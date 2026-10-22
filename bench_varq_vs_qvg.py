"""Three-arm KV-cache comparison on real sglang caches: BF16 / QVG(PRQ) / VAR-Q.

Cache-level, same shape, same data. Sweeps the window length, because the
compression ratio depends on how much of the window stays bf16 (the current chunk
plus keep_recent_chunks): a short window flatters bf16, a long one approaches the
packed format's asymptote.

Reports resident bytes, reconstruction error against the dense window, and the
packed-path replay cost.
"""

import statistics
import sys
import time

import torch

from sglang.multimodal_gen.configs.quantization.qvg_kv import QVGKVQuantArgs
from sglang.multimodal_gen.configs.quantization.varq_kv import VARQKVQuantArgs
from sglang.multimodal_gen.runtime.layers.kvcache.causal_attention_cache import (
    CausalSelfAttentionKVCache,
)
from sglang.multimodal_gen.runtime.layers.kvcache.qvg_packed_cache import (
    QVGPackedCausalKVCache,
)
from sglang.multimodal_gen.runtime.layers.kvcache.varq_packed_cache import (
    VARQPackedCausalKVCache,
)

B, C, H, D = 1, 1560, 12, 128
CHUNKS = 21  # Self-Forcing 5s = 21 latent frames
SINK = C  # one frame of sink
WINDOWS = [6, 12, 21]  # KV window, in chunks
DTYPE = torch.bfloat16
DEV = "cuda"


def dense_cache(window):
    return CausalSelfAttentionKVCache(
        k=torch.zeros(B, window, H, D, dtype=DTYPE, device=DEV),
        v=torch.zeros(B, window, H, D, dtype=DTYPE, device=DEV),
        global_end_index=torch.zeros(1, dtype=torch.long, device=DEV),
        local_end_index=torch.zeros(1, dtype=torch.long, device=DEV),
        cache_size=window,
        sink_tokens=SINK,
        attention_window_size=window,
    )


def packed_cache(kind, window, quant_args):
    cls = QVGPackedCausalKVCache if kind == "qvg" else VARQPackedCausalKVCache
    return cls(
        batch_size=B,
        cache_size=window,
        num_heads=H,
        head_dim=D,
        dtype=DTYPE,
        device=torch.device(DEV),
        global_end_index=torch.zeros(1, dtype=torch.long, device=DEV),
        local_end_index=torch.zeros(1, dtype=torch.long, device=DEV),
        sink_tokens=SINK,
        attention_window_size=window,
        quant_args=quant_args,
    )


torch.manual_seed(0)

# A100 (SM80) has no E4M3 fp8 in Triton, so QVG's default fp8 scale cannot run
# here. Force bf16 scales for QVG and disclose it in the footer.
try:
    import quant_videogen.functions as _qf

    _orig_prq_q = _qf.triton_prq_quantize_tensor

    def _prq_q(*a, **kw):
        kw["scale_precision"] = torch.bfloat16
        return _orig_prq_q(*a, **kw)

    _qf.triton_prq_quantize_tensor = _prq_q
    QVG_SCALE = "bf16 (A100 has no fp8e4nv)"
except Exception as e:  # pragma: no cover
    QVG_SCALE = f"unpatched ({e})"

data = [
    (
        torch.randn(B, C, H, D, dtype=DTYPE, device=DEV),
        torch.randn(B, C, H, D, dtype=DTYPE, device=DEV),
    )
    for _ in range(CHUNKS)
]

ARMS = [
    ("qvg", "QVG/PRQ", lambda b: QVGKVQuantArgs.from_dict({"kv_cache_quant": f"int{b}"})),
    ("varq", "VAR-Q", lambda b: VARQKVQuantArgs.from_dict({"kv_cache_varq": f"int{b}"})),
]
BITS = (8, 6, 4, 3, 2)

notes = []
for win_chunks in WINDOWS:
    window = C * win_chunks

    dense = dense_cache(window)
    t0 = time.perf_counter()
    for ci, (k, v) in enumerate(data):
        dview = dense.update_and_get_attention_kv(
            key=k, value=v, current_chunk_start=ci * C, recent_window_tokens=window
        )
    torch.cuda.synchronize()
    dense_ms = (time.perf_counter() - t0) * 1e3
    ref_k = dview.k.clone()
    ref_v = dview.v.clone()
    ref_rms = ref_k.float().pow(2).mean().sqrt().item()
    dense_bytes = dense.k.numel() * dense.k.element_size() * 2

    print(
        f"\n### window = {win_chunks} chunks ({window} tokens) | "
        f"dense bf16 {dense_bytes / 2**20:.1f} MB | replay {dense_ms:.1f} ms\n"
    )
    print(
        f"| {'arm':10s} | {'prec':6s} | {'resident':>10s} | {'vs bf16':>7s} | "
        f"{'recon err':>9s} | {'replay ms':>9s} |"
    )
    print(
        "|" + "-" * 12 + "|" + "-" * 8 + "|" + "-" * 12 + "|" + "-" * 9 + "|"
        + "-" * 11 + "|" + "-" * 11 + "|"
    )
    print(
        f"| {'BF16 dense':10s} | {'bf16':6s} | {dense_bytes / 2**20:8.1f} MB | "
        f"{'1.00x':>7s} | {'0.00%':>9s} | {dense_ms:9.1f} |"
    )

    for kind, label, factory in ARMS:
        for bits in BITS:
            try:
                qa = factory(bits)
            except Exception:
                continue  # precision not offered by this quantizer
            tag = f"{label} int{bits} w{win_chunks}"
            try:
                cache = packed_cache(kind, window, qa)
                for ci, (k, v) in enumerate(data):  # warm up
                    cache.update_and_get_attention_kv(
                        key=k, value=v, current_chunk_start=ci * C,
                        recent_window_tokens=window,
                    )
                torch.cuda.synchronize()
                samples = []
                for _ in range(3):
                    cache.reset_indices()
                    t0 = time.perf_counter()
                    for ci, (k, v) in enumerate(data):
                        pview = cache.update_and_get_attention_kv(
                            key=k, value=v, current_chunk_start=ci * C,
                            recent_window_tokens=window,
                        )
                    torch.cuda.synchronize()
                    samples.append(time.perf_counter() - t0)
                if pview.k.shape != ref_k.shape:
                    notes.append(
                        f"{tag}: window {tuple(pview.k.shape)} vs "
                        f"{tuple(ref_k.shape)}"
                    )
                    continue
                err_k = (pview.k.float() - ref_k.float()).pow(2).mean().sqrt().item()
                err_v = (pview.v.float() - ref_v.float()).pow(2).mean().sqrt().item()
                err = 100.0 * ((err_k + err_v) / 2) / max(ref_rms, 1e-9)
                resident = cache.resident_nbytes()
                print(
                    f"| {label:10s} | {'int' + str(bits):6s} | "
                    f"{resident / 2**20:8.1f} MB | "
                    f"{dense_bytes / resident:6.2f}x | {err:8.3f}% | "
                    f"{statistics.median(samples) * 1e3:9.1f} |"
                )
            except Exception as e:
                notes.append(f"{tag}: {type(e).__name__}: {str(e)[:120]}")

print(
    f"\nshape: B={B} chunk={C} tokens  heads={H}  head_dim={D}  "
    f"chunks={CHUNKS}  dtype={DTYPE}"
)
print(f"peak alloc: {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")
print(f"QVG scale precision: {QVG_SCALE}")
print(
    "note: QVG/PRQ offers int4 and int2 only in this sglang integration; VAR-Q "
    "rows at int8/int6/int3 have no PRQ counterpart."
)
print(
    "note: reconstruction error here is measured on random Gaussian data, which "
    "has no per-channel outliers or cross-chunk structure; it is not a quality "
    "result for either method."
)
if notes:
    print(f"\n{len(notes)} arm(s) did not produce a comparable window:")
    for n in notes:
        print("  -", n)
