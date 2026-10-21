"""Three-arm KV-cache comparison on real sglang caches: BF16 / QVG(PRQ) / VAR-Q.

Cache-level, same shape, same data, same window. Reports resident bytes,
reconstruction error against the dense window, and the pack / reconstruct cost
that the packed caches add per layer per chunk.
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
CHUNKS = 12
SINK = C  # one frame of sink
WINDOW = C * 6
DTYPE = torch.bfloat16
DEV = "cuda"


def dense_cache():
    return CausalSelfAttentionKVCache(
        k=torch.zeros(B, WINDOW, H, D, dtype=DTYPE, device=DEV),
        v=torch.zeros(B, WINDOW, H, D, dtype=DTYPE, device=DEV),
        global_end_index=torch.zeros(1, dtype=torch.long, device=DEV),
        local_end_index=torch.zeros(1, dtype=torch.long, device=DEV),
        cache_size=WINDOW,
        sink_tokens=SINK,
        attention_window_size=WINDOW,
    )


def packed_cache(kind, quant_args):
    cls = QVGPackedCausalKVCache if kind == "qvg" else VARQPackedCausalKVCache
    return cls(
        batch_size=B,
        cache_size=WINDOW,
        num_heads=H,
        head_dim=D,
        dtype=DTYPE,
        device=torch.device(DEV),
        global_end_index=torch.zeros(1, dtype=torch.long, device=DEV),
        local_end_index=torch.zeros(1, dtype=torch.long, device=DEV),
        sink_tokens=SINK,
        attention_window_size=WINDOW,
        quant_args=quant_args,
    )


def steady(times):
    """median of the timed samples, ms"""
    return statistics.median(times) * 1e3


torch.manual_seed(0)

# A100 (SM80) has no E4M3 fp8 in Triton, so QVG's default fp8 scale cannot run
# here. Force bf16 scales for QVG and disclose it in the table footer.
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

# ---- dense reference -------------------------------------------------------
dense = dense_cache()
t0 = time.perf_counter()
for ci, (k, v) in enumerate(data):
    view = dense.update_and_get_attention_kv(
        key=k, value=v, current_chunk_start=ci * C, recent_window_tokens=WINDOW
    )
torch.cuda.synchronize()
dense_ms = (time.perf_counter() - t0) * 1e3
ref_k = view.k.clone()
ref_v = view.v.clone()
dense_bytes = dense.k.numel() * dense.k.element_size() * 2
print(f"dense: window={ref_k.shape[1]} tok  resident={dense_bytes/2**20:.1f} MB  "
      f"replay={dense_ms:.1f} ms")

# ---- packed arms -----------------------------------------------------------
results = []
for kind, args_factory, label in [
    ("qvg", lambda b: QVGKVQuantArgs.from_dict({"kv_cache_quant": f"int{b}"}), "QVG/PRQ"),
    ("varq", lambda b: VARQKVQuantArgs.from_dict({"kv_cache_varq": f"int{b}"}), "VAR-Q"),
]:
    for bits in (8, 6, 4, 3, 2):
        try:
            qa = args_factory(bits)
        except Exception as e:
            print(f"{label} int{bits}: config rejected ({e})")
            continue
        try:
            cache = packed_cache(kind, qa)
            # warm up one full pass, then time a steady-state pass
            for ci, (k, v) in enumerate(data):
                cache.update_and_get_attention_kv(
                    key=k, value=v, current_chunk_start=ci * C,
                    recent_window_tokens=WINDOW,
                )
            torch.cuda.synchronize()
            samples = []
            for _ in range(3):
                cache.reset_indices()
                t0 = time.perf_counter()
                for ci, (k, v) in enumerate(data):
                    view = cache.update_and_get_attention_kv(
                        key=k, value=v, current_chunk_start=ci * C,
                        recent_window_tokens=WINDOW,
                    )
                torch.cuda.synchronize()
                samples.append(time.perf_counter() - t0)
            pk, pv = view.k, view.v
            assert pk.shape == ref_k.shape, f"{label} int{bits} window {pk.shape} vs {ref_k.shape}"
            err_k = (pk.float() - ref_k.float()).pow(2).mean().sqrt().item()
            err_v = (pv.float() - ref_v.float()).pow(2).mean().sqrt().item()
            ref_rms = ref_k.float().pow(2).mean().sqrt().item()
            resident = cache.resident_nbytes()
            results.append(dict(
                label=label, bits=bits,
                resident_mb=resident / 2**20,
                ratio=dense_bytes / resident,
                err_rel=100.0 * ((err_k + err_v) / 2) / max(ref_rms, 1e-9),
                replay_ms=steady(samples),
            ))
        except Exception as e:
            torch.cuda.synchronize()
            print(f"{label} int{bits}: FAILED {type(e).__name__}: {str(e)[:160]}")

print()
hdr = f"| {'arm':12s} | {'KV precision':12s} | {'resident':>11s} | {'vs bf16':>8s} | {'recon err':>9s} | {'replay ms':>9s} |"
print(hdr)
print("|" + "-" * 14 + "|" + "-" * 14 + "|" + "-" * 13 + "|" + "-" * 10 + "|" + "-" * 11 + "|" + "-" * 11 + "|")
print(f"| {'BF16 dense':12s} | {'bf16':12s} | {dense_bytes/2**20:9.1f} MB | {'1.00x':>8s} | {'0.00%':>9s} | {dense_ms:9.1f} |")
for r in results:
    print(f"| {r['label']:12s} | {'int'+str(r['bits']):12s} | "
          f"{r['resident_mb']:7.1f} MB | {r['ratio']:7.2f}x | "
          f"{r['err_rel']:8.3f}% | {r['replay_ms']:9.1f} |")

print()
print(f"shape: B={B} chunk={C} tokens  heads={H}  head_dim={D}  chunks={CHUNKS}  "
      f"window={WINDOW}  dtype={DTYPE}")
print(f"peak alloc: {torch.cuda.max_memory_allocated()/2**20:.0f} MB")
print(f"QVG scale precision: {QVG_SCALE}")
if any(r["label"].startswith("QVG") for r in results):
    print("note: QVG/PRQ offers int4 and int2 only in this sglang integration; "
          "VAR-Q rows at int8/int6/int3 have no PRQ counterpart.")
