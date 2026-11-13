# SPDX-License-Identifier: Apache-2.0
"""VAR-Q KV cache with symmetric-quantized storage for completed chunks.

Storage model (fitted to SGLang's ``update_and_get_attention_kv`` contract, and
deliberately the same shape as the sibling ``qvg_packed_cache``):

  * The retained window is split into chunk-aligned *segments* in global token
    order. Each segment is either bf16 (``k``/``v`` tensors resident) or
    VAR-Q-quantized (``packed_k``/``packed_v`` metadata resident, bf16 freed).
  * The current (still-denoising) chunk and the newest ``keep_recent_chunks``
    completed chunks stay bf16 (rewritten each denoise step / attended cleanly).
  * Older completed segments are quantized once and their bf16 freed, so the
    resident footprint drops to ~(sink + recent) bf16 + quantized tail.
  * On read the visible window is reconstructed densely on the fly
    (dequantize quantized segments + copy bf16 ones) and returned to attention;
    that transient dense tensor is freed after the layer's attention, so only
    ONE layer is dense at a time instead of all ``num_layers`` of them.

Difference from QVG: VAR-Q is tuning-free. There is no calibration pass and no
k-means clustering; quantization is symmetric with no zero point, and each
segment carries one scale per ``(head, channel)`` — i.e. the per-(chunk, head,
channel) granularity the method is built on. Segments are stored in SGLang's
native ``BLHc`` layout, so pack and dequant are permute-free.

Scope: causal chunked video attention (sliding window + sink, chunk-aligned
writes, optional head-slice, ``recent_window_tokens`` None or a non-negative
int). Unsupported base-class features raise NotImplementedError rather than
silently diverge.
"""

from __future__ import annotations

from functools import cache
from typing import Any, cast

import torch

from sglang.multimodal_gen.configs.quantization.varq_kv import VARQKVQuantArgs
from sglang.multimodal_gen.runtime.layers.kvcache.causal_attention_cache import (
    CausalAttentionKVView,
)
from sglang.multimodal_gen.runtime.layers.kvcache.qvg_packed_cache import (
    QVGPackedCausalKVCache,
    _Segment,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# SGLang stores causal KV as [B, S, H, D]; VAR-Q calls that layout "BLHc".
_QKV_FORMAT = "BLHc"


@cache
def _varq_functions():
    try:
        from VAR_Q import dequantize_tensor, quantize_tensor
    except ImportError as e:
        raise ImportError(
            "VAR-Q KV-cache quantization requires the optional VAR-Q runtime "
            "package exposing the `VAR_Q` module. Install that package separately "
            "and make it available on PYTHONPATH before enabling "
            "--kv-cache-varq."
        ) from e
    return quantize_tensor, dequantize_tensor


def _packed_nbytes(q: dict | None) -> int:
    if not q:
        return 0
    total = 0
    for key in ("packed", "scale"):
        t = q.get(key)
        if isinstance(t, torch.Tensor):
            total += t.numel() * t.element_size()
    return total


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sort and coalesce half-open intervals so no token is visited twice."""
    if not intervals:
        return []
    ordered = sorted((lo, hi) for lo, hi in intervals if hi > lo)
    merged: list[tuple[int, int]] = []
    for lo, hi in ordered:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


class VARQPackedCausalKVCache(QVGPackedCausalKVCache):
    """Share QVG's segment lifecycle, but keep VAR-Q storage and view semantics.

    The inherited write/eviction paths use only segment boundaries, dense K/V,
    and the shared recency/sink policy. Quantization, byte accounting and dense
    reconstruction stay here because the packed dictionaries and layouts differ.
    """

    def __init__(
        self,
        *,
        batch_size: int,
        cache_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        global_end_index: torch.Tensor,
        local_end_index: torch.Tensor,
        use_int_indices: bool = False,
        sink_tokens: int = 0,
        attention_window_size: int | None = None,
        quant_args: VARQKVQuantArgs,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            cache_size=cache_size,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
            global_end_index=global_end_index,
            local_end_index=local_end_index,
            use_int_indices=use_int_indices,
            sink_tokens=sink_tokens,
            attention_window_size=attention_window_size,
            # The lifecycle reads only the shared recency/sink policy fields.
            quant_args=cast(Any, quant_args),
        )
        self.reset_pinned_sink()

    # ------------------------------------------------------------------ api
    def reset_indices(self) -> None:
        super().reset_indices()
        self.reset_pinned_sink()

    def reset_pinned_sink(self) -> None:
        self._pinned_start = -1
        self._pinned_len = 0

    def pin_current_chunk(self, current_num_tokens: int) -> None:
        """Pin the leading ``sink_tokens`` tokens of the current chunk.

        Unlike the QVG sibling this is implemented, not refused: the pinned range
        is served by the same interval merge as the sliding window, so it stays
        readable after the chunk is quantized.

        Note it only adds tokens when the attention window is smaller than the
        current chunk; when the window already covers the whole current chunk
        this is a no-op, exactly as in the dense cache.
        """
        if self.sink_tokens <= 0 or current_num_tokens <= 0:
            self.reset_pinned_sink()
            return
        self._pinned_start = self._global_end - current_num_tokens
        self._pinned_len = min(self.sink_tokens, current_num_tokens)

    def resident_nbytes(self) -> int:
        # _Segment.nbytes() interprets packed dictionaries as QVG metadata.
        total = 0
        for seg in self._all_segments():
            if seg.packed:
                total += _packed_nbytes(seg.packed_k) + _packed_nbytes(seg.packed_v)
            else:
                total += seg.nbytes()
        return total

    # -------------------------------------------------------------- helpers
    def _quant_kwargs(self) -> dict:
        # The base constructor stores the VAR-Q args passed by this adapter.
        q = cast(VARQKVQuantArgs, self.q)
        bits = q.bits
        if bits is None:
            raise RuntimeError("VAR-Q packed cache constructed with quantization off")
        return {
            "quant_bits": int(bits),
            "qkv_format": _QKV_FORMAT,
            "quant_method": q.quant_method,
            "pack_to_int32": q.pack_to_int32,
            "dequant_dtype": q.dequant_dtype,
            "compression_ratio": q.compression_ratio,
            "max_scale_seq_len": q.max_scale_seq_len,
        }

    def _pack(self, seg: _Segment) -> None:
        if seg.packed or seg.k is None:
            return
        quantize_tensor, _ = _varq_functions()
        kwargs = self._quant_kwargs()
        seg.packed_k = quantize_tensor(seg.k, kv_role="k", **kwargs)
        seg.packed_v = quantize_tensor(seg.v, kv_role="v", **kwargs)
        seg.k = None
        seg.v = None
        logger.info_once(f"Using VAR-Q packed KV cache: {self.q.describe()}")

    def _dequant(self, packed: dict) -> torch.Tensor:
        _, dequantize_tensor = _varq_functions()
        return dequantize_tensor(
            packed["packed"],
            packed["scale"],
            packed.get("pack_meta"),
            self.dtype,
            quant_meta=packed.get("quant_meta"),
        )

    # -------------------------------------------------------------- contract
    def update_and_get_attention_kv(
        self,
        *,
        key: torch.Tensor,
        value: torch.Tensor,
        current_chunk_start: int,
        cache_head_start: int | None = None,
        recent_window_tokens: int | None = None,
        debug_name: str = "VAR-Q packed KV cache",
    ) -> CausalAttentionKVView:
        view = super().update_and_get_attention_kv(
            key=key,
            value=value,
            current_chunk_start=current_chunk_start,
            cache_head_start=cache_head_start,
            recent_window_tokens=recent_window_tokens,
            debug_name=debug_name,
        )
        # VAR-Q indexes the current chunk at the tail of the reconstructed view,
        # not at the beginning as QVG's returned metadata does.
        view.local_start_index = max(0, view.k.shape[1] - key.shape[1])
        view.local_end_index = view.k.shape[1]
        return view

    def _visible_intervals(
        self,
        current_chunk_start: int,
        recent_window_tokens: int | None,
    ) -> list[tuple[int, int]]:
        """Global token intervals the current chunk may attend to, ascending."""
        sink_end = self._sink_end()
        if recent_window_tokens is None:
            tail_start = self._tail_start()
        else:
            if recent_window_tokens < 0:
                raise ValueError("recent_window_tokens must be >= 0 or None")
            tail_start = max(sink_end, current_chunk_start - recent_window_tokens)

        intervals: list[tuple[int, int]] = []
        if sink_end > 0:
            intervals.append((0, sink_end))
        if self._pinned_len > 0 and self._pinned_start >= 0:
            intervals.append(
                (self._pinned_start, self._pinned_start + self._pinned_len)
            )
        intervals.append((tail_start, self._global_end))
        return _merge_intervals(intervals)

    def _reconstruct(
        self,
        current_chunk_start: int,
        recent_window_tokens: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dense visible window in ascending global order.

        The last interval always ends at ``_global_end``, i.e. at the end of the
        current chunk, so the current chunk occupies the tail of the returned
        tensor and causal attention lines up bottom-right.
        """
        intervals = self._visible_intervals(current_chunk_start, recent_window_tokens)

        visible_segments: list[tuple[_Segment, int, int]] = []
        visible_tokens = 0
        for g_lo, g_hi in intervals:
            for seg in self._all_segments():
                a = max(g_lo, seg.g0)
                b = min(g_hi, seg.g1)
                if b <= a:
                    continue
                visible_segments.append((seg, a - seg.g0, b - seg.g0))
                visible_tokens += b - a

        if not visible_segments:
            empty = torch.empty(
                self.batch_size,
                0,
                self.num_heads,
                self.head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            return empty, empty.clone()

        if len(visible_segments) == 1 and not visible_segments[0][0].packed:
            seg, i0, i1 = visible_segments[0]
            return seg.k[:, i0:i1], seg.v[:, i0:i1]

        output_shape = (
            self.batch_size,
            visible_tokens,
            self.num_heads,
            self.head_dim,
        )
        vk = torch.empty(output_shape, dtype=self.dtype, device=self.device)
        vv = torch.empty_like(vk)
        output_start = 0

        # Dequantize one tensor at a time so reconstruction needs only the final
        # dense view plus one segment-sized temporary.
        for seg, i0, i1 in visible_segments:
            output_end = output_start + i1 - i0
            if seg.packed:
                assert seg.packed_k is not None and seg.packed_v is not None
                dequantized = self._dequant(seg.packed_k)
                vk[:, output_start:output_end].copy_(dequantized[:, i0:i1])
                del dequantized

                dequantized = self._dequant(seg.packed_v)
                vv[:, output_start:output_end].copy_(dequantized[:, i0:i1])
                del dequantized
            else:
                vk[:, output_start:output_end].copy_(seg.k[:, i0:i1])
                vv[:, output_start:output_end].copy_(seg.v[:, i0:i1])
            output_start = output_end

        return vk, vv
