# SPDX-License-Identifier: Apache-2.0
"""VAR-Q KV cache with symmetric-quantized storage for completed chunks.

Storage model (fitted to SGLang's ``update_and_get_attention_kv`` contract, and
deliberately the same shape as the sibling ``qvg_packed_cache``):

  * The retained window is split into chunk-aligned *segments* in global token
    order. Each segment is either bf16 (``k``/``v`` tensors resident) or
    VAR-Q-quantized (``qk``/``qv`` metadata resident, bf16 freed).
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

from dataclasses import dataclass
from functools import cache

import torch

from sglang.multimodal_gen.configs.quantization.varq_kv import VARQKVQuantArgs
from sglang.multimodal_gen.runtime.layers.kvcache.causal_attention_cache import (
    CausalAttentionKVView,
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
            "VAR-Q KV-cache quantization requires its optional runtime "
            "dependency. Install it with: "
            "pip install 'sglang[diffusion-varq]' && "
            "pip install VAR-Q."
        ) from e
    return quantize_tensor, dequantize_tensor


@dataclass
class _Segment:
    g0: int  # global start token (inclusive)
    g1: int  # global end token (exclusive)
    is_sink: bool  # sink segments are never evicted
    k: torch.Tensor | None = None
    v: torch.Tensor | None = None
    qk: dict | None = None
    qv: dict | None = None

    @property
    def packed(self) -> bool:
        return self.qk is not None

    def nbytes(self) -> int:
        if self.packed:
            return _packed_nbytes(self.qk) + _packed_nbytes(self.qv)
        return (
            self.k.numel() * self.k.element_size()
            + self.v.numel() * self.v.element_size()
        )


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


class VARQPackedCausalKVCache:
    """Chunk-segmented causal KV cache with VAR-Q-quantized cold segments."""

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
        self.batch_size = batch_size
        self.cache_size = cache_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.sink_tokens = sink_tokens
        self.global_sink_tokens = 0
        self.attention_window_size = attention_window_size or cache_size
        self.q = quant_args
        # kept for API compatibility with the dense cache (indices are not read
        # by consumers, but reset/patterns touch them)
        self.global_end_index = global_end_index
        self.local_end_index = local_end_index
        self.global_end_index_int = 0 if use_int_indices else None
        self.local_end_index_int = 0 if use_int_indices else None

        self._segments: list[_Segment] = []  # completed, global-ordered
        self._cur: _Segment | None = None  # current (mutable) chunk
        self._global_end = 0
        self._chunk_tokens = 0  # inferred from first advance
        self._pinned_start = -1  # global token index of the pinned sink
        self._pinned_len = 0

    # ------------------------------------------------------------------ api
    def reset_indices(self) -> None:
        self._segments = []
        self._cur = None
        self._global_end = 0
        self.reset_pinned_sink()
        if self.global_end_index_int is not None:
            self.global_end_index_int = 0
            self.local_end_index_int = 0
        self.global_end_index.zero_()
        self.local_end_index.zero_()

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

    def can_direct_current_attention(self, num_new_tokens: int) -> bool:
        return (
            self.sink_tokens == 0
            and self.cache_size == num_new_tokens
            and self.attention_window_size == num_new_tokens
        )

    @property
    def num_cache_heads(self) -> int:
        return self.num_heads

    def resident_nbytes(self) -> int:
        total = sum(s.nbytes() for s in self._segments)
        if self._cur is not None:
            total += self._cur.nbytes()
        return total

    # -------------------------------------------------------------- helpers
    def _quant_kwargs(self) -> dict:
        bits = self.q.bits
        if bits is None:
            raise RuntimeError("VAR-Q packed cache constructed with quantization off")
        return {
            "quant_bits": int(bits),
            "qkv_format": _QKV_FORMAT,
            "quant_method": self.q.quant_method,
            "pack_to_int32": self.q.pack_to_int32,
            "dequant_dtype": self.q.dequant_dtype,
            "compression_ratio": self.q.compression_ratio,
            "max_scale_seq_len": self.q.max_scale_seq_len,
        }

    def _new_bf16_segment(self, g0: int, g1: int, is_sink: bool) -> _Segment:
        n = g1 - g0
        k = torch.zeros(
            self.batch_size,
            n,
            self.num_heads,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        v = torch.zeros_like(k)
        return _Segment(g0=g0, g1=g1, is_sink=is_sink, k=k, v=v)

    def _write(self, seg: _Segment, key, value, head_slice) -> None:
        if head_slice is None:
            seg.k.copy_(key)
            seg.v.copy_(value)
        else:
            seg.k[:, :, head_slice, :] = key
            seg.v[:, :, head_slice, :] = value

    def _pack(self, seg: _Segment) -> None:
        if seg.packed or seg.k is None:
            return
        quantize_tensor, _ = _varq_functions()
        kwargs = self._quant_kwargs()
        seg.qk = quantize_tensor(seg.k, kv_role="k", **kwargs)
        seg.qv = quantize_tensor(seg.v, kv_role="v", **kwargs)
        seg.k = None
        seg.v = None
        logger.info_once(f"Using VAR-Q packed KV cache: {self.q.describe()}")

    def _dequant(self, q: dict) -> torch.Tensor:
        _, dequantize_tensor = _varq_functions()
        return dequantize_tensor(
            q["packed"],
            q["scale"],
            q.get("pack_meta"),
            self.dtype,
            quant_meta=q.get("quant_meta"),
        )

    def _all_segments(self) -> list[_Segment]:
        segs = list(self._segments)
        if self._cur is not None:
            segs.append(self._cur)
        return segs

    def _sink_end(self) -> int:
        return min(self.sink_tokens, self._global_end)

    def _tail_start(self) -> int:
        """Global start of the rolling recent tail (sink occupies its own budget
        at the window front, matching the dense cache's roll)."""
        sink_end = self._sink_end()
        recent_budget = max(0, self.attention_window_size - sink_end)
        return max(sink_end, self._global_end - recent_budget)

    def _pack_and_evict(self) -> None:
        """Quantize completed segments older than the recency guard; drop
        segments that have slid entirely out of the window (sink never evicted)."""
        tail_start = self._tail_start()

        kept = []
        for s in self._segments:
            if not s.is_sink and s.g1 <= tail_start:
                continue
            kept.append(s)
        self._segments = kept

        if not self.q.enabled:
            return
        recent = self.q.keep_recent_chunks
        nonsink = [s for s in self._segments if not s.is_sink]
        cutoff_idx = len(nonsink) - recent
        for i, s in enumerate(nonsink):
            if i < cutoff_idx:
                self._pack(s)
        if self.q.sink:
            sink_keep_tokens = self.q.sink_keep_chunks * max(1, self._chunk_tokens)
            for s in self._segments:
                if s.is_sink and s.g0 >= sink_keep_tokens:
                    self._pack(s)

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
        num_new = key.shape[1]
        num_input_heads = key.shape[2]
        head_slice = None
        if num_input_heads != self.num_heads:
            if cache_head_start is None:
                raise ValueError(
                    f"{debug_name}: cache_head_start required for head slice"
                )
            head_slice = slice(cache_head_start, cache_head_start + num_input_heads)
        cend = current_chunk_start + num_new

        if self._cur is not None and current_chunk_start == self._cur.g0:
            # rewrite current chunk in place (denoise step)
            if cend != self._cur.g1:
                raise NotImplementedError(
                    f"{debug_name}: current-chunk rewrite size changed"
                )
            self._write(self._cur, key, value, head_slice)
        elif current_chunk_start == self._global_end:
            # advance: finalize current chunk, start a new one
            if self._cur is not None:
                self._segments.append(self._cur)
            is_sink = current_chunk_start < self.sink_tokens
            if self._chunk_tokens == 0:
                self._chunk_tokens = num_new
            self._cur = self._new_bf16_segment(current_chunk_start, cend, is_sink)
            self._write(self._cur, key, value, head_slice)
            self._global_end = cend
            self._pack_and_evict()
        else:
            raise NotImplementedError(
                f"{debug_name}: non-sequential write current_start="
                f"{current_chunk_start} global_end={self._global_end} "
                f"cur={None if self._cur is None else self._cur.g0}"
            )

        local_end = min(self._global_end, self.cache_size)
        if self.global_end_index_int is not None:
            self.global_end_index_int = self._global_end
            self.local_end_index_int = local_end
        else:
            self.global_end_index.fill_(self._global_end)
            self.local_end_index.fill_(local_end)

        vk, vv = self._reconstruct(current_chunk_start, recent_window_tokens)
        visible_tokens = vk.shape[1]
        return CausalAttentionKVView(
            k=vk,
            v=vv,
            local_start_index=max(0, visible_tokens - num_new),
            local_end_index=visible_tokens,
            visible_local_end=local_end,
            visible_global_end=self._global_end,
        )

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
            intervals.append((self._pinned_start, self._pinned_start + self._pinned_len))
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
                dequantized = self._dequant(seg.qk)
                vk[:, output_start:output_end].copy_(dequantized[:, i0:i1])
                del dequantized

                dequantized = self._dequant(seg.qv)
                vv[:, output_start:output_end].copy_(dequantized[:, i0:i1])
                del dequantized
            else:
                vk[:, output_start:output_end].copy_(seg.k[:, i0:i1])
                vv[:, output_start:output_end].copy_(seg.v[:, i0:i1])
            output_start = output_end

        return vk, vv
