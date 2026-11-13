# SPDX-License-Identifier: Apache-2.0
"""VAR-Q adapter for the shared segmented causal KV-cache lifecycle."""

from __future__ import annotations

from functools import cache
from typing import cast

import torch

from sglang.multimodal_gen.configs.quantization.varq_kv import VARQKVQuantArgs
from sglang.multimodal_gen.runtime.layers.kvcache.qvg_packed_cache import (
    QVGPackedCausalKVCache,
    _merge_intervals,
)


@cache
def _varq_functions():
    try:
        from VAR_Q import dequantize_tensor, quantize_tensor
    except ImportError as e:
        raise ImportError(
            "VAR-Q KV-cache quantization requires the optional VAR_Q package; "
            "install it and add it to PYTHONPATH before enabling --kv-cache-varq."
        ) from e
    return quantize_tensor, dequantize_tensor


class VARQPackedCausalKVCache(QVGPackedCausalKVCache):
    """Shared lifecycle with VAR-Q's BLHc codec and pinned-sink view."""

    _cache_label = "VAR-Q"

    def _reset_extra_state(self) -> None:
        self._pinned_start, self._pinned_len = -1, 0

    def reset_pinned_sink(self) -> None:
        self._reset_extra_state()

    def pin_current_chunk(self, current_num_tokens: int) -> None:
        if self.sink_tokens <= 0 or current_num_tokens <= 0:
            self._reset_extra_state()
        else:
            self._pinned_start = self._global_end - current_num_tokens
            self._pinned_len = min(self.sink_tokens, current_num_tokens)

    def _packed_nbytes(self, packed: dict | None) -> int:
        if not packed:
            return 0
        return sum(
            value.numel() * value.element_size()
            for value in (packed.get("packed"), packed.get("scale"))
            if isinstance(value, torch.Tensor)
        )

    def _pack_tensor(self, tensor: torch.Tensor, role: str) -> dict:
        quantize_tensor, _ = _varq_functions()
        q = cast(VARQKVQuantArgs, self.q)
        if q.bits is None:
            raise RuntimeError("VAR-Q packed cache constructed with quantization off")
        return quantize_tensor(
            tensor,
            kv_role=role,
            quant_bits=int(q.bits),
            qkv_format="BLHc",
            quant_method=q.quant_method,
            pack_to_int32=q.pack_to_int32,
            dequant_dtype=q.dequant_dtype,
            compression_ratio=q.compression_ratio,
            max_scale_seq_len=q.max_scale_seq_len,
        )

    def _dequant(self, packed: dict) -> torch.Tensor:
        _, dequantize_tensor = _varq_functions()
        return dequantize_tensor(**packed, dequant_dtype=self.dtype)

    def _decoded_slice(self, decoded: torch.Tensor, i0: int, i1: int) -> torch.Tensor:
        return decoded[:, i0:i1]

    def _visible_ranges(
        self, current_chunk_start: int, recent_window_tokens: int | None
    ) -> list[tuple[int, int]]:
        ranges = super()._visible_ranges(current_chunk_start, recent_window_tokens)
        if self._pinned_len > 0 and self._pinned_start >= 0:
            ranges.append((self._pinned_start, self._pinned_start + self._pinned_len))
        return _merge_intervals(ranges)

    def _view_indices(
        self, visible_tokens: int, num_new_tokens: int
    ) -> tuple[int, int]:
        return max(0, visible_tokens - num_new_tokens), visible_tokens
