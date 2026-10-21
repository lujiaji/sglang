# SPDX-License-Identifier: Apache-2.0
"""CLI-facing configuration for VAR-Q KV-cache quantization.

Mirrors the sibling ``qvg_kv`` config: the on/off switch and tuning knobs live on
a typed config object carried by ``ServerArgs`` (``kv_cache_varq_config``) rather
than a pile of raw environment variables.

VAR-Q is tuning-free: it needs no calibration pass and no clustering. Quantization
is symmetric, has no zero point, and picks one scale per ``(generation chunk,
head, channel)``. In this integration the chunk boundary is the causal KV-cache
segment boundary, so one segment carries exactly one scale per channel.
"""

from __future__ import annotations

from dataclasses import dataclass

_BITS = {
    "off": None,
    "none": None,
    "bf16": None,
    "int8": 8,
    "int6": 6,
    "int4": 4,
    "int3": 3,
    "int2": 2,
}


def _parse_bits(val: str | int | None) -> int | None:
    if val is None:
        return None
    if isinstance(val, int):
        return val
    key = str(val).strip().lower()
    if key not in _BITS:
        raise ValueError(
            f"kv-cache-varq must be one of {list(_BITS)}, got {val!r}"
        )
    return _BITS[key]


@dataclass
class VARQKVQuantArgs:
    """VAR-Q symmetric per-(chunk, head, channel) KV-cache quantization settings.

    ``bits is None`` means quantization is OFF (plain bf16 cache).
    """

    bits: int | None = None  # None => off; 8/6/4/3/2
    quant_method: str = "VARQ"  # VAR-Q grouping; see VAR_Q.quant
    # One scale per (chunk, head, channel). 1.0 means "the whole segment is one
    # scale group", which is what the causal segment boundary already encodes.
    compression_ratio: float = 1.0
    max_scale_seq_len: int | None = None
    pack_to_int32: bool = True
    dequant_dtype: str = "native"
    keep_recent_chunks: int = 1  # completed chunks kept bf16 (recency guard)
    sink: bool = True  # quantize the attention sink too
    sink_keep_chunks: int = 0  # leading sink chunks kept bf16 forever

    @property
    def enabled(self) -> bool:
        return self.bits is not None

    def validate(self) -> VARQKVQuantArgs:
        if self.bits not in (None, 2, 3, 4, 6, 8):
            raise ValueError(
                f"kv-cache-varq bits must be one of 2/3/4/6/8, got {self.bits}"
            )
        if self.bits is not None and not self.pack_to_int32 and self.bits in (2, 3, 6):
            raise ValueError(
                f"kv-cache-varq bits={self.bits} requires pack_to_int32"
            )
        if not (0.0 < float(self.compression_ratio)):
            raise ValueError("kv-cache-varq compression_ratio must be positive")
        if self.max_scale_seq_len is not None and self.max_scale_seq_len <= 0:
            raise ValueError("kv-cache-varq max_scale_seq_len must be positive")
        if self.keep_recent_chunks < 0 or self.sink_keep_chunks < 0:
            raise ValueError(
                "keep_recent_chunks / sink_keep_chunks must be >= 0"
            )
        return self

    def describe(self) -> str:
        if not self.enabled:
            return "off"
        return (
            f"int{self.bits} method={self.quant_method} "
            f"r={self.compression_ratio} "
            f"max_scale_seq={self.max_scale_seq_len} "
            f"pack={self.pack_to_int32} recent={self.keep_recent_chunks} "
            f"sink={self.sink} sink_keep={self.sink_keep_chunks}"
        )

    @classmethod
    def from_dict(cls, kwargs: dict) -> VARQKVQuantArgs:
        """Build from flat CLI kwargs (dest names ``kv_cache_varq*``)."""
        master = kwargs.get("kv_cache_varq")
        if master is None:
            stray = sorted(
                k
                for k, v in kwargs.items()
                if k.startswith("kv_cache_varq_")
                and k != "kv_cache_varq_config"
                and v is not None
            )
            if stray:
                raise ValueError(
                    f"{stray} require --kv-cache-varq to be set "
                    f"(off|int8|int6|int4|int3|int2)"
                )
            return cls()
        inst = cls(bits=_parse_bits(master))
        if kwargs.get("kv_cache_varq_quant_method") is not None:
            inst.quant_method = str(kwargs["kv_cache_varq_quant_method"])
        if kwargs.get("kv_cache_varq_compression_ratio") is not None:
            inst.compression_ratio = float(kwargs["kv_cache_varq_compression_ratio"])
        if kwargs.get("kv_cache_varq_max_scale_seq_len") is not None:
            inst.max_scale_seq_len = int(kwargs["kv_cache_varq_max_scale_seq_len"])
        if kwargs.get("kv_cache_varq_pack_to_int32") is not None:
            inst.pack_to_int32 = bool(kwargs["kv_cache_varq_pack_to_int32"])
        if kwargs.get("kv_cache_varq_dequant_dtype") is not None:
            inst.dequant_dtype = str(kwargs["kv_cache_varq_dequant_dtype"])
        if kwargs.get("kv_cache_varq_keep_recent") is not None:
            inst.keep_recent_chunks = int(kwargs["kv_cache_varq_keep_recent"])
        if kwargs.get("kv_cache_varq_sink_keep") is not None:
            inst.sink_keep_chunks = int(kwargs["kv_cache_varq_sink_keep"])
        if kwargs.get("kv_cache_varq_sink") is not None:
            inst.sink = bool(kwargs["kv_cache_varq_sink"])
        return inst.validate()
