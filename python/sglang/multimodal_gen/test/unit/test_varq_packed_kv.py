# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the VAR-Q packed KV-cache (VARQPackedCausalKVCache).

- config parsing + defaults: CPU, no deps.
- packed-storage equivalence vs the dense cache (quantization disabled -> bf16
  segments): CPU-only, exercises the segment / sliding-window / eviction /
  reconstruction logic bit-exactly without needing VAR-Q or a GPU.
- quantized memory saving + reconstruction: requires CUDA + VAR-Q.
"""

import importlib.util
import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.quantization.qvg_kv import QVGKVQuantArgs
from sglang.multimodal_gen.configs.quantization.varq_kv import VARQKVQuantArgs
from sglang.multimodal_gen.runtime.layers.kvcache.causal_attention_cache import (
    CausalSelfAttentionKVCache,
)
from sglang.multimodal_gen.runtime.layers.kvcache.varq_packed_cache import (
    VARQPackedCausalKVCache,
)

_HAS_VARQ = importlib.util.find_spec("VAR_Q") is not None
_HAS_CUDA = torch.cuda.is_available()


def _base(B, W, H, D, sink, dev):
    return CausalSelfAttentionKVCache(
        k=torch.zeros(B, W, H, D, device=dev),
        v=torch.zeros(B, W, H, D, device=dev),
        global_end_index=torch.zeros(1, dtype=torch.long, device=dev),
        local_end_index=torch.zeros(1, dtype=torch.long, device=dev),
        cache_size=W,
        sink_tokens=sink,
        attention_window_size=W,
    )


def _packed(B, W, H, D, sink, dev, quant, use_int_indices=False):
    return VARQPackedCausalKVCache(
        batch_size=B,
        cache_size=W,
        num_heads=H,
        head_dim=D,
        dtype=torch.float32,
        device=torch.device(dev),
        use_int_indices=use_int_indices,
        global_end_index=torch.zeros(1, dtype=torch.long, device=dev),
        local_end_index=torch.zeros(1, dtype=torch.long, device=dev),
        sink_tokens=sink,
        attention_window_size=W,
        quant_args=quant,
    )


def _replay(cache, chunks, C, nsteps, rwt, data):
    views = []
    for ci in range(chunks):
        g = ci * C
        for step in range(nsteps):
            k, v = data[(ci, step)]
            view = cache.update_and_get_attention_kv(
                key=k, value=v, current_chunk_start=g, recent_window_tokens=rwt
            )
        views.append((view.k.clone(), view.v.clone()))
    return views


class TestVARQKVQuantArgs(unittest.TestCase):
    def test_off_by_default(self):
        self.assertFalse(VARQKVQuantArgs().enabled)
        self.assertFalse(VARQKVQuantArgs.from_dict({}).enabled)

    def test_master_flag_defaults(self):
        a = VARQKVQuantArgs.from_dict({"kv_cache_varq": "int4"})
        self.assertTrue(a.enabled)
        self.assertEqual(
            (
                a.bits,
                a.quant_method,
                a.compression_ratio,
                a.max_scale_seq_len,
                a.pack_to_int32,
                a.keep_recent_chunks,
                a.sink,
                a.sink_keep_chunks,
            ),
            (4, "VARQ", 1.0, None, True, 1, True, 0),
        )

    def test_all_supported_bit_widths(self):
        for name, bits in (
            ("int8", 8),
            ("int6", 6),
            ("int4", 4),
            ("int3", 3),
            ("int2", 2),
        ):
            self.assertEqual(
                VARQKVQuantArgs.from_dict({"kv_cache_varq": name}).bits, bits
            )

    def test_overrides_and_validation(self):
        a = VARQKVQuantArgs.from_dict(
            {
                "kv_cache_varq": "int2",
                "kv_cache_varq_keep_recent": 3,
                "kv_cache_varq_sink": 0,
            }
        )
        self.assertEqual((a.bits, a.keep_recent_chunks, a.sink), (2, 3, False))
        with self.assertRaises(ValueError):
            VARQKVQuantArgs.from_dict({"kv_cache_varq": "int5"})
        with self.assertRaises(ValueError):
            VARQKVQuantArgs(bits=4, compression_ratio=0.0).validate()
        with self.assertRaises(ValueError):
            VARQKVQuantArgs(bits=4, keep_recent_chunks=-1).validate()

    def test_sub_flags_without_master_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "require --kv-cache-varq"):
            VARQKVQuantArgs.from_dict({"kv_cache_varq_keep_recent": 3})

    def test_describe_mentions_bits(self):
        self.assertEqual(VARQKVQuantArgs().describe(), "off")
        self.assertIn("int4", VARQKVQuantArgs(bits=4).describe())


class TestPackedStorageEquivalence(unittest.TestCase):
    """quantization disabled -> packed cache stores bf16 segments; must match the
    dense cache bit-for-bit across the sliding-window + eviction lifecycle."""

    def _run(self, rwt):
        B, H, D, C, sink, chunks, nsteps = 1, 2, 4, 8, 8, 8, 3
        W = C * 5
        dev = "cpu"
        torch.manual_seed(0)
        data = {
            (ci, s): (torch.randn(B, C, H, D), torch.randn(B, C, H, D))
            for ci in range(chunks)
            for s in range(nsteps)
        }
        base = _base(B, W, H, D, sink, dev)
        packed = _packed(B, W, H, D, sink, dev, VARQKVQuantArgs())  # disabled
        ob = _replay(base, chunks, C, nsteps, rwt, data)
        op = _replay(packed, chunks, C, nsteps, rwt, data)
        for i, ((bk, bv), (pk, pv)) in enumerate(zip(ob, op)):
            self.assertEqual(bk.shape, pk.shape, f"chunk {i} shape")
            self.assertTrue(torch.equal(bk, pk), f"chunk {i} K mismatch")
            self.assertTrue(torch.equal(bv, pv), f"chunk {i} V mismatch")

    def test_full_window(self):
        self._run(rwt=None)

    def test_recent_window_selection(self):
        self._run(rwt=16)

    def test_zero_recent_window_keeps_sink_plus_current(self):
        self._run(rwt=0)

    def test_head_slice_writes_only_its_heads(self):
        B, H, D, C, W, dev = 1, 4, 4, 4, 12, "cpu"
        cache = _packed(B, W, H, D, sink=0, dev=dev, quant=VARQKVQuantArgs())
        key = torch.randn(B, C, 2, D)
        value = torch.randn(B, C, 2, D)
        view = cache.update_and_get_attention_kv(
            key=key,
            value=value,
            current_chunk_start=0,
            cache_head_start=1,
        )
        # heads 1..2 written, heads 0 and 3 stay zero
        self.assertTrue(torch.equal(view.k[:, :, 1:3, :], key))
        self.assertEqual(float(view.k[:, :, 0, :].abs().sum()), 0.0)
        self.assertEqual(float(view.k[:, :, 3, :].abs().sum()), 0.0)

    def test_view_indices_and_debug_name(self):
        cache = _packed(1, 8, 2, 4, sink=0, dev="cpu", quant=VARQKVQuantArgs())
        key = torch.zeros(1, 4, 1, 4)
        with self.assertRaisesRegex(ValueError, "VAR-Q packed KV cache"):
            cache.update_and_get_attention_kv(key=key, value=key, current_chunk_start=0)
        with self.assertRaisesRegex(ValueError, "custom VAR-Q"):
            cache.update_and_get_attention_kv(
                key=key,
                value=key,
                current_chunk_start=0,
                debug_name="custom VAR-Q",
            )
        view = cache.update_and_get_attention_kv(
            key=torch.zeros(1, 4, 2, 4),
            value=torch.zeros(1, 4, 2, 4),
            current_chunk_start=0,
        )
        self.assertEqual((view.local_start_index, view.local_end_index), (0, 4))
        view = cache.update_and_get_attention_kv(
            key=torch.ones(1, 4, 2, 4),
            value=torch.ones(1, 4, 2, 4),
            current_chunk_start=4,
        )
        self.assertEqual((view.local_start_index, view.local_end_index), (4, 8))
        self.assertEqual((view.visible_local_end, view.visible_global_end), (8, 8))

    def test_enabled_cpu_codec_layout_bytes_and_reset(self):
        calls = []

        def encode(tensor, **kwargs):
            calls.append((tensor.shape, kwargs["kv_role"]))
            return {
                "packed": tensor.to(torch.int8),
                "scale": torch.ones(1, 1, tensor.shape[2], tensor.shape[3]),
                "pack_meta": None,
                "quant_meta": {"qkv_format": "BLHc"},
            }

        def decode(packed, scale, pack_meta, quant_meta, dequant_dtype):
            del scale, pack_meta, quant_meta
            return packed.to(dequant_dtype)

        args = VARQKVQuantArgs(
            bits=4, keep_recent_chunks=0, sink=False, pack_to_int32=False
        )
        cache = _packed(1, 12, 2, 3, sink=0, dev="cpu", quant=args)
        with patch(
            "sglang.multimodal_gen.runtime.layers.kvcache.varq_packed_cache._varq_functions",
            return_value=(encode, decode),
        ):
            chunk = torch.arange(24, dtype=torch.float32).reshape(1, 4, 2, 3)
            cache.update_and_get_attention_kv(
                key=chunk, value=chunk, current_chunk_start=0
            )
            cache.update_and_get_attention_kv(
                key=chunk, value=chunk, current_chunk_start=4
            )

        self.assertEqual(calls, [((1, 4, 2, 3), "k"), ((1, 4, 2, 3), "v")])
        packed_bytes = 2 * (4 * 2 * 3 + 2 * 3 * 4)
        current_bytes = 2 * chunk.numel() * chunk.element_size()
        self.assertEqual(cache.resident_nbytes(), packed_bytes + current_bytes)
        cache.pin_current_chunk(4)
        cache.reset_indices()
        self.assertEqual(cache.resident_nbytes(), 0)
        self.assertEqual((cache._pinned_start, cache._pinned_len), (-1, 0))

    def test_host_index_cursors_reset_without_device_updates(self):
        cache = _packed(
            B=1,
            W=16,
            H=2,
            D=4,
            sink=0,
            dev="cpu",
            quant=VARQKVQuantArgs(),
            use_int_indices=True,
        )
        cache.update_and_get_attention_kv(
            key=torch.ones(1, 4, 2, 4),
            value=torch.ones(1, 4, 2, 4),
            current_chunk_start=0,
        )
        self.assertEqual(cache.global_end_index_int, 4)
        self.assertEqual(cache.local_end_index_int, 4)
        self.assertEqual(int(cache.global_end_index.item()), 0)
        self.assertEqual(int(cache.local_end_index.item()), 0)

        cache.reset_indices()
        self.assertEqual(cache.global_end_index_int, 0)
        self.assertEqual(cache.local_end_index_int, 0)


@unittest.skipUnless(
    _HAS_CUDA and _HAS_VARQ,
    "needs CUDA + VAR-Q",
)
class TestPackedQuantMemory(unittest.TestCase):
    def test_saves_memory_and_reconstructs(self):
        B, H, D, C, sink, chunks, nsteps = 1, 8, 128, 512, 512, 12, 2
        W = C * 6
        dev = "cuda"
        q = VARQKVQuantArgs.from_dict({"kv_cache_varq": "int4"})
        torch.manual_seed(1)
        data = {
            (ci, s): (
                torch.randn(B, C, H, D, device=dev),
                torch.randn(B, C, H, D, device=dev),
            )
            for ci in range(chunks)
            for s in range(nsteps)
        }
        base = _base(B, W, H, D, sink, dev)
        packed = _packed(B, W, H, D, sink, dev, q)
        ob = _replay(base, chunks, C, nsteps, None, data)
        op = _replay(packed, chunks, C, nsteps, None, data)
        dense_bytes = base.k.numel() * base.k.element_size() * 2
        self.assertLess(
            packed.resident_nbytes(),
            dense_bytes * 0.7,
            "packed cache should save >30% resident memory at int4",
        )
        bk, _ = ob[-1]
        pk, _ = op[-1]
        self.assertEqual(bk.shape, pk.shape)
        mse = ((bk - pk) ** 2).mean().item()
        self.assertLess(mse, 1.0, "int4 reconstruction error too large")


if __name__ == "__main__":
    unittest.main()
