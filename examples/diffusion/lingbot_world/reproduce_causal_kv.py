#!/usr/bin/env python3
"""Reproduce one 81-frame LingBot World 2.0 causal session.

This runner is intentionally explicit: it performs no model import, download, or
GPU work unless --execute is supplied. The seven requests share one realtime
session and seed; decoded chunk lengths must be 9,12,12,12,12,12,12.
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

MODEL_ID = "robbyant/lingbot-world-v2-14b-causal-fast-diffusers"
EXPECTED_CHUNK_FRAMES = (9, 12, 12, 12, 12, 12, 12)
FLOW_SHIFT = 5.0
DMD_STEPS = (1000, 750, 500, 250)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("bf16", "varq", "qvg"))
    p.add_argument("--image", type=Path, required=True, help="Official examples/03/image.jpg or another reviewer-supplied image")
    p.add_argument("--prompt", required=True, help="Prompt used for every causal chunk")
    p.add_argument("--model-path", default=MODEL_ID)
    p.add_argument("--output", type=Path, required=True, help="Fresh output directory")
    p.add_argument("--num-gpus", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--execute", action="store_true", help="Run inference; without this flag print the manifest only")
    return p


def mode_config(mode: str):
    """Return the typed cache config; imports are deferred to --execute."""
    if mode == "bf16":
        return None, None
    from sglang.multimodal_gen.configs.quantization.qvg_kv import QVGKVQuantArgs
    from sglang.multimodal_gen.configs.quantization.varq_kv import VARQKVQuantArgs
    if mode == "varq":
        return VARQKVQuantArgs.from_dict({"kv_cache_varq": "int4"}), None
    return None, QVGKVQuantArgs.from_dict({"kv_cache_quant": "int4"})


def manifest(args: argparse.Namespace) -> dict:
    return {
        "mode": args.mode,
        "model_id": MODEL_ID,
        "model_path": str(args.model_path),
        "image": str(args.image),
        "prompt": args.prompt,
        "seed": args.seed,
        "num_gpus": args.num_gpus,
        "flow_shift": FLOW_SHIFT,
        "dmd_denoising_steps": list(DMD_STEPS),
        "warp": True,
        "chunk_frames": list(EXPECTED_CHUNK_FRAMES),
        "total_frames": 81,
        "session": "one realtime_session_id; block_idx 0..6; cache window 18 frames",
        "parity": "forward-key trajectory and realtime per-chunk RNG; not official poses/noise replay",
        "timing_scope": "per-request wall time includes cold/JIT work and output decode/write; do not treat as warmed benchmark",
    }


def execute(args: argparse.Namespace) -> dict:
    if not args.image.is_file():
        raise FileNotFoundError(f"image not found: {args.image}; supply official examples/03/image.jpg (no teaser fallback)")
    if args.num_gpus < 1 or 40 % args.num_gpus:
        raise ValueError("--num-gpus must be a positive divisor of 40 attention heads")
    if args.output.exists():
        raise FileExistsError(f"output exists: {args.output}; use a fresh directory")

    from PIL import Image
    import imageio.v2 as imageio
    from sglang.multimodal_gen.configs.pipeline_configs.lingbot_world import LingBotWorldV2CausalDMDConfig
    from sglang.multimodal_gen.runtime.entrypoints import diffusion_generator as dg

    with Image.open(args.image) as image:
        image.verify()
    varq_config, qvg_config = mode_config(args.mode)
    if args.mode == "qvg":
        import torch
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 9:
            raise RuntimeError("This runner requires Hopper or newer for QVG; the experimental A100 BF16-scale patch is not included")

    config = LingBotWorldV2CausalDMDConfig()
    assert config.flow_shift == FLOW_SHIFT and config.dmd_denoising_steps == list(DMD_STEPS)
    assert config.warp_denoising_step is True
    args.output.mkdir(parents=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest(args), indent=2), encoding="utf-8")
    kwargs = dict(model_path=args.model_path, model_id=MODEL_ID,
        pipeline_class_name="LingBotWorldCausalDMDPipeline", pipeline_config=config,
        backend="sglang", num_gpus=args.num_gpus, ulysses_degree=args.num_gpus,
        ring_degree=1, dit_precision="bf16", vae_precision="bf16",
        dit_cpu_offload=False, text_encoder_cpu_offload=False, vae_cpu_offload=False,
        enable_torch_compile=False, warmup_mode="off")
    if varq_config is not None:
        kwargs["kv_cache_varq_config"] = varq_config
    if qvg_config is not None:
        kwargs["kv_cache_quant_config"] = qvg_config

    generator = dg.DiffGenerator.from_pretrained(**kwargs)
    original_prepare = dg.prepare_request
    session_id = f"lingbot-review-{args.mode}-{uuid.uuid4()}"
    current_block = 0
    records = []
    try:
        def prepare(**kwargs):
            request = original_prepare(**kwargs)
            request.realtime_session_id = session_id
            request.block_idx = current_block
            request.realtime_causal_kv_cache_num_frames = 18
            return request
        dg.prepare_request = prepare
        for current_block in range(7):
            name = f"{args.mode}_chunk{current_block}.mp4"
            started = time.perf_counter()
            result = generator.generate({"prompt": args.prompt, "image_path": str(args.image),
                "task_type": "i2v", "width": 832, "height": 480, "num_frames": 81,
                "chunk_size": 3, "actions": [["w"], ["w"], ["w"]],
                "guidance_scale": 1.0, "guidance_scale_2": 1.0, "num_inference_steps": 4,
                "seed": args.seed, "fps": 16, "realtime_causal_kv_cache_num_frames": 18,
                "quality": "exact", "return_file_paths_only": True,
                "output_path": str(args.output), "output_file_name": name,
                "perf_dump_path": str(args.output / f"{args.mode}_chunk{current_block}_perf.json")})
            if result is None:
                raise RuntimeError(f"generation failed at chunk {current_block}")
            count = 0
            with imageio.get_reader(args.output / name) as reader:
                for _ in reader:
                    count += 1
            record = {"chunk": current_block, "frames": count, "seconds": time.perf_counter() - started}
            records.append(record)
            if count != EXPECTED_CHUNK_FRAMES[current_block]:
                raise RuntimeError(f"decoded chunk has {count} frames, expected {EXPECTED_CHUNK_FRAMES[current_block]}")
            (args.output / "progress.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    finally:
        dg.prepare_request = original_prepare
        generator.shutdown()
    summary = {"mode": args.mode, "session_id": session_id, "chunks": records,
               "decoded_frames": sum(r["frames"] for r in records), "timing_scope": manifest(args)["timing_scope"]}
    if summary["decoded_frames"] != 81:
        raise RuntimeError(f"final decoded frame count is {summary['decoded_frames']}, expected 81")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parser().parse_args()
    print(json.dumps(execute(args) if args.execute else manifest(args), indent=2))


if __name__ == "__main__":
    main()
