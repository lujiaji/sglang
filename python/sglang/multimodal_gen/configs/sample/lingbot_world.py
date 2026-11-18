# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
# Adapted from: https://github.com/Robbyant/lingbot-world

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Any

from sglang.multimodal_gen.configs.sample.wan import Wan2_2_I2V_A14B_SamplingParam


@dataclass
class LingBotWorldSamplingParams(Wan2_2_I2V_A14B_SamplingParam):
    negative_prompt: str | None = None
    actions: list[list[str]] | None = None
    chunk_size: int | None = None
    guidance_scale: float = 5.0
    guidance_scale_2: float = 5.0
    num_inference_steps: int = 70
    num_frames: int = 117
    fps: int = 16
    realtime_causal_kv_cache_num_frames: int | None = None

    def _adjust(self, server_args):
        enable_sequence_shard = self.enable_sequence_shard
        if enable_sequence_shard is None or enable_sequence_shard:
            self.adjust_frames = False
        super()._adjust(server_args)
        if enable_sequence_shard is None or enable_sequence_shard:
            self.enable_sequence_shard = True
            self.adjust_frames = False
        if self.chunk_size is None:
            self.chunk_size = max(
                1,
                int(
                    server_args.pipeline_config.dit_config.arch_config.num_frames_per_block
                ),
            )
        if self.actions is not None:
            self.condition_inputs["camera_actions"] = self.actions
        if self.chunk_size is not None:
            self.realtime_chunk_size = self.chunk_size
        if self.realtime_causal_kv_cache_num_frames is None:
            self.realtime_causal_kv_cache_num_frames = getattr(
                server_args.pipeline_config,
                "realtime_causal_kv_cache_num_frames",
                None,
            )

    def apply_request_extra(self, req: Any) -> None:
        super().apply_request_extra(req)
        if self.realtime_causal_kv_cache_num_frames is not None:
            req.realtime_causal_kv_cache_num_frames = (
                self.realtime_causal_kv_cache_num_frames
            )
