# SPDX-License-Identifier: Apache-2.0
"""Causal Wan T2V pipeline driven by the normal (non-DMD) denoising schedule.

`WanCausalDMDPipeline` is bound to `SelfForcingWanT2V480PConfig`, which hardcodes
the 4-step DMD schedule used by distilled Self-Forcing checkpoints. Running base
Wan weights through that schedule yields noise, so this pipeline pairs the same
causal stage with a config that keeps the sampler's normal step count.

Each pipeline module must expose its own ``EntryClass``; the pipeline registry
discovers one pipeline per module.
"""

from sglang.multimodal_gen.configs.pipeline_configs.wan import CausalWanT2V480PConfig
from sglang.multimodal_gen.runtime.pipelines.wan_causal_dmd_pipeline import (
    WanCausalDMDPipeline,
)


class CausalWanT2VPipeline(WanCausalDMDPipeline):
    pipeline_name = "CausalWanT2VPipeline"
    pipeline_config_cls = CausalWanT2V480PConfig


EntryClass = CausalWanT2VPipeline
