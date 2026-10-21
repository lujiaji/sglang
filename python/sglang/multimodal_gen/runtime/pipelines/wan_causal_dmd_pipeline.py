# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Wan causal DMD pipeline implementation.

This module wires the causal DMD denoising stage into the modular pipeline.
"""

from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline import LoRAPipeline

# isort: off
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    CausalDMDDenoisingStage,
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs

# isort: on


class WanCausalDMDDenoisingStage(CausalDMDDenoisingStage):
    def _supports_qvg_kv_cache_quantization(self) -> bool:
        return True

    def _supports_varq_kv_cache_quantization(self) -> bool:
        return True


class WanCausalDMDPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "WanCausalDMDPipeline"

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self.add_stage(InputValidationStage())
        self.add_standard_text_encoding_stage()
        self.add_standard_latent_preparation_stage()

        self.add_stage(
            WanCausalDMDDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        self.add_standard_decoding_stage()


EntryClass = WanCausalDMDPipeline
