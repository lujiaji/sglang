"""Regression for full repo IDs selecting native LingBot config defaults."""
import pytest

from sglang.multimodal_gen import registry


@pytest.mark.parametrize("model_id", [
    "robbyant/lingbot-world-v2-14b-causal-fast-diffusers",
    "lingbot-world-v2-14b-causal-fast-diffusers",
])
def test_lingbot_explicit_model_id_preserves_arch_defaults(model_id):
    registry._discover_and_register_configs()
    info = registry._get_config_info("/local/renamed-lingbot-checkpoint", model_id)
    assert info is not None
    assert info.pipeline_config_cls.__name__ == "LingBotWorldV2CausalDMDConfig"
    assert info.sampling_param_cls.__name__ == "LingBotWorldSamplingParams"
    config = info.pipeline_config_cls()
    assert config.dit_config.text_len == 512
    assert config.dit_config.num_frames_per_block == 3
    assert config.flow_shift == 5.0
