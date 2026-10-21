"""A small real LLaVA model for network-free integration checks."""

import pytest
import torch
from torch import nn

from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM


class TinyVision(nn.Module):
    def forward(self, pixels):
        pooled = pixels.mean(dim=(-1, -2))
        fourth = pooled.mean(dim=1, keepdim=True)
        patch = torch.cat((pooled, fourth), dim=1)
        return torch.stack((patch, patch * 0.5), dim=1)


@pytest.fixture
def model():
    torch.manual_seed(73)
    config = LlavaConfig(
        vocab_size=41,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    config.mm_use_im_start_end = False
    config.tune_mm_mlp_adapter = False
    result = LlavaLlamaForCausalLM(config).eval()
    result.model.vision_tower = TinyVision()
    result.model.mm_projector = nn.Linear(4, 32, bias=False)
    return result
