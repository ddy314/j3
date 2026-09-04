from __future__ import annotations

import pytest

from src.model import ModelConfig


@pytest.fixture()
def tiny_config() -> ModelConfig:
    return ModelConfig(
        model_name="tiny-test",
        vocab_size=64,
        d_model=32,
        depth=2,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=8,
        mlp_hidden_dim=48,
        output_residual_rank=3,
        max_seq_len=16,
    )
