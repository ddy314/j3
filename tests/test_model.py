from __future__ import annotations

import torch

from src.model import DecoderLM, ModelConfig
from src.model.attention import GQAAttention
from src.model.mlp import XIELU
from src.model.rope import RotaryEmbedding


def test_frozen_parameter_count_and_breakdown() -> None:
    config = ModelConfig()
    model = DecoderLM(config)
    report = model.parameter_report()
    assert report["total"] == 49_001_408
    assert report["target"] == 49_001_408
    assert report["matches_target"] is True
    assert report["breakdown"]["total"] == 49_001_408


def test_output_shape_and_loss(tiny_config: ModelConfig) -> None:
    model = DecoderLM(tiny_config)
    input_ids = torch.randint(tiny_config.vocab_size, (2, 7))
    output = model(input_ids, input_ids.roll(-1, dims=1))
    assert output.logits.shape == (2, 7, tiny_config.vocab_size)
    assert output.loss is not None and output.loss.ndim == 0


def test_causal_mask_blocks_future_tokens(tiny_config: ModelConfig) -> None:
    torch.manual_seed(1)
    model = DecoderLM(tiny_config).eval()
    prefix = torch.tensor([[1, 2, 3]])
    first = torch.cat((prefix, torch.tensor([[4, 5, 6, 7]])), dim=1)
    second = torch.cat((prefix, torch.tensor([[12, 13, 14, 15]])), dim=1)
    with torch.no_grad():
        first_logits = model(first).logits
        second_logits = model(second).logits
    torch.testing.assert_close(first_logits[:, :3], second_logits[:, :3], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(first_logits[:, 3], second_logits[:, 3])


def test_gqa_dimensions(tiny_config: ModelConfig) -> None:
    attention = GQAAttention(tiny_config)
    assert attention.num_q_heads == 4
    assert attention.num_kv_heads == 2
    x = torch.randn(2, 5, tiny_config.d_model)
    assert attention(x).shape == x.shape


def test_rope_preserves_shape(tiny_config: ModelConfig) -> None:
    rope = RotaryEmbedding(tiny_config.head_dim, 16)
    q = torch.randn(2, 4, 5, 8)
    k = torch.randn(2, 2, 5, 8)
    rotated_q, rotated_k = rope(q, k)
    assert rotated_q.shape == q.shape
    assert rotated_k.shape == k.shape
    assert not torch.equal(rotated_q, q)


def test_xielu_is_differentiable() -> None:
    activation = XIELU()
    x = torch.tensor([[-2.0, -0.1, 0.0, 0.1, 2.0]], requires_grad=True)
    y = activation(x)
    y.sum().backward()
    assert torch.isfinite(y).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert activation.alpha_p.grad is not None
    assert activation.alpha_n.grad is not None


def test_tied_embedding_and_output_residual(tiny_config: ModelConfig) -> None:
    model = DecoderLM(tiny_config)
    assert model.token_embedding.weight.requires_grad
    assert model.parameter_count == tiny_config.parameter_count
    before = model.output_up.weight.detach().clone()
    model.output_up.weight.data.add_(0.1)
    assert not torch.equal(before, model.output_up.weight)
    assert model.output_down.weight.shape == (tiny_config.output_residual_rank, tiny_config.d_model)
    assert model.output_up.weight.shape == (tiny_config.vocab_size, tiny_config.output_residual_rank)


def test_backward_and_bf16_autocast(tiny_config: ModelConfig) -> None:
    model = DecoderLM(tiny_config)
    input_ids = torch.randint(tiny_config.vocab_size, (2, 8))
    labels = torch.randint(tiny_config.vocab_size, (2, 8))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(input_ids, labels)
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    assert model.token_embedding.weight.grad is not None
    assert torch.isfinite(model.token_embedding.weight.grad).all()


def test_generation(tiny_config: ModelConfig) -> None:
    model = DecoderLM(tiny_config).eval()
    prompt = torch.tensor([[1, 2, 3]])
    generated = model.generate(prompt, max_new_tokens=2)
    assert generated.shape == (1, 5)
