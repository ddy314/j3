from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd.profiler import record_function

from .attention import GQAAttention
from .config import ModelConfig
from .mlp import MLP
from .norm import RMSNorm


class CausalLMOutput(NamedTuple):
    logits: torch.Tensor
    loss: torch.Tensor | None = None


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attention = GQAAttention(config)
        self.mlp_norm = RMSNorm(config.d_model, config.norm_eps)
        self.mlp = MLP(
            config.d_model,
            config.mlp_hidden_dim,
            config.use_bias,
            config.xielu_alpha_p_init,
            config.xielu_alpha_n_init,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attn_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class DecoderLM(nn.Module):
    """D32-1216-R12 decoder-only causal language model."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.blocks = nn.ModuleList(DecoderBlock(self.config) for _ in range(self.config.depth))
        self.final_norm = RMSNorm(self.config.d_model, self.config.norm_eps)
        self.output_down = nn.Linear(self.config.d_model, self.config.output_residual_rank, bias=False)
        self.output_up = nn.Linear(self.config.output_residual_rank, self.config.vocab_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = self.config.initializer_range
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                bias = getattr(module, "bias", None)
                if bias is not None:
                    nn.init.zeros_(bias)
        for module in self.modules():
            if isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
        # Keep the low-rank residual head a small correction at initialization.
        nn.init.normal_(self.output_down.weight, mean=0.0, std=std)
        nn.init.normal_(self.output_up.weight, mean=0.0, std=std / math.sqrt(self.config.output_residual_rank))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def parameter_report(self) -> dict[str, object]:
        breakdown = self.config.parameter_breakdown()
        actual = self.parameter_count
        return {
            "model_name": self.config.model_name,
            "total": actual,
            "target": self.config.parameter_count,
            "matches_target": actual == self.config.parameter_count,
            "breakdown": breakdown,
        }

    @staticmethod
    def estimate_flops(config: ModelConfig, batch_size: int, seq_len: int) -> dict[str, float]:
        """Estimate forward and train FLOPs; useful for interpreting throughput."""

        tokens = batch_size * seq_len
        projection_flops = 2 * config.d_model * (4 * config.d_model + 2 * config.kv_dim + 2 * config.mlp_hidden_dim)
        attention_flops = 4 * seq_len * config.q_dim
        per_token_forward = config.depth * (projection_flops + attention_flops)
        lm_head_flops = 2 * config.d_model * config.vocab_size
        residual_flops = 2 * config.d_model * config.output_residual_rank + 2 * config.output_residual_rank * config.vocab_size
        forward = tokens * per_token_forward + tokens * (lm_head_flops + residual_flops)
        return {
            "forward": float(forward),
            "train_approx_3x_forward": float(3 * forward),
            "forward_per_token": float(forward / tokens),
        }

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError(f"sequence length exceeds max_seq_len={self.config.max_seq_len}")
        hidden = self.token_embedding(input_ids)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)
        if getattr(self, "profile_ranges", False):
            with record_function("model.lm_head_loss"):
                logits = F.linear(hidden, self.token_embedding.weight)
                logits = logits + self.output_up(self.output_down(hidden))
        else:
            logits = F.linear(hidden, self.token_embedding.weight)
            logits = logits + self.output_up(self.output_down(hidden))
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            if getattr(self, "profile_ranges", False):
                with record_function("model.loss"):
                    loss = F.cross_entropy(logits.float().reshape(-1, self.config.vocab_size), labels.reshape(-1))
            else:
                loss = F.cross_entropy(logits.float().reshape(-1, self.config.vocab_size), labels.reshape(-1))
        return CausalLMOutput(logits=logits, loss=loss)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.max_seq_len :]
            logits = self(context).logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            input_ids = torch.cat((input_ids, next_token), dim=1)
        return input_ids
