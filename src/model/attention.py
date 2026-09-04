from __future__ import annotations

import inspect

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd.profiler import record_function

from .norm import HeadwiseRMSNorm
from .rope import RotaryEmbedding


class GQAAttention(nn.Module):
    """Grouped-query self-attention backed by PyTorch SDPA."""

    def __init__(self, config) -> None:
        super().__init__()
        self.num_q_heads = config.num_q_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.q_dim = config.q_dim
        self.kv_dim = config.kv_dim
        self.q_proj = nn.Linear(config.d_model, self.q_dim, bias=config.use_bias)
        self.k_proj = nn.Linear(config.d_model, self.kv_dim, bias=config.use_bias)
        self.v_proj = nn.Linear(config.d_model, self.kv_dim, bias=config.use_bias)
        # QK-RMSNorm is applied independently to each head vector. The
        # per-projected-channel scales preserve the frozen parameter count.
        self.q_norm = HeadwiseRMSNorm(self.num_q_heads, self.head_dim, config.norm_eps)
        self.k_norm = HeadwiseRMSNorm(self.num_kv_heads, self.head_dim, config.norm_eps)
        self.rope = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_theta)
        self.out_proj = nn.Linear(self.q_dim, config.d_model, bias=config.use_bias)
        try:
            self._native_gqa = "enable_gqa" in inspect.signature(F.scaled_dot_product_attention).parameters
        except (TypeError, ValueError):
            # Builtin signatures are unavailable on some PyTorch builds. The
            # project baseline is >=2.14, where enable_gqa is present.
            self._native_gqa = tuple(int(part) for part in torch.__version__.split(".")[:2]) >= (2, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, "profile_ranges", False):
            with record_function("model.attention"):
                return self._forward_impl(x)
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_q_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k)
        if self._native_gqa:
            attended = F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=True, enable_gqa=True
            )
        else:
            k = k.repeat_interleave(self.num_q_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_q_heads // self.num_kv_heads, dim=1)
            attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        attended = attended.transpose(1, 2).contiguous().view(batch, seq_len, self.q_dim)
        return self.out_proj(attended)
