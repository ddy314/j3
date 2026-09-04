from __future__ import annotations

import torch
from torch import nn
from torch.autograd.profiler import record_function


class RotaryEmbedding(nn.Module):
    """Cached half-rotation RoPE for tensors shaped [B, H, T, D]."""

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE head_dim must be even")
        self.head_dim = head_dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cache(max_seq_len, device=inv_freq.device)

    def _set_cache(self, seq_len: int, device: torch.device) -> None:
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq.to(device=device))
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def _cache(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.shape[0] or self.cos_cached.device != device:
            self._set_cache(max(seq_len, self.cos_cached.shape[0]), device)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if getattr(self, "profile_ranges", False):
            with record_function("model.rope"):
                return self._forward_impl(q, k)
        return self._forward_impl(q, k)

    def _forward_impl(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[-2]
        cos, sin = self._cache(seq_len, q.device)
        cos = cos.to(dtype=q.dtype)[None, None, :, :]
        sin = sin.to(dtype=q.dtype)[None, None, :, :]
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)

    @staticmethod
    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        even = x[..., 0::2]
        odd = x[..., 1::2]
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)
