from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd.profiler import record_function


class RMSNorm(nn.Module):
    """RMSNorm using PyTorch's native implementation when available."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, "profile_ranges", False):
            with record_function("model.rms_norm"):
                return self._forward_impl(x)
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        native = getattr(F, "rms_norm", None)
        if native is not None:
            # Autocast keeps parameters in FP32. Passing an FP32 weight to
            # the BF16 native kernel disables its fused implementation, so
            # make the dtype conversion explicit and let compile fold it.
            weight = self.weight if self.weight.dtype == x.dtype else self.weight.to(dtype=x.dtype)
            return native(x, (self.dim,), weight, self.eps)
        variance = x.float().square().mean(dim=-1, keepdim=True)
        return (x * torch.rsqrt(variance + self.eps).to(dtype=x.dtype)) * self.weight


class HeadwiseRMSNorm(nn.Module):
    """RMSNorm over the last dimension of each attention head.

    The scale is kept per projected channel (``[num_heads, head_dim]``), so
    the frozen model retains the original Q/K parameter budget while the RMS
    statistic is correctly computed independently for every head.
    """

    def __init__(self, num_heads: int, head_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_heads, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, "profile_ranges", False):
            with record_function("model.rms_norm"):
                return self._forward_impl(x)
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.num_heads, self.head_dim):
            raise ValueError(
                f"HeadwiseRMSNorm expected [..., {self.num_heads}, {self.head_dim}], got {tuple(x.shape)}"
            )
        variance = x.float().square().mean(dim=-1, keepdim=True)
        weight = self.weight if self.weight.dtype == x.dtype else self.weight.to(dtype=x.dtype)
        return (x * torch.rsqrt(variance + self.eps).to(dtype=x.dtype)) * weight
