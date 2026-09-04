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
