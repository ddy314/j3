from __future__ import annotations

import torch
from torch import nn
from torch.autograd.profiler import record_function


class XIELU(nn.Module):
    """Trainable Expanded Integral of ELU activation.

    The scalar parameters are deliberately per block, matching the frozen
    architecture count. The piecewise function is continuous at zero:

        x > 0:  alpha_p * x^2 + 0.5*x
        x <= 0: alpha_n*(exp(x)-1) - alpha_n*x + 0.5*x
    """

    def __init__(self, alpha_p: float = 0.5, alpha_n: float = 1.0) -> None:
        super().__init__()
        self.alpha_p = nn.Parameter(torch.tensor(float(alpha_p)))
        self.alpha_n = nn.Parameter(torch.tensor(float(alpha_n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, "profile_ranges", False):
            with record_function("model.xielu"):
                return self._forward_impl(x)
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        positive = self.alpha_p * x.square() + 0.5 * x
        negative = self.alpha_n * torch.expm1(x) - self.alpha_n * x + 0.5 * x
        return torch.where(x > 0, positive, negative)


class MLP(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, bias: bool, alpha_p: float, alpha_n: float) -> None:
        super().__init__()
        self.fc_in = nn.Linear(d_model, hidden_dim, bias=bias)
        self.activation = XIELU(alpha_p, alpha_n)
        self.fc_out = nn.Linear(hidden_dim, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, "profile_ranges", False):
            with record_function("model.mlp"):
                return self._forward_impl(x)
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_out(self.activation(self.fc_in(x)))
