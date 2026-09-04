from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd.profiler import record_function


def _inverse_softplus(value: float) -> float:
    if value <= 0 or not math.isfinite(value):
        raise ValueError("inverse softplus requires a finite positive value")
    return value + math.log(-math.expm1(-value))


class XIELU(nn.Module):
    """Standard constrained trainable Expanded Integral of ELU activation.

    The scalar parameters are deliberately per block, matching the frozen
    architecture count. The stored parameters are unconstrained logits; the
    effective coefficients are ``softplus(alpha_p) > 0`` and
    ``beta + softplus(alpha_n) > beta``. This preserves xIELU's intended
    positive-side slope and negative-side gradient behavior during training.

        x > 0:  a_p * x^2 + beta*x
        x <= 0: a_n*(exp(min(x, eps))-1) - a_n*x + beta*x
    """

    def __init__(
        self,
        alpha_p: float = 0.5,
        alpha_n: float = 1.0,
        beta: float = 0.5,
        eps: float = -1.0e-6,
    ) -> None:
        super().__init__()
        if beta <= 0 or not math.isfinite(beta):
            raise ValueError("xIELU beta must be finite and positive")
        if alpha_n <= beta:
            raise ValueError("xIELU alpha_n must be greater than beta")
        if eps > 0 or not math.isfinite(eps):
            raise ValueError("xIELU eps must be finite and non-positive")
        self.alpha_p = nn.Parameter(torch.tensor(_inverse_softplus(float(alpha_p))))
        self.alpha_n = nn.Parameter(torch.tensor(_inverse_softplus(float(alpha_n - beta))))
        self.register_buffer("beta", torch.tensor(float(beta)), persistent=False)
        self.register_buffer("eps", torch.tensor(float(eps)), persistent=False)

    @property
    def effective_alpha_p(self) -> torch.Tensor:
        return F.softplus(self.alpha_p)

    @property
    def effective_alpha_n(self) -> torch.Tensor:
        return self.beta + F.softplus(self.alpha_n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, "profile_ranges", False):
            with record_function("model.xielu"):
                return self._forward_impl(x)
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        alpha_p = self.effective_alpha_p.to(dtype=x.dtype)
        alpha_n = self.effective_alpha_n.to(dtype=x.dtype)
        beta = self.beta.to(dtype=x.dtype)
        eps = self.eps.to(dtype=x.dtype)
        positive = alpha_p * x.square() + beta * x
        negative = alpha_n * torch.expm1(torch.minimum(x, eps)) - alpha_n * x + beta * x
        return torch.where(x > 0, positive, negative)


class MLP(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        bias: bool,
        alpha_p: float,
        alpha_n: float,
        beta: float,
        eps: float,
    ) -> None:
        super().__init__()
        self.fc_in = nn.Linear(d_model, hidden_dim, bias=bias)
        self.activation = XIELU(alpha_p, alpha_n, beta, eps)
        self.fc_out = nn.Linear(hidden_dim, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, "profile_ranges", False):
            with record_function("model.mlp"):
                return self._forward_impl(x)
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_out(self.activation(self.fc_in(x)))
