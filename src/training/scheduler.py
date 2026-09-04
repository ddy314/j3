from __future__ import annotations

import math

import torch


class TokenCosineScheduler:
    """Warmup + cosine schedule indexed by consumed training tokens."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        peak_lr: float,
        min_lr: float,
        warmup_tokens: int,
        total_tokens: int,
    ) -> None:
        self.optimizer = optimizer
        self.peak_lr = float(peak_lr)
        self.min_lr = float(min_lr)
        self.warmup_tokens = max(0, int(warmup_tokens))
        self.total_tokens = max(1, int(total_tokens))
        self.last_tokens = 0
        self.last_lr = 0.0
        self.step(0)

    def _lr(self, tokens_seen: int) -> float:
        tokens_seen = max(0, int(tokens_seen))
        if self.warmup_tokens and tokens_seen < self.warmup_tokens:
            return self.peak_lr * tokens_seen / self.warmup_tokens
        progress = (tokens_seen - self.warmup_tokens) / max(1, self.total_tokens - self.warmup_tokens)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.peak_lr - self.min_lr) * cosine

    def step(self, tokens_seen: int) -> float:
        self.last_tokens = int(tokens_seen)
        self.last_lr = self._lr(tokens_seen)
        for group in self.optimizer.param_groups:
            group["lr"] = self.last_lr
        return self.last_lr

    def get_last_lr(self) -> list[float]:
        return [self.last_lr]

    def state_dict(self) -> dict[str, float | int]:
        return {
            "peak_lr": self.peak_lr,
            "min_lr": self.min_lr,
            "warmup_tokens": self.warmup_tokens,
            "total_tokens": self.total_tokens,
            "last_tokens": self.last_tokens,
            "last_lr": self.last_lr,
        }

    def load_state_dict(self, state: dict[str, float | int]) -> None:
        for key in ("peak_lr", "min_lr", "warmup_tokens", "total_tokens"):
            if key in state and getattr(self, key) != state[key]:
                raise ValueError(f"scheduler {key} differs from checkpoint")
        self.step(int(state.get("last_tokens", 0)))
