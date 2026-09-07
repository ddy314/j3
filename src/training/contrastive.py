from __future__ import annotations

from typing import Literal

import torch
from torch.nn import functional as F

from src.data.multiple_choice import MultipleChoiceBatch


NegativeMode = Literal["mean", "hardest"]


def continuation_scores(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    continuation_mask: torch.Tensor,
) -> torch.Tensor:
    """Return length-normalized log P(choice | prompt) for each candidate."""

    if input_ids.ndim != 2 or continuation_mask.ndim != 2:
        raise ValueError("input_ids and continuation_mask must be rank-2 tensors")
    if input_ids.shape[0] != continuation_mask.shape[0]:
        raise ValueError("candidate and mask batch dimensions differ")
    if input_ids.shape[1] != continuation_mask.shape[1] + 1:
        raise ValueError("continuation_mask must have sequence_length - 1 columns")
    output = model(input_ids)
    logits = output.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    if logits.shape[:2] != targets.shape:
        raise ValueError("model logits do not align with candidate targets")
    # Cross entropy computes -log softmax(target) without materializing a
    # separate full-FP32 vocabulary tensor. The mask selects only answer
    # continuation tokens; prompt and right-padding tokens are ignored.
    negative_log_probs = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    mask = continuation_mask.to(dtype=negative_log_probs.dtype)
    lengths = mask.sum(dim=1).clamp_min(1.0)
    return -(negative_log_probs * mask).sum(dim=1) / lengths


def pairwise_ranking_loss(
    scores: torch.Tensor,
    group_offsets: torch.Tensor,
    positive_indices: torch.Tensor,
    *,
    temperature: float,
    negative_mode: NegativeMode = "mean",
) -> torch.Tensor:
    """Compute mean or hardest-negative logistic ranking loss."""

    if temperature <= 0:
        raise ValueError("ranking temperature must be positive")
    if negative_mode not in {"mean", "hardest"}:
        raise ValueError(f"unsupported ranking negative mode: {negative_mode}")
    if group_offsets.ndim != 1 or positive_indices.ndim != 1:
        raise ValueError("group_offsets and positive_indices must be rank-1 tensors")
    if len(group_offsets) != len(positive_indices) + 1:
        raise ValueError("group_offsets must have one more element than positive_indices")
    losses: list[torch.Tensor] = []
    for group_index, positive in enumerate(positive_indices.tolist()):
        start = int(group_offsets[group_index])
        end = int(group_offsets[group_index + 1])
        if not start <= positive < end:
            raise ValueError("positive candidate is outside its multiple-choice group")
        negatives = torch.cat((scores[start:positive], scores[positive + 1 : end]))
        if negatives.numel() == 0:
            raise ValueError("each multiple-choice group needs at least one negative")
        margins = (scores[positive] - negatives) / temperature
        pair_losses = F.softplus(-margins)
        losses.append(pair_losses.mean() if negative_mode == "mean" else pair_losses.max())
    if not losses:
        raise ValueError("ranking loss received no groups")
    return torch.stack(losses).mean()


def ranking_diagnostics(
    scores: torch.Tensor,
    group_offsets: torch.Tensor,
    positive_indices: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return accuracy and margin tensors for logging/evaluation."""

    correct: list[torch.Tensor] = []
    margins: list[torch.Tensor] = []
    for group_index, positive in enumerate(positive_indices.tolist()):
        start = int(group_offsets[group_index])
        end = int(group_offsets[group_index + 1])
        negatives = torch.cat((scores[start:positive], scores[positive + 1 : end]))
        if negatives.numel() == 0:
            raise ValueError("each multiple-choice group needs at least one negative")
        positive_score = scores[positive]
        # Match the formal scorer's argmax tie behavior (first option wins).
        correct.append(torch.argmax(scores[start:end]).eq(positive - start))
        margins.append(positive_score - negatives.mean())
    return {
        "correct": torch.stack(correct).to(dtype=torch.float32).sum(),
        "examples": torch.tensor(float(len(correct)), device=scores.device),
        "mean_positive_margin": torch.stack(margins).mean(),
    }


def score_batch(model: torch.nn.Module, batch: MultipleChoiceBatch) -> torch.Tensor:
    return continuation_scores(model, batch.input_ids, batch.continuation_mask)


def batch_ranking_loss(
    model: torch.nn.Module,
    batch: MultipleChoiceBatch,
    *,
    temperature: float,
    negative_mode: NegativeMode,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    scores = score_batch(model, batch)
    loss = pairwise_ranking_loss(
        scores,
        batch.group_offsets,
        batch.positive_indices,
        temperature=temperature,
        negative_mode=negative_mode,
    )
    return loss, ranking_diagnostics(scores.detach(), batch.group_offsets, batch.positive_indices)
