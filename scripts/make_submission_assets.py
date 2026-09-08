"""Create small, reproducible PNG audit assets for the competition submission."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/submission/assets"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def make_training_curve() -> None:
    rows = [
        json.loads(line)
        for line in (ROOT / "runs/20260906-225321/metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    steps = [row["tokens_seen"] / 1_000_000 for row in rows]
    train = [row["smoothed_loss"] for row in rows]
    validation = [
        (row["tokens_seen"] / 1_000_000, row["validation_loss"])
        for row in rows
        if row.get("validation_loss") is not None
    ]
    figure, axis = plt.subplots(figsize=(10, 5.5), dpi=160)
    axis.plot(steps, train, color="#2563eb", linewidth=1.2, label="smoothed train loss")
    if validation:
        axis.plot(
            [point[0] for point in validation],
            [point[1] for point in validation],
            color="#dc2626",
            linewidth=1.4,
            label="validation loss",
        )
    axis.set_title("J3 Stage 3 training record")
    axis.set_xlabel("tokens seen (millions)")
    axis.set_ylabel("loss")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(OUT / "training-loss.png", bbox_inches="tight")
    plt.close(figure)


def make_data_mixture() -> None:
    manifest = _read_json(ROOT / "data/stage3_mixed_tokenized/manifest.json")
    source_tokens = manifest["mixing"]["train_source_chunk_counts_before_shuffle"]
    chunk_tokens = manifest["mixing"]["chunk_tokens"]
    names = list(source_tokens)
    values = [source_tokens[name] * chunk_tokens / 1_000_000 for name in names]
    labels = [
        "FinePDF",
        "FineWeb-Edu",
        "Ultra-FineWeb normal",
        "Cosmopedia stories",
        "Cosmopedia WikiHow",
        "WikiText-103",
    ]
    # Keep the public figure in the same order as the source disclosure table.
    order = ["finepdfs_300m", "fineweb_edu", "stage2_natural_web", "stage2_narrative", "stage2_procedural", "wikitext_103"]
    values_by_name = dict(zip(names, values))
    values = [values_by_name[name] for name in order]
    figure, axis = plt.subplots(figsize=(10, 5.5), dpi=160)
    bars = axis.barh(labels, values, color=["#0f766e", "#2563eb", "#64748b", "#7c3aed", "#ea580c", "#16a34a"])
    axis.invert_yaxis()
    axis.set_title("J3 Stage 3 materialized training mixture")
    axis.set_xlabel("training tokens (millions)")
    axis.grid(axis="x", alpha=0.22)
    for bar, value in zip(bars, values):
        axis.text(value + 4, bar.get_y() + bar.get_height() / 2, f"{value:.1f}M", va="center")
    figure.tight_layout()
    figure.savefig(OUT / "data-mixture.png", bbox_inches="tight")
    plt.close(figure)


def make_evaluation_summary() -> None:
    results = _read_json(ROOT / "docs/evaluation/results/j3-gibc.json")["results"]["results"]
    labels = ["HellaSwag", "ARC-Easy", "PIQA", "WinoGrande"]
    keys = ["hellaswag", "arc_easy", "piqa", "winogrande"]
    scores = [
        results[key].get("acc_norm,none", results[key].get("acc,none"))
        for key in keys
    ]
    figure, (accuracy_axis, perplexity_axis) = plt.subplots(1, 2, figsize=(10, 4.8), dpi=160)
    accuracy_axis.bar(labels, scores, color="#2563eb")
    accuracy_axis.set_ylim(0, 1)
    accuracy_axis.set_ylabel("zero-shot accuracy")
    accuracy_axis.set_title("GIBC multiple-choice")
    accuracy_axis.tick_params(axis="x", rotation=35)
    accuracy_axis.grid(axis="y", alpha=0.22)
    for index, score in enumerate(scores):
        accuracy_axis.text(index, score + 0.025, f"{score:.3f}", ha="center")

    wiki = results["j3_wikitext_103"]
    perplexity_axis.bar(
        ["word PPL", "byte PPL"],
        [wiki["word_perplexity,none"], wiki["byte_perplexity,none"]],
        color=["#dc2626", "#f97316"],
    )
    perplexity_axis.set_title("WikiText-103 held-out")
    perplexity_axis.set_ylabel("perplexity (lower is better)")
    perplexity_axis.grid(axis="y", alpha=0.22)
    figure.suptitle("J3 final evaluation summary", y=1.02)
    figure.tight_layout()
    figure.savefig(OUT / "evaluation-summary.png", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    make_training_curve()
    make_data_mixture()
    make_evaluation_summary()
    print(f"wrote submission assets under {OUT}")


if __name__ == "__main__":
    main()
