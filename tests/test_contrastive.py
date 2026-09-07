from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from scripts.prepare_mc_posttrain import prepare
from scripts.score_multiple_choice import score
from src.data.multiple_choice import (
    MultipleChoiceExample,
    MultipleChoiceStream,
    collate_multiple_choice,
)
from src.model import DecoderLM, ModelConfig
from src.training.config import DataConfig, TrainConfig
from src.training.checkpoint import CheckpointManager
from src.training.contrastive import pairwise_ranking_loss, ranking_diagnostics
from src.training.trainer import Trainer


def _mc_manifest(tmp_path: Path, *, count: int = 4) -> Path:
    records = [
        {
            "id": f"fixture-{index}",
            "category": "fixture",
            "source": "fixture",
            "prompt_ids": [4, 5],
            "choice_ids": [[6], [7]],
            "answer": index % 2,
        }
        for index in range(count)
    ]
    (tmp_path / "train.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    manifest = {
        "version": 1,
        "type": "multiple_choice_tokenized_v1",
        "tokenizer_hash": None,
        "vocab_size": 32,
        "pad_id": 0,
        "max_seq_len": 4,
        "train_file": "train.jsonl",
        "train_count": len(records),
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_collate_masks_only_choice_tokens() -> None:
    example = MultipleChoiceExample(
        example_id="x",
        prompt_ids=(4, 5),
        choice_ids=((6, 7), (8,)),
        answer=0,
    )
    batch = collate_multiple_choice([example], pad_id=0, pad_to_length=5)
    assert batch.input_ids.tolist() == [[4, 5, 6, 7, 0], [4, 5, 8, 0, 0]]
    assert batch.continuation_mask.tolist() == [
        [False, True, True, False],
        [False, True, False, False],
    ]
    assert batch.group_offsets.tolist() == [0, 2]
    assert batch.positive_indices.tolist() == [0]

    truncated = collate_multiple_choice(
        [
            MultipleChoiceExample(
                example_id="truncated",
                prompt_ids=(1, 2, 3, 4),
                choice_ids=((5, 6), (7,)),
                answer=0,
            )
        ],
        pad_id=0,
        pad_to_length=5,
    )
    assert truncated.input_ids.tolist() == [[2, 3, 4, 5, 6], [2, 3, 4, 7, 0]]


def test_pairwise_ranking_loss_and_diagnostics() -> None:
    scores = torch.tensor([0.9, 0.1, 0.2, 0.3, 0.2], requires_grad=True)
    offsets = torch.tensor([0, 3, 5])
    positives = torch.tensor([0, 3])
    loss = pairwise_ranking_loss(
        scores,
        offsets,
        positives,
        temperature=0.1,
        negative_mode="mean",
    )
    loss.backward()
    diagnostics = ranking_diagnostics(scores.detach(), offsets, positives)
    assert torch.isfinite(loss)
    assert diagnostics["correct"].item() == 2
    assert diagnostics["examples"].item() == 2
    assert diagnostics["mean_positive_margin"].item() > 0


def test_prepare_mc_artifact_and_stream(tmp_path: Path) -> None:
    raw = tmp_path / "records.jsonl"
    raw.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "train-a",
                        "prompt": "A short question",
                        "choices": [" answer one", " answer two"],
                        "answer": "A",
                        "split": "train",
                    }
                ),
                json.dumps(
                    {
                        "id": "val-a",
                        "prompt": "Another short question",
                        "choices": [" answer one", " answer two"],
                        "answer": 1,
                        "split": "validation",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "version: 1",
                "name: fixture-mc",
                "max_seq_len: 32",
                "val_ratio: 0",
                "sources:",
                "  - name: fixture",
                "    category: fixture",
                f"    path: {raw}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "tokenized"
    repository = Path(__file__).resolve().parents[1]
    manifest = prepare(
        config,
        tokenizer_path=repository / "data/tokenized/tokenizer.json",
        output_dir=output_dir,
        decontamination_index=None,
    )
    assert manifest["train_count"] == 1
    assert manifest["val_count"] == 1
    stream = MultipleChoiceStream(output_dir / "manifest.json", split="train", shuffle=False)
    assert stream.next_batch(1)[0].answer == 0
    val_stream = MultipleChoiceStream(output_dir / "manifest.json", split="val", shuffle=False)
    assert val_stream.next_batch(1)[0].answer == 1


def test_score_multiple_choice_uses_one_epoch_and_actual_groups(tmp_path: Path) -> None:
    mc_dir = tmp_path / "mc"
    mc_dir.mkdir()
    mc_manifest = _mc_manifest(mc_dir, count=4)
    model_config = _contrastive_model_config()
    config_path = tmp_path / "score.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": model_config.to_dict(),
                "training": {
                    "device": "cpu",
                    "precision": "fp32",
                    "sequence_length": 4,
                    "micro_batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "total_tokens": 4,
                    "max_tokens": 4,
                    "mc_sequence_length": 4,
                },
                "data": {"mc_train_manifest": str(mc_manifest)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    model = DecoderLM(model_config)
    manager = CheckpointManager(run_dir)
    checkpoint = manager.save(
        "step_000000001",
        {"model": model.state_dict()},
        {"model_config": model_config.to_dict()},
    )
    result = score(
        config_path=config_path,
        checkpoint=checkpoint,
        manifest_path=mc_manifest,
        split="train",
        device_spec="cpu",
        batch_size=2,
        max_batches=None,
    )
    assert result["examples"] == 4
    assert sum(result["option_count_histogram"].values()) == 4


def _contrastive_config(max_tokens: int) -> TrainConfig:
    return TrainConfig(
        seed=77,
        device="cpu",
        precision="fp32",
        sequence_length=4,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        max_tokens=max_tokens,
        total_tokens=64,
        learning_rate=1e-3,
        min_learning_rate=1e-4,
        warmup_tokens=4,
        log_every_steps=100,
        eval_every_tokens=0,
        checkpoint_every_steps=0,
        checkpoint_every_minutes=0,
        permanent_checkpoint_every_tokens=0,
        keep_last_n=3,
        pin_memory=False,
        objective="contrastive_mc",
        lm_loss_weight=1.0,
        ranking_loss_weight=0.5,
        ranking_temperature=0.1,
        ranking_negative_mode="mean",
        mc_sequence_length=4,
    )


def _contrastive_model_config() -> ModelConfig:
    return ModelConfig(
        model_name="contrastive-test",
        vocab_size=32,
        d_model=16,
        depth=1,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=8,
        mlp_hidden_dim=24,
        output_residual_rank=2,
        max_seq_len=8,
    )


def test_contrastive_exact_resume_restores_both_streams(tmp_path: Path) -> None:
    mc_dir = tmp_path / "mc"
    mc_dir.mkdir()
    mc_manifest = _mc_manifest(mc_dir)
    data_config = DataConfig(
        eval_batches=0,
        synthetic=True,
        mc_train_manifest=str(mc_manifest),
        mc_batch_size=1,
        mc_eval_batches=0,
    )
    model_config = _contrastive_model_config()
    first_tokens = 2 * 4
    final_tokens = 5 * 4

    torch.manual_seed(123)
    first = Trainer(
        DecoderLM(model_config),
        model_config,
        _contrastive_config(first_tokens),
        data_config,
        tmp_path / "resumed",
    )
    first.run()
    checkpoint = first.checkpoints.latest()
    assert checkpoint is not None

    torch.manual_seed(999)
    resumed = Trainer(
        DecoderLM(model_config),
        model_config,
        _contrastive_config(final_tokens),
        data_config,
        tmp_path / "resumed",
    )
    resumed.load_checkpoint(checkpoint)
    resumed.run()

    torch.manual_seed(123)
    uninterrupted = Trainer(
        DecoderLM(model_config),
        model_config,
        _contrastive_config(final_tokens),
        data_config,
        tmp_path / "uninterrupted",
    )
    uninterrupted.run()

    for left, right in zip(resumed.raw_model.parameters(), uninterrupted.raw_model.parameters()):
        torch.testing.assert_close(left, right, atol=0.0, rtol=0.0)
    assert resumed.state.tokens_seen == uninterrupted.state.tokens_seen == final_tokens
    assert resumed.mc_stream is not None and uninterrupted.mc_stream is not None
    assert resumed.mc_stream.state_dict() == uninterrupted.mc_stream.state_dict()
