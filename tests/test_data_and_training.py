from __future__ import annotations

import json
import random
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from src.data.mmap_dataset import MMapTokenStream, SyntheticTokenStream
from src.model import DecoderLM, ModelConfig
from src.data.tokenizer import load_tokenizer
from src.training.checkpoint import CheckpointManager, capture_rng_state, restore_rng_state
from src.training.config import DataConfig, TrainConfig, load_config
from src.training.metrics import MetricsWriter
from src.training.scheduler import TokenCosineScheduler
from src.training.trainer import Trainer
from scripts.lr_range_test import build_proxy_config


def _tiny_train_config(max_tokens: int) -> TrainConfig:
    return TrainConfig(
        seed=77,
        device="cpu",
        precision="fp32",
        sequence_length=8,
        micro_batch_size=2,
        gradient_accumulation_steps=1,
        max_tokens=max_tokens,
        total_tokens=512,
        learning_rate=1e-3,
        min_learning_rate=1e-4,
        warmup_tokens=16,
        log_every_steps=100,
        eval_every_tokens=0,
        checkpoint_every_steps=0,
        checkpoint_every_minutes=0,
        permanent_checkpoint_every_tokens=0,
        keep_last_n=3,
        pin_memory=False,
    )


def _tiny_model_config() -> ModelConfig:
    return ModelConfig(
        model_name="resume-test",
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


def test_mmap_stream_cross_shard_and_exact_state(tmp_path: Path) -> None:
    first = np.arange(10, dtype=np.uint16)
    second = np.arange(10, 24, dtype=np.uint16)
    first.tofile(tmp_path / "shard_0.bin")
    second.tofile(tmp_path / "shard_1.bin")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "dtype": "uint16",
                "train_shards": [
                    {"path": "shard_0.bin", "token_count": len(first)},
                    {"path": "shard_1.bin", "token_count": len(second)},
                ],
            }
        )
    )
    stream = MMapTokenStream(manifest, shuffle_shards=False)
    inputs, targets = stream.next_batch(1, 7)
    np.testing.assert_array_equal(inputs[0], np.arange(7, dtype=np.uint16))
    np.testing.assert_array_equal(targets[0], np.arange(1, 8, dtype=np.uint16))
    state = stream.state_dict()
    expected = stream.next_batch(1, 7)
    np.testing.assert_array_equal(expected[0], np.arange(7, 14, dtype=np.uint16)[None, :])
    np.testing.assert_array_equal(expected[1], np.arange(8, 15, dtype=np.uint16)[None, :])
    restored = MMapTokenStream(manifest, shuffle_shards=False)
    restored.load_state_dict(state)
    actual = restored.next_batch(1, 7)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])

    wrapped = MMapTokenStream(manifest, shuffle_shards=False)
    wrapped_inputs, wrapped_targets = wrapped.next_batch(1, 24)
    np.testing.assert_array_equal(wrapped_inputs[0], np.arange(24, dtype=np.uint16))
    np.testing.assert_array_equal(wrapped_targets[0], np.concatenate((np.arange(1, 24), np.array([0]))))
    next_inputs, _ = wrapped.next_batch(1, 1)
    np.testing.assert_array_equal(next_inputs, np.array([[0]], dtype=np.uint16))


def test_offline_tokenizer_and_prepare_data_paths(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "alpha beta gamma delta epsilon\n"
        "a second document with enough repeated text for a tokenizer\n"
        "第三个文档用于验证离线分片路径。\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "tokenized"
    repository = Path(__file__).resolve().parents[1]
    tokenizer_result = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts/train_tokenizer.py"),
            "--input",
            str(corpus),
            "--output-dir",
            str(output_dir),
            "--vocab-size",
            "64",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert tokenizer_result.returncode == 0, tokenizer_result.stderr
    data_result = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts/prepare_data.py"),
            "--input",
            str(corpus),
            "--tokenizer",
            str(output_dir / "tokenizer.model"),
            "--output-dir",
            str(output_dir),
            "--shard-tokens",
            "5",
            "--val-ratio",
            "0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert data_result.returncode == 0, data_result.stderr
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["dataset"]["input"] == [str(corpus.resolve())]
    assert 0.0 <= manifest["statistics"]["unk_rate"] <= 1.0
    assert manifest["train_shards"][0]["path"] == "shard_000000.bin"
    stream = MMapTokenStream(output_dir / "manifest.json", shuffle_shards=False)
    inputs, targets = stream.next_batch(1, 4)
    assert inputs.shape == targets.shape == (1, 4)
    assert int(inputs.max()) < 64


def test_byte_level_bpe_handles_unicode_and_literal_unk(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        ("English text with repeated words and a small byte-level tokenizer. 中文🙂 <unk>\n" * 20),
        encoding="utf-8",
    )
    output_dir = tmp_path / "byte-tokenizer"
    repository = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts/train_tokenizer.py"),
            "--input",
            str(corpus),
            "--output-dir",
            str(output_dir),
            "--tokenizer-type",
            "byte-bpe",
            "--vocab-size",
            "512",
            "--min-token-frequency",
            "1",
            "--max-token-length",
            "32",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    tokenizer = load_tokenizer(output_dir / "tokenizer.json")
    ids = tokenizer.encode("中文🙂 <unk>")
    assert tokenizer.unk_id not in ids
    assert max(ids) < tokenizer.vocab_size
    metadata = json.loads((output_dir / "tokenizer_meta.json").read_text())
    assert metadata["tokenizer_type"] == "byte-bpe"


def test_scheduler_state_resume() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    scheduler = TokenCosineScheduler(optimizer, 1e-3, 1e-4, 10, 100)
    scheduler.step(45)
    state = scheduler.state_dict()
    other = TokenCosineScheduler(torch.optim.AdamW([torch.nn.Parameter(torch.ones(()))], lr=1e-3), 1e-3, 1e-4, 10, 100)
    other.load_state_dict(state)
    assert other.last_tokens == 45
    assert other.last_lr == scheduler.last_lr


def test_lr_proxy_preserves_formal_batch_and_schedule() -> None:
    formal = TrainConfig(
        device="cpu",
        sequence_length=1024,
        micro_batch_size=8,
        gradient_accumulation_steps=16,
        global_batch_tokens=131072,
        total_tokens=1_100_000_000,
        warmup_tokens=20_000_000,
    )
    proxy = build_proxy_config(formal, proxy_tokens=20_000_000, learning_rate=5e-4)
    assert proxy.effective_global_batch_tokens == formal.effective_global_batch_tokens == 131072
    assert proxy.total_tokens == formal.total_tokens
    assert proxy.warmup_tokens == formal.warmup_tokens
    assert proxy.max_tokens == 20_000_000
    assert proxy.learning_rate == 5e-4


def test_formal_pretrain_uses_first_stage_sequence_length() -> None:
    repository = Path(__file__).resolve().parents[1]
    _, training, _, _ = load_config(repository / "configs/pretrain_1b.yaml")
    assert training.sequence_length == 512
    assert training.micro_batch_size == 16
    assert training.gradient_accumulation_steps == 16
    assert training.effective_global_batch_tokens == 131072


def test_checkpoint_manager_atomic_roundtrip(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path / "run", keep_last_n=2)
    path = manager.save("step_000000001", {"value": torch.arange(4)}, {"step": 1, "created_at": "now"})
    assert path.is_dir()
    payload, loaded_path = manager.load("auto")
    assert loaded_path == path
    torch.testing.assert_close(payload["value"], torch.arange(4))
    assert manager.latest() == path


def test_checkpoint_restores_model_outputs(tmp_path: Path) -> None:
    config = _tiny_model_config()
    torch.manual_seed(12)
    model = DecoderLM(config).eval()
    inputs = torch.randint(config.vocab_size, (2, config.max_seq_len))
    with torch.no_grad():
        expected = model(inputs).logits

    manager = CheckpointManager(tmp_path / "run")
    path = manager.save(
        "step_000000001",
        {"model": model.state_dict()},
        {"step": 1, "created_at": "now", "model_config": config.to_dict()},
    )
    restored = DecoderLM(config).eval()
    payload, loaded_path = manager.load(path)
    restored.load_state_dict(payload["model"])
    with torch.no_grad():
        actual = restored(inputs).logits
    assert loaded_path == path
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_rng_state_roundtrip() -> None:
    random.seed(2)
    np.random.seed(2)
    torch.manual_seed(2)
    state = capture_rng_state()
    expected = (random.random(), float(np.random.rand()), float(torch.rand(())))
    restore_rng_state(state)
    actual = (random.random(), float(np.random.rand()), float(torch.rand(())))
    assert expected == actual


def test_exact_resume_matches_uninterrupted(tmp_path: Path) -> None:
    model_config = _tiny_model_config()
    data_config = DataConfig(eval_batches=0, synthetic=True)
    tokens_per_step = 16
    first_tokens = 4 * tokens_per_step
    final_tokens = 9 * tokens_per_step

    torch.manual_seed(123)
    first = Trainer(
        DecoderLM(model_config),
        model_config,
        _tiny_train_config(first_tokens),
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
        _tiny_train_config(final_tokens),
        data_config,
        tmp_path / "resumed",
    )
    resumed.load_checkpoint(checkpoint)
    resumed.run()

    torch.manual_seed(123)
    uninterrupted = Trainer(
        DecoderLM(model_config),
        model_config,
        _tiny_train_config(final_tokens),
        data_config,
        tmp_path / "uninterrupted",
    )
    uninterrupted.run()

    for left, right in zip(resumed.raw_model.parameters(), uninterrupted.raw_model.parameters()):
        torch.testing.assert_close(left, right, atol=0.0, rtol=0.0)
    assert resumed.state.tokens_seen == uninterrupted.state.tokens_seen == final_tokens
    assert resumed.state.step == uninterrupted.state.step == 9


def test_exact_mmap_resume_matches_uninterrupted(tmp_path: Path) -> None:
    values = np.arange(256, dtype=np.uint16) % 32
    values[:96].tofile(tmp_path / "train_0.bin")
    values[96:].tofile(tmp_path / "train_1.bin")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "dtype": "uint16",
                "train_shards": [
                    {"path": "train_0.bin", "token_count": 96},
                    {"path": "train_1.bin", "token_count": 160},
                ],
            }
        )
    )
    model_config = _tiny_model_config()
    data_config = DataConfig(
        train_manifest=str(manifest),
        eval_batches=0,
        synthetic=False,
        shuffle_shards=False,
    )
    first_tokens = 4 * 16
    final_tokens = 9 * 16

    torch.manual_seed(321)
    first = Trainer(
        DecoderLM(model_config),
        model_config,
        _tiny_train_config(first_tokens),
        data_config,
        tmp_path / "mmap-resumed",
    )
    first.run()
    checkpoint = first.checkpoints.latest()
    assert checkpoint is not None

    torch.manual_seed(999)
    resumed = Trainer(
        DecoderLM(model_config),
        model_config,
        _tiny_train_config(final_tokens),
        data_config,
        tmp_path / "mmap-resumed",
    )
    resumed.load_checkpoint(checkpoint)
    resumed.run()

    torch.manual_seed(321)
    uninterrupted = Trainer(
        DecoderLM(model_config),
        model_config,
        _tiny_train_config(final_tokens),
        data_config,
        tmp_path / "mmap-uninterrupted",
    )
    uninterrupted.run()

    for left, right in zip(resumed.raw_model.parameters(), uninterrupted.raw_model.parameters()):
        torch.testing.assert_close(left, right, atol=0.0, rtol=0.0)
    assert resumed.train_stream.state_dict() == uninterrupted.train_stream.state_dict()


def test_emergency_checkpoint_on_requested_stop(tmp_path: Path) -> None:
    model_config = _tiny_model_config()
    trainer = Trainer(
        DecoderLM(model_config),
        model_config,
        _tiny_train_config(3 * 16),
        DataConfig(eval_batches=0, synthetic=True),
        tmp_path / "stopped",
    )
    trainer.request_stop("SIGINT")
    state = trainer.run()
    assert state.status == "stopped"
    checkpoints = list((tmp_path / "stopped" / "checkpoints").glob("step_*/metadata.json"))
    assert len(checkpoints) == 1
    metadata = json.loads(checkpoints[0].read_text())
    assert metadata["reason"] == "emergency"


def test_metrics_writer_appends_jsonl(tmp_path: Path) -> None:
    with MetricsWriter(tmp_path) as writer:
        writer.write({"step": 3, "tokens_seen": 48, "tokens_per_sec": 123.0})
    lines = (tmp_path / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["step"] == 3
