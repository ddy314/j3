from __future__ import annotations

from pathlib import Path

import yaml

from scripts.prepare_stage3 import load_stage3_config


def test_stage3_mix_has_requested_categories_and_exact_target() -> None:
    repository = Path(__file__).resolve().parents[1]
    config = load_stage3_config(repository / "configs/stage3_1b.yaml")
    assert config["target_train_tokens"] == 1_000_000_000
    sources = config["sources"]
    assert sum(
        int(item.get("token_budget", item.get("target_train_tokens", 0)))
        for item in sources
    ) == 1_000_206_468
    assert {item["category"] for item in sources} == {
        "finepdf",
        "stage1_clean_web_edu",
        "stage2_natural_web",
        "stage2_narrative",
        "stage2_procedural",
        "wikipedia_encyclopedic",
    }
    assert [item["kind"] for item in sources].count("hf") == 2
    assert [item["kind"] for item in sources].count("existing_source") == 3
    assert [item["kind"] for item in sources].count("tokenized_manifest") == 1


def test_stage3_stage2_slices_sum_to_250m() -> None:
    repository = Path(__file__).resolve().parents[1]
    config = load_stage3_config(repository / "configs/stage3_1b.yaml")
    stage2 = [item for item in config["sources"] if item["kind"] == "existing_source"]
    assert sum(int(item["token_budget"]) for item in stage2) == 250_000_000


def test_stage3_training_uses_global_mix_and_lower_transfer_lr() -> None:
    repository = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((repository / "configs/pretrain_stage3_1b.yaml").read_text())
    assert config["data"]["train_manifest"] == "data/stage3_mixed_tokenized/manifest.json"
    assert config["data"]["val_manifest"] == "data/stage3_mixed_tokenized/manifest.json"
    assert config["training"]["learning_rate"] == 1.0e-4
    assert config["training"]["min_learning_rate"] == 1.0e-5
