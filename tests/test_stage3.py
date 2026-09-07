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


def test_final_stage_recipe_is_exactly_500m_and_excludes_generic_web() -> None:
    repository = Path(__file__).resolve().parents[1]
    config = load_stage3_config(repository / "configs/stage3_final_500m.yaml")
    assert config["target_train_tokens"] == 500_000_000
    assert sum(
        int(item.get("token_budget", item.get("target_train_tokens", 0)))
        for item in config["sources"]
    ) == 500_000_000
    assert {item["category"] for item in config["sources"]} == {
        "textbook_longform_pdf",
        "synthetic_textbook",
        "synthetic_procedural_knowledge",
        "synthetic_coherent_narrative",
        "scientific_explanations",
        "encyclopedia",
        "qa_knowledge",
        "synthetic_verified_reasoning",
    }
    assert not any("web" in str(item["category"]).lower() for item in config["sources"])
    reasoning = next(item for item in config["sources"] if item["category"] == "synthetic_verified_reasoning")
    assert reasoning["filters"]["correct_in"] == [True]
    assert reasoning["token_budget"] == 5_000_000
    qa = next(item for item in config["sources"] if item["category"] == "qa_knowledge")
    assert qa["token_budget"] == 55_000_000


def test_final_stage_training_config_is_stage4_and_uses_final_manifest() -> None:
    repository = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (repository / "configs/pretrain_stage4_final_500m.yaml").read_text()
    )
    assert config["training"]["total_tokens"] == 500_000_000
    assert config["data"]["train_manifest"] == "data/stage3_final_mixed_tokenized/manifest.json"
    assert config["data"]["val_manifest"] == "data/stage3_final_mixed_tokenized/manifest.json"
