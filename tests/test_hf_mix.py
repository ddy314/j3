from __future__ import annotations

from pathlib import Path

from src.data.hf_mix import HFMixSource, load_mix_config, row_matches_filters, source_slug
from scripts import prepare_hf_mix
from scripts.prepare_hf_mix import take_exact


def test_requested_hf_mix_config_has_exact_budgets() -> None:
    repository = Path(__file__).resolve().parents[1]
    _, sources = load_mix_config(repository / "configs/hf_mix_1b.yaml")
    assert [source.name for source in sources] == [
        "dclm_edu_3plus",
        "fineweb_edu",
        "cosmopedia_v2",
        "wikitext_103",
    ]
    assert [source.token_budget for source in sources] == [
        600_000_000,
        300_000_000,
        100_000_000,
        100_000_000,
    ]
    assert sum(source.token_budget for source in sources) == 1_100_000_000


def test_stage2_mix_has_exact_balanced_budgets() -> None:
    repository = Path(__file__).resolve().parents[1]
    _, sources = load_mix_config(repository / "configs/hf_mix_stage2_1b.yaml")
    assert sum(source.token_budget for source in sources) == 1_000_000_000
    l1 = [source for source in sources if source.name.startswith("ultra_fineweb_l1_")]
    assert len(l1) == 6
    assert sum(source.token_budget for source in l1) == 300_000_000
    l2 = [source for source in sources if source.name.startswith("ultra_fineweb_l2_")]
    assert sum(source.token_budget for source in l2) == 250_000_000
    assert l2[0].filters == {"score_min": 0.55, "score_max": 0.8}
    assert l2[-1].token_budget == 37_500_000


def test_hf_mix_filters_use_english_and_dclm_score() -> None:
    repository = Path(__file__).resolve().parents[1]
    _, sources = load_mix_config(repository / "configs/hf_mix_1b.yaml")
    dclm = sources[0]
    assert row_matches_filters({"language": "en", "edu_int_score": 3}, dclm.filters)
    assert not row_matches_filters({"language": "en", "edu_int_score": 2}, dclm.filters)
    assert not row_matches_filters({"language": "fr", "edu_int_score": 5}, dclm.filters)
    assert row_matches_filters({"metadata": {"language": "en"}}, {"language": "en"})
    assert row_matches_filters({"score": 0.7, "format": "story"}, {"score_min": 0.55, "score_max": 0.8, "format_in": ["story"]})
    assert not row_matches_filters({"score": 0.95}, {"score_min": 0.55, "score_max": 0.8})
    assert row_matches_filters({"meta": '{"language": "en", "language_score": 0.9}'}, {"language": "en", "language_score_min": 0.8})


def test_take_exact_keeps_an_eos_boundary() -> None:
    assert take_exact([4, 5, 6, 3], 4, 3) == [4, 5, 6, 3]
    assert take_exact([4, 5, 6, 3], 3, 3) == [4, 5, 3]
    assert take_exact([4, 5, 6, 3], 1, 3) == [3]
    assert take_exact([4, 5], 0, 3) == []


def test_source_slug_is_stable_and_path_safe() -> None:
    assert source_slug("Source A / synthetic") == "source-a-synthetic"


def test_prepare_source_writes_exact_token_quota(monkeypatch, tmp_path: Path) -> None:
    class FakeTokenizer:
        vocab_size = 16
        unk_id = 1
        eos_id = 3

        def encode(self, text: str, add_bos: bool = False, add_eos: bool = True) -> list[int]:
            del add_bos, add_eos
            return [4 + (len(text) % 3), 3]

    source = HFMixSource(
        name="fake_source",
        dataset="fake/dataset",
        config=None,
        split="train",
        text_field="text",
        token_budget=5,
        filters={"language": "en"},
    )
    rows = iter(
        [
            (1, {"text": "a", "language": "en"}),
            (2, {"text": "bb", "language": "en"}),
            (3, {"text": "ccc", "language": "fr"}),
            (4, {"text": "dddd", "language": "en"}),
        ]
    )
    monkeypatch.setattr(prepare_hf_mix, "iter_hf_rows", lambda *_args, **_kwargs: rows)
    result = prepare_hf_mix._prepare_source(
        source,
        tokenizer=FakeTokenizer(),
        output_dir=tmp_path,
        scale=1.0,
        val_ratio=0.0,
        seed=1337,
        shard_tokens=3,
        deduplicate=False,
        max_retries=0,
        retry_backoff_seconds=0,
    )
    assert result["train_token_count"] == 5
    assert result["statistics"]["filtered_rows"] == 1
    assert sum(entry["token_count"] for entry in result["train_shards"]) == 5
    for entry in result["train_shards"]:
        assert (tmp_path / entry["path"]).stat().st_size == 2 * entry["token_count"]
