from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.hf_mix import iter_parquet_rows, load_mix_config, row_matches_filters, row_text, source_slug
from scripts.train_tokenizer import clean_training_text, training_text_segments
from src.data.tokenizer import SPECIAL_TOKEN_PIECES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a trained byte-level BPE vocabulary against its sample corpus")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--mix-config", default="configs/hf_mix_1b.yaml")
    parser.add_argument("--parquet-root", required=True)
    parser.add_argument("--max-documents-per-source", type=int, default=10_000)
    parser.add_argument("--max-sentence-length", type=int, default=16_384)
    parser.add_argument("--max-token-length", type=int, default=32)
    parser.add_argument("--rare-frequency", type=int, default=5)
    parser.add_argument("--json-output", default=None)
    return parser.parse_args()


def _byte_decoder() -> dict[str, int]:
    """Return the inverse of the GPT-2/Hugging Face ByteLevel mapping."""

    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))
    unicode_values = byte_values[:]
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            unicode_values.append(256 + extra)
            extra += 1
    return {chr(unicode_value): byte_value for byte_value, unicode_value in zip(byte_values, unicode_values)}


def _is_cjk(char: str) -> bool:
    value = ord(char)
    return (
        0x3400 <= value <= 0x4DBF
        or 0x4E00 <= value <= 0x9FFF
        or 0xF900 <= value <= 0xFAFF
        or 0x20000 <= value <= 0x2FA1F
    )


def _iter_training_segments(
    parquet_root: Path,
    sources: Iterable[Any],
    max_documents_per_source: int,
    max_sentence_length: int,
) -> Iterator[tuple[str, bool]]:
    for source in sources:
        accepted = 0
        source_dir = parquet_root / source_slug(source.name)
        paths = sorted(source_dir.glob("*.parquet"))
        if not paths:
            raise FileNotFoundError(f"no parquet shards found for {source.name!r} under {source_dir}")
        for _, row in iter_parquet_rows(paths):
            if not row_matches_filters(row, source.filters):
                continue
            text = row_text(row, source.text_field)
            if text is None:
                continue
            accepted += 1
            cleaned = clean_training_text(text)
            if cleaned:
                for segment_index, segment in enumerate(
                    training_text_segments(cleaned, max_sentence_length)
                ):
                    yield segment, segment_index == 0
            if accepted >= max_documents_per_source:
                break


def _raw_bytes(token: str, decoder: dict[str, int]) -> bytes:
    try:
        return bytes(decoder[char] for char in token)
    except KeyError as exc:
        raise ValueError(f"token contains a non-ByteLevel symbol: {token!r}") from exc


def _as_record(index: int, token: str, frequency: int, decoder: dict[str, int], byte_tokens: set[str]) -> dict[str, Any]:
    raw = _raw_bytes(token, decoder)
    try:
        decoded = raw.decode("utf-8")
        valid_utf8 = True
    except UnicodeDecodeError:
        decoded = raw.decode("utf-8", errors="replace")
        valid_utf8 = False
    return {
        "id": index,
        "token": token,
        "raw_utf8": raw.decode("utf-8", errors="replace"),
        "decoded": decoded,
        "byte_length": len(raw),
        "frequency": frequency,
        "is_byte_alphabet": token in byte_tokens,
        "valid_utf8": valid_utf8,
    }


def audit_tokenizer(
    tokenizer_path: Path,
    parquet_root: Path,
    mix_config: Path,
    max_documents_per_source: int,
    max_sentence_length: int,
    max_token_length: int,
    rare_frequency: int,
) -> dict[str, Any]:
    if (
        max_documents_per_source <= 0
        or max_sentence_length <= 0
        or max_token_length <= 0
        or rare_frequency < 0
    ):
        raise ValueError("document, sentence, and frequency limits must be valid")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab = sorted(((int(index), token) for token, index in tokenizer.get_vocab().items()), key=lambda item: item[0])
    decoder = _byte_decoder()
    special_tokens = set(SPECIAL_TOKEN_PIECES.values())
    byte_tokens = {token for _, token in vocab if len(token) == 1 and token in decoder}
    frequencies: Counter[int] = Counter()
    document_count = 0
    training_line_count = 0
    character_count = 0
    unknown_count = 0
    unknown_id = tokenizer.token_to_id(SPECIAL_TOKEN_PIECES["unk"])
    if unknown_id is None:
        raise ValueError("tokenizer is missing the configured unknown token")
    batch: list[str] = []
    _, sources = load_mix_config(mix_config)
    for segment, is_new_document in _iter_training_segments(
        parquet_root,
        sources,
        max_documents_per_source,
        max_sentence_length,
    ):
        batch.append(segment)
        if is_new_document:
            document_count += 1
        training_line_count += 1
        character_count += len(segment)
        if len(batch) < 256:
            continue
        for encoding in tokenizer.encode_batch(batch):
            frequencies.update(encoding.ids)
            unknown_count += encoding.ids.count(unknown_id)
        batch.clear()
    if batch:
        for encoding in tokenizer.encode_batch(batch):
            frequencies.update(encoding.ids)
            unknown_count += encoding.ids.count(unknown_id)

    records = [
        _as_record(index, token, frequencies[index], decoder, byte_tokens)
        for index, token in vocab
        if token not in special_tokens
    ]
    learned = [record for record in records if not record["is_byte_alphabet"]]
    base64_pattern = re.compile(r"^[ \t]*[A-Za-z0-9+/]{16,}={0,2}$")
    hex_pattern = re.compile(r"^[ \t]*[0-9A-Fa-f]{16,}$")
    word_pattern = re.compile(r"^[ \t]*[A-Za-z][A-Za-z-]{3,}$")

    def raw(record: dict[str, Any]) -> bytes:
        return _raw_bytes(record["token"], decoder)

    base64_like = [
        record
        for record in learned
        if base64_pattern.fullmatch(raw(record).decode("ascii", errors="ignore"))
        and any(char.isdigit() for char in record["decoded"])
    ]
    hex_like = [
        record
        for record in learned
        if hex_pattern.fullmatch(raw(record).decode("ascii", errors="ignore"))
    ]
    cjk_merges = [record for record in learned if any(_is_cjk(char) for char in record["decoded"])]
    control_merges = [
        record
        for record in learned
        if any(byte < 32 and byte not in {9, 10, 13, 32} for byte in raw(record))
    ]
    long_tokens = [record for record in learned if record["byte_length"] > max_token_length]
    rare_words = [
        record
        for record in learned
        if record["frequency"] <= rare_frequency and word_pattern.fullmatch(record["decoded"])
    ]
    low_frequency = [record for record in learned if record["frequency"] <= rare_frequency]

    report = {
        "tokenizer": str(tokenizer_path.resolve()),
        "vocab_size": len(vocab),
        "special_token_count": sum(token in special_tokens for _, token in vocab),
        "byte_alphabet_count": len(byte_tokens),
        "learned_merge_count": len(learned),
        "sample": {
            "document_count": document_count,
            "training_line_count": training_line_count,
            "character_count": character_count,
            "token_count": sum(frequencies.values()),
            "unknown_token_count": unknown_count,
            "unknown_rate": unknown_count / max(1, sum(frequencies.values())),
            "characters_per_token": character_count / max(1, sum(frequencies.values())),
        },
        "vocabulary": {
            "base64_like": base64_like,
            "hex_like": hex_like,
            "cjk_merges": cjk_merges,
            "control_merges": control_merges,
        "long_tokens_over_limit": long_tokens,
            "rare_word_merges": rare_words,
            "low_frequency_merges": low_frequency,
        },
        "checks": {
            "byte_alphabet_is_complete": len(byte_tokens) == 256,
            "no_base64_like_learned_merges": not base64_like,
            "no_hex_like_learned_merges": not hex_like,
            "no_cjk_learned_merges": not cjk_merges,
            "no_control_byte_learned_merges": not control_merges,
            "no_overlong_learned_merges": not long_tokens,
            "no_unknown_tokens_on_sample": unknown_count == 0,
        },
    }
    report["passed"] = all(report["checks"].values())
    return report


def main() -> None:
    args = parse_args()
    report = audit_tokenizer(
        Path(args.tokenizer),
        Path(args.parquet_root).resolve(),
        Path(args.mix_config),
        args.max_documents_per_source,
        args.max_sentence_length,
        args.max_token_length,
        args.rare_frequency,
    )
    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
