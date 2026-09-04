from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.hf_mix import (
    HFMixSource,
    iter_hf_rows,
    iter_parquet_rows,
    load_mix_config,
    row_matches_filters,
    row_text,
    source_slug,
)
from src.data.raw import iter_file_documents
from src.data.tokenizer import (
    SPECIAL_TOKEN_PIECES,
    SPECIAL_TOKENS,
    clean_text_for_tokenization,
    tokenizer_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the offline 16,384-piece SentencePiece BPE tokenizer")
    parser.add_argument("--input", nargs="*", default=[])
    parser.add_argument("--hf-dataset", default=None)
    parser.add_argument("--hf-config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--output-dir", default="data/tokenized")
    parser.add_argument("--vocab-size", type=int, default=16_384)
    parser.add_argument(
        "--tokenizer-type",
        choices=("sentencepiece", "byte-bpe"),
        default="sentencepiece",
        help="tokenizer implementation; byte-bpe starts from the UTF-8 byte alphabet",
    )
    parser.add_argument("--max-documents", type=int, default=2_000_000)
    parser.add_argument(
        "--mix-config",
        default=None,
        help="YAML HF mix; samples max-documents-per-source from each configured source",
    )
    parser.add_argument("--max-documents-per-source", type=int, default=50_000)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-backoff-seconds", type=float, default=5.0)
    parser.add_argument(
        "--parquet-root",
        default=None,
        help=(
            "read downloaded parquet shards from <root>/<source-name>/ instead of Hub streaming; "
            "useful with aria2c or another resumable downloader"
        ),
    )
    parser.add_argument(
        "--character-coverage",
        type=float,
        default=0.9995,
        help="fraction of normalized characters retained by SentencePiece",
    )
    parser.add_argument(
        "--byte-fallback",
        action="store_true",
        help="encode rare out-of-coverage characters as UTF-8 byte pieces",
    )
    parser.add_argument(
        "--max-sentence-length",
        type=int,
        default=16_384,
        help="maximum normalized characters per training line before SentencePiece skips it",
    )
    parser.add_argument(
        "--min-token-frequency",
        type=int,
        default=10,
        help="minimum corpus frequency for byte-level BPE merges",
    )
    parser.add_argument(
        "--max-token-length",
        type=int,
        default=32,
        help="maximum byte-level BPE token length",
    )
    return parser.parse_args()


def _iter_mix_documents(args: argparse.Namespace, mix_config: str) -> Iterator[str]:
    _, sources = load_mix_config(mix_config)
    if args.max_documents_per_source <= 0:
        raise ValueError("--max-documents-per-source must be positive")
    for source in sources:
        accepted = 0
        if args.parquet_root:
            source_dir = Path(args.parquet_root).resolve() / source_slug(source.name)
            parquet_paths = sorted(source_dir.glob("*.parquet"))
            if not parquet_paths:
                raise FileNotFoundError(f"no parquet shards found for {source.name!r} under {source_dir}")
            print(
                f"[tokenizer] {source.name}: reading {len(parquet_paths)} local parquet shard(s)",
                flush=True,
            )
            rows = iter_parquet_rows(parquet_paths)
        else:
            rows = iter_hf_rows(
                source,
                max_retries=args.max_retries,
                retry_backoff_seconds=args.retry_backoff_seconds,
            )
        for _, row in rows:
            if not row_matches_filters(row, source.filters):
                continue
            text = row_text(row, source.text_field)
            if text is None:
                continue
            yield text
            accepted += 1
            if accepted >= args.max_documents_per_source:
                break


def clean_training_text(text: str) -> str:
    """Normalize text and remove invisible/control code points before training."""

    return clean_text_for_tokenization(text)


def training_text_segments(text: str, max_chars: int) -> Iterator[str]:
    """Split long documents so SentencePiece does not silently drop them."""

    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    for start in range(0, len(text), max_chars):
        segment = text[start : start + max_chars]
        if segment:
            yield segment


def train_byte_level_bpe(
    corpus_path: Path,
    output_path: Path,
    vocab_size: int,
    *,
    min_token_frequency: int,
    max_token_length: int,
) -> int:
    """Train a true byte-level BPE tokenizer and save its JSON artifact."""

    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

    byte_alphabet = pre_tokenizers.ByteLevel.alphabet()
    special_tokens = [SPECIAL_TOKEN_PIECES[name] for name in SPECIAL_TOKENS]
    if vocab_size < len(byte_alphabet) + len(special_tokens):
        raise ValueError(
            "byte-level BPE vocab_size must fit 256 byte pieces plus four special tokens"
        )
    if min_token_frequency <= 0:
        raise ValueError("min_token_frequency must be positive")
    if max_token_length <= 0:
        raise ValueError("max_token_length must be positive")
    tokenizer = Tokenizer(models.BPE(unk_token=SPECIAL_TOKEN_PIECES["unk"]))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_token_frequency,
        initial_alphabet=byte_alphabet,
        special_tokens=special_tokens,
        max_token_length=max_token_length,
        show_progress=True,
    )
    tokenizer.train([str(corpus_path)], trainer=trainer)
    tokenizer.save(str(output_path))
    return tokenizer.get_vocab_size()


def main() -> None:
    args = parse_args()
    source_modes = sum(bool(value) for value in (args.input, args.hf_dataset, args.mix_config))
    if source_modes != 1:
        raise SystemExit("provide exactly one of --input, --hf-dataset, or --mix-config")
    if not 0 < args.character_coverage <= 1:
        raise SystemExit("--character-coverage must be in (0, 1]")
    if args.max_sentence_length <= 0:
        raise SystemExit("--max-sentence-length must be positive")
    if args.mix_config:
        documents = _iter_mix_documents(args, args.mix_config)
        source = {
            "mix_config": str(Path(args.mix_config).resolve()),
            "max_documents_per_source": args.max_documents_per_source,
        }
    elif args.hf_dataset:
        source_spec = HFMixSource(
            name="single_hf_source",
            dataset=args.hf_dataset,
            config=args.hf_config,
            split=args.split,
            text_field=args.text_field,
            token_budget=1,
            filters={},
        )
        documents = (
            text
            for _, row in iter_hf_rows(
                source_spec,
                max_retries=args.max_retries,
                retry_backoff_seconds=args.retry_backoff_seconds,
            )
            if (text := row_text(row, source_spec.text_field)) is not None
        )
        source = {"hf_dataset": args.hf_dataset, "split": args.split, "config": args.hf_config}
    else:
        documents = iter_file_documents(args.input, args.text_field)
        source = {"input": [str(Path(item).resolve()) for item in args.input]}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus = tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", prefix="j3-spm-", delete=False)
    corpus_path = Path(corpus.name)
    document_count = 0
    sentence_count = 0
    try:
        with corpus:
            for text in documents:
                if args.max_documents is not None and document_count >= args.max_documents:
                    break
                cleaned = clean_training_text(text)
                if not cleaned:
                    continue
                for segment in training_text_segments(cleaned, args.max_sentence_length):
                    corpus.write(segment + "\n")
                    sentence_count += 1
                document_count += 1
        if document_count == 0:
            raise RuntimeError("no documents found")
        if args.tokenizer_type == "sentencepiece":
            import sentencepiece as spm

            prefix = output_dir / "tokenizer"
            spm.SentencePieceTrainer.Train(
                input=str(corpus_path),
                model_prefix=str(prefix),
                vocab_size=args.vocab_size,
                model_type="bpe",
                character_coverage=args.character_coverage,
                normalization_rule_name="nfkc",
                byte_fallback=args.byte_fallback,
                hard_vocab_limit=False,
                max_sentence_length=args.max_sentence_length,
                pad_id=SPECIAL_TOKENS["pad"],
                unk_id=SPECIAL_TOKENS["unk"],
                bos_id=SPECIAL_TOKENS["bos"],
                eos_id=SPECIAL_TOKENS["eos"],
                pad_piece="<pad>",
                unk_piece="<unk>",
                bos_piece="<bos>",
                eos_piece="<eos>",
                # SentencePiece accepts either 0 (use all sentences) or a sample
                # size greater than 100; the 0-document-to-100-document case is
                # common in smoke tests and small validation corpora.
                input_sentence_size=0 if sentence_count <= 100 else min(sentence_count, 10_000_000),
                shuffle_input_sentence=True,
            )
            tokenizer_path = prefix.with_suffix(".model")
            actual_vocab_size = int(
                spm.SentencePieceProcessor(model_file=str(tokenizer_path)).get_piece_size()
            )
        else:
            tokenizer_path = output_dir / "tokenizer.json"
            actual_vocab_size = train_byte_level_bpe(
                corpus_path,
                tokenizer_path,
                args.vocab_size,
                min_token_frequency=args.min_token_frequency,
                max_token_length=args.max_token_length,
            )
        metadata = {
            "version": 1,
            "type": "sentencepiece_bpe" if args.tokenizer_type == "sentencepiece" else "byte_level_bpe",
            "vocab_size": args.vocab_size,
            "actual_vocab_size": actual_vocab_size,
            "special_tokens": SPECIAL_TOKENS,
            "special_token_pieces": SPECIAL_TOKEN_PIECES,
            "document_count": document_count,
            "training_line_count": sentence_count,
            "source": source,
            "parquet_root": str(Path(args.parquet_root).resolve()) if args.parquet_root else None,
            "tokenizer": str(tokenizer_path.resolve()),
            "tokenizer_type": args.tokenizer_type,
            "character_coverage": args.character_coverage
            if args.tokenizer_type == "sentencepiece"
            else None,
            "byte_fallback": args.byte_fallback if args.tokenizer_type == "sentencepiece" else False,
            "min_token_frequency": args.min_token_frequency
            if args.tokenizer_type == "byte-bpe"
            else None,
            "max_token_length": args.max_token_length if args.tokenizer_type == "byte-bpe" else None,
            "max_sentence_length": args.max_sentence_length,
            "text_cleanup": "NFKC; remove Cc/Cf/Co; collapse whitespace",
            "model_sha256": tokenizer_sha256(tokenizer_path),
        }
        (output_dir / "tokenizer_meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
        print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
    finally:
        corpus_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
