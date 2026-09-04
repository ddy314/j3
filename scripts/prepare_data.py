from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.raw import iter_file_documents, iter_hf_documents
from src.data.shards import ShardWriter
from src.data.tokenizer import clean_text_for_tokenization, load_tokenizer, tokenizer_sha256
from src.training.metrics import atomic_json_write


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize raw documents once into uint16 mmap shards")
    parser.add_argument("--input", nargs="*", default=[])
    parser.add_argument("--hf-dataset", default=None)
    parser.add_argument("--hf-config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", default="data/tokenized")
    parser.add_argument("--shard-tokens", type=int, default=4_194_304)
    parser.add_argument("--val-ratio", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--deduplicate", action="store_true")
    parser.add_argument("--max-documents", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input and not args.hf_dataset:
        raise SystemExit("provide --input files/directories or --hf-dataset")
    if not 0 <= args.val_ratio < 1:
        raise SystemExit("--val-ratio must be in [0, 1)")
    tokenizer = load_tokenizer(args.tokenizer)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_writer = ShardWriter(output_dir, "shard", args.shard_tokens, output_dir)
    val_writer = ShardWriter(output_dir / "val", "shard", args.shard_tokens, output_dir)
    rng = random.Random(args.seed)
    seen: set[str] = set()
    frequency = np.zeros(tokenizer.vocab_size, dtype=np.int64)
    stats = {
        "document_count": 0,
        "train_document_count": 0,
        "val_document_count": 0,
        "duplicate_count": 0,
        "empty_document_count": 0,
        "unk_token_count": 0,
        "language_distribution": {"en_like": 0, "other_or_mixed": 0},
    }
    if args.hf_dataset:
        documents = iter_hf_documents(
            args.hf_dataset,
            config=args.hf_config,
            split=args.split,
            text_field=args.text_field,
            max_documents=args.max_documents,
        )
        source: object = {"hf_dataset": args.hf_dataset, "config": args.hf_config, "split": args.split}
    else:
        documents = iter_file_documents(args.input, args.text_field)
        source = {"input": [str(Path(item).resolve()) for item in args.input]}
    try:
        for index, text in enumerate(documents):
            if args.max_documents is not None and index >= args.max_documents:
                break
            stats["document_count"] += 1
            normalized = clean_text_for_tokenization(text)
            if not normalized:
                stats["empty_document_count"] += 1
                continue
            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if args.deduplicate and digest in seen:
                stats["duplicate_count"] += 1
                continue
            if args.deduplicate:
                seen.add(digest)
            ids = tokenizer.encode(normalized, add_bos=False, add_eos=True)
            stats["unk_token_count"] += ids.count(tokenizer.unk_id)
            frequency += np.bincount(ids, minlength=tokenizer.vocab_size)
            if sum(char.isascii() and char.isalpha() for char in normalized) >= 0.7 * max(1, sum(char.isalpha() for char in normalized)):
                stats["language_distribution"]["en_like"] += 1
            else:
                stats["language_distribution"]["other_or_mixed"] += 1
            if args.val_ratio and rng.random() < args.val_ratio:
                val_writer.write(ids)
                stats["val_document_count"] += 1
            else:
                train_writer.write(ids)
                stats["train_document_count"] += 1
    finally:
        train_writer.close()
        val_writer.close()
    if train_writer.total_tokens < 2:
        raise RuntimeError("training split has fewer than two tokens")
    if args.val_ratio > 0 and val_writer.total_tokens < 2:
        raise RuntimeError("validation split is empty; use more documents or a larger --val-ratio")
    manifest = {
        "version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset": source,
        "source": source,
        "tokenizer": str(Path(args.tokenizer).resolve()),
        "tokenizer_hash": tokenizer_sha256(args.tokenizer),
        "vocab_size": tokenizer.vocab_size,
        "dtype": "uint16",
        "shard_token_target": args.shard_tokens,
        "train_shards": train_writer.entries,
        "val_shards": val_writer.entries,
        "train_token_count": train_writer.total_tokens,
        "val_token_count": val_writer.total_tokens,
        "total_token_count": train_writer.total_tokens + val_writer.total_tokens,
        "statistics": {
            **stats,
            "average_document_tokens": (train_writer.total_tokens + val_writer.total_tokens) / max(1, stats["train_document_count"] + stats["val_document_count"]),
            "duplicate_filtering": "enabled" if args.deduplicate else "disabled",
            "unk_rate": stats["unk_token_count"] / max(1, train_writer.total_tokens + val_writer.total_tokens),
            "token_frequency_top_100": [
                {"token_id": int(token_id), "count": int(frequency[token_id])}
                for token_id in np.argsort(frequency)[-100:][::-1]
                if frequency[token_id] > 0
            ],
        },
    }
    atomic_json_write(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
