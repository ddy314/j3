from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.raw import iter_file_documents, iter_hf_documents
from src.data.tokenizer import SPECIAL_TOKENS, tokenizer_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the offline 16,384-piece SentencePiece BPE tokenizer")
    parser.add_argument("--input", nargs="*", default=[])
    parser.add_argument("--hf-dataset", default=None)
    parser.add_argument("--hf-config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--output-dir", default="data/tokenized")
    parser.add_argument("--vocab-size", type=int, default=16_384)
    parser.add_argument("--max-documents", type=int, default=2_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input and not args.hf_dataset:
        raise SystemExit("provide --input files/directories or --hf-dataset")
    if args.hf_dataset:
        documents = iter_hf_documents(
            args.hf_dataset,
            config=args.hf_config,
            split=args.split,
            text_field=args.text_field,
            max_documents=args.max_documents,
        )
        source = {"hf_dataset": args.hf_dataset, "split": args.split, "config": args.hf_config}
    else:
        documents = iter_file_documents(args.input, args.text_field)
        source = {"input": [str(Path(item).resolve()) for item in args.input]}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus = tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", prefix="j3-spm-", delete=False)
    corpus_path = Path(corpus.name)
    count = 0
    try:
        with corpus:
            for text in documents:
                if args.max_documents is not None and count >= args.max_documents:
                    break
                corpus.write(text.replace("\n", " ") + "\n")
                count += 1
        if count == 0:
            raise RuntimeError("no documents found")
        import sentencepiece as spm

        prefix = output_dir / "tokenizer"
        spm.SentencePieceTrainer.Train(
            input=str(corpus_path),
            model_prefix=str(prefix),
            vocab_size=args.vocab_size,
            model_type="bpe",
            character_coverage=1.0,
            normalization_rule_name="nfkc",
            hard_vocab_limit=False,
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
            input_sentence_size=0 if count <= 100 else min(count, 10_000_000),
            shuffle_input_sentence=True,
        )
        model_path = prefix.with_suffix(".model")
        metadata = {
            "version": 1,
            "type": "sentencepiece_bpe",
            "vocab_size": args.vocab_size,
            "actual_vocab_size": int(spm.SentencePieceProcessor(model_file=str(model_path)).get_piece_size()),
            "special_tokens": SPECIAL_TOKENS,
            "document_count": count,
            "source": source,
            "model_sha256": tokenizer_sha256(model_path),
        }
        (output_dir / "tokenizer_meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
        print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
    finally:
        corpus_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
