from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path
from typing import Iterable


SPECIAL_TOKENS = {"pad": 0, "unk": 1, "bos": 2, "eos": 3}
SPECIAL_TOKEN_PIECES = {
    "pad": "<|j3_pad|>",
    "unk": "<|j3_unk|>",
    "bos": "<|j3_bos|>",
    "eos": "<|j3_eos|>",
}


def tokenizer_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_text_for_tokenization(text: str) -> str:
    """Apply the shared text cleanup used for tokenizer training and data shards."""

    normalized = unicodedata.normalize("NFKC", text)
    cleaned: list[str] = []
    for char in normalized:
        category = unicodedata.category(char)
        if category in {"Cc", "Cf", "Co"}:
            if char in {"\n", "\t", "\r"}:
                cleaned.append(" ")
            continue
        cleaned.append(char)
    return " ".join("".join(cleaned).split())


class SentencePieceTokenizer:
    """Small wrapper that keeps tokenizer use out of the training hot path."""

    def __init__(self, model_path: str | Path) -> None:
        import sentencepiece as spm

        self.path = Path(model_path)
        self._processor = spm.SentencePieceProcessor(model_file=str(self.path))
        self.vocab_size = self._processor.get_piece_size()
        self.sha256 = tokenizer_sha256(self.path)

    @property
    def pad_id(self) -> int:
        return SPECIAL_TOKENS["pad"]

    @property
    def unk_id(self) -> int:
        return SPECIAL_TOKENS["unk"]

    @property
    def bos_id(self) -> int:
        return SPECIAL_TOKENS["bos"]

    @property
    def eos_id(self) -> int:
        return SPECIAL_TOKENS["eos"]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = True) -> list[int]:
        ids = list(self._processor.encode(text, out_type=int))
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def encode_iter(self, texts: Iterable[str], add_bos: bool = False, add_eos: bool = True):
        for text in texts:
            yield self.encode(text, add_bos=add_bos, add_eos=add_eos)


class ByteLevelBPETokenizer:
    """Hugging Face byte-level BPE wrapper with the same training interface."""

    def __init__(self, tokenizer_path: str | Path) -> None:
        from tokenizers import Tokenizer

        self.path = Path(tokenizer_path)
        self._tokenizer = Tokenizer.from_file(str(self.path))
        self.vocab_size = self._tokenizer.get_vocab_size()
        self.sha256 = tokenizer_sha256(self.path)
        self._special_ids = {
            name: self._tokenizer.token_to_id(SPECIAL_TOKEN_PIECES[name]) for name in SPECIAL_TOKENS
        }
        missing = [name for name, value in self._special_ids.items() if value is None]
        if missing:
            raise ValueError(f"byte-level tokenizer is missing special tokens: {missing}")

    @property
    def pad_id(self) -> int:
        return int(self._special_ids["pad"])

    @property
    def unk_id(self) -> int:
        return int(self._special_ids["unk"])

    @property
    def bos_id(self) -> int:
        return int(self._special_ids["bos"])

    @property
    def eos_id(self) -> int:
        return int(self._special_ids["eos"])

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = True) -> list[int]:
        ids = [int(value) for value in self._tokenizer.encode(text, add_special_tokens=False).ids]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def encode_iter(self, texts: Iterable[str], add_bos: bool = False, add_eos: bool = True):
        for text in texts:
            yield self.encode(text, add_bos=add_bos, add_eos=add_eos)


def load_tokenizer(path: str | Path) -> SentencePieceTokenizer | ByteLevelBPETokenizer:
    """Load the tokenizer format selected by its artifact suffix."""

    tokenizer_path = Path(path)
    if tokenizer_path.suffix.lower() == ".json":
        return ByteLevelBPETokenizer(tokenizer_path)
    return SentencePieceTokenizer(tokenizer_path)
