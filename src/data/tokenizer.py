from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


SPECIAL_TOKENS = {"pad": 0, "unk": 1, "bos": 2, "eos": 3}


def tokenizer_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
