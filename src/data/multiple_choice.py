from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch


def manifest_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class MultipleChoiceExample:
    """One pre-tokenized multiple-choice example.

    ``prompt_ids`` and every ``choice_ids`` sequence are tokenized separately.
    This makes the boundary used by training and evaluation explicit and
    reproducible; callers should include any required leading space in the
    choice text before tokenization.
    """

    example_id: str
    prompt_ids: tuple[int, ...]
    choice_ids: tuple[tuple[int, ...], ...]
    answer: int
    category: str = "unspecified"
    source: str = "unspecified"

    def __post_init__(self) -> None:
        if not self.example_id:
            raise ValueError("multiple-choice example_id must be non-empty")
        if not self.prompt_ids:
            raise ValueError(f"multiple-choice example {self.example_id!r} has an empty prompt")
        if len(self.choice_ids) < 2:
            raise ValueError(f"multiple-choice example {self.example_id!r} needs at least two choices")
        if not 0 <= self.answer < len(self.choice_ids):
            raise ValueError(
                f"multiple-choice example {self.example_id!r} answer {self.answer} is out of range"
            )
        if any(not choice for choice in self.choice_ids):
            raise ValueError(f"multiple-choice example {self.example_id!r} has an empty choice")


@dataclass(frozen=True)
class MultipleChoiceBatch:
    input_ids: torch.Tensor
    continuation_mask: torch.Tensor
    group_offsets: torch.Tensor
    positive_indices: torch.Tensor
    example_ids: tuple[str, ...]


def _as_int_sequence(value: Any, field: str, location: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location} field {field!r} must be a non-empty list")
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location} field {field!r} contains a non-integer token") from exc
    if any(item < 0 for item in result):
        raise ValueError(f"{location} field {field!r} contains a negative token")
    return result


def example_from_record(
    record: dict[str, Any],
    *,
    location: str,
    vocab_size: int | None = None,
) -> MultipleChoiceExample:
    """Parse and validate one line from a tokenized MC JSONL file."""

    raw_id = record.get("id", record.get("example_id"))
    example_id = str(raw_id).strip() if raw_id is not None else ""
    prompt_ids = _as_int_sequence(record.get("prompt_ids"), "prompt_ids", location)
    raw_choices = record.get("choice_ids")
    if not isinstance(raw_choices, list) or len(raw_choices) < 2:
        raise ValueError(f"{location} field 'choice_ids' must contain at least two lists")
    choice_ids = tuple(
        _as_int_sequence(choice, f"choice_ids[{index}]", location)
        for index, choice in enumerate(raw_choices)
    )
    raw_answer = record.get("answer", record.get("label"))
    try:
        answer = int(raw_answer)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location} answer/label must be an integer") from exc
    if vocab_size is not None:
        for field, sequences in (("prompt_ids", (prompt_ids,)), ("choice_ids", choice_ids)):
            maximum = max(token for sequence in sequences for token in sequence)
            if maximum >= vocab_size:
                raise ValueError(
                    f"{location} field {field!r} has token {maximum} >= vocab_size {vocab_size}"
                )
    return MultipleChoiceExample(
        example_id=example_id,
        prompt_ids=prompt_ids,
        choice_ids=choice_ids,
        answer=answer,
        category=str(record.get("category", "unspecified")),
        source=str(record.get("source", "unspecified")),
    )


class MultipleChoiceStream:
    """Deterministically shuffled, exact-resume stream over tokenized MC data."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str = "train",
        shuffle: bool = True,
        shuffle_seed: int = 1337,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("type") != "multiple_choice_tokenized_v1":
            raise ValueError(f"unsupported multiple-choice manifest type: {self.manifest_path}")
        self.manifest_hash = manifest_sha256(self.manifest_path)
        self.tokenizer_hash = self.manifest.get("tokenizer_hash")
        self.vocab_size = int(self.manifest.get("vocab_size", 0))
        if self.vocab_size <= 0:
            raise ValueError(f"multiple-choice manifest has invalid vocab_size: {self.manifest_path}")
        self.pad_id = int(self.manifest.get("pad_id", 0))
        self.split = split
        self.shuffle = bool(shuffle)
        self.shuffle_seed = int(shuffle_seed)
        records_key = f"{split}_file"
        records_value = self.manifest.get(records_key)
        if records_value is None and split == "val":
            records_value = self.manifest.get("validation_file")
        if not isinstance(records_value, str):
            raise ValueError(f"manifest has no {split} records file: {self.manifest_path}")
        records_path = Path(records_value)
        if not records_path.is_absolute():
            records_path = self.manifest_path.parent / records_path
        if not records_path.is_file():
            raise FileNotFoundError(records_path)
        expected_records_hash = self.manifest.get(f"{split}_file_sha256")
        if expected_records_hash is not None and manifest_sha256(records_path) != expected_records_hash:
            raise ValueError(f"multiple-choice records hash mismatch: {records_path}")
        self.records_path = records_path
        self.examples = self._read_examples()
        if not self.examples:
            raise ValueError(f"multiple-choice records file is empty: {records_path}")
        self.epoch = 0
        self.order = list(range(len(self.examples)))
        self.order_position = 0
        self._reset_order()

    def _read_examples(self) -> list[MultipleChoiceExample]:
        examples: list[MultipleChoiceExample] = []
        with self.records_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {self.records_path}:{line_number}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"record at {self.records_path}:{line_number} is not an object")
                examples.append(
                    example_from_record(
                        record,
                        location=f"{self.records_path}:{line_number}",
                        vocab_size=self.vocab_size,
                    )
                )
        expected = self.manifest.get(f"{self.split}_count")
        if expected is not None and int(expected) != len(examples):
            raise ValueError(
                f"{self.split} record count mismatch: {len(examples)} != {int(expected)}"
            )
        return examples

    def _reset_order(self) -> None:
        self.order = list(range(len(self.examples)))
        if self.shuffle:
            random.Random(self.shuffle_seed + self.epoch).shuffle(self.order)
        self.order_position = 0

    def _next_index(self) -> int:
        if self.order_position >= len(self.order):
            self.epoch += 1
            self._reset_order()
        index = self.order[self.order_position]
        self.order_position += 1
        return index

    def next_batch(self, batch_size: int) -> list[MultipleChoiceExample]:
        if batch_size <= 0:
            raise ValueError("multiple-choice batch_size must be positive")
        return [self.examples[self._next_index()] for _ in range(batch_size)]

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "manifest_hash": self.manifest_hash,
            "split": self.split,
            "epoch": self.epoch,
            "order": list(self.order),
            "order_position": self.order_position,
            "shuffle": self.shuffle,
            "shuffle_seed": self.shuffle_seed,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("manifest_hash") != self.manifest_hash:
            raise ValueError("multiple-choice manifest hash does not match checkpoint")
        if state.get("split", "train") != self.split:
            raise ValueError("multiple-choice split differs from checkpoint")
        if bool(state.get("shuffle", self.shuffle)) != self.shuffle:
            raise ValueError("multiple-choice shuffle setting differs from checkpoint")
        if int(state.get("shuffle_seed", self.shuffle_seed)) != self.shuffle_seed:
            raise ValueError("multiple-choice shuffle seed differs from checkpoint")
        order = [int(item) for item in state.get("order", [])]
        if sorted(order) != list(range(len(self.examples))):
            raise ValueError("invalid multiple-choice order in checkpoint")
        position = int(state.get("order_position", 0))
        if not 0 <= position <= len(order):
            raise ValueError("invalid multiple-choice order_position in checkpoint")
        self.epoch = int(state.get("epoch", 0))
        self.order = order
        self.order_position = position


def collate_multiple_choice(
    examples: Sequence[MultipleChoiceExample],
    *,
    pad_id: int = 0,
    pad_to_length: int | None = None,
) -> MultipleChoiceBatch:
    """Flatten choices while retaining group offsets for pairwise ranking."""

    if not examples:
        raise ValueError("cannot collate an empty multiple-choice batch")
    if pad_to_length is not None and pad_to_length <= 1:
        raise ValueError("pad_to_length must be at least 2")
    max_length = pad_to_length or max(
        len(example.prompt_ids) + len(choice)
        for example in examples
        for choice in example.choice_ids
    )
    if max_length <= 1:
        raise ValueError("multiple-choice sequences must contain at least two tokens")

    sequences: list[tuple[list[int], int]] = []
    group_offsets = [0]
    positive_indices: list[int] = []
    for example in examples:
        group_start = len(sequences)
        max_choice_length = max(len(choice) for choice in example.choice_ids)
        available_prompt = max_length - max_choice_length
        if available_prompt <= 0:
            raise ValueError(
                f"choices for {example.example_id!r} leave no prompt at length {max_length}"
            )
        # Every candidate for one question must see the same truncated
        # context. Cropping separately per choice would give shorter answers
        # extra prompt tokens and make the comparison unfair.
        prompt = list(example.prompt_ids)[-available_prompt:]
        for choice in example.choice_ids:
            sequence = prompt + list(choice)
            if len(sequence) > max_length:
                raise AssertionError("multiple-choice truncation invariant violated")
            sequences.append((sequence, len(prompt)))
        group_offsets.append(len(sequences))
        positive_indices.append(group_start + example.answer)

    input_ids = torch.full((len(sequences), max_length), int(pad_id), dtype=torch.long)
    continuation_mask = torch.zeros((len(sequences), max_length - 1), dtype=torch.bool)
    for row, (sequence, prompt_length) in enumerate(sequences):
        input_ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        # The target at full-sequence position t is predicted from t-1. Only
        # positions belonging to the choice continuation contribute to s_i.
        start = prompt_length - 1
        end = len(sequence) - 1
        continuation_mask[row, start:end] = True

    return MultipleChoiceBatch(
        input_ids=input_ids,
        continuation_mask=continuation_mask,
        group_offsets=torch.tensor(group_offsets, dtype=torch.long),
        positive_indices=torch.tensor(positive_indices, dtype=torch.long),
        example_ids=tuple(example.example_id for example in examples),
    )


def combined_manifest_hash(*hashes: str | None) -> str:
    values = [value for value in hashes if value]
    if not values:
        return ""
    return hashlib.sha256("\n".join(values).encode("ascii")).hexdigest()
