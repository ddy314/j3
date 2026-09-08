"""Run the GIBC V2 Track 01 evaluation with lm-evaluation-harness.

The harness does not know about J3's native PyTorch model, so this file keeps
the adapter in the repository. It uses local checkpoint weights and the
checked-in byte-level tokenizer; no hosted inference API is involved.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm_eval.api.instance import Instance  # noqa: E402
from lm_eval.api.model import LM  # noqa: E402
from lm_eval.evaluator import simple_evaluate  # noqa: E402
from lm_eval.tasks import TaskManager  # noqa: E402

from src.data.tokenizer import load_tokenizer, tokenizer_sha256  # noqa: E402
from src.model import DecoderLM  # noqa: E402
from src.training.checkpoint import CheckpointManager  # noqa: E402
from src.training.config import load_config  # noqa: E402
from src.training.trainer import configure_torch  # noqa: E402


def _resolve_checkpoint(value: str | Path) -> tuple[Path, Path]:
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    if path.name == "latest" and path.is_symlink():
        path = path.resolve()
    if path.is_dir() and path.name != "checkpoints" and (path / "checkpoints").is_dir():
        run_dir = path
        resolved = CheckpointManager(run_dir).resolve("auto")
    elif path.parent.name == "checkpoints":
        run_dir = path.parent.parent
        resolved = CheckpointManager(run_dir).resolve(path)
    elif path.is_file() and path.name == "state.pt" and path.parent.parent.name == "checkpoints":
        run_dir = path.parent.parent.parent
        resolved = CheckpointManager(run_dir).resolve(path.parent)
    else:
        raise FileNotFoundError(f"checkpoint must be a run directory or checkpoint path: {path}")
    if resolved is None:
        raise FileNotFoundError(f"no complete checkpoint found under {run_dir}")
    return run_dir, resolved


class J3HarnessLM(LM):
    """Minimal local adapter for lm-evaluation-harness log-likelihood tasks."""

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        config: str | Path,
        device: str,
        batch_size: int,
    ) -> None:
        super().__init__()
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        model_config, _, data_config, _ = load_config(config)
        configure_torch(True)
        self._device = torch.device(device)
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        run_dir, resolved = _resolve_checkpoint(checkpoint)
        payload, _ = CheckpointManager(run_dir).load(resolved)
        self._model = DecoderLM(model_config).to(self._device)
        self._model.load_state_dict(payload["model"])
        self._model.eval()
        if not data_config.tokenizer:
            raise ValueError("evaluation config must declare data.tokenizer")
        self._tokenizer = load_tokenizer(data_config.tokenizer)
        self._tokenizer_hash = tokenizer_sha256(data_config.tokenizer)
        self._model_name = model_config.model_name
        self.max_length = model_config.max_seq_len
        self.batch_size = batch_size
        self._pad_id = self._tokenizer.pad_id
        self._bos_id = self._tokenizer.bos_id
        self.checkpoint = str(resolved)

    @property
    def tokenizer_name(self) -> str:
        return f"j3-byte-bpe:{self._tokenizer_hash}"

    def _autocast(self):
        if self._device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_bos=False, add_eos=False)

    def _sequence_for_continuation(self, context: str, continuation: str) -> tuple[list[int], int]:
        continuation_ids = self._encode(continuation)
        if not continuation_ids:
            return [self._bos_id], 1
        context_ids = self._encode(context)
        max_context = self.max_length - len(continuation_ids)
        if max_context < 1:
            continuation_ids = continuation_ids[-(self.max_length - 1) :]
            context_ids = []
            max_context = 1
        context_tail = context_ids[-(max_context - 1) :] if max_context > 1 else []
        prefix = [self._bos_id] + context_tail
        return prefix + continuation_ids, len(prefix)

    def _score_sequences(
        self, sequences: list[tuple[list[int], int]]
    ) -> list[tuple[float, bool]]:
        outputs: list[tuple[float, bool]] = []
        for offset in range(0, len(sequences), self.batch_size):
            batch = sequences[offset : offset + self.batch_size]
            length = max(len(ids) for ids, _ in batch)
            input_ids = torch.full(
                (len(batch), length), self._pad_id, dtype=torch.long, device=self._device
            )
            for row, (ids, _) in enumerate(batch):
                input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=self._device)
            with torch.inference_mode(), self._autocast():
                logits = self._model(input_ids).logits.float()
            for row, (ids, continuation_start) in enumerate(batch):
                if continuation_start >= len(ids):
                    outputs.append((0.0, True))
                    continue
                positions = torch.arange(
                    continuation_start - 1, len(ids) - 1, device=self._device
                )
                targets = torch.tensor(
                    ids[continuation_start:], dtype=torch.long, device=self._device
                )
                selected = logits[row, positions]
                log_probs = selected.log_softmax(dim=-1)
                token_scores = log_probs.gather(1, targets[:, None]).squeeze(1)
                greedy = bool((selected.argmax(dim=-1) == targets).all().item())
                outputs.append((float(token_scores.sum().item()), greedy))
            del logits, input_ids
        return outputs

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        sequences = [self._sequence_for_continuation(*request.args) for request in requests]
        return self._score_sequences(sequences)

    def _rolling_windows(self, text: str) -> list[tuple[list[int], int]]:
        tokens = self._encode(text)
        if not tokens:
            return []
        stride = self.max_length - 1
        windows: list[tuple[list[int], int]] = []
        previous_end = 0
        while previous_end < len(tokens):
            end = min(len(tokens), previous_end + stride)
            begin = previous_end - 1 if previous_end else 0
            sequence = [self._bos_id] + tokens[begin:end]
            score_start = 1 + previous_end - begin
            windows.append((sequence, score_start))
            previous_end = end
        return windows

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        all_windows: list[tuple[list[int], int]] = []
        spans: list[tuple[int, int]] = []
        for request in requests:
            start = len(all_windows)
            all_windows.extend(self._rolling_windows(request.args[0]))
            spans.append((start, len(all_windows)))
        scores = self._score_sequences(all_windows)
        return [sum(score for score, _ in scores[start:end]) for start, end in spans]

    def generate_until(self, requests: list[Instance]) -> list[str]:
        results: list[str] = []
        for request in requests:
            context, generation_kwargs = request.args
            kwargs = generation_kwargs or {}
            max_new_tokens = int(kwargs.get("max_gen_toks", kwargs.get("max_new_tokens", 32)))
            input_ids = torch.tensor(
                [[self._bos_id] + self._encode(context)[-(self.max_length - 1) :]],
                dtype=torch.long,
                device=self._device,
            )
            with torch.inference_mode(), self._autocast():
                generated = self._model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=1.0,
                    do_sample=False,
                )[0].tolist()
            new_tokens = generated[input_ids.shape[1] :]
            if hasattr(self._tokenizer, "_tokenizer"):
                text = self._tokenizer._tokenizer.decode(new_tokens)
            else:
                text = self._tokenizer._processor.decode(new_tokens)
            for stop in kwargs.get("until", []) or []:
                if stop in text:
                    text = text.split(stop, 1)[0]
            results.append(text)
        return results


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (torch.Tensor,)):
        return value.item() if value.ndim == 0 else value.tolist()
    if callable(value):
        return getattr(value, "__name__", repr(value))
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate J3 on the GIBC V2 Track 01 tasks")
    parser.add_argument("--checkpoint", default="runs/j3-stage3-final/checkpoints/latest")
    parser.add_argument("--config", default="configs/pretrain_stage3_1b.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["hellaswag", "arc_easy", "piqa", "winogrande", "j3_wikitext_103"],
    )
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--output", default="docs/evaluation/results/j3-gibc.json")
    parser.add_argument("--log-samples", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")

    model = J3HarnessLM(
        checkpoint=args.checkpoint,
        config=args.config,
        device=args.device,
        batch_size=args.batch_size,
    )
    task_manager = TaskManager(include_path=Path(__file__).resolve().parents[1] / "eval_tasks")
    results = simple_evaluate(
        model=model,
        tasks=args.tasks,
        num_fewshot=0,
        batch_size=args.batch_size,
        limit=args.limit,
        log_samples=args.log_samples,
        task_manager=task_manager,
        random_seed=1337,
        numpy_random_seed=1337,
        torch_random_seed=1337,
        fewshot_random_seed=1337,
    )
    if results is None:
        raise RuntimeError("lm-evaluation-harness returned no results")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "J3",
        "checkpoint": model.checkpoint,
        "tokenizer_sha256": model._tokenizer_hash,
        "tasks": args.tasks,
        "num_fewshot": 0,
        "limit": args.limit,
        "harness": "lm-evaluation-harness",
        "harness_version": "0.4.13-or-newer",
        "results": results,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n")
    print(json.dumps(payload["results"], ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
