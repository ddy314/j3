# J3

J3 is a decoder-only language model trained from scratch for the **Global Innovation Build Challenge V2**, Track 01 TECH. The repository is prepared around one public submission candidate: the Stage 3 1B-token run (`20260906-225321`). The later Stage 4, capability-CPT, and multiple-choice post-training experiments are documented as non-selected experiments and are not part of J3.

## Submission status

| Item | Value |
| --- | --- |
| Public model name | **J3** |
| Selected training run | `20260906-225321` |
| Trainable parameters | `49,001,408` |
| Training tokens | `1,000,079,360` seen; `1,000,000,000` target |
| Validation loss / perplexity | `3.177146828174591` / `23.978241816690886` |
| GIBC evaluation | HellaSwag `0.27246` acc-norm; ARC-Easy `0.40699` acc; PIQA `0.59684` acc; WinoGrande `0.51618` acc; WikiText-103 `96.1230` word-PPL |
| Checkpoint | 588,414,613 bytes; SHA-256 is recorded in [`release-manifest.json`](docs/submission/release-manifest.json) |
| Training hardware | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GiB class |
| External pretrained weights | None |

The internal shape shorthand `D32-1216-R12` appears in historical run records only. It is not the submitted model name.

The competition rules require a public source repository, a model trained from scratch, no more than 50M trainable parameters, and evaluation with `lm-evaluation-harness` on HellaSwag, ARC-Easy, PIQA, WinoGrande, plus held-out WikiText-103 perplexity. The repository keeps those requirements and the remaining submission items in [`docs/submission/checklist.md`](docs/submission/checklist.md). See the [official competition rules](https://gibc-v2.devpost.com/rules).

## Model

J3 uses the fixed 49M architecture in [`configs/pretrain_stage3_1b.yaml`](configs/pretrain_stage3_1b.yaml): 32 decoder blocks, model width 384, 6 query heads with 2 key/value heads, 64-dimensional heads, 1,216-dimensional MLPs, a rank-12 output residual head, tied input/output embeddings, Q/K RMS normalization, RoPE, and xIELU activations. The maximum model context is 1,024 tokens; the selected training sequence length was 512.

The model is instantiated locally by `src/model`. No hosted inference endpoint or third-party model API is used by training or evaluation.

## Data and disclosure

The selected one-billion-token training stream is a deterministic, globally mixed stream of 8,192-token chunks with seed `1337`. Its exact quotas, filters, source provenance, tokenizer hash, validation split, benchmark decontamination rule, and license notes are in [`docs/data/datasets.md`](docs/data/datasets.md) and [`docs/data/stage3-mixed-manifest.json`](docs/data/stage3-mixed-manifest.json).

The repository discloses all local training runs that were present when the record was exported. Raw JSON metadata, commands, environment snapshots, status files, metrics, and checkpoint metadata are under [`docs/training/raw-runs`](docs/training/raw-runs); the searchable register is [`docs/training/experiment-register.json`](docs/training/experiment-register.json). Checkpoint tensor files are intentionally excluded from Git. Their selected release hash and publication instructions are in [`docs/submission/release-manifest.json`](docs/submission/release-manifest.json).

The full zero-shot evaluation completed with `lm-evaluation-harness` 0.4.13, `num_fewshot=0`, and the default 100,000 bootstrap iterations. The complete JSON, including standard errors and harness metadata, is [`docs/evaluation/results/j3-gibc.json`](docs/evaluation/results/j3-gibc.json). The accuracy values above use the conventional normalized score where the task exposes `acc_norm`; both raw and normalized values are preserved in the result file.

The team used AI-assisted coding/documentation tooling during repository preparation. The model, data decisions, experiment selection, measurements, and disclosure remain the team's responsibility and are recorded as facts or explicitly marked as pending/unknown.

## Reproduce and verify

```bash
uv sync --locked
uv run python scripts/verify_submission.py
uv run pytest -q
```

The selected local checkpoint is expected at `runs/j3-stage3-final/checkpoints/latest/state.pt`. The evaluation extra installs the required harness:

```bash
uv sync --locked --extra eval
uv run python scripts/evaluate_gibc.py \
  --checkpoint runs/j3-stage3-final/checkpoints/latest \
  --config configs/pretrain_stage3_1b.yaml \
  --tasks hellaswag arc_easy piqa winogrande j3_wikitext_103 \
  --output docs/evaluation/results/j3-gibc.json
```

The evaluation adapter is intentionally checked in at [`scripts/evaluate_gibc.py`](scripts/evaluate_gibc.py). It loads local J3 weights and the checked-in tokenizer, and emits a JSON record with the exact harness arguments and results. A small smoke run can be made with `--limit 2` before running the full evaluation.

To launch a fresh run, use the selected config and the verified manifest. The exact selected run was initialized from the preceding in-house Stage 2 continuation; that history is preserved in the training register and raw records. This is still from-scratch training in the competition sense: no external pretrained checkpoint, fine-tuning, or distillation was used.

## Repository map

- [`configs/`](configs): authoritative model, data, and training configurations.
- [`src/`](src): tokenizer, data stream, model, checkpoint, and training loop.
- [`scripts/`](scripts): data preparation, verification, training-record export, and evaluation utilities.
- [`docs/data/`](docs/data): manifests, tokenizer audit, verification artifacts, and dataset disclosure.
- [`docs/training/`](docs/training): complete run register and raw experiment records.
- [`docs/evaluation/`](docs/evaluation): benchmark protocol and result artifacts.
- [`docs/submission/`](docs/submission): competition description, release manifest, checklist, and screenshots.

## License and redistribution

The repository contains source code and audit metadata, not the raw web/PDF corpora. Dataset cards and source licenses are linked in [`docs/data/datasets.md`](docs/data/datasets.md); derived web content and the local FinePDF source still require the original providers' terms to be checked before redistribution. Do not treat a dataset-card license as a blanket license for every underlying document.

## Competition materials

The English project description, Built With disclosure, screenshot placeholders, and final checklist are kept in [`docs/submission`](docs/submission). The official submission deadline and required fields can change, so the final Devpost form must be checked against the [official GIBC V2 rules](https://gibc-v2.devpost.com/rules) immediately before submission.
