# Built With

- **Language/runtime:** Python 3.13, PyTorch 2.14.0+cu130, CUDA 13.0, Triton 3.8.0.
- **Model implementation:** native PyTorch decoder-only transformer in `src/model`.
- **Training:** resumable PyTorch loop in `src/training`, BF16 mixed precision, AdamW, deterministic seed 1337, checkpoint metadata and JSONL metrics.
- **Tokenizer:** checked-in 16,384-vocabulary byte-level BPE tokenizer; hash recorded in the release manifest.
- **Data tooling:** local Parquet/JSONL materialization, uint16 token shards, manifest-driven streams, deterministic global chunk mixing, duplicate filtering, and benchmark decontamination.
- **Evaluation:** `lm-evaluation-harness` with a checked-in adapter for the native model; no hosted model API.
- **Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU with 7,834.375 MiB reported memory.
- **Repository tooling:** `uv`, pytest, YAML configurations, JSON/JSONL audit records, and Git.
- **AI assistance:** AI-assisted coding and documentation were used during project preparation. No external pretrained model weights were used for J3 training, and the assistance is disclosed here rather than presented as an unassisted implementation.

Exact versions, commands, timestamps, and environment snapshots are in [`docs/training/experiment-register.json`](../training/experiment-register.json) and [`docs/training/raw-runs`](../training/raw-runs).
