# J3 — GIBC V2 project description

## Overview

J3 is a compact decoder-only language model built and trained from scratch under a strict 50M trainable-parameter budget. The project focuses on making a small model's training path inspectable: the source code, data mixture manifest, tokenizer audit, decontamination index, training metrics, environment records, checkpoint hash, and benchmark adapter are kept together in one public repository.

## What was built

J3 has 49,001,408 trainable parameters. Its fixed architecture has 32 decoder blocks, width 384, grouped-query attention with 6 query heads and 2 key/value heads, 64-dimensional heads, 1,216-dimensional MLPs, tied embeddings, Q/K RMS normalization, RoPE, xIELU activations, and a rank-12 output residual head. Training is implemented with a resumable native PyTorch loop, BF16 support, deterministic seeds, checkpoint metadata, validation logging, and explicit tokenizer/data hashes.

## Data and training

The selected Stage 3 stream contains exactly 1,000,000,000 training tokens after materialization: FinePDF local text (299,793,532), FineWeb-Edu (350,206,468), ordinary-score Ultra-FineWeb (125,000,000), Cosmopedia V2 stories (75,000,000), Cosmopedia V2 WikiHow (50,000,000), and WikiText-103 (100,000,000). The data was deduplicated and decontaminated against public benchmark shingles, then globally shuffled in 8,192-token chunks with seed 1337.

The selected run saw 1,000,079,360 tokens, completed 7,630 optimizer steps, and recorded validation loss 3.177146828174591 (perplexity 23.978241816690886) on the training-loop validation stream. It ran for 24,595.263 seconds on an NVIDIA GeForce RTX 4060 Laptop GPU, using BF16 and an effective batch of 131,072 tokens.

## Evaluation

The full zero-shot run used `lm-evaluation-harness` 0.4.13 with the default 100,000 bootstrap iterations. HellaSwag accuracy was 0.27246 (normalized 0.28082), ARC-Easy accuracy was 0.40699, PIQA accuracy was 0.59684, WinoGrande accuracy was 0.51618, and WikiText-103 word perplexity was 96.1230 (byte perplexity 2.41322). The complete JSON, including standard errors and harness metadata, is stored in `docs/evaluation/results/j3-gibc.json`. No hosted inference API is used.

## Honest scope

The selected release is the Stage 3 1B-token model. A later Stage 4 500M-token continuation and auxiliary capability/multiple-choice post-training runs were explored but are explicitly not selected. Their raw records remain available for audit, while their active code, generated data, and checkpoint tensors are removed from the release path.

## Disclosure

AI-assisted coding and documentation tools were used while preparing this repository. The team selected the model, data mixture, experiments, and release boundary, and is responsible for verifying the measurements and complying with the source licenses. Dataset cards and unresolved license questions are listed in `docs/data/datasets.md`.

## Submission placeholders

- Demo video (2–5 minutes, English audio or subtitles): **TO BE RECORDED**.
- Team members and roles: **TO BE COMPLETED**.
- Final screenshots: see `docs/submission/assets/`; replace or supplement the generated audit screenshots with product/demo screenshots before submitting.
