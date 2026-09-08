# J3 data disclosure

This document describes the data that actually entered the selected Stage 3 training artifact. It distinguishes source intent from materialized token counts and does not claim rights beyond the source providers' terms.

## Selected training mixture

The runtime manifest is [`stage3-mixed-manifest.json`](stage3-mixed-manifest.json). The assembled, pre-mix manifest is [`stage3-assembled-manifest.json`](stage3-assembled-manifest.json). The corresponding verification files record the observed token totals, ID range, missing-shard checks, and tokenizer hash.

| Materialized source | Category | Train tokens | Validation tokens | Selection/configuration |
| --- | --- | ---: | ---: | --- |
| FinePDF local artifact | PDF/textbook-like documents | 299,793,532 | 214,382 | `finepdfs_300m_tokenized`; first complete document after the 300M-token target, lexicographic part order and Parquet row order |
| FineWeb-Edu | clean web/educational text | 350,206,468 | 3,502,065 | `HuggingFaceFW/fineweb-edu`, `sample-10BT`, English, 350,206,468-token quota |
| Ultra-FineWeb L2 normal web | ordinary web | 125,000,000 | included in assembled validation accounting | `openbmb/Ultra-FineWeb`, English, score `0.55 <= score < 0.80` |
| Cosmopedia V2 stories | narrative text | 75,000,000 | included in assembled validation accounting | `HuggingFaceTB/smollm-corpus`, `cosmopedia-v2`, formats `story`, `story_forums`, `story_reddit`, `story_life_lessons` |
| Cosmopedia V2 WikiHow | procedural text | 50,000,000 | included in assembled validation accounting | `HuggingFaceTB/smollm-corpus`, `cosmopedia-v2`, format `wikihow` |
| WikiText-103 | encyclopedic text | 100,000,000 | included in assembled validation accounting | `Salesforce/wikitext`, `wikitext-103-v1`, train source |
| **Total** |  | **1,000,000,000** | **7,216,447** | exact train target after materialization |

The selected stream is not the earlier broad Stage 1 configuration. In particular, DCLM-Edu, peS2o, Common Corpus, TinyStories, Ultra-FineWeb elevated/high bands, and the unused Stage 1 Cosmopedia quota are not part of the selected Stage 3 artifact. The Stage 2 manifest contains more sources than the selected Stage 3 slice; only the explicitly listed Stage 2 source names were copied into Stage 3.

## Processing and audit

- Text cleanup used by the tokenizer pipeline: NFKC normalization, removal of `Cc`/`Cf`/`Co` characters, and whitespace collapsing.
- Tokenizer: byte-level BPE, vocabulary 16,384, four special tokens, tokenizer SHA-256 `416267157bccdfa2bc6d5025aebf3b9a6a82271f4664f04f0d963337fec35023`.
- Token IDs in the verified selected artifact are in the observed range `3..16383`; the verification artifact reports no missing listed binary shards.
- Stage 3 assembly enables deduplication and uses the public-benchmark index. The decontamination index contains HellaSwag (59,950 rows), PIQA (21,035), ARC (7,787), and WinoGrande (43,432). A candidate is rejected when it contains at least two exact normalized word shingles of width 12 from the same benchmark.
- The final assembly is globally chunk-shuffled before training: 8,192-token chunks, seed `1337`, 122,073 train chunks, and 766 source switches in the first 1,024 chunks. Runtime shard shuffling is an additional deterministic permutation.
- The validation stream is retained in the same manifest and is not used as train tokens. The model's reported Stage 3 validation perplexity is the training-loop validation metric; it is not the competition's held-out WikiText-103 result.

## Provenance and license notes

The public source cards used for the named sources are:

- [FineWeb-Edu dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) — the card states ODC-By for the dataset; underlying web documents may carry their own terms.
- [Ultra-FineWeb dataset card](https://huggingface.co/datasets/openbmb/Ultra-FineWeb) — the card describes the dataset and its Apache-2.0 repository license; that does not automatically relicense every crawled source document.
- [SmolLM Cosmopedia V2 card](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/tree/main/cosmopedia-v2) — the card states ODC-By for the dataset.
- [WikiText dataset card](https://huggingface.co/datasets/Salesforce/wikitext) — the card identifies the WikiText licensing information; verify the exact version and redistribution terms before publishing derived artifacts.

The FinePDF input is a local `finepdfs-world/clean` artifact. The materialized manifest preserves the source-file byte counts and SHA-256 values, but the upstream license/permission record is not present in this repository. We therefore disclose it as a used source, do not redistribute its raw files here, and leave redistribution as an explicit legal verification item.

The repository also preserves the tokenizer and manifests as audit artifacts. These are not a substitute for the licenses of the underlying source documents.
