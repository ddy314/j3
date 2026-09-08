# Non-selected experiments

These experiments are disclosed so that the repository does not present a selective history. They are not part of the J3 release candidate.

| Experiment | Observed outcome | Release decision |
| --- | --- | --- |
| Stage 4 / final 500M candidate, `20260907-112712` | Completed 500,039,680 tokens on a different high-information mixture; final recorded validation perplexity about 19.3842 | Not selected. It was a different continuation/data protocol, so its number is not a controlled comparison against Stage 3. The run record is retained under `docs/training/raw-runs/20260907-112712`. |
| Capability CPT, `capability-cpt-125m` | Completed 125,042,688 tokens | Not selected; capability-CPT implementation and data were removed from the active project surface. |
| Multiple-choice post-training, `contrastive-mc-posttrain` | Failed with an out-of-memory condition | Not selected; failure log and metadata are retained. |
| Multiple-choice post-training v2, `contrastive-mc-posttrain-v2` | Completed only a short 589,824-token run | Not selected; not a competition-ready finalization. |
| Multiple-choice post-training v3, `contrastive-mc-posttrain-v3` | Completed 50,003,968 tokens | Not selected; auxiliary post-training was removed from the public release path. |

The user's final selection is therefore Stage 3 1B, not the later Stage 4/CPT/multiple-choice branches. Removing their active code, local data, and checkpoint tensors reduces ambiguity without deleting the raw audit records.
