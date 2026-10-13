# Crafter self-branch scale-v3 evaluation curve: 600k–1.5M

![100-episode natural score curve](score-curve.svg)

## Evaluation protocol

Every plotted point uses 100 real natural episodes, namespace 241701, policy seed 241703, and the original scale-v3 evaluation protocol. Each entry was validated against checkpoint step and agent SHA-256, completed status, 100 episode records, action log hash, and summary hash. Full episode and action records remain in the remote project outputs; paths are listed in scores.csv.

The 600k point is the shared baseline checkpoint (bd026df853ee976d780bec13824dc4a21aeaed587673aa3170735d9a128b4443), evaluated using the same 100-episode protocol. Both arms were initialized from the same 610k checkpoint, but there is no standardized evaluation at 610k; the plotted 600k point is a baseline reference, not the measured initialization result.

There is a coverage gap from 600k to the first surviving scale-v3 checkpoint: branch 822,987, source 826,855. The plot does not interpolate over this gap. Previously deleted 610k–790k weights cannot be re-evaluated.

## Scores

| Point | Branch score | Source score |
|---|---:|---:|
| Shared baseline 600k | 11.87 | 11.87 |
| 1.0M | 19.14 | 17.99 |
| 1.5M | 23.66 | 17.83 |

Best measured surviving points: branch **24.94** at 1,300,000; source **20.22** at 1,370,000. Local fluctuations should be read cautiously because each score is an estimate over 100 natural episodes.

Selected achievement rates (% of episodes):

| Point | Collect iron | Collect coal | Make stone pickaxe | Make iron pickaxe | Collect diamond |
|---|---:|---:|---:|---:|---:|
| Shared baseline 600k | 0.0 | 14.0 | 0.0 | 0.0 | 0.0 |
| Branch 1M | 1.0 | 23.0 | 34.0 | 0.0 | 0.0 |
| Source 1M | 4.0 | 27.0 | 28.0 | 0.0 | 0.0 |
| Branch 1.5M | 10.0 | 34.0 | 56.0 | 0.0 | 0.0 |
| Source 1.5M | 1.0 | 20.0 | 40.0 | 0.0 | 0.0 |

## Historical evaluation and checkpoint cleanup

The surviving pre-1M audit covers **31 unique checkpoints** (16 branch, 15 source), all evaluated on 100 natural episodes with verified records. The resumed cleanup manifest records 16 remaining path deletions and 6 retained canonical copies. An earlier cleanup had already removed 33 duplicate rolling/eval-pending paths; these match prior retention/checkpoint-cleanup logs and the preserved pre-delete manifest. No evaluation output, episode/action record, training log, or ledger row was deleted.

The retained pre-1M targets are the nearest 800k, 900k, and 1M checkpoints per arm, plus any better global-best checkpoint. For 1M–1.5M, exact 100k milestones, the best node (branch 1.3M, source 1.37M), and the official final 1.5M run checkpoints are retained. A final dedup pass removed 10 evaluated duplicate copies and preserved the SHA-verified canonical files. Machine-readable details are in cleanup-audit.json; full path/hash manifests remain in the report root.

## Files

- scores.csv — checkpoint scores, selected skill rates, hashes, seeds, and relative evaluation output paths.
- score-curve.svg — the figure.
- cleanup-audit.json — checkpoint cleanup and prior deletion reconciliation.
