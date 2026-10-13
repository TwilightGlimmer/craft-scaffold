# Crafter self-branch scale-v3 evaluation curve: 600k–1.5M

![100-episode natural score curve](score-curve.svg)

## Evaluation protocol

Every plotted point is a completed evaluation of 100 real natural episodes with namespace 241701 and policy seed 241703. The CSV records checkpoint step, score, selected achievement rates, agent SHA-256, seeds, evaluation target, and output location. Evaluations were validated against checkpoint step/hash, completed status, episode count, action rows, and summary hashes. Training rewards are not evaluation scores.

The 600k point is the measured shared baseline (11.87). Both routes start from the same complete 610k checkpoint (12.41). The Branch and Source arms use the same DreamerV3/Crafter training and curiosity-replay setup; their training replay banks are separately collected and audited. The policy receives no expert labels or privileged state.

## Checkpoint coverage

- 610k–800k: both routes have verified natural evaluations every 10k actions, with 100 episodes per point. The 620k–800k rows were cross-checked against the continuation and gap-fill ledgers; checkpoint hashes and scores agree.
- 810k–900k: each route was continued from its own verified 800k checkpoint and evaluated every 10k actions for 100 natural episodes. The 900k results are Branch 19.319 and Source 16.514.
- 800k–1M: route-specific gap-fill covers 810k–900k; surviving historical checkpoints elsewhere in this interval have irregular actual steps. The figure plots measured steps and does not invent missing points.
- 1M–1.5M: both routes have verified natural evaluations at 10k-step intervals through the 1.5M endpoint.

The curve has one training seed per route. It documents this paired engineering run and does not establish multi-seed statistical reliability. The chart omits the irregular pre-1M audit markers to keep the main trajectories readable; those verified rows remain in the CSV. A line between 900k and 1.0M connects two measured endpoints across a 100k interval and does not imply intermediate evaluations.

## Scores

| Checkpoint | Branch | Source |
|---|---:|---:|
| Shared baseline 600k | 11.87 | 11.87 |
| Shared initialization 610k | 12.41 | 12.41 |
| 700k | 14.04 | 14.27 |
| 800k | 19.52 | 12.17 |
| 900k | 19.32 | 16.51 |
| 1.0M | 19.14 | 17.99 |
| 1.5M | 23.66 | 17.83 |

Best measured points: Branch 24.94 at 1,300,000; Source 20.22 at 1,370,000. Scores can fluctuate between checkpoints, so interpret them as estimates over 100 natural episodes.

### Selected high-level achievements

| Point | Route | Collect iron | Collect coal | Stone pickaxe | Iron pickaxe | Diamond |
|---|---|---:|---:|---:|---:|---:|
| 800k | Branch | 4% | 22% | 27% | 0% | 0% |
| 800k | Source | 0% | 18% | 1% | 0% | 0% |
| 1.5M | Branch | 10% | 34% | 56% | 0% | 0% |
| 1.5M | Source | 1% | 20% | 40% | 0% | 0% |

Stone-pickaxe progress is visible, while iron-pickaxe and diamond collection remained at zero at these checkpoints.

## Data and regeneration

- scores.csv — 191 verified rows, including shared start references and all evaluated route checkpoints.
- score-curve.svg — plotted score curve.
- tools/plot_scale_v3_curve.py — regenerates the SVG from the CSV using only the Python standard library.
- cleanup-audit.json — checkpoint retention and cleanup audit. Full episode/action records and checkpoint manifests remain in remote experiment reports; model weights are not included in GitHub.

Regenerate the plot from the repository root:

~~~bash
python tools/plot_scale_v3_curve.py
~~~
