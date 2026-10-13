# Reproducing the scale-v3 Branch/Source experiment

This page separates the versioned software procedure from access to historical run artifacts. GitHub does not contain model checkpoints or latent replay-bank arrays.

## What is versioned

- The self-branch Dreamer continuation trainer and its adapters and replay implementation.
- Four portable JSON templates: 610k to 1M and 1M to 1.5M for both arms.
- The Dreamer configuration used for the 610k starting run, with only its machine-specific output path replaced.
- Small audit JSON files describing bank sequences and expected SHA256 values.
- The 100-episode-per-checkpoint score table, cleanup audit, and score-curve figure.

## Required external artifacts

To continue the historical run from the same states, provide:

1. A complete 610k source checkpoint for the first segment, agent SHA256 11f405f70c7b70fc27e5a974f64b46a4f6a1a9657ed5990db9910e6b3cdc13ac.
2. The two 1M checkpoints: Branch agent SHA256 d266b5cebd84e5f3c5c25730569ddee1d715962d4ee293700f8fea91bf14f9b4; Source agent SHA256 9b8c85fc50d5f9e8ef77b33bd033dce8ff6fd26443e79b4487694ec2c3781d59.
3. Branch and Source latent replay-bank NPZ files. Individual file hashes and expected sequence counts are in the bundled bank audits; the trainer refuses files with mismatched hashes.
4. To re-evaluate historical nodes, their complete checkpoint directories or archived model artifacts. The score CSV preserves agent hashes after weights are removed.

Checkpoint files and replay-bank arrays are deliberately excluded from Git. A future Hugging Face artifact release can provide a checksum-verified bundle without changing this repository. A clean clone can reproduce analysis, source code, protocol, and validation steps, but cannot recreate the historical numerical curve until these external artifacts are obtained. Do not describe the measurements as end-to-end reproducible from GitHub alone.

## Environment

Use Linux or WSL2, Python 3.11, and an NVIDIA GPU supported by the pinned CUDA/JAX requirements. CPU setup supports code and template tests, not model training.

~~~bash
git clone https://github.com/TwilightGlimmer/craft-scaffold.git
cd craft-scaffold
bash setup.sh cuda
./python.sh tools/doctor.py
./python.sh tools/validate_runtime.py --gpu
./python.sh -m unittest tests.test_scale_v3_protocol tests.test_curiosity_agent tests.test_curiosity_replay
~~~

The experiment used JAX 0.4.38, the vendored DreamerV3 implementation, seed 23, natural-only interaction, curiosity replay, and block resets every 2,500 retained environment actions. Hyperparameters and artifact validation rules are in the templates.

## Materialize and run

Set CRAFT_DATA_ROOT to a directory containing external artifacts at the relative paths encoded in the chosen template. Keep run outputs outside the Git checkout. The materializer checks source checkpoint completeness, source step and agent SHA256, source config SHA256, every audit SHA256, and every NPZ SHA256 before writing a resolved protocol.

~~~bash
export CRAFT_DATA_ROOT=/absolute/path/to/crafter-artifacts
./python.sh tools/materialize_scale_v3_protocol.py \
  --template configs/scale-v3/610k-to-1m-branch.template.json \
  --output "$CRAFT_DATA_ROOT/protocols/branch-610k-to-1m.json"
~~~

Choose the Source template for the matched arm. The first segment command is:

~~~bash
DEADLINE=$(python3 -c 'import time; print(int(time.time()) + 86400)')
./python.sh scripts/student_replay_train.py \
  --experimental --deadline "$DEADLINE" \
  --target 1000000 \
  --protocol "$CRAFT_DATA_ROOT/protocols/branch-610k-to-1m.json" \
  --logdir "$CRAFT_DATA_ROOT/runs/scale-v3/branch-610k-to-1m"
~~~

For the second segment, materialize the corresponding 1M-to-1.5M template with that arm's 1M checkpoint. Keep each arm's run directory and protocol fixed across resumptions. The trainer rejects protocol changes on resume; only the deadline and outer process supervision may change.

Evaluate a checkpoint with the same natural protocol using the saved run config:

~~~bash
./python.sh scripts/continuation_evaluate.py \
  --checkpoint "$CRAFT_DATA_ROOT/models/scale-v3-checkpoints/branch/step-1500000" \
  --config "$CRAFT_DATA_ROOT/models/scale-v3-checkpoints/branch/config.yaml" \
  --output "$CRAFT_DATA_ROOT/evaluations/branch-1500000" \
  --kind natural --episodes 100 --namespace 241701 --policy-seed 241703 --experimental
~~~

Use the corresponding Source paths for that arm. The evaluator writes status, summary, per-episode records, and per-action records; only a completed result whose recorded checkpoint step and agent hash match the input should be added to the score table. The all-kind capability suite can be requested with --kind all, but those grouped task runs are not the natural-score curve.

The published evaluation used 100 real natural episodes per checkpoint, environment namespace 241701, and policy seed 241703. Each score is an estimate over 100 episodes, not a separate training seed.

## Limits

Historical 610k and 1M checkpoints, NPZ banks, raw episode/action logs, and per-run release locks are external or archival artifacts. This keeps weights out of Git, but a clean clone cannot reproduce the exact historical training run today. If artifacts are later published on Hugging Face, add immutable revision IDs and checksums here before claiming end-to-end reproducibility.
