#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$ROOT/.env.local" ]]; then
  set -a
  source "$ROOT/.env.local"
  set +a
fi
export CRAFT_DATA_ROOT="${CRAFT_DATA_ROOT:-$ROOT/.artifacts}"
export CRAFT_RUN="${CRAFT_RUN:-portable-v2}" CRAFT_GPU="${CRAFT_GPU:-0}" CRAFT_THREADS="${CRAFT_THREADS:-2}"
export TMPDIR="$CRAFT_DATA_ROOT/tmp" TMP="$CRAFT_DATA_ROOT/tmp" TEMP="$CRAFT_DATA_ROOT/tmp"
export XDG_CACHE_HOME="$CRAFT_DATA_ROOT/cache" XDG_CONFIG_HOME="$CRAFT_DATA_ROOT/config"
export PIP_CACHE_DIR="$CRAFT_DATA_ROOT/cache/pip" HF_HOME="$CRAFT_DATA_ROOT/cache/huggingface"
export TORCH_HOME="$CRAFT_DATA_ROOT/cache/torch" CUDA_CACHE_PATH="$CRAFT_DATA_ROOT/cache/cuda"
export MPLCONFIGDIR="$CRAFT_DATA_ROOT/cache/matplotlib" NUMBA_CACHE_DIR="$CRAFT_DATA_ROOT/cache/numba"
export JAX_COMPILATION_CACHE_DIR="$CRAFT_DATA_ROOT/cache/jax"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$CRAFT_GPU"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="$CRAFT_THREADS" MKL_NUM_THREADS="$CRAFT_THREADS" OPENBLAS_NUM_THREADS="$CRAFT_THREADS"
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME"
PY="${CRAFT_PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then echo "Python missing: $PY. Run bash setup.sh cpu|cuda, or set CRAFT_PYTHON." >&2; exit 2; fi
cd "$ROOT"
exec "$PY" -B "$@"
