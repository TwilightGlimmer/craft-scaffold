#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-cpu}"
case "$MODE" in cpu|cuda) ;; *) echo "Usage: bash setup.sh cpu|cuda"; exit 2;; esac
export CRAFT_DATA_ROOT="${CRAFT_DATA_ROOT:-$ROOT/.artifacts}"
export TMPDIR="$CRAFT_DATA_ROOT/tmp" PIP_CACHE_DIR="$CRAFT_DATA_ROOT/cache/pip"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
BOOT="${CRAFT_BOOTSTRAP_PYTHON:-python3.11}"
"$BOOT" -c 'import sys; assert sys.version_info[:2] == (3,11), "Python 3.11 required"'
if [[ ! -d "$ROOT/.venv" ]]; then "$BOOT" -m venv "$ROOT/.venv"; fi
"$ROOT/.venv/bin/python" -m pip install -r "$ROOT/requirements/$MODE.txt"
"$ROOT/.venv/bin/python" -m pip check
echo "Environment installed. Run ./python.sh tools/doctor.py"
