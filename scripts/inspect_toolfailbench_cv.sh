#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CV_DIR="${1:-$PROJECT_ROOT/.runs/adhoc-toolfailbench-cv}"
FOLDS="${2:-5}"
EXPECTED_K="${3:-5}"
JSON="${4:-0}"

cmd=("$PYTHON_BIN" "$PROJECT_ROOT/scripts/inspect_toolfailbench_cv_splits.py" "$CV_DIR" --folds "$FOLDS" --expected-k "$EXPECTED_K")
if [ "$JSON" = "1" ]; then
  cmd+=("--json")
fi

PYTHONPATH="$PROJECT_ROOT/src" "${cmd[@]}"
