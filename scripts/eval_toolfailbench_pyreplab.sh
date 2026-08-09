#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SAMPLES="${SAMPLES:-50}"
SEED="${SEED:-99}"

if command -v pyreplab-harness-outcome-model >/dev/null 2>&1; then
  PYREPLAB=(pyreplab-harness-outcome-model)
  PYTHONPATH=""
else
  PYREPLAB=("$PYTHON_BIN" -m pyreplab_harness.outcome_model)
  PYTHONPATH="${PROJECT_ROOT}/src"
fi

DATASET="${1:-$PROJECT_ROOT/.runs/adhoc-toolfailbench-dataset.jsonl}"
ARTIFACT="${2:-$PROJECT_ROOT/.runs/adhoc-toolfailbench-model}"

PYTHONPATH="$PYTHONPATH" ${PYREPLAB[@]} evaluate "$DATASET" "$ARTIFACT" --num-samples "$SAMPLES" --seed "$SEED"
