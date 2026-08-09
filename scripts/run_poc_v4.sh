#!/usr/bin/env bash
set -euo pipefail

# Stratified 24-pair proof-of-concept gate for policy v4.
#
# Each family contributes six tasks: three train, one validation and two test
# tasks under dataset.task_split.  Python-repair seeds additionally cover all
# three implemented templates.  Cells are interleaved to reduce correlation
# between family and shared-model/cron load.  The batch runner's default resume
# behavior makes this script safe to restart with the same output and run root.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="${1:-$PROJECT_ROOT/.runs/poc-v4.jsonl}"
HOST="${HOST:-${PYREPLAB_HARNESS_HOST:-ubuntu-local}}"
REMOTE_PROJECT="${REMOTE_PROJECT:-${PYREPLAB_REMOTE_PROJECT:-}}"
REMOTE_ROOT="${REMOTE_ROOT:-${PYREPLAB_REMOTE_RUN_ROOT:-}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PI_BIN="${PYREPLAB_PI:-pi}"
PI_PROVIDER="${PYREPLAB_PI_PROVIDER:-ubuntu-gemma}"
PI_MODEL="${PYREPLAB_PI_MODEL:-gemma-4-26b-a4b}"
PI_THINKING="${PYREPLAB_PI_THINKING:-off}"
MODEL_SWITCH_EXTENSION="${PYREPLAB_MODEL_SWITCH_EXTENSION:-}"

if [[ -z "$REMOTE_PROJECT" || "$REMOTE_PROJECT" != /* || "$REMOTE_PROJECT" == "/" ]]; then
  echo "error: set REMOTE_PROJECT or PYREPLAB_REMOTE_PROJECT to an absolute remote path other than '/'" >&2
  exit 2
fi
if [[ -z "$REMOTE_ROOT" ]]; then
  REMOTE_ROOT="$REMOTE_PROJECT/.runs/poc-v4"
fi
if [[ "$REMOTE_ROOT" != /* || "$REMOTE_ROOT" == "/" ]]; then
  echo "error: set REMOTE_ROOT or PYREPLAB_REMOTE_RUN_ROOT to an absolute remote path other than '/'" >&2
  exit 2
fi

run_cell() {
  local family="$1"
  local difficulty="$2"
  local seeds="$3"

  local optional_args=()
  if [[ -n "$MODEL_SWITCH_EXTENSION" ]]; then
    optional_args+=(--model-switch-extension "$MODEL_SWITCH_EXTENSION")
  fi

  PYTHONPATH="$PROJECT_ROOT/src" "$PYTHON_BIN" -m pyreplab_harness.batch \
    --host "$HOST" \
    --remote-project "$REMOTE_PROJECT" \
    --policy-version 4 \
    --remote-run-root "$REMOTE_ROOT" \
    --pi "$PI_BIN" \
    --provider "$PI_PROVIDER" \
    --model "$PI_MODEL" \
    --thinking "$PI_THINKING" \
    --families "$family" \
    --difficulties "$difficulty" \
    --seeds "$seeds" \
    --output "$OUTPUT" \
    "${optional_args[@]}"
}

run_cell artifact easy "2,3"
run_cell sqlite hard "3,8"
run_cell shell medium "2,3"
run_cell python_repair easy "1,2"
run_cell sqlite easy "1,12"
run_cell artifact hard "5,30"
run_cell python_repair medium "7,11"
run_cell shell easy "1,4"
run_cell artifact medium "4,1"
run_cell sqlite medium "2,6"
run_cell shell hard "5,22"
run_cell python_repair hard "5,6"
