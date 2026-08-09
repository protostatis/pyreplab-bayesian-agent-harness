#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BATCH_OUTPUT="${1:-$PROJECT_ROOT/.runs/poc-v4.jsonl}"
GATE_OUTPUT="${GATE_OUTPUT:-$PROJECT_ROOT/.runs/poc-v4-gate.json}"
HOST="${HOST:-${PYREPLAB_HARNESS_HOST:-ubuntu-local}}"
REMOTE_PROJECT="${REMOTE_PROJECT:-${PYREPLAB_REMOTE_PROJECT:-}}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-${PYREPLAB_REMOTE_RUN_ROOT:-}}"

if [[ -z "$REMOTE_PROJECT" || "$REMOTE_PROJECT" != /* || "$REMOTE_PROJECT" == "/" ]]; then
  echo "error: set REMOTE_PROJECT or PYREPLAB_REMOTE_PROJECT to an absolute remote path other than '/'" >&2
  exit 2
fi
if [[ -z "$REMOTE_RUN_ROOT" ]]; then
  REMOTE_RUN_ROOT="$REMOTE_PROJECT/.runs/poc-v4"
fi
if [[ "$REMOTE_RUN_ROOT" != /* || "$REMOTE_RUN_ROOT" == "/" ]]; then
  echo "error: set REMOTE_RUN_ROOT or PYREPLAB_REMOTE_RUN_ROOT to an absolute remote path other than '/'" >&2
  exit 2
fi

REMOTE_PROJECT_Q="$(printf '%q' "$REMOTE_PROJECT")"
REMOTE_RUN_ROOT_Q="$(printf '%q' "$REMOTE_RUN_ROOT")"

# Exit 2 without touching model artifacts if completion or diversity gates fail.
PYTHONPATH="$PROJECT_ROOT/src" python3 -m pyreplab_harness.poc_gate \
  "$BATCH_OUTPUT" \
  --expected-jobs 24 \
  --policy-version 4 \
  --min-disagreement 0.15 \
  --output "$GATE_OUTPUT"

# Training remains on CPU and never unloads or replaces the shared Gemma model.
ssh -T -o BatchMode=yes -o ConnectTimeout=10 "$HOST" "
  set -eu
  cd $REMOTE_PROJECT_Q
  PYTHONPATH=src python3 -m pyreplab_harness.dataset \
    $REMOTE_RUN_ROOT_Q $REMOTE_RUN_ROOT_Q/dataset.jsonl
  PYTHONPATH=src .venv/bin/python -m pyreplab_harness.outcome_model train \
    $REMOTE_RUN_ROOT_Q/dataset.jsonl $REMOTE_RUN_ROOT_Q/model --device cpu
  PYTHONPATH=src .venv/bin/python -m pyreplab_harness.allocator_eval \
    $REMOTE_RUN_ROOT_Q/dataset.jsonl $REMOTE_RUN_ROOT_Q/model \
    $REMOTE_RUN_ROOT_Q/allocator-test.json --split test
  PYTHONPATH=src python3 -m pyreplab_harness.dashboard \
    $REMOTE_RUN_ROOT_Q/dataset.jsonl $REMOTE_RUN_ROOT_Q/dashboard.html \
    --metrics $REMOTE_RUN_ROOT_Q/model/metrics.json \
    --baselines $REMOTE_RUN_ROOT_Q/allocator-test.json \
    --title 'Terminal Gym v4 Proof of Concept'
"
