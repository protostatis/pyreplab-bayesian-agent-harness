#!/usr/bin/env bash
set -euo pipefail

HOST="${1:-${PYREPLAB_HARNESS_HOST:-ubuntu-local}}"
REMOTE_PROJECT="${2:-${PYREPLAB_REMOTE_PROJECT:-}}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "$REMOTE_PROJECT" || "$REMOTE_PROJECT" != /* || "$REMOTE_PROJECT" == "/" ]]; then
  echo "error: pass an absolute remote project path other than '/' as argument 2 or PYREPLAB_REMOTE_PROJECT" >&2
  exit 2
fi

ssh -o BatchMode=yes "$HOST" "mkdir -p $(printf '%q' "$REMOTE_PROJECT")"
rsync -az \
  --exclude '.git/' \
  --exclude '.env' \
  --exclude '.env.*' \
  --exclude '.pi/' \
  --exclude '.runs/' \
  --exclude 'runs/' \
  --exclude '.venv/' \
  --exclude '*.egg-info/' \
  --exclude '__pycache__/' \
  --exclude 'notes/local/' \
  --exclude 'notes/opencode-session-outcome-training-history.md' \
  --exclude '.DS_Store' \
  "$PROJECT_ROOT/" "$HOST:$REMOTE_PROJECT/"

ssh -o BatchMode=yes "$HOST" \
  "cd $(printf '%q' "$REMOTE_PROJECT") && { PYTHON_BIN=.venv/bin/python; [ -x \"\$PYTHON_BIN\" ] || PYTHON_BIN=python3; PYTHONPATH=src \"\$PYTHON_BIN\" -m unittest discover -s tests -v; }"
