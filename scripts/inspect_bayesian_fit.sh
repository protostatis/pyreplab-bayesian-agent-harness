#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

PYTHONPATH="$PROJECT_ROOT/src" "$PYTHON_BIN" "$PROJECT_ROOT/scripts/inspect_bayesian_fit.py" "$@"
