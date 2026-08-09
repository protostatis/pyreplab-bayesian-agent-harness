#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SAMPLES="${SAMPLES:-20}"
SEED_BASE="${SEED_BASE:-2026}"

if command -v pyreplab-harness-outcome-model >/dev/null 2>&1; then
  PYREPLAB=(pyreplab-harness-outcome-model)
  PYTHONPATH=""
else
  PYREPLAB=("$PYTHON_BIN" -m pyreplab_harness.outcome_model)
  PYTHONPATH="${PROJECT_ROOT}/src"
fi

BASE_DIR="${PROJECT_ROOT}/.runs/adhoc-toolfailbench-cv"

AP=()
ACC=()
for fold in 1 2 3 4 5; do
  dataset="$BASE_DIR/fold-${fold}.jsonl"
  artifact="$BASE_DIR/fold-${fold}"
  tmp="$(mktemp)"
  if [ -n "${PYTHONPATH:-}" ]; then
    PYTHONPATH="$PYTHONPATH" "${PYREPLAB[@]}" evaluate "$dataset" "$artifact" --num-samples "$SAMPLES" --seed "$((SEED_BASE + fold))" >"$tmp"
  else
    "${PYREPLAB[@]}" evaluate "$dataset" "$artifact" --num-samples "$SAMPLES" --seed "$((SEED_BASE + fold))" >"$tmp"
  fi
  ap=$(jq -r '.metrics.validation.average_precision' "$tmp")
  acc=$(jq -r '.metrics.validation.accuracy_05' "$tmp")
  brier=$(jq -r '.metrics.validation.brier' "$tmp")
  echo "fold-${fold}: AP=${ap} ACC=${acc} BRIER=${brier}"
  AP+=("$ap")
  ACC+=("$acc")
  rm "$tmp"
done

ap_list="${AP[*]}"
acc_list="${ACC[*]}"
awk -v ap="$ap_list" -v acc="$acc_list" '
BEGIN {
  n=split(ap, A, " ");
  m=split(acc, B, " ");
  if (n!=5 || m!=5) {
    print "Failed to parse fold metrics";
    exit 1;
  }
  ap_sum = 0; acc_sum = 0;
  for (i=1; i<=n; i++) {
    x = A[i] + 0;
    y = B[i] + 0;
    ap_sum += x;
    acc_sum += y;
  }
  ap_mean = ap_sum / n;
  acc_mean = acc_sum / m;
  ap_var = 0; acc_var = 0;
  for (i=1; i<=n; i++) {
    x = A[i] + 0;
    y = B[i] + 0;
    ap_var += (x - ap_mean)^2;
    acc_var += (y - acc_mean)^2;
  }
  ap_sd = sqrt(ap_var / (n - 1));
  acc_sd = sqrt(acc_var / (m - 1));
  printf("AP mean=%.4f std=%.4f\n", ap_mean, ap_sd);
  printf("ACC mean=%.4f std=%.4f\n", acc_mean, acc_sd);
}
'
