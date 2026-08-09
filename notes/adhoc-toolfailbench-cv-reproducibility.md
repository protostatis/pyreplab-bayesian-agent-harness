# Reproducibility: ad-hoc SoHarshh/ToolFailBench CV run

## Interrogating results with pyreplab

Use the `outcome_model` CLI entry point (pure-`pyreplab` path) to print per-split
and per-policy metrics, including `average_precision`.

```bash
PYTHONPATH=src python3 -m pyreplab_harness.outcome_model evaluate \
  .runs/adhoc-toolfailbench-dataset.jsonl \
  .runs/adhoc-toolfailbench-model --num-samples 50 --seed 99
```

For CV split safety checks (no task appears in both train+validation within a fold, and validation fold assignments are a 5-way partition):

```bash
cd /path/to/pyreplab-bayesian-agent-harness
./scripts/inspect_toolfailbench_cv.sh .runs/adhoc-toolfailbench-cv 5 5 1
```

For Bayesian posterior diagnostics and posterior/prior predictive checks:

```bash
cd /path/to/pyreplab-bayesian-agent-harness
./scripts/inspect_bayesian_fit.sh .runs/adhoc-toolfailbench-model \
  --dataset .runs/adhoc-toolfailbench-dataset.jsonl \
  --posterior-samples 64 --prior-samples 64 --max-rows 400
```

For 5-fold task-level CV artifacts (datasets at `.runs/adhoc-toolfailbench-cv/fold-*.jsonl`
and artifacts at `.runs/adhoc-toolfailbench-cv/fold-*`):

```bash
PYTHONPATH=src python3 - <<'PY'
import json
from pathlib import Path
from pyreplab_harness import outcome_model as om

root = Path('.runs/adhoc-toolfailbench-cv')
rows = []
for i in range(1, 6):
    out = om.evaluate_model(root / f'fold-{i}.jsonl', root / f'fold-{i}')
    rows.append(out['metrics']['validation'])

for idx, m in enumerate(rows, start=1):
    print(f"fold-{idx}: n={m['n']} ap={m['average_precision']:.4f} acc={m['accuracy_05']:.4f} brier={m['brier']:.4f}")

for key in ('accuracy_05', 'average_precision', 'precision', 'recall', 'f1', 'brier'):
    vals = [m[key] for m in rows]
    mu = sum(vals) / len(vals)
    sd = (sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
    print(f"{key}: {mu:.4f} ± {sd:.4f}")
PY
```
The CV results were generated from an ad-hoc script (not yet formalized into a
single CLI) using `python3` directly. For strict reproducibility, rerun inside a
fresh `uv` environment with pinned versions.

## One-time environment setup

```bash
cd /path/to/pyreplab-bayesian-agent-harness
uv venv .venv --python 3.9
source .venv/bin/activate
uv pip install --upgrade pip setuptools wheel
uv pip install -r requirements-cv.txt
PYTHONPATH=src
export PYTHONPATH=src
```

## Reproducible runtime config used

- Python: 3.9.6
- NumPy: 2.0.2
- PyTorch: 2.8.0 (CPU)
- scikit-learn: 1.6.1
- requests: 2.32.5
- Data file: `.runs/adhoc-toolfailbench-dataset.jsonl`
  - SHA-256: `13250b6c8ccbfe32678eaf33a9f60ca70f3784471d327b84d156222ec9e6579b`

## Command-level reproducibility checklist

1. Use the dataset file exactly as built for this run (or rerun your own
   conversion script before this step).
2. Keep task-level 5-fold split (by sorted `task_id` index modulo 5).
3. Fix seeds:
   - fold seed base `2026` for training/eval calls,
   - all train runs use `epochs=7`, `batch-size=128`, `patience=3`, `num-samples=20`
   - split-level seed offsets as shown in the original script.
4. Pin all randomizers (Python/Torch via model defaults already set by
   `outcome_model.train_model`).

## Quick check commands

```bash
# show help/compatibility in env
PYTHONPATH=src python -m pyreplab_harness.outcome_model --help

# smoke sanity check (single dataset row inference)
PYTHONPATH=src python -m pyreplab_harness.outcome_model evaluate \
  .runs/adhoc-toolfailbench-dataset.jsonl \
  .runs/adhoc-toolfailbench-model
```

## Recommended hardening

- Use `scripts/run_toolfailbench_cv.py` for exact run replay:
  `python3 scripts/run_toolfailbench_cv.py --dataset .runs/adhoc-toolfailbench-dataset.jsonl --cv-dir .runs/adhoc-toolfailbench-cv --folds 5`.
- Save `requirements-cv.txt`, an environment lock file, and the dataset hash next to
  the run artifacts for future audits.
