# Outcome Model Descriptor Learn-Smoke

> Status: synthetic-only, treatment-held-out mechanism probe. It never
> establishes real-agent effectiveness, calibration, or deployment readiness.

## What It Tests

The probe creates noisy Bernoulli outcomes from a known synthetic policy grammar
and asks whether the Bayesian outcome model can rank unseen treatment bundles.

- A deterministic coverage-aware split trains on 26 grammar bundles and holds
  out 10 bundles.
- `policy_id`, `policy_version`, and descriptor `bundle_id` are replaced with a
  constant placeholder in every model input, preventing identity lookup.
- Every held-out grammar factor level is represented in training descriptors.
- The task-side `task_variant` is visible before action and modulates grammar
  factors, so optimal policies can differ across tasks.
- Evaluation uses held-out tasks and treatments, a predeclared static held-out
  baseline, random allocation, known synthetic `true_p`, aggregate rank
  recovery, and a fixed representative ranking panel.

The model never receives `true_p`; it is available only to score this synthetic
diagnostic after prediction.

## Canonical Run

| Field | Value |
| --- | --- |
| Registry | Full 36-cell grammar, seed `20260809` |
| Registry hash | `1d26bdbd9bb77a9b1f70391c5ccc7fa84f3159278ea127aeaee401c327a95bac` |
| Train / held-out treatments | 26 / 10 |
| Rows | 4,288: train 2,912, validation 936, test 440 |
| Test tasks | 44 complete held-out panels |
| Data / train seeds | `20260809` / `42` |
| Training | 80 configured epochs; stopped at 16, best epoch 6 |

## Result

| Measure | Result |
| --- | --- |
| Test Brier, model / train-rate baseline | `0.2453` / `0.2537` |
| Expected allocation, model / random | `0.4757` / `0.4476` |
| Expected lift over random | `+0.0282` |
| Expected lift over predeclared held-out baseline | `+0.0532` |
| Held-out Spearman rank recovery | `0.2385` |
| Predeclared rank threshold | `0.3` |
| Descriptor-only top-1 match | `1 / 12` |
| Final verdict | `descriptor_learned_something_usable: false` |

The model has partial synthetic signal: it improves Brier and expected allocation
over random. It does not reliably recover the held-out policy ordering, so the
predeclared gate correctly remains a non-pass.

## Representative Ranking

The fixed lexicographically selected test task was
`synthetic-artifact-easy-t17`. Its model-highest and model-lowest held-out
treatments happened to match the synthetic oracle ranks:

| Model rank | Policy | Predicted success | Synthetic `true_p` | Oracle rank |
| --- | --- | ---: | ---: | ---: |
| 1 | `deliberate-final-single-pass-generous` | `0.507685` | `0.664123` | 1 |
| 10 | `decompose-incremental-retry-on-failure-tight` | `0.507281` | `0.400594` | 10 |

That isolated agreement is not enough to overturn the aggregate result. The
predicted range is only `0.000405`, far smaller than the approximate posterior
standard deviation of `0.017` for those policies, and the broader held-out
ranking/top-1 tests fail.

## Reproduce

```bash
PYTHONPATH=src .venv/bin/python -m pyreplab_harness.treatments generate \
  .runs/learn-smoke-treatments.json --count 36 --seed 20260809

PYTHONPATH=src .venv/bin/python -m pyreplab_harness.outcome_model_learn_smoke \
  .runs/learn-smoke-treatments.json \
  --output-dir .runs/learn-smoke \
  --data-seed 20260809 --train-seed 42
```

The command exits `0` only if its predeclared descriptor-held-out gate passes;
an exit code of `2` is an informative non-pass. Retained datasets and artifacts
belong under ignored `.runs/` and must never be merged into a real experiment
dataset.
