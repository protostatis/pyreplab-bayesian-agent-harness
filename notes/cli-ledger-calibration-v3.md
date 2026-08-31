# CLI Ledger Calibration v3

**Date:** 2026-08-30
**Purpose:** local-only calibration of the Bayesian outcome engine on the
completed CLI evaluation ledger. No model provider was called.

## Data boundary

The dataset contains 296 completed, verified CLI attempts:

- 188 records from the coding-smoke evaluation rounds;
- 48 records from the earlier crash-novel collection; and
- 60 records from bounded-generalization-v1.

The 29 records from the budget-cancelled bounded-generalization-v2 run were
excluded. The v1 records are also not clean confirmatory evidence because six
of their ten task IDs had older target-model records; they are retained only as
an exploratory holdout.

The model uses only pre-action fields: task prompt, broad task family, prompt
length, family index, model/profile identity, and prompt fingerprint. It does
not use token usage, latency, tool calls, engine errors, or verifier evidence.

## Split and result

The completed ledger was split by task group into 196 training rows, 40
validation rows, and 60 exploratory holdout rows. Training and early stopping
used only the first two splits.

| Split | Rows | Brier score | Expected calibration error | What it means |
|---|---:|---:|---:|---|
| Training | 196 | 0.091 | 0.076 | Fit quality on the rows used to learn the model. |
| Validation | 40 | 0.103 | 0.113 | Calibration on held-out historical task groups used for model selection. |
| Exploratory v1 holdout | 60 | 0.075 | 0.050 | An exploratory check on the completed v1 collection, not a clean confirmation. |

On the exploratory holdout, the engine classified 56 of 60 attempts correctly
at a 0.5 threshold. Its overall expected calibration error was 0.050, meaning
the predicted probabilities differed from observed success frequencies by about
five percentage points on average across probability bins. This encouraging
number must not be read as evidence for a model ranking: the holdout itself is
contaminated by the v1 task-history issue and the suite has a high success
ceiling.

Per-profile exploratory holdout expected calibration errors were 0.036 for
Nemotron, 0.086 for GPT-4o-mini, and 0.182 for Qwen3 Coder 30B. Each profile
has only 20 holdout rows, so these values are diagnostic rather than stable
performance estimates.

## Decision

The engine is technically runnable on the accumulated CLI ledger and produces
uncertainty-aware predictions. The data are still too narrow for routing or a
general model ranking. Keep the model in analysis/reporting mode, add no new
provider calls while the account balance is about $1, and use mock transports
for the next policy prototype.

The generated dataset and artifact are in the ignored local path
`.runs/cli_ledger_validation_v3/`. The dataset SHA-256 is
`6f79602fc059d9375a4d8e0b1e809a06393d83a4aa6eba50608dfbe8753c6f1c`.
