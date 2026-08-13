# O1 Offline Architecture Experiment - Result

Date: 2026-08-11

## 1. Experiment summary

The O1 SWE-bench Verified paired releases were imported, the 455-pair outcome
matrix was reproduced, and two neural configurations plus a diagnostic TF-IDF
model were evaluated as closed-set allocators. **Every tested selector tied
always-native.** The tested representations found no task-dependent allocation
signal; this does not establish that no richer representation could do so.

## 2. Data

Projected from the two HuggingFace CC-BY-4.0 releases using
`dataset_viewer.parquet`:

| Source | Rows | Shared IDs | Complete pairs |
|---|---:|---:|---:|
| Baseline (text protocol) | 495 | 495 | 455 |
| Native (tool interface) | 500 | | |

The 40 excluded tasks have null `resolved` in at least one arm.

Reproduced 2x2 outcome matrix (exact match to prior audit):

| | Native pass | Native fail | Total |
|---|---:|---:|---:|
| Baseline pass | 110 | 34 | 144 |
| Baseline fail | 96 | 215 | 311 |
| Total | 206 | 249 | 455 |

- Baseline rate: 31.6%
- Native rate: 45.3%
- Paired oracle: 52.7% (240/455)
- Maximum observed allocation headroom: 7.5 percentage points (34 tasks)

## 3. Models trained

### Neural outcome model (harness-native)

Architecture: bag-of-words text embedding + categorical embeddings +
treatment descriptor fusion + variational Bayesian linear head.

Two configurations:
- v1: 5000 vocab, 512 max tokens, 64-dim text, 64-dim fusion, dropout 0.3
- v2: 8000 vocab, 1024 max tokens, 128-dim text, 128-dim fusion, dropout 0.1

Both produced all probabilities below 0.5 at the classification threshold. That
threshold behavior is not itself an allocator result. The relevant result is
that the posterior mean for native was higher on every evaluated task, so
argmax selected native for 100% of test tasks in both configurations.

### Logistic regression with interaction features (scikit-learn)

TF-IDF (500 features, 1-2 grams) + repo one-hot + problem-statement length +
explicit treatment-x-feature interactions. This was an ad hoc diagnostic, not
the predeclared nested evaluation: the unsupervised feature vocabulary and
scaling were fit before the task-grouped folds, and no durable fold artifact was
saved.

Result: predicted native probability exceeded baseline for every task in every
fold. The allocator selected native 100% of the time. Gain over always-native:
0.000 in all 5 folds.

## 4. Allocator evaluation (test split, 55 tasks)

| Strategy | Successes | Rate | Allocation |
|---|---:|---:|---|
| Always baseline | 19/55 | 34.5% | 55 baseline |
| Always native | 30/55 | 54.5% | 55 native |
| Neural allocator (v1) | 30/55 | 54.5% | 55 native |
| Neural allocator (v2) | 30/55 | 54.5% | 55 native |
| Uniform random | 24.5/55 | 44.5% | 27.5 each |
| Paired oracle | 33/55 | 60.0% | 41 baseline, 14 native |

The neural allocator exactly matches always-native. The 3-task (5.5pp) oracle
headroom on this split is not realized.

## 5. 5-fold cross-validation (455 tasks, logistic regression)

| Fold | n | Allocator | Always-native | Oracle | Gain |
|---:|---:|---:|---:|---:|---:|
| 0 | 91 | 0.505 | 0.505 | 0.538 | +0.000 |
| 1 | 91 | 0.429 | 0.429 | 0.505 | +0.000 |
| 2 | 91 | 0.527 | 0.527 | 0.637 | +0.000 |
| 3 | 91 | 0.396 | 0.396 | 0.451 | +0.000 |
| 4 | 91 | 0.407 | 0.407 | 0.505 | +0.000 |
| Mean | | 0.453 | 0.453 | 0.527 | +0.000 |

Zero allocation gain in every fold. Baseline was selected zero times across all
folds.

## 6. What the result does and does not show

### Native dominates in every repo

| Repo | Baseline | Native | Diff | n |
|---|---:|---:|---:|---:|
| django | 0.354 | 0.505 | +0.150 | 206 |
| sympy | 0.282 | 0.451 | +0.169 | 71 |
| sphinx | 0.256 | 0.308 | +0.051 | 39 |
| matplotlib | 0.258 | 0.387 | +0.129 | 31 |
| scikit-learn | 0.433 | 0.667 | +0.233 | 30 |

There is no repo where baseline outperforms native. A repo-conditioned strategy
would also always select native.

### The 34 baseline-only wins are not feature-distinguishable

Baseline-only tasks have mean problem-statement length 1475 characters,
indistinguishable from native-only tasks at 1499 characters. The TF-IDF and
bag-of-words representations cannot separate the two groups.

### `R=1` cannot identify the cause

The outcomes are positively correlated because the two arms share task
difficulty. With only one outcome per task-arm cell, the data cannot separate
stable task-treatment heterogeneity from within-cell stochastic variation. The
observed disagreement matrix therefore does not prove that disagreements are
noise, nor does it prove that they are predictable treatment interactions.

## 7. Verdict

**No-go for the tested allocator implementations on this data.** Both neural
configurations and the diagnostic logistic regression collapse to always-native.
The observed 7.5-point realized-oracle gap was not captured by these models. It
is not an expectation, and this experiment does not rule out every richer
pre-action representation.

## 8. Implications for the project

### The aggregate association is measured; heterogeneity is unresolved

Native is 13.7 points better overall in this observational paired release. The
realized oracle is 7.5 points above always-native. The tested representations do
not predict the interaction between task features and arm outcome, while `R=1`
prevents separating stable heterogeneity from stochastic outcome variation.

### This motivates, but does not validate, replication

The M3 preregistration required repeated rollouts specifically to measure
discordance. O1 has `R=1`, so it cannot test whether repeated outcomes recover a
stable task-arm ordering. That question is evaluated separately in
[`repeated-outcome-smoke-result.md`](repeated-outcome-smoke-result.md).

### What would need to change

1. **Multiple attempts per task-treatment cell** to determine whether stable
   cell means exist. Repetition improves measurement but is not automatically an
   allocator solution.
2. **Treatments with comparable base rates.** The 13.7-point native advantage
   means any model can get most of the benefit by always picking native. An
   allocator experiment needs treatments with similar average performance but
   different task-specific strengths.
3. **Richer task features.** The tested bag-of-words and diagnostic TF-IDF
   representations did not capture task-arm interactions. A pretrained encoder
   is a candidate, not an established remedy, and should be tested only after
   replication and treatment-comparability issues are resolved.

### Relationship to M3

This result does not change the M3 no-go. It shows that the current outcome
models do not exploit O1's one-shot paired outcomes. It does not independently
prove that stochastic noise caused the M3 failure. The M3 mechanical-enforcement
fixes remain necessary before additional replicated outcome collection is
scientifically useful.
