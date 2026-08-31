# Repeated-Outcome Smoke Test Result

Date: 2026-08-11

Verdict: **inconclusive**

## Decision

Multiple outcomes appear useful for measuring task-arm differences, but this
smoke does not show that they solve the allocator problem. The point estimate
improves as more calibration repeats are used, yet every task-bootstrap interval
includes zero and most task choices remain tied.

The defensible position is:

> Repetition is a measurement prerequisite when outcomes are stochastic. It is
> not sufficient without behaviorally distinct treatments, stable task-specific
> heterogeneity, and a model that transfers that heterogeneity to new tasks.

## Evidence and Scope

Primary artifact: `yoonholee/terminalbench-trajectories`, Apache-2.0, revision
`04e8940f5b6736a7ce8d22224fe2f2af74163ed2`.

Exact slice:

- model: `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8@together_ai`;
- arms: `openhands` and `terminus-2`;
- 1,807 raw source rows;
- 908 logical trials after dropping 899 consistent duplicate rows;
- 89 tasks seen;
- 68 tasks retained with exactly five logical trials per arm;
- 21 tasks excluded for unequal repeat counts;
- OpenHands: 87/340 successes (25.59%);
- Terminus: 78/340 successes (22.94%).

The source is observational. The two scaffold arms were not seed-paired,
contemporaneously randomized, or guaranteed to share every configuration. This
tests repeat stability in a fixed historical slice, not a causal scaffold
effect.

## Method

The plan is in [`repeated-outcome-smoke-plan.md`](repeated-outcome-smoke-plan.md).
For each calibration size `k`, every calibration-position subset for OpenHands
was crossed with every independent subset for Terminus. The selector chose the
arm with more calibration successes for that task; ties were scored 50/50. A
global fixed arm was selected from the same calibration data and both strategies
were scored only on held repetitions.

Tasks were resampled as whole clusters for 10,000 bootstrap draws. The fixed arm
was reselected inside each draw and split pair.

## Results

| Calibration repeats per arm | Held repeats | Split pairs | Selector | Calibrated fixed | Lift | 95% task-bootstrap CI | Tie rate | Non-tied sign concordance |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 25 | 25.56% | 25.00% | +0.56 pp | [-1.88, +2.54] pp | 80.65% | 163/226 (72.12%) |
| **2** | **3** | **100** | **26.35%** | **25.36%** | **+1.00 pp** | **[-1.68, +4.02] pp** | **72.35%** | **862/1164 (74.05%)** |
| 3 | 2 | 100 | 26.91% | 25.59% | +1.32 pp | [-1.51, +5.14] pp | 67.06% | 862/1164 (74.05%) |
| 4 | 1 | 25 | 27.21% | 25.59% | +1.62 pp | [-1.71, +6.44] pp | 63.00% | 163/226 (72.12%) |

The primary `k=2` result fails the frozen support rule because its lower bound
is below zero. It also falls below the project's existing five-point practical
improvement target; even its upper interval endpoint is about four points.

The increasing point estimates are suggestive but not an independent dose
response: every row of the learning curve reuses the same five observed trials.
Tie rates decline mechanically as more binary outcomes are observed and remain
high even at `k=4`.

Sign concordance is conditional on both calibration and held differences being
nonzero. It covers only a selected minority of task-split cases and is not a
74% allocator accuracy claim.

## What This Corrects About O1

The O1 experiment had one outcome per task-arm cell. From `R=1` data, stable
task-treatment heterogeneity and within-cell stochastic noise cannot be
separated. Positive correlation between the two arms is expected from shared
task difficulty and does not prove that disagreements are noise.

The valid O1 conclusion is narrower: the tested bag-of-words neural models and
the diagnostic TF-IDF model selected the globally stronger native arm for every
task. O1 did not establish why they failed or that every richer feature model
must fail.

This repeated-trial smoke fills part of that identification gap. It finds weak
same-task persistence, but not enough evidence to conclude that repetition alone
creates a useful allocator.

## Hypothesis Disposition

| Hypothesis | Result |
|---|---|
| Stable task-specific heterogeneity | Plausible but not established; positive point lift, interval crosses zero |
| Fixed-arm dominance | Not rejected; calibrated fixed remains within sampling uncertainty of the selector |
| Historical configuration confounding | Unresolved and cannot be removed with this artifact |

## Implication for M3

Do not proceed by merely increasing replica count under the current M3 policy
grammar. The original pilot already showed that observation, verification, and
recovery labels were not behaviorally enforced. Repeating a weak or nonexistent
intervention estimates that weakness more precisely.

The next native evidence sequence should be:

1. Mechanically enforce or remove every policy factor before collecting outcome
   evidence.
2. Use only two treatments with similar aggregate performance and a credible
   reason for complementary task strengths.
3. Run a small excluded `R=2` manipulation/stable-disagreement canary first.
4. Only after that passes, freeze a prospective repeated panel with separate
   calibration and held repetitions, a minimum useful lift of five points, and
   task-clustered sequential efficacy/futility boundaries.
5. For the actual new-task allocator claim, train on repeated cell means from
   training tasks and evaluate treatment selection on entirely held-out tasks.
   Same-task calibration followed by another attempt is a narrower recurring-
   task use case.

## Artifacts

- Raw projected slice:
  `.runs/terminalbench-repeat-smoke.raw.jsonl`
- Raw SHA-256:
  `68c197c4f9a902e5c7228e62b7f19ef970252802a608aec038819b095bf8666e`
- Pairing-corrected analysis:
  `.runs/terminalbench-repeat-smoke.analysis.json`
- Analysis SHA-256:
  `818c055494acb68caf701ff0a98ca7b870513d0e676cb8423865ea2ffd034834`

The `.runs` artifacts are ignored and are not part of the Git worktree.

## Follow-up

The next evidence step is documented in
[`m3-observation-canary-result.md`](m3-observation-canary-result.md). A fresh
mechanics panel established 100% first-observation enforcement, then a frozen
24-attempt `R=2` screen initially returned `futility_no_go`. Post-run raw-event
audit found a controller bug that killed persistent browser processes after 30
seconds and affected 18/19 failures. The screen is therefore invalid and does
not provide convergent outcome evidence. Its useful contribution is the
infrastructure falsification that must be repaired before another repeated
panel.
