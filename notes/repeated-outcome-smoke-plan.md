# Repeated-Outcome Smoke Test Plan

Date: 2026-08-11

Status: frozen before trial-level outcome analysis; pairing correction recorded
before the final sensitivity run

## Decision Question

Do multiple outcomes per task-treatment cell reveal stable treatment
heterogeneity that a selector can exploit on held-out repetitions, or do they
only estimate the same globally best fixed arm more precisely?

## Competing Hypotheses

1. **Stable heterogeneity:** early repetitions identify the better arm for a
   task, and that choice beats the best fixed arm on untouched repetitions.
2. **Fixed-arm dominance:** repetitions reduce uncertainty but task-specific
   choices do not beat a fixed arm selected from the same calibration data.
3. **Historical confounding:** apparent repeat stability reflects scaffold,
   date, or configuration differences rather than a transportable policy
   effect.

The third hypothesis cannot be eliminated by this observational dataset. This
smoke can support or reject repeat stability in the fixed historical slice; it
cannot establish a causal prompt effect or transport to Gemma/Unbrowser.

## Data Slice

Source: `yoonholee/terminalbench-trajectories`, Apache-2.0, revision
`04e8940f5b6736a7ce8d22224fe2f2af74163ed2`.

Filter:

- model exactly
  `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8@together_ai`;
- arms exactly `openhands` and `terminus-2`;
- deduplicate logical trials by `(agent, task_name, trial_name)`;
- require duplicate source rows for one logical trial to agree on binary
  `reward`;
- retain only tasks with exactly five unique logical trials for each arm.

The prior audit already reported that this complete slice contains 68 tasks,
340 trials per arm, and aggregate passes of 87 versus 78. Those aggregate
counts are known before this analysis. Trial-level split outcomes are not.

## Deterministic Trial Ordering

Trials have no shared random-seed pairing across arms. Within each task-arm
cell, order the five logical trials by SHA-256 of:

```text
20260811|task_name|agent|trial_name
```

The hash order is outcome-blind. A position subset selects calibration trials;
the complement is held out.

## Primary Test

Use `k=2` calibration repeats and three held repeats. Each arm has ten choices
of two positions from five. Because trials are not seed-paired across arms,
evaluate the Cartesian product of those choices: 100 independent arm-A/arm-B
calibration-subset pairs.

For each arm-A/arm-B position-split pair:

1. Count calibration successes separately for both arms on each task.
2. Select the arm with more calibration successes for that task.
3. If calibration counts tie, allocate fractionally 50/50 rather than using an
   arm label.
4. Score the selection on only the three held repetitions.
5. Select a global fixed arm from aggregate calibration successes only; score
   it on the same held repetitions. Split a global tie fractionally.
6. Also report both always-arm strategies and the held-out realized oracle.

Average over the ten position splits. Resample tasks as whole clusters with
10,000 bootstrap draws and seed `20260811`. Recompute the calibrated fixed arm
inside every bootstrap draw. Report the 95% percentile interval for selector
lift over the calibrated fixed arm.

## Secondary Learning Curve

Repeat the same Cartesian-product cross-fit for `k=1`, `k=3`, and `k=4`.
Report:

- selector held success;
- calibrated-fixed held success;
- selector lift;
- fraction of task decisions assigned to each arm, with ties fractional;
- sign concordance when both calibration and held arm differences are nonzero;
- task-cluster bootstrap interval for lift.

The learning curve is descriptive because position splits reuse trials. The
task bootstrap is conditional on the observed five-trial cells and does not
represent trial-level or temporal sampling uncertainty.

## Decision Rule

- **Supports repeats as a solution:** primary `k=2` lift is positive with a
  task-bootstrap 95% lower bound above zero, and lift does not disappear as
  calibration repeats increase.
- **Does not support repeats:** primary lift is nonpositive, or choices collapse
  to the calibrated fixed arm without held-repeat improvement. This is not an
  equivalence or futility conclusion without a predeclared margin.
- **Inconclusive:** primary point lift is positive but its interval crosses
  zero, or the learning curve is materially inconsistent.

Even a pass would establish only that repeated outcomes expose stable
same-task arm ordering in this slice. It would not show that a task-feature
model can predict the ordering for a new task.

Statistical support above zero would not by itself establish practical utility.
The existing project target of five percentage points is reported separately,
and calibration-attempt cost is outside this smoke's success-rate estimand.

## Post-Review Pairing Amendment

The initial implementation used the same hash-position subset for both arms.
An independent statistical review identified that as an arbitrary cross-arm
pairing because the historical trials have no shared seeds. Before the final
reported sensitivity run, the analysis was changed to average every independent
arm-A/arm-B subset pair. This changes the number of splits from `C(5,k)` to
`C(5,k)^2` and makes averaged point metrics invariant to independent position
permutations within either arm.

The first, same-position result is retained only as an implementation diagnostic
and is not the decision result. It had a primary `k=2` lift of 1.03 percentage
points with a task-bootstrap interval of `[-1.42, 4.12]`. The final conclusion
uses only the pairing-corrected analysis.
