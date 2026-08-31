# Decision: Run the O1 Offline Architecture Experiment (Path B)

Date: 2026-08-11

## 1. Where we are

### M0 (terminal gym)

Completed: four-family procedural task gym, paired Direct/Deliberate Gemma
rollouts, independent semantic verification, leakage-safe dataset export,
neural outcome model with variational Bayesian head, held-out allocator
evaluation, static dashboard. 445-test remote suite passes.

Stopped: the 24-pair v4 gate was paused at 5/24 pairs. The terminal gym is
infrastructure proof, not the product target (browser/UnchainedSky).

### M1-M2 (unbrowser fixture gym + policy grammar)

Completed: deterministic confined fixture-verified browser-task gym, 72-cell
policy grammar (3x2x2x2x3), immutable treatment registry, confined execution
mode, M2 grammar-factor export.

### M3 (meta-learning allocator)

Completed: full preregistration, 72-policy registry, frozen split manifest,
attentive CNP infrastructure, headroom pilot (96 attempts).

Result: **NO-GO** on six independent checks:
- 18.75% repeat discordance (ceiling 10%)
- negative cross-replica lift (-0.5833)
- observation adherence 20-63% (ceiling 75%)
- verification adherence 0% for both levels
- recovery retry difference 12.5 points (required 25)
- tool-cap compliance 29% raw, 93% corrected (required 100%)

Post-no-go exploratory screen (232 attempts, all permanently excluded):
narrowed to `decompose + submit_directly + diagnose_retry_once + expanded`
but found 20% neutral discordance and no resolved single winner.

## 2. The core struggle

The M3 no-go was not a statistical-power problem. It was a mechanical-control
problem: the grammar factors are *instructed but not enforced*. The model
receives different prompts but produces similar behaviors. When the treatment
manipulation fails, `P(Y | x, do(policy))` is unidentified — no amount of
meta-learning can extract a policy effect from policies that do not
behaviorally differ.

Three paths were considered:

- **Path A:** Fix the mechanical enforcement, rerun the frozen pilot.
  Risk: even with perfect enforcement, Gemma-26B-IQ4 on fixture tasks may not
  produce enough stable disagreement. Costs ~96 native attempts to find out.

- **Path B (chosen):** Run the O1 SWE-bench offline experiment.
  455 complete paired tasks with two same-model bundles (text protocol vs
  native tool interface), real executable verification, 7.5-point oracle
  headroom. Tests whether `P(success | task, treatment)` has predictable
  structure at minimal cost.

- **Path C:** Simplify to a budget-only allocator.
  Reframes around an enforceable axis but drops the unseen-policy
  generalization thesis.

## 3. Why Path B first

1. **Cheapest test of the current modeling path.** If the current representations
   cannot beat a fixed arm on 455 real paired tasks, more native collection is
   premature. This diagnoses the current architecture; it cannot by itself kill
   the broader control thesis.

2. **No native compute required.** The data is CC-BY-4.0, the importer exists,
   and the modeling runs on CPU.

3. **It validates the estimand before spending on mechanics.** Path A's fixes
   are worth the compute only if the modeling approach works at all.

4. **It does not foreclose Path A.** If O1 shows predictable headroom, the
   mechanical fixes become worth the investment. If O1 fails too, the project
   needs a fundamentally different approach before more native spend.

## 4. What O1 can and cannot answer

**Can answer:** Do the tested representations recover task-dependent allocation
signal between two known complete treatment bundles on real software-engineering
tasks?

**Cannot answer:** Does the descriptor encoder generalize to unseen policy
combinations? (Two arms is a closed-set test, not a transfer test.)

## 5. Predeclared evaluation plan

### Data

- Project required columns from the two pinned HF releases plus canonical
  SWE-bench Verified tasks.
- Reproduce the documented 110/34/96/215 paired outcome matrix before any
  fitting.
- Task-grouped outer folds; inner validation for preprocessing and early
  stopping; exactly one outer prediction per task-arm.

### Baselines

- task-only predictor
- treatment-only predictor
- regularized task-plus-treatment logistic
- ID-only neural model
- descriptor-only neural model
- hybrid ID-plus-descriptor neural model

### Allocator strategies

- always-baseline
- always-native
- cost-matched random
- learned selector (each model above)
- paired oracle (ceiling, not a baseline)

### Metrics

- arm-specific Brier, log loss, calibration
- paired 2x2 outcome table
- selected-arm success rate
- regret versus paired oracle
- task-clustered bootstrap uncertainty

### Go/no-go for the modeling approach

- **Go:** learned selector beats always-native by a nonzero margin with
  task-clustered bootstrap lower bound above zero.
- **No-go:** always-native is within bootstrap noise of the learned selector.
- **Inconclusive:** point estimate favors the selector but intervals cross
  zero.

## 6. Infrastructure gaps to close

1. O1 importer does not emit the identity fields the generalized allocator
   requires (`treatment_bundle_id`, `treatment_bundle_hash`,
   `treatment_registry_hash`, compatible registry format).
2. O1 treatment descriptor omits budget numerics (4 of 10 keys).
3. No O1 source data is projected locally.
4. Local `.venv` has no PyTorch/numpy.

## 7. Claim boundary

O1 tests a closed-set architecture selection question. Its treatment effect is
not transferable to native Gemma/Pi policies. A live allocator remains gated
on frozen native evidence beating the best fixed native policy.

## 8. Execution Outcome and Deviations

The source Parquet artifacts matched the pinned revisions and reproduced the
documented 455-pair matrix exactly. The tested neural configurations and an ad
hoc TF-IDF diagnostic selected native for every evaluated task and tied the best
fixed arm.

This was not the full predeclared architecture comparison:

- the neural models used one deterministic train/validation/test split rather
  than nested grouped outer evaluation;
- task-only, treatment-only, ID-only, descriptor-only, and hybrid variants were
  not all run as separate controlled baselines;
- the TF-IDF diagnostic fit its unsupervised feature vocabulary before grouped
  folds and was not retained as a reproducible fold artifact;
- the generalized `TreatmentSpec` bridge uses canonical source-reference text
  rather than the full source prompts, plus equal placeholder budgets that were
  not observed in the releases.

The outcome is therefore a diagnostic no-go for the current tested selectors,
not a general no-go for every task representation. Because O1 has one outcome
per task-arm cell, it also cannot determine whether disagreement is stable
heterogeneity or stochastic variation. See
[`repeated-outcome-smoke-result.md`](repeated-outcome-smoke-result.md) for the
separate repeated-trial test.
