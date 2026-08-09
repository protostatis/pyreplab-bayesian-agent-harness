# External treatment-outcome dataset decision

## Decision question

Can a public dataset cheaply validate a small model of

```text
P(verified success | predecision task, fixed harness treatment)
```

before we generate a larger Direct-vs-Deliberate corpus with Gemma?

The ideal source has:

1. the same task IDs evaluated under two or a few predeclared treatments;
2. the same base model across treatments when possible;
3. an exact, immutable treatment definition (prompt, tools, runner, budget);
4. a machine-checked terminal binary outcome;
5. only pre-action task information in `model_input`;
6. task-grouped evaluation and a usable license; and
7. enough independent task groups, not merely many correlated attempts.

## Current position

No audited source is a large randomized *prompt-string-only* A/B experiment.
The strongest available primary experiment is nevertheless close enough to
validate the closed-set treatment-conditioned modeling workflow:

> Use the paired SWE-bench Verified O1 baseline and native-tool-calling
> releases as an offline architecture experiment. Treat each arm as one
> immutable prompt-plus-tool-interface bundle. Do not transfer its estimated
> treatment effect to Direct/Deliberate.

The external and native evidence planes must stay separate:

- External rows test preprocessing, grouped evaluation, treatment interactions,
  uncertainty plumbing, and whether allocation has predictable headroom.
- Native Gemma rows estimate the effects of this project's Direct and Deliberate
  treatments.
- A live allocator should be enabled only after a frozen native test beats the
  best fixed native policy, not merely a random policy mix.

## Exploration result snapshot (2026-08-09)

The row count alone is not the statistical sample size. For policy allocation,
the important counts are independent task groups, comparable treatment bundles,
and tasks observed under more than one treatment.

| Source | Outcome rows | Independent tasks | Comparable treatments | Outcome | Decision |
|---|---:|---:|---:|---|---|
| O1 SWE-bench paired releases | 910 admitted rows | 455 | 2 same-model interface bundles | executable SWE-bench `resolved` | primary closed-set A/B architecture test |
| TerminalBench Qwen matched slice | about 908 logical trials | 89 | 2 same-model scaffolds | task verifier reward | repeated observational robustness test |
| ToolFailBench headline panel | up to 19,000 cells before censoring | 1,000 | 19 model/deployment stacks | reproducible benchmark rule/ensemble | multi-arm stress test after leakage-safe rebuild |
| Nebius SWE-agent trajectories | 80,036 attempts | 3,591 | 3 incompletely specified model labels | end-to-end pipeline target | observational prediction only |
| WorkBench committed results | 23,460 responses | 690 tasks / 69 templates | 34 coarse model/tool-menu cells | requires pinned sandbox replay | not a ready binary-outcome import |
| AWS Bench datasets | 0 attempts | 134 task definitions | 0 observed | verifier definitions only | rerun asset, not outcome training data |

### Main exploration conclusion

No audited, clearly licensed source simultaneously provides:

1. many independently varied policy prompts;
2. the same base model and tasks crossed over those prompts;
3. exact prompt/tool/budget descriptors;
4. machine-verified terminal outcomes; and
5. enough held-out policies to validate zero-shot policy-embedding transfer.

Therefore the public datasets answer two narrower questions:

- O1 answers whether a small model can learn task-dependent allocation between
  two known treatment bundles.
- ToolFailBench answers whether the architecture can handle many known
  categorical treatments, but its treatments are model stacks rather than
  generated harness policies.

Neither proves that prompt text can place a never-executed policy correctly in a
treatment embedding space. That claim requires a native controlled-policy corpus
and leave-one-policy-out evaluation.

### Harness implementation checkpoint

The local MVP now supports the experiment required by that conclusion:

- deterministic sampling from a 36-cell planning × verification × execution ×
  budget grammar;
- deeply immutable treatment metadata plus exact bundle and registry hashes;
- arbitrary registered-treatment execution and resumable task-matrix batches;
- leakage-safe export with separate task and treatment descriptors;
- hybrid identity/text/tool/interface/budget model features; and
- strict multi-treatment held-out evaluation using posterior-mean argmax,
  fixed-treatment baselines, uniform random assignment, and a clearly labeled
  `hindsight_realized_oracle`.

Generalized evaluation deliberately has no implicit quota. Observed costs are
reporting-only until a deployment constraint and leakage-safe ex-ante cost model
are specified. It aborts on incomplete panels, duplicate attempts, identity or
hash drift, unequal task-side inputs, or task IDs crossing splits. The full
393-test suite passes in the remote CPU/PyTorch environment. This validates the
software path, not unseen-policy generalization; the next evidence requirement
is an actual frozen generated-policy corpus with policy-held-out evaluation.

## Ranked candidates

### 1. O1 SWE-bench Verified paired releases — primary A/B experiment

Repositories:

- `AlexCuadron/SWE-Bench-Verified-O1-reasoning-high-results`
- `AlexCuadron/SWE-Bench-Verified-O1-native-tool-calling-reasoning-high-results`

Audited properties:

- License: CC-BY-4.0 on both dataset cards.
- Model: `o1-2024-12-17` in every sampled response.
- Harness family: OpenHands CodeAct v2.2; both releases claim high reasoning.
- Shared tasks: 495 issue IDs.
- Strict complete pairs: 455 issues with boolean `resolved` in both arms.
- Baseline: 144/455 resolved (31.6%).
- Native tools: 206/455 resolved (45.3%).
- Exact paired outcome matrix:

  | Baseline | Native pass | Native fail | Total |
  |---|---:|---:|---:|
  | Pass | 110 | 34 | 144 |
  | Fail | 96 | 215 | 311 |
  | Total | 206 | 249 | 455 |

  The paired oracle succeeds on 240/455 tasks (52.7%), so the maximum observed
  allocation headroom over always-native is 34 tasks, or 7.5 percentage points.
  A learned selector realizes that headroom only if it can identify baseline-only
  wins without giving back too many of the 96 native-only wins.
- Treatment is visible before the first action:
  - baseline has a 5,070-character system prompt with textual tool definitions,
    a manual function-call protocol, and a worked example in the first user
    message;
  - native mode has a 617-character system prompt and three native tool schemas
    in request `kwargs.tools`.
- Outcome provenance: SWE-bench executable tests, with `tests_status` and
  per-issue evaluation artifacts available for audit.

Interpretation:

- This is a same-model, same-task **harness/interface bundle** comparison.
- It is not an atomic prompt-only intervention: system text, few-shot framing,
  tool transport, action formatting, and response parsing all change.
- There is only one observed run per task-arm, so run-level stochasticity is not
  separately estimable.
- The marginal 13.6-point native advantage does not prove allocator headroom.
  Predictability of the 34 baseline-only wins determines whether
  task-conditioned selection can beat always-native.

Recommended use: import now for offline closed-set A/B architecture validation.

### 2. TerminalBench 2.0 matched scaffold slices — repeated robustness test

Strongest audited slice:

```text
model: Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8@together_ai
arm A: OpenHands 0.60.0
arm B: Terminus 2.0.0
tasks: 89 shared
repeats: approximately five trials per task-arm
```

The derivative `yoonholee/terminalbench-trajectories` contains binary `reward`,
and the arm prompts are recoverable for almost every selected trial. It is useful
for repeated-trial/hierarchical modeling, but the treatment is the whole scaffold
and assignment dates/configuration are observational and incompletely recorded.

Recommended use: secondary robustness analysis, not a causal prompt-effect claim.

### 3. ToolFailBench — multi-arm architecture stress test

`SoHarshh/toolfailbench-traces` provides a complete crossed panel:

- 1,000 tasks;
- 22 raw model-stack files, with 19 headline treatments;
- the exact same 1,000 task payloads in every raw file;
- Apache-2.0 dataset and code;
- rule-defined terminal classifications and optional judge ensembles.

This is excellent for testing a categorical treatment model across many known
arms. The treatment is a model/deployment stack, not a harness-prompt A/B, and it
cannot estimate or transfer Direct/Deliberate effects.

#### Current local ToolFail artifact is quarantined

`.runs/adhoc-toolfailbench-dataset.jsonl` is not leakage-safe for the intended
predecision estimand. Its `model_input.public_metadata` includes privileged or
unavailable fields derived from `mock_tool_return` (for example prices and P/E
ratios) plus `target_failure_mode_code`. Metrics from this file must not be cited.

The current CV also uses each outer validation fold both for early stopping and
for reported held-out metrics. Rebuild with an explicit semantic allowlist, then
use nested grouped CV or independently fixed epochs.

Recommended use: rebuild later as a multi-arm stress test.

### 4. BFCL Qwen prompt/function-calling comparisons — license hold

Audited result archives appear to contain same-model paired prompt-versus-native
function-calling outcomes and a 26-arm formatting study. The BFCL code is
Apache-2.0, but the separate response archive does not declare a clear license.

Recommended use: do not import or redistribute until the result license is
clarified.

## Audited exclusions and secondary assets

| Candidate | Finding | Valid use |
|---|---|---|
| `nebius/SWE-agent-trajectories` | 80,036 attempts but only 3,591 tasks; three model labels, adaptive collection, incomplete treatment provenance | observational model-conditioned prediction only |
| `SWE-bench/SWE-smith-trajectories` | `tool/xml/ticks` are post-outcome serialization transforms; shared rows copy the same outcome | format robustness, not treatment effects |
| `SWE-Gym/OpenHands-Verifier-Trajectories` | verifier-training data with source-model and task IDs dropped | verifier/critic training only |
| `olly-styles/WorkBench` | task CSV `outcome` is a reference action list, not observed success; binary labels require deterministic sandbox replay; historical A/B provenance is weak | task/model difficulty study after pinned rescoring |
| `aws-bench/aws-bench-datasets` | 134 task/evaluator definitions and zero observed attempts/outcomes | future rerun benchmark only |
| AgentBoard and `tau2-bench-data` | strong task/evaluator assets but no released treatment attempts | controlled future reruns |
| WebRewardBench / WebPRM / WPRM | step preferences/checklists, often with future-trajectory leakage; not terminal multi-policy outcomes | auxiliary critic research after license review |
| ClawBench V2 | binary rescored outcomes and exact effective prompts, but only one published harness family | labeled outcome prior/benchmark, not prompt A/B |
| OSWorld verified trajectories | some same-task step-budget overlap, but exact historical prompts/agent revisions absent | conditional budget-treatment analysis |
| Workspace-Bench Lite | aggregate harness pass counts with good nominal overlap, but raw runs/prompts/judge versions and data license are incomplete | hold pending provenance |
| OpenCUA / TerminalWorld | demonstrations or task/verifier definitions, not observed multi-treatment outcomes | representation learning or reruns |
| ATBench / AgentDefense-Bench | safety labels/taxonomies rather than verified task completion | separate safety model only |

## Minimum O1 importer contract

Emit one row per issue-arm with:

```text
schema_version
source_dataset_id
source_revision
source_license
task_id / pair_id                 # SWE-bench issue ID
attempt_id
treatment_id                      # immutable categorical bundle ID
treatment_bundle_hash/reference
verified_success                  # source field: resolved
verifier_id / verifier_version
design = paired_complete
split_group = task_id
model_input:
  text                            # canonical SWE-bench problem statement
  family / template / difficulty  # only if known before execution
  public_metadata                 # semantic allowlist only
  policy_id                       # treatment ID for current model compatibility
  policy_version                  # immutable bundle revision
  treatment:
    text                          # exact prompt or pinned treatment description
    bundle_id                     # composite ID + content hash
    max_output_tokens
    tool_call_limit
    command_timeout_seconds
    wall_time_limit_seconds
    tool_interface
    allowed_tools_signature
```

Also write:

- a treatment registry containing the exact system prompt, tool schemas, model,
  runner, budget/reasoning declarations, and source hashes;
- an exclusion ledger for all shared IDs not admitted to the 455 complete pairs;
- source outcome/audit fields outside `model_input`.

Never put conversation turns, patches, test results, `tests_status`, success
flags, or gold SWE-bench patches/tests into `model_input`.

For the fixed O1 A/B experiment, categorical treatment IDs remain the defensible
primary representation. A generalized model may additionally encode prompt text
and structured treatment attributes, but two O1 bundles cannot validate semantic
generalization to unseen policies. The ID embedding should act as a seen-policy
residual; text, tools, interface, and budgets provide the only path for a novel
bundle to differ from the shared `UNK` identity. Trust in that path requires many
controlled policies and leave-one-treatment-out evaluation.

## Implication for generated-policy outcome training

The native generalized-policy experiment should use a finite, interpretable
policy grammar rather than unconstrained random prose. Candidate dimensions are:

```text
planning mode
verification mode
execution/retry style
tool-call budget
output-token budget
command and wall-time budgets
```

Every generated bundle must be frozen before outcomes are observed and carry:

```text
policy ID + version
exact system prompt
allowed tools and interface
structured budgets
canonical bundle hash
generator seed and factor settings
```

The evaluation sequence is:

1. task-grouped held-out evaluation for known policies;
2. leave-one-policy-out evaluation for descriptor-based transfer;
3. a task-only baseline, ID-only baseline, descriptor-only model, and hybrid
   ID-plus-descriptor model;
4. posterior uncertainty and abstention checks on held-out policies; and
5. no live selection until the hybrid beats the best fixed policy on a frozen
   native test.

An independent policy-generating agent can reduce researcher cherry-picking,
but independence alone does not create identifiability. The generator must be
task-outcome blind, constrained by the declared grammar, and followed by actual
executions and independent verification.

## How this becomes a harness

### Plane A — offline architecture validation

```text
two pinned external releases
  -> importer + treatment registry + exclusion ledger
  -> 455 complete task pairs / 910 rows
  -> task-grouped outer folds
  -> inner validation for preprocessing, early stopping, and tuning
  -> exactly one outer prediction per task-arm
  -> paired allocator comparison
```

Required baselines:

- task-only predictor;
- treatment-only predictor;
- regularized task-plus-treatment logistic model;
- current small NN plus Bayesian head;
- always-baseline, always-native, random mix, and paired oracle.

Primary evaluation should report arm-specific Brier/log loss/calibration, the
paired 2x2 outcome matrix, selected-arm success, regret versus the paired oracle,
and task-clustered/bootstrap uncertainty. Add a repository-held-out stress test.

### Plane B — native evidence collection

```text
deterministic task generator
  -> one task contract
  -> execute immutable Direct-v4 and Deliberate-v4 on fresh isolated attempts
  -> independent semantic verification
  -> leakage-safe paired rows
  -> native-only model and frozen paired evaluation
```

The external rows must not be pooled as if their O1 treatments were Direct and
Deliberate. External representation reuse would be a separate transfer
experiment and must be judged on an untouched native test.

### Plane C — eventual live allocation

```text
new task predecision state
  -> construct one counterfactual model_input per registered native treatment
  -> predict success distribution and expected cost
  -> choose one policy under budget/uncertainty/exploration constraints
  -> execute only that policy
  -> independently verify and log assignment probability + outcome
  -> periodically retrain while retaining randomized/shadow evaluation
```

Live single-arm observations are not equivalent to paired evaluation. Preserve
positivity, log assignment propensities, and retain randomized controls or
paired shadow attempts on a subset.

## Review of the local demo-task walkthrough

The three-phase description (collect evidence, fit model, evaluate/select) is a
good operational summary of the repository. The dataset, model, `inspect`,
allocator, dashboard, and `predict` commands exist.

Corrections:

1. `orchestrator --pair` **executes** two pre-authored policy bundles; it does not
   generate policies.
2. `sqlite/hard/seed=3` was selected because a prior run disagreed. A rerun is a
   post-hoc stochastic-stability demo, not confirmatory or independent evidence.
3. A demo should use its own `--remote-run-root`; otherwise its attempts can
   contaminate an evidence corpus.
4. Repeated attempts of one task must stay in the same task group, and selection
   on prior disagreement must be logged.
5. Five completed v4 pairs demonstrate plumbing only. They cannot support a
   learned allocator claim.

A correctly labeled demo flow is:

```text
select deterministic task spec
  -> load Direct-v4 and Deliberate-v4 from immutable files
  -> run fresh counterbalanced attempts in a dedicated demo root
  -> verify both final workspaces
  -> display success, cost, timing, and disagreement
```

If `sqlite/hard/seed=3` is repeated, mark it:

```text
study_role = post_hoc_targeted_replication
selected_on_prior_disagreement = true
selection_reason = v4_direct_deliberate_disagreement
```

## Next steps

1. Freeze the generalized treatment-registry and external-row schema.
2. Project the O1 source columns, run the paired importer, and reproduce the
   documented 2x2 outcome table before fitting.
3. Add grouped nested evaluation and task/ID/descriptor/hybrid baselines.
4. Generate a small controlled native policy menu, execute it on predeclared
   paired task blocks, and reserve complete policies for leave-policy-out tests.
5. Rebuild ToolFailBench only after removing privileged fields; quarantine the
   current metrics.
6. Resume or replace the Direct-v4/Deliberate-v4 rollout only after the offline
   architecture decision.

## Primary sources

- O1 baseline: <https://huggingface.co/datasets/AlexCuadron/SWE-Bench-Verified-O1-reasoning-high-results>
- O1 native tools: <https://huggingface.co/datasets/AlexCuadron/SWE-Bench-Verified-O1-native-tool-calling-reasoning-high-results>
- TerminalBench derivative: <https://huggingface.co/datasets/yoonholee/terminalbench-trajectories>
- Official TerminalBench leaderboard: <https://www.tbench.ai/leaderboard/terminal-bench/2.0>
- ToolFailBench: <https://huggingface.co/datasets/SoHarshh/toolfailbench-traces>
- ToolFailBench code: <https://github.com/SoHarshh/ToolFailBench>
- WorkBench: <https://github.com/olly-styles/WorkBench>
- AWS Bench datasets: <https://github.com/aws-bench/aws-bench-datasets>
