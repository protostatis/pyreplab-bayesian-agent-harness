# Unbrowser M3 Preregistration: Few-Shot Unseen-Policy Allocation

> **Status:** Frozen preregistration. No model architecture, grammar, split,
> calibration protocol, context schema, cost definition, baselines, metrics, or
> stop gates may be revised after the final-policy stage is unlocked. Corrections
> before rollout are recorded in the revision log at the bottom.
> The headroom-pilot manifest is separately locked at its first non-canary
> attempt; any later change invalidates that entire pilot and requires a visibly
> versioned replacement manifest while preserving the aborted data.
>
> **Scope:** M3 is the outcome-model training milestone. It depends on M0
> (interactive plumbing spike), M1 (deterministic interactive fixtures and
> isolation), and M2 (seeded task templates, multi-policy execution, independent
> verification, leakage-safe export). These milestones are summarized below but
> specified elsewhere.

## 1. Frozen Claim

For the frozen Gemma model, Pi runtime, Unbrowser v0.0.19 binary, verifier
version, task-generator version, and policy registry, estimate:

```text
mu_p  = E[Y | task ~ target pool, policy p, rollout seed]
c_p   = E[output_token_cost | task ~ target pool, policy p, rollout seed]
```

and the value of a predeclared allocator that, given a policy withheld from
meta-training, its structured descriptor, and at most k=8 calibration outcomes,
chooses among candidate policies on unseen deterministic Unbrowser fixture
tasks.

Primary claim: the allocator improves the success-cost frontier over the
strongest eligible baseline.

Secondary claims: global policy ranking accuracy; per-task selection success at
a predeclared operational budget.

### 1.1 What "unseen" means

A held-out policy is an **unseen combination of known grammar-factor levels**.
M3 tests factorial interpolation, not transfer to new instruction semantics, new
tools, new interfaces, or arbitrary natural-language policies.

## 2. Dependency Chain

```text
M0  Interactive Wikipedia plumbing spike (disposable-host, not security boundary)
M1  Deterministic seeded interactive fixtures; filesystem isolation; nonce-backed
    trace verification
M2  Multiple seeded task templates (>=8); multi-policy execution; resumable batch;
    independent semantic verifier; leakage-safe dataset export with grammar factors
M3  Few-shot unseen-policy allocation (this document)
```

M3 must not begin until M1 fixtures are deterministic and M2 export carries
structured grammar factors in `model_input`.

## 3. Policy Grammar

A mechanically composed 72-cell full factorial. Prompts are assembled from
frozen factor clauses; no creative per-policy wording.

| Factor | Levels | Identifier |
|---|---|---|
| Planning mode | direct / brief-plan / decompose | `planning` |
| Initial observation strategy | text-first / structure-first / targeted-query-first | `observation` |
| Verification mode | submit-directly / final-reobserve | `verification` |
| Recovery mode | fail-fast / diagnose-retry-once | `recovery` |
| Tool-call cap | lean(6) / expanded(12) | `tool_cap` |

```text
3 x 3 x 2 x 2 x 2 = 72 policies
```

### 3.1 Controlled constants

The following are held constant across all 72 policies to avoid confounding:

```text
base model             frozen Gemma 26B
Pi runtime             pinned package, version, CLI SHA-256, thinking=off
Unbrowser binary       pinned v0.0.19 with SHA-256 digest
allowed tools          bash, unbrowser
tool interface         native_bash_unbrowser_interactive_v1
max_output_tokens      held constant (not a grammar factor)
command_timeout        held constant
wall_time_limit        held constant
safety suffix          fixed text appended to every prompt
allowed URL set        controller-fixed per task template
```

Only `tool_cap` varies as a resource factor. Token budget and wall timeout do
not co-vary with it; this makes the tool-call-cap effect identifiable instead
of confounding four resource limits.

### 3.2 Behavioral adherence checks

Each factor must be either mechanically enforced or have a measurable
manipulation check recorded in every attempt:

- Pre-tool output checked against explicit planning markers: no preamble for
  `direct`, one `PLAN:` line for `brief_plan`, and at least `STEP 1:` plus
  `STEP 2:` lines for `decompose`.
- First successful post-navigation observation checked against `observation`:
  `text`, `blockmap`, or `query`, respectively.
- A repeated read-only observation before result submission checked against
  `verification`.
- A successful same-tool retry after the first eligible failure checked against
  `recovery`.
- Admitted tool-call count checked against the enforced `tool_cap`; rejected
  over-cap calls are recorded separately.

Non-adherence is recorded but does **not** exclude the attempt. The estimand is
intention-to-treat performance of the bundle.

### 3.3 Frozen registry

All 72 policies are generated, hashed, and committed to an immutable registry
file before any rollout. The registry hash is recorded in every exported row.

## 4. Policy Split

```text
48 meta-training policies
12 policy-development policies
12 final held-out policies
```

Split constraints (determined before any outcomes are observed):

- Every factor level strongly represented in meta-training.
- Each 12-policy holdout balanced: four per three-level factor, six per
  binary factor.
- Pairwise factor coverage maximized in both holdout sets.
- No policy selected based on outcomes.
- No duplicate or near-identical bundles across splits.

The split is generated by a deterministic constrained-search script and
recorded in the registry manifest.

## 5. Task Design

### 5.1 Templates

At least eight semantic interactive templates spanning distinct structures:

1. Single-page semantic extraction
2. Table filtering or sorting
3. Multi-page link navigation
4. Search or filter controls
5. Form entry with validation
6. Cross-page comparison and synthesis
7. Stateful multi-step workflow
8. Distractor, stale-state, or recovery scenario

Each template has easy, medium, and hard variants. Difficulty varies through
steps, distractors, ambiguity, page count, required state transitions, and
verification conditions, not merely text length.

All fixtures are harness-owned deterministic pages with generated nonces so that
correct answers cannot come from pretrained knowledge.

### 5.2 Task splits

```text
T_meta      meta-training task pool
T_dev_cal   policy-development calibration pool (disjoint)
T_dev_tgt   policy-development target pool (disjoint)
T_fin_cal   final calibration pool (disjoint)
T_fin_known final target, known templates
T_fin_held  final target, two structurally held-out templates
T_canary    operational validation only; permanently excluded
T_pilot     headroom/manipulation screening only; permanently excluded
```

The exporter assigns `T_canary` to `canary_excluded` and `T_pilot` to
`pilot_excluded`; neither role may enter any training, calibration,
development, or final-evaluation pool.

Template holdout: the six known templates are `single_page_extraction`,
`table_filter_sort`, `multi_page_navigation`, `search_filter_controls`,
`form_entry_validation`, and `distractor_recovery`. The two structurally held
templates are `cross_page_comparison` and `stateful_workflow`; they remain
unseen until final evaluation. Known-template and held-template results are
reported separately.

Final calibration uses known templates only. Held-template categories never
enter final calibration; the two held templates first appear in `T_fin_held`.

### 5.3 Final target pool

```text
54 unseen-seed tasks on known templates
18 tasks on two structurally held-out templates
total 72 final target tasks
```

## 6. Calibration Protocol

### 6.1 Frozen calibration panels

Every development and final policy receives one frozen ordered panel of 16
unique tasks drawn from its calibration pool. The same ordered panel is used for
every policy within a split to make policies comparable.

Nested prefixes:

```text
k = 0   zero-shot (descriptor only)
k = 4   first 4 calibration tasks
k = 8   first 8 calibration tasks  (PRIMARY)
k = 16  full calibration panel
```

`k=8` is the sole primary result. Other values are reported as calibration
sensitivity.

### 6.2 Constraints

- Calibration tasks never appear in any target evaluation pool.
- No model weights change during calibration; the CNP is applied, not trained.
- Context statistics (normalization, cost standardization) use meta-training
  data only.
- The calibration order is frozen before any target outcomes are observed.

## 7. Context Schema

### 7.1 Primary context (outcome-only)

Each calibration row contributes:

```text
task predecision features
    task prompt embedding (frozen encoder, projected)
    structured template, difficulty, interaction-type, form/state complexity
verified_success            binary
termination_class           coarse categorical (see below)
output_token_cost           primary cost, log-scaled and standardized
missingness_masks           for any absent field
```

Coarse termination classes:

```text
normal_completion
tool_call_limit
wall_timeout
invalid_or_tool_error
model_runtime_failure
verifier_declared_unsuccessful
```

### 7.2 Trace-summary ablation context

A separate predeclared ablation may add only frozen aggregate counts:

```text
calls by tool-operation type
tool_error_count
empty_result_count
retry_count
state_changing_action_count
read_only_action_count
termination_timing_bucket
```

### 7.3 Forbidden context fields

The following must never enter calibration context, model input, or descriptors:

```text
page text or DOM strings
URLs or domains beyond the task-fixed allowed set
selector strings
final answer text
verifier diagnostics or expected targets
private oracle or hidden-test results
policy identity (id, version, bundle_id, registry position, hash)
```

### 7.4 Policy descriptor

Primary descriptor uses structured grammar factors only:

```text
one-hot: planning, observation, verification, recovery, tool_cap
numeric: enforced tool-call cap
categorical: tool_interface, allowed_tools_signature
```

A frozen system-prompt embedding may be used in a predeclared ablation, but the
primary result must survive with structured descriptors alone.

## 8. Model Specification

### 8.1 Primary model: attentive Conditional Neural Process

```text
inputs:
    target task features      h_x(target)
    policy descriptor         h_p
    calibration context set   {(h_x_i, h_p, y_i, term_i, cost_i, mask_i)}

context element encoder:
    phi(h_x_i, h_p, y_i, onehot(term_i), log(1+cost_i), mask_i) -> e_i

global DeepSets summary:
    r_global = rho(h_p, mean(e_i), variance(e_i), log(1+k))

target-conditioned attention:
    r_local = Attention(q=h_x(target), k=h_x_i, v=e_i)

decoder input:
    [h_x(target), h_p, r_global, r_local, h_x(target) . h_p]

heads:
    success        -> Bernoulli logit
    cost           -> log-normal (mu_c, log_sigma_c) on log(1+C)
    termination    -> categorical logits (auxiliary, lambda_t ~ 0.2)
```

At `k=0`, learned empty-context vectors replace `r_global` and `r_local`.

Constraints:

- Frozen text encoder; trainable parameters kept below approximately one
  million (excluding the frozen encoder).
- One to two attention heads.
- Modest dropout in the decoder MLP.

### 8.2 Episodic training

Each episode:

1. Sample one meta-training policy uniformly (not proportional to row count).
2. Sample `k` from `{0, 4, 8, 16}` with weights favoring `k=8`.
3. Draw `k` calibration tasks and disjoint target tasks.
4. Compute episode loss; normalize per policy.

Context and target task IDs never overlap within an episode.

### 8.3 Uncertainty

Five independently initialized CNP ensemble members. Each trained on a
stratified bootstrap of whole meta-training policies. Within selected policies,
whole task clusters are resampled, not individual rows. Grammar-factor coverage
is preserved in each member.

Predictive distribution is the ensemble mixture. Temperature scaling for
success and interval calibration for cost are fitted separately for each `k` on
policy-development episodes and locked before final evaluation.

Ensemble spread is an operational epistemic approximation, not a Bayesian
posterior.

### 8.4 Cost expectation

```text
E[C] = exp(mu_c + sigma_c^2 / 2) - 1
```

Timeouts, failures, and runtime errors remain valid rows with their consumed
cost.

### 8.5 What the descriptor encoder learns and its limitations

The policy descriptor encoder (`h_p`) has no separate label. Its weights update
only through the outcome loss (success, cost, termination). There is no
contrastive or similarity objective. The expectation is that the outcome signal
teaches it to place similar-behaving policies near each other in the learned
embedding space.

#### Two fused signals, not one

The model's estimate of a policy is formed by two complementary paths:

1. **Descriptor prior** (`h_p` from grammar factors): what the recipe implies
   before any execution. This is the only information available at `k=0`.
2. **Calibration context** (the k observed outcomes): what the policy actually
   does on real tasks. At `k=8` this evidence can override a wrong prior.

The CNP fuses these: the descriptor gives a starting estimate, calibration
outcomes refine it. The attention mechanism lets calibration examples from
similar tasks weigh more heavily for a given target task.

#### Signal-thinness risk

With 48 meta-training policies and binary outcomes, the signal for learning
rich factor interactions is thin. The model can likely learn main effects
("plan-first is generally better") but may struggle with interactions
("plan-first wins on multi-page tasks but loses on simple extraction").

Mitigations:

- Descriptor input is structured one-hot factors, not free-form text. The input
  space is small (3x3x2x2x2 = 72 cells) and well-defined.
- The encoder is tiny (two-layer MLP, 64 dimensions) to avoid overfitting.
- The `k=0` ablation directly tests whether the descriptor alone carries
  transferable signal.
- The hierarchical residual model is a mandatory baseline. If a simpler
  factorial model matches or beats the CNP, the learned embedding space added
  nothing, and the meta-encoder claim is not supported.

#### What "meta" means here

The learner's target is the adaptation skill itself, not any single policy's
quality. It is learning "given a recipe and a few examples of how it behaves,
predict its behavior on a new task." This is a transferable skill evaluated on
policies withheld from training.

## 9. Cost Definition

### 9.1 Primary cost

Generated output tokens (model completions). This is the resource the allocator
most directly controls.

### 9.2 Reporting-only secondary costs

```text
provider turns
browser (unbrowser) tool calls
cache-read tokens
wall-clock latency
```

Total logical tokens must never be presented as physical compute. Logical-token
sums mix cached prefix reads with generated completions and are not comparable
across configurations.

### 9.3 Missing cost

Missing primary cost in a final evaluation panel is an integrity failure and
aborts the affected panel. It is not silently excluded or mean-imputed.

## 10. Baselines

All baselines are evaluated at `k={0,4,8,16}` with `k=8` primary.

### 10.1 Global baselines

```text
uniform random policy
descriptor-only structured ridge/logistic
calibration-only Beta-Binomial policy success estimate
calibration-best fixed global policy
hierarchical descriptor-prior plus policy residual   (mandatory strong baseline)
current hybrid model retrained with calibration rows
descriptor-only CNP at k=0
hindsight best fixed policy                          (ceiling, not a baseline)
```

### 10.2 Per-task baselines

```text
uniform random, cost-matched
calibration-best global fixed policy applied per task
family/difficulty lookup allocator
structured ridge/GLMM allocator
hierarchical residual allocator
current hybrid retrained allocator
CNP allocator                                         (primary)
realized per-task oracle                              (ceiling)
```

### 10.3 Required CNP ablations

```text
DeepSets global summary without target attention
no policy descriptors (context-only)
descriptor only, no calibration context
outcome-only context
outcome + termination
outcome + cost
full primary context
full primary context + safe trace summary
structured descriptor vs frozen prompt embedding
shuffled calibration contexts (leakage negative control)
```

### 10.4 Stop rule

If the **hierarchical descriptor-prior plus policy residual** model matches or
beats the CNP on policy-development ranking and frontier value, the meta-encoder
claim is not supported. The CNP may remain an evaluated artifact, but no claim
that meta-encoding adds value may be made from the final rollout.

## 11. Metrics

### 11.1 Global ranking

```text
Spearman rho
Kendall tau_b
pairwise ranking accuracy                (primary ranking metric)
top-1 and top-3                           (descriptive only)
success-rank and cost-rank metrics        (reported separately)
ranking at k = 0, 4, 8, 16
calibration gain from k=0 to k=8
```

Predictions are averaged over the fixed 72-task target distribution before
comparing to the empirical complete-panel means.

### 11.2 Per-task success-cost frontier

For frozen lambda values selected on policy-development:

```text
p*(task, lambda) = argmax_p [ predicted_success(task, p) - lambda * predicted_cost(task, p) ]
```

Selection uses predictions only; scoring uses the observed complete-panel cell.

Primary metrics:

```text
area under non-dominated success-vs-log-token-cost frontier
verified success at one predeclared operational token budget
pure-success selection at lambda = 0
observed budget deviation
frontier regret vs calibration-selected fixed policy
realized oracle frontier                   (ceiling)
results by template and difficulty
```

## 12. Go/No-Go Gates

### 12.1 Before full meta-training rollout

- Every grammar factor is mechanically enforced or has a measurable adherence
  check.
- Planning-marker adherence is at least 75% separately for `direct`,
  `brief_plan`, and `decompose`.
- First-observation adherence is at least 75% separately for `text_first`,
  `structure_first`, and `targeted_query_first`.
- Among attempts that reach a result-submission opportunity, the repeated-read
  rate for `final_reobserve` exceeds `submit_directly` by at least 25 percentage
  points, with at least eight opportunities per level. Intention-to-treat rates
  over all attempts are also reported.
- On the two frozen recovery tasks, eligibility requires that the designated
  `page_0` click is the first interaction/error and returns HTTP 503. There must
  be exactly eight eligible attempts per recovery level. The successful
  immediate same-tool retry rate for `diagnose_retry_once` must exceed
  `fail_fast` by at least 25 percentage points. Incidental errors on other tasks
  do not enter this manipulation check.
- Enforced tool-cap compliance is 100%.
- No page/target/verifier leakage enters descriptors or contexts.
- CNP validates on synthetic data (plumbing evidence only).

### 12.2 Headroom pilot gate (96 attempts)

Run four maximally different policies across 12 unique tasks, with two rollout
replicas for every policy-task cell (`12 x 4 x 2 = 96 attempts`), before any
large rollout. The 12 tasks contain two tasks from each known template and four
tasks at each difficulty. The 24 sequential panels use every four-policy
execution permutation once; within each template, every policy occupies every
execution position once. Templates are interleaved over execution time. Replica
chronology is counterbalanced: replica 0 runs first for six tasks and replica 1
runs first for six.

Every panel has a unique frozen sampling seed, shared across its four policies
as a common-random-number control; the other replica of the same task has a
different seed. The Pi provider is pinned to `openai-completions`, and the
extension writes these values into every provider request:

```text
temperature=0.8  top_p=0.95  top_k=40  min_p=0.05
repeat_penalty=1.0  presence_penalty=0.0  frequency_penalty=0.0
thinking=off
```

A sanitized provider-request receipt containing only the panel seed and these
sampling values is required whenever at least one provider turn occurred.

The two `distractor_recovery` tasks require a frozen first probe that returns a
recoverable non-200 status, yielding eight designed recovery-eligible attempts
per recovery level across replicas. These two tasks are manipulation-only: they
are excluded from non-degeneracy, stable-disagreement, cross-replica-lift, and
cost-range calculations. Stop unless all hold:

- Completeness is exactly 12 frozen tasks, 24 frozen panels, four expected policy
  bundles per panel, two replicas per policy-task cell, 96 unique measured
  attempts, no infrastructure-error records, and a non-missing output-token
  cost for every attempt.
- Across the ten manipulation-neutral tasks (20 rollout outcomes per policy),
  outcomes are non-degenerate: every policy has between 1 and 19 successes.
- Repeat discordance is at most 10% across the 48 repeated policy-task cells.
- At least 2 of the 10 neutral tasks show stable disagreement: one policy
  succeeds on both replicas and another fails on both replicas. Requiring two
  task clusters prevents a single seed from establishing headroom while retaining
  a 20% screening threshold.
- Cross-replica allocation has at least one more expected neutral-task success
  than the best fixed policy (`1/10 = 10` points): select from replica 0 and
  score on replica 1, repeat in the opposite direction, then average. Binary
  selector ties are scored uniformly and fractionally over all tied winners;
  policy labels never break ties.
- On neutral tasks only,
  `max(policy mean output tokens) / min(policy mean output tokens) >= 1.20`.

Model/runtime failures with a prepared attempt and usable event/cost record are
intention-to-treat failures and are never rerun. A controller, SSH, fixture-port,
or persistence failure that prevents a measured row invalidates the complete
96-attempt panel: stop immediately, preserve the aborted run and manifest, and
restart all 96 attempts in a fresh run root under the same manifest. Partial
panels are never combined. Failure classification must use infrastructure logs,
not observed task outcomes.

### 12.3 Before unlocking final policies

All must hold on policy-development split:

- Every meta-training policy has at least 32 unique task rows.
- CNP `k=8` beats its `k=0` version on policy-development log loss and pairwise
  ranking accuracy.
- CNP pairwise ranking accuracy is at least 0.62 and above chance under
  task-cluster bootstrap.
- CNP frontier area exceeds the strongest eligible baseline.
- CNP improves success by at least 3 points at the development operational
  budget.
- Shuffled calibration contexts remove the apparent calibration gain (negative
  control passes).
- Safe trace summaries do not produce implausibly large gains (leakage check).
- Repeat discordance is at most 10%; otherwise require complete R=2 evaluation.

### 12.4 Final go gate (k=8, frozen test)

All must hold:

1. Frontier-area improvement over the strongest eligible baseline, with
   task-cluster bootstrap 95% lower bound above zero.
2. At the predeclared operational budget, at least 5 percentage points higher
   success than the strongest cost-matched baseline, with lower bound above
   zero.
3. Held-policy pairwise ranking accuracy at least 0.65 and Spearman at least
   0.4.
4. Unconstrained pure success no more than 2 points below the strongest
   eligible baseline.

If point estimates pass but intervals cross zero, M3 is classified
**inconclusive**, not passed. Gates are not weakened after seeing results.

## 13. Bootstrap and Uncertainty

Preserve all dependence structures. Never bootstrap individual rows.

### 13.1 Conditional-calibration interval

- Hold realized `k=8` calibration contexts fixed.
- Resample final target tasks as whole clusters.
- Carry all policies and repeats for a selected task together.
- Recompute ranks, assignments, and frontier.

### 13.2 Full-protocol interval

- Stratified resampling of calibration tasks.
- Rebuild CNP contexts.
- Resample final target-task clusters.
- Recompute the full allocator.

With 12 final policies, policy-bootstrap intervals are too unstable for a broad
policy-population claim. Final inference is principally about the fixed held-out
menu.

## 14. Phased Spend Plan

```text
Phase 0  Validate CNP on synthetic policy outcomes           (no native spend)
Phase 1  Headroom pilot: 4 policies x 12 tasks x R=2 = 96 attempts (stop gate)
Phase 2  Method pilot: ~12 train + 4 dev policies, ~300-350 attempts
Phase 3  Frozen development: 48 meta-train + 12 dev policies, full rows
Phase 4  Final held-out: 12 policies, calibration + complete panel, once
Phase 5  Repeat expansion only per predeclared discordance rule
```

### 14.1 Minimum native attempt budget

```text
Meta-training (balanced incomplete block):
    48 policies x 32 unique tasks = 1,536 attempts
    (preferred 48 x 48 = 2,304)

Final calibration: 12 x 16 = 192
Final complete panel: 12 x 72 = 864
25% repeat audit: 216

Minimum before development overhead: ~2,808 attempts
```

These are expensive because policy count, not episode count, determines
meta-learning evidence. Replaying the same rows across training episodes does
not create new policy samples.

## 15. Claim Boundaries

### 15.1 What M3 may claim

Under the frozen harness and calibration protocol, the allocator ranked and
selected among held-out combinations of the Unbrowser policy grammar on
deterministic fixture tasks at a measured output-token cost.

### 15.2 What M3 may not claim

```text
arbitrary natural-language policy generalization
transfer to unseen tools, interfaces, or grammar levels
public-web or live-site effectiveness
sequential per-step dynamic control
causal effects of individual budget components (only tool_cap varies)
universal calibrated Bayesian posterior intervals
```

### 15.3 Live canaries

Public-site canaries (Wikipedia or others) validate operational compatibility
only. They are excluded from training, model selection, calibration, and all
effectiveness evaluation.

## 16. Implementation Changes Required

```text
new 72-cell Unbrowser policy grammar generator
export structured grammar factors into model_input
add policy split, calibration role/order, rollout-replica fields to dataset
add calibration context builder with leakage audit
add attentive CNP training and inference path
add separate cost prediction head
add success-cost frontier evaluator with frozen lambda grid
add policy/task-clustered bootstrap evaluator
preserve current hybrid model and hierarchical residual as baselines
add frozen negative-control (shuffled context) evaluator
```

## 17. Revision Log

| Date | Change | Reason |
|---|---|---|
| 2026-08-10 | Created the initial pre-canary 72-policy registry at hash `4cfd05631af3907d42b5a6c64cfca5205fcd1aef3a583886e7ccccc9dd5bcfe7` and split manifest `edd169f46b71ee17cbe771a9a16cc1353946ba37435e9d049849ee3f1dea668d`; this version is superseded and excluded. | Record policy identities before the operational v0.0.19 canary. |
| 2026-08-10 | After the excluded canary and pre-pilot semantic audit, bumped the fixture generator/verifier to v2, removed generic nonce shortcuts, made the workflow state-bearing, removed comparison labels that revealed the winner, added fixture-specific Pi instructions, made planning markers measurable, and froze grammar `m3-v2` / policy version `2`. Final registry hash: `2d3b6c3d956fed9d255782bef264f6333129e803fc6853b1fcebb4486a8a2d3f`; balanced split manifest hash: `b8853ae708b2e1943c2097ae546c47ad046a1a0c2769ccc55bba9a1b6510485f`; split seed remains `20260810`. | Close validity and instrumentation defects before any headroom-pilot outcome is observed; all v1 canaries remain excluded. |
| 2026-08-10 | Replaced the unreplicated 24-task screening layout before any v2 outcome with 12 tasks × 4 policies × 2 rollout replicas. Added cross-replica lift, repeat-concordance, stable-disagreement, planning-adherence, deterministic recovery probes, per-template execution-position balance, interleaved execution, explicit `T_pilot` exclusion, attempt replica metadata, and pinned `thinking=off`. Frozen headroom manifest hash: `d5d22bb742ed91cfc58c1c224ef4fdda1aae9f73967a4f23a8201ad1c62b4ad6`. | Prevent stochastic rollout noise from manufacturing disagreement/oracle lift and remove post-outcome discretion from manipulation eligibility and pilot ordering. |
| 2026-08-10 | Superseded the still-unused `d5d22bb…` headroom manifest before any v2 canary or pilot outcome. Pinned `@earendil-works/pi-coding-agent` `0.84.1` and CLI SHA-256 `840d1e8e689ed9e4937bcb00b9a810e02a8567d9afb10a47097f11ca93ea1521`; made executed-policy, verifier, and trajectory identity structural gate requirements; and added a durable active-panel marker that blocks resume after an interrupted panel. Then-current headroom manifest hash: `e466bbf8282f1e12f2dbd460f4dba8f8dc9536057415bebe99ae69e80d73729f`. | Close runtime-identity and partial-panel resume paths before collecting evidence. |
| 2026-08-10 | Superseded all still-unused intermediate headroom manifests before any v2 canary or pilot outcome. Excluded the two forced-recovery tasks from outcome/cost headroom; moved headroom thresholds to 2/10 neutral stable-disagreement tasks and 1/10 uniform-tie cross-replica lift; made recovery eligibility exact and task-specific; required planning/observation adherence per factor level and verification opportunities; assigned 24 explicit panel seeds with fixed sampling parameters; counterbalanced replica chronology 6/6; captured sanitized outgoing-request receipts; and bound the Pi provider endpoint to exact remote executable/model paths and hashes. Current headroom manifest hash: `e7f257c48548245b3aaba965c81bff4e548c06fca06b16c84e10eadf2daf0931`. | Prevent designed recovery behavior, label order, provider defaults, endpoint drift, or pooled manipulation rates from manufacturing a pilot pass. |
| 2026-08-10 | Ran the single permitted v2 operational canary outside all pilot coordinates: `T_canary`, `distractor_recovery`, task seed `990000001`, sampling seed `1900000001`, policy `ub-direct-structure_first-submit_directly-diagnose_retry_once-expanded@2-a18fd8c7`. Verification passed with output-token cost `2626`, 11 admitted calls, exact HTTP-503 probe detection, Unbrowser `0.0.19`, and the frozen sampling receipt. Export produced exactly one `canary_excluded` row with 32-D task features, 13-D treatment features, and no leakage violations. Adherence truthfully recorded planning/probe/submit/cap as passing and structure-first/immediate-retry as failing. No frozen threshold or pilot coordinate changed after observing it. | Validate operational plumbing and manipulation instrumentation only; this row is permanently inadmissible as headroom or allocator-effectiveness evidence. |

No revisions are permitted after the final-policy stage is unlocked.

## 18. Plain-Language Reference

This section explains the jargon used above in concrete terms. It is reference
material; the formal spec is Sections 1–17.

### 18.1 The goal in one sentence

We want to teach a system to pick the right instruction set for the AI agent on
each web task — and to do that even for instruction sets it has never seen
before, by letting it practice on a few warm-up tasks first.

### 18.2 What "policy" means

A policy is a frozen recipe of instructions given to Gemma before it starts a
task. It is a system prompt plus a tool budget. Example of one policy's prompt:

> Plan before acting. Read the page text first. Re-check the page before
> submitting your answer. If a step fails, try once more. You have 12 actions.

That entire bundle is one policy.

### 18.3 What "policy grammar" means

"Grammar" is the menu of choices. Think of it like building a burger: each row
is a dimension, and you pick one option per row.

| Dimension | Options |
|---|---|
| Planning | just-do-it / plan-first / break-into-steps |
| Observation | read-text / inspect-structure / search-directly |
| Verification | submit-immediately / re-check-before-submit |
| Recovery | give-up-on-fail / retry-once |
| Tool budget | 6 actions / 12 actions |

One pick from each row is one policy. All possible picks equals 72 policies.

### 18.4 What "combination of which" means

A policy such as `plan-first + read-text + re-check + retry + 12-actions` is one
combination. M3 holds out some combinations from training. "Unseen combination"
means the system has seen each ingredient before (it knows what plan-first
tends to do, what retry tends to do) but has never seen that exact bundle. The
research question is: can it predict how the bundle performs?

### 18.5 What "allocation" means

Allocation means choosing which policy to give the agent for a given task.

- Task A might be "extract one number from a table" — a lean, fast policy wins.
- Task B might be "compare info across three pages" — a plan-first, bigger-budget
  policy wins.

The allocator is the model that looks at the task and picks the policy.

### 18.6 What "few-shot calibration" means

Before the allocator ranks a never-seen policy, it runs that policy on a small
number of warm-up tasks. Those results are the few-shot calibration. The
allocator uses them to estimate: "this policy is generally decent at reading
tasks but burns tokens on multi-page tasks."

`k=8` means 8 warm-up examples. It is the primary condition.

### 18.7 What "success-cost frontier" means

Two things matter on each task:

- Success: did it get the right answer?
- Cost: how many tokens did it burn?

A policy that gets 100 percent success by spending 10x tokens is not always
better. The frontier is the curve of best success at each cost level. We want
the allocator to push that curve up and to the left: more success, less cost.

### 18.8 What "meta-training" means

We train the allocator on 48 policies we have seen, using thousands of task
attempts. The goal is to teach it the general skill of "look at a policy recipe
plus a few practice results, then predict how it will do." Then we test that
skill on 12 policies it never trained on.

### 18.9 What the meta-policy learner targets

There are two levels.

**Training target (what the model optimizes during meta-training):** Given a
policy descriptor plus k calibration outcomes, predict verified success and
token cost on a held-out task for that same policy. The model learns the
transferable skill: "given a recipe and a few examples of how it behaves,
predict its behavior on a new task." It is not memorizing "policy X is good"; it
is learning how to rapidly estimate any policy's task-conditional performance.

**Downstream target (what we use it for):** After meta-training, use the learned
predictor to rank policies globally and select the best policy per task,
maximizing success minus lambda times cost. The allocator's target is the
success-cost frontier.

The word "meta" is the key: the learner's target is the adaptation skill itself,
not any single policy's quality.

### 18.10 The whole thing in one example

1. A new policy arrives: `break-into-steps + search-directly + submit-immediately + give-up + 6-actions`.
2. We run it on 8 warm-up tasks. It got 5 out of 8 right and burned few tokens.
3. The allocator predicts: "this policy is fast but shaky; use it on simple
   extraction tasks, not multi-page comparison."
4. On 72 real tasks, it picks this policy for the easy ones and a sturdier
   policy for the hard ones.
5. M3 passes if that smart picking beats both "always use one fixed policy" and
   "pick randomly" on the success-cost curve.

### 18.11 What M3 does not claim

It cannot handle a totally new instruction never put in the menu, such as "use a
brand-new tool" or "write your answer in Chinese." It only interpolates among
ingredients it has already seen.
