# M3 Utility-Routing Smoke Plan

Date: 2026-08-13

Status: **prospective staged plan; no routing-smoke outcome may be inspected
before the corresponding manifest and gate are frozen**

## 1. Decision And Claim Boundary

The next experiment tests whether a controller-owned, pre-treatment structural
probe can support practical closed-set routing between the frozen table and form
semantic specialists.

It does not amend the valid semantic `replication_no_go`, establish strict
capability isolation, unlock the original 72-policy M3 Phase 2, validate public
websites, or establish transfer to an unseen policy. The existing semantic rows
remain `T_canary` / `canary_excluded` and are used only to motivate this new
prospective design.

The practical decision is:

```text
selected_policy(task) = argmax_policy utility(task, policy)

utility = predicted_success - lambda * predicted_output_cost_units
predicted_output_cost_units = predicted_output_tokens / 10,000
```

Primary `lambda = 1.0`, equivalent to `0.0001` per raw output token. One unit of
predicted success is therefore exchangeable with 10,000 generated output
tokens. Frozen sensitivity values are `0.0`, `0.25`, `0.5`, `1.0`, and `2.0`.
Only `lambda = 1.0` determines the smoke decision.

Ties are resolved by higher predicted success, then lower predicted output
cost, then immutable registry order. Observed success or cost must never enter
selection.

## 2. Hypotheses

### H-M: Probe Mechanics

The neutral probe is deterministic, bounded, treatment-independent, available
before assignment, and free of page content or private task information.

### H-R: Prospective Routing

A frozen request-plus-probe router chooses the useful initial specialist on
pure, mixed, and misleading-cue tasks more reliably and efficiently than either
fixed specialist.

### H-U: Learned Utility

After a prospective corpus exists, separate success and output-cost predictions
can select policies with higher held-out utility than the best fixed specialist
and the frozen routing heuristic.

H-M and H-R authorize corpus collection. H-U is evaluated only on the untouched
test split of that corpus.

Stage B identifies the value of the frozen request-plus-probe routing rule. It
does not, by itself, identify the probe's incremental contribution because the
public request is also legitimate routing input. Combined-minus-prompt-only
success and utility are therefore frozen secondary reports, not Stage-B pass
criteria.

## 3. Neutral Probe Contract

The controller obtains the initial public HTML before starting Pi or assigning a
treatment. It parses the bytes with a deterministic stdlib parser and exposes
only bounded integers or booleans:

```text
element_count
max_dom_depth
table_count
table_row_count
table_cell_count
max_table_columns
form_count
control_count
required_control_count
get_form_count
post_form_count
text_input_count
select_count
textarea_count
button_count
anchor_count
```

All counts are capped by the schema before entering model input. The audit-only
receipt contains the probe schema/mechanism, source byte count, source HTML
SHA-256, canonical feature SHA-256, and delivery status.

The model-visible object must not contain page text, labels, names, values,
headers, IDs, classes, URLs, domains, form actions, links, selectors, element
references, source hashes, task seed, template ID, difficulty label, route
label, oracle, expected answer, verifier data, treatment identity, or any
post-action field.

The task request remains legitimate predecision input. Frozen request features
may describe declared operations, but they are derived from the public contract
without reading the fixture oracle or design label. Prompt-only and probe-only
baselines are reported separately.

The probe never mutates page state, invokes a semantic specialist, submits a
form, or consumes model output tokens. Probe latency and bytes are reported as
secondary costs.

## 4. Task Strata

Fresh routing fixtures use one versioned template family with private design
labels that never enter model input:

1. `pure_table`: a relevant table operation with a bounded irrelevant form cue.
2. `pure_form`: a relevant validated form with a bounded irrelevant table cue.
3. `mixed`: both operations are required, balanced between table-first and
   form-first dependency order.
4. `ambiguous`: the larger or more numerous visible structure is irrelevant;
   the requested bottleneck is the opposite capability.

Every stratum contains balanced table-preferred and form-preferred tasks across
easy, medium, and hard structural complexity. The preferred specialist means
the capability expected to reduce the first required bottleneck. Both policies
retain generic Unbrowser fallback, so failure of the nonmatching policy is not a
requirement.

## 5. Stage A: Zero-Outcome Probe Gate

Stage A creates 32 fixtures, eight per stratum, and performs three cold probe
replays per fixture. It also runs synthetic utility matrices covering dominance,
success-cost tradeoffs, ties, missing/non-finite predictions, and every frozen
lambda.

Stage A passes only if all hold:

- 96/96 probe receipts are valid;
- each fixture's model-visible features are byte-identical across three replays;
- every count is within its frozen bound;
- the privacy/provenance audit has zero violations;
- changing only visible text while preserving DOM structure leaves features
  identical and changes the source hash;
- probe execution precedes treatment assignment by construction;
- the frozen router agrees with the sealed first-bottleneck label on at least
  28/32 fixtures and at least 6/8 in each stratum;
- prompt-only, probe-only, and combined heuristic results are all reported; and
- all synthetic utility and tie-break checks pass.

Any failed check is `probe_no_go`. No agent outcome smoke or corpus execution is
authorized.

## 6. Stage B: Excluded Crossed Outcome Smoke

After Stage A passes, freeze 24 new tasks in two independently seeded blocks.
Each block contains three tasks from each stratum and balances preferred
specialist, difficulty, replica chronology, and arm execution position.

Run both frozen specialists on every task with two independent replicas:

```text
24 tasks x 2 policies x 2 replicas = 96 attempts
```

The same initial public HTML hash and probe feature hash must bind all four
attempts for one task. Every row is permanently `T_canary` /
`canary_excluded`. No outcome from this smoke may train or tune the later model.
The routed arm is reconstructed from the frozen pre-outcome routing receipt and
the complete crossed panel; it is not executed as a third arm.

Immediately before each panel, the controller generates the frozen task and
runs a model-free remote commitment command. That command starts the
harness-owned fixture server itself, fetches the exact opaque routing URL, and
recomputes the source HTML, probe-feature, and probe-receipt hashes. All three
must match the manifest before treatment execution. A port collision, changed
URL/status, malformed commitment, or hash drift fails closed.

For each task-policy cell, smoke scoring uses the replica mean:

```text
observed_utility = mean(success) - lambda * mean(output_tokens / 10,000)
```

Probe output-token cost is zero. Probe latency and transferred bytes are added
to reporting, not the primary utility.

Stage B passes only if all hold:

- 96/96 planned attempts are present with exact identities and valid sampling
  receipts;
- zero infrastructure, structural, mechanism, probe-hash, or verifier errors;
- no selective reruns, early outcome stopping, or outcome-driven replacement;
- every task has a complete two-policy, two-replica panel;
- repeat-discordant policy-task cells are at most 12/48 overall and at most 7/24
  in either block;
- routed verified success is at least 80%;
- routed success exceeds the hindsight-better fixed specialist by at least 10
  percentage points;
- primary pooled utility lift over the hindsight-better fixed specialist is at
  least 0.08 and its one-sided 90% task-cluster bootstrap lower bound is above
  zero;
- utility lift is at least 0.04 in each independent block;
- utility lift is nonnegative at sensitivity lambdas `0.5` and `2.0`; and
- mixed and ambiguous strata each have nonnegative primary utility lift.

For every comparison, "hindsight-better fixed specialist" means one specialist
used for every task in the relevant evaluation subset, selected after observing
that subset. It never means a per-task hindsight oracle, which a router could
not exceed. Success and utility comparators are selected independently. Ties
use higher success, then lower output-token cost, then immutable registry order.
The pooled comparison uses all 24 tasks; each block, stratum, and sensitivity
comparison recomputes the best fixed specialist within that declared subset.

The primary confidence bound uses 100,000 deterministic task-cluster bootstrap
draws with seed `2026081302`. Each draw resamples three whole tasks with
replacement inside each `block x stratum` cell, preserving both policies, both
replicas, and the frozen route. The best fixed utility specialist is recomputed
inside every draw. The one-sided 90% lower bound is the nearest-rank empirical
10th percentile and must be strictly above zero.

Attempt classification is frozen before execution:

- a verifier-returned false result, malformed model answer, refusal, admitted
  tool misuse, model budget exhaustion, or wall-time exhaustion with a valid
  record is an intention-to-treat failure and is never rerun;
- an explicit controller, provider-transport, or browser-transport failure
  before a valid attempt record is `infrastructure_invalid`;
- probe/hash/order failures are `probe_invalid`;
- wrong treatment/interface or invalid specialist receipts are
  `mechanism_invalid`;
- verifier crashes or verifier identity mismatches are `verifier_invalid`;
- manifest, schedule, sampling, identity, completeness, duplicate, governance,
  and artifact-hash failures are `protocol_invalid`; and
- unclassified failures fail closed as `unclassified_invalid`.

Only `infrastructure_invalid` can activate a replacement, and only through a
complete contingency block frozen before the first primary attempt. The
contingency retains the same task HTML, oracle, route, policies, and block
identity while using pre-frozen fresh attempt IDs, sampling seeds, chronology,
and arm positions. All primary-block attempts are then quarantined, the whole
contingency block is run once, and no second replacement is permitted.

Live chronology is also frozen. The controller walks the global primary panel
chronology once, omitting the rest of a block after its first explicit
infrastructure trigger while continuing the other primary block. After primary
chronology is exhausted, activated blocks run in the global contingency
chronology. The replacement receipt binds the trigger's attempt ID, controller
phase, and raw-record hash. Resumes must replay this exact state machine; an
unfinished active-panel marker or out-of-order raw ledger requires adjudication
and never silently reruns an uncertain attempt ID.

Execution requires a fresh clean-worktree runtime preflight on every start or
resume, with exact local/remote source-tree parity and frozen Pi, model,
llama-server, Unbrowser, Bubblewrap, provider, and remote-path identities. Raw
panel records are append-only and restricted. A separate whitelist-only export
emits the 96 selected safe rows, binds each to its raw-record hash, and excludes
stderr, model text/thinking, tool arguments/payloads, verifier diagnostics,
HTML, oracle data, and quarantined primary outcomes.

Pooling cannot rescue a negative block. A mechanically valid failure is
`routing_smoke_no_go`; the full corpus is not run.

An infrastructure-invalid block may be rerun once only as a complete block with
fresh seeds frozen before execution. Model or task failures with valid records
remain intention-to-treat failures and are never rerun.

## 7. Full Outcome-Model Corpus

The full corpus is authorized only by byte-verified Stage A and Stage B pass
reports. Before execution, freeze and hash the generator, archetypes, policy
bundles, probe schema, router, verifier, cost definition, lambda grid, split
manifest, exclusion rules, model specifications, and dataset contract.

The target corpus is:

```text
4 strata x 10 archetypes x 8 tasks = 320 independent tasks
320 tasks x 2 policies x 2 replicas = 1,280 attempts
```

Split whole archetypes, not individual rows: six train, two validation, and two
test archetypes per stratum. This yields 192 train, 64 validation, and 64 test
tasks. Every policy and replica for one task stays in one split. Test outcomes
remain sealed until all model choices and thresholds are frozen.

The approved execution ceiling is `1,280 x Stage-B P90 output tokens per
attempt`. A projection above 120% of that ceiling stops and invalidates the
complete corpus rather than silently truncating expensive cells.

The verified package must contain:

- one leakage-safe row per attempt with `train`, `validation`, or `test` split;
- complete task-policy-replica panels;
- model-visible request features, structural probe features, and a structured
  capability descriptor;
- top-level verified success, output-token cost, termination, provenance, probe
  receipt hash, treatment/registry identity, and sampling identity;
- an inclusion ledger, immutable schedule, source/file hashes, privacy audit,
  and explicit absence of excluded smoke rows; and
- a deterministic package verifier and byte-identical rebuild test.

## 8. Model And Final Gate

Fit separate predictors from predecision features only:

```text
p_hat(task, probe, policy) -> verified success probability
c_hat(task, probe, policy) -> expected output tokens
```

Required baselines are best fixed specialist, uniform random, prompt-only
lookup, probe-only lookup, combined frozen heuristic, structured logistic/ridge,
and per-policy mean cost. The initial neural outcome model is optional until a
simple model demonstrates held-out signal.

The untouched test passes only if the learned utility allocator:

- improves primary utility by at least 0.03 over both the best fixed specialist
  and frozen heuristic, with one-sided 95% task-cluster bootstrap lower bound
  above zero;
- has success no more than 0.02 below the strongest baseline;
- has nonnegative utility lift separately on mixed and ambiguous strata;
- has success Brier no worse than the best declared baseline by more than 0.005
  and ECE at most 0.10; and
- reduces cost MAE by at least 10% versus the train-fitted per-policy mean-cost
  baseline.

If the frozen heuristic matches or beats the learned allocator, the heuristic is
the practical result and model complexity has not earned deployment.

## 9. Implementation Sequence

1. Implement and test the pure structural parser, receipt, privacy audit, and
   exact utility/tie-break function.
2. Implement the versioned routing fixture strata and sealed design labels.
3. Implement immutable Stage-A manifest, validator, analyzer, and gate.
4. Run Stage A locally without model execution.
5. Only on Stage-A pass, freeze and run the 96-attempt excluded Stage B panel.
6. Build and verify its aggregate report and private package.
7. Only on Stage-B pass, freeze the full corpus manifest and execute all 1,280
   attempts without outcome peeking.
8. Build and verify the outcome-model dataset, then train baselines before any
   neural architecture expansion.
