# Policy Lab Technical Design

Date: 2026-08-14

Status: **design confirmed; implementation not started; no autonomous live
search or sealed-test execution is authorized by this document**

## 1. Decision And Claim Boundary

The Policy Lab is a reusable control plane for drafting, compiling, probing,
and selecting one universal agent policy across the harness's supported task
families.

The selected design has these fixed properties:

- one universal strategy rather than one prompt per task family;
- separate strategy and execution layers combined into one treatment;
- versioned capability contracts visible to both the agent and guardrails;
- joint Bayesian search over named strategy and execution factors;
- success-first promotion, with cost considered only among success-qualified
  candidates;
- autonomous search within a frozen budget;
- an automatic, one-time sealed final test;
- frozen task banks split by whole archetype; and
- support for all current task families, while only benchmark-ready families
  may influence search or promotion.

"Universal" has two separate claims:

- **Compilation and deployment:** every candidate must compile for every
  non-deprecated supported family, and must pass a non-decision smoke check in
  each family before it may enter search.
- **Efficacy:** the terminal success claim applies only to the exact eligible
  final-bank target mixture bound by the charter.

The final estimand is finite-bank performance, with family, archetype, and
difficulty weights frozen in the final-bank manifest. Exploration and
validation archetypes support candidate selection but are not part of the
terminal efficacy claim. No result is interpreted as inference to unregistered
or future archetypes.

Adding a task family, capability, factor, contract version, or target mixture
starts a new search generation. It does not imply transfer to arbitrary future
tasks or tools.

This design does not authorize reuse of M3 canary outcomes, retrospective
promotion from existing exploratory runs, arbitrary prompt rewriting, or a
large live search before benchmark-readiness and simulator gates pass.

## 2. Design Principles

1. **Behavior before labels.** A policy factor enters search only when its
   behavior is observable, enforceable, or both.
2. **Success before efficiency.** Cheap failure never outranks expensive
   success.
3. **Intention to treat.** Valid non-adherent attempts remain outcome evidence;
   analysis never conditions success on observed adherence.
4. **No pooling rescue.** Overall strength cannot hide failure in one eligible
   family or required archetype.
5. **Outcome-blind freezing.** Candidate spaces, banks, priors, acquisition,
   budgets, thresholds, and final-test rules are immutable before their
   outcomes become visible.
6. **Append-only evidence.** Every candidate, assignment, attempt, invalidity,
   posterior update, rejection, and decision remains auditable.
7. **One-way sealing.** Final-test outcomes cannot return to search.
8. **Fail closed.** Unknown states, incompatible capabilities, hash drift,
   uncertain attempts, and unclassified failures stop progression.

## 3. Existing Foundations

The implementation should generalize existing mechanisms rather than create a
parallel harness.

| Need | Existing foundation | Reuse decision |
|---|---|---|
| Task and attempt wire types | `contracts.py` | Preserve existing types; add Policy Lab types alongside them. |
| Family generation and verification | `gym_registry.py` | Keep as execution dispatch; add family contracts above it. |
| Immutable treatment identity | `treatments.py` | Reuse canonical payloads, bundle hashes, and registries. |
| Deterministic clause composition | `meta_grammar.py` | Reuse its clause-library and prompt-composition pattern. |
| Registered treatment execution | `orchestrator.py` | Keep as the atomic live runner. |
| Sequential batches and resume | `batch.py` | Reuse job expansion and atomic append semantics. |
| Safe predecision exports | `dataset.py` | Extend with bank and evidence bindings; do not weaken leakage rules. |
| Outcome modeling utilities | `outcome_model.py` | Reuse CPU PyTorch conventions and metrics, not the V1 decision model. |
| Calibration leakage audits | `calibration.py` | Reuse forbidden-field and deterministic split utilities where applicable. |
| Complete-panel evaluation | `allocator_eval.py` | Reuse strict pairing and held-out comparison patterns. |
| Frozen manifests and gates | M3 modules | Promote proven self-hash, contingency, safe-export, and gate patterns. |

The existing neural outcome model and MetaCNP remain research components. They
are not the V1 autonomous search posterior because current real evidence is too
sparse to support their larger representation spaces reliably.

## 4. System Architecture

```text
                    committed control plane

  family contracts ---- capability contracts ---- runtime pins
          |                     |                       |
          +---------- compatibility resolver ----------+
                                |
  strategy card + execution profile + factor assignment
                                |
                    deterministic compiler
                                |
                       treatment candidate
                                |
            search charter + frozen task banks
                                |
                    Bayesian batch planner
                                |
                 frozen candidate-task batch
                                |
            orchestrator -> verifier -> safe export
                                |
                     append-only evidence
                                |
                       posterior snapshot
                                |
                  stop / continue / validate
                                |
             one-time validation -> sealed final
```

The control plane runs locally. Remote workers continue to generate tasks,
prepare workspaces, execute Pi, record events, and verify attempts through the
existing CLI and orchestrator boundaries.

## 5. Module Boundaries

V1 should use a small set of cohesive modules.

| Proposed module | Responsibility |
|---|---|
| `policy_lab_contracts.py` | Typed schemas, canonical serialization, self-hashes, the safe-row whitelist, and the sole forbidden-field registry. |
| `policy_lab_compiler.py` | Capability compatibility, policy compilation, visible contract text, and compile receipts. |
| `policy_lab_banks.py` | Immutable task banks, role separation, private-oracle commitments, and consumption. |
| `policy_lab_ledger.py` | Hash-chained append-only records, active markers, replay, reconciliation, and safe exports. |
| `policy_lab_model.py` | Design matrix, MAP fit, Laplace posterior, fixed-packet behavior models, cost model, and snapshots. |
| `policy_lab_search.py` | Search state machine, warm-up design, acquisition, scheduling, stopping, and resume. |
| `policy_lab_gate.py` | Validation gate, automatic final trigger, sealed evaluation, and decision reports. |
| `policy_lab_cli.py` | Controller-facing commands that delegate to the modules above. |

Do not move the remote execution surface in V1. `cli.py`, `worker.py`, family
gyms, verifiers, and `orchestrator.run_registered_treatments` remain the live
execution primitives.

## 6. Core Contracts

All contracts use strict JSON types, schema versions, canonical key ordering,
SHA-256 self-hashes, immutable-write semantics, and explicit validation.

### 6.1 Family Contract

`FamilyContract` declares whether and how one family participates.

Required fields:

- `family_id`, `version`, and `maturity`;
- generator module, version, and source hash;
- verifier ID, version, source hash, and execution safety class;
- generator mode: `procedural`, `finite_frozen`, or `live_fixed`;
- supported archetypes and family-local difficulty levels;
- required capability names and compatible execution modes;
- workspace mutation scope and required output artifacts;
- network, platform, sandbox, reset, and stateful-session requirements;
- outcome, mechanism, infrastructure, and timeout taxonomies;
- trace and runtime-commitment requirements; and
- `search_eligibility`: `eligible`, `diagnostic_only`, `smoke_only`,
  `consumed`, or `deprecated`.

`maturity` and `search_eligibility` are governance fields. The Bayesian model
must not merely assign zero objective weight to ineligible families; their rows
must be absent from the decision fit so they cannot alter shared coefficients.

### 6.2 Capability Contract

`CapabilityContract` is the single source for agent-visible guidance and
guardrail enforcement.

Required fields:

- capability name, version, and safety class;
- compatible tool interface and required tool set;
- applicability signals available before action;
- required and optional input fields with JSON types;
- boundedness rules and maximum request shape;
- output shape and sufficient-result evidence;
- recoverable failures and allowed corrections;
- non-recoverable failures and forbidden uses;
- fallback compatibility;
- expected cost class; and
- machine fields from which concise model-visible text is generated.

Capability text may not contain task answers, worked solutions, task-specific
identifiers, or hidden evaluation details. A contract-text audit enforces size,
field provenance, and forbidden patterns. Handwritten visible text separate
from machine enforcement is not allowed.

Each capability also carries a predecision opportunity predicate derived only
from task-bank and contract fields. Adherence and intervention denominators use
this predicate, never whether the agent happened to reach a tool-call state.

### 6.3 Strategy Card

`StrategyCard` describes the universal domain-neutral operating method.

Required factors include:

- task interpretation depth;
- planning depth;
- capability-selection rule;
- first-action rule;
- result-validation rule;
- recovery rule;
- fallback rule;
- stopping rule; and
- submission rule.

Each factor references a committed clause ID. V1 does not search arbitrary free
text. A clause library deterministically renders the strategy prompt.

### 6.4 Execution Profile

`ExecutionProfile` describes runtime behavior independently of strategy text.

Required factors include:

- capability exposure rule;
- input enforcement mode: warn or reject;
- maximum corrections;
- maximum fallbacks;
- finishing-action reservation;
- tool-call, output-token, command-time, and wall-time limits;
- budget enforcement mode;
- duplicate-action rejection; and
- sandbox and network profile.

Every factor level carries an `evidence_mode` of `enforced`, `observable`, or
`both`, plus the enforcement-adapter and observation-schema hashes. A factor
whose runtime adapter does not yet exist cannot enter the candidate menu. The
initial menu is restricted to factors proven enforceable or observable by the
current Pi extension and worker stack; new guardrails are implementation
prerequisites, not aspirational metadata.

An `ObservationAdapter` binds a predecision opportunity predicate, immutable
event fields, a deterministic behavior classifier, explicit `unknown`
semantics, and an observation-receipt schema. Before a factor is marked
observable, model-free fixtures must prove that its adapter distinguishes the
declared levels without reading outcome or private task fields.

The charter also freezes each factor's endpoint mapping. `observable` levels
contribute to voluntary-adherence endpoints, `enforced` levels contribute only
to intervention endpoints, and `both` levels contribute to both. The adapter
runs inside the restricted raw-to-safe boundary: it may inspect approved raw
trace fields, including tool routing fields, but emits only bounded labels and
receipt hashes. Raw fields never become predictors or safe-export columns.

### 6.5 Universal Candidate

`UniversalCandidate` binds:

- one strategy-card hash;
- one execution-profile hash;
- one capability-registry hash;
- one family-registry hash;
- factor values and declared interactions;
- compiler version and source hash; and
- a candidate content hash.

The universal candidate has one parent identity. Compilation for a family
produces a family-resolved `CompiledTreatment` whose hash additionally binds
the compatible capabilities, generated system prompt, active tool schemas,
Pi base-prompt hash, command arguments, environment pins, and runtime profile.

Every candidate must compile for every eligible family. Candidate-specific
incompatibility eliminates the candidate globally rather than removing its
difficult cells. It must also compile and pass non-decision mechanics checks in
every non-deprecated supported family. Only candidate-independent bank defects
may remove a task from the target population.

Every treatment-varying capability, tool, registry, prompt, or runtime choice
must be represented by a frozen factor column, and candidate factor vectors
must be unique. Registry hashes may bind provenance but cannot create an
unmodeled treatment difference between otherwise identical vectors.

### 6.6 Search Charter

`SearchCharter` is the authorization boundary for autonomous spend. It binds:

- eligible and diagnostic family-contract hashes;
- capability-registry and clause-library hashes;
- candidate factor levels, enforcement/observation adapter hashes, and allowed
  interactions;
- baseline candidate and finite feasible candidate menu;
- exploration, validation, final, and contingency bank hashes;
- search-standardization weights and the finite final-bank estimand;
- runtime, provider, model, sampling, and source-tree pins;
- warm-up design, screening and promotion coverage matrices, all three
  acquisition allocators, quotas, behavior-audit packet schema, and random
  seeds;
- prior scales, block-dimension and per-task candidate caps, posterior draws,
  nested-integration draws, and numerical tolerances;
- success, finalist-equivalence, family/archetype noninferiority, behavior error
  budget, adherence, intervention, stability, and cost thresholds;
- attempt, output-token, elapsed-time, batch, candidate, and ledger-record
  budgets;
- infrastructure replacement and invalidity rules;
- stopping, futility, complete validation/final decision tables, process-access
  manifests, and automatic-final rules; and
- safe-export and final-report schemas.

The search process can consume the charter but cannot rewrite or supersede it.
A changed charter creates a new search ID and no prior outcome is silently
carried over.

The clause library, factor menu, compiler, priors, and statistical protocol are
committed before validation or final task coordinates, prompts, seeds, or
archetype identities are available to candidate designers or the acquisition
process. A bank-custodian receipt records this ordering and all access.

### 6.7 Compiled Treatment And Receipt

`CompiledTreatment` is the only live-runner input accepted from the Policy Lab.
It contains:

- schema version and compiled-treatment hash;
- universal-candidate and factor-assignment hashes;
- family, capability-registry, clause-library, and compiler hashes;
- generated system-prompt bytes and SHA-256;
- Pi base-prompt, extension, and tool-schema hashes;
- ordered active tools and exact interface identity;
- strategy and guardrail receipts;
- output, tool-call, correction, fallback, command, and wall-time limits;
- sandbox, network, provider, model, sampling, and runtime pins;
- exact ordered command arguments and approved environment variables; and
- source-tree and dependency hashes.

The compile receipt repeats the input hashes, output hash, and deterministic
compiler identity, then self-hashes. Recompilation in a fresh process must
produce byte-identical treatment and receipt bytes.

## 7. Compatibility And Compilation

Compatibility is checked before task generation or attempt reservation.

```text
family required capabilities
  subset of candidate-resolvable capabilities
  subset of runtime-supported capabilities
```

The resolver also checks platform, network, sandbox, state-reset, verifier, and
output-artifact requirements. A mismatch in an eligible family eliminates the
candidate globally. A mismatch caused by a family or bank defect blocks that
family's readiness rather than becoming an outcome. No candidate may improve
its comparison population by being incompatible with hard tasks.

Compilation is pure and deterministic:

```text
compile(strategy, execution, family, capability registry, runtime pins)
  -> CompiledTreatment
  -> compile receipt
```

The receipt includes every model-visible prompt byte, tool schema, CLI
argument, environment pin, budget, and source hash. Recompiling the same inputs
must produce byte-identical output. The live runner accepts only a validated
compile receipt and verifies it remotely before starting Pi.

## 8. Task Banks

### 8.1 Roles

Each task belongs permanently to exactly one role:

- `exploration`: adaptive acquisition and posterior updates;
- `validation`: one complete finalist-versus-baseline evaluation;
- `final`: one automatic sealed evaluation;
- `diagnostic`: mechanics only, never decision evidence;
- `smoke`: operational connectivity only; or
- `excluded`: prior pilot, canary, or consumed research data.

Roles split whole archetypes, not individual generated instances. Every policy,
replica, and contingency cell for a task stays in the same role.

`BankRole` remains distinct from the existing dataset split field. Safe export
uses this explicit mapping:

| Bank role | Dataset split | Selection eligibility |
|---|---|---|
| `exploration` | `train` | Adaptive decision evidence. |
| `validation` | `validation` | One-time validation evidence only. |
| `final` | `final_sealed` | One-time terminal evidence only; rejected by existing train/evaluate loaders. |
| `diagnostic`, `smoke`, `excluded` | dedicated excluded split | Never decision evidence. |

An unrecognized role or split mapping fails closed. Existing canary and pilot
exclusions remain exclusions and are never remapped to exploration.
The Policy Lab exporter assigns this mapping from the custodian-bound role and
does not call the existing hash-based `task_split`. Existing `test` rows are
legacy evidence and map only to exclusion; they are never validation or final
rows. `final_sealed` is accepted only by the sealed final loader, while existing
training and evaluation loaders must reject it explicitly.

### 8.2 Manifest

`TaskBankManifest` binds:

- bank ID, version, role, and creation time;
- family, archetype, seed, difficulty, and task coordinates;
- generator/source/runtime commitments;
- public task and initial-workspace hashes;
- private verifier/oracle commitment hashes;
- attempt and contingency schedules;
- exposure and consumption state; and
- a bank manifest self-hash.

The public bank view excludes private answers, verifier data, nonces, and
future-role membership where revealing it creates leakage. Private verifier
bundles remain controller-only and never mount into an agent workspace.

Bank role assignment is performed by a custodian after the candidate clause
library and statistical protocol are committed. The search process receives
only the exploration view. Validation and final runners receive their bank
views through separate roots and process credentials after the corresponding
one-way trigger.

### 8.3 Readiness

A family becomes `eligible` only when it has:

- at least four genuinely distinct archetypes, assigned as at least two
  exploration archetypes, one disjoint validation archetype, and one disjoint
  final archetype, with six recommended for a production claim;
- at least two independently generated or frozen task clusters per
  decision-role archetype, with larger minima set by Phase 0 power analysis;
- deterministic generation or a fully frozen finite bank;
- stable family-local difficulty semantics;
- hidden and independently reproducible verification;
- reset and state-isolation guarantees;
- a model-free capability check;
- no known answer leakage; and
- successful excluded mechanics and non-degeneracy probes.

Initial family treatment is:

| Family | Framework support | Decision eligibility |
|---|---|---|
| `artifact` | Supported | Blocked pending archetype expansion. |
| `sqlite` | Supported | Blocked pending archetype expansion. |
| `shell` | Supported | Blocked pending archetype and verifier hardening. |
| `python_repair` | Supported | Blocked pending explicit archetype banking. |
| `unbrowser_fixture` | Supported | Blocked pending frozen banks and stronger oracle secrecy. |
| `unbrowser` | Supported | Permanent smoke-only live dependency. |
| `unbrowser_interactive` | Supported | Permanent smoke-only plumbing check. |
| `routing_fixture` | Supported | Consumed canary; permanent exclusion from selection. |

No production autonomous search begins while the eligible set is empty or
below the charter's minimum family count.

## 9. Evidence And Storage

### 9.1 Ledger

The search ledger is hash-chained JSONL. Every record carries:

- schema version and search ID;
- monotonic sequence number;
- previous-record hash;
- event type and event payload;
- creation time; and
- record hash.

V1 uses a single controller writer and the existing temp-file, `fsync`, and
atomic-replace pattern. The charter sets a maximum ledger-record count for
which full-file rewrite remains acceptable. A later storage backend may replace
it without changing the logical record contract.

The durable ordering for one attempt is:

1. atomically write and directory-`fsync` the active marker;
2. append and `fsync` the attempt reservation;
3. append and `fsync` the local Pi-launch receipt before spawning Pi;
4. durably spool restricted Pi output and append the Pi-exit receipt;
5. append and `fsync` the raw attempt record;
6. append and `fsync` the safe-export commitment; and
7. remove the active marker and directory-`fsync` its parent.

The marker is never removed before the ledger and raw record are durable.

Search-ledger event types include:

- search and charter validation;
- candidate compilation;
- batch proposal and freeze;
- attempt reservation, start, completion, and adjudication;
- safe-export commitment;
- posterior fit and snapshot;
- candidate qualification status;
- exploration stop;
- finalist receipt emitted and self-hash-verified; and
- search no-go or invalid.

The validation ledger separately records trigger verification, bank opening,
attempt lifecycle and safe export, decision, and final-trigger emission or
validation no-go/invalid. The final ledger separately records trigger
verification, bank opening and consumption, attempt lifecycle and safe export,
and final pass/no-go/invalid. Cross-process receipts bind predecessor ledger
hashes, but no runner appends to another runner's ledger.

Search state is reconstructed only by replaying the ledger. Mutable summary
files are caches and never authoritative.

### 9.2 Raw And Safe Views

Raw attempt artifacts remain restricted and preserve model text, event traces,
tool payloads, verifier diagnostics, and private execution details. A
whitelist-only safe export contains only fields approved for modeling and
decision reports.

Every safe row binds:

- raw-record hash;
- task-bank and task-entry hashes;
- candidate and compiled-treatment hashes;
- charter, batch, runtime, and verifier identities;
- predecision factor values;
- verified binary outcome for valid attempts;
- termination class;
- adherence and guardrail-intervention summaries; and
- consumption counters.

The safe exporter must not expose raw text, private oracle fields, selectors,
URLs, hidden tests, or final-bank outcomes before the final gate.

### 9.3 Posterior Snapshots

Each immutable `PosteriorSnapshot` binds:

- charter and evidence-ledger prefix hashes;
- included sequence range and row count;
- model schema, design-matrix and full-rank-basis schemas, priors, and fitting
  seed;
- MAP parameters, elimination-tree and block-factor hashes, Schur-complement
  and covariance-solve diagnostics;
- posterior-draw seed and count;
- task-cluster behavior and cost model state;
- training metrics and simulation calibration version; and
- source and dependency hashes.

Evaluation never overwrites a model artifact. New evidence creates a new
snapshot ID.

## 10. State Machines

Search, validation, and final evaluation are separate state machines and
process identities.

```text
search:
  draft -> charter_frozen -> banks_verified -> warmup_running
        -> adaptive_running -> exploration_stopped -> finalist_emitted

  warmup_running | adaptive_running -> no_go
  any nonterminal search state      -> invalid

validation:
  ready -> running -> passed -> final_trigger_emitted
                  -> no_go | invalid

final:
  ready -> running -> passed | no_go | invalid
```

`finalist_emitted` is terminal for the search process. Validation and final
runners cannot append to the search ledger, and the search runner has no read
path to their roots. An orchestration envelope may summarize all three state
machines but cannot alter their transitions.

Within one batch:

```text
proposed -> frozen -> reserved -> running -> raw_committed
         -> safe_exported -> posterior_eligible
```

Posterior updates occur only after the whole frozen batch is safe-exported.
Outcome-driven early stopping inside a batch is forbidden.

## 11. Bayesian Model

### 11.1 V1 Choice

V1 uses a finite-menu hierarchical Bayesian logistic model fitted in CPU
PyTorch float64 by regularized maximum a posteriori estimation, followed by a
joint block-structured Laplace approximation at the mode.

This model is selected because it:

- matches sparse binary outcome data;
- represents named factors and a small interaction set directly;
- provides posterior uncertainty for acquisition and stopping;
- can be reproduced from the standard-library package plus the existing
  pinned training environment (`numpy` and CPU `torch`); and
- is easier to calibrate in simulation than the current neural models.

Float64 MAP, block Hessian, Cholesky, and Laplace code is new; only dependency,
device, seed, and metric conventions are reused from the existing PyTorch
modules. The full parameter-by-parameter Hessian and covariance are never
materialized. Local effects are instantiated only for observed ledger cells and
organized in the fixed nesting tree task -> archetype -> family -> global. MAP
Newton solves, posterior covariance solves, and Gaussian sampling eliminate
task blocks first and pass Schur complements upward, then sample downward from
the conditional blocks. This uses dense Cholesky only within bounded local
blocks and the small global block, requiring no new sparse-library dependency.
Couplings induced by weighted contrast bases are carried as bounded low-rank
messages at the corresponding archetype or family separator; their fill is
included in the largest-block and runtime benchmark. The message rank is at
most one plus the frozen parent-effect count introduced by residualization.

The charter freezes maximum candidates, observed candidate-task cells, total
parameters, largest local block, estimated peak bytes, fit deadline, posterior
draw count, and nested-integration draw count. A pre-charter benchmark and
capacity check must pass under the bound CPU/BLAS/thread environment; a full
dense fallback is forbidden. Snapshots bind these identities and
cross-process reproducibility tolerances. Numeric fits are
tolerance-reproducible under the bound environment, not assumed byte-identical
across CPU reduction implementations.

### 11.2 Success Model

Index a valid intention-to-treat attempt by candidate `c`, family `f`,
archetype `a` nested in family, task `t` nested in archetype, and replica `r`:

```text
Y_c,f,a,t,r ~ Bernoulli(sigmoid(eta_c,f,a,t,r))

eta_c,f,a,t,r = mu
                + family_f
                + archetype_f,a
                + task_f,a,t
                + replica_block_t,r
                + difficulty_f,d
                + strategy_factors_c
                + execution_factors_c
                + declared_interactions_c,f,d
                + candidate_combo_c
                + candidate_family_c,f
                + candidate_archetype_c,f,a
                + candidate_task_c,f,a,t
```

`replica_block_t,r` is shared only when candidates are run on the same frozen
task-replica panel. Its usefulness is stress-tested because a common sampling
seed does not guarantee strongly correlated trajectories under different
prompts.

Categorical factors use committed sum-to-zero contrast matrices. Archetype and
task effects are nested as shown. Difficulty is encoded within family, not as a
globally comparable easy/medium/hard number. `candidate_combo_c` is a strongly
shrunk exact-candidate residual whose one-hot columns are projected into the
orthogonal complement of the committed factor and declared-interaction design.
It captures otherwise unmodeled high-order combinations without replacing the
interpretable factor coefficients.

`candidate_family_c,f`, `candidate_archetype_c,f,a`, and
`candidate_task_c,f,a,t` are strongly shrunk candidate-specific treatment
contrasts relative to baseline. They use weighted sum-to-zero constraints
across families within candidate, archetypes within candidate-family, and tasks
within candidate-archetype, respectively. The family term prevents an exact
high-order candidate combination from borrowing its way past a family
regression; the archetype term exposes within-family reversals; and the task
term prevents within-archetype heterogeneity from producing overconfident
common-treatment tails. The task is the effective efficacy replication unit;
replicas are repeated measurements, not independent task evidence. A
candidate-archetype gate requires direct candidate-versus-baseline coverage on
at least two distinct tasks in that archetype. Task contrasts for untested bank
tasks are integrated over the frozen prior rather than treated as observed,
and no task-specific claim is made; prior-only evidence never satisfies the
archetype coverage gate.

Every constrained parameter group is encoded in an explicit full-rank
orthonormal basis, not a redundant projected one-hot matrix. The
exact-candidate basis is the null space of the intercept plus every committed
candidate-level factor and declared-interaction column. Family, archetype, and
task bases use the same frozen weights as their g-computation targets. Baseline
treatment contrasts are fixed to zero. Each nested exact-candidate basis is
also residualized against its parent effects and every declared interaction at
the same scope, and the combined design rank is validated before fitting.

Let a frozen full-bank constrained effect be `u = B z`, where `B` is its
full-rank basis and `z` has the frozen spherical Gaussian prior. For an observed
cell subset `O`, the instantiated prior is the exact marginal with covariance
`sigma^2 * B_O * transpose(B_O)`. Re-centering only the observed subset or
constructing a new local sum-to-zero basis is forbidden. Unobserved rows retain
their conditional or marginal prior contribution for prediction and
acquisition.

All fitting and sampling operate in the full-rank `z` coordinates, or an
algebraically equivalent constrained/KKT representation. The row-space
covariance may be singular when a complete constrained group is observed and is
never inverted directly.

A frozen but unobserved exploration-bank cell and a genuinely new population
draw are different prediction cases. The former uses the exact conditional
distribution of its coordinate in the committed full-bank basis, including
cross-covariance with observed cells. The latter draws a new archetype, task,
replica, and candidate deviation from the charter's hierarchical population
prior without assigning it a coordinate in, or re-centering, the frozen bank.
Acquisition scores only committed exploration-bank cells; search-standardized
population success uses the new-draw construction.

Suggested fixed prior scales for simulator calibration are:

| Parameter group | Prior standard deviation on log odds |
|---|---:|
| Global intercept | 1.25 |
| Family effect | 0.50 |
| Archetype effect | 0.70 |
| Task effect | 0.50 |
| Family-local difficulty effect | 0.50 |
| Strategy or execution main effect | 0.50 |
| Strategy by family/difficulty | 0.35 |
| Strategy by guardrail | 0.25 |
| Other declared interaction | 0.25 |
| Exact-candidate combination residual | 0.25 |
| Candidate by family contrast | 0.25 |
| Candidate by archetype contrast | 0.25 |
| Candidate by task contrast | 0.20 |
| Task-replica block effect | 0.25 |

The first live charter freezes these scales rather than estimating many
variance hyperparameters from sparse data. Synthetic operating-characteristic
tests may revise them before the first charter, not after live outcomes.

The fitter uses a deterministic optimizer seed, explicit convergence criteria,
gradient and Hessian checks, positive-definite Cholesky diagnostics, and frozen
jitter rules. A failed fit is invalid, never a reason to fall back silently to
point estimates.

For each posterior draw, search-standardized success uses the joint posterior
for global and candidate-by-family effects and integrates a new archetype,
task, replica, and corresponding candidate deviations over their frozen
conditional Gaussian distributions. It uses the charter's family and
family-local difficulty weights without reading validation or final archetype
identities. This is the search promotion estimand. Every family and directly
observed exploration archetype also supplies a candidate-specific
noninferiority contrast, g-computed across its frozen exploration-task weights
from the same joint Gaussian draw. Descriptive observed-task averages are
reported separately.

The terminal estimand is recomputed from final-bank outcomes only over the
finite final-bank task weights. It does not reuse this adaptive posterior. The
simulator must compare Laplace tail probabilities near every decision threshold
to a higher-fidelity posterior reference and run prior-sensitivity checks;
successful Cholesky decomposition alone is not calibration evidence.

### 11.3 Validity And Behavior

Only demonstrably candidate-independent infrastructure failures may be
excluded from the success likelihood and activate a frozen contingency. A
candidate-induced compile, guardrail, tool, mechanism, budget, timeout,
refusal, malformed-answer, or verifier-false result is a valid
intention-to-treat outcome with `Y=0`. A protocol or verifier defect that
prevents unbiased adjudication invalidates the complete batch or search rather
than selectively deleting a row.

Valid non-adherent attempts remain in the success model. Diagnostic-only
families may have separate descriptive models, but their rows never update the
decision posterior.

Behavior is modeled at the task-candidate audit packet, never as independent
tool events or opportunistically accumulated replica rows. Before dispatch, a
`BehaviorAuditPacket` binds the task, candidate-balanced primary replica IDs,
applicable predecision opportunity IDs, endpoint mappings, and packet hash.
Extra replicas allocated for repeat stability cannot enter the adherence or
intervention packet. For each endpoint `b` and packet `(c,f,a,t)`, the
observation adapter emits one binary value:

```text
Z_adherence,c,f,a,t = 1 only when every observable opportunity in the frozen
                      primary audit packet was followed voluntarily
Z_intervention,c,f,a,t = 1 when any opportunity required guardrail rejection
                         or correction
Z_discordance,c,f,a,t = 1 when any predeclared repeated pair disagreed on
                        success
```

The adherence and intervention denominators contain only fixed audit packets
whose immutable task metadata declares at least one mapped opportunity before
execution. Their packet sizes and task mixture are identical across compared
candidates. The discordance denominator contains only task clusters assigned
to the frozen repeat-audit schedule. All events in one packet collapse into its
single conservative endpoint value, so additional tool events or repeat
replicas cannot change the behavior estimand or create pseudo-replication.

Each endpoint uses a separate task-level hierarchical logistic model:

```text
Z_b,c,f,a,t ~ Bernoulli(sigmoid(zeta_b,c,f,a,t))

zeta_b,c,f,a,t = mu_b
                 + candidate_b,c
                 + family_b,f
                 + candidate_family_b,c,f
                 + archetype_b,f,a
                 + task_b,f,a,t
```

The charter freezes sum-to-zero contrasts, weak fixed prior scales, minimum
distinct-task and family coverage, Laplace diagnostics, and the direction of
each endpoint. A candidate with too few distinct task clusters fails the
information prerequisite; a prior-only behavior estimate cannot pass. These
models provide family-specific marginal behavior probabilities but are not
asserted to form a joint posterior with success or with one another.
The behavior estimand and gates are family-marginal; V1 makes no separate
archetype- or task-specific behavior claim.

Post-treatment events such as "the agent reached the opportunity" are not
predictors and never define a denominator. An adapter-classification `unknown`
where a declared opportunity should be observable is a mechanism-recording
defect handled by the invalidity rules, not a missing behavior row.

### 11.4 Cost Model

Cost ranks candidates only after full qualification, but its estimand is
intention-to-treat resource use across all valid attempts:

```text
log(1 + output_tokens)
  ~ Normal(family + family-local difficulty + factor main effects
           + declared strategy-by-guardrail effects
           + exact-candidate combination residual
           + exact-candidate-by-family residual, residual variance)
```

Expected cost is standardized over the same fixed family and family-local
difficulty weights as success; it does not claim task-level cost effects absent
from this model. Its exact-candidate and candidate-family residuals use the same
full-rank weighted-basis construction recomputed for the cost design, with
separately frozen weak priors.
This prevents a policy from appearing cheap because its expensive attempts
fail. Cost per success and cost conditional on success may be reported as
secondary descriptions but never select or promote a candidate in V1.

All consumed attempts, including failures and invalid attempts, count against
the hard search budget and are reported separately.

## 12. Candidate Space And Warm-Up

The finite candidate menu is the Cartesian product of frozen factor levels,
minus compatibility and charter exclusions. An illustrative first generation
contains no more than six binary factors after simulator power analysis.

Before menu construction, a factor registry proves that every level is
observable or enforced by a bound runtime adapter. Levels that exist only as
prompt labels are rejected.

Candidate factors may include:

- direct action versus short planning;
- capability-first versus inspect-first;
- candidate validation versus immediate submission;
- one correction versus one fallback;
- sufficient-evidence stop versus additional exploration;
- warning versus rejection of unbounded requests; and
- one versus two reserved finishing actions.

The first 20 to 30 percent of the exploration budget is a balanced fractional
factorial warm-up. Its committed design matrix must have the required rank for
all main effects and declared interactions under the planned family coverage.
It must:

- cover every factor level and make every declared effect estimable;
- include the frozen baseline in every connected block;
- pair candidates on the same task and sampling seed;
- counterbalance candidate execution order;
- use new seeds for replicas; and
- cover every eligible family, archetype role, and difficulty before adaptive
  acquisition begins.

The candidate comparison graph must remain connected. Every new challenger has
a direct shared-task path to the baseline or incumbent.

Warm-up rows may satisfy exact screening cells when their candidate, task,
paired baseline, packet, and replica identities match the screening matrix.
Fractional-factor level coverage alone never implies that every exact candidate
is screening-complete.

## 13. Acquisition And Scheduling

Acquisition eligibility is deliberately weaker than qualification. A candidate
is acquisition-eligible when it is in the frozen menu, compiles for every
eligible family, has valid immutable treatment/runtime identities, and has not
triggered a search-wide protocol invalidity. No observed success probability,
behavior probability, direct-comparison minimum, or promotion gate is an
acquisition prerequisite. Thus a zero-coverage candidate can acquire the
evidence needed to become coverage-complete.

The charter freezes a promotion-coverage matrix. For every candidate it lists
the minimum distinct paired-baseline task clusters by eligible family,
exploration archetype, and family-local difficulty; the minimum opportunity
audit packets for each behavior endpoint and family; and the minimum
repeat-audit clusters. It also freezes a lower screening matrix that every
acquisition-eligible challenger must complete before global futility can be
evaluated. `coverage_complete(c)` and `screening_complete(c)` are deterministic
ledger queries. Promotion coverage is a promotion and incumbent prerequisite,
not an acquisition prerequisite. The baseline is exempt from self-pairing; its
reference coverage is the distinct task clusters on which it served as the
paired reference for challengers.

After warm-up, V1 uses three separately defined slot allocators over the finite
candidate menu.

At batch boundary `k`:

1. Compute `q_success,c` for reporting and incumbent selection. The incumbent
   is the coverage-complete candidate with largest `q_success,c`, then largest
   posterior mean search-target success, then smallest candidate hash. The
   posterior mean is an acquisition/incumbent tie-break, not a substitute for
   promotion coverage. Until one challenger is coverage-complete, the frozen
   baseline is the incumbent.
2. Fill coverage slots without posterior gating. Enumerate every unmet
   candidate-coverage requirement and compatible unused task block. Sort tuples
   by unmet screening requirement before unmet promotion-only requirement,
   candidate completed paired clusters ascending, requirement completed
   clusters ascending, factor-design leverage descending, candidate hash,
   requirement hash, then task-entry hash. Select the first tuple and pair its
   candidate directly with baseline. Recompute deficits only from the ledger
   prefix plus already reserved slots, never from outcomes pending in the
   current batch. A tuple is legal only when adding its candidate to that task
   stays within the charter's per-task candidate cap and largest-local-block
   bound.
3. Fill repeat-audit slots without posterior gating. Enumerate every
   predeclared candidate-task repeat cell with an unmet replica requirement.
   Sort by candidate completed repeat clusters ascending, family completed
   repeat clusters ascending, candidate hash, then task-entry hash. Select the
   first tuple and schedule its next frozen replica with baseline under the
   precommitted seed and order block.
4. Select one batch-level discrimination pair. Draw one joint
   success-posterior sample; `winner(draw)` is the target-standardized
   sampled-success maximizer among all acquisition-eligible candidates and need
   not satisfy promotion constraints. This is challenger one. Redraw up to 100
   times until the sampled winner differs; if no different winner appears,
   challenger two is the non-winner with highest posterior probability of
   beating challenger one, then candidate hash. If no distinct challenger
   exists, the receipt records that fact and uses challenger one alone.
5. For each posterior-discrimination slot, draw
   `B ~ Bernoulli(beta)` using the charter RNG, with simulation default
   `beta = 0.5`, and assign challenger one when `B=1` or challenger two when
   `B=0`. If challenger two does not exist, assign challenger one. Thus all
   discrimination slots use the same batch-level pair.
6. Pair each allocated discrimination challenger with the incumbent and, when
   different, the frozen baseline on the same task-replica block. A batch
   contains at most two distinct discrimination challengers; baseline and
   coverage assignments do not count against that cap.
7. For task choices not fixed by a coverage or repeat requirement, retain only
   blocks that stay within the per-task candidate and largest-local-block caps,
   then score them for candidate contrast `delta_x` by:

```text
information_score = p_bar * (1 - p_bar)
                    * transpose(delta_x) * covariance * delta_x
```

   `delta_x` is the ordered prospective design-row difference for challenger
   versus baseline. When incumbent differs from baseline, the task score is the
   charter-frozen weighted sum of challenger-versus-baseline and challenger-
   versus-incumbent contrast scores. The covariance quadratic form is evaluated
   by a deterministic block solve;
   the full covariance is not materialized. For an unused candidate-task cell,
   the scorer temporarily augments the relevant task/archetype/family blocks
   with the prospective design row and its exact conditional prior from the
   frozen full-bank bases, then computes the predictive contrast variance by
   the same Schur solve. This includes prior variance and cross-covariance for
   every uninstantiated local or global effect; zero-filling or observed-subset
   re-centering is forbidden. The augmented design, variance components, and
   score are bound in the proposal receipt. Connectivity and predecision
   opportunity bonuses frozen in the charter are then added. Ties use
   task-entry hash.
8. Fill exact quota counts computed from the charter's batch cardinality. Any
   rounding remainder is assigned in the fixed order: coverage, repeat audit,
   posterior discrimination. If one slot class has no legal tuple, its slots
   move in that same fixed order; the receipt records the empty set and reason.
9. Freeze candidate order, task order, attempt IDs, sampling seeds, primary
   schedule, and complete contingency schedule before dispatch.

Post-warm-up allocation defaults for simulation are:

- 70 percent posterior discrimination;
- 20 percent mandatory family/archetype/connectivity coverage; and
- 10 percent repeated-run stability audit.

The planner selects complete batches, not individual replacements after seeing
outcomes. Pending cells are reserved before dispatch, and concurrent planners
are forbidden in V1 because the model runtime is single-slot. The posterior,
candidate menu, coverage matrix and deficits, RNG state, quota counts, legal
tuple sets, batch-level top-two draws, and every tie-break input are bound in the
batch proposal receipt. Before charter freezing, a deterministic capacity check
proves that task and attempt budgets can complete every screening requirement.
No candidate is silently pruned for a low posterior; only the frozen global
budget or post-screening futility rule can stop its opportunity to acquire
evidence.

## 14. Promotion, Stopping, And Futility

The charter defines equal family weights by default and frozen family-local
difficulty weights. Search predictions integrate a new archetype/task draw;
the hidden final-bank weights define only the terminal estimand. Alternative
weights require explicit justification before bank creation.

For success-posterior draw `m`, candidate `c` satisfies the joint success event
`S_c,m` when:

```text
overall_success_c,m - overall_success_baseline,m >= delta_success

and for every eligible family f:
  family_success_c,f,m - family_success_baseline,f,m >= -epsilon_family,f

and for every observed exploration archetype a with minimum coverage:
  archetype_success_c,a,m - archetype_success_baseline,a,m
    >= -epsilon_archetype,a
```

All overall, family, and candidate-specific archetype contrasts in `S_c,m` are
computed from the same Gaussian success-posterior draw. Deterministic
prerequisites are checked outside `S_c,m`: complete universal compatibility,
minimum direct paired-baseline coverage for every claimed family and observed
exploration archetype, complete safe rows, valid source/runtime identities, and
zero protocol, treatment-delivery, verifier, or unclassified invalidity.

Joint success qualification probability is:

```text
q_success,c = mean_m indicator(S_c,m)
```

A candidate is success-qualified only when `q_success,c` meets the charter
threshold, with 0.95 as the simulation default. This is one joint probability
for the efficacy constraints, not separately thresholded overall, family, and
archetype probabilities.

Behavior is a separate conservative gate because the audit-packet behavior
models are not a joint generative model with success. For each eligible family
with minimum information, compute:

```text
g_adherence,c,f   = P(adherence_c,f >= adherence_min)
g_intervention,c,f = P(intervention_c,f <= intervention_max)
g_discordance,c,f = P(discordance_c,f <= discordance_max)
```

Every one of the `G` required family-endpoint probabilities must meet its frozen
marginal threshold. The simulator starts with the conservative Bonferroni-style
bound `1 - alpha_behavior / G`, may make it stricter to meet the global
false-promotion target, and may never relax it after live outcomes. This rule
does not label the separately fitted marginals a joint posterior. Missing
minimum distinct audit-packet coverage fails the behavior gate.

A candidate qualifies only when the joint success probability, every behavior
gate, and every deterministic prerequisite pass. The term `qualified` always
means this conjunction; `success-qualified` refers only to `q_success,c`.

Among fully qualified candidates whose posterior mean search-target success is
within the frozen equivalence margin of the best qualified candidate, choose
the candidate with the lowest target-standardized intention-to-treat cost, then
smallest candidate hash. This deterministic result is
`selected_finalist(snapshot)`. Never exchange a material success loss for lower
cost. If no candidate is fully qualified, `selected_finalist(snapshot)` returns
`None`, and the stability stop is false.

Autonomous exploration stops when:

- `selected_finalist(snapshot)` returns the same candidate at two consecutive
  batch boundaries, separated by at least one frozen minimum increment of new
  complete evidence, after promotion coverage;
- the hard budget is exhausted; or
- after every acquisition-eligible challenger is screening-complete,
  `mean_m indicator(any acquisition-eligible candidate satisfies S_c,m)` is
  below 0.05 at two consecutive boundaries separated by the same minimum
  evidence increment.

Budget exhaustion or futility produces `no_go`, not an invitation to expand the
budget. Consecutive-boundary finalist selection is stopping hysteresis, not an
independent replication claim. Futility is disabled before complete screening;
the charter is invalid if its frozen bank or budget cannot guarantee the
screening matrix under the primary schedule and complete contingency policy.

## 15. Validation And Automatic Final Test

### 15.1 One-Way Process Separation

The search process never reads validation or final outcomes.

1. Search emits one immutable finalist receipt and exits.
2. A separate validation runner verifies the receipt and consumes the
   validation bank once.
3. Validation emits pass, no-go, or invalid.
4. On validation pass, a separate final runner automatically verifies the
   frozen trigger and consumes the final bank once.
5. Final evaluation emits the terminal decision and has no acquisition API.

Automatic chaining is allowed, but returning to `adaptive_running` is not.
Validation and final runners use separate artifact roots and process
credentials. The search root contains only immutable trigger hashes, never
validation or final rows, labels, reports, or private bank paths.

The charter contains complete decision tables for both one-time stages:

| Stage | Data | Model | Pass action | Failure action |
|---|---|---|---|---|
| Validation | Complete paired finalist/baseline validation bank only | Frozen validation likelihood, weak priors, margins, and posterior seed | Emit one self-hashed final-trigger receipt | Terminal `no_go`; never return to search |
| Final | Complete paired finalist/baseline final bank only | Frozen final likelihood, weak priors, margins, and posterior seed | Terminal `passed` | Terminal `no_go` |

Identity, sealing, completeness, protocol, treatment-delivery, verifier, or
analysis defects produce `invalid`, not a substantive no-go. Every valid
attempt must carry a measured intention-to-treat cost; missing cost is an
integrity failure rather than an invitation to fit from an incomplete subset.

### 15.2 Final Design

The final bank runs a complete paired finalist-versus-baseline panel with
frozen replicas, sampling seeds, chronology, and contingency blocks. The final
analysis does not reuse the adaptive or validation posterior. It applies the
separately frozen final model and weak priors to final outcomes only.

Validation and final use the same indexed likelihood shape restricted to two
treatments: baseline and finalist. A finalist indicator and its declared
family interaction plus finalist-by-archetype and finalist-by-task treatment
contrasts are the only treatment contrasts; family, archetype, task, and
task-replica block effects account for the complete paired bank. Each
archetype contrast therefore uses direct finalist-versus-baseline evidence
rather than an archetype intercept, and finite-bank g-computation averages the
task-specific treatment contrasts rather than imposing a common within-
archetype effect. The task is the effective replication unit. Priors, contrast
matrices, posterior draws, and g-computation over the observed finite bank are
frozen separately for each stage.

The finalist family, archetype, and task contrasts use the same full-rank
nested weighted bases as search. When a parent has only one child in a bank,
the child's deviation basis has zero columns and the parent contrast carries
that cell mean; redundant family/archetype/task columns are never retained.

The terminal claim is limited to the finite, weighted final-bank mixture.
Family and archetype results refer only to family and archetype cells present
in that bank. They are not population claims about unseen archetypes.

Final pass requires all of:

- one joint success-posterior probability, computed exactly as specified by the
  final decision table, at or above the final threshold;
- overall success lift plus every final-bank family and archetype
  noninferiority margin inside that joint event;
- zero protocol, treatment-delivery, verifier, and unclassified invalidity;
- adherence, guardrail-intervention, and repeat-stability gates; and
- after success passes, the frozen intention-to-treat cost rule.

Final behavior gates are recomputed from final-bank task clusters only using
the separately frozen task-level models and conservative marginal thresholds.
They are not folded into, or described as part of, the joint success posterior.

Any failed substantive gate is `no_go`. Any broken identity, completeness,
sealing, or analysis invariant is `invalid`. The final bank remains consumed in
both cases and its outcomes never authorize another search generation.

On first execution, `final run` accepts only the unique validation-pass receipt,
expected final-bank unconsumed receipt, exact charter hash, and expected final
runner source hash. On replay after consumption, it verifies the existing final
ledger and re-emits the stored terminal decision without reopening the bank or
rechecking the former unconsumed precondition. This ledger-defined replay is
idempotent. There is no force or manual-override flag.

## 16. Failure And Resume Semantics

Valid model, guardrail rejection, wrong action sequence, tool misuse, budget,
refusal, timeout-with-valid-record, malformed-answer, and verifier-false
outcomes are candidate-induced intention-to-treat failures with `Y=0` and are
never rerun. A candidate's auditable failure to follow the intended mechanism
is an outcome, not missing data. Delivery of the wrong treatment/interface or
a corrupt mechanism receipt is a protocol defect and invalidates the batch.

Only demonstrably candidate-independent infrastructure failure may activate a
replacement, and only through a primary/contingency unit frozen in the batch
manifest. Examples are a controller failure before Pi starts, a provider outage
before any provider turn, or a common fixture-server failure before candidate
execution. V1 uses complete contingency blocks rather than selective cell
reruns. Triggered primary outcomes are quarantined from the posterior.

A protocol, mechanism-recording, verifier, identity, or unclassified defect
that prevents unbiased outcome adjudication invalidates the complete batch or
search. It never deletes only the affected candidate's failure.

Before every attempt, the controller writes an active marker containing the
charter, bank, batch, candidate, task, attempt, runtime, local spool, and Pi
lifecycle hashes. Before spawning local Pi it durably appends a launch receipt
with process identity and start time. Pi stdout and stderr stream to restricted
durable local spool files. On exit, the controller appends the exit status and
spool hashes before remote event recording or verification. It appends a raw
record and `fsync`s it before removing the marker.

On restart:

- completed records are never rerun;
- malformed or hash-invalid records fail closed;
- an active marker creates uncertain state and requires evidence-based
  adjudication;
- only an active attempt with no durable Pi-launch receipt may be classified as
  a pre-execution controller error eligible for frozen contingency;
- any durable Pi-launch receipt without a complete recoverable result
  invalidates the affected batch, even when the remote attempt remains
  `prepared` and no event file exists;
- any possibility of a valid but unrecorded attempt invalidates the affected
  batch; and
- posterior snapshots are rebuilt from the ledger prefix rather than trusted
  as mutable state.

## 17. Runtime Layout

Recommended controller layout:

```text
policy-lab/
  contracts/
    families/
    capabilities/
    clauses/
  banks/
    <bank-id>/
      manifest.json
      public/
      private-commitments.json
  searches/
    <search-id>/
      charter.json
      candidate-registry.json
      compile-receipts/
      batches/
      raw/
      safe/
      ledger.jsonl
      active.json
      posterior/
      finalist-receipt.json
  validations/
    <validation-id>/
      trigger.json
      ledger.jsonl
      decision.json
      final-trigger.json
  finals/
    <final-id>/
      trigger.json
      consumption.json
      ledger.jsonl
      decision.json
```

Committed schema and clause definitions live in the repository. Frozen task
banks and run artifacts live under the ignored run root or a configured durable
artifact store. Search, validation, and final roots have separate access
profiles. Private verifier data lives only on the trusted execution host.

## 18. Controller CLI

Add one controller entry point, `pyreplab-policy-lab`, with subcommands:

- `contracts validate`;
- `bank freeze`, `bank verify`, and `bank status`;
- `candidate compile` and `candidate validate`;
- `charter freeze` and `charter validate`;
- `search simulate`, `search run`, `search resume`, and `search status`;
- `validation run`;
- `final run`; and
- `decision validate`.

Commands print only safe summaries. Raw outcomes, private bank data, and final
labels are not emitted by status commands.

`search run` refuses an existing ledger. `search resume` validates the charter,
replays the ledger, reconciles active state, and then continues. Runtime host,
project, provider, model, and sampling values come only from the charter and
cannot be overridden by CLI flags. Search RNG uses the explicit
`--search-seed` only during charter freezing; task and sampling seeds come from
bank and batch manifests. `final run` requires the unique final-trigger receipt
and exposes no override or force option.

## 19. Security And Leakage Controls

- Agent workspaces never contain verifier or oracle files.
- Capability contracts derive visible text from audited machine fields.
- The clause library, factor menu, compiler, priors, and statistical protocol
  are committed before validation/final task contents or coordinates are
  accessible to candidate designers.
- Search features are predecision family, archetype, difficulty, candidate
  factor, and runtime identities only.
- Raw model text, tool payloads, post-action traces, and verifier diagnostics do
  not become search predictors.
- Final-role membership and outcomes are unavailable to the acquisition
  process.
- Validation and final artifacts live outside the search process's readable
  root and are opened only by receipt-bound runners.
- Bank nonces use unreleased random salt commitments, not only public seeds.
- Runtime source, provider, model, tool binary, sandbox, and prompt identities
  are checked before every start or resume.
- Every safe export has a whitelist schema and binds exact raw hashes.
- Excluded, smoke-only, consumed, and retrospective rows fail closed if offered
  to the decision fitter.

## 20. Implementation Phases

### Phase 0: Statistical Protocol And Simulator

- Freeze the candidate design matrix, priors, target weighting, acquisition,
  promotion event, budgets, and final gate in a simulator-only protocol.
- Simulate null, single-winner, interaction, family-regression,
  archetype-reversal, within-archetype task-reversal, clustered-behavior,
  unequal raw repeat exposure, zero-coverage acquisition, sparse-success,
  invalidity, and cost-only scenarios.
- Establish false-promotion, power, budget, convergence, and calibration
  operating characteristics.

### Phase 1: Contracts And Compatibility

- Implement family, capability, strategy, execution, candidate, and charter
  contracts.
- Consolidate task-policy-runtime compatibility checks.
- Add deterministic clause and capability-text rendering.
- Add explicit archetype identity to every bank-eligible family.
- Inventory every proposed execution factor against an enforcement or
  observation adapter; defer unsupported factors.

### Phase 2: Task Banks

- Add bank freezing, verification, role separation, oracle commitments, and
  consumption.
- Expand core-family archetypes and harden `unbrowser_fixture` secrecy.
- Keep every existing family non-eligible until its readiness gate passes.

### Phase 3: Compiler And Ledger

- Implement a pure shadow compiler and compile-receipt contract without
  changing `orchestrator._run_pi`.
- Compare shadow output byte-for-byte with the existing command path across
  every supported family and treatment interface.
- Rewire the orchestrator only after a separately frozen parity gate passes;
  retain the prior path for existing experiment manifests.
- Add the hash-chained ledger, active markers, replay, safe export, and crash
  reconciliation.
- Add durable local Pi lifecycle and restricted output spooling before any live
  Policy Lab attempt.

### Phase 4: Bayesian Model

- Implement design-matrix generation, MAP fitting, Hessian/Laplace posterior,
  fixed-packet adherence/intervention and task-cluster discordance models, cost
  model, and
  immutable snapshots.
- Reproduce simulator operating characteristics in unit and integration tests.

### Phase 5: Autonomous Dry Search

- Run the entire state machine against synthetic outcomes and mocked runners.
- Exercise budget, futility, qualification, validation, final, invalidity,
  interruption, and resume paths.

### Phase 6: Excluded Live Pilot

- Run a small permanently excluded search over benchmark-ready families.
- Validate mechanics, posterior calibration, comparison connectivity, and
  spending limits.
- Do not use this pilot to make the universal-policy claim.

### Phase 7: First Authorized Search

- Freeze a fresh charter and untouched banks only after all previous gates
  pass.
- Execute autonomous exploration, one-time validation, and automatic final
  under the fixed claim boundary.

## 21. Verification Plan

### Contract And Integrity Tests

- Round-trip every schema and reject unknown, missing, extra, or wrong-type
  fields.
- Reject boolean-as-integer values, non-finite numbers, duplicate IDs, and hash
  drift.
- Verify deterministic candidate and compile hashes across processes.
- Verify the complete `CompiledTreatment` and compile-receipt schemas and
  byte-identical shadow compilation for every supported family.
- Verify every observation adapter against model-free positive, negative,
  no-opportunity, and unknown fixtures without outcome/private-field access.
- Reject duplicate factor vectors and any treatment-varying registry or runtime
  field without a committed factor column.
- Reject any candidate that fails compilation in one eligible family.
- Tamper every bound artifact and assert fail-closed behavior.

### Bank And Leakage Tests

- Prove whole-archetype role separation and complete task grouping.
- Prove private fields never enter public manifests, prompts, model features, or
  safe exports.
- Reject ineligible-family evidence at the fitter boundary.
- Prove final-bank consumption is one-time and irreversible.
- Prove the clause library and statistical protocol predate any access to
  validation/final task contents.
- Prove the BankRole-to-dataset-split mapping cannot map final or excluded rows
  into exploration.
- Prove existing train/evaluate loaders reject `final_sealed` and legacy `test`
  rows cannot become validation or final evidence.

### Model Tests

- Recover known factor and interaction effects in synthetic panels.
- Verify every orthonormal contrast basis has the frozen rank, weighted
  centering, baseline-zero convention, and no redundant columns.
- Verify every partially observed local-effect prior equals the restricted
  full-bank basis covariance and reject observed-subset re-centering.
- Verify complete constrained groups are solved in full-rank coordinates without
  attempting to invert their singular row-space covariance.
- Distinguish frozen-unobserved-cell conditional predictions from genuinely new
  population draws and match both to small reference calculations.
- Verify the combined global/nested design rank after residualizing against all
  declared interaction spaces.
- Match block-Schur MAP solves, covariance quadratic forms, and posterior draws
  to a dense reference on small fixtures; reject any production dense fallback.
- Benchmark the charter's maximum candidate/task dimensions against frozen
  separator fill, peak-memory, and fit/draw deadlines.
- Check posterior coverage and calibration under sparse and imbalanced data.
- Check separation, singular Hessian, and low-success behavior.
- Compare decision-tail probabilities to a higher-fidelity posterior reference
  and run frozen prior-sensitivity cases.
- Prove family regression cannot be hidden by pooled improvement.
- Prove an exact-candidate family regression cannot be hidden by factor or
  global candidate effects and cannot pass without direct family coverage.
- Prove a candidate-specific archetype reversal cannot be hidden by family or
  archetype intercepts and cannot pass without direct paired coverage.
- Prove within-archetype candidate-by-task reversals inflate uncertainty and
  are averaged correctly in final finite-bank g-computation.
- Prove repeated opportunities and replicas outside a frozen audit packet
  cannot alter adherence/intervention or create pseudo-replication, and
  calibrate every conservative marginal behavior gate under cross-endpoint
  dependence.
- Prove cheap failures cannot win the cost tie-break.
- Prove candidate-induced invalid behavior is scored as failure rather than
  missing outcome.

### Search Tests

- Reproduce every acquisition from the same ledger and seeds.
- Reproduce the exact coverage, repeat-audit, and top-two allocators, legal tuple
  sets, redraw cap, tie-breaks, information scores, quotas, and RNG state.
- Reproduce the frozen baseline/incumbent contrast weights in every information
  score.
- Match prospective unused-cell variance scores to an explicitly augmented
  dense reference on small fixtures, including prior cross-covariances.
- Prove a zero-coverage candidate remains acquisition-eligible and receives
  deterministic coverage slots without first passing a posterior gate.
- Prove futility is impossible before every challenger completes screening and
  that the charter rejects insufficient screening budget or bank capacity.
- Prove all discrimination slots use one batch-level top-two pair and never
  exceed the two-challenger cap.
- Prove coverage and discrimination reject tuples that exceed the per-task
  candidate or largest-local-block cap.
- Maintain a connected comparison graph and frozen coverage quotas.
- Never update from a partial batch.
- Stop exactly at budget, stable deterministic-finalist, or futility boundaries.
- Reject self-budget expansion and charter mutation.

### Crash And Resume Tests

- Inject crashes before and after every state transition.
- Preserve uncertain active attempts without silent reruns.
- Inject crashes before Pi launch, during Pi, after Pi exit, during local spool
  persistence, before remote event recording, and before marker removal.
- Prove a prepared-only remote directory cannot be treated as proof that Pi did
  not run.
- Rebuild state and posterior snapshots from ledger replay.
- Validate full-block contingency and quarantine semantics.

### Final-Test Tests

- Prove the search process cannot read validation or final outcomes.
- Prove validation and final roots are absent from the search process's access
  profile.
- Prove validation failure cannot return to search.
- Prove automatic final opens only from one exact immutable trigger.
- Prove `final run` is idempotent and has no force path.
- Prove search, validation, and final event types cannot enter another process's
  ledger.
- Validate report decision, self-hash, bank consumption, and artifact bindings.

## 22. Acceptance Gates

No autonomous live search is authorized until:

- all contracts and compiler outputs are self-validating and reproducible;
- every candidate compiles for every eligible family and passes non-decision
  mechanics in every non-deprecated supported family;
- at least the charter's minimum eligible family set has readiness-passing task
  banks;
- the simulator meets frozen false-promotion, power, calibration, and budget
  targets;
- the nested block-Laplace benchmark meets frozen parameter, memory, fit-time,
  posterior-draw, and integration-draw capacity bounds without dense fallback;
- the primary and complete-contingency budgets can complete every candidate's
  screening matrix;
- every dry-run state, failure, contingency, and resume path passes;
- safe-export and leakage audits report zero violations;
- posterior decisions reproduce from a fresh replay;
- automatic validation and final tests are one-way and one-time; and
- an independent review finds no unresolved critical or high-severity protocol
  defect.

The first live search charter must separately freeze exact numeric thresholds,
budgets, factor levels, interactions, target weights, bank sizes, and the
baseline candidate. They are intentionally not universal constants in this
architecture document.

## 23. Primary Risks

1. **Joint-search sparsity.** Strategy and execution interactions can exceed the
   available evidence. Limit V1 to a small finite menu and fixed shrinkage.
2. **False universality.** Generic fallback can hide capability weakness. Enforce
   family and archetype gates.
3. **Contract-as-prompt leakage.** Rich capability text can become a covert
   domain policy. Generate it from bounded machine fields and audit it.
4. **Adaptive aliasing.** Bayesian acquisition can confound factors with task
   difficulty. Preserve warm-up coverage and a connected comparison graph.
5. **Laplace overconfidence.** Sparse logistic posteriors can be too narrow.
   Calibrate on simulation and fail closed on poor Hessian diagnostics.
6. **Resource-model misspecification.** Heavy-tailed all-attempt costs can make
   the Gaussian log-cost model unstable. Calibrate robust diagnostics and keep
   cost secondary to success.
7. **Benchmark weakness.** Too few role-specific archetypes or tasks cannot
   support a universal claim. Keep families ineligible until expanded.
8. **Automatic-final irreversibility.** A sealing or gate bug consumes the bank.
   Require complete dry-run and tamper coverage first.
9. **Runtime drift.** Model, provider, tool, prompt, or source changes can
   dominate policy effects. Verify exact pins at every start and resume.
10. **Mechanism/outcome confusion.** Enforcement can make weak instructions look
    adherent. Report voluntary behavior and guardrail intervention separately.
11. **Candidate-dependent missingness.** Misclassified mechanism or tool
    failures can make a weak candidate look strong. Only independent
    infrastructure may be excluded.
12. **Sealed-content access.** Human or process access to validation/final task
    content before policy freezing can overfit the benchmark without outcome
    leakage. Require custody and access receipts.
13. **Treatment heterogeneity.** Shared task intercepts can hide policy-by-task
    reversals. Retain shrunk treatment-by-archetype and treatment-by-task
    contrasts and calibrate their tails.
14. **Behavior exposure drift.** Adaptive repeats can change all/any endpoints.
    Bind fixed candidate-balanced audit packets before dispatch.

## 24. Non-Goals For V1

- Free-form LLM rewriting of complete policies.
- Adding policy factors during a running search.
- Mid-attempt posterior updates or policy switching.
- Concurrent model workloads or multiple search planners.
- Neural task embeddings as the primary acquisition model.
- Transfer learning from smoke-only, retrospective, or excluded data.
- Estimating many hierarchical variance parameters from sparse panels.
- Generalizing the final claim beyond the frozen eligible-family universe.
- Treating support for smoke-only families as evidence of efficacy.

## 25. Remaining Charter-Level Decisions

The architecture is complete, but the first simulator protocol must still
choose:

- the exact initial factor levels and allowed interactions;
- the universal baseline strategy and execution profile;
- eligible family readiness thresholds and first bank composition;
- total attempt, token, time, batch, and candidate budgets;
- overall improvement, finalist-equivalence, and family/archetype
  noninferiority margins;
- fixed prior scales for exact-candidate, treatment-by-family,
  treatment-by-archetype, and treatment-by-task terms;
- adherence, intervention, and repeat-stability thresholds;
- behavior error budget, marginal gate probabilities, and minimum distinct-task
  coverage by endpoint and family;
- warm-up fraction, screening and promotion coverage matrices, acquisition
  quota values, and futility screening increment;
- block-Laplace per-task candidate, separator-fill, dimension, memory, and time
  bounds, posterior and nested-integration draw counts, and numerical
  tolerances;
- validation and final bank sizes; and
- automatic validation and final promotion probabilities.

These decisions must be justified through simulator operating characteristics
and frozen before any live search outcome is observed.

Phase 0 must emit a separate frozen statistical protocol containing the exact
indexed likelihood, full-rank contrast bases, target standardization,
observed-cell marginal-prior, genuinely-new-draw, and prospective-variance
formulas, block-elimination and conditional-sampling algorithm, behavior-gate
cluster definitions and marginal thresholds, joint success event, acquisition
pseudocode parameters and legal tuple orderings, validation decision table, and
final decision table. This architecture document does not substitute for that
experiment-specific protocol.
