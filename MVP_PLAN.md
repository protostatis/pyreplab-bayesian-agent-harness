# Pyreplab Bayesian Agent Harness — MVP Plan

> **Status:** Proposed implementation plan for the first controlled toy experiment.
>
> **Primary objective:** Build a local, procedurally generated, independently verifiable task gym; use the frozen Gemma 26B agent through Pi to generate policy-outcome data; then train and evaluate a policy-conditioned neural success model.

## 1. MVP Hypothesis

For a mixed family of deterministic terminal tasks, a learned allocator can use information available before execution to choose between two fixed Gemma execution policies and achieve higher verified success than a cost-matched fixed or random policy.

The first MVP tests a one-decision contextual allocator. It does not yet test full sequential control.

## 2. Fixed Decisions

```text
Rollout agent       = Gemma 26B, frozen during the experiment
Agent runtime       = Pi 0.84.1 with an explicit Gemma provider/model
Execution host      = ubuntu-local (example SSH alias) for model inference, task sandboxes,
                      verification, corpus storage, and NN training
Gym strategy        = procedurally generated mixed micro-terminal gym
Policy decision     = one policy selected before an attempt begins
Initial policies    = Direct and Deliberate
Outcome             = independent terminal verification
Harness model       = policy-conditioned multimodal neural outcome model
```

The exact Pi provider, model identifier, Pi version, model settings, and policy versions must be resolved and pinned in each experiment manifest. Production data must not depend only on a mutable default-model setting.

### Verified environment snapshot

The Phase 0 infrastructure audit on 2026-08-09 established:

```text
Pi client                 = 0.84.1 on the development Mac
Pi provider               = ubuntu-gemma
Pi model                  = gemma-4-26b-a4b
Pi API mode               = openai-completions
Pi model transport        = local port 18081 through an SSH tunnel

ubuntu-local OS           = Ubuntu 22.04.5 LTS
CPU                       = 16 logical CPUs
RAM                       = 62 GiB
Swap                      = 127 GiB
Root storage              = 1.8 TiB total, approximately 1.2 TiB free
GPU                       = NVIDIA GeForce RTX 3080 Ti, 12 GiB VRAM

Gemma server              = llama-server on 127.0.0.1:8081
Gemma parameters          = approximately 25.23B
Gemma quantization        = IQ4_NL
Gemma context             = 65,536 tokens
Gemma server parallelism  = 1
Gemma model size          = approximately 13.6 GB

Remote Python             = 3.12.13; PyTorch is not yet installed
Remote Pi                 = not currently installed
Remote Node.js            = 13.12.0 and too old for the planned Pi runtime
Docker                    = installed, but the SSH user cannot access its daemon
Bubblewrap                = installed and verified with unprivileged namespaces
systemd user scopes       = available for resource limits
```

Pi JSON mode was verified to identify the provider and model, report token usage, emit structured tool-call and tool-result events, and complete a multi-turn tool loop. The observed provider reports no separate reasoning-token channel, so the Deliberate policy must be defined through its prompt, workflow, and budget rather than relying on Pi's thinking-level setting.

## 3. Scope

### Included

- Four general-purpose local task families: structured artifacts, SQLite,
  Python repair, and shell/filesystem operations.
- One optional fixed-page, read-only Unbrowser control smoke, isolated from the
  general-purpose task-network boundary.
- Seeded task generation with multiple templates and difficulty levels.
- Disposable task environments and hidden programmatic verifiers.
- Paired Direct and Deliberate Gemma rollouts.
- Complete Pi trajectory, cost, latency, and outcome logging.
- A neural representation of text and structured pre-decision features.
- Bayesian or ensemble uncertainty over the outcome model.
- Held-out allocator evaluation at matched average cost.

### Excluded from the first MVP

- Fine-tuning Gemma.
- General live-web/browser tasks and browser benchmarks. The sole exception is
  the fixed `https://example.com/` read-only plumbing smoke; it is not a
  benchmark or research corpus.
- External benchmarks such as Terminal-Bench or SWE-bench.
- Human escalation and user-response simulation.
- Multiple harness decisions within one trajectory.
- Durable distributed scheduling or a production database.
- Claims of cross-domain generalization beyond the generated gym.

## 4. System Overview

```text
seed + template
      |
      v
task generator ---------> private oracle/verifier bundle
      |
      v
public task contract + isolated initial workspace
      |
      +-------------> Direct Pi/Gemma attempt ------+
      |                                             |
      +-------------> Deliberate Pi/Gemma attempt --+--> independent verifier
                                                    |
                                                    v
                                      normalized AttemptRecord dataset
                                                    |
                                                    v
                                   policy-conditioned neural outcome model
                                                    |
                                                    v
                                      cost-constrained policy allocator
```

Each policy receives a fresh copy of the same initial workspace. Attempt order is randomized, and no state is shared between attempts.

## 5. Core Data Contracts

### `TaskSpec`

```text
id                    stable generated-task identifier
family                artifact | sqlite | python_repair | shell
template_id           versioned generator template
generator_version     task-generator code version
seed                  deterministic generation seed
prompt                user-facing task request
contract              explicit success criteria and required artifacts
public_metadata       information legitimately visible before policy choice
private_metadata      oracle and analysis fields excluded from the model
workspace_ref         immutable initial workspace snapshot
verifier_ref          private verifier bundle
split                 train | validation | test
```

### `PolicySpec`

```text
id                    direct | deliberate
version               immutable policy version
system_prompt         policy-specific execution instructions
allowed_tools         Pi tool allowlist
token_limit           maximum generated tokens
tool_call_limit       maximum tool calls
wall_time_limit       attempt timeout
thinking_setting      pinned Pi/model setting, when supported
```

### `AttemptRecord`

```text
run_id
task_id
policy_id and version
Pi version
provider and exact model ID
model settings and seed, when supported
pre-decision state snapshot
raw Pi JSON event stream reference
normalized messages and tool events
token and tool usage
wall-clock latency
termination reason
verifier identity and version
verified success label
verifier diagnostics
safety or sandbox events
```

Only the pre-decision public fields may be used to train the allocator. Verifier diagnostics, private generator metadata, and trajectory information produced after policy assignment must not leak into the initial policy decision.

## 6. Mixed Task Gym

All task families implement the same generator and verifier protocols:

```text
generate(seed, template, difficulty) -> TaskSpec + workspace + private oracle
verify(task_id, submitted workspace) -> VerificationResult
```

### 6.1 Structured artifact tasks

Example operations:

- Join CSV and JSON records under natural-language constraints.
- Aggregate logs into a required JSON schema.
- Reconcile conflicting records using declared precedence rules.
- Transform multiple source files while preserving types and ordering rules.

Verification parses the submitted artifact and independently recomputes the expected semantic result. It must not rely on raw string equality when ordering or formatting is irrelevant.

### 6.2 SQLite tasks

Example operations:

- Update records while preserving database invariants.
- Deduplicate or reconcile related tables.
- Produce a report from multi-table conditions.
- Apply a migration described by a task contract.

Verification runs private SQL queries against the submitted database and checks schema, row-level conditions, aggregates, and invariants.

### 6.3 Python repair tasks

Example mutations:

- Boundary-condition errors.
- Incorrect operators or branch conditions.
- Wrong dictionary keys or field mappings.
- Mishandled empty, missing, or duplicate values.
- Small cross-function consistency errors.

The workspace may include basic public tests, but final verification uses a hidden `pytest` suite in a separate verifier sandbox. The verifier checks behavior rather than requiring the reference patch.

### 6.4 Shell and filesystem tasks

Example operations:

- Reorganize a generated file tree according to explicit rules.
- Build or extract archives with required contents.
- Rename, deduplicate, or classify files using content and metadata.
- Repair permissions or generate checksums and manifests.

Verification checks the resulting tree, file content, hashes, permissions, and declared invariants rather than the commands used.

### 6.5 Difficulty and diversity

Each family should have several independently versioned templates with `easy`, `medium`, and `hard` parameter ranges. Difficulty can vary through:

```text
number of files or records
number of constraints
number of joins or dependent steps
presence of missing, duplicate, or conflicting values
required output-schema complexity
amount of irrelevant evidence
number of plausible but incorrect shortcuts
```

Private generator difficulty may be used for corpus balancing and analysis but should not be a model feature unless the harness can derive the same information before execution.

## 7. Verification and Sandbox Boundary

The verifier is part of the experimental measurement system, not part of either execution policy.

Required properties:

- The same verifier evaluates Direct and Deliberate attempts.
- Oracle data and hidden tests are never mounted in the agent environment.
- Verification is deterministic for a fixed submitted workspace.
- The verifier runs after Pi exits or reaches its budget.
- Timeout, crash, malformed submission, and verifier failure are distinct outcomes.
- Executing submitted Python code occurs in a separate restricted verifier sandbox.

For the first MVP, the preferred attempt environment is a Bubblewrap sandbox executed on `ubuntu-local`. Bubblewrap is already available to the SSH user, works without daemon or root access, and can isolate tool execution from both the host and the model network.

```text
Pi process and model connection remain outside the task sandbox
Pi built-in host tools are disabled
only experiment-specific remote workspace tools are exposed
selected runtime directories are mounted read-only
the generated task workspace is the only writable bind
/home and project-private paths are not mounted
network namespace is unshared
process, IPC, user, and mount namespaces are unshared
CPU, memory, process, and wall-time limits use systemd user scopes
```

The verifier runs in a second Bubblewrap sandbox with the submitted workspace and private verifier bundle. Agent and verifier sandboxes must never share the private oracle mount.

Docker remains a possible later backend, but it is not an MVP dependency. Docker 19.03.12 is installed on `ubuntu-local`, but the current SSH user is not a member of the Docker group and cannot access the daemon. Granting Docker access is effectively granting root-level host control and should be a deliberate infrastructure decision rather than a prerequisite for the toy experiment.

## 8. Pi and Gemma Integration

Pi owns the model/tool loop. In the fastest safe MVP topology, Pi remains on the development Mac while all heavy computation and task execution remain on `ubuntu-local`:

```text
Pi 0.84.1 on development Mac
  -> ubuntu-gemma provider
  -> SSH tunnel: 127.0.0.1:18081 to ubuntu-local:8081
  -> Gemma llama-server on ubuntu-local

Pi experiment tool extension
  -> persistent SSH JSON-RPC connection
  -> gym worker on ubuntu-local
  -> Bubblewrap-isolated task workspace
```

This avoids upgrading remote Node.js or installing Pi before the vertical slice. Moving Pi onto `ubuntu-local` can be evaluated later; it would require a supported Node.js runtime and an experiment-specific Pi installation.

The experiment must disable Pi's built-in host filesystem and shell tools. A dedicated Pi extension exposes only gym tools whose implementations forward structured requests to a persistent worker over SSH. The worker validates task IDs and paths, then executes shell and filesystem operations inside the task's Bubblewrap sandbox. The model must never receive a generic SSH or host-shell tool.

Required Pi behavior:

```text
non-interactive print mode
JSON output mode
no persisted session
no inherited AGENTS.md or other context files
no unrelated skills, prompt templates, or auto-discovered extensions
only the explicit model-switch and sandbox-tool extensions
explicit tool allowlist
policy-specific system prompt
explicit provider ubuntu-gemma
explicit model gemma-4-26b-a4b
```

The raw Pi JSON stream is retained unchanged for auditability. A separate normalizer extracts messages, tool calls, token usage, timing, errors, and final status into `AttemptRecord`.

The audit has already established that Pi can invoke the configured Gemma model, emit JSON events, report usage, produce a structured tool call, consume its result, and finish the turn. The remaining integration spike must establish:

1. The Pi experiment extension can maintain a remote worker connection.
2. Built-in Pi tools are unavailable during an experiment attempt.
3. Remote tool actions affect only the assigned Bubblewrap workspace.
4. The task sandbox has no network or access to host home directories.
5. A separate verifier sandbox can evaluate the submitted workspace.
6. Repeated `--no-session` attempts share no conversation or workspace state.
7. The experiment manifest records exact Pi, provider, model, server, policy, and sandbox versions.

## 9. Initial Execution Policies

### Direct policy

```text
- Solve immediately using the available tools.
- No required planning stage.
- Small token, tool-call, and time budget.
- Stop after producing the requested artifact or change.
```

### Deliberate policy

```text
- Inspect the task and relevant inputs before editing.
- Form an explicit internal plan.
- Use Python or shell computation when useful.
- Inspect or test the completed work before stopping.
- Receive a larger token, tool-call, and time budget.
```

Both policies receive the same contract, initial workspace, base model, and tool types. The intervention intentionally includes execution procedure and resource budget because the harness is learning resource allocation.

Policy prompts and limits must be versioned. Changing a policy creates a new treatment rather than silently modifying an existing policy ID.

## 10. Rollout and Dataset Design

For each task:

1. Generate and freeze the task, initial workspace, and private oracle.
2. Randomize whether Direct or Deliberate runs first.
3. Run each policy in a fresh environment.
4. Keep Gemma, Pi, tools, and verifier versions fixed.
5. Verify each final workspace independently.
6. Store one attempt row per `(task, policy)` pair.

Paired attempts remove policy-assignment confounding for this toy environment and reveal tasks where policy choice changes the result. Start with deterministic or fixed model settings for reproducibility. Later, repeat selected tasks with multiple model seeds or sampling settings to estimate outcome variability.

### Pilot corpus

```text
4 task families
3 difficulty bands
10 seeds per family/difficulty cell
2 policies per task
= 240 attempts
```

The pilot is for calibrating the gym, not proving the allocator.

### Initial training corpus

Target approximately:

```text
300–1,000 tasks
600–2,000 paired attempts
```

Final volume depends on measured Gemma throughput and the diversity of outcomes. Scale only after the pilot demonstrates meaningful Direct/Deliberate disagreement.

### Dataset splits

- Never split paired attempts from the same task across datasets.
- Split by generator template and seed, not by random attempt rows.
- Reserve unseen seeds for in-template generalization.
- Reserve at least one subtemplate per family for structural generalization.
- Keep a final test set frozen before model selection.

## 11. Harness Neural Outcome Model

The binary outcome distribution may remain simple while its conditional success function is neural and multimodal:

```text
Y | x, policy ~ Bernoulli(q(x, policy))
```

Initial architecture:

```text
task prompt + contract + public workspace summary
  -> pretrained text encoder

file counts, sizes, types, budgets, and environment fields
  -> normalized numeric encoder with missingness indicators

task family and candidate policy
  -> learned embeddings

all representations
  -> policy-conditioned fusion network
  -> Bayesian outcome head, Bayesian adapters, or small deep ensemble
  -> posterior samples or calibrated intervals for success
```

The workspace summary must be generated before policy assignment and must not inspect private oracle data. The first model can freeze most of the text encoder while training the numeric tower, policy embedding, fusion layers, and uncertainty-aware head. Broader neural fine-tuning can follow once the dataset is large enough.

Record cost and latency as separate outcomes. The first decision rule may select Deliberate only when its expected success uplift justifies its additional expected cost.

## 12. Baselines and Allocator Evaluation

Compare at least:

1. Always Direct.
2. Always Deliberate.
3. A random Direct/Deliberate mixture matched to the allocator's average cost.
4. A transparent family/numeric logistic or GAM baseline.
5. The policy-conditioned neural model.
6. The uncertainty-aware neural allocator.

Model metrics:

```text
log loss
Brier score
calibration and reliability
selective accuracy under deferral
success-uplift estimation
OOD/support behavior
```

Decision metrics:

```text
verified success rate
average token, tool, time, and compute cost
success at matched average cost
regret relative to the best observed policy per task
results by task family and difficulty band
task-level bootstrap uncertainty
```

## 13. Implementation Phases

### Phase 0 — Environment and Pi spike

- Treat the verified environment snapshot above as the infrastructure baseline.
- Implement a persistent SSH JSON-RPC gym worker on `ubuntu-local`.
- Implement an explicit Pi extension exposing only sandboxed gym tools.
- Execute tool commands through Bubblewrap with network isolation.
- Apply memory, CPU, process, and wall-time limits with systemd user scopes.
- Normalize a Pi JSON event stream into an `AttemptRecord`.
- Complete one artifact task and one independent verifier run end to end.

### Phase 1 — End-to-end vertical slice

- Implement `TaskSpec`, `PolicySpec`, and `AttemptRecord`.
- Implement the disposable attempt runner.
- Add Direct and Deliberate policy definitions.
- Add one artifact generator and hidden verifier.
- Run ten paired attempts end to end.

Exit criterion: seeded task generation, both policy runs, independent verification, and complete normalized records are reproducible.

### Phase 2 — Mixed gym

- Add SQLite generators and verifiers.
- Add Python mutation generators and hidden tests.
- Add shell/filesystem generators and invariant checks.
- Add multiple templates and difficulty controls per family.
- Add verifier determinism and isolation tests.

### Phase 3 — Pilot calibration

- Produce the 240-attempt pilot.
- Inspect success rates and paired-policy disagreement.
- Remove broken, trivial, ambiguous, or impossible templates.
- Adjust policy budgets and task difficulty without using the final test set.

Exit criterion: neither policy dominates everywhere, and policy outcomes differ on a meaningful subset of tasks.

### Phase 4 — Corpus generation

- Freeze generator, policy, Pi, Gemma, and verifier versions.
- Generate the training, validation, and frozen test tasks.
- Run paired attempts and monitor failures and resource use.
- Export normalized training tables while retaining raw trajectories.

### Phase 5 — Outcome-model training

- Train transparent baselines.
- Train the multimodal neural model.
- Add Bayesian-head, adapter, or ensemble uncertainty.
- Calibrate on the validation set.
- Freeze the model and decision rule before test evaluation.

Rollout generation and harness-model training must initially run sequentially. The loaded Gemma server currently consumes roughly 10 GiB of the RTX 3080 Ti's 12 GiB VRAM, leaving insufficient headroom for meaningful NN training. Unload Gemma before the training phase, then reload the pinned model for additional rollouts.

### Phase 6 — Allocator evaluation

- Evaluate on the frozen test set.
- Match the allocator and random-mixture baseline by average cost.
- Report aggregate and family-specific results with task-level uncertainty.
- Record failure modes and decide whether to refine the gym, policies, representation, or uncertainty model.

## 14. Acceptance Criteria

### Engineering acceptance

- All four task families use one gym interface.
- Every task is reproducible from its generator version and seed.
- Verifiers are deterministic and inaccessible to Gemma.
- Pi attempts are isolated and leave no cross-attempt state.
- Raw and normalized trajectory records are complete.
- Exact model, policy, tool, environment, and verifier versions are recorded.

### Dataset acceptance

- Direct and Deliberate both succeed on nontrivial subsets.
- At least approximately 15–20% of paired tasks produce different binary outcomes.
- No private, verifier, or post-treatment feature enters the policy model.
- Test tasks are separated by seed and subtemplate from training data.

### Research go/no-go criterion

The neural allocator should improve verified success over a cost-matched random policy on held-out tasks. A useful target is at least a five-percentage-point improvement, reported with a task-level bootstrap interval. It should also improve probability quality over transparent baselines, or clearly demonstrate where the neural representation adds value.

If these criteria fail, first inspect task diversity, verifier quality, policy dominance, and feature leakage before increasing model complexity.

## 15. Primary Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Direct and Deliberate almost always agree | Calibrate task difficulty and policy budgets before scaling |
| Synthetic shortcuts dominate | Hold out templates, perturb names/content, and use semantic verifiers |
| Hidden oracle leaks to the agent | Separate public workspaces from private verifier storage |
| Generated code attacks the host | Disable Pi built-ins and execute attempts and hidden tests in separate Bubblewrap sandboxes |
| Pi carries state across attempts | Use ephemeral processes, no sessions, and isolated workspaces |
| Mutable default model invalidates comparisons | Resolve, pin, and log exact Pi/Gemma identity per run |
| Pi extension exposes the development host | Register only structured remote gym tools; never expose generic local bash or SSH |
| NN memorizes generator IDs | Exclude private metadata and evaluate on held-out subtemplates |
| Dataset is too small for text fine-tuning | Begin with a pretrained encoder and train smaller fusion/adaptor layers |
| GPU contention or OOM | Unload Gemma and separate rollout-generation from model-training phases |

## 16. Remaining Infrastructure Questions

1. Does the configured llama-server endpoint honor an explicit generation seed, or only sampling parameters?
2. What is the measured tokens-per-second and wall time for Direct and Deliberate rollouts?
3. Which small text encoder and PyTorch/CUDA environment should be installed for harness-model training?
4. What initial token, tool-call, and wall-time limits should define each policy?
5. After the vertical slice, is moving Pi onto `ubuntu-local` worth upgrading Node.js and maintaining a second Pi installation?

The first implementation action is now the remote-worker and Bubblewrap vertical slice. No gym scale-up should begin until path isolation, network isolation, independent verification, event normalization, and reproducibility are demonstrated.

## 17. Implementation Status — 2026-08-09

The Phase 0 vertical slice and the main MVP software path are implemented.

### Implemented

```text
Gym registry
  -> artifact
  -> SQLite
  -> Python repair
  -> shell/filesystem

Execution
  -> Pi 0.84.1 on the development Mac
  -> explicit ubuntu-gemma / gemma-4-26b-a4b selection
  -> project-explicit gym extension; no global or auto-discovered tool changes
  -> persistent SSH JSONL worker
  -> Bubblewrap + systemd user-scope isolation on ubuntu-local

Measurement
  -> paired Direct and Deliberate attempts
  -> independent family-specific semantic verifiers
  -> raw Pi events plus normalized usage records
  -> resumable sequential batch runner
  -> deterministic leakage-safe JSONL dataset exporter

Learning and evaluation
  -> trainable text embedding, categorical embeddings, numeric tower, and fusion MLP
  -> variational Bayesian Bernoulli outcome head
  -> posterior policy counterfactual scoring
  -> paired allocator evaluation against fixed, random-mix, and oracle baselines
  -> privacy-whitelisted standalone HTML research dashboard
```

The restrictive gym `bash` tool is loaded only by explicit orchestrator flags from `pi_extensions/gym-tools.ts`. Pi's built-in tools and unrelated cron-driven Pi sessions are not changed globally or per-directory.

### Verified

- The full suite passes on the configured Linux runner: **445 tests**, including PyTorch, Bubblewrap, network isolation, hidden-verifier isolation, the fixed-page Unbrowser boundary, synthetic theta-model smoke coverage, CLI subprocesses, privacy, and dashboard tests.
- Pi JSON mode reports the pinned provider/model, token usage, tool calls, tool results, and terminal messages.
- Paired Gemma smoke attempts ran on all four task families.
- The fixed-page live Unbrowser runner observes the predeclared negative-control
  failure and positive-control success with exact tool-call traces; this is
  plumbing evidence only.
- The synthetic two-policy theta-model smoke fits, reloads, and scores complete
  panels with the predeclared `extract-h1` ranking. Its labels are generated and
  it is not outcome-model evidence.
- The synthetic descriptor-held-out learn-smoke uses 26 training and 10 held-out
  grammar bundles with identity fields neutralized. Its canonical run improves
  expected allocation lift over random but fails its predeclared held-out
  ranking threshold (`rho=0.238 < 0.3`); it is an explicit non-pass, not
  allocator evidence.
- The resumable batch runner completed a real paired artifact job end to end in approximately 203 seconds.
- Artifact, SQLite, and Python-repair smoke pairs passed under both policies.
- The shell/filesystem smoke pair produced measured verifier failures under both policies, demonstrating useful failure capture but also indicating that this family needs calibration.
- Dataset export, CPU PyTorch training, posterior prediction, allocator evaluation, and standalone dashboard generation completed end to end.

### Smoke results are not evidence

The current smoke corpus contains only 11 usable attempt rows across six tasks. It has no frozen test split, and its allocator smoke evaluation used the validation split that also informed model selection. These outputs establish only that the pipeline works. They must not be used to claim predictive value, calibration, causal policy improvement, or allocator superiority.

### Operational constraint discovered

The same single-slot Gemma service is used by existing cron jobs, including a job scheduled every 15 minutes. The batch runner is intentionally sequential, but queueing can still increase latency or trigger attempt timeouts. Before the 240-attempt pilot:

1. Choose a low-contention execution window or establish a shared model-resource lock without silently modifying other projects.
2. Do not unload Gemma for GPU training while cron jobs may need it.
3. Use CPU training for the small outcome model until coordinated GPU windows exist.

At the measured artifact-smoke rate, 120 paired jobs would require at least roughly seven hours before accounting for harder tasks and cron contention. The full pilot should therefore be treated as a scheduled batch, not an interactive smoke command.

### Next research milestone

Run the predeclared 240-attempt pilot, then enforce these gates before scaling:

```text
verifier determinism and sandbox isolation remain clean
both policies have non-degenerate success rates
approximately 15–20% or more paired outcomes disagree
task-level train/validation/test splits all contain usable pairs
shell-task contract and policy behavior are calibrated
no private or post-treatment field enters model_input
```

Only after those gates pass should the outcome model be retrained and the allocator evaluated once on the frozen test split.
