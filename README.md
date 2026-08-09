# Pyreplab Bayesian Agent Harness

> **Current status:** The end-to-end implementation covers four verifiable task
> families, immutable treatment registries, sequential Pi execution,
> leakage-safe export, descriptor-aware Bayesian outcome modeling, strict
> complete-panel allocator evaluation, and a static dashboard. The full remote
> CPU/PyTorch suite passes 393 tests. No generated-policy corpus or
> leave-one-policy-out result exists yet, so this validates software plumbing,
> not the research thesis.

Detailed documents:

- [`DESIGN.md`](DESIGN.md) — theory, causal scope, uncertainty, and long-term design.
- [`MVP_PLAN.md`](MVP_PLAN.md) — implementation plan, verified environment, acceptance criteria, and current status.
- [`POC_NOTES.md`](POC_NOTES.md) — terminal-gym calibration results, throughput lessons, and the staged v4 proof-of-concept gate.
- [`notes/agent-task-success-dataset-exploration.md`](notes/agent-task-success-dataset-exploration.md) — audited external dataset decision, exclusions, and the offline/native/live harness boundary.

## How to use

### 1. Install and verify the core package

Requirements: Python 3.11 or newer. Linux with Bubblewrap is required for the
real sandbox-isolation path; non-Linux systems can still run the pure unit tests
and data/model utilities.

```bash
git clone https://github.com/protostatis/pyreplab-bayesian-agent-harness.git
cd pyreplab-bayesian-agent-harness

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

python -m unittest discover -s tests -p "test_*.py" -v
```

Generate a deterministic local task to confirm the CLI works:

```bash
pyreplab-harness generate \
  --family artifact \
  --root .runs/demo \
  --seed 7 \
  --difficulty easy
```

The core gym has no required third-party Python dependencies. Neural training
and allocator evaluation additionally require the pinned CPU environment:

```bash
python -m pip install -r requirements-train.txt
```

### 2. Configure optional Pi + SSH execution

End-to-end agent execution is optional. It requires:

- a disposable Linux SSH host with Python 3.11+ and Bubblewrap;
- the Pi CLI installed on the controller machine;
- a Pi provider/model already configured by the operator; and
- absolute remote project and run-root paths owned by a least-privilege user.

Copy the placeholder configuration and edit it locally. `.env` is ignored and
must never contain credentials committed to Git.

```bash
cp .env.example .env
set -a
source .env
set +a

bash scripts/deploy_ubuntu.sh \
  "$PYREPLAB_HARNESS_HOST" \
  "$PYREPLAB_REMOTE_PROJECT"
```

Run one paired Direct/Deliberate task:

```bash
python -m pyreplab_harness.orchestrator \
  --host "$PYREPLAB_HARNESS_HOST" \
  --remote-project "$PYREPLAB_REMOTE_PROJECT" \
  --remote-run-root "$PYREPLAB_REMOTE_RUN_ROOT" \
  --provider "$PYREPLAB_PI_PROVIDER" \
  --model "$PYREPLAB_PI_MODEL" \
  --family artifact \
  --difficulty easy \
  --seed 21 \
  --pair
```

If your Pi setup needs a provider-switch extension, set
`PYREPLAB_MODEL_SWITCH_EXTENSION` in `.env`; otherwise leave it empty.

### 3. Try the generalized treatment path

```bash
python -m pyreplab_harness.treatments generate \
  .runs/generated-treatments.json --count 4 --seed 42

python -m pyreplab_harness.orchestrator \
  --host "$PYREPLAB_HARNESS_HOST" \
  --remote-project "$PYREPLAB_REMOTE_PROJECT" \
  --remote-run-root "$PYREPLAB_REMOTE_RUN_ROOT" \
  --provider "$PYREPLAB_PI_PROVIDER" \
  --model "$PYREPLAB_PI_MODEL" \
  --family artifact --difficulty easy --seed 21 \
  --treatment-registry .runs/generated-treatments.json \
  --treatments all
```

Each treatment is an immutable prompt/tool/budget bundle. `--treatments all`
runs every selected bundle sequentially on the same task and can be expensive.

> **Safety:** Agent-authored commands execute on the configured SSH host inside
> the Linux sandbox path. Use a disposable non-production host, a
> least-privilege SSH identity, strict resource limits, and an environment that
> contains no secrets. Treat `.runs/`, datasets, trajectories, and model
> artifacts as potentially sensitive. See [`SECURITY.md`](SECURITY.md).

This is the short thesis. See [DESIGN.md](./DESIGN.md) for the detailed working notes, definitions, model framing, research context, and validation direction.

## Thesis

An agent harness should not only execute a user request. It should learn how to allocate the agent's reasoning, tool, verification, and human-escalation budget so that verified task completion becomes more likely.

Pyreplab is the computational workspace for this loop: it preserves intermediate computation, evidence, and model state while a task is in progress. The harness is the control layer above it.

The core claim is:

```text
Given a task state and a set of available execution policies,
the harness can learn which intervention is most likely to improve
the probability of verified task completion within a budget.
```

This is not a domain model of Zillow, debugging, research, or travel. It is a model of how an agent's execution policy affects task outcomes.

## Minimal Bayesian Neural Formulation

For task attempt `i`:

```text
x_i      = information known before the policy decision
           (task contract, current trajectory/history, environment, budget)

pi_i     = selected execution policy

Y_i      = terminal verified outcome
           1 = task contract satisfied
           0 = task contract not satisfied

theta    = latent neural-network weights

q_theta(x_i, pi_i)
         = neural prediction of P(Y_i = 1 | x_i, pi_i, theta)
```

The prior is a distribution over neural weights:

```text
p0(theta)
```

Historical, verified attempts form the dataset:

```text
D = {(x_i, pi_i, Y_i)} for i = 1...N
```

The posterior is:

```text
p(theta | D)
  proportional to
p0(theta) * product_i P(Y_i | x_i, pi_i, theta)
```

For a new task, the harness predicts success by averaging across plausible weight settings:

```text
P(Y = 1 | x, pi, D)
  = integral of P(Y = 1 | x, pi, theta) * p(theta | D)
    over theta
```

## Posterior Contraction

The goal is not to make every individual neural weight certain. Many weight settings can express the same function.

The goal is to reduce uncertainty about the *predicted success function* where verified evidence supports it:

```text
familiar task state + familiar policy + reliable verification
  -> narrower predictive uncertainty

novel task state, policy, environment, or contract
  -> uncertainty remains wide
```

Strictly, data does not "shrink the prior." It updates the prior `p0(theta)` into a posterior `p(theta | D)`. The desired property is calibrated posterior contraction: confidence narrows only where evidence and transferable similarity justify it.

## Control Claim

Prediction alone says whether a task appears difficult. Control requires estimating how a chosen intervention changes the outcome distribution:

```text
P(Y = 1 | current state, do(policy = pi))
```

The harness then chooses among policies such as direct execution, verify-first, compute-first, ask-the-user, delegate, or defer, subject to budget, latency, and safety constraints.

The first validation target is intentionally small: compare a fixed, low-cost policy against a fixed, more deliberate policy using predeclared task contracts and verified outcomes.

## Implemented MVP

```text
procedural task generator
  -> artifact | SQLite | Python repair | shell/filesystem task
  -> fresh Bubblewrap workspace on a configured Linux SSH host
  -> Direct and Deliberate Pi/Gemma attempts
  -> independent semantic verifier
  -> leakage-safe paired dataset
  -> multimodal neural representation + variational Bayesian head
  -> held-out allocator evaluation
  -> standalone static HTML dashboard
```

Security boundaries include disabled Pi host tools, an explicit sandbox-only replacement `bash` tool, unshared task networking, hidden verifier bundles, separate Python-verifier isolation, and no unsafe verifier fallback.

## Quick Verification

Run the local tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p "test_*.py" -v
```

Deploy to the SSH host configured in `.env` and run the Linux isolation suite
(`ubuntu-local` is only an example SSH alias):

```bash
set -a; source .env; set +a
bash scripts/deploy_ubuntu.sh \
  "$PYREPLAB_HARNESS_HOST" "$PYREPLAB_REMOTE_PROJECT"
```

Run one paired task:

```bash
PYTHONPATH=src python3 -m pyreplab_harness.orchestrator \
  --host "$PYREPLAB_HARNESS_HOST" \
  --remote-project "$PYREPLAB_REMOTE_PROJECT" \
  --remote-run-root "$PYREPLAB_REMOTE_RUN_ROOT" \
  --provider "$PYREPLAB_PI_PROVIDER" \
  --model "$PYREPLAB_PI_MODEL" \
  --family artifact --difficulty easy --seed 21 --pair
```

## Generalized Treatment MVP

Generate a deterministic sample from the controlled policy grammar:

```bash
PYTHONPATH=src python3 -m pyreplab_harness.treatments generate \
  .runs/generated-treatments.json --count 4 --seed 42

PYTHONPATH=src python3 -m pyreplab_harness.treatments inspect \
  .runs/generated-treatments.json
```

Each registry entry freezes its exact system prompt, tools, interface, budgets,
generator factors, and canonical SHA-256 bundle identity. Execute a predeclared
menu on one generated task with:

```bash
PYTHONPATH=src python3 -m pyreplab_harness.orchestrator \
  --family artifact --difficulty easy --seed 21 \
  --treatment-registry .runs/generated-treatments.json \
  --treatments all
```

`--treatments all` can be expensive: each registry treatment receives a fresh
attempt on the same task and the shared model has one inference slot. Use a
small registry and a dedicated `--remote-run-root`.

The resumable batch runner accepts the same registry for a task matrix:

```bash
PYTHONPATH=src python3 -m pyreplab_harness.batch \
  --families artifact,sqlite \
  --difficulties easy,medium \
  --seeds 1-3 \
  --treatment-registry .runs/generated-treatments.json \
  --treatments all \
  --host "$PYREPLAB_HARNESS_HOST" \
  --remote-project "$PYREPLAB_REMOTE_PROJECT" \
  --remote-run-root "$PYREPLAB_REMOTE_PROJECT/.runs/generated-policy-mvp" \
  --provider "$PYREPLAB_PI_PROVIDER" \
  --model "$PYREPLAB_PI_MODEL" \
  --output .runs/generated-policy-mvp.jsonl
```

Export verified attempts with the same registry so the dataset contains
separate task and treatment descriptors:

```bash
PYTHONPATH=src python3 -m pyreplab_harness.dataset \
  <run_root> .runs/generated-policy-dataset.jsonl \
  --treatment-registry .runs/generated-treatments.json
```

The treatment-aware model combines a closed-set policy identity with a separate
encoder over prompt text, tools, interface, and budgets. Unseen IDs map to
`UNK`, while their descriptors can still map to different representations.
This is an experimental mechanism, not evidence of valid unseen-policy
generalization; that requires leave-one-policy-out outcomes.

Train and evaluate the descriptor-enabled allocator only after every evaluated
task has exactly one verified row for every candidate treatment:

```bash
PYTHONPATH=src python3 -m pyreplab_harness.outcome_model train \
  .runs/generated-policy-dataset.jsonl \
  .runs/generated-policy-model

PYTHONPATH=src python3 -m pyreplab_harness.allocator_eval \
  .runs/generated-policy-dataset.jsonl \
  .runs/generated-policy-model \
  .runs/generated-policy-evaluation.json \
  --split test \
  --treatment-registry .runs/generated-treatments.json \
  --treatments all
```

Generalized evaluation chooses the largest saved posterior-predictive mean per
task. It has no implicit resource quota: observed tokens/tool calls are
reporting-only. Missing cells, duplicate attempts, descriptor/hash drift,
cross-split task IDs, or unequal task-side inputs abort evaluation rather than
silently changing the evaluated population. `hindsight_realized_oracle` is an
optimistic ceiling on this one-attempt panel, not a causal oracle.

Run a resumable sequential pilot:

```bash
PYTHONPATH=src python3 -m pyreplab_harness.batch \
  --families artifact,sqlite,shell,python_repair \
  --difficulties easy,medium,hard \
  --seeds 1-10 \
  --output .runs/pilot.jsonl
```

The reference environment uses one inference slot. If your endpoint is shared,
schedule large pilots deliberately and avoid model reloads or concurrent jobs
that would confound timing and outcomes.

When the native experiment is resumed, run or resume the smaller stratified
proof-of-concept gate before the full pilot:

```bash
nohup bash scripts/run_poc_v4.sh >.runs/poc-v4.log 2>&1 &
```

This gate contains 24 paired tasks with preselected train/validation/test
coverage. Policy versions are separate treatments; never append another policy
version to the same batch output. It is currently paused after five completed
pairs; do not resume it automatically while the external-data decision is being
validated.

After all 24 pairs finish, gate and finalize the CPU-only model demonstration:

```bash
bash scripts/finalize_poc_v4.sh
```

The current ad-hoc ToolFailBench dataset/model/CV artifacts are quarantined:
their model input contains privileged mock-return and target-mode fields, and
the outer validation fold was reused for early stopping. Do not cite or rerun
those metrics until the importer is rebuilt with a semantic allowlist and
nested grouped evaluation.

## Reproducible environment for experiments

For ad-hoc experiments (including the ToolFailBench CV run), create a local, pinned
environment before running any training/evaluation so results are rerunnable:

```bash
cd /path/to/pyreplab-bayesian-agent-harness
uv venv .venv --python 3.9
source .venv/bin/activate
uv pip install -r requirements-cv.txt
PYTHONPATH=src
```

Then run the same commands/scripts against explicit artifacts and capture:

- pinned dependency freeze (`uv pip freeze --python .venv/bin/python`),
- dataset hash, and
- exact CLI/seed parameters.

For this run’s reproducibility notes and data hash, see:
`notes/adhoc-toolfailbench-cv-reproducibility.md`.

You can also sanity-check CV splits for task-level leakage:

```bash
cd /path/to/pyreplab-bayesian-agent-harness
./scripts/inspect_toolfailbench_cv.sh .runs/adhoc-toolfailbench-cv 5 5 1
```

For Bayesian posterior diagnostics (prior/posterior head diagnostics and sampled
predictive metrics), use the new official `inspect` CLI subcommand:

```bash
cd /path/to/pyreplab-bayesian-agent-harness
python3 -m pyreplab_harness.outcome_model inspect .runs/adhoc-toolfailbench-model \
  --dataset .runs/adhoc-toolfailbench-dataset.jsonl \
  --posterior-samples 64 --prior-samples 64 --max-rows 400
```

or use the compatibility wrapper:

```bash
./scripts/inspect_bayesian_fit.sh .runs/adhoc-toolfailbench-model \
  --dataset .runs/adhoc-toolfailbench-dataset.jsonl \
  --posterior-samples 64 --prior-samples 64 --max-rows 400
```

## License

No open-source license has been selected yet. Public visibility of the source
does not grant permission to copy, modify, or redistribute it.
