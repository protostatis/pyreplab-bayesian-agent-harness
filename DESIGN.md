# Pyreplab Bayesian Agent Harness — Detailed Working Notes

> **Status:** exploratory concept document, not an implementation specification.
>
> The aim is to define the thesis and mathematical objects clearly before choosing a framework, model architecture, or product scope.

## 1. The High-Level Thesis

Most agent harnesses execute a task with a fixed or heuristic loop. This concept proposes a harness that learns how to allocate an agent's limited resources:

```text
reasoning
tool calls
verification
computation
delegation
human escalation
time, token, money, and risk budget
```

The central claim is:

```text
Given a task state and a set of available execution policies,
the harness can learn which intervention is most likely to improve
the probability of verified task completion within a budget.
```

This is not primarily a domain model of Zillow, travel, debugging, research, or procurement. It is a model of the relationship between an agent's execution choices and the resulting task outcome.

At the highest level, the harness is an **epistemic control plane**:

```text
task
  -> current task state
  -> choose how to think, search, compute, verify, ask, act, or stop
  -> observe results
  -> update the task state and long-run model
  -> produce or revise a recommendation
```

## 2. What This Is and Is Not

### It is

- A learned controller for allocating an agent's execution budget.
- A system that treats verified task outcomes as learning signals.
- A way to distinguish "this task looks difficult" from "this policy is likely to help."
- A compute-and-evidence workspace plus a control policy.

### It is not

- A universal model of the external world.
- A claim that every task can be reduced to a real-estate-style feature table.
- A generic agent framework replacement.
- A guarantee that uncertainty estimates make a system safe.
- A claim that a neural network or Bayesian vocabulary automatically produces calibrated decisions.

## 3. System Layers

```text
User / conversation layer
  - supplies the request, preferences, constraints, and approvals

Task-contract layer
  - declares what completion means for this attempt

Harness control layer
  - selects an execution policy and later selects next actions

Pyreplab compute layer
  - stores calculations, evidence, intermediate artifacts, and model state

Tools / environment layer
  - browser, search, files, APIs, tests, code execution, human input

Verification layer
  - decides whether declared success criteria were met

Learning layer
  - updates beliefs about how policies perform in task contexts
```

Pyreplab is important, but it is not the whole harness. It is the agent's computational instrument and working surface. The harness decides how that instrument should be used.

## 4. Task Semantics

### 4.1 Raw prompt and task contract

```text
u_i = raw user prompt for task attempt i

c_i = explicit task contract derived from u_i
```

The raw prompt is natural language. The contract is a compact statement of what can later be evaluated.

For example, a Zillow-like task contract could require:

```text
- satisfy stated hard constraints
- provide evidence and freshness information
- state the ranking criterion
- disclose material uncertainty
- return the requested decision artifact
```

It should not initially claim to verify an unobservable statement such as "objectively best house."

### 4.2 Trajectory and state

```text
e_i     = execution environment for attempt i
          (available tools, model version, permissions, limits)

a_i,t   = action taken at time t
          (search, compute, verify, ask, delegate, act, stop, etc.)

o_i,t   = observation received at time t
          (tool result, user response, file content, test result, verifier result)

h_i,t   = history known before action a_i,t

tau_i   = complete trajectory for attempt i
```

In plain text:

```text
h_i,t =
  (u_i, c_i, e_i, o_i,0, a_i,0, o_i,1, ..., a_i,t-1, o_i,t)

tau_i =
  (h_i,0, a_i,0, o_i,1, ..., a_i,T-1, o_i,T)
```

Only information available before a decision may be used to choose that decision. Later observations can guide later actions, but cannot justify an earlier choice retroactively.

## 5. Terminal Outcome and `Y`

Use a deliberately narrow definition first:

```text
Y_i = terminal verified task-success label

Y_i = 1  -> the task contract was satisfied
Y_i = 0  -> the task contract was not satisfied
Y_i = ?  -> unresolved, censored, or not yet verified
```

`Y` is not the entire task outcome. The full terminal result may be represented as:

```text
R_i = rich outcome vector
      (quality, cost, latency, safety events, user acceptance,
       verifier result, retry count, artifacts, and failure mode)

Y_i = contract-level binary label derived from R_i and tau_i
```

Equivalently:

```text
Y_i = V(c_i, tau_i)
```

where `V` is the declared verifier or evaluation process.

This separation matters:

```text
o_t   = intermediate observation
tau   = full sequence of actions and observations
R     = rich terminal outcome
Y     = did the attempt satisfy the task contract?
```

## 6. Bayesian Neural Formulation

### 6.1 Parameters and prior

```text
theta   = latent neural-network weights
p0(theta)
        = prior distribution over possible weight settings
```

The prior is not "the weights." The weights are the latent parameters; the prior is a probability distribution over them.

An ordinary neural network learns one point estimate of `theta`. A Bayesian neural formulation maintains or approximates a posterior distribution:

```text
p(theta | D)
```

### 6.2 Minimal discriminative model

For a first policy-selection model:

```text
x_i      = information available before selecting a policy
            (task contract, trajectory prefix, environment, budget)

pi_i     = selected execution policy

q_theta(x_i, pi_i)
          = P(Y_i = 1 | x_i, pi_i, theta)
```

Historical verified attempts form the dataset:

```text
D = {(x_i, pi_i, Y_i)} for i = 1...N
```

The posterior is:

```text
p(theta | D)
  proportional to
p0(theta) * product_i P(Y_i | x_i, pi_i, theta)
```

For a new task, the posterior predictive success probability is:

```text
P(Y = 1 | x, pi, D)
  = integral of P(Y = 1 | x, pi, theta) * p(theta | D)
    over theta
```

This is the simplest useful reading of the model:

```text
X tells the model what kind of decision context it is seeing.
Y tells the model whether the chosen policy ultimately worked.
theta stores shared, learned structure across attempts.
```

### 6.3 Trajectory model

The more general formulation proposed in this discussion models the distribution of trajectories, not only a binary label:

```text
p_theta(tau_i | u_i, c_i, e_i, pi_i)
```

That means:

```text
Given the task, contract, environment, and policy,
the model assigns probability to possible execution trajectories.
```

A stepwise factorization is:

```text
p_theta(tau_i | u_i, c_i, e_i, pi_i)
  = product over time t of:

      pi_i(a_i,t | h_i,t)
      *
      p_theta(o_i,t+1 | h_i,t, a_i,t)
```

In this formulation, `Y_i` may be a deterministic evaluation of the completed trajectory:

```text
Y_i = V(c_i, tau_i)
```

The discriminative success model is a practical first approximation to this richer trajectory model.

### 6.4 Current task inference

For a current unfinished task, the harness has an observed history prefix `h_*,t`. It predicts possible futures under a candidate policy:

```text
p(future trajectory, final Y
  | h_*,t, candidate policy, D)
```

Conceptually:

```text
past verified trajectories
  -> posterior over shared model weights

current trajectory prefix
  -> condition the current prediction

candidate policy
  -> predicts a different distribution of possible futures
```

## 7. Global Parameters Versus Current-Task State

The harness should distinguish two kinds of uncertainty.

```text
theta
  = long-lived shared model parameters
  = learned slowly across many attempts

z_t
  = latent state of the current unfinished task
  = updated quickly as the current trajectory produces observations
```

At a high level, `z_t` may represent:

```text
what remains unknown
which interpretation of the task is plausible
which strategy is viable
whether the task is likely to complete within budget
```

The intended distinction is:

```text
The current task should update the current-task belief state.
Verified task histories should gradually update the shared model.
```

This avoids treating every tool result as a reason to immediately rewrite the global neural weights.

## 8. Posterior Contraction

The phrase "shrink the uncertainty of the prior" should be stated more precisely:

```text
Evidence does not shrink the prior itself.
Evidence updates p0(theta) into p(theta | D).
```

The goal is not to make every individual neural weight certain. Neural networks can have many functionally equivalent weight settings.

The desired property is **posterior contraction in predictive function space**:

```text
For familiar task states and policies with reliable evidence,
the posterior distribution over predicted success should narrow.

For novel tasks, policies, environments, or contracts,
uncertainty should remain wide or trigger a conservative response.
```

The meaningful object is:

```text
q_theta(x, pi) = P(Y = 1 | x, pi, theta)
```

not the variance of each raw coordinate of `theta`.

## 9. Uncertainty Taxonomy

The harness should not collapse all uncertainty into one scalar confidence score.

| Type | Meaning | Desired response |
|---|---|---|
| Aleatoric uncertainty | Outcome remains variable even with a known model | Act if one policy still dominates; do not expect more modeling to remove it |
| Epistemic uncertainty | Insufficient evidence about the success function in this region | Verify, explore, collect data, or defer |
| OOD/model uncertainty | The context is outside reliable training support or the model is misspecified | Use a conservative policy; do not trust a narrow-looking posterior alone |
| Decision uncertainty | Plausible models recommend different actions | Gather decision-relevant evidence or ask the user |
| Safety uncertainty | A low-probability but unacceptable outcome is possible | Treat as a constraint or escalation condition, not merely an average penalty |

For binary `Y`, a single mean success probability is insufficient. The harness should retain uncertainty about the success function `q_theta(x, pi)`, not only a marginal probability that success occurs.

## 10. From Prediction to Control

Prediction alone is not control.

```text
P(Y = 1 | current state)
```

may only reveal that a task looks easy or hard.

The control claim requires a policy-specific, causal quantity:

```text
P(Y = 1 | current state, do(policy = pi))
```

In plain language:

```text
If the harness selects this policy now,
how does that change the distribution of possible task outcomes?
```

The harness controls its interventions, not the world directly:

```text
current state H_t
  + chosen intervention A_t
  -> next observation O_t+1
  -> updated state H_t+1
  -> later intervention
  -> terminal outcome Y
```

The complete theory contains three increasingly demanding bets:

```text
Prediction bet:
Can the harness predict which attempts will succeed?

Control bet:
Can it predict which policy improves success for this context?

Generalization bet:
Do those policy effects transfer to new task families?
```

Only the control and generalization bets establish a genuinely smart harness.

## 11. Learning and Evaluation Contract

To learn policy effects rather than accidental correlations, the harness needs a disciplined record for each attempt:

```text
- immutable task contract and success criteria
- verifier identity and version
- decision-time state snapshot and feature-schema version
- full candidate policy set available at the decision point
- selected policy and its assignment probability
- model, prompt, tool, and environment versions
- offered and consumed token/time/money budget
- safety events and permission state
- terminal result, verifier result, and censoring/timeout reason
- task family and repeated-attempt grouping
- prediction recorded before post-action evidence arrives
```

The assignment probability matters because historical agent logs are usually selective:

```text
expensive policies may be assigned to difficult tasks
easy tasks may never receive verification
deferred tasks may never receive terminal labels
retries on one task are not independent examples
```

For a policy allocator, the first credible data collection design is controlled or randomized policy assignment within a stable policy menu.

## 12. Modeling Stance

### Neural representation versus Bayesian scope

A simple terminal outcome distribution does not require a simple predictor. The binary outcome may remain:

```text
Y_i | x_i, pi_i, theta
  ~ Bernoulli(q_theta(x_i, pi_i))
```

while `q_theta` is a multimodal neural model. This matters because the decision-time state may combine raw text, a variable-length trajectory, derived numbers, categorical metadata, environment state, and a candidate policy. A useful decomposition is:

```text
text and trajectory       -> pretrained text/trajectory encoder
derived numeric features  -> numeric encoder with missingness indicators
environment and budget    -> structured embeddings
candidate policy          -> policy embedding
                              |
                              v
                     multimodal fusion
                              |
                              v
               policy-conditioned outcome heads
```

Conceptually:

```text
z_x = F_psi(
        E_text(x_text),
        E_num(x_numeric),
        E_env(x_environment)
      )

q(x, pi) = sigmoid(g_phi(z_x, E_policy(pi)))
```

The neural representation, the Bayesian scope, and causal identification are separate design choices:

```text
Neural representation
  -> compresses heterogeneous, partly unstructured task state

Bayesian inference
  -> represents epistemic uncertainty over some or all of the
     representation, fusion, and outcome-model parameters

Causal identification
  -> comes from controlled assignment or defensible causal assumptions;
     it is not supplied by the neural network
```

A neural outcome model may therefore be important from the first contextual allocator even when a full weight-space BNN is not yet practical. An initial implementation can use a pretrained encoder with trainable adapters, a numeric tower, policy-conditioned fusion, and Bayesian or ensemble treatment of the adapters and outcome heads. A Bayesian last layer is cheaper, but may miss representation uncertainty, so it should be paired with explicit OOD/support checks and compared with encoder or adapter ensembles.

### Ideal long-term model

```text
A constrained, partially observed sequential control model:

- belief over current task state
- posterior over shared model parameters
- outcome and trajectory predictions
- policy-specific value estimates
- explicit cost, latency, risk, and human-approval constraints
- decision-aware exploration
```

### Practical first model

Do not confuse the need for neural representation with a requirement to place an approximate posterior over every neural-network weight from the start.

Start with:

```text
- two or a few fixed execution policies
- versioned task contracts and verifiers
- randomized or carefully logged policy assignment
- a hierarchical Bayesian logistic model or GAM as a transparent baseline
- a pretrained text/trajectory encoder when raw text or variable-length
  history is part of the decision state
- a numeric-feature tower and policy-conditioned fusion
- Bayesian adapters, a Bayesian outcome head, or a bootstrapped ensemble
- calibrated samples or intervals for policy-specific success
- a separate support/OOD signal
- explicit verify/defer/ask-the-user behavior
```

As evidence and data volume grow, the next step may be:

```text
- jointly fine-tuned task/trajectory adapters
- richer multimodal and policy-conditioned interactions
- broader Bayesian treatment of representation and fusion parameters
- explicit calibration and temporal drift monitoring
```

A full BNN is justified when it materially improves calibration, selective risk, and decision utility over partial-Bayesian and ensemble alternatives. Neural representation itself does not depend on clearing that bar.

## 13. Role of Pyreplab

Pyreplab is useful as the agent's local computational workbench:

```text
- preserve variables and expensive computations across commands
- maintain evidence tables and intermediate artifacts
- run model updates and simulations
- keep an execution history while a session is active
- support incremental analysis rather than repeated process startup
```

Important current boundary:

```text
Pyreplab preserves its Python namespace while the daemon remains alive.
It is not yet a durable state system across daemon termination and restart.
```

`history.md` can support recovery and auditing, but a production harness should explicitly persist its decision state, evidence ledger, model version, and artifacts. It should not rely on raw REPL history alone as a durable recovery mechanism.

Pyreplab also does not automatically record browser navigation, external-tool provenance, or decisions. The harness must intentionally write normalized task state and evidence into its workspace.

## 14. Zillow as a Reference Scenario, Not the Model

The Zillow task is useful because it exposes the desired control behavior:

```text
User request:
"Find the best deal for a 4 bed / 3 bath house for sale now in 60451."
```

The generalized harness should reason about possible policies, not hard-code a property taxonomy:

```text
- direct listing search and rank
- ask the user to resolve an ambiguous definition of "best deal"
- verify freshness/status before deeper research
- gather comparable evidence before ranking
- stop when further evidence is unlikely to change the recommendation
- defer when evidence quality is inadequate
```

The underlying product question is:

```text
Which intervention is most likely to improve the task contract outcome,
given the present task state and limited budget?
```

Live Zillow is not an ideal first benchmark because data access can be blocked, facts change, and "best deal" lacks immediate ground truth. A frozen Zillow-like environment with staged evidence and a known evaluator is better for an early controlled experiment.

## 15. Existing Landscape and Positioning

Several adjacent systems already exist:

| Capability | Examples | Relation to this concept |
|---|---|---|
| Durable state and orchestration | [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview), [Pydantic AI durable execution](https://pydantic.dev/docs/ai/capabilities/durable_execution/overview/), [AutoGen state](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/state.html), [OpenAI Agents SDK sessions](https://openai.github.io/openai-agents-python/sessions/) | Provide useful runtime substrate; do not natively provide a general utility/VoI decision controller |
| Stateful agent memory | [Letta](https://github.com/letta-ai/letta) | Strong memory focus; not a general policy-value model |
| Agent code sandboxes | [E2B](https://e2b.dev/docs) | Provides execution environment, not decision control |
| Bayesian cross-task skill evolution | [Bayesian-Agent](https://github.com/DataArcTech/Bayesian-Agent) | Closest reference; currently updates beliefs about Skills/SOPs across verified trajectories |
| Bayesian uncertain reasoning | [BayesAgent / vPGM](https://arxiv.org/abs/2406.05516) | Addresses probabilistic reasoning under uncertainty, not a full durable tool-task control harness |
| Information-seeking policy theory | [pymdp](https://github.com/infer-actively/pymdp) | Formal active-inference machinery for explicit MDP environments, not a practical LLM/web-agent harness |

The positioning should therefore not be simply "a Bayesian agent harness." A more specific hypothesis is:

```text
An evidence-backed, within-task decision control plane for agents.
```

The closest Bayesian-Agent project is especially important. Its current model treats Skills and SOPs as hypotheses about success across verified trajectories. Its published roadmap includes richer uncertainty-aware Skill selection and Bayesian decision policies. This makes it both a valuable reference and a potential future overlap.

## 16. First Validation Experiment

The smallest defensible experiment is a one-decision policy allocator, not full sequential reinforcement learning.

```text
Policy A:
direct, low-cost execution

Policy B:
more deliberate execution
(for example: verify-first, pyreplab-assisted, or ask-before-act)
```

For a predeclared family of tasks:

```text
1. Define a task contract and independent verifier.
2. Randomly assign A or B, or log defensible assignment propensities.
3. Record cost, latency, safety events, and verified outcome.
4. Fit a simple policy-specific success model.
5. Freeze it.
6. Compare its allocation decisions with the best fixed policy at equal average cost.
7. Test on future tasks and at least one held-out task family.
```

Report separately:

```text
- verified success
- user acceptance, if available
- cost and latency
- false-safe rate
- deferral / verification rate
- calibration and reliability
- decision utility or regret
- behavior under task, policy, verifier, and time shifts
```

## 17. Open Questions

1. What is the first stable task family with an auditable verifier?
2. What are the first two fixed policies worth comparing?
3. What information is legitimately available at each decision point?
4. Which outcomes are verified, user-rated, censored, or delayed?
5. What representation of task context can transfer without erasing important minority-task differences?
6. What should force verification, user escalation, or refusal to act?
7. Is the early product about within-task control, cross-task skill learning, or a deliberate combination of both?
8. What durable artifact must exist beyond Pyreplab's live session so another agent or later session can resume safely?

## 18. Current Decision

The next step is not to build a full neural controller.

The next step is to define a narrow, verifiable two-policy experiment and the associated evaluation contract. That experiment should determine whether policy-specific success prediction contains useful signal before the project expands into trajectory models, Bayesian neural networks, or sequential control.
