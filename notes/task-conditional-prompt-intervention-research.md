# Task-Conditional Prompt Interventions in Tool-Using LLM Agents

Date: 2026-08-15

Research cutoff: 2026-08-15

Status: **complete-fit-for-purpose research synthesis; candidate methodological
positioning only; no live model execution is authorized by this document**

## 0. One-Sentence Conclusion

The harness should not claim to invent prompt optimization or prompt routing;
its defensible research direction is a controlled protocol that first tests
prompt-only manipulation and average lift in a frozen tool agent, then requires
any complementary prompt ordering to replicate on disjoint tasks before testing
with selection-valid inference whether a pre-action router beats every constant
overlay on structurally held-out tasks. Empty-overlay and locked-comparator
results remain separately reported operational comparisons.

## 1. Decision Context

The harness thesis in [`DESIGN.md`](../DESIGN.md) is broader than ordinary
success prediction:

```text
Given a task state and a set of available execution policies,
the harness can learn which intervention is most likely to improve
the probability of verified task completion within a budget.
```

This report asks the narrow question that should be resolved before building a
general policy allocator:

> Can information available before execution predict which immutable system-
> prompt overlay improves a frozen tool-using agent's verified task success,
> and can the resulting router outperform the constant-overlay class on fresh
> tasks, with no-overlay and locked-comparator results reported separately?

The next decision this research must support is whether to invest in a prompt-
policy router, continue only with universal policy optimization, or stop this
branch because there is no stable prompt-responsive heterogeneity to route.

### 1.1 In scope

- One policy decision before a tool-using attempt starts.
- A finite library of exact, immutable system-prompt overlays.
- The same frozen model, base prompt, tools, interface, runtime, verifier, and
  offered resource limits across prompt-only arms.
- Binary external terminal verification plus separately recorded cost and
  failure mechanisms.
- Complete crossed outcome collection on fresh tasks, with attempt slots
  randomized or counterbalanced; or known-propensity randomized incomplete
  blocks at larger scale.
- Task-clustered treatment effects, structural holdouts, and fixed-policy
  comparators.

### 1.2 Out of scope

- Arbitrary prompt generation during deployment.
- Per-step sequential control within an attempt.
- Model routing, model fine-tuning, or changing tool interfaces by arm.
- Treating larger budgets, extra tools, or a different cache mode as prompt-only
  interventions.
- Claims about arbitrary future task families, public websites, or models.
- Using the completed empty-overlay screen as prompt-lift evidence or training
  data.

## 2. Research Run

| Field | Value |
|---|---|
| Mode | Normal Research plus Application |
| Depth | L4 applied research |
| Audience | Harness research and engineering |
| Output | Maintainable repository Markdown report |
| Channels | Primary papers, canonical proceedings, author preprints, canonical repositories, and immutable local experiment notes |
| Access | Public, read-only sources only; no login, paywall bypass, private data, or third-party code execution |
| Evidence contract | Every Must claim requires a locatable primary source or immutable local artifact; preprints are marked; same-lineage work is not counted as independent corroboration |
| Terminal status | `complete-fit-for-purpose` |
| Stop reason | The literature is sufficient to reject broad novelty, define the correct causal objects, and specify the next falsification gate; remaining uncertainty is empirical rather than resolvable by more literature alone |

Evidence labels used below:

| Label | Meaning |
|---|---|
| A1 | Peer-reviewed or canonical primary research with directly relevant method or result |
| A2 | Primary preprint or technical report; useful but provisional |
| L | Immutable local experiment or design artifact |
| S | Synthesis or design implication derived from cited evidence, not a source's own claim |

## 3. Core Findings

1. **Broad algorithmic novelty is ruled out.** Global prompt optimization,
   per-input instruction selection, mixture-of-prompt routing, strategy routing,
   contextual-bandit system-prompt routing, and learned harness configuration
   already exist.
2. **The exact evaluation conjunction remains differentiated, not proven
   unique.** The candidate contribution is prompt-only isolation in a frozen
   multi-turn tool agent, external terminal verification, replicated task-arm
   outcomes with task-clustered analysis, structural holdouts, and selection-
   valid comparison with empty and fixed policies.
3. **Prompt difficulty is not prompt headroom.** The valid V5 empty-overlay
   screen identifies challenge candidates, but no non-empty prompt was tested in
   that screen. It provides no causal prompt-effect or routeability evidence.
4. **Oracle opportunity is not realizable value.** A hindsight oracle can exploit
   outcomes unavailable to a deployable router. The relevant endpoint is held-
   out policy value using only pre-action signals, with selection-valid
   superiority to the full constant-action class.
5. **The unit hierarchy must remain explicit.** A rollout attempt is the
   treatment-execution unit. The task instance is the primary clustering and
   generalization unit for a new-task claim within represented generators.
   Replicas improve task-arm measurement; they do not create more tasks,
   templates, or families.
6. **Prediction and treatment-effect estimation are different.** A model can
   predict easy and hard tasks accurately while learning no prompt-by-task
   interaction and producing no routing gain.
7. **Universal optimization and conditional routing are separate claims.** The
   current Policy Lab searches for one policy over a frozen target mixture. A
   router learns a function from pre-treatment task features to a policy. They
   can share infrastructure and explicitly designated development evidence, but
   not estimands, sealed evaluation data, or terminal claims.
8. **Do not build a router yet.** The next scientific gate is a small excluded,
   crossed, prompt-only headroom pilot. Router fitting is justified only after
   pilot evidence of manipulation and lift plus complementary ordering that
   replicates on disjoint tasks.

## 4. Concept Map And Terminology

```text
prompt optimization
  one prompt or program configuration for a task distribution
  examples: APE, OPRO, MIPRO
                 |
                 v
per-input prompt or strategy selection
  choose prompt content, demonstrations, or a scaffold from input features
  examples: Auto-Instruct, APS, MoP, Adaptive-RAG
                 |
                 v
online prompt or harness routing
  learn a policy over prompt/harness actions from outcome feedback
  examples: CCLUB, frozen-agent harness control, adaptive tutoring
                 |
                 v
causal policy learning
  identify arm-specific response surfaces and optimize held-out policy value
  requirements: fixed treatments, overlap, assignment records, no leakage
                 |
                 v
this harness's candidate contribution
  controlled prompt-only treatment effects in a frozen tool agent
  + external terminal verifier
  + replicated task-arm outcomes with task-clustered analysis
  + structural holdout
  + empty, locked, and selection-valid constant-policy comparisons
```

| Term | Meaning here | Important boundary |
|---|---|---|
| Prompt headroom | Project shorthand for improvement available over the empty overlay | Not a standardized research term and not established by baseline failures |
| Prompt-induced uplift | Difference in expected verified success caused by one exact overlay relative to another | Requires controlled assignment or defensible causal assumptions |
| Task-conditional treatment effect | How an overlay contrast varies with pre-treatment task information | Not the difference between two noisy single rollouts |
| Prompt-policy routing | Pre-execution selection among immutable overlays | Different from generating a new prompt or choosing a model |
| Scaffold routing | Selection among broader workflows such as retrieval, planning, memory, tools, or verification | If any of these vary, call the arm a treatment bundle, not prompt-only |
| Universal policy | One treatment applied to every task in a declared target mixture | This is the current Policy Lab's primary estimand |
| Contextual policy | A mapping from pre-action task features to treatments | This is the router estimand |
| Best fixed policy | The highest-value constant action in the declared action class, including the empty overlay | A validation-selected action is not necessarily the true best fixed policy |
| Locked fixed comparator | A constant action selected on development data and frozen before sealed evaluation | Beating it is narrower than selection-valid superiority to every constant action |
| Realized oracle | Chooses an arm after observing noisy outcomes | Optimistic diagnostic ceiling; not deployable and not causal |
| Signal-restricted oracle | Best policy possible from a declared pre-action signal | Upper bound on realizable routing for that signal class |

## 5. Causal Objects

Let `i` index a task instance sampled under a declared task-generation scheme,
`a` an immutable overlay, and `r` a fresh draw from a preregistered rollout-
generation distribution, including sampling-seed generation and the handling of
admissible runtime randomness:

```text
Y_i,a,r in {0, 1}
  = externally verified success for task i under overlay a on rollout r

p_i(a)
  = E_r[Y_i,a,r | task i]

delta_i(a, b)
  = p_i(a) - p_i(b)

tau_a,b(x)
  = E[delta_i(a, b) | X_i = x]
```

`delta_i(a, b)` is a task-specific expected contrast. It is not an observed
effect from one pass/fail pair. A rollout attempt is the smallest treatment-
execution unit. Repeated fresh attempts can estimate `p_i(a)`, but they do not
create additional task instances or reveal a unique effect for one singular
stochastic trajectory. Task instances are clustered within any shared template,
archetype, family, and execution-session structure; those higher-level units
govern structural-transfer claims.

For a pre-action policy `pi(X_i)`, utility `U_i(a)`, and action class `A` that
includes the empty overlay and every eligible non-empty overlay:

```text
V(pi) = E_i[U_i(pi(X_i))]

V_fixed* = max_{a in A} V(pi_a), where pi_a(x) = a for every x
```

For a development-selected fixed action `a_lock` frozen before sealed
evaluation, the operational deployment comparison is:

```text
V(router) - V(pi_a_lock) > 0
```

The terminal scientific routing-value claim is `V(router) - V_fixed* > 0`. It
requires simultaneous or otherwise selection-valid comparison with every
constant action; superiority to `pi_a_lock` alone does not establish value from
conditional routing. Neither comparison is:

```text
realized per-task oracle - fixed policy > 0
```

### 5.1 Success and cost

The objective must be frozen before outcomes. Two coherent choices are:

1. Success-first: require success qualification, then compare cost among
   success-equivalent policies. This matches the current Policy Lab design.
2. Scalar utility: `U_i(a) = value * p_i(a) - lambda * E[cost_i(a)]`, with
   `lambda` frozen before collection.

Even when offered limits are identical, overlays can consume different tokens,
tool calls, and wall time. Cost therefore remains an outcome. Router feature
extraction and decision latency also count as deployment cost.

For the pilot in Section 10, verified success is the sole decision endpoint;
cost and failure mechanisms are descriptive safety outcomes. Before any routing
evaluation, freeze either a success-first rule with explicit qualification
margins or one scalar utility. Do not select the more favorable objective after
observing outcomes.

### 5.2 Why ordinary prediction is insufficient

An ordinary outcome model estimates:

```text
q(x, a) = P(Y = 1 | X = x, A = a)
```

The causal response surface is:

```text
mu_a(x) = P(Y(a) = 1 | X = x)
```

They coincide only with consistent treatments, randomized assignment or
conditional exchangeability, overlap, and no relevant interference. A neural
network, Bayesian posterior, or low prediction loss does not create these
conditions. Task difficulty can dominate prediction while treatment interactions
remain unlearned.

Complete crossing observes every task-arm cell and supports paired finite-bank
comparisons. It does not by itself establish a task-sampling model, justify
generalization beyond the bank, or remove chronology and carryover concerns.

## 6. Evidence Matrix

### 6.1 Prompt optimization and routing

| ID | Source and status | Locatable evidence | Supported claim | Transfer boundary |
|---|---|---|---|---|
| E-P01 | [APE: Large Language Models Are Human-Level Prompt Engineers](https://iclr.cc/virtual/2023/poster/10850), ICLR 2023, A1 | Abstract: treats an instruction as a program and selects from an LLM-proposed pool by downstream score | Automatic instruction generation and selection are established | Optimizes an instruction for static NLP tasks; not pre-action routing of multi-turn tool policies |
| E-P02 | [OPRO: Large Language Models as Optimizers](https://arxiv.org/abs/2309.03409), ICLR 2024, A1 | Abstract: iteratively proposes solutions and evaluates instructions that maximize task accuracy | LLM-driven global prompt search is established | Produces a high-scoring instruction for a task distribution, not a task-conditional router |
| E-P03 | [Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs (MIPRO)](https://aclanthology.org/2024.emnlp-main.525/), EMNLP 2024, A1 | Abstract: jointly optimizes free-form instructions and demonstrations across multi-stage programs against a downstream metric | Prompt optimization extends to multi-stage LM programs | Returns a program configuration; does not identify per-task causal policy effects |
| E-P04 | [Auto-Instruct](https://arxiv.org/abs/2310.13127), EMNLP 2023 Findings, A1 | Figure 1 and Sections 4.1-4.2: generate candidate instructions, then rank and select one for each test example; evaluated on 118 out-of-domain NLP tasks | Per-example instruction selection predates this harness | Static NLP outputs, no tool trajectory, external environment, or randomized treatment-effect design |
| E-P05 | [Automatic Prompt Selection](https://link.springer.com/chapter/10.1007/978-981-96-8180-8_8), PAKDD 2025, A1 | Abstract: cluster training data, generate candidates, synthesize input-prompt-output tuples, train an evaluator, and select from a finite set for each new input | Direct precedent for input-conditioned fixed-prompt selection | Five QA datasets; offline task outputs; no multi-turn tool verifier or structural task-family claim |
| E-P06 | [Mixture-of-Prompts](https://arxiv.org/abs/2407.00256), ICML 2024, A1 | Abstract and Sections 4.1-4.3: partition the input space, assign expert prompts and demonstrations, and route inputs by semantic similarity | One global prompt need not be optimal; expert-prompt routing is established | Experts are constructed from demonstrations and static benchmark scores, not randomized tool-agent outcomes |
| E-P07 | [Adaptive-RAG](https://aclanthology.org/2024.naacl-long.389/), NAACL 2024, A1 | Abstract and Section 3.2: a classifier selects no retrieval, single-step retrieval, or iterative retrieval from predicted query complexity | Pre-execution routing among fixed reasoning/retrieval strategies is established | Routes retrieval scaffolds, not prompt-only arms; labels combine outcomes and dataset priors |
| E-P08 | [RouteLLM: Learning to Route LLMs from Preference Data](https://iclr.cc/virtual/2025/poster/30737), ICLR 2025, A1 | Abstract: selects between a stronger and weaker model to balance response quality and cost | Pre-query routing and cost-quality evaluation are established | The treatment changes the model, so it is adjacent rather than prompt-policy evidence |
| E-P09 | [Steering Frozen LLMs: Adaptive Social Alignment via Online Prompt Routing](https://arxiv.org/abs/2603.15647), 2026 preprint, A2 | Abstract: CCLUB performs contextual-bandit routing among system prompts for a frozen LLM, with safety and utility reward models | Literal online system-prompt routing already exists | Proxy reward and same-distribution text queries; no deterministic tool-task terminal verifier or structural holdout |
| E-P10 | [Learning to Prompt](https://arxiv.org/abs/2606.20138), 2026 preprint, A2 | Abstract and Sections 3-5: adaptive selection among pedagogical prompt strategies; simulation plus student A/B test | Multi-turn adaptive prompt selection has deployment evidence | Primary quality reward is model-evaluated pedagogy; static comparisons do not establish selection-valid superiority over every constant strategy in the full prompt pool |
| E-P11 | [A Control System, a Dataset, and a Recipe for Making Frozen LLM Agents Learn a Domain](https://arxiv.org/abs/2607.25415), 2026 preprint, A2 | Section 3.1 defines 729 prompt/tool/memory/planning/verification/budget configurations; Tables 1-3 show DSPy-static leading pooled tool-use/retrieval and all Sonnet checks, while adaptive methods lead some model-domain cells | Closest broad precedent for learned frozen-agent harness control and evidence that adaptation did not consistently beat a strong static baseline at tested budgets | Changes full harness bundles, not prompt-only arms; no documented whole-archetype/family holdout; single-author unreviewed preprint |
| E-P12 | [Opportunity Is Not Realizability](https://arxiv.org/abs/2608.08265), 2026 preprint, A2 | Abstract and Sections 3-4 distinguish outcome-oracle, signal-restricted, and learned-router gains; Section 7 reports routers recover only a minority of oracle opportunity | Oracle gaps can coexist with weak realizable routing; selecting a fixed comparator on the same test data invalidates naive paired inference | Routes models rather than prompt policies; very recent unreviewed result |

### 6.2 Causal effects and policy learning

| ID | Source and status | Locatable evidence | Supported claim | Harness implication |
|---|---|---|---|---|
| E-C01 | [Neyman, randomized field experiments](https://projecteuclid.org/journals/statistical-science/volume-5/issue-4/On-the-Application-of-Probability-Theory-to-Agricultural-Experiments/10.1214/ss/1177012031.full), 1923/1990 translation, A1 | Abstract: only one potential yield is observed on a plot, while randomization identifies average contrasts | Potential outcomes and randomized finite-population contrasts are foundational | Treat rollout attempt slots as treatment-execution units, tasks as paired blocks/clusters, and templates/archetypes/families as structural generalization units |
| E-C02 | [Meta-learners for Heterogeneous Treatment Effects](https://arxiv.org/abs/1706.03461), PNAS 2019, A1 | Abstract: S/T/X meta-algorithms estimate CATE; none is uniformly best in simulations | Outcome learners require explicit treatment-effect constructions and no learner dominates universally | Compare simple treatment interactions and meta-learners only after enough independent tasks exist |
| E-C03 | [Causal Forests](https://arxiv.org/abs/1510.04342), JASA 2018, A1 | Abstract: pointwise CATE consistency and asymptotic inference under unconfoundedness and regularity | Flexible HTE estimation is possible with adequate support and sample size | Do not feed replicas as independent forest rows; small task counts make this a later-stage method |
| E-C04 | [Policy Learning with Observational Data](https://arxiv.org/abs/1702.02896), Econometrica 2021, A1 | Abstract: optimize constrained assignment policies from identified causal scores with regret guarantees | The terminal object is policy value/regret within a declared policy class | Include all constant-overlay policies in the comparator class and evaluate the router against them |
| E-C05 | [Doubly Robust Policy Evaluation and Learning](https://arxiv.org/abs/1103.4601), ICML 2011, A1 | Abstract: combine reward and logging-policy models for contextual-bandit policy evaluation | Selective logs require assignment support and policy-aware evaluation | Use direct complete-panel evaluation when available; use DR methods only for genuinely missing arms with logged propensities |
| E-C06 | [Counterfactual Risk Minimization](https://proceedings.mlr.press/v37/swaminathan15.html), ICML 2015, A1 | Abstract: propensity scoring and variance-aware generalization for learning from logged bandit feedback | Ordinary supervised learning on selectively assigned arms is insufficient | Log exact assignment probabilities before adaptive collection |
| E-C07 | [Confidence Intervals for Policy Evaluation in Adaptive Experiments](https://arxiv.org/abs/1911.02768), PNAS 2021, A1 | Abstract: naive means can be biased and inverse-propensity estimators heavy-tailed as adaptive probabilities decay | Adaptive acquisition changes inference and requires persistent support | Start balanced; batch decisions; keep an exploration floor; do not apply iid intervals to adaptive logs |
| E-C08 | [Calibration Error for Heterogeneous Treatment Effects](https://proceedings.mlr.press/v151/xu22c.html), AISTATS 2022, A1 | Abstract: HTE methods often over- or under-estimate effects; proposes a robust calibration error | Outcome calibration does not establish treatment-effect calibration | Report arm probability calibration, effect calibration, and policy value separately |

The Wager/Athey causal-forest, policy-learning, and adaptive-experiment papers
share an author and methodological lineage. They supply complementary methods,
not three independent replications of one empirical claim.

### 6.3 Tool-agent scaffolding and counterevidence

| ID | Source and status | Locatable evidence | Supported claim | Transfer boundary |
|---|---|---|---|---|
| E-A01 | [ReAct](https://arxiv.org/abs/2210.03629), ICLR 2023, A1 | Abstract: interleaves reasoning and actions; external actions gather information and improve several interactive benchmarks | Prompted reasoning/action scaffolds can help when actions expose useful evidence | ReAct was not uniformly best on every reported task; some comparisons use extra examples or sampling |
| E-A02 | [Large Language Models Cannot Self-Correct Reasoning Yet](https://arxiv.org/abs/2310.01798), ICLR 2024, A1 | Abstract: intrinsic self-correction without external feedback often fails or degrades performance | Reflection is not a universally beneficial prompt intervention | Does not refute correction grounded in tests, environment state, or external verification |
| E-A03 | [WorkArena and BrowserGym](https://proceedings.mlr.press/v235/drouin24a.html), ICML 2024, A1 | Sections 3 and 5 use browser tasks, final-state validators, repeated seeds, and ablations where extra history can hurt while chain-of-thought helps some weaker models | Web-agent scaffold effects depend on model, observation history, and task | Configurations were tuned within the benchmark; not causal evidence for prompt routing across unseen families |
| E-A04 | [SWE-agent](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5a7c947568c1b1328ccc5230172e1e7c-Abstract-Conference.html), NeurIPS 2024, A1 | Abstract and Section 5 ablations show large effects from the agent-computer interface, editor, linting, and history choices | Tool interface and guardrails can matter as much as prompt wording | ACI changes are treatment bundles; they cannot be attributed to system prompts alone |

The practical synthesis is conditional rather than universal:

- Prompts can improve planning, state tracking, output discipline, bounded
  recovery, and use of visible external feedback.
- Prompts cannot by themselves repair missing observations, broken browser
  lifecycle, provider failures, invalid verifiers, inaccessible facts, or
  insufficient base-model capability.
- Additional reflection, memory, context, or search can hurt when feedback is
  intrinsic, noisy, or mismatched to the task.

## 7. Local Evidence

| ID | Artifact | Direct observation | What it supports | What it does not support |
|---|---|---|---|---|
| E-L01 | [`unbrowser-empty-overlay-baseline-audit.md`](unbrowser-empty-overlay-baseline-audit.md), V5 | 36 tasks, two replicas, 44/72 successful rollouts, 28/36 task agreements; `form_entry_validation` 3/12 and `distractor_recovery` 4/12 | A lifecycle-valid empty-overlay baseline for its recorded substrate and two screening-selected challenge templates | No prompt lift, task-conditional effect, router signal, or pinned cache identity was measured |
| E-L02 | Same V5 audit | All 28 failures were `missing_output`; 18 exhausted tool budget, 7 ended in validation error, and 3 omitted submission | Candidate mechanisms include completion discipline, validation recovery, and budget reservation | A prompt is not proven capable of repairing any mechanism |
| E-L03 | [`o1-experiment-result.md`](o1-experiment-result.md) | In a 455-pair observational SWE-bench dataset, TF-IDF chose native in every grouped fold; both neural models chose native on the 55-task held-out split; each matched always-native | Realized oracle gaps need not be predictable from tested features | Does not rule out richer features or establish that disagreement was only noise |
| E-L04 | [`repeated-outcome-smoke-result.md`](repeated-outcome-smoke-result.md) | Repeated historical scaffold outcomes produced small positive selector estimates, but every task-bootstrap interval included zero | Replication is a measurement prerequisite but not a routing solution | Historical arms were not randomized or configuration-identical |
| E-L05 | [`m3-unbrowser-preregistration.md`](m3-unbrowser-preregistration.md) and V5 audit | The frozen 96-attempt M3 headroom pilot was a no-go and was later found lifecycle-corrupted; its policies also weakly enforced intended behaviors | Mechanical manipulation and substrate validity must precede effect estimation | The run cannot estimate empty-overlay lift or valid prompt effects |
| E-L06 | [`policy-lab-technical-design.md`](policy-lab-technical-design.md) | The design searches for one universal candidate over a fixed target mixture and limits the terminal claim to a finite final bank | Strong governance, treatment identity, and fixed-policy optimization design already exist locally | Candidate-by-task random effects do not define a deployable new-task router |
| E-L07 | [`m3-utility-routing-smoke-plan.md`](m3-utility-routing-smoke-plan.md) and [Stage B decision](../.runs/m3-utility-routing-stage-b-20260814.commit-1a67504.final-decision.json) | Stage A's structural probe passed; valid 96-attempt Stage B returned `routing_smoke_no_go`: 41.7% routed success versus 66.7% best fixed and utility lift -0.2434 | A valid local pre-action semantic-specialist router did not beat its fixed comparator | Arms had identical prompt hashes but different semantic tool interfaces, so this is not prompt-overlay evidence |

The V5 Wilson interval over 72 rollout rows is useful descriptive screening, but
it is not the primary efficacy interval for a new-task claim. Confirmatory
uncertainty must preserve all replicas and arms within a task cluster and account
for shared template, family, and session structure. A new-family claim requires
multiple independently sampled families and family-level uncertainty; holding
out one named family is a structural stress test, not general family inference.

## 8. Novelty Assessment

### 8.1 Claims that are not defensible

- First automatic prompt optimizer.
- First per-input prompt selector.
- First prompt mixture or prompt router.
- First contextual bandit over system prompts.
- First learned controller over frozen-agent harness configurations.
- First cost-aware LLM router.
- First demonstration that agent scaffolds have task-dependent effects.

### 8.2 Candidate methodological contribution

Use wording no stronger than:

> We do not introduce prompt optimization, prompt routing, contextual-bandit
> harness control, or model routing. We contribute and evaluate a controlled
> prompt-only protocol for determining whether a frozen tool-using LLM benefits
> from pre-execution selection among immutable, human-auditable system-prompt
> overlays. The protocol holds the model, tools, interface, runtime, and offered
> resource limits fixed; permits only pre-action routing signals; uses complete
> or known-propensity outcome collection with external terminal verification;
> separates attempt-level treatment execution, task-clustered efficacy, rollout
> replication, and structural-transfer units; and distinguishes observed oracle
> opportunity, signal-realizable value, and held-out router gain against an
> empty overlay, a locked fixed comparator, and selection-valid constant-policy
> comparisons.

This is a **candidate** contribution, not a priority or "first" claim. It becomes
an empirical contribution only if the completed experiment actually satisfies
every clause and produces a positive held-out result.

### 8.3 Closest precedents

| Precedent | Strong overlap | Remaining difference |
|---|---|---|
| Auto-Instruct / APS / MoP | Per-input selection among prompts or prompt experts | Static NLP tasks using offline task outputs or validation scores, no multi-turn tool verification |
| Adaptive-RAG | Pre-execution selection among fixed scaffolds | Retrieval regime changes rather than prompt-only treatment |
| CCLUB | Online contextual-bandit system-prompt routing for a frozen LLM | Proxy utility/safety rewards, same-distribution queries, no tool-task verifier |
| Frozen-agent harness control | Learned policy over a human-legible harness action space with verifiable domains | Full treatment bundles, no consistent adaptive advantage at tested budgets, no documented structural holdout |
| Opportunity Is Not Realizability | Signal-restricted opportunity and selection-valid best-fixed/router inference | Model routing rather than prompt policies; new unreviewed preprint |

The remaining difference is a conjunction of experimental controls and
evaluation discipline. It is not a new learning paradigm.

## 9. Falsification Ladder

The project should not jump from baseline failures to a neural router.

| Stage | Question | Required evidence | Current status |
|---|---|---|---|
| F0 Substrate identity | Are model, tools, runtime, verifier, cache mode, and budgets fixed and healthy? | Immutable identities, preflights, isolation, complete records | V5 passed lifecycle/completion gates for its recorded substrate, but cache identity remained unresolved; a new generation must pin and receipt it |
| F1 Baseline challenge | Does empty-overlay performance leave nontrivial room without infrastructure failure? | Fresh empty-overlay screen with repeated tasks | Passed for screening only; form and distractor are challenge templates |
| F2 Behavioral manipulation | Do candidate overlays change a predeclared intended-behavior diagnostic? | Treatment-blind trace classifiers, manipulation thresholds, and intention-to-treat retention | Not established for new overlays; passing is not causal mediation evidence |
| F3 Prompt lift | Does at least one overlay improve verified success over empty with all other treatment fields fixed? | Crossed fresh-task prompt-only outcomes | Not measured |
| F4 Replicated interaction | Does a predeclared arm-by-structure ordering replicate on disjoint tasks? | Pilot interaction followed by independent task-held-out replication; no task-winner classification from two replicas | Not measured |
| F5 Signal realizability | Can pre-action features predict those differences on untouched tasks? | Frozen simple router, grouped validation, effect/policy-value metrics | Not measured |
| F6 Router value | Is the frozen router selection-validly superior to every constant action, including empty? | Sealed task-clustered comparison against the full constant-action class; report the locked comparator separately | Not measured |
| F7 Transfer and drift | Does value persist under archetype, family, time, model, or tool shift? | Separately declared holdouts and fallback behavior | Not measured |

Cache configuration, reset policy, and scheduling must not vary by arm. Observed
cache utilization may differ because prompt bytes differ; record it separately
rather than treating configuration drift as a prompt effect. Unless a separately
authorized cache-invariance canary passes, a prompt-effect experiment should use
one explicit, conservative cache configuration. The existing cache substrate
preflight does not authorize model use.

## 10. Smallest Useful Pre-Router Experiment

This is a research recommendation, not an execution authorization or frozen
preregistration.

### 10.1 Purpose

Test prompt-only manipulation and average lift cheaply, and screen one
predeclared arm-by-template interaction before implementing a router or running
a large Policy Lab search.

### 10.2 Candidate design

```text
12 fresh challenge tasks
  6 from the form_entry_validation template generator
  6 from the distractor_recovery template generator

3 prompt-only arms
  empty overlay
  completion/validation discipline overlay
  bounded diagnosis/recovery overlay

2 fresh rollout replicas per task-arm cell

12 x 3 x 2 = 72 attempts
```

The primary estimand is the equally template-weighted finite-bank difference in
mean verified success over these 12 tasks and the frozen rollout-generation
distribution. The only contextual pilot contrast is a predeclared arm-by-
template interaction or a fully prespecified template lookup. A superpopulation
interpretation is limited to new task instances from these exact registered
generators under the same substrate. Because V5 outcomes selected the two
templates, this pilot does not estimate prompt lift for the full harness mixture.

Template identity is an analysis stratum unless its derivation is explicitly
whitelisted, available before treatment, and legal in the target deployment.
Otherwise, template-lookup value is only an interaction/opportunity diagnostic,
not contextual policy value.

The overlay descriptions are mechanism hypotheses, not prompt text. Exact text,
behavior adapters, treatment hashes, task seeds, sampling seeds, chronology,
analysis, and thresholds must be frozen before any outcome is observed.

### Execution audit (2026-08-16)

Prompt-only generation v4 consumed its single-use authorization but failed at
readiness before the first cell. The isolated model server loaded and listened,
but the pinned llama.cpp `GET /slots` response uses `id` and `is_processing`
while the v4 parser required the legacy `id_slot` and `state` fields. The
readiness gate therefore timed out, zero inference attempts ran, no outcome
ledger was created, all owned processes were torn down, all three experiment
ports were released, and the protected `gemma.service` identity remained
unchanged. V4 is infrastructure-invalid and cannot be resumed or analyzed for
efficacy. Its claim, consumed marker, controller/server logs, and self-hashed
failure record remain audit evidence.

Generation v5 corrects only that pinned endpoint contract and related receipt
validation/diagnostics, advances generation-bound schemas, and uses a fresh
task, sampling, schedule, and simulator seed bank. No arm outcome was observed
or used to select this repair. V5 still requires a new immutable freeze,
no-model preflights, and separately authored single-use authorization.

### 10.3 Controls

- Use new tasks and seeds disjoint from V2-V5 and every prior canary.
- Keep the model, base prompt, tools, tool schemas, interface, verifier, runtime,
  sampling distribution, offered limits, workspace reset, and cache mode fixed.
- Change only the exact system-prompt overlay bytes.
- Run every arm on every task in fresh isolated state.
- Randomize and counterbalance arm order within blocked task-replica panels.
- A shared sampling seed may be used as a prospectively defined common-random-
  number block, but it neither guarantees correlated trajectories nor adds
  independent task information.
- Predeclare behavior classifiers that cannot read treatment identity, verifier
  outcome, or private task fields. Retain every assigned attempt in the
  intention-to-treat efficacy analysis regardless of observed adherence.
- Use the exact slot/order assignment mechanism for any randomization test.
  Complete crossing supports paired finite-bank contrasts but does not by itself
  identify a broader population effect.
- Use these rows only for preregistered stage decisions, including which tested
  mechanism advances. Exclude them from router fitting, calibration, and
  terminal efficacy evaluation. Any prompt or policy advanced by the gate is
  selected and requires independent re-estimation and evaluation.
- Analyze task-level contrasts; do not fit a router.

### 10.4 Screening decisions

The exact numeric thresholds need simulation before freezing. The screen should
nevertheless have five non-negotiable decision categories:

| Observation | Decision |
|---|---|
| Neither overlay changes the measured intended behavior | Do not advance the intended-mechanism interpretation; still report all intention-to-treat efficacy outcomes |
| Behavior changes but neither overlay shows material success lift over empty | Do not advance these mechanisms under the current budget; do not interpret the pilot as proof of zero effect |
| One arm is materially better in both sampled templates without a predeclared harm signal | Prioritize that fixed arm in an independent study; do not claim universal dominance or that routing is unnecessary |
| The sampled templates show complementary arm ordering, but no outcome-blind rule was prespecified | Treat it as exploratory interaction evidence requiring replication on new tasks |
| A fully prespecified lookup using a legal pre-action signal beats every fixed arm under the calibrated finite-bank gate | Permit design of a larger disjoint-task routing study; if template identity is not legal, its stratified contrast permits only interaction replication |

Before freezing the gate, simulate the joint task-arm-replica outcome structure,
including between-task heterogeneity, within-task arm dependence, replica
dependence, and plausible session/order effects. Calibrate false-go probability
and power for the complete five-category decision rule, multiplicity, and a
declared minimum relevant average lift and interaction. Choose thresholds from
those operating characteristics rather than adopting an uncalibrated percentage-
point bar, and do not weaken them after seeing outcomes.

Two replicas make this only a cheap no-go and mechanistic pilot. They cannot
support individual task-winner labels, stable task-specific ordering, routeability,
or precise task-specific success probabilities. Any observed template
interaction is pilot evidence for independent replication. Ceiling-harm and
structural-transfer tests belong in the next stage if the pilot passes; spending
scarce first-screen cells on V5's perfect ceiling templates would weaken the
initial falsification test.

## 11. Evaluation Contract After A Headroom Pass

### 11.1 Data collection

- Declare the target task-generation scheme and split whole templates,
  archetypes, or families before feature design, tuning, or outcome inspection.
- Keep every arm and replica for one task in one split.
- Use a balanced complete task-by-treatment panel while the menu is small.
- If complete panels become unaffordable, use randomized incomplete blocks with
  known nonzero assignment probabilities.
- Batch adaptive assignments before observing their outcomes and preserve a
  randomized exploration floor.
- Record offered and consumed cost, exact treatment identity, assignment
  probability, chronology, and all failure classes.

### 11.2 Direct estimand

For arm `a` relative to `b`, with frozen task weights `w_i`:

```text
Delta_a,b =
  sum_i w_i * (mean_r(Y_i,a,r) - mean_r(Y_i,b,r))
  / sum_i w_i
```

Resample or randomize at the target generalization unit. Use task clusters for a
new-task claim within represented generators. A whole-group holdout reports a
stress test on that specific template, archetype, or family; a general new-family
claim requires multiple independently sampled families and family-level
uncertainty. Never bootstrap individual rollout rows.

### 11.3 Modeling progression

1. Raw task-level arm contrasts and exact or randomization-based checks tied to
   the actual assignment mechanism.
2. A small hierarchical Bernoulli model or regularized logistic/GAM treatment-
   interaction baseline with g-computed risk differences.
3. S/T/X learners only after independent task support is adequate.
4. Causal forests or neural treatment models only after they beat simple
   baselines under grouped outer evaluation.
5. Off-policy estimators only when arms are actually missing and propensities
   are valid; do not replace direct complete-panel evaluation with OPE.

At small `n`, flexible CATE learners, neural routers, and asymptotic cluster
intervals can manufacture confidence. Prior sensitivity, raw task contrasts,
and exact or permutation analyses should accompany any hierarchical result.

### 11.4 Required baselines

| Baseline | Role |
|---|---|
| Empty overlay | Measures absolute prompt lift |
| Every individual fixed action, including empty | Prevents hiding a dominant treatment |
| Primary locked fixed comparator | Development-selected deployable constant policy frozen before final evaluation |
| Outcome-selected fixed benchmark | Optimistic in-sample estimate of fixed-policy performance; neither a router ceiling nor a primary comparator |
| Uniform random router | Tests whether model structure adds value |
| Predeclared legal-feature lookup | Strong low-complexity contextual baseline; otherwise report only a subgroup diagnostic |
| Regularized logistic or GAM router | Transparent learned baseline |
| Universal Policy Lab finalist | Prompt-only comparator only if its search used the same overlay library and frozen substrate; otherwise a separate deployment benchmark |
| Learned contextual router | Candidate policy under test |
| Signal-restricted expected oracle | Upper bound for declared features |
| Realized-outcome oracle | Optimistic, non-deployable ceiling |

The locked comparator should be selected on development/validation data and
frozen before final evaluation. Beating it supports only that operational
comparison. The terminal routing-value claim must compare the router
simultaneously against every fixed arm or use a genuinely selection-valid
procedure. Selecting the "best fixed" on final outcomes and then applying an
ordinary paired interval is invalid.

### 11.5 Required metrics

- Verified success and task-level arm contrasts.
- Selection-valid policy value relative to every constant action, with empty and
  locked-comparator results reported separately.
- Regret relative to the declared policy class and diagnostic oracles.
- Arm-level Brier score, log loss, and probability calibration.
- Treatment-effect calibration and ranking, reported separately from outcome
  prediction.
- Output tokens, tool calls, wall time, router overhead, and cost per success.
- Invalid actions, tool-validation errors, budget exhaustion, missing
  submission, and correct-to-incorrect revision rate.
- Results by archetype, difficulty, chronology, and declared shift.
- Support/OOD diagnostics and fallback frequency.

## 12. Harness Architecture Implications

The smallest coherent architecture separates evidence generation from universal
optimization and contextual routing.

```text
immutable treatment registry
        |
        v
task-bank custodian ---- pre-action feature builder
        |                         |
        v                         |
crossed assignment runner         |
        |                         |
        v                         |
external verifier + cost ledger   |
        |                         |
        v                         |
headroom / interaction gate -------+
        |
        +--> universal optimizer U: choose one fixed policy
        |
        +--> contextual learner R: choose policy from X
                         |
                         v
             separately sealed policy evaluation
                         |
                         v
              deploy or fall back to fixed policy
```

| Component | Responsibility | Existing foundation |
|---|---|---|
| Treatment registry | Exact overlay bytes, empty overlay, base-prompt hash, tool/runtime pins, and immutable treatment hash | `treatments.py`, M3 registries, and command-template/preflight receipts; the full compile receipt remains Policy Lab design work |
| Task-bank custodian | Whole-archetype roles, private oracle commitments, exposure, and one-way sealing | Policy Lab bank design and M3 manifests |
| Assignment recorder | Complete-block schedule or exact propensity, order, sampling, and cost commitments | `batch.py` and M3 ledgers support complete schedules; exact propensity logging for incomplete/adaptive assignment is not yet implemented |
| Pre-action feature builder | Whitelist-only task features available before treatment; no verifier or trajectory leakage | Declared projection from `dataset.py` plus `structural_probe.py`; raw template IDs require an explicit legality decision and exclusion for structural-transfer claims |
| Outcome ledger | Raw restricted events, safe export, verifier result, failure class, and measured cost | `events.py`, `orchestrator.py`, Policy Lab ledger design |
| Headroom gate | Manipulation, prompt lift, pilot interaction, independent replication, and stage decision | New analysis boundary; do not overload allocator evaluation |
| Universal optimizer U | Search and select one policy for the declared target mixture | Policy Lab design |
| Contextual learner R | Estimate arm value from `X` and select one arm before execution | Existing outcome-model utilities plus new causal/policy contract |
| Sealed evaluator | Compare the frozen router with every constant action on untouched tasks; report empty and locked comparators separately | `allocator_eval.py` is a single-row-per-task-arm foundation; grouped-replica evaluation remains to be implemented |
| Support monitor | Detect feature shift and fall back to a locked fixed policy | New; uncertainty alone is not an OOD guarantee |

### 12.1 Reconciliation with Policy Lab

The Policy Lab should remain **architecturally parallel but evidentially staged**.

| Track | Question | Estimand | Terminal claim |
|---|---|---|---|
| Universal track U | Which one candidate should every task receive? | Weighted finite-bank value for the frozen eligible final-bank mixture | One universal finalist beats its baseline under the charter |
| Routing track R | Which candidate should this task receive from pre-action `X`? | Held-out contextual policy value relative to constant policies | Frozen router is selection-validly superior to every constant action; empty and locked comparisons are reported separately |

Recommended sequence:

1. Run the small prompt-only headroom pilot after separate authorization.
2. If it shows no material lift, stop these mechanisms; if one arm leads both
   sampled templates, prioritize an independently evaluated fixed policy.
3. If a predeclared interaction appears, replicate it on disjoint tasks before
   fitting a router or claiming signal realizability.
4. Use bounded universal search on development banks only after headroom exists.
   If it is to supply a prompt-only comparator, restrict it to the same overlay
   library and frozen execution substrate.
5. Freeze the universal finalist, routing menu, features, router, and all
   thresholds before opening routing validation/final banks.
6. Evaluate U and R as separate claims. A U win does not prove R, and a
   candidate-by-task random effect in U is not a new-task router.

A broader Policy Lab finalist that varies strategy and execution factors is a
treatment bundle. It may be reported as a separate deployment benchmark, but it
cannot define the best fixed action for the prompt-only routing estimand.

Running the full universal search before any prompt-headroom evidence risks
large spend, optimization toward one dominant arm, and contamination of the
task structure needed to test routing.

## 13. Risks And Counterexamples

| Risk | Failure mode | Required control |
|---|---|---|
| Baseline difficulty mistaken for headroom | Hard tasks fail under every overlay | Crossed prompt-only screen against empty |
| Realized oracle mistaken for a policy | Router appears valuable only after observing outcomes | Signal-restricted and held-out policy evaluation |
| Pseudoreplication | Dozens of rollouts from a few tasks create narrow intervals | Cluster and split by task; preserve template/family structure and require multiple sampled families for family-level inference |
| Treatment drift | Prompt, base prompt, tool schema, model, budget, or cache changes under one ID | Immutable complete treatment bundle and new generation on any change |
| Weak manipulation | Overlay labels do not produce different behavior | Predeclared behavior adapters before outcome collection |
| Apparent universal dominance | One overlay leads both sampled templates | Prioritize a fixed-policy replication; do not generalize beyond the sampled generators |
| Unpredictable heterogeneity | Complementary wins exist but are not encoded in `X` | Report opportunity without realizability; do not deploy a router |
| Feature leakage | Raw template ID, oracle, page answers, post-action traces, or verifier fields reveal the arm winner | Whitelist legal pre-action features; exclude raw IDs for structural-transfer claims; run negative-control audits |
| Adaptive-selection bias | Later arm means and intervals become biased | Log propensities, keep support, and use adaptive-design estimators |
| Verifier gaming | Prompt learns to satisfy a weak check without task completion | Independent verifier audits and semantic invariants |
| Intrinsic reflection harm | The agent revises correct work without external evidence | Gate revision on observable validation; preserve prior candidate |
| Substrate confounding | Browser, provider, or cache behavior varies by arm or chronology | Preflights, fresh state, counterbalanced order, and explicit cache identity |
| Structural overclaim | Random seed holdout is presented as unseen-family transfer | Split whole archetypes/families and state the exact target population |

## 14. Decision Matrix

| Decision | Current judgment | Confidence | Condition that changes it |
|---|---|---|---|
| Is generic prompt routing novel? | No | High | None expected; multiple direct precedents exist |
| Is the exact protocol a possible contribution? | Yes, as a candidate evaluation/identification contribution | Medium | An earlier source is found matching the full control and evaluation conjunction |
| Is prompt headroom established locally? | No | High | Fresh crossed prompt-only outcomes show lift over empty |
| Should a router be implemented now? | No | High | Pilot manipulation/lift and a predeclared interaction are followed by complementary ordering replicated on disjoint tasks |
| Should full Policy Lab search run first? | No | Medium-high | A simulator or non-live argument shows search is required to create any credible prompt arms |
| Are form and distractor proven stable failure templates? | No; they are screening-selected challenge templates | High | More independent tasks establish stable generator-level failure rates |
| Can V5 train or tune the router? | No | High | It remains excluded by design and has only the empty arm |
| Are neural CATE models justified at current sample size? | No | High | A sufficiently large independent-task corpus and simple-baseline gains exist |

## 15. Claim Ledger

| Claim ID | Claim | Status | Confidence | Evidence |
|---|---|---|---|---|
| C-01 | Global prompt optimization is established prior art | Supported | High | E-P01, E-P02, E-P03 |
| C-02 | Per-input prompt/strategy routing is established prior art | Supported | High | E-P04 through E-P10 |
| C-03 | Learned frozen-agent harness control is established prior art | Supported, provisional source | Medium-high | E-P11 |
| C-04 | The exact controlled tool-agent evaluation conjunction remains differentiated | Provisional | Medium | Contrast across E-P04 through E-P12; no exact match found |
| C-05 | Ordinary success prediction does not by itself identify prompt effects | Supported | High | E-C01 through E-C08 |
| C-06 | Attempts execute treatments; task clusters govern new-task efficacy claims, and replicas do not add tasks | Supported | High | E-C01 plus E-L01, E-L04, Policy Lab design |
| C-07 | Scaffold effects are conditional and external feedback matters | Supported | High | E-A01 through E-A04 |
| C-08 | V5 establishes baseline challenge but not prompt headroom | Supported | High | E-L01, E-L02 |
| C-09 | Local historical oracle gaps have not yielded credible routing value | Supported with scope boundary | High | E-L03, E-L04, E-L05, E-L07 |
| C-10 | Universal policy search and contextual routing require separate claims | Supported synthesis | High | E-C04, E-L06, E-P11 |
| C-11 | A router should be built only after prompt lift and a pre-action interaction replicate on disjoint tasks | Applied judgment | High | C-05 through C-10 |

## 16. Framework Change Log

| Event | Type | Before | After | Evidence and rationale |
|---|---|---|---|---|
| CE-01 | Invalidate | Prompt-policy routing may be broadly novel | Broad routing novelty is false | Auto-Instruct, APS, MoP, CCLUB, and frozen-agent control directly cover adjacent or overlapping methods |
| CE-02 | Split | Prompt optimization and prompt routing are one problem | Global optimization, per-input selection, and causal policy learning are distinct layers | APE/OPRO/MIPRO versus Auto-Instruct/APS/MoP versus causal-policy sources |
| CE-03 | Refine | A paired rollout difference is a task effect | The causal object is a difference in task-specific expected success probabilities | Neyman lineage, repeated-outcome local result, stochastic agent behavior |
| CE-04 | Challenge | A realized oracle gap indicates allocator potential | Oracle opportunity may be mostly unrealizable from allowed signals | E-P12 plus O1 and repeated-outcome local no-gain results |
| CE-05 | Split | Policy Lab can stand in for a router | Policy Lab U and router R share infrastructure but have different estimands and sealed evaluations | Policy Lab design, policy-learning literature |
| CE-06 | Refine | The next step is model or router implementation | The next step is a no-model-fitting finite-bank prompt-only pilot | V5 provides difficulty only; all Must causal gaps begin with treatment manipulation and lift |

## 17. Residual Gaps And Overturn Conditions

| Gap | Why it matters | Closure criterion | Status |
|---|---|---|---|
| G-01 Prompt responsiveness | Without causal lift, there is no prompt intervention to allocate | Fresh prompt-only crossed screen shows material lift over empty | Open; highest-value empirical gap |
| G-02 Stable complementarity | A dominant overlay makes a fixed policy preferable in the target population | A predeclared interaction replicates across disjoint tasks with adequate independent-task support | Open |
| G-03 Pre-action signal value | Heterogeneity may be real but unpredictable | Frozen simple router transfers to untouched tasks and separately declared structural groups | Open |
| G-04 Verifier and benchmark transfer | A router can exploit fixture artifacts | Semantic verifier audit and structural holdouts | Open |
| G-05 Exact-prior-art freshness | The 2026 literature is moving quickly | Repeat focused prior-art search before any publication claim | Open but does not block the next experiment |
| G-06 Shift robustness | A frozen router may fail after model/tool changes | Separate model, tool, time, or family shift evaluation | Deferred until F6 passes |

The candidate methodological position should be reconsidered if any of the
following occurs:

- A primary source is found that already combines immutable prompt-only arms,
  frozen multi-turn tool agents, randomized or complete outcome collection,
  external terminal verification, structural holdouts, and selection-valid
  comparison with empty and all constant policies.
- The prompt-only pilot and an appropriately powered independent replication find
  no material lift for the tested mechanisms.
- One overlay dominates across the declared target population with adequate
  task and structural support, eliminating allocation value.
- Complementary effects exist but no legal pre-action signal predicts them.
- The held-out router is not selection-validly superior to every constant action
  after complete cost accounting; locked-comparator performance is reported
  separately.
- Useful routing depends on post-action traces, candidate responses, verifier
  data, treatment identity leakage, or changed budgets/tools.

## 18. Source Lineage And Limitations

- APE, OPRO, and MIPRO are distinct systems but belong to the global prompt-
  optimization lineage.
- Auto-Instruct, APS, and MoP all use offline outcome information to build
  per-input prompt selection mechanisms; they are independent implementations,
  not evidence for causal prompt effects in this harness.
- MoP and Adaptive-RAG share an author and a semantic-routing perspective.
- ReAct and SWE-agent share authors and should not be counted as independent
  replication of a universal scaffold claim.
- The Wager/Athey causal papers share a methodological lineage.
- CCLUB, Learning to Prompt, frozen-agent harness control, and Opportunity Is Not
  Realizability are 2026 preprints. They are important novelty evidence but
  should not carry the core causal-design argument alone.
- This review used public primary sources and canonical local artifacts. It did
  not search paid databases, private communities, or unpublished internal work.
- Absence of an exact match is not proof of priority. Any future public novelty
  statement needs a fresh systematic search and appropriately qualified wording.

## 19. Research Exit

Terminal status: `complete-fit-for-purpose`.

The literature has answered the design question sufficiently:

```text
Do not implement a contextual prompt router from V5.
Do not run full universal search merely to discover whether headroom exists.
First run a finite-bank pilot of prompt-only manipulation, average lift, and one
predeclared template interaction on fresh tasks under a separately frozen and
authorized protocol. Then replicate any complementary ordering on disjoint tasks
before fitting a router.
```

If that screen fails, the useful output is a negative result about the tested
prompt mechanisms under the pilot budget. If one overlay leads both sampled
templates, prioritize independent fixed-policy evaluation. If complementary
effects replicate on disjoint tasks and are predictable from legal pre-action
signals, proceed to a separate structurally held-out routing charter.
