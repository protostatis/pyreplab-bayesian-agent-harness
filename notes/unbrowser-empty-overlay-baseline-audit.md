# Unbrowser Empty-Overlay Baseline Audit

Date: 2026-08-14

Evidence updated: 2026-08-15 UTC

Status: v2, v3, and v4 executions permanently invalid; v5 completed and
validated as the first usable empty-overlay baseline

## Decision question

The product decision is not whether one universal policy is better than another.
It is whether a pre-execution outcome model can choose an optional system-prompt
overlay that improves verified task success over both:

1. the same agent with no additional policy overlay, and
2. the best fixed overlay used for every task.

The first grounding question is therefore:

> On which task archetypes does the empty-overlay agent fail repeatedly for
> behavioral reasons, under a healthy and otherwise fixed execution substrate?

A task that is merely hard is not enough. It must later show policy headroom:
at least one prompt overlay must improve outcomes while tools, budgets, model,
sampling, runtime, and task contract remain fixed.

## Baseline definition

The empty-overlay baseline keeps the platform substrate but adds no experimental
system prompt. It is not a literally prompt-free model invocation.

Frozen inputs:

- the same provider, model, thinking mode, and sampling parameters;
- the same plain interactive Unbrowser interface and allowed tools;
- the same output-token, tool-call, command, and wall-time budgets;
- the same sandbox, safety boundary, fixture runtime, and verifier;
- the same high-level user task and result-file contract;
- no context files, skills, prompt templates, or unrelated extensions.

Treatment-specific input:

- `system_prompt=""`;
- no `--append-system-prompt` CLI argument.

The empty string must remain part of the hashed treatment payload so that the
baseline has an immutable identity even though no overlay is sent to the model.

## Existing-evidence audit

At the start of this audit there was no empty-overlay Unbrowser treatment, and
there is still no historical run that answers the baseline question.

- All 80 pre-existing Unbrowser treatments use non-empty system prompts.
- Before this implementation, `TreatmentSpec` rejected an empty
  `system_prompt` and the orchestrator always sent `--append-system-prompt`.
- Existing fixture tasks use detailed task contracts. The
  `distractor_recovery` contract additionally tells the agent to follow an
  assigned recovery policy, so that prompt is not suitable for an empty-policy
  screen without a new task-generator version.

### Corrupted headroom proxy

`.runs/m3-headroom-pilot-e7f257c4.jsonl` contains 96 attempts across four
non-empty policy bundles. It can generate hypotheses, but it cannot estimate an
empty-overlay failure rate or prompt lift.

| Template | Verified success | Broken-pipe attempts | Success among attempts without a broken pipe |
| --- | ---: | ---: | ---: |
| `single_page_extraction` | 16/16 | 2/16 | 14/14 |
| `multi_page_navigation` | 8/16 | 7/16 | 8/9 |
| `search_filter_controls` | 4/16 | 6/16 | 4/10 |
| `table_filter_sort` | 4/16 | 13/16 | 1/3 |
| `form_entry_validation` | 1/16 | 11/16 | 0/5 |
| `distractor_recovery` | 0/16 | 13/16 | 0/3 |

Overall, 52/96 attempts contained a `BrokenPipeError`, including 46/63 failed
attempts. The persistent browser had been killed after about 30 seconds by a
process-wide GNU `timeout`; see `notes/m3-browser-lifecycle-audit.md`.

Conditioning on the absence of a broken pipe does not repair the estimates.
Longer trajectories were more likely to cross the defective lifetime boundary,
so the remaining subset is selected by task and policy behavior. The subset is
shown only to identify candidate mechanisms. All 17 failures in it had
`missing_output`; traces commonly ended after several successful interactions
with a malformed or empty-details tool error instead of writing `result.json`.

The same proxy suggested panel-oracle headroom for
`search_filter_controls` (best fixed 50%, panel oracle 100%) and
`table_filter_sort` (best fixed 50%, panel oracle 75%). These are not valid
effect estimates because the run changed prompts and budgets together, had no
empty overlay, and used the defective browser lifecycle.

### Valid specialist-capability evidence

The repaired semantic replication contains 96 infrastructure-valid attempts,
but it changes the tool interface rather than only the system prompt.

| Task | Matching specialist | Non-matching specialist |
| --- | ---: | ---: |
| `form_entry_validation` | form: 24/24 | table: 4/24 |
| `table_filter_sort` | table: 23/24 | form: 11/24 |

This establishes task-treatment heterogeneity for capabilities. It does not
establish empty-overlay difficulty or system-prompt responsiveness.

## Provisional archetype hypotheses

These rankings are inputs to a fresh screen, not conclusions.

| Archetype | Provisional role | Reason |
| --- | --- | --- |
| `single_page_extraction` | Ceiling control | Historical policies solved it consistently, including long enough attempts to expose some lifecycle failures. |
| `search_filter_controls` | Primary challenge candidate | Multiple clean traces completed most interactions but failed to submit; historical panel-oracle spread was largest. |
| `table_filter_sort` | Challenge candidate, uncertain | Strong specialist capability result, but almost all old plain-interface attempts were lifecycle-corrupted. |
| `form_entry_validation` | Floor-risk candidate | Matching semantic capability solved it, while old plain-interface traces often interacted correctly but produced no output. |
| `multi_page_navigation` | Moderate/easy control candidate | Eight of nine old non-broken attempts succeeded, but there was no empty overlay. |
| `distractor_recovery` | Redesign before screening | Old task contract embeds the treatment's recovery instruction and old attempts were heavily corrupted. |
| `cross_page_comparison` | Reserved structural holdout | Frozen preregistration requires this template to remain unseen until final evaluation. |
| `stateful_workflow` | Reserved structural holdout | Frozen preregistration requires this template to remain unseen until final evaluation. |

## Fresh excluded baseline screen

### Purpose

Measure baseline challenge and failure mechanisms only. This screen must not be
used as evidence that any system prompt helps, and its task seeds must not be
reused for confirmatory allocator evaluation.

### Recommended frozen cell

- one treatment with `system_prompt=""`;
- `native_bash_unbrowser_interactive_v1`;
- allowed tools `bash,unbrowser`;
- 4,096 maximum output tokens;
- 12 total tool calls and 12 Unbrowser calls;
- 60-second command timeout and 600-second wall timeout;
- pinned provider/model/thinking/sampling values already used by the harness;
- fresh fixture seeds and repaired Unbrowser lifecycle;
- no optional observation or semantic capability enforcement.

The generous fixed budget is intentional. The first screen should find
behavioral failures, not failures manufactured by varying resource caps.

### Task bank

Use a new generator version whose user prompt contains only:

- the desired real-world outcome;
- the allowed fixture URL/tool assignment;
- the result schema and completion criterion;
- the invariant untrusted-content safety boundary.

Do not include planning, observation order, verification, retry, or recovery
strategy. In particular, replace the current `distractor_recovery` instruction
to follow an assigned policy with an outcome-only instruction to locate the
correct diagnostics page despite stale or misleading links.

Recommended excluded screen size:

- 6 known archetypes;
- 3 difficulty levels;
- 2 fresh task seeds per archetype/difficulty cell;
- 2 rollout replicas per task;
- 72 attempts total.

The structurally held `cross_page_comparison` and `stateful_workflow` templates
must not enter this screen. They first appear in `T_fin_held` under the frozen
preregistration.

Use distinct sampling seeds for the two replicas and preserve those replica
coordinates for later paired exploratory overlays on a separate task bank.

### Mechanics gate

Do not interpret outcomes unless all of the following hold:

- the fixture server and pinned runtime pass a preflight probe;
- a confined persistent-browser stress probe succeeds after more than 35 seconds;
- every attempt has a sampling receipt, normalized trace, and verifier result;
- no attempt has an infrastructure marker, broken pipe, startup failure,
  response timeout, or browser-state overflow;
- the preflight command-template receipt confirms that
  `--append-system-prompt` is absent in the same builder used for execution.

Any infrastructure failure invalidates the screen rather than counting as an
agent failure.

### Analysis

Report by archetype and difficulty:

- verified success count and a binomial uncertainty interval;
- replica agreement for each task seed;
- verifier failure code;
- terminal mechanism such as malformed tool call, exhausted budget, wrong
  navigation, incomplete state transition, wrong answer, or missing submission;
- tool calls, output tokens, and elapsed time;
- first divergence point among replicated failures.

Classify archetypes provisionally as:

- ceiling controls when baseline success is at least 80%;
- challenge candidates when baseline success is between 20% and 80% and
  behavioral failures repeat across seeds;
- floor-risk candidates when baseline success is at most 20%;
- unstable when replicas disagree too often to support a stable label.

The thresholds are screening heuristics, not confirmatory gates. Policy
responsiveness requires a subsequent paired prompt-only experiment on fresh
tasks, while retaining ceiling controls to measure harm.

## Implementation and gate status

Implemented and verified:

- exact empty `system_prompt` support with whitespace-only prompts still invalid;
- omission of `--append-system-prompt` for the empty treatment;
- opt-in `unbrowser-fixture-v3` outcome-only task prompts while v2 remains the
  default for every persisted experiment;
- one immutable empty-overlay treatment and a 36-task/72-panel manifest;
- deterministic generation and content/workspace/oracle commitments for every
  v3 task;
- atomic committed attempt preparation with attempt-local oracle snapshots;
- a command-template receipt proving the empty-overlay and confinement flags;
- a confined lifecycle probe that performed two requests in the same Unbrowser
  0.0.19 process with a 36-second wait;
- a dedicated sequential runner with an exclusive lock, immutable single-use
  claim, active marker, deterministic attempt IDs, self-hashed records,
  fail-closed infrastructure handling, and completion receipt;
- a hard rejection in the generic treatment runner so the frozen baseline
  registry cannot bypass the hash-bound authorization gate;
- provider-backed turn accounting that excludes only Pi's exact linked,
  zero-token terminal-abort event while preserving lookalike events;
- separate accounting for attempted, budget-admitted, executed, and rejected
  tool calls;
- local and deployed-remote unit suites: 1,532 tests passed, 8 skipped.

### Invalid v2 execution

The v2 mechanics gates passed and a separately authored authorization was
claimed for one execution. The runner wrote two completed records and then
failed closed on the third record with
`malformed_normalized_events=trajectory.provider_turn_count exceeds the frozen
bound`. There is no completion receipt.

The model server journal records 7, 8, and 13 proxied requests for those three
attempts. The third raw trace has 13 provider-backed assistant messages followed
by Pi's local zero-token terminal `This operation was aborted` assistant event,
which the v2 normalizer incorrectly counted as a fourteenth provider turn. The
frozen 13-request bound was not exceeded.

The incomplete v2 screen is permanently infrastructure-invalid. Its records
must not be pooled, analyzed as baseline outcomes, repaired in place, or
resumed. The authorization was single-use and has already been consumed.

Frozen v2 artifacts and evidence:

- `.runs/m3-empty-overlay-baseline-20260814-v2.registry.json`
  - registry and treatment hash:
    `ff32e43f0e25006038c4b87d715e8c34e8352dc1f70e79ddd972524f03c5db03`;
- `.runs/m3-empty-overlay-baseline-20260814-v2.manifest.json`
  - manifest hash:
    `1c3063bcbe420cf1d5487049e1c202628b2f8f65a1ec9fd92f77e6225ef24a80`;
- `.runs/m3-empty-overlay-baseline-20260814-v2.local-preflight.json`
  - local preflight hash:
    `dda43eabda99146118b04b1117a548bd4aa223127c4a6bf67c7f2e6f6851cd14`;
- `.runs/m3-empty-overlay-baseline-20260814-v2.remote-preflight.json`
  - remote preflight hash:
    `5e7556f7a198be89c8bf433b86d32933feebf8da1a003328c6a2f42eb2355de1`;
  - lifecycle receipt hash:
    `70a2862a5e4d65ba4dc63aa8fe4ee91fa86d873d63aae6740ef2ffef601afb76`;
- `.runs/m3-empty-overlay-baseline-20260814-v2.authorization-request.json`
  - non-authorizing request hash:
    `583b024809d9943ea6073a5d82691c10828639171190aff2ad49766c7244bdef`;
- `.runs/m3-empty-overlay-baseline-20260814-v2.authorization.json`
  - consumed single-use authorization hash:
    `60b016ad086c67f03fabb2c6f4362416f953e4314bdc16328d3db21dc888a96a`;
- `.runs/m3-empty-overlay-baseline-20260814-v2.results.jsonl`
  - three-record ledger hash:
    `64ab89a138f78661cbdc5ff592b8cedf923254a88e5290e79d6d7362d2a98c81`;
- `.runs/m3-empty-overlay-baseline-20260814-v2.failure-receipt.json`
  - failed-execution receipt hash:
    `b5161fb49d961fafb78aa75634820fecfe5ab919ccb6b382a0eb13950f409042`;
- all v2 artifacts bind source tree hash:
  `f1302fadd3992625f22e64f342d10a393f7f30a67ddd44773f2fbc7ffa618727`.

The v1 artifacts are superseded and must not be used.

### Invalid v3 execution

V3 uses disjoint task, sampling, and schedule seeds. Its local and remote gates
bind the corrected source tree. The deployed remote suite passed all 1,532
tests. The remote lifecycle receipt records `same_session=true`,
`confined=true`, `navigation_status=200`, and 36.055 seconds elapsed. The model
artifact, llama-server binary, Pi CLI, provider and sampling settings, and
Unbrowser binary all matched the frozen identities without executing the model.

The v3 manifest, both preflights, and authorization request record
`live_model_execution_authorized=false`. The request reserves at most 72 model
attempts, 936 provider-backed turns, 3,833,856 output tokens, 936 tool attempts,
864 budget-admitted tool attempts, and 43,200 model wall seconds. It is not an
authorization and cannot be promoted by the harness.

A fresh, separately authored, time-bounded, single-use authorization bound every
v3 identity. Authorization
`10afb17074bd23aeb12aa3666f6b24c1354b6fa5d33931ccb25f7106221b89fb`
was claimed once and is consumed.

Frozen v3 artifacts:

- `.runs/m3-empty-overlay-baseline-20260815-v3.registry.json`
  - registry and treatment hash:
    `ff32e43f0e25006038c4b87d715e8c34e8352dc1f70e79ddd972524f03c5db03`;
- `.runs/m3-empty-overlay-baseline-20260815-v3.manifest.json`
  - manifest hash:
    `33b02404dafa3338496e890cf6c1c1db0b22e0cb6cf3c30af591c7616a6c72ba`;
- `.runs/m3-empty-overlay-baseline-20260815-v3.local-preflight.json`
  - local preflight hash:
    `8a7bb86a5d77a9415d0f780294d1286968692a8e4fc98a26483718f38b337d83`;
- `.runs/m3-empty-overlay-baseline-20260815-v3.remote-preflight.json`
  - remote preflight hash:
    `2ef5ec044ce6e4f4b9cfeddfb5f9c5b71457095765350a2613c7d494535933eb`;
  - lifecycle receipt hash:
    `cf2702f5a25820a773e58b7d90708f1d6e5a52914afc3a06cfb4d957d4e260dd`;
- `.runs/m3-empty-overlay-baseline-20260815-v3.authorization-request.json`
  - non-authorizing request hash:
    `e02211b9235d7c4d466d4a79dd75822254a2cee90d66f311718651d015b3041f`;
- all v3 artifacts bind source tree hash:
  `58e65e130a91d090394dec9c2806bc68c966b9db52573c3ad19b14b798fcb853`.

The v3 runner wrote 26 completed records and failed closed on panel 27,
`unbrowser-fixture-v3-distractor_recovery-easy-2026089032/replica=0`. The
failed attempt contained 14 real provider-backed turns, 14 tool attempts, one
pre-execution schema rejection, 12 extension-admitted/executed calls, one
terminal budget block, and one synthetic terminal assistant message. Server
logs independently record 14 proxied model requests for that attempt.

Pi performs schema validation before emitting the extension's `tool_call` hook.
The schema-invalid request therefore consumed a provider turn and tool attempt
without consuming the 12-call extension budget. The model could make a
fourteenth real request before the extension blocked the next valid tool call.
The v3 assumption that provider turns and tool attempts were bounded by
`tool_limit + 1` was false, and the normalizer also misclassified the schema
rejection as budget-admitted.

The incomplete v3 screen is permanently infrastructure-invalid. Its records
must not be pooled, analyzed as baseline outcomes, repaired in place, or
resumed. Its consumed single-use authorization must never be reused.

- `.runs/m3-empty-overlay-baseline-20260815-v3.results.jsonl`
  - 27-record ledger hash:
    `c46ac2edff27e77bcb974ea5ef7a12a5b6b3d072147354bf5e8a0c8181922e2c`;
- `.runs/m3-empty-overlay-baseline-20260815-v3.failure-receipt.json`
  - failed-execution receipt hash:
    `8f75cc95275bdb03f1aaa82c763854fef523b248d48500d0e27384ba1cb73be8`;
- `.runs/m3-empty-overlay-baseline-20260815-v3.failure-evidence/`
  - immutable source, all 27 attempt directories, and server request logs.

Any recovery must use fresh disjoint tasks, sampling seeds, source hash,
preflights, result path, and authorization. It must independently cap provider
requests before the HTTP call; changing only post-hoc accounting is not valid.

### Fresh v4 authorization boundary

V4 uses disjoint task seeds starting at `2026090001`, sampling seeds starting at
`1900008001`, and schedule seed `2026081502`. It loads
`gym-budget-v3.ts`, which independently enforces:

- at most 13 provider HTTP admissions and one local blocked gate check;
- at most 13 tool-execution attempts;
- at most 12 schema-valid tool admissions and executions;
- atomic rejection of a tool batch that cannot fit the remaining attempt cap;
- a machine-readable receipt reconciling provider checks, attempted,
  pre-admission rejected, admitted, executed, suppressed, and duplicate-ID
  events.

Pinned Pi 0.84.1 conformance tests prove four boundary cases: a schema rejection
cannot admit HTTP request 14; a 14-call batch executes no siblings; accepted
parallel calls reconcile when completion order differs from start order; and
duplicate IDs increment the monotonic attempt count and fail closed. All four
ran and passed on the Mac controller where Pi executes.

The local suite passed 1,543 tests with 8 skips. The deployed Ubuntu suite
passed 1,543 tests with 4 expected skips because Pi is not installed on the
remote tool host. The remote lifecycle receipt records `same_session=true`,
`confined=true`, `navigation_status=200`, and 36.073 seconds elapsed. No
baseline model attempt was admitted by either preflight.

The v4 request reserves at most 72 model attempts, 936 provider-backed turns,
1,008 provider-gate checks (the extra check is local and cannot reach HTTP),
3,833,856 output tokens, 936 tool attempts, 864 budget-admitted tool attempts,
and 43,200 model wall seconds. It records
`live_model_execution_authorized=false` and cannot be promoted by the harness.

Frozen v4 artifacts:

- `.runs/m3-empty-overlay-baseline-20260815-v4.registry.json`
  - registry and treatment hash:
    `ff32e43f0e25006038c4b87d715e8c34e8352dc1f70e79ddd972524f03c5db03`;
- `.runs/m3-empty-overlay-baseline-20260815-v4.manifest.json`
  - manifest hash:
    `c2e7283d8669b73380eb0340aa349e3e4545caaeb680d6fb8d15b7fc08e15607`;
- `.runs/m3-empty-overlay-baseline-20260815-v4.local-preflight.json`
  - local preflight hash:
    `0c4e96e8de73fa1dfe9552740cee943c87ab422b0801956a3313280277cd9e21`;
- `.runs/m3-empty-overlay-baseline-20260815-v4.remote-preflight.json`
  - remote preflight hash:
    `ec52799b46b135564b6a93caa29955e33b7b8e7b49bc0ab3e63870055de76cab`;
  - lifecycle receipt hash:
    `5c549b889334015a381c0873ed244c23c7e258d185f6871317cf6029c11fcd80`;
- `.runs/m3-empty-overlay-baseline-20260815-v4.authorization-request.json`
  - non-authorizing request hash:
    `d739152235b588f62fde267e3a2cd811c64d32a8b1c3456cea30ae7e8423d471`;
- all v4 artifacts bind source tree hash:
  `06f078fd785265cdc1d0dbf41badfd3f2c1779668b104ff3da73cead0712cb83`.

V4 authorization
`d23b8aa414aee7d088ea4755fd742ab796e6573054f29106ff6bc2bd8db163ab`
was claimed once. The external API shell terminated the controller after 44
durable completed records, before panel 45 could finish and before a completion
receipt was written. No controller process remained. The active marker binds
panel 45 to
`unbrowser-fixture-v3-single_page_extraction-hard-2026090005/replica=0`.

The incomplete v4 screen is permanently invalid even though none of its 44
records was marked infrastructure-invalid. Its records must not be pooled,
analyzed as baseline outcomes, repaired in place, or resumed. Its consumed
single-use authorization must never be reused.

- `.runs/m3-empty-overlay-baseline-20260815-v4.results.jsonl`
  - 44-record ledger hash:
    `f26872bdc74ddc76d6ef63cff0bc050a04c644aebf001c85b056dd987e6b322a`;
- `.runs/m3-empty-overlay-baseline-20260815-v4.failure-receipt.json`
  - failed-execution receipt hash:
    `cc37f16e8f9863486521d0f9c37aacab57a87be402dba033642bb0da3c41e78c`;
- `.runs/m3-empty-overlay-baseline-20260815-v4.failure-evidence/`
  - immutable source, all 44 remote attempt directories, and the 459-request
    server log window.

Any next screen must again use fresh disjoint tasks, sampling seeds, result
path, preflights, and authorization. Its controller must be detached from the
API shell and expose durable PID, status, and log files so the shell timeout
cannot terminate a valid long-running experiment.

## V5 Recovery Screen

V5 uses a new screen, new remote run root, new result path, fresh task seeds
`2026091001` through `2026091036`, fresh sampling seeds `1900009001` through
`1900009072`, and schedule seed `2026081503`. It preserves the exact empty
overlay, treatment hash, model/runtime identity, task/verifier semantics, and
v3 budget extension from v4.

The controller now has a source-bound `launch-detached` boundary. It starts a
new process session with stdout/stderr redirected to an immutable sibling log,
records the PID, process group, exact command and command hash, and waits for a
PID-bound child claim before reporting `claim_observed`. Existing run or launch
state blocks spawn. A child that exits before claiming is durably recorded and
reported as an error. Independent review found no high- or medium-severity
issue after the startup-handshake correction.

Frozen v5 identities:

- source tree:
  `2bcefe958b4f8af44fe2f196a33bb761a40ec1a651ccb183188b54af8f6affd3`;
- treatment registry:
  `ff32e43f0e25006038c4b87d715e8c34e8352dc1f70e79ddd972524f03c5db03`;
- manifest:
  `b52304eb39e76980d6851b54cc063f9768a127f6f3b8cd1ed5747c6705c6e739`;
- local preflight:
  `9202b952b2d319f6d6b400c25335e0d1ae638353e459a3e43cc09db482cfc357`;
- remote preflight:
  `063c426230c0247eae0e79e6aa05526d3a0ee5ed87003f795001804217f73c89`;
- 36-second lifecycle receipt:
  `848c9c298d8e60642190bdac6320dd7f76f2100f64b4682f4b73d92dd63e5268`;
- non-authorizing request:
  `80a0d3b0f120d7cf15373b2b424c6e6134874268fd97bc912b44d9fb5fc60eca`.

Local verification passed 1,539 tests with 8 expected skips, plus all four
real-Pi budget conformance cases. The exact deployed remote source passed 1,547
unittest cases with 4 expected Pi-unavailable skips. Remote source identity
exactly matched local, the fresh remote run root was absent before preflight,
and the no-model preflight reported `ready_for_authorization=true`.

The user explicitly authorized exactly one v5 execution. Authorization
`a14c11477fb7346480e2ae373cd3a33cdcb267a29145656d374b6f8327f6f478`
was claimed once by controller PID/process group `45569`. Detached launch
`995bcee4a5cf911fef2be31fcf1cacecea67393f42b6e3b65697284031a9978b`
reported `claim_observed`. The controller ran all 72 panels without retries or
skips, exited normally, removed the active marker, and wrote a completion
receipt.

Validated completion identities:

- 72-record ledger SHA-256:
  `9e93dad3c4a2de56e0c62e1296d4f05476a3f6fa48e35f20144a6d4f1f2c408e`;
- completion receipt hash:
  `781ee3db82009d92907b1927a2c7c90f2dacaaec65beaf6554317233fb121e46`;
- frozen analysis hash:
  `94b93ea6df7648dd5756f4c56d04cde36d7352b933e7eb479364d0f5ad33eaa8`;
- analysis-file SHA-256:
  `427d61d8dda2e50a9bb612673513aa63fd6f779f59a493f57e4bcf1cab610f06`.

All 72 records have status `completed`; there are no infrastructure-invalid
records. The analyzer revalidated the authorization, manifest, registry,
preflights, source identity, self-hashed ledger records, exact 72-panel prefix,
ledger hash, and completion receipt before reading outcomes.

### V5 Screening Result

The exact empty-overlay baseline succeeded on 44 of 72 attempts (61.1%; Wilson
95% interval 49.6% to 71.5%). Replicas agreed on 28 of 36 tasks (77.8%); 8 tasks
were discordant. All 28 unsuccessful attempts had verifier code
`missing_output`. Their terminal mechanisms were 18 tool-budget exhaustion, 7
tool-validation error, and 3 missing submission.

Template screening:

- `single_page_extraction`: 12/12, 6/6 task agreement, no stable failure;
  ceiling;
- `multi_page_navigation`: 12/12, 6/6 task agreement, no stable failure;
  ceiling;
- `search_filter_controls`: 10/12, 4/6 task agreement, no stable failure;
  ceiling under the preregistered screen;
- `form_entry_validation`: 3/12, 5/6 task agreement, 4 stable-failure tasks;
  challenge candidate. Easy was 3/4, while medium and hard were each 0/4. All
  nine failures exhausted the tool budget;
- `distractor_recovery`: 4/12, 4/6 task agreement, 3 stable-failure tasks;
  challenge candidate. Six failures exhausted the tool budget and two ended in
  tool-validation errors;
- `table_filter_sort`: 3/12 with 3 stable-failure tasks but also 3 discordant
  tasks; classified unstable. Medium was repeatably 0/4, but the template-wide
  result is not stable enough to treat as a single challenge family without a
  narrower mechanism probe.

This is screening evidence, not prompt-lift or causal evidence. The next
prompt-overlay headroom screen should prioritize the stable
`form_entry_validation` and `distractor_recovery` failures while preserving all
substrate and budget identities. `table_filter_sort` needs a narrower
mechanism/stability probe; ceiling templates should not consume initial
headroom budget.

Completion evidence is archived in
`.runs/m3-empty-overlay-baseline-20260815-v5.completion-evidence/`:

- exact source snapshot SHA-256:
  `cf6d86918edc7262cc4de0f5c0d36ae694006f57eda85415acc2af199223f303`;
- all 72 remote attempt directories SHA-256:
  `eab02549b90b0f12af905702e8a497544993c374cd992c0bedc74b7a1f5d2e54`;
- 712 retained model-service proxy log lines SHA-256:
  `1904081b4cfe6a99b362421c2b0f385c58795a772780793dd0b288945af794af`.
