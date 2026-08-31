# KV Cache And Branch Exploration Design Spec

Date: 2026-08-14

Status: Stage 1 isolated substrate implemented and preflighted; live cache
canary manifest not built or authorized; Stage 2 not started

Implementation evidence updated: 2026-08-15 UTC

## 0. Implementation Status

Stage 1 now has a no-model, GET-only runtime capability probe, per-provider-turn
observability records, provider-turn cache receipt schema, and a pure offline
cache-off/cache-on invariance comparator. No cache-aware scheduling, cache
reuse control, live cache canary, logical checkpoint, branch execution, or
native KV persistence was added.

Implemented boundaries:

- `pi-events-normalized-v4` emits one privacy-safe `provider_turns` record for
  every non-synthetic provider turn. Missing cache usage remains explicitly
  `null` with missing-field names rather than becoming a silent zero. Invalid
  numeric usage fails closed. Model-authored content is represented only by a
  deterministic hash that excludes volatile provider IDs and signatures;
- `pyreplab-cache-runtime-receipt-v1` binds Pi, llama-server, model, launch
  configuration, help contract, endpoint status, sensitive-artifact policy,
  and separate common/cell configuration hashes;
- `pyreplab-provider-turn-cache-receipt-v1` keeps Pi `cacheRead` as a provider
  usage observation. It cannot relabel that counter as server-verified prefix
  reuse. Exact request bytes, reused-prefix tokens, prompt/generation timing,
  and slot identity must each be separately observed before
  `mechanics_valid=true`;
- `pyreplab-cache-canary-report-v1` independently gates input equivalence,
  sampling equivalence, behavioral invariance, mechanics observability, and
  performance evidence. Reordered/missing pairs, request-hash drift, sampling
  drift, output/tool/verifier divergence, cache reuse in the off cell, missing
  timing, or savings below the frozen threshold all produce `no_go`;
- `pyreplab-cache-proxy-turn-receipt-v1` is emitted by a loopback-only,
  per-attempt instrumentation proxy. The proxy persists no prompt, response, or
  authorization header; it records exact incoming bytes and logical request
  hashes, injects only the frozen cache mode and slot zero, streams the response,
  and extracts server-originated `cache_n`, `prompt_n`, `prompt_ms`,
  `predicted_n`, and `predicted_ms`;
- `pyreplab-cache-canary-substrate-v1` freezes dedicated ports, sequential
  single-slot execution, slot clearing between attempts, no native slot-save
  path, and two server commands whose only delta is `--cache-prompt` versus
  `--no-cache-prompt`. It explicitly forbids mutation of `gemma.service`;
- CLIs are available as `pyreplab-cache-mechanics`, `pyreplab-cache-proxy`, and
  `pyreplab-cache-canary-substrate`.

The corrected current-runtime probe is
`.runs/cache-mechanics-runtime-probe-20260815-v2.json`:

- embedded receipt hash:
  `687a241f63f2b9786505017169c75ddf351b3c14a5ee94e63e0eba1c99b7fba6`;
- file SHA-256:
  `7519f4ec05f474d61405b612b6c853605049444a846d5b2dedfd5f5d399ef527`;
- model state: `sleeping`;
- cache mode: `unresolved`;
- metrics and slots endpoints: `blocked_while_sleeping` rather than falsely
  classified as unsupported;
- canary eligibility: `false`.

The current server explicitly pins context size, parallelism, and sleep policy,
but relies on defaults for prompt caching, K/V types, cache RAM, context
checkpoints, checkpoint spacing, idle-slot caching, cache reuse, unified KV,
metrics, slots, and slot-save policy. The GET-only probe also correctly reports
that exact serialized request bytes and per-request server timing/reuse/slot
evidence are outside its observation boundary. The earlier v1 probe artifact is
retained as superseded evidence; it incorrectly mixed runtime and per-turn
eligibility, and must not be used for a canary decision.

The isolated non-authorizing substrate is frozen at:

- source tree:
  `16c86678b257fc0604f6ccf6367e0316be67941f39266cf3e1886dd0201b0651`;
- manifest hash:
  `ecbdfa958d6012a2ec4ca6205f7f5ce59abe65f2fdef2f39e91165f06dc4dd8c`;
- manifest file SHA-256:
  `b35516061370a8398450fce0157ec308ab7de4cf28dfe654c547ea3f3be563c9`;
- no-model preflight hash:
  `8019a19917440f65e4d7b159c799cc4cc075a56ced76ccc11c1457496c3593f9`;
- preflight file SHA-256:
  `036cb89be573f8613f9d1a263601b6e260ba335613d9c6cc275bbe5105c5b7ac`.

The preflight re-hashed the 13.6 GB model and llama-server binary, validated
every required flag against the pinned help contract, confirmed remote port
18082 and local ports 18083/18084 were unused, and recorded
`active_service_mutated=false`, `model_loaded_or_invoked=false`, and
`ready_for_live_model_execution=false`. It reports only
`substrate_ready_for_canary_manifest_construction=true`.

The provider/server accounting contract is schema-specific and fail closed:
Pi `cacheRead`, `input`, and `output` must equal llama-server `cache_n`,
`prompt_n`, and `predicted_n`, respectively. Historical v5 logs match the
prompt and prediction equalities on consecutive turns. Any future mismatch is
classified as telemetry/accounting invalidity, not as cache-caused behavioral
divergence.

Local verification passes 1,573 tests with 8 expected skips and 463 subtests.
Independent reviews found no remaining high-severity issue; all identified
medium implementation issues were corrected before the substrate was frozen.

The next gate is a separate mechanics-only canary manifest: choose excluded
tasks, freeze counterbalanced off/on pair order, exact prompts and seeds,
aggregate budget, slot-clear receipts, server/tunnel/proxy lifecycle, and
detached execution governance. That manifest and this substrate remain
non-authorizing. No server launch or model call may occur without a new exact,
expiring, single-use authorization from the user.

## 1. Background

The harness currently executes one Pi process per attempt against a pinned
llama-server model endpoint. Pi sends the complete conversation on each provider
turn. The pinned server uses one model slot and a 65,536-token context. Its
current launch arguments do not explicitly pin prompt-cache behavior, cache
types, cache RAM, context checkpoints, slot persistence, metrics, or server
sleep behavior.

The event normalizer already records `cache_read` and `cache_write` counters
when Pi provider events expose them. Those counters are not yet sufficient as a
llama-server cache receipt: they can remain zero even when the server reuses an
identical prefix, and they do not report prompt-evaluation time, cached-prefix
length, slot identity, eviction, or checkpoint identity.

KV reuse has two materially different uses:

1. transparent prefix reuse can lower prompt-prefill compute without changing
   the agent policy; and
2. branching from an intermediate state creates a new search policy, even if
   cached model state makes each continuation cheaper.

These uses must remain separate in experiment design and reporting.

## 2. Problem And Goal

The goal is to use model-prefix caching to reduce repeated prefill work and,
later, make multi-branch exploration economically viable without weakening
experimental provenance or confusing a systems optimization with behavioral
uplift.

The design must answer four questions independently:

- Does transparent cache reuse preserve the exact measured behavior?
- How much prompt-evaluation compute and latency does it save?
- Can an agent checkpoint be reconstructed without cross-branch state leakage?
- Does branching improve verified success after charging all branch generation,
  tool, replay, and selection costs?

## 3. Roles

- Experiment controller: freezes cache and branch manifests, validates receipts,
  and enforces budgets.
- Pi agent: produces the conversation and tool trajectory.
- llama-server: owns model slots, KV state, prompt-prefix reuse, and optional
  native slot checkpoints.
- Remote gym worker: owns task, attempt workspace, verifier, and normalized
  event artifacts.
- Unbrowser runtime: owns browser process and page interaction state.
- Offline analyzer: measures cache mechanics, branch headroom, selection value,
  and total cost without feeding verifier outcomes back into live execution.

## 4. Constraints

- The frozen empty-overlay baseline must not gain branch search or cache-aware
  scheduling.
- Held templates remain unseen until their preregistered final role.
- Sampling seeds, prompt bytes, model, tokenizer, chat template, tools, and
  budgets must be fixed when testing cache invariance.
- Model KV state does not include filesystem or browser state.
- Branch generation cost still scales with branch count even when prefix
  prefill is cheap.
- Cache artifacts can contain task prompts, tool observations, and untrusted
  page content and must be treated as sensitive experiment data.
- One pinned llama-server slot means branches execute sequentially unless the
  runtime design is deliberately changed and revalidated.

## 5. Options Compared

### Option A: Transparent Prefix Reuse

- Keep the existing prompts, trajectories, schedule, and outcomes unchanged.
- Explicitly pin prompt-cache settings and emit cache-mechanics receipts.
- Strength: lowest-risk leverage and compatible with causal comparisons.
- Cost: does not create additional exploration or solution diversity.
- Risk: cache-aware attempt reordering can confound chronology and treatment
  effects, so the frozen randomized schedule cannot be reordered for hits.

### Option B: Replayable Logical Checkpoints

- Freeze a conversation prefix, workspace snapshot, and browser-action replay
  log, then run multiple seeded continuations from the reconstructed state.
- Let llama-server reuse the identical serialized conversation prefix rather
  than depending on an opaque persisted KV file.
- Strength: auditable, portable, and testable independently of one server build.
- Cost: browser replay and workspace cloning add controller complexity.
- Risk: incomplete reconstruction creates false branch diversity or hidden
  cross-branch leakage.

### Option C: Native KV Slot Save And Restore

- Assign explicit slot IDs and use llama-server slot save/restore plus context
  checkpoints and a slot-save path.
- Strength: minimizes repeated prefill and is closest to a true model-state fork.
- Cost: requires provider payload controls, slot lifecycle management, and
  server-version-specific APIs.
- Risk: a valid model KV snapshot still does not restore browser or filesystem
  state; opaque cache files are difficult to audit and highly runtime-specific.

## 6. Recommended Design

Use a staged design:

1. Stage 1 implements transparent cache observability and an on/off mechanics
   canary without changing the baseline policy or schedule.
2. Stage 2 implements replayable logical checkpoints as a separate branch-search
   treatment on excluded tasks.
3. Native KV slot save/restore remains a later optimization experiment. It may
   replace logical prefix replay only after Stage 2 demonstrates branch value
   and native restore passes equivalence tests.

The empty-overlay baseline may run on a server with transparent prompt caching,
provided cache configuration is pinned for every treatment and cache state does
not alter attempt ordering or model inputs. Branching is never part of the
baseline.

## 7. Core Flow

### Stage 1: Transparent Cache Mechanics

1. Freeze an explicit server cache configuration and configuration hash.
2. Run a mechanics-only task bank twice, once with prompt caching disabled and
   once enabled, using identical prompt bytes and sampling seeds.
3. Preserve manifest order; do not group attempts to manufacture cache hits.
4. Capture a receipt for every provider turn containing prompt tokens, reused
   prefix tokens when available, prompt-evaluation duration, generation duration,
   slot/cache identity, eviction or miss reason, and server configuration hash.
5. Compare final output, tool sequence, verifier outcome, normalized usage, and
   sampling receipt between cache-off and cache-on cells.
6. Report cache hit rate and prefill savings separately from task success.

### Stage 2: Replayable Logical Branches

1. Run one trunk trajectory to a frozen, eligible checkpoint.
2. Require the checkpoint to occur after a complete provider/tool boundary and,
   initially, before an irreversible side effect.
3. Persist the exact serialized conversation prefix and its token/hash receipt.
4. Snapshot the attempt workspace into immutable branch inputs.
5. Persist the ordered Unbrowser action log and the resulting observation hashes.
6. For each branch, create a fresh attempt namespace, restore the workspace,
   replay browser actions, verify all replayed observation hashes, and only then
   request a continuation with a distinct frozen sampling seed.
7. Execute branches sequentially under one aggregate branch budget.
8. Verify every branch independently.
9. Compute an offline oracle upper bound to establish branch headroom. Do not use
   verifier results to select a branch during live execution.
10. Only after headroom exists, evaluate a deployable branch selector using
    pre-verification features such as model self-score, trajectory features, or
    the outcome model.

### Deferred Native KV Optimization

1. Pin a llama-server build that supports explicit slot identity and slot
   save/restore.
2. Bind cache files to model hash, tokenizer/chat-template hash, server config,
   context length, cache K/V types, and exact token-prefix hash.
3. Restore a saved slot and compare logits or deterministic continuation output
   against logical prefix replay.
4. Keep workspace/browser restoration unchanged; native KV only replaces model
   prefill replay.

## 8. Artifact Model

### Cache Runtime Receipt

- schema and receipt hash;
- llama-server binary and model hashes;
- tokenizer/chat-template identity;
- context size, parallel slots, cache K/V types, cache RAM, prompt-cache flag,
  context-checkpoint settings, slot-save path policy, and sleep policy;
- metrics/slots endpoint availability;
- sensitive-artifact retention policy.

### Provider-Turn Cache Receipt

- attempt, panel, provider-turn, and sampling-seed identities;
- exact serialized-prefix hash and prompt token count;
- reused-prefix token count or an explicit `unobservable` value;
- prompt-evaluation and generation durations;
- cache hit/miss/eviction reason when available;
- slot and checkpoint identity when exposed;
- cache runtime receipt hash.

### Logical Checkpoint Receipt

- trunk attempt and checkpoint IDs;
- task, manifest, policy, and source hashes;
- ordered conversation-event hash and serialized token-prefix hash;
- workspace tree hash;
- ordered browser-action log and observation hashes;
- checkpoint eligibility classification;
- admitted tool calls and consumed budget before the checkpoint;
- declaration that no verifier or oracle result informed checkpoint selection.

### Branch Result

- checkpoint receipt hash and branch ID;
- branch sampling seed and execution order;
- restored workspace and browser-replay receipts;
- cache receipt chain;
- full normalized attempt, verifier, usage, and timing result;
- aggregate trunk-plus-branch cost allocation.

## 9. Error And Human-Review States

- Cache receipt unavailable: mechanics result is invalid, not a zero-token hit.
- Cache eviction or server sleep: record a cache miss; do not retry selectively.
- Cache-on/off output divergence: stop the invariance canary and investigate
  token serialization, sampling, slot reuse, and numerical cache settings.
- Workspace hash mismatch: block the branch before model execution.
- Browser replay mismatch: block the branch before model execution.
- Checkpoint after an unreplayable side effect: mark ineligible rather than
  approximating state.
- Crash after branch admission: preserve an active marker and require the same
  at-most-once adjudication used by other live attempts.
- Native cache restore mismatch: discard the native cache and retain logical
  replay as the authoritative path.

## 10. Non-Goals

- Reordering the frozen baseline to maximize cache hits.
- Treating cache savings as task-success improvement.
- Treating an offline branch oracle as a deployable selector.
- Sharing cache state across models, tokenizers, chat templates, system prompts,
  task roles, or untrusted experiment boundaries.
- Persisting raw KV files outside controlled local experiment storage.
- Adding concurrent model slots before single-slot reproducibility is established.

## 11. Validation And Acceptance

### Stage 1

- Every provider turn has either a valid cache receipt or an explicit mechanics
  invalidation; silent zeroes are not accepted.
- Cache-on and cache-off cells have byte-identical model inputs and matching
  sampling receipts.
- Under deterministic replay, cache-on and cache-off produce identical model
  output, tool trajectory, and verifier result.
- Prompt-evaluation tokens and time are reported independently from generated
  tokens and end-to-end wall time.
- The observed savings are large enough to justify retaining the added runtime
  complexity; no success claim depends on the savings threshold.

### Stage 2

- Every branch reconstructs the exact frozen workspace and browser observation
  hashes before model admission.
- Branches cannot read or mutate sibling branch state.
- Aggregate trunk, replay, branch generation, tool, and selection costs stay
  within one frozen authorization budget.
- The branch oracle upper bound improves verified success on excluded challenge
  tasks before a selector is implemented.
- A deployable selector is evaluated separately against single-trajectory and
  fixed-branch baselines using fresh excluded tasks.

### Native KV Optimization

- Native restore and logical replay are continuation-equivalent under a
  deterministic seed.
- A cache artifact with any runtime or token-prefix mismatch is rejected.
- Native restore reduces prefill cost beyond transparent logical replay without
  changing branch outcomes.

## 12. Risks

- Current Pi usage events may not expose llama-server prefix-cache savings.
- Server defaults can change across builds unless every cache option is explicit.
- Quantized or shifted KV caches may introduce small numerical differences.
- Browser replay can be expensive or impossible for stateful third-party sites.
- Branch count can multiply generation cost faster than cache reduces prefill.
- Cache-optimized scheduling can create treatment-order confounding.
- Raw KV and conversation checkpoints can retain sensitive or untrusted content.

## 13. Open Questions

- Which pinned llama-server endpoint or metrics path can provide per-request
  reused-prefix tokens and prompt-evaluation time through the current router?
- Does Pi preserve provider-specific timing fields in raw events, or is a narrow
  local proxy/extension required?
- What checkpoint eligibility rule best predicts useful branch headroom without
  reading verifier outcomes?
- Should the first branch selector be model self-scoring or the existing outcome
  model?
- What aggregate branch budget produces useful diversity without making the
  comparison trivially favor more compute?

## 14. Handoff Recommendation

First implement a no-model cache-capability probe and Stage 1 receipt schema.
Do not add branch execution until the receipt path and cache invariance canary
pass. Then freeze a separate Stage 2 logical-checkpoint protocol and pressure-test
its state reconstruction before authorizing any branch-search model attempts.
