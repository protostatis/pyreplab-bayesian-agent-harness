# M3 Observation-Enforcement Canary Plan

Date: 2026-08-11

Status: complete; outcome screen later invalidated by browser-lifecycle audit

## Decision being tested

The repeated-outcome smoke showed that replication is useful for measurement but
did not establish useful allocation lift. The next question is narrower:

> After making two treatments behaviorally distinct, do any task-specific
> advantages repeat in both directions?

This is an excluded screening/futility canary. It cannot amend the frozen M3
no-go, unlock Phase 2, establish a causal observation-modality effect, or support
an allocator-effectiveness claim.

## Treatments

The canary clones the two narrowed v2 submit/retry/expanded treatments into an
isolated v3 registry. Planning, verification, recovery, output limits, timeouts,
allowed tools, and the 12-call cap are unchanged. Only the initial observation
wrapper differs.

| Arm | Parent | Enforced interface |
|---|---|---|
| Text first | `ub-decompose-text_first-submit_directly-diagnose_retry_once-expanded@2-439044f8` | `native_bash_unbrowser_interactive_text_first_v1` |
| Structure first | `ub-decompose-structure_first-submit_directly-diagnose_retry_once-expanded@2-e7e84c83` | `native_bash_unbrowser_interactive_structure_first_v1` |

On the first successful direct `navigate`, the worker automatically executes the
assigned observation before returning:

- text first: `text` with the fixed selector `body`;
- structure first: `blockmap` without a selector.

Navigation and the assigned observation are one combined model-visible tool
call. The result contains the observation payload and a receipt binding its
action, selector, canonical payload byte count, and SHA-256. Later use of either
observation modality is allowed.

This tests the performance of the enforced wrapper, not a pure content-modality
effect. Text and blockmap can differ in payload size, latency, and downstream
model behavior.

Registry SHA-256 identity:
`8f54b33f54b469f133d84a7f5ec43b2b710783197382b639d4cc333de46eb216`.

The isolated split manifest exists only to make the generic screen runner fail
closed on policy eligibility. Its `meta_train` label does not admit these rows to
M3 training. Split manifest hash:
`dc35867891fce9e423eb67698afbd3c15f0b0bd5aa5c2bee1ffe0b0f2806ab7b`.

## Stage 1: mechanics dry run

The sacrificial dry run has two fresh tasks, both treatments, and one replica:
four attempts total. Its manifest hash is
`bc8171de043bc6068ab2c03027a427f3c1e7e5fd0d8f87d374d32296a60edddf`.

It passes only if all four attempts have:

- complete runtime, registry, executed-policy, sampling, and verifier identity;
- `pi_return_code == 0` and a finite nonnegative output-token count;
- a valid auto-observation receipt with the assigned action and matching payload
  hash and byte count;
- expected first-observation adherence; and
- hard-cap compliance.

Any failure makes the dry run invalid. Preserve it, fix the mechanism, and use
new task and sampling coordinates. Its outcomes have no evidentiary role.

## Stage 2: outcome screen

Run Stage 2 only after Stage 1 passes. The frozen screen has six fresh targeted
tasks, two treatments, and two replicas: 24 attempts total. Its manifest hash is
`899139a755bcfff430c6dbc15a64adc4d1d4bf98743fe1b271d7ad8ebdd150fb`.

The task families are table filtering, search/filter controls, and form entry.
They were chosen because prior permanently excluded screens suggested possible
complementarity. This is targeted confirmation on a selected panel, not a
population sample or a task-family effect estimate.

Run every frozen panel regardless of intermediate outcomes. Do not replace tasks
or rerun outcome failures. Infrastructure invalidity makes the screen invalid.

After all mechanics checks pass, the screen passes only if:

1. At most one of 12 policy-task cells is discordant across its two replicas.
2. At least one task is stably text-only: text succeeds twice and structure fails
   twice.
3. At least one task is stably structure-only: structure succeeds twice and text
   fails twice.

Aggregate arm success and cost are descriptive only. There is no aggregate
balance gate.

Decision meanings:

- `screen_pass`: reproducible bidirectional niches exist on this frozen panel;
  qualify a larger prospective repeated panel.
- `futility_no_go`: mechanics pass, but reproducibility or bidirectional niches
  fail; stop rather than merely adding replicas.
- `invalid`: mechanism, identity, completeness, or accounting failed; no outcome
  interpretation is allowed.

## Artifacts

- Registry: `policies/m3-unbrowser-observation-canary-v1.json`
- Isolated runner split: `policies/m3-unbrowser-observation-canary-split-v1.json`
- Mechanics spec: `policies/m3-unbrowser-observation-mechanics-v1.spec.json`
- Outcome spec: `policies/m3-unbrowser-observation-outcome-v1.spec.json`
- Mechanics manifest: `.runs/m3-observation-mechanics-20260811-v1.manifest.json`
- Outcome manifest: `.runs/m3-observation-outcome-20260811-v1.manifest.json`

All task rows use `T_canary` and must export only as `canary_excluded`. Dataset
export materializes that split from the task role; raw screen-result rows do not
carry `canary_excluded` directly. The `.runs` manifests and future results are
ignored local artifacts.

## Execution history

The v1 mechanics run is preserved as an invalid diagnostic run. It exposed two
harness-reporting defects before the outcome screen: the exploratory preflight
stored the global `R=2` pin instead of the mechanics manifest's `R=1`, and an
empty blocked call serialized with `operation_aborted=false` was counted as an
admitted thirteenth call. No v1 outcome was interpreted.

After fixing those fail-closed checks, fresh v2 mechanics coordinates were
frozen at manifest hash
`1674aa60cd0065d3cd37334d215f6d42f8c6f3b362cd27fd7a03bfe692105864`.
All four v2 attempts passed the mechanics gate and exported as
`canary_excluded`.

The 24-attempt outcome screen initially returned `futility_no_go`. A later
raw-event audit found that a 30-second per-action timeout had been applied to the
entire persistent browser process, producing broken pipes in 21/24 attempts and
18/19 failures. The outcome screen is therefore invalid, not evidence of
futility. See
[`m3-observation-canary-result.md`](m3-observation-canary-result.md).
