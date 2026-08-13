# M3 Observation-Enforcement Canary Result

Date: 2026-08-11

Verdict: **invalid after post-run browser-lifecycle audit**

## Decision

The original gate returned `futility_no_go`, but a post-run raw-event audit found
that the controller killed each persistent confined Unbrowser process after 30
seconds of total process lifetime. The configured 30-second per-action timeout
had incorrectly been applied to the whole process. Later actions then surfaced
as `BrokenPipeError` tool results rather than infrastructure failures.

This invalidates the outcome interpretation. Do not use the observed stability
matrix, arm totals, or costs for a treatment conclusion. Do not train an
allocator from these rows or unlock M3 Phase 2.

The lifecycle failure affected 21/24 attempts and 18/19 observed failures. Every
recorded broken pipe occurred at least about 29.8 seconds after successful
navigation, matching the erroneous process-lifetime boundary. The sole failure
without a broken pipe followed a result-size overflow that also killed session
state.

The result is limited to this enforced wrapper pair and frozen targeted panel.
It does not show that every possible policy intervention lacks stable niches.

## Mechanics qualification

The original four-attempt mechanics v1 run was invalidated before outcome use.
It exposed incorrect `R=1` preflight binding and ambiguous empty blocked-call
classification. The run and gate are preserved; its task outcomes are ignored.

A fresh v2 mechanics panel used new task, sampling, and run coordinates. It
passed every predeclared check:

- two complete panels and four unique attempts;
- exact registry, executed-policy, sampling, verifier, source, and runtime
  identities;
- four valid payload-bound first-observation receipts;
- expected text/blockmap delivery in 4/4 attempts;
- no model-runtime or infrastructure failure; and
- hard-cap compliance in 4/4 attempts.

All four rows exported as `canary_excluded`. That split label is materialized by
dataset export from `task.role == T_canary`; it is not a direct field in the raw
screen-result JSONL.

Mechanics v2 manifest hash:
`1674aa60cd0065d3cd37334d215f6d42f8c6f3b362cd27fd7a03bfe692105864`.

## Outcome screen: preserved but invalid

The frozen outcome manifest contained six tasks, both enforced treatments, and
two replicas per policy-task cell: 24 attempts. Every panel was run regardless
of intermediate outcomes.

The original gate classified all 24 attempts as mechanically valid because the
worker serialized browser-process failures as ordinary tool errors. That check
was incomplete. The rows remain `canary_excluded` through the role-derived
dataset split, but they are also invalid for outcome interpretation.

After repairing lifecycle-error propagation, the same immutable results were
re-evaluated without changing any outcome row. The repaired gate returns
`invalid` and detects 24 infrastructure-error tool entries across 21 attempts,
independently confirming the raw-event audit.

| Frozen task | Text first | Structure first |
|---|---:|---:|
| Table filtering, easy | 1/2 | 0/2 |
| Table filtering, hard | 0/2 | 0/2 |
| Search/filter, easy | 1/2 | 0/2 |
| Search/filter, medium | 1/2 | 0/2 |
| Form entry, easy | 1/2 | 1/2 |
| Form entry, medium | 0/2 | 0/2 |
| **Total** | **4/12 (33.3%)** | **1/12 (8.3%)** |

On the easy form task, both treatments succeeded on replica 0 and failed on
replica 1. Across the 12 paired task-replica panels there were three text-only
successes, zero structure-only successes, one both-success, and eight both-fail
panels.

Mean output-token cost was 3,192.6 for text first and 2,547.3 for structure
first. These aggregate outcomes and costs are descriptive only.

## Original frozen gate

| Check | Required | Observed | Result |
|---|---:|---:|---|
| Mechanically valid attempts | 24/24 | 24/24 | Pass |
| Discordant policy-task cells | At most 1/12 | 5/12 | Fail |
| Stable text-only tasks | At least 1 | 0 | Fail |
| Stable structure-only tasks | At least 1 | 0 | Fail |

The table above records what the original gate returned. It is retained for
auditability and is not recomputed post hoc. Because its mechanics check omitted
the browser-lifecycle failure, `futility_no_go` is superseded by `invalid` for
scientific interpretation.

## Implication

Repair and stress-test the shared browser substrate before collecting more
outcomes. The invalid screen cannot establish whether the enforced wrapper pair
has stable niches. A new capability experiment must use fresh coordinates and
must classify browser-process death or result overflow as an invalid mechanism,
not an ordinary policy failure.

No fixed-policy or allocator conclusion may be drawn from the 4/12 versus 1/12
descriptive totals below. A future allocator experiment needs repaired shared
infrastructure and a mechanically distinct intervention with an independently
credible mechanism for complementary task strengths, followed by a fresh
excluded stability gate.

The wrappers also induced different downstream trajectories, including sharply
different descriptive recovery-adherence rates. That is part of wrapper
performance and another reason not to interpret this as a pure observation-
modality effect.

The frozen M3 headroom no-go and Phase 2 lock remain unchanged.

## Artifact identities

- Canary registry hash:
  `8f54b33f54b469f133d84a7f5ec43b2b710783197382b639d4cc333de46eb216`
- Executable/protocol source-tree hash:
  `d89be52abed1f3a1fbfbf845de6a67b72f08e5f4233f1c5dc638022b9f7b55d5`
- Mechanics v2 result SHA-256:
  `2f2859619ee3e655f9590ac274c696f668e889061c64732e7c22b3b1b0e3634d`
- Mechanics v2 preflight SHA-256:
  `263d9246f7994181f1f8f8bcfac925d65b9c3e23dc381ec5ca362a2c1e666f7c`
- Mechanics v2 gate SHA-256:
  `ee89509b1923a2fa5a3de7c9814990b6a5f3c5ac013cb181539c1381ee72edfc`
- Mechanics v2 dataset SHA-256:
  `fbdfbf2c072a4e9a4ffb0855cc8086a4a6256cf2cc80e30238dd872c0635e8b0`
- Outcome manifest hash:
  `899139a755bcfff430c6dbc15a64adc4d1d4bf98743fe1b271d7ad8ebdd150fb`
- Outcome result SHA-256:
  `4b36f64fe234d454bec0aaf16e74c923858e2c76e2fda974ac1dc594cc9899a8`
- Outcome preflight SHA-256:
  `b2c758a34c50aab3c50009694ae1e986d59156ee5c5c40b8a2bc625a8fd55479`
- Outcome analysis SHA-256:
  `23bff4d44fdd7ae04b9d2bb333d47fb2b9276610be24f805fb5b158acef4c531`
- Outcome gate SHA-256:
  `f90e1d600a0b187a5692ca711c3cf4781e0e5fe5006ff30797145e0ebc41dfb6`
- Outcome dataset SHA-256:
  `241ae9ee178d8a8bc8e401cf7fbeebbf095ea943a82b05a0bedf8908145cecde`

Local manifests, results, analyses, preflights, and gates are under `.runs/` and
remain ignored. Dataset artifacts are on the disposable remote runner and are
also excluded from the Git worktree.
