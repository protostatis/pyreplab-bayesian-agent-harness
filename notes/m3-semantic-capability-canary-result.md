# M3 Semantic-Capability Canary Result

Date: 2026-08-12

Verdict: **screen pass for the semantic capability family**

## Decision

The excluded semantic table/form canary passed its frozen screening gate. The
result qualifies this controller-owned capability family for a larger fresh,
prospective panel. It does not select either arm, establish a causal treatment
effect, justify allocator training, amend the frozen M3 no-go, or unlock Phase
2.

All 32 outcome attempts remain `T_canary` / `canary_excluded` and are excluded
from meta-training, calibration, development, and final evaluation pools.

## Mechanics Qualification

Fresh mechanics v2 passed before outcome execution:

- two complete panels and four unique attempts;
- four valid specialist-task mechanics checks;
- exact manifest, registry, runtime, source, and sampling identities;
- zero infrastructure, mechanism, or structural errors; and
- hard 12-call cap compliance in all attempts.

Mechanics v2 manifest hash:
`9bd665f308253fc04377f2aaf03cd81429a552666ec4124aafc70ee08ee8cd7c`.

## Outcome Screen

The frozen outcome manifest contained eight tasks, both capability arms, and
two replicas per policy-task cell: 32 attempts in 16 paired panels. Every panel
completed, with no early stopping or replacement.

| Frozen task | Table arm | Form arm |
|---|---:|---:|
| Table filtering, easy | 2/2 | 1/2 |
| Table filtering, medium, seed 2026087102 | 2/2 | 0/2 |
| Table filtering, hard | 2/2 | 0/2 |
| Table filtering, medium, seed 2026087104 | 2/2 | 1/2 |
| Form entry, easy | 2/2 | 2/2 |
| Form entry, medium, seed 2026087106 | 0/2 | 2/2 |
| Form entry, hard | 0/2 | 2/2 |
| Form entry, medium, seed 2026087108 | 0/2 | 2/2 |
| **Total** | **10/16** | **10/16** |

The repaired gate passed every predeclared outcome check:

| Check | Required | Observed | Result |
|---|---:|---:|---|
| Complete unique attempts | 32 | 32 | Pass |
| Discordant policy-task cells | At most 2 | 2 | Pass |
| Stable table-only tasks | At least 1 | 2 | Pass |
| Stable form-only tasks | At least 1 | 3 | Pass |
| Successes per arm | 2 to 14 | 10, 10 | Pass |
| Absolute arm difference | At most 4 | 0 | Pass |
| Infrastructure errors | 0 | 0 | Pass |
| Mechanism errors | 0 | 0 | Pass |
| Structural errors | 0 | 0 | Pass |

Mean output-token cost was 1,999.8 for the table arm and 3,573.4 for the form
arm. These costs and arm totals are descriptive screening results, not
treatment-effect estimates.

## Accounting Repair

The first outcome gate report returned `invalid` because it made two
deterministic classification errors:

- it treated two bounded 64 KiB result overflows as infrastructure failures
  despite their explicit `details.infrastructure_error=false` marker; and
- it counted three Pi schema-validation or truncated-argument rejections as
  executed calls, even though raw events showed they did not reach the budget
  hook and a later blocked call proved that exactly 12 calls were admitted.

No outcome row was changed and no outcome attempt was rerun. The gate now gives
an explicit structured infrastructure marker precedence over legacy text
matching. Future traces also record `pre_execution_rejected`; compatibility
inference for old traces is limited to an absent field, a later exact abort,
and exactly the configured cap of otherwise admitted calls.

The original invalid gate remains preserved at
`.runs/m3-semantic-outcome-20260812-v1.gate.json`. The superseding derived
report is
`.runs/m3-semantic-outcome-20260812-v1.accounting-repair.gate.json`; it returns
`screen_pass` with zero infrastructure, mechanism, and structural errors.

## Implication

The repeat-stable complementary niches clear the predeclared futility screen:
the table arm has two stable table-only tasks, and the form arm has three stable
form-only tasks. This supports considering the capability family in a larger
fresh panel with prospectively frozen tasks and analysis.

It does not show that either arm is globally superior. The equal 10/16 totals,
small task panel, task-family alignment, and higher descriptive form-arm token
cost all argue against selecting a fixed winner or training an allocator from
these rows. Any next experiment must retain the repaired lifecycle and
accounting semantics and use fresh excluded coordinates.

The frozen M3 headroom no-go and Phase 2 lock remain unchanged.

## Artifact Identities

- Semantic registry hash:
  `8787f3126c5e38f2d6d87187897301716d84703b35642c1216837d650934f777`
- Outcome manifest hash:
  `e8f1a941a80a8a1c0bb0141398adb12cc5e277d18c187540d2e0e84d4beeef72`
- Mechanics v2 result SHA-256:
  `c24a6685bcd5ebf666eb0806cd165b7480ac54c6c19b420bb5e1de3fec89dccb`
- Mechanics v2 preflight SHA-256:
  `cb734f354ae8f51c1f88e3daa139c075cec1326bcb437260a8a46be65d8374f5`
- Mechanics v2 gate SHA-256:
  `22b859f2eda2d5157f671058299bbbe466e184c3c99d1b84223fa8f3f6a3601c`
- Outcome result SHA-256:
  `27eed4fcaf119e717f4157c12991ca7bfac2a4f0acbc6cd767b4afcd4b274f74`
- Outcome preflight SHA-256:
  `c6402979c673d72173e5677eeee68340b46e76c55bc0d633b2dc9b3eed2b8f85`
- Repaired outcome gate SHA-256:
  `c694d368274f2c50a9156a9b8525cb44cb0142334413a4b2a0e8099f72593591`

Local manifests, results, preflights, and gate reports remain under ignored
`.runs/` storage.
