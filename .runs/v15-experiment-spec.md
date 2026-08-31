# v15 Experiment Specification — Panel-Major Arm-Screen (DRAFT FOR FREEZE)

> **STATUS (2026-08-30): UNEXECUTED — SUPERSEDED BY PRODUCT-STRATEGY CHANGE.**
> Never frozen, never launched. The operator redirected the first deliverable
> to the openrouter-agent-cli product integration (see governance log and
> `notes/harness-cli-integration-plan.md`). Spec and scheduler code are
> preserved intact should the screening program be revived.

**Status:** Drafted 2026-08-30 per advisory ruling (v14 terminal-invalid; option c
ratified). v15 is an **independent fresh screen**, NOT a completion of v14.
No pooling of v11–v14 prefixes (none was prospectively authorized).

## 1. Purpose

Deliver the arm-selection decision under the frozen decisiveness rule with a
schedule whose prefixes serve the actual estimand (task-clustered contrast on
complete panels), fixing v14's two defects:

1. **Authorization horizon:** v14 died at a 24h TTL minted for a ~40h run.
2. **Panel-completion pace:** the v14 stratified interleave left 4/24 complete
   panels at 48 cells — contrasts computable but powerless.

## 2. Frozen (unchanged from v13/v14)

- Task bank: identical 12 fixtures (seeds 2026093001–12), 2 replicas × 3 arms.
- Arms: E `baseline_execution`, C `execution_discipline`, R
  `execution_recovery_discipline` (registry `pilot-excluded-v1`).
- Instrument: G5-qualified v2 stack (mutation detector, behavior-receipt-v2).
- Estimands: C−E and R−C primary, R−E secondary; task-clustered SE at 12
  tasks; paired within panel; ITT; decisiveness = CI excludes 0 AND sign-test
  p<0.01; cell-24/cell-48 frozen checkpoint snapshots; discordance >25pp ⇒
  audit-not-stop.
- Per-cell protocol: tool-call limit 13, wall 3300s, slot-clear, v2 receipts.
- **Any interruption ⇒ generation terminal-invalid; no resume** (reaffirmed).

## 3. Schedule amendment (v15-only)

**Panel-major within difficulty superblocks:**

- 24 panels = 12 tasks × 2 replicas. Group panels into 8 superblocks of
  3 panels each: {1 easy, 1 medium, 1 hard} per superblock (24 panels = 8×3).
- Fisher–Yates shuffle **within each stratum's panel list** (new
  `SCHEDULE_SEED`, difficulty-mixed, deterministic — no Python `hash()`).
- Superblock order itself shuffled (seeded).
- Within a panel, the 3 arm-cells run in **counterbalanced randomized order**
  (Latin-square-ish rotation across panels so each arm occupies early/mid/late
  positions equally; residual randomness seeded). Guards temporal drift,
  rate limits, and fixture carryover from becoming arm effects.
- **Tradeoff (documented):** difficulty balance at arbitrary stops is
  panel-approximate (bounded ±1 panel), not cell-exact as in v14. Accepted —
  the estimand needs complete panels more than cell-level balance.
- Expected prefix yield: at 48 cells ≈ **16 complete panels** (vs v14's 4).

## 4. Authorization & operational gates (mandated)

1. TTL ≥ max(72h, 1.5 × conservative projected runtime + buffer), measured
   from expected **launch** time, not mint time. Minting script asserts this.
2. Controller re-verifies `expires_at` against trusted UTC immediately before
   each model admission; refuses closed otherwise (v14 behavior, retained).
3. All other expiring dependencies validated over the same horizon at
   preflight: endpoint lease, credentials, quota, host power policy
   (`caffeinate`), disk/log capacity for ~72 attempt dirs.
4. Pre-freeze machine-check: all 72 cells scheduled, 24 panels complete-able,
   arm and difficulty counts exact, arm-order counterbalance satisfied,
   hash-chain uniqueness, collision freedom.
5. ETA-vs-expiry monitoring: checkpoint monitor gains an expiry alert
   (projected completion vs remaining TTL at each poll).
6. Second-person review of the frozen manifest + authorization before launch.

## 5. Fallback decision rule (frozen BEFORE launch)

If v15 completes 72/72 and no primary contrast meets the decisiveness bar:
declare the screen **inconclusive**, select the arm with the highest ITT
success rate among E/C (excluding R unless R leads both contrasts), label the
selection "screen-inconclusive fallback", and proceed to training collection
with that arm plus the no-selection default documented in the v15 analysis
preregistration. Rationale: the training milestone outranks a third screening
generation; an inconclusive screen must not trigger an indefinite loop.

## 6. Prefix handling (audit-only, non-pooling)

v14 (48 cells, all strata) and v13 (24 cells, easy) remain integrity-valid,
audit-only evidence. Qualitative comparability notes may cite them; no
statistical pooling; no decision-ineligible data enters the v15 analysis.

## 7. Duration

~30 min/cell × 72 ≈ 36h expected (v14 actual: 30.6 min/cell incl. hard);
TTL 72h covers 1.5× with margin. Plan disk for ~72 attempt dirs.

— Next step on approval: implement panel-major scheduler + TTL assertions
with tests → second-person review → freeze ceremony → mirror → preflight →
probe → authorization → hardened launch.
