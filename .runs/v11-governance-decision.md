# Governance Decision Memo — M3 Prompt-Only Pilot v11

**Date:** 2026-08-22
**Run:** `m3-prompt-only-pilot-20260816-v11` (controller PID 14580, ~8/72 cells done, infra healthy)
**Status:** Decision memo for the v11 disposition and the authorization of any follow-on generation.
**Disposition (recommended):** **Continue v11 unchanged; treat as a QUALIFIED SUCCESSFUL SCREEN; authorize training-data collection only after separately-gated detector qualification.** A full 72-cell rerun is unnecessary.

---

## 1. Summary

v11 is a **screen** (instrument validation + arm-contrast estimation), not a training-data run — its frozen clause excludes `T_pilot` from all training/calibration pools and fits no model. During the run a completion-label defect was root-caused: a harness-side detection bug (`pi_extensions/gym-tools.ts:752`) mis-flags any bash command mentioning `result.json` as a submission, so the agent's verify-read (`cat /workspace/result.json`) is counted as a second write → classifier fails closed to `unknown` on most verified-success cells.

**This defect touches only the oracle-blind completion label** (a diagnostic manipulation check). The binding outcomes — `verification` (exact-match nonce, oracle-judged) and `recovery` class — are oracle/fixture-derived and are **unaffected**. The defect is fully fixable in source and recoverable post-hoc for already-collected cells (corrected detector in the analysis layer, no frozen-source change).

---

## 2. Evidence base

- **Slot-clear fix validated (engineering deliverable):** v10 died at cell 5/72 on a single-shot slot-clear transport timeout. v11 hardened it with bounded wait-for-idle polling. All 8 finished v11 cells show `status=completed` with valid slot-clear receipts; no early termination. (Caveat: 8/72 is thin — see §6.)
- **Defect is diagnostic-only, not run-affecting:** `result_submission` is consumed solely in post-hoc modules (`m3_prompt_behavior.classify_completion`, `m3_semantic_diagnostic`, `m3_adherence`) — confirmed by source grep. It is **not** in any live control path (stopping, retries, budget, rewards, controller behavior). The buggy flag therefore could not have altered cell execution or the binding endpoints.
- **Immutable trajectories permit deterministic relabel:** raw `pi-events.jsonl` (command strings, exit codes, ordering) are persisted and frozen; the corrected classifier can replay all v11 trajectories without altering raw records. A rerun would add little information.
- **Post-hoc recovery validated on current cells:** 5/5 double-write `unknown` cells reclassify to `post_submission_tool_activity` (1 real write each) under the corrected write-redirect detector; failures stay `no_submission`. See `.runs/v11-cell-features.json`.

---

## 3. Critical audit result — model-visible coupling

**Question:** did the erroneous `result_submission` event leak into observations, stopping, retries, rewards, or controller behavior?

**Finding:** The flag is NOT used in any live control flow (§2 grep). The only model-visible surface is the agent's own tool-result observation, which included `details.result_submission: true` (benign metadata about the agent's own action; the agent already knows it ran the command). This does not change task semantics or controller behavior. The "binding-endpoint independence" gate is satisfied for control flow.

**Residual note:** `verification` and `recovery` must be confirmed to not consume the same faulty event or any state derived from it. Both are computed from oracle/fixture data independent of the trajectory's `result_submission` marker (verification compares the written nonce to the oracle; recovery classifies from tool-error sequences). This holds by construction; flagged for the final independence audit (gate G4).

---

## 4. Recommended disposition

1. **Complete v11 under frozen code.** Do not patch `gym-tools.ts` mid-run (would break the frozen-source hash and post-run audit binding). Record a protocol deviation note for the completion-label defect.
2. **Version the corrected classifier** and replay all v11 trajectories (post-hoc), reporting original vs corrected labels with full provenance. Raw records stay immutable.
3. **Qualify the prospective detector fix** (require a write-redirect, not mere path mention) with replay tests PLUS quarantined canaries exercising write/read/tee/Python-file-write submission forms.
4. **Authorize training-data collection only after** qualification + instrument freeze, on a task population **disjoint** from `T_pilot`.

---

## 5. Gates (must all pass before training collection)

- **G1** All 72 v11 cells complete, or every loss is classified under prespecified handling.
- **G2** Raw artifacts hashed and frozen; deviation report approved.
- **G3** Corrected labels reconcile with controller receipts; discrepancies below a prespecified threshold.
- **G4** Binding-endpoint independence audit passes (verification/recovery do not consume the faulty event).
- **G5** Detector conformance tests + quarantined canaries pass under production-like load.
- **G6** Training manifest freezes policy, task population, inclusion rules, stop rules, and software versions before collection.
- **G7** Training tasks are disjoint from v11; canary/qualification data excluded from training.
- **G8** No training collection begins while the detector is still being changed.

---

## 6. Watch-outs (failure modes)

- **Thin validation:** 8/72 cells is not production reliability. Even 0 failures across all 72 bounds a failure rate to ~4% at 95% CI. Call slot-clear "qualified under the observed panel," not universally validated.
- **Detector robustness:** the write-redirect heuristic still misses `tee`, Python file writes, `mv`, and other write mechanisms. Prefer a structured filesystem/controller receipt, or validate the detector against every permitted submission form.
- **MNAR / selection bias:** `unknown` is behavior-dependent. Never exclude or condition endpoint analysis on completion labels; compare corrected-label disagreement by arm, outcome, attempts, and fixture.
- **Dataset shift:** instrument qualification transfers only to the same substrate, tool protocol, task family, and concurrency regime. New fixtures/workload require canaries and explicit transportability limits.
- **Pilot leakage:** v11 tasks/attempts must remain outside training. Aggregate screen results may inform arm choice without violating the manifest's "development" exclusion.
- **Overclaiming:** arm contrasts are screening estimates, not proof of superiority.

---

## 7. Recommended next-generation structure (DECISION REQUIRED)

The advisor recommends predeclaring the follow-on shape now. Recommended:

- **v12 = detector qualification.** Ship the `gym-tools.ts` fix + versioned corrected classifier; run quarantined canaries covering all submission forms; validate completion-label reconciliation on the full v11 corpus (post-hoc). No training data collected.
- **v13 = authorized training-data collection.** Under the frozen, qualified instrument, on a task population disjoint from `T_pilot`, with the v11 governance exclusions honored. This cleanly separates "is the instrument correct?" (v12) from "collect the dataset" (v13).

Alternative (if a faster path is preferred): a single two-phase v12 that does qualification first, then collection within the same generation using excluded canaries — but this mixes validation and collection and is weaker on gate isolation. **Recommend the v12/v13 split.**

---

## 8. Decision log

- 2026-08-22: Disposition proposed — qualified successful screen; continue v11; training collection gated on v12 qualification. Next-gen structure recommended as v12 (qualify) / v13 (collect). Awaiting user ratification of §7.
- **2026-08-22 — §7 RATIFIED (predeclared, per advisory recommendation):** The follow-on structure is locked as a **staged split**:
  - **v12 = detector qualification only.** Conditionally allowed to begin **after v11 closure + gates G1–G4 pass**. Scope: ship the `gym-tools.ts` write-redirect fix, version-bump the classifier, run conformance tests + quarantined canaries, validate completion-label reconciliation on the full v11 corpus (post-hoc). **Collects no training data.**
  - **v13 = training-data collection only.** **Separately authorized after v12 satisfies gates G5–G8** (detector conformance + canaries pass, training manifest frozen, task population disjoint from v11, instrument frozen). Not pre-authorized now.
  - **Execution of neither v12 nor v13 is pre-authorized.** Locking the roles and gate order now prevents post-hoc relabeling of qualification data as training data, while deferring v13's detailed design avoids committing before v11 evidence exists.
  - **Constraint:** no source edits (even "for v12") in the active workspace during the v11 run — provenance must stay clean. v12/v13 design is documentary until v11 ends.
- **2026-08-22 — v12 detector design DECIDED (advisory):** v12 implements **Option B (structured controller receipt)**, not the regex stopgap. The controller (not the bash wrapper) emits the submission signal from the actual finalized filesystem mutation of `/workspace/result.json`; the controller receipt is the single authoritative source. Rationale: qualification must exercise the durable path v13 will use; shipping A in v12 would leave the collection instrument unqualified. Detector and classifier are versioned **separately**. Acceptance is **deterministic zero-mismatch** (not statistical), with ≥12 quarantined canaries (6 v11-replay + 6 adversarial). Missing monitor events / sequence gaps / controller failures classify a cell as **infrastructure-invalid**, not "unknown". See `v12_qualification_plan.md` §4–§6.
- **2026-08-23 — v11 CLOSED: TERMINAL, INTERRUPTED, GENERATION INVALID (advisory ruling).** The run terminated at 43/72 cells (local controller death + remote model-endpoint loss, last cell finished 13:20:57Z). The frozen design forbids resume/relaunch after any server launch or interrupted cell; a single resume attempt was correctly refused by the collision gate (recorded as a deviation; no side effects). The 43-cell prefix is retained as **audit-only evidence** with exploratory (non-confirmatory) analyses. G1 status = loss accounting complete; design completeness failed; generation invalid. Closure package: `v11-closure-invalid-run.md` (hashes, UTC timeline, loss classification, deviation report incl. detector defect and resume attempt, task-level missingness/selection-bias table, training-exclusion confirmation).
- **2026-08-23 — Staged structure AMENDED at v12's entrance criterion only:** v12 (detector qualification) may begin after approval of the invalid-run closure + G2–G4 evidence; its reconciliation corpus is the immutable **43-cell ledger** + quarantined canaries (replaces the assumed full 72-cell corpus). v13 remains gated behind G5–G8 with an explicit arm-selection decision required. A "remaining-29-cells" continuation under a fresh freeze is ruled out — it is a new experiment (different frozen seeds/schedule) and cannot repair v11.
- **Screening takeaway carried forward (non-confirmatory):** baseline_execution leads the observed prefix (E 8/14 vs R 6/15 vs C 3/14); C−E = −0.375 [−0.682, −0.068] excludes zero but sign-test p=0.125 fails the locked decisiveness bar; **zero hard-stratum panels were reached** — this must not drive v13 arm selection without a fresh balanced experiment.
- **Implementation correction (advisory catch):** checkpoint monitor clustering fixed from panel-level (`pilot_panel_id`, overcounted 14 clusters) to unique-task level (`task_id`, 8 covered tasks); all status artifacts regenerated with corrected contrasts.
- **2026-08-23 — v12 QUALIFICATION EXECUTED: G5 SATISFIED.** Option B mutation-based detector implemented across executor (`worker.py` content-snapshot receipts), tool layer (`gym-tools.ts` authoritative predicate incl. deletion-is-not-a-submission), and receipt versioning (`behavior-receipt-v2`; detector `result-submission-mutation-v2` versioned separately from classifier). Evidence: deterministic conformance matrix 9 tests + 17 subtests zero-mismatch; quarantined canaries **12/12 zero-mismatch** (6 v11-replay idioms incl. the double-write trigger + 6 adversarial forms); corpus reconciliation on the immutable 43-cell ledger = 16 discrepant / 0 unexplained (G3 holds). Report: `v12-qualification-report.md`. Residual note: re-run matrix against real BubblewrapSandbox on the Linux substrate before any collection generation. G6–G8 remain open (v13 phase, deferred).
- **2026-08-23 — G5 residual CLOSED: substrate re-validation passed.** The full 18-case matrix re-run on the production Linux executor (real `BubblewrapSandbox`, bwrap + `systemd-run --user`, ubuntu-local — same substrate as all 43 v11 cells): **18/18 zero-mismatch** (`v12-substrate-report.json`). macOS FakeSandbox suite remains as fast CI tier. No open caveats remain on G5.
- **2026-08-23 — STAGED STRUCTURE AMENDED: v13 = fresh balanced arm-screen (spec drafted).** Per the advisory escape hatch ("authorize a fresh balanced experiment" once arm selection becomes consequential — it now has), v13 is re-scoped from training-collection to **screen-only**: reuse of the identical 12-task bank (incl. all four HARD fixtures v11 never reached: form_entry hard 3005/06, distractor_recovery hard 3011/12), seeded stratified-randomized execution order (bounded-damage vs interruption), G5-qualified v2 instrument, pre-registered no-resume loss handling, tmux+caffeinate launch hardening, model-server-on-18082 as blocking pre-launch check. Training collection deferred to v14+, still gated behind G6–G8 with explicit arm-selection input from v13. Spec: `v13-experiment-spec.md` (draft for freeze; awaiting ceremony run).
- **2026-08-30 — v14 TERMINAL-INVALID AT 48/72: AUTHORIZATION-EXPIRY CLEAN BOUNDARY (advisory ruling).** Root cause: execution authorization minted with 24h TTL for a ~40h projected run (operator minting error); controller attempted cell 48's model admission 30 min past `expires_at` (2026-08-30T13:15:49Z+00:00) and failed closed. **New loss category** (distinct from v11's mid-cell infrastructure loss): clean cell-boundary stop, all 48 records complete with slot-clear receipts, zero orphan attempts, integrity verified. Prefix is **integrity-valid but decision-ineligible and audit-only**; T_pilot remains excluded from training regardless. Balanced difficulty coverage achieved (16 easy / 16 medium / 16 hard — first generation to observe the hard stratum); however the stratified interleave left only 4/24 complete panels, so the task-clustered contrasts rest on ~4 clusters and cannot meet the decisiveness bar (C−E +0.250 [−0.24,0.74] p=1.0; R−C −0.250 [−0.74,0.24] p=1.0; ITT: E 7/13, C 8/14, R 6/21). Resume with a fresh authorization ruled out under all readings (single-use chain + ratified no-resume rule; v11's refused move). Arm selection from v14+v13 ruled out (weak post-hoc pooling bypasses the preregistered decisiveness rule).
- **2026-08-30 — v15 RATIFIED (advisory, option c): fresh 72-cell PANEL-MAJOR screen; NOT a completion of v14.** Schedule amendment for the new generation only (prompts, estimands, thresholds, instrument, outcome definitions all frozen): panels scheduled **panel-major within difficulty superblocks** (one easy + one medium + one hard panel per 9 cells; arm order counterbalanced/randomized within panels) so any prefix carries ~3× the complete panels of the v14 schedule at equal cell count. Documented tradeoff: stratum balance at arbitrary stops becomes panel-approximate (bounded ±1 panel). **Fallback decision rule must be frozen BEFORE launch** for the case "v15 completes but contrasts inconclusive." Launch gates mandated: (1) authorization TTL ≥ max(72h, 1.5× conservative projected runtime + buffer), measured from expected launch, not mint time; (2) controller re-verifies `expires_at` against trusted UTC immediately before each admission and refuses otherwise; (3) all other expiring dependencies validated over the same horizon (endpoint lease, credentials, quota, host power policy, disk/log capacity); (4) machine-check of all 72 scheduled cells / 24 panels / arm+difficulty counts / counterbalanced arm order / hash-chain uniqueness before freeze; (5) ETA-vs-expiry monitoring with alerts + second-person review of frozen manifest and authorization; (6) reaffirm any-interruption-⇒-terminal-invalid for v15.
- **2026-08-30 — v15 UNEXECUTED, SUPERSEDED BY PRODUCT-STRATEGY CHANGE (operator directive + advisory).** The operator directed that the first product deliverable be a measurement/evaluation layer embedded in the openrouter-agent-cli product (see `notes/harness-cli-integration-plan.md`). Per advisory: v15 (fresh 72-run panel-major screen, spec preserved at `v15-experiment-spec.md`, DRAFT status, never frozen, never launched) is marked **unexecuted and superseded** — not failed, not invalid. No cells were run under the v15 generation. Should the operator revive the screening program, the spec and the v15 scheduler code (committed 34a6f08) remain ready for freeze. The 115 completed screening runs (v11 43 + v13 24 + v14 48) remain integrity-valid, audit-only, T_pilot-excluded evidence; they continue to support two substrate-specific findings — the v2 detector works in production, and the recovery-discipline prompt *reduced* success (29% vs 54–57% for the other two prompts on 48 all-strata runs) — but per the transferability warning (§6) they do NOT select defaults for the CLI product's different model/substrate/task space. The M3 outcome-model milestone is deferred: a **new CLI-specific preregistration** (not an amendment) will govern any future data collection on the product's execution loop, starting with a minimal model×profile comparison rather than a factorial.
