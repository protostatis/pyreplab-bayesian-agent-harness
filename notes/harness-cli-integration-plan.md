# Harness → agent-cli Integration Plan (product v1)

**Status:** plan per operator directive ("first version of the product is a
core part of our agent-cli") + advisory ruling 2026-08-30. Target repo:
`~/Projects/openrouter-agent-cli`.

## What each side has

- **agent-cli (product):** terminal agent for any OpenRouter model; tools
  (bash, files, web via pyunbrowser); permission gating; session persistence;
  an ad-hoc A/B script that runs prompt variants on text task lists but
  "grades" answers with regex heuristics, not ground truth; its own
  duplicated mini agent loop inside the A/B script.
- **This repo (research):** deterministic tasks with real pass/fail checkers;
  the v2 finish-detector and behavior receipts; discipline prompt variants;
  paired-comparison statistics; 115 completed screening runs of evidence.

## v1 definition (one vertical slice, ~1–2 weeks)

1. **One execution path.** The interactive CLI, `--prompt` mode, and the
   evaluation runner all call the same headless agent loop. The A/B script's
   private loop is retired — evaluation observes the product loop, never
   replaces it.
2. **Factual run records.** Every run appends a versioned record (model,
   sampling settings, prompt-version fingerprint, timestamps, token/cost
   totals, end reason, per-tool-call outcomes) to a local append-only trace,
   separate from compactable chat history. Viewed via `openrouter-agent runs
   show`. Records state facts only; "success" is never claimed without a
   verifier.
3. **A minimal suite + verifier contract.** A suite manifest describes tasks:
   id, prompt, fresh disposable workspace setup, a trusted verifier command
   (kept outside the agent-writable directory), timeout, and an optional
   group id linking task copies. Verifier verdicts: `pass`, `task_fail`, or
   `infrastructure_error`, with evidence. Ship one small coding-task suite
   first (repo-state checks); browser nonce suites port later if warranted.
4. **Paired comparison, descriptive only.** Both prompt profiles run on the
   same fresh task state, order counterbalanced, nothing dropped. Report:
   both-pass / only-default / only-candidate / neither, pass-rate difference,
   cost, latency. No confidence-interval claims in v1 (few independent tasks
   ⇒ no valid clustering).
5. **Discipline prompts as experimental overlays.** Default prompt unchanged.
   Execution-discipline available as an explicit experimental overlay.
   Recovery-discipline ships as an experimental file only — in the 48-run
   all-strata screening prefix it succeeded 29% vs 54–57% for the others, so
   it must not present as recommended.

**Explicitly cut from v1:** model routing via the outcome model, any
telemetry upload, statistical significance claims, cryptographic receipt
chains, general benchmark catalogs.

## Architecture: port, don't depend

Adapt the needed ideas as standalone modules inside the CLI repo (run-events,
suite contract, verifiers, pairing, reporting) — with provenance comments and
their own tests. Do **not** make the CLI depend on this research package
(Python 3.11-only vs CLI's 3.10; governance/sandbox machinery no product
should carry; frozen research code must not churn for product reasons). If
stable contracts emerge on both sides after a release or two, extract a
small neutral shared package then.

## Research-program consequences (recorded in governance log)

- v15 screen: **unexecuted, superseded** (spec preserved; scheduler code
  ready if revived).
- The 115 screening runs: evidence only, substrate-specific; do not pick
  product defaults across models.
- Future data collection on the CLI loop needs a **new preregistration**
  (small model×profile factorial first), after the measurement layer proves
  stable. Opt-in real-usage data is not an initial training corpus.

## First acceptance test

One coding task runs through the real CLI engine in two fresh workspaces
(default vs execution-discipline overlay), produces complete structured run
records, and is independently passed/failed by a host-side verifier — with
the A/B script's duplicate loop deleted.

## Top risks and mitigations

1. **Evaluation drifts from product behavior** (already true: the A/B script
   owns a second agent loop) → one headless execution engine, evaluation
   observes it without hidden nudges or extra tools.
2. **Verifiers get tampered with, are flaky, or damage real work** → fresh
   disposable workspaces per attempt; verifier and expected state outside the
   writable directory; immutable suite manifests; timeouts; separate
   `task_fail` from `infrastructure_error`; report every scheduled attempt.
3. **Derived receipts presented as truth** → the CLI currently guesses tool
   failure by searching output text for words like "failed" — normalize
   structured outcomes at each tool boundary instead; version derived
   classifications; allow `unknown`; keep receipts local and quiet.
