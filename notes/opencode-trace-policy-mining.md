# OpenCode Trace Mining for Policy Shaping (exploratory, non-confirmatory)

**Status:** exploratory analysis. Documentary only — no source edits, no
training use. Produced while v14 runs (2026-08-30). Cross-substrate evidence:
different agent (opencode coding agent), different task family (software
engineering), different tool surface. Used to shape *clause wording and
priors*, never as confirmatory evidence.

## 1. Corpus

Source: `~/.local/share/opencode/opencode.db` (SQLite, read-only URI mode;
only `session`, `message`, `part` tables read — `account`/`credential` tables
untouched).

| Metric | Value |
|---|---|
| Sessions | 3,185 |
| Messages | 180,861 |
| Parts | 884,051 (241,607 tool calls) |
| Tool errors | 4,306 |
| Window | 2026-04-05 → 2026-08-30 |
| Top projects | panicradar (640), unchainedsky (411), unchained (377), unbrowser-fin-terminal (258), pyreplab (230) |

Extraction: one full pass over `part` (111 s) filtering
`json_extract(data,'$.type') IN ('tool','text','step-finish')`, plus a
targeted pass for error states. Compact caches in `.runs/opencode_probe/`
(`tool_parts.jsonl` 69 MB, `text_parts.jsonl` 21 MB, `step_finish.jsonl`
26 MB, `part_distributions.json`).

## 2. Findings mapped to the M3 policy grammar

### 2.1 Recovery factor (`recovery: fail-fast | diagnose-retry-once`)

Post-error next-action over 4,305 classifiable errors:

| Next action | Rate |
|---|---|
| route change (different tool) | **46.2%** |
| corrected retry (same tool, changed input) | **37.8%** |
| unchanged repeat (identical call) | **15.1%** ← anti-pattern |
| session end after error | 0.9% |

**By error family** (the headline table):

| Error family | repeat | corrected | route | n |
|---|---|---|---|---|
| stale-context write (lines not found) | 24.8% | 0.0% | **75.2%** | 400 |
| stale-context write (multiple matches) | **52.4%** | 0.0% | 47.6% | 42 |
| read stale-assumption (offset/not-found) | 9.8% | **53.9%** | 35.3% | 814 |
| ripgrep failure (pattern/limit) | 6.8% | 32.6% | **59.7%** | 620 |
| transient timeout (MCP/webfetch) | 8.1% | 14.5% | **76.9%** | 346 |
| tool-argument schema error | 28.9% | 44.4% | 24.4% | 45 |
| other | 18.1% | 45.0% | 35.9% | 2038 |

**Implications for the R-arm clause ("inspect the returned error before
acting; never repeat an unchanged failed request; at most one corrected
retry; then different route or stop"):**

1. The clause is consistent with expert behavior in aggregate (83.9%
   corrected-or-route vs 15.1% repeat).
2. Family nuance the data adds: experts treat **write/patch staleness
   failures as route-only** (75%+ switch; ~0% same-tool corrected retry) and
   **read/argument failures as corrected-retry** (fix offset/args). A
   candidate clause refinement: *"After a write/patch failure caused by stale
   content, re-observe the target before any retry; do not resubmit the same
   patch."* The dominant unchanged-repeat cases (52% on multiple-match,
   25% on not-found) are exactly the staleness loop the clause forbids.
3. Transient timeouts: experts route 77% of the time — supports the clause's
   "different route" over blind re-issue.

### 2.2 Planning factor (`planning: direct | brief-plan | decompose`)

- Explicit `PLAN`/`STEP n` markers in the first assistant text: **2.5%** of
  sessions (78/3,167). Naturalistic expert behavior rarely uses literal
  markers — adherence checks that demand markers measure *instruction
  compliance*, not natural style. No change to the grammar; supports keeping
  marker adherence as a manipulation check rather than an outcome predictor.

### 2.3 Verification factor (`verification: submit-directly | final-reobserve`)

- **52.4%** of completed writes (10,188/19,451) had a read/grep/glob within
  the preceding 3 calls. Verify-before-write is roughly a coin flip
  naturalistically — the `final-reobserve` level formalizes the behavior of
  roughly half of expert writes.

### 2.4 Tool-cap factor (`tool_cap: lean(6) | expanded(12)`)

- Tool calls per session: median 33, p90 166, p99 952, max 7,769. Long-horizon
  coding sessions, so not directly comparable to 20-min fixture cells; but the
  tail confirms agents consume any cap offered. Our fixture-cell usage
  (7–9 attempts under cap 12–13 in v14 so far) sits far below these budgets —
  cap effects likely bind only on hard strata.

### 2.5 Error taxonomy (top signatures)

1. `apply_patch` verification failed — expected lines not found (270+41):
   stale-context writes.
2. ripgrep execution failed / record exceeded bytes (257+184+177):
   pattern/limit tooling errors.
3. MCP/browser tool timeouts (165+45+36+34): transient infrastructure.
4. `edit` oldString not found / multiple matches / identical (87+42+37).
5. `read` offset out of range / file not found (87+50+42).

Families 1/4/5 are the "act on stale state" cluster — the exact behavior the
R-arm recovery clause targets, and the same family our browser fixtures
elicit via stale-page distractors.

## 3. Implications beyond clause wording

- **Outcome-model feature prior (exploratory):** error-family counts and
  post-error class counts per prefix are candidate prefix-safe features.
  Naturalistic base rates above give informative priors for their expected
  ranges and predictive value (e.g., unchanged-repeat count should be
  negatively associated with success — to be verified on harness data, not
  assumed).
- **Harness recovery classifier mapping:** our receipts use
  `retry_loop | corrected_once_success | no_opportunity`. The trace taxonomy
  adds a `route_change` class our fixtures currently fold into other labels;
  worth tracking as a candidate receipt extension **after** v14 (no
  instrument edits now — G8).

## 4. Method notes & caveats

- Sequence reconstruction: per-session tool parts sorted by `time_created`;
  "unchanged repeat" = next call same tool + first-80-chars of canonical
  input identical. Approximate (truncated inputs may overcount repeats).
- Preamble-length stats invalid (300-char extraction cap); marker rates valid.
- Text/patch contents were not extracted beyond 300-char caps; nothing from
  `account`, `credential`, or token/secret material was read or cached.
- Cache-first design: `.runs/opencode_probe/` holds the only derived copies;
  the 55 GB source DB is never written to.
- Cross-substrate validity: findings are priors for clause design on a
  different agent, model, and task family. They must not be cited as evidence
  of arm effects in the harness (that is v14's job).
