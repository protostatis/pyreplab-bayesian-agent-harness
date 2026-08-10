# Terminal Gym Proof-of-Concept Notes

> **Purpose:** Finish and learn from the local terminal gym before adapting the
> harness to a third-party browser benchmark. These results are calibration and
> plumbing evidence, not a claim that the learned allocator generalizes.

> **Current status (2026-08-09):** The v4 gate is intentionally paused after
> five completed pairs. The immediate milestone is an offline architecture
> experiment using 455 complete same-task O1 SWE-bench A/B pairs. External
> outcomes will not be pooled as Direct/Deliberate labels. See
> `notes/agent-task-success-dataset-exploration.md`.

> **Generalized harness checkpoint:** The controlled treatment-registry path is
> now wired through generation, execution, resumable batches, dataset export,
> descriptor-aware training, and strict complete-panel allocator evaluation.
> The remote CPU/PyTorch suite passes all 445 tests. A fixed-page live
> Pi/Unbrowser control smoke also produced the predeclared negative failure and
> positive success with exact tool traces. This is implementation evidence
> only: no generated-policy outcome corpus or leave-one-policy-out result
> exists yet, and no learned live allocator selector should be enabled.

> **Theta descriptor probe:** A deliberately labeled two-policy model sharply
> ranks the two seen bundles, but six unseen-ID exact-clone and paraphrase
> candidates collapse to near-identical low scores. The predeclared result is
> therefore `inconclusive`, not descriptor-generalization evidence. The
> mean-pooled encoder also ties an `h1, not p` versus `p, not h1` collision pair,
> confirming that token order is not represented by this smoke setup.

> **Held-out descriptor learn-smoke:** A 36-policy, noisy synthetic run trained
> on 26 bundles and ranked 10 identity-neutralized held-out bundles. It improved
> expected allocation lift over random by `0.028`, but its held-out rank recovery
> was only `rho=0.238` and top-1 matched `1/12`; the predeclared verdict is
> therefore a non-pass. This records partial signal, not substantial transferable
> policy learning.

## 1. v1 Calibration Result

The first calibration batch ran eight paired jobs: four families, medium and
hard difficulty, seed 21.

```text
pairs                         8
attempts                     16
infrastructure errors         0
total wall time              34.2 minutes
median pair wall time       241.8 seconds

both policies succeeded       4
Direct only succeeded         1
Deliberate only succeeded      1
both policies failed          2
paired disagreement          25%

Direct success                5/8
Deliberate success            5/8
```

The 25% disagreement rate clears the preliminary 15–20% calibration target,
but the sample is far too small for a model or allocator claim. Artifact and
Python-repair were all-pass in this slice, SQLite produced disagreement in both
directions, and shell hard/medium failed under both policies.

## 2. Throughput Diagnosis

Task generation, SSH orchestration, event recording, and verification are not
the throughput bottleneck. Instrumented pairs spend only about 2–3 seconds
outside Pi/model execution. The dominant cost is serialized autoregressive
inference across repeated provider turns.

Prefix caching is already active: `cache_read` dominates logical prompt tokens
after early turns. Moving Pi to Ubuntu or optimizing task-file generation is
therefore unlikely to matter materially.

The largest avoidable v1 cost was budget overrun behavior. The original tool
returned an error after its nominal limit, but Gemma could repeatedly request
another rejected call:

```text
Direct output tokens in calibration          35,425
Direct output after first limit rejection    15,127 (42.7%)
Direct rejected calls                            15
Direct attempts with a rejection                 6/8

Deliberate output after first rejection         821 (2.3%)
Deliberate rejected calls                           1
```

The harness now records provider turns, tool calls, length stops, limit
rejections, and per-attempt phase timing. Batch files are policy-version
specific and refuse mixed-version resume.

## 3. Policy Calibration Lessons

### v2 — rejected as too lean

Policy v2 reduced per-turn output and tool budgets aggressively. On four
matched diagnostic pairs it reduced wall time from approximately 18.1 minutes
under v1 to 9.9 minutes, a roughly 46% improvement. However, it also changed
the outcome matrix from useful disagreement to two both-pass and two both-fail
pairs. SQLite policies spent the small budget inspecting and never edited the
database. The lower per-turn cap also cut off Gemma before some generated tool
calls became executable.

Conclusion: fastest rows per hour is the wrong objective if the policy
intervention becomes degenerate.

### v3/v4 — preserve headroom, remove tail waste

The moderate policy restores v1-sized per-turn output headroom, gives Direct
seven tool calls and Deliberate eight, and explicitly discourages version,
existence, and other redundant probes. It retains a hard-stop companion
extension for calls beyond the treatment budget.

The v3 SQLite-medium diagnostic took 252 seconds versus 365 seconds in the v1
calibration and retained a policy disagreement (Deliberate passed, Direct
failed). The shell-hard diagnostic took 340 seconds versus 415 seconds; both
still failed, but Deliberate progressed from a wrong file tree to a semantically
correct tree with a manifest mismatch.

Policy v4 pins the corrected hard-stop lifecycle behavior as a new immutable
treatment version. v1, v2, v3, and v4 rows must not be pooled as if they were
the same treatment.

## 4. Shell Generator Calibration

The v1 shell contract did not state clearly that misfiled inputs must be renamed
with the category's canonical extension, and it did not explicitly prohibit
leftover helper scripts. The verifier did enforce both rules. This produced
measurement failures that mixed task ambiguity with agent capability.

Shell generator/template v2 now states:

- canonical extensions are image `.img`, note `.txt`, data `.csv`, and script
  `.sh`, regardless of the original extension;
- no temporary/helper files may remain; and
- the exact allowed final workspace shape.

## 5. Stratified 24-Pair Gate

`scripts/run_poc_v4.sh` runs the next proof-of-concept gate. It uses a fresh
remote run root, policy v4 only, and 24 paired tasks. Cells are interleaved to
reduce family/time confounding from the shared single-slot model.

The run currently has five completed records and no active rollout/finalizer
process. Rerunning the script is resumable, but should wait until the external
architecture experiment determines whether the current model/evaluation path is
worth further native inference cost.

The seeds are preselected to produce:

```text
train tasks          12  (24 attempts)
validation tasks      4  ( 8 attempts)
test tasks            8  (16 attempts)
total tasks           24  (48 attempts)
```

Every family contributes three train, one validation, and two test tasks.
Python-repair additionally covers all three implemented templates twice.

Run or resume:

```bash
nohup bash scripts/run_poc_v4.sh >.runs/poc-v4.log 2>&1 &
```

## 6. Gate Before Model Interpretation

Proceed through dataset export, training, and held-out evaluation only if:

1. all 24 jobs finish without infrastructure errors;
2. both policies have non-degenerate success rates;
3. paired disagreement remains approximately 15% or higher;
4. failures are semantic outcomes rather than verifier ambiguity;
5. every expected train/validation/test pair is present; and
6. no task mixes policy versions or generator versions under one identifier.

With only 24 tasks, model metrics and bootstrap intervals remain descriptive.
The run proves the workflow and exposes failure modes; it cannot validate a
neural allocator thesis.

## 7. Shortcuts Not Taken

- Do not use a smaller model's outcomes as Gemma training labels.
- External model/harness outcomes may validate the modeling architecture in a
  separate experiment, but must not be relabeled or pooled as Gemma
  Direct/Deliberate treatment outcomes.
- Do not poll the hidden verifier during execution to stop successful attempts;
  that creates an oracle-assisted treatment unavailable in deployment.
- Do not enable concurrent llama-server slots without a separate memory and
  interference benchmark; the 13.6 GB quant already uses about 10 GB of a
  12 GB GPU allocation with CPU offload.
- Do not use adaptive or single-arm sampling for the frozen test set. A later
  training corpus may use randomized single-arm assignment with logged
  propensities, while evaluation remains paired.
- Do not treat summed logical `total_tokens` as physical compute. Report output
  tokens, cache-read tokens, provider turns, and wall time separately.

## 8. Post-Gate Commands

Use the remote CPU PyTorch environment at `.venv/bin/python` after rollout
generation completes:

```bash
bash scripts/finalize_poc_v4.sh
```

The finalizer first runs the descriptive completion/diversity gate and exits
without training if it fails. Its expanded steps are:

```bash
# Export 48 leakage-safe attempt rows.
PYTHONPATH=src python3 -m pyreplab_harness.dataset \
  .runs/poc-v4 .runs/poc-v4/dataset.jsonl

# Train without competing for Gemma's GPU allocation.
PYTHONPATH=src .venv/bin/python -m pyreplab_harness.outcome_model train \
  .runs/poc-v4/dataset.jsonl .runs/poc-v4/model --device cpu

# Evaluate exactly once on the preselected frozen test pairs.
PYTHONPATH=src .venv/bin/python -m pyreplab_harness.allocator_eval \
  .runs/poc-v4/dataset.jsonl .runs/poc-v4/model \
  .runs/poc-v4/allocator-test.json --split test

# Render the descriptive workbench.
PYTHONPATH=src python3 -m pyreplab_harness.dashboard \
  .runs/poc-v4/dataset.jsonl .runs/poc-v4/dashboard.html \
  --metrics .runs/poc-v4/model/metrics.json \
  --baselines .runs/poc-v4/allocator-test.json \
  --title "Terminal Gym v4 Proof of Concept"
```
