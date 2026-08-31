# M3 Semantic Replication Diagnostic

Date: 2026-08-13

Status: **post-hoc descriptive error analysis; frozen `replication_no_go`
unchanged**

## Purpose And Boundary

This report explains the unstable table cells in the verified 96-attempt M3
semantic replication. It does not amend the preregistered gate, qualify a
capability family, justify an allocator, or unlock Phase 2.

The analysis uses the verified private dataset package and emits aggregate
counts only. It excludes model messages, reasoning, result bodies, URLs,
selectors, element references, diagnostics, private oracle values, and absolute
paths. Every source row remains `T_canary` / `canary_excluded` and ineligible
for training, calibration, development, and final evaluation.

## Frozen Result

The authoritative gate remains a valid `replication_no_go`:

| Check | Observed | Required | Result |
|---|---:|---:|---|
| Favorable table tasks | 8/8 | At least 7/8 | Pass |
| Favorable form tasks | 8/8 | At least 7/8 | Pass |
| Stable table-only tasks | 0 | At least 2 | **Fail** |
| Stable form-only tasks | 6 | At least 2 | Pass |
| Discordant policy-task cells | 10/32 | At most 4/32 | **Fail** |

The diagnostic below explains those failures. It does not replace them with a
different decision rule.

## Action Paths

| Task family and arm | Success | Mean admitted calls | Mean total tokens | Mean seconds | At cap | Aborted |
|---|---:|---:|---:|---:|---:|---:|
| Table, table specialist | 23/24 | 5.792 | 34,282 | 55.6 | 1 | 1 |
| Table, form specialist | 11/24 | 10.750 | 102,782 | 154.8 | 14 | 13 |
| Form, form specialist | 24/24 | 4.167 | 23,837 | 31.6 | 0 | 0 |
| Form, table specialist | 4/24 | 11.375 | 75,973 | 71.6 | 20 | 19 |

The matching specialist succeeded in 47/48 attempts. The nonmatching
specialist succeeded in 15/48. That is strong exploratory routing evidence, but
it is not the repeat-stable isolation required by the frozen gate.

Both treatments retained generic Unbrowser actions. The form arm therefore had
a manual fallback path on table tasks even though it lacked `semantic_table`.
Across its 24 table attempts, that arm requested 144 generic query calls and
succeeded 11 times. The matching table arm requested 13 generic query calls and
succeeded 23 times.

The nonmatching table path was much less efficient:

| Form specialist on table tasks | Successes | Mean admitted calls | Generic queries | At cap | Aborted | Submitted |
|---|---:|---:|---:|---:|---:|---:|
| Successful attempts | 11/11 | 9.545 | 41 | 2 | 1 | 11 |
| Failed attempts | 0/13 | 11.769 | 103 | 12 | 12 | 0 |

This is the direct explanation for zero stable table-only tasks. One
nonmatching success in any of three replicas is enough to defeat the strict
stable-only definition.

## Unstable Table Cells

The form specialist on table tasks had seven discordant task cells and one
stable-failure cell. It had no stable-success cell:

| Replica outcome pattern | Task cells |
|---|---:|
| Failure / failure / failure | 1 |
| Failure / failure / success | 2 |
| Failure / success / failure | 1 |
| Failure / success / success | 1 |
| Success / failure / success | 3 |

The table specialist had seven stable-success table cells and one discordant
cell with success / success / failure. The frozen gate therefore saw eight
discordant table policy-task cells in total: seven from the nonmatching arm and
one from the matching arm. The other two discordant cells were on form tasks.

Post-hoc execution slices for the form arm on table tasks were:

| Slice | Success |
|---|---:|
| Execution position 0 | 3/11 |
| Execution position 1 | 8/13 |
| Replica 0 | 3/8 |
| Replica 1 | 2/8 |
| Replica 2 | 6/8 |

These groups are small and observational. They do not support a carryover,
order, or replica effect claim.

## Sole Matching Failure

The sole matching table failure used one unbounded `semantic_table` request,
then made eight generic queries, reached 12 admitted calls, aborted, and never
submitted a result. A successful comparator on the same task used a fully
bounded semantic-table request and submitted in six admitted calls.

This is an isolated action-selection failure, not an infrastructure or tool
delivery failure. The replication had zero infrastructure, mechanism, and
structural errors.

## Interpretation

The replication answers two different questions differently:

1. **Did matching specialists provide useful capability?** Yes. The matching
   arm won every task-level comparison and succeeded in 47/48 attempts.
2. **Were the two capabilities strictly isolated and repeat-stable under the
   frozen rule?** No. Generic Unbrowser actions let nonmatching arms sometimes
   solve the other task family, especially tables, and ten cells were
   discordant.

The valid no-go must therefore remain in force. The result does not support an
automatic chooser because these rows contain only pure-family synthetic tasks,
not mixed pages or prospective task-family classification.

## Recommended Next Step

Do not buy another confirmatory replication yet. First choose and preregister
which claim the next design should test:

1. **Strict capability isolation:** remove or constrain generic fallback paths
   in each specialist and require bounded semantic requests prospectively.
2. **Efficient task routing:** retain generic fallbacks, but replace the
   isolation gate with a prospective mixed-task design that measures routing
   classification, success, tool calls, tokens, and latency against fixed
   always-table and always-form policies.

The second path better matches the observed evidence. It treats fallback
success as robustness with a cost penalty rather than as proof that
specialization is absent. It still requires a new frozen protocol and fresh
tasks; this post-hoc dataset cannot supply allocator evidence.

## Reproduction

Analyzer:

`src/pyreplab_harness/m3_semantic_diagnostic.py`

Command:

```bash
.venv/bin/python -m pyreplab_harness.m3_semantic_diagnostic \
  .runs/m3-semantic-replication-20260812-v1.dataset-v1.0.0 \
  --output .runs/m3-semantic-replication-20260812-v1.diagnostic.json
```

Generated report SHA-256:

`a99088d335a196306e8731046ab7b4b28ce801a46582e712081fd9dd1746a271`

The report was rebuilt independently and was byte-identical. Its source hashes
bind the 96 curated rows, 96 normalized executions, 96 raw event traces, gate
report, package manifest, and raw inventory without publishing their contents.
