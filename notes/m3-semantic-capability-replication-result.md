# M3 Semantic-Capability Replication Result

Date: 2026-08-12

Verdict: **valid `replication_no_go`**

## Plain-English Result

The larger test did not fully confirm the earlier small result.

The table specialist did better on every table task, and the form specialist
did better on every form task. However, the repeated table results were not
stable enough for the strict rule frozen before the run. This means the
directional specialization is real and useful for research, but it is not yet
reliable enough to justify an automatic chooser or Phase 2.

## Design

The replication was frozen before outcome execution:

- 16 fresh synthetic tasks: eight table and eight form tasks;
- two easy, four medium, and two hard tasks per family;
- two specialist arms and three replicas per task-arm cell;
- 48 paired panels and 96 attempts;
- panel-common sampling seeds and exactly balanced first/second execution;
- no early outcome stop, replacement, or seed substitution; and
- every row permanently `T_canary` / `canary_excluded`.

Fresh four-attempt mechanics passed before the replication. The mechanics run
had zero infrastructure, mechanism, or structural errors and retained four raw
event traces.

## Frozen Gate

`confirmation_pass` required all of the following:

| Check | Required | Observed | Result |
|---|---:|---:|---|
| Complete attempts | 96 | 96 | Pass |
| Infrastructure errors | 0 | 0 | Pass |
| Mechanism errors | 0 | 0 | Pass |
| Structural errors | 0 | 0 | Pass |
| Favorable table tasks | At least 7/8 | 8/8 | Pass |
| Adverse table tasks | At most 1/8 | 0/8 | Pass |
| Favorable form tasks | At least 7/8 | 8/8 | Pass |
| Adverse form tasks | At most 1/8 | 0/8 | Pass |
| Stable table-only tasks | At least 2 | 0 | **Fail** |
| Stable form-only tasks | At least 2 | 6 | Pass |
| Discordant policy-task cells | At most 4/32 | 10/32 | **Fail** |

A stable-only task required the matching arm to succeed in all three replicas
and the nonmatching arm to fail in all three. The table specialist itself was
3/3 on seven table tasks and 2/3 on one. The strict table-only count was zero
because the form arm succeeded at least once on seven table tasks, while the
remaining table task had a 2/3 table result.

## Outcomes

| Task family | Table specialist | Form specialist |
|---|---:|---:|
| Table tasks | 23/24 (95.8%) | 11/24 (45.8%) |
| Form tasks | 4/24 (16.7%) | 24/24 (100%) |
| **All tasks** | **27/48 (56.3%)** | **35/48 (72.9%)** |

The matching specialist was better on all 16 task-level comparisons. Across
the 48 paired task-replica panels there were 12 table-only successes, 20
form-only successes, 15 both-success panels, and one both-fail panel.

The form specialist used a mean 2,962.9 output tokens per attempt; the table
specialist used 1,889.6. These totals are descriptive and were not gate
criteria. Across both arms the dataset contains 62 verified successes and 34
failures.

## Interpretation

The larger run overturns the small screen's progression qualification. Do not
build or train an automatic chooser from these rows, and do not claim the
specialization is repeat-stable under the frozen standard.

The directional result remains strong: each matching specialist won every
task-level comparison. The failure is specifically about repeat stability and
strict isolation, especially on table tasks. Future work should first study
why the nonmatching form arm intermittently solves table tasks and why one
table-specialist cell failed, rather than immediately collecting another
confirmatory panel.

The result applies only to the two synthetic fixture templates. It does not
establish mixed-page routing, real-web performance, a causal specialist effect,
or allocator effectiveness. The frozen M3 no-go and Phase 2 lock remain
unchanged.

## Dataset Release

The full run is retained as an audit-ready private dataset package at:

`.runs/m3-semantic-replication-20260812-v1.dataset-v1.0.0`

The package contains:

- 96 privacy-scanned attempt rows with task prompts, contracts, public task
  metadata, treatment identity, execution position, outcome, failure code,
  usage, timing, mechanism checks, and raw-event hashes;
- 96 inclusion-ledger rows;
- the authoritative gate report;
- 591 raw-source files with relative paths, byte sizes, and SHA-256 hashes;
- a quality audit and dataset card; and
- explicit false eligibility for training, calibration, development, and final
  evaluation on every row.

The standard safe exporter independently produced 96 rows, zero skips, and 96
`canary_excluded` labels. The package verifier passed file hashes, raw hashes,
all-cell coverage, row governance, ledger alignment, gate validity, and privacy
checks.

The package is internal-research-only until model-output redistribution terms
are reviewed. Raw traces are restricted because they may contain model
messages, thinking, tool payloads, diagnostics, and local paths.

## Artifact Identities

- Canonical dataset contract file:
  `policies/m3-unbrowser-semantic-replication-20260812-v1.dataset-contract.json`
- Dataset contract file SHA-256:
  `97f58c15d9af80bbd0c59d4083eeab55f5f66039d4905d612e082abebaeb34e8`
- Dataset contract hash:
  `1c41142e78af868aa5fbb25bfb832706419e580ebdc6d2528d3d0cfd3e291af6`
- Replication manifest hash:
  `efaa3ee3d623d4bfd35a30ef9271aa7998ce7bbf0117b80b5aa1898801a04aaa`
- Frozen execution source-tree hash:
  `b0fc294a41d280cabb80a93acb5e4e0f3096e9f3c46ee6bd2497a8599ebd2f55`
- Mechanics manifest hash:
  `61a16b77953d302fb83bc9c95ebe4c86fb479fe86ed2bdee3c56cb57ecc3e3b7`
- Mechanics result SHA-256:
  `5a6bedc03b94b38a87d0e3836e6e60af0e6bfedccfdc9dd5ca9d916ed70f0fd1`
- Mechanics gate SHA-256:
  `1ac8b58aaf4a4502d307a52e01108baef445ebe5d2a8e58dbe0f3e0b70159f4f`
- Replication result SHA-256:
  `992951e0b248c3ddcad5b7fbb5c608309e7755c014726c3210b703a254525460`
- Replication preflight SHA-256:
  `ee61d1eb9efe6dffb75dcdf982fa6900a21432c75b13c3b47ffc6eb7166534e3`
- Replication gate SHA-256:
  `7630568507e82c2d33b3e01f7cc89833f8b8b1d78f6c74e3a2e3e43a7c670afd`
- Standard safe export SHA-256:
  `3914416d5b833a13bcbf7ee710f53307105371201e26b67fb31b9448d0cb2daf`
- Dataset package manifest SHA-256:
  `25a6da26bbc9032b1cb3cd87e67874c3ab2ebdd24ff1bbd559957a665dd777a5`
- Curated 96-row dataset SHA-256:
  `f01f2f918384abfa1c2a5a08006deb7d93ddf9883f77f112e5e7a3b61ffa75a1`
- Raw inventory SHA-256:
  `22575e2423a5565bdf073ae54abe29fd351a8dbbfa2181052a0631cbefe3204e`
- Compressed package SHA-256:
  `58894e0a43a3de6e4c4a55cab8ef3ebe0a81c9051852e8d1ae2f1e2b64f0dac6`

The ignored `.runs/` directory remains the private artifact store. The
committed policies and this note provide the public audit pointer without
publishing raw model traces.

The contract hash is the canonical hash of the contract's logical JSON content
with its `contract_hash` field omitted. The contract file SHA-256 hashes the
actual formatted file bytes; both are recorded intentionally.
