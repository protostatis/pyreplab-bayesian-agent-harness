# M3 Semantic Replication Dataset Card

Release: `m3-semantic-replication-20260812-v1.dataset-v1.0.0`

Status: private, verified, internal research only

## Summary

This release contains 96 completed browser-agent attempts from 16 fresh
synthetic tasks, two specialist arms, and three replicas. It is designed for
audit, mechanics regression, descriptive screening, and error analysis.

Every row is `T_canary` / `canary_excluded` and explicitly ineligible for the
current M3 training, calibration, development, and final-evaluation pools.
Current model and allocator loaders fail closed on excluded split labels.

## Location

- Package directory:
  `.runs/m3-semantic-replication-20260812-v1.dataset-v1.0.0`
- Compressed package:
  `.runs/m3-semantic-replication-20260812-v1.dataset-v1.0.0.tar.gz`
- Full generated card:
  `.runs/m3-semantic-replication-20260812-v1.dataset-v1.0.0/DATASET_CARD.md`
- Quality audit:
  `.runs/m3-semantic-replication-20260812-v1.dataset-v1.0.0/QUALITY_AUDIT.json`

## Data Files

- `data/attempts.jsonl`: 96 privacy-scanned rows, one per planned cell.
- `analysis/inclusion-ledger.jsonl`: 96 governance decisions.
- `analysis/gate-report.json`: authoritative `replication_no_go` report.
- `raw/inventory.jsonl`: 591 content-addressed raw artifacts.
- `MANIFEST.json`: package schedule, source identities, counts, and hashes.

The curated rows include task prompt and contract, safe public metadata,
treatment and bundle identity, replica and sampling seed, execution position,
verification outcome and failure code, token usage, timing, specialist receipt
status, admitted/rejected call counts, and the relative raw event hash.

They exclude raw assistant messages and thinking, tool arguments and payloads,
stderr, verifier diagnostics, private oracle data, and absolute paths.

## Use Boundary

Allowed now:

- reproducibility and audit;
- accounting and mechanics regression tests;
- descriptive task/arm/replica analysis;
- failure and cost analysis; and
- planning a new prospectively frozen experiment.

Prohibited in current M3:

- training or fine-tuning;
- calibration;
- development or policy selection;
- final evaluation;
- allocator training; and
- causal or population-generalization claims.

A future experiment may name this exact immutable release as historical
training or prior information only before collecting a fresh untouched
evaluation panel. That does not make these already observed rows prospective
evidence.

## Verification

`python -m pyreplab_harness.m3_semantic_dataset verify <package>` passes:

- package file hashes;
- all 591 raw artifact hashes;
- exact 96-cell coverage;
- row schema and exclusion metadata;
- inclusion-ledger alignment;
- authoritative valid gate status; and
- privacy scanning of every derived file.

Key release hashes:

- Logical dataset contract:
  `1c41142e78af868aa5fbb25bfb832706419e580ebdc6d2528d3d0cfd3e291af6`
- Formatted contract file:
  `97f58c15d9af80bbd0c59d4083eeab55f5f66039d4905d612e082abebaeb34e8`
- Package manifest:
  `25a6da26bbc9032b1cb3cd87e67874c3ab2ebdd24ff1bbd559957a665dd777a5`
- Curated attempt rows:
  `f01f2f918384abfa1c2a5a08006deb7d93ddf9883f77f112e5e7a3b61ffa75a1`
- Raw inventory:
  `22575e2423a5565bdf073ae54abe29fd351a8dbbfa2181052a0631cbefe3204e`
- Compressed package:
  `58894e0a43a3de6e4c4a55cab8ef3ebe0a81c9051852e8d1ae2f1e2b64f0dac6`

## Limitations

The release covers only harness-owned synthetic table and form fixtures. It is
targeted rather than population sampled. Replicas are repeated measurements,
not independent tasks. The valid replication gate returned no-go because
repeat discordance and strict table-only stability missed their thresholds.
See `m3-semantic-capability-replication-result.md` for interpretation.
