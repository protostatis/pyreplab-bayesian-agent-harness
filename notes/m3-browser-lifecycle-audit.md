# M3 Browser Lifecycle Audit

Date: 2026-08-12

Status: confirmed validity defect

## Finding

The confined Unbrowser launch command wrapped the persistent browser process in
GNU `timeout` using the configured 30-second per-action timeout. GNU `timeout`
measures total process lifetime, while the host RPC deadline correctly measures
each request. After about 30 seconds, the wrapper killed the otherwise persistent
process and the next RPC write surfaced as `BrokenPipeError`.

The worker returned those failures as ordinary model-visible tool errors. The
observation-canary gate therefore reported zero infrastructure errors even
though browser infrastructure had died.

## Retained-trace evidence

- Observation outcome screen: 24 broken-pipe events across 21/24 attempts.
- Broken pipes affected 18/19 observed failures.
- Earliest broken pipe was about 29.8 seconds after successful navigation;
  median was about 34.3 seconds.
- Earlier exploratory Stage 2: 86 broken pipes across 65/96 attempts, affecting
  53/54 failures; the same approximately 30-second boundary appears.
- The only observation-screen failure without a broken pipe followed a 64 KiB
  result overflow that killed browser session state.

Navigation occurs after process launch, so elapsed time from navigation is a
conservative lower bound on process age.

## Decision impact

The original immutable gates remain preserved, but outcome interpretations that
treated these browser deaths as policy failures are invalid. In particular, the
observation-enforcement screen's `futility_no_go` is superseded by `invalid`.
The earlier exploratory rankings are also descriptive artifacts of a corrupted
execution substrate and cannot support policy selection.

## Repair

The persistent-process lifetime is now separated from the per-request deadline.
The process-wide GNU `timeout` wrapper was removed; each RPC request retains its
30-second host deadline and the enclosing agent attempt retains its frozen
wall-clock limit. Pre-write process-exit, broken-pipe, startup, and response-
timeout failures now carry a structured infrastructure marker through the
worker and Pi tool details. The observation and semantic gates inspect that
marker even when Pi renders the failure as a model-visible tool result.
Oversized model results no longer destroy session state, allowing a narrower
retry.

A real Ubuntu Bubblewrap probe navigated a fixture, waited 35 seconds, and then
completed a second RPC in the same process. The full local suite passed 1,131
tests with 83 skips; the deployed Ubuntu suite passed 1,130 tests with no skips.

Re-evaluating the preserved 24-attempt outcome artifact with the repaired gate
returns `invalid`, with 24 infrastructure-error entries across 21 attempts. The
original artifact and original gate remain unchanged; the repaired audit report
is a derived reclassification, not a rewrite of frozen evidence.

No prior outcome artifact is rewritten. Any new evidence uses fresh task,
sampling, manifest, and run coordinates after deterministic lifecycle stress
tests pass.
