# Plain-Language Glossary — This Project's Terms

Reference for every term used in reports. Each entry: the term, then what it
actually means. Terms marked **(coined)** were invented during this project
(or by the agent mid-session) and have no meaning outside this repo. Terms
marked **(standard)** are ordinary statistics/experimental-design words.

## The experiment itself, in one paragraph

We give a coding agent (a Gemma model) small website tasks — fill in a form
correctly, or find a number in a table while ignoring decoy links. Each task
has a built-in checker that says pass or fail. We compare three versions of
the agent's instruction prompt to see which finishes tasks successfully more
often. The full comparison is 12 tasks × 2 copies of each task × 3 prompt
versions = **72 runs**. The 12 tasks are permanently barred from being used
as training data later — they exist only for this comparison.

## Core vocabulary

- **run / cell** — one timed attempt: one task, one copy, one prompt version,
  with its pass/fail outcome. 72 total.
- **panel** **(project term)** — the group of 3 runs that share the same task
  and copy (one per prompt version). Comparisons are fairest *within* a
  panel, because everything except the prompt is identical.
- **prompt version / arm** **("arm" is standard clinical-trial wording)** —
  which of the 3 instruction prompts a run used:
  - **baseline** (code E): plain instructions, no discipline rules.
  - **execution-discipline** (code C): adds "save one tool call for writing
    the result file; stop once it's written."
  - **execution + recovery discipline** (code R): adds "when a tool call
    fails, read the error, never repeat the same failed call, fix it once,
    then try a different approach or stop."
- **screen / arm-screen** **(project term)** — the whole comparison
  experiment whose only job is to pick which prompt version to use later.
- **generation (v11, v12, … v15)** **(project term)** — one fresh attempt at
  the 72-run comparison. Each generation re-locks code and settings from
  scratch; a failed generation is never "continued", only replaced.
- **the 12-task bank / T_pilot** — the fixed task set above. **(project
  term)** Permanently excluded from all model training.

## Rules and mechanics

- **freeze** — lock the exact code and settings before a run starts.
- **bundle hash / drift** — the frozen code's fingerprint. "Drift" = code
  changed after the freeze, which invalidates the run.
- **slot-clear** **(coined)** — wait until the previous run's sandbox is
  fully shut down before starting the next run.
- **authorization TTL** — how long the run's permission ticket stays valid.
  The v14 run died because the ticket was minted with 24 hours left for a
  ~40-hour run.
- **checkpoint snapshot** — a frozen copy of results at 24 and 48 runs, used
  for mid-run review.

## Measurement

- **success rate** — share of runs where the checker passed, per prompt
  version. Always say the counts ("8 of 14 runs") not just the percent.
- **intention-to-treat / ITT** **(standard)** — count every run as-is,
  including misbehaving ones; never re-run or drop failures.
- **confidence interval (95% CI)** **(standard)** — the plausible range for
  the true difference. If the range includes zero, the data can't rule out
  "no difference".
- **p-value / sign test** **(standard)** — chance of seeing a gap this big if
  the prompts were actually equal.
- **task-clustered uncertainty** **(standard method, project application)** —
  runs of the same task correlate, so uncertainty is computed per task (only
  12 groups) rather than per run. This is why gaps look "noisy" — 12 groups
  is very little data.
- **decisiveness rule** **(project term)** — a prompt difference only
  "counts" if its confidence range excludes zero AND the sign test gives
  p < 0.01. Deliberately strict.
- **completion label / detector** — whether the run actually wrote its result
  file. The v1 detector had a bug that produced "unknown"; the v2 detector
  (v12) watches the actual file and fixed it.
- **discordance flag** — an audit trigger when prompt versions' success rates
  differ by more than 25 points; audit, not stop.

## Schedule and failure handling

- **panel-major / superblock** **(coined, v15)** — the new run order: finish
  all 3 prompt versions of one task back-to-back, cycling through groups of
  one easy + one medium + one hard task. Means useful comparisons accumulate
  from the start instead of only near the end.
- **bounded-damage ordering** **(coined, v14)** — order runs so an
  interruption spreads loss across task difficulties instead of losing one
  difficulty entirely (what killed v11's usefulness).
- **clean-boundary stop** **(coined, v14 closure)** — the run stopped exactly
  between two runs, not mid-run, so every finished record is complete and
  trustworthy.
- **terminal-invalid / no-resume** — if a generation is interrupted, it is
  closed forever; the finished prefix is kept as evidence only and the
  comparison is redone from scratch in a new generation.

## Governance gates (G1–G8) **(project terms)**

Checkpoints that must pass before model training data may be collected:
G1–G4 close out a failed run cleanly (losses explained, artifacts locked,
labels reconciled, no detector contamination); G5 = the fixed detector
passed its test battery; G6 = the training plan is frozen before collection;
G7 = training tasks don't overlap screening tasks; G8 = the measuring
instrument stays unchanged during collection.
