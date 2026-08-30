# AGENTS.md — Working Rules for This Repo

## Communication style (feedback from the operator, 2026-08-30)

Reports are read by the project operator, not by another agent. Shorthand and
compressed jargon that assume the whole experiment is loaded in the reader's
head do not help. Rules:

- **No letter-code labels without the full name.** Write "the
  recovery-discipline prompt arm" — never bare "R-arm", "the R arm", "C−E".
  On first use in a message, give the plain name; the short code may follow
  in parentheses if a table needs it.
- **No cryptic noun piles.** "Expert post-error behavior" means nothing on
  its own. Write out the sentence: "When a tool call failed, what did the
  agent do next?"
- **Explain numbers, don't just report them.** For every statistic, say in
  plain words what it measures and why it matters (e.g., "across the 48
  finished tasks, the baseline prompt succeeded 54% of the time, while the
  recovery-discipline prompt succeeded only 29%").
- **Lead with the plain conclusion.** First sentence = what happened and
  whether it's good or bad. Evidence and numbers come after.
- **Say what a term means the first time it appears in a session**, even if
  it was defined in earlier sessions (e.g., "the 72-task test run", "the
  result-detection fix").
- **Tables need captions.** A table of rates without one sentence saying what
  the rows and columns are forces the reader to reverse-engineer it.
- Same rules apply to commit messages and notes documents: a reader six
  months from now should understand them without this chat's context.
- **Do not coin jargon.** Most of this project's vocabulary was invented
  here ("panel", "slot-clear", "panel-major", "bounded-damage"). If a plain
  sentence does the job, write the sentence instead of minting a term. New
  project terms are allowed only when they are added to
  `notes/glossary-plain-language.md` with a plain-language definition — and
  every first use in a report must still be glossed.
- **The glossary is the bridge:** `notes/glossary-plain-language.md` maps
  every project term to everyday language. Read it before writing reports;
  when in doubt, use the glossary's phrasing verbatim.

## Operational rules carried from governance

- During an active frozen run, do not edit `src/`, `policies/`, or tests —
  any edit invalidates the frozen bundle hash. Analysis work lives in
  `.runs/` and `notes/`.
- Governance decisions and gate rulings are appended to
  `.runs/v11-governance-decision.md`; experiment specs live in `.runs/` as
  `vN-experiment-spec.md` before each freeze.
- Long runs launch inside tmux wrapped in `caffeinate -ism`; authorizations
  must have TTL ≥ max(72h, 1.5× projected runtime) from launch.
