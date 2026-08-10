# Outcome Model Learn-Smoke: Did the Brain Learn Anything?

> A noisy synthetic test that asks one question: **can the Bayesian outcome
> model recover a real policy→success signal from noisy outcomes — or does it
> just memorize trivial labels?**
>
> **Status:** synthetic mechanism probe. Proves the *learning machinery* works.
> Says nothing about real agents. Never merge this data into a real experiment.

---

## TL;DR

We planted a known, noisy policy→success structure into 1,792 synthetic rows,
showed the model only the 0/1 outcomes, and asked it to pick the best policy per
task. On held-out data:

| Metric | Result | Verdict |
|---|---|---|
| Allocator success (vs known truth) | **0.679** vs 0.465 random, 0.695 oracle | captures **98%** of the oracle's lift |
| Top-1 policy recommendation | **12 / 12** match ground truth | the decision is right, every time |
| Ranking recovery (Spearman) | **0.74** | it re-learns the latent ordering |
| Brier (model vs base-rate) | **0.239 vs 0.250** | beats the trivial baseline |
| Posterior spread (mean std) | **0.033** | ⚠️ **overconfident** — intervals too tight |

**Answer: yes, it learned something usable** — for the *decision*. The raw
probability *values* and the uncertainty bands are not yet trustworthy.

---

## The Setup

**1. Generate real policies** from the controlled grammar
(3 planning × 2 verification × 2 execution × 3 budgets = 36 possible):

```bash
python -m pyreplab_harness.treatments generate \
  .runs/learn-smoke-treatments.json --count 8 --seed 42
```

This yields 8 immutable treatments like
`decompose-incremental-single-pass-generous` and
`direct-final-retry-on-failure-generous`, each with a SHA-256 identity.

**2. Build a noisy world.** Four task families (`artifact`, `sqlite`,
`python_repair`, `shell`) × three difficulties. Crucially, **different families
reward different policies** — that's what makes the allocation problem
non-trivial:

- 🐍 **python_repair** rewards careful `decompose` + `incremental` + `generous`
- 🗄️ **sqlite** rewards `deliberate` + `incremental` + `moderate`
- 📁 **shell** rewards `direct` + `generous` (many commands, low reasoning)
- 📦 **artifact** rewards `deliberate` + `retry` + `moderate`

**3. The data-generating process (the honest part).** For every
(task, treatment) cell we compute a *planted* latent success probability:

```
true_p = sigmoid( 0.2 + 0.75 * affinity_sum + difficulty_penalty )
```

then draw `verified_success ~ Bernoulli(true_p)`. The model **never sees
`true_p`** — only the noisy 0/1 outcomes. So "learning" means: can it
reconstruct the planted function through ~20 noisy examples per cell?

> ⚠️ **Read this before citing any number:** `true_p` is *constructed by us*,
> not measured. It is a smooth additive function of features the model also
> receives (family + grammar factors). So "can it fit this?" is **partly
> guaranteed feasible by design**. The genuinely non-trivial parts are (a) the
> optimum *differs by family*, requiring a learned family×treatment interaction,
> and (b) doing it through noisy Bernoulli draws, not deterministic labels. This
> is a *mechanism* test, not real-world evidence.

**4. Train + evaluate.** 1,792 rows = 224 tasks × 8 treatments.
Train **1,136** / validation **320** / **test 336** (42 tasks × 8). The model is
a descriptor-aware variational Bayesian Bernoulli head
(text embed + categorical embeds + numeric tower + fusion MLP + variational last
layer). Held-out tasks are split by task, never shared across splits.

---

## The Results

### The allocator is the headline

Because this is synthetic, we *know* the true `p` of every pick, so we score
decision quality against ground truth (not the noisy single draw):

```
               success rate (vs known true_p)
  oracle  ───────────────────────────  0.695   ← hindsight ceiling
  model   ──────────────────────────   0.679   ← what the brain chose
  random  ──────────────               0.465   ← pick a policy by dice roll
```

The model grabs **+0.214** of success rate over random, leaving only **0.016** on
the table vs an oracle that sees the future. That's **98% of the available lift**.

### It also ranks and predicts sensibly

- **Ranking recovery:** Spearman **0.74** between predicted and true success
  across all 32 (family × treatment) cells. MAE **0.060**.
- **Beats the base-rate baseline:** Brier **0.239** vs **0.250** (predicting the
  global mean for everyone).

---

## "Ask the Model" — a Concrete Example

Hold one `python_repair` (hard) task fixed. Show the model all 8 policies and
ask: *which should I use?*

```
rank  policy (planning/verif/exec/budget)      pred   [5–95%]   true_p
 1    decompose/incremental/single-pass/generous  0.581  [.54–.63]  0.605   ✅
 2    decompose/final/single-pass/moderate        0.456  [.42–.49]  0.315
 3    decompose/final/retry/tight                 0.440  [.40–.48]  0.401
 4    direct/final/retry/generous                 0.430  [.37–.50]  0.349
 5    decompose/incremental/single-pass/tight     0.363  [.31–.41]  0.332
 6    direct/incremental/retry/tight              0.360  [.31–.41]  0.284
 7    deliberate/incremental/single-pass/moderate 0.353  [.29–.42]  0.401
 8    deliberate/final/single-pass/tight          0.320  [.27–.37]  0.122
      → model's pick = true best ✅
```

It nailed the decision: **decompose + incremental + generous** for a repair task
— exactly the "careful debugging needs care and budget" intuition we planted. The
same model correctly switches to `direct + generous` for `shell` tasks and
`deliberate + incremental` for `sqlite`. That cross-family switching is the real
test, and it passes 12/12.

The warts are visible too: ranks 2 and 7 are out of order, and the `[5–95%]`
bands are too narrow to cover the true `p` (overconfidence — see below).

---

## What This Proves — and What It Doesn't

**✅ Proves (mechanism):**
- The architecture can fit a smooth, cross-family policy→success function from
  noisy Bernoulli outcomes.
- The allocator decision it produces is genuinely better than random and
  near-optimal on this problem class.
- The descriptor path works: it routes policy identity + budgets into usable
  predictions.

**❌ Does *not* prove:**
- Anything about real Gemma agents or real task families. The signal here is
  smoother and more additive than reality.
- Generalization to **unseen policies**. We trained and tested on the same 8.
  A leave-one-treatment-out test is the real generalization probe (the README
  flags this as the open gap) — and it's where our planted-signal advantage
  disappears.

**⚠️ Known weakness to fix:**
- **Overconfident posterior.** Mean predictive std ≈ **0.033** — the variational
  head collapses to near-deterministic. Point predictions rank well, but the
  uncertainty bands are miscalibrated (too tight). Any feature that depends on
  calibrated uncertainty — selective prediction, deferral, "ask the user" — would
  be unreliable until this is addressed. Likely fixes: stronger KL / larger
  prior sigma, an ensemble, or MC dropout alongside the variational head.

---

## How to Reproduce

```bash
# 1. Environment (one time)
uv venv .venv --python 3.9
uv pip install -r requirements-cv.txt   # torch 2.8.0, numpy, scikit-learn

# 2. Generate the policy registry
PYTHONPATH=src python -m pyreplab_harness.treatments generate \
  .runs/learn-smoke-treatments.json --count 8 --seed 42

# 3. Run the noisy learn-smoke (writes dataset + model + this result)
PYTHONPATH=src .venv/bin/python -m pyreplab_harness.outcome_model_learn_smoke \
  .runs/learn-smoke-treatments.json --output-dir .runs/learn-smoke
```

The run prints a full JSON verdict and exits `0` if
`learned_something_usable` is true.

### Canonical run (this document)

| Field | Value |
|---|---|
| Registry hash | `4607ff665554e9b0ca511b61609b42c3586e4fd5372796fafb4777df8d5580d3` |
| Dataset sha256 | `4c86dbc53ba8040da3f3cbb94cc9cc0b17996eadda82611167f2fec034228278` |
| Rows | 1,792 (train 1,136 / val 320 / test 336) |
| Seeds | data=2024, train=42 |
| Hyperparams | epochs 80 (stopped 25, best 15), batch 32, patience 10, lr 1.5e-3, signal_scale 0.75 |
| Torch | 2.8.0 (CPU) |

> Determinism note: torch CPU + seeded `random` make this reproducible on the
> same torch version. Cross-version torch may drift slightly. The dataset hash
> pins the exact inputs regardless.
