"""Noisy synthetic descriptor-learning smoke for the treatment-aware outcome model.

Purpose
-------
This probe generates a **treatment-held-out** split over a policy grammar
registry.  The model sees training/validation rows only for *training
treatments*; the test split contains *held-out treatments* on *held-out
tasks*.  All identity-bearing categoricals (``policy_id``, ``policy_version``,
treatment ``bundle_id``) are neutralized to constant placeholders in every
``model_input`` so the model **cannot** memorize treatment identity — it must
learn from descriptor text, numeric budget fields, and tool-interface /
allowed-tools-signature categories.

The treatment-held-out split is **deterministic and coverage-aware**:
training treatments are selected via a seeded greedy algorithm that ensures
every grammar-factor level used by any held-out treatment is also represented
in at least one training treatment.  No random shuffle-and-pray.

Evaluation quantifies how well the learned descriptor representation
generalizes to unseen treatment bundles, measured by:

* "expected p_star allocation lift" over a predeclared held-out baseline
  and random, scored against the known true ``p_star``,
* treatment-held-out ranking recovery (a stdlib tie-aware Spearman ρ),
* and a descriptor-only counterfactual top-1 match rate.

The data-generating process is noisy Bernoulli(p_star) where p_star depends
on latent per-family affinities, difficulty, and an additional per-task
continuous variation (``task_variant``) visible in the model input.
``task_variant`` modulates the contribution of each grammar factor
differently, so treatment rankings genuinely flip across tasks.  All
labels and rankings are synthetic.  Scores are research-sandbox evidence
about the *descriptor-learning mechanism*, never about any real agent or
task family.

CLI
---
``python -m pyreplab_harness.outcome_model_learn_smoke REGISTRY.json [options]``
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import outcome_model as om
from .treatments import (
    TreatmentRegistry,
    TreatmentSpec,
    treatment_model_input_descriptor,
)

# ---------------------------------------------------------------------------
# Synthetic task families and latent factor affinities.
# Families mirror real gym names so the categorical schema is identical,
# but the tasks and labels here are entirely synthetic.
# ---------------------------------------------------------------------------

FAMILIES = ("artifact", "sqlite", "python_repair", "shell")
DIFFICULTIES = ("easy", "medium", "hard")

# Per-family logit affinity for each grammar factor level.
_FAMILY_AFFINITIES: dict[str, dict[str, dict[str, float]]] = {
    "python_repair": {
        "planning": {"direct": -0.8, "deliberate": 0.4, "decompose": 1.0},
        "verification": {"final": -0.4, "incremental": 0.7},
        "execution": {"single-pass": -0.6, "retry-on-failure": 0.9},
        "budget": {"tight": -0.7, "moderate": 0.3, "generous": 0.8},
    },
    "sqlite": {
        "planning": {"direct": -0.3, "deliberate": 0.8, "decompose": 0.5},
        "verification": {"final": -0.5, "incremental": 0.6},
        "execution": {"single-pass": -0.2, "retry-on-failure": 0.5},
        "budget": {"tight": -0.6, "moderate": 0.6, "generous": 0.4},
    },
    "shell": {
        "planning": {"direct": 0.7, "deliberate": -0.2, "decompose": -0.6},
        "verification": {"final": 0.4, "incremental": -0.3},
        "execution": {"single-pass": 0.3, "retry-on-failure": 0.2},
        "budget": {"tight": -0.5, "moderate": 0.1, "generous": 0.8},
    },
    "artifact": {
        "planning": {"direct": -0.4, "deliberate": 0.7, "decompose": 0.2},
        "verification": {"final": 0.2, "incremental": 0.3},
        "execution": {"single-pass": -0.3, "retry-on-failure": 0.6},
        "budget": {"tight": -0.4, "moderate": 0.5, "generous": 0.3},
    },
}

_DIFFICULTY_PENALTY = {"easy": 0.0, "medium": -0.5, "hard": -1.2}

_BASE_INTERCEPT = 0.2

# Per-task variant score: each task gets a different continuous modifier,
# visible in public_metadata.  The variant modulates every grammar factor
# via _factor_modulation so the model must learn factor × variant interaction.
_TASK_VARIANT_RANGE = (-0.5, 0.5)

_GRAMMAR_FACTORS = ("planning", "verification", "execution", "budget")


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _factor_modulation(variant: float, offset: float) -> float:
    """Smooth periodic modulation so treatment rankings can flip across task
    variants.  Each grammar factor (planning/verification/execution/budget)
    receives a variant-dependent weight via a cosine with a phase offset,
    ensuring that for some tasks planning dominates while for others budget
    or execution dominates.
    """
    return math.cos(variant * math.pi + offset)


def treatment_logit(
    treatment: TreatmentSpec,
    family: str,
    difficulty: str,
    *,
    task_variant: float = 0.0,
    signal_scale: float = 0.75,
) -> float:
    """Latent logit success for a (treatment, family, difficulty, variant) cell.

    ``task_variant`` modulates the contribution of each grammar factor
    differently (via ``_factor_modulation`` with distinct phase offsets),
    so the *relative ordering* of treatments can genuinely flip across task
    variants.  The variant value itself is exposed in
    ``public_metadata.task_variant`` so the model can learn this interaction.
    """
    meta = treatment.generator_metadata
    aff = _FAMILY_AFFINITIES[family]
    mod_p = _factor_modulation(task_variant, 0.0)
    mod_v = _factor_modulation(task_variant, math.pi / 2)
    mod_e = _factor_modulation(task_variant, math.pi)
    mod_b = _factor_modulation(task_variant, 3 * math.pi / 2)
    raw = (
        aff["planning"][str(meta.get("planning"))] * mod_p
        + aff["verification"][str(meta.get("verification"))] * mod_v
        + aff["execution"][str(meta.get("execution"))] * mod_e
        + aff["budget"][str(meta.get("budget"))] * mod_b
    )
    return (
        _BASE_INTERCEPT
        + signal_scale * raw
        + _DIFFICULTY_PENALTY[difficulty]
    )


def treatment_true_p(
    treatment: TreatmentSpec,
    family: str,
    difficulty: str,
    *,
    task_variant: float = 0.0,
    signal_scale: float = 0.75,
) -> float:
    """True latent Bernoulli success probability for a cell."""
    return _sigmoid(
        treatment_logit(
            treatment, family, difficulty,
            task_variant=task_variant, signal_scale=signal_scale,
        )
    )


# ---------------------------------------------------------------------------
# Synthetic model_input construction
# ---------------------------------------------------------------------------


def _task_prompt(family: str, difficulty: str, seed: int, variant: float) -> str:
    return (
        f"Synthetic {family} task ({difficulty}, seed={seed}, "
        f"variant={variant:+.3f}). "
        f"Produce the required {family} artifact following the stated contract. "
        f"Difficulty band: {difficulty}."
    )


#: Constant placeholder used to neutralize direct treatment-identity
#: categoricals (policy_id, policy_version, treatment bundle_id) so the
#: model cannot cheat by learning a policy-id -> outcome lookup.  All
#: between-treatment variation must come through the descriptor text and
#: numeric/tool categorical fields.
_IDENTITY_PLACEHOLDER = "synthetic-treatment"


def _model_input(
    treatment: TreatmentSpec,
    family: str,
    difficulty: str,
    seed: int,
    variant: float,
) -> dict[str, Any]:
    """Build a ``model_input`` with neutralized identity categoricals.

    ``policy_id`` and ``policy_version`` are set to a constant placeholder
    so the model cannot memorize per-ID success rates.  The treatment
    descriptor retains text, numeric budget fields, ``tool_interface``, and
    ``allowed_tools_signature``, but its ``bundle_id`` is also neutralized.
    """
    prompt = _task_prompt(family, difficulty, seed, variant)
    desc = treatment_model_input_descriptor(treatment, task_text=prompt)
    # Neutralize treatment-identity categoricals so the model learns from
    # descriptors only.
    return {
        "text": prompt,
        "family": family,
        "template_id": f"{family}-{difficulty}-v1",
        "difficulty": difficulty,
        "public_metadata": {
            "seed": float(seed),
            "difficulty_index": float(DIFFICULTIES.index(difficulty)),
            "task_variant": float(variant),
        },
        # Neutralized identity fields — same value for every row.
        "policy_id": _IDENTITY_PLACEHOLDER,
        "policy_version": "1",
        "treatment": {
            "text": desc["text"],
            "max_output_tokens": desc["max_output_tokens"],
            "tool_call_limit": desc["tool_call_limit"],
            "command_timeout_seconds": desc["command_timeout_seconds"],
            "wall_time_limit_seconds": desc["wall_time_limit_seconds"],
            "tool_interface": desc["tool_interface"],
            "allowed_tools_signature": desc["allowed_tools_signature"],
            # Neutralized bundle_id — same placeholder for every treatment.
            "bundle_id": _IDENTITY_PLACEHOLDER,
            "policy_id": _IDENTITY_PLACEHOLDER,
            "policy_version": "1",
        },
    }


# ---------------------------------------------------------------------------
# Panel + dataset construction with treatment-held-out splits
# ---------------------------------------------------------------------------


def _deterministic_coverage_split(
    treatments: list[TreatmentSpec],
    *,
    data_seed: int,
    train_treatment_frac: float,
) -> tuple[list[TreatmentSpec], list[TreatmentSpec]]:
    """Deterministic coverage-aware train/held-out treatment split.

    Treatments are processed in a seeded permutation order.  Training
    treatments are greedily selected until every grammar-factor level that
    appears in the full registry is represented in the training set.
    The training set is then filled to the requested fraction from the same
    seeded order. Remaining treatments (at least 2, verified by the caller)
    become held-out. This preserves factor coverage without starving training
    of policy combinations.
    """
    if len(treatments) < 3:
        raise ValueError(
            f"registry must contain at least 3 treatments to support "
            f"coverage-aware held-out split, got {len(treatments)}"
        )

    # Collect all factor levels present anywhere in the registry.
    all_levels: set[tuple[str, str]] = set()
    for t in treatments:
        meta = t.generator_metadata
        for factor in _GRAMMAR_FACTORS:
            all_levels.add((factor, str(meta[factor])))

    # Seeded permutation of indices.
    rng = random.Random(data_seed)
    indices = list(range(len(treatments)))
    rng.shuffle(indices)

    # Greedy: add treatments to train until all levels are covered.
    train_set: set[int] = set()
    covered: set[tuple[str, str]] = set()
    for idx in indices:
        if covered >= all_levels:
            break
        t = treatments[idx]
        train_set.add(idx)
        for factor in _GRAMMAR_FACTORS:
            covered.add((factor, str(t.generator_metadata[factor])))

    target_train = min(
        max(len(train_set), math.ceil(len(treatments) * train_treatment_frac)),
        len(treatments) - 2,
    )
    for idx in indices:
        if len(train_set) >= target_train:
            break
        train_set.add(idx)

    # All remaining become held-out.
    held_out_indices = [i for i in indices if i not in train_set]

    if len(held_out_indices) < 2:
        raise ValueError(
            f"after coverage-aware split only {len(held_out_indices)} held-out "
            f"treatment(s) remain (need >= 2). Increase registry size or "
            f"reduce the number of unique factor levels."
        )

    train_treatments = [treatments[i] for i in sorted(train_set)]
    held_out_treatments = [treatments[i] for i in sorted(held_out_indices)]
    return train_treatments, held_out_treatments


def build_treatment_held_out_rows(
    registry: TreatmentRegistry,
    *,
    tasks_per_cell: tuple[int, int, int] = (16, 12, 8),
    signal_scale: float = 0.75,
    data_seed: int = 2024,
    train_treatment_frac: float = 0.7,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str, float], float], set[str], set[str]]:
    """Build noisy panels with treatment-held-out splits.

    Registry treatments are split into *training treatments* and *held-out
    treatments* by a deterministic coverage-aware algorithm (see
    :func:`_deterministic_coverage_split`).  Training/validation rows only
    contain training treatments; test rows contain held-out treatments on
    held-out tasks.

    Because ``model_input`` identity categoricals are neutralized to
    :data:`_IDENTITY_PLACEHOLDER`, neither the preprocessor vocabulary nor the
    model can distinguish treatments by ID — all between-treatment variation
    must come through descriptor text, numeric budget, and tool categories.

    Every task is a complete panel (all applicable treatments).  Tasks are
    assigned to splits deterministically, stratified so every
    ``(family, difficulty)`` cell contributes to every split.

    Returns
    -------
    rows : list[dict]
        All data rows with ``split`` in {``"train"``, ``"validation"``, ``"test"``}.
    true_p : dict
        Lookup ``(family, difficulty, treatment.bundle_id, variant) -> p_star``.
    train_bundle_ids : set
        Bundle IDs of training treatments.
    held_out_bundle_ids : set
        Bundle IDs of held-out treatments.
    """
    _validate_build_inputs(
        tasks_per_cell,
        signal_scale,
        train_treatment_frac,
        train_frac,
        val_frac,
    )

    rng = random.Random(data_seed)
    treatments = list(registry.treatments)

    for t in treatments:
        meta = t.generator_metadata
        if not meta:
            raise ValueError(
                f"treatment {t.bundle_id} has empty generator_metadata"
            )
        for factor in _GRAMMAR_FACTORS:
            if factor not in meta:
                raise ValueError(
                    f"treatment {t.bundle_id} is missing grammar factor {factor!r} "
                    f"in generator_metadata"
                )

    train_treatments, held_out_treatments = _deterministic_coverage_split(
        treatments,
        data_seed=data_seed,
        train_treatment_frac=train_treatment_frac,
    )

    train_bundle_ids = {t.bundle_id for t in train_treatments}
    held_out_bundle_ids = {t.bundle_id for t in held_out_treatments}

    # Pre-compute true p_star for every (family, difficulty, treatment, variant).
    true_p: dict[tuple[str, str, str, float], float] = {}

    rows: list[dict[str, Any]] = []
    task_global = 0

    for family in FAMILIES:
        for difficulty, count in zip(DIFFICULTIES, tasks_per_cell):
            # Pre-generate variant values for this cell so splits are
            # deterministic given data_seed only.
            variants = [rng.uniform(*_TASK_VARIANT_RANGE) for _ in range(count)]

            # Reserve one task for each split before applying the requested
            # proportions, so valid extreme fractions cannot crowd out test.
            n_train_cell = min(max(1, int(count * train_frac)), count - 2)
            n_val_cell = min(
                max(1, int(count * val_frac)), count - n_train_cell - 1
            )
            n_test_cell = count - n_train_cell - n_val_cell

            cell_splits: list[str] = []
            cell_splits.extend(["train"] * n_train_cell)
            cell_splits.extend(["validation"] * n_val_cell)
            cell_splits.extend(["test"] * n_test_cell)
            rng.shuffle(cell_splits)

            for i in range(count):
                variant = variants[i]
                split = cell_splits[i]
                task_id = f"synthetic-{family}-{difficulty}-t{task_global}"

                # Pre-compute p_star for all treatments.
                for treatment in treatments:
                    key = (family, difficulty, treatment.bundle_id, variant)
                    if key not in true_p:
                        true_p[key] = treatment_true_p(
                            treatment, family, difficulty,
                            task_variant=variant, signal_scale=signal_scale,
                        )

                # Training and validation splits only contain training treatments.
                if split in ("train", "validation"):
                    panel_treatments = train_treatments
                else:
                    panel_treatments = held_out_treatments

                for treatment in panel_treatments:
                    key = (family, difficulty, treatment.bundle_id, variant)
                    p = true_p[key]
                    success = 1 if rng.random() < p else 0
                    rows.append({
                        "task_id": task_id,
                        "attempt_id": f"{task_id}-{treatment.bundle_id}",
                        "split": split,
                        "family": family,
                        "difficulty": difficulty,
                        "bundle_id": treatment.bundle_id,
                        "true_p": p,
                        "task_variant": variant,
                        "verified_success": bool(success),
                        "treatment_bundle_id": treatment.bundle_id,
                        "treatment_bundle_hash": treatment.bundle_hash,
                        "treatment_registry_hash": registry.registry_hash,
                        "model_input": _model_input(
                            treatment, family, difficulty, i, variant,
                        ),
                    })
                task_global += 1

    return rows, true_p, train_bundle_ids, held_out_bundle_ids


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _validate_build_inputs(
    tasks_per_cell: tuple[int, int, int],
    signal_scale: float,
    train_treatment_frac: float,
    train_frac: float,
    val_frac: float,
) -> None:
    """Reject configurations that cannot support rank evaluation or descriptor
    extrapolation before any rows are generated."""
    if not isinstance(tasks_per_cell, (tuple, list)) or len(tasks_per_cell) != 3:
        raise ValueError(
            f"tasks_per_cell must be a 3-tuple, got {tasks_per_cell!r}"
        )
    for idx, v in enumerate(tasks_per_cell):
        if not isinstance(v, int):
            raise TypeError(
                f"tasks_per_cell[{idx}] must be int, got {type(v).__name__}"
            )
        if v < 3:
            raise ValueError(
                f"tasks_per_cell values must all be >= 3, got {tasks_per_cell}"
            )
    if not isinstance(signal_scale, (int, float)) or not math.isfinite(float(signal_scale)):
        raise ValueError(
            f"signal_scale must be a finite number, got {signal_scale!r}"
        )
    if signal_scale <= 0.0:
        raise ValueError(
            f"signal_scale must be > 0, got {signal_scale}"
        )
    for name, value in (
        ("train_treatment_frac", train_treatment_frac),
        ("train_frac", train_frac),
        ("val_frac", val_frac),
    ):
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(
                f"{name} must be a finite number, got {value!r}"
            )
    for name, value in (("train_frac", train_frac), ("val_frac", val_frac)):
        if not (0.0 < value < 1.0):
            raise ValueError(f"{name} must be in (0, 1), got {value}")
    if train_frac + val_frac >= 1.0:
        raise ValueError(
            f"train_frac + val_frac ({train_frac} + {val_frac}) must be < 1"
        )


def _validate_run_inputs(
    epochs: int,
    batch_size: int,
    patience: int,
    num_samples: int,
) -> None:
    """Reject invalid training hyperparameters before launching."""
    for name, value in (
        ("epochs", epochs),
        ("batch_size", batch_size),
        ("patience", patience),
        ("num_samples", num_samples),
    ):
        if not isinstance(value, int):
            raise TypeError(f"{name} must be int, got {type(value).__name__}")
        if value < 1:
            raise ValueError(f"{name} must be >= 1, got {value}")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Stdlib tie-aware Spearman rank correlation (no SciPy dependency)
# ---------------------------------------------------------------------------


def _rank_values(values: list[float]) -> list[float]:
    """Return fractional (average) ranks for a list of values.

    Ties share the mean rank of their positions; the result is computed
    with the standard library only.
    """
    n = len(values)
    if n == 0:
        return []
    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        v = indexed[i][1]
        j = i
        while j < n and indexed[j][1] == v:
            j += 1
        rank = (i + j + 1) / 2.0  # average rank (1-indexed)
        for k in range(i, j):
            ranks[indexed[k][0]] = rank
        i = j
    return ranks


def spearman_rho(xs: list[float], ys: list[float]) -> float:
    """Deterministic tie-aware Spearman rank correlation using stdlib only.

    Returns ``float('nan')`` when fewer than 3 points or all values tied.
    """
    if len(xs) < 3 or len(ys) < 3:
        return float("nan")
    rx = _rank_values(xs)
    ry = _rank_values(ys)
    n = len(rx)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry))
    den_x = math.sqrt(sum((a - mean_rx) ** 2 for a in rx))
    den_y = math.sqrt(sum((b - mean_ry) ** 2 for b in ry))
    den = den_x * den_y
    if den == 0.0:
        return float("nan")
    return num / den


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _predict_test_rows(
    model: "om.OutcomeModel",
    pre: om.Preprocessor,
    test_rows: list[dict[str, Any]],
    *,
    num_samples: int,
    seed: int,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    out = []
    for row in test_rows:
        pred = om.predict_single(
            model,
            pre,
            row["model_input"],
            num_samples=num_samples,
            seed=seed,
            device=device,
        )
        out.append({
            "task_id": row["task_id"],
            "family": row["family"],
            "difficulty": row["difficulty"],
            "bundle_id": row["bundle_id"],
            "task_variant": row.get("task_variant"),
            "true_p": row["true_p"],
            "verified_success": int(bool(row["verified_success"])),
            "pred_mean": pred["mean"],
            "pred_std": pred["std"],
        })
    return out


def _predeclared_held_out_baseline(held_out_bundle_ids: set[str]) -> str:
    """Deterministic held-out baseline chosen *before* any labels are observed.

    The lexicographically-first held-out bundle_id is used as a static
    comparator so the adaptive allocator can be evaluated against a
    non-random alternative that exists in every test panel.
    """
    if not held_out_bundle_ids:
        raise ValueError("at least one held-out treatment is required")
    return sorted(held_out_bundle_ids)[0]


def _allocator_lift(
    preds: list[dict[str, Any]],
    held_out_bundle_ids: set[str],
) -> dict[str, Any]:
    """Compare model allocator against a predeclared held-out baseline and random.

    Two views are reported:

    * ``realized`` — scored against the noisy single-draw outcome (high
      variance).
    * ``expected`` — scored against the known true ``p_star`` (low-variance
      measure of decision quality).

    The ``predeclared_held_out_baseline`` is a single held-out treatment
    selected deterministically before labels (lexicographically-first
    ``bundle_id``).  Because every test panel contains it, the comparison is
    never degenerate.
    """
    baseline_bundle = _predeclared_held_out_baseline(held_out_bundle_ids)

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in preds:
        by_task[p["task_id"]].append(p)

    n_tasks = len(by_task)
    r_model = r_random = r_oracle = r_baseline = 0.0
    e_model = e_random = e_oracle = e_baseline = 0.0

    for items in by_task.values():
        model_pick = max(items, key=lambda it: it["pred_mean"])
        # Realized (noisy draw).
        r_model += model_pick["verified_success"]
        r_random += sum(it["verified_success"] for it in items) / len(items)
        r_oracle += max(it["verified_success"] for it in items)
        # Predeclared held-out baseline — must be present in every panel.
        bl_items = [it for it in items if it["bundle_id"] == baseline_bundle]
        if not bl_items:
            raise ValueError(
                f"predeclared baseline {baseline_bundle!r} is missing from "
                f"task panel; every test panel must contain it"
            )
        r_baseline += bl_items[0]["verified_success"]
        # Expected (true p_star): the decision-quality view.
        e_model += model_pick["true_p"]
        e_random += sum(it["true_p"] for it in items) / len(items)
        e_oracle += max(it["true_p"] for it in items)
        e_baseline += bl_items[0]["true_p"]

    def _div(num: float) -> float | None:
        return num / n_tasks if n_tasks else None

    exp_model = _div(e_model)
    exp_random = _div(e_random)
    exp_baseline = _div(e_baseline)
    exp_oracle = _div(e_oracle)

    alloc_lift_abs = (exp_model - exp_random) if (exp_model is not None and exp_random is not None) else None
    alloc_lift_over_baseline = (exp_model - exp_baseline) if (exp_model is not None and exp_baseline is not None) else None

    return {
        "n_test_tasks": n_tasks,
        "realized_model": _div(r_model),
        "realized_random": _div(r_random),
        "realized_predeclared_baseline": _div(r_baseline),
        "realized_oracle_hindsight": _div(r_oracle),
        "expected_model_vs_true_pstar": exp_model,
        "expected_random_vs_true_pstar": exp_random,
        "expected_predeclared_baseline_vs_true_pstar": exp_baseline,
        "expected_oracle_vs_true_pstar": exp_oracle,
        "expected_allocator_lift_over_random": alloc_lift_abs,
        "expected_allocator_lift_over_predeclared_baseline": alloc_lift_over_baseline,
        "predeclared_baseline_bundle": baseline_bundle,
    }


def _ranking_recovery(preds: list[dict[str, Any]]) -> dict[str, Any]:
    """Spearman correlation between predicted and true success, aggregated.

    Per (family, bundle_id) mean predictions vs true p_star.
    """
    agg: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"pred": [], "true": [], "succ": []}
    )
    for p in preds:
        key = (p["family"], p["bundle_id"])
        agg[key]["pred"].append(p["pred_mean"])
        agg[key]["true"].append(p["true_p"])
        agg[key]["succ"].append(p["verified_success"])

    pred_means = [sum(v["pred"]) / len(v["pred"]) for v in agg.values()]
    true_means = [sum(v["true"]) / len(v["true"]) for v in agg.values()]
    realized = [sum(v["succ"]) / len(v["succ"]) for v in agg.values()]

    return {
        "n_family_treatment_cells": len(agg),
        "spearman_pred_vs_true_pstar": spearman_rho(pred_means, true_means),
        "spearman_pred_vs_realized": spearman_rho(pred_means, realized),
        "spearman_realized_vs_true_pstar": spearman_rho(realized, true_means),
        "mae_pred_vs_true_pstar": (
            sum(abs(a - b) for a, b in zip(pred_means, true_means)) / len(pred_means)
            if pred_means
            else None
        ),
    }


def _counterfactual_model_input(
    base_input: dict[str, Any],
    treatment: TreatmentSpec,
) -> dict[str, Any]:
    """Build a descriptor-only counterfactual ``model_input`` that preserves
    the neutralized identity fields (``policy_id``, ``policy_version``,
    treatment ``bundle_id``) while replacing the treatment descriptor text,
    numeric budget fields, and categorical tool fields.

    This avoids leaking real treatment identity through
    ``score_treatment_counterfactuals()``, which would override the
    neutralized placeholders with real IDs.
    """
    prompt_text = base_input.get("text", "")
    desc = treatment_model_input_descriptor(treatment, task_text=prompt_text)
    cf = dict(base_input)
    cf["policy_id"] = _IDENTITY_PLACEHOLDER
    cf["policy_version"] = "1"
    cf["treatment"] = {
        "text": desc["text"],
        "max_output_tokens": desc["max_output_tokens"],
        "tool_call_limit": desc["tool_call_limit"],
        "command_timeout_seconds": desc["command_timeout_seconds"],
        "wall_time_limit_seconds": desc["wall_time_limit_seconds"],
        "tool_interface": desc["tool_interface"],
        "allowed_tools_signature": desc["allowed_tools_signature"],
        "bundle_id": _IDENTITY_PLACEHOLDER,
        "policy_id": _IDENTITY_PLACEHOLDER,
        "policy_version": "1",
    }
    return cf


def _score_descriptor_counterfactuals(
    model: "om.OutcomeModel",
    pre: om.Preprocessor,
    base_input: dict[str, Any],
    treatments: list[TreatmentSpec],
    *,
    num_samples: int,
    seed: int,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Score treatment descriptors while keeping identity fields neutralized."""
    scores: list[dict[str, Any]] = []
    for treatment in treatments:
        prediction = om.predict_single(
            model,
            pre,
            _counterfactual_model_input(base_input, treatment),
            num_samples=num_samples,
            seed=seed,
            device=device,
        )
        scores.append(
            {
                "policy_id": treatment.id,
                "bundle_id": treatment.bundle_id,
                "mean": prediction["mean"],
                "std": prediction["std"],
            }
        )
    return scores


def _representative_held_out_ranking(
    model: "om.OutcomeModel",
    pre: om.Preprocessor,
    registry: TreatmentRegistry,
    test_rows: list[dict[str, Any]],
    held_out_bundle_ids: set[str],
    *,
    num_samples: int,
    seed: int,
    device: str = "cpu",
) -> dict[str, Any]:
    """Return a fixed test-task policy ranking for human smoke inspection.

    The task is selected lexicographically before scoring. ``true_p`` is
    included only because this is a synthetic diagnostic; it never enters the
    model input or ranking calculation.
    """
    rows_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in test_rows:
        rows_by_task[row["task_id"]].append(row)
    if not rows_by_task:
        raise ValueError("at least one test task is required for a ranking")

    task_id = min(rows_by_task)
    panel = rows_by_task[task_id]
    task_row = panel[0]
    held_out_treatments = [
        treatment
        for treatment in registry.treatments
        if treatment.bundle_id in held_out_bundle_ids
    ]
    scores_by_bundle = {
        score["bundle_id"]: score
        for score in _score_descriptor_counterfactuals(
            model,
            pre,
            task_row["model_input"],
            held_out_treatments,
            num_samples=num_samples,
            seed=seed,
            device=device,
        )
    }
    true_p_by_bundle = {row["bundle_id"]: row["true_p"] for row in panel}
    ranked = []
    for bundle_id, score in scores_by_bundle.items():
        ranked.append({**score, "true_p": true_p_by_bundle[bundle_id]})
    ranked.sort(key=lambda score: (-float(score["mean"]), str(score["bundle_id"])))
    for rank, score in enumerate(ranked, start=1):
        score["model_rank"] = rank
    for rank, score in enumerate(
        sorted(ranked, key=lambda score: (-float(score["true_p"]), str(score["bundle_id"]))),
        start=1,
    ):
        score["oracle_rank"] = rank
    return {
        "task_id": task_id,
        "family": task_row["family"],
        "difficulty": task_row["difficulty"],
        "task_variant": task_row["task_variant"],
        "ranked": ranked,
    }


def _counterfactual_top1_match(
    model: "om.OutcomeModel",
    pre: om.Preprocessor,
    registry: TreatmentRegistry,
    test_rows: list[dict[str, Any]],
    true_p: dict[tuple[str, str, str, float], float],
    held_out_bundle_ids: set[str],
    *,
    num_samples: int,
    seed: int,
    device: str = "cpu",
    n_probes: int = 12,
    signal_scale: float = 0.75,
) -> dict[str, Any]:
    """For a sample of *unique test tasks*: does the model pick the best
    held-out treatment according to true ``p_star``?

    Each probe task is scored against every held-out treatment via
    descriptor-only counterfactual inputs (see
    :func:`_counterfactual_model_input`), so the model cannot use identity
    leakage to rank treatments.
    """
    if not test_rows:
        return {"n_probes": 0, "top1_match_rate": None}
    rng = random.Random(seed + 7)

    # Sample unique task IDs, not raw rows (one row per treatment per task).
    unique_tasks = sorted({r["task_id"] for r in test_rows})
    if len(unique_tasks) == 0:
        return {"n_probes": 0, "top1_match_rate": None}
    sampled_tasks = rng.sample(
        unique_tasks, min(n_probes, len(unique_tasks)),
    )

    # Map task_id -> one representative row (any treatment row for that task).
    task_row: dict[str, dict[str, Any]] = {}
    for r in test_rows:
        if r["task_id"] not in task_row:
            task_row[r["task_id"]] = r

    # True-best held-out bundle per (family, difficulty, variant).
    best_cache: dict[tuple[str, str, float], str] = {}
    for r in test_rows:
        key = (r["family"], r["difficulty"], r.get("task_variant", 0.0))
        if key in best_cache:
            continue
        best_bundle = None
        best_p = -1.0
        for treatment in registry.treatments:
            if treatment.bundle_id not in held_out_bundle_ids:
                continue
            tkey = (r["family"], r["difficulty"], treatment.bundle_id, key[2])
            p = true_p.get(tkey)
            if p is None:
                p = treatment_true_p(
                    treatment, r["family"], r["difficulty"],
                    task_variant=key[2], signal_scale=signal_scale,
                )
            if p > best_p:
                best_p = p
                best_bundle = treatment.bundle_id
        best_cache[key] = best_bundle

    held_out_treatments = [
        t for t in registry.treatments if t.bundle_id in held_out_bundle_ids
    ]

    matches = 0
    for task_id in sampled_tasks:
        row = task_row[task_id]
        true_best = best_cache.get(
            (row["family"], row["difficulty"], row.get("task_variant", 0.0))
        )

        cfs = _score_descriptor_counterfactuals(
            model,
            pre,
            row["model_input"],
            held_out_treatments,
            num_samples=num_samples,
            seed=seed,
            device=device,
        )
        top = max(cfs, key=lambda c: c["mean"])
        if true_best and top["bundle_id"] == true_best:
            matches += 1

    return {
        "n_probes": len(sampled_tasks),
        "top1_match_rate": matches / len(sampled_tasks) if sampled_tasks else None,
        "held_out_bundle_count": len(held_out_bundle_ids),
    }


def _verify_identity_neutralization(
    test_rows: list[dict[str, Any]],
    train_bundle_ids: set[str],
    held_out_bundle_ids: set[str],
) -> dict[str, Any]:
    """Verify structural identity-neutralization integrity.

    Because model-input identity categoricals are neutralized to a constant
    placeholder (see :data:`_IDENTITY_PLACEHOLDER`), no preprocessor UNK
    test is meaningful — the placeholder is always in vocabulary.  Instead
    this function verifies the *structural* facts:

    * Test rows use held-out treatment ``bundle_id`` values that never
      appear in training rows.
    * Every test-row ``model_input`` consistently carries the neutralized
      placeholders for ``policy_id`` and treatment ``bundle_id``.
    """
    test_bundle_ids = {r["bundle_id"] for r in test_rows}
    disjoint = len(test_bundle_ids & train_bundle_ids) == 0
    all_in_held_out = test_bundle_ids.issubset(held_out_bundle_ids)
    placeholder_ok = True
    for row in test_rows[:50]:
        mi = row["model_input"]
        if mi.get("policy_id") != _IDENTITY_PLACEHOLDER:
            placeholder_ok = False
            break
        if mi.get("treatment", {}).get("bundle_id") != _IDENTITY_PLACEHOLDER:
            placeholder_ok = False
            break
    return {
        "structural_test_disjoint_from_train": disjoint,
        "all_test_bundle_ids_in_held_out_set": all_in_held_out,
        "placeholder_neutralized_in_model_input": placeholder_ok,
        "n_test_bundle_ids": len(test_bundle_ids),
        "n_train_bundle_ids": len(train_bundle_ids),
        "n_held_out_bundle_ids": len(held_out_bundle_ids),
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_learn_smoke(
    registry: TreatmentRegistry,
    output_dir: str | Path,
    *,
    tasks_per_cell: tuple[int, int, int] = (20, 16, 12),
    signal_scale: float = 0.75,
    data_seed: int = 2024,
    train_treatment_frac: float = 0.7,
    train_seed: int = 42,
    epochs: int = 80,
    batch_size: int = 32,
    patience: int = 10,
    num_samples: int = 64,
    verbose: bool = False,
) -> dict[str, Any]:
    """Generate treatment-held-out data, train model, evaluate generalization.

    Returns a structured result dict with verdict fields suitable for
    automated smoke testing.
    """
    if not om.TORCH_AVAILABLE:
        raise RuntimeError("learn-smoke requires PyTorch")

    _validate_run_inputs(epochs, batch_size, patience, num_samples)

    root = Path(output_dir).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"output directory must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)

    rows, true_p, train_bundle_ids, held_out_bundle_ids = build_treatment_held_out_rows(
        registry,
        tasks_per_cell=tasks_per_cell,
        signal_scale=signal_scale,
        data_seed=data_seed,
        train_treatment_frac=train_treatment_frac,
    )
    dataset_path = root / "learn-smoke-dataset.jsonl"
    artifact_dir = root / "model"
    registry_path = root / "treatments.json"
    _write_jsonl(dataset_path, rows)
    registry.save(registry_path)

    # p_star distribution summary.
    all_p = [r["true_p"] for r in rows]
    split_counts = {
        s: sum(1 for r in rows if r["split"] == s)
        for s in ("train", "validation", "test")
    }

    # Validate split integrity.
    if split_counts["train"] == 0:
        raise ValueError("no training rows — reduce train_frac or increase panel size")
    if split_counts["test"] == 0:
        raise ValueError("no test rows — increase tasks_per_cell")

    training = om.train_model(
        dataset_path,
        artifact_dir,
        epochs=epochs,
        batch_size=batch_size,
        seed=train_seed,
        max_vocab=512,
        max_tokens=64,
        text_dim=24,
        cat_dim=12,
        numeric_hidden=16,
        fusion_hidden=48,
        dropout=0.1,
        lr=1.5e-3,
        patience=patience,
        num_samples=num_samples,
        verbose=verbose,
    )

    _config, pre, model = om.load_artifacts(artifact_dir, device="cpu")
    model.eval()
    test_rows = [r for r in rows if r["split"] == "test"]
    train_rows = [r for r in rows if r["split"] == "train"]

    identity_check = _verify_identity_neutralization(
        test_rows, train_bundle_ids, held_out_bundle_ids,
    )

    # Naive baseline: global train success rate.
    train_rate = (
        sum(1 for r in train_rows if r["verified_success"]) / max(1, len(train_rows))
    )

    preds = _predict_test_rows(
        model, pre, test_rows,
        num_samples=num_samples, seed=train_seed + 500,
    )
    y_true = [p["verified_success"] for p in preds]
    y_pred = [p["pred_mean"] for p in preds]
    y_std = [p["pred_std"] for p in preds]

    model_metrics = om.compute_metrics(y_true, y_pred, posterior_std=y_std)
    naive_metrics = om.compute_metrics(
        y_true, [train_rate] * len(y_true), posterior_std=None,
    )

    alloc = _allocator_lift(preds, held_out_bundle_ids)
    ranking = _ranking_recovery(preds)
    cf = _counterfactual_top1_match(
        model, pre, registry, test_rows, true_p,
        held_out_bundle_ids,
        num_samples=num_samples,
        seed=train_seed + 900,
        signal_scale=signal_scale,
    )
    representative_ranking = _representative_held_out_ranking(
        model,
        pre,
        registry,
        test_rows,
        held_out_bundle_ids,
        num_samples=num_samples,
        seed=train_seed + 901,
    )

    # -- verdict heuristics (advisory, not research claims) --
    alloc_lift_abs = alloc["expected_allocator_lift_over_random"]
    alloc_lift_baseline = alloc["expected_allocator_lift_over_predeclared_baseline"]

    verdict = {
        "predicts_better_than_naive_brier": (
            model_metrics["brier"] is not None
            and naive_metrics["brier"] is not None
            and model_metrics["brier"] < naive_metrics["brier"]
        ),
        "expected_allocator_beats_random": (
            alloc_lift_abs is not None and alloc_lift_abs > 0.01
        ),
        "expected_allocator_beats_predeclared_baseline": (
            alloc_lift_baseline is not None and alloc_lift_baseline > 0.0
        ),
        "ranking_spearman_positive": (
            ranking["spearman_pred_vs_true_pstar"] is not None
            and not math.isnan(ranking["spearman_pred_vs_true_pstar"])
            and ranking["spearman_pred_vs_true_pstar"] > 0.3
        ),
        "test_treatments_are_held_out": identity_check["structural_test_disjoint_from_train"],
        "expected_allocator_lift_over_random": alloc_lift_abs,
        "expected_allocator_lift_over_predeclared_baseline": alloc_lift_baseline,
    }
    verdict["descriptor_learned_something_usable"] = (
        verdict["predicts_better_than_naive_brier"]
        and verdict["expected_allocator_beats_random"]
        and verdict["ranking_spearman_positive"]
        and verdict["test_treatments_are_held_out"]
    )

    return {
        "synthetic_only": True,
        "warning": (
            "Labels are sampled from a noisy synthetic data-generating process. "
            "This probes the model's descriptor-learning mechanism, NOT any real "
            "agent or task family. Never merge into an experiment dataset."
        ),
        "config": {
            "registry_hash": registry.registry_hash,
            "n_treatments": len(registry.treatments),
            "n_train_treatments": len(train_bundle_ids),
            "n_held_out_treatments": len(held_out_bundle_ids),
            "tasks_per_cell": list(tasks_per_cell),
            "signal_scale": signal_scale,
            "data_seed": data_seed,
            "train_treatment_frac": train_treatment_frac,
            "train_seed": train_seed,
            "epochs": epochs,
            "identity_placeholder": _IDENTITY_PLACEHOLDER,
        },
        "data": {
            "total_rows": len(rows),
            "split_counts": split_counts,
            "p_star_min": min(all_p),
            "p_star_max": max(all_p),
            "p_star_mean": sum(all_p) / len(all_p),
            "train_success_rate": train_rate,
            "train_treatment_bundle_ids": sorted(train_bundle_ids),
            "held_out_treatment_bundle_ids": sorted(held_out_bundle_ids),
        },
        "identity_neutralization": identity_check,
        "training": {
            key: training["training"][key]
            for key in ("best_epoch", "epochs_run", "train_rows", "validation_rows", "test_rows")
        },
        "test_metrics_model": {
            k: model_metrics[k]
            for k in ("n", "log_loss", "brier", "accuracy_05", "ece",
                       "average_precision", "mean_posterior_std")
        },
        "test_metrics_naive_baseline": {
            k: naive_metrics[k]
            for k in ("n", "log_loss", "brier", "accuracy_05", "ece")
        },
        "ranking_recovery": ranking,
        "allocator_lift": alloc,
        "counterfactual_top1": cf,
        "representative_held_out_ranking": representative_ranking,
        "verdict": verdict,
        "output_dir": str(root),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyreplab-outcome-model-learn-smoke",
        description="Treatment-held-out descriptor-learning smoke for the Bayesian outcome model.",
    )
    parser.add_argument(
        "registry",
        help="path to a treatment registry JSON (e.g. from treatments generate)",
    )
    parser.add_argument(
        "--output-dir",
        default=".runs/learn-smoke",
        help="empty directory for synthetic dataset + model artifacts",
    )
    parser.add_argument("--signal-scale", type=float, default=0.75)
    parser.add_argument("--data-seed", type=int, default=2024)
    parser.add_argument("--train-treatment-frac", type=float, default=0.7)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    registry = TreatmentRegistry.load(args.registry)
    try:
        result = run_learn_smoke(
            registry,
            args.output_dir,
            signal_scale=args.signal_scale,
            data_seed=args.data_seed,
            train_treatment_frac=args.train_treatment_frac,
            train_seed=args.train_seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            num_samples=args.num_samples,
            verbose=args.verbose,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["verdict"]["descriptor_learned_something_usable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
