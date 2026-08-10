"""Noisy synthetic learn-smoke for the descriptor-aware outcome model.

Purpose
-------
Unlike :mod:`outcome_model_smoke` (whose labels are deterministic and therefore
only exercise plumbing), this probe builds a *noisy, graded* data-generating
process over a **generated** treatment registry and asks a genuine question:

    Can the variational Bayesian outcome model recover a non-trivial
    policy->success mapping from noisy Bernoulli outcomes, produce calibrated
    predictions, and yield a usable per-task allocator?

It is deliberately self-contained and synthetic.  Its data MUST NEVER be mixed
into a real experiment dataset, and its scores are research-sandBox evidence
about the *learning mechanism*, not about any real agent or task family.

Data-generating process
-----------------------
Each treatment in the supplied registry carries grammar factors
(planning / verification / execution / budget).  Each synthetic task family has
its own latent affinity for each factor level, so the optimal treatment differs
by family (a non-trivial allocation problem).  For every (task, treatment) cell
we compute a true latent success probability ``p_star`` in ``[0, 1]`` and sample
``verified_success ~ Bernoulli(p_star)``.  The signal is therefore partial: the
same treatment on the same family can succeed or fail across seeds.

CLI
---
``python -m pyreplab_harness.outcome_model_learn_smoke REGISTRY.json [options]``
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import tempfile
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
# Families intentionally mirror the real gym names so the categorical schema is
# identical, but the tasks and labels here are entirely synthetic.
# ---------------------------------------------------------------------------

FAMILIES = ("artifact", "sqlite", "python_repair", "shell")
DIFFICULTIES = ("easy", "medium", "hard")

# Per-family logit affinity for each grammar factor level.
# python_repair rewards careful/decompose/retry/generous policies.
# sqlite rewards deliberate/incremental/retry with moderate budget.
# shell rewards direct/single-pass/generous (lots of file commands, low reasoning).
# artifact rewards deliberate/retry/moderate.
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


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def treatment_logit(
    treatment: TreatmentSpec,
    family: str,
    difficulty: str,
    *,
    signal_scale: float = 0.75,
) -> float:
    """Latent logit success for a (treatment, family, difficulty) cell."""
    meta = treatment.generator_metadata
    aff = _FAMILY_AFFINITIES[family]
    raw = (
        aff["planning"][str(meta.get("planning"))]
        + aff["verification"][str(meta.get("verification"))]
        + aff["execution"][str(meta.get("execution"))]
        + aff["budget"][str(meta.get("budget"))]
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
    signal_scale: float = 0.75,
) -> float:
    """True latent Bernoulli success probability for a cell."""
    return _sigmoid(
        treatment_logit(treatment, family, difficulty, signal_scale=signal_scale)
    )


# ---------------------------------------------------------------------------
# Synthetic model_input construction
# ---------------------------------------------------------------------------


def _task_prompt(family: str, difficulty: str, seed: int) -> str:
    return (
        f"Synthetic {family} task ({difficulty}, seed={seed}). "
        f"Produce the required {family} artifact following the stated contract. "
        f"Difficulty band: {difficulty}."
    )


def _model_input(
    treatment: TreatmentSpec,
    family: str,
    difficulty: str,
    seed: int,
) -> dict[str, Any]:
    prompt = _task_prompt(family, difficulty, seed)
    return {
        "text": prompt,
        "family": family,
        "template_id": f"{family}-{difficulty}-v1",
        "difficulty": difficulty,
        "public_metadata": {
            "seed": float(seed),
            "difficulty_index": float(DIFFICULTIES.index(difficulty)),
        },
        "policy_id": treatment.id,
        "policy_version": treatment.version,
        "treatment": treatment_model_input_descriptor(treatment),
    }


# ---------------------------------------------------------------------------
# Panel + dataset construction
# ---------------------------------------------------------------------------


def build_noisy_rows(
    registry: TreatmentRegistry,
    *,
    tasks_per_cell: tuple[int, int, int] = (12, 12, 8),
    signal_scale: float = 0.75,
    seed: int = 2024,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], float]]:
    """Build complete noisy panels plus the true p_star lookup.

    Every task receives all registry treatments (a complete panel), so the
    allocator evaluation population is well-defined.  Tasks are split by a
    seeded shuffle into train/validation/test; tasks never span splits.
    """
    rng = random.Random(seed)
    treatments = list(registry.treatments)
    rows: list[dict[str, Any]] = []
    true_p: dict[tuple[str, str, str], float] = {}

    # Pre-compute true p_star per (family, difficulty, bundle_id).
    for family in FAMILIES:
        for difficulty, _count in zip(DIFFICULTIES, tasks_per_cell):
            for treatment in treatments:
                true_p[(family, difficulty, treatment.bundle_id)] = (
                    treatment_true_p(
                        treatment, family, difficulty, signal_scale=signal_scale
                    )
                )

    # Enumerate tasks per (family, difficulty) and assign splits by task.
    task_global = 0
    for family in FAMILIES:
        for difficulty, count in zip(DIFFICULTIES, tasks_per_cell):
            for i in range(count):
                task_id = f"synthetic-{family}-{difficulty}-t{task_global}"
                # Deterministic per-task split via the seeded rng stream.
                u = rng.random()
                if u < train_frac:
                    split = "train"
                elif u < train_frac + val_frac:
                    split = "validation"
                else:
                    split = "test"
                task_global += 1
                for treatment in treatments:
                    p = true_p[(family, difficulty, treatment.bundle_id)]
                    success = 1 if rng.random() < p else 0
                    rows.append(
                        {
                            "task_id": task_id,
                            "attempt_id": f"{task_id}-{treatment.bundle_id}",
                            "split": split,
                            "family": family,
                            "difficulty": difficulty,
                            "bundle_id": treatment.bundle_id,
                            "true_p": p,
                            "verified_success": bool(success),
                            "treatment_bundle_id": treatment.bundle_id,
                            "treatment_bundle_hash": treatment.bundle_hash,
                            "treatment_registry_hash": registry.registry_hash,
                            "model_input": _model_input(
                                treatment, family, difficulty, i
                            ),
                        }
                    )
    return rows, true_p


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation; returns nan when undefined."""
    try:
        from scipy.stats import spearmanr  # type: ignore
    except Exception:  # pragma: no cover - scipy is a pinned transitive dep
        return float("nan")
    if len(xs) < 3:
        return float("nan")
    corr, _p = spearmanr(xs, ys)
    return float(corr)


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
        out.append(
            {
                "task_id": row["task_id"],
                "family": row["family"],
                "difficulty": row["difficulty"],
                "bundle_id": row["bundle_id"],
                "true_p": row["true_p"],
                "verified_success": int(bool(row["verified_success"])),
                "pred_mean": pred["mean"],
                "pred_std": pred["std"],
            }
        )
    return out


def _allocator_lift(preds: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare the model allocator against random / fixed-best / oracle on test.

    Two complementary views are reported:

    * ``realized`` — scored against the *noisy single draw* outcome, as a real
      held-out evaluation would see it (high variance on small panels).
    * ``expected`` — scored against the known true ``p_star`` of the picked
      treatment.  Because this is synthetic data, ``p_star`` is ground truth, so
      the expected view is the low-variance measure of *decision quality*.
    """
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in preds:
        by_task[p["task_id"]].append(p)

    n_tasks = len(by_task)
    r_model = r_random = r_oracle = 0.0
    e_model = e_random = e_oracle = 0.0
    # Fixed-best-global: treatment with highest mean *realized* success on the
    # full prediction set (an optimistic static rule that sees test outcomes).
    by_bundle_success: dict[str, list[int]] = defaultdict(list)
    for p in preds:
        by_bundle_success[p["bundle_id"]].append(p["verified_success"])
    fixed_best_bundle = max(
        by_bundle_success,
        key=lambda b: sum(by_bundle_success[b]) / len(by_bundle_success[b]),
    )

    for items in by_task.values():
        model_pick = max(items, key=lambda it: it["pred_mean"])
        # Realized (noisy draw).
        r_model += model_pick["verified_success"]
        r_random += sum(it["verified_success"] for it in items) / len(items)
        r_oracle += max(it["verified_success"] for it in items)
        # Expected (true p_star): the decision-quality view.
        e_model += model_pick["true_p"]
        e_random += sum(it["true_p"] for it in items) / len(items)
        e_oracle += max(it["true_p"] for it in items)

    def _div(num: float) -> float | None:
        return num / n_tasks if n_tasks else None

    return {
        "n_test_tasks": n_tasks,
        "realized_model": _div(r_model),
        "realized_random": _div(r_random),
        "realized_oracle_hindsight": _div(r_oracle),
        "expected_model_vs_true_pstar": _div(e_model),
        "expected_random_vs_true_pstar": _div(e_random),
        "expected_oracle_vs_true_pstar": _div(e_oracle),
        "fixed_best_global_bundle": fixed_best_bundle,
        "fixed_best_global_realized": (
            sum(p["verified_success"] for p in preds if p["bundle_id"] == fixed_best_bundle)
            / sum(1 for p in preds if p["bundle_id"] == fixed_best_bundle)
        ),
    }


def _ranking_recovery(preds: list[dict[str, Any]]) -> dict[str, Any]:
    """Spearman correlation between predicted and true success, aggregated."""
    # Per (family, bundle_id): mean predicted vs true p_star.
    agg: dict[tuple[str, str], list[float]] = defaultdict(
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
        "spearman_pred_vs_true_pstar": _spearman(pred_means, true_means),
        "spearman_pred_vs_realized": _spearman(pred_means, realized),
        "spearman_realized_vs_true_pstar": _spearman(realized, true_means),
        "mae_pred_vs_true_pstar": (
            sum(abs(a - b) for a, b in zip(pred_means, true_means)) / len(pred_means)
            if pred_means
            else None
        ),
    }


def _counterfactual_top1_match(
    model: "om.OutcomeModel",
    pre: om.Preprocessor,
    registry: TreatmentRegistry,
    test_rows: list[dict[str, Any]],
    true_p: dict[tuple[str, str, str], float],
    *,
    num_samples: int,
    seed: int,
    device: str = "cpu",
    n_probes: int = 12,
) -> dict[str, Any]:
    """For a sample of test tasks, does the model's top-1 treatment equal the
    family's true-best treatment?"""
    if not test_rows:
        return {"n_probes": 0, "top1_match_rate": None}
    rng = random.Random(seed + 7)
    sample = rng.sample(test_rows, min(n_probes, len(test_rows)))
    matches = 0
    family_best: dict[str, str] = {}
    # True-best bundle per family (over registry treatments, averaged over difficulties).
    for family in FAMILIES:
        best_bundle = None
        best_p = -1.0
        for treatment in registry.treatments:
            ps = [
                true_p[(family, d, treatment.bundle_id)] for d in DIFFICULTIES
            ]
            mp = sum(ps) / len(ps)
            if mp > best_p:
                best_p = mp
                best_bundle = treatment.bundle_id
        family_best[family] = best_bundle

    for row in sample:
        family = row["family"]
        cfs = om.score_treatment_counterfactuals(
            model,
            pre,
            row["model_input"],
            list(registry.treatments),
            num_samples=num_samples,
            seed=seed,
            device=device,
        )
        top = max(cfs, key=lambda c: c["mean"])
        if top["bundle_id"] == family_best[family]:
            matches += 1
    return {
        "n_probes": len(sample),
        "top1_match_rate": matches / len(sample),
        "family_true_best_bundle": dict(family_best),
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_learn_smoke(
    registry: TreatmentRegistry,
    output_dir: str | Path,
    *,
    tasks_per_cell: tuple[int, int, int] = (20, 20, 16),
    signal_scale: float = 0.75,
    data_seed: int = 2024,
    train_seed: int = 42,
    epochs: int = 80,
    batch_size: int = 32,
    patience: int = 10,
    num_samples: int = 64,
    verbose: bool = False,
) -> dict[str, Any]:
    """Generate noisy data, train the model, and evaluate learning quality."""
    if not om.TORCH_AVAILABLE:
        raise RuntimeError("learn-smoke requires PyTorch")

    root = Path(output_dir).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"output directory must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)

    rows, true_p = build_noisy_rows(
        registry,
        tasks_per_cell=tasks_per_cell,
        signal_scale=signal_scale,
        seed=data_seed,
    )
    dataset_path = root / "learn-smoke-dataset.jsonl"
    artifact_dir = root / "model"
    registry_path = root / "treatments.json"
    _write_jsonl(dataset_path, rows)
    registry.save(registry_path)

    # p_star distribution summary (so the noise level is auditable).
    all_p = [r["true_p"] for r in rows]
    split_counts = {
        s: sum(1 for r in rows if r["split"] == s) for s in ("train", "validation", "test")
    }

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
    test_rows = [r for r in rows if r["split"] == "test"]

    # Naive baseline: predict global train success rate for every test row.
    train_rate = sum(1 for r in rows if r["split"] == "train" and r["verified_success"]) / max(
        1, split_counts["train"]
    )

    preds = _predict_test_rows(
        model, pre, test_rows, num_samples=num_samples, seed=train_seed + 500
    )
    y_true = [p["verified_success"] for p in preds]
    y_pred = [p["pred_mean"] for p in preds]
    y_std = [p["pred_std"] for p in preds]

    model_metrics = om.compute_metrics(y_true, y_pred, posterior_std=y_std)
    naive_metrics = om.compute_metrics(
        y_true, [train_rate] * len(y_true), posterior_std=None
    )

    alloc = _allocator_lift(preds)
    ranking = _ranking_recovery(preds)
    cf = _counterfactual_top1_match(
        model,
        pre,
        registry,
        test_rows,
        true_p,
        num_samples=num_samples,
        seed=train_seed + 900,
    )

    # Verdict heuristics (advisory, not research claims).
    # Primary "usable" test: the expected allocator (scored against known true
    # p_star) must beat the random allocator by a non-trivial margin, and the
    # model must predict better than the base-rate baseline.
    exp_model = alloc["expected_model_vs_true_pstar"]
    exp_random = alloc["expected_random_vs_true_pstar"]
    exp_oracle = alloc["expected_oracle_vs_true_pstar"]
    alloc_lift_abs = (exp_model - exp_random) if (exp_model is not None and exp_random is not None) else None
    oracle_gap = (exp_oracle - exp_model) if (exp_model is not None and exp_oracle is not None) else None
    verdict = {
        "predicts_better_than_naive_brier": (
            model_metrics["brier"] is not None
            and naive_metrics["brier"] is not None
            and model_metrics["brier"] < naive_metrics["brier"]
        ),
        "expected_allocator_beats_random": (
            alloc_lift_abs is not None and alloc_lift_abs > 0.02
        ),
        "ranking_spearman_positive": (
            ranking["spearman_pred_vs_true_pstar"] is not None
            and not math.isnan(ranking["spearman_pred_vs_true_pstar"])
            and ranking["spearman_pred_vs_true_pstar"] > 0.4
        ),
        "expected_allocator_lift_abs": alloc_lift_abs,
        "expected_allocator_oracle_gap": oracle_gap,
    }
    verdict["learned_something_usable"] = (
        verdict["predicts_better_than_naive_brier"]
        and verdict["expected_allocator_beats_random"]
        and verdict["ranking_spearman_positive"]
    )

    return {
        "synthetic_only": True,
        "warning": (
            "Labels are sampled from a noisy synthetic data-generating process. "
            "This probes the model's learning mechanism, NOT any real agent or "
            "task family. Never merge into an experiment dataset."
        ),
        "config": {
            "registry_hash": registry.registry_hash,
            "n_treatments": len(registry.treatments),
            "tasks_per_cell": list(tasks_per_cell),
            "signal_scale": signal_scale,
            "data_seed": data_seed,
            "train_seed": train_seed,
            "epochs": epochs,
        },
        "data": {
            "total_rows": len(rows),
            "split_counts": split_counts,
            "p_star_min": min(all_p),
            "p_star_max": max(all_p),
            "p_star_mean": sum(all_p) / len(all_p),
            "train_success_rate": train_rate,
        },
        "training": {
            key: training["training"][key]
            for key in ("best_epoch", "epochs_run", "train_rows", "validation_rows", "test_rows")
        },
        "test_metrics_model": {
            k: model_metrics[k]
            for k in ("n", "log_loss", "brier", "accuracy_05", "ece", "average_precision", "mean_posterior_std")
        },
        "test_metrics_naive_baseline": {
            k: naive_metrics[k] for k in ("n", "log_loss", "brier", "accuracy_05", "ece")
        },
        "ranking_recovery": ranking,
        "allocator_lift": alloc,
        "counterfactual_top1": cf,
        "verdict": verdict,
        "output_dir": str(root),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyreplab-outcome-model-learn-smoke",
        description="Noisy synthetic learn-smoke over a generated treatment registry.",
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
    return 0 if result["verdict"]["learned_something_usable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
