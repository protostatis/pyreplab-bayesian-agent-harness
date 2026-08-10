"""Success-cost frontier evaluator for meta-policy allocation.

Evaluates allocations against the success-cost frontier defined in the
M3 preregistration (Section 11.2). Selection uses predictions only; scoring
uses observed complete-panel outcomes.

The lambda grid is frozen before final evaluation; lambda=0 is included for
pure-success selection.
"""

from __future__ import annotations

import math
import random
from typing import Any


def compute_frontier(
    task_predictions: list[list[dict[str, Any]]],
    task_outcomes: list[list[dict[str, Any]]],
    lambda_grid: list[float],
) -> dict[str, Any]:
    """Compute success-cost frontier metrics for a frozen lambda grid.

    Args:
        task_predictions: task_predictions[i][j] is a dict with keys
            "success_prob", "cost_mean" for task i, policy j.
        task_outcomes: task_outcomes[i][j] is a dict with keys
            "verified_success" (0|1), "cost" (float) for task i, policy j.
        lambda_grid: list of lambda values for the success-cost tradeoff.
            lambda=0 means pure-success selection.

    Returns:
        dict with frontier metrics.
    """
    if not task_predictions or not task_outcomes:
        raise ValueError("empty task_predictions or task_outcomes")
    if len(task_predictions) != len(task_outcomes):
        raise ValueError(
            f"mismatched lengths: {len(task_predictions)} predictions vs "
            f"{len(task_outcomes)} outcomes"
        )
    if not lambda_grid:
        raise ValueError("lambda_grid must not be empty")
    if 0.0 not in lambda_grid:
        raise ValueError("lambda_grid must include 0.0")
    if any(not math.isfinite(float(lam)) or float(lam) < 0.0 for lam in lambda_grid):
        raise ValueError("lambda_grid values must be finite and nonnegative")

    n_tasks = len(task_predictions)
    n_policies = len(task_predictions[0]) if n_tasks > 0 else 0
    if n_policies == 0:
        raise ValueError("task panels must contain at least one policy")

    for i in range(n_tasks):
        if len(task_predictions[i]) != n_policies:
            raise ValueError(f"task {i}: mismatched policy count in predictions")
        if len(task_outcomes[i]) != n_policies:
            raise ValueError(f"task {i}: mismatched policy count in outcomes")

    # Build observed data structures.
    observed_success: list[list[float]] = []
    observed_cost: list[list[float]] = []
    for i in range(n_tasks):
        obs_s: list[float] = []
        obs_c: list[float] = []
        for j in range(n_policies):
            prediction = task_predictions[i][j]
            outcome = task_outcomes[i][j]
            if "success_prob" not in prediction or "cost_mean" not in prediction:
                raise ValueError(f"task {i}, policy {j}: incomplete prediction")
            if "verified_success" not in outcome or "cost" not in outcome:
                raise ValueError(f"task {i}, policy {j}: incomplete observed outcome")

            success_prob = float(prediction["success_prob"])
            predicted_cost = float(prediction["cost_mean"])
            observed_cost_value = float(outcome["cost"])
            if not math.isfinite(success_prob) or not 0.0 <= success_prob <= 1.0:
                raise ValueError(
                    f"task {i}, policy {j}: success_prob must be finite and in [0, 1]"
                )
            if not math.isfinite(predicted_cost) or predicted_cost < 0.0:
                raise ValueError(
                    f"task {i}, policy {j}: cost_mean must be finite and nonnegative"
                )
            if not math.isfinite(observed_cost_value) or observed_cost_value < 0.0:
                raise ValueError(
                    f"task {i}, policy {j}: observed cost must be finite and nonnegative"
                )

            observed_success_value = outcome["verified_success"]
            if observed_success_value not in (False, True, 0, 1):
                raise ValueError(
                    f"task {i}, policy {j}: verified_success must be boolean or binary"
                )
            obs_s.append(1.0 if observed_success_value else 0.0)
            obs_c.append(observed_cost_value)
        observed_success.append(obs_s)
        observed_cost.append(obs_c)

    # Per-lambda metrics.
    lambda_results: list[dict[str, Any]] = []
    for lam in lambda_grid:
        # Select argmax(predicted_success - lambda * predicted_cost) per task.
        selected_indices: list[int] = []
        for i in range(n_tasks):
            scores = [
                task_predictions[i][j]["success_prob"]
                - lam * task_predictions[i][j]["cost_mean"]
                for j in range(n_policies)
            ]
            selected_indices.append(
                max(
                    range(n_policies),
                    key=lambda j, s=scores, preds=task_predictions[i]: (
                        s[j], -float(preds[j]["cost_mean"]), -j,
                    ),
                )
            )

        # Score against observed outcomes.
        success_obs = [observed_success[i][selected_indices[i]] for i in range(n_tasks)]
        cost_obs = [observed_cost[i][selected_indices[i]] for i in range(n_tasks)]

        lambda_results.append({
            "lambda": lam,
            "selected_success": sum(success_obs) / n_tasks if n_tasks else 0.0,
            "mean_cost": sum(cost_obs) / n_tasks if n_tasks else 0.0,
            "selected_indices": selected_indices,
        })

    # Frontier area: area under observed success vs log(cost) curve.
    max_observed_cost = max(max(task_costs) for task_costs in observed_cost)
    frontier_area = _compute_frontier_area(
        observed_success,
        observed_cost,
        selected_by_lambda=lambda_results,
        max_cost=max_observed_cost,
    )

    # Pure-success (lambda=0) result.
    lambda0 = next(r for r in lambda_results if r["lambda"] == 0.0)

    # Oracle frontier (best possible per task).
    oracle = _compute_oracle_frontier(
        observed_success, observed_cost, lambda_grid, max_cost=max_observed_cost,
    )

    return {
        "n_tasks": n_tasks,
        "n_policies": n_policies,
        "lambda_grid": lambda_grid,
        "lambda_results": lambda_results,
        "frontier_points": _operating_frontier_points(lambda_results),
        "frontier_area": frontier_area,
        "pure_success_rate": lambda0["selected_success"],
        "pure_success_mean_cost": lambda0["mean_cost"],
        "oracle_frontier_area": oracle["area"],
        "oracle_pure_success_rate": oracle["pure_success"],
        "regret_vs_oracle": oracle["pure_success"] - lambda0["selected_success"],
    }


def _compute_frontier_area(
    observed_success: list[list[float]],
    observed_cost: list[list[float]],
    selected_by_lambda: list[dict[str, Any]],
    *,
    max_cost: float | None = None,
) -> float:
    """Compute area under the success-vs-log(1+cost) frontier curve.

    Only aggregate allocator operating points selected by the frozen lambda
    grid belong on this frontier. Individual task-policy cells are not
    comparable to aggregate operating points and are used only to establish a
    common observed-cost upper bound.
    """
    if not selected_by_lambda:
        return 0.0

    nondominated = _operating_frontier_points(selected_by_lambda)
    if not nondominated:
        return 0.0

    if max_cost is None:
        max_cost = max(
            (cost for task_costs in observed_cost for cost in task_costs),
            default=max(point[0] for point in nondominated),
        )
    max_cost = max(float(max_cost), nondominated[-1][0])

    # Integrate the piecewise-linear frontier and extend its last attained
    # success horizontally to the common observed-cost upper bound.
    area = 0.0
    for idx in range(1, len(nondominated)):
        x0 = math.log1p(nondominated[idx - 1][0])
        x1 = math.log1p(nondominated[idx][0])
        y0 = nondominated[idx - 1][1]
        y1 = nondominated[idx][1]
        area += (y0 + y1) * (x1 - x0) / 2.0
    last_cost, last_success = nondominated[-1]
    area += last_success * (math.log1p(max_cost) - math.log1p(last_cost))
    return area


def _operating_frontier_points(
    selected_by_lambda: list[dict[str, Any]],
) -> list[tuple[float, float]]:
    """Return sorted nondominated aggregate allocator operating points."""
    points = [
        (float(result["mean_cost"]), float(result["selected_success"]))
        for result in selected_by_lambda
    ]
    return sorted(_pareto_frontier(points), key=lambda point: point[0])


def _pareto_frontier(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Return the nondominated (Pareto) frontier: maximize success, minimize cost.

    Points are (cost, success) tuples.  A point (c1, s1) dominates (c2, s2)
    if s1 >= s2 AND c1 <= c2, with at least one strict; that is, equal or
    better success at equal or lower cost.
    """
    if not points:
        return []
    points = sorted(set(points))
    dominated = [False] * len(points)
    for i in range(len(points)):
        ci, si = points[i]  # (cost, success)
        for j in range(len(points)):
            if i == j or dominated[j]:
                continue
            cj, sj = points[j]
            if si >= sj and ci <= cj and (si > sj or ci < cj):
                dominated[j] = True
    return [p for i, p in enumerate(points) if not dominated[i]]


def _compute_oracle_frontier(
    observed_success: list[list[float]],
    observed_cost: list[list[float]],
    lambda_grid: list[float],
    *,
    max_cost: float,
) -> dict[str, Any]:
    """Compute the oracle (upper-bound) frontier using realized outcomes.

    The oracle knows all observed outcomes and can select optimally.
    """
    n_tasks = len(observed_success)
    if n_tasks == 0:
        return {"area": 0.0, "pure_success": 0.0}

    lambda_results: list[dict[str, Any]] = []
    for lam in lambda_grid:
        selected_indices = [
            max(
                range(len(observed_success[i])),
                key=lambda j, task=i: (
                    observed_success[task][j] - lam * observed_cost[task][j],
                    -observed_cost[task][j],
                    -j,
                ),
            )
            for i in range(n_tasks)
        ]
        lambda_results.append({
            "lambda": lam,
            "selected_success": sum(
                observed_success[i][selected_indices[i]] for i in range(n_tasks)
            ) / n_tasks,
            "mean_cost": sum(
                observed_cost[i][selected_indices[i]] for i in range(n_tasks)
            ) / n_tasks,
            "selected_indices": selected_indices,
        })

    area = _compute_frontier_area(
        observed_success,
        observed_cost,
        lambda_results,
        max_cost=max_cost,
    )

    # Pure success (lambda=0): best possible success per task.
    pure_success = sum(
        max(observed_success[i]) for i in range(n_tasks)
    ) / n_tasks

    return {
        "area": area,
        "pure_success": pure_success,
        "lambda_results": lambda_results,
    }


# ---------------------------------------------------------------------------
# Global ranking metrics
# ---------------------------------------------------------------------------


def _rank_values(values: list[float]) -> list[float]:
    """Return fractional (average) ranks. Ties share the mean rank."""
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
    """Tie-aware Spearman rank correlation."""
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


def kendall_tau(xs: list[float], ys: list[float]) -> float:
    """Kendall tau-b correlation."""
    n = len(xs)
    if n < 2:
        return float("nan")
    concordant = discordant = ties_x = ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            x_diff = xs[i] - xs[j]
            y_diff = ys[i] - ys[j]
            if x_diff == 0:
                ties_x += 1
            if y_diff == 0:
                ties_y += 1
            if x_diff * y_diff > 0:
                concordant += 1
            elif x_diff * y_diff < 0:
                discordant += 1
    total = n * (n - 1) / 2.0
    if total == 0:
        return float("nan")
    denom = math.sqrt((total - ties_x) * (total - ties_y))
    if denom == 0:
        return float("nan")
    return (concordant - discordant) / denom


def pairwise_ranking_accuracy(
    predicted_ranks: list[float],
    true_ranks: list[float],
) -> float:
    """Fraction of pairs where predicted ordering matches true ordering."""
    n = len(predicted_ranks)
    if n < 2:
        return float("nan")
    correct = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            pred_diff = predicted_ranks[i] - predicted_ranks[j]
            true_diff = true_ranks[i] - true_ranks[j]
            total += 1
            if pred_diff * true_diff > 0 or (pred_diff == 0 and true_diff == 0):
                correct += 1
    return correct / total if total > 0 else float("nan")


def compute_global_ranking_metrics(
    predicted_means: list[float],
    true_means: list[float],
) -> dict[str, Any]:
    """Compute global policy ranking metrics.

    Args:
        predicted_means: per-policy predicted success rates (averaged over tasks)
        true_means: per-policy observed success rates (averaged over tasks)

    Returns:
        dict with spearman, kendall, pairwise_accuracy, top1, top3
    """
    n = len(predicted_means)
    if n < 2:
        return {
            "n": n,
            "spearman_rho": float("nan"),
            "kendall_tau": float("nan"),
            "pairwise_accuracy": float("nan"),
            "top1_correct": False,
            "top3_correct": False,
        }

    rho = spearman_rho(predicted_means, true_means)
    tau = kendall_tau(predicted_means, true_means)
    pair_acc = pairwise_ranking_accuracy(predicted_means, true_means)

    # Top-1 and Top-3.
    pred_order = sorted(range(n), key=lambda i: -predicted_means[i])
    true_order = sorted(range(n), key=lambda i: -true_means[i])
    top1 = pred_order[0] == true_order[0]
    top3 = any(true_order[0] == idx for idx in pred_order[:3])

    return {
        "n": n,
        "spearman_rho": rho,
        "kendall_tau": tau,
        "pairwise_accuracy": pair_acc,
        "top1_correct": top1,
        "top3_correct": top3,
    }


# ---------------------------------------------------------------------------
# Full evaluator
# ---------------------------------------------------------------------------


def evaluate_allocator(
    predictions: list[list[dict[str, Any]]],
    outcomes: list[list[dict[str, Any]]],
    policies: list[Any],
    lambda_grid: list[float],
    *,
    task_labels: list[dict[str, str]] | None = None,
    bootstrap_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Full evaluation of a meta-policy allocator.

    Args:
        predictions: predictions[i][j] for task i, policy j
        outcomes: outcomes[i][j] for task i, policy j
        policies: list of policy objects/descriptors
        lambda_grid: frozen lambda values for frontier
        task_labels: optional per-task metadata for stratified reporting
        bootstrap_config: dict with keys "seed", "num_trials" for bootstrap

    Returns:
        dict with frontier, ranking, per-template, per-difficulty, bootstrap
    """
    n_tasks = len(predictions)
    n_policies = len(policies)
    if n_tasks == 0:
        raise ValueError("no tasks to evaluate")

    # Validate complete panel.
    _validate_complete_panel(predictions, outcomes, n_tasks, n_policies)

    # Compute frontier.
    frontier = compute_frontier(predictions, outcomes, lambda_grid)

    # Compute global ranking.
    predicted_means: list[float] = []
    true_means: list[float] = []
    for j in range(n_policies):
        pred_sum = sum(predictions[i][j]["success_prob"] for i in range(n_tasks))
        true_sum = sum(
            1.0 if outcomes[i][j].get("verified_success", False) else 0.0
            for i in range(n_tasks)
        )
        predicted_means.append(pred_sum / n_tasks)
        true_means.append(true_sum / n_tasks)

    ranking = compute_global_ranking_metrics(predicted_means, true_means)

    # Per-template and per-difficulty breakdown.
    per_template: dict[str, Any] = {}
    per_difficulty: dict[str, Any] = {}
    if task_labels:
        for i in range(n_tasks):
            template = task_labels[i].get("template", "unknown")
            difficulty = task_labels[i].get("difficulty", "unknown")

            # Accumulate template-level data.
            if template not in per_template:
                per_template[template] = {
                    "task_indices": [],
                    "predictions": [],
                    "outcomes": [],
                }
            per_template[template]["task_indices"].append(i)
            per_template[template]["predictions"].append(predictions[i])
            per_template[template]["outcomes"].append(outcomes[i])

            # Accumulate difficulty-level data.
            if difficulty not in per_difficulty:
                per_difficulty[difficulty] = {
                    "task_indices": [],
                    "predictions": [],
                    "outcomes": [],
                }
            per_difficulty[difficulty]["task_indices"].append(i)
            per_difficulty[difficulty]["predictions"].append(predictions[i])
            per_difficulty[difficulty]["outcomes"].append(outcomes[i])

    template_results: dict[str, Any] = {}
    for template, data in per_template.items():
        tf = compute_frontier(data["predictions"], data["outcomes"], lambda_grid)
        template_results[template] = {
            "n_tasks": len(data["task_indices"]),
            "frontier_area": tf["frontier_area"],
            "pure_success_rate": tf["pure_success_rate"],
        }

    difficulty_results: dict[str, Any] = {}
    for difficulty, data in per_difficulty.items():
        df = compute_frontier(data["predictions"], data["outcomes"], lambda_grid)
        difficulty_results[difficulty] = {
            "n_tasks": len(data["task_indices"]),
            "frontier_area": df["frontier_area"],
            "pure_success_rate": df["pure_success_rate"],
        }

    # Bootstrap intervals (task-cluster resampling).
    bootstrap_intervals: dict[str, Any] = {}
    if bootstrap_config:
        seed = bootstrap_config.get("seed", 42)
        num_trials = bootstrap_config.get("num_trials", 500)
        bootstrap_intervals = _bootstrap_eval(
            predictions, outcomes, lambda_grid, seed, num_trials,
        )

    return {
        "n_tasks": n_tasks,
        "n_policies": n_policies,
        "frontier": frontier,
        "global_ranking": ranking,
        "per_template": template_results,
        "per_difficulty": difficulty_results,
        "bootstrap": bootstrap_intervals,
    }


def _validate_complete_panel(
    predictions: list[list[dict[str, Any]]],
    outcomes: list[list[dict[str, Any]]],
    n_tasks: int,
    n_policies: int,
) -> None:
    """Reject missing cells, duplicate attempts, or unequal inputs.

    Follows allocator_eval.py's strict complete-panel validation pattern.
    """
    if not predictions or not outcomes:
        raise ValueError("empty predictions or outcomes")
    if len(predictions) != n_tasks or len(outcomes) != n_tasks:
        raise ValueError("mismatched task count")

    for i in range(n_tasks):
        if len(predictions[i]) != n_policies:
            raise ValueError(
                f"task {i}: expected {n_policies} predictions, "
                f"got {len(predictions[i])}"
            )
        if len(outcomes[i]) != n_policies:
            raise ValueError(
                f"task {i}: expected {n_policies} outcomes, "
                f"got {len(outcomes[i])}"
            )


def _bootstrap_eval(
    predictions: list[list[dict[str, Any]]],
    outcomes: list[list[dict[str, Any]]],
    lambda_grid: list[float],
    seed: int,
    num_trials: int,
) -> dict[str, Any]:
    """Task-cluster bootstrap: resample whole tasks, not individual rows."""
    n_tasks = len(predictions)
    if n_tasks < 2 or num_trials < 1:
        return {"trials": 0}

    rng = random.Random(seed)
    frontier_areas: list[float] = []
    pure_success_rates: list[float] = []

    for _ in range(num_trials):
        indices = rng.choices(range(n_tasks), k=n_tasks)
        sub_predictions = [predictions[i] for i in indices]
        sub_outcomes = [outcomes[i] for i in indices]
        f = compute_frontier(sub_predictions, sub_outcomes, lambda_grid)
        frontier_areas.append(f["frontier_area"])
        pure_success_rates.append(f["pure_success_rate"])

    frontier_areas.sort()
    pure_success_rates.sort()

    def _pctile(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        idx = int(math.floor(p / 100.0 * (len(values) - 1)))
        idx2 = min(idx + 1, len(values) - 1)
        frac = p / 100.0 * (len(values) - 1) - idx
        return values[idx] * (1.0 - frac) + values[idx2] * frac

    return {
        "trials": num_trials,
        "frontier_area": {
            "mean": sum(frontier_areas) / len(frontier_areas),
            "ci_95_lower": _pctile(frontier_areas, 2.5),
            "ci_95_upper": _pctile(frontier_areas, 97.5),
        },
        "pure_success_rate": {
            "mean": sum(pure_success_rates) / len(pure_success_rates),
            "ci_95_lower": _pctile(pure_success_rates, 2.5),
            "ci_95_upper": _pctile(pure_success_rates, 97.5),
        },
    }


def compare_to_baselines(
    cnp_eval: dict[str, Any],
    baseline_evals: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compare CNP allocator against baselines.

    Args:
        cnp_eval: CNP evaluation result dict
        baseline_evals: dict mapping baseline name -> evaluation result dict

    Returns:
        comparison table with deltas vs each baseline.
    """
    cnp_frontier = cnp_eval.get("frontier", {})
    cnp_ranking = cnp_eval.get("global_ranking", {})

    comparison: dict[str, dict[str, Any]] = {}
    for name, eval_result in baseline_evals.items():
        baseline_frontier = eval_result.get("frontier", {})
        baseline_ranking = eval_result.get("global_ranking", {})

        fa_delta = None
        if "frontier_area" in cnp_frontier and "frontier_area" in baseline_frontier:
            fa_delta = cnp_frontier["frontier_area"] - baseline_frontier["frontier_area"]

        success_delta = None
        if "pure_success_rate" in cnp_frontier and "pure_success_rate" in baseline_frontier:
            success_delta = cnp_frontier["pure_success_rate"] - baseline_frontier["pure_success_rate"]

        rank_delta = None
        if (
            "pairwise_accuracy" in cnp_ranking
            and "pairwise_accuracy" in baseline_ranking
        ):
            rank_delta = cnp_ranking["pairwise_accuracy"] - baseline_ranking["pairwise_accuracy"]

        comparison[name] = {
            "frontier_area_delta": fa_delta,
            "pure_success_delta": success_delta,
            "pairwise_accuracy_delta": rank_delta,
            "cnp_frontier_area": cnp_frontier.get("frontier_area"),
            f"{name}_frontier_area": baseline_frontier.get("frontier_area"),
            "cnp_pure_success": cnp_frontier.get("pure_success_rate"),
            f"{name}_pure_success": baseline_frontier.get("pure_success_rate"),
        }

    return {
        "comparisons": comparison,
        "cnp_beats_all": all(
            (info.get("frontier_area_delta") or 0) > 0 for info in comparison.values()
        ),
        "cnp_beats_any": any(
            (info.get("frontier_area_delta") or 0) > 0 for info in comparison.values()
        ),
    }


__all__ = [
    "compute_frontier",
    "evaluate_allocator",
    "compare_to_baselines",
    "compute_global_ranking_metrics",
    "spearman_rho",
    "kendall_tau",
    "pairwise_ranking_accuracy",
]
