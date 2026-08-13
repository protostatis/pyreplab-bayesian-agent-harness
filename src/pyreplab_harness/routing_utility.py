"""Utility-based routing helper utilities for treatment selection.

The module implements a closed-form scorer used by routing workstreams:

``utility = predicted_success - lambda * (predicted_output_tokens / 10_000)``

Only predecision predictions are used; no observed outcomes/costs enter the
selection rule. Selection is deterministic and rejects malformed inputs:

* candidate IDs must be a non-empty immutable order of unique IDs
* each supplied candidate must be present exactly once in that order
* ``predicted_success`` must be finite and in ``[0, 1]``
* ``predicted_output_tokens`` must be finite and non-negative
* lambda must be finite and non-negative
* boolean inputs are invalid for numeric fields

Decision receipts are aggregate-safe: each entry records only candidate IDs,
their predictions, and computed utility.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence, Tuple


PRIMARY_LAMBDA = 1.0
"""Primary lambda used by the smoke and routing workstream."""

FROZEN_LAMBDA_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0)
"""Frozen sensitivity grid used by routing smoke expectations."""


def _coerce_non_empty_str(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a non-empty string, got {type(value)!r}")
    if not value:
        raise ValueError(f"{field} must be non-empty")
    return value


def _coerce_scalar(
    value: Any,
    *,
    field: str,
    lower: float | None = None,
    upper: float | None = None,
    allow_negative: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{field} must be a finite numeric value (int/float, not bool), got {type(value)!r}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite, got {value!r}")
    if lower is not None and number < lower:
        raise ValueError(f"{field} must be >= {lower}, got {number}")
    if upper is not None and number > upper:
        raise ValueError(f"{field} must be <= {upper}, got {number}")
    if not allow_negative and number < 0:
        raise ValueError(f"{field} must be non-negative, got {number}")
    return number


def _coerce_non_negative_scalar(value: Any, *, field: str) -> float:
    return _coerce_scalar(value, field=field, lower=0.0, allow_negative=False)


def _coerce_success_probability(value: Any) -> float:
    return _coerce_scalar(value, field="predicted_success", lower=0.0, upper=1.0, allow_negative=False)


def _coerce_lambda(value: Any) -> float:
    return _coerce_non_negative_scalar(value, field="lambda")


def _coerce_cost_tokens(value: Any) -> float:
    return _coerce_non_negative_scalar(value, field="predicted_output_tokens")


def _extract_candidate_id(candidate: Mapping[str, Any]) -> str:
    if "candidate_id" not in candidate and "id" not in candidate:
        raise ValueError("each candidate must include 'candidate_id' or 'id'")
    if "candidate_id" in candidate:
        return _coerce_non_empty_str(candidate["candidate_id"], field="candidate_id")
    return _coerce_non_empty_str(candidate["id"], field="id")


def _extract_candidate_success(candidate: Mapping[str, Any]) -> float:
    if "predicted_success" not in candidate:
        raise ValueError(f"candidate {candidate!r} missing 'predicted_success'")
    return _coerce_success_probability(candidate["predicted_success"])


def _extract_candidate_cost_tokens(candidate: Mapping[str, Any]) -> float:
    if "predicted_output_tokens" not in candidate:
        raise ValueError(f"candidate {candidate!r} missing 'predicted_output_tokens'")
    return _coerce_cost_tokens(candidate["predicted_output_tokens"])


def _candidate_map_by_id(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise TypeError("candidates must be a sequence of mapping-like objects")

    by_id: dict[str, Mapping[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise TypeError(
                f"each candidate must be a mapping, got {type(candidate)!r}; full candidate was {candidate!r}"
            )
        candidate_id = _extract_candidate_id(candidate)
        if candidate_id in by_id:
            raise ValueError(f"duplicate candidate id in candidates: {candidate_id!r}")
        by_id[candidate_id] = candidate
    return by_id


def _coerce_candidate_order(candidate_order: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(candidate_order, tuple):
        raise TypeError("candidate_order must be an immutable tuple of IDs")
    canonical = tuple(_coerce_non_empty_str(candidate_id, field="candidate_order") for candidate_id in candidate_order)
    if len(set(canonical)) != len(canonical):
        raise ValueError("candidate_order must contain unique candidate IDs")
    return canonical


def _validate_candidate_coverage(candidate_ids: set[str], order: tuple[str, ...]) -> None:
    missing = [candidate_id for candidate_id in order if candidate_id not in candidate_ids]
    if missing:
        raise ValueError(f"candidate order references missing candidate IDs: {missing}")
    extra = sorted(candidate_ids - set(order))
    if extra:
        raise ValueError(f"extra candidate IDs not present in order: {extra}")


def score_candidates(
    candidates: Sequence[Mapping[str, Any]],
    lam: float,
    *,
    candidate_order: Sequence[str],
) -> Tuple[dict[str, float | str | int], ...]:
    """Score candidates and return deterministic, aggregate-safe receipts.

    Parameters
    ----------
    candidates:
        Iterable of objects each containing a candidate id and predictions.
    lam:
        Utility penalty coefficient (lambda).
    candidate_order:
        Explicit immutable candidate order (IDs only). The function fails closed if
        this set differs from candidates.

    Returns
    -------
    tuple of receipts in ``candidate_order`` order, each containing:
        candidate_id, predicted_success, predicted_output_tokens,
        predicted_output_cost_units, utility, registry_position.
    """
    lam_value = _coerce_lambda(lam)
    order = _coerce_candidate_order(candidate_order)
    by_id = _candidate_map_by_id(candidates)
    _validate_candidate_coverage(set(by_id), order)

    scored: list[dict[str, float | str | int]] = []
    for position, candidate_id in enumerate(order):
        candidate = by_id[candidate_id]
        success = _extract_candidate_success(candidate)
        output_tokens = _extract_candidate_cost_tokens(candidate)
        output_cost_units = output_tokens / 10_000.0
        utility = success - (lam_value * output_cost_units)
        scored.append(
            {
                "candidate_id": candidate_id,
                "registry_position": position,
                "predicted_success": success,
                "predicted_output_tokens": output_tokens,
                "predicted_output_cost_units": output_cost_units,
                "utility": utility,
            }
        )
    return tuple(scored)


def _is_better(candidate_a: Mapping[str, Any], candidate_b: Mapping[str, Any]) -> bool:
    """Deterministic argmax-style comparator.

    Returns True iff ``candidate_a`` outranks ``candidate_b`` under the required
    key order:

    1) higher utility
    2) higher predicted_success
    3) lower predicted_output_cost_units
    4) lower registry_position (explicit order tie-break)
    """
    utility_a = float(candidate_a["utility"])
    utility_b = float(candidate_b["utility"])
    if utility_a != utility_b:
        return utility_a > utility_b

    success_a = float(candidate_a["predicted_success"])
    success_b = float(candidate_b["predicted_success"])
    if success_a != success_b:
        return success_a > success_b

    cost_a = float(candidate_a["predicted_output_cost_units"])
    cost_b = float(candidate_b["predicted_output_cost_units"])
    if cost_a != cost_b:
        return cost_a < cost_b

    return int(candidate_a["registry_position"]) < int(candidate_b["registry_position"])


def select_candidate(
    candidates: Sequence[Mapping[str, Any]],
    lam: float,
    *,
    candidate_order: Sequence[str],
) -> dict[str, Any]:
    """Score and choose the utility-optimal candidate in one closed-form pass.

    Returns an aggregate-safe decision receipt with only candidate IDs,
    predictions, computed utility, and the selected candidate id.
    """
    scored = score_candidates(candidates, lam, candidate_order=candidate_order)
    if not scored:
        raise ValueError("no candidates provided")

    best_index = 0
    for index in range(1, len(scored)):
        if _is_better(scored[index], scored[best_index]):
            best_index = index

    best = scored[best_index]
    return {
        "lambda": _coerce_lambda(lam),
        "candidates": scored,
        "selected_candidate_id": str(best["candidate_id"]),
        "selected_candidate_utility": best["utility"],
    }


def _run_case(
    candidates: Sequence[Mapping[str, Any]],
    lam: float,
    candidate_order: tuple[str, ...],
    *,
    expected: str,
) -> dict[str, Any]:
    receipt = select_candidate(candidates, lam, candidate_order=candidate_order)
    selected = receipt["selected_candidate_id"]
    return {
        "lambda": lam,
        "selected_candidate_id": selected,
        "expected_candidate_id": expected,
        "passed": selected == expected,
    }


def run_utility_scoring_smoke_matrix(
    *,
    lambda_grid: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Run synthetic utility selection checks for documentation and testability.

    The matrix is fully synthetic and uses no observed outcomes. It validates:

    * dominance under monotone better utility,
    * a trade-off case where preference switches across the lambda grid,
    * exact utility ties and registry-order tie-breaks,
    * failure-closed invalid inputs.

    Returns a structured report that is intended to be inspected in tests.
    """
    lambdas = tuple(
        _coerce_lambda(value) for value in (lambda_grid or FROZEN_LAMBDA_GRID)
    )

    # --- Case 1: dominance ---
    # Candidate D dominates all dimensions; selected for every finite lambda >= 0.
    dominance_order = ("best", "worse")
    dominance_candidates = (
        {"candidate_id": "best", "predicted_success": 0.95, "predicted_output_tokens": 50},
        {"candidate_id": "worse", "predicted_success": 0.50, "predicted_output_tokens": 1_000},
    )
    dominance_results = [
        _run_case(
            dominance_candidates,
            lam,
            dominance_order,
            expected="best",
        )
        for lam in lambdas
    ]

    # --- Case 2: tradeoff ---
    # A good-but-expensive candidate is preferred only at very small lambda.
    tradeoff_order = ("latency", "economy")
    tradeoff_candidates = (
        {"candidate_id": "latency", "predicted_success": 0.90, "predicted_output_tokens": 20_000},
        {"candidate_id": "economy", "predicted_success": 0.60, "predicted_output_tokens": 0},
    )
    expected_by_lambda = {
        lam: ("latency" if lam == 0.0 else "economy") for lam in lambdas
    }
    tradeoff_results = [
        _run_case(
            tradeoff_candidates,
            lam,
            tradeoff_order,
            expected=expected_by_lambda[lam],
        )
        for lam in lambdas
    ]

    # --- Case 3: exact utility tie + secondary tie-breaks ---
    # - For equal utility & equal predictions: registry order wins.
    tie_order = ("alpha", "zeta")
    tie_candidates = (
        {"candidate_id": "alpha", "predicted_success": 0.71, "predicted_output_tokens": 10_000},
        {"candidate_id": "zeta", "predicted_success": 0.71, "predicted_output_tokens": 10_000},
    )
    tie_results = [
        _run_case(tie_candidates, lam, tie_order, expected="alpha") for lam in lambdas
    ]

    # --- Case 4: direct higher success + lower cost candidate
    # high_succ should always win without reaching secondary/tertiary tie logic.
    success_tie_order = ("high_succ", "low_succ")
    success_tie_candidates = (
        {
            "candidate_id": "high_succ",
            "predicted_success": 0.9,
            "predicted_output_tokens": 1000,
        },
        {
            "candidate_id": "low_succ",
            "predicted_success": 0.8,
            "predicted_output_tokens": 2000,
        },
    )
    success_tie_results = [
        _run_case(success_tie_candidates, lam, success_tie_order, expected="high_succ")
        for lam in lambdas
    ]

    invalid_cases: dict[str, bool] = {}

    def _captures_validation_error(callback: Any) -> None:
        try:
            callback()
        except (ValueError, TypeError):
            return
        except Exception as error:
            raise ValueError("smoke matrix expected a validation error") from error
        raise ValueError("smoke matrix expected ValueError but no exception was raised")

    _captures_validation_error(
        lambda: select_candidate(
            ("not-a-mapping",),
            1.0,
            candidate_order=("x",),
        )
    )
    invalid_cases["missing_candidate_in_order"] = True

    _captures_validation_error(
        lambda: select_candidate(
            ({"candidate_id": "a", "predicted_success": 0.5, "predicted_output_tokens": 100}),
            1.0,
            candidate_order=("a", "b"),
        )
    )
    invalid_cases["extra_candidate_for_order"] = True

    _captures_validation_error(
        lambda: select_candidate(
            (
                {"candidate_id": "a", "predicted_success": 0.5, "predicted_output_tokens": 100},
                {"candidate_id": "a", "predicted_success": 0.6, "predicted_output_tokens": 200},
            ),
            1.0,
            candidate_order=("a",),
        )
    )
    invalid_cases["duplicate_candidate_ids"] = True

    _captures_validation_error(
        lambda: select_candidate(
            ({"candidate_id": "a", "predicted_success": 0.5, "predicted_output_tokens": -1},),
            1.0,
            candidate_order=("a",),
        )
    )
    invalid_cases["negative_cost"] = True

    _captures_validation_error(
        lambda: select_candidate(
            ({"candidate_id": "a", "predicted_success": float("nan"), "predicted_output_tokens": 10},),
            1.0,
            candidate_order=("a",),
        )
    )
    invalid_cases["nonfinite_success"] = True

    _captures_validation_error(
        lambda: select_candidate(
            ({"candidate_id": "a", "predicted_success": 0.5, "predicted_output_tokens": True},),
            1.0,
            candidate_order=("a",),
        )
    )
    invalid_cases["bool_cost"] = True

    _captures_validation_error(
        lambda: select_candidate(
            ({"candidate_id": "a", "predicted_success": 0.5, "predicted_output_tokens": 10},),
            float("inf"),
            candidate_order=("a",),
        )
    )
    invalid_cases["invalid_lambda"] = True

    passed = all(
        (
            all(item["passed"] for item in dominance_results),
            all(item["passed"] for item in tradeoff_results),
            all(item["passed"] for item in tie_results),
            all(item["passed"] for item in success_tie_results),
            all(invalid_cases.values()),
        )
    )

    return {
        "lambda_grid": list(lambdas),
        "primary_lambda": PRIMARY_LAMBDA,
        "cases": {
            "dominance": dominance_results,
            "tradeoff": tradeoff_results,
            "tie_by_order": tie_results,
            "tie_by_success": success_tie_results,
        },
        "invalid_cases": invalid_cases,
        "passed": passed,
    }


def synthetic_smoke_validator(_report: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Run the authoritative utility smoke for Stage-A integration.

    ``_report`` is accepted so the function can be used as a gate callback, but
    is intentionally ignored: validation is recomputed from this module's
    frozen scorer rather than trusting a caller-supplied report.
    """
    return run_utility_scoring_smoke_matrix()


__all__ = [
    "FROZEN_LAMBDA_GRID",
    "PRIMARY_LAMBDA",
    "run_utility_scoring_smoke_matrix",
    "synthetic_smoke_validator",
    "score_candidates",
    "select_candidate",
]
