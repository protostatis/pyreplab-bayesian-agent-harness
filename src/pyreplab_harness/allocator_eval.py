"""Held-out evaluation of learned task-to-policy allocators.

The legacy allocator is a deterministic decision rule over ``(task, policy)``: given
predecision features (``model_input``) it picks either the ``direct`` or the
``deliberate`` policy for each task.  This module evaluates that rule on
*paired held-out* task rows -- one verified ``direct`` attempt and one
verified ``deliberate`` attempt for the same ``task_id`` -- without any causal
leakage:

* Pairing requires exactly one usable row per policy per task in the evaluated
  split.  Duplicate attempts are rejected by default (ambiguity) or collapsed
  to the lexicographically-first ``attempt_id`` with ``duplicate_attempts``,
  and every exclusion is reported.
* The two policies are scored for every paired task with the *saved* outcome
  model via :func:`pyreplab_harness.outcome_model.score_policy_counterfactuals`
  -- only ``model_input.policy_id`` / ``model_input.policy_version`` are
  replaced, so the prediction for a task never sees the observed validation or
  test outcome or cost.
* Selection ranks tasks by predicted success uplift (posterior mean of
  ``deliberate`` minus ``direct``), ties broken by ``task_id``, and assigns
  exactly ``k`` tasks to ``deliberate`` for a configurable
  ``deliberate_fraction`` budget.  Optionally the uplift is divided by the
  positive train-estimated incremental token cost; because that is a global
  constant it preserves the ranking.
* Cost summaries are fitted on TRAIN rows only.  ``usage.total_tokens`` is the
  token cost, with a fallback to the ``input + output + cache`` totals.  A task
  with an unobservable cost is excluded from the cost means of the strategies
  that assign it, but stays in the success-rate denominator wherever possible.
* Strategies are compared on the observed paired potential outcomes: Always
  Direct, Always Deliberate, a cost-matched Monte Carlo Random Mix (exact-``k``
  assignments, deterministic seed, mean + 95% quantiles), the Neural
  Allocator, and an Oracle Upper Bound at the same ``k`` (selection by observed
  uplift; clearly labeled as an upper bound).  The learned selector's observed
  cost is *reporting only* and never enters selection.
* Deterministic strategies get task-bootstrap 95% CIs; a warning is emitted for
  small ``n`` so no statistical significance is claimed.

The output is dashboard-compatible: a top-level ``strategies`` dict whose
entries carry ``success_rate``, ``n``, ``mean_tokens``, ``mean_tool_calls``,
``mean_messages`` plus optional ``ci_95``, ``selected_deliberate`` and
``regret``, together with ``metadata``/``exclusions``/``model``/``split``/
``budget``.  No per-task predictions or outcomes are ever written (privacy).

The evaluation helpers that do not need PyTorch (pairing, cost extraction and
fitting, selection, strategy aggregation, bootstrap and Monte-Carlo baselines)
are pure and testable with injected predictions; only ``evaluate_allocator``
and :func:`score_pairs` require the saved torch artifacts.

Descriptor-enabled artifacts can additionally be evaluated against an
immutable multi-treatment registry.  That mode requires a strict complete
panel (one observed row for every candidate treatment on every evaluated
task), chooses the maximum saved posterior-predictive mean per task, and
reports observed cost only.  Its hindsight strategy is explicitly a realized
panel ceiling rather than a causal or expected-performance oracle.

CLI: ``python -m pyreplab_harness.allocator_eval DATASET ARTIFACT_DIR [OUTPUT]
[--split test] [--deliberate-fraction 0.5] ...``.  Add
``--treatment-registry REGISTRY [--treatments all|REF,...]`` for generalized
treatment mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

from . import outcome_model as om
from .io_utils import write_json
from .treatments import (
    TreatmentRegistry,
    TreatmentSpec,
    treatment_model_input_descriptor,
)

#: The two canonical experiment policies.
_DIRECT = "direct"
_DELIBERATE = "deliberate"
_POLICIES = (_DIRECT, _DELIBERATE)
_SPLITS = ("train", "validation", "test")
#: Emission order for the strategies dict (JSON files are re-sorted on write).
_STRATEGY_ORDER = (
    "always_direct",
    "always_deliberate",
    "random_mix",
    "neural_allocator",
    "oracle_upper_bound",
)
#: Global cost keys accepted as the token-cost fallback.
_COST_FALLBACK_KEYS = ("output", "input", "cache", "cache_creation", "cache_read")
#: Below this task count no significance claim is made.
_SMALL_N = 30

# ---------------------------------------------------------------------------
# Row-level cost extraction (pure, no torch)
# ---------------------------------------------------------------------------


def _as_int(value: Any) -> int | None:
    """Coerce a JSON scalar to an int, rejecting bools and non-finite floats."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def task_tokens(row: dict[str, Any]) -> int | None:
    """Token cost of one row: ``usage.total_tokens`` or the fallback sum.

    ``usage.total_tokens`` is preferred; otherwise the ``output + input +
    cache`` (plus ``cache_creation``/``cache_read``) totals are summed.  A row
    with no usable counter returns ``None`` (missing cost).
    """
    usage = row.get("usage")
    if not isinstance(usage, dict):
        return None
    total = _as_int(usage.get("total_tokens"))
    if total is not None:
        return total
    values = [_as_int(usage.get(key)) for key in _COST_FALLBACK_KEYS]
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present)


def _row_policy(row: dict[str, Any]) -> str | None:
    mi = row.get("model_input")
    policy = mi.get("policy_id") if isinstance(mi, dict) else row.get("policy_id")
    if policy is None:
        return None
    return str(policy)


def _row_split(row: dict[str, Any]) -> str:
    split = row.get("split")
    return str(split) if split in _SPLITS else "train"


def fit_train_cost(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit per-policy mean token costs on the TRAIN split only.

    Rows outside the train split, rows with an unknown policy, and rows with an
    unobservable cost never enter the fitted values.  The returned
    ``incremental_tokens_per_task`` (deliberate minus direct) is the global
    constant used by the optional cost-adjusted uplift.
    """
    buckets: dict[str, list[int]] = {_DIRECT: [], _DELIBERATE: []}
    for row in rows:
        if _row_split(row) != "train":
            continue
        policy = _row_policy(row)
        if policy not in buckets:
            continue
        tokens = task_tokens(row)
        if tokens is not None:
            buckets[policy].append(tokens)
    mean_direct = _mean(buckets[_DIRECT])
    mean_deliberate = _mean(buckets[_DELIBERATE])
    incremental: float | None = None
    if mean_direct is not None and mean_deliberate is not None:
        incremental = mean_deliberate - mean_direct
    return {
        "fit_split": "train",
        "n_direct": len(buckets[_DIRECT]),
        "n_deliberate": len(buckets[_DELIBERATE]),
        "mean_tokens_direct": mean_direct,
        "mean_tokens_deliberate": mean_deliberate,
        "incremental_tokens_per_task": incremental,
    }


# ---------------------------------------------------------------------------
# Pair grouping (pure, no torch)
# ---------------------------------------------------------------------------


def _resolve_attempt(rows: list[dict[str, Any]], duplicate_attempts: str) -> dict[str, Any] | None:
    if len(rows) == 1:
        return rows[0]
    if duplicate_attempts == "first":
        return min(rows, key=lambda row: str(row.get("attempt_id") or ""))
    # Ambiguity: reject the whole task.
    return None


def build_task_pairs(
    rows: list[dict[str, Any]], split: str, duplicate_attempts: str = "reject"
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Group rows of one split into paired ``(direct, deliberate)`` tasks.

    Only rows whose ``model_input.policy_id`` is exactly ``direct`` or
    ``deliberate`` are considered.  A task must provide exactly one usable row
    per policy: duplicate attempts are rejected (the task is excluded and
    reported) unless ``duplicate_attempts="first"``, which deterministically
    keeps the smallest ``attempt_id``.  Returns ``(pairs, exclusions)`` where
    each pair is ``{"task_id", "split", "direct", "deliberate"}`` and pairs are
    sorted by ``task_id``.
    """
    if duplicate_attempts not in ("reject", "first"):
        raise ValueError(
            f"duplicate_attempts must be 'reject' or 'first', got {duplicate_attempts!r}"
        )
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        if _row_split(row) != split:
            continue
        policy = _row_policy(row)
        if policy not in _POLICIES:
            continue
        task_id = row.get("task_id")
        if not isinstance(task_id, str):
            continue
        grouped.setdefault(task_id, {_DIRECT: [], _DELIBERATE: []})[policy].append(row)

    pairs: list[dict[str, Any]] = []
    missing_pair: list[str] = []
    duplicate_tasks: list[str] = []
    for task_id in sorted(grouped):
        direct_rows = grouped[task_id][_DIRECT]
        deliberate_rows = grouped[task_id][_DELIBERATE]
        if not direct_rows or not deliberate_rows:
            missing_pair.append(task_id)
            continue
        direct = _resolve_attempt(direct_rows, duplicate_attempts)
        deliberate = _resolve_attempt(deliberate_rows, duplicate_attempts)
        if direct is None or deliberate is None:
            duplicate_tasks.append(task_id)
            continue
        pairs.append(
            {
                "task_id": task_id,
                "split": split,
                _DIRECT: direct,
                _DELIBERATE: deliberate,
            }
        )

    exclusions: dict[str, Any] = {
        "split": split,
        "tasks_in_split": len(grouped),
        "evaluated": len(pairs),
        "missing_pair": {"count": len(missing_pair), "task_ids": missing_pair},
        "duplicate_attempts": {"count": len(duplicate_tasks), "task_ids": duplicate_tasks},
    }
    return pairs, exclusions


# ---------------------------------------------------------------------------
# Generalized treatment-panel construction (pure, no torch)
# ---------------------------------------------------------------------------


def _resolve_treatment_reference(
    registry: TreatmentRegistry, reference: str
) -> TreatmentSpec:
    """Resolve a registry reference without weakening identity checks."""
    reference = str(reference).strip()
    if not reference:
        raise KeyError("empty treatment reference")
    try:
        return registry.by_bundle_id(reference)
    except KeyError:
        pass
    try:
        return registry.by_hash(reference)
    except KeyError:
        pass
    if "@" in reference:
        treatment_id, version = reference.rsplit("@", 1)
        try:
            return registry.by_id_version(treatment_id, version)
        except KeyError:
            pass
    return registry.by_id(reference)


def resolve_treatment_candidates(
    registry: TreatmentRegistry,
    references: str | list[str] | tuple[str, ...] | None = None,
) -> list[TreatmentSpec]:
    """Resolve and canonically order a non-duplicated treatment menu.

    ``None`` and ``"all"`` select the entire registry.  Canonical bundle-id
    ordering makes argmax tie-breaking independent of registry file order.
    """
    if references is None or (
        isinstance(references, str) and references.strip().lower() == "all"
    ):
        selected = list(registry)
    else:
        if isinstance(references, str):
            values = [value.strip() for value in references.split(",") if value.strip()]
        else:
            values = [str(value).strip() for value in references if str(value).strip()]
        if not values:
            raise ValueError("at least one treatment reference is required")
        if any(value.lower() == "all" for value in values):
            raise ValueError("'all' cannot be combined with other treatment references")
        selected = []
        seen: set[str] = set()
        for reference in values:
            try:
                treatment = _resolve_treatment_reference(registry, reference)
            except KeyError as error:
                raise ValueError(f"unknown treatment reference: {reference!r}") from error
            if treatment.bundle_id in seen:
                raise ValueError(f"duplicate treatment selection: {reference!r}")
            seen.add(treatment.bundle_id)
            selected.append(treatment)
    if not selected:
        raise ValueError("treatment registry contains no treatments")
    return sorted(selected, key=lambda treatment: treatment.bundle_id)


def treatment_candidate_set_hash(treatments: list[TreatmentSpec]) -> str:
    """SHA-256 commitment to an ordered-independent treatment menu."""
    payload = json.dumps(
        sorted(treatment.bundle_hash for treatment in treatments),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_task_model_input(model_input: dict[str, Any]) -> str:
    task_input = dict(model_input)
    for field in ("policy_id", "policy_version", "treatment"):
        task_input.pop(field, None)
    return json.dumps(
        task_input,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _validate_treatment_row(
    row: dict[str, Any], treatment: TreatmentSpec, registry_hash: str
) -> None:
    """Validate that one observed row is the exact registered bundle."""
    task_id = row.get("task_id")
    label = f"task {task_id!r}, treatment {treatment.bundle_id!r}"
    if str(row.get("policy_id")) != treatment.id:
        raise ValueError(f"policy_id mismatch for {label}")
    if str(row.get("policy_version")) != treatment.version:
        raise ValueError(f"policy_version mismatch for {label}")
    if row.get("treatment_bundle_id") != treatment.bundle_id:
        raise ValueError(f"treatment_bundle_id mismatch for {label}")
    if row.get("treatment_bundle_hash") != treatment.bundle_hash:
        raise ValueError(f"treatment_bundle_hash mismatch for {label}")
    observed_registry_hash = row.get("treatment_registry_hash")
    if str(observed_registry_hash) != registry_hash:
        raise ValueError(f"treatment_registry_hash mismatch for {label}")

    model_input = row.get("model_input")
    if not isinstance(model_input, dict):
        raise ValueError(f"model_input must be an object for {label}")
    if str(model_input.get("policy_id")) != treatment.id:
        raise ValueError(f"model_input.policy_id mismatch for {label}")
    if str(model_input.get("policy_version")) != treatment.version:
        raise ValueError(f"model_input.policy_version mismatch for {label}")
    if model_input.get("treatment") != treatment_model_input_descriptor(treatment):
        raise ValueError(f"model_input.treatment descriptor mismatch for {label}")


def build_treatment_panels(
    rows: list[dict[str, Any]],
    split: str,
    treatments: list[TreatmentSpec],
    registry_hash: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build strict complete held-out panels for an immutable treatment menu.

    Every task represented by a candidate row in ``split`` must have exactly
    one row for every candidate.  Missing cells, duplicate attempts, identity
    drift, cross-split task IDs, and unequal task-side model inputs are hard
    errors rather than silent complete-case exclusions.
    """
    if split not in _SPLITS:
        raise ValueError(f"split must be one of {_SPLITS}, got {split!r}")
    if not treatments:
        raise ValueError("at least one treatment is required")
    by_identity = {
        (treatment.id, treatment.version): treatment for treatment in treatments
    }
    if len(by_identity) != len(treatments):
        raise ValueError("candidate treatments must have unique id/version pairs")

    candidate_bundle_ids = {treatment.bundle_id for treatment in treatments}
    candidate_bundle_hashes = {treatment.bundle_hash for treatment in treatments}
    candidate_splits: dict[str, set[str]] = {}
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        task_id = row.get("task_id")
        if not isinstance(task_id, str):
            continue
        identity = (str(row.get("policy_id")), str(row.get("policy_version")))
        treatment = by_identity.get(identity)
        if treatment is None:
            # A candidate bundle marker with another identity is tampering, not
            # an unrelated treatment row that can safely be ignored.
            if (
                row.get("treatment_bundle_id") in candidate_bundle_ids
                or row.get("treatment_bundle_hash") in candidate_bundle_hashes
            ):
                raise ValueError(
                    f"candidate treatment identity mismatch for task {task_id!r}"
                )
            continue
        row_split = row.get("split")
        if row_split not in _SPLITS:
            raise ValueError(
                f"invalid split {row_split!r} for candidate task {task_id!r}"
            )
        row_split = str(row_split)
        candidate_splits.setdefault(task_id, set()).add(row_split)
        if row_split != split:
            continue
        _validate_treatment_row(row, treatment, registry_hash)
        grouped.setdefault(task_id, {}).setdefault(treatment.bundle_id, []).append(row)

    leaked = sorted(
        task_id for task_id, task_splits in candidate_splits.items() if len(task_splits) > 1
    )
    if leaked:
        raise ValueError(f"candidate task ids occur in multiple splits: {leaked}")

    panels: list[dict[str, Any]] = []
    expected_ids = [treatment.bundle_id for treatment in treatments]
    for task_id in sorted(grouped):
        cells = grouped[task_id]
        missing = [bundle_id for bundle_id in expected_ids if not cells.get(bundle_id)]
        if missing:
            raise ValueError(
                f"incomplete treatment panel for task {task_id!r}; missing {missing}"
            )
        duplicates = [
            bundle_id for bundle_id in expected_ids if len(cells[bundle_id]) != 1
        ]
        if duplicates:
            raise ValueError(
                f"duplicate treatment attempts for task {task_id!r}: {duplicates}"
            )
        panel_rows = {bundle_id: cells[bundle_id][0] for bundle_id in expected_ids}
        canonical_inputs = {
            _canonical_task_model_input(row["model_input"])
            for row in panel_rows.values()
        }
        if len(canonical_inputs) != 1:
            raise ValueError(
                f"task-side model_input differs across treatments for task {task_id!r}"
            )
        panels.append({"task_id": task_id, "split": split, "rows": panel_rows})

    exclusions = {
        "split": split,
        "tasks_in_split": len(grouped),
        "evaluated": len(panels),
        "missing_cells": {"count": 0, "task_ids": []},
        "duplicate_attempts": {"count": 0, "task_ids": []},
        "policy": "strict complete panel; any integrity failure aborts evaluation",
    }
    return panels, exclusions


# ---------------------------------------------------------------------------
# Pair-level observables (pure)
# ---------------------------------------------------------------------------


def pair_outcome(pair: dict[str, Any], policy: str) -> bool:
    """Observed verified success of ``policy`` on this paired task."""
    return bool(pair[policy]["verified_success"])


def pair_tokens(pair: dict[str, Any], policy: str) -> int | None:
    return task_tokens(pair[policy])


def _pair_int(row: dict[str, Any], key: str) -> int | None:
    value = row.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def pair_tool_calls(pair: dict[str, Any], policy: str) -> int | None:
    return _pair_int(pair[policy], "tool_call_count")


def pair_messages(pair: dict[str, Any], policy: str) -> int | None:
    return _pair_int(pair[policy], "assistant_message_count")


def _mean(values: list[Any]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return sum(cleaned) / len(cleaned) if cleaned else None


def _percentile(values: list[float], percentile: float) -> float:
    """Linear-interpolation percentile of ``values`` (sorted internally)."""
    if not values:
        raise ValueError("cannot compute a percentile of an empty list")
    sorted_values = sorted(float(value) for value in values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = float(percentile) / 100.0 * (len(sorted_values) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return sorted_values[lower]
    fraction = rank - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _quantiles(values: list[Any], lo: float = 2.5, hi: float = 97.5) -> list[float | None]:
    present = [float(value) for value in values if value is not None]
    if not present:
        return [None, None]
    return [_percentile(present, lo), _percentile(present, hi)]


# ---------------------------------------------------------------------------
# Strategy aggregation (pure)
# ---------------------------------------------------------------------------


def _aggregate_with_assignment(
    pairs: list[dict[str, Any]], assignment: list[str]
) -> dict[str, Any]:
    """Aggregate observed outcomes and costs for one per-task policy assignment.

    ``assignment[i]`` is ``direct`` or ``deliberate``.  Success is computed over
    all tasks; each cost dimension is averaged only over tasks whose assigned
    row makes it observable (missing costs exclude a task from that mean but
    never from the success denominator).
    """
    n = len(pairs)
    if len(assignment) != n:
        raise ValueError("assignment length must equal the number of pairs")
    successes = 0
    tokens: list[int] = []
    tools: list[int] = []
    messages: list[int] = []
    for index, policy in enumerate(assignment):
        pair = pairs[index]
        if policy not in _POLICIES:
            raise ValueError(f"invalid policy in assignment: {policy!r}")
        if pair_outcome(pair, policy):
            successes += 1
        token_value = pair_tokens(pair, policy)
        if token_value is not None:
            tokens.append(token_value)
        tool_value = pair_tool_calls(pair, policy)
        if tool_value is not None:
            tools.append(tool_value)
        message_value = pair_messages(pair, policy)
        if message_value is not None:
            messages.append(message_value)
    return {
        "n": n,
        "successes": successes,
        "success_rate": successes / n if n else None,
        "mean_tokens": _mean(tokens),
        "mean_tool_calls": _mean(tools),
        "mean_messages": _mean(messages),
        "cost_n": len(tokens),
    }


def _per_task_oracle(pairs: list[dict[str, Any]]) -> int:
    """Number of tasks where at least one policy succeeds (per-task oracle)."""
    return sum(
        1
        for pair in pairs
        if pair_outcome(pair, _DIRECT) or pair_outcome(pair, _DELIBERATE)
    )


def _regret(pairs: list[dict[str, Any]], assignment: list[str]) -> float:
    """Regret vs the per-task oracle: oracle successes minus strategy successes."""
    return float(_per_task_oracle(pairs) - _aggregate_with_assignment(pairs, assignment)["successes"])


def task_bootstrap_ci(
    pairs: list[dict[str, Any]],
    assignment: list[str],
    seed: int,
    bootstrap_trials: int,
) -> list[float] | None:
    """Task-level bootstrap 95% CI of a deterministic strategy's success rate.

    Tasks (with their already-assigned policy) are resampled with replacement
    and the strategy's success rate is recomputed each time; the 2.5% and 97.5%
    percentiles are returned.  Returns ``None`` when fewer than two tasks or no
    trials are requested.  Deterministic under a fixed ``seed``.
    """
    n = len(pairs)
    if n < 2 or bootstrap_trials < 1:
        return None
    rng = random.Random(seed + 1000)
    rates: list[float] = []
    for _ in range(bootstrap_trials):
        indices = rng.choices(range(n), k=n)
        sub_pairs = [pairs[index] for index in indices]
        sub_assignment = [assignment[index] for index in indices]
        rates.append(_aggregate_with_assignment(sub_pairs, sub_assignment)["success_rate"])
    return [_percentile(rates, 2.5), _percentile(rates, 97.5)]


def random_mix_aggregate(
    pairs: list[dict[str, Any]], k: int, seed: int, random_trials: int
) -> dict[str, Any]:
    """Monte Carlo exact-``k`` random mix (cost-matched baseline).

    Each trial draws a uniformly random subset of exactly ``k`` tasks to assign
    to ``deliberate`` (the rest stay ``direct``).  Reports the mean of each
    trial-level aggregate plus 2.5%/97.5% quantiles under a deterministic
    ``seed``.
    """
    n = len(pairs)
    if k < 0 or k > n:
        raise ValueError(f"k={k} is outside [0, {n}]")
    if random_trials < 1:
        raise ValueError("random_trials must be >= 1")
    rng = random.Random(seed)
    success_rates: list[float] = []
    token_means: list[float] = []
    tool_means: list[float] = []
    message_means: list[float] = []
    regrets: list[float] = []
    for _ in range(random_trials):
        selected = set(rng.sample(range(n), k))
        assignment = [_DELIBERATE if index in selected else _DIRECT for index in range(n)]
        aggregate = _aggregate_with_assignment(pairs, assignment)
        success_rates.append(aggregate["success_rate"])
        token_means.append(aggregate["mean_tokens"])
        tool_means.append(aggregate["mean_tool_calls"])
        message_means.append(aggregate["mean_messages"])
        regrets.append(_regret(pairs, assignment))
    return {
        "n": n,
        "success_rate": _mean(success_rates),
        "mean_tokens": _mean(token_means),
        "mean_tool_calls": _mean(tool_means),
        "mean_messages": _mean(message_means),
        "selected_deliberate": k,
        "regret": _mean(regrets),
        "quantiles": {
            "success_rate": _quantiles(success_rates),
            "mean_tokens": _quantiles(token_means),
            "mean_tool_calls": _quantiles(tool_means),
            "mean_messages": _quantiles(message_means),
        },
        "random_trials": random_trials,
    }


def _budget_k(deliberate_fraction: float, n: int) -> int:
    """Exact deliberate-task budget: half-up rounding of ``fraction * n``."""
    return max(0, min(n, int(deliberate_fraction * n + 0.5)))


def _validate_fraction(deliberate_fraction: Any) -> float:
    try:
        fraction = float(deliberate_fraction)
    except (TypeError, ValueError):
        raise ValueError(
            f"deliberate_fraction must be a number in [0, 1], got {deliberate_fraction!r}"
        )
    if not (0.0 <= fraction <= 1.0):
        raise ValueError(f"deliberate_fraction must be in [0, 1], got {fraction}")
    return fraction


def select_deliberate_tasks(
    pairs: list[dict[str, Any]],
    uplift: list[float],
    deliberate_fraction: float,
) -> tuple[list[str], int]:
    """Select exactly ``k`` tasks for ``deliberate`` by predicted success uplift.

    Tasks are ranked by ``uplift`` descending; ties are broken deterministically
    by ``task_id`` ascending.  The top ``k`` become ``deliberate`` (all others
    ``direct``), with ``k = round(deliberate_fraction * n)``.  The returned
    task ids are in ``pairs`` order.  This function never reads observed
    outcomes or costs, so selection cannot leak the evaluation labels.
    """
    n = len(pairs)
    if n == 0:
        raise ValueError("cannot select deliberate tasks from an empty pair list")
    if len(uplift) != n:
        raise ValueError("uplift must have one value per paired task")
    fraction = _validate_fraction(deliberate_fraction)
    k = _budget_k(fraction, n)
    ranked = sorted(range(n), key=lambda index: (-float(uplift[index]), pairs[index]["task_id"]))
    selected_set = set(ranked[:k])
    selected = [pairs[index]["task_id"] for index in range(n) if index in selected_set]
    return selected, k


def compare_strategies(
    pairs: list[dict[str, Any]],
    selected_ids: list[str],
    *,
    seed: int = 42,
    random_trials: int = 1000,
    bootstrap_trials: int = 1000,
) -> dict[str, Any]:
    """Compare all aggregate strategies on the observed paired outcomes.

    ``selected_ids`` is the neural allocator's (prediction-driven) choice of
    deliberate tasks; the random mix and the oracle upper bound share its exact
    budget ``k``.  The oracle upper bound selects by *observed* uplift and is
    labeled as an upper bound.  No per-task data is returned.
    """
    n = len(pairs)
    if n == 0:
        raise ValueError("no paired tasks to compare")
    if random_trials < 1:
        raise ValueError("random_trials must be >= 1")
    if bootstrap_trials < 1:
        raise ValueError("bootstrap_trials must be >= 1")
    selected_set = set(selected_ids)
    known = {pair["task_id"] for pair in pairs}
    unknown = selected_set - known
    if unknown:
        raise ValueError(f"selected task ids are not among the paired tasks: {sorted(unknown)}")

    neural_assignment = [
        _DELIBERATE if pair["task_id"] in selected_set else _DIRECT for pair in pairs
    ]
    direct_assignment = [_DIRECT] * n
    deliberate_assignment = [_DELIBERATE] * n

    # Oracle upper bound at the same k: select by observed uplift, ties by id.
    k = len(selected_set)
    oracle_ranked = sorted(
        range(n),
        key=lambda index: (
            -(
                (1 if pair_outcome(pairs[index], _DELIBERATE) else 0)
                - (1 if pair_outcome(pairs[index], _DIRECT) else 0)
            ),
            pairs[index]["task_id"],
        ),
    )
    oracle_selected = set(oracle_ranked[:k])
    oracle_assignment = [
        _DELIBERATE if index in oracle_selected else _DIRECT for index in range(n)
    ]

    strategies: dict[str, Any] = {}
    for name, assignment in (
        ("always_direct", direct_assignment),
        ("always_deliberate", deliberate_assignment),
        ("neural_allocator", neural_assignment),
        ("oracle_upper_bound", oracle_assignment),
    ):
        aggregate = _aggregate_with_assignment(pairs, assignment)
        strategies[name] = {
            "n": aggregate["n"],
            "success_rate": aggregate["success_rate"],
            "mean_tokens": aggregate["mean_tokens"],
            "mean_tool_calls": aggregate["mean_tool_calls"],
            "mean_messages": aggregate["mean_messages"],
            "cost_n": aggregate["cost_n"],
            "selected_deliberate": n if name == "always_deliberate" else (k if name in ("neural_allocator", "oracle_upper_bound") else 0),
            "regret": _regret(pairs, assignment),
            "ci_95": task_bootstrap_ci(pairs, assignment, seed, bootstrap_trials),
        }
    strategies["random_mix"] = random_mix_aggregate(pairs, k, seed, random_trials)

    ordered: dict[str, Any] = {}
    for name in _STRATEGY_ORDER:
        ordered[name] = strategies[name]
    return ordered


# ---------------------------------------------------------------------------
# Generalized treatment allocation and aggregation (pure, no torch)
# ---------------------------------------------------------------------------


def _panel_row(panel: dict[str, Any], bundle_id: str) -> dict[str, Any]:
    rows = panel.get("rows")
    if not isinstance(rows, dict) or bundle_id not in rows:
        raise ValueError(
            f"panel {panel.get('task_id')!r} has no row for {bundle_id!r}"
        )
    return rows[bundle_id]


def _aggregate_treatment_assignment(
    panels: list[dict[str, Any]], assignment: list[str]
) -> dict[str, Any]:
    if len(assignment) != len(panels):
        raise ValueError("assignment length must equal the number of treatment panels")
    successes = 0
    tokens: list[int] = []
    tools: list[int] = []
    messages: list[int] = []
    counts: dict[str, int] = {}
    for panel, bundle_id in zip(panels, assignment):
        row = _panel_row(panel, bundle_id)
        counts[bundle_id] = counts.get(bundle_id, 0) + 1
        if bool(row.get("verified_success")):
            successes += 1
        token_value = task_tokens(row)
        if token_value is not None:
            tokens.append(token_value)
        tool_value = _pair_int(row, "tool_call_count")
        if tool_value is not None:
            tools.append(tool_value)
        message_value = _pair_int(row, "assistant_message_count")
        if message_value is not None:
            messages.append(message_value)
    n = len(panels)
    return {
        "n": n,
        "successes": successes,
        "success_rate": successes / n if n else None,
        "mean_tokens": _mean(tokens),
        "mean_tool_calls": _mean(tools),
        "mean_messages": _mean(messages),
        "metric_n": {
            "tokens": len(tokens),
            "tool_calls": len(tools),
            "messages": len(messages),
        },
        "allocation_counts": dict(sorted(counts.items())),
    }


def _treatment_oracle_successes(
    panels: list[dict[str, Any]], bundle_ids: list[str]
) -> int:
    return sum(
        1
        for panel in panels
        if any(bool(_panel_row(panel, bundle_id).get("verified_success")) for bundle_id in bundle_ids)
    )


def _treatment_regret(
    panels: list[dict[str, Any]], assignment: list[str], bundle_ids: list[str]
) -> float:
    aggregate = _aggregate_treatment_assignment(panels, assignment)
    return float(_treatment_oracle_successes(panels, bundle_ids) - aggregate["successes"])


def treatment_task_bootstrap_ci(
    panels: list[dict[str, Any]],
    assignment: list[str],
    *,
    seed: int,
    bootstrap_trials: int,
) -> list[float] | None:
    """Task-bootstrap interval conditional on a fixed fitted allocator."""
    n = len(panels)
    if n < 2 or bootstrap_trials < 1:
        return None
    rng = random.Random(seed + 2000)
    rates: list[float] = []
    for _ in range(bootstrap_trials):
        indices = rng.choices(range(n), k=n)
        sub_panels = [panels[index] for index in indices]
        sub_assignment = [assignment[index] for index in indices]
        rates.append(
            float(
                _aggregate_treatment_assignment(sub_panels, sub_assignment)[
                    "success_rate"
                ]
            )
        )
    return [_percentile(rates, 2.5), _percentile(rates, 97.5)]


def select_treatment_argmax(
    panels: list[dict[str, Any]],
    predictions: list[dict[str, dict[str, float]]],
    treatments: list[TreatmentSpec],
) -> list[str]:
    """Choose maximum predicted success per task, ties by bundle ID."""
    if not panels:
        raise ValueError("cannot allocate an empty treatment panel list")
    if len(predictions) != len(panels):
        raise ValueError("predictions must have one entry per treatment panel")
    bundle_ids = sorted(treatment.bundle_id for treatment in treatments)
    if not bundle_ids:
        raise ValueError("at least one candidate treatment is required")
    assignment: list[str] = []
    for panel, prediction in zip(panels, predictions):
        missing = [bundle_id for bundle_id in bundle_ids if bundle_id not in prediction]
        if missing:
            raise ValueError(
                f"missing treatment predictions for task {panel.get('task_id')!r}: {missing}"
            )
        means: dict[str, float] = {}
        for bundle_id in bundle_ids:
            try:
                mean = float(prediction[bundle_id]["mean"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid prediction for task {panel.get('task_id')!r}, "
                    f"treatment {bundle_id!r}"
                ) from error
            if not math.isfinite(mean):
                raise ValueError(
                    f"non-finite prediction for task {panel.get('task_id')!r}, "
                    f"treatment {bundle_id!r}"
                )
            means[bundle_id] = mean
        assignment.append(min(bundle_ids, key=lambda bundle_id: (-means[bundle_id], bundle_id)))
    return assignment


def uniform_random_treatment_aggregate(
    panels: list[dict[str, Any]],
    treatments: list[TreatmentSpec],
    *,
    seed: int,
    random_trials: int,
) -> dict[str, Any]:
    """Independent uniform treatment baseline with exact expected success."""
    if not panels:
        raise ValueError("cannot aggregate an empty treatment panel list")
    if random_trials < 1:
        raise ValueError("random_trials must be >= 1")
    bundle_ids = sorted(treatment.bundle_id for treatment in treatments)
    if not bundle_ids:
        raise ValueError("at least one candidate treatment is required")
    n = len(panels)
    exact_success_rate = sum(
        1.0 if bool(_panel_row(panel, bundle_id).get("verified_success")) else 0.0
        for panel in panels
        for bundle_id in bundle_ids
    ) / (n * len(bundle_ids))
    oracle_successes = _treatment_oracle_successes(panels, bundle_ids)

    rng = random.Random(seed)
    trial_aggregates: list[dict[str, Any]] = []
    for _ in range(random_trials):
        assignment = [rng.choice(bundle_ids) for _panel in panels]
        trial_aggregates.append(_aggregate_treatment_assignment(panels, assignment))
    success_rates = [float(item["success_rate"]) for item in trial_aggregates]
    token_means = [item["mean_tokens"] for item in trial_aggregates]
    tool_means = [item["mean_tool_calls"] for item in trial_aggregates]
    message_means = [item["mean_messages"] for item in trial_aggregates]
    return {
        "strategy_type": "randomized_uniform",
        "n": n,
        "successes": exact_success_rate * n,
        "success_rate": exact_success_rate,
        "mean_tokens": _mean(token_means),
        "mean_tool_calls": _mean(tool_means),
        "mean_messages": _mean(message_means),
        "metric_n": {
            metric: _mean([item["metric_n"][metric] for item in trial_aggregates])
            for metric in ("tokens", "tool_calls", "messages")
        },
        "allocation_counts": {
            bundle_id: n / len(bundle_ids) for bundle_id in bundle_ids
        },
        "regret": float(oracle_successes - exact_success_rate * n),
        "randomization_quantiles_95": {
            "success_rate": _quantiles(success_rates),
            "mean_tokens": _quantiles(token_means),
            "mean_tool_calls": _quantiles(tool_means),
            "mean_messages": _quantiles(message_means),
        },
        "random_trials": random_trials,
    }


def compare_treatment_strategies(
    panels: list[dict[str, Any]],
    neural_assignment: list[str],
    treatments: list[TreatmentSpec],
    *,
    seed: int = 42,
    random_trials: int = 1000,
    bootstrap_trials: int = 1000,
) -> dict[str, Any]:
    """Aggregate fixed, uniform-random, neural, and realized-oracle values."""
    if not panels:
        raise ValueError("no treatment panels to compare")
    if random_trials < 1:
        raise ValueError("random_trials must be >= 1")
    if bootstrap_trials < 1:
        raise ValueError("bootstrap_trials must be >= 1")
    bundle_ids = sorted(treatment.bundle_id for treatment in treatments)
    if len(set(bundle_ids)) != len(bundle_ids) or not bundle_ids:
        raise ValueError("candidate treatments must have unique bundle IDs")
    if len(neural_assignment) != len(panels):
        raise ValueError("neural assignment length must equal panel count")
    unknown = sorted(set(neural_assignment) - set(bundle_ids))
    if unknown:
        raise ValueError(f"neural assignment contains unknown treatments: {unknown}")

    def deterministic_strategy(
        strategy_type: str,
        assignment: list[str],
        *,
        uses_outcomes: bool = False,
    ) -> dict[str, Any]:
        aggregate = _aggregate_treatment_assignment(panels, assignment)
        return {
            "strategy_type": strategy_type,
            **aggregate,
            "regret": _treatment_regret(panels, assignment, bundle_ids),
            "ci_95": treatment_task_bootstrap_ci(
                panels,
                assignment,
                seed=seed,
                bootstrap_trials=bootstrap_trials,
            ),
            "uses_heldout_outcomes_for_selection": uses_outcomes,
        }

    strategies: dict[str, Any] = {}
    for bundle_id in bundle_ids:
        strategies[f"always::{bundle_id}"] = deterministic_strategy(
            "always", [bundle_id] * len(panels)
        )

    strategies["uniform_random"] = uniform_random_treatment_aggregate(
        panels,
        treatments,
        seed=seed,
        random_trials=random_trials,
    )
    strategies["neural_argmax"] = deterministic_strategy(
        "neural_argmax", neural_assignment
    )
    hindsight_assignment: list[str] = []
    for panel in panels:
        successful = [
            bundle_id
            for bundle_id in bundle_ids
            if bool(_panel_row(panel, bundle_id).get("verified_success"))
        ]
        hindsight_assignment.append(successful[0] if successful else bundle_ids[0])
    strategies["hindsight_realized_oracle"] = deterministic_strategy(
        "hindsight_realized_oracle",
        hindsight_assignment,
        uses_outcomes=True,
    )
    return strategies


# ---------------------------------------------------------------------------
# Neural scoring (torch-dependent)
# ---------------------------------------------------------------------------


def score_pairs(
    model: Any,
    pre: om.Preprocessor,
    pairs: list[dict[str, Any]],
    *,
    posterior_samples: int = 50,
    seed: int | None = None,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Score both policies for every paired task through the saved model.

    For each task the *direct* row's ``model_input`` is reused and only
    ``policy_id``/``policy_version`` are swapped (the existing
    ``score_policy_counterfactuals`` path).  Observed outcomes and costs never
    enter the feature path.  Returns one dict per pair of the form
    ``{"direct": {"mean", "std"}, "deliberate": {"mean", "std"}}``.
    """
    if not om.TORCH_AVAILABLE:
        raise RuntimeError(
            "scoring the neural allocator requires PyTorch, which is not installed"
        )
    if posterior_samples < 1:
        raise ValueError("posterior_samples must be >= 1")
    if seed is not None:
        om.set_seed(seed)
    predictions: list[dict[str, Any]] = []
    for pair in pairs:
        base = pair[_DIRECT]["model_input"]
        policies = [
            (_DIRECT, str(pair[_DIRECT].get("policy_version") or "1")),
            (_DELIBERATE, str(pair[_DELIBERATE].get("policy_version") or "1")),
        ]
        scored = om.score_policy_counterfactuals(
            model,
            pre,
            base,
            policies=policies,
            num_samples=posterior_samples,
            seed=None,
            device=device,
        )
        by_policy: dict[str, dict[str, float]] = {}
        for entry in scored:
            policy_id = entry.get("policy_id")
            if policy_id in _POLICIES:
                by_policy[policy_id] = {
                    "mean": float(entry.get("mean")),
                    "std": float(entry.get("std")),
                }
        if _DIRECT not in by_policy or _DELIBERATE not in by_policy:
            raise RuntimeError(
                f"policy counterfactual scoring did not return both policies for {pair['task_id']}"
            )
        predictions.append(by_policy)
    return predictions


def score_treatment_panels(
    model: Any,
    pre: om.Preprocessor,
    panels: list[dict[str, Any]],
    treatments: list[TreatmentSpec],
    *,
    posterior_samples: int = 50,
    seed: int | None = None,
    device: str = "cpu",
) -> list[dict[str, dict[str, float]]]:
    """Score every complete treatment bundle while holding task input fixed."""
    if not om.TORCH_AVAILABLE:
        raise RuntimeError(
            "scoring the neural allocator requires PyTorch, which is not installed"
        )
    if not pre.treatment_enabled:
        raise ValueError(
            "generalized treatment allocation requires a descriptor-enabled artifact"
        )
    if posterior_samples < 1:
        raise ValueError("posterior_samples must be >= 1")
    if not treatments:
        raise ValueError("at least one candidate treatment is required")
    ordered = sorted(treatments, key=lambda treatment: treatment.bundle_id)
    predictions: list[dict[str, dict[str, float]]] = []
    for panel in panels:
        first_bundle_id = ordered[0].bundle_id
        base = _panel_row(panel, first_bundle_id)["model_input"]
        scored = om.score_treatment_counterfactuals(
            model,
            pre,
            base,
            ordered,
            num_samples=posterior_samples,
            seed=seed,
            device=device,
        )
        by_bundle: dict[str, dict[str, float]] = {}
        for entry in scored:
            bundle_id = str(entry.get("bundle_id"))
            if bundle_id in by_bundle:
                raise RuntimeError(
                    f"duplicate treatment score for {bundle_id!r} on task "
                    f"{panel.get('task_id')!r}"
                )
            by_bundle[bundle_id] = {
                "mean": float(entry.get("mean")),
                "std": float(entry.get("std")),
            }
        expected = {treatment.bundle_id for treatment in ordered}
        if set(by_bundle) != expected:
            raise RuntimeError(
                f"treatment counterfactual scoring returned the wrong menu for "
                f"task {panel.get('task_id')!r}"
            )
        predictions.append(by_bundle)
    return predictions


# ---------------------------------------------------------------------------
# Output assembly (pure)
# ---------------------------------------------------------------------------


def assemble_output(
    *,
    strategies: dict[str, Any],
    metadata: dict[str, Any],
    exclusions: dict[str, Any],
    model: dict[str, Any],
    split: str,
    budget: dict[str, Any],
    cost_model: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the dashboard-compatible evaluation dict.

    Only aggregate strategy rows and metadata enter the result; per-task
    predictions, outcomes, prompts and usage dicts are never included here.
    """
    return {
        "strategies": strategies,
        "metadata": metadata,
        "exclusions": exclusions,
        "model": model,
        "cost_model": cost_model,
        "split": split,
        "budget": budget,
    }


# ---------------------------------------------------------------------------
# Orchestration (torch-dependent)
# ---------------------------------------------------------------------------


def evaluate_allocator(
    dataset_path: str | Path,
    artifact_dir: str | Path,
    *,
    split: str = "test",
    deliberate_fraction: float = 0.5,
    posterior_samples: int = 50,
    seed: int = 42,
    random_trials: int = 1000,
    bootstrap_trials: int = 1000,
    duplicate_attempts: str = "reject",
    use_cost_adjusted_uplift: bool = False,
) -> dict[str, Any]:
    """Evaluate the learned allocator on paired held-out task rows.

    Loads the deterministic JSONL dataset, pairs tasks in ``split``, fits cost
    summaries on TRAIN rows only, scores both policies through the saved
    artifacts (never using the observed split outcome or cost to choose), and
    compares the aggregate strategies.  Returns the dashboard-compatible dict.
    """
    if split not in _SPLITS:
        raise ValueError(f"split must be one of {_SPLITS}, got {split!r}")
    fraction = _validate_fraction(deliberate_fraction)
    if posterior_samples < 1:
        raise ValueError("posterior_samples must be >= 1")
    if random_trials < 1:
        raise ValueError("random_trials must be >= 1")
    if bootstrap_trials < 1:
        raise ValueError("bootstrap_trials must be >= 1")

    dataset = Path(dataset_path).expanduser().resolve()
    artifact = Path(artifact_dir).expanduser().resolve()

    rows = om.load_dataset_rows(dataset)
    pairs, exclusions = build_task_pairs(rows, split, duplicate_attempts)
    if not pairs:
        raise ValueError(
            f"no paired tasks found in split {split!r} of {dataset}; allocator "
            "evaluation requires exactly one direct and one deliberate row per task"
        )

    cost_model = fit_train_cost(rows)
    config, pre, model = om.load_artifacts(artifact)
    predictions = score_pairs(
        model, pre, pairs, posterior_samples=posterior_samples, seed=seed
    )

    uplift = [
        predictions[index][_DELIBERATE]["mean"] - predictions[index][_DIRECT]["mean"]
        for index in range(len(pairs))
    ]
    warnings: list[str] = []
    if use_cost_adjusted_uplift:
        incremental = cost_model.get("incremental_tokens_per_task")
        if incremental is not None and incremental > 0:
            uplift = [value / incremental for value in uplift]
        else:
            warnings.append(
                "cost-adjusted uplift requested but the train-estimated incremental "
                "token cost is not positive; using raw predicted success uplift"
            )

    selected_ids, k = select_deliberate_tasks(pairs, uplift, fraction)
    strategies = compare_strategies(
        pairs,
        selected_ids,
        seed=seed,
        random_trials=random_trials,
        bootstrap_trials=bootstrap_trials,
    )

    n = len(pairs)
    statistical_warning: str | None = None
    if n < _SMALL_N:
        statistical_warning = (
            f"task-bootstrap intervals are descriptive only; n={n} is too small "
            "to support any significance claim"
        )
        warnings.append(statistical_warning)

    metadata: dict[str, Any] = {
        "evaluator": "allocator_eval",
        "version": 1,
        "dataset_path": str(dataset),
        "artifact_dir": str(artifact),
        "split": split,
        "seed": seed,
        "deliberate_fraction": fraction,
        "k": k,
        "n_tasks": n,
        "posterior_samples": posterior_samples,
        "random_trials": random_trials,
        "bootstrap_trials": bootstrap_trials,
        "duplicate_attempts": duplicate_attempts,
        "cost_adjusted_uplift": use_cost_adjusted_uplift,
        "selection_rule": "rank tasks by predicted success uplift; ties by task_id",
        "observed_cost_role": "reporting only; observed cost never influences selection",
        "warnings": warnings,
        "statistical_warning": statistical_warning,
    }
    model_meta: dict[str, Any] = {
        "artifact_dir": str(artifact),
        "policies": [_DIRECT, _DELIBERATE],
        "posterior_samples": posterior_samples,
        "config": {
            "vocab_size": int(config["model"].get("vocab_size", 0)),
            "max_tokens": int(config["model"].get("max_tokens", 0)),
            "num_numeric": int(config["model"].get("num_numeric", 0)),
        },
        "training": {
            "seed": config.get("training", {}).get("seed"),
            "num_samples": config.get("training", {}).get("num_samples"),
        },
    }
    budget: dict[str, Any] = {
        "deliberate_fraction": fraction,
        "k": k,
        "n": n,
        "rule": "exactly k tasks assigned to deliberate; all others direct",
    }
    return assemble_output(
        strategies=strategies,
        metadata=metadata,
        exclusions=exclusions,
        model=model_meta,
        split=split,
        budget=budget,
        cost_model=cost_model,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_fingerprints(artifact: Path) -> dict[str, str]:
    fingerprints: dict[str, str] = {}
    for name in ("config.json", "preprocessor.json", "model.pt"):
        path = artifact / name
        if path.is_file():
            fingerprints[name] = _sha256_file(path)
    return fingerprints


def evaluate_treatment_allocator(
    dataset_path: str | Path,
    artifact_dir: str | Path,
    *,
    treatment_registry: TreatmentRegistry | str | Path,
    treatments: str | list[str] | tuple[str, ...] | None = None,
    split: str = "test",
    posterior_samples: int = 50,
    seed: int = 42,
    random_trials: int = 1000,
    bootstrap_trials: int = 1000,
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate posterior-mean argmax over an immutable treatment menu.

    The empirical value is computed only on strict complete panels.  It is
    conditional on one realized observed attempt per task/treatment cell and
    on the saved fitted artifact; it is not automatically a population-causal
    estimate.  No observed outcome or cost enters neural selection.
    """
    if split not in _SPLITS:
        raise ValueError(f"split must be one of {_SPLITS}, got {split!r}")
    if posterior_samples < 1:
        raise ValueError("posterior_samples must be >= 1")
    if random_trials < 1:
        raise ValueError("random_trials must be >= 1")
    if bootstrap_trials < 1:
        raise ValueError("bootstrap_trials must be >= 1")

    dataset = Path(dataset_path).expanduser().resolve()
    artifact = Path(artifact_dir).expanduser().resolve()
    if isinstance(treatment_registry, TreatmentRegistry):
        registry = treatment_registry
        registry_path: str | None = None
    else:
        registry_file = Path(treatment_registry).expanduser().resolve()
        registry = TreatmentRegistry.load(registry_file)
        registry_path = str(registry_file)
    candidates = resolve_treatment_candidates(registry, treatments)

    rows = om.load_dataset_rows(dataset)
    panels, exclusions = build_treatment_panels(
        rows,
        split,
        candidates,
        registry.registry_hash,
    )
    if not panels:
        raise ValueError(
            f"no complete treatment panels found in split {split!r} of {dataset}"
        )

    config, pre, model = om.load_artifacts(artifact, device=device)
    if not pre.treatment_enabled:
        raise ValueError(
            "generalized treatment allocation requires an artifact trained with "
            "model_input.treatment descriptors"
        )
    predictions = score_treatment_panels(
        model,
        pre,
        panels,
        candidates,
        posterior_samples=posterior_samples,
        seed=seed,
        device=device,
    )
    neural_assignment = select_treatment_argmax(panels, predictions, candidates)
    strategies = compare_treatment_strategies(
        panels,
        neural_assignment,
        candidates,
        seed=seed,
        random_trials=random_trials,
        bootstrap_trials=bootstrap_trials,
    )

    n = len(panels)
    warnings: list[str] = [
        "values are realized complete-panel estimates conditional on one observed "
        "attempt per task/treatment cell; they are not population-causal estimates"
    ]
    statistical_warning: str | None = None
    if n < _SMALL_N:
        statistical_warning = (
            f"task-bootstrap intervals are descriptive only; n={n} is too small "
            "to support any significance claim"
        )
        warnings.append(statistical_warning)
    if split == "validation":
        warnings.append(
            "validation is a tuning split because training may use it for early stopping"
        )

    bundle_vocab = pre.treatment_cat_vocab.get("bundle_id", {})
    candidate_rows: list[dict[str, Any]] = []
    unseen: list[str] = []
    for treatment in candidates:
        train_n = sum(
            1
            for row in rows
            if row.get("split") == "train"
            and str(row.get("policy_id")) == treatment.id
            and str(row.get("policy_version")) == treatment.version
            and row.get("treatment_bundle_hash") == treatment.bundle_hash
        )
        seen_in_training = treatment.bundle_id in bundle_vocab
        if not seen_in_training:
            unseen.append(treatment.bundle_id)
        candidate_rows.append(
            {
                "policy_id": treatment.id,
                "policy_version": treatment.version,
                "bundle_id": treatment.bundle_id,
                "bundle_hash": treatment.bundle_hash,
                "train_n": train_n,
                "seen_in_training": seen_in_training,
            }
        )
    if unseen:
        warnings.append(
            "candidate bundles unseen by the fitted categorical vocabulary are "
            f"descriptor-based extrapolations: {unseen}"
        )

    metadata: dict[str, Any] = {
        "evaluator": "allocator_eval",
        "mode": "generalized_treatments",
        "schema_version": 2,
        "dataset_path": str(dataset),
        "dataset_sha256": _sha256_file(dataset),
        "artifact_dir": str(artifact),
        "split": split,
        "seed": seed,
        "n_tasks": n,
        "posterior_samples": posterior_samples,
        "random_trials": random_trials,
        "bootstrap_trials": bootstrap_trials,
        "estimand": "realized complete-panel success value",
        "selection_rule": "maximize saved posterior predictive mean per task",
        "tie_break": "lexicographic bundle_id",
        "observed_cost_role": "reporting only; observed cost never influences selection",
        "uncertainty_scope": (
            "task bootstrap conditional on the fitted artifact; random baseline "
            "quantiles describe assignment randomization"
        ),
        "warnings": warnings,
        "statistical_warning": statistical_warning,
    }
    model_meta: dict[str, Any] = {
        "artifact_dir": str(artifact),
        "artifact_sha256": _artifact_fingerprints(artifact),
        "posterior_samples": posterior_samples,
        "treatment_encoder_enabled": pre.treatment_enabled,
        "config": {
            "vocab_size": int(config["model"].get("vocab_size", 0)),
            "max_tokens": int(config["model"].get("max_tokens", 0)),
            "num_numeric": int(config["model"].get("num_numeric", 0)),
            "num_treatment_numeric": int(
                config["model"].get("num_treatment_numeric", 0)
            ),
        },
        "training": {
            "seed": config.get("training", {}).get("seed"),
            "num_samples": config.get("training", {}).get("num_samples"),
            "feature_schema_version": config.get("training", {}).get(
                "feature_schema_version"
            ),
        },
    }
    treatment_set: dict[str, Any] = {
        "registry_path": registry_path,
        "registry_hash": registry.registry_hash,
        "candidate_set_hash": treatment_candidate_set_hash(candidates),
        "candidates": candidate_rows,
    }
    budget: dict[str, Any] = {
        "type": "none",
        "objective": "maximize predicted success",
        "observed_cost_used_for_selection": False,
    }
    return {
        "strategies": strategies,
        "metadata": metadata,
        "exclusions": exclusions,
        "model": model_meta,
        "treatment_set": treatment_set,
        "cost_model": {
            "role": "not used for generalized selection; observed costs are reporting only"
        },
        "split": split,
        "budget": budget,
    }


def write_evaluation(result: dict[str, Any], output_path: str | Path) -> dict[str, Any]:
    """Atomically write an evaluation dict as sorted, deterministic JSON."""
    if not isinstance(result, dict) or "strategies" not in result:
        raise ValueError("evaluation result must be a dict with a 'strategies' key")
    path = Path(output_path).expanduser().resolve()
    write_json(path, result)
    metadata = result.get("metadata") or {}
    return {
        "output_path": str(path),
        "n_tasks": metadata.get("n_tasks"),
        "strategies": list(sorted(result.get("strategies", {}))),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyreplab-harness-allocator-eval",
        description=(
            "Evaluate a learned allocator on paired legacy rows or strict "
            "complete generalized-treatment panels without causal leakage."
        ),
    )
    parser.add_argument("dataset", help="deterministic JSONL dataset from dataset.py")
    parser.add_argument("artifact_dir", help="artifact directory produced by outcome_model train")
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="output JSON path (omitting it prints the full result to stdout)",
    )
    parser.add_argument(
        "--treatment-registry",
        default=None,
        help="immutable registry enabling generalized complete-panel evaluation",
    )
    parser.add_argument(
        "--treatments",
        default=None,
        help="comma-separated registry references or 'all' (default: all)",
    )
    parser.add_argument("--split", default="test", help="evaluation split (default: test)")
    parser.add_argument("--deliberate-fraction", type=float, default=0.5)
    parser.add_argument("--posterior-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-trials", type=int, default=1000)
    parser.add_argument("--bootstrap-trials", type=int, default=1000)
    parser.add_argument(
        "--duplicate-attempts",
        choices=["reject", "first"],
        default="reject",
        help="how to handle duplicate attempts per task and policy",
    )
    parser.add_argument(
        "--cost-adjusted-uplift",
        action="store_true",
        help="divide predicted uplift by the positive train-estimated incremental cost",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    try:
        if args.treatments and not args.treatment_registry:
            raise ValueError("--treatments requires --treatment-registry")
        if args.treatment_registry:
            def option_present(name: str) -> bool:
                return any(
                    token == name or token.startswith(f"{name}=")
                    for token in raw_argv
                )

            incompatible = [
                name
                for name in ("--deliberate-fraction", "--cost-adjusted-uplift", "--duplicate-attempts")
                if option_present(name)
            ]
            if incompatible:
                raise ValueError(
                    "generalized treatment mode does not accept legacy options: "
                    + ", ".join(incompatible)
                )
            result = evaluate_treatment_allocator(
                args.dataset,
                args.artifact_dir,
                treatment_registry=args.treatment_registry,
                treatments=args.treatments,
                split=args.split,
                posterior_samples=args.posterior_samples,
                seed=args.seed,
                random_trials=args.random_trials,
                bootstrap_trials=args.bootstrap_trials,
            )
        else:
            result = evaluate_allocator(
                args.dataset,
                args.artifact_dir,
                split=args.split,
                deliberate_fraction=args.deliberate_fraction,
                posterior_samples=args.posterior_samples,
                seed=args.seed,
                random_trials=args.random_trials,
                bootstrap_trials=args.bootstrap_trials,
                duplicate_attempts=args.duplicate_attempts,
                use_cost_adjusted_uplift=args.cost_adjusted_uplift,
            )
        if args.output:
            summary = write_evaluation(result, args.output)
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            print(json.dumps(result, indent=2, sort_keys=True))
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "assemble_output",
    "build_parser",
    "build_task_pairs",
    "build_treatment_panels",
    "compare_strategies",
    "compare_treatment_strategies",
    "evaluate_allocator",
    "evaluate_treatment_allocator",
    "fit_train_cost",
    "main",
    "pair_messages",
    "pair_outcome",
    "pair_tokens",
    "pair_tool_calls",
    "random_mix_aggregate",
    "resolve_treatment_candidates",
    "score_pairs",
    "score_treatment_panels",
    "select_deliberate_tasks",
    "select_treatment_argmax",
    "task_bootstrap_ci",
    "task_tokens",
    "treatment_candidate_set_hash",
    "treatment_task_bootstrap_ci",
    "uniform_random_treatment_aggregate",
    "write_evaluation",
]
