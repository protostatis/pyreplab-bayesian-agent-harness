"""Calibration context builder and split utilities for meta-policy learning.

Follows the M3 preregistration calibration protocol (Section 6):
- Frozen ordered panels with nested prefixes (k=4 subset of k=8 subset of k=16)
- Context statistics normalized using meta-training data only
- Leakage audit forbids page text, URLs, selectors, answers, verifier diagnostics,
  and policy identity from entering calibration context
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import Any

# ---------------------------------------------------------------------------
# Forbidden field names for leakage audit
# ---------------------------------------------------------------------------

_FORBIDDEN_PATTERNS = (
    "page_text",
    "dom_string",
    "url",
    "domain",
    "selector",
    "answer",
    "final_answer",
    "verifier_diag",
    "verifier_diagnostic",
    "expected_target",
    "hidden_test",
    "policy_id",
    "policy_version",
    "bundle_id",
    "bundle_hash",
    "registry_position",
    "policy_identity",
)


def _contains_forbidden(value: Any, path: str = "") -> list[str]:
    """Recursively check for forbidden field names. Returns list of violations."""
    violations: list[str] = []
    if isinstance(value, dict):
        for key in value:
            full_path = f"{path}.{key}" if path else str(key)
            for pattern in _FORBIDDEN_PATTERNS:
                if pattern in str(key).lower():
                    violations.append(f"{full_path} (matches forbidden pattern '{pattern}')")
            violations.extend(_contains_forbidden(value[key], full_path))
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            violations.extend(_contains_forbidden(item, f"{path}[{i}]"))
    return violations


def audit_context_leakage(context: dict[str, Any]) -> list[str]:
    """Audit a calibration context for forbidden fields.

    Returns a list of violation descriptions. An empty list means the context
    is clean — no page text, URLs, selectors, answers, verifier diagnostics,
    or policy identity have leaked in.

    Args:
        context: calibration context dict or row dict.

    Returns:
        list of violation strings (empty if clean).
    """
    return _contains_forbidden(context)


# ---------------------------------------------------------------------------
# Calibration context building
# ---------------------------------------------------------------------------


def build_calibration_context(
    policy_rows: list[dict[str, Any]],
    k: int,
    normalization_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a calibration context tensor from the first k rows of a frozen
    ordered panel.

    Each calibration row is encoded as:
        (task_features, outcome, termination, cost, mask)

    Args:
        policy_rows: ordered list of rows for one policy (first k are used).
        k: number of calibration rows to include (0 <= k <= len(policy_rows)).
        normalization_stats: dict with "cost_mean", "cost_std" fitted on
            meta-training ``log1p(cost)`` values. If None, returns log1p(cost)
            without standardization.

    Returns:
        dict with keys "success", "term_onehot", "cost", "mask",
        "task_feature_vectors", "k_actual".
    """
    if k < 0:
        raise ValueError(f"k must be nonnegative, got {k}")
    k_actual = min(k, len(policy_rows))
    if k_actual == 0:
        return {
            "success": [],
            "term_onehot": [],
            "cost": [],
            "mask": [],
            "task_feature_vectors": [],
            "k_actual": 0,
        }

    cost_mean = float(normalization_stats.get("cost_mean", 0.0)) if normalization_stats else 0.0
    cost_std = float(normalization_stats.get("cost_std", 1.0)) if normalization_stats else 1.0
    if cost_std <= 0:
        cost_std = 1.0

    success_list: list[float] = []
    term_list: list[list[float]] = []
    cost_list: list[float] = []
    mask_list: list[float] = []
    task_emb_list: list[Any] = []

    for i in range(k):
        if i < len(policy_rows):
            row = policy_rows[i]
            success_list.append(1.0 if row.get("verified_success", False) else 0.0)
            if "cost" in row:
                cost_val = float(row["cost"])
                if "output_token_cost" in row and cost_val != float(
                    row["output_token_cost"]
                ):
                    raise ValueError(
                        f"calibration row {i} has conflicting cost fields"
                    )
            elif "output_token_cost" in row:
                cost_val = float(row["output_token_cost"])
            else:
                raise ValueError(f"calibration row {i} is missing cost")
            if not math.isfinite(cost_val) or cost_val < 0.0:
                raise ValueError(f"calibration row {i} has invalid cost {cost_val!r}")
            cost_list.append((math.log1p(cost_val) - cost_mean) / cost_std)
            mask_list.append(1.0)

            # Termination one-hot (6 classes).
            term_class = _classify_termination(row)
            term_list.append(term_class)

            # Task features (precomputed embedding or structured features).
            task_emb = row.get("task_embedding", None)
            if task_emb is None:
                # Try nested M3/CNP location: model_input.task.task_embedding.vector
                mi = row.get("model_input", {})
                if isinstance(mi, dict):
                    task_info = mi.get("task", {})
                    if isinstance(task_info, dict):
                        emb_info = task_info.get("task_embedding", {})
                        if isinstance(emb_info, dict):
                            task_emb = emb_info.get("vector", None)
            task_emb_list.append(task_emb)
        else:
            success_list.append(0.0)
            cost_list.append(0.0)
            mask_list.append(0.0)
            term_list.append([0.0] * 6)
            task_emb_list.append(None)

    return {
        "success": success_list,
        "term_onehot": term_list,
        "cost": cost_list,
        "mask": mask_list,
        "task_feature_vectors": task_emb_list,
        "k_actual": k_actual,
    }


def _classify_termination(row: dict[str, Any]) -> list[float]:
    """Encode termination class as 6-dim one-hot.

    Classes: normal_completion(0), tool_call_limit(1), wall_timeout(2),
    invalid_or_tool_error(3), model_runtime_failure(4),
    verifier_declared_unsuccessful(5).
    """
    onehot = [0.0] * 6
    term = str(row.get("termination_class", row.get("failure_code", "normal_completion")))
    mapping = {
        "normal_completion": 0,
        "tool_call_limit": 1,
        "wall_timeout": 2,
        "invalid_or_tool_error": 3,
        "model_runtime_failure": 4,
        "verifier_declared_unsuccessful": 5,
    }
    idx = mapping.get(term, 0)
    onehot[idx] = 1.0
    return onehot


# ---------------------------------------------------------------------------
# Frozen calibration splits
# ---------------------------------------------------------------------------


def frozen_calibration_split(
    tasks: list[str],
    k_sizes: tuple[int, ...] = (0, 4, 8, 16),
    seed: int = 42,
) -> dict[str, Any]:
    """Create a deterministic ordered task selection for calibration.

    Nested prefixes: k=4 is the prefix of k=8 is the prefix of k=16.

    Args:
        tasks: list of task identifiers (strings).
        k_sizes: tuple of k values, must be non-decreasing.
        seed: deterministic seed.

    Returns:
        dict with ordered task list and per-k subsets.
    """
    if not tasks:
        raise ValueError("tasks list must be non-empty")
    if not k_sizes:
        raise ValueError("k_sizes must not be empty")
    if any(k < 0 for k in k_sizes):
        raise ValueError(f"k_sizes must be nonnegative: {k_sizes}")

    for i in range(1, len(k_sizes)):
        if k_sizes[i] < k_sizes[i - 1]:
            raise ValueError(f"k_sizes must be non-decreasing: {k_sizes}")

    if len(tasks) != len(set(tasks)):
        raise ValueError("calibration tasks must be unique")

    max_k = max(k_sizes)
    if len(tasks) < max_k:
        raise ValueError(
            f"need at least {max_k} unique calibration tasks, got {len(tasks)}"
        )

    rng = random.Random(seed)
    ordered = list(tasks)
    rng.shuffle(ordered)

    ordered = ordered[:max_k]

    result: dict[str, Any] = {"ordered_tasks": ordered, "seed": seed}
    for k in k_sizes:
        result[f"k_{k}"] = ordered[:k]

    return result


# ---------------------------------------------------------------------------
# Task and policy splits
# ---------------------------------------------------------------------------


def policy_task_split(
    tasks: list[str],
    policies: list[str],
    ratios: dict[str, float],
    seed: int,
) -> dict[str, Any]:
    """Split tasks and policies into meta_train, dev, and final subsets.

    No task or policy appears in multiple splits.

    Args:
        tasks: list of task IDs.
        policies: list of policy IDs.
        ratios: dict with keys "task_meta_train", "task_dev_cal", "task_dev_target",
            "task_final_cal", "task_final_known", "task_final_held",
            "policy_meta_train", "policy_development", "policy_final".
            Values are fractions summing to <= 1 per dimension.
        seed: deterministic seed.

    Returns:
        dict with split assignments.
    """
    required_keys = {
        "task_meta_train", "task_dev_cal", "task_dev_target",
        "task_final_cal", "task_final_known", "task_final_held",
        "policy_meta_train", "policy_development", "policy_final",
    }
    missing = required_keys - set(ratios)
    if missing:
        raise ValueError(f"missing ratio keys: {missing}")
    if len(tasks) != len(set(tasks)):
        raise ValueError("tasks must be unique")
    if len(policies) != len(set(policies)):
        raise ValueError("policies must be unique")

    rng = random.Random(seed)

    # Split tasks.
    shuffled_tasks = list(tasks)
    rng.shuffle(shuffled_tasks)

    n_tasks = len(tasks)
    task_keys = [
        "task_meta_train", "task_dev_cal", "task_dev_target",
        "task_final_cal", "task_final_known", "task_final_held",
    ]
    if n_tasks < len(task_keys):
        raise ValueError(
            f"need at least {len(task_keys)} tasks for non-empty task splits, "
            f"got {n_tasks}"
        )
    if any(
        not math.isfinite(float(ratios[key])) or float(ratios[key]) <= 0
        for key in task_keys
    ):
        raise ValueError("task split ratios must be finite and positive")
    task_counts = {
        key: max(1, int(n_tasks * ratios[key] / sum(ratios[k] for k in task_keys)))
        for key in task_keys
    }

    # Adjust to sum to n_tasks.
    total = sum(task_counts.values())
    if total < n_tasks:
        task_counts["task_meta_train"] += n_tasks - total
    elif total > n_tasks:
        task_counts["task_meta_train"] -= total - n_tasks
    if sum(task_counts.values()) != n_tasks or any(
        count < 1 for count in task_counts.values()
    ):
        raise ValueError("task ratios cannot produce non-empty splits")

    task_splits: dict[str, list[str]] = {}
    offset = 0
    for key in task_keys:
        count = min(task_counts[key], len(shuffled_tasks) - offset)
        task_splits[key] = shuffled_tasks[offset : offset + count]
        offset += count

    # Split policies.
    shuffled_policies = list(policies)
    rng.shuffle(shuffled_policies)

    n_policies = len(policies)
    policy_keys = ["policy_meta_train", "policy_development", "policy_final"]
    if n_policies < len(policy_keys):
        raise ValueError(
            f"need at least {len(policy_keys)} policies for non-empty policy splits, "
            f"got {n_policies}"
        )
    if any(
        not math.isfinite(float(ratios[key])) or float(ratios[key]) <= 0
        for key in policy_keys
    ):
        raise ValueError("policy split ratios must be finite and positive")
    policy_counts = {
        key: max(1, int(n_policies * ratios[key] / sum(ratios[k] for k in policy_keys)))
        for key in policy_keys
    }
    total_p = sum(policy_counts.values())
    if total_p < n_policies:
        policy_counts["policy_meta_train"] += n_policies - total_p
    elif total_p > n_policies:
        policy_counts["policy_meta_train"] -= total_p - n_policies
    if sum(policy_counts.values()) != n_policies or any(
        count < 1 for count in policy_counts.values()
    ):
        raise ValueError("policy ratios cannot produce non-empty splits")

    policy_splits: dict[str, list[str]] = {}
    offset_p = 0
    for key in policy_keys:
        count = min(policy_counts[key], len(shuffled_policies) - offset_p)
        policy_splits[key] = shuffled_policies[offset_p : offset_p + count]
        offset_p += count

    # Verify every unique input was assigned exactly once.
    all_tasks: list[str] = []
    for key in task_keys:
        all_tasks.extend(task_splits.get(key, []))
    all_policies = [
        policy
        for key in policy_keys
        for policy in policy_splits.get(key, [])
    ]
    if len(all_tasks) != n_tasks or len(set(all_tasks)) != n_tasks:
        raise RuntimeError("task split assignment is incomplete or overlapping")
    if len(all_policies) != n_policies or len(set(all_policies)) != n_policies:
        raise RuntimeError("policy split assignment is incomplete or overlapping")

    return {
        "tasks": task_splits,
        "policies": policy_splits,
        "seed": seed,
        "ratios": ratios,
    }


# ---------------------------------------------------------------------------
# Normalization statistics
# ---------------------------------------------------------------------------


def fit_normalization_stats(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fit log-cost normalization statistics on meta-training data only.

    ``cost_mean`` and ``cost_std`` describe ``log1p(raw_cost)``. Returns a
    ``cost_transform`` marker so callers cannot mistake them for raw-cost
    moments.
    """
    cost_values: list[float] = []
    for row_index, row in enumerate(rows):
        if "cost" in row:
            cost = float(row["cost"])
            if "output_token_cost" in row and cost != float(
                row["output_token_cost"]
            ):
                raise ValueError(
                    f"normalization row {row_index} has conflicting cost fields"
                )
        elif "output_token_cost" in row:
            cost = float(row["output_token_cost"])
        else:
            continue
        if math.isfinite(cost) and cost >= 0.0:
            cost_values.append(math.log1p(cost))

    if not cost_values:
        return {
            "cost_mean": 0.0,
            "cost_std": 1.0,
            "cost_transform": "log1p_zscore",
            "n_rows": 0,
        }

    n = len(cost_values)
    mean = sum(cost_values) / n
    if n > 1:
        variance = sum((c - mean) ** 2 for c in cost_values) / (n - 1)
    else:
        variance = 0.0
    std = math.sqrt(variance) if variance > 0 else 1.0

    return {
        "cost_mean": mean,
        "cost_std": std,
        "cost_transform": "log1p_zscore",
        "n_rows": n,
    }


__all__ = [
    "build_calibration_context",
    "frozen_calibration_split",
    "policy_task_split",
    "audit_context_leakage",
    "fit_normalization_stats",
]
