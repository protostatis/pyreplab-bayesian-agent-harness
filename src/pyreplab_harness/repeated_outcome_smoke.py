"""Repeated-outcome two-arm smoke analysis.

The module implements a frozen protocol for task-level repeated outcomes where each
task has multiple logical trials per arm. It performs deterministic split-and-hold
evaluation of per-task selection rules against a calibrated global fixed arm.

Usage:
    python -m pyreplab_harness.repeated_outcome_smoke INPUT_JSONL OUTPUT_JSON \
        --arm-a <arm-a> --arm-b <arm-b> --model <model>
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from .io_utils import write_json


SCHEMA_VERSION = "repeated-outcome-smoke-v1"


def trial_hash(seed: int, task_name: str, arm: str, trial_name: str) -> str:
    """Deterministic outcome-blind ordering key for a logical trial.

    Ordered by SHA-256 of `seed|task_name|arm|trial_name` as specified in the
    protocol.
    """

    message = f"{seed}|{task_name}|{arm}|{trial_name}"
    return hashlib.sha256(message.encode("utf-8")).hexdigest()


def _input_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line_no} is not JSON: {error}") from error
            if not isinstance(item, dict):
                raise ValueError(f"line {line_no} is not an object")
            rows.append(item)
    return rows


def _coerce_reward(raw_reward: Any, line_no: int | None = None) -> int:
    if not isinstance(raw_reward, int) or isinstance(raw_reward, bool) or raw_reward not in (0, 1):
        label = f" on line {line_no}" if line_no is not None else ""
        raise ValueError(f"reward must be 0 or 1{label}")
    return int(raw_reward)


def _coerce_str(value: Any, label: str, row_no: int | None = None) -> str:
    if not isinstance(value, str):
        position = f" on row {row_no}" if row_no is not None else ""
        raise ValueError(f"{label} must be a string{position}")
    if not value:
        position = f" on row {row_no}" if row_no is not None else ""
        raise ValueError(f"{label} must be non-empty{position}")
    return value


def filter_rows(
    rows: Sequence[dict[str, Any]],
    *,
    model: str,
    arm_a: str,
    arm_b: str,
) -> list[dict[str, Any]]:
    """Filter exact model and the two requested arms.

    Raises on malformed required fields.
    """

    requested_arms = {arm_a, arm_b}
    out: list[dict[str, Any]] = []

    for row_no, row in enumerate(rows, 1):
        model_name = _coerce_str(row.get("model"), "model", row_no)
        if model_name != model:
            continue

        arm = _coerce_str(row.get("agent"), "agent", row_no)
        if arm not in requested_arms:
            continue

        reward = _coerce_reward(row.get("reward"), row_no)
        task_name = _coerce_str(row.get("task_name"), "task_name", row_no)
        trial_name = _coerce_str(row.get("trial_name"), "trial_name", row_no)

        out.append(
            {
                "task_name": task_name,
                "agent": arm,
                "model": model_name,
                "reward": reward,
                "trial_name": trial_name,
            }
        )
    return out


def deduplicate_trials(
    rows: Sequence[dict[str, Any]],
    *,
    arm_a: str,
    arm_b: str,
) -> tuple[list[dict[str, Any]], int]:
    """Deduplicate logical trials by (agent, task_name, trial_name).

    Conflicting duplicates are a hard error.

    Returns the deduplicated rows (sorted for deterministic output) and the number
    of duplicate rows removed.
    """

    canonical: dict[tuple[str, str, str], int] = {}
    duplicates = 0

    for row in rows:
        arm = row["agent"]
        if arm not in {arm_a, arm_b}:
            raise ValueError(f"unexpected arm {arm!r}")

        key = (row["agent"], row["task_name"], row["trial_name"])
        reward = int(row["reward"])
        if key not in canonical:
            canonical[key] = reward
            continue
        if canonical[key] != reward:
            raise ValueError(
                f"conflicting duplicates for (agent={key[0]!r}, task_name={key[1]!r}, "
                f"trial_name={key[2]!r}): {canonical[key]!r} != {reward!r}"
            )
        duplicates += 1

    deduped = [
        {
            "agent": agent,
            "task_name": task_name,
            "trial_name": trial_name,
            "reward": reward,
        }
        for (agent, task_name, trial_name), reward in canonical.items()
    ]
    deduped.sort(key=lambda item: (item["task_name"], item["agent"], item["trial_name"]))
    return deduped, duplicates


def _ordered_task_trials(
    rows: Sequence[dict[str, Any]],
    *,
    seed: int,
    task_name: str,
    arm: str,
) -> list[int]:
    ordered = sorted(
        rows,
        key=lambda row: (trial_hash(seed, task_name, arm, row["trial_name"]), row["trial_name"]),
    )
    return [int(row["reward"]) for row in ordered]


def build_complete_task_panels(
    deduped_rows: Sequence[dict[str, Any]],
    *,
    repeats: int,
    seed: int,
    arm_a: str,
    arm_b: str,
) -> tuple[list[tuple[str, list[int], list[int]]], dict[str, int]]:
    """Retain only tasks with exactly `repeats` logical trials for both arms.

    Returns:
        - complete task panels as (task_name, arm_a_rewards, arm_b_rewards)
        - exclusion counters keyed by reason
    """

    by_task: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {arm_a: [], arm_b: []})
    for row in deduped_rows:
        by_task[str(row["task_name"])][str(row["agent"])].append(row)

    complete: list[tuple[str, list[int], list[int]]] = []
    exclusions = {"missing_arm": 0, "wrong_repeat_count": 0}

    for task_name, by_arm in by_task.items():
        rows_a = by_arm.get(arm_a, [])
        rows_b = by_arm.get(arm_b, [])

        if len(rows_a) != repeats or len(rows_b) != repeats:
            if len(rows_a) != repeats and len(rows_b) != repeats:
                if len(rows_a) == 0 or len(rows_b) == 0:
                    exclusions["missing_arm"] += 1
                else:
                    exclusions["wrong_repeat_count"] += 1
            elif len(rows_a) != repeats or len(rows_b) != repeats:
                if len(rows_a) == 0 or len(rows_b) == 0:
                    exclusions["missing_arm"] += 1
                else:
                    exclusions["wrong_repeat_count"] += 1
            continue

        ordered_a = _ordered_task_trials(rows_a, seed=seed, task_name=task_name, arm=arm_a)
        ordered_b = _ordered_task_trials(rows_b, seed=seed, task_name=task_name, arm=arm_b)
        complete.append((task_name, ordered_a, ordered_b))

    return complete, exclusions


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    """Compute a quantile with linear interpolation using only stdlib tools."""

    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])

    sorted_values = sorted(float(v) for v in values)
    if q <= 0.0:
        return sorted_values[0]
    if q >= 1.0:
        return sorted_values[-1]

    position = q * (len(sorted_values) - 1)
    left = math.floor(position)
    right = math.ceil(position)
    if left == right:
        return sorted_values[left]
    weight = position - left
    return sorted_values[left] * (1.0 - weight) + sorted_values[right] * weight


def analyze_position_split(
    task_panels: Sequence[tuple[str, list[int], list[int]]],
    *,
    repeats: int,
    calibration_positions_a: tuple[int, ...],
    calibration_positions_b: tuple[int, ...],
) -> dict[str, Any]:
    """Analyze one (calibration, held) split for all tasks.

    calibration_positions_a and calibration_positions_b are the 0-based
    trial indices used for calibration for arm A and arm B respectively.
    Held complements are arm-specific, so the Cartesian product of arm-A
    and arm-B combos is evaluated independently.
    """

    calibration_a = set(calibration_positions_a)
    calibration_b = set(calibration_positions_b)
    held_positions_a = [idx for idx in range(repeats) if idx not in calibration_a]
    held_positions_b = [idx for idx in range(repeats) if idx not in calibration_b]

    selector_a_alloc = 0.0
    tie_count = 0
    sign_num = 0
    sign_den = 0
    selector_success = 0.0
    fixed_cal_a = 0
    fixed_cal_b = 0
    held_a = 0.0
    held_b = 0.0
    oracle_success = 0.0

    task_count = len(task_panels)
    for _, rewards_a, rewards_b in task_panels:
        cal_a = sum(rewards_a[pos] for pos in calibration_positions_a)
        cal_b = sum(rewards_b[pos] for pos in calibration_positions_b)
        hold_a_val = sum(rewards_a[pos] for pos in held_positions_a)
        hold_b_val = sum(rewards_b[pos] for pos in held_positions_b)

        cal_diff = cal_a - cal_b
        hold_diff = hold_a_val - hold_b_val
        if cal_a > cal_b:
            selector = 1.0
        elif cal_a < cal_b:
            selector = 0.0
        else:
            selector = 0.5

        selector_a_alloc += selector
        if selector == 0.5:
            tie_count += 1
        selector_success += selector * hold_a_val + (1.0 - selector) * hold_b_val

        if cal_diff and hold_diff:
            sign_den += 1
            sign_num += 1 if (cal_diff > 0) == (hold_diff > 0) else 0

        fixed_cal_a += cal_a
        fixed_cal_b += cal_b
        held_a += hold_a_val
        held_b += hold_b_val

        if hold_a_val > hold_b_val:
            oracle_success += hold_a_val
        elif hold_a_val < hold_b_val:
            oracle_success += hold_b_val
        else:
            oracle_success += 0.5 * (hold_a_val + hold_b_val)

    held_per_task = repeats - len(calibration_positions_a)
    held_trials = task_count * held_per_task
    if held_trials == 0:
        selector_rate = 0.0
        fixed_rate = 0.0
        always_a_rate = 0.0
        always_b_rate = 0.0
        oracle_rate = 0.0
    else:
        selector_rate = selector_success / held_trials
        always_a_rate = held_a / held_trials
        always_b_rate = held_b / held_trials
        oracle_rate = oracle_success / held_trials

    if fixed_cal_a > fixed_cal_b:
        fixed_prob_a = 1.0
    elif fixed_cal_a < fixed_cal_b:
        fixed_prob_a = 0.0
    else:
        fixed_prob_a = 0.5
    fixed_success = fixed_prob_a * held_a + (1.0 - fixed_prob_a) * held_b
    fixed_rate = fixed_success / held_trials if held_trials else 0.0

    return {
        "calibration_positions_a": list(calibration_positions_a),
        "calibration_positions_b": list(calibration_positions_b),
        "held_repetitions": held_per_task,
        "selector": {
            "held_success": selector_success,
            "held_success_rate": selector_rate,
        },
        "calibrated_fixed": {
            "held_success": fixed_success,
            "held_success_rate": fixed_rate,
            "calibration_alloc_a": fixed_prob_a,
            "calibration_alloc_b": 1.0 - fixed_prob_a,
        },
        "always_arm_a": {
            "held_success": held_a,
            "held_success_rate": always_a_rate,
        },
        "always_arm_b": {
            "held_success": held_b,
            "held_success_rate": always_b_rate,
        },
        "oracle": {
            "held_success": oracle_success,
            "held_success_rate": oracle_rate,
            "per_task_fraction": 0.0,
        },
        "allocation": {
            "task_alloc_a": selector_a_alloc,
            "task_alloc_b": task_count - selector_a_alloc,
            "task_alloc_a_rate": selector_a_alloc / task_count if task_count else 0.0,
            "task_alloc_b_rate": (task_count - selector_a_alloc) / task_count if task_count else 0.0,
            "tie_rate": tie_count / task_count if task_count else 0.0,
            "task_count": task_count,
        },
        "sign_concordance": {
            "matches": sign_num,
            "denominator": sign_den,
            "rate": sign_num / sign_den if sign_den else 0.0,
        },
        "lift": {
            "selector_minus_fixed": selector_rate - fixed_rate,
        },
    }


def analyze_k(
    task_panels: Sequence[tuple[str, list[int], list[int]]],
    *,
    repeats: int,
    k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Average split metrics for a fixed k across all position combinations.

    Uses the Cartesian product of arm-A calibration combos and arm-B
    calibration combos (C(R,k)^2 pairings) because each arm has
    independently collected trials with no shared seed/pairing.
    """

    if not task_panels:
        return {
            "k": k,
            "position_combinations": 0,
            "held_repetitions": repeats - k,
            "selector": {"held_success_rate": 0.0},
            "calibrated_fixed": {"held_success_rate": 0.0},
            "lift": {"selector_minus_fixed": 0.0},
            "always_arm_a": {"held_success_rate": 0.0},
            "always_arm_b": {"held_success_rate": 0.0},
            "oracle": {"held_success_rate": 0.0},
            "allocation": {"task_alloc_a_rate": 0.0, "task_alloc_b_rate": 0.0, "tie_rate": 0.0, "task_count": 0},
            "sign_concordance": {"matches": 0, "denominator": 0, "rate": 0.0},
        }, []

    combos = tuple(itertools.combinations(range(repeats), k))
    split_results = [
        analyze_position_split(task_panels, repeats=repeats,
                               calibration_positions_a=combo_a,
                               calibration_positions_b=combo_b)
        for combo_a in combos
        for combo_b in combos
    ]

    avg = {
        "k": k,
        "position_combinations": len(split_results),
        "held_repetitions": repeats - k,
        "selector": {
            "held_success": _mean(sr["selector"]["held_success"] for sr in split_results),
            "held_success_rate": _mean(sr["selector"]["held_success_rate"] for sr in split_results),
        },
        "calibrated_fixed": {
            "held_success": _mean(
                sr["calibrated_fixed"]["held_success"] for sr in split_results
            ),
            "held_success_rate": _mean(
                sr["calibrated_fixed"]["held_success_rate"] for sr in split_results
            ),
        },
        "lift": {
            "selector_minus_fixed": _mean(
                sr["lift"]["selector_minus_fixed"] for sr in split_results
            ),
        },
        "always_arm_a": {
            "held_success_rate": _mean(
                sr["always_arm_a"]["held_success_rate"] for sr in split_results
            ),
        },
        "always_arm_b": {
            "held_success_rate": _mean(
                sr["always_arm_b"]["held_success_rate"] for sr in split_results
            ),
        },
        "oracle": {
            "held_success_rate": _mean(sr["oracle"]["held_success_rate"] for sr in split_results),
        },
        "allocation": {
            "arm_a": _mean(sr["allocation"]["task_alloc_a_rate"] for sr in split_results),
            "arm_b": _mean(sr["allocation"]["task_alloc_b_rate"] for sr in split_results),
            "tie_rate": _mean(sr["allocation"]["tie_rate"] for sr in split_results),
            "task_count": split_results[0]["allocation"]["task_count"] if split_results else 0,
            "denominator": split_results[0]["allocation"]["task_count"] if split_results else 0,
        },
        "sign_concordance": {},
    }
    sign_matches = sum(sr["sign_concordance"]["matches"] for sr in split_results)
    sign_denominator = sum(
        sr["sign_concordance"]["denominator"] for sr in split_results
    )
    avg["sign_concordance"] = {
        "matches": sign_matches,
        "denominator": sign_denominator,
        "rate": sign_matches / sign_denominator if sign_denominator else 0.0,
    }
    return avg, list(split_results)


def bootstrap_task_lift_ci(
    task_panels: Sequence[tuple[str, list[int], list[int]]],
    *,
    repeats: int,
    k: int,
    bootstrap_trials: int,
    seed: int,
) -> dict[str, float | int]:
    """Bootstrap tasks as clusters with replacement and return lift CI.

    Uses the Cartesian product of arm-A and arm-B calibration combos
    (C(R,k)^2 pairings).  Per-task calibration-sum arrays are
    precomputed so that the inner loop is array lookups and scalar
    arithmetic (stdlib only, exact task-bootstrap semantics).
    """

    if not task_panels or bootstrap_trials <= 0:
        return {"trials": int(bootstrap_trials), "lower_2_5": 0.0, "upper_97_5": 0.0}

    rng = random.Random(seed)
    combo_pool = tuple(itertools.combinations(range(repeats), k))
    n_combos = len(combo_pool)
    task_count = len(task_panels)
    held_per_task = repeats - k
    held_trials = task_count * held_per_task

    # ----- precompute per-task per-combo calibration sums -----
    task_totals_a: list[int] = []
    task_totals_b: list[int] = []
    task_cal_a: list[list[int]] = []   # [task_idx][combo_idx]
    task_cal_b: list[list[int]] = []   # [task_idx][combo_idx]
    for _, rewards_a, rewards_b in task_panels:
        task_totals_a.append(sum(rewards_a))
        task_totals_b.append(sum(rewards_b))
        task_cal_a.append([sum(rewards_a[p] for p in combo) for combo in combo_pool])
        task_cal_b.append([sum(rewards_b[p] for p in combo) for combo in combo_pool])

    lifts: list[float] = []
    sample_cal_a = [0] * n_combos
    sample_cal_b = [0] * n_combos
    sample_hold_a = [0] * n_combos
    sample_hold_b = [0] * n_combos

    for _ in range(bootstrap_trials):
        sampled = [rng.randrange(task_count) for _ in range(task_count)]

        # --- per-combo aggregates for this bootstrap sample (fixed-arm decision) ---
        for ca in range(n_combos):
            sample_cal_a[ca] = 0
            sample_hold_a[ca] = 0
        for cb in range(n_combos):
            sample_cal_b[cb] = 0
            sample_hold_b[cb] = 0

        for idx in sampled:
            tot_a = task_totals_a[idx]
            tot_b = task_totals_b[idx]
            cal_a_row = task_cal_a[idx]
            cal_b_row = task_cal_b[idx]
            for ca in range(n_combos):
                cal = cal_a_row[ca]
                sample_cal_a[ca] += cal
                sample_hold_a[ca] += tot_a - cal
            for cb in range(n_combos):
                cal = cal_b_row[cb]
                sample_cal_b[cb] += cal
                sample_hold_b[cb] += tot_b - cal

        # --- Cartesian-product splits for this bootstrap sample ---
        split_lifts: list[float] = []
        for ca in range(n_combos):
            for cb in range(n_combos):
                selector_success = 0.0
                for idx in sampled:
                    a_cal = task_cal_a[idx][ca]
                    b_cal = task_cal_b[idx][cb]
                    a_hold = task_totals_a[idx] - a_cal
                    b_hold = task_totals_b[idx] - b_cal
                    if a_cal > b_cal:
                        selector_success += a_hold
                    elif a_cal < b_cal:
                        selector_success += b_hold
                    else:
                        selector_success += 0.5 * (a_hold + b_hold)

                selector_rate = selector_success / held_trials

                # calibrated fixed arm for this (ca, cb) split pair
                if sample_cal_a[ca] > sample_cal_b[cb]:
                    fixed_success = sample_hold_a[ca]
                elif sample_cal_a[ca] < sample_cal_b[cb]:
                    fixed_success = sample_hold_b[cb]
                else:
                    fixed_success = 0.5 * (sample_hold_a[ca] + sample_hold_b[cb])
                fixed_rate = fixed_success / held_trials

                split_lifts.append(selector_rate - fixed_rate)

        lifts.append(_mean(split_lifts))

    return {
        "trials": bootstrap_trials,
        "lower_2_5": percentile(lifts, 0.025),
        "upper_97_5": percentile(lifts, 0.975),
    }


def analyze_repeated_outcome(
    rows: Sequence[dict[str, Any]],
    *,
    arm_a: str,
    arm_b: str,
    model: str,
    repeats: int = 5,
    seed: int = 20260811,
    bootstrap_trials: int = 10000,
) -> dict[str, Any]:
    """Core pure analysis function over already-read rows."""

    if repeats < 2:
        raise ValueError("--repeats must be at least 2")

    filtered = filter_rows(rows, model=model, arm_a=arm_a, arm_b=arm_b)
    deduped, duplicate_count = deduplicate_trials(filtered, arm_a=arm_a, arm_b=arm_b)
    task_panels, exclusions = build_complete_task_panels(
        deduped,
        repeats=repeats,
        seed=seed,
        arm_a=arm_a,
        arm_b=arm_b,
    )

    counts: dict[str, Any] = {
        "input_rows": len(rows),
        "raw_rows": len(filtered),
        "logical_rows": len(deduped),
        "filter": {
            "model": model,
            "arms": [arm_a, arm_b],
            "rows_after_filter": len(filtered),
        },
        "completeness": {
            "tasks_seen": len({row["task_name"] for row in filtered}),
            "tasks_retained": len(task_panels),
            "tasks_excluded": len({row["task_name"] for row in filtered}) - len(task_panels),
            "repeats_required": repeats,
        },
        "exclusion": {
            "missing_arm": exclusions["missing_arm"],
            "wrong_repeat_count": exclusions["wrong_repeat_count"],
            "duplicate_rows_dropped": duplicate_count,
        },
        "retained_outcomes": {
            arm_a: {
                "attempts": len(task_panels) * repeats,
                "successes": sum(sum(rewards_a) for _, rewards_a, _ in task_panels),
            },
            arm_b: {
                "attempts": len(task_panels) * repeats,
                "successes": sum(sum(rewards_b) for _, _, rewards_b in task_panels),
            },
        },
    }

    learning: dict[str, Any] = {}
    for k in range(1, repeats):
        average, split_results = analyze_k(task_panels, repeats=repeats, k=k)
        average["bootstrap_ci"] = bootstrap_task_lift_ci(
            task_panels,
            repeats=repeats,
            k=k,
            bootstrap_trials=bootstrap_trials,
            seed=seed + k,
        )
        average["split_results"] = split_results
        learning[str(k)] = average

    primary_k = min(2, repeats - 1)
    primary = learning[str(primary_k)]
    primary_lift = primary["lift"]["selector_minus_fixed"]
    primary_lower = primary["bootstrap_ci"]["lower_2_5"]
    later_lifts = [
        learning[str(k)]["lift"]["selector_minus_fixed"]
        for k in range(primary_k, repeats)
    ]
    non_disappearing = all(lift >= 0.0 for lift in later_lifts)
    if primary_lift > 0.0 and primary_lower > 0.0 and non_disappearing:
        verdict = "supports_repeats"
    elif primary_lift <= 0.0:
        verdict = "does_not_support_repeats"
    else:
        verdict = "inconclusive"

    warnings = [
        "This is observational, repeated-outcome evidence and does not control for "
        "historical confounding.",
        "Trials are ordered per arm-task by SHA-256 of seed|task_name|arm|trial_name.",
        "Learning curves reuse trials and are descriptive by construction.",
    ]
    if counts["exclusion"]["missing_arm"] or counts["exclusion"]["wrong_repeat_count"]:
        warnings.append(
            f"{counts['exclusion']['missing_arm']} tasks missing one arm and "
            f"{counts['exclusion']['wrong_repeat_count']} tasks with wrong repeat counts "
            f"were excluded from complete-task analysis."
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "args": {
            "arm_a": arm_a,
            "arm_b": arm_b,
            "model": model,
            "repeats": repeats,
            "seed": seed,
            "bootstrap_trials": bootstrap_trials,
        },
        "counts": counts,
        "decision": {
            "verdict": verdict,
            "primary_k": primary_k,
            "primary_lift": primary_lift,
            "primary_ci_95": [
                primary["bootstrap_ci"]["lower_2_5"],
                primary["bootstrap_ci"]["upper_97_5"],
            ],
            "non_disappearing_from_primary_k": non_disappearing,
            "practical_target": 0.05,
            "primary_upper_below_practical_target": (
                primary["bootstrap_ci"]["upper_97_5"] < 0.05
            ),
            "rule": (
                "Support requires positive primary lift, bootstrap lower bound above "
                "zero, and nonnegative lift for all larger k. A nonpositive point "
                "estimate does not establish equivalence or futility."
            ),
        },
        "warnings": warnings,
        "learning_curve": learning,
    }


def run_smoke(
    input_jsonl: str | Path,
    output_json: str | Path,
    *,
    arm_a: str,
    arm_b: str,
    model: str,
    repeats: int = 5,
    seed: int = 20260811,
    bootstrap_trials: int = 10000,
) -> dict[str, Any]:
    input_path = Path(input_jsonl).expanduser().resolve()
    output_path = Path(output_json).expanduser().resolve()

    if input_path.is_dir():
        raise ValueError("INPUT_JSONL must be a file")
    if output_path.exists() and output_path.is_dir():
        raise ValueError("OUTPUT_JSON must be a file path, not a directory")

    raw_rows = _read_jsonl(input_path)
    report = analyze_repeated_outcome(
        raw_rows,
        arm_a=arm_a,
        arm_b=arm_b,
        model=model,
        repeats=repeats,
        seed=seed,
        bootstrap_trials=bootstrap_trials,
    )

    report["input_jsonl"] = str(input_path)
    report["output_json"] = str(output_path)
    report["input_sha256"] = _input_sha256(input_path)

    if report["counts"]["completeness"]["tasks_retained"] == 0:
        raise ValueError("No complete tasks remain after filtering and deduplication")

    write_json(output_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyreplab-repeated-outcome-smoke",
        description="Analyze repeated-outcome two-arm outcome stability protocol.",
    )
    parser.add_argument("input_jsonl")
    parser.add_argument("output_json")
    parser.add_argument("--arm-a", required=True)
    parser.add_argument("--arm-b", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--bootstrap-trials", type=int, default=10000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_smoke(
            args.input_jsonl,
            args.output_json,
            arm_a=args.arm_a,
            arm_b=args.arm_b,
            model=args.model,
            repeats=args.repeats,
            seed=args.seed,
            bootstrap_trials=args.bootstrap_trials,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, sort_keys=True))
        return 1

    print(json.dumps({"ok": True, **result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
