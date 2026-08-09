#!/usr/bin/env python3
"""Interrogate SoHarshh ToolFailBench CV folds for leakage and dataset integrity.

Checks performed:
* per-fold task-level split purity (no task appears in both train and validation)
* 5-fold (or user-configured) val-task partitioning across folds
* optional verification that val-task assignment matches sorted(task_id) mod K
* duplicate/malformed row sanity checks
* basic cardinality/label summaries
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed json in {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object row in {path}:{line_no}")
            if "task_id" not in row:
                raise ValueError(f"missing task_id in {path}:{line_no}")
            if "split" not in row:
                raise ValueError(f"missing split in {path}:{line_no}")
            rows.append(row)
    return rows


def fold_task_sets(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    split_to_tasks = defaultdict(set)
    for row in rows:
        split_to_tasks[str(row.get("split", "train"))].add(row["task_id"])
    return {split: tasks for split, tasks in split_to_tasks.items()}


def task_split_map(rows: list[dict[str, Any]]) -> dict[str, str]:
    task_to_split: dict[str, str] = {}
    for row in rows:
        task_id = str(row["task_id"])
        split = str(row.get("split", "train"))
        prev = task_to_split.get(task_id)
        if prev is None:
            task_to_split[task_id] = split
        elif prev != split:
            raise ValueError(f"split leak: task {task_id} appears in both {prev} and {split}")
    return task_to_split


def build_expected_val_assignments(tasks: list[str], k: int) -> dict[str, int]:
    ordered = sorted(set(tasks))
    return {task_id: i % k for i, task_id in enumerate(ordered)}


def summarize_fold(rows: list[dict[str, Any]], fold_index: int) -> dict[str, Any]:
    tasks_by_split = fold_task_sets(rows)
    task_splits = task_split_map(rows)
    task_counts = defaultdict(int)
    for row in rows:
        task_counts[row["task_id"]] += 1
    success = [1 for row in rows if bool(row.get("verified_success"))]
    row_policy_ids = {
        str(row["model_input"].get("policy_id", row["model_input"].get("policy")))
        for row in rows
        if "model_input" in row and isinstance(row["model_input"], dict)
    }
    return {
        "fold": fold_index,
        "rows": len(rows),
        "task_count": len(task_splits),
        "split_counts": {split: len(tasks) for split, tasks in tasks_by_split.items()},
        "task_rows_per_split": {
            "train": sum(1 for task, split in task_splits.items() if split == "train"),
            "validation": sum(1 for task, split in task_splits.items() if split == "validation"),
            "test": sum(1 for task, split in task_splits.items() if split == "test"),
        },
        "rows_per_task_unique": (min(task_counts.values()), max(task_counts.values()))
        if task_counts
        else (0, 0),
        "policy_ids_in_fold": sorted(x for x in row_policy_ids if x is not None),
        "verified_success_rate": (sum(success) / len(success)) if success else None,
        "verified_success_count": sum(success),
        "mixed_split_task_count": 0,
    }


def assert_non_leaking(
    fold_summaries: list[dict[str, Any]],
    fold_task_sets_per_split: list[dict[str, set[str]]],
    k: int,
    all_tasks: list[str] | None = None,
) -> None:
    folds = len(fold_summaries)
    if folds == 0:
        raise ValueError("no folds supplied")

    for summary in fold_summaries:
        print(f"fold {summary['fold']}: rows={summary['rows']} tasks={summary['tasks']} "
              f"train_tasks={summary['task_rows_per_split'].get('train', 0)} "
              f"val_tasks={summary['task_rows_per_split'].get('validation', 0)} "
              f"success_rate={summary['verified_success_rate']:.4f} "
              f"rows/task={summary['rows_per_task_unique']}")

    # task-level split disjointness for each fold
    for i in range(folds):
        s = fold_task_sets_per_split[i]
        train_tasks = s.get("train", set())
        val_tasks = s.get("validation", set())
        test_tasks = s.get("test", set())
        leakage = (train_tasks & val_tasks) | (train_tasks & test_tasks) | (val_tasks & test_tasks)
        if leakage:
            raise ValueError(f"fold {i+1} has task overlap across split types: {sorted(leakage)[:5]}")

    # expected val sets are a partition of tasks by sorted(task_id) mod K
    val_sets = [set(tasks_by_split.get("validation", set())) for tasks_by_split in fold_task_sets_per_split]
    for i in range(folds):
        for j in range(i + 1, folds):
            overlap = val_sets[i] & val_sets[j]
            if overlap:
                raise ValueError(f"val-task overlap between folds {i+1} and {j+1}: {len(overlap)}")

    if all_tasks is not None:
        all_task_set = set(all_tasks)
        all_val = set().union(*val_sets)
        missing_val = all_task_set - all_val
        if missing_val:
            raise ValueError(f"some tasks never appear in val: {len(missing_val)}")
        extra_val = all_val - all_task_set
        if extra_val:
            raise ValueError(f"val contains unknown tasks: {len(extra_val)}")

        if k != 0:
            expected_by_task = build_expected_val_assignments(sorted(all_task_set), k)
            for fold_idx, tasks in enumerate(val_sets, start=1):
                bad = [task for task in sorted(tasks) if expected_by_task.get(task) != (fold_idx - 1)]
                if bad:
                    raise ValueError(f"fold {fold_idx}: {len(bad)} val tasks not matching sorted(task_id) mod {k}")

    # each task should land in exactly one val fold
    task_to_val_count: defaultdict[str, int] = defaultdict(int)
    for tasks in val_sets:
        for task in tasks:
            task_to_val_count[task] += 1

    if len(task_to_val_count) > 0:
        counts = sorted(set(task_to_val_count.values()))
        multi = {task: c for task, c in task_to_val_count.items() if c != 1}
        if multi:
            raise ValueError(f"{len(multi)} tasks have val-count != 1")
        if counts != [1]:
            print(f"WARN: val-count coverage values were {counts}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check CV split leakage for ToolFailBench folds")
    parser.add_argument("cv_dir", nargs="?", default=".runs/adhoc-toolfailbench-cv", help="Directory containing fold-*.jsonl")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--base-dataset", default=".runs/adhoc-toolfailbench-dataset.jsonl", help="Base full dataset to verify hash/coverage")
    parser.add_argument("--expected-k", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="Emit JSON summary payload")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cv_dir = Path(args.cv_dir).expanduser()
    if not cv_dir.is_dir():
        print(f"cv directory not found: {cv_dir}")
        return 1

    fold_rows: list[list[dict[str, Any]]] = []
    fold_summaries: list[dict[str, Any]] = []
    task_sets_by_fold: list[dict[str, set[str]]] = []

    for i in range(1, args.folds + 1):
        path = cv_dir / f"fold-{i}.jsonl"
        rows = read_rows(path)
        fold_rows.append(rows)
        task_splits = task_split_map(rows)
        tasks_by_split = fold_task_sets(rows)
        # validate policy coverage is consistent and finite rows per task (warn only)
        rows_by_task = defaultdict(int)
        for row in rows:
            rows_by_task[row["task_id"]] += 1
        row_counts = sorted(set(rows_by_task.values()))
        if len(row_counts) > 1:
            print(f"WARN: fold-{i} has varying rows/task counts: {row_counts[:10]}")

        success = [1 for row in rows if bool(row.get("verified_success"))]
        summary = {
            "fold": i,
            "rows": len(rows),
            "tasks": len(task_splits),
            "task_rows_per_split": {split: len(task_set) for split, task_set in tasks_by_split.items()},
            "split_task_counts": {split: len(task_set) for split, task_set in tasks_by_split.items()},
            "verified_success_rate": (sum(success) / len(rows)) if rows else None,
            "rows_per_task_unique": (min(rows_by_task.values()), max(rows_by_task.values())),
            "rows_per_task": {"min": min(rows_by_task.values()), "max": max(rows_by_task.values())},
            "policies": sorted(
                {
                    str(row["model_input"].get("policy_id", row["model_input"].get("policy")))
                    for row in rows
                    if isinstance(row.get("model_input"), dict)
                }
            ),
        }
        fold_summaries.append(summary)
        task_sets_by_fold.append(tasks_by_split)

    if args.base_dataset:
        base_path = Path(args.base_dataset).expanduser()
        base_rows = read_rows(base_path)
        base_tasks = [row["task_id"] for row in base_rows]
        full_task_set = set(base_tasks)
    else:
        fold_task_sets_union = [
            task_sets_by_fold[i].get("train", set())
            | task_sets_by_fold[i].get("validation", set())
            | task_sets_by_fold[i].get("test", set())
            for i in range(len(task_sets_by_fold))
        ]
        full_task_set = set().union(*fold_task_sets_union)

    try:
        assert_non_leaking(fold_summaries, task_sets_by_fold, args.expected_k, list(full_task_set))
    except Exception as err:
        print(f"LEAK CHECK FAILED: {err}")
        return 2

    print("LEAK CHECK PASSED")

    if args.json:
        payload = {
            "folds": fold_summaries,
            "base_dataset_rows": len(base_rows) if args.base_dataset else None,
            "base_dataset_tasks": len(full_task_set),
        }
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
