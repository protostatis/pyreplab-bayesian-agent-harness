#!/usr/bin/env python3
"""Run the end-to-end ToolFailBench 5-fold outcome-model workflow.

The harness performs:

1) optional dataset build from a run root
2) deterministic task-level fold split by sorted task_id
3) per-fold model training and held-out validation evaluation
4) cross-fold summary with aggregate validation metrics

All steps are intentionally explicit and deterministic so runs are reproducible.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pyreplab_harness import outcome_model as om
from pyreplab_harness.dataset import write_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministic full-run ToolFailBench CV pipeline: build dataset, "
            "create fold datasets, train/evaluate outcome models, and print "
            "cross-fold validation summary."
        )
    )
    parser.add_argument(
        "--run-root",
        help=(
            "optional run root containing tasks/ and attempts/; if provided "
            "the dataset is rebuilt from this root"
        ),
    )
    parser.add_argument(
        "--dataset",
        default=".runs/adhoc-toolfailbench-dataset.jsonl",
        help="dataset JSONL path (written when --run-root is provided)",
    )
    parser.add_argument(
        "--cv-dir",
        default=".runs/adhoc-toolfailbench-cv",
        help="directory to write fold datasets and artifacts",
    )
    parser.add_argument("--folds", type=int, default=5, help="number of CV folds")
    parser.add_argument(
        "--expected-k",
        type=int,
        default=5,
        help="expected validation modulus check; must equal --folds for strict CI-style check",
    )
    parser.add_argument("--train-seed-base", type=int, default=2026)
    parser.add_argument("--eval-seed-base", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--train-num-samples", type=int, default=20)
    parser.add_argument("--eval-num-samples", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--summary", help="optional JSON output path for the final summary")
    parser.add_argument(
        "--train",
        dest="train",
        action="store_true",
        default=True,
        help="(default) run per-fold training and evaluation",
    )
    parser.add_argument(
        "--no-train",
        dest="train",
        action="store_false",
        help="skip per-fold training and evaluation",
    )
    parser.add_argument(
        "--skip-splits-check",
        action="store_true",
        help="skip strict fold split integrity check",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow overwriting existing fold datasets and artifact directories",
    )
    return parser.parse_args()


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        raise FileNotFoundError(f"dataset not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(f"malformed JSON at {path}:{line_no}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"non-object row at {path}:{line_no}")
            rows.append(row)
    return rows


def write_jsonl_rows(path: Path, rows: list[dict[str, Any]], overwrite: bool = False) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"would overwrite existing dataset: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_fold_assignments(task_ids: list[str], folds: int) -> dict[str, int]:
    if folds < 1:
        raise ValueError(f"folds must be >= 1, got {folds}")
    return {task_id: index % folds for index, task_id in enumerate(task_ids)}


def build_folds(
    dataset_rows: list[dict[str, Any]],
    folds: int,
    cv_dir: Path,
    *,
    overwrite: bool,
) -> list[dict[str, Any]]:
    if not dataset_rows:
        raise ValueError("dataset is empty")

    task_ids = sorted({str(row["task_id"]) for row in dataset_rows if "task_id" in row})
    if not task_ids:
        raise ValueError("dataset has no task_id values")

    if folds > len(task_ids):
        raise ValueError(f"folds ({folds}) > task count ({len(task_ids)})")

    assignment = build_fold_assignments(task_ids, folds)
    fold_rows: list[list[dict[str, Any]]] = [[] for _ in range(folds)]
    for row in dataset_rows:
        task_id = str(row["task_id"])
        task_fold = assignment[task_id]
        for fold_id in range(folds):
            row_copy = dict(row)
            row_copy["split"] = "validation" if fold_id == task_fold else "train"
            fold_rows[fold_id].append(row_copy)

    summaries: list[dict[str, Any]] = []
    for fold_index, rows in enumerate(fold_rows, start=1):
        # Deterministic rewrite for each fold by task assignment.
        path = cv_dir / f"fold-{fold_index}.jsonl"
        write_jsonl_rows(path, rows, overwrite=overwrite)
        summaries.append(
            {
                "fold": fold_index,
                "path": str(path),
                "rows": len(rows),
                "tasks": len({str(row["task_id"]) for row in rows}),
            }
        )
    return summaries


def check_fold_integrity(fold_rows: list[tuple[int, list[dict[str, Any]], int]], *, expected_k: int) -> None:
    folds = len(fold_rows)
    if folds == 0:
        raise ValueError("no folds to check")

    if expected_k and expected_k != folds:
        raise ValueError(
            f"expected_k ({expected_k}) must match folds ({folds}) for deterministic check"
        )

    # Partition the dataset tasks by their validation fold.
    validation_per_fold: list[set[str]] = []
    expected_all_tasks: set[str] | None = None
    for fold_id, rows, _ in fold_rows:
        validation_tasks: set[str] = set()
        train_tasks: set[str] = set()
        for row in rows:
            split = str(row.get("split", ""))
            task = str(row["task_id"])
            if split == "validation":
                validation_tasks.add(task)
            elif split == "train":
                train_tasks.add(task)
            else:
                raise ValueError(f"unexpected split {split!r} in fold {fold_id}")

            if task in validation_tasks and task in train_tasks:
                raise ValueError(
                    f"task {task} appears as both validation and train in fold {fold_id}"
                )

        if expected_all_tasks is None:
            expected_all_tasks = set(train_tasks) | set(validation_tasks)
        elif train_tasks | validation_tasks != expected_all_tasks:
            raise ValueError(f"fold {fold_id} has a different task set than other folds")

        validation_per_fold.append(validation_tasks)

    assert expected_all_tasks is not None
    sorted_tasks = sorted(expected_all_tasks)
    expected = {task: index % folds for index, task in enumerate(sorted_tasks)}

    # Each task is validation in exactly one fold.
    seen_validation: dict[str, int] = {}
    for fold_id, validation_tasks in enumerate(validation_per_fold, start=1):
        for task in validation_tasks:
            prior = seen_validation.get(task)
            if prior is None:
                seen_validation[task] = fold_id
                continue
            raise ValueError(
                f"task {task} is validation in folds {prior} and {fold_id}"
            )

    for task in sorted_tasks:
        observed = seen_validation.get(task)
        expected_fold = expected[task] + 1
        if observed is None:
            raise ValueError(f"task {task} missing validation assignment")
        if observed != expected_fold:
            raise ValueError(
                f"task {task} assigned to validation fold {observed} "
                f"but expected {expected_fold}"
            )


def summarize_series(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None}
    if len(values) == 1:
        value = float(values[0])
        return {"mean": value, "std": 0.0}
    mean = statistics.fmean(values)
    std = statistics.pstdev(values)
    return {"mean": mean, "std": std}


def summarize_cv_results(fold_evals: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "accuracy_05",
        "average_precision",
        "brier",
        "ece",
        "precision",
        "recall",
        "f1",
    ]
    per_metric: dict[str, list[float]] = {key: [] for key in keys}
    for fold in fold_evals:
        metrics = fold["validation_metrics"]
        for key in keys:
            value = metrics.get(key)
            if value is None:
                continue
            per_metric[key].append(float(value))

    summary: dict[str, Any] = {}
    for key, values in per_metric.items():
        summary[key] = summarize_series(values)
        summary[key]["n"] = len(values)
    return summary


def train_and_eval_fold(
    fold: int,
    dataset_path: Path,
    artifact_dir: Path,
    *,
    overwrite: bool,
    train_seed: int,
    eval_seed: int,
    batch_size: int,
    epochs: int,
    patience: int,
    train_num_samples: int,
    eval_num_samples: int,
    device: str,
) -> dict[str, Any]:
    if artifact_dir.exists() and any(artifact_dir.iterdir()) and not overwrite:
        # Keep explicit and predictable; this mirrors fold dataset handling above.
        raise FileExistsError(f"would overwrite existing artifact directory: {artifact_dir}")

    training = om.train_model(
        str(dataset_path),
        str(artifact_dir),
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        num_samples=train_num_samples,
        seed=train_seed,
        device=device,
        verbose=False,
    )

    evaluation = om.evaluate_model(
        str(dataset_path),
        str(artifact_dir),
        device=device,
        num_samples=eval_num_samples,
        seed=eval_seed,
    )
    validation_metrics = evaluation["metrics"]["validation"]
    return {
        "fold": fold,
        "artifact_dir": str(artifact_dir),
        "dataset_path": str(dataset_path),
        "training": training["training"],
        "validation_metrics": validation_metrics,
    }


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset).expanduser().resolve()
    cv_dir = Path(args.cv_dir).expanduser().resolve()
    folds = args.folds

    if args.run_root is not None:
        run_root = Path(args.run_root).expanduser().resolve()
        if not run_root.is_dir():
            print(f"run root does not exist: {run_root}", file=sys.stderr)
            return 1
        write_dataset(run_root, dataset_path)

    if not dataset_path.exists():
        print(f"dataset not found: {dataset_path}", file=sys.stderr)
        return 1

    dataset_rows = read_jsonl_rows(dataset_path)
    if not dataset_rows:
        print(f"dataset is empty: {dataset_path}", file=sys.stderr)
        return 1

    cv_dir.mkdir(parents=True, exist_ok=True)
    fold_summaries = build_folds(dataset_rows, folds, cv_dir, overwrite=args.overwrite)

    fold_records: list[tuple[int, list[dict[str, Any]], int]] = []
    for fold_idx in range(1, folds + 1):
        fold_path = cv_dir / f"fold-{fold_idx}.jsonl"
        rows = read_jsonl_rows(fold_path)
        nrows = len(rows)
        fold_records.append((fold_idx, rows, nrows))

    if not args.skip_splits_check:
        check_fold_integrity(fold_records, expected_k=args.expected_k)

    fold_results: list[dict[str, Any]] = []
    if args.train:
        for fold_idx in range(1, folds + 1):
            print(f"running fold {fold_idx}/{folds}...", file=sys.stderr)
            artifact_dir = cv_dir / f"fold-{fold_idx}"
            if not args.overwrite and artifact_dir.exists() and any(artifact_dir.iterdir()):
                raise FileExistsError(f"would overwrite existing artifact directory: {artifact_dir}")
            artifact_dir.mkdir(parents=True, exist_ok=True)
            fold_dataset_path = cv_dir / f"fold-{fold_idx}.jsonl"
            result = train_and_eval_fold(
                fold_idx,
                fold_dataset_path,
                artifact_dir,
                overwrite=args.overwrite,
                train_seed=args.train_seed_base + fold_idx,
                eval_seed=args.eval_seed_base + fold_idx,
                batch_size=args.batch_size,
                epochs=args.epochs,
                patience=args.patience,
                train_num_samples=args.train_num_samples,
                eval_num_samples=args.eval_num_samples,
                device=args.device,
            )
            fold_results.append(result)

    cv_summary = {
        "dataset": {
            "path": str(dataset_path),
            "rows": len(dataset_rows),
            "tasks": len({str(row["task_id"]) for row in dataset_rows}),
        },
        "folds": folds,
        "expected_k": args.expected_k,
        "fold_datasets": fold_summaries,
        "folds_ran": len(fold_results),
        "fold_results": fold_results,
    }

    if fold_results:
        cv_summary["validation"] = summarize_cv_results(fold_results)

    if args.summary:
        summary_path = Path(args.summary).expanduser().resolve()
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(cv_summary, handle, indent=2, sort_keys=True)

    print(json.dumps(cv_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
