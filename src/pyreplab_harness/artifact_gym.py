from __future__ import annotations

import csv
import json
import random
import re
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

from datetime import datetime, timezone

try:
    from datetime import UTC
except ImportError:  # Python < 3.11 compatibility
    UTC = timezone.utc

from .contracts import AttemptRecord, TaskSpec, VerificationResult
from .io_utils import read_json, write_json

GENERATOR_VERSION = "artifact-reconciliation-v1"
TEMPLATE_ID = "customer-paid-orders-v1"
VERIFIER_ID = "artifact-semantic-json"
VERIFIER_VERSION = "1"
DIFFICULTIES = {"easy", "medium", "hard"}
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@-]{0,127}$")


def _safe_id(value: str, label: str) -> str:
    if not SAFE_ID.fullmatch(value):
        raise ValueError(f"invalid {label}: {value!r}")
    return value


def _root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _task_dir(root: Path, task_id: str) -> Path:
    return root / "tasks" / _safe_id(task_id, "task id")


def _attempt_dir(root: Path, attempt_id: str) -> Path:
    return root / "attempts" / _safe_id(attempt_id, "attempt id")


def _difficulty_shape(difficulty: str) -> tuple[int, int, int]:
    if difficulty == "easy":
        return 7, 22, 18_000
    if difficulty == "medium":
        return 14, 65, 42_000
    if difficulty == "hard":
        return 25, 145, 75_000
    raise ValueError(f"unsupported difficulty: {difficulty!r}")


def _build_dataset(seed: int, difficulty: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    customer_count, order_count, threshold = _difficulty_shape(difficulty)
    rng = random.Random(seed)
    regions = ["north", "south", "east", "west"]
    target_region = rng.choice(regions)

    customers: list[dict[str, Any]] = []
    for index in range(customer_count):
        customers.append(
            {
                "customer_id": f"C{index + 1:03d}",
                "status": "active" if rng.random() < 0.72 else "inactive",
                "region": rng.choice(regions),
            }
        )

    # Guarantee at least one qualifying answer while retaining distractors.
    customers[0]["status"] = "active"
    customers[0]["region"] = target_region

    orders: list[dict[str, Any]] = []
    statuses = ["paid", "paid", "paid", "pending", "cancelled", "refunded"]
    for index in range(order_count):
        orders.append(
            {
                "order_id": f"O{index + 1:04d}",
                "customer_id": rng.choice(customers)["customer_id"],
                "status": rng.choice(statuses),
                "amount_cents": rng.randint(500, max(2_000, threshold // 2)),
            }
        )

    orders.extend(
        [
            {
                "order_id": "OG001",
                "customer_id": customers[0]["customer_id"],
                "status": "paid",
                "amount_cents": threshold,
            },
            {
                "order_id": "OG002",
                "customer_id": customers[0]["customer_id"],
                "status": "paid",
                "amount_cents": threshold // 2,
            },
            {
                "order_id": "OG003",
                "customer_id": customers[0]["customer_id"],
                "status": "refunded",
                "amount_cents": threshold * 3,
            },
        ]
    )

    active_target_ids = {
        customer["customer_id"]
        for customer in customers
        if customer["status"] == "active" and customer["region"] == target_region
    }
    totals: dict[str, tuple[int, int]] = {customer_id: (0, 0) for customer_id in active_target_ids}
    for order in orders:
        customer_id = order["customer_id"]
        if customer_id in totals and order["status"] == "paid":
            total, count = totals[customer_id]
            totals[customer_id] = (total + int(order["amount_cents"]), count + 1)

    expected = [
        {
            "customer_id": customer_id,
            "paid_total_cents": total,
            "paid_order_count": count,
        }
        for customer_id, (total, count) in totals.items()
        if total >= threshold
    ]
    expected.sort(key=lambda item: (-item["paid_total_cents"], item["customer_id"]))
    oracle = {
        "target_region": target_region,
        "threshold_cents": threshold,
        "expected": expected,
    }
    return customers, orders, oracle


def generate_artifact_task(root: str | Path, seed: int, difficulty: str = "medium") -> TaskSpec:
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {sorted(DIFFICULTIES)}")
    root_path = _root(root)
    task_id = f"artifact-{TEMPLATE_ID}-{difficulty}-{seed}"
    task_path = _task_dir(root_path, task_id)
    manifest_path = task_path / "task.json"
    if manifest_path.exists():
        return TaskSpec.from_dict(read_json(manifest_path))

    initial = task_path / "initial"
    private = task_path / "private"
    initial.mkdir(parents=True, exist_ok=False)
    private.mkdir(parents=True, exist_ok=False)

    customers, orders, oracle = _build_dataset(seed, difficulty)
    write_json(initial / "customers.json", customers)
    with (initial / "orders.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["order_id", "customer_id", "status", "amount_cents"],
        )
        writer.writeheader()
        writer.writerows(orders)

    contract = (
        "Read customers.json and orders.csv from the task workspace.",
        f"Include only active customers in region {oracle['target_region']!r}.",
        "Count and sum only orders whose status is exactly 'paid'.",
        f"Include customers whose paid total is at least {oracle['threshold_cents']} cents.",
        "Write result.json as a JSON array with exactly customer_id, paid_total_cents, and paid_order_count.",
        "Sort by paid_total_cents descending, then customer_id ascending.",
    )
    prompt = (
        "Complete the structured-data task in the isolated /workspace directory.\n\n"
        + "\n".join(f"- {item}" for item in contract)
        + "\n\nThe task is complete only when /workspace/result.json exists and satisfies every rule."
    )
    (initial / "TASK.md").write_text(prompt + "\n", encoding="utf-8")
    write_json(private / "oracle.json", oracle)

    spec = TaskSpec(
        id=task_id,
        family="artifact",
        template_id=TEMPLATE_ID,
        generator_version=GENERATOR_VERSION,
        seed=seed,
        difficulty=difficulty,
        prompt=prompt,
        contract=contract,
        public_metadata={
            "input_files": ["customers.json", "orders.csv", "TASK.md"],
            "customer_count": len(customers),
            "order_count": len(orders),
            "required_output": "result.json",
        },
        workspace_ref=str(initial),
        verifier_ref=str(private / "oracle.json"),
    )
    write_json(manifest_path, spec.to_dict())
    return spec


def load_task(root: str | Path, task_id: str) -> TaskSpec:
    return TaskSpec.from_dict(read_json(_task_dir(_root(root), task_id) / "task.json"))


def prepare_attempt(
    root: str | Path,
    task_id: str,
    attempt_id: str,
    policy_id: str,
    policy_version: str = "1",
    treatment_bundle_hash: str | None = None,
    treatment_registry_hash: str | None = None,
    rollout_replica: int | None = None,
    sampling_seed: int | None = None,
    pilot_manifest_hash: str | None = None,
    pilot_panel_id: str | None = None,
    expected_task_commitment_hash: str | None = None,
) -> AttemptRecord:
    root_path = _root(root)
    spec = load_task(root_path, task_id)
    attempt_path = _attempt_dir(root_path, attempt_id)
    if attempt_path.exists():
        raise FileExistsError(f"attempt already exists: {attempt_id}")
    if rollout_replica is not None and (
        isinstance(rollout_replica, bool)
        or not isinstance(rollout_replica, int)
        or rollout_replica < 0
    ):
        raise ValueError("rollout_replica must be a non-negative integer")
    if sampling_seed is not None and (
        isinstance(sampling_seed, bool)
        or not isinstance(sampling_seed, int)
        or sampling_seed < 0
        or sampling_seed > 2_147_483_647
    ):
        raise ValueError("sampling_seed must be an integer in [0, 2147483647]")
    workspace = attempt_path / "workspace"
    attempt_path.mkdir(parents=True)
    oracle_snapshot_ref: str | None = None
    oracle_snapshot_sha256: str | None = None
    try:
        shutil.copytree(spec.workspace_ref, workspace)
        if expected_task_commitment_hash is not None:
            from .unbrowser_fixture_gym import (
                unbrowser_fixture_snapshot_commitment,
                unbrowser_fixture_task_commitment,
            )

            source_commitment = unbrowser_fixture_task_commitment(
                root_path, task_id
            )
            if source_commitment["commitment_hash"] != expected_task_commitment_hash:
                raise ValueError("fixture task commitment mismatch before snapshot")
            oracle_snapshot = attempt_path / "oracle.snapshot.json"
            shutil.copy2(spec.verifier_ref, oracle_snapshot)
            snapshot_commitment = unbrowser_fixture_snapshot_commitment(
                spec, workspace, oracle_snapshot
            )
            if snapshot_commitment != source_commitment:
                raise ValueError("fixture task changed while snapshotting attempt")
            oracle_snapshot_ref = str(oracle_snapshot)
            oracle_snapshot_sha256 = str(snapshot_commitment["oracle_sha256"])
    except Exception:
        shutil.rmtree(attempt_path, ignore_errors=True)
        raise
    record = AttemptRecord(
        attempt_id=attempt_id,
        task_id=task_id,
        policy_id=_safe_id(policy_id, "policy id"),
        policy_version=_safe_id(policy_version, "policy version"),
        workspace_ref=str(workspace),
        created_at=datetime.now(UTC).isoformat(),
        treatment_bundle_hash=treatment_bundle_hash,
        treatment_registry_hash=treatment_registry_hash,
        rollout_replica=rollout_replica,
        sampling_seed=sampling_seed,
        pilot_manifest_hash=pilot_manifest_hash,
        pilot_panel_id=pilot_panel_id,
        task_commitment_hash=expected_task_commitment_hash,
        oracle_snapshot_ref=oracle_snapshot_ref,
        oracle_snapshot_sha256=oracle_snapshot_sha256,
    )
    write_json(attempt_path / "attempt.json", record.to_dict())
    return record


def load_attempt(root: str | Path, attempt_id: str) -> AttemptRecord:
    return AttemptRecord.from_dict(read_json(_attempt_dir(_root(root), attempt_id) / "attempt.json"))


def record_pi_events(root: str | Path, attempt_id: str, raw_events: str, normalized: dict[str, Any]) -> AttemptRecord:
    root_path = _root(root)
    attempt_path = _attempt_dir(root_path, attempt_id)
    record = load_attempt(root_path, attempt_id)
    raw_path = attempt_path / "pi-events.jsonl"
    raw_path.write_text(raw_events, encoding="utf-8")
    normalized_path = attempt_path / "pi-events.normalized.json"
    write_json(normalized_path, normalized)
    updated = replace(
        record,
        status="executed",
        pi_events_ref=str(raw_path),
        normalized_events_ref=str(normalized_path),
    )
    write_json(attempt_path / "attempt.json", updated.to_dict())
    return updated


def verify_artifact_attempt(root: str | Path, task_id: str, attempt_id: str) -> VerificationResult:
    root_path = _root(root)
    spec = load_task(root_path, task_id)
    attempt = load_attempt(root_path, attempt_id)
    if attempt.task_id != spec.id:
        raise ValueError("attempt does not belong to task")

    oracle = read_json(Path(spec.verifier_ref))
    output_path = Path(attempt.workspace_ref) / "result.json"
    if not output_path.exists():
        result = VerificationResult(
            success=False,
            verifier_id=VERIFIER_ID,
            verifier_version=VERIFIER_VERSION,
            failure_code="missing_output",
            diagnostics={"required_output": "result.json"},
        )
    else:
        try:
            actual = read_json(output_path)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as error:
            result = VerificationResult(
                success=False,
                verifier_id=VERIFIER_ID,
                verifier_version=VERIFIER_VERSION,
                failure_code="invalid_json",
                diagnostics={"error": str(error)},
            )
        else:
            expected = oracle["expected"]
            success = actual == expected
            result = VerificationResult(
                success=success,
                verifier_id=VERIFIER_ID,
                verifier_version=VERIFIER_VERSION,
                failure_code=None if success else "semantic_mismatch",
                diagnostics={
                    "expected_rows": len(expected),
                    "actual_rows": len(actual) if isinstance(actual, list) else None,
                },
            )

    attempt_path = _attempt_dir(root_path, attempt_id)
    verification_path = attempt_path / "verification.json"
    write_json(verification_path, result.to_dict())
    updated = replace(
        attempt,
        status="verified",
        verification_ref=str(verification_path),
    )
    write_json(attempt_path / "attempt.json", updated.to_dict())
    return result
