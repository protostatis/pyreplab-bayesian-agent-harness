"""SQLite state-transformation task family.

The agent receives a workspace containing ``TASK.md`` and a seeded
``store.db`` (warehouse inventory).  It must modify ``store.db`` in place:
merge duplicate inventory rows, apply stock adjustments (clamping at zero),
and recompute reorder flags.  Verification never compares bytes or a
reference patch: it opens the submitted database with a private copy of the
initial dataset (the oracle), reruns the same reconciliation as private
queries/invariants, and checks schema, reference data, and final inventory
semantics.
"""

from __future__ import annotations

import itertools
import random
import re
import shutil
import sqlite3
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from .artifact_gym import load_attempt, load_task  # Generic loaders shared with the artifact gym.
from .contracts import TaskSpec, VerificationResult
from .io_utils import read_json, write_json

GENERATOR_VERSION = "sqlite-inventory-reconcile-v1"
TEMPLATE_ID = "inventory-reconcile-v1"
VERIFIER_ID = "sqlite-semantic-queries"
VERIFIER_VERSION = "1"
DIFFICULTIES = {"easy", "medium", "hard"}
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

DB_FILENAME = "store.db"
TABLES = ("warehouses", "skus", "inventory", "adjustments")

SCHEMA_SQL = """
CREATE TABLE warehouses (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    region TEXT NOT NULL
);
CREATE TABLE skus (
    id INTEGER PRIMARY KEY,
    sku TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    unit_cost_cents INTEGER NOT NULL
);
CREATE TABLE inventory (
    id INTEGER PRIMARY KEY,
    warehouse_id INTEGER NOT NULL REFERENCES warehouses(id),
    sku_id INTEGER NOT NULL REFERENCES skus(id),
    quantity INTEGER NOT NULL,
    reorder_level INTEGER NOT NULL,
    reorder_flag INTEGER NOT NULL
);
CREATE TABLE adjustments (
    id INTEGER PRIMARY KEY,
    warehouse_id INTEGER NOT NULL REFERENCES warehouses(id),
    sku_id INTEGER NOT NULL REFERENCES skus(id),
    delta INTEGER NOT NULL,
    reason TEXT NOT NULL
);
"""

TABLE_COLUMNS = {
    "warehouses": ["id", "name", "region"],
    "skus": ["id", "sku", "name", "category", "unit_cost_cents"],
    "inventory": ["id", "warehouse_id", "sku_id", "quantity", "reorder_level", "reorder_flag"],
    "adjustments": ["id", "warehouse_id", "sku_id", "delta", "reason"],
}


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


def _difficulty_shape(difficulty: str) -> dict[str, int]:
    if difficulty == "easy":
        return {"warehouses": 4, "skus": 8, "pairs": 7, "adjustments": 5}
    if difficulty == "medium":
        return {"warehouses": 6, "skus": 16, "pairs": 14, "adjustments": 10}
    if difficulty == "hard":
        return {"warehouses": 10, "skus": 30, "pairs": 22, "adjustments": 18}
    raise ValueError(f"unsupported difficulty: {difficulty!r}")


def _duplicate_size_options(difficulty: str) -> tuple[int, ...]:
    if difficulty == "easy":
        return (1, 1, 2, 2, 3)
    if difficulty == "medium":
        return (1, 1, 2, 2, 3)
    return (1, 2, 2, 3, 3, 4)


def _expected_inventory(
    initial: list[dict[str, Any]], adjustments: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Recompute the reconciled inventory from initial rows and adjustments.

    This is the shared semantic oracle used both at generation time and by
    the verifier, so the verifier genuinely reruns the reconciliation against
    the private initial dataset instead of trusting a stored answer.
    """
    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in initial:
        groups.setdefault((row["warehouse_id"], row["sku_id"]), []).append(row)

    deltas: dict[tuple[int, int], int] = {}
    for adjustment in adjustments:
        key = (adjustment["warehouse_id"], adjustment["sku_id"])
        deltas[key] = deltas.get(key, 0) + int(adjustment["delta"])

    expected: list[dict[str, Any]] = []
    for key in sorted(groups):
        rows = groups[key]
        kept = min(rows, key=lambda row: row["id"])
        quantity = sum(int(row["quantity"]) for row in rows) + deltas.get(key, 0)
        quantity = max(0, quantity)  # quantities never go negative.
        reorder_level = max(int(row["reorder_level"]) for row in rows)
        expected.append(
            {
                "id": int(kept["id"]),
                "warehouse_id": key[0],
                "sku_id": key[1],
                "quantity": quantity,
                "reorder_level": reorder_level,
                "reorder_flag": 1 if quantity <= reorder_level else 0,
            }
        )
    return expected


def _build_dataset(seed: int, difficulty: str) -> dict[str, Any]:
    shape = _difficulty_shape(difficulty)
    size_options = _duplicate_size_options(difficulty)
    rng = random.Random(seed)

    warehouses = [
        {
            "id": index + 1,
            "name": f"WH-{index + 1:02d}",
            "region": rng.choice(["north", "south", "east", "west"]),
        }
        for index in range(shape["warehouses"])
    ]
    categories = ["tools", "fasteners", "electrical", "plumbing", "lumber", "paint"]
    skus = [
        {
            "id": index + 1,
            "sku": f"SKU-{index + 1:04d}",
            "name": f"Item {index + 1:03d}",
            "category": rng.choice(categories),
            "unit_cost_cents": rng.randint(250, 25_000),
        }
        for index in range(shape["skus"])
    ]

    # Distinct (warehouse, sku) pairs, deterministically sampled.
    candidates = list(
        itertools.product(range(1, shape["warehouses"] + 1), range(1, shape["skus"] + 1))
    )
    pairs = rng.sample(candidates, shape["pairs"])

    # Group sizes: most pairs are unique, a guaranteed subset is duplicated.
    sizes = [rng.choice(size_options) for _ in pairs]
    duplicate_count = sum(1 for size in sizes if size >= 2)
    if duplicate_count < 2:  # Guarantee at least two duplicate groups.
        fixed = 0
        for index in range(len(sizes)):
            if fixed >= 2 - duplicate_count:
                break
            if sizes[index] < 2:
                sizes[index] = 2
                fixed += 1

    inventory: list[dict[str, Any]] = []
    next_id = 1
    for (warehouse_id, sku_id), size in zip(pairs, sizes):
        reorder_level = rng.choice([0, 0, 5, 10, 20])
        quantities = [rng.randint(2, 60)]
        for _ in range(size - 1):
            quantities.append(rng.randint(1, 40))
        for quantity in quantities:
            inventory.append(
                {
                    "id": next_id,
                    "warehouse_id": warehouse_id,
                    "sku_id": sku_id,
                    "quantity": quantity,
                    "reorder_level": reorder_level,
                    "reorder_flag": 1 if quantity <= reorder_level else 0,
                }
            )
            next_id += 1

    # Adjustments reference existing pairs; multiple adjustments per pair are allowed.
    duplicate_pairs = {pair for pair, size in zip(pairs, sizes) if size >= 2}
    adjustments: list[dict[str, Any]] = []
    next_adj_id = 1
    reasons = ["received", "damaged", "sold", "counted", "returned"]
    for index in range(shape["adjustments"]):
        if index == 0:
            delta = rng.randint(-40, -10)  # Guarantee at least one negative delta.
        elif index == 1:
            delta = rng.randint(5, 60)  # Guarantee at least one positive delta.
        else:
            delta = rng.choice([0, rng.randint(-30, 60)])
        pair = rng.choice(pairs)
        adjustments.append(
            {
                "id": next_adj_id,
                "warehouse_id": pair[0],
                "sku_id": pair[1],
                "delta": delta,
                "reason": rng.choice(reasons),
            }
        )
        next_adj_id += 1

    # Edge case: at least one duplicate group is untouched by adjustments.
    if duplicate_pairs and all(
        (adj["warehouse_id"], adj["sku_id"]) in duplicate_pairs for adj in adjustments
    ):
        singleton = [pair for pair in pairs if pair not in duplicate_pairs]
        fallback = singleton[0] if singleton else sorted(duplicate_pairs)[0]
        adjustments[0] = {
            **adjustments[0],
            "warehouse_id": fallback[0],
            "sku_id": fallback[1],
        }

    expected = _expected_inventory(inventory, adjustments)

    # Edge case: force at least one clamp-to-zero through negative adjustments.
    if not any(row["quantity"] == 0 for row in expected):
        candidate = max(expected, key=lambda row: row["quantity"] - row["reorder_level"])
        merged = sum(
            int(row["quantity"])
            for row in inventory
            if (row["warehouse_id"], row["sku_id"]) == (candidate["warehouse_id"], candidate["sku_id"])
        )
        applied = sum(
            int(adj["delta"])
            for adj in adjustments
            if (adj["warehouse_id"], adj["sku_id"]) == (candidate["warehouse_id"], candidate["sku_id"])
        )
        if merged + applied > 0:
            adjustments.append(
                {
                    "id": next_adj_id,
                    "warehouse_id": candidate["warehouse_id"],
                    "sku_id": candidate["sku_id"],
                    "delta": -(merged + applied),
                    "reason": "damaged",
                }
            )
            expected = _expected_inventory(inventory, adjustments)

    # Edge cases: guarantee at least one final reorder_flag both set and clear.
    # These are realized by editing the initial inventory rows (the source of
    # truth), so the verifier's independent recomputation matches the oracle.
    if not any(row["reorder_flag"] == 1 for row in expected):
        target = min(expected, key=lambda item: item["quantity"])
        for initial_row in inventory:
            if (initial_row["warehouse_id"], initial_row["sku_id"]) == (
                target["warehouse_id"],
                target["sku_id"],
            ):
                initial_row["reorder_level"] = max(
                    initial_row["reorder_level"], target["quantity"]
                )
        expected = _expected_inventory(inventory, adjustments)
    if not any(row["reorder_flag"] == 0 for row in expected):
        target = max(expected, key=lambda item: item["quantity"])
        for initial_row in inventory:
            if (initial_row["warehouse_id"], initial_row["sku_id"]) == (
                target["warehouse_id"],
                target["sku_id"],
            ):
                initial_row["reorder_level"] = 0
        expected = _expected_inventory(inventory, adjustments)

    return {
        "warehouses": warehouses,
        "skus": skus,
        "inventory": inventory,
        "adjustments": adjustments,
        "expected_inventory": expected,
        "duplicate_groups": sum(1 for size in sizes if size >= 2),
    }


def _write_db(path: Path, dataset: dict[str, Any]) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(SCHEMA_SQL)
        connection.executemany(
            "INSERT INTO warehouses (id, name, region) VALUES (?, ?, ?)",
            [(row["id"], row["name"], row["region"]) for row in dataset["warehouses"]],
        )
        connection.executemany(
            "INSERT INTO skus (id, sku, name, category, unit_cost_cents) VALUES (?, ?, ?, ?, ?)",
            [
                (row["id"], row["sku"], row["name"], row["category"], row["unit_cost_cents"])
                for row in dataset["skus"]
            ],
        )
        connection.executemany(
            "INSERT INTO inventory (id, warehouse_id, sku_id, quantity, reorder_level, reorder_flag) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    row["id"],
                    row["warehouse_id"],
                    row["sku_id"],
                    row["quantity"],
                    row["reorder_level"],
                    row["reorder_flag"],
                )
                for row in dataset["inventory"]
            ],
        )
        connection.executemany(
            "INSERT INTO adjustments (id, warehouse_id, sku_id, delta, reason) VALUES (?, ?, ?, ?, ?)",
            [
                (row["id"], row["warehouse_id"], row["sku_id"], row["delta"], row["reason"])
                for row in dataset["adjustments"]
            ],
        )
        connection.commit()
    finally:
        connection.close()


def generate_sqlite_task(root: str | Path, seed: int, difficulty: str = "medium") -> TaskSpec:
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {sorted(DIFFICULTIES)}")
    root_path = _root(root)
    task_id = f"sqlite-{TEMPLATE_ID}-{difficulty}-{seed}"
    task_path = _task_dir(root_path, task_id)
    manifest_path = task_path / "task.json"
    if manifest_path.exists():
        return TaskSpec.from_dict(read_json(manifest_path))

    initial = task_path / "initial"
    private = task_path / "private"
    initial.mkdir(parents=True, exist_ok=False)
    private.mkdir(parents=True, exist_ok=False)

    dataset = _build_dataset(seed, difficulty)
    db_path = initial / DB_FILENAME
    _write_db(db_path, dataset)

    contract = (
        "Open /workspace/store.db with the sqlite3 module and modify it in place.",
        "Merge duplicate inventory rows: for each (warehouse_id, sku_id) keep exactly the row "
        "with the smallest id, set its quantity to the sum of the group, set its reorder_level "
        "to the maximum reorder_level in the group, set its reorder_flag to 0, and delete the "
        "other rows of the group.",
        "Apply every adjustments row: add its delta to the quantity of the matching "
        "(warehouse_id, sku_id) inventory row, and clamp any negative result up to 0.",
        "After merging and adjustments, set reorder_flag to 1 when quantity <= reorder_level "
        "and to 0 otherwise.",
        "Do not modify warehouses, skus, or adjustments data, and do not change the schema: "
        "keep the same tables and columns, with no additions or removals.",
        "The final inventory table must have exactly one row per (warehouse_id, sku_id), all "
        "quantities non-negative, and every warehouse_id/sku_id referencing an existing row.",
    )
    prompt = (
        "Complete the SQLite data-reconciliation task in the isolated /workspace directory.\n\n"
        + "\n".join(f"- {item}" for item in contract)
        + "\n\nThe task is complete only when /workspace/store.db satisfies every rule "
        "(schema, reference data, and inventory state)."
    )
    (initial / "TASK.md").write_text(prompt + "\n", encoding="utf-8")

    oracle = {
        "warehouses": dataset["warehouses"],
        "skus": dataset["skus"],
        "adjustments": dataset["adjustments"],
        "initial_inventory": dataset["inventory"],
        "expected_inventory": dataset["expected_inventory"],
        "tables": list(TABLES),
        "table_columns": TABLE_COLUMNS,
    }
    write_json(private / "oracle.json", oracle)

    spec = TaskSpec(
        id=task_id,
        family="sqlite",
        template_id=TEMPLATE_ID,
        generator_version=GENERATOR_VERSION,
        seed=seed,
        difficulty=difficulty,
        prompt=prompt,
        contract=contract,
        public_metadata={
            "input_files": ["TASK.md", DB_FILENAME],
            "database": DB_FILENAME,
            "tables": list(TABLES),
            "warehouse_count": len(dataset["warehouses"]),
            "sku_count": len(dataset["skus"]),
            "inventory_rows": len(dataset["inventory"]),
            "adjustment_rows": len(dataset["adjustments"]),
            "duplicate_groups": dataset["duplicate_groups"],
        },
        workspace_ref=str(initial),
        verifier_ref=str(private / "oracle.json"),
    )
    write_json(manifest_path, spec.to_dict())
    return spec


def _open_submitted_db(db_path: Path) -> tuple[sqlite3.Connection, tempfile.TemporaryDirectory[str]]:
    """Open the submitted database on a private copy (handles WAL journals).

    Opening a copy keeps the verifier read-only against the agent's file and
    lets SQLite recover ``-wal``/``-shm`` sidecar files if the agent left the
    database in WAL mode.
    """
    temporary = tempfile.TemporaryDirectory()
    copy = Path(temporary.name) / DB_FILENAME
    shutil.copy2(db_path, copy)
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(db_path) + suffix)
        if sidecar.exists():
            shutil.copy2(sidecar, Path(str(copy) + suffix))
    connection = sqlite3.connect(copy)
    return connection, temporary


def _check_schema(connection: sqlite3.Connection, oracle: dict[str, Any]) -> str | None:
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    tables = {name for (name,) in rows if name != "sqlite_sequence"}
    if tables != set(oracle["tables"]):
        return "schema_mismatch"
    for table in oracle["tables"]:
        columns = [
            row[1]
            for row in connection.execute(f"PRAGMA table_info({_safe_id(table, 'table name')})").fetchall()
        ]
        if columns != oracle["table_columns"][table]:
            return "schema_mismatch"
    return None


def _check_reference_data(connection: sqlite3.Connection, oracle: dict[str, Any]) -> str | None:
    for table in ("warehouses", "skus", "adjustments"):
        rows = connection.execute(f"SELECT * FROM {_safe_id(table, 'table name')} ORDER BY id").fetchall()
        expected = oracle[table]
        if len(rows) != len(expected):
            return "reference_data_mismatch"
        columns = oracle["table_columns"][table]
        for row, item in zip(rows, expected):
            if list(row) != [item[column] for column in columns]:
                return "reference_data_mismatch"
    return None


def _check_inventory(connection: sqlite3.Connection, oracle: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    diagnostics: dict[str, Any] = {}

    duplicates = connection.execute(
        "SELECT warehouse_id, sku_id, COUNT(*) FROM inventory "
        "GROUP BY warehouse_id, sku_id HAVING COUNT(*) > 1"
    ).fetchall()
    if duplicates:
        diagnostics["duplicate_pairs"] = len(duplicates)
        return "semantic_mismatch", diagnostics

    negative = connection.execute(
        "SELECT COUNT(*) FROM inventory WHERE quantity < 0"
    ).fetchone()[0]
    if negative:
        diagnostics["negative_quantities"] = negative
        return "semantic_mismatch", diagnostics

    bad_fk = connection.execute(
        "SELECT COUNT(*) FROM inventory i "
        "LEFT JOIN warehouses w ON i.warehouse_id = w.id "
        "LEFT JOIN skus s ON i.sku_id = s.id "
        "WHERE w.id IS NULL OR s.id IS NULL"
    ).fetchone()[0]
    if bad_fk:
        diagnostics["invalid_foreign_keys"] = bad_fk
        return "semantic_mismatch", diagnostics

    initial_pairs = {(row["warehouse_id"], row["sku_id"]) for row in oracle["initial_inventory"]}
    submitted_pairs = {
        (warehouse_id, sku_id)
        for warehouse_id, sku_id in connection.execute(
            "SELECT warehouse_id, sku_id FROM inventory"
        ).fetchall()
    }
    if submitted_pairs != initial_pairs:
        diagnostics["pair_set_mismatch"] = sorted(
            submitted_pairs.symmetric_difference(initial_pairs)
        )
        return "semantic_mismatch", diagnostics

    expected = _expected_inventory(oracle["initial_inventory"], oracle["adjustments"])
    for item in expected:
        row = connection.execute(
            "SELECT id, quantity, reorder_level, reorder_flag FROM inventory "
            "WHERE warehouse_id = ? AND sku_id = ?",
            (item["warehouse_id"], item["sku_id"]),
        ).fetchone()
        if row is None:
            return "semantic_mismatch", diagnostics
        if list(row) != [item["id"], item["quantity"], item["reorder_level"], item["reorder_flag"]]:
            diagnostics["mismatched_rows"] = diagnostics.get("mismatched_rows", 0) + 1
            return "semantic_mismatch", diagnostics

    diagnostics["expected_rows"] = len(expected)
    diagnostics["actual_rows"] = len(submitted_pairs)
    diagnostics["reorder_flags_set"] = connection.execute(
        "SELECT COUNT(*) FROM inventory WHERE reorder_flag = 1"
    ).fetchone()[0]
    return None, diagnostics


def verify_sqlite_attempt(root: str | Path, task_id: str, attempt_id: str) -> VerificationResult:
    root_path = _root(root)
    spec = load_task(root_path, task_id)
    attempt = load_attempt(root_path, attempt_id)
    if attempt.task_id != spec.id:
        raise ValueError("attempt does not belong to task")

    oracle = read_json(Path(spec.verifier_ref))
    db_path = Path(attempt.workspace_ref) / DB_FILENAME

    if not db_path.exists():
        result = VerificationResult(
            success=False,
            verifier_id=VERIFIER_ID,
            verifier_version=VERIFIER_VERSION,
            failure_code="missing_db",
            diagnostics={"required_db": DB_FILENAME},
        )
    else:
        connection = None
        temporary = None
        try:
            connection, temporary = _open_submitted_db(db_path)
            connection.execute("SELECT name FROM sqlite_master").fetchall()  # Probe validity.
            failure_code = _check_schema(connection, oracle)
            diagnostics: dict[str, Any] = {}
            if failure_code is None:
                failure_code = _check_reference_data(connection, oracle)
            if failure_code is None:
                failure_code, diagnostics = _check_inventory(connection, oracle)
            result = VerificationResult(
                success=failure_code is None,
                verifier_id=VERIFIER_ID,
                verifier_version=VERIFIER_VERSION,
                failure_code=failure_code,
                diagnostics=diagnostics,
            )
        except (sqlite3.DatabaseError, OSError) as error:
            result = VerificationResult(
                success=False,
                verifier_id=VERIFIER_ID,
                verifier_version=VERIFIER_VERSION,
                failure_code="invalid_db",
                diagnostics={"error": str(error)},
            )
        finally:
            if connection is not None:
                connection.close()
            if temporary is not None:
                temporary.cleanup()

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
