from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from pyreplab_harness.artifact_gym import prepare_attempt  # Generic helper must work with the sqlite manifest.
from pyreplab_harness.io_utils import read_json
from pyreplab_harness.sqlite_gym import generate_sqlite_task, verify_sqlite_attempt

DB_FILENAME = "store.db"
INVENTORY_COLUMNS = ["id", "warehouse_id", "sku_id", "quantity", "reorder_level", "reorder_flag"]


def _read_tables(db_path: Path) -> dict[str, list[tuple[object, ...]]]:
    connection = sqlite3.connect(db_path)
    try:
        return {
            table: connection.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()
            for table in ("warehouses", "skus", "inventory", "adjustments")
        }
    finally:
        connection.close()


def _apply_expected(workspace: Path, oracle: dict[str, object]) -> None:
    """Write the oracle-derived reconciled inventory into the submitted DB."""
    connection = sqlite3.connect(workspace / DB_FILENAME)
    try:
        connection.execute("DELETE FROM inventory")
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
                for row in oracle["expected_inventory"]
            ],
        )
        connection.commit()
    finally:
        connection.close()


class SqliteGymTest(unittest.TestCase):
    def test_generation_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            task_a = generate_sqlite_task(first, 42, "medium")
            task_b = generate_sqlite_task(second, 42, "medium")
            self.assertEqual(task_a.id, task_b.id)
            self.assertEqual(task_a.family, "sqlite")
            self.assertEqual(task_a.prompt, task_b.prompt)
            self.assertEqual(task_a.contract, task_b.contract)
            self.assertEqual(
                read_json(Path(task_a.verifier_ref)),
                read_json(Path(task_b.verifier_ref)),
            )
            self.assertEqual(
                _read_tables(Path(task_a.workspace_ref) / DB_FILENAME),
                _read_tables(Path(task_b.workspace_ref) / DB_FILENAME),
            )
            self.assertGreaterEqual(task_a.public_metadata["duplicate_groups"], 2)

    def test_correct_oracle_derived_submission_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_sqlite_task(directory, 7, "easy")
            attempt = prepare_attempt(directory, task.id, "attempt-correct", "direct")
            oracle = read_json(Path(task.verifier_ref))
            # The reconciliation must be nontrivial: merged state differs from initial state.
            self.assertNotEqual(oracle["expected_inventory"], oracle["initial_inventory"])
            _apply_expected(Path(attempt.workspace_ref), oracle)

            result = verify_sqlite_attempt(directory, task.id, attempt.attempt_id)
            self.assertTrue(result.success)
            self.assertIsNone(result.failure_code)

            attempt_dir = Path(directory).resolve() / "attempts" / attempt.attempt_id
            self.assertEqual(read_json(attempt_dir / "verification.json"), result.to_dict())
            record = read_json(attempt_dir / "attempt.json")
            self.assertEqual(record["status"], "verified")
            self.assertEqual(record["verification_ref"], str(attempt_dir / "verification.json"))

    def test_no_modification_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_sqlite_task(directory, 7, "medium")
            attempt = prepare_attempt(directory, task.id, "attempt-none", "direct")
            result = verify_sqlite_attempt(directory, task.id, attempt.attempt_id)
            self.assertFalse(result.success)
            self.assertEqual(result.failure_code, "semantic_mismatch")

    def test_wrong_modification_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_sqlite_task(directory, 7, "medium")
            attempt = prepare_attempt(directory, task.id, "attempt-wrong", "direct")
            connection = sqlite3.connect(Path(attempt.workspace_ref) / DB_FILENAME)
            try:
                # Removes duplicates but keeps unmerged quantities and wrong flags:
                # a plausible but incorrect reconciliation.
                connection.execute(
                    "DELETE FROM inventory WHERE id NOT IN "
                    "(SELECT MIN(id) FROM inventory GROUP BY warehouse_id, sku_id)"
                )
                connection.execute("UPDATE inventory SET reorder_flag = 0")
                connection.commit()
            finally:
                connection.close()
            result = verify_sqlite_attempt(directory, task.id, attempt.attempt_id)
            self.assertFalse(result.success)
            self.assertEqual(result.failure_code, "semantic_mismatch")

    def test_missing_db_has_distinct_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_sqlite_task(directory, 9, "medium")
            attempt = prepare_attempt(directory, task.id, "attempt-missing", "direct")
            (Path(attempt.workspace_ref) / DB_FILENAME).unlink()
            result = verify_sqlite_attempt(directory, task.id, attempt.attempt_id)
            self.assertFalse(result.success)
            self.assertEqual(result.failure_code, "missing_db")

    def test_malformed_db_has_distinct_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_sqlite_task(directory, 9, "easy")
            attempt = prepare_attempt(directory, task.id, "attempt-malformed", "direct")
            (Path(attempt.workspace_ref) / DB_FILENAME).write_text(
                "definitely not a sqlite file", encoding="utf-8"
            )
            result = verify_sqlite_attempt(directory, task.id, attempt.attempt_id)
            self.assertFalse(result.success)
            self.assertEqual(result.failure_code, "invalid_db")

    def test_hard_difficulty_scales_and_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            easy = generate_sqlite_task(directory, 3, "easy")
            hard = generate_sqlite_task(directory, 3, "hard")
            self.assertNotEqual(easy.id, hard.id)
            self.assertGreater(
                hard.public_metadata["inventory_rows"],
                easy.public_metadata["inventory_rows"],
            )
            attempt = prepare_attempt(directory, hard.id, "attempt-hard", "deliberate")
            oracle = read_json(Path(hard.verifier_ref))
            _apply_expected(Path(attempt.workspace_ref), oracle)
            result = verify_sqlite_attempt(directory, hard.id, attempt.attempt_id)
            self.assertTrue(result.success)

    def test_unsafe_task_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            generate_sqlite_task(directory, 5, "easy")
            with self.assertRaises(ValueError):
                verify_sqlite_attempt(directory, "../evil", "anything")


if __name__ == "__main__":
    unittest.main()
