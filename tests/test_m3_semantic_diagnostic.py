from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from pyreplab_harness.m3_semantic_diagnostic import (
    REPORT_SCHEMA,
    build_diagnostic,
    main,
)
from pyreplab_harness.m3_semantic_dataset import privacy_scan


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _make_package(root: Path) -> Path:
    package = root / "package"
    attempt_id = "attempt-001"
    normalized_rel = f"raw/attempts/{attempt_id}/pi-events.normalized.json"
    raw_rel = f"raw/attempts/{attempt_id}/pi-events.jsonl"
    normalized_path = package / normalized_rel
    raw_path = package / raw_rel
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(
        "\n".join(
            json.dumps(event, sort_keys=True)
            for event in [
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "toolCall",
                                "name": "semantic_table",
                                "arguments": {
                                    "filters": [{"column": "secret", "value": "private"}],
                                    "limit": 1,
                                    "projection": ["secret"],
                                    "sort": {"column": "secret", "direction": "asc"},
                                },
                            },
                            {
                                "type": "toolCall",
                                "name": "unbrowser",
                                "arguments": {"action": "query", "selector": "secret"},
                            },
                        ],
                    },
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        normalized_path,
        {
            "tool_call_count": 4,
            "tool_executions": [
                {
                    "tool_name": "unbrowser",
                    "budget_rejected": False,
                    "operation_aborted": False,
                    "pre_execution_rejected": False,
                    "result": {"details": {"action": "navigate", "url": "private"}},
                },
                {
                    "tool_name": "semantic_table",
                    "budget_rejected": False,
                    "operation_aborted": False,
                    "pre_execution_rejected": False,
                    "result": {"details": {"semantic_payload": {"secret": "private"}}},
                },
                {
                    "tool_name": "unbrowser",
                    "budget_rejected": False,
                    "operation_aborted": False,
                    "pre_execution_rejected": False,
                    "result": {"details": {"action": "query", "selector": "secret"}},
                },
                {
                    "tool_name": "bash",
                    "budget_rejected": False,
                    "operation_aborted": False,
                    "pre_execution_rejected": False,
                    "result": {"details": {"result_submission": True}},
                },
            ],
        },
    )
    row = {
        "raw": {"path": raw_rel},
        "task": {"template": "table_filter_sort"},
        "treatment": {"capability": "table_specialist", "tool_call_limit": 12},
        "execution": {
            "tool_call_count": 4,
            "termination_class": "normal_completion",
            "usage": {"total_tokens": 100},
            "timing": {"total_seconds": 2.5},
        },
        "mechanism": {
            "admitted_tool_call_count": 4,
            "rejected_tool_call_count": 0,
        },
        "outcome": {"failure_code": None, "output_tokens": 20, "success": True},
        "panel": {"execution_position": 0, "rollout_replica": 0},
    }
    attempts_path = package / "data" / "attempts.jsonl"
    attempts_path.parent.mkdir(parents=True, exist_ok=True)
    attempts_path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")

    inventory = [
        {
            "bytes": normalized_path.stat().st_size,
            "path": normalized_rel,
            "sha256": _sha256(normalized_path),
        },
        {
            "bytes": raw_path.stat().st_size,
            "path": raw_rel,
            "sha256": _sha256(raw_path),
        },
    ]
    inventory_path = package / "raw" / "inventory.jsonl"
    inventory_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in inventory),
        encoding="utf-8",
    )
    gate_path = package / "analysis" / "gate-report.json"
    _write_json(
        gate_path,
        {
            "authoritative_gate": {
                "decision": "replication_no_go",
                "passed": False,
                "reasons": ["stable table-only tasks 0 < minimum 2"],
                "stability": {
                    "discordant_cell_count": 10,
                    "favorable_form_count": 8,
                    "favorable_table_count": 8,
                    "stable_form_only_count": 6,
                    "stable_table_only_count": 0,
                },
            }
        },
    )
    _write_json(
        package / "MANIFEST.json",
        {
            "contract_hash": "c" * 64,
            "counts": {"cells": 1},
            "dataset_id": "test-dataset",
            "governance_role": "canary_excluded",
            "identities": {"manifest_hash": "m" * 64},
            "task_role": "T_canary",
        },
    )
    return package


class SemanticDiagnosticTests(unittest.TestCase):
    def test_builds_deterministic_privacy_safe_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            package = _make_package(Path(temp))
            with patch(
                "pyreplab_harness.m3_semantic_diagnostic.verify_package",
                return_value={"passed": True},
            ):
                first = build_diagnostic(package)
                second = build_diagnostic(package)

            self.assertEqual(first, second)
            self.assertEqual(first["schema_version"], REPORT_SCHEMA)
            self.assertEqual(first["aggregate"]["attempts"], 1)
            self.assertEqual(first["aggregate"]["successes"], 1)
            self.assertEqual(
                first["aggregate"]["action_counts"],
                {
                    "result_submission": 1,
                    "semantic_table": 1,
                    "unbrowser_navigate": 1,
                    "unbrowser_query": 1,
                },
            )
            self.assertEqual(
                first["aggregate"]["request_shape_counts"],
                {"bounded_semantic_table": 1, "generic_unbrowser_query": 1},
            )
            self.assertEqual(
                first["table_cell_stability"]["table_specialist"][
                    "replica_outcome_patterns"
                ],
                {"success": 1},
            )
            self.assertEqual(privacy_scan(first), [])
            rendered = json.dumps(first, sort_keys=True)
            for private_value in ("private", "secret"):
                self.assertNotIn(private_value, rendered)

    def test_rejects_unverified_package(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            package = _make_package(Path(temp))
            with patch(
                "pyreplab_harness.m3_semantic_diagnostic.verify_package",
                return_value={"passed": False},
            ):
                with self.assertRaisesRegex(ValueError, "verification failed"):
                    build_diagnostic(package)

    def test_rejects_normalized_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            package = _make_package(Path(temp))
            normalized = next((package / "raw" / "attempts").glob("*/pi-events.normalized.json"))
            normalized.write_text("{}", encoding="utf-8")
            with patch(
                "pyreplab_harness.m3_semantic_diagnostic.verify_package",
                return_value={"passed": True},
            ):
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    build_diagnostic(package)

    def test_rejects_boolean_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            package = _make_package(Path(temp))
            attempts_path = package / "data" / "attempts.jsonl"
            row = json.loads(attempts_path.read_text())
            row["treatment"]["tool_call_limit"] = True
            attempts_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with patch(
                "pyreplab_harness.m3_semantic_diagnostic.verify_package",
                return_value={"passed": True},
            ):
                with self.assertRaisesRegex(ValueError, "counts must be integers"):
                    build_diagnostic(package)

    def test_cli_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            package = _make_package(root)
            output = root / "report.json"
            with patch(
                "pyreplab_harness.m3_semantic_diagnostic.verify_package",
                return_value={"passed": True},
            ):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(main([str(package), "--output", str(output)]), 0)
            self.assertEqual(json.loads(output.read_text())["schema_version"], REPORT_SCHEMA)


if __name__ == "__main__":
    unittest.main()
