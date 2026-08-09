from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pyreplab_harness.poc_gate import evaluate_gate, main


def _record(index: int, direct: bool, deliberate: bool, *, version: str = "4") -> dict:
    attempts = {}
    for policy, outcome in (("direct", direct), ("deliberate", deliberate)):
        attempts[policy] = {
            "policy": {"id": policy, "version": version},
            "verification": {"success": outcome},
        }
    return {
        "key": f"pair/artifact/easy/seed={index}",
        "status": "completed",
        "mode": "pair",
        "policy_version": version,
        "duration_seconds": 10 + index,
        "result": {"attempts": attempts},
    }


def _write(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


class PocGateTest(unittest.TestCase):
    def test_passes_complete_diverse_batch(self) -> None:
        records = [
            _record(1, True, True),
            _record(2, True, False),
            _record(3, False, True),
            _record(4, False, False),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "batch.jsonl"
            _write(path, records)
            report = evaluate_gate(path, expected_jobs=4, min_disagreement=0.25)
        self.assertTrue(report["passed"])
        self.assertEqual(report["usable_pairs"], 4)
        self.assertEqual(report["disagreement_rate"], 0.5)
        self.assertEqual(report["matrix"]["direct_only"], 1)
        self.assertEqual(report["matrix"]["deliberate_only"], 1)

    def test_fails_incomplete_or_degenerate_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "batch.jsonl"
            _write(path, [_record(1, True, True)])
            report = evaluate_gate(path, expected_jobs=2)
        self.assertFalse(report["passed"])
        self.assertTrue(any("expected 2" in reason for reason in report["reasons"]))
        self.assertTrue(any("degenerate" in reason for reason in report["reasons"]))

    def test_fails_mixed_policy_version(self) -> None:
        records = [_record(1, True, False, version="3"), _record(2, False, True)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "batch.jsonl"
            _write(path, records)
            report = evaluate_gate(path, expected_jobs=2)
        self.assertFalse(report["passed"])
        self.assertTrue(any("policy version" in reason for reason in report["reasons"]))

    def test_cli_writes_report_and_uses_gate_exit_code(self) -> None:
        records = [
            _record(1, True, True),
            _record(2, True, False),
            _record(3, False, True),
            _record(4, False, False),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            batch = Path(tmp) / "batch.jsonl"
            output = Path(tmp) / "gate.json"
            _write(batch, records)
            rc = main(
                [
                    str(batch),
                    "--expected-jobs",
                    "4",
                    "--min-disagreement",
                    "0.25",
                    "--output",
                    str(output),
                ]
            )
            saved = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(rc, 0)
        self.assertTrue(saved["passed"])


if __name__ == "__main__":
    unittest.main()
