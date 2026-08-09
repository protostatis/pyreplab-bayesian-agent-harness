from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from pyreplab_harness import external_o1 as o1


def _source(task_id: str, resolved: bool | None, project: str = "org") -> dict:
    return {
        "issue_name": task_id,
        "project": project,
        "resolved": resolved,
        # Must be ignored: both real releases default this field to False.
        "success": False,
        "full_conversation_jsonl": "POST-ACTION-DO-NOT-IMPORT",
        "patch": "POST-ACTION-DO-NOT-IMPORT",
        "tests_status": "POST-ACTION-DO-NOT-IMPORT",
    }


def _task(task_id: str) -> dict:
    return {
        "instance_id": task_id,
        "repo": "org/repo",
        "problem_statement": f"Fix the behavior described by {task_id}.",
        "patch": "GOLD-DO-NOT-IMPORT",
        "test_patch": "GOLD-DO-NOT-IMPORT",
    }


class O1NormalizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = [
            _source("both-pass", True),
            _source("baseline-only", True),
            _source("native-only", False),
            _source("both-fail", False),
            _source("baseline-missing-report", None),
            _source("both-missing-report", None),
        ]
        self.native = [
            _source("both-pass", True),
            _source("baseline-only", False),
            _source("native-only", True),
            _source("both-fail", False),
            _source("baseline-missing-report", True),
            _source("both-missing-report", None),
            _source("native-only-row", True),
        ]
        self.tasks = [
            _task(task_id)
            for task_id in (
                "both-pass",
                "baseline-only",
                "native-only",
                "both-fail",
                "baseline-missing-report",
                "both-missing-report",
                "native-only-row",
            )
        ]

    def test_complete_pairs_and_paired_table(self) -> None:
        rows, registry, exclusions, summary = o1.build_o1_dataset(
            self.baseline, self.native, self.tasks
        )
        self.assertEqual(summary["paired_tasks"], 4)
        self.assertEqual(summary["rows"], 8)
        self.assertEqual(
            summary["paired_outcomes"],
            {
                "both_pass": 1,
                "baseline_only_pass": 1,
                "native_only_pass": 1,
                "both_fail": 1,
            },
        )
        self.assertEqual(summary["success"], {"baseline": 2, "native": 2})
        self.assertEqual(
            {item["treatment_id"] for item in registry["treatments"]},
            {o1.BASELINE_TREATMENT_ID, o1.NATIVE_TREATMENT_ID},
        )
        self.assertEqual(exclusions["excluded_task_count"], 3)
        self.assertEqual(
            exclusions["reason_counts"],
            {
                "PAIR_EXCLUDE_BASELINE_REPORT_MISSING": 2,
                "PAIR_EXCLUDE_BASELINE_ROW_ABSENT": 1,
                "PAIR_EXCLUDE_NATIVE_REPORT_MISSING": 1,
            },
        )

    def test_pair_features_are_identical_except_treatment(self) -> None:
        rows, _registry, _exclusions, _summary = o1.build_o1_dataset(
            self.baseline, self.native, self.tasks
        )
        pair = [row for row in rows if row["task_id"] == "native-only"]
        self.assertEqual(len(pair), 2)
        left = dict(pair[0]["model_input"])
        right = dict(pair[1]["model_input"])
        for key in ("policy_id", "policy_version", "treatment"):
            left.pop(key)
            right.pop(key)
        self.assertEqual(left, right)
        self.assertNotEqual(
            pair[0]["model_input"]["policy_id"],
            pair[1]["model_input"]["policy_id"],
        )
        self.assertEqual(pair[0]["split"], pair[1]["split"])
        self.assertNotEqual(
            pair[0]["model_input"]["treatment"]["text"],
            pair[1]["model_input"]["treatment"]["text"],
        )
        self.assertNotEqual(
            pair[0]["model_input"]["treatment"]["tool_interface"],
            pair[1]["model_input"]["treatment"]["tool_interface"],
        )

    def test_post_action_and_oracle_fields_never_enter_model_input(self) -> None:
        rows, _registry, _exclusions, _summary = o1.build_o1_dataset(
            self.baseline, self.native, self.tasks
        )
        forbidden = {
            "success",
            "resolved",
            "full_conversation_jsonl",
            "patch",
            "test_patch",
            "tests_status",
            "verified_success",
            "failure_code",
        }
        for row in rows:
            self.assertTrue(forbidden.isdisjoint(row["model_input"]))
            self.assertEqual(row["model_input"]["public_metadata"], {})
            self.assertTrue(forbidden.isdisjoint(row["model_input"]["treatment"]))

    def test_missing_canonical_task_excludes_whole_pair(self) -> None:
        tasks = [task for task in self.tasks if task["instance_id"] != "both-pass"]
        rows, _registry, exclusions, summary = o1.build_o1_dataset(
            self.baseline, self.native, tasks
        )
        self.assertEqual(summary["paired_tasks"], 3)
        self.assertFalse(any(row["task_id"] == "both-pass" for row in rows))
        entry = next(item for item in exclusions["excluded"] if item["task_id"] == "both-pass")
        self.assertEqual(entry["reasons"], ["PAIR_EXCLUDE_CANONICAL_TASK_MISSING"])

    def test_duplicate_or_non_boolean_outcomes_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate baseline ID"):
            o1.build_o1_dataset(
                self.baseline + [_source("both-pass", True)], self.native, self.tasks
            )
        malformed = list(self.baseline)
        malformed[0] = _source("both-pass", 1)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "must be bool or null"):
            o1.build_o1_dataset(malformed, self.native, self.tasks)


class O1WriterCliTest(unittest.TestCase):
    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    def test_writer_is_deterministic_and_cli_emits_summary(self) -> None:
        baseline = [_source("a", True), _source("b", False)]
        native = [_source("a", False), _source("b", True)]
        tasks = [_task("a"), _task("b")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_path = root / "baseline.jsonl"
            native_path = root / "native.jsonl"
            tasks_path = root / "tasks.jsonl"
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            self._write_jsonl(baseline_path, baseline)
            self._write_jsonl(native_path, native)
            self._write_jsonl(tasks_path, tasks)

            summary = o1.write_o1_dataset(
                baseline_path, native_path, tasks_path, first
            )
            o1.write_o1_dataset(baseline_path, native_path, tasks_path, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(summary["paired_tasks"], 2)
            self.assertTrue(Path(summary["registry_path"]).is_file())
            self.assertTrue(Path(summary["exclusions_path"]).is_file())

            output = root / "cli.jsonl"
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = o1.main(
                    [
                        str(baseline_path),
                        str(native_path),
                        str(tasks_path),
                        str(output),
                    ]
                )
            self.assertEqual(code, 0)
            payload = json.loads(buffer.getvalue())
            self.assertEqual(payload["rows"], 4)
            self.assertTrue(output.is_file())

    def test_parquet_requires_projection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.parquet"
            path.write_bytes(b"not parquet")
            with self.assertRaisesRegex(ValueError, "Parquet projection is required"):
                o1.read_records(path)


if __name__ == "__main__":
    unittest.main()
