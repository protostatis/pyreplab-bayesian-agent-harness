from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pyreplab_harness.artifact_gym import (
    generate_artifact_task,
    prepare_attempt,
    verify_artifact_attempt,
)
from pyreplab_harness.io_utils import read_json, write_json


class ArtifactGymTest(unittest.TestCase):
    def test_generation_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            task_a = generate_artifact_task(first, 42, "medium")
            task_b = generate_artifact_task(second, 42, "medium")
            self.assertEqual(task_a.id, task_b.id)
            self.assertEqual(task_a.prompt, task_b.prompt)
            self.assertEqual(
                read_json(Path(task_a.verifier_ref)),
                read_json(Path(task_b.verifier_ref)),
            )

    def test_correct_result_passes_semantic_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_artifact_task(directory, 7, "easy")
            attempt = prepare_attempt(directory, task.id, "attempt-1", "direct")
            expected = read_json(Path(task.verifier_ref))["expected"]
            write_json(Path(attempt.workspace_ref) / "result.json", expected)
            result = verify_artifact_attempt(directory, task.id, attempt.attempt_id)
            self.assertTrue(result.success)
            self.assertIsNone(result.failure_code)

    def test_wrong_result_fails_semantic_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_artifact_task(directory, 7, "hard")
            attempt = prepare_attempt(directory, task.id, "attempt-2", "direct")
            write_json(Path(attempt.workspace_ref) / "result.json", [])
            result = verify_artifact_attempt(directory, task.id, attempt.attempt_id)
            self.assertFalse(result.success)
            self.assertEqual(result.failure_code, "semantic_mismatch")

    def test_missing_output_has_distinct_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_artifact_task(directory, 9, "medium")
            attempt = prepare_attempt(directory, task.id, "attempt-3", "direct")
            result = verify_artifact_attempt(directory, task.id, attempt.attempt_id)
            self.assertFalse(result.success)
            self.assertEqual(result.failure_code, "missing_output")


if __name__ == "__main__":
    unittest.main()
