from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pyreplab_harness.artifact_gym import prepare_attempt
from pyreplab_harness.io_utils import write_json
from pyreplab_harness.unbrowser_gym import (
    EXPECTED_RESULT,
    generate_unbrowser_task,
    verify_unbrowser_attempt,
)
from pyreplab_harness.unbrowser_rpc import UNBROWSER_SMOKE_URL


class UnbrowserGymTest(unittest.TestCase):
    def test_generation_is_deterministic_and_url_is_fixed(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            task_a = generate_unbrowser_task(first, 7, "easy")
            task_b = generate_unbrowser_task(second, 7, "easy")
            self.assertEqual(task_a.id, task_b.id)
            self.assertEqual(task_a.prompt, task_b.prompt)
            self.assertEqual(task_a.public_metadata["allowed_url"], UNBROWSER_SMOKE_URL)
            self.assertNotIn(EXPECTED_RESULT["heading"], task_a.prompt)

    def test_expected_heading_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_unbrowser_task(directory, 7, "easy")
            attempt = prepare_attempt(directory, task.id, "ub-pass", "correct")
            write_json(Path(attempt.workspace_ref) / "result.json", EXPECTED_RESULT)
            result = verify_unbrowser_attempt(directory, task.id, attempt.attempt_id)
            self.assertTrue(result.success)
            self.assertIsNone(result.failure_code)

    def test_wrong_heading_and_missing_output_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_unbrowser_task(directory, 7, "easy")
            wrong = prepare_attempt(directory, task.id, "ub-wrong", "wrong")
            write_json(
                Path(wrong.workspace_ref) / "result.json",
                {"heading": "Wrong paragraph"},
            )
            mismatch = verify_unbrowser_attempt(
                directory, task.id, wrong.attempt_id
            )
            self.assertFalse(mismatch.success)
            self.assertEqual(mismatch.failure_code, "semantic_mismatch")

            missing = prepare_attempt(directory, task.id, "ub-missing", "wrong")
            missing_result = verify_unbrowser_attempt(
                directory, task.id, missing.attempt_id
            )
            self.assertFalse(missing_result.success)
            self.assertEqual(missing_result.failure_code, "missing_output")


if __name__ == "__main__":
    unittest.main()
