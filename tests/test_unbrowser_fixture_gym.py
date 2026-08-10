from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pyreplab_harness.artifact_gym import prepare_attempt
from pyreplab_harness.fixture_templates import TEMPLATES, generate_nonce
from pyreplab_harness.io_utils import write_json
from pyreplab_harness.unbrowser_fixture_gym import (
    FIXTURE_BASE_URL,
    generate_unbrowser_fixture_task,
    verify_unbrowser_fixture_attempt,
)


class UnbrowserFixtureGymTest(unittest.TestCase):
    def test_generation_is_deterministic_and_url_is_fixed(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            task_a = generate_unbrowser_fixture_task(first, 7, "easy", "single_page_extraction")
            task_b = generate_unbrowser_fixture_task(second, 7, "easy", "single_page_extraction")
            self.assertEqual(task_a.id, task_b.id)
            self.assertEqual(task_a.prompt, task_b.prompt)
            self.assertEqual(task_a.family, "unbrowser_fixture")
            allowed_url = task_a.public_metadata["allowed_url"]
            self.assertTrue(allowed_url.startswith(FIXTURE_BASE_URL))
            self.assertIn("single_page_extraction", allowed_url)
            self.assertIn("/7/", allowed_url)
            self.assertIn("/easy", allowed_url)

    def test_oracle_contains_expected_nonce(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_unbrowser_fixture_task(directory, 42, "medium", "table_filter_sort")
            oracle_path = Path(task.verifier_ref)
            self.assertTrue(oracle_path.exists())
            oracle = json.loads(oracle_path.read_text())
            self.assertIn("nonce", oracle)
            self.assertIn("expected_answer", oracle)
            self.assertEqual(oracle["nonce"], oracle["expected_answer"])
            self.assertEqual(oracle["verification_type"], "exact_match")
            self.assertTrue(oracle["nonce"].startswith("KEY_"))

    def test_different_seeds_produce_different_oracles(self) -> None:
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            task_a = generate_unbrowser_fixture_task(d1, 7, "easy", "single_page_extraction")
            task_b = generate_unbrowser_fixture_task(d2, 42, "easy", "single_page_extraction")
            oracle_a = json.loads(Path(task_a.verifier_ref).read_text())
            oracle_b = json.loads(Path(task_b.verifier_ref).read_text())
            self.assertNotEqual(oracle_a["nonce"], oracle_b["nonce"])
            self.assertNotEqual(task_a.id, task_b.id)

    def test_different_templates_produce_different_oracles(self) -> None:
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            task_a = generate_unbrowser_fixture_task(d1, 7, "easy", "single_page_extraction")
            task_b = generate_unbrowser_fixture_task(d2, 7, "easy", "table_filter_sort")
            oracle_a = json.loads(Path(task_a.verifier_ref).read_text())
            oracle_b = json.loads(Path(task_b.verifier_ref).read_text())
            self.assertNotEqual(oracle_a["nonce"], oracle_b["nonce"])
            self.assertNotEqual(task_a.template_id, task_b.template_id)

    def test_verifier_accepts_correct_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_unbrowser_fixture_task(directory, 7, "easy", "single_page_extraction")
            oracle = json.loads(Path(task.verifier_ref).read_text())
            nonce = oracle["nonce"]

            attempt = prepare_attempt(directory, task.id, "ub-fix-pass", "correct")
            write_json(
                Path(attempt.workspace_ref) / "result.json",
                {"verification_key": nonce},
            )
            result = verify_unbrowser_fixture_attempt(
                directory, task.id, attempt.attempt_id
            )
            self.assertTrue(result.success)
            self.assertIsNone(result.failure_code)

    def test_verifier_rejects_wrong_nonce(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_unbrowser_fixture_task(directory, 7, "easy", "single_page_extraction")

            attempt = prepare_attempt(directory, task.id, "ub-fix-wrong", "wrong")
            write_json(
                Path(attempt.workspace_ref) / "result.json",
                {"verification_key": "KEY_wrong0001"},
            )
            result = verify_unbrowser_fixture_attempt(
                directory, task.id, attempt.attempt_id
            )
            self.assertFalse(result.success)
            self.assertEqual(result.failure_code, "nonce_mismatch")

    def test_verifier_rejects_missing_result_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_unbrowser_fixture_task(directory, 7, "easy", "single_page_extraction")
            attempt = prepare_attempt(directory, task.id, "ub-fix-missing", "wrong")
            # Do not write result.json
            result = verify_unbrowser_fixture_attempt(
                directory, task.id, attempt.attempt_id
            )
            self.assertFalse(result.success)
            self.assertEqual(result.failure_code, "missing_output")
            attempt_dir = Path(directory) / "attempts" / attempt.attempt_id
            self.assertTrue((attempt_dir / "verification.json").is_file())
            persisted_attempt = json.loads((attempt_dir / "attempt.json").read_text())
            self.assertEqual(persisted_attempt["status"], "verified")

    def test_verifier_rejects_malformed_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_unbrowser_fixture_task(directory, 7, "easy", "single_page_extraction")
            attempt = prepare_attempt(directory, task.id, "ub-fix-malformed", "wrong")
            (Path(attempt.workspace_ref) / "result.json").write_text(
                "not valid json {{", encoding="utf-8"
            )
            result = verify_unbrowser_fixture_attempt(
                directory, task.id, attempt.attempt_id
            )
            self.assertFalse(result.success)
            self.assertEqual(result.failure_code, "invalid_json")

    def test_verifier_rejects_missing_verification_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_unbrowser_fixture_task(directory, 7, "easy", "single_page_extraction")
            attempt = prepare_attempt(directory, task.id, "ub-fix-nokey", "wrong")
            write_json(
                Path(attempt.workspace_ref) / "result.json",
                {"heading": "something else"},
            )
            result = verify_unbrowser_fixture_attempt(
                directory, task.id, attempt.attempt_id
            )
            self.assertFalse(result.success)
            self.assertEqual(result.failure_code, "missing_key")

    def test_verifier_rejects_result_that_is_not_dict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_unbrowser_fixture_task(directory, 7, "easy", "single_page_extraction")
            attempt = prepare_attempt(directory, task.id, "ub-fix-list", "wrong")
            write_json(
                Path(attempt.workspace_ref) / "result.json",
                ["KEY_something"],
            )
            result = verify_unbrowser_fixture_attempt(
                directory, task.id, attempt.attempt_id
            )
            self.assertFalse(result.success)
            self.assertEqual(result.failure_code, "wrong_type")

    def test_verifier_rejects_non_string_verification_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_unbrowser_fixture_task(directory, 7, "easy", "single_page_extraction")
            attempt = prepare_attempt(directory, task.id, "ub-fix-int", "wrong")
            write_json(
                Path(attempt.workspace_ref) / "result.json",
                {"verification_key": 12345},
            )
            result = verify_unbrowser_fixture_attempt(
                directory, task.id, attempt.attempt_id
            )
            self.assertFalse(result.success)
            self.assertEqual(result.failure_code, "wrong_key_type")

    def test_deterministic_given_template_seed_difficulty(self) -> None:
        """Same (template, seed, difficulty) always produces the same task and oracle."""
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            task1 = generate_unbrowser_fixture_task(d1, 99, "hard", "distractor_recovery")
            task2 = generate_unbrowser_fixture_task(d2, 99, "hard", "distractor_recovery")
            self.assertEqual(task1.id, task2.id)
            self.assertEqual(task1.prompt, task2.prompt)
            oracle1 = json.loads(Path(task1.verifier_ref).read_text())
            oracle2 = json.loads(Path(task2.verifier_ref).read_text())
            self.assertEqual(oracle1, oracle2)

    def test_all_templates_generate_valid_tasks(self) -> None:
        """Every template produces a valid task with a fixture URL and oracle."""
        for template in TEMPLATES:
            with tempfile.TemporaryDirectory() as directory, self.subTest(template=template):
                task = generate_unbrowser_fixture_task(directory, 1, "easy", template)
                self.assertEqual(task.family, "unbrowser_fixture")
                self.assertEqual(task.template_id, template)
                # Verify the fixture URL is valid
                url = task.public_metadata["allowed_url"]
                self.assertTrue(url.startswith(FIXTURE_BASE_URL))
                # Verify oracle exists and has nonce
                oracle = json.loads(Path(task.verifier_ref).read_text())
                self.assertIn("nonce", oracle)
                self.assertTrue(oracle["nonce"].startswith("KEY_"))
                # Verify TASK.md exists
                task_md = Path(task.workspace_ref) / "TASK.md"
                self.assertTrue(task_md.exists())
                # Verify contract mentions the fixture URL
                self.assertIn(FIXTURE_BASE_URL, "\n".join(task.contract))

    def test_verifier_handles_nonexistent_task(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = verify_unbrowser_fixture_attempt(
                directory, "nonexistent-task", "nonexistent-attempt"
            )
            self.assertFalse(result.success)
            self.assertEqual(result.failure_code, "task_not_found")

    def test_verifier_handles_nonexistent_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_unbrowser_fixture_task(directory, 7, "easy", "single_page_extraction")
            result = verify_unbrowser_fixture_attempt(
                directory, task.id, "nonexistent-attempt"
            )
            self.assertFalse(result.success)
            self.assertEqual(result.failure_code, "attempt_not_found")

    def test_rejects_invalid_difficulty(self) -> None:
        with self.assertRaisesRegex(ValueError, "difficulty must be"):
            generate_unbrowser_fixture_task(tempfile.mkdtemp(), 7, "impossible")

    def test_rejects_invalid_template(self) -> None:
        with self.assertRaisesRegex(ValueError, "template must be"):
            generate_unbrowser_fixture_task(
                tempfile.mkdtemp(), 7, "easy", "nonexistent_template"
            )

    def test_subsequent_call_returns_cached_task(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task1 = generate_unbrowser_fixture_task(directory, 7, "medium", "form_entry_validation")
            task2 = generate_unbrowser_fixture_task(directory, 7, "medium", "form_entry_validation")
            self.assertIsNot(task1, task2)
            self.assertEqual(task1.id, task2.id)
            self.assertEqual(task1.to_dict(), task2.to_dict())

    def test_task_role_is_frozen_and_cached_role_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_unbrowser_fixture_task(
                directory,
                7,
                "easy",
                "single_page_extraction",
                task_role="T_pilot",
            )
            self.assertEqual(task.public_metadata["task_role"], "T_pilot")
            with self.assertRaisesRegex(ValueError, "cached task role mismatch"):
                generate_unbrowser_fixture_task(
                    directory, 7, "easy", "single_page_extraction"
                )


if __name__ == "__main__":
    unittest.main()
