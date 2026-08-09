from __future__ import annotations

import base64
import shutil
import tempfile
import unittest
from pathlib import Path

from pyreplab_harness.artifact_gym import load_attempt, load_task, prepare_attempt
from pyreplab_harness.io_utils import read_json, write_json
from pyreplab_harness.shell_gym import generate_shell_task, verify_shell_attempt


def _oracle(task) -> dict:
    return read_json(Path(task.verifier_ref))


def _tree_snapshot(directory: Path) -> dict[str, bytes]:
    snapshot: dict[str, bytes] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            snapshot[path.relative_to(directory).as_posix()] = path.read_bytes()
    return snapshot


def _materialize_correct(attempt, oracle: dict) -> Path:
    """Replay the expected transformation using only oracle data."""
    workspace = Path(attempt.workspace_ref)
    shutil.rmtree(workspace / "incoming", ignore_errors=True)
    for rel, info in oracle["files"].items():
        target = workspace / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(base64.b64decode(info["content_b64"]))
    manifest = {
        "schema_version": 1,
        "files": [
            {
                "path": rel,
                "category": info["category"],
                "sha256": info["sha256"],
                "size": info["size"],
                "original_name": info["original_name"],
            }
            for rel, info in sorted(oracle["files"].items())
        ],
        "duplicates": [dict(group) for group in oracle["duplicates"]],
        "category_counts": dict(oracle["counts"]),
    }
    write_json(workspace / "manifest.json", manifest)
    return workspace


class ShellGymTest(unittest.TestCase):
    def test_generation_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            task_a = generate_shell_task(first, 42, "medium")
            task_b = generate_shell_task(second, 42, "medium")
            self.assertEqual(task_a.id, task_b.id)
            self.assertEqual(task_a.family, "shell")
            self.assertEqual(task_a.prompt, task_b.prompt)
            self.assertEqual(task_a.contract, task_b.contract)
            self.assertEqual(
                read_json(Path(task_a.verifier_ref)),
                read_json(Path(task_b.verifier_ref)),
            )
            self.assertEqual(
                _tree_snapshot(Path(task_a.workspace_ref)),
                _tree_snapshot(Path(task_b.workspace_ref)),
            )

    def test_difficulty_scales_file_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            easy = generate_shell_task(directory, 5, "easy")
            medium = generate_shell_task(directory, 5, "medium")
            hard = generate_shell_task(directory, 5, "hard")
            easy_count = len(_oracle(easy)["files"])
            medium_count = len(_oracle(medium)["files"])
            hard_count = len(_oracle(hard)["files"])
            self.assertLess(easy_count, medium_count)
            self.assertLess(medium_count, hard_count)
            self.assertGreaterEqual(len(_oracle(easy)["duplicates"]), 1)
            self.assertGreaterEqual(len(_oracle(hard)["duplicates"]), 1)

    def test_contract_explicitly_matches_strict_file_set_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_shell_task(directory, 21, "hard")
            contract = " ".join(task.contract)
            self.assertIn("canonical extension", contract)
            self.assertIn("Do not leave temporary or helper files", contract)
            self.assertIn("image=.img", contract)

    def test_generic_helpers_work_with_shell_task(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_shell_task(directory, 9, "easy")
            attempt = prepare_attempt(directory, task.id, "attempt-g", "direct")
            self.assertEqual(load_task(directory, task.id).id, task.id)
            loaded = load_attempt(directory, attempt.attempt_id)
            self.assertEqual(loaded.status, "prepared")
            self.assertEqual(loaded.task_id, task.id)

    def test_correct_transformed_workspace_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_shell_task(directory, 7, "medium")
            attempt = prepare_attempt(directory, task.id, "attempt-1", "direct")
            workspace = _materialize_correct(attempt, _oracle(task))
            self.assertTrue((workspace / "manifest.json").exists())

            result = verify_shell_attempt(directory, task.id, attempt.attempt_id)
            self.assertTrue(result.success)
            self.assertIsNone(result.failure_code)

            # Attempt status and verification ref are updated consistently.
            record = load_attempt(directory, attempt.attempt_id)
            self.assertEqual(record.status, "verified")
            self.assertIsNotNone(record.verification_ref)
            self.assertTrue(Path(record.verification_ref).exists())
            self.assertTrue(
                read_json(Path(record.verification_ref)) == result.to_dict()
            )

    def test_untouched_workspace_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_shell_task(directory, 9, "medium")
            attempt = prepare_attempt(directory, task.id, "attempt-2", "direct")
            result = verify_shell_attempt(directory, task.id, attempt.attempt_id)
            self.assertFalse(result.success)
            self.assertEqual(result.failure_code, "file_set_mismatch")
            self.assertTrue(result.diagnostics["missing"])
            self.assertTrue(result.diagnostics["extra"])

    def test_wrong_content_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_shell_task(directory, 7, "hard")
            attempt = prepare_attempt(directory, task.id, "attempt-3", "direct")
            oracle = _oracle(task)
            workspace = _materialize_correct(attempt, oracle)
            target = next(iter(oracle["files"]))
            (workspace / target).write_bytes(b"corrupted bytes\n")
            result = verify_shell_attempt(directory, task.id, attempt.attempt_id)
            self.assertFalse(result.success)
            self.assertEqual(result.failure_code, "content_mismatch")
            self.assertEqual(result.diagnostics["path"], target)

    def test_missing_manifest_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_shell_task(directory, 11, "easy")
            attempt = prepare_attempt(directory, task.id, "attempt-4", "direct")
            oracle = _oracle(task)
            workspace = _materialize_correct(attempt, oracle)
            (workspace / "manifest.json").unlink()
            result = verify_shell_attempt(directory, task.id, attempt.attempt_id)
            self.assertFalse(result.success)
            self.assertEqual(result.failure_code, "missing_manifest")

    def test_malformed_manifest_is_distinct_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_shell_task(directory, 13, "medium")
            attempt = prepare_attempt(directory, task.id, "attempt-5", "direct")
            workspace = _materialize_correct(attempt, _oracle(task))
            (workspace / "manifest.json").write_text("{ this is not json", encoding="utf-8")
            result = verify_shell_attempt(directory, task.id, attempt.attempt_id)
            self.assertFalse(result.success)
            self.assertEqual(result.failure_code, "invalid_manifest")

    def test_bad_schema_manifest_is_distinct_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_shell_task(directory, 17, "easy")
            attempt = prepare_attempt(directory, task.id, "attempt-6", "direct")
            workspace = _materialize_correct(attempt, _oracle(task))
            write_json(
                workspace / "manifest.json",
                {"schema_version": 1, "files": "not-a-list", "duplicates": [], "category_counts": {}},
            )
            result = verify_shell_attempt(directory, task.id, attempt.attempt_id)
            self.assertFalse(result.success)
            self.assertEqual(result.failure_code, "invalid_manifest")
            self.assertNotEqual(result.failure_code, "manifest_mismatch")

    def test_semantically_wrong_manifest_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_shell_task(directory, 19, "medium")
            attempt = prepare_attempt(directory, task.id, "attempt-7", "direct")
            oracle = _oracle(task)
            workspace = _materialize_correct(attempt, oracle)
            manifest = read_json(workspace / "manifest.json")
            manifest["files"][0]["sha256"] = "0" * 64
            write_json(workspace / "manifest.json", manifest)
            result = verify_shell_attempt(directory, task.id, attempt.attempt_id)
            self.assertFalse(result.success)
            self.assertEqual(result.failure_code, "manifest_mismatch")

    def test_duplicate_kept_alongside_original_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_shell_task(directory, 23, "medium")
            attempt = prepare_attempt(directory, task.id, "attempt-8", "direct")
            oracle = _oracle(task)
            workspace = _materialize_correct(attempt, oracle)
            # Keep a duplicate file back in the tree: the expected tree must not
            # contain any file beyond the kept set.
            group = oracle["duplicates"][0]
            original = group["original_names"][0]
            kept = workspace / group["kept"]
            extra = workspace / original
            extra.write_bytes(kept.read_bytes())
            result = verify_shell_attempt(directory, task.id, attempt.attempt_id)
            self.assertFalse(result.success)
            self.assertEqual(result.failure_code, "file_set_mismatch")
            self.assertIn(original, result.diagnostics["extra"])


if __name__ == "__main__":
    unittest.main()
