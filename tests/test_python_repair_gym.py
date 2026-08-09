from __future__ import annotations

import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pyreplab_harness.artifact_gym import load_attempt, prepare_attempt  # Generic helpers must work with the python-repair manifest.
from pyreplab_harness.io_utils import read_json
from pyreplab_harness.python_repair_gym import (
    _result_from_report,
    _verifier_command,
    generate_python_repair_task,
    verify_python_repair_attempt,
)
from pyreplab_harness.sandbox import sandbox_available

# Textual fixes that turn each template's medium-difficulty buggy source into
# the correct implementation. Used by test_expected_fix_passes.
_FIXES = {
    "range-boundary-v1": [
        ("if low < v < high", "if low <= v <= high"),
    ],
    "eligibility-v1": [
        ("return age >= MIN_AGE or has_id", "return age >= MIN_AGE and has_id"),
    ],
    "total-bill-v1": [
        ("order[\"unit_price_cents\"] for order", "order[\"qty\"] * order[\"unit_price_cents\"] for order"),
    ],
}

# Behavioral tests exercise the OS-level sandbox, so they are skipped where
# bwrap/systemd are genuinely unavailable (e.g. local macOS). Pure
# command-construction and runner-semantics tests run everywhere.
_REQUIRES_SANDBOX = "requires bwrap and a systemd user session"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class PythonRepairGymTest(unittest.TestCase):
    def test_generation_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            task_a = generate_python_repair_task(first, 42, "medium")
            task_b = generate_python_repair_task(second, 42, "medium")
            self.assertEqual(task_a.id, task_b.id)
            self.assertEqual(task_a.family, "python_repair")
            self.assertEqual(task_a.prompt, task_b.prompt)
            self.assertEqual(task_a.contract, task_b.contract)
            self.assertEqual(task_a.public_metadata, task_b.public_metadata)
            for name in ("TASK.md", "solution.py", "test_public.py"):
                self.assertEqual(
                    _read_text(Path(task_a.workspace_ref) / name),
                    _read_text(Path(task_b.workspace_ref) / name),
                )
            for name in ("reference.py", "verify_runner.py"):
                self.assertEqual(
                    _read_text(Path(task_a.verifier_ref) / name),
                    _read_text(Path(task_b.verifier_ref) / name),
                )
            self.assertEqual(
                read_json(Path(task_a.verifier_ref) / "hidden_cases.json"),
                read_json(Path(task_b.verifier_ref) / "hidden_cases.json"),
            )
            self.assertGreaterEqual(task_a.public_metadata["hidden_case_count"], 1)

    def test_generation_is_idempotent_in_same_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_python_repair_task(directory, 5, "easy")
            again = generate_python_repair_task(directory, 5, "easy")
            self.assertEqual(task.id, again.id)
            self.assertEqual(task.prompt, again.prompt)

    def test_difficulty_scales_hidden_case_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            easy = generate_python_repair_task(directory, 3, "easy")
            hard = generate_python_repair_task(directory, 3, "hard")
            self.assertNotEqual(easy.id, hard.id)
            self.assertGreater(
                hard.public_metadata["hidden_case_count"],
                easy.public_metadata["hidden_case_count"],
            )

    @unittest.skipUnless(sandbox_available(), _REQUIRES_SANDBOX)
    def test_reference_source_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_python_repair_task(directory, 7, "medium")
            attempt = prepare_attempt(directory, task.id, "attempt-correct", "direct")
            reference = _read_text(Path(task.verifier_ref) / "reference.py")
            (Path(attempt.workspace_ref) / "solution.py").write_text(
                reference, encoding="utf-8"
            )

            result = verify_python_repair_attempt(directory, task.id, attempt.attempt_id)
            self.assertTrue(result.success)
            self.assertIsNone(result.failure_code)
            self.assertGreaterEqual(result.diagnostics["cases_run"], 1)
            self.assertEqual(
                result.diagnostics["cases_passed"],
                result.diagnostics["cases_run"],
            )

            attempt_dir = Path(directory).resolve() / "attempts" / attempt.attempt_id
            self.assertEqual(read_json(attempt_dir / "verification.json"), result.to_dict())
            record = load_attempt(directory, attempt.attempt_id)
            self.assertEqual(record.status, "verified")
            self.assertEqual(record.verification_ref, str(attempt_dir / "verification.json"))

    @unittest.skipUnless(sandbox_available(), _REQUIRES_SANDBOX)
    def test_expected_fix_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_python_repair_task(directory, 11, "medium")
            template_id = task.public_metadata["template_id"]
            self.assertIn(template_id, _FIXES)
            attempt = prepare_attempt(directory, task.id, "attempt-fix", "direct")
            source = _read_text(Path(attempt.workspace_ref) / "solution.py")
            for old, new in _FIXES[template_id]:
                self.assertIn(old, source)
                source = source.replace(old, new)
            (Path(attempt.workspace_ref) / "solution.py").write_text(
                source, encoding="utf-8"
            )

            result = verify_python_repair_attempt(directory, task.id, attempt.attempt_id)
            self.assertTrue(result.success)
            self.assertIsNone(result.failure_code)

    @unittest.skipUnless(sandbox_available(), _REQUIRES_SANDBOX)
    def test_untouched_bug_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_python_repair_task(directory, 7, "medium")
            attempt = prepare_attempt(directory, task.id, "attempt-bug", "direct")
            result = verify_python_repair_attempt(directory, task.id, attempt.attempt_id)
            self.assertFalse(result.success)
            # Behavioral failure only: never missing_file/timeout/success.
            self.assertIn(result.failure_code, {"test_failure", "runtime_error"})

    @unittest.skipUnless(sandbox_available(), _REQUIRES_SANDBOX)
    def test_syntax_error_is_distinct_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_python_repair_task(directory, 7, "medium")
            attempt = prepare_attempt(directory, task.id, "attempt-syntax", "direct")
            (Path(attempt.workspace_ref) / "solution.py").write_text(
                "def count_in_range(values, low, high):\n    return sum(1 for v in values if)\n",
                encoding="utf-8",
            )
            result = verify_python_repair_attempt(directory, task.id, attempt.attempt_id)
            self.assertFalse(result.success)
            self.assertEqual(result.failure_code, "syntax_error")

    def test_missing_file_is_distinct_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_python_repair_task(directory, 9, "medium")
            attempt = prepare_attempt(directory, task.id, "attempt-missing", "direct")
            (Path(attempt.workspace_ref) / "solution.py").unlink()
            result = verify_python_repair_attempt(directory, task.id, attempt.attempt_id)
            self.assertFalse(result.success)
            self.assertEqual(result.failure_code, "missing_file")

    @unittest.skipUnless(sandbox_available(), _REQUIRES_SANDBOX)
    def test_timeout_is_distinct_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_python_repair_task(directory, 9, "medium")
            attempt = prepare_attempt(directory, task.id, "attempt-hang", "direct")
            (Path(attempt.workspace_ref) / "solution.py").write_text(
                "while True:\n    pass\n", encoding="utf-8"
            )
            result = verify_python_repair_attempt(
                directory, task.id, attempt.attempt_id, timeout_seconds=2
            )
            self.assertFalse(result.success)
            self.assertEqual(result.failure_code, "timeout")

    def test_unsafe_task_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            generate_python_repair_task(directory, 5, "easy")
            with self.assertRaises(ValueError):
                verify_python_repair_attempt(directory, "../evil", "anything")

    def test_sandbox_unavailable_is_distinct_with_no_host_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_python_repair_task(directory, 9, "medium")
            attempt = prepare_attempt(directory, task.id, "attempt-nosandbox", "direct")
            with mock.patch(
                "pyreplab_harness.python_repair_gym.sandbox_available", return_value=False
            ):
                result = verify_python_repair_attempt(
                    directory, task.id, attempt.attempt_id
                )
            self.assertFalse(result.success)
            self.assertEqual(result.failure_code, "sandbox_unavailable")
            self.assertIn("never executed on the host", result.diagnostics["reason"])
            record = load_attempt(directory, attempt.attempt_id)
            self.assertEqual(record.status, "verified")

    def test_verifier_command_uses_only_in_sandbox_paths(self) -> None:
        command = _verifier_command("/usr/bin/python3", "solution", "count_in_range")
        self.assertIn("-I -S -B", command)
        self.assertIn("/private/verify_runner.py", command)
        self.assertIn("/private/hidden_cases.json", command)
        self.assertIn("/output/verification.out.json", command)
        self.assertIn("/workspace", command)
        # No host mount flags or host paths leak into the sandboxed command.
        self.assertNotIn("--ro-bind", command)
        self.assertNotIn("--bind", command)

    def test_result_from_report_maps_taxonomy_and_caps_diagnostics(self) -> None:
        success = _result_from_report(
            {"status": "success", "cases_run": 4, "cases_passed": 4}
        )
        self.assertTrue(success.success)
        self.assertIsNone(success.failure_code)

        syntax = _result_from_report({"status": "syntax_error", "detail": "bad code"})
        self.assertEqual(syntax.failure_code, "syntax_error")
        self.assertEqual(syntax.diagnostics["detail"], "bad code")

        runtime = _result_from_report({"status": "runtime_error", "detail": "boom"})
        self.assertEqual(runtime.failure_code, "runtime_error")

        long_detail = _result_from_report(
            {"status": "runtime_error", "detail": "x" * 5000}
        )
        self.assertLessEqual(len(long_detail.diagnostics["detail"]), 2000)

        test_failure = _result_from_report(
            {
                "status": "test_failure",
                "cases_run": 3,
                "cases_passed": 1,
                "failures": [
                    {"args": [1], "expected": 2, "actual": 3},
                    {"args": ["y" * 5000], "expected": "e" * 5000, "actual": "a" * 5000},
                ],
            }
        )
        self.assertEqual(test_failure.failure_code, "test_failure")
        self.assertEqual(len(test_failure.diagnostics["failures"]), 2)
        self.assertLessEqual(len(test_failure.diagnostics["failures"][1]["args"]), 402)
        self.assertLessEqual(len(test_failure.diagnostics["failures"][1]["expected"]), 402)

        unknown = _result_from_report({"status": "alien"})
        self.assertEqual(unknown.failure_code, "runtime_error")
        self.assertIn("unknown verifier status", unknown.diagnostics["error"])

        bare = _result_from_report({"status": "syntax_error"})
        self.assertIsNone(bare.diagnostics["detail"])

    @unittest.skipUnless(sandbox_available(), _REQUIRES_SANDBOX)
    def test_security_isolation_of_submitted_code(self) -> None:
        """Submitted code that can read /home, reach the localhost model
        endpoint, or write outside the allowed mounts must fail verification;
        isolated submitted code must pass every hidden case."""
        with tempfile.TemporaryDirectory() as directory:
            task = generate_python_repair_task(directory, 21, "medium")
            attempt = prepare_attempt(directory, task.id, "attempt-sec", "direct")
            reference = _read_text(Path(task.verifier_ref) / "reference.py")
            canary = Path.home() / f".pyreplab-gym-canary-{os.getpid()}.txt"
            canary.write_text("secret", encoding="utf-8")
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            try:
                probe_prefix = (
                    "import os, socket\n"
                    "_pyreplab_leak = 0\n"
                    f"try:\n"
                    f"    with open({str(canary)!r}, 'r', encoding='utf-8') as handle:\n"
                    f"        if handle.read() == 'secret':\n"
                    f"            _pyreplab_leak += 1\n"
                    f"except Exception:\n"
                    f"    pass\n"
                    f"try:\n"
                    f"    conn = socket.create_connection(('127.0.0.1', {port}), timeout=3)\n"
                    f"    conn.close()\n"
                    f"    _pyreplab_leak += 1\n"
                    f"except Exception:\n"
                    f"    pass\n"
                    f"try:\n"
                    f"    with open('/etc/pyreplab-evil', 'w', encoding='utf-8') as handle:\n"
                    f"        handle.write('x')\n"
                    f"    _pyreplab_leak += 1\n"
                    f"except Exception:\n"
                    f"    pass\n"
                    f"if _pyreplab_leak:\n"
                    f"    raise RuntimeError('sandbox isolation leak detected')\n\n"
                )
                (Path(attempt.workspace_ref) / "solution.py").write_text(
                    probe_prefix + reference, encoding="utf-8"
                )
                result = verify_python_repair_attempt(
                    directory, task.id, attempt.attempt_id
                )
                self.assertTrue(result.success, result.diagnostics)
                self.assertEqual(
                    result.diagnostics["cases_passed"],
                    result.diagnostics["cases_run"],
                )
            finally:
                listener.close()
                try:
                    canary.unlink()
                except OSError:
                    pass


if __name__ == "__main__":
    unittest.main()
