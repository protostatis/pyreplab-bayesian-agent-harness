"""Deterministic conformance tests for the v2 result-submission detector.

Zero-mismatch contract (v12 qualification, G5): for every case in the
write/read mechanism matrix, the detector's label must equal the expected
label exactly — any mismatch fails qualification.

The fake sandbox executes real shell commands via subprocess in a temp
workspace (mirroring BubblewrapSandbox semantics without requiring bwrap on
macOS), so mutation detection is exercised against real filesystem effects.
"""
from __future__ import annotations

import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any

from pyreplab_harness.sandbox import SandboxResult
from pyreplab_harness.worker import (
    RESULT_FILE_BASENAME,
    _result_mutated,
    _snapshot_result_file,
    handle_request,
)


class FakeSandbox:
    """Executes commands with subprocess in ``workspace`` (host-side)."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    def execute(self, command: str, timeout_seconds: int | None = None) -> SandboxResult:
        proc = subprocess.run(
            ["bash", "-lc", command],
            cwd=self.workspace,
            capture_output=True,
            text=True,
            timeout=timeout_seconds or 30,
        )
        return SandboxResult(
            stdout=proc.stdout,
            stderr=proc.stderr,
            exit_code=proc.returncode,
            timed_out=False,
            truncated=False,
        )


class ResultSubmissionDetectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.sandbox = FakeSandbox(self.workspace)
        self.config = Any  # unused by exec branch assertions below

    # -- helpers ----------------------------------------------------------
    def _run_exec(self, command: str) -> dict[str, Any]:
        response, _stop = handle_request(
            self.sandbox,
            {"id": "t1", "method": "exec", "params": {"command": command}},
            type("C", (), {"max_timeout": 30})(),
            unbrowser=None,
        )
        assert response["ok"] is True, response
        return response["result"]

    def _submission(self, command: str) -> tuple[bool, dict[str, Any]]:
        """Mirror the TS-layer gate exactly (the single authoritative predicate):
        submission = exit 0 AND mutated AND target exists after the call."""
        result = self._run_exec(command)
        rf = result.get("result_file") or {}
        submission = (
            result.get("exit_code") == 0
            and rf.get("mutated") is True
            and (rf.get("after") or {}).get("exists") is True
        )
        return bool(submission), result

    # -- WRITE mechanisms: every one must flag a submission -----------------
    def test_write_matrix_all_flag_submission(self) -> None:
        key = '{"verification_key": "KEY_test"}'
        cases = {
            "redirect_overwrite": f"echo '{key}' > {RESULT_FILE_BASENAME}",
            "redirect_append_new_file": f"printf '%s' '{key}' >> {RESULT_FILE_BASENAME}",
            "tee": f"tee {RESULT_FILE_BASENAME} <<< '{key}' >/dev/null",
            "copy_from_other": (
                f"echo x > draft.json && cp draft.json {RESULT_FILE_BASENAME}"
            ),
            "move_over_target": (
                f"echo y > tmp.json && mv tmp.json {RESULT_FILE_BASENAME}"
            ),
            "heredoc": f"cat <<EOF > {RESULT_FILE_BASENAME}\n{key}\nEOF",
            "variable_path": (
                f"F=$PWD/{RESULT_FILE_BASENAME}; echo '{key}' > \"$F\""
            ),
            "symlink_target_write": (
                f"ln -s $PWD/{RESULT_FILE_BASENAME} rlink.json "
                f"&& echo '{key}' > rlink.json"
            ),
            "python_write": (
                "python3 -c \"open('"
                + RESULT_FILE_BASENAME + "','w').write('py')\""
            ),
        }
        for name, command in cases.items():
            with self.subTest(case=name):
                # each case starts from a clean workspace
                for p in self.workspace.iterdir():
                    p.unlink()
                submission, result = self._submission(command)
                self.assertTrue(submission, f"{name} must be a submission")
                rf = result["result_file"]
                self.assertTrue(rf["after"]["exists"])
                self.assertIsInstance(rf["before"], dict)
                write = result  # receipt hash pair present at worker level
                if submission:
                    self.assertNotEqual(
                        (rf["before"] or {}).get("sha256"),
                        rf["after"].get("sha256"),
                    )

    # -- READ mechanisms: none may flag a submission ------------------------
    def test_read_matrix_none_flag_submission(self) -> None:
        self._run_exec(f"echo seed > {RESULT_FILE_BASENAME}")
        read_cases = {
            "cat": f"cat {RESULT_FILE_BASENAME}",
            "head": f"head -n 5 {RESULT_FILE_BASENAME}",
            "wc": f"wc -c {RESULT_FILE_BASENAME}",
            "grep": f"grep verification {RESULT_FILE_BASENAME} || true",
            "ls": f"ls -la {RESULT_FILE_BASENAME}",
            "python_read": (
                "python3 -c \"open('" + RESULT_FILE_BASENAME + "').read()\""
            ),
            "unrelated_path_mention": "echo result.json",
            "touch_only": f"touch {RESULT_FILE_BASENAME}",
        }
        for name, command in read_cases.items():
            with self.subTest(case=name):
                submission, _ = self._submission(command)
                self.assertFalse(submission, f"{name} must NOT be a submission")

    # -- failure / dedupe / partial semantics -------------------------------
    def test_failed_write_is_not_submission_even_when_mutated(self) -> None:
        before_snapshot = _snapshot_result_file(self.workspace)
        proc = subprocess.run(
            ["bash", "-lc", f"echo partial > {RESULT_FILE_BASENAME} && false"],
            cwd=self.workspace, capture_output=True, text=True,
        )
        after_snapshot = _snapshot_result_file(self.workspace)
        self.assertNotEqual(proc.returncode, 0)
        mutated = _result_mutated(before_snapshot, after_snapshot)
        # File DID mutate, but exit != 0 => not a valid submission.
        self.assertTrue(mutated)
        self.assertFalse(proc.returncode == 0 and mutated)

    def test_repeated_identical_rewrite_dedupes_to_false(self) -> None:
        cmd = f"echo same > {RESULT_FILE_BASENAME}"
        first, _ = self._submission(cmd)
        second, _ = self._submission(cmd)
        self.assertTrue(first)
        self.assertFalse(second, "byte-identical rewrite must dedupe to False")

    def test_ordering_via_hash_pair_chain(self) -> None:
        _, r1 = self._submission(f"echo v1 > {RESULT_FILE_BASENAME}")
        _, r2 = self._submission(f"echo v2 > {RESULT_FILE_BASENAME}")
        h1 = r1["result_file"]["after"]["sha256"]
        h2_before = r2["result_file"]["before"]["sha256"]
        h2_after = r2["result_file"]["after"]["sha256"]
        self.assertEqual(h1, h2_before, "receipts must chain across calls")
        self.assertNotEqual(h2_before, h2_after)

    def test_missing_then_created_counts_as_mutation(self) -> None:
        submission, result = self._submission(
            f"echo new > {RESULT_FILE_BASENAME}"
        )
        self.assertTrue(submission)
        rf = result["result_file"]
        self.assertFalse(rf["before"]["exists"])
        self.assertTrue(rf["after"]["exists"])

    def test_deleted_file_counts_as_mutation_but_not_submission_shape(self) -> None:
        self._run_exec(f"echo gone > {RESULT_FILE_BASENAME}")
        submission, result = self._submission(f"rm {RESULT_FILE_BASENAME}")
        rf = result["result_file"]
        self.assertTrue(_result_mutated(rf["before"], rf["after"]))
        self.assertFalse(submission, "deletion is not a submission")

    def test_worker_payload_shape_stable(self) -> None:
        result = self._run_exec("echo hi")
        self.assertIn("result_file", result)
        rf = result["result_file"]
        self.assertIn("mutated", rf)
        self.assertIn("exists", rf["before"])
        # Absent file: snapshot is exactly {"exists": False} (no hash fields).
        self.assertFalse(rf["before"]["exists"])
        self.assertNotIn("sha256", rf["before"])

    def test_hash_function_consistency(self) -> None:
        target = self.workspace / RESULT_FILE_BASENAME
        target.write_bytes(b"abc")
        snap = _snapshot_result_file(self.workspace)
        self.assertEqual(snap["sha256"], hashlib.sha256(b"abc").hexdigest())


if __name__ == "__main__":
    unittest.main()
