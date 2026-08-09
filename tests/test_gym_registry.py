from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

from pyreplab_harness.cli import main as cli_main
from pyreplab_harness.gym_registry import FAMILIES, generate_task, prepare_attempt, verify_attempt
from pyreplab_harness.orchestrator import (
    RemoteConfig,
    _pair_order,
    _remote_command,
    _summary_ok,
    build_parser,
    run_smoke,
)


class RegistryTest(unittest.TestCase):
    def test_canonical_families(self) -> None:
        self.assertEqual(FAMILIES, ("artifact", "sqlite", "shell", "python_repair"))

    def test_generate_task_rejects_unknown_family(self) -> None:
        with self.assertRaises(ValueError) as context:
            generate_task("bogus", "/tmp/whatever", 7, "easy")
        message = str(context.exception)
        self.assertIn("unknown family", message)
        self.assertIn("artifact", message)
        self.assertIn("python_repair", message)

    def test_verify_attempt_rejects_unknown_family(self) -> None:
        with self.assertRaises(ValueError):
            verify_attempt("bogus", "/tmp/whatever", "task-1", "attempt-1")

    def test_every_family_generates_prepares_and_untouched_verify_fails(self) -> None:
        for family in FAMILIES:
            with self.subTest(family=family):
                with tempfile.TemporaryDirectory() as directory:
                    task = generate_task(family, directory, 42, "easy")
                    self.assertEqual(task.family, family)
                    self.assertEqual(task.difficulty, "easy")
                    attempt = prepare_attempt(
                        directory, task.id, f"attempt-{family}", "direct"
                    )
                    result = verify_attempt(
                        family, directory, task.id, attempt.attempt_id
                    )
                    self.assertFalse(result.success)
                    self.assertIsNotNone(result.failure_code)

    def test_python_repair_untouched_verify_fails_without_sandbox(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_task("python_repair", directory, 3, "medium")
            attempt = prepare_attempt(directory, task.id, "attempt-pr", "deliberate")
            result = verify_attempt(
                "python_repair", directory, task.id, attempt.attempt_id
            )
            self.assertFalse(result.success)
            # Locally bwrap/systemd are unavailable, so the verifier returns
            # sandbox_unavailable; on a host with a sandbox an untouched buggy
            # module fails with test_failure/runtime_error instead. Either way
            # it counts as a valid failed untouched attempt.
            self.assertIn(
                result.failure_code,
                {"sandbox_unavailable", "test_failure", "runtime_error"},
            )

    def test_generate_task_is_idempotent_within_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = generate_task("sqlite", directory, 11, "hard")
            second = generate_task("sqlite", directory, 11, "hard")
            self.assertEqual(first.id, second.id)
            self.assertEqual(first.to_dict(), second.to_dict())

    def test_generic_cli_generate_and_verify_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = cli_main(
                    [
                        "generate",
                        "--family",
                        "sqlite",
                        "--root",
                        directory,
                        "--seed",
                        "7",
                        "--difficulty",
                        "easy",
                    ]
                )
            self.assertEqual(code, 0)
            task = json.loads(buffer.getvalue())
            self.assertEqual(task["family"], "sqlite")

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = cli_main(
                    [
                        "prepare-attempt",
                        "--root",
                        directory,
                        "--task-id",
                        task["id"],
                        "--attempt-id",
                        "cli-att",
                        "--policy-id",
                        "direct",
                    ]
                )
            self.assertEqual(code, 0)

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = cli_main(
                    [
                        "verify",
                        "--family",
                        "sqlite",
                        "--root",
                        directory,
                        "--task-id",
                        task["id"],
                        "--attempt-id",
                        "cli-att",
                    ]
                )
            self.assertEqual(code, 0)
            result = json.loads(buffer.getvalue())
            self.assertFalse(result["success"])

    def test_legacy_artifact_commands_still_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = cli_main(
                    [
                        "generate-artifact",
                        "--root",
                        directory,
                        "--seed",
                        "5",
                        "--difficulty",
                        "medium",
                    ]
                )
            self.assertEqual(code, 0)
            task = json.loads(buffer.getvalue())
            self.assertEqual(task["family"], "artifact")

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = cli_main(
                    [
                        "prepare-attempt",
                        "--root",
                        directory,
                        "--task-id",
                        task["id"],
                        "--attempt-id",
                        "legacy-att",
                        "--policy-id",
                        "direct",
                    ]
                )
            self.assertEqual(code, 0)

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = cli_main(
                    [
                        "verify-artifact",
                        "--root",
                        directory,
                        "--task-id",
                        task["id"],
                        "--attempt-id",
                        "legacy-att",
                    ]
                )
            self.assertEqual(code, 0)
            result = json.loads(buffer.getvalue())
            self.assertFalse(result["success"])


class OrchestratorTest(unittest.TestCase):
    def test_pair_order_is_deterministic_from_seed(self) -> None:
        first = _pair_order(7, ["direct", "deliberate"])
        second = _pair_order(7, ["direct", "deliberate"])
        self.assertEqual(first, second)
        self.assertEqual(sorted(first), ["deliberate", "direct"])
        self.assertEqual(len(first), 2)

    def test_remote_commands_use_generic_family_dispatch(self) -> None:
        config = RemoteConfig(host="ubuntu-local", project="/p", run_root="/r")
        generate_remote = _remote_command(
            config,
            ["generate", "--family", "sqlite", "--root", "/r", "--seed", "7", "--difficulty", "easy"],
        )[-1]
        self.assertIn("generate --family sqlite", generate_remote)
        verify_remote = _remote_command(
            config,
            ["verify", "--family", "artifact", "--root", "/r", "--task-id", "t1", "--attempt-id", "a1"],
        )[-1]
        self.assertIn("verify --family artifact", verify_remote)

    def test_summary_ok(self) -> None:
        self.assertTrue(_summary_ok({"verification": {"success": True}}))
        self.assertFalse(_summary_ok({"verification": {"success": False}}))
        self.assertTrue(
            _summary_ok(
                {
                    "attempts": {
                        "direct": {"verification": {"success": True}},
                        "deliberate": {"verification": {"success": True}},
                    }
                }
            )
        )
        self.assertFalse(
            _summary_ok(
                {
                    "attempts": {
                        "direct": {"verification": {"success": True}},
                        "deliberate": {"verification": {"success": False}},
                    }
                }
            )
        )

    @mock.patch("pyreplab_harness.orchestrator._run_pi")
    @mock.patch("pyreplab_harness.orchestrator.remote_json")
    def test_run_single_uses_generic_commands_and_reports_pi_failure(
        self, remote_json: mock.Mock, run_pi: mock.Mock
    ) -> None:
        run_pi.return_value = mock.Mock(returncode=3, stdout="", stderr="pi exploded")
        commands: list[list[str]] = []

        def fake_remote(config, arguments, input_text=None, timeout=120):
            commands.append(arguments)
            head = arguments[0]
            if head == "generate":
                return {"id": "t9", "prompt": "p"}
            if head == "prepare-attempt":
                return {"workspace_ref": "/r/ws"}
            if head == "verify":
                return {
                    "success": False,
                    "verifier_id": "v",
                    "verifier_version": "1",
                    "failure_code": "missing_output",
                    "diagnostics": {},
                }
            raise AssertionError(f"unexpected remote command: {arguments}")

        remote_json.side_effect = fake_remote
        args = build_parser().parse_args(
            [
                "--family",
                "artifact",
                "--seed",
                "9",
                "--difficulty",
                "easy",
                "--host",
                "ubuntu-local",
                "--remote-project",
                "/p",
                "--remote-run-root",
                "/r",
            ]
        )
        result = run_smoke(args)

        generate = [c for c in commands if c[0] == "generate"]
        verify = [c for c in commands if c[0] == "verify"]
        self.assertEqual(len(generate), 1)
        self.assertEqual(generate[0][2], "artifact")
        self.assertEqual(len(verify), 1)
        # Pi failure and verification failure are both still reported.
        self.assertFalse(result["verification"]["success"])
        self.assertEqual(result["pi_return_code"], 3)
        self.assertEqual(result["pi_stderr"], "pi exploded")
        self.assertIn("task_id", result)
        self.assertIn("attempt_id", result)
        self.assertIn("policy", result)

    @mock.patch("pyreplab_harness.orchestrator._run_pi")
    @mock.patch("pyreplab_harness.orchestrator.remote_json")
    def test_run_pair_generates_once_and_runs_both_policies(
        self, remote_json: mock.Mock, run_pi: mock.Mock
    ) -> None:
        run_pi.return_value = mock.Mock(returncode=0, stdout="", stderr="")
        commands: list[list[str]] = []

        def fake_remote(config, arguments, input_text=None, timeout=120):
            commands.append(arguments)
            head = arguments[0]
            if head == "generate":
                return {"id": "pair-task", "prompt": "do it"}
            if head == "prepare-attempt":
                return {"workspace_ref": "/r/workspace"}
            if head == "verify":
                return {
                    "success": True,
                    "verifier_id": "v",
                    "verifier_version": "1",
                    "failure_code": None,
                    "diagnostics": {},
                }
            if head == "normalize-events":
                return {"usage": {"input": 100, "output": 50, "total_tokens": 150}}
            if head == "record-events":
                return {}
            raise AssertionError(f"unexpected remote command: {arguments}")

        remote_json.side_effect = fake_remote
        args = build_parser().parse_args(
            [
                "--pair",
                "--family",
                "shell",
                "--seed",
                "7",
                "--difficulty",
                "medium",
                "--host",
                "ubuntu-local",
                "--remote-project",
                "/p",
                "--remote-run-root",
                "/r",
            ]
        )
        result = run_smoke(args)

        generate = [c for c in commands if c[0] == "generate"]
        prepare = [c for c in commands if c[0] == "prepare-attempt"]
        verify = [c for c in commands if c[0] == "verify"]
        self.assertEqual(len(generate), 1)  # Generation is never duplicated.
        self.assertEqual(generate[0][2], "shell")
        self.assertEqual(len(prepare), 2)  # Fresh attempt per policy.
        self.assertEqual(len(verify), 2)

        self.assertEqual(result["mode"], "pair")
        self.assertEqual(result["task_id"], "pair-task")
        self.assertEqual(result["execution_order"], _pair_order(7, ["direct", "deliberate"]))
        self.assertEqual(sorted(result["attempts"]), ["deliberate", "direct"])
        self.assertNotEqual(
            result["attempts"]["direct"]["attempt_id"],
            result["attempts"]["deliberate"]["attempt_id"],
        )
        for item in result["attempts"].values():
            self.assertTrue(item["verification"]["success"])
            self.assertEqual(item["usage"]["input"], 100)


if __name__ == "__main__":
    unittest.main()
