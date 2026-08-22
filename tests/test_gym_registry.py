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
from pyreplab_harness.routing_fixtures import build_stage_b_design


def _routing_fixture_seed(difficulty: str = "easy") -> int:
    """Return a valid frozen Stage-B seed for the requested difficulty."""
    for coord in build_stage_b_design():
        if coord["difficulty"] == difficulty:
            return coord["seed"]
    raise AssertionError(f"no Stage-B coordinate with difficulty {difficulty!r}")


class RegistryTest(unittest.TestCase):
    def test_canonical_families(self) -> None:
        self.assertEqual(
            FAMILIES,
            (
                "artifact",
                "sqlite",
                "shell",
                "python_repair",
                "unbrowser",
                "unbrowser_interactive",
                "unbrowser_fixture",
                "routing_fixture",
            ),
        )

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
                    seed = 42
                    difficulty = "easy"
                    if family == "routing_fixture":
                        # routing_fixture selects an exact frozen Stage-B
                        # coordinate by seed, so seed 42 is not valid.
                        seed = _routing_fixture_seed("easy")
                    task = generate_task(family, directory, seed, difficulty)
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

    def test_cli_generate_passes_fixture_template_to_unbrowser_fixture(self) -> None:
        """--fixture-template reaches the generated task's public_metadata."""
        with tempfile.TemporaryDirectory() as directory:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = cli_main(
                    [
                        "generate",
                        "--family",
                        "unbrowser_fixture",
                        "--root",
                        directory,
                        "--seed",
                        "42",
                        "--difficulty",
                        "easy",
                        "--fixture-template",
                        "table_filter_sort",
                    ]
                )
            self.assertEqual(code, 0)
            task = json.loads(buffer.getvalue())
            self.assertEqual(task["family"], "unbrowser_fixture")
            self.assertEqual(task["template_id"], "table_filter_sort")
            metadata = task["public_metadata"]
            self.assertEqual(metadata["template"], "table_filter_sort")
            self.assertIn("table_filter_sort", metadata["fixture_url"])

    def test_cli_generate_ignores_fixture_template_for_non_fixture_families(self) -> None:
        """--fixture-template is silently ignored for non-fixture families."""
        with tempfile.TemporaryDirectory() as directory:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = cli_main(
                    [
                        "generate",
                        "--family",
                        "artifact",
                        "--root",
                        directory,
                        "--seed",
                        "7",
                        "--difficulty",
                        "easy",
                        "--fixture-template",
                        "cross_page_comparison",
                    ]
                )
            self.assertEqual(code, 0)
            task = json.loads(buffer.getvalue())
            self.assertEqual(task["family"], "artifact")

    def test_generate_task_kwarg_forwards_to_fixture_only(self) -> None:
        """generate_task only passes fixture_template to unbrowser_fixture."""
        with tempfile.TemporaryDirectory() as directory:
            fixture_task = generate_task(
                "unbrowser_fixture", directory, 1, "easy",
                fixture_template="distractor_recovery",
            )
            self.assertEqual(
                fixture_task.public_metadata["template"], "distractor_recovery"
            )
            # Non-fixture family silently ignores the template kwarg.
            artifact_task = generate_task(
                "artifact", directory, 1, "easy",
                fixture_template="bogus_ignored",
            )
            self.assertEqual(artifact_task.family, "artifact")

    def test_generate_task_forwards_outcome_only_fixture_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_task(
                "unbrowser_fixture",
                directory,
                1,
                "easy",
                fixture_template="single_page_extraction",
                fixture_generator_version="unbrowser-fixture-v3",
            )
            self.assertEqual(task.generator_version, "unbrowser-fixture-v3")
            self.assertEqual(
                task.public_metadata["prompt_profile"], "outcome_only_v1"
            )

    def test_cli_generate_accepts_outcome_only_fixture_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = cli_main(
                    [
                        "generate",
                        "--family",
                        "unbrowser_fixture",
                        "--root",
                        directory,
                        "--seed",
                        "7",
                        "--difficulty",
                        "easy",
                        "--fixture-generator-version",
                        "unbrowser-fixture-v3",
                    ]
                )
            self.assertEqual(code, 0)
            task = json.loads(buffer.getvalue())
            self.assertEqual(task["generator_version"], "unbrowser-fixture-v3")

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = cli_main(
                    [
                        "fixture-task-commitment",
                        "--root",
                        directory,
                        "--task-id",
                        task["id"],
                    ]
                )
            self.assertEqual(code, 0)
            commitment = json.loads(buffer.getvalue())
            self.assertEqual(commitment["task"]["id"], task["id"])
            self.assertEqual(len(commitment["workspace_sha256"]), 64)
            self.assertEqual(len(commitment["oracle_sha256"]), 64)
            self.assertEqual(len(commitment["commitment_hash"]), 64)

    def test_task_role_is_fixture_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_task(
                "unbrowser_fixture",
                directory,
                3,
                "easy",
                task_role="T_pilot",
            )
            self.assertEqual(task.public_metadata["task_role"], "T_pilot")
            with self.assertRaisesRegex(ValueError, "only supported"):
                generate_task(
                    "artifact", directory, 4, "easy", task_role="T_pilot"
                )

    def test_routing_fixture_accepts_task_role(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            seed = _routing_fixture_seed("easy")
            task = generate_task(
                "routing_fixture",
                directory,
                seed,
                "easy",
                task_role="T_canary",
            )
            self.assertEqual(task.family, "routing_fixture")
            self.assertEqual(task.public_metadata["task_role"], "T_canary")

    def test_routing_fixture_rejects_wrong_difficulty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            seed = _routing_fixture_seed("easy")
            with self.assertRaisesRegex(ValueError, "difficulty"):
                generate_task("routing_fixture", directory, seed, "hard")

    def test_cli_generate_routing_fixture_dispatches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            seed = _routing_fixture_seed("easy")
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = cli_main(
                    [
                        "generate",
                        "--family",
                        "routing_fixture",
                        "--root",
                        directory,
                        "--seed",
                        str(seed),
                        "--difficulty",
                        "easy",
                        "--task-role",
                        "T_canary",
                    ]
                )
            self.assertEqual(code, 0)
            task = json.loads(buffer.getvalue())
            self.assertEqual(task["family"], "routing_fixture")
            self.assertEqual(task["public_metadata"]["task_role"], "T_canary")
            self.assertTrue(task["id"].startswith("routing-fixture-"))
            self.assertIn(
                "/routing/", task["public_metadata"]["allowed_url"]
            )


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
            if head == "record-events":
                return {"recorded": 0}
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
        record = [c for c in commands if c[0] == "record-events"]
        self.assertEqual(len(generate), 1)
        self.assertEqual(generate[0][2], "artifact")
        self.assertEqual(len(verify), 1)
        self.assertEqual(len(record), 1)
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
