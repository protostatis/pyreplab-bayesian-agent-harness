from __future__ import annotations

import argparse
import io
import json
import unittest
from unittest import mock

from pyreplab_harness.sandbox import SandboxResult
from pyreplab_harness import worker
from pyreplab_harness.unbrowser_rpc import UnbrowserProtocolError
from pyreplab_harness.worker import WorkerConfig, handle_request, serve


class FakeSandbox:
    def execute(self, command: str, timeout: int) -> SandboxResult:
        return SandboxResult(
            stdout=f"{command}:{timeout}",
            stderr="",
            exit_code=0,
            timed_out=False,
            truncated=False,
        )


class FakeUnbrowser:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def execute(self, params: dict) -> dict:
        self.calls.append(params)
        return {"action": params["action"], "result": "Example Domain"}

    def close(self) -> None:
        pass


class FailingUnbrowser(FakeUnbrowser):
    def execute(self, params: dict) -> dict:
        raise UnbrowserProtocolError(
            "unbrowser process connection broken (exit_code=124)",
            infrastructure_error=True,
        )


class WorkerProtocolTest(unittest.TestCase):
    def test_ping(self) -> None:
        response, stop = handle_request(FakeSandbox(), {"id": 1, "method": "ping"}, WorkerConfig())
        self.assertTrue(response["ok"])
        self.assertEqual(response["result"]["network"], "isolated")
        self.assertFalse(stop)

    def test_exec_clamps_timeout(self) -> None:
        response, stop = handle_request(
            FakeSandbox(),
            {"id": 2, "method": "exec", "params": {"command": "pwd", "timeout": 500}},
            WorkerConfig(max_timeout=20),
        )
        self.assertEqual(response["result"]["stdout"], "pwd:20")
        self.assertFalse(stop)

    def test_shutdown(self) -> None:
        response, stop = handle_request(FakeSandbox(), {"id": 3, "method": "shutdown"}, WorkerConfig())
        self.assertTrue(response["ok"])
        self.assertTrue(stop)

    def test_unbrowser_fails_closed_unless_enabled(self) -> None:
        request = {"id": 4, "method": "unbrowser", "params": {"action": "navigate"}}
        with self.assertRaisesRegex(ValueError, "not enabled"):
            handle_request(FakeSandbox(), request, WorkerConfig())

        unbrowser = FakeUnbrowser()
        response, stop = handle_request(
            FakeSandbox(), request, WorkerConfig(), unbrowser  # type: ignore[arg-type]
        )
        self.assertEqual(response["result"]["action"], "navigate")
        self.assertEqual(unbrowser.calls, [{"action": "navigate"}])
        self.assertFalse(stop)

    def test_fixture_server_bind_failure_is_not_ignored(self) -> None:
        worker._fixture_server_instance = None
        with mock.patch.object(worker, "FixtureServer", side_effect=OSError("busy")):
            with self.assertRaisesRegex(OSError, "busy"):
                worker.ensure_fixture_server()
        self.assertIsNone(worker._fixture_server_instance)

    def test_worker_serializes_infrastructure_failure_marker(self) -> None:
        output = io.StringIO()
        serve(
            FakeSandbox(),  # type: ignore[arg-type]
            io.StringIO(json.dumps({
                "id": 7,
                "method": "unbrowser",
                "params": {"action": "navigate"},
            }) + "\n"),
            output,
            30,
            FailingUnbrowser(),  # type: ignore[arg-type]
        )
        response = json.loads(output.getvalue())
        self.assertFalse(response["ok"])
        self.assertTrue(response["error"]["infrastructure_error"])


class WorkerRequiredObservationTest(unittest.TestCase):
    """Tests for the --unbrowser-required-first-observation CLI plumbing."""

    def test_arg_registered_with_choices(self) -> None:
        parser = argparse.ArgumentParser()
        worker.add_worker_arguments(parser)
        args = parser.parse_args([
            "--root", "/root",
            "--workspace", "/workspace",
            "--unbrowser-required-first-observation", "text",
        ])
        self.assertEqual(args.unbrowser_required_first_observation, "text")

    def test_arg_rejects_invalid_choice(self) -> None:
        parser = argparse.ArgumentParser()
        worker.add_worker_arguments(parser)
        with self.assertRaises(SystemExit):
            parser.parse_args([
                "--root", "/root",
                "--workspace", "/workspace",
                "--unbrowser-required-first-observation", "invalid",
            ])

    def test_arg_accepts_blockmap(self) -> None:
        parser = argparse.ArgumentParser()
        worker.add_worker_arguments(parser)
        args = parser.parse_args([
            "--root", "/root",
            "--workspace", "/workspace",
            "--unbrowser-required-first-observation", "blockmap",
        ])
        self.assertEqual(args.unbrowser_required_first_observation, "blockmap")

    def test_arg_defaults_to_none(self) -> None:
        parser = argparse.ArgumentParser()
        worker.add_worker_arguments(parser)
        args = parser.parse_args([
            "--root", "/root",
            "--workspace", "/workspace",
        ])
        self.assertIsNone(args.unbrowser_required_first_observation)


if __name__ == "__main__":
    unittest.main()
