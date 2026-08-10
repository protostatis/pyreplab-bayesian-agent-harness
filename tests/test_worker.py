from __future__ import annotations

import unittest

from pyreplab_harness.sandbox import SandboxResult
from pyreplab_harness.worker import WorkerConfig, handle_request


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


if __name__ == "__main__":
    unittest.main()
