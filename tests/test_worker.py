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


if __name__ == "__main__":
    unittest.main()
