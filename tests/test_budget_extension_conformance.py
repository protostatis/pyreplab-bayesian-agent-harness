from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from pyreplab_harness.events import normalize_pi_events
from pyreplab_harness.orchestrator import _parse_budget_receipt


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _ScriptedOpenAIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler contract
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length))
        if self.path != "/v1/chat/completions" or payload.get("stream") is not True:
            self.send_error(400)
            return

        server = self.server
        with server.request_lock:  # type: ignore[attr-defined]
            request_index = server.request_count  # type: ignore[attr-defined]
            server.request_count += 1  # type: ignore[attr-defined]

        script_mode = server.script_mode  # type: ignore[attr-defined]
        if script_mode == "accepted-batch" and request_index > 0:
            tool_calls = []
            response_delta = {"role": "assistant", "content": "done"}
            finish_reason = "stop"
        elif script_mode == "batch":
            tool_calls = [
                {
                    "index": index,
                    "id": f"call-{index}",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"command": "true"}),
                    },
                }
                for index in range(14)
            ]
            response_delta = {"role": "assistant", "tool_calls": tool_calls}
            finish_reason = "tool_calls"
        elif script_mode == "accepted-batch":
            tool_calls = [
                {
                    "index": 0,
                    "id": "slow",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"command": "sleep 0.2"}),
                    },
                },
                {
                    "index": 1,
                    "id": "fast",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"command": "true"}),
                    },
                },
            ]
            response_delta = {"role": "assistant", "tool_calls": tool_calls}
            finish_reason = "tool_calls"
        elif script_mode == "duplicate":
            tool_calls = [
                {
                    "index": index,
                    "id": "duplicate",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"command": "true"}),
                    },
                }
                for index in range(2)
            ]
            response_delta = {"role": "assistant", "tool_calls": tool_calls}
            finish_reason = "tool_calls"
        else:
            arguments = {} if request_index == 0 else {"command": "true"}
            tool_calls = [
                {
                    "index": 0,
                    "id": f"call-{request_index}",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps(arguments),
                    },
                }
            ]
            response_delta = {"role": "assistant", "tool_calls": tool_calls}
            finish_reason = "tool_calls"
        created = int(time.time())
        chunks = [
            {
                "id": f"chatcmpl-{request_index}",
                "object": "chat.completion.chunk",
                "created": created,
                "model": "mock-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": response_delta,
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": f"chatcmpl-{request_index}",
                "object": "chat.completion.chunk",
                "created": created,
                "model": "mock-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason,
                    }
                ],
            },
            {
                "id": f"chatcmpl-{request_index}",
                "object": "chat.completion.chunk",
                "created": created,
                "model": "mock-model",
                "choices": [],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        ]
        body = "".join(
            f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n"
            for chunk in chunks
        ) + "data: [DONE]\n\n"
        encoded = body.encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(encoded)))
        self.send_header("connection", "close")
        self.end_headers()
        self.wfile.write(encoded)


@unittest.skipUnless(shutil.which("pi"), "pinned Pi CLI is unavailable")
class BudgetExtensionConformanceTest(unittest.TestCase):
    def _run_script(self, script_mode: str) -> tuple[subprocess.CompletedProcess[str], int]:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _ScriptedOpenAIHandler)
        server.request_count = 0  # type: ignore[attr-defined]
        server.request_lock = threading.Lock()  # type: ignore[attr-defined]
        server.script_mode = script_mode  # type: ignore[attr-defined]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                agent_dir = root / "agent"
                agent_dir.mkdir()
                (agent_dir / "models.json").write_text(
                    json.dumps(
                        {
                            "providers": {
                                "mock": {
                                    "baseUrl": (
                                        f"http://127.0.0.1:{server.server_port}/v1"
                                    ),
                                    "api": "openai-completions",
                                    "apiKey": "test",
                                    "models": [
                                        {
                                            "id": "mock-model",
                                            "name": "Mock model",
                                            "reasoning": False,
                                            "contextWindow": 65536,
                                            "maxTokens": 4096,
                                        }
                                    ],
                                }
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                flag_extension = root / "register-tool-limit.ts"
                flag_extension.write_text(
                    """
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
export default function (pi: ExtensionAPI): void {
  pi.registerFlag("gym-tool-limit", {
    description: "Conformance tool limit",
    type: "string",
    default: "12",
  });
}
""".strip()
                    + "\n",
                    encoding="utf-8",
                )
                command = [
                    str(shutil.which("pi")),
                    "--provider",
                    "mock",
                    "--model",
                    "mock-model",
                    "--api-key",
                    "test",
                    "--thinking",
                    "off",
                    "--mode",
                    "json",
                    "--print",
                    "--no-session",
                    "--tools",
                    "bash",
                    "--no-context-files",
                    "--no-skills",
                    "--no-prompt-templates",
                    "--no-extensions",
                    "--no-approve",
                    "--extension",
                    str(flag_extension),
                    "--extension",
                    str(PROJECT_ROOT / "pi_extensions" / "gym-budget-v3.ts"),
                    "--gym-tool-limit",
                    "12",
                    "--gym-provider-turn-limit",
                    "13",
                    "Exercise the scripted tool sequence.",
                ]
                environment = {
                    **os.environ,
                    "PI_CODING_AGENT_DIR": str(agent_dir),
                }
                completed = subprocess.run(
                    command,
                    cwd=root,
                    env=environment,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=30,
                    check=False,
                )
            return completed, server.request_count  # type: ignore[attr-defined]
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_schema_rejection_cannot_admit_provider_request_fourteen(self) -> None:
        completed, request_count = self._run_script("schema-rejection")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(request_count, 13)
        receipt = _parse_budget_receipt(completed.stderr)
        self.assertIsNotNone(receipt, completed.stderr)
        assert receipt is not None
        normalized = normalize_pi_events(completed.stdout, receipt)
        self.assertEqual(receipt["provider_request_admissions"], 13)
        self.assertEqual(receipt["provider_request_blocks"], 1)
        self.assertEqual(receipt["provider_gate_checks"], 14)
        self.assertEqual(receipt["tool_attempt_count"], 13)
        self.assertEqual(receipt["admitted_tool_call_count"], 12, receipt)
        self.assertEqual(receipt["executed_tool_call_count"], 12, receipt)
        self.assertEqual(receipt["pre_admission_rejected_tool_call_count"], 1)
        self.assertEqual(receipt["suppressed_tool_request_count"], 0)
        self.assertEqual(receipt["invariant_violations"], [])
        self.assertEqual(normalized["provider_turn_count"], 13)
        self.assertEqual(normalized["synthetic_assistant_message_count"], 1)
        self.assertEqual(normalized["tool_call_count"], 13)

    def test_parallel_batch_cannot_emit_tool_attempt_fourteen(self) -> None:
        completed, request_count = self._run_script("batch")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(request_count, 1, completed.stderr)
        receipt = _parse_budget_receipt(completed.stderr)
        self.assertIsNotNone(receipt, completed.stderr)
        assert receipt is not None
        normalized = normalize_pi_events(completed.stdout, receipt)
        self.assertEqual(receipt["provider_request_admissions"], 1)
        self.assertEqual(receipt["provider_request_blocks"], 0)
        self.assertEqual(receipt["tool_attempt_count"], 1, receipt)
        self.assertEqual(receipt["admitted_tool_call_count"], 0)
        self.assertEqual(receipt["executed_tool_call_count"], 0)
        self.assertEqual(receipt["pre_admission_rejected_tool_call_count"], 1)
        self.assertEqual(receipt["suppressed_tool_request_count"], 13)
        self.assertEqual(receipt["invariant_violations"], [])
        self.assertEqual(normalized["provider_turn_count"], 1)
        self.assertEqual(normalized["synthetic_assistant_message_count"], 1)
        self.assertEqual(normalized["tool_call_count"], 1)

    def test_parallel_completion_order_reconciles_by_identity(self) -> None:
        completed, request_count = self._run_script("accepted-batch")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(request_count, 2)
        receipt = _parse_budget_receipt(completed.stderr)
        self.assertIsNotNone(receipt, completed.stderr)
        assert receipt is not None
        normalized = normalize_pi_events(completed.stdout, receipt)
        self.assertEqual(receipt["provider_request_admissions"], 2)
        self.assertEqual(receipt["provider_request_blocks"], 0)
        self.assertEqual(receipt["tool_attempt_count"], 2)
        self.assertEqual(receipt["tool_attempt_ids"], ["slow", "fast"])
        self.assertEqual(receipt["admitted_tool_call_count"], 2)
        self.assertEqual(receipt["executed_tool_call_count"], 2)
        self.assertEqual(receipt["executed_tool_call_ids"], ["fast", "slow"])
        self.assertEqual(receipt["pre_admission_rejected_tool_call_count"], 0)
        self.assertEqual(receipt["suppressed_tool_request_count"], 0)
        self.assertEqual(receipt["invariant_violations"], [])
        self.assertEqual(normalized["provider_turn_count"], 2)
        self.assertEqual(normalized["synthetic_assistant_message_count"], 0)
        self.assertEqual(normalized["tool_call_count"], 2)

    def test_duplicate_ids_abort_without_undercounting_attempt_events(self) -> None:
        completed, request_count = self._run_script("duplicate")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(request_count, 2, completed.stderr)
        receipt = _parse_budget_receipt(completed.stderr)
        self.assertIsNotNone(receipt, completed.stderr)
        assert receipt is not None
        self.assertEqual(receipt["provider_request_admissions"], 2)
        self.assertEqual(receipt["tool_attempt_count"], 2)
        self.assertEqual(receipt["tool_attempt_ids"], ["duplicate"])
        self.assertIn(
            "duplicate_tool_attempt_id:duplicate",
            receipt["invariant_violations"],
        )
        self.assertLessEqual(receipt["tool_attempt_count"], 13)


if __name__ == "__main__":
    unittest.main()
