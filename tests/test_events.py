from __future__ import annotations

import json
import unittest

from pyreplab_harness.events import normalize_pi_events


class EventNormalizerTest(unittest.TestCase):
    def test_normalizes_usage_tools_and_final_text_without_double_counting(self) -> None:
        events = [
            {"type": "session", "version": 3, "id": "s1", "cwd": "/tmp"},
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "provider": "ubuntu-gemma",
                    "model": "gemma-4-26b-a4b",
                    "content": [{"type": "toolCall", "name": "bash"}],
                    "usage": {"input": 10, "output": 4, "totalTokens": 14},
                    "stopReason": "toolUse",
                },
            },
            {
                "type": "tool_execution_end",
                "toolCallId": "t1",
                "toolName": "bash",
                "result": {"content": [{"type": "text", "text": "ok"}]},
            },
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "provider": "ubuntu-gemma",
                    "model": "gemma-4-26b-a4b",
                    "content": [{"type": "text", "text": "DONE"}],
                    "usage": {
                        "input": 3,
                        "output": 1,
                        "cacheRead": 8,
                        "totalTokens": 12,
                    },
                    "stopReason": "length",
                },
            },
        ]
        raw = "\n".join(json.dumps(event) for event in events)
        normalized = normalize_pi_events(raw)
        self.assertEqual(normalized["provider"], "ubuntu-gemma")
        self.assertEqual(normalized["model"], "gemma-4-26b-a4b")
        self.assertEqual(normalized["usage"]["input"], 13)
        self.assertEqual(normalized["usage"]["output"], 5)
        self.assertEqual(normalized["usage"]["total_tokens"], 26)
        self.assertEqual(normalized["final_text"], "DONE")
        self.assertEqual(len(normalized["tool_executions"]), 1)
        self.assertEqual(normalized["provider_turn_count"], 2)
        self.assertEqual(normalized["tool_call_count"], 1)
        self.assertEqual(normalized["length_stop_count"], 1)
        self.assertEqual(normalized["stop_reasons"], {"length": 1, "toolUse": 1})
        self.assertEqual(normalized["tool_limit_rejection_count"], 0)
        self.assertEqual(
            normalized["planning_preamble"],
            {
                "present": False,
                "line_count": 0,
                "character_count": 0,
                "plan_marker": False,
                "step_marker_count": 0,
            },
        )

    def test_counts_tool_limit_rejections(self) -> None:
        raw = json.dumps(
            {
                "type": "tool_execution_end",
                "toolCallId": "t1",
                "toolName": "bash",
                "result": {
                    "content": [
                        {"type": "text", "text": "Tool call limit reached (4)."}
                    ]
                },
            }
        )
        normalized = normalize_pi_events(raw)
        self.assertEqual(normalized["tool_limit_rejection_count"], 1)
        self.assertTrue(normalized["tool_executions"][0]["budget_rejected"])

    def test_summarizes_pre_tool_planning_without_retaining_text(self) -> None:
        raw = json.dumps(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "STEP 1: inspect\nSTEP 2: submit"},
                        {"type": "toolCall", "name": "unbrowser"},
                        {"type": "text", "text": "not pre-tool"},
                    ],
                },
            }
        )
        normalized = normalize_pi_events(raw)
        self.assertEqual(normalized["planning_preamble"]["step_marker_count"], 2)
        self.assertEqual(normalized["planning_preamble"]["line_count"], 2)
        self.assertNotIn("inspect", json.dumps(normalized["planning_preamble"]))

    def test_rejects_non_json_lines(self) -> None:
        with self.assertRaisesRegex(ValueError, "line 2"):
            normalize_pi_events('{"type":"session"}\nnot-json\n')


if __name__ == "__main__":
    unittest.main()
