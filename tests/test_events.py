from __future__ import annotations

import json
import re
import unittest

from pyreplab_harness.events import (
    BUDGET_RECEIPT_SCHEMA_VERSION,
    NORMALIZED_EVENT_SCHEMA_VERSION,
    PROVIDER_TURN_SEMANTICS,
    normalize_pi_events,
)


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
        self.assertEqual(
            normalized["schema_version"], NORMALIZED_EVENT_SCHEMA_VERSION
        )
        self.assertEqual(
            normalized["provider_turn_semantics"], PROVIDER_TURN_SEMANTICS
        )
        self.assertEqual(normalized["provider"], "ubuntu-gemma")
        self.assertEqual(normalized["model"], "gemma-4-26b-a4b")
        self.assertEqual(normalized["usage"]["input"], 13)
        self.assertEqual(normalized["usage"]["output"], 5)
        self.assertEqual(normalized["usage"]["total_tokens"], 26)
        self.assertEqual(normalized["final_text"], "DONE")
        self.assertEqual(len(normalized["tool_executions"]), 1)
        self.assertEqual(normalized["provider_turn_count"], 2)
        self.assertEqual(len(normalized["provider_turns"]), 2)
        first_turn = normalized["provider_turns"][0]
        self.assertEqual(first_turn["turn_index"], 1)
        self.assertEqual(first_turn["source_event_index"], 1)
        self.assertFalse(first_turn["usage"]["complete"])
        self.assertIsNone(first_turn["usage"]["cache_read"])
        self.assertTrue(re.fullmatch(r"[0-9a-f]{64}", first_turn["assistant_content_sha256"]))
        self.assertEqual(normalized["synthetic_assistant_message_count"], 0)
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

    def test_marks_operation_aborted_without_calling_it_a_budget_rejection(self) -> None:
        raw = json.dumps(
            {
                "type": "tool_execution_end",
                "toolCallId": "t1",
                "toolName": "bash",
                "isError": True,
                "result": {
                    "content": [{"type": "text", "text": "Operation aborted"}],
                    "details": {},
                },
            }
        )
        normalized = normalize_pi_events(raw)
        execution = normalized["tool_executions"][0]
        self.assertTrue(execution["operation_aborted"])
        self.assertFalse(execution["budget_rejected"])

    def test_excludes_exact_terminal_local_abort_from_provider_turns(self) -> None:
        events = [
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "provider": "ubuntu-gemma",
                    "model": "gemma-4-26b-a4b",
                    "content": [
                        {
                            "type": "toolCall",
                            "id": "over-cap",
                            "name": "unbrowser",
                        }
                    ],
                    "usage": {
                        "input": 10,
                        "output": 4,
                        "cacheRead": 0,
                        "cacheWrite": 0,
                        "totalTokens": 14,
                    },
                    "stopReason": "toolUse",
                },
            },
            {
                "type": "tool_execution_end",
                "toolCallId": "over-cap",
                "toolName": "unbrowser",
                "isError": True,
                "result": {
                    "content": [{"type": "text", "text": "Operation aborted"}],
                    "details": {},
                },
            },
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "provider": "ubuntu-gemma",
                    "model": "gemma-4-26b-a4b",
                    "content": [],
                    "usage": {
                        "input": 0,
                        "output": 0,
                        "cacheRead": 0,
                        "cacheWrite": 0,
                        "totalTokens": 0,
                    },
                    "stopReason": "error",
                    "errorMessage": "This operation was aborted",
                },
            },
            {"type": "agent_end"},
        ]
        normalized = normalize_pi_events(
            "\n".join(json.dumps(event) for event in events)
        )
        self.assertEqual(normalized["assistant_message_count"], 2)
        self.assertEqual(normalized["provider_turn_count"], 1)
        self.assertEqual(len(normalized["provider_turns"]), 1)
        self.assertEqual(normalized["synthetic_assistant_message_count"], 1)
        self.assertEqual(normalized["stop_reasons"], {"error": 1, "toolUse": 1})

    def test_does_not_exclude_abort_lookalikes(self) -> None:
        base = {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "provider": "ubuntu-gemma",
                "model": "gemma-4-26b-a4b",
                "content": [],
                "usage": {
                    "input": 0,
                    "output": 0,
                    "cacheRead": 0,
                    "cacheWrite": 0,
                    "totalTokens": 0,
                },
                "stopReason": "error",
                "errorMessage": "This operation was aborted",
            },
        }
        aborted_tool = {
            "type": "tool_execution_end",
            "toolName": "unbrowser",
            "isError": True,
            "result": {
                "content": [{"type": "text", "text": "Operation aborted"}],
                "details": {},
            },
        }
        later_assistant = {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "later"}],
                "usage": {"input": 1, "output": 1, "totalTokens": 2},
                "stopReason": "stop",
            },
        }
        cases = {
            "no_preceding_abort": [base, {"type": "agent_end"}],
            "not_terminal": [aborted_tool, base, later_assistant],
            "different_error": [
                aborted_tool,
                {
                    **base,
                    "message": {
                        **base["message"],
                        "errorMessage": "Provider request failed",
                    },
                },
            ],
        }
        for label, events in cases.items():
            with self.subTest(label=label):
                normalized = normalize_pi_events(
                    "\n".join(json.dumps(event) for event in events)
                )
                self.assertEqual(normalized["synthetic_assistant_message_count"], 0)
                self.assertGreaterEqual(normalized["provider_turn_count"], 1)

    def test_provider_gate_receipt_excludes_local_pre_http_abort(self) -> None:
        provider_message = {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "provider": "ubuntu-gemma",
                "model": "gemma-4-26b-a4b",
                "content": [{"type": "text", "text": "working"}],
                "usage": {
                    "input": 1,
                    "output": 1,
                    "cacheRead": 0,
                    "cacheWrite": 0,
                    "totalTokens": 2,
                },
                "stopReason": "toolUse",
            },
        }
        local_abort = {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "provider": "ubuntu-gemma",
                "model": "gemma-4-26b-a4b",
                "content": [],
                "usage": {
                    "input": 0,
                    "output": 0,
                    "cacheRead": 0,
                    "cacheWrite": 0,
                    "totalTokens": 0,
                },
                "stopReason": "aborted",
                "errorMessage": "Request aborted",
            },
        }
        raw = "\n".join(json.dumps(event) for event in [provider_message, local_abort])
        without_receipt = normalize_pi_events(raw)
        with_receipt = normalize_pi_events(
            raw,
            {
                "schema_version": BUDGET_RECEIPT_SCHEMA_VERSION,
                "provider_request_blocks": 1,
            },
        )
        self.assertEqual(without_receipt["provider_turn_count"], 2)
        self.assertEqual(with_receipt["provider_turn_count"], 1)
        self.assertEqual(with_receipt["synthetic_assistant_message_count"], 1)

    def test_counts_thirteen_provider_turns_before_terminal_budget_abort(self) -> None:
        events = []
        for index in range(13):
            events.append(
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "provider": "ubuntu-gemma",
                        "model": "gemma-4-26b-a4b",
                        "content": [
                            {
                                "type": "toolCall",
                                "id": f"call-{index}",
                                "name": "unbrowser",
                            }
                        ],
                        "usage": {
                            "input": 1,
                            "output": 1,
                            "cacheRead": 0,
                            "cacheWrite": 0,
                            "totalTokens": 2,
                        },
                        "stopReason": "toolUse",
                    },
                }
            )
            result = (
                {
                    "content": [{"type": "text", "text": "Operation aborted"}],
                    "details": {},
                }
                if index == 12
                else {"content": [{"type": "text", "text": "ok"}], "details": {}}
            )
            events.append(
                {
                    "type": "tool_execution_end",
                    "toolCallId": f"call-{index}",
                    "toolName": "unbrowser",
                    "isError": index == 12,
                    "result": result,
                }
            )
        events.extend(
            [
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "provider": "ubuntu-gemma",
                        "model": "gemma-4-26b-a4b",
                        "content": [],
                        "usage": {
                            "input": 0,
                            "output": 0,
                            "cacheRead": 0,
                            "cacheWrite": 0,
                            "totalTokens": 0,
                        },
                        "stopReason": "error",
                        "errorMessage": "This operation was aborted",
                    },
                },
                {"type": "agent_end"},
            ]
        )
        normalized = normalize_pi_events(
            "\n".join(json.dumps(event) for event in events)
        )
        self.assertEqual(normalized["assistant_message_count"], 14)
        self.assertEqual(normalized["provider_turn_count"], 13)
        self.assertEqual(normalized["synthetic_assistant_message_count"], 1)
        self.assertEqual(normalized["tool_call_count"], 13)

    def test_does_not_mark_incidental_operation_aborted_text(self) -> None:
        raw = json.dumps(
            {
                "type": "tool_execution_end",
                "toolCallId": "t1",
                "toolName": "bash",
                "isError": True,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": "Remote operation aborted unexpectedly",
                        }
                    ],
                    "details": {},
                },
            }
        )
        normalized = normalize_pi_events(raw)
        self.assertFalse(
            normalized["tool_executions"][0]["operation_aborted"]
        )

    def test_marks_schema_validation_as_pre_execution_rejected(self) -> None:
        raw = json.dumps({
            "type": "tool_execution_end",
            "toolCallId": "t1",
            "toolName": "unbrowser",
            "isError": True,
            "result": {
                "content": [{
                    "type": "text",
                    "text": "Validation failed for tool \"unbrowser\": bad action",
                }],
                "details": {},
            },
        })
        execution = normalize_pi_events(raw)["tool_executions"][0]
        self.assertTrue(execution["pre_execution_rejected"])

    def test_marks_truncated_arguments_as_pre_execution_rejected(self) -> None:
        raw = json.dumps({
            "type": "tool_execution_end",
            "toolCallId": "t1",
            "toolName": "bash",
            "isError": True,
            "result": {
                "content": [{
                    "type": "text",
                    "text": "Tool call \"bash\" was not executed: arguments were truncated.",
                }],
                "details": {},
            },
        })
        execution = normalize_pi_events(raw)["tool_executions"][0]
        self.assertTrue(execution["pre_execution_rejected"])

    def test_marks_unknown_tool_as_pre_execution_rejected(self) -> None:
        raw = json.dumps({
            "type": "tool_execution_end",
            "toolCallId": "t1",
            "toolName": "unknown",
            "isError": True,
            "result": {
                "content": [{"type": "text", "text": "Tool unknown not found"}],
                "details": {},
            },
        })
        execution = normalize_pi_events(raw)["tool_executions"][0]
        self.assertTrue(execution["pre_execution_rejected"])

    def test_does_not_mark_tool_authored_validation_text(self) -> None:
        raw = json.dumps({
            "type": "tool_execution_end",
            "toolCallId": "t1",
            "toolName": "bash",
            "isError": True,
            "result": {
                "content": [{
                    "type": "text",
                    "text": "App validation failed for tool input",
                }],
                "details": {},
            },
        })
        execution = normalize_pi_events(raw)["tool_executions"][0]
        self.assertFalse(execution["pre_execution_rejected"])

    def test_does_not_mark_pi_text_with_tool_details(self) -> None:
        raw = json.dumps({
            "type": "tool_execution_end",
            "toolCallId": "t1",
            "toolName": "bash",
            "isError": True,
            "result": {
                "content": [{
                    "type": "text",
                    "text": "Validation failed for tool \"bash\": app output",
                }],
                "details": {"exit_code": 1},
            },
        })
        execution = normalize_pi_events(raw)["tool_executions"][0]
        self.assertFalse(execution["pre_execution_rejected"])

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

    def test_provider_turn_usage_rejects_invalid_values(self) -> None:
        for value in (True, -1, 1.5, "1"):
            with self.subTest(value=value):
                raw = json.dumps(
                    {
                        "type": "message_end",
                        "message": {
                            "role": "assistant",
                            "content": [],
                            "usage": {"input": value},
                        },
                    }
                )
                with self.assertRaisesRegex(ValueError, "non-negative integer"):
                    normalize_pi_events(raw)

    def test_provider_turn_hash_omits_volatile_ids_and_signatures(self) -> None:
        def normalized(call_id: str, signature: str) -> str:
            raw = json.dumps(
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": "plan",
                                "thinkingSignature": signature,
                            },
                            {
                                "type": "toolCall",
                                "id": call_id,
                                "name": "bash",
                                "arguments": {"command": "pwd"},
                            },
                        ],
                    },
                }
            )
            return normalize_pi_events(raw)["provider_turns"][0][
                "assistant_content_sha256"
            ]

        self.assertEqual(normalized("a", "x"), normalized("b", "y"))

    def test_rejects_non_json_lines(self) -> None:
        with self.assertRaisesRegex(ValueError, "line 2"):
            normalize_pi_events('{"type":"session"}\nnot-json\n')


if __name__ == "__main__":
    unittest.main()
