from __future__ import annotations

import unittest
from pathlib import Path

from pyreplab_harness.orchestrator import UNBROWSER_INTERACTIVE_TOOL_INTERFACE
from pyreplab_harness.treatments import TreatmentRegistry
from pyreplab_harness.unbrowser_interactive_smoke import (
    NEGATIVE_POLICY_ID,
    POSITIVE_POLICY_ID,
    _evaluate_interactive_smoke,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _attempt(policy_id: str, actions: list[str], success: bool) -> dict:
    tool_trace = []
    for action in actions:
        tool_trace.append(
            {
                "tool_name": "unbrowser",
                "is_error": False,
                "details": {
                    "action": action,
                    "runtime_version": "0.0.18",
                },
            }
        )
    tool_trace.append(
        {"tool_name": "bash", "is_error": False, "details": {"exit_code": 0}}
    )
    return {
        "policy": {"id": policy_id},
        "pi_return_code": 0,
        "verification": {
            "success": success,
            "failure_code": None if success else "semantic_mismatch",
        },
        "trajectory": {
            "tool_trace": tool_trace,
        },
    }


class LiveUnbrowserInteractiveSmokeContractTest(unittest.TestCase):
    def test_frozen_registry_loads_with_expected_controls(self) -> None:
        registry = TreatmentRegistry.load(
            PROJECT_ROOT / "policies" / "unbrowser-interactive-treatments.json"
        )
        self.assertEqual(len(registry), 2)
        self.assertEqual(
            {treatment.id for treatment in registry},
            {NEGATIVE_POLICY_ID, POSITIVE_POLICY_ID},
        )
        for treatment in registry:
            self.assertEqual(
                treatment.tool_interface, UNBROWSER_INTERACTIVE_TOOL_INTERFACE
            )
            self.assertEqual(
                treatment.allowed_tools, ("bash", "unbrowser")
            )
            self.assertEqual(treatment.tool_call_limit, 12)

    def test_expected_failure_and_success_are_a_passing_smoke(self) -> None:
        positive_actions = ["navigate", "type", "submit", "text", "click", "text"]
        negative_actions = ["navigate", "type", "submit", "text"]
        result = {
            "attempts": {
                "negative": _attempt(
                    NEGATIVE_POLICY_ID, negative_actions, False
                ),
                "positive": _attempt(
                    POSITIVE_POLICY_ID, positive_actions, True
                ),
            }
        }
        assessment = _evaluate_interactive_smoke(result)
        self.assertTrue(assessment["success"])
        self.assertEqual(assessment["problems"], [])

    def test_positive_must_include_interactive_actions(self) -> None:
        # Missing click action
        positive_no_click = ["navigate", "type", "submit", "text"]
        result = {
            "attempts": {
                "negative": _attempt(
                    NEGATIVE_POLICY_ID, ["navigate", "type", "submit", "text"], False
                ),
                "positive": _attempt(
                    POSITIVE_POLICY_ID, positive_no_click, True
                ),
            }
        }
        assessment = _evaluate_interactive_smoke(result)
        self.assertFalse(assessment["success"])
        self.assertTrue(
            any("click" in p for p in assessment["problems"]),
            f"Expected a problem about missing 'click'; got {assessment['problems']}",
        )

    def test_wrong_polarity_fails_assessment(self) -> None:
        result = {
            "attempts": {
                "negative": _attempt(
                    NEGATIVE_POLICY_ID,
                    ["navigate", "type", "submit", "text"],
                    True,  # Should be False
                ),
                "positive": _attempt(
                    POSITIVE_POLICY_ID,
                    ["navigate", "type", "submit", "text", "click", "text"],
                    True,
                ),
            }
        }
        assessment = _evaluate_interactive_smoke(result)
        self.assertFalse(assessment["success"])
        self.assertGreaterEqual(len(assessment["problems"]), 1)


if __name__ == "__main__":
    unittest.main()
