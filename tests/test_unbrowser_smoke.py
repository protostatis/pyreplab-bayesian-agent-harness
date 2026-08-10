from __future__ import annotations

import unittest
from pathlib import Path

from pyreplab_harness.orchestrator import UNBROWSER_TOOL_INTERFACE
from pyreplab_harness.treatments import TreatmentRegistry
from pyreplab_harness.unbrowser_smoke import (
    NEGATIVE_POLICY_ID,
    POSITIVE_POLICY_ID,
    _evaluate_smoke,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _attempt(policy_id: str, selector: str, success: bool) -> dict:
    return {
        "policy": {"id": policy_id},
        "pi_return_code": 0,
        "verification": {
            "success": success,
            "failure_code": None if success else "semantic_mismatch",
        },
        "trajectory": {
            "tool_trace": [
                {
                    "tool_name": "unbrowser",
                    "is_error": False,
                    "details": {
                        "action": "navigate",
                        "selector": None,
                        "runtime_version": "0.0.18",
                    },
                },
                {
                    "tool_name": "unbrowser",
                    "is_error": False,
                    "details": {
                        "action": "text",
                        "selector": selector,
                        "runtime_version": "0.0.18",
                    },
                },
                {"tool_name": "bash", "is_error": False, "details": {"exit_code": 0}},
            ]
        },
    }


class LiveUnbrowserSmokeContractTest(unittest.TestCase):
    def test_frozen_registry_loads_with_expected_controls(self) -> None:
        registry = TreatmentRegistry.load(
            PROJECT_ROOT / "policies" / "unbrowser-smoke-treatments.json"
        )
        self.assertEqual(len(registry), 2)
        self.assertEqual(
            {treatment.id for treatment in registry},
            {NEGATIVE_POLICY_ID, POSITIVE_POLICY_ID},
        )
        for treatment in registry:
            self.assertEqual(
                treatment.tool_interface, UNBROWSER_TOOL_INTERFACE
            )
            self.assertEqual(
                treatment.allowed_tools, ("bash", "unbrowser")
            )
            self.assertEqual(treatment.tool_call_limit, 3)

    def test_expected_failure_and_success_are_a_passing_smoke(self) -> None:
        result = {
            "attempts": {
                "negative": _attempt(NEGATIVE_POLICY_ID, "p", False),
                "positive": _attempt(POSITIVE_POLICY_ID, "h1", True),
            }
        }
        assessment = _evaluate_smoke(result)
        self.assertTrue(assessment["success"])
        self.assertEqual(assessment["problems"], [])

    def test_wrong_polarity_or_selector_fails_assessment(self) -> None:
        result = {
            "attempts": {
                "negative": _attempt(NEGATIVE_POLICY_ID, "h1", True),
                "positive": _attempt(POSITIVE_POLICY_ID, "h1", True),
            }
        }
        assessment = _evaluate_smoke(result)
        self.assertFalse(assessment["success"])
        self.assertGreaterEqual(len(assessment["problems"]), 2)


if __name__ == "__main__":
    unittest.main()
