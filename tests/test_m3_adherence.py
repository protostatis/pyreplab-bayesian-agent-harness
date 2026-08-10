from __future__ import annotations

import unittest

from pyreplab_harness.m3_adherence import assess_policy_adherence
from pyreplab_harness.meta_grammar import enumerate_unbrowser_grammar


def _treatment(
    planning: str,
    observation: str,
    verification: str,
    recovery: str,
    tool_cap: str = "lean",
):
    for treatment in enumerate_unbrowser_grammar(version="2"):
        metadata = treatment.generator_metadata
        if all(
            metadata[key] == value
            for key, value in {
                "planning": planning,
                "observation": observation,
                "verification": verification,
                "recovery": recovery,
                "tool_cap": tool_cap,
            }.items()
        ):
            return treatment
    raise AssertionError("treatment not found")


def _entry(tool: str, **details):
    return {
        "tool_name": tool,
        "is_error": False,
        "budget_rejected": False,
        "details": details,
    }


class M3AdherenceTest(unittest.TestCase):
    def test_direct_text_submit_failfast_adheres(self) -> None:
        treatment = _treatment(
            "direct", "text_first", "submit_directly", "fail_fast"
        )
        trajectory = {
            "planning_preamble": {"present": False},
            "tool_trace": [
                _entry("unbrowser", action="navigate"),
                _entry("unbrowser", action="text", selector="body"),
                _entry("bash", exit_code=0),
            ],
        }
        result = assess_policy_adherence(treatment, trajectory)
        self.assertTrue(result["planning_adherent"])
        self.assertTrue(result["observation_adherent"])
        self.assertTrue(result["verification_adherent"])
        self.assertIsNone(result["recovery_adherent"])
        self.assertTrue(result["tool_cap_compliant"])

    def test_marked_plan_reobserve_and_retry_are_detected(self) -> None:
        treatment = _treatment(
            "brief_plan",
            "structure_first",
            "final_reobserve",
            "diagnose_retry_once",
            "expanded",
        )
        failed = _entry("unbrowser", action="blockmap", error="temporary")
        trajectory = {
            "planning_preamble": {
                "present": True,
                "line_count": 1,
                "plan_marker": True,
                "step_marker_count": 0,
            },
            "tool_trace": [
                _entry("unbrowser", action="navigate"),
                failed,
                _entry("unbrowser", action="blockmap"),
                _entry("unbrowser", action="text", selector="#result"),
                _entry("unbrowser", action="text", selector="#result"),
                _entry("bash", exit_code=0),
            ],
        }
        result = assess_policy_adherence(treatment, trajectory)
        self.assertTrue(result["planning_adherent"])
        self.assertTrue(result["observation_adherent"])
        self.assertTrue(result["repeated_final_read"])
        self.assertTrue(result["recovery_eligible"])
        self.assertTrue(result["successful_same_tool_retry"])
        self.assertTrue(result["recovery_adherent"])

    def test_budget_rejection_does_not_count_as_admitted(self) -> None:
        treatment = _treatment(
            "decompose", "targeted_query_first", "submit_directly", "fail_fast"
        )
        trace = [
            _entry("unbrowser", action="navigate"),
            _entry("unbrowser", action="query", selector="table"),
            _entry("bash", exit_code=0),
            _entry("bash", exit_code=0),
            _entry("bash", exit_code=0),
            _entry("bash", exit_code=0),
            {
                "tool_name": "bash",
                "is_error": False,
                "budget_rejected": True,
                "details": {"exit_code": -1},
            },
        ]
        result = assess_policy_adherence(
            treatment,
            {
                "planning_preamble": {"present": True, "step_marker_count": 2},
                "tool_trace": trace,
            },
        )
        self.assertEqual(result["admitted_tool_call_count"], 6)
        self.assertEqual(result["budget_rejection_count"], 1)
        self.assertTrue(result["tool_cap_compliant"])

    def test_bash_nonzero_is_not_automatically_recovery_eligible(self) -> None:
        treatment = _treatment(
            "direct", "text_first", "submit_directly", "fail_fast"
        )
        result = assess_policy_adherence(
            treatment,
            {
                "planning_preamble": {"present": False},
                "tool_trace": [
                    _entry("unbrowser", action="navigate"),
                    _entry("unbrowser", action="text", selector="body"),
                    _entry("bash", exit_code=1),
                ],
            },
        )
        self.assertFalse(result["recovery_eligible"])

    def test_observation_must_precede_second_navigation(self) -> None:
        treatment = _treatment(
            "direct", "text_first", "submit_directly", "fail_fast"
        )
        result = assess_policy_adherence(
            treatment,
            {
                "planning_preamble": {"present": False},
                "tool_trace": [
                    _entry("unbrowser", action="navigate"),
                    _entry("unbrowser", action="navigate"),
                    _entry("unbrowser", action="text", selector="body"),
                ],
            },
        )
        self.assertIsNone(result["first_observation"])
        self.assertFalse(result["observation_adherent"])

    def test_non_200_navigation_is_recovery_eligible(self) -> None:
        treatment = _treatment(
            "brief_plan",
            "text_first",
            "submit_directly",
            "diagnose_retry_once",
        )
        result = assess_policy_adherence(
            treatment,
            {
                "planning_preamble": {
                    "present": True,
                    "line_count": 1,
                    "plan_marker": True,
                },
                "tool_trace": [
                    _entry("unbrowser", action="navigate"),
                    _entry("unbrowser", action="text", selector="body"),
                    _entry("unbrowser", action="click", status=503),
                    _entry("unbrowser", action="navigate", status=200),
                ],
            },
        )
        self.assertTrue(result["recovery_eligible"])
        self.assertTrue(result["successful_same_tool_retry"])

    def test_required_recovery_probe_must_be_exact_and_first(self) -> None:
        treatment = _treatment(
            "brief_plan",
            "text_first",
            "submit_directly",
            "diagnose_retry_once",
        )
        expected_url = (
            "http://127.0.0.1:18090/distractor_recovery/7/easy/page_0"
        )
        base = [
            _entry("unbrowser", action="navigate", status=200),
            _entry("unbrowser", action="text", selector="body"),
        ]
        valid = assess_policy_adherence(
            treatment,
            {
                "planning_preamble": {
                    "present": True,
                    "line_count": 1,
                    "plan_marker": True,
                },
                "tool_trace": [
                    *base,
                    _entry(
                        "unbrowser",
                        action="click",
                        status=503,
                        url=expected_url,
                    ),
                    _entry("unbrowser", action="navigate", status=200),
                ],
            },
            required_recovery_probe_url=expected_url,
            required_recovery_probe_status=503,
        )
        self.assertTrue(valid["recovery_probe_adherent"])
        self.assertTrue(valid["recovery_eligible"])

        for invalid_trace in (
            [
                *base,
                _entry(
                    "unbrowser",
                    action="click",
                    status=503,
                    url=expected_url.replace("page_0", "page_1"),
                ),
            ],
            [
                *base,
                _entry("unbrowser", action="click", status=200, url="target"),
                _entry(
                    "unbrowser",
                    action="click",
                    status=503,
                    url=expected_url,
                ),
            ],
        ):
            result = assess_policy_adherence(
                treatment,
                {
                    "planning_preamble": {
                        "present": True,
                        "line_count": 1,
                        "plan_marker": True,
                    },
                    "tool_trace": invalid_trace,
                },
                required_recovery_probe_url=expected_url,
                required_recovery_probe_status=503,
            )
            self.assertFalse(result["recovery_probe_adherent"])
            self.assertFalse(result["recovery_eligible"])


if __name__ == "__main__":
    unittest.main()
