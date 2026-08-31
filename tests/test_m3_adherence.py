from __future__ import annotations

import hashlib
import json
import unittest

from pyreplab_harness.m3_adherence import (
    _navigate_receipt_action,
    _valid_auto_first_observation,
    assess_policy_adherence,
)
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

    def test_unmarked_aborted_call_after_cap_is_inferred_as_rejected(self) -> None:
        treatment = _treatment(
            "direct", "text_first", "submit_directly", "fail_fast"
        )
        trace = [_entry("bash", exit_code=0) for _ in range(6)]
        trace.append(
            {
                "tool_name": "bash",
                "is_error": True,
                "budget_rejected": False,
                "details": {},
            }
        )
        result = assess_policy_adherence(treatment, {"tool_trace": trace})
        self.assertEqual(result["admitted_tool_call_count"], 6)
        self.assertEqual(result["budget_rejection_count"], 1)
        self.assertTrue(result["tool_cap_compliant"])

    def test_marked_aborted_call_after_cap_is_inferred_as_rejected(self) -> None:
        treatment = _treatment(
            "direct", "text_first", "submit_directly", "fail_fast"
        )
        trace = [_entry("bash", exit_code=0) for _ in range(6)]
        trace.append(
            {
                "tool_name": "bash",
                "is_error": True,
                "budget_rejected": False,
                "operation_aborted": True,
                "details": {},
            }
        )
        result = assess_policy_adherence(treatment, {"tool_trace": trace})
        self.assertEqual(result["admitted_tool_call_count"], 6)
        self.assertEqual(result["budget_rejection_count"], 1)
        self.assertTrue(result["tool_cap_compliant"])

    def test_empty_error_before_cap_remains_admitted(self) -> None:
        treatment = _treatment(
            "direct", "text_first", "submit_directly", "fail_fast"
        )
        trace = [
            {
                "tool_name": "unbrowser",
                "is_error": True,
                "budget_rejected": False,
                "details": {},
            },
            *[_entry("bash", exit_code=0) for _ in range(5)],
        ]
        result = assess_policy_adherence(treatment, {"tool_trace": trace})
        self.assertEqual(result["admitted_tool_call_count"], 6)
        self.assertEqual(result["budget_rejection_count"], 0)
        self.assertTrue(result["tool_cap_compliant"])

    def test_explicit_pre_execution_rejection_is_not_admitted(self) -> None:
        treatment = _treatment(
            "direct", "text_first", "submit_directly", "fail_fast"
        )
        trace = [_entry("bash", exit_code=0) for _ in range(5)]
        trace.append({
            "tool_name": "unbrowser",
            "is_error": True,
            "budget_rejected": False,
            "operation_aborted": False,
            "pre_execution_rejected": True,
            "details": {},
        })
        result = assess_policy_adherence(treatment, {"tool_trace": trace})
        self.assertEqual(result["admitted_tool_call_count"], 5)
        self.assertEqual(result["budget_rejection_count"], 1)

    def test_successful_call_after_cap_remains_noncompliant(self) -> None:
        treatment = _treatment(
            "direct", "text_first", "submit_directly", "fail_fast"
        )
        trace = [_entry("bash", exit_code=0) for _ in range(6)]
        trace.append(_entry("unbrowser", action="navigate", status=200))
        trace.append(
            {
                "tool_name": "bash",
                "is_error": True,
                "budget_rejected": False,
                "details": {},
            }
        )
        result = assess_policy_adherence(treatment, {"tool_trace": trace})
        self.assertEqual(result["admitted_tool_call_count"], 7)
        self.assertEqual(result["budget_rejection_count"], 1)
        self.assertFalse(result["tool_cap_compliant"])

    def test_empty_error_after_cap_with_false_abort_marker_is_rejected(self) -> None:
        treatment = _treatment(
            "direct", "text_first", "submit_directly", "fail_fast"
        )
        trace = [_entry("bash", exit_code=0) for _ in range(6)]
        trace.append(
            {
                "tool_name": "bash",
                "is_error": True,
                "budget_rejected": False,
                "operation_aborted": False,
                "details": {},
            }
        )
        result = assess_policy_adherence(treatment, {"tool_trace": trace})
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


class ReceiptAdherenceTest(unittest.TestCase):
    """Tests for receipt-based first-observation detection."""

    def _receipt_entry(self, delivered_action: str):
        payload = "Example Domain"
        encoded = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return {
            "tool_name": "unbrowser",
            "is_error": False,
            "budget_rejected": False,
            "details": {
                "action": "navigate",
                "required_first_observation_receipt": {
                    "schema_version": "pyreplab-required-first-observation-v1",
                    "mechanism": "auto_delivered_first_observation",
                    "required_action": delivered_action,
                    "delivered_action": delivered_action,
                    "selector": "body" if delivered_action == "text" else None,
                    "delivered": True,
                    "payload_bytes": len(encoded),
                    "payload_sha256": hashlib.sha256(encoded).hexdigest(),
                },
                "auto_delivered_observation": payload,
            },
        }

    def test_receipt_navigate_counts_as_text_observation(self) -> None:
        treatment = _treatment(
            "direct", "text_first", "submit_directly", "fail_fast"
        )
        trajectory = {
            "planning_preamble": {"present": False},
            "tool_trace": [
                self._receipt_entry("text"),
                _entry("bash", exit_code=0),
            ],
        }
        result = assess_policy_adherence(treatment, trajectory)
        self.assertEqual(result["first_observation"], "text")
        self.assertTrue(result["observation_adherent"])
        self.assertEqual(result["receipt_mechanism"], "auto_delivered_first_observation")
        self.assertTrue(result["first_observation_receipt_valid"])

    def test_receipt_navigate_counts_as_blockmap_observation(self) -> None:
        treatment = _treatment(
            "direct", "structure_first", "submit_directly", "fail_fast"
        )
        trajectory = {
            "planning_preamble": {"present": False},
            "tool_trace": [
                self._receipt_entry("blockmap"),
                _entry("bash", exit_code=0),
            ],
        }
        result = assess_policy_adherence(treatment, trajectory)
        self.assertEqual(result["first_observation"], "blockmap")
        self.assertTrue(result["observation_adherent"])

    def test_malformed_receipt_schema_is_not_adherent(self) -> None:
        treatment = _treatment(
            "direct", "text_first", "submit_directly", "fail_fast"
        )
        bad_entry = {
            "tool_name": "unbrowser",
            "is_error": False,
            "budget_rejected": False,
            "details": {
                "action": "navigate",
                "required_first_observation_receipt": {
                    "schema_version": "wrong-schema-v1",
                    "mechanism": "auto_delivered_first_observation",
                    "required_action": "text",
                    "delivered_action": "text",
                    "delivered": True,
                    "payload_bytes": 42,
                    "payload_sha256": "a" * 64,
                },
            },
        }
        trajectory = {
            "planning_preamble": {"present": False},
            "tool_trace": [bad_entry, _entry("bash", exit_code=0)],
        }
        result = assess_policy_adherence(treatment, trajectory)
        self.assertIsNone(result["first_observation"])
        self.assertFalse(result["observation_adherent"])
        self.assertFalse(result["first_observation_receipt_valid"])

    def test_receipt_payload_hash_mismatch_is_not_adherent(self) -> None:
        treatment = _treatment(
            "direct", "text_first", "submit_directly", "fail_fast"
        )
        entry = self._receipt_entry("text")
        entry["details"]["required_first_observation_receipt"][
            "payload_sha256"
        ] = "0" * 64
        trajectory = {
            "planning_preamble": {"present": False},
            "tool_trace": [
                entry,
                _entry("unbrowser", action="text", selector="body"),
            ],
        }
        result = assess_policy_adherence(treatment, trajectory)
        self.assertIsNone(result["first_observation"])
        self.assertFalse(result["observation_adherent"])
        self.assertFalse(result["first_observation_receipt_valid"])

    def test_wrong_receipt_mechanism_is_non_adherent(self) -> None:
        treatment = _treatment(
            "direct", "text_first", "submit_directly", "fail_fast"
        )
        bad_entry = {
            "tool_name": "unbrowser",
            "is_error": False,
            "budget_rejected": False,
            "details": {
                "action": "navigate",
                "required_first_observation_receipt": {
                    "schema_version": "pyreplab-required-first-observation-v1",
                    "mechanism": "some_other_mechanism",
                    "required_action": "text",
                    "delivered_action": "text",
                    "delivered": True,
                    "payload_bytes": 42,
                    "payload_sha256": "a" * 64,
                },
            },
        }
        trajectory = {
            "planning_preamble": {"present": False},
            "tool_trace": [bad_entry, _entry("bash", exit_code=0)],
        }
        result = assess_policy_adherence(treatment, trajectory)
        self.assertIsNone(result["first_observation"])

    def test_no_receipt_historical_trace_works_unchanged(self) -> None:
        """Historical traces without receipts must produce the same results."""
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
        self.assertEqual(result["first_observation"], "text")
        self.assertTrue(result["observation_adherent"])
        self.assertIsNone(result["receipt_mechanism"])

    def test_navigate_receipt_action_helper(self) -> None:
        """Verify the helper correctly extracts the delivered action."""
        entry = self._receipt_entry("text")
        self.assertEqual(_navigate_receipt_action(entry), "text")

        # Non-navigate entry returns None.
        non_nav = _entry("unbrowser", action="blockmap")
        self.assertIsNone(_navigate_receipt_action(non_nav))

        # Entry without receipt returns None.
        no_receipt = _entry("unbrowser", action="navigate")
        self.assertIsNone(_navigate_receipt_action(no_receipt))


if __name__ == "__main__":
    unittest.main()
