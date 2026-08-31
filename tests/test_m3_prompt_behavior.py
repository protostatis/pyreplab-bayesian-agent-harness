"""Focused tests for the treatment-blind behavior classifier + producer.

Covers every completion and recovery label with model-free synthetic fixtures
(both the strict flattened restricted-evidence schema and the real
details-nested trajectory + raw Pi JSONL adapter path), malformed/unknown
semantics, tool-aware status, treatment-blind function signatures, deterministic
hashing, receipt tamper detection, and the recursive privacy validator.
"""

from __future__ import annotations

import inspect
import json
import unittest

from pyreplab_harness.m3_prompt_behavior import (
    BEHAVIOR_RECEIPT_SCHEMA_VERSION,
    CLASSIFIER_SOURCE,
    COMPLETION_LABELS,
    PILOT_TOOLS,
    RECOVERY_LABELS,
    RESULT_JSON_PATH,
    RESULT_WRITE_PILOT_SCOPE,
    RESULT_WRITE_RECEIPT_SCHEMA_VERSION,
    RESTRICTED_EVIDENCE_SCHEMA_VERSION,
    RestrictedEvidenceError,
    analyze_attempt,
    build_restricted_evidence,
    canonical_args_hash,
    classify_completion,
    classify_recovery,
    extract_request_args_hashes,
    module_source_sha256,
    privacy_scan,
    validate_behavior_receipt,
    validate_evidence,
    validate_result_write_receipt,
)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _entry(
    tool_name: str = "bash",
    *,
    is_error: bool = False,
    budget_rejected: bool = False,
    operation_aborted: bool = False,
    pre_execution_rejected: bool = False,
    result_submission: bool = False,
    infrastructure_error: bool = False,
    status=None,
    error_class: str | None = None,
    args=None,
    request_args_hash: str | None = None,
) -> dict:
    """Build one flattened restricted-evidence tool-trace entry (model-free)."""
    return {
        "tool_name": tool_name,
        "is_error": is_error,
        "budget_rejected": budget_rejected,
        "operation_aborted": operation_aborted,
        "pre_execution_rejected": pre_execution_rejected,
        "result_submission": result_submission,
        "infrastructure_error": infrastructure_error,
        "error_class": error_class,
        "status": status,
        "request_args_hash": (
            request_args_hash
            if request_args_hash is not None
            else (None if args is None else canonical_args_hash(args))
        ),
    }


def _evidence(*entries: dict, provider_turn_count: int = 1) -> dict:
    return {
        "schema_version": RESTRICTED_EVIDENCE_SCHEMA_VERSION,
        "provider_turn_count": provider_turn_count,
        "tool_trace": [dict(entry) for entry in entries],
    }


def _write(args: dict | None = None, **overrides) -> dict:
    entry = _entry(
        args=args if args is not None else {"command": "cat > result.json"},
        result_submission=True,
    )
    entry.update(overrides)
    return entry


def _read(tool: str = "unbrowser") -> dict:
    return _entry(tool)


def _valid_receipt(
    content_sha256: str | None = None, operation: str = "created"
) -> dict:
    return {
        "schema_version": RESULT_WRITE_RECEIPT_SCHEMA_VERSION,
        "pilot_scope": RESULT_WRITE_PILOT_SCOPE,
        "path": RESULT_JSON_PATH,
        "operation": operation,
        "content_sha256": content_sha256 or ("ab" * 32),
        "shape": "json_object",
        "verification_key": {"present": True, "type": "string"},
    }


# Real nested trajectory + raw JSONL helpers (for the adapter).


def _nested_entry(
    tool_name: str,
    tool_call_id: str | None = None,
    *,
    is_error: bool = False,
    budget_rejected: bool = False,
    operation_aborted: bool = False,
    pre_execution_rejected: bool = False,
    details: dict | None = None,
) -> dict:
    entry: dict = {
        "tool_name": tool_name,
        "is_error": is_error,
        "budget_rejected": budget_rejected,
        "operation_aborted": operation_aborted,
        "pre_execution_rejected": pre_execution_rejected,
        "details": details,
    }
    if tool_call_id is not None:
        entry["tool_call_id"] = tool_call_id
    return entry


def _trajectory(*entries: dict, provider_turn_count: int = 1) -> dict:
    return {
        "provider_turn_count": provider_turn_count,
        "tool_trace": list(entries),
    }


def _raw_tool_call(tool_call_id: str, name: str, arguments) -> dict:
    return {
        "type": "message_end",
        "message": {
            "role": "assistant",
            "content": [
                {"type": "toolCall", "id": tool_call_id, "name": name,
                 "arguments": arguments}
            ],
        },
    }


def _raw_jsonl(*events: dict) -> str:
    return "\n".join(json.dumps(event) for event in events)


class CompletionLabelFixturesTest(unittest.TestCase):
    """Exhaustive completion-label fixtures on flattened evidence."""

    def test_submitted_before_budget_block(self) -> None:
        evidence = _evidence(_read(), _write())
        result = classify_completion(evidence)
        self.assertEqual(result["label"], "submitted_before_budget_block")
        self.assertEqual(result["result_write_count"], 1)
        self.assertTrue(result["intended_behavior"])
        self.assertFalse(result["prior_budget_block"])
        self.assertFalse(result["prior_eligible_error"])
        self.assertEqual(result["post_submission_tool_attempts"], 0)

    def test_submitted_before_budget_block_with_valid_receipt(self) -> None:
        evidence = _evidence(_read(), _write())
        result = classify_completion(evidence, _valid_receipt())
        self.assertEqual(result["label"], "submitted_before_budget_block")
        self.assertTrue(result["intended_behavior"])

    def test_submitted_after_prior_error_is_intended(self) -> None:
        evidence = _evidence(
            _entry(error_class="tool_error", status=1),
            _read(),
            _write(),
        )
        result = classify_completion(evidence)
        self.assertEqual(result["label"], "submitted_after_prior_error")
        self.assertTrue(result["prior_eligible_error"])
        self.assertFalse(result["prior_budget_block"])
        self.assertTrue(result["intended_behavior"])

    def test_submitted_after_prior_budget_block_is_not_intended(self) -> None:
        evidence = _evidence(
            _entry(budget_rejected=True),
            _write(),
        )
        result = classify_completion(evidence)
        self.assertEqual(result["label"], "submitted_after_prior_error")
        self.assertTrue(result["prior_budget_block"])
        self.assertFalse(result["intended_behavior"])

    def test_no_submission(self) -> None:
        result = classify_completion(_evidence())
        self.assertEqual(result["label"], "no_submission")
        self.assertFalse(result["intended_behavior"])
        self.assertEqual(
            classify_completion(_evidence(_read()))["label"], "no_submission"
        )

    def test_multiple_submissions(self) -> None:
        evidence = _evidence(_write(), _write())
        result = classify_completion(evidence)
        self.assertEqual(result["label"], "multiple_submissions")
        self.assertEqual(result["result_write_count"], 2)
        self.assertFalse(result["intended_behavior"])

    def test_post_submission_tool_activity(self) -> None:
        evidence = _evidence(_write(), _read())
        result = classify_completion(evidence)
        self.assertEqual(result["label"], "post_submission_tool_activity")
        self.assertEqual(result["post_submission_tool_attempts"], 1)
        self.assertFalse(result["intended_behavior"])

    def test_post_submission_tool_activity_even_when_blocked(self) -> None:
        evidence = _evidence(_write(), _entry(budget_rejected=True))
        result = classify_completion(evidence)
        self.assertEqual(result["label"], "post_submission_tool_activity")
        self.assertEqual(result["post_submission_tool_attempts"], 1)

    def test_prior_error_but_later_activity_is_post_submission(self) -> None:
        evidence = _evidence(
            _entry(error_class="tool_error", status=1),
            _write(),
            _read(),
        )
        result = classify_completion(evidence)
        self.assertEqual(result["label"], "post_submission_tool_activity")
        self.assertFalse(result["intended_behavior"])

    def test_errored_submission_never_counts_as_a_valid_write(self) -> None:
        evidence = _evidence(
            _entry(result_submission=True, is_error=True, status=1)
        )
        self.assertEqual(classify_completion(evidence)["label"], "no_submission")

    def test_blocked_submission_never_counts_as_a_valid_write(self) -> None:
        evidence = _evidence(
            _entry(result_submission=True, budget_rejected=True)
        )
        self.assertEqual(classify_completion(evidence)["label"], "no_submission")

    def test_pre_rejected_submission_never_counts_as_a_valid_write(self) -> None:
        evidence = _evidence(
            _entry(result_submission=True, pre_execution_rejected=True)
        )
        self.assertEqual(classify_completion(evidence)["label"], "no_submission")


class RecoveryLabelFixturesTest(unittest.TestCase):
    """Exhaustive recovery-label fixtures on flattened evidence."""

    def test_no_opportunity(self) -> None:
        evidence = _evidence(_read(), _write())
        result = classify_recovery(evidence)
        self.assertEqual(result["label"], "no_opportunity")
        self.assertEqual(result["opportunity_count"], 0)

    def test_corrected_once_success(self) -> None:
        evidence = _evidence(
            _entry(args={"command": "false"}, error_class="tool_error", status=1),
            _entry(args={"command": "true"}, status=0),
        )
        result = classify_recovery(evidence)
        self.assertEqual(result["label"], "corrected_once_success")
        self.assertEqual(result["retry_count"], 1)
        self.assertEqual(result["changed_retry_count"], 1)
        self.assertEqual(result["unchanged_repeat_count"], 0)
        self.assertTrue(result["later_success"])

    def test_corrected_once_failed_then_stopped(self) -> None:
        evidence = _evidence(
            _entry(args={"command": "false"}, error_class="tool_error", status=1),
            _entry(args={"command": "true -x"}, error_class="tool_error", status=2),
        )
        result = classify_recovery(evidence)
        self.assertEqual(result["label"], "corrected_once_failed_then_stopped")
        self.assertFalse(result["later_success"])

    def test_unchanged_repeat(self) -> None:
        evidence = _evidence(
            _entry(args={"command": "false"}, error_class="tool_error", status=1),
            _entry(args={"command": "false"}, error_class="tool_error", status=1),
        )
        result = classify_recovery(evidence)
        self.assertEqual(result["label"], "unchanged_repeat")
        self.assertEqual(result["unchanged_repeat_count"], 1)
        self.assertEqual(result["changed_retry_count"], 0)

    def test_unchanged_repeat_that_succeeded_is_still_unchanged(self) -> None:
        evidence = _evidence(
            _entry(args={"command": "flaky"}, error_class="tool_error", status=1),
            _entry(args={"command": "flaky"}, status=0),
        )
        result = classify_recovery(evidence)
        self.assertEqual(result["label"], "unchanged_repeat")
        self.assertTrue(result["later_success"])

    def test_retry_loop_many_retries(self) -> None:
        evidence = _evidence(
            _entry(args={"command": "false"}, error_class="tool_error", status=1),
            _entry(args={"command": "false"}, error_class="tool_error", status=1),
            _entry(args={"command": "false"}, error_class="tool_error", status=1),
        )
        result = classify_recovery(evidence)
        self.assertEqual(result["label"], "retry_loop")
        self.assertEqual(result["retry_count"], 2)

    def test_retry_loop_two_changed_retries(self) -> None:
        evidence = _evidence(
            _entry(args={"command": "a"}, error_class="tool_error", status=1),
            _entry(args={"command": "b"}, error_class="tool_error", status=1),
            _entry(args={"command": "c"}, error_class="tool_error", status=1),
        )
        result = classify_recovery(evidence)
        self.assertEqual(result["label"], "retry_loop")
        self.assertEqual(result["changed_retry_count"], 2)

    def test_abandoned_after_error(self) -> None:
        evidence = _evidence(_read(), _entry(error_class="tool_error", status=404))
        result = classify_recovery(evidence)
        self.assertEqual(result["label"], "abandoned_after_error")
        self.assertEqual(result["opportunity_count"], 1)
        self.assertEqual(result["retry_count"], 0)

    def test_budget_rejected_flag_is_not_an_opportunity(self) -> None:
        evidence = _evidence(_entry(budget_rejected=True))
        self.assertEqual(classify_recovery(evidence)["label"], "no_opportunity")

    def test_budget_limit_error_class_is_not_an_opportunity(self) -> None:
        evidence = _evidence(_entry(error_class="budget_limit", is_error=True))
        self.assertEqual(classify_recovery(evidence)["label"], "no_opportunity")

    def test_infrastructure_flag_is_not_an_opportunity(self) -> None:
        evidence = _evidence(_entry(infrastructure_error=True, is_error=True))
        self.assertEqual(classify_recovery(evidence)["label"], "no_opportunity")

    def test_infrastructure_error_class_is_not_an_opportunity(self) -> None:
        evidence = _evidence(_entry(error_class="infrastructure", is_error=True))
        self.assertEqual(classify_recovery(evidence)["label"], "no_opportunity")

    def test_pre_execution_rejection_is_not_an_opportunity(self) -> None:
        evidence = _evidence(_entry(pre_execution_rejected=True))
        self.assertEqual(classify_recovery(evidence)["label"], "no_opportunity")

    def test_abandoned_recovery_does_not_need_args_hashes(self) -> None:
        evidence = _evidence(_entry(error_class="tool_error", status=1))
        self.assertEqual(classify_recovery(evidence)["label"], "abandoned_after_error")

    def test_unknown_when_retries_lack_fingerprints(self) -> None:
        evidence = _evidence(
            _entry(args={"command": "false"}, error_class="tool_error", status=1),
            _entry(error_class="tool_error", status=1),
        )
        result = classify_recovery(evidence)
        self.assertEqual(result["label"], "unknown")
        self.assertIn("request_args_hash", result["diagnostics"][0])

    def test_unknown_when_moved_on_without_retrying(self) -> None:
        evidence = _evidence(
            _entry(args={"command": "false"}, error_class="tool_error", status=1),
            _read(),
            _write(),
        )
        result = classify_recovery(evidence)
        self.assertEqual(result["label"], "unknown")
        self.assertTrue(result["later_success"])

    def test_unknown_when_changed_retry_failed_but_continued(self) -> None:
        evidence = _evidence(
            _entry(args={"command": "a"}, error_class="tool_error", status=1),
            _entry(args={"command": "b"}, error_class="tool_error", status=2),
            _read(),
        )
        self.assertEqual(classify_recovery(evidence)["label"], "unknown")


class ToolAwareStatusTest(unittest.TestCase):
    """bash exit-code 0 and browser HTTP 200 succeed; other values fail."""

    def test_bash_status_zero_is_success(self) -> None:
        evidence = _evidence(_entry(status=0))
        self.assertEqual(classify_recovery(evidence)["label"], "no_opportunity")

    def test_bash_nonzero_status_is_eligible_error(self) -> None:
        evidence = _evidence(_entry(status=1))
        result = classify_recovery(evidence)
        self.assertEqual(result["label"], "abandoned_after_error")
        self.assertEqual(result["opportunity_count"], 1)

    def test_unbrowser_http_200_is_success(self) -> None:
        evidence = _evidence(_entry("unbrowser", status=200))
        self.assertEqual(classify_recovery(evidence)["label"], "no_opportunity")

    def test_unbrowser_http_404_is_eligible_error(self) -> None:
        evidence = _evidence(_entry("unbrowser", status=404))
        result = classify_recovery(evidence)
        self.assertEqual(result["label"], "abandoned_after_error")
        self.assertEqual(result["opportunity_count"], 1)

    def test_browser_200_does_not_count_as_bash_error(self) -> None:
        browser = classify_recovery(_evidence(_entry("unbrowser", status=200)))
        bash = classify_recovery(_evidence(_entry("bash", status=200)))
        self.assertEqual(browser["label"], "no_opportunity")
        self.assertEqual(bash["label"], "abandoned_after_error")

    def test_is_error_fallback_without_status(self) -> None:
        evidence = _evidence(_entry(is_error=True))
        result = classify_recovery(evidence)
        self.assertEqual(result["label"], "abandoned_after_error")
        self.assertEqual(result["opportunity_count"], 1)

    def test_error_class_tool_error_without_status(self) -> None:
        evidence = _evidence(_entry(error_class="tool_error"))
        result = classify_recovery(evidence)
        self.assertEqual(result["label"], "abandoned_after_error")
        self.assertEqual(result["opportunity_count"], 1)


class MalformedAndFailClosedTest(unittest.TestCase):
    """Malformed/ambiguous evidence fails closed to unknown."""

    def test_non_mapping_evidence(self) -> None:
        self.assertEqual(classify_completion(None)["label"], "unknown")
        self.assertEqual(classify_recovery("nope")["label"], "unknown")

    def test_unknown_top_level_key(self) -> None:
        evidence = _evidence(_read())
        evidence["policy_identity"] = "leak"
        self.assertEqual(classify_completion(evidence)["label"], "unknown")

    def test_wrong_schema_version(self) -> None:
        evidence = _evidence(_read())
        evidence["schema_version"] = "future-version"
        self.assertEqual(classify_completion(evidence)["label"], "unknown")

    def test_trace_is_not_a_list(self) -> None:
        evidence = _evidence(_read())
        evidence["tool_trace"] = {"tool_name": "bash"}
        self.assertEqual(classify_completion(evidence)["label"], "unknown")

    def test_entry_unknown_key(self) -> None:
        evidence = _evidence(_read(), _write())
        evidence["tool_trace"][1]["args"] = {"command": "cat"}
        self.assertEqual(classify_completion(evidence)["label"], "unknown")

    def test_entry_missing_tool_name(self) -> None:
        evidence = _evidence(_read(), _write())
        evidence["tool_trace"][0].pop("tool_name")
        self.assertEqual(classify_completion(evidence)["label"], "unknown")

    def test_out_of_vocabulary_tool_name_is_rejected(self) -> None:
        evidence = _evidence(_entry("semantic_table", status=200))
        self.assertEqual(classify_completion(evidence)["label"], "unknown")

    def test_tool_call_id_key_is_rejected(self) -> None:
        # tool_call_id is transient-only and is not part of the schema.
        evidence = _evidence(_write())
        evidence["tool_trace"][0]["tool_call_id"] = "c0"
        self.assertEqual(classify_completion(evidence)["label"], "unknown")

    def test_raw_details_key_is_rejected(self) -> None:
        evidence = _evidence(_write())
        evidence["tool_trace"][0]["details"] = {"exit_code": 0}
        self.assertEqual(classify_completion(evidence)["label"], "unknown")

    def test_raw_error_string_key_is_rejected(self) -> None:
        evidence = _evidence(_write())
        evidence["tool_trace"][0]["error"] = "something failed"
        self.assertEqual(classify_completion(evidence)["label"], "unknown")

    def test_unknown_error_class_is_rejected(self) -> None:
        evidence = _evidence(_entry(error_class="mystery_error"))
        self.assertEqual(classify_completion(evidence)["label"], "unknown")

    def test_malformed_request_args_hash(self) -> None:
        evidence = _evidence(
            _entry(args={"command": "a"}, error_class="tool_error", status=1),
            _entry(args={"command": "b"}, error_class="tool_error", status=1),
        )
        evidence["tool_trace"][1]["request_args_hash"] = "not-a-hash"
        self.assertEqual(classify_recovery(evidence)["label"], "unknown")

    def test_invalid_result_write_receipt(self) -> None:
        receipt = _valid_receipt()
        receipt["shape"] = "json_array"
        self.assertEqual(
            classify_completion(_evidence(_read(), _write()), receipt)["label"],
            "unknown",
        )

    def test_receipt_inconsistent_with_trace(self) -> None:
        evidence = _evidence(_write(), _write())
        self.assertEqual(
            classify_completion(evidence, _valid_receipt())["label"], "unknown"
        )

    def test_receipt_without_trace_write_is_unknown(self) -> None:
        evidence = _evidence(_read())
        self.assertEqual(
            classify_completion(evidence, _valid_receipt())["label"], "unknown"
        )

    def test_receipt_missing_key_descriptor_is_invalid(self) -> None:
        receipt = _valid_receipt()
        receipt["verification_key"] = {"present": False, "type": "string"}
        self.assertTrue(validate_result_write_receipt(receipt))
        self.assertEqual(
            classify_completion(_evidence(_read(), _write()), receipt)["label"],
            "unknown",
        )

    def test_receipt_replacement_operation_is_valid(self) -> None:
        result = classify_completion(
            _evidence(_read(), _write()), _valid_receipt(operation="replaced")
        )
        self.assertEqual(result["label"], "submitted_before_budget_block")

    def test_receipt_wrong_pilot_scope_is_invalid(self) -> None:
        receipt = _valid_receipt()
        receipt["pilot_scope"] = "some-other-pilot"
        self.assertTrue(validate_result_write_receipt(receipt))

    def test_validate_evidence_reports_empty_trace_as_valid(self) -> None:
        self.assertEqual(validate_evidence(_evidence()), [])


class TreatmentBlindSignatureTest(unittest.TestCase):
    """Public APIs accept only evidence/trajectory + optional inputs."""

    FORBIDDEN_PARAMETER_TERMS = (
        "treatment",
        "policy",
        "verifier",
        "verified",
        "success",
        "oracle",
        "answer",
        "key",
        "template",
        "prompt",
        "task",
        "nonce",
        "outcome",
    )

    def test_public_signatures_are_treatment_blind(self) -> None:
        functions = {
            "classify_completion": classify_completion,
            "classify_recovery": classify_recovery,
            "analyze_attempt": analyze_attempt,
            "build_restricted_evidence": build_restricted_evidence,
            "extract_request_args_hashes": extract_request_args_hashes,
            "canonical_args_hash": canonical_args_hash,
            "privacy_scan": privacy_scan,
        }
        for name, function in functions.items():
            parameters = inspect.signature(function).parameters
            for parameter_name in parameters:
                lowered = parameter_name.casefold()
                for term in self.FORBIDDEN_PARAMETER_TERMS:
                    self.assertNotIn(
                        term,
                        lowered,
                        f"{name} parameter {parameter_name!r} violates "
                        f"treatment-blindness (matches {term!r})",
                    )

    def test_known_labels_are_exactly_the_declared_sets(self) -> None:
        self.assertEqual(
            set(COMPLETION_LABELS),
            {
                "submitted_before_budget_block",
                "submitted_after_prior_error",
                "no_submission",
                "multiple_submissions",
                "post_submission_tool_activity",
                "unknown",
            },
        )
        self.assertEqual(
            set(RECOVERY_LABELS),
            {
                "no_opportunity",
                "corrected_once_success",
                "corrected_once_failed_then_stopped",
                "unchanged_repeat",
                "retry_loop",
                "abandoned_after_error",
                "unknown",
            },
        )
        self.assertEqual(PILOT_TOOLS, frozenset({"bash", "unbrowser"}))


class DeterministicHashTest(unittest.TestCase):
    """Canonical args hashing and receipt hashing are deterministic."""

    def test_canonical_args_hash_ignores_key_order(self) -> None:
        first = canonical_args_hash({"command": "true", "cwd": "/tmp"})
        second = canonical_args_hash({"cwd": "/tmp", "command": "true"})
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_canonical_args_hash_changes_with_content(self) -> None:
        self.assertNotEqual(
            canonical_args_hash({"command": "true"}),
            canonical_args_hash({"command": "false"}),
        )

    def test_analyze_attempt_is_deterministic(self) -> None:
        evidence = _evidence(_read(), _write())
        first = analyze_attempt(evidence, _valid_receipt())
        second = analyze_attempt(evidence, _valid_receipt())
        self.assertEqual(first, second)
        self.assertEqual(first["receipt_hash"], second["receipt_hash"])

    def test_different_evidence_yields_different_receipt_hash(self) -> None:
        clean = analyze_attempt(_evidence(_read(), _write()))
        noisy = analyze_attempt(_evidence(_read(), _write(), _read()))
        self.assertNotEqual(clean["receipt_hash"], noisy["receipt_hash"])

    def test_module_source_identity_is_deterministic(self) -> None:
        self.assertEqual(module_source_sha256(), module_source_sha256())
        self.assertEqual(len(module_source_sha256()), 64)

    def test_receipt_is_self_consistent(self) -> None:
        receipt = analyze_attempt(_evidence(_read(), _write()))
        self.assertEqual(validate_behavior_receipt(receipt), [])


class TamperDetectionTest(unittest.TestCase):
    """Self-hashed receipts detect any modification."""

    def test_tampered_completion_label_is_detected(self) -> None:
        receipt = analyze_attempt(_evidence(_read(), _write()))
        tampered = dict(receipt)
        tampered["completion"] = {**receipt["completion"], "label": "no_submission"}
        violations = validate_behavior_receipt(tampered)
        self.assertTrue(any("receipt_hash" in v for v in violations), violations)

    def test_tampered_counter_is_detected(self) -> None:
        receipt = analyze_attempt(_evidence(_read(), _write()))
        tampered = dict(receipt)
        tampered["recovery"] = {**receipt["recovery"], "retry_count": 99}
        violations = validate_behavior_receipt(tampered)
        self.assertTrue(any("receipt_hash" in v for v in violations), violations)

    def test_tampered_intended_behavior_is_detected(self) -> None:
        receipt = analyze_attempt(_evidence(_read(), _write()))
        tampered = dict(receipt)
        tampered["completion"] = {
            **receipt["completion"],
            "intended_behavior": False,
        }
        violations = validate_behavior_receipt(tampered)
        self.assertTrue(any("receipt_hash" in v for v in violations), violations)

    def test_tampered_receipt_hash_itself_is_detected(self) -> None:
        receipt = analyze_attempt(_evidence(_read(), _write()))
        tampered = dict(receipt)
        tampered["receipt_hash"] = "00" * 32
        violations = validate_behavior_receipt(tampered)
        self.assertTrue(any("receipt_hash" in v for v in violations), violations)

    def test_unknown_schema_version_is_detected(self) -> None:
        receipt = analyze_attempt(_evidence(_read(), _write()))
        tampered = dict(receipt)
        tampered["schema_version"] = "tampered"
        violations = validate_behavior_receipt(tampered)
        self.assertTrue(any("schema_version" in v for v in violations), violations)

    def test_injected_forbidden_field_is_detected_in_receipt(self) -> None:
        receipt = analyze_attempt(_evidence(_read(), _write()))
        tampered = dict(receipt)
        tampered["policy_id"] = "lean-6"
        violations = validate_behavior_receipt(tampered)
        self.assertTrue(any("forbidden field" in v for v in violations), violations)


class PrivacyTest(unittest.TestCase):
    """The recursive privacy validator rejects structured leakage."""

    def test_privacy_scan_rejects_policy_identity(self) -> None:
        evidence = _evidence(_read(), _write())
        evidence["policy_id"] = "lean-6"
        self.assertTrue(privacy_scan(evidence))
        self.assertEqual(classify_completion(evidence)["label"], "unknown")

    def test_privacy_scan_rejects_verification_key_value(self) -> None:
        evidence = _evidence(_read(), _write())
        evidence["verification_key"] = "supersecret-nonce"
        self.assertTrue(privacy_scan(evidence))
        self.assertEqual(classify_completion(evidence)["label"], "unknown")

    def test_privacy_scan_rejects_oracle_and_answer(self) -> None:
        evidence = _evidence(_read(), _write())
        evidence["oracle"] = {"nonce": "abc"}
        evidence["answer"] = "the-correct-answer"
        violations = privacy_scan(evidence)
        self.assertTrue(violations)
        self.assertEqual(classify_recovery(evidence)["label"], "unknown")

    def test_privacy_scan_rejects_task_template(self) -> None:
        evidence = _evidence(_read(), _write())
        evidence["task_template"] = "single_page_extraction"
        self.assertTrue(privacy_scan(evidence))
        self.assertEqual(classify_completion(evidence)["label"], "unknown")

    def test_privacy_scan_rejects_verifier_outcome_fields(self) -> None:
        for field in ("success", "verification", "verified_success", "failure_code"):
            evidence = _evidence(_read(), _write())
            evidence[field] = True if field != "failure_code" else "nonce_mismatch"
            self.assertTrue(privacy_scan(evidence), f"{field} should be flagged")
            self.assertEqual(
                classify_completion(evidence)["label"], "unknown", field
            )

    def test_privacy_scan_rejects_system_prompt(self) -> None:
        evidence = _evidence(_read(), _write())
        evidence["system_prompt"] = "decompose and retry once"
        self.assertTrue(privacy_scan(evidence))

    def test_privacy_scan_rejects_raw_request_args(self) -> None:
        evidence = _evidence(_read(), _write())
        evidence["request_args"] = {"command": "echo secret-key-value"}
        self.assertTrue(privacy_scan(evidence))
        self.assertEqual(classify_completion(evidence)["label"], "unknown")

    def test_privacy_scan_allows_hashed_args_field(self) -> None:
        evidence = _evidence(
            _entry(args={"command": "true"}, error_class="tool_error", status=1),
            _entry(args={"command": "true"}, error_class="tool_error", status=1),
        )
        self.assertEqual(privacy_scan(evidence), [])

    def test_privacy_scan_accepts_receipt_key_descriptor(self) -> None:
        self.assertEqual(privacy_scan(_valid_receipt()), [])

    def test_privacy_scan_rejects_receipt_carrying_key_value(self) -> None:
        receipt = _valid_receipt()
        receipt["verification_key"] = "literal-nonce-value"
        violations = privacy_scan(receipt)
        self.assertTrue(
            any("verification_key" in v for v in violations), violations
        )
        self.assertEqual(
            classify_completion(_evidence(_read(), _write()), receipt)["label"],
            "unknown",
        )

    def test_privacy_scan_is_recursive_into_nested_structures(self) -> None:
        payload = {"tool_trace": [{"nested": {"treatment_bundle_hash": "deadbeef"}}]}
        violations = privacy_scan(payload)
        self.assertTrue(
            any("treatment_bundle_hash" in v for v in violations), violations
        )

    def test_privacy_scan_avoids_false_positive_on_benign_source_keys(self) -> None:
        self.assertEqual(
            privacy_scan({"prompt_tokens": 12, "completion_tokens": 3}), []
        )

    def test_privacy_scan_emits_no_duplicate_violations(self) -> None:
        violations = privacy_scan({"verifier_id": "v1"})
        self.assertEqual(len(violations), 1, violations)

    def test_behavior_receipt_is_privacy_clean(self) -> None:
        receipt = analyze_attempt(
            _evidence(
                _entry(args={"command": "a"}, error_class="tool_error", status=1),
                _entry(args={"command": "b"}, error_class="tool_error", status=1),
                _write(),
            )
        )
        self.assertEqual(privacy_scan(receipt), [])

    def test_behavior_receipt_contains_only_bounded_fields(self) -> None:
        receipt = analyze_attempt(
            _evidence(
                _entry(args={"command": "a"}, error_class="tool_error", status=1),
                _entry(args={"command": "b"}, error_class="tool_error", status=1),
                _write(),
            )
        )
        serialized = json.dumps(receipt, sort_keys=True)
        for secret_fragment in ("command", "secret", "a1b2c3d4", "result.json"):
            self.assertNotIn(secret_fragment, serialized)
        # Request-argument hashes/fingerprints are never persisted into the
        # behavior receipt (they can leak low-entropy keys); output is bounded
        # labels/counters only.
        self.assertNotIn("retry_fingerprints", receipt["recovery"])
        self.assertNotIn("args_sha256", serialized)
        self.assertNotIn("request_args_hash", serialized)

    def test_behavior_receipt_never_persists_request_hashes(self) -> None:
        receipt = analyze_attempt(
            _evidence(
                _entry(args={"command": "secret-key-value"}, error_class="tool_error", status=1),
                _entry(args={"command": "secret-key-value"}, error_class="tool_error", status=1),
                _write(),
            )
        )
        serialized = json.dumps(receipt, sort_keys=True)
        self.assertNotIn("secret", serialized)
        self.assertNotIn("args_sha256", serialized)
        self.assertEqual(
            set(receipt["recovery"]),
            {
                "label",
                "opportunity_count",
                "retry_count",
                "changed_retry_count",
                "unchanged_repeat_count",
                "later_success",
            },
        )

    def test_receipt_records_unconditional_itt_inclusion(self) -> None:
        receipt = analyze_attempt(_evidence(_entry(error_class="tool_error", status=1)))
        self.assertEqual(receipt["itt_inclusion"], "unconditional")
        receipt_loop = analyze_attempt(
            _evidence(
                _entry(args={"command": "a"}, error_class="tool_error", status=1),
                _entry(args={"command": "a"}, error_class="tool_error", status=1),
                _entry(args={"command": "a"}, error_class="tool_error", status=1),
            )
        )
        self.assertEqual(receipt_loop["recovery"]["label"], "retry_loop")
        self.assertEqual(receipt_loop["itt_inclusion"], "unconditional")


class ProducerAdapterTest(unittest.TestCase):
    """The producer converts real nested trajectory + raw JSONL to evidence."""

    def test_details_nested_submission(self) -> None:
        trajectory = _trajectory(
            _nested_entry(
                "bash", "c0", details={"exit_code": 0, "result_submission": True}
            ),
        )
        evidence = build_restricted_evidence(trajectory)
        self.assertEqual(validate_evidence(evidence), [])
        self.assertEqual(privacy_scan(evidence), [])
        entry = evidence["tool_trace"][0]
        self.assertTrue(entry["result_submission"])
        self.assertEqual(entry["status"], 0)
        self.assertIsNone(entry["error_class"])
        self.assertEqual(
            classify_completion(evidence)["label"], "submitted_before_budget_block"
        )

    def test_errored_submission_is_not_a_valid_write(self) -> None:
        trajectory = _trajectory(
            _nested_entry(
                "bash", "c0", details={"exit_code": 1, "result_submission": True}
            ),
        )
        evidence = build_restricted_evidence(trajectory)
        self.assertEqual(evidence["tool_trace"][0]["error_class"], "tool_error")
        self.assertEqual(classify_completion(evidence)["label"], "no_submission")

    def test_bash_status_zero_succeeds(self) -> None:
        trajectory = _trajectory(
            _nested_entry(
                "bash", "c0", details={"exit_code": 0, "result_submission": False}
            )
        )
        evidence = build_restricted_evidence(trajectory)
        self.assertEqual(evidence["tool_trace"][0]["status"], 0)
        self.assertIsNone(evidence["tool_trace"][0]["error_class"])
        self.assertEqual(classify_recovery(evidence)["label"], "no_opportunity")

    def test_bash_nonzero_exit_code_is_ordinary_error(self) -> None:
        trajectory = _trajectory(_nested_entry("bash", "c0", details={"exit_code": 1}))
        evidence = build_restricted_evidence(trajectory)
        self.assertEqual(evidence["tool_trace"][0]["error_class"], "tool_error")
        result = classify_recovery(evidence)
        self.assertEqual(result["label"], "abandoned_after_error")
        self.assertEqual(result["opportunity_count"], 1)

    def test_bash_status_fallback_when_exit_code_absent(self) -> None:
        # exit_code first; fall back to integer status.
        success = build_restricted_evidence(
            _trajectory(_nested_entry("bash", "c0", details={"status": 0}))
        )
        self.assertEqual(success["tool_trace"][0]["status"], 0)
        self.assertIsNone(success["tool_trace"][0]["error_class"])
        failure = build_restricted_evidence(
            _trajectory(_nested_entry("bash", "c0", details={"status": 2}))
        )
        self.assertEqual(failure["tool_trace"][0]["status"], 2)
        self.assertEqual(failure["tool_trace"][0]["error_class"], "tool_error")

    def test_bash_exit_code_takes_precedence_over_status(self) -> None:
        trajectory = _trajectory(
            _nested_entry("bash", "c0", details={"exit_code": 0, "status": 1})
        )
        evidence = build_restricted_evidence(trajectory)
        self.assertEqual(evidence["tool_trace"][0]["status"], 0)
        self.assertIsNone(evidence["tool_trace"][0]["error_class"])

    def test_success_status_dominates_benign_error_text(self) -> None:
        trajectory = _trajectory(
            _nested_entry(
                "bash", "c0",
                details={"exit_code": 0, "error": "some benign warning text"},
            )
        )
        evidence = build_restricted_evidence(trajectory)
        self.assertIsNone(evidence["tool_trace"][0]["error_class"])

    def test_infrastructure_flag_dominates_success_status(self) -> None:
        trajectory = _trajectory(
            _nested_entry(
                "bash", "c0",
                details={"exit_code": 0, "infrastructure_error": True,
                         "error": "warning"},
            )
        )
        evidence = build_restricted_evidence(trajectory)
        self.assertEqual(evidence["tool_trace"][0]["error_class"], "infrastructure")

    def test_nonzero_status_creates_tool_error(self) -> None:
        trajectory = _trajectory(_nested_entry("bash", "c0", details={"exit_code": 3}))
        evidence = build_restricted_evidence(trajectory)
        self.assertEqual(evidence["tool_trace"][0]["error_class"], "tool_error")

    def test_is_error_creates_tool_error(self) -> None:
        trajectory = _trajectory(
            _nested_entry("bash", "c0", is_error=True, details={"exit_code": 0})
        )
        evidence = build_restricted_evidence(trajectory)
        self.assertEqual(evidence["tool_trace"][0]["error_class"], "tool_error")

    def test_pre_execution_rejection_error_class_from_flag(self) -> None:
        trajectory = _trajectory(
            _nested_entry(
                "bash", "c0", pre_execution_rejected=True, details={"exit_code": 0}
            )
        )
        evidence = build_restricted_evidence(trajectory)
        self.assertEqual(
            evidence["tool_trace"][0]["error_class"], "pre_execution_rejection"
        )
        self.assertEqual(classify_recovery(evidence)["label"], "no_opportunity")

    def test_unbrowser_benign_details_and_http_status(self) -> None:
        trajectory = _trajectory(
            _nested_entry(
                "unbrowser",
                "c0",
                details={
                    "action": "navigate",
                    "selector": None,
                    "allowed_url": "http://127.0.0.1:18090/",
                    "runtime_version": "0.0.19",
                    "status": 200,
                    "url": "http://127.0.0.1:18090/foo",
                },
            ),
        )
        evidence = build_restricted_evidence(trajectory)
        self.assertEqual(privacy_scan(evidence), [])
        entry = evidence["tool_trace"][0]
        self.assertEqual(entry["status"], 200)
        self.assertIsNone(entry["error_class"])
        self.assertNotIn("url", entry)
        self.assertNotIn("selector", entry)
        self.assertEqual(classify_recovery(evidence)["label"], "no_opportunity")

    def test_unbrowser_infrastructure_error_class(self) -> None:
        trajectory = _trajectory(
            _nested_entry(
                "unbrowser",
                "c0",
                details={"action": "text", "error": "connection reset",
                         "infrastructure_error": True},
            ),
        )
        evidence = build_restricted_evidence(trajectory)
        entry = evidence["tool_trace"][0]
        self.assertTrue(entry["infrastructure_error"])
        self.assertEqual(entry["error_class"], "infrastructure")
        self.assertNotIn("error", entry)
        self.assertEqual(classify_recovery(evidence)["label"], "no_opportunity")

    def test_bash_budget_limit_error_class(self) -> None:
        trajectory = _trajectory(
            _nested_entry(
                "bash", "c0",
                details={"exit_code": -1, "error": "shared_tool_limit"},
            ),
        )
        evidence = build_restricted_evidence(trajectory)
        self.assertEqual(evidence["tool_trace"][0]["error_class"], "budget_limit")
        self.assertEqual(classify_recovery(evidence)["label"], "no_opportunity")

    # ---- recovery labels reached via raw toolCall argument hashes ----

    def test_corrected_once_success_via_raw_args(self) -> None:
        raw = _raw_jsonl(
            _raw_tool_call("c0", "bash", {"command": "false"}),
            _raw_tool_call("c1", "bash", {"command": "true"}),
        )
        trajectory = _trajectory(
            _nested_entry("bash", "c0", details={"exit_code": 1}),
            _nested_entry("bash", "c1", details={"exit_code": 0}),
        )
        evidence = build_restricted_evidence(trajectory, raw)
        result = classify_recovery(evidence)
        self.assertEqual(result["label"], "corrected_once_success")
        self.assertEqual(result["changed_retry_count"], 1)

    def test_corrected_once_failed_then_stopped_via_raw_args(self) -> None:
        raw = _raw_jsonl(
            _raw_tool_call("c0", "bash", {"command": "false"}),
            _raw_tool_call("c1", "bash", {"command": "false -x"}),
        )
        trajectory = _trajectory(
            _nested_entry("bash", "c0", details={"exit_code": 1}),
            _nested_entry("bash", "c1", details={"exit_code": 1}),
        )
        evidence = build_restricted_evidence(trajectory, raw)
        self.assertEqual(
            classify_recovery(evidence)["label"], "corrected_once_failed_then_stopped"
        )

    def test_unchanged_repeat_via_raw_args(self) -> None:
        raw = _raw_jsonl(
            _raw_tool_call("c0", "bash", {"command": "false"}),
            _raw_tool_call("c1", "bash", {"command": "false"}),
        )
        trajectory = _trajectory(
            _nested_entry("bash", "c0", details={"exit_code": 1}),
            _nested_entry("bash", "c1", details={"exit_code": 1}),
        )
        evidence = build_restricted_evidence(trajectory, raw)
        result = classify_recovery(evidence)
        self.assertEqual(result["label"], "unchanged_repeat")
        self.assertEqual(result["unchanged_repeat_count"], 1)

    def test_retry_loop_via_raw_args(self) -> None:
        raw = _raw_jsonl(
            _raw_tool_call("c0", "bash", {"command": "false"}),
            _raw_tool_call("c1", "bash", {"command": "false"}),
            _raw_tool_call("c2", "bash", {"command": "false"}),
        )
        trajectory = _trajectory(
            _nested_entry("bash", "c0", details={"exit_code": 1}),
            _nested_entry("bash", "c1", details={"exit_code": 1}),
            _nested_entry("bash", "c2", details={"exit_code": 1}),
        )
        evidence = build_restricted_evidence(trajectory, raw)
        self.assertEqual(classify_recovery(evidence)["label"], "retry_loop")

    def test_abandoned_after_error_via_raw_args(self) -> None:
        raw = _raw_jsonl(_raw_tool_call("c0", "bash", {"command": "false"}))
        trajectory = _trajectory(_nested_entry("bash", "c0", details={"exit_code": 1}))
        evidence = build_restricted_evidence(trajectory, raw)
        self.assertEqual(classify_recovery(evidence)["label"], "abandoned_after_error")

    def test_no_opportunity_via_raw_args(self) -> None:
        raw = _raw_jsonl(_raw_tool_call("c0", "bash", {"command": "true"}))
        trajectory = _trajectory(_nested_entry("bash", "c0", details={"exit_code": 0}))
        evidence = build_restricted_evidence(trajectory, raw)
        self.assertEqual(classify_recovery(evidence)["label"], "no_opportunity")

    def test_missing_hash_fails_closed_to_unknown(self) -> None:
        trajectory = _trajectory(
            _nested_entry("bash", "c0", details={"exit_code": 1}),
            _nested_entry("bash", "c1", details={"exit_code": 1}),
        )
        evidence = build_restricted_evidence(trajectory)
        self.assertIsNone(evidence["tool_trace"][0]["request_args_hash"])
        result = classify_recovery(evidence)
        self.assertEqual(result["label"], "unknown")

    def test_arguments_as_json_string_form(self) -> None:
        raw = _raw_jsonl(
            _raw_tool_call("c0", "bash", json.dumps({"command": "false"})),
            _raw_tool_call("c1", "bash", json.dumps({"command": "true"})),
        )
        trajectory = _trajectory(
            _nested_entry("bash", "c0", details={"exit_code": 1}),
            _nested_entry("bash", "c1", details={"exit_code": 0}),
        )
        evidence = build_restricted_evidence(trajectory, raw)
        self.assertEqual(
            classify_recovery(evidence)["label"], "corrected_once_success"
        )

    def test_extract_request_args_hashes_matches_ids(self) -> None:
        raw = [
            _raw_tool_call("alpha", "bash", {"command": "a"}),
            _raw_tool_call("beta", "bash", {"command": "b"}),
        ]
        hashes = extract_request_args_hashes(raw)
        self.assertEqual(set(hashes), {"alpha", "beta"})
        self.assertEqual(hashes["alpha"], canonical_args_hash({"command": "a"}))
        self.assertEqual(hashes["beta"], canonical_args_hash({"command": "b"}))

    def test_literal_key_and_args_never_emitted(self) -> None:
        secret = "supersecret-verification-nonce-abc123"
        raw = _raw_jsonl(
            _raw_tool_call("c0", "bash", {"command": f"cat {secret} > result.json"}),
            _raw_tool_call(
                "c1",
                "bash",
                {"command": f"printf '{{\"verification_key\": \"{secret}\"}}' > result.json"},
            ),
        )
        trajectory = _trajectory(
            _nested_entry("bash", "c0", details={"exit_code": 0}),
            _nested_entry(
                "bash", "c1",
                details={"exit_code": 0, "result_submission": True},
            ),
        )
        evidence = build_restricted_evidence(trajectory, raw)
        receipt = analyze_attempt(evidence, _valid_receipt())
        for blob in (
            json.dumps(evidence, sort_keys=True),
            json.dumps(receipt, sort_keys=True),
        ):
            self.assertNotIn(secret, blob)
            self.assertNotIn("result.json", blob)
            self.assertNotIn("verification_key", blob)
        self.assertEqual(receipt["completion"]["label"], "submitted_before_budget_block")

    def test_adapter_output_is_valid_and_clean(self) -> None:
        trajectory = _trajectory(
            _nested_entry("bash", "c0", details={"exit_code": 1}),
            _nested_entry("bash", "c1", details={"exit_code": 0}),
            _nested_entry(
                "bash", "c2", details={"exit_code": 0, "result_submission": True}
            ),
        )
        raw = _raw_jsonl(
            _raw_tool_call("c0", "bash", {"command": "false"}),
            _raw_tool_call("c1", "bash", {"command": "true"}),
            _raw_tool_call("c2", "bash", {"command": "cat > result.json"}),
        )
        evidence = build_restricted_evidence(trajectory, raw)
        self.assertEqual(validate_evidence(evidence), [])
        self.assertEqual(privacy_scan(evidence), [])
        receipt = analyze_attempt(evidence)
        self.assertEqual(receipt["completion"]["label"], "submitted_after_prior_error")
        self.assertTrue(receipt["completion"]["intended_behavior"])
        self.assertEqual(receipt["recovery"]["label"], "corrected_once_success")

    def test_tool_call_id_is_not_emitted(self) -> None:
        raw = _raw_jsonl(_raw_tool_call("c0", "bash", {"command": "true"}))
        trajectory = _trajectory(_nested_entry("bash", "c0", details={"exit_code": 0}))
        evidence = build_restricted_evidence(trajectory, raw)
        self.assertNotIn("tool_call_id", evidence["tool_trace"][0])
        self.assertIsNotNone(evidence["tool_trace"][0]["request_args_hash"])


class ProducerFailClosedTest(unittest.TestCase):
    """The producer raises on malformed source input; no invalid evidence."""

    def test_malformed_raw_json_raises(self) -> None:
        raw = (
            '{"type": "message_end", "message": {"role": "assistant", '
            '"content": [{"type": "toolCall", "id": "c0", "name": "bash", '
            '"arguments": {"command": "false"}}]}}\n'
            "NOT VALID JSON\n"
        )
        trajectory = _trajectory(_nested_entry("bash", "c0", details={"exit_code": 1}))
        with self.assertRaises(RestrictedEvidenceError):
            build_restricted_evidence(trajectory, raw)

    def test_non_object_raw_event_raises(self) -> None:
        raw = '["this", "is", "a", "list"]\n'
        trajectory = _trajectory(_nested_entry("bash", "c0", details={"exit_code": 1}))
        with self.assertRaises(RestrictedEvidenceError):
            build_restricted_evidence(trajectory, raw)

    def test_duplicate_tool_call_ids_raise(self) -> None:
        raw = _raw_jsonl(
            _raw_tool_call("c0", "bash", {"command": "true"}),
            _raw_tool_call("c0", "bash", {"command": "true"}),
        )
        with self.assertRaises(RestrictedEvidenceError) as context:
            extract_request_args_hashes(raw)
        self.assertIn("duplicate", str(context.exception))

    def test_conflicting_tool_call_ids_raise(self) -> None:
        raw = _raw_jsonl(
            _raw_tool_call("c0", "bash", {"command": "true"}),
            _raw_tool_call("c0", "bash", {"command": "false"}),
        )
        with self.assertRaises(RestrictedEvidenceError) as context:
            extract_request_args_hashes(raw)
        self.assertIn("conflicting", str(context.exception))

    def test_malformed_trajectory_entry_raises(self) -> None:
        trajectory = _trajectory("not-an-object")
        with self.assertRaises(RestrictedEvidenceError):
            build_restricted_evidence(trajectory)

    def test_missing_tool_name_raises(self) -> None:
        entry = _nested_entry("bash", "c0", details={"exit_code": 0})
        entry.pop("tool_name")
        trajectory = _trajectory(entry)
        with self.assertRaises(RestrictedEvidenceError):
            build_restricted_evidence(trajectory)

    def test_out_of_vocabulary_tool_name_raises(self) -> None:
        trajectory = _trajectory(
            _nested_entry("semantic_table", "c0", details={"status": 200})
        )
        with self.assertRaises(RestrictedEvidenceError):
            build_restricted_evidence(trajectory)

    def test_non_mapping_trajectory_raises(self) -> None:
        with self.assertRaises(RestrictedEvidenceError):
            build_restricted_evidence(None)  # type: ignore[arg-type]

    def test_malformed_provider_turn_count_raises(self) -> None:
        trajectory = _trajectory(_nested_entry("bash", "c0", details={"exit_code": 0}))
        trajectory["provider_turn_count"] = "not-an-int"
        with self.assertRaises(RestrictedEvidenceError):
            build_restricted_evidence(trajectory)

    def test_negative_provider_turn_count_raises(self) -> None:
        trajectory = _trajectory(_nested_entry("bash", "c0", details={"exit_code": 0}))
        trajectory["provider_turn_count"] = -1
        with self.assertRaises(RestrictedEvidenceError):
            build_restricted_evidence(trajectory)


class BehaviorReceiptIntegrationTest(unittest.TestCase):
    """End-to-end behavior receipt shape."""

    def test_receipt_carries_completion_and_recovery_labels(self) -> None:
        receipt = analyze_attempt(
            _evidence(
                _entry(args={"command": "false"}, error_class="tool_error", status=1),
                _entry(args={"command": "true"}, status=0),
                _write(),
            )
        )
        self.assertEqual(receipt["schema_version"], BEHAVIOR_RECEIPT_SCHEMA_VERSION)
        self.assertEqual(receipt["classifier_source"], CLASSIFIER_SOURCE)
        self.assertEqual(receipt["completion"]["label"], "submitted_after_prior_error")
        self.assertTrue(receipt["completion"]["intended_behavior"])
        self.assertEqual(receipt["recovery"]["label"], "corrected_once_success")
        self.assertEqual(receipt["provider_turn_count"], 1)
        self.assertEqual(validate_behavior_receipt(receipt), [])

    def test_receipt_fails_closed_for_leaked_evidence(self) -> None:
        evidence = _evidence(_read(), _write())
        evidence["oracle"] = {"nonce": "x"}
        receipt = analyze_attempt(evidence)
        self.assertEqual(receipt["completion"]["label"], "unknown")
        self.assertEqual(receipt["recovery"]["label"], "unknown")

    def test_analyze_attempt_accepts_non_mapping_evidence(self) -> None:
        receipt = analyze_attempt(None)
        self.assertEqual(receipt["completion"]["label"], "unknown")
        self.assertEqual(validate_behavior_receipt(receipt), [])

    def test_behavior_receipt_excludes_result_write_receipt_content(self) -> None:
        # The result-write receipt is ephemeral: its content hash and shape must
        # never leak into the self-hashed behavior receipt.
        receipt = analyze_attempt(_evidence(_read(), _write()), _valid_receipt())
        serialized = json.dumps(receipt, sort_keys=True)
        for fragment in (
            "content_sha256",
            "verification_key",
            "pilot_scope",
            RESULT_JSON_PATH,
            "shape",
            "created",
        ):
            self.assertNotIn(fragment, serialized)
        self.assertEqual(validate_behavior_receipt(receipt), [])


if __name__ == "__main__":
    unittest.main()
