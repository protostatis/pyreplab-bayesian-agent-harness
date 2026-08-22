from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock
from uuid import uuid4

from pyreplab_harness.m3_empty_overlay_baseline import (
    EXPECTED_ATTEMPTS,
    REMOTE_PREFLIGHT_SCHEMA_VERSION,
    build_baseline_manifest,
    build_empty_overlay_registry,
    build_local_preflight,
)
from pyreplab_harness.m3_empty_overlay_execution import (
    AUTHORIZATION_SCHEMA_VERSION,
    AUTHORIZATION_STATEMENT,
    PANEL_RESULT_SCHEMA_VERSION,
    _append_record,
    _attempt_budget_consumption,
    _classify_attempt,
    _existing_completed_records,
    _load_ledger,
    _record_binds,
    _budget_reservation,
    _sha256_file,
    _validate_record,
    _worst_case_budget,
    _write_claim,
    _write_completion_receipt,
    analyze_baseline_results,
    build_authorization_request,
    deterministic_attempt_id,
    launch_authorized_baseline_detached,
    run_authorized_baseline,
    validate_execution_authorization,
)
from pyreplab_harness.m3_pilot import _canonical_hash, source_tree_hash
from pyreplab_harness.orchestrator import RemoteConfig, policy_spec_from_treatment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_IDENTITY = {
    "host": "ubuntu-local",
    "project": "/remote/project",
    "run_root": "/remote/project/.runs/empty-overlay",
    "python": "python3",
}
RESULT_FILENAME = "baseline.jsonl"

# Deterministic per-template outcome pattern for the synthetic analysis
# fixture. Each template has exactly six tasks (easy/medium/hard x two seeds);
# "discordant" means replica 0 succeeds and replica 1 fails.
TEMPLATE_OUTCOMES = {
    "single_page_extraction": ["both_success"] * 6,
    "table_filter_sort": ["both_fail"] * 6,
    "multi_page_navigation": ["discordant"] * 4 + ["both_success"] * 2,
    "search_filter_controls": ["both_success"] * 3 + ["both_fail"] * 3,
    "form_entry_validation": ["both_success"] * 3 + ["both_fail"] + ["discordant"] * 2,
    "distractor_recovery": ["both_success"] * 3 + ["both_fail"] * 2 + ["discordant"],
}


def _lifecycle_receipt() -> dict:
    payload = {
        "schema_version": "m3-unbrowser-lifecycle-stress-v1",
        "checked_at": "2026-08-14T00:00:00+00:00",
        "wait_seconds": 36.0,
        "elapsed_seconds": 36.1,
        "fixture_url": "http://127.0.0.1:18090/single_page_extraction/2026091001/easy",
        "navigation_status": 200,
        "post_wait_observation_sha256": "a" * 64,
        "runtime_version": "0.0.19",
        "confined": True,
        "same_session": True,
        "passed": True,
    }
    return {**payload, "receipt_hash": _canonical_hash(payload)}


def _build_artifacts():
    """Synthetic (but fully valid) registry/manifest/preflights, no model calls."""
    registry = build_empty_overlay_registry()
    manifest = build_baseline_manifest(
        registry, REMOTE_IDENTITY, registry_file="registry.json"
    )
    local_preflight = build_local_preflight(manifest, registry, PROJECT_ROOT)
    source = source_tree_hash(PROJECT_ROOT)
    command_receipt = local_preflight["command_template_receipt"]

    runtime = {
        "source_tree_hash": source,
        "runtime_pins": json.loads(json.dumps(manifest["runtime_pins"])),
        "code_revision": "abc123",
        "worktree_clean": True,
        "worktree_status_hash": "0" * 64,
        "checked_at": "2026-08-14T00:00:00+00:00",
    }

    remote_payload = {
        "schema_version": REMOTE_PREFLIGHT_SCHEMA_VERSION,
        "manifest_hash": manifest["manifest_hash"],
        "registry_hash": registry.registry_hash,
        "local_preflight_hash": local_preflight["preflight_hash"],
        "runtime": runtime,
        "lifecycle_receipt": _lifecycle_receipt(),
        "command_template_receipt": command_receipt,
        "live_runtime_checked": True,
        "live_model_execution_authorized": False,
        "ready_for_authorization": True,
    }
    remote_preflight = {
        **remote_payload,
        "preflight_hash": _canonical_hash(remote_payload),
    }
    return registry, manifest, local_preflight, remote_preflight, source, runtime


ARTIFACTS = _build_artifacts()
_REGISTRY, _MANIFEST, _LOCAL, _REMOTE, _SOURCE, _RUNTIME = ARTIFACTS
_POLICY_DICT = policy_spec_from_treatment(_REGISTRY.treatments[0]).to_dict()
_BUNDLE_ID = _REGISTRY.treatments[0].bundle_id
_SAMPLING_PARAMS = _MANIFEST["runtime_pins"]["sampling"]["parameters"]


def _finalize(payload: dict, field: str) -> dict:
    payload[field] = _canonical_hash(
        {key: value for key, value in payload.items() if key != field}
    )
    return payload


def _make_request(
    result_filename: str = RESULT_FILENAME,
    *,
    result_path: Path | None = None,
) -> dict:
    result = result_path or (PROJECT_ROOT / ".runs" / result_filename)
    return build_authorization_request(
        _MANIFEST,
        _REGISTRY,
        _LOCAL,
        _REMOTE,
        project_root=PROJECT_ROOT,
        result_path=result,
    )


def _make_authorization(
    request: dict,
    *,
    approved_by: str = "test-operator",
    approved_at: str | None = None,
    expires_at: str | None = None,
    expires_seconds: int = 3600,
    authorization_id: str | None = None,
) -> tuple[dict, str]:
    now = datetime.now(timezone.utc)
    payload = {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "authorization_id": authorization_id or f"auth-test-{uuid4().hex[:12]}",
        "screen_id": request["screen_id"],
        "manifest_hash": request["manifest_hash"],
        "registry_hash": request["registry_hash"],
        "local_preflight_hash": request["local_preflight_hash"],
        "remote_preflight_hash": request["remote_preflight_hash"],
        "source_tree_hash": request["source_tree_hash"],
        "remote_identity": request["remote_identity"],
        "result_filename": request["result_filename"],
        "result_path": request["result_path"],
        "max_attempts": EXPECTED_ATTEMPTS,
        "budget": request["budget"],
        "approved_by": approved_by,
        "approved_at": approved_at or now.isoformat(),
        "expires_at": expires_at or (now + timedelta(seconds=expires_seconds)).isoformat(),
        "authorization_statement": AUTHORIZATION_STATEMENT,
        "live_model_execution_authorized": True,
        "single_use": True,
    }
    authorization_hash = _canonical_hash(payload)
    return {**payload, "authorization_hash": authorization_hash}, authorization_hash


def _validate(
    authorization,
    expected_hash,
    *,
    result_filename=RESULT_FILENAME,
    result_path=None,
    source=None,
):
    result = result_path or (PROJECT_ROOT / ".runs" / result_filename)
    return validate_execution_authorization(
        authorization,
        expected_authorization_hash=expected_hash,
        manifest_hash=_MANIFEST["manifest_hash"],
        registry_hash=_REGISTRY.registry_hash,
        local_preflight_hash=_LOCAL["preflight_hash"],
        remote_preflight_hash=_REMOTE["preflight_hash"],
        source_tree_hash=_SOURCE if source is None else source,
        remote_identity=_MANIFEST["remote_identity"],
        result_filename=result_filename,
        result_path=result,
    )


def _trace_entry(
    tool_name: str,
    *,
    is_error: bool = False,
    tool_call_id: str = "",
) -> dict:
    return {
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "is_error": is_error,
        "budget_rejected": False,
        "operation_aborted": False,
        "pre_execution_rejected": False,
        "details": {},
    }


def _make_attempt_item(
    attempt_id: str,
    sampling_seed: int,
    *,
    success: bool = True,
    pi_return_code: int = 0,
    failure_code: str | None = None,
    tool_trace=None,
    pi_stderr: str = "",
    provider_turn_count: int = 1,
    synthetic_assistant_message_count: int = 0,
    provider_request_blocks: int = 0,
    suppressed_tool_request_ids: list[str] | None = None,
) -> dict:
    if tool_trace is None:
        tool_trace = []
    for index, entry in enumerate(tool_trace):
        if not entry.get("tool_call_id"):
            entry["tool_call_id"] = f"call-{index}"
    attempt_ids = [entry["tool_call_id"] for entry in tool_trace]
    admitted_ids = [
        entry["tool_call_id"]
        for entry in tool_trace
        if not entry.get("budget_rejected")
        and not entry.get("operation_aborted")
        and not entry.get("pre_execution_rejected")
    ]
    rejected_ids = [item for item in attempt_ids if item not in set(admitted_ids)]
    suppressed_tool_request_ids = suppressed_tool_request_ids or []
    budget_receipt = {
        "schema_version": _MANIFEST["event_accounting"][
            "budget_receipt_schema_version"
        ],
        "provider_turn_limit": 13,
        "provider_request_admissions": provider_turn_count,
        "provider_request_blocks": provider_request_blocks,
        "provider_gate_checks": provider_turn_count + provider_request_blocks,
        "tool_attempt_limit": 13,
        "tool_attempt_count": len(attempt_ids),
        "tool_attempt_ids": attempt_ids,
        "tool_admission_limit": 12,
        "admitted_tool_call_count": len(admitted_ids),
        "admitted_tool_call_ids": admitted_ids,
        "executed_tool_call_count": len(admitted_ids),
        "executed_tool_call_ids": admitted_ids,
        "pre_admission_rejected_tool_call_count": len(rejected_ids),
        "pre_admission_rejected_tool_call_ids": rejected_ids,
        "suppressed_tool_request_count": len(suppressed_tool_request_ids),
        "suppressed_tool_request_ids": suppressed_tool_request_ids,
        "invariant_violations": [],
    }
    return {
        "attempt_id": attempt_id,
        "policy": _POLICY_DICT,
        "pi_return_code": pi_return_code,
        "pi_stderr": pi_stderr,
        "sampling_receipt": {"seed": sampling_seed, "parameters": _SAMPLING_PARAMS},
        "verification": {
            "success": success,
            "verifier_id": "unbrowser-fixture-nonce",
            "verifier_version": "2",
            "failure_code": None
            if success
            else (failure_code if failure_code is not None else "nonce_mismatch"),
            "diagnostics": {},
        },
        "usage": {"output": 100.0, "input": 50.0},
        "trajectory": {
            "normalizer_schema_version": _MANIFEST["event_accounting"][
                "normalizer_schema_version"
            ],
            "provider_turn_semantics": _MANIFEST["event_accounting"][
                "provider_turn_semantics"
            ],
            "budget_receipt": budget_receipt,
            "assistant_message_count": (
                provider_turn_count + synthetic_assistant_message_count
            ),
            "provider_turn_count": provider_turn_count,
            "synthetic_assistant_message_count": (
                synthetic_assistant_message_count
            ),
            "tool_call_count": len(tool_trace),
            "tool_limit_rejection_count": 0,
            "length_stop_count": 0,
            "stop_reasons": [],
            "planning_preamble": {},
            "tool_trace": tool_trace,
        },
        "timing": {
            "prepare_seconds": 0.1,
            "pi_seconds": 0.2,
            "record_seconds": 0.05,
            "verify_seconds": 0.05,
            "usage_seconds": 0.01,
            "total_seconds": 0.41,
        },
    }


def _one_policy_result(task, attempt, panel, registry_hash, manifest_hash):
    return {
        "task_id": task["task_id"],
        "mode": "treatment_set",
        "execution_order": [_BUNDLE_ID],
        "attempts": {_BUNDLE_ID: attempt},
        "treatment_registry_hash": registry_hash,
        "rollout_replica": panel["rollout_replica"],
        "sampling_seed": panel["sampling_seed"],
        "pilot_manifest_hash": manifest_hash,
        "pilot_panel_id": panel["panel_id"],
    }


def _make_completed_record(panel, task, panel_index, auth_hash, binds) -> dict:
    attempt_id = deterministic_attempt_id(auth_hash, panel["panel_id"])
    attempt = _make_attempt_item(attempt_id, panel["sampling_seed"], success=True)
    result = _one_policy_result(
        task, attempt, panel, binds["registry_hash"], binds["manifest_hash"]
    )
    return {
        "schema_version": PANEL_RESULT_SCHEMA_VERSION,
        "authorization_hash": auth_hash,
        "manifest_hash": binds["manifest_hash"],
        "registry_hash": binds["registry_hash"],
        "local_preflight_hash": binds["local_preflight_hash"],
        "remote_preflight_hash": binds["remote_preflight_hash"],
        "source_tree_hash": binds["source_tree_hash"],
        "panel_id": panel["panel_id"],
        "panel_index": panel_index,
        "task": task,
        "task_commitment_hash": task["task_commitment_hash"],
        "panel": panel,
        "attempt_id": attempt_id,
        "status": "completed",
        "budget": {
            **_budget_reservation(panel_index),
            "consumed": _attempt_budget_consumption(attempt),
        },
        "started_at": "2026-08-14T00:00:00+00:00",
        "finished_at": "2026-08-14T00:00:01+00:00",
        "duration_seconds": 0.5,
        "result": result,
    }


def _make_binds(auth_hash: str) -> dict:
    return _record_binds(
        authorization_hash=auth_hash,
        manifest_hash=_MANIFEST["manifest_hash"],
        registry_hash=_REGISTRY.registry_hash,
        local_preflight_hash=_LOCAL["preflight_hash"],
        remote_preflight_hash=_REMOTE["preflight_hash"],
        source_tree_hash=_SOURCE,
    )


class AuthorizationRequestTest(unittest.TestCase):
    def test_request_is_non_authorizing_and_binds_frozen_artifacts(self) -> None:
        request = _make_request()
        self.assertIs(request["live_model_execution_authorized"], False)
        self.assertEqual(request["manifest_hash"], _MANIFEST["manifest_hash"])
        self.assertEqual(request["registry_hash"], _REGISTRY.registry_hash)
        self.assertEqual(request["local_preflight_hash"], _LOCAL["preflight_hash"])
        self.assertEqual(request["remote_preflight_hash"], _REMOTE["preflight_hash"])
        self.assertEqual(request["source_tree_hash"], _SOURCE)
        self.assertEqual(request["remote_identity"], REMOTE_IDENTITY)
        self.assertEqual(request["result_filename"], RESULT_FILENAME)
        self.assertEqual(
            request["result_path"],
            str((PROJECT_ROOT / ".runs" / RESULT_FILENAME).resolve()),
        )
        self.assertEqual(request["max_attempts"], EXPECTED_ATTEMPTS)
        self.assertEqual(request["budget"], _worst_case_budget())
        self.assertEqual(
            request["budget"]["provider_backed_turns_per_attempt"], 13
        )
        self.assertEqual(request["budget"]["tool_attempts_per_attempt"], 13)
        self.assertEqual(
            request["budget"]["budget_admitted_tool_attempts_per_attempt"], 12
        )
        self.assertEqual(request["budget"]["provider_gate_checks_per_attempt"], 14)
        self.assertEqual(request["budget"]["total_output_tokens"], 72 * 13 * 4096)
        self.assertIn("request_hash", request)

    def test_request_cannot_be_used_as_an_authorization(self) -> None:
        request = _make_request()
        with self.assertRaises(ValueError):
            _validate(request, request["request_hash"])


class AuthorizationValidationTest(unittest.TestCase):
    def test_valid_authorization_is_accepted(self) -> None:
        request = _make_request()
        authorization, authorization_hash = _make_authorization(request)
        self.assertEqual(_validate(authorization, authorization_hash), authorization_hash)

    def test_rejects_live_model_execution_authorized_false(self) -> None:
        request = _make_request()
        authorization, _ = _make_authorization(request)
        authorization = _finalize(
            {**authorization, "live_model_execution_authorized": False},
            "authorization_hash",
        )
        with self.assertRaisesRegex(ValueError, "does not enable live model execution"):
            _validate(authorization, authorization["authorization_hash"])

    def test_rejects_tampered_authorization(self) -> None:
        request = _make_request()
        authorization, authorization_hash = _make_authorization(request)
        tampered = dict(authorization)
        tampered["approved_by"] = "evil-operator"
        with self.assertRaisesRegex(ValueError, "authorization_hash"):
            _validate(tampered, authorization_hash)

    def test_rejects_expected_hash_mismatch(self) -> None:
        request = _make_request()
        authorization, _ = _make_authorization(request)
        with self.assertRaisesRegex(ValueError, "expected hash"):
            _validate(authorization, "f" * 64)

    def test_rejects_expired_authorization(self) -> None:
        request = _make_request()
        now = datetime.now(timezone.utc)
        authorization, authorization_hash = _make_authorization(
            request,
            approved_at=(now - timedelta(hours=2)).isoformat(),
            expires_at=(now - timedelta(hours=1)).isoformat(),
        )
        with self.assertRaisesRegex(ValueError, "expired"):
            _validate(authorization, authorization_hash)

    def test_rejects_expires_before_approved(self) -> None:
        request = _make_request()
        now = datetime.now(timezone.utc)
        authorization, authorization_hash = _make_authorization(
            request,
            approved_at=(now - timedelta(hours=1)).isoformat(),
            expires_at=(now - timedelta(hours=2)).isoformat(),
        )
        with self.assertRaisesRegex(ValueError, "after approved_at"):
            _validate(authorization, authorization_hash)

    def test_rejects_source_drift(self) -> None:
        request = _make_request()
        authorization, authorization_hash = _make_authorization(request)
        with self.assertRaisesRegex(ValueError, "source"):
            _validate(authorization, authorization_hash, source="f" * 64)

    def test_rejects_budget_mismatch(self) -> None:
        request = _make_request()
        authorization, _ = _make_authorization(request)
        authorization = _finalize(
            {**authorization, "budget": {"max_attempts": 1}}, "authorization_hash"
        )
        with self.assertRaisesRegex(ValueError, "budget"):
            _validate(authorization, authorization["authorization_hash"])

    def test_rejects_max_attempts_mismatch(self) -> None:
        request = _make_request()
        authorization, _ = _make_authorization(request)
        authorization = _finalize(
            {**authorization, "max_attempts": EXPECTED_ATTEMPTS - 1},
            "authorization_hash",
        )
        with self.assertRaisesRegex(ValueError, "max_attempts"):
            _validate(authorization, authorization["authorization_hash"])

    def test_rejects_unknown_authorization_field(self) -> None:
        request = _make_request()
        authorization, _ = _make_authorization(request)
        authorization = _finalize(
            {**authorization, "unexpected_override": True}, "authorization_hash"
        )
        with self.assertRaisesRegex(ValueError, "fields mismatch"):
            _validate(authorization, authorization["authorization_hash"])

    def test_rejects_future_approval(self) -> None:
        request = _make_request()
        now = datetime.now(timezone.utc)
        authorization, authorization_hash = _make_authorization(
            request,
            approved_at=(now + timedelta(hours=1)).isoformat(),
            expires_at=(now + timedelta(hours=2)).isoformat(),
        )
        with self.assertRaisesRegex(ValueError, "future"):
            _validate(authorization, authorization_hash)

    def test_rejects_result_filename_mismatch(self) -> None:
        request = _make_request()
        authorization, authorization_hash = _make_authorization(request)
        with self.assertRaisesRegex(ValueError, "filename"):
            _validate(authorization, authorization_hash, result_filename="other.jsonl")

    def test_rejects_same_filename_at_a_different_result_path(self) -> None:
        request = _make_request()
        authorization, authorization_hash = _make_authorization(request)
        with self.assertRaisesRegex(ValueError, "result path"):
            _validate(
                authorization,
                authorization_hash,
                result_path=PROJECT_ROOT / "other" / RESULT_FILENAME,
            )

    def test_rejects_single_use_false(self) -> None:
        request = _make_request()
        authorization, _ = _make_authorization(request)
        authorization = _finalize({**authorization, "single_use": False}, "authorization_hash")
        with self.assertRaisesRegex(ValueError, "single_use"):
            _validate(authorization, authorization["authorization_hash"])

    def test_rejects_empty_approved_by(self) -> None:
        request = _make_request()
        authorization, _ = _make_authorization(request)
        authorization = _finalize({**authorization, "approved_by": "  "}, "authorization_hash")
        with self.assertRaisesRegex(ValueError, "approved_by"):
            _validate(authorization, authorization["authorization_hash"])

    def test_rejects_statement_mismatch(self) -> None:
        request = _make_request()
        authorization, _ = _make_authorization(request)
        authorization = _finalize(
            {**authorization, "authorization_statement": "not the statement"},
            "authorization_hash",
        )
        with self.assertRaisesRegex(ValueError, "statement"):
            _validate(authorization, authorization["authorization_hash"])


class DeterministicAttemptIdTest(unittest.TestCase):
    def test_attempt_ids_are_deterministic_safe_and_distinct(self) -> None:
        first = deterministic_attempt_id("a" * 64, "panel-1")
        self.assertEqual(first, deterministic_attempt_id("a" * 64, "panel-1"))
        self.assertNotEqual(first, deterministic_attempt_id("b" * 64, "panel-1"))
        self.assertNotEqual(first, deterministic_attempt_id("a" * 64, "panel-2"))
        self.assertRegex(first, r"^[A-Za-z0-9][A-Za-z0-9_.@-]{0,127}$")


class AttemptClassificationTest(unittest.TestCase):
    def _classify(self, item) -> tuple:
        return _classify_attempt(
            item,
            expected_policy=_POLICY_DICT,
            expected_sampling_receipt={
                "seed": item["sampling_receipt"]["seed"],
                "parameters": _SAMPLING_PARAMS,
            },
            runtime_pins=_MANIFEST["runtime_pins"],
        )

    def test_ordinary_verifier_failure_is_completed(self) -> None:
        item = _make_attempt_item("attempt-1", 1, success=False, failure_code="nonce_mismatch")
        status, reason = self._classify(item)
        self.assertEqual(status, "completed")
        self.assertIsNone(reason)

    def test_verifier_substrate_failure_is_infrastructure(self) -> None:
        for code in (
            "task_not_found",
            "attempt_not_found",
            "attempt_task_mismatch",
            "oracle_unreadable",
            "oracle_missing_nonce",
            "oracle_commitment_mismatch",
        ):
            item = _make_attempt_item("attempt-1", 1, success=False, failure_code=code)
            status, reason = self._classify(item)
            self.assertEqual(status, "infrastructure_invalid", code)
            self.assertIn("verifier_substrate", reason)

    def test_embedded_browser_infra_marker_is_detected(self) -> None:
        trace = [_trace_entry("unbrowser")]
        trace.append(
            {
                "tool_name": "unbrowser",
                "is_error": True,
                "budget_rejected": False,
                "operation_aborted": False,
                "pre_execution_rejected": False,
                "details": {"infrastructure_error": True},
            }
        )
        item = _make_attempt_item("attempt-1", 1, success=True, tool_trace=trace)
        status, reason = self._classify(item)
        self.assertEqual(status, "infrastructure_invalid")
        self.assertIn("browser_infrastructure_marker", reason)

    def test_pi_return_minus_one_is_infrastructure_even_with_false_verification(self) -> None:
        item = _make_attempt_item(
            "attempt-1", 1, success=False, pi_return_code=-1, failure_code="nonce_mismatch"
        )
        status, reason = self._classify(item)
        self.assertEqual(status, "infrastructure_invalid")
        self.assertEqual(reason, "ambiguous_wall_timeout")

    def test_pi_return_minus_one_with_true_verification_is_infra(self) -> None:
        item = _make_attempt_item("attempt-1", 1, success=True, pi_return_code=-1)
        status, reason = self._classify(item)
        self.assertEqual(status, "infrastructure_invalid")
        self.assertEqual(reason, "ambiguous_wall_timeout")

    def test_pi_return_code_out_of_bounds_is_infrastructure(self) -> None:
        item = _make_attempt_item("attempt-1", 1, success=True, pi_return_code=2)
        status, reason = self._classify(item)
        self.assertEqual(status, "infrastructure_invalid")
        self.assertIn("pi_return_code=2", reason)

    def test_provider_transport_marker_is_infrastructure(self) -> None:
        item = _make_attempt_item("attempt-1", 1, success=True, pi_stderr="Connection refused")
        status, reason = self._classify(item)
        self.assertEqual(status, "infrastructure_invalid")
        self.assertIn("provider_transport", reason)

    def test_malformed_normalized_events_is_infrastructure(self) -> None:
        item = _make_attempt_item("attempt-1", 1, success=True)
        del item["usage"]
        status, reason = self._classify(item)
        self.assertEqual(status, "infrastructure_invalid")
        self.assertIn("malformed_normalized_events", reason)

    def test_missing_sampling_receipt_is_infrastructure(self) -> None:
        item = _make_attempt_item("attempt-1", 1, success=False)
        item["sampling_receipt"] = None
        status, reason = _classify_attempt(
            item,
            expected_policy=_POLICY_DICT,
            expected_sampling_receipt={"seed": 1, "parameters": _SAMPLING_PARAMS},
            runtime_pins=_MANIFEST["runtime_pins"],
        )
        self.assertEqual(status, "infrastructure_invalid")
        self.assertIn("sampling receipt", reason)

    def test_zero_provider_turns_is_infrastructure(self) -> None:
        item = _make_attempt_item(
            "attempt-1", 1, success=False, provider_turn_count=0
        )
        status, reason = self._classify(item)
        self.assertEqual(status, "infrastructure_invalid")
        self.assertIn("provider_turn_count", reason)

    def test_exact_budget_boundary_with_terminal_abort_is_completed(self) -> None:
        trace = [_trace_entry("unbrowser") for _ in range(12)]
        terminal_abort = _trace_entry("unbrowser", is_error=True)
        terminal_abort["operation_aborted"] = True
        trace.append(terminal_abort)
        item = _make_attempt_item(
            "attempt-1",
            1,
            success=False,
            tool_trace=trace,
            provider_turn_count=13,
            synthetic_assistant_message_count=1,
        )
        status, reason = self._classify(item)
        self.assertEqual(status, "completed")
        self.assertIsNone(reason)
        self.assertEqual(
            _attempt_budget_consumption(item),
            {
                "model_attempts": 1,
                "provider_backed_turns": 13,
                "provider_gate_checks": 13,
                "provider_gate_blocks": 0,
                "output_tokens": 100.0,
                "tool_attempts": 13,
                "budget_admitted_tool_attempts": 12,
                "executed_tool_calls": 12,
                "rejected_tool_attempts": 1,
                "budget_blocked_tool_attempts": 1,
                "suppressed_tool_requests": 0,
                "model_wall_seconds": 0.2,
            },
        )

    def test_schema_rejection_consumes_turn_before_provider_gate_stop(self) -> None:
        schema_rejection = _trace_entry("unbrowser", is_error=True)
        schema_rejection["pre_execution_rejected"] = True
        trace = [schema_rejection]
        trace.extend(_trace_entry("unbrowser") for _ in range(12))
        item = _make_attempt_item(
            "attempt-1",
            1,
            success=False,
            tool_trace=trace,
            provider_turn_count=13,
            synthetic_assistant_message_count=1,
            provider_request_blocks=1,
        )
        status, reason = self._classify(item)
        self.assertEqual(status, "completed")
        self.assertIsNone(reason)
        consumed = _attempt_budget_consumption(item)
        self.assertEqual(consumed["provider_backed_turns"], 13)
        self.assertEqual(consumed["provider_gate_checks"], 14)
        self.assertEqual(consumed["provider_gate_blocks"], 1)
        self.assertEqual(consumed["tool_attempts"], 13)
        self.assertEqual(consumed["budget_admitted_tool_attempts"], 12)
        self.assertEqual(consumed["executed_tool_calls"], 12)
        self.assertEqual(consumed["rejected_tool_attempts"], 1)

    def test_budget_receipt_invariant_violation_is_infrastructure(self) -> None:
        item = _make_attempt_item("attempt-1", 1)
        item["trajectory"]["budget_receipt"]["invariant_violations"] = [
            "tool_attempt_limit_bypassed"
        ]
        status, reason = self._classify(item)
        self.assertEqual(status, "infrastructure_invalid")
        self.assertIn("budget_receipt invariant", reason)

    def test_oversized_parallel_batch_is_a_terminal_budget_outcome(self) -> None:
        terminal_abort = _trace_entry("bash", is_error=True)
        terminal_abort["operation_aborted"] = True
        item = _make_attempt_item(
            "attempt-1",
            1,
            success=False,
            tool_trace=[terminal_abort],
            provider_turn_count=1,
            synthetic_assistant_message_count=1,
            suppressed_tool_request_ids=[f"suppressed-{index}" for index in range(13)],
        )
        status, reason = self._classify(item)
        self.assertEqual(status, "completed")
        self.assertIsNone(reason)

    def test_parallel_completion_order_reconciles_by_tool_identity(self) -> None:
        trace = [
            _trace_entry("bash", tool_call_id="fast"),
            _trace_entry("bash", tool_call_id="slow"),
        ]
        item = _make_attempt_item(
            "attempt-1",
            1,
            tool_trace=trace,
            provider_turn_count=2,
        )
        receipt = item["trajectory"]["budget_receipt"]
        receipt["tool_attempt_ids"] = ["slow", "fast"]
        receipt["admitted_tool_call_ids"] = ["slow", "fast"]
        status, reason = self._classify(item)
        self.assertEqual(status, "completed")
        self.assertIsNone(reason)

    def test_nonterminal_operation_abort_is_infrastructure(self) -> None:
        trace = [_trace_entry("unbrowser") for _ in range(11)]
        aborted = _trace_entry("unbrowser", is_error=True)
        aborted["operation_aborted"] = True
        trace.extend([aborted, _trace_entry("unbrowser")])
        item = _make_attempt_item(
            "attempt-1", 1, success=False, tool_trace=trace, provider_turn_count=13
        )
        status, reason = self._classify(item)
        self.assertEqual(status, "infrastructure_invalid")
        self.assertIn("budget-blocked", reason)

    def test_assistant_message_accounting_mismatch_is_infrastructure(self) -> None:
        item = _make_attempt_item("attempt-1", 1)
        item["trajectory"]["assistant_message_count"] = 2
        status, reason = self._classify(item)
        self.assertEqual(status, "infrastructure_invalid")
        self.assertIn("assistant-message accounting", reason)

    def test_synthetic_abort_without_blocked_tool_is_infrastructure(self) -> None:
        item = _make_attempt_item(
            "attempt-1", 1, synthetic_assistant_message_count=1
        )
        status, reason = self._classify(item)
        self.assertEqual(status, "infrastructure_invalid")
        self.assertIn("synthetic assistant", reason)


class RecordValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.auth_hash = "a" * 64
        self.binds = _make_binds(self.auth_hash)
        self.panel = _MANIFEST["panels"][0]
        self.task = next(
            t for t in _MANIFEST["tasks"] if t["task_id"] == self.panel["task_id"]
        )

    def _record(self) -> dict:
        return _make_completed_record(self.panel, self.task, 0, self.auth_hash, self.binds)

    def _write(self, path: Path, *records: dict) -> None:
        path.write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in records),
            encoding="utf-8",
        )

    def test_valid_record_passes_strict_validation(self) -> None:
        record = _finalize(self._record(), "record_hash")
        self.assertEqual(
            _validate_record(
                record,
                binds=self.binds,
                manifest=_MANIFEST,
                registry_hash=_REGISTRY.registry_hash,
            ),
            [],
        )

    def test_tampered_record_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            self._write(path, _finalize(self._record(), "record_hash"))
            tampered = dict(self._record())
            tampered["duration_seconds"] = 9999.0
            self._write(path, tampered)
            with self.assertRaisesRegex(ValueError, "record_hash"):
                _load_ledger(
                    path,
                    binds=self.binds,
                    manifest=_MANIFEST,
                    registry_hash=_REGISTRY.registry_hash,
                )

    def test_duplicate_panel_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            self._write(
                path,
                _finalize(self._record(), "record_hash"),
                _finalize(self._record(), "record_hash"),
            )
            with self.assertRaisesRegex(ValueError, "duplicate panel"):
                _load_ledger(
                    path,
                    binds=self.binds,
                    manifest=_MANIFEST,
                    registry_hash=_REGISTRY.registry_hash,
                )

    def test_duplicate_attempt_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            first = _finalize(self._record(), "record_hash")
            second_panel = _MANIFEST["panels"][1]
            second_task = next(
                t for t in _MANIFEST["tasks"] if t["task_id"] == second_panel["task_id"]
            )
            second = _make_completed_record(
                second_panel, second_task, 1, self.auth_hash, self.binds
            )
            second["attempt_id"] = first["attempt_id"]
            second["result"]["attempts"][_BUNDLE_ID]["attempt_id"] = first["attempt_id"]
            second = _finalize(second, "record_hash")
            self._write(path, first, second)
            with self.assertRaisesRegex(ValueError, "duplicate attempt id"):
                _load_ledger(
                    path,
                    binds=self.binds,
                    manifest=_MANIFEST,
                    registry_hash=_REGISTRY.registry_hash,
                )

    def test_mixed_manifest_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            record = self._record()
            record["manifest_hash"] = "f" * 64
            self._write(path, _finalize(record, "record_hash"))
            with self.assertRaisesRegex(ValueError, "manifest_hash"):
                _load_ledger(
                    path,
                    binds=self.binds,
                    manifest=_MANIFEST,
                    registry_hash=_REGISTRY.registry_hash,
                )

    def test_result_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            record = self._record()
            record["result"] = dict(record["result"])
            record["result"]["mode"] = "pair"
            self._write(path, _finalize(record, "record_hash"))
            with self.assertRaisesRegex(ValueError, "mode"):
                _load_ledger(
                    path,
                    binds=self.binds,
                    manifest=_MANIFEST,
                    registry_hash=_REGISTRY.registry_hash,
                )

    def test_non_deterministic_attempt_id_is_rejected(self) -> None:
        record = self._record()
        record["attempt_id"] = "different-safe-id"
        record["result"]["attempts"][_BUNDLE_ID]["attempt_id"] = (
            "different-safe-id"
        )
        record = _finalize(record, "record_hash")
        errors = _validate_record(
            record,
            binds=self.binds,
            manifest=_MANIFEST,
            registry_hash=_REGISTRY.registry_hash,
        )
        self.assertTrue(any("deterministic" in error for error in errors))

    def test_non_prefix_ledger_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            panel = _MANIFEST["panels"][1]
            task = next(
                t for t in _MANIFEST["tasks"] if t["task_id"] == panel["task_id"]
            )
            self._write(
                path,
                _finalize(
                    _make_completed_record(
                        panel, task, 1, self.auth_hash, self.binds
                    ),
                    "record_hash",
                ),
            )
            with self.assertRaisesRegex(ValueError, "manifest-order prefix"):
                _load_ledger(
                    path,
                    binds=self.binds,
                    manifest=_MANIFEST,
                    registry_hash=_REGISTRY.registry_hash,
                )

    def test_infrastructure_invalid_record_blocks_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            record = self._record()
            record["status"] = "infrastructure_invalid"
            record.pop("result", None)
            record["error"] = {
                "type": "AttemptExecutionError",
                "message": "boom",
                "error_class": "infrastructure_invalid",
                "error_code": "controller_error",
                "phase": "test",
                "attempt_id": record["attempt_id"],
            }
            self._write(path, _finalize(record, "record_hash"))
            with self.assertRaisesRegex(RuntimeError, "infrastructure_invalid"):
                _existing_completed_records(
                    path,
                    binds=self.binds,
                    manifest=_MANIFEST,
                    registry_hash=_REGISTRY.registry_hash,
                )


class RunnerGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()
        self.result_path = self.tmp / RESULT_FILENAME
        self.registry_path = self.tmp / "registry.json"
        self.manifest_path = self.tmp / "manifest.json"
        self.local_path = self.tmp / "local.json"
        self.remote_path = self.tmp / "remote.json"
        self.auth_path = self.tmp / "authorization.json"
        _REGISTRY.save(self.registry_path)
        self.manifest_path.write_text(json.dumps(_MANIFEST), encoding="utf-8")
        self.local_path.write_text(json.dumps(_LOCAL), encoding="utf-8")
        self.remote_path.write_text(json.dumps(_REMOTE), encoding="utf-8")
        self.request = _make_request(result_path=self.result_path)
        self.authorization, self.authorization_hash = _make_authorization(self.request)
        self.auth_path.write_text(json.dumps(self.authorization), encoding="utf-8")
        self.config = RemoteConfig(**REMOTE_IDENTITY)
        self.runtime = _RUNTIME

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run_kwargs(self) -> dict:
        return dict(
            manifest_path=self.manifest_path,
            registry_path=self.registry_path,
            local_preflight_path=self.local_path,
            remote_preflight_path=self.remote_path,
            authorization_path=self.auth_path,
            expected_authorization_hash=self.authorization_hash,
            result_path=self.result_path,
            config=self.config,
            pi_binary="pi",
            provider="ubuntu-gemma",
            model="gemma-4-26b-a4b",
            thinking="off",
            unbrowser_binary="/unbrowser",
            model_artifact="/model.gguf",
            llama_server_binary="/llama-server",
        )

    def _invoke(
        self,
        attempt_side_effect,
        task_side_effect=None,
        commitment_side_effect=None,
    ):
        """Patch the three low-level functions and run the baseline runner."""

        def fake_task(config, args):
            if task_side_effect is not None:
                task_side_effect(config, args)
            task_id = (
                f"{args.fixture_generator_version}-{args.fixture_template}-"
                f"{args.difficulty}-{args.seed}"
            )
            expected = next(
                item["task"]
                for item in _LOCAL["generated_tasks"]
                if item["task"]["id"] == task_id
            )
            return {
                **expected,
                "workspace_ref": "/ws",
                "verifier_ref": "/oracle",
            }

        def fake_commitment(config, arguments, **kwargs):
            task_id = arguments[arguments.index("--task-id") + 1]
            commitment = next(
                item
                for item in _LOCAL["generated_tasks"]
                if item["task"]["id"] == task_id
            )
            if commitment_side_effect is not None:
                return commitment_side_effect(commitment)
            return commitment

        with mock.patch(
            "pyreplab_harness.m3_empty_overlay_execution.runtime_preflight",
            return_value=self.runtime,
        ), mock.patch(
            "pyreplab_harness.m3_empty_overlay_execution._task_json",
            side_effect=fake_task,
        ), mock.patch(
            "pyreplab_harness.m3_empty_overlay_execution.remote_json",
            side_effect=fake_commitment,
        ), mock.patch(
            "pyreplab_harness.m3_empty_overlay_execution._run_attempt",
            side_effect=attempt_side_effect,
        ):
            return run_authorized_baseline(**self._run_kwargs())

    def test_existing_claim_without_ledger_fails_closed(self) -> None:
        _write_claim(
            self.tmp / "baseline.jsonl.claim.json",
            self.authorization_hash,
            self.result_path,
            RESULT_FILENAME,
        )
        with mock.patch(
            "pyreplab_harness.m3_empty_overlay_execution.runtime_preflight",
            return_value=self.runtime,
        ):
            with self.assertRaisesRegex(RuntimeError, "no ledger"):
                run_authorized_baseline(**self._run_kwargs())

    def test_active_marker_fails_closed(self) -> None:
        (self.tmp / "baseline.jsonl.active.json").write_text(
            json.dumps({"panel_id": "stale"}), encoding="utf-8"
        )
        with mock.patch(
            "pyreplab_harness.m3_empty_overlay_execution.runtime_preflight",
            return_value=self.runtime,
        ):
            with self.assertRaisesRegex(RuntimeError, "active panel marker"):
                run_authorized_baseline(**self._run_kwargs())

    def test_lock_contention_fails_closed(self) -> None:
        (self.tmp / "baseline.jsonl.lock").write_text("locked", encoding="utf-8")
        with mock.patch(
            "pyreplab_harness.m3_empty_overlay_execution.runtime_preflight",
            return_value=self.runtime,
        ):
            with self.assertRaisesRegex(RuntimeError, "lock"):
                run_authorized_baseline(**self._run_kwargs())

    def test_detached_launch_writes_immutable_process_receipt(self) -> None:
        process = mock.Mock(pid=4321)
        process.poll.return_value = None

        def spawn(*args, **kwargs):
            _write_claim(
                self.tmp / "baseline.jsonl.claim.json",
                self.authorization_hash,
                self.result_path,
                RESULT_FILENAME,
                controller_pid=4321,
            )
            return process

        with mock.patch(
            "pyreplab_harness.m3_empty_overlay_execution.subprocess.Popen",
            side_effect=spawn,
        ) as popen:
            receipt = launch_authorized_baseline_detached(**self._run_kwargs())

        launch_path = self.tmp / "baseline.jsonl.launch.json"
        log_path = self.tmp / "baseline.jsonl.controller.log"
        self.assertTrue(launch_path.is_file())
        self.assertTrue(log_path.is_file())
        self.assertFalse((self.tmp / "baseline.jsonl.launch.lock").exists())
        self.assertEqual(receipt["controller_pid"], 4321)
        self.assertEqual(receipt["controller_process_group"], 4321)
        self.assertEqual(receipt["startup_state"], "claim_observed")
        self.assertIsNone(receipt["controller_return_code"])
        self.assertTrue(receipt["detached_session"])
        self.assertEqual(
            receipt["launch_hash"],
            _canonical_hash(
                {key: value for key, value in receipt.items() if key != "launch_hash"}
            ),
        )
        command = popen.call_args.args[0]
        self.assertEqual(command[3], "run")
        self.assertEqual(
            command[command.index("--authorization-hash") + 1],
            self.authorization_hash,
        )
        self.assertEqual(
            command[command.index("--result") + 1], str(self.result_path)
        )
        self.assertIs(popen.call_args.kwargs["stdin"], subprocess.DEVNULL)
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertTrue(popen.call_args.kwargs["close_fds"])

        with self.assertRaisesRegex(RuntimeError, "fresh result path"):
            launch_authorized_baseline_detached(**self._run_kwargs())

    def test_detached_launch_spawn_failure_releases_reservation(self) -> None:
        with mock.patch(
            "pyreplab_harness.m3_empty_overlay_execution.subprocess.Popen",
            side_effect=OSError("spawn failed"),
        ):
            with self.assertRaisesRegex(OSError, "spawn failed"):
                launch_authorized_baseline_detached(**self._run_kwargs())

        self.assertFalse((self.tmp / "baseline.jsonl.launch.lock").exists())
        self.assertFalse((self.tmp / "baseline.jsonl.controller.log").exists())
        self.assertFalse((self.tmp / "baseline.jsonl.launch.json").exists())

    def test_detached_launch_records_exit_before_claim(self) -> None:
        process = mock.Mock(pid=4321)
        process.poll.return_value = 2
        with mock.patch(
            "pyreplab_harness.m3_empty_overlay_execution.subprocess.Popen",
            return_value=process,
        ):
            with self.assertRaisesRegex(RuntimeError, "before claiming"):
                launch_authorized_baseline_detached(**self._run_kwargs())

        receipt = json.loads(
            (self.tmp / "baseline.jsonl.launch.json").read_text(encoding="utf-8")
        )
        self.assertEqual(receipt["startup_state"], "exited_before_claim")
        self.assertEqual(receipt["controller_return_code"], 2)

    def test_detached_launch_rejects_existing_runner_lock(self) -> None:
        (self.tmp / "baseline.jsonl.lock").write_text("locked\n", encoding="utf-8")
        with mock.patch(
            "pyreplab_harness.m3_empty_overlay_execution.subprocess.Popen"
        ) as popen:
            with self.assertRaisesRegex(RuntimeError, "fresh result path"):
                launch_authorized_baseline_detached(**self._run_kwargs())
        popen.assert_not_called()

    def test_run_proves_v3_one_panel_then_interruption(self) -> None:
        state = {"generator_versions": [], "attempts": 0}

        def fake_attempt(project_root, config, task, policy, attempt_id, args, **kwargs):
            state["attempts"] += 1
            if state["attempts"] > 1:
                raise KeyboardInterrupt()
            return _make_attempt_item(attempt_id, args.sampling_seed, success=True)

        def record_generator_version(config, args):
            state["generator_versions"].append(args.fixture_generator_version)

        with self.assertRaises(KeyboardInterrupt):
            self._invoke(fake_attempt, task_side_effect=record_generator_version)

        self.assertTrue(state["generator_versions"])
        self.assertTrue(
            all(v == "unbrowser-fixture-v3" for v in state["generator_versions"])
        )
        ledger = [
            line
            for line in self.result_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(ledger), 1)
        self.assertTrue((self.tmp / "baseline.jsonl.active.json").exists())
        self.assertTrue((self.tmp / "baseline.jsonl.claim.json").exists())
        self.assertFalse((self.tmp / "baseline.jsonl.receipt.json").exists())

    def test_run_stops_on_verifier_substrate_failure(self) -> None:
        def fake_attempt(project_root, config, task, policy, attempt_id, args, **kwargs):
            return _make_attempt_item(
                attempt_id, args.sampling_seed, success=False, failure_code="task_not_found"
            )

        with self.assertRaisesRegex(RuntimeError, "infrastructure error"):
            self._invoke(fake_attempt)
        record = json.loads(self.result_path.read_text(encoding="utf-8").splitlines()[-1])
        self.assertEqual(record["status"], "infrastructure_invalid")

    def test_run_rechecks_authorization_expiry_before_each_model_admission(self) -> None:
        calls = {"attempts": 0}

        def fake_attempt(project_root, config, task, policy, attempt_id, args, **kwargs):
            calls["attempts"] += 1
            return _make_attempt_item(attempt_id, args.sampling_seed, success=True)

        with mock.patch(
            "pyreplab_harness.m3_empty_overlay_execution._require_authorization_active",
            side_effect=[None, None, RuntimeError("expired")],
        ):
            with self.assertRaisesRegex(RuntimeError, "expired"):
                self._invoke(fake_attempt)

        self.assertEqual(calls["attempts"], 1)
        self.assertEqual(
            len(self.result_path.read_text(encoding="utf-8").splitlines()), 1
        )
        self.assertFalse((self.tmp / "baseline.jsonl.active.json").exists())

    def test_run_rejects_poisoned_remote_oracle_before_model_admission(self) -> None:
        calls = {"attempts": 0}

        def fake_attempt(project_root, config, task, policy, attempt_id, args, **kwargs):
            calls["attempts"] += 1
            return _make_attempt_item(attempt_id, args.sampling_seed, success=True)

        def poison(commitment):
            return {**commitment, "oracle_sha256": "f" * 64}

        with self.assertRaisesRegex(RuntimeError, "infrastructure error"):
            self._invoke(fake_attempt, commitment_side_effect=poison)
        self.assertEqual(calls["attempts"], 0)
        record = json.loads(
            self.result_path.read_text(encoding="utf-8").splitlines()[-1]
        )
        self.assertEqual(record["status"], "infrastructure_invalid")
        self.assertIn("oracle drifted", record["error"]["message"])

    def test_resume_skips_completed_and_writes_receipt(self) -> None:
        binds = _make_binds(self.authorization_hash)
        task_by_id = {t["task_id"]: t for t in _MANIFEST["tasks"]}
        _write_claim(
            self.tmp / "baseline.jsonl.claim.json",
            self.authorization_hash,
            self.result_path,
            RESULT_FILENAME,
        )
        for index, panel in enumerate(_MANIFEST["panels"][: EXPECTED_ATTEMPTS - 1]):
            task = task_by_id[panel["task_id"]]
            _append_record(
                self.result_path,
                _make_completed_record(panel, task, index, self.authorization_hash, binds),
            )

        calls = {"attempts": 0}

        def fake_attempt(project_root, config, task, policy, attempt_id, args, **kwargs):
            calls["attempts"] += 1
            return _make_attempt_item(attempt_id, args.sampling_seed, success=True)

        report = self._invoke(fake_attempt)
        self.assertEqual(calls["attempts"], 1)
        self.assertEqual(report["panels_run"], 1)
        self.assertEqual(report["panels_skipped"], EXPECTED_ATTEMPTS - 1)
        self.assertIsNotNone(report["completion_receipt"])
        self.assertTrue((self.tmp / "baseline.jsonl.receipt.json").exists())
        ledger = [
            line
            for line in self.result_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(ledger), EXPECTED_ATTEMPTS)


class AnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()
        self.result_path = self.tmp / RESULT_FILENAME
        self.registry_path = self.tmp / "registry.json"
        self.manifest_path = self.tmp / "manifest.json"
        self.local_path = self.tmp / "local.json"
        self.remote_path = self.tmp / "remote.json"
        _REGISTRY.save(self.registry_path)
        self.manifest_path.write_text(json.dumps(_MANIFEST), encoding="utf-8")
        self.local_path.write_text(json.dumps(_LOCAL), encoding="utf-8")
        self.remote_path.write_text(json.dumps(_REMOTE), encoding="utf-8")
        self.auth_hash = "a" * 64
        self.binds = _make_binds(self.auth_hash)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _failure_trace(self, template: str, index: int, replica: int) -> list:
        if template == "table_filter_sort" and index == 0:
            if replica == 0:
                return [
                    _trace_entry("unbrowser"),
                    _trace_entry("unbrowser"),
                    _trace_entry("bash"),
                ]
            return [_trace_entry("unbrowser"), _trace_entry("bash")]
        return [_trace_entry("unbrowser")]

    def _write_synthetic_ledger(self) -> None:
        task_by_id = {t["task_id"]: t for t in _MANIFEST["tasks"]}
        template_task_ids = {}
        for task in _MANIFEST["tasks"]:
            template_task_ids.setdefault(task["template"], []).append(task["task_id"])
        task_index_by_id = {}
        for template, ids in template_task_ids.items():
            for index, task_id in enumerate(ids):
                task_index_by_id[task_id] = index

        for panel_index, panel in enumerate(_MANIFEST["panels"]):
            task = task_by_id[panel["task_id"]]
            template = task["template"]
            index = task_index_by_id[task["task_id"]]
            outcome = TEMPLATE_OUTCOMES[template][index]
            replica = panel["rollout_replica"]
            if outcome == "both_success":
                success = True
            elif outcome == "both_fail":
                success = False
            else:
                success = replica == 0
            attempt_id = deterministic_attempt_id(self.auth_hash, panel["panel_id"])
            trace = [] if success else self._failure_trace(template, index, replica)
            attempt = _make_attempt_item(
                attempt_id, panel["sampling_seed"], success=success, tool_trace=trace
            )
            result = _one_policy_result(
                task,
                attempt,
                panel,
                self.binds["registry_hash"],
                self.binds["manifest_hash"],
            )
            record = {
                "schema_version": PANEL_RESULT_SCHEMA_VERSION,
                "authorization_hash": self.auth_hash,
                "manifest_hash": self.binds["manifest_hash"],
                "registry_hash": self.binds["registry_hash"],
                "local_preflight_hash": self.binds["local_preflight_hash"],
                "remote_preflight_hash": self.binds["remote_preflight_hash"],
                "source_tree_hash": self.binds["source_tree_hash"],
                "panel_id": panel["panel_id"],
                "panel_index": panel_index,
                "task": task,
                "task_commitment_hash": task["task_commitment_hash"],
                "panel": panel,
                "attempt_id": attempt_id,
                "status": "completed",
                "budget": {
                    **_budget_reservation(panel_index),
                    "consumed": _attempt_budget_consumption(attempt),
                },
                "started_at": "2026-08-14T00:00:00+00:00",
                "finished_at": "2026-08-14T00:00:01+00:00",
                "duration_seconds": 0.5,
                "result": result,
            }
            _append_record(self.result_path, record)

    def _write_receipt(self) -> None:
        _write_completion_receipt(
            self.tmp / "baseline.jsonl.receipt.json",
            binds=self.binds,
            result_filename=RESULT_FILENAME,
            ledger_sha256=_sha256_file(self.result_path),
            record_count=EXPECTED_ATTEMPTS,
        )

    def test_analysis_requires_receipt(self) -> None:
        with self.assertRaisesRegex(ValueError, "receipt"):
            analyze_baseline_results(
                self.manifest_path,
                self.registry_path,
                self.local_path,
                self.remote_path,
                self.result_path,
            )

    def test_analysis_rejects_incomplete_ledger(self) -> None:
        task_by_id = {t["task_id"]: t for t in _MANIFEST["tasks"]}
        for index, panel in enumerate(_MANIFEST["panels"][: EXPECTED_ATTEMPTS - 1]):
            task = task_by_id[panel["task_id"]]
            _append_record(
                self.result_path,
                _make_completed_record(panel, task, index, self.auth_hash, self.binds),
            )
        self._write_receipt()
        with self.assertRaisesRegex(ValueError, "exactly"):
            analyze_baseline_results(
                self.manifest_path,
                self.registry_path,
                self.local_path,
                self.remote_path,
                self.result_path,
            )

    def test_analysis_on_complete_72_records(self) -> None:
        self._write_synthetic_ledger()
        self._write_receipt()
        analysis = analyze_baseline_results(
            self.manifest_path,
            self.registry_path,
            self.local_path,
            self.remote_path,
            self.result_path,
        )

        overall = analysis["overall"]
        self.assertEqual(overall["attempts"], EXPECTED_ATTEMPTS)
        self.assertEqual(overall["successes"], 41)
        self.assertAlmostEqual(overall["success_rate"], 41 / 72, places=9)
        self.assertGreater(overall["wilson_95_upper"], overall["wilson_95_lower"])
        self.assertEqual(overall["replica_agreement"]["total_tasks"], 36)

        by_template = {entry["template"]: entry for entry in analysis["by_template"]}
        self.assertEqual(by_template["single_page_extraction"]["classification"], "ceiling")
        self.assertEqual(by_template["table_filter_sort"]["classification"], "floor_risk")
        self.assertEqual(
            by_template["multi_page_navigation"]["classification"], "unstable"
        )
        self.assertEqual(
            by_template["search_filter_controls"]["classification"], "challenge_candidate"
        )
        self.assertEqual(
            by_template["form_entry_validation"]["classification"],
            "insufficient_repeatability",
        )

        self.assertEqual(analysis["failure_codes"]["nonce_mismatch"], 31)
        self.assertEqual(analysis["terminal_mechanisms"]["success"], 41)
        self.assertEqual(analysis["terminal_mechanisms"]["incorrect_answer"], 31)

        # Every cell (template x difficulty) is present with exactly 4 attempts.
        self.assertEqual(len(analysis["by_template_difficulty"]), 18)
        self.assertTrue(
            all(cell["attempts"] == 4 for cell in analysis["by_template_difficulty"])
        )
        self.assertTrue(
            all(
                cell["replica_agreement"]["total_tasks"] == 2
                for cell in analysis["by_template_difficulty"]
            )
        )

        # Resource summaries cover all 72 attempts.
        self.assertEqual(analysis["resource_summaries"]["output_tokens"]["count"], 72)
        self.assertAlmostEqual(
            analysis["resource_summaries"]["output_tokens"]["mean"], 100.0, places=6
        )
        for metric in (
            "provider_backed_turns",
            "tool_attempts",
            "budget_admitted_tool_attempts",
            "executed_tool_calls",
            "rejected_tool_attempts",
            "budget_blocked_tool_attempts",
        ):
            self.assertEqual(
                analysis["resource_summaries"][metric]["count"], 72, metric
            )

        # Replicated-failure divergence detects the diverging first tool-trace step.
        divergence = {
            entry["task_id"]: entry
            for entry in analysis["replicated_failure_divergence"]
        }
        self.assertTrue(divergence)
        table_task = next(
            t for t in _MANIFEST["tasks"] if t["template"] == "table_filter_sort"
        )
        entry = divergence[table_task["task_id"]]
        self.assertFalse(entry["identical"])
        self.assertEqual(entry["divergence_index"], 1)


if __name__ == "__main__":
    unittest.main()
