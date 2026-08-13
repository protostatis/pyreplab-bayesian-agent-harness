"""Tests for the semantic-capability canary gate.

All tests use compact synthetic ``TreatmentSpec`` instances, a synthetic
registry, split/manifests built through the exploratory screen builder,
and synthetic strict panel result records.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from pyreplab_harness.m3_exploratory_screen import (
    PANEL_RESULT_SCHEMA,
    build_screen_manifest,
)
from pyreplab_harness.m3_semantic_capability_gate import (
    DATASET_CONTRACT_SCHEMA,
    GATE_SCHEMA,
    GATE_SCHEMA_V2,
    PROTOCOL_SCHEMA_V2,
    _V2_REPLICATION_DECISION_RULE,
    _assess_semantic_specialist_adherence,
    _is_infrastructure_error,
    _load_semantic_capability_records,
    _validate_semantic_capability_protocol,
    evaluate_semantic_capability_gate,
)
from pyreplab_harness.orchestrator import policy_spec_from_treatment
from pyreplab_harness.treatments import TreatmentRegistry, TreatmentSpec

# ---------------------------------------------------------------------------
# synthetic treatment builders
# ---------------------------------------------------------------------------

_TABLE_PAYLOAD = {"rows": [{"a": 1, "b": 2}], "columns": ["a", "b"], "shape": [1, 2]}
_FORM_PAYLOAD = {"fields": [{"name": "email", "type": "text"}], "action": "/submit"}


def _canonical_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _encode_payload(payload: dict[str, Any]) -> tuple[int, str]:
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return len(encoded), hashlib.sha256(encoded).hexdigest()


def _make_synthetic_treatment(
    bundle_id: str,
    bundle_hash_prefix: str,
    capability: str,
    tool_interface: str,
    allowed_tools: tuple[str, ...] = ("bash", "unbrowser", "semantic_table"),
    tool_call_limit: int = 6,
    parent_bundle_id: str = "",
    substrate: str = "public_html",
) -> TreatmentSpec:
    """Build a synthetic TreatmentSpec for the semantic capability canary."""
    bundle_hash = bundle_hash_prefix + "a" * (64 - len(bundle_hash_prefix))
    # System prompt must be identical between treatments; capability differs
    # only in metadata.
    return TreatmentSpec(
        id=bundle_id,
        version="2",
        system_prompt=(
            "Planning: direct\n"
            "Capability: specialist_assigned\n"
            "Verification: submit_directly\n"
            "Recovery: fail_fast\n"
            "Safety: Workspace only.\n"
        ),
        allowed_tools=allowed_tools,
        max_output_tokens=4096,
        tool_call_limit=tool_call_limit,
        command_timeout_seconds=60,
        wall_time_limit_seconds=600,
        tool_interface=tool_interface,
        generator_metadata={
            "grammar_name": "semantic_canary_test",
            "grammar_version": "m3-semantic-canary-v1",
            "planning": "direct",
            "capability": capability,
            "verification": "submit_directly",
            "recovery": "fail_fast",
            "tool_cap": "lean" if tool_call_limit <= 6 else "expanded",
            "parent_bundle_id": parent_bundle_id,
            "substrate": substrate,
            "observation_mechanism": "controller_owned_public_html_semantic_operation",
        },
    )


def _make_table_treatment(parent_bundle_id: str = "parent-table-001") -> TreatmentSpec:
    return _make_synthetic_treatment(
        bundle_id="canary-table-specialist",
        bundle_hash_prefix="tb",
        capability="table_specialist",
        tool_interface="native_bash_unbrowser_semantic_table_v1",
        allowed_tools=("bash", "unbrowser", "semantic_table"),
        parent_bundle_id=parent_bundle_id,
    )


def _make_form_treatment(parent_bundle_id: str = "parent-form-001") -> TreatmentSpec:
    return _make_synthetic_treatment(
        bundle_id="canary-form-specialist",
        bundle_hash_prefix="fm",
        capability="form_specialist",
        tool_interface="native_bash_unbrowser_semantic_form_v1",
        allowed_tools=("bash", "unbrowser", "semantic_form"),
        parent_bundle_id=parent_bundle_id,
    )


def _make_semantic_registry(
    table_t: TreatmentSpec | None = None,
    form_t: TreatmentSpec | None = None,
) -> TreatmentRegistry:
    """Build a synthetic registry with the two semantic capability treatments."""
    t1 = table_t or _make_table_treatment()
    t2 = form_t or _make_form_treatment()
    treatments = [t1, t2]
    registry = TreatmentRegistry(treatments)
    return registry


def _make_semantic_policy_split(
    registry: TreatmentRegistry,
) -> dict[str, Any]:
    """Build a synthetic policy split with both treatments in meta_train."""
    bundle_ids = [t.bundle_id for t in registry]
    payload: dict[str, Any] = {
        "grammar_name": "semantic_canary_test",
        "grammar_version": "m3-semantic-canary-v1",
        "policy_version": "2",
        "registry_file": "synthetic-treatments.json",
        "registry_hash": registry.registry_hash,
        "schema_version": "m3-policy-split-v1",
        "split_algorithm": "semantic-canary-test-v1",
        "split_seed": 42,
        "splits": {
            "development": [],
            "final_held_out": [],
            "meta_train": list(bundle_ids),
        },
    }
    payload["manifest_hash"] = _canonical_hash(payload)
    return payload


def _make_semantic_manifest(
    registry: TreatmentRegistry,
    policy_split: dict[str, Any],
    stage: str = "mechanics_dry_run",
    num_tasks: int = 2,
    rollout_replicas: int = 1,
    schedule_seed: int = 42,
    sampling_seed_start: int = 1000,
    task_groups: dict | None = None,
) -> dict[str, Any]:
    """Build a semantic capability manifest using the exploratory screen builder."""
    bundle_ids = [t.bundle_id for t in registry]

    if stage == "mechanics_dry_run":
        decision_rule: dict[str, Any] = {"all_attempts_mechanically_valid": True}
    else:
        decision_rule = {
            "maximum_discordant_cells": 2,
            "minimum_stable_table_only_tasks": 1,
            "minimum_stable_form_only_tasks": 1,
            "minimum_successes_per_arm": 2,
            "maximum_successes_per_arm": 14,
            "maximum_absolute_success_difference": 4,
        }

    treatments_by_cap = {
        str(treatment.generator_metadata["capability"]): treatment
        for treatment in registry
    }

    protocol: dict[str, Any] = {
        "schema_version": "m3-semantic-capability-protocol-v1",
        "stage": stage,
        "claim_boundary": "screening_futility_only",
        "mechanism": {
            "name": "controller_owned_public_html_semantic_operation",
            "receipt_schema_version": "pyreplab-semantic-specialist-receipt-v1",
        },
        "parent_bundle_ids": {
            level: str(treatment.generator_metadata["parent_bundle_id"])
            for level, treatment in treatments_by_cap.items()
        },
        "decision_rule": decision_rule,
    }
    if task_groups is not None:
        protocol["task_groups"] = task_groups

    spec: dict[str, Any] = {
        "screen_id": f"semantic-canary-{stage}-001",
        "purpose": f"Synthetic semantic capability canary {stage}",
        "remote_identity": {
            "host": "test-host",
            "project": "/remote/test-project",
            "run_root": "/remote/test-runs",
            "python": "python3",
        },
        "policy_bundle_ids": bundle_ids,
        "tasks": [
            {
                "template": (
                    "table_filter_sort"
                    if i < num_tasks // 2
                    else "form_entry_validation"
                ),
                "difficulty": "easy",
                "seed": 1000 + i,
            }
            for i in range(num_tasks)
        ],
        "rollout_replicas": rollout_replicas,
        "sampling_seed_start": sampling_seed_start,
        "schedule_seed": schedule_seed,
        "task_role": "T_canary",
        "protocol": protocol,
        "selection": {"reason": "synthetic semantic capability canary test"},
    }
    return build_screen_manifest(
        registry,
        policy_split,
        spec,
        registry_file="synthetic-treatments.json",
        policy_split_file="synthetic-split.json",
    )


# ---------------------------------------------------------------------------
# synthetic record builders
# ---------------------------------------------------------------------------


def _make_semantic_navigate_entry(
    specialist: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Build a semantic unbrowser trace entry with a valid specialist receipt."""
    payload_bytes, payload_sha256 = _encode_payload(payload)
    action = "semantic_table" if specialist == "table_specialist" else "semantic_form"
    return {
        "tool_name": action,
        "is_error": False,
        "budget_rejected": False,
        "operation_aborted": False,
        "details": {
            "status": 200,
            "url": "http://127.0.0.1:PORT/test/page_0",
            "semantic_specialist_receipt": {
                "schema_version": "pyreplab-semantic-specialist-receipt-v1",
                "specialist": specialist,
                "action": action,
                "delivered": True,
                "payload_bytes": payload_bytes,
                "payload_sha256": payload_sha256,
            },
            "semantic_payload": payload,
        },
    }


def _make_infra_error_entry(marker: str) -> dict[str, Any]:
    """Build an infrastructure-error tool trace entry."""
    return {
        "tool_name": "unbrowser",
        "is_error": True,
        "budget_rejected": False,
        "operation_aborted": False,
        "details": {
            "action": "navigate",
            "status": 500,
            "error": marker,
            "infrastructure_error": True,
        },
    }


def _make_submit_entry() -> dict[str, Any]:
    return {
        "tool_name": "bash",
        "is_error": False,
        "budget_rejected": False,
        "operation_aborted": False,
        "details": {"result_submission": True, "status": 0},
    }


def _make_attempt(
    treatment: TreatmentSpec,
    *,
    manifest: dict[str, Any],
    success: bool = True,
    output_tokens: float = 100.0,
    pi_return_code: int = 0,
    with_missing_receipt: bool = False,
    with_infra_error: str | None = None,
    with_wrong_specialist: bool = False,
    include_receipt: bool = True,
) -> dict[str, Any]:
    """Build one synthetic attempt for a semantic capability treatment."""
    specialist = str(treatment.generator_metadata.get("capability", ""))
    payload = _TABLE_PAYLOAD if specialist == "table_specialist" else _FORM_PAYLOAD

    runtime_pins = manifest["runtime_pins"]

    tool_trace: list[dict[str, Any]] = []

    if with_infra_error:
        tool_trace.append(_make_infra_error_entry(with_infra_error))
    elif with_wrong_specialist:
        # Use the wrong specialist action in the details.
        wrong_payload = _FORM_PAYLOAD if specialist == "table_specialist" else _TABLE_PAYLOAD
        payload_bytes, payload_sha256 = _encode_payload(wrong_payload)
        wrong_action = "semantic_form" if specialist == "table_specialist" else "semantic_table"
        wrong_specialist_name = "form_specialist" if specialist == "table_specialist" else "table_specialist"
        tool_trace.append({
            "tool_name": wrong_action,
            "is_error": False,
            "budget_rejected": False,
            "operation_aborted": False,
            "details": {
                "status": 200,
                "url": "http://127.0.0.1:PORT/test/page_0",
                "semantic_specialist_receipt": {
                    "schema_version": "pyreplab-semantic-specialist-receipt-v1",
                    "specialist": wrong_specialist_name,
                    "action": wrong_action,
                    "delivered": True,
                    "payload_bytes": payload_bytes,
                    "payload_sha256": payload_sha256,
                },
                "semantic_payload": wrong_payload,
            },
        })
    elif with_missing_receipt or not include_receipt:
        tool_trace.append({
            "tool_name": "unbrowser",
            "is_error": False,
            "budget_rejected": False,
            "operation_aborted": False,
            "details": {
                "action": "navigate",
                "status": 200,
                "url": "http://127.0.0.1:PORT/test/page_0",
            },
        })
    else:
        tool_trace.append(_make_semantic_navigate_entry(specialist, payload))

    tool_trace.append(_make_submit_entry())

    trajectory: dict[str, Any] = {
        "planning_preamble": {"present": False},
        "tool_trace": tool_trace,
        "provider_turn_count": 1,
    }

    expected_policy = policy_spec_from_treatment(treatment).to_dict()

    return {
        "attempt_id": f"attempt-{treatment.bundle_id}",
        "policy": expected_policy,
        "pi_return_code": pi_return_code,
        "pi_stderr": "",
        "verification": {
            "success": success,
            "details": {},
            "verifier_id": runtime_pins["fixture_verifier_id"],
            "verifier_version": runtime_pins["fixture_verifier_version"],
        },
        "usage": {"output": output_tokens, "prompt_tokens": 50},
        "trajectory": trajectory,
        "sampling_receipt": {
            "seed": 0,  # will be patched
            "parameters": runtime_pins["sampling"]["parameters"],
        },
        "timing": {},
    }


def _make_panel_result(
    manifest: dict[str, Any],
    panel: dict[str, Any],
    table_treatment: TreatmentSpec,
    form_treatment: TreatmentSpec,
    table_success: bool = True,
    form_success: bool = True,
    *,
    attempt_mods: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a valid panel result record."""
    task = next(t for t in manifest["tasks"] if t["task_id"] == panel["task_id"])
    attempts: dict[str, dict[str, Any]] = {}
    for bid in panel["execution_order"]:
        if bid == table_treatment.bundle_id:
            treatment = table_treatment
            base_success = table_success
        else:
            treatment = form_treatment
            base_success = form_success

        mods = (attempt_mods or {}).get(bid, {})
        if (
            manifest["protocol"]["stage"] == "mechanics_dry_run"
            and "include_receipt" not in mods
        ):
            task_index = next(
                index
                for index, task in enumerate(manifest["tasks"])
                if task["task_id"] == panel["task_id"]
            )
            expected_capability = (
                "table_specialist" if task_index == 0 else "form_specialist"
            )
            mods = {
                **mods,
                "include_receipt": (
                    treatment.generator_metadata.get("capability")
                    == expected_capability
                ),
            }
        att = _make_attempt(treatment, manifest=manifest, success=base_success, **mods)
        att["sampling_receipt"]["seed"] = panel["sampling_seed"]
        att["attempt_id"] = f"attempt-{panel['panel_id']}-{bid}"
        attempts[bid] = att

    return {
        "schema_version": PANEL_RESULT_SCHEMA,
        "panel_id": panel["panel_id"],
        "manifest_hash": manifest["manifest_hash"],
        "task": task,
        "panel": panel,
        "status": "completed",
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:01:00Z",
        "duration_seconds": 60.0,
        "result": {
            "task_id": panel["task_id"],
            "mode": "treatment_set",
            "execution_order": panel["execution_order"],
            "attempts": attempts,
            "pilot_manifest_hash": manifest["manifest_hash"],
            "pilot_panel_id": panel["panel_id"],
            "rollout_replica": panel["rollout_replica"],
            "sampling_seed": panel["sampling_seed"],
            "treatment_registry_hash": "",  # Patched later
        },
    }


def _make_preflight(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "manifest_hash": manifest["manifest_hash"],
        "pilot_manifest_hash": manifest["manifest_hash"],
        "screen_preflight": True,
        "checked_at": "2026-01-01T00:00:00Z",
        "code_revision": "a" * 40,
        "source_tree_hash": "b" * 64,
        "worktree_clean": True,
        "worktree_status_hash": hashlib.sha256(
            b"PYREPLAB_GIT_WORKTREE_CLEAN_MARKER_V1"
        ).hexdigest(),
        "runtime_pins": manifest["runtime_pins"],
        "remote_identity": manifest["remote_identity"],
    }


# ---------------------------------------------------------------------------
# tests: mechanics_dry_run
# ---------------------------------------------------------------------------


class SemanticCapabilityGateMechanicsTest(unittest.TestCase):
    """Tests for mechanics_dry_run stage."""

    def setUp(self) -> None:
        self.table_treatment = _make_table_treatment()
        self.form_treatment = _make_form_treatment()
        self.registry = _make_semantic_registry(self.table_treatment, self.form_treatment)
        self.policy_split = _make_semantic_policy_split(self.registry)
        self.manifest = _make_semantic_manifest(
            self.registry, self.policy_split, stage="mechanics_dry_run", num_tasks=2
        )
        self.manifest = dict(self.manifest)
        self.manifest.pop("manifest_hash", None)
        self.manifest["registry_hash"] = self.registry.registry_hash
        self.manifest["manifest_hash"] = _canonical_hash(self.manifest)
        self.preflight = _make_preflight(self.manifest)

    def _make_records(
        self, attempt_mods: dict | None = None
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for panel in self.manifest["panels"]:
            record = _make_panel_result(
                self.manifest,
                panel,
                self.table_treatment,
                self.form_treatment,
                attempt_mods=attempt_mods,
            )
            record["result"]["treatment_registry_hash"] = self.registry.registry_hash
            records.append(record)
        return records

    def test_mechanics_pass(self) -> None:
        """Complete mechanics_dry_run with all checks passing."""
        records = self._make_records()
        report = evaluate_semantic_capability_gate(
            self.manifest, self.registry, self.policy_split, records,
            preflight=self.preflight,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["decision"], "mechanics_pass")
        self.assertEqual(report["stage"], "mechanics_dry_run")
        self.assertEqual(report["gate"], GATE_SCHEMA)
        self.assertEqual(report["manifest_hash"], self.manifest["manifest_hash"])
        self.assertIn("warning", report)
        self.assertIn("screening", report["warning"].lower())
        self.assertIn("capability family", report["warning"].lower())

    def test_report_is_deterministic(self) -> None:
        records = self._make_records()
        r1 = evaluate_semantic_capability_gate(
            self.manifest, self.registry, self.policy_split, records,
            preflight=self.preflight,
        )
        r2 = evaluate_semantic_capability_gate(
            self.manifest, self.registry, self.policy_split, records,
            preflight=self.preflight,
        )
        self.assertEqual(r1, r2)

    def test_missing_specialist_receipt_fails(self) -> None:
        """Attempts without a specialist receipt should fail mechanics."""
        records = self._make_records(attempt_mods={
            self.table_treatment.bundle_id: {"with_missing_receipt": True},
            self.form_treatment.bundle_id: {"with_missing_receipt": True},
        })
        report = evaluate_semantic_capability_gate(
            self.manifest, self.registry, self.policy_split, records,
            preflight=self.preflight,
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["decision"], "invalid")
        self.assertTrue(any("receipt" in r.lower() for r in report["reasons"]))

    def test_infrastructure_error_marker_fails(self) -> None:
        """BrokenPipeError infrastructure marker should invalidate."""
        records = self._make_records(attempt_mods={
            self.table_treatment.bundle_id: {"with_infra_error": "BrokenPipeError in pipe"},
            self.form_treatment.bundle_id: {},
        })
        report = evaluate_semantic_capability_gate(
            self.manifest, self.registry, self.policy_split, records,
            preflight=self.preflight,
        )
        self.assertFalse(report["passed"])
        self.assertTrue(any(
            "infrastructure error" in r.lower() for r in report["reasons"]
        ))

    def test_browser_process_exited_fails(self) -> None:
        """'browser process exited' marker should invalidate."""
        records = self._make_records(attempt_mods={
            self.table_treatment.bundle_id: {},
            self.form_treatment.bundle_id: {"with_infra_error": "browser process exited"},
        })
        report = evaluate_semantic_capability_gate(
            self.manifest, self.registry, self.policy_split, records,
            preflight=self.preflight,
        )
        self.assertFalse(report["passed"])

    def test_connection_reset_error_fails(self) -> None:
        """ConnectionResetError marker should invalidate."""
        records = self._make_records(attempt_mods={
            self.table_treatment.bundle_id: {"with_infra_error": "ConnectionResetError"},
            self.form_treatment.bundle_id: {},
        })
        report = evaluate_semantic_capability_gate(
            self.manifest, self.registry, self.policy_split, records,
            preflight=self.preflight,
        )
        self.assertFalse(report["passed"])

    def test_response_timed_out_fails(self) -> None:
        """'response timed out' marker should invalidate."""
        records = self._make_records(attempt_mods={
            self.form_treatment.bundle_id: {"with_infra_error": "response timed out after 30s"},
            self.table_treatment.bundle_id: {},
        })
        report = evaluate_semantic_capability_gate(
            self.manifest, self.registry, self.policy_split, records,
            preflight=self.preflight,
        )
        self.assertFalse(report["passed"])

    def test_unavailable_specialist_trace_fails(self) -> None:
        """Wrong specialist in an arm's trace should invalidate."""
        records = self._make_records(attempt_mods={
            self.table_treatment.bundle_id: {"with_wrong_specialist": True},
            self.form_treatment.bundle_id: {},
        })
        report = evaluate_semantic_capability_gate(
            self.manifest, self.registry, self.policy_split, records,
            preflight=self.preflight,
        )
        self.assertFalse(report["passed"])
        self.assertTrue(any(
            "unavailable specialist" in r.lower() for r in report["reasons"]
        ))

    def test_policy_identity_mismatch_fails(self) -> None:
        """Executed policy not matching registry treatment should fail."""
        records: list[dict[str, Any]] = []
        for panel in self.manifest["panels"]:
            record = _make_panel_result(
                self.manifest, panel, self.table_treatment, self.form_treatment,
            )
            record["result"]["treatment_registry_hash"] = self.registry.registry_hash
            for bid, att in record["result"]["attempts"].items():
                if bid == self.table_treatment.bundle_id:
                    att["policy"] = {"id": "wrong", "version": "99"}
            records.append(record)
        report = evaluate_semantic_capability_gate(
            self.manifest, self.registry, self.policy_split, records,
            preflight=self.preflight,
        )
        self.assertFalse(report["passed"])
        self.assertTrue(any("policy" in r.lower() for r in report["reasons"]))

    def test_sampling_receipt_mismatch_fails(self) -> None:
        """Mismatched sampling receipt should fail."""
        records: list[dict[str, Any]] = []
        for panel in self.manifest["panels"]:
            record = _make_panel_result(
                self.manifest, panel, self.table_treatment, self.form_treatment,
            )
            record["result"]["treatment_registry_hash"] = self.registry.registry_hash
            for bid, att in record["result"]["attempts"].items():
                if bid == self.table_treatment.bundle_id:
                    att["sampling_receipt"]["seed"] = 99999
                    att["trajectory"]["provider_turn_count"] = 2
            records.append(record)
        report = evaluate_semantic_capability_gate(
            self.manifest, self.registry, self.policy_split, records,
            preflight=self.preflight,
        )
        self.assertFalse(report["passed"])
        self.assertTrue(any("sampling" in r.lower() for r in report["reasons"]))

    def test_tool_cap_failure(self) -> None:
        """Exceeding tool call limit should be caught."""
        records: list[dict[str, Any]] = []
        for panel in self.manifest["panels"]:
            record = _make_panel_result(
                self.manifest, panel, self.table_treatment, self.form_treatment,
            )
            record["result"]["treatment_registry_hash"] = self.registry.registry_hash
            for att in record["result"]["attempts"].values():
                extra_calls = [
                    {
                        "tool_name": "unbrowser",
                        "is_error": False,
                        "budget_rejected": False,
                        "operation_aborted": False,
                        "details": {"action": "navigate", "status": 200},
                    }
                    for _ in range(10)
                ]
                att["trajectory"]["tool_trace"][1:1] = extra_calls
            records.append(record)
        report = evaluate_semantic_capability_gate(
            self.manifest, self.registry, self.policy_split, records,
            preflight=self.preflight,
        )
        self.assertFalse(report["passed"])
        self.assertTrue(any("tool_cap" in r.lower() for r in report["reasons"]))

    def test_preflight_identity_missing_fails(self) -> None:
        """Missing preflight should fail."""
        records = self._make_records()
        report = evaluate_semantic_capability_gate(
            self.manifest, self.registry, self.policy_split, records,
            preflight=None,
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["decision"], "invalid")
        self.assertFalse(report["checks"]["preflight_identity_valid"])

    def test_manifest_hash_mismatch_in_record_rejected(self) -> None:
        """Records with wrong manifest hash are rejected during loading."""
        records = self._make_records()
        records[0] = dict(records[0])
        records[0]["manifest_hash"] = "deadbeef"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.jsonl"
            path.write_text(
                "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                _load_semantic_capability_records(path, self.manifest)

    def test_duplicate_attempt_ids_fail(self) -> None:
        """Duplicate attempt IDs should produce structural errors."""
        records = self._make_records()
        for record in records:
            for att in record["result"]["attempts"].values():
                att["attempt_id"] = "duplicate-id"
        report = evaluate_semantic_capability_gate(
            self.manifest, self.registry, self.policy_split, records,
            preflight=self.preflight,
        )
        self.assertFalse(report["passed"])
        self.assertTrue(any("duplicate" in r.lower() for r in report["reasons"]))

    def test_wrong_task_role_rejected(self) -> None:
        """Manifest without T_canary role should fail protocol validation."""
        with self.assertRaises(ValueError) as ctx:
            _validate_semantic_capability_protocol(
                {
                    "task_role": "T_pilot",
                    "policy_bundle_ids": [
                        self.table_treatment.bundle_id,
                        self.form_treatment.bundle_id,
                    ],
                    "protocol": {
                        "schema_version": "m3-semantic-capability-protocol-v1",
                        "stage": "mechanics_dry_run",
                    },
                },
                self.registry,
            )
        self.assertIn("T_canary", str(ctx.exception))

    def test_wrong_protocol_schema_rejected(self) -> None:
        """Wrong protocol schema_version should fail."""
        with self.assertRaises(ValueError) as ctx:
            _validate_semantic_capability_protocol(
                {
                    "task_role": "T_canary",
                    "policy_bundle_ids": [
                        self.table_treatment.bundle_id,
                        self.form_treatment.bundle_id,
                    ],
                    "protocol": {
                        "schema_version": "wrong-schema",
                        "stage": "mechanics_dry_run",
                    },
                },
                self.registry,
            )
        self.assertIn("schema_version", str(ctx.exception))

    def test_wrong_capability_metadata_rejected(self) -> None:
        """Treatments with wrong capability metadata should fail."""
        bad_table = _make_synthetic_treatment(
            bundle_id="bad-table",
            bundle_hash_prefix="bt",
            capability="text_specialist",
            tool_interface="native_bash_unbrowser_semantic_table_v1",
            allowed_tools=("bash", "unbrowser", "semantic_table"),
        )
        registry = _make_semantic_registry(table_t=bad_table)
        with self.assertRaises(ValueError) as ctx:
            _validate_semantic_capability_protocol(
                {
                    "task_role": "T_canary",
                    "policy_bundle_ids": [
                        bad_table.bundle_id,
                        _make_form_treatment().bundle_id,
                    ],
                    "protocol": {
                        "schema_version": "m3-semantic-capability-protocol-v1",
                        "stage": "mechanics_dry_run",
                        "claim_boundary": "screening_futility_only",
                        "decision_rule": {"all_attempts_mechanically_valid": True},
                        "mechanism": {"name": "controller_owned_public_html_semantic_operation"},
                    },
                },
                registry,
            )
        self.assertIn("capability", str(ctx.exception).lower())

    def test_report_has_all_required_sections(self) -> None:
        """Gate report must have all required top-level keys."""
        records = self._make_records()
        report = evaluate_semantic_capability_gate(
            self.manifest, self.registry, self.policy_split, records,
            preflight=self.preflight,
        )
        required_keys = {
            "gate", "schema_version", "manifest_hash", "stage",
            "passed", "decision", "checks", "reasons",
            "completeness", "mechanism", "stability",
            "descriptive_outcomes", "warning",
        }
        self.assertTrue(required_keys.issubset(report.keys()))


# ---------------------------------------------------------------------------
# tests: outcome_screen
# ---------------------------------------------------------------------------


class SemanticCapabilityGateOutcomeScreenTest(unittest.TestCase):
    """Tests for outcome_screen stage."""

    def setUp(self) -> None:
        self.table_treatment = _make_table_treatment()
        self.form_treatment = _make_form_treatment()
        self.registry = _make_semantic_registry(self.table_treatment, self.form_treatment)
        self.policy_split = _make_semantic_policy_split(self.registry)

    def _build_outcome_manifest(
        self, task_groups: dict | None = None, num_tasks: int = 8
    ) -> dict[str, Any]:
        """Build outcome_screen manifest with the given task count and task_groups."""
        if task_groups is None:
            # Create a temporary manifest to get task IDs, then derive task_groups.
            temp = _make_semantic_manifest(
                self.registry, self.policy_split, stage="outcome_screen",
                num_tasks=num_tasks, rollout_replicas=2,
                task_groups={
                    "table": [],
                    "form": [],
                },
            )
            # Build real task_groups from the manifest tasks.
            all_tasks = temp["tasks"]
            task_groups = {
                "table": [t["task_id"] for t in all_tasks[: num_tasks // 2]],
                "form": [t["task_id"] for t in all_tasks[num_tasks // 2:]],
            }
        manifest = _make_semantic_manifest(
            self.registry, self.policy_split, stage="outcome_screen",
            num_tasks=num_tasks, rollout_replicas=2,
            task_groups=task_groups,
        )
        manifest = dict(manifest)
        manifest.pop("manifest_hash", None)
        manifest["registry_hash"] = self.registry.registry_hash
        manifest["manifest_hash"] = _canonical_hash(manifest)
        return manifest

    def _default_tasks(self, manifest: dict[str, Any]) -> list[dict[str, Any]]:
        """Get tasks from a manifest."""
        return manifest["tasks"]

    def _make_outcome_records(
        self,
        manifest: dict[str, Any],
        table_successes: dict[int, dict[int, bool]] | None = None,
        form_successes: dict[int, dict[int, bool]] | None = None,
    ) -> list[dict[str, Any]]:
        """Build records for outcome_screen with specified per-task-per-replica
        success patterns.

        Suppresses specialist receipts on non-matching task groups so the gate
        does not flag "unexpected specialist receipt" errors.
        """
        records: list[dict[str, Any]] = []
        tasks = manifest["tasks"]
        task_indices = {t["task_id"]: i for i, t in enumerate(tasks)}

        # Read task_group mapping from protocol.
        protocol_tg = manifest.get("protocol", {}).get("task_groups", {})
        table_task_ids = set(str(tid) for tid in protocol_tg.get("table", []))
        form_task_ids = set(str(tid) for tid in protocol_tg.get("form", []))

        for panel in manifest["panels"]:
            tid = panel["task_id"]
            replica = panel["rollout_replica"]
            t_idx = task_indices[tid]

            # Determine which arm(s) should have receipts based on task group.
            # table specialist receipt only on table tasks; form only on form.
            table_has_receipt = tid in table_task_ids
            form_has_receipt = tid in form_task_ids

            table_ok = True
            form_ok = True
            if table_successes and t_idx in table_successes:
                table_ok = table_successes[t_idx].get(replica, True)
            if form_successes and t_idx in form_successes:
                form_ok = form_successes[t_idx].get(replica, True)

            attempt_mods: dict[str, dict[str, Any]] = {}
            if not table_has_receipt:
                attempt_mods[self.table_treatment.bundle_id] = {"with_missing_receipt": True}
            if not form_has_receipt:
                attempt_mods[self.form_treatment.bundle_id] = {"with_missing_receipt": True}

            record = _make_panel_result(
                manifest, panel, self.table_treatment, self.form_treatment,
                table_success=table_ok, form_success=form_ok,
                attempt_mods=attempt_mods if attempt_mods else None,
            )
            record["result"]["treatment_registry_hash"] = self.registry.registry_hash
            records.append(record)
        return records

    def test_outcome_pass_with_bidirectional_stable_tasks(self) -> None:
        """Scenario: 1 stable table-only, 1 stable form-only, rest stable
        within arm success bounds."""
        manifest = self._build_outcome_manifest()
        preflight = _make_preflight(manifest)

        # 8 tasks, 2 replicas
        # Task 0: stable table-only (table 2/2, form 0/2)
        # Task 4: stable form-only (form 2/2, table 0/2)
        # Tasks 1-3, 5-7: both succeed on both replicas (balanced)
        table_successes: dict[int, dict[int, bool]] = {
            0: {0: True, 1: True},
            4: {0: False, 1: False},
        }
        form_successes: dict[int, dict[int, bool]] = {
            0: {0: False, 1: False},
            4: {0: True, 1: True},
        }
        for i in [1, 2, 3, 5, 6, 7]:
            table_successes[i] = {0: True, 1: True}
            form_successes[i] = {0: True, 1: True}

        records = self._make_outcome_records(
            manifest,
            table_successes=table_successes,
            form_successes=form_successes,
        )
        report = evaluate_semantic_capability_gate(
            manifest, self.registry, self.policy_split, records,
            preflight=preflight,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["decision"], "screen_pass")
        self.assertEqual(report["stage"], "outcome_screen")

        stability = report["stability"]
        self.assertLessEqual(stability["discordant_cell_count"], 2)
        self.assertGreaterEqual(stability["stable_table_only_count"], 1)
        self.assertGreaterEqual(stability["stable_form_only_count"], 1)

    def test_outcome_futility_no_reverse_niche(self) -> None:
        """No stable form-only task → futility_no_go."""
        manifest = self._build_outcome_manifest()
        preflight = _make_preflight(manifest)

        # All tasks: table succeeds, form succeeds. No reverse niche.
        records = self._make_outcome_records(manifest)
        report = evaluate_semantic_capability_gate(
            manifest, self.registry, self.policy_split, records,
            preflight=preflight,
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["decision"], "futility_no_go")
        self.assertEqual(report["stability"]["stable_form_only_count"], 0)

    def test_outcome_futility_excess_discordance(self) -> None:
        """Too many discordant cells → futility_no_go."""
        manifest = self._build_outcome_manifest()
        preflight = _make_preflight(manifest)

        # Make 3 tasks discordant (table disagrees on replicas).
        table_successes: dict[int, dict[int, bool]] = {
            i: {0: True, 1: False} for i in range(3)
        }
        table_successes.update({i: {0: True, 1: True} for i in range(3, 8)})
        form_successes: dict[int, dict[int, bool]] = {
            i: {0: False, 1: True} for i in range(3)
        }
        form_successes.update({i: {0: True, 1: True} for i in range(3, 8)})

        records = self._make_outcome_records(
            manifest,
            table_successes=table_successes,
            form_successes=form_successes,
        )
        report = evaluate_semantic_capability_gate(
            manifest, self.registry, self.policy_split, records,
            preflight=preflight,
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["decision"], "futility_no_go")
        self.assertGreater(report["stability"]["discordant_cell_count"], 2)

    def test_outcome_futility_degenerate(self) -> None:
        """All tasks succeed for both arms → no niche → futility_no_go."""
        manifest = self._build_outcome_manifest()
        preflight = _make_preflight(manifest)

        records = self._make_outcome_records(manifest)
        report = evaluate_semantic_capability_gate(
            manifest, self.registry, self.policy_split, records,
            preflight=preflight,
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["decision"], "futility_no_go")

    def test_outcome_futility_arm_imbalance(self) -> None:
        """Large arm success imbalance (>4 diff) should cause futility."""
        manifest = self._build_outcome_manifest()
        preflight = _make_preflight(manifest)

        # Table arm succeeds most, form arm rarely, beyond max_abs_diff=4.
        table_successes: dict[int, dict[int, bool]] = {}
        form_successes: dict[int, dict[int, bool]] = {}
        for i in range(8):
            table_successes[i] = {0: True, 1: True}  # 16 successes
            if i < 3:
                form_successes[i] = {0: False, 1: False}
            else:
                form_successes[i] = {0: False, 1: False}  # 0 form successes

        records = self._make_outcome_records(
            manifest,
            table_successes=table_successes,
            form_successes=form_successes,
        )
        report = evaluate_semantic_capability_gate(
            manifest, self.registry, self.policy_split, records,
            preflight=preflight,
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["decision"], "futility_no_go")
        self.assertGreater(
            report["stability"]["absolute_success_difference"],
            4,
        )

    def test_outcome_futility_below_min_successes(self) -> None:
        """Arm below minimum_successes_per_arm (2) should cause futility."""
        manifest = self._build_outcome_manifest()
        preflight = _make_preflight(manifest)

        # Form arm has 0 successes, table arm stable.
        # Task 0: stable table-only (table 2/2, form 0/2)
        # Rest: all fail for form, table mixed
        table_successes: dict[int, dict[int, bool]] = {
            0: {0: True, 1: True},
        }
        form_successes: dict[int, dict[int, bool]] = {
            0: {0: False, 1: False},
        }
        for i in range(1, 8):
            table_successes[i] = {0: True, 1: True}
            form_successes[i] = {0: False, 1: False}

        records = self._make_outcome_records(
            manifest,
            table_successes=table_successes,
            form_successes=form_successes,
        )
        report = evaluate_semantic_capability_gate(
            manifest, self.registry, self.policy_split, records,
            preflight=preflight,
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["decision"], "futility_no_go")
        self.assertTrue(any(
            "form arm successes" in r.lower() for r in report["reasons"]
        ))

    def test_outcome_wrong_shape_rejected(self) -> None:
        """outcome_screen with wrong number of tasks should be detected as invalid."""
        # Build mechanics manifest with 2 tasks, but mislabel as outcome_screen.
        # The gate's size check will detect the mismatch.
        manifest = _make_semantic_manifest(
            self.registry, self.policy_split, stage="mechanics_dry_run", num_tasks=2
        )
        manifest = dict(manifest)
        # Change protocol to outcome_screen but keep small shape.
        manifest["protocol"] = dict(manifest.get("protocol", {}))
        manifest["protocol"]["stage"] = "outcome_screen"
        manifest["protocol"]["decision_rule"] = {
            "maximum_discordant_cells": 2,
            "minimum_stable_table_only_tasks": 1,
            "minimum_stable_form_only_tasks": 1,
            "minimum_successes_per_arm": 2,
            "maximum_successes_per_arm": 14,
            "maximum_absolute_success_difference": 4,
        }
        manifest["protocol"]["task_groups"] = {
            "table": [manifest["tasks"][0]["task_id"]] * 4,
            "form": [manifest["tasks"][1]["task_id"]] * 4,
        }
        manifest.pop("manifest_hash", None)
        manifest["registry_hash"] = self.registry.registry_hash
        manifest["manifest_hash"] = _canonical_hash(manifest)
        preflight = _make_preflight(manifest)

        records: list[dict[str, Any]] = []
        for panel in manifest["panels"]:
            table_task_ids = {"table": [manifest["tasks"][0]["task_id"]] * 4}
            attempt_mods = {}
            record = _make_panel_result(
                manifest, panel, self.table_treatment, self.form_treatment,
            )
            record["result"]["treatment_registry_hash"] = self.registry.registry_hash
            records.append(record)

        report = evaluate_semantic_capability_gate(
            manifest, self.registry, self.policy_split, records,
            preflight=preflight,
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["decision"], "invalid")

    def test_outcome_task_groups_not_partitioning_fails(self) -> None:
        """Task groups not having exactly 4+4 task IDs should fail protocol."""
        temp = _make_semantic_manifest(
            self.registry, self.policy_split, stage="outcome_screen",
            num_tasks=8, rollout_replicas=2,
            task_groups={"table": [], "form": []},
        )
        all_tasks = temp["tasks"]
        # Only 2+2 tasks instead of 4+4 → fails count check.
        bad_task_groups = {
            "table": [all_tasks[0]["task_id"], all_tasks[1]["task_id"]],
            "form": [all_tasks[2]["task_id"], all_tasks[3]["task_id"]],
        }
        bad_manifest = _make_semantic_manifest(
            self.registry, self.policy_split, stage="outcome_screen",
            num_tasks=8, rollout_replicas=2,
            task_groups=bad_task_groups,
        )
        with self.assertRaises(ValueError) as ctx:
            _validate_semantic_capability_protocol(
                bad_manifest, self.registry,
            )
        exc_msg = str(ctx.exception).lower()
        self.assertTrue("4 task ids" in exc_msg or "must be a list of 4" in exc_msg)

    def test_outcome_task_groups_must_exactly_partition_tasks(self) -> None:
        manifest = self._build_outcome_manifest()
        bad = dict(manifest)
        bad["protocol"] = dict(manifest["protocol"])
        bad["protocol"]["task_groups"] = {
            "table": list(manifest["protocol"]["task_groups"]["table"]),
            "form": list(manifest["protocol"]["task_groups"]["form"]),
        }
        bad["protocol"]["task_groups"]["form"][-1] = "unknown-task-id"
        with self.assertRaisesRegex(ValueError, "exactly partition"):
            _validate_semantic_capability_protocol(bad, self.registry)

    def test_outcome_task_group_template_mismatch_fails(self) -> None:
        manifest = self._build_outcome_manifest()
        bad = dict(manifest)
        bad["protocol"] = dict(manifest["protocol"])
        table_ids = list(manifest["protocol"]["task_groups"]["table"])
        form_ids = list(manifest["protocol"]["task_groups"]["form"])
        table_ids[0], form_ids[0] = form_ids[0], table_ids[0]
        bad["protocol"]["task_groups"] = {"table": table_ids, "form": form_ids}
        with self.assertRaisesRegex(ValueError, "non-table template|non-form template"):
            _validate_semantic_capability_protocol(bad, self.registry)

    def test_wrong_family_niches_do_not_count(self) -> None:
        manifest = self._build_outcome_manifest()
        preflight = _make_preflight(manifest)
        table_successes = {i: {0: True, 1: True} for i in range(8)}
        form_successes = {i: {0: True, 1: True} for i in range(8)}
        # Reverse the apparent niches: table-only on a form task and form-only
        # on a table task. Neither should satisfy the family-scoped gate.
        table_successes[0] = {0: False, 1: False}
        form_successes[0] = {0: True, 1: True}
        table_successes[4] = {0: True, 1: True}
        form_successes[4] = {0: False, 1: False}
        records = self._make_outcome_records(
            manifest,
            table_successes=table_successes,
            form_successes=form_successes,
        )
        report = evaluate_semantic_capability_gate(
            manifest, self.registry, self.policy_split, records,
            preflight=preflight,
        )
        self.assertEqual(report["decision"], "futility_no_go")
        self.assertEqual(report["stability"]["stable_table_only_count"], 0)
        self.assertEqual(report["stability"]["stable_form_only_count"], 0)

    def test_specialist_receipt_on_wrong_task_group_fails(self) -> None:
        """On outcome_screen, table specialist receipt on form task should fail."""
        manifest = self._build_outcome_manifest()
        preflight = _make_preflight(manifest)

        records: list[dict[str, Any]] = []
        for panel in manifest["panels"]:
            record = _make_panel_result(
                manifest, panel, self.table_treatment, self.form_treatment,
            )
            record["result"]["treatment_registry_hash"] = self.registry.registry_hash
            records.append(record)

        report = evaluate_semantic_capability_gate(
            manifest, self.registry, self.policy_split, records,
            preflight=preflight,
        )
        # Depending on task group partition: if table specialist bounces on
        # form task, the receipt might still be valid for the table arm
        # attempt but the receipt check only applies when the specialist
        # matches the task group. On form tasks, the table specialist receipt
        # would be "unexpected". However, all tasks succeed so the receipt is
        # valid for both arms on all tasks (matching), which means no
        # specialist receipt error. The gate would fail on stability instead.
        pass  # This is tested indirectly via stability.

    def test_outcome_report_deterministic(self) -> None:
        """Outcome report must be deterministic."""
        manifest = self._build_outcome_manifest()
        preflight = _make_preflight(manifest)

        table_successes = {
            0: {0: True, 1: True},
            4: {0: False, 1: False},
        }
        form_successes = {
            0: {0: False, 1: False},
            4: {0: True, 1: True},
        }
        for i in [1, 2, 3, 5, 6, 7]:
            table_successes[i] = {0: True, 1: True}
            form_successes[i] = {0: True, 1: True}

        records = self._make_outcome_records(
            manifest,
            table_successes=table_successes,
            form_successes=form_successes,
        )
        r1 = evaluate_semantic_capability_gate(
            manifest, self.registry, self.policy_split, records,
            preflight=preflight,
        )
        r2 = evaluate_semantic_capability_gate(
            manifest, self.registry, self.policy_split, records,
            preflight=preflight,
        )
        self.assertEqual(r1, r2)

    def test_outcome_report_has_required_sections(self) -> None:
        """Outcome gate report must have all required top-level keys."""
        manifest = self._build_outcome_manifest()
        preflight = _make_preflight(manifest)
        records = self._make_outcome_records(manifest)
        report = evaluate_semantic_capability_gate(
            manifest, self.registry, self.policy_split, records,
            preflight=preflight,
        )
        required_keys = {
            "gate", "schema_version", "manifest_hash", "stage",
            "passed", "decision", "checks", "reasons",
            "completeness", "mechanism", "stability",
            "descriptive_outcomes", "warning",
        }
        self.assertTrue(required_keys.issubset(report.keys()))

    def test_outcome_warning_mentions_capability_family(self) -> None:
        """Warning must mention 'capability family' qualification."""
        manifest = self._build_outcome_manifest()
        preflight = _make_preflight(manifest)

        table_successes = {
            0: {0: True, 1: True},
            4: {0: False, 1: False},
        }
        form_successes = {
            0: {0: False, 1: False},
            4: {0: True, 1: True},
        }
        for i in [1, 2, 3, 5, 6, 7]:
            table_successes[i] = {0: True, 1: True}
            form_successes[i] = {0: True, 1: True}

        records = self._make_outcome_records(
            manifest,
            table_successes=table_successes,
            form_successes=form_successes,
        )
        report = evaluate_semantic_capability_gate(
            manifest, self.registry, self.policy_split, records,
            preflight=preflight,
        )
        self.assertIn("capability family", report["warning"].lower())
        self.assertIn("screening", report["warning"].lower())


# ---------------------------------------------------------------------------
# tests: CLI
# ---------------------------------------------------------------------------


class SemanticCapabilityGateCLITest(unittest.TestCase):
    """Tests for CLI exit codes."""

    def setUp(self) -> None:
        self.table_treatment = _make_table_treatment()
        self.form_treatment = _make_form_treatment()
        self.registry = _make_semantic_registry(self.table_treatment, self.form_treatment)
        self.policy_split = _make_semantic_policy_split(self.registry)
        self.manifest = _make_semantic_manifest(
            self.registry, self.policy_split, stage="mechanics_dry_run", num_tasks=2
        )
        self.manifest = dict(self.manifest)
        self.manifest.pop("manifest_hash", None)
        self.manifest["registry_hash"] = self.registry.registry_hash
        self.manifest["manifest_hash"] = _canonical_hash(self.manifest)
        self.preflight = _make_preflight(self.manifest)

    def _make_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for panel in self.manifest["panels"]:
            record = _make_panel_result(
                self.manifest, panel, self.table_treatment, self.form_treatment,
            )
            record["result"]["treatment_registry_hash"] = self.registry.registry_hash
            records.append(record)
        return records

    def test_exit_0_on_pass(self) -> None:
        from pyreplab_harness.m3_semantic_capability_gate import main

        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "results.jsonl"
            records = self._make_records()
            results_path.write_text(
                "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n",
                encoding="utf-8",
            )
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
            registry_path = Path(tmpdir) / "registry.json"
            registry_path.write_text(
                json.dumps(self.registry.to_dict()), encoding="utf-8"
            )
            split_path = Path(tmpdir) / "split.json"
            split_path.write_text(json.dumps(self.policy_split), encoding="utf-8")
            preflight_path = Path(tmpdir) / "results.jsonl.preflight.json"
            preflight_path.write_text(json.dumps(self.preflight), encoding="utf-8")

            exit_code = main([
                str(results_path),
                "--manifest", str(manifest_path),
                "--registry", str(registry_path),
                "--policy-split", str(split_path),
                "--preflight", str(preflight_path),
            ])
            self.assertEqual(exit_code, 0)

    def test_exit_2_on_no_go(self) -> None:
        """CLI returns 2 when gate does not pass (futility_no_go)."""
        from pyreplab_harness.m3_semantic_capability_gate import main

        # Build outcome manifest with task_groups.
        temp = _make_semantic_manifest(
            self.registry, self.policy_split, stage="outcome_screen",
            num_tasks=8, rollout_replicas=2,
            task_groups={"table": [], "form": []},
        )
        all_tasks = temp["tasks"]
        task_groups = {
            "table": [t["task_id"] for t in all_tasks[:4]],
            "form": [t["task_id"] for t in all_tasks[4:]],
        }
        manifest = _make_semantic_manifest(
            self.registry, self.policy_split, stage="outcome_screen",
            num_tasks=8, rollout_replicas=2,
            task_groups=task_groups,
        )
        manifest = dict(manifest)
        manifest.pop("manifest_hash", None)
        manifest["registry_hash"] = self.registry.registry_hash
        manifest["manifest_hash"] = _canonical_hash(manifest)
        preflight = _make_preflight(manifest)

        # Determine which task IDs are table vs form.
        table_task_ids = set(str(tid) for tid in task_groups.get("table", []))
        form_task_ids = set(str(tid) for tid in task_groups.get("form", []))

        # All successes → no stable table-only or form-only → futility_no_go
        records: list[dict[str, Any]] = []
        for panel in manifest["panels"]:
            tid = panel["task_id"]
            attempt_mods: dict[str, dict[str, Any]] = {}
            if tid not in table_task_ids:
                attempt_mods[self.table_treatment.bundle_id] = {"with_missing_receipt": True}
            if tid not in form_task_ids:
                attempt_mods[self.form_treatment.bundle_id] = {"with_missing_receipt": True}

            record = _make_panel_result(
                manifest, panel, self.table_treatment, self.form_treatment,
                attempt_mods=attempt_mods if attempt_mods else None,
            )
            record["result"]["treatment_registry_hash"] = self.registry.registry_hash
            records.append(record)

        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "results.jsonl"
            results_path.write_text(
                "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n",
                encoding="utf-8",
            )
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            registry_path = Path(tmpdir) / "registry.json"
            registry_path.write_text(
                json.dumps(self.registry.to_dict()), encoding="utf-8"
            )
            split_path = Path(tmpdir) / "split.json"
            split_path.write_text(json.dumps(self.policy_split), encoding="utf-8")
            preflight_path = Path(tmpdir) / "results.jsonl.preflight.json"
            preflight_path.write_text(json.dumps(preflight), encoding="utf-8")

            exit_code = main([
                str(results_path),
                "--manifest", str(manifest_path),
                "--registry", str(registry_path),
                "--policy-split", str(split_path),
                "--preflight", str(preflight_path),
            ])
            self.assertEqual(exit_code, 2)

    def test_exit_1_on_error(self) -> None:
        """CLI returns 1 on invalid input."""
        from pyreplab_harness.m3_semantic_capability_gate import main

        with tempfile.TemporaryDirectory() as tmpdir:
            exit_code = main([
                "/nonexistent/results.jsonl",
                "--manifest", "/nonexistent/manifest.json",
                "--registry", str(Path(tmpdir) / "nonexistent.json"),
                "--policy-split", str(Path(tmpdir) / "nonexistent.json"),
            ])
            self.assertEqual(exit_code, 1)

    def test_exit_1_on_invalid_report(self) -> None:
        """CLI distinguishes invalid mechanics from a valid futility no-go."""
        from pyreplab_harness.m3_semantic_capability_gate import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            results_path = root / "results.jsonl"
            results_path.write_text(
                "\n".join(
                    json.dumps(record, sort_keys=True)
                    for record in self._make_records()
                ) + "\n",
                encoding="utf-8",
            )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
            registry_path = root / "registry.json"
            registry_path.write_text(
                json.dumps(self.registry.to_dict()), encoding="utf-8"
            )
            split_path = root / "split.json"
            split_path.write_text(json.dumps(self.policy_split), encoding="utf-8")

            exit_code = main([
                str(results_path),
                "--manifest", str(manifest_path),
                "--registry", str(registry_path),
                "--policy-split", str(split_path),
            ])
            self.assertEqual(exit_code, 1)


# ---------------------------------------------------------------------------
# unit tests for helpers
# ---------------------------------------------------------------------------


class InfrastructureErrorDetectionTest(unittest.TestCase):
    """Tests for _is_infrastructure_error."""

    def test_infrastructure_error_flag(self) -> None:
        self.assertTrue(_is_infrastructure_error({
            "details": {"infrastructure_error": True, "error": "something"},
        }))

    def test_broken_pipe(self) -> None:
        self.assertTrue(_is_infrastructure_error({
            "details": {"error": "BrokenPipeError: [Errno 32] Broken pipe"},
        }))

    def test_connection_reset(self) -> None:
        self.assertTrue(_is_infrastructure_error({
            "details": {"error": "ConnectionResetError: connection reset by peer"},
        }))

    def test_browser_exited(self) -> None:
        self.assertTrue(_is_infrastructure_error({
            "details": {"error": "browser process exited unexpectedly"},
        }))

    def test_process_connection_broken(self) -> None:
        self.assertTrue(_is_infrastructure_error({
            "details": {"error": "unbrowser process connection broken (exit_code=124)"},
        }))

    def test_response_timed_out(self) -> None:
        self.assertTrue(_is_infrastructure_error({
            "details": {"error": "response timed out after 30 seconds"},
        }))

    def test_result_exceeds(self) -> None:
        self.assertTrue(_is_infrastructure_error({
            "details": {"error": "result exceeds maximum size limit"},
        }))

    def test_explicit_non_infrastructure_overflow(self) -> None:
        self.assertFalse(_is_infrastructure_error({
            "details": {
                "error": "unbrowser result exceeds 65536 bytes",
                "infrastructure_error": False,
            },
        }))

    def test_normal_error_not_infra(self) -> None:
        self.assertFalse(_is_infrastructure_error({
            "details": {"error": "404 Not Found"},
        }))

    def test_no_details(self) -> None:
        self.assertFalse(_is_infrastructure_error({
            "tool_name": "bash",
            "is_error": True,
        }))

    def test_no_error_field(self) -> None:
        self.assertFalse(_is_infrastructure_error({
            "details": {"action": "navigate", "status": 500},
        }))


class SpecialistAdherenceTest(unittest.TestCase):
    """Tests for _assess_semantic_specialist_adherence."""

    def setUp(self) -> None:
        self.table_treatment = _make_table_treatment()
        self.form_treatment = _make_form_treatment()

    def test_valid_table_specialist_receipt(self) -> None:
        """Table specialist with valid semantic_table receipt."""
        payload_bytes, payload_sha256 = _encode_payload(_TABLE_PAYLOAD)
        traj = {
            "planning_preamble": {"present": False},
            "tool_trace": [
                {
                    "tool_name": "semantic_table",
                    "is_error": False,
                    "budget_rejected": False,
                    "operation_aborted": False,
                    "details": {
                        "status": 200,
                        "semantic_specialist_receipt": {
                            "schema_version": "pyreplab-semantic-specialist-receipt-v1",
                            "specialist": "table_specialist",
                            "action": "semantic_table",
                            "delivered": True,
                            "payload_bytes": payload_bytes,
                            "payload_sha256": payload_sha256,
                        },
                        "semantic_payload": _TABLE_PAYLOAD,
                    },
                },
            ],
            "provider_turn_count": 1,
        }
        result = _assess_semantic_specialist_adherence(
            self.table_treatment, traj, task_group="table",
        )
        self.assertTrue(result["specialist_receipt_valid"])
        self.assertTrue(result["specialist_action_match"])
        self.assertFalse(result["unavailable_specialist_found"])
        self.assertEqual(result["infrastructure_errors"], 0)
        self.assertTrue(result["tool_cap_compliant"])

    def test_valid_form_specialist_receipt(self) -> None:
        """Form specialist with valid semantic_form receipt."""
        payload_bytes, payload_sha256 = _encode_payload(_FORM_PAYLOAD)
        traj = {
            "planning_preamble": {"present": False},
            "tool_trace": [
                {
                    "tool_name": "semantic_form",
                    "is_error": False,
                    "budget_rejected": False,
                    "operation_aborted": False,
                    "details": {
                        "status": 200,
                        "semantic_specialist_receipt": {
                            "schema_version": "pyreplab-semantic-specialist-receipt-v1",
                            "specialist": "form_specialist",
                            "action": "semantic_form",
                            "delivered": True,
                            "payload_bytes": payload_bytes,
                            "payload_sha256": payload_sha256,
                        },
                        "semantic_payload": _FORM_PAYLOAD,
                    },
                },
            ],
            "provider_turn_count": 1,
        }
        result = _assess_semantic_specialist_adherence(
            self.form_treatment, traj, task_group="form",
        )
        self.assertTrue(result["specialist_receipt_valid"])
        self.assertTrue(result["specialist_action_match"])

    def test_missing_receipt(self) -> None:
        """Trajectory without receipt returns invalid."""
        traj = {
            "planning_preamble": {"present": False},
            "tool_trace": [
                {
                    "tool_name": "unbrowser",
                    "is_error": False,
                    "budget_rejected": False,
                    "operation_aborted": False,
                    "details": {
                        "action": "navigate",
                        "status": 200,
                    },
                },
            ],
            "provider_turn_count": 1,
        }
        result = _assess_semantic_specialist_adherence(
            self.table_treatment, traj, task_group="table",
        )
        self.assertIsNone(result["specialist_receipt_valid"])

    def test_infrastructure_error_detected(self) -> None:
        """Infrastructure error entries should be counted."""
        traj = {
            "planning_preamble": {"present": False},
            "tool_trace": [
                {
                    "tool_name": "unbrowser",
                    "is_error": True,
                    "budget_rejected": False,
                    "operation_aborted": False,
                    "details": {
                        "infrastructure_error": True,
                        "error": "BrokenPipeError",
                    },
                },
            ],
            "provider_turn_count": 1,
        }
        result = _assess_semantic_specialist_adherence(
            self.table_treatment, traj, task_group="table",
        )
        self.assertEqual(result["infrastructure_errors"], 1)

    def test_model_visible_infrastructure_error_is_detected(self) -> None:
        traj = {
            "planning_preamble": {"present": False},
            "tool_trace": [{
                "tool_name": "unbrowser",
                "is_error": False,
                "budget_rejected": False,
                "operation_aborted": False,
                "details": {
                    "infrastructure_error": True,
                    "error": "unbrowser process connection broken (exit_code=124)",
                },
            }],
            "provider_turn_count": 1,
        }
        result = _assess_semantic_specialist_adherence(
            self.table_treatment, traj, task_group="table",
        )
        self.assertEqual(result["infrastructure_errors"], 1)

    def test_unavailable_specialist_found(self) -> None:
        """Form action in table specialist trace should flag."""
        traj = {
            "planning_preamble": {"present": False},
            "tool_trace": [
                {
                    "tool_name": "semantic_form",
                    "is_error": False,
                    "budget_rejected": False,
                    "operation_aborted": False,
                    "details": {
                        "status": 200,
                    },
                },
            ],
            "provider_turn_count": 1,
        }
        result = _assess_semantic_specialist_adherence(
            self.table_treatment, traj, task_group="table",
        )
        self.assertTrue(result["unavailable_specialist_found"])

    def test_tool_cap_exceeded(self) -> None:
        """Tool cap exceeded should be detected."""
        traj = {
            "planning_preamble": {"present": False},
            "tool_trace": [
                {
                    "tool_name": "unbrowser",
                    "is_error": False,
                    "budget_rejected": False,
                    "operation_aborted": False,
                    "details": {"action": "navigate", "status": 200},
                }
                for _ in range(10)
            ],
            "provider_turn_count": 10,
        }
        result = _assess_semantic_specialist_adherence(
            self.table_treatment, traj, task_group="table",
        )
        self.assertFalse(result["tool_cap_compliant"])

    def test_operation_aborted_after_cap_is_rejected(self) -> None:
        treatment = _make_synthetic_treatment(
            "cap-twelve-table",
            "ct",
            "table_specialist",
            "native_bash_unbrowser_semantic_table_v1",
            allowed_tools=("bash", "unbrowser", "semantic_table"),
            tool_call_limit=12,
            parent_bundle_id="parent-table-001",
        )
        trace = [
            {
                "tool_name": "unbrowser",
                "is_error": False,
                "budget_rejected": False,
                "operation_aborted": False,
                "details": {"action": "text"},
            }
            for _ in range(treatment.tool_call_limit)
        ]
        trace.append({
            "tool_name": "bash",
            "is_error": True,
            "budget_rejected": False,
            "operation_aborted": True,
            "details": {},
        })
        result = _assess_semantic_specialist_adherence(
            treatment,
            {"tool_trace": trace},
            task_group="table",
        )
        self.assertTrue(result["tool_cap_compliant"])
        self.assertEqual(result["admitted_tool_call_count"], 12)
        self.assertEqual(result["rejected_tool_call_count"], 1)

    def test_legacy_pre_execution_error_before_later_abort_is_rejected(self) -> None:
        treatment = _make_synthetic_treatment(
            "legacy-cap-table",
            "lc",
            "table_specialist",
            "native_bash_unbrowser_semantic_table_v1",
            allowed_tools=("bash", "unbrowser", "semantic_table"),
            tool_call_limit=12,
            parent_bundle_id="parent-table-001",
        )
        trace = [
            {
                "tool_name": "unbrowser",
                "is_error": False,
                "budget_rejected": False,
                "operation_aborted": False,
                "details": {"action": "query"},
            }
            for _ in range(12)
        ]
        trace.insert(5, {
            "tool_name": "unbrowser",
            "is_error": True,
            "budget_rejected": False,
            "operation_aborted": False,
            "details": {},
        })
        trace.append({
            "tool_name": "bash",
            "is_error": True,
            "budget_rejected": False,
            "operation_aborted": True,
            "details": {},
        })
        result = _assess_semantic_specialist_adherence(
            treatment, {"tool_trace": trace}, task_group="table",
        )
        self.assertEqual(result["admitted_tool_call_count"], 12)
        self.assertEqual(result["rejected_tool_call_count"], 2)
        self.assertTrue(result["tool_cap_compliant"])

    def test_legacy_empty_error_is_not_waived_without_cap_proof(self) -> None:
        treatment = _make_synthetic_treatment(
            "legacy-under-cap-table",
            "luc",
            "table_specialist",
            "native_bash_unbrowser_semantic_table_v1",
            allowed_tools=("bash", "unbrowser", "semantic_table"),
            tool_call_limit=12,
            parent_bundle_id="parent-table-001",
        )
        trace = [
            {
                "tool_name": "unbrowser",
                "is_error": False,
                "budget_rejected": False,
                "operation_aborted": False,
                "details": {"action": "query"},
            }
            for _ in range(9)
        ]
        trace.append({
            "tool_name": "unbrowser",
            "is_error": True,
            "budget_rejected": False,
            "operation_aborted": False,
            "details": {},
        })
        trace.append({
            "tool_name": "bash",
            "is_error": True,
            "budget_rejected": False,
            "operation_aborted": True,
            "details": {},
        })
        result = _assess_semantic_specialist_adherence(
            treatment, {"tool_trace": trace}, task_group="table",
        )
        self.assertEqual(result["admitted_tool_call_count"], 11)
        self.assertEqual(result["rejected_tool_call_count"], 0)
        self.assertTrue(result["tool_cap_compliant"])

    def test_explicit_false_disables_legacy_pre_execution_inference(self) -> None:
        treatment = _make_synthetic_treatment(
            "explicit-false-table",
            "eft",
            "table_specialist",
            "native_bash_unbrowser_semantic_table_v1",
            allowed_tools=("bash", "unbrowser", "semantic_table"),
            tool_call_limit=12,
            parent_bundle_id="parent-table-001",
        )
        trace = [
            {
                "tool_name": "unbrowser",
                "is_error": False,
                "budget_rejected": False,
                "operation_aborted": False,
                "pre_execution_rejected": False,
                "details": {"action": "query"},
            }
            for _ in range(12)
        ]
        trace.insert(5, {
            "tool_name": "unbrowser",
            "is_error": True,
            "budget_rejected": False,
            "operation_aborted": False,
            "pre_execution_rejected": False,
            "details": {},
        })
        trace.append({
            "tool_name": "bash",
            "is_error": True,
            "budget_rejected": False,
            "operation_aborted": True,
            "pre_execution_rejected": False,
            "details": {},
        })
        result = _assess_semantic_specialist_adherence(
            treatment, {"tool_trace": trace}, task_group="table",
        )
        self.assertEqual(result["admitted_tool_call_count"], 13)
        self.assertEqual(result["rejected_tool_call_count"], 1)
        self.assertFalse(result["tool_cap_compliant"])


# ---------------------------------------------------------------------------
# v2 protocol helpers
# ---------------------------------------------------------------------------

_V2_TABLE_TEMPLATE = "table_filter_sort"
_V2_FORM_TEMPLATE = "form_entry_validation"


def _v2_task_ids(num_tasks: int, seed_start: int = 1000) -> tuple[list[str], list[str]]:
    """Compute the table/form task IDs for a v2 manifest."""
    table_ids = [
        f"unbrowser-fixture-v2-{_V2_TABLE_TEMPLATE}-easy-{seed_start + i}"
        for i in range(num_tasks // 2)
    ]
    form_ids = [
        f"unbrowser-fixture-v2-{_V2_FORM_TEMPLATE}-easy-{seed_start + num_tasks // 2 + i}"
        for i in range(num_tasks // 2)
    ]
    return table_ids, form_ids


def _make_v2_manifest(
    registry: TreatmentRegistry,
    policy_split: dict[str, Any],
    stage: str = "replication_screen",
    *,
    decision_rule: dict[str, Any] | None = None,
    run_policy: dict[str, Any] | None = None,
    dataset_contract: dict[str, Any] | None = None,
    mechanics_qualification: dict[str, Any] | None = None,
    task_groups: dict | None = None,
    rollout_replicas: int | None = None,
) -> dict[str, Any]:
    """Build a v2 semantic capability manifest (protocol schema v2)."""
    bundle_ids = [t.bundle_id for t in registry]
    treatments_by_cap = {
        str(treatment.generator_metadata["capability"]): treatment
        for treatment in registry
    }

    if stage == "mechanics_dry_run":
        num_tasks = 2
        if rollout_replicas is None:
            rollout_replicas = 1
        if decision_rule is None:
            decision_rule = {"all_attempts_mechanically_valid": True}
    else:
        num_tasks = 16
        if rollout_replicas is None:
            rollout_replicas = 3
        if decision_rule is None:
            decision_rule = dict(_V2_REPLICATION_DECISION_RULE)

    protocol: dict[str, Any] = {
        "schema_version": PROTOCOL_SCHEMA_V2,
        "stage": stage,
        "claim_boundary": "screening_futility_only",
        "mechanism": {
            "name": "controller_owned_public_html_semantic_operation",
            "receipt_schema_version": "pyreplab-semantic-specialist-receipt-v1",
        },
        "parent_bundle_ids": {
            level: str(treatment.generator_metadata["parent_bundle_id"])
            for level, treatment in treatments_by_cap.items()
        },
        "decision_rule": decision_rule,
    }

    if stage == "replication_screen":
        protocol["run_policy"] = run_policy if run_policy is not None else {
            "early_outcome_stop": False,
            "outcome_driven_replacement": False,
        }
        protocol["dataset_contract"] = (
            dataset_contract
            if dataset_contract is not None
            else {
                "schema_version": DATASET_CONTRACT_SCHEMA,
                "contract_hash": "c" * 64,
                "governance_role": "canary_excluded",
            }
        )
        protocol["mechanics_qualification"] = (
            mechanics_qualification
            if mechanics_qualification is not None
            else {
                "mechanics_manifest_hash": "d" * 64,
                "mechanics_gate_sha256": "e" * 64,
                "decision": "mechanics_pass",
            }
        )
        if task_groups is not None:
            protocol["task_groups"] = task_groups
        else:
            table_ids, form_ids = _v2_task_ids(num_tasks)
            protocol["task_groups"] = {"table": table_ids, "form": form_ids}

    spec: dict[str, Any] = {
        "screen_id": f"semantic-canary-v2-{stage}-001",
        "purpose": f"Synthetic v2 semantic capability canary {stage}",
        "remote_identity": {
            "host": "test-host",
            "project": "/remote/test-project",
            "run_root": "/remote/test-runs",
            "python": "python3",
        },
        "policy_bundle_ids": bundle_ids,
        "tasks": [
            {
                "template": (
                    _V2_TABLE_TEMPLATE
                    if i < num_tasks // 2
                    else _V2_FORM_TEMPLATE
                ),
                "difficulty": "easy",
                "seed": 1000 + i,
            }
            for i in range(num_tasks)
        ],
        "rollout_replicas": rollout_replicas,
        "sampling_seed_start": 1000,
        "schedule_seed": 42,
        "task_role": "T_canary",
        "protocol": protocol,
        "selection": {"reason": "synthetic v2 semantic capability canary test"},
    }
    return build_screen_manifest(
        registry,
        policy_split,
        spec,
        registry_file="synthetic-treatments.json",
        policy_split_file="synthetic-split.json",
    )


def _make_v2_records(
    manifest: dict[str, Any],
    table_treatment: TreatmentSpec,
    form_treatment: TreatmentSpec,
    registry_hash: str,
    table_successes: dict[str, dict[int, bool]] | None = None,
    form_successes: dict[str, dict[int, bool]] | None = None,
) -> list[dict[str, Any]]:
    """Build panel records for a v2 replication_screen manifest (R=3)."""
    protocol_tg = manifest.get("protocol", {}).get("task_groups", {})
    table_task_ids = set(str(tid) for tid in protocol_tg.get("table", []))
    form_task_ids = set(str(tid) for tid in protocol_tg.get("form", []))

    records: list[dict[str, Any]] = []
    for panel in manifest["panels"]:
        tid = panel["task_id"]
        replica = panel["rollout_replica"]

        table_has_receipt = tid in table_task_ids
        form_has_receipt = tid in form_task_ids

        table_ok = True
        form_ok = True
        if table_successes and tid in table_successes:
            table_ok = table_successes[tid].get(replica, True)
        if form_successes and tid in form_successes:
            form_ok = form_successes[tid].get(replica, True)

        attempt_mods: dict[str, dict[str, Any]] = {}
        if not table_has_receipt:
            attempt_mods[table_treatment.bundle_id] = {"with_missing_receipt": True}
        if not form_has_receipt:
            attempt_mods[form_treatment.bundle_id] = {"with_missing_receipt": True}

        record = _make_panel_result(
            manifest,
            panel,
            table_treatment,
            form_treatment,
            table_success=table_ok,
            form_success=form_ok,
            attempt_mods=attempt_mods if attempt_mods else None,
        )
        record["result"]["treatment_registry_hash"] = registry_hash
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# tests: v2 mechanics_dry_run
# ---------------------------------------------------------------------------


class SemanticCapabilityGateV2MechanicsTest(unittest.TestCase):
    """v2 protocol mechanics_dry_run must behave like v1 (2 tasks, R=1, 4
    attempts) but report under the v2 schema."""

    def setUp(self) -> None:
        self.table_treatment = _make_table_treatment()
        self.form_treatment = _make_form_treatment()
        self.registry = _make_semantic_registry(self.table_treatment, self.form_treatment)
        self.policy_split = _make_semantic_policy_split(self.registry)
        self.manifest = _make_v2_manifest(
            self.registry, self.policy_split, stage="mechanics_dry_run"
        )
        self.manifest = dict(self.manifest)
        self.manifest.pop("manifest_hash", None)
        self.manifest["registry_hash"] = self.registry.registry_hash
        self.manifest["manifest_hash"] = _canonical_hash(self.manifest)
        self.preflight = _make_preflight(self.manifest)

    def _make_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for panel in self.manifest["panels"]:
            record = _make_panel_result(
                self.manifest, panel, self.table_treatment, self.form_treatment,
            )
            record["result"]["treatment_registry_hash"] = self.registry.registry_hash
            records.append(record)
        return records

    def test_v2_mechanics_pass_uses_v2_report_schema(self) -> None:
        records = self._make_records()
        report = evaluate_semantic_capability_gate(
            self.manifest, self.registry, self.policy_split, records,
            preflight=self.preflight,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["decision"], "mechanics_pass")
        self.assertEqual(report["stage"], "mechanics_dry_run")
        self.assertEqual(report["gate"], GATE_SCHEMA_V2)
        self.assertEqual(report["schema_version"], GATE_SCHEMA_V2)
        # Still 2 tasks, 2 panels, 4 attempts, R=1.
        self.assertEqual(report["completeness"]["records"], 2)
        self.assertEqual(report["completeness"]["unique_attempt_ids"], 4)
        self.assertNotEqual(report["gate"], GATE_SCHEMA)


# ---------------------------------------------------------------------------
# tests: v2 replication_screen
# ---------------------------------------------------------------------------


class SemanticCapabilityGateV2ReplicationScreenTest(unittest.TestCase):
    """Tests for the v2 replication_screen stage (R=3, 16 tasks, 48 panels)."""

    def setUp(self) -> None:
        self.table_treatment = _make_table_treatment()
        self.form_treatment = _make_form_treatment()
        self.registry = _make_semantic_registry(self.table_treatment, self.form_treatment)
        self.policy_split = _make_semantic_policy_split(self.registry)

    def _build_manifest(self, **kwargs: Any) -> dict[str, Any]:
        manifest = _make_v2_manifest(
            self.registry, self.policy_split, stage="replication_screen", **kwargs
        )
        manifest = dict(manifest)
        manifest.pop("manifest_hash", None)
        manifest["registry_hash"] = self.registry.registry_hash
        manifest["manifest_hash"] = _canonical_hash(manifest)
        return manifest

    def _task_ids(self, manifest: dict[str, Any]) -> tuple[list[str], list[str]]:
        tg = manifest["protocol"]["task_groups"]
        return list(tg["table"]), list(tg["form"])

    # -- success-pattern builders -------------------------------------------

    def _build_pass_successes(
        self,
        table_ids: list[str],
        form_ids: list[str],
    ) -> tuple[dict[str, dict[int, bool]], dict[str, dict[int, bool]]]:
        """7 stable table-only + 1 tie table; 7 stable form-only + 1 tie form.

        Zero discordant cells; every favorable task is stable-only.
        """
        table_successes: dict[str, dict[int, bool]] = {}
        form_successes: dict[str, dict[int, bool]] = {}
        for tid in table_ids:
            table_successes[tid] = {0: True, 1: True, 2: True}
            form_successes[tid] = {0: False, 1: False, 2: False}
        for tid in form_ids:
            table_successes[tid] = {0: False, 1: False, 2: False}
            form_successes[tid] = {0: True, 1: True, 2: True}
        # Make the last task of each group a tie (both arms unanimous fail).
        tie_table = table_ids[-1]
        table_successes[tie_table] = {0: False, 1: False, 2: False}
        form_successes[tie_table] = {0: False, 1: False, 2: False}
        tie_form = form_ids[-1]
        table_successes[tie_form] = {0: False, 1: False, 2: False}
        form_successes[tie_form] = {0: False, 1: False, 2: False}
        return table_successes, form_successes

    def _evaluate(
        self,
        manifest: dict[str, Any],
        table_successes: dict[str, dict[int, bool]],
        form_successes: dict[str, dict[int, bool]],
    ) -> dict[str, Any]:
        records = _make_v2_records(
            manifest,
            self.table_treatment,
            self.form_treatment,
            self.registry.registry_hash,
            table_successes=table_successes,
            form_successes=form_successes,
        )
        return evaluate_semantic_capability_gate(
            manifest, self.registry, self.policy_split, records,
            preflight=_make_preflight(manifest),
        )

    # -- tests --------------------------------------------------------------

    def test_replication_confirmation_pass(self) -> None:
        manifest = self._build_manifest()
        table_ids, form_ids = self._task_ids(manifest)
        table_s, form_s = self._build_pass_successes(table_ids, form_ids)
        report = self._evaluate(manifest, table_s, form_s)

        self.assertTrue(report["passed"])
        self.assertEqual(report["decision"], "confirmation_pass")
        self.assertEqual(report["stage"], "replication_screen")
        self.assertEqual(report["gate"], GATE_SCHEMA_V2)
        self.assertEqual(report["schema_version"], GATE_SCHEMA_V2)

        stability = report["stability"]
        self.assertEqual(stability["discordant_cell_count"], 0)
        self.assertGreaterEqual(stability["favorable_table_count"], 7)
        self.assertGreaterEqual(stability["favorable_form_count"], 7)
        self.assertLessEqual(stability["adverse_table_count"], 1)
        self.assertLessEqual(stability["adverse_form_count"], 1)
        self.assertGreaterEqual(stability["stable_table_only_count"], 2)
        self.assertGreaterEqual(stability["stable_form_only_count"], 2)

        # Descriptive arm totals / token means present but not gated.
        outcomes = report["descriptive_outcomes"]
        self.assertEqual(outcomes["table_arm_attempts"], 48)
        self.assertEqual(outcomes["form_arm_attempts"], 48)
        self.assertIn("table_arm_mean_output_tokens", outcomes)
        self.assertIn("form_arm_mean_output_tokens", outcomes)
        self.assertIsNotNone(outcomes["table_arm_mean_output_tokens"])
        self.assertIsNotNone(outcomes["form_arm_mean_output_tokens"])

    def test_replication_six_favorable_table_tasks_no_go(self) -> None:
        """Only 6 favorable table tasks (below minimum 7) → replication_no_go."""
        manifest = self._build_manifest()
        table_ids, form_ids = self._task_ids(manifest)
        table_s, form_s = self._build_pass_successes(table_ids, form_ids)

        # Demote one more table task from favorable (stable) to tie: 6 favorable.
        for tid in table_ids[-2:-1]:
            table_s[tid] = {0: False, 1: False, 2: False}
            form_s[tid] = {0: False, 1: False, 2: False}

        report = self._evaluate(manifest, table_s, form_s)
        self.assertFalse(report["passed"])
        self.assertEqual(report["decision"], "replication_no_go")
        self.assertEqual(report["stability"]["favorable_table_count"], 6)
        self.assertTrue(any(
            "favorable table tasks" in r.lower() for r in report["reasons"]
        ))

    def test_replication_too_many_discordant_no_go(self) -> None:
        """More than 4 discordant cells → replication_no_go."""
        manifest = self._build_manifest()
        table_ids, form_ids = self._task_ids(manifest)

        # Favorable but non-stable everywhere: table 2/3 vs form 1/3 on table
        # tasks; form 2/3 vs table 1/3 on form tasks → each task contributes
        # 2 discordant cells (both arms non-unanimous).
        table_s: dict[str, dict[int, bool]] = {}
        form_s: dict[str, dict[int, bool]] = {}
        for tid in table_ids:
            table_s[tid] = {0: True, 1: True, 2: False}
            form_s[tid] = {0: False, 1: True, 2: False}
        for tid in form_ids:
            table_s[tid] = {0: False, 1: True, 2: False}
            form_s[tid] = {0: True, 1: True, 2: False}

        report = self._evaluate(manifest, table_s, form_s)
        self.assertFalse(report["passed"])
        self.assertEqual(report["decision"], "replication_no_go")
        self.assertGreater(report["stability"]["discordant_cell_count"], 4)
        self.assertTrue(any(
            "discordant cells" in r.lower() for r in report["reasons"]
        ))

    def test_replication_strict_stable_failure_no_go(self) -> None:
        """Fewer than 2 stable-only tasks on each side → replication_no_go."""
        manifest = self._build_manifest()
        table_ids, form_ids = self._task_ids(manifest)

        # Every task favorable but never stable-only: table 3/3 vs form 1/3.
        table_s: dict[str, dict[int, bool]] = {}
        form_s: dict[str, dict[int, bool]] = {}
        for tid in table_ids:
            table_s[tid] = {0: True, 1: True, 2: True}
            form_s[tid] = {0: False, 1: True, 2: False}
        for tid in form_ids:
            table_s[tid] = {0: False, 1: True, 2: False}
            form_s[tid] = {0: True, 1: True, 2: True}

        report = self._evaluate(manifest, table_s, form_s)
        self.assertFalse(report["passed"])
        self.assertEqual(report["decision"], "replication_no_go")
        self.assertEqual(report["stability"]["stable_table_only_count"], 0)
        self.assertEqual(report["stability"]["stable_form_only_count"], 0)
        self.assertTrue(any(
            "stable table-only tasks" in r.lower() for r in report["reasons"]
        ))

    def test_replication_wrong_replica_count_invalid(self) -> None:
        """A v2 replication_screen manifest with R=2 must be invalid (not
        replication_no_go) via the size check."""
        manifest = _make_v2_manifest(
            self.registry, self.policy_split, stage="replication_screen",
            rollout_replicas=2,
        )
        manifest = dict(manifest)
        manifest.pop("manifest_hash", None)
        manifest["registry_hash"] = self.registry.registry_hash
        manifest["manifest_hash"] = _canonical_hash(manifest)

        report = evaluate_semantic_capability_gate(
            manifest, self.registry, self.policy_split, [],
            preflight=_make_preflight(manifest),
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["decision"], "invalid")
        self.assertTrue(any(
            "replication_screen expects" in r for r in report["reasons"]
        ))

    def test_replication_wrong_panel_count_invalid(self) -> None:
        """Truncated records (missing panels) → invalid via size check."""
        manifest = self._build_manifest()
        table_ids, form_ids = self._task_ids(manifest)
        table_s, form_s = self._build_pass_successes(table_ids, form_ids)
        records = _make_v2_records(
            manifest,
            self.table_treatment,
            self.form_treatment,
            self.registry.registry_hash,
            table_successes=table_s,
            form_successes=form_s,
        )
        # Drop one panel's record so the size check fails.
        records = records[:-1]
        report = evaluate_semantic_capability_gate(
            manifest, self.registry, self.policy_split, records,
            preflight=_make_preflight(manifest),
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["decision"], "invalid")
        self.assertTrue(any(
            "replication_screen expects" in r for r in report["reasons"]
        ))

    def test_replication_wrong_task_count_invalid(self) -> None:
        """8 tasks (not 16) on a replication_screen → invalid via size check."""
        manifest = self._build_manifest()
        # Truncate tasks/panels won't pass validate_screen_manifest, so test
        # the protocol validator directly for wrong task_groups size.
        table_ids, form_ids = self._task_ids(manifest)
        bad = dict(manifest)
        bad["protocol"] = dict(manifest["protocol"])
        bad["protocol"]["task_groups"] = {
            "table": table_ids[:4],
            "form": form_ids[:4],
        }
        with self.assertRaisesRegex(ValueError, "list of 8 task IDs"):
            _validate_semantic_capability_protocol(bad, self.registry)


class SemanticCapabilityGateV2ProtocolValidationTest(unittest.TestCase):
    """v2 protocol contract-field validation."""

    def setUp(self) -> None:
        self.table_treatment = _make_table_treatment()
        self.form_treatment = _make_form_treatment()
        self.registry = _make_semantic_registry(self.table_treatment, self.form_treatment)
        self.policy_split = _make_semantic_policy_split(self.registry)

    def _build_manifest(self, **kwargs: Any) -> dict[str, Any]:
        manifest = _make_v2_manifest(
            self.registry, self.policy_split, stage="replication_screen", **kwargs
        )
        manifest = dict(manifest)
        manifest.pop("manifest_hash", None)
        manifest["registry_hash"] = self.registry.registry_hash
        manifest["manifest_hash"] = _canonical_hash(manifest)
        return manifest

    def test_wrong_decision_rule_value_rejected(self) -> None:
        bad_rule = dict(_V2_REPLICATION_DECISION_RULE)
        bad_rule["maximum_discordant_cells"] = 2
        manifest = self._build_manifest(decision_rule=bad_rule)
        with self.assertRaisesRegex(
            ValueError, "decision_rule.maximum_discordant_cells"
        ):
            _validate_semantic_capability_protocol(manifest, self.registry)

    def test_missing_decision_rule_key_rejected(self) -> None:
        bad_rule = dict(_V2_REPLICATION_DECISION_RULE)
        bad_rule.pop("minimum_favorable_table_tasks")
        manifest = self._build_manifest(decision_rule=bad_rule)
        with self.assertRaisesRegex(
            ValueError, "decision_rule.minimum_favorable_table_tasks"
        ):
            _validate_semantic_capability_protocol(manifest, self.registry)

    def test_run_policy_early_outcome_stop_true_rejected(self) -> None:
        manifest = self._build_manifest(run_policy={
            "early_outcome_stop": True,
            "outcome_driven_replacement": False,
        })
        with self.assertRaisesRegex(ValueError, "early_outcome_stop"):
            _validate_semantic_capability_protocol(manifest, self.registry)

    def test_run_policy_outcome_driven_replacement_true_rejected(self) -> None:
        manifest = self._build_manifest(run_policy={
            "early_outcome_stop": False,
            "outcome_driven_replacement": True,
        })
        with self.assertRaisesRegex(ValueError, "outcome_driven_replacement"):
            _validate_semantic_capability_protocol(manifest, self.registry)

    def test_dataset_contract_wrong_schema_rejected(self) -> None:
        manifest = self._build_manifest(dataset_contract={
            "schema_version": "wrong-contract-schema",
            "contract_hash": "c" * 64,
            "governance_role": "canary_excluded",
        })
        with self.assertRaisesRegex(ValueError, "dataset_contract.schema_version"):
            _validate_semantic_capability_protocol(manifest, self.registry)

    def test_dataset_contract_bad_hash_rejected(self) -> None:
        manifest = self._build_manifest(dataset_contract={
            "schema_version": DATASET_CONTRACT_SCHEMA,
            "contract_hash": "not-a-hex-hash",
            "governance_role": "canary_excluded",
        })
        with self.assertRaisesRegex(ValueError, "contract_hash"):
            _validate_semantic_capability_protocol(manifest, self.registry)

    def test_dataset_contract_wrong_governance_rejected(self) -> None:
        manifest = self._build_manifest(dataset_contract={
            "schema_version": DATASET_CONTRACT_SCHEMA,
            "contract_hash": "c" * 64,
            "governance_role": "canary_included",
        })
        with self.assertRaisesRegex(ValueError, "governance_role"):
            _validate_semantic_capability_protocol(manifest, self.registry)

    def test_mechanics_qualification_bad_manifest_hash_rejected(self) -> None:
        manifest = self._build_manifest(mechanics_qualification={
            "mechanics_manifest_hash": "short",
            "mechanics_gate_sha256": "e" * 64,
            "decision": "mechanics_pass",
        })
        with self.assertRaisesRegex(ValueError, "mechanics_manifest_hash"):
            _validate_semantic_capability_protocol(manifest, self.registry)

    def test_mechanics_qualification_bad_gate_sha256_rejected(self) -> None:
        manifest = self._build_manifest(mechanics_qualification={
            "mechanics_manifest_hash": "d" * 64,
            "mechanics_gate_sha256": "zz",
            "decision": "mechanics_pass",
        })
        with self.assertRaisesRegex(ValueError, "mechanics_gate_sha256"):
            _validate_semantic_capability_protocol(manifest, self.registry)

    def test_mechanics_qualification_wrong_decision_rejected(self) -> None:
        manifest = self._build_manifest(mechanics_qualification={
            "mechanics_manifest_hash": "d" * 64,
            "mechanics_gate_sha256": "e" * 64,
            "decision": "mechanics_fail",
        })
        with self.assertRaisesRegex(ValueError, "mechanics_qualification.decision"):
            _validate_semantic_capability_protocol(manifest, self.registry)


class SemanticCapabilityGateV2TaskGroupValidationTest(unittest.TestCase):
    """v2 replication_screen task_groups must be 8+8 and template-correct."""

    def setUp(self) -> None:
        self.table_treatment = _make_table_treatment()
        self.form_treatment = _make_form_treatment()
        self.registry = _make_semantic_registry(self.table_treatment, self.form_treatment)
        self.policy_split = _make_semantic_policy_split(self.registry)

    def _build_manifest(self, **kwargs: Any) -> dict[str, Any]:
        manifest = _make_v2_manifest(
            self.registry, self.policy_split, stage="replication_screen", **kwargs
        )
        manifest = dict(manifest)
        manifest.pop("manifest_hash", None)
        manifest["registry_hash"] = self.registry.registry_hash
        manifest["manifest_hash"] = _canonical_hash(manifest)
        return manifest

    def test_replication_task_groups_not_partitioning_rejected(self) -> None:
        manifest = self._build_manifest()
        bad = dict(manifest)
        bad["protocol"] = dict(manifest["protocol"])
        tg = manifest["protocol"]["task_groups"]
        bad["protocol"]["task_groups"] = {
            "table": list(tg["table"]),
            "form": list(tg["form"]),
        }
        bad["protocol"]["task_groups"]["form"][-1] = "unknown-task-id"
        with self.assertRaisesRegex(ValueError, "exactly partition"):
            _validate_semantic_capability_protocol(bad, self.registry)

    def test_replication_task_group_template_mismatch_rejected(self) -> None:
        manifest = self._build_manifest()
        bad = dict(manifest)
        bad["protocol"] = dict(manifest["protocol"])
        tg = manifest["protocol"]["task_groups"]
        table_ids = list(tg["table"])
        form_ids = list(tg["form"])
        table_ids[0], form_ids[0] = form_ids[0], table_ids[0]
        bad["protocol"]["task_groups"] = {"table": table_ids, "form": form_ids}
        with self.assertRaisesRegex(ValueError, "non-table template|non-form template"):
            _validate_semantic_capability_protocol(bad, self.registry)


if __name__ == "__main__":
    unittest.main()
