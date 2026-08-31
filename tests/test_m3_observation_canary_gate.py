"""Tests for the observation-enforcement canary gate.

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
from pyreplab_harness.m3_observation_canary_gate import (
    GATE_SCHEMA,
    _is_infrastructure_error,
    _load_canary_records,
    _validate_canary_protocol,
    evaluate_observation_canary_gate,
)
from pyreplab_harness.orchestrator import policy_spec_from_treatment
from pyreplab_harness.treatments import TreatmentRegistry, TreatmentSpec

# ---------------------------------------------------------------------------
# synthetic treatment builders
# ---------------------------------------------------------------------------

_OBS_PAYLOAD_TEXT = {"text_content": "Hello, world!", "title": "Test Page"}
_OBS_PAYLOAD_STRUCT = {"elements": [{"tag": "div", "children": 5}], "format": "DOM"}


def _canonical_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _encode_obs_payload(payload: dict[str, Any]) -> tuple[int, str]:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return len(encoded), hashlib.sha256(encoded).hexdigest()


def _make_synthetic_treatment(
    bundle_id: str,
    bundle_hash_prefix: str,
    observation: str,
    tool_interface: str,
    tool_call_limit: int = 6,
    parent_bundle_id: str = "",
) -> TreatmentSpec:
    """Build a synthetic TreatmentSpec for the observation canary."""
    bundle_hash = bundle_hash_prefix + "a" * (64 - len(bundle_hash_prefix))
    return TreatmentSpec(
        id=bundle_id,
        version="2",
        system_prompt=(
            f"Planning: direct\n"
            f"Observation: {observation}\n"
            f"Verification: submit_directly\n"
            f"Recovery: fail_fast\n"
            f"Safety: Workspace only.\n"
        ),
        allowed_tools=("bash", "unbrowser"),
        max_output_tokens=4096,
        tool_call_limit=tool_call_limit,
        command_timeout_seconds=60,
        wall_time_limit_seconds=600,
        tool_interface=tool_interface,
        generator_metadata={
            "grammar_name": "canary_test",
            "grammar_version": "m3-canary-v1",
            "planning": "direct",
            "observation": observation,
            "verification": "submit_directly",
            "recovery": "fail_fast",
            "tool_cap": "lean" if tool_call_limit <= 6 else "expanded",
            "parent_bundle_id": parent_bundle_id,
            "observation_mechanism": "auto_delivered_first_observation",
        },
    )


def _make_text_treatment(parent_bundle_id: str = "parent-text-001") -> TreatmentSpec:
    return _make_synthetic_treatment(
        bundle_id="canary-text-first",
        bundle_hash_prefix="tx",
        observation="text_first",
        tool_interface="native_bash_unbrowser_interactive_text_first_v1",
        parent_bundle_id=parent_bundle_id,
    )


def _make_structure_treatment(parent_bundle_id: str = "parent-struct-001") -> TreatmentSpec:
    return _make_synthetic_treatment(
        bundle_id="canary-structure-first",
        bundle_hash_prefix="sx",
        observation="structure_first",
        tool_interface="native_bash_unbrowser_interactive_structure_first_v1",
        parent_bundle_id=parent_bundle_id,
    )


def _make_canary_registry(
    text_t: TreatmentSpec | None = None,
    struct_t: TreatmentSpec | None = None,
) -> TreatmentRegistry:
    """Build a synthetic registry with the two canary treatments."""
    t1 = text_t or _make_text_treatment()
    t2 = struct_t or _make_structure_treatment()
    treatments = [t1, t2]
    registry = TreatmentRegistry(treatments)
    return registry


def _make_canary_policy_split(
    registry: TreatmentRegistry,
) -> dict[str, Any]:
    """Build a synthetic policy split with both treatments in meta_train."""
    bundle_ids = [t.bundle_id for t in registry]
    payload = {
        "grammar_name": "canary_test",
        "grammar_version": "m3-canary-v1",
        "policy_version": "2",
        "registry_file": "synthetic-treatments.json",
        "registry_hash": registry.registry_hash,
        "schema_version": "m3-policy-split-v1",
        "split_algorithm": "canary-test-v1",
        "split_seed": 42,
        "splits": {
            "development": [],
            "final_held_out": [],
            "meta_train": list(bundle_ids),
        },
    }
    payload["manifest_hash"] = _canonical_hash(payload)
    return payload


def _make_canary_manifest(
    registry: TreatmentRegistry,
    policy_split: dict[str, Any],
    stage: str = "mechanics_dry_run",
    num_tasks: int = 2,
    rollout_replicas: int = 1,
    schedule_seed: int = 42,
    sampling_seed_start: int = 1000,
) -> dict[str, Any]:
    """Build a canary manifest using the exploratory screen builder."""
    bundle_ids = [t.bundle_id for t in registry]
    decision_rule = (
        {"all_attempts_mechanically_valid": True}
        if stage == "mechanics_dry_run"
        else {
            "maximum_discordant_cells": 1,
            "minimum_stable_text_only_tasks": 1,
            "minimum_stable_structure_only_tasks": 1,
        }
    )
    treatments_by_observation = {
        str(treatment.generator_metadata["observation"]): treatment
        for treatment in registry
    }
    spec: dict[str, Any] = {
        "screen_id": f"canary-{stage}-001",
        "purpose": f"Synthetic canary {stage}",
        "remote_identity": {
            "host": "test-host",
            "project": "/remote/test-project",
            "run_root": "/remote/test-runs",
            "python": "python3",
        },
        "policy_bundle_ids": bundle_ids,
        "tasks": [
            {"template": "single_page_extraction", "difficulty": "easy", "seed": 1000 + i}
            for i in range(num_tasks)
        ],
        "rollout_replicas": rollout_replicas,
        "sampling_seed_start": sampling_seed_start,
        "schedule_seed": schedule_seed,
        "task_role": "T_canary",
        "protocol": {
            "schema_version": "m3-observation-canary-protocol-v1",
            "stage": stage,
            "claim_boundary": "screening_futility_only",
            "mechanism": {
                "name": "auto_delivered_first_observation",
                "receipt_schema_version": "pyreplab-required-first-observation-v1",
                "combined_navigation_observation_tool_call": True,
                "text_selector": "body",
                "later_cross_modal_observations_allowed": True,
            },
            "parent_bundle_ids": {
                level: str(treatment.generator_metadata["parent_bundle_id"])
                for level, treatment in treatments_by_observation.items()
            },
            "decision_rule": decision_rule,
        },
        "selection": {"reason": "synthetic canary test"},
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


def _make_navigate_trace_entry(
    action: str,
    obs_payload: dict[str, Any],
) -> dict[str, Any]:
    """Build a navigate unbrowser trace entry with a valid first-observation receipt."""
    payload_bytes, payload_sha256 = _encode_obs_payload(obs_payload)
    selector = "body" if action == "text" else None
    return {
        "tool_name": "unbrowser",
        "is_error": False,
        "budget_rejected": False,
        "operation_aborted": False,
        "details": {
            "action": "navigate",
            "status": 200,
            "url": "http://127.0.0.1:PORT/single_page_extraction/0/easy/page_0",
            "required_first_observation_receipt": {
                "schema_version": "pyreplab-required-first-observation-v1",
                "mechanism": "auto_delivered_first_observation",
                "delivered": True,
                "delivered_action": action,
                "required_action": action,
                "selector": selector,
                "payload_bytes": payload_bytes,
                "payload_sha256": payload_sha256,
            },
            "auto_delivered_observation": obs_payload,
        },
    }


def _make_later_observation_entry(
    action: str,
) -> dict[str, Any]:
    """Build a later cross-modal observation entry (should not invalidate receipt)."""
    return {
        "tool_name": "unbrowser",
        "is_error": False,
        "budget_rejected": False,
        "operation_aborted": False,
        "details": {
            "action": action,
            "status": 200,
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
    with_later_observations: bool = False,
    with_missing_receipt: bool = False,
) -> dict[str, Any]:
    """Build one synthetic attempt for a canary treatment.

    If *with_missing_receipt* is True, the navigate entry will not have a
    required_first_observation_receipt (invalid mechanics).

    *manifest* is required so verifier identity and sampling parameters are
    taken directly from the frozen runtime pins.
    """
    obs_level = str(treatment.generator_metadata.get("observation", ""))
    action = "text" if obs_level == "text_first" else "blockmap"
    obs_payload = _OBS_PAYLOAD_TEXT if action == "text" else _OBS_PAYLOAD_STRUCT

    runtime_pins = manifest["runtime_pins"]

    tool_trace: list[dict[str, Any]] = []
    if with_missing_receipt:
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
        tool_trace.append(_make_navigate_trace_entry(action, obs_payload))

    if with_later_observations:
        later_action = "blockmap" if action == "text" else "text"
        tool_trace.append(_make_later_observation_entry(later_action))

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
            "seed": 0,  # will be patched to panel seed by _make_panel_result
            "parameters": runtime_pins["sampling"]["parameters"],
        },
        "timing": {},
    }


def _make_panel_result(
    manifest: dict[str, Any],
    panel: dict[str, Any],
    text_treatment: TreatmentSpec,
    structure_treatment: TreatmentSpec,
    text_success: bool = True,
    struct_success: bool = True,
    *,
    attempt_mods: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a valid panel result record."""
    task = next(t for t in manifest["tasks"] if t["task_id"] == panel["task_id"])
    attempts: dict[str, dict[str, Any]] = {}
    for bid in panel["execution_order"]:
        if bid == text_treatment.bundle_id:
            treatment = text_treatment
            base_success = text_success
        else:
            treatment = structure_treatment
            base_success = struct_success

        mods = (attempt_mods or {}).get(bid, {})
        att = _make_attempt(treatment, manifest=manifest, success=base_success, **mods)
        # Patch sampling_receipt seed to match panel
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
            "treatment_registry_hash": _canonical_hash({}),  # Will be patched
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
# tests
# ---------------------------------------------------------------------------


class ObservationCanaryGateMechanicsTest(unittest.TestCase):
    """Tests for mechanics_dry_run stage."""

    def setUp(self) -> None:
        self.text_treatment = _make_text_treatment()
        self.structure_treatment = _make_structure_treatment()
        self.registry = _make_canary_registry(self.text_treatment, self.structure_treatment)
        self.policy_split = _make_canary_policy_split(self.registry)
        self.manifest = _make_canary_manifest(
            self.registry, self.policy_split, stage="mechanics_dry_run", num_tasks=2
        )
        # Patch the manifest registry_hash to match
        self.manifest = dict(self.manifest)
        self.manifest.pop("manifest_hash", None)
        self.manifest["registry_hash"] = self.registry.registry_hash
        self.manifest["manifest_hash"] = _canonical_hash(self.manifest)
        self.preflight = _make_preflight(self.manifest)

    def _make_records(self, attempt_mods: dict | None = None) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for panel in self.manifest["panels"]:
            record = _make_panel_result(
                self.manifest,
                panel,
                self.text_treatment,
                self.structure_treatment,
                attempt_mods=attempt_mods,
            )
            # Patch registry hash
            record["result"]["treatment_registry_hash"] = self.registry.registry_hash
            records.append(record)
        return records

    def test_mechanics_pass(self) -> None:
        """Complete mechanics_dry_run with all checks passing."""
        records = self._make_records()
        report = evaluate_observation_canary_gate(
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

    def test_report_is_deterministic(self) -> None:
        records = self._make_records()
        r1 = evaluate_observation_canary_gate(
            self.manifest, self.registry, self.policy_split, records,
            preflight=self.preflight,
        )
        r2 = evaluate_observation_canary_gate(
            self.manifest, self.registry, self.policy_split, records,
            preflight=self.preflight,
        )
        self.assertEqual(r1, r2)

    def test_missing_first_observation_receipt_fails(self) -> None:
        """Attempts without a first-observation receipt should fail mechanics."""
        records = self._make_records(attempt_mods={
            self.text_treatment.bundle_id: {"with_missing_receipt": True},
            self.structure_treatment.bundle_id: {"with_missing_receipt": True},
        })
        report = evaluate_observation_canary_gate(
            self.manifest, self.registry, self.policy_split, records,
            preflight=self.preflight,
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["decision"], "invalid")
        self.assertTrue(any("receipt" in r.lower() for r in report["reasons"]))

    def test_wrong_observation_receipt_action_fails(self) -> None:
        """Receipt with mismatched action should fail."""
        records: list[dict[str, Any]] = []
        for panel in self.manifest["panels"]:
            record = _make_panel_result(
                self.manifest, panel, self.text_treatment, self.structure_treatment,
            )
            record["result"]["treatment_registry_hash"] = self.registry.registry_hash
            # Corrupt text treatment: swap receipt action
            for bid, att in record["result"]["attempts"].items():
                if bid == self.text_treatment.bundle_id:
                    nav = att["trajectory"]["tool_trace"][0]
                    nav["details"]["required_first_observation_receipt"]["delivered_action"] = "blockmap"
                    nav["details"]["required_first_observation_receipt"]["required_action"] = "blockmap"
                    nav["details"]["required_first_observation_receipt"]["selector"] = None
            records.append(record)
        report = evaluate_observation_canary_gate(
            self.manifest, self.registry, self.policy_split, records,
            preflight=self.preflight,
        )
        self.assertFalse(report["passed"])
        self.assertTrue(any(
            "observation" in r.lower() and "adherent" in r.lower()
            for r in report["reasons"]
        ))

    def test_policy_identity_mismatch_fails(self) -> None:
        """Executed policy not matching registry treatment should fail."""
        records: list[dict[str, Any]] = []
        for panel in self.manifest["panels"]:
            record = _make_panel_result(
                self.manifest, panel, self.text_treatment, self.structure_treatment,
            )
            record["result"]["treatment_registry_hash"] = self.registry.registry_hash
            # Corrupt policy identity on text treatment
            for bid, att in record["result"]["attempts"].items():
                if bid == self.text_treatment.bundle_id:
                    att["policy"] = {"id": "wrong", "version": "99", "tool_interface": "bad"}
            records.append(record)
        report = evaluate_observation_canary_gate(
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
                self.manifest, panel, self.text_treatment, self.structure_treatment,
            )
            record["result"]["treatment_registry_hash"] = self.registry.registry_hash
            # Corrupt sampling receipt on text treatment
            for bid, att in record["result"]["attempts"].items():
                if bid == self.text_treatment.bundle_id:
                    att["sampling_receipt"]["seed"] = 99999
                    att["trajectory"]["provider_turn_count"] = 2
            records.append(record)
        report = evaluate_observation_canary_gate(
            self.manifest, self.registry, self.policy_split, records,
            preflight=self.preflight,
        )
        self.assertFalse(report["passed"])
        self.assertTrue(any("sampling" in r.lower() for r in report["reasons"]))

    def test_tool_cap_failure(self) -> None:
        """Exceeding tool call limit should be caught by adherence check."""
        records: list[dict[str, Any]] = []
        for panel in self.manifest["panels"]:
            record = _make_panel_result(
                self.manifest, panel, self.text_treatment, self.structure_treatment,
            )
            record["result"]["treatment_registry_hash"] = self.registry.registry_hash
            # Add excessive tool calls
            for att in record["result"]["attempts"].values():
                extra_calls = [
                    {
                        "tool_name": "unbrowser",
                        "is_error": False,
                        "budget_rejected": False,
                        "operation_aborted": False,
                        "details": {"action": "text", "status": 200},
                    }
                    for _ in range(10)
                ]
                att["trajectory"]["tool_trace"][1:1] = extra_calls
            records.append(record)
        report = evaluate_observation_canary_gate(
            self.manifest, self.registry, self.policy_split, records,
            preflight=self.preflight,
        )
        self.assertFalse(report["passed"])
        self.assertTrue(any("tool_cap" in r.lower() for r in report["reasons"]))

    def test_preflight_identity_missing_fails(self) -> None:
        """Missing preflight should fail."""
        records = self._make_records()
        report = evaluate_observation_canary_gate(
            self.manifest, self.registry, self.policy_split, records,
            preflight=None,
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["decision"], "invalid")
        self.assertFalse(report["checks"]["preflight_identity_valid"])
        self.assertTrue(any("preflight" in reason for reason in report["reasons"]))

    def test_completed_attempt_with_browser_death_is_invalid(self) -> None:
        records = self._make_records()
        attempt = next(iter(records[0]["result"]["attempts"].values()))
        attempt["trajectory"]["tool_trace"].append({
            "tool_name": "unbrowser",
            "is_error": False,
            "budget_rejected": False,
            "operation_aborted": False,
            "details": {
                "error": "UnbrowserProtocolError: unbrowser process connection broken (exit_code=124)",
                "infrastructure_error": True,
            },
        })
        report = evaluate_observation_canary_gate(
            self.manifest, self.registry, self.policy_split, records,
            preflight=self.preflight,
        )
        self.assertEqual(report["decision"], "invalid")
        self.assertEqual(report["completeness"]["infrastructure_errors"], 1)

    def test_legacy_connection_broken_text_is_infrastructure_error(self) -> None:
        self.assertTrue(_is_infrastructure_error({
            "details": {
                "error": "UnbrowserProtocolError: unbrowser process connection broken (exit_code=124)"
            },
        }))

    def test_explicit_non_infrastructure_overrides_legacy_text(self) -> None:
        self.assertFalse(_is_infrastructure_error({
            "details": {
                "error": "UnbrowserProtocolError: browser process exited",
                "infrastructure_error": False,
            },
        }))

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
                _load_canary_records(path, self.manifest)

    def test_duplicate_attempt_ids_fail(self) -> None:
        """Duplicate attempt IDs should produce structural errors."""
        records = self._make_records()
        # Make all attempts share the same ID
        for record in records:
            for att in record["result"]["attempts"].values():
                att["attempt_id"] = "duplicate-id"
        report = evaluate_observation_canary_gate(
            self.manifest, self.registry, self.policy_split, records,
            preflight=self.preflight,
        )
        self.assertFalse(report["passed"])
        self.assertTrue(any("duplicate" in r.lower() for r in report["reasons"]))

    def test_wrong_stage_in_manifest_fails(self) -> None:
        """Protocol with wrong stage is rejected."""
        manifest = _make_canary_manifest(
            self.registry, self.policy_split, stage="wrong_stage"
        )
        manifest = dict(manifest)
        manifest.pop("manifest_hash", None)
        manifest["registry_hash"] = self.registry.registry_hash
        manifest["manifest_hash"] = _canonical_hash(manifest)
        records = self._make_records()
        with self.assertRaises(ValueError) as ctx:
            evaluate_observation_canary_gate(
                manifest, self.registry, self.policy_split, records,
                preflight=_make_preflight(manifest),
            )
        self.assertIn("stage", str(ctx.exception))

    def test_wrong_task_role_rejected(self) -> None:
        """Manifest without T_canary role should fail protocol validation."""
        spec: dict[str, Any] = {
            "screen_id": "wrong-role-001",
            "purpose": "Wrong role test",
            "remote_identity": {
                "host": "test-host",
                "project": "/remote/test-project",
                "run_root": "/remote/test-runs",
                "python": "python3",
            },
            "policy_bundle_ids": [self.text_treatment.bundle_id, self.structure_treatment.bundle_id],
            "tasks": [
                {"template": "single_page_extraction", "difficulty": "easy", "seed": 1},
                {"template": "table_filter_sort", "difficulty": "medium", "seed": 2},
            ],
            "rollout_replicas": 1,
            "sampling_seed_start": 1000,
            "schedule_seed": 42,
            "task_role": "T_pilot",  # wrong for canary gate
            "protocol": {
                "schema_version": "m3-observation-canary-protocol-v1",
                "stage": "mechanics_dry_run",
                "decision_rule": {
                    "maximum_discordant_cells": 1,
                    "minimum_stable_text_only_tasks": 1,
                    "minimum_stable_structure_only_tasks": 1,
                },
            },
            "selection": {"reason": "test"},
        }
        # _validate_canary_protocol should raise ValueError for non-T_canary
        with self.assertRaises(ValueError) as ctx:
            _validate_canary_protocol(
                {"task_role": "T_pilot",
                 "policy_bundle_ids": [self.text_treatment.bundle_id, self.structure_treatment.bundle_id],
                 "protocol": {"schema_version": "m3-observation-canary-protocol-v1", "stage": "mechanics_dry_run"}},
                self.registry,
            )
        self.assertIn("T_canary", str(ctx.exception))


class ObservationCanaryGateOutcomeScreenTest(unittest.TestCase):
    """Tests for outcome_screen stage."""

    def setUp(self) -> None:
        self.text_treatment = _make_text_treatment()
        self.structure_treatment = _make_structure_treatment()
        self.registry = _make_canary_registry(self.text_treatment, self.structure_treatment)
        self.policy_split = _make_canary_policy_split(self.registry)
        self.manifest = _make_canary_manifest(
            self.registry, self.policy_split, stage="outcome_screen",
            num_tasks=6, rollout_replicas=2,
        )
        self.manifest = dict(self.manifest)
        self.manifest.pop("manifest_hash", None)
        self.manifest["registry_hash"] = self.registry.registry_hash
        self.manifest["manifest_hash"] = _canonical_hash(self.manifest)
        self.preflight = _make_preflight(self.manifest)

    def _make_outcome_records(
        self,
        text_successes: dict[int, dict[int, bool]] | None = None,
        struct_successes: dict[int, dict[int, bool]] | None = None,
    ) -> list[dict[str, Any]]:
        """Build records for outcome_screen with specified per-task-per-replica
        success patterns.

        *text_successes*: {task_index: {replica: success_bool}}
        """
        records: list[dict[str, Any]] = []
        tasks = self.manifest["tasks"]
        task_by_id = {t["task_id"]: t for t in tasks}
        task_indices = {t["task_id"]: i for i, t in enumerate(tasks)}

        for panel in self.manifest["panels"]:
            tid = panel["task_id"]
            replica = panel["rollout_replica"]
            t_idx = task_indices[tid]

            text_ok = True
            struct_ok = True
            if text_successes and t_idx in text_successes:
                text_ok = text_successes[t_idx].get(replica, True)
            if struct_successes and t_idx in struct_successes:
                struct_ok = struct_successes[t_idx].get(replica, True)

            record = _make_panel_result(
                self.manifest, panel, self.text_treatment, self.structure_treatment,
                text_success=text_ok, struct_success=struct_ok,
            )
            record["result"]["treatment_registry_hash"] = self.registry.registry_hash
            records.append(record)
        return records

    def test_outcome_pass_with_bidirectional_stable_tasks(self) -> None:
        """Scenario: one stable text-only task, one stable structure-only task,
        5 fully stable tasks (text+struct both succeed), <=1 discordant cell."""
        # 6 tasks, 2 replicas each = 12 panels = 24 attempts
        # Task 0: stable text-only (text both success, struct both fail)
        # Task 1: stable structure-only (struct both success, text both fail)
        # Tasks 2-5: both succeed on both replicas (stable)
        text_successes = {
            0: {0: True, 1: True},
            1: {0: False, 1: False},
            2: {0: True, 1: True},
            3: {0: True, 1: True},
            4: {0: True, 1: True},
            5: {0: True, 1: True},
        }
        struct_successes = {
            0: {0: False, 1: False},
            1: {0: True, 1: True},
            2: {0: True, 1: True},
            3: {0: True, 1: True},
            4: {0: True, 1: False},  # one discordant cell on task 4
            5: {0: True, 1: True},
        }

        records = self._make_outcome_records(
            text_successes=text_successes,
            struct_successes=struct_successes,
        )
        report = evaluate_observation_canary_gate(
            self.manifest, self.registry, self.policy_split, records,
            preflight=self.preflight,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["decision"], "screen_pass")
        self.assertEqual(report["stage"], "outcome_screen")

        stability = report["stability"]
        self.assertLessEqual(stability["discordant_cell_count"], 1)
        self.assertGreaterEqual(stability["stable_text_only_count"], 1)
        self.assertGreaterEqual(stability["stable_structure_only_count"], 1)

    def test_outcome_futility_no_reverse_niche(self) -> None:
        """No stable structure-only task → futility_no_go."""
        # All tasks: text succeeds, structure succeeds. No reverse niche.
        records = self._make_outcome_records()
        # All successes by default
        report = evaluate_observation_canary_gate(
            self.manifest, self.registry, self.policy_split, records,
            preflight=self.preflight,
        )
        # Should be futility because no stable structure-only task
        self.assertFalse(report["passed"])
        self.assertEqual(report["decision"], "futility_no_go")
        self.assertEqual(report["stability"]["stable_structure_only_count"], 0)

    def test_outcome_futility_excess_discordance(self) -> None:
        """Too many discordant cells → futility_no_go."""
        # Make 3 tasks discordant (text and struct disagree on replicas).
        text_successes = {
            i: {0: True, 1: False} for i in range(3)
        }
        text_successes.update({
            i: {0: True, 1: True} for i in range(3, 6)
        })
        struct_successes = {
            i: {0: False, 1: True} for i in range(3)
        }
        struct_successes.update({
            i: {0: True, 1: True} for i in range(3, 6)
        })

        records = self._make_outcome_records(
            text_successes=text_successes,
            struct_successes=struct_successes,
        )
        report = evaluate_observation_canary_gate(
            self.manifest, self.registry, self.policy_split, records,
            preflight=self.preflight,
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["decision"], "futility_no_go")
        self.assertGreater(report["stability"]["discordant_cell_count"], 1)

    def test_aggregate_imbalance_not_a_gate(self) -> None:
        """Large arm success imbalance should not cause failure alone."""
        # Text succeeds most of the time, structure rarely.
        # Tasks 0-4: stable text-only (text both success, struct both fail)
        text_successes: dict[int, dict[int, bool]] = {}
        struct_successes: dict[int, dict[int, bool]] = {}
        for i in range(5):
            text_successes[i] = {0: True, 1: True}
            struct_successes[i] = {0: False, 1: False}
        # Task 5: stable structure-only (text both fail, struct both success)
        text_successes[5] = {0: False, 1: False}
        struct_successes[5] = {0: True, 1: True}

        records = self._make_outcome_records(
            text_successes=text_successes,
            struct_successes=struct_successes,
        )
        report = evaluate_observation_canary_gate(
            self.manifest, self.registry, self.policy_split, records,
            preflight=self.preflight,
        )
        # The gate should pass because stability is met (5 stable text-only,
        # 1 stable structure-only, 0 discordant) despite huge arm imbalance.
        self.assertTrue(report["passed"])
        self.assertEqual(report["decision"], "screen_pass")
        self.assertEqual(report["stability"]["discordant_cell_count"], 0)
        self.assertEqual(report["stability"]["stable_text_only_count"], 5)  # tasks 0-4
        self.assertEqual(report["stability"]["stable_structure_only_count"], 1)  # task 5

        # Verify descriptive outcomes show the imbalance
        desc = report["descriptive_outcomes"]
        self.assertEqual(desc["text_first_successes"], 10)
        self.assertEqual(desc["text_first_attempts"], 12)
        self.assertEqual(desc["structure_first_successes"], 2)
        self.assertEqual(desc["structure_first_attempts"], 12)
        self.assertIn("descriptive", desc["note"].lower())

    def test_wrong_outcome_stage_shape_rejected(self) -> None:
        """outcome_screen with wrong number of tasks should add reason."""
        manifest = _make_canary_manifest(
            self.registry, self.policy_split, stage="outcome_screen",
            num_tasks=3, rollout_replicas=2,  # should be 6
        )
        manifest = dict(manifest)
        manifest.pop("manifest_hash", None)
        manifest["registry_hash"] = self.registry.registry_hash
        manifest["manifest_hash"] = _canonical_hash(manifest)
        preflight = _make_preflight(manifest)

        records: list[dict[str, Any]] = []
        for panel in manifest["panels"]:
            record = _make_panel_result(
                manifest, panel, self.text_treatment, self.structure_treatment,
            )
            record["result"]["treatment_registry_hash"] = self.registry.registry_hash
            records.append(record)

        report = evaluate_observation_canary_gate(
            manifest, self.registry, self.policy_split, records,
            preflight=preflight,
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["decision"], "invalid")

    def test_report_has_all_required_sections(self) -> None:
        """Gate report must have all required top-level keys."""
        records = self._make_outcome_records()
        report = evaluate_observation_canary_gate(
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

    def test_later_cross_modal_observations_do_not_invalidate(self) -> None:
        """Later cross-modal observations should not break receipt validity."""
        records: list[dict[str, Any]] = []
        for panel in self.manifest["panels"]:
            record = _make_panel_result(
                self.manifest, panel, self.text_treatment, self.structure_treatment,
                attempt_mods={
                    self.text_treatment.bundle_id: {"with_later_observations": True},
                    self.structure_treatment.bundle_id: {"with_later_observations": True},
                },
            )
            record["result"]["treatment_registry_hash"] = self.registry.registry_hash
            records.append(record)

        report = evaluate_observation_canary_gate(
            self.manifest, self.registry, self.policy_split, records,
            preflight=self.preflight,
        )
        # Check there are no receipt-related errors
        receipt_errors = [r for r in report["reasons"] if "receipt" in r.lower()]
        self.assertEqual(receipt_errors, [])

    def test_protocol_validation_rejects_extra_treatments(self) -> None:
        """_validate_canary_protocol should reject manifests with != 2 bundle IDs."""
        from pyreplab_harness.m3_observation_canary_gate import _validate_canary_protocol
        with self.assertRaisesRegex(ValueError, "exactly 2"):
            _validate_canary_protocol(
                {
                    "task_role": "T_canary",
                    "protocol": {
                        "schema_version": "m3-observation-canary-protocol-v1",
                        "stage": "mechanics_dry_run",
                    },
                    "policy_bundle_ids": [
                        self.text_treatment.bundle_id,
                        self.structure_treatment.bundle_id,
                        "extra-bundle-id@2-00000000",
                    ],
                },
                self.registry,
            )


class ObservationCanaryGateCLITest(unittest.TestCase):
    """Tests for CLI exit codes."""

    def setUp(self) -> None:
        self.text_treatment = _make_text_treatment()
        self.structure_treatment = _make_structure_treatment()
        self.registry = _make_canary_registry(self.text_treatment, self.structure_treatment)
        self.policy_split = _make_canary_policy_split(self.registry)
        self.manifest = _make_canary_manifest(
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
                self.manifest, panel, self.text_treatment, self.structure_treatment,
            )
            record["result"]["treatment_registry_hash"] = self.registry.registry_hash
            records.append(record)
        return records

    def test_exit_0_on_pass(self) -> None:
        from pyreplab_harness.m3_observation_canary_gate import main

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
            registry_path.write_text(json.dumps(self.registry.to_dict()), encoding="utf-8")
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
        from pyreplab_harness.m3_observation_canary_gate import main

        manifest = _make_canary_manifest(
            self.registry, self.policy_split, stage="outcome_screen",
            num_tasks=6, rollout_replicas=2,
        )
        manifest = dict(manifest)
        manifest.pop("manifest_hash", None)
        manifest["registry_hash"] = self.registry.registry_hash
        manifest["manifest_hash"] = _canonical_hash(manifest)
        preflight = _make_preflight(manifest)

        # All successes → no stable structure-only or text-only → futility_no_go
        records: list[dict[str, Any]] = []
        for panel in manifest["panels"]:
            record = _make_panel_result(
                manifest, panel, self.text_treatment, self.structure_treatment,
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
            registry_path.write_text(json.dumps(self.registry.to_dict()), encoding="utf-8")
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
        from pyreplab_harness.m3_observation_canary_gate import main

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
        from pyreplab_harness.m3_observation_canary_gate import main

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


if __name__ == "__main__":
    unittest.main()
