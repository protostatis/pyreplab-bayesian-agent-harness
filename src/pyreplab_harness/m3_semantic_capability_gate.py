"""Executable screening/futility gate for the semantic-capability canary.

This gate validates that the specialist receipt mechanism is mechanically
functional across two treatments whose only factor-level difference is
capability specialist (table_specialist vs form_specialist).  It is
exclusively a screening and futility detector; it is not causal or
allocator-effectiveness evidence.

CLI::

    pyreplab-m3-semantic-capability-gate RESULTS \\
        --manifest MANIFEST --registry REGISTRY --policy-split SPLIT \\
        [--preflight PREFLIGHT] [--output OUTPUT]

Exit codes: 0 = pass, 2 = valid futility no-go, 1 = invalid / exception.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import string
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .io_utils import write_json
from .m3_exploratory_screen import (
    PANEL_RESULT_SCHEMA,
    _validate_panel_result_strict,
    validate_screen_manifest,
)
from .m3_pilot import _load_json
from .orchestrator import policy_spec_from_treatment
from .treatments import TreatmentRegistry

GATE_SCHEMA = "m3-semantic-capability-gate-v1"
GATE_SCHEMA_V2 = "m3-semantic-capability-gate-v2"

PROTOCOL_SCHEMA_V1 = "m3-semantic-capability-protocol-v1"
PROTOCOL_SCHEMA_V2 = "m3-semantic-capability-protocol-v2"
DATASET_CONTRACT_SCHEMA = "m3-semantic-dataset-contract-v1"

# Frozen v2 replication_screen decision rule.  These thresholds are the only
# valid values for the protocol-v2 replication screen; any drift is invalid.
_V2_REPLICATION_DECISION_RULE: dict[str, int] = {
    "maximum_discordant_cells": 4,
    "minimum_favorable_table_tasks": 7,
    "maximum_adverse_table_tasks": 1,
    "minimum_favorable_form_tasks": 7,
    "maximum_adverse_form_tasks": 1,
    "minimum_stable_table_only_tasks": 2,
    "minimum_stable_form_only_tasks": 2,
}

# ---------------------------------------------------------------------------
# infrastructure-error detection markers
# ---------------------------------------------------------------------------

_INFRA_ERROR_MARKERS = (
    "BrokenPipeError",
    "ConnectionResetError",
    "browser process exited",
    "process connection broken",
    "response timed out",
    "result exceeds",
)


def _is_infrastructure_error(entry: Mapping[str, Any]) -> bool:
    """Return True if the tool-trace entry signals an infrastructure error."""
    details = entry.get("details")
    if not isinstance(details, Mapping):
        return False
    if details.get("infrastructure_error") is True:
        return True
    if details.get("infrastructure_error") is False:
        return False
    error_text = str(details.get("error", "")).casefold()
    for marker in _INFRA_ERROR_MARKERS:
        if marker.casefold() in error_text:
            return True
    return False


# ---------------------------------------------------------------------------
# specialist receipt helpers
# ---------------------------------------------------------------------------


def _is_hex_digest(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in string.hexdigits for character in value)
    )


def _canonical_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _specialist_from_treatment(treatment: Any) -> str:
    """Extract the specialist capability string from a treatment's metadata."""
    meta = treatment.generator_metadata
    cap = str(meta.get("capability", ""))
    return cap


# ---------------------------------------------------------------------------
# specialist adherence assessment
# ---------------------------------------------------------------------------


def _assess_semantic_specialist_adherence(
    treatment: Any,
    trajectory: Mapping[str, Any] | None,
    *,
    task_group: str,
) -> dict[str, Any]:
    """Assess one trajectory against its assigned specialist capability.

    Returns a dict with keys:
    * ``specialist`` — the capability string from treatment metadata
    * ``specialist_receipt_valid`` — bool or None
    * ``specialist_action_match`` — bool
    * ``unavailable_specialist_found`` — bool (wrong specialist in trace)
    * ``infrastructure_errors`` — count of infra-error entries
    * ``tool_cap_compliant`` — whether tool calls within limit
    * ``admitted_tool_call_count`` — number of non-rejected tool calls
    """
    meta = treatment.generator_metadata
    specialist = str(meta.get("capability", ""))
    cap = int(treatment.tool_call_limit)

    trace_value = (trajectory or {}).get("tool_trace", [])
    trace = [entry for entry in trace_value if isinstance(entry, Mapping)]

    admitted: list[Mapping[str, Any]] = []
    rejected_count = 0
    infra_errors = 0
    unavailable_specialist_found = False
    specialist_receipt_valid: bool | None = None
    specialist_action_match = False
    budgeted_tools = frozenset(
        {"bash", "unbrowser", "semantic_table", "semantic_form"}
    )
    legacy_pre_execution_candidates = {
        index
        for index, entry in enumerate(trace)
        if "pre_execution_rejected" not in entry
        and entry.get("tool_name") in budgeted_tools
        and bool(entry.get("is_error"))
        and not entry.get("details")
        and not entry.get("operation_aborted")
        and any(
            bool(later.get("operation_aborted"))
            for later in trace[index + 1 :]
        )
    }
    otherwise_admitted_count = sum(
        1
        for index, entry in enumerate(trace)
        if entry.get("tool_name") in budgeted_tools
        and index not in legacy_pre_execution_candidates
        and not bool(entry.get("budget_rejected"))
        and entry.get("pre_execution_rejected") is not True
        and not bool(entry.get("operation_aborted"))
    )
    if otherwise_admitted_count != cap:
        legacy_pre_execution_candidates.clear()

    # Expected action prefix for this specialist.
    if specialist == "table_specialist":
        expected_specialist_action = "semantic_table"
        opposite_actions = frozenset({"semantic_form"})
    elif specialist == "form_specialist":
        expected_specialist_action = "semantic_form"
        opposite_actions = frozenset({"semantic_table"})
    else:
        expected_specialist_action = None
        opposite_actions = frozenset()

    for index, entry in enumerate(trace):
        # Infrastructure error detection.
        if _is_infrastructure_error(entry):
            infra_errors += 1
            admitted.append(entry)
            continue

        # Budget check.
        rejected = bool(entry.get("budget_rejected"))
        if entry.get("pre_execution_rejected") is True:
            rejected = True
        elif not rejected and index in legacy_pre_execution_candidates:
            # Legacy traces lack pre_execution_rejected. Exactly ``cap`` other
            # admitted calls followed by an abort prove this empty error never
            # reached the budget hook.
            rejected = True
        if (
            not rejected
            and len(admitted) >= cap
            and entry.get("tool_name") in budgeted_tools
            and bool(entry.get("is_error"))
            and (
                bool(entry.get("operation_aborted"))
                or not entry.get("details")
            )
        ):
            rejected = True
        if rejected:
            rejected_count += 1
            continue

        admitted.append(entry)

        # Check for unavailable specialist usage.
        tool_name = entry.get("tool_name")
        if isinstance(tool_name, str) and tool_name in opposite_actions:
            unavailable_specialist_found = True

        # Check for specialist receipt.
        if (
            tool_name == expected_specialist_action
            and not entry.get("is_error")
            and not entry.get("budget_rejected")
            and not entry.get("operation_aborted")
            and specialist_receipt_valid is None
        ):
            details = entry.get("details")
            if not isinstance(details, Mapping):
                continue
            receipt = details.get("semantic_specialist_receipt")
            if not isinstance(receipt, Mapping):
                continue
            # Validate receipt schema and integrity.
            if receipt.get("schema_version") != "pyreplab-semantic-specialist-receipt-v1":
                continue
            if receipt.get("delivered") is not True:
                continue
            if receipt.get("specialist") != specialist:
                continue
            action = receipt.get("action")
            if not isinstance(action, str) or not action.startswith("semantic_"):
                continue
            # Verify payload integrity.
            payload = details.get("semantic_payload")
            if payload is None:
                continue
            try:
                encoded = json.dumps(
                    payload,
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            except (TypeError, ValueError):
                continue
            payload_bytes = receipt.get("payload_bytes")
            if (
                isinstance(payload_bytes, bool)
                or not isinstance(payload_bytes, int)
                or payload_bytes != len(encoded)
            ):
                continue
            if receipt.get("payload_sha256") != hashlib.sha256(encoded).hexdigest():
                continue
            specialist_receipt_valid = True
            specialist_action_match = action == expected_specialist_action

    tool_cap_compliant = len(admitted) <= cap

    return {
        "specialist": specialist,
        "specialist_receipt_valid": specialist_receipt_valid,
        "specialist_action_match": specialist_action_match,
        "expected_specialist_action": expected_specialist_action,
        "unavailable_specialist_found": unavailable_specialist_found,
        "infrastructure_errors": infra_errors,
        "tool_cap": cap,
        "admitted_tool_call_count": len(admitted),
        "rejected_tool_call_count": rejected_count,
        "tool_cap_compliant": tool_cap_compliant,
    }


# ---------------------------------------------------------------------------
# protocol validation
# ---------------------------------------------------------------------------


def _validate_semantic_capability_protocol(
    manifest: Mapping[str, Any],
    registry: TreatmentRegistry,
) -> tuple[list[Any], list[Any], dict[str, list[str]]]:
    """Validate the semantic-capability protocol contract.

    Returns ``(table_treatments, form_treatments, task_groups)``.
    """
    errors: list[str] = []

    # ---- task_role must be T_canary -----------------------------------------
    task_role = manifest.get("task_role")
    if task_role != "T_canary":
        errors.append(f"manifest task_role must be T_canary, got {task_role!r}")

    # ---- protocol must be present and have the right schema_version ----------
    protocol = manifest.get("protocol")
    if not isinstance(protocol, Mapping):
        errors.append("manifest.protocol must be an object")
        if errors:
            raise ValueError("; ".join(errors))
        raise ValueError("; ".join(errors))

    protocol_schema = protocol.get("schema_version")
    if protocol_schema == PROTOCOL_SCHEMA_V2:
        protocol_version = "v2"
    elif protocol_schema == PROTOCOL_SCHEMA_V1:
        protocol_version = "v1"
    else:
        protocol_version = "unknown"
        errors.append(
            "protocol.schema_version must be "
            f"{PROTOCOL_SCHEMA_V1} or {PROTOCOL_SCHEMA_V2}, "
            f"got {protocol_schema!r}"
        )

    # ---- stage --------------------------------------------------------------
    stage = protocol.get("stage")
    if protocol_version == "v2":
        valid_stages = ("mechanics_dry_run", "replication_screen")
        stage_desc = "mechanics_dry_run or replication_screen"
    else:
        valid_stages = ("mechanics_dry_run", "outcome_screen")
        stage_desc = "mechanics_dry_run or outcome_screen"
    if stage not in valid_stages:
        errors.append(
            f"protocol.stage must be {stage_desc}, got {stage!r}"
        )

    # ---- mechanism ----------------------------------------------------------
    mechanism = protocol.get("mechanism")
    if not isinstance(mechanism, Mapping):
        errors.append("protocol.mechanism must be an object")
    else:
        if mechanism.get("name") != "controller_owned_public_html_semantic_operation":
            errors.append(
                "protocol.mechanism.name must be "
                "'controller_owned_public_html_semantic_operation', "
                f"got {mechanism.get('name')!r}"
            )

    if protocol.get("claim_boundary") != "screening_futility_only":
        errors.append(
            "protocol.claim_boundary must be 'screening_futility_only'"
        )

    # ---- decision_rule ------------------------------------------------------
    decision_rule = protocol.get("decision_rule")
    if not isinstance(decision_rule, Mapping):
        errors.append("protocol.decision_rule must be an object")
    elif stage == "mechanics_dry_run":
        if decision_rule.get("all_attempts_mechanically_valid") is not True:
            errors.append(
                "mechanics decision rule must require all attempts mechanically valid"
            )
    elif stage == "outcome_screen":
        expected_rules = {
            "maximum_discordant_cells": 2,
            "minimum_stable_table_only_tasks": 1,
            "minimum_stable_form_only_tasks": 1,
            "minimum_successes_per_arm": 2,
            "maximum_successes_per_arm": 14,
            "maximum_absolute_success_difference": 4,
        }
        for key, expected in expected_rules.items():
            value = decision_rule.get(key)
            if isinstance(value, bool) or value != expected:
                errors.append(
                    f"protocol.decision_rule.{key} must equal {expected}, got {value!r}"
                )
    elif stage == "replication_screen":
        for key, expected in _V2_REPLICATION_DECISION_RULE.items():
            value = decision_rule.get(key)
            if isinstance(value, bool) or value != expected:
                errors.append(
                    f"protocol.decision_rule.{key} must equal {expected}, got {value!r}"
                )

    # ---- v2 replication_screen contract fields -------------------------------
    if protocol_version == "v2" and stage == "replication_screen":
        # run_policy must pin early-out / replacement behavior.
        run_policy = protocol.get("run_policy")
        if not isinstance(run_policy, Mapping):
            errors.append("protocol.run_policy must be an object")
        else:
            if run_policy.get("early_outcome_stop") is not False:
                errors.append(
                    "protocol.run_policy.early_outcome_stop must be false, "
                    f"got {run_policy.get('early_outcome_stop')!r}"
                )
            if run_policy.get("outcome_driven_replacement") is not False:
                errors.append(
                    "protocol.run_policy.outcome_driven_replacement must be false, "
                    f"got {run_policy.get('outcome_driven_replacement')!r}"
                )

        # dataset_contract pins the canary-excluded dataset provenance.
        dataset_contract = protocol.get("dataset_contract")
        if not isinstance(dataset_contract, Mapping):
            errors.append("protocol.dataset_contract must be an object")
        else:
            if dataset_contract.get("schema_version") != DATASET_CONTRACT_SCHEMA:
                errors.append(
                    "protocol.dataset_contract.schema_version must be "
                    f"{DATASET_CONTRACT_SCHEMA!r}, "
                    f"got {dataset_contract.get('schema_version')!r}"
                )
            if not _is_hex_digest(dataset_contract.get("contract_hash"), 64):
                errors.append(
                    "protocol.dataset_contract.contract_hash must be a "
                    "64-char hex digest"
                )
            if dataset_contract.get("governance_role") != "canary_excluded":
                errors.append(
                    "protocol.dataset_contract.governance_role must be "
                    f"'canary_excluded', "
                    f"got {dataset_contract.get('governance_role')!r}"
                )

        # mechanics_qualification must confirm the prior mechanics pass.
        mechanics_qualification = protocol.get("mechanics_qualification")
        if not isinstance(mechanics_qualification, Mapping):
            errors.append("protocol.mechanics_qualification must be an object")
        else:
            if not _is_hex_digest(
                mechanics_qualification.get("mechanics_manifest_hash"), 64
            ):
                errors.append(
                    "protocol.mechanics_qualification.mechanics_manifest_hash "
                    "must be a 64-char hex digest"
                )
            if not _is_hex_digest(
                mechanics_qualification.get("mechanics_gate_sha256"), 64
            ):
                errors.append(
                    "protocol.mechanics_qualification.mechanics_gate_sha256 "
                    "must be a 64-char hex digest"
                )
            if mechanics_qualification.get("decision") != "mechanics_pass":
                errors.append(
                    "protocol.mechanics_qualification.decision must be "
                    f"'mechanics_pass', "
                    f"got {mechanics_qualification.get('decision')!r}"
                )

    # ---- task_groups (required for outcome_screen / replication_screen) -----
    task_groups: dict[str, list[str]] = {"table": [], "form": []}
    tg = protocol.get("task_groups")
    grouped_stage = stage in ("outcome_screen", "replication_screen")
    expected_group_size = 8 if stage == "replication_screen" else 4
    if grouped_stage:
        if not isinstance(tg, Mapping):
            errors.append(f"protocol.task_groups must be an object for {stage}")
        else:
            table_ids = tg.get("table")
            form_ids = tg.get("form")
            if not isinstance(table_ids, list) or len(table_ids) != expected_group_size:
                errors.append(
                    f"protocol.task_groups.table must be a list of "
                    f"{expected_group_size} task IDs"
                )
            else:
                task_groups["table"] = [str(tid) for tid in table_ids]
            if not isinstance(form_ids, list) or len(form_ids) != expected_group_size:
                errors.append(
                    f"protocol.task_groups.form must be a list of "
                    f"{expected_group_size} task IDs"
                )
            else:
                task_groups["form"] = [str(tid) for tid in form_ids]
            if set(task_groups["table"]) & set(task_groups["form"]):
                errors.append("protocol.task_groups.table and form must be disjoint")
            manifest_tasks = manifest.get("tasks")
            if isinstance(manifest_tasks, list):
                manifest_ids = {
                    str(task.get("task_id"))
                    for task in manifest_tasks
                    if isinstance(task, Mapping)
                }
                grouped_ids = set(task_groups["table"]) | set(task_groups["form"])
                if grouped_ids != manifest_ids:
                    errors.append(
                        "protocol.task_groups must exactly partition manifest task IDs"
                    )
                task_by_id = {
                    str(task.get("task_id")): task
                    for task in manifest_tasks
                    if isinstance(task, Mapping)
                }
                for task_id in task_groups["table"]:
                    task = task_by_id.get(task_id)
                    if task is not None and task.get("template") != "table_filter_sort":
                        errors.append(
                            f"table task group contains non-table template: {task_id!r}"
                        )
                for task_id in task_groups["form"]:
                    task = task_by_id.get(task_id)
                    if task is not None and task.get("template") != "form_entry_validation":
                        errors.append(
                            f"form task group contains non-form template: {task_id!r}"
                        )

    # ---- exactly two policy bundle IDs --------------------------------------
    bundle_ids = manifest.get("policy_bundle_ids", [])
    if not isinstance(bundle_ids, list) or len(bundle_ids) != 2:
        errors.append("manifest must have exactly 2 policy_bundle_ids")

    if errors:
        raise ValueError("; ".join(errors))

    # ---- fetch treatments ---------------------------------------------------
    treatments = list(registry.by_bundle_id(str(bid)) for bid in bundle_ids)

    # ---- capability levels must be table_specialist and form_specialist -----
    cap_levels = set()
    for t in treatments:
        cap = str(t.generator_metadata.get("capability", ""))
        cap_levels.add(cap)
    if cap_levels != {"table_specialist", "form_specialist"}:
        errors.append(
            "capability levels must be exactly "
            "{table_specialist, form_specialist}, "
            f"got {sorted(cap_levels)}"
        )

    # ---- partition treatments by specialist ---------------------------------
    table_t = [
        t for t in treatments
        if t.generator_metadata.get("capability") == "table_specialist"
    ]
    form_t = [
        t for t in treatments
        if t.generator_metadata.get("capability") == "form_specialist"
    ]
    if len(table_t) != 1 or len(form_t) != 1:
        errors.append(
            "must have exactly one table_specialist and one form_specialist treatment"
        )

    # ---- tool interfaces must be exact --------------------------------------
    if table_t:
        ti_table = table_t[0].tool_interface
        if ti_table != "native_bash_unbrowser_semantic_table_v1":
            errors.append(
                "table_specialist treatment must use "
                "native_bash_unbrowser_semantic_table_v1, "
                f"got {ti_table!r}"
            )
    if form_t:
        ti_form = form_t[0].tool_interface
        if ti_form != "native_bash_unbrowser_semantic_form_v1":
            errors.append(
                "form_specialist treatment must use "
                "native_bash_unbrowser_semantic_form_v1, "
                f"got {ti_form!r}"
            )

    # ---- system_prompt must be identical ------------------------------------
    if table_t and form_t:
        if table_t[0].system_prompt != form_t[0].system_prompt:
            errors.append("system_prompt must be identical between treatments")

    # ---- allowed_tools must differ only by specialist assignment ------------
    if table_t and form_t:
        table_tools = set(table_t[0].allowed_tools)
        form_tools = set(form_t[0].allowed_tools)
        common = table_tools & form_tools
        table_only = table_tools - form_tools
        form_only = form_tools - table_tools
        if not common:
            errors.append("treatments must share a common allowed_tools subset")
        if len(table_only) > 1 or len(form_only) > 1:
            errors.append(
                "allowed_tools may differ by at most one specialist tool per arm"
            )

    # ---- resource fields must be identical ----------------------------------
    if table_t and form_t:
        comparable_fields = (
            "max_output_tokens",
            "tool_call_limit",
            "command_timeout_seconds",
            "wall_time_limit_seconds",
        )
        for field in comparable_fields:
            if getattr(table_t[0], field) != getattr(form_t[0], field):
                errors.append(f"treatment resource field {field!r} differs")

    # ---- parent/common substrate metadata -----------------------------------
    for t in treatments:
        meta = t.generator_metadata
        if not isinstance(meta.get("parent_bundle_id"), str) or not meta["parent_bundle_id"].strip():
            errors.append(f"treatment {t.bundle_id} missing parent_bundle_id in metadata")
        if not isinstance(meta.get("substrate"), str) or not meta["substrate"].strip():
            errors.append(
                f"treatment {t.bundle_id} missing substrate in metadata"
            )

    # ---- verify parent_bundle_ids in protocol match treatments ---------------
    parent_bundle_ids = protocol.get("parent_bundle_ids")
    if not isinstance(parent_bundle_ids, Mapping):
        errors.append("protocol.parent_bundle_ids must be an object")
    else:
        for level, treatment_list in (
            ("table_specialist", table_t),
            ("form_specialist", form_t),
        ):
            expected_parent = parent_bundle_ids.get(level)
            actual_parent = (
                treatment_list[0].generator_metadata.get("parent_bundle_id")
                if treatment_list
                else None
            )
            if not isinstance(expected_parent, str) or not expected_parent:
                errors.append(f"protocol.parent_bundle_ids.{level} must be nonempty")
            elif actual_parent != expected_parent:
                errors.append(
                    f"{level} parent bundle mismatch: {actual_parent!r} != "
                    f"{expected_parent!r}"
                )

    if errors:
        raise ValueError("; ".join(errors))

    return table_t, form_t, task_groups


# ---------------------------------------------------------------------------
# record loading
# ---------------------------------------------------------------------------


def _load_semantic_capability_records(
    path: Path,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Load strictly-validated panel results for the semantic capability gate."""
    manifest_hash = manifest["manifest_hash"]
    panel_by_id = {str(p["panel_id"]): p for p in manifest["panels"]}
    task_by_id = {str(t["task_id"]): t for t in manifest["tasks"]}
    records: list[dict[str, Any]] = []
    seen_panels: set[str] = set()

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid JSONL line {line_number}: {error}"
            ) from error
        if not isinstance(record, Mapping):
            raise ValueError(f"line {line_number}: record must be an object")

        sch = record.get("schema_version")
        if sch is None:
            raise ValueError(f"line {line_number}: record missing schema_version")
        if sch != PANEL_RESULT_SCHEMA:
            raise ValueError(f"line {line_number}: unknown schema {sch!r}")

        pid = str(record.get("panel_id", ""))
        if pid in seen_panels:
            raise ValueError(f"line {line_number}: duplicate panel {pid}")
        seen_panels.add(pid)

        if record.get("manifest_hash") != manifest_hash:
            raise ValueError(f"line {line_number}: manifest hash mismatch")

        mpanel = panel_by_id.get(pid)
        mpanel_task = task_by_id.get(mpanel["task_id"]) if mpanel else None

        errors = _validate_panel_result_strict(
            record, manifest, mpanel, mpanel_task
        )
        if errors:
            raise ValueError(
                f"line {line_number}: invalid panel result: {'; '.join(errors)}"
            )

        records.append(record)

    # Check completeness.
    expected = {str(p["panel_id"]) for p in manifest["panels"]}
    found = {str(r["panel_id"]) for r in records}
    missing = expected - found
    extra = found - expected
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing panels: {sorted(missing)}")
        if extra:
            parts.append(f"extra panels: {sorted(extra)}")
        raise ValueError("; ".join(parts))

    return records


# ---------------------------------------------------------------------------
# gate evaluation
# ---------------------------------------------------------------------------


def evaluate_semantic_capability_gate(
    manifest: Mapping[str, Any],
    registry: TreatmentRegistry,
    policy_split: Mapping[str, Any],
    records: list[dict[str, Any]],
    *,
    preflight: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Evaluate the semantic-capability canary gate.

    Returns a deterministic machine-readable report.
    """
    # ---- 1. validate manifest / registry / split ----------------------------
    validate_screen_manifest(manifest, registry, policy_split)
    manifest_hash = str(manifest["manifest_hash"])

    # ---- 2. validate semantic capability protocol ---------------------------
    table_treatments, form_treatments, task_groups = (
        _validate_semantic_capability_protocol(manifest, registry)
    )

    table_treatment = table_treatments[0]
    form_treatment = form_treatments[0]
    table_bid = table_treatment.bundle_id
    form_bid = form_treatment.bundle_id
    bundle_ids = [table_bid, form_bid]
    bundle_by_id: dict[str, Any] = {
        table_bid: table_treatment,
        form_bid: form_treatment,
    }

    protocol = manifest["protocol"]
    stage = protocol["stage"]
    protocol_version = (
        "v2" if protocol.get("schema_version") == PROTOCOL_SCHEMA_V2 else "v1"
    )
    report_schema = GATE_SCHEMA_V2 if protocol_version == "v2" else GATE_SCHEMA

    # ---- 3. validate preflight identity -------------------------------------
    runtime_preflight_ok = bool(
        preflight
        and preflight.get("screen_preflight") is True
        and preflight.get("manifest_hash") == manifest_hash
        and preflight.get("runtime_pins") == manifest.get("runtime_pins")
        and preflight.get("remote_identity") == manifest.get("remote_identity")
        and _is_hex_digest(preflight.get("code_revision"), 40)
        and _is_hex_digest(preflight.get("source_tree_hash"), 64)
        and isinstance(preflight.get("worktree_clean"), bool)
        and _is_hex_digest(preflight.get("worktree_status_hash"), 64)
    )

    # ---- 4. extract and validate all attempts -------------------------------
    task_by_id: dict[str, dict[str, Any]] = {
        str(t["task_id"]): t for t in manifest["tasks"]
    }
    panel_by_id: dict[str, dict[str, Any]] = {
        str(p["panel_id"]): p for p in manifest["panels"]
    }
    expected_bundle_set = {table_bid, form_bid}

    reasons: list[str] = []
    mechanisms_errors: list[str] = []

    attempt_ids: set[str] = set()
    task_outcomes: dict[str, dict[str, dict[int, bool]]] = {}
    output_costs: dict[str, list[float]] = {
        table_bid: [],
        form_bid: [],
    }
    # Per-arm success counts.
    table_successes = 0
    form_successes = 0
    table_total = 0
    form_total = 0
    # Paired panel outcomes (panel_id -> {bid: success_bool}).
    paired_outcomes: dict[str, dict[str, bool]] = {}

    for task in manifest["tasks"]:
        tid = str(task["task_id"])
        task_outcomes[tid] = {bid: {} for bid in bundle_ids}

    infrastructure_errors = 0
    structural_errors: list[str] = []

    # Build task-group lookup.
    grouped_stage = stage in ("outcome_screen", "replication_screen")
    table_task_ids = set(task_groups.get("table", [])) if grouped_stage else set()
    form_task_ids = set(task_groups.get("form", [])) if grouped_stage else set()

    for record in records:
        pid = str(record.get("panel_id", ""))
        if record.get("status") == "error":
            infrastructure_errors += 1
            structural_errors.append(f"{pid}: infrastructure error")
            continue
        if record.get("status") != "completed":
            structural_errors.append(f"{pid}: unknown status")
            continue

        result = record.get("result")
        if not isinstance(result, Mapping):
            structural_errors.append(f"{pid}: result missing")
            continue

        task_id = str(result.get("task_id", ""))
        task = task_by_id.get(task_id)
        if task is None:
            structural_errors.append(f"{pid}: unknown task_id {task_id!r}")
            continue
        replica = int(result.get("rollout_replica", -1))

        # Determine task group for this panel.
        if grouped_stage:
            if task_id in table_task_ids:
                panel_task_group = "table"
            elif task_id in form_task_ids:
                panel_task_group = "form"
            else:
                structural_errors.append(f"{pid}: task_id not in task_groups")
                panel_task_group = ""
        else:
            panel_task_group = ""

        # ---- registry hash --------------------------------------------------
        if result.get("treatment_registry_hash") != registry.registry_hash:
            structural_errors.append(f"{pid}: registry hash mismatch")

        # ---- execution order ------------------------------------------------
        expected_order = list(panel_by_id[pid]["execution_order"])
        if result.get("execution_order") != expected_order:
            structural_errors.append(f"{pid}: execution order mismatch")

        # ---- per-attempt validation -----------------------------------------
        attempts = result.get("attempts")
        if not isinstance(attempts, Mapping) or set(attempts) != expected_bundle_set:
            structural_errors.append(f"{pid}: attempts set is incomplete")
            continue

        panel_successes: dict[str, bool] = {}

        for bid in expected_order:
            item = attempts.get(bid)
            if not isinstance(item, Mapping):
                structural_errors.append(f"{pid}/{bid}: attempt malformed")
                continue

            # attempt ID unique
            aid = item.get("attempt_id")
            if not isinstance(aid, str) or not aid:
                structural_errors.append(f"{pid}/{bid}: attempt id missing")
                continue
            if aid in attempt_ids:
                structural_errors.append(f"duplicate attempt id: {aid}")
            attempt_ids.add(aid)

            treatment = bundle_by_id[bid]

            # executed policy identity
            expected_policy = policy_spec_from_treatment(treatment).to_dict()
            if item.get("policy") != expected_policy:
                structural_errors.append(f"{pid}/{bid}: executed policy mismatch")

            # verification
            verif = item.get("verification")
            if not isinstance(verif, Mapping) or not isinstance(verif.get("success"), bool):
                structural_errors.append(f"{pid}/{bid}: verification missing")

            # pi_return_code must be 0
            pi_rc = item.get("pi_return_code")
            if isinstance(pi_rc, bool) or not isinstance(pi_rc, int):
                structural_errors.append(f"{pid}/{bid}: pi_return_code not int")
            elif pi_rc != 0:
                structural_errors.append(f"{pid}/{bid}: pi_return_code = {pi_rc}")

            # usage.output finite nonnegative
            output_cost: float | None = None
            usage = item.get("usage")
            if isinstance(usage, Mapping):
                output_val = usage.get("output")
                if (
                    output_val is None
                    or isinstance(output_val, bool)
                    or not isinstance(output_val, (int, float))
                    or not math.isfinite(float(output_val))
                    or float(output_val) < 0
                ):
                    structural_errors.append(f"{pid}/{bid}: usage.output invalid")
                else:
                    output_cost = float(output_val)
                    output_costs[bid].append(output_cost)
            else:
                structural_errors.append(f"{pid}/{bid}: usage missing")

            # trajectory structural validity
            traj = item.get("trajectory")
            ptc: Any = None
            if not isinstance(traj, Mapping):
                structural_errors.append(f"{pid}/{bid}: trajectory missing")
            else:
                pp = traj.get("planning_preamble")
                tt = traj.get("tool_trace")
                ptc = traj.get("provider_turn_count")
                if not isinstance(pp, Mapping):
                    structural_errors.append(f"{pid}/{bid}: planning_preamble missing")
                if not isinstance(tt, list):
                    structural_errors.append(f"{pid}/{bid}: tool_trace missing")
                elif any(
                    not isinstance(e, Mapping)
                    or not isinstance(e.get("tool_name"), str)
                    or not isinstance(e.get("is_error"), bool)
                    or not isinstance(e.get("budget_rejected"), bool)
                    or (
                        "pre_execution_rejected" in e
                        and not isinstance(e.get("pre_execution_rejected"), bool)
                    )
                    or not isinstance(e.get("details"), Mapping)
                    for e in tt
                ):
                    structural_errors.append(f"{pid}/{bid}: tool_trace entry malformed")
                if isinstance(ptc, bool) or not isinstance(ptc, int) or ptc < 0:
                    structural_errors.append(f"{pid}/{bid}: provider_turn_count invalid")

            # sampling receipt
            expected_sampling_receipt = {
                "seed": panel_by_id[pid]["sampling_seed"],
                "parameters": manifest["runtime_pins"]["sampling"]["parameters"],
            }
            if (
                isinstance(ptc, int) and ptc > 0
                and item.get("sampling_receipt") != expected_sampling_receipt
            ):
                structural_errors.append(f"{pid}/{bid}: sampling receipt mismatch")

            # verifier identity
            runtime_pins = manifest["runtime_pins"]
            if (
                isinstance(verif, Mapping)
                and (
                    verif.get("verifier_id") != runtime_pins.get("fixture_verifier_id")
                    or verif.get("verifier_version") != runtime_pins.get("fixture_verifier_version")
                )
            ):
                structural_errors.append(f"{pid}/{bid}: verifier identity mismatch")

            # ---- semantic specialist adherence ------------------------------
            specialist = _specialist_from_treatment(treatment)
            # Determine expected task group for this bid + task combination.
            # For mechanics_dry_run: table specialist gets table task, form gets form task.
            # For outcome_screen: use task_groups from protocol.
            if stage == "mechanics_dry_run":
                # mechanics: 2 tasks — first is table-oriented, second is form-oriented.
                task_index = next(
                    (i for i, t in enumerate(manifest["tasks"]) if t["task_id"] == task_id),
                    -1,
                )
                expected_group = "table" if task_index == 0 else "form"
            else:
                expected_group = panel_task_group

            adh = _assess_semantic_specialist_adherence(
                treatment,
                traj,
                task_group=expected_group,
            )

            # Infrastructure error detection (lifecycle death markers).
            if adh["infrastructure_errors"] > 0:
                mechanisms_errors.append(
                    f"{pid}/{bid}: infrastructure error detected "
                    f"({adh['infrastructure_errors']} entries)"
                )

            # Specialist receipt must be valid when task group matches.
            expected_specialist_for_group = (
                "table_specialist" if expected_group == "table" else "form_specialist"
            )
            if specialist == expected_specialist_for_group:
                if adh["specialist_receipt_valid"] is not True:
                    mechanisms_errors.append(
                        f"{pid}/{bid}: specialist receipt invalid or missing "
                        f"on matching {expected_group!r} task"
                    )
                elif not adh["specialist_action_match"]:
                    mechanisms_errors.append(
                        f"{pid}/{bid}: specialist action mismatch "
                        f"(expected {adh['expected_specialist_action']!r})"
                    )
            elif adh["specialist_receipt_valid"] is True:
                mechanisms_errors.append(
                    f"{pid}/{bid}: unexpected specialist receipt on "
                    f"non-matching {expected_group!r} task"
                )

            # Unavailable specialist in trace.
            if adh["unavailable_specialist_found"]:
                mechanisms_errors.append(
                    f"{pid}/{bid}: unavailable specialist found in trace"
                )

            # Tool cap compliance.
            if not adh["tool_cap_compliant"]:
                mechanisms_errors.append(
                    f"{pid}/{bid}: tool_cap not compliant "
                    f"(admitted={adh['admitted_tool_call_count']}, "
                    f"cap={adh['tool_cap']})"
                )

            # Record success for stability analysis.
            if isinstance(verif, Mapping):
                success = bool(verif.get("success", False))
                task_outcomes[task_id][bid][replica] = success
                panel_successes[bid] = success
                if bid == table_bid:
                    table_total += 1
                    if success:
                        table_successes += 1
                else:
                    form_total += 1
                    if success:
                        form_successes += 1

        paired_outcomes[pid] = panel_successes

    # ---- 5. compute mechanics -----------------------------------------------
    mechanics_valid = (
        len(structural_errors) == 0
        and len(mechanisms_errors) == 0
        and infrastructure_errors == 0
        and runtime_preflight_ok
    )
    if not mechanics_valid:
        reasons.extend(structural_errors)
        reasons.extend(mechanisms_errors)
        if infrastructure_errors:
            reasons.append(f"infrastructure_errors: {infrastructure_errors}")
        if not runtime_preflight_ok:
            reasons.append("runtime preflight identity invalid")

    # ---- 6. stage-specific analysis -----------------------------------------
    stability: dict[str, Any] = {}
    descriptive_outcomes: dict[str, Any] = {}

    if stage == "mechanics_dry_run":
        # Compute gate count assertions.
        expected_tasks = 2
        expected_panels = 2
        expected_attempts = 4
        expected_replicas = 1

        size_ok = (
            len(manifest["tasks"]) == expected_tasks
            and len(manifest["panels"]) == expected_panels
            and len(records) == expected_panels
            and len(manifest["panels"]) * len(bundle_ids) == expected_attempts
            and len(attempt_ids) == expected_attempts
            and manifest.get("rollout_replicas") == expected_replicas
            and all(
                set(replicas) == {0}
                for policies in task_outcomes.values()
                for replicas in policies.values()
            )
        )
        if not size_ok:
            reasons.append(
                f"mechanics_dry_run expects {expected_tasks} tasks, "
                f"{expected_panels} panels, {expected_attempts} attempts, "
                f"R={expected_replicas}"
            )

        passed = mechanics_valid and size_ok
        decision = "mechanics_pass" if passed else "invalid"

    elif stage == "outcome_screen":
        # Compute count assertions.
        expected_tasks = 8
        expected_replicas = 2
        expected_panels = 16
        expected_attempts = 32

        size_ok = (
            len(manifest["tasks"]) == expected_tasks
            and manifest.get("rollout_replicas") == expected_replicas
            and len(manifest["panels"]) == expected_panels
            and len(records) == expected_panels
            and len(manifest["panels"]) * len(bundle_ids) == expected_attempts
            and len(attempt_ids) == expected_attempts
            and all(
                set(replicas) == {0, 1}
                for policies in task_outcomes.values()
                for replicas in policies.values()
            )
        )
        if not size_ok:
            reasons.append(
                f"outcome_screen expects {expected_tasks} tasks, "
                f"R={expected_replicas}, {expected_panels} panels, "
                f"{expected_attempts} attempts"
            )

        decision_rule = protocol["decision_rule"]

        # ---- discordant cells (16 policy-task cells) ------------------------
        # A policy-task cell is discordant when two replicas disagree.
        discordant_cells: list[str] = []
        for tid, policies in task_outcomes.items():
            for bid in bundle_ids:
                replicas = policies[bid]
                if set(replicas.keys()) == {0, 1}:
                    if replicas[0] != replicas[1]:
                        discordant_cells.append(f"{tid}/{bid}")

        max_discordant = int(decision_rule["maximum_discordant_cells"])

        # ---- stable table-only tasks ----------------------------------------
        # Table arm 2/2 success, form arm 0/2 success.
        stable_table_only: list[str] = []
        stable_form_only: list[str] = []

        for tid, policies in task_outcomes.items():
            if set(policies[table_bid].keys()) != {0, 1} or set(policies[form_bid].keys()) != {0, 1}:
                continue
            table_both_ok = policies[table_bid][0] and policies[table_bid][1]
            form_both_ok = policies[form_bid][0] and policies[form_bid][1]
            table_both_fail = not policies[table_bid][0] and not policies[table_bid][1]
            form_both_fail = not policies[form_bid][0] and not policies[form_bid][1]

            if (
                tid in task_groups["table"]
                and table_both_ok
                and form_both_fail
            ):
                stable_table_only.append(tid)
            if (
                tid in task_groups["form"]
                and form_both_ok
                and table_both_fail
            ):
                stable_form_only.append(tid)

        min_stable_table = int(decision_rule["minimum_stable_table_only_tasks"])
        min_stable_form = int(decision_rule["minimum_stable_form_only_tasks"])

        # ---- arm totals -----------------------------------------------------
        min_successes = int(decision_rule["minimum_successes_per_arm"])
        max_successes = int(decision_rule["maximum_successes_per_arm"])
        max_abs_diff = int(decision_rule["maximum_absolute_success_difference"])

        arm_table_ok = min_successes <= table_successes <= max_successes
        arm_form_ok = min_successes <= form_successes <= max_successes
        abs_diff = abs(table_successes - form_successes)
        balance_ok = abs_diff <= max_abs_diff

        stability_ok = (
            len(discordant_cells) <= max_discordant
            and len(stable_table_only) >= min_stable_table
            and len(stable_form_only) >= min_stable_form
            and arm_table_ok
            and arm_form_ok
            and balance_ok
        )

        stability = {
            "discordant_cells": discordant_cells,
            "discordant_cell_count": len(discordant_cells),
            "maximum_discordant_cells": max_discordant,
            "stable_table_only_tasks": stable_table_only,
            "stable_table_only_count": len(stable_table_only),
            "minimum_stable_table_only_tasks": min_stable_table,
            "stable_form_only_tasks": stable_form_only,
            "stable_form_only_count": len(stable_form_only),
            "minimum_stable_form_only_tasks": min_stable_form,
            "table_arm_successes": table_successes,
            "table_arm_attempts": table_total,
            "form_arm_successes": form_successes,
            "form_arm_attempts": form_total,
            "absolute_success_difference": abs_diff,
            "maximum_absolute_success_difference": max_abs_diff,
            "minimum_successes_per_arm": min_successes,
            "maximum_successes_per_arm": max_successes,
        }

        # ---- descriptive outcomes -------------------------------------------
        descriptive_outcomes = {
            "table_arm_successes": table_successes,
            "table_arm_attempts": table_total,
            "form_arm_successes": form_successes,
            "form_arm_attempts": form_total,
            "table_arm_mean_output_tokens": (
                sum(output_costs[table_bid]) / len(output_costs[table_bid])
                if output_costs[table_bid]
                else None
            ),
            "form_arm_mean_output_tokens": (
                sum(output_costs[form_bid]) / len(output_costs[form_bid])
                if output_costs[form_bid]
                else None
            ),
            "paired_panel_count": len(paired_outcomes),
            "note": (
                "Arm totals and their absolute difference are predeclared "
                "futility gates, not treatment-effect estimates."
            ),
        }

        # ---- decision --------------------------------------------------------
        if not mechanics_valid:
            decision = "invalid"
            passed = False
        elif not size_ok:
            decision = "invalid"
            passed = False
        elif not stability_ok:
            decision = "futility_no_go"
            passed = False
            if len(discordant_cells) > max_discordant:
                reasons.append(
                    f"discordant cells {len(discordant_cells)} > "
                    f"maximum {max_discordant}"
                )
            if len(stable_table_only) < min_stable_table:
                reasons.append(
                    f"stable table-only tasks {len(stable_table_only)} < "
                    f"minimum {min_stable_table}"
                )
            if len(stable_form_only) < min_stable_form:
                reasons.append(
                    f"stable form-only tasks {len(stable_form_only)} < "
                    f"minimum {min_stable_form}"
                )
            if not arm_table_ok:
                reasons.append(
                    f"table arm successes {table_successes} outside "
                    f"[{min_successes}, {max_successes}]"
                )
            if not arm_form_ok:
                reasons.append(
                    f"form arm successes {form_successes} outside "
                    f"[{min_successes}, {max_successes}]"
                )
            if not balance_ok:
                reasons.append(
                    f"absolute success difference {abs_diff} > "
                    f"maximum {max_abs_diff}"
                )
        else:
            decision = "screen_pass"
            passed = True

    elif stage == "replication_screen":
        # ---- count assertions -----------------------------------------------
        expected_tasks = 16
        expected_replicas = 3
        expected_panels = 48
        expected_attempts = 96

        size_ok = (
            len(manifest["tasks"]) == expected_tasks
            and manifest.get("rollout_replicas") == expected_replicas
            and len(manifest["panels"]) == expected_panels
            and len(records) == expected_panels
            and len(manifest["panels"]) * len(bundle_ids) == expected_attempts
            and len(attempt_ids) == expected_attempts
            and all(
                set(replicas) == {0, 1, 2}
                for policies in task_outcomes.values()
                for replicas in policies.values()
            )
        )
        if not size_ok:
            reasons.append(
                f"replication_screen expects {expected_tasks} tasks, "
                f"R={expected_replicas}, {expected_panels} panels, "
                f"{expected_attempts} attempts"
            )

        decision_rule = protocol["decision_rule"]

        # ---- discordant cells (32 policy-task cells) ------------------------
        # A policy-task cell is discordant when the 3 replicas are not unanimous.
        discordant_cells: list[str] = []
        for tid, policies in task_outcomes.items():
            for bid in bundle_ids:
                replicas = policies[bid]
                if set(replicas.keys()) == {0, 1, 2}:
                    values = [replicas[r] for r in (0, 1, 2)]
                    if not (values[0] == values[1] == values[2]):
                        discordant_cells.append(f"{tid}/{bid}")

        max_discordant = int(decision_rule["maximum_discordant_cells"])

        # ---- favorable / adverse / tie per task -----------------------------
        # For a table task, favorable means table success count > form count,
        # adverse the reverse, tie equal.  For a form task the mirror holds.
        favorable_table_tasks: list[str] = []
        adverse_table_tasks: list[str] = []
        tie_table_tasks: list[str] = []
        favorable_form_tasks: list[str] = []
        adverse_form_tasks: list[str] = []
        tie_form_tasks: list[str] = []

        def _arm_success_counts(tid: str) -> tuple[int, int] | None:
            policies = task_outcomes.get(tid, {})
            tset = policies.get(table_bid, {})
            fset = policies.get(form_bid, {})
            if set(tset.keys()) != {0, 1, 2} or set(fset.keys()) != {0, 1, 2}:
                return None
            table_count = sum(1 for r in (0, 1, 2) if tset.get(r))
            form_count = sum(1 for r in (0, 1, 2) if fset.get(r))
            return table_count, form_count

        for tid in task_groups["table"]:
            counts = _arm_success_counts(tid)
            if counts is None:
                continue
            table_count, form_count = counts
            if table_count > form_count:
                favorable_table_tasks.append(tid)
            elif form_count > table_count:
                adverse_table_tasks.append(tid)
            else:
                tie_table_tasks.append(tid)

        for tid in task_groups["form"]:
            counts = _arm_success_counts(tid)
            if counts is None:
                continue
            table_count, form_count = counts
            if form_count > table_count:
                favorable_form_tasks.append(tid)
            elif table_count > form_count:
                adverse_form_tasks.append(tid)
            else:
                tie_form_tasks.append(tid)

        min_favorable_table = int(decision_rule["minimum_favorable_table_tasks"])
        max_adverse_table = int(decision_rule["maximum_adverse_table_tasks"])
        min_favorable_form = int(decision_rule["minimum_favorable_form_tasks"])
        max_adverse_form = int(decision_rule["maximum_adverse_form_tasks"])

        # ---- stable-only tasks ----------------------------------------------
        # Matching arm 3/3 success and opposite arm 0/3 success.
        stable_table_only: list[str] = []
        stable_form_only: list[str] = []

        for tid in task_groups["table"]:
            policies = task_outcomes.get(tid, {})
            tset = policies.get(table_bid, {})
            fset = policies.get(form_bid, {})
            if set(tset.keys()) == {0, 1, 2} and set(fset.keys()) == {0, 1, 2}:
                table_all_ok = all(tset.get(r) for r in (0, 1, 2))
                form_all_fail = not any(fset.get(r) for r in (0, 1, 2))
                if table_all_ok and form_all_fail:
                    stable_table_only.append(tid)

        for tid in task_groups["form"]:
            policies = task_outcomes.get(tid, {})
            tset = policies.get(table_bid, {})
            fset = policies.get(form_bid, {})
            if set(tset.keys()) == {0, 1, 2} and set(fset.keys()) == {0, 1, 2}:
                form_all_ok = all(fset.get(r) for r in (0, 1, 2))
                table_all_fail = not any(tset.get(r) for r in (0, 1, 2))
                if form_all_ok and table_all_fail:
                    stable_form_only.append(tid)

        min_stable_table = int(decision_rule["minimum_stable_table_only_tasks"])
        min_stable_form = int(decision_rule["minimum_stable_form_only_tasks"])

        discordance_ok = len(discordant_cells) <= max_discordant
        favorable_table_ok = len(favorable_table_tasks) >= min_favorable_table
        adverse_table_ok = len(adverse_table_tasks) <= max_adverse_table
        favorable_form_ok = len(favorable_form_tasks) >= min_favorable_form
        adverse_form_ok = len(adverse_form_tasks) <= max_adverse_form
        stable_table_ok = len(stable_table_only) >= min_stable_table
        stable_form_ok = len(stable_form_only) >= min_stable_form

        outcome_criteria_ok = (
            discordance_ok
            and favorable_table_ok
            and adverse_table_ok
            and favorable_form_ok
            and adverse_form_ok
            and stable_table_ok
            and stable_form_ok
        )

        stability = {
            "discordant_cells": discordant_cells,
            "discordant_cell_count": len(discordant_cells),
            "maximum_discordant_cells": max_discordant,
            "favorable_table_tasks": favorable_table_tasks,
            "favorable_table_count": len(favorable_table_tasks),
            "minimum_favorable_table_tasks": min_favorable_table,
            "adverse_table_tasks": adverse_table_tasks,
            "adverse_table_count": len(adverse_table_tasks),
            "maximum_adverse_table_tasks": max_adverse_table,
            "favorable_form_tasks": favorable_form_tasks,
            "favorable_form_count": len(favorable_form_tasks),
            "minimum_favorable_form_tasks": min_favorable_form,
            "adverse_form_tasks": adverse_form_tasks,
            "adverse_form_count": len(adverse_form_tasks),
            "maximum_adverse_form_tasks": max_adverse_form,
            "tie_table_tasks": tie_table_tasks,
            "tie_form_tasks": tie_form_tasks,
            "stable_table_only_tasks": stable_table_only,
            "stable_table_only_count": len(stable_table_only),
            "minimum_stable_table_only_tasks": min_stable_table,
            "stable_form_only_tasks": stable_form_only,
            "stable_form_only_count": len(stable_form_only),
            "minimum_stable_form_only_tasks": min_stable_form,
            "table_arm_successes": table_successes,
            "table_arm_attempts": table_total,
            "form_arm_successes": form_successes,
            "form_arm_attempts": form_total,
        }

        # ---- descriptive outcomes (NOT gated on global totals) --------------
        descriptive_outcomes = {
            "table_arm_successes": table_successes,
            "table_arm_attempts": table_total,
            "form_arm_successes": form_successes,
            "form_arm_attempts": form_total,
            "table_arm_mean_output_tokens": (
                sum(output_costs[table_bid]) / len(output_costs[table_bid])
                if output_costs[table_bid]
                else None
            ),
            "form_arm_mean_output_tokens": (
                sum(output_costs[form_bid]) / len(output_costs[form_bid])
                if output_costs[form_bid]
                else None
            ),
            "paired_panel_count": len(paired_outcomes),
            "note": (
                "Arm totals and token means are descriptive only; the "
                "replication screen gates on per-task favorable/adverse/"
                "stable-only and discordance criteria, not global totals."
            ),
        }

        # ---- decision --------------------------------------------------------
        if not mechanics_valid:
            decision = "invalid"
            passed = False
        elif not size_ok:
            decision = "invalid"
            passed = False
        elif not outcome_criteria_ok:
            decision = "replication_no_go"
            passed = False
            if not discordance_ok:
                reasons.append(
                    f"discordant cells {len(discordant_cells)} > "
                    f"maximum {max_discordant}"
                )
            if not favorable_table_ok:
                reasons.append(
                    f"favorable table tasks {len(favorable_table_tasks)} < "
                    f"minimum {min_favorable_table}"
                )
            if not adverse_table_ok:
                reasons.append(
                    f"adverse table tasks {len(adverse_table_tasks)} > "
                    f"maximum {max_adverse_table}"
                )
            if not favorable_form_ok:
                reasons.append(
                    f"favorable form tasks {len(favorable_form_tasks)} < "
                    f"minimum {min_favorable_form}"
                )
            if not adverse_form_ok:
                reasons.append(
                    f"adverse form tasks {len(adverse_form_tasks)} > "
                    f"maximum {max_adverse_form}"
                )
            if not stable_table_ok:
                reasons.append(
                    f"stable table-only tasks {len(stable_table_only)} < "
                    f"minimum {min_stable_table}"
                )
            if not stable_form_ok:
                reasons.append(
                    f"stable form-only tasks {len(stable_form_only)} < "
                    f"minimum {min_stable_form}"
                )
        else:
            decision = "confirmation_pass"
            passed = True
    else:
        decision = "invalid"
        passed = False
        reasons.append(f"unknown stage: {stage!r}")

    # ---- 7. assemble report -------------------------------------------------
    checks = {
        "manifest_valid": True,
        "protocol_valid": True,
        "preflight_identity_valid": runtime_preflight_ok,
        "registry_hash_match": True,
        "completeness": (
            not structural_errors
            and infrastructure_errors == 0
            and runtime_preflight_ok
        ),
        "mechanics_valid": mechanics_valid,
    }
    if stage == "mechanics_dry_run":
        checks["stage_size"] = size_ok if 'size_ok' in dir() else False
        if 'size_ok' in dir():
            checks["stage_size"] = size_ok
        else:
            checks["stage_size"] = True
    elif stage == "outcome_screen":
        if 'size_ok' in dir():
            checks["stage_size"] = size_ok
        else:
            checks["stage_size"] = True
        checks["discordance_ok"] = len(discordant_cells) <= max_discordant
        checks["stable_table_only_ok"] = len(stable_table_only) >= min_stable_table
        checks["stable_form_only_ok"] = len(stable_form_only) >= min_stable_form
        checks["arm_table_ok"] = arm_table_ok
        checks["arm_form_ok"] = arm_form_ok
        checks["balance_ok"] = balance_ok
    elif stage == "replication_screen":
        if 'size_ok' in dir():
            checks["stage_size"] = size_ok
        else:
            checks["stage_size"] = True
        checks["discordance_ok"] = discordance_ok
        checks["favorable_table_ok"] = favorable_table_ok
        checks["adverse_table_ok"] = adverse_table_ok
        checks["favorable_form_ok"] = favorable_form_ok
        checks["adverse_form_ok"] = adverse_form_ok
        checks["stable_table_only_ok"] = stable_table_ok
        checks["stable_form_only_ok"] = stable_form_ok

    if not passed:
        for name, val in list(checks.items()):
            if not val:
                if name not in reasons and name != "stage_size":
                    reasons.append(f"check failed: {name}")

    # deduplicate and sort reasons
    reasons = sorted(set(reasons))

    return {
        "gate": report_schema,
        "schema_version": report_schema,
        "manifest_hash": manifest_hash,
        "stage": stage,
        "passed": passed,
        "decision": decision,
        "checks": checks,
        "reasons": reasons,
        "completeness": {
            "records": len(records),
            "expected_panels": len(manifest["panels"]),
            "infrastructure_errors": infrastructure_errors,
            "structural_errors_count": len(structural_errors),
            "mechanisms_errors_count": len(mechanisms_errors),
            "unique_attempt_ids": len(attempt_ids),
            "runtime_preflight": runtime_preflight_ok,
        },
        "mechanism": {
            "structural_errors": structural_errors,
            "mechanisms_errors": mechanisms_errors,
        },
        "stability": stability,
        "descriptive_outcomes": descriptive_outcomes,
        "warning": (
            "This is a screening / futility gate for the semantic-capability "
            "canary only; qualifies capability family only, not allocator/causal/"
            "generalization evidence. It cannot justify a final treatment or "
            "policy selection."
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyreplab-m3-semantic-capability-gate"
    )
    parser.add_argument("results", help="Path to screen results JSONL")
    parser.add_argument("--manifest", required=True, help="Path to screen manifest JSON")
    parser.add_argument("--registry", required=True, help="Path to treatment registry")
    parser.add_argument("--policy-split", required=True, help="Path to policy split JSON")
    parser.add_argument(
        "--preflight",
        default=None,
        help="Path to preflight JSON (default: results.jsonl.preflight.json)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the gate report JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        results_path = Path(args.results).expanduser().resolve()
        preflight_path = (
            Path(args.preflight).expanduser().resolve()
            if args.preflight
            else results_path.with_suffix(results_path.suffix + ".preflight.json")
        )
        manifest = _load_json(args.manifest)
        registry = TreatmentRegistry.load(args.registry)
        policy_split = _load_json(args.policy_split)
        records = _load_semantic_capability_records(results_path, manifest)
        preflight = _load_json(preflight_path) if preflight_path.is_file() else None

        report = evaluate_semantic_capability_gate(
            manifest,
            registry,
            policy_split,
            records,
            preflight=preflight,
        )
        if args.output:
            write_json(Path(args.output).expanduser().resolve(), report)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"semantic capability gate error: {error}", file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2, sort_keys=True))
    if report["decision"] == "invalid":
        return 1
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GATE_SCHEMA",
    "GATE_SCHEMA_V2",
    "PROTOCOL_SCHEMA_V1",
    "PROTOCOL_SCHEMA_V2",
    "DATASET_CONTRACT_SCHEMA",
    "_V2_REPLICATION_DECISION_RULE",
    "evaluate_semantic_capability_gate",
    "main",
    "_load_semantic_capability_records",
    "_validate_semantic_capability_protocol",
    "_assess_semantic_specialist_adherence",
    "_is_infrastructure_error",
]
