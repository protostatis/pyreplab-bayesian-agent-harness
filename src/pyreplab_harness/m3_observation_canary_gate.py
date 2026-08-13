"""Executable screening/futility gate for the observation-enforcement canary.

This gate validates that the first-observation receipt mechanism is
mechanically functional across two treatments whose only factor-level
difference is observation (text_first vs structure_first).  It is
exclusively a screening and futility detector; it is not causal or
allocator-effectiveness evidence.

CLI::

    pyreplab-m3-observation-canary-gate RESULTS \\
        --manifest MANIFEST --registry REGISTRY --policy-split SPLIT \\
        [--preflight PREFLIGHT] [--output OUTPUT]

Exit codes: 0 = pass, 2 = valid no-go, 1 = invalid / exception.
"""

from __future__ import annotations

import argparse
import json
import math
import string
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .io_utils import write_json
from .m3_adherence import _OBSERVATION_ACTION, assess_policy_adherence
from .m3_exploratory_screen import (
    PANEL_RESULT_SCHEMA,
    _validate_panel_result_strict,
    validate_screen_manifest,
)
from .m3_pilot import _load_json
from .orchestrator import (
    UNBROWSER_INTERACTIVE_STRUCTURE_FIRST_INTERFACE,
    UNBROWSER_INTERACTIVE_TEXT_FIRST_INTERFACE,
    policy_spec_from_treatment,
)
from .treatments import TreatmentRegistry

GATE_SCHEMA = "m3-observation-canary-gate-v1"

_FACTOR_NAMES = ("planning", "observation", "verification", "recovery", "tool_cap")

_INFRA_ERROR_MARKERS = (
    "BrokenPipeError",
    "ConnectionResetError",
    "browser process exited",
    "process connection broken",
    "response timed out",
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _is_hex_digest(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in string.hexdigits for character in value)
    )


def _is_infrastructure_error(entry: Mapping[str, Any]) -> bool:
    """Return whether one tool result records a browser/transport failure."""
    details = entry.get("details")
    if not isinstance(details, Mapping):
        return False
    if details.get("infrastructure_error") is True:
        return True
    if details.get("infrastructure_error") is False:
        return False
    error_text = str(details.get("error", "")).casefold()
    return any(marker.casefold() in error_text for marker in _INFRA_ERROR_MARKERS)


# ---------------------------------------------------------------------------
# protocol validation
# ---------------------------------------------------------------------------


def _validate_canary_protocol(
    manifest: Mapping[str, Any],
    registry: TreatmentRegistry,
) -> tuple[list[Any], list[Any]]:
    """Validate the observation-canary protocol contract and return
    ``(text_treatments, structure_treatments)``, each containing exactly
    one ``TreatmentSpec``."""
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
        raise ValueError("; ".join(errors))  # unreachable but safe
    if protocol.get("schema_version") != "m3-observation-canary-protocol-v1":
        errors.append(
            f"protocol.schema_version must be m3-observation-canary-protocol-v1, "
            f"got {protocol.get('schema_version')!r}"
        )

    # ---- stage --------------------------------------------------------------
    stage = protocol.get("stage")
    if stage not in ("mechanics_dry_run", "outcome_screen"):
        errors.append(
            f"protocol.stage must be mechanics_dry_run or outcome_screen, "
            f"got {stage!r}"
        )

    mechanism = protocol.get("mechanism")
    if not isinstance(mechanism, Mapping):
        errors.append("protocol.mechanism must be an object")
    else:
        expected_mechanism = {
            "name": "auto_delivered_first_observation",
            "receipt_schema_version": "pyreplab-required-first-observation-v1",
            "combined_navigation_observation_tool_call": True,
            "text_selector": "body",
            "later_cross_modal_observations_allowed": True,
        }
        for key, expected in expected_mechanism.items():
            if mechanism.get(key) != expected:
                errors.append(
                    f"protocol.mechanism.{key} must be {expected!r}, "
                    f"got {mechanism.get(key)!r}"
                )

    if protocol.get("claim_boundary") != "screening_futility_only":
        errors.append(
            "protocol.claim_boundary must be 'screening_futility_only'"
        )

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
            "maximum_discordant_cells": 1,
            "minimum_stable_text_only_tasks": 1,
            "minimum_stable_structure_only_tasks": 1,
        }
        for key, expected in expected_rules.items():
            value = decision_rule.get(key)
            if isinstance(value, bool) or value != expected:
                errors.append(
                    f"protocol.decision_rule.{key} must equal {expected}, got {value!r}"
                )

    # ---- exactly two policy bundle IDs --------------------------------------
    bundle_ids = manifest.get("policy_bundle_ids", [])
    if not isinstance(bundle_ids, list) or len(bundle_ids) != 2:
        errors.append("manifest must have exactly 2 policy_bundle_ids")

    if errors:
        raise ValueError("; ".join(errors))

    # ---- fetch treatments ---------------------------------------------------
    treatments = list(registry.by_bundle_id(str(bid)) for bid in bundle_ids)

    # ---- observation levels must be text_first and structure_first -----------
    obs_levels = set()
    for t in treatments:
        obs = str(t.generator_metadata.get("observation", ""))
        obs_levels.add(obs)
    if obs_levels != {"text_first", "structure_first"}:
        errors.append(
            f"observation levels must be exactly {{text_first, structure_first}}, "
            f"got {sorted(obs_levels)}"
        )

    # ---- all other five-factor dimensions equal ------------------------------
    text_t = [t for t in treatments
              if t.generator_metadata.get("observation") == "text_first"]
    structure_t = [t for t in treatments
                   if t.generator_metadata.get("observation") == "structure_first"]
    if len(text_t) != 1 or len(structure_t) != 1:
        errors.append("must have exactly one text_first and one structure_first treatment")
    else:
        for factor in _FACTOR_NAMES:
            if factor == "observation":
                continue
            if text_t[0].generator_metadata.get(factor) != structure_t[0].generator_metadata.get(factor):
                errors.append(
                    f"non-observation factor {factor!r} differs between treatments"
                )

    # ---- tool interfaces must be exact --------------------------------------
    if text_t:
        ti_text = text_t[0].tool_interface
        if ti_text != UNBROWSER_INTERACTIVE_TEXT_FIRST_INTERFACE:
            errors.append(
                f"text_first treatment must use "
                f"native_bash_unbrowser_interactive_text_first_v1, "
                f"got {ti_text!r}"
            )
    if structure_t:
        ti_struct = structure_t[0].tool_interface
        if ti_struct != UNBROWSER_INTERACTIVE_STRUCTURE_FIRST_INTERFACE:
            errors.append(
                f"structure_first treatment must use "
                f"native_bash_unbrowser_interactive_structure_first_v1, "
                f"got {ti_struct!r}"
            )

    # ---- metadata: parent_bundle_id and observation_mechanism ---------------
    for t in treatments:
        meta = t.generator_metadata
        if not isinstance(meta.get("parent_bundle_id"), str) or not meta["parent_bundle_id"].strip():
            errors.append(f"treatment {t.bundle_id} missing parent_bundle_id in metadata")
        if meta.get("observation_mechanism") != "auto_delivered_first_observation":
            errors.append(
                f"treatment {t.bundle_id} observation_mechanism must be "
                f"auto_delivered_first_observation, got {meta.get('observation_mechanism')!r}"
            )

    parent_bundle_ids = protocol.get("parent_bundle_ids")
    if not isinstance(parent_bundle_ids, Mapping):
        errors.append("protocol.parent_bundle_ids must be an object")
    else:
        for level, treatment_list in (
            ("text_first", text_t),
            ("structure_first", structure_t),
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

    if len(text_t) == 1 and len(structure_t) == 1:
        comparable_fields = (
            "allowed_tools",
            "max_output_tokens",
            "tool_call_limit",
            "command_timeout_seconds",
            "wall_time_limit_seconds",
        )
        for field in comparable_fields:
            if getattr(text_t[0], field) != getattr(structure_t[0], field):
                errors.append(f"treatment resource field {field!r} differs")

    if errors:
        raise ValueError("; ".join(errors))

    return text_t, structure_t


# ---------------------------------------------------------------------------
# record loading
# ---------------------------------------------------------------------------


def _load_canary_records(
    path: Path,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Load strictly-validated panel results.

    Uses the same ``_validate_panel_result_strict`` from the exploratory
    screen for base validation (schema, hash, structure), then returns
    the validated records.
    """
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


def evaluate_observation_canary_gate(
    manifest: Mapping[str, Any],
    registry: TreatmentRegistry,
    policy_split: Mapping[str, Any],
    records: list[dict[str, Any]],
    *,
    preflight: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Evaluate the observation-enforcement canary gate.

    Returns a deterministic machine-readable report.
    """
    # ---- 1. validate manifest / registry / split ----------------------------
    validate_screen_manifest(manifest, registry, policy_split)
    manifest_hash = str(manifest["manifest_hash"])

    # ---- 2. validate canary protocol ----------------------------------------
    text_treatments, structure_treatments = _validate_canary_protocol(manifest, registry)

    text_treatment = text_treatments[0]
    structure_treatment = structure_treatments[0]
    text_bid = text_treatment.bundle_id
    structure_bid = structure_treatment.bundle_id
    bundle_ids = [text_bid, structure_bid]

    protocol = manifest["protocol"]
    stage = protocol["stage"]

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
    bundle_by_id: dict[str, Any] = {
        text_bid: text_treatment,
        structure_bid: structure_treatment,
    }

    task_by_id: dict[str, dict[str, Any]] = {
        str(t["task_id"]): t for t in manifest["tasks"]
    }
    panel_by_id: dict[str, dict[str, Any]] = {
        str(p["panel_id"]): p for p in manifest["panels"]
    }
    expected_bundle_set = {text_bid, structure_bid}

    reasons: list[str] = []
    mechanisms_errors: list[str] = []

    attempt_ids: set[str] = set()
    task_outcomes: dict[str, dict[str, dict[int, bool]]] = {}
    output_costs: dict[str, list[float]] = {
        text_bid: [],
        structure_bid: [],
    }

    for task in manifest["tasks"]:
        tid = str(task["task_id"])
        task_outcomes[tid] = {bid: {} for bid in bundle_ids}

    infrastructure_errors = 0
    structural_errors: list[str] = []

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

            # assess adherence (includes first-observation receipt check)
            adh = assess_policy_adherence(
                treatment,
                traj,
            )

            if isinstance(traj, Mapping):
                trace_entries = traj.get("tool_trace")
                if isinstance(trace_entries, list):
                    attempt_infrastructure_errors = sum(
                        1
                        for entry in trace_entries
                        if isinstance(entry, Mapping)
                        and _is_infrastructure_error(entry)
                    )
                    if attempt_infrastructure_errors:
                        infrastructure_errors += attempt_infrastructure_errors
                        mechanisms_errors.append(
                            f"{pid}/{bid}: infrastructure error detected "
                            f"({attempt_infrastructure_errors} entries)"
                        )

            # first-observation receipt must be valid
            if adh.get("first_observation_receipt_valid") is not True:
                mechanisms_errors.append(
                    f"{pid}/{bid}: first-observation receipt invalid or missing"
                )

            # expected observation must match treatment
            expected_obs_level = str(treatment.generator_metadata.get("observation", ""))
            expected_obs_action = _OBSERVATION_ACTION.get(expected_obs_level)
            if adh.get("expected_first_observation") != expected_obs_action:
                mechanisms_errors.append(
                    f"{pid}/{bid}: expected_observation mismatch "
                    f"(treatment says {expected_obs_action!r}, "
                    f"assess says {adh.get('expected_first_observation')!r})"
                )
            if adh.get("first_observation") != expected_obs_action:
                mechanisms_errors.append(
                    f"{pid}/{bid}: first_observation {adh.get('first_observation')!r} "
                    f"!= expected {expected_obs_action!r}"
                )

            # observation_adherent is required
            if adh.get("observation_adherent") is not True:
                mechanisms_errors.append(
                    f"{pid}/{bid}: observation not adherent"
                )

            # tool_cap_compliant
            if adh.get("tool_cap_compliant") is not True:
                mechanisms_errors.append(
                    f"{pid}/{bid}: tool_cap not compliant "
                    f"(admitted={adh.get('admitted_tool_call_count')}, "
                    f"cap={adh.get('tool_cap')})"
                )

            # record success for stability analysis
            if isinstance(verif, Mapping):
                success = bool(verif.get("success", False))
                task_outcomes[task_id][bid][replica] = success

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
        # compute gate count assertions
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
        # compute count assertions
        expected_tasks = 6
        expected_replicas = 2
        expected_panels = 12
        expected_attempts = 24

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

        # ---- discordant cells -----------------------------------------------
        # A policy-task cell is discordant when two replicas disagree
        # (one success, one failure).
        discordant_cells: list[str] = []
        for tid, policies in task_outcomes.items():
            for bid in bundle_ids:
                replicas = policies[bid]
                if set(replicas.keys()) == {0, 1}:
                    if replicas[0] != replicas[1]:
                        discordant_cells.append(f"{tid}/{bid}")

        decision_rule = protocol["decision_rule"]
        max_discordant = int(decision_rule["maximum_discordant_cells"])

        # ---- stable tasks ---------------------------------------------------
        # A task is "stable text-only" when text treatment succeeds on both
        # replicas and structure treatment fails on both replicas.
        stable_text_only: list[str] = []
        # A task is "stable structure-only" when structure treatment succeeds
        # on both replicas and text treatment fails on both replicas.
        stable_structure_only: list[str] = []

        for tid, policies in task_outcomes.items():
            if set(policies[text_bid].keys()) != {0, 1} or set(policies[structure_bid].keys()) != {0, 1}:
                continue
            text_both_ok = policies[text_bid][0] and policies[text_bid][1]
            struct_both_ok = policies[structure_bid][0] and policies[structure_bid][1]
            text_both_fail = not policies[text_bid][0] and not policies[text_bid][1]
            struct_both_fail = not policies[structure_bid][0] and not policies[structure_bid][1]

            if text_both_ok and struct_both_fail:
                stable_text_only.append(tid)
            if struct_both_ok and text_both_fail:
                stable_structure_only.append(tid)

        min_stable_text = int(decision_rule["minimum_stable_text_only_tasks"])
        min_stable_structure = int(
            decision_rule["minimum_stable_structure_only_tasks"]
        )

        stability_ok = (
            len(discordant_cells) <= max_discordant
            and len(stable_text_only) >= min_stable_text
            and len(stable_structure_only) >= min_stable_structure
        )

        stability = {
            "discordant_cells": discordant_cells,
            "discordant_cell_count": len(discordant_cells),
            "maximum_discordant_cells": max_discordant,
            "stable_text_only_tasks": stable_text_only,
            "stable_text_only_count": len(stable_text_only),
            "minimum_stable_text_only_tasks": min_stable_text,
            "stable_structure_only_tasks": stable_structure_only,
            "stable_structure_only_count": len(stable_structure_only),
            "minimum_stable_structure_only_tasks": min_stable_structure,
        }

        # ---- aggregate descriptive outcomes ----------------------------------
        text_successes = 0
        struct_successes = 0
        text_total = 0
        struct_total = 0
        for tid, policies in task_outcomes.items():
            for rep in policies[text_bid]:
                text_total += 1
                if policies[text_bid][rep]:
                    text_successes += 1
            for rep in policies[structure_bid]:
                struct_total += 1
                if policies[structure_bid][rep]:
                    struct_successes += 1

        descriptive_outcomes = {
            "text_first_successes": text_successes,
            "text_first_attempts": text_total,
            "structure_first_successes": struct_successes,
            "structure_first_attempts": struct_total,
            "text_first_mean_output_tokens": (
                sum(output_costs[text_bid]) / len(output_costs[text_bid])
                if output_costs[text_bid]
                else None
            ),
            "structure_first_mean_output_tokens": (
                sum(output_costs[structure_bid]) / len(output_costs[structure_bid])
                if output_costs[structure_bid]
                else None
            ),
            "note": (
                "Aggregate arm successes are descriptive only and never a "
                "balance gate. Do not interpret as treatment-effect evidence."
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
            if len(stable_text_only) < min_stable_text:
                reasons.append(
                    f"stable text-only tasks {len(stable_text_only)} < "
                    f"minimum {min_stable_text}"
                )
            if len(stable_structure_only) < min_stable_structure:
                reasons.append(
                    f"stable structure-only tasks {len(stable_structure_only)} < "
                    f"minimum {min_stable_structure}"
                )
        else:
            decision = "screen_pass"
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
        checks["stage_size"] = size_ok
    elif stage == "outcome_screen":
        checks["stage_size"] = size_ok
        checks["discordance_ok"] = len(discordant_cells) <= max_discordant
        checks["stable_text_only_ok"] = len(stable_text_only) >= min_stable_text
        checks["stable_structure_only_ok"] = len(stable_structure_only) >= min_stable_structure

    if not passed:
        for name, val in list(checks.items()):
            if not val:
                if name not in reasons and name != "stage_size":
                    reasons.append(f"check failed: {name}")

    # deduplicate and sort reasons
    reasons = sorted(set(reasons))

    return {
        "gate": GATE_SCHEMA,
        "schema_version": GATE_SCHEMA,
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
            "This is a screening / futility gate for the observation-enforcement "
            "canary only; it is not causal or allocator-effectiveness evidence. "
            "It cannot justify a final treatment or policy selection."
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyreplab-m3-observation-canary-gate"
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
        records = _load_canary_records(results_path, manifest)
        preflight = _load_json(preflight_path) if preflight_path.is_file() else None

        report = evaluate_observation_canary_gate(
            manifest,
            registry,
            policy_split,
            records,
            preflight=preflight,
        )
        if args.output:
            write_json(Path(args.output).expanduser().resolve(), report)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"observation canary gate error: {error}", file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2, sort_keys=True))
    if report["decision"] == "invalid":
        return 1
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GATE_SCHEMA",
    "evaluate_observation_canary_gate",
    "main",
    "_load_canary_records",
    "_is_infrastructure_error",
    "_validate_canary_protocol",
]
