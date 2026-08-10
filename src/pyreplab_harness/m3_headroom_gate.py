"""Executable go/no-go gate for the frozen 96-attempt M3 headroom pilot."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .io_utils import write_json
from .m3_adherence import assess_policy_adherence
from .m3_pilot import validate_headroom_manifest
from .orchestrator import policy_spec_from_treatment
from .treatments import TreatmentRegistry


def _load_json(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def read_pilot_records(path: str | Path) -> list[dict[str, Any]]:
    """Read a complete pilot JSONL without tolerating malformed lines."""
    source = Path(path).expanduser().resolve()
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid pilot JSONL line {line_number}: {error}") from error
        if not isinstance(record, dict):
            raise ValueError(f"pilot JSONL line {line_number} is not an object")
        records.append(record)
    return records


def _rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def _uniform_tie_score(
    policies: Mapping[str, Mapping[int, bool]],
    ordered_bundles: list[str],
    selector_replica: int,
    evaluation_replica: int,
) -> float:
    """Score the evaluation replica under uniform ties on selector outcomes."""
    best = max(int(policies[bundle].get(selector_replica, False)) for bundle in ordered_bundles)
    winners = [
        bundle
        for bundle in ordered_bundles
        if int(policies[bundle].get(selector_replica, False)) == best
    ]
    return sum(
        int(policies[bundle].get(evaluation_replica, False)) for bundle in winners
    ) / len(winners)


def evaluate_headroom_gate(
    manifest: Mapping[str, Any],
    registry: TreatmentRegistry,
    policy_split: Mapping[str, Any],
    records: list[dict[str, Any]],
    *,
    preflight: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Evaluate completeness, headroom, cost, and manipulation checks."""
    validate_headroom_manifest(manifest, registry, policy_split)
    manifest_hash = str(manifest["manifest_hash"])
    expected_tasks = {str(task["task_id"]): task for task in manifest["tasks"]}
    headroom_task_ids = {
        task_id
        for task_id, task in expected_tasks.items()
        if task.get("template") != "distractor_recovery"
    }
    expected_panels = {
        str(panel["panel_id"]): panel for panel in manifest["panels"]
    }
    policy_labels = dict(manifest["policy_labels"])
    expected_bundles = set(policy_labels.values())
    gates = manifest["gates"]
    reasons: list[str] = []

    runtime_preflight = bool(
        preflight
        and preflight.get("pilot_manifest_hash") == manifest_hash
        and preflight.get("runtime_pins") == manifest.get("runtime_pins")
        and isinstance(preflight.get("code_revision"), str)
        and len(str(preflight.get("code_revision"))) == 40
        and isinstance(preflight.get("source_tree_hash"), str)
        and len(str(preflight.get("source_tree_hash"))) == 64
    )

    records_by_key: dict[str, dict[str, Any]] = {}
    infrastructure_errors = 0
    structural_errors: list[str] = []
    for record in records:
        key = str(record.get("key"))
        if key in records_by_key:
            reasons.append(f"duplicate panel result: {key}")
            continue
        records_by_key[key] = record
        if record.get("schema_version") != "m3-headroom-task-result-v1":
            structural_errors.append(f"{key}: result schema mismatch")
        if record.get("status") == "error":
            infrastructure_errors += 1

    missing_panels = sorted(set(expected_panels) - set(records_by_key))
    extra_panels = sorted(set(records_by_key) - set(expected_panels))
    attempts: list[dict[str, Any]] = []
    attempt_ids: set[str] = set()
    task_outcomes: dict[str, dict[str, dict[int, bool]]] = {
        task_id: {bundle_id: {} for bundle_id in expected_bundles}
        for task_id in expected_tasks
    }

    for panel_id, panel in expected_panels.items():
        task_id = str(panel["task_id"])
        task = expected_tasks[task_id]
        replica = int(panel["rollout_replica"])
        record = records_by_key.get(panel_id)
        if record is None or record.get("status") != "completed":
            continue
        if record.get("pilot_manifest_hash") != manifest_hash:
            structural_errors.append(f"{panel_id}: manifest hash mismatch")
            continue
        if record.get("task") != task:
            structural_errors.append(f"{panel_id}: frozen task entry mismatch")
            continue
        if record.get("panel") != panel:
            structural_errors.append(f"{panel_id}: frozen panel entry mismatch")
            continue
        result = record.get("result")
        if not isinstance(result, Mapping):
            structural_errors.append(f"{panel_id}: result is missing")
            continue
        expected_order = [policy_labels[label] for label in panel["execution_order"]]
        if result.get("task_id") != task_id:
            structural_errors.append(f"{panel_id}: generated task id mismatch")
        if result.get("treatment_registry_hash") != registry.registry_hash:
            structural_errors.append(f"{panel_id}: registry hash mismatch")
        if result.get("execution_order") != expected_order:
            structural_errors.append(f"{panel_id}: execution order mismatch")
        if result.get("rollout_replica") != replica:
            structural_errors.append(f"{panel_id}: rollout replica mismatch")
        if result.get("sampling_seed") != panel.get("sampling_seed"):
            structural_errors.append(f"{panel_id}: sampling seed mismatch")
        if result.get("pilot_manifest_hash") != manifest_hash:
            structural_errors.append(f"{panel_id}: result manifest hash mismatch")
        if result.get("pilot_panel_id") != panel_id:
            structural_errors.append(f"{panel_id}: result panel id mismatch")
        result_attempts = result.get("attempts")
        if not isinstance(result_attempts, Mapping) or set(result_attempts) != expected_bundles:
            structural_errors.append(f"{panel_id}: policy panel is incomplete")
            continue

        for bundle_id in expected_order:
            item = result_attempts[bundle_id]
            if not isinstance(item, Mapping):
                structural_errors.append(f"{panel_id}/{bundle_id}: attempt is malformed")
                continue
            attempt_id = item.get("attempt_id")
            if not isinstance(attempt_id, str) or not attempt_id:
                structural_errors.append(f"{panel_id}/{bundle_id}: attempt id missing")
                continue
            if attempt_id in attempt_ids:
                structural_errors.append(f"duplicate attempt id: {attempt_id}")
            attempt_ids.add(attempt_id)
            treatment = registry.by_bundle_id(bundle_id)
            if item.get("policy") != policy_spec_from_treatment(treatment).to_dict():
                structural_errors.append(
                    f"{panel_id}/{bundle_id}: executed policy mismatch"
                )
                continue
            verification = item.get("verification")
            if not isinstance(verification, Mapping) or not isinstance(
                verification.get("success"), bool
            ):
                structural_errors.append(f"{panel_id}/{bundle_id}: outcome missing")
                continue
            runtime_pins = manifest["runtime_pins"]
            if (
                verification.get("verifier_id")
                != runtime_pins["fixture_verifier_id"]
                or verification.get("verifier_version")
                != runtime_pins["fixture_verifier_version"]
            ):
                structural_errors.append(
                    f"{panel_id}/{bundle_id}: verifier identity mismatch"
                )
                continue
            usage = item.get("usage")
            output_cost = usage.get("output") if isinstance(usage, Mapping) else None
            if (
                isinstance(output_cost, bool)
                or not isinstance(output_cost, (int, float))
                or not math.isfinite(float(output_cost))
                or float(output_cost) < 0
            ):
                structural_errors.append(f"{panel_id}/{bundle_id}: output cost missing")
                continue
            trajectory = item.get("trajectory")
            if not isinstance(trajectory, Mapping):
                structural_errors.append(f"{panel_id}/{bundle_id}: trajectory missing")
                continue
            planning_preamble = trajectory.get("planning_preamble")
            tool_trace = trajectory.get("tool_trace")
            provider_turn_count = trajectory.get("provider_turn_count")
            if not isinstance(planning_preamble, Mapping) or not isinstance(
                tool_trace, list
            ) or isinstance(provider_turn_count, bool) or not isinstance(
                provider_turn_count, int
            ) or provider_turn_count < 0 or any(
                not isinstance(entry, Mapping)
                or not isinstance(entry.get("tool_name"), str)
                or not isinstance(entry.get("is_error"), bool)
                or not isinstance(entry.get("budget_rejected"), bool)
                or not isinstance(entry.get("details"), Mapping)
                for entry in tool_trace
            ):
                structural_errors.append(
                    f"{panel_id}/{bundle_id}: trajectory is malformed"
                )
                continue
            expected_sampling_receipt = {
                "seed": panel["sampling_seed"],
                "parameters": manifest["runtime_pins"]["sampling"]["parameters"],
            }
            if provider_turn_count > 0 and item.get(
                "sampling_receipt"
            ) != expected_sampling_receipt:
                structural_errors.append(
                    f"{panel_id}/{bundle_id}: provider sampling receipt mismatch"
                )
                continue
            success = bool(verification["success"])
            task_outcomes[task_id][bundle_id][replica] = success
            recovery_probe_url = task.get("recovery_probe_url")
            recovery_probe_status = task.get("recovery_probe_status")
            attempts.append(
                {
                    "task_id": task_id,
                    "task_template": task["template"],
                    "panel_id": panel_id,
                    "rollout_replica": replica,
                    "bundle_id": bundle_id,
                    "success": success,
                    "output_cost": float(output_cost),
                    "adherence": assess_policy_adherence(
                        treatment,
                        trajectory,
                        required_recovery_probe_url=(
                            str(recovery_probe_url)
                            if isinstance(recovery_probe_url, str)
                            else None
                        ),
                        required_recovery_probe_status=(
                            int(recovery_probe_status)
                            if isinstance(recovery_probe_status, int)
                            and not isinstance(recovery_probe_status, bool)
                            else None
                        ),
                    ),
                }
            )

    for task_id, policies in task_outcomes.items():
        for bundle_id, replicas in policies.items():
            if set(replicas) != {0, 1}:
                structural_errors.append(
                    f"{task_id}/{bundle_id}: rollout replicas are incomplete"
                )

    complete = bool(
        runtime_preflight
        and not missing_panels
        and not extra_panels
        and infrastructure_errors == 0
        and not structural_errors
        and len(records) == int(gates["panels"])
        and len(attempts) == int(gates["attempts"])
        and len(attempt_ids) == int(gates["attempts"])
    )

    all_successes_by_policy: dict[str, int] = {}
    successes_by_policy: dict[str, int] = {}
    costs_by_policy: dict[str, list[float]] = {}
    for bundle_id in expected_bundles:
        policy_attempts = [attempt for attempt in attempts if attempt["bundle_id"] == bundle_id]
        headroom_attempts = [
            attempt
            for attempt in policy_attempts
            if attempt["task_id"] in headroom_task_ids
        ]
        all_successes_by_policy[bundle_id] = sum(
            attempt["success"] for attempt in policy_attempts
        )
        successes_by_policy[bundle_id] = sum(
            attempt["success"] for attempt in headroom_attempts
        )
        costs_by_policy[bundle_id] = [
            attempt["output_cost"] for attempt in headroom_attempts
        ]
    maximum_nondegenerate_successes = len(headroom_task_ids) * 2 - 1
    nondegenerate = complete and all(
        1 <= successes <= maximum_nondegenerate_successes
        for successes in successes_by_policy.values()
    )

    repeated_cells = [
        replicas
        for policies in task_outcomes.values()
        for replicas in policies.values()
        if set(replicas) == {0, 1}
    ]
    discordant_cells = sum(replicas[0] != replicas[1] for replicas in repeated_cells)
    repeat_discordance_rate = (
        discordant_cells / len(repeated_cells) if repeated_cells else None
    )
    repeat_concordance = bool(
        complete
        and repeat_discordance_rate is not None
        and repeat_discordance_rate
        <= float(gates["maximum_repeat_discordance_rate"])
    )

    stable_disagreement_tasks = 0
    for task_id in sorted(headroom_task_ids):
        policies = task_outcomes[task_id]
        stable_success = any(
            replicas.get(0) is True and replicas.get(1) is True
            for replicas in policies.values()
        )
        stable_failure = any(
            replicas.get(0) is False and replicas.get(1) is False
            for replicas in policies.values()
        )
        stable_disagreement_tasks += stable_success and stable_failure
    stable_disagreement = complete and stable_disagreement_tasks >= int(
        gates["minimum_stable_disagreement_tasks"]
    )

    ordered_bundles = [policy_labels[label] for label in ("A", "B", "C", "D")]
    cross_replica_total = 0.0
    for task_id in sorted(headroom_task_ids):
        policies = task_outcomes[task_id]
        for selector_replica, evaluation_replica in ((0, 1), (1, 0)):
            cross_replica_total += _uniform_tie_score(
                policies,
                ordered_bundles,
                selector_replica,
                evaluation_replica,
            )
    cross_replica_successes = cross_replica_total / 2.0
    best_fixed_successes = max(successes_by_policy.values(), default=0) / 2.0
    cross_replica_lift_successes = cross_replica_successes - best_fixed_successes
    cross_replica_lift = complete and cross_replica_lift_successes >= float(
        gates["minimum_cross_replica_lift_successes"]
    )

    policy_mean_costs = {
        bundle_id: sum(costs) / len(costs) if costs else None
        for bundle_id, costs in costs_by_policy.items()
    }
    finite_means = [value for value in policy_mean_costs.values() if value is not None]
    cost_mean_ratio = (
        max(finite_means) / min(finite_means)
        if len(finite_means) == 4 and min(finite_means) > 0
        else None
    )
    cost_range = bool(
        complete
        and cost_mean_ratio is not None
        and cost_mean_ratio >= float(gates["minimum_cost_mean_ratio"])
    )

    adherence_rows = [attempt["adherence"] for attempt in attempts]
    planning_rates = {
        level: _rate(
            [
                bool(row["planning_adherent"])
                for row in adherence_rows
                if row["planning_level"] == level
            ]
        )
        for level in ("direct", "brief_plan", "decompose")
    }
    planning_rate = _rate(
        [bool(row["planning_adherent"]) for row in adherence_rows]
    )
    planning_adherence = bool(
        complete
        and all(
            rate is not None
            and rate >= float(gates["minimum_planning_adherence"])
            for rate in planning_rates.values()
        )
    )
    observation_rates = {
        level: _rate(
            [
                bool(row["observation_adherent"])
                for row in adherence_rows
                if row["observation_level"] == level
            ]
        )
        for level in ("text_first", "structure_first", "targeted_query_first")
    }
    observation_rate = _rate([bool(row["observation_adherent"]) for row in adherence_rows])
    observation_adherence = bool(
        complete
        and all(
            rate is not None
            and rate >= float(gates["minimum_observation_adherence"])
            for rate in observation_rates.values()
        )
    )

    repeated_read_rates_itt = {
        level: _rate(
            [
                bool(row["repeated_final_read"])
                for row in adherence_rows
                if row["verification_level"] == level
            ]
        )
        for level in ("submit_directly", "final_reobserve")
    }
    verification_opportunity_counts = {
        level: sum(
            row["verification_level"] == level and row["verification_opportunity"]
            for row in adherence_rows
        )
        for level in ("submit_directly", "final_reobserve")
    }
    repeated_read_rates = {
        level: _rate(
            [
                bool(row["repeated_final_read"])
                for row in adherence_rows
                if row["verification_level"] == level
                and row["verification_opportunity"]
            ]
        )
        for level in ("submit_directly", "final_reobserve")
    }
    verification_difference = None
    if all(value is not None for value in repeated_read_rates.values()):
        verification_difference = (
            float(repeated_read_rates["final_reobserve"])
            - float(repeated_read_rates["submit_directly"])
        )
    verification_separation = bool(
        complete
        and min(verification_opportunity_counts.values(), default=0)
        >= int(gates["minimum_verification_opportunities_per_level"])
        and verification_difference is not None
        and verification_difference
        >= float(gates["minimum_verification_rate_difference"])
    )

    recovery_eligible_counts: dict[str, int] = {}
    recovery_retry_rates: dict[str, float | None] = {}
    for level in ("fail_fast", "diagnose_retry_once"):
        eligible = [
            row
            for row in adherence_rows
            if row["recovery_level"] == level
            and row["recovery_probe_required"]
            and row["recovery_eligible"]
        ]
        recovery_eligible_counts[level] = len(eligible)
        recovery_retry_rates[level] = _rate(
            [bool(row["successful_same_tool_retry"]) for row in eligible]
        )
    recovery_difference = None
    if all(value is not None for value in recovery_retry_rates.values()):
        recovery_difference = (
            float(recovery_retry_rates["diagnose_retry_once"])
            - float(recovery_retry_rates["fail_fast"])
        )
    required_recovery = int(gates["required_recovery_eligible_per_level"])
    recovery_separation = bool(
        complete
        and all(
            count == required_recovery
            for count in recovery_eligible_counts.values()
        )
        and recovery_difference is not None
        and recovery_difference >= float(gates["minimum_recovery_rate_difference"])
    )
    tool_cap_rate = _rate(
        [bool(row["tool_cap_compliant"]) for row in adherence_rows]
    )
    tool_cap_compliance = bool(
        complete
        and tool_cap_rate is not None
        and tool_cap_rate >= float(gates["tool_cap_compliance"])
    )

    checks = {
        "complete": complete,
        "nondegenerate": nondegenerate,
        "repeat_concordance": repeat_concordance,
        "stable_disagreement": stable_disagreement,
        "cross_replica_lift": cross_replica_lift,
        "cost_range": cost_range,
        "planning_adherence": planning_adherence,
        "observation_adherence": observation_adherence,
        "verification_separation": verification_separation,
        "recovery_separation": recovery_separation,
        "tool_cap_compliance": tool_cap_compliance,
    }
    reasons.extend(name for name, passed in checks.items() if not passed)
    return {
        "gate": "m3-headroom-pilot-v1",
        "manifest_hash": manifest_hash,
        "passed": all(checks.values()),
        "checks": checks,
        "reasons": sorted(set(reasons)),
        "completeness": {
            "records": len(records),
            "attempts": len(attempts),
            "unique_attempt_ids": len(attempt_ids),
            "missing_panels": missing_panels,
            "extra_panels": extra_panels,
            "infrastructure_errors": infrastructure_errors,
            "structural_errors": structural_errors,
            "runtime_preflight": runtime_preflight,
        },
        "headroom": {
            "headroom_task_count": len(headroom_task_ids),
            "headroom_excluded_templates": ["distractor_recovery"],
            "successes_by_policy": successes_by_policy,
            "all_task_successes_by_policy": all_successes_by_policy,
            "discordant_policy_task_cells": discordant_cells,
            "repeat_discordance_rate": repeat_discordance_rate,
            "stable_disagreement_tasks": stable_disagreement_tasks,
            "two_direction_cross_replica_successes": cross_replica_successes,
            "best_fixed_successes": best_fixed_successes,
            "cross_replica_lift_successes": cross_replica_lift_successes,
            "policy_mean_output_tokens": policy_mean_costs,
            "cost_mean_ratio": cost_mean_ratio,
        },
        "manipulation": {
            "planning_adherence_rates": planning_rates,
            "planning_adherence_rate": planning_rate,
            "observation_adherence_rates": observation_rates,
            "observation_adherence_rate": observation_rate,
            "repeated_read_rates_itt": repeated_read_rates_itt,
            "repeated_read_rates": repeated_read_rates,
            "verification_opportunity_counts": verification_opportunity_counts,
            "verification_rate_difference": verification_difference,
            "recovery_eligible_counts": recovery_eligible_counts,
            "same_tool_retry_rates": recovery_retry_rates,
            "recovery_rate_difference": recovery_difference,
            "tool_cap_compliance_rate": tool_cap_rate,
        },
        "warning": (
            "This gate establishes headroom and manipulation behavior only; "
            "it is not allocator-effectiveness evidence."
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyreplab-m3-headroom-gate")
    parser.add_argument("results")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--policy-split", required=True)
    parser.add_argument("--preflight")
    parser.add_argument("--output")
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
        records = read_pilot_records(results_path)
        preflight = _load_json(preflight_path) if preflight_path.is_file() else None
        report = evaluate_headroom_gate(
            manifest,
            registry,
            policy_split,
            records,
            preflight=preflight,
        )
        if args.output:
            write_json(Path(args.output).expanduser().resolve(), report)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"headroom gate error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["evaluate_headroom_gate", "main", "read_pilot_records"]
