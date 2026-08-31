"""Sequential exploratory variable-policy M3 screen runner.

Freeze, validate, run, and analyze screens that test arbitrary subsets of
the immutable 72-policy registry as T_pilot/pilot_excluded, without
modifying the frozen m3_pilot/headroom manifest semantics.

CLI subcommands: freeze, validate, run, analyze.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .m3_adherence import assess_policy_adherence
from .m3_pilot import (
    FIXTURE_PORT,
    HELD_TEMPLATES,
    KNOWN_TEMPLATES,
    _RUNTIME_PINS,
    _canonical_hash,
    _load_json,
    _write_immutable_json,
    runtime_preflight,
)
from .meta_grammar import _GRAMMAR_VERSION
from .orchestrator import (
    RemoteConfig,
    run_registered_treatments,
)
from .treatments import TreatmentRegistry, TreatmentSpec

SCHEMA_VERSION = "m3-exploratory-screen-v1"
PANEL_RESULT_SCHEMA = "m3-exploratory-panel-result-v1"
ANALYSIS_SCHEMA_VERSION = "m3-exploratory-analysis-v1"

_FACTOR_NAMES = ("planning", "observation", "verification", "recovery", "tool_cap")

# ---------------------------------------------------------------------------
# spec / manifest helpers
# ---------------------------------------------------------------------------


def _require_nonempty_safe(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    stripped = value.strip()
    if any(ord(ch) < 32 for ch in stripped):
        raise ValueError(f"{label} must not contain control characters")
    return stripped


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer, got {value!r}")
    return value


def _validate_schedule_seed(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"schedule_seed must be a non-negative integer, got {value!r}")
    return value


def _generate_panel_sampling_seeds(
    num_panels: int,
    seed_start: int,
) -> list[int]:
    """Deterministic unique sampling seeds for each panel."""
    return list(range(seed_start, seed_start + num_panels))


def _generate_execution_orders(
    bundle_ids: list[str],
    num_panels: int,
    schedule_seed: int,
) -> list[list[str]]:
    """Balanced deterministic execution orders across panels.

    For 4 policies with exactly 24 panels (or exact multiples), uses
    every permutation once per 24-panel block (shuffled with
    schedule_seed).  For all other cases, uses cyclic rotations of a
    shuffled base order for guaranteed balance.
    """
    rng = random.Random(schedule_seed)
    p = len(bundle_ids)
    if p == 4 and num_panels % 24 == 0 and num_panels > 0:
        all_perms = list(itertools.permutations(bundle_ids))
        rng.shuffle(all_perms)
        blocks = num_panels // 24
        if blocks == 1:
            return [list(perm) for perm in all_perms]
        orders: list[list[str]] = []
        for block in range(blocks):
            block_perms = list(all_perms)
            rng = random.Random(schedule_seed + block + 1)
            rng.shuffle(block_perms)
            orders.extend(list(perm) for perm in block_perms[:24])
        return orders

    # General case: cyclic rotations for balanced position occupancy.
    base_order = list(bundle_ids)
    rng.shuffle(base_order)
    orders = []
    for i in range(num_panels):
        rotation = i % p
        order = base_order[rotation:] + base_order[:rotation]
        orders.append(order)
    return orders


def _validate_spec_body(spec: Mapping[str, Any]) -> None:
    """Validate the input spec before manifest construction."""
    _require_nonempty_safe(str(spec.get("screen_id", "")), "screen_id")
    _require_nonempty_safe(str(spec.get("purpose", "")), "purpose")

    remote = spec.get("remote_identity")
    if not isinstance(remote, Mapping):
        raise ValueError("remote_identity must be an object")
    _require_nonempty_safe(str(remote.get("host", "")), "remote_identity.host")
    _require_nonempty_safe(str(remote.get("python", "python3")), "remote_identity.python")
    for key in ("project", "run_root"):
        value = str(remote.get(key, ""))
        if not value or not value.startswith("/") or value == "/":
            raise ValueError(
                f"remote_identity.{key} must be an explicit absolute path"
            )

    bundle_ids = spec.get("policy_bundle_ids")
    if not isinstance(bundle_ids, list) or len(bundle_ids) < 2:
        raise ValueError("policy_bundle_ids must be a list with >= 2 entries")
    if len(set(bundle_ids)) != len(bundle_ids):
        raise ValueError("policy_bundle_ids must be unique")

    # Validate optional task_role (defaults to T_pilot).
    task_role = spec.get("task_role", "T_pilot")
    if task_role not in ("T_pilot", "T_canary"):
        raise ValueError(f"task_role must be T_pilot or T_canary, got {task_role!r}")

    # Validate optional protocol object.
    protocol = spec.get("protocol")
    if protocol is not None and not isinstance(protocol, dict):
        raise ValueError("protocol must be an object if present")

    tasks = spec.get("tasks")
    if not isinstance(tasks, list) or len(tasks) < 1:
        raise ValueError("tasks must be a non-empty list")
    task_coords: set[tuple[str, str, int]] = set()
    for task in tasks:
        if not isinstance(task, Mapping):
            raise ValueError("each task must be an object")
        template = str(task.get("template", ""))
        if template not in KNOWN_TEMPLATES:
            raise ValueError(f"unknown or held template: {template!r}")
        difficulty = str(task.get("difficulty", ""))
        if difficulty not in ("easy", "medium", "hard"):
            raise ValueError(f"invalid difficulty: {difficulty!r}")
        seed = _require_positive_int(task.get("seed"), "task.seed")
        coord = (template, difficulty, seed)
        if coord in task_coords:
            raise ValueError(f"duplicate task coordinate: {coord}")
        task_coords.add(coord)

    _require_positive_int(spec.get("rollout_replicas", 1), "rollout_replicas")
    _validate_schedule_seed(spec.get("schedule_seed", 0))
    _validate_schedule_seed(spec.get("sampling_seed_start", 0))


def _build_task_entry(
    task: Mapping[str, Any],
    *,
    screen_task_role: str = "T_pilot",
) -> dict[str, Any]:
    template = str(task["template"])
    difficulty = str(task["difficulty"])
    seed = int(task["seed"])
    task_id = f"unbrowser-fixture-v2-{template}-{difficulty}-{seed}"
    entry: dict[str, Any] = {
        "task_id": task_id,
        "role": screen_task_role,
        "template": template,
        "difficulty": difficulty,
        "seed": seed,
    }
    if template == "distractor_recovery":
        entry["recovery_probe_url"] = (
            f"http://127.0.0.1:{FIXTURE_PORT}/{template}/{seed}/"
            f"{difficulty}/page_0"
        )
        entry["recovery_probe_status"] = 503
    return entry


def _build_panels(
    tasks: list[dict[str, Any]],
    bundle_ids: list[str],
    rollout_replicas: int,
    sampling_seed_start: int,
    schedule_seed: int,
) -> list[dict[str, Any]]:
    """Generate one panel per task x replica with interleaved schedule.

    Task x replica coordinates are shuffled deterministically from
    schedule_seed so replicas are not always adjacent.  Execution orders
    and sampling seeds are assigned in the resulting panel sequence.
    """
    num_panels = len(tasks) * rollout_replicas
    execution_orders = _generate_execution_orders(
        bundle_ids, num_panels, schedule_seed
    )
    sampling_seeds = _generate_panel_sampling_seeds(num_panels, sampling_seed_start)

    # Build all (task_idx, replica) coordinates and shuffle.
    coords = [
        (task_idx, replica)
        for task_idx in range(len(tasks))
        for replica in range(rollout_replicas)
    ]
    rng = random.Random(schedule_seed)
    rng.shuffle(coords)

    panels: list[dict[str, Any]] = []
    for panel_index, (task_idx, replica) in enumerate(coords):
        task = tasks[task_idx]
        panels.append(
            {
                "panel_id": f"{task['task_id']}/replica={replica}",
                "task_id": task["task_id"],
                "rollout_replica": replica,
                "sampling_seed": sampling_seeds[panel_index],
                "execution_order": execution_orders[panel_index],
            }
        )
    return panels


# ---------------------------------------------------------------------------
# build / freeze
# ---------------------------------------------------------------------------


def build_screen_manifest(
    registry: TreatmentRegistry,
    policy_split: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    registry_file: str,
    policy_split_file: str,
) -> dict[str, Any]:
    """Build a self-hashed exploratory screen manifest from a spec."""
    from .m3_pilot import _verify_embedded_hash

    _verify_embedded_hash(policy_split, "manifest_hash")
    if policy_split.get("registry_hash") != registry.registry_hash:
        raise ValueError("policy split registry hash does not match registry")

    _validate_spec_body(spec)

    meta_train = set(policy_split.get("splits", {}).get("meta_train", []))
    bundle_ids = [str(bid) for bid in spec["policy_bundle_ids"]]
    selected: dict[str, TreatmentSpec] = {}
    for bid in bundle_ids:
        if bid not in meta_train:
            raise ValueError(f"policy {bid!r} is not in meta_train")
        treatment = registry.by_bundle_id(bid)
        selected[bid] = treatment

    dev = set(policy_split.get("splits", {}).get("development", []))
    final_held = set(policy_split.get("splits", {}).get("final_held_out", []))
    for bid in bundle_ids:
        if bid in dev:
            raise ValueError(f"policy {bid!r} is in development (must be meta_train only)")
        if bid in final_held:
            raise ValueError(f"policy {bid!r} is in final_held_out (must be meta_train only)")

    screen_task_role = str(spec.get("task_role", "T_pilot"))
    tasks = [_build_task_entry(task, screen_task_role=screen_task_role) for task in spec["tasks"]]
    rollout_replicas = int(spec.get("rollout_replicas", 1))
    schedule_seed = int(spec.get("schedule_seed", 0))
    sampling_seed_start = int(spec.get("sampling_seed_start", 0))

    panels = _build_panels(
        tasks, bundle_ids, rollout_replicas, sampling_seed_start, schedule_seed
    )

    # Build runtime pins adjusted for this screen.
    screen_pins = dict(_RUNTIME_PINS)
    screen_pins["rollout_replicas"] = rollout_replicas

    # Policy factors (only the grammar dimensions, not metadata extras).
    factor_names = _FACTOR_NAMES
    policy_meta = {
        bid: {
            f: str(treatment.generator_metadata.get(f, ""))
            for f in factor_names
        }
        for bid, treatment in selected.items()
    }

    # Hamming-1 pairs.
    hamming_1_pairs: list[dict[str, Any]] = []
    for bid_a, bid_b in itertools.combinations(bundle_ids, 2):
        meta_a = policy_meta[bid_a]
        meta_b = policy_meta[bid_b]
        distance = sum(
            str(meta_a.get(f)) != str(meta_b.get(f)) for f in factor_names
        )
        if distance == 1:
            changed = [
                f for f in factor_names
                if str(meta_a.get(f)) != str(meta_b.get(f))
            ]
            hamming_1_pairs.append({
                "policy_a": bid_a,
                "policy_b": bid_b,
                "factor_changed": changed[0],
                "level_a": str(meta_a.get(changed[0])),
                "level_b": str(meta_b.get(changed[0])),
            })

    selection = spec.get("selection", {})
    if not isinstance(selection, Mapping):
        selection = {}

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "screen_id": str(spec["screen_id"]),
        "purpose": str(spec["purpose"]),
        "registry_file": registry_file,
        "registry_hash": registry.registry_hash,
        "policy_split_file": policy_split_file,
        "policy_split_manifest_hash": policy_split["manifest_hash"],
        "grammar_version": _GRAMMAR_VERSION,
        "policy_bundle_ids": bundle_ids,
        "policy_factors": {
            bid: dict(policy_meta[bid]) for bid in bundle_ids
        },
        "known_templates": list(KNOWN_TEMPLATES),
        "held_templates": list(HELD_TEMPLATES),
        "tasks": tasks,
        "panels": panels,
        "rollout_replicas": rollout_replicas,
        "sampling_seed_start": sampling_seed_start,
        "schedule_seed": schedule_seed,
        "runtime_pins": screen_pins,
        "remote_identity": {
            "host": str(spec["remote_identity"]["host"]),
            "project": str(spec["remote_identity"]["project"]),
            "run_root": str(spec["remote_identity"]["run_root"]),
            "python": str(spec["remote_identity"].get("python", "python3")),
        },
        "selection": {
            "eligible_split": "meta_train",
            "factor_order": list(factor_names),
            "hamming_1_pairs": hamming_1_pairs,
            "spec_provenance": selection,
        },
        "gates": {
            "policies": len(bundle_ids),
            "tasks": len(tasks),
            "panels": len(panels),
            "attempts": len(panels) * len(bundle_ids),
            "rollout_replicas": rollout_replicas,
        },
        "exclusion": (
            f"{screen_task_role} tasks and all attempts are permanently excluded from "
            "meta-training, calibration, development, and final evaluation pools."
        ),
    }
    # Optional protocol object hash-bound into the manifest.
    protocol = spec.get("protocol")
    if protocol is not None and isinstance(protocol, dict):
        payload["protocol"] = dict(protocol)
    if screen_task_role != "T_pilot":
        payload["task_role"] = screen_task_role
    return {**payload, "manifest_hash": _canonical_hash(payload)}


def validate_screen_manifest(
    manifest: Mapping[str, Any],
    registry: TreatmentRegistry,
    policy_split: Mapping[str, Any],
) -> None:
    """Fail closed on any structural or integrity drift."""
    from .m3_pilot import _verify_embedded_hash

    _verify_embedded_hash(manifest, "manifest_hash")
    _verify_embedded_hash(policy_split, "manifest_hash")

    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported screen manifest schema")

    if manifest.get("registry_hash") != registry.registry_hash:
        raise ValueError("screen manifest registry hash mismatch")

    if manifest.get("policy_split_manifest_hash") != policy_split.get("manifest_hash"):
        raise ValueError("screen manifest policy split hash mismatch")

    if manifest.get("grammar_version") != _GRAMMAR_VERSION:
        raise ValueError("grammar version mismatch")

    _require_nonempty_safe(str(manifest.get("screen_id", "")), "screen_id")
    _require_nonempty_safe(str(manifest.get("purpose", "")), "purpose")

    remote = manifest.get("remote_identity")
    if not isinstance(remote, Mapping):
        raise ValueError("remote_identity must be an object")
    _require_nonempty_safe(str(remote.get("host", "")), "remote_identity.host")
    _require_nonempty_safe(str(remote.get("python", "python3")), "remote_identity.python")
    for key in ("project", "run_root"):
        value = str(remote.get(key, ""))
        if not value or not value.startswith("/") or value == "/":
            raise ValueError(f"remote_identity.{key} must be an absolute path")

    bundle_ids = manifest.get("policy_bundle_ids")
    if not isinstance(bundle_ids, list) or len(bundle_ids) < 2:
        raise ValueError("policy_bundle_ids must be a list with >= 2 entries")
    if len(set(bundle_ids)) != len(bundle_ids):
        raise ValueError("policy_bundle_ids must be unique")

    screen_task_role = manifest.get("task_role", "T_pilot")
    if screen_task_role not in ("T_pilot", "T_canary"):
        raise ValueError("task_role must be T_pilot or T_canary")
    protocol = manifest.get("protocol")
    if protocol is not None and not isinstance(protocol, Mapping):
        raise ValueError("protocol must be an object if present")

    meta_train = set(policy_split.get("splits", {}).get("meta_train", []))
    dev = set(policy_split.get("splits", {}).get("development", []))
    final_held = set(policy_split.get("splits", {}).get("final_held_out", []))
    for bid in bundle_ids:
        bid_str = str(bid)
        if bid_str not in meta_train:
            raise ValueError(f"policy {bid_str!r} is not in meta_train")
        if bid_str in dev:
            raise ValueError(f"policy {bid_str!r} is in development (must be meta_train only)")
        if bid_str in final_held:
            raise ValueError(f"policy {bid_str!r} is in final_held_out (must be meta_train only)")

    # Verify policy factors match registry.
    factors = manifest.get("policy_factors")
    if not isinstance(factors, Mapping):
        raise ValueError("policy_factors must be an object")
    for bid in bundle_ids:
        if str(bid) not in factors:
            raise ValueError(f"policy_factors missing entry for {bid!r}")
        treatment = registry.by_bundle_id(str(bid))
        expected_factors = {
            f: str(treatment.generator_metadata.get(f, ""))
            for f in _FACTOR_NAMES
        }
        if dict(factors[str(bid)]) != expected_factors:
            raise ValueError(f"policy_factors mismatch for {bid}")

    # Verify known/held templates exactly.
    if manifest.get("known_templates") != list(KNOWN_TEMPLATES):
        raise ValueError("known_templates mismatch")
    if manifest.get("held_templates") != list(HELD_TEMPLATES):
        raise ValueError("held_templates mismatch")

    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or len(tasks) < 1:
        raise ValueError("tasks must be a non-empty list")
    task_ids: set[str] = set()
    task_coords: set[tuple[str, str, int]] = set()
    for task in tasks:
        if not isinstance(task, Mapping):
            raise ValueError("each task must be an object")
        template = str(task.get("template", ""))
        if template not in KNOWN_TEMPLATES:
            raise ValueError(f"unknown or held task template: {template!r}")
        if template in HELD_TEMPLATES:
            raise ValueError(f"held template {template!r} must not enter a screen")
        difficulty = str(task.get("difficulty", ""))
        if difficulty not in ("easy", "medium", "hard"):
            raise ValueError(f"invalid task difficulty: {difficulty!r}")
        seed = _require_positive_int(task.get("seed"), "task.seed")
        coord = (template, difficulty, seed)
        if coord in task_coords:
            raise ValueError(f"duplicate task coordinate: {coord}")
        task_coords.add(coord)
        expected_id = f"unbrowser-fixture-v2-{template}-{difficulty}-{seed}"
        if task.get("task_id") != expected_id:
            raise ValueError(f"task_id mismatch: expected {expected_id!r}")
        if task.get("role") != screen_task_role:
            raise ValueError(
                f"task role mismatch: expected {screen_task_role!r}"
            )
        if template == "distractor_recovery":
            expected_probe = (
                f"http://127.0.0.1:{FIXTURE_PORT}/distractor_recovery/"
                f"{seed}/{difficulty}/page_0"
            )
            if task.get("recovery_probe_url") != expected_probe:
                raise ValueError("recovery_probe_url mismatch for distractor_recovery")
            if task.get("recovery_probe_status") != 503:
                raise ValueError("recovery_probe_status must be 503 for distractor_recovery")
        else:
            if task.get("recovery_probe_url") is not None:
                raise ValueError(
                    "non-distractor_recovery task must not have recovery_probe_url"
                )
        task_ids.add(expected_id)

    rollout_replicas = _require_positive_int(
        manifest.get("rollout_replicas"), "rollout_replicas"
    )
    schedule_seed = _validate_schedule_seed(manifest.get("schedule_seed"))
    sampling_seed_start = _validate_schedule_seed(manifest.get("sampling_seed_start"))

    # Rebuild panels to verify exact schedule/order match.
    rebuillt_panels = _build_panels(
        tasks, bundle_ids, rollout_replicas, sampling_seed_start, schedule_seed
    )
    manifest_panels = manifest.get("panels")
    if not isinstance(manifest_panels, list) or len(manifest_panels) != len(rebuillt_panels):
        raise ValueError("panel count mismatch on rebuild")
    for i, (mp, rp) in enumerate(zip(manifest_panels, rebuillt_panels)):
        if mp != rp:
            raise ValueError(f"panel {i} mismatch on rebuild: manifest={mp}, rebuilt={rp}")

    # Verify execution order exact lengths.
    for panel in manifest_panels:
        exec_order = panel.get("execution_order")
        if not isinstance(exec_order, list) or len(exec_order) != len(bundle_ids):
            raise ValueError(
                f"panel {panel.get('panel_id')} execution order must have "
                f"exactly {len(bundle_ids)} entries"
            )
        if set(exec_order) != set(bundle_ids):
            raise ValueError(
                f"panel {panel.get('panel_id')} execution order must contain "
                f"each policy exactly once"
            )

    # Verify sampling seeds are unique.
    panel_seeds = [panel.get("sampling_seed") for panel in manifest_panels]
    if len(set(panel_seeds)) != len(panel_seeds):
        raise ValueError("panel sampling seeds must be unique")

    # Verify execution orders are balanced.
    positions: list[dict[str, int]] = [{} for _ in range(len(bundle_ids))]
    for panel in manifest_panels:
        for pos, bid in enumerate(panel["execution_order"]):
            positions[pos][bid] = positions[pos].get(bid, 0) + 1
    for pos in range(len(bundle_ids)):
        counts = list(positions[pos].values())
        if not counts or max(counts) - min(counts) > 1:
            raise ValueError(
                f"execution orders not balanced at position {pos}: {positions[pos]}"
            )

    # Verify expected counts.
    gates = manifest.get("gates", {})
    expected_attempts = len(manifest_panels) * len(bundle_ids)
    if gates.get("policies") != len(bundle_ids):
        raise ValueError("gates.policies mismatch")
    if gates.get("tasks") != len(tasks):
        raise ValueError("gates.tasks mismatch")
    if gates.get("panels") != len(manifest_panels):
        raise ValueError("gates.panels mismatch")
    if gates.get("attempts") != expected_attempts:
        raise ValueError("gates.attempts mismatch")
    if gates.get("rollout_replicas") != rollout_replicas:
        raise ValueError("gates.rollout_replicas mismatch")

    # Verify runtime pins: exact copy of _RUNTIME_PINS with only rollout_replicas adjusted.
    expected_pins = dict(_RUNTIME_PINS)
    expected_pins["rollout_replicas"] = rollout_replicas
    if manifest.get("runtime_pins") != expected_pins:
        raise ValueError("runtime_pins mismatch")

    # Verify exclusion text.
    expected_exclusion = (
        f"{screen_task_role} tasks and all attempts are permanently excluded from "
        "meta-training, calibration, development, and final evaluation pools."
    )
    if manifest.get("exclusion") != expected_exclusion:
        raise ValueError("manifest exclusion warning mismatch")

    # Recompute manifest hash as final gate.
    _verify_embedded_hash(manifest, "manifest_hash")


def freeze_screen_manifest(
    output_path: str | Path,
    registry_path: str | Path,
    policy_split_path: str | Path,
    spec_path: str | Path,
) -> dict[str, Any]:
    """Read inputs, build manifest, validate, and immutably write."""
    registry_file = Path(registry_path).expanduser().resolve()
    split_file = Path(policy_split_path).expanduser().resolve()
    spec_file = Path(spec_path).expanduser().resolve()

    registry = TreatmentRegistry.load(registry_file)
    policy_split = _load_json(split_file)
    spec = _load_json(spec_file)

    manifest = build_screen_manifest(
        registry,
        policy_split,
        spec,
        registry_file=registry_file.name,
        policy_split_file=split_file.name,
    )
    validate_screen_manifest(manifest, registry, policy_split)

    output = Path(output_path).expanduser().resolve()
    _write_immutable_json(output, manifest)

    return {
        "manifest": str(output),
        "manifest_hash": manifest["manifest_hash"],
        "screen_id": manifest["screen_id"],
        "policies": len(manifest["policy_bundle_ids"]),
        "tasks": len(manifest["tasks"]),
        "panels": len(manifest["panels"]),
        "attempts": manifest["gates"]["attempts"],
    }


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

# ---- shared strict result validation ---------------------------------------


def _validate_panel_result_strict(
    record: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_panel: dict[str, Any] | None,
    manifest_task: dict[str, Any] | None,
) -> list[str]:
    """Validate one panel result record against the manifest.

    Returns a list of error strings (empty = valid). Used by both
    ``_existing_result_keys`` and ``_load_result_jsonl`` to share the
    same strict checks.
    """
    errors: list[str] = []
    pid = str(record.get("panel_id", ""))

    if record.get("schema_version") != PANEL_RESULT_SCHEMA:
        errors.append("unknown result schema version")
        return errors

    if record.get("manifest_hash") != manifest["manifest_hash"]:
        errors.append("manifest hash mismatch")
        return errors

    status = record.get("status")
    if status not in ("completed", "error"):
        errors.append(f"unknown status: {status!r}")
        return errors

    if manifest_panel is None:
        errors.append(f"unknown panel_id {pid!r}")
        return errors

    # Check panel / task metadata in record matches manifest.
    rec_panel = record.get("panel")
    if not isinstance(rec_panel, Mapping):
        errors.append("record.panel missing or not an object")
    else:
        for key in ("panel_id", "task_id", "rollout_replica",
                     "sampling_seed", "execution_order"):
            if rec_panel.get(key) != manifest_panel.get(key):
                errors.append(f"record.panel.{key} mismatch")

    rec_task = record.get("task")
    if not isinstance(rec_task, Mapping):
        errors.append("record.task missing or not an object")
    elif manifest_task is not None:
        for key in ("task_id", "template", "difficulty", "seed", "role"):
            if rec_task.get(key) != manifest_task.get(key):
                errors.append(f"record.task.{key} mismatch")

    if status == "error":
        err_info = record.get("error")
        if not isinstance(err_info, Mapping) or "type" not in err_info:
            errors.append("error record missing error.type")
        return errors

    # status == "completed" — deep result validation.
    result = record.get("result")
    if not isinstance(result, Mapping):
        errors.append("completed record missing result object")
        return errors

    if result.get("mode") != "treatment_set":
        errors.append("result.mode must be treatment_set")

    bundle_ids = manifest["policy_bundle_ids"]
    expected_bid_set = set(bundle_ids)

    # Check result-level metadata.
    for field in ("pilot_manifest_hash", "pilot_panel_id",
                   "rollout_replica", "sampling_seed"):
        result_val = result.get(field)
        if field == "pilot_manifest_hash":
            if result_val != manifest["manifest_hash"]:
                errors.append(f"result.{field} mismatch")
        elif field == "pilot_panel_id":
            if result_val != pid:
                errors.append(f"result.{field} mismatch")
        elif field == "rollout_replica":
            if result_val != manifest_panel.get("rollout_replica"):
                errors.append(f"result.{field} mismatch")
        elif field == "sampling_seed":
            if result_val != manifest_panel.get("sampling_seed"):
                errors.append(f"result.{field} mismatch")

    if result.get("execution_order") != manifest_panel.get("execution_order"):
        errors.append("result.execution_order mismatch")

    attempts = result.get("attempts")
    if not isinstance(attempts, Mapping):
        errors.append("result.attempts missing or not an object")
        return errors

    found_bids = set(str(k) for k in attempts)
    if found_bids != expected_bid_set:
        extra = found_bids - expected_bid_set
        missing = expected_bid_set - found_bids
        if extra:
            errors.append(f"result.attempts has extra policies: {sorted(extra)}")
        if missing:
            errors.append(f"result.attempts missing policies: {sorted(missing)}")

    for bid, attempt in attempts.items():
        if not isinstance(attempt, Mapping):
            errors.append(f"attempt for {bid} is not an object")
            continue

        # --- verification.success must be bool ---------------------------------
        verif = attempt.get("verification")
        if not isinstance(verif, Mapping):
            errors.append(f"attempt {bid} missing verification object")
        elif not isinstance(verif.get("success"), bool):
            errors.append(f"attempt {bid} verification.success must be bool")

        # --- pi_return_code must be int, not bool ------------------------------
        pi_rc = attempt.get("pi_return_code")
        if isinstance(pi_rc, bool) or not isinstance(pi_rc, int):
            errors.append(
                f"attempt {bid} pi_return_code must be int, got "
                f"{type(pi_rc).__name__!r}"
            )

        # --- usage.output must be present, numeric-not-bool, finite, >= 0 -------
        usage = attempt.get("usage")
        if not isinstance(usage, Mapping):
            errors.append(f"attempt {bid} missing usage object")
        else:
            output_tokens = usage.get("output")
            if output_tokens is None:
                errors.append(f"attempt {bid} usage.output is required but missing")
            elif isinstance(output_tokens, bool):
                errors.append(
                    f"attempt {bid} usage.output must be numeric, got bool"
                )
            elif not isinstance(output_tokens, (int, float)):
                errors.append(
                    f"attempt {bid} usage.output must be numeric, got "
                    f"{type(output_tokens).__name__!r}"
                )
            elif not math.isfinite(float(output_tokens)) or float(output_tokens) < 0:
                errors.append(
                    f"attempt {bid} usage.output must be finite and nonnegative"
                )

        # --- trajectory must be present and structurally sound -----------------
        traj = attempt.get("trajectory")
        if not isinstance(traj, Mapping):
            errors.append(
                f"attempt {bid} trajectory must be an object (required for adherence)"
            )
        else:
            # planning_preamble must be a Mapping.
            pp = traj.get("planning_preamble")
            if not isinstance(pp, Mapping):
                errors.append(
                    f"attempt {bid} trajectory.planning_preamble "
                    f"must be an object"
                )
            # tool_trace must be a list.
            tt = traj.get("tool_trace")
            if not isinstance(tt, list):
                errors.append(
                    f"attempt {bid} trajectory.tool_trace must be a list"
                )
            else:
                for ti, entry in enumerate(tt):
                    if not isinstance(entry, Mapping):
                        errors.append(
                            f"attempt {bid} trajectory.tool_trace[{ti}] "
                            f"must be an object"
                        )
                        continue
                    tn = entry.get("tool_name")
                    if not isinstance(tn, str):
                        errors.append(
                            f"attempt {bid} trajectory.tool_trace[{ti}] "
                            f"tool_name must be str"
                        )
                    for bool_key in (
                        "is_error",
                        "budget_rejected",
                        "operation_aborted",
                    ):
                        val = entry.get(bool_key)
                        if not isinstance(val, bool):
                            errors.append(
                                f"attempt {bid} trajectory.tool_trace[{ti}] "
                                f"{bool_key} must be bool"
                            )
                    if (
                        "pre_execution_rejected" in entry
                        and not isinstance(entry.get("pre_execution_rejected"), bool)
                    ):
                        errors.append(
                            f"attempt {bid} trajectory.tool_trace[{ti}] "
                            "pre_execution_rejected must be bool when present"
                        )
                    det = entry.get("details")
                    if not isinstance(det, Mapping):
                        errors.append(
                            f"attempt {bid} trajectory.tool_trace[{ti}] "
                            f"details must be an object"
                        )
            # provider_turn_count must be nonnegative int (not bool).
            ptc = traj.get("provider_turn_count")
            if isinstance(ptc, bool) or not isinstance(ptc, int) or ptc < 0:
                errors.append(
                    f"attempt {bid} trajectory.provider_turn_count "
                    f"must be a nonnegative int"
                )

    return errors


# ---- resume helpers --------------------------------------------------------


def _existing_result_keys(
    path: Path,
    manifest: Mapping[str, Any],
) -> set[str]:
    """Return completed panel keys that pass strict validation.

    Missing-schema records are rejected (not skipped). Duplicate panel
    IDs, unsafe records, and structural mismatches raise ValueError.
    Infrastructure errors that pass validation are reported but raise
    RuntimeError to require adjudication.
    """
    if not path.exists():
        return set()
    manifest_hash = manifest["manifest_hash"]
    panel_by_id = {str(p["panel_id"]): p for p in manifest["panels"]}
    task_by_id = {str(t["task_id"]): t for t in manifest["tasks"]}
    completed: set[str] = set()
    seen_panels: set[str] = set()

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid screen JSONL line {line_number}: {error}"
            ) from error
        if not isinstance(record, Mapping):
            raise ValueError(f"line {line_number}: record must be an object")

        # Schema must be present and exact — never skip.
        sch = record.get("schema_version")
        if sch is None:
            raise ValueError(
                f"line {line_number}: record missing schema_version"
            )
        if sch != PANEL_RESULT_SCHEMA:
            raise ValueError(
                f"line {line_number}: unknown schema {sch!r}"
            )

        pid = str(record.get("panel_id", ""))
        if pid in seen_panels:
            raise ValueError(
                f"line {line_number}: duplicate panel {pid}"
            )
        seen_panels.add(pid)

        if record.get("manifest_hash") != manifest_hash:
            raise ValueError(
                f"line {line_number}: manifest hash mismatch"
            )

        mpanel = panel_by_id.get(pid)
        mpanel_task = task_by_id.get(mpanel["task_id"]) if mpanel else None

        errors = _validate_panel_result_strict(
            record, manifest, mpanel, mpanel_task
        )
        if errors:
            raise ValueError(
                f"line {line_number}: invalid panel result: {'; '.join(errors)}"
            )

        if record.get("status") == "error":
            raise RuntimeError(
                "existing screen infrastructure error requires adjudication"
            )
        if record.get("status") == "completed":
            completed.add(pid)

    return completed


# ---- run -------------------------------------------------------------------


def run_screen(
    manifest_path: str | Path,
    registry_path: str | Path,
    policy_split_path: str | Path,
    output_path: str | Path,
    config: RemoteConfig,
    *,
    pi_binary: str,
    provider: str,
    model: str,
    thinking: str,
    unbrowser_binary: str,
    model_artifact: str,
    llama_server_binary: str,
) -> dict[str, Any]:
    """Run every panel in the screen sequentially, stop on infrastructure error."""
    project_root = Path(__file__).resolve().parents[2]
    registry = TreatmentRegistry.load(registry_path)
    policy_split = _load_json(policy_split_path)
    manifest = _load_json(manifest_path)
    validate_screen_manifest(manifest, registry, policy_split)

    # Pin provider/model/thinking.
    runtime_pins = manifest["runtime_pins"]
    if provider != runtime_pins["provider"] or model != runtime_pins["model_alias"]:
        raise ValueError("provider/model do not match the frozen runtime pins")
    if thinking != runtime_pins["thinking"]:
        raise ValueError("thinking mode does not match the frozen runtime pins")

    # Verify RemoteConfig matches frozen remote_identity exactly.
    remote = manifest["remote_identity"]
    if not str(config.host).strip():
        raise ValueError("host must not be empty")
    if config.host != remote["host"]:
        raise ValueError(f"host mismatch: {config.host!r} vs {remote['host']!r}")
    if config.project != remote["project"]:
        raise ValueError(f"project mismatch: {config.project!r} vs {remote['project']!r}")
    if config.run_root != remote["run_root"]:
        raise ValueError(f"run_root mismatch: {config.run_root!r} vs {remote['run_root']!r}")
    if config.python != remote["python"]:
        raise ValueError(f"python mismatch: {config.python!r} vs {remote['python']!r}")

    output = Path(output_path).expanduser().resolve()
    active_path = output.with_suffix(output.suffix + ".active.json")
    if active_path.exists():
        raise RuntimeError(
            "unfinished screen panel marker requires a fresh output/run root"
        )

    runtime = runtime_preflight(
        project_root,
        config,
        pi_binary=pi_binary,
        thinking=thinking,
        unbrowser_binary=unbrowser_binary,
        model_artifact=model_artifact,
        llama_server_binary=llama_server_binary,
        require_clean=False,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    preflight_path = output.with_suffix(output.suffix + ".preflight.json")
    preflight_payload = {
        "manifest_hash": manifest["manifest_hash"],
        "screen_preflight": True,
        **runtime,
        "runtime_pins": manifest["runtime_pins"],
        "remote_identity": {
            "host": config.host,
            "project": config.project,
            "run_root": config.run_root,
            "python": config.python,
        },
    }
    if preflight_path.exists():
        existing_preflight = _load_json(preflight_path)
        stable_keys = (
            "manifest_hash",
            "code_revision",
            "source_tree_hash",
            "worktree_status_hash",
            "worktree_clean",
            "runtime_pins",
            "remote_identity",
        )
        if any(
            existing_preflight.get(key) != preflight_payload.get(key)
            for key in stable_keys
        ):
            raise RuntimeError("screen preflight identity changed across resume")
    else:
        _write_immutable_json(preflight_path, preflight_payload)

    completed = _existing_result_keys(output, manifest)
    bundle_ids = manifest["policy_bundle_ids"]
    task_by_id = {task["task_id"]: task for task in manifest["tasks"]}
    panel_by_id = {str(p["panel_id"]): p for p in manifest["panels"]}
    ran = skipped = 0

    screen_task_role = str(manifest.get("task_role", "T_pilot"))
    for panel in manifest["panels"]:
        task = task_by_id[panel["task_id"]]
        panel_id = str(panel["panel_id"])
        if panel_id in completed:
            skipped += 1
            continue

        args = argparse.Namespace(
            family="unbrowser_fixture",
            seed=int(task["seed"]),
            difficulty=str(task["difficulty"]),
            fixture_template=str(task["template"]),
            task_role=screen_task_role,
            rollout_replica=int(panel["rollout_replica"]),
            sampling_seed=int(panel["sampling_seed"]),
            pilot_manifest_hash=str(manifest["manifest_hash"]),
            pilot_panel_id=panel_id,
            treatment_registry=str(Path(registry_path).expanduser().resolve()),
            treatments=",".join(panel["execution_order"]),
            preserve_treatment_order=True,
            pi=pi_binary,
            provider=provider,
            model=model,
            thinking=thinking,
            model_switch_extension=None,
            unbrowser_binary=unbrowser_binary,
        )
        started = time.monotonic()
        record: dict[str, Any] = {
            "schema_version": PANEL_RESULT_SCHEMA,
            "panel_id": panel_id,
            "manifest_hash": manifest["manifest_hash"],
            "task": task,
            "panel": panel,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_immutable_json(
            active_path,
            {
                "manifest_hash": manifest["manifest_hash"],
                "panel_id": panel_id,
                "started_at": record["started_at"],
            },
        )

        # --- run and validate ------------------------------------------------
        try:
            result = run_registered_treatments(project_root, config, args)
        except Exception as error:
            record.update(
                {
                    "status": "error",
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "duration_seconds": round(time.monotonic() - started, 3),
                    "error": {"type": type(error).__name__,
                               "message": str(error)},
                }
            )
            from .batch import _append_result

            _append_result(output, record)
            active_path.unlink()
            raise RuntimeError(
                f"screen stopped after infrastructure error on {panel_id}"
            ) from error

        record.update(
            {
                "status": "completed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": round(time.monotonic() - started, 3),
                "result": result,
            }
        )

        # strict validation of the completed record before durable append
        mpanel = panel_by_id.get(panel_id)
        mpanel_task = task_by_id.get(mpanel["task_id"]) if mpanel else None
        validation_errors = _validate_panel_result_strict(
            record, manifest, mpanel, mpanel_task
        )
        if validation_errors:
            # malformed orchestrator result — record as infra error and stop
            record["status"] = "error"
            record.pop("result", None)
            record["finished_at"] = datetime.now(timezone.utc).isoformat()
            record["duration_seconds"] = round(time.monotonic() - started, 3)
            record["error"] = {
                "type": "MalformedPanelResult",
                "message": "; ".join(validation_errors),
            }
            from .batch import _append_result

            _append_result(output, record)
            active_path.unlink()
            raise RuntimeError(
                f"screen stopped after malformed result on {panel_id}: "
                f"{'; '.join(validation_errors)}"
            )

        from .batch import _append_result

        _append_result(output, record)
        active_path.unlink()
        ran += 1
        print(
            f"  [{ran + skipped}/{len(manifest['panels'])}] {panel_id}",
            file=sys.stderr,
        )

    return {
        "manifest_hash": manifest["manifest_hash"],
        "tasks_total": len(manifest["tasks"]),
        "panels_total": len(manifest["panels"]),
        "policies_total": len(bundle_ids),
        "attempts_total": len(manifest["panels"]) * len(bundle_ids),
        "panels_run": ran,
        "panels_skipped": skipped,
        "output": str(output),
        "preflight": str(preflight_path),
    }


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


def _beta_mean(successes: int, trials: int) -> float:
    """Beta(1, 1) posterior mean = (successes + 1) / (trials + 2)."""
    if trials == 0:
        return 0.5
    return (successes + 1) / (trials + 2)


def _load_result_jsonl(
    path: Path,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Load and strictly validate all panel results against the manifest.

    Unknown schemas are rejected. Duplicates, missing/extra panels, and
    malformed records raise ValueError.
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

        # Schema must be present and exact — never skip.
        sch = record.get("schema_version")
        if sch is None:
            raise ValueError(
                f"line {line_number}: record missing schema_version"
            )
        if sch != PANEL_RESULT_SCHEMA:
            raise ValueError(
                f"line {line_number}: unknown schema {sch!r}"
            )

        pid = str(record.get("panel_id", ""))
        if pid in seen_panels:
            raise ValueError(f"line {line_number}: duplicate panel {pid}")
        seen_panels.add(pid)

        if record.get("manifest_hash") != manifest_hash:
            raise ValueError(
                f"line {line_number}: manifest hash mismatch"
            )

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


def _infer_success(verification: Any) -> bool:
    """Extract success boolean from a verification dict."""
    if isinstance(verification, Mapping):
        return bool(verification.get("success"))
    return False


def analyze_screen(
    manifest_path: str | Path,
    results_path: str | Path,
    registry_path: str | Path,
    policy_split_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Read manifest, registry, and results; compute exploratory analysis."""
    manifest = _load_json(manifest_path)
    manifest_hash = str(manifest["manifest_hash"])
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported manifest schema version")

    registry = TreatmentRegistry.load(registry_path)
    policy_split = _load_json(policy_split_path)
    validate_screen_manifest(manifest, registry, policy_split)

    records = _load_result_jsonl(Path(results_path), manifest)
    bundle_ids = manifest["policy_bundle_ids"]
    tasks = manifest["tasks"]
    policy_factors = manifest.get("policy_factors", {})

    # ---- accumulate ---------------------------------------------------------

    infra_errors = 0
    model_failures = 0
    successes = 0
    per_policy: dict[str, dict[str, Any]] = {}
    for bid in bundle_ids:
        per_policy[bid] = {
            "bundle_id": bid,
            "factors": dict(policy_factors.get(bid, {})),
            "attempts": 0,
            "successes": 0,
            "total_output_tokens": 0.0,
            "output_token_count": 0,
            "admitted_tool_calls": 0,
            "admitted_count": 0,
            "template_results": {},
            "adherence": {
                "planning_adherent": 0,
                "observation_adherent": 0,
                "verification_adherent": 0,
                "recovery_adherent": 0,
                "recovery_eligible": 0,
                "tool_cap_compliant": 0,
                "adherence_checks": 0,
            },
        }

    # Track per-template outcomes globally.
    template_outcomes: dict[str, dict[str, Any]] = {}
    for task in tasks:
        tmpl = str(task["template"])
        if tmpl not in template_outcomes:
            template_outcomes[tmpl] = {
                "attempts": 0,
                "successes": 0,
                "total_output_tokens": 0.0,
                "output_token_count": 0,
            }

    # Track per-task outcomes.
    task_outcomes: dict[str, dict[str, Any]] = {}
    for task in tasks:
        tid = str(task["task_id"])
        task_outcomes[tid] = {
            "task_id": tid,
            "template": str(task["template"]),
            "difficulty": str(task["difficulty"]),
            "attempts": 0,
            "successes": 0,
            "total_output_tokens": 0.0,
            "output_token_count": 0,
        }

    # Paired panel data for Hamming-1 analysis.
    paired_panels: dict[str, dict[str, dict[str, bool | None]]] = {}
    # panel_id -> {bid -> success_bool_or_None}

    for record in records:
        pid = str(record.get("panel_id", ""))
        if record.get("status") == "error":
            infra_errors += 1
            continue
        result = record.get("result")
        if not isinstance(result, Mapping):
            continue
        attempts = result.get("attempts")
        if not isinstance(attempts, Mapping):
            continue

        task = record.get("task", {})
        template = str(task.get("template", ""))
        tid = str(task.get("task_id", ""))

        panel_successes: dict[str, bool] = {}
        for bid, attempt in attempts.items():
            if not isinstance(attempt, Mapping):
                continue
            if bid not in per_policy:
                continue
            pp = per_policy[bid]
            pp["attempts"] += 1

            verif = attempt.get("verification", {})
            ok = _infer_success(verif)
            if ok:
                pp["successes"] += 1

            panel_successes[bid] = ok

            # Usage: orchestrator key is "output", not "output_tokens".
            usage = attempt.get("usage")
            if isinstance(usage, Mapping):
                tokens = usage.get("output")
                if isinstance(tokens, (int, float)):
                    tval = float(tokens)
                    if math.isfinite(tval) and tval >= 0:
                        pp["total_output_tokens"] += tval
                        pp["output_token_count"] += 1

            pi_rc = attempt.get("pi_return_code")
            if isinstance(pi_rc, int) and pi_rc != 0:
                # Model runtime failure, but still counts for other stats.
                pass

            # Track template/task outcomes.
            if template and template in template_outcomes:
                template_outcomes[template]["attempts"] += 1
                if ok:
                    template_outcomes[template]["successes"] += 1
                if isinstance(tokens, (int, float)):
                    tval = float(tokens)
                    if math.isfinite(tval) and tval >= 0:
                        template_outcomes[template]["total_output_tokens"] += tval
                        template_outcomes[template]["output_token_count"] += 1

            if tid and tid in task_outcomes:
                task_outcomes[tid]["attempts"] += 1
                if ok:
                    task_outcomes[tid]["successes"] += 1
                if isinstance(tokens, (int, float)):
                    tval = float(tokens)
                    if math.isfinite(tval) and tval >= 0:
                        task_outcomes[tid]["total_output_tokens"] += tval
                        task_outcomes[tid]["output_token_count"] += 1

            # Adherence (computed once per attempt using registry).
            treatment = registry.by_bundle_id(bid)
            recovery_url = task.get("recovery_probe_url")
            recovery_status = task.get("recovery_probe_status")
            adh = assess_policy_adherence(
                treatment,
                attempt.get("trajectory"),
                required_recovery_probe_url=recovery_url,
                required_recovery_probe_status=recovery_status,
            )
            pp["adherence"]["adherence_checks"] += 1
            if adh.get("planning_adherent"):
                pp["adherence"]["planning_adherent"] += 1
            if adh.get("observation_adherent"):
                pp["adherence"]["observation_adherent"] += 1
            if adh.get("verification_adherent"):
                pp["adherence"]["verification_adherent"] += 1
            if adh.get("recovery_adherent") is not None:
                pp["adherence"]["recovery_eligible"] += 1
                if adh.get("recovery_adherent"):
                    pp["adherence"]["recovery_adherent"] += 1
            if adh.get("tool_cap_compliant"):
                pp["adherence"]["tool_cap_compliant"] += 1

            admitted = adh.get("admitted_tool_call_count", 0)
            pp["admitted_tool_calls"] += admitted
            pp["admitted_count"] += 1  # always count, even zero

        # Record paired successes for this panel.
        if pid not in paired_panels:
            paired_panels[pid] = {}
        for bid, ok_val in panel_successes.items():
            paired_panels[pid][bid] = ok_val

    # ---- compute derived stats ----------------------------------------------

    total_attempts = sum(pp["attempts"] for pp in per_policy.values())
    global_successes = sum(pp["successes"] for pp in per_policy.values())

    for pp in per_policy.values():
        a = pp["attempts"]
        pp["success_rate"] = pp["successes"] / a if a > 0 else 0.0
        oc = pp["output_token_count"]
        pp["mean_output_tokens"] = pp["total_output_tokens"] / oc if oc > 0 else 0.0
        ac = pp["admitted_count"]
        pp["mean_admitted_tool_calls"] = (
            pp["admitted_tool_calls"] / ac if ac > 0 else 0.0
        )
        ach = pp["adherence"]
        n_ach = ach["adherence_checks"]
        pp["planning_adherence_rate"] = (
            ach["planning_adherent"] / n_ach if n_ach else 0.0
        )
        pp["observation_adherence_rate"] = (
            ach["observation_adherent"] / n_ach if n_ach else 0.0
        )
        pp["verification_adherence_rate"] = (
            ach["verification_adherent"] / n_ach if n_ach else 0.0
        )
        pp["recovery_adherence_rate"] = (
            ach["recovery_adherent"] / ach["recovery_eligible"]
            if ach["recovery_eligible"] else None
        )
        pp["tool_cap_compliance_rate"] = (
            ach["tool_cap_compliant"] / n_ach if n_ach else 0.0
        )

    for tmpl, to in template_outcomes.items():
        a = to["attempts"]
        to["success_rate"] = to["successes"] / a if a > 0 else 0.0
        oc = to["output_token_count"]
        to["mean_output_tokens"] = to["total_output_tokens"] / oc if oc > 0 else 0.0

    for tid, tko in task_outcomes.items():
        a = tko["attempts"]
        tko["success_rate"] = tko["successes"] / a if a > 0 else 0.0
        oc = tko["output_token_count"]
        tko["mean_output_tokens"] = tko["total_output_tokens"] / oc if oc > 0 else 0.0

    # ---- marginal factor analysis (including adherence) ---------------------

    marginal: dict[str, dict[str, dict[str, Any]]] = {}
    all_pp = list(per_policy.values())
    for factor in _FACTOR_NAMES:
        levels: dict[str, dict[str, Any]] = {}
        for pp_entry in all_pp:
            level = str(pp_entry["factors"].get(factor, ""))
            if not level:
                continue
            if level not in levels:
                levels[level] = {
                    "attempts": 0,
                    "successes": 0,
                    "total_output_tokens": 0.0,
                    "output_token_count": 0,
                    # factor-specific adherence (only from policies at this level)
                    "planning_adherent": 0,
                    "observation_adherent": 0,
                    "verification_adherent": 0,
                    "recovery_adherent": 0,
                    "recovery_eligible": 0,
                    "tool_cap_compliant": 0,
                    "adherence_checks": 0,
                }
            levels[level]["attempts"] += pp_entry["attempts"]
            levels[level]["successes"] += pp_entry["successes"]
            levels[level]["total_output_tokens"] += pp_entry["total_output_tokens"]
            levels[level]["output_token_count"] += pp_entry["output_token_count"]

            ach = pp_entry["adherence"]
            levels[level]["adherence_checks"] += ach["adherence_checks"]
            levels[level]["planning_adherent"] += ach["planning_adherent"]
            levels[level]["observation_adherent"] += ach["observation_adherent"]
            levels[level]["verification_adherent"] += ach["verification_adherent"]
            levels[level]["recovery_adherent"] += ach["recovery_adherent"]
            levels[level]["recovery_eligible"] += ach["recovery_eligible"]
            levels[level]["tool_cap_compliant"] += ach["tool_cap_compliant"]

        for level, stats in levels.items():
            stats["success_rate"] = (
                stats["successes"] / stats["attempts"]
                if stats["attempts"] else 0.0
            )
            stats["mean_output_tokens"] = (
                stats["total_output_tokens"] / stats["output_token_count"]
                if stats["output_token_count"] else 0.0
            )
            n_ach = stats["adherence_checks"]
            # Assign the appropriate adherence rate per factor.
            if factor == "planning":
                stats["planning_adherence_rate"] = (
                    stats["planning_adherent"] / n_ach if n_ach else 0.0
                )
            elif factor == "observation":
                stats["observation_adherence_rate"] = (
                    stats["observation_adherent"] / n_ach if n_ach else 0.0
                )
            elif factor == "verification":
                stats["verification_adherence_rate"] = (
                    stats["verification_adherent"] / n_ach if n_ach else 0.0
                )
            elif factor == "recovery":
                stats["recovery_adherence_rate"] = (
                    stats["recovery_adherent"] / stats["recovery_eligible"]
                    if stats["recovery_eligible"] else None
                )
                stats["recovery_eligible_count"] = stats["recovery_eligible"]
            elif factor == "tool_cap":
                stats["tool_cap_compliance_rate"] = (
                    stats["tool_cap_compliant"] / n_ach if n_ach else 0.0
                )
        marginal[factor] = levels

    # ---- Hamming-1 pairs (truly paired by common panel) ---------------------

    h1_pairs: list[dict[str, Any]] = []
    for bid_a, bid_b in itertools.combinations(bundle_ids, 2):
        fa = dict(policy_factors.get(bid_a, {}))
        fb = dict(policy_factors.get(bid_b, {}))
        distance = sum(fa.get(f) != fb.get(f) for f in _FACTOR_NAMES)
        if distance != 1:
            continue
        changed = [f for f in _FACTOR_NAMES if fa.get(f) != fb.get(f)][0]

        # Paired accounting across common panels.
        pair_count = 0
        a_wins = 0  # A success, B fail
        b_wins = 0  # B success, A fail
        both_success = 0
        both_fail = 0
        for pid, bids in paired_panels.items():
            ok_a = bids.get(bid_a)
            ok_b = bids.get(bid_b)
            if ok_a is None or ok_b is None:
                continue
            pair_count += 1
            if ok_a and not ok_b:
                a_wins += 1
            elif ok_b and not ok_a:
                b_wins += 1
            elif ok_a and ok_b:
                both_success += 1
            else:
                both_fail += 1

        pp_a = per_policy.get(bid_a, {})
        pp_b = per_policy.get(bid_b, {})
        h1_pairs.append({
            "policy_a": bid_a,
            "policy_b": bid_b,
            "factor_changed": changed,
            "level_a": str(fa.get(changed, "")),
            "level_b": str(fb.get(changed, "")),
            "pair_count": pair_count,
            "a_wins": a_wins,
            "b_wins": b_wins,
            "both_success": both_success,
            "both_fail": both_fail,
            "paired_success_difference": (
                (a_wins - b_wins) / pair_count if pair_count > 0 else 0.0
            ),
            "a_success_rate": pp_a.get("success_rate", 0.0),
            "b_success_rate": pp_b.get("success_rate", 0.0),
            "a_successes": pp_a.get("successes", 0),
            "b_successes": pp_b.get("successes", 0),
            "a_attempts": pp_a.get("attempts", 0),
            "b_attempts": pp_b.get("attempts", 0),
        })

    # ---- two-factor combination cells ---------------------------------------

    two_factor: dict[str, dict[str, Any]] = {}
    for f1, f2 in itertools.combinations(_FACTOR_NAMES, 2):
        pair_key = f"{f1}_x_{f2}"
        cells: dict[str, dict[str, Any]] = {}
        for pp_entry in all_pp:
            l1 = str(pp_entry["factors"].get(f1, ""))
            l2 = str(pp_entry["factors"].get(f2, ""))
            if not l1 or not l2:
                continue
            cell_key = f"{l1}|{l2}"
            if cell_key not in cells:
                cells[cell_key] = {
                    "attempts": 0,
                    "successes": 0,
                    "total_output_tokens": 0.0,
                    "token_count": 0,
                }
            cells[cell_key]["attempts"] += pp_entry["attempts"]
            cells[cell_key]["successes"] += pp_entry["successes"]
            cells[cell_key]["total_output_tokens"] += pp_entry["total_output_tokens"]
            cells[cell_key]["token_count"] += pp_entry["output_token_count"]
        for ck, cs in cells.items():
            cs["success_rate"] = (
                cs["successes"] / cs["attempts"] if cs["attempts"] else 0.0
            )
            cs["mean_output_tokens"] = (
                cs["total_output_tokens"] / cs["token_count"]
                if cs["token_count"] else 0.0
            )
        two_factor[pair_key] = {
            "factor_a": f1,
            "factor_b": f2,
            "cells": cells,
        }

    # ---- candidate ranking --------------------------------------------------

    ranked = sorted(
        all_pp,
        key=lambda pp: (
            -_beta_mean(pp["successes"], pp["attempts"]),
            pp.get("mean_output_tokens", float("inf")),
        ),
    )
    ranking = []
    for rank, pp in enumerate(ranked, 1):
        ranking.append({
            "rank": rank,
            "bundle_id": pp["bundle_id"],
            "factors": pp["factors"],
            "beta_mean": round(_beta_mean(pp["successes"], pp["attempts"]), 6),
            "successes": pp["successes"],
            "attempts": pp["attempts"],
            "success_rate": pp["success_rate"],
            "mean_output_tokens": pp["mean_output_tokens"],
        })

    # ---- assemble analysis --------------------------------------------------

    # Count model runtime failures (pi_return_code != 0).
    model_failure_count = 0
    for record in records:
        if record.get("status") != "completed":
            continue
        result = record.get("result")
        if not isinstance(result, Mapping):
            continue
        attempts = result.get("attempts")
        if not isinstance(attempts, Mapping):
            continue
        for bid, attempt in attempts.items():
            if not isinstance(attempt, Mapping):
                continue
            pi_rc = attempt.get("pi_return_code")
            if isinstance(pi_rc, int) and pi_rc != 0:
                model_failure_count += 1

    analysis: dict[str, Any] = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "screen_id": manifest["screen_id"],
        "manifest_hash": manifest_hash,
        "completeness": {
            "panels_expected": len(manifest["panels"]),
            "panels_found": len(records),
            "complete": len(records) == len(manifest["panels"]),
        },
        "summary": {
            "total_attempts": total_attempts,
            "infra_errors": infra_errors,
            "model_runtime_failures": model_failure_count,
            "successes": global_successes,
            "overall_success_rate": (
                global_successes / total_attempts if total_attempts else 0.0
            ),
            "mean_output_tokens": (
                sum(pp["total_output_tokens"] for pp in all_pp)
                / sum(pp["output_token_count"] for pp in all_pp)
                if any(pp["output_token_count"] for pp in all_pp)
                else 0.0
            ),
        },
        "task_outcomes": task_outcomes,
        "template_outcomes": template_outcomes,
        "per_policy": {
            bid: {
                "bundle_id": pp["bundle_id"],
                "factors": pp["factors"],
                "successes": pp["successes"],
                "attempts": pp["attempts"],
                "success_rate": pp["success_rate"],
                "mean_output_tokens": pp["mean_output_tokens"],
                "mean_admitted_tool_calls": pp["mean_admitted_tool_calls"],
                "planning_adherence_rate": pp["planning_adherence_rate"],
                "observation_adherence_rate": pp["observation_adherence_rate"],
                "verification_adherence_rate": pp["verification_adherence_rate"],
                "recovery_adherence_rate": pp["recovery_adherence_rate"],
                "tool_cap_compliance_rate": pp["tool_cap_compliance_rate"],
            }
            for bid, pp in per_policy.items()
        },
        "marginal": marginal,
        "hamming_1_pairs": h1_pairs,
        "two_factor": two_factor,
        "ranking": ranking,
        "ranking_label": "exploratory-not-causal-confirmatory",
        "warnings": [
            "All data are exploratory and permanently excluded from meta-training "
            "and final evaluation pools.",
            "These results cannot repair the frozen no-go gates.",
            "Candidate rankings use Beta(1,1) posterior mean and are descriptive "
            "only; they are not causal or confirmatory.",
            "Do not use these results for final model selection decisions.",
        ],
    }

    if output_path is not None:
        out = Path(output_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(f"{out}.tmp")
        content = json.dumps(analysis, indent=2, sort_keys=True, ensure_ascii=False)
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, out)

    print(json.dumps(analysis, indent=2, sort_keys=True, ensure_ascii=False))
    return analysis


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyreplab-m3-exploratory-screen")
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--spec", required=True, help="screen spec JSON")
    freeze.add_argument("--registry", required=True)
    freeze.add_argument("--policy-split", required=True)
    freeze.add_argument("--output", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--registry", required=True)
    validate.add_argument("--policy-split", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--manifest", required=True)
    run.add_argument("--registry", required=True)
    run.add_argument("--policy-split", required=True)
    run.add_argument("--output", required=True)
    run.add_argument(
        "--host",
        default=os.environ.get("PYREPLAB_HARNESS_HOST", "ubuntu-local"),
    )
    run.add_argument("--remote-project", required=True)
    run.add_argument("--remote-run-root", required=True)
    run.add_argument("--remote-python", default="python3")
    run.add_argument("--pi", default=os.environ.get("PYREPLAB_PI", "pi"))
    run.add_argument(
        "--provider",
        default=os.environ.get("PYREPLAB_PI_PROVIDER", "ubuntu-gemma"),
    )
    run.add_argument(
        "--model",
        default=os.environ.get("PYREPLAB_PI_MODEL", "gemma-4-26b-a4b"),
    )
    run.add_argument(
        "--thinking",
        default=os.environ.get("PYREPLAB_PI_THINKING", "off"),
    )
    run.add_argument("--unbrowser-binary", required=True)
    run.add_argument("--model-artifact", required=True)
    run.add_argument(
        "--llama-server-binary",
        default="/usr/local/lib/ollama/llama-server",
    )

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--manifest", required=True)
    analyze.add_argument("--registry", required=True)
    analyze.add_argument("--policy-split", required=True)
    analyze.add_argument("--results", required=True)
    analyze.add_argument("--output", default=None, help="optional output JSON path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "freeze":
        report = freeze_screen_manifest(
            args.output, args.registry, args.policy_split, args.spec
        )
    elif args.command == "validate":
        registry = TreatmentRegistry.load(args.registry)
        policy_split = _load_json(args.policy_split)
        manifest = _load_json(args.manifest)
        validate_screen_manifest(manifest, registry, policy_split)
        report = {
            "valid": True,
            "manifest_hash": manifest["manifest_hash"],
            "screen_id": manifest["screen_id"],
        }
    elif args.command == "run":
        report = run_screen(
            args.manifest,
            args.registry,
            args.policy_split,
            args.output,
            RemoteConfig(
                args.host,
                args.remote_project,
                args.remote_run_root,
                args.remote_python,
            ),
            pi_binary=args.pi,
            provider=args.provider,
            model=args.model,
            thinking=args.thinking,
            unbrowser_binary=args.unbrowser_binary,
            model_artifact=args.model_artifact,
            llama_server_binary=args.llama_server_binary,
        )
    elif args.command == "analyze":
        report = analyze_screen(
            args.manifest,
            args.results,
            args.registry,
            args.policy_split,
            output_path=args.output,
        )
        return 0  # analyze prints JSON directly
    else:
        print(f"unknown command: {args.command}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SCHEMA_VERSION",
    "PANEL_RESULT_SCHEMA",
    "build_screen_manifest",
    "freeze_screen_manifest",
    "validate_screen_manifest",
    "run_screen",
    "analyze_screen",
]
