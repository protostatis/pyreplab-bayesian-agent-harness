"""Audit-ready semantic dataset package builder (stdlib-only).

This module freezes the ``T_canary`` semantic-capability screen into a
deterministic, leakage-safe, audit-ready dataset package.  It joins eight
inputs into a single output tree:

* a self-hashed **dataset contract** (declares the planned tasks / arms /
  replicas and the frozen source identities),
* the **frozen screen manifest**,
* a complete or partial **local panel JSONL** (the screen results),
* the runtime **preflight**,
* the authoritative **gate report**,
* the immutable **treatment registry**,
* the **policy split**, and
* a locally copied **raw run root**.

The emitted package contains exactly::

    data/attempts.jsonl
    raw/inventory.jsonl
    analysis/inclusion-ledger.jsonl
    analysis/gate-report.json
    QUALITY_AUDIT.json
    DATASET_CARD.md
    MANIFEST.json

``data/attempts.jsonl`` carries one row for *every* planned task x arm x
replica cell, and every row is ``task_role=T_canary``,
``governance_role=split=canary_excluded`` with a fully-``false``
``eligibility`` object.  Rows carry schema/version, task/panel/treatment,
execution/outcome/mechanism summaries, relative raw references plus hashes,
and provenance.  Raw message/thinking/tool payloads, stderr, diagnostics, and
absolute paths never appear in any derived file; the raw payloads live only in
the copied ``raw/`` tree (listed by ``raw/inventory.jsonl`` as relative path /
bytes / SHA-256 only).

The builder validates, fail-closed, the contract embedded SHA-256, exact
planned counts and task/bundle identities, panel/attempt joins, raw hashes,
complete cell coverage, source/registry/manifest identities, and gate-report
validity (``confirmation_pass`` or ``replication_no_go``, never ``invalid``).
A deterministic rebuild produces byte-identical files (no wall-clock, no
iteration-order dependence).

CLI::

    python -m pyreplab_harness.m3_semantic_dataset build \\
        --contract C --manifest M --results R --preflight P --gate G \\
        --registry REG --policy-split S --raw-root RAW --output OUT
    python -m pyreplab_harness.m3_semantic_dataset verify PACKAGE
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .m3_exploratory_screen import (
    PANEL_RESULT_SCHEMA,
    _validate_panel_result_strict,
    validate_screen_manifest,
)
from .m3_pilot import _load_json, _verify_embedded_hash
from .m3_semantic_capability_gate import _assess_semantic_specialist_adherence
from .treatments import TreatmentRegistry

# ---------------------------------------------------------------------------
# schema identifiers
# ---------------------------------------------------------------------------

PACKAGE_SCHEMA = "m3-semantic-dataset-package-v1"
ATTEMPT_SCHEMA = "m3-semantic-dataset-attempt-v1"
LEDGER_SCHEMA = "m3-semantic-dataset-ledger-v1"
GATE_REPORT_SCHEMA = "m3-semantic-dataset-gate-report-v1"
CONTRACT_SCHEMA = "m3-semantic-dataset-contract-v1"
AUDIT_SCHEMA = "m3-semantic-dataset-audit-v1"

_GATE_SCHEMAS = frozenset({
    "m3-semantic-capability-gate-v1",
    "m3-semantic-capability-gate-v2",
})
#: Authoritative gate decisions that have a valid dataset-package
#: interpretation.  ``invalid`` (or anything else) fails closed.
_CONFIRMATION_DECISIONS = frozenset({
    "mechanics_pass",
    "screen_pass",
    "confirmation_pass",
})
_REPLICATION_DECISIONS = frozenset({"futility_no_go", "replication_no_go"})
_VALID_GATE_DECISIONS = _CONFIRMATION_DECISIONS | _REPLICATION_DECISIONS

#: Excluded task roles and their reserved governance split labels.
_ROLE_TO_SPLIT: dict[str, str] = {
    "T_canary": "canary_excluded",
    "T_pilot": "pilot_excluded",
}

#: The four current governance pools; an excluded row is ineligible for all.
_ELIGIBILITY_POOLS: tuple[str, ...] = (
    "training",
    "calibration",
    "development",
    "final",
)

#: Keys that must never appear in any derived (non-raw) package file.
_FORBIDDEN_KEYS: frozenset[str] = frozenset(
    {
        "stderr",
        "pi_stderr",
        "tool_trace",
        "system_prompt",
        "thinking",
        "message",
        "messages",
        "diagnostics",
        "trajectory",
        "planning_preamble",
        "semantic_payload",
        "details",
        "tool_payload",
        "raw_message",
        "raw_thinking",
        "oracle",
        "private_metadata",
        "verifier_ref",
        "workspace_ref",
        "pi_events_ref",
        "normalized_events_ref",
        "verification_ref",
    }
)

#: The derived (non-raw) files that are subject to the privacy scan.
_DERIVED_FILES: tuple[str, ...] = (
    "data/attempts.jsonl",
    "raw/inventory.jsonl",
    "analysis/inclusion-ledger.jsonl",
    "analysis/gate-report.json",
    "QUALITY_AUDIT.json",
    "DATASET_CARD.md",
    "MANIFEST.json",
)


# ---------------------------------------------------------------------------
# deterministic serialisation helpers
# ---------------------------------------------------------------------------


def _dump_json(value: Any) -> str:
    """Deterministic single-line JSON serialisation (no key-order dependence)."""
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )


def _dump_jsonl_rows(rows: list[dict[str, Any]]) -> str:
    return "".join(_dump_json(row) + "\n" for row in rows)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(data)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _write_text(path: Path, text: str) -> None:
    _write_bytes(path, text.encode("utf-8"))


# ---------------------------------------------------------------------------
# privacy scanning
# ---------------------------------------------------------------------------


def _is_absolute_path(value: str) -> bool:
    if not value:
        return False
    if value.startswith("/"):
        return True
    # Windows-style drive paths (e.g. ``C:\\foo``, ``C:/foo``).
    if len(value) >= 3 and value[0].isalpha() and value[1] == ":" and value[2] in "/\\":
        return True
    return False


def privacy_scan(value: Any, prefix: str = "<root>") -> list[str]:
    """Recursively report forbidden keys and absolute-path string values."""
    violations: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str) and key in _FORBIDDEN_KEYS:
                violations.append(f"forbidden key {key!r} at {prefix}")
            violations.extend(privacy_scan(item, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            violations.extend(privacy_scan(item, f"{prefix}[{index}]"))
    elif isinstance(value, str):
        if _is_absolute_path(value):
            violations.append(f"absolute path value at {prefix}: {value!r}")
    return violations


def _privacy_scan_derived_files(base: Path) -> list[str]:
    """Scan every derived (non-raw) package file under ``base`` for privacy
    violations (forbidden keys and absolute-path values)."""
    violations: list[str] = []
    for rel in _DERIVED_FILES:
        path = base / rel
        if not path.is_file():
            continue
        if rel.endswith(".json"):
            violations.extend(
                privacy_scan(json.loads(path.read_text(encoding="utf-8")), rel)
            )
        elif rel.endswith(".jsonl"):
            violations.extend(
                privacy_scan(
                    [
                        json.loads(line)
                        for line in path.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ],
                    rel,
                )
            )
        else:
            violations.extend(privacy_scan(path.read_text(encoding="utf-8"), rel))
    return violations


# ---------------------------------------------------------------------------
# input loading and validation
# ---------------------------------------------------------------------------


def load_contract(path: str | Path) -> dict[str, Any]:
    """Load and validate the self-hashed dataset contract."""
    contract = _load_json(path)
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise ValueError(
            f"contract schema must be {CONTRACT_SCHEMA!r}, "
            f"got {contract.get('schema_version')!r}"
        )
    _verify_embedded_hash(contract, "contract_hash")
    return contract


def _validate_contract_against_sources(
    contract: dict[str, Any],
    manifest: dict[str, Any],
    registry: TreatmentRegistry,
    policy_split: dict[str, Any],
) -> None:
    """Validate the contract's planned identities and counts against sources."""
    contract_manifest_hash = contract.get("manifest_hash")
    if (
        contract_manifest_hash is not None
        and contract_manifest_hash != manifest.get("manifest_hash")
    ):
        raise ValueError("contract.manifest_hash does not match manifest")
    if contract.get("registry_hash") != registry.registry_hash:
        raise ValueError("contract.registry_hash does not match registry")
    if contract.get("policy_split_manifest_hash") != policy_split.get("manifest_hash"):
        raise ValueError(
            "contract.policy_split_manifest_hash does not match policy split"
        )
    protocol = manifest.get("protocol")
    dataset_contract = (
        protocol.get("dataset_contract") if isinstance(protocol, Mapping) else None
    )
    if not isinstance(dataset_contract, Mapping):
        raise ValueError("manifest protocol does not bind a dataset contract")
    if dataset_contract.get("schema_version") != CONTRACT_SCHEMA:
        raise ValueError("manifest dataset-contract schema does not match contract")
    if dataset_contract.get("contract_hash") != contract.get("contract_hash"):
        raise ValueError("manifest dataset-contract hash does not match contract")
    if dataset_contract.get("governance_role") != "canary_excluded":
        raise ValueError("manifest dataset governance role must be canary_excluded")

    task_role = contract.get("task_role")
    if task_role != manifest.get("task_role"):
        raise ValueError("contract.task_role does not match manifest task_role")
    if task_role not in _ROLE_TO_SPLIT:
        raise ValueError(f"unsupported excluded task_role: {task_role!r}")

    planned = contract.get("planned")
    if not isinstance(planned, Mapping):
        raise ValueError("contract.planned must be an object")

    planned_tasks = planned.get("tasks")
    if not isinstance(planned_tasks, list) or not planned_tasks:
        raise ValueError("contract.planned.tasks must be a non-empty list")
    manifest_tasks = [str(task["task_id"]) for task in manifest["tasks"]]
    if [str(t) for t in planned_tasks] != manifest_tasks:
        raise ValueError("contract.planned.tasks does not match manifest tasks")

    planned_bundles = planned.get("bundle_ids")
    if not isinstance(planned_bundles, list) or not planned_bundles:
        raise ValueError("contract.planned.bundle_ids must be a non-empty list")
    manifest_bundles = [str(bid) for bid in manifest["policy_bundle_ids"]]
    if [str(bid) for bid in planned_bundles] != manifest_bundles:
        raise ValueError(
            "contract.planned.bundle_ids does not match manifest policy_bundle_ids"
        )

    rollout_replicas = int(manifest.get("rollout_replicas", 1))
    if planned.get("rollout_replicas") != rollout_replicas:
        raise ValueError("contract.planned.rollout_replicas does not match manifest")
    expected_attempts = len(manifest_tasks) * len(manifest_bundles) * rollout_replicas
    if planned.get("attempts") != expected_attempts:
        raise ValueError(
            f"contract.planned.attempts {planned.get('attempts')!r} != "
            f"expected {expected_attempts}"
        )
    if manifest.get("gates", {}).get("attempts") != expected_attempts:
        raise ValueError("manifest gates.attempts does not match planned attempts")


def _governance_split(task_role: str) -> str:
    if task_role not in _ROLE_TO_SPLIT:
        raise ValueError(f"unsupported excluded task_role: {task_role!r}")
    return _ROLE_TO_SPLIT[task_role]


def _task_groups(manifest: dict[str, Any]) -> dict[str, set[str]]:
    protocol = manifest.get("protocol")
    stage = protocol.get("stage") if isinstance(protocol, Mapping) else None
    if stage in ("outcome_screen", "replication_screen"):
        tg = protocol.get("task_groups")
        if isinstance(tg, Mapping):
            return {
                "table": {str(tid) for tid in tg.get("table", [])},
                "form": {str(tid) for tid in tg.get("form", [])},
            }
    tasks = manifest.get("tasks", [])
    table = {str(tasks[0]["task_id"])} if tasks else set()
    form = {str(tasks[1]["task_id"])} if len(tasks) > 1 else set()
    return {"table": table, "form": form}


def _task_group_for(task_id: str, groups: dict[str, set[str]]) -> str:
    if task_id in groups["table"]:
        return "table"
    if task_id in groups["form"]:
        return "form"
    return ""


def _planned_cells(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for panel in manifest["panels"]:
        for execution_position, bundle_id in enumerate(panel["execution_order"]):
            cells.append(
                {
                    "panel_id": str(panel["panel_id"]),
                    "task_id": str(panel["task_id"]),
                    "bundle_id": str(bundle_id),
                    "rollout_replica": int(panel["rollout_replica"]),
                    "sampling_seed": int(panel["sampling_seed"]),
                    "execution_position": execution_position,
                }
            )
    return cells


def _load_safe_export(
    raw_root: str | Path,
    *,
    expected_attempt_ids: set[str],
    registry: TreatmentRegistry,
) -> tuple[dict[str, dict[str, Any]], str]:
    """Load and validate the standard leakage-safe attempt export."""
    path = Path(raw_root).expanduser().resolve() / "attempts.safe.jsonl"
    if not path.is_file():
        raise ValueError("raw run root is missing attempts.safe.jsonl")
    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"safe export line {line_number} is not an object")
        attempt_id = row.get("attempt_id")
        if not isinstance(attempt_id, str) or not attempt_id:
            raise ValueError(f"safe export line {line_number} has no attempt_id")
        if attempt_id in rows:
            raise ValueError(f"duplicate safe-export attempt id: {attempt_id}")
        if (
            row.get("task_role") != "T_canary"
            or row.get("split") != "canary_excluded"
            or row.get("governance_role") != "canary_excluded"
            or row.get("eligibility")
            != {pool: False for pool in _ELIGIBILITY_POOLS}
        ):
            raise ValueError(
                f"safe export governance mismatch for attempt {attempt_id!r}"
            )
        bundle_id = row.get("treatment_bundle_id")
        treatment = registry.by_bundle_id(str(bundle_id))
        if (
            row.get("treatment_bundle_hash") != treatment.bundle_hash
            or row.get("treatment_registry_hash") != registry.registry_hash
            or row.get("policy_id") != treatment.id
            or row.get("policy_version") != treatment.version
        ):
            raise ValueError(
                f"safe export treatment identity mismatch for attempt {attempt_id!r}"
            )
        rows[attempt_id] = row
    if set(rows) != expected_attempt_ids:
        raise ValueError(
            "safe export attempt IDs do not exactly match panel results: "
            f"missing={sorted(expected_attempt_ids - set(rows))}, "
            f"extra={sorted(set(rows) - expected_attempt_ids)}"
        )
    return rows, _sha256_file(path)


# ---------------------------------------------------------------------------
# raw run root copy + inventory
# ---------------------------------------------------------------------------


def _copy_raw_root(raw_root: str | Path, output_dir: Path) -> list[dict[str, Any]]:
    """Copy every file under ``raw_root`` into ``output_dir`` preserving
    relative structure; return relative-path/bytes/SHA-256 inventory rows."""
    root = Path(raw_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"raw run root does not exist: {root}")
    inventory: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        dest = output_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        data = path.read_bytes()
        inventory.append(
            {
                "path": f"raw/{rel}",
                "bytes": len(data),
                "sha256": _sha256_bytes(data),
            }
        )
    return inventory


def _raw_index(inventory: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map attempt IDs to their raw event inventory entry."""
    index: dict[str, dict[str, Any]] = {}
    for entry in inventory:
        parts = Path(entry["path"]).parts
        if len(parts) >= 4 and parts[0] == "raw" and parts[1] == "attempts":
            attempt_id = parts[2]
            if parts[-1] == "pi-events.jsonl":
                index[attempt_id] = entry
    return index


# ---------------------------------------------------------------------------
# attempt extraction
# ---------------------------------------------------------------------------


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _output_tokens_or_none(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and math.isfinite(value) and value >= 0:
        return value
    return None


def _build_attempt_row(
    cell: dict[str, Any],
    record: Mapping[str, Any] | None,
    *,
    registry: TreatmentRegistry,
    manifest: dict[str, Any],
    task_groups: dict[str, set[str]],
    raw_index: dict[str, dict[str, Any]],
    contract: dict[str, Any],
    preflight: dict[str, Any],
    governance_split: str,
    task_by_id: dict[str, dict[str, Any]],
    safe_by_attempt_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    bundle_id = cell["bundle_id"]
    task_id = cell["task_id"]
    treatment = registry.by_bundle_id(bundle_id)
    meta = treatment.generator_metadata
    capability = str(meta.get("capability", ""))
    task = task_by_id[task_id]
    group = _task_group_for(task_id, task_groups)

    if record is None:
        status = "not_run"
        attempt: Mapping[str, Any] | None = None
    elif record.get("status") == "error":
        status = "error"
        attempt = None
    else:
        status = "completed"
        attempts = record.get("result", {}).get("attempts", {})
        attempt = attempts.get(bundle_id)
        if not isinstance(attempt, Mapping):
            raise ValueError(
                f"completed panel {cell['panel_id']!r} is missing attempt "
                f"for {bundle_id!r}"
            )

    safe_row = (
        safe_by_attempt_id.get(str(attempt.get("attempt_id")))
        if attempt is not None
        else None
    )
    if attempt is not None and safe_row is None:
        raise ValueError(
            f"safe export row missing for attempt {attempt.get('attempt_id')!r}"
        )
    public_metadata_keys = (
        "difficulty",
        "fixture_url",
        "network_mode",
        "page_description",
        "required_output",
        "task_role",
        "template",
    )
    safe_public_metadata = (
        safe_row.get("public_metadata") if isinstance(safe_row, Mapping) else {}
    )
    safe_public_metadata = (
        safe_public_metadata if isinstance(safe_public_metadata, Mapping) else {}
    )

    row: dict[str, Any] = {
        "schema_version": ATTEMPT_SCHEMA,
        "dataset_id": contract["dataset_id"],
        "contract_hash": contract["contract_hash"],
        "task_role": contract["task_role"],
        "governance_role": governance_split,
        "split": governance_split,
        "eligibility": {pool: False for pool in _ELIGIBILITY_POOLS},
        "task": {
            "task_id": task_id,
            "family": safe_row.get("family") if safe_row else "unbrowser_fixture",
            "template": task["template"],
            "difficulty": task["difficulty"],
            "seed": task["seed"],
            "task_group": group,
            "generator_version": (
                safe_row.get("generator_version") if safe_row else None
            ),
            "prompt": safe_row.get("prompt") if safe_row else None,
            "contract": safe_row.get("contract") if safe_row else None,
            "public_metadata": {
                key: safe_public_metadata[key]
                for key in public_metadata_keys
                if key in safe_public_metadata
            },
        },
        "panel": {
            "panel_id": cell["panel_id"],
            "rollout_replica": cell["rollout_replica"],
            "sampling_seed": cell["sampling_seed"],
            "execution_position": cell["execution_position"],
        },
        "treatment": {
            "bundle_id": bundle_id,
            "bundle_hash": treatment.bundle_hash,
            "policy_id": treatment.id,
            "policy_version": treatment.version,
            "tool_interface": treatment.tool_interface,
            "capability": capability,
            "parent_bundle_id": str(meta.get("parent_bundle_id", "")),
            "substrate": str(meta.get("substrate", "")),
            "allowed_tools": list(treatment.allowed_tools),
            "max_output_tokens": treatment.max_output_tokens,
            "tool_call_limit": treatment.tool_call_limit,
            "command_timeout_seconds": treatment.command_timeout_seconds,
            "wall_time_limit_seconds": treatment.wall_time_limit_seconds,
        },
        "provenance": {
            "manifest_hash": manifest["manifest_hash"],
            "registry_hash": registry.registry_hash,
            "policy_split_manifest_hash": manifest.get("policy_split_manifest_hash"),
            "code_revision": preflight.get("code_revision"),
            "source_tree_hash": preflight.get("source_tree_hash"),
            "worktree_status_hash": preflight.get("worktree_status_hash"),
        },
    }

    if attempt is None:
        row["execution"] = {
            "status": status,
            "attempt_id": None,
            "pi_return_code": None,
            "provider_turn_count": None,
            "tool_call_count": None,
            "tool_limit_rejection_count": None,
            "length_stop_count": None,
        }
        row["outcome"] = {
            "success": None,
            "verifier_id": None,
            "verifier_version": None,
            "output_tokens": None,
        }
        row["mechanism"] = {
            "specialist": capability,
            "specialist_receipt_valid": None,
            "specialist_action_match": None,
            "unavailable_specialist_found": None,
            "infrastructure_errors": None,
            "tool_cap_compliant": None,
            "admitted_tool_call_count": None,
            "rejected_tool_call_count": None,
        }
        row["raw"] = None
        return row

    traj = attempt.get("trajectory")
    traj = traj if isinstance(traj, Mapping) else {}
    row["execution"] = {
        "status": status,
        "attempt_id": attempt.get("attempt_id"),
        "pi_return_code": _int_or_none(attempt.get("pi_return_code")),
        "provider_turn_count": _int_or_none(traj.get("provider_turn_count")),
        "tool_call_count": _int_or_none(traj.get("tool_call_count")),
        "tool_limit_rejection_count": _int_or_none(
            traj.get("tool_limit_rejection_count")
        ),
        "length_stop_count": _int_or_none(traj.get("length_stop_count")),
        "assistant_message_count": (
            _int_or_none(safe_row.get("assistant_message_count")) if safe_row else None
        ),
        "termination_class": (
            safe_row.get("termination_class") if safe_row else None
        ),
        "usage": dict(safe_row.get("usage", {})) if safe_row else {},
        "timing": (
            dict(attempt.get("timing", {}))
            if isinstance(attempt.get("timing"), Mapping)
            else {}
        ),
        "sampling_receipt_valid": attempt.get("sampling_receipt")
        == {
            "seed": cell["sampling_seed"],
            "parameters": manifest["runtime_pins"]["sampling"]["parameters"],
        },
    }

    verif = attempt.get("verification")
    verif = verif if isinstance(verif, Mapping) else {}
    usage = attempt.get("usage")
    usage = usage if isinstance(usage, Mapping) else {}
    row["outcome"] = {
        "success": (
            bool(verif["success"]) if isinstance(verif.get("success"), bool) else None
        ),
        "verifier_id": verif.get("verifier_id"),
        "verifier_version": verif.get("verifier_version"),
        "failure_code": safe_row.get("failure_code") if safe_row else None,
        "output_tokens": _output_tokens_or_none(usage.get("output")),
    }

    adherence = _assess_semantic_specialist_adherence(
        treatment, traj, task_group=group
    )
    row["mechanism"] = {
        "specialist": adherence["specialist"],
        "specialist_receipt_valid": adherence["specialist_receipt_valid"],
        "specialist_action_match": adherence["specialist_action_match"],
        "unavailable_specialist_found": adherence["unavailable_specialist_found"],
        "infrastructure_errors": adherence["infrastructure_errors"],
        "tool_cap_compliant": adherence["tool_cap_compliant"],
        "admitted_tool_call_count": adherence["admitted_tool_call_count"],
        "rejected_tool_call_count": adherence["rejected_tool_call_count"],
    }

    raw_entry = raw_index.get(str(attempt.get("attempt_id") or ""))
    row["raw"] = (
        {
            "path": raw_entry["path"],
            "bytes": raw_entry["bytes"],
            "sha256": raw_entry["sha256"],
        }
        if raw_entry is not None
        else None
    )
    return row


def _build_ledger_row(
    cell: dict[str, Any],
    attempt_row: dict[str, Any],
    *,
    contract: dict[str, Any],
    governance_split: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "schema_version": LEDGER_SCHEMA,
        "dataset_id": contract["dataset_id"],
        "contract_hash": contract["contract_hash"],
        "task_id": cell["task_id"],
        "bundle_id": cell["bundle_id"],
        "panel_id": cell["panel_id"],
        "attempt_id": attempt_row["execution"]["attempt_id"],
        "attempt_status": attempt_row["execution"]["status"],
        "governance_role": governance_split,
        "eligible_for_training": False,
        "eligible_for_calibration": False,
        "eligible_for_development": False,
        "eligible_for_final": False,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# gate report
# ---------------------------------------------------------------------------


def _validate_gate_report(
    gate: Mapping[str, Any], manifest_hash: str
) -> tuple[str, str]:
    """Validate the authoritative gate report; return ``(verdict, decision)``.

    ``verdict`` is ``confirmation_pass`` for a passing mechanics/screen decision
    and ``replication_no_go`` for a valid no-go. Any other decision
    (including ``invalid``) fails closed.
    """
    if gate.get("schema_version") not in _GATE_SCHEMAS:
        raise ValueError(
            f"gate report schema must be one of {sorted(_GATE_SCHEMAS)!r}, "
            f"got {gate.get('schema_version')!r}"
        )
    if gate.get("manifest_hash") != manifest_hash:
        raise ValueError("gate report manifest_hash does not match manifest")
    decision = gate.get("decision")
    if decision in _CONFIRMATION_DECISIONS:
        verdict = "confirmation_pass"
    elif decision in _REPLICATION_DECISIONS:
        verdict = "replication_no_go"
    else:
        raise ValueError(
            f"gate report decision {decision!r} is invalid; expected one of "
            f"{sorted(_VALID_GATE_DECISIONS)}"
        )
    if isinstance(gate.get("passed"), bool) is False:
        raise ValueError("gate report 'passed' must be a boolean")
    expected_passed = decision in _CONFIRMATION_DECISIONS
    if gate.get("passed") is not expected_passed:
        raise ValueError("gate report passed flag is inconsistent with decision")
    return verdict, str(decision)


# ---------------------------------------------------------------------------
# audit assembly
# ---------------------------------------------------------------------------


def _check(name: str, passed: bool, detail: str = "") -> dict[str, Any]:
    entry: dict[str, Any] = {"name": name, "passed": bool(passed)}
    if detail:
        entry["detail"] = detail
    return entry


def _build_audit(
    checks: list[dict[str, Any]],
    *,
    contract: dict[str, Any],
    manifest: dict[str, Any],
    governance_split: str,
    counts: dict[str, int],
) -> dict[str, Any]:
    return {
        "schema_version": AUDIT_SCHEMA,
        "dataset_id": contract["dataset_id"],
        "contract_hash": contract["contract_hash"],
        "task_role": contract["task_role"],
        "governance_role": governance_split,
        "manifest_hash": manifest["manifest_hash"],
        "counts": dict(sorted(counts.items())),
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# package builder
# ---------------------------------------------------------------------------


def _load_results_records(
    results_path: str | Path,
    manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Load and strictly validate the (complete or partial) panel JSONL.

    Returns ``{panel_id: record}``.  Present records are validated against the
    manifest with the shared strict validator; duplicate/unknown panels,
    schema drift, and malformed records fail closed (never skipped).
    """
    path = Path(results_path).expanduser().resolve()
    panel_by_id = {str(p["panel_id"]): p for p in manifest["panels"]}
    task_by_id = {str(t["task_id"]): t for t in manifest["tasks"]}
    records: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        raise FileNotFoundError(f"results JSONL does not exist: {path}")
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid results JSONL line {line_number}: {error}") from error
        if not isinstance(record, Mapping):
            raise ValueError(f"line {line_number}: record must be an object")
        if record.get("schema_version") != PANEL_RESULT_SCHEMA:
            raise ValueError(
                f"line {line_number}: unknown results schema "
                f"{record.get('schema_version')!r}"
            )
        pid = str(record.get("panel_id", ""))
        if pid in records:
            raise ValueError(f"line {line_number}: duplicate panel {pid}")
        if pid not in panel_by_id:
            raise ValueError(f"line {line_number}: unknown panel_id {pid!r}")
        if record.get("manifest_hash") != manifest["manifest_hash"]:
            raise ValueError(f"line {line_number}: manifest hash mismatch")
        mpanel = panel_by_id[pid]
        mpanel_task = task_by_id[mpanel["task_id"]]
        errors = _validate_panel_result_strict(record, manifest, mpanel, mpanel_task)
        if errors:
            raise ValueError(
                f"line {line_number}: invalid panel result: {'; '.join(errors)}"
            )
        records[pid] = dict(record)
    return records


def _validate_attempt_join(
    records: dict[str, dict[str, Any]], manifest: dict[str, Any]
) -> None:
    """Validate panel/attempt joins: every attempt joins to its panel's
    execution order, and attempt IDs are unique."""
    bundle_by_panel = {
        str(p["panel_id"]): set(str(bid) for bid in p["execution_order"])
        for p in manifest["panels"]
    }
    seen_attempt_ids: set[str] = set()
    for pid, record in records.items():
        if record.get("status") != "completed":
            continue
        attempts = record.get("result", {}).get("attempts", {})
        expected = bundle_by_panel[pid]
        found = {str(bid) for bid in attempts}
        if found != expected:
            extra = found - expected
            missing = expected - found
            parts = []
            if extra:
                parts.append(f"extra attempts {sorted(extra)}")
            if missing:
                parts.append(f"missing attempts {sorted(missing)}")
            raise ValueError(f"panel {pid!r} attempt join mismatch: {'; '.join(parts)}")
        for item in attempts.values():
            if not isinstance(item, Mapping):
                continue
            aid = item.get("attempt_id")
            if not isinstance(aid, str) or not aid:
                raise ValueError(f"panel {pid!r}: attempt id missing")
            if aid in seen_attempt_ids:
                raise ValueError(f"duplicate attempt id: {aid}")
            seen_attempt_ids.add(aid)


def build_package(
    *,
    contract_path: str | Path,
    manifest_path: str | Path,
    results_path: str | Path,
    preflight_path: str | Path,
    gate_path: str | Path,
    registry_path: str | Path,
    policy_split_path: str | Path,
    raw_root: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Build the deterministic audit-ready dataset package; return the audit.

    All files are staged in a sibling temporary directory and atomically moved
    into ``output`` only after every validation (including the privacy scan)
    passes.  Any failure raises and leaves no partial package behind.
    """
    contract = load_contract(contract_path)
    manifest = _load_json(manifest_path)
    registry = TreatmentRegistry.load(registry_path)
    policy_split = _load_json(policy_split_path)
    preflight = _load_json(preflight_path)
    gate = _load_json(gate_path)

    # ---- source identities -------------------------------------------------
    validate_screen_manifest(manifest, registry, policy_split)
    _validate_contract_against_sources(contract, manifest, registry, policy_split)
    if preflight.get("manifest_hash") != manifest["manifest_hash"]:
        raise ValueError("preflight.manifest_hash does not match manifest")
    if preflight.get("screen_preflight") is not True:
        raise ValueError("preflight.screen_preflight is not true")
    gate_verdict, _decision = _validate_gate_report(gate, manifest["manifest_hash"])
    gate_sha256 = _sha256_file(Path(gate_path).expanduser().resolve())
    preflight_sha256 = _sha256_file(Path(preflight_path).expanduser().resolve())

    governance_split = _governance_split(str(contract["task_role"]))
    task_role = str(contract["task_role"])
    if task_role != str(manifest.get("task_role")):
        raise ValueError("contract task_role does not match manifest task_role")

    task_by_id = {str(t["task_id"]): t for t in manifest["tasks"]}
    groups = _task_groups(manifest)
    cells = _planned_cells(manifest)
    records = _load_results_records(results_path, manifest)
    _validate_attempt_join(records, manifest)
    expected_panels = {str(panel["panel_id"]) for panel in manifest["panels"]}
    if set(records) != expected_panels:
        missing = sorted(expected_panels - set(records))
        raise ValueError(
            "results must contain every planned panel before packaging; "
            f"missing={missing}"
        )
    expected_attempt_ids = {
        str(item["attempt_id"])
        for record in records.values()
        for item in record.get("result", {}).get("attempts", {}).values()
        if isinstance(item, Mapping) and isinstance(item.get("attempt_id"), str)
    }
    safe_by_attempt_id, safe_export_sha256 = _load_safe_export(
        raw_root,
        expected_attempt_ids=expected_attempt_ids,
        registry=registry,
    )

    # ---- stage the package --------------------------------------------------
    output_dir = Path(output).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging.", dir=output_dir.parent)
    )
    try:
        raw_dir = staging / "raw"
        inventory = _copy_raw_root(raw_root, raw_dir)
        raw_index = _raw_index(inventory)

        reason = (
            f"{task_role} task; permanently excluded from meta-training, "
            "calibration, development, and final evaluation pools."
        )
        rows: list[dict[str, Any]] = []
        ledger: list[dict[str, Any]] = []
        for cell in cells:
            record = records.get(cell["panel_id"])
            row = _build_attempt_row(
                cell,
                record,
                registry=registry,
                manifest=manifest,
                task_groups=groups,
                raw_index=raw_index,
                contract=contract,
                preflight=preflight,
                governance_split=governance_split,
                task_by_id=task_by_id,
                safe_by_attempt_id=safe_by_attempt_id,
            )
            rows.append(row)
            if row["execution"]["status"] != "completed":
                raise ValueError(
                    "every planned cell must be completed before packaging: "
                    f"{cell['panel_id']}/{cell['bundle_id']}"
                )
            if row["raw"] is None:
                raise ValueError(
                    "raw Pi events are missing for completed attempt: "
                    f"{row['execution']['attempt_id']!r}"
                )
            ledger.append(
                _build_ledger_row(
                    cell,
                    row,
                    contract=contract,
                    governance_split=governance_split,
                    reason=reason,
                )
            )

        safe_export = Path(raw_root).expanduser().resolve() / "attempts.safe.jsonl"
        safe_export_rows = len(safe_by_attempt_id)
        if safe_export_rows != len(cells):
            raise ValueError(
                f"standard safe export has {safe_export_rows} rows; "
                f"expected {len(cells)}"
            )

        # ---- write data / raw / analysis files ------------------------------
        _write_text(staging / "data" / "attempts.jsonl", _dump_jsonl_rows(rows))
        _write_text(staging / "raw" / "inventory.jsonl", _dump_jsonl_rows(inventory))
        _write_text(
            staging / "analysis" / "inclusion-ledger.jsonl",
            _dump_jsonl_rows(ledger),
        )
        gate_report = {
            "schema_version": GATE_REPORT_SCHEMA,
            "dataset_id": contract["dataset_id"],
            "contract_hash": contract["contract_hash"],
            "verdict": gate_verdict,
            "manifest_hash": manifest["manifest_hash"],
            "authoritative_gate_sha256": gate_sha256,
            "authoritative_gate": gate,
        }
        _write_text(
            staging / "analysis" / "gate-report.json", _dump_json(gate_report) + "\n"
        )

        # ---- counts ---------------------------------------------------------
        counts = {
            "tasks": len(manifest["tasks"]),
            "arms": len(manifest["policy_bundle_ids"]),
            "rollout_replicas": int(manifest.get("rollout_replicas", 1)),
            "cells": len(cells),
            "completed": sum(1 for r in rows if r["execution"]["status"] == "completed"),
            "error": sum(1 for r in rows if r["execution"]["status"] == "error"),
            "not_run": sum(1 for r in rows if r["execution"]["status"] == "not_run"),
            "raw_files": len(inventory),
            "records": len(records),
            "safe_export_rows": safe_export_rows,
        }

        checks: list[dict[str, Any]] = [
            _check(
                "contract_embedded_sha256",
                True,
                f"contract {contract['dataset_id']!r} hash-verified",
            ),
            _check(
                "planned_counts_and_ids",
                True,
                f"{len(manifest['tasks'])} tasks, "
                f"{len(manifest['policy_bundle_ids'])} arms, "
                f"{counts['rollout_replicas']} replicas, {counts['cells']} cells",
            ),
            _check(
                "panel_attempt_join",
                True,
                f"{len(records)} panel records joined",
            ),
            _check(
                "all_cells_covered",
                len(rows) == len(cells),
                f"{len(rows)} rows for {len(cells)} planned cells",
            ),
            _check(
                "no_exporter_skips",
                len(rows) == len(cells)
                and len(ledger) == len(cells)
                and counts["completed"] == len(cells)
                and counts["error"] == 0
                and counts["not_run"] == 0,
                f"{len(rows)} attempt rows, {len(ledger)} ledger rows",
            ),
            _check(
                "standard_safe_export_complete",
                counts["safe_export_rows"] == len(cells),
                f"{counts['safe_export_rows']} canary-excluded staging rows",
            ),
            _check(
                "source_identities",
                True,
                "manifest/registry/policy-split/preflight/gate identities match",
            ),
            _check(
                "gate_report_valid",
                gate_verdict in ("confirmation_pass", "replication_no_go"),
                f"verdict {gate_verdict!r}",
            ),
        ]

        audit = _build_audit(
            checks,
            contract=contract,
            manifest=manifest,
            governance_split=governance_split,
            counts=counts,
        )
        if not audit["passed"]:
            raise ValueError(
                "audit failed: "
                + "; ".join(
                    check["name"] for check in checks if not check["passed"]
                )
            )

        # ---- write audit, card, manifest ------------------------------------
        _write_text(staging / "QUALITY_AUDIT.json", _dump_json(audit) + "\n")
        card = _dataset_card(
            contract=contract,
            manifest=manifest,
            governance_split=governance_split,
            gate_verdict=gate_verdict,
            gate_sha256=gate_sha256,
            preflight_sha256=preflight_sha256,
            preflight=preflight,
            safe_export_sha256=safe_export_sha256,
            counts=counts,
        )
        _write_text(staging / "DATASET_CARD.md", card)

        manifest_payload = _package_manifest(
            contract=contract,
            manifest=manifest,
            registry=registry,
            governance_split=governance_split,
            gate_sha256=gate_sha256,
            preflight_sha256=preflight_sha256,
            preflight=preflight,
            source_hashes={
                "contract_file_sha256": _sha256_file(
                    Path(contract_path).expanduser().resolve()
                ),
                "screen_manifest_file_sha256": _sha256_file(
                    Path(manifest_path).expanduser().resolve()
                ),
                "results_file_sha256": _sha256_file(
                    Path(results_path).expanduser().resolve()
                ),
                "registry_file_sha256": _sha256_file(
                    Path(registry_path).expanduser().resolve()
                ),
                "policy_split_file_sha256": _sha256_file(
                    Path(policy_split_path).expanduser().resolve()
                ),
                "safe_export_file_sha256": safe_export_sha256,
                "builder_source_sha256": _sha256_file(Path(__file__).resolve()),
            },
            counts=counts,
            staging=staging,
        )
        _write_text(staging / "MANIFEST.json", _dump_json(manifest_payload) + "\n")

        # ---- final privacy scan over every derived file ---------------------
        final_violations = _privacy_scan_derived_files(staging)
        if final_violations:
            raise ValueError(
                "privacy scan failed: " + "; ".join(final_violations)
            )

        # ---- atomically move staging into place -----------------------------
        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return audit


def _dataset_card(
    *,
    contract: dict[str, Any],
    manifest: dict[str, Any],
    governance_split: str,
    gate_verdict: str,
    gate_sha256: str,
    preflight_sha256: str,
    preflight: dict[str, Any],
    safe_export_sha256: str,
    counts: dict[str, int],
) -> str:
    lines = [
        "# M3 Semantic-Capability Canary Dataset Package",
        "",
        f"- Dataset ID: `{contract['dataset_id']}`",
        f"- Contract SHA-256: `{contract['contract_hash']}`",
        f"- Task role: `{contract['task_role']}`",
        f"- Governance role / split: `{governance_split}`",
        f"- Manifest SHA-256: `{manifest['manifest_hash']}`",
        f"- Registry SHA-256: `{manifest['registry_hash']}`",
        f"- Policy-split manifest SHA-256: `{manifest['policy_split_manifest_hash']}`",
        f"- Authoritative gate SHA-256: `{gate_sha256}`",
        f"- Gate verdict: `{gate_verdict}`",
        f"- Preflight SHA-256: `{preflight_sha256}`",
        f"- Code revision: `{preflight.get('code_revision')}`",
        f"- Source-tree SHA-256: `{preflight.get('source_tree_hash')}`",
        f"- Standard safe-export SHA-256: `{safe_export_sha256}`",
        "",
        "## Eligibility",
        "",
        "Every row is permanently excluded from meta-training, calibration,",
        "development, and final evaluation pools.",
        "",
        "```json",
        json.dumps(
            {pool: False for pool in _ELIGIBILITY_POOLS},
            sort_keys=True,
            ensure_ascii=False,
        ),
        "```",
        "",
        "## Counts",
        "",
    ]
    for key, value in sorted(counts.items()):
        lines.append(f"- {key}: {value}")
    lines += [
        "",
        "## Boundary",
        "",
        "This package is audit-only screening evidence for the excluded",
        "semantic-capability canary. It does not qualify an allocator, a causal",
        "treatment effect, or a population generalization.",
        "",
        "## Collection",
        "",
        "The dataset contains 16 fresh synthetic browser tasks: eight table",
        "filter/sort tasks and eight form-entry tasks. Each task was run with",
        "both specialist arms across three replicas using a panel-common sampling",
        "seed. All 48 panels and 96 attempts ran sequentially with no early stop",
        "or outcome-driven replacement.",
        "",
        "## Outcome And Validity",
        "",
        f"The frozen replication gate returned `{gate_verdict}`. This is a valid",
        "complete run, not an infrastructure-invalid run. Read the bundled gate",
        "report before interpreting task outcomes.",
        "",
        "## Data Layers",
        "",
        "- `data/attempts.jsonl`: privacy-scanned, one row per planned cell.",
        "- `analysis/inclusion-ledger.jsonl`: explicit exclusion decision per row.",
        "- `analysis/gate-report.json`: authoritative preregistered gate output.",
        "- `raw/`: restricted source artifacts plus a SHA-256 inventory.",
        "",
        "## Privacy And Security",
        "",
        "Derived rows omit model messages and thinking, tool payloads, stderr,",
        "verifier diagnostics, private oracle data, and absolute paths. Raw files",
        "remain restricted and may contain those fields.",
        "",
        "## License And Access",
        "",
        "Internal research only pending review of model-output redistribution",
        "terms. Synthetic task fixtures are harness-owned; raw model traces are",
        "not approved for public redistribution.",
        "",
        "## Future Use",
        "",
        "A future experiment may cite this immutable release for audit, regression,",
        "error analysis, or prospective design. It may use these rows for fitting",
        "only under a new preregistration that names this exact release hash and",
        "keeps a fresh untouched evaluation panel. These rows never become",
        "prospective evidence for the current M3 experiment.",
        "",
        "## Limitations",
        "",
        "The tasks are targeted synthetic fixtures from two templates. The task is",
        "the independent generalization unit; replicas are repeated measurements.",
        "The results do not establish real-web performance, mixed-page routing,",
        "allocator effectiveness, or a broad semantic-capability effect.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _package_manifest(
    *,
    contract: dict[str, Any],
    manifest: dict[str, Any],
    registry: TreatmentRegistry,
    governance_split: str,
    gate_sha256: str,
    preflight_sha256: str,
    preflight: dict[str, Any],
    source_hashes: dict[str, str],
    counts: dict[str, int],
    staging: Path,
) -> dict[str, Any]:
    schedule = [
        {
            "panel_id": str(p["panel_id"]),
            "task_id": str(p["task_id"]),
            "rollout_replica": int(p["rollout_replica"]),
            "sampling_seed": int(p["sampling_seed"]),
            "execution_order": [str(bid) for bid in p["execution_order"]],
        }
        for p in manifest["panels"]
    ]
    files: dict[str, dict[str, Any]] = {}
    for rel in sorted(_DERIVED_FILES):
        if rel == "MANIFEST.json":
            continue  # MANIFEST cannot hash itself
        path = staging / rel
        data = path.read_bytes()
        files[rel] = {"bytes": len(data), "sha256": _sha256_bytes(data)}
    return {
        "schema_version": PACKAGE_SCHEMA,
        "dataset_id": contract["dataset_id"],
        "contract_hash": contract["contract_hash"],
        "task_role": contract["task_role"],
        "governance_role": governance_split,
        "eligibility": {pool: False for pool in _ELIGIBILITY_POOLS},
        "identities": {
            "manifest_hash": manifest["manifest_hash"],
            "registry_hash": registry.registry_hash,
            "policy_split_manifest_hash": manifest["policy_split_manifest_hash"],
            "authoritative_gate_sha256": gate_sha256,
            "preflight_sha256": preflight_sha256,
            "code_revision": preflight.get("code_revision"),
            "source_tree_hash": preflight.get("source_tree_hash"),
            "worktree_status_hash": preflight.get("worktree_status_hash"),
            **dict(sorted(source_hashes.items())),
        },
        "counts": dict(sorted(counts.items())),
        "schedule": schedule,
        "files": files,
    }


# ---------------------------------------------------------------------------
# package verification (from the package alone)
# ---------------------------------------------------------------------------


def verify_package(package: str | Path) -> dict[str, Any]:
    """Re-validate a built package from its own files; return the audit."""
    root = Path(package).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"package does not exist: {root}")
    manifest_path = root / "MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"MANIFEST.json missing from {root}")
    pkg = json.loads(manifest_path.read_text(encoding="utf-8"))
    if pkg.get("schema_version") != PACKAGE_SCHEMA:
        raise ValueError("unknown package schema_version")
    if not isinstance(pkg.get("contract_hash"), str) or len(pkg["contract_hash"]) != 64:
        raise ValueError("MANIFEST.json contract_hash is not a 64-hex digest")

    dataset_id = pkg["dataset_id"]
    governance_split = pkg.get("governance_role")
    task_role = pkg.get("task_role")
    checks: list[dict[str, Any]] = []

    # ---- file hashes --------------------------------------------------------
    files = pkg.get("files", {})
    expected_files = {rel for rel in _DERIVED_FILES if rel != "MANIFEST.json"}
    hash_ok = True
    hash_details: list[str] = []
    if set(files) != expected_files:
        hash_ok = False
        hash_details.append(
            f"file manifest set mismatch: {sorted(set(files) ^ expected_files)}"
        )
    for rel in sorted(files):
        path = root / rel
        if not path.is_file():
            hash_ok = False
            hash_details.append(f"missing file {rel}")
            continue
        data = path.read_bytes()
        expected = files[rel]
        if (
            expected.get("bytes") != len(data)
            or expected.get("sha256") != _sha256_bytes(data)
        ):
            hash_ok = False
            hash_details.append(f"hash mismatch for {rel}")
    checks.append(_check("file_hashes", hash_ok, "; ".join(hash_details) or "clean"))

    # ---- gate report validity ----------------------------------------------
    gate_path = root / "analysis" / "gate-report.json"
    gate_report = json.loads(gate_path.read_text(encoding="utf-8"))
    gate = gate_report.get("authoritative_gate")
    gate_ok = (
        gate_report.get("schema_version") == GATE_REPORT_SCHEMA
        and gate_report.get("contract_hash") == pkg["contract_hash"]
        and gate_report.get("manifest_hash") == pkg["identities"]["manifest_hash"]
        and isinstance(gate, Mapping)
        and gate.get("schema_version") in _GATE_SCHEMAS
        and gate.get("decision") in _VALID_GATE_DECISIONS
        and gate_report.get("verdict") in ("confirmation_pass", "replication_no_go")
    )
    checks.append(
        _check(
            "gate_report_valid",
            gate_ok,
            f"verdict {gate_report.get('verdict')!r}",
        )
    )

    # ---- inventory hashes ---------------------------------------------------
    inventory_path = root / "raw" / "inventory.jsonl"
    inventory = [
        json.loads(line)
        for line in inventory_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    inv_ok = True
    inv_details: list[str] = []
    for entry in inventory:
        rel = entry.get("path")
        if not isinstance(rel, str) or rel.startswith("/"):
            inv_ok = False
            inv_details.append(f"non-relative raw path {rel!r}")
            continue
        path = root / rel
        if not path.is_file():
            inv_ok = False
            inv_details.append(f"missing raw file {rel}")
            continue
        data = path.read_bytes()
        if (
            entry.get("bytes") != len(data)
            or entry.get("sha256") != _sha256_bytes(data)
        ):
            inv_ok = False
            inv_details.append(f"raw hash mismatch for {rel}")
    checks.append(
        _check("raw_hashes", inv_ok, "; ".join(inv_details) or "clean")
    )

    # ---- attempts coverage and joins ---------------------------------------
    attempts_path = root / "data" / "attempts.jsonl"
    attempts = [
        json.loads(line)
        for line in attempts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    schedule = pkg.get("schedule", [])
    expected_cells = []
    for panel in schedule:
        for bid in panel["execution_order"]:
            expected_cells.append((str(panel["panel_id"]), str(bid)))
    actual_cells = [(str(r["panel"]["panel_id"]), str(r["treatment"]["bundle_id"])) for r in attempts]
    cells_ok = (
        len(expected_cells) == len(attempts) == pkg.get("counts", {}).get("cells")
        and expected_cells == actual_cells
    )
    checks.append(
        _check(
            "all_cells_covered",
            cells_ok,
            f"{len(attempts)} rows for {len(expected_cells)} planned cells",
        )
    )
    rows_ok = all(
        r.get("schema_version") == ATTEMPT_SCHEMA
        and r.get("dataset_id") == dataset_id
        and r.get("contract_hash") == pkg["contract_hash"]
        and r.get("task_role") == task_role
        and r.get("governance_role") == governance_split
        and r.get("split") == governance_split
        and r.get("eligibility") == {pool: False for pool in _ELIGIBILITY_POOLS}
        for r in attempts
    )
    checks.append(_check("row_schema_and_eligibility", rows_ok, "all rows consistent"))

    # ---- ledger alignment ---------------------------------------------------
    ledger_path = root / "analysis" / "inclusion-ledger.jsonl"
    ledger = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ledger_ok = len(ledger) == len(attempts) and all(
        l.get("schema_version") == LEDGER_SCHEMA
        and l.get("governance_role") == governance_split
        and l.get("eligible_for_training") is False
        and l.get("eligible_for_calibration") is False
        and l.get("eligible_for_development") is False
        and l.get("eligible_for_final") is False
        for l in ledger
    )
    checks.append(
        _check("inclusion_ledger_aligned", ledger_ok, f"{len(ledger)} ledger rows")
    )

    # ---- privacy scan over every derived file -------------------------------
    violations = _privacy_scan_derived_files(root)
    checks.append(
        _check(
            "privacy_no_forbidden_keys_or_absolute_paths",
            not violations,
            "; ".join(violations) if violations else "clean",
        )
    )

    audit = {
        "schema_version": AUDIT_SCHEMA,
        "dataset_id": dataset_id,
        "contract_hash": pkg["contract_hash"],
        "task_role": task_role,
        "governance_role": governance_split,
        "manifest_hash": pkg["identities"]["manifest_hash"],
        "counts": pkg.get("counts", {}),
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }
    return audit


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyreplab-m3-semantic-dataset",
        description="Build or verify an audit-ready semantic dataset package.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build a deterministic dataset package")
    build.add_argument("--contract", required=True, help="self-hashed dataset contract JSON")
    build.add_argument("--manifest", required=True, help="frozen screen manifest JSON")
    build.add_argument("--results", required=True, help="local panel results JSONL (complete or partial)")
    build.add_argument("--preflight", required=True, help="runtime preflight JSON")
    build.add_argument("--gate", required=True, help="authoritative gate report JSON")
    build.add_argument("--registry", required=True, help="immutable treatment registry JSON")
    build.add_argument("--policy-split", required=True, help="policy split JSON")
    build.add_argument("--raw-root", required=True, help="raw run root directory to copy")
    build.add_argument("--output", required=True, help="package output directory")

    verify = subparsers.add_parser("verify", help="re-validate a built package")
    verify.add_argument("package", help="package directory produced by build")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build":
            audit = build_package(
                contract_path=args.contract,
                manifest_path=args.manifest,
                results_path=args.results,
                preflight_path=args.preflight,
                gate_path=args.gate,
                registry_path=args.registry,
                policy_split_path=args.policy_split,
                raw_root=args.raw_root,
                output=args.output,
            )
            print(_dump_json(audit))
        elif args.command == "verify":
            audit = verify_package(args.package)
            print(_dump_json(audit))
        else:  # pragma: no cover - argparse enforces subcommand
            return 1
    except (OSError, ValueError, RuntimeError) as error:
        print(f"m3 semantic dataset error: {error}", file=sys.stderr)
        return 1
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ATTEMPT_SCHEMA",
    "AUDIT_SCHEMA",
    "CONTRACT_SCHEMA",
    "GATE_REPORT_SCHEMA",
    "LEDGER_SCHEMA",
    "PACKAGE_SCHEMA",
    "build_package",
    "build_parser",
    "load_contract",
    "main",
    "privacy_scan",
    "verify_package",
]
