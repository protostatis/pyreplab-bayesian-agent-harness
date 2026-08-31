"""Dedicated authorized execution and analysis layer for the empty-overlay baseline.

This module is the only live-model execution path for the frozen
``m3_empty_overlay_baseline`` screen. The generic orchestrator refuses any
treatment whose ``generator_metadata.execution_path`` equals
``RESTRICTED_BASELINE_EXECUTION_PATH``; that boundary is not bypassed here but
satisfied with a separately authored, hash-bound execution authorization.

Governance model
----------------
* :func:`build_authorization_request` emits a **non-authorizing** request
  (``live_model_execution_authorized=False``) that binds the frozen manifest,
  registry, both preflights, the current source tree, the remote identity, the
  exact result filename, and the exact worst-case budget. There is no function
  or CLI in this module that turns a request into a valid authorization.
* :func:`validate_execution_authorization` accepts only a separately authored
  canonical JSON artifact with ``live_model_execution_authorized=True`` and
  ``single_use=True`` whose embedded hash exactly equals the operator-supplied
  expected hash. This is a governance gate, not a cryptographic signature.

The runner executes the 72 panels strictly sequentially under an exclusive
ephemeral lock, a persistent immutable claim, and a per-panel active marker.
Every JSONL record is self-hashed and bound to the authorization, manifest,
registry, both preflights, the source tree, and its exact panel coordinates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .batch import _append_result
from .events import (
    BUDGET_RECEIPT_SCHEMA_VERSION,
    NORMALIZED_EVENT_SCHEMA_VERSION,
    PROVIDER_TURN_SEMANTICS,
)
from .m3_empty_overlay_baseline import (
    EXPECTED_ATTEMPTS,
    LOCAL_PREFLIGHT_SCHEMA_VERSION,
    REMOTE_PREFLIGHT_SCHEMA_VERSION,
    SCREEN_ID,
    build_empty_overlay_registry,
    validate_baseline_manifest,
    validate_local_preflight,
    validate_remote_preflight,
)
from .m3_pilot import (
    _canonical_hash,
    _load_json,
    _verify_embedded_hash,
    _write_immutable_json,
    runtime_preflight,
    source_tree_hash,
)
from .m3_semantic_capability_gate import _is_infrastructure_error
from .orchestrator import (
    AttemptExecutionError,
    RemoteConfig,
    _run_attempt,
    _task_json,
    policy_spec_from_treatment,
    remote_json,
)
from .treatments import TreatmentRegistry
from .unbrowser_fixture_gym import (
    OUTCOME_ONLY_GENERATOR_VERSION,
    task_content_receipt,
)

# ---------------------------------------------------------------------------
# Frozen schemas and constants
# ---------------------------------------------------------------------------

REQUEST_SCHEMA_VERSION = "m3-empty-overlay-authorization-request-v4"
AUTHORIZATION_SCHEMA_VERSION = "m3-empty-overlay-execution-authorization-v4"
PANEL_RESULT_SCHEMA_VERSION = "m3-empty-overlay-panel-result-v4"
COMPLETION_RECEIPT_SCHEMA_VERSION = "m3-empty-overlay-completion-receipt-v4"
ANALYSIS_SCHEMA_VERSION = "m3-empty-overlay-analysis-v4"
CLAIM_SCHEMA_VERSION = "m3-empty-overlay-claim-v2"
DETACHED_LAUNCH_SCHEMA_VERSION = "m3-empty-overlay-detached-launch-v1"

AUTHORIZATION_STATEMENT = (
    "This artifact authorizes exactly one dedicated empty-overlay baseline "
    "execution of the frozen 36-task/72-panel manifest under the pinned "
    "runtime identity. It is a governance gate, not a cryptographic "
    "signature, and it authorizes live model execution exactly once."
)

SCREENING_NOTE = (
    "Screening-only descriptive analysis of the empty-overlay baseline. "
    "This is repeatability and floor/ceiling triage, not prompt-lift or "
    "causal evidence; it cannot justify a final treatment or policy "
    "selection."
)

# Exact worst-case budget, matching the frozen baseline treatment limits.
MAX_ATTEMPTS = EXPECTED_ATTEMPTS
OUTPUT_TOKENS_PER_INVOCATION = 4096
TOOL_CALLS_PER_INVOCATION = 12
WALL_SECONDS_PER_INVOCATION = 600
PROVIDER_BACKED_TURNS_PER_INVOCATION = TOOL_CALLS_PER_INVOCATION + 1
TOOL_ATTEMPTS_PER_INVOCATION = TOOL_CALLS_PER_INVOCATION + 1
PROVIDER_GATE_CHECKS_PER_INVOCATION = PROVIDER_BACKED_TURNS_PER_INVOCATION + 1

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@-]{0,127}$")
_DIFFICULTIES = ("easy", "medium", "hard")

# Verifier substrate failures are infrastructure errors, not behavioral
# outcomes: the harness substrate could not read the task/attempt.
_VERIFIER_SUBSTRATE_CODES = frozenset(
    {
        "task_not_found",
        "attempt_not_found",
        "attempt_task_mismatch",
        "oracle_unreadable",
        "oracle_missing_nonce",
        "oracle_commitment_mismatch",
    }
)

# Ordinary verifier failures are completed behavioral outcomes.
_ORDINARY_VERIFIER_FAILURE_CODES = frozenset(
    {
        "missing_output",
        "invalid_json",
        "wrong_type",
        "missing_key",
        "wrong_key_type",
        "nonce_mismatch",
    }
)

# Provider startup / transport failure markers (casefolded, matched on stderr).
_PROVIDER_STARTUP_TRANSPORT_MARKERS = (
    "connection refused",
    "connection reset",
    "connection timed out",
    "failed to connect",
    "no route to host",
    "transport error",
    "could not resolve host",
    "provider startup failed",
    "timed out waiting",
)

# Legacy browser infrastructure markers (casefolded, matched on stderr).
_BROWSER_INFRA_MARKERS = (
    "brokenpipeerror",
    "connectionreseterror",
    "browser process exited",
    "process connection broken",
    "response timed out",
    "result exceeds",
)

_AUTHORIZATION_FIELDS = frozenset(
    {
        "schema_version",
        "authorization_id",
        "screen_id",
        "manifest_hash",
        "registry_hash",
        "local_preflight_hash",
        "remote_preflight_hash",
        "source_tree_hash",
        "remote_identity",
        "result_filename",
        "result_path",
        "max_attempts",
        "budget",
        "approved_by",
        "approved_at",
        "expires_at",
        "authorization_statement",
        "live_model_execution_authorized",
        "single_use",
        "authorization_hash",
    }
)


def _worst_case_budget() -> dict[str, Any]:
    return {
        "model_attempts": MAX_ATTEMPTS,
        "provider_backed_turns_per_attempt": PROVIDER_BACKED_TURNS_PER_INVOCATION,
        "total_provider_backed_turns": (
            MAX_ATTEMPTS * PROVIDER_BACKED_TURNS_PER_INVOCATION
        ),
        "provider_gate_checks_per_attempt": PROVIDER_GATE_CHECKS_PER_INVOCATION,
        "total_provider_gate_checks": (
            MAX_ATTEMPTS * PROVIDER_GATE_CHECKS_PER_INVOCATION
        ),
        "output_tokens_per_provider_backed_turn": OUTPUT_TOKENS_PER_INVOCATION,
        "total_output_tokens": (
            MAX_ATTEMPTS
            * PROVIDER_BACKED_TURNS_PER_INVOCATION
            * OUTPUT_TOKENS_PER_INVOCATION
        ),
        "tool_attempts_per_attempt": TOOL_ATTEMPTS_PER_INVOCATION,
        "total_tool_attempts": MAX_ATTEMPTS * TOOL_ATTEMPTS_PER_INVOCATION,
        "budget_admitted_tool_attempts_per_attempt": TOOL_CALLS_PER_INVOCATION,
        "total_budget_admitted_tool_attempts": (
            MAX_ATTEMPTS * TOOL_CALLS_PER_INVOCATION
        ),
        "model_wall_seconds_per_attempt": WALL_SECONDS_PER_INVOCATION,
        "total_wall_seconds": MAX_ATTEMPTS * WALL_SECONDS_PER_INVOCATION,
    }


def _reserved_budget() -> dict[str, int]:
    return {
        "model_attempts": 1,
        "provider_backed_turns": PROVIDER_BACKED_TURNS_PER_INVOCATION,
        "provider_gate_checks": PROVIDER_GATE_CHECKS_PER_INVOCATION,
        "output_tokens": (
            PROVIDER_BACKED_TURNS_PER_INVOCATION * OUTPUT_TOKENS_PER_INVOCATION
        ),
        "tool_attempts": TOOL_ATTEMPTS_PER_INVOCATION,
        "budget_admitted_tool_attempts": TOOL_CALLS_PER_INVOCATION,
        "model_wall_seconds": WALL_SECONDS_PER_INVOCATION,
    }


def _remaining_budget(attempts_remaining: int) -> dict[str, int]:
    if attempts_remaining < 0 or attempts_remaining > MAX_ATTEMPTS:
        raise ValueError("remaining attempt count is outside the authorization")
    per_attempt = _reserved_budget()
    return {
        key: value * attempts_remaining for key, value in per_attempt.items()
    }


def _budget_reservation(panel_index: int) -> dict[str, Any]:
    if panel_index < 0 or panel_index >= MAX_ATTEMPTS:
        raise ValueError("panel index is outside the authorization")
    return {
        "reserved_capacity_before": _remaining_budget(
            MAX_ATTEMPTS - panel_index
        ),
        "reserved_for_panel": _reserved_budget(),
        "reserved_capacity_after": _remaining_budget(
            MAX_ATTEMPTS - panel_index - 1
        ),
    }


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _is_hex_digest(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_tz_aware(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a timezone-aware ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{label} must be a valid ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return parsed


def _require_authorization_active(expires_at: datetime) -> None:
    """Check expiry immediately before admitting another model attempt.

    An attempt admitted before expiry may finish after expiry; no subsequent
    panel may start.
    """
    if datetime.now(timezone.utc) >= expires_at:
        raise RuntimeError("execution authorization expired before model admission")


def _validate_result_filename(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("result filename must be a non-empty string")
    name = value.strip()
    if "/" in name or "\\" in name or "\x00" in name:
        raise ValueError("result filename must be a bare filename without separators")
    return name


def _record_binds(
    *,
    authorization_hash: str,
    manifest_hash: str,
    registry_hash: str,
    local_preflight_hash: str,
    remote_preflight_hash: str,
    source_tree_hash: str,
) -> dict[str, str]:
    return {
        "authorization_hash": authorization_hash,
        "manifest_hash": manifest_hash,
        "registry_hash": registry_hash,
        "local_preflight_hash": local_preflight_hash,
        "remote_preflight_hash": remote_preflight_hash,
        "source_tree_hash": source_tree_hash,
    }


def _sibling_paths(output: Path) -> dict[str, Path]:
    return {
        "lock": output.with_name(output.name + ".lock"),
        "launch_lock": output.with_name(output.name + ".launch.lock"),
        "launch": output.with_name(output.name + ".launch.json"),
        "controller_log": output.with_name(output.name + ".controller.log"),
        "claim": output.with_name(output.name + ".claim.json"),
        "active": output.with_name(output.name + ".active.json"),
        "receipt": output.with_name(output.name + ".receipt.json"),
        "runtime_preflight": output.with_name(output.name + ".runtime-preflight.json"),
    }


def deterministic_attempt_id(authorization_hash: str, panel_id: str) -> str:
    """Deterministic, safe attempt id bound to the authorization and panel."""
    digest = hashlib.sha256(
        f"{authorization_hash}:{panel_id}".encode("utf-8")
    ).hexdigest()
    return f"eob-{digest[:16]}"


def _append_record(output: Path, record: dict[str, Any]) -> None:
    """Self-hash and durably append exactly one JSONL record."""
    payload = {key: value for key, value in record.items() if key != "record_hash"}
    record["record_hash"] = _canonical_hash(payload)
    _append_result(output, record)


# ---------------------------------------------------------------------------
# authorization request (non-authorizing)
# ---------------------------------------------------------------------------


def build_authorization_request(
    manifest: Mapping[str, Any],
    registry: TreatmentRegistry,
    local_preflight: Mapping[str, Any],
    remote_preflight: Mapping[str, Any],
    *,
    project_root: str | Path,
    result_path: str | Path,
    pi_executable: str = "pi",
) -> dict[str, Any]:
    """Build a non-authorizing request bound to every frozen artifact.

    ``live_model_execution_authorized`` is hard-coded to ``False`` and there is
    no function in this module that can promote a request into a valid
    authorization.
    """
    project = Path(project_root).expanduser().resolve()
    validate_baseline_manifest(manifest, registry)
    validate_local_preflight(
        local_preflight, manifest, registry, project, pi_executable=pi_executable
    )
    validate_remote_preflight(
        remote_preflight,
        manifest,
        registry,
        local_preflight,
        project,
        pi_executable=pi_executable,
    )
    source = source_tree_hash(project)
    if source != local_preflight.get("source_tree_hash"):
        raise RuntimeError("source tree drifted from the frozen local preflight")
    result = Path(result_path).expanduser().resolve()
    filename = _validate_result_filename(result.name)

    payload: dict[str, Any] = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "screen_id": manifest["screen_id"],
        "purpose": manifest["purpose"],
        "manifest_hash": manifest["manifest_hash"],
        "registry_hash": registry.registry_hash,
        "local_preflight_hash": local_preflight["preflight_hash"],
        "remote_preflight_hash": remote_preflight["preflight_hash"],
        "source_tree_hash": source,
        "remote_identity": dict(manifest["remote_identity"]),
        "result_filename": filename,
        "result_path": str(result),
        "max_attempts": MAX_ATTEMPTS,
        "budget": _worst_case_budget(),
        "required_authorization_fields": [
            "schema_version",
            "authorization_id",
            "screen_id",
            "manifest_hash",
            "registry_hash",
            "local_preflight_hash",
            "remote_preflight_hash",
            "source_tree_hash",
            "remote_identity",
            "result_filename",
            "result_path",
            "max_attempts",
            "budget",
            "approved_by",
            "approved_at",
            "expires_at",
            "authorization_statement",
            "live_model_execution_authorized",
            "single_use",
            "authorization_hash",
        ],
        "live_model_execution_authorized": False,
        "authorization_boundary": (
            "This request is non-authorizing. Live execution requires a "
            "separately authored authorization artifact whose "
            "live_model_execution_authorized field is exactly true, whose "
            "single_use field is exactly true, and whose authorization_hash "
            "matches the operator-supplied expected hash."
        ),
    }
    return {**payload, "request_hash": _canonical_hash(payload)}


# ---------------------------------------------------------------------------
# execution authorization validation (governance gate)
# ---------------------------------------------------------------------------


def validate_execution_authorization(
    authorization: Mapping[str, Any],
    *,
    expected_authorization_hash: str,
    manifest_hash: str,
    registry_hash: str,
    local_preflight_hash: str,
    remote_preflight_hash: str,
    source_tree_hash: str,
    remote_identity: Mapping[str, Any],
    result_filename: str,
    result_path: str | Path,
) -> str:
    """Validate a separately authored execution authorization.

    This is a governance gate (exact field equality against frozen artifacts and
    an operator-supplied expected hash), not a cryptographic signature.
    """
    _verify_embedded_hash(authorization, "authorization_hash")

    if set(authorization) != _AUTHORIZATION_FIELDS:
        missing = sorted(_AUTHORIZATION_FIELDS - set(authorization))
        extra = sorted(set(authorization) - _AUTHORIZATION_FIELDS)
        raise ValueError(
            f"authorization fields mismatch: missing={missing!r}, extra={extra!r}"
        )

    if authorization.get("schema_version") != AUTHORIZATION_SCHEMA_VERSION:
        raise ValueError("unsupported authorization schema")
    authorization_id = authorization.get("authorization_id")
    if not isinstance(authorization_id, str) or not _SAFE_ID.fullmatch(authorization_id):
        raise ValueError("authorization_id must be a unique safe identifier")
    if authorization.get("screen_id") != SCREEN_ID:
        raise ValueError("authorization screen_id mismatch")
    if authorization.get("manifest_hash") != manifest_hash:
        raise ValueError("authorization manifest hash mismatch")
    if authorization.get("registry_hash") != registry_hash:
        raise ValueError("authorization registry hash mismatch")
    if authorization.get("local_preflight_hash") != local_preflight_hash:
        raise ValueError("authorization local preflight hash mismatch")
    if authorization.get("remote_preflight_hash") != remote_preflight_hash:
        raise ValueError("authorization remote preflight hash mismatch")
    if authorization.get("source_tree_hash") != source_tree_hash:
        raise ValueError("authorization source tree hash mismatch (source drift)")
    remote = authorization.get("remote_identity")
    if not isinstance(remote, Mapping) or dict(remote) != dict(remote_identity):
        raise ValueError("authorization remote identity mismatch")
    if authorization.get("result_filename") != result_filename:
        raise ValueError("authorization result filename mismatch")
    if authorization.get("result_path") != str(
        Path(result_path).expanduser().resolve()
    ):
        raise ValueError("authorization result path mismatch")
    if authorization.get("max_attempts") != MAX_ATTEMPTS:
        raise ValueError("authorization max_attempts mismatch")
    if authorization.get("budget") != _worst_case_budget():
        raise ValueError("authorization budget mismatch")

    approved_by = authorization.get("approved_by")
    if not isinstance(approved_by, str) or not approved_by.strip():
        raise ValueError("authorization approved_by must be non-empty")
    approved_at = _parse_tz_aware(authorization.get("approved_at"), "approved_at")
    expires_at = _parse_tz_aware(authorization.get("expires_at"), "expires_at")
    if expires_at <= approved_at:
        raise ValueError("authorization expires_at must be after approved_at")
    now = datetime.now(timezone.utc)
    if approved_at > now:
        raise ValueError("authorization approved_at cannot be in the future")
    if expires_at <= now:
        raise ValueError("authorization has expired")

    if authorization.get("authorization_statement") != AUTHORIZATION_STATEMENT:
        raise ValueError("authorization statement mismatch")
    if authorization.get("live_model_execution_authorized") is not True:
        raise ValueError("authorization does not enable live model execution")
    if authorization.get("single_use") is not True:
        raise ValueError("authorization single_use must be exactly true")

    computed = str(authorization["authorization_hash"])
    if computed != expected_authorization_hash:
        raise ValueError(
            "authorization hash does not match the operator-supplied expected hash"
        )
    return computed


# ---------------------------------------------------------------------------
# claim / lock / active marker
# ---------------------------------------------------------------------------


def _write_claim(
    claim_path: Path,
    authorization_hash: str,
    output: Path,
    result_filename: str,
    *,
    controller_pid: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": CLAIM_SCHEMA_VERSION,
        "authorization_hash": authorization_hash,
        "result_path": str(output),
        "result_filename": result_filename,
        "controller_pid": controller_pid or os.getpid(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    claim = {**payload, "claim_hash": _canonical_hash(payload)}
    _write_immutable_json(claim_path, claim)
    return claim


def _validate_claim(
    claim: Mapping[str, Any],
    authorization_hash: str,
    output: Path,
    result_filename: str,
) -> None:
    _verify_embedded_hash(claim, "claim_hash")
    if claim.get("schema_version") != CLAIM_SCHEMA_VERSION:
        raise ValueError("unsupported claim schema")
    if claim.get("authorization_hash") != authorization_hash:
        raise ValueError("claim authorization hash mismatch")
    if claim.get("result_path") != str(output):
        raise ValueError("claim result path mismatch")
    if claim.get("result_filename") != result_filename:
        raise ValueError("claim result filename mismatch")
    controller_pid = claim.get("controller_pid")
    if not isinstance(controller_pid, int) or isinstance(controller_pid, bool):
        raise ValueError("claim controller_pid must be an integer")
    if controller_pid <= 0:
        raise ValueError("claim controller_pid must be positive")


def _prepare_claim(
    claim_path: Path, output: Path, authorization_hash: str, result_filename: str
) -> dict[str, Any]:
    if claim_path.exists():
        claim = _load_json(claim_path)
        _validate_claim(claim, authorization_hash, output, result_filename)
        if not output.exists():
            raise RuntimeError(
                "existing baseline claim has no ledger; adjudication required"
            )
        return claim
    if output.exists():
        raise RuntimeError("ledger exists without a claim; adjudication required")
    return _write_claim(claim_path, authorization_hash, output, result_filename)


def _acquire_lock(lock_path: Path) -> None:
    try:
        descriptor = os.open(
            str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
        )
    except FileExistsError as error:
        raise RuntimeError(
            "baseline execution lock is already held; concurrent runs are forbidden"
        ) from error
    try:
        os.write(descriptor, b"locked\n")
    finally:
        os.close(descriptor)


def _write_active_marker(
    active_path: Path,
    *,
    authorization_hash: str,
    panel_id: str,
    attempt_id: str,
    started_at: str,
    budget_reserved: Mapping[str, Any],
) -> None:
    _write_immutable_json(
        active_path,
        {
            "authorization_hash": authorization_hash,
            "panel_id": panel_id,
            "attempt_id": attempt_id,
            "started_at": started_at,
            "budget_reserved": dict(budget_reserved),
        },
    )


# ---------------------------------------------------------------------------
# attempt-level classification and structural validation
# ---------------------------------------------------------------------------


def _tool_attempt_accounting(
    trace: Any, budget_receipt: Any = None
) -> dict[str, int]:
    entries = [entry for entry in (trace or []) if isinstance(entry, Mapping)]
    blocked = [
        entry
        for entry in entries
        if entry.get("budget_rejected") is True
        or entry.get("operation_aborted") is True
    ]
    derived_executed = [
        entry
        for entry in entries
        if entry.get("budget_rejected") is not True
        and entry.get("operation_aborted") is not True
        and entry.get("pre_execution_rejected") is not True
    ]
    if isinstance(budget_receipt, Mapping):
        admitted = budget_receipt.get("admitted_tool_call_count")
        executed = budget_receipt.get("executed_tool_call_count")
        pre_admission_rejected = budget_receipt.get(
            "pre_admission_rejected_tool_call_count"
        )
        suppressed = budget_receipt.get("suppressed_tool_request_count")
    else:
        admitted = len(derived_executed)
        executed = len(derived_executed)
        pre_admission_rejected = len(entries) - len(derived_executed)
        suppressed = 0
    return {
        "tool_attempts": len(entries),
        "budget_admitted_tool_attempts": admitted,
        "executed_tool_calls": executed,
        "rejected_tool_attempts": pre_admission_rejected,
        "budget_blocked_tool_attempts": len(blocked),
        "suppressed_tool_requests": suppressed,
    }


def _attempt_budget_consumption(item: Mapping[str, Any]) -> dict[str, Any]:
    usage = item.get("usage") if isinstance(item.get("usage"), Mapping) else {}
    trajectory = (
        item.get("trajectory") if isinstance(item.get("trajectory"), Mapping) else {}
    )
    timing = item.get("timing") if isinstance(item.get("timing"), Mapping) else {}
    trace = trajectory.get("tool_trace")
    budget_receipt = trajectory.get("budget_receipt")
    tool_accounting = _tool_attempt_accounting(trace, budget_receipt)
    return {
        "model_attempts": 1,
        "provider_backed_turns": trajectory.get("provider_turn_count"),
        "provider_gate_checks": (
            budget_receipt.get("provider_gate_checks")
            if isinstance(budget_receipt, Mapping)
            else None
        ),
        "provider_gate_blocks": (
            budget_receipt.get("provider_request_blocks")
            if isinstance(budget_receipt, Mapping)
            else None
        ),
        "output_tokens": usage.get("output"),
        **tool_accounting,
        "model_wall_seconds": timing.get("pi_seconds"),
    }


_BUDGET_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "provider_turn_limit",
        "provider_request_admissions",
        "provider_request_blocks",
        "provider_gate_checks",
        "tool_attempt_limit",
        "tool_attempt_count",
        "tool_attempt_ids",
        "tool_admission_limit",
        "admitted_tool_call_count",
        "admitted_tool_call_ids",
        "executed_tool_call_count",
        "executed_tool_call_ids",
        "pre_admission_rejected_tool_call_count",
        "pre_admission_rejected_tool_call_ids",
        "suppressed_tool_request_count",
        "suppressed_tool_request_ids",
        "invariant_violations",
    }
)


def _budget_receipt_errors(
    receipt: Any,
    tool_trace: Any,
    provider_turns: Any,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(receipt, Mapping):
        return ["trajectory.budget_receipt missing"]
    if set(receipt) != _BUDGET_RECEIPT_FIELDS:
        errors.append("trajectory.budget_receipt fields mismatch")
    if receipt.get("schema_version") != BUDGET_RECEIPT_SCHEMA_VERSION:
        errors.append("trajectory.budget_receipt schema mismatch")

    count_keys = (
        "provider_turn_limit",
        "provider_request_admissions",
        "provider_request_blocks",
        "provider_gate_checks",
        "tool_attempt_limit",
        "tool_attempt_count",
        "tool_admission_limit",
        "admitted_tool_call_count",
        "executed_tool_call_count",
        "pre_admission_rejected_tool_call_count",
        "suppressed_tool_request_count",
    )
    counts: dict[str, int] = {}
    for key in count_keys:
        value = receipt.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            errors.append(f"trajectory.budget_receipt {key} invalid")
        else:
            counts[key] = value

    id_keys = (
        "tool_attempt_ids",
        "admitted_tool_call_ids",
        "executed_tool_call_ids",
        "pre_admission_rejected_tool_call_ids",
        "suppressed_tool_request_ids",
    )
    id_lists: dict[str, list[str]] = {}
    for key in id_keys:
        value = receipt.get(key)
        if (
            not isinstance(value, list)
            or any(not isinstance(item, str) or not item for item in value)
            or len(set(value)) != len(value)
        ):
            errors.append(f"trajectory.budget_receipt {key} invalid")
        else:
            id_lists[key] = value

    violations = receipt.get("invariant_violations")
    if violations != []:
        errors.append("trajectory.budget_receipt invariant violation")

    if counts.get("provider_turn_limit") != PROVIDER_BACKED_TURNS_PER_INVOCATION:
        errors.append("trajectory.budget_receipt provider turn limit mismatch")
    admissions = counts.get("provider_request_admissions")
    blocks = counts.get("provider_request_blocks")
    checks = counts.get("provider_gate_checks")
    if admissions is not None and not (1 <= admissions <= PROVIDER_BACKED_TURNS_PER_INVOCATION):
        errors.append("trajectory.budget_receipt provider admissions out of bounds")
    if blocks is not None and blocks not in (0, 1):
        errors.append("trajectory.budget_receipt provider blocks out of bounds")
    if (
        admissions is not None
        and blocks is not None
        and checks != admissions + blocks
    ):
        errors.append("trajectory.budget_receipt provider gate accounting mismatch")
    if checks is not None and checks > PROVIDER_GATE_CHECKS_PER_INVOCATION:
        errors.append("trajectory.budget_receipt provider gate checks exceed bound")
    if admissions is not None and provider_turns != admissions:
        errors.append("trajectory provider turns do not match budget receipt")

    if counts.get("tool_attempt_limit") != TOOL_ATTEMPTS_PER_INVOCATION:
        errors.append("trajectory.budget_receipt tool attempt limit mismatch")
    if counts.get("tool_admission_limit") != TOOL_CALLS_PER_INVOCATION:
        errors.append("trajectory.budget_receipt tool admission limit mismatch")
    attempts = counts.get("tool_attempt_count")
    admitted = counts.get("admitted_tool_call_count")
    executed = counts.get("executed_tool_call_count")
    rejected = counts.get("pre_admission_rejected_tool_call_count")
    if attempts is not None and attempts > TOOL_ATTEMPTS_PER_INVOCATION:
        errors.append("trajectory.budget_receipt tool attempts exceed bound")
    if admitted is not None and admitted > TOOL_CALLS_PER_INVOCATION:
        errors.append("trajectory.budget_receipt tool admissions exceed bound")
    if executed is not None and admitted is not None and executed > admitted:
        errors.append("trajectory.budget_receipt executions exceed admissions")
    if attempts is not None and admitted is not None and rejected != attempts - admitted:
        errors.append("trajectory.budget_receipt rejection accounting mismatch")

    list_count_pairs = (
        ("tool_attempt_ids", "tool_attempt_count"),
        ("admitted_tool_call_ids", "admitted_tool_call_count"),
        ("executed_tool_call_ids", "executed_tool_call_count"),
        (
            "pre_admission_rejected_tool_call_ids",
            "pre_admission_rejected_tool_call_count",
        ),
        ("suppressed_tool_request_ids", "suppressed_tool_request_count"),
    )
    for list_key, count_key in list_count_pairs:
        if list_key in id_lists and count_key in counts:
            if len(id_lists[list_key]) != counts[count_key]:
                errors.append(
                    f"trajectory.budget_receipt {list_key} count mismatch"
                )

    attempt_ids = id_lists.get("tool_attempt_ids", [])
    admitted_ids = id_lists.get("admitted_tool_call_ids", [])
    executed_ids = id_lists.get("executed_tool_call_ids", [])
    rejected_ids = id_lists.get("pre_admission_rejected_tool_call_ids", [])
    suppressed_ids = id_lists.get("suppressed_tool_request_ids", [])
    if any(item not in attempt_ids for item in admitted_ids):
        errors.append("trajectory.budget_receipt admission without attempt")
    if any(item not in admitted_ids for item in executed_ids):
        errors.append("trajectory.budget_receipt execution without admission")
    if rejected_ids != [item for item in attempt_ids if item not in set(admitted_ids)]:
        errors.append("trajectory.budget_receipt rejected id partition mismatch")
    if set(suppressed_ids) & set(attempt_ids):
        errors.append("trajectory.budget_receipt suppressed request was attempted")

    if isinstance(tool_trace, list) and all(
        isinstance(entry, Mapping) for entry in tool_trace
    ):
        trace_ids = [entry.get("tool_call_id") for entry in tool_trace]
        if len(trace_ids) != len(attempt_ids) or set(trace_ids) != set(attempt_ids):
            errors.append("trajectory tool trace does not match budget receipt")
        derived_executed = [
            entry.get("tool_call_id")
            for entry in tool_trace
            if entry.get("budget_rejected") is not True
            and entry.get("operation_aborted") is not True
            and entry.get("pre_execution_rejected") is not True
        ]
        if derived_executed != executed_ids:
            errors.append("trajectory executed tools do not match budget receipt")
    return errors


def _attempt_structural_errors(
    item: Mapping[str, Any],
    *,
    expected_policy: Mapping[str, Any],
    expected_sampling_receipt: Mapping[str, Any],
    runtime_pins: Mapping[str, Any],
) -> list[str]:
    """Return structural problems in one attempt item (empty == valid)."""
    errors: list[str] = []
    if item.get("policy") != expected_policy:
        errors.append("empty policy receipt mismatch")
    policy = item.get("policy")
    if isinstance(policy, Mapping) and policy.get("system_prompt") != "":
        errors.append("empty policy receipt has a non-empty system prompt")

    rc = item.get("pi_return_code")
    if isinstance(rc, bool) or not isinstance(rc, int) or rc not in (0, -1):
        errors.append(f"pi_return_code invalid: {rc!r}")

    verification = item.get("verification")
    if not isinstance(verification, Mapping) or not isinstance(
        verification.get("success"), bool
    ):
        errors.append("verification missing or malformed")
    else:
        if verification.get("verifier_id") != runtime_pins.get("fixture_verifier_id"):
            errors.append("verifier identity mismatch")
        if verification.get("verifier_version") != runtime_pins.get(
            "fixture_verifier_version"
        ):
            errors.append("verifier version mismatch")

    usage = item.get("usage")
    if not isinstance(usage, Mapping):
        errors.append("usage missing")
    else:
        output = usage.get("output")
        if (
            output is None
            or isinstance(output, bool)
            or not isinstance(output, (int, float))
            or not math.isfinite(float(output))
            or float(output) < 0
        ):
            errors.append("usage.output invalid")

    trajectory = item.get("trajectory")
    if not isinstance(trajectory, Mapping):
        errors.append("trajectory missing")
    else:
        if (
            trajectory.get("normalizer_schema_version")
            != NORMALIZED_EVENT_SCHEMA_VERSION
        ):
            errors.append("trajectory normalizer schema mismatch")
        if trajectory.get("provider_turn_semantics") != PROVIDER_TURN_SEMANTICS:
            errors.append("trajectory provider-turn semantics mismatch")
        if not isinstance(trajectory.get("planning_preamble"), Mapping):
            errors.append("trajectory.planning_preamble missing")
        tool_trace = trajectory.get("tool_trace")
        if not isinstance(tool_trace, list):
            errors.append("trajectory.tool_trace missing")
        else:
            for index, entry in enumerate(tool_trace):
                if (
                    not isinstance(entry, Mapping)
                    or not isinstance(entry.get("tool_call_id"), str)
                    or not entry.get("tool_call_id")
                    or not isinstance(entry.get("tool_name"), str)
                    or not isinstance(entry.get("is_error"), bool)
                    or not isinstance(entry.get("budget_rejected"), bool)
                    or not isinstance(entry.get("operation_aborted"), bool)
                    or not isinstance(entry.get("pre_execution_rejected"), bool)
                    or not isinstance(entry.get("details"), Mapping)
                ):
                    errors.append(f"trajectory.tool_trace[{index}] malformed")
        provider_turns = trajectory.get("provider_turn_count")
        assistant_messages = trajectory.get("assistant_message_count")
        synthetic_messages = trajectory.get("synthetic_assistant_message_count")
        if (
            isinstance(assistant_messages, bool)
            or not isinstance(assistant_messages, int)
            or assistant_messages < 0
        ):
            errors.append("trajectory.assistant_message_count invalid")
        if (
            isinstance(synthetic_messages, bool)
            or not isinstance(synthetic_messages, int)
            or synthetic_messages < 0
            or synthetic_messages > 1
        ):
            errors.append("trajectory.synthetic_assistant_message_count invalid")
        if (
            isinstance(provider_turns, bool)
            or not isinstance(provider_turns, int)
            or provider_turns < 0
        ):
            errors.append("trajectory.provider_turn_count invalid")
        elif provider_turns < 1:
            errors.append("trajectory.provider_turn_count must be positive")
        elif provider_turns > PROVIDER_BACKED_TURNS_PER_INVOCATION:
            errors.append("trajectory.provider_turn_count exceeds the frozen bound")
        if (
            isinstance(assistant_messages, int)
            and not isinstance(assistant_messages, bool)
            and isinstance(provider_turns, int)
            and not isinstance(provider_turns, bool)
            and isinstance(synthetic_messages, int)
            and not isinstance(synthetic_messages, bool)
            and assistant_messages != provider_turns + synthetic_messages
        ):
            errors.append("trajectory assistant-message accounting mismatch")
        if item.get("sampling_receipt") != expected_sampling_receipt:
            errors.append("sampling receipt mismatch")

        tool_calls = trajectory.get("tool_call_count")
        if (
            isinstance(tool_calls, bool)
            or not isinstance(tool_calls, int)
            or tool_calls < 0
            or tool_calls > TOOL_ATTEMPTS_PER_INVOCATION
        ):
            errors.append("trajectory.tool_call_count invalid")
        elif isinstance(tool_trace, list) and tool_calls != len(tool_trace):
            errors.append("trajectory.tool_call_count does not match tool_trace")
        if isinstance(tool_trace, list) and all(
            isinstance(entry, Mapping) for entry in tool_trace
        ):
            budget_receipt = trajectory.get("budget_receipt")
            accounting = _tool_attempt_accounting(tool_trace, budget_receipt)
            if (
                accounting["budget_admitted_tool_attempts"]
                > TOOL_CALLS_PER_INVOCATION
            ):
                errors.append(
                    "trajectory budget-admitted tool attempts exceed the frozen bound"
                )
            blocked_indices = [
                index
                for index, entry in enumerate(tool_trace)
                if entry.get("budget_rejected") is True
                or entry.get("operation_aborted") is True
            ]
            suppressed_requests = (
                budget_receipt.get("suppressed_tool_request_count")
                if isinstance(budget_receipt, Mapping)
                else None
            )
            if blocked_indices and (
                len(blocked_indices) != 1
                or blocked_indices[0] != len(tool_trace) - 1
                or (
                    accounting["budget_admitted_tool_attempts"]
                    != TOOL_CALLS_PER_INVOCATION
                    and not (
                        isinstance(suppressed_requests, int)
                        and not isinstance(suppressed_requests, bool)
                        and suppressed_requests > 0
                    )
                )
            ):
                errors.append("trajectory budget-blocked tool attempt is not terminal")
            provider_blocks = (
                budget_receipt.get("provider_request_blocks")
                if isinstance(budget_receipt, Mapping)
                else None
            )
            if (
                isinstance(synthetic_messages, int)
                and not isinstance(synthetic_messages, bool)
                and isinstance(provider_blocks, int)
                and not isinstance(provider_blocks, bool)
                and synthetic_messages
                != accounting["budget_blocked_tool_attempts"] + provider_blocks
            ):
                errors.append(
                    "trajectory synthetic assistant and local-block counts mismatch"
                )
            errors.extend(
                _budget_receipt_errors(
                    budget_receipt,
                    tool_trace,
                    provider_turns,
                )
            )

    timing = item.get("timing")
    if not isinstance(timing, Mapping):
        errors.append("timing missing")
    else:
        for key in (
            "prepare_seconds",
            "pi_seconds",
            "record_seconds",
            "verify_seconds",
            "usage_seconds",
            "total_seconds",
        ):
            value = timing.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                errors.append(f"timing.{key} invalid")
    if isinstance(usage, Mapping) and isinstance(trajectory, Mapping):
        output = usage.get("output")
        provider_turns = trajectory.get("provider_turn_count")
        if (
            isinstance(output, (int, float))
            and not isinstance(output, bool)
            and isinstance(provider_turns, int)
            and not isinstance(provider_turns, bool)
            and provider_turns > 0
            and float(output) > provider_turns * OUTPUT_TOKENS_PER_INVOCATION
        ):
            errors.append("usage.output exceeds the provider-request token bound")
    return errors


def _classify_attempt(
    item: Mapping[str, Any],
    *,
    expected_policy: Mapping[str, Any],
    expected_sampling_receipt: Mapping[str, Any],
    runtime_pins: Mapping[str, Any],
) -> tuple[str, str | None]:
    """Classify one completed :func:`_run_attempt` item.

    Returns ``(status, reason)`` where status is either ``"completed"`` or
    ``"infrastructure_invalid"``. ``reason`` is ``None`` for a completed
    behavioral outcome.
    """
    rc = item.get("pi_return_code")
    if isinstance(rc, bool) or not isinstance(rc, int):
        return "infrastructure_invalid", "pi_return_code_not_int"
    if rc not in (0, -1):
        return "infrastructure_invalid", f"pi_return_code={rc}"
    if rc == -1:
        return "infrastructure_invalid", "ambiguous_wall_timeout"

    verification = item.get("verification")
    failure_code = (
        verification.get("failure_code")
        if isinstance(verification, Mapping)
        else None
    )
    if failure_code in _VERIFIER_SUBSTRATE_CODES:
        return "infrastructure_invalid", f"verifier_substrate={failure_code}"
    if failure_code is not None and failure_code not in _ORDINARY_VERIFIER_FAILURE_CODES:
        return "infrastructure_invalid", f"verifier_unknown_code={failure_code}"

    trajectory = item.get("trajectory")
    if isinstance(trajectory, Mapping):
        for entry in trajectory.get("tool_trace") or []:
            if isinstance(entry, Mapping) and _is_infrastructure_error(entry):
                return "infrastructure_invalid", "browser_infrastructure_marker"

    stderr = str(item.get("pi_stderr") or "").casefold()
    for marker in _PROVIDER_STARTUP_TRANSPORT_MARKERS:
        if marker in stderr:
            return "infrastructure_invalid", f"provider_transport={marker}"
    for marker in _BROWSER_INFRA_MARKERS:
        if marker in stderr:
            return "infrastructure_invalid", f"browser_legacy={marker}"

    structural = _attempt_structural_errors(
        item,
        expected_policy=expected_policy,
        expected_sampling_receipt=expected_sampling_receipt,
        runtime_pins=runtime_pins,
    )
    if structural:
        return (
            "infrastructure_invalid",
            f"malformed_normalized_events={'|'.join(structural)}",
        )

    return "completed", None


# ---------------------------------------------------------------------------
# one-policy result shape (equivalent to orchestrator treatment_set output)
# ---------------------------------------------------------------------------


def _one_policy_result(
    task: Mapping[str, Any],
    attempt: Mapping[str, Any],
    bundle_id: str,
    registry_hash: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "task_id": task["id"],
        "mode": "treatment_set",
        "execution_order": [bundle_id],
        "attempts": {
            bundle_id: {
                "attempt_id": attempt["attempt_id"],
                "policy": attempt["policy"],
                "pi_return_code": attempt["pi_return_code"],
                "pi_stderr": attempt["pi_stderr"],
                "sampling_receipt": attempt.get("sampling_receipt"),
                "verification": attempt["verification"],
                "usage": attempt.get("usage"),
                "trajectory": attempt.get("trajectory"),
                "timing": attempt.get("timing"),
            }
        },
        "treatment_registry_hash": registry_hash,
        "rollout_replica": args.rollout_replica,
        "sampling_seed": args.sampling_seed,
        "pilot_manifest_hash": args.pilot_manifest_hash,
        "pilot_panel_id": args.pilot_panel_id,
    }


def _verify_generated_task(
    generated_task: Any,
    expected_task: Mapping[str, Any],
    expected_commitment: Mapping[str, Any],
) -> None:
    if not isinstance(generated_task, Mapping):
        raise ValueError("task generation returned a non-object")
    if generated_task.get("id") != expected_task["task_id"]:
        raise ValueError(
            f"generated task id drifted: {generated_task.get('id')!r}"
        )
    if generated_task.get("generator_version") != OUTCOME_ONLY_GENERATOR_VERSION:
        raise ValueError("generated task generator version drifted")
    if generated_task.get("family") != "unbrowser_fixture":
        raise ValueError("generated task family drifted")
    if generated_task.get("template_id") != expected_task["template"]:
        raise ValueError("generated task template drifted")
    if generated_task.get("seed") != expected_task["seed"]:
        raise ValueError("generated task seed drifted")
    if generated_task.get("difficulty") != expected_task["difficulty"]:
        raise ValueError("generated task difficulty drifted")
    metadata = generated_task.get("public_metadata")
    if not isinstance(metadata, Mapping) or metadata.get("prompt_profile") != "outcome_only_v1":
        raise ValueError("generated task omitted the outcome-only prompt profile")
    if task_content_receipt(generated_task) != expected_commitment.get("task"):
        raise ValueError("generated task content drifted from the local commitment")


# ---------------------------------------------------------------------------
# strict JSONL record validation (shared by resume and analysis)
# ---------------------------------------------------------------------------


def _validate_completed_result(
    result: Any,
    *,
    panel: Mapping[str, Any],
    task: Mapping[str, Any],
    panel_id: str,
    attempt_id: str,
    bundle_id: str,
    expected_policy: Mapping[str, Any],
    runtime_pins: Mapping[str, Any],
    registry_hash: str,
    manifest_hash: str,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(result, Mapping):
        errors.append("result must be an object")
        return errors
    if result.get("mode") != "treatment_set":
        errors.append("result.mode must be treatment_set")
    if result.get("task_id") != task["task_id"]:
        errors.append("result.task_id mismatch")
    if result.get("execution_order") != [bundle_id]:
        errors.append("result.execution_order mismatch")
    if result.get("treatment_registry_hash") != registry_hash:
        errors.append("result.treatment_registry_hash mismatch")
    if result.get("rollout_replica") != panel["rollout_replica"]:
        errors.append("result.rollout_replica mismatch")
    if result.get("sampling_seed") != panel["sampling_seed"]:
        errors.append("result.sampling_seed mismatch")
    if result.get("pilot_manifest_hash") != manifest_hash:
        errors.append("result.pilot_manifest_hash mismatch")
    if result.get("pilot_panel_id") != panel_id:
        errors.append("result.pilot_panel_id mismatch")

    attempts = result.get("attempts")
    if not isinstance(attempts, Mapping) or set(attempts) != {bundle_id}:
        errors.append("result.attempts must contain exactly the baseline bundle")
        return errors
    item = attempts[bundle_id]
    if not isinstance(item, Mapping):
        errors.append("attempt item must be an object")
        return errors
    if item.get("attempt_id") != attempt_id:
        errors.append("attempt_id mismatch")
    expected_sampling_receipt = {
        "seed": int(panel["sampling_seed"]),
        "parameters": runtime_pins["sampling"]["parameters"],
    }
    errors.extend(
        _attempt_structural_errors(
            item,
            expected_policy=expected_policy,
            expected_sampling_receipt=expected_sampling_receipt,
            runtime_pins=runtime_pins,
        )
    )
    return errors


def _validate_record(
    record: Mapping[str, Any],
    *,
    binds: Mapping[str, str],
    manifest: Mapping[str, Any],
    registry_hash: str,
) -> list[str]:
    """Validate one panel record's integrity and binding (empty == valid)."""
    errors: list[str] = []
    try:
        _verify_embedded_hash(record, "record_hash")
    except ValueError as error:
        return [f"record_hash: {error}"]

    if record.get("schema_version") != PANEL_RESULT_SCHEMA_VERSION:
        errors.append("unknown record schema version")
        return errors

    for key, expected in binds.items():
        if record.get(key) != expected:
            errors.append(f"record.{key} mismatch")

    panel_by_id = {str(p["panel_id"]): p for p in manifest["panels"]}
    task_by_id = {str(t["task_id"]): t for t in manifest["tasks"]}
    panel_index_by_id = {
        str(p["panel_id"]): index for index, p in enumerate(manifest["panels"])
    }

    pid = str(record.get("panel_id", ""))
    panel = panel_by_id.get(pid)
    if panel is None:
        errors.append(f"unknown panel_id {pid!r}")
        return errors
    if record.get("panel_index") != panel_index_by_id[pid]:
        errors.append("record.panel_index mismatch")
    if record.get("panel") != panel:
        errors.append("record.panel mismatch")
    task = task_by_id.get(str(panel["task_id"]))
    if record.get("task") != task:
        errors.append("record.task mismatch")
    if record.get("task_commitment_hash") != task.get("task_commitment_hash"):
        errors.append("record.task_commitment_hash mismatch")

    attempt_id = record.get("attempt_id")
    if not isinstance(attempt_id, str) or not _SAFE_ID.fullmatch(attempt_id):
        errors.append("record.attempt_id invalid")
    elif attempt_id != deterministic_attempt_id(binds["authorization_hash"], pid):
        errors.append("record.attempt_id is not the deterministic panel id")

    status = record.get("status")
    if status not in ("completed", "infrastructure_invalid"):
        errors.append(f"unknown record status {status!r}")
        return errors

    budget = record.get("budget")
    expected_reservation = _budget_reservation(panel_index_by_id[pid])
    if not isinstance(budget, Mapping):
        errors.append("record.budget missing")
    else:
        for field, expected in expected_reservation.items():
            if budget.get(field) != expected:
                errors.append(f"record.budget {field} mismatch")

    started_at = record.get("started_at")
    finished_at = record.get("finished_at")
    if not isinstance(started_at, str) or not started_at:
        errors.append("record.started_at missing")
    if not isinstance(finished_at, str) or not finished_at:
        errors.append("record.finished_at missing")
    duration = record.get("duration_seconds")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration < 0:
        errors.append("record.duration_seconds invalid")

    if status == "infrastructure_invalid":
        error = record.get("error")
        if not isinstance(error, Mapping) or not isinstance(error.get("type"), str) or not error.get("type"):
            errors.append("infrastructure_invalid record missing error.type")
        if not isinstance(error, Mapping) or not isinstance(error.get("message"), str):
            errors.append("infrastructure_invalid record missing error.message")
        if not isinstance(error, Mapping) or error.get("error_class") != "infrastructure_invalid":
            errors.append("infrastructure_invalid record has invalid error_class")
        if not isinstance(error, Mapping) or not isinstance(error.get("error_code"), str):
            errors.append("infrastructure_invalid record missing error_code")
        if not isinstance(error, Mapping) or not isinstance(error.get("phase"), str):
            errors.append("infrastructure_invalid record missing phase")
        if isinstance(error, Mapping) and error.get("attempt_id") != attempt_id:
            errors.append("infrastructure_invalid record attempt_id mismatch")
        return errors

    # status == "completed" -> deep result validation + budget reconciliation.
    baseline_treatment = build_empty_overlay_registry().treatments[0]
    bundle_id = baseline_treatment.bundle_id
    expected_policy = policy_spec_from_treatment(baseline_treatment).to_dict()
    runtime_pins = manifest["runtime_pins"]

    errors.extend(
        _validate_completed_result(
            record.get("result"),
            panel=panel,
            task=task,
            panel_id=pid,
            attempt_id=attempt_id,
            bundle_id=bundle_id,
            expected_policy=expected_policy,
            runtime_pins=runtime_pins,
            registry_hash=registry_hash,
            manifest_hash=binds["manifest_hash"],
        )
    )

    result = record.get("result")
    if isinstance(result, Mapping) and isinstance(result.get("attempts"), Mapping):
        item = result["attempts"].get(bundle_id)
        if isinstance(item, Mapping) and isinstance(budget, Mapping):
            if budget.get("consumed") != _attempt_budget_consumption(item):
                errors.append("record.budget consumed mismatch")
    return errors


# ---------------------------------------------------------------------------
# ledger loading (strict)
# ---------------------------------------------------------------------------


def _load_ledger(
    path: Path,
    *,
    binds: Mapping[str, str],
    manifest: Mapping[str, Any],
    registry_hash: str,
) -> list[dict[str, Any]]:
    """Load and strictly validate every ledger record.

    Rejects malformed lines, unknown/duplicate panels, duplicate attempt ids,
    mixed binds, and result drift. Returns all valid records (completed and
    infrastructure_invalid).
    """
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    seen_panels: set[str] = set()
    seen_attempt_ids: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid baseline JSONL line {line_number}: {error}"
            ) from error
        if not isinstance(record, Mapping):
            raise ValueError(f"line {line_number}: record must be an object")
        pid = str(record.get("panel_id", ""))
        if pid in seen_panels:
            raise ValueError(f"line {line_number}: duplicate panel {pid}")
        attempt_id = str(record.get("attempt_id", ""))
        if attempt_id in seen_attempt_ids:
            raise ValueError(
                f"line {line_number}: duplicate attempt id {attempt_id}"
            )
        errors = _validate_record(
            record, binds=binds, manifest=manifest, registry_hash=registry_hash
        )
        if errors:
            raise ValueError(
                f"line {line_number}: invalid baseline panel record: "
                f"{'; '.join(errors)}"
            )
        if record.get("panel_index") != len(records):
            raise ValueError(
                f"line {line_number}: ledger is not a manifest-order prefix"
            )
        seen_panels.add(pid)
        seen_attempt_ids.add(attempt_id)
        records.append(record)
    return records


def _existing_completed_records(
    path: Path,
    *,
    binds: Mapping[str, str],
    manifest: Mapping[str, Any],
    registry_hash: str,
) -> set[str]:
    records = _load_ledger(
        path, binds=binds, manifest=manifest, registry_hash=registry_hash
    )
    for record in records:
        if record.get("status") == "infrastructure_invalid":
            raise RuntimeError(
                "existing infrastructure_invalid record requires adjudication"
            )
    return {str(record["panel_id"]) for record in records}


# ---------------------------------------------------------------------------
# completion receipt
# ---------------------------------------------------------------------------


def _write_completion_receipt(
    receipt_path: Path,
    *,
    binds: Mapping[str, str],
    result_filename: str,
    ledger_sha256: str,
    record_count: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": COMPLETION_RECEIPT_SCHEMA_VERSION,
        **dict(binds),
        "result_filename": result_filename,
        "record_count": record_count,
        "ledger_sha256": ledger_sha256,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    receipt = {**payload, "receipt_hash": _canonical_hash(payload)}
    _write_immutable_json(receipt_path, receipt)
    return receipt


def _validate_completion_receipt(
    receipt: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    registry_hash: str,
    local_preflight_hash: str,
    remote_preflight_hash: str,
    source_tree_hash: str,
    result_filename: str,
    ledger_sha256: str,
) -> str:
    """Validate a completion receipt; return the bound authorization hash."""
    _verify_embedded_hash(receipt, "receipt_hash")
    if receipt.get("schema_version") != COMPLETION_RECEIPT_SCHEMA_VERSION:
        raise ValueError("unsupported completion receipt schema")
    if receipt.get("manifest_hash") != manifest["manifest_hash"]:
        raise ValueError("completion receipt manifest hash mismatch")
    if receipt.get("registry_hash") != registry_hash:
        raise ValueError("completion receipt registry hash mismatch")
    if receipt.get("local_preflight_hash") != local_preflight_hash:
        raise ValueError("completion receipt local preflight hash mismatch")
    if receipt.get("remote_preflight_hash") != remote_preflight_hash:
        raise ValueError("completion receipt remote preflight hash mismatch")
    if receipt.get("source_tree_hash") != source_tree_hash:
        raise ValueError("completion receipt source tree hash mismatch")
    if receipt.get("result_filename") != result_filename:
        raise ValueError("completion receipt result filename mismatch")
    if receipt.get("record_count") != EXPECTED_ATTEMPTS:
        raise ValueError("completion receipt record count mismatch")
    if receipt.get("ledger_sha256") != ledger_sha256:
        raise ValueError("completion receipt ledger sha256 mismatch")
    authorization_hash = receipt.get("authorization_hash")
    if not _is_hex_digest(authorization_hash, 64):
        raise ValueError("completion receipt authorization hash invalid")
    return str(authorization_hash)


# ---------------------------------------------------------------------------
# authorized baseline runner
# ---------------------------------------------------------------------------


def _require_remote_identity_match(
    config: RemoteConfig, remote: Mapping[str, Any]
) -> None:
    if not str(config.host).strip():
        raise ValueError("host must not be empty")
    expected = {
        "host": str(remote["host"]),
        "project": str(remote["project"]),
        "run_root": str(remote["run_root"]),
        "python": str(remote["python"]),
    }
    actual = {
        "host": config.host,
        "project": config.project,
        "run_root": config.run_root,
        "python": config.python,
    }
    if actual != expected:
        raise ValueError(
            f"RemoteConfig does not match the manifest remote identity: "
            f"{actual} != {expected}"
        )


def _build_args(
    panel: Mapping[str, Any],
    task: Mapping[str, Any],
    bundle_id: str,
    manifest: Mapping[str, Any],
    *,
    pi_binary: str,
    provider: str,
    model: str,
    thinking: str,
    unbrowser_binary: str,
) -> argparse.Namespace:
    return argparse.Namespace(
        family="unbrowser_fixture",
        seed=int(task["seed"]),
        difficulty=str(task["difficulty"]),
        fixture_template=str(task["template"]),
        fixture_generator_version=OUTCOME_ONLY_GENERATOR_VERSION,
        task_role=str(task["role"]),
        rollout_replica=int(panel["rollout_replica"]),
        sampling_seed=int(panel["sampling_seed"]),
        pilot_manifest_hash=str(manifest["manifest_hash"]),
        pilot_panel_id=str(panel["panel_id"]),
        expected_task_commitment_hash=str(task["task_commitment_hash"]),
        bundle_id=bundle_id,
        pi=pi_binary,
        provider=provider,
        model=model,
        thinking=thinking,
        model_switch_extension=None,
        unbrowser_binary=unbrowser_binary,
    )


def run_authorized_baseline(
    manifest_path: str | Path,
    registry_path: str | Path,
    local_preflight_path: str | Path,
    remote_preflight_path: str | Path,
    authorization_path: str | Path,
    expected_authorization_hash: str,
    result_path: str | Path,
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
    """Run the 72-panel baseline strictly sequentially under an authorization.

    Executes one model invocation per panel. Stops permanently on any
    infrastructure-invalid outcome. Never retries. Safe resume skips only
    strictly valid completed records.
    """
    project_root = Path(__file__).resolve().parents[2]
    registry = TreatmentRegistry.load(registry_path)
    manifest = _load_json(manifest_path)
    local_preflight = _load_json(local_preflight_path)
    remote_preflight = _load_json(remote_preflight_path)
    authorization = _load_json(authorization_path)

    validate_baseline_manifest(manifest, registry)
    validate_local_preflight(
        local_preflight, manifest, registry, project_root, pi_executable=pi_binary
    )
    validate_remote_preflight(
        remote_preflight,
        manifest,
        registry,
        local_preflight,
        project_root,
        pi_executable=pi_binary,
    )

    source_hash = source_tree_hash(project_root)
    if source_hash != local_preflight.get("source_tree_hash"):
        raise RuntimeError("source tree drifted from the frozen local preflight")

    runtime_pins = manifest["runtime_pins"]
    if provider != runtime_pins["provider"] or model != runtime_pins["model_alias"]:
        raise ValueError("provider/model do not match the frozen runtime pins")
    if thinking != runtime_pins["thinking"]:
        raise ValueError("thinking mode does not match the frozen runtime pins")

    remote = manifest["remote_identity"]
    _require_remote_identity_match(config, remote)

    output = Path(result_path).expanduser().resolve()
    paths = _sibling_paths(output)

    authorization_hash = validate_execution_authorization(
        authorization,
        expected_authorization_hash=expected_authorization_hash,
        manifest_hash=manifest["manifest_hash"],
        registry_hash=registry.registry_hash,
        local_preflight_hash=local_preflight["preflight_hash"],
        remote_preflight_hash=remote_preflight["preflight_hash"],
        source_tree_hash=source_hash,
        remote_identity=remote,
        result_filename=output.name,
        result_path=output,
    )
    authorization_expires_at = _parse_tz_aware(
        authorization["expires_at"], "expires_at"
    )

    # Re-run the no-model runtime preflight and require identity stability with
    # the frozen remote preflight. The lifecycle stress test is NOT rerun.
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
    if runtime.get("source_tree_hash") != remote_preflight["runtime"]["source_tree_hash"]:
        raise RuntimeError(
            "runtime source identity drifted from the frozen remote preflight"
        )
    if runtime.get("runtime_pins") != remote_preflight["runtime"]["runtime_pins"]:
        raise RuntimeError("runtime identity drifted from the frozen remote preflight")

    output.parent.mkdir(parents=True, exist_ok=True)
    runtime_preflight_payload: dict[str, Any] = {
        "authorization_hash": authorization_hash,
        "manifest_hash": manifest["manifest_hash"],
        "source_tree_hash": source_hash,
        **runtime,
    }
    if paths["runtime_preflight"].exists():
        existing = _load_json(paths["runtime_preflight"])
        for key in (
            "authorization_hash",
            "manifest_hash",
            "source_tree_hash",
            "runtime_pins",
            "code_revision",
        ):
            if existing.get(key) != runtime_preflight_payload.get(key):
                raise RuntimeError("runtime preflight identity changed across resume")
    else:
        _write_immutable_json(paths["runtime_preflight"], runtime_preflight_payload)

    binds = _record_binds(
        authorization_hash=authorization_hash,
        manifest_hash=manifest["manifest_hash"],
        registry_hash=registry.registry_hash,
        local_preflight_hash=local_preflight["preflight_hash"],
        remote_preflight_hash=remote_preflight["preflight_hash"],
        source_tree_hash=source_hash,
    )

    _acquire_lock(paths["lock"])
    try:
        return _run_locked(
            output=output,
            paths=paths,
            manifest=manifest,
            registry=registry,
            local_preflight=local_preflight,
            remote_preflight=remote_preflight,
            binds=binds,
            authorization_hash=authorization_hash,
            authorization_expires_at=authorization_expires_at,
            source_hash=source_hash,
            config=config,
            pi_binary=pi_binary,
            provider=provider,
            model=model,
            thinking=thinking,
            unbrowser_binary=unbrowser_binary,
        )
    finally:
        # The lock is ephemeral and released even on interruption; the active
        # marker (not the lock) is the durable fail-closed signal.
        paths["lock"].unlink(missing_ok=True)


def launch_authorized_baseline_detached(
    manifest_path: str | Path,
    registry_path: str | Path,
    local_preflight_path: str | Path,
    remote_preflight_path: str | Path,
    authorization_path: str | Path,
    expected_authorization_hash: str,
    result_path: str | Path,
    config: RemoteConfig,
    *,
    pi_binary: str,
    provider: str,
    model: str,
    thinking: str,
    unbrowser_binary: str,
    model_artifact: str,
    llama_server_binary: str,
    python_executable: str | None = None,
    startup_timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Launch the existing authorized runner in a durable detached session."""
    if startup_timeout_seconds < 0:
        raise ValueError("startup_timeout_seconds must be non-negative")
    project_root = Path(__file__).resolve().parents[2]
    output = Path(result_path).expanduser().resolve()
    paths = _sibling_paths(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    existing = [
        path
        for path in (
            output,
            paths["lock"],
            paths["launch_lock"],
            paths["launch"],
            paths["controller_log"],
            paths["claim"],
            paths["active"],
            paths["receipt"],
            paths["runtime_preflight"],
        )
        if path.exists()
    ]
    if existing:
        rendered = ", ".join(str(path) for path in existing)
        raise RuntimeError(
            f"detached launch requires a fresh result path; found: {rendered}"
        )

    command = [
        python_executable or sys.executable,
        "-m",
        "pyreplab_harness.m3_empty_overlay_execution",
        "run",
        "--authorization-hash",
        expected_authorization_hash,
        "--authorization",
        str(Path(authorization_path).expanduser().resolve()),
        "--manifest",
        str(Path(manifest_path).expanduser().resolve()),
        "--registry",
        str(Path(registry_path).expanduser().resolve()),
        "--local-preflight",
        str(Path(local_preflight_path).expanduser().resolve()),
        "--remote-preflight",
        str(Path(remote_preflight_path).expanduser().resolve()),
        "--result",
        str(output),
        "--host",
        config.host,
        "--remote-project",
        config.project,
        "--remote-run-root",
        config.run_root,
        "--remote-python",
        config.python,
        "--pi",
        pi_binary,
        "--provider",
        provider,
        "--model",
        model,
        "--thinking",
        thinking,
        "--unbrowser-binary",
        unbrowser_binary,
        "--model-artifact",
        model_artifact,
        "--llama-server-binary",
        llama_server_binary,
    ]

    _acquire_lock(paths["launch_lock"])
    spawned = False
    receipt_written = False
    try:
        with paths["controller_log"].open("x", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=project_root,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
            spawned = True

        startup_state = "pending"
        controller_return_code: int | None = None
        deadline = time.monotonic() + startup_timeout_seconds
        while True:
            if paths["claim"].is_file():
                claim = _load_json(paths["claim"])
                _validate_claim(
                    claim,
                    expected_authorization_hash,
                    output,
                    output.name,
                )
                if claim["controller_pid"] != process.pid:
                    raise RuntimeError(
                        "observed claim belongs to a different controller process"
                    )
                startup_state = "claim_observed"
                break
            controller_return_code = process.poll()
            if controller_return_code is not None:
                startup_state = "exited_before_claim"
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)

        payload: dict[str, Any] = {
            "schema_version": DETACHED_LAUNCH_SCHEMA_VERSION,
            "screen_id": SCREEN_ID,
            "authorization_hash": expected_authorization_hash,
            "result_path": str(output),
            "controller_log_path": str(paths["controller_log"]),
            "project_root": str(project_root),
            "controller_pid": process.pid,
            "controller_process_group": process.pid,
            "startup_state": startup_state,
            "controller_return_code": controller_return_code,
            "detached_session": True,
            "command": command,
            "command_hash": _canonical_hash(command),
            "launched_at": datetime.now(timezone.utc).isoformat(),
        }
        receipt = {**payload, "launch_hash": _canonical_hash(payload)}
        _write_immutable_json(paths["launch"], receipt)
        receipt_written = True
        if startup_state == "exited_before_claim":
            raise RuntimeError(
                "detached controller exited before claiming the authorization; "
                f"inspect {paths['controller_log']}"
            )
        return receipt
    finally:
        # Keep the reservation if a process escaped without a durable receipt.
        # That state requires adjudication rather than a duplicate launch.
        if not spawned:
            paths["controller_log"].unlink(missing_ok=True)
        if receipt_written or not spawned:
            paths["launch_lock"].unlink(missing_ok=True)


def _run_locked(
    *,
    output: Path,
    paths: Mapping[str, Path],
    manifest: Mapping[str, Any],
    registry: TreatmentRegistry,
    local_preflight: Mapping[str, Any],
    remote_preflight: Mapping[str, Any],
    binds: Mapping[str, str],
    authorization_hash: str,
    authorization_expires_at: datetime,
    source_hash: str,
    config: RemoteConfig,
    pi_binary: str,
    provider: str,
    model: str,
    thinking: str,
    unbrowser_binary: str,
) -> dict[str, Any]:
    if paths["active"].exists():
        raise RuntimeError(
            "active panel marker exists; adjudication required before resume"
        )
    _prepare_claim(paths["claim"], output, authorization_hash, output.name)

    completed = _existing_completed_records(
        output,
        binds=binds,
        manifest=manifest,
        registry_hash=registry.registry_hash,
    )

    treatment = registry.treatments[0]
    bundle_id = treatment.bundle_id
    policy = policy_spec_from_treatment(treatment)
    expected_policy = policy.to_dict()
    runtime_pins = manifest["runtime_pins"]
    task_by_id = {str(task["task_id"]): task for task in manifest["tasks"]}
    task_commitment_by_id = {
        str(item["task"]["id"]): item
        for item in local_preflight["generated_tasks"]
    }
    panel_index_by_id = {
        str(panel["panel_id"]): index
        for index, panel in enumerate(manifest["panels"])
    }
    project_root = Path(__file__).resolve().parents[2]

    ran = skipped = 0
    for panel in manifest["panels"]:
        panel_id = str(panel["panel_id"])
        if panel_id in completed:
            skipped += 1
            continue
        task = task_by_id[str(panel["task_id"])]
        expected_task_commitment = task_commitment_by_id[task["task_id"]]
        attempt_id = deterministic_attempt_id(authorization_hash, panel_id)
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.monotonic()
        reservation = _budget_reservation(panel_index_by_id[panel_id])

        record: dict[str, Any] = {
            "schema_version": PANEL_RESULT_SCHEMA_VERSION,
            "authorization_hash": authorization_hash,
            "manifest_hash": binds["manifest_hash"],
            "registry_hash": binds["registry_hash"],
            "local_preflight_hash": binds["local_preflight_hash"],
            "remote_preflight_hash": binds["remote_preflight_hash"],
            "source_tree_hash": binds["source_tree_hash"],
            "panel_id": panel_id,
            "panel_index": panel_index_by_id[panel_id],
            "task": task,
            "task_commitment_hash": expected_task_commitment["commitment_hash"],
            "panel": panel,
            "attempt_id": attempt_id,
            "budget": dict(reservation),
            "started_at": started_at,
        }

        _require_authorization_active(authorization_expires_at)
        _write_active_marker(
            paths["active"],
            authorization_hash=authorization_hash,
            panel_id=panel_id,
            attempt_id=attempt_id,
            started_at=started_at,
            budget_reserved=reservation["reserved_for_panel"],
        )

        args = _build_args(
            panel,
            task,
            bundle_id,
            manifest,
            pi_binary=pi_binary,
            provider=provider,
            model=model,
            thinking=thinking,
            unbrowser_binary=unbrowser_binary,
        )

        caught_error: dict[str, Any] | None = None
        attempt: Mapping[str, Any] | None = None
        generated_task: Any = None
        try:
            generated_task = _task_json(config, args)
            _verify_generated_task(
                generated_task, task, expected_task_commitment
            )
            remote_task_commitment = remote_json(
                config,
                [
                    "fixture-task-commitment",
                    "--root",
                    config.run_root,
                    "--task-id",
                    task["task_id"],
                ],
            )
            if remote_task_commitment != expected_task_commitment:
                raise ValueError(
                    "remote task workspace or oracle drifted from the local commitment"
                )
            _require_authorization_active(authorization_expires_at)
            attempt = _run_attempt(
                project_root,
                config,
                generated_task,
                policy,
                attempt_id,
                args,
                with_usage=True,
                registry_hash=registry.registry_hash,
                require_complete_event_summary=True,
                before_model_admission=lambda: _require_authorization_active(
                    authorization_expires_at
                ),
            )
        except AttemptExecutionError as error:
            caught_error = {
                "type": type(error).__name__,
                "message": str(error),
                "error_class": "infrastructure_invalid",
                "source_error_class": error.error_class,
                "error_code": error.error_code,
                "phase": error.phase,
                "attempt_id": error.attempt_id or attempt_id,
            }
        except KeyboardInterrupt:
            raise
        except Exception as error:  # noqa: BLE001 - typed into infra record.
            caught_error = {
                "type": type(error).__name__,
                "message": str(error),
                "error_class": "infrastructure_invalid",
                "error_code": "controller_error",
                "phase": "generate_or_execute",
                "attempt_id": attempt_id,
            }

        finished_at = datetime.now(timezone.utc).isoformat()
        duration = round(time.monotonic() - started, 3)

        if caught_error is not None:
            status = "infrastructure_invalid"
            reason = f"{caught_error['type']}: {caught_error['message']}"
            consumed = None
        else:
            status, reason = _classify_attempt(
                attempt,
                expected_policy=expected_policy,
                expected_sampling_receipt={
                    "seed": int(panel["sampling_seed"]),
                    "parameters": runtime_pins["sampling"]["parameters"],
                },
                runtime_pins=runtime_pins,
            )
            consumed = (
                _attempt_budget_consumption(attempt)
                if attempt is not None
                else None
            )

        if status == "completed":
            result = _one_policy_result(
                generated_task,
                attempt,
                bundle_id,
                registry.registry_hash,
                args,
            )
            record.update(
                {
                    "status": "completed",
                    "finished_at": finished_at,
                    "duration_seconds": duration,
                    "budget": {
                        **record["budget"],
                        "consumed": consumed,
                    },
                    "result": result,
                }
            )
            # Self-hash before strict validation so _validate_record can check
            # the embedded record_hash; _append_record recomputes the same value.
            record["record_hash"] = _canonical_hash(
                {key: value for key, value in record.items() if key != "record_hash"}
            )
            validation_errors = _validate_record(
                record,
                binds=binds,
                manifest=manifest,
                registry_hash=registry.registry_hash,
            )
            if validation_errors:
                record["status"] = "infrastructure_invalid"
                record.pop("result", None)
                record["budget"] = {**record["budget"], "consumed": consumed}
                record["error"] = {
                    "type": "MalformedPanelResult",
                    "message": "; ".join(validation_errors),
                    "error_class": "infrastructure_invalid",
                    "error_code": "malformed_panel_result",
                    "phase": "validate_result",
                    "attempt_id": attempt_id,
                }
                _append_record(output, record)
                paths["active"].unlink()
                raise RuntimeError(
                    f"baseline stopped after malformed result on {panel_id}: "
                    f"{'; '.join(validation_errors)}"
                )
            _append_record(output, record)
            paths["active"].unlink()
            ran += 1
        else:
            record.update(
                {
                    "status": "infrastructure_invalid",
                    "finished_at": finished_at,
                    "duration_seconds": duration,
                    "budget": {
                        **record["budget"],
                        "consumed": consumed,
                    },
                    "error": caught_error
                    or {
                        "type": "InfrastructureInvalidAttempt",
                        "message": str(reason),
                        "error_class": "infrastructure_invalid",
                        "error_code": "attempt_infrastructure_marker",
                        "phase": "validate_attempt",
                        "attempt_id": attempt_id,
                    },
                }
            )
            _append_record(output, record)
            paths["active"].unlink()
            raise RuntimeError(
                f"baseline stopped after infrastructure error on {panel_id}: {reason}"
            )

    # Completion receipt (at-most-once).
    records = _load_ledger(
        output, binds=binds, manifest=manifest, registry_hash=registry.registry_hash
    )
    receipt_path = paths["receipt"]
    receipt = None
    if len(records) == EXPECTED_ATTEMPTS and all(
        record.get("status") == "completed" for record in records
    ):
        ledger_sha256 = _sha256_file(output)
        if receipt_path.exists():
            receipt = _load_json(receipt_path)
            _validate_completion_receipt(
                receipt,
                manifest=manifest,
                registry_hash=registry.registry_hash,
                local_preflight_hash=local_preflight["preflight_hash"],
                remote_preflight_hash=remote_preflight["preflight_hash"],
                source_tree_hash=source_hash,
                result_filename=output.name,
                ledger_sha256=ledger_sha256,
            )
        else:
            receipt = _write_completion_receipt(
                receipt_path,
                binds=binds,
                result_filename=output.name,
                ledger_sha256=ledger_sha256,
                record_count=len(records),
            )

    return {
        "authorization_hash": authorization_hash,
        "manifest_hash": manifest["manifest_hash"],
        "result": str(output),
        "panels_total": EXPECTED_ATTEMPTS,
        "panels_run": ran,
        "panels_skipped": skipped,
        "completion_receipt": str(receipt_path) if receipt is not None else None,
    }


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------


def _wilson_interval(
    successes: int, trials: int, z: float = 1.96
) -> tuple[float, float]:
    if trials == 0:
        return (0.0, 1.0)
    phat = successes / trials
    z2 = z * z
    denom = 1 + z2 / trials
    center = (phat + z2 / (2 * trials)) / denom
    margin = (
        z
        * math.sqrt(phat * (1 - phat) / trials + z2 / (4 * trials * trials))
        / denom
    )
    return (center - margin, center + margin)


def _classify_template(
    successes: int,
    attempts: int,
    discordant_tasks: int,
    total_tasks: int,
    stable_failure_tasks: int,
) -> str:
    rate = successes / attempts if attempts else 0.0
    discordance_rate = discordant_tasks / total_tasks if total_tasks else 0.0
    if discordance_rate > 1.0 / 3.0:
        return "unstable"
    if rate >= 0.80:
        return "ceiling"
    if rate <= 0.20:
        return "floor_risk"
    if stable_failure_tasks >= 2:
        return "challenge_candidate"
    return "insufficient_repeatability"


def _numeric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "min": None, "max": None, "total": None, "count": 0}
    return {
        "mean": round(sum(values) / len(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "total": round(sum(values), 3),
        "count": len(values),
    }


def _replica_agreement(
    task_ids: list[str], task_outcomes: Mapping[str, Mapping[int, bool]]
) -> dict[str, Any]:
    complete = [
        task_outcomes[task_id]
        for task_id in task_ids
        if set(task_outcomes.get(task_id, {})) == {0, 1}
    ]
    discordant = sum(outcomes[0] != outcomes[1] for outcomes in complete)
    agreed = len(complete) - discordant
    return {
        "total_tasks": len(task_ids),
        "complete_tasks": len(complete),
        "agreed_tasks": agreed,
        "discordant_tasks": discordant,
        "agreement_rate": agreed / len(complete) if complete else None,
        "discordance_rate": discordant / len(complete) if complete else None,
    }


def _terminal_mechanism(item: Mapping[str, Any]) -> str:
    verification = item.get("verification")
    if isinstance(verification, Mapping) and verification.get("success") is True:
        return "success"
    if item.get("pi_return_code") == -1:
        return "wall_time_exceeded"
    trajectory = item.get("trajectory")
    trace = trajectory.get("tool_trace") if isinstance(trajectory, Mapping) else []
    entries = [entry for entry in (trace or []) if isinstance(entry, Mapping)]
    if any(
        entry.get("budget_rejected") is True
        or entry.get("operation_aborted") is True
        for entry in entries
    ):
        return "tool_budget_exhaustion"
    if any(entry.get("pre_execution_rejected") is True for entry in entries):
        return "tool_validation_error"
    failure_code = (
        verification.get("failure_code")
        if isinstance(verification, Mapping)
        else None
    )
    if failure_code == "missing_output":
        return "missing_submission"
    if failure_code in {"invalid_json", "wrong_type", "missing_key", "wrong_key_type"}:
        return "malformed_submission"
    if failure_code == "nonce_mismatch":
        return "incorrect_answer"
    if entries and entries[-1].get("is_error") is True:
        return "terminal_tool_error"
    return "unknown_failure"


def _trace_signature(entry: Mapping[str, Any]) -> dict[str, Any]:
    details = entry.get("details")
    return {
        "tool_name": entry.get("tool_name"),
        "is_error": entry.get("is_error"),
        "budget_rejected": entry.get("budget_rejected"),
        "operation_aborted": entry.get("operation_aborted"),
        "pre_execution_rejected": entry.get("pre_execution_rejected"),
        "error": details.get("error") if isinstance(details, Mapping) else None,
        "status": details.get("status") if isinstance(details, Mapping) else None,
    }


def _replicated_failure_divergence(
    records: list[dict[str, Any]],
    manifest: Mapping[str, Any],
    bundle_id: str,
    task_outcomes: Mapping[str, Mapping[int, bool]],
) -> list[dict[str, Any]]:
    records_by_task: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        records_by_task.setdefault(str(record["task"]["task_id"]), []).append(record)

    divergences: list[dict[str, Any]] = []
    for task in manifest["tasks"]:
        task_id = str(task["task_id"])
        outcomes = task_outcomes.get(task_id, {})
        if set(outcomes) != {0, 1} or outcomes[0] or outcomes[1]:
            continue  # only replicated failures (both replicas failed)
        traces: dict[int, list[dict[str, Any]]] = {}
        for record in records_by_task.get(task_id, []):
            replica = int(record["panel"]["rollout_replica"])
            item = record["result"]["attempts"][bundle_id]
            trajectory = (
                item.get("trajectory")
                if isinstance(item.get("trajectory"), Mapping)
                else {}
            )
            tool_trace = trajectory.get("tool_trace") or []
            traces[replica] = [
                _trace_signature(entry)
                for entry in tool_trace
                if isinstance(entry, Mapping)
            ]
        trace_0 = traces.get(0, [])
        trace_1 = traces.get(1, [])
        divergence_index: int | None = None
        for index in range(max(len(trace_0), len(trace_1))):
            left: dict[str, Any] | None = (
                trace_0[index] if index < len(trace_0) else None
            )
            right: dict[str, Any] | None = (
                trace_1[index] if index < len(trace_1) else None
            )
            if left != right:
                divergence_index = index
                break
        divergences.append(
            {
                "task_id": task_id,
                "template": task["template"],
                "difficulty": task["difficulty"],
                "divergence_index": divergence_index,
                "identical": divergence_index is None,
                "replica_traces": {"0": trace_0, "1": trace_1},
            }
        )
    return divergences


def _compute_analysis(
    records: list[dict[str, Any]],
    manifest: Mapping[str, Any],
    registry: TreatmentRegistry,
    binds: Mapping[str, str],
    receipt: Mapping[str, Any],
    ledger_sha256: str,
) -> dict[str, Any]:
    bundle_id = registry.treatments[0].bundle_id
    template_order = list(manifest["known_templates"])
    accounting_metrics = (
        "provider_backed_turns",
        "tool_attempts",
        "budget_admitted_tool_attempts",
        "executed_tool_calls",
        "rejected_tool_attempts",
        "budget_blocked_tool_attempts",
        "suppressed_tool_requests",
        "provider_gate_blocks",
    )

    overall_successes = 0
    overall_attempts = 0
    per_template: dict[str, dict[str, Any]] = {
        template: {
            "successes": 0,
            "attempts": 0,
            "failure_codes": {},
            "terminal_mechanisms": {},
            **{metric: [] for metric in accounting_metrics},
            "output_tokens": [],
            "elapsed": [],
        }
        for template in template_order
    }
    per_cell: dict[tuple[str, str], dict[str, Any]] = {
        (template, difficulty): {
            "successes": 0,
            "attempts": 0,
            "failure_codes": {},
            "terminal_mechanisms": {},
            **{metric: [] for metric in accounting_metrics},
            "output_tokens": [],
            "elapsed": [],
        }
        for template in template_order
        for difficulty in _DIFFICULTIES
    }
    failure_codes: dict[str, int] = {}
    terminal_mechanisms: dict[str, int] = {}
    overall_accounting: dict[str, list[float]] = {
        metric: [] for metric in accounting_metrics
    }
    output_tokens: list[float] = []
    elapsed: list[float] = []
    task_outcomes: dict[str, dict[int, bool]] = {}

    for record in records:
        task = record["task"]
        panel = record["panel"]
        template = str(task["template"])
        difficulty = str(task["difficulty"])
        task_id = str(task["task_id"])
        replica = int(panel["rollout_replica"])
        item = record["result"]["attempts"][bundle_id]
        verification = item["verification"]
        success = bool(verification.get("success"))

        overall_attempts += 1
        if success:
            overall_successes += 1

        per_template[template]["successes"] += int(success)
        per_template[template]["attempts"] += 1
        per_cell[(template, difficulty)]["successes"] += int(success)
        per_cell[(template, difficulty)]["attempts"] += 1

        task_outcomes.setdefault(task_id, {})[replica] = success

        mechanism = _terminal_mechanism(item)
        terminal_mechanisms[mechanism] = terminal_mechanisms.get(mechanism, 0) + 1
        for bucket in (per_template[template], per_cell[(template, difficulty)]):
            bucket_mechanisms = bucket["terminal_mechanisms"]
            bucket_mechanisms[mechanism] = bucket_mechanisms.get(mechanism, 0) + 1
        if not success:
            failure_code = verification.get("failure_code")
            code = failure_code if isinstance(failure_code, str) else "unknown"
            failure_codes[code] = failure_codes.get(code, 0) + 1
            for bucket in (per_template[template], per_cell[(template, difficulty)]):
                bucket_codes = bucket["failure_codes"]
                bucket_codes[code] = bucket_codes.get(code, 0) + 1

        usage = item.get("usage") if isinstance(item.get("usage"), Mapping) else {}
        trajectory = (
            item.get("trajectory")
            if isinstance(item.get("trajectory"), Mapping)
            else {}
        )
        timing = item.get("timing") if isinstance(item.get("timing"), Mapping) else {}
        output = usage.get("output")
        if isinstance(output, (int, float)) and not isinstance(output, bool):
            output_tokens.append(float(output))
            per_template[template]["output_tokens"].append(float(output))
            per_cell[(template, difficulty)]["output_tokens"].append(float(output))
        trace = trajectory.get("tool_trace")
        budget_receipt = trajectory.get("budget_receipt")
        accounting = {
            "provider_backed_turns": trajectory.get("provider_turn_count"),
            "provider_gate_blocks": (
                budget_receipt.get("provider_request_blocks")
                if isinstance(budget_receipt, Mapping)
                else None
            ),
            **_tool_attempt_accounting(trace, budget_receipt),
        }
        for metric in accounting_metrics:
            value = accounting.get(metric)
            if isinstance(value, int) and not isinstance(value, bool):
                overall_accounting[metric].append(float(value))
                per_template[template][metric].append(float(value))
                per_cell[(template, difficulty)][metric].append(float(value))
        total = timing.get("total_seconds")
        if isinstance(total, (int, float)) and not isinstance(total, bool):
            elapsed.append(float(total))
            per_template[template]["elapsed"].append(float(total))
            per_cell[(template, difficulty)]["elapsed"].append(float(total))

    overall_wilson = _wilson_interval(overall_successes, overall_attempts)

    by_template: list[dict[str, Any]] = []
    for template in template_order:
        stats = per_template[template]
        task_ids = [
            str(task["task_id"])
            for task in manifest["tasks"]
            if task["template"] == template
        ]
        agreement = _replica_agreement(task_ids, task_outcomes)
        stable_failures = 0
        for task_id in task_ids:
            outcomes = task_outcomes.get(task_id, {})
            if set(outcomes) == {0, 1}:
                if not outcomes[0] and not outcomes[1]:
                    stable_failures += 1
        wilson = _wilson_interval(stats["successes"], stats["attempts"])
        by_template.append(
            {
                "template": template,
                "successes": stats["successes"],
                "attempts": stats["attempts"],
                "success_rate": stats["successes"] / stats["attempts"]
                if stats["attempts"]
                else 0.0,
                "wilson_95_lower": wilson[0],
                "wilson_95_upper": wilson[1],
                "replica_agreement": agreement,
                "stable_failure_tasks": stable_failures,
                "failure_codes": dict(sorted(stats["failure_codes"].items())),
                "terminal_mechanisms": dict(
                    sorted(stats["terminal_mechanisms"].items())
                ),
                "resource_summaries": {
                    **{
                        metric: _numeric_summary(stats[metric])
                        for metric in accounting_metrics
                    },
                    "output_tokens": _numeric_summary(stats["output_tokens"]),
                    "elapsed_seconds": _numeric_summary(stats["elapsed"]),
                },
                "classification": _classify_template(
                    stats["successes"],
                    stats["attempts"],
                    agreement["discordant_tasks"],
                    len(task_ids),
                    stable_failures,
                ),
            }
        )

    by_cell: list[dict[str, Any]] = []
    for template in template_order:
        for difficulty in _DIFFICULTIES:
            cell = per_cell[(template, difficulty)]
            wilson = _wilson_interval(cell["successes"], cell["attempts"])
            task_ids = [
                str(task["task_id"])
                for task in manifest["tasks"]
                if task["template"] == template
                and task["difficulty"] == difficulty
            ]
            by_cell.append(
                {
                    "template": template,
                    "difficulty": difficulty,
                    "successes": cell["successes"],
                    "attempts": cell["attempts"],
                    "success_rate": cell["successes"] / cell["attempts"]
                    if cell["attempts"]
                    else 0.0,
                    "wilson_95_lower": wilson[0],
                    "wilson_95_upper": wilson[1],
                    "replica_agreement": _replica_agreement(
                        task_ids, task_outcomes
                    ),
                    "failure_codes": dict(sorted(cell["failure_codes"].items())),
                    "terminal_mechanisms": dict(
                        sorted(cell["terminal_mechanisms"].items())
                    ),
                    "resource_summaries": {
                        **{
                            metric: _numeric_summary(cell[metric])
                            for metric in accounting_metrics
                        },
                        "output_tokens": _numeric_summary(cell["output_tokens"]),
                        "elapsed_seconds": _numeric_summary(cell["elapsed"]),
                    },
                }
            )

    divergence = _replicated_failure_divergence(
        records, manifest, bundle_id, task_outcomes
    )

    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "authorization_hash": binds["authorization_hash"],
        "manifest_hash": binds["manifest_hash"],
        "registry_hash": binds["registry_hash"],
        "ledger_sha256": ledger_sha256,
        "completion_receipt_hash": receipt["receipt_hash"],
        "screening_note": SCREENING_NOTE,
        "overall": {
            "successes": overall_successes,
            "attempts": overall_attempts,
            "success_rate": overall_successes / overall_attempts
            if overall_attempts
            else 0.0,
            "wilson_95_lower": overall_wilson[0],
            "wilson_95_upper": overall_wilson[1],
            "replica_agreement": _replica_agreement(
                [str(task["task_id"]) for task in manifest["tasks"]],
                task_outcomes,
            ),
        },
        "by_template": by_template,
        "by_template_difficulty": by_cell,
        "failure_codes": failure_codes,
        "terminal_mechanisms": terminal_mechanisms,
        "resource_summaries": {
            **{
                metric: _numeric_summary(overall_accounting[metric])
                for metric in accounting_metrics
            },
            "output_tokens": _numeric_summary(output_tokens),
            "elapsed_seconds": _numeric_summary(elapsed),
        },
        "replicated_failure_divergence": divergence,
    }


def analyze_baseline_results(
    manifest_path: str | Path,
    registry_path: str | Path,
    local_preflight_path: str | Path,
    remote_preflight_path: str | Path,
    results_path: str | Path,
    *,
    output_path: str | Path | None = None,
    pi_executable: str = "pi",
) -> dict[str, Any]:
    """Validate a complete 72-record ledger and compute screening analysis."""
    project_root = Path(__file__).resolve().parents[2]
    registry = TreatmentRegistry.load(registry_path)
    manifest = _load_json(manifest_path)
    local_preflight = _load_json(local_preflight_path)
    remote_preflight = _load_json(remote_preflight_path)

    validate_baseline_manifest(manifest, registry)
    validate_local_preflight(
        local_preflight, manifest, registry, project_root, pi_executable=pi_executable
    )
    validate_remote_preflight(
        remote_preflight,
        manifest,
        registry,
        local_preflight,
        project_root,
        pi_executable=pi_executable,
    )
    source_hash = local_preflight["source_tree_hash"]

    output = Path(results_path).expanduser().resolve()
    receipt_path = output.with_name(output.name + ".receipt.json")
    if not receipt_path.is_file():
        raise ValueError("completion receipt is required for analysis")
    receipt = _load_json(receipt_path)
    ledger_sha256 = _sha256_file(output)
    authorization_hash = _validate_completion_receipt(
        receipt,
        manifest=manifest,
        registry_hash=registry.registry_hash,
        local_preflight_hash=local_preflight["preflight_hash"],
        remote_preflight_hash=remote_preflight["preflight_hash"],
        source_tree_hash=source_hash,
        result_filename=output.name,
        ledger_sha256=ledger_sha256,
    )

    binds = _record_binds(
        authorization_hash=authorization_hash,
        manifest_hash=manifest["manifest_hash"],
        registry_hash=registry.registry_hash,
        local_preflight_hash=local_preflight["preflight_hash"],
        remote_preflight_hash=remote_preflight["preflight_hash"],
        source_tree_hash=source_hash,
    )
    records = _load_ledger(
        output, binds=binds, manifest=manifest, registry_hash=registry.registry_hash
    )
    if len(records) != EXPECTED_ATTEMPTS:
        raise ValueError(
            f"analysis requires exactly {EXPECTED_ATTEMPTS} valid records, "
            f"got {len(records)}"
        )
    if any(record.get("status") != "completed" for record in records):
        raise ValueError(
            "analysis requires a fully completed ledger with no "
            "infrastructure_invalid records"
        )

    analysis = _compute_analysis(
        records, manifest, registry, binds, receipt, ledger_sha256
    )
    analysis = {
        **analysis,
        "analysis_hash": _canonical_hash(
            {key: value for key, value in analysis.items() if key != "analysis_hash"}
        ),
    }
    if output_path:
        _write_immutable_json(Path(output_path).expanduser().resolve(), analysis)
    return analysis


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--authorization-hash", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--local-preflight", required=True)
    parser.add_argument("--remote-preflight", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument(
        "--host", default=os.environ.get("PYREPLAB_HARNESS_HOST", "ubuntu-local")
    )
    parser.add_argument("--remote-project", required=True)
    parser.add_argument("--remote-run-root", required=True)
    parser.add_argument("--remote-python", default="python3")
    parser.add_argument("--pi", default=os.environ.get("PYREPLAB_PI", "pi"))
    parser.add_argument(
        "--provider", default=os.environ.get("PYREPLAB_PI_PROVIDER", "ubuntu-gemma")
    )
    parser.add_argument(
        "--model", default=os.environ.get("PYREPLAB_PI_MODEL", "gemma-4-26b-a4b")
    )
    parser.add_argument(
        "--thinking", default=os.environ.get("PYREPLAB_PI_THINKING", "off")
    )
    parser.add_argument("--unbrowser-binary", required=True)
    parser.add_argument("--model-artifact", required=True)
    parser.add_argument(
        "--llama-server-binary", default="/usr/local/lib/ollama/llama-server"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyreplab-m3-empty-overlay-execution")
    subparsers = parser.add_subparsers(dest="command", required=True)

    request = subparsers.add_parser(
        "authorization-request",
        help="write a non-authorizing execution authorization request",
    )
    request.add_argument("--manifest", required=True)
    request.add_argument("--registry", required=True)
    request.add_argument("--local-preflight", required=True)
    request.add_argument("--remote-preflight", required=True)
    request.add_argument("--root", required=True)
    request.add_argument("--result", required=True)
    request.add_argument("--output", required=True)
    request.add_argument("--pi", default=os.environ.get("PYREPLAB_PI", "pi"))

    validate = subparsers.add_parser(
        "validate-authorization",
        help="validate a separately authored execution authorization",
    )
    validate.add_argument("--authorization", required=True)
    validate.add_argument("--authorization-hash", required=True)
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--registry", required=True)
    validate.add_argument("--local-preflight", required=True)
    validate.add_argument("--remote-preflight", required=True)
    validate.add_argument("--root", required=True)
    validate.add_argument("--result", required=True)
    validate.add_argument("--pi", default=os.environ.get("PYREPLAB_PI", "pi"))

    run = subparsers.add_parser("run", help="run the authorized baseline")
    _add_run_arguments(run)

    launch = subparsers.add_parser(
        "launch-detached",
        help="launch the authorized baseline in a durable detached session",
    )
    _add_run_arguments(launch)

    analyze = subparsers.add_parser(
        "analyze", help="analyze a complete 72-record baseline ledger"
    )
    analyze.add_argument("--manifest", required=True)
    analyze.add_argument("--registry", required=True)
    analyze.add_argument("--local-preflight", required=True)
    analyze.add_argument("--remote-preflight", required=True)
    analyze.add_argument("--results", required=True)
    analyze.add_argument("--output", default=None)
    analyze.add_argument("--pi", default=os.environ.get("PYREPLAB_PI", "pi"))

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "authorization-request":
        registry = TreatmentRegistry.load(args.registry)
        manifest = _load_json(args.manifest)
        local_preflight = _load_json(args.local_preflight)
        remote_preflight = _load_json(args.remote_preflight)
        report = build_authorization_request(
            manifest,
            registry,
            local_preflight,
            remote_preflight,
            project_root=args.root,
            result_path=args.result,
            pi_executable=args.pi,
        )
        _write_immutable_json(Path(args.output).expanduser().resolve(), report)
    elif args.command == "validate-authorization":
        registry = TreatmentRegistry.load(args.registry)
        manifest = _load_json(args.manifest)
        local_preflight = _load_json(args.local_preflight)
        remote_preflight = _load_json(args.remote_preflight)
        authorization = _load_json(args.authorization)
        project_root = Path(args.root).expanduser().resolve()
        validate_baseline_manifest(manifest, registry)
        validate_local_preflight(
            local_preflight,
            manifest,
            registry,
            project_root,
            pi_executable=args.pi,
        )
        validate_remote_preflight(
            remote_preflight,
            manifest,
            registry,
            local_preflight,
            project_root,
            pi_executable=args.pi,
        )
        source_hash = source_tree_hash(project_root)
        report = {
            "valid": True,
            "authorization_hash": validate_execution_authorization(
                authorization,
                expected_authorization_hash=args.authorization_hash,
                manifest_hash=manifest["manifest_hash"],
                registry_hash=registry.registry_hash,
                local_preflight_hash=local_preflight["preflight_hash"],
                remote_preflight_hash=remote_preflight["preflight_hash"],
                source_tree_hash=source_hash,
                remote_identity=manifest["remote_identity"],
                result_filename=Path(args.result).expanduser().resolve().name,
                result_path=args.result,
            ),
        }
    elif args.command in {"run", "launch-detached"}:
        runner = (
            run_authorized_baseline
            if args.command == "run"
            else launch_authorized_baseline_detached
        )
        report = runner(
            args.manifest,
            args.registry,
            args.local_preflight,
            args.remote_preflight,
            args.authorization,
            args.authorization_hash,
            args.result,
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
        report = analyze_baseline_results(
            args.manifest,
            args.registry,
            args.local_preflight,
            args.remote_preflight,
            args.results,
            output_path=args.output,
            pi_executable=args.pi,
        )
    else:  # pragma: no cover - argparse enforces the command set.
        raise AssertionError(f"unsupported command: {args.command}")
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANALYSIS_SCHEMA_VERSION",
    "AUTHORIZATION_SCHEMA_VERSION",
    "AUTHORIZATION_STATEMENT",
    "COMPLETION_RECEIPT_SCHEMA_VERSION",
    "DETACHED_LAUNCH_SCHEMA_VERSION",
    "MAX_ATTEMPTS",
    "PANEL_RESULT_SCHEMA_VERSION",
    "REQUEST_SCHEMA_VERSION",
    "analyze_baseline_results",
    "build_authorization_request",
    "deterministic_attempt_id",
    "launch_authorized_baseline_detached",
    "main",
    "run_authorized_baseline",
    "validate_execution_authorization",
]
