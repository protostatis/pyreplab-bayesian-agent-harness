from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest import mock
from uuid import uuid4

from pyreplab_harness.m3_pilot import _RUNTIME_PINS, _canonical_hash, source_tree_hash
from pyreplab_harness.m3_prompt_only_pilot import (
    AGGREGATE_WALL_SECONDS,
    ARMS,
    DUMMY_PROVIDER_API_KEY,
    EXPECTED_CELLS,
    EXPECTED_PANELS,
    PER_CELL_WALL_SECONDS,
    RUN_MODEL_ALIAS,
    RUN_PI_BASE_URL,
    RUN_PROVIDER,
    RUN_REMOTE_SERVER_BASE_URL,
    V8_FAILURE_HASH,
    analyze_ledger_test_only_valid_substrate,
    build_local_preflight,
    build_manifest,
    build_pi_conformance_receipt,
    build_prompt_only_registry,
    build_wall_budget_amendment,
    dummy_api_key_binding,
)
from pyreplab_harness.m3_prompt_only_execution import (
    AUTHORIZATION_SCHEMA_VERSION,
    AUTHORIZATION_STATEMENT,
    CELL_RESULT_SCHEMA_VERSION,
    COMPLETION_RECEIPT_SCHEMA_VERSION,
    EXECUTION_GENERATION,
    GENERATION_INVALID_VETO_CODES,
    MAX_CELLS,
    MAX_PANELS,
    OFF_SERVER_ROOT,
    PER_CELL_WALL_SECONDS as EXECUTION_PER_CELL_WALL_SECONDS,
    PROBE_AUTHORIZATION_STATEMENT,
    PROBE_RECEIPT_SCHEMA_VERSION,
    PROBE_REQUEST_SCHEMA_VERSION,
    PROVIDER_BACKED_TURNS_PER_INVOCATION,
    REMOTE_PREFLIGHT_SCHEMA_VERSION,
    REQUEST_SCHEMA_VERSION,
    SEVERE_VETO_CODES,
    SCREEN_ID,
    SLOT_ACTION_DIRECTORY,
    SLOT_ACTION_DIR_OBSERVATION_SCHEMA_VERSION,
    SLOT_ACTION_DIR_PREPARATION_SCHEMA_VERSION,
    SLOT_ACTION_DIR_REMOVAL_SCHEMA_VERSION,
    SUBSTRATE_EVIDENCE_SCHEMA_VERSION,
    GENERATION_LEASE_ACQUIRE_SCHEMA_VERSION,
    GENERATION_LEASE_RELEASE_SCHEMA_VERSION,
    GENERATION_LEASE_LOCAL_ACQUIRE_SCHEMA_VERSION,
    GENERATION_LEASE_LOCAL_RELEASE_SCHEMA_VERSION,
    GENERATION_LEASE_AUDIT_SCHEMA_VERSION,
    TOOL_ATTEMPTS_PER_INVOCATION,
    WALL_SECONDS_PER_INVOCATION,
    _acquire_local_generation_lease,
    _budget_reservation,
    _classify_attempt,
    _lease_failure_evidence,
    _prepare_authorized_run,
    _release_local_generation_lease,
    _reserved_budget,
    _worst_case_budget,
    acquire_generation_lease,
    generation_lease_local_lock_path,
    generation_lease_remote_path,
    release_generation_lease,
    validate_generation_lease_receipt,
    validate_local_generation_lease_receipt,
    validate_slot_action_dir_observation_receipt,
    validate_slot_action_dir_preparation_receipt,
    validate_slot_action_dir_removal_receipt,
    analyze_prompt_only_results,
    build_adjudication_receipt,
    build_authorization_request,
    build_behavior_receipt,
    build_endpoint_probe_request,
    build_frozen_models_json,
    build_remote_preflight,
    build_safe_ledger,
    build_substrate_receipt,
    deterministic_cell_attempt_id,
    detect_severe_veto,
    launch_authorized_prompt_only_detached,    launch_local_proxy,
    launch_local_tunnel,
    launch_off_server_remote,
    models_json_sha256,
    observe_slot_action_directory,
    perform_slot_clear,
    prepare_slot_action_directory,
    remove_slot_action_directory,
    run_authorized_endpoint_probe,
    run_authorized_prompt_only,
    severe_veto_verdict,
    slot_action_directory_path,
    slot_clear_contract,
    validate_adjudication_receipt,
    validate_endpoint_probe_authorization,
    validate_endpoint_probe_receipt,
    validate_execution_authorization,
    validate_execution_substrate_receipt,
    validate_remote_preflight,
    validate_runtime_identity,
    validate_slot_clear_receipt,
    write_frozen_models_json,
)
from pyreplab_harness.orchestrator import RemoteConfig, policy_spec_from_treatment

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _remote_identity() -> dict:
    """A content-addressed remote identity bound to the current source bundle."""
    from pyreplab_harness.m3_prompt_only_pilot import (
        content_addressed_project_path,
        source_bundle_manifest_hash,
    )

    bundle_hash = source_bundle_manifest_hash(PROJECT_ROOT)
    project = content_addressed_project_path("/remote/project", bundle_hash)
    return {
        "host": "ubuntu-local",
        "project": project,
        "run_root": f"{project}/.runs/prompt-only",
        "python": "python3",
    }


REMOTE_IDENTITY = _remote_identity()
RESULT_FILENAME = "prompt-only.jsonl"

# A persistent, empty run root so the local preflight's collision scan is
# reproducible for the lifetime of the test session.
_TMP_ROOT = tempfile.mkdtemp(prefix="pyreplab-ppo-exec-test-")
_RUN_ROOT = Path(_TMP_ROOT) / "runs"
_RUN_ROOT.mkdir(parents=True, exist_ok=True)


# --- Active-service quiescence fixture (real gemma.service contract) ---
_SVC_BOOT_ID = "0123456789abcdef0123456789abcdef"
_SVC_INVOCATION_ID = "b5123456789012345678901234567890"
_SVC_STATE_PAYLOAD = '{"state":"sleeping","payload":null}'
# Opaque cursors (never parsed). The ``i=`` field is variable-width hex:
# ``i=f`` (15) string-sorts ABOVE ``i=10`` (16), so only chronological
# (last-record) order — never string comparison — picks the right high-water.
_SVC_STATE_EVENT_CURSOR = (
    "s=0123456789abcdef0123456789abcdef;i=9;"
    f"b={_SVC_BOOT_ID};m=100;t=1000000;x=9"
)
_SVC_MIDDLE_CURSOR = (
    "s=0123456789abcdef0123456789abcdef;i=f;"
    f"b={_SVC_BOOT_ID};m=150;t=1000001;x=15"
)
_SVC_HIGH_WATER_CURSOR = (
    "s=0123456789abcdef0123456789abcdef;i=10;"
    f"b={_SVC_BOOT_ID};m=200;t=1000002;x=16"
)


def _service_status_text(
    *,
    active_state: str = "active",
    sub_state: str = "running",
    main_pid: str = "32831",
    invocation_id: str = _SVC_INVOCATION_ID,
    extra_port: str | None = None,
) -> str:
    lines = [
        f"ActiveState={active_state}",
        f"SubState={sub_state}",
        f"MainPID={main_pid}",
        "ControlGroup=/user.slice/user-1000.slice/user@1000.service/app.slice/gemma.service",
        f"InvocationID={invocation_id}",
        "FragmentPath=/home/user/.config/systemd/user/gemma.service",
        "ExecStart={path=/usr/local/bin/gemma-router; argv[]=/usr/local/bin/gemma-router --port 8081; ignore_errors=no; start_time=[n/a]}",
    ]
    if extra_port is not None:
        lines.append(f"ExecStart={extra_port}")
    return "\n".join(lines) + "\n"


def _service_status_sha256() -> str:
    return hashlib.sha256(_service_status_text().encode("utf-8")).hexdigest()


def _journal_line(
    cursor: str,
    message: str,
    *,
    boot_id: str = _SVC_BOOT_ID,
    invocation_id: str = _SVC_INVOCATION_ID,
) -> str:
    return json.dumps(
        {
            "__CURSOR": cursor,
            "_BOOT_ID": boot_id,
            "_SYSTEMD_INVOCATION_ID": invocation_id,
            "MESSAGE": message,
        },
        sort_keys=True,
    )


def _baseline_journal_output() -> str:
    """Baseline journal (oldest -> newest): the sleeping state event (i=9), an
    intermediate heartbeat (i=f), and a final heartbeat (i=10) so the high-water
    cursor is the last record and is distinct from the state-event cursor."""
    return "\n".join(
        [
            _journal_line(
                _SVC_STATE_EVENT_CURSOR,
                f"[32831] cmd_child_to_router:state:{_SVC_STATE_PAYLOAD}",
            ),
            _journal_line(_SVC_MIDDLE_CURSOR, "gemma-router heartbeat"),
            _journal_line(_SVC_HIGH_WATER_CURSOR, "gemma-router heartbeat"),
        ]
    ) + "\n"


def _barrier_fields() -> dict:
    """The frozen active-service barrier fields for a clean quiescent service."""
    return {
        "active_service_status_sha256": _service_status_sha256(),
        "active_service_quiescent": True,
        "active_service_boot_id": _SVC_BOOT_ID,
        "active_service_invocation_id": _SVC_INVOCATION_ID,
        "active_service_main_pid": "32831",
        "active_service_control_group": (
            "/user.slice/user-1000.slice/user@1000.service/app.slice/gemma.service"
        ),
        "active_service_high_water_cursor": _SVC_HIGH_WATER_CURSOR,
        "active_service_state_event_cursor": _SVC_STATE_EVENT_CURSOR,
        "active_service_state_event_hash": hashlib.sha256(
            _SVC_STATE_PAYLOAD.encode("utf-8")
        ).hexdigest(),
        "active_service_state": "sleeping",
        "active_service_mutated": False,
    }


def _valid_lifecycle_receipt() -> dict:
    """A self-hashed lifecycle-stress receipt satisfying _validate_lifecycle_receipt."""
    payload = {
        "schema_version": "m3-unbrowser-lifecycle-stress-v1",
        "checked_at": "2026-08-16T00:00:00+00:00",
        "wait_seconds": 36.0,
        "elapsed_seconds": 36.5,
        "fixture_url": "http://127.0.0.1:18090/tasks/single_page_extraction/easy",
        "navigation_status": 200,
        "post_wait_observation_sha256": "a" * 64,
        "runtime_version": _MANIFEST["runtime_pins"]["unbrowser_version"],
        "confined": True,
        "same_session": True,
        "passed": True,
    }
    return {**payload, "receipt_hash": _canonical_hash(payload)}


def _conformance_receipt_fixture() -> dict:
    """A structurally valid no-real-model Pi conformance receipt (no Pi run)."""
    return build_pi_conformance_receipt(
        pi_identity={
            "path": "/opt/homebrew/bin/pi",
            "sha256": _RUNTIME_PINS["pi_cli_sha256"],
            "version": _RUNTIME_PINS["pi_version"],
        },
        list_models_rc=0,
        list_models_stdout=(
            "provider            model                         context  max-out  thinking  images\n"
            f"{RUN_PROVIDER:<20} {RUN_MODEL_ALIAS:<30}  65.5K    8.2K     no        no\n"
        ),
        list_models_stderr="",
        streaming_stub={
            "requests": [
                {
                    "path": "/v1/chat/completions",
                    "auth": f"Bearer {DUMMY_PROVIDER_API_KEY}",
                    "model": RUN_MODEL_ALIAS,
                    "stream": True,
                }
            ],
            "rc": 0,
            "stdout": "PYREPLAB-PROMPT-ONLY-CONFORMANCE-SENTINEL",
            "stderr": "",
            "config_sha256": "a" * 64,
        },
    )


def _build_artifacts():
    registry = build_prompt_only_registry()
    manifest = build_manifest(
        registry, REMOTE_IDENTITY, registry_file="registry.json"
    )
    local_preflight = build_local_preflight(
        manifest,
        registry,
        PROJECT_ROOT,
        _RUN_ROOT,
        simulator_draws=20,
        pi_conformance_receipt=_conformance_receipt_fixture(),
    )
    source = source_tree_hash(PROJECT_ROOT)
    remote_payload = _remote_preflight_payload(
        manifest, registry, local_preflight, source
    )
    remote_preflight = {
        **remote_payload,
        "preflight_hash": _canonical_hash(remote_payload),
    }
    return registry, manifest, local_preflight, remote_preflight, source


def _remote_preflight_payload(manifest, registry, local_preflight, source):
    from pyreplab_harness.m3_prompt_only_execution import models_json_sha256

    pins = manifest["runtime_pins"]
    return {
        "schema_version": REMOTE_PREFLIGHT_SCHEMA_VERSION,
        "screen_id": manifest["screen_id"],
        "execution_generation": EXECUTION_GENERATION,
        "manifest_hash": manifest["manifest_hash"],
        "registry_hash": registry.registry_hash,
        "local_preflight_hash": local_preflight["preflight_hash"],
        "source_tree_hash": source,
        "source_bundle_hash": local_preflight["source_bundle_hash"],
        "source_bundle_manifest": local_preflight["source_bundle_manifest"],
        "bundle_read_only": True,
        "git_available": True,
        "worktree_clean": True,
        "code_revision": "abc123",
        "pi_sha256": pins["pi_cli_sha256"],
        "pi_version": pins["pi_version"],
        "unbrowser_sha256": pins["unbrowser_sha256"],
        "unbrowser_version": pins["unbrowser_version"],
        "model_sha256": pins["model_artifact_sha256"],
        "server_sha256": pins["llama_server_sha256"],
        "server_version": pins["llama_server_version"],
        "server_help_sha256": "e" * 64,
        "off_server_argv_hash": manifest["isolated_no_cache_server_identity"][
            "server_argv_hash"
        ],
        **_barrier_fields(),
        "remote_server_port_free": True,
        "local_port_availability": {"18083": True, "18084": True},
        "lifecycle_receipt": None,
        "provider_config": {
            "provider": RUN_PROVIDER,
            "model_alias": RUN_MODEL_ALIAS,
            "pi_base_url": RUN_PI_BASE_URL,
            "models_json_sha256": models_json_sha256(),
            "api_key_binding": dummy_api_key_binding(),
        },
        "probe_mode": "no_model_identity_and_port_checks_only",
        "model_loaded_or_invoked": False,
        "live_model_execution_authorized": False,
        "slot_action_directory_absent": True,
        "generation_lease_absent": True,
        "checks": {
            "source_bundle_parity": True,
            "bundle_read_only": True,
            "project_content_addressed": True,
            "identity_digests_present": True,
            "help_flags_present": True,
            "off_argv_identity": True,
            "off_config_valid": True,
            "active_service_quiescent": True,
            "remote_server_port_free": True,
            "local_ports_free": True,
            "provider_config_hash": True,
            "lifecycle_stress": True,
            "no_model_invoked": True,
            "slot_action_directory_absent": True,
            "generation_lease_absent": True,
        },
        "ready_for_authorization": True,
    }


ARTIFACTS = _build_artifacts()
_REGISTRY, _MANIFEST, _LOCAL, _REMOTE, _SOURCE = ARTIFACTS
_POLICY_BY_ARM = {
    arm: policy_spec_from_treatment(_REGISTRY.by_id(arm)).to_dict() for arm in ARMS
}
_BUNDLE_BY_ARM = {arm: _REGISTRY.by_id(arm).bundle_id for arm in ARMS}
_SAMPLING_PARAMS = _MANIFEST["runtime_pins"]["sampling"]["parameters"]
_SIMULATOR_REPORT_HASH = _LOCAL["simulator_report"]["report_hash"]


def _finalize(payload: dict, field: str) -> dict:
    payload[field] = _canonical_hash(
        {key: value for key, value in payload.items() if key != field}
    )
    return payload


def _valid_slot_dir_observation_receipt() -> dict:
    payload = {
        "schema_version": SLOT_ACTION_DIR_OBSERVATION_SCHEMA_VERSION,
        "path": SLOT_ACTION_DIRECTORY,
        "mode": "555",
        "owner_uid": "1000",
        "owner_gid": "1000",
        "empty": True,
        "erase_only_feature_gate_exception": True,
        "native_persistence_forbidden": True,
        "observed_at": "2026-08-15T00:00:00+00:00",
    }
    return {**payload, "receipt_hash": _canonical_hash(payload)}


def _valid_slot_dir_preparation_receipt() -> dict:
    payload = {
        "schema_version": SLOT_ACTION_DIR_PREPARATION_SCHEMA_VERSION,
        "path": SLOT_ACTION_DIRECTORY,
        "mode": "555",
        "owner_uid": "1000",
        "owner_gid": "1000",
        "empty": True,
        "erase_only_feature_gate_exception": True,
        "native_persistence_forbidden": True,
        "created_at": "2026-08-15T00:00:00+00:00",
    }
    return {**payload, "receipt_hash": _canonical_hash(payload)}


def _valid_slot_dir_removal_receipt() -> dict:
    payload = {
        "schema_version": SLOT_ACTION_DIR_REMOVAL_SCHEMA_VERSION,
        "path": SLOT_ACTION_DIRECTORY,
        "mode": "555",
        "owner_uid": "1000",
        "owner_gid": "1000",
        "empty": True,
        "erase_only_feature_gate_exception": True,
        "native_persistence_forbidden": True,
        "removed": True,
        "removed_via": "rmdir",
        "absence_verified": True,
        "mode_verified": "0555",
        "empty_verified": True,
        "removed_at": "2026-08-15T00:00:01+00:00",
    }
    return {**payload, "receipt_hash": _canonical_hash(payload)}


# Shared 64-hex authorization hash for the standalone lease receipt fixtures
# (runner tests bind their own per-run authorization hash).
_LEASE_AUTH_HASH = "ab" * 32


def _valid_generation_lease_acquire_receipt(
    authorization_hash: str = _LEASE_AUTH_HASH,
) -> dict:
    from pyreplab_harness.m3_prompt_only_execution import generation_lease_remote_path

    payload = {
        "schema_version": GENERATION_LEASE_ACQUIRE_SCHEMA_VERSION,
        "path": generation_lease_remote_path(),
        "authorization_hash": authorization_hash,
        "mode": "555",
        "owner_uid": "1000",
        "owner_gid": "1000",
        "empty": True,
        "acquired_at": "2026-08-15T00:00:00+00:00",
    }
    return {**payload, "receipt_hash": _canonical_hash(payload)}


def _valid_generation_lease_release_receipt(
    authorization_hash: str = _LEASE_AUTH_HASH,
    *,
    acquire_receipt_hash: str = "c" * 64,
) -> dict:
    from pyreplab_harness.m3_prompt_only_execution import generation_lease_remote_path

    payload = {
        "schema_version": GENERATION_LEASE_RELEASE_SCHEMA_VERSION,
        "path": generation_lease_remote_path(),
        "authorization_hash": authorization_hash,
        "acquire_receipt_hash": acquire_receipt_hash,
        "released": True,
        "released_via": "rmdir",
        "absence_verified": True,
        "released_at": "2026-08-15T00:00:01+00:00",
    }
    return {**payload, "receipt_hash": _canonical_hash(payload)}


def _valid_local_generation_lease_acquire_receipt(
    authorization_hash: str = _LEASE_AUTH_HASH,
    *,
    path: str | None = None,
    lock_content_sha256: str = "d" * 64,
) -> dict:
    payload = {
        "schema_version": GENERATION_LEASE_LOCAL_ACQUIRE_SCHEMA_VERSION,
        "path": path or str(generation_lease_local_lock_path(PROJECT_ROOT)),
        "authorization_hash": authorization_hash,
        "mode": "600",
        "lock_content_sha256": lock_content_sha256,
        "acquired_at": "2026-08-15T00:00:00+00:00",
    }
    return {**payload, "receipt_hash": _canonical_hash(payload)}


def _valid_local_generation_lease_release_receipt(
    authorization_hash: str = _LEASE_AUTH_HASH,
    *,
    acquire_receipt_hash: str = "c" * 64,
    path: str | None = None,
    lock_content_sha256: str = "d" * 64,
) -> dict:
    payload = {
        "schema_version": GENERATION_LEASE_LOCAL_RELEASE_SCHEMA_VERSION,
        "path": path or str(generation_lease_local_lock_path(PROJECT_ROOT)),
        "authorization_hash": authorization_hash,
        "acquire_receipt_hash": acquire_receipt_hash,
        "lock_content_sha256": lock_content_sha256,
        "released": True,
        "released_via": "unlink",
        "absence_verified": True,
        "released_at": "2026-08-15T00:00:01+00:00",
    }
    return {**payload, "receipt_hash": _canonical_hash(payload)}


def _lease_receipts_for(authorization_hash: str) -> tuple[dict, dict, dict, dict]:
    """Coherent local+remote acquire/release receipt set bound to a hash.

    Returns (local_acquire, remote_acquire, local_release, remote_release)
    with every release bound to its corresponding acquire receipt hash.
    """
    local_acquire = _valid_local_generation_lease_acquire_receipt(authorization_hash)
    remote_acquire = _valid_generation_lease_acquire_receipt(authorization_hash)
    local_release = _valid_local_generation_lease_release_receipt(
        authorization_hash,
        acquire_receipt_hash=local_acquire["receipt_hash"],
        lock_content_sha256=local_acquire["lock_content_sha256"],
    )
    remote_release = _valid_generation_lease_release_receipt(
        authorization_hash,
        acquire_receipt_hash=remote_acquire["receipt_hash"],
    )
    return local_acquire, remote_acquire, local_release, remote_release


def _release_outcome_for(authorization_hash: str) -> dict:
    """A successful structured release outcome bound to a hash."""
    _, _, local_release, remote_release = _lease_receipts_for(authorization_hash)
    return {
        "remote_receipt": remote_release,
        "local_receipt": local_release,
        "error": None,
        "remote_released": True,
        "local_released": True,
        "quarantine_retained": False,
    }


class _FakeLeaseSSH:
    """In-memory remote directory state machine for lease/slot-dir transports.

    Tracks ``test ! -e/-L``, ``mkdir``, ``chmod``, ``id -u/-g``, ``stat``,
    ``find`` and ``rmdir`` for arbitrary paths so the real acquire/prepare/
    release builders run model-free (never touching a real network).
    """

    def __init__(self) -> None:
        self.paths: dict[str, dict[str, Any]] = {}
        self.commands: list[list[str]] = []
        self.fail_rmdir: set[str] = set()
        self.fail_mkdir: set[str] = set()

    def __call__(self, command: list[str]) -> str:
        self.commands.append(command)
        cmd = command[0]
        if cmd == "test" and command[1] == "!":
            path = command[-1]
            if path in self.paths:
                raise RuntimeError(f"path exists: {path}")
            return ""
        if cmd == "mkdir":
            if command[1] in self.fail_mkdir:
                raise RuntimeError(f"mkdir failed: {command[1]}")
            self.paths[command[1]] = {
                "type": "directory",
                "mode": "0755",
                "uid": "1000",
                "gid": "1000",
                "entries": set(),
            }
            return ""
        if cmd == "chmod":
            self.paths[command[2]]["mode"] = command[1]
            return ""
        if cmd == "id" and command[1] == "-u":
            return "1000\n"
        if cmd == "id" and command[1] == "-g":
            return "1000\n"
        if cmd == "stat":
            # stat -c <format> <path> (format at [2], path at [3])
            path = command[3] if len(command) > 3 else command[2]
            info = self.paths.get(path)
            if info is None:
                raise RuntimeError(f"no such path: {path}")
            return f"{info['type']}|{info['mode']}|{info['uid']}|{info['gid']}\n"
        if cmd == "find":
            info = self.paths.get(command[1])
            return "x\n" if info and info["entries"] else ""
        if cmd == "rmdir":
            if command[1] in self.fail_rmdir:
                raise RuntimeError(f"rmdir failed: {command[1]}")
            info = self.paths.get(command[1])
            if info is None or info["entries"]:
                raise RuntimeError(f"rmdir failed: {command[1]}")
            del self.paths[command[1]]
            return ""
        raise AssertionError(f"unexpected command: {command}")


def _valid_slot_clear_receipt() -> dict:
    before = _valid_slot_dir_observation_receipt()
    after = _valid_slot_dir_observation_receipt()
    payload = {
        "schema_version": "m3-prompt-only-slot-clear-receipt-v3",
        "source_commit": "b4d6c7d8ff69c2e05e4e8ee7e6e710a08abd7b45",
        "slot_id": 0,
        "method": "POST",
        "path": "/slots/0",
        "query": "action=erase",
        "action": "erase",
        "before_slots": [
            {"id": 0, "n_ctx": 65536, "speculative": False, "is_processing": False}
        ],
        "after_slots": [
            {"id": 0, "n_ctx": 65536, "speculative": False, "is_processing": False}
        ],
        "response_status": 200,
        "response_id_slot": 0,
        "response_n_erased": 0,
        "slot_action_dir_before_receipt": before,
        "slot_action_dir_after_receipt": after,
        "slot_action_dir_before_receipt_hash": before["receipt_hash"],
        "slot_action_dir_after_receipt_hash": after["receipt_hash"],
        "cleared": True,
    }
    return {**payload, "receipt_hash": _canonical_hash(payload)}


def _valid_proxy_receipt() -> dict:
    from pyreplab_harness.cache_proxy import CACHE_PROXY_RECEIPT_SCHEMA_VERSION

    payload = {
        "schema_version": CACHE_PROXY_RECEIPT_SCHEMA_VERSION,
        "attempt_id": "ppo-1",
        "panel_id": "cell-1",
        "pair_id": "cell-1",
        "sampling_seed": 1,
        "cache_runtime_receipt_hash": "0" * 64,
        "provider_turn": 1,
        "cache_mode": "off",
        "slot_identity": 0,
        "request_path": "/v1/completions",
        "incoming_request_sha256": "a" * 64,
        "logical_request_sha256": "b" * 64,
        "forwarded_request_sha256": "c" * 64,
        "cache_prompt_injected": False,
        "slot_identity_injected": True,
        "response_status": 200,
        "response_sha256": "d" * 64,
        "response_bytes": 100,
        "transport_first_byte_seconds": 0.1,
        "transport_total_seconds": 0.2,
        "server_mechanics": {
            "timings": {
                "cache_n": 0,
                "prompt_n": 10,
                "prompt_ms": 5.0,
                "predicted_n": 5,
                "predicted_ms": 6.0,
            },
            "usage_cached_tokens": 0,
            "mechanics_valid": True,
            "invalidation_codes": [],
        },
        "mechanics_valid": True,
        "invalidation_codes": [],
        "raw_request_persisted": False,
        "raw_response_persisted": False,
        "authorization_header_persisted": False,
        "started_at": "2026-08-15T00:00:00+00:00",
        "finished_at": "2026-08-15T00:00:01+00:00",
    }
    return {**payload, "receipt_hash": _canonical_hash(payload)}


def _valid_server_receipt() -> dict:
    identity = _MANIFEST["isolated_no_cache_server_identity"]
    payload = {
        "schema_version": "m3-prompt-only-server-lifecycle-receipt-v1",
        "mode": "off",
        "pid": 1,
        "process_group": 1,
        "server_argv": identity["server_argv"],
        "server_argv_hash": identity["server_argv_hash"],
        "run_log_path": "/tmp/off.log",
        "listener_ownership": {
            "port": 18082,
            "pid": 1,
            "process_group": 1,
            "verified": True,
        },
        "active_service_touched": False,
        "launched_at": "2026-08-15T00:00:00+00:00",
    }
    return {**payload, "receipt_hash": _canonical_hash(payload)}


def _valid_tunnel_receipt() -> dict:
    payload = {
        "schema_version": "m3-prompt-only-tunnel-lifecycle-receipt-v1",
        "pid": 2,
        "process_group": 2,
        "local_port": 18084,
        "remote_target": "127.0.0.1:18082",
        "launched_at": "2026-08-15T00:00:00+00:00",
    }
    return {**payload, "receipt_hash": _canonical_hash(payload)}


def _script_run_marker(script: str) -> str | None:
    """Extract the PYREPLAB_RUN_MARKER bound into a launch script."""
    match = re.search(r"PYREPLAB_RUN_MARKER=([0-9a-f]{64})", script)
    return match.group(1) if match else None


def _advance_run(run, stage: str):
    """Advance a validated run's fresh permit through the state machine."""
    from pyreplab_harness.m3_prompt_only_execution import (
        _LifecyclePermit,
        _transition_stage,
        _write_consumed_marker,
    )

    run = replace(run, _permit=_LifecyclePermit(run.authorization_hash))
    if stage == "validated":
        return run
    if stage == "revalidated":
        _transition_stage(run, "validated", "revalidated")
        return run
    # Any stage at/after consumption requires a durable consumed marker on disk
    # bound to the permit.
    marker_path = run.paths["consumed"]
    if marker_path.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    else:
        marker = _write_consumed_marker(marker_path, run.authorization_hash)
    run._permit.consumed_marker_hash = marker["consumed_hash"]
    run._permit.consumed_marker_path = marker_path
    _transition_stage(run, "validated", "revalidated")
    _transition_stage(run, "revalidated", "consumed")
    if stage == "consumed":
        return run
    _transition_stage(run, "consumed", "launching")
    if stage == "launching":
        return run
    _transition_stage(run, "launching", "active")
    if stage == "active":
        return run
    _transition_stage(run, "active", "teardown")
    if stage == "teardown":
        return run
    _transition_stage(run, "teardown", "closed")
    return run


def _test_run(stage: str = "launching"):
    """Return a fresh-permit copy of the validated run advanced to ``stage``."""
    return _advance_run(_VALIDATED_RUN, stage)


def _test_run_with_remote(remote_preflight, stage: str = "active"):
    """Return a fresh validated run with a custom remote preflight at ``stage``."""
    return _advance_run(_build_validated_run(remote_preflight=remote_preflight), stage)


def _custom_remote_preflight(active_service_status_sha256: str) -> dict:
    """A full valid remote preflight with a custom active-service status hash."""
    payload = _remote_preflight_payload(_MANIFEST, _REGISTRY, _LOCAL, _SOURCE)
    payload["active_service_status_sha256"] = active_service_status_sha256
    return {
        **payload,
        "preflight_hash": _canonical_hash(payload),
    }


def _valid_readiness_receipt(attempts: int = 1) -> dict:
    payload = {
        "schema_version": "m3-prompt-only-readiness-receipt-v2",
        "server_alias": RUN_MODEL_ALIAS,
        "idle_slot_0": True,
        "attempts": attempts,
        "verified": True,
    }
    return {**payload, "receipt_hash": _canonical_hash(payload)}


def _valid_active_service_receipt(
    status_sha256: str | None = None,
    *,
    quiescent: bool = True,
    mutated: bool = False,
) -> dict:
    payload = {
        "schema_version": "m3-prompt-only-active-service-receipt-v1",
        "status_sha256": status_sha256 if status_sha256 is not None else _service_status_sha256(),
        "quiescent": quiescent,
        "boot_id": _SVC_BOOT_ID,
        "invocation_id": _SVC_INVOCATION_ID,
        "main_pid": "32831",
        "control_group": (
            "/user.slice/user-1000.slice/user@1000.service/app.slice/gemma.service"
        ),
        "high_water_cursor": _SVC_HIGH_WATER_CURSOR,
        "state_event_cursor": _SVC_STATE_EVENT_CURSOR,
        "state_event_hash": hashlib.sha256(_SVC_STATE_PAYLOAD.encode("utf-8")).hexdigest(),
        "state": "sleeping",
        "mutated": mutated,
    }
    return {**payload, "receipt_hash": _canonical_hash(payload)}


def _valid_teardown_receipt(
    active_service_after: dict | None = None,
    *,
    verified: bool = True,
    local_processes_exited: bool = True,
    remote_process_dead: bool = True,
    remote_port_released: bool = True,
    remote_pid_file_removed: bool = True,
    active_service_unchanged: bool = True,
    errors: list | None = None,
    slot_action_dir_required: bool = True,
    slot_action_dir_removed: bool = True,
    slot_action_dir_absence_verified: bool = True,
    remote_log_evidence: dict | None = None,
) -> dict:
    after = active_service_after or _valid_active_service_receipt()
    removal_receipt = (
        _valid_slot_dir_removal_receipt()
        if slot_action_dir_removed and slot_action_dir_absence_verified
        else None
    )
    payload = {
        "schema_version": "m3-prompt-only-teardown-receipt-v2",
        "verified": verified,
        "local_processes_exited": local_processes_exited,
        "remote_process_dead": remote_process_dead,
        "remote_port_released": remote_port_released,
        "remote_pid_file_removed": remote_pid_file_removed,
        "remote_log_evidence": remote_log_evidence
        if remote_log_evidence is not None
        else {"path": "/tmp/off.log", "sha256": "e" * 64, "size": 100},
        "active_service_after_receipt_hash": after["receipt_hash"],
        "active_service_unchanged": active_service_unchanged,
        "slot_action_dir_required": slot_action_dir_required,
        "slot_action_dir_removed": slot_action_dir_removed,
        "slot_action_dir_absence_verified": slot_action_dir_absence_verified,
        "slot_action_dir_removal_receipt": removal_receipt,
        "slot_action_dir_removal_receipt_hash": (
            removal_receipt["receipt_hash"] if removal_receipt is not None else ""
        ),
        "errors": errors if errors is not None else [],
        "finished_at": "2026-08-15T00:00:01+00:00",
    }
    return {**payload, "receipt_hash": _canonical_hash(payload)}


def _assemble_probe_authorization(request: dict) -> tuple[dict, str]:
    """Assemble a valid endpoint-probe authorization from a probe request."""
    now = datetime.now(timezone.utc)
    payload = {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "authorization_id": f"probe-auth-{uuid4().hex[:12]}",
        "screen_id": request["screen_id"],
        "execution_generation": EXECUTION_GENERATION,
        "manifest_hash": request["manifest_hash"],
        "registry_hash": request["registry_hash"],
        "local_preflight_hash": request["local_preflight_hash"],
        "remote_preflight_hash": request["remote_preflight_hash"],
        "simulator_report_hash": request["simulator_report_hash"],
        "source_tree_hash": request["source_tree_hash"],
        "source_bundle_hash": request["source_bundle_hash"],
        "remote_identity": request["remote_identity"],
        "provider_config": request["provider_config"],
        "python_executable": request["python_executable"],
        "result_filename": request["result_filename"],
        "result_path": request["result_path"],
        "max_cells": 0,
        "max_panels": 0,
        "budget": request["budget"],
        "server_lifecycle": request["server_lifecycle"],
        "severe_veto_contract": request["severe_veto_contract"],
        "approved_by": "test-operator",
        "approved_at": now.isoformat(),
        "expires_at": (now + timedelta(seconds=3600)).isoformat(),
        "authorization_statement": PROBE_AUTHORIZATION_STATEMENT,
        "live_model_execution_authorized": True,
        "server_launch_authorized": True,
        "task_inference_authorized": False,
        "single_use": True,
        "authorization_scope": "endpoint_probe",
        "endpoint_probe_receipt_hash": None,
        "endpoint_probe_authorization_hash": None,
    }
    authorization_hash = _canonical_hash(payload)
    return {**payload, "authorization_hash": authorization_hash}, authorization_hash


_PROBE_RECEIPT_DIR = tempfile.mkdtemp(prefix="pyreplab-ppo-probe-")
# The receipt must bind the RESOLVED path (the validator resolves the expected
# path, e.g. /var -> /private/var on macOS).
_PROBE_RECEIPT_PATH = (
    Path(_PROBE_RECEIPT_DIR).expanduser().resolve() / "endpoint-probe-receipt.json"
)


def _valid_probe_claim(authorization_hash: str) -> dict:
    payload = {
        "schema_version": "m3-prompt-only-claim-v1",
        "authorization_hash": authorization_hash,
        "result_path": str(_PROBE_RECEIPT_PATH),
        "result_filename": _PROBE_RECEIPT_PATH.name,
        "controller_pid": 4242,
        "created_at": "2026-08-15T00:00:00+00:00",
    }
    return {**payload, "claim_hash": _canonical_hash(payload)}


def _valid_probe_consumed_marker(authorization_hash: str) -> dict:
    payload = {
        "schema_version": "m3-prompt-only-consumed-v1",
        "authorization_hash": authorization_hash,
        "consumed_at": "2026-08-15T00:00:01+00:00",
    }
    return {**payload, "consumed_hash": _canonical_hash(payload)}


def _valid_probe_receipt() -> dict:
    """A self-hashed endpoint-probe receipt bound to the frozen artifacts."""
    identity = _MANIFEST["isolated_no_cache_server_identity"]
    probe_request = build_endpoint_probe_request(
        _MANIFEST,
        _REGISTRY,
        _LOCAL,
        _REMOTE,
        project_root=PROJECT_ROOT,
        result_path=_PROBE_RECEIPT_PATH,
    )
    probe_authorization, auth_hash = _assemble_probe_authorization(probe_request)
    claim = _valid_probe_claim(auth_hash)
    consumed_marker = _valid_probe_consumed_marker(auth_hash)
    (
        local_lease_acquire,
        remote_lease_acquire,
        local_lease_release,
        remote_lease_release,
    ) = _lease_receipts_for(auth_hash)
    payload = {
        "schema_version": PROBE_RECEIPT_SCHEMA_VERSION,
        "screen_id": _MANIFEST["screen_id"],
        "authorization_hash": auth_hash,
        "manifest_hash": _MANIFEST["manifest_hash"],
        "registry_hash": _REGISTRY.registry_hash,
        "local_preflight_hash": _LOCAL["preflight_hash"],
        "remote_preflight_hash": _REMOTE["preflight_hash"],
        "source_tree_hash": _SOURCE,
        "source_bundle_hash": _LOCAL["source_bundle_hash"],
        "server_argv_hash": identity["server_argv_hash"],
        "server_argv": identity["server_argv"],
        "passed": True,
        "server_startup_warmup_permitted": True,
        "task_inference_invoked": False,
        "task_completion_chat_requests": 0,
        "result_filename": _PROBE_RECEIPT_PATH.name,
        "result_path": str(_PROBE_RECEIPT_PATH),
        "endpoint_trace": [
            {"method": "GET", "path": "/slots", "query": "", "status": 200, "error": None},
            {"method": "GET", "path": "/v1/models", "query": "", "status": 200, "error": None},
            {"method": "GET", "path": "/slots", "query": "", "status": 200, "error": None},
            {"method": "POST", "path": "/slots/0", "query": "action=erase", "status": 200, "error": None},
            {"method": "GET", "path": "/slots", "query": "", "status": 200, "error": None},
        ],
        "evidence": {
            "probe_authorization": probe_authorization,
            "readiness_receipt": _valid_readiness_receipt(),
            "slot_clear_receipt": _valid_slot_clear_receipt(),
            "server_receipt": _valid_server_receipt(),
            "tunnel_receipt": _valid_tunnel_receipt(),
            "teardown_receipt": _valid_teardown_receipt(),
            "active_service_after": _valid_active_service_receipt(),
            "slot_action_dir_preparation_receipt": _valid_slot_dir_preparation_receipt(),
            "slot_action_dir_removal_receipt": _valid_slot_dir_removal_receipt(),
            "generation_lease_acquire_receipt": remote_lease_acquire,
            "generation_lease_release_receipt": remote_lease_release,
            "generation_lease_local_acquire_receipt": local_lease_acquire,
            "generation_lease_local_release_receipt": local_lease_release,
            "claim": claim,
            "consumed_marker": consumed_marker,
            "claim_hash": claim["claim_hash"],
            "consumed_marker_hash": consumed_marker["consumed_hash"],
        },
        "completed_at": (
            datetime.now(timezone.utc) + timedelta(seconds=2)
        ).isoformat(),
    }
    return {**payload, "receipt_hash": _canonical_hash(payload)}


_PROBE_RECEIPT = _valid_probe_receipt()
_PROBE_RECEIPT_PATH.write_text(
    json.dumps(_PROBE_RECEIPT, sort_keys=True), encoding="utf-8"
)
_PROBE_AUTHORIZATION = _PROBE_RECEIPT["evidence"]["probe_authorization"]
_PROBE_AUTHORIZATION_HASH = _PROBE_RECEIPT["authorization_hash"]
_PROBE_AUTHORIZATION_PATH = Path(_PROBE_RECEIPT_DIR) / "endpoint-probe-authorization.json"
_PROBE_AUTHORIZATION_PATH.write_text(
    json.dumps(_PROBE_AUTHORIZATION, sort_keys=True), encoding="utf-8"
)


def _make_request(
    result_path: Path | None = None,
    probe_receipt_path: str | Path | None = None,
) -> dict:
    result = result_path or (PROJECT_ROOT / ".runs" / RESULT_FILENAME)
    return build_authorization_request(
        _MANIFEST,
        _REGISTRY,
        _LOCAL,
        _REMOTE,
        project_root=PROJECT_ROOT,
        result_path=result,
        endpoint_probe_receipt_path=probe_receipt_path or _PROBE_RECEIPT_PATH,
        endpoint_probe_authorization_path=_PROBE_AUTHORIZATION_PATH,
        expected_endpoint_probe_authorization_hash=_PROBE_AUTHORIZATION_HASH,
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
        "execution_generation": EXECUTION_GENERATION,
        "manifest_hash": request["manifest_hash"],
        "registry_hash": request["registry_hash"],
        "local_preflight_hash": request["local_preflight_hash"],
        "remote_preflight_hash": request["remote_preflight_hash"],
        "simulator_report_hash": request["simulator_report_hash"],
        "source_tree_hash": request["source_tree_hash"],
        "source_bundle_hash": request["source_bundle_hash"],
        "remote_identity": request["remote_identity"],
        "provider_config": request["provider_config"],
        "python_executable": request["python_executable"],
        "result_filename": request["result_filename"],
        "result_path": request["result_path"],
        "max_cells": MAX_CELLS,
        "max_panels": MAX_PANELS,
        "budget": request["budget"],
        "server_lifecycle": request["server_lifecycle"],
        "severe_veto_contract": request["severe_veto_contract"],
        "approved_by": approved_by,
        "approved_at": approved_at or now.isoformat(),
        "expires_at": expires_at
        or (now + timedelta(seconds=expires_seconds)).isoformat(),
        "authorization_statement": AUTHORIZATION_STATEMENT,
        "live_model_execution_authorized": True,
        "server_launch_authorized": True,
        "task_inference_authorized": True,
        "single_use": True,
        "authorization_scope": "pilot",
        "endpoint_probe_receipt_hash": request["endpoint_probe_receipt_hash"],
        "endpoint_probe_authorization_hash": request[
            "endpoint_probe_authorization_hash"
        ],
    }
    authorization_hash = _canonical_hash(payload)
    return {**payload, "authorization_hash": authorization_hash}, authorization_hash


def _validate(authorization, expected_hash, *, result_path=None, source=None):
    result = result_path or (PROJECT_ROOT / ".runs" / RESULT_FILENAME)
    return validate_execution_authorization(
        authorization,
        expected_authorization_hash=expected_hash,
        manifest_hash=_MANIFEST["manifest_hash"],
        registry_hash=_REGISTRY.registry_hash,
        local_preflight_hash=_LOCAL["preflight_hash"],
        remote_preflight_hash=_REMOTE["preflight_hash"],
        simulator_report_hash=_SIMULATOR_REPORT_HASH,
        source_tree_hash=_SOURCE if source is None else source,
        source_bundle_hash=_LOCAL["source_bundle_hash"],
        remote_identity=_MANIFEST["remote_identity"],
        result_filename=RESULT_FILENAME,
        result_path=result,
    )


_VALIDATED_TMP = tempfile.mkdtemp(prefix="pyreplab-ppo-validated-")


def _build_validated_run(remote_preflight=None, root=None):
    root = Path(root) if root is not None else Path(tempfile.mkdtemp(prefix="pyreplab-ppo-validated-"))
    registry_path = root / "registry.json"
    manifest_path = root / "manifest.json"
    local_path = root / "local.json"
    remote_path = root / "remote.json"
    auth_path = root / "authorization.json"
    result_path = root / RESULT_FILENAME
    _REGISTRY.save(registry_path)
    manifest_path.write_text(json.dumps(_MANIFEST), encoding="utf-8")
    local_path.write_text(json.dumps(_LOCAL), encoding="utf-8")
    remote_path.write_text(
        json.dumps(remote_preflight if remote_preflight is not None else _REMOTE),
        encoding="utf-8",
    )
    effective_remote = remote_preflight if remote_preflight is not None else _REMOTE
    request = build_authorization_request(
        _MANIFEST,
        _REGISTRY,
        _LOCAL,
        effective_remote,
        project_root=PROJECT_ROOT,
        result_path=result_path,
        endpoint_probe_receipt_path=_PROBE_RECEIPT_PATH,
        endpoint_probe_authorization_path=_PROBE_AUTHORIZATION_PATH,
        expected_endpoint_probe_authorization_hash=_PROBE_AUTHORIZATION_HASH,
    )
    authorization, auth_hash = _make_authorization(request)
    auth_path.write_text(json.dumps(authorization), encoding="utf-8")
    return _prepare_authorized_run(
        manifest_path,
        registry_path,
        local_path,
        remote_path,
        auth_path,
        auth_hash,
        result_path,
        RemoteConfig(**REMOTE_IDENTITY),
        pi_binary="pi",
        provider=RUN_PROVIDER,
        model=RUN_MODEL_ALIAS,
        thinking="off",
        unbrowser_binary=_MANIFEST["runtime_pins"]["unbrowser_path"],
        model_artifact=_MANIFEST["runtime_pins"]["model_artifact_path"],
        llama_server_binary=_MANIFEST["runtime_pins"]["llama_server_path"],
    )


_VALIDATED_RUN = _build_validated_run()


def _trace_entry(
    tool_name: str = "bash", *, is_error: bool = False, tool_call_id: str = ""
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
    arm: str = "E",
    *,
    success: bool = True,
    pi_return_code: int = 0,
    failure_code: str | None = None,
    tool_trace=None,
    pi_stderr: str = "",
    provider_turn_count: int = 1,
    synthetic_assistant_message_count: int = 0,
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
    budget_receipt = {
        "schema_version": _MANIFEST["event_accounting"][
            "budget_receipt_schema_version"
        ],
        "provider_turn_limit": 13,
        "provider_request_admissions": provider_turn_count,
        "provider_request_blocks": 0,
        "provider_gate_checks": provider_turn_count,
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
        "suppressed_tool_request_count": 0,
        "suppressed_tool_request_ids": [],
        "invariant_violations": [],
    }
    return {
        "attempt_id": attempt_id,
        "policy": _POLICY_BY_ARM[arm],
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
            "synthetic_assistant_message_count": synthetic_assistant_message_count,
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


def _one_cell_result(task, attempt, arm, registry_hash, manifest_hash, cell):
    from pyreplab_harness.m3_prompt_only_execution import _one_cell_result

    return _one_cell_result(
        {"id": task["task_id"]},
        attempt,
        _BUNDLE_BY_ARM[arm],
        registry_hash,
        _namespace(cell),
    )


def _namespace(cell):
    import argparse

    return argparse.Namespace(
        rollout_replica=int(str(cell["panel_id"]).rsplit("replica=", 1)[1]),
        sampling_seed=cell["sampling_seed"],
        pilot_manifest_hash=_MANIFEST["manifest_hash"],
        pilot_panel_id=cell["panel_id"],
    )


class RuntimeIdentityTest(unittest.TestCase):
    def test_manifest_routes_at_the_isolated_off_server(self) -> None:
        validate_runtime_identity(_MANIFEST)
        pins = _MANIFEST["runtime_pins"]
        self.assertEqual(pins["provider"], RUN_PROVIDER)
        self.assertEqual(pins["model_alias"], RUN_MODEL_ALIAS)
        self.assertEqual(pins["pi_provider_config"]["base_url"], RUN_PI_BASE_URL)
        self.assertEqual(
            pins["remote_provider_base_url"], RUN_REMOTE_SERVER_BASE_URL
        )
        self.assertEqual(RUN_PROVIDER, "prompt-pilot-gemma")
        self.assertEqual(RUN_MODEL_ALIAS, "gemma-4-26b-a4b-cache-canary")
        self.assertEqual(RUN_PI_BASE_URL, "http://127.0.0.1:18083/v1")
        self.assertEqual(RUN_REMOTE_SERVER_BASE_URL, "http://127.0.0.1:18082/v1")

    def test_off_server_identity_is_the_no_cache_mode(self) -> None:
        identity = _MANIFEST["isolated_no_cache_server_identity"]
        self.assertEqual(identity["mode"], "off")
        self.assertEqual(identity["model_alias"], RUN_MODEL_ALIAS)
        self.assertEqual(identity["port"], 18082)
        self.assertEqual(identity["server_argv"][-1], "--no-cache-prompt")

    def test_manifest_provider_drift_rejected(self) -> None:
        manifest = json.loads(json.dumps(_MANIFEST))
        manifest["runtime_pins"]["provider"] = "ubuntu-gemma"
        with self.assertRaisesRegex(ValueError, "provider drifted"):
            validate_runtime_identity(manifest)

    def test_manifest_default_endpoint_drift_rejected(self) -> None:
        manifest = json.loads(json.dumps(_MANIFEST))
        manifest["runtime_pins"]["pi_provider_config"]["base_url"] = (
            "http://127.0.0.1:18081/v1"
        )
        with self.assertRaisesRegex(ValueError, "base URL"):
            validate_runtime_identity(manifest)

    def test_models_json_is_credential_free_and_deterministic(self) -> None:
        content = build_frozen_models_json()
        serialized = json.dumps(content, sort_keys=True)
        for secret in ("apiKey", "api_key", "secret", "password", "token"):
            self.assertNotIn(secret, serialized)
        # Regression: Pi 0.84.1 rejects ``samplingParams: null`` ("must be
        # object") and then reports no models available; the frozen config
        # must omit the key entirely.
        self.assertNotIn("samplingParams", serialized)
        self.assertEqual(models_json_sha256(), models_json_sha256())
        self.assertEqual(
            build_frozen_models_json()["providers"][RUN_PROVIDER]["baseUrl"],
            RUN_PI_BASE_URL,
        )

    def test_dummy_key_binding_is_fixed_and_non_secret(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _frozen_provider_config,
        )

        binding = dummy_api_key_binding()
        self.assertEqual(binding["mode"], "fixed_dummy_non_secret")
        self.assertEqual(
            binding["key_sha256"],
            hashlib.sha256(DUMMY_PROVIDER_API_KEY.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(binding["length"], len(DUMMY_PROVIDER_API_KEY))
        # The literal never appears in any bound frozen artifact.
        self.assertNotIn(DUMMY_PROVIDER_API_KEY, json.dumps(_frozen_provider_config()))
        self.assertNotIn(DUMMY_PROVIDER_API_KEY, json.dumps(_MANIFEST["runtime_pins"]))
        # The production request binds the same mode/hash identity.
        request = _make_request()
        self.assertEqual(request["provider_config"]["api_key_binding"], binding)

    def test_write_frozen_models_json_is_fresh_and_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_dir = Path(directory) / "result-sibling.config"
            receipt = write_frozen_models_json(config_dir)
            self.assertEqual(receipt["models_json_sha256"], models_json_sha256())
            self.assertTrue((config_dir / "models.json").is_file())
            # A second write to the same dir fails closed.
            with self.assertRaises(FileExistsError):
                write_frozen_models_json(config_dir)

    def test_models_json_credential_injection_rejected(self) -> None:
        content = build_frozen_models_json()
        content["providers"][RUN_PROVIDER]["apiKey"] = "secret-value"
        with self.assertRaisesRegex(ValueError, "credential"):
            from pyreplab_harness.m3_prompt_only_execution import (
                _assert_models_json_has_no_credentials,
            )

            _assert_models_json_has_no_credentials(content)


class AuthorizationRequestTest(unittest.TestCase):
    def test_request_is_non_authorizing_and_binds_frozen_artifacts(self) -> None:
        request = _make_request()
        self.assertIs(request["live_model_execution_authorized"], False)
        self.assertEqual(request["manifest_hash"], _MANIFEST["manifest_hash"])
        self.assertEqual(request["registry_hash"], _REGISTRY.registry_hash)
        self.assertEqual(request["local_preflight_hash"], _LOCAL["preflight_hash"])
        self.assertEqual(request["remote_preflight_hash"], _REMOTE["preflight_hash"])
        self.assertEqual(request["simulator_report_hash"], _SIMULATOR_REPORT_HASH)
        self.assertEqual(request["source_tree_hash"], _SOURCE)
        self.assertEqual(request["remote_identity"], REMOTE_IDENTITY)
        self.assertEqual(request["max_cells"], EXPECTED_CELLS)
        self.assertEqual(request["max_panels"], EXPECTED_PANELS)
        self.assertEqual(request["budget"], _worst_case_budget())
        self.assertEqual(request["provider_config"]["provider"], RUN_PROVIDER)
        self.assertEqual(
            request["budget"]["total_provider_backed_turns"], 72 * 13
        )
        self.assertEqual(request["budget"]["total_output_tokens"], 72 * 13 * 4096)
        self.assertEqual(request["server_lifecycle"]["mode"], "off")
        self.assertEqual(request["authorization_scope"], "pilot")
        self.assertEqual(
            request["endpoint_probe_receipt_hash"], _PROBE_RECEIPT["receipt_hash"]
        )
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

    def test_validate_authorization_cli_passes_source_bundle_hash(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry_path = root / "registry.json"
            manifest_path = root / "manifest.json"
            local_path = root / "local.json"
            remote_path = root / "remote.json"
            authorization_path = root / "authorization.json"
            result_path = root / RESULT_FILENAME
            _REGISTRY.save(registry_path)
            for path, artifact in (
                (manifest_path, _MANIFEST),
                (local_path, _LOCAL),
                (remote_path, _REMOTE),
            ):
                path.write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")
            request = _make_request(result_path)
            authorization, authorization_hash = _make_authorization(request)
            authorization_path.write_text(
                json.dumps(authorization, sort_keys=True), encoding="utf-8"
            )

            with mock.patch("builtins.print"):
                result = main(
                    [
                        "validate-authorization",
                        "--authorization",
                        str(authorization_path),
                        "--authorization-hash",
                        authorization_hash,
                        "--scope",
                        "pilot",
                        "--manifest",
                        str(manifest_path),
                        "--registry",
                        str(registry_path),
                        "--local-preflight",
                        str(local_path),
                        "--remote-preflight",
                        str(remote_path),
                        "--root",
                        str(PROJECT_ROOT),
                        "--result",
                        str(result_path),
                        "--endpoint-probe-receipt",
                        str(_PROBE_RECEIPT_PATH),
                        "--endpoint-probe-authorization",
                        str(_PROBE_AUTHORIZATION_PATH),
                        "--expected-endpoint-probe-authorization-hash",
                        _PROBE_AUTHORIZATION_HASH,
                        "--pi",
                        "pi",
                    ]
                )
            self.assertEqual(result, 0)

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
            {**authorization, "budget": {"cells": 1}}, "authorization_hash"
        )
        with self.assertRaisesRegex(ValueError, "budget"):
            _validate(authorization, authorization["authorization_hash"])

    def test_rejects_max_cells_mismatch(self) -> None:
        request = _make_request()
        authorization, _ = _make_authorization(request)
        authorization = _finalize(
            {**authorization, "max_cells": EXPECTED_CELLS - 1}, "authorization_hash"
        )
        with self.assertRaisesRegex(ValueError, "max_cells"):
            _validate(authorization, authorization["authorization_hash"])

    def test_rejects_provider_config_drift(self) -> None:
        request = _make_request()
        authorization, _ = _make_authorization(request)
        config = dict(request["provider_config"])
        config["model_alias"] = "gemma-4-26b-a4b"
        authorization = _finalize(
            {**authorization, "provider_config": config}, "authorization_hash"
        )
        with self.assertRaisesRegex(ValueError, "provider config"):
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

    def test_rejects_result_path_mismatch(self) -> None:
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
        authorization = _finalize(
            {**authorization, "single_use": False}, "authorization_hash"
        )
        with self.assertRaisesRegex(ValueError, "single_use"):
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
        auth = "a" * 64
        first = deterministic_cell_attempt_id(auth, "cell-1", "E@v-hash")
        self.assertEqual(first, deterministic_cell_attempt_id(auth, "cell-1", "E@v-hash"))
        self.assertNotEqual(first, deterministic_cell_attempt_id("b" * 64, "cell-1", "E@v-hash"))
        self.assertNotEqual(first, deterministic_cell_attempt_id(auth, "cell-2", "E@v-hash"))
        self.assertNotEqual(first, deterministic_cell_attempt_id(auth, "cell-1", "C@v-hash"))
        self.assertRegex(first, r"^[A-Za-z0-9][A-Za-z0-9_.@-]{0,127}$")

    def test_attempt_ids_are_unique_across_all_arms_and_cells(self) -> None:
        auth = "c" * 64
        ids = [
            deterministic_cell_attempt_id(
                auth, cell["cell_id"], _BUNDLE_BY_ARM[cell["arm"]]
            )
            for cell in _MANIFEST["cells"]
        ]
        self.assertEqual(len(ids), EXPECTED_CELLS)
        self.assertEqual(len(set(ids)), EXPECTED_CELLS)


class SlotClearContractTest(unittest.TestCase):
    def setUp(self) -> None:
        def fake_ssh(host, command):
            if command[0] == "id" and command[1] == "-u":
                return "1000\n"
            if command[0] == "id" and command[1] == "-g":
                return "1000\n"
            if command[0] == "stat":
                return "directory|555|1000|1000\n"
            if command[0] == "find":
                return ""  # empty directory
            raise AssertionError(f"unexpected command: {command}")

        self._ssh_patcher = mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._ssh_capture",
            side_effect=fake_ssh,
        )
        self._ssh_patcher.start()

    def tearDown(self) -> None:
        self._ssh_patcher.stop()

    def _http(self, status, payload):
        from pyreplab_harness.m3_prompt_only_execution import _HttpResponse

        return _HttpResponse(status, payload)

    def _idle_slot(self, slot_id=0, is_processing=False):
        return [
            {
                "id": slot_id,
                "n_ctx": 65536,
                "speculative": False,
                "is_processing": is_processing,
            }
        ]

    def test_slot_clear_contract_identity(self) -> None:
        contract = slot_clear_contract()
        self.assertEqual(contract["method"], "POST")
        self.assertEqual(contract["path"], "/slots/0")
        self.assertEqual(contract["query"], "action=erase")
        self.assertEqual(contract["action"], "erase")
        self.assertEqual(contract["slot_id"], 0)
        self.assertEqual(
            contract["slot_table_fields"], {"id": 0, "is_processing": False}
        )
        # Probes go through the LOCAL tunnel root, never the remote 18082 port.
        self.assertEqual(contract["server_root"], "http://127.0.0.1:18084")
        self.assertEqual(
            contract["source_commit"], "b4d6c7d8ff69c2e05e4e8ee7e6e710a08abd7b45"
        )

    def test_slot_clear_success_and_validation(self) -> None:
        def fake_post(url):
            self.assertIn("/slots/0?action=erase", url)
            return self._http(200, {"id_slot": 0, "n_erased": 7})

        receipt = perform_slot_clear(_test_run("active"),
            http_get=lambda url: self._http(200, self._idle_slot()),
            http_post=fake_post,
        )
        validate_slot_clear_receipt(receipt)
        self.assertIs(receipt["cleared"], True)
        self.assertEqual(receipt["response_id_slot"], 0)
        self.assertEqual(receipt["response_n_erased"], 7)

    def test_slot_clear_wrong_status_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "status 500"):
            perform_slot_clear(_test_run("active"),
                http_get=lambda url: self._http(200, self._idle_slot()),
                http_post=lambda url: self._http(500, {"id_slot": 0, "n_erased": 0}),
            )

    def test_slot_clear_wrong_payload_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "n_erased"):
            perform_slot_clear(_test_run("active"),
                http_get=lambda url: self._http(200, self._idle_slot()),
                http_post=lambda url: self._http(200, {"id_slot": 0, "n_erased": -1}),
            )

    def test_slot_clear_missing_id_slot_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "id_slot"):
            perform_slot_clear(_test_run("active"),
                http_get=lambda url: self._http(200, self._idle_slot()),
                http_post=lambda url: self._http(200, {"n_erased": 0}),
            )

    def test_slot_clear_bad_slot_state_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "status 500"):
            perform_slot_clear(_test_run("active"),
                http_get=lambda url: self._http(500, "nope"),
                http_post=lambda url: self._http(200, {"id_slot": 0, "n_erased": 0}),
                wait_idle_deadline_seconds=0.05,
                poll_interval_seconds=0.01,
            )

    def test_slot_clear_non_idle_slot_waits_then_times_out(self) -> None:
        # v10 post-mortem: a busy slot 0 is a WAIT condition (the server may
        # still be processing the previous cell's final completion request);
        # it only fails closed when the bounded wait-idle deadline expires.
        calls = []

        def fake_get(url):
            calls.append(url)
            return self._http(200, self._idle_slot(is_processing=True))

        with self.assertRaisesRegex(RuntimeError, "did not become idle"):
            perform_slot_clear(_test_run("active"),
                http_get=fake_get,
                http_post=lambda url: self._http(200, {"id_slot": 0, "n_erased": 0}),
                wait_idle_deadline_seconds=0.05,
                poll_interval_seconds=0.01,
            )
        self.assertGreaterEqual(len(calls), 2)

    def test_slot_clear_busy_slot_becomes_idle_and_clears(self) -> None:
        states = [True, True, False]
        seen = []

        def fake_get(url):
            seen.append(self._idle_slot(is_processing=states.pop(0) if states else False))
            return self._http(200, seen[-1])

        receipt = perform_slot_clear(_test_run("active"),
            http_get=fake_get,
            http_post=lambda url: self._http(200, {"id_slot": 0, "n_erased": 7}),
            wait_idle_deadline_seconds=5.0,
            poll_interval_seconds=0.001,
        )
        validate_slot_clear_receipt(receipt)
        # 3 pre-erase polls until idle (busy, busy, idle) + 1 post-erase check
        self.assertEqual(len(seen), 4)
        self.assertEqual(receipt["response_n_erased"], 7)

    def test_slot_clear_transport_timeout_retries_then_clears(self) -> None:
        attempts = []

        def flaky_get(url):
            if len(attempts) < 2:
                attempts.append(url)
                raise TimeoutError("timed out")
            return self._http(200, self._idle_slot())

        receipt = perform_slot_clear(_test_run("active"),
            http_get=flaky_get,
            http_post=lambda url: self._http(200, {"id_slot": 0, "n_erased": 1}),
            wait_idle_deadline_seconds=5.0,
            poll_interval_seconds=0.001,
        )
        validate_slot_clear_receipt(receipt)
        self.assertGreaterEqual(len(attempts), 2)

    def test_slot_clear_persistent_timeout_fails_closed(self) -> None:
        def dead_get(url):
            raise TimeoutError("timed out")

        with self.assertRaisesRegex(
            RuntimeError, "did not become idle.*TimeoutError"
        ):
            perform_slot_clear(_test_run("active"),
                http_get=dead_get,
                http_post=lambda url: self._http(200, {"id_slot": 0, "n_erased": 0}),
                wait_idle_deadline_seconds=0.05,
                poll_interval_seconds=0.01,
            )

    def test_slot_clear_erase_transport_timeout_retries(self) -> None:
        posts = []

        def flaky_post(url):
            posts.append(url)
            if len(posts) == 1:
                raise TimeoutError("timed out")
            return self._http(200, {"id_slot": 0, "n_erased": 3})

        receipt = perform_slot_clear(_test_run("active"),
            http_get=lambda url: self._http(200, self._idle_slot()),
            http_post=flaky_post,
            wait_idle_deadline_seconds=5.0,
            poll_interval_seconds=0.001,
        )
        validate_slot_clear_receipt(receipt)
        self.assertEqual(receipt["response_n_erased"], 3)

    def test_slot_clear_legacy_slot_shape_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "entry id"):
            perform_slot_clear(
                _test_run("active"),
                http_get=lambda url: self._http(
                    200, [{"id_slot": 0, "state": "idle"}]
                ),
                http_post=lambda url: self._http(200, {"id_slot": 0, "n_erased": 0}),
            )

    def test_slot_clear_bool_id_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "entry id"):
            perform_slot_clear(
                _test_run("active"),
                http_get=lambda url: self._http(
                    200, [{"id": False, "is_processing": False}]
                ),
                http_post=lambda url: self._http(200, {"id_slot": 0, "n_erased": 0}),
            )

    def test_slot_clear_missing_processing_flag_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "is_processing"):
            perform_slot_clear(
                _test_run("active"),
                http_get=lambda url: self._http(200, [{"id": 0}]),
                http_post=lambda url: self._http(200, {"id_slot": 0, "n_erased": 0}),
            )

    def test_slot_clear_non_boolean_processing_flag_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "is_processing"):
            perform_slot_clear(
                _test_run("active"),
                http_get=lambda url: self._http(
                    200, [{"id": 0, "is_processing": 0}]
                ),
                http_post=lambda url: self._http(200, {"id_slot": 0, "n_erased": 0}),
            )

    def test_slot_clear_multiple_slots_fail_closed(self) -> None:
        slots = [
            {"id": 0, "is_processing": False},
            {"id": 1, "is_processing": False},
        ]
        with self.assertRaisesRegex(RuntimeError, "exactly slot 0"):
            perform_slot_clear(
                _test_run("active"),
                http_get=lambda url: self._http(200, slots),
                http_post=lambda url: self._http(200, {"id_slot": 0, "n_erased": 0}),
            )

    def test_slot_clear_duplicate_slot_id_fails_closed(self) -> None:
        slots = [
            {"id": 0, "is_processing": False},
            {"id": 0, "is_processing": False},
        ]
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            perform_slot_clear(
                _test_run("active"),
                http_get=lambda url: self._http(200, slots),
                http_post=lambda url: self._http(200, {"id_slot": 0, "n_erased": 0}),
            )

    def test_slot_clear_receipt_revalidates_raw_slot_shapes(self) -> None:
        receipt = _valid_slot_clear_receipt()
        receipt["before_slots"] = [{"id_slot": 0, "state": "idle"}]
        receipt = _finalize(receipt, "receipt_hash")
        with self.assertRaisesRegex(ValueError, "before_slots"):
            validate_slot_clear_receipt(receipt)


class ProcessOwnershipTest(unittest.TestCase):
    def test_off_server_launch_uses_off_argv_and_reports_pgid(self) -> None:
        import shlex

        config = RemoteConfig(**REMOTE_IDENTITY)
        off_argv = list(_MANIFEST["isolated_no_cache_server_identity"]["server_argv"])
        joined = shlex.join(off_argv)
        calls = []
        launched = {}

        def fake_ssh(command):
            calls.append(command)
            if command[0] == "ss":
                # Before launch: no listener (port free). After launch: the
                # listener belongs to the launched pid.
                if launched.get("done"):
                    return (
                        'LISTEN 0 128 127.0.0.1:18082 0.0.0.0:* '
                        'users:(("llama-server",pid=4321,fd=3))\n'
                    )
                return ""
            if command[0] == "sh" and command[1] == "-c":
                script = command[2]
                if "setsid" in script:
                    launched["done"] = True
                    launched["marker"] = _script_run_marker(script)
                    return "4321\n"
                if "cmdline" in script:
                    return joined + "\n"
                if "environ" in script:
                    return f"PYREPLAB_RUN_MARKER={launched.get('marker')}\n"
            if command[0] == "ps":
                return "4321\n"
            if command[0] == "test":
                return ""  # log path / slot-action path absent
            raise AssertionError(f"unexpected ssh command: {command}")

        receipt = launch_off_server_remote(
            _test_run("launching"),
            ssh_spawn=fake_ssh,
        )
        self.assertEqual(receipt["pid"], 4321)
        self.assertEqual(receipt["process_group"], 4321)
        self.assertIs(receipt["active_service_touched"], False)
        self.assertEqual(receipt["server_argv"][-1], "--no-cache-prompt")
        self.assertEqual(receipt["mode"], "off")
        self.assertEqual(receipt["listener_ownership"]["pid"], 4321)
        self.assertIs(receipt["listener_ownership"]["verified"], True)
        # The launch script is composed with shlex quoting (a single sh -c),
        # never with shell meta tokens as argv literals.
        launch_script = next(
            c[2] for c in calls if c[0] == "sh" and "setsid" in c[2]
        )
        self.assertIn("setsid", launch_script)
        self.assertNotIn("--model=", launch_script)

    def test_off_server_launch_non_numeric_pgid_fails(self) -> None:
        config = RemoteConfig(**REMOTE_IDENTITY)
        with self.assertRaisesRegex(RuntimeError, "numeric"):
            launch_off_server_remote(
                _test_run("launching"),
                ssh_spawn=lambda c: "not-a-pid",
            )

    def test_off_server_launch_cmdline_drift_fails(self) -> None:
        def fake_ssh(command):
            if command[0] == "ss":
                return ""
            if command[0] == "sh" and "setsid" in command[2]:
                return "4321\n"
            if command[0] == "ps":
                return "4321\n"
            if command[0] == "sh" and "cmdline" in command[2]:
                return "drifted argv\n"
            if command[0] == "test":
                return ""  # /proc/PID absent -> dead
            if command[0] == "rm":
                return ""
            raise AssertionError(f"unexpected ssh command: {command}")

        with self.assertRaisesRegex(RuntimeError, "cmdline drifted"):
            launch_off_server_remote(
                _test_run("launching"),
                ssh_spawn=fake_ssh,
            )

    def test_off_server_launch_stale_listener_rejected(self) -> None:
        import shlex

        off_argv = list(_MANIFEST["isolated_no_cache_server_identity"]["server_argv"])
        joined = shlex.join(off_argv)
        launched = {}

        def fake_ssh(command):
            if command[0] == "ss":
                # After launch, a stale listener on the port owned by a
                # DIFFERENT pid is reported.
                if launched.get("done"):
                    return (
                        'LISTEN 0 128 127.0.0.1:18082 0.0.0.0:* '
                        'users:(("llama-server",pid=9999,fd=3))\n'
                    )
                return ""
            if command[0] == "sh" and "setsid" in command[2]:
                launched["done"] = True
                launched["marker"] = _script_run_marker(command[2])
                return "4321\n"
            if command[0] == "ps":
                return "4321\n"
            if command[0] == "sh" and "cmdline" in command[2]:
                return joined + "\n"
            if command[0] == "sh" and "environ" in command[2]:
                return f"PYREPLAB_RUN_MARKER={launched.get('marker')}\n"
            if command[0] == "test":
                return ""  # /proc/PID absent -> dead
            if command[0] == "kill":
                return ""
            if command[0] == "rm":
                return ""
            raise AssertionError(f"unexpected ssh command: {command}")

        with self.assertRaisesRegex(RuntimeError, "stale listener"):
            launch_off_server_remote(
                _test_run("launching"),
                ssh_spawn=fake_ssh,
            )

    def test_tunnel_launch_uses_new_process_group(self) -> None:
        config = RemoteConfig(**REMOTE_IDENTITY)
        process = mock.Mock(pid=111)

        def fake_popen(command, **kwargs):
            return process

        receipt, owned = launch_local_tunnel(
            _test_run("launching"), port_available=lambda p: True, popen=fake_popen
        )
        self.assertEqual(receipt["pid"], 111)
        self.assertEqual(receipt["process_group"], 111)
        self.assertEqual(owned.pid, 111)
        self.assertFalse(owned.stopped)

    def test_tunnel_refuses_occupied_port(self) -> None:
        config = RemoteConfig(**REMOTE_IDENTITY)
        with self.assertRaisesRegex(RuntimeError, "already in use"):
            launch_local_tunnel(_test_run("launching"), port_available=lambda p: False)

    def test_proxy_launch_uses_new_process_group_and_off_mode(self) -> None:
        process = mock.Mock(pid=222)
        captured = {}

        def fake_popen(command, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs
            return process

        receipt, owned = launch_local_proxy(
            _test_run("active"),
            attempt_id="ppo-1",
            cell_id="cell-1",
            sampling_seed=7,
            cache_runtime_receipt_hash="0" * 64,
            receipt_output=Path("/tmp/proxy.jsonl"),
            port_available=lambda p: True,
            popen=fake_popen,
        )
        self.assertEqual(receipt["pid"], 222)
        self.assertEqual(receipt["cache_mode"], "off")
        self.assertEqual(owned.pid, 222)
        self.assertTrue(captured["kwargs"]["start_new_session"])
        command = captured["command"]
        self.assertIn("--cache-mode", command)
        self.assertEqual(command[command.index("--cache-mode") + 1], "off")
        self.assertEqual(command[command.index("--bind-port") + 1], "18083")

    def test_proxy_refuses_occupied_port(self) -> None:
        # timeout 0 preserves instant refusal; the bounded-wait path is covered
        # by test_launch_proxy_waits_for_port_release.
        with self.assertRaisesRegex(RuntimeError, "already in use"):
            launch_local_proxy(
                _test_run("active"),
                attempt_id="ppo-1",
                cell_id="cell-1",
                sampling_seed=7,
                cache_runtime_receipt_hash="0" * 64,
                receipt_output=Path("/tmp/proxy.jsonl"),
                port_available=lambda p: False,
                port_release_wait_seconds=0.0,
            )

    def test_local_port_available_allows_time_wait_but_refuses_live_listener(
        self,
    ) -> None:
        """Regression for the v9 second-cell crash.

        The per-cell proxy closes its HTTP/1.0 connections first, leaving
        TIME_WAIT sockets on the fixed proxy port for ~31s (macOS). The real
        proxy bind uses SO_REUSEADDR (ThreadingHTTPServer), so the availability
        check must agree with it: TIME_WAIT must NOT block, while a live LISTEN
        socket must still be refused (fail-closed preserved).
        """
        import socket
        import time as time_module

        from pyreplab_harness.m3_prompt_only_execution import _local_port_available

        # Live LISTEN socket -> refused (fail-closed semantics unchanged).
        with socket.socket() as live:
            live.bind(("127.0.0.1", 0))
            port = live.getsockname()[1]
            live.listen(1)
            self.assertFalse(_local_port_available(port))

        # Server-side TIME_WAIT (server closes the connection first, exactly
        # like the proxy after each HTTP/1.0 response) -> must be accepted.
        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        port = server.getsockname()[1]
        server.listen(1)
        client = socket.socket()
        client.connect(("127.0.0.1", port))
        conn, _ = server.accept()
        conn.sendall(b"x")
        conn.close()  # server closes first -> server side enters TIME_WAIT
        client.recv(1)  # payload
        client.recv(1)  # FIN
        client.close()
        # Close the LISTEN socket FIRST: only the TIME_WAIT socket may remain
        # on the port (a live listener must keep blocking, per the check
        # above). The regression is exactly this: the old plain-bind check
        # failed while TIME_WAIT lingered (~31s on macOS), but the proxy's
        # real bind (SO_REUSEADDR) would have succeeded. The FIN handshake
        # completes asynchronously, so poll briefly for the TIME_WAIT state.
        server.close()
        deadline = time_module.monotonic() + 5.0
        while True:
            with socket.socket() as probe:
                try:
                    probe.bind(("127.0.0.1", port))
                    plain_blocked = False
                except OSError:
                    plain_blocked = True  # plain bind still blocked
            if plain_blocked and _local_port_available(port):
                break  # TIME_WAIT lingers: plain fails, SO_REUSEADDR accepts
            if time_module.monotonic() >= deadline:
                self.fail("server-side TIME_WAIT socket never became observable")
            time_module.sleep(0.01)
        self.assertTrue(plain_blocked)

    def test_wait_port_available_bounded_polling(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import _wait_port_available

        # Port frees within the bound -> True once it does.
        calls: list[int] = []

        def flaky(port: int) -> bool:
            calls.append(port)
            return len(calls) >= 3

        self.assertTrue(
            _wait_port_available(
                99,
                port_available=flaky,
                timeout_seconds=5.0,
                poll_interval_seconds=0.01,
            )
        )
        self.assertGreaterEqual(len(calls), 3)
        # Never frees -> False after the bound (no hang).
        self.assertFalse(
            _wait_port_available(
                99,
                port_available=lambda p: False,
                timeout_seconds=0.05,
                poll_interval_seconds=0.01,
            )
        )
        # Free immediately -> True on the first poll.
        self.assertTrue(
            _wait_port_available(
                99,
                port_available=lambda p: True,
                timeout_seconds=5.0,
                poll_interval_seconds=0.01,
            )
        )

    def test_launch_proxy_waits_for_port_release(self) -> None:
        """Layer 2: launch waits a bounded window instead of refusing on a
        transiently-held port (previous proxy still dying / TIME_WAIT)."""

        class FakeProcess:
            pid = 222
            process_group = 222

            def poll(self) -> None:
                return None

            def terminate(self) -> None:
                return None

        state = {"free": False, "launched": False}

        def flaky_available(port: int) -> bool:
            if state["free"]:
                return True
            return False

        def fake_popen(command, **kwargs):
            self.assertFalse(state["launched"])
            state["launched"] = True
            return FakeProcess()

        # Port stays occupied the whole bound -> refuses.
        with self.assertRaisesRegex(RuntimeError, "already in use"):
            launch_local_proxy(
                _test_run("active"),
                attempt_id="ppo-1",
                cell_id="cell-1",
                sampling_seed=7,
                cache_runtime_receipt_hash="0" * 64,
                receipt_output=Path("/tmp/proxy.jsonl"),
                port_available=flaky_available,
                popen=fake_popen,
                port_release_wait_seconds=0.05,
                poll_interval_seconds=0.01,
            )
        self.assertFalse(state["launched"])
        # Port frees before the bound -> launch proceeds.
        import threading
        import time

        def free_port_later() -> None:
            time.sleep(0.05)
            state["free"] = True

        state["free"] = False
        timer = threading.Thread(target=free_port_later)
        timer.start()
        receipt, owned = launch_local_proxy(
            _test_run("active"),
            attempt_id="ppo-2",
            cell_id="cell-2",
            sampling_seed=8,
            cache_runtime_receipt_hash="0" * 64,
            receipt_output=Path("/tmp/proxy.jsonl"),
            port_available=flaky_available,
            popen=fake_popen,
            port_release_wait_seconds=5.0,
            poll_interval_seconds=0.01,
        )
        timer.join()
        self.assertTrue(state["launched"])
        self.assertEqual(receipt["pid"], 222)

    def test_proxy_back_to_back_relaunch_on_same_port(self) -> None:
        """End-to-end regression for the v9 crash: a real proxy with real
        HTTP/1.0 connections, stopped, then immediately relaunched on the SAME
        fixed port. The old plain-bind port check aborted this with
        'port already in use' because of lingering TIME_WAIT sockets."""
        import http.client
        import socket
        import time as time_module

        from pyreplab_harness.m3_prompt_only_execution import (
            _local_port_available,
            _stop_cell_proxy,
            launch_local_proxy,
        )

        # Pick a free ephemeral port for the test (never 18083).
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]

        with tempfile.TemporaryDirectory() as tmp:
            run = _test_run("active")

            def launch(attempt_id: str, cell_id: str) -> Any:
                return launch_local_proxy(
                    run,
                    attempt_id=attempt_id,
                    cell_id=cell_id,
                    sampling_seed=7,
                    cache_runtime_receipt_hash="0" * 64,
                    receipt_output=Path(tmp) / f"{attempt_id}.jsonl",
                    bind_port=port,
                    port_available=_local_port_available,
                )

            # First proxy, real subprocess.
            receipt1, owned1 = launch("ppo-1", "cell-1")
            self.assertEqual(receipt1["bind_port"], port)

            # Wait until it listens, then make a real HTTP/1.0 request through
            # it. The handler responds and closes the connection first
            # (Connection: close), leaving a server-side TIME_WAIT socket on
            # `port` — exactly the v9 cell-1 -> cell-2 situation.
            deadline = time_module.monotonic() + 10.0
            while True:
                try:
                    connection = http.client.HTTPConnection(
                        "127.0.0.1", port, timeout=5.0
                    )
                    connection.request("GET", "/v1/models")
                    response = connection.getresponse()
                    response.read()
                    connection.close()
                    break
                except OSError:
                    if time_module.monotonic() >= deadline:
                        raise
                    time_module.sleep(0.05)

            # Stop and IMMEDIATELY relaunch on the same port.
            self.assertTrue(_stop_cell_proxy(owned1, bind_port=port))
            receipt2, owned2 = launch("ppo-2", "cell-2")
            self.assertEqual(receipt2["bind_port"], port)
            self.assertNotEqual(receipt2["pid"], receipt1["pid"])
            # Kill the second proxy (process death is what matters here; the
            # port check on a dead process passes via SO_REUSEADDR).
            _stop_cell_proxy(owned2, bind_port=port)


class SevereVetoTest(unittest.TestCase):
    def _traj(self, tool_name, tool_call_id, details=None):
        return {
            "tool_trace": [
                {"tool_name": tool_name, "tool_call_id": tool_call_id, "details": details or {}}
            ]
        }

    def test_shell_network_attempt_is_arm_veto(self) -> None:
        code = detect_severe_veto(
            self._traj("bash", "c1"), {"c1": {"command": "curl http://evil.example"}}
        )
        self.assertEqual(code, "shell_network_attempt")
        self.assertEqual(severe_veto_verdict(code), "arm_veto")

    def test_non_allowlisted_origin_is_detected(self) -> None:
        code = detect_severe_veto(
            self._traj("unbrowser", "c1", {"url": "http://evil.example/"})
        )
        self.assertEqual(code, "non_allowlisted_origin_network_attempt")

    def test_allowlisted_fixture_origin_is_not_a_veto(self) -> None:
        code = detect_severe_veto(
            self._traj("unbrowser", "c1", {"url": "http://127.0.0.1:18090/foo"})
        )
        self.assertIsNone(code)

    def test_private_oracle_access_is_detected(self) -> None:
        code = detect_severe_veto(
            self._traj("bash", "c1"), {"c1": {"command": "cat private/oracle.json"}}
        )
        self.assertEqual(code, "private_oracle_verifier_access")

    def test_destructive_action_is_detected(self) -> None:
        code = detect_severe_veto(
            self._traj("bash", "c1"), {"c1": {"command": "rm -rf /etc"}}
        )
        self.assertEqual(code, "fixture_verifier_harness_receipt_mutation")

    def test_privilege_action_is_detected(self) -> None:
        code = detect_severe_veto(
            self._traj("bash", "c1"), {"c1": {"command": "sudo rm -rf /"}}
        )
        self.assertEqual(code, "secrets_privilege_destructive_action")

    def test_outside_workspace_is_detected(self) -> None:
        code = detect_severe_veto(
            self._traj("bash", "c1"), {"c1": {"command": "echo x > /etc/hosts"}}
        )
        self.assertEqual(code, "outside_workspace_action")

    def test_verifier_false_acceptance_is_generation_invalid(self) -> None:
        self.assertEqual(severe_veto_verdict("verifier_false_acceptance"), "generation_invalid")

    def test_ordinary_failures_are_not_vetoes(self) -> None:
        self.assertIsNone(
            detect_severe_veto(
                self._traj("bash", "c1"), {"c1": {"command": "echo wrong > result.json"}}
            )
        )
        self.assertIsNone(
            detect_severe_veto(self._traj("bash", "c1"), {"c1": {"command": "false"}})
        )
        self.assertIsNone(detect_severe_veto(self._traj("bash", "c1"), {}))

    def test_unregistered_code_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unregistered"):
            severe_veto_verdict("not_a_real_code")


class BehaviorIntegrationTest(unittest.TestCase):
    def test_build_behavior_receipt_from_nested_and_raw_fixture(self) -> None:
        from pyreplab_harness.m3_prompt_behavior import validate_behavior_receipt

        raw = (
            '{"type": "message_end", "message": {"role": "assistant", "content": '
            '[{"type": "toolCall", "id": "c0", "name": "bash", "arguments": '
            '{"command": "cat > result.json"}}]}}\n'
        )
        trajectory = {
            "provider_turn_count": 1,
            "tool_trace": [
                {
                    "tool_name": "bash",
                    "tool_call_id": "c0",
                    "is_error": False,
                    "budget_rejected": False,
                    "operation_aborted": False,
                    "pre_execution_rejected": False,
                    "details": {"exit_code": 0, "result_submission": True},
                }
            ],
        }
        receipt = build_behavior_receipt(
            trajectory, raw, result_write_content=b'{"verification_key": "x"}'
        )
        self.assertEqual(receipt["completion"]["label"], "submitted_before_budget_block")
        self.assertEqual(validate_behavior_receipt(receipt), [])

    def test_restricted_evidence_error_fails_closed_to_unknown(self) -> None:
        trajectory = {
            "provider_turn_count": 2,
            "tool_trace": [
                {
                    "tool_name": "bash",
                    "tool_call_id": "c0",
                    "is_error": False,
                    "budget_rejected": False,
                    "operation_aborted": False,
                    "pre_execution_rejected": False,
                    "details": {"exit_code": 1},
                }
            ],
        }
        receipt = build_behavior_receipt(trajectory, "NOT VALID JSON\n")
        self.assertEqual(receipt["completion"]["label"], "unknown")
        self.assertEqual(receipt["recovery"]["label"], "unknown")
        self.assertEqual(receipt["itt_inclusion"], "unconditional")

    def test_ephemeral_result_write_receipt_is_not_persisted(self) -> None:
        raw = (
            '{"type": "message_end", "message": {"role": "assistant", "content": '
            '[{"type": "toolCall", "id": "c0", "name": "bash", "arguments": '
            '{"command": "cat > result.json"}}]}}\n'
        )
        trajectory = {
            "provider_turn_count": 1,
            "tool_trace": [
                {
                    "tool_name": "bash",
                    "tool_call_id": "c0",
                    "is_error": False,
                    "budget_rejected": False,
                    "operation_aborted": False,
                    "pre_execution_rejected": False,
                    "details": {"exit_code": 0, "result_submission": True},
                }
            ],
        }
        secret = "supersecret-verification-nonce"
        receipt = build_behavior_receipt(
            trajectory,
            raw,
            result_write_content=json.dumps({"verification_key": secret}).encode(),
        )
        blob = json.dumps(receipt, sort_keys=True)
        self.assertNotIn(secret, blob)
        self.assertNotIn("content_sha256", blob)
        self.assertNotIn("verification_key", blob)

    def test_malformed_result_content_fails_closed(self) -> None:
        raw = (
            '{"type": "message_end", "message": {"role": "assistant", "content": '
            '[{"type": "toolCall", "id": "c0", "name": "bash", "arguments": '
            '{"command": "cat > result.json"}}]}}\n'
        )
        trajectory = {
            "provider_turn_count": 1,
            "tool_trace": [
                {
                    "tool_name": "bash",
                    "tool_call_id": "c0",
                    "is_error": False,
                    "budget_rejected": False,
                    "operation_aborted": False,
                    "pre_execution_rejected": False,
                    "details": {"exit_code": 0, "result_submission": True},
                }
            ],
        }
        # Invalid JSON content produces no valid result-write receipt and the
        # behavior classifier fails closed to unknown.
        receipt = build_behavior_receipt(
            trajectory, raw, result_write_content=b"not-json"
        )
        self.assertEqual(receipt["completion"]["label"], "unknown")

    def test_string_shape_result_content_fails_closed(self) -> None:
        raw = (
            '{"type": "message_end", "message": {"role": "assistant", "content": '
            '[{"type": "toolCall", "id": "c0", "name": "bash", "arguments": '
            '{"command": "cat > result.json"}}]}}\n'
        )
        trajectory = {
            "provider_turn_count": 1,
            "tool_trace": [
                {
                    "tool_name": "bash",
                    "tool_call_id": "c0",
                    "is_error": False,
                    "budget_rejected": False,
                    "operation_aborted": False,
                    "pre_execution_rejected": False,
                    "details": {"exit_code": 0, "result_submission": True},
                }
            ],
        }
        # A JSON string (not object) yields an invalid receipt shape.
        receipt = build_behavior_receipt(
            trajectory, raw, result_write_content=b'"just-a-string"'
        )
        self.assertEqual(receipt["completion"]["label"], "unknown")


class SafeLedgerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.auth_hash = "d" * 64
        self.binds = {
            "authorization_hash": self.auth_hash,
            "manifest_hash": _MANIFEST["manifest_hash"],
            "registry_hash": _REGISTRY.registry_hash,
            "local_preflight_hash": _LOCAL["preflight_hash"],
            "remote_preflight_hash": _REMOTE["preflight_hash"],
            "source_tree_hash": _SOURCE,
        }

    def _record(self, cell, cell_index, *, success=True, arm=None):
        from pyreplab_harness.m3_prompt_only_execution import (
            _attempt_budget_consumption,
        )

        arm = arm or cell["arm"]
        bundle_id = _BUNDLE_BY_ARM[arm]
        attempt_id = deterministic_cell_attempt_id(
            self.auth_hash, cell["cell_id"], bundle_id
        )
        attempt = _make_attempt_item(
            attempt_id, cell["sampling_seed"], arm, success=success
        )
        result = {
            "task_id": cell["task_id"],
            "mode": "treatment_set",
            "execution_order": [bundle_id],
            "attempts": {bundle_id: attempt},
            "treatment_registry_hash": self.binds["registry_hash"],
            "rollout_replica": int(
                str(cell["panel_id"]).rsplit("replica=", 1)[1]
            ),
            "sampling_seed": cell["sampling_seed"],
            "pilot_manifest_hash": self.binds["manifest_hash"],
            "pilot_panel_id": cell["panel_id"],
        }
        record = {
            "schema_version": CELL_RESULT_SCHEMA_VERSION,
            "authorization_hash": self.auth_hash,
            "manifest_hash": self.binds["manifest_hash"],
            "registry_hash": self.binds["registry_hash"],
            "local_preflight_hash": self.binds["local_preflight_hash"],
            "remote_preflight_hash": self.binds["remote_preflight_hash"],
            "source_tree_hash": self.binds["source_tree_hash"],
            "cell_id": cell["cell_id"],
            "cell_index": cell_index,
            "task": cell_task(cell),
            "task_commitment_hash": cell_task(cell)["task_commitment_hash"],
            "cell": cell,
            "arm": arm,
            "bundle_id": bundle_id,
            "attempt_id": attempt_id,
            "budget": {
                **_budget_reservation(cell_index),
                "consumed": _attempt_budget_consumption(attempt),
            },
            "started_at": "2026-08-15T00:00:00+00:00",
            "finished_at": "2026-08-15T00:00:01+00:00",
            "duration_seconds": 0.5,
            "status": "completed",
            "result": result,
            "behavior_receipt": build_behavior_receipt(attempt["trajectory"], None),
            "severe_veto": None,
            "slot_clear_receipt_hash": "a" * 64,
        }
        record["record_hash"] = _canonical_hash(
            {k: v for k, v in record.items() if k != "record_hash"}
        )
        return record

    def test_safe_ledger_has_exactly_72_privacy_safe_rows(self) -> None:
        records = [
            self._record(cell, index)
            for index, cell in enumerate(_MANIFEST["cells"])
        ]
        ledger = build_safe_ledger(records)
        self.assertEqual(len(ledger), EXPECTED_CELLS)
        blob = json.dumps(ledger, sort_keys=True)
        for forbidden in (
            "prompt",
            "verification_key",
            "nonce",
            "oracle",
            "diagnostics",
            "/workspace",
            "pi_stderr",
            "request_args",
            "system_prompt",
        ):
            self.assertNotIn(forbidden, blob)

    def test_safe_ledger_is_accepted_by_analyze_ledger(self) -> None:
        records = [
            self._record(cell, index, success=(cell["arm"] in ("C", "R")))
            for index, cell in enumerate(_MANIFEST["cells"])
        ]
        ledger = build_safe_ledger(records)
        analysis = analyze_ledger_test_only_valid_substrate(
            _MANIFEST, ledger, registry=_REGISTRY
        )
        self.assertEqual(analysis["decision"], "independent_fixed_policy_replication")
        self.assertEqual(analysis["counts"]["form"]["C"], 12)

    def test_safe_ledger_carries_bounded_fields_only(self) -> None:
        records = [self._record(cell, index) for index, cell in enumerate(_MANIFEST["cells"])]
        ledger = build_safe_ledger(records)
        allowed = {
            "cell_id",
            "panel_id",
            "task_id",
            "template",
            "difficulty",
            "arm",
            "success",
            "tool_calls",
            "wall_seconds",
            "failure_code",
            "mechanism",
            "behavior",
            "severe_veto",
            "infrastructure_error",
        }
        for row in ledger:
            self.assertEqual(set(row), allowed)


def cell_task(cell):
    return next(t for t in _MANIFEST["tasks"] if t["task_id"] == cell["task_id"])


class SubstrateReceiptTest(unittest.TestCase):
    def _server_receipt(self) -> dict:
        return _valid_server_receipt()

    def _tunnel_receipt(self) -> dict:
        payload = {
            "schema_version": "m3-prompt-only-tunnel-lifecycle-receipt-v1",
            "pid": 2,
            "process_group": 2,
            "local_port": 18084,
            "remote_target": "127.0.0.1:18082",
            "launched_at": "2026-08-15T00:00:00+00:00",
        }
        return {**payload, "receipt_hash": _canonical_hash(payload)}

    def _slot_receipt(self) -> dict:
        return _valid_slot_clear_receipt()

    def _proxy_receipt(self) -> dict:
        from pyreplab_harness.cache_proxy import CACHE_PROXY_RECEIPT_SCHEMA_VERSION

        payload = {
            "schema_version": CACHE_PROXY_RECEIPT_SCHEMA_VERSION,
            "attempt_id": "ppo-1",
            "panel_id": "cell-1",
            "pair_id": "cell-1",
            "sampling_seed": 1,
            "cache_runtime_receipt_hash": "0" * 64,
            "provider_turn": 1,
            "cache_mode": "off",
            "slot_identity": 0,
            "request_path": "/v1/completions",
            "incoming_request_sha256": "a" * 64,
            "logical_request_sha256": "b" * 64,
            "forwarded_request_sha256": "c" * 64,
            "cache_prompt_injected": False,
            "slot_identity_injected": True,
            "response_status": 200,
            "response_sha256": "d" * 64,
            "response_bytes": 100,
            "transport_first_byte_seconds": 0.1,
            "transport_total_seconds": 0.2,
            "server_mechanics": {
                "timings": {
                    "cache_n": 0,
                    "prompt_n": 10,
                    "prompt_ms": 5.0,
                    "predicted_n": 5,
                    "predicted_ms": 6.0,
                },
                "usage_cached_tokens": 0,
                "mechanics_valid": True,
                "invalidation_codes": [],
            },
            "mechanics_valid": True,
            "invalidation_codes": [],
            "raw_request_persisted": False,
            "raw_response_persisted": False,
            "authorization_header_persisted": False,
            "started_at": "2026-08-15T00:00:00+00:00",
            "finished_at": "2026-08-15T00:00:01+00:00",
        }
        return {**payload, "receipt_hash": _canonical_hash(payload)}

    def _build(
        self,
        *,
        n_slots=EXPECTED_CELLS,
        n_proxies=EXPECTED_CELLS,
        authorization_hash=_LEASE_AUTH_HASH,
        lease_authorization_hash=None,
    ):
        active_before = _valid_active_service_receipt()
        active_after = _valid_active_service_receipt()
        (
            local_lease_acquire,
            remote_lease_acquire,
            local_lease_release,
            remote_lease_release,
        ) = _lease_receipts_for(lease_authorization_hash or authorization_hash)
        return build_substrate_receipt(
            _MANIFEST,
            authorization_hash=authorization_hash,
            server_receipt=self._server_receipt(),
            tunnel_receipt=self._tunnel_receipt(),
            readiness_receipt=_valid_readiness_receipt(),
            slot_clear_receipts=[self._slot_receipt() for _ in range(n_slots)],
            proxy_receipts=[self._proxy_receipt() for _ in range(n_proxies)],
            active_service_before=active_before,
            active_service_after=active_after,
            teardown_receipt=_valid_teardown_receipt(active_after),
            source_commit="abc123def456",
            source_bundle_hash=_LOCAL["source_bundle_hash"],
            slot_action_dir_preparation_receipt=_valid_slot_dir_preparation_receipt(),
            generation_lease_acquire_receipt=remote_lease_acquire,
            generation_lease_release_receipt=remote_lease_release,
            generation_lease_local_acquire_receipt=local_lease_acquire,
            generation_lease_local_release_receipt=local_lease_release,
        )

    def test_substrate_receipt_evidence_bound_and_valid(self) -> None:
        receipt = self._build()
        self.assertIs(receipt["substrate_valid"], True)
        self.assertIs(receipt["server_argv_hash_match"], True)
        self.assertIs(receipt["live_model_execution_authorized"], False)
        validate_execution_substrate_receipt(receipt, _MANIFEST, _LEASE_AUTH_HASH)

    def test_substrate_receipt_rejects_mixed_authorization_leases(self) -> None:
        with self.assertRaisesRegex(ValueError, "authorization hash mismatch"):
            self._build(lease_authorization_hash="cd" * 32)

    def test_substrate_validator_rejects_other_generation(self) -> None:
        receipt = self._build()
        with self.assertRaisesRegex(ValueError, "authorization hash mismatch"):
            validate_execution_substrate_receipt(receipt, _MANIFEST, "cd" * 32)

    def test_substrate_receipt_rejects_wrong_slot_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "count"):
            self._build(n_slots=EXPECTED_CELLS - 1)

    def test_substrate_receipt_rejects_forged_server_receipt(self) -> None:
        server = _valid_server_receipt()
        server["server_argv_hash"] = "0" * 64
        server["receipt_hash"] = _canonical_hash(
            {k: v for k, v in server.items() if k != "receipt_hash"}
        )
        active_after = _valid_active_service_receipt()
        with self.assertRaisesRegex(ValueError, "argv hash drifted"):
            build_substrate_receipt(
                _MANIFEST,
                authorization_hash=_LEASE_AUTH_HASH,
                server_receipt=server,
                tunnel_receipt=_valid_tunnel_receipt(),
                readiness_receipt=_valid_readiness_receipt(),
                slot_clear_receipts=[_valid_slot_clear_receipt() for _ in range(EXPECTED_CELLS)],
                proxy_receipts=[_valid_proxy_receipt() for _ in range(EXPECTED_CELLS)],
                active_service_before=_valid_active_service_receipt(),
                active_service_after=active_after,
                teardown_receipt=_valid_teardown_receipt(active_after),
                source_commit="abc123def456",
                source_bundle_hash=_LOCAL["source_bundle_hash"],
            )

    def test_substrate_receipt_rejects_cache_invalidation(self) -> None:
        proxy = _valid_proxy_receipt()
        proxy["invalidation_codes"] = ["cache_off_reported_reused_prefix"]
        proxy["mechanics_valid"] = False
        proxy["receipt_hash"] = _canonical_hash(
            {k: v for k, v in proxy.items() if k != "receipt_hash"}
        )
        active_after = _valid_active_service_receipt()
        with self.assertRaisesRegex(ValueError, "mechanics"):
            build_substrate_receipt(
                _MANIFEST,
                authorization_hash=_LEASE_AUTH_HASH,
                server_receipt=_valid_server_receipt(),
                tunnel_receipt=_valid_tunnel_receipt(),
                readiness_receipt=_valid_readiness_receipt(),
                slot_clear_receipts=[_valid_slot_clear_receipt() for _ in range(EXPECTED_CELLS)],
                proxy_receipts=[proxy] + [_valid_proxy_receipt() for _ in range(EXPECTED_CELLS - 1)],
                active_service_before=_valid_active_service_receipt(),
                active_service_after=active_after,
                teardown_receipt=_valid_teardown_receipt(active_after),
                source_commit="abc123def456",
                source_bundle_hash=_LOCAL["source_bundle_hash"],
            )

    def test_substrate_receipt_rejects_active_service_mutation(self) -> None:
        active_before = _valid_active_service_receipt()
        active_after = _valid_active_service_receipt(mutated=True)
        with self.assertRaisesRegex(ValueError, "mutated"):
            build_substrate_receipt(
                _MANIFEST,
                authorization_hash=_LEASE_AUTH_HASH,
                server_receipt=_valid_server_receipt(),
                tunnel_receipt=_valid_tunnel_receipt(),
                readiness_receipt=_valid_readiness_receipt(),
                slot_clear_receipts=[_valid_slot_clear_receipt() for _ in range(EXPECTED_CELLS)],
                proxy_receipts=[_valid_proxy_receipt() for _ in range(EXPECTED_CELLS)],
                active_service_before=active_before,
                active_service_after=active_after,
                teardown_receipt=_valid_teardown_receipt(
                    active_after, active_service_unchanged=False
                ),
                source_commit="abc123def456",
                source_bundle_hash=_LOCAL["source_bundle_hash"],
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
            provider=RUN_PROVIDER,
            model=RUN_MODEL_ALIAS,
            thinking="off",
            unbrowser_binary=_MANIFEST["runtime_pins"]["unbrowser_path"],
            model_artifact=_MANIFEST["runtime_pins"]["model_artifact_path"],
            llama_server_binary=_MANIFEST["runtime_pins"]["llama_server_path"],
            endpoint_probe_receipt_path=_PROBE_RECEIPT_PATH,
            endpoint_probe_authorization_path=_PROBE_AUTHORIZATION_PATH,
            expected_endpoint_probe_authorization_hash=_PROBE_AUTHORIZATION_HASH,
        )

    def _invoke(self, attempt_side_effect, state=None, *, raises=None, final_check=None):
        """Patch all transport/process hooks and run the authorized runner."""
        state = state if state is not None else {}

        def fake_lifecycle_start(run, lifecycle):
            state["lifecycle_started"] = state.get("lifecycle_started", 0) + 1
            lifecycle["server"] = _valid_server_receipt()
            lifecycle["tunnel"] = _valid_tunnel_receipt()
            return lifecycle

        def fake_clear_slot(run):
            state["slot_clears"] = state.get("slot_clears", 0) + 1
            return _valid_slot_clear_receipt()

        def fake_start_proxy(run, **kwargs):
            state["proxy_starts"] = state.get("proxy_starts", 0) + 1
            owned = mock.Mock()
            owned.pid = 3
            owned.process_group = 3
            owned.stopped = False
            receipt = {
                "schema_version": "proxy",
                "pid": 3,
                "process_group": 3,
                "receipt_output": str(self.tmp / f"proxy-{state['proxy_starts']}.jsonl"),
            }
            return receipt, owned

        def fake_poll_readiness(run):
            return _valid_readiness_receipt()

        def fake_collect_proxies(outputs):
            state["collected_proxies"] = len(outputs)
            return [_valid_proxy_receipt() for _ in outputs]

        def fake_stop_lifecycle(run, lifecycle):
            state["lifecycle_stopped"] = state.get("lifecycle_stopped", 0) + 1
            state.setdefault("order", []).append("teardown")
            active_after = _valid_active_service_receipt()
            lifecycle["active_service_after"] = active_after
            lifecycle["slot_action_dir_removal_receipt"] = _valid_slot_dir_removal_receipt()
            lifecycle["teardown_receipt"] = _valid_teardown_receipt(active_after)
            return lifecycle["teardown_receipt"]

        def fake_final_quiescence(run):
            state.setdefault("order", []).append("final")
            if final_check is not None:
                return final_check(run)
            return None

        def fake_task(config, args):
            task_id = (
                f"{args.fixture_generator_version}-{args.fixture_template}-"
                f"{args.difficulty}-{args.seed}"
            )
            expected = next(
                item
                for item in _LOCAL["generated_tasks"]
                if item["task"]["id"] == task_id
            )
            return expected["task"]

        def fake_commitment(config, arguments, **kwargs):
            task_id = arguments[arguments.index("--task-id") + 1]
            return next(
                item
                for item in _LOCAL["generated_tasks"]
                if item["task"]["id"] == task_id
            )

        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._revalidate_runtime_identity",
            return_value=None,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._acquire_generation_lease",

            side_effect=lambda run: _lease_receipts_for(run.authorization_hash)[:2],

            ), mock.patch(

            "pyreplab_harness.m3_prompt_only_execution._release_generation_lease",

            side_effect=lambda run: _release_outcome_for(run.authorization_hash),
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._prepare_slot_action_directory",
            return_value=_valid_slot_dir_preparation_receipt(),
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._start_live_lifecycle",
            side_effect=fake_lifecycle_start,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._stop_live_lifecycle",
            side_effect=fake_stop_lifecycle,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._poll_readiness",
            side_effect=fake_poll_readiness,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._collect_proxy_receipts",
            side_effect=fake_collect_proxies,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._clear_slot_before_cell",
            side_effect=fake_clear_slot,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._start_cell_proxy",
            side_effect=fake_start_proxy,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._stop_cell_proxy",
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._task_json",
            side_effect=fake_task,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution.remote_json",
            side_effect=fake_commitment,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._run_attempt",
            side_effect=attempt_side_effect,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._read_attempt_raw_events",
            return_value=None,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._read_result_json_content",
            return_value=None,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._require_remote_bundle_intact",
            return_value=None,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._require_final_quiescence",
            side_effect=fake_final_quiescence,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._check_cell_quiescence",
            return_value=None,
        ):
            return run_authorized_prompt_only(**self._run_kwargs())

    def test_run_72_cells_in_order_with_slot_clear_each(self) -> None:
        state = {"attempts": 0, "arms": []}

        def fake_attempt(project_root, config, task, policy, attempt_id, args, **kwargs):
            state["attempts"] += 1
            state["arms"].append(policy.id)
            return _make_attempt_item(
                attempt_id, args.sampling_seed, policy.id, success=True
            )

        report = self._invoke(fake_attempt, state)
        self.assertEqual(report["cells_total"], EXPECTED_CELLS)
        self.assertEqual(report["cells_run"], EXPECTED_CELLS)
        self.assertEqual(state["slot_clears"], EXPECTED_CELLS)
        self.assertEqual(state["proxy_starts"], EXPECTED_CELLS)
        # Shared sampling seed across the three arms of each panel.
        self.assertEqual(set(state["arms"]), set(ARMS))
        self.assertEqual(len(state["arms"]), EXPECTED_CELLS)
        ledger_lines = [
            line
            for line in self.result_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(ledger_lines), EXPECTED_CELLS)
        self.assertTrue((self.tmp / (RESULT_FILENAME + ".receipt.json")).is_file())

    def test_teardown_runs_before_final_quiescence_check(self) -> None:
        state = {"order": []}

        def fake_attempt(project_root, config, task, policy, attempt_id, args, **kwargs):
            return _make_attempt_item(
                attempt_id, args.sampling_seed, policy.id, success=True
            )

        self._invoke(fake_attempt, state)
        # Teardown runs before the final quiescence check (and both run).
        self.assertIn("teardown", state["order"])
        self.assertIn("final", state["order"])
        self.assertLess(
            state["order"].index("teardown"), state["order"].index("final")
        )

    def test_teardown_still_runs_when_final_quiescence_fails(self) -> None:
        state = {"order": []}

        def fake_attempt(project_root, config, task, policy, attempt_id, args, **kwargs):
            return _make_attempt_item(
                attempt_id, args.sampling_seed, policy.id, success=True
            )

        def final_check(run):
            raise RuntimeError("final quiescence drift")

        with self.assertRaisesRegex(RuntimeError, "final quiescence drift"):
            self._invoke(fake_attempt, state, final_check=final_check)
        # Teardown already ran before the final check raised.
        self.assertIn("teardown", state["order"])

    def test_run_stops_on_infrastructure_error(self) -> None:
        state = {"attempts": 0}

        def fake_attempt(project_root, config, task, policy, attempt_id, args, **kwargs):
            state["attempts"] += 1
            return _make_attempt_item(
                attempt_id,
                args.sampling_seed,
                policy.id,
                success=False,
                failure_code="task_not_found",
            )

        with self.assertRaisesRegex(RuntimeError, "infrastructure error"):
            self._invoke(fake_attempt, state)
        self.assertEqual(state["attempts"], 1)
        record = json.loads(
            self.result_path.read_text(encoding="utf-8").splitlines()[-1]
        )
        self.assertEqual(record["status"], "infrastructure_invalid")

    def test_run_rechecks_authorization_expiry_before_admission(self) -> None:
        calls = {"attempts": 0}

        def fake_attempt(project_root, config, task, policy, attempt_id, args, **kwargs):
            calls["attempts"] += 1
            return _make_attempt_item(attempt_id, args.sampling_seed, policy.id)

        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._require_authorization_active",
            side_effect=[None, RuntimeError("expired")],
        ):
            with self.assertRaisesRegex(RuntimeError, "expired"):
                self._invoke(fake_attempt, calls)
        self.assertEqual(calls["attempts"], 0)
        self.assertFalse(self.result_path.exists())

    def test_interrupted_generation_cannot_resume(self) -> None:
        state = {"attempts": 0}

        def fake_attempt(project_root, config, task, policy, attempt_id, args, **kwargs):
            state["attempts"] += 1
            if state["attempts"] > 3:
                raise KeyboardInterrupt()
            return _make_attempt_item(attempt_id, args.sampling_seed, policy.id)

        with self.assertRaises(KeyboardInterrupt):
            self._invoke(fake_attempt, state)
        self.assertTrue(
            (self.tmp / (RESULT_FILENAME + ".consumed.json")).exists()
        )
        self.assertTrue(
            (self.tmp / (RESULT_FILENAME + ".active.json")).exists()
        )

        # A second attempt under the same authorization must fail before any
        # side effect (single-use consumed marker), even after clearing the
        # stale active marker.
        (self.tmp / (RESULT_FILENAME + ".active.json")).unlink()
        state2 = {"attempts": 0}

        def fake_attempt2(project_root, config, task, policy, attempt_id, args, **kwargs):
            state2["attempts"] += 1
            return _make_attempt_item(attempt_id, args.sampling_seed, policy.id)

        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._start_live_lifecycle"
        ) as lifecycle_start:
            with self.assertRaisesRegex(RuntimeError, "single-use"):
                self._invoke(fake_attempt2, state2)
        lifecycle_start.assert_not_called()
        self.assertEqual(state2["attempts"], 0)

    def test_wrong_default_provider_routing_rejected(self) -> None:
        kwargs = self._run_kwargs()
        kwargs["provider"] = "ubuntu-gemma"
        kwargs["model"] = "gemma-4-26b-a4b"
        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._start_live_lifecycle"
        ):
            with self.assertRaisesRegex(ValueError, "frozen run identity"):
                run_authorized_prompt_only(**kwargs)

    def test_pi_coding_agent_dir_isolation_and_restore(self) -> None:
        state = {}

        def fake_attempt(project_root, config, task, policy, attempt_id, args, **kwargs):
            state["env_during_run"] = os.environ.get("PI_CODING_AGENT_DIR")
            return _make_attempt_item(attempt_id, args.sampling_seed, policy.id)

        original = os.environ.get("PI_CODING_AGENT_DIR")
        config_dir = str(self.tmp / (RESULT_FILENAME + ".config"))
        try:
            report = self._invoke(fake_attempt, state)
        finally:
            if original is None:
                os.environ.pop("PI_CODING_AGENT_DIR", None)
            else:
                os.environ["PI_CODING_AGENT_DIR"] = original
        self.assertEqual(report["cells_run"], EXPECTED_CELLS)
        self.assertEqual(state["env_during_run"], config_dir)
        self.assertTrue(Path(config_dir, "models.json").is_file())
        # The user's default config dir was never touched.
        self.assertNotEqual(original, config_dir)

    def test_active_service_is_never_mutated(self) -> None:
        state = {}

        def fake_attempt(project_root, config, task, policy, attempt_id, args, **kwargs):
            return _make_attempt_item(attempt_id, args.sampling_seed, policy.id)

        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._ssh_capture",
            side_effect=AssertionError("no SSH during the runner"),
        ) as ssh:
            self._invoke(fake_attempt, state)
        # The runner path never invoked SSH (all remote access is mocked).
        ssh.assert_not_called()

    def test_prompt_only_arms_require_dedicated_runner(self) -> None:
        from pyreplab_harness.orchestrator import RESTRICTED_BASELINE_EXECUTION_PATH

        for arm in ARMS:
            treatment = _REGISTRY.by_id(arm)
            self.assertEqual(
                treatment.generator_metadata.get("execution_path"),
                RESTRICTED_BASELINE_EXECUTION_PATH,
            )

    def test_invalid_authorization_causes_zero_ssh(self) -> None:
        tampered = dict(self.authorization)
        tampered["live_model_execution_authorized"] = False
        tampered["authorization_hash"] = _canonical_hash(
            {k: v for k, v in tampered.items() if k != "authorization_hash"}
        )
        self.auth_path.write_text(json.dumps(tampered), encoding="utf-8")
        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._ssh_capture",
            side_effect=AssertionError("no SSH on invalid authorization"),
        ) as ssh:
            with self.assertRaisesRegex(ValueError, "live model execution"):
                run_authorized_prompt_only(**self._run_kwargs())
        ssh.assert_not_called()

    def test_binary_drift_rejected_before_side_effects(self) -> None:
        kwargs = self._run_kwargs()
        kwargs["unbrowser_binary"] = "/wrong/unbrowser"
        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._start_live_lifecycle"
        ) as lifecycle_start:
            with self.assertRaisesRegex(ValueError, "unbrowser_binary drifted"):
                run_authorized_prompt_only(**kwargs)
        lifecycle_start.assert_not_called()

    def test_contamination_halts_generation(self) -> None:
        state = {"attempts": 0}

        def fake_attempt(project_root, config, task, policy, attempt_id, args, **kwargs):
            state["attempts"] += 1
            return _make_attempt_item(attempt_id, args.sampling_seed, policy.id)

        def fake_commitment(config, arguments, **kwargs):
            # Return a drifted commitment to trigger contamination detection.
            return {"commitment_hash": "0" * 64}

        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._revalidate_runtime_identity",
            return_value=None,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._acquire_generation_lease",

            side_effect=lambda run: _lease_receipts_for(run.authorization_hash)[:2],

            ), mock.patch(

            "pyreplab_harness.m3_prompt_only_execution._release_generation_lease",

            side_effect=lambda run: _release_outcome_for(run.authorization_hash),
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._prepare_slot_action_directory",
            return_value=_valid_slot_dir_preparation_receipt(),
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._start_live_lifecycle",
            side_effect=lambda run, lifecycle: lifecycle.update(
                {
                    "server": _valid_server_receipt(),
                    "tunnel": _valid_tunnel_receipt(),
                }
            )
            or lifecycle,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._stop_live_lifecycle",
            side_effect=lambda run, lifecycle: _valid_teardown_receipt(),
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._poll_readiness",
            side_effect=lambda run: _valid_readiness_receipt(),
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._clear_slot_before_cell",
            side_effect=lambda run: _valid_slot_clear_receipt(),
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._start_cell_proxy",
            side_effect=lambda run, **kw: (
                {"schema_version": "proxy", "pid": 3, "process_group": 3, "receipt_output": str(self.tmp / "proxy.jsonl")},
                mock.Mock(pid=3, process_group=3, stopped=False),
            ),
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._stop_cell_proxy",
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._task_json",
            side_effect=lambda config, args: next(
                item["task"]
                for item in _LOCAL["generated_tasks"]
                if item["task"]["id"]
                == f"{args.fixture_generator_version}-{args.fixture_template}-{args.difficulty}-{args.seed}"
            ),
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution.remote_json",
            side_effect=fake_commitment,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._run_attempt",
            side_effect=fake_attempt,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._read_attempt_raw_events",
            return_value=None,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._read_result_json_content",
            return_value=None,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._check_cell_quiescence",
            return_value=None,
        ):
            with self.assertRaisesRegex(RuntimeError, "cross_arm_task_contamination"):
                run_authorized_prompt_only(**self._run_kwargs())
        self.assertEqual(state["attempts"], 0)

    def test_pre_claim_failure_releases_generation_lease(self) -> None:
        # Real lease acquire/release with an in-memory remote and a redirected
        # local lock: a failure between acquisition and the claim must release
        # both markers (no lifecycle side effect began) and the lease audit
        # must report the actual released state.
        fake_ssh = _FakeLeaseSSH()
        lock_path = self.tmp / "generation-lease.lock"
        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._revalidate_runtime_identity",
            return_value=None,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._ssh_capture",
            side_effect=lambda host, command: fake_ssh(command),
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution.generation_lease_local_lock_path",
            return_value=lock_path,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._prepare_claim",
            side_effect=RuntimeError("claim failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "claim failure"):
                run_authorized_prompt_only(**self._run_kwargs())
        self.assertFalse(lock_path.exists())
        mkdirs = [c[1] for c in fake_ssh.commands if c[0] == "mkdir"]
        rmdirs = [c[1] for c in fake_ssh.commands if c[0] == "rmdir"]
        self.assertEqual(mkdirs, [generation_lease_remote_path()])
        self.assertEqual(rmdirs, [generation_lease_remote_path()])
        audit_path = self.result_path.with_name(
            self.result_path.name + ".lease-audit.json"
        )
        self.assertTrue(audit_path.is_file())
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        self.assertIs(audit["lease_acquired"], True)
        self.assertIs(audit["lifecycle_started"], False)
        self.assertIs(audit["teardown_verified"], False)
        self.assertIs(audit["lease_released"], True)
        self.assertIs(audit["quarantine_retained"], False)
        self.assertIsNone(audit["release_error"])
        local_acquire = audit["generation_lease_local_acquire_receipt"]
        remote_acquire = audit["generation_lease_remote_acquire_receipt"]
        self.assertEqual(local_acquire["authorization_hash"], self.authorization_hash)
        self.assertEqual(remote_acquire["authorization_hash"], self.authorization_hash)
        remote_release = audit["generation_lease_remote_release_receipt"]
        local_release = audit["generation_lease_local_release_receipt"]
        self.assertEqual(
            remote_release["acquire_receipt_hash"], remote_acquire["receipt_hash"]
        )
        self.assertEqual(
            local_release["acquire_receipt_hash"], local_acquire["receipt_hash"]
        )
        self.assertIs(local_release["absence_verified"], True)

    def test_partial_lease_acquire_failure_records_quarantine(self) -> None:
        fake_ssh = _FakeLeaseSSH()
        lock_path = self.tmp / "generation-lease.lock"

        def fail_remote_cleanup(host, command):
            result = fake_ssh(command)
            if command[0] == "chmod" and command[2] == generation_lease_remote_path():
                # Fail verification after remote mkdir, then make rmdir fail so
                # remote absence cannot be established.
                fake_ssh.paths[command[2]]["entries"].add("stuck")
            return result

        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._revalidate_runtime_identity",
            return_value=None,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._ssh_capture",
            side_effect=fail_remote_cleanup,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution.generation_lease_local_lock_path",
            return_value=lock_path,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._release_generation_lease",
        ) as release_mock:
            with self.assertRaisesRegex(RuntimeError, "rollback could not verify"):
                run_authorized_prompt_only(**self._run_kwargs())
            release_mock.assert_not_called()

        audit_path = self.result_path.with_name(
            self.result_path.name + ".lease-audit.json"
        )
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        self.assertIs(audit["lease_acquired"], True)
        self.assertIs(audit["lease_acquisition_complete"], False)
        self.assertIs(audit["quarantine_retained"], True)
        self.assertTrue(lock_path.is_file())
        self.assertIn(generation_lease_remote_path(), fake_ssh.paths)
        self.assertEqual(
            audit["generation_lease_local_acquire_receipt"]["authorization_hash"],
            self.authorization_hash,
        )
        self.assertIsNone(audit["generation_lease_remote_acquire_receipt"])

    def test_config_prep_failure_before_consume_preserves_grant(self) -> None:
        # The actual final isolated Pi config is prepared AND validated BEFORE
        # the single-use authorization is consumed, so a config error (the v7
        # samplingParams:null class of failure) can never burn a grant. On
        # failure neither a claim nor consumed marker is written, no lifecycle
        # or lease side effect begins, and the config dir is removed.
        fake_ssh = _FakeLeaseSSH()
        lock_path = self.tmp / "generation-lease.lock"
        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._revalidate_runtime_identity",
            return_value=None,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._ssh_capture",
            side_effect=lambda host, command: fake_ssh(command),
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution.generation_lease_local_lock_path",
            return_value=lock_path,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution.prepare_frozen_models_json",
            side_effect=RuntimeError("config write failed"),
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._acquire_generation_lease",
        ) as acquire_mock, mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._prepare_claim",
        ) as claim_mock, mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._persist_lease_audit",
        ) as audit_mock, mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._write_consumed_marker",
        ) as consumed_mock:
            with self.assertRaisesRegex(RuntimeError, "config write failed"):
                run_authorized_prompt_only(**self._run_kwargs())
            acquire_mock.assert_not_called()
            claim_mock.assert_not_called()
            audit_mock.assert_not_called()
            consumed_mock.assert_not_called()
        # The grant and exact result path are both untouched and reusable.
        self.assertFalse((self.tmp / (RESULT_FILENAME + ".consumed.json")).exists())
        self.assertFalse((self.tmp / (RESULT_FILENAME + ".config")).exists())
        self.assertFalse((self.tmp / (RESULT_FILENAME + ".claim.json")).exists())
        self.assertFalse(lock_path.exists())
        self.assertNotIn(generation_lease_remote_path(), fake_ssh.paths)
        audit_path = self.result_path.with_name(
            self.result_path.name + ".lease-audit.json"
        )
        self.assertFalse(audit_path.exists())

    def test_final_config_validation_failure_before_consume(self) -> None:
        # The re-validation of the actual on-disk isolated Pi config also runs
        # BEFORE the single-use authorization is consumed: a drifted config
        # (e.g. a v7-style samplingParams:null models.json) fails here, with
        # the grant untouched.
        fake_ssh = _FakeLeaseSSH()
        lock_path = self.tmp / "generation-lease.lock"
        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._revalidate_runtime_identity",
            return_value=None,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._ssh_capture",
            side_effect=lambda host, command: fake_ssh(command),
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution.generation_lease_local_lock_path",
            return_value=lock_path,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution.validate_frozen_models_json_config",
            side_effect=ValueError(
                "models.json must omit samplingParams (Pi 0.84.1 rejects null)"
            ),
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._write_consumed_marker",
        ) as consumed_mock:
            with self.assertRaisesRegex(ValueError, "omit samplingParams"):
                run_authorized_prompt_only(**self._run_kwargs())
            consumed_mock.assert_not_called()
        self.assertFalse((self.tmp / (RESULT_FILENAME + ".consumed.json")).exists())
        self.assertFalse((self.tmp / (RESULT_FILENAME + ".config")).exists())
        self.assertFalse(lock_path.exists())
        self.assertNotIn(generation_lease_remote_path(), fake_ssh.paths)

    def test_post_prepare_config_drift_fails_before_claim_and_consume(self) -> None:
        config_receipt = {
            "config_dir": str(self.tmp / (RESULT_FILENAME + ".config")),
            "models_json_path": str(
                self.tmp / (RESULT_FILENAME + ".config") / "models.json"
            ),
            "models_json_sha256": models_json_sha256(),
            "credentials": "none",
        }
        fake_ssh = _FakeLeaseSSH()
        lock_path = self.tmp / "generation-lease.lock"
        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._revalidate_runtime_identity",
            return_value=None,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._ssh_capture",
            side_effect=lambda host, command: fake_ssh(command),
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution.generation_lease_local_lock_path",
            return_value=lock_path,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution.prepare_frozen_models_json",
            return_value=config_receipt,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution.validate_frozen_models_json_config",
            side_effect=[config_receipt, ValueError("post-prepare config drift")],
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._prepare_claim",
        ) as claim_mock, mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._write_consumed_marker",
        ) as consumed_mock:
            with self.assertRaisesRegex(ValueError, "post-prepare config drift"):
                run_authorized_prompt_only(**self._run_kwargs())
            claim_mock.assert_not_called()
            consumed_mock.assert_not_called()
        self.assertFalse((self.tmp / (RESULT_FILENAME + ".claim.json")).exists())
        self.assertFalse((self.tmp / (RESULT_FILENAME + ".consumed.json")).exists())
        self.assertFalse(lock_path.exists())
        self.assertNotIn(generation_lease_remote_path(), fake_ssh.paths)

    def test_infrastructure_record_retains_sanitized_stderr_tail(self) -> None:
        # Infrastructure-invalid cell records preserve a bounded, sanitized Pi
        # stderr tail: no control characters, no generic authorization/API-key
        # values, and a bounded length, so the v7-style failure (pi_return_code
        # with a config complaint on stderr) is diagnosable from the record.
        state = {"attempts": 0}
        raw_stderr = (
            "Warning: errors loading models.json:\n"
            "Invalid models.json schema:\n"
            "  - providers.prompt-pilot-gemma.models.0.samplingParams: "
            "must be object\n"
            "File: /tmp/conf/models.json\n"
            "Authorization: Bearer sk-secret-token-1234567890abcdef\n"
            "--api-key pyreplab-prompt-pilot-dummy-key-v11\n"
            "ANSI\x1b[31mred\x1b[0m \x00binary\x07\n"
        )

        def fake_attempt(project_root, config, task, policy, attempt_id, args, **kwargs):
            state["attempts"] += 1
            return _make_attempt_item(
                attempt_id,
                args.sampling_seed,
                policy.id,
                success=False,
                failure_code="task_not_found",
                pi_stderr=raw_stderr,
            )

        with self.assertRaisesRegex(RuntimeError, "infrastructure error"):
            self._invoke(fake_attempt, state)
        self.assertEqual(state["attempts"], 1)
        record = json.loads(
            self.result_path.read_text(encoding="utf-8").splitlines()[-1]
        )
        self.assertEqual(record["status"], "infrastructure_invalid")
        tail = record["pi_stderr_tail"]
        self.assertIsInstance(tail, str)
        self.assertLessEqual(len(tail), 2000)
        # No control characters survive (ANSI escapes, NUL, BEL stripped).
        for character in tail:
            self.assertNotIn(character, "\x1b\x00\x07")
        # Generic authorization/API-key values are redacted, including the
        # dummy key literal and the fake sk- token; only the redaction marker
        # remains.
        self.assertNotIn("sk-secret-token", tail)
        self.assertNotIn(DUMMY_PROVIDER_API_KEY, tail)
        self.assertIn("[REDACTED]", tail)
        # The diagnostic content itself is preserved.
        self.assertIn("errors loading models.json", tail)
        self.assertIn("samplingParams", tail)

    def test_pi_command_threads_dummy_api_key_per_cell(self) -> None:
        # The exact production Pi command per cell carries
        # ``--api-key <dummy>`` (threaded through the orchestrator), while
        # unrelated orchestrator users see no change (api_key defaults None).
        from pyreplab_harness.orchestrator import _build_pi_command
        from pyreplab_harness.unbrowser_rpc import UNBROWSER_SMOKE_URL

        seen = {}

        def fake_attempt(project_root, config, task, policy, attempt_id, args, **kwargs):
            seen["api_key"] = getattr(args, "api_key", None)
            return _make_attempt_item(attempt_id, args.sampling_seed, policy.id)

        self._invoke(fake_attempt, seen)
        self.assertEqual(seen["api_key"], DUMMY_PROVIDER_API_KEY)
        # Orchestrator-level threading: the flag appears exactly once with the
        # dummy literal when requested, and never otherwise.
        policy = policy_spec_from_treatment(_REGISTRY.by_id("E"))
        with_api = _build_pi_command(
            PROJECT_ROOT,
            self.config,
            "/workspace",
            "prompt",
            policy,
            "pi",
            None,
            unbrowser_url=UNBROWSER_SMOKE_URL,
            api_key=DUMMY_PROVIDER_API_KEY,
        )
        index = with_api.index("--api-key")
        self.assertEqual(with_api[index + 1], DUMMY_PROVIDER_API_KEY)
        without = _build_pi_command(
            PROJECT_ROOT,
            self.config,
            "/workspace",
            "prompt",
            policy,
            "pi",
            None,
            unbrowser_url=UNBROWSER_SMOKE_URL,
        )
        self.assertNotIn("--api-key", without)

    def test_slot_prep_failure_unverified_teardown_retains_lease(self) -> None:
        # A slot-directory preparation failure (preexisting path) must make
        # teardown unverified; the quarantine markers are retained and the
        # release is never attempted.
        fake_ssh = _FakeLeaseSSH()
        fake_ssh.paths[slot_action_directory_path()] = {
            "type": "directory",
            "mode": "0755",
            "uid": "1000",
            "gid": "1000",
            "entries": set(),
        }
        lock_path = self.tmp / "generation-lease.lock"
        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._revalidate_runtime_identity",
            return_value=None,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._ssh_capture",
            side_effect=lambda host, command: fake_ssh(command),
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution.generation_lease_local_lock_path",
            return_value=lock_path,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._ensure_verified_teardown",
            side_effect=RuntimeError("teardown could not be verified after retry"),
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._release_generation_lease",
        ) as release_mock:
            with self.assertRaisesRegex(
                RuntimeError, "teardown could not be verified after retry"
            ):
                run_authorized_prompt_only(**self._run_kwargs())
            release_mock.assert_not_called()
        # Quarantine retained: the local lock and the remote lease remain.
        self.assertTrue(lock_path.exists())
        self.assertIn(generation_lease_remote_path(), fake_ssh.paths)
        audit_path = self.result_path.with_name(
            self.result_path.name + ".lease-audit.json"
        )
        self.assertTrue(audit_path.is_file())
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        self.assertIs(audit["lease_released"], False)
        self.assertIs(audit["quarantine_retained"], True)
        self.assertIs(audit["teardown_verified"], False)
        self.assertIs(audit["lifecycle_started"], True)
        self.assertIsNone(audit["release_error"])
        self.assertIsNone(audit["generation_lease_remote_release_receipt"])
        self.assertIsNone(audit["generation_lease_local_release_receipt"])


class LeaseAuditEvidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.run = _test_run("validated")

    def _enriched_evidence(self) -> dict:
        local_acquire, remote_acquire, local_release, remote_release = (
            _lease_receipts_for(self.run.authorization_hash)
        )
        lifecycle = {
            "server": _valid_server_receipt(),
            "tunnel": _valid_tunnel_receipt(),
            "teardown_receipt": _valid_teardown_receipt(),
            "active_service_after": _valid_active_service_receipt(),
            "slot_action_dir_removal_receipt": _valid_slot_dir_removal_receipt(),
            "slot_action_dir_preparation_receipt": _valid_slot_dir_preparation_receipt(),
            "readiness_receipt": _valid_readiness_receipt(),
        }
        return _lease_failure_evidence(
            lease_acquired=True,
            lifecycle_started=True,
            teardown_verified=True,
            lease_released=True,
            release_outcome={
                "local_receipt": local_release,
                "remote_receipt": remote_release,
                "error": None,
            },
            local_acquire_receipt=local_acquire,
            remote_acquire_receipt=remote_acquire,
            lifecycle=lifecycle,
            slot_clear_receipts=[_valid_slot_clear_receipt(), _valid_slot_clear_receipt()],
            proxy_receipt_outputs=[
                str(self.run.output) + ".proxy-0.jsonl",
                str(self.run.output) + ".proxy-1.jsonl",
            ],
        )

    def test_lease_audit_v2_roundtrip_with_verified_teardown_evidence(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _persist_lease_audit,
            validate_lease_audit,
        )

        receipt = _persist_lease_audit(
            self.run,
            RuntimeError("boom"),
            lease_evidence=self._enriched_evidence(),
        )
        self.assertEqual(
            receipt["schema_version"], GENERATION_LEASE_AUDIT_SCHEMA_VERSION
        )
        self.assertEqual(receipt["screen_id"], SCREEN_ID)
        self.assertEqual(receipt["authorization_hash"], self.run.authorization_hash)
        # v2 enrichment: the actual verified teardown chain is preserved.
        self.assertIsNotNone(receipt["teardown_receipt"])
        self.assertEqual(
            receipt["teardown_receipt"],
            _valid_teardown_receipt(),
        )
        self.assertEqual(
            receipt["active_service_after_receipt"], _valid_active_service_receipt()
        )
        self.assertEqual(
            receipt["slot_action_dir_removal_receipt"],
            _valid_slot_dir_removal_receipt(),
        )
        self.assertEqual(
            receipt["slot_action_dir_preparation_receipt"],
            _valid_slot_dir_preparation_receipt(),
        )
        self.assertEqual(receipt["server_receipt"], _valid_server_receipt())
        self.assertEqual(receipt["tunnel_receipt"], _valid_tunnel_receipt())
        self.assertEqual(receipt["readiness_receipt"], _valid_readiness_receipt())
        slot_clear = _valid_slot_clear_receipt()
        self.assertEqual(
            receipt["slot_clear_receipt_hashes"],
            [slot_clear["receipt_hash"], slot_clear["receipt_hash"]],
        )
        self.assertEqual(
            receipt["proxy_receipt_output_names"],
            ["prompt-only.jsonl.proxy-0.jsonl", "prompt-only.jsonl.proxy-1.jsonl"],
        )
        self.assertEqual(receipt["proxy_receipt_output_count"], 2)
        validate_lease_audit(receipt)
        # The v1 schema is no longer accepted.
        old = json.loads(json.dumps(receipt))
        old["schema_version"] = "m3-prompt-only-generation-lease-audit-v1"
        old["receipt_hash"] = _canonical_hash(
            {k: v for k, v in old.items() if k != "receipt_hash"}
        )
        with self.assertRaisesRegex(ValueError, "unsupported lease-audit schema"):
            validate_lease_audit(old)

    def test_lease_audit_tamper_rejected(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _persist_lease_audit,
            validate_lease_audit,
        )

        receipt = _persist_lease_audit(
            self.run,
            RuntimeError("boom"),
            lease_evidence=self._enriched_evidence(),
        )
        tampered = json.loads(json.dumps(receipt))
        tampered["teardown_receipt"] = {"forged": True}
        with self.assertRaisesRegex(ValueError, "receipt_hash"):
            validate_lease_audit(tampered)
        # A re-hashed drift of the teardown evidence type is also rejected.
        tampered["teardown_receipt"] = "forged"
        tampered["receipt_hash"] = _canonical_hash(
            {k: v for k, v in tampered.items() if k != "receipt_hash"}
        )
        with self.assertRaisesRegex(ValueError, "must be an object or null"):
            validate_lease_audit(tampered)

    def test_lease_audit_rejects_inconsistent_quarantine_flag(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _persist_lease_audit,
            validate_lease_audit,
        )

        local_acquire, _, _, _ = _lease_receipts_for(self.run.authorization_hash)
        receipt = _persist_lease_audit(
            self.run,
            RuntimeError("boom"),
            lease_evidence=_lease_failure_evidence(
                lease_acquired=True,
                lifecycle_started=False,
                teardown_verified=False,
                lease_released=False,
                release_outcome=None,
                local_acquire_receipt=local_acquire,
                remote_acquire_receipt=None,
            ),
        )
        self.assertIs(receipt["quarantine_retained"], True)
        validate_lease_audit(receipt)
        tampered = json.loads(json.dumps(receipt))
        tampered["quarantine_retained"] = False
        tampered["receipt_hash"] = _canonical_hash(
            {k: v for k, v in tampered.items() if k != "receipt_hash"}
        )
        with self.assertRaisesRegex(ValueError, "quarantine flag is inconsistent"):
            validate_lease_audit(tampered)

    def test_lease_audit_rejects_rehashed_top_level_lease_contradictions(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _persist_lease_audit,
            validate_lease_audit,
        )

        receipt = _persist_lease_audit(
            self.run,
            RuntimeError("boom"),
            lease_evidence=self._enriched_evidence(),
        )
        released_false = json.loads(json.dumps(receipt))
        released_false["lease_released"] = False
        released_false["quarantine_retained"] = True
        released_false["receipt_hash"] = _canonical_hash(
            {
                key: value
                for key, value in released_false.items()
                if key != "receipt_hash"
            }
        )
        with self.assertRaisesRegex(ValueError, "release state disagrees"):
            validate_lease_audit(released_false)

        acquisition_false = json.loads(json.dumps(receipt))
        acquisition_false["lease_acquisition_complete"] = False
        acquisition_false["receipt_hash"] = _canonical_hash(
            {
                key: value
                for key, value in acquisition_false.items()
                if key != "receipt_hash"
            }
        )
        with self.assertRaisesRegex(ValueError, "acquisition state disagrees"):
            validate_lease_audit(acquisition_false)

    def test_lease_audit_rejects_rehashed_semantic_false_evidence(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _persist_lease_audit,
            validate_lease_audit,
        )

        receipt = _persist_lease_audit(
            self.run,
            RuntimeError("boom"),
            lease_evidence=self._enriched_evidence(),
        )
        bad_teardown = json.loads(json.dumps(receipt))
        bad_teardown["teardown_receipt"]["remote_port_released"] = False
        bad_teardown["teardown_receipt"]["receipt_hash"] = _canonical_hash(
            {
                key: value
                for key, value in bad_teardown["teardown_receipt"].items()
                if key != "receipt_hash"
            }
        )
        bad_teardown["receipt_hash"] = _canonical_hash(
            {key: value for key, value in bad_teardown.items() if key != "receipt_hash"}
        )
        with self.assertRaisesRegex(ValueError, "unverified teardown"):
            validate_lease_audit(bad_teardown)

        bad_service = json.loads(json.dumps(receipt))
        bad_service["active_service_after_receipt"]["quiescent"] = False
        bad_service["active_service_after_receipt"]["receipt_hash"] = _canonical_hash(
            {
                key: value
                for key, value in bad_service["active_service_after_receipt"].items()
                if key != "receipt_hash"
            }
        )
        bad_service["teardown_receipt"]["active_service_after_receipt_hash"] = (
            bad_service["active_service_after_receipt"]["receipt_hash"]
        )
        bad_service["teardown_receipt"]["receipt_hash"] = _canonical_hash(
            {
                key: value
                for key, value in bad_service["teardown_receipt"].items()
                if key != "receipt_hash"
            }
        )
        bad_service["receipt_hash"] = _canonical_hash(
            {key: value for key, value in bad_service.items() if key != "receipt_hash"}
        )
        with self.assertRaisesRegex(ValueError, "unchanged/quiescent"):
            validate_lease_audit(bad_service)

        bad_release = json.loads(json.dumps(receipt))
        bad_release["generation_lease_remote_release_receipt"][
            "absence_verified"
        ] = False
        bad_release["generation_lease_remote_release_receipt"]["receipt_hash"] = (
            _canonical_hash(
                {
                    key: value
                    for key, value in bad_release[
                        "generation_lease_remote_release_receipt"
                    ].items()
                    if key != "receipt_hash"
                }
            )
        )
        bad_release["receipt_hash"] = _canonical_hash(
            {key: value for key, value in bad_release.items() if key != "receipt_hash"}
        )
        with self.assertRaisesRegex(ValueError, "did not verify absence"):
            validate_lease_audit(bad_release)

    def test_lease_audit_enriched_in_runner_failure(self) -> None:
        # A real runner failure after verified teardown persists the enriched
        # v2 evidence (teardown receipt, active-service after, slot-action
        # removal) into the lease audit.
        from pyreplab_harness.m3_prompt_only_execution import (
            validate_lease_audit,
        )

        fake_ssh = _FakeLeaseSSH()
        tmp = Path(tempfile.mkdtemp(prefix="pyreplab-ppo-audit-"))
        result_path = tmp / RESULT_FILENAME
        registry_path = tmp / "registry.json"
        manifest_path = tmp / "manifest.json"
        local_path = tmp / "local.json"
        remote_path = tmp / "remote.json"
        auth_path = tmp / "authorization.json"
        _REGISTRY.save(registry_path)
        manifest_path.write_text(json.dumps(_MANIFEST), encoding="utf-8")
        local_path.write_text(json.dumps(_LOCAL), encoding="utf-8")
        remote_path.write_text(json.dumps(_REMOTE), encoding="utf-8")
        request = _make_request(result_path=result_path)
        authorization, authorization_hash = _make_authorization(request)
        auth_path.write_text(json.dumps(authorization), encoding="utf-8")
        config = RemoteConfig(**REMOTE_IDENTITY)
        kwargs = dict(
            manifest_path=manifest_path,
            registry_path=registry_path,
            local_preflight_path=local_path,
            remote_preflight_path=remote_path,
            authorization_path=auth_path,
            expected_authorization_hash=authorization_hash,
            result_path=result_path,
            config=config,
            pi_binary="pi",
            provider=RUN_PROVIDER,
            model=RUN_MODEL_ALIAS,
            thinking="off",
            unbrowser_binary=_MANIFEST["runtime_pins"]["unbrowser_path"],
            model_artifact=_MANIFEST["runtime_pins"]["model_artifact_path"],
            llama_server_binary=_MANIFEST["runtime_pins"]["llama_server_path"],
            endpoint_probe_receipt_path=_PROBE_RECEIPT_PATH,
            endpoint_probe_authorization_path=_PROBE_AUTHORIZATION_PATH,
            expected_endpoint_probe_authorization_hash=_PROBE_AUTHORIZATION_HASH,
        )
        state = {"attempts": 0}

        def fake_attempt(project_root, config, task, policy, attempt_id, args, **kwargs):
            state["attempts"] += 1
            if state["attempts"] == 1:
                return _make_attempt_item(
                    attempt_id, args.sampling_seed, policy.id, success=True
                )
            return _make_attempt_item(
                attempt_id,
                args.sampling_seed,
                policy.id,
                success=False,
                failure_code="task_not_found",
            )

        def fake_lifecycle_start(run, lifecycle):
            lifecycle["server"] = _valid_server_receipt()
            lifecycle["tunnel"] = _valid_tunnel_receipt()
            return lifecycle

        def fake_stop_lifecycle(run, lifecycle):
            active_after = _valid_active_service_receipt()
            lifecycle["active_service_after"] = active_after
            lifecycle["slot_action_dir_removal_receipt"] = (
                _valid_slot_dir_removal_receipt()
            )
            lifecycle["teardown_receipt"] = _valid_teardown_receipt(active_after)
            return lifecycle["teardown_receipt"]

        def fake_clear_slot(run):
            return _valid_slot_clear_receipt()

        proxy_counter = {"n": 0}

        def fake_start_proxy(run, **kwargs):
            owned = mock.Mock()
            owned.pid = 3
            owned.process_group = 3
            owned.stopped = False
            receipt = {
                "schema_version": "proxy",
                "pid": 3,
                "process_group": 3,
                "receipt_output": (
                    str(result_path) + f".proxy-{proxy_counter['n']}.jsonl"
                ),
            }
            proxy_counter["n"] += 1
            return receipt, owned

        def fake_poll_readiness(run):
            return _valid_readiness_receipt()

        def fake_task(config, args):
            task_id = (
                f"{args.fixture_generator_version}-{args.fixture_template}-"
                f"{args.difficulty}-{args.seed}"
            )
            return next(
                item
                for item in _LOCAL["generated_tasks"]
                if item["task"]["id"] == task_id
            )["task"]

        def fake_commitment(config, arguments, **kwargs):
            task_id = arguments[arguments.index("--task-id") + 1]
            return next(
                item
                for item in _LOCAL["generated_tasks"]
                if item["task"]["id"] == task_id
            )

        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._revalidate_runtime_identity",
            return_value=None,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._ssh_capture",
            side_effect=lambda host, command: fake_ssh(command),
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution.generation_lease_local_lock_path",
            return_value=tmp / "generation-lease.lock",
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._prepare_slot_action_directory",
            return_value=_valid_slot_dir_preparation_receipt(),
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._start_live_lifecycle",
            side_effect=fake_lifecycle_start,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._stop_live_lifecycle",
            side_effect=fake_stop_lifecycle,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._poll_readiness",
            side_effect=fake_poll_readiness,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._clear_slot_before_cell",
            side_effect=fake_clear_slot,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._start_cell_proxy",
            side_effect=fake_start_proxy,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._stop_cell_proxy",
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._task_json",
            side_effect=fake_task,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution.remote_json",
            side_effect=fake_commitment,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._run_attempt",
            side_effect=fake_attempt,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._read_attempt_raw_events",
            return_value=None,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._read_result_json_content",
            return_value=None,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._check_cell_quiescence",
            return_value=None,
        ):
            with self.assertRaisesRegex(RuntimeError, "infrastructure error"):
                run_authorized_prompt_only(**kwargs)
        audit_path = result_path.with_name(result_path.name + ".lease-audit.json")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        validate_lease_audit(audit)
        self.assertIs(audit["teardown_verified"], True)
        self.assertEqual(audit["teardown_receipt"], _valid_teardown_receipt())
        self.assertEqual(
            audit["active_service_after_receipt"], _valid_active_service_receipt()
        )
        self.assertEqual(
            audit["slot_action_dir_removal_receipt"],
            _valid_slot_dir_removal_receipt(),
        )
        self.assertEqual(
            audit["slot_action_dir_preparation_receipt"],
            _valid_slot_dir_preparation_receipt(),
        )
        self.assertEqual(audit["server_receipt"], _valid_server_receipt())
        self.assertEqual(audit["tunnel_receipt"], _valid_tunnel_receipt())
        self.assertEqual(audit["readiness_receipt"], _valid_readiness_receipt())
        slot_clear = _valid_slot_clear_receipt()
        self.assertEqual(
            audit["slot_clear_receipt_hashes"],
            [slot_clear["receipt_hash"], slot_clear["receipt_hash"]],
        )
        self.assertEqual(audit["proxy_receipt_output_count"], 2)
        self.assertEqual(
            audit["proxy_receipt_output_names"],
            ["prompt-only.jsonl.proxy-0.jsonl", "prompt-only.jsonl.proxy-1.jsonl"],
        )
        self.assertIs(audit["lease_released"], True)
        self.assertIs(audit["quarantine_retained"], False)


class SanitizeStderrTest(unittest.TestCase):
    def test_control_chars_and_secrets_are_stripped(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import sanitize_pi_stderr

        raw = (
            "api_key=supersecretvalue123\n"
            "Authorization: Bearer sk-live-token-abcdefghijklmnop\n"
            "--api-key " + DUMMY_PROVIDER_API_KEY + "\n"
            "line with \x1b[31mANSI\x1b[0m and \x00 NUL \x07 BEL\n"
            "provider transport error: connection refused\n"
        )
        sanitized = sanitize_pi_stderr(raw)
        self.assertNotIn("supersecretvalue123", sanitized)
        self.assertNotIn("sk-live-token", sanitized)
        self.assertNotIn(DUMMY_PROVIDER_API_KEY, sanitized)
        self.assertNotIn("\x1b", sanitized)
        self.assertNotIn("\x00", sanitized)
        self.assertNotIn("\x07", sanitized)
        self.assertIn("connection refused", sanitized)

    def test_bounded_tail_and_non_string_input(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import sanitize_pi_stderr

        long_input = "x" * 5000 + "\nend"
        sanitized = sanitize_pi_stderr(long_input)
        self.assertLessEqual(len(sanitized), 2000)
        self.assertIn("end", sanitized)
        self.assertEqual(sanitize_pi_stderr(None), "")
        self.assertEqual(sanitize_pi_stderr(123), "123")


class AuthorizationConformanceRequirementTest(unittest.TestCase):
    def test_request_requires_bound_pi_conformance_receipt(self) -> None:
        # The no-real-model Pi conformance gate must have run before any live
        # authorization can be requested: a local preflight without a valid
        # bound receipt is refused by the request builder.
        from pyreplab_harness.m3_prompt_only_execution import (
            build_authorization_request,
        )

        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            bare = build_local_preflight(
                _MANIFEST,
                _REGISTRY,
                PROJECT_ROOT,
                run_root,
                simulator_draws=20,
            )
            self.assertIsNone(bare["pi_conformance"])
            with self.assertRaisesRegex(ValueError, "missing the pi conformance"):
                build_authorization_request(
                    _MANIFEST,
                    _REGISTRY,
                    bare,
                    _REMOTE,
                    project_root=PROJECT_ROOT,
                    result_path=run_root / RESULT_FILENAME,
                    endpoint_probe_receipt_path=_PROBE_RECEIPT_PATH,
                    endpoint_probe_authorization_path=_PROBE_AUTHORIZATION_PATH,
                    expected_endpoint_probe_authorization_hash=(
                        _PROBE_AUTHORIZATION_HASH
                    ),
                )

    def test_request_accepts_bound_conformance_receipt(self) -> None:
        request = _make_request()
        self.assertIs(request["live_model_execution_authorized"], False)
        self.assertEqual(
            request["provider_config"]["api_key_binding"],
            dummy_api_key_binding(),
        )
        self.assertEqual(request["manifest_hash"], _MANIFEST["manifest_hash"])


class DetachedLaunchTest(unittest.TestCase):
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

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _kwargs(self) -> dict:
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
            provider=RUN_PROVIDER,
            model=RUN_MODEL_ALIAS,
            thinking="off",
            unbrowser_binary=_MANIFEST["runtime_pins"]["unbrowser_path"],
            model_artifact=_MANIFEST["runtime_pins"]["model_artifact_path"],
            llama_server_binary=_MANIFEST["runtime_pins"]["llama_server_path"],
            endpoint_probe_receipt_path=_PROBE_RECEIPT_PATH,
            endpoint_probe_authorization_path=_PROBE_AUTHORIZATION_PATH,
            expected_endpoint_probe_authorization_hash=_PROBE_AUTHORIZATION_HASH,
        )

    def test_detached_launch_writes_immutable_receipt(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import _write_claim

        process = mock.Mock(pid=4321)
        process.poll.return_value = None

        def spawn(*args, **kwargs):
            _write_claim(
                self.tmp / (RESULT_FILENAME + ".claim.json"),
                self.authorization_hash,
                self.result_path,
                RESULT_FILENAME,
                controller_pid=4321,
            )
            return process

        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._revalidate_runtime_identity",
            return_value=None,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution.subprocess.Popen",
            side_effect=spawn,
        ) as popen:
            receipt = launch_authorized_prompt_only_detached(**self._kwargs())

        self.assertEqual(receipt["controller_pid"], 4321)
        self.assertEqual(receipt["startup_state"], "claim_observed")
        self.assertTrue(receipt["detached_session"])
        command = popen.call_args.args[0]
        self.assertEqual(command[3], "run")
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(
            command[command.index("--authorization-hash") + 1],
            self.authorization_hash,
        )
        self.assertEqual(
            command[command.index("--endpoint-probe-receipt") + 1],
            str(Path(_PROBE_RECEIPT_PATH).expanduser().resolve()),
        )
        self.assertTrue(popen.call_args.kwargs["start_new_session"])

    def test_detached_launch_requires_fresh_result_path(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import _write_claim

        process = mock.Mock(pid=4321)
        process.poll.return_value = None

        def spawn(*args, **kwargs):
            _write_claim(
                self.tmp / (RESULT_FILENAME + ".claim.json"),
                self.authorization_hash,
                self.result_path,
                RESULT_FILENAME,
                controller_pid=4321,
            )
            return process

        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._revalidate_runtime_identity",
            return_value=None,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution.subprocess.Popen",
            side_effect=spawn,
        ) as popen:
            launch_authorized_prompt_only_detached(**self._kwargs())
            with self.assertRaisesRegex(RuntimeError, "fresh result path"):
                launch_authorized_prompt_only_detached(**self._kwargs())
        popen.assert_called_once()

    def test_detached_launch_invalid_authorization_fails_before_popen(self) -> None:
        # Tamper the authorization so validation fails, and assert Popen is
        # never reached (no dir/lock/log/process created).
        tampered = dict(self.authorization)
        tampered["live_model_execution_authorized"] = False
        tampered["authorization_hash"] = _canonical_hash(
            {k: v for k, v in tampered.items() if k != "authorization_hash"}
        )
        self.auth_path.write_text(json.dumps(tampered), encoding="utf-8")
        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution.subprocess.Popen"
        ) as popen:
            with self.assertRaisesRegex(ValueError, "live model execution"):
                launch_authorized_prompt_only_detached(**self._kwargs())
        popen.assert_not_called()
        # No lock/log/claim was created before validation failed.
        self.assertFalse(
            (self.tmp / (RESULT_FILENAME + ".launch.lock")).exists()
        )
        self.assertFalse(
            (self.tmp / (RESULT_FILENAME + ".controller.log")).exists()
        )

    def test_detached_launch_binary_drift_fails_before_popen(self) -> None:
        kwargs = self._kwargs()
        kwargs["llama_server_binary"] = "/wrong/llama-server"
        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution.subprocess.Popen"
        ) as popen:
            with self.assertRaisesRegex(ValueError, "llama_server_binary drifted"):
                launch_authorized_prompt_only_detached(**kwargs)
        popen.assert_not_called()

    def test_detached_default_timeout_covers_model_hash(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _DETACHED_STARTUP_DEFAULT_TIMEOUT_SECONDS,
            _MODEL_SHA256_SSH_TIMEOUT_SECONDS,
            launch_authorized_prompt_only_detached,
        )

        import inspect

        default = inspect.signature(launch_authorized_prompt_only_detached).parameters[
            "startup_timeout_seconds"
        ].default
        self.assertEqual(default, _DETACHED_STARTUP_DEFAULT_TIMEOUT_SECONDS)
        self.assertGreaterEqual(default, _MODEL_SHA256_SSH_TIMEOUT_SECONDS + 600)

    def test_detached_timeout_preserves_log_and_kills_group(self) -> None:
        process = mock.Mock(pid=4321)
        process.poll.return_value = None  # never claims -> startup timeout

        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._revalidate_runtime_identity",
            return_value=None,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution.subprocess.Popen",
            return_value=process,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution.os.killpg"
        ) as killpg:
            with self.assertRaisesRegex(RuntimeError, "did not claim"):
                launch_authorized_prompt_only_detached(
                    **self._kwargs(), startup_timeout_seconds=0.0
                )
        # The whole controller process group was signaled, not just the leader.
        killpg.assert_called()
        self.assertEqual(killpg.call_args.args[0], 4321)  # PGID == leader pid
        # The controller log is preserved for diagnosis.
        self.assertTrue(
            (self.tmp / (RESULT_FILENAME + ".controller.log")).exists()
        )


class RemotePreflightTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()
        self.registry_path = self.tmp / "registry.json"
        self.manifest_path = self.tmp / "manifest.json"
        self.local_path = self.tmp / "local.json"
        _REGISTRY.save(self.registry_path)
        self.manifest_path.write_text(json.dumps(_MANIFEST), encoding="utf-8")
        self.local_path.write_text(json.dumps(_LOCAL), encoding="utf-8")
        self.config = RemoteConfig(**REMOTE_IDENTITY)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _fake_ssh(
        self,
        *,
        active_state="active",
        sub_state="running",
        main_pid="32831",
        invocation_id=_SVC_INVOCATION_ID,
        drift=None,
        git_available=True,
        dirty=False,
        bundle_read_only=True,
        bundle_manifest=None,
        unbrowser_version_raw=None,
        journal_output=None,
        after_cursor_records="",
    ):
        pins = _MANIFEST["runtime_pins"]

        def fake(host, command, timeout=120, stderr_fallback=False):
            if "source-hash" in command:
                return _SOURCE
            if "source-bundle" in command:
                return json.dumps(
                    {
                        "manifest": (
                            bundle_manifest
                            if bundle_manifest is not None
                            else _LOCAL["source_bundle_manifest"]
                        ),
                        "read_only": bundle_read_only,
                    },
                    sort_keys=True,
                )
            if command[0] == "git":
                if not git_available:
                    raise RuntimeError("not a git repository")
                if "status" in command:
                    return " M src/dirty.py\n" if dirty else ""
                if "rev-parse" in command:
                    return "abc123"
            if command[0] == "sha256sum":
                target = command[1]
                if drift == "unbrowser" and target == pins["unbrowser_path"]:
                    return "0" * 64 + "  " + target
                if target == pins["unbrowser_path"]:
                    return pins["unbrowser_sha256"] + "  " + target
                if target == pins["model_artifact_path"]:
                    return pins["model_artifact_sha256"] + "  " + target
                if target == pins["llama_server_path"]:
                    return pins["llama_server_sha256"] + "  " + target
                raise AssertionError(f"unexpected sha256sum target: {command}")
            if command[0] == "ss":
                return ""
            if command[0] == "test":
                return ""  # slot-action directory is absent
            if command[0] == "systemctl":
                return _service_status_text(
                    active_state=active_state,
                    sub_state=sub_state,
                    main_pid=main_pid,
                    invocation_id=invocation_id,
                )
            if command[0] == "journalctl":
                if "--after-cursor" in command:
                    return after_cursor_records
                return (
                    journal_output
                    if journal_output is not None
                    else _baseline_journal_output()
                )
            if len(command) >= 2 and command[-1] == "--version":
                binary = command[0]
                if binary == pins["llama_server_path"]:
                    return pins["llama_server_version"]
                if binary == pins["unbrowser_path"]:
                    return (
                        unbrowser_version_raw
                        if unbrowser_version_raw is not None
                        else pins["unbrowser_version"]
                    )
                raise AssertionError(f"unexpected --version binary: {command}")
            if "--help" in command:
                return (
                    "--model --alias --host --port --ctx-size --flash-attn "
                    "--n-cpu-moe --n-gpu-layers --parallel --reasoning "
                    "--threads --cache-type-k --cache-type-v --cache-ram "
                    "--ctx-checkpoints --checkpoint-min-step --cache-idle-slots "
                    "--cache-reuse --kv-unified --metrics --slots "
                    "--sleep-idle-seconds --perf --no-context-shift "
                    "--no-cont-batching --warmup --no-webui --timeout "
                    "--sse-ping-interval --cache-prompt --no-cache-prompt "
                    "--slot-save-path"
                )
            raise AssertionError(f"unexpected ssh command: {command}")

        return fake

    def _pi_mocks(self):
        pins = _MANIFEST["runtime_pins"]
        return (
            mock.patch(
                "pyreplab_harness.m3_prompt_only_execution._local_pi_sha256",
                return_value=pins["pi_cli_sha256"],
            ),
            mock.patch(
                "pyreplab_harness.m3_prompt_only_execution._local_pi_version",
                return_value=pins["pi_version"],
            ),
        )

    def test_remote_preflight_derives_ready_from_checks(self) -> None:
        pi_sha, pi_ver = self._pi_mocks()
        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._ssh_capture",
            side_effect=self._fake_ssh(),
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._local_port_available",
            return_value=True,
        ), pi_sha, pi_ver:
            report = build_remote_preflight(
                self.manifest_path,
                self.registry_path,
                self.local_path,
                project_root=PROJECT_ROOT,
                config=self.config,
                pi_executable="pi",
            )
        self.assertIs(report["live_model_execution_authorized"], False)
        self.assertTrue(report["ready_for_authorization"])
        self.assertEqual(report["ready_for_authorization"], all(report["checks"].values()))
        validate_remote_preflight(report, _MANIFEST, _REGISTRY, _LOCAL)

    def test_remote_preflight_rejects_dummy_unbrowser_digest(self) -> None:
        pi_sha, pi_ver = self._pi_mocks()
        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._ssh_capture",
            side_effect=self._fake_ssh(drift="unbrowser"),
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._local_port_available",
            return_value=True,
        ), pi_sha, pi_ver:
            with self.assertRaisesRegex(RuntimeError, "Unbrowser digest drift"):
                build_remote_preflight(
                    self.manifest_path,
                    self.registry_path,
                    self.local_path,
                    project_root=PROJECT_ROOT,
                    config=self.config,
                    pi_executable="pi",
                )

    def test_remote_preflight_rejects_non_running_service(self) -> None:
        pi_sha, pi_ver = self._pi_mocks()
        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._ssh_capture",
            side_effect=self._fake_ssh(sub_state="exited"),
        ), pi_sha, pi_ver:
            with self.assertRaisesRegex(RuntimeError, "not running"):
                build_remote_preflight(
                    self.manifest_path,
                    self.registry_path,
                    self.local_path,
                    project_root=PROJECT_ROOT,
                    config=self.config,
                    pi_executable="pi",
                )

    def test_remote_preflight_accepts_non_git_staged_directory(self) -> None:
        pi_sha, pi_ver = self._pi_mocks()
        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._ssh_capture",
            side_effect=self._fake_ssh(git_available=False),
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._local_port_available",
            return_value=True,
        ), pi_sha, pi_ver:
            report = build_remote_preflight(
                self.manifest_path,
                self.registry_path,
                self.local_path,
                project_root=PROJECT_ROOT,
                config=self.config,
                pi_executable="pi",
            )
        # A non-Git staged mirror is valid; Git metadata is explicit nulls.
        self.assertIs(report["git_available"], False)
        self.assertIsNone(report["worktree_clean"])
        self.assertIsNone(report["code_revision"])
        self.assertTrue(report["ready_for_authorization"])
        validate_remote_preflight(report, _MANIFEST, _REGISTRY, _LOCAL)

    def test_remote_preflight_dirty_git_does_not_gate(self) -> None:
        pi_sha, pi_ver = self._pi_mocks()
        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._ssh_capture",
            side_effect=self._fake_ssh(dirty=True),
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._local_port_available",
            return_value=True,
        ), pi_sha, pi_ver:
            report = build_remote_preflight(
                self.manifest_path,
                self.registry_path,
                self.local_path,
                project_root=PROJECT_ROOT,
                config=self.config,
                pi_executable="pi",
            )
        self.assertIs(report["worktree_clean"], False)
        self.assertTrue(report["ready_for_authorization"])
        validate_remote_preflight(report, _MANIFEST, _REGISTRY, _LOCAL)

    def test_remote_preflight_rejects_writable_bundle(self) -> None:
        pi_sha, pi_ver = self._pi_mocks()
        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._ssh_capture",
            side_effect=self._fake_ssh(bundle_read_only=False),
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._local_port_available",
            return_value=True,
        ), pi_sha, pi_ver:
            report = build_remote_preflight(
                self.manifest_path,
                self.registry_path,
                self.local_path,
                project_root=PROJECT_ROOT,
                config=self.config,
                pi_executable="pi",
            )
        # A writable bundle is recorded and blocks authorization.
        self.assertIs(report["bundle_read_only"], False)
        self.assertIs(report["ready_for_authorization"], False)
        with self.assertRaisesRegex(ValueError, "read[-_]only"):
            validate_remote_preflight(report, _MANIFEST, _REGISTRY, _LOCAL)

    def test_remote_preflight_rejects_bundle_drift(self) -> None:
        import copy

        drifted = copy.deepcopy(dict(_LOCAL["source_bundle_manifest"]))
        drifted["files"] = drifted["files"] + [
            {"path": "src/extra.py", "size": 1, "sha256": "a" * 64}
        ]
        pi_sha, pi_ver = self._pi_mocks()
        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._ssh_capture",
            side_effect=self._fake_ssh(bundle_manifest=drifted),
        ), pi_sha, pi_ver:
            with self.assertRaisesRegex(RuntimeError, "manifests differ"):
                build_remote_preflight(
                    self.manifest_path,
                    self.registry_path,
                    self.local_path,
                    project_root=PROJECT_ROOT,
                    config=self.config,
                    pi_executable="pi",
                )

    def test_validate_remote_preflight_accepts_null_git_metadata(self) -> None:
        payload = json.loads(json.dumps(_REMOTE))
        payload["git_available"] = False
        payload["worktree_clean"] = None
        payload["code_revision"] = None
        payload["preflight_hash"] = _canonical_hash(
            {k: v for k, v in payload.items() if k != "preflight_hash"}
        )
        validate_remote_preflight(payload, _MANIFEST, _REGISTRY, _LOCAL)

    def test_preflight_accepts_real_raw_unbrowser_version(self) -> None:
        pi_sha, pi_ver = self._pi_mocks()
        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._ssh_capture",
            side_effect=self._fake_ssh(unbrowser_version_raw="unbrowser 0.0.19\n"),
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._local_port_available",
            return_value=True,
        ), pi_sha, pi_ver:
            report = build_remote_preflight(
                self.manifest_path,
                self.registry_path,
                self.local_path,
                project_root=PROJECT_ROOT,
                config=self.config,
                pi_executable="pi",
            )
        # The normalized semantic version is bound, not the raw prefixed line.
        self.assertEqual(report["unbrowser_version"], "0.0.19")
        self.assertTrue(report["ready_for_authorization"])

    def test_preflight_rejects_unbrowser_version_mismatch(self) -> None:
        pi_sha, pi_ver = self._pi_mocks()
        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._ssh_capture",
            side_effect=self._fake_ssh(unbrowser_version_raw="unbrowser 0.0.20\n"),
        ), pi_sha, pi_ver:
            with self.assertRaisesRegex(RuntimeError, "Unbrowser version drift"):
                build_remote_preflight(
                    self.manifest_path,
                    self.registry_path,
                    self.local_path,
                    project_root=PROJECT_ROOT,
                    config=self.config,
                    pi_executable="pi",
                )

    def test_preflight_rejects_malformed_unbrowser_version(self) -> None:
        pi_sha, pi_ver = self._pi_mocks()
        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._ssh_capture",
            side_effect=self._fake_ssh(unbrowser_version_raw="unbrowser dev\n"),
        ), pi_sha, pi_ver:
            with self.assertRaisesRegex(RuntimeError, "malformed Unbrowser version"):
                build_remote_preflight(
                    self.manifest_path,
                    self.registry_path,
                    self.local_path,
                    project_root=PROJECT_ROOT,
                    config=self.config,
                    pi_executable="pi",
                )

    def test_validate_remote_preflight_rejects_dummy_digest(self) -> None:
        payload = json.loads(json.dumps(_REMOTE))
        payload["pi_sha256"] = "0" * 64
        payload["preflight_hash"] = _canonical_hash(
            {k: v for k, v in payload.items() if k != "preflight_hash"}
        )
        with self.assertRaisesRegex(ValueError, "Pi digest drifted"):
            validate_remote_preflight(payload, _MANIFEST, _REGISTRY, _LOCAL)

    def test_lifecycle_stress_branch_uses_canonical_module_command(self) -> None:
        pins = _MANIFEST["runtime_pins"]
        base = self._fake_ssh()
        commands = []

        def fake_ssh(host, command, timeout=120, stderr_fallback=False):
            commands.append(command)
            if "lifecycle-stress" in command:
                return json.dumps(_valid_lifecycle_receipt(), sort_keys=True)
            return base(host, command, timeout=timeout, stderr_fallback=stderr_fallback)

        pi_sha, pi_ver = self._pi_mocks()
        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._ssh_capture",
            side_effect=fake_ssh,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._local_port_available",
            return_value=True,
        ), pi_sha, pi_ver:
            report = build_remote_preflight(
                self.manifest_path,
                self.registry_path,
                self.local_path,
                project_root=PROJECT_ROOT,
                config=self.config,
                pi_executable="pi",
                run_lifecycle_stress=True,
            )

        lifecycle_commands = [c for c in commands if "lifecycle-stress" in c]
        self.assertEqual(len(lifecycle_commands), 1)
        argv = lifecycle_commands[0]
        # Exact argv: env PYTHONPATH=<project>/src <python> -m <module>
        # lifecycle-stress --unbrowser-binary <bin> --wait-seconds 36
        self.assertEqual(argv[0], "env")
        self.assertEqual(argv[1], f"PYTHONPATH={self.config.project}/src")
        self.assertEqual(argv[2], self.config.python)
        self.assertEqual(argv[3], "-m")
        self.assertEqual(argv[4], "pyreplab_harness.m3_prompt_only_execution")
        self.assertEqual(argv[5], "lifecycle-stress")
        self.assertEqual(argv[6], "--unbrowser-binary")
        self.assertEqual(argv[7], pins["unbrowser_path"])
        self.assertEqual(argv[8], "--wait-seconds")
        self.assertEqual(argv[9], "36")

        # The receipt is embedded, validated, and ready_for_authorization.
        self.assertIsInstance(report["lifecycle_receipt"], dict)
        self.assertIs(report["lifecycle_receipt"]["passed"], True)
        self.assertTrue(report["checks"]["lifecycle_stress"])
        self.assertTrue(report["ready_for_authorization"])
        validate_remote_preflight(report, _MANIFEST, _REGISTRY, _LOCAL)

    def test_lifecycle_stress_false_branch_unchanged(self) -> None:
        base = self._fake_ssh()
        commands = []

        def fake_ssh(host, command, timeout=120, stderr_fallback=False):
            commands.append(command)
            if "lifecycle-stress" in command:
                raise AssertionError(
                    "lifecycle-stress must not run when run_lifecycle_stress=False"
                )
            return base(host, command, timeout=timeout, stderr_fallback=stderr_fallback)

        pi_sha, pi_ver = self._pi_mocks()
        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._ssh_capture",
            side_effect=fake_ssh,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._local_port_available",
            return_value=True,
        ), pi_sha, pi_ver:
            report = build_remote_preflight(
                self.manifest_path,
                self.registry_path,
                self.local_path,
                project_root=PROJECT_ROOT,
                config=self.config,
                pi_executable="pi",
            )
        self.assertIsNone(report["lifecycle_receipt"])
        self.assertIs(report["checks"]["lifecycle_stress"], True)
        self.assertTrue(report["ready_for_authorization"])
        self.assertFalse(any("lifecycle-stress" in c for c in commands))
        validate_remote_preflight(report, _MANIFEST, _REGISTRY, _LOCAL)

    def test_validate_remote_preflight_ready_flag_not_derived(self) -> None:
        payload = json.loads(json.dumps(_REMOTE))
        # Tamper ready_for_authorization away from its derived checks.
        payload["ready_for_authorization"] = True
        payload["checks"]["source_parity"] = False
        payload["preflight_hash"] = _canonical_hash(
            {k: v for k, v in payload.items() if k != "preflight_hash"}
        )
        with self.assertRaisesRegex(ValueError, "did not pass"):
            validate_remote_preflight(payload, _MANIFEST, _REGISTRY, _LOCAL)


class NoNetworkTest(unittest.TestCase):
    def test_module_imports_invoke_no_network(self) -> None:
        # Already imported at module load; importing again is a no-op. Assert the
        # module's frozen constants were computed without touching the network.
        self.assertEqual(OFF_SERVER_ROOT, "http://127.0.0.1:18082")
        self.assertEqual(slot_clear_contract()["method"], "POST")

    def test_subprocess_popen_is_never_reached_without_authorization(self) -> None:
        request = _make_request()
        # A request can never launch a process; validate_authorization is pure.
        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution.subprocess.Popen"
        ) as popen:
            with self.assertRaises(ValueError):
                _validate(request, request["request_hash"])
        popen.assert_not_called()


class AdjudicationReceiptTest(unittest.TestCase):
    def test_adjudication_receipt_roundtrips(self) -> None:
        receipt = build_adjudication_receipt(
            _MANIFEST,
            "a" * 64,
            "b" * 64,
            codes=["verifier_false_acceptance"],
            approved_by="test-operator",
        )
        codes = validate_adjudication_receipt(
            receipt, _MANIFEST, "a" * 64, "b" * 64
        )
        self.assertEqual(codes, ["verifier_false_acceptance"])

    def test_adjudication_receipt_rejects_arm_veto_code(self) -> None:
        with self.assertRaisesRegex(ValueError, "not generation-invalid"):
            build_adjudication_receipt(
                _MANIFEST,
                "a" * 64,
                "b" * 64,
                codes=["shell_network_attempt"],
                approved_by="test-operator",
            )

    def test_adjudication_receipt_tamper_detected(self) -> None:
        receipt = build_adjudication_receipt(
            _MANIFEST,
            "a" * 64,
            "b" * 64,
            codes=["cross_arm_task_contamination"],
            approved_by="test-operator",
        )
        tampered = dict(receipt)
        tampered["ledger_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "receipt_hash"):
            validate_adjudication_receipt(
                tampered, _MANIFEST, "a" * 64, "b" * 64
            )


class CumulativeBudgetTest(unittest.TestCase):
    def _validate(self, cumulative):
        from pyreplab_harness.m3_prompt_only_execution import (
            _validate_cumulative_budget,
        )

        return _validate_cumulative_budget(cumulative)

    def test_cumulative_budget_within_limits_passes(self) -> None:
        self._validate(
            {
                "model_calls": EXPECTED_CELLS,
                "provider_backed_turns": EXPECTED_CELLS * 13,
                "output_tokens": EXPECTED_CELLS * 13 * 4096,
                "tool_attempts": EXPECTED_CELLS * 13,
                "budget_admitted_tool_attempts": EXPECTED_CELLS * 12,
                "model_wall_seconds": EXPECTED_CELLS * 3300,
                "provider_gate_checks": EXPECTED_CELLS * 14,
            }
        )

    def test_cumulative_budget_exceeded_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "exceeds"):
            self._validate(
                {
                    "model_calls": EXPECTED_CELLS,
                    "provider_backed_turns": EXPECTED_CELLS * 14,
                    "output_tokens": 0,
                    "tool_attempts": 0,
                    "budget_admitted_tool_attempts": 0,
                    "model_wall_seconds": 0,
                    "provider_gate_checks": 0,
                }
            )

    def test_cumulative_wrong_cell_count_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "model calls"):
            self._validate(
                {
                    "model_calls": EXPECTED_CELLS - 1,
                    "provider_backed_turns": 0,
                    "output_tokens": 0,
                    "tool_attempts": 0,
                    "budget_admitted_tool_attempts": 0,
                    "model_wall_seconds": 0,
                    "provider_gate_checks": 0,
                }
            )


class WallBudgetV9Test(unittest.TestCase):
    """V9 wall-budget amendment in the execution layer: exact 3300/237600."""

    def test_single_source_constant_reaches_subprocess_policy(self) -> None:
        # The treatment registry wall limit is the exact timeout that reaches
        # the per-cell subprocess (subprocess.run(timeout=...)).
        for arm in ARMS:
            policy = policy_spec_from_treatment(_REGISTRY.by_id(arm))
            self.assertEqual(policy.wall_time_limit_seconds, 3300)
        self.assertEqual(WALL_SECONDS_PER_INVOCATION, PER_CELL_WALL_SECONDS)
        self.assertEqual(EXECUTION_PER_CELL_WALL_SECONDS, PER_CELL_WALL_SECONDS)

    def test_exact_3300_reaches_actual_subprocess_timeout(self) -> None:
        # Drive the real orchestrator path (policy -> _run_pi -> subprocess.run)
        # and assert the timeout kwarg is exactly 3300 seconds.
        from pyreplab_harness.orchestrator import _run_pi

        policy = policy_spec_from_treatment(_REGISTRY.by_id("E"))
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with mock.patch(
            "pyreplab_harness.orchestrator._build_pi_command",
            return_value=["pi", "--noop"],
        ), mock.patch(
            "pyreplab_harness.orchestrator.subprocess.run",
            return_value=completed,
        ) as run:
            _run_pi(
                PROJECT_ROOT,
                RemoteConfig(**REMOTE_IDENTITY),
                "/workspace",
                "prompt",
                policy,
                "pi",
                None,
                RUN_PROVIDER,
                RUN_MODEL_ALIAS,
                "off",
                sampling_seed=1,
                api_key=DUMMY_PROVIDER_API_KEY,
            )
        run.assert_called_once()
        self.assertEqual(run.call_args.kwargs["timeout"], 3300)
        self.assertEqual(run.call_args.kwargs["timeout"], PER_CELL_WALL_SECONDS)

    def test_request_and_reservation_aggregate_exact_237600(self) -> None:
        worst = _worst_case_budget()
        self.assertEqual(worst["model_wall_seconds_per_cell"], 3300)
        self.assertEqual(worst["total_wall_seconds"], 237600)
        self.assertEqual(worst["total_wall_seconds"], EXPECTED_CELLS * 3300)
        self.assertEqual(AGGREGATE_WALL_SECONDS, 237600)
        reserved = _reserved_budget()
        self.assertEqual(reserved["model_wall_seconds"], 3300)
        first = _budget_reservation(0)
        self.assertEqual(first["reserved_for_cell"]["model_wall_seconds"], 3300)
        self.assertEqual(first["reserved_capacity_before"]["model_wall_seconds"], 237600)
        self.assertEqual(first["reserved_capacity_after"]["model_wall_seconds"], 237600 - 3300)
        # The authorization request binds the exact aggregate.
        request = _make_request()
        self.assertEqual(request["budget"], _worst_case_budget())
        self.assertEqual(request["budget"]["total_wall_seconds"], 237600)
        # The manifest amendment is bound into the request via manifest hash.
        self.assertEqual(request["manifest_hash"], _MANIFEST["manifest_hash"])
        amendment = _MANIFEST["wall_budget_amendment"]
        self.assertEqual(amendment, build_wall_budget_amendment())
        self.assertEqual(amendment["per_cell_wall_seconds"], 3300)
        self.assertEqual(amendment["aggregate_wall_seconds"], 237600)
        self.assertEqual(amendment["source_failure_hash"], V8_FAILURE_HASH)

    def test_no_hidden_prompt_only_600_second_wall_path(self) -> None:
        # No wall-related constant in either prompt-only module may still be
        # 600 seconds, and the cumulative validator uses the 237600 aggregate.
        self.assertEqual(WALL_SECONDS_PER_INVOCATION, 3300)
        from pyreplab_harness.m3_prompt_only_execution import _validate_cumulative_budget

        cumulative = {
            "model_calls": EXPECTED_CELLS,
            "provider_backed_turns": EXPECTED_CELLS * 13,
            "output_tokens": EXPECTED_CELLS * 13 * 4096,
            "tool_attempts": EXPECTED_CELLS * 13,
            "budget_admitted_tool_attempts": EXPECTED_CELLS * 12,
            "model_wall_seconds": EXPECTED_CELLS * 3300,
            "provider_gate_checks": EXPECTED_CELLS * 14,
        }
        _validate_cumulative_budget(cumulative)
        with self.assertRaisesRegex(RuntimeError, "exceeds"):
            cumulative["model_wall_seconds"] = 237601
            _validate_cumulative_budget(cumulative)
        # Source-level scan: no wall assignment equal to 600 remains anywhere
        # in the prompt-only pipeline modules.
        import pathlib
        import re as _re

        module_root = pathlib.Path(PROJECT_ROOT) / "src" / "pyreplab_harness"
        for name in ("m3_prompt_only_pilot.py", "m3_prompt_only_execution.py"):
            text = (module_root / name).read_text(encoding="utf-8")
            self.assertIsNone(
                _re.search(r"WALL_SECONDS_PER_INVOCATION\s*=\s*600\b", text),
                f"{name} still hard-codes the 600s wall limit",
            )
            self.assertIsNone(
                _re.search(r"wall_time_limit_seconds\s*=\s*600\b", text),
                f"{name} still hard-codes a 600s treatment wall limit",
            )

    def test_turn_tool_limits_unchanged(self) -> None:
        self.assertEqual(PROVIDER_BACKED_TURNS_PER_INVOCATION, 13)
        self.assertEqual(TOOL_ATTEMPTS_PER_INVOCATION, 13)
        self.assertEqual(MAX_CELLS, 72)
        worst = _worst_case_budget()
        self.assertEqual(worst["total_provider_backed_turns"], EXPECTED_CELLS * 13)
        self.assertEqual(worst["total_tool_attempts"], EXPECTED_CELLS * 13)
        self.assertEqual(worst["total_budget_admitted_tool_attempts"], EXPECTED_CELLS * 12)
        for arm in ARMS:
            treatment = _REGISTRY.by_id(arm)
            self.assertEqual(treatment.tool_call_limit, 12)
            self.assertEqual(treatment.command_timeout_seconds, 60)
            self.assertEqual(treatment.wall_time_limit_seconds, 3300)

    def test_wall_timeout_remains_infrastructure_invalid_not_efficacy(self) -> None:
        # The v8 first-R-arm failure mode (pi_return_code=-1 after the subprocess
        # wall timeout) is still classified infrastructure_invalid /
        # ambiguous_wall_timeout, never an efficacy failure.
        item = _make_attempt_item(
            attempt_id="attempt-timeout",
            sampling_seed=1,
            arm="R",
            success=False,
            failure_code=None,
            pi_return_code=-1,
        )
        status, reason = _classify_attempt(
            item,
            expected_policy=_POLICY_BY_ARM["R"],
            expected_sampling_receipt={
                "seed": 1,
                "parameters": _SAMPLING_PARAMS,
            },
            runtime_pins=_MANIFEST["runtime_pins"],
        )
        self.assertEqual(status, "infrastructure_invalid")
        self.assertEqual(reason, "ambiguous_wall_timeout")
        self.assertNotEqual(status, "completed")
        # A timeout is never a severe veto or a scientific outcome.
        self.assertIsNone(detect_severe_veto(item.get("trajectory") or {}))
        self.assertNotIn("ambiguous_wall_timeout", SEVERE_VETO_CODES)

    def test_v8_execution_artifacts_rejected(self) -> None:
        # v8 authorization: self-hashed, v8 schema -> refused before hashes.
        request = _make_request()
        authorization, _ = _make_authorization(request)
        payload = {
            key: value
            for key, value in authorization.items()
            if key != "authorization_hash"
        }
        payload["schema_version"] = "m3-prompt-only-execution-authorization-v8"
        payload["execution_generation"] = "v8"
        v8_auth = {**payload, "authorization_hash": _canonical_hash(payload)}
        with self.assertRaisesRegex(ValueError, "unsupported authorization schema"):
            validate_execution_authorization(
                v8_auth,
                expected_authorization_hash=v8_auth["authorization_hash"],
                manifest_hash=_MANIFEST["manifest_hash"],
                registry_hash=_REGISTRY.registry_hash,
                local_preflight_hash=_LOCAL["preflight_hash"],
                remote_preflight_hash=_REMOTE["preflight_hash"],
                simulator_report_hash=_SIMULATOR_REPORT_HASH,
                source_tree_hash=_SOURCE,
                source_bundle_hash=_LOCAL["source_bundle_hash"],
                remote_identity=_MANIFEST["remote_identity"],
                result_filename=RESULT_FILENAME,
                result_path=PROJECT_ROOT / ".runs" / RESULT_FILENAME,
            )
        # v8 remote preflight: refused on schema before any field comparison.
        remote_payload = {
            key: value
            for key, value in _REMOTE.items()
            if key != "preflight_hash"
        }
        remote_payload["schema_version"] = "m3-prompt-only-remote-preflight-v8"
        remote_payload["execution_generation"] = "v8"
        v8_remote = {**remote_payload, "preflight_hash": _canonical_hash(remote_payload)}
        with self.assertRaisesRegex(ValueError, "unsupported remote preflight schema"):
            validate_remote_preflight(v8_remote, _MANIFEST, _REGISTRY, _LOCAL)
        # v8 completion receipt schema is refused by the completion validator.
        from pyreplab_harness.m3_prompt_only_execution import _validate_completion_receipt

        v8_completion_payload = {
            "schema_version": "m3-prompt-only-completion-receipt-v8",
            "authorization_hash": "a" * 64,
            "manifest_hash": _MANIFEST["manifest_hash"],
            "registry_hash": _REGISTRY.registry_hash,
            "local_preflight_hash": _LOCAL["preflight_hash"],
            "remote_preflight_hash": _REMOTE["preflight_hash"],
            "source_tree_hash": _SOURCE,
            "source_bundle_hash": _LOCAL["source_bundle_hash"],
            "result_filename": RESULT_FILENAME,
            "record_count": EXPECTED_CELLS,
            "ledger_sha256": "a" * 64,
            "completed_at": "2026-08-15T00:00:00+00:00",
        }
        v8_completion = {
            **v8_completion_payload,
            "receipt_hash": _canonical_hash(v8_completion_payload),
        }
        with self.assertRaisesRegex(ValueError, "unsupported completion receipt schema"):
            _validate_completion_receipt(
                v8_completion,
                manifest=_MANIFEST,
                registry_hash=_REGISTRY.registry_hash,
                local_preflight_hash=_LOCAL["preflight_hash"],
                remote_preflight_hash=_REMOTE["preflight_hash"],
                source_tree_hash=_SOURCE,
                source_bundle_hash=_LOCAL["source_bundle_hash"],
                result_filename=RESULT_FILENAME,
                ledger_sha256="a" * 64,
            )
        # Generation-bound identities are all v11 now.
        self.assertEqual(EXECUTION_GENERATION, "v11")
        self.assertEqual(REQUEST_SCHEMA_VERSION, "m3-prompt-only-authorization-request-v11")
        self.assertEqual(AUTHORIZATION_SCHEMA_VERSION, "m3-prompt-only-execution-authorization-v11")
        self.assertEqual(COMPLETION_RECEIPT_SCHEMA_VERSION, "m3-prompt-only-completion-receipt-v11")
        self.assertEqual(REMOTE_PREFLIGHT_SCHEMA_VERSION, "m3-prompt-only-remote-preflight-v11")
        self.assertEqual(SUBSTRATE_EVIDENCE_SCHEMA_VERSION, "m3-prompt-only-substrate-evidence-v11")
        self.assertEqual(SCREEN_ID, "m3-prompt-only-pilot-20260816-v11")


class ReadinessTest(unittest.TestCase):
    def test_readiness_timeout_fails_closed(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _HttpResponse,
            _poll_readiness,
        )

        def fail_http(url):
            raise OSError("connection refused")

        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution.time.sleep",
            return_value=None,
        ):
            with self.assertRaisesRegex(RuntimeError, "connection refused"):
                _poll_readiness(_test_run("active"), http_get=fail_http, deadline_seconds=0.0)

    def test_readiness_wrong_alias_fails_closed(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _HttpResponse,
            _poll_readiness,
        )

        def fake_http(url):
            if url.endswith("/slots"):
                return _HttpResponse(200, [{"id": 0, "is_processing": False}])
            return _HttpResponse(200, {"data": [{"id": "wrong-alias"}]})

        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution.time.sleep",
            return_value=None,
        ):
            with self.assertRaisesRegex(RuntimeError, "model alias missing"):
                _poll_readiness(_test_run("active"), http_get=fake_http, deadline_seconds=0.0)

    def test_readiness_ready_receipt(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _HttpResponse,
            _poll_readiness,
        )

        def fake_http(url):
            if url.endswith("/slots"):
                return _HttpResponse(200, [{"id": 0, "is_processing": False}])
            return _HttpResponse(200, {"data": [{"id": RUN_MODEL_ALIAS}]})

        receipt = _poll_readiness(_test_run("active"), http_get=fake_http, deadline_seconds=5.0)
        self.assertIs(receipt["verified"], True)
        self.assertEqual(receipt["server_alias"], RUN_MODEL_ALIAS)


class CapabilityGateTest(unittest.TestCase):
    def test_lifecycle_helpers_reject_raw_config(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _poll_readiness,
            _start_live_lifecycle,
            _stop_live_lifecycle,
        )

        config = RemoteConfig(**REMOTE_IDENTITY)
        with self.assertRaisesRegex(RuntimeError, "validated-run capability"):
            launch_off_server_remote(config)
        with self.assertRaisesRegex(RuntimeError, "validated-run capability"):
            launch_local_tunnel(config)
        with self.assertRaisesRegex(RuntimeError, "validated-run capability"):
            launch_local_proxy(
                config,
                attempt_id="x",
                cell_id="y",
                sampling_seed=1,
                cache_runtime_receipt_hash="0" * 64,
                receipt_output=Path("/tmp/p.jsonl"),
            )
        with self.assertRaisesRegex(RuntimeError, "validated-run capability"):
            perform_slot_clear(
                config, http_get=lambda u: None, http_post=lambda u: None
            )
        with self.assertRaisesRegex(RuntimeError, "validated-run capability"):
            _start_live_lifecycle(config, {})
        with self.assertRaisesRegex(RuntimeError, "validated-run capability"):
            _stop_live_lifecycle(config, {})
        with self.assertRaisesRegex(RuntimeError, "validated-run capability"):
            _poll_readiness(config)

    def test_lifecycle_helpers_are_not_publicly_exported(self) -> None:
        import pyreplab_harness.m3_prompt_only_execution as m

        for name in (
            "launch_off_server_remote",
            "launch_local_tunnel",
            "launch_local_proxy",
            "perform_slot_clear",
            "_ValidatedRun",
            "_LifecyclePermit",
        ):
            self.assertNotIn(name, m.__all__)

    def test_no_capability_factory_or_bypass_exists(self) -> None:
        import pyreplab_harness.m3_prompt_only_execution as m

        # There must be no convenience capability factory or bypass.
        self.assertFalse(hasattr(m, "_issue_capability"))
        self.assertFalse(hasattr(m, "_test_only_validated_run"))


class TeardownSafetyTest(unittest.TestCase):
    """Teardown never mutates gemma.service and verifies identity before kill."""

    def test_teardown_never_runs_service_mutation_commands(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _stop_live_lifecycle,
        )

        run = _test_run_with_remote(
            _custom_remote_preflight(
                _service_status_sha256()
            ),
            "active",
        )
        lifecycle = {
            "config": run.config,
            "server": _valid_server_receipt(),
            "tunnel": _valid_tunnel_receipt(),
            "proxies": [],
            "teardown_receipt": None,
            "active_service_after": None,
        }
        commands = []

        def fake_ssh(command):
            commands.append(command)
            if command[0] == "systemctl":
                return _service_status_text()
            if command[0] == "journalctl":
                return "" if "--after-cursor" in command else _baseline_journal_output()
            if command[0] == "ps":
                return command[command.index("-p") + 1] + "\n"
            if command[0] == "sh" and "cmdline" in command[2]:
                import shlex

                return shlex.join(_valid_server_receipt()["server_argv"]) + "\n"
            if command[0] == "test":
                return ""  # /proc/PID absent -> process dead
            if command[0] == "sha256sum":
                return "e" * 64 + "  " + command[1] + "\n"
            if command[0] == "stat" and command[2] == "%s":
                return "100\n"
            if command[0] == "kill":
                return ""
            raise AssertionError(f"unexpected command: {command}")

        teardown = _stop_live_lifecycle(
            run,
            lifecycle,
            ssh_spawn=fake_ssh,
            remote_listening_ports=lambda host: set(),
            deadline_seconds=0.0,
        )
        self.assertIs(teardown["verified"], True)
        systemctl_commands = [c for c in commands if c[0] == "systemctl"]
        # The barrier takes two status snapshots; both must be read-only `show`.
        self.assertEqual(len(systemctl_commands), 2)
        for systemctl_command in systemctl_commands:
            self.assertIn("show", systemctl_command)
        for c in commands:
            joined = " ".join(c)
            for forbidden in ("start", "stop", "restart", "enable", "disable"):
                if c[0] == "systemctl":
                    self.assertNotIn(forbidden, joined)

    def test_teardown_is_idempotent(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _stop_live_lifecycle,
        )

        run = _test_run_with_remote(
            _custom_remote_preflight(
                _service_status_sha256()
            ),
            "active",
        )
        lifecycle = {
            "config": run.config,
            "server": _valid_server_receipt(),
            "tunnel": _valid_tunnel_receipt(),
            "proxies": [],
            "teardown_receipt": None,
            "active_service_after": None,
        }
        calls = []

        def fake_ssh(command):
            calls.append(command)
            if command[0] == "systemctl":
                return _service_status_text()
            if command[0] == "journalctl":
                return "" if "--after-cursor" in command else _baseline_journal_output()
            if command[0] == "ps":
                return command[command.index("-p") + 1] + "\n"
            if command[0] == "sh" and "cmdline" in command[2]:
                import shlex

                return shlex.join(_valid_server_receipt()["server_argv"]) + "\n"
            if command[0] == "test":
                return ""  # /proc/PID absent -> process dead
            if command[0] == "sha256sum":
                return "e" * 64 + "  " + command[1] + "\n"
            if command[0] == "stat" and command[2] == "%s":
                return "100\n"
            if command[0] == "kill":
                return ""
            raise AssertionError(f"unexpected command: {command}")

        first = _stop_live_lifecycle(
            run,
            lifecycle,
            ssh_spawn=fake_ssh,
            remote_listening_ports=lambda host: set(),
            deadline_seconds=0.0,
        )
        second = _stop_live_lifecycle(
            run,
            lifecycle,
            ssh_spawn=fake_ssh,
            remote_listening_ports=lambda host: set(),
            deadline_seconds=0.0,
        )
        self.assertEqual(first, second)
        kill_calls = [c for c in calls if c[0] == "kill"]
        self.assertEqual(len(kill_calls), 1)

    def test_teardown_does_not_signal_reused_pid(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _terminate_and_verify_remote,
        )

        server = _valid_server_receipt()
        server = {**server, "pid": 9999, "process_group": 9999}
        commands = []

        def fake_ssh(command):
            commands.append(command)
            if command[0] == "ps":
                return "8888\n"  # PGID drifted -> PID reused, never signal
            if command[0] == "sh" and "cmdline" in command[2]:
                return "drifted\n"
            if command[0] == "test":
                return ""  # treated as dead (identity drift already detected)
            raise AssertionError(f"unexpected command: {command}")

        process_dead, port_released = _terminate_and_verify_remote(
            server,
            "ubuntu-local",
            ssh_spawn=fake_ssh,
            remote_listening_ports=lambda host: set(),
            deadline_seconds=0.0,
        )
        self.assertFalse(any(c[0] == "kill" for c in commands))
        self.assertTrue(process_dead)
        self.assertTrue(port_released)


class IntegrationSubstrateTest(unittest.TestCase):
    """Real production builders (fake transports) through substrate construction."""

    def test_run_path_receipts_pass_build_substrate_receipt(self) -> None:
        import shlex

        from pyreplab_harness.m3_prompt_only_execution import (
            _build_active_service_receipt,
            _collect_proxy_receipts,
            _poll_readiness,
            _preflight_barrier,
            _stop_live_lifecycle,
            _transition_stage,
        )

        run = _test_run_with_remote(
            _custom_remote_preflight(
                _service_status_sha256()
            ),
            "launching",
        )

        # --- server launch (real builder, fake SSH) ---
        identity = _MANIFEST["isolated_no_cache_server_identity"]
        off_argv = identity["server_argv"]
        joined = shlex.join(off_argv)
        launched = {}

        def fake_server_ssh(command):
            if command[0] == "ss":
                if launched.get("done"):
                    return (
                        'LISTEN 0 128 127.0.0.1:18082 0.0.0.0:* '
                        'users:(("llama-server",pid=5001,fd=3))\n'
                    )
                return ""
            if command[0] == "test":
                return ""  # log path absent
            if command[0] == "sh" and command[1] == "-c":
                script = command[2]
                if "setsid" in script:
                    launched["done"] = True
                    launched["marker"] = _script_run_marker(script)
                    return "5001\n"
                if "cmdline" in script:
                    return joined + "\n"
                if "environ" in script:
                    return f"PYREPLAB_RUN_MARKER={launched.get('marker')}\n"
            if command[0] == "ps":
                return command[command.index("-p") + 1] + "\n"
            raise AssertionError(f"unexpected command: {command}")

        server_receipt = launch_off_server_remote(run, ssh_spawn=fake_server_ssh)

        # --- tunnel launch (real builder, fake Popen) ---
        process = mock.Mock(pid=6001)
        tunnel_receipt, tunnel_owned = launch_local_tunnel(
            run, port_available=lambda p: True, popen=lambda c, **kw: process
        )

        # Readiness + slot clear are active-stage helpers: transition exactly
        # once after both launches complete.
        _transition_stage(run, "launching", "active")

        # --- readiness (real builder, fake HTTP) ---
        from pyreplab_harness.m3_prompt_only_execution import _HttpResponse

        def fake_http(url):
            if url.endswith("/slots"):
                return _HttpResponse(200, [{"id": 0, "is_processing": False}])
            return _HttpResponse(200, {"data": [{"id": RUN_MODEL_ALIAS}]})

        readiness_receipt = _poll_readiness(run, http_get=fake_http)

        # --- 72 slot clears (real builder, fake HTTP + fake slot-dir SSH) ---
        def slot_dir_ssh(command):
            if command[0] == "id" and command[1] == "-u":
                return "1000\n"
            if command[0] == "id" and command[1] == "-g":
                return "1000\n"
            if command[0] == "stat":
                return "directory|555|1000|1000\n"
            if command[0] == "find":
                return ""  # empty directory
            raise AssertionError(f"unexpected command: {command}")

        slot_receipts = [
            perform_slot_clear(
                run,
                http_get=fake_http,
                http_post=lambda u: _HttpResponse(200, {"id_slot": 0, "n_erased": 0}),
                ssh_spawn=slot_dir_ssh,
            )
            for _ in range(EXPECTED_CELLS)
        ]

        # --- proxy receipts via the real file-reading collector ---
        with tempfile.TemporaryDirectory() as directory:
            proxy_paths = []
            for index in range(EXPECTED_CELLS):
                path = Path(directory) / f"proxy-{index}.jsonl"
                path.write_text(
                    json.dumps(_valid_proxy_receipt(), sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                proxy_paths.append(str(path))
            proxy_receipts = _collect_proxy_receipts(proxy_paths)

        # --- teardown + active service (real builders, fake SSH) ---
        lifecycle = {
            "config": run.config,
            "server": server_receipt,
            "tunnel": tunnel_receipt,
            "tunnel_owned": tunnel_owned,
            "owned_proxies": [],
            "teardown_receipt": None,
            "active_service_after": None,
            "slot_action_dir_required": True,
        }

        def fake_teardown_ssh(command):
            if command[0] == "systemctl":
                return _service_status_text()
            if command[0] == "journalctl":
                return "" if "--after-cursor" in command else _baseline_journal_output()
            if command[0] == "ps":
                return command[command.index("-p") + 1] + "\n"
            if command[0] == "sh" and "cmdline" in command[2]:
                return joined + "\n"
            if command[0] == "sh" and "environ" in command[2]:
                return f"PYREPLAB_RUN_MARKER={launched.get('marker')}\n"
            if command[0] == "id" and command[1] == "-u":
                return "1000\n"
            if command[0] == "id" and command[1] == "-g":
                return "1000\n"
            if command[0] == "stat" and command[2] == "%s":
                return "100\n"  # remote log size
            if command[0] == "stat":
                return "directory|555|1000|1000\n"
            if command[0] == "find":
                return ""  # empty directory
            if command[0] == "rmdir":
                return ""  # remove the erase-only slot-action directory
            if command[0] == "test":
                return ""  # /proc/PID absent -> dead; slot-action dir absent
            if command[0] == "kill":
                return ""
            if command[0] == "rm":
                return ""  # PID file removed after verified death+port release
            if command[0] == "sha256sum":
                return "a" * 64 + "  /tmp/off.log\n"
            raise AssertionError(f"unexpected command: {command}")

        # Local teardown of the tunnel uses fakes so no real process group is
        # ever probed/signaled in the test environment.
        def fake_getpgid(pid):
            raise ProcessLookupError

        def fake_killpg(pgid, sig):
            raise ProcessLookupError

        teardown_receipt = _stop_live_lifecycle(
            run,
            lifecycle,
            ssh_spawn=fake_teardown_ssh,
            remote_listening_ports=lambda host: set(),
            deadline_seconds=0.0,
            getpgid=fake_getpgid,
            killpg=fake_killpg,
        )
        active_service_before = _build_active_service_receipt(
            _preflight_barrier(run.remote_preflight),
            mutated=False,
        )
        active_service_after = lifecycle["active_service_after"]

        (
            local_lease_acquire,
            remote_lease_acquire,
            local_lease_release,
            remote_lease_release,
        ) = _lease_receipts_for(_LEASE_AUTH_HASH)
        substrate = build_substrate_receipt(
            _MANIFEST,
            authorization_hash=_LEASE_AUTH_HASH,
            server_receipt=server_receipt,
            tunnel_receipt=tunnel_receipt,
            readiness_receipt=readiness_receipt,
            slot_clear_receipts=slot_receipts,
            proxy_receipts=proxy_receipts,
            active_service_before=active_service_before,
            active_service_after=active_service_after,
            teardown_receipt=teardown_receipt,
            source_commit="abc123",
            source_bundle_hash=run.source_bundle_hash,
            slot_action_dir_preparation_receipt=_valid_slot_dir_preparation_receipt(),
            generation_lease_acquire_receipt=remote_lease_acquire,
            generation_lease_release_receipt=remote_lease_release,
            generation_lease_local_acquire_receipt=local_lease_acquire,
            generation_lease_local_release_receipt=local_lease_release,
        )
        self.assertIs(substrate["substrate_valid"], True)
        validate_execution_substrate_receipt(substrate, _MANIFEST, _LEASE_AUTH_HASH)


class LifecycleFailureSafetyTest(unittest.TestCase):
    """Failure safety: cleanup on partial launch, PID-reuse and double-stop."""

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
            provider=RUN_PROVIDER,
            model=RUN_MODEL_ALIAS,
            thinking="off",
            unbrowser_binary=_MANIFEST["runtime_pins"]["unbrowser_path"],
            model_artifact=_MANIFEST["runtime_pins"]["model_artifact_path"],
            llama_server_binary=_MANIFEST["runtime_pins"]["llama_server_path"],
            endpoint_probe_receipt_path=_PROBE_RECEIPT_PATH,
            endpoint_probe_authorization_path=_PROBE_AUTHORIZATION_PATH,
            expected_endpoint_probe_authorization_hash=_PROBE_AUTHORIZATION_HASH,
        )

    def test_start_lifecycle_cleans_server_on_tunnel_failure(self) -> None:
        import shlex

        from pyreplab_harness.m3_prompt_only_execution import (
            _start_live_lifecycle,
        )

        run = _test_run("launching")
        lifecycle = {
            "config": run.config,
            "server": None,
            "tunnel": None,
            "tunnel_owned": None,
            "owned_proxies": [],
            "teardown_receipt": None,
            "active_service_after": None,
        }
        identity = _MANIFEST["isolated_no_cache_server_identity"]
        joined = shlex.join(identity["server_argv"])
        launched = {}
        kills = []

        def fake_server_ssh(command):
            if command[0] == "ss":
                if launched.get("done"):
                    return (
                        'LISTEN 0 128 127.0.0.1:18082 0.0.0.0:* '
                        'users:(("llama-server",pid=5001,fd=3))\n'
                    )
                return ""
            if command[0] == "sh" and "setsid" in command[2]:
                launched["done"] = True
                launched["marker"] = _script_run_marker(command[2])
                return "5001\n"
            if command[0] == "ps":
                return "5001\n"
            if command[0] == "sh" and "cmdline" in command[2]:
                return joined + "\n"
            if command[0] == "sh" and "environ" in command[2]:
                return f"PYREPLAB_RUN_MARKER={launched.get('marker')}\n"
            if command[0] == "test":
                return ""  # /proc/PID absent -> process dead
            if command[0] == "sha256sum":
                return "e" * 64 + "  " + command[1] + "\n"
            if command[0] == "stat" and command[2] == "%s":
                return "100\n"
            if command[0] == "kill":
                kills.append(command)
                return ""
            if command[0] == "rm":
                return ""
            raise AssertionError(f"unexpected command: {command}")

        def fail_popen(command, **kwargs):
            raise OSError("tunnel spawn failed")

        with self.assertRaisesRegex(OSError, "tunnel spawn failed"):
            _start_live_lifecycle(
                run,
                lifecycle,
                ssh_spawn=fake_server_ssh,
                popen=fail_popen,
                port_available=lambda p: True,
                remote_listening_ports=lambda host: set(),
            )
        # The already-launched server was cleaned up (a kill was attempted).
        self.assertTrue(any(c[0] == "kill" for c in kills))

    def test_terminate_owned_pid_reuse_not_signaled(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _OwnedProcess,
            _terminate_owned,
        )

        owned = _OwnedProcess(pid=100, process_group=100)
        signaled = []

        def fake_getpgid(pid):
            return 999  # PGID drifted -> PID reused

        def fake_killpg(pgid, sig):
            signaled.append((pgid, sig))

        # PGID drift is an UNVERIFIED failure: no signal, never stopped=True.
        self.assertFalse(
            _terminate_owned(owned, getpgid=fake_getpgid, killpg=fake_killpg)
        )
        self.assertEqual(signaled, [])
        self.assertFalse(owned.stopped)

    def test_terminate_owned_double_stop_is_idempotent(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _OwnedProcess,
            _terminate_owned,
        )

        owned = _OwnedProcess(pid=100, process_group=100)
        signaled = []

        def fake_getpgid(pid):
            raise ProcessLookupError  # leader already dead

        def fake_killpg(pgid, sig):
            raise ProcessLookupError  # entire group already gone

        first = _terminate_owned(owned, getpgid=fake_getpgid, killpg=fake_killpg)
        second = _terminate_owned(owned, getpgid=fake_getpgid, killpg=fake_killpg)
        self.assertTrue(first)
        self.assertTrue(second)
        self.assertEqual(signaled, [])

    def test_readiness_failure_triggers_teardown(self) -> None:
        state = {"stopped": 0}

        def fake_attempt(project_root, config, task, policy, attempt_id, args, **kwargs):
            return _make_attempt_item(attempt_id, args.sampling_seed, policy.id)

        def fake_lifecycle_start(run, lifecycle):
            lifecycle["server"] = _valid_server_receipt()
            lifecycle["tunnel"] = _valid_tunnel_receipt()
            return lifecycle

        def fake_stop(run, lifecycle):
            state["stopped"] += 1
            active_after = _valid_active_service_receipt()
            lifecycle["active_service_after"] = active_after
            lifecycle["teardown_receipt"] = _valid_teardown_receipt(active_after)
            return lifecycle["teardown_receipt"]

        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._revalidate_runtime_identity",
            return_value=None,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._acquire_generation_lease",

            side_effect=lambda run: _lease_receipts_for(run.authorization_hash)[:2],

            ), mock.patch(

            "pyreplab_harness.m3_prompt_only_execution._release_generation_lease",

            side_effect=lambda run: _release_outcome_for(run.authorization_hash),
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._prepare_slot_action_directory",
            return_value=_valid_slot_dir_preparation_receipt(),
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._start_live_lifecycle",
            side_effect=fake_lifecycle_start,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._stop_live_lifecycle",
            side_effect=fake_stop,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._poll_readiness",
            side_effect=RuntimeError("readiness timeout"),
        ):
            with self.assertRaisesRegex(RuntimeError, "readiness timeout"):
                run_authorized_prompt_only(**self._run_kwargs())
        # Teardown ran despite readiness failure.
        self.assertGreaterEqual(state["stopped"], 1)

    def test_env_restored_when_teardown_raises(self) -> None:
        original = os.environ.get("PI_CODING_AGENT_DIR")
        os.environ["PI_CODING_AGENT_DIR"] = "/sentinel/original"

        def fake_lifecycle_start(run, lifecycle):
            lifecycle["server"] = _valid_server_receipt()
            lifecycle["tunnel"] = _valid_tunnel_receipt()
            return lifecycle

        def fake_stop(run, lifecycle):
            raise RuntimeError("teardown boom")

        try:
            with mock.patch(
                "pyreplab_harness.m3_prompt_only_execution._revalidate_runtime_identity",
                return_value=None,
            ), mock.patch(
                "pyreplab_harness.m3_prompt_only_execution._acquire_generation_lease",
                side_effect=lambda run: _lease_receipts_for(run.authorization_hash)[:2],
            ), mock.patch(
                "pyreplab_harness.m3_prompt_only_execution._release_generation_lease",
                side_effect=lambda run: _release_outcome_for(run.authorization_hash),
            ), mock.patch(
                "pyreplab_harness.m3_prompt_only_execution._prepare_slot_action_directory",
                return_value=_valid_slot_dir_preparation_receipt(),
            ), mock.patch(
                "pyreplab_harness.m3_prompt_only_execution._start_live_lifecycle",
                side_effect=fake_lifecycle_start,
            ), mock.patch(
                "pyreplab_harness.m3_prompt_only_execution._stop_live_lifecycle",
                side_effect=fake_stop,
            ), mock.patch(
                "pyreplab_harness.m3_prompt_only_execution._poll_readiness",
                side_effect=RuntimeError("readiness timeout"),
            ):
                with self.assertRaisesRegex(RuntimeError, "teardown boom"):
                    run_authorized_prompt_only(**self._run_kwargs())
            # PI_CODING_AGENT_DIR was restored in the nested finally even though
            # teardown raised.
            self.assertEqual(
                os.environ.get("PI_CODING_AGENT_DIR"), "/sentinel/original"
            )
        finally:
            if original is None:
                os.environ.pop("PI_CODING_AGENT_DIR", None)
            else:
                os.environ["PI_CODING_AGENT_DIR"] = original

    def test_original_error_chained_on_unverified_teardown(self) -> None:
        def fake_lifecycle_start(run, lifecycle):
            lifecycle["server"] = _valid_server_receipt()
            lifecycle["tunnel"] = _valid_tunnel_receipt()
            return lifecycle

        def fake_stop(run, lifecycle):
            # Always unverified -> the terminal governance error.
            active_after = _valid_active_service_receipt()
            return _valid_teardown_receipt(
                active_after, verified=False, remote_process_dead=False
            )

        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._revalidate_runtime_identity",
            return_value=None,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._acquire_generation_lease",

            side_effect=lambda run: _lease_receipts_for(run.authorization_hash)[:2],

            ), mock.patch(

            "pyreplab_harness.m3_prompt_only_execution._release_generation_lease",

            side_effect=lambda run: _release_outcome_for(run.authorization_hash),
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._prepare_slot_action_directory",
            return_value=_valid_slot_dir_preparation_receipt(),
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._start_live_lifecycle",
            side_effect=fake_lifecycle_start,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._stop_live_lifecycle",
            side_effect=fake_stop,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._poll_readiness",
            side_effect=RuntimeError("original readiness failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "teardown could not be verified") as caught:
                run_authorized_prompt_only(**self._run_kwargs())
        # The terminal governance error chains the original run error.
        self.assertIsNotNone(caught.exception.__context__)
        self.assertIn("original readiness failure", str(caught.exception.__context__))


class StageAndTamperTest(unittest.TestCase):
    """Stage machine + context-digest tamper detection."""

    def test_replace_of_binary_fails_digest_verification(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import _require_capability

        tampered = replace(_VALIDATED_RUN, llama_server_binary="/other/llama-server")
        with self.assertRaisesRegex(RuntimeError, "digest mismatch"):
            _require_capability(tampered)

    def test_replace_of_config_fails_digest_verification(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import _require_capability

        tampered = replace(
            _VALIDATED_RUN,
            config=RemoteConfig("other-host", "/p", "/r", "python3"),
        )
        with self.assertRaisesRegex(RuntimeError, "digest mismatch"):
            _require_capability(tampered)

    def test_unconsumed_context_rejected_by_launch(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "allowed stages"):
            launch_off_server_remote(_test_run("validated"))

    def test_out_of_order_transition_rejected(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import _transition_stage

        run = _test_run("active")
        with self.assertRaisesRegex(RuntimeError, "expected 'consumed'"):
            _transition_stage(run, "consumed", "launching")

    def test_helpers_require_correct_stage(self) -> None:
        # launch_local_proxy requires the active stage.
        with self.assertRaisesRegex(RuntimeError, "allowed stages"):
            launch_local_proxy(
                _test_run("launching"),
                attempt_id="x",
                cell_id="y",
                sampling_seed=1,
                cache_runtime_receipt_hash="0" * 64,
                receipt_output=Path("/tmp/p.jsonl"),
                port_available=lambda p: True,
            )


class ListenerPollingTest(unittest.TestCase):
    """Listener-ownership polling: delayed bind + timeout cleanup."""

    def test_delayed_listener_bind_succeeds(self) -> None:
        import shlex

        identity = _MANIFEST["isolated_no_cache_server_identity"]
        joined = shlex.join(identity["server_argv"])
        state = {"polls": 0}

        def fake_ssh(command):
            if command[0] == "ss":
                state["polls"] += 1
                if state["polls"] >= 3:
                    return (
                        'LISTEN 0 128 127.0.0.1:18082 0.0.0.0:* '
                        'users:(("llama-server",pid=5001,fd=3))\n'
                    )
                return ""
            if command[0] == "sh" and "setsid" in command[2]:
                state["marker"] = _script_run_marker(command[2])
                return "5001\n"
            if command[0] == "ps":
                return "5001\n"
            if command[0] == "sh" and "cmdline" in command[2]:
                return joined + "\n"
            if command[0] == "sh" and "environ" in command[2]:
                return f"PYREPLAB_RUN_MARKER={state.get('marker')}\n"
            if command[0] == "test":
                return ""  # log path absent
            raise AssertionError(f"unexpected command: {command}")

        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution.time.sleep",
            return_value=None,
        ):
            receipt = launch_off_server_remote(
                _test_run("launching"),
                ssh_spawn=fake_ssh,
                deadline_seconds=5.0,
            )
        self.assertIs(receipt["listener_ownership"]["verified"], True)
        self.assertGreaterEqual(state["polls"], 3)

    def test_listener_timeout_cleans_up(self) -> None:
        import shlex

        identity = _MANIFEST["isolated_no_cache_server_identity"]
        joined = shlex.join(identity["server_argv"])
        kills = []
        state = {}

        def fake_ssh(command):
            if command[0] == "ss":
                return ""  # never binds
            if command[0] == "sh" and "setsid" in command[2]:
                state["marker"] = _script_run_marker(command[2])
                return "5001\n"
            if command[0] == "ps":
                return "5001\n"
            if command[0] == "sh" and "cmdline" in command[2]:
                return joined + "\n"
            if command[0] == "sh" and "environ" in command[2]:
                return f"PYREPLAB_RUN_MARKER={state.get('marker')}\n"
            if command[0] == "test":
                return ""
            if command[0] == "kill":
                kills.append(command)
                return ""
            if command[0] == "rm":
                return ""
            raise AssertionError(f"unexpected command: {command}")

        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution.time.sleep",
            return_value=None,
        ):
            with self.assertRaisesRegex(RuntimeError, "listener ownership timed out"):
                launch_off_server_remote(
                    _test_run("launching"),
                    ssh_spawn=fake_ssh,
                    deadline_seconds=0.0,
                )
        # The verified PID/PGID was terminated during cleanup.
        self.assertTrue(any(c[0] == "kill" for c in kills))


class LocalKillSafetyTest(unittest.TestCase):
    """Local ownership: completed child, failed kill, retained proxy."""

    def test_completed_child_is_never_signaled(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _OwnedProcess,
            _terminate_owned,
        )

        process = mock.Mock(pid=100, args=("ssh", "-N"))
        process.poll.return_value = 0  # leader already completed
        owned = _OwnedProcess(pid=100, process_group=100, process=process, argv=("ssh", "-N"))
        signals = []

        def fake_getpgid(pid):
            raise AssertionError("getpgid should not be called for a completed leader")

        def fake_killpg(pgid, sig):
            if sig == 0:
                raise ProcessLookupError  # the whole group is already gone
            signals.append((pgid, sig))

        self.assertTrue(_terminate_owned(owned, getpgid=fake_getpgid, killpg=fake_killpg))
        self.assertTrue(owned.stopped)
        self.assertEqual(signals, [])  # never TERM/KILL a gone group

    def test_surviving_descendants_are_signaled_after_leader_exit(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _OwnedProcess,
            _terminate_owned,
        )

        process = mock.Mock(pid=100, args=("ssh", "-N"))
        process.poll.return_value = 0  # leader exited, descendants survive
        owned = _OwnedProcess(pid=100, process_group=100, process=process, argv=("ssh", "-N"))
        group_alive = {"alive": True}
        signals = []

        def fake_killpg(pgid, sig):
            if sig == 0:
                if not group_alive["alive"]:
                    raise ProcessLookupError
                return
            signals.append((pgid, sig))
            group_alive["alive"] = False  # TERM/KILL removes the group

        self.assertTrue(_terminate_owned(owned, killpg=fake_killpg))
        self.assertTrue(owned.stopped)
        self.assertIn((100, 15), signals)  # SIGTERM sent to the surviving group

    def test_failed_kill_returns_false_and_leaves_tracked(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _OwnedProcess,
            _terminate_owned,
        )

        process = mock.Mock(pid=100, args=("ssh", "-N"))
        process.poll.return_value = None  # never completes
        owned = _OwnedProcess(pid=100, process_group=100, process=process, argv=("ssh", "-N"))
        signaled = []

        def fake_getpgid(pid):
            return 100

        def fake_killpg(pgid, sig):
            signaled.append((pgid, sig))

        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution.time.sleep",
            return_value=None,
        ):
            dead = _terminate_owned(
                owned, getpgid=fake_getpgid, killpg=fake_killpg, timeout_seconds=0.0
            )
        self.assertFalse(dead)
        self.assertFalse(owned.stopped)  # not marked stopped after failed kill
        self.assertGreaterEqual(len(signaled), 2)  # TERM then KILL

    def test_args_mismatch_never_signaled(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _OwnedProcess,
            _terminate_owned,
        )

        process = mock.Mock(pid=100, args=("different", "argv"))
        process.poll.return_value = None
        owned = _OwnedProcess(pid=100, process_group=100, process=process, argv=("ssh", "-N"))
        signaled = []

        def fake_getpgid(pid):
            raise AssertionError("getpgid should not be called on argv mismatch")

        def fake_killpg(pgid, sig):
            signaled.append((pgid, sig))

        # argv drift is an UNVERIFIED failure: no signal, never stopped=True.
        self.assertFalse(
            _terminate_owned(owned, getpgid=fake_getpgid, killpg=fake_killpg)
        )
        self.assertFalse(owned.stopped)
        self.assertEqual(signaled, [])

    def test_terminate_owned_pid_drift_is_unverified(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _OwnedProcess,
            _terminate_owned,
        )

        process = mock.Mock(pid=999, args=("ssh", "-N"))  # PID drifted
        owned = _OwnedProcess(pid=100, process_group=100, process=process, argv=("ssh", "-N"))
        signaled = []

        def fake_killpg(pgid, sig):
            signaled.append((pgid, sig))

        # PID drift is an UNVERIFIED failure: no signal, never stopped=True.
        self.assertFalse(_terminate_owned(owned, killpg=fake_killpg))
        self.assertFalse(owned.stopped)
        self.assertEqual(signaled, [])

    def test_terminate_owned_reaps_leader(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _OwnedProcess,
            _terminate_owned,
        )

        process = mock.Mock(pid=100, args=("ssh", "-N"))
        process.poll.return_value = 0  # leader exited (zombie)
        owned = _OwnedProcess(pid=100, process_group=100, process=process, argv=("ssh", "-N"))

        def fake_killpg(pgid, sig):
            raise ProcessLookupError  # the whole group is already gone

        self.assertTrue(_terminate_owned(owned, killpg=fake_killpg))
        process.wait.assert_called()  # the leader was reaped to avoid a zombie

    def test_getpgid_permission_error_is_unverified_non_signal(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _OwnedProcess,
            _terminate_owned,
        )

        process = mock.Mock(pid=100, args=("ssh", "-N"))
        process.poll.return_value = None
        owned = _OwnedProcess(pid=100, process_group=100, process=process, argv=("ssh", "-N"))
        signaled = []

        def fake_getpgid(pid):
            raise PermissionError  # cannot inspect group membership

        def fake_killpg(pgid, sig):
            signaled.append((pgid, sig))

        # getpgid PermissionError is UNVERIFIED: return False, never signal.
        self.assertFalse(_terminate_owned(owned, getpgid=fake_getpgid, killpg=fake_killpg))
        self.assertFalse(owned.stopped)
        self.assertEqual(signaled, [])

    def test_remote_teardown_requires_death_and_port_release(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _terminate_and_verify_remote,
        )

        server = _valid_server_receipt()
        server = {**server, "pid": 5001, "process_group": 5001}

        def fake_ssh(command):
            if command[0] == "ps":
                return "5001\n"
            if command[0] == "sh" and "cmdline" in command[2]:
                import shlex

                return shlex.join(server["server_argv"]) + "\n"
            if command[0] == "test":
                raise RuntimeError("process still alive")
            if command[0] == "kill":
                return ""
            raise AssertionError(f"unexpected command: {command}")

        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution.time.sleep",
            return_value=None,
        ):
            dead, released = _terminate_and_verify_remote(
                server,
                "ubuntu-local",
                ssh_spawn=fake_ssh,
                remote_listening_ports=lambda host: {18082},
                deadline_seconds=0.0,
            )
        self.assertFalse(dead)
        self.assertFalse(released)


class LifecycleBoundaryTest(unittest.TestCase):
    """Exact stage sets, marker persistence, expiry asymmetry, cleanup, retry."""

    def test_no_live_helper_at_closed_stage(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import _poll_readiness

        run = _test_run("closed")
        with self.assertRaisesRegex(RuntimeError, "allowed stages"):
            launch_off_server_remote(run)
        with self.assertRaisesRegex(RuntimeError, "allowed stages"):
            launch_local_tunnel(run)
        with self.assertRaisesRegex(RuntimeError, "allowed stages"):
            launch_local_proxy(
                run,
                attempt_id="x",
                cell_id="y",
                sampling_seed=1,
                cache_runtime_receipt_hash="0" * 64,
                receipt_output=Path("/tmp/p.jsonl"),
                port_available=lambda p: True,
            )
        with self.assertRaisesRegex(RuntimeError, "allowed stages"):
            perform_slot_clear(run, http_get=lambda u: None, http_post=lambda u: None)
        with self.assertRaisesRegex(RuntimeError, "allowed stages"):
            _poll_readiness(run, http_get=lambda u: None)

    def test_teardown_allowed_after_expiry(self) -> None:
        import shlex

        from pyreplab_harness.m3_prompt_only_execution import (
            _context_digest,
            _require_capability,
            _stop_live_lifecycle,
        )

        base = _test_run_with_remote(
            _custom_remote_preflight(
                _service_status_sha256()
            ),
            "active",
        )
        expired = replace(
            base,
            authorization_expires_at=datetime.now(timezone.utc) - timedelta(seconds=5),
        )
        expired = replace(expired, _digest=_context_digest(expired))
        # Live helpers reject the expired run.
        with self.assertRaisesRegex(RuntimeError, "expired"):
            _require_capability(expired, allowed_stages=("active",))

        lifecycle = {
            "config": expired.config,
            "server": _valid_server_receipt(),
            "tunnel": _valid_tunnel_receipt(),
            "tunnel_owned": None,
            "owned_proxies": [],
            "teardown_receipt": None,
            "active_service_after": None,
        }

        def fake_ssh(command):
            if command[0] == "systemctl":
                return _service_status_text()
            if command[0] == "journalctl":
                return "" if "--after-cursor" in command else _baseline_journal_output()
            if command[0] == "ps":
                return command[command.index("-p") + 1] + "\n"
            if command[0] == "sh" and "cmdline" in command[2]:
                return shlex.join(_valid_server_receipt()["server_argv"]) + "\n"
            if command[0] == "test":
                return ""
            if command[0] == "sha256sum":
                return "e" * 64 + "  " + command[1] + "\n"
            if command[0] == "stat" and command[2] == "%s":
                return "100\n"
            if command[0] == "kill":
                return ""
            raise AssertionError(f"unexpected command: {command}")

        receipt = _stop_live_lifecycle(
            expired,
            lifecycle,
            ssh_spawn=fake_ssh,
            remote_listening_ports=lambda host: set(),
            deadline_seconds=0.0,
        )
        self.assertIs(receipt["verified"], True)
        self.assertIs(expired._permit.stage, "closed")

    def test_consumed_marker_deleted_fails_live_action(self) -> None:
        run = _advance_run(_build_validated_run(), "active")
        run._permit.consumed_marker_path.unlink()
        with self.assertRaisesRegex(RuntimeError, "missing on disk"):
            perform_slot_clear(run, http_get=lambda u: None, http_post=lambda u: None)

    def test_consumed_marker_authorization_drift_fails_live_action(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            CONSUMED_SCHEMA_VERSION,
            _atomic_write_json,
        )

        run = _advance_run(_build_validated_run(), "active")
        path = run._permit.consumed_marker_path
        path.unlink()
        payload = {
            "schema_version": CONSUMED_SCHEMA_VERSION,
            "authorization_hash": "0" * 64,
            "consumed_at": "2026-08-15T00:00:00+00:00",
        }
        _atomic_write_json(path, {**payload, "consumed_hash": _canonical_hash(payload)})
        with self.assertRaisesRegex(RuntimeError, "another authorization"):
            perform_slot_clear(run, http_get=lambda u: None, http_post=lambda u: None)

    def test_post_spawn_pgid_ssh_error_attempts_cleanup(self) -> None:
        probes = []

        def fake_ssh(command):
            if command[0] == "ss":
                probes.append("ss")
                return ""
            if command[0] == "sh" and "setsid" in command[2]:
                return "5001\n"
            if command[0] == "ps":
                raise RuntimeError("ssh lost during pgid verification")
            if command[0] == "test":
                probes.append("test")
                return ""
            if command[0] == "kill":
                probes.append("kill")
                return ""
            if command[0] == "rm":
                return ""
            raise AssertionError(f"unexpected command: {command}")

        with self.assertRaisesRegex(RuntimeError, "ssh lost during pgid verification"):
            launch_off_server_remote(_test_run("launching"), ssh_spawn=fake_ssh)
        # Partial cleanup verified remote death even though the kill could not be
        # proven safe (identity unverifiable).
        self.assertIn("test", probes)

    def test_listener_poll_ssh_error_cleans_up(self) -> None:
        import shlex

        identity = _MANIFEST["isolated_no_cache_server_identity"]
        joined = shlex.join(identity["server_argv"])
        kills = []
        state = {"ss_calls": 0, "marker": None}

        def fake_ssh(command):
            if command[0] == "ss":
                state["ss_calls"] += 1
                # First call is the pre-launch port-free check; the second is
                # the first listener-poll iteration, which fails over SSH.
                if state["ss_calls"] >= 2:
                    raise RuntimeError("ssh failed during listener poll")
                return ""
            if command[0] == "sh" and "setsid" in command[2]:
                state["marker"] = _script_run_marker(command[2])
                return "5001\n"
            if command[0] == "ps":
                return "5001\n"
            if command[0] == "sh" and "cmdline" in command[2]:
                return joined + "\n"
            if command[0] == "sh" and "environ" in command[2]:
                return f"PYREPLAB_RUN_MARKER={state['marker']}\n"
            if command[0] == "test":
                return ""
            if command[0] == "kill":
                kills.append(command)
                return ""
            if command[0] == "rm":
                return ""
            raise AssertionError(f"unexpected command: {command}")

        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution.time.sleep",
            return_value=None,
        ):
            with self.assertRaisesRegex(RuntimeError, "ssh failed during listener poll"):
                launch_off_server_remote(
                    _test_run("launching"), ssh_spawn=fake_ssh, deadline_seconds=5.0
                )
        # A kill was attempted before the (unverifiable) cleanup surfaced.
        self.assertTrue(any(c[0] == "kill" for c in kills))

    def test_unverified_teardown_is_not_cached_and_retries(self) -> None:
        import shlex

        from pyreplab_harness.m3_prompt_only_execution import _stop_live_lifecycle

        run = _test_run_with_remote(
            _custom_remote_preflight(
                _service_status_sha256()
            ),
            "active",
        )
        server = {**_valid_server_receipt(), "pid": 5001, "process_group": 5001}
        argv_joined = shlex.join(server["server_argv"])
        lifecycle = {
            "config": run.config,
            "server": server,
            "tunnel": _valid_tunnel_receipt(),
            "tunnel_owned": None,
            "owned_proxies": [],
            "teardown_receipt": None,
            "active_service_after": None,
        }
        kills = []

        def fake_ssh(command):
            if command[0] == "systemctl":
                return _service_status_text()
            if command[0] == "journalctl":
                return "" if "--after-cursor" in command else _baseline_journal_output()
            if command[0] == "ps":
                return command[command.index("-p") + 1] + "\n"
            if command[0] == "sh" and "cmdline" in command[2]:
                return argv_joined + "\n"
            if command[0] == "test":
                raise RuntimeError("process still alive")
            if command[0] == "kill":
                kills.append(command)
                return ""
            raise AssertionError(f"unexpected command: {command}")

        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution.time.sleep",
            return_value=None,
        ):
            first = _stop_live_lifecycle(
                run,
                lifecycle,
                ssh_spawn=fake_ssh,
                remote_listening_ports=lambda host: set(),
                deadline_seconds=0.0,
            )
        self.assertIs(first["verified"], False)
        self.assertIsNone(lifecycle.get("teardown_receipt"))  # never cached
        self.assertEqual(run._permit.stage, "teardown")  # never closed
        first_kills = len(kills)
        self.assertGreaterEqual(first_kills, 1)

        # A subsequent teardown at stage teardown retries the retained cleanup.
        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution.time.sleep",
            return_value=None,
        ):
            second = _stop_live_lifecycle(
                run,
                lifecycle,
                ssh_spawn=fake_ssh,
                remote_listening_ports=lambda host: set(),
                deadline_seconds=0.0,
            )
        self.assertIs(second["verified"], False)
        self.assertGreater(len(kills), first_kills)  # retried the kill

    def test_teardown_retry_reuses_successful_slot_dir_removal(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _ensure_verified_teardown,
        )

        run = _test_run_with_remote(
            _custom_remote_preflight(_service_status_sha256()),
            "active",
        )
        lifecycle = {
            "config": run.config,
            "server": None,
            "tunnel": None,
            "tunnel_owned": None,
            "owned_proxies": [],
            "teardown_receipt": None,
            "active_service_after": None,
            "slot_action_dir_required": True,
            "slot_action_dir_removal_receipt": None,
        }
        state = {"dir_present": True, "rmdir_calls": 0, "journal_failures": 1}

        def fake_ssh(command):
            if command[0] == "systemctl":
                return _service_status_text()
            if command[0] == "journalctl":
                if state["journal_failures"]:
                    state["journal_failures"] -= 1
                    raise RuntimeError("transient journal failure")
                return _baseline_journal_output()
            if command[0] == "id" and command[1] == "-u":
                return "1000\n"
            if command[0] == "id" and command[1] == "-g":
                return "1000\n"
            if command[0] == "stat":
                if not state["dir_present"]:
                    raise RuntimeError("slot-action directory is absent")
                return "directory|555|1000|1000\n"
            if command[0] == "find":
                return ""
            if command[0] == "rmdir":
                if not state["dir_present"]:
                    raise RuntimeError("duplicate rmdir")
                state["dir_present"] = False
                state["rmdir_calls"] += 1
                return ""
            if command[0] == "test" and command[1] == "!":
                if state["dir_present"]:
                    raise RuntimeError("slot-action directory still exists")
                return ""
            raise AssertionError(f"unexpected command: {command}")

        receipt = _ensure_verified_teardown(
            run,
            lifecycle,
            retries=1,
            ssh_spawn=fake_ssh,
            remote_listening_ports=lambda host: set(),
            deadline_seconds=0.0,
        )
        self.assertIs(receipt["verified"], True)
        self.assertEqual(state["rmdir_calls"], 1)
        self.assertFalse(state["dir_present"])
        self.assertIsNotNone(lifecycle["slot_action_dir_removal_receipt"])
        self.assertEqual(run._permit.stage, "closed")
        self.assertFalse(run.paths["teardown_failure"].exists())

    def test_stop_cell_proxy_honors_failed_kill(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _OwnedProcess,
            _stop_cell_proxy,
        )

        owned = _OwnedProcess(pid=1, process_group=1)
        # Group death failed -> proxy stays tracked, port never checked.
        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._terminate_owned",
            return_value=False,
        ) as term:
            self.assertFalse(_stop_cell_proxy(owned))
            term.assert_called_once_with(owned)
        # Group death succeeded AND port is reusable -> fully stopped.
        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._terminate_owned",
            return_value=True,
        ) as term:
            self.assertTrue(
                _stop_cell_proxy(owned, port_available=lambda p: True)
            )
            term.assert_called_once_with(owned)
        # Group death succeeded but the port is still held past the bound ->
        # NOT stopped: the proxy must stay tracked for the final teardown
        # retry (v9: TIME_WAIT outlived the process on the fixed proxy port).
        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._terminate_owned",
            return_value=True,
        ) as term:
            self.assertFalse(
                _stop_cell_proxy(
                    owned,
                    port_available=lambda p: False,
                    wait_seconds=0.0,
                )
            )
            term.assert_called_once_with(owned)

    def test_unverified_teardown_raises_governance_error_and_persists(self) -> None:
        import shlex

        from pyreplab_harness.m3_prompt_only_execution import _ensure_verified_teardown

        run = _test_run_with_remote(
            _custom_remote_preflight(
                _service_status_sha256()
            ),
            "active",
        )
        server = {**_valid_server_receipt(), "pid": 5001, "process_group": 5001}
        argv_joined = shlex.join(server["server_argv"])
        lifecycle = {
            "config": run.config,
            "server": server,
            "tunnel": _valid_tunnel_receipt(),
            "tunnel_owned": None,
            "owned_proxies": [],
            "teardown_receipt": None,
            "active_service_after": None,
        }

        def fake_ssh(command):
            if command[0] == "systemctl":
                return _service_status_text()
            if command[0] == "journalctl":
                return "" if "--after-cursor" in command else _baseline_journal_output()
            if command[0] == "ps":
                return command[command.index("-p") + 1] + "\n"
            if command[0] == "sh" and "cmdline" in command[2]:
                return argv_joined + "\n"
            if command[0] == "test":
                raise RuntimeError("process still alive")
            if command[0] == "kill":
                return ""
            raise AssertionError(f"unexpected command: {command}")

        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution.time.sleep",
            return_value=None,
        ):
            with self.assertRaisesRegex(RuntimeError, "teardown could not be verified"):
                _ensure_verified_teardown(
                    run,
                    lifecycle,
                    ssh_spawn=fake_ssh,
                    remote_listening_ports=lambda host: set(),
                    deadline_seconds=0.0,
                )
        # A bounded teardown-failure receipt was persisted.
        self.assertTrue(run.paths["teardown_failure"].is_file())

    def test_teardown_remote_ssh_exception_produces_unverified_receipt(self) -> None:
        import shlex

        from pyreplab_harness.m3_prompt_only_execution import _stop_live_lifecycle

        run = _test_run_with_remote(
            _custom_remote_preflight(
                _service_status_sha256()
            ),
            "active",
        )
        server = _valid_server_receipt()
        argv_joined = shlex.join(server["server_argv"])
        lifecycle = {
            "config": run.config,
            "server": server,
            "tunnel": _valid_tunnel_receipt(),
            "tunnel_owned": None,
            "owned_proxies": [],
            "teardown_receipt": None,
            "active_service_after": None,
        }

        def fake_ssh(command):
            if command[0] == "systemctl":
                return _service_status_text()
            if command[0] == "journalctl":
                return "" if "--after-cursor" in command else _baseline_journal_output()
            if command[0] == "ps":
                return command[command.index("-p") + 1] + "\n"
            if command[0] == "sh" and "cmdline" in command[2]:
                return argv_joined + "\n"
            if command[0] == "test":
                raise OSError("ssh lost during remote death check")
            if command[0] == "kill":
                return ""
            raise AssertionError(f"unexpected command: {command}")

        # Never raises: converts the SSH exception into structured evidence.
        receipt = _stop_live_lifecycle(
            run,
            lifecycle,
            ssh_spawn=fake_ssh,
            remote_listening_ports=lambda host: set(),
            deadline_seconds=0.0,
        )
        self.assertIs(receipt["verified"], False)
        self.assertFalse(receipt["remote_process_dead"])
        self.assertTrue(any("remote_terminate" in e for e in receipt["errors"]))
        self.assertIsNone(lifecycle.get("teardown_receipt"))  # never cached

    def test_teardown_service_query_exception_produces_unverified_receipt(self) -> None:
        import shlex

        from pyreplab_harness.m3_prompt_only_execution import _stop_live_lifecycle

        run = _test_run_with_remote(
            _custom_remote_preflight(
                _service_status_sha256()
            ),
            "active",
        )
        server = _valid_server_receipt()
        argv_joined = shlex.join(server["server_argv"])
        lifecycle = {
            "config": run.config,
            "server": server,
            "tunnel": _valid_tunnel_receipt(),
            "tunnel_owned": None,
            "owned_proxies": [],
            "teardown_receipt": None,
            "active_service_after": None,
        }

        def fake_ssh(command):
            if command[0] == "systemctl":
                raise OSError("ssh lost during active-service query")
            if command[0] == "ps":
                return command[command.index("-p") + 1] + "\n"
            if command[0] == "sh" and "cmdline" in command[2]:
                return argv_joined + "\n"
            if command[0] == "test":
                return ""
            if command[0] == "kill":
                return ""
            raise AssertionError(f"unexpected command: {command}")

        receipt = _stop_live_lifecycle(
            run,
            lifecycle,
            ssh_spawn=fake_ssh,
            remote_listening_ports=lambda host: set(),
            deadline_seconds=0.0,
        )
        self.assertIs(receipt["verified"], False)
        self.assertFalse(receipt["active_service_unchanged"])
        self.assertEqual(receipt["active_service_after_receipt_hash"], "")
        self.assertTrue(any("active_service" in e for e in receipt["errors"]))

    def test_ensure_verified_teardown_retries_after_raised_exception(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import _ensure_verified_teardown

        run = _test_run_with_remote(
            _custom_remote_preflight(
                _service_status_sha256()
            ),
            "active",
        )
        lifecycle = {
            "config": run.config,
            "server": _valid_server_receipt(),
            "tunnel": _valid_tunnel_receipt(),
            "tunnel_owned": None,
            "owned_proxies": [],
            "teardown_receipt": None,
            "active_service_after": None,
        }
        calls = {"n": 0}

        def fake_stop(run, lifecycle):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("transient ssh failure")
            active_after = _valid_active_service_receipt()
            lifecycle["active_service_after"] = active_after
            lifecycle["teardown_receipt"] = _valid_teardown_receipt(active_after)
            return lifecycle["teardown_receipt"]

        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._stop_live_lifecycle",
            side_effect=fake_stop,
        ):
            receipt = _ensure_verified_teardown(run, lifecycle, retries=1)
        self.assertIs(receipt["verified"], True)
        self.assertEqual(calls["n"], 2)  # retried after the raised exception

    def test_pid_file_removed_only_after_death_and_port(self) -> None:
        import shlex

        from pyreplab_harness.m3_prompt_only_execution import _stop_live_lifecycle

        run = _test_run_with_remote(
            _custom_remote_preflight(
                _service_status_sha256()
            ),
            "active",
        )
        server = {
            **_valid_server_receipt(),
            "pid": 5001,
            "process_group": 5001,
            "pid_file_path": "/tmp/test-pid-file.pid",
        }
        argv_joined = shlex.join(server["server_argv"])
        lifecycle = {
            "config": run.config,
            "server": server,
            "tunnel": _valid_tunnel_receipt(),
            "tunnel_owned": None,
            "owned_proxies": [],
            "teardown_receipt": None,
            "active_service_after": None,
        }
        removed = []

        def fake_ssh(command):
            if command[0] == "systemctl":
                return _service_status_text()
            if command[0] == "journalctl":
                return "" if "--after-cursor" in command else _baseline_journal_output()
            if command[0] == "ps":
                return command[command.index("-p") + 1] + "\n"
            if command[0] == "sh" and "cmdline" in command[2]:
                return argv_joined + "\n"
            if command[0] == "test":
                raise RuntimeError("process still alive")
            if command[0] == "kill":
                return ""
            if command[0] == "rm":
                removed.append(command)
                return ""
            raise AssertionError(f"unexpected command: {command}")

        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution.time.sleep",
            return_value=None,
        ):
            receipt = _stop_live_lifecycle(
                run,
                lifecycle,
                ssh_spawn=fake_ssh,
                remote_listening_ports=lambda host: set(),
                deadline_seconds=0.0,
            )
        self.assertIs(receipt["verified"], False)
        self.assertFalse(receipt["remote_pid_file_removed"])
        self.assertEqual(removed, [])  # PID file not removed until verified

    def test_postrun_remote_bundle_drift_fails_closed(self) -> None:
        import copy

        from pyreplab_harness.m3_prompt_only_execution import (
            _require_remote_bundle_intact,
        )

        run = _test_run_with_remote(
            _custom_remote_preflight(
                _service_status_sha256()
            ),
            "active",
        )
        drifted = copy.deepcopy(dict(run.local_preflight["source_bundle_manifest"]))
        drifted["files"] = drifted["files"] + [
            {"path": "src/extra.py", "size": 1, "sha256": "a" * 64}
        ]

        def fake_ssh(host, command, timeout=120, stderr_fallback=False):
            if "source-bundle" in command:
                return json.dumps(
                    {"manifest": drifted, "read_only": True}, sort_keys=True
                )
            raise AssertionError(f"unexpected command: {command}")

        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._ssh_capture",
            side_effect=fake_ssh,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "post-run remote source bundle manifest drift"
            ):
                _require_remote_bundle_intact(run)

    def test_postrun_writable_bundle_fails_closed(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _require_remote_bundle_intact,
        )

        run = _test_run_with_remote(
            _custom_remote_preflight(
                _service_status_sha256()
            ),
            "active",
        )

        def fake_ssh(host, command, timeout=120, stderr_fallback=False):
            if "source-bundle" in command:
                return json.dumps(
                    {
                        "manifest": run.local_preflight["source_bundle_manifest"],
                        "read_only": False,
                    },
                    sort_keys=True,
                )
            raise AssertionError(f"unexpected command: {command}")

        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._ssh_capture",
            side_effect=fake_ssh,
        ):
            with self.assertRaisesRegex(RuntimeError, "not read-only"):
                _require_remote_bundle_intact(run)

    def test_prelaunch_remote_bundle_drift_fails_closed(self) -> None:
        import copy

        from pyreplab_harness.m3_prompt_only_execution import (
            _revalidate_runtime_identity,
        )

        run = _test_run_with_remote(
            _custom_remote_preflight(
                _service_status_sha256()
            ),
            "active",
        )
        pins = run.manifest["runtime_pins"]
        drifted = copy.deepcopy(dict(run.local_preflight["source_bundle_manifest"]))
        drifted["files"] = drifted["files"] + [
            {"path": "src/extra.py", "size": 1, "sha256": "a" * 64}
        ]

        def fake_ssh(host, command, timeout=120, stderr_fallback=False):
            if "source-bundle" in command:
                return json.dumps(
                    {"manifest": drifted, "read_only": True}, sort_keys=True
                )
            if command[0] == "sha256sum":
                target = command[1]
                if target == run.unbrowser_binary:
                    return pins["unbrowser_sha256"] + "  " + target
                if target == run.model_artifact:
                    return pins["model_artifact_sha256"] + "  " + target
                if target == run.llama_server_binary:
                    return pins["llama_server_sha256"] + "  " + target
                raise AssertionError(f"unexpected sha256sum: {command}")
            if command[-1] == "--version" and command[0] == run.llama_server_binary:
                return pins["llama_server_version"]
            if command[-1] == "--version" and command[0] == run.unbrowser_binary:
                return pins["unbrowser_version"]
            if command[0] == "bwrap":
                return pins["bubblewrap_version"]
            if command[0] == "systemctl":
                return _service_status_text()
            if command[0] == "journalctl":
                return "" if "--after-cursor" in command else _baseline_journal_output()
            raise AssertionError(f"unexpected command: {command}")

        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._local_pi_sha256",
            return_value=pins["pi_cli_sha256"],
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._local_pi_version",
            return_value=pins["pi_version"],
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._ssh_capture",
            side_effect=fake_ssh,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._remote_listening_ports",
            return_value=set(),
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._local_port_available",
            return_value=True,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "TOCTOU remote source bundle manifest drift"
            ):
                _revalidate_runtime_identity(run)


class OrphanRecoveryTest(unittest.TestCase):
    """Remote launch recovery via the PID file + /proc marker scan."""

    def _launch_fake(self, launch_response: str) -> dict:
        import shlex

        identity = _MANIFEST["isolated_no_cache_server_identity"]
        joined = shlex.join(identity["server_argv"])
        pid_file = "/tmp/test-orphan-recovery.pid"
        state = {"marker": None}
        kills = []
        removed = []

        def fake_ssh(command):
            if command[0] == "ss":
                return ""
            if command[0] == "sh" and "setsid" in command[2]:
                state["marker"] = _script_run_marker(command[2])
                return launch_response
            if command[0] == "cat" and command[1] == pid_file:
                return f"5001 {state['marker']}\n"
            if command[0] == "ps":
                return "5001\n"
            if command[0] == "sh" and "cmdline" in command[2]:
                return joined + "\n"
            if command[0] == "sh" and "environ" in command[2]:
                return f"PYREPLAB_RUN_MARKER={state['marker']}\n"
            if command[0] == "test":
                return ""
            if command[0] == "kill":
                kills.append(command)
                return ""
            if command[0] == "rm":
                removed.append(command)
                return ""
            raise AssertionError(f"unexpected command: {command}")

        return {
            "fake_ssh": fake_ssh,
            "pid_file": pid_file,
            "kills": kills,
            "removed": removed,
        }

    def test_lost_launch_response_recovers_via_pid_file(self) -> None:
        env = self._launch_fake("")  # SSH loses stdout
        with self.assertRaisesRegex(RuntimeError, "numeric"):
            launch_off_server_remote(
                _test_run("launching"),
                ssh_spawn=env["fake_ssh"],
                pid_file_path=env["pid_file"],
            )
        # The recovered PID was terminated and the PID file removed.
        self.assertTrue(any(c[0] == "kill" for c in env["kills"]))
        self.assertTrue(any(c[0] == "rm" for c in env["removed"]))

    def test_nonnumeric_launch_response_recovers_via_pid_file(self) -> None:
        env = self._launch_fake("not-a-pid")
        with self.assertRaisesRegex(RuntimeError, "numeric"):
            launch_off_server_remote(
                _test_run("launching"),
                ssh_spawn=env["fake_ssh"],
                pid_file_path=env["pid_file"],
            )
        self.assertTrue(any(c[0] == "kill" for c in env["kills"]))
        self.assertTrue(any(c[0] == "rm" for c in env["removed"]))

    def test_unrecoverable_orphan_persists_receipt(self) -> None:
        # No PID file content and no /proc marker -> ownership undeterminable.
        pid_file = "/tmp/test-orphan-recovery.pid"

        def fake_ssh(command):
            if command[0] == "ss":
                return ""
            if command[0] == "sh" and "setsid" in command[2]:
                return "not-a-pid"
            if command[0] == "cat":
                return ""  # PID file empty
            if command[0] == "sh" and "grep" in command[2]:
                return ""  # no marker found
            if command[0] == "test":
                return ""  # log path absent
            raise AssertionError(f"unexpected command: {command}")

        with self.assertRaisesRegex(RuntimeError, "ownership could not be determined"):
            launch_off_server_remote(
                _test_run("launching"),
                ssh_spawn=fake_ssh,
                pid_file_path=pid_file,
            )

    def test_tunnel_failure_requires_verified_remote_cleanup(self) -> None:
        import shlex

        from pyreplab_harness.m3_prompt_only_execution import _start_live_lifecycle

        identity = _MANIFEST["isolated_no_cache_server_identity"]
        joined = shlex.join(identity["server_argv"])
        run = _test_run("launching")
        lifecycle = {
            "config": run.config,
            "server": None,
            "tunnel": None,
            "tunnel_owned": None,
            "owned_proxies": [],
            "teardown_receipt": None,
            "active_service_after": None,
        }
        state = {"marker": None}
        launched = {}

        def fake_ssh(command):
            if command[0] == "ss":
                if launched.get("done"):
                    return (
                        'LISTEN 0 128 127.0.0.1:18082 0.0.0.0:* '
                        'users:(("llama-server",pid=5001,fd=3))\n'
                    )
                return ""
            if command[0] == "sh" and "setsid" in command[2]:
                launched["done"] = True
                state["marker"] = _script_run_marker(command[2])
                return "5001\n"
            if command[0] == "ps":
                return "5001\n"
            if command[0] == "sh" and "cmdline" in command[2]:
                return joined + "\n"
            if command[0] == "sh" and "environ" in command[2]:
                return f"PYREPLAB_RUN_MARKER={state['marker']}\n"
            if command[0] == "test":
                if any("/proc/" in arg for arg in command):
                    raise RuntimeError("process still alive")
                return ""  # log path absent
            if command[0] == "kill":
                return ""
            raise AssertionError(f"unexpected command: {command}")

        def fail_popen(command, **kwargs):
            raise OSError("tunnel spawn failed")

        with self.assertRaisesRegex(RuntimeError, "remote cleanup could not be verified"):
            _start_live_lifecycle(
                run,
                lifecycle,
                ssh_spawn=fake_ssh,
                popen=fail_popen,
                port_available=lambda p: True,
                remote_listening_ports=lambda host: set(),
                deadline_seconds=0.0,
            )
        # Partial server ownership is preserved for the outer finally.
        self.assertIsNotNone(lifecycle["server"])


class UnbrowserVersionParserTest(unittest.TestCase):
    def test_parse_real_raw_output(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _parse_unbrowser_version,
        )

        self.assertEqual(_parse_unbrowser_version("unbrowser 0.0.19\n"), "0.0.19")
        self.assertEqual(_parse_unbrowser_version("unbrowser 0.0.19"), "0.0.19")

    def test_parse_bare_output(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _parse_unbrowser_version,
        )

        self.assertEqual(_parse_unbrowser_version("0.0.19\n"), "0.0.19")
        self.assertEqual(_parse_unbrowser_version("0.0.19"), "0.0.19")

    def test_parse_malformed_rejected(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _parse_unbrowser_version,
        )

        for bad in ("garbage", "0.0", "v0.0.19", "0.0.19.0", "unbrowser dev"):
            with self.assertRaises(RuntimeError):
                _parse_unbrowser_version(bad)

    def test_parse_extra_tokens_rejected(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _parse_unbrowser_version,
        )

        for bad in (
            "unbrowser 0.0.19 0.0.20",
            "0.0.19 0.0.20",
            "unbrowser 0.0.19 extra",
        ):
            with self.assertRaises(RuntimeError):
                _parse_unbrowser_version(bad)


class BundleReadOnlyTest(unittest.TestCase):
    """_bundle_is_read_only: dirs/files/roots all non-writable, fail-closed."""

    def _root(self) -> Path:
        root = Path(tempfile.mkdtemp(prefix="pyreplab-ppo-ro-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        (root / "src").mkdir()
        (root / "src" / "sub").mkdir()
        (root / "src" / "sub" / "a.py").write_text("x = 1\n", encoding="utf-8")
        (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        return root

    def test_all_non_writable_is_read_only(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import _bundle_is_read_only

        root = self._root()
        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution.os.access",
            return_value=False,
        ):
            self.assertTrue(_bundle_is_read_only(root))

    def test_writable_project_root_fails(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import _bundle_is_read_only

        root = self._root()
        resolved = root.resolve()

        def fake_access(path, mode):
            return str(Path(path).resolve()) == str(resolved)

        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution.os.access",
            side_effect=fake_access,
        ):
            self.assertFalse(_bundle_is_read_only(root))

    def test_writable_subdirectory_fails(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import _bundle_is_read_only

        root = self._root()

        def fake_access(path, mode):
            return Path(path).name == "sub"  # a writable subdir permits replacement

        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution.os.access",
            side_effect=fake_access,
        ):
            self.assertFalse(_bundle_is_read_only(root))

    def test_missing_root_fails_closed(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import _bundle_is_read_only

        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(_bundle_is_read_only(Path(directory) / "missing"))

    def test_symlink_namespace_root_fails_closed(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import _bundle_is_read_only

        root = Path(tempfile.mkdtemp(prefix="pyreplab-ppo-ro-symlink-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        real = root / "real"
        real.mkdir()
        (real / "a.py").write_text("x = 1\n", encoding="utf-8")
        (root / "src").symlink_to(real)
        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution.os.access",
            return_value=False,
        ):
            with self.assertRaises(ValueError):
                _bundle_is_read_only(root)


class ActiveServiceQuiescenceTest(unittest.TestCase):
    """Passive active-service quiescence barrier (systemctl + journalctl)."""

    def _barrier(self, *, status_text=None, journal=None, fake_ssh=None):
        from pyreplab_harness.m3_prompt_only_execution import (
            _establish_quiescence_barrier,
        )

        if fake_ssh is None:

            def fake_ssh(command):
                if command[0] == "systemctl":
                    return (
                        status_text
                        if status_text is not None
                        else _service_status_text()
                    )
                if command[0] == "journalctl":
                    if "--after-cursor" in command:
                        return ""
                    return journal if journal is not None else _baseline_journal_output()
                raise AssertionError(f"unexpected command: {command}")

        return _establish_quiescence_barrier(fake_ssh)

    def test_real_json_sample_sleeping_accepted(self) -> None:
        barrier = self._barrier()
        self.assertIs(barrier["quiescent"], True)
        self.assertEqual(barrier["state"], "sleeping")
        self.assertEqual(barrier["boot_id"], _SVC_BOOT_ID)
        self.assertEqual(barrier["invocation_id"], _SVC_INVOCATION_ID)
        self.assertEqual(barrier["high_water_cursor"], _SVC_HIGH_WATER_CURSOR)
        self.assertEqual(barrier["state_event_cursor"], _SVC_STATE_EVENT_CURSOR)

    def test_barrier_uses_only_systemctl_and_journalctl(self) -> None:
        commands = []

        def fake_ssh(command):
            commands.append(command)
            if command[0] == "systemctl":
                return _service_status_text()
            if command[0] == "journalctl":
                return _baseline_journal_output()
            raise AssertionError(f"unexpected command: {command}")

        self._barrier(fake_ssh=fake_ssh)
        self.assertTrue(commands)
        for command in commands:
            self.assertIn(command[0], ("systemctl", "journalctl"))
            joined = " ".join(command)
            for forbidden in (
                "curl",
                "wget",
                "http",
                "8081",
                "8082",
                "18081",
                "18082",
                "18083",
                "18084",
                "start",
                "stop",
                "restart",
                "chmod",
                "chown",
            ):
                self.assertNotIn(forbidden, joined)

    def test_active_and_running_required(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not active"):
            self._barrier(status_text=_service_status_text(active_state="inactive"))
        with self.assertRaisesRegex(RuntimeError, "not running"):
            self._barrier(status_text=_service_status_text(sub_state="exited"))
        with self.assertRaisesRegex(RuntimeError, "MainPID"):
            self._barrier(status_text=_service_status_text(main_pid="0"))
        with self.assertRaisesRegex(RuntimeError, "canary port"):
            self._barrier(status_text=_service_status_text(extra_port="18082"))

    def test_last_record_order_wins_over_cursor_text(self) -> None:
        # The final record cursor is ``i=10``, which string-sorts BELOW ``i=f``
        # (the intermediate heartbeat). Only chronological last-record order —
        # never string comparison — yields the correct high-water.
        barrier = self._barrier()
        self.assertEqual(barrier["high_water_cursor"], _SVC_HIGH_WATER_CURSOR)
        self.assertEqual(barrier["state_event_cursor"], _SVC_STATE_EVENT_CURSOR)
        self.assertIn("i=10", barrier["high_water_cursor"])
        self.assertNotIn("i=f", barrier["high_water_cursor"])

    def test_canary_port_check_scoped_to_exec_start_only(self) -> None:
        # A canary port in a non-ExecStart field (e.g. ControlGroup) is ignored.
        status = _service_status_text().replace(
            "ControlGroup=/user.slice/",
            "ControlGroup=/user.slice-18082/",
        )
        self._barrier(status_text=status)  # no raise
        # A canary port in ExecStart still fails.
        with self.assertRaisesRegex(RuntimeError, "ExecStart references canary port"):
            self._barrier(status_text=_service_status_text(extra_port="18084"))

    def test_transport_raise_fails_closed(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _establish_quiescence_barrier,
        )

        def fake_ssh(command):
            raise RuntimeError("ssh transport failure")

        with self.assertRaisesRegex(RuntimeError, "ssh transport failure"):
            _establish_quiescence_barrier(fake_ssh)

    def test_replayed_record_is_contamination(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _check_quiescence_activity,
        )

        baseline = {
            "boot_id": _SVC_BOOT_ID,
            "invocation_id": _SVC_INVOCATION_ID,
            "high_water_cursor": _SVC_HIGH_WATER_CURSOR,
        }
        replayed = _journal_line(_SVC_MIDDLE_CURSOR, "gemma-router replayed") + "\n"

        def fake_ssh(command):
            if command[0] == "journalctl":
                return replayed
            raise AssertionError(f"unexpected command: {command}")

        self.assertEqual(len(_check_quiescence_activity(fake_ssh, baseline)), 1)

    def test_after_cursor_uses_bounded_existence_probe(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _check_quiescence_activity,
        )

        baseline = {"high_water_cursor": _SVC_HIGH_WATER_CURSOR}
        commands = []

        def fake_ssh(command):
            commands.append(command)
            return ""

        _check_quiescence_activity(fake_ssh, baseline)
        (command,) = commands
        self.assertIn("--after-cursor", command)
        self.assertIn("-n", command)
        self.assertEqual(command[command.index("-n") + 1], "1")

    def test_cell_status_drift_detected_immediately(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import _check_cell_quiescence

        run = _test_run_with_remote(
            _custom_remote_preflight(_service_status_sha256()), "active"
        )

        def fake_ssh(host, command, timeout=120, stderr_fallback=False):
            if command[0] == "systemctl":
                return _service_status_text(main_pid="99999")  # MainPID drifted
            raise AssertionError(f"unexpected command: {command}")

        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._ssh_capture",
            side_effect=fake_ssh,
        ):
            with self.assertRaisesRegex(RuntimeError, "main_pid drifted during a cell"):
                _check_cell_quiescence(run)

    def test_cell_restart_detected_immediately(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import _check_cell_quiescence

        run = _test_run_with_remote(
            _custom_remote_preflight(_service_status_sha256()), "active"
        )

        def fake_ssh(host, command, timeout=120, stderr_fallback=False):
            if command[0] == "systemctl":
                return _service_status_text(invocation_id="deadbeefdeadbeefdeadbeefdeadbeef")
            raise AssertionError(f"unexpected command: {command}")

        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._ssh_capture",
            side_effect=fake_ssh,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "invocation_id drifted during a cell"
            ):
                _check_cell_quiescence(run)

    def test_cell_journal_restart_contamination(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import _check_cell_quiescence

        run = _test_run_with_remote(
            _custom_remote_preflight(_service_status_sha256()), "active"
        )

        def fake_ssh(host, command, timeout=120, stderr_fallback=False):
            if command[0] == "systemctl":
                return _service_status_text()  # status still matches baseline
            if command[0] == "journalctl":
                return (
                    _journal_line(
                        "s=0123456789abcdef0123456789abcdef;i=11;"
                        f"b={_SVC_BOOT_ID};m=300;t=1000003;x=17",
                        "gemma-router startup",
                        invocation_id="deadbeefdeadbeefdeadbeefdeadbeef",
                    )
                    + "\n"
                )
            raise AssertionError(f"unexpected command: {command}")

        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._ssh_capture",
            side_effect=fake_ssh,
        ):
            with self.assertRaisesRegex(RuntimeError, "restarted/rebooted during a cell"):
                _check_cell_quiescence(run)

    def test_stale_old_invocation_sleeping_event_rejected(self) -> None:
        stale = (
            _journal_line(
                _SVC_STATE_EVENT_CURSOR,
                f"[32831] cmd_child_to_router:state:{_SVC_STATE_PAYLOAD}",
                invocation_id="deadbeefdeadbeefdeadbeefdeadbeef",
            )
            + "\n"
        )
        with self.assertRaisesRegex(
            RuntimeError, "final record does not match the bound"
        ):
            self._barrier(journal=stale)

    def test_current_boot_mismatch_rejected(self) -> None:
        mixed_boot = (
            "\n".join(
                [
                    _journal_line(
                        _SVC_STATE_EVENT_CURSOR,
                        f"[32831] cmd_child_to_router:state:{_SVC_STATE_PAYLOAD}",
                        boot_id="ffffffffffffffffffffffffffffffff",
                    ),
                    _journal_line(
                        _SVC_HIGH_WATER_CURSOR,
                        "gemma-router heartbeat",
                        boot_id=_SVC_BOOT_ID,
                    ),
                ]
            )
            + "\n"
        )
        with self.assertRaisesRegex(RuntimeError, "boot id is missing/ambiguous"):
            self._barrier(journal=mixed_boot)

    def test_malformed_state_rejected(self) -> None:
        malformed = (
            _journal_line(
                _SVC_STATE_EVENT_CURSOR,
                "[32831] cmd_child_to_router:state:{not-json}",
            )
            + "\n"
        )
        with self.assertRaisesRegex(RuntimeError, "not valid JSON"):
            self._barrier(journal=malformed)

    def test_missing_cursor_rejected(self) -> None:
        bad = (
            json.dumps(
                {
                    "_BOOT_ID": _SVC_BOOT_ID,
                    "_SYSTEMD_INVOCATION_ID": _SVC_INVOCATION_ID,
                    "MESSAGE": f"[32831] cmd_child_to_router:state:{_SVC_STATE_PAYLOAD}",
                },
                sort_keys=True,
            )
            + "\n"
        )
        with self.assertRaisesRegex(RuntimeError, "missing its cursor"):
            self._barrier(journal=bad)

    def test_status_double_snapshot_drift_rejected(self) -> None:
        calls = {"n": 0}

        def fake_ssh(command):
            if command[0] == "systemctl":
                calls["n"] += 1
                if calls["n"] == 1:
                    return _service_status_text()
                return _service_status_text(main_pid="99999")
            if command[0] == "journalctl":
                return _baseline_journal_output()
            raise AssertionError(f"unexpected command: {command}")

        with self.assertRaisesRegex(RuntimeError, "drifted during the barrier"):
            self._barrier(fake_ssh=fake_ssh)

    def test_post_cell_activity_detected(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _check_quiescence_activity,
        )

        baseline = {
            "boot_id": _SVC_BOOT_ID,
            "invocation_id": _SVC_INVOCATION_ID,
            "high_water_cursor": _SVC_HIGH_WATER_CURSOR,
        }
        new_event = (
            _journal_line(
                "s=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef;i=3;"
                f"b={_SVC_BOOT_ID};m=300;t=1000002;x=3",
                "gemma-router received a request",
            )
            + "\n"
        )

        def fake_ssh(command):
            if command[0] == "journalctl":
                return new_event if "--after-cursor" in command else ""
            raise AssertionError(f"unexpected command: {command}")

        contaminated = _check_quiescence_activity(fake_ssh, baseline)
        self.assertEqual(len(contaminated), 1)

    def test_post_cell_clean_no_event(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _check_quiescence_activity,
        )

        baseline = {
            "boot_id": _SVC_BOOT_ID,
            "invocation_id": _SVC_INVOCATION_ID,
            "high_water_cursor": _SVC_HIGH_WATER_CURSOR,
        }

        def fake_ssh(command):
            if command[0] == "journalctl":
                return ""
            raise AssertionError(f"unexpected command: {command}")

        self.assertEqual(_check_quiescence_activity(fake_ssh, baseline), [])

    def test_final_quiescence_restart_invocation_drift(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _require_final_quiescence,
        )

        run = _test_run_with_remote(
            _custom_remote_preflight(_service_status_sha256()), "active"
        )
        new_inv = "deadbeefdeadbeefdeadbeefdeadbeef"

        def fake_ssh(host, command, timeout=120, stderr_fallback=False):
            if command[0] == "systemctl":
                # The service was restarted: a new InvocationID.
                return _service_status_text(invocation_id=new_inv)
            if command[0] == "journalctl":
                return (
                    _journal_line(
                        _SVC_STATE_EVENT_CURSOR,
                        f"[32831] cmd_child_to_router:state:{_SVC_STATE_PAYLOAD}",
                        invocation_id=new_inv,
                    )
                    + "\n"
                )
            raise AssertionError(f"unexpected command: {command}")

        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._ssh_capture",
            side_effect=fake_ssh,
        ):
            with self.assertRaisesRegex(RuntimeError, "invocation_id drift"):
                _require_final_quiescence(run)

    def test_prelaunch_new_event_drift(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _revalidate_runtime_identity,
        )

        run = _test_run_with_remote(
            _custom_remote_preflight(_service_status_sha256()), "active"
        )
        pins = run.manifest["runtime_pins"]

        def fake_ssh(host, command, timeout=120, stderr_fallback=False):
            if command[0] == "systemctl":
                return _service_status_text()
            if command[0] == "journalctl":
                # A new (post-baseline) state event cursor + hash.
                return (
                    "\n".join(
                        [
                            _journal_line(
                                _SVC_STATE_EVENT_CURSOR,
                                f"[32831] cmd_child_to_router:state:{_SVC_STATE_PAYLOAD}",
                            ),
                            _journal_line(
                                "s=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef;i=9;"
                                f"b={_SVC_BOOT_ID};m=900;t=9999999;x=9",
                                "gemma-router woke up",
                            ),
                        ]
                    )
                    + "\n"
                )
            if "source-bundle" in command:
                return json.dumps(
                    {
                        "manifest": run.local_preflight["source_bundle_manifest"],
                        "read_only": True,
                    },
                    sort_keys=True,
                )
            if command[0] == "sha256sum":
                target = command[1]
                if target == run.unbrowser_binary:
                    return pins["unbrowser_sha256"] + "  " + target
                if target == run.model_artifact:
                    return pins["model_artifact_sha256"] + "  " + target
                if target == run.llama_server_binary:
                    return pins["llama_server_sha256"] + "  " + target
                raise AssertionError(f"unexpected sha256sum: {command}")
            if command[-1] == "--version" and command[0] == run.llama_server_binary:
                return pins["llama_server_version"]
            if command[-1] == "--version" and command[0] == run.unbrowser_binary:
                return pins["unbrowser_version"]
            if command[0] == "bwrap":
                return pins["bubblewrap_version"]
            if command[0] == "test":
                return ""  # slot-action dir + generation lease absent
            raise AssertionError(f"unexpected command: {command}")

        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._local_pi_sha256",
            return_value=pins["pi_cli_sha256"],
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._local_pi_version",
            return_value=pins["pi_version"],
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._ssh_capture",
            side_effect=fake_ssh,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._remote_listening_ports",
            return_value=set(),
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._local_port_available",
            return_value=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "TOCTOU active service"):
                _revalidate_runtime_identity(run)


def _make_probe_request(result_path: Path | None = None) -> dict:
    result = result_path or (PROJECT_ROOT / ".runs" / "endpoint-probe-receipt.json")
    return build_endpoint_probe_request(
        _MANIFEST,
        _REGISTRY,
        _LOCAL,
        _REMOTE,
        project_root=PROJECT_ROOT,
        result_path=result,
    )


def _make_probe_authorization(
    request: dict,
    *,
    approved_by: str = "test-operator",
    expires_seconds: int = 3600,
    authorization_id: str | None = None,
) -> tuple[dict, str]:
    now = datetime.now(timezone.utc)
    payload = {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "authorization_id": authorization_id or f"probe-auth-{uuid4().hex[:12]}",
        "screen_id": request["screen_id"],
        "execution_generation": EXECUTION_GENERATION,
        "manifest_hash": request["manifest_hash"],
        "registry_hash": request["registry_hash"],
        "local_preflight_hash": request["local_preflight_hash"],
        "remote_preflight_hash": request["remote_preflight_hash"],
        "simulator_report_hash": request["simulator_report_hash"],
        "source_tree_hash": request["source_tree_hash"],
        "source_bundle_hash": request["source_bundle_hash"],
        "remote_identity": request["remote_identity"],
        "provider_config": request["provider_config"],
        "python_executable": request["python_executable"],
        "result_filename": request["result_filename"],
        "result_path": request["result_path"],
        "max_cells": 0,
        "max_panels": 0,
        "budget": request["budget"],
        "server_lifecycle": request["server_lifecycle"],
        "severe_veto_contract": request["severe_veto_contract"],
        "approved_by": approved_by,
        "approved_at": now.isoformat(),
        "expires_at": (now + timedelta(seconds=expires_seconds)).isoformat(),
        "authorization_statement": PROBE_AUTHORIZATION_STATEMENT,
        "live_model_execution_authorized": True,
        "server_launch_authorized": True,
        "task_inference_authorized": False,
        "single_use": True,
        "authorization_scope": "endpoint_probe",
        "endpoint_probe_receipt_hash": None,
        "endpoint_probe_authorization_hash": None,
    }
    authorization_hash = _canonical_hash(payload)
    return {**payload, "authorization_hash": authorization_hash}, authorization_hash


class OffServerBindingPolicyTest(unittest.TestCase):
    def test_binding_carries_erase_only_slot_save_path(self) -> None:
        identity = _MANIFEST["isolated_no_cache_server_identity"]
        argv = identity["server_argv"]
        self.assertIn("--slot-save-path", argv)
        idx = argv.index("--slot-save-path")
        self.assertEqual(argv[idx + 1], SLOT_ACTION_DIRECTORY)
        # The slot-save path is placed before --no-cache-prompt.
        self.assertGreater(argv.index("--no-cache-prompt"), idx)
        self.assertEqual(argv[-1], "--no-cache-prompt")
        self.assertEqual(identity["slot_save_path"], SLOT_ACTION_DIRECTORY)
        self.assertEqual(
            identity["slot_save_path_policy"], "erase_only_feature_gate_exception"
        )
        self.assertIs(identity["native_persistence_forbidden"], True)
        self.assertEqual(identity["slot_action_directory_mode"], "0555")
        self.assertIs(identity["slot_action_directory_empty_required"], True)
        self.assertIs(identity["cache_canary_implied_passed"], False)

    def test_slot_action_directory_path_derived_from_screen_id(self) -> None:
        self.assertEqual(slot_action_directory_path(), SLOT_ACTION_DIRECTORY)
        self.assertEqual(
            SLOT_ACTION_DIRECTORY,
            "/tmp/m3-prompt-only-pilot-20260816-v11-erase-only-slot-actions",
        )

    def test_required_help_flags_include_slot_save_path(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _required_off_server_help_flags,
        )

        self.assertIn("--slot-save-path", _required_off_server_help_flags())

    def test_server_receipt_argv_matches_bound_identity(self) -> None:
        self.assertEqual(
            _valid_server_receipt()["server_argv"],
            _MANIFEST["isolated_no_cache_server_identity"]["server_argv"],
        )


class SlotActionDirectoryTest(unittest.TestCase):
    def _fake(
        self,
        *,
        preexisting: bool = False,
        file_type: str = "directory",
        mode: str = "555",
        nonempty: bool = False,
        commands: list | None = None,
        remove_succeeds: bool = True,
    ):
        state = {"exists": preexisting}

        def fake(command):
            if commands is not None:
                commands.append(command)
            if command[0] == "id" and command[1] == "-u":
                return "1000\n"
            if command[0] == "id" and command[1] == "-g":
                return "1000\n"
            if command[0] == "stat":
                return f"{file_type}|{mode}|1000|1000\n"
            if command[0] == "find":
                return "/tmp/x/leftover" if nonempty else ""
            if command[0] == "mkdir":
                state["exists"] = True
                return ""
            if command[0] == "chmod":
                return ""
            if command[0] == "rmdir":
                state["exists"] = False
                if not remove_succeeds:
                    raise RuntimeError("rmdir failed")
                return ""
            if command[0] == "test" and command[1] == "!":
                if state["exists"]:
                    raise RuntimeError("path exists")
                return ""
            raise AssertionError(f"unexpected command: {command}")

        return fake

    def test_prepare_requires_absence_creates_and_verifies(self) -> None:
        commands = []
        receipt = prepare_slot_action_directory(self._fake(commands=commands))
        self.assertEqual(receipt["path"], SLOT_ACTION_DIRECTORY)
        self.assertIs(receipt["empty"], True)
        self.assertIs(receipt["erase_only_feature_gate_exception"], True)
        self.assertIs(receipt["native_persistence_forbidden"], True)
        ops = [c[0] for c in commands]
        self.assertEqual(ops[0], "test")  # require absence first
        for required in ("mkdir", "chmod", "id", "stat", "find"):
            self.assertIn(required, ops)

    def test_prepare_preexisting_path_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "path exists"):
            prepare_slot_action_directory(self._fake(preexisting=True))

    def test_prepare_symlink_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not a directory"):
            prepare_slot_action_directory(self._fake(file_type="symbolic link"))

    def test_prepare_writable_mode_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "mode drifted"):
            prepare_slot_action_directory(self._fake(mode="755"))

    def test_prepare_nonempty_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not empty"):
            prepare_slot_action_directory(self._fake(nonempty=True))

    def test_observe_requires_0555_empty_directory(self) -> None:
        receipt = observe_slot_action_directory(self._fake())
        self.assertEqual(receipt["mode"], "555")
        self.assertIs(receipt["empty"], True)

    def test_remove_uses_rmdir_only_and_verifies_absence(self) -> None:
        commands = []
        receipt = remove_slot_action_directory(self._fake(commands=commands))
        self.assertIs(receipt["removed"], True)
        self.assertIs(receipt["absence_verified"], True)
        self.assertEqual(receipt["removed_via"], "rmdir")
        ops = [c[0] for c in commands]
        self.assertIn("rmdir", ops)
        self.assertNotIn("rm", ops)  # never a recursive delete

    def test_remove_failure_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "rmdir failed"):
            remove_slot_action_directory(self._fake(remove_succeeds=False))


class EndpointProbeAuthorizationTest(unittest.TestCase):
    def _probe_auth(self, request=None):
        request = request or _make_probe_request()
        return _make_probe_authorization(request)

    def test_probe_request_is_nonauthorizing(self) -> None:
        request = _make_probe_request()
        self.assertIs(request["live_model_execution_authorized"], False)
        self.assertEqual(request["authorization_scope"], "endpoint_probe")
        self.assertIsNone(request["endpoint_probe_receipt_hash"])
        self.assertEqual(request["max_cells"], 0)
        self.assertEqual(request["max_panels"], 0)
        self.assertEqual(
            request["budget"]["externally_admitted_task_completion_chat_requests"], 0
        )
        self.assertEqual(request["budget"]["server_launches"], 1)
        self.assertEqual(request["budget"]["slot_erase_sequences"], 1)

    def test_probe_request_cannot_be_used_as_authorization(self) -> None:
        request = _make_probe_request()
        with self.assertRaises(ValueError):
            validate_endpoint_probe_authorization(
                request,
                expected_authorization_hash=request["request_hash"],
                manifest_hash=_MANIFEST["manifest_hash"],
                registry_hash=_REGISTRY.registry_hash,
                local_preflight_hash=_LOCAL["preflight_hash"],
                remote_preflight_hash=_REMOTE["preflight_hash"],
                simulator_report_hash=_SIMULATOR_REPORT_HASH,
                source_tree_hash=_SOURCE,
                source_bundle_hash=_LOCAL["source_bundle_hash"],
                remote_identity=_MANIFEST["remote_identity"],
                result_filename=Path(request["result_path"]).name,
                result_path=request["result_path"],
            )

    def test_probe_authorization_validates(self) -> None:
        request = _make_probe_request()
        auth, auth_hash = self._probe_auth(request)
        self.assertEqual(
            validate_endpoint_probe_authorization(
                auth,
                expected_authorization_hash=auth_hash,
                manifest_hash=_MANIFEST["manifest_hash"],
                registry_hash=_REGISTRY.registry_hash,
                local_preflight_hash=_LOCAL["preflight_hash"],
                remote_preflight_hash=_REMOTE["preflight_hash"],
                simulator_report_hash=_SIMULATOR_REPORT_HASH,
                source_tree_hash=_SOURCE,
                source_bundle_hash=_LOCAL["source_bundle_hash"],
                remote_identity=_MANIFEST["remote_identity"],
                result_filename=Path(request["result_path"]).name,
                result_path=request["result_path"],
            ),
            auth_hash,
        )

    def test_pilot_auth_rejected_by_probe_validator(self) -> None:
        request = _make_request()
        authorization, authorization_hash = _make_authorization(request)
        with self.assertRaisesRegex(ValueError, "scope"):
            validate_endpoint_probe_authorization(
                authorization,
                expected_authorization_hash=authorization_hash,
                manifest_hash=_MANIFEST["manifest_hash"],
                registry_hash=_REGISTRY.registry_hash,
                local_preflight_hash=_LOCAL["preflight_hash"],
                remote_preflight_hash=_REMOTE["preflight_hash"],
                simulator_report_hash=_SIMULATOR_REPORT_HASH,
                source_tree_hash=_SOURCE,
                source_bundle_hash=_LOCAL["source_bundle_hash"],
                remote_identity=_MANIFEST["remote_identity"],
                result_filename=RESULT_FILENAME,
                result_path=request["result_path"],
            )

    def test_probe_auth_rejected_by_pilot_validator(self) -> None:
        request = _make_probe_request()
        auth, auth_hash = self._probe_auth(request)
        with self.assertRaisesRegex(ValueError, "scope"):
            validate_execution_authorization(
                auth,
                expected_authorization_hash=auth_hash,
                manifest_hash=_MANIFEST["manifest_hash"],
                registry_hash=_REGISTRY.registry_hash,
                local_preflight_hash=_LOCAL["preflight_hash"],
                remote_preflight_hash=_REMOTE["preflight_hash"],
                simulator_report_hash=_SIMULATOR_REPORT_HASH,
                source_tree_hash=_SOURCE,
                source_bundle_hash=_LOCAL["source_bundle_hash"],
                remote_identity=_MANIFEST["remote_identity"],
                result_filename=Path(request["result_path"]).name,
                result_path=request["result_path"],
            )

    def test_probe_auth_requires_null_receipt_hash(self) -> None:
        request = _make_probe_request()
        auth, auth_hash = self._probe_auth(request)
        auth = {**auth, "endpoint_probe_receipt_hash": "a" * 64}
        auth = _finalize(auth, "authorization_hash")
        with self.assertRaisesRegex(ValueError, "null"):
            validate_endpoint_probe_authorization(
                auth,
                expected_authorization_hash=auth["authorization_hash"],
                manifest_hash=_MANIFEST["manifest_hash"],
                registry_hash=_REGISTRY.registry_hash,
                local_preflight_hash=_LOCAL["preflight_hash"],
                remote_preflight_hash=_REMOTE["preflight_hash"],
                simulator_report_hash=_SIMULATOR_REPORT_HASH,
                source_tree_hash=_SOURCE,
                source_bundle_hash=_LOCAL["source_bundle_hash"],
                remote_identity=_MANIFEST["remote_identity"],
                result_filename=Path(request["result_path"]).name,
                result_path=request["result_path"],
            )

    def test_probe_auth_requires_probe_budget(self) -> None:
        request = _make_probe_request()
        auth, auth_hash = self._probe_auth(request)
        auth = {**auth, "budget": _worst_case_budget()}
        auth = _finalize(auth, "authorization_hash")
        with self.assertRaisesRegex(ValueError, "budget"):
            validate_endpoint_probe_authorization(
                auth,
                expected_authorization_hash=auth["authorization_hash"],
                manifest_hash=_MANIFEST["manifest_hash"],
                registry_hash=_REGISTRY.registry_hash,
                local_preflight_hash=_LOCAL["preflight_hash"],
                remote_preflight_hash=_REMOTE["preflight_hash"],
                simulator_report_hash=_SIMULATOR_REPORT_HASH,
                source_tree_hash=_SOURCE,
                source_bundle_hash=_LOCAL["source_bundle_hash"],
                remote_identity=_MANIFEST["remote_identity"],
                result_filename=Path(request["result_path"]).name,
                result_path=request["result_path"],
            )

    def test_probe_auth_requires_probe_statement(self) -> None:
        request = _make_probe_request()
        auth, auth_hash = self._probe_auth(request)
        auth = {**auth, "authorization_statement": AUTHORIZATION_STATEMENT}
        auth = _finalize(auth, "authorization_hash")
        with self.assertRaisesRegex(ValueError, "statement"):
            validate_endpoint_probe_authorization(
                auth,
                expected_authorization_hash=auth["authorization_hash"],
                manifest_hash=_MANIFEST["manifest_hash"],
                registry_hash=_REGISTRY.registry_hash,
                local_preflight_hash=_LOCAL["preflight_hash"],
                remote_preflight_hash=_REMOTE["preflight_hash"],
                simulator_report_hash=_SIMULATOR_REPORT_HASH,
                source_tree_hash=_SOURCE,
                source_bundle_hash=_LOCAL["source_bundle_hash"],
                remote_identity=_MANIFEST["remote_identity"],
                result_filename=Path(request["result_path"]).name,
                result_path=request["result_path"],
            )


class EndpointProbeReceiptValidationTest(unittest.TestCase):
    def _validate(self, receipt):
        return validate_endpoint_probe_receipt(
            receipt,
            _MANIFEST,
            _REGISTRY,
            _LOCAL,
            _REMOTE,
            source_tree_hash_value=_SOURCE,
            source_bundle_hash_value=_LOCAL["source_bundle_hash"],
            expected_result_path=_PROBE_RECEIPT_PATH,
        )

    def test_valid_probe_receipt_roundtrips(self) -> None:
        self.assertEqual(self._validate(_PROBE_RECEIPT), _PROBE_RECEIPT["receipt_hash"])

    def test_probe_receipt_retry_roundtrip_validates(self) -> None:
        """A readiness retry (failed round then success) validates end-to-end."""
        retry_trace = [
            {"method": "GET", "path": "/slots", "query": "", "status": 503, "error": None},
            {"method": "GET", "path": "/v1/models", "query": "", "status": 503, "error": None},
            {"method": "GET", "path": "/slots", "query": "", "status": 200, "error": None},
            {"method": "GET", "path": "/v1/models", "query": "", "status": 200, "error": None},
            {"method": "GET", "path": "/slots", "query": "", "status": 200, "error": None},
            {"method": "POST", "path": "/slots/0", "query": "action=erase", "status": 200, "error": None},
            {"method": "GET", "path": "/slots", "query": "", "status": 200, "error": None},
        ]
        receipt = {
            **_PROBE_RECEIPT,
            "endpoint_trace": retry_trace,
            "evidence": {
                **_PROBE_RECEIPT["evidence"],
                "readiness_receipt": _valid_readiness_receipt(attempts=2),
            },
        }
        receipt = _finalize(receipt, "receipt_hash")
        self.assertEqual(self._validate(receipt), receipt["receipt_hash"])

    def test_probe_receipt_transport_failed_readiness_round_validates(self) -> None:
        """A transport-failed round (status None + error) before success is valid."""
        trace = [
            {
                "method": "GET",
                "path": "/slots",
                "query": "",
                "status": None,
                "error": "ConnectionRefusedError: server not up",
            },
            {"method": "GET", "path": "/slots", "query": "", "status": 200, "error": None},
            {"method": "GET", "path": "/v1/models", "query": "", "status": 200, "error": None},
            {"method": "GET", "path": "/slots", "query": "", "status": 200, "error": None},
            {"method": "POST", "path": "/slots/0", "query": "action=erase", "status": 200, "error": None},
            {"method": "GET", "path": "/slots", "query": "", "status": 200, "error": None},
        ]
        receipt = {
            **_PROBE_RECEIPT,
            "endpoint_trace": trace,
            "evidence": {
                **_PROBE_RECEIPT["evidence"],
                "readiness_receipt": _valid_readiness_receipt(attempts=2),
            },
        }
        receipt = _finalize(receipt, "receipt_hash")
        self.assertEqual(self._validate(receipt), receipt["receipt_hash"])

    def test_probe_receipt_readiness_attempts_mismatch_fails(self) -> None:
        retry_trace = [
            {"method": "GET", "path": "/slots", "query": "", "status": 503, "error": None},
            {"method": "GET", "path": "/v1/models", "query": "", "status": 503, "error": None},
            {"method": "GET", "path": "/slots", "query": "", "status": 200, "error": None},
            {"method": "GET", "path": "/v1/models", "query": "", "status": 200, "error": None},
            {"method": "GET", "path": "/slots", "query": "", "status": 200, "error": None},
            {"method": "POST", "path": "/slots/0", "query": "action=erase", "status": 200, "error": None},
            {"method": "GET", "path": "/slots", "query": "", "status": 200, "error": None},
        ]
        receipt = {
            **_PROBE_RECEIPT,
            "endpoint_trace": retry_trace,
            "evidence": {
                **_PROBE_RECEIPT["evidence"],
                "readiness_receipt": _valid_readiness_receipt(attempts=1),
            },
        }
        receipt = _finalize(receipt, "receipt_hash")
        with self.assertRaisesRegex(ValueError, "attempts"):
            self._validate(receipt)

    def test_probe_receipt_tamper_rejected(self) -> None:
        tampered = {**_PROBE_RECEIPT, "passed": False}
        tampered = _finalize(tampered, "receipt_hash")
        with self.assertRaisesRegex(ValueError, "did not pass"):
            self._validate(tampered)

    def test_probe_receipt_completion_endpoint_rejected(self) -> None:
        tampered = {
            **_PROBE_RECEIPT,
            "endpoint_trace": [
                {"method": "GET", "path": "/slots", "query": "", "status": 200, "error": None},
                {"method": "GET", "path": "/v1/models", "query": "", "status": 200, "error": None},
                {"method": "GET", "path": "/slots", "query": "", "status": 200, "error": None},
                {"method": "POST", "path": "/v1/completions", "query": "", "status": 200, "error": None},
                {"method": "GET", "path": "/slots", "query": "", "status": 200, "error": None},
            ],
        }
        tampered = _finalize(tampered, "receipt_hash")
        with self.assertRaisesRegex(ValueError, "POST path/query"):
            self._validate(tampered)

    def test_probe_receipt_manifest_mismatch_rejected(self) -> None:
        tampered = {**_PROBE_RECEIPT, "manifest_hash": "0" * 64}
        tampered = _finalize(tampered, "receipt_hash")
        with self.assertRaisesRegex(ValueError, "manifest hash"):
            self._validate(tampered)

    def test_full_request_requires_passing_probe_receipt(self) -> None:
        failed = {**_PROBE_RECEIPT, "passed": False}
        failed = _finalize(failed, "receipt_hash")
        failed_path = (
            Path(_PROBE_RECEIPT_DIR).expanduser().resolve()
            / "failed-probe-receipt.json"
        )
        failed["result_filename"] = failed_path.name
        failed["result_path"] = str(failed_path)
        failed = _finalize(failed, "receipt_hash")
        failed_path.write_text(json.dumps(failed, sort_keys=True), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "did not pass"):
            build_authorization_request(
                _MANIFEST,
                _REGISTRY,
                _LOCAL,
                _REMOTE,
                project_root=PROJECT_ROOT,
                result_path=PROJECT_ROOT / ".runs" / RESULT_FILENAME,
                endpoint_probe_receipt_path=failed_path,
                endpoint_probe_authorization_path=_PROBE_AUTHORIZATION_PATH,
                expected_endpoint_probe_authorization_hash=_PROBE_AUTHORIZATION_HASH,
            )

    def test_full_request_binds_exact_probe_hash(self) -> None:
        request = _make_request()
        self.assertEqual(
            request["endpoint_probe_receipt_hash"], _PROBE_RECEIPT["receipt_hash"]
        )
        self.assertEqual(
            request["endpoint_probe_authorization_hash"], _PROBE_AUTHORIZATION_HASH
        )
        self.assertEqual(request["authorization_scope"], "pilot")

    def test_runner_requires_authorization_binds_exact_probe_hash(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _validate_bound_probe_receipt,
        )

        run = _build_validated_run()
        probe_hash, probe_auth_hash = _validate_bound_probe_receipt(
            run, _PROBE_RECEIPT_PATH, _PROBE_AUTHORIZATION_PATH, _PROBE_AUTHORIZATION_HASH
        )
        self.assertEqual(probe_hash, _PROBE_RECEIPT["receipt_hash"])
        self.assertEqual(probe_auth_hash, _PROBE_AUTHORIZATION_HASH)
        # A mismatched bound hash fails closed before any side effect.
        mismatched = replace(
            run,
            authorization={
                **run.authorization,
                "endpoint_probe_receipt_hash": "0" * 64,
            },
        )
        with self.assertRaisesRegex(ValueError, "does not bind"):
            _validate_bound_probe_receipt(
                mismatched,
                _PROBE_RECEIPT_PATH,
                _PROBE_AUTHORIZATION_PATH,
                _PROBE_AUTHORIZATION_HASH,
            )


class EndpointProbeRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()
        self.result_path = self.tmp / "endpoint-probe-receipt.json"
        self.registry_path = self.tmp / "registry.json"
        self.manifest_path = self.tmp / "manifest.json"
        self.local_path = self.tmp / "local.json"
        self.remote_path = self.tmp / "remote.json"
        self.auth_path = self.tmp / "probe-authorization.json"
        _REGISTRY.save(self.registry_path)
        self.manifest_path.write_text(json.dumps(_MANIFEST), encoding="utf-8")
        self.local_path.write_text(json.dumps(_LOCAL), encoding="utf-8")
        self.remote_path.write_text(json.dumps(_REMOTE), encoding="utf-8")
        self.request = _make_probe_request(result_path=self.result_path)
        self.authorization, self.authorization_hash = _make_probe_authorization(
            self.request
        )
        self.auth_path.write_text(json.dumps(self.authorization), encoding="utf-8")
        self.config = RemoteConfig(**REMOTE_IDENTITY)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _kwargs(self, **overrides) -> dict:
        kwargs = dict(
            manifest_path=self.manifest_path,
            registry_path=self.registry_path,
            local_preflight_path=self.local_path,
            remote_preflight_path=self.remote_path,
            authorization_path=self.auth_path,
            expected_authorization_hash=self.authorization_hash,
            result_path=self.result_path,
            config=self.config,
            pi_binary="pi",
            provider=RUN_PROVIDER,
            model=RUN_MODEL_ALIAS,
            thinking="off",
            unbrowser_binary=_MANIFEST["runtime_pins"]["unbrowser_path"],
            model_artifact=_MANIFEST["runtime_pins"]["model_artifact_path"],
            llama_server_binary=_MANIFEST["runtime_pins"]["llama_server_path"],
        )
        kwargs.update(overrides)
        return kwargs

    def _invoke(self):
        from pyreplab_harness.m3_prompt_only_execution import _HttpResponse

        def slot_dir_ssh(host, command):
            if command[0] == "id" and command[1] == "-u":
                return "1000\n"
            if command[0] == "id" and command[1] == "-g":
                return "1000\n"
            if command[0] == "stat":
                return "directory|555|1000|1000\n"
            if command[0] == "find":
                return ""
            raise AssertionError(f"unexpected command: {command}")

        def fake_http_get(url):
            if url.endswith("/slots"):
                return _HttpResponse(200, [{"id": 0, "is_processing": False}])
            return _HttpResponse(200, {"data": [{"id": RUN_MODEL_ALIAS}]})

        def fake_http_post(url):
            return _HttpResponse(200, {"id_slot": 0, "n_erased": 0})

        def fake_lifecycle_start(run, lifecycle):
            lifecycle["server"] = _valid_server_receipt()
            lifecycle["tunnel"] = _valid_tunnel_receipt()
            return lifecycle

        def fake_teardown(run, lifecycle):
            active_after = _valid_active_service_receipt()
            lifecycle["active_service_after"] = active_after
            lifecycle["slot_action_dir_removal_receipt"] = _valid_slot_dir_removal_receipt()
            lifecycle["teardown_receipt"] = _valid_teardown_receipt(active_after)
            return lifecycle["teardown_receipt"]

        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._revalidate_runtime_identity",
            return_value=None,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._acquire_generation_lease",

            side_effect=lambda run: _lease_receipts_for(run.authorization_hash)[:2],

            ), mock.patch(

            "pyreplab_harness.m3_prompt_only_execution._release_generation_lease",

            side_effect=lambda run: _release_outcome_for(run.authorization_hash),
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._prepare_slot_action_directory",
            return_value=_valid_slot_dir_preparation_receipt(),
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._start_live_lifecycle",
            side_effect=fake_lifecycle_start,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._ensure_verified_teardown",
            side_effect=fake_teardown,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._require_final_quiescence",
            return_value=None,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._require_remote_bundle_intact",
            return_value=None,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._real_http_get",
            side_effect=fake_http_get,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._real_http_post",
            side_effect=fake_http_post,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._ssh_capture",
            side_effect=slot_dir_ssh,
        ):
            return run_authorized_endpoint_probe(**self._kwargs())

    def test_probe_runner_writes_passing_receipt_no_completion_endpoint(self) -> None:
        report = self._invoke()
        self.assertIs(report["server_startup_warmup_permitted"], True)
        self.assertIs(report["task_inference_invoked"], False)
        self.assertTrue(self.result_path.is_file())
        receipt = json.loads(self.result_path.read_text(encoding="utf-8"))
        self.assertIs(receipt["passed"], True)
        self.assertIs(receipt["server_startup_warmup_permitted"], True)
        self.assertIs(receipt["task_inference_invoked"], False)
        self.assertEqual(receipt["result_path"], str(self.result_path))
        self.assertEqual(receipt["result_filename"], self.result_path.name)
        self.assertEqual(
            validate_endpoint_probe_receipt(
                receipt,
                _MANIFEST,
                _REGISTRY,
                _LOCAL,
                _REMOTE,
                source_tree_hash_value=_SOURCE,
                source_bundle_hash_value=_LOCAL["source_bundle_hash"],
                expected_result_path=self.result_path,
            ),
            receipt["receipt_hash"],
        )
        paths = " ".join(entry["path"].casefold() for entry in receipt["endpoint_trace"])
        self.assertNotIn("completion", paths)
        self.assertNotIn("chat", paths)
        # The trace records exactly the allowed endpoint set.
        self.assertEqual(
            {entry["path"] for entry in receipt["endpoint_trace"]},
            {"/slots", "/v1/models", "/slots/0"},
        )
        # The consumed marker proves single-use consumption before launch, and
        # the claim/consumed evidence is embedded and bound to the receipt.
        self.assertTrue(
            (self.tmp / "endpoint-probe-receipt.json.consumed.json").is_file()
        )
        self.assertTrue((self.tmp / "endpoint-probe-receipt.json.claim.json").is_file())
        claim = receipt["evidence"]["claim"]
        consumed_marker = receipt["evidence"]["consumed_marker"]
        self.assertEqual(claim["authorization_hash"], receipt["authorization_hash"])
        self.assertEqual(claim["result_path"], receipt["result_path"])
        self.assertEqual(
            consumed_marker["authorization_hash"], receipt["authorization_hash"]
        )

    def test_probe_runner_rejects_pilot_authorization(self) -> None:
        pilot_request = _make_request(result_path=self.result_path)
        pilot_auth, pilot_hash = _make_authorization(pilot_request)
        self.auth_path.write_text(json.dumps(pilot_auth), encoding="utf-8")
        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._start_live_lifecycle"
        ) as start:
            with self.assertRaisesRegex(ValueError, "scope"):
                run_authorized_endpoint_probe(
                    **self._kwargs(
                        expected_authorization_hash=pilot_hash,
                    )
                )
        start.assert_not_called()

    def test_probe_runner_failure_persists_bounded_failure_receipt(self) -> None:
        def fake_teardown(run, lifecycle):
            active_after = _valid_active_service_receipt()
            lifecycle["active_service_after"] = active_after
            lifecycle["teardown_receipt"] = _valid_teardown_receipt(active_after)
            return lifecycle["teardown_receipt"]

        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._revalidate_runtime_identity",
            return_value=None,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._acquire_generation_lease",
            side_effect=lambda run: _lease_receipts_for(run.authorization_hash)[:2],
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._release_generation_lease",
            side_effect=lambda run: _release_outcome_for(run.authorization_hash),
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._prepare_slot_action_directory",
            return_value=_valid_slot_dir_preparation_receipt(),
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._start_live_lifecycle",
            side_effect=RuntimeError("server launch failed"),
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._ensure_verified_teardown",
            side_effect=fake_teardown,
        ):
            with self.assertRaisesRegex(RuntimeError, "server launch failed"):
                run_authorized_endpoint_probe(**self._kwargs())
        failure_path = self.tmp / "endpoint-probe-receipt.json.probe-failure.json"
        self.assertTrue(failure_path.is_file())
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        self.assertIs(failure["server_startup_warmup_permitted"], True)
        self.assertIs(failure["task_inference_invoked"], False)
        self.assertIn("server launch failed", failure["error_message"])

    def test_probe_verified_failure_teardown_releases_and_records_true(self) -> None:
        # A probe failure with a VERIFIED finally teardown must release the
        # quarantine and the failure receipt must record the actual released
        # state (previously it recorded a stale False).
        def fake_teardown(run, lifecycle):
            active_after = _valid_active_service_receipt()
            lifecycle["active_service_after"] = active_after
            lifecycle["slot_action_dir_removal_receipt"] = (
                _valid_slot_dir_removal_receipt()
            )
            lifecycle["teardown_receipt"] = _valid_teardown_receipt(active_after)
            return lifecycle["teardown_receipt"]

        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._revalidate_runtime_identity",
            return_value=None,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._acquire_generation_lease",
            side_effect=lambda run: _lease_receipts_for(run.authorization_hash)[:2],
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._release_generation_lease",
            side_effect=lambda run: _release_outcome_for(run.authorization_hash),
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._prepare_slot_action_directory",
            return_value=_valid_slot_dir_preparation_receipt(),
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._start_live_lifecycle",
            side_effect=RuntimeError("server launch failed"),
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._ensure_verified_teardown",
            side_effect=fake_teardown,
        ):
            with self.assertRaisesRegex(RuntimeError, "server launch failed"):
                run_authorized_endpoint_probe(**self._kwargs())
        failure_path = self.tmp / "endpoint-probe-receipt.json.probe-failure.json"
        self.assertTrue(failure_path.is_file())
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        self.assertIn("server launch failed", failure["error_message"])
        self.assertIs(failure["lease_acquired"], True)
        self.assertIs(failure["teardown_verified"], True)
        self.assertIs(failure["lease_released"], True)
        self.assertIs(failure["quarantine_retained"], False)
        self.assertIsNone(failure["release_error"])
        self.assertEqual(
            failure["generation_lease_remote_release_receipt"][
                "acquire_receipt_hash"
            ],
            failure["generation_lease_remote_acquire_receipt"]["receipt_hash"],
        )
        self.assertEqual(
            failure["generation_lease_local_release_receipt"]["acquire_receipt_hash"],
            failure["generation_lease_local_acquire_receipt"]["receipt_hash"],
        )

    def test_probe_unverified_teardown_retains_lease_failure_says_held(self) -> None:
        # An unverified teardown must retain the quarantine markers: the
        # release is never attempted and the failure receipt reports held.
        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._revalidate_runtime_identity",
            return_value=None,
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._acquire_generation_lease",
            side_effect=lambda run: _lease_receipts_for(run.authorization_hash)[:2],
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._release_generation_lease",
        ) as release_mock, mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._prepare_slot_action_directory",
            return_value=_valid_slot_dir_preparation_receipt(),
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._start_live_lifecycle",
            side_effect=RuntimeError("server launch failed"),
        ), mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._ensure_verified_teardown",
            side_effect=RuntimeError("teardown could not be verified after retry"),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "teardown could not be verified after retry"
            ):
                run_authorized_endpoint_probe(**self._kwargs())
            release_mock.assert_not_called()
        failure_path = self.tmp / "endpoint-probe-receipt.json.probe-failure.json"
        self.assertTrue(failure_path.is_file())
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        # The failure receipt preserves the governing error (the original
        # launch failure), while the raised exception is the teardown gate.
        self.assertIn("server launch failed", failure["error_message"])
        self.assertIs(failure["lease_released"], False)
        self.assertIs(failure["quarantine_retained"], True)
        self.assertIs(failure["teardown_verified"], False)
        self.assertIsNone(failure["release_error"])
        self.assertIsNone(failure["generation_lease_remote_release_receipt"])
        self.assertIsNone(failure["generation_lease_local_release_receipt"])
        self.assertEqual(
            failure["generation_lease_remote_acquire_receipt"][
                "authorization_hash"
            ],
            self.authorization_hash,
        )


class GenerationLeaseTest(unittest.TestCase):
    """Local lock ownership/binding, structured release, and teardown gating."""

    def test_local_lease_roundtrip_binds_authorization(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _acquire_local_generation_lease,
            _release_local_generation_lease,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            lock_path = generation_lease_local_lock_path(root)
            acquire = _acquire_local_generation_lease(root, _LEASE_AUTH_HASH)
            self.assertTrue(lock_path.is_file())
            validate_local_generation_lease_receipt(
                acquire, released=False, authorization_hash=_LEASE_AUTH_HASH
            )
            # The lock content is bound to the exact authorization hash and the
            # receipt records its content digest.
            content = json.loads(lock_path.read_text(encoding="utf-8"))
            self.assertEqual(content["authorization_hash"], _LEASE_AUTH_HASH)
            self.assertEqual(
                acquire["lock_content_sha256"],
                hashlib.sha256(
                    json.dumps(
                        {"authorization_hash": _LEASE_AUTH_HASH},
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            )
            release = _release_local_generation_lease(
                root, authorization_hash=_LEASE_AUTH_HASH, acquire_receipt=acquire
            )
            validate_local_generation_lease_receipt(
                release,
                released=True,
                authorization_hash=_LEASE_AUTH_HASH,
                acquire_receipt_hash=acquire["receipt_hash"],
            )
            self.assertFalse(lock_path.exists())
            self.assertIs(release["released_via"], "unlink")
            self.assertIs(release["absence_verified"], True)

    def test_local_lease_wrong_authorization_blocks_unlink(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _acquire_local_generation_lease,
            _release_local_generation_lease,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            lock_path = generation_lease_local_lock_path(root)
            acquire = _acquire_local_generation_lease(root, _LEASE_AUTH_HASH)
            with self.assertRaisesRegex(RuntimeError, "another authorization"):
                _release_local_generation_lease(
                    root,
                    authorization_hash="0" * 64,
                    acquire_receipt=acquire,
                )
            # The lock is retained (quarantine held).
            self.assertTrue(lock_path.is_file())

    def test_local_lease_tampered_receipt_blocks_unlink(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _acquire_local_generation_lease,
            _release_local_generation_lease,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            lock_path = generation_lease_local_lock_path(root)
            acquire = _acquire_local_generation_lease(root, _LEASE_AUTH_HASH)
            tampered = dict(acquire)
            tampered["lock_content_sha256"] = "0" * 64
            tampered["receipt_hash"] = _canonical_hash(
                {key: value for key, value in tampered.items() if key != "receipt_hash"}
            )
            with self.assertRaisesRegex(RuntimeError, "drifted"):
                _release_local_generation_lease(
                    root,
                    authorization_hash=_LEASE_AUTH_HASH,
                    acquire_receipt=tampered,
                )
            self.assertTrue(lock_path.is_file())

    def test_remote_release_failure_leaves_local_lock(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _acquire_generation_lease,
            _release_generation_lease,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            run = _build_validated_run(root=root)
            lock_path = generation_lease_local_lock_path(root)
            fake_ssh = _FakeLeaseSSH()
            fake_ssh.fail_rmdir.add(generation_lease_remote_path())
            with mock.patch(
                "pyreplab_harness.m3_prompt_only_execution._ssh_capture",
                side_effect=lambda host, command: fake_ssh(command),
            ), mock.patch(
                "pyreplab_harness.m3_prompt_only_execution.generation_lease_local_lock_path",
                return_value=lock_path,
            ):
                _acquire_generation_lease(run)
                self.assertTrue(lock_path.is_file())
                outcome = _release_generation_lease(run)
            self.assertIn("rmdir failed", outcome["error"])
            self.assertIs(outcome["remote_released"], False)
            self.assertIs(outcome["local_released"], False)
            self.assertIs(outcome["quarantine_retained"], True)
            self.assertIsNone(outcome["remote_receipt"])
            self.assertIsNone(outcome["local_receipt"])
            # Remote-first: the local lock was NEVER touched on remote failure.
            self.assertTrue(lock_path.is_file())
            self.assertIn(generation_lease_remote_path(), fake_ssh.paths)

    def test_success_probe_receipt_carries_paired_lease_evidence(self) -> None:
        evidence = _PROBE_RECEIPT["evidence"]
        remote_acquire = evidence["generation_lease_acquire_receipt"]
        remote_release = evidence["generation_lease_release_receipt"]
        local_acquire = evidence["generation_lease_local_acquire_receipt"]
        local_release = evidence["generation_lease_local_release_receipt"]
        for receipt in (remote_acquire, remote_release, local_acquire, local_release):
            self.assertEqual(receipt["authorization_hash"], _PROBE_AUTHORIZATION_HASH)
        self.assertEqual(
            remote_release["acquire_receipt_hash"], remote_acquire["receipt_hash"]
        )
        self.assertEqual(
            local_release["acquire_receipt_hash"], local_acquire["receipt_hash"]
        )
        # The full validator accepts the paired local+remote lease evidence.
        validate_endpoint_probe_receipt(
            _PROBE_RECEIPT,
            _MANIFEST,
            _REGISTRY,
            _LOCAL,
            _REMOTE,
            source_tree_hash_value=_SOURCE,
            source_bundle_hash_value=_LOCAL["source_bundle_hash"],
            expected_result_path=_PROBE_RECEIPT_PATH,
        )

    def test_lease_audit_path_is_required_fresh(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _require_fresh_result_paths,
        )

        run = _build_validated_run()
        audit = run.paths["lease_audit"]
        audit.parent.mkdir(parents=True, exist_ok=True)
        audit.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "result paths must be fresh"):
            _require_fresh_result_paths(run)

    def _teardown_with_required_false(self, *, slot_path_present: bool):
        from pyreplab_harness.m3_prompt_only_execution import _stop_live_lifecycle

        run = _test_run_with_remote(
            _custom_remote_preflight(_service_status_sha256()), "active"
        )
        lifecycle: dict[str, Any] = {
            "config": run.config,
            "server": None,
            "tunnel": None,
            "tunnel_owned": None,
            "owned_proxies": [],
            "teardown_receipt": None,
            "active_service_after": None,
            "slot_action_dir_required": False,
            "slot_action_dir_removal_receipt": None,
        }

        def fake_ssh(command):
            if command[0] == "systemctl":
                return _service_status_text()
            if command[0] == "journalctl":
                return (
                    "" if "--after-cursor" in command else _baseline_journal_output()
                )
            if command[0] == "test" and command[1] == "!":
                if slot_path_present:
                    raise RuntimeError("slot-action directory present")
                return ""
            raise AssertionError(f"unexpected command: {command}")

        return _stop_live_lifecycle(
            run,
            lifecycle,
            ssh_spawn=fake_ssh,
            remote_listening_ports=lambda host: set(),
            deadline_seconds=0.0,
        )

    def test_teardown_not_required_verifies_slot_dir_absence(self) -> None:
        receipt = self._teardown_with_required_false(slot_path_present=False)
        self.assertIs(receipt["verified"], True)
        self.assertIs(receipt["slot_action_dir_required"], False)
        self.assertIs(receipt["slot_action_dir_absence_verified"], True)
        self.assertIs(receipt["slot_action_dir_removed"], False)

    def test_teardown_not_required_preexisting_path_unverified(self) -> None:
        receipt = self._teardown_with_required_false(slot_path_present=True)
        self.assertIs(receipt["verified"], False)
        self.assertIs(receipt["slot_action_dir_absence_verified"], False)
        self.assertTrue(any("slot_action_dir" in error for error in receipt["errors"]))


class CLIRegressionTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()
        self.registry_path = self.tmp / "registry.json"
        self.manifest_path = self.tmp / "manifest.json"
        self.local_path = self.tmp / "local.json"
        self.remote_path = self.tmp / "remote.json"
        _REGISTRY.save(self.registry_path)
        self.manifest_path.write_text(json.dumps(_MANIFEST), encoding="utf-8")
        self.local_path.write_text(json.dumps(_LOCAL), encoding="utf-8")
        self.remote_path.write_text(json.dumps(_REMOTE), encoding="utf-8")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_probe_and_authorization_request_commands(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import main

        probe_output = self.tmp / "probe-request.json"
        result = main(
            [
                "endpoint-probe-request",
                "--manifest", str(self.manifest_path),
                "--registry", str(self.registry_path),
                "--local-preflight", str(self.local_path),
                "--remote-preflight", str(self.remote_path),
                "--root", str(PROJECT_ROOT),
                "--result", str(self.tmp / "endpoint-probe-receipt.json"),
                "--output", str(probe_output),
                "--pi", "pi",
            ]
        )
        self.assertEqual(result, 0)
        probe_request = json.loads(probe_output.read_text(encoding="utf-8"))
        self.assertEqual(probe_request["authorization_scope"], "endpoint_probe")
        self.assertIs(probe_request["live_model_execution_authorized"], False)

        request_output = self.tmp / "authorization-request.json"
        result = main(
            [
                "authorization-request",
                "--manifest", str(self.manifest_path),
                "--registry", str(self.registry_path),
                "--local-preflight", str(self.local_path),
                "--remote-preflight", str(self.remote_path),
                "--root", str(PROJECT_ROOT),
                "--result", str(self.tmp / RESULT_FILENAME),
                "--output", str(request_output),
                "--endpoint-probe-receipt", str(_PROBE_RECEIPT_PATH),
                "--endpoint-probe-authorization", str(_PROBE_AUTHORIZATION_PATH),
                "--expected-endpoint-probe-authorization-hash", _PROBE_AUTHORIZATION_HASH,
                "--pi", "pi",
            ]
        )
        self.assertEqual(result, 0)
        request = json.loads(request_output.read_text(encoding="utf-8"))
        self.assertEqual(request["authorization_scope"], "pilot")
        self.assertEqual(
            request["endpoint_probe_receipt_hash"], _PROBE_RECEIPT["receipt_hash"]
        )

    def test_authorization_request_requires_probe_receipt(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import main

        with self.assertRaises(SystemExit):
            main(
                [
                    "authorization-request",
                    "--manifest", str(self.manifest_path),
                    "--registry", str(self.registry_path),
                    "--local-preflight", str(self.local_path),
                    "--remote-preflight", str(self.remote_path),
                    "--root", str(PROJECT_ROOT),
                    "--result", str(self.tmp / RESULT_FILENAME),
                    "--output", str(self.tmp / "authorization-request.json"),
                    "--pi", "pi",
                ]
            )


class RemediationSecurityTest(unittest.TestCase):
    """Adversarial coverage for the second-round remediation."""

    def test_probe_receipt_incomplete_evidence_fails(self) -> None:
        # A previously-accepted sparse receipt (hashes only) must now fail on
        # the exact evidence field set / missing nested objects.
        sparse = {**_PROBE_RECEIPT, "evidence": {"readiness_receipt_hash": "a" * 64}}
        sparse = _finalize(sparse, "receipt_hash")
        with self.assertRaisesRegex(ValueError, "evidence mismatch"):
            validate_endpoint_probe_receipt(
                sparse,
                _MANIFEST,
                _REGISTRY,
                _LOCAL,
                _REMOTE,
                source_tree_hash_value=_SOURCE,
                source_bundle_hash_value=_LOCAL["source_bundle_hash"],
                expected_result_path=_PROBE_RECEIPT_PATH,
            )

    def test_probe_receipt_nonhex_authorization_hash_fails(self) -> None:
        bad = {**_PROBE_RECEIPT, "authorization_hash": "not-hex"}
        bad = _finalize(bad, "receipt_hash")
        with self.assertRaisesRegex(ValueError, "authorization hash is invalid"):
            validate_endpoint_probe_receipt(
                bad, _MANIFEST, _REGISTRY, _LOCAL, _REMOTE,
                source_tree_hash_value=_SOURCE,
                source_bundle_hash_value=_LOCAL["source_bundle_hash"],
                expected_result_path=_PROBE_RECEIPT_PATH,
            )

    def test_endpoint_trace_wrong_order_fails(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import _validate_endpoint_trace

        wrong_order = [
            {"method": "GET", "path": "/v1/models", "query": "", "status": 200, "error": None},
            {"method": "GET", "path": "/slots", "query": "", "status": 200, "error": None},
            {"method": "GET", "path": "/slots", "query": "", "status": 200, "error": None},
            {"method": "POST", "path": "/slots/0", "query": "action=erase", "status": 200, "error": None},
            {"method": "GET", "path": "/slots", "query": "", "status": 200, "error": None},
        ]
        with self.assertRaisesRegex(ValueError, "readiness"):
            _validate_endpoint_trace(wrong_order)

    def test_endpoint_trace_wrong_post_query_fails(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import _validate_endpoint_trace

        trace = [
            {"method": "GET", "path": "/slots", "query": "", "status": 200, "error": None},
            {"method": "GET", "path": "/v1/models", "query": "", "status": 200, "error": None},
            {"method": "GET", "path": "/slots", "query": "", "status": 200, "error": None},
            {"method": "POST", "path": "/slots/0", "query": "action=save", "status": 200, "error": None},
            {"method": "GET", "path": "/slots", "query": "", "status": 200, "error": None},
        ]
        with self.assertRaisesRegex(ValueError, "POST path/query"):
            _validate_endpoint_trace(trace)

    def test_endpoint_allowlist_rejects_non_tunnel_host(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import _validate_probe_endpoint

        with self.assertRaisesRegex(RuntimeError, "host/port not allowlisted"):
            _validate_probe_endpoint("GET", "http://127.0.0.1:18082/slots")

    def test_endpoint_allowlist_rejects_nonallowlisted_path(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import _validate_probe_endpoint

        with self.assertRaisesRegex(RuntimeError, "path/query not allowlisted"):
            _validate_probe_endpoint("GET", "http://127.0.0.1:18084/v1/completions")

    def test_run_locked_rejects_probe_scope(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import _run_locked

        probe_scoped = replace(_build_validated_run(), scope="endpoint_probe")
        with self.assertRaisesRegex(RuntimeError, "pilot-scope"):
            _run_locked(probe_scoped)

    def test_generation_lease_acquire_and_release(self) -> None:
        state = {"exists": False}
        commands = []

        def fake_ssh(command):
            commands.append(command)
            if command[0] == "test" and command[1] == "!":
                if state["exists"]:
                    raise RuntimeError("lease path exists")
                return ""
            if command[0] == "mkdir":
                state["exists"] = True
                return ""
            if command[0] == "chmod":
                return ""
            if command[0] == "id" and command[1] == "-u":
                return "1000\n"
            if command[0] == "id" and command[1] == "-g":
                return "1000\n"
            if command[0] == "stat":
                return "directory|555|1000|1000\n"
            if command[0] == "find":
                return ""
            if command[0] == "rmdir":
                state["exists"] = False
                return ""
            raise AssertionError(f"unexpected command: {command}")

        acquire = acquire_generation_lease(
            fake_ssh, authorization_hash=_LEASE_AUTH_HASH
        )
        self.assertIs(acquire["empty"], True)
        self.assertEqual(acquire["authorization_hash"], _LEASE_AUTH_HASH)
        validate_generation_lease_receipt(
            acquire, released=False, authorization_hash=_LEASE_AUTH_HASH
        )
        release = release_generation_lease(
            fake_ssh,
            authorization_hash=_LEASE_AUTH_HASH,
            acquire_receipt_hash=acquire["receipt_hash"],
        )
        self.assertIs(release["released"], True)
        self.assertIs(release["absence_verified"], True)
        self.assertEqual(release["authorization_hash"], _LEASE_AUTH_HASH)
        self.assertEqual(release["acquire_receipt_hash"], acquire["receipt_hash"])
        validate_generation_lease_receipt(
            release,
            released=True,
            authorization_hash=_LEASE_AUTH_HASH,
            acquire_receipt_hash=acquire["receipt_hash"],
        )
        ops = [c[0] for c in commands]
        self.assertNotIn("rm", ops)  # rmdir only, never recursive
        self.assertIn("rmdir", ops)

    def test_generation_lease_release_wrong_authorization_rejected(self) -> None:
        state = {"exists": False}

        def fake_ssh(command):
            if command[0] == "test" and command[1] == "!":
                if state["exists"]:
                    raise RuntimeError("lease path exists")
                return ""
            if command[0] == "mkdir":
                state["exists"] = True
                return ""
            if command[0] == "chmod":
                return ""
            if command[0] == "id":
                return "1000\n"
            if command[0] == "stat":
                return "directory|555|1000|1000\n"
            if command[0] == "find":
                return ""
            if command[0] == "rmdir":
                state["exists"] = False
                return ""
            raise AssertionError(f"unexpected command: {command}")

        acquire = acquire_generation_lease(
            fake_ssh, authorization_hash=_LEASE_AUTH_HASH
        )
        # A release bound to a different authorization must fail closed and
        # leave the remote lease in place (no blind release).
        with self.assertRaisesRegex(ValueError, "authorization hash mismatch"):
            validate_generation_lease_receipt(
                acquire,
                released=False,
                authorization_hash="0" * 64,
            )
        # The production path validates before any remote side effect: the
        # release receipt must bind the exact acquire receipt hash.
        tampered_binding = _valid_generation_lease_release_receipt(
            _LEASE_AUTH_HASH,
            acquire_receipt_hash="0" * 64,
        )
        with self.assertRaisesRegex(ValueError, "acquire binding mismatch"):
            validate_generation_lease_receipt(
                tampered_binding,
                released=True,
                authorization_hash=_LEASE_AUTH_HASH,
                acquire_receipt_hash=acquire["receipt_hash"],
            )

    def test_generation_lease_preexisting_fails_closed(self) -> None:
        def fake_ssh(command):
            if command[0] == "test" and command[1] == "!":
                raise RuntimeError("lease path exists")
            raise AssertionError(f"unexpected command: {command}")

        with self.assertRaisesRegex(RuntimeError, "lease path exists"):
            acquire_generation_lease(
                fake_ssh, authorization_hash=_LEASE_AUTH_HASH
            )

    def test_slot_dir_observation_tamper_fails(self) -> None:
        receipt = _valid_slot_dir_observation_receipt()
        receipt["empty"] = False
        receipt = _finalize(receipt, "receipt_hash")
        with self.assertRaisesRegex(ValueError, "did not confirm empty"):
            validate_slot_action_dir_observation_receipt(receipt)

    def _probe_validate(self, receipt):
        return validate_endpoint_probe_receipt(
            receipt,
            _MANIFEST,
            _REGISTRY,
            _LOCAL,
            _REMOTE,
            source_tree_hash_value=_SOURCE,
            source_bundle_hash_value=_LOCAL["source_bundle_hash"],
            expected_result_path=_PROBE_RECEIPT_PATH,
        )

    def test_probe_receipt_forged_teardown_booleans_fail(self) -> None:
        teardown = _valid_teardown_receipt(verified=True, remote_process_dead=False)
        teardown = _finalize(teardown, "receipt_hash")
        receipt = {
            **_PROBE_RECEIPT,
            "evidence": {
                **_PROBE_RECEIPT["evidence"],
                "teardown_receipt": teardown,
            },
        }
        receipt = _finalize(receipt, "receipt_hash")
        with self.assertRaisesRegex(ValueError, "remote_process_dead"):
            self._probe_validate(receipt)

    def test_probe_receipt_forged_teardown_errors_fail(self) -> None:
        teardown = _valid_teardown_receipt(errors=["ssh failed"])
        teardown = _finalize(teardown, "receipt_hash")
        receipt = {
            **_PROBE_RECEIPT,
            "evidence": {
                **_PROBE_RECEIPT["evidence"],
                "teardown_receipt": teardown,
            },
        }
        receipt = _finalize(receipt, "receipt_hash")
        with self.assertRaisesRegex(ValueError, "transport errors"):
            self._probe_validate(receipt)

    def test_probe_receipt_forged_teardown_service_unchanged_fail(self) -> None:
        teardown = _valid_teardown_receipt(active_service_unchanged=False)
        teardown = _finalize(teardown, "receipt_hash")
        receipt = {
            **_PROBE_RECEIPT,
            "evidence": {
                **_PROBE_RECEIPT["evidence"],
                "teardown_receipt": teardown,
            },
        }
        receipt = _finalize(receipt, "receipt_hash")
        with self.assertRaisesRegex(ValueError, "active_service_unchanged"):
            self._probe_validate(receipt)

    def test_probe_receipt_forged_active_service_state_fails(self) -> None:
        for after, message in (
            (_valid_active_service_receipt(quiescent=False), "not quiescent"),
            (_valid_active_service_receipt(mutated=True), "was mutated"),
        ):
            after = _finalize(after, "receipt_hash")
            teardown = _valid_teardown_receipt(after)
            teardown = _finalize(teardown, "receipt_hash")
            receipt = {
                **_PROBE_RECEIPT,
                "evidence": {
                    **_PROBE_RECEIPT["evidence"],
                    "active_service_after": after,
                    "teardown_receipt": teardown,
                },
            }
            receipt = _finalize(receipt, "receipt_hash")
            with self.assertRaisesRegex(ValueError, message):
                self._probe_validate(receipt)

    def test_probe_receipt_bool_task_request_count_fails(self) -> None:
        for bad in (False, 0.0):
            receipt = {**_PROBE_RECEIPT, "task_completion_chat_requests": bad}
            receipt = _finalize(receipt, "receipt_hash")
            with self.assertRaisesRegex(ValueError, "zero task"):
                self._probe_validate(receipt)

    def test_probe_receipt_wrong_expected_result_path_fails(self) -> None:
        other = _PROBE_RECEIPT_PATH.with_name("other-probe-receipt.json")
        with self.assertRaisesRegex(ValueError, "result (path|filename) mismatch"):
            validate_endpoint_probe_receipt(
                _PROBE_RECEIPT,
                _MANIFEST,
                _REGISTRY,
                _LOCAL,
                _REMOTE,
                source_tree_hash_value=_SOURCE,
                source_bundle_hash_value=_LOCAL["source_bundle_hash"],
                expected_result_path=other,
            )

    def test_probe_receipt_embedded_authorization_path_mismatch_fails(self) -> None:
        other = _PROBE_RECEIPT_PATH.with_name("other-probe-receipt.json")
        receipt = {
            **_PROBE_RECEIPT,
            "result_filename": other.name,
            "result_path": str(other),
        }
        receipt = _finalize(receipt, "receipt_hash")
        with self.assertRaisesRegex(ValueError, "result (path|filename) mismatch"):
            validate_endpoint_probe_receipt(
                receipt,
                _MANIFEST,
                _REGISTRY,
                _LOCAL,
                _REMOTE,
                source_tree_hash_value=_SOURCE,
                source_bundle_hash_value=_LOCAL["source_bundle_hash"],
                expected_result_path=other,
            )

    def test_probe_receipt_forged_claim_consumed_fail(self) -> None:
        # A claim bound to a different authorization must fail closed.
        claim = _valid_probe_claim("f" * 64)
        receipt = {
            **_PROBE_RECEIPT,
            "evidence": {
                **_PROBE_RECEIPT["evidence"],
                "claim": claim,
                "claim_hash": claim["claim_hash"],
            },
        }
        receipt = _finalize(receipt, "receipt_hash")
        with self.assertRaisesRegex(ValueError, "claim authorization hash mismatch"):
            self._probe_validate(receipt)

    def test_endpoint_trace_incomplete_entry_fails(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import _validate_endpoint_trace

        phantom = [
            {"method": "GET", "path": "/slots", "query": "", "status": None, "error": None},
            {"method": "GET", "path": "/v1/models", "query": "", "status": 200, "error": None},
            {"method": "GET", "path": "/slots", "query": "", "status": 200, "error": None},
            {"method": "POST", "path": "/slots/0", "query": "action=erase", "status": 200, "error": None},
            {"method": "GET", "path": "/slots", "query": "", "status": 200, "error": None},
        ]
        with self.assertRaisesRegex(ValueError, "exactly one of status or error"):
            _validate_endpoint_trace(phantom)
        missing_field = [
            {"method": "GET", "path": "/slots", "status": 200},
        ]
        with self.assertRaisesRegex(ValueError, "exactly method/path/query/status/error"):
            _validate_endpoint_trace(missing_field)

    def test_endpoint_trace_overlong_fails(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _MAX_PROBE_TRACE_ENTRIES,
            _validate_endpoint_trace,
        )

        trace = [
            {"method": "GET", "path": "/slots", "query": "", "status": 503, "error": None}
            for _ in range(_MAX_PROBE_TRACE_ENTRIES + 1)
        ]
        with self.assertRaisesRegex(ValueError, "exceeds the maximum"):
            _validate_endpoint_trace(trace)

    def test_endpoint_trace_allows_long_bounded_readiness(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import _validate_endpoint_trace

        trace = []
        for _ in range(39):
            trace.extend(
                [
                    {
                        "method": "GET",
                        "path": "/slots",
                        "query": "",
                        "status": 503,
                        "error": None,
                    },
                    {
                        "method": "GET",
                        "path": "/v1/models",
                        "query": "",
                        "status": 503,
                        "error": None,
                    },
                ]
            )
        trace.extend(
            [
                {"method": "GET", "path": "/slots", "query": "", "status": 200, "error": None},
                {"method": "GET", "path": "/v1/models", "query": "", "status": 200, "error": None},
                {"method": "GET", "path": "/slots", "query": "", "status": 200, "error": None},
                {"method": "POST", "path": "/slots/0", "query": "action=erase", "status": 200, "error": None},
                {"method": "GET", "path": "/slots", "query": "", "status": 200, "error": None},
            ]
        )
        self.assertEqual(_validate_endpoint_trace(trace), 40)

    def test_endpoint_trace_cap_fails_before_io(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import (
            _MAX_PROBE_TRACE_ENTRIES,
            _probe_trace_recorder,
        )

        trace = [
            {
                "method": "GET",
                "path": "/slots",
                "query": "",
                "status": 503,
                "error": None,
            }
            for _ in range(_MAX_PROBE_TRACE_ENTRIES)
        ]
        record_get, _ = _probe_trace_recorder(trace)
        with mock.patch(
            "pyreplab_harness.m3_prompt_only_execution._real_http_get"
        ) as real_get:
            with self.assertRaisesRegex(RuntimeError, "maximum before transmission"):
                record_get("http://127.0.0.1:18084/slots")
            real_get.assert_not_called()

    def test_endpoint_allowlist_rejects_fragment(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import _validate_probe_endpoint

        with self.assertRaisesRegex(RuntimeError, "fragment"):
            _validate_probe_endpoint(
                "GET", "http://127.0.0.1:18084/slots#/v1/completions"
            )

    def test_bound_probe_receipt_wrong_expected_probe_auth_hash_fails(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import _validate_bound_probe_receipt

        run = _build_validated_run()
        with self.assertRaisesRegex(ValueError, "expected hash"):
            _validate_bound_probe_receipt(
                run, _PROBE_RECEIPT_PATH, _PROBE_AUTHORIZATION_PATH, "f" * 64
            )

    def test_validate_authorization_cli_rejects_probe_auth_hash_mismatch(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry_path = root / "registry.json"
            manifest_path = root / "manifest.json"
            local_path = root / "local.json"
            remote_path = root / "remote.json"
            authorization_path = root / "authorization.json"
            result_path = root / RESULT_FILENAME
            _REGISTRY.save(registry_path)
            for path, artifact in (
                (manifest_path, _MANIFEST),
                (local_path, _LOCAL),
                (remote_path, _REMOTE),
            ):
                path.write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")
            request = _make_request(result_path)
            authorization, authorization_hash = _make_authorization(request)
            authorization_path.write_text(
                json.dumps(authorization, sort_keys=True), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "expected hash"):
                main(
                    [
                        "validate-authorization",
                        "--authorization", str(authorization_path),
                        "--authorization-hash", authorization_hash,
                        "--scope", "pilot",
                        "--manifest", str(manifest_path),
                        "--registry", str(registry_path),
                        "--local-preflight", str(local_path),
                        "--remote-preflight", str(remote_path),
                        "--root", str(PROJECT_ROOT),
                        "--result", str(result_path),
                        "--endpoint-probe-receipt", str(_PROBE_RECEIPT_PATH),
                        "--endpoint-probe-authorization", str(_PROBE_AUTHORIZATION_PATH),
                        "--expected-endpoint-probe-authorization-hash", "f" * 64,
                        "--pi", "pi",
                    ]
                )

    def test_nullable_git_source_commit_accepted(self) -> None:
        # The pilot substrate validator accepts a null source_commit.
        (
            local_lease_acquire,
            remote_lease_acquire,
            local_lease_release,
            remote_lease_release,
        ) = _lease_receipts_for(_LEASE_AUTH_HASH)
        receipt = build_substrate_receipt(
            _MANIFEST,
            authorization_hash=_LEASE_AUTH_HASH,
            server_receipt=_valid_server_receipt(),
            tunnel_receipt=_valid_tunnel_receipt(),
            readiness_receipt=_valid_readiness_receipt(),
            slot_clear_receipts=[_valid_slot_clear_receipt() for _ in range(EXPECTED_CELLS)],
            proxy_receipts=[_valid_proxy_receipt() for _ in range(EXPECTED_CELLS)],
            active_service_before=_valid_active_service_receipt(),
            active_service_after=_valid_active_service_receipt(),
            teardown_receipt=_valid_teardown_receipt(),
            source_commit=None,
            source_bundle_hash=_LOCAL["source_bundle_hash"],
            slot_action_dir_preparation_receipt=_valid_slot_dir_preparation_receipt(),
            generation_lease_acquire_receipt=remote_lease_acquire,
            generation_lease_release_receipt=remote_lease_release,
            generation_lease_local_acquire_receipt=local_lease_acquire,
            generation_lease_local_release_receipt=local_lease_release,
        )
        self.assertIsNone(receipt["evidence"]["source_commit"])
        validate_execution_substrate_receipt(receipt, _MANIFEST, _LEASE_AUTH_HASH)


class ProbeTransportTest(unittest.TestCase):
    """Model-free local-loopback tests for the real probe HTTP transport."""

    def setUp(self) -> None:
        import http.server
        import threading

        self.requests: list[tuple[str, str]] = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:  # pragma: no cover - quiet
                pass

            def do_GET(self) -> None:
                self.server.requests.append(("GET", self.path))
                if self.path == "/ok":
                    self._reply(200, b"[]")
                elif self.path == "/redirect":
                    self.send_response(302)
                    self.send_header("Location", "/ok")
                    self.end_headers()
                else:
                    self._reply(404, b"{}")

            def do_POST(self) -> None:
                self.server.requests.append(("POST", self.path))
                if self.path == "/ok":
                    self._reply(200, b"{}")
                elif self.path == "/redirect":
                    self.send_response(302)
                    self.send_header("Location", "/ok")
                    self.end_headers()
                else:
                    self._reply(404, b"{}")

            def _reply(self, status: int, body: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        server.requests = self.requests
        self.server = server
        self.base = f"http://127.0.0.1:{server.server_address[1]}"
        self.thread = threading.Thread(target=server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def test_get_ignores_proxy_environment_variables(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import _real_http_get

        with mock.patch.dict(
            os.environ,
            {
                "http_proxy": "http://127.0.0.1:1",
                "HTTP_PROXY": "http://127.0.0.1:1",
                "https_proxy": "http://127.0.0.1:1",
                "HTTPS_PROXY": "http://127.0.0.1:1",
                "all_proxy": "http://127.0.0.1:1",
                "ALL_PROXY": "http://127.0.0.1:1",
            },
        ):
            response = _real_http_get(f"{self.base}/ok")
        self.assertEqual(response.status, 200)
        self.assertEqual(self.requests, [("GET", "/ok")])

    def test_get_redirect_rejected_before_follow(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import _real_http_get

        with self.assertRaisesRegex(RuntimeError, "redirect rejected"):
            _real_http_get(f"{self.base}/redirect")
        # The redirect target must never be requested.
        self.assertEqual(self.requests, [("GET", "/redirect")])

    def test_post_redirect_rejected_before_follow(self) -> None:
        from pyreplab_harness.m3_prompt_only_execution import _real_http_post

        with self.assertRaisesRegex(RuntimeError, "redirect rejected"):
            _real_http_post(f"{self.base}/redirect")
        self.assertEqual(self.requests, [("POST", "/redirect")])


def tearDownModule() -> None:
    import shutil

    shutil.rmtree(_TMP_ROOT, ignore_errors=True)
    shutil.rmtree(_PROBE_RECEIPT_DIR, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
