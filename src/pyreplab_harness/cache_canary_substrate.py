"""Freeze and preflight an isolated cache-canary substrate without model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import socket
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .cache_mechanics import (
    CACHE_RUNTIME_RECEIPT_SCHEMA_VERSION,
    canonical_receipt_hash,
    parse_cache_launch_configuration,
    validate_cache_runtime_receipt,
)
from .cache_proxy import CACHE_PROXY_RECEIPT_SCHEMA_VERSION
from .m3_pilot import _ssh_capture, _write_immutable_json, source_tree_hash

CACHE_CANARY_SUBSTRATE_SCHEMA_VERSION = "pyreplab-cache-canary-substrate-v1"
CACHE_CANARY_SUBSTRATE_PREFLIGHT_SCHEMA_VERSION = (
    "pyreplab-cache-canary-substrate-preflight-v1"
)
SCREEN_ID = "cache-mechanics-canary-substrate-20260815-v1"

_SERVER_PORT = 18082
_PROXY_PORT = 18083
_TUNNEL_PORT = 18084
_MODEL_ALIAS = "gemma-4-26b-a4b-cache-canary"
_ALLOWED_CACHE_PARSE_INVALIDATIONS = frozenset(
    {"implicit_cache_argument:slot_save_path"}
)
_REQUIRED_HELP_FLAGS = (
    "--model",
    "--alias",
    "--host",
    "--port",
    "--ctx-size",
    "--flash-attn",
    "--n-cpu-moe",
    "--n-gpu-layers",
    "--parallel",
    "--reasoning",
    "--threads",
    "--cache-type-k",
    "--cache-type-v",
    "--cache-ram",
    "--ctx-checkpoints",
    "--checkpoint-min-step",
    "--cache-idle-slots",
    "--cache-reuse",
    "--kv-unified",
    "--metrics",
    "--slots",
    "--sleep-idle-seconds",
    "--perf",
    "--no-context-shift",
    "--no-cont-batching",
    "--warmup",
    "--no-webui",
    "--timeout",
    "--sse-ping-interval",
    "--cache-prompt",
    "--no-cache-prompt",
)


def _common_server_argv(server_binary: str, model_artifact: str) -> list[str]:
    return [
        server_binary,
        "--model",
        model_artifact,
        "--alias",
        _MODEL_ALIAS,
        "--host",
        "127.0.0.1",
        "--port",
        str(_SERVER_PORT),
        "--ctx-size",
        "65536",
        "--flash-attn",
        "on",
        "--n-cpu-moe",
        "16",
        "--n-gpu-layers",
        "all",
        "--parallel",
        "1",
        "--reasoning",
        "on",
        "--threads",
        "8",
        "--cache-type-k",
        "f16",
        "--cache-type-v",
        "f16",
        "--cache-ram",
        "8192",
        "--ctx-checkpoints",
        "32",
        "--checkpoint-min-step",
        "8192",
        "--cache-idle-slots",
        "--cache-reuse",
        "0",
        "--kv-unified",
        "--metrics",
        "--slots",
        "--sleep-idle-seconds",
        "-1",
        "--perf",
        "--no-context-shift",
        "--no-cont-batching",
        "--warmup",
        "--no-webui",
        "--timeout",
        "900",
        "--sse-ping-interval",
        "-1",
    ]


def build_cache_canary_substrate_manifest(
    *,
    project_root: str | Path,
    runtime_probe: Mapping[str, Any],
    server_binary: str,
    model_artifact: str,
) -> dict[str, Any]:
    """Build a non-authorizing isolated substrate manifest."""
    validate_cache_runtime_receipt(runtime_probe)
    runtime = runtime_probe["runtime"]
    common_argv = _common_server_argv(server_binary, model_artifact)
    cells = []
    for mode, flag in (("off", "--no-cache-prompt"), ("on", "--cache-prompt")):
        argv = [*common_argv, flag]
        cells.append(
            {
                "cache_mode": mode,
                "server_argv": argv,
                "server_argv_hash": canonical_receipt_hash(argv),
                "proxy_injected_cache_prompt": mode == "on",
                "slot_id": 0,
            }
        )
    common_config = {
        "server_argv_without_cache_mode": common_argv,
        "slot_save_path_policy": "disabled_by_omission_and_forbidden",
        "raw_kv_persistence": False,
    }
    payload: dict[str, Any] = {
        "schema_version": CACHE_CANARY_SUBSTRATE_SCHEMA_VERSION,
        "screen_id": SCREEN_ID,
        "purpose": (
            "Isolated mechanics substrate for a future authorized cache "
            "invariance canary; this artifact does not authorize execution."
        ),
        "source_tree_hash": source_tree_hash(Path(project_root).expanduser().resolve()),
        "identity_source_runtime_receipt_hash": runtime_probe["receipt_hash"],
        "runtime_identity": {
            "pi_version": runtime["pi_version"],
            "pi_sha256": runtime["pi_sha256"],
            "llama_server_version": runtime["llama_server_version"],
            "llama_server_sha256": runtime["llama_server_sha256"],
            "model_sha256": runtime["model_sha256"],
            "server_binary": server_binary,
            "model_artifact": model_artifact,
            "model_alias": _MODEL_ALIAS,
        },
        "network": {
            "remote_server": {
                "host": "127.0.0.1",
                "port": _SERVER_PORT,
            },
            "local_upstream_tunnel": {
                "host": "127.0.0.1",
                "port": _TUNNEL_PORT,
                "remote_target": f"127.0.0.1:{_SERVER_PORT}",
            },
            "local_instrumentation_proxy": {
                "host": "127.0.0.1",
                "port": _PROXY_PORT,
                "upstream": f"127.0.0.1:{_TUNNEL_PORT}",
            },
            "pi_provider_base_url": f"http://127.0.0.1:{_PROXY_PORT}/v1",
        },
        "common_configuration": common_config,
        "common_config_hash": canonical_receipt_hash(common_config),
        "cells": cells,
        "only_allowed_cell_delta": "cache_prompt",
        "execution_protocol": {
            "cell_execution": "sequential",
            "attempt_execution": "sequential_single_slot",
            "preserve_manifest_order": True,
            "clear_slot_before_each_attempt": True,
            "slot_clear_receipt_required": True,
            "preserve_cache_within_attempt": True,
            "selective_retry_forbidden": True,
            "active_service_mutation_forbidden": True,
            "active_service_name": "gemma.service",
        },
        "telemetry_contract": {
            "proxy_receipt_schema": CACHE_PROXY_RECEIPT_SCHEMA_VERSION,
            "runtime_receipt_schema": CACHE_RUNTIME_RECEIPT_SCHEMA_VERSION,
            "incoming_request_hash_required": True,
            "logical_request_hash_required": True,
            "server_timing_fields": [
                "cache_n",
                "prompt_n",
                "prompt_ms",
                "predicted_n",
                "predicted_ms",
            ],
            "accounting_semantics": "llama-cpp-openai-usage-vs-timings-v1",
            "required_accounting_equalities": [
                "provider.cache_read == timings.cache_n",
                "provider.input == timings.prompt_n",
                "provider.output == timings.predicted_n",
                "provider.logical_prompt_tokens == timings.prompt_n + timings.cache_n",
            ],
            "accounting_mismatch_policy": "mechanics_invalid_no_go",
            "metrics_endpoint_required": True,
            "slots_endpoint_required": True,
            "raw_request_persistence": False,
            "raw_response_persistence": False,
            "authorization_header_persistence": False,
        },
        "live_model_execution_authorized": False,
        "authorization_boundary": (
            "No server launch, model request, or cache canary is authorized by "
            "this manifest or its no-model preflight."
        ),
        "integrity_trust_boundary": (
            "Artifact hashes detect accidental corruption under trusted local "
            "storage; they are not signatures and do not authenticate a writer."
        ),
    }
    return {**payload, "manifest_hash": canonical_receipt_hash(payload)}


def validate_cache_canary_substrate_manifest(manifest: Mapping[str, Any]) -> None:
    observed = manifest.get("manifest_hash")
    payload = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if observed != canonical_receipt_hash(payload):
        raise ValueError("cache substrate manifest hash mismatch")
    if manifest.get("schema_version") != CACHE_CANARY_SUBSTRATE_SCHEMA_VERSION:
        raise ValueError("unsupported cache substrate manifest schema")
    if manifest.get("screen_id") != SCREEN_ID:
        raise ValueError("cache substrate screen mismatch")
    if manifest.get("live_model_execution_authorized") is not False:
        raise ValueError("cache substrate manifest must remain non-authorizing")
    common = manifest.get("common_configuration")
    if not isinstance(common, Mapping):
        raise ValueError("cache substrate common configuration is missing")
    if manifest.get("common_config_hash") != canonical_receipt_hash(common):
        raise ValueError("cache substrate common configuration hash mismatch")
    if common.get("slot_save_path_policy") != "disabled_by_omission_and_forbidden":
        raise ValueError("native slot persistence must be disabled")
    common_argv = common.get("server_argv_without_cache_mode")
    if not isinstance(common_argv, list) or any(
        not isinstance(value, str) for value in common_argv
    ):
        raise ValueError("cache substrate common argv is invalid")
    if "--slot-save-path" in common_argv:
        raise ValueError("cache substrate must not configure slot persistence")
    if "--cache-prompt" in common_argv or "--no-cache-prompt" in common_argv:
        raise ValueError("cache mode leaked into the common argv")
    cells = manifest.get("cells")
    if not isinstance(cells, list) or len(cells) != 2:
        raise ValueError("cache substrate requires exactly two cells")
    expected = (("off", "--no-cache-prompt"), ("on", "--cache-prompt"))
    for cell, (mode, flag) in zip(cells, expected, strict=True):
        if not isinstance(cell, Mapping) or cell.get("cache_mode") != mode:
            raise ValueError("cache substrate cell order or mode drifted")
        argv = cell.get("server_argv")
        if argv != [*common_argv, flag]:
            raise ValueError("cache substrate cells differ beyond cache mode")
        if cell.get("server_argv_hash") != canonical_receipt_hash(argv):
            raise ValueError("cache substrate cell argv hash mismatch")
        if cell.get("proxy_injected_cache_prompt") is not (mode == "on"):
            raise ValueError("proxy and server cache modes differ")
        if cell.get("slot_id") != 0:
            raise ValueError("cache substrate must use slot zero")
    protocol = manifest.get("execution_protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("cache substrate execution protocol is missing")
    for field in (
        "preserve_manifest_order",
        "clear_slot_before_each_attempt",
        "slot_clear_receipt_required",
        "preserve_cache_within_attempt",
        "selective_retry_forbidden",
        "active_service_mutation_forbidden",
    ):
        if protocol.get(field) is not True:
            raise ValueError(f"cache substrate protocol must require {field}")


def _help_mentions(help_text: str, flag: str) -> bool:
    return re.search(rf"(?<!\S){re.escape(flag)}(?=[\s,=]|$)", help_text) is not None


def _local_port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        # Mirror the real proxy bind: ThreadingHTTPServer sets SO_REUSEADDR
        # (HTTPServer.allow_reuse_address=1). A plain bind() spuriously fails
        # while TIME_WAIT sockets from the previous cell's proxy connections
        # linger on the fixed proxy port (~31s on macOS), which is exactly the
        # v9 second-cell crash. SO_REUSEADDR makes the check agree with the
        # actual bind while a live LISTEN socket still blocks (fail-closed).
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _remote_listening_ports(host: str) -> set[int]:
    output = _ssh_capture(host, ["ss", "-H", "-ltn"])
    ports: set[int] = set()
    for line in output.splitlines():
        columns = line.split()
        if len(columns) < 4:
            continue
        local_address = columns[3]
        try:
            ports.add(int(local_address.rsplit(":", 1)[1]))
        except (ValueError, IndexError):
            continue
    return ports


def preflight_cache_canary_substrate(
    manifest: Mapping[str, Any],
    *,
    project_root: str | Path,
    host: str,
) -> dict[str, Any]:
    """Validate identities, flags, and unused ports without launching a model."""
    validate_cache_canary_substrate_manifest(manifest)
    root = Path(project_root).expanduser().resolve()
    source_hash = source_tree_hash(root)
    if source_hash != manifest.get("source_tree_hash"):
        raise RuntimeError("cache substrate source tree drifted")
    identity = manifest["runtime_identity"]
    server_binary = identity["server_binary"]
    server_version = _ssh_capture(
        host, [server_binary, "--version"], stderr_fallback=True
    ).splitlines()[0]
    server_hash = _ssh_capture(host, ["sha256sum", server_binary]).split()[0]
    if server_version != identity["llama_server_version"]:
        raise RuntimeError("cache substrate llama-server version drift")
    if server_hash != identity["llama_server_sha256"]:
        raise RuntimeError("cache substrate llama-server hash drift")
    _ssh_capture(host, ["test", "-r", identity["model_artifact"]])
    model_hash = _ssh_capture(
        host,
        ["sha256sum", identity["model_artifact"]],
        timeout=900,
    ).split()[0]
    if model_hash != identity["model_sha256"]:
        raise RuntimeError("cache substrate model artifact hash drift")
    help_text = _ssh_capture(
        host, [server_binary, "--help"], timeout=120, stderr_fallback=True
    )
    missing_help_flags = [
        flag for flag in _REQUIRED_HELP_FLAGS if not _help_mentions(help_text, flag)
    ]
    if missing_help_flags:
        raise RuntimeError(
            f"cache substrate flags absent from pinned help: {missing_help_flags!r}"
        )
    for cell in manifest["cells"]:
        parsed = parse_cache_launch_configuration(cell["server_argv"], help_text)
        unexpected = set(parsed["invalidation_codes"]) - set(
            _ALLOWED_CACHE_PARSE_INVALIDATIONS
        )
        if unexpected:
            raise RuntimeError(
                f"cache substrate cell configuration invalid: {sorted(unexpected)!r}"
            )
    remote_ports = _remote_listening_ports(host)
    if _SERVER_PORT in remote_ports:
        raise RuntimeError("isolated remote cache-canary port is already in use")
    local_ports = {
        str(port): _local_port_available(port) for port in (_PROXY_PORT, _TUNNEL_PORT)
    }
    if not all(local_ports.values()):
        raise RuntimeError("isolated local cache-canary port is already in use")
    service_status = _ssh_capture(
        host,
        [
            "systemctl",
            "--user",
            "show",
            "gemma.service",
            "--property=ActiveState",
            "--property=FragmentPath",
            "--property=ExecStart",
            "--no-pager",
        ],
    )
    if any(
        str(port) in service_status
        for port in (_SERVER_PORT, _PROXY_PORT, _TUNNEL_PORT)
    ):
        raise RuntimeError("active service unexpectedly references canary ports")
    payload: dict[str, Any] = {
        "schema_version": CACHE_CANARY_SUBSTRATE_PREFLIGHT_SCHEMA_VERSION,
        "screen_id": SCREEN_ID,
        "manifest_hash": manifest["manifest_hash"],
        "source_tree_hash": source_hash,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "probe_mode": "no_model_identity_and_port_checks_only",
        "server_help_sha256": hashlib.sha256(help_text.encode("utf-8")).hexdigest(),
        "server_version": server_version,
        "server_sha256": server_hash,
        "model_sha256": model_hash,
        "remote_server_port_available": True,
        "local_port_availability": local_ports,
        "active_service_status_sha256": hashlib.sha256(
            service_status.encode("utf-8")
        ).hexdigest(),
        "active_service_mutated": False,
        "model_loaded_or_invoked": False,
        "live_model_execution_authorized": False,
        "substrate_ready_for_canary_manifest_construction": True,
        "ready_for_live_model_execution": False,
    }
    return {**payload, "preflight_hash": canonical_receipt_hash(payload)}


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyreplab-cache-canary-substrate")
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--root", required=True)
    freeze.add_argument("--runtime-probe", required=True)
    freeze.add_argument("--llama-server-binary", required=True)
    freeze.add_argument("--model-artifact", required=True)
    freeze.add_argument("--output", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--manifest", required=True)
    preflight.add_argument("--root", required=True)
    preflight.add_argument("--host", default="ubuntu-local")
    preflight.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "freeze":
        report = build_cache_canary_substrate_manifest(
            project_root=args.root,
            runtime_probe=_load_json(args.runtime_probe),
            server_binary=args.llama_server_binary,
            model_artifact=args.model_artifact,
        )
    else:
        report = preflight_cache_canary_substrate(
            _load_json(args.manifest), project_root=args.root, host=args.host
        )
    _write_immutable_json(Path(args.output).expanduser().resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CACHE_CANARY_SUBSTRATE_PREFLIGHT_SCHEMA_VERSION",
    "CACHE_CANARY_SUBSTRATE_SCHEMA_VERSION",
    "SCREEN_ID",
    "build_cache_canary_substrate_manifest",
    "preflight_cache_canary_substrate",
    "validate_cache_canary_substrate_manifest",
]
