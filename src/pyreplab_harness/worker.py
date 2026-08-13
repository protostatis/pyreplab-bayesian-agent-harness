from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any, TextIO

from .fixture_server import FixtureServer
from .sandbox import BubblewrapSandbox, SandboxLimits
from .unbrowser_rpc import DEFAULT_MAX_RESULT_BYTES, FIXTURE_URL_PREFIX, UnbrowserSession

PROTOCOL_VERSION = 1
FIXTURE_PORT = 18090

_fixture_server_instance: FixtureServer | None = None


def ensure_fixture_server() -> FixtureServer:
    """Start the fixture server on the fixed port if not already running.

    Raises if the fixed port cannot be bound. A different process on the port
    must never be mistaken for the harness-owned deterministic fixture server.
    """
    global _fixture_server_instance
    if _fixture_server_instance is None:
        _fixture_server_instance = FixtureServer(port=FIXTURE_PORT)
    return _fixture_server_instance


def stop_fixture_server() -> None:
    """Shut down the fixture server if it was started by this worker."""
    global _fixture_server_instance
    if _fixture_server_instance is not None:
        _fixture_server_instance.stop()
        _fixture_server_instance = None


@dataclass(frozen=True)
class WorkerConfig:
    max_timeout: int = 30
    semantic_capability: str | None = None


def handle_request(
    sandbox: BubblewrapSandbox,
    request: dict[str, Any],
    config: WorkerConfig,
    unbrowser: UnbrowserSession | None = None,
) -> tuple[dict[str, Any], bool]:
    request_id = request.get("id")
    method = request.get("method")
    params = request.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError("params must be an object")

    if method == "ping":
        return (
            {
                "id": request_id,
                "ok": True,
                "result": {
                    "protocol_version": PROTOCOL_VERSION,
                    "workspace": "/workspace",
                    "network": "isolated",
                    "unbrowser": {
                        "enabled": unbrowser is not None,
                        "mode": "fixed-page-read-only" if unbrowser is not None else None,
                    },
                },
            },
            False,
        )
    if method == "shutdown":
        return {"id": request_id, "ok": True, "result": {"stopped": True}}, True
    if method == "unbrowser":
        if unbrowser is None:
            raise ValueError("unbrowser is not enabled for this treatment")
        result = unbrowser.execute(params)
        return {"id": request_id, "ok": True, "result": result}, False
    if method == "semantic_table":
        if unbrowser is None:
            raise ValueError("semantic_table requires an active unbrowser session")
        result = unbrowser.semantic_table(params)
        return {"id": request_id, "ok": True, "result": result}, False
    if method == "semantic_form":
        if unbrowser is None:
            raise ValueError("semantic_form requires an active unbrowser session")
        result = unbrowser.semantic_form(params)
        return {"id": request_id, "ok": True, "result": result}, False
    if method != "exec":
        raise ValueError(f"unknown method: {method!r}")

    command = params.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ValueError("exec command must be a non-empty string")
    requested_timeout = params.get("timeout", config.max_timeout)
    if isinstance(requested_timeout, bool) or not isinstance(requested_timeout, (int, float)):
        raise ValueError("timeout must be numeric")
    timeout = max(1, min(int(requested_timeout), config.max_timeout))
    result = sandbox.execute(command, timeout)
    return {"id": request_id, "ok": True, "result": result.to_dict()}, False


def serve(
    sandbox: BubblewrapSandbox,
    input_stream: TextIO,
    output_stream: TextIO,
    max_timeout: int,
    unbrowser: UnbrowserSession | None = None,
    semantic_capability: str | None = None,
) -> int:
    config = WorkerConfig(max_timeout=max_timeout, semantic_capability=semantic_capability)
    try:
        for line in input_stream:
            if not line.strip():
                continue
            request_id: Any = None
            stop = False
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("request must be an object")
                request_id = request.get("id")
                response, stop = handle_request(sandbox, request, config, unbrowser)
            except Exception as error:  # The protocol must return structured failures.
                response = {
                    "id": request_id,
                    "ok": False,
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                        "infrastructure_error": bool(
                            getattr(error, "infrastructure_error", False)
                            or isinstance(error, (BrokenPipeError, ConnectionResetError, TimeoutError))
                        ),
                    },
                }
            output_stream.write(json.dumps(response, separators=(",", ":")) + "\n")
            output_stream.flush()
            if stop:
                break
    finally:
        if unbrowser is not None:
            unbrowser.close()
    return 0


def add_worker_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--max-timeout", type=int, default=30)
    parser.add_argument("--memory-max", default="1G")
    parser.add_argument("--tasks-max", type=int, default=64)
    parser.add_argument("--cpu-quota", default="200%")
    parser.add_argument("--unbrowser-url", default=None)
    parser.add_argument("--unbrowser-binary", default="/usr/local/bin/unbrowser")
    parser.add_argument("--unbrowser-timeout", type=int, default=30)
    parser.add_argument(
        "--unbrowser-max-result-bytes", type=int, default=DEFAULT_MAX_RESULT_BYTES
    )
    parser.add_argument(
        "--unbrowser-interactive",
        action="store_true",
        default=False,
        help="enable interactive Unbrowser actions (click, type, submit)",
    )
    parser.add_argument(
        "--confine-unbrowser",
        action="store_true",
        default=False,
        help="launch unbrowser inside a Bubblewrap sandbox (filesystem isolation)",
    )
    parser.add_argument(
        "--unbrowser-required-first-observation",
        choices=["text", "blockmap"],
        default=None,
        help="auto-deliver a text or blockmap observation on the first successful "
        "direct navigate (interactive sessions only)",
    )
    parser.add_argument(
        "--semantic-capability",
        choices=["table", "form"],
        default=None,
        help="enable a controller-side semantic table or form specialist "
        "(requires interactive fixture stack)",
    )


def run_from_args(args: argparse.Namespace) -> int:
    limits = SandboxLimits(
        max_timeout_seconds=args.max_timeout,
        memory_max=args.memory_max,
        tasks_max=args.tasks_max,
        cpu_quota=args.cpu_quota,
    )
    sandbox = BubblewrapSandbox(args.root, args.workspace, limits)

    unbrowser_url: str | None = args.unbrowser_url
    is_fixture = bool(unbrowser_url and unbrowser_url.startswith(FIXTURE_URL_PREFIX))
    if is_fixture:
        ensure_fixture_server()

    # Auto-confine for fixture tasks unless explicitly disabled.
    confine_unbrowser = bool(args.confine_unbrowser) or is_fixture

    required_first_obs = getattr(args, "unbrowser_required_first_observation", None)
    if required_first_obs is not None and not unbrowser_url:
        raise ValueError(
            "--unbrowser-required-first-observation requires --unbrowser-url"
        )
    semantic_cap = getattr(args, "semantic_capability", None)
    if semantic_cap is not None:
        if not unbrowser_url:
            raise ValueError(
                "--semantic-capability requires --unbrowser-url"
            )
        if not is_fixture:
            raise ValueError(
                "--semantic-capability requires a fixture URL "
                f"(must start with {FIXTURE_URL_PREFIX})"
            )
        if not args.unbrowser_interactive:
            raise ValueError(
                "--semantic-capability requires --unbrowser-interactive"
            )
    unbrowser = None
    if unbrowser_url:
        unbrowser = UnbrowserSession(
            args.unbrowser_binary,
            unbrowser_url,
            timeout_seconds=args.unbrowser_timeout,
            max_result_bytes=args.unbrowser_max_result_bytes,
            interactive=bool(args.unbrowser_interactive or is_fixture),
            confined=confine_unbrowser,
            required_first_observation=(
                str(required_first_obs) if required_first_obs is not None else None
            ),
            semantic_capability=str(semantic_cap) if semantic_cap is not None else None,
        )
    try:
        return serve(sandbox, sys.stdin, sys.stdout, args.max_timeout, unbrowser,
                     semantic_capability=semantic_cap)
    finally:
        if is_fixture:
            stop_fixture_server()
