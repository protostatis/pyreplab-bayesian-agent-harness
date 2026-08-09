from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any, TextIO

from .sandbox import BubblewrapSandbox, SandboxLimits

PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class WorkerConfig:
    max_timeout: int = 30


def handle_request(
    sandbox: BubblewrapSandbox,
    request: dict[str, Any],
    config: WorkerConfig,
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
                },
            },
            False,
        )
    if method == "shutdown":
        return {"id": request_id, "ok": True, "result": {"stopped": True}}, True
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


def serve(sandbox: BubblewrapSandbox, input_stream: TextIO, output_stream: TextIO, max_timeout: int) -> int:
    config = WorkerConfig(max_timeout=max_timeout)
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
            response, stop = handle_request(sandbox, request, config)
        except Exception as error:  # The protocol must return structured failures.
            response = {
                "id": request_id,
                "ok": False,
                "error": {"type": type(error).__name__, "message": str(error)},
            }
        output_stream.write(json.dumps(response, separators=(",", ":")) + "\n")
        output_stream.flush()
        if stop:
            break
    return 0


def add_worker_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--max-timeout", type=int, default=30)
    parser.add_argument("--memory-max", default="1G")
    parser.add_argument("--tasks-max", type=int, default=64)
    parser.add_argument("--cpu-quota", default="200%")


def run_from_args(args: argparse.Namespace) -> int:
    limits = SandboxLimits(
        max_timeout_seconds=args.max_timeout,
        memory_max=args.memory_max,
        tasks_max=args.tasks_max,
        cpu_quota=args.cpu_quota,
    )
    sandbox = BubblewrapSandbox(args.root, args.workspace, limits)
    return serve(sandbox, sys.stdin, sys.stdout, args.max_timeout)
