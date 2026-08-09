from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import subprocess
import sys
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import PolicySpec
from .gym_registry import FAMILIES
from .treatments import TreatmentRegistry, TreatmentSpec, to_policy_spec_kwargs


@dataclass(frozen=True)
class RemoteConfig:
    host: str
    project: str
    run_root: str
    python: str = "python3"


def validate_remote_config(config: RemoteConfig) -> None:
    """Reject missing or dangerous remote execution locations."""
    if not str(config.host).strip():
        raise ValueError("--host must not be empty")
    for label, value in (
        ("--remote-project", config.project),
        ("--remote-run-root", config.run_root),
    ):
        if not value or not str(value).startswith("/") or str(value) == "/":
            raise ValueError(
                f"{label} must be an explicit absolute remote path other than '/'"
            )


def _remote_command(config: RemoteConfig, arguments: list[str]) -> list[str]:
    command = [
        "env",
        f"PYTHONPATH={config.project}/src",
        config.python,
        "-m",
        "pyreplab_harness",
        *arguments,
    ]
    return [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        config.host,
        shlex.join(command),
    ]


def remote_json(
    config: RemoteConfig,
    arguments: list[str],
    *,
    input_text: str | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    completed = subprocess.run(
        _remote_command(config, arguments),
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"remote command failed ({completed.returncode}): "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("remote command returned no JSON")
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as error:
        raise RuntimeError(f"remote command returned invalid JSON: {lines[-1]!r}") from error


def policy_spec(project_root: Path, policy_id: str, version: str = "1") -> PolicySpec:
    if version not in {"1", "2", "3", "4"}:
        raise ValueError(f"unsupported policy version: {version!r}")

    if policy_id == "direct" and version == "1":
        return PolicySpec(
            id="direct",
            version="1",
            system_prompt=(project_root / "policies" / "direct.md").read_text(encoding="utf-8").strip(),
            allowed_tools=("bash",),
            max_output_tokens=1536,
            tool_call_limit=6,
            command_timeout_seconds=30,
            wall_time_limit_seconds=300,
        )
    if policy_id == "deliberate" and version == "1":
        return PolicySpec(
            id="deliberate",
            version="1",
            system_prompt=(project_root / "policies" / "deliberate.md").read_text(encoding="utf-8").strip(),
            allowed_tools=("bash",),
            max_output_tokens=2560,
            tool_call_limit=10,
            command_timeout_seconds=45,
            wall_time_limit_seconds=480,
        )
    if policy_id == "direct" and version == "2":
        return PolicySpec(
            id="direct",
            version="2",
            system_prompt=(project_root / "policies" / "direct-v2.md")
            .read_text(encoding="utf-8")
            .strip(),
            allowed_tools=("bash",),
            max_output_tokens=1024,
            tool_call_limit=4,
            command_timeout_seconds=30,
            wall_time_limit_seconds=240,
        )
    if policy_id == "deliberate" and version == "2":
        return PolicySpec(
            id="deliberate",
            version="2",
            system_prompt=(project_root / "policies" / "deliberate-v2.md")
            .read_text(encoding="utf-8")
            .strip(),
            allowed_tools=("bash",),
            max_output_tokens=1536,
            tool_call_limit=6,
            command_timeout_seconds=45,
            wall_time_limit_seconds=360,
        )
    if policy_id == "direct" and version == "3":
        return PolicySpec(
            id="direct",
            version="3",
            system_prompt=(project_root / "policies" / "direct-v3.md")
            .read_text(encoding="utf-8")
            .strip(),
            allowed_tools=("bash",),
            max_output_tokens=1536,
            tool_call_limit=7,
            command_timeout_seconds=30,
            wall_time_limit_seconds=300,
        )
    if policy_id == "deliberate" and version == "3":
        return PolicySpec(
            id="deliberate",
            version="3",
            system_prompt=(project_root / "policies" / "deliberate-v3.md")
            .read_text(encoding="utf-8")
            .strip(),
            allowed_tools=("bash",),
            max_output_tokens=2560,
            tool_call_limit=8,
            command_timeout_seconds=45,
            wall_time_limit_seconds=420,
        )
    if policy_id == "direct" and version == "4":
        return PolicySpec(
            id="direct",
            version="4",
            system_prompt=(project_root / "policies" / "direct-v4.md")
            .read_text(encoding="utf-8")
            .strip(),
            allowed_tools=("bash",),
            max_output_tokens=1536,
            tool_call_limit=7,
            command_timeout_seconds=30,
            wall_time_limit_seconds=300,
        )
    if policy_id == "deliberate" and version == "4":
        return PolicySpec(
            id="deliberate",
            version="4",
            system_prompt=(project_root / "policies" / "deliberate-v4.md")
            .read_text(encoding="utf-8")
            .strip(),
            allowed_tools=("bash",),
            max_output_tokens=2560,
            tool_call_limit=8,
            command_timeout_seconds=45,
            wall_time_limit_seconds=420,
        )
    raise ValueError(f"unsupported policy: {policy_id!r}")


def policy_spec_from_treatment(treatment: TreatmentSpec) -> PolicySpec:
    """Convert one immutable registry entry into an executable policy.

    The MVP Pi gym exposes only the native sandboxed ``bash`` interface. Other
    interfaces remain valid registry descriptors but cannot be executed by this
    orchestrator until a matching tool adapter exists.
    """

    if treatment.tool_interface != "native_bash":
        raise ValueError(
            f"unsupported treatment tool interface: {treatment.tool_interface!r}"
        )
    if tuple(treatment.allowed_tools) != ("bash",):
        raise ValueError(
            "the current gym can execute only allowed_tools=['bash']; got "
            f"{list(treatment.allowed_tools)!r}"
        )
    return PolicySpec(
        **to_policy_spec_kwargs(treatment),
        tool_interface=treatment.tool_interface,
        bundle_hash=treatment.bundle_hash,
        enforce_budget=True,
    )


def _run_pi(
    project_root: Path,
    config: RemoteConfig,
    workspace: str,
    prompt: str,
    policy: PolicySpec,
    pi_executable: str,
    model_switch_extension: Path | None,
    provider: str = "ubuntu-gemma",
    model: str = "gemma-4-26b-a4b",
    thinking: str = "off",
) -> subprocess.CompletedProcess[str]:
    # Keep the extension outside .pi/extensions so normal Pi sessions in this
    # repository never auto-discover the restrictive gym tool configuration.
    gym_extension = project_root / "pi_extensions" / "gym-tools.ts"
    budget_extension = project_root / "pi_extensions" / "gym-budget-v2.ts"
    command = [
        pi_executable,
        "--provider",
        provider,
        "--model",
        model,
        "--thinking",
        thinking,
        "--mode",
        "json",
        "--print",
        "--no-session",
        "--no-builtin-tools",
        "--tools",
        "bash",
        "--no-context-files",
        "--no-skills",
        "--no-prompt-templates",
        "--no-extensions",
        "--no-approve",
    ]
    if model_switch_extension is not None:
        command.extend(["--extension", str(model_switch_extension)])
    command.extend(["--extension", str(gym_extension)])
    if policy.enforce_budget or policy.version in {"2", "3", "4"}:
        command.extend(["--extension", str(budget_extension)])
    command.extend(
        [
            "--gym-host",
            config.host,
            "--gym-python",
            config.python,
            "--gym-project",
            config.project,
            "--gym-root",
            config.run_root,
            "--gym-workspace",
            workspace,
            "--gym-tool-limit",
            str(policy.tool_call_limit),
            "--gym-command-timeout",
            str(policy.command_timeout_seconds),
            "--gym-memory-max",
            "1G",
            "--gym-tasks-max",
            "64",
            "--gym-cpu-quota",
            "200%",
            "--gym-max-output-tokens",
            str(policy.max_output_tokens),
            "--append-system-prompt",
            policy.system_prompt,
            prompt,
        ]
    )
    return subprocess.run(
        command,
        cwd=project_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=policy.wall_time_limit_seconds,
    )


def _run_pi_checked(
    project_root: Path,
    config: RemoteConfig,
    workspace: str,
    prompt: str,
    policy: PolicySpec,
    pi_executable: str,
    model_switch_extension: Path | None,
    provider: str = "ubuntu-gemma",
    model: str = "gemma-4-26b-a4b",
    thinking: str = "off",
) -> subprocess.CompletedProcess[str]:
    """Run Pi, converting a wall-clock timeout into a failed run.

    A hung Pi process is still an attempt, not a harness failure: verification
    must still inspect the final workspace and report diagnostics.
    """
    try:
        return _run_pi(
            project_root,
            config,
            workspace,
            prompt,
            policy,
            pi_executable,
            model_switch_extension,
            provider,
            model,
            thinking,
        )
    except subprocess.TimeoutExpired as error:

        def _text(value: str | bytes | None) -> str:
            if value is None:
                return ""
            if isinstance(value, str):
                return value
            return value.decode("utf-8", errors="replace")

        return subprocess.CompletedProcess(
            args=error.cmd or [],
            returncode=-1,
            stdout=_text(error.stdout),
            stderr=_text(error.stderr)
            or f"pi process timed out after {policy.wall_time_limit_seconds}s",
        )


def _attempt_event_summary(config: RemoteConfig, attempt_id: str) -> dict[str, Any] | None:
    """Best-effort normalized summary from the recorded Pi events (raw JSONL).

    ``record-events`` persists the raw JSONL at
    ``<run_root>/attempts/<attempt_id>/pi-events.jsonl``; re-normalizing it
    with the generic ``normalize-events`` command yields the event summary.
    Returns ``None`` when no events were recorded or the file is unreachable.
    """
    events_path = f"{config.run_root}/attempts/{attempt_id}/pi-events.jsonl"
    try:
        return remote_json(config, ["normalize-events", events_path])
    except RuntimeError:
        return None


def _pair_order(seed: int | str, policies: Sequence[str]) -> list[str]:
    """Deterministic per-task execution order for the two policies."""
    ordered = list(policies)
    random.Random(seed).shuffle(ordered)
    return ordered


def _task_json(config: RemoteConfig, args: argparse.Namespace) -> dict[str, Any]:
    return remote_json(
        config,
        [
            "generate",
            "--family",
            args.family,
            "--root",
            config.run_root,
            "--seed",
            str(args.seed),
            "--difficulty",
            args.difficulty,
        ],
    )


def _run_attempt(
    project_root: Path,
    config: RemoteConfig,
    task: dict[str, Any],
    policy: PolicySpec,
    attempt_id: str,
    args: argparse.Namespace,
    *,
    with_usage: bool,
    registry_hash: str | None = None,
) -> dict[str, Any]:
    """Prepare a fresh attempt, run the policy in Pi, record events, verify."""
    attempt_started = time.monotonic()
    phase_started = attempt_started
    prepare_arguments = [
        "prepare-attempt",
        "--root",
        config.run_root,
        "--task-id",
        task["id"],
        "--attempt-id",
        attempt_id,
        "--policy-id",
        policy.id,
        "--policy-version",
        policy.version,
    ]
    if policy.bundle_hash is not None:
        prepare_arguments.extend(
            ["--treatment-bundle-hash", policy.bundle_hash]
        )
    if registry_hash is not None:
        prepare_arguments.extend(
            ["--treatment-registry-hash", registry_hash]
        )
    attempt = remote_json(config, prepare_arguments)
    prepare_seconds = time.monotonic() - phase_started

    phase_started = time.monotonic()
    switch_value = getattr(args, "model_switch_extension", None)
    switch_extension = (
        Path(switch_value).expanduser().resolve() if switch_value else None
    )
    if switch_extension is not None and not switch_extension.is_file():
        raise ValueError(
            f"model switch extension does not exist: {switch_extension}"
        )
    completed = _run_pi_checked(
        project_root,
        config,
        attempt["workspace_ref"],
        task["prompt"],
        policy,
        args.pi,
        switch_extension,
        getattr(args, "provider", "ubuntu-gemma"),
        getattr(args, "model", "gemma-4-26b-a4b"),
        getattr(args, "thinking", "off"),
    )
    pi_seconds = time.monotonic() - phase_started

    record_seconds = 0.0
    if completed.stdout.strip():
        phase_started = time.monotonic()
        remote_json(
            config,
            ["record-events", "--root", config.run_root, "--attempt-id", attempt_id],
            input_text=completed.stdout,
        )
        record_seconds = time.monotonic() - phase_started
    # Verification always runs, even when Pi failed, so the final workspace is
    # inspected and reported; a failed verification never suppresses the report.
    phase_started = time.monotonic()
    verification = remote_json(
        config,
        [
            "verify",
            "--family",
            args.family,
            "--root",
            config.run_root,
            "--task-id",
            task["id"],
            "--attempt-id",
            attempt_id,
        ],
    )
    verify_seconds = time.monotonic() - phase_started
    result = {
        "task_id": task["id"],
        "attempt_id": attempt_id,
        "policy": policy.to_dict(),
        "pi_return_code": completed.returncode,
        "pi_stderr": completed.stderr[-4000:],
        "verification": verification,
    }
    usage_seconds = 0.0
    if with_usage:
        phase_started = time.monotonic()
        event_summary = _attempt_event_summary(config, attempt_id)
        usage_seconds = time.monotonic() - phase_started
        usage = event_summary.get("usage") if isinstance(event_summary, dict) else None
        result["usage"] = usage if isinstance(usage, dict) else None
        if isinstance(event_summary, dict):
            result["trajectory"] = {
                "provider_turn_count": event_summary.get("provider_turn_count"),
                "tool_call_count": event_summary.get("tool_call_count"),
                "tool_limit_rejection_count": event_summary.get(
                    "tool_limit_rejection_count"
                ),
                "length_stop_count": event_summary.get("length_stop_count"),
                "stop_reasons": event_summary.get("stop_reasons"),
            }
    result["timing"] = {
        "prepare_seconds": round(prepare_seconds, 3),
        "pi_seconds": round(pi_seconds, 3),
        "record_seconds": round(record_seconds, 3),
        "verify_seconds": round(verify_seconds, 3),
        "usage_seconds": round(usage_seconds, 3),
        "total_seconds": round(time.monotonic() - attempt_started, 3),
    }
    return result


def run_single(
    project_root: Path, config: RemoteConfig, args: argparse.Namespace
) -> dict[str, Any]:
    policy = policy_spec(project_root, args.policy, getattr(args, "policy_version", "1"))
    task = _task_json(config, args)
    attempt_id = args.attempt_id or f"smoke-{policy.id}-{uuid.uuid4().hex[:12]}"
    return _run_attempt(project_root, config, task, policy, attempt_id, args, with_usage=False)


def run_pair(
    project_root: Path, config: RemoteConfig, args: argparse.Namespace
) -> dict[str, Any]:
    """Direct + Deliberate on fresh attempts for the same task.

    The task is generated once; the policies share it. Execution order is
    randomized deterministically from the seed.
    """
    policy_version = getattr(args, "policy_version", "1")
    policies = {
        "direct": policy_spec(project_root, "direct", policy_version),
        "deliberate": policy_spec(project_root, "deliberate", policy_version),
    }
    return _run_policy_set(project_root, config, args, policies, mode="pair")


def _run_policy_set(
    project_root: Path,
    config: RemoteConfig,
    args: argparse.Namespace,
    policies: dict[str, PolicySpec],
    *,
    mode: str,
    registry_hash: str | None = None,
) -> dict[str, Any]:
    """Execute an arbitrary fixed treatment menu on one generated task."""

    if not policies:
        raise ValueError("at least one policy treatment is required")
    task = _task_json(config, args)
    policy_versions = {policy.version for policy in policies.values()}
    order_key: int | str = (
        task["id"] if policy_versions != {"1"} or mode != "pair" else args.seed
    )
    execution_order = _pair_order(order_key, list(policies))
    attempts: list[tuple[str, dict[str, Any]]] = []
    for treatment_ref in execution_order:
        policy = policies[treatment_ref]
        prefix = "pair" if mode == "pair" else "treatment"
        attempt_id = f"{prefix}-{policy.id}-{uuid.uuid4().hex[:12]}"
        attempts.append(
            (
                treatment_ref,
                _run_attempt(
                    project_root,
                    config,
                    task,
                    policy,
                    attempt_id,
                    args,
                    with_usage=True,
                    registry_hash=registry_hash,
                ),
            )
        )
    result = {
        "task_id": task["id"],
        "mode": mode,
        "execution_order": execution_order,
        "attempts": {
            treatment_ref: {
                "attempt_id": attempt["attempt_id"],
                "policy": attempt["policy"],
                "pi_return_code": attempt["pi_return_code"],
                "pi_stderr": attempt["pi_stderr"],
                "verification": attempt["verification"],
                "usage": attempt.get("usage"),
                "trajectory": attempt.get("trajectory"),
                "timing": attempt.get("timing"),
            }
            for treatment_ref, attempt in attempts
        },
    }
    if registry_hash is not None:
        result["treatment_registry_hash"] = registry_hash
    return result


def _resolve_treatment_reference(
    registry: TreatmentRegistry, reference: str
) -> TreatmentSpec:
    try:
        return registry.by_bundle_id(reference)
    except KeyError:
        pass
    if "@" in reference:
        treatment_id, version = reference.rsplit("@", 1)
        try:
            return registry.by_id_version(treatment_id, version)
        except KeyError:
            pass
    return registry.by_id(reference)


def run_registered_treatments(
    project_root: Path,
    config: RemoteConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Load and execute the requested treatments from an immutable registry."""

    if not args.treatment_registry:
        raise ValueError("--treatment-registry is required")
    registry = TreatmentRegistry.load(args.treatment_registry)
    if str(args.treatments or "").strip().lower() == "all":
        references = [treatment.bundle_id for treatment in registry]
    else:
        references = [
            value.strip()
            for value in str(args.treatments or "").split(",")
            if value.strip()
        ]
    if not references:
        raise ValueError("--treatments must list at least one registry treatment")
    selected: dict[str, PolicySpec] = {}
    for reference in references:
        try:
            treatment = _resolve_treatment_reference(registry, reference)
        except KeyError as error:
            raise ValueError(f"unknown treatment reference: {reference!r}") from error
        if treatment.bundle_id in selected:
            raise ValueError(f"duplicate treatment selection: {reference!r}")
        selected[treatment.bundle_id] = policy_spec_from_treatment(treatment)
    return _run_policy_set(
        project_root,
        config,
        args,
        selected,
        mode="treatment_set",
        registry_hash=registry.registry_hash,
    )


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    config = RemoteConfig(
        host=args.host,
        project=args.remote_project,
        run_root=args.remote_run_root,
        python=args.remote_python,
    )
    validate_remote_config(config)
    if getattr(args, "treatment_registry", None) or getattr(args, "treatments", None):
        if getattr(args, "pair", False):
            raise ValueError("--pair cannot be combined with a treatment registry")
        return run_registered_treatments(project_root, config, args)
    if args.pair:
        return run_pair(project_root, config, args)
    return run_single(project_root, config, args)


def _summary_ok(result: dict[str, Any]) -> bool:
    if "verification" in result:  # Single-policy result.
        return bool(result["verification"]["success"])
    attempts = result.get("attempts") or {}
    return bool(attempts) and all(
        bool(item["verification"]["success"]) for item in attempts.values()
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyreplab-harness-smoke")
    remote_project = os.environ.get("PYREPLAB_REMOTE_PROJECT") or None
    remote_run_root = os.environ.get("PYREPLAB_REMOTE_RUN_ROOT") or (
        f"{remote_project}/.runs" if remote_project else None
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("PYREPLAB_HARNESS_HOST", "ubuntu-local"),
        help="SSH host or alias for the disposable Linux runner",
    )
    parser.add_argument(
        "--remote-project",
        default=remote_project,
        help="absolute project path on the remote host (or PYREPLAB_REMOTE_PROJECT)",
    )
    parser.add_argument(
        "--remote-run-root",
        default=remote_run_root,
        help="absolute remote run root (or PYREPLAB_REMOTE_RUN_ROOT)",
    )
    parser.add_argument("--remote-python", default="python3")
    parser.add_argument("--pi", default=os.environ.get("PYREPLAB_PI", "pi"))
    parser.add_argument(
        "--provider",
        default=os.environ.get("PYREPLAB_PI_PROVIDER", "ubuntu-gemma"),
        help="Pi provider name (or PYREPLAB_PI_PROVIDER)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("PYREPLAB_PI_MODEL", "gemma-4-26b-a4b"),
        help="Pi model name (or PYREPLAB_PI_MODEL)",
    )
    parser.add_argument(
        "--thinking",
        default=os.environ.get("PYREPLAB_PI_THINKING", "off"),
        help="Pi thinking level (or PYREPLAB_PI_THINKING)",
    )
    parser.add_argument(
        "--model-switch-extension",
        default=os.environ.get("PYREPLAB_MODEL_SWITCH_EXTENSION") or None,
        help="optional local Pi provider-switch extension",
    )
    parser.add_argument("--policy", choices=["direct", "deliberate"], default="direct")
    parser.add_argument(
        "--policy-version",
        choices=["1", "2", "3", "4"],
        default="1",
        help="immutable Direct/Deliberate policy version (default: 1)",
    )
    parser.add_argument("--family", choices=FAMILIES, default="artifact")
    parser.add_argument(
        "--pair",
        action="store_true",
        help="run both direct and deliberate policies on fresh attempts for the "
        "same generated task; execution order is derived deterministically from "
        "--seed",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    parser.add_argument("--attempt-id")
    parser.add_argument(
        "--treatment-registry",
        default=None,
        help="immutable JSON treatment registry for generalized policy execution",
    )
    parser.add_argument(
        "--treatments",
        default=None,
        help="comma-separated registry IDs/id@version/bundle IDs, or 'all'",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_smoke(args)
    except (RuntimeError, subprocess.TimeoutExpired, ValueError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 1
    ok = _summary_ok(result)
    print(json.dumps({"ok": ok, **result}, indent=2, sort_keys=True))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
