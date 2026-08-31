"""Freeze and preflight the excluded Unbrowser empty-overlay baseline screen.

This module deliberately has no live-attempt runner. It prepares immutable
registry/manifest artifacts and two preflights:

* ``local-preflight`` verifies all deterministic v3 task prompts and the exact
  Pi command template without invoking Pi or Unbrowser.
* ``preflight`` verifies the pinned remote runtime and a confined persistent
  Unbrowser session across the historical 30-second lifecycle boundary. It
  does not invoke the model.

Live model execution requires a separate, explicit authorization artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .events import (
    BUDGET_RECEIPT_SCHEMA_VERSION,
    NORMALIZED_EVENT_SCHEMA_VERSION,
    PROVIDER_TURN_SEMANTICS,
)
from .fixture_server import FixtureServer
from .m3_pilot import (
    HELD_TEMPLATES,
    KNOWN_TEMPLATES,
    _RUNTIME_PINS,
    _canonical_hash,
    _load_json,
    _ssh_capture,
    _verify_embedded_hash,
    _write_immutable_json,
    runtime_preflight,
    source_tree_hash,
)
from .orchestrator import (
    RESTRICTED_BASELINE_EXECUTION_PATH,
    UNBROWSER_INTERACTIVE_TOOL_INTERFACE,
    RemoteConfig,
    _build_pi_command,
    policy_spec_from_treatment,
)
from .treatments import TreatmentRegistry, TreatmentSpec
from .unbrowser_fixture_gym import (
    FIXTURE_BASE_URL,
    OUTCOME_ONLY_GENERATOR_VERSION,
    generate_unbrowser_fixture_task,
    unbrowser_fixture_task_commitment,
)
from .unbrowser_rpc import UnbrowserSession

MANIFEST_SCHEMA_VERSION = "m3-empty-overlay-baseline-manifest-v4"
LOCAL_PREFLIGHT_SCHEMA_VERSION = "m3-empty-overlay-local-preflight-v5"
REMOTE_PREFLIGHT_SCHEMA_VERSION = "m3-empty-overlay-remote-preflight-v5"
LIFECYCLE_RECEIPT_SCHEMA_VERSION = "m3-unbrowser-lifecycle-stress-v1"
COMMAND_RECEIPT_SCHEMA_VERSION = "m3-empty-overlay-command-template-v2"

SCREEN_ID = "m3-empty-overlay-baseline-20260815-v5"
TASK_ROLE = "T_pilot"
TASK_SPLIT = "pilot_excluded"
DIFFICULTIES = ("easy", "medium", "hard")
TASK_SEEDS_PER_CELL = 2
ROLLOUT_REPLICAS = 2
TASK_SEED_START = 2026091001
SAMPLING_SEED_START = 1900009001
SCHEDULE_SEED = 2026081503
EXPECTED_TASKS = len(KNOWN_TEMPLATES) * len(DIFFICULTIES) * TASK_SEEDS_PER_CELL
EXPECTED_ATTEMPTS = EXPECTED_TASKS * ROLLOUT_REPLICAS

BASELINE_TREATMENT_ID = "ub-empty-overlay"
BASELINE_TREATMENT_VERSION = "1"
_TASK_PROMPT_PLACEHOLDER = "__PYREPLAB_TASK_PROMPT__"
_WORKSPACE_PLACEHOLDER = "__PYREPLAB_WORKSPACE__"
_POLICY_LEAKAGE_MARKERS = (
    "assigned recovery policy",
    "fail-fast policies",
    "retry policies",
    "first open the link",
    "if navigation reports",
)


def build_empty_overlay_registry() -> TreatmentRegistry:
    treatment = TreatmentSpec(
        id=BASELINE_TREATMENT_ID,
        version=BASELINE_TREATMENT_VERSION,
        system_prompt="",
        allowed_tools=("bash", "unbrowser"),
        max_output_tokens=4096,
        tool_call_limit=12,
        command_timeout_seconds=60,
        wall_time_limit_seconds=600,
        tool_interface=UNBROWSER_INTERACTIVE_TOOL_INTERFACE,
        generator_metadata={
            "experimental_variable": "system_prompt_overlay",
            "execution_path": RESTRICTED_BASELINE_EXECUTION_PATH,
            "prompt_overlay": "empty",
            "task_prompt_profile": "outcome_only_v1",
            "treatment_kind": "baseline",
        },
    )
    return TreatmentRegistry((treatment,))


def _validate_registry(registry: TreatmentRegistry) -> TreatmentSpec:
    expected = build_empty_overlay_registry()
    if registry.to_dict() != expected.to_dict():
        raise ValueError("baseline treatment registry drifted")
    treatment = registry.treatments[0]
    if treatment.system_prompt != "":
        raise ValueError("baseline treatment must have an exact empty system prompt")
    return treatment


def _validate_remote_identity(value: Mapping[str, Any]) -> dict[str, str]:
    host = str(value.get("host", "")).strip()
    python = str(value.get("python", "python3")).strip()
    project = str(value.get("project", ""))
    run_root = str(value.get("run_root", ""))
    if not host or not python:
        raise ValueError("remote host and python must be non-empty")
    for label, path in (("project", project), ("run_root", run_root)):
        if not path.startswith("/") or path == "/":
            raise ValueError(f"remote {label} must be an absolute path other than '/'")
    return {
        "host": host,
        "project": project,
        "run_root": run_root,
        "python": python,
    }


def _build_tasks() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for template in KNOWN_TEMPLATES:
        for difficulty in DIFFICULTIES:
            for seed_replica in range(TASK_SEEDS_PER_CELL):
                seed = TASK_SEED_START + len(tasks)
                tasks.append(
                    {
                        "task_id": (
                            f"{OUTCOME_ONLY_GENERATOR_VERSION}-{template}-"
                            f"{difficulty}-{seed}"
                        ),
                        "role": TASK_ROLE,
                        "split": TASK_SPLIT,
                        "template": template,
                        "difficulty": difficulty,
                        "seed": seed,
                        "seed_replica": seed_replica,
                        "generator_version": OUTCOME_ONLY_GENERATOR_VERSION,
                    }
                )
    return tasks


def _build_panels(
    tasks: list[dict[str, Any]], bundle_id: str
) -> list[dict[str, Any]]:
    coordinates = [
        (task_index, rollout_replica)
        for task_index in range(len(tasks))
        for rollout_replica in range(ROLLOUT_REPLICAS)
    ]
    random.Random(SCHEDULE_SEED).shuffle(coordinates)
    panels: list[dict[str, Any]] = []
    for panel_index, (task_index, rollout_replica) in enumerate(coordinates):
        task = tasks[task_index]
        panels.append(
            {
                "panel_id": f"{task['task_id']}/replica={rollout_replica}",
                "task_id": task["task_id"],
                "rollout_replica": rollout_replica,
                "sampling_seed": SAMPLING_SEED_START + panel_index,
                "execution_order": [bundle_id],
            }
        )
    return panels


def _baseline_runtime_pins() -> dict[str, Any]:
    pins = json.loads(json.dumps(_RUNTIME_PINS))
    pins["rollout_replicas"] = ROLLOUT_REPLICAS
    return pins


def _treatment_descriptor(
    treatment: TreatmentSpec, registry: TreatmentRegistry
) -> dict[str, Any]:
    return {
        "id": treatment.id,
        "version": treatment.version,
        "bundle_id": treatment.bundle_id,
        "bundle_hash": treatment.bundle_hash,
        "registry_hash": registry.registry_hash,
        "system_prompt_length": 0,
        "system_prompt_sha256": hashlib.sha256(b"").hexdigest(),
        "append_system_prompt": False,
        "allowed_tools": list(treatment.allowed_tools),
        "tool_interface": treatment.tool_interface,
        "max_output_tokens": treatment.max_output_tokens,
        "tool_call_limit": treatment.tool_call_limit,
        "command_timeout_seconds": treatment.command_timeout_seconds,
        "wall_time_limit_seconds": treatment.wall_time_limit_seconds,
    }


def build_baseline_manifest(
    registry: TreatmentRegistry,
    remote_identity: Mapping[str, Any],
    *,
    registry_file: str,
) -> dict[str, Any]:
    treatment = _validate_registry(registry)
    remote = _validate_remote_identity(remote_identity)
    tasks = _build_tasks()
    task_commitments = _build_task_commitments({"tasks": tasks})
    commitment_by_id = {
        str(item["task"]["id"]): item for item in task_commitments
    }
    for task in tasks:
        task["task_commitment_hash"] = commitment_by_id[task["task_id"]][
            "commitment_hash"
        ]
    panels = _build_panels(tasks, treatment.bundle_id)
    payload: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "screen_id": SCREEN_ID,
        "purpose": (
            "Measure repeatable behavioral failures under an exact empty system-"
            "prompt overlay before any prompt-policy comparison."
        ),
        "registry_file": registry_file,
        "registry_hash": registry.registry_hash,
        "baseline_treatment": _treatment_descriptor(treatment, registry),
        "known_templates": list(KNOWN_TEMPLATES),
        "held_templates": list(HELD_TEMPLATES),
        "task_role": TASK_ROLE,
        "task_split": TASK_SPLIT,
        "task_prompt_profile": "outcome_only_v1",
        "task_generator_version": OUTCOME_ONLY_GENERATOR_VERSION,
        "tasks": tasks,
        "panels": panels,
        "rollout_replicas": ROLLOUT_REPLICAS,
        "task_seed_start": TASK_SEED_START,
        "sampling_seed_start": SAMPLING_SEED_START,
        "schedule_seed": SCHEDULE_SEED,
        "runtime_pins": _baseline_runtime_pins(),
        "event_accounting": {
            "normalizer_schema_version": NORMALIZED_EVENT_SCHEMA_VERSION,
            "provider_turn_semantics": PROVIDER_TURN_SEMANTICS,
            "budget_receipt_schema_version": BUDGET_RECEIPT_SCHEMA_VERSION,
        },
        "remote_identity": remote,
        "command_contract": {
            "append_system_prompt_flag": "absent",
            "task_prompt_position": "final_positional_argument",
            "context_files": "disabled",
            "skills": "disabled",
            "prompt_templates": "disabled",
            "builtin_tools": "disabled",
            "optional_observation_enforcement": "disabled",
            "semantic_capability": "disabled",
            "provider_turn_limit": 13,
            "tool_attempt_limit": 13,
            "tool_admission_limit": 12,
        },
        "gates": {
            "templates": len(KNOWN_TEMPLATES),
            "difficulties": len(DIFFICULTIES),
            "task_seeds_per_cell": TASK_SEEDS_PER_CELL,
            "tasks": EXPECTED_TASKS,
            "rollout_replicas": ROLLOUT_REPLICAS,
            "panels": EXPECTED_ATTEMPTS,
            "attempts": EXPECTED_ATTEMPTS,
            "infrastructure_errors_allowed": 0,
            "minimum_lifecycle_stress_seconds": 35,
        },
        "authorization_boundary": {
            "live_model_execution_authorized": False,
            "required_next_artifact": "explicit hash-bound execution authorization",
        },
        "exclusion": (
            "T_pilot tasks and all attempts are permanently excluded from meta-"
            "training, calibration, development, and final evaluation pools."
        ),
    }
    return {**payload, "manifest_hash": _canonical_hash(payload)}


def validate_baseline_manifest(
    manifest: Mapping[str, Any], registry: TreatmentRegistry
) -> None:
    _verify_embedded_hash(manifest, "manifest_hash")
    registry_file = manifest.get("registry_file")
    if not isinstance(registry_file, str) or not registry_file:
        raise ValueError("manifest registry_file must be a non-empty string")
    remote = manifest.get("remote_identity")
    if not isinstance(remote, Mapping):
        raise ValueError("manifest remote_identity must be an object")
    expected = build_baseline_manifest(
        registry,
        remote,
        registry_file=registry_file,
    )
    if dict(manifest) != expected:
        raise ValueError("baseline manifest drifted from its frozen design")
    task_templates = {task["template"] for task in manifest["tasks"]}
    if task_templates != set(KNOWN_TEMPLATES):
        raise ValueError("baseline manifest must use all and only known templates")
    if task_templates & set(HELD_TEMPLATES):
        raise ValueError("held templates must remain unseen until final evaluation")


def freeze_baseline_artifacts(
    registry_output: str | Path,
    manifest_output: str | Path,
    remote_identity: Mapping[str, Any],
) -> dict[str, Any]:
    registry = build_empty_overlay_registry()
    registry_path = Path(registry_output).expanduser().resolve()
    manifest_path = Path(manifest_output).expanduser().resolve()
    manifest = build_baseline_manifest(
        registry,
        remote_identity,
        registry_file=registry_path.name,
    )
    validate_baseline_manifest(manifest, registry)
    _write_immutable_json(registry_path, registry.to_dict())
    _write_immutable_json(manifest_path, manifest)
    return {
        "registry": str(registry_path),
        "registry_hash": registry.registry_hash,
        "manifest": str(manifest_path),
        "manifest_hash": manifest["manifest_hash"],
        "tasks": EXPECTED_TASKS,
        "attempts": EXPECTED_ATTEMPTS,
        "live_model_execution_authorized": False,
    }


def _value_after(arguments: list[str], flag: str) -> str | None:
    try:
        return arguments[arguments.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def build_command_template_receipt(
    manifest: Mapping[str, Any],
    registry: TreatmentRegistry,
    controller_project_root: str | Path,
    *,
    pi_executable: str = "pi",
) -> dict[str, Any]:
    validate_baseline_manifest(manifest, registry)
    treatment = _validate_registry(registry)
    policy = policy_spec_from_treatment(treatment)
    remote = manifest["remote_identity"]
    first_panel = manifest["panels"][0]
    first_task = next(
        task for task in manifest["tasks"] if task["task_id"] == first_panel["task_id"]
    )
    project_root = Path(controller_project_root).expanduser().resolve()
    command = _build_pi_command(
        project_root,
        RemoteConfig(
            str(remote["host"]),
            str(remote["project"]),
            str(remote["run_root"]),
            str(remote["python"]),
        ),
        f"{remote['run_root']}/{_WORKSPACE_PLACEHOLDER}",
        _TASK_PROMPT_PLACEHOLDER,
        policy,
        pi_executable,
        None,
        str(manifest["runtime_pins"]["provider"]),
        str(manifest["runtime_pins"]["model_alias"]),
        str(manifest["runtime_pins"]["thinking"]),
        (
            f"{FIXTURE_BASE_URL}/{first_task['template']}/"
            f"{first_task['seed']}/{first_task['difficulty']}"
        ),
        str(manifest["runtime_pins"]["unbrowser_path"]),
        True,
        True,
        int(first_panel["sampling_seed"]),
    )
    checks = {
        "append_system_prompt_absent": "--append-system-prompt" not in command,
        "task_prompt_is_final": command[-1] == _TASK_PROMPT_PLACEHOLDER,
        "no_context_files": "--no-context-files" in command,
        "no_skills": "--no-skills" in command,
        "no_prompt_templates": "--no-prompt-templates" in command,
        "no_builtin_tools": "--no-builtin-tools" in command,
        "plain_interactive_interface": (
            "--gym-unbrowser-interactive" in command
            and "--gym-unbrowser-required-first-observation" not in command
            and "--gym-semantic-capability" not in command
        ),
        "tool_limit_frozen": (
            _value_after(command, "--gym-tool-limit")
            == str(treatment.tool_call_limit)
        ),
        "provider_turn_limit_frozen": (
            _value_after(command, "--gym-provider-turn-limit")
            == str(treatment.tool_call_limit + 1)
        ),
        "budget_v3_extension_loaded": any(
            argument.endswith("/pi_extensions/gym-budget-v3.ts")
            for argument in command
        ),
        "output_limit_frozen": (
            _value_after(command, "--gym-max-output-tokens")
            == str(treatment.max_output_tokens)
        ),
        "command_timeout_frozen": (
            _value_after(command, "--gym-command-timeout")
            == str(treatment.command_timeout_seconds)
        ),
        "unbrowser_limit_frozen": (
            _value_after(command, "--gym-unbrowser-tool-limit") == "12"
        ),
        "filesystem_confinement_enabled": (
            _value_after(command, "--gym-confine-unbrowser") == "true"
        ),
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"baseline Pi command contract failed: {failed!r}")
    payload: dict[str, Any] = {
        "schema_version": COMMAND_RECEIPT_SCHEMA_VERSION,
        "manifest_hash": manifest["manifest_hash"],
        "registry_hash": registry.registry_hash,
        "treatment_bundle_hash": treatment.bundle_hash,
        "argv": command,
        "argv_sha256": hashlib.sha256(
            json.dumps(command, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
        "checks": checks,
    }
    return {**payload, "receipt_hash": _canonical_hash(payload)}


def _build_task_commitments(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    commitments: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="pyreplab-empty-overlay-") as directory:
        for entry in manifest["tasks"]:
            task = generate_unbrowser_fixture_task(
                directory,
                int(entry["seed"]),
                str(entry["difficulty"]),
                str(entry["template"]),
                task_role=TASK_ROLE,
                generator_version=OUTCOME_ONLY_GENERATOR_VERSION,
            )
            if task.id != entry["task_id"]:
                raise ValueError(f"generated task id drifted: {task.id!r}")
            if task.public_metadata.get("prompt_profile") != "outcome_only_v1":
                raise ValueError(f"task omitted outcome-only prompt profile: {task.id}")
            prompt_lower = task.prompt.casefold()
            leaked = [
                marker
                for marker in _POLICY_LEAKAGE_MARKERS
                if marker.casefold() in prompt_lower
            ]
            if leaked:
                raise ValueError(
                    f"task prompt contains policy leakage {leaked!r}: {task.id}"
                )
            commitments.append(
                unbrowser_fixture_task_commitment(directory, task.id)
            )
            expected_hash = entry.get("task_commitment_hash")
            if (
                expected_hash is not None
                and commitments[-1]["commitment_hash"] != expected_hash
            ):
                raise ValueError(f"task commitment drifted: {task.id}")
    return commitments


def build_local_preflight(
    manifest: Mapping[str, Any],
    registry: TreatmentRegistry,
    project_root: str | Path,
    *,
    pi_executable: str = "pi",
) -> dict[str, Any]:
    validate_baseline_manifest(manifest, registry)
    task_receipts = _build_task_commitments(manifest)
    command_receipt = build_command_template_receipt(
        manifest,
        registry,
        project_root,
        pi_executable=pi_executable,
    )
    payload: dict[str, Any] = {
        "schema_version": LOCAL_PREFLIGHT_SCHEMA_VERSION,
        "manifest_hash": manifest["manifest_hash"],
        "registry_hash": registry.registry_hash,
        "source_tree_hash": source_tree_hash(project_root),
        "generated_task_count": len(task_receipts),
        "generated_tasks_sha256": _canonical_hash({"tasks": task_receipts}),
        "generated_tasks": task_receipts,
        "command_template_receipt": command_receipt,
        "held_templates_consumed": False,
        "policy_leakage_markers_found": 0,
        "live_runtime_checked": False,
        "live_model_execution_authorized": False,
    }
    return {**payload, "preflight_hash": _canonical_hash(payload)}


def validate_local_preflight(
    preflight: Mapping[str, Any],
    manifest: Mapping[str, Any],
    registry: TreatmentRegistry,
    project_root: str | Path,
    *,
    pi_executable: str = "pi",
) -> None:
    _verify_embedded_hash(preflight, "preflight_hash")
    if preflight.get("schema_version") != LOCAL_PREFLIGHT_SCHEMA_VERSION:
        raise ValueError("unsupported local preflight schema")
    if preflight.get("manifest_hash") != manifest.get("manifest_hash"):
        raise ValueError("local preflight manifest hash mismatch")
    if preflight.get("registry_hash") != registry.registry_hash:
        raise ValueError("local preflight registry hash mismatch")
    if preflight.get("source_tree_hash") != source_tree_hash(project_root):
        raise ValueError("local preflight source tree hash mismatch")
    if preflight.get("generated_task_count") != EXPECTED_TASKS:
        raise ValueError("local preflight task count mismatch")
    task_commitments = preflight.get("generated_tasks")
    if not isinstance(task_commitments, list):
        raise ValueError("local preflight task commitments are missing")
    if preflight.get("generated_tasks_sha256") != _canonical_hash(
        {"tasks": task_commitments}
    ):
        raise ValueError("local preflight task commitment hash mismatch")
    if task_commitments != _build_task_commitments(manifest):
        raise ValueError("local preflight task commitments drifted")
    command_receipt = preflight.get("command_template_receipt")
    if command_receipt != build_command_template_receipt(
        manifest,
        registry,
        project_root,
        pi_executable=pi_executable,
    ):
        raise ValueError("local preflight command receipt drifted")
    if preflight.get("held_templates_consumed") is not False:
        raise ValueError("local preflight consumed held templates")
    if preflight.get("policy_leakage_markers_found") != 0:
        raise ValueError("local preflight found policy leakage")
    if preflight.get("live_model_execution_authorized") is not False:
        raise ValueError("local preflight must not authorize model execution")


def write_local_preflight(
    output_path: str | Path,
    manifest_path: str | Path,
    registry_path: str | Path,
    project_root: str | Path,
    *,
    pi_executable: str = "pi",
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    registry = TreatmentRegistry.load(registry_path)
    preflight = build_local_preflight(
        manifest,
        registry,
        project_root,
        pi_executable=pi_executable,
    )
    validate_local_preflight(
        preflight,
        manifest,
        registry,
        project_root,
        pi_executable=pi_executable,
    )
    output = Path(output_path).expanduser().resolve()
    _write_immutable_json(output, preflight)
    return {
        "local_preflight": str(output),
        "preflight_hash": preflight["preflight_hash"],
        "generated_tasks": preflight["generated_task_count"],
        "live_model_execution_authorized": False,
    }


def run_lifecycle_stress(
    unbrowser_binary: str,
    *,
    wait_seconds: float = 36.0,
) -> dict[str, Any]:
    if wait_seconds < 35:
        raise ValueError("lifecycle stress must wait at least 35 seconds")
    server = FixtureServer(port=18090)
    session: UnbrowserSession | None = None
    started = time.monotonic()
    try:
        url = server.url_for("single_page_extraction", TASK_SEED_START, "easy")
        session = UnbrowserSession(
            unbrowser_binary,
            url,
            timeout_seconds=30,
            interactive=True,
            confined=True,
        )
        navigation = session.execute({"action": "navigate"})
        status = navigation.get("result", {}).get("status")
        if status != 200:
            raise RuntimeError(f"lifecycle navigation returned status {status!r}")
        time.sleep(wait_seconds)
        observation = session.execute({"action": "text", "selector": "h1"})
        text = observation.get("result")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("lifecycle post-wait observation was empty")
        payload: dict[str, Any] = {
            "schema_version": LIFECYCLE_RECEIPT_SCHEMA_VERSION,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "wait_seconds": wait_seconds,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "fixture_url": url,
            "navigation_status": status,
            "post_wait_observation_sha256": hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest(),
            "runtime_version": session.runtime_version,
            "confined": True,
            "same_session": True,
            "passed": True,
        }
        return {**payload, "receipt_hash": _canonical_hash(payload)}
    finally:
        if session is not None:
            session.close()
        server.stop()


def _validate_lifecycle_receipt(receipt: Mapping[str, Any]) -> None:
    _verify_embedded_hash(receipt, "receipt_hash")
    if receipt.get("schema_version") != LIFECYCLE_RECEIPT_SCHEMA_VERSION:
        raise ValueError("unsupported lifecycle receipt schema")
    if receipt.get("passed") is not True or receipt.get("same_session") is not True:
        raise ValueError("lifecycle stress did not pass in one session")
    if float(receipt.get("wait_seconds", 0)) < 35:
        raise ValueError("lifecycle receipt did not cross the old timeout boundary")
    if float(receipt.get("elapsed_seconds", 0)) < 35:
        raise ValueError("lifecycle receipt measured less than 35 elapsed seconds")
    if receipt.get("navigation_status") != 200:
        raise ValueError("lifecycle receipt navigation status was not 200")
    if receipt.get("confined") is not True:
        raise ValueError("lifecycle receipt did not use filesystem confinement")
    if receipt.get("runtime_version") != _RUNTIME_PINS["unbrowser_version"]:
        raise ValueError("lifecycle receipt Unbrowser runtime version drifted")
    observation_hash = receipt.get("post_wait_observation_sha256")
    if (
        not isinstance(observation_hash, str)
        or len(observation_hash) != 64
        or any(character not in "0123456789abcdef" for character in observation_hash)
    ):
        raise ValueError("lifecycle receipt observation digest is invalid")


def validate_remote_preflight(
    preflight: Mapping[str, Any],
    manifest: Mapping[str, Any],
    registry: TreatmentRegistry,
    local_preflight: Mapping[str, Any],
    project_root: str | Path,
    *,
    pi_executable: str = "pi",
) -> None:
    validate_baseline_manifest(manifest, registry)
    validate_local_preflight(
        local_preflight,
        manifest,
        registry,
        project_root,
        pi_executable=pi_executable,
    )
    _verify_embedded_hash(preflight, "preflight_hash")
    if preflight.get("schema_version") != REMOTE_PREFLIGHT_SCHEMA_VERSION:
        raise ValueError("unsupported remote preflight schema")
    if preflight.get("manifest_hash") != manifest.get("manifest_hash"):
        raise ValueError("remote preflight manifest hash mismatch")
    if preflight.get("registry_hash") != registry.registry_hash:
        raise ValueError("remote preflight registry hash mismatch")
    if preflight.get("local_preflight_hash") != local_preflight.get("preflight_hash"):
        raise ValueError("remote preflight local receipt hash mismatch")
    lifecycle = preflight.get("lifecycle_receipt")
    if not isinstance(lifecycle, Mapping):
        raise ValueError("remote preflight lifecycle receipt must be an object")
    _validate_lifecycle_receipt(lifecycle)
    expected_command = build_command_template_receipt(
        manifest,
        registry,
        project_root,
        pi_executable=pi_executable,
    )
    if preflight.get("command_template_receipt") != expected_command:
        raise ValueError("remote preflight command receipt drifted")
    runtime = preflight.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("remote preflight runtime must be an object")
    if runtime.get("source_tree_hash") != local_preflight.get("source_tree_hash"):
        raise ValueError("remote preflight source tree hash mismatch")
    if runtime.get("runtime_pins") != manifest["runtime_pins"]:
        raise ValueError("remote preflight runtime pins drifted")
    if preflight.get("live_runtime_checked") is not True:
        raise ValueError("remote preflight did not check the live runtime")
    if preflight.get("live_model_execution_authorized") is not False:
        raise ValueError("remote preflight must not authorize model execution")
    if preflight.get("ready_for_authorization") is not True:
        raise ValueError("remote preflight is not ready for authorization")


def run_remote_preflight(
    output_path: str | Path,
    manifest_path: str | Path,
    registry_path: str | Path,
    local_preflight_path: str | Path,
    project_root: str | Path,
    config: RemoteConfig,
    *,
    pi_binary: str,
    thinking: str,
    unbrowser_binary: str,
    model_artifact: str,
    llama_server_binary: str,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    registry = TreatmentRegistry.load(registry_path)
    local_preflight = _load_json(local_preflight_path)
    validate_baseline_manifest(manifest, registry)
    validate_local_preflight(
        local_preflight,
        manifest,
        registry,
        project_root,
        pi_executable=pi_binary,
    )
    if dict(manifest["remote_identity"]) != {
        "host": config.host,
        "project": config.project,
        "run_root": config.run_root,
        "python": config.python,
    }:
        raise ValueError("remote preflight identity does not match manifest")
    runtime = runtime_preflight(
        Path(project_root).expanduser().resolve(),
        config,
        pi_binary=pi_binary,
        thinking=thinking,
        unbrowser_binary=unbrowser_binary,
        model_artifact=model_artifact,
        llama_server_binary=llama_server_binary,
        require_clean=False,
    )
    if runtime.get("runtime_pins") != manifest["runtime_pins"]:
        raise RuntimeError("remote runtime identity drifted from baseline pins")
    if runtime.get("source_tree_hash") != local_preflight.get("source_tree_hash"):
        raise RuntimeError("local preflight source tree differs from remote runtime")

    module_command = [
        "env",
        f"PYTHONPATH={config.project}/src",
        config.python,
        "-m",
        "pyreplab_harness.m3_empty_overlay_baseline",
    ]
    lifecycle = json.loads(
        _ssh_capture(
            config.host,
            [
                *module_command,
                "lifecycle-stress",
                "--unbrowser-binary",
                unbrowser_binary,
                "--wait-seconds",
                "36",
            ],
            timeout=90,
        )
    )
    if not isinstance(lifecycle, Mapping):
        raise RuntimeError("remote lifecycle stress returned a non-object")
    _validate_lifecycle_receipt(lifecycle)
    command_receipt = build_command_template_receipt(
        manifest,
        registry,
        project_root,
        pi_executable=pi_binary,
    )
    payload: dict[str, Any] = {
        "schema_version": REMOTE_PREFLIGHT_SCHEMA_VERSION,
        "manifest_hash": manifest["manifest_hash"],
        "registry_hash": registry.registry_hash,
        "local_preflight_hash": local_preflight["preflight_hash"],
        "runtime": runtime,
        "lifecycle_receipt": dict(lifecycle),
        "command_template_receipt": command_receipt,
        "live_runtime_checked": True,
        "live_model_execution_authorized": False,
        "ready_for_authorization": True,
    }
    preflight = {**payload, "preflight_hash": _canonical_hash(payload)}
    validate_remote_preflight(
        preflight,
        manifest,
        registry,
        local_preflight,
        project_root,
        pi_executable=pi_binary,
    )
    output = Path(output_path).expanduser().resolve()
    _write_immutable_json(output, preflight)
    return {
        "remote_preflight": str(output),
        "preflight_hash": preflight["preflight_hash"],
        "ready_for_authorization": True,
        "live_model_execution_authorized": False,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyreplab-m3-empty-overlay-baseline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--registry-output", required=True)
    freeze.add_argument("--manifest-output", required=True)
    freeze.add_argument("--host", default=os.environ.get("PYREPLAB_HARNESS_HOST", "ubuntu-local"))
    freeze.add_argument("--remote-project", required=True)
    freeze.add_argument("--remote-run-root", required=True)
    freeze.add_argument("--remote-python", default="python3")

    validate = subparsers.add_parser("validate")
    validate.add_argument("--registry", required=True)
    validate.add_argument("--manifest", required=True)

    local = subparsers.add_parser("local-preflight")
    local.add_argument("--registry", required=True)
    local.add_argument("--manifest", required=True)
    local.add_argument("--root", required=True)
    local.add_argument("--output", required=True)
    local.add_argument("--pi", default=os.environ.get("PYREPLAB_PI", "pi"))

    lifecycle = subparsers.add_parser("lifecycle-stress")
    lifecycle.add_argument("--unbrowser-binary", required=True)
    lifecycle.add_argument("--wait-seconds", type=float, default=36.0)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--registry", required=True)
    preflight.add_argument("--manifest", required=True)
    preflight.add_argument("--local-preflight", required=True)
    preflight.add_argument("--root", required=True)
    preflight.add_argument("--output", required=True)
    preflight.add_argument("--host", default=os.environ.get("PYREPLAB_HARNESS_HOST", "ubuntu-local"))
    preflight.add_argument("--remote-project", required=True)
    preflight.add_argument("--remote-run-root", required=True)
    preflight.add_argument("--remote-python", default="python3")
    preflight.add_argument("--pi", default=os.environ.get("PYREPLAB_PI", "pi"))
    preflight.add_argument("--thinking", default=os.environ.get("PYREPLAB_PI_THINKING", "off"))
    preflight.add_argument("--unbrowser-binary", required=True)
    preflight.add_argument("--model-artifact", required=True)
    preflight.add_argument(
        "--llama-server-binary", default="/usr/local/lib/ollama/llama-server"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "freeze":
        report = freeze_baseline_artifacts(
            args.registry_output,
            args.manifest_output,
            {
                "host": args.host,
                "project": args.remote_project,
                "run_root": args.remote_run_root,
                "python": args.remote_python,
            },
        )
    elif args.command == "validate":
        registry = TreatmentRegistry.load(args.registry)
        manifest = _load_json(args.manifest)
        validate_baseline_manifest(manifest, registry)
        report = {
            "valid": True,
            "registry_hash": registry.registry_hash,
            "manifest_hash": manifest["manifest_hash"],
        }
    elif args.command == "local-preflight":
        report = write_local_preflight(
            args.output,
            args.manifest,
            args.registry,
            args.root,
            pi_executable=args.pi,
        )
    elif args.command == "lifecycle-stress":
        report = run_lifecycle_stress(
            args.unbrowser_binary,
            wait_seconds=args.wait_seconds,
        )
    elif args.command == "preflight":
        report = run_remote_preflight(
            args.output,
            args.manifest,
            args.registry,
            args.local_preflight,
            args.root,
            RemoteConfig(
                args.host,
                args.remote_project,
                args.remote_run_root,
                args.remote_python,
            ),
            pi_binary=args.pi,
            thinking=args.thinking,
            unbrowser_binary=args.unbrowser_binary,
            model_artifact=args.model_artifact,
            llama_server_binary=args.llama_server_binary,
        )
    else:  # pragma: no cover - argparse enforces the command set.
        raise AssertionError(f"unsupported command: {args.command}")
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_ATTEMPTS",
    "EXPECTED_TASKS",
    "build_baseline_manifest",
    "build_command_template_receipt",
    "build_empty_overlay_registry",
    "build_local_preflight",
    "freeze_baseline_artifacts",
    "run_lifecycle_stress",
    "run_remote_preflight",
    "validate_baseline_manifest",
    "validate_local_preflight",
    "validate_remote_preflight",
    "write_local_preflight",
]
