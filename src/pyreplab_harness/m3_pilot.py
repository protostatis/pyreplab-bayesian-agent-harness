"""Frozen manifest and sequential runner for the M3 headroom pilot."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from .batch import _append_result
from .meta_grammar import _GRAMMAR_VERSION
from .orchestrator import (
    PINNED_SAMPLING_PARAMETERS,
    RemoteConfig,
    run_registered_treatments,
    validate_remote_config,
)
from .treatments import TreatmentRegistry, TreatmentSpec
from .unbrowser_fixture_gym import (
    FIXTURE_PORT,
    GENERATOR_VERSION,
    VERIFIER_ID,
    VERIFIER_VERSION,
)

SCHEMA_VERSION = "m3-headroom-pilot-v1"
RESULT_SCHEMA_VERSION = "m3-headroom-task-result-v1"

KNOWN_TEMPLATES = (
    "single_page_extraction",
    "table_filter_sort",
    "multi_page_navigation",
    "search_filter_controls",
    "form_entry_validation",
    "distractor_recovery",
)
HELD_TEMPLATES = ("cross_page_comparison", "stateful_workflow")

_POLICY_FACTORS: dict[str, dict[str, str]] = {
    "A": {
        "planning": "direct",
        "observation": "text_first",
        "verification": "submit_directly",
        "recovery": "fail_fast",
        "tool_cap": "lean",
    },
    "B": {
        "planning": "direct",
        "observation": "structure_first",
        "verification": "submit_directly",
        "recovery": "diagnose_retry_once",
        "tool_cap": "expanded",
    },
    "C": {
        "planning": "brief_plan",
        "observation": "structure_first",
        "verification": "final_reobserve",
        "recovery": "fail_fast",
        "tool_cap": "expanded",
    },
    "D": {
        "planning": "decompose",
        "observation": "targeted_query_first",
        "verification": "final_reobserve",
        "recovery": "diagnose_retry_once",
        "tool_cap": "lean",
    },
}

_RUNTIME_PINS: dict[str, Any] = {
    "pi_version": "0.84.1",
    "pi_package": "@earendil-works/pi-coding-agent",
    "pi_cli_sha256": "840d1e8e689ed9e4937bcb00b9a810e02a8567d9afb10a47097f11ca93ea1521",
    "pi_provider_config": {
        "api": "openai-completions",
        "base_url": "http://127.0.0.1:18081/v1",
        "context_window": 65536,
        "max_tokens": 8192,
        "reasoning": False,
        "sampling_params": None,
    },
    "provider": "ubuntu-gemma",
    "model_alias": "gemma-4-26b-a4b",
    "thinking": "off",
    "rollout_replicas": 2,
    "sampling": {
        "seed_scope": "panel-common-across-policies",
        "parameters": PINNED_SAMPLING_PARAMETERS,
    },
    "model_artifact": "gemma-4-26B-A4B-it-UD-IQ4_NL.gguf",
    "model_artifact_path": "/home/zhimin90/llm_models/gemma-4-26B-A4B-it-UD-IQ4_NL.gguf",
    "model_artifact_sha256": "eeb867f279ea5a3d52a0dc15fe8ada677b3328a328530957f3f9a5da93cb10b8",
    "llama_server_version": "version: 1 (b4d6c7d8f)",
    "llama_server_path": "/usr/local/lib/ollama/llama-server",
    "llama_server_sha256": "234b05b2138264f8fb263c3205e85f4c290e8afe5067e280a4f6f90cdac5696b",
    "remote_provider_base_url": "http://127.0.0.1:8081/v1",
    "llama_server_config": {
        "ctx_size": 65536,
        "flash_attention": "on",
        "n_cpu_moe": 16,
        "n_gpu_layers": "all",
        "parallel": 1,
        "reasoning": "on",
        "threads": 8,
    },
    "llama_server_required_args": [
        "--alias gemma-4-26b-a4b",
        "--ctx-size 65536",
        "--flash-attn on",
        "--n-cpu-moe 16",
        "--n-gpu-layers all",
        "--parallel 1",
        "--reasoning on",
        "--threads 8",
    ],
    "unbrowser_version": "0.0.19",
    "unbrowser_path": "/home/zhimin90/Projects/pyreplab_bayesian_agent_harness/.tools/unbrowser-v0.0.19/unbrowser",
    "unbrowser_sha256": "3c9b0b59ee2f7cdc04970db5ccd7fd17374ea1634a8a6d27dea3c0baec77f6c4",
    "bubblewrap_version": "bubblewrap 0.6.1",
    "fixture_generator_version": GENERATOR_VERSION,
    "fixture_verifier_id": VERIFIER_ID,
    "fixture_verifier_version": VERIFIER_VERSION,
}


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _verify_embedded_hash(payload: Mapping[str, Any], field: str) -> None:
    expected = payload.get(field)
    if not isinstance(expected, str):
        raise ValueError(f"missing {field}")
    unhashed = {key: value for key, value in payload.items() if key != field}
    actual = _canonical_hash(unhashed)
    if actual != expected:
        raise ValueError(f"{field} mismatch: expected {expected}, computed {actual}")


def _load_json(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def _pi_provider_identity(config_path: Path, provider: str, model: str) -> dict[str, Any]:
    config = _load_json(config_path)
    providers = config.get("providers")
    provider_config = providers.get(provider) if isinstance(providers, Mapping) else None
    if not isinstance(provider_config, Mapping):
        raise RuntimeError(f"Pi provider is missing from {config_path}: {provider!r}")
    models = provider_config.get("models")
    matches = [
        entry
        for entry in models if isinstance(entry, Mapping) and entry.get("id") == model
    ] if isinstance(models, list) else []
    if len(matches) != 1:
        raise RuntimeError(f"Pi model identity is ambiguous or missing: {provider}/{model}")
    model_config = matches[0]
    return {
        "api": model_config.get("api") or provider_config.get("api"),
        "base_url": provider_config.get("baseUrl"),
        "context_window": model_config.get("contextWindow"),
        "max_tokens": model_config.get("maxTokens"),
        "reasoning": model_config.get("reasoning", False),
        "sampling_params": model_config.get("samplingParams"),
    }


def _contains_argument_sequence(arguments: list[str], required: str) -> bool:
    tokens = shlex.split(required)
    return any(
        arguments[index : index + len(tokens)] == tokens
        for index in range(len(arguments) - len(tokens) + 1)
    )


def _argument_value(arguments: list[str], *flags: str) -> str | None:
    for index, value in enumerate(arguments):
        if value in flags and index + 1 < len(arguments):
            return arguments[index + 1]
        for flag in flags:
            prefix = f"{flag}="
            if value.startswith(prefix):
                return value[len(prefix) :]
    return None


def _model_endpoint_entry(base_url: str, model_alias: str) -> dict[str, Any]:
    try:
        with urlopen(f"{base_url.rstrip('/')}/models", timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"pinned model endpoint is unavailable: {error}") from error
    entries = payload.get("data") if isinstance(payload, Mapping) else None
    matches = [
        dict(entry)
        for entry in entries
        if isinstance(entry, Mapping) and entry.get("id") == model_alias
    ] if isinstance(entries, list) else []
    if len(matches) != 1:
        raise RuntimeError(f"model endpoint omitted unique alias {model_alias!r}")
    return matches[0]


def _find_policy(
    registry: TreatmentRegistry,
    factors: Mapping[str, str],
) -> TreatmentSpec:
    matches = [
        treatment
        for treatment in registry
        if all(treatment.generator_metadata.get(key) == value for key, value in factors.items())
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one policy for factors {dict(factors)!r}, got {len(matches)}")
    return matches[0]


def _factor_distance(first: TreatmentSpec, second: TreatmentSpec) -> int:
    return sum(
        first.generator_metadata.get(factor) != second.generator_metadata.get(factor)
        for factor in ("planning", "observation", "verification", "recovery", "tool_cap")
    )


def _pilot_layout() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    order_groups = (
        (("A", "B", "C", "D"), ("B", "A", "D", "C"), ("C", "D", "A", "B"), ("D", "C", "B", "A")),
        (("A", "B", "D", "C"), ("B", "A", "C", "D"), ("C", "D", "B", "A"), ("D", "C", "A", "B")),
        (("A", "C", "B", "D"), ("B", "D", "A", "C"), ("C", "A", "D", "B"), ("D", "B", "C", "A")),
        (("A", "C", "D", "B"), ("B", "D", "C", "A"), ("C", "A", "B", "D"), ("D", "B", "A", "C")),
        (("A", "D", "B", "C"), ("B", "C", "A", "D"), ("C", "B", "D", "A"), ("D", "A", "C", "B")),
        (("A", "D", "C", "B"), ("B", "C", "D", "A"), ("C", "B", "A", "D"), ("D", "A", "B", "C")),
    )
    difficulty_patterns = (
        ("easy", "medium"),
        ("hard", "easy"),
        ("medium", "hard"),
    )
    tasks: list[dict[str, Any]] = []
    panel_groups: list[list[dict[str, Any]]] = []
    task_index = 0
    for template_index, template in enumerate(KNOWN_TEMPLATES):
        template_tasks: list[dict[str, Any]] = []
        for within_template in range(2):
            difficulty = difficulty_patterns[template_index % 3][within_template]
            seed = 2026081001 + task_index
            task_id = f"unbrowser-fixture-v2-{template}-{difficulty}-{seed}"
            task = {
                "task_id": task_id,
                "role": "T_pilot",
                "template": template,
                "difficulty": difficulty,
                "seed": seed,
            }
            if template == "distractor_recovery":
                task["recovery_probe_url"] = (
                    f"http://127.0.0.1:{FIXTURE_PORT}/{template}/{seed}/"
                    f"{difficulty}/page_0"
                )
                task["recovery_probe_status"] = 503
            tasks.append(task)
            template_tasks.append(task)
            task_index += 1
        group_panels: list[dict[str, Any]] = []
        for slot, order in enumerate(order_groups[template_index]):
            within_template = slot // 2
            task = template_tasks[within_template]
            task_offset = template_index * 2 + within_template
            first_replica = task_offset % 2
            replica = first_replica if slot % 2 == 0 else 1 - first_replica
            group_panels.append(
                {
                    "panel_id": f"{task['task_id']}/replica={replica}",
                    "task_id": task["task_id"],
                    "rollout_replica": replica,
                    "sampling_seed": 2026082001 + task_offset * 2 + replica,
                    "execution_order": list(order),
                }
            )
        panel_groups.append(group_panels)

    # Interleave templates over time while retaining per-template position balance.
    panels = [
        panel_groups[template_index][slot]
        for slot in range(4)
        for template_index in range(len(KNOWN_TEMPLATES))
    ]
    return tasks, panels


def build_headroom_manifest(
    registry: TreatmentRegistry,
    policy_split: Mapping[str, Any],
    *,
    registry_file: str,
    policy_split_file: str,
) -> dict[str, Any]:
    """Build the frozen four-policy, 12-task replicated headroom manifest."""
    _verify_embedded_hash(policy_split, "manifest_hash")
    if policy_split.get("registry_hash") != registry.registry_hash:
        raise ValueError("policy split registry hash does not match registry")
    meta_train = set(policy_split.get("splits", {}).get("meta_train", []))
    selected = {
        label: _find_policy(registry, factors)
        for label, factors in _POLICY_FACTORS.items()
    }
    if not all(treatment.bundle_id in meta_train for treatment in selected.values()):
        raise ValueError("headroom policies must all belong to meta_train")

    distances = [
        _factor_distance(first, second)
        for first, second in itertools.combinations(selected.values(), 2)
    ]
    tasks, panels = _pilot_layout()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "registry_file": registry_file,
        "registry_hash": registry.registry_hash,
        "policy_split_file": policy_split_file,
        "policy_split_manifest_hash": policy_split["manifest_hash"],
        "grammar_version": _GRAMMAR_VERSION,
        "policy_labels": {
            label: treatment.bundle_id for label, treatment in selected.items()
        },
        "selection": {
            "eligible_split": "meta_train",
            "factor_order": [
                "planning",
                "observation",
                "verification",
                "recovery",
                "tool_cap",
            ],
            "pairwise_hamming_distances": distances,
            "minimum_pairwise_distance": min(distances),
            "total_pairwise_distance": sum(distances),
            "tie_break": "uniform-fractional-over-selector-replica-winners",
        },
        "known_templates": list(KNOWN_TEMPLATES),
        "held_templates": list(HELD_TEMPLATES),
        "tasks": tasks,
        "panels": panels,
        "runtime_pins": _RUNTIME_PINS,
        "gates": {
            "attempts": 96,
            "tasks": 12,
            "headroom_tasks": 10,
            "panels": 24,
            "policies_per_task": 4,
            "rollout_replicas": 2,
            "maximum_repeat_discordance_rate": 0.10,
            "minimum_stable_disagreement_tasks": 2,
            "minimum_cross_replica_lift_successes": 1.0,
            "minimum_cost_mean_ratio": 1.20,
            "minimum_planning_adherence": 0.75,
            "minimum_observation_adherence": 0.75,
            "minimum_verification_rate_difference": 0.25,
            "minimum_verification_opportunities_per_level": 8,
            "minimum_recovery_rate_difference": 0.25,
            "required_recovery_eligible_per_level": 8,
            "tool_cap_compliance": 1.0,
        },
        "exclusion": (
            "T_pilot tasks and all attempts are permanently excluded from "
            "meta-training, calibration, development, and final evaluation pools."
        ),
    }
    return {**payload, "manifest_hash": _canonical_hash(payload)}


def validate_headroom_manifest(
    manifest: Mapping[str, Any],
    registry: TreatmentRegistry,
    policy_split: Mapping[str, Any],
) -> None:
    """Fail closed on any drift in the frozen pilot design."""
    _verify_embedded_hash(manifest, "manifest_hash")
    _verify_embedded_hash(policy_split, "manifest_hash")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported headroom manifest schema")
    if manifest.get("registry_hash") != registry.registry_hash:
        raise ValueError("headroom manifest registry hash mismatch")
    if manifest.get("policy_split_manifest_hash") != policy_split.get("manifest_hash"):
        raise ValueError("headroom manifest policy split hash mismatch")
    if manifest.get("runtime_pins") != _RUNTIME_PINS:
        raise ValueError("headroom runtime pins drifted")

    policy_labels = manifest.get("policy_labels")
    if not isinstance(policy_labels, Mapping) or set(policy_labels) != set(_POLICY_FACTORS):
        raise ValueError("headroom manifest must define policy labels A-D")
    meta_train = set(policy_split.get("splits", {}).get("meta_train", []))
    bundle_ids = list(policy_labels.values())
    if len(set(bundle_ids)) != 4 or not set(bundle_ids) <= meta_train:
        raise ValueError("headroom policies must be four unique meta_train bundles")
    for label, factors in _POLICY_FACTORS.items():
        treatment = registry.by_bundle_id(str(policy_labels[label]))
        if any(treatment.generator_metadata.get(key) != value for key, value in factors.items()):
            raise ValueError(f"policy label {label} factor assignment drifted")

    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 12:
        raise ValueError("headroom manifest must contain exactly 12 tasks")
    coordinates = {
        (task.get("template"), task.get("difficulty"), task.get("seed"))
        for task in tasks
        if isinstance(task, Mapping)
    }
    if len(coordinates) != 12:
        raise ValueError("headroom task coordinates must be unique")
    if {task.get("template") for task in tasks} != set(KNOWN_TEMPLATES):
        raise ValueError("headroom task templates do not match known templates")
    if set(HELD_TEMPLATES) & {task.get("template") for task in tasks}:
        raise ValueError("held templates must not enter the headroom pilot")
    if any(
        sum(task.get("template") == template for task in tasks) != 2
        for template in KNOWN_TEMPLATES
    ):
        raise ValueError("each known template must have exactly two pilot tasks")
    difficulty_counts = {
        difficulty: sum(task.get("difficulty") == difficulty for task in tasks)
        for difficulty in ("easy", "medium", "hard")
    }
    if set(difficulty_counts.values()) != {4}:
        raise ValueError(f"headroom difficulties are unbalanced: {difficulty_counts}")
    panels = manifest.get("panels")
    if not isinstance(panels, list) or len(panels) != 24:
        raise ValueError("headroom manifest must contain exactly 24 panels")
    task_ids = {str(task["task_id"]) for task in tasks}
    panel_ids = {str(panel.get("panel_id")) for panel in panels}
    if len(panel_ids) != 24:
        raise ValueError("headroom panel ids must be unique")
    for task_id in task_ids:
        replicas = {
            panel.get("rollout_replica")
            for panel in panels
            if panel.get("task_id") == task_id
        }
        if replicas != {0, 1}:
            raise ValueError(f"task {task_id} must have rollout replicas 0 and 1")
    sampling_seeds = [panel.get("sampling_seed") for panel in panels]
    if (
        len(set(sampling_seeds)) != 24
        or any(
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
            or seed > 2_147_483_647
            for seed in sampling_seeds
        )
    ):
        raise ValueError("headroom panels must have 24 unique valid sampling seeds")
    if {str(panel.get("task_id")) for panel in panels} != task_ids:
        raise ValueError("headroom panels reference unknown or missing tasks")
    observed_orders = [tuple(panel.get("execution_order", [])) for panel in panels]
    expected_orders = set(itertools.permutations(tuple(_POLICY_FACTORS)))
    if set(observed_orders) != expected_orders or len(set(observed_orders)) != 24:
        raise ValueError("headroom execution orders must contain all 24 permutations once")
    task_by_id = {str(task["task_id"]): task for task in tasks}
    first_replica_counts = {0: 0, 1: 0}
    for task_id in task_ids:
        task_panels = [panel for panel in panels if panel["task_id"] == task_id]
        first = min(task_panels, key=panels.index)
        first_replica_counts[int(first["rollout_replica"])] += 1
    if first_replica_counts != {0: 6, 1: 6}:
        raise ValueError(
            f"replica chronology is not counterbalanced: {first_replica_counts}"
        )
    for template in KNOWN_TEMPLATES:
        template_panels = [
            panel
            for panel in panels
            if task_by_id[str(panel["task_id"])]["template"] == template
        ]
        for position in range(4):
            if {
                panel["execution_order"][position] for panel in template_panels
            } != set(_POLICY_FACTORS):
                raise ValueError(
                    f"execution positions are unbalanced within template {template}"
                )
    for task in tasks:
        expected_task_id = (
            f"unbrowser-fixture-v2-{task['template']}-{task['difficulty']}-{task['seed']}"
        )
        if task.get("task_id") != expected_task_id or task.get("role") != "T_pilot":
            raise ValueError(f"invalid frozen task entry: {task!r}")
        expected_probe = (
            f"http://127.0.0.1:{FIXTURE_PORT}/distractor_recovery/"
            f"{task['seed']}/{task['difficulty']}/page_0"
            if task["template"] == "distractor_recovery"
            else None
        )
        if task.get("recovery_probe_url") != expected_probe:
            raise ValueError(f"invalid frozen recovery probe entry: {task!r}")
        expected_probe_status = 503 if task["template"] == "distractor_recovery" else None
        if task.get("recovery_probe_status") != expected_probe_status:
            raise ValueError(f"invalid frozen recovery probe status: {task!r}")


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_headroom_manifest(
    output_path: str | Path,
    registry_path: str | Path,
    policy_split_path: str | Path,
) -> dict[str, Any]:
    registry_file = Path(registry_path).expanduser().resolve()
    split_file = Path(policy_split_path).expanduser().resolve()
    registry = TreatmentRegistry.load(registry_file)
    policy_split = _load_json(split_file)
    manifest = build_headroom_manifest(
        registry,
        policy_split,
        registry_file=registry_file.name,
        policy_split_file=split_file.name,
    )
    validate_headroom_manifest(manifest, registry, policy_split)
    output = Path(output_path).expanduser().resolve()
    _write_immutable_json(output, manifest)
    return {
        "manifest": str(output),
        "manifest_hash": manifest["manifest_hash"],
        "policies": len(manifest["policy_labels"]),
        "tasks": len(manifest["tasks"]),
        "panels": len(manifest["panels"]),
        "attempts": len(manifest["panels"]) * len(manifest["policy_labels"]),
    }


def source_tree_hash(root: str | Path) -> str:
    """Hash executable source and frozen protocol assets by relative path."""
    project = Path(root).expanduser().resolve()
    files: list[Path] = []
    suffixes = {
        "src": {".py"},
        "pi_extensions": {".ts"},
        "policies": {".json", ".md"},
    }
    for directory, allowed_suffixes in suffixes.items():
        base = project / directory
        if base.is_dir():
            files.extend(
                path
                for path in base.rglob("*")
                if path.is_file() and path.suffix in allowed_suffixes
            )
    for name in ("pyproject.toml", "requirements-train.txt"):
        path = project / name
        if path.is_file():
            files.append(path)
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(project).as_posix()):
        relative = path.relative_to(project).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _run_checked(
    command: list[str],
    *,
    timeout: int = 120,
    cwd: Path | None = None,
    stderr_fallback: bool = False,
) -> str:
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
        cwd=cwd,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    stdout = completed.stdout.strip()
    return stdout or (completed.stderr.strip() if stderr_fallback else "")


def _ssh_capture(
    host: str,
    command: list[str],
    *,
    timeout: int = 120,
    stderr_fallback: bool = False,
) -> str:
    return _run_checked(
        [
            "ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            host,
            shlex.join(command),
        ],
        timeout=timeout,
        stderr_fallback=stderr_fallback,
    )


def runtime_preflight(
    project_root: Path,
    config: RemoteConfig,
    *,
    pi_binary: str,
    thinking: str,
    unbrowser_binary: str,
    model_artifact: str,
    llama_server_binary: str,
    require_clean: bool = True,
) -> dict[str, Any]:
    """Verify frozen binaries, source parity, Git state, and fixture port.

    When ``require_clean`` is True (the default for the frozen headroom
    pilot), a dirty worktree is a hard failure.  When False (exploratory
    screens), the preflight records but does not reject a dirty tree.
    """
    validate_remote_config(config)
    dirty_text = _run_checked(
        ["git", "status", "--porcelain"], timeout=30, cwd=project_root
    )
    worktree_clean = not bool(dirty_text.strip())
    _CLEAN_MARKER_HASH = (
        hashlib.sha256(b"PYREPLAB_GIT_WORKTREE_CLEAN_MARKER_V1").hexdigest()
    )
    worktree_status_hash = (
        _CLEAN_MARKER_HASH
        if worktree_clean
        else hashlib.sha256(dirty_text.encode("utf-8")).hexdigest()
    )
    if require_clean and not worktree_clean:
        raise RuntimeError("headroom pilot requires a clean Git worktree")
    code_revision = _run_checked(
        ["git", "rev-parse", "HEAD"], timeout=30, cwd=project_root
    )
    pi_version = _run_checked([pi_binary, "--version"], timeout=30)
    if pi_version != _RUNTIME_PINS["pi_version"]:
        raise RuntimeError(f"Pi version drift: {pi_version!r}")
    resolved_pi = shutil.which(pi_binary)
    if resolved_pi is None:
        candidate = Path(pi_binary).expanduser()
        resolved_pi = str(candidate.resolve()) if candidate.is_file() else None
    if resolved_pi is None:
        raise RuntimeError(f"Pi executable not found: {pi_binary!r}")
    pi_path = Path(resolved_pi).resolve()
    package_marker = f"/node_modules/{_RUNTIME_PINS['pi_package']}/"
    if package_marker not in pi_path.as_posix():
        raise RuntimeError(f"Pi package drift: {pi_path}")
    if _sha256_file(pi_path) != _RUNTIME_PINS["pi_cli_sha256"]:
        raise RuntimeError("Pi CLI digest drift")
    pi_config_dir = Path(
        os.environ.get("PI_CODING_AGENT_DIR", "~/.pi/agent")
    ).expanduser().resolve()
    provider_identity = _pi_provider_identity(
        pi_config_dir / "models.json",
        str(_RUNTIME_PINS["provider"]),
        str(_RUNTIME_PINS["model_alias"]),
    )
    if provider_identity != _RUNTIME_PINS["pi_provider_config"]:
        raise RuntimeError(f"Pi provider configuration drift: {provider_identity!r}")
    if thinking != _RUNTIME_PINS["thinking"]:
        raise RuntimeError(f"thinking mode drift: {thinking!r}")
    if unbrowser_binary != _RUNTIME_PINS["unbrowser_path"]:
        raise RuntimeError("Unbrowser path drift")
    if model_artifact != _RUNTIME_PINS["model_artifact_path"]:
        raise RuntimeError("model artifact path drift")
    if llama_server_binary != _RUNTIME_PINS["llama_server_path"]:
        raise RuntimeError("llama-server path drift")

    remote_python = config.python
    module_command = [
        "env",
        f"PYTHONPATH={config.project}/src",
        remote_python,
        "-m",
        "pyreplab_harness.m3_pilot",
    ]
    remote_source_hash = _ssh_capture(
        config.host,
        [*module_command, "source-hash", "--root", config.project],
    )
    local_source_hash = source_tree_hash(project_root)
    if remote_source_hash != local_source_hash:
        raise RuntimeError("local and remote source-tree hashes differ")

    confined_version = _ssh_capture(
        config.host,
        [*module_command, "confined-unbrowser-check", "--binary", unbrowser_binary],
    )
    if confined_version != f"unbrowser {_RUNTIME_PINS['unbrowser_version']}":
        raise RuntimeError(f"confined Unbrowser version drift: {confined_version!r}")
    unbrowser_hash = _ssh_capture(config.host, ["sha256sum", unbrowser_binary]).split()[0]
    model_hash = _ssh_capture(
        config.host, ["sha256sum", model_artifact], timeout=300
    ).split()[0]
    server_hash = _ssh_capture(
        config.host, ["sha256sum", llama_server_binary]
    ).split()[0]
    server_version = _ssh_capture(
        config.host,
        [llama_server_binary, "--version"],
        stderr_fallback=True,
    ).splitlines()[0]
    bwrap_version = _ssh_capture(config.host, ["bwrap", "--version"])
    _ssh_capture(config.host, [*module_command, "fixture-port-check"])
    expected = _RUNTIME_PINS
    if unbrowser_hash != expected["unbrowser_sha256"]:
        raise RuntimeError("Unbrowser digest drift")
    if model_hash != expected["model_artifact_sha256"]:
        raise RuntimeError("model artifact digest drift")
    if server_hash != expected["llama_server_sha256"]:
        raise RuntimeError("llama-server digest drift")
    if server_version != expected["llama_server_version"]:
        raise RuntimeError("llama-server version drift")
    endpoint_model = _model_endpoint_entry(
        str(expected["pi_provider_config"]["base_url"]),
        str(expected["model_alias"]),
    )
    remote_endpoint_model = json.loads(
        _ssh_capture(
            config.host,
            [
                *module_command,
                "model-endpoint-entry",
                "--base-url",
                str(expected["remote_provider_base_url"]),
                "--model-alias",
                str(expected["model_alias"]),
            ],
        )
    )
    status = endpoint_model.get("status")
    server_arguments = status.get("args") if isinstance(status, Mapping) else None
    remote_status = remote_endpoint_model.get("status")
    remote_arguments = (
        remote_status.get("args") if isinstance(remote_status, Mapping) else None
    )
    if (
        not isinstance(server_arguments, list)
        or any(not isinstance(value, str) for value in server_arguments)
        or server_arguments != remote_arguments
        or not server_arguments
        or server_arguments[0] != llama_server_binary
        or _argument_value(server_arguments, "--model", "-m") != model_artifact
        or any(
            not _contains_argument_sequence(server_arguments, required)
            for required in expected["llama_server_required_args"]
        )
    ):
        raise RuntimeError("llama-server endpoint configuration drift")
    if bwrap_version != expected["bubblewrap_version"]:
        raise RuntimeError("Bubblewrap version drift")
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": code_revision,
        "source_tree_hash": local_source_hash,
        "worktree_clean": worktree_clean,
        "worktree_status_hash": worktree_status_hash,
        "runtime_pins": expected,
    }


def _existing_result_keys(path: Path, manifest_hash: str) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid pilot JSONL line {line_number}: {error}") from error
        if record.get("pilot_manifest_hash") != manifest_hash:
            raise ValueError("pilot output mixes manifest hashes")
        if record.get("status") == "error":
            raise RuntimeError("existing pilot infrastructure error requires adjudication")
        if record.get("status") == "completed":
            completed.add(str(record.get("key")))
    return completed


def run_headroom_pilot(
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
    """Run the frozen matrix sequentially and stop on infrastructure error."""
    project_root = Path(__file__).resolve().parents[2]
    registry = TreatmentRegistry.load(registry_path)
    policy_split = _load_json(policy_split_path)
    manifest = _load_json(manifest_path)
    validate_headroom_manifest(manifest, registry, policy_split)
    if provider != _RUNTIME_PINS["provider"] or model != _RUNTIME_PINS["model_alias"]:
        raise ValueError("provider/model do not match the frozen runtime pins")
    output = Path(output_path).expanduser().resolve()
    active_path = output.with_suffix(output.suffix + ".active.json")
    if active_path.exists():
        raise RuntimeError(
            "unfinished pilot panel marker requires a fresh output/run root"
        )
    runtime = runtime_preflight(
        project_root,
        config,
        pi_binary=pi_binary,
        thinking=thinking,
        unbrowser_binary=unbrowser_binary,
        model_artifact=model_artifact,
        llama_server_binary=llama_server_binary,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    preflight_path = output.with_suffix(output.suffix + ".preflight.json")
    preflight_payload = {
        "pilot_manifest_hash": manifest["manifest_hash"],
        **runtime,
    }
    if preflight_path.exists():
        existing_preflight = _load_json(preflight_path)
        stable_keys = (
            "pilot_manifest_hash",
            "code_revision",
            "source_tree_hash",
            "runtime_pins",
        )
        if any(
            existing_preflight.get(key) != preflight_payload.get(key)
            for key in stable_keys
        ):
            raise RuntimeError("pilot preflight identity changed across resume")
    else:
        _write_immutable_json(preflight_path, preflight_payload)
    completed = _existing_result_keys(output, str(manifest["manifest_hash"]))
    policy_labels = manifest["policy_labels"]
    task_by_id = {task["task_id"]: task for task in manifest["tasks"]}
    ran = skipped = 0
    for panel in manifest["panels"]:
        task = task_by_id[panel["task_id"]]
        key = str(panel["panel_id"])
        if key in completed:
            skipped += 1
            continue
        ordered_refs = [policy_labels[label] for label in panel["execution_order"]]
        args = argparse.Namespace(
            family="unbrowser_fixture",
            seed=int(task["seed"]),
            difficulty=str(task["difficulty"]),
            fixture_template=str(task["template"]),
            task_role="T_pilot",
            rollout_replica=int(panel["rollout_replica"]),
            sampling_seed=int(panel["sampling_seed"]),
            pilot_manifest_hash=str(manifest["manifest_hash"]),
            pilot_panel_id=key,
            treatment_registry=str(Path(registry_path).expanduser().resolve()),
            treatments=",".join(ordered_refs),
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
            "schema_version": RESULT_SCHEMA_VERSION,
            "key": key,
            "pilot_manifest_hash": manifest["manifest_hash"],
            "task": task,
            "panel": panel,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_immutable_json(
            active_path,
            {
                "pilot_manifest_hash": manifest["manifest_hash"],
                "panel_id": key,
                "started_at": record["started_at"],
            },
        )
        try:
            result = run_registered_treatments(project_root, config, args)
        except Exception as error:
            record.update(
                {
                    "status": "error",
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "duration_seconds": round(time.monotonic() - started, 3),
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
            )
            _append_result(output, record)
            raise RuntimeError(f"pilot stopped after infrastructure error on {key}") from error
        record.update(
            {
                "status": "completed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": round(time.monotonic() - started, 3),
                "result": result,
            }
        )
        _append_result(output, record)
        active_path.unlink()
        ran += 1
    return {
        "manifest_hash": manifest["manifest_hash"],
        "tasks_total": len(manifest["tasks"]),
        "panels_total": len(manifest["panels"]),
        "panels_run": ran,
        "panels_skipped": skipped,
        "output": str(output),
        "preflight": str(preflight_path),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyreplab-m3-pilot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--registry", required=True)
    freeze.add_argument("--policy-split", required=True)
    freeze.add_argument("--output", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--registry", required=True)
    validate.add_argument("--policy-split", required=True)

    source_hash = subparsers.add_parser("source-hash")
    source_hash.add_argument("--root", required=True)

    confined = subparsers.add_parser("confined-unbrowser-check")
    confined.add_argument("--binary", required=True)

    subparsers.add_parser("fixture-port-check")

    endpoint = subparsers.add_parser("model-endpoint-entry")
    endpoint.add_argument("--base-url", required=True)
    endpoint.add_argument("--model-alias", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--root", required=True)
    preflight.add_argument("--host", default=os.environ.get("PYREPLAB_HARNESS_HOST", "ubuntu-local"))
    preflight.add_argument("--remote-project", required=True)
    preflight.add_argument("--remote-run-root", required=True)
    preflight.add_argument("--remote-python", default="python3")
    preflight.add_argument("--pi", default=os.environ.get("PYREPLAB_PI", "pi"))
    preflight.add_argument("--thinking", default=os.environ.get("PYREPLAB_PI_THINKING", "off"))
    preflight.add_argument("--unbrowser-binary", required=True)
    preflight.add_argument("--model-artifact", required=True)
    preflight.add_argument("--llama-server-binary", default="/usr/local/lib/ollama/llama-server")

    run = subparsers.add_parser("run")
    run.add_argument("--manifest", required=True)
    run.add_argument("--registry", required=True)
    run.add_argument("--policy-split", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--host", default=os.environ.get("PYREPLAB_HARNESS_HOST", "ubuntu-local"))
    run.add_argument("--remote-project", required=True)
    run.add_argument("--remote-run-root", required=True)
    run.add_argument("--remote-python", default="python3")
    run.add_argument("--pi", default=os.environ.get("PYREPLAB_PI", "pi"))
    run.add_argument("--provider", default=os.environ.get("PYREPLAB_PI_PROVIDER", "ubuntu-gemma"))
    run.add_argument("--model", default=os.environ.get("PYREPLAB_PI_MODEL", "gemma-4-26b-a4b"))
    run.add_argument("--thinking", default=os.environ.get("PYREPLAB_PI_THINKING", "off"))
    run.add_argument("--unbrowser-binary", required=True)
    run.add_argument("--model-artifact", required=True)
    run.add_argument("--llama-server-binary", default="/usr/local/lib/ollama/llama-server")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "freeze":
        report = freeze_headroom_manifest(args.output, args.registry, args.policy_split)
    elif args.command == "validate":
        registry = TreatmentRegistry.load(args.registry)
        policy_split = _load_json(args.policy_split)
        manifest = _load_json(args.manifest)
        validate_headroom_manifest(manifest, registry, policy_split)
        report = {"valid": True, "manifest_hash": manifest["manifest_hash"]}
    elif args.command == "source-hash":
        print(source_tree_hash(args.root))
        return 0
    elif args.command == "confined-unbrowser-check":
        from .unbrowser_sandbox import UnbrowserSandbox

        print(_run_checked(UnbrowserSandbox(args.binary).build_command("--version")))
        return 0
    elif args.command == "fixture-port-check":
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", FIXTURE_PORT))
        print("available")
        return 0
    elif args.command == "model-endpoint-entry":
        print(json.dumps(_model_endpoint_entry(args.base_url, args.model_alias), sort_keys=True))
        return 0
    elif args.command == "preflight":
        report = runtime_preflight(
            Path(args.root).expanduser().resolve(),
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
    else:
        report = run_headroom_pilot(
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
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "HELD_TEMPLATES",
    "KNOWN_TEMPLATES",
    "build_headroom_manifest",
    "freeze_headroom_manifest",
    "run_headroom_pilot",
    "runtime_preflight",
    "source_tree_hash",
    "validate_headroom_manifest",
]
