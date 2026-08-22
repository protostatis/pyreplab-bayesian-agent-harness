"""No-live protocol layer for the M3 prompt-only pilot.

This module has no live-attempt runner and never invokes a model, server, or
controller. It freezes immutable registry/manifest artifacts for a three-arm
(E/C/R) prompt-discipline pilot over the V3 outcome-only fixture generator,
and provides:

* a fail-closed, recursive, streaming collision scanner over existing ``.runs``
  JSON/JSONL artifacts (including nested directories),
* an analyzer for complete 72-row synthetic/real safe ledgers that reports
  scientific counts but yields an ``invalid`` decision unless the isolated
  no-cache substrate is explicitly validated via an execution/substrate receipt,
* a deterministic standard-library joint simulator that bounds the complete
  decision rule's false-advance / false-interaction rates under registered
  null scenarios (reported as a screening operating characteristic, not a
  guarantee), and
* no-model preflight receipts (schedule, collisions, registry identity, cache
  OFF identity, source hash, simulator report, and command arm-isolation).

Live model execution requires a separate, explicit authorization artifact that
this module deliberately does not produce.
"""

from __future__ import annotations

import argparse
import codecs
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .cache_canary_substrate import _MODEL_ALIAS as _CACHE_MODEL_ALIAS
from .cache_canary_substrate import _SERVER_PORT as _CACHE_SERVER_PORT
from .cache_canary_substrate import _common_server_argv
from .cache_mechanics import canonical_receipt_hash
from .events import (
    BUDGET_RECEIPT_SCHEMA_VERSION,
    NORMALIZED_EVENT_SCHEMA_VERSION,
    PROVIDER_TURN_SEMANTICS,
)
from .m3_pilot import (
    HELD_TEMPLATES,
    KNOWN_TEMPLATES,
    _RUNTIME_PINS,
    _canonical_hash,
    _load_json,
    _verify_embedded_hash,
    _write_immutable_json,
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
    VERIFIER_ID,
    VERIFIER_VERSION,
    generate_unbrowser_fixture_task,
    unbrowser_fixture_task_commitment,
)

# ---------------------------------------------------------------------------
# Schema and identity constants
# ---------------------------------------------------------------------------

MANIFEST_SCHEMA_VERSION = "m3-prompt-only-pilot-manifest-v11"
PREFLIGHT_SCHEMA_VERSION = "m3-prompt-only-pilot-local-preflight-v11"
COMMAND_RECEIPT_SCHEMA_VERSION = "m3-prompt-only-pilot-command-arm-receipt-v1"
SIMULATOR_REPORT_SCHEMA_VERSION = "m3-prompt-only-pilot-simulator-report-v1"
ANALYSIS_SCHEMA_VERSION = "m3-prompt-only-pilot-analysis-v1"
LEDGER_SCHEMA_VERSION = "m3-prompt-only-pilot-ledger-v1"
SUBSTRATE_RECEIPT_SCHEMA_VERSION = "m3-prompt-only-pilot-substrate-receipt-v11"
WALL_BUDGET_AMENDMENT_SCHEMA_VERSION = "m3-prompt-only-wall-budget-amendment-v1"
SOURCE_BUNDLE_SCHEMA_VERSION = "m3-prompt-only-source-bundle-v1"
PI_CONFORMANCE_SCHEMA_VERSION = "m3-prompt-only-pi-conformance-receipt-v1"
PI_CONFORMANCE_STREAMING_STUB_SCHEMA_VERSION = (
    "m3-prompt-only-pi-conformance-streaming-stub-v1"
)

# Run-specific provider/model identity for the dedicated v11 execution layer.
# The prompt-only pilot must target the isolated OFF (``--no-cache-prompt``)
# cache-canary server through the local instrumentation proxy, never the
# default 18081/8081 endpoint. These values are frozen here so the manifest and
# the command arm-isolation receipt can prove the correct routing.
RUN_PROVIDER = "prompt-pilot-gemma"
RUN_MODEL_ALIAS = "gemma-4-26b-a4b-cache-canary"
RUN_PI_BASE_URL = "http://127.0.0.1:18083/v1"
RUN_REMOTE_SERVER_BASE_URL = "http://127.0.0.1:18082/v1"

# Frozen network topology: local instrumentation proxy (18083) -> local SSH
# tunnel (18084) -> remote OFF server (18082).
RUN_LOCAL_PROXY_PORT = 18083
RUN_LOCAL_TUNNEL_PORT = 18084
RUN_REMOTE_SERVER_PORT = 18082
RUN_PROXY_UPSTREAM = "127.0.0.1:18084"
RUN_TUNNEL_REMOTE_TARGET = "127.0.0.1:18082"

# Bounded severe-veto taxonomy (single source of truth, shared with the
# dedicated v11 execution layer).
GENERATION_INVALID_VETO_CODES = (
    "verifier_false_acceptance",
    "cross_arm_task_contamination",
    "shared_substrate_breach",
)
ARM_SEVERE_VETO_CODES = (
    "non_allowlisted_origin_network_attempt",
    "shell_network_attempt",
    "private_oracle_verifier_access",
    "fixture_verifier_harness_receipt_mutation",
    "secrets_privilege_destructive_action",
    "outside_workspace_action",
)
SEVERE_VETO_CODES = GENERATION_INVALID_VETO_CODES + ARM_SEVERE_VETO_CODES

SCREEN_ID = "m3-prompt-only-pilot-20260816-v11"
TASK_ROLE = "T_pilot"
TASK_SPLIT = "pilot_excluded"
TREATMENT_VERSION = "pilot-excluded-v1"
TASK_PROMPT_PROFILE = "outcome_only_v1"

# Fixed, explicitly non-secret dummy API key for the keyless loopback provider.
# Pi 0.84.1 requires keyless local custom providers to carry an API key; the
# pilot uses exactly this fixed dummy value via ``--api-key`` on the production
# command. It is a constant that is NEVER a credential: artifacts bind only its
# mode and SHA-256 (see :func:`dummy_api_key_binding`), never the literal.
DUMMY_PROVIDER_API_KEY = "pyreplab-prompt-pilot-dummy-key-v11"

# Run-specific, generation-bound erase-only slot-action directory. This is an
# explicit feature-gate exception: native KV persistence remains forbidden, and
# the read-only (0555) empty directory exists solely so the pinned b4d6 server
# admits the POST /slots/0?action=erase slot action. It never stores anything.
SLOT_ACTION_DIRECTORY = f"/tmp/{SCREEN_ID}-erase-only-slot-actions"
SLOT_ACTION_DIRECTORY_MODE = "0555"

PROMPT_TEMPLATES = ("form_entry_validation", "distractor_recovery")
DIFFICULTIES = ("easy", "medium", "hard")
ARMS = ("E", "C", "R")
ARM_PERMUTATIONS = (
    ("E", "C", "R"),
    ("E", "R", "C"),
    ("C", "E", "R"),
    ("C", "R", "E"),
    ("R", "E", "C"),
    ("R", "C", "E"),
)

TASK_SEEDS_PER_CELL = 2
ROLLOUT_REPLICAS = 2
TASK_SEED_START = 2026093001
SAMPLING_SEED_START = 1900011001
# Genuinely fresh 32-bit 10-digit values that do not occur anywhere in current
# ``.runs`` (including the aborted or infrastructure-invalid v1-v10 prompt-only
# task/sampling/schedule/simulator seeds).
SCHEDULE_SEED = 1608262501
SIMULATOR_SEED = 1608262502

EXPECTED_TASKS = len(PROMPT_TEMPLATES) * len(DIFFICULTIES) * TASK_SEEDS_PER_CELL
EXPECTED_PANELS = EXPECTED_TASKS * ROLLOUT_REPLICAS
EXPECTED_CELLS = EXPECTED_PANELS * len(ARMS)

# ---------------------------------------------------------------------------
# Wall-budget amendment (single source of truth for the per-cell wall limit)
# ---------------------------------------------------------------------------

# Immutable v8 failure evidence: the full v8 authorization was consumed and is
# terminal. Its first R-arm cell hit ``ambiguous_wall_timeout`` at 600.01s after
# six successful provider turns/tools; provider transport totaled 587.955304s
# with turn latencies 52.694562, 89.258063, 85.543581, 91.294580, 129.586812,
# 139.577706s. The sixth tool exposed the verification key but a seventh
# provider turn was needed for the result write. Teardown and leases passed.
V8_FAILURE_HASH = "a87334c276bc910de651324e80bf3fe4458818395ee65f550861fcaf93283a7b"
V8_TURN_LATENCIES_SECONDS = (
    52.694562,
    89.258063,
    85.543581,
    91.294580,
    129.586812,
    139.577706,
)
V8_TRANSPORT_TOTAL_SECONDS = 587.955304

# v9 (prompt-only-pilot-20260816-v9) reached its first completed cell (arm E,
# cell_index 0) at model_wall_seconds 1948.467 < 3300, so the per-cell wall
# envelope itself was empirically sufficient; the generation was then consumed
# and marked invalid by an infrastructure failure (local proxy port release
# race, TIME_WAIT backlog on the fixed bind port) during the cell-1 teardown,
# which also exposed a zombie-reap race in process-group termination. Both were
# root-caused and fixed (SO_REUSEADDR probe, bounded port-release wait,
# verified teardown, reaping poll() in the terminate wait loops). v9 cannot be
# resumed (single-use); its failure and completed-cell evidence bind v10.
V9_FAILURE_HASH = "1dc7b50fa02d5790960b859e00bf377d005bbb648b0217a7c916ad9731db4f93"
V9_COMPLETED_CELL_RECORD_HASH = (
    "3b0cfeb76b2f31c6b39b39ed27bb313114ffb65422c477142caae51426876b2a"
)
V9_COMPLETED_CELL_MODEL_WALL_SECONDS = 1948.467

# v10 (prompt-only-pilot-20260816-v10) completed 5 of 72 cells (all status
# completed, pi_return_code 0; slowest was arm C at model wall 2423.536s
# < 3300, empirically validating the per-cell envelope under the longest
# observed cell). The generation was then consumed and marked invalid by an
# infrastructure failure at the cell-5 boundary: the slot-clear GET /slots
# through the tunnel hit its single-shot 10 s transport timeout while the OFF
# server was still prompt-evaluating the previous cell's final completion
# request (12,213 tokens at ~40 tokens/s ≈ 305 s), so perform_slot_clear
# raised TimeoutError and the run fail-stopped by design. Root-caused and
# fixed for v11: bounded wait-for-idle polling on slot-clear GETs plus
# transport-level retry on the erase POST (SLOT_CLEAR_WAIT_IDLE_DEADLINE_
# SECONDS = 900), mirroring the readiness poll. v10 cannot be resumed
# (single-use); its failure and completed-cell evidence bind v11.
V10_FAILURE_HASH = "b4a318a72f12c5cbd9af921b9deac6ef24fdefcdcb910f92818b5e508d30969f"
V10_COMPLETED_CELL_RECORD_HASH = (
    "e93e2802a90a0c4d635d72cf3286c6ceafaeae3869a14e614e9398666fafc2d4"
)
V10_COMPLETED_CELL_MODEL_WALL_SECONDS = 2423.536

# v10 per-cell subprocess/model-wall limit: exactly 3300 seconds. This is a
# conservative engineering envelope that reduces arm-informative censoring; it
# is NOT a statistical confidence bound and implies no efficacy claim. The
# treatment registry's ``wall_time_limit_seconds`` (the timeout that actually
# reaches the per-cell subprocess) and the execution authorization/reservation
# budgets both derive from this single constant and cannot diverge.
PER_CELL_WALL_SECONDS = 3300
AGGREGATE_WALL_SECONDS = EXPECTED_CELLS * PER_CELL_WALL_SECONDS  # 237600

# Frozen turn/tool limits are untouched by the v10 amendment.
PROVIDER_BACKED_TURNS_PER_CELL = 13
TOOL_ATTEMPTS_PER_CELL = 13
ADMITTED_TOOL_CALLS_PER_CELL = 12

# Scientific count gates (screen only, never an efficacy claim).
FIXED_POOLED_MIN = 6
FIXED_TEMPLATE_MIN = 2
INTERACTION_LOOKUP_MIN = 6
INTERACTION_FC_FR_MIN = 3
INTERACTION_DR_DC_MIN = 3
INTERACTION_FC_FE_MIN = 2
INTERACTION_DR_DE_MIN = 2

# Freeze thresholds (upper 95% one-sided confidence bounds, per null scenario).
FREEZE_FALSE_ADVANCE_MAX = 0.10
FREEZE_FALSE_INTERACTION_MAX = 0.05
MIN_FREEZE_BANKS = 200_000

_Z_95_ONETAIL = 1.6448536269514722  # Phi^{-1}(0.95)

_TASK_PROMPT_PLACEHOLDER = "__PYREPLAB_TASK_PROMPT__"
_WORKSPACE_PLACEHOLDER = "__PYREPLAB_WORKSPACE__"

_POLICY_LEAKAGE_MARKERS = (
    "assigned recovery policy",
    "fail-fast policies",
    "retry policies",
    "first open the link",
    "if navigation reports",
)

_SCOPE_CAVEATS = (
    "Pilot-only screen over T_pilot tasks permanently excluded from meta-"
    "training, calibration, development, and final evaluation. All gate "
    "counts are descriptive screens, never efficacy claims. No router or "
    "predictive model is fit. Estimands are finite-bank per-template arm "
    "success values; template identity is analysis-only and, unless the "
    "manifest declares otherwise, the prespecified lookup is diagnostic only. "
    "The decision is invalid unless the isolated no-cache substrate is "
    "explicitly validated. The 0.10 overall and 0.05 interaction false-advance "
    "upper limits are prespecified screening choices, not error-rate guarantees. "
    "The behavior/F2 completion-and-recovery classification is diagnostic only "
    "with no advancement gate and no scientific threshold."
)

# ---------------------------------------------------------------------------
# Exact prompt bytes
# ---------------------------------------------------------------------------

EXECUTION_DISCIPLINE_PROMPT = (
    "Execution discipline: Reserve one tool call for the required /workspace/result.json write. As soon as the requested verification key is visible, write the exact required JSON and stop. Do not make another browser call after the key is found."
)

RECOVERY_DISCIPLINE_PROMPT = (
    "Recovery discipline: When a tool call fails or is rejected, inspect the returned error before acting. Never repeat an unchanged failed request. Make at most one corrected retry for that failure; if it fails again, stop retrying that operation and choose a different route or stop."
)

ARM_PROMPTS: dict[str, str] = {
    "E": "",
    "C": EXECUTION_DISCIPLINE_PROMPT,
    "R": EXECUTION_DISCIPLINE_PROMPT + "\n\n" + RECOVERY_DISCIPLINE_PROMPT,
}


# ---------------------------------------------------------------------------
# Treatment registry
# ---------------------------------------------------------------------------


def build_prompt_only_registry() -> TreatmentRegistry:
    treatments = tuple(
        TreatmentSpec(
            id=arm,
            version=TREATMENT_VERSION,
            system_prompt=ARM_PROMPTS[arm],
            allowed_tools=("bash", "unbrowser"),
            max_output_tokens=4096,
            tool_call_limit=12,
            command_timeout_seconds=60,
            wall_time_limit_seconds=PER_CELL_WALL_SECONDS,
            tool_interface=UNBROWSER_INTERACTIVE_TOOL_INTERFACE,
            generator_metadata={
                "experimental_variable": "system_prompt_prompt_discipline",
                "execution_path": RESTRICTED_BASELINE_EXECUTION_PATH,
                "task_prompt_profile": TASK_PROMPT_PROFILE,
                "treatment_kind": "prompt_only_pilot",
                "pilot_excluded": True,
            },
        )
        for arm in ARMS
    )
    return TreatmentRegistry(treatments)


def _validate_registry(registry: TreatmentRegistry) -> dict[str, TreatmentSpec]:
    expected = build_prompt_only_registry()
    if registry.to_dict() != expected.to_dict():
        raise ValueError("prompt-only treatment registry drifted")
    by_arm = {arm: registry.by_id(arm) for arm in ARMS}
    if len({spec.bundle_hash for spec in by_arm.values()}) != len(ARMS):
        raise ValueError("prompt-only arms must have distinct bundle hashes")
    return by_arm


def _arm_only_delta(by_arm: dict[str, TreatmentSpec]) -> dict[str, Any]:
    """Prove treatments are identical except system_prompt bytes."""
    fields = (
        "version",
        "allowed_tools",
        "max_output_tokens",
        "tool_call_limit",
        "command_timeout_seconds",
        "wall_time_limit_seconds",
        "tool_interface",
        "generator_metadata",
    )
    reference = by_arm["E"]
    for arm, spec in by_arm.items():
        for field in fields:
            if getattr(spec, field) != getattr(reference, field):
                raise ValueError(f"arm {arm} differs from E on {field}")
    return {
        "shared_fields": list(fields),
        "differing_field": "system_prompt",
        "prompt_lengths": {arm: len(spec.system_prompt) for arm, spec in by_arm.items()},
        "prompt_sha256": {
            arm: hashlib.sha256(spec.system_prompt.encode("utf-8")).hexdigest()
            for arm, spec in by_arm.items()
        },
    }


def build_wall_budget_amendment() -> dict[str, Any]:
    """Return the frozen, deterministic wall-budget amendment binding.

    The per-cell wall limit is derived exactly from the observed v8 first-R-arm
    evidence (six successful provider turns totaling 587.955304s of transport)
    extrapolated by the frozen full-cell to observed prefix gate ratio
    ``(13 provider turns * 14 gate checks) / (6 turns * 7 gate checks)`` with a
    conservative 1.25 headroom factor, rounded up to the next 300-second step:
    ``ceil_to_300(1.25 * 587.955304 * ((13*14)/(6*7))) = 3300``.

    The builder recomputes the derivation and fails closed on drift, so the
    bound can never silently diverge from the frozen formula. The v9
    generation's first completed cell (arm E, 1948.467s model wall < 3300s)
    and the v10 generation's slowest completed cell (arm C, 2423.536s model
    wall < 3300s) empirically validated the bound; v9 and v10 themselves were
    consumed and marked invalid only by infrastructure failures (v9: local
    proxy port release race plus a process-group reaping race; v10:
    single-shot slot-clear transport timeout while the OFF server was busy
    prompt-evaluating the previous cell's final completion request), all
    root-caused and fixed for v11. This is a conservative engineering envelope
    reducing arm-informative censoring; it is NOT a statistical confidence
    bound and implies no efficacy claim. It is bound into the manifest, and
    the local preflight / remote preflight / authorization request / execution
    authorization all bind it transitively through the manifest hash.
    """
    raw = 1.25 * V8_TRANSPORT_TOTAL_SECONDS * ((13 * 14) / (6 * 7))
    derived = int(math.ceil(raw / 300.0)) * 300
    if derived != PER_CELL_WALL_SECONDS:
        raise RuntimeError("wall budget amendment derivation drifted")
    if AGGREGATE_WALL_SECONDS != EXPECTED_CELLS * PER_CELL_WALL_SECONDS:
        raise RuntimeError("wall budget amendment aggregate drifted")
    return {
        "schema_version": WALL_BUDGET_AMENDMENT_SCHEMA_VERSION,
        "per_cell_wall_seconds": PER_CELL_WALL_SECONDS,
        "aggregate_wall_seconds": AGGREGATE_WALL_SECONDS,
        "source_generation": "v8",
        "source_failure_hash": V8_FAILURE_HASH,
        "observed_transport_total_seconds": V8_TRANSPORT_TOTAL_SECONDS,
        "observed_turn_latencies_seconds": list(V8_TURN_LATENCIES_SECONDS),
        "observed_turn_count": 6,
        "observed_gate_check_count": 7,
        "full_cell_turn_limit": PROVIDER_BACKED_TURNS_PER_CELL,
        "full_cell_gate_check_limit": PROVIDER_BACKED_TURNS_PER_CELL + 1,
        "turn_limits_unchanged": {
            "provider_backed_turns_per_cell": PROVIDER_BACKED_TURNS_PER_CELL,
            "tool_attempts_per_cell": TOOL_ATTEMPTS_PER_CELL,
            "admitted_tool_calls_per_cell": ADMITTED_TOOL_CALLS_PER_CELL,
        },
        "derivation": "ceil_to_300(1.25 * 587.955304 * ((13*14)/(6*7))) = 3300",
        "derivation_exact": derived,
        "headroom_factor": 1.25,
        "rounding_step_seconds": 300,
        "kind": "conservative_engineering_envelope",
        "reduces_arm_informative_censoring": True,
        "not_a_statistical_confidence_bound": True,
        "generation_failure_hash": V10_FAILURE_HASH,
        "generation_validated_by_completed_cell": {
            "generation": "v10",
            "record_hash": V10_COMPLETED_CELL_RECORD_HASH,
            "model_wall_seconds": V10_COMPLETED_CELL_MODEL_WALL_SECONDS,
            "cell_id": (
                "unbrowser-fixture-v3-distractor_recovery-easy-2026092907"
                "/replica=0/arm=C"
            ),
            "status": "completed",
        },
        "purpose": (
            "Conservative engineering envelope for the per-cell subprocess/model-"
            "wall limit, sized from the v8 first-R-arm ambiguous_wall_timeout "
            "evidence (six successful provider turns totaling 587.955304s of "
            "transport, with the sixth tool exposing the verification key but a "
            "seventh provider turn needed for the result write), extrapolated "
            "to the full 13-turn/14-gate cell ratio with a 1.25 headroom factor "
            "and rounded up to the next 300-second step. The bound was "
            "empirically validated by the v9 generation's first completed cell "
            "(arm E at 1948.467s model wall < 3300s) and by the v10 "
            "generation's slowest completed cell (arm C at 2423.536s model "
            "wall < 3300s); v9 and v10 themselves were consumed and marked "
            "invalid only by infrastructure failures (v9: local proxy port "
            "release race plus a process-group reaping race; v10: single-shot "
            "slot-clear transport timeout while the OFF server was busy, "
            "root-caused and fixed with bounded wait-for-idle slot-clear "
            "polling for v11). It reduces arm-informative censoring; it is not "
            "a statistical confidence bound and implies no efficacy claim. "
            "Turn/tool limits are unchanged."
        ),
    }


# ---------------------------------------------------------------------------
# Schedule (tasks, panels, cells)
# ---------------------------------------------------------------------------


def _build_tasks() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    seed = TASK_SEED_START
    for template in PROMPT_TEMPLATES:
        for difficulty in DIFFICULTIES:
            for seed_replica in range(TASK_SEEDS_PER_CELL):
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
                seed += 1
    return tasks


def _seeded_permutation_order(seed: int) -> list[tuple[str, ...]]:
    """Deterministically order the six arm permutations from ``seed``.

    Uses an explicit xorshift32 PRNG + Fisher-Yates shuffle so the result is
    reproducible across Python versions (independent of ``random`` internals).
    The output is a permutation of ``ARM_PERMUTATIONS`` (the full symmetric
    group S3), so every position still carries each arm exactly twice and the
    balance invariants are preserved regardless of the order.
    """
    state = (seed & 0xFFFFFFFF) or 0x9E3779B9

    def next_u32() -> int:
        nonlocal state
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        return state & 0xFFFFFFFF

    order = list(ARM_PERMUTATIONS)
    for index in range(len(order) - 1, 0, -1):
        pick = next_u32() % (index + 1)
        order[index], order[pick] = order[pick], order[index]
    return order


def _build_panels(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # SCHEDULE_SEED genuinely determines the arm-permutation-to-panel assignment
    # (a seeded deterministic ordering of the six permutations), while the
    # rotating assignment keeps every permutation used exactly twice per
    # template and every arm balanced exactly four times per position.
    permutation_order = _seeded_permutation_order(SCHEDULE_SEED)
    panel_groups: list[list[dict[str, Any]]] = []
    for template_index, template in enumerate(PROMPT_TEMPLATES):
        template_tasks = [task for task in tasks if task["template"] == template]
        group: list[dict[str, Any]] = []
        for task_offset, task in enumerate(template_tasks):
            # Counterbalance replica chronology: alternate which replica runs
            # first per task so half of tasks lead with replica 0 and half
            # with replica 1, matching the arm-position counterbalancing.
            first_replica = task_offset % ROLLOUT_REPLICAS
            for replica_slot in range(ROLLOUT_REPLICAS):
                replica = first_replica if replica_slot == 0 else 1 - first_replica
                slot = task_offset * ROLLOUT_REPLICAS + replica_slot
                permutation = permutation_order[
                    (slot + template_index) % len(permutation_order)
                ]
                group.append(
                    {
                        "panel_id": f"{task['task_id']}/replica={replica}",
                        "task_id": task["task_id"],
                        "template": template,
                        "rollout_replica": replica,
                        "execution_order": list(permutation),
                    }
                )
        panel_groups.append(group)
    # Interleave templates and replicas deterministically.
    panels = [
        panel_groups[template_index][slot]
        for slot in range(len(panel_groups[0]))
        for template_index in range(len(panel_groups))
    ]
    for panel_index, panel in enumerate(panels):
        panel["sampling_seed"] = SAMPLING_SEED_START + panel_index
    return panels


def _build_cells(
    tasks: list[dict[str, Any]], panels: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    task_by_id = {task["task_id"]: task for task in tasks}
    cells: list[dict[str, Any]] = []
    for panel in panels:
        task = task_by_id[panel["task_id"]]
        for position, arm in enumerate(panel["execution_order"]):
            cells.append(
                {
                    "cell_id": f"{panel['panel_id']}/arm={arm}",
                    "panel_id": panel["panel_id"],
                    "task_id": task["task_id"],
                    "template": task["template"],
                    "difficulty": task["difficulty"],
                    "arm": arm,
                    "position": position,
                    "sampling_seed": panel["sampling_seed"],
                }
            )
    return cells


def build_schedule() -> dict[str, Any]:
    tasks = _build_tasks()
    panels = _build_panels(tasks)
    cells = _build_cells(tasks, panels)
    return {"tasks": tasks, "panels": panels, "cells": cells}


def _normalize_schedule(schedule: Mapping[str, Any]) -> dict[str, Any]:
    """Strip analysis-added fields (e.g. task_commitment_hash) for comparison."""
    tasks = [
        {key: value for key, value in task.items() if key != "task_commitment_hash"}
        for task in schedule["tasks"]
    ]
    return {"tasks": tasks, "panels": schedule["panels"], "cells": schedule["cells"]}


def validate_schedule(schedule: Mapping[str, Any]) -> None:
    expected = build_schedule()
    if _normalize_schedule(schedule) != expected:
        raise ValueError("prompt-only schedule drifted from its frozen design")
    tasks = schedule["tasks"]
    panels = schedule["panels"]
    cells = schedule["cells"]
    if len(tasks) != EXPECTED_TASKS:
        raise ValueError("prompt-only schedule must contain exactly 12 tasks")
    if len(panels) != EXPECTED_PANELS:
        raise ValueError("prompt-only schedule must contain exactly 24 panels")
    if len(cells) != EXPECTED_CELLS:
        raise ValueError("prompt-only schedule must contain exactly 72 cells")
    if [task["seed"] for task in tasks] != list(range(TASK_SEED_START, TASK_SEED_START + EXPECTED_TASKS)):
        raise ValueError("task seeds must be consecutive starting at TASK_SEED_START")
    if [task["template"] for task in tasks[:6]] != ["form_entry_validation"] * 6:
        raise ValueError("first six tasks must be form_entry_validation")
    if [task["template"] for task in tasks[6:]] != ["distractor_recovery"] * 6:
        raise ValueError("last six tasks must be distractor_recovery")
    for template in PROMPT_TEMPLATES:
        difficulties = [
            task["difficulty"] for task in tasks if task["template"] == template
        ]
        if difficulties != [
            difficulty for difficulty in DIFFICULTIES for _ in range(TASK_SEEDS_PER_CELL)
        ]:
            raise ValueError(f"{template} must have two seeds per difficulty")
    if [panel["sampling_seed"] for panel in panels] != list(
        range(SAMPLING_SEED_START, SAMPLING_SEED_START + EXPECTED_PANELS)
    ):
        raise ValueError("sampling seeds must be consecutive from SAMPLING_SEED_START")
    for template in PROMPT_TEMPLATES:
        template_panels = [p for p in panels if p["template"] == template]
        orders = [tuple(p["execution_order"]) for p in template_panels]
        counts: dict[tuple[str, ...], int] = {}
        for order in orders:
            counts[order] = counts.get(order, 0) + 1
        if set(counts) != set(ARM_PERMUTATIONS):
            raise ValueError(f"{template} must use all six arm permutations")
        if set(counts.values()) != {2}:
            raise ValueError(f"{template} must use each arm permutation exactly twice")
        for position in range(len(ARMS)):
            arms_at_position = [order[position] for order in orders]
            if {arms_at_position.count(arm) for arm in ARMS} != {4}:
                raise ValueError(
                    f"{template} must balance each arm across position {position}"
                )
    task_by_id = {task["task_id"]: task for task in tasks}
    for task_id in task_by_id:
        replicas = {
            panel["rollout_replica"]
            for panel in panels
            if panel["task_id"] == task_id
        }
        if replicas != {0, 1}:
            raise ValueError(f"task {task_id} must have rollout replicas 0 and 1")
    first_replica_counts: dict[int, int] = {0: 0, 1: 0}
    for task_id in task_by_id:
        task_panels = [panel for panel in panels if panel["task_id"] == task_id]
        first = min(task_panels, key=panels.index)
        first_replica_counts[int(first["rollout_replica"])] += 1
    if first_replica_counts != {0: 6, 1: 6}:
        raise ValueError(
            f"replica chronology is not counterbalanced: {first_replica_counts}"
        )


# ---------------------------------------------------------------------------
# Collision scanning (fail closed)
# ---------------------------------------------------------------------------


def _forbidden_identifiers(manifest: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    """Return (substring identifiers, exact seed strings) from a manifest."""
    identifiers: set[str] = set()
    seeds: set[str] = set()
    for task in manifest["tasks"]:
        identifiers.add(str(task["task_id"]))
        seeds.add(str(task["seed"]))
    for panel in manifest["panels"]:
        identifiers.add(str(panel["panel_id"]))
        seeds.add(str(panel["sampling_seed"]))
    for cell in manifest["cells"]:
        identifiers.add(str(cell["cell_id"]))
    seeds.add(str(manifest["schedule_seed"]))
    seeds.add(str(manifest["simulator_seed"]))
    return identifiers, seeds


# Fail-closed collision scan configuration. Files larger than this ceiling are
# refused rather than partially scanned; JSON/JSONL is streamed in bounded
# chunks so arbitrarily large artifacts never load fully into memory.
MAX_SCAN_FILE_BYTES = 1 << 30  # 1 GiB
_SCAN_CHUNK_BYTES = 1 << 20  # 1 MiB streaming chunk
_SCAN_OVERLAP_BYTES = 512  # larger than the longest candidate identifier


_ID_RE = re.compile(
    r"unbrowser-fixture-v3-(?:form_entry_validation|distractor_recovery)"
    r"-(?:easy|medium|hard)-(\d{10})(?:/replica=\d(?:/arm=[ECR])?)?"
)
_DIGIT_RE = re.compile(r"\d+")


def _iter_json_artifact_files(
    run_root: str | Path, exclude_paths: Sequence[str | Path]
):
    """Yield every ``.json``/``.jsonl`` file under the run root, recursively."""
    root = Path(run_root).expanduser().resolve()
    excluded = {Path(path).expanduser().resolve() for path in exclude_paths}
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path in excluded:
            continue
        if path.suffix in {".json", ".jsonl"}:
            yield path


def _scan_file_collisions(
    path: Path, seed_set: set[str]
) -> list[dict[str, Any]]:
    """Fail-closed stream scan of one artifact file for candidate IDs/seeds.

    Returns collision records. Raises (rather than skipping) on unreadable or
    oversized files, and never skips a digit run when looking for an embedded
    candidate seed. Files are streamed in bounded chunks so arbitrarily large
    artifacts never load fully into memory.
    """
    try:
        size = path.stat().st_size
    except OSError as error:
        raise ValueError(f"collision scan cannot stat artifact: {path}") from error
    if size > MAX_SCAN_FILE_BYTES:
        raise ValueError(
            f"collision scan refuses oversized artifact ({size} bytes > "
            f"{MAX_SCAN_FILE_BYTES}): {path}"
        )
    collisions: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    carry = ""
    try:
        with path.open("rb") as handle:
            decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
            while True:
                chunk = handle.read(_SCAN_CHUNK_BYTES)
                if not chunk:
                    break
                try:
                    text = decoder.decode(chunk, final=False)
                except UnicodeDecodeError as error:
                    raise ValueError(
                        f"collision scan cannot decode artifact as UTF-8: {path}"
                    ) from error
                window = carry + text
                carry = window[-_SCAN_OVERLAP_BYTES:]
                for match in _ID_RE.finditer(window):
                    if match.group(1) in seed_set:
                        key = ("identifier", match.group())
                        if key not in seen:
                            seen.add(key)
                            collisions.append(
                                {
                                    "file": path.as_posix(),
                                    "kind": "identifier",
                                    "value": match.group(),
                                }
                            )
                for match in _DIGIT_RE.finditer(window):
                    run = match.group()
                    if len(run) < 10:
                        continue
                    if run in seed_set:
                        key = ("seed", run)
                        if key not in seen:
                            seen.add(key)
                            collisions.append(
                                {
                                    "file": path.as_posix(),
                                    "kind": "seed",
                                    "value": run,
                                    "ambiguous": False,
                                }
                            )
                        continue
                    for offset in range(len(run) - 9):
                        piece = run[offset : offset + 10]
                        if piece in seed_set:
                            key = ("seed", piece)
                            if key not in seen:
                                seen.add(key)
                                collisions.append(
                                    {
                                        "file": path.as_posix(),
                                        "kind": "seed",
                                        "value": piece,
                                        "ambiguous": True,
                                    }
                                )
            try:
                decoder.decode(b"", final=True)
            except UnicodeDecodeError as error:
                raise ValueError(
                    f"collision scan cannot decode artifact as UTF-8: {path}"
                ) from error
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"collision scan cannot read artifact: {path}") from error
    return collisions


def scan_collisions(
    manifest: Mapping[str, Any],
    run_root: str | Path,
    *,
    exclude_paths: Sequence[str | Path] = (),
) -> list[dict[str, Any]]:
    """Return any identifier/seed collisions found in prior .runs artifacts.

    Only identifiers and seeds are extracted; prior outcomes are never parsed
    into analysis. Every ``.json``/``.jsonl`` file under the run root is
    stream-scanned recursively, including nested directories. Any unreadable,
    oversized, or candidate-containing file is a hard failure for the caller.
    """
    identifiers, seeds = _forbidden_identifiers(manifest)
    _ = identifiers  # identifiers embed their task seed; seed sweep subsumes them
    seed_set = {value for value in seeds if len(value) == 10 and value.isdigit()}
    collisions: list[dict[str, Any]] = []
    for path in _iter_json_artifact_files(run_root, exclude_paths):
        collisions.extend(_scan_file_collisions(path, seed_set))
    return collisions


def _normalized_exclude_paths(exclude_paths: Sequence[str | Path]) -> list[str]:
    """Return the sorted, deduplicated, resolved exact paths to exclude.

    Only exact file paths are ever excluded — never directories or wildcards —
    so a caller cannot gain exclusion by supplying a broad prefix.
    """
    return sorted({str(Path(path).expanduser().resolve()) for path in exclude_paths})


def _derive_bound_artifact_exclusions(
    run_root: str | Path, *artifact_paths: str | Path
) -> list[str]:
    """Derive the exact bound-artifact paths that lie under ``run_root``.

    The freshness re-scan excludes exactly the pilot's own immutable artifacts
    (manifest, registry, local preflight) when they are stored under the scanned
    run root, so standard outputs inside ``.runs`` do not self-collide. Paths
    outside the run root are ignored (the scan never visits them). Only exact,
    already-supplied artifact paths are ever excluded — no broad directories and
    no user-controlled wildcard bypasses.
    """
    root = Path(run_root).expanduser().resolve()
    result: set[str] = set()
    for path in artifact_paths:
        if path is None or path == "":
            continue
        resolved = Path(path).expanduser().resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        result.add(str(resolved))
    return sorted(result)


def assert_no_collisions(
    manifest: Mapping[str, Any],
    run_root: str | Path,
    *,
    exclude_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    collisions = scan_collisions(manifest, run_root, exclude_paths=exclude_paths)
    if collisions:
        detail = ", ".join(
            f"{item['kind']}={item['value']!r} in {item['file']}" for item in collisions[:8]
        )
        raise ValueError(f"prompt-only pilot collides with prior artifacts: {detail}")
    return {
        "collisions": 0,
        "scanned_run_root": str(Path(run_root).expanduser().resolve()),
        "excluded_paths": _normalized_exclude_paths(exclude_paths),
    }


# ---------------------------------------------------------------------------
# Canonical source bundle manifest (authoritative prompt-only source identity)
# ---------------------------------------------------------------------------

# Exact cache/build-junk exclusion rules for the source bundle. Only these
# directories (matched by exact directory name relative to a covered namespace,
# never by absolute parent components), directories whose basename ends in
# ``.egg-info``, and these exact file suffixes are excluded; everything else
# that is a regular file under a covered namespace is part of the identity.
_SOURCE_BUNDLE_NAMESPACES = ("src", "pi_extensions", "policies")
_SOURCE_BUNDLE_TOP_FILES = ("pyproject.toml", "requirements-train.txt")
_SOURCE_BUNDLE_EXCLUDED_DIRS = frozenset(
    {
        "__pycache__",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "venv",
        "node_modules",
        ".eggs",
        "build",
        "dist",
        "__pypackages__",
    }
)
_SOURCE_BUNDLE_EXCLUDED_SUFFIXES = frozenset(
    {".pyc", ".pyo", ".pyd", ".so", ".dylib", ".o", ".a", ".pyi~"}
)


def _is_excluded_dir_component(name: str) -> bool:
    """Return True when a single relative directory component is cache/build junk."""
    return name in _SOURCE_BUNDLE_EXCLUDED_DIRS or name.endswith(".egg-info")


def build_source_bundle_manifest(root: str | Path) -> dict[str, Any]:
    """Build the canonical, content-addressed source bundle manifest.

    Covers every regular file under ``src``, ``pi_extensions`` and ``policies``
    plus ``pyproject.toml`` and ``requirements-train.txt``. Files are listed by
    sorted relative path with byte size and SHA-256. Symlinks, non-regular
    files, unreadable files, and symlinked/non-directory namespace roots are
    hard failures. Exclusion checks inspect only path components relative to
    each covered namespace (never absolute parent components), so a checkout
    whose ancestor directory is named ``venv``/``.git``/etc. still produces the
    full manifest. Only the exact cache/build exclusions above are skipped. The
    embedded ``bundle_hash`` is the canonical hash of the manifest payload and
    is the authoritative source identity.
    """
    project = Path(root).expanduser().resolve()
    entries: list[dict[str, Any]] = []
    seen: set[Path] = set()

    def _collect(path: Path) -> None:
        if path.is_symlink():
            raise ValueError(f"source bundle refuses symlink: {path}")
        if not path.is_file():
            raise ValueError(f"source bundle refuses non-regular file: {path}")
        if path in seen:
            return
        seen.add(path)
        relative = path.relative_to(project).as_posix()
        try:
            content = path.read_bytes()
        except OSError as error:
            raise ValueError(f"source bundle cannot read file: {path}") from error
        entries.append(
            {
                "path": relative,
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )

    for directory in _SOURCE_BUNDLE_NAMESPACES:
        base = project / directory
        if base.is_symlink():
            raise ValueError(f"source bundle refuses symlink namespace root: {base}")
        if base.exists() and not base.is_dir():
            raise ValueError(
                f"source bundle refuses non-directory namespace root: {base}"
            )
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"source bundle refuses symlink: {path}")
            if not path.is_file():
                continue
            relative = path.relative_to(base)
            # Inspect only components relative to the namespace root, never
            # absolute parent components.
            if any(
                _is_excluded_dir_component(part) for part in relative.parts[:-1]
            ):
                continue
            if path.suffix in _SOURCE_BUNDLE_EXCLUDED_SUFFIXES:
                continue
            _collect(path)
    for name in _SOURCE_BUNDLE_TOP_FILES:
        path = project / name
        if path.exists():
            _collect(path)
    entries.sort(key=lambda item: item["path"])
    payload: dict[str, Any] = {
        "schema_version": SOURCE_BUNDLE_SCHEMA_VERSION,
        "files": entries,
    }
    return {**payload, "bundle_hash": _canonical_hash(payload)}


def source_bundle_manifest_hash(root: str | Path) -> str:
    """Return the authoritative source bundle hash for ``root``."""
    return str(build_source_bundle_manifest(root)["bundle_hash"])


def _is_hex_digest_64(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def content_addressed_project_path(project: str, bundle_hash: str) -> str:
    """Derive a content-addressed remote project path from the bundle hash."""
    return f"{str(project).rstrip('/')}-{bundle_hash}"


def project_is_content_addressed(project: str, bundle_hash: str) -> bool:
    """Return True when the remote project basename is content-addressed."""
    return str(project).endswith(f"-{bundle_hash}")


# ---------------------------------------------------------------------------
# Cache OFF (isolated no-cache server) binding
# ---------------------------------------------------------------------------


def build_cache_off_server_binding(
    server_binary: str,
    model_artifact: str,
) -> dict[str, Any]:
    """Reconstruct the cache substrate's OFF server argv/hash deterministically.

    This binds the OFF (``--no-cache-prompt``) server identity as a *source
    input*. It does not imply the cache canary itself passed; that remains an
    independent, non-authorizing status. The prompt-only OFF binding adds an
    exact ``--slot-save-path`` erase-only feature-gate directory: native KV
    persistence stays forbidden, and the read-only empty directory exists only
    to unlock the pinned server's slot-action gate.
    """
    common_argv = _common_server_argv(server_binary, model_artifact)
    off_argv = [
        *common_argv,
        "--slot-save-path",
        SLOT_ACTION_DIRECTORY,
        "--no-cache-prompt",
    ]
    return {
        "source_screen": "cache-mechanics-canary-substrate-20260815-v1",
        "mode": "off",
        "model_alias": _CACHE_MODEL_ALIAS,
        "host": "127.0.0.1",
        "port": _CACHE_SERVER_PORT,
        "server_binary": server_binary,
        "model_artifact": model_artifact,
        "server_argv": off_argv,
        "server_argv_hash": canonical_receipt_hash(off_argv),
        "slot_save_path": SLOT_ACTION_DIRECTORY,
        "slot_save_path_policy": "erase_only_feature_gate_exception",
        "native_persistence_forbidden": True,
        "slot_action_directory_mode": SLOT_ACTION_DIRECTORY_MODE,
        "slot_action_directory_empty_required": True,
        "cache_canary_implied_passed": False,
    }


# ---------------------------------------------------------------------------
# Dummy keyless-provider API key binding (never a secret)
# ---------------------------------------------------------------------------


def dummy_api_key_binding() -> dict[str, Any]:
    """Return the frozen mode/hash binding for the dummy keyless-provider key.

    Artifacts bind only the mode and a SHA-256 of the fixed non-secret literal;
    the literal itself never appears in a bound artifact (the exact production
    ``--api-key <dummy>`` command is the only place the constant is threaded).
    """
    return {
        "mode": "fixed_dummy_non_secret",
        "key_sha256": hashlib.sha256(
            DUMMY_PROVIDER_API_KEY.encode("utf-8")
        ).hexdigest(),
        "length": len(DUMMY_PROVIDER_API_KEY),
    }


# ---------------------------------------------------------------------------
# Frozen provider config (models.json) — no credentials
# ---------------------------------------------------------------------------


def build_frozen_models_json() -> dict[str, Any]:
    """Return the exact run-specific Pi ``models.json`` content (no secrets).

    Pi 0.84.1 rejects a model whose ``samplingParams`` is ``null`` (the schema
    requires an object), so the key is omitted entirely rather than emitted as
    ``null``.
    """
    return {
        "providers": {
            RUN_PROVIDER: {
                "baseUrl": RUN_PI_BASE_URL,
                "models": [
                    {
                        "id": RUN_MODEL_ALIAS,
                        "api": "openai-completions",
                        "contextWindow": 65536,
                        "maxTokens": 8192,
                        "reasoning": False,
                    }
                ],
            }
        }
    }


def models_json_sha256() -> str:
    return canonical_receipt_hash(build_frozen_models_json())


def _assert_models_json_has_no_credentials(models_json: Mapping[str, Any]) -> None:
    forbidden = frozenset(
        {"apikey", "api_key", "token", "secret", "password", "authorization", "credential"}
    )

    def scan(value: Any, prefix: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                label = f"{prefix}.{key}" if prefix else str(key)
                if isinstance(key, str) and key.casefold() in forbidden:
                    raise ValueError(f"frozen models.json carries a credential key: {label}")
                scan(item, label)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                scan(item, f"{prefix}[{index}]")

    scan(models_json, "")


def write_frozen_models_json(config_dir: str | Path) -> dict[str, Any]:
    """Write the frozen, credential-free ``models.json`` into a fresh config dir.

    Never touches the user's default Pi config directory. The caller must set
    ``PI_CODING_AGENT_DIR`` to this directory for the execution.
    """
    directory = Path(config_dir).expanduser().resolve()
    if directory.exists():
        raise FileExistsError(f"config dir must be fresh: {directory}")
    directory.mkdir(parents=True, exist_ok=False)
    content = build_frozen_models_json()
    _assert_models_json_has_no_credentials(content)
    path = directory / "models.json"
    _write_immutable_json(path, content)
    path.chmod(0o444)
    return {
        "config_dir": str(directory),
        "models_json_path": str(path),
        "models_json_sha256": models_json_sha256(),
        "credentials": "none",
    }


def prepare_frozen_models_json(config_dir: str | Path) -> dict[str, Any]:
    """Prepare the frozen ``models.json`` config dir, idempotent across resume.

    On first run it writes the fresh dir; on resume it verifies the existing
    content hash matches the frozen hash and fails closed on drift.
    """
    directory = Path(config_dir).expanduser().resolve()
    path = directory / "models.json"
    if not directory.exists():
        return write_frozen_models_json(directory)
    if not path.is_file():
        raise ValueError(f"config dir exists without a frozen models.json: {directory}")
    try:
        observed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("models.json is not valid JSON") from error
    if canonical_receipt_hash(observed) != models_json_sha256():
        raise ValueError("models.json drifted from the frozen run-specific content")
    return {
        "config_dir": str(directory),
        "models_json_path": str(path),
        "models_json_sha256": models_json_sha256(),
        "credentials": "none",
    }


def validate_frozen_models_json_config(config_dir: str | Path) -> dict[str, Any]:
    """Fail-closed re-validation of the actual isolated Pi config on disk.

    Re-reads the exact ``models.json`` that Pi will load, verifying the frozen
    content hash, zero credential keys, the exact run-specific provider/model
    identity, the loopback base URL, and (critically for Pi 0.84.1) that
    ``samplingParams`` is omitted rather than ``null``.
    """
    directory = Path(config_dir).expanduser().resolve()
    path = directory / "models.json"
    if not path.is_file():
        raise ValueError(f"frozen models.json is missing: {path}")
    if path.stat().st_mode & 0o777 != 0o444:
        raise ValueError("frozen models.json mode must be 0444")
    try:
        observed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("models.json is not valid JSON") from error
    if canonical_receipt_hash(observed) != models_json_sha256():
        raise ValueError("models.json drifted from the frozen run-specific content")
    _assert_models_json_has_no_credentials(observed)
    providers = observed.get("providers")
    provider = providers.get(RUN_PROVIDER) if isinstance(providers, Mapping) else None
    if not isinstance(provider, Mapping):
        raise ValueError("models.json omitted the run-specific provider")
    if provider.get("baseUrl") != RUN_PI_BASE_URL:
        raise ValueError("models.json base URL drifted from the frozen loopback proxy")
    models = provider.get("models")
    matches = [
        entry
        for entry in models
        if isinstance(entry, Mapping) and entry.get("id") == RUN_MODEL_ALIAS
    ] if isinstance(models, list) else []
    if len(matches) != 1:
        raise ValueError("models.json must declare exactly one run-specific model")
    if "samplingParams" in matches[0]:
        raise ValueError(
            "models.json must omit samplingParams (Pi 0.84.1 rejects null)"
        )
    return {
        "config_dir": str(directory),
        "models_json_path": str(path),
        "models_json_sha256": models_json_sha256(),
        "credentials": "none",
    }


# ---------------------------------------------------------------------------
# No-real-model Pi conformance gate (local preflight)
# ---------------------------------------------------------------------------

# Deterministic sentinel the loopback streaming stub returns; pi --print must
# echo it into stdout for the gate to pass.
_PI_CONFORMANCE_SENTINEL = "PYREPLAB-PROMPT-ONLY-CONFORMANCE-SENTINEL"
# Config warnings are emitted on stderr by Pi 0.84.1 when models.json is
# invalid; "no models available" appears on stdout and ALSO exits 0, so the
# gate scans both streams and never trusts the return code alone.
_PI_CONFIG_WARNING_MARKERS = (
    "warning:",
    "errors loading models.json",
    "invalid models.json schema",
)
_PI_MAX_CONFORMANCE_TEXT = 1 << 20  # 1 MiB bounded capture


def _pi_binary_identity(pi_executable: str) -> dict[str, str]:
    resolved = shutil.which(pi_executable)
    if resolved is None:
        raise RuntimeError(f"Pi executable not found: {pi_executable!r}")
    binary = Path(resolved).resolve()
    digest = hashlib.sha256()
    with binary.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    completed = subprocess.run(
        [str(binary), "--version"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    version = (completed.stdout or completed.stderr).strip()
    return {
        "path": str(binary),
        "sha256": digest.hexdigest(),
        "version": version,
    }


def _sanitized_pi_environment(config_dir: str | Path) -> dict[str, str]:
    """Minimal ``PI_OFFLINE=1`` environment for the no-model conformance gate.

    Only generic locale/temp/PATH variables survive; every PI_*, PYREPLAB_*,
    provider credential, and project variable is dropped so the gate can never
    inherit a real key or routing.
    """
    keep = ("PATH", "HOME", "TMPDIR", "TMP", "TEMP", "LANG", "LC_ALL", "LANGUAGE")
    environment = {key: os.environ[key] for key in keep if key in os.environ}
    environment["PI_CODING_AGENT_DIR"] = str(config_dir)
    environment["PI_OFFLINE"] = "1"
    return environment


def _list_models_warnings(stderr: str, stdout: str) -> list[str]:
    """Collect bounded config warnings from the list-models invocation."""
    warnings: list[str] = []
    for line in (stderr or "").splitlines():
        lowered = line.casefold()
        if any(marker in lowered for marker in _PI_CONFIG_WARNING_MARKERS):
            warnings.append(line.strip()[:240])
    if "no models available" in (stdout or "").casefold():
        warnings.append("no models available")
    return warnings[:8]


def _parse_list_models_rows(stdout: str) -> tuple[list[dict[str, str]], list[str]]:
    """Parse the ``pi --list-models`` table into provider/model rows.

    Returns ``(rows, unrecognized)`` where every row carries exactly
    ``{"provider", "model"}`` from the two leading columns. Header lines,
    documentation paths, and the "No models available." message are ignored;
    any other non-empty line that does not yield two tokens is reported as
    unrecognized so a drifted output format fails closed instead of passing.
    """
    rows: list[dict[str, str]] = []
    unrecognized: list[str] = []
    for line in (stdout or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.casefold()
        if lowered.startswith("no models available"):
            continue
        if "use /login" in lowered:
            continue
        if lowered.startswith("provider") and lowered.split()[0] == "provider":
            continue  # table header
        tokens = stripped.split()
        if len(tokens) >= 2:
            rows.append({"provider": tokens[0], "model": tokens[1]})
        else:
            unrecognized.append(stripped[:240])
    return rows, unrecognized


def _conformance_failures(
    *,
    pi_identity: Mapping[str, Any],
    list_models_rc: int,
    list_models_stdout: str,
    list_models_stderr: str,
    stub_observations: Mapping[str, Any] | None,
) -> list[str]:
    """Fail-closed verdict for the no-model Pi conformance gate."""
    failures: list[str] = []
    if pi_identity.get("sha256") != _RUNTIME_PINS["pi_cli_sha256"]:
        failures.append(
            f"pi digest drift from the pinned binary "
            f"({pi_identity.get('sha256')!r})"
        )
    if pi_identity.get("version") != _RUNTIME_PINS["pi_version"]:
        failures.append(
            f"pi version drift from the pinned runtime "
            f"({pi_identity.get('version')!r} != {_RUNTIME_PINS['pi_version']!r})"
        )
    if list_models_rc != 0:
        failures.append(f"list-models return code {list_models_rc}")
    warnings = _list_models_warnings(list_models_stderr, list_models_stdout)
    if warnings:
        failures.append(f"config warnings: {' | '.join(warnings)}")
    rows, unrecognized = _parse_list_models_rows(list_models_stdout)
    expected = [
        row
        for row in rows
        if row["provider"] == RUN_PROVIDER and row["model"] == RUN_MODEL_ALIAS
    ]
    if len(expected) != 1:
        failures.append(
            f"expected exactly one {RUN_PROVIDER}/{RUN_MODEL_ALIAS} model row, "
            f"got {len(expected)}"
        )
    if unrecognized:
        failures.append(f"unrecognized list-models output: {unrecognized[:3]!r}")
    if stub_observations is None:
        failures.append("streaming stub observations are missing")
    else:
        failures.extend(_verify_conformance_stub_observations(stub_observations))
    return failures


def _conformance_stub_observations(
    pi_binary: str,
    workdir: str | Path,
    env: Mapping[str, str],
    *,
    timeout: float,
) -> dict[str, Any]:
    """Run Pi once, non-interactively, against a loopback OpenAI streaming stub.

    No tools, extensions, context files, skills, or session are enabled. The
    stub records every request (path, bearer header, model, stream flag) and
    answers with a deterministic sentinel. Returns bounded observations;
    ``requests`` is empty and ``rc`` is ``None`` when pi could not be invoked.
    """
    sentinel = _PI_CONFORMANCE_SENTINEL

    class _StubHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, _format: str, *_args: object) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler contract
            length = int(self.headers.get("content-length", "0"))
            try:
                payload = json.loads(self.rfile.read(length))
            except json.JSONDecodeError:
                payload = {}
            server = self.server
            with server.obs_lock:  # type: ignore[attr-defined]
                server.observations["requests"].append(  # type: ignore[attr-defined]
                    {
                        "path": self.path,
                        "auth": self.headers.get("Authorization"),
                        "model": payload.get("model"),
                        "stream": payload.get("stream"),
                    }
                )
            chunks = [
                {
                    "id": "chatcmpl-conformance-1",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": RUN_MODEL_ALIAS,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": sentinel},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "chatcmpl-conformance-1",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": RUN_MODEL_ALIAS,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
            ]
            body = "".join(
                f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n"
                for chunk in chunks
            ) + "data: [DONE]\n\n"
            encoded = body.encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(encoded)))
            self.send_header("connection", "close")
            self.end_headers()
            self.wfile.write(encoded)

    observations: dict[str, Any] = {"requests": [], "rc": None}
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
    server.observations = observations  # type: ignore[attr-defined]
    server.obs_lock = threading.Lock()  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        # The stub config mirrors the frozen models.json exactly except for the
        # loopback base URL; it is a validation artifact, never a bound config.
        stub_config = build_frozen_models_json()
        stub_config["providers"][RUN_PROVIDER]["baseUrl"] = (
            f"http://127.0.0.1:{server.server_port}/v1"
        )
        stub_config_dir = Path(workdir) / "stub-config"
        stub_config_dir.mkdir(parents=True, exist_ok=False)
        (stub_config_dir / "models.json").write_text(
            json.dumps(stub_config, sort_keys=True), encoding="utf-8"
        )
        stub_env = {**dict(env), "PI_CODING_AGENT_DIR": str(stub_config_dir)}
        command = [
            pi_binary,
            "--provider",
            RUN_PROVIDER,
            "--model",
            RUN_MODEL_ALIAS,
            "--api-key",
            DUMMY_PROVIDER_API_KEY,
            "--thinking",
            "off",
            "--mode",
            "json",
            "--print",
            "--no-session",
            "--no-builtin-tools",
            "--no-context-files",
            "--no-skills",
            "--no-prompt-templates",
            "--no-extensions",
            "--no-approve",
            f"Reply with exactly this sentinel: {sentinel}",
        ]
        completed = subprocess.run(
            command,
            cwd=workdir,
            env=stub_env,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        observations["rc"] = completed.returncode
        observations["stdout"] = completed.stdout[:_PI_MAX_CONFORMANCE_TEXT]
        observations["stderr"] = completed.stderr[:_PI_MAX_CONFORMANCE_TEXT]
        observations["config_sha256"] = canonical_receipt_hash(stub_config)
        observations["stub_models_json"] = stub_config
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    return observations


def _verify_conformance_stub_observations(
    observations: Mapping[str, Any],
) -> list[str]:
    """Fail-closed verdict over the loopback streaming-stub observations."""
    failures: list[str] = []
    requests = observations.get("requests")
    if not isinstance(requests, list) or len(requests) != 1:
        failures.append(
            f"stub expected exactly one /v1/chat/completions request, "
            f"got {len(requests) if isinstance(requests, list) else 'none'}"
        )
        return failures
    request = requests[0]
    if not isinstance(request, Mapping):
        failures.append("stub request is malformed")
        return failures
    if request.get("path") != "/v1/chat/completions":
        failures.append(f"stub request path is wrong: {request.get('path')!r}")
    if request.get("model") != RUN_MODEL_ALIAS:
        failures.append(f"stub request model is wrong: {request.get('model')!r}")
    if request.get("stream") is not True:
        failures.append(f"stub request stream flag is wrong: {request.get('stream')!r}")
    expected_auth = f"Bearer {DUMMY_PROVIDER_API_KEY}"
    if request.get("auth") != expected_auth:
        failures.append("stub request bearer header does not match the dummy key")
    if observations.get("rc") != 0:
        failures.append(f"stub invocation return code {observations.get('rc')!r}")
    stdout = str(observations.get("stdout") or "")
    if _PI_CONFORMANCE_SENTINEL not in stdout:
        failures.append("stub sentinel response was not echoed into pi stdout")
    stderr = str(observations.get("stderr") or "")
    warnings = _list_models_warnings(stderr, "")
    if warnings:
        failures.append(f"stub config warnings: {' | '.join(warnings)}")
    return failures


def build_pi_conformance_receipt(
    *,
    pi_identity: Mapping[str, Any],
    list_models_rc: int,
    list_models_stdout: str,
    list_models_stderr: str,
    streaming_stub: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the normalized, self-hashed conformance receipt (pure builder).

    Raw output is never persisted: only bounded structural fields and hashes
    are bound. ``receipt_hash`` covers the normalized payload (schema, screen,
    api-key binding, executable identity, frozen config hash, list-models
    verdict, and the optional streaming-stub verdict), so the same gate
    invocation revalidates deterministically without rerunning Pi.
    """
    warnings = _list_models_warnings(list_models_stderr, list_models_stdout)
    rows, _ = _parse_list_models_rows(list_models_stdout)
    normalized: dict[str, Any] = {
        "schema_version": PI_CONFORMANCE_SCHEMA_VERSION,
        "screen_id": SCREEN_ID,
        "api_key_binding": dummy_api_key_binding(),
        "pi_version": str(pi_identity["version"]),
        "pi_sha256": str(pi_identity["sha256"]),
        "models_json_sha256": models_json_sha256(),
        "list_models": {
            "rc": int(list_models_rc),
            "warnings_count": len(warnings),
            "warnings_sha256": hashlib.sha256(
                "\n".join(warnings).encode("utf-8")
            ).hexdigest(),
            "rows": rows,
            "expected_row_count": 1,
            "stdout_sha256": hashlib.sha256(
                (list_models_stdout or "")[: _PI_MAX_CONFORMANCE_TEXT].encode("utf-8")
            ).hexdigest(),
        },
        "streaming_stub": None,
    }
    if streaming_stub is not None:
        requests = streaming_stub.get("requests")
        request = requests[0] if isinstance(requests, list) and requests else {}
        request = request if isinstance(request, Mapping) else {}
        normalized["streaming_stub"] = {
            "schema_version": PI_CONFORMANCE_STREAMING_STUB_SCHEMA_VERSION,
            "request_count": len(requests) if isinstance(requests, list) else 0,
            "request_path": request.get("path"),
            "request_model": request.get("model"),
            "request_stream": request.get("stream"),
            "request_auth_sha256": hashlib.sha256(
                str(request.get("auth") or "").encode("utf-8")
            ).hexdigest(),
            "sentinel_present": (
                _PI_CONFORMANCE_SENTINEL
                in str(streaming_stub.get("stdout") or "")
            ),
            "rc": streaming_stub.get("rc"),
            "config_sha256": streaming_stub.get("config_sha256"),
            "stdout_sha256": hashlib.sha256(
                str(streaming_stub.get("stdout") or "")
                [:_PI_MAX_CONFORMANCE_TEXT].encode("utf-8")
            ).hexdigest(),
        }
    return {**normalized, "receipt_hash": _canonical_hash(normalized)}


def run_pi_conformance(
    pi_executable: str = "pi",
    *,
    include_streaming_stub: bool = True,
    timeout: float = 90.0,
) -> dict[str, Any]:
    """Run the no-real-model Pi conformance gate and return its receipt.

    Uses the pinned Pi binary with an isolated temporary
    ``PI_CODING_AGENT_DIR``, a sanitized ``PI_OFFLINE=1`` environment, and the
    frozen credential-free ``models.json``. At minimum it invokes
    ``pi --api-key <dummy> --list-models`` and requires return code 0, zero
    config warnings, and exactly the expected provider/model row. It also
    invokes Pi
    non-interactively (no tools/extensions/context/skills/session) against a
    deterministic loopback OpenAI-compatible streaming stub, verifying exactly
    one ``/v1/chat/completions`` request with the exact model, stream mode,
    dummy bearer header, sentinel response, and clean stub shutdown. Fails
    closed (raises) on any failure; the returned receipt is always valid.
    """
    if not include_streaming_stub:
        raise ValueError("the loopback streaming stub is required for conformance")
    pi_identity = _pi_binary_identity(pi_executable)
    stub_observations: Mapping[str, Any] | None = None
    with tempfile.TemporaryDirectory(prefix="pyreplab-pi-conformance-") as directory:
        config_dir = Path(directory) / "config"
        write_frozen_models_json(config_dir)
        env = _sanitized_pi_environment(config_dir)
        list_argv = [
            str(pi_identity["path"]),
            "--provider",
            RUN_PROVIDER,
            "--model",
            RUN_MODEL_ALIAS,
            "--api-key",
            DUMMY_PROVIDER_API_KEY,
            "--list-models",
        ]
        completed = subprocess.run(
            list_argv,
            cwd=directory,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        stub_observations = _conformance_stub_observations(
            str(pi_identity["path"]),
            directory,
            env,
            timeout=timeout,
        )
    failures = _conformance_failures(
        pi_identity=pi_identity,
        list_models_rc=completed.returncode,
        list_models_stdout=completed.stdout[:_PI_MAX_CONFORMANCE_TEXT],
        list_models_stderr=completed.stderr[:_PI_MAX_CONFORMANCE_TEXT],
        stub_observations=stub_observations,
    )
    if failures:
        raise RuntimeError("pi conformance gate failed: " + "; ".join(failures))
    return build_pi_conformance_receipt(
        pi_identity=pi_identity,
        list_models_rc=completed.returncode,
        list_models_stdout=completed.stdout[:_PI_MAX_CONFORMANCE_TEXT],
        list_models_stderr=completed.stderr[:_PI_MAX_CONFORMANCE_TEXT],
        streaming_stub=stub_observations,
    )


def validate_pi_conformance_receipt(receipt: Mapping[str, Any]) -> None:
    """Structurally revalidate a bound conformance receipt without rerunning.

    Verifies the embedded normalized hash, schema, screen, dummy-key binding,
    pinned executable identity, frozen config hash, the exact list-models
    verdict, and (when present) the exact streaming-stub verdict.
    """
    _verify_embedded_hash(receipt, "receipt_hash")
    if receipt.get("schema_version") != PI_CONFORMANCE_SCHEMA_VERSION:
        raise ValueError("unsupported pi conformance receipt schema")
    if receipt.get("screen_id") != SCREEN_ID:
        raise ValueError("pi conformance receipt screen mismatch")
    if receipt.get("api_key_binding") != dummy_api_key_binding():
        raise ValueError("pi conformance receipt dummy-key binding drifted")
    if receipt.get("pi_sha256") != _RUNTIME_PINS["pi_cli_sha256"]:
        raise ValueError(
            "pi conformance receipt does not bind the pinned Pi executable"
        )
    if receipt.get("pi_version") != _RUNTIME_PINS["pi_version"]:
        raise ValueError("pi conformance receipt Pi version drifted")
    if receipt.get("models_json_sha256") != models_json_sha256():
        raise ValueError("pi conformance receipt frozen config hash drifted")
    listed = receipt.get("list_models")
    if not isinstance(listed, Mapping):
        raise ValueError("pi conformance receipt is missing the list-models verdict")
    if listed.get("rc") != 0:
        raise ValueError("pi conformance receipt list-models rc must be 0")
    if listed.get("warnings_count") != 0:
        raise ValueError("pi conformance receipt must have zero config warnings")
    if listed.get("warnings_sha256") != hashlib.sha256(b"").hexdigest():
        raise ValueError("pi conformance receipt warnings hash is inconsistent")
    rows = listed.get("rows")
    if rows != [{"provider": RUN_PROVIDER, "model": RUN_MODEL_ALIAS}]:
        raise ValueError(
            "pi conformance receipt must bind exactly the expected provider/model row"
        )
    if listed.get("expected_row_count") != 1:
        raise ValueError("pi conformance receipt expected-row count drifted")
    if not _is_hex_digest_64(listed.get("stdout_sha256")):
        raise ValueError("pi conformance receipt list-models stdout hash is invalid")
    stub = receipt.get("streaming_stub")
    if not isinstance(stub, Mapping):
        raise ValueError("pi conformance receipt requires the streaming stub verdict")
    if stub is not None:
        if stub.get("schema_version") != PI_CONFORMANCE_STREAMING_STUB_SCHEMA_VERSION:
            raise ValueError("pi conformance receipt streaming stub schema drifted")
        if stub.get("request_count") != 1:
            raise ValueError(
                "pi conformance receipt must bind exactly one stub request"
            )
        if stub.get("request_path") != "/v1/chat/completions":
            raise ValueError("pi conformance receipt stub request path drifted")
        if stub.get("request_model") != RUN_MODEL_ALIAS:
            raise ValueError("pi conformance receipt stub request model drifted")
        if stub.get("request_stream") is not True:
            raise ValueError("pi conformance receipt stub stream flag drifted")
        expected_auth_hash = hashlib.sha256(
            f"Bearer {DUMMY_PROVIDER_API_KEY}".encode("utf-8")
        ).hexdigest()
        if stub.get("request_auth_sha256") != expected_auth_hash:
            raise ValueError(
                "pi conformance receipt stub bearer header does not bind the dummy key"
            )
        if stub.get("sentinel_present") is not True:
            raise ValueError("pi conformance receipt stub sentinel is missing")
        if stub.get("rc") != 0:
            raise ValueError("pi conformance receipt stub rc must be 0")
        for field in ("config_sha256", "stdout_sha256"):
            if not _is_hex_digest_64(stub.get(field)):
                raise ValueError(
                    f"pi conformance receipt stub {field} hash is invalid"
                )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _prompt_only_runtime_pins() -> dict[str, Any]:
    pins = json.loads(json.dumps(_RUNTIME_PINS))
    pins["fixture_generator_version"] = OUTCOME_ONLY_GENERATOR_VERSION
    pins["rollout_replicas"] = ROLLOUT_REPLICAS
    # Route the pilot at the isolated OFF cache-canary server through the
    # local instrumentation proxy, never the default 18081/8081 endpoint.
    pins["provider"] = RUN_PROVIDER
    pins["model_alias"] = RUN_MODEL_ALIAS
    pins["pi_provider_config"]["base_url"] = RUN_PI_BASE_URL
    pins["remote_provider_base_url"] = RUN_REMOTE_SERVER_BASE_URL
    pins["llama_server_required_args"] = [
        f"--alias {RUN_MODEL_ALIAS}",
        "--ctx-size 65536",
        "--flash-attn on",
        "--n-cpu-moe 16",
        "--n-gpu-layers all",
        "--parallel 1",
        "--reasoning on",
        "--threads 8",
        f"--slot-save-path {SLOT_ACTION_DIRECTORY}",
        "--no-cache-prompt",
    ]
    pins["llama_server_config"]["cache_prompt"] = False
    # Pi 0.84.1 requires keyless local custom providers to carry an API key;
    # the run threads the fixed non-secret dummy key via ``--api-key``. Only
    # the mode/hash binding is frozen into the manifest (never the literal).
    pins["pi_api_key"] = dummy_api_key_binding()
    return pins


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


def _treatment_descriptor(
    treatment: TreatmentSpec, registry: TreatmentRegistry
) -> dict[str, Any]:
    return {
        "id": treatment.id,
        "version": treatment.version,
        "bundle_id": treatment.bundle_id,
        "bundle_hash": treatment.bundle_hash,
        "registry_hash": registry.registry_hash,
        "system_prompt_sha256": hashlib.sha256(
            treatment.system_prompt.encode("utf-8")
        ).hexdigest(),
        "system_prompt_length": len(treatment.system_prompt),
        "append_system_prompt": bool(treatment.system_prompt),
        "allowed_tools": list(treatment.allowed_tools),
        "tool_interface": treatment.tool_interface,
        "max_output_tokens": treatment.max_output_tokens,
        "tool_call_limit": treatment.tool_call_limit,
        "command_timeout_seconds": treatment.command_timeout_seconds,
        "wall_time_limit_seconds": treatment.wall_time_limit_seconds,
    }


def _build_task_commitments(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    commitments: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="pyreplab-prompt-only-") as directory:
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
            if task.public_metadata.get("prompt_profile") != TASK_PROMPT_PROFILE:
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
            commitments.append(unbrowser_fixture_task_commitment(directory, task.id))
            expected_hash = entry.get("task_commitment_hash")
            if (
                expected_hash is not None
                and commitments[-1]["commitment_hash"] != expected_hash
            ):
                raise ValueError(f"task commitment drifted: {task.id}")
    return commitments


def build_manifest(
    registry: TreatmentRegistry,
    remote_identity: Mapping[str, Any],
    *,
    registry_file: str,
    cache_server_binary: str | None = None,
    cache_model_artifact: str | None = None,
    declare_template_identity_available: bool = False,
) -> dict[str, Any]:
    by_arm = _validate_registry(registry)
    remote = _validate_remote_identity(remote_identity)
    schedule = build_schedule()
    tasks = schedule["tasks"]
    panels = schedule["panels"]
    cells = schedule["cells"]
    commitments = _build_task_commitments({"tasks": tasks})
    commitment_by_id = {str(item["task"]["id"]): item for item in commitments}
    for task in tasks:
        task["task_commitment_hash"] = commitment_by_id[task["task_id"]][
            "commitment_hash"
        ]
    runtime_pins = _prompt_only_runtime_pins()
    cache_binding = build_cache_off_server_binding(
        cache_server_binary or str(runtime_pins["llama_server_path"]),
        cache_model_artifact or str(runtime_pins["model_artifact_path"]),
    )
    payload: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "screen_id": SCREEN_ID,
        "generation_id": SCREEN_ID,
        "purpose": (
            "No-live, three-arm (E/C/R) prompt-discipline pilot over the V3 "
            "outcome-only fixture generator. Freezes immutable schedule and "
            "identity artifacts; never authorizes model execution."
        ),
        "registry_file": registry_file,
        "registry_hash": registry.registry_hash,
        "treatments": {
            arm: _treatment_descriptor(by_arm[arm], registry) for arm in ARMS
        },
        "arm_only_delta": _arm_only_delta(by_arm),
        "task_role": TASK_ROLE,
        "task_split": TASK_SPLIT,
        "task_prompt_profile": TASK_PROMPT_PROFILE,
        "task_generator_version": OUTCOME_ONLY_GENERATOR_VERSION,
        "fixture_verifier": {"id": VERIFIER_ID, "version": VERIFIER_VERSION},
        "known_templates": list(PROMPT_TEMPLATES),
        "untested_templates": [
            template for template in KNOWN_TEMPLATES if template not in PROMPT_TEMPLATES
        ],
        "held_templates": list(HELD_TEMPLATES),
        "tasks": tasks,
        "panels": panels,
        "cells": cells,
        "rollout_replicas": ROLLOUT_REPLICAS,
        "task_seed_start": TASK_SEED_START,
        "sampling_seed_start": SAMPLING_SEED_START,
        "schedule_seed": SCHEDULE_SEED,
        "simulator_seed": SIMULATOR_SEED,
        "wall_budget_amendment": build_wall_budget_amendment(),
        "runtime_pins": runtime_pins,
        "event_accounting": {
            "normalizer_schema_version": NORMALIZED_EVENT_SCHEMA_VERSION,
            "provider_turn_semantics": PROVIDER_TURN_SEMANTICS,
            "budget_receipt_schema_version": BUDGET_RECEIPT_SCHEMA_VERSION,
        },
        "remote_identity": remote,
        "isolated_no_cache_server_identity": cache_binding,
        "estimands": {
            "finite_bank": {
                "form_entry_validation": {"E": None, "C": None, "R": None},
                "distractor_recovery": {"E": None, "C": None, "R": None},
            },
            "finite_bank_note": (
                "Per-template arm success values; unknown pre-action and filled "
                "only by post-action analysis."
            ),
            "legal_lookup": {
                "form_entry_validation": "C",
                "distractor_recovery": "R",
            },
            "template_identity_available_pre_action": declare_template_identity_available,
            "lookup_value_is_diagnostic_only": not declare_template_identity_available,
        },
        "diagnostics": {
            "behavior_classification": {
                "source": "pyreplab_harness.m3_prompt_behavior",
                "role": "diagnostic_only",
                "advancement_gate": None,
                "scientific_threshold": None,
                "note": (
                    "The behavior/F2 completion-and-recovery classification is "
                    "reported purely as descriptive diagnostics. It has no "
                    "advancement gate and no scientific threshold is invented "
                    "for it."
                ),
            }
        },
        "command_contract": {
            "append_system_prompt_delta_only": True,
            "task_prompt_position": "final_positional_argument",
            "context_files": "disabled",
            "skills": "disabled",
            "prompt_templates": "disabled",
            "builtin_tools": "disabled",
            "tool_interface": UNBROWSER_INTERACTIVE_TOOL_INTERFACE,
        },
        "gates": {
            "tasks": EXPECTED_TASKS,
            "panels": EXPECTED_PANELS,
            "cells": EXPECTED_CELLS,
            "fixed_pooled_min": FIXED_POOLED_MIN,
            "fixed_template_min": FIXED_TEMPLATE_MIN,
            "interaction_lookup_min": INTERACTION_LOOKUP_MIN,
            "interaction_fc_fr_min": INTERACTION_FC_FR_MIN,
            "interaction_dr_dc_min": INTERACTION_DR_DC_MIN,
            "interaction_fc_fe_min": INTERACTION_FC_FE_MIN,
            "interaction_dr_de_min": INTERACTION_DR_DE_MIN,
            "gate_is_screen_not_efficacy_claim": True,
        },
        "authorization_boundary": {
            "live_model_execution_authorized": False,
            "cache_canary_implied_passed": False,
            "required_next_artifact": "explicit hash-bound execution authorization",
        },
        "exclusion": (
            "T_pilot tasks and all attempts are permanently excluded from meta-"
            "training, calibration, development, and final evaluation pools."
        ),
    }
    return {**payload, "manifest_hash": _canonical_hash(payload)}


def validate_manifest(manifest: Mapping[str, Any], registry: TreatmentRegistry) -> None:
    _verify_embedded_hash(manifest, "manifest_hash")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported prompt-only manifest schema")
    if manifest.get("screen_id") != SCREEN_ID:
        raise ValueError("prompt-only screen mismatch")
    if manifest.get("registry_hash") != registry.registry_hash:
        raise ValueError("prompt-only manifest registry hash mismatch")
    if manifest.get("wall_budget_amendment") != build_wall_budget_amendment():
        raise ValueError("prompt-only manifest wall budget amendment drifted")
    remote = manifest.get("remote_identity")
    if not isinstance(remote, Mapping):
        raise ValueError("manifest remote_identity must be an object")
    registry_file = manifest.get("registry_file")
    if not isinstance(registry_file, str) or not registry_file:
        raise ValueError("manifest registry_file must be a non-empty string")
    declared = manifest.get("estimands", {}).get("template_identity_available_pre_action")
    if not isinstance(declared, bool):
        raise ValueError("template_identity_available_pre_action must be boolean")
    if manifest["estimands"].get("lookup_value_is_diagnostic_only") is not (not declared):
        raise ValueError("lookup diagnostic flag must be the inverse of the declaration")
    expected = build_manifest(
        registry,
        remote,
        registry_file=registry_file,
        cache_server_binary=manifest["isolated_no_cache_server_identity"]["server_binary"],
        cache_model_artifact=manifest["isolated_no_cache_server_identity"]["model_artifact"],
        declare_template_identity_available=declared,
    )
    if dict(manifest) != expected:
        raise ValueError("prompt-only manifest drifted from its frozen design")
    validate_schedule(
        {"tasks": manifest["tasks"], "panels": manifest["panels"], "cells": manifest["cells"]}
    )
    if manifest["isolated_no_cache_server_identity"]["cache_canary_implied_passed"] is not False:
        raise ValueError("manifest must not imply the cache canary passed")
    if manifest["authorization_boundary"]["live_model_execution_authorized"] is not False:
        raise ValueError("manifest must remain non-authorizing")


# ---------------------------------------------------------------------------
# Command arm-isolation receipt
# ---------------------------------------------------------------------------


def _strip_append_system_prompt(argv: list[str]) -> list[str]:
    stripped: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--append-system-prompt":
            index += 2  # skip flag and its value
            continue
        stripped.append(token)
        index += 1
    return stripped


def _value_after(arguments: list[str], flag: str) -> str | None:
    try:
        return arguments[arguments.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def build_command_arm_receipt(
    manifest: Mapping[str, Any],
    registry: TreatmentRegistry,
    project_root: str | Path,
    *,
    pi_executable: str = "pi",
) -> dict[str, Any]:
    validate_manifest(manifest, registry)
    by_arm = _validate_registry(registry)
    remote = manifest["remote_identity"]
    runtime_pins = manifest["runtime_pins"]
    first_panel = manifest["panels"][0]
    first_task = next(
        task for task in manifest["tasks"] if task["task_id"] == first_panel["task_id"]
    )
    project_root_path = Path(project_root).expanduser().resolve()
    config = RemoteConfig(
        str(remote["host"]),
        str(remote["project"]),
        str(remote["run_root"]),
        str(remote["python"]),
    )
    arm_argv: dict[str, list[str]] = {}
    for arm in ARMS:
        treatment = by_arm[arm]
        policy = policy_spec_from_treatment(treatment)
        arm_argv[arm] = _build_pi_command(
            project_root_path,
            config,
            f"{remote['run_root']}/{_WORKSPACE_PLACEHOLDER}",
            _TASK_PROMPT_PLACEHOLDER,
            policy,
            pi_executable,
            None,
            str(runtime_pins["provider"]),
            str(runtime_pins["model_alias"]),
            str(runtime_pins["thinking"]),
            (
                f"{FIXTURE_BASE_URL}/{first_task['template']}/"
                f"{first_task['seed']}/{first_task['difficulty']}"
            ),
            str(runtime_pins["unbrowser_path"]),
            True,
            True,
            int(first_panel["sampling_seed"]),
            api_key=DUMMY_PROVIDER_API_KEY,
        )
    stripped = {arm: _strip_append_system_prompt(arm_argv[arm]) for arm in ARMS}
    if len({json.dumps(value, ensure_ascii=False) for value in stripped.values()}) != 1:
        raise ValueError("arm argv differ beyond append-system-prompt bytes")
    common = stripped["E"]
    public_common = list(common)
    api_key_index = public_common.index("--api-key") + 1
    public_common[api_key_index] = "<bound-dummy-api-key>"
    checks = {
        "arm_argv_equal_after_stripping_append_prompt": True,
        "append_system_prompt_present": {
            arm: "--append-system-prompt" in arm_argv[arm] for arm in ARMS
        },
        "tools_equal": all(
            _value_after(arm_argv[arm], "--tools") == "bash,unbrowser" for arm in ARMS
        ),
        "tool_limit_frozen": all(
            _value_after(arm_argv[arm], "--gym-tool-limit")
            == str(by_arm[arm].tool_call_limit)
            for arm in ARMS
        ),
        "command_timeout_frozen": all(
            _value_after(arm_argv[arm], "--gym-command-timeout")
            == str(by_arm[arm].command_timeout_seconds)
            for arm in ARMS
        ),
        "output_limit_frozen": all(
            _value_after(arm_argv[arm], "--gym-max-output-tokens")
            == str(by_arm[arm].max_output_tokens)
            for arm in ARMS
        ),
        "provider_turn_limit_frozen": all(
            _value_after(arm_argv[arm], "--gym-provider-turn-limit")
            == str(by_arm[arm].tool_call_limit + 1)
            for arm in ARMS
        ),
        "budget_v3_extension_loaded": all(
            any(argument.endswith("/pi_extensions/gym-budget-v3.ts") for argument in arm_argv[arm])
            for arm in ARMS
        ),
        "runtime_identity_equal": all(
            _value_after(arm_argv[arm], "--provider") == runtime_pins["provider"]
            and _value_after(arm_argv[arm], "--model") == runtime_pins["model_alias"]
            and _value_after(arm_argv[arm], "--thinking") == runtime_pins["thinking"]
            for arm in ARMS
        ),
        "run_specific_provider_model_frozen": all(
            _value_after(arm_argv[arm], "--provider") == RUN_PROVIDER
            and _value_after(arm_argv[arm], "--model") == RUN_MODEL_ALIAS
            for arm in ARMS
        ),
        "dummy_keyless_api_key_present": all(
            _value_after(arm_argv[arm], "--api-key") == DUMMY_PROVIDER_API_KEY
            for arm in ARMS
        ),
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"prompt-only arm command contract failed: {failed!r}")
    payload: dict[str, Any] = {
        "schema_version": COMMAND_RECEIPT_SCHEMA_VERSION,
        "manifest_hash": manifest["manifest_hash"],
        "registry_hash": registry.registry_hash,
        "common_argv": public_common,
        "common_argv_sha256": hashlib.sha256(
            json.dumps(common, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "arm_argv_sha256": {
            arm: hashlib.sha256(
                json.dumps(arm_argv[arm], ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest()
            for arm in ARMS
        },
        "arm_system_prompt_sha256": {
            arm: hashlib.sha256(
                by_arm[arm].system_prompt.encode("utf-8")
            ).hexdigest()
            for arm in ARMS
        },
        "api_key_binding": dummy_api_key_binding(),
        "checks": checks,
    }
    return {**payload, "receipt_hash": _canonical_hash(payload)}


# ---------------------------------------------------------------------------
# Gate evaluation (shared by analyzer and simulator)
# ---------------------------------------------------------------------------


def select_fixed_arm(
    counts: Mapping[str, Mapping[str, int]],
    *,
    disqualified_arms: Sequence[str] = (),
) -> str | None:
    """Deterministic fixed-arm tie-break: pooled, minimum template contrast, C.

    ``counts`` is ``{"form": {"E","C","R": int}, "distractor": {...}}`` of
    success counts (0..12 per template per arm). A C or R arm carrying a
    treatment-attributable severe veto is disqualified and can never advance;
    ``None`` is returned when both C and R are disqualified.
    """
    disqualified = set(disqualified_arms)
    candidates = [arm for arm in ("C", "R") if arm not in disqualified]

    def key(arm: str) -> tuple[int, int, int]:
        pooled = (counts["form"][arm] + counts["distractor"][arm]) - (
            counts["form"]["E"] + counts["distractor"]["E"]
        )
        min_contrast = min(
            counts["form"][arm] - counts["form"]["E"],
            counts["distractor"][arm] - counts["distractor"]["E"],
        )
        lexical = 0 if arm == "C" else -1
        return (pooled, min_contrast, lexical)

    if not candidates:
        return None
    return max(candidates, key=key)


def evaluate_fixed_gate(
    counts: Mapping[str, Mapping[str, int]],
    *,
    severe_vetos: Sequence[str] = (),
    disqualified_arms: Sequence[str] = (),
) -> dict[str, Any]:
    arm = select_fixed_arm(counts, disqualified_arms=disqualified_arms)
    if arm is None:
        return {
            "candidate_arm": None,
            "pooled_difference": None,
            "form_difference": None,
            "distractor_difference": None,
            "severe_vetos": list(severe_vetos),
            "disqualified_arms": list(disqualified_arms),
            "passed": False,
        }
    pooled = (counts["form"][arm] + counts["distractor"][arm]) - (
        counts["form"]["E"] + counts["distractor"]["E"]
    )
    form_diff = counts["form"][arm] - counts["form"]["E"]
    distractor_diff = counts["distractor"][arm] - counts["distractor"]["E"]
    passed = (
        pooled >= FIXED_POOLED_MIN
        and form_diff >= FIXED_TEMPLATE_MIN
        and distractor_diff >= FIXED_TEMPLATE_MIN
        and not severe_vetos
    )
    return {
        "candidate_arm": arm,
        "pooled_difference": pooled,
        "form_difference": form_diff,
        "distractor_difference": distractor_diff,
        "severe_vetos": list(severe_vetos),
        "disqualified_arms": list(disqualified_arms),
        "passed": passed,
    }


def evaluate_interaction_gate(
    counts: Mapping[str, Mapping[str, int]],
    *,
    severe_vetos: Sequence[str] = (),
    disqualified_arms: Sequence[str] = (),
) -> dict[str, Any]:
    vetos = list(severe_vetos)
    disqualified = set(disqualified_arms)
    if "C" in disqualified:
        vetos.append("form_lookup_disqualified")
    if "R" in disqualified:
        vetos.append("distractor_lookup_disqualified")
    if counts["form"]["C"] < counts["form"]["R"]:
        vetos.append("form_lookup_below_recovery")
    if counts["distractor"]["R"] < counts["distractor"]["C"]:
        vetos.append("distractor_lookup_below_execution")
    lookup_vs_empty = (counts["form"]["C"] + counts["distractor"]["R"]) - (
        counts["form"]["E"] + counts["distractor"]["E"]
    )
    fc_fr = counts["form"]["C"] - counts["form"]["R"]
    dr_dc = counts["distractor"]["R"] - counts["distractor"]["C"]
    fc_fe = counts["form"]["C"] - counts["form"]["E"]
    dr_de = counts["distractor"]["R"] - counts["distractor"]["E"]
    passed = (
        lookup_vs_empty >= INTERACTION_LOOKUP_MIN
        and fc_fr >= INTERACTION_FC_FR_MIN
        and dr_dc >= INTERACTION_DR_DC_MIN
        and fc_fe >= INTERACTION_FC_FE_MIN
        and dr_de >= INTERACTION_DR_DE_MIN
        and not vetos
    )
    return {
        "lookup_vs_empty": lookup_vs_empty,
        "fc_fr": fc_fr,
        "dr_dc": dr_dc,
        "fc_fe": fc_fe,
        "dr_de": dr_de,
        "vetos": vetos,
        "disqualified_arms": list(disqualified_arms),
        "passed": passed,
    }


def evaluate_decision(
    counts: Mapping[str, Mapping[str, int]],
    *,
    substrate_valid: bool,
    severe_vetos: Sequence[str] = (),
    disqualified_arms: Sequence[str] = (),
) -> dict[str, Any]:
    fixed = evaluate_fixed_gate(
        counts, severe_vetos=severe_vetos, disqualified_arms=disqualified_arms
    )
    interaction = evaluate_interaction_gate(
        counts, severe_vetos=severe_vetos, disqualified_arms=disqualified_arms
    )
    if not substrate_valid:
        decision = "invalid"
    elif interaction["passed"]:
        decision = "independent_interaction_replication"
    elif fixed["passed"]:
        decision = "independent_fixed_policy_replication"
    else:
        decision = "stop"
    return {"decision": decision, "fixed": fixed, "interaction": interaction}


# ---------------------------------------------------------------------------
# Ledger analyzer (never fits a router)
# ---------------------------------------------------------------------------


def _coerce_ledger(ledger: Any) -> list[dict[str, Any]]:
    if isinstance(ledger, list):
        rows = ledger
    elif isinstance(ledger, Mapping):
        if ledger.get("schema_version") != LEDGER_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported ledger schema: {ledger.get('schema_version')!r}"
            )
        cells = ledger.get("cells")
        if not isinstance(cells, list):
            raise ValueError("ledger must contain a cells list")
        rows = cells
    else:
        raise ValueError("ledger must be a list of cells or a ledger object")
    coerced: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(
                f"ledger row {index} must be a JSON object, got {type(row).__name__}"
            )
        coerced.append(dict(row))
    return coerced


_REQUIRED_CELL_FIELDS = (
    "cell_id",
    "panel_id",
    "task_id",
    "template",
    "difficulty",
    "arm",
    "success",
)


def _validate_cell_rows(
    cells: list[dict[str, Any]], expected_cells: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Validate row structure before any counting (descriptive errors only)."""
    observed_set: set[str] = set()
    duplicates: list[str] = []
    extras: list[str] = []
    invalid: list[dict[str, Any]] = []
    infrastructure_errors = 0
    for index, cell in enumerate(cells):
        missing = [field for field in _REQUIRED_CELL_FIELDS if field not in cell]
        if missing:
            invalid.append({"row": index, "reason": f"missing fields {missing}"})
            continue
        cell_id = cell["cell_id"]
        if not isinstance(cell_id, str) or not cell_id:
            invalid.append({"row": index, "reason": "cell_id must be a non-empty string"})
            continue
        if cell_id in observed_set:
            duplicates.append(cell_id)
        observed_set.add(cell_id)
        if cell_id not in expected_cells:
            extras.append(cell_id)
            continue
        expected = expected_cells[cell_id]
        for field in ("panel_id", "task_id", "template", "difficulty", "arm"):
            if cell.get(field) != expected[field]:
                invalid.append(
                    {
                        "row": index,
                        "cell_id": cell_id,
                        "reason": f"{field} does not match schedule",
                    }
                )
        if not isinstance(cell.get("success"), bool):
            invalid.append(
                {"row": index, "cell_id": cell_id, "reason": "success must be a boolean"}
            )
        if cell.get("arm") not in ARMS:
            invalid.append(
                {"row": index, "cell_id": cell_id, "reason": "arm must be E, C, or R"}
            )
        if cell.get("infrastructure_error") is True:
            infrastructure_errors += 1
    missing_cells = sorted(set(expected_cells) - observed_set)
    return {
        "duplicates": duplicates,
        "extras": extras,
        "invalid": invalid,
        "missing": missing_cells,
        "infrastructure_errors": infrastructure_errors,
    }


def _counts_from_cells(
    cells: list[dict[str, Any]], tasks: list[dict[str, Any]]
) -> dict[str, dict[str, int]]:
    template_by_id = {task["task_id"]: task["template"] for task in tasks}
    key = {
        "form_entry_validation": "form",
        "distractor_recovery": "distractor",
    }
    counts: dict[str, dict[str, int]] = {
        "form": {"E": 0, "C": 0, "R": 0},
        "distractor": {"E": 0, "C": 0, "R": 0},
    }
    for cell in cells:
        template = template_by_id[cell["task_id"]]
        bucket = key[template]
        if cell["success"]:
            counts[bucket][cell["arm"]] += 1
    return counts


def validate_substrate_receipt(
    receipt: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    """Validate an explicit execution/substrate receipt against the manifest.

    Fail closed: the receipt must be self-hashed, schema-bound, manifest-bound,
    confirm the isolated no-cache server argv hash, declare ``substrate_valid``,
    remain non-authorizing, and be evidence-bound to the lifecycle, slot-clear,
    proxy, tunnel, server, and active-service receipts (with recomputed hashes,
    exact counts, OFF argv identity, server alias, tunnel/proxy topology,
    active-service non-mutation, cache-invalidation freedom, and teardown).
    """
    _verify_embedded_hash(receipt, "receipt_hash")
    if receipt.get("schema_version") != SUBSTRATE_RECEIPT_SCHEMA_VERSION:
        raise ValueError("unsupported substrate receipt schema")
    if receipt.get("manifest_hash") != manifest.get("manifest_hash"):
        raise ValueError("substrate receipt manifest hash mismatch")
    if receipt.get("substrate_valid") is not True:
        raise ValueError("substrate receipt does not declare a valid substrate")
    if receipt.get("server_argv_hash_match") is not True:
        raise ValueError("substrate receipt does not confirm the server argv hash")
    if receipt.get("isolated_no_cache_server_identity") != manifest[
        "isolated_no_cache_server_identity"
    ]:
        raise ValueError("substrate receipt server identity drifted")
    if receipt.get("live_model_execution_authorized") is not False:
        raise ValueError("substrate receipt must remain non-authorizing")
    authorization_hash = receipt.get("authorization_hash")
    if not _is_hex_digest_64(authorization_hash):
        raise ValueError("substrate receipt authorization hash is missing/invalid")

    evidence = receipt.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("substrate receipt is missing its evidence binding")
    if evidence.get("authorization_hash") != authorization_hash:
        raise ValueError("substrate receipt evidence authorization hash mismatch")

    for key in (
        "server_receipt_hash",
        "tunnel_receipt_hash",
        "active_service_receipt_hash",
    ):
        value = evidence.get(key)
        if not (
            isinstance(value, str)
            and len(value) == 64
            and all(char in "0123456789abcdef" for char in value)
        ):
            raise ValueError(f"substrate receipt evidence {key} is missing/invalid")

    for key in ("slot_clear_receipt_hashes", "proxy_receipt_hashes"):
        hashes = evidence.get(key)
        if not isinstance(hashes, list) or len(hashes) != EXPECTED_CELLS:
            raise ValueError(f"substrate receipt evidence {key} count mismatch")
        if not all(
            isinstance(value, str)
            and len(value) == 64
            and all(char in "0123456789abcdef" for char in value)
            for value in hashes
        ):
            raise ValueError(f"substrate receipt evidence {key} carries a bad hash")

    if evidence.get("off_server_argv_hash") != manifest[
        "isolated_no_cache_server_identity"
    ]["server_argv_hash"]:
        raise ValueError("substrate receipt evidence OFF argv identity drifted")
    if evidence.get("server_alias") != RUN_MODEL_ALIAS:
        raise ValueError("substrate receipt evidence server alias drifted")
    if evidence.get("server_readiness_verified") is not True:
        raise ValueError("substrate receipt evidence did not verify server readiness")
    if evidence.get("tunnel_topology") != {
        "local_port": RUN_LOCAL_TUNNEL_PORT,
        "remote_target": RUN_TUNNEL_REMOTE_TARGET,
    }:
        raise ValueError("substrate receipt evidence tunnel topology drifted")
    if evidence.get("proxy_topology") != {
        "local_port": RUN_LOCAL_PROXY_PORT,
        "upstream": RUN_PROXY_UPSTREAM,
    }:
        raise ValueError("substrate receipt evidence proxy topology drifted")
    if evidence.get("active_service_unchanged") is not True:
        raise ValueError("substrate receipt evidence active service mutated")
    if evidence.get("cache_invalidation_free") is not True:
        raise ValueError("substrate receipt evidence has cache invalidations")
    if evidence.get("teardown_verified") is not True:
        raise ValueError("substrate receipt evidence did not verify teardown")
    if evidence.get("slot_action_dir_removed") is not True:
        raise ValueError("substrate receipt evidence did not remove the slot-action dir")
    if evidence.get("slot_action_dir_absence_verified") is not True:
        raise ValueError("substrate receipt evidence did not verify slot-action dir absence")
    for key in ("slot_action_dir_preparation_receipt", "slot_action_dir_removal_receipt"):
        obj = evidence.get(key)
        if not isinstance(obj, Mapping):
            raise ValueError(f"substrate receipt evidence {key} is missing")
    for key in ("generation_lease_acquire_receipt", "generation_lease_release_receipt"):
        obj = evidence.get(key)
        if not isinstance(obj, Mapping):
            raise ValueError(f"substrate receipt evidence {key} is missing")
    for key in (
        "generation_lease_local_acquire_receipt",
        "generation_lease_local_release_receipt",
    ):
        obj = evidence.get(key)
        if not isinstance(obj, Mapping):
            raise ValueError(f"substrate receipt evidence {key} is missing")
    if evidence.get("infrastructure_invalid_cells") != 0:
        raise ValueError("substrate receipt evidence has infrastructure-invalid cells")
    # The remote Git source_commit is a nullable diagnostic: require nonempty
    # only when it is present.
    source_commit = evidence.get("source_commit")
    if source_commit is not None and (
        not isinstance(source_commit, str) or not source_commit.strip()
    ):
        raise ValueError("substrate receipt evidence source commit must be non-empty or null")
    source_bundle_hash = evidence.get("source_bundle_hash")
    if not _is_hex_digest_64(source_bundle_hash):
        raise ValueError("substrate receipt evidence source bundle hash is missing/invalid")
    remote_project = str(manifest["remote_identity"]["project"])
    if not project_is_content_addressed(remote_project, str(source_bundle_hash)):
        raise ValueError(
            "substrate receipt remote project is not content-addressed by the source bundle hash"
        )


def _classify_vetoes(
    cells: Sequence[Mapping[str, Any]],
) -> tuple[list[str], set[str], bool]:
    """Classify bounded severe-veto codes into (codes, vetoed arms, gen-invalid).

    Generation-invalid codes (verifier false acceptance, cross-arm/task
    contamination, shared-substrate breach) invalidate the whole generation.
    A treatment-attributable severe action vetoes only its arm, except that an
    E-arm severe action also invalidates the generation (the comparator is
    compromised). ITT rows are never deleted; vetoes only gate advancement.
    """
    veto_codes: list[str] = []
    vetoed_arms: set[str] = set()
    generation_invalid = False
    for cell in cells:
        code = cell.get("severe_veto")
        if not isinstance(code, str) or not code:
            continue
        if code not in SEVERE_VETO_CODES:
            raise ValueError(f"unknown severe veto code: {code!r}")
        veto_codes.append(code)
        if code in GENERATION_INVALID_VETO_CODES:
            generation_invalid = True
        elif code in ARM_SEVERE_VETO_CODES:
            vetoed_arms.add(str(cell.get("arm", "")))
    if "E" in vetoed_arms:
        generation_invalid = True
        vetoed_arms.discard("E")
    return veto_codes, vetoed_arms, generation_invalid


def _analyze_ledger_impl(
    manifest: Mapping[str, Any],
    ledger: Any,
    *,
    registry: TreatmentRegistry | None = None,
    substrate_receipt: Mapping[str, Any] | None = None,
    substrate_valid: bool,
) -> dict[str, Any]:
    """Internal analysis implementation shared by the public entry points.

    Scientific counts, gate screens, and diagnostics are always reported. The
    top-level decision is ``invalid`` unless ``substrate_valid`` is True, which
    production callers may only reach via a validated ``substrate_receipt``.
    """
    # 1. Validate manifest schema/screen and, when available, full registry.
    _verify_embedded_hash(manifest, "manifest_hash")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported prompt-only manifest schema")
    if manifest.get("screen_id") != SCREEN_ID:
        raise ValueError("prompt-only screen mismatch")
    if registry is not None:
        validate_manifest(manifest, registry)
    else:
        for section in ("tasks", "panels", "cells"):
            if not isinstance(manifest.get(section), list):
                raise ValueError(f"manifest is missing the {section} section")
        if not isinstance(manifest.get("estimands"), Mapping):
            raise ValueError("manifest is missing the estimands section")
        if not isinstance(manifest.get("estimands", {}).get("legal_lookup"), Mapping):
            raise ValueError("manifest is missing the legal lookup")
        if not isinstance(manifest.get("isolated_no_cache_server_identity"), Mapping):
            raise ValueError("manifest is missing the server identity section")

    # 2. Resolve substrate validity (fail closed by default).
    substrate_receipt_hash: str | None = None
    if substrate_receipt is not None:
        validate_substrate_receipt(substrate_receipt, manifest)
        substrate_valid = True
        substrate_receipt_hash = substrate_receipt.get("receipt_hash")
    substrate_valid = bool(substrate_valid)

    # 3. Coerce and structurally validate rows before any counting.
    cells = _coerce_ledger(ledger)
    tasks = manifest["tasks"]
    expected_cells = {cell["cell_id"]: cell for cell in manifest["cells"]}
    validation = _validate_cell_rows(cells, expected_cells)
    if (
        validation["extras"]
        or validation["duplicates"]
        or validation["invalid"]
        or validation["missing"]
    ):
        raise ValueError(
            "ledger is not a complete, exact 72-row safe ledger: "
            f"extras={validation['extras']!r} "
            f"duplicates={validation['duplicates']!r} "
            f"invalid={validation['invalid']!r} "
            f"missing={validation['missing']!r}"
        )

    veto_codes, vetoed_arms, generation_invalid = _classify_vetoes(cells)
    severe_vetos: list[str] = list(veto_codes)
    if validation["infrastructure_errors"]:
        severe_vetos.append("infrastructure_error")
    disqualified_arms = sorted(vetoed_arms)

    # 4. Counts and estimates (now safe on validated rows).
    counts = _counts_from_cells(cells, tasks)
    task_by_id = {task["task_id"]: task for task in tasks}
    raw_task_vectors: list[dict[str, Any]] = []
    replica_means: dict[str, Any] = {}
    for task in tasks:
        task_cells = [cell for cell in cells if cell["task_id"] == task["task_id"]]
        arm_outcomes: dict[str, list[bool]] = {arm: [] for arm in ARMS}
        for cell in task_cells:
            arm_outcomes[cell["arm"]].append(bool(cell["success"]))
        arm_means = {
            arm: sum(values) / len(values) if values else None
            for arm, values in arm_outcomes.items()
        }
        raw_task_vectors.append(
            {
                "task_id": task["task_id"],
                "template": task["template"],
                "difficulty": task["difficulty"],
                "arm_outcomes": arm_outcomes,
                "arm_means": arm_means,
            }
        )
        replica_means[task["task_id"]] = arm_means

    finite_bank: dict[str, dict[str, float | None]] = {}
    template_arm_values: dict[str, dict[str, float | None]] = {}
    for template in ("form_entry_validation", "distractor_recovery"):
        bucket = "form" if template == "form_entry_validation" else "distractor"
        n = 12  # six tasks x two replicas
        template_arm_values[template] = {
            arm: counts[bucket][arm] / n for arm in ARMS
        }
        finite_bank[template] = dict(template_arm_values[template])

    lookup = manifest["estimands"]["legal_lookup"]
    lookup_arm_values = {
        template: template_arm_values[template][lookup[template]]
        for template in ("form_entry_validation", "distractor_recovery")
    }
    lookup_vs_empty = (
        counts["form"]["C"] + counts["distractor"]["R"]
    ) - (counts["form"]["E"] + counts["distractor"]["E"])

    fixed = evaluate_fixed_gate(
        counts, severe_vetos=severe_vetos, disqualified_arms=disqualified_arms
    )
    interaction = evaluate_interaction_gate(
        counts, severe_vetos=severe_vetos, disqualified_arms=disqualified_arms
    )
    decision = evaluate_decision(
        counts,
        substrate_valid=substrate_valid,
        severe_vetos=severe_vetos,
        disqualified_arms=disqualified_arms,
    )
    if generation_invalid:
        decision["decision"] = "invalid"
        decision["generation_invalid"] = True

    tool_calls = [
        cell.get("tool_calls")
        for cell in cells
        if isinstance(cell.get("tool_calls"), int)
    ]
    wall_seconds = [
        cell.get("wall_seconds")
        for cell in cells
        if isinstance(cell.get("wall_seconds"), (int, float))
    ]
    failure_codes = [
        cell.get("failure_code")
        for cell in cells
        if isinstance(cell.get("failure_code"), str) and cell.get("failure_code")
    ]

    payload: dict[str, Any] = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "manifest_hash": manifest["manifest_hash"],
        "ledger_sha256": _canonical_hash(cells),
        "substrate": {
            "substrate_valid": substrate_valid,
            "substrate_receipt_hash": substrate_receipt_hash,
            "decision_invalid_unless_substrate_validated": True,
        },
        "validation": {
            "cells": len(cells),
            "exact_cells": True,
            "reruns": 0,
            "extras": len(validation["extras"]),
            "duplicates": len(validation["duplicates"]),
            "missing": len(validation["missing"]),
            "infrastructure_errors": validation["infrastructure_errors"],
            "invalid": len(validation["invalid"]),
            "severe_veto_codes": veto_codes,
        },
        "counts": counts,
        "replica_means": replica_means,
        "template_arm_values": template_arm_values,
        "finite_bank": finite_bank,
        "raw_task_vectors": raw_task_vectors,
        "lookup_diagnostics": {
            "legal_lookup": dict(lookup),
            "template_identity_available_pre_action": manifest["estimands"][
                "template_identity_available_pre_action"
            ],
            "lookup_value_is_diagnostic_only": manifest["estimands"][
                "lookup_value_is_diagnostic_only"
            ],
            "lookup_arm_values": lookup_arm_values,
            "lookup_vs_empty": lookup_vs_empty,
        },
        "gates": {
            "fixed": fixed,
            "interaction": interaction,
        },
        "decision": decision["decision"],
        "cost_failures": {
            "tool_calls_total": sum(tool_calls),
            "tool_calls_by_arm": {
                arm: sum(
                    cell.get("tool_calls") or 0
                    for cell in cells
                    if cell.get("arm") == arm and isinstance(cell.get("tool_calls"), int)
                )
                for arm in ARMS
            },
            "wall_seconds_total": round(sum(wall_seconds), 3),
            "failure_total": len(failure_codes),
            "failure_codes": {
                code: failure_codes.count(code) for code in sorted(set(failure_codes))
            },
        },
        "scope_caveats": _SCOPE_CAVEATS,
        "severe_vetos": {
            "codes": veto_codes,
            "vetoed_arms": sorted(vetoed_arms),
            "disqualified_arms": disqualified_arms,
            "generation_invalid": generation_invalid,
            "itt_rows_retained": True,
        },
        "fitted_router": False,
        "decision_rule_fitted_model": "none",
        "gate_is_screen_not_efficacy_claim": True,
    }
    return {**payload, "analysis_hash": _canonical_hash(payload)}


def analyze_ledger(
    manifest: Mapping[str, Any],
    ledger: Any,
    *,
    registry: TreatmentRegistry | None = None,
    substrate_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze a complete 72-row safe ledger against a frozen manifest.

    Scientific counts, gate screens, and diagnostics are always reported, but
    the top-level decision is ``invalid`` unless the isolated no-cache
    substrate has been explicitly validated via a validated
    ``substrate_receipt``. There is no production flag that declares the
    substrate valid without such a receipt.
    """
    return _analyze_ledger_impl(
        manifest,
        ledger,
        registry=registry,
        substrate_receipt=substrate_receipt,
        substrate_valid=False,
    )


def analyze_ledger_test_only_valid_substrate(
    manifest: Mapping[str, Any],
    ledger: Any,
    *,
    registry: TreatmentRegistry | None = None,
) -> dict[str, Any]:
    """TEST-ONLY helper declaring a valid substrate without a receipt.

    Explicitly named so it can never be mistaken for a production path; the
    production :func:`analyze_ledger` has no such bypass.
    """
    return _analyze_ledger_impl(
        manifest,
        ledger,
        registry=registry,
        substrate_receipt=None,
        substrate_valid=True,
    )


# ---------------------------------------------------------------------------
# Deterministic joint simulator
# ---------------------------------------------------------------------------

_SCENARIOS: dict[str, dict[str, Any]] = {
    "null_flat": {
        "kind": "null",
        "mu": 0.0,
        "sigma_task": 0.0,
        "sigma_panel": 0.0,
        "sigma_eps": 0.5,
        "order_slope": 0.1,
        "arm_effects": {
            "form": {"E": 0.0, "C": 0.0, "R": 0.0},
            "distractor": {"E": 0.0, "C": 0.0, "R": 0.0},
        },
    },
    "null_heterogeneous": {
        "kind": "null",
        "mu": 0.0,
        "sigma_task": 1.5,
        "sigma_panel": 1.0,
        "sigma_eps": 0.6,
        "order_slope": 0.15,
        "arm_effects": {
            "form": {"E": 0.0, "C": 0.0, "R": 0.0},
            "distractor": {"E": 0.0, "C": 0.0, "R": 0.0},
        },
    },
    "alt_fixed": {
        "kind": "alternative",
        "mu": 0.0,
        "sigma_task": 1.5,
        "sigma_panel": 1.0,
        "sigma_eps": 0.6,
        "order_slope": 0.15,
        "arm_effects": {
            "form": {"E": 0.0, "C": 1.5, "R": 0.0},
            "distractor": {"E": 0.0, "C": 1.5, "R": 0.0},
        },
    },
    "alt_interaction": {
        "kind": "alternative",
        "mu": 0.0,
        "sigma_task": 1.5,
        "sigma_panel": 1.0,
        "sigma_eps": 0.6,
        "order_slope": 0.15,
        "arm_effects": {
            "form": {"E": 0.0, "C": 1.5, "R": 0.0},
            "distractor": {"E": 0.0, "C": 0.0, "R": 1.5},
        },
    },
}

REGISTERED_SCENARIOS = tuple(_SCENARIOS)
NULL_SCENARIOS = tuple(name for name, spec in _SCENARIOS.items() if spec["kind"] == "null")


def _wilson_upper_95(successes: int, trials: int) -> float:
    if trials <= 0:
        return 1.0
    if successes >= trials:
        return 1.0
    z = _Z_95_ONETAIL
    phat = successes / trials
    z2 = z * z
    denominator = 1 + z2 / trials
    center = (phat + z2 / (2 * trials)) / denominator
    halfwidth = z * math.sqrt((phat * (1 - phat) + z2 / (4 * trials)) / trials) / denominator
    return min(1.0, center + halfwidth)


def _bank_structure(
    manifest: Mapping[str, Any],
) -> tuple[list[str], list[tuple[int, list[tuple[str, int]]]]]:
    tasks = manifest["tasks"]
    task_index = {task["task_id"]: index for index, task in enumerate(tasks)}
    templates = [
        "form" if task["template"] == "form_entry_validation" else "distractor"
        for task in tasks
    ]
    panels: list[tuple[int, list[tuple[str, int]]]] = []
    for panel in manifest["panels"]:
        arms_positions = [
            (arm, position) for position, arm in enumerate(panel["execution_order"])
        ]
        panels.append((task_index[panel["task_id"]], arms_positions))
    return templates, panels


def _simulate_bank(
    rng: random.Random,
    scenario: Mapping[str, Any],
    templates: Sequence[str],
    panels: Sequence[tuple[int, list[tuple[str, int]]]],
) -> dict[str, dict[str, int]]:
    task_latent = [rng.gauss(0.0, scenario["sigma_task"]) for _ in templates]
    panel_latent = [rng.gauss(0.0, scenario["sigma_panel"]) for _ in panels]
    counts: dict[str, dict[str, int]] = {
        "form": {"E": 0, "C": 0, "R": 0},
        "distractor": {"E": 0, "C": 0, "R": 0},
    }
    effects = scenario["arm_effects"]
    for panel_index, (task_index, arms_positions) in enumerate(panels):
        template = templates[task_index]
        baseline = scenario["mu"] + task_latent[task_index] + panel_latent[panel_index]
        for arm, position in arms_positions:
            utility = (
                baseline
                + effects[template][arm]
                - scenario["order_slope"] * position
                + rng.gauss(0.0, scenario["sigma_eps"])
            )
            if utility > 0:
                counts[template][arm] += 1
    return counts


def _scenario_report(
    manifest: Mapping[str, Any],
    scenario_name: str,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    scenario = _SCENARIOS[scenario_name]
    templates, panels = _bank_structure(manifest)
    rng = random.Random(seed)
    decision_counts = {
        "stop": 0,
        "independent_fixed_policy_replication": 0,
        "independent_interaction_replication": 0,
        "invalid": 0,
    }
    for _ in range(draws):
        counts = _simulate_bank(rng, scenario, templates, panels)
        result = evaluate_decision(counts, substrate_valid=True, severe_vetos=())
        decision_counts[result["decision"]] += 1
    advance = (
        decision_counts["independent_fixed_policy_replication"]
        + decision_counts["independent_interaction_replication"]
    )
    interaction = decision_counts["independent_interaction_replication"]
    is_null = scenario["kind"] == "null"
    return {
        "scenario": scenario_name,
        "kind": scenario["kind"],
        "draws": draws,
        "seed": seed,
        "decision_counts": decision_counts,
        "advance_rate": advance / draws,
        "interaction_rate": interaction / draws,
        "advance_rate_interpretation": (
            "false_advance_under_null" if is_null else "power"
        ),
        "interaction_rate_interpretation": (
            "false_interaction_under_null" if is_null else "power"
        ),
        "advance_upper_95": _wilson_upper_95(advance, draws),
        "interaction_upper_95": _wilson_upper_95(interaction, draws),
    }


def run_simulator(
    manifest: Mapping[str, Any],
    *,
    draws: int = MIN_FREEZE_BANKS,
    seed: int = SIMULATOR_SEED,
    scenarios: Sequence[str] = REGISTERED_SCENARIOS,
) -> dict[str, Any]:
    _verify_embedded_hash(manifest, "manifest_hash")
    for scenario in scenarios:
        if scenario not in _SCENARIOS:
            raise ValueError(f"unregistered simulator scenario: {scenario!r}")
    reports = [
        _scenario_report(manifest, name, draws=draws, seed=seed) for name in scenarios
    ]
    freeze_met = check_freeze_requirements(reports)
    null_reports = [report for report in reports if report["kind"] == "null"]
    payload: dict[str, Any] = {
        "schema_version": SIMULATOR_REPORT_SCHEMA_VERSION,
        "manifest_hash": manifest["manifest_hash"],
        "seed": seed,
        "draws_per_scenario": draws,
        "min_freeze_banks": MIN_FREEZE_BANKS,
        "freeze_requirements_met": freeze_met,
        "freeze_thresholds": {
            "false_advance_max": FREEZE_FALSE_ADVANCE_MAX,
            "false_interaction_max": FREEZE_FALSE_INTERACTION_MAX,
            "screening_limits_are_choices_not_guarantees": True,
        },
        "operating_characteristic": {
            "screening_limits_are_choices_not_guarantees": True,
            "null_scenarios": [
                {
                    "scenario": report["scenario"],
                    "false_advance_rate": report["advance_rate"],
                    "false_advance_upper_95": report["advance_upper_95"],
                    "false_interaction_rate": report["interaction_rate"],
                    "false_interaction_upper_95": report["interaction_upper_95"],
                }
                for report in null_reports
            ],
            "interpretation": (
                "Monte Carlo operating characteristic of the complete decision "
                "rule under the registered null scenarios. The 0.10 overall and "
                "0.05 interaction upper limits are prespecified screening "
                "choices, not guarantees of any specific error rate in the live "
                "experiment."
            ),
        },
        "scenarios": reports,
    }
    return {**payload, "report_hash": _canonical_hash(payload)}


def check_freeze_requirements(reports: Sequence[Mapping[str, Any]]) -> bool:
    for report in reports:
        if report.get("kind") != "null":
            continue
        if report.get("advance_upper_95", 1.0) > FREEZE_FALSE_ADVANCE_MAX:
            return False
        if report.get("interaction_upper_95", 1.0) > FREEZE_FALSE_INTERACTION_MAX:
            return False
    return True


def validate_simulator_report(
    report: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    expected_draws: int,
    required_scenarios: Sequence[str] = REGISTERED_SCENARIOS,
    require_freeze_met: bool = False,
) -> bool:
    """Validate a simulator report structurally rather than trusting a boolean.

    Checks schema, self-hash, manifest binding, draw count (and the 200k floor
    when freezing), scenario membership/order, and recomputes the freeze
    thresholds from the raw per-scenario bounds. Returns whether the freeze
    requirements are met.
    """
    _verify_embedded_hash(report, "report_hash")
    if report.get("schema_version") != SIMULATOR_REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported simulator report schema")
    if report.get("manifest_hash") != manifest.get("manifest_hash"):
        raise ValueError("simulator report manifest hash mismatch")
    if report.get("draws_per_scenario") != expected_draws:
        raise ValueError("simulator report draws mismatch")
    scenarios = report.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError("simulator report scenarios must be a list")
    observed = [scenario.get("scenario") for scenario in scenarios]
    if tuple(observed) != tuple(required_scenarios):
        raise ValueError(
            f"simulator report scenarios mismatch: {observed!r} != {list(required_scenarios)!r}"
        )
    for scenario in scenarios:
        if not isinstance(scenario, Mapping):
            raise ValueError("simulator scenario must be an object")
        if scenario.get("draws") != expected_draws:
            raise ValueError("simulator scenario draws mismatch")
        if scenario.get("kind") not in {"null", "alternative"}:
            raise ValueError("simulator scenario kind is invalid")
        for field in ("advance_upper_95", "interaction_upper_95"):
            if not isinstance(scenario.get(field), (int, float)):
                raise ValueError(f"simulator scenario {field} must be numeric")
        # Recompute rates and Wilson upper bounds from the raw decision counts.
        decision_counts = scenario.get("decision_counts")
        if not isinstance(decision_counts, Mapping):
            raise ValueError("simulator scenario decision_counts must be an object")
        try:
            stop = int(decision_counts.get("stop", 0))
            fixed = int(decision_counts.get("independent_fixed_policy_replication", 0))
            interaction = int(
                decision_counts.get("independent_interaction_replication", 0)
            )
            invalid = int(decision_counts.get("invalid", 0))
        except (TypeError, ValueError) as error:
            raise ValueError("simulator scenario decision counts must be integers") from error
        if stop + fixed + interaction + invalid != expected_draws:
            raise ValueError(
                "simulator scenario decision counts do not sum to the draws"
            )
        advance = fixed + interaction
        expected_advance_rate = advance / expected_draws
        expected_interaction_rate = interaction / expected_draws
        if not math.isclose(scenario.get("advance_rate"), expected_advance_rate, abs_tol=1e-12):
            raise ValueError("simulator scenario advance_rate drifted from decision counts")
        if not math.isclose(scenario.get("interaction_rate"), expected_interaction_rate, abs_tol=1e-12):
            raise ValueError("simulator scenario interaction_rate drifted from decision counts")
        expected_advance_upper = _wilson_upper_95(advance, expected_draws)
        expected_interaction_upper = _wilson_upper_95(interaction, expected_draws)
        if not math.isclose(scenario.get("advance_upper_95"), expected_advance_upper, abs_tol=1e-12):
            raise ValueError("simulator scenario advance_upper_95 drifted from decision counts")
        if not math.isclose(scenario.get("interaction_upper_95"), expected_interaction_upper, abs_tol=1e-12):
            raise ValueError("simulator scenario interaction_upper_95 drifted from decision counts")
    thresholds = report.get("freeze_thresholds")
    if not isinstance(thresholds, Mapping):
        raise ValueError("simulator report freeze thresholds are missing")
    if thresholds.get("false_advance_max") != FREEZE_FALSE_ADVANCE_MAX:
        raise ValueError("simulator report false-advance threshold drifted")
    if thresholds.get("false_interaction_max") != FREEZE_FALSE_INTERACTION_MAX:
        raise ValueError("simulator report false-interaction threshold drifted")
    if thresholds.get("screening_limits_are_choices_not_guarantees") is not True:
        raise ValueError("simulator report must mark screening limits as choices")
    freeze_met = check_freeze_requirements(scenarios)
    if report.get("freeze_requirements_met") != freeze_met:
        raise ValueError("simulator report freeze flag is inconsistent")
    if require_freeze_met:
        if expected_draws < MIN_FREEZE_BANKS:
            raise ValueError(
                f"freezing requires at least {MIN_FREEZE_BANKS} simulator banks per "
                f"scenario, got {expected_draws}"
            )
        if not freeze_met:
            raise ValueError("simulator report fails freeze thresholds")
    return freeze_met


# ---------------------------------------------------------------------------
# Local no-model preflight
# ---------------------------------------------------------------------------


def build_local_preflight(
    manifest: Mapping[str, Any],
    registry: TreatmentRegistry,
    project_root: str | Path,
    run_root: str | Path,
    *,
    pi_executable: str = "pi",
    simulator_draws: int = MIN_FREEZE_BANKS,
    simulator_report: Mapping[str, Any] | None = None,
    exclude_paths: Sequence[str | Path] = (),
    pi_conformance_receipt: Mapping[str, Any] | None = None,
    run_pi_gate: bool = False,
) -> dict[str, Any]:
    validate_manifest(manifest, registry)
    root = Path(project_root).expanduser().resolve()
    schedule = {
        "tasks": manifest["tasks"],
        "panels": manifest["panels"],
        "cells": manifest["cells"],
    }
    validate_schedule(schedule)
    collisions = assert_no_collisions(
        manifest, run_root, exclude_paths=exclude_paths
    )
    task_commitments = _build_task_commitments(manifest)
    command_receipt = build_command_arm_receipt(
        manifest, registry, root, pi_executable=pi_executable
    )
    report = (
        run_simulator(manifest, draws=simulator_draws)
        if simulator_report is None
        else dict(simulator_report)
    )
    # The canonical source bundle is the authoritative prompt-only source
    # identity; the remote project must be content-addressed by its hash.
    bundle_manifest = build_source_bundle_manifest(root)
    bundle_hash = str(bundle_manifest["bundle_hash"])
    remote_project = str(manifest["remote_identity"]["project"])
    if not project_is_content_addressed(remote_project, bundle_hash):
        raise ValueError(
            "manifest remote project is not content-addressed by the "
            "authoritative source bundle hash"
        )
    # No-real-model Pi conformance gate: run it now when requested, or bind an
    # already-produced receipt; the default stays model-free so offline
    # validation never invokes Pi. Authorization flows fail closed on a
    # preflight without a valid conformance receipt.
    conformance: dict[str, Any] | None = None
    if pi_conformance_receipt is not None:
        validate_pi_conformance_receipt(pi_conformance_receipt)
        conformance = dict(pi_conformance_receipt)
    elif run_pi_gate:
        conformance = run_pi_conformance(pi_executable)
    payload: dict[str, Any] = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "manifest_hash": manifest["manifest_hash"],
        "registry_hash": registry.registry_hash,
        "source_tree_hash": source_tree_hash(root),
        "source_bundle_hash": bundle_hash,
        "source_bundle_manifest": bundle_manifest,
        "schedule": {
            "tasks": len(manifest["tasks"]),
            "panels": len(manifest["panels"]),
            "cells": len(manifest["cells"]),
            "schedule_seed": manifest["schedule_seed"],
        },
        "collision_scan": collisions,
        "registry_identity": {
            "registry_hash": registry.registry_hash,
            "arm_only_delta": manifest["arm_only_delta"],
        },
        "cache_off_identity": manifest["isolated_no_cache_server_identity"],
        "generated_tasks_sha256": _canonical_hash({"tasks": task_commitments}),
        "generated_tasks": task_commitments,
        "command_arm_receipt": command_receipt,
        "simulator_report": report,
        "pi_conformance": conformance,
        "held_templates_consumed": False,
        "policy_leakage_markers_found": 0,
        "no_model_invoked": True,
        "live_runtime_checked": False,
        "live_model_execution_authorized": False,
    }
    return {**payload, "preflight_hash": _canonical_hash(payload)}


def validate_local_preflight(
    preflight: Mapping[str, Any],
    manifest: Mapping[str, Any],
    registry: TreatmentRegistry,
    project_root: str | Path,
    run_root: str | Path,
    *,
    pi_executable: str = "pi",
    simulator_draws: int = MIN_FREEZE_BANKS,
    exclude_paths: Sequence[str | Path] = (),
    pi_conformance_receipt: Mapping[str, Any] | None = None,
    require_pi_conformance: bool = False,
) -> None:
    _verify_embedded_hash(preflight, "preflight_hash")
    if preflight.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        raise ValueError("unsupported prompt-only preflight schema")
    if preflight.get("manifest_hash") != manifest.get("manifest_hash"):
        raise ValueError("prompt-only preflight manifest hash mismatch")
    if preflight.get("registry_hash") != registry.registry_hash:
        raise ValueError("prompt-only preflight registry hash mismatch")
    root = Path(project_root).expanduser().resolve()
    if preflight.get("source_tree_hash") != source_tree_hash(root):
        raise ValueError("prompt-only preflight source tree hash mismatch")
    if preflight.get("source_bundle_manifest") != build_source_bundle_manifest(root):
        raise ValueError("prompt-only preflight source bundle manifest drifted")
    bundle_hash = preflight.get("source_bundle_hash")
    if not _is_hex_digest_64(bundle_hash):
        raise ValueError("prompt-only preflight source bundle hash is invalid")
    if not project_is_content_addressed(
        str(manifest["remote_identity"]["project"]), str(bundle_hash)
    ):
        raise ValueError(
            "manifest remote project is not content-addressed by the "
            "preflight source bundle hash"
        )
    schedule = {
        "tasks": manifest["tasks"],
        "panels": manifest["panels"],
        "cells": manifest["cells"],
    }
    validate_schedule(schedule)
    if preflight.get("schedule") != {
        "tasks": len(manifest["tasks"]),
        "panels": len(manifest["panels"]),
        "cells": len(manifest["cells"]),
        "schedule_seed": manifest["schedule_seed"],
    }:
        raise ValueError("prompt-only preflight schedule receipt drifted")
    if preflight.get("collision_scan") != assert_no_collisions(
        manifest, run_root, exclude_paths=exclude_paths
    ):
        raise ValueError("prompt-only preflight collision receipt drifted")
    if preflight.get("registry_identity") != {
        "registry_hash": registry.registry_hash,
        "arm_only_delta": manifest["arm_only_delta"],
    }:
        raise ValueError("prompt-only preflight registry identity drifted")
    if preflight.get("cache_off_identity") != manifest["isolated_no_cache_server_identity"]:
        raise ValueError("prompt-only preflight cache OFF identity drifted")
    task_commitments = _build_task_commitments(manifest)
    if preflight.get("generated_tasks_sha256") != _canonical_hash(
        {"tasks": task_commitments}
    ):
        raise ValueError("prompt-only preflight task commitment hash mismatch")
    if preflight.get("generated_tasks") != task_commitments:
        raise ValueError("prompt-only preflight task commitments drifted")
    command_receipt = build_command_arm_receipt(
        manifest, registry, root, pi_executable=pi_executable
    )
    if preflight.get("command_arm_receipt") != command_receipt:
        raise ValueError("prompt-only preflight command receipt drifted")
    simulator_report = preflight.get("simulator_report")
    if not isinstance(simulator_report, Mapping):
        raise ValueError("prompt-only preflight simulator report is missing")
    validate_simulator_report(
        simulator_report,
        manifest,
        expected_draws=simulator_draws,
        require_freeze_met=False,
    )
    if preflight.get("held_templates_consumed") is not False:
        raise ValueError("prompt-only preflight consumed held templates")
    if preflight.get("policy_leakage_markers_found") != 0:
        raise ValueError("prompt-only preflight found policy leakage")
    if preflight.get("no_model_invoked") is not True:
        raise ValueError("prompt-only preflight must not invoke a model")
    if preflight.get("live_model_execution_authorized") is not False:
        raise ValueError("prompt-only preflight must not authorize model execution")
    # No-real-model Pi conformance receipt: revalidate it structurally (never
    # rerun the gate here); a production preflight must carry one.
    embedded = preflight.get("pi_conformance")
    if pi_conformance_receipt is not None:
        validate_pi_conformance_receipt(pi_conformance_receipt)
        if embedded != dict(pi_conformance_receipt):
            raise ValueError("prompt-only preflight pi conformance receipt drifted")
    elif embedded is not None:
        validate_pi_conformance_receipt(embedded)
    elif require_pi_conformance:
        raise ValueError(
            "prompt-only preflight is missing the pi conformance receipt"
        )


# ---------------------------------------------------------------------------
# Freeze (immutable writes, non-authorizing)
# ---------------------------------------------------------------------------


def freeze_prompt_only_artifacts(
    registry_output: str | Path,
    manifest_output: str | Path,
    remote_identity: Mapping[str, Any],
    *,
    project_root: str | Path,
    run_root: str | Path = ".runs",
    pi_executable: str = "pi",
    simulator_draws: int = MIN_FREEZE_BANKS,
    preflight_output: str | Path | None = None,
    declare_template_identity_available: bool = False,
    run_pi_conformance: bool = False,
) -> dict[str, Any]:
    if simulator_draws < MIN_FREEZE_BANKS:
        raise ValueError(
            f"freezing requires at least {MIN_FREEZE_BANKS} simulator banks per "
            f"scenario, got {simulator_draws}"
        )
    registry = build_prompt_only_registry()
    registry_path = Path(registry_output).expanduser().resolve()
    manifest_path = Path(manifest_output).expanduser().resolve()
    # The remote project is content-addressed by the authoritative source bundle
    # hash: derive it (idempotently) before building the manifest so the frozen
    # manifest binds the exact staged bundle path.
    bundle_hash = source_bundle_manifest_hash(project_root)
    project = str(remote_identity["project"])
    if not project_is_content_addressed(project, bundle_hash):
        project = content_addressed_project_path(project, bundle_hash)
    addressed = {**dict(remote_identity), "project": project}
    manifest = build_manifest(
        registry,
        addressed,
        registry_file=registry_path.name,
        declare_template_identity_available=declare_template_identity_available,
    )
    validate_manifest(manifest, registry)
    # Exclude exactly the bound frozen artifacts themselves when they share the
    # scanned run root, so the freeze scan and the execution re-scan agree.
    exclude = _derive_bound_artifact_exclusions(
        run_root,
        registry_path,
        manifest_path,
        preflight_output,
    )
    assert_no_collisions(manifest, run_root, exclude_paths=exclude)
    simulator_report = run_simulator(manifest, draws=simulator_draws)
    validate_simulator_report(
        simulator_report,
        manifest,
        expected_draws=simulator_draws,
        require_freeze_met=True,
    )
    _write_immutable_json(registry_path, registry.to_dict())
    _write_immutable_json(manifest_path, manifest)
    report: dict[str, Any] = {
        "registry": str(registry_path),
        "registry_hash": registry.registry_hash,
        "manifest": str(manifest_path),
        "manifest_hash": manifest["manifest_hash"],
        "tasks": EXPECTED_TASKS,
        "panels": EXPECTED_PANELS,
        "cells": EXPECTED_CELLS,
        "live_model_execution_authorized": False,
        "cache_canary_implied_passed": False,
        "freeze_requirements_met": simulator_report["freeze_requirements_met"],
    }
    if preflight_output is not None:
        preflight = build_local_preflight(
            manifest,
            registry,
            project_root,
            run_root,
            pi_executable=pi_executable,
            simulator_draws=simulator_draws,
            simulator_report=simulator_report,
            exclude_paths=exclude,
            run_pi_gate=run_pi_conformance,
        )
        validate_local_preflight(
            preflight,
            manifest,
            registry,
            project_root,
            run_root,
            pi_executable=pi_executable,
            simulator_draws=simulator_draws,
            exclude_paths=exclude,
            require_pi_conformance=run_pi_conformance,
        )
        preflight_path = Path(preflight_output).expanduser().resolve()
        _write_immutable_json(preflight_path, preflight)
        report["preflight"] = str(preflight_path)
        report["preflight_hash"] = preflight["preflight_hash"]
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyreplab-m3-prompt-only-pilot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser("simulate")
    simulate.add_argument("--manifest", required=True)
    simulate.add_argument("--draws", type=int, default=200)
    simulate.add_argument("--seed", type=int, default=SIMULATOR_SEED)
    simulate.add_argument("--scenario", action="append", default=None)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--registry-output", required=True)
    freeze.add_argument("--manifest-output", required=True)
    freeze.add_argument("--host", default=os.environ.get("PYREPLAB_HARNESS_HOST", "ubuntu-local"))
    freeze.add_argument("--remote-project", required=True)
    freeze.add_argument("--remote-run-root", required=True)
    freeze.add_argument("--remote-python", default="python3")
    freeze.add_argument("--root", required=True)
    freeze.add_argument("--run-root", default=".runs")
    freeze.add_argument("--simulator-draws", type=int, default=MIN_FREEZE_BANKS)
    freeze.add_argument("--preflight-output", default=None)
    freeze.add_argument("--declare-template-identity-available", action="store_true")
    freeze.add_argument(
        "--run-pi-conformance",
        action="store_true",
        help=(
            "run the no-real-model Pi conformance gate (pinned Pi binary, "
            "isolated PI_CODING_AGENT_DIR, PI_OFFLINE=1) and bind its receipt "
            "into the local preflight"
        ),
    )

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--registry", required=True)
    preflight.add_argument("--manifest", required=True)
    preflight.add_argument("--root", required=True)
    preflight.add_argument("--run-root", default=".runs")
    preflight.add_argument("--output", required=True)
    preflight.add_argument("--simulator-draws", type=int, default=MIN_FREEZE_BANKS)
    preflight.add_argument("--pi", default=os.environ.get("PYREPLAB_PI", "pi"))
    preflight.add_argument(
        "--run-pi-conformance",
        action="store_true",
        help=(
            "run the no-real-model Pi conformance gate and bind its receipt "
            "into the local preflight"
        ),
    )

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--manifest", required=True)
    analyze.add_argument("--ledger", required=True)
    analyze.add_argument("--registry", required=True)
    analyze.add_argument("--substrate-receipt", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--registry", required=True)
    validate.add_argument("--manifest", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "simulate":
        manifest = _load_json(args.manifest)
        scenarios = tuple(args.scenario) if args.scenario else REGISTERED_SCENARIOS
        report = run_simulator(
            manifest, draws=args.draws, seed=args.seed, scenarios=scenarios
        )
    elif args.command == "freeze":
        report = freeze_prompt_only_artifacts(
            args.registry_output,
            args.manifest_output,
            {
                "host": args.host,
                "project": args.remote_project,
                "run_root": args.remote_run_root,
                "python": args.remote_python,
            },
            project_root=args.root,
            run_root=args.run_root,
            simulator_draws=args.simulator_draws,
            preflight_output=args.preflight_output,
            declare_template_identity_available=args.declare_template_identity_available,
            run_pi_conformance=args.run_pi_conformance,
        )
    elif args.command == "preflight":
        registry = TreatmentRegistry.load(args.registry)
        manifest = _load_json(args.manifest)
        preflight = build_local_preflight(
            manifest,
            registry,
            args.root,
            args.run_root,
            pi_executable=args.pi,
            simulator_draws=args.simulator_draws,
            run_pi_gate=args.run_pi_conformance,
        )
        validate_local_preflight(
            preflight,
            manifest,
            registry,
            args.root,
            args.run_root,
            pi_executable=args.pi,
            simulator_draws=args.simulator_draws,
            require_pi_conformance=args.run_pi_conformance,
        )
        _write_immutable_json(Path(args.output).expanduser().resolve(), preflight)
        report = {
            "preflight": str(Path(args.output).expanduser().resolve()),
            "preflight_hash": preflight["preflight_hash"],
            "live_model_execution_authorized": False,
        }
    elif args.command == "analyze":
        manifest = _load_json(args.manifest)
        registry = TreatmentRegistry.load(args.registry)
        ledger = _load_json(args.ledger)
        substrate_receipt = _load_json(args.substrate_receipt)
        report = analyze_ledger(
            manifest,
            ledger,
            registry=registry,
            substrate_receipt=substrate_receipt,
        )
    else:
        registry = TreatmentRegistry.load(args.registry)
        manifest = _load_json(args.manifest)
        validate_manifest(manifest, registry)
        report = {
            "valid": True,
            "registry_hash": registry.registry_hash,
            "manifest_hash": manifest["manifest_hash"],
        }
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ADMITTED_TOOL_CALLS_PER_CELL",
    "AGGREGATE_WALL_SECONDS",
    "ANALYSIS_SCHEMA_VERSION",
    "ARMS",
    "ARM_PERMUTATIONS",
    "ARM_PROMPTS",
    "ARM_SEVERE_VETO_CODES",
    "COMMAND_RECEIPT_SCHEMA_VERSION",
    "DUMMY_PROVIDER_API_KEY",
    "EXECUTION_DISCIPLINE_PROMPT",
    "EXPECTED_CELLS",
    "EXPECTED_PANELS",
    "EXPECTED_TASKS",
    "FREEZE_FALSE_ADVANCE_MAX",
    "FREEZE_FALSE_INTERACTION_MAX",
    "GENERATION_INVALID_VETO_CODES",
    "LEDGER_SCHEMA_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "MAX_SCAN_FILE_BYTES",
    "MIN_FREEZE_BANKS",
    "NULL_SCENARIOS",
    "PER_CELL_WALL_SECONDS",
    "PI_CONFORMANCE_SCHEMA_VERSION",
    "PREFLIGHT_SCHEMA_VERSION",
    "PROMPT_TEMPLATES",
    "PROVIDER_BACKED_TURNS_PER_CELL",
    "RECOVERY_DISCIPLINE_PROMPT",
    "REGISTERED_SCENARIOS",
    "RUN_LOCAL_PROXY_PORT",
    "RUN_LOCAL_TUNNEL_PORT",
    "RUN_MODEL_ALIAS",
    "RUN_PI_BASE_URL",
    "RUN_PROVIDER",
    "RUN_PROXY_UPSTREAM",
    "RUN_REMOTE_SERVER_BASE_URL",
    "RUN_REMOTE_SERVER_PORT",
    "RUN_TUNNEL_REMOTE_TARGET",
    "SAMPLING_SEED_START",
    "SCHEDULE_SEED",
    "SCREEN_ID",
    "SEVERE_VETO_CODES",
    "SIMULATOR_REPORT_SCHEMA_VERSION",
    "SLOT_ACTION_DIRECTORY",
    "SLOT_ACTION_DIRECTORY_MODE",
    "SOURCE_BUNDLE_SCHEMA_VERSION",
    "SUBSTRATE_RECEIPT_SCHEMA_VERSION",
    "TASK_SEED_START",
    "TASK_SPLIT",
    "TASK_ROLE",
    "TOOL_ATTEMPTS_PER_CELL",
    "TREATMENT_VERSION",
    "V8_FAILURE_HASH",
    "V8_TRANSPORT_TOTAL_SECONDS",
    "V8_TURN_LATENCIES_SECONDS",
    "V10_COMPLETED_CELL_MODEL_WALL_SECONDS",
    "V10_COMPLETED_CELL_RECORD_HASH",
    "V10_FAILURE_HASH",
    "V9_COMPLETED_CELL_MODEL_WALL_SECONDS",
    "V9_COMPLETED_CELL_RECORD_HASH",
    "V9_FAILURE_HASH",
    "WALL_BUDGET_AMENDMENT_SCHEMA_VERSION",
    "analyze_ledger",
    "analyze_ledger_test_only_valid_substrate",
    "assert_no_collisions",
    "build_cache_off_server_binding",
    "build_command_arm_receipt",
    "build_frozen_models_json",
    "build_local_preflight",
    "build_manifest",
    "build_pi_conformance_receipt",
    "build_prompt_only_registry",
    "build_schedule",
    "build_source_bundle_manifest",
    "build_wall_budget_amendment",
    "check_freeze_requirements",
    "content_addressed_project_path",
    "dummy_api_key_binding",
    "evaluate_decision",
    "evaluate_fixed_gate",
    "evaluate_interaction_gate",
    "freeze_prompt_only_artifacts",
    "models_json_sha256",
    "prepare_frozen_models_json",
    "project_is_content_addressed",
    "run_pi_conformance",
    "run_simulator",
    "scan_collisions",
    "select_fixed_arm",
    "source_bundle_manifest_hash",
    "validate_frozen_models_json_config",
    "validate_local_preflight",
    "validate_manifest",
    "validate_pi_conformance_receipt",
    "validate_schedule",
    "validate_simulator_report",
    "validate_substrate_receipt",
    "write_frozen_models_json",
]
