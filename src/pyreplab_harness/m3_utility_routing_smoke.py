"""M3 utility-routing Stage-B contract, live runner, exporter, and gate.

This module implements Section 6 of
``notes/m3-utility-routing-smoke-plan.md`` end to end. Manifest freezing and
authorization remain outcome-blind. The explicit ``run`` path executes the
authorized crossed panel sequentially, records a restricted raw ledger,
activates only pre-frozen whole-block contingencies, and emits leakage-safe
attempt rows. Outcome analysis is a separate post-run operation.

What this module provides:

* ``default_spec(...)`` -- a practical spec builder that loads the production
  policy registry and the byte-verified Stage-A evidence (manifest + gate
  report) and binds their hashes.
* ``build_manifest(...)`` -- derives the exact 24-task Stage-B design from
  ``routing_fixtures.build_stage_b_design`` with the frozen seed/block seeds,
  generates each public fixture, runs the neutral structural probe, derives
  bounded request features, routes every task with the *frozen* Stage-A
  combined and prompt-only heuristics, and binds source/prompt/request/probe/
  oracle/private-coordinate hashes plus a self-hashed route receipt.  The
  manifest never exposes HTML, page text, oracle values, or nonces.
* ``validate_manifest(...)`` -- fail-closed validation. The manifest is
  self-contained: every task, route receipt, panel, and attempt is rebuilt
  from the manifest's own spec/seed/block seeds and compared exactly.  When
  the production registry and/or Stage-A evidence are supplied, their byte
  and embedded hash bindings are verified as well.
* ``build_authorization(...)`` / ``validate_authorization(...)`` -- load the
  Stage-A manifest and gate report, run ``m3_routing_probe_gate``
  ``validate_manifest`` / ``validate_gate_report``, require ``probe_pass``,
  check the exact byte and embedded hash bindings against the Stage-B
  manifest, and emit a self-hashed authorization record binding the Stage-B
  manifest hash.
* ``run_stage_b(...)`` -- runtime-preflighted sequential execution with exact
  planned attempt IDs, append-only raw records, and whole-block contingency.
* ``export_safe_attempts(...)`` -- whitelist-only raw-to-safe conversion.
* ``analyze_stage_b(...)`` -- pure post-run outcome analysis and gate report.

Frozen protocol facts (plan Section 6):

* 24 tasks in two independently seeded blocks; three tasks per stratum per
  block (one easy / one medium / one hard); six table-preferred and six
  form-preferred tasks per block.
* Two policies (the table and form semantic specialists) in immutable
  registry order mapped by capability.
* 2 replicas x 2 policies x 24 tasks = 96 attempts per schedule; 48
  (task, replica) panels per schedule.  Within each task, replica 1 reverses
  replica 0's arm positions; execution chronology is deterministic.
* A full contingency schedule is frozen alongside the primary schedule: the
  same tasks/routes/policies/blocks with fresh attempt IDs, sampling seeds,
  chronology, and arm positions.
* Every row is permanently ``T_canary`` / ``canary_excluded``.
* The routed arm is reconstructed from the frozen pre-outcome route receipt
  and the complete crossed panel; it is never executed as a third arm.

Only the standard library and sibling harness modules are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import string
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import m3_routing_probe_gate as gate
from .batch import _append_result
from .m3_pilot import _RUNTIME_PINS, runtime_preflight, source_tree_hash
from .orchestrator import (
    AttemptExecutionError,
    PINNED_SAMPLING_PARAMETERS,
    RemoteConfig,
    policy_spec_from_treatment,
    run_registered_treatments,
)
from .routing_fixtures import (
    BLOCK_COUNT,
    DEFAULT_STAGE_B_SEED,
    DIFFICULTIES,
    PREFERRED_CAPABILITIES,
    STRATA,
    TASKS_PER_BLOCK,
    TASKS_PER_STRATUM_PER_BLOCK,
    build_stage_b_design,
    generate_routing_fixture,
)
from .routing_fixture_gym import (
    GENERATOR_VERSION as GYM_GENERATOR_VERSION,
    VERIFIER_ID,
    VERIFIER_VERSION,
)
from .structural_probe import (
    FEATURE_KEYS,
    audit_features,
    audit_receipt,
    structural_probe,
)
from .treatments import TreatmentRegistry

# ---------------------------------------------------------------------------
# frozen schemas and constants
# ---------------------------------------------------------------------------

SPEC_SCHEMA = "m3-utility-routing-smoke-stage-b-spec-v1"
MANIFEST_SCHEMA = "m3-utility-routing-smoke-stage-b-manifest-v1"
AUTHORIZATION_SCHEMA = "m3-utility-routing-smoke-stage-b-authorization-v1"
GATE_SCHEMA = "m3-utility-routing-smoke-stage-b-gate-v1"
SAFE_ATTEMPT_SCHEMA = "m3-utility-routing-smoke-safe-attempt-v1"
SAFE_EXPORT_SCHEMA = "m3-utility-routing-smoke-safe-export-v1"
RUNTIME_PREFLIGHT_SCHEMA = "m3-utility-routing-smoke-runtime-preflight-v1"
RAW_PANEL_SCHEMA = "m3-utility-routing-smoke-raw-panel-v1"
EXECUTION_RECEIPT_SCHEMA = "m3-utility-routing-smoke-stage-b-execution-receipt-v1"
GATE_REPORT_SCHEMA = "m3-utility-routing-smoke-stage-b-gate-report-v1"

DECISION_PASS = "routing_smoke_pass"
DECISION_NO_GO = "routing_smoke_no_go"
DECISION_INVALID = "invalid"

BLOCKS = BLOCK_COUNT
TASKS = TASKS_PER_BLOCK * BLOCK_COUNT  # 24
STRATA_COUNT = len(STRATA)  # 4
REPLICAS = 2
REPLICAS_0 = 0
REPLICAS_1 = 1
POLICIES_PER_TASK = 2
PANELS_PER_SCHEDULE = TASKS * REPLICAS  # 48
ATTEMPTS_PER_SCHEDULE = PANELS_PER_SCHEDULE * POLICIES_PER_TASK  # 96
TOTAL_PANELS = PANELS_PER_SCHEDULE * 2
TOTAL_ATTEMPTS = ATTEMPTS_PER_SCHEDULE * 2

TEXT_VARIANT = 0
"""Frozen Stage-B text variant (Stage A uses its own variant handling)."""

LAMBDA_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0)
PRIMARY_LAMBDA = 1.0
COST_UNITS_PER_TOKEN = 10000
MAX_SAMPLING_SEED = 2_147_483_647

DEFAULT_REMOTE_PROJECT = "/home/zhimin90/Projects/pyreplab_bayesian_agent_harness"
DEFAULT_REMOTE_HOST = "ubuntu-local"

BOOTSTRAP_DRAWS = 100000
BOOTSTRAP_SEED = 2026081302

CANARY_ROW_STATUS = "T_canary"
CANARY_EXCLUSION = "canary_excluded"

SPECIALIST_CAPABILITIES: tuple[str, ...] = ("table_specialist", "form_specialist")

#: Frozen error taxonomy (plan Section 6, "Attempt classification is frozen
#: before execution").
ERROR_TAXONOMY: dict[str, Any] = {
    "intention_to_treat_failure": {
        "description": (
            "A mechanically valid attempt that did not succeed; never rerun."
        ),
        "codes": [
            "verifier_false_result",
            "malformed_model_answer",
            "refusal",
            "admitted_tool_misuse",
            "model_budget_exhaustion",
            "wall_time_exhaustion_valid_record",
        ],
    },
    "infrastructure_invalid": {
        "description": (
            "Explicit controller, provider-transport, or browser-transport "
            "failure before a valid attempt record."
        ),
        "codes": [
            "controller_error",
            "provider_transport_error",
            "browser_transport_error",
        ],
        "replacement_allowed": True,
    },
    "probe_invalid": {
        "description": "Probe, probe-hash, or probe-order failures.",
        "codes": ["probe_failure", "probe_hash_mismatch", "probe_order_violation"],
    },
    "mechanism_invalid": {
        "description": "Wrong treatment/interface or invalid specialist receipts.",
        "codes": ["wrong_treatment", "wrong_interface", "invalid_specialist_receipt"],
    },
    "verifier_invalid": {
        "description": "Verifier crashes or verifier identity mismatches.",
        "codes": ["verifier_crash", "verifier_identity_mismatch"],
    },
    "protocol_invalid": {
        "description": (
            "Manifest, schedule, sampling, identity, completeness, duplicate, "
            "governance, and artifact-hash failures."
        ),
        "codes": [
            "manifest_failure",
            "schedule_failure",
            "sampling_failure",
            "identity_failure",
            "completeness_failure",
            "duplicate_failure",
            "governance_failure",
            "artifact_hash_failure",
        ],
    },
    "unclassified_invalid": {
        "description": "Unclassified failures fail closed.",
        "codes": ["unclassified_failure"],
        "fail_closed": True,
    },
}

#: Frozen Stage-B thresholds (plan Section 6 pass criteria).
THRESHOLDS: dict[str, Any] = {
    "planned_attempts_per_schedule": ATTEMPTS_PER_SCHEDULE,
    "valid_attempt_receipts_per_schedule": ATTEMPTS_PER_SCHEDULE,
    "max_repeat_discordant_cells": 12,
    "max_repeat_discordant_cells_per_block": 7,
    "routed_verified_success_min": 0.80,
    "routed_success_lift_over_hindsight_best_fixed_min": 0.10,
    "primary_pooled_utility_lift_min": 0.08,
    "bootstrap_one_sided_lower_bound_positive": True,
    "block_utility_lift_min": 0.04,
    "sensitivity_lambda_lift_nonnegative": [0.5, 2.0],
    "mixed_and_ambiguous_strata_lift_nonnegative": True,
}

BOOTSTRAP: dict[str, Any] = {
    "draws": BOOTSTRAP_DRAWS,
    "seed": BOOTSTRAP_SEED,
    "stratification": "block x stratum",
    "resample_unit": "whole task",
    "resample_rule": (
        "draw three whole tasks with replacement inside each block x stratum cell"
    ),
    "preserved": ["both policies", "both replicas", "frozen route"],
    "statistic": (
        "pooled routed utility lift over the hindsight-better fixed utility "
        "specialist, recomputed inside every draw"
    ),
    "bound": "one-sided 90% lower bound",
    "quantile": "nearest-rank empirical 10th percentile",
    "requirement": "strictly above zero",
}

COMPARATORS: dict[str, Any] = {
    "observed_utility": (
        "mean(success) - lambda * mean(output_tokens / 10000) over the two "
        "replicas of a task-policy cell"
    ),
    "lambda_grid": list(LAMBDA_GRID),
    "primary_lambda": PRIMARY_LAMBDA,
    "cost_units_per_token": COST_UNITS_PER_TOKEN,
    "tie_break": "higher success, then lower output-token cost, then immutable registry order",
    "hindsight_better_fixed_specialist": (
        "one specialist used for every task in the relevant evaluation subset, "
        "selected after observing that subset; never a per-task hindsight oracle"
    ),
    "success_comparator": "independent of the utility comparator",
    "pooled_comparison": "all 24 tasks",
    "subset_comparisons": (
        "each block, stratum, and sensitivity comparison recomputes the best "
        "fixed specialist within that declared subset"
    ),
}

RUN_POLICY: dict[str, Any] = {
    "canary": {
        "row_status": CANARY_ROW_STATUS,
        "exclusion": CANARY_EXCLUSION,
        "training_impact": (
            "none; no outcome from this smoke may train or tune the later model"
        ),
    },
    "routed_arm_reconstruction": (
        "routed arm is reconstructed from the frozen pre-outcome routing receipt "
        "and the complete crossed panel; never executed as a third arm"
    ),
    "replacement": "only infrastructure_invalid activates a replacement",
    "contingency_use": (
        "complete contingency schedule frozen before the first primary attempt; "
        "same tasks, routes, policies, and block identity with fresh attempt "
        "IDs, sampling seeds, chronology, and arm positions"
    ),
    "on_contingency_use": (
        "all attempts from the corresponding primary block are quarantined and "
        "that complete contingency block is run once"
    ),
    "live_chronology": (
        "run primary panels in the frozen primary chronology, skipping the "
        "remainder of a block after its first infrastructure-invalid trigger; "
        "after the primary chronology is exhausted, run activated contingency "
        "blocks in the frozen contingency chronology"
    ),
    "second_replacement": False,
    "outcome_peeking": False,
    "selective_reruns": False,
    "early_outcome_stopping": False,
    "outcome_driven_replacement": False,
    "intention_to_treat": True,
    "pooling_rescue_of_negative_block": False,
}

GATE_CHECKS: tuple[str, ...] = (
    "all 96 planned attempts present with exact frozen identities and valid "
    "sampling receipts",
    "zero infrastructure, structural, mechanism, probe-hash, or verifier errors",
    "no selective reruns, early outcome stopping, or outcome-driven replacement",
    "every task has a complete two-policy two-replica panel",
    "repeat-discordant policy-task cells at most 12/48 overall and at most 7/24 "
    "in either block",
    "routed verified success at least 80%",
    "routed success exceeds the hindsight-better fixed specialist by at least "
    "10 percentage points",
    "primary pooled utility lift over hindsight-better fixed specialist at "
    "least 0.08 with one-sided 90% task-cluster bootstrap lower bound above zero",
    "utility lift at least 0.04 in each independent block",
    "utility lift nonnegative at sensitivity lambdas 0.5 and 2.0",
    "mixed and ambiguous strata each have nonnegative primary utility lift",
)

#: Frozen required fields of a runtime Stage-B safe-attempt record.  The
#: schedule freezes the pre-treatment subset; the record contract itself is
#: declared here so the future executor cannot drift.
SAFE_ATTEMPT_REQUIRED_FIELDS: tuple[str, ...] = (
    "schema_version",
    "attempt_id",
    "panel_id",
    "schedule",
    "task_id",
    "task_ordinal",
    "block",
    "stratum",
    "difficulty",
    "fixture_id",
    "replica",
    "policy_capability",
    "policy_bundle_id",
    "policy_bundle_hash",
    "arm_position",
    "execution_order",
    "sampling_seed",
    "source_sha256",
    "probe_receipt_sha256",
    "route_receipt_sha256",
    "canary_row_status",
    "canary_exclusion",
    "status",
    "error_class",
    "error_code",
    "sampling_receipt",
    "verification",
    "usage",
    "mechanism",
    "provenance",
)

#: The pre-treatment identity subset of a runtime safe-attempt row.  These
#: must match the frozen planned attempt exactly: schema, IDs, task/block/
#: stratum/difficulty/fixture/replica, policy/bundle, arm position/execution
#: order, sampling/source/probe/route hashes, and governance (canary) fields.
ATTEMPT_IDENTITY_FIELDS: tuple[str, ...] = (
    "schema_version",
    "attempt_id",
    "panel_id",
    "schedule",
    "task_id",
    "task_ordinal",
    "block",
    "stratum",
    "difficulty",
    "fixture_id",
    "replica",
    "policy_capability",
    "policy_bundle_id",
    "policy_bundle_hash",
    "arm_position",
    "execution_order",
    "sampling_seed",
    "source_sha256",
    "probe_receipt_sha256",
    "route_receipt_sha256",
    "canary_row_status",
    "canary_exclusion",
)

#: Never allowed anywhere in a serialized manifest (would indicate leakage of
#: HTML, page text, oracle, reference, or nonce material).
_FORBIDDEN_TOKENS: tuple[str, ...] = (
    "<html",
    "<table",
    "<form",
    "RF-",
    "REF-",
    "PENDING",
    "LOCKED",
    "unlock_query_param",
    "expected_answer",
)

_HEX_DIGITS = frozenset(string.hexdigits)


# ---------------------------------------------------------------------------
# canonical serialization and immutable writes
# ---------------------------------------------------------------------------


def canonical_json(value: Any) -> str:
    """Deterministic, key-sorted compact JSON serialization used for hashing."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def canonical_hash(value: Any) -> str:
    """SHA-256 over the canonical serialization of ``value``."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _json_equal(first: Any, second: Any) -> bool:
    """Type-sensitive equality for JSON-shaped protocol values.

    Native Python equality treats booleans as integers (``True == 1``), which
    is unsafe for frozen identities and deterministic rebuild comparisons.
    Canonical JSON preserves that type distinction.
    """
    try:
        return canonical_json(first) == canonical_json(second)
    except (TypeError, ValueError):
        return False


def file_sha256(path: str | Path) -> str:
    """SHA-256 over the exact file bytes at ``path``."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def immutable_write(path: str | Path, value: Any) -> None:
    """Write ``value`` with immutable-write semantics.

    Refuses to overwrite an existing file unless the prospective bytes are
    byte-identical (idempotent no-op).  New files are written atomically via a
    same-directory temporary file and ``os.replace``.
    """
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() == payload:
            return
        raise FileExistsError(
            f"refusing to overwrite {target}: it already exists with different "
            f"bytes (immutable-write semantics; remove the file to regenerate)"
        )
    fd, temporary = tempfile.mkstemp(
        dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# deterministic derivation helpers
# ---------------------------------------------------------------------------


def _digest_hex(*parts: object) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _derive_seed(*parts: object) -> int:
    """Deterministic non-negative integer derived from namespaced parts."""
    return int(_digest_hex("seed", *parts)[:16], 16)


def _panel_sampling_seed(
    schedule_name: str, schedule_seed: int, panel_index: int
) -> int:
    """Return a provider-valid, collision-free panel-common sampling seed.

    Primary and contingency schedules occupy disjoint halves of the signed
    31-bit provider seed space. Within each half, a deterministic start plus
    the panel index guarantees uniqueness without probabilistic collision
    handling.
    """
    if schedule_name not in ("primary", "contingency"):
        raise ValueError(f"unknown schedule name {schedule_name!r}")
    panel_index = _require_int(panel_index, "panel_index")
    if not (0 <= panel_index < PANELS_PER_SCHEDULE):
        raise ValueError(
            f"panel_index must be in 0..{PANELS_PER_SCHEDULE - 1}"
        )
    namespace_size = (MAX_SAMPLING_SEED + 1) // 2
    namespace_start = 0 if schedule_name == "primary" else namespace_size
    max_offset = namespace_size - PANELS_PER_SCHEDULE
    offset = int(
        _digest_hex("panel-seed-start", schedule_name, schedule_seed)[:8], 16
    ) % (max_offset + 1)
    return namespace_start + offset + panel_index


def _embed_self_hash(payload: dict[str, Any], field: str) -> str:
    digest = canonical_hash({key: value for key, value in payload.items() if key != field})
    payload[field] = digest
    return digest


def _verify_self_hash(payload: Mapping[str, Any], field: str) -> list[str]:
    expected = payload.get(field)
    if not isinstance(expected, str) or len(expected) != 64:
        return [f"missing or malformed {field}"]
    unhashed = {key: value for key, value in payload.items() if key != field}
    actual = canonical_hash(unhashed)
    if actual != expected:
        return [f"{field} mismatch: stored {expected}, computed {actual}"]
    return []


def _is_hex_digest(value: Any, length: int = 64) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and set(value).issubset(_HEX_DIGITS)
    )


def _load_json(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def _sha256_bytes(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer, got {value!r}")
    return value


def _stage_b_block_seeds(seed: int) -> tuple[int, int]:
    """Mirror the backend's frozen block-seed derivation.

    ``routing_fixtures.build_stage_b_design`` derives its two block seeds from
    the master seed when none are supplied.  This mirrors that derivation so
    the manifest can freeze the exact block seeds and become independent of
    backend internals.  ``build_manifest`` verifies the mirror by requiring
    the two call forms to produce identical coordinates.
    """
    seed = _require_int(seed, "seed")
    return tuple(
        int(_digest_hex("seed", "stage-b", seed, "block", index)[:16], 16)
        for index in range(BLOCK_COUNT)
    )


# ---------------------------------------------------------------------------
# spec builder
# ---------------------------------------------------------------------------


def default_spec(
    registry_path: str | Path,
    stage_a_manifest_path: str | Path,
    stage_a_report_path: str | Path,
    *,
    seed: int = DEFAULT_STAGE_B_SEED,
    text_variant: int = TEXT_VARIANT,
    remote_host: str = DEFAULT_REMOTE_HOST,
    remote_project: str = DEFAULT_REMOTE_PROJECT,
    remote_run_root: str | None = None,
    remote_python: str = "python3",
) -> dict[str, Any]:
    """Build a practical Stage-B spec from production files.

    Loads the policy registry (with full hash verification) and the Stage-A
    manifest/report evidence, binding their byte SHA-256 and embedded hashes.
    The Stage-A gate report is *not* required to have passed here -- pass
    enforcement belongs to ``build_authorization``.
    """
    registry_path = _resolve(registry_path)
    manifest_path = _resolve(stage_a_manifest_path)
    report_path = _resolve(stage_a_report_path)
    if not registry_path.is_file():
        raise ValueError(f"registry file does not exist: {registry_path}")
    if not manifest_path.is_file():
        raise ValueError(f"stage-a manifest file does not exist: {manifest_path}")
    if not report_path.is_file():
        raise ValueError(f"stage-a gate report file does not exist: {report_path}")

    registry = TreatmentRegistry.load(registry_path)
    stage_a_manifest = _load_json(manifest_path)
    stage_a_report = _load_json(report_path)
    if remote_run_root is None:
        remote_run_root = f"{remote_project}/.runs/m3-utility-routing-smoke-stage-b-{seed}"
    remote = RemoteConfig(
        host=str(remote_host),
        project=str(remote_project),
        run_root=str(remote_run_root),
        python=str(remote_python),
    )
    from .orchestrator import validate_remote_config

    validate_remote_config(remote)

    manifest_hash = stage_a_manifest.get("manifest_hash")
    stage_a_id = stage_a_manifest.get("stage_a_id")
    router = stage_a_manifest.get("router")
    probe_schema = stage_a_manifest.get("probe_schema")
    report_hash = stage_a_report.get("report_hash")
    if not all(
        isinstance(value, str) and value
        for value in (manifest_hash, report_hash, stage_a_id)
    ):
        raise ValueError("stage-a evidence is missing embedded manifest/report hashes")
    if not isinstance(router, Mapping):
        raise ValueError("stage-a manifest is missing the frozen router config")
    if not isinstance(probe_schema, Mapping):
        raise ValueError("stage-a manifest is missing the frozen probe schema")

    return {
        "schema_version": SPEC_SCHEMA,
        "stage_b_id": f"stage-b-{seed}",
        "seed": _require_int(seed, "seed"),
        "text_variant": _require_int(text_variant, "text_variant"),
        "registry": {
            "registry_path": str(registry_path),
            "registry_hash": registry.registry_hash,
            "registry_sha256": file_sha256(registry_path),
        },
        "remote_identity": {
            "host": remote.host,
            "project": remote.project,
            "run_root": remote.run_root,
            "python": remote.python,
        },
        "stage_a": {
            "manifest_path": str(manifest_path),
            "report_path": str(report_path),
            "manifest_sha256": file_sha256(manifest_path),
            "report_sha256": file_sha256(report_path),
            "manifest_hash": manifest_hash,
            "report_hash": report_hash,
            "stage_a_id": stage_a_id,
            "router": dict(router),
            "probe_schema": dict(probe_schema),
        },
    }


def _coerce_registry(registry: Any) -> TreatmentRegistry:
    if isinstance(registry, TreatmentRegistry):
        return registry
    if isinstance(registry, Mapping):
        return TreatmentRegistry.from_dict(registry, verify_hashes=True)
    raise ValueError(
        "registry must be a TreatmentRegistry instance or a registry mapping"
    )


def _policy_binding(treatment: Any) -> dict[str, Any]:
    policy = policy_spec_from_treatment(treatment)
    return {
        "capability": str(treatment.generator_metadata.get("capability", "")),
        "bundle_id": treatment.bundle_id,
        "bundle_hash": treatment.bundle_hash,
        "policy_id": treatment.id,
        "version": treatment.version,
        "tool_interface": treatment.tool_interface,
        "allowed_tools": list(treatment.allowed_tools),
        "system_prompt_sha256": _sha256_bytes(policy.system_prompt),
        "max_output_tokens": policy.max_output_tokens,
        "tool_call_limit": policy.tool_call_limit,
        "command_timeout_seconds": policy.command_timeout_seconds,
        "wall_time_limit_seconds": policy.wall_time_limit_seconds,
        "enforce_budget": policy.enforce_budget,
    }


def _policies_by_capability(
    registry: TreatmentRegistry,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Map the two specialist capabilities to registry policies.

    Returns ``(policies_by_capability, policy_order)`` where ``policy_order``
    is the immutable registry order ``[table_bundle_id, form_bundle_id]``.
    """
    found: dict[str, dict[str, Any]] = {}
    for treatment in registry.treatments:
        binding = _policy_binding(treatment)
        capability = binding["capability"]
        if capability not in SPECIALIST_CAPABILITIES:
            continue
        if capability in found:
            raise ValueError(
                f"registry contains more than one {capability} policy; "
                "exactly one per capability is required"
            )
        found[capability] = binding
    missing = [cap for cap in SPECIALIST_CAPABILITIES if cap not in found]
    if missing:
        raise ValueError(
            f"registry must contain exactly one policy per capability; "
            f"missing: {missing}"
        )
    policy_order = [found["table_specialist"]["bundle_id"], found["form_specialist"]["bundle_id"]]
    return found, policy_order


def _capability_for_bundle(
    policy_bindings: Mapping[str, Mapping[str, Any]], bundle_id: str
) -> str:
    for capability, binding in policy_bindings.items():
        if binding["bundle_id"] == bundle_id:
            return capability
    raise ValueError(f"bundle {bundle_id!r} is not a bound policy bundle")


# ---------------------------------------------------------------------------
# request-feature validation (mirrors the Stage-A gate contract)
# ---------------------------------------------------------------------------


def _validate_request_features(stratum: str, flags: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    table_op = flags.get("table_operation")
    form_op = flags.get("form_operation")
    first = flags.get("first_operation")
    if stratum == "pure_table":
        if table_op is not True or form_op is not False:
            errors.append("pure_table must declare table_operation only")
        if first is not None:
            errors.append("pure_table must not declare first_operation")
    elif stratum == "pure_form":
        if table_op is not False or form_op is not True:
            errors.append("pure_form must declare form_operation only")
        if first is not None:
            errors.append("pure_form must not declare first_operation")
    elif stratum == "mixed":
        if table_op is not True or form_op is not True:
            errors.append("mixed must declare both operations")
        if first not in ("table", "form"):
            errors.append("mixed must declare first_operation table or form")
    elif stratum == "ambiguous":
        if table_op is form_op:
            errors.append("ambiguous must declare exactly one operation")
        if first is not None:
            errors.append("ambiguous must not declare first_operation")
    else:
        errors.append(f"unknown stratum {stratum!r}")
    return errors


# ---------------------------------------------------------------------------
# task derivation (shared by freeze and rebuild validation)
# ---------------------------------------------------------------------------


def _task_entries_and_receipts(
    coords: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
    router: Mapping[str, Any],
    probe_schema: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Derive the exact task bindings and self-hashed route receipts.

    Every task binds: source/prompt/request/probe/receipt/oracle/private
    coordinate hashes and a route receipt (combined + prompt-only routes)
    computed with the *frozen* Stage-A heuristics.  Only public fixture data
    and bounded probe features ever appear in the outputs.
    """
    if set(probe_schema) != set(FEATURE_KEYS):
        raise ValueError("probe_schema must define exactly the 16 probe features")
    text_variant = _require_int(spec.get("text_variant"), "spec.text_variant")
    if text_variant != TEXT_VARIANT:
        raise ValueError(
            f"Stage-B text_variant must be frozen at {TEXT_VARIANT}, "
            f"got {text_variant!r}"
        )

    tasks: list[dict[str, Any]] = []
    receipts: dict[str, dict[str, Any]] = {}
    for ordinal, coord in enumerate(coords):
        stratum = coord.get("stratum")
        if stratum not in STRATA:
            raise ValueError(f"{coord.get('fixture_id')!r}: unknown stratum {stratum!r}")
        difficulty = coord.get("difficulty")
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"{coord.get('fixture_id')!r}: unknown difficulty {difficulty!r}")
        capability = coord.get("preferred_capability")
        if capability not in PREFERRED_CAPABILITIES:
            raise ValueError(f"{coord.get('fixture_id')!r}: unknown preferred capability")

        fixture = generate_routing_fixture(coord, text_variant=text_variant)
        if not isinstance(fixture, Mapping):
            raise ValueError(f"{coord.get('fixture_id')!r}: fixture must be an object")
        html = fixture.get("html")
        prompt = fixture.get("prompt")
        if not isinstance(html, str) or not isinstance(prompt, str):
            raise ValueError(
                f"{coord.get('fixture_id')!r}: fixture must supply html and public prompt"
            )

        source_sha = _sha256_bytes(html)
        if fixture.get("source_sha256") not in (None, source_sha):
            raise ValueError(f"{coord.get('fixture_id')!r}: generated source_sha256 mismatch")
        prompt_sha = _sha256_bytes(prompt)

        request_features = gate.derive_request_features(prompt)
        request_errors = _validate_request_features(stratum, request_features)
        if request_errors:
            raise ValueError(
                f"{coord.get('fixture_id')!r}: public request features invalid: "
                + "; ".join(request_errors)
            )

        probe = structural_probe(html)
        if not isinstance(probe, Mapping):
            raise ValueError(f"{coord.get('fixture_id')!r}: probe result must be an object")
        features = probe.get("features")
        receipt = probe.get("receipt")
        if not isinstance(features, Mapping) or not isinstance(receipt, Mapping):
            raise ValueError(f"{coord.get('fixture_id')!r}: probe must supply features and receipt")
        feature_errors = audit_features(features)
        if feature_errors:
            raise ValueError(f"{coord.get('fixture_id')!r}: probe features invalid: " + "; ".join(feature_errors))
        receipt_errors = audit_receipt(receipt, features)
        if receipt_errors:
            raise ValueError(f"{coord.get('fixture_id')!r}: probe receipt invalid: " + "; ".join(receipt_errors))
        if receipt.get("source_sha256") != source_sha:
            raise ValueError(f"{coord.get('fixture_id')!r}: probe receipt source hash mismatch")

        combined_route = gate.frozen_heuristic(request_features, features, router)["choice"]
        prompt_only_route = gate.prompt_only_heuristic(request_features, features, router)["choice"]

        fixture_id = str(coord["fixture_id"])
        task_id = f"routing-fixture-{fixture_id}"
        receipt_entry: dict[str, Any] = {
            "task_id": task_id,
            "fixture_id": fixture_id,
            "block": _require_int(coord.get("block"), f"{fixture_id} block"),
            "stratum": stratum,
            "difficulty": difficulty,
            "preferred_capability": capability,
            "request_features": dict(request_features),
            "probe_features": dict(features),
            "combined_route": combined_route,
            "prompt_only_route": prompt_only_route,
        }
        _embed_self_hash(receipt_entry, "route_receipt_sha256")

        tasks.append(
            {
                "task_id": task_id,
                "ordinal": ordinal,
                "seed": _require_int(coord.get("seed"), f"{fixture_id} seed"),
                "block": _require_int(coord.get("block"), f"{fixture_id} block"),
                "stratum": stratum,
                "difficulty": difficulty,
                "preferred_capability": capability,
                "fixture_id": fixture_id,
                "text_variant": text_variant,
                "source_sha256": source_sha,
                "prompt_sha256": prompt_sha,
                "request_features_sha256": canonical_hash(request_features),
                "probe_features_sha256": canonical_hash(features),
                "probe_receipt_sha256": canonical_hash(receipt),
                "oracle_sha256": canonical_hash(coord.get("oracle")),
                "private_seal": canonical_hash(dict(coord)),
                "route_receipt_sha256": receipt_entry["route_receipt_sha256"],
            }
        )
        receipts[task_id] = receipt_entry
    return tasks, receipts


# ---------------------------------------------------------------------------
# schedule derivation (shared by freeze and rebuild validation)
# ---------------------------------------------------------------------------


def _build_schedule(
    schedule_name: str,
    schedule_seed: int,
    tasks: Sequence[Mapping[str, Any]],
    policy_order: Sequence[str],
    policy_bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Derive one frozen schedule: 48 panels and 96 attempts.

    Task chronology is a deterministic shuffle seeded per schedule.  Within a
    task, replica 1 reverses replica 0's arm positions; a task's orientation
    alternates with its ordinal so each block keeps both capabilities balanced
    at every arm position.  Attempt IDs, panel seeds, attempt seeds, and the
    chronology are namespaced by ``schedule_name`` so the primary and
    contingency schedules are fresh by construction.
    """
    by_ordinal = {task["ordinal"]: task for task in tasks}
    if len(by_ordinal) != len(tasks):
        raise ValueError("task ordinals must be unique")
    panel_coordinates = [
        (ordinal, replica)
        for ordinal in range(len(tasks))
        for replica in (REPLICAS_0, REPLICAS_1)
    ]
    random.Random(
        _derive_seed("chronology", schedule_name, schedule_seed)
    ).shuffle(panel_coordinates)
    chronology = [
        {"task_ordinal": ordinal, "replica": replica}
        for ordinal, replica in panel_coordinates
    ]

    panels: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    execution_position = 0
    for ordinal, replica in panel_coordinates:
        task = by_ordinal[ordinal]
        base = list(policy_order)
        if (ordinal % 2 == 1) != (schedule_name == "contingency"):
            base.reverse()
        order = list(base) if replica == REPLICAS_0 else list(reversed(base))
        panel_seed = _panel_sampling_seed(
            schedule_name, schedule_seed, len(panels)
        )
        panel_id = f"{schedule_name}-sb{ordinal:02d}-r{replica}"
        attempt_ids: list[str] = []
        execution_positions: list[int] = []
        for arm_position, bundle_id in enumerate(order):
            capability = _capability_for_bundle(policy_bindings, bundle_id)
            attempt_id = (
                f"{schedule_name}-sb{ordinal:02d}-r{replica}-"
                f"{capability.rsplit('_specialist', 1)[0]}-"
                f"{_digest_hex('attempt-id', schedule_name, schedule_seed, len(panels), arm_position)[:8]}"
            )
            attempts.append(
                {
                    "attempt_id": attempt_id,
                    "schema_version": SAFE_ATTEMPT_SCHEMA,
                    "schedule": schedule_name,
                    "panel_id": panel_id,
                    "task_id": task["task_id"],
                    "task_ordinal": ordinal,
                    "block": task["block"],
                    "stratum": task["stratum"],
                    "difficulty": task["difficulty"],
                    "fixture_id": task["fixture_id"],
                    "replica": replica,
                    "policy_capability": capability,
                    "policy_bundle_id": bundle_id,
                    "policy_bundle_hash": policy_bindings[capability]["bundle_hash"],
                    "arm_position": arm_position,
                    "execution_order": execution_position,
                    "sampling_seed": panel_seed,
                    "source_sha256": task["source_sha256"],
                    "probe_receipt_sha256": task["probe_receipt_sha256"],
                    "route_receipt_sha256": task["route_receipt_sha256"],
                    "canary_row_status": CANARY_ROW_STATUS,
                    "canary_exclusion": CANARY_EXCLUSION,
                }
            )
            attempt_ids.append(attempt_id)
            execution_positions.append(execution_position)
            execution_position += 1
        panels.append(
            {
                "panel_id": panel_id,
                "schedule": schedule_name,
                "block": task["block"],
                "task_id": task["task_id"],
                "task_ordinal": ordinal,
                "stratum": task["stratum"],
                "difficulty": task["difficulty"],
                "replica": replica,
                "sampling_seed": panel_seed,
                "execution_order": list(order),
                "execution_positions": execution_positions,
                "attempt_ids": attempt_ids,
            }
        )
    return {
        "seed": schedule_seed,
        "chronology": chronology,
        "panels": panels,
        "attempts": attempts,
    }


def _build_schedules(
    seed: int,
    tasks: Sequence[Mapping[str, Any]],
    policy_order: Sequence[str],
    policy_bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    primary_seed = _derive_seed("stage-b-schedule", seed, "primary")
    contingency_seed = _derive_seed("stage-b-schedule", seed, "contingency")
    primary = _build_schedule("primary", primary_seed, tasks, policy_order, policy_bindings)
    contingency = _build_schedule(
        "contingency", contingency_seed, tasks, policy_order, policy_bindings
    )

    if contingency["chronology"] == primary["chronology"]:
        raise ValueError("contingency chronology must differ from the primary chronology")
    primary_ids = {attempt["attempt_id"] for attempt in primary["attempts"]}
    contingency_ids = {attempt["attempt_id"] for attempt in contingency["attempts"]}
    if primary_ids & contingency_ids:
        raise ValueError("contingency attempt IDs must be disjoint from primary")
    primary_panel_seeds = {panel["sampling_seed"] for panel in primary["panels"]}
    primary_attempt_seeds = {attempt["sampling_seed"] for attempt in primary["attempts"]}
    contingency_panel_seeds = {panel["sampling_seed"] for panel in contingency["panels"]}
    contingency_attempt_seeds = {attempt["sampling_seed"] for attempt in contingency["attempts"]}
    if primary_panel_seeds & contingency_panel_seeds:
        raise ValueError("contingency panel sampling seeds must be disjoint from primary")
    if primary_attempt_seeds & contingency_attempt_seeds:
        raise ValueError("contingency attempt sampling seeds must be disjoint from primary")

    return {"primary": primary, "contingency": contingency}


# ---------------------------------------------------------------------------
# manifest freezer
# ---------------------------------------------------------------------------


def build_manifest(
    spec: Mapping[str, Any],
    registry: Any,
    stage_a_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the self-hashed, immutable Stage-B manifest from ``spec``.

    The exact 24-task design comes from ``routing_fixtures.build_stage_b_design``
    using the frozen master seed and the frozen block seeds derived from it.
    Every task is bound to its public fixture source/prompt hashes, bounded
    request and probe features, oracle/private coordinate commitments, and a
    self-hashed route receipt computed with the frozen Stage-A combined and
    prompt-only heuristics.  The manifest also freezes the primary and
    contingency schedules (48 panels / 96 attempts each), the registry binding,
    the Stage-A byte and embedded hash bindings, the lambda grid, thresholds,
    comparators, bootstrap, error taxonomy, and run policy.
    """
    if not isinstance(spec, Mapping):
        raise ValueError("spec must be an object")
    if spec.get("schema_version") != SPEC_SCHEMA:
        raise ValueError(
            f"spec schema must be {SPEC_SCHEMA!r}, got {spec.get('schema_version')!r}"
        )
    seed = _require_int(spec.get("seed"), "spec.seed")
    text_variant = _require_int(spec.get("text_variant"), "spec.text_variant")
    if text_variant != TEXT_VARIANT:
        raise ValueError(
            f"spec.text_variant must be frozen at {TEXT_VARIANT}, "
            f"got {text_variant!r}"
        )
    stage_b_id = spec.get("stage_b_id")
    if not isinstance(stage_b_id, str) or not stage_b_id.strip():
        raise ValueError("spec.stage_b_id must be a non-empty string")

    remote_identity = spec.get("remote_identity")
    if not isinstance(remote_identity, Mapping):
        raise ValueError("spec.remote_identity must be an object")
    remote = RemoteConfig(
        host=str(remote_identity.get("host", "")),
        project=str(remote_identity.get("project", "")),
        run_root=str(remote_identity.get("run_root", "")),
        python=str(remote_identity.get("python", "")),
    )
    from .orchestrator import validate_remote_config

    validate_remote_config(remote)
    if not remote.python.strip() or any(character in remote.python for character in "\r\n\0"):
        raise ValueError("spec.remote_identity.python must be a safe non-empty string")

    # ---- registry binding ---------------------------------------------------
    reg_spec = spec.get("registry")
    if not isinstance(reg_spec, Mapping):
        raise ValueError("spec.registry must be an object")
    registry_obj = _coerce_registry(registry)
    if reg_spec.get("registry_hash") != registry_obj.registry_hash:
        raise ValueError(
            "spec.registry.registry_hash does not match the supplied registry"
        )
    registry_path = reg_spec.get("registry_path")
    if not isinstance(registry_path, str) or not _resolve(registry_path).is_file():
        raise ValueError("spec.registry.registry_path must name the frozen registry file")
    if reg_spec.get("registry_sha256") != file_sha256(registry_path):
        raise ValueError("spec.registry.registry_sha256 does not match registry bytes")
    policy_bindings, policy_order = _policies_by_capability(registry_obj)

    # ---- stage-a evidence binding -------------------------------------------
    sa_spec = spec.get("stage_a")
    if not isinstance(sa_spec, Mapping):
        raise ValueError("spec.stage_a must be an object")
    manifest_path = sa_spec.get("manifest_path")
    report_path = sa_spec.get("report_path")
    if not isinstance(manifest_path, str) or not isinstance(report_path, str):
        raise ValueError("spec.stage_a must carry manifest_path and report_path")
    if file_sha256(manifest_path) != sa_spec.get("manifest_sha256"):
        raise ValueError("stage-a manifest bytes do not match spec.stage_a.manifest_sha256")
    if file_sha256(report_path) != sa_spec.get("report_sha256"):
        raise ValueError("stage-a report bytes do not match spec.stage_a.report_sha256")

    sa_errors = gate.validate_manifest(stage_a_manifest)
    if sa_errors:
        raise ValueError("stage-a manifest invalid: " + "; ".join(sa_errors))
    if stage_a_manifest.get("manifest_hash") != sa_spec.get("manifest_hash"):
        raise ValueError("stage-a embedded manifest_hash does not match spec")
    if stage_a_manifest.get("stage_a_id") != sa_spec.get("stage_a_id"):
        raise ValueError("stage-a stage_a_id does not match spec")
    if stage_a_manifest.get("router") != sa_spec.get("router"):
        raise ValueError("stage-a router config does not match spec")
    if stage_a_manifest.get("probe_schema") != sa_spec.get("probe_schema"):
        raise ValueError("stage-a probe schema does not match spec")

    stage_a_report = _load_json(report_path)
    report_errors = gate.validate_gate_report(stage_a_report, stage_a_manifest)
    if report_errors:
        raise ValueError("stage-a gate report invalid: " + "; ".join(report_errors))
    if stage_a_report.get("report_hash") != sa_spec.get("report_hash"):
        raise ValueError("stage-a embedded report_hash does not match spec")
    if stage_a_report.get("manifest_hash") != sa_spec.get("manifest_hash"):
        raise ValueError("stage-a report manifest_hash does not match spec")
    if stage_a_report.get("stage_a_id") != sa_spec.get("stage_a_id"):
        raise ValueError("stage-a report stage_a_id does not match spec")
    if not isinstance(stage_a_report.get("fixture_commitments"), Sequence):
        raise ValueError("stage-a report must carry fixture_commitments")
    fixture_commitments_sha256 = canonical_hash(stage_a_report["fixture_commitments"])

    router = dict(stage_a_manifest["router"])
    probe_schema = dict(stage_a_manifest["probe_schema"])

    # ---- exact Stage-B design ------------------------------------------------
    block_seeds = _stage_b_block_seeds(seed)
    default_coords = build_stage_b_design(seed=seed)
    explicit_coords = build_stage_b_design(seed=seed, block_seeds=block_seeds)
    if default_coords != explicit_coords:
        raise ValueError("frozen block-seed derivation diverged from the design backend")
    tasks, receipts = _task_entries_and_receipts(explicit_coords, spec, router, probe_schema)

    # ---- frozen schedules ----------------------------------------------------
    schedule = _build_schedules(seed, tasks, policy_order, policy_bindings)

    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "stage_b_id": stage_b_id,
        "seed": seed,
        "block_seeds": list(block_seeds),
        "text_variant": text_variant,
        "counts": {
            "strata": STRATA_COUNT,
            "blocks": BLOCKS,
            "tasks": TASKS,
            "tasks_per_block": TASKS_PER_BLOCK,
            "tasks_per_stratum_per_block": TASKS_PER_STRATUM_PER_BLOCK,
            "replicas": REPLICAS,
            "policies": POLICIES_PER_TASK,
            "panels_per_schedule": PANELS_PER_SCHEDULE,
            "attempts_per_schedule": ATTEMPTS_PER_SCHEDULE,
            "total_panels": TOTAL_PANELS,
            "total_attempts": TOTAL_ATTEMPTS,
        },
        "strata": list(STRATA),
        "difficulties": list(DIFFICULTIES),
        "spec": dict(spec),
        "stage_a": {
            "stage_a_id": sa_spec["stage_a_id"],
            "manifest_path": manifest_path,
            "report_path": report_path,
            "manifest_sha256": sa_spec["manifest_sha256"],
            "report_sha256": sa_spec["report_sha256"],
            "manifest_hash": sa_spec["manifest_hash"],
            "report_hash": sa_spec["report_hash"],
            "fixture_commitments_sha256": fixture_commitments_sha256,
        },
        "registry": {
            "registry_path": reg_spec["registry_path"],
            "registry_hash": registry_obj.registry_hash,
            "registry_sha256": reg_spec["registry_sha256"],
            "policy_order": list(policy_order),
            "policies": dict(policy_bindings),
        },
        "implementation": {
            "fixture_generator_version": explicit_coords[0]["generator_version"],
            "gym_generator_version": GYM_GENERATOR_VERSION,
            "verifier_id": VERIFIER_ID,
            "verifier_version": VERIFIER_VERSION,
            "probe_schema_version": gate.PROBE_RECEIPT_SCHEMA,
            "probe_mechanism": gate.PROBE_MECHANISM,
            "source_tree_hash": source_tree_hash(Path(__file__).resolve().parents[2]),
        },
        "runtime": {
            "remote_identity": {
                "host": remote.host,
                "project": remote.project,
                "run_root": remote.run_root,
                "python": remote.python,
            },
            "pins": dict(_RUNTIME_PINS),
            "sampling": {
                "seed_scope": "panel-common-across-policies",
                "parameters": dict(PINNED_SAMPLING_PARAMETERS),
            },
        },
        "router": router,
        "probe_schema": probe_schema,
        "canary": {
            "row_status": CANARY_ROW_STATUS,
            "exclusion": CANARY_EXCLUSION,
        },
        "tasks": tasks,
        "route_receipts": receipts,
        "schedule": schedule,
        "safe_attempt": {
            "schema_version": SAFE_ATTEMPT_SCHEMA,
            "required_fields": list(SAFE_ATTEMPT_REQUIRED_FIELDS),
            "intention_to_treat_failure_codes": ERROR_TAXONOMY[
                "intention_to_treat_failure"
            ]["codes"],
            "never_rerun": True,
        },
        "analysis": {
            "lambda_grid": list(LAMBDA_GRID),
            "primary_lambda": PRIMARY_LAMBDA,
            "cost_units_per_token": COST_UNITS_PER_TOKEN,
            "comparators": COMPARATORS,
            "bootstrap": BOOTSTRAP,
            "thresholds": THRESHOLDS,
            "error_taxonomy": ERROR_TAXONOMY,
            "run_policy": RUN_POLICY,
        },
        "gate": {
            "schema_version": GATE_SCHEMA,
            "stage": "contract_frozen",
            "implementation": (
                "manifest freeze and authorization are outcome-blind; live "
                "execution, safe export, and post-run analysis are explicit "
                "separate commands"
            ),
            "checks": list(GATE_CHECKS),
        },
        "gates": {
            "tasks": TASKS,
            "blocks": BLOCKS,
            "replicas": REPLICAS,
            "policies": POLICIES_PER_TASK,
            "panels_per_schedule": PANELS_PER_SCHEDULE,
            "attempts_per_schedule": ATTEMPTS_PER_SCHEDULE,
            "deterministic_rebuild": True,
            "tamper_rejection": True,
            "no_secret_leakage": True,
        },
    }
    _embed_self_hash(manifest, "manifest_hash")
    return manifest


# ---------------------------------------------------------------------------
# manifest validation
# ---------------------------------------------------------------------------


def _rebuild_manifest_parts(
    manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Rebuild tasks and route receipts from the manifest's own frozen inputs."""
    spec = manifest["spec"]
    seed = _require_int(spec.get("seed"), "spec.seed")
    block_seeds = manifest.get("block_seeds")
    if (
        not isinstance(block_seeds, Sequence)
        or isinstance(block_seeds, (str, bytes))
        or len(block_seeds) != BLOCK_COUNT
    ):
        raise ValueError(f"block_seeds must be a sequence of {BLOCK_COUNT} integers")
    expected_block_seeds = _stage_b_block_seeds(seed)
    if not _json_equal(list(block_seeds), list(expected_block_seeds)):
        raise ValueError("block_seeds do not derive from the frozen spec seed")
    coords = build_stage_b_design(
        seed=seed, block_seeds=tuple(_require_int(value, "block seed") for value in block_seeds)
    )
    return _task_entries_and_receipts(coords, spec, manifest["router"], manifest["probe_schema"])


def _rebuild_schedule(
    manifest: Mapping[str, Any], schedule_name: str
) -> dict[str, Any]:
    schedule = manifest["schedule"]
    if not isinstance(schedule, Mapping) or schedule_name not in schedule:
        raise ValueError(f"manifest.schedule.{schedule_name} must be an object")
    stored = schedule[schedule_name]
    registry_section = manifest["registry"]
    policies = registry_section["policies"]
    policy_bindings = {capability: dict(binding) for capability, binding in policies.items()}
    policy_order = list(registry_section["policy_order"])
    rebuilt = _build_schedule(
        schedule_name,
        _require_int(stored["seed"], f"{schedule_name} schedule seed"),
        manifest["tasks"],
        policy_order,
        policy_bindings,
    )
    expected_derived_seed = _derive_seed("stage-b-schedule", manifest["spec"]["seed"], schedule_name)
    if rebuilt["seed"] != expected_derived_seed:
        raise ValueError(
            f"{schedule_name} schedule seed does not derive from the spec seed"
        )
    return rebuilt


def _validate_counts(manifest: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping):
        return ["counts must be an object"]
    expected = {
        "strata": STRATA_COUNT,
        "blocks": BLOCKS,
        "tasks": TASKS,
        "tasks_per_block": TASKS_PER_BLOCK,
        "tasks_per_stratum_per_block": TASKS_PER_STRATUM_PER_BLOCK,
        "replicas": REPLICAS,
        "policies": POLICIES_PER_TASK,
        "panels_per_schedule": PANELS_PER_SCHEDULE,
        "attempts_per_schedule": ATTEMPTS_PER_SCHEDULE,
        "total_panels": TOTAL_PANELS,
        "total_attempts": TOTAL_ATTEMPTS,
    }
    for field, expected_value in expected.items():
        if not _json_equal(counts.get(field), expected_value):
            errors.append(f"counts.{field} must equal {expected_value!r}")
    return errors


def _validate_tasks(
    manifest: Mapping[str, Any],
    rebuilt_tasks: Sequence[Mapping[str, Any]],
) -> list[str]:
    errors: list[str] = []
    tasks = manifest.get("tasks")
    if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes)):
        return ["tasks must be a list"]
    if len(tasks) != TASKS:
        return [f"tasks must contain {TASKS} entries"]
    if not _json_equal(list(tasks), list(rebuilt_tasks)):
        return ["tasks do not match the deterministic rebuild from the manifest spec"]

    per_block: dict[int, int] = {}
    per_block_stratum: dict[tuple[int, str], dict[str, int]] = {}
    per_block_stratum_difficulty: dict[tuple[int, str], set[str]] = {}
    per_block_capability: dict[tuple[int, str], int] = {}
    seen_ids: set[str] = set()
    seen_fixture_ids: set[str] = set()
    seen_ordinals: set[int] = set()
    for index, entry in enumerate(tasks):
        if not isinstance(entry, Mapping):
            errors.append(f"tasks[{index}] must be an object")
            continue
        task_id = entry.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            errors.append(f"tasks[{index}] missing task_id")
            continue
        if task_id in seen_ids:
            errors.append(f"duplicate task_id {task_id!r}")
        seen_ids.add(task_id)
        fixture_id = entry.get("fixture_id")
        if not isinstance(fixture_id, str) or not fixture_id:
            errors.append(f"{task_id}: missing fixture_id")
        elif fixture_id in seen_fixture_ids:
            errors.append(f"duplicate fixture_id {fixture_id!r}")
        seen_fixture_ids.add(fixture_id)
        ordinal = entry.get("ordinal")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int):
            errors.append(f"{task_id}: ordinal must be an integer")
        elif ordinal in seen_ordinals or not (0 <= ordinal < TASKS):
            errors.append(f"{task_id}: ordinal must be a unique value in 0..{TASKS - 1}")
        seen_ordinals.add(ordinal)
        block = entry.get("block")
        if isinstance(block, bool) or not isinstance(block, int) or block not in (0, 1):
            errors.append(f"{task_id}: block must be 0 or 1")
        else:
            per_block[block] = per_block.get(block, 0) + 1
        stratum = entry.get("stratum")
        if stratum not in STRATA:
            errors.append(f"{task_id}: unknown stratum {stratum!r}")
        difficulty = entry.get("difficulty")
        if difficulty not in DIFFICULTIES:
            errors.append(f"{task_id}: unknown difficulty {difficulty!r}")
        capability = entry.get("preferred_capability")
        if capability not in PREFERRED_CAPABILITIES:
            errors.append(f"{task_id}: unknown preferred capability {capability!r}")
        if block in (0, 1) and stratum in STRATA and difficulty in DIFFICULTIES:
            key = (block, stratum)
            per_block_stratum[key] = per_block_stratum.get(key, {})
            per_block_stratum[key][difficulty] = per_block_stratum[key].get(difficulty, 0) + 1
            per_block_stratum_difficulty.setdefault(key, set()).add(difficulty)
        if block in (0, 1) and capability in PREFERRED_CAPABILITIES:
            cap_key = (block, capability)
            per_block_capability[cap_key] = per_block_capability.get(cap_key, 0) + 1
        for field in (
            "source_sha256",
            "prompt_sha256",
            "request_features_sha256",
            "probe_features_sha256",
            "probe_receipt_sha256",
            "oracle_sha256",
            "private_seal",
            "route_receipt_sha256",
        ):
            if not _is_hex_digest(entry.get(field)):
                errors.append(f"{task_id}: {field} must be a sha256 hex digest")

    for block in (0, 1):
        if per_block.get(block) != TASKS_PER_BLOCK:
            errors.append(f"block {block} must contain {TASKS_PER_BLOCK} tasks")
    for block in (0, 1):
        for stratum in STRATA:
            key = (block, stratum)
            counts = per_block_stratum.get(key, {})
            if sum(counts.values()) != TASKS_PER_STRATUM_PER_BLOCK:
                errors.append(
                    f"block {block} stratum {stratum} must contain "
                    f"{TASKS_PER_STRATUM_PER_BLOCK} tasks"
                )
            if per_block_stratum_difficulty.get(key) != set(DIFFICULTIES):
                errors.append(
                    f"block {block} stratum {stratum} must contain one task per difficulty"
                )
    for block in (0, 1):
        for capability in PREFERRED_CAPABILITIES:
            expected_count = TASKS_PER_BLOCK // len(PREFERRED_CAPABILITIES)
            if per_block_capability.get((block, capability)) != expected_count:
                errors.append(
                    f"block {block} preferred capability {capability} must be "
                    f"balanced at {expected_count} tasks"
                )
    return errors


def _validate_receipts(
    manifest: Mapping[str, Any],
    rebuilt_receipts: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    errors: list[str] = []
    receipts = manifest.get("route_receipts")
    if not isinstance(receipts, Mapping):
        return ["route_receipts must be an object"]
    if len(receipts) != TASKS:
        errors.append(f"route_receipts must contain {TASKS} entries")
    if not _json_equal(receipts, dict(rebuilt_receipts)):
        return ["route_receipts do not match the deterministic rebuild from the manifest"]
    tasks = manifest.get("tasks")
    task_by_id: dict[str, Mapping[str, Any]] = {}
    if isinstance(tasks, Sequence) and not isinstance(tasks, (str, bytes)):
        for index, task in enumerate(tasks):
            if not isinstance(task, Mapping):
                errors.append(f"tasks[{index}] must be an object")
                continue
            task_id = task.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                errors.append(f"tasks[{index}] missing task_id")
                continue
            task_by_id[task_id] = task
    for task_id, entry in receipts.items():
        if task_id not in task_by_id:
            errors.append(f"route_receipts key {task_id!r} is not a manifest task")
            continue
        if entry.get("task_id") != task_id:
            errors.append(f"route_receipts[{task_id}].task_id mismatch")
        errors.extend(_verify_self_hash(entry, "route_receipt_sha256"))
        if task_by_id[task_id].get("route_receipt_sha256") != entry.get("route_receipt_sha256"):
            errors.append(f"tasks[{task_id}].route_receipt_sha256 mismatch")
        if entry.get("combined_route") not in SPECIALIST_CAPABILITIES:
            errors.append(f"route_receipts[{task_id}].combined_route invalid")
        if entry.get("prompt_only_route") not in SPECIALIST_CAPABILITIES:
            errors.append(f"route_receipts[{task_id}].prompt_only_route invalid")
    return errors


def _validate_schedule(
    manifest: Mapping[str, Any],
    schedule_name: str,
    rebuilt: Mapping[str, Any],
    *,
    other: Mapping[str, Any] | None,
) -> list[str]:
    errors: list[str] = []
    stored = manifest["schedule"][schedule_name]
    if not _json_equal(stored, dict(rebuilt)):
        return [f"schedule.{schedule_name} does not match the deterministic rebuild"]

    panels = stored.get("panels")
    attempts = stored.get("attempts")
    chronology = stored.get("chronology")
    expected_coordinates = {
        (ordinal, replica)
        for ordinal in range(TASKS)
        for replica in (REPLICAS_0, REPLICAS_1)
    }
    chronology_coordinates: set[tuple[int, int]] = set()
    if (
        not isinstance(chronology, Sequence)
        or isinstance(chronology, (str, bytes))
        or len(chronology) != PANELS_PER_SCHEDULE
    ):
        errors.append(
            f"schedule.{schedule_name}.chronology must contain "
            f"{PANELS_PER_SCHEDULE} task-replica coordinates"
        )
    else:
        for index, coordinate in enumerate(chronology):
            if not isinstance(coordinate, Mapping):
                errors.append(
                    f"schedule.{schedule_name}.chronology[{index}] must be an object"
                )
                continue
            ordinal = coordinate.get("task_ordinal")
            replica = coordinate.get("replica")
            if (
                isinstance(ordinal, bool)
                or not isinstance(ordinal, int)
                or isinstance(replica, bool)
                or not isinstance(replica, int)
                or replica not in (REPLICAS_0, REPLICAS_1)
            ):
                errors.append(
                    f"schedule.{schedule_name}.chronology[{index}] is invalid"
                )
                continue
            chronology_coordinates.add((ordinal, replica))
        if chronology_coordinates != expected_coordinates:
            errors.append(
                f"schedule.{schedule_name}.chronology must contain each "
                "task-replica coordinate exactly once"
            )
    if not isinstance(panels, Sequence) or len(panels) != PANELS_PER_SCHEDULE:
        errors.append(
            f"schedule.{schedule_name} must contain {PANELS_PER_SCHEDULE} panels"
        )
    if not isinstance(attempts, Sequence) or len(attempts) != ATTEMPTS_PER_SCHEDULE:
        errors.append(
            f"schedule.{schedule_name} must contain {ATTEMPTS_PER_SCHEDULE} attempts"
        )

    task_by_id: dict[str, Mapping[str, Any]] = {}
    tasks = manifest.get("tasks")
    if isinstance(tasks, Sequence) and not isinstance(tasks, (str, bytes)):
        for index, task in enumerate(tasks):
            if not isinstance(task, Mapping):
                errors.append(f"tasks[{index}] must be an object")
                continue
            task_id = task.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                errors.append(f"tasks[{index}] missing task_id")
                continue
            task_by_id[task_id] = task
    panel_ids: set[str] = set()
    attempt_ids: set[str] = set()
    panel_seeds: set[int] = set()
    attempt_seeds: set[int] = set()
    execution_positions: set[int] = set()
    replica_panels: dict[int, dict[str, list[int]]] = {}

    for panel in panels or []:
        if not isinstance(panel, Mapping):
            errors.append(f"{schedule_name}: panel must be an object")
            continue
        panel_id = panel.get("panel_id")
        if not isinstance(panel_id, str) or not panel_id:
            errors.append(f"{schedule_name}: panel missing panel_id")
        elif panel_id in panel_ids:
            errors.append(f"{schedule_name}: duplicate panel_id {panel_id!r}")
        panel_ids.add(panel_id)
        if panel.get("schedule") != schedule_name:
            errors.append(f"{panel_id}: schedule mismatch")
        panel_replica = panel.get("replica")
        if (
            isinstance(panel_replica, bool)
            or not isinstance(panel_replica, int)
            or panel_replica not in (0, 1)
        ):
            errors.append(f"{panel_id}: replica must be 0 or 1")
        panel_block = panel.get("block")
        if (
            isinstance(panel_block, bool)
            or not isinstance(panel_block, int)
            or panel_block not in (0, 1)
        ):
            errors.append(f"{panel_id}: block must be 0 or 1")
        task = task_by_id.get(panel.get("task_id"))
        if task is None:
            errors.append(f"{panel_id}: unknown task_id {panel.get('task_id')!r}")
        elif task.get("block") != panel.get("block"):
            errors.append(f"{panel_id}: block does not match its task")
        seed = panel.get("sampling_seed")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not (0 <= seed <= MAX_SAMPLING_SEED)
        ):
            errors.append(
                f"{panel_id}: sampling_seed must be an integer in "
                f"[0, {MAX_SAMPLING_SEED}]"
            )
        elif seed in panel_seeds:
            errors.append(f"{schedule_name}: duplicate panel sampling_seed {seed}")
        panel_seeds.add(seed)
        order = panel.get("execution_order")
        if not isinstance(order, Sequence) or sorted(order) != sorted(
            manifest["registry"]["policy_order"]
        ):
            errors.append(f"{panel_id}: execution_order must be the two bound policies")
        attempt_ids_in_panel = panel.get("attempt_ids")
        if (
            not isinstance(attempt_ids_in_panel, Sequence)
            or len(attempt_ids_in_panel) != POLICIES_PER_TASK
        ):
            errors.append(f"{panel_id}: attempt_ids must contain {POLICIES_PER_TASK} ids")
        if (
            task is not None
            and isinstance(panel_replica, int)
            and not isinstance(panel_replica, bool)
            and panel_replica in (0, 1)
        ):
            replica_panels.setdefault(panel["replica"], {}).setdefault(task["task_id"], []).append(
                list(order)
            )

    for attempt in attempts or []:
        if not isinstance(attempt, Mapping):
            errors.append(f"{schedule_name}: attempt must be an object")
            continue
        attempt_id = attempt.get("attempt_id")
        if not isinstance(attempt_id, str) or not attempt_id:
            errors.append(f"{schedule_name}: attempt missing attempt_id")
        elif attempt_id in attempt_ids:
            errors.append(f"duplicate attempt_id {attempt_id!r}")
        attempt_ids.add(attempt_id)
        if attempt.get("schema_version") != SAFE_ATTEMPT_SCHEMA:
            errors.append(f"{attempt_id}: schema_version mismatch")
        if attempt.get("schedule") != schedule_name:
            errors.append(f"{attempt_id}: schedule mismatch")
        if attempt.get("panel_id") not in panel_ids:
            errors.append(f"{attempt_id}: unknown panel_id")
        task = task_by_id.get(attempt.get("task_id"))
        if task is None:
            errors.append(f"{attempt_id}: unknown task_id")
        else:
            for field in ("source_sha256", "probe_receipt_sha256", "route_receipt_sha256"):
                if attempt.get(field) != task.get(field):
                    errors.append(f"{attempt_id}: {field} does not bind its task")
        attempt_replica = attempt.get("replica")
        if (
            isinstance(attempt_replica, bool)
            or not isinstance(attempt_replica, int)
            or attempt_replica not in (0, 1)
        ):
            errors.append(f"{attempt_id}: replica must be 0 or 1")
        capability = attempt.get("policy_capability")
        if capability not in SPECIALIST_CAPABILITIES:
            errors.append(f"{attempt_id}: policy_capability invalid")
        arm_position = attempt.get("arm_position")
        if (
            isinstance(arm_position, bool)
            or not isinstance(arm_position, int)
            or arm_position not in (0, 1)
        ):
            errors.append(f"{attempt_id}: arm_position must be 0 or 1")
        position = attempt.get("execution_order")
        if isinstance(position, bool) or not isinstance(position, int):
            errors.append(f"{attempt_id}: execution_order must be an integer")
        elif position in execution_positions or not (0 <= position < ATTEMPTS_PER_SCHEDULE):
            errors.append(f"{schedule_name}: execution_order must be a unique 0..{ATTEMPTS_PER_SCHEDULE - 1}")
        execution_positions.add(position)
        attempt_seed = attempt.get("sampling_seed")
        if (
            isinstance(attempt_seed, bool)
            or not isinstance(attempt_seed, int)
            or not (0 <= attempt_seed <= MAX_SAMPLING_SEED)
        ):
            errors.append(
                f"{attempt_id}: sampling_seed must be an integer in "
                f"[0, {MAX_SAMPLING_SEED}]"
            )
        else:
            panel = next(
                (
                    candidate
                    for candidate in panels or []
                    if isinstance(candidate, Mapping)
                    and candidate.get("panel_id") == attempt.get("panel_id")
                ),
                None,
            )
            if isinstance(panel, Mapping) and attempt_seed != panel.get("sampling_seed"):
                errors.append(
                    f"{attempt_id}: sampling_seed must match its panel-common seed"
                )
        attempt_seeds.add(attempt_seed)

    # ---- arm-position crossover and balance ---------------------------------
    for task_id, orders in replica_panels.get(0, {}).items():
        replica_1_orders = replica_panels.get(1, {}).get(task_id, [])
        if len(orders) != 1 or len(replica_1_orders) != 1:
            errors.append(f"{task_id}: task must have exactly one panel per replica")
            continue
        if list(reversed(orders[0])) != list(replica_1_orders[0]):
            errors.append(
                f"{task_id}: replica 1 must reverse replica 0's arm positions"
            )
    for block in (0, 1):
        for replica in (0, 1):
            arm_zero_capabilities: dict[str, int] = {}
            for attempt in attempts or []:
                if not isinstance(attempt, Mapping):
                    continue
                if (
                    attempt.get("block") == block
                    and attempt.get("replica") == replica
                    and attempt.get("arm_position") == 0
                ):
                    cap = attempt.get("policy_capability")
                    arm_zero_capabilities[cap] = arm_zero_capabilities.get(cap, 0) + 1
            for capability in SPECIALIST_CAPABILITIES:
                expected = TASKS_PER_BLOCK // len(SPECIALIST_CAPABILITIES)
                if arm_zero_capabilities.get(capability) != expected:
                    errors.append(
                        f"block {block} replica {replica}: capability {capability} "
                        f"must start at arm position 0 in {expected} panels"
                    )

    # ---- contingency freshness ----------------------------------------------
    if schedule_name == "contingency" and other is not None:
        other_ids = {attempt["attempt_id"] for attempt in other["attempts"]}
        other_panel_seeds = {panel["sampling_seed"] for panel in other["panels"]}
        other_attempt_seeds = {attempt["sampling_seed"] for attempt in other["attempts"]}
        if attempt_ids & other_ids:
            errors.append("contingency attempt IDs must be disjoint from primary")
        if panel_seeds & other_panel_seeds:
            errors.append("contingency panel sampling seeds must be disjoint from primary")
        if attempt_seeds & other_attempt_seeds:
            errors.append("contingency attempt sampling seeds must be disjoint from primary")
        if stored.get("chronology") == other.get("chronology"):
            errors.append("contingency chronology must differ from the primary chronology")

    # ---- every task-policy cell bound by all four attempts ------------------
    by_task_policy: dict[tuple[str, str], set[int]] = {}
    for attempt in attempts or []:
        if not isinstance(attempt, Mapping):
            continue
        by_task_policy.setdefault(
            (attempt.get("task_id"), attempt.get("policy_capability")), set()
        ).add(attempt.get("replica"))
    for task_id, capability in by_task_policy:
        if by_task_policy[(task_id, capability)] != {0, 1}:
            errors.append(
                f"{task_id}/{capability}: every task-policy cell must have both replicas"
            )
    return errors


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    registry: Any = None,
    stage_a_manifest: Mapping[str, Any] | None = None,
    stage_a_report: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return a list of validation errors; empty means the manifest is valid.

    Validation fails closed:

    * schema identity and the embedded ``manifest_hash`` self-hash;
    * exact frozen counts and block/stratum/difficulty/capability balance;
    * an exact deterministic rebuild of every task, route receipt, panel, and
      attempt from the manifest's own spec/seed/block seeds;
    * schedule identities: unique attempt/panel IDs and sampling seeds, global
      execution positions, per-task arm reversal, per-block arm balance, and
      contingency freshness;
    * the frozen registry binding (when ``registry`` is supplied);
    * the Stage-A byte/embedded hash bindings (when the Stage-A manifest and
      gate report are supplied); and
    * the frozen analysis/run-policy constants and secret-leakage scan.
    """
    errors: list[str] = []
    if not isinstance(manifest, Mapping):
        return ["manifest must be an object"]

    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        errors.append(
            f"schema_version must be {MANIFEST_SCHEMA!r}, "
            f"got {manifest.get('schema_version')!r}"
        )
    errors.extend(_verify_self_hash(manifest, "manifest_hash"))

    if not _json_equal(manifest.get("strata"), list(STRATA)):
        errors.append(f"strata must equal {list(STRATA)}")
    if not _json_equal(manifest.get("difficulties"), list(DIFFICULTIES)):
        errors.append(f"difficulties must equal {list(DIFFICULTIES)}")
    if not _json_equal(manifest.get("canary"), {
        "row_status": CANARY_ROW_STATUS,
        "exclusion": CANARY_EXCLUSION,
    }):
        errors.append("canary row status/exclusion must be frozen")

    errors.extend(_validate_counts(manifest))

    gates_section = manifest.get("gates")
    if not isinstance(gates_section, Mapping):
        errors.append("gates must be an object")
    else:
        expected_gates = {
            "tasks": TASKS,
            "blocks": BLOCKS,
            "replicas": REPLICAS,
            "policies": POLICIES_PER_TASK,
            "panels_per_schedule": PANELS_PER_SCHEDULE,
            "attempts_per_schedule": ATTEMPTS_PER_SCHEDULE,
        }
        for field, expected_value in expected_gates.items():
            if not _json_equal(gates_section.get(field), expected_value):
                errors.append(f"gates.{field} must equal {expected_value!r}")
        for field in ("deterministic_rebuild", "tamper_rejection", "no_secret_leakage"):
            if gates_section.get(field) is not True:
                errors.append(f"gates.{field} must be true")

    spec = manifest.get("spec")
    if not isinstance(spec, Mapping) or spec.get("schema_version") != SPEC_SCHEMA:
        errors.append(f"spec must carry schema_version {SPEC_SCHEMA!r}")
    if not isinstance(manifest.get("stage_b_id"), str) or not manifest["stage_b_id"]:
        errors.append("stage_b_id must be a non-empty string")

    # ---- rebuild everything from the self-contained manifest -----------------
    try:
        rebuilt_tasks, rebuilt_receipts = _rebuild_manifest_parts(manifest)
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"deterministic rebuild failed: {error}")
        rebuilt_tasks, rebuilt_receipts = [], {}
    errors.extend(_validate_tasks(manifest, rebuilt_tasks))
    errors.extend(_validate_receipts(manifest, rebuilt_receipts))

    schedule = manifest.get("schedule")
    if not isinstance(schedule, Mapping) or set(schedule) != {"primary", "contingency"}:
        errors.append("schedule must contain exactly primary and contingency")
    else:
        primary = schedule.get("primary")
        contingency = schedule.get("contingency")
        if not isinstance(primary, Mapping) or not isinstance(contingency, Mapping):
            errors.append("schedule.primary and schedule.contingency must be objects")
        else:
            try:
                rebuilt_primary = _rebuild_schedule(manifest, "primary")
            except (KeyError, TypeError, ValueError) as error:
                errors.append(f"primary schedule rebuild failed: {error}")
                rebuilt_primary = {}
            try:
                rebuilt_contingency = _rebuild_schedule(manifest, "contingency")
            except (KeyError, TypeError, ValueError) as error:
                errors.append(f"contingency schedule rebuild failed: {error}")
                rebuilt_contingency = {}
            errors.extend(
                _validate_schedule(
                    manifest, "primary", rebuilt_primary, other=rebuilt_contingency
                )
            )
            errors.extend(
                _validate_schedule(
                    manifest, "contingency", rebuilt_contingency, other=rebuilt_primary
                )
            )

    # ---- registry binding ----------------------------------------------------
    registry_section = manifest.get("registry")
    if not isinstance(registry_section, Mapping):
        errors.append("registry must be an object")
    else:
        if not _is_hex_digest(registry_section.get("registry_hash")):
            errors.append("registry.registry_hash must be a sha256 hex digest")
        if not _is_hex_digest(registry_section.get("registry_sha256")):
            errors.append("registry.registry_sha256 must be a sha256 hex digest")
        policies = registry_section.get("policies")
        policy_order = registry_section.get("policy_order")
        if not isinstance(policies, Mapping):
            errors.append("registry.policies must be an object")
        elif set(policies) != set(SPECIALIST_CAPABILITIES):
            errors.append(
                f"registry.policies must map exactly {list(SPECIALIST_CAPABILITIES)}"
            )
        else:
            for capability in SPECIALIST_CAPABILITIES:
                binding = policies.get(capability)
                if not isinstance(binding, Mapping):
                    errors.append(f"registry.policies.{capability} must be an object")
                    continue
                for field in (
                    "capability",
                    "bundle_id",
                    "policy_id",
                    "version",
                    "tool_interface",
                ):
                    if not isinstance(binding.get(field), str) or not binding[field]:
                        errors.append(
                            f"registry.policies.{capability}.{field} must be a non-empty string"
                        )
                if not _is_hex_digest(binding.get("bundle_hash")):
                    errors.append(
                        f"registry.policies.{capability}.bundle_hash must be a sha256 hex digest"
                    )
                if not _is_hex_digest(binding.get("system_prompt_sha256")):
                    errors.append(
                        f"registry.policies.{capability}.system_prompt_sha256 must "
                        "be a sha256 hex digest"
                    )
                for field in (
                    "max_output_tokens",
                    "tool_call_limit",
                    "command_timeout_seconds",
                    "wall_time_limit_seconds",
                ):
                    value = binding.get(field)
                    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                        errors.append(
                            f"registry.policies.{capability}.{field} must be a positive integer"
                        )
                if binding.get("enforce_budget") is not True:
                    errors.append(
                        f"registry.policies.{capability}.enforce_budget must be true"
                    )
                if not isinstance(binding.get("allowed_tools"), Sequence) or isinstance(
                    binding.get("allowed_tools"), (str, bytes)
                ):
                    errors.append(
                        f"registry.policies.{capability}.allowed_tools must be a list"
                    )
                if binding.get("capability") != capability:
                    errors.append(
                        f"registry.policies.{capability}.capability mismatch"
                    )
        expected_order = [
            policies.get("table_specialist", {}).get("bundle_id"),
            policies.get("form_specialist", {}).get("bundle_id"),
        ]
        if (
            not isinstance(policy_order, Sequence)
            or isinstance(policy_order, (str, bytes))
            or list(policy_order) != expected_order
        ):
            errors.append(
                "registry.policy_order must list the table then form bundle in "
                "immutable registry order"
            )
        if registry is not None:
            try:
                registry_obj = _coerce_registry(registry)
                if registry_section.get("registry_hash") != registry_obj.registry_hash:
                    errors.append("registry.registry_hash does not match the supplied registry")
                registry_path = registry_section.get("registry_path")
                if (
                    not isinstance(registry_path, str)
                    or not _resolve(registry_path).is_file()
                    or file_sha256(registry_path)
                    != registry_section.get("registry_sha256")
                ):
                    errors.append("registry.registry_sha256 does not match registry bytes")
                rebuilt_bindings, rebuilt_order = _policies_by_capability(registry_obj)
                if dict(policies) != rebuilt_bindings:
                    errors.append("registry.policies do not match the supplied registry")
                if list(policy_order) != rebuilt_order:
                    errors.append("registry.policy_order does not match the supplied registry")
            except ValueError as error:
                errors.append(f"registry binding invalid: {error}")

    # ---- stage-a binding -----------------------------------------------------
    stage_a_section = manifest.get("stage_a")
    if not isinstance(stage_a_section, Mapping):
        errors.append("stage_a must be an object")
    else:
        for field in (
            "manifest_path",
            "report_path",
            "stage_a_id",
        ):
            if not isinstance(stage_a_section.get(field), str) or not stage_a_section[field]:
                errors.append(f"stage_a.{field} must be a non-empty string")
        for field in (
            "manifest_sha256",
            "report_sha256",
            "manifest_hash",
            "report_hash",
            "fixture_commitments_sha256",
        ):
            if not _is_hex_digest(stage_a_section.get(field)):
                errors.append(f"stage_a.{field} must be a sha256 hex digest")
        if isinstance(spec, Mapping):
            spec_stage_a = spec.get("stage_a")
            if not isinstance(spec_stage_a, Mapping):
                errors.append("spec.stage_a must be an object")
            else:
                for field in (
                    "manifest_sha256",
                    "report_sha256",
                    "manifest_hash",
                    "report_hash",
                    "stage_a_id",
                ):
                    if spec_stage_a.get(field) != stage_a_section.get(field):
                        errors.append(f"spec.stage_a.{field} does not match manifest.stage_a")

        if stage_a_manifest is not None or stage_a_report is not None:
            if stage_a_manifest is None or stage_a_report is None:
                errors.append("stage-a manifest and gate report must be supplied together")
            else:
                if gate.validate_manifest(stage_a_manifest):
                    errors.append("bound stage-a manifest is invalid")
                report_errors = gate.validate_gate_report(stage_a_report, stage_a_manifest)
                if report_errors:
                    errors.append("bound stage-a gate report is invalid")
                if stage_a_manifest.get("manifest_hash") != stage_a_section.get("manifest_hash"):
                    errors.append("stage-a manifest_hash binding mismatch")
                if stage_a_report.get("report_hash") != stage_a_section.get("report_hash"):
                    errors.append("stage-a report_hash binding mismatch")
                if stage_a_manifest.get("stage_a_id") != stage_a_section.get("stage_a_id"):
                    errors.append("stage-a stage_a_id binding mismatch")
                if stage_a_report.get("stage_a_id") != stage_a_section.get("stage_a_id"):
                    errors.append("stage-a report stage_a_id binding mismatch")
                if manifest.get("router") != stage_a_manifest.get("router"):
                    errors.append("router does not match the bound stage-a manifest")
                if manifest.get("probe_schema") != stage_a_manifest.get("probe_schema"):
                    errors.append("probe_schema does not match the bound stage-a manifest")
                commitments = stage_a_report.get("fixture_commitments")
                if not isinstance(commitments, Sequence):
                    errors.append("stage-a report fixture_commitments missing")
                elif canonical_hash(commitments) != stage_a_section.get(
                    "fixture_commitments_sha256"
                ):
                    errors.append("stage-a fixture_commitments_sha256 binding mismatch")

    # ---- frozen analysis constants ------------------------------------------
    implementation = manifest.get("implementation")
    expected_implementation = {
        "fixture_generator_version": build_stage_b_design()[0]["generator_version"],
        "gym_generator_version": GYM_GENERATOR_VERSION,
        "verifier_id": VERIFIER_ID,
        "verifier_version": VERIFIER_VERSION,
        "probe_schema_version": gate.PROBE_RECEIPT_SCHEMA,
        "probe_mechanism": gate.PROBE_MECHANISM,
        "source_tree_hash": source_tree_hash(Path(__file__).resolve().parents[2]),
    }
    if not _json_equal(implementation, expected_implementation):
        errors.append("implementation identities or source_tree_hash are not frozen")

    runtime = manifest.get("runtime")
    expected_runtime = {
        "remote_identity": dict(manifest.get("spec", {}).get("remote_identity", {})),
        "pins": dict(_RUNTIME_PINS),
        "sampling": {
            "seed_scope": "panel-common-across-policies",
            "parameters": dict(PINNED_SAMPLING_PARAMETERS),
        },
    }
    if not _json_equal(runtime, expected_runtime):
        errors.append("runtime pins or sampling contract are not frozen")

    analysis = manifest.get("analysis")
    if not isinstance(analysis, Mapping):
        errors.append("analysis must be an object")
    else:
        if not _json_equal(analysis.get("lambda_grid"), list(LAMBDA_GRID)):
            errors.append(f"analysis.lambda_grid must equal {list(LAMBDA_GRID)}")
        if not _json_equal(analysis.get("primary_lambda"), PRIMARY_LAMBDA):
            errors.append("analysis.primary_lambda must equal 1.0")
        if not _json_equal(
            analysis.get("cost_units_per_token"), COST_UNITS_PER_TOKEN
        ):
            errors.append("analysis.cost_units_per_token must equal 10000")
        if not _json_equal(analysis.get("comparators"), COMPARATORS):
            errors.append("analysis.comparators are not frozen")
        if not _json_equal(analysis.get("bootstrap"), BOOTSTRAP):
            errors.append("analysis.bootstrap is not frozen")
        if not _json_equal(analysis.get("thresholds"), THRESHOLDS):
            errors.append("analysis.thresholds are not frozen")
        if not _json_equal(analysis.get("error_taxonomy"), ERROR_TAXONOMY):
            errors.append("analysis.error_taxonomy is not frozen")
        if not _json_equal(analysis.get("run_policy"), RUN_POLICY):
            errors.append("analysis.run_policy is not frozen")

    safe_attempt = manifest.get("safe_attempt")
    if not isinstance(safe_attempt, Mapping):
        errors.append("safe_attempt must be an object")
    elif (
        safe_attempt.get("schema_version") != SAFE_ATTEMPT_SCHEMA
        or safe_attempt.get("required_fields") != list(SAFE_ATTEMPT_REQUIRED_FIELDS)
    ):
        errors.append("safe_attempt contract is not frozen")

    gate_section = manifest.get("gate")
    if not isinstance(gate_section, Mapping) or gate_section.get("schema_version") != GATE_SCHEMA:
        errors.append(f"gate schema_version must be {GATE_SCHEMA!r}")

    # ---- secret-leakage scan --------------------------------------------------
    serialized = canonical_json(manifest)
    for token in _FORBIDDEN_TOKENS:
        if token in serialized:
            errors.append(f"secret-leakage: forbidden content {token!r} found in manifest")

    return errors


# ---------------------------------------------------------------------------
# authorization layer
# ---------------------------------------------------------------------------


def _verify_stage_a_bindings(
    stage_a_manifest_path: str | Path,
    stage_a_report_path: str | Path,
    stage_a_manifest: Mapping[str, Any],
    stage_a_report: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    """Fail-closed Stage-A evidence check shared by build/validate.

    Raises ``ValueError`` on the first detected violation.
    """
    stage_a_section = manifest.get("stage_a")
    if not isinstance(stage_a_section, Mapping):
        raise ValueError("manifest.stage_a must be an object")

    manifest_sha = file_sha256(stage_a_manifest_path)
    report_sha = file_sha256(stage_a_report_path)
    if stage_a_section.get("manifest_sha256") != manifest_sha:
        raise ValueError(
            f"stage-a manifest byte SHA-256 mismatch: stored "
            f"{stage_a_section.get('manifest_sha256')}, computed {manifest_sha}"
        )
    if stage_a_section.get("report_sha256") != report_sha:
        raise ValueError(
            f"stage-a report byte SHA-256 mismatch: stored "
            f"{stage_a_section.get('report_sha256')}, computed {report_sha}"
        )

    errors = gate.validate_manifest(stage_a_manifest)
    if errors:
        raise ValueError("stage-a manifest invalid: " + "; ".join(errors))
    errors = gate.validate_gate_report(stage_a_report, stage_a_manifest)
    if errors:
        raise ValueError("stage-a gate report invalid: " + "; ".join(errors))

    if stage_a_report.get("decision") != "probe_pass":
        raise ValueError(
            f"stage-a gate decision must be probe_pass, got "
            f"{stage_a_report.get('decision')!r}"
        )
    if stage_a_report.get("passed") is not True:
        raise ValueError("stage-a gate report passed must be True")

    if stage_a_manifest.get("manifest_hash") != stage_a_section.get("manifest_hash"):
        raise ValueError("stage-a embedded manifest_hash does not match the manifest binding")
    if stage_a_report.get("report_hash") != stage_a_section.get("report_hash"):
        raise ValueError("stage-a embedded report_hash does not match the manifest binding")
    if stage_a_manifest.get("stage_a_id") != stage_a_section.get("stage_a_id"):
        raise ValueError("stage-a stage_a_id does not match the manifest binding")
    if stage_a_report.get("stage_a_id") != stage_a_section.get("stage_a_id"):
        raise ValueError("stage-a report stage_a_id does not match the manifest binding")
    if stage_a_report.get("manifest_hash") != stage_a_section.get("manifest_hash"):
        raise ValueError("stage-a report manifest_hash does not match the manifest binding")
    if manifest.get("router") != stage_a_manifest.get("router"):
        raise ValueError("router does not match the bound stage-a manifest")

    commitments = stage_a_report.get("fixture_commitments")
    if not isinstance(commitments, Sequence):
        raise ValueError("stage-a report must carry fixture_commitments")
    if canonical_hash(commitments) != stage_a_section.get("fixture_commitments_sha256"):
        raise ValueError("stage-a fixture_commitments binding mismatch")


def build_authorization(
    stage_a_manifest_path: str | Path,
    stage_a_report_path: str | Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a self-hashed authorization record for a Stage-B manifest.

    Loads the Stage-A manifest and gate report, runs ``m3_routing_probe_gate``
    ``validate_manifest`` / ``validate_gate_report``, requires ``probe_pass``,
    checks the exact byte and embedded hash bindings against ``manifest``, and
    emits a self-hashed authorization record binding the Stage-B manifest
    hash.  The Stage-B manifest itself must already validate.
    """
    manifest_errors = validate_manifest(manifest)
    if manifest_errors:
        raise ValueError("stage-b manifest invalid: " + "; ".join(manifest_errors))

    manifest_path = _resolve(stage_a_manifest_path)
    report_path = _resolve(stage_a_report_path)
    stage_a_manifest = _load_json(manifest_path)
    stage_a_report = _load_json(report_path)
    _verify_stage_a_bindings(manifest_path, report_path, stage_a_manifest, stage_a_report, manifest)

    stage_a_section = manifest["stage_a"]
    authorization: dict[str, Any] = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "manifest_hash": manifest["manifest_hash"],
        "stage_b_id": manifest.get("stage_b_id"),
        "stage_a": {
            "manifest_path": str(manifest_path),
            "report_path": str(report_path),
            "manifest_sha256": stage_a_section["manifest_sha256"],
            "report_sha256": stage_a_section["report_sha256"],
            "manifest_hash": stage_a_section["manifest_hash"],
            "report_hash": stage_a_section["report_hash"],
            "stage_a_id": stage_a_section["stage_a_id"],
            "fixture_commitments_sha256": stage_a_section["fixture_commitments_sha256"],
        },
        "decision": "probe_pass",
        "passed": True,
        "basis": (
            "m3_routing_probe_gate.validate_manifest + validate_gate_report on "
            "byte-verified Stage-A evidence; probe_pass required"
        ),
    }
    _embed_self_hash(authorization, "authorization_hash")
    return authorization


def validate_authorization(
    authorization: Mapping[str, Any],
    manifest: Mapping[str, Any],
    stage_a_manifest_path: str | Path,
    stage_a_report_path: str | Path,
) -> list[str]:
    """Validate an authorization record against its Stage-B manifest and the
    Stage-A evidence files.  Returns a list of errors; empty means valid."""
    errors: list[str] = []
    if not isinstance(authorization, Mapping):
        return ["authorization must be an object"]
    if authorization.get("schema_version") != AUTHORIZATION_SCHEMA:
        errors.append(
            f"schema_version must be {AUTHORIZATION_SCHEMA!r}, "
            f"got {authorization.get('schema_version')!r}"
        )
    errors.extend(_verify_self_hash(authorization, "authorization_hash"))
    if authorization.get("manifest_hash") != manifest.get("manifest_hash"):
        errors.append("authorization.manifest_hash does not match the Stage-B manifest")
    if authorization.get("stage_b_id") != manifest.get("stage_b_id"):
        errors.append("authorization.stage_b_id does not match the Stage-B manifest")
    if authorization.get("decision") != "probe_pass" or authorization.get("passed") is not True:
        errors.append("authorization must carry decision probe_pass / passed True")

    stage_a_binding = authorization.get("stage_a")
    if not isinstance(stage_a_binding, Mapping):
        return errors + ["authorization.stage_a must be an object"]
    try:
        stage_a_manifest = _load_json(stage_a_manifest_path)
        stage_a_report = _load_json(stage_a_report_path)
        _verify_stage_a_bindings(
            stage_a_manifest_path, stage_a_report_path, stage_a_manifest, stage_a_report, manifest
        )
    except (OSError, ValueError, RuntimeError) as error:
        return errors + [f"stage-a evidence verification failed: {error}"]

    stage_a_section = manifest.get("stage_a")
    if not isinstance(stage_a_section, Mapping):
        errors.append("manifest.stage_a must be an object")
    else:
        for field in (
            "manifest_path",
            "report_path",
            "manifest_sha256",
            "report_sha256",
            "manifest_hash",
            "report_hash",
            "stage_a_id",
            "fixture_commitments_sha256",
        ):
            if stage_a_binding.get(field) != stage_a_section.get(field):
                errors.append(
                    f"authorization.stage_a.{field} does not match manifest.stage_a"
                )
    return errors


# ---------------------------------------------------------------------------
# live runtime preflight
# ---------------------------------------------------------------------------


def _remote_config_from_manifest(manifest: Mapping[str, Any]) -> RemoteConfig:
    remote = manifest.get("runtime", {}).get("remote_identity")
    if not isinstance(remote, Mapping):
        raise ValueError("manifest.runtime.remote_identity must be an object")
    return RemoteConfig(
        host=str(remote.get("host", "")),
        project=str(remote.get("project", "")),
        run_root=str(remote.get("run_root", "")),
        python=str(remote.get("python", "")),
    )


def validate_runtime_preflight(
    preflight: Mapping[str, Any],
    manifest: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> list[str]:
    """Validate the byte-bound Stage-B live-runtime preflight receipt."""
    errors: list[str] = []
    if not isinstance(preflight, Mapping):
        return ["runtime preflight must be an object"]
    if preflight.get("schema_version") != RUNTIME_PREFLIGHT_SCHEMA:
        errors.append(
            f"runtime preflight schema_version must be {RUNTIME_PREFLIGHT_SCHEMA!r}"
        )
    errors.extend(_verify_self_hash(preflight, "preflight_hash"))
    if preflight.get("manifest_hash") != manifest.get("manifest_hash"):
        errors.append("runtime preflight does not bind the Stage-B manifest")
    if preflight.get("authorization_hash") != authorization.get("authorization_hash"):
        errors.append("runtime preflight does not bind the authorization")
    if preflight.get("stage_b_id") != manifest.get("stage_b_id"):
        errors.append("runtime preflight stage_b_id mismatch")
    if preflight.get("source_tree_hash") != manifest.get("implementation", {}).get(
        "source_tree_hash"
    ):
        errors.append("runtime preflight source_tree_hash mismatch")
    if not _json_equal(
        preflight.get("runtime_pins"), manifest.get("runtime", {}).get("pins")
    ):
        errors.append("runtime preflight runtime_pins mismatch")
    remote = manifest.get("runtime", {}).get("remote_identity")
    if not _json_equal(preflight.get("remote_identity"), remote):
        errors.append("runtime preflight remote_identity mismatch")
    if preflight.get("worktree_clean") is not True:
        errors.append("Stage-B runtime preflight requires a clean worktree")
    if not _is_hex_digest(preflight.get("code_revision"), 40):
        errors.append("runtime preflight code_revision must be a 40-char hex digest")
    if not _is_hex_digest(preflight.get("worktree_status_hash")):
        errors.append("runtime preflight worktree_status_hash must be a sha256 digest")
    return errors


def build_runtime_preflight(
    manifest: Mapping[str, Any],
    authorization: Mapping[str, Any],
    *,
    pi_binary: str,
    thinking: str,
    unbrowser_binary: str,
    model_artifact: str,
    llama_server_binary: str,
) -> dict[str, Any]:
    """Run the frozen environment checks and emit a self-hashed receipt."""
    manifest_errors = validate_manifest(manifest)
    if manifest_errors:
        raise ValueError("stage-b manifest invalid: " + "; ".join(manifest_errors))
    auth_errors = _check_authorization_binding(authorization, manifest)
    if auth_errors:
        raise ValueError("authorization invalid: " + "; ".join(auth_errors))
    project_root = Path(__file__).resolve().parents[2]
    config = _remote_config_from_manifest(manifest)
    runtime = runtime_preflight(
        project_root,
        config,
        pi_binary=pi_binary,
        thinking=thinking,
        unbrowser_binary=unbrowser_binary,
        model_artifact=model_artifact,
        llama_server_binary=llama_server_binary,
        require_clean=True,
    )
    receipt: dict[str, Any] = {
        "schema_version": RUNTIME_PREFLIGHT_SCHEMA,
        "manifest_hash": manifest["manifest_hash"],
        "authorization_hash": authorization["authorization_hash"],
        "stage_b_id": manifest["stage_b_id"],
        **runtime,
        "runtime_pins": manifest["runtime"]["pins"],
        "remote_identity": manifest["runtime"]["remote_identity"],
    }
    _embed_self_hash(receipt, "preflight_hash")
    errors = validate_runtime_preflight(receipt, manifest, authorization)
    if errors:
        raise RuntimeError("runtime preflight receipt invalid: " + "; ".join(errors))
    return receipt


# ---------------------------------------------------------------------------
# execution receipt (schedule selection + quarantine)
# ---------------------------------------------------------------------------


def _normalize_block_schedules(
    manifest: Mapping[str, Any], block_schedules: Any
) -> dict[str, str]:
    """Normalize a ``{block: schedule}`` selection to string keys.

    Accepts integer or string block keys and requires a selection for both
    blocks 0 and 1 (each ``"primary"`` or ``"contingency"``).
    """
    if not isinstance(block_schedules, Mapping):
        raise ValueError("block_schedules must be a mapping of block -> schedule")
    normalized: dict[str, str] = {}
    for block, schedule_name in block_schedules.items():
        block_str = str(block)
        if block_str not in ("0", "1"):
            raise ValueError(f"block_schedules key must be 0 or 1, got {block!r}")
        if schedule_name not in ("primary", "contingency"):
            raise ValueError(
                f"block {block_str}: schedule must be 'primary' or 'contingency', "
                f"got {schedule_name!r}"
            )
        if block_str in normalized:
            raise ValueError(f"block {block_str}: duplicate schedule selection")
        normalized[block_str] = schedule_name
    if set(normalized) != {"0", "1"}:
        raise ValueError("block_schedules must select a schedule for both blocks 0 and 1")
    return normalized


def _selected_block_attempt_ids(
    manifest: Mapping[str, Any], block: int, schedule_name: str
) -> list[str]:
    """The ordered attempt IDs of one block within one frozen schedule."""
    return [
        attempt["attempt_id"]
        for attempt in manifest["schedule"][schedule_name]["attempts"]
        if attempt["block"] == block
    ]


def _validate_replacement_triggers(
    manifest: Mapping[str, Any],
    normalized: Mapping[str, str],
    triggers: Any,
) -> list[str]:
    """Return errors for a replacement-trigger mapping (empty means valid).

    Contingency may replace a block only with a structured trigger whose
    ``error_class`` is ``infrastructure_invalid`` and whose ``error_code``
    comes from the manifest's frozen infrastructure taxonomy.  A primary block
    must not carry a trigger, and there is no second replacement or mixing.
    """
    errors: list[str] = []
    if not isinstance(triggers, Mapping):
        return ["replacement_triggers must be an object"]
    taxonomy = manifest["analysis"]["error_taxonomy"]
    infrastructure = taxonomy["infrastructure_invalid"]
    infrastructure_codes = list(infrastructure["codes"])
    for block_str in ("0", "1"):
        schedule_name = normalized[block_str]
        trigger = triggers.get(block_str)
        if schedule_name == "contingency":
            if not isinstance(trigger, Mapping):
                errors.append(
                    f"block {block_str}: contingency replacement requires a "
                    "structured trigger"
                )
                continue
            if trigger.get("error_class") != "infrastructure_invalid":
                errors.append(
                    f"block {block_str}: replacement trigger error_class must be "
                    "'infrastructure_invalid'"
                )
            if trigger.get("error_code") not in infrastructure_codes:
                errors.append(
                    f"block {block_str}: replacement trigger error_code "
                    f"{trigger.get('error_code')!r} is not in the infrastructure "
                    "taxonomy"
                )
            attempt_id = trigger.get("attempt_id")
            phase = trigger.get("phase")
            raw_record_hash = trigger.get("raw_record_hash")
            if not isinstance(attempt_id, str) or not attempt_id:
                errors.append(
                    f"block {block_str}: replacement trigger attempt_id is required"
                )
            if not isinstance(phase, str) or not phase:
                errors.append(
                    f"block {block_str}: replacement trigger phase is required"
                )
            if not _is_hex_digest(raw_record_hash):
                errors.append(
                    f"block {block_str}: replacement trigger raw_record_hash is required"
                )
        elif trigger is not None:
            errors.append(f"block {block_str}: primary schedule cannot carry a replacement trigger")
    for extra in sorted(set(triggers) - {"0", "1"}):
        errors.append(f"replacement_triggers has unexpected key {extra!r}")
    return errors


def build_execution_receipt(
    manifest: Mapping[str, Any],
    authorization: Mapping[str, Any],
    block_schedules: Mapping[Any, str],
    replacement_triggers: Mapping[Any, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a self-hashed execution receipt selecting the schedule per block.

    The receipt binds the manifest and authorization hashes, records which of
    the primary/contingency schedules runs for each block, quarantines the
    exact full set of planned primary attempt IDs for every contingency block,
    and requires a structured infrastructure-invalid replacement trigger for
    each contingency block.  No outcome is inspected and no model is executed.
    """
    manifest_errors = validate_manifest(manifest)
    if manifest_errors:
        raise ValueError("stage-b manifest invalid: " + "; ".join(manifest_errors))
    auth_errors = _check_authorization_binding(authorization, manifest)
    if auth_errors:
        raise ValueError("authorization invalid: " + "; ".join(auth_errors))

    normalized = _normalize_block_schedules(manifest, block_schedules)
    raw_triggers: dict[str, Any] = {}
    if replacement_triggers is not None:
        if not isinstance(replacement_triggers, Mapping):
            raise ValueError("replacement_triggers must be a mapping")
        raw_triggers = {str(block): trigger for block, trigger in replacement_triggers.items()}
    trigger_errors = _validate_replacement_triggers(manifest, normalized, raw_triggers)
    if trigger_errors:
        raise ValueError("replacement triggers invalid: " + "; ".join(trigger_errors))

    selected_by_block: dict[str, list[str]] = {}
    quarantined: dict[str, list[str]] = {}
    triggers_out: dict[str, dict[str, Any]] = {}
    for block_str in ("0", "1"):
        block = int(block_str)
        schedule_name = normalized[block_str]
        selected_by_block[block_str] = _selected_block_attempt_ids(
            manifest, block, schedule_name
        )
        if schedule_name == "contingency":
            quarantined[block_str] = _selected_block_attempt_ids(manifest, block, "primary")
            triggers_out[block_str] = {
                "error_class": raw_triggers[block_str]["error_class"],
                "error_code": raw_triggers[block_str]["error_code"],
                "attempt_id": raw_triggers[block_str]["attempt_id"],
                "phase": raw_triggers[block_str]["phase"],
                "raw_record_hash": raw_triggers[block_str]["raw_record_hash"],
            }

    receipt: dict[str, Any] = {
        "schema_version": EXECUTION_RECEIPT_SCHEMA,
        "manifest_hash": manifest["manifest_hash"],
        "authorization_hash": authorization["authorization_hash"],
        "stage_b_id": manifest["stage_b_id"],
        "block_schedules": normalized,
        "replacement_triggers": triggers_out,
        "quarantined_primary_attempt_ids": quarantined,
        "selected_attempt_ids_by_block": selected_by_block,
        "selected_attempt_ids": [
            attempt_id
            for block_str in ("0", "1")
            for attempt_id in selected_by_block[block_str]
        ],
    }
    _embed_self_hash(receipt, "execution_receipt_hash")
    return receipt


def validate_execution_receipt(
    receipt: Mapping[str, Any],
    manifest: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> list[str]:
    """Validate an execution receipt against its manifest and authorization.

    Returns a list of errors; empty means valid.
    """
    errors: list[str] = []
    if not isinstance(receipt, Mapping):
        return ["execution receipt must be an object"]
    if receipt.get("schema_version") != EXECUTION_RECEIPT_SCHEMA:
        errors.append(
            f"schema_version must be {EXECUTION_RECEIPT_SCHEMA!r}, "
            f"got {receipt.get('schema_version')!r}"
        )
    errors.extend(_verify_self_hash(receipt, "execution_receipt_hash"))
    if receipt.get("manifest_hash") != manifest.get("manifest_hash"):
        errors.append("execution receipt does not bind the supplied manifest")
    if receipt.get("authorization_hash") != authorization.get("authorization_hash"):
        errors.append("execution receipt does not bind the supplied authorization")
    if receipt.get("stage_b_id") != manifest.get("stage_b_id"):
        errors.append("execution receipt stage_b_id does not match the manifest")

    block_schedules = receipt.get("block_schedules")
    try:
        normalized = _normalize_block_schedules(manifest, block_schedules)
    except ValueError as error:
        return errors + [f"block_schedules invalid: {error}"]
    if block_schedules != normalized:
        errors.append("block_schedules are not normalized to {'0': ..., '1': ...}")

    errors.extend(
        _validate_replacement_triggers(manifest, normalized, receipt.get("replacement_triggers"))
    )

    selected_by_block = receipt.get("selected_attempt_ids_by_block")
    quarantined = receipt.get("quarantined_primary_attempt_ids")
    if not isinstance(selected_by_block, Mapping):
        errors.append("selected_attempt_ids_by_block must be an object")
    if not isinstance(quarantined, Mapping):
        errors.append("quarantined_primary_attempt_ids must be an object")

    for block_str in ("0", "1"):
        block = int(block_str)
        schedule_name = normalized[block_str]
        expected_ids = _selected_block_attempt_ids(manifest, block, schedule_name)
        if isinstance(selected_by_block, Mapping) and selected_by_block.get(block_str) != expected_ids:
            errors.append(
                f"block {block_str}: selected_attempt_ids_by_block do not match the "
                f"selected {schedule_name} schedule"
            )
        if schedule_name == "contingency":
            expected_quarantine = _selected_block_attempt_ids(manifest, block, "primary")
            if isinstance(quarantined, Mapping) and quarantined.get(block_str) != expected_quarantine:
                errors.append(
                    f"block {block_str}: quarantine must be the exact full set of "
                    "planned primary attempt IDs"
                )
        elif isinstance(quarantined, Mapping) and quarantined.get(block_str) not in (None, []):
            errors.append(f"block {block_str}: primary block must not be quarantined")

    if isinstance(selected_by_block, Mapping):
        full = [
            attempt_id
            for block_str in ("0", "1")
            for attempt_id in (selected_by_block.get(block_str) or [])
        ]
        if receipt.get("selected_attempt_ids") != full:
            errors.append("selected_attempt_ids do not match the derived block selection")
    return errors


# ---------------------------------------------------------------------------
# raw live panel records and leakage-safe export
# ---------------------------------------------------------------------------


def _raw_record_hash(record: Mapping[str, Any]) -> str:
    return canonical_hash({key: value for key, value in record.items() if key != "raw_record_hash"})


def _load_raw_panel_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    source = _resolve(path)
    if not source.exists():
        return records
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid raw panel JSONL line {line_number}: {error}") from error
        if not isinstance(record, dict):
            raise ValueError(f"raw panel line {line_number} must be an object")
        records.append(record)
    return records


def _panel_lookup(manifest: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {
        (schedule_name, panel["panel_id"]): panel
        for schedule_name in ("primary", "contingency")
        for panel in manifest["schedule"][schedule_name]["panels"]
    }


def _validate_raw_panel_record(
    record: Mapping[str, Any],
    manifest: Mapping[str, Any],
    authorization: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    if not isinstance(record, Mapping):
        return ["raw panel record must be an object"]
    if record.get("schema_version") != RAW_PANEL_SCHEMA:
        errors.append(f"raw panel schema_version must be {RAW_PANEL_SCHEMA!r}")
    if record.get("manifest_hash") != manifest.get("manifest_hash"):
        errors.append("raw panel manifest_hash mismatch")
    if record.get("authorization_hash") != authorization.get("authorization_hash"):
        errors.append("raw panel authorization_hash mismatch")
    if record.get("preflight_hash") != preflight.get("preflight_hash"):
        errors.append("raw panel preflight_hash mismatch")
    expected_hash = record.get("raw_record_hash")
    if not _is_hex_digest(expected_hash) or expected_hash != _raw_record_hash(record):
        errors.append("raw panel raw_record_hash mismatch")

    schedule_name = record.get("schedule")
    panel_id = record.get("panel_id")
    panel = _panel_lookup(manifest).get((schedule_name, panel_id))
    if panel is None:
        errors.append(f"unknown raw panel {schedule_name!r}/{panel_id!r}")
        return errors
    if record.get("block") != panel.get("block"):
        errors.append(f"{panel_id}: raw panel block mismatch")
    if record.get("task_id") != panel.get("task_id"):
        errors.append(f"{panel_id}: raw panel task_id mismatch")
    if record.get("replica") != panel.get("replica"):
        errors.append(f"{panel_id}: raw panel replica mismatch")

    status = record.get("status")
    if status not in ("completed", "infrastructure_invalid"):
        errors.append(f"{panel_id}: raw panel status invalid")
        return errors
    if status == "infrastructure_invalid":
        failure = record.get("failure")
        infrastructure_codes = manifest["analysis"]["error_taxonomy"][
            "infrastructure_invalid"
        ]["codes"]
        if not isinstance(failure, Mapping):
            errors.append(f"{panel_id}: infrastructure failure must be an object")
        else:
            if failure.get("error_class") != "infrastructure_invalid":
                errors.append(f"{panel_id}: infrastructure error_class mismatch")
            if failure.get("error_code") not in infrastructure_codes:
                errors.append(f"{panel_id}: infrastructure error_code invalid")
        if record.get("result") is not None:
            errors.append(f"{panel_id}: infrastructure-invalid record must not carry a result")
        if isinstance(failure, Mapping):
            if not isinstance(failure.get("attempt_id"), str) or not failure[
                "attempt_id"
            ]:
                errors.append(f"{panel_id}: infrastructure failure attempt_id is required")
            if not isinstance(failure.get("phase"), str) or not failure["phase"]:
                errors.append(f"{panel_id}: infrastructure failure phase is required")
        return errors

    result = record.get("result")
    if not isinstance(result, Mapping):
        return errors + [f"{panel_id}: completed raw panel result must be an object"]
    if result.get("task_id") != panel.get("task_id"):
        errors.append(f"{panel_id}: result task_id mismatch")
    if result.get("mode") != "treatment_set":
        errors.append(f"{panel_id}: result mode must be treatment_set")
    if result.get("execution_order") != panel.get("execution_order"):
        errors.append(f"{panel_id}: result execution_order mismatch")
    if result.get("sampling_seed") != panel.get("sampling_seed"):
        errors.append(f"{panel_id}: result sampling_seed mismatch")
    if result.get("rollout_replica") != panel.get("replica"):
        errors.append(f"{panel_id}: result rollout_replica mismatch")
    if result.get("pilot_manifest_hash") != manifest.get("manifest_hash"):
        errors.append(f"{panel_id}: result manifest provenance mismatch")
    if result.get("pilot_panel_id") != panel_id:
        errors.append(f"{panel_id}: result panel provenance mismatch")
    if result.get("treatment_registry_hash") != manifest["registry"]["registry_hash"]:
        errors.append(f"{panel_id}: result registry hash mismatch")
    task_commitments = result.get("task_commitments")
    task = next(
        (entry for entry in manifest["tasks"] if entry["task_id"] == panel["task_id"]),
        None,
    )
    expected_commitments = (
        {
            "source_sha256": task["source_sha256"],
            "probe_features_sha256": task["probe_features_sha256"],
            "probe_receipt_sha256": task["probe_receipt_sha256"],
        }
        if isinstance(task, Mapping)
        else None
    )
    if not _json_equal(task_commitments, expected_commitments):
        errors.append(f"{panel_id}: runtime source/probe commitments mismatch")

    attempts = result.get("attempts")
    if not isinstance(attempts, Mapping) or set(attempts) != set(panel["execution_order"]):
        errors.append(f"{panel_id}: result attempts must exactly match the panel policies")
        return errors
    planned = {
        attempt["policy_bundle_id"]: attempt
        for attempt in manifest["schedule"][schedule_name]["attempts"]
        if attempt["panel_id"] == panel_id
    }
    for bundle_id in panel["execution_order"]:
        item = attempts.get(bundle_id)
        expected = planned[bundle_id]
        if not isinstance(item, Mapping):
            errors.append(f"{panel_id}/{bundle_id}: raw attempt must be an object")
            continue
        if item.get("attempt_id") != expected["attempt_id"]:
            errors.append(f"{panel_id}/{bundle_id}: attempt_id mismatch")
        capability = expected["policy_capability"]
        treatment = manifest["registry"]["policies"][capability]
        policy = item.get("policy")
        if not isinstance(policy, Mapping):
            errors.append(f"{panel_id}/{bundle_id}: policy receipt missing")
            continue
        expected_policy = {
            "id": treatment["policy_id"],
            "version": treatment["version"],
            "allowed_tools": treatment["allowed_tools"],
            "max_output_tokens": treatment["max_output_tokens"],
            "tool_call_limit": treatment["tool_call_limit"],
            "command_timeout_seconds": treatment["command_timeout_seconds"],
            "wall_time_limit_seconds": treatment["wall_time_limit_seconds"],
            "tool_interface": treatment["tool_interface"],
            "bundle_hash": treatment["bundle_hash"],
            "enforce_budget": treatment["enforce_budget"],
        }
        if policy.get("bundle_hash") != treatment["bundle_hash"]:
            errors.append(f"{panel_id}/{bundle_id}: policy bundle hash mismatch")
        if policy.get("tool_interface") != treatment["tool_interface"]:
            errors.append(f"{panel_id}/{bundle_id}: policy interface mismatch")
        for field, expected_value in expected_policy.items():
            if not _json_equal(policy.get(field), expected_value):
                errors.append(
                    f"{panel_id}/{bundle_id}: policy field {field!r} mismatch"
                )
        system_prompt = policy.get("system_prompt")
        if not isinstance(system_prompt, str) or _sha256_bytes(system_prompt) != treatment[
            "system_prompt_sha256"
        ]:
            errors.append(f"{panel_id}/{bundle_id}: policy system prompt hash mismatch")
    return errors


def _result_infrastructure_failure(
    result: Mapping[str, Any], manifest: Mapping[str, Any]
) -> tuple[dict[str, Any] | None, list[str]]:
    attempts = result.get("attempts")
    if not isinstance(attempts, Mapping):
        return None, []
    non_infrastructure_errors: list[str] = []
    for bundle_id, item in attempts.items():
        if not isinstance(item, Mapping):
            continue
        return_code = item.get("pi_return_code")
        trajectory = item.get("trajectory")
        trace = trajectory.get("tool_trace") if isinstance(trajectory, Mapping) else []
        provider_turn_count = (
            trajectory.get("provider_turn_count")
            if isinstance(trajectory, Mapping)
            else None
        )
        if (
            isinstance(return_code, int)
            and not isinstance(return_code, bool)
            and return_code not in (0, -1)
            and provider_turn_count == 0
        ):
            return {
                "error_class": "infrastructure_invalid",
                "error_code": "provider_transport_error",
                "phase": "provider_request",
                "attempt_id": item.get("attempt_id"),
                "type": "ProviderTransportError",
            }, []
        for entry in trace or []:
            if not isinstance(entry, Mapping):
                continue
            details = entry.get("details")
            if not isinstance(details, Mapping) or details.get("infrastructure_error") is not True:
                continue
            if details.get("error") == "disabled":
                non_infrastructure_errors.append(
                    f"{bundle_id}: disabled semantic specialist surfaced as infrastructure"
                )
                continue
            return {
                "error_class": "infrastructure_invalid",
                "error_code": "browser_transport_error",
                "phase": "model_tool_execution",
                "attempt_id": item.get("attempt_id"),
                "type": "BrowserTransportError",
            }, []
    return None, non_infrastructure_errors


def _specialist_mechanism(
    item: Mapping[str, Any], planned: Mapping[str, Any]
) -> dict[str, Any]:
    expected_tool = (
        "semantic_table"
        if planned["policy_capability"] == "table_specialist"
        else "semantic_form"
    )
    opposite_tool = (
        "semantic_form" if expected_tool == "semantic_table" else "semantic_table"
    )
    trajectory = item.get("trajectory")
    trace = trajectory.get("tool_trace") if isinstance(trajectory, Mapping) else None
    entries = [entry for entry in (trace or []) if isinstance(entry, Mapping)]
    infrastructure_errors = sum(
        1
        for entry in entries
        if isinstance(entry.get("details"), Mapping)
        and entry["details"].get("infrastructure_error") is True
    )
    unavailable = any(entry.get("tool_name") == opposite_tool for entry in entries)
    receipt_valid: bool | None = None
    action_match: bool | None = None
    specialist_called = False
    for entry in entries:
        if entry.get("tool_name") != expected_tool or entry.get("is_error") is True:
            continue
        specialist_called = True
        details = entry.get("details")
        receipt = details.get("semantic_specialist_receipt") if isinstance(details, Mapping) else None
        payload = details.get("semantic_payload") if isinstance(details, Mapping) else None
        if not isinstance(receipt, Mapping) or payload is None:
            continue
        try:
            encoded = canonical_json(payload).encode("utf-8")
        except (TypeError, ValueError):
            continue
        valid = (
            receipt.get("schema_version") == "pyreplab-semantic-specialist-receipt-v1"
            and receipt.get("delivered") is True
            and receipt.get("specialist") == planned["policy_capability"]
            and receipt.get("payload_bytes") == len(encoded)
            and receipt.get("payload_sha256") == hashlib.sha256(encoded).hexdigest()
        )
        receipt_valid = valid
        action_match = valid and receipt.get("action") == expected_tool
        break
    if specialist_called and receipt_valid is None:
        receipt_valid = False
        action_match = False
    policy = item.get("policy") if isinstance(item.get("policy"), Mapping) else {}
    admitted = sum(
        1
        for entry in entries
        if entry.get("tool_name") in {"bash", "unbrowser", "semantic_table", "semantic_form"}
        and entry.get("budget_rejected") is not True
        and entry.get("pre_execution_rejected") is not True
        and entry.get("operation_aborted") is not True
    )
    limit = policy.get("tool_call_limit")
    tool_cap_compliant = (
        not isinstance(limit, bool)
        and isinstance(limit, int)
        and admitted <= limit
    )
    return {
        "infrastructure_errors": infrastructure_errors,
        "unavailable_specialist_found": unavailable,
        "tool_cap_compliant": tool_cap_compliant,
        "specialist_receipt_valid": receipt_valid,
        "specialist_action_match": action_match,
    }


def _itt_error_code(item: Mapping[str, Any]) -> str:
    trajectory = item.get("trajectory")
    trace = trajectory.get("tool_trace") if isinstance(trajectory, Mapping) else []
    if any(
        isinstance(entry, Mapping)
        and (
            entry.get("budget_rejected") is True
            or entry.get("pre_execution_rejected") is True
            or entry.get("operation_aborted") is True
        )
        for entry in (trace or [])
    ):
        return "model_budget_exhaustion"
    if item.get("pi_return_code") == -1:
        return "wall_time_exhaustion_valid_record"
    verification = item.get("verification")
    failure_code = verification.get("failure_code") if isinstance(verification, Mapping) else None
    if failure_code in {"invalid_json", "wrong_type", "missing_key", "wrong_key_type"}:
        return "malformed_model_answer"
    if failure_code == "missing_output":
        return "refusal"
    if failure_code == "nonce_mismatch":
        return "verifier_false_result"
    raise ValueError(
        f"failure_code {failure_code!r} is not a frozen intention-to-treat outcome"
    )


def _safe_row_from_raw_attempt(
    item: Mapping[str, Any],
    planned: Mapping[str, Any],
    manifest: Mapping[str, Any],
    authorization: Mapping[str, Any],
    raw_record_hash: str,
) -> dict[str, Any]:
    verification = item.get("verification")
    if not isinstance(verification, Mapping) or not isinstance(verification.get("success"), bool):
        raise ValueError(f"{planned['attempt_id']}: verification is not mechanically valid")
    if verification.get("verifier_id") != manifest["implementation"]["verifier_id"]:
        raise ValueError(f"{planned['attempt_id']}: verifier identity mismatch")
    if verification.get("verifier_version") != manifest["implementation"]["verifier_version"]:
        raise ValueError(f"{planned['attempt_id']}: verifier version mismatch")
    usage = item.get("usage")
    output_tokens = usage.get("output") if isinstance(usage, Mapping) else None
    if isinstance(output_tokens, bool) or not isinstance(output_tokens, int) or output_tokens < 0:
        raise ValueError(f"{planned['attempt_id']}: usage.output is not a non-negative integer")
    if not _json_equal(item.get("sampling_receipt"), {
        "seed": planned["sampling_seed"],
        "parameters": manifest["runtime"]["sampling"]["parameters"],
    }):
        raise ValueError(f"{planned['attempt_id']}: sampling receipt mismatch")
    return_code = item.get("pi_return_code")
    if (
        isinstance(return_code, bool)
        or not isinstance(return_code, int)
        or return_code not in (0, -1)
    ):
        raise ValueError(f"{planned['attempt_id']}: unexplained Pi return code")
    trajectory = item.get("trajectory")
    provider_turn_count = (
        trajectory.get("provider_turn_count")
        if isinstance(trajectory, Mapping)
        else None
    )
    if (
        isinstance(provider_turn_count, bool)
        or not isinstance(provider_turn_count, int)
        or provider_turn_count < 0
    ):
        raise ValueError(f"{planned['attempt_id']}: provider_turn_count is invalid")
    if return_code == -1 and verification["success"] is not False:
        raise ValueError(
            f"{planned['attempt_id']}: wall-time exit cannot have successful verification"
        )
    mechanism = _specialist_mechanism(item, planned)
    if mechanism["infrastructure_errors"]:
        raise ValueError(f"{planned['attempt_id']}: browser infrastructure error in valid record")
    success = verification["success"]
    if mechanism["specialist_receipt_valid"] is False:
        raise ValueError(f"{planned['attempt_id']}: invalid specialist receipt")
    if mechanism["unavailable_specialist_found"]:
        raise ValueError(f"{planned['attempt_id']}: unavailable specialist was used")
    if mechanism["tool_cap_compliant"] is not True:
        raise ValueError(f"{planned['attempt_id']}: specialist tool cap was not compliant")
    itt_code = None if success else _itt_error_code(item)
    row = craft_attempt_row(
        planned,
        manifest,
        authorization,
        success=success,
        output_tokens=output_tokens,
        error_class=None if success else "intention_to_treat_failure",
        error_code=itt_code,
        specialist_receipt_valid=mechanism["specialist_receipt_valid"],
        specialist_action_match=mechanism["specialist_action_match"],
    )
    row["mechanism"]["unavailable_specialist_found"] = mechanism[
        "unavailable_specialist_found"
    ]
    row["mechanism"]["tool_cap_compliant"] = mechanism["tool_cap_compliant"]
    row["provenance"]["raw_record_hash"] = raw_record_hash
    return row


def export_safe_attempts(
    manifest: Mapping[str, Any],
    authorization: Mapping[str, Any],
    preflight: Mapping[str, Any],
    raw_records: Sequence[Mapping[str, Any]],
    execution_receipt: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate raw panel results and whitelist only safe per-attempt fields."""
    errors = validate_manifest(manifest)
    errors.extend(_check_authorization_binding(authorization, manifest))
    errors.extend(validate_runtime_preflight(preflight, manifest, authorization))
    errors.extend(validate_execution_receipt(execution_receipt, manifest, authorization))
    if errors:
        raise ValueError("safe export prerequisites invalid: " + "; ".join(errors))
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
        raise ValueError("raw_records must be a sequence")

    selected_schedules = execution_receipt["block_schedules"]
    records_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in raw_records:
        record_errors = _validate_raw_panel_record(
            record, manifest, authorization, preflight
        )
        if record_errors:
            raise ValueError("raw panel invalid: " + "; ".join(record_errors))
        key = (record["schedule"], record["panel_id"])
        if key in records_by_key:
            raise ValueError(f"duplicate raw panel record {key!r}")
        records_by_key[key] = record

    rows: list[dict[str, Any]] = []
    raw_hashes: list[str] = []
    for block in (0, 1):
        schedule_name = selected_schedules[str(block)]
        panels = [
            panel
            for panel in manifest["schedule"][schedule_name]["panels"]
            if panel["block"] == block
        ]
        planned_by_id = {
            attempt["attempt_id"]: attempt
            for attempt in manifest["schedule"][schedule_name]["attempts"]
            if attempt["block"] == block
        }
        for panel in panels:
            record = records_by_key.get((schedule_name, panel["panel_id"]))
            if record is None:
                raise ValueError(
                    f"missing selected raw panel {schedule_name}/{panel['panel_id']}"
                )
            if record.get("status") != "completed":
                raise ValueError(
                    f"selected raw panel {schedule_name}/{panel['panel_id']} is not completed"
                )
            raw_hash = record["raw_record_hash"]
            raw_hashes.append(raw_hash)
            for bundle_id in panel["execution_order"]:
                item = record["result"]["attempts"][bundle_id]
                planned = planned_by_id[item["attempt_id"]]
                rows.append(
                    _safe_row_from_raw_attempt(
                        item, planned, manifest, authorization, raw_hash
                    )
                )
    rows.sort(key=lambda row: row["execution_order"])
    export_receipt: dict[str, Any] = {
        "schema_version": SAFE_EXPORT_SCHEMA,
        "manifest_hash": manifest["manifest_hash"],
        "authorization_hash": authorization["authorization_hash"],
        "preflight_hash": preflight["preflight_hash"],
        "execution_receipt_hash": execution_receipt["execution_receipt_hash"],
        "rows": len(rows),
        "attempt_ids_sha256": canonical_hash([row["attempt_id"] for row in rows]),
        "raw_record_hashes_sha256": canonical_hash(raw_hashes),
    }
    _embed_self_hash(export_receipt, "safe_export_hash")
    return rows, export_receipt


def immutable_write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Atomically write JSONL once, allowing only byte-identical idempotence."""
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(canonical_json(dict(row)) + "\n" for row in rows).encode("utf-8")
    if target.exists():
        if target.read_bytes() == payload:
            return
        raise FileExistsError(f"refusing to overwrite {target} with different bytes")
    fd, temporary = tempfile.mkstemp(
        dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _existing_raw_state(
    path: Path,
    manifest: Mapping[str, Any],
    authorization: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[int, dict[str, Any]]]:
    records_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    primary_triggers: dict[int, dict[str, Any]] = {}
    records = _load_raw_panel_records(path)
    trigger_seen: set[int] = set()
    contingency_started = False
    observed_by_schedule: dict[str, list[str]] = {"primary": [], "contingency": []}
    for record in records:
        errors = _validate_raw_panel_record(record, manifest, authorization, preflight)
        if errors:
            raise ValueError("existing raw panel invalid: " + "; ".join(errors))
        key = (record["schedule"], record["panel_id"])
        if key in records_by_key:
            raise ValueError(f"duplicate existing raw panel record {key!r}")
        records_by_key[key] = record
        schedule_name = record["schedule"]
        block = record["block"]
        if schedule_name == "primary":
            if contingency_started:
                raise ValueError("raw ledger returned to primary after contingency started")
            if block in trigger_seen:
                raise ValueError(
                    f"block {block}: primary panel recorded after infrastructure trigger"
                )
        else:
            contingency_started = True
            if block not in trigger_seen:
                raise ValueError(
                    f"block {block}: contingency panel precedes a primary trigger"
                )
        observed_by_schedule[schedule_name].append(record["panel_id"])
        if record["schedule"] == "primary" and record["status"] == "infrastructure_invalid":
            if block in primary_triggers:
                raise ValueError(f"block {block}: more than one primary replacement trigger")
            primary_triggers[block] = {
                "error_class": record["failure"]["error_class"],
                "error_code": record["failure"]["error_code"],
                "attempt_id": record["failure"]["attempt_id"],
                "phase": record["failure"]["phase"],
                "raw_record_hash": record["raw_record_hash"],
            }
            trigger_seen.add(block)
        if record["schedule"] == "contingency" and record["status"] != "completed":
            raise RuntimeError(
                f"block {record['block']}: contingency infrastructure failure; no second replacement"
            )

    expected_primary = [
        panel["panel_id"]
        for panel in manifest["schedule"]["primary"]["panels"]
        if panel["block"] not in trigger_seen
        or any(
            record["panel_id"] == panel["panel_id"]
            and record["status"] == "infrastructure_invalid"
            for record in records
        )
        or panel["panel_id"] in observed_by_schedule["primary"]
    ]
    observed_primary = observed_by_schedule["primary"]
    if observed_primary != expected_primary[: len(observed_primary)] or (
        contingency_started and observed_primary != expected_primary
    ):
        raise ValueError("raw primary ledger does not follow the frozen chronology")
    expected_contingency = [
        panel["panel_id"]
        for panel in manifest["schedule"]["contingency"]["panels"]
        if panel["block"] in trigger_seen
    ]
    observed_contingency = observed_by_schedule["contingency"]
    if observed_contingency != expected_contingency[: len(observed_contingency)]:
        raise ValueError("raw contingency ledger does not follow the frozen chronology")
    return records_by_key, primary_triggers


def _planned_attempts_for_panel(
    manifest: Mapping[str, Any], schedule_name: str, panel_id: str
) -> dict[str, Mapping[str, Any]]:
    return {
        attempt["policy_bundle_id"]: attempt
        for attempt in manifest["schedule"][schedule_name]["attempts"]
        if attempt["panel_id"] == panel_id
    }


def _raw_panel_record(
    manifest: Mapping[str, Any],
    authorization: Mapping[str, Any],
    preflight: Mapping[str, Any],
    schedule_name: str,
    panel: Mapping[str, Any],
    *,
    started_at: str,
    duration_seconds: float,
    result: Mapping[str, Any] | None = None,
    failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    status = "completed" if result is not None else "infrastructure_invalid"
    record: dict[str, Any] = {
        "schema_version": RAW_PANEL_SCHEMA,
        "manifest_hash": manifest["manifest_hash"],
        "authorization_hash": authorization["authorization_hash"],
        "preflight_hash": preflight["preflight_hash"],
        "stage_b_id": manifest["stage_b_id"],
        "schedule": schedule_name,
        "panel_id": panel["panel_id"],
        "block": panel["block"],
        "task_id": panel["task_id"],
        "replica": panel["replica"],
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": duration_seconds,
        "status": status,
        "result": dict(result) if result is not None else None,
        "failure": dict(failure) if failure is not None else None,
    }
    record["raw_record_hash"] = _raw_record_hash(record)
    return record


def _run_live_panel(
    manifest: Mapping[str, Any],
    authorization: Mapping[str, Any],
    preflight: Mapping[str, Any],
    registry_path: Path,
    config: RemoteConfig,
    schedule_name: str,
    panel: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    pi_binary: str,
    provider: str,
    model: str,
    thinking: str,
    unbrowser_binary: str,
) -> dict[str, Any]:
    planned = _planned_attempts_for_panel(manifest, schedule_name, panel["panel_id"])
    args = argparse.Namespace(
        family="routing_fixture",
        seed=int(task["seed"]),
        difficulty=str(task["difficulty"]),
        task_role=CANARY_ROW_STATUS,
        rollout_replica=int(panel["replica"]),
        sampling_seed=int(panel["sampling_seed"]),
        pilot_manifest_hash=str(manifest["manifest_hash"]),
        pilot_panel_id=str(panel["panel_id"]),
        treatment_registry=str(registry_path),
        treatments=",".join(panel["execution_order"]),
        preserve_treatment_order=True,
        attempt_ids_by_treatment={
            bundle_id: planned[bundle_id]["attempt_id"]
            for bundle_id in panel["execution_order"]
        },
        pi=pi_binary,
        provider=provider,
        model=model,
        thinking=thinking,
        model_switch_extension=None,
        unbrowser_binary=unbrowser_binary,
    )
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    try:
        result = run_registered_treatments(
            Path(__file__).resolve().parents[2], config, args
        )
    except AttemptExecutionError as error:
        if error.error_class != "infrastructure_invalid":
            raise RuntimeError(
                f"non-replaceable live panel failure on {panel['panel_id']}: {error}"
            ) from error
        return _raw_panel_record(
            manifest,
            authorization,
            preflight,
            schedule_name,
            panel,
            started_at=started_at,
            duration_seconds=round(time.monotonic() - started, 3),
            failure={
                "error_class": error.error_class,
                "error_code": error.error_code,
                "phase": error.phase,
                "attempt_id": error.attempt_id
                or next(iter(planned.values()))["attempt_id"],
                "type": type(error).__name__,
            },
        )
    except Exception as error:
        raise RuntimeError(
            f"unclassified live panel failure on {panel['panel_id']}: {error}"
        ) from error
    infrastructure_failure, mechanism_errors = _result_infrastructure_failure(
        result, manifest
    )
    if mechanism_errors:
        raise RuntimeError("mechanism-invalid live result: " + "; ".join(mechanism_errors))
    if infrastructure_failure is not None:
        return _raw_panel_record(
            manifest,
            authorization,
            preflight,
            schedule_name,
            panel,
            started_at=started_at,
            duration_seconds=round(time.monotonic() - started, 3),
            failure=infrastructure_failure,
        )
    record = _raw_panel_record(
        manifest,
        authorization,
        preflight,
        schedule_name,
        panel,
        started_at=started_at,
        duration_seconds=round(time.monotonic() - started, 3),
        result=result,
    )
    errors = _validate_raw_panel_record(record, manifest, authorization, preflight)
    if errors:
        raise RuntimeError("live panel result invalid: " + "; ".join(errors))
    return record


def run_stage_b(
    manifest_path: str | Path,
    authorization_path: str | Path,
    registry_path: str | Path,
    stage_a_manifest_path: str | Path,
    stage_a_report_path: str | Path,
    raw_output_path: str | Path,
    safe_output_path: str | Path,
    execution_receipt_path: str | Path,
    safe_export_receipt_path: str | Path,
    preflight_path: str | Path,
    *,
    pi_binary: str,
    provider: str,
    model: str,
    thinking: str,
    unbrowser_binary: str,
    model_artifact: str,
    llama_server_binary: str,
) -> dict[str, Any]:
    """Execute the authorized Stage-B panel sequentially and export safe rows.

    Execution never computes aggregate outcomes. A primary infrastructure
    failure quarantines that complete block and activates its frozen
    contingency schedule. Contingency failure, unclassified failure, or any
    malformed completed record fails closed with no second replacement.
    """
    manifest = _load_json(manifest_path)
    authorization = _load_json(authorization_path)
    registry_file = _resolve(registry_path)
    registry = TreatmentRegistry.load(registry_file)
    manifest_errors = validate_manifest(manifest, registry=registry)
    if manifest_errors:
        raise ValueError("stage-b manifest invalid: " + "; ".join(manifest_errors))
    if manifest.get("seed") != DEFAULT_STAGE_B_SEED:
        raise ValueError(
            "live Stage-B execution is restricted to the frozen production seed "
            f"{DEFAULT_STAGE_B_SEED}"
        )
    authorization_errors = validate_authorization(
        authorization, manifest, stage_a_manifest_path, stage_a_report_path
    )
    if authorization_errors:
        raise ValueError("stage-b authorization invalid: " + "; ".join(authorization_errors))
    if file_sha256(registry_file) != manifest["registry"]["registry_sha256"]:
        raise ValueError("registry bytes do not match the frozen manifest")
    pins = manifest["runtime"]["pins"]
    if (
        provider != pins["provider"]
        or model != pins["model_alias"]
        or thinking != pins["thinking"]
    ):
        raise ValueError("provider/model/thinking do not match frozen runtime pins")

    preflight_file = _resolve(preflight_path)
    preflight = build_runtime_preflight(
        manifest,
        authorization,
        pi_binary=pi_binary,
        thinking=thinking,
        unbrowser_binary=unbrowser_binary,
        model_artifact=model_artifact,
        llama_server_binary=llama_server_binary,
    )
    if preflight_file.exists():
        existing_preflight = _load_json(preflight_file)
        errors = validate_runtime_preflight(
            existing_preflight, manifest, authorization
        )
        if errors:
            raise RuntimeError("existing runtime preflight invalid: " + "; ".join(errors))
        stable_keys = (
            "manifest_hash",
            "authorization_hash",
            "stage_b_id",
            "code_revision",
            "source_tree_hash",
            "worktree_clean",
            "worktree_status_hash",
            "runtime_pins",
            "remote_identity",
        )
        if any(
            not _json_equal(existing_preflight.get(key), preflight.get(key))
            for key in stable_keys
        ):
            raise RuntimeError("runtime preflight identity changed across resume")
        preflight = existing_preflight
    else:
        immutable_write(preflight_file, preflight)

    raw_output = _resolve(raw_output_path)
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    active_path = raw_output.with_suffix(raw_output.suffix + ".active.json")
    if active_path.exists():
        raise RuntimeError(
            "unfinished Stage-B panel marker creates uncertain attempt state; "
            "adjudicate before resuming"
        )
    records_by_key, triggers = _existing_raw_state(
        raw_output, manifest, authorization, preflight
    )
    task_by_id = {task["task_id"]: task for task in manifest["tasks"]}
    config = _remote_config_from_manifest(manifest)
    panels_run = 0

    def execute_schedule(schedule_name: str, active_blocks: set[int]) -> None:
        nonlocal panels_run
        for panel in manifest["schedule"][schedule_name]["panels"]:
            block = panel["block"]
            if block not in active_blocks:
                continue
            key = (schedule_name, panel["panel_id"])
            existing = records_by_key.get(key)
            if existing is not None:
                if existing["status"] == "infrastructure_invalid":
                    if schedule_name == "contingency":
                        raise RuntimeError(
                            f"block {block}: contingency already failed; no second replacement"
                        )
                    triggers[block] = {
                        "error_class": existing["failure"]["error_class"],
                        "error_code": existing["failure"]["error_code"],
                        "attempt_id": existing["failure"]["attempt_id"],
                        "phase": existing["failure"]["phase"],
                        "raw_record_hash": existing["raw_record_hash"],
                    }
                    active_blocks.discard(block)
                continue
            immutable_write(
                active_path,
                {
                    "manifest_hash": manifest["manifest_hash"],
                    "preflight_hash": preflight["preflight_hash"],
                    "schedule": schedule_name,
                    "panel_id": panel["panel_id"],
                },
            )
            record = _run_live_panel(
                manifest,
                authorization,
                preflight,
                registry_file,
                config,
                schedule_name,
                panel,
                task_by_id[panel["task_id"]],
                pi_binary=pi_binary,
                provider=provider,
                model=model,
                thinking=thinking,
                unbrowser_binary=unbrowser_binary,
            )
            _append_result(raw_output, record)
            active_path.unlink()
            records_by_key[key] = record
            panels_run += 1
            if record["status"] == "infrastructure_invalid":
                if schedule_name == "contingency":
                    raise RuntimeError(
                        f"block {block}: contingency infrastructure failure; no second replacement"
                    )
                triggers[block] = {
                    "error_class": record["failure"]["error_class"],
                    "error_code": record["failure"]["error_code"],
                    "attempt_id": record["failure"]["attempt_id"],
                    "phase": record["failure"]["phase"],
                    "raw_record_hash": record["raw_record_hash"],
                }
                active_blocks.discard(block)

    primary_blocks = {block for block in (0, 1) if block not in triggers}
    execute_schedule("primary", primary_blocks)
    contingency_blocks = set(triggers)
    execute_schedule("contingency", contingency_blocks)

    block_schedules = {
        block: ("contingency" if block in triggers else "primary")
        for block in (0, 1)
    }
    execution_receipt = build_execution_receipt(
        manifest, authorization, block_schedules, triggers
    )
    immutable_write(execution_receipt_path, execution_receipt)
    raw_records = list(records_by_key.values())
    safe_rows, export_receipt = export_safe_attempts(
        manifest, authorization, preflight, raw_records, execution_receipt
    )
    immutable_write_jsonl(safe_output_path, safe_rows)
    immutable_write(safe_export_receipt_path, export_receipt)
    return {
        "manifest_hash": manifest["manifest_hash"],
        "authorization_hash": authorization["authorization_hash"],
        "preflight_hash": preflight["preflight_hash"],
        "execution_receipt_hash": execution_receipt["execution_receipt_hash"],
        "safe_export_hash": export_receipt["safe_export_hash"],
        "block_schedules": execution_receipt["block_schedules"],
        "panels_run": panels_run,
        "safe_attempts": len(safe_rows),
        "raw_output": str(raw_output),
        "safe_output": str(_resolve(safe_output_path)),
    }


# ---------------------------------------------------------------------------
# safe-attempt row loading and crafting
# ---------------------------------------------------------------------------


def load_safe_attempts_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file of flat runtime safe-attempt rows.

    Blank lines are ignored; every other line must decode to a JSON object.
    """
    rows: list[dict[str, Any]] = []
    text = Path(path).expanduser().resolve().read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on line {lineno}: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"line {lineno}: each safe-attempt row must be a JSON object")
        rows.append(row)
    return rows


def craft_attempt_row(
    planned: Mapping[str, Any],
    manifest: Mapping[str, Any],
    authorization: Mapping[str, Any],
    *,
    success: bool,
    output_tokens: int = 0,
    error_class: str | None = None,
    error_code: str | None = None,
    specialist_receipt_valid: bool | None = None,
    specialist_action_match: bool | None = None,
) -> dict[str, Any]:
    """Craft one runtime safe-attempt row satisfying the frozen row contract.

    ``success`` drives ``verification.success``.  A successful row requires
    ``error_class``/``error_code`` of ``None``; a failed row must use an
    intention-to-treat class/code.  Tests tweak fields to exercise the
    invalid/replacement paths.
    """
    binding = manifest["registry"]["policies"][planned["policy_capability"]]
    if success is False and error_class is None and error_code is None:
        error_class = "intention_to_treat_failure"
        error_code = "verifier_false_result"
    return {
        "schema_version": planned["schema_version"],
        "attempt_id": planned["attempt_id"],
        "panel_id": planned["panel_id"],
        "schedule": planned["schedule"],
        "task_id": planned["task_id"],
        "task_ordinal": planned["task_ordinal"],
        "block": planned["block"],
        "stratum": planned["stratum"],
        "difficulty": planned["difficulty"],
        "fixture_id": planned["fixture_id"],
        "replica": planned["replica"],
        "policy_capability": planned["policy_capability"],
        "policy_bundle_id": planned["policy_bundle_id"],
        "policy_bundle_hash": planned["policy_bundle_hash"],
        "arm_position": planned["arm_position"],
        "execution_order": planned["execution_order"],
        "sampling_seed": planned["sampling_seed"],
        "source_sha256": planned["source_sha256"],
        "probe_receipt_sha256": planned["probe_receipt_sha256"],
        "route_receipt_sha256": planned["route_receipt_sha256"],
        "canary_row_status": planned["canary_row_status"],
        "canary_exclusion": planned["canary_exclusion"],
        "status": "completed",
        "error_class": error_class,
        "error_code": error_code,
        "sampling_receipt": {
            "seed": planned["sampling_seed"],
            "parameters": dict(manifest["runtime"]["sampling"]["parameters"]),
        },
        "verification": {
            "success": success,
            "verifier_id": manifest["implementation"]["verifier_id"],
            "verifier_version": manifest["implementation"]["verifier_version"],
        },
        "usage": {"output": output_tokens},
        "mechanism": {
            "specialist": planned["policy_capability"],
            "bundle": planned["policy_bundle_id"],
            "interface": binding["tool_interface"],
            "infrastructure_errors": 0,
            "unavailable_specialist_found": False,
            "tool_cap_compliant": True,
            "specialist_receipt_valid": specialist_receipt_valid,
            "specialist_action_match": specialist_action_match,
        },
        "provenance": {
            "manifest_hash": manifest["manifest_hash"],
            "authorization_hash": authorization["authorization_hash"],
            "registry_hash": manifest["registry"]["registry_hash"],
            "source_tree_hash": manifest["implementation"]["source_tree_hash"],
        },
    }


def craft_runtime_rows(
    manifest: Mapping[str, Any],
    authorization: Mapping[str, Any],
    *,
    block_schedules: Mapping[Any, str] | None = None,
    success: Any,
    output_tokens: Any = 0,
    error_class: Any = None,
    error_code: Any = None,
) -> list[dict[str, Any]]:
    """Craft the full set of 96 runtime rows for the selected schedules.

    ``success``, ``output_tokens``, ``error_class``, and ``error_code`` may be
    constants or callables receiving the planned attempt (letting a test assign
    per-task-policy outcomes).  When ``block_schedules`` is omitted, both
    blocks run their primary schedule.
    """
    if block_schedules is None:
        normalized = {"0": "primary", "1": "primary"}
    else:
        normalized = _normalize_block_schedules(manifest, block_schedules)
    rows: list[dict[str, Any]] = []
    for block in (0, 1):
        schedule_name = normalized[str(block)]
        for attempt in manifest["schedule"][schedule_name]["attempts"]:
            if attempt["block"] != block:
                continue
            success_value = success(attempt) if callable(success) else success
            tokens_value = output_tokens(attempt) if callable(output_tokens) else output_tokens
            error_class_value = error_class(attempt) if callable(error_class) else error_class
            error_code_value = error_code(attempt) if callable(error_code) else error_code
            rows.append(
                craft_attempt_row(
                    attempt,
                    manifest,
                    authorization,
                    success=success_value,
                    output_tokens=tokens_value,
                    error_class=error_class_value,
                    error_code=error_code_value,
                )
            )
    return rows


# ---------------------------------------------------------------------------
# Stage-B outcome-metric analyzer and gate report
# ---------------------------------------------------------------------------


def _check_authorization_binding(
    authorization: Any, manifest: Mapping[str, Any]
) -> list[str]:
    """Lightweight authorization binding check (schema/self-hash/manifest)."""
    errors: list[str] = []
    if not isinstance(authorization, Mapping):
        return ["authorization must be an object"]
    if authorization.get("schema_version") != AUTHORIZATION_SCHEMA:
        errors.append(
            f"authorization schema_version must be {AUTHORIZATION_SCHEMA!r}"
        )
    errors.extend(_verify_self_hash(authorization, "authorization_hash"))
    if authorization.get("manifest_hash") != manifest.get("manifest_hash"):
        errors.append("authorization does not bind the supplied manifest")
    if authorization.get("decision") != "probe_pass" or authorization.get("passed") is not True:
        errors.append("authorization must carry decision probe_pass / passed True")
    return errors


def _validate_attempt_identity(
    row: Mapping[str, Any], planned: Mapping[str, Any]
) -> list[str]:
    errors: list[str] = []
    attempt_id = planned["attempt_id"]
    for field in ATTEMPT_IDENTITY_FIELDS:
        if not _json_equal(row.get(field), planned.get(field)):
            errors.append(
                f"{attempt_id}: identity field {field!r} does not match the "
                "planned attempt"
            )
    return errors


def _validate_attempt_runtime(
    row: Mapping[str, Any],
    planned: Mapping[str, Any],
    manifest: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    attempt_id = planned["attempt_id"]
    for field in SAFE_ATTEMPT_REQUIRED_FIELDS:
        if field not in row:
            errors.append(f"{attempt_id}: missing required field {field!r}")
    if errors:
        return errors

    if row.get("status") != "completed":
        errors.append(f"{attempt_id}: status must be 'completed'")

    expected_sampling = {
        "seed": planned["sampling_seed"],
        "parameters": manifest["runtime"]["sampling"]["parameters"],
    }
    if not _json_equal(row.get("sampling_receipt"), expected_sampling):
        errors.append(
            f"{attempt_id}: sampling_receipt must carry the planned panel-common "
            "seed and the frozen sampling parameters"
        )

    verification = row.get("verification")
    success: Any = None
    if not isinstance(verification, Mapping):
        errors.append(f"{attempt_id}: verification must be an object")
    else:
        success = verification.get("success")
        if not isinstance(success, bool):
            errors.append(f"{attempt_id}: verification.success must be a boolean")
        if verification.get("verifier_id") != manifest["implementation"]["verifier_id"]:
            errors.append(
                f"{attempt_id}: verification.verifier_id does not match the "
                "manifest implementation"
            )
        if verification.get("verifier_version") != manifest["implementation"]["verifier_version"]:
            errors.append(
                f"{attempt_id}: verification.verifier_version does not match the "
                "manifest implementation"
            )

    usage = row.get("usage")
    if not isinstance(usage, Mapping):
        errors.append(f"{attempt_id}: usage must be an object")
    else:
        output = usage.get("output")
        if isinstance(output, bool) or not isinstance(output, int) or output < 0:
            errors.append(
                f"{attempt_id}: usage.output must be a non-negative integer "
                "(not a boolean)"
            )

    provenance = row.get("provenance")
    if not isinstance(provenance, Mapping):
        errors.append(f"{attempt_id}: provenance must be an object")
    else:
        expected_provenance = {
            "manifest_hash": manifest["manifest_hash"],
            "authorization_hash": authorization["authorization_hash"],
            "registry_hash": manifest["registry"]["registry_hash"],
            "source_tree_hash": manifest["implementation"]["source_tree_hash"],
        }
        for key, value in expected_provenance.items():
            if provenance.get(key) != value:
                errors.append(f"{attempt_id}: provenance.{key} does not bind the frozen value")

    mechanism = row.get("mechanism")
    if not isinstance(mechanism, Mapping):
        errors.append(f"{attempt_id}: mechanism must be an object")
    else:
        binding = manifest["registry"]["policies"].get(planned["policy_capability"])
        if mechanism.get("specialist") != planned["policy_capability"]:
            errors.append(
                f"{attempt_id}: mechanism.specialist must be the assigned specialist"
            )
        if mechanism.get("bundle") != planned["policy_bundle_id"]:
            errors.append(f"{attempt_id}: mechanism.bundle must be the assigned policy bundle")
        if binding is None or mechanism.get("interface") != binding.get("tool_interface"):
            errors.append(f"{attempt_id}: mechanism.interface must match the policy tool interface")
        if mechanism.get("infrastructure_errors") != 0:
            errors.append(f"{attempt_id}: mechanism.infrastructure_errors must be 0")
        if mechanism.get("unavailable_specialist_found") is not False:
            errors.append(f"{attempt_id}: mechanism.unavailable_specialist_found must be False")
        if mechanism.get("tool_cap_compliant") is not True:
            errors.append(f"{attempt_id}: mechanism.tool_cap_compliant must be True")
        specialist_receipt_valid = mechanism.get("specialist_receipt_valid")
        if specialist_receipt_valid not in (None, True):
            errors.append(
                f"{attempt_id}: mechanism.specialist_receipt_valid must be None or True"
            )
        if specialist_receipt_valid is True and mechanism.get("specialist_action_match") is not True:
            errors.append(
                f"{attempt_id}: mechanism.specialist_action_match must be True when "
                "specialist_receipt_valid is True"
            )

    error_class = row.get("error_class")
    error_code = row.get("error_code")
    itt_codes = manifest["safe_attempt"]["intention_to_treat_failure_codes"]
    if success is True:
        if error_class is not None or error_code is not None:
            errors.append(
                f"{attempt_id}: successful rows require error_class and error_code "
                "to be None"
            )
    elif success is False:
        if error_class != "intention_to_treat_failure":
            errors.append(
                f"{attempt_id}: failed verification must use error_class "
                "'intention_to_treat_failure'"
            )
        if error_code not in itt_codes:
            errors.append(
                f"{attempt_id}: error_code {error_code!r} is not an allowed "
                "intention-to-treat code"
            )
    return errors


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _subset_routed(
    task_metrics: Mapping[str, Mapping[str, Any]],
    task_ids: Sequence[str],
    route_field: str,
    lam: float,
) -> dict[str, float]:
    successes = [
        task_metrics[t]["cells"][task_metrics[t][route_field]]["success_mean"]
        for t in task_ids
    ]
    tokens = [
        task_metrics[t]["cells"][task_metrics[t][route_field]]["token_mean"]
        for t in task_ids
    ]
    success_mean = _mean(successes)
    token_mean = _mean(tokens)
    return {
        "success_mean": success_mean,
        "token_mean": token_mean,
        "utility": success_mean - lam * (token_mean / COST_UNITS_PER_TOKEN),
    }


def _subset_fixed_arms(
    task_metrics: Mapping[str, Mapping[str, Any]],
    task_ids: Sequence[str],
    lam: float,
) -> dict[str, dict[str, float]]:
    arms: dict[str, dict[str, float]] = {}
    for capability in SPECIALIST_CAPABILITIES:
        successes = [
            task_metrics[t]["cells"][capability]["success_mean"] for t in task_ids
        ]
        tokens = [
            task_metrics[t]["cells"][capability]["token_mean"] for t in task_ids
        ]
        success_mean = _mean(successes)
        token_mean = _mean(tokens)
        arms[capability] = {
            "success_mean": success_mean,
            "token_mean": token_mean,
            "utility": success_mean - lam * (token_mean / COST_UNITS_PER_TOKEN),
        }
    return arms


def _best_fixed_policy(
    arms: Mapping[str, Mapping[str, float]],
    capability_order: Sequence[str],
    *,
    metric: str,
) -> str:
    """Select the best fixed specialist within a declared subset.

    Tie-break: higher success, then lower mean tokens, then immutable registry
    order.  The success and utility comparators are selected independently.
    """

    def key(capability: str) -> tuple[Any, ...]:
        arm = arms[capability]
        index = capability_order.index(capability)
        if metric == "success":
            return (arm["success_mean"], -arm["token_mean"], -index)
        return (arm["utility"], arm["success_mean"], -arm["token_mean"], -index)

    return max(arms, key=key)


def _bootstrap_utility_lower_bound(
    task_metrics: Mapping[str, Mapping[str, Any]],
    capability_order: Sequence[str],
    seed: int,
    draws: int,
    primary_lambda: float,
    cost_units: int,
) -> float:
    """One-sided 90% lower bound on pooled routed utility lift.

    Mirrors the frozen bootstrap: ``draws`` draws with seed ``seed``, each draw
    resampling three whole tasks with replacement inside every block x stratum
    cell (retaining both policies, both replicas, and the frozen route), the
    hindsight-better fixed specialist recomputed inside every draw, and the
    nearest-rank empirical 10th percentile as the one-sided 90% lower bound.
    """
    if isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0:
        raise ValueError("bootstrap draws must be a positive integer")
    if isinstance(cost_units, bool) or not isinstance(cost_units, int) or cost_units <= 0:
        raise ValueError("bootstrap cost units must be a positive integer")

    table = "table_specialist"
    form = "form_specialist"
    cells: dict[tuple[int, str], list[str]] = {}
    for task_id, metric in task_metrics.items():
        task = metric["task"]
        cells.setdefault((task["block"], task["stratum"]), []).append(task_id)
    expected_cells = {(block, stratum) for block in (0, 1) for stratum in STRATA}
    if set(cells) != expected_cells or any(
        len(task_ids) != TASKS_PER_STRATUM_PER_BLOCK for task_ids in cells.values()
    ):
        raise ValueError(
            "bootstrap requires exactly three tasks in every block x stratum cell"
        )
    cell_keys = sorted(cells)

    pre: dict[str, tuple[float, float, float, float, float, float]] = {}
    for task_id, metric in task_metrics.items():
        c = metric["cells"]
        pre[task_id] = (
            c[metric["route"]]["success_mean"],
            c[metric["route"]]["token_mean"],
            c[table]["success_mean"],
            c[table]["token_mean"],
            c[form]["success_mean"],
            c[form]["token_mean"],
        )

    table_index = capability_order.index(table)
    form_index = capability_order.index(form)
    lam = primary_lambda
    rng = random.Random(seed)
    total = len(cell_keys) * 3
    lifts: list[float] = []
    for _ in range(draws):
        routed_success = routed_tokens = 0.0
        table_success = table_tokens = 0.0
        form_success = form_tokens = 0.0
        for key in cell_keys:
            task_ids = cells[key]
            for _ in range(3):
                (
                    r_success,
                    r_tokens,
                    t_success,
                    t_tokens,
                    f_success,
                    f_tokens,
                ) = pre[rng.choice(task_ids)]
                routed_success += r_success
                routed_tokens += r_tokens
                table_success += t_success
                table_tokens += t_tokens
                form_success += f_success
                form_tokens += f_tokens
        routed_success /= total
        routed_tokens /= total
        table_success /= total
        table_tokens /= total
        form_success /= total
        form_tokens /= total
        routed_utility = routed_success - lam * routed_tokens / cost_units
        table_utility = table_success - lam * table_tokens / cost_units
        form_utility = form_success - lam * form_tokens / cost_units
        table_key = (table_utility, table_success, -table_tokens, -table_index)
        form_key = (form_utility, form_success, -form_tokens, -form_index)
        best_utility = form_utility if form_key > table_key else table_utility
        lifts.append(routed_utility - best_utility)
    lifts.sort()
    rank = math.ceil(0.10 * len(lifts))
    return lifts[rank - 1]


def _finalize_report(
    manifest: Mapping[str, Any],
    authorization: Mapping[str, Any],
    receipt_hash: str | None,
    draws: int,
    *,
    valid: bool,
    gates_passed: bool,
    errors: Sequence[str],
    analysis: Mapping[str, Any],
    checks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    decision = (
        DECISION_INVALID
        if not valid
        else (DECISION_PASS if gates_passed else DECISION_NO_GO)
    )
    report: dict[str, Any] = {
        "schema_version": GATE_REPORT_SCHEMA,
        "manifest_hash": manifest.get("manifest_hash"),
        "authorization_hash": authorization.get("authorization_hash"),
        "stage_b_id": manifest.get("stage_b_id"),
        "execution_receipt_hash": receipt_hash,
        "configured_draws": draws,
        "valid": valid,
        "gates_passed": gates_passed,
        "decision": decision,
        "errors": sorted(set(errors)),
        "analysis": dict(analysis),
        "checks": list(checks),
    }
    _embed_self_hash(report, "report_hash")
    return report


def analyze_stage_b(
    manifest: Mapping[str, Any],
    authorization: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    execution_receipt: Mapping[str, Any] | None = None,
    _bootstrap_draws: int | None = None,
) -> dict[str, Any]:
    """Analyze runtime safe-attempt rows and emit a self-hashed gate report.

    This is the pure Stage-B outcome-metric analyzer: no agent model is
    executed, no outcome is peeking-adjusted, and the routed arm is
    reconstructed from the frozen route receipt plus the complete crossed
    panel (never executed as a third arm).

    The report's ``decision`` is ``routing_smoke_pass`` when every frozen
    threshold is met, ``routing_smoke_no_go`` when the rows are mechanically
    valid but a threshold fails, and ``invalid`` when the manifest,
    authorization, execution receipt, or any row violates the frozen contract
    (missing/duplicate/extra rows, identity, governance, sampling, cost,
    verifier, mechanism, or non-ITT error classes).

    ``_bootstrap_draws`` is a private test-only override of the number of
    bootstrap draws; the authoritative report uses the manifest's frozen
    ``bootstrap.draws`` (100000) and ``validate_gate_report`` rejects reports
    whose ``configured_draws`` differ from the manifest.
    """
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be an object")
    if not isinstance(authorization, Mapping):
        raise ValueError("authorization must be an object")

    try:
        manifest_draws = _require_int(
            manifest["analysis"]["bootstrap"]["draws"], "bootstrap.draws"
        )
    except (KeyError, TypeError, ValueError):
        manifest_draws = BOOTSTRAP_DRAWS
    draws = manifest_draws if _bootstrap_draws is None else _require_int(
        _bootstrap_draws, "_bootstrap_draws"
    )
    if draws <= 0:
        raise ValueError("bootstrap draws must be a positive integer")

    structural_errors: list[str] = []
    structural_errors.extend(validate_manifest(manifest))
    structural_errors.extend(_check_authorization_binding(authorization, manifest))

    receipt_hash: str | None = None
    block_schedules: dict[str, str] = {"0": "primary", "1": "primary"}
    if execution_receipt is not None:
        receipt_errors = validate_execution_receipt(
            execution_receipt, manifest, authorization
        )
        if receipt_errors:
            structural_errors.extend(
                f"execution receipt: {error}" for error in receipt_errors
            )
        else:
            receipt_hash = execution_receipt.get("execution_receipt_hash")
            block_schedules = dict(execution_receipt["block_schedules"])

    if structural_errors:
        return _finalize_report(
            manifest,
            authorization,
            receipt_hash,
            draws,
            valid=False,
            gates_passed=False,
            errors=structural_errors,
            analysis={},
            checks=[
                {
                    "id": "structural_validity",
                    "kind": "structural",
                    "label": (
                        "manifest, authorization, and execution-receipt validity"
                    ),
                    "passed": False,
                    "detail": {},
                }
            ],
        )

    # ---- selected schedules and planned attempts --------------------------
    planned_by_id: dict[str, dict[str, Any]] = {}
    for block in (0, 1):
        schedule_name = block_schedules[str(block)]
        for attempt in manifest["schedule"][schedule_name]["attempts"]:
            if attempt["block"] == block:
                planned_by_id[attempt["attempt_id"]] = attempt
    expected_ids = set(planned_by_id)

    # ---- completeness ------------------------------------------------------
    completeness_errors: list[str] = []
    rows_by_id: dict[str, dict[str, Any]] = {}
    if not isinstance(attempts, Sequence) or isinstance(attempts, (str, bytes)):
        completeness_errors.append("attempts must be a sequence of safe-attempt rows")
        rows: list[Mapping[str, Any]] = []
    else:
        rows = list(attempts)
    if len(rows) != ATTEMPTS_PER_SCHEDULE:
        completeness_errors.append(
            f"expected exactly {ATTEMPTS_PER_SCHEDULE} safe-attempt rows, got {len(rows)}"
        )
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            completeness_errors.append(f"row {index} must be an object")
            continue
        attempt_id = row.get("attempt_id")
        if not isinstance(attempt_id, str) or not attempt_id:
            completeness_errors.append(f"row {index} is missing a string attempt_id")
            continue
        if attempt_id in seen:
            completeness_errors.append(f"duplicate attempt_id {attempt_id!r}")
        seen.add(attempt_id)
        rows_by_id[attempt_id] = dict(row)
    for attempt_id in sorted(expected_ids - set(rows_by_id)):
        completeness_errors.append(f"missing planned attempt {attempt_id}")
    for attempt_id in sorted(set(rows_by_id) - expected_ids):
        completeness_errors.append(f"unexpected attempt {attempt_id}")

    # ---- identity and runtime contract -------------------------------------
    identity_errors: list[str] = []
    runtime_errors: list[str] = []
    for attempt_id in sorted(expected_ids):
        row = rows_by_id.get(attempt_id)
        if row is None:
            continue
        identity_errors.extend(_validate_attempt_identity(row, planned_by_id[attempt_id]))
        runtime_errors.extend(
            _validate_attempt_runtime(row, planned_by_id[attempt_id], manifest, authorization)
        )

    structural_errors = completeness_errors + identity_errors + runtime_errors
    if structural_errors:
        return _finalize_report(
            manifest,
            authorization,
            receipt_hash,
            draws,
            valid=False,
            gates_passed=False,
            errors=structural_errors,
            analysis={},
            checks=[
                {
                    "id": "attempt_completeness",
                    "kind": "structural",
                    "label": (
                        "exactly 96 planned attempts with no duplicate, missing, "
                        "or extra rows"
                    ),
                    "passed": not completeness_errors,
                    "detail": {},
                },
                {
                    "id": "attempt_identity",
                    "kind": "structural",
                    "label": "every identity field matches the planned attempt",
                    "passed": not identity_errors,
                    "detail": {},
                },
                {
                    "id": "attempt_runtime_contract",
                    "kind": "structural",
                    "label": "runtime row contract (status, sampling, verification, usage, provenance, mechanism, error coherence)",
                    "passed": not runtime_errors,
                    "detail": {},
                },
            ],
        )

    # ---- metrics ------------------------------------------------------------
    primary_lambda = manifest["analysis"]["primary_lambda"]
    cost_units = manifest["analysis"]["cost_units_per_token"]
    capability_order: list[str] = []
    for bundle_id in manifest["registry"]["policy_order"]:
        for capability, binding in manifest["registry"]["policies"].items():
            if binding["bundle_id"] == bundle_id:
                capability_order.append(capability)

    cell_data: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    for attempt_id, planned in planned_by_id.items():
        row = rows_by_id[attempt_id]
        key = (planned["task_id"], planned["policy_capability"])
        cell_data.setdefault(key, {})[planned["replica"]] = {
            "success": row["verification"]["success"],
            "output": row["usage"]["output"],
        }

    task_metrics: dict[str, dict[str, Any]] = {}
    for task in manifest["tasks"]:
        task_id = task["task_id"]
        cells: dict[str, dict[str, float]] = {}
        for capability in SPECIALIST_CAPABILITIES:
            replicas = cell_data[(task_id, capability)]
            cells[capability] = {
                "success_mean": (
                    replicas[0]["success"] + replicas[1]["success"]
                )
                / 2.0,
                "token_mean": (replicas[0]["output"] + replicas[1]["output"])
                / 2.0,
            }
        task_metrics[task_id] = {
            "task": task,
            "route": manifest["route_receipts"][task_id]["combined_route"],
            "prompt_only_route": manifest["route_receipts"][task_id]["prompt_only_route"],
            "cells": cells,
        }

    discordant_overall = 0
    discordant_per_block: dict[int, int] = {0: 0, 1: 0}
    for (task_id, _capability), replicas in cell_data.items():
        if replicas[0]["success"] != replicas[1]["success"]:
            discordant_overall += 1
            discordant_per_block[task_metrics[task_id]["task"]["block"]] += 1

    def subset_ids(
        block: int | None = None, stratum: str | None = None
    ) -> list[str]:
        return [
            task_id
            for task_id, metric in task_metrics.items()
            if (block is None or metric["task"]["block"] == block)
            and (stratum is None or metric["task"]["stratum"] == stratum)
        ]

    pooled_ids = subset_ids()
    routed_success = _subset_routed(
        task_metrics, pooled_ids, "route", primary_lambda
    )["success_mean"]
    pooled_arms = _subset_fixed_arms(task_metrics, pooled_ids, primary_lambda)
    best_fixed_success = _best_fixed_policy(
        pooled_arms, capability_order, metric="success"
    )
    best_fixed_success_value = pooled_arms[best_fixed_success]["success_mean"]
    success_lift = routed_success - best_fixed_success_value

    routed_utility = _subset_routed(
        task_metrics, pooled_ids, "route", primary_lambda
    )["utility"]
    best_fixed_utility = _best_fixed_policy(
        pooled_arms, capability_order, metric="utility"
    )
    best_fixed_utility_value = pooled_arms[best_fixed_utility]["utility"]
    primary_utility_lift = routed_utility - best_fixed_utility_value

    block_results: dict[str, dict[str, Any]] = {}
    for block in (0, 1):
        ids = subset_ids(block=block)
        routed = _subset_routed(task_metrics, ids, "route", primary_lambda)
        arms = _subset_fixed_arms(task_metrics, ids, primary_lambda)
        best = _best_fixed_policy(arms, capability_order, metric="utility")
        block_results[str(block)] = {
            "routed": routed,
            "fixed": arms,
            "best_fixed_policy": best,
            "lift": routed["utility"] - arms[best]["utility"],
        }

    stratum_results: dict[str, dict[str, Any]] = {}
    for stratum in STRATA:
        ids = subset_ids(stratum=stratum)
        routed = _subset_routed(task_metrics, ids, "route", primary_lambda)
        arms = _subset_fixed_arms(task_metrics, ids, primary_lambda)
        best = _best_fixed_policy(arms, capability_order, metric="utility")
        stratum_results[stratum] = {
            "routed": routed,
            "fixed": arms,
            "best_fixed_policy": best,
            "lift": routed["utility"] - arms[best]["utility"],
        }

    sensitivity_results: dict[str, dict[str, Any]] = {}
    for lam in manifest["analysis"]["thresholds"]["sensitivity_lambda_lift_nonnegative"]:
        routed = _subset_routed(task_metrics, pooled_ids, "route", lam)
        arms = _subset_fixed_arms(task_metrics, pooled_ids, lam)
        best = _best_fixed_policy(arms, capability_order, metric="utility")
        sensitivity_results[str(lam)] = {
            "lambda": lam,
            "routed": routed,
            "fixed": arms,
            "best_fixed_policy": best,
            "lift": routed["utility"] - arms[best]["utility"],
        }

    bootstrap_lower_bound = _bootstrap_utility_lower_bound(
        task_metrics, capability_order, BOOTSTRAP_SEED, draws, primary_lambda, cost_units
    )

    prompt_only_routed = _subset_routed(
        task_metrics, pooled_ids, "prompt_only_route", primary_lambda
    )
    prompt_only_arms = _subset_fixed_arms(task_metrics, pooled_ids, primary_lambda)
    prompt_only_best = _best_fixed_policy(
        prompt_only_arms, capability_order, metric="utility"
    )

    analysis: dict[str, Any] = {
        "schedules": {str(block): block_schedules[str(block)] for block in (0, 1)},
        "attempts": {
            "rows": len(rows),
            "per_block": {"0": ATTEMPTS_PER_SCHEDULE // 2, "1": ATTEMPTS_PER_SCHEDULE // 2},
        },
        "discordant_cells": {
            "overall": {
                "count": discordant_overall,
                "total": PANELS_PER_SCHEDULE,
            },
            "per_block": {
                str(block): {
                    "count": discordant_per_block[block],
                    "total": PANELS_PER_SCHEDULE // 2,
                }
                for block in (0, 1)
            },
        },
        "success": {
            "routed": routed_success,
            "fixed": {
                capability: pooled_arms[capability]["success_mean"]
                for capability in SPECIALIST_CAPABILITIES
            },
            "best_fixed_policy": best_fixed_success,
            "best_fixed": best_fixed_success_value,
            "lift": success_lift,
        },
        "utility": {
            "lambda": primary_lambda,
            "pooled": {
                "routed": routed_utility,
                "fixed": {
                    capability: pooled_arms[capability]["utility"]
                    for capability in SPECIALIST_CAPABILITIES
                },
                "best_fixed_policy": best_fixed_utility,
                "best_fixed": best_fixed_utility_value,
                "lift": primary_utility_lift,
            },
            "blocks": block_results,
            "strata": stratum_results,
            "sensitivity": sensitivity_results,
            "bootstrap": {
                "draws": draws,
                "seed": BOOTSTRAP_SEED,
                "lower_bound": bootstrap_lower_bound,
            },
        },
        "prompt_only": {
            "lambda": primary_lambda,
            "routed": prompt_only_routed,
            "fixed": prompt_only_arms,
            "best_fixed_policy": prompt_only_best,
            "lift": prompt_only_routed["utility"]
            - prompt_only_arms[prompt_only_best]["utility"],
        },
    }

    # ---- gates --------------------------------------------------------------
    thresholds = manifest["analysis"]["thresholds"]
    gate_checks: list[dict[str, Any]] = []
    gates_passed = True

    def _gate(
        check_id: str, label: str, passed: bool, detail: dict[str, Any]
    ) -> None:
        nonlocal gates_passed
        gates_passed = gates_passed and passed
        gate_checks.append(
            {"id": check_id, "kind": "gate", "label": label, "passed": passed, "detail": detail}
        )

    _gate(
        "discordant_cells_overall",
        "repeat-discordant policy-task cells (overall)",
        discordant_overall <= thresholds["max_repeat_discordant_cells"],
        {
            "count": discordant_overall,
            "total": PANELS_PER_SCHEDULE,
            "threshold": thresholds["max_repeat_discordant_cells"],
        },
    )
    for block in (0, 1):
        _gate(
            f"discordant_cells_block_{block}",
            f"repeat-discordant policy-task cells (block {block})",
            discordant_per_block[block]
            <= thresholds["max_repeat_discordant_cells_per_block"],
            {
                "count": discordant_per_block[block],
                "total": PANELS_PER_SCHEDULE // 2,
                "threshold": thresholds["max_repeat_discordant_cells_per_block"],
            },
        )
    _gate(
        "routed_success",
        "routed verified success",
        routed_success >= thresholds["routed_verified_success_min"],
        {"value": routed_success, "threshold": thresholds["routed_verified_success_min"]},
    )
    _gate(
        "routed_success_lift",
        "routed success lift over the global hindsight-better fixed specialist",
        success_lift >= thresholds["routed_success_lift_over_hindsight_best_fixed_min"],
        {
            "value": success_lift,
            "threshold": thresholds["routed_success_lift_over_hindsight_best_fixed_min"],
        },
    )
    _gate(
        "primary_pooled_utility_lift",
        "primary pooled utility lift over the hindsight-better fixed specialist",
        primary_utility_lift >= thresholds["primary_pooled_utility_lift_min"],
        {
            "value": primary_utility_lift,
            "threshold": thresholds["primary_pooled_utility_lift_min"],
        },
    )
    _gate(
        "bootstrap_lower_bound",
        "one-sided 90% task-cluster bootstrap lower bound above zero",
        bootstrap_lower_bound > 0.0,
        {"value": bootstrap_lower_bound, "threshold": 0.0},
    )
    for block in (0, 1):
        _gate(
            f"block_{block}_utility_lift",
            f"block {block} utility lift over the block-local hindsight-better fixed specialist",
            block_results[str(block)]["lift"] >= thresholds["block_utility_lift_min"],
            {
                "value": block_results[str(block)]["lift"],
                "threshold": thresholds["block_utility_lift_min"],
            },
        )
    for lam in thresholds["sensitivity_lambda_lift_nonnegative"]:
        _gate(
            f"sensitivity_lambda_{str(lam)}_lift",
            f"pooled utility lift at sensitivity lambda {lam}",
            sensitivity_results[str(lam)]["lift"] >= 0.0,
            {"value": sensitivity_results[str(lam)]["lift"], "threshold": 0.0},
        )
    for stratum in ("mixed", "ambiguous"):
        _gate(
            f"{stratum}_stratum_utility_lift",
            f"{stratum} stratum primary utility lift",
            stratum_results[stratum]["lift"] >= 0.0,
            {"value": stratum_results[stratum]["lift"], "threshold": 0.0},
        )

    checks: list[dict[str, Any]] = [
        {
            "id": "attempt_completeness",
            "kind": "structural",
            "label": "exactly 96 planned attempts with no duplicate, missing, or extra rows",
            "passed": True,
            "detail": {},
        },
        {
            "id": "attempt_identity",
            "kind": "structural",
            "label": "every identity field matches the planned attempt",
            "passed": True,
            "detail": {},
        },
        {
            "id": "attempt_runtime_contract",
            "kind": "structural",
            "label": "runtime row contract (status, sampling, verification, usage, provenance, mechanism, error coherence)",
            "passed": True,
            "detail": {},
        },
    ] + gate_checks

    return _finalize_report(
        manifest,
        authorization,
        receipt_hash,
        draws,
        valid=True,
        gates_passed=gates_passed,
        errors=[],
        analysis=analysis,
        checks=checks,
    )


def validate_gate_report(
    report: Mapping[str, Any],
    manifest: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> list[str]:
    """Validate a self-hashed Stage-B gate report.

    Checks schema identity, self-hash, manifest/authorization bindings, the
    authoritative bootstrap draw count, and decision/check coherence (the
    ``decision`` must match the ``valid``/``gates_passed`` flags and those
    flags must match the structural/gate check results).
    """
    errors: list[str] = []
    if not isinstance(report, Mapping):
        return ["gate report must be an object"]
    if report.get("schema_version") != GATE_REPORT_SCHEMA:
        errors.append(
            f"schema_version must be {GATE_REPORT_SCHEMA!r}, "
            f"got {report.get('schema_version')!r}"
        )
    errors.extend(_verify_self_hash(report, "report_hash"))
    if report.get("manifest_hash") != manifest.get("manifest_hash"):
        errors.append("gate report does not bind the supplied manifest")
    if report.get("authorization_hash") != authorization.get("authorization_hash"):
        errors.append("gate report does not bind the supplied authorization")
    if report.get("stage_b_id") != manifest.get("stage_b_id"):
        errors.append("gate report stage_b_id does not match the manifest")

    try:
        manifest_draws = _require_int(
            manifest["analysis"]["bootstrap"]["draws"], "bootstrap.draws"
        )
    except (KeyError, TypeError, ValueError):
        manifest_draws = BOOTSTRAP_DRAWS
    if not _json_equal(report.get("configured_draws"), manifest_draws):
        errors.append(
            f"configured_draws must equal the manifest bootstrap draws "
            f"{manifest_draws!r}, got {report.get('configured_draws')!r}"
        )
    if report.get("decision") not in (DECISION_PASS, DECISION_NO_GO, DECISION_INVALID):
        errors.append(
            "decision must be routing_smoke_pass, routing_smoke_no_go, or invalid"
        )

    checks = report.get("checks")
    if not isinstance(checks, Sequence) or isinstance(checks, (str, bytes)):
        errors.append("checks must be a list")
    else:
        for index, check in enumerate(checks):
            if not isinstance(check, Mapping):
                errors.append(f"checks[{index}] must be an object")
                continue
            if type(check.get("passed")) is not bool:
                errors.append(f"checks[{index}].passed must be a boolean")
        if type(report.get("valid")) is not bool:
            errors.append("valid must be a boolean")
        if type(report.get("gates_passed")) is not bool:
            errors.append("gates_passed must be a boolean")
        structural_checks = [
            check
            for check in checks
            if isinstance(check, Mapping) and check.get("kind") == "structural"
        ]
        gate_checks = [
            check
            for check in checks
            if isinstance(check, Mapping) and check.get("kind") == "gate"
        ]
        all_structural_passed = all(check.get("passed") is True for check in structural_checks)
        all_gates_passed = all(check.get("passed") is True for check in gate_checks)
        if not _json_equal(report.get("valid"), all_structural_passed):
            errors.append("valid flag does not match the structural check results")
        if report.get("valid") is True:
            if not _json_equal(report.get("gates_passed"), all_gates_passed):
                errors.append("gates_passed flag does not match the gate check results")
            expected = DECISION_PASS if report.get("gates_passed") else DECISION_NO_GO
        else:
            if report.get("gates_passed") is not False:
                errors.append("gates_passed must be False when the report is invalid")
            expected = DECISION_INVALID
        if report.get("decision") != expected:
            errors.append(
                f"decision {report.get('decision')!r} is incoherent with "
                f"valid/gates_passed (expected {expected!r})"
            )
    return errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyreplab-m3-utility-routing-smoke",
        description=(
            "Freeze, authorize, execute, safely export, analyze, and validate "
            "the M3 utility-routing Stage-B smoke. Freeze/authorize commands "
            "remain outcome-blind; run and analyze are explicit operations."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser(
        "freeze", help="freeze the self-hashed Stage-B manifest"
    )
    freeze.add_argument("spec", help="path to the Stage-B spec JSON")
    freeze.add_argument("registry", help="path to the policy registry JSON")
    freeze.add_argument("stage_a_manifest", help="path to the Stage-A manifest JSON")
    freeze.add_argument("output", help="path to write the frozen manifest JSON")

    authorize = subparsers.add_parser(
        "authorize", help="emit a self-hashed authorization for a Stage-B manifest"
    )
    authorize.add_argument("stage_a_manifest", help="path to the Stage-A manifest JSON")
    authorize.add_argument("stage_a_report", help="path to the Stage-A gate report JSON")
    authorize.add_argument("stage_b_manifest", help="path to the Stage-B manifest JSON")
    authorize.add_argument("output", help="path to write the authorization JSON")

    validate = subparsers.add_parser(
        "validate", help="validate a Stage-B manifest (and optionally its authorization)"
    )
    validate.add_argument("manifest", help="path to the Stage-B manifest JSON")
    validate.add_argument("--authorization", default=None, help="path to the authorization JSON")
    validate.add_argument(
        "--stage-a-manifest",
        default=None,
        help="path to the Stage-A manifest (required with --authorization)",
    )
    validate.add_argument(
        "--stage-a-report",
        default=None,
        help="path to the Stage-A gate report (required with --authorization)",
    )

    analyze = subparsers.add_parser(
        "analyze", help="analyze runtime safe-attempts and emit a self-hashed gate report"
    )
    analyze.add_argument("manifest", help="path to the Stage-B manifest JSON")
    analyze.add_argument("authorization", help="path to the Stage-B authorization JSON")
    analyze.add_argument("attempts", help="path to the runtime safe-attempts JSONL")
    analyze.add_argument("output", help="path to write the gate report JSON")
    analyze.add_argument(
        "--execution-receipt",
        default=None,
        help="optional path to the execution receipt JSON",
    )

    validate_gate = subparsers.add_parser(
        "validate-gate", help="validate a self-hashed Stage-B gate report"
    )
    validate_gate.add_argument("report", help="path to the gate report JSON")
    validate_gate.add_argument("manifest", help="path to the Stage-B manifest JSON")
    validate_gate.add_argument("authorization", help="path to the Stage-B authorization JSON")

    run = subparsers.add_parser(
        "run", help="run the authorized Stage-B panel sequentially and export safe rows"
    )
    run.add_argument("manifest", help="path to the Stage-B manifest JSON")
    run.add_argument("authorization", help="path to the Stage-B authorization JSON")
    run.add_argument("registry", help="path to the frozen policy registry JSON")
    run.add_argument("stage_a_manifest", help="path to the bound Stage-A manifest")
    run.add_argument("stage_a_report", help="path to the bound Stage-A report")
    run.add_argument("raw_output", help="append-only raw panel JSONL output")
    run.add_argument("safe_output", help="immutable safe-attempt JSONL output")
    run.add_argument("execution_receipt", help="execution receipt JSON output")
    run.add_argument("safe_export_receipt", help="safe export receipt JSON output")
    run.add_argument("preflight", help="runtime preflight JSON output")
    run.add_argument("--pi", default=os.environ.get("PYREPLAB_PI", "pi"))
    run.add_argument(
        "--provider", default=os.environ.get("PYREPLAB_PI_PROVIDER", "ubuntu-gemma")
    )
    run.add_argument(
        "--model", default=os.environ.get("PYREPLAB_PI_MODEL", "gemma-4-26b-a4b")
    )
    run.add_argument(
        "--thinking", default=os.environ.get("PYREPLAB_PI_THINKING", "off")
    )
    run.add_argument("--unbrowser-binary", required=True)
    run.add_argument("--model-artifact", required=True)
    run.add_argument(
        "--llama-server-binary", default="/usr/local/lib/ollama/llama-server"
    )

    export = subparsers.add_parser(
        "export-safe", help="rebuild the safe-attempt export from raw panel records"
    )
    export.add_argument("manifest")
    export.add_argument("authorization")
    export.add_argument("preflight")
    export.add_argument("raw_output")
    export.add_argument("execution_receipt")
    export.add_argument("safe_output")
    export.add_argument("safe_export_receipt")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "freeze":
            spec = _load_json(args.spec)
            registry = TreatmentRegistry.load(args.registry)
            stage_a_manifest = _load_json(args.stage_a_manifest)
            spec_stage_a = spec.get("stage_a")
            if not isinstance(spec_stage_a, Mapping):
                raise ValueError("spec.stage_a must be an object")
            if file_sha256(args.stage_a_manifest) != spec_stage_a.get("manifest_sha256"):
                raise ValueError(
                    "the supplied Stage-A manifest file does not match "
                    "spec.stage_a.manifest_sha256"
                )
            manifest = build_manifest(spec, registry, stage_a_manifest)
            manifest_errors = validate_manifest(manifest, registry=registry)
            if manifest_errors:
                raise ValueError("manifest failed self-validation: " + "; ".join(manifest_errors))
            immutable_write(Path(args.output), manifest)
            print(
                json.dumps(
                    {
                        "command": "freeze",
                        "manifest": str(_resolve(args.output)),
                        "manifest_hash": manifest["manifest_hash"],
                        "stage_b_id": manifest["stage_b_id"],
                        "tasks": TASKS,
                        "attempts_per_schedule": ATTEMPTS_PER_SCHEDULE,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        if args.command == "authorize":
            stage_a_manifest_path = _resolve(args.stage_a_manifest)
            stage_a_report_path = _resolve(args.stage_a_report)
            manifest = _load_json(args.stage_b_manifest)
            authorization = build_authorization(
                stage_a_manifest_path, stage_a_report_path, manifest
            )
            immutable_write(Path(args.output), authorization)
            print(
                json.dumps(
                    {
                        "command": "authorize",
                        "authorization": str(_resolve(args.output)),
                        "authorization_hash": authorization["authorization_hash"],
                        "manifest_hash": manifest["manifest_hash"],
                        "decision": authorization["decision"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        if args.command == "validate":
            manifest = _load_json(args.manifest)
            errors: list[str] = []
            if args.authorization:
                authorization = _load_json(args.authorization)
                stage_a_manifest_path = args.stage_a_manifest
                stage_a_report_path = args.stage_a_report
                auth_stage_a = authorization.get("stage_a")
                if stage_a_manifest_path is None and isinstance(auth_stage_a, Mapping):
                    stage_a_manifest_path = auth_stage_a.get("manifest_path")
                if stage_a_report_path is None and isinstance(auth_stage_a, Mapping):
                    stage_a_report_path = auth_stage_a.get("report_path")
                if stage_a_manifest_path is None or stage_a_report_path is None:
                    raise ValueError(
                        "--authorization requires --stage-a-manifest and --stage-a-report "
                        "(or an authorization that records them)"
                    )
                errors.extend(
                    validate_authorization(
                        authorization,
                        manifest,
                        stage_a_manifest_path,
                        stage_a_report_path,
                    )
                )
            errors.extend(validate_manifest(manifest))
            result = {
                "command": "validate",
                "valid": not errors,
                "manifest_hash": manifest.get("manifest_hash"),
                "errors": sorted(set(errors)),
            }
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if not errors else 1

        if args.command == "analyze":
            manifest = _load_json(args.manifest)
            authorization = _load_json(args.authorization)
            attempts = load_safe_attempts_jsonl(args.attempts)
            execution_receipt = None
            if args.execution_receipt:
                execution_receipt = _load_json(args.execution_receipt)
            report = analyze_stage_b(
                manifest, authorization, attempts, execution_receipt=execution_receipt
            )
            immutable_write(Path(args.output), report)
            decision = report["decision"]
            print(
                json.dumps(
                    {
                        "command": "analyze",
                        "report": str(_resolve(args.output)),
                        "report_hash": report["report_hash"],
                        "decision": decision,
                        "valid": report["valid"],
                        "gates_passed": report["gates_passed"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return {
                DECISION_PASS: 0,
                DECISION_NO_GO: 2,
                DECISION_INVALID: 1,
            }[decision]

        if args.command == "validate-gate":
            report = _load_json(args.report)
            manifest = _load_json(args.manifest)
            authorization = _load_json(args.authorization)
            errors = validate_gate_report(report, manifest, authorization)
            result = {
                "command": "validate-gate",
                "valid": not errors,
                "report_hash": report.get("report_hash"),
                "errors": sorted(set(errors)),
            }
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if not errors else 1

        if args.command == "run":
            result = run_stage_b(
                args.manifest,
                args.authorization,
                args.registry,
                args.stage_a_manifest,
                args.stage_a_report,
                args.raw_output,
                args.safe_output,
                args.execution_receipt,
                args.safe_export_receipt,
                args.preflight,
                pi_binary=args.pi,
                provider=args.provider,
                model=args.model,
                thinking=args.thinking,
                unbrowser_binary=args.unbrowser_binary,
                model_artifact=args.model_artifact,
                llama_server_binary=args.llama_server_binary,
            )
            print(json.dumps({"command": "run", **result}, indent=2, sort_keys=True))
            return 0

        if args.command == "export-safe":
            manifest = _load_json(args.manifest)
            authorization = _load_json(args.authorization)
            preflight = _load_json(args.preflight)
            execution_receipt = _load_json(args.execution_receipt)
            raw_records = _load_raw_panel_records(args.raw_output)
            rows, receipt = export_safe_attempts(
                manifest,
                authorization,
                preflight,
                raw_records,
                execution_receipt,
            )
            immutable_write_jsonl(args.safe_output, rows)
            immutable_write(args.safe_export_receipt, receipt)
            print(
                json.dumps(
                    {
                        "command": "export-safe",
                        "safe_output": str(_resolve(args.safe_output)),
                        "safe_attempts": len(rows),
                        "safe_export_hash": receipt["safe_export_hash"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        raise ValueError(f"unknown command {args.command!r}")
    except (OSError, ValueError, RuntimeError) as error:
        print(f"utility-routing smoke error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SPEC_SCHEMA",
    "MANIFEST_SCHEMA",
    "AUTHORIZATION_SCHEMA",
    "GATE_SCHEMA",
    "GATE_REPORT_SCHEMA",
    "EXECUTION_RECEIPT_SCHEMA",
    "RUNTIME_PREFLIGHT_SCHEMA",
    "RAW_PANEL_SCHEMA",
    "SAFE_ATTEMPT_SCHEMA",
    "SAFE_EXPORT_SCHEMA",
    "DEFAULT_STAGE_B_SEED",
    "TASKS",
    "BLOCKS",
    "REPLICAS",
    "PANELS_PER_SCHEDULE",
    "ATTEMPTS_PER_SCHEDULE",
    "LAMBDA_GRID",
    "PRIMARY_LAMBDA",
    "COST_UNITS_PER_TOKEN",
    "MAX_SAMPLING_SEED",
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "CANARY_ROW_STATUS",
    "CANARY_EXCLUSION",
    "THRESHOLDS",
    "ERROR_TAXONOMY",
    "RUN_POLICY",
    "GATE_CHECKS",
    "SAFE_ATTEMPT_REQUIRED_FIELDS",
    "ATTEMPT_IDENTITY_FIELDS",
    "DECISION_PASS",
    "DECISION_NO_GO",
    "DECISION_INVALID",
    "canonical_json",
    "canonical_hash",
    "file_sha256",
    "immutable_write",
    "default_spec",
    "build_manifest",
    "validate_manifest",
    "build_authorization",
    "validate_authorization",
    "build_runtime_preflight",
    "validate_runtime_preflight",
    "build_execution_receipt",
    "validate_execution_receipt",
    "export_safe_attempts",
    "immutable_write_jsonl",
    "run_stage_b",
    "load_safe_attempts_jsonl",
    "craft_attempt_row",
    "craft_runtime_rows",
    "analyze_stage_b",
    "validate_gate_report",
    "build_parser",
    "main",
]
