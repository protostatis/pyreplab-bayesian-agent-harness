"""Safe corpus exporter for a Pyreplab run root.

The exporter reads the on-disk task and attempt manifests produced by the gym
(``<root>/tasks/<task_id>/task.json``, ``<root>/attempts/<attempt_id>/``) and
joins each *verified* attempt to its ``TaskSpec`` into a flat, deterministic,
leakage-safe row.

Row contents
------------
Each row carries the task identity fields (``task_id``, ``family``,
``template_id``, ``generator_version``, ``seed``, ``difficulty``, ``prompt``,
``contract``, raw ``public_metadata``), the attempt identity
(``attempt_id``, ``policy_id``, ``policy_version``), a whole-task group split,
the verified outcome (``verified_success``, ``failure_code``, verifier ids),
a coarse ``termination_class``, ``output_token_cost`` from ``usage.output``,
post-action cost counters from the normalized Pi events (``usage``,
``assistant_message_count``, ``provider_turn_count``, ``tool_call_count``,
``tool_limit_rejection_count``, ``length_stop_count``, ``final_text_length``)
and an explicit nested ``model_input`` that contains *only* predecision
information.

For Unbrowser grammar treatments (``native_bash_unbrowser_interactive_v1``)
the ``model_input`` is identity-free and follows an M3/CNP nested schema:
``model_input.task`` (32-d deterministic embedding, template, difficulty,
family, public_metadata) and ``model_input.treatment`` (grammar factor
labels, 13-d vector, ``enforced_tool_call_cap``, ``tool_interface``,
``allowed_tools_signature``).  The top-level row keeps identity fields for
join/audit use.

Leakage boundary
----------------
The following never appear in a row: ``verifier_ref``, oracle/private
metadata, verifier diagnostics, workspace or reference paths, raw trajectory
text, or final answer text. Post-action fields (usage, message/tool counts,
outcome) are exposed at the top level for supervised learning and cost
analysis but are *excluded from* ``model_input``, which is the only accepted
predecision model input. ``model_input`` holds the ``text`` (prompt +
contract), ``family``/``template_id``/``difficulty``, a recursively flattened
dictionary of finite numeric/bool ``public_metadata`` leaves (arrays and
strings are never silently converted into labels) and the policy identity.

Splits
------
The split is computed per whole task from a stable SHA-256 of
``<template_id>|<seed>`` bucketed into approximately train 70% / validation
15% / test 15%. It never depends on the attempt, so both policies of a pair
always land in the same split. The stored ``TaskSpec.split`` field is ignored.

``T_canary`` / ``T_pilot`` tasks are mapped to the reserved excluded splits
``canary_excluded`` / ``pilot_excluded``.  Those rows additionally carry a
``governance_role`` equal to the excluded split and an ``eligibility`` object
whose ``training`` / ``calibration`` / ``development`` / ``final`` booleans are
all ``False``, so an excluded row can never silently enter a current
meta-training, calibration, development, or final-evaluation pool.

Robustness
----------
Attempts that are not yet verified are skipped by default; the summary
returned by :func:`write_dataset` reports the skip counts by reason. Malformed
records (invalid JSON, missing or mistyped fields, an attempt claiming a
verified status without a verification file, or an attempt whose task manifest
is missing) raise a path-specific :class:`ValueError`/``OSError`` instead of
silently contaminating the dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator

from .meta_grammar import grammar_factor_vector as _meta_grammar_factor_vector
from .treatments import (
    TreatmentRegistry,
    TreatmentSpec,
    treatment_model_input_descriptor,
)

#: Cumulative bucket bounds for the ~70/15/15 whole-task split.
_SPLIT_BOUNDS: tuple[tuple[str, int], ...] = (("train", 70), ("validation", 85))

#: Split labels that mark a whole task as permanently excluded from every
#: current meta-training / calibration / development / final pool.  Rows with
#: one of these splits carry a ``governance_role`` equal to the split and an
#: ``eligibility`` object whose four booleans are all ``False``.
_EXCLUDED_SPLITS: frozenset[str] = frozenset({"canary_excluded", "pilot_excluded"})

#: The four current governance pools; an excluded row must be ineligible for
#: every one of them.
_ELIGIBILITY_POOLS: tuple[str, ...] = (
    "training",
    "calibration",
    "development",
    "final",
)

_TASK_REQUIRED: dict[str, type] = {
    "id": str,
    "family": str,
    "template_id": str,
    "generator_version": str,
    "seed": int,
    "difficulty": str,
    "prompt": str,
}

_ATTEMPT_REQUIRED: dict[str, type] = {
    "attempt_id": str,
    "task_id": str,
    "policy_id": str,
    "policy_version": str,
    "status": str,
}

_VERIFICATION_REQUIRED: dict[str, type] = {
    "success": bool,
    "verifier_id": str,
    "verifier_version": str,
}


def task_split(template_id: str, seed: int) -> str:
    """Deterministic whole-task group split from ``template_id`` and ``seed``.

    SHA-256 of ``<template_id>|<seed>`` bucketed into approximately
    ``train`` 70%, ``validation`` 15%, ``test`` 15%. The key contains no
    attempt identity, so every attempt of a task shares its split.
    """
    key = f"{template_id}|{seed}".encode("utf-8")
    bucket = int(hashlib.sha256(key).hexdigest()[:8], 16) % 100
    for name, bound in _SPLIT_BOUNDS:
        if bucket < bound:
            return name
    return "test"


def flatten_public_metadata(
    metadata: dict[str, Any], prefix: str = ""
) -> dict[str, int | float | bool]:
    """Recursively flatten finite numeric/bool leaves of ``public_metadata``.

    Nested dicts are flattened under dotted keys. Arrays, strings, ``None``
    and non-finite numbers are excluded rather than converted into labels;
    callers that want counts must add them under clearly named derived keys.
    """
    flattened: dict[str, int | float | bool] = {}
    for key in sorted(metadata):
        value = metadata[key]
        full = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, bool):
            flattened[full] = value
        elif isinstance(value, int):
            flattened[full] = value
        elif isinstance(value, float) and math.isfinite(value):
            flattened[full] = value
        elif isinstance(value, dict):
            flattened.update(flatten_public_metadata(value, full))
    return flattened


_UNBROWSER_GRAMMAR_INTERFACE = "native_bash_unbrowser_interactive_v1"

# All three interactive Unbrowser interfaces share the same identity-free
# grammar model_input schema.
_UNBROWSER_GRAMMAR_INTERFACES = frozenset({
    _UNBROWSER_GRAMMAR_INTERFACE,
    "native_bash_unbrowser_interactive_text_first_v1",
    "native_bash_unbrowser_interactive_structure_first_v1",
})

# DDL-1 (semantic_table) and DDL-2 (semantic_form) tool interfaces carry
# distinct tool schemas and do NOT share the 72-cell grammar factor
# structure.  Their rows therefore use the legacy generic model_input path
# (non-identity-free) with the treatment descriptor augmented by any
# available generator_metadata labels.  This is intentional; the M3/CNP
# identity-free schema requires a well-defined factor vector that these
# purpose-specific interfaces do not yet define.
_SEMANTIC_SPECIALIST_INTERFACES: frozenset[str] = frozenset({
    "native_bash_unbrowser_semantic_table_v1",
    "native_bash_unbrowser_semantic_form_v1",
})

_UNBROWSER_GRAMMAR_FACTOR_KEYS = (
    "planning",
    "observation",
    "verification",
    "recovery",
    "tool_cap",
)


def _compute_task_embedding(text: str) -> dict[str, Any]:
    """Deterministic 32-d task-text embedding via ASCII-token SHA-256 projection.

    Tokenises the text into ASCII space-delimited tokens, XOR-accumulates each
    token's SHA-256 digest bytewise, converts the 32 accumulated bytes into
    signed floats in [-1, 1], and L2-normalises the resulting vector.

    The result is process-stable (no *hash()*, no iteration order).  It is
    plumbing-only — not semantic-effectiveness evidence.
    """
    raw_text = text.encode("ascii", errors="replace").decode("ascii")
    tokens = raw_text.split()
    if not tokens:
        return {
            "encoder": "sha256_ascii_projection_v1",
            "version": 1,
            "vector": [0.0] * 32,
        }
    # XOR-accumulate each token's 32-byte SHA-256 digest.
    acc = bytearray(32)
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for i in range(32):
            acc[i] ^= digest[i]
    # Convert bytes -> floats in [-1, 1], then L2-normalise.
    raw = [((b / 127.5) - 1.0) for b in acc]
    norm = math.sqrt(sum(v * v for v in raw))
    if norm > 0:
        vector = [v / norm for v in raw]
    else:
        vector = [0.0] * 32
    return {
        "encoder": "sha256_ascii_projection_v1",
        "version": 1,
        "vector": vector,
    }


_TERMINATION_KNOWN = frozenset(
    {"tool_call_limit", "wall_timeout", "invalid_or_tool_error", "model_runtime_failure"}
)


def _derive_termination_class(
    verified_success: bool,
    failure_code: str | None,
    tool_limit_rejection_count: int,
    length_stop_count: int,
) -> str:
    """Coarse termination class from verifier outcome and Pi-event signals."""
    if failure_code and failure_code in _TERMINATION_KNOWN:
        return failure_code
    if tool_limit_rejection_count > 0:
        return "tool_call_limit"
    if length_stop_count > 0:
        return "wall_timeout"
    if verified_success:
        return "normal_completion"
    if failure_code is not None:
        return "verifier_declared_unsuccessful"
    return "normal_completion"


def _grammar_factors_from_treatment(
    treatment: TreatmentSpec,
) -> dict[str, str] | None:
    """Extract behavioural grammar factor levels from a treatment's
    ``generator_metadata``, or ``None`` when the treatment is not an
    Unbrowser grammar treatment.

    The returned dict contains only the five behavioural factor labels;
    identity/hash fields (``grammar_version``, ``grammar_size``,
    ``grammar_name``, ``index``, ``policy_id``, ``bundle_id``, ``bundle_hash``)
    are excluded.
    """
    meta = dict(treatment.generator_metadata)
    if not meta or "planning" not in meta:
        return None
    return {
        key: str(meta[key])
        for key in _UNBROWSER_GRAMMAR_FACTOR_KEYS
        if key in meta
    }


def build_model_input(
    task: dict[str, Any],
    policy_id: str,
    policy_version: str,
    treatment: TreatmentSpec | None = None,
) -> dict[str, Any]:
    """Predecision-only model input for one attempt.

    Only information available before policy assignment may enter here: the
    prompt/contract text, family/template/difficulty, the numeric/bool public
    metadata and the chosen policy identity. No post-action fields.

    For Unbrowser grammar treatments (``native_bash_unbrowser_interactive_v1``)
    the output follows the M3/CNP leakage-safe nested schema:

    * ``model_input.task`` — task embedding, template, difficulty, family,
      public_metadata.
    * ``model_input.treatment`` — grammar factor labels, 13-d factor vector,
      ``enforced_tool_call_cap``, ``tool_interface``,
      ``allowed_tools_signature``.

    Neither ``policy_id``/``version``/``bundle_id``/``bundle_hash`` nor system
    prompt text appear inside ``model_input`` for grammar rows.
    """
    parts = [task["prompt"]]
    contract = task["contract"]
    if contract:
        parts.append("\n".join(contract))
    task_text = "\n\n".join(parts)
    public_meta = flatten_public_metadata(task["public_metadata"])

    is_grammar = (
        treatment is not None
        and treatment.tool_interface in _UNBROWSER_GRAMMAR_INTERFACES
    )

    if is_grammar:
        if treatment.id != policy_id or treatment.version != policy_version:  # type: ignore[union-attr]
            raise ValueError(
                "attempt policy identity does not match treatment registry entry: "
                f"{policy_id}@{policy_version} != {treatment.id}@{treatment.version}"  # type: ignore[union-attr]
            )
        factors = _grammar_factors_from_treatment(treatment)  # type: ignore[arg-type]
        factor_vec = _meta_grammar_factor_vector(treatment)  # 13-d  # type: ignore[arg-type]
        model_input: dict[str, Any] = {
            "task": {
                "task_embedding": _compute_task_embedding(task_text),
                "template": task["template_id"],
                "difficulty": task["difficulty"],
                "family": task["family"],
                "public_metadata": public_meta,
            },
            "treatment": {
                "grammar_factors": factors if factors is not None else {},
                "grammar_factor_vector": factor_vec,
                "enforced_tool_call_cap": treatment.tool_call_limit,  # type: ignore[union-attr]
                "tool_interface": treatment.tool_interface,  # type: ignore[union-attr]
                "allowed_tools_signature": treatment.allowed_tools_signature,  # type: ignore[union-attr]
            },
        }
        return model_input

    # ---------- legacy generic path (non-grammar treatments) -----------------
    model_input = {
        "text": task_text,
        "family": task["family"],
        "template_id": task["template_id"],
        "difficulty": task["difficulty"],
        "public_metadata": public_meta,
        "policy_id": policy_id,
        "policy_version": policy_version,
    }
    if treatment is not None:
        if treatment.id != policy_id or treatment.version != policy_version:
            raise ValueError(
                "attempt policy identity does not match treatment registry entry: "
                f"{policy_id}@{policy_version} != {treatment.id}@{treatment.version}"
            )
        treatment_desc = treatment_model_input_descriptor(treatment)
        # ------------------------------------------------------------------
        # ADDITIVE: attach grammar factor labels for Unbrowser policy cells.
        # Non-grammar treatments (interface != _UNBROWSER_GRAMMAR_INTERFACE)
        # are unaffected — their model_input.treatment dict remains unchanged.
        # ------------------------------------------------------------------
        if treatment.tool_interface in _UNBROWSER_GRAMMAR_INTERFACES:
            factors = _grammar_factors_from_treatment(treatment)
            if factors is not None:
                treatment_desc["grammar_factors"] = factors
        # ------------------------------------------------------------------
        # ADDITIVE: attach the structured specialist capability identity for
        # the two semantic interfaces (DDL-1 semantic_table / DDL-2
        # semantic_form).  These rows follow the generic model_input path, so
        # the capability family, parent bundle and substrate are surfaced as a
        # structured ``semantic`` descriptor from generator_metadata.
        # ------------------------------------------------------------------
        if treatment.tool_interface in _SEMANTIC_SPECIALIST_INTERFACES:
            meta = treatment.generator_metadata
            treatment_desc["semantic"] = {
                "capability": str(meta.get("capability", "")),
                "parent_bundle_id": str(meta.get("parent_bundle_id", "")),
                "substrate": str(meta.get("substrate", "")),
            }
        model_input["treatment"] = treatment_desc
    return model_input


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError as error:
            raise ValueError(f"malformed JSON at {path}: {error}") from error


def _require(value: Any, expected_type: type, label: str, path: Path) -> None:
    if not isinstance(value, expected_type):
        raise ValueError(
            f"malformed {label}: expected {expected_type.__name__}, "
            f"got {type(value).__name__} at {path}"
        )


def _parse_task(path: Path) -> dict[str, Any]:
    raw = _read_json(path)
    _require(raw, dict, "task manifest", path)
    for key, expected_type in _TASK_REQUIRED.items():
        if key not in raw:
            raise ValueError(f"malformed task manifest at {path}: missing key {key!r}")
        _require(raw[key], expected_type, f"task manifest field {key!r}", path)
    contract = raw["contract"]
    _require(contract, list, "task manifest field 'contract'", path)
    for item in contract:
        _require(item, str, "task manifest contract item", path)
    public_metadata = raw["public_metadata"]
    _require(public_metadata, dict, "task manifest field 'public_metadata'", path)
    return {
        "task_id": str(raw["id"]),
        "family": str(raw["family"]),
        "template_id": str(raw["template_id"]),
        "generator_version": str(raw["generator_version"]),
        "seed": int(raw["seed"]),
        "difficulty": str(raw["difficulty"]),
        "prompt": str(raw["prompt"]),
        "contract": list(contract),
        "public_metadata": dict(public_metadata),
    }


def _parse_attempt(path: Path) -> dict[str, Any]:
    raw = _read_json(path)
    _require(raw, dict, "attempt manifest", path)
    for key, expected_type in _ATTEMPT_REQUIRED.items():
        if key not in raw:
            raise ValueError(f"malformed attempt manifest at {path}: missing key {key!r}")
        _require(raw[key], expected_type, f"attempt manifest field {key!r}", path)
    return dict(raw)


def _parse_verification(path: Path) -> dict[str, Any]:
    raw = _read_json(path)
    _require(raw, dict, "verification result", path)
    for key, expected_type in _VERIFICATION_REQUIRED.items():
        if key not in raw:
            raise ValueError(f"malformed verification result at {path}: missing key {key!r}")
        _require(raw[key], expected_type, f"verification field {key!r}", path)
    failure_code = raw.get("failure_code")
    if failure_code is not None:
        _require(failure_code, str, "verification field 'failure_code'", path)
    return {
        "verified_success": bool(raw["success"]),
        "failure_code": failure_code,
        "verifier_id": str(raw["verifier_id"]),
        "verifier_version": str(raw["verifier_version"]),
    }


def _parse_normalized_events(path: Path) -> dict[str, Any]:
    raw = _read_json(path)
    _require(raw, dict, "normalized events", path)
    usage = raw.get("usage")
    _require(usage, dict, "normalized events field 'usage'", path)
    parsed_usage: dict[str, int] = {}
    for key, value in usage.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"malformed normalized events at {path}: usage.{key} is not an int"
            )
        parsed_usage[str(key)] = int(value)
    def _nonnegative_int(field: str, default: int = 0) -> int:
        value = raw.get(field, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"malformed normalized events at {path}: {field!r} is not a non-negative int"
            )
        return int(value)

    assistant_message_count = _nonnegative_int("assistant_message_count")
    provider_turn_count = _nonnegative_int(
        "provider_turn_count", assistant_message_count
    )
    tool_executions = raw.get("tool_executions", [])
    _require(tool_executions, list, "normalized events field 'tool_executions'", path)
    final_text = raw.get("final_text", "")
    _require(final_text, str, "normalized events field 'final_text'", path)
    return {
        "usage": parsed_usage,
        "assistant_message_count": assistant_message_count,
        "provider_turn_count": provider_turn_count,
        "tool_call_count": len(tool_executions),
        "tool_limit_rejection_count": _nonnegative_int(
            "tool_limit_rejection_count"
        ),
        "length_stop_count": _nonnegative_int("length_stop_count"),
        "final_text_length": len(final_text),
    }


def _build_row(
    task: dict[str, Any],
    attempt: dict[str, Any],
    verification: dict[str, Any],
    normalized: dict[str, Any],
    split: str,
    treatment: TreatmentSpec | None = None,
    treatment_registry_hash: str | None = None,
) -> dict[str, Any]:
    policy_id = str(attempt["policy_id"])
    policy_version = str(attempt["policy_version"])
    # --- output_token_cost (must be finite non-negative) ----------------------
    output_cost = normalized["usage"].get("output")
    if output_cost is None or not isinstance(output_cost, int):
        raise ValueError(
            f"missing or non-integer usage.output for attempt {attempt['attempt_id']}"
        )
    if output_cost < 0 or not math.isfinite(output_cost):
        raise ValueError(
            f"non-finite or negative usage.output {output_cost!r} "
            f"for attempt {attempt['attempt_id']}"
        )
    # --- termination_class (coarse) -------------------------------------------
    termination_class = _derive_termination_class(
        verification["verified_success"],
        verification["failure_code"],
        normalized["tool_limit_rejection_count"],
        normalized["length_stop_count"],
    )
    row = {
        "task_id": task["task_id"],
        "family": task["family"],
        "template_id": task["template_id"],
        "generator_version": task["generator_version"],
        "seed": task["seed"],
        "difficulty": task["difficulty"],
        "prompt": task["prompt"],
        "contract": list(task["contract"]),
        "public_metadata": dict(task["public_metadata"]),
        "attempt_id": str(attempt["attempt_id"]),
        "policy_id": policy_id,
        "policy_version": policy_version,
        "split": split,
        "verified_success": verification["verified_success"],
        "failure_code": verification["failure_code"],
        "verifier_id": verification["verifier_id"],
        "verifier_version": verification["verifier_version"],
        "output_token_cost": output_cost,
        "termination_class": termination_class,
        "usage": dict(normalized["usage"]),
        "assistant_message_count": normalized["assistant_message_count"],
        "provider_turn_count": normalized["provider_turn_count"],
        "tool_call_count": normalized["tool_call_count"],
        "tool_limit_rejection_count": normalized["tool_limit_rejection_count"],
        "length_stop_count": normalized["length_stop_count"],
        "final_text_length": normalized["final_text_length"],
        "model_input": build_model_input(
            task, policy_id, policy_version, treatment=treatment
        ),
    }
    task_role = task["public_metadata"].get("task_role")
    if isinstance(task_role, str) and task_role:
        row["task_role"] = task_role
    if split in _EXCLUDED_SPLITS:
        row["governance_role"] = split
        row["eligibility"] = {pool: False for pool in _ELIGIBILITY_POOLS}
    for field in (
        "rollout_replica",
        "sampling_seed",
        "pilot_manifest_hash",
        "pilot_panel_id",
    ):
        value = attempt.get(field)
        if value is not None:
            row[field] = value
    if treatment is not None:
        if not treatment_registry_hash:
            raise ValueError(
                "treatment_registry_hash is required for a registry-enriched row"
            )
        row["treatment_bundle_id"] = treatment.bundle_id
        row["treatment_bundle_hash"] = treatment.bundle_hash
        row["treatment_registry_hash"] = treatment_registry_hash
    return row


def _coerce_registry(
    treatment_registry: TreatmentRegistry | str | Path | None,
) -> TreatmentRegistry | None:
    if treatment_registry is None or isinstance(treatment_registry, TreatmentRegistry):
        return treatment_registry
    return TreatmentRegistry.load(treatment_registry)


def _scan(
    root: str | Path,
    treatment_registry: TreatmentRegistry | str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return ``(rows, summary)`` for a run root.

    Rows are sorted by ``(task_id, policy_id, attempt_id)``. Attempts that are
    not yet verified, or verified attempts without a normalized event file,
    are skipped and reported in ``summary["skipped"]``.
    """
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"run root does not exist: {root_path}")
    attempts_root = root_path / "attempts"
    tasks_root = root_path / "tasks"

    task_cache: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    attempts_found = 0
    tasks_seen: set[str] = set()
    registry = _coerce_registry(treatment_registry)

    if attempts_root.is_dir():
        for manifest_path in sorted(attempts_root.glob("*/attempt.json")):
            attempts_found += 1
            attempt = _parse_attempt(manifest_path)
            task_id = attempt["task_id"]
            if task_id not in task_cache:
                task_cache[task_id] = _parse_task(tasks_root / task_id / "task.json")
            task = task_cache[task_id]
            tasks_seen.add(task_id)

            attempt_dir = manifest_path.parent
            verification_path = attempt_dir / "verification.json"
            if not verification_path.exists():
                if attempt.get("status") == "verified":
                    raise ValueError(
                        f"malformed attempt at {manifest_path}: status is 'verified' "
                        f"but {verification_path} is missing"
                    )
                skipped["unverified"] = skipped.get("unverified", 0) + 1
                continue
            verification = _parse_verification(verification_path)

            events_path = attempt_dir / "pi-events.normalized.json"
            if not events_path.exists():
                skipped["missing_events"] = skipped.get("missing_events", 0) + 1
                continue
            normalized = _parse_normalized_events(events_path)

            task_role = task["public_metadata"].get("task_role")
            excluded_roles = {
                "T_pilot": "pilot_excluded",
                "T_canary": "canary_excluded",
            }
            split = excluded_roles.get(
                task_role,
                task_split(task["template_id"], task["seed"]),
            )
            treatment: TreatmentSpec | None = None
            if registry is not None:
                try:
                    treatment = registry.by_id_version(
                        str(attempt["policy_id"]), str(attempt["policy_version"])
                    )
                except KeyError as error:
                    raise ValueError(
                        "attempt treatment is missing from the supplied registry: "
                        f"{attempt['policy_id']}@{attempt['policy_version']} at "
                        f"{manifest_path}"
                    ) from error
                observed_bundle_hash = attempt.get("treatment_bundle_hash")
                if (
                    observed_bundle_hash is not None
                    and str(observed_bundle_hash) != treatment.bundle_hash
                ):
                    raise ValueError(
                        "attempt treatment bundle hash does not match registry: "
                        f"{observed_bundle_hash!r} != {treatment.bundle_hash!r} at "
                        f"{manifest_path}"
                    )
                observed_registry_hash = attempt.get("treatment_registry_hash")
                if (
                    observed_registry_hash is not None
                    and str(observed_registry_hash) != registry.registry_hash
                ):
                    raise ValueError(
                        "attempt treatment registry hash does not match supplied registry: "
                        f"{observed_registry_hash!r} != {registry.registry_hash!r} at "
                        f"{manifest_path}"
                    )
            rows.append(
                _build_row(
                    task,
                    attempt,
                    verification,
                    normalized,
                    split,
                    treatment=treatment,
                    treatment_registry_hash=(
                        registry.registry_hash if registry is not None else None
                    ),
                )
            )

    rows.sort(key=lambda row: (row["task_id"], row["policy_id"], row["attempt_id"]))

    split_counts: dict[str, int] = {}
    for row in rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
    summary = {
        "tasks": len(tasks_seen),
        "attempts_found": attempts_found,
        "rows": len(rows),
        "skipped": dict(sorted(skipped.items())),
        "split": dict(sorted(split_counts.items())),
        "treatment_registry_hash": registry.registry_hash if registry else None,
    }
    return rows, summary


def iter_dataset_rows(
    root: str | Path,
    treatment_registry: TreatmentRegistry | str | Path | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield sorted, leakage-safe dataset rows for every verified attempt."""
    rows, _ = _scan(root, treatment_registry)
    yield from rows


def build_dataset(
    root: str | Path,
    treatment_registry: TreatmentRegistry | str | Path | None = None,
) -> list[dict[str, Any]]:
    """Build the full sorted dataset row list for ``root``."""
    rows, _ = _scan(root, treatment_registry)
    return rows


def write_dataset(
    root: str | Path,
    output_path: str | Path,
    treatment_registry: TreatmentRegistry | str | Path | None = None,
) -> dict[str, Any]:
    """Write the dataset as deterministic JSONL and return a summary dict.

    Each line is one row serialized with ``sort_keys=True`` so the file is
    byte-for-byte reproducible. The write is atomic (temp file + rename).
    """
    rows, summary = _scan(root, treatment_registry)
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{target.name}.",
        delete=False,
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, target)
    summary["output_path"] = str(target)
    summary["rows_written"] = len(rows)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyreplab-harness-dataset",
        description="Export a verified-attempt corpus from a run root as deterministic JSONL.",
    )
    parser.add_argument("root", help="run root directory containing tasks/ and attempts/")
    parser.add_argument(
        "output",
        nargs="?",
        help="output JSONL path (omitting it prints the summary only)",
    )
    parser.add_argument(
        "--treatment-registry",
        default=None,
        help="optional immutable treatment registry used to enrich model_input",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output:
        summary = write_dataset(
            args.root, args.output, treatment_registry=args.treatment_registry
        )
    else:
        _rows, summary = _scan(args.root, args.treatment_registry)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "_compute_task_embedding",
    "_derive_termination_class",
    "_grammar_factors_from_treatment",
    "_UNBROWSER_GRAMMAR_INTERFACE",
    "build_dataset",
    "build_model_input",
    "build_parser",
    "flatten_public_metadata",
    "iter_dataset_rows",
    "main",
    "task_split",
    "write_dataset",
]
