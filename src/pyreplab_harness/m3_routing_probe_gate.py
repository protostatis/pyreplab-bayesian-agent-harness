"""Stage A zero-outcome probe manifest, analyzer, and gate for M3 utility routing.

This module implements Section 5 of ``notes/m3-utility-routing-smoke-plan.md``:
freeze an immutable, self-hashed Stage-A manifest from a deterministic spec,
validate it, then run three cold structural-probe replays over 32 versioned
routing fixtures (8 per stratum) and gate the result against frozen thresholds
*without executing any agent model*.

The frozen router is a transparent heuristic over (a) declared-operation flags
carried in the spec and (b) bounded structural-probe counts.  The sealed
first-bottleneck label is consulted only after every router decision has been
finalized; labels never enter the decision function.

Gates (frozen in the plan):

* 96/96 probe receipts valid;
* model-visible features byte-identical across three replays per fixture;
* every probe count within its frozen bound;
* privacy/provenance audit has zero violations;
* changing only visible text preserves features while changing the source hash;
* the probe precedes treatment assignment by construction;
* the frozen combined router agrees with the sealed first-bottleneck label on
  at least 28/32 fixtures and at least 6/8 in each stratum;
* prompt-only, probe-only, and combined heuristic results are all reported; and
* every synthetic utility and tie-break check passes.

Exit codes (CLI ``run``): ``0`` pass, ``2`` valid no-go, ``1`` invalid.

Forthcoming backend APIs are imported lazily and are optional for the
``run_stage_a`` entry point (they may be injected as adapters):

* ``routing_fixtures.build_stage_a_design()`` -> list of 32 private design
  coordinates; ``routing_fixtures.generate_routing_fixture(coord)`` -> public
  initial HTML plus private design fields.
* ``structural_probe.structural_probe(html)`` -> ``{"features": {...16
  counts...}, "receipt": {...}}``.
* ``routing_utility`` exposes a synthetic smoke validator for the utility
  function and tie-break (optional cross-check).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import string
import sys
import tempfile
from collections.abc import Mapping, Sequence
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable

from .routing_utility import (
    FROZEN_LAMBDA_GRID,
    PRIMARY_LAMBDA as ROUTING_PRIMARY_LAMBDA,
    run_utility_scoring_smoke_matrix,
)
from .structural_probe import (
    FEATURE_CAPS,
    FEATURE_KEYS,
    MECHANISM as PROBE_MECHANISM,
    SCHEMA_VERSION as PROBE_RECEIPT_SCHEMA,
    audit_receipt,
)

# ---------------------------------------------------------------------------
# frozen constants
# ---------------------------------------------------------------------------

MANIFEST_SCHEMA = "m3-routing-probe-stage-a-v1"
GATE_SCHEMA = "m3-routing-probe-stage-a-gate-v1"
STRATA: tuple[str, ...] = ("pure_table", "pure_form", "mixed", "ambiguous")
SPECIALISTS: tuple[str, ...] = ("table_specialist", "form_specialist")

PLAN_FIXTURES = 32
PLAN_PER_STRATUM = 8
PLAN_REPLAYS = 3
PLAN_PROBES = PLAN_FIXTURES * PLAN_REPLAYS  # 96
PLAN_AGREEMENT_MIN = 28
PLAN_AGREEMENT_MIN_PER_STRATUM = 6
PLAN_PRIVACY_MAX = 0

LAMBDA_GRID: tuple[float, ...] = FROZEN_LAMBDA_GRID
PRIMARY_LAMBDA = ROUTING_PRIMARY_LAMBDA
COST_UNITS_PER_TOKEN = 10000

# Frozen heuristic weights (prompt = declared operations, probe = structure).
ROUTER_PROMPT_WEIGHT = 1.0
ROUTER_PROBE_WEIGHT = 0.5
ROUTER_FIRST_OPERATION_BONUS = 0.5

# Probe schema: feature name -> frozen upper bound.  All counts are capped by
# this schema before entering model input.
PROBE_SCHEMA: dict[str, dict[str, Any]] = {
    name: {"type": "int", "cap": FEATURE_CAPS[name]} for name in FEATURE_KEYS
}
PROBE_FEATURE_NAMES: tuple[str, ...] = tuple(PROBE_SCHEMA)

# Structural-mass grouping used by the frozen router.
TABLE_MASS_FIELDS = (
    "table_count",
    "table_row_count",
    "table_cell_count",
    "max_table_columns",
)
FORM_MASS_FIELDS = (
    "form_count",
    "control_count",
    "required_control_count",
    "get_form_count",
    "post_form_count",
    "text_input_count",
    "select_count",
    "textarea_count",
    "button_count",
)

# Privacy: model-visible objects must not contain these keys (exact match).
_FORBIDDEN_KEYS = frozenset(
    {
        "html",
        "text",
        "page_text",
        "label",
        "route_label",
        "name",
        "value",
        "header",
        "class",
        "url",
        "domain",
        "host",
        "form_action",
        "action",
        "link",
        "selector",
        "element",
        "ref",
        "source_hash",
        "source_sha256",
        "seed",
        "task_seed",
        "template",
        "template_id",
        "difficulty",
        "route",
        "oracle",
        "expected",
        "expected_answer",
        "answer",
        "verifier",
        "treatment",
        "bundle",
        "bundle_id",
        "policy",
        "policy_id",
        "registry",
        "hash",
        "first_bottleneck",
        "bottleneck",
    }
)

_HEX_DIGITS = frozenset(string.hexdigits)

_REQUEST_MARKERS: dict[str, tuple[str, ...]] = {
    "table": (
        "access code for",
        "directory table",
        "records table",
        "look up the access code",
        "locate the confirmation key",
    ),
    "form": (
        "complete the request form",
        "complete and submit the verification form",
        "enter it into the verification form",
        "submit it",
    ),
}


def derive_request_features(prompt: str) -> dict[str, Any]:
    """Derive bounded routing flags from the public task request only."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("public prompt must be a non-empty string")
    normalized = " ".join(prompt.casefold().split())
    table_positions = [
        normalized.find(marker)
        for marker in _REQUEST_MARKERS["table"]
        if marker in normalized
    ]
    form_positions = [
        normalized.find(marker)
        for marker in _REQUEST_MARKERS["form"]
        if marker in normalized
    ]
    table_operation = bool(table_positions)
    form_operation = bool(form_positions)
    first_operation: str | None = None
    if table_operation and form_operation:
        first_operation = (
            "table" if min(table_positions) < min(form_positions) else "form"
        )
    return {
        "table_operation": table_operation,
        "form_operation": form_operation,
        "first_operation": first_operation,
    }


# ---------------------------------------------------------------------------
# deterministic canonical serialization and self-hashing
# ---------------------------------------------------------------------------


def canonical_json(value: Any) -> str:
    """Deterministic, key-sorted JSON serialization used for every hash."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def canonical_hash(value: Any) -> str:
    """SHA-256 over the canonical serialization of ``value``."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _embed_self_hash(payload: dict[str, Any], field: str) -> str:
    """Embed a self-hash into ``payload`` and return the digest."""
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


def _load_json(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def _is_hex_digest(value: Any, length: int = 64) -> bool:
    """True if ``value`` is a lowercase hex digest of exactly ``length`` chars."""
    return (
        isinstance(value, str)
        and len(value) == length
        and set(value).issubset(_HEX_DIGITS)
    )


def _coordinate_commitment(coord: Mapping[str, Any]) -> str:
    """Deterministic private seal over a full private design coordinate.

    This is a coordinate commitment: it binds the exact private coordinate
    (stratum, capability, bottleneck, seed, template, nonce, oracle) without
    revealing any of its fields.  Only the digest is stored in the manifest.
    """
    return canonical_hash(dict(coord))


def _immutable_write(path: Path, value: Any) -> None:
    """Write ``value`` with immutable-write semantics.

    Refuses to overwrite an existing file unless the prospective bytes are
    byte-identical to what is already on disk (idempotent no-op).  New files are
    written atomically via a same-directory temporary file and ``os.replace``.
    """
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() == payload:
            return  # byte-identical: idempotent, nothing to do
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


def _design_fixtures(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return the manifest's bound per-fixture design commitments (or [])."""
    design = manifest.get("design")
    if not isinstance(design, Mapping):
        return []
    fixtures = design.get("fixtures")
    if not isinstance(fixtures, Sequence) or isinstance(fixtures, (str, bytes)):
        return []
    return [entry for entry in fixtures if isinstance(entry, Mapping)]


# ---------------------------------------------------------------------------
# spec -> manifest
# ---------------------------------------------------------------------------


def default_spec(*, seed: int = 20260813) -> dict[str, Any]:
    """Return a deterministic, plan-conformant Stage-A spec."""
    return {
        "schema_version": "m3-routing-probe-stage-a-spec-v1",
        "stage_a_id": f"stage-a-{seed}",
        "seed": seed,
        "strata": {
            "pure_table": {
                "table_operation": True,
                "form_operation": False,
                "first_operation": None,
            },
            "pure_form": {
                "table_operation": False,
                "form_operation": True,
                "first_operation": None,
            },
            "mixed": {
                "table_operation": True,
                "form_operation": True,
                "first_operation": "balanced",
            },
            "ambiguous": {
                "table_operation": None,
                "form_operation": None,
                "first_operation": None,
            },
        },
    }


def build_manifest(
    spec: Mapping[str, Any],
    *,
    design_adapter: Any = None,
) -> dict[str, Any]:
    """Deterministically derive a self-hashed immutable manifest from ``spec``.

    The spec supplies the operation-flag vocabulary per stratum and the master
    seed; the frozen probe schema, router weights, lambda grid, and gate
    thresholds are plan constants injected here (not user-editable) so the
    manifest cannot be weakened after freezing.

    The manifest additionally binds the *exact* generated Stage-A design and
    public fixtures through deterministic commitments: the generator version,
    the probe implementation/schema identities, and, per fixture, its opaque id,
    a private coordinate seal, the source HTML hash, and the public prompt
    hash.  ``design_adapter`` defaults to the production ``routing_fixtures``
    backend; tests may inject an equivalent adapter explicitly.
    """
    if not isinstance(spec, Mapping):
        raise ValueError("spec must be an object")
    if spec.get("schema_version") != "m3-routing-probe-stage-a-spec-v1":
        raise ValueError(
            f"spec schema must be m3-routing-probe-stage-a-spec-v1, "
            f"got {spec.get('schema_version')!r}"
        )
    seed = spec.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("spec.seed must be an integer")
    stage_a_id = spec.get("stage_a_id")
    if not isinstance(stage_a_id, str) or not stage_a_id.strip():
        raise ValueError("spec.stage_a_id must be a non-empty string")

    strata_spec = spec.get("strata")
    if not isinstance(strata_spec, Mapping) or set(strata_spec) != set(STRATA):
        raise ValueError(f"spec.strata must define exactly {list(STRATA)}")

    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "stage_a_id": str(stage_a_id),
        "seed": int(seed),
        "spec": dict(spec),
        "strata": list(STRATA),
        "specialists": list(SPECIALISTS),
        "probe_schema": dict(PROBE_SCHEMA),
        "probe_feature_names": list(PROBE_FEATURE_NAMES),
        "replays": PLAN_REPLAYS,
        "fixtures_per_stratum": PLAN_PER_STRATUM,
        "total_fixtures": PLAN_FIXTURES,
        "total_probes": PLAN_PROBES,
        "router": {
            "prompt_weight": ROUTER_PROMPT_WEIGHT,
            "probe_weight": ROUTER_PROBE_WEIGHT,
            "first_operation_bonus": ROUTER_FIRST_OPERATION_BONUS,
            "registry_order": list(SPECIALISTS),
            "table_mass_fields": list(TABLE_MASS_FIELDS),
            "form_mass_fields": list(FORM_MASS_FIELDS),
        },
        "utility": {
            "lambda_grid": list(LAMBDA_GRID),
            "primary_lambda": PRIMARY_LAMBDA,
            "cost_units_per_token": COST_UNITS_PER_TOKEN,
            "tie_break": "success_then_cost_then_registry_order",
        },
        "gates": {
            "valid_receipts": PLAN_PROBES,
            "router_agreement_min": PLAN_AGREEMENT_MIN,
            "router_agreement_min_per_stratum": PLAN_AGREEMENT_MIN_PER_STRATUM,
            "privacy_violations_max": PLAN_PRIVACY_MAX,
            "deterministic_features": True,
            "counts_within_bounds": True,
            "text_invariance": True,
            "probe_precedes_treatment": True,
            "synthetic_utility_pass": True,
        },
    }

    design = design_adapter if design_adapter is not None else _lazy_design_backend()
    generator_version = getattr(design, "GENERATOR_VERSION", None)
    if not isinstance(generator_version, str) or not generator_version.strip():
        raise ValueError(
            "design adapter must expose a non-empty GENERATOR_VERSION "
            "(the Stage-A generator identity)"
        )
    coordinates = list(design.build_stage_a_design(seed=int(seed)))
    coordinate_errors = _validate_design_coordinates(manifest, coordinates)
    if coordinate_errors:
        raise ValueError("design invalid: " + "; ".join(coordinate_errors))

    fixture_commitments: list[dict[str, Any]] = []
    for coord in coordinates:
        generated = design.generate_routing_fixture(coord)
        if (
            not isinstance(generated, Mapping)
            or not isinstance(generated.get("html"), str)
            or not isinstance(generated.get("prompt"), str)
        ):
            raise ValueError(
                f"{coord.get('fixture_id')!r}: fixture must supply html and public prompt"
            )
        html = generated["html"]
        prompt = generated["prompt"]
        source_sha = _sha256_bytes(html.encode("utf-8"))
        if generated.get("source_sha256") not in (None, source_sha):
            raise ValueError(
                f"{coord.get('fixture_id')!r}: generated source_sha256 mismatch"
            )
        fixture_commitments.append(
            {
                "fixture_id": str(coord["fixture_id"]),
                "private_seal": _coordinate_commitment(coord),
                "source_sha256": source_sha,
                "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
            }
        )

    manifest["design"] = {
        "generator_version": generator_version,
        "probe_receipt_schema": PROBE_RECEIPT_SCHEMA,
        "probe_mechanism": PROBE_MECHANISM,
        "fixtures": fixture_commitments,
    }
    _embed_self_hash(manifest, "manifest_hash")
    return manifest


# ---------------------------------------------------------------------------
# manifest validation
# ---------------------------------------------------------------------------


def validate_manifest(manifest: Mapping[str, Any]) -> list[str]:
    """Return a list of validation errors; empty means the manifest is valid."""
    errors: list[str] = []
    if not isinstance(manifest, Mapping):
        return ["manifest must be an object"]

    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        errors.append(
            f"schema_version must be {MANIFEST_SCHEMA!r}, "
            f"got {manifest.get('schema_version')!r}"
        )
    errors.extend(_verify_self_hash(manifest, "manifest_hash"))

    if set(manifest.get("strata", [])) != set(STRATA):
        errors.append(f"strata must equal {list(STRATA)}")
    if manifest.get("specialists") != list(SPECIALISTS):
        errors.append(f"specialists must equal {list(SPECIALISTS)}")

    probe_schema = manifest.get("probe_schema")
    if not isinstance(probe_schema, Mapping) or set(probe_schema) != set(PROBE_SCHEMA):
        errors.append(f"probe_schema must define exactly the 16 probe features")
    else:
        for name, entry in PROBE_SCHEMA.items():
            observed = probe_schema.get(name)
            if not isinstance(observed, Mapping):
                errors.append(f"probe_schema.{name} must be an object")
                continue
            if observed.get("type") != entry["type"]:
                errors.append(f"probe_schema.{name}.type must be {entry['type']!r}")
            cap = observed.get("cap")
            if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
                errors.append(f"probe_schema.{name}.cap must be a positive integer")

    if manifest.get("replays") != PLAN_REPLAYS:
        errors.append(f"replays must be {PLAN_REPLAYS}")
    if manifest.get("fixtures_per_stratum") != PLAN_PER_STRATUM:
        errors.append(f"fixtures_per_stratum must be {PLAN_PER_STRATUM}")
    if manifest.get("total_fixtures") != PLAN_FIXTURES:
        errors.append(f"total_fixtures must be {PLAN_FIXTURES}")
    if manifest.get("total_probes") != PLAN_PROBES:
        errors.append(f"total_probes must be {PLAN_PROBES}")

    router = manifest.get("router")
    if not isinstance(router, Mapping):
        errors.append("router must be an object")
    else:
        for field, expected in (
            ("prompt_weight", ROUTER_PROMPT_WEIGHT),
            ("probe_weight", ROUTER_PROBE_WEIGHT),
            ("first_operation_bonus", ROUTER_FIRST_OPERATION_BONUS),
        ):
            value = router.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value != expected:
                errors.append(f"router.{field} must equal {expected!r}")
        if router.get("registry_order") != list(SPECIALISTS):
            errors.append(f"router.registry_order must equal {list(SPECIALISTS)}")

    utility = manifest.get("utility")
    if not isinstance(utility, Mapping):
        errors.append("utility must be an object")
    else:
        if utility.get("lambda_grid") != list(LAMBDA_GRID):
            errors.append(f"utility.lambda_grid must equal {list(LAMBDA_GRID)}")
        if utility.get("primary_lambda") != PRIMARY_LAMBDA:
            errors.append(f"utility.primary_lambda must equal {PRIMARY_LAMBDA}")
        if utility.get("cost_units_per_token") != COST_UNITS_PER_TOKEN:
            errors.append(f"utility.cost_units_per_token must equal {COST_UNITS_PER_TOKEN}")
        if utility.get("tie_break") != "success_then_cost_then_registry_order":
            errors.append("utility.tie_break must be success_then_cost_then_registry_order")

    gates = manifest.get("gates")
    if not isinstance(gates, Mapping):
        errors.append("gates must be an object")
    else:
        for field, expected in (
            ("valid_receipts", PLAN_PROBES),
            ("router_agreement_min", PLAN_AGREEMENT_MIN),
            ("router_agreement_min_per_stratum", PLAN_AGREEMENT_MIN_PER_STRATUM),
            ("privacy_violations_max", PLAN_PRIVACY_MAX),
        ):
            if gates.get(field) != expected:
                errors.append(f"gates.{field} must equal {expected!r}")
        for field in (
            "deterministic_features",
            "counts_within_bounds",
            "text_invariance",
            "probe_precedes_treatment",
            "synthetic_utility_pass",
        ):
            if gates.get(field) is not True:
                errors.append(f"gates.{field} must be true")

    design = manifest.get("design")
    if not isinstance(design, Mapping):
        errors.append("design must be an object")
    else:
        generator_version = design.get("generator_version")
        if not isinstance(generator_version, str) or not generator_version.strip():
            errors.append("design.generator_version must be a non-empty string")
        if design.get("probe_receipt_schema") != PROBE_RECEIPT_SCHEMA:
            errors.append(
                f"design.probe_receipt_schema must equal {PROBE_RECEIPT_SCHEMA!r}"
            )
        if design.get("probe_mechanism") != PROBE_MECHANISM:
            errors.append(f"design.probe_mechanism must equal {PROBE_MECHANISM!r}")
        fixtures = design.get("fixtures")
        if not isinstance(fixtures, Sequence) or isinstance(fixtures, (str, bytes)):
            errors.append("design.fixtures must be a list")
        elif len(fixtures) != PLAN_FIXTURES:
            errors.append(f"design.fixtures must contain {PLAN_FIXTURES} entries")
        else:
            seen_ids: set[str] = set()
            for index, entry in enumerate(fixtures):
                if not isinstance(entry, Mapping):
                    errors.append(f"design.fixtures[{index}] must be an object")
                    continue
                fixture_id = entry.get("fixture_id")
                if not isinstance(fixture_id, str) or not fixture_id:
                    errors.append(f"design.fixtures[{index}] missing fixture_id")
                    continue
                if fixture_id in seen_ids:
                    errors.append(f"duplicate fixture_id {fixture_id!r} in design.fixtures")
                seen_ids.add(fixture_id)
                for field in ("private_seal", "source_sha256", "prompt_sha256"):
                    if not _is_hex_digest(entry.get(field)):
                        errors.append(
                            f"design.fixtures[{index}].{field} must be a sha256 hex digest"
                        )

    return errors


# ---------------------------------------------------------------------------
# frozen routing heuristic
# ---------------------------------------------------------------------------


def _mass(probe_features: Mapping[str, Any], fields: Sequence[str]) -> int:
    total = 0
    for field in fields:
        value = probe_features.get(field, 0)
        if isinstance(value, int) and not isinstance(value, bool):
            total += value
    return total


def _heuristic_breakdown(
    operation_flags: Mapping[str, Any],
    probe_features: Mapping[str, Any],
    *,
    prompt_weight: float,
    probe_weight: float,
    registry_order: Sequence[str],
) -> dict[str, Any]:
    """Transparent frozen heuristic: declared-operation evidence plus bounded
    structural-mass evidence, scored linearly, with a deterministic tie-break."""
    table_op = operation_flags.get("table_operation") is True
    form_op = operation_flags.get("form_operation") is True
    first = operation_flags.get("first_operation")

    prompt_table = 1.0 if table_op else 0.0
    prompt_form = 1.0 if form_op else 0.0
    if table_op and form_op and first in ("table", "form"):
        bonus = ROUTER_FIRST_OPERATION_BONUS
        if first == "table":
            prompt_table += bonus
        else:
            prompt_form += bonus

    table_mass = _mass(probe_features, TABLE_MASS_FIELDS)
    form_mass = _mass(probe_features, FORM_MASS_FIELDS)
    if table_mass == 0 and form_mass == 0:
        probe_table, probe_form = 0.0, 0.0
    elif table_mass > form_mass:
        probe_table, probe_form = 1.0, 0.0
    elif form_mass > table_mass:
        probe_table, probe_form = 0.0, 1.0
    else:
        probe_table, probe_form = 0.5, 0.5

    score_table = prompt_weight * prompt_table + probe_weight * probe_table
    score_form = prompt_weight * prompt_form + probe_weight * probe_form

    if score_table > score_form:
        choice = "table_specialist"
    elif score_form > score_table:
        choice = "form_specialist"
    elif prompt_table != prompt_form:
        choice = "table_specialist" if prompt_table > prompt_form else "form_specialist"
    else:
        choice = registry_order[0]

    return {
        "choice": choice,
        "prompt_table": prompt_table,
        "prompt_form": prompt_form,
        "probe_table": probe_table,
        "probe_form": probe_form,
        "table_mass": table_mass,
        "form_mass": form_mass,
        "score_table": score_table,
        "score_form": score_form,
    }


def frozen_heuristic(
    operation_flags: Mapping[str, Any],
    probe_features: Mapping[str, Any],
    router: Mapping[str, Any],
) -> dict[str, Any]:
    """Combined frozen router: operation flags plus probe counts."""
    return _heuristic_breakdown(
        operation_flags,
        probe_features,
        prompt_weight=float(router["prompt_weight"]),
        probe_weight=float(router["probe_weight"]),
        registry_order=list(router["registry_order"]),
    )


def prompt_only_heuristic(
    operation_flags: Mapping[str, Any],
    probe_features: Mapping[str, Any],
    router: Mapping[str, Any],
) -> dict[str, Any]:
    """Prompt-only baseline: declared operations only, probe weight zero."""
    return _heuristic_breakdown(
        operation_flags,
        probe_features,
        prompt_weight=1.0,
        probe_weight=0.0,
        registry_order=list(router["registry_order"]),
    )


def probe_only_heuristic(
    operation_flags: Mapping[str, Any],
    probe_features: Mapping[str, Any],
    router: Mapping[str, Any],
) -> dict[str, Any]:
    """Probe-only baseline: structural counts only, prompt weight zero."""
    return _heuristic_breakdown(
        operation_flags,
        probe_features,
        prompt_weight=0.0,
        probe_weight=1.0,
        registry_order=list(router["registry_order"]),
    )


# ---------------------------------------------------------------------------
# utility function and tie-break
# ---------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def utility(success: Any, cost_tokens: Any, lambda_value: float) -> float:
    """``success - lambda * (cost_tokens / 10000)`` with validation."""
    if not _is_number(success):
        raise ValueError(f"predicted success must be a finite number, got {success!r}")
    if not 0.0 <= float(success) <= 1.0:
        raise ValueError(f"predicted success must be in [0, 1], got {success!r}")
    if not _is_number(cost_tokens) or float(cost_tokens) < 0:
        raise ValueError(
            f"predicted output cost must be a finite non-negative number, "
            f"got {cost_tokens!r}"
        )
    if not _is_number(lambda_value):
        raise ValueError(f"lambda must be a finite number, got {lambda_value!r}")
    return float(success) - float(lambda_value) * (float(cost_tokens) / COST_UNITS_PER_TOKEN)


def select_policy(
    policies: Sequence[str],
    predictions: Mapping[str, Mapping[str, Any]],
    lambda_value: float,
    *,
    registry_order: Sequence[str] | None = None,
) -> str:
    """Argmax utility with tie-break: higher success, then lower cost, then
    immutable registry order.  Missing/non-finite predictions raise."""
    order = list(registry_order) if registry_order else list(policies)
    if set(policies) != set(order):
        raise ValueError("registry_order must be a permutation of policies")

    best_key: tuple[float, float, float, int] | None = None
    best_policy: str | None = None
    for policy in order:
        pred = predictions.get(policy)
        if not isinstance(pred, Mapping):
            raise ValueError(f"policy {policy!r}: missing prediction")
        success = pred.get("success")
        cost = pred.get("cost_tokens")
        value = utility(success, cost, lambda_value)
        # key: (utility, success, -cost, -registry_index) -> maximize lexicographic
        key = (
            value,
            float(success),
            -float(cost),
            -order.index(policy),
        )
        if best_key is None or key > best_key:
            best_key, best_policy = key, policy
    assert best_policy is not None
    return best_policy


# ---------------------------------------------------------------------------
# synthetic utility smoke
# ---------------------------------------------------------------------------


def run_synthetic_utility_smoke() -> dict[str, Any]:
    """Deterministic synthetic utility matrices covering dominance, success-cost
    tradeoffs, ties, missing/non-finite predictions, and every frozen lambda."""
    reasons: list[str] = []
    cases: dict[str, Any] = {}

    policies = ["A", "B"]

    # 1. dominance: A strictly better success and cheaper cost than B.
    dominance = {"A": {"success": 0.8, "cost_tokens": 5000},
                 "B": {"success": 0.5, "cost_tokens": 9000}}
    dominance_winners = {
        lam: select_policy(policies, dominance, lam) for lam in LAMBDA_GRID
    }
    dominance_ok = all(winner == "A" for winner in dominance_winners.values())
    cases["dominance"] = {"passed": dominance_ok, "winners_by_lambda": dominance_winners}
    if not dominance_ok:
        reasons.append("dominance: A should win at every lambda")

    # 2. success-cost tradeoff: A higher success but much higher cost.
    tradeoff = {"A": {"success": 0.9, "cost_tokens": 30000},
                "B": {"success": 0.6, "cost_tokens": 1000}}
    tradeoff_winners = {
        lam: select_policy(policies, tradeoff, lam) for lam in LAMBDA_GRID
    }
    # A wins at lambda=0; B wins at every positive lambda.
    tradeoff_ok = (
        tradeoff_winners[0.0] == "A"
        and all(tradeoff_winners[lam] == "B" for lam in LAMBDA_GRID if lam > 0)
    )
    cases["success_cost_tradeoff"] = {
        "passed": tradeoff_ok,
        "winners_by_lambda": tradeoff_winners,
    }
    if not tradeoff_ok:
        reasons.append("success_cost_tradeoff: expected A at lambda=0 and B otherwise")

    # 3. exact tie -> higher success, then lower cost, then registry order.
    tie_cases: dict[str, Any] = {}
    # identical predictions -> registry order (A).
    identical = {"A": {"success": 0.7, "cost_tokens": 10000},
                 "B": {"success": 0.7, "cost_tokens": 10000}}
    tie_cases["identical"] = {
        "winner": select_policy(policies, identical, 1.0),
        "expected": "A",
    }
    # equal utility from success and cost tradeoffs is impossible when
    # utilities match exactly; verify the pure cost tie-break instead: same
    # success, B cheaper -> B.
    cheaper = {"A": {"success": 0.7, "cost_tokens": 20000},
               "B": {"success": 0.7, "cost_tokens": 10000}}
    tie_cases["lower_cost_wins"] = {
        "winner": select_policy(policies, cheaper, 1.0),
        "expected": "B",
    }
    tie_ok = all(case["winner"] == case["expected"] for case in tie_cases.values())
    cases["tie_break"] = {"passed": tie_ok, "cases": tie_cases}
    if not tie_ok:
        reasons.append("tie_break: tie-break ordering violated")

    # 4. missing / non-finite predictions must be detected, not silently chosen.
    invalid_predictions = [
        {"A": {"success": None, "cost_tokens": 1000}, "B": {"success": 0.5, "cost_tokens": 1000}},
        {"A": {"success": 0.5, "cost_tokens": None}, "B": {"success": 0.5, "cost_tokens": 1000}},
        {"A": {"success": float("nan"), "cost_tokens": 1000}, "B": {"success": 0.5, "cost_tokens": 1000}},
        {"A": {"success": 0.5, "cost_tokens": float("inf")}, "B": {"success": 0.5, "cost_tokens": 1000}},
        {"A": {"success": 0.5, "cost_tokens": -5}, "B": {"success": 0.5, "cost_tokens": 1000}},
        {"A": {"success": 1.5, "cost_tokens": 1000}, "B": {"success": 0.5, "cost_tokens": 1000}},
    ]
    detected = 0
    for pred in invalid_predictions:
        try:
            select_policy(policies, pred, 1.0)
        except ValueError:
            detected += 1
    missing_ok = detected == len(invalid_predictions)
    cases["missing_nonfinite_detected"] = {
        "passed": missing_ok,
        "detected": detected,
        "total": len(invalid_predictions),
    }
    if not missing_ok:
        reasons.append(
            f"missing_nonfinite_detected: expected {len(invalid_predictions)} "
            f"detections, got {detected}"
        )

    # 5. lambda-grid consistency: primary lambda present and monotone cost
    # sensitivity verified in the tradeoff case above.
    grid_ok = (
        PRIMARY_LAMBDA in LAMBDA_GRID
        and tuple(sorted(LAMBDA_GRID)) == LAMBDA_GRID
        and all(lam >= 0 for lam in LAMBDA_GRID)
    )
    cases["lambda_grid_consistency"] = {"passed": grid_ok}
    if not grid_ok:
        reasons.append("lambda_grid_consistency: grid must be sorted non-negative")

    passed = not reasons
    return {
        "passed": passed,
        "cases": cases,
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# privacy audit
# ---------------------------------------------------------------------------


def _looks_like_leak(value: str) -> bool:
    if not value:
        return False
    if value.startswith("/") or re.match(r"^[A-Za-z]:[/\\]", value):
        return True  # absolute path
    if re.match(r"^https?://", value):
        return True  # URL
    if len(value) == 64 and set(value).issubset(_HEX_DIGITS):
        return True  # hex digest (source/feature hash)
    return False


def privacy_scan(value: Any, prefix: str = "<root>") -> list[str]:
    """Recursively report forbidden keys and leak-like string values."""
    violations: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str) and key in _FORBIDDEN_KEYS:
                violations.append(f"forbidden key {key!r} at {prefix}")
            violations.extend(privacy_scan(item, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            violations.extend(privacy_scan(item, f"{prefix}[{index}]"))
    elif isinstance(value, str):
        if _looks_like_leak(value):
            violations.append(f"leak-like string at {prefix}: {value!r}")
    return violations


def _model_visible(fixture: Mapping[str, Any]) -> dict[str, Any]:
    """Return only derived request features and bounded probe counts."""
    return {
        "request_features": fixture["request_features"],
        "probe_features": fixture["replays"][0]["features"],
    }


# ---------------------------------------------------------------------------
# design validation
# ---------------------------------------------------------------------------


def _validate_operation_flags(stratum: str, flags: Mapping[str, Any]) -> list[str]:
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


def _validate_design_coordinates(manifest: Mapping[str, Any], design: Sequence[Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(design, Sequence) or isinstance(design, (str, bytes)):
        return ["design must be a sequence of coordinates"]
    if len(design) != PLAN_FIXTURES:
        return [f"design must contain {PLAN_FIXTURES} coordinates, got {len(design)}"]

    per_stratum: dict[str, int] = {stratum: 0 for stratum in STRATA}
    seen_ids: set[str] = set()
    for index, coord in enumerate(design):
        if not isinstance(coord, Mapping):
            errors.append(f"design[{index}] is not an object")
            continue
        fixture_id = coord.get("fixture_id")
        if not isinstance(fixture_id, str) or not fixture_id:
            errors.append(f"design[{index}] missing string fixture_id")
            continue
        if fixture_id in seen_ids:
            errors.append(f"duplicate fixture_id {fixture_id!r}")
        seen_ids.add(fixture_id)
        stratum = coord.get("stratum")
        if stratum not in STRATA:
            errors.append(f"{fixture_id}: unknown stratum {stratum!r}")
            continue
        per_stratum[stratum] += 1
        label = coord.get("first_bottleneck")
        if label not in SPECIALISTS:
            errors.append(f"{fixture_id}: first_bottleneck must be one of {SPECIALISTS}")
        flags = coord.get("operation_flags")
        if not isinstance(flags, Mapping):
            errors.append(f"{fixture_id}: operation_flags must be an object")
        else:
            errors.extend(
                f"{fixture_id}: {error}" for error in _validate_operation_flags(stratum, flags)
            )

    for stratum in STRATA:
        if per_stratum[stratum] != PLAN_PER_STRATUM:
            errors.append(
                f"stratum {stratum!r} must contain {PLAN_PER_STRATUM} fixtures, "
                f"got {per_stratum[stratum]}"
            )
    return errors


# ---------------------------------------------------------------------------
# probe execution (cold replays, no model)
# ---------------------------------------------------------------------------


def _probe(html: str, probe_adapter: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Call the probe adapter and normalize to ``(features, receipt)``."""
    raw = probe_adapter.structural_probe(html)
    if isinstance(raw, tuple) and len(raw) == 2:
        features, receipt = raw
    elif isinstance(raw, Mapping) and "features" in raw and "receipt" in raw:
        features, receipt = raw["features"], raw["receipt"]
    else:
        raise ValueError("probe adapter must return (features, receipt) or a mapping with both")
    if not isinstance(features, Mapping):
        raise ValueError("probe features must be an object")
    if not isinstance(receipt, Mapping):
        raise ValueError("probe receipt must be an object")
    return dict(features), dict(receipt)


def _validate_features(manifest: Mapping[str, Any], features: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    probe_schema = manifest["probe_schema"]
    if set(features) != set(PROBE_FEATURE_NAMES):
        errors.append(
            f"features must define exactly the 16 probe fields, got {sorted(features)}"
        )
    for name, entry in probe_schema.items():
        value = features.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            errors.append(f"feature {name!r} must be a non-negative integer, got {value!r}")
            continue
        cap = int(entry["cap"])
        if value > cap:
            errors.append(f"feature {name!r} = {value} exceeds frozen bound {cap}")
    return errors


class _TextMutator(HTMLParser):
    """Rewrite visible text only, preserving tags, attributes, and nesting."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self._out: list[str] = []

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        self._out.append(self.get_starttag_text())

    def handle_startendtag(self, tag: str, attrs: Any) -> None:
        self._out.append(self.get_starttag_text())

    def handle_endtag(self, tag: str) -> None:
        self._out.append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        if data and data.strip():
            self._out.append(re.sub(r"\S", "X", data))
        else:
            self._out.append(data)

    def handle_entityref(self, name: str) -> None:
        self._out.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self._out.append(f"&#{name};")

    def handle_comment(self, data: str) -> None:
        self._out.append(f"<!--{data}-->")

    def handle_decl(self, decl: str) -> None:
        self._out.append(f"<!{decl}>")

    def handle_pi(self, data: str) -> None:
        self._out.append(f"<?{data}>")

    def unknown_decl(self, data: str) -> None:
        self._out.append(f"<![{data}]>")

    def text(self) -> str:
        return "".join(self._out)


def _mutate_text(html: str) -> str:
    parser = _TextMutator()
    parser.feed(html)
    parser.close()
    return parser.text()


class _SealedLabels:
    """Private first-bottleneck labels readable only after decisions finalize."""

    def __init__(self, labels: Mapping[str, str]) -> None:
        self._labels = dict(labels)
        self._open = False

    def unseal(self) -> "_SealedLabels":
        self._open = True
        return self

    def get(self, fixture_id: str) -> str:
        if not self._open:
            raise RuntimeError("sealed labels read before decisions were finalized")
        return self._labels[fixture_id]


# ---------------------------------------------------------------------------
# run (collect probes) and analyze (compute gate)
# ---------------------------------------------------------------------------


def run_stage_a(
    manifest: Mapping[str, Any],
    *,
    design_adapter: Any = None,
    probe_adapter: Any = None,
    utility_validator: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run Stage A locally: generate fixtures, perform three cold probe replays
    per fixture plus a text-invariance probe, then analyze.  No model executes.

    The production design is generated from ``manifest["seed"]`` (never the
    backend default), and every generated fixture is authorized fail-closed
    against the manifest's bound design commitments (fixture id, private seal,
    source hash, prompt hash).

    ``design_adapter`` and ``probe_adapter`` default to lazily-imported
    ``routing_fixtures`` and ``structural_probe`` backends; inject mocks in
    tests (injected adapters must accept ``build_stage_a_design(seed=...)``).
    ``utility_validator`` is an optional external synthetic-smoke cross-check
    (see ``routing_utility``).
    """
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("manifest invalid: " + "; ".join(errors))

    design = design_adapter if design_adapter is not None else _lazy_design_backend()
    probe = probe_adapter if probe_adapter is not None else _lazy_probe_backend()

    seed = int(manifest["seed"])
    coordinates = list(design.build_stage_a_design(seed=seed))
    coordinate_errors = _validate_design_coordinates(manifest, coordinates)
    if coordinate_errors:
        raise ValueError("design invalid: " + "; ".join(coordinate_errors))

    # ---- artifact authorization: fail closed against bound manifest entries ---
    bound = {entry["fixture_id"]: entry for entry in _design_fixtures(manifest)}

    fixtures: list[dict[str, Any]] = []
    for coord in coordinates:
        generated = design.generate_routing_fixture(coord)
        if (
            not isinstance(generated, Mapping)
            or not isinstance(generated.get("html"), str)
            or not isinstance(generated.get("prompt"), str)
        ):
            raise ValueError(
                f"{coord.get('fixture_id')!r}: fixture must supply html and public prompt"
            )
        html = generated["html"]
        request_features = derive_request_features(generated["prompt"])
        request_errors = _validate_operation_flags(
            str(coord["stratum"]), request_features
        )
        if request_errors:
            raise ValueError(
                f"{coord.get('fixture_id')!r}: public request features invalid: "
                + "; ".join(request_errors)
            )
        source_sha = _sha256_bytes(html.encode("utf-8"))
        if generated.get("source_sha256") not in (None, source_sha):
            raise ValueError(f"{coord.get('fixture_id')!r}: source_sha256 mismatch")

        commitment = bound.get(coord["fixture_id"])
        if commitment is None:
            raise ValueError(
                f"{coord.get('fixture_id')!r}: not bound by the manifest design"
            )
        if commitment.get("private_seal") != _coordinate_commitment(coord):
            raise ValueError(
                f"{coord.get('fixture_id')!r}: private coordinate seal mismatch "
                f"against manifest"
            )
        if commitment.get("source_sha256") != source_sha:
            raise ValueError(
                f"{coord.get('fixture_id')!r}: source_sha256 not authorized by manifest"
            )
        prompt_sha = _sha256_bytes(generated["prompt"].encode("utf-8"))
        if commitment.get("prompt_sha256") != prompt_sha:
            raise ValueError(
                f"{coord.get('fixture_id')!r}: prompt_sha256 not authorized by manifest"
            )
        fixtures.append(
            {
                "fixture_id": coord["fixture_id"],
                "stratum": coord["stratum"],
                "first_bottleneck": coord["first_bottleneck"],
                "request_features": request_features,
                "html": html,
                "source_sha256": source_sha,
                "source_byte_count": len(html.encode("utf-8")),
            }
        )

    probe_results: list[dict[str, Any]] = []
    for fixture in fixtures:
        replays: list[dict[str, Any]] = []
        for _ in range(PLAN_REPLAYS):
            features, receipt = _probe(fixture["html"], probe)
            replays.append(
                {
                    "features": features,
                    "feature_sha256": canonical_hash(features),
                    "receipt": receipt,
                }
            )
        # text-invariance probe: same structure, different visible text.
        text_variant_html = _mutate_text(fixture["html"])
        text_variant: dict[str, Any] | None = None
        if text_variant_html != fixture["html"]:
            variant_features, variant_receipt = _probe(text_variant_html, probe)
            text_variant = {
                "source_sha256": _sha256_bytes(text_variant_html.encode("utf-8")),
                "features": variant_features,
                "feature_sha256": canonical_hash(variant_features),
                "receipt": variant_receipt,
            }
        probe_results.append(
            {
                "fixture_id": fixture["fixture_id"],
                "stratum": fixture["stratum"],
                "first_bottleneck": fixture["first_bottleneck"],
                "request_features": fixture["request_features"],
                "source_sha256": fixture["source_sha256"],
                "source_byte_count": fixture["source_byte_count"],
                "replays": replays,
                "text_variant": text_variant,
            }
        )

    return analyze_stage_a(manifest, probe_results, utility_validator=utility_validator)


def analyze_stage_a(
    manifest: Mapping[str, Any],
    probe_results: Sequence[Mapping[str, Any]],
    *,
    utility_validator: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Pure, deterministic gate computation over collected probe results.

    Decisions are computed from public fields only; the sealed first-bottleneck
    labels are revealed afterward for scoring (labels never enter the router).
    """
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("manifest invalid: " + "; ".join(errors))
    if not isinstance(probe_results, Sequence) or isinstance(probe_results, (str, bytes)):
        raise ValueError("probe_results must be a sequence")
    if len(probe_results) != PLAN_FIXTURES:
        raise ValueError(
            f"probe_results must contain {PLAN_FIXTURES} fixtures, got {len(probe_results)}"
        )

    router = manifest["router"]
    gates = manifest["gates"]

    mechanical_errors: list[str] = []
    threshold_reasons: list[str] = []
    privacy_violations: list[str] = []
    receipt_errors: list[str] = []
    feature_errors: list[str] = []

    # ---- phase 0: mechanical validation of probe results ---------------------
    normalized: list[dict[str, Any]] = []
    labels: dict[str, str] = {}
    seen_ids: set[str] = set()
    strata_counts: dict[str, int] = {stratum: 0 for stratum in STRATA}

    for fixture in probe_results:
        if not isinstance(fixture, Mapping):
            mechanical_errors.append("a probe result is not an object")
            continue
        fixture_id = fixture.get("fixture_id")
        if not isinstance(fixture_id, str) or not fixture_id:
            mechanical_errors.append("probe result missing fixture_id")
            continue
        if fixture_id in seen_ids:
            mechanical_errors.append(f"duplicate probe result for {fixture_id!r}")
        seen_ids.add(fixture_id)
        stratum = fixture.get("stratum")
        if stratum not in STRATA:
            mechanical_errors.append(f"{fixture_id}: unknown stratum {stratum!r}")
            continue
        strata_counts[stratum] += 1
        label = fixture.get("first_bottleneck")
        if label not in SPECIALISTS:
            mechanical_errors.append(f"{fixture_id}: missing first_bottleneck label")
        else:
            labels[fixture_id] = str(label)
        flags = fixture.get("request_features")
        if not isinstance(flags, Mapping):
            mechanical_errors.append(f"{fixture_id}: request_features missing")
            continue
        mechanical_errors.extend(
            f"{fixture_id}: {error}" for error in _validate_operation_flags(stratum, flags)
        )
        replays = fixture.get("replays")
        if not isinstance(replays, Sequence) or len(replays) != PLAN_REPLAYS:
            mechanical_errors.append(
                f"{fixture_id}: expected {PLAN_REPLAYS} replays, got "
                f"{len(replays) if isinstance(replays, Sequence) else 'n/a'}"
            )
            continue
        normalized_replays: list[dict[str, Any]] = []
        for replay in replays:
            if not isinstance(replay, Mapping):
                mechanical_errors.append(f"{fixture_id}: malformed replay")
                continue
            features = replay.get("features")
            if not isinstance(features, Mapping):
                mechanical_errors.append(f"{fixture_id}: replay features missing")
                continue
            feature_errors.extend(
                f"{fixture_id}: {error}" for error in _validate_features(manifest, features)
            )
            receipt = replay.get("receipt")
            if not isinstance(receipt, Mapping):
                mechanical_errors.append(f"{fixture_id}: replay receipt missing")
                continue
            receipt_errors.extend(
                f"{fixture_id}: {error}"
                for error in audit_receipt(receipt, features)
            )
            if receipt.get("source_sha256") != fixture.get("source_sha256"):
                receipt_errors.append(
                    f"{fixture_id}: receipt.source_sha256 does not match fixture source"
                )
            if receipt.get("source_bytes") != fixture.get("source_byte_count"):
                receipt_errors.append(
                    f"{fixture_id}: receipt.source_bytes does not match fixture source"
                )
            normalized_replays.append(
                {
                    "features": dict(features),
                    "feature_sha256": str(
                        receipt.get("canonical_feature_sha256", "")
                    ),
                    "receipt": dict(receipt),
                }
            )
        normalized.append(
            {
                "fixture_id": fixture_id,
                "stratum": stratum,
                "source_sha256": fixture.get("source_sha256"),
                "request_features": dict(flags),
                "replays": normalized_replays,
                "text_variant": (
                    dict(fixture["text_variant"])
                    if isinstance(fixture.get("text_variant"), Mapping)
                    else None
                ),
            }
        )

    for stratum in STRATA:
        if strata_counts[stratum] != PLAN_PER_STRATUM:
            mechanical_errors.append(
                f"probe_results stratum {stratum!r} must contain "
                f"{PLAN_PER_STRATUM} fixtures, got {strata_counts[stratum]}"
            )

    valid_receipt_count = PLAN_PROBES - len(receipt_errors)
    deterministic_count = 0
    for fixture in normalized:
        hashes = {replay["feature_sha256"] for replay in fixture["replays"]}
        features_sets = [replay["features"] for replay in fixture["replays"]]
        if len(hashes) == 1 and all(f == features_sets[0] for f in features_sets[1:]):
            deterministic_count += 1

    # ---- text invariance -----------------------------------------------------
    text_invariant_count = 0
    for fixture in normalized:
        variant = fixture["text_variant"]
        if variant is None:
            continue
        base = fixture["replays"][0]["features"]
        variant_features = variant.get("features")
        if (
            isinstance(variant_features, Mapping)
            and dict(variant_features) == base
            and variant.get("source_sha256")
            and variant.get("source_sha256") != fixture.get("source_sha256")
        ):
            text_invariant_count += 1

    # ---- privacy audit (model-visible objects only) --------------------------
    for fixture in normalized:
        model_visible = _model_visible(fixture)
        privacy_violations.extend(
            f"{fixture['fixture_id']}: {violation}"
            for violation in privacy_scan(model_visible)
        )

    # ---- phase 1: router decisions from public fields only -------------------
    sealed = _SealedLabels(labels)
    combined_decisions: dict[str, str] = {}
    prompt_only_decisions: dict[str, str] = {}
    probe_only_decisions: dict[str, str] = {}
    for fixture in normalized:
        flags = fixture["request_features"]
        features = fixture["replays"][0]["features"]
        combined_decisions[fixture["fixture_id"]] = frozen_heuristic(flags, features, router)[
            "choice"
        ]
        prompt_only_decisions[fixture["fixture_id"]] = prompt_only_heuristic(
            flags, features, router
        )["choice"]
        probe_only_decisions[fixture["fixture_id"]] = probe_only_heuristic(
            flags, features, router
        )["choice"]

    # ---- phase 2: reveal labels and score (post-decision only) ---------------
    sealed.unseal()
    agreement_by_stratum: dict[str, int] = {stratum: 0 for stratum in STRATA}
    prompt_agreement_by_stratum: dict[str, int] = {stratum: 0 for stratum in STRATA}
    probe_agreement_by_stratum: dict[str, int] = {stratum: 0 for stratum in STRATA}
    decisions: list[dict[str, Any]] = []
    for fixture in normalized:
        fid = fixture["fixture_id"]
        label = sealed.get(fid)
        choice = combined_decisions[fid]
        correct = choice == label
        agreement_by_stratum[fixture["stratum"]] += int(correct)
        prompt_agreement_by_stratum[fixture["stratum"]] += int(
            prompt_only_decisions[fid] == label
        )
        probe_agreement_by_stratum[fixture["stratum"]] += int(
            probe_only_decisions[fid] == label
        )
        decisions.append(
            {
                "fixture_id": fid,
                "stratum": fixture["stratum"],
                "combined_choice": choice,
                "prompt_only_choice": prompt_only_decisions[fid],
                "probe_only_choice": probe_only_decisions[fid],
                "sealed_label": label,
                "correct": correct,
            }
        )

    combined_agreement = sum(agreement_by_stratum.values())
    prompt_only_agreement = sum(prompt_agreement_by_stratum.values())
    probe_only_agreement = sum(probe_agreement_by_stratum.values())
    min_per_stratum = min(agreement_by_stratum.values(), default=0)

    # ---- synthetic utility smoke --------------------------------------------
    utility_smoke = run_utility_scoring_smoke_matrix(lambda_grid=LAMBDA_GRID)
    external_utility: dict[str, Any] | None = None
    if utility_validator is not None:
        result = utility_validator(utility_smoke)
        if not isinstance(result, Mapping):
            raise ValueError("utility validator must return a mapping")
        external_utility = dict(result)
        if external_utility.get("passed") is not True:
            utility_smoke.setdefault("reasons", []).append(
                "external synthetic utility validator failed"
            )
            utility_smoke["passed"] = False

    # ---- assemble checks -----------------------------------------------------
    receipt_valid = len(receipt_errors) == 0 and valid_receipt_count == PLAN_PROBES
    deterministic_ok = deterministic_count == PLAN_FIXTURES
    counts_ok = len(feature_errors) == 0
    privacy_ok = len(privacy_violations) == 0
    text_invariance_ok = text_invariant_count == PLAN_FIXTURES
    router_ok = (
        combined_agreement >= gates["router_agreement_min"]
        and min_per_stratum >= gates["router_agreement_min_per_stratum"]
    )
    utility_ok = utility_smoke["passed"]

    mechanics_valid = not mechanical_errors and receipt_valid

    # ---- phase/order receipt (by construction: probes -> routes -> labels) ---
    phase_receipt = {
        "order": ["probe_collection", "router_decision", "label_reveal"],
        "probe_collection": {
            "position": 0,
            "produces": "96 cold probe replays + text-invariance variants",
            "precedes": ["router_decision", "label_reveal", "treatment_assignment"],
        },
        "router_decision": {
            "position": 1,
            "produces": "frozen combined / prompt-only / probe-only routes",
            "follows": ["probe_collection"],
            "precedes": ["label_reveal", "treatment_assignment"],
        },
        "label_reveal": {
            "position": 2,
            "produces": "post-decision agreement scoring against sealed labels",
            "follows": ["probe_collection", "router_decision"],
        },
        "treatment_assignment": None,
        "probe_precedes_treatment": True,
        "zero_outcome": True,
    }

    checks = {
        "manifest_valid": True,
        "design_mechanics_valid": not mechanical_errors,
        "receipts_valid": receipt_valid,
        "counts_within_bounds": counts_ok,
        "deterministic_features": deterministic_ok,
        "text_invariance": text_invariance_ok,
        "privacy_zero": privacy_ok,
        "probe_precedes_treatment": phase_receipt["probe_precedes_treatment"] is True,
        "router_agreement": router_ok,
        "synthetic_utility_pass": utility_ok,
    }

    if not mechanics_valid:
        decision = "invalid"
        passed = False
    elif not all(checks.values()):
        decision = "probe_no_go"
        passed = False
    else:
        decision = "probe_pass"
        passed = True

    threshold_reasons = [name for name, ok in checks.items() if not ok]
    if not counts_ok:
        threshold_reasons.extend(feature_errors)
    if not deterministic_ok:
        threshold_reasons.append(
            f"deterministic features in {deterministic_count}/{PLAN_FIXTURES} fixtures"
        )
    if not text_invariance_ok:
        threshold_reasons.append(
            f"text invariance demonstrated in {text_invariant_count}/{PLAN_FIXTURES} fixtures"
        )
    if not privacy_ok:
        threshold_reasons.extend(privacy_violations)
    if not router_ok:
        threshold_reasons.append(
            f"combined router agreement {combined_agreement}/{PLAN_FIXTURES} "
            f"(min {gates['router_agreement_min']}) with per-stratum minimum "
            f"{min_per_stratum} (min {gates['router_agreement_min_per_stratum']})"
        )
    if not utility_ok:
        threshold_reasons.extend(utility_smoke.get("reasons", []))

    reasons = sorted(set(mechanical_errors + receipt_errors + threshold_reasons))

    exit_code = 0 if decision == "probe_pass" else (2 if decision == "probe_no_go" else 1)

    # ---- per-fixture audit commitments for Stage B authorization -------------
    # Source hash, canonical feature hash, and the frozen combined + prompt-only
    # routes.  No HTML, page text, oracle, nonce, or label is exposed here.
    fixture_commitments: list[dict[str, Any]] = []
    for fixture in normalized:
        fid = fixture["fixture_id"]
        fixture_commitments.append(
            {
                "fixture_id": fid,
                "source_sha256": fixture["source_sha256"],
                "canonical_feature_sha256": fixture["replays"][0]["feature_sha256"],
                "combined_route": combined_decisions[fid],
                "prompt_only_route": prompt_only_decisions[fid],
            }
        )

    report: dict[str, Any] = {
        "gate": GATE_SCHEMA,
        "schema_version": GATE_SCHEMA,
        "stage_a_id": manifest.get("stage_a_id"),
        "manifest_hash": manifest["manifest_hash"],
        "passed": passed,
        "decision": decision,
        "exit_code": exit_code,
        "checks": checks,
        "reasons": reasons,
        "completeness": {
            "fixtures": PLAN_FIXTURES,
            "replays_per_fixture": PLAN_REPLAYS,
            "total_probes": PLAN_PROBES,
            "valid_receipts": valid_receipt_count,
            "receipt_errors": receipt_errors,
            "mechanical_errors": mechanical_errors,
        },
        "determinism": {
            "deterministic_fixtures": deterministic_count,
            "text_invariant_fixtures": text_invariant_count,
            "count_out_of_bounds_errors": feature_errors,
        },
        "privacy": {
            "violations": privacy_violations,
            "violation_count": len(privacy_violations),
        },
        "routing": {
            "combined_agreement": combined_agreement,
            "combined_agreement_by_stratum": agreement_by_stratum,
            "prompt_only_agreement": prompt_only_agreement,
            "prompt_only_by_stratum": prompt_agreement_by_stratum,
            "probe_only_agreement": probe_only_agreement,
            "probe_only_by_stratum": probe_agreement_by_stratum,
            "required_min": gates["router_agreement_min"],
            "required_min_per_stratum": gates["router_agreement_min_per_stratum"],
            "decisions": decisions,
        },
        "phases": phase_receipt,
        "fixture_commitments": fixture_commitments,
        "synthetic_utility": utility_smoke,
        "external_utility_validator": external_utility,
        "warning": (
            "Stage A is a zero-outcome probe gate; it authorizes corpus collection "
            "only. Passing does not establish routing effectiveness or allocator value."
        ),
    }
    _embed_self_hash(report, "report_hash")
    return report


def validate_gate_report(
    report: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> list[str]:
    """Return a list of gate-report validation errors; empty means valid.

    Verifies, fail-closed: gate schema identity, the embedded ``report_hash``
    self-hash, manifest binding (``manifest_hash`` and ``stage_a_id``), and the
    pass semantics of ``decision`` / ``passed`` / ``exit_code`` / ``checks``.
    """
    errors: list[str] = []
    if not isinstance(report, Mapping):
        return ["gate report must be an object"]

    if report.get("gate") != GATE_SCHEMA:
        errors.append(f"gate must equal {GATE_SCHEMA!r}, got {report.get('gate')!r}")
    if report.get("schema_version") != GATE_SCHEMA:
        errors.append(
            f"schema_version must equal {GATE_SCHEMA!r}, "
            f"got {report.get('schema_version')!r}"
        )
    errors.extend(_verify_self_hash(report, "report_hash"))

    if report.get("manifest_hash") != manifest.get("manifest_hash"):
        errors.append("report.manifest_hash does not match the manifest")
    if report.get("stage_a_id") != manifest.get("stage_a_id"):
        errors.append("report.stage_a_id does not match the manifest")

    decision = report.get("decision")
    passed = report.get("passed")
    exit_code = report.get("exit_code")
    expected = {
        "probe_pass": (True, 0),
        "probe_no_go": (False, 2),
        "invalid": (False, 1),
    }
    if decision not in expected:
        errors.append(f"unknown decision {decision!r}")
    else:
        exp_passed, exp_exit = expected[decision]
        if passed is not exp_passed:
            errors.append(
                f"decision {decision!r} implies passed={exp_passed}, got {passed!r}"
            )
        if exit_code != exp_exit:
            errors.append(
                f"decision {decision!r} implies exit_code={exp_exit}, got {exit_code!r}"
            )

    checks = report.get("checks")
    if not isinstance(checks, Mapping):
        errors.append("checks must be an object")
    elif passed is True and any(ok is not True for ok in checks.values()):
        errors.append("passed=True but at least one check is not True")
    elif passed is False and decision == "probe_pass":
        errors.append("passed=False contradicts decision probe_pass")

    return errors


# ---------------------------------------------------------------------------
# lazy backend imports (forthcoming APIs)
# ---------------------------------------------------------------------------


def _try_import(*names: str) -> Any:
    for name in names:
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    raise ImportError("none of %s are importable" % ", ".join(names))


def _load_module_from_file(path: str) -> Any:
    """Load a Python file as a throwaway module (for backend injection)."""
    resolved = str(Path(path).expanduser().resolve())
    spec = importlib.util.spec_from_file_location("_gate_backend", resolved)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load backend module from {resolved!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _lazy_design_backend() -> Any:
    try:
        return _try_import("pyreplab_harness.routing_fixtures", "routing_fixtures")
    except ImportError as error:
        raise RuntimeError(
            "routing_fixtures backend unavailable; inject a design adapter "
            "(build_stage_a_design/generate_routing_fixture)"
        ) from error


def _lazy_probe_backend() -> Any:
    try:
        return _try_import("pyreplab_harness.structural_probe", "structural_probe")
    except ImportError as error:
        raise RuntimeError(
            "structural_probe backend unavailable; inject a probe adapter "
            "(structural_probe(html) -> features + receipt)"
        ) from error


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyreplab-m3-routing-probe-gate",
        description="Freeze, validate, and run the Stage A zero-outcome probe gate.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze", help="derive and freeze a self-hashed manifest")
    freeze.add_argument("spec", help="path to the Stage-A spec JSON")
    freeze.add_argument("output", help="path to write the frozen manifest JSON")

    validate = subparsers.add_parser("validate", help="validate a frozen manifest")
    validate.add_argument("manifest", help="path to the manifest JSON")

    run = subparsers.add_parser("run", help="run Stage A locally and emit the gate report")
    run.add_argument("manifest", help="path to the manifest JSON")
    run.add_argument("--output", default=None, help="path to write the gate report JSON")
    run.add_argument(
        "--design-module",
        default=None,
        help="path to a Python file exposing build_stage_a_design/generate_routing_fixture",
    )
    run.add_argument(
        "--probe-module",
        default=None,
        help="path to a Python file exposing structural_probe(html)",
    )
    run.add_argument(
        "--utility-module",
        default=None,
        help="path to a Python file exposing synthetic_smoke_validator(report)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "freeze":
            spec = _load_json(args.spec)
            manifest = build_manifest(spec)
            manifest_errors = validate_manifest(manifest)
            if manifest_errors:
                raise ValueError("manifest failed self-validation: " + "; ".join(manifest_errors))
            _immutable_write(Path(args.output), manifest)
            summary = {
                "command": "freeze",
                "manifest": str(Path(args.output).expanduser().resolve()),
                "manifest_hash": manifest["manifest_hash"],
                "stage_a_id": manifest["stage_a_id"],
            }
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0

        if args.command == "validate":
            manifest = _load_json(args.manifest)
            manifest_errors = validate_manifest(manifest)
            result = {
                "command": "validate",
                "valid": not manifest_errors,
                "manifest_hash": manifest.get("manifest_hash"),
                "errors": manifest_errors,
            }
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if not manifest_errors else 1

        if args.command == "run":
            manifest = _load_json(args.manifest)
            design_adapter = (
                _load_module_from_file(args.design_module) if args.design_module else None
            )
            probe_adapter = (
                _load_module_from_file(args.probe_module) if args.probe_module else None
            )
            utility_validator = None
            if args.utility_module:
                module = _load_module_from_file(args.utility_module)
                utility_validator = getattr(module, "synthetic_smoke_validator", None)
                if not callable(utility_validator):
                    raise ValueError(
                        "utility module must expose callable synthetic_smoke_validator"
                    )
            report = run_stage_a(
                manifest,
                design_adapter=design_adapter,
                probe_adapter=probe_adapter,
                utility_validator=utility_validator,
            )
            if args.output:
                _immutable_write(Path(args.output), report)
            print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
            return int(report["exit_code"])

        raise ValueError(f"unknown command {args.command!r}")
    except (OSError, ValueError, RuntimeError) as error:
        print(f"routing probe gate error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GATE_SCHEMA",
    "MANIFEST_SCHEMA",
    "PROBE_SCHEMA",
    "PROBE_FEATURE_NAMES",
    "STRATA",
    "SPECIALISTS",
    "PLAN_FIXTURES",
    "PLAN_PROBES",
    "PLAN_AGREEMENT_MIN",
    "PLAN_AGREEMENT_MIN_PER_STRATUM",
    "LAMBDA_GRID",
    "PRIMARY_LAMBDA",
    "canonical_hash",
    "default_spec",
    "build_manifest",
    "validate_manifest",
    "frozen_heuristic",
    "prompt_only_heuristic",
    "probe_only_heuristic",
    "utility",
    "select_policy",
    "run_synthetic_utility_smoke",
    "privacy_scan",
    "run_stage_a",
    "analyze_stage_a",
    "validate_gate_report",
    "build_parser",
    "main",
]
