"""72-cell Unbrowser interactive policy grammar generator.

Generates the full factorial of Unbrowser-specific policy factors following
the M3 preregistration grammar (Section 3). Each factor contributes a frozen
description clause to a mechanically composed system prompt.

Factors (3x3x2x2x2 = 72 policies):
    planning:    direct / brief_plan / decompose
    observation: text_first / structure_first / targeted_query_first
    verification: submit_directly / final_reobserve
    recovery:    fail_fast / diagnose_retry_once
    tool_cap:    lean(6) / expanded(12)

All treatments carry a deterministic ``bundle_hash`` (SHA-256) and use the
``native_bash_unbrowser_interactive_v1`` tool interface. ``max_output_tokens``,
``command_timeout``, and ``wall_time_limit`` are held CONSTANT across all 72
policies so only ``tool_call_limit`` varies.

This module creates ``TreatmentSpec`` instances directly (following the
immutable spec pattern from :mod:`pyreplab_harness.treatments`), ensuring the
existing registry and evaluator tooling can consume Unbrowser treatments
without modification.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path
from typing import Any

from .treatments import TreatmentRegistry, TreatmentSpec

# ---------------------------------------------------------------------------
# Frozen Unbrowser grammar factor definitions
# ---------------------------------------------------------------------------

_PLANNING = [
    (
        "direct",
        "Solve the task directly without pre-planning. Do not emit any planning "
        "text before the first tool call; start with the tool immediately.",
    ),
    (
        "brief_plan",
        "Before the first tool call, emit exactly one line beginning 'PLAN:' "
        "with a one-sentence approach, then execute.",
    ),
    (
        "decompose",
        "Before the first tool call, emit at least two decomposition lines "
        "beginning 'STEP 1:' and 'STEP 2:'. Solve the steps independently, "
        "then combine the results.",
    ),
]

_OBSERVATION = [
    (
        "text_first",
        "After navigate, the first successful observation must use the text "
        "action. Use text extraction before inspecting structure.",
    ),
    (
        "structure_first",
        "After navigate, the first successful observation must use the blockmap "
        "action. Inspect structure before extracting text.",
    ),
    (
        "targeted_query_first",
        "After navigate, the first successful observation must use the query "
        "action with a targeted selector. Locate specific information first.",
    ),
]

_VERIFICATION = [
    (
        "submit_directly",
        "Submit directly after obtaining the candidate answer. Do not repeat a "
        "read-only observation before writing result.json.",
    ),
    (
        "final_reobserve",
        "After obtaining the candidate answer and before writing result.json, "
        "repeat a relevant read-only observation to confirm it.",
    ),
]

_RECOVERY = [
    (
        "fail_fast",
        "If a tool call fails or returns no useful information, make no further "
        "tool calls; stop and report the failure.",
    ),
    (
        "diagnose_retry_once",
        "If a tool call fails, diagnose the problem and retry that tool once "
        "with a corrected approach before giving up.",
    ),
]

_TOOL_CAP = [
    ("lean", 6),
    ("expanded", 12),
]

# Frozen constants (NOT grammar factors):
_MAX_OUTPUT_TOKENS = 4096  # constant across all 72
_COMMAND_TIMEOUT = 60      # constant across all 72
_WALL_TIME_LIMIT = 600     # constant across all 72
_SAFETY_SUFFIX = (
    "\n\n---\n"
    "Safety: Always work in the assigned task workspace. "
    "Do not modify system files or access locations outside the allowed URL set. "
    "Clean up any temporary state after completion."
)

_TOOL_INTERFACE = "native_bash_unbrowser_interactive_v1"
_ALLOWED_TOOLS: tuple[str, ...] = ("bash", "unbrowser")

_GRAMMAR_VERSION = "m3-v2"
_GRAMMAR_SIZE = len(_PLANNING) * len(_OBSERVATION) * len(_VERIFICATION) * len(_RECOVERY) * len(_TOOL_CAP)  # 72
_DEFAULT_SPLIT_SEED = 20260810
_DEFAULT_POLICY_VERSION = "2"
_SPLIT_SCHEMA_VERSION = "m3-policy-split-v1"


def _compute_bundle_hash(payload: str) -> str:
    """SHA-256 hex digest of a canonical JSON payload."""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_treatment_payload(treatment: TreatmentSpec) -> str:
    """Deterministic sorted-key JSON of treatment fields (excluding hash)."""
    return json.dumps(
        {
            "id": treatment.id,
            "version": treatment.version,
            "system_prompt": treatment.system_prompt,
            "allowed_tools": sorted(treatment.allowed_tools),
            "max_output_tokens": treatment.max_output_tokens,
            "tool_call_limit": treatment.tool_call_limit,
            "command_timeout_seconds": treatment.command_timeout_seconds,
            "wall_time_limit_seconds": treatment.wall_time_limit_seconds,
            "tool_interface": treatment.tool_interface,
            "generator_metadata": treatment.generator_metadata,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _assemble_system_prompt(combo: dict[str, Any]) -> str:
    """Mechanically assemble a system prompt from frozen factor clauses."""
    prompt_parts = [
        f"Planning: {combo['planning_desc']}",
        f"Observation: {combo['observation_desc']}",
        f"Verification: {combo['verification_desc']}",
        f"Recovery: {combo['recovery_desc']}",
        _SAFETY_SUFFIX,
    ]
    return "\n\n".join(prompt_parts)


def _build_treatment_from_combo(combo: dict[str, Any], index: int, version: str) -> TreatmentSpec:
    """Build one ``TreatmentSpec`` from Unbrowser grammar combination metadata.

    The treatment ID is composed from factor levels deterministically.
    ``bundle_hash`` is computed from the canonical payload, matching the
    mechanism in :mod:`pyreplab_harness.treatments`.
    """
    treatment_id = (
        f"ub-{combo['planning']}-{combo['observation']}-"
        f"{combo['verification']}-{combo['recovery']}-{combo['tool_cap']}"
    )

    system_prompt = _assemble_system_prompt(combo)

    spec = TreatmentSpec(
        id=treatment_id,
        version=version,
        system_prompt=system_prompt,
        allowed_tools=_ALLOWED_TOOLS,
        max_output_tokens=_MAX_OUTPUT_TOKENS,
        tool_call_limit=combo["tool_call_limit_int"],
        command_timeout_seconds=_COMMAND_TIMEOUT,
        wall_time_limit_seconds=_WALL_TIME_LIMIT,
        tool_interface=_TOOL_INTERFACE,
        generator_metadata={
            "grammar_version": _GRAMMAR_VERSION,
            "grammar_size": _GRAMMAR_SIZE,
            "grammar_name": "unbrowser_interactive",
            "index": index,
            "planning": combo["planning"],
            "observation": combo["observation"],
            "verification": combo["verification"],
            "recovery": combo["recovery"],
            "tool_cap": combo["tool_cap"],
        },
    )
    return spec


def _enumerate_combinations() -> list[dict[str, Any]]:
    """Produce every grammar combination in deterministic factor order."""
    combinations: list[dict[str, Any]] = []
    for planning_key, planning_desc in _PLANNING:
        for obs_key, obs_desc in _OBSERVATION:
            for ver_key, ver_desc in _VERIFICATION:
                for rec_key, rec_desc in _RECOVERY:
                    for cap_key, cap_int in _TOOL_CAP:
                        combinations.append({
                            "planning": planning_key,
                            "planning_desc": planning_desc,
                            "observation": obs_key,
                            "observation_desc": obs_desc,
                            "verification": ver_key,
                            "verification_desc": ver_desc,
                            "recovery": rec_key,
                            "recovery_desc": rec_desc,
                            "tool_cap": cap_key,
                            "tool_call_limit_int": cap_int,
                        })
    return combinations


def enumerate_unbrowser_grammar(version: str = "1") -> list[TreatmentSpec]:
    """Return all 72 Unbrowser policy combinations in deterministic order.

    The combination order is fixed: planning > observation > verification >
    recovery > tool_cap. IDs are composed from factor levels.
    """
    combos = _enumerate_combinations()
    return [
        _build_treatment_from_combo(combo, i, version)
        for i, combo in enumerate(combos)
    ]


def generate_unbrowser_treatments(
    count: int,
    seed: int,
    *,
    version: str = "1",
) -> list[TreatmentSpec]:
    """Sample ``count`` treatments without replacement from the 72-cell grammar.

    The grammar is enumerated in deterministic order, then shuffled with
    ``random.Random(seed)``. The first ``count`` entries are returned.
    Calls with the same (count, seed) are reproducible.

    Raises ``ValueError`` when ``count`` exceeds 72.
    """
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    if count > _GRAMMAR_SIZE:
        raise ValueError(
            f"count {count} exceeds Unbrowser grammar size {_GRAMMAR_SIZE}"
        )

    combos = _enumerate_combinations()
    rng = random.Random(seed)
    rng.shuffle(combos)

    return [
        _build_treatment_from_combo(combos[i], i, version)
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# Policy splits: 48 meta-train, 12 development, 12 final held-out
# ---------------------------------------------------------------------------


def _factor_levels_from_treatment(treatment: TreatmentSpec) -> dict[str, str]:
    """Extract grammar factor levels from a treatment's generator_metadata."""
    meta = treatment.generator_metadata
    return {
        "planning": str(meta.get("planning", "")),
        "observation": str(meta.get("observation", "")),
        "verification": str(meta.get("verification", "")),
        "recovery": str(meta.get("recovery", "")),
        "tool_cap": str(meta.get("tool_cap", "")),
    }


def _vrt_from_treatment(treatment: TreatmentSpec) -> int:
    """Encode (verification, recovery, tool_cap) as a 3-bit integer.

    v=MSB, r=middle, t=LSB.  0 = submit_directly / fail_fast / lean.
    """
    levels = _factor_levels_from_treatment(treatment)
    v = 0 if levels["verification"] == "submit_directly" else 1
    r = 0 if levels["recovery"] == "fail_fast" else 1
    t = 0 if levels["tool_cap"] == "lean" else 1
    return (v << 2) | (r << 1) | t


def _vrt_counts(
    vrt_indices: list[int],
    treatments: list[TreatmentSpec],
) -> dict[str, dict[int, int]]:
    """Count v=0, v=1, r=0, r=1, t=0, t=1 across a list of treatment indices."""
    counts: dict[str, dict[int, int]] = {"v": {0: 0, 1: 0}, "r": {0: 0, 1: 0}, "t": {0: 0, 1: 0}}
    for idx in vrt_indices:
        vrt = _vrt_from_treatment(treatments[idx])
        v = (vrt >> 2) & 1
        r = (vrt >> 1) & 1
        t_ = vrt & 1
        counts["v"][v] += 1
        counts["r"][r] += 1
        counts["t"][t_] += 1
    return counts


def split_policies(
    treatments: list[TreatmentSpec],
    seed: int,
) -> tuple[list[TreatmentSpec], list[TreatmentSpec], list[TreatmentSpec]]:
    """Split 72 policies into (meta_train, development, final_held_out).

    Split: 48 meta-train, 12 development, 12 final.

    Uses a deterministic combinatorial construction that guarantees:
    - Every factor level represented in meta-train.
    - Each holdout exactly balanced: 4 per three-level factor, 6 per binary factor.
    - No overlaps between splits.
    - Deterministic for a given seed; different seeds generally differ.

    Algorithm: partition the 72 treatments into 9 buckets (planning x observation),
    each with 8 (verification x recovery x tool_cap) combos.  For each bucket,
    assign a precomputed balanced selection of combos to dev and final using
    a 3x3 diagonal pattern.  The remaining combos go to meta-train.

    Raises ``ValueError`` if the treatment list is not 72 entries.
    """
    if len(treatments) != _GRAMMAR_SIZE:
        raise ValueError(
            f"split_policies requires exactly {_GRAMMAR_SIZE} treatments, "
            f"got {len(treatments)}"
        )

    # --- level-name to index mapping -----------------------------------------
    planning_levels = [p[0] for p in _PLANNING]
    obs_levels = [o[0] for o in _OBSERVATION]

    # --- bucket treatments by (planning_idx, observation_idx) ----------------
    # buckets[(p_idx, o_idx)] = list of (treatment_index, vrt_code)
    buckets: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for idx, t in enumerate(treatments):
        levels = _factor_levels_from_treatment(t)
        p_idx = planning_levels.index(levels["planning"])
        o_idx = obs_levels.index(levels["observation"])
        vrt = _vrt_from_treatment(t)
        buckets.setdefault((p_idx, o_idx), []).append((idx, vrt))

    # --- deterministic seed-based ordering -----------------------------------
    rng = random.Random(seed)
    for bucket_list in buckets.values():
        rng.shuffle(bucket_list)

    # Pick a diagonal pattern (permutation of obs indices for each planning row).
    patterns = list(itertools.permutations(range(3)))
    rng.shuffle(patterns)
    dev_pattern = patterns[0]
    # Use the same pattern for final to guarantee complementary disjointness.
    # (Different patterns would require a more complex non-conflicting search.)

    # --- precomputed balanced (v,r,t) assignments ----------------------------
    # dev: 3 diagonal buckets pick 2 combos each; 6 off-diagonal pick 1 each.
    _DIAG_DEV = [{0b000, 0b111}, {0b001, 0b110}, {0b010, 0b101}]
    _DIAG_FINAL = [{0b001, 0b110}, {0b010, 0b101}, {0b000, 0b111}]
    _OFF_DEV = [0b011, 0b100, 0b111, 0b000, 0b110, 0b001]
    _OFF_FINAL = [0b100, 0b011, 0b000, 0b111, 0b001, 0b110]

    diag_pairs = [(p, dev_pattern[p]) for p in range(3)]
    off_pairs = [(p, o) for p in range(3) for o in range(3) if o != dev_pattern[p]]

    dev_indices: list[int] = []
    final_indices: list[int] = []

    # Diagonal buckets: 2 for dev, 2 for final from each (8 total per bucket).
    for slot, (p, o) in enumerate(diag_pairs):
        bucket = buckets[(p, o)]
        dev_vrts = set(_DIAG_DEV[slot])
        fin_vrts = set(_DIAG_FINAL[slot])
        for idx, vrt in bucket:
            if vrt in dev_vrts:
                dev_indices.append(idx)
                dev_vrts.discard(vrt)
            elif vrt in fin_vrts:
                final_indices.append(idx)
                fin_vrts.discard(vrt)
        if dev_vrts or fin_vrts:
            raise RuntimeError(
                f"bucket ({p},{o}) missing expected vrt combos: "
                f"dev_missing={sorted(dev_vrts)}, final_missing={sorted(fin_vrts)}"
            )

    # Off-diagonal buckets: 1 for dev, 1 for final from each.
    for slot, (p, o) in enumerate(off_pairs):
        bucket = buckets[(p, o)]
        dev_target = _OFF_DEV[slot]
        fin_target = _OFF_FINAL[slot]
        dev_found = False
        fin_found = False
        for idx, vrt in bucket:
            if not dev_found and vrt == dev_target:
                dev_indices.append(idx)
                dev_found = True
            elif not fin_found and vrt == fin_target:
                final_indices.append(idx)
                fin_found = True
        if not dev_found or not fin_found:
            raise RuntimeError(
                f"bucket ({p},{o}) missing off-diagonal vrt combo: "
                f"dev_target={dev_target}, fin_target={fin_target}"
            )

    # --- build meta-train from the remaining 48 indices ----------------------
    used = set(dev_indices) | set(final_indices)
    all_idx = set(range(len(treatments)))
    meta_indices = sorted(all_idx - used)

    # --- verify constraints --------------------------------------------------
    if len(dev_indices) != 12:
        raise RuntimeError(f"dev size {len(dev_indices)} != 12")
    if len(final_indices) != 12:
        raise RuntimeError(f"final size {len(final_indices)} != 12")
    if len(meta_indices) != 48:
        raise RuntimeError(f"meta-train size {len(meta_indices)} != 48")
    if len(set(dev_indices) & set(final_indices)) != 0:
        raise RuntimeError("dev and final have overlapping treatments")
    if len(set(dev_indices) & set(meta_indices)) != 0:
        raise RuntimeError("dev and meta-train have overlapping treatments")
    if len(set(final_indices) & set(meta_indices)) != 0:
        raise RuntimeError("final and meta-train have overlapping treatments")

    # Verify exact factor balance in each holdout.
    for name, indices in [("dev", dev_indices), ("final", final_indices)]:
        counts = _vrt_counts(indices, treatments)
        for dim, expected in [("v", 6), ("r", 6), ("t", 6)]:
            for level in (0, 1):
                actual = counts[dim][level]
                if actual != expected:
                    raise RuntimeError(
                        f"{name} {dim}={level}: count={actual}, expected={expected}"
                    )

    # planning / observation balance (4 each).
    for name, indices in [("dev", dev_indices), ("final", final_indices)]:
        p_counts: dict[str, int] = {}
        o_counts: dict[str, int] = {}
        for idx in indices:
            levels = _factor_levels_from_treatment(treatments[idx])
            p_counts[levels["planning"]] = p_counts.get(levels["planning"], 0) + 1
            o_counts[levels["observation"]] = o_counts.get(levels["observation"], 0) + 1
        for dim_name, counts_dict in [("planning", p_counts), ("observation", o_counts)]:
            for level in counts_dict:
                if counts_dict[level] != 4:
                    raise RuntimeError(
                        f"{name} {dim_name}={level}: count={counts_dict[level]}, expected=4"
                    )

    # Meta-train must cover all factor levels.
    meta_levels: dict[str, set[str]] = {}
    for idx in meta_indices:
        levels = _factor_levels_from_treatment(treatments[idx])
        for f in ("planning", "observation", "verification", "recovery", "tool_cap"):
            meta_levels.setdefault(f, set()).add(levels[f])
    all_levels: dict[str, set[str]] = {}
    for t in treatments:
        levels = _factor_levels_from_treatment(t)
        for f in ("planning", "observation", "verification", "recovery", "tool_cap"):
            all_levels.setdefault(f, set()).add(levels[f])
    for f in all_levels:
        if meta_levels.get(f, set()) != all_levels[f]:
            raise RuntimeError(f"meta-train missing factor levels for {f}: "
                               f"have {sorted(meta_levels.get(f, set()))}, "
                               f"need {sorted(all_levels[f])}")

    # --- assemble output -----------------------------------------------------
    meta_train = [treatments[i] for i in sorted(meta_indices)]
    development = [treatments[i] for i in sorted(dev_indices)]
    final_held = [treatments[i] for i in sorted(final_indices)]

    return meta_train, development, final_held


def build_policy_split_manifest(
    treatments: list[TreatmentSpec],
    seed: int,
    *,
    registry_file: str,
) -> dict[str, Any]:
    """Build a hash-protected manifest for the frozen policy split."""
    registry = TreatmentRegistry(tuple(treatments))
    meta_train, development, final_held = split_policies(treatments, seed)
    payload: dict[str, Any] = {
        "schema_version": _SPLIT_SCHEMA_VERSION,
        "grammar_name": "unbrowser_interactive",
        "grammar_version": _GRAMMAR_VERSION,
        "policy_version": treatments[0].version,
        "registry_file": registry_file,
        "registry_hash": registry.registry_hash,
        "split_algorithm": "balanced-combinatorial-v1",
        "split_seed": seed,
        "splits": {
            "meta_train": [treatment.bundle_id for treatment in meta_train],
            "development": [treatment.bundle_id for treatment in development],
            "final_held_out": [treatment.bundle_id for treatment in final_held],
        },
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return {
        **payload,
        "manifest_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    """Create a frozen JSON artifact, allowing only byte-identical reruns."""
    content = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def freeze_policy_registry(
    registry_path: str | Path,
    manifest_path: str | Path,
    *,
    split_seed: int = _DEFAULT_SPLIT_SEED,
    policy_version: str = _DEFAULT_POLICY_VERSION,
) -> dict[str, Any]:
    """Freeze the full M3 registry and deterministic split manifest."""
    registry_file = Path(registry_path).expanduser().resolve()
    manifest_file = Path(manifest_path).expanduser().resolve()
    treatments = enumerate_unbrowser_grammar(version=policy_version)
    registry = TreatmentRegistry(tuple(treatments))
    manifest = build_policy_split_manifest(
        treatments,
        split_seed,
        registry_file=registry_file.name,
    )
    _write_immutable_json(registry_file, registry.to_dict())
    _write_immutable_json(manifest_file, manifest)
    return {
        "registry": str(registry_file),
        "registry_hash": registry.registry_hash,
        "manifest": str(manifest_file),
        "manifest_hash": manifest["manifest_hash"],
        "split_seed": split_seed,
        "treatments": len(treatments),
    }


def export_grammar_factors(treatment: TreatmentSpec) -> dict[str, Any]:
    """Export one-hot encoded grammar factors for model input.

    Returns a dict with:
    - ``one_hot``: dict mapping factor_name -> one-hot vector
    - ``numeric``: dict with tool_call_limit and derived features
    - ``factor_labels``: original string labels for each factor

    The one-hot encoding is: planning (3), observation (3), verification (2),
    recovery (2), tool_cap (2) = 12 dimensions total.
    """
    meta = treatment.generator_metadata
    planning = str(meta.get("planning", ""))
    observation = str(meta.get("observation", ""))
    verification = str(meta.get("verification", ""))
    recovery = str(meta.get("recovery", ""))
    tool_cap = str(meta.get("tool_cap", ""))

    planning_levels = [p[0] for p in _PLANNING]
    observation_levels = [o[0] for o in _OBSERVATION]
    verification_levels = [v[0] for v in _VERIFICATION]
    recovery_levels = [r[0] for r in _RECOVERY]
    tool_cap_levels = [t[0] for t in _TOOL_CAP]

    return {
        "one_hot": {
            "planning": [1.0 if planning == level else 0.0 for level in planning_levels],
            "observation": [1.0 if observation == level else 0.0 for level in observation_levels],
            "verification": [1.0 if verification == level else 0.0 for level in verification_levels],
            "recovery": [1.0 if recovery == level else 0.0 for level in recovery_levels],
            "tool_cap": [1.0 if tool_cap == level else 0.0 for level in tool_cap_levels],
        },
        "numeric": {
            "tool_call_limit": float(treatment.tool_call_limit),
        },
        "factor_labels": {
            "planning": planning,
            "observation": observation,
            "verification": verification,
            "recovery": recovery,
            "tool_cap": tool_cap,
        },
    }


def grammar_factor_vector(treatment: TreatmentSpec) -> list[float]:
    """Return a flat 13-dimensional vector: 12 one-hot + 1 numeric tool_cap.

    The order is: planning[3] + observation[3] + verification[2] +
    recovery[2] + tool_cap[2] + tool_call_limit_normalized.
    """
    exported = export_grammar_factors(treatment)
    vec: list[float] = []
    for factor in ("planning", "observation", "verification", "recovery", "tool_cap"):
        vec.extend(exported["one_hot"][factor])
    # Normalize tool_call_limit to [0, 1] range (6 -> 0.0, 12 -> 1.0).
    vec.append((exported["numeric"]["tool_call_limit"] - 6.0) / 6.0)
    return vec


__all__ = [
    "enumerate_unbrowser_grammar",
    "generate_unbrowser_treatments",
    "split_policies",
    "build_policy_split_manifest",
    "freeze_policy_registry",
    "export_grammar_factors",
    "grammar_factor_vector",
    "_GRAMMAR_SIZE",
    "_GRAMMAR_VERSION",
    "_DEFAULT_SPLIT_SEED",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze the M3 Unbrowser treatment registry and policy split."
    )
    parser.add_argument("--registry", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split-seed", type=int, default=_DEFAULT_SPLIT_SEED)
    parser.add_argument("--policy-version", default=_DEFAULT_POLICY_VERSION)
    args = parser.parse_args(argv)
    report = freeze_policy_registry(
        args.registry,
        args.manifest,
        split_seed=args.split_seed,
        policy_version=args.policy_version,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
