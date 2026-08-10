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

import hashlib
import json
import random
from typing import Any

from .treatments import TreatmentSpec

# ---------------------------------------------------------------------------
# Frozen Unbrowser grammar factor definitions
# ---------------------------------------------------------------------------

_PLANNING = [
    (
        "direct",
        "Solve the task directly without pre-planning. Start working immediately.",
    ),
    (
        "brief_plan",
        "Before acting, briefly think through the approach. State your plan "
        "in one sentence, then execute.",
    ),
    (
        "decompose",
        "Break the task into sub-problems. Solve each independently, then "
        "combine the results into a final answer.",
    ),
]

_OBSERVATION = [
    (
        "text_first",
        "Start by reading the page text first. Use text extraction before "
        "inspecting the page structure.",
    ),
    (
        "structure_first",
        "Start by inspecting the page structure first. Understand the DOM "
        "layout before extracting text.",
    ),
    (
        "targeted_query_first",
        "Start by running a targeted query or search first. Locate the "
        "specific information you need before exploring the full page.",
    ),
]

_VERIFICATION = [
    (
        "submit_directly",
        "Submit the answer directly after arriving at a result. Do not "
        "re-observe the page before submitting.",
    ),
    (
        "final_reobserve",
        "After arriving at a result and before submitting, re-observe the "
        "relevant page state to confirm the answer is still correct.",
    ),
]

_RECOVERY = [
    (
        "fail_fast",
        "If a step fails or returns no useful information, stop the attempt "
        "and report the failure. Do not retry.",
    ),
    (
        "diagnose_retry_once",
        "If a step fails, diagnose the problem and retry once with a "
        "corrected approach before giving up.",
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

_GRAMMAR_VERSION = "m3-v1"
_GRAMMAR_SIZE = len(_PLANNING) * len(_OBSERVATION) * len(_VERIFICATION) * len(_RECOVERY) * len(_TOOL_CAP)  # 72


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


def _check_factor_coverage(
    treatments: list[TreatmentSpec],
    factors: list[str],
    required_per_level: dict[tuple[str, str], int],
) -> dict[str, dict[str, int]]:
    """Count factor level occurrences in a treatment list."""
    counts: dict[str, dict[str, int]] = {f: {} for f in factors}
    for t in treatments:
        levels = _factor_levels_from_treatment(t)
        for f in factors:
            level = levels[f]
            counts[f][level] = counts[f].get(level, 0) + 1
    return counts


def split_policies(
    treatments: list[TreatmentSpec],
    seed: int,
) -> tuple[list[TreatmentSpec], list[TreatmentSpec], list[TreatmentSpec]]:
    """Split 72 policies into (meta_train, development, final_held_out).

    Split: 48 meta-train, 12 development, 12 final.

    Constraints:
    - Every factor level represented in meta-train.
    - Each holdout balanced: 4 per three-level factor, 6 per binary factor.
    - Deterministic given seed.

    Raises ``ValueError`` if the treatment list is not 72 entries or if
    factor coverage cannot be satisfied.
    """
    if len(treatments) != _GRAMMAR_SIZE:
        raise ValueError(
            f"split_policies requires exactly {_GRAMMAR_SIZE} treatments, "
            f"got {len(treatments)}"
        )

    factors = ["planning", "observation", "verification", "recovery", "tool_cap"]
    factor_arity = {
        "planning": 3, "observation": 3,
        "verification": 2, "recovery": 2, "tool_cap": 2,
    }

    rng = random.Random(seed)
    indices = list(range(len(treatments)))
    rng.shuffle(indices)

    # Greedy meta-train: ensure every factor level is covered.
    meta_train_idx: set[int] = set()
    covered: set[tuple[str, str]] = set()
    for idx in indices:
        if len(meta_train_idx) >= 48:
            break
        t = treatments[idx]
        levels = _factor_levels_from_treatment(t)
        for f in factors:
            covered.add((f, levels[f]))
        meta_train_idx.add(idx)

    # If coverage incomplete after 48, it's a bug in the grammar definition.
    all_levels: set[tuple[str, str]] = set()
    for f in factors:
        for level in set(
            _factor_levels_from_treatment(t)[f] for t in treatments
        ):
            all_levels.add((f, level))
    if not covered >= all_levels:
        raise RuntimeError(
            f"factor coverage not satisfied after greedy meta-train selection: "
            f"covered {len(covered)}/{len(all_levels)}"
        )

    # Fill to exactly 48 meta-train.
    for idx in indices:
        if len(meta_train_idx) >= 48:
            break
        if idx not in meta_train_idx:
            meta_train_idx.add(idx)

    remaining = [i for i in indices if i not in meta_train_idx]

    # Split remaining 24 into 12 dev + 12 final with balanced coverage.
    dev_idx: set[int] = set()
    final_idx: set[int] = set()

    # Greedy: alternate dev/final, trying to maintain balance.
    dev_counts: dict[str, dict[str, int]] = {f: {} for f in factors}
    final_counts: dict[str, dict[str, int]] = {f: {} for f in factors}

    for i in remaining:
        t = treatments[i]
        levels = _factor_levels_from_treatment(t)

        # Compute which side would improve balance more.
        def _imbalance(counts: dict[str, dict[str, int]]) -> float:
            score = 0.0
            for f in factors:
                target = factor_arity[f]
                for level in set(
                    _factor_levels_from_treatment(tt)[f] for tt in treatments
                ):
                    current = counts[f].get(level, 0)
                    score += (current / max(1, sum(counts[f].values()))) if sum(counts[f].values()) else 0
                # Perfect balance for arity A means each level has 12*arity/A
                # For 12 policies: 3-level -> 4 each, 2-level -> 6 each.
                target_each = 12 // target
                for level in set(
                    _factor_levels_from_treatment(tt)[f] for tt in treatments
                ):
                    current = counts[f].get(level, 0)
                    score += abs(current - target_each)
            return score

        dev_imbalance_before = _imbalance(dev_counts)
        final_imbalance_before = _imbalance(final_counts)

        # Prefer adding to the side with more capacity and less imbalance.
        if len(dev_idx) < 12 and len(final_idx) < 12:
            if dev_imbalance_before <= final_imbalance_before:
                dev_idx.add(i)
                for f in factors:
                    dev_counts[f][levels[f]] = dev_counts[f].get(levels[f], 0) + 1
            else:
                final_idx.add(i)
                for f in factors:
                    final_counts[f][levels[f]] = final_counts[f].get(levels[f], 0) + 1
        elif len(dev_idx) < 12:
            dev_idx.add(i)
            for f in factors:
                dev_counts[f][levels[f]] = dev_counts[f].get(levels[f], 0) + 1
        elif len(final_idx) < 12:
            final_idx.add(i)
            for f in factors:
                final_counts[f][levels[f]] = final_counts[f].get(levels[f], 0) + 1

    if len(dev_idx) != 12 or len(final_idx) != 12 or len(meta_train_idx) != 48:
        raise RuntimeError(
            f"split sizes incorrect: meta_train={len(meta_train_idx)}, "
            f"dev={len(dev_idx)}, final={len(final_idx)}"
        )

    meta_train = [treatments[i] for i in sorted(meta_train_idx)]
    development = [treatments[i] for i in sorted(dev_idx)]
    final_held = [treatments[i] for i in sorted(final_idx)]

    return meta_train, development, final_held


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
    "export_grammar_factors",
    "grammar_factor_vector",
    "_GRAMMAR_SIZE",
    "_GRAMMAR_VERSION",
]
