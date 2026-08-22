"""Fail-closed cache observability and Stage 1 invariance comparison.

The probe in this module is GET-only and never invokes a model. Provider usage
counters are retained as provider observations; they are not promoted to
server-verified prompt-cache mechanics without separate instrumentation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .m3_pilot import _ssh_capture, _write_immutable_json

CACHE_RUNTIME_RECEIPT_SCHEMA_VERSION = "pyreplab-cache-runtime-receipt-v1"
PROVIDER_TURN_CACHE_RECEIPT_SCHEMA_VERSION = (
    "pyreplab-provider-turn-cache-receipt-v1"
)
CACHE_CANARY_CELL_SCHEMA_VERSION = "pyreplab-cache-canary-cell-v1"
CACHE_CANARY_REPORT_SCHEMA_VERSION = "pyreplab-cache-canary-report-v1"

_CACHE_ARGUMENTS: dict[str, tuple[tuple[str, Any], ...]] = {
    "prompt_cache": (("--cache-prompt", True), ("--no-cache-prompt", False)),
    "cache_type_k": (("--cache-type-k", "value"), ("-ctk", "value")),
    "cache_type_v": (("--cache-type-v", "value"), ("-ctv", "value")),
    "cache_ram_mib": (("--cache-ram", "integer"), ("-cram", "integer")),
    "ctx_checkpoints": (
        ("--ctx-checkpoints", "integer"),
        ("--swa-checkpoints", "integer"),
        ("-ctxcp", "integer"),
    ),
    "checkpoint_min_step": (
        ("--checkpoint-min-step", "integer"),
        ("-cms", "integer"),
    ),
    "cache_idle_slots": (
        ("--cache-idle-slots", True),
        ("--no-cache-idle-slots", False),
    ),
    "cache_reuse": (("--cache-reuse", "integer"),),
    "kv_unified": (
        ("--kv-unified", True),
        ("-kvu", True),
        ("--no-kv-unified", False),
        ("-no-kvu", False),
    ),
    "metrics": (("--metrics", True),),
    "slots": (("--slots", True), ("--no-slots", False)),
    "slot_save_path": (("--slot-save-path", "value"),),
    "sleep_idle_seconds": (("--sleep-idle-seconds", "integer"),),
    "parallel": (("--parallel", "integer"), ("-np", "integer")),
    "ctx_size": (("--ctx-size", "integer"), ("-c", "integer")),
}

_DOCUMENTED_DEFAULTS: dict[str, Any] = {
    "prompt_cache": True,
    "cache_type_k": "f16",
    "cache_type_v": "f16",
    "cache_ram_mib": 8192,
    "ctx_checkpoints": 32,
    "checkpoint_min_step": 8192,
    "cache_idle_slots": True,
    "cache_reuse": 0,
    "metrics": False,
    "slots": True,
    "slot_save_path": "disabled",
    "sleep_idle_seconds": -1,
    "parallel": -1,
}


def canonical_receipt_hash(value: Any) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _with_hash(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {**payload, field: canonical_receipt_hash(payload)}


def _verify_hash(value: Mapping[str, Any], field: str) -> None:
    observed = value.get(field)
    if not isinstance(observed, str) or len(observed) != 64:
        raise ValueError(f"{field} must be a SHA-256 hex digest")
    payload = {key: item for key, item in value.items() if key != field}
    if canonical_receipt_hash(payload) != observed:
        raise ValueError(f"{field} mismatch")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _argument_occurrences(argv: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    aliases = {
        alias: (name, kind)
        for name, values in _CACHE_ARGUMENTS.items()
        for alias, kind in values
    }
    occurrences: dict[str, list[dict[str, Any]]] = {
        name: [] for name in _CACHE_ARGUMENTS
    }
    index = 0
    while index < len(argv):
        token = argv[index]
        alias = token
        inline_value: str | None = None
        if token.startswith("--") and "=" in token:
            alias, inline_value = token.split("=", 1)
        spec = aliases.get(alias)
        if spec is None:
            index += 1
            continue
        name, kind = spec
        if kind in {True, False}:
            if inline_value is not None:
                raise ValueError(f"boolean cache argument cannot take a value: {token}")
            value = kind
        else:
            if inline_value is None:
                if index + 1 >= len(argv):
                    raise ValueError(f"cache argument requires a value: {token}")
                inline_value = argv[index + 1]
                index += 1
            if kind == "integer":
                try:
                    value = int(inline_value)
                except ValueError as error:
                    raise ValueError(
                        f"cache argument requires an integer: {token}"
                    ) from error
            else:
                value = inline_value
        occurrences[name].append({"argument": alias, "value": value})
        index += 1
    return occurrences


def parse_cache_launch_configuration(
    argv: Sequence[str], help_text: str
) -> dict[str, Any]:
    """Parse explicit cache controls without treating documented defaults as pins."""
    if not argv or any(not isinstance(value, str) for value in argv):
        raise ValueError("server argv must be a non-empty string list")
    occurrences = _argument_occurrences(argv)
    fields: dict[str, Any] = {}
    invalidation_codes: list[str] = []
    for name, values in occurrences.items():
        distinct = {canonical_receipt_hash(item["value"]) for item in values}
        if len(values) > 1:
            invalidation_codes.append(f"duplicate_cache_argument:{name}")
        if len(distinct) > 1:
            invalidation_codes.append(f"conflicting_cache_argument:{name}")
        aliases = [alias for alias, _ in _CACHE_ARGUMENTS[name]]
        supported = any(
            re.search(rf"(?<!\S){re.escape(alias)}(?=[\s,=]|$)", help_text)
            is not None
            for alias in aliases
        )
        fields[name] = {
            "explicit": len(values) == 1,
            "value": values[0]["value"] if len(values) == 1 else None,
            "occurrences": values,
            "documented_default": _DOCUMENTED_DEFAULTS.get(name),
            "supported_by_help": supported,
        }
        if not values:
            invalidation_codes.append(f"implicit_cache_argument:{name}")
        if not supported:
            invalidation_codes.append(f"cache_argument_not_in_help:{name}")
    return {
        "fields": fields,
        "all_cache_fields_explicit": all(
            field["explicit"] for field in fields.values()
        ),
        "invalidation_codes": sorted(set(invalidation_codes)),
    }


def _endpoint_classification(
    observation: Mapping[str, Any], model_state: str | None
) -> dict[str, Any]:
    status = observation.get("http_status")
    if status == 200:
        classification = "available"
    elif status == 400 and model_state == "sleeping":
        classification = "blocked_while_sleeping"
    elif status is None:
        classification = "unreachable"
    else:
        classification = "unavailable"
    return {
        "status": classification,
        "http_status": status,
        "response_bytes": observation.get("response_bytes"),
        "response_sha256": observation.get("response_sha256"),
    }


def build_cache_runtime_receipt(observation: Mapping[str, Any]) -> dict[str, Any]:
    argv = observation.get("server_argv")
    help_text = observation.get("server_help")
    if not isinstance(argv, list) or not isinstance(help_text, str):
        raise ValueError("probe observation requires server_argv and server_help")
    launch = parse_cache_launch_configuration(argv, help_text)
    fields = launch["fields"]
    prompt_field = fields["prompt_cache"]
    cache_mode = (
        "on"
        if prompt_field["explicit"] and prompt_field["value"] is True
        else "off"
        if prompt_field["explicit"] and prompt_field["value"] is False
        else "unresolved"
    )
    model_state = observation.get("model_state")
    endpoints = {
        name: _endpoint_classification(
            observation.get(f"{name}_endpoint", {}),
            model_state if isinstance(model_state, str) else None,
        )
        for name in ("metrics", "slots")
    }
    runtime = {
        "pi_version": observation.get("pi_version"),
        "pi_sha256": observation.get("pi_sha256"),
        "llama_server_version": observation.get("llama_server_version"),
        "llama_server_sha256": observation.get("llama_server_sha256"),
        "model_sha256": observation.get("model_sha256"),
        "model_state": model_state,
        "tokenizer_identity": {
            "status": "unobservable",
            "reason": "router_model_status_omits_tokenizer_hash",
        },
        "chat_template_identity": {
            "status": "unobservable",
            "reason": "router_model_status_omits_chat_template_hash",
        },
    }
    capabilities = {
        "exact_request_bytes": {
            "status": "unobservable",
            "reason": "pi_event_contract_does_not_expose_request_bytes",
        },
        "provider_cache_usage": {
            "status": "observed",
            "semantics": "provider_reported_usage_not_server_verified_reuse",
        },
        "reused_prefix_tokens": {
            "status": "unobservable",
            "reason": "cacheRead_is_not_a_server_mechanics_receipt",
        },
        "prompt_evaluation_duration": {
            "status": "unobservable",
            "reason": "provider_events_omit_prompt_evaluation_timing",
        },
        "generation_duration": {
            "status": "unobservable",
            "reason": "provider_events_omit_generation_timing",
        },
        "slot_identity": {
            "status": "unobservable",
            "reason": "provider_events_omit_slot_identity",
        },
    }
    invalidation_codes = list(launch["invalidation_codes"])
    if not launch["all_cache_fields_explicit"]:
        invalidation_codes.append("cache_configuration_uses_implicit_defaults")
    for name, endpoint in endpoints.items():
        if endpoint["status"] != "available":
            invalidation_codes.append(f"{name}_endpoint_{endpoint['status']}")

    common_configuration = {
        "runtime": runtime,
        "launch_fields": {
            name: value for name, value in fields.items() if name != "prompt_cache"
        },
        "help_sha256": canonical_receipt_hash(help_text),
    }
    cell_configuration = {
        **common_configuration,
        "prompt_cache": prompt_field,
    }
    payload: dict[str, Any] = {
        "schema_version": CACHE_RUNTIME_RECEIPT_SCHEMA_VERSION,
        "checked_at": observation.get("checked_at")
        or datetime.now(timezone.utc).isoformat(),
        "probe_mode": "no_model_get_only",
        "cache_mode": cache_mode,
        "common_config_hash": canonical_receipt_hash(common_configuration),
        "cell_config_hash": canonical_receipt_hash(cell_configuration),
        "runtime": runtime,
        "launch": {
            "argv_sha256": canonical_receipt_hash(argv),
            "help_sha256": canonical_receipt_hash(help_text),
            **launch,
        },
        "endpoints": endpoints,
        "capabilities": capabilities,
        "sensitive_artifact_policy": {
            "raw_prompts_in_receipt": False,
            "raw_kv_persistence": False,
            "retention": "local_excluded_experiment_artifacts",
        },
        "acceptance": {
            "eligible_for_canary": not invalidation_codes,
            "invalidation_codes": sorted(set(invalidation_codes)),
        },
    }
    return _with_hash(payload, "receipt_hash")


def validate_cache_runtime_receipt(receipt: Mapping[str, Any]) -> None:
    _verify_hash(receipt, "receipt_hash")
    if receipt.get("schema_version") != CACHE_RUNTIME_RECEIPT_SCHEMA_VERSION:
        raise ValueError("unsupported cache runtime receipt schema")
    if receipt.get("probe_mode") != "no_model_get_only":
        raise ValueError("cache runtime probe must be no-model GET-only")
    if receipt.get("cache_mode") not in {"on", "off", "unresolved"}:
        raise ValueError("invalid cache mode")
    acceptance = receipt.get("acceptance")
    if not isinstance(acceptance, Mapping):
        raise ValueError("cache runtime receipt acceptance must be an object")
    codes = acceptance.get("invalidation_codes")
    if not isinstance(codes, list) or any(not isinstance(code, str) for code in codes):
        raise ValueError("cache runtime invalidation codes must be strings")
    if acceptance.get("eligible_for_canary") is not (not codes):
        raise ValueError("cache runtime eligibility contradicts invalidation codes")
    launch = receipt.get("launch")
    runtime = receipt.get("runtime")
    if not isinstance(launch, Mapping) or not isinstance(runtime, Mapping):
        raise ValueError("cache runtime launch and runtime must be objects")
    fields = launch.get("fields")
    if not isinstance(fields, Mapping) or "prompt_cache" not in fields:
        raise ValueError("cache runtime launch fields are incomplete")
    common_configuration = {
        "runtime": dict(runtime),
        "launch_fields": {
            name: value for name, value in fields.items() if name != "prompt_cache"
        },
        "help_sha256": launch.get("help_sha256"),
    }
    cell_configuration = {
        **common_configuration,
        "prompt_cache": fields["prompt_cache"],
    }
    if receipt.get("common_config_hash") != canonical_receipt_hash(
        common_configuration
    ):
        raise ValueError("cache runtime common configuration hash mismatch")
    if receipt.get("cell_config_hash") != canonical_receipt_hash(cell_configuration):
        raise ValueError("cache runtime cell configuration hash mismatch")
    prompt_field = fields["prompt_cache"]
    if not isinstance(prompt_field, Mapping):
        raise ValueError("prompt-cache field must be an object")
    expected_mode = (
        "on"
        if prompt_field.get("explicit") is True and prompt_field.get("value") is True
        else "off"
        if prompt_field.get("explicit") is True and prompt_field.get("value") is False
        else "unresolved"
    )
    if receipt.get("cache_mode") != expected_mode:
        raise ValueError("cache mode contradicts the launch configuration")


def _observed(value: Any) -> dict[str, Any]:
    return {"status": "observed", "value": value}


def _unobservable(reason: str) -> dict[str, Any]:
    return {"status": "unobservable", "reason": reason}


def _mechanics_observation(
    mechanics: Mapping[str, Any],
    key: str,
    *,
    validator: Any,
    reason: str,
) -> dict[str, Any]:
    value = mechanics.get(key)
    return _observed(value) if validator(value) else _unobservable(reason)


def build_provider_turn_cache_receipt(
    provider_turn: Mapping[str, Any],
    *,
    attempt_id: str,
    panel_id: str,
    pair_id: str,
    sampling_seed: int,
    cache_runtime_receipt: Mapping[str, Any],
    mechanics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_cache_runtime_receipt(cache_runtime_receipt)
    mechanics = mechanics or {}
    usage = provider_turn.get("usage")
    if not isinstance(usage, Mapping):
        raise ValueError("provider turn usage must be an object")
    assistant_hash = provider_turn.get("assistant_content_sha256")
    if not isinstance(assistant_hash, str) or len(assistant_hash) != 64:
        raise ValueError("provider turn assistant hash is invalid")

    exact_request = _mechanics_observation(
        mechanics,
        "exact_serialized_request_sha256",
        validator=lambda value: isinstance(value, str) and len(value) == 64,
        reason="pi_event_contract_does_not_expose_request_bytes",
    )
    reused = _mechanics_observation(
        mechanics,
        "reused_prefix_tokens",
        validator=lambda value: isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0,
        reason="server_reused_prefix_tokens_unobservable",
    )
    prompt_seconds = _mechanics_observation(
        mechanics,
        "prompt_evaluation_seconds",
        validator=lambda value: isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value >= 0,
        reason="server_prompt_evaluation_timing_unobservable",
    )
    generation_seconds = _mechanics_observation(
        mechanics,
        "generation_seconds",
        validator=lambda value: isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value >= 0,
        reason="server_generation_timing_unobservable",
    )
    slot_identity = _mechanics_observation(
        mechanics,
        "slot_identity",
        validator=lambda value: isinstance(value, (str, int))
        and not isinstance(value, bool),
        reason="server_slot_identity_unobservable",
    )
    server_prompt_tokens = _mechanics_observation(
        mechanics,
        "server_prompt_tokens",
        validator=lambda value: isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0,
        reason="server_prompt_token_count_unobservable",
    )
    server_predicted_tokens = _mechanics_observation(
        mechanics,
        "server_predicted_tokens",
        validator=lambda value: isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0,
        reason="server_predicted_token_count_unobservable",
    )
    observations = {
        "exact_serialized_request_sha256": exact_request,
        "reused_prefix_tokens": reused,
        "prompt_evaluation_seconds": prompt_seconds,
        "generation_seconds": generation_seconds,
        "slot_identity": slot_identity,
        "server_prompt_tokens": server_prompt_tokens,
        "server_predicted_tokens": server_predicted_tokens,
    }
    invalidation_codes = [
        f"{name}_unobservable"
        for name, observation in observations.items()
        if observation["status"] != "observed"
    ]
    if usage.get("complete") is not True:
        invalidation_codes.append("provider_usage_incomplete")
    runtime_acceptance = cache_runtime_receipt["acceptance"]
    if runtime_acceptance["eligible_for_canary"] is not True:
        invalidation_codes.append("cache_runtime_ineligible")
    cache_read = usage.get("cache_read")
    provider_cache_observation = (
        _observed(cache_read)
        if isinstance(cache_read, int) and not isinstance(cache_read, bool)
        else _unobservable("provider_cacheRead_missing")
    )
    payload: dict[str, Any] = {
        "schema_version": PROVIDER_TURN_CACHE_RECEIPT_SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "panel_id": panel_id,
        "pair_id": pair_id,
        "provider_turn": provider_turn.get("turn_index"),
        "sampling_seed": sampling_seed,
        "cache_runtime_receipt_hash": cache_runtime_receipt["receipt_hash"],
        "provider": provider_turn.get("provider"),
        "model": provider_turn.get("model"),
        "provider_usage": dict(usage),
        "provider_reported_cache_read_tokens": provider_cache_observation,
        **observations,
        "assistant_content_sha256": assistant_hash,
        "schema_valid": True,
        "mechanics_valid": not invalidation_codes,
        "invalidation_codes": sorted(set(invalidation_codes)),
    }
    return _with_hash(payload, "receipt_hash")


def validate_provider_turn_cache_receipt(receipt: Mapping[str, Any]) -> None:
    _verify_hash(receipt, "receipt_hash")
    if receipt.get("schema_version") != PROVIDER_TURN_CACHE_RECEIPT_SCHEMA_VERSION:
        raise ValueError("unsupported provider-turn cache receipt schema")
    if receipt.get("schema_valid") is not True:
        raise ValueError("provider-turn cache receipt is structurally invalid")
    turn = receipt.get("provider_turn")
    if not isinstance(turn, int) or isinstance(turn, bool) or turn < 1:
        raise ValueError("provider turn index must be positive")
    codes = receipt.get("invalidation_codes")
    if not isinstance(codes, list) or any(not isinstance(code, str) for code in codes):
        raise ValueError("provider-turn invalidation codes must be strings")
    if receipt.get("mechanics_valid") is not (not codes):
        raise ValueError("provider-turn mechanics validity contradicts invalidations")


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _validate_canary_cell(cell: Mapping[str, Any], expected_mode: str) -> None:
    _verify_hash(cell, "cell_hash")
    if cell.get("schema_version") != CACHE_CANARY_CELL_SCHEMA_VERSION:
        raise ValueError("unsupported cache canary cell schema")
    if cell.get("cache_mode") != expected_mode:
        raise ValueError(f"expected cache-{expected_mode} cell")
    runtime = cell.get("runtime_receipt")
    if not isinstance(runtime, Mapping):
        raise ValueError("canary cell runtime receipt must be an object")
    validate_cache_runtime_receipt(runtime)
    if runtime.get("cache_mode") != expected_mode:
        raise ValueError("canary cell and runtime cache modes differ")
    attempts = cell.get("attempts")
    if not isinstance(attempts, list) or any(
        not isinstance(attempt, Mapping) for attempt in attempts
    ):
        raise ValueError("canary attempts must be objects")


def _status(errors: list[str]) -> dict[str, Any]:
    return {"passed": not errors, "errors": sorted(set(errors))}


def _observed_value(receipt: Mapping[str, Any], field: str) -> Any | None:
    observation = receipt.get(field)
    if not isinstance(observation, Mapping) or observation.get("status") != "observed":
        return None
    return observation.get("value")


def compare_cache_cells(
    cache_off: Mapping[str, Any],
    cache_on: Mapping[str, Any],
    pair_order: Sequence[str],
    *,
    minimum_prompt_savings_fraction: float = 0.0,
) -> dict[str, Any]:
    """Compare frozen cache cells without weakening missing-evidence failures."""
    _validate_canary_cell(cache_off, "off")
    _validate_canary_cell(cache_on, "on")
    if not pair_order or any(not isinstance(pair_id, str) for pair_id in pair_order):
        raise ValueError("pair_order must be a non-empty string list")
    if len(set(pair_order)) != len(pair_order):
        raise ValueError("pair_order must not contain duplicates")
    if (
        not isinstance(minimum_prompt_savings_fraction, (int, float))
        or isinstance(minimum_prompt_savings_fraction, bool)
        or minimum_prompt_savings_fraction < 0
        or minimum_prompt_savings_fraction >= 1
    ):
        raise ValueError("minimum prompt savings fraction must be in [0, 1)")

    input_errors: list[str] = []
    sampling_errors: list[str] = []
    behavior_errors: list[str] = []
    mechanics_errors: list[str] = []
    performance_errors: list[str] = []
    off_attempts = cache_off["attempts"]
    on_attempts = cache_on["attempts"]
    off_order = [attempt.get("pair_id") for attempt in off_attempts]
    on_order = [attempt.get("pair_id") for attempt in on_attempts]
    expected_order = list(pair_order)
    if off_order != expected_order:
        input_errors.append("cache_off_manifest_order_mismatch")
    if on_order != expected_order:
        input_errors.append("cache_on_manifest_order_mismatch")

    off_runtime = cache_off["runtime_receipt"]
    on_runtime = cache_on["runtime_receipt"]
    if off_runtime["common_config_hash"] != on_runtime["common_config_hash"]:
        input_errors.append("runtime_common_configuration_mismatch")
    if off_runtime["acceptance"]["eligible_for_canary"] is not True:
        mechanics_errors.append("cache_off_runtime_ineligible")
    if on_runtime["acceptance"]["eligible_for_canary"] is not True:
        mechanics_errors.append("cache_on_runtime_ineligible")

    total_off_prompt_seconds = 0.0
    total_on_prompt_seconds = 0.0
    observed_on_reused_tokens = 0
    compared_turns = 0
    for index, pair_id in enumerate(expected_order):
        if index >= len(off_attempts) or index >= len(on_attempts):
            input_errors.append(f"missing_pair:{pair_id}")
            continue
        off_attempt = off_attempts[index]
        on_attempt = on_attempts[index]
        if off_attempt.get("pair_id") != pair_id or on_attempt.get("pair_id") != pair_id:
            continue
        off_sampling = off_attempt.get("sampling_receipt")
        on_sampling = on_attempt.get("sampling_receipt")
        if not isinstance(off_sampling, Mapping) or not isinstance(
            on_sampling, Mapping
        ):
            sampling_errors.append(f"sampling_receipt_missing:{pair_id}")
        elif off_sampling != on_sampling:
            sampling_errors.append(f"sampling_receipt_mismatch:{pair_id}")
        for field in (
            "final_output_sha256",
            "tool_trajectory_sha256",
            "verifier_result_sha256",
        ):
            off_value = off_attempt.get(field)
            on_value = on_attempt.get(field)
            if (
                not isinstance(off_value, str)
                or len(off_value) != 64
                or not isinstance(on_value, str)
                or len(on_value) != 64
            ):
                behavior_errors.append(f"{field}_missing:{pair_id}")
            elif off_value != on_value:
                behavior_errors.append(f"{field}_mismatch:{pair_id}")

        off_turns = off_attempt.get("provider_turn_receipts")
        on_turns = on_attempt.get("provider_turn_receipts")
        if not isinstance(off_turns, list) or not isinstance(on_turns, list):
            mechanics_errors.append(f"provider_turn_receipts_missing:{pair_id}")
            continue
        if len(off_turns) != len(on_turns):
            input_errors.append(f"provider_turn_count_mismatch:{pair_id}")
            continue
        for turn_index, (off_turn, on_turn) in enumerate(
            zip(off_turns, on_turns, strict=True), start=1
        ):
            validate_provider_turn_cache_receipt(off_turn)
            validate_provider_turn_cache_receipt(on_turn)
            if (
                off_turn.get("provider_turn") != turn_index
                or on_turn.get("provider_turn") != turn_index
            ):
                mechanics_errors.append(
                    f"provider_turn_correlation_mismatch:{pair_id}:{turn_index}"
                )
            if off_turn.get("pair_id") != pair_id or on_turn.get("pair_id") != pair_id:
                input_errors.append(f"provider_turn_pair_mismatch:{pair_id}:{turn_index}")
            if isinstance(off_sampling, Mapping) and isinstance(on_sampling, Mapping):
                expected_seed = off_sampling.get("seed")
                if (
                    off_turn.get("sampling_seed") != expected_seed
                    or on_turn.get("sampling_seed") != expected_seed
                ):
                    sampling_errors.append(
                        f"provider_turn_sampling_seed_mismatch:{pair_id}:{turn_index}"
                    )
            if off_turn["cache_runtime_receipt_hash"] != off_runtime["receipt_hash"]:
                mechanics_errors.append(
                    f"cache_off_runtime_receipt_mismatch:{pair_id}:{turn_index}"
                )
            if on_turn["cache_runtime_receipt_hash"] != on_runtime["receipt_hash"]:
                mechanics_errors.append(
                    f"cache_on_runtime_receipt_mismatch:{pair_id}:{turn_index}"
                )
            off_request = _observed_value(
                off_turn, "exact_serialized_request_sha256"
            )
            on_request = _observed_value(on_turn, "exact_serialized_request_sha256")
            if off_request is None or on_request is None:
                input_errors.append(f"request_hash_unobservable:{pair_id}:{turn_index}")
            elif off_request != on_request:
                input_errors.append(f"request_hash_mismatch:{pair_id}:{turn_index}")
            if (
                off_turn.get("assistant_content_sha256")
                != on_turn.get("assistant_content_sha256")
            ):
                behavior_errors.append(
                    f"assistant_content_mismatch:{pair_id}:{turn_index}"
                )
            off_usage = off_turn.get("provider_usage", {})
            on_usage = on_turn.get("provider_usage", {})
            for field in ("logical_prompt_tokens", "output", "total_tokens"):
                if off_usage.get(field) != on_usage.get(field):
                    input_errors.append(
                        f"provider_usage_{field}_mismatch:{pair_id}:{turn_index}"
                    )
            if off_turn.get("mechanics_valid") is not True:
                mechanics_errors.append(
                    f"cache_off_mechanics_invalid:{pair_id}:{turn_index}"
                )
            if on_turn.get("mechanics_valid") is not True:
                mechanics_errors.append(
                    f"cache_on_mechanics_invalid:{pair_id}:{turn_index}"
                )
            off_reused = _observed_value(off_turn, "reused_prefix_tokens")
            on_reused = _observed_value(on_turn, "reused_prefix_tokens")
            if off_reused != 0:
                mechanics_errors.append(
                    f"cache_off_reused_prefix_nonzero_or_unobservable:{pair_id}:{turn_index}"
                )
            if isinstance(on_reused, int):
                observed_on_reused_tokens += on_reused
            else:
                mechanics_errors.append(
                    f"cache_on_reused_prefix_unobservable:{pair_id}:{turn_index}"
                )
            if isinstance(off_reused, int) and off_usage.get("cache_read") != off_reused:
                mechanics_errors.append(
                    f"cache_off_provider_server_reuse_mismatch:{pair_id}:{turn_index}"
                )
            if isinstance(on_reused, int) and on_usage.get("cache_read") != on_reused:
                mechanics_errors.append(
                    f"cache_on_provider_server_reuse_mismatch:{pair_id}:{turn_index}"
                )
            off_prompt_tokens = _observed_value(off_turn, "server_prompt_tokens")
            on_prompt_tokens = _observed_value(on_turn, "server_prompt_tokens")
            off_predicted_tokens = _observed_value(
                off_turn, "server_predicted_tokens"
            )
            on_predicted_tokens = _observed_value(
                on_turn, "server_predicted_tokens"
            )
            if all(
                isinstance(value, int)
                for value in (
                    off_reused,
                    on_reused,
                    off_prompt_tokens,
                    on_prompt_tokens,
                )
            ):
                off_logical_prompt = off_prompt_tokens + off_reused
                on_logical_prompt = on_prompt_tokens + on_reused
                if off_logical_prompt != on_logical_prompt:
                    input_errors.append(
                        f"server_logical_prompt_tokens_mismatch:{pair_id}:{turn_index}"
                    )
                if off_usage.get("logical_prompt_tokens") != off_logical_prompt:
                    mechanics_errors.append(
                        f"cache_off_provider_server_prompt_mismatch:{pair_id}:{turn_index}"
                    )
                if on_usage.get("logical_prompt_tokens") != on_logical_prompt:
                    mechanics_errors.append(
                        f"cache_on_provider_server_prompt_mismatch:{pair_id}:{turn_index}"
                    )
                if off_usage.get("input") != off_prompt_tokens:
                    mechanics_errors.append(
                        f"cache_off_provider_server_input_mismatch:{pair_id}:{turn_index}"
                    )
                if on_usage.get("input") != on_prompt_tokens:
                    mechanics_errors.append(
                        f"cache_on_provider_server_input_mismatch:{pair_id}:{turn_index}"
                    )
            if not isinstance(off_predicted_tokens, int) or not isinstance(
                on_predicted_tokens, int
            ):
                mechanics_errors.append(
                    f"server_predicted_tokens_unobservable:{pair_id}:{turn_index}"
                )
            else:
                if off_predicted_tokens != on_predicted_tokens:
                    behavior_errors.append(
                        f"server_predicted_tokens_mismatch:{pair_id}:{turn_index}"
                    )
                if off_usage.get("output") != off_predicted_tokens:
                    mechanics_errors.append(
                        f"cache_off_provider_server_output_mismatch:{pair_id}:{turn_index}"
                    )
                if on_usage.get("output") != on_predicted_tokens:
                    mechanics_errors.append(
                        f"cache_on_provider_server_output_mismatch:{pair_id}:{turn_index}"
                    )
            off_prompt = _observed_value(off_turn, "prompt_evaluation_seconds")
            on_prompt = _observed_value(on_turn, "prompt_evaluation_seconds")
            if not isinstance(off_prompt, (int, float)) or not isinstance(
                on_prompt, (int, float)
            ):
                performance_errors.append(
                    f"prompt_timing_unobservable:{pair_id}:{turn_index}"
                )
            else:
                total_off_prompt_seconds += float(off_prompt)
                total_on_prompt_seconds += float(on_prompt)
            compared_turns += 1

    if len(off_attempts) != len(expected_order) or len(on_attempts) != len(
        expected_order
    ):
        input_errors.append("canary_pair_count_mismatch")
    if mechanics_errors:
        performance_errors.append("mechanics_invalid_for_performance_claim")

    savings_seconds: float | None = None
    savings_fraction: float | None = None
    if not performance_errors and total_off_prompt_seconds > 0:
        savings_seconds = total_off_prompt_seconds - total_on_prompt_seconds
        savings_fraction = savings_seconds / total_off_prompt_seconds
        if savings_fraction <= float(minimum_prompt_savings_fraction):
            performance_errors.append("prompt_evaluation_savings_below_threshold")
    elif not performance_errors:
        performance_errors.append("cache_off_prompt_evaluation_time_not_positive")

    statuses = {
        "input_equivalence": _status(input_errors),
        "sampling_equivalence": _status(sampling_errors),
        "behavior_invariance": _status(behavior_errors),
        "mechanics_observability": _status(mechanics_errors),
        "performance_evidence": _status(performance_errors),
    }
    accepted = all(status["passed"] for status in statuses.values())
    if not statuses["performance_evidence"]["passed"]:
        savings_seconds = None
        savings_fraction = None
    payload: dict[str, Any] = {
        "schema_version": CACHE_CANARY_REPORT_SCHEMA_VERSION,
        "cache_off_cell_hash": cache_off["cell_hash"],
        "cache_on_cell_hash": cache_on["cell_hash"],
        "pair_order_hash": canonical_receipt_hash(expected_order),
        "pair_count": len(expected_order),
        "compared_provider_turns": compared_turns,
        **statuses,
        "performance_summary": {
            "minimum_prompt_savings_fraction": float(
                minimum_prompt_savings_fraction
            ),
            "cache_on_reused_prefix_tokens": observed_on_reused_tokens
            if statuses["mechanics_observability"]["passed"]
            else None,
            "cache_off_prompt_evaluation_seconds": total_off_prompt_seconds
            if statuses["performance_evidence"]["passed"]
            else None,
            "cache_on_prompt_evaluation_seconds": total_on_prompt_seconds
            if statuses["performance_evidence"]["passed"]
            else None,
            "prompt_evaluation_savings_seconds": savings_seconds,
            "prompt_evaluation_savings_fraction": savings_fraction,
        },
        "stage1_acceptance": {
            "passed": accepted,
            "decision": "retain_transparent_cache" if accepted else "no_go",
        },
    }
    return _with_hash(payload, "report_hash")


def _http_get_observation(url: str) -> dict[str, Any]:
    request = Request(url, method="GET")
    try:
        with urlopen(request, timeout=10) as response:
            body = response.read()
            status = response.status
    except HTTPError as error:
        body = error.read()
        status = error.code
    except URLError as error:
        return {
            "http_status": None,
            "response_bytes": None,
            "response_sha256": None,
            "transport_error": type(error.reason).__name__,
        }
    return {
        "http_status": status,
        "response_bytes": len(body),
        "response_sha256": hashlib.sha256(body).hexdigest(),
    }


def _model_endpoint_entry(base_url: str, model_alias: str) -> dict[str, Any]:
    request = Request(f"{base_url.rstrip('/')}/models", method="GET")
    with urlopen(request, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    values = payload.get("data") if isinstance(payload, Mapping) else None
    if not isinstance(values, list):
        raise RuntimeError("model endpoint response omitted data")
    matches = [
        value
        for value in values
        if isinstance(value, Mapping) and value.get("id") == model_alias
    ]
    if len(matches) != 1:
        raise RuntimeError("model endpoint did not return exactly one alias")
    return matches[0]


def _remote_model_endpoint_entry(
    host: str, base_url: str, model_alias: str
) -> dict[str, Any]:
    payload = json.loads(
        _ssh_capture(
            host,
            ["curl", "-fsS", f"{base_url.rstrip('/')}/models"],
        )
    )
    values = payload.get("data") if isinstance(payload, Mapping) else None
    if not isinstance(values, list):
        raise RuntimeError("remote model endpoint response omitted data")
    matches = [
        value
        for value in values
        if isinstance(value, Mapping) and value.get("id") == model_alias
    ]
    if len(matches) != 1:
        raise RuntimeError("remote model endpoint did not return exactly one alias")
    return matches[0]


def _remote_http_get_observation(host: str, url: str) -> dict[str, Any]:
    try:
        value = _ssh_capture(
            host,
            [
                "curl",
                "-sS",
                "--max-time",
                "10",
                "-o",
                "/dev/null",
                "-w",
                "%{http_code}",
                url,
            ],
        )
        status = int(value)
    except (OSError, RuntimeError, ValueError):
        status = None
    return {
        "http_status": status,
        "response_bytes": None,
        "response_sha256": None,
    }


def probe_cache_runtime(
    *,
    host: str,
    provider_base_url: str,
    remote_provider_base_url: str | None = None,
    model_alias: str,
    pi_binary: str,
    llama_server_binary: str,
    model_artifact: str,
) -> dict[str, Any]:
    """Perform a no-model capability probe using only GET and identity commands."""
    resolved_pi = shutil.which(pi_binary)
    if resolved_pi is None:
        raise RuntimeError(f"Pi executable not found: {pi_binary}")
    pi_path = Path(resolved_pi).resolve()
    model_entry = (
        _remote_model_endpoint_entry(host, remote_provider_base_url, model_alias)
        if remote_provider_base_url is not None
        else _model_endpoint_entry(provider_base_url, model_alias)
    )
    status = model_entry.get("status")
    if not isinstance(status, Mapping):
        raise RuntimeError("model endpoint omitted status")
    server_argv = status.get("args")
    if not isinstance(server_argv, list):
        raise RuntimeError("model endpoint omitted server argv")
    if not server_argv or server_argv[0] != llama_server_binary:
        raise RuntimeError("serving process llama-server path mismatch")

    def argument_value(*names: str) -> str | None:
        for index, token in enumerate(server_argv):
            for name in names:
                if token == name and index + 1 < len(server_argv):
                    return server_argv[index + 1]
                if token.startswith(f"{name}="):
                    return token.split("=", 1)[1]
        return None

    if argument_value("--model", "-m") != model_artifact:
        raise RuntimeError("serving process model artifact path mismatch")
    server_version = _ssh_capture(
        host, [llama_server_binary, "--version"], stderr_fallback=True
    ).splitlines()[0]
    server_help = _ssh_capture(
        host,
        [llama_server_binary, "--help"],
        timeout=120,
        stderr_fallback=True,
    )
    server_sha = _ssh_capture(host, ["sha256sum", llama_server_binary]).split()[0]
    model_sha = _ssh_capture(
        host, ["sha256sum", model_artifact], timeout=900
    ).split()[0]
    endpoint_base_url = remote_provider_base_url or provider_base_url
    root_url = endpoint_base_url.rstrip("/")
    if root_url.endswith("/v1"):
        root_url = root_url[:-3]
    observation = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "pi_version": _ssh_free_pi_version(pi_binary),
        "pi_sha256": _sha256_file(pi_path),
        "llama_server_version": server_version,
        "llama_server_sha256": server_sha,
        "model_sha256": model_sha,
        "model_state": status.get("value"),
        "server_argv": server_argv,
        "server_help": server_help,
        "metrics_endpoint": (
            _remote_http_get_observation(host, f"{root_url}/metrics")
            if remote_provider_base_url is not None
            else _http_get_observation(f"{root_url}/metrics")
        ),
        "slots_endpoint": (
            _remote_http_get_observation(host, f"{root_url}/slots")
            if remote_provider_base_url is not None
            else _http_get_observation(f"{root_url}/slots")
        ),
    }
    return build_cache_runtime_receipt(observation)


def _ssh_free_pi_version(pi_binary: str) -> str:
    completed = subprocess.run(
        [pi_binary, "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed.stdout.strip() or completed.stderr.strip()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyreplab-cache-mechanics")
    subparsers = parser.add_subparsers(dest="command", required=True)
    probe = subparsers.add_parser("probe")
    probe.add_argument("--host", default="ubuntu-local")
    probe.add_argument("--provider-base-url", required=True)
    probe.add_argument("--remote-provider-base-url", default=None)
    probe.add_argument("--model", required=True)
    probe.add_argument("--pi", default="pi")
    probe.add_argument("--llama-server-binary", required=True)
    probe.add_argument("--model-artifact", required=True)
    probe.add_argument("--output", required=True)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--cache-off", required=True)
    compare.add_argument("--cache-on", required=True)
    compare.add_argument("--pair-order", required=True)
    compare.add_argument("--minimum-prompt-savings-fraction", type=float, default=0.0)
    compare.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "probe":
        report = probe_cache_runtime(
            host=args.host,
            provider_base_url=args.provider_base_url,
            remote_provider_base_url=args.remote_provider_base_url,
            model_alias=args.model,
            pi_binary=args.pi,
            llama_server_binary=args.llama_server_binary,
            model_artifact=args.model_artifact,
        )
    else:
        pair_order = _load_json(args.pair_order)
        if not isinstance(pair_order, list):
            raise ValueError("pair-order artifact must be a JSON list")
        report = compare_cache_cells(
            _load_json(args.cache_off),
            _load_json(args.cache_on),
            pair_order,
            minimum_prompt_savings_fraction=args.minimum_prompt_savings_fraction,
        )
    _write_immutable_json(Path(args.output).expanduser().resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CACHE_CANARY_CELL_SCHEMA_VERSION",
    "CACHE_CANARY_REPORT_SCHEMA_VERSION",
    "CACHE_RUNTIME_RECEIPT_SCHEMA_VERSION",
    "PROVIDER_TURN_CACHE_RECEIPT_SCHEMA_VERSION",
    "build_cache_runtime_receipt",
    "build_provider_turn_cache_receipt",
    "canonical_receipt_hash",
    "compare_cache_cells",
    "parse_cache_launch_configuration",
    "probe_cache_runtime",
    "validate_cache_runtime_receipt",
    "validate_provider_turn_cache_receipt",
]
