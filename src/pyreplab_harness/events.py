from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable


NORMALIZED_EVENT_SCHEMA_VERSION = "pi-events-normalized-v4"
PROVIDER_TURN_SEMANTICS = "provider-backed-assistant-message-v1"
BUDGET_RECEIPT_SCHEMA_VERSION = "pyreplab-gym-budget-v3-receipt-v1"

_PROVIDER_USAGE_FIELDS = {
    "input": "input",
    "output": "output",
    "cache_read": "cacheRead",
    "cache_write": "cacheWrite",
    "reasoning": "reasoning",
    "total_tokens": "totalTokens",
}


def _canonical_sha256(value: Any) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _assistant_content_for_hash(content: Any) -> list[dict[str, Any]]:
    """Remove provider-generated identifiers while retaining authored content."""
    if not isinstance(content, list):
        raise ValueError("assistant message content must be a list")
    normalized: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            raise ValueError("assistant content items must be objects")
        normalized.append(
            {
                key: value
                for key, value in item.items()
                if key not in {"id", "thinkingSignature"}
            }
        )
    return normalized


def _provider_usage(value: Any) -> dict[str, Any]:
    raw = value if isinstance(value, dict) else {}
    usage: dict[str, int | None] = {}
    missing: list[str] = []
    for normalized_name, source_name in _PROVIDER_USAGE_FIELDS.items():
        if source_name not in raw:
            usage[normalized_name] = None
            missing.append(normalized_name)
            continue
        observed = raw[source_name]
        if (
            not isinstance(observed, (int, float))
            or isinstance(observed, bool)
            or not float(observed).is_integer()
            or observed < 0
        ):
            raise ValueError(
                f"assistant usage {source_name} must be a non-negative integer"
            )
        usage[normalized_name] = int(observed)
    logical_prompt_tokens = None
    if usage["input"] is not None and usage["cache_read"] is not None:
        logical_prompt_tokens = usage["input"] + usage["cache_read"]
    return {
        **usage,
        "logical_prompt_tokens": logical_prompt_tokens,
        "complete": not missing,
        "missing_fields": missing,
    }


def _provider_turn(
    message: dict[str, Any], *, turn_index: int, source_event_index: int
) -> dict[str, Any]:
    timestamp = message.get("timestamp")
    timestamp_observation = (
        {
            "status": "observed",
            "value": timestamp,
            "source": "message.timestamp",
        }
        if isinstance(timestamp, (str, int, float)) and not isinstance(timestamp, bool)
        else {
            "status": "unobservable",
            "reason": "message_timestamp_missing",
        }
    )
    content = _assistant_content_for_hash(message.get("content", []))
    return {
        "turn_index": turn_index,
        "source_event_index": source_event_index,
        "timestamp": timestamp_observation,
        "provider": message.get("provider"),
        "model": message.get("model"),
        "usage": _provider_usage(message.get("usage")),
        "assistant_content_sha256": _canonical_sha256(content),
        "stop_reason": message.get("stopReason"),
    }


def _contains_tool_limit_rejection(value: Any) -> bool:
    """Return whether a JSON event value contains the budget rejection marker."""
    try:
        payload = json.dumps(value, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        payload = str(value)
    lowered = payload.casefold()
    return "tool call limit" in lowered or "unbrowser call limit" in lowered


def _is_operation_aborted_result(value: Any) -> bool:
    """Return whether Pi reduced a blocked tool call to its exact abort result."""
    if not isinstance(value, dict) or value.get("details") != {}:
        return False
    return value.get("content") == [{"type": "text", "text": "Operation aborted"}]


def _is_pre_execution_rejection(value: Any) -> bool:
    """Return whether Pi rejected a tool call before the tool executed."""
    if not isinstance(value, dict) or value.get("details") != {}:
        return False
    content = value.get("content")
    if not isinstance(content, list):
        return False
    texts = [
        item.get("text")
        for item in content
        if isinstance(item, dict)
        and item.get("type") == "text"
        and isinstance(item.get("text"), str)
    ]
    return any(
        text.startswith("Validation failed for tool ")
        or (text.startswith("Tool ") and text.endswith(" not found"))
        or (
            text.startswith("Tool call \"")
            and " was not executed: " in text
        )
        for text in texts
    )


def _planning_preamble_shape(text: str) -> dict[str, Any]:
    """Return marker/count features without retaining model-authored text."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return {
        "present": bool(text.strip()),
        "line_count": len(lines),
        "character_count": len(text),
        "plan_marker": any(line.upper().startswith("PLAN:") for line in lines),
        "step_marker_count": len(
            re.findall(r"(?im)^\s*STEP\s+\d+\s*:", text)
        ),
    }


def _has_zero_token_usage(message: dict[str, Any]) -> bool:
    usage = message.get("usage")
    if not isinstance(usage, dict):
        return False
    required = ("input", "output", "cacheRead", "cacheWrite", "totalTokens")
    return all(
        key in usage
        and isinstance(usage[key], (int, float))
        and not isinstance(usage[key], bool)
        and usage[key] == 0
        for key in required
    )


def _is_terminal_synthetic_abort(
    events: list[dict[str, Any]],
    index: int,
    budget_receipt: dict[str, Any] | None = None,
) -> bool:
    """Identify Pi's local post-budget abort, not a provider response.

    The pinned Pi emits this exact terminal assistant event after the budget
    extension aborts an over-cap tool call. Requiring the preceding abort and
    terminal position prevents generic zero-usage provider failures from being
    silently removed from accounting.
    """
    message = events[index].get("message") or {}
    if (
        message.get("role") != "assistant"
        or message.get("content") != []
        or not isinstance(message.get("provider"), str)
        or not isinstance(message.get("model"), str)
        or not _has_zero_token_usage(message)
    ):
        return False

    for later in events[index + 1 :]:
        if later.get("type") in {"tool_execution_start", "tool_execution_end"}:
            return False
        if (
            later.get("type") == "message_end"
            and (later.get("message") or {}).get("role") == "assistant"
        ):
            return False

    # The v3 budget extension blocks provider request N+1 before any HTTP call.
    # Its receipt is authoritative for the otherwise context-free local abort.
    if (
        isinstance(budget_receipt, dict)
        and budget_receipt.get("schema_version") == BUDGET_RECEIPT_SCHEMA_VERSION
        and budget_receipt.get("provider_request_blocks") == 1
        and message.get("stopReason") == "aborted"
        and message.get("errorMessage") == "Request aborted"
    ):
        return True

    tool_abort_shape = (
        message.get("stopReason") == "error"
        and message.get("errorMessage") == "This operation was aborted"
    ) or (
        message.get("stopReason") == "aborted"
        and message.get("errorMessage") == "Request aborted"
    )
    if not tool_abort_shape:
        return False

    abort_index: int | None = None
    for previous_index in range(index - 1, -1, -1):
        previous = events[previous_index]
        if previous.get("type") == "tool_execution_end":
            if _is_operation_aborted_result(previous.get("result")):
                abort_index = previous_index
            break
        if (
            previous.get("type") == "message_end"
            and (previous.get("message") or {}).get("role") == "assistant"
        ):
            break
    if abort_index is None:
        return False

    tool_call_id = events[abort_index].get("toolCallId")
    if not isinstance(tool_call_id, str) or not tool_call_id:
        return False
    linked_provider_tool_call = False
    for previous in reversed(events[:abort_index]):
        if (
            previous.get("type") != "message_end"
            or (previous.get("message") or {}).get("role") != "assistant"
        ):
            continue
        linked_provider_tool_call = any(
            isinstance(item, dict)
            and item.get("type") in {"toolCall", "tool_use", "tool-call"}
            and item.get("id") == tool_call_id
            for item in (previous.get("message") or {}).get("content", [])
        )
        break
    if not linked_provider_tool_call:
        return False

    return True


def normalize_pi_events(
    lines: str | Iterable[str],
    budget_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    iterable = lines.splitlines() if isinstance(lines, str) else lines
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(iterable, start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid Pi JSON on line {line_number}: {error}") from error
        if not isinstance(event, dict):
            raise ValueError(f"invalid Pi JSON object on line {line_number}")
        events.append(event)

    synthetic_abort_indices = {
        index
        for index, event in enumerate(events)
        if event.get("type") == "message_end"
        and (event.get("message") or {}).get("role") == "assistant"
        and _is_terminal_synthetic_abort(events, index, budget_receipt)
    }
    session: dict[str, Any] = {}
    assistant_messages: list[dict[str, Any]] = []
    provider_turns: list[dict[str, Any]] = []
    provider_turn_count = 0
    tool_executions: list[dict[str, Any]] = []
    final_text = ""
    usage = {
        "input": 0,
        "output": 0,
        "cache_read": 0,
        "cache_write": 0,
        "reasoning": 0,
        "total_tokens": 0,
    }
    provider: str | None = None
    model: str | None = None
    stop_reasons: dict[str, int] = {}
    tool_limit_rejection_count = 0
    pre_tool_text: list[str] = []
    tool_requested = False

    for event_index, event in enumerate(events):
        event_type = event.get("type")
        if event_type == "session":
            session = {
                "id": event.get("id"),
                "version": event.get("version"),
                "timestamp": event.get("timestamp"),
                "cwd": event.get("cwd"),
            }
        elif event_type == "message_end":
            message = event.get("message") or {}
            if message.get("role") != "assistant":
                continue
            assistant_messages.append(message)
            if event_index not in synthetic_abort_indices:
                provider_turn_count += 1
                provider_turns.append(
                    _provider_turn(
                        message,
                        turn_index=provider_turn_count,
                        source_event_index=event_index,
                    )
                )
            provider = message.get("provider") or provider
            model = message.get("model") or model
            current = message.get("usage") or {}
            usage["input"] += int(current.get("input") or 0)
            usage["output"] += int(current.get("output") or 0)
            usage["cache_read"] += int(current.get("cacheRead") or 0)
            usage["cache_write"] += int(current.get("cacheWrite") or 0)
            usage["reasoning"] += int(current.get("reasoning") or 0)
            usage["total_tokens"] += int(current.get("totalTokens") or 0)
            stop_reason = message.get("stopReason")
            if isinstance(stop_reason, str) and stop_reason:
                stop_reasons[stop_reason] = stop_reasons.get(stop_reason, 0) + 1
            for item in message.get("content", []):
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type in {"toolCall", "tool_use", "tool-call"}:
                    tool_requested = True
                elif item_type == "text" and not tool_requested:
                    text = item.get("text")
                    if isinstance(text, str) and text:
                        pre_tool_text.append(text)
            texts = [
                item.get("text", "")
                for item in message.get("content", [])
                if item.get("type") == "text"
            ]
            if texts:
                final_text = "\n".join(texts)
        elif event_type == "tool_execution_end":
            result = event.get("result")
            budget_rejected = _contains_tool_limit_rejection(result)
            if budget_rejected:
                tool_limit_rejection_count += 1
            tool_executions.append(
                {
                    "tool_call_id": event.get("toolCallId"),
                    "tool_name": event.get("toolName"),
                    "result": result,
                    "is_error": event.get("isError", False),
                    "budget_rejected": budget_rejected,
                    "operation_aborted": _is_operation_aborted_result(result),
                    "pre_execution_rejected": _is_pre_execution_rejection(result),
                }
            )

    return {
        "schema_version": NORMALIZED_EVENT_SCHEMA_VERSION,
        "provider_turn_semantics": PROVIDER_TURN_SEMANTICS,
        "budget_receipt": dict(budget_receipt) if budget_receipt is not None else None,
        "session": session,
        "provider": provider,
        "model": model,
        "usage": usage,
        "assistant_message_count": len(assistant_messages),
        "provider_turn_count": provider_turn_count,
        "provider_turns": provider_turns,
        "synthetic_assistant_message_count": len(synthetic_abort_indices),
        "tool_executions": tool_executions,
        "tool_call_count": len(tool_executions),
        "tool_limit_rejection_count": tool_limit_rejection_count,
        "length_stop_count": stop_reasons.get("length", 0),
        "stop_reasons": dict(sorted(stop_reasons.items())),
        "planning_preamble": _planning_preamble_shape("\n".join(pre_tool_text)),
        "final_text": final_text,
    }
