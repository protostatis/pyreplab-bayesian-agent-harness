from __future__ import annotations

import json
import re
from typing import Any, Iterable


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
    content = value.get("content")
    return isinstance(content, list) and any(
        isinstance(item, dict)
        and item.get("type") == "text"
        and item.get("text") == "Operation aborted"
        for item in content
    )


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


def normalize_pi_events(lines: str | Iterable[str]) -> dict[str, Any]:
    iterable = lines.splitlines() if isinstance(lines, str) else lines
    session: dict[str, Any] = {}
    assistant_messages: list[dict[str, Any]] = []
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

    for line_number, line in enumerate(iterable, start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid Pi JSON on line {line_number}: {error}") from error
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
        "session": session,
        "provider": provider,
        "model": model,
        "usage": usage,
        "assistant_message_count": len(assistant_messages),
        "provider_turn_count": len(assistant_messages),
        "tool_executions": tool_executions,
        "tool_call_count": len(tool_executions),
        "tool_limit_rejection_count": tool_limit_rejection_count,
        "length_stop_count": stop_reasons.get("length", 0),
        "stop_reasons": dict(sorted(stop_reasons.items())),
        "planning_preamble": _planning_preamble_shape("\n".join(pre_tool_text)),
        "final_text": final_text,
    }
