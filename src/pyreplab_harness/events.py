from __future__ import annotations

import json
from typing import Any, Iterable


def _contains_tool_limit_rejection(value: Any) -> bool:
    """Return whether a JSON event value contains the budget rejection marker."""
    try:
        payload = json.dumps(value, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        payload = str(value)
    return "Tool call limit reached" in payload


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
            texts = [
                item.get("text", "")
                for item in message.get("content", [])
                if item.get("type") == "text"
            ]
            if texts:
                final_text = "\n".join(texts)
        elif event_type == "tool_execution_end":
            if _contains_tool_limit_rejection(event.get("result")):
                tool_limit_rejection_count += 1
            tool_executions.append(
                {
                    "tool_call_id": event.get("toolCallId"),
                    "tool_name": event.get("toolName"),
                    "result": event.get("result"),
                    "is_error": event.get("isError", False),
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
        "final_text": final_text,
    }
