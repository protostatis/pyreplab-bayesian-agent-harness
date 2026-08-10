"""Leakage-safe manipulation checks for M3 Unbrowser policy factors."""

from __future__ import annotations

from typing import Any, Mapping

from .treatments import TreatmentSpec

_READ_ACTIONS = frozenset({"text", "query", "blockmap"})
_STATE_ACTIONS = frozenset({"navigate", "click", "type", "submit"})
_OBSERVATION_ACTION = {
    "text_first": "text",
    "structure_first": "blockmap",
    "targeted_query_first": "query",
}


def _details(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    value = entry.get("details")
    return value if isinstance(value, Mapping) else {}


def _budget_rejected(entry: Mapping[str, Any]) -> bool:
    if bool(entry.get("budget_rejected")):
        return True
    error = str(_details(entry).get("error") or "").casefold()
    return "tool_limit" in error or "tool call limit" in error


def _tool_error(entry: Mapping[str, Any]) -> bool:
    if _budget_rejected(entry):
        return False
    details = _details(entry)
    status = details.get("status")
    if isinstance(status, int) and status != 200:
        return True
    if bool(entry.get("is_error")) or bool(details.get("error")):
        return True
    return False


def _successful_unbrowser(entry: Mapping[str, Any]) -> bool:
    return (
        entry.get("tool_name") == "unbrowser"
        and not _budget_rejected(entry)
        and not _tool_error(entry)
        and isinstance(_details(entry).get("action"), str)
    )


def _planning_adherent(level: str, shape: Mapping[str, Any]) -> bool:
    present = bool(shape.get("present"))
    if level == "direct":
        return not present
    if level == "brief_plan":
        return bool(shape.get("plan_marker")) and shape.get("line_count") == 1
    if level == "decompose":
        return int(shape.get("step_marker_count") or 0) >= 2
    return False


def assess_policy_adherence(
    treatment: TreatmentSpec,
    trajectory: Mapping[str, Any] | None,
    *,
    required_recovery_probe_url: str | None = None,
    required_recovery_probe_status: int | None = None,
) -> dict[str, Any]:
    """Assess one measured trajectory against its frozen grammar factors."""
    metadata = treatment.generator_metadata
    trace_value = (trajectory or {}).get("tool_trace", [])
    trace = [entry for entry in trace_value if isinstance(entry, Mapping)]
    admitted = [entry for entry in trace if not _budget_rejected(entry)]

    planning_shape = (trajectory or {}).get("planning_preamble", {})
    if not isinstance(planning_shape, Mapping):
        planning_shape = {}
    planning_level = str(metadata.get("planning", ""))

    navigate_index = next(
        (
            index
            for index, entry in enumerate(trace)
            if _successful_unbrowser(entry)
            and _details(entry).get("action") == "navigate"
        ),
        None,
    )
    first_observation = None
    if navigate_index is not None:
        for entry in trace[navigate_index + 1 :]:
            if not _successful_unbrowser(entry):
                continue
            action = _details(entry).get("action")
            if action == "navigate":
                break
            if action in _READ_ACTIONS:
                first_observation = str(action)
                break
    observation_level = str(metadata.get("observation", ""))
    expected_observation = _OBSERVATION_ACTION.get(observation_level)

    submission_index = next(
        (
            index
            for index, entry in enumerate(trace)
            if entry.get("tool_name") == "bash"
            and bool(_details(entry).get("result_submission"))
            and not _tool_error(entry)
        ),
        len(trace),
    )
    verification_opportunity = submission_index < len(trace)
    successful_browser = [
        entry
        for entry in trace[:submission_index]
        if _successful_unbrowser(entry)
    ]
    last_state_index = -1
    for index, entry in enumerate(successful_browser):
        if _details(entry).get("action") in _STATE_ACTIONS:
            last_state_index = index
    final_reads = [
        entry
        for entry in successful_browser[last_state_index + 1 :]
        if _details(entry).get("action") in _READ_ACTIONS
    ]
    signatures = [
        (
            _details(entry).get("action"),
            _details(entry).get("selector"),
            _details(entry).get("ref"),
        )
        for entry in final_reads
    ]
    repeated_read = bool(signatures) and signatures[-1] in signatures[:-1]
    verification_level = str(metadata.get("verification", ""))
    verification_adherent = (
        repeated_read
        if verification_level == "final_reobserve"
        else not repeated_read
    )

    recovery_probe_adherent: bool | None = None
    if required_recovery_probe_url is None:
        first_error_index = next(
            (index for index, entry in enumerate(admitted) if _tool_error(entry)),
            None,
        )
    else:
        candidate_index = next(
            (
                index
                for index, entry in enumerate(admitted)
                if entry.get("tool_name") == "unbrowser"
                and _details(entry).get("action") == "click"
                and _details(entry).get("url") == required_recovery_probe_url
                and _details(entry).get("status") == required_recovery_probe_status
                and _tool_error(entry)
            ),
            None,
        )
        recovery_probe_adherent = bool(
            candidate_index is not None
            and not any(
                _tool_error(entry)
                or (
                    entry.get("tool_name") == "unbrowser"
                    and _details(entry).get("action") in {"click", "type", "submit"}
                )
                for entry in admitted[:candidate_index]
            )
        )
        first_error_index = candidate_index if recovery_probe_adherent else None
    recovery_level = str(metadata.get("recovery", ""))
    recovery_eligible = first_error_index is not None
    continued_after_error = False
    successful_same_tool_retry = False
    if first_error_index is not None:
        error_entry = admitted[first_error_index]
        later = admitted[first_error_index + 1 :]
        continued_after_error = bool(later)
        same_tool = (
            later[0]
            if later and later[0].get("tool_name") == error_entry.get("tool_name")
            else None
        )
        successful_same_tool_retry = (
            same_tool is not None and not _tool_error(same_tool)
        )
    recovery_adherent: bool | None
    if not recovery_eligible:
        recovery_adherent = None
    elif recovery_level == "fail_fast":
        recovery_adherent = not continued_after_error
    else:
        recovery_adherent = successful_same_tool_retry

    cap = int(treatment.tool_call_limit)
    rejected_count = len(trace) - len(admitted)
    return {
        "planning_level": planning_level,
        "planning_adherent": _planning_adherent(planning_level, planning_shape),
        "observation_level": observation_level,
        "expected_first_observation": expected_observation,
        "first_observation": first_observation,
        "observation_adherent": first_observation == expected_observation,
        "verification_level": verification_level,
        "verification_opportunity": verification_opportunity,
        "repeated_final_read": repeated_read,
        "verification_adherent": verification_adherent,
        "recovery_level": recovery_level,
        "recovery_probe_required": required_recovery_probe_url is not None,
        "recovery_probe_adherent": recovery_probe_adherent,
        "recovery_eligible": recovery_eligible,
        "continued_after_error": continued_after_error,
        "successful_same_tool_retry": successful_same_tool_retry,
        "recovery_adherent": recovery_adherent,
        "tool_cap": cap,
        "admitted_tool_call_count": len(admitted),
        "budget_rejection_count": rejected_count,
        "tool_cap_compliant": len(admitted) <= cap,
    }


__all__ = ["assess_policy_adherence"]
