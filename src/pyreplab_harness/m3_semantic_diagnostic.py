"""Privacy-safe post-hoc diagnostics for an M3 semantic dataset package.

The report is aggregate-only. It consumes curated attempt metadata and the
normalized execution files retained inside a verified package, but never emits
messages, reasoning, result bodies, URLs, selectors, refs, or absolute paths.
It is descriptive error analysis and cannot amend the frozen replication gate.

Usage::

    python -m pyreplab_harness.m3_semantic_diagnostic PACKAGE --output REPORT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .io_utils import write_json
from .m3_semantic_dataset import privacy_scan, verify_package


REPORT_SCHEMA = "m3-semantic-action-path-diagnostic-v1"
_HEX_DIGITS = frozenset("0123456789abcdef")
_FAMILY_TO_MATCHING_CAPABILITY = {
    "form_entry_validation": "form_specialist",
    "table_filter_sort": "table_specialist",
}
_ALLOWED_ACTIONS = frozenset(
    {
        "blockmap",
        "click",
        "extract_table",
        "navigate",
        "query",
        "query_text",
        "submit",
        "text",
        "type",
    }
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path.name}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON objects in {path.name}")
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_HEX_DIGITS)
    )


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _details(execution: Mapping[str, Any]) -> Mapping[str, Any]:
    result = execution.get("result")
    if not isinstance(result, Mapping):
        return {}
    details = result.get("details")
    return details if isinstance(details, Mapping) else {}


def _canonical_token(execution: Mapping[str, Any]) -> str:
    tool_name = execution.get("tool_name")
    details = _details(execution)
    if tool_name == "unbrowser":
        action = details.get("action")
        if action in _ALLOWED_ACTIONS:
            return f"unbrowser_{action}"
        return "unbrowser_other"
    if tool_name in {"semantic_form", "semantic_table"}:
        return str(tool_name)
    if tool_name == "bash":
        return "result_submission" if details.get("result_submission") else "bash_other"
    return "other_tool"


def _request_shape_counts(raw_events_path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    events = _read_jsonl(raw_events_path)
    for event in events:
        if event.get("type") != "message_end":
            continue
        message = event.get("message")
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, Mapping) or item.get("type") != "toolCall":
                continue
            name = item.get("name")
            arguments = item.get("arguments")
            if not isinstance(arguments, Mapping):
                arguments = {}
            if name == "semantic_table":
                bounded = (
                    isinstance(arguments.get("filters"), list)
                    and bool(arguments["filters"])
                    and isinstance(arguments.get("limit"), int)
                    and not isinstance(arguments["limit"], bool)
                    and isinstance(arguments.get("projection"), list)
                    and bool(arguments["projection"])
                    and isinstance(arguments.get("sort"), Mapping)
                    and bool(arguments["sort"])
                )
                if bounded:
                    counts["bounded_semantic_table"] += 1
                else:
                    counts["unbounded_semantic_table"] += 1
            elif name == "unbrowser" and arguments.get("action") == "query":
                counts["generic_unbrowser_query"] += 1
    return counts


def _classify_attempt(
    row: Mapping[str, Any],
    normalized: Mapping[str, Any],
    request_shape_counts: Counter[str],
) -> dict[str, Any]:
    task = row.get("task")
    treatment = row.get("treatment")
    execution = row.get("execution")
    mechanism = row.get("mechanism")
    outcome = row.get("outcome")
    panel = row.get("panel")
    if not all(
        isinstance(value, Mapping)
        for value in (task, treatment, execution, mechanism, outcome, panel)
    ):
        raise ValueError("attempt row is missing required mappings")

    family = task.get("template")
    capability = treatment.get("capability")
    matching = _FAMILY_TO_MATCHING_CAPABILITY.get(str(family))
    if matching is None:
        raise ValueError(f"unsupported task family {family!r}")
    if capability not in {"form_specialist", "table_specialist"}:
        raise ValueError(f"unsupported capability {capability!r}")

    tool_executions = normalized.get("tool_executions")
    if not isinstance(tool_executions, list) or not all(
        isinstance(item, Mapping) for item in tool_executions
    ):
        raise ValueError("normalized execution has invalid tool_executions")
    if not _is_int(normalized.get("tool_call_count")):
        raise ValueError("normalized execution tool_call_count must be an integer")
    if normalized.get("tool_call_count") != len(tool_executions):
        raise ValueError("normalized execution tool_call_count mismatch")
    if not _is_int(execution.get("tool_call_count")):
        raise ValueError("curated execution tool_call_count must be an integer")
    if execution.get("tool_call_count") != len(tool_executions):
        raise ValueError("curated and normalized tool_call_count mismatch")

    tokens: list[str] = []
    action_counts: Counter[str] = Counter()
    operation_aborted = False
    pre_execution_rejected = False
    budget_rejected = False
    submission_executions = 0
    for item in tool_executions:
        token = _canonical_token(item)
        tokens.append(token)
        action_counts[token] += 1
        if item.get("operation_aborted") is True:
            operation_aborted = True
        if item.get("pre_execution_rejected") is True:
            pre_execution_rejected = True
        if item.get("budget_rejected") is True:
            budget_rejected = True
        if _details(item).get("result_submission") is True:
            submission_executions += 1

    success = outcome.get("success")
    if not isinstance(success, bool):
        raise ValueError("attempt outcome.success must be boolean")
    admitted_calls = mechanism.get("admitted_tool_call_count")
    rejected_calls = mechanism.get("rejected_tool_call_count")
    tool_call_limit = treatment.get("tool_call_limit")
    if not all(_is_int(value) for value in (admitted_calls, rejected_calls, tool_call_limit)):
        raise ValueError("attempt tool-call counts must be integers")

    total_tokens = execution.get("usage", {}).get("total_tokens")
    output_tokens = outcome.get("output_tokens")
    total_seconds = execution.get("timing", {}).get("total_seconds")
    if not _is_int(total_tokens) or not _is_int(output_tokens):
        raise ValueError("attempt token counts must be integers")
    if not isinstance(total_seconds, (int, float)) or isinstance(total_seconds, bool):
        raise ValueError("attempt total_seconds must be numeric")
    execution_position = panel.get("execution_position")
    replica = panel.get("rollout_replica")
    if not _is_int(execution_position) or not _is_int(replica):
        raise ValueError("attempt execution position and replica must be integers")

    return {
        "action_counts": action_counts,
        "admitted_calls": admitted_calls,
        "at_tool_cap": admitted_calls == tool_call_limit,
        "budget_rejected": budget_rejected,
        "capability": capability,
        "execution_position": execution_position,
        "failure_code": outcome.get("failure_code"),
        "family": family,
        "matching": capability == matching,
        "operation_aborted": operation_aborted,
        "output_tokens": output_tokens,
        "pre_execution_rejected": pre_execution_rejected,
        "rejected_calls": rejected_calls,
        "replica": replica,
        "request_shape_counts": request_shape_counts,
        "submission_attempted": submission_executions > 0,
        "submission_executions": submission_executions,
        "success": success,
        "task_id": task.get("task_id"),
        "termination": execution.get("termination_class"),
        "tokens": tokens,
        "total_seconds": float(total_seconds),
        "total_tokens": total_tokens,
    }


def _aggregate(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts: Counter[str] = Counter()
    request_shape_counts: Counter[str] = Counter()
    terminations: Counter[str] = Counter()
    failure_codes: Counter[str] = Counter()
    for attempt in attempts:
        action_counts.update(attempt["action_counts"])
        request_shape_counts.update(attempt["request_shape_counts"])
        terminations[str(attempt["termination"])] += 1
        if attempt["failure_code"] is not None:
            failure_codes[str(attempt["failure_code"])] += 1

    count = len(attempts)
    return {
        "action_counts": dict(sorted(action_counts.items())),
        "admitted_tool_calls": sum(item["admitted_calls"] for item in attempts),
        "attempts": count,
        "budget_rejected_attempts": sum(item["budget_rejected"] for item in attempts),
        "failure_codes": dict(sorted(failure_codes.items())),
        "mean_admitted_tool_calls": (
            round(sum(item["admitted_calls"] for item in attempts) / count, 3)
            if count
            else None
        ),
        "mean_output_tokens": (
            round(sum(item["output_tokens"] for item in attempts) / count, 3)
            if count
            else None
        ),
        "mean_total_seconds": (
            round(sum(item["total_seconds"] for item in attempts) / count, 3)
            if count
            else None
        ),
        "mean_total_tokens": (
            round(sum(item["total_tokens"] for item in attempts) / count, 3)
            if count
            else None
        ),
        "operation_aborted_attempts": sum(
            item["operation_aborted"] for item in attempts
        ),
        "pre_execution_rejected_attempts": sum(
            item["pre_execution_rejected"] for item in attempts
        ),
        "rejected_tool_calls": sum(item["rejected_calls"] for item in attempts),
        "request_shape_counts": dict(sorted(request_shape_counts.items())),
        "submission_attempts": sum(item["submission_attempted"] for item in attempts),
        "submission_executions": sum(
            item["submission_executions"] for item in attempts
        ),
        "successes": sum(item["success"] for item in attempts),
        "terminations": dict(sorted(terminations.items())),
        "tool_cap_attempts": sum(item["at_tool_cap"] for item in attempts),
        "total_output_tokens": sum(item["output_tokens"] for item in attempts),
        "total_seconds": round(sum(item["total_seconds"] for item in attempts), 3),
        "total_tokens": sum(item["total_tokens"] for item in attempts),
    }


def _task_family_summary(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for family in sorted(_FAMILY_TO_MATCHING_CAPABILITY):
        family_attempts = [item for item in attempts if item["family"] == family]
        by_capability = {}
        for capability in sorted({item["capability"] for item in family_attempts}):
            arm = [item for item in family_attempts if item["capability"] == capability]
            by_capability[capability] = {
                "all": _aggregate(arm),
                "failure": _aggregate([item for item in arm if not item["success"]]),
                "success": _aggregate([item for item in arm if item["success"]]),
            }
        summary[family] = {
            "matching_capability": _FAMILY_TO_MATCHING_CAPABILITY[family],
            "by_capability": by_capability,
        }
    return summary


def _slice_summary(attempts: list[dict[str, Any]], field: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for attempt in attempts:
        grouped[str(attempt[field])].append(attempt)
    return {key: _aggregate(grouped[key]) for key in sorted(grouped)}


def _table_cell_stability(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    table_attempts = [
        item for item in attempts if item["family"] == "table_filter_sort"
    ]
    summary: dict[str, Any] = {}
    for capability in sorted({item["capability"] for item in table_attempts}):
        cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for attempt in table_attempts:
            if attempt["capability"] == capability:
                cells[str(attempt["task_id"])].append(attempt)

        patterns: Counter[str] = Counter()
        discordant_attempts: list[dict[str, Any]] = []
        stable_failure_cells = 0
        stable_success_cells = 0
        for cell in cells.values():
            replicas = [item["replica"] for item in cell]
            if not all(isinstance(replica, int) for replica in replicas):
                raise ValueError("table cell replica must be an integer")
            if len(replicas) != len(set(replicas)):
                raise ValueError("table cell contains duplicate replicas")
            ordered = sorted(cell, key=lambda item: item["replica"])
            outcomes = [item["success"] for item in ordered]
            patterns["/".join("success" if value else "failure" for value in outcomes)] += 1
            if all(outcomes):
                stable_success_cells += 1
            elif not any(outcomes):
                stable_failure_cells += 1
            else:
                discordant_attempts.extend(ordered)

        summary[capability] = {
            "cells": len(cells),
            "discordant_cell_attempts": _aggregate(discordant_attempts),
            "discordant_cells": len(cells) - stable_success_cells - stable_failure_cells,
            "replica_outcome_patterns": dict(sorted(patterns.items())),
            "stable_failure_cells": stable_failure_cells,
            "stable_success_cells": stable_success_cells,
        }
    return summary


def build_diagnostic(package: str | Path) -> dict[str, Any]:
    """Build a deterministic aggregate diagnostic from a verified package."""
    root = Path(package).expanduser().resolve()
    audit = verify_package(root)
    if not audit.get("passed"):
        raise ValueError("dataset package verification failed")

    manifest_path = root / "MANIFEST.json"
    attempts_path = root / "data" / "attempts.jsonl"
    inventory_path = root / "raw" / "inventory.jsonl"
    gate_path = root / "analysis" / "gate-report.json"
    manifest = _read_json(manifest_path)
    rows = _read_jsonl(attempts_path)
    inventory = {entry["path"]: entry for entry in _read_jsonl(inventory_path)}
    gate_report = _read_json(gate_path)
    gate = gate_report.get("authoritative_gate")
    if not isinstance(gate, Mapping):
        raise ValueError("gate report has no authoritative_gate mapping")

    classified: list[dict[str, Any]] = []
    normalized_hashes: list[str] = []
    raw_event_hashes: list[str] = []
    for row in rows:
        raw = row.get("raw")
        if not isinstance(raw, Mapping) or not isinstance(raw.get("path"), str):
            raise ValueError("attempt row has no raw event path")
        normalized_rel = str(Path(raw["path"]).with_name("pi-events.normalized.json"))
        entry = inventory.get(normalized_rel)
        if not isinstance(entry, Mapping) or not _is_sha256(entry.get("sha256")):
            raise ValueError(f"normalized execution is absent from raw inventory: {normalized_rel}")
        normalized_path = root / normalized_rel
        if _sha256_file(normalized_path) != entry["sha256"]:
            raise ValueError(f"normalized execution hash mismatch: {normalized_rel}")
        normalized_hashes.append(str(entry["sha256"]))
        raw_entry = inventory.get(raw["path"])
        if not isinstance(raw_entry, Mapping) or not _is_sha256(raw_entry.get("sha256")):
            raise ValueError(f"raw execution is absent from raw inventory: {raw['path']}")
        raw_events_path = root / raw["path"]
        if _sha256_file(raw_events_path) != raw_entry["sha256"]:
            raise ValueError(f"raw execution hash mismatch: {raw['path']}")
        raw_event_hashes.append(str(raw_entry["sha256"]))
        classified.append(
            _classify_attempt(
                row,
                _read_json(normalized_path),
                _request_shape_counts(raw_events_path),
            )
        )

    table_nonmatching = [
        item
        for item in classified
        if item["family"] == "table_filter_sort" and not item["matching"]
    ]
    form_nonmatching = [
        item
        for item in classified
        if item["family"] == "form_entry_validation" and not item["matching"]
    ]
    table_matching = [
        item
        for item in classified
        if item["family"] == "table_filter_sort" and item["matching"]
    ]

    failures = [item for item in table_matching if not item["success"]]
    failed_task_ids = {item["task_id"] for item in failures}
    same_task_successes = [
        item
        for item in table_matching
        if item["success"]
        and item["task_id"] in failed_task_ids
        and item["request_shape_counts"]["bounded_semantic_table"]
    ]
    comparator = (
        min(
            same_task_successes,
            key=lambda item: (
                item["admitted_calls"],
                item["total_tokens"],
                item["replica"],
            ),
        )
        if same_task_successes
        else None
    )
    matching_table_failure = {
        "admitted_tool_calls": sum(item["admitted_calls"] for item in failures),
        "attempts": len(failures),
        "canonical_path_counts": dict(
            sorted(Counter(" > ".join(item["tokens"]) for item in failures).items())
        ),
        "operation_aborted_attempts": sum(
            item["operation_aborted"] for item in failures
        ),
        "result_submission_attempts": sum(
            item["submission_attempted"] for item in failures
        ),
        "same_task_bounded_success_comparator": {
            "admitted_tool_calls": (
                comparator["admitted_calls"] if comparator is not None else None
            ),
            "bounded_semantic_table_calls": (
                comparator["request_shape_counts"]["bounded_semantic_table"]
                if comparator is not None
                else 0
            ),
            "result_submission_attempts": (
                int(comparator["submission_attempted"])
                if comparator is not None
                else 0
            ),
        },
        "semantic_table_calls": sum(
            item["action_counts"]["semantic_table"] for item in failures
        ),
        "unbounded_semantic_table_calls": sum(
            item["request_shape_counts"]["unbounded_semantic_table"]
            for item in failures
        ),
        "tool_cap_attempts": sum(item["at_tool_cap"] for item in failures),
        "unbrowser_query_calls": sum(
            item["action_counts"]["unbrowser_query"] for item in failures
        ),
    }

    matching_attempts = [item for item in classified if item["matching"]]
    nonmatching_attempts = [item for item in classified if not item["matching"]]
    matching_successes = sum(item["success"] for item in matching_attempts)
    nonmatching_successes = sum(item["success"] for item in nonmatching_attempts)
    failure_query_calls = sum(
        item["action_counts"]["unbrowser_query"] for item in failures
    )
    failure_semantic_calls = sum(
        item["action_counts"]["semantic_table"] for item in failures
    )
    if failures:
        matching_failure_explanation = (
            f"The {len(failures)} matching table failure made "
            f"{failure_semantic_calls} semantic-table call, then "
            f"{failure_query_calls} generic queries, reached the admitted-call "
            "cap, and never submitted a result."
        )
    else:
        matching_failure_explanation = "No matching table failure was observed."

    expected_attempts = manifest.get("counts", {}).get("cells")
    if expected_attempts != len(classified):
        raise ValueError("diagnostic attempt count does not match package manifest")

    report = {
        "schema_version": REPORT_SCHEMA,
        "dataset_id": manifest.get("dataset_id"),
        "contract_hash": manifest.get("contract_hash"),
        "manifest_hash": manifest.get("identities", {}).get("manifest_hash"),
        "source_hashes": {
            "attempts_sha256": _sha256_file(attempts_path),
            "gate_report_sha256": _sha256_file(gate_path),
            "normalized_execution_count": len(normalized_hashes),
            "normalized_execution_hashes_sha256": hashlib.sha256(
                "\n".join(sorted(normalized_hashes)).encode("ascii")
            ).hexdigest(),
            "package_manifest_sha256": _sha256_file(manifest_path),
            "raw_event_count": len(raw_event_hashes),
            "raw_event_hashes_sha256": hashlib.sha256(
                "\n".join(sorted(raw_event_hashes)).encode("ascii")
            ).hexdigest(),
            "raw_inventory_sha256": _sha256_file(inventory_path),
        },
        "governance": {
            "eligibility": {
                "calibration": False,
                "development": False,
                "final": False,
                "training": False,
            },
            "frozen_gate_decision": gate.get("decision"),
            "frozen_gate_passed": gate.get("passed"),
            "governance_role": manifest.get("governance_role"),
            "task_role": manifest.get("task_role"),
        },
        "frozen_gate": {
            "decision": gate.get("decision"),
            "discordant_cell_count": gate.get("stability", {}).get(
                "discordant_cell_count"
            ),
            "favorable_form_count": gate.get("stability", {}).get(
                "favorable_form_count"
            ),
            "favorable_table_count": gate.get("stability", {}).get(
                "favorable_table_count"
            ),
            "reasons": list(gate.get("reasons", [])),
            "stable_form_only_count": gate.get("stability", {}).get(
                "stable_form_only_count"
            ),
            "stable_table_only_count": gate.get("stability", {}).get(
                "stable_table_only_count"
            ),
        },
        "aggregate": _aggregate(classified),
        "task_families": _task_family_summary(classified),
        "table_cell_stability": _table_cell_stability(classified),
        "exploratory_slices": {
            "nonmatching_form_on_table_by_execution_position": _slice_summary(
                table_nonmatching, "execution_position"
            ),
            "nonmatching_form_on_table_by_replica": _slice_summary(
                table_nonmatching, "replica"
            ),
            "nonmatching_table_on_form_by_execution_position": _slice_summary(
                form_nonmatching, "execution_position"
            ),
            "nonmatching_table_on_form_by_replica": _slice_summary(
                form_nonmatching, "replica"
            ),
        },
        "matching_table_failure": matching_table_failure,
        "interpretation": {
            "isolation_explanation": (
                "Both arms retained generic Unbrowser actions. The nonmatching "
                "form arm therefore had a slower manual path for table tasks, "
                "and any one success prevented strict stable-table-only status."
            ),
            "matching_table_failure_explanation": matching_failure_explanation,
            "routing_signal": (
                f"The matching specialist succeeded in {matching_successes} of "
                f"{len(matching_attempts)} attempts; the nonmatching specialist "
                f"succeeded in {nonmatching_successes} of "
                f"{len(nonmatching_attempts)}. This is "
                "exploratory routing evidence, not repeat-stable isolation or "
                "allocator evidence."
            ),
        },
        "warnings": [
            "Post-hoc descriptive error analysis only; it does not amend or override the frozen replication gate.",
            "Execution-position and replica slices are small and cannot support causal attribution.",
            "All rows remain T_canary / canary_excluded and ineligible for training, calibration, development, and final evaluation.",
            "The report is aggregate-only and omits messages, reasoning, result bodies, URLs, selectors, refs, and absolute paths.",
        ],
    }
    violations = privacy_scan(report)
    if violations:
        raise ValueError("diagnostic privacy scan failed: " + "; ".join(violations))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyreplab-m3-semantic-diagnostic",
        description="Build an aggregate privacy-safe semantic replication diagnostic.",
    )
    parser.add_argument("package", help="verified semantic dataset package")
    parser.add_argument("--output", help="optional JSON output path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_diagnostic(args.package)
        if args.output:
            write_json(Path(args.output).expanduser().resolve(), report)
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    except (OSError, ValueError, RuntimeError) as error:
        print(f"m3 semantic diagnostic error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["REPORT_SCHEMA", "build_diagnostic", "build_parser", "main"]
