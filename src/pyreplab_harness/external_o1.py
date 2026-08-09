"""Normalize the paired O1 SWE-bench releases into harness dataset rows.

This module deliberately separates source extraction from semantic import.  The
two Hugging Face releases are large Parquet files because they embed complete
conversations.  Callers should first project each release to JSONL containing
at least ``issue_name`` and ``resolved``, and project SWE-bench Verified to
JSONL containing ``instance_id``, ``problem_statement`` and ``repo``.  This
normalizer then:

* keeps only issue IDs with a boolean ``resolved`` outcome in both arms;
* joins both arms to one canonical pre-action problem statement;
* emits two rows with identical task features and different categorical
  treatment IDs;
* writes an explicit treatment registry and exclusion ledger; and
* reports the paired 2x2 outcome table before any model is trained.

The source ``success`` field is intentionally ignored.  In both releases it is
defaulted to ``False`` by conversion code rather than populated by the SWE-bench
evaluator.  ``resolved`` is the terminal outcome.

Example::

    python -m pyreplab_harness.external_o1 \
      baseline-projection.jsonl native-projection.jsonl \
      swebench-verified-projection.jsonl o1-paired.jsonl

No conversation, patch, test status, gold patch or evaluator detail enters
``model_input``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .io_utils import write_json

SCHEMA_VERSION = 1
DATASET_ID = "o1-swebench-verified-paired-v1"
SOURCE_LICENSE = "cc-by-4.0"
FAMILY = "swe_bench_verified"
VERIFIER_ID = "swe-bench-resolved"
VERIFIER_VERSION = "source-report-v1"

BASELINE_SOURCE_ID = "AlexCuadron/SWE-Bench-Verified-O1-reasoning-high-results"
BASELINE_SOURCE_REVISION = "a57cfefbb319d0119bb92b3c81d150653747d12e"
NATIVE_SOURCE_ID = (
    "AlexCuadron/SWE-Bench-Verified-O1-native-tool-calling-reasoning-high-results"
)
NATIVE_SOURCE_REVISION = "1467c51c3b4f506d212509489704a1f2934843a1"

BASELINE_TREATMENT_ID = "o1_baseline_text_protocol"
NATIVE_TREATMENT_ID = "o1_native_tool_interface"


def _descriptor_hash(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _treatment_descriptors() -> list[dict[str, Any]]:
    """Pinned closed-set treatment descriptors.

    Exact prompt bodies remain in the source release.  Their hashes, lengths,
    locations and source revisions make drift detectable without copying the
    multi-megabyte trajectory field into every normalized row.
    """

    descriptors = [
        {
            "treatment_id": BASELINE_TREATMENT_ID,
            "label": "O1 OpenHands textual tool protocol",
            "source_dataset_id": BASELINE_SOURCE_ID,
            "source_revision": BASELINE_SOURCE_REVISION,
            "source_url": f"https://huggingface.co/datasets/{BASELINE_SOURCE_ID}",
            "model": "o1-2024-12-17",
            "harness": "OpenHands CodeAct v2.2",
            "protocol": "textual_tool_protocol_with_worked_example",
            "system_prompt": {
                "length": 5070,
                "sha256": "fb3da48cc7f156876a98b0393a54edaf855c900ed6d1e35ddff4377900b7aea4",
                "source_field": "full_conversation_jsonl initial request messages[0]",
            },
            "tool_interface": {
                "transport": "text_protocol",
                "tools": ["execute_bash", "finish", "str_replace_editor"],
            },
            "known_bundle_scope": [
                "system_prompt",
                "worked_example",
                "tool_transport",
                "action_format",
                "response_parser",
            ],
        },
        {
            "treatment_id": NATIVE_TREATMENT_ID,
            "label": "O1 OpenHands native tool interface",
            "source_dataset_id": NATIVE_SOURCE_ID,
            "source_revision": NATIVE_SOURCE_REVISION,
            "source_url": f"https://huggingface.co/datasets/{NATIVE_SOURCE_ID}",
            "model": "o1-2024-12-17",
            "harness": "OpenHands CodeAct v2.2",
            "protocol": "native_tool_calling",
            "system_prompt": {
                "length": 617,
                "sha256": "410c821910805c18c46e11567c2fcaeaa67524eafd7e58ab735c33e0098ed81f",
                "source_field": "full_conversation_jsonl initial request messages[0]",
            },
            "tool_interface": {
                "transport": "native_tools",
                "source_field": "full_conversation_jsonl initial request kwargs.tools",
                "tools": ["execute_bash", "finish", "str_replace_editor"],
            },
            "known_bundle_scope": [
                "system_prompt",
                "tool_transport",
                "action_format",
                "response_parser",
            ],
        },
    ]
    for descriptor in descriptors:
        descriptor["registry_fingerprint"] = _descriptor_hash(descriptor)
        descriptor["treatment_version"] = descriptor["registry_fingerprint"][:16]
    return descriptors


def build_treatment_registry() -> dict[str, Any]:
    treatments = _treatment_descriptors()
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "source_license": SOURCE_LICENSE,
        "estimand": "closed-set prompt-plus-tool-interface bundle association",
        "treatments": treatments,
        "limitations": [
            "not an atomic prompt-only intervention",
            "one observed attempt per task and treatment",
            "reasoning_effort is declared by the releases but absent from sampled request fields",
            "external treatment effects do not identify Direct/Deliberate effects",
        ],
    }


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSON array or JSONL object stream.

    Parquet is rejected explicitly so callers cannot accidentally assume the
    standard library read a projected source when it did not.
    """

    source = Path(path).expanduser().resolve()
    if source.suffix.lower() == ".parquet":
        raise ValueError(
            f"Parquet projection is required before normalization: {source}; "
            "export only issue_name/resolved (and the canonical task fields) to JSONL"
        )
    if not source.is_file():
        raise FileNotFoundError(f"input does not exist: {source}")

    with source.open("r", encoding="utf-8") as handle:
        first = ""
        while True:
            character = handle.read(1)
            if not character:
                return []
            if not character.isspace():
                first = character
                break
        handle.seek(0)
        if first == "[":
            data = json.load(handle)
            if not isinstance(data, list):
                raise ValueError(f"expected a JSON array at {source}")
            rows = data
        else:
            rows = []
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"malformed JSON at {source}:{line_number}: {error}"
                    ) from error

    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"record {index} at {source} is not an object")
    return rows


def _index_unique(
    rows: Iterable[dict[str, Any]], keys: tuple[str, ...], label: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        raw_id = next((row.get(key) for key in keys if row.get(key) is not None), None)
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise ValueError(f"{label} row {index} has no non-empty ID in {keys}")
        row_id = raw_id.strip()
        if row_id in indexed:
            raise ValueError(f"duplicate {label} ID: {row_id}")
        indexed[row_id] = row
    return indexed


def _resolved(row: dict[str, Any], *, source: str, task_id: str) -> bool | None:
    value = row.get("resolved")
    if value is None:
        return None
    if type(value) is not bool:  # bool only; reject integers such as 0/1.
        raise ValueError(
            f"{source} resolved value for {task_id} must be bool or null, "
            f"got {type(value).__name__}"
        )
    return value


def grouped_task_split(task_id: str) -> str:
    """Stable whole-task 70/15/15 split for smoke use.

    Final architecture experiments should replace this with grouped outer folds
    and inner validation; every arm for a task still receives the same value.
    """

    bucket = int(hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def _task_text(task: dict[str, Any], task_id: str) -> str:
    value = task.get("problem_statement")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"canonical task {task_id} has no non-empty problem_statement")
    return value.strip()


def _task_repo(task: dict[str, Any], source_row: dict[str, Any]) -> str:
    value = task.get("repo") or source_row.get("project") or "unknown-repo"
    return str(value)


def _normalized_row(
    *,
    task_id: str,
    task: dict[str, Any],
    source_row: dict[str, Any],
    success: bool,
    treatment: dict[str, Any],
) -> dict[str, Any]:
    treatment_id = str(treatment["treatment_id"])
    treatment_version = str(treatment["treatment_version"])
    text = _task_text(task, task_id)
    repo = _task_repo(task, source_row)
    tool_interface = str(treatment["tool_interface"]["transport"])
    tools = sorted(str(tool) for tool in treatment["tool_interface"]["tools"])
    treatment_descriptor = {
        "text": (
            f"{treatment['label']}. Protocol: {treatment['protocol']}. "
            f"Known bundle scope: {', '.join(treatment['known_bundle_scope'])}. "
            f"Pinned system prompt sha256: {treatment['system_prompt']['sha256']}."
        ),
        "bundle_id": (
            f"{treatment_id}@{treatment_version}-"
            f"{str(treatment['registry_fingerprint'])[:8]}"
        ),
        "tool_interface": tool_interface,
        "allowed_tools_signature": ",".join(tools),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "source_dataset_id": treatment["source_dataset_id"],
        "source_revision": treatment["source_revision"],
        "source_license": SOURCE_LICENSE,
        "source_outcome_field": "resolved",
        "task_id": task_id,
        "pair_id": task_id,
        "split_group": task_id,
        "attempt_id": f"{treatment_id}::{task_id}",
        "trial_index": 0,
        "design": "paired_complete",
        "family": FAMILY,
        "template_id": repo,
        "difficulty": "unknown",
        "prompt": text,
        "contract": [],
        "public_metadata": {},
        "policy_id": treatment_id,
        "policy_version": treatment_version,
        "treatment_registry_fingerprint": treatment["registry_fingerprint"],
        "split": grouped_task_split(task_id),
        "verified_success": success,
        "failure_code": None if success else "swe_bench_unresolved",
        "verifier_id": VERIFIER_ID,
        "verifier_version": VERIFIER_VERSION,
        "model_input": {
            "text": text,
            "family": FAMILY,
            "template_id": repo,
            "difficulty": "unknown",
            "public_metadata": {},
            "policy_id": treatment_id,
            "policy_version": treatment_version,
            "treatment": treatment_descriptor,
        },
    }


def build_o1_dataset(
    baseline_rows: list[dict[str, Any]],
    native_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build normalized rows, registry, exclusion ledger and summary."""

    baseline = _index_unique(baseline_rows, ("issue_name",), "baseline")
    native = _index_unique(native_rows, ("issue_name",), "native")
    tasks = _index_unique(task_rows, ("instance_id", "issue_name"), "canonical task")

    registry = build_treatment_registry()
    treatment_by_id = {
        str(item["treatment_id"]): item for item in registry["treatments"]
    }

    normalized: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    paired = {
        "both_pass": 0,
        "baseline_only_pass": 0,
        "native_only_pass": 0,
        "both_fail": 0,
    }

    for task_id in sorted(set(baseline) | set(native)):
        reasons: list[str] = []
        baseline_row = baseline.get(task_id)
        native_row = native.get(task_id)
        baseline_y: bool | None = None
        native_y: bool | None = None

        if baseline_row is None:
            reasons.append("PAIR_EXCLUDE_BASELINE_ROW_ABSENT")
        else:
            baseline_y = _resolved(baseline_row, source="baseline", task_id=task_id)
            if baseline_y is None:
                reasons.append("PAIR_EXCLUDE_BASELINE_REPORT_MISSING")

        if native_row is None:
            reasons.append("PAIR_EXCLUDE_NATIVE_ROW_ABSENT")
        else:
            native_y = _resolved(native_row, source="native", task_id=task_id)
            if native_y is None:
                reasons.append("PAIR_EXCLUDE_NATIVE_REPORT_MISSING")

        task = tasks.get(task_id)
        if task is None:
            reasons.append("PAIR_EXCLUDE_CANONICAL_TASK_MISSING")

        if reasons:
            reasons = sorted(set(reasons))
            excluded.append({"task_id": task_id, "reasons": reasons})
            reason_counts.update(reasons)
            continue

        assert baseline_row is not None and native_row is not None and task is not None
        assert baseline_y is not None and native_y is not None
        if baseline_y and native_y:
            paired["both_pass"] += 1
        elif baseline_y:
            paired["baseline_only_pass"] += 1
        elif native_y:
            paired["native_only_pass"] += 1
        else:
            paired["both_fail"] += 1

        normalized.append(
            _normalized_row(
                task_id=task_id,
                task=task,
                source_row=baseline_row,
                success=baseline_y,
                treatment=treatment_by_id[BASELINE_TREATMENT_ID],
            )
        )
        normalized.append(
            _normalized_row(
                task_id=task_id,
                task=task,
                source_row=native_row,
                success=native_y,
                treatment=treatment_by_id[NATIVE_TREATMENT_ID],
            )
        )

    normalized.sort(key=lambda row: (row["task_id"], row["policy_id"]))
    exclusion_ledger = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "excluded_task_count": len(excluded),
        "reason_counts": dict(sorted(reason_counts.items())),
        "excluded": excluded,
    }
    paired_tasks = sum(paired.values())
    summary = {
        "dataset_id": DATASET_ID,
        "baseline_source_rows": len(baseline),
        "native_source_rows": len(native),
        "canonical_tasks": len(tasks),
        "paired_tasks": paired_tasks,
        "rows": len(normalized),
        "excluded_tasks": len(excluded),
        "paired_outcomes": paired,
        "success": {
            "baseline": paired["both_pass"] + paired["baseline_only_pass"],
            "native": paired["both_pass"] + paired["native_only_pass"],
        },
        "split_rows": dict(
            sorted(Counter(str(row["split"]) for row in normalized).items())
        ),
    }
    return normalized, registry, exclusion_ledger, summary


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_o1_dataset(
    baseline_path: str | Path,
    native_path: str | Path,
    tasks_path: str | Path,
    output_path: str | Path,
    *,
    registry_path: str | Path | None = None,
    exclusions_path: str | Path | None = None,
) -> dict[str, Any]:
    output = Path(output_path).expanduser().resolve()
    registry_target = (
        Path(registry_path).expanduser().resolve()
        if registry_path
        else output.with_name(f"{output.stem}.treatments.json")
    )
    exclusions_target = (
        Path(exclusions_path).expanduser().resolve()
        if exclusions_path
        else output.with_name(f"{output.stem}.exclusions.json")
    )
    rows, registry, exclusions, summary = build_o1_dataset(
        read_records(baseline_path), read_records(native_path), read_records(tasks_path)
    )
    _write_jsonl(output, rows)
    write_json(registry_target, registry)
    write_json(exclusions_target, exclusions)
    return {
        **summary,
        "output_path": str(output),
        "registry_path": str(registry_target),
        "exclusions_path": str(exclusions_target),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyreplab-harness-external-o1",
        description="Normalize paired O1 SWE-bench release projections.",
    )
    parser.add_argument("baseline", help="baseline release JSON/JSONL projection")
    parser.add_argument("native", help="native-tools release JSON/JSONL projection")
    parser.add_argument("tasks", help="canonical SWE-bench task JSON/JSONL projection")
    parser.add_argument("output", help="normalized harness JSONL output")
    parser.add_argument("--registry", help="treatment registry output JSON path")
    parser.add_argument("--exclusions", help="exclusion ledger output JSON path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = write_o1_dataset(
            args.baseline,
            args.native,
            args.tasks,
            args.output,
            registry_path=args.registry,
            exclusions_path=args.exclusions,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BASELINE_TREATMENT_ID",
    "DATASET_ID",
    "NATIVE_TREATMENT_ID",
    "build_o1_dataset",
    "build_parser",
    "build_treatment_registry",
    "grouped_task_split",
    "main",
    "read_records",
    "write_o1_dataset",
]
