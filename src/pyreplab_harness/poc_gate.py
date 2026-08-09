"""Descriptive completion gate for a paired terminal-gym proof of concept.

This gate validates experiment plumbing and outcome diversity before the small
neural-model demonstration runs. It is not a statistical hypothesis test and
must not be used to claim allocator effectiveness.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .io_utils import write_json

_POLICIES = ("direct", "deliberate")


def _read_records(path: str | Path) -> tuple[list[dict[str, Any]], list[str]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"batch output does not exist: {source}")
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            errors.append(f"line {line_number}: malformed JSON: {error.msg}")
            continue
        if not isinstance(value, dict):
            errors.append(f"line {line_number}: record is not an object")
            continue
        records.append(value)
    return records, errors


def evaluate_gate(
    batch_path: str | Path,
    *,
    expected_jobs: int = 24,
    policy_version: str = "4",
    min_disagreement: float = 0.15,
) -> dict[str, Any]:
    """Return a JSON-safe descriptive gate report for one paired batch."""
    if expected_jobs <= 0:
        raise ValueError("expected_jobs must be positive")
    if not math.isfinite(min_disagreement) or not 0 <= min_disagreement <= 1:
        raise ValueError("min_disagreement must be between 0 and 1")

    records, parse_errors = _read_records(batch_path)
    reasons = list(parse_errors)
    seen_keys: set[str] = set()
    matrix: Counter[str] = Counter()
    successes: Counter[str] = Counter()
    attempts: Counter[str] = Counter()
    paired = 0
    total_wall_seconds = 0.0

    for index, record in enumerate(records, 1):
        key = record.get("key")
        label = str(key) if isinstance(key, str) else f"record {index}"
        if not isinstance(key, str):
            reasons.append(f"{label}: missing string key")
        elif key in seen_keys:
            reasons.append(f"{label}: duplicate completed job key")
        else:
            seen_keys.add(key)

        if record.get("status") != "completed":
            reasons.append(f"{label}: status is not completed")
            continue
        if record.get("mode") != "pair":
            reasons.append(f"{label}: mode is not pair")
            continue
        if str(record.get("policy_version")) != policy_version:
            reasons.append(
                f"{label}: policy version {record.get('policy_version')!r}, "
                f"expected {policy_version!r}"
            )

        duration = record.get("duration_seconds")
        if isinstance(duration, (int, float)) and not isinstance(duration, bool):
            if math.isfinite(float(duration)) and duration >= 0:
                total_wall_seconds += float(duration)

        result = record.get("result")
        pair_attempts = result.get("attempts") if isinstance(result, Mapping) else None
        if not isinstance(pair_attempts, Mapping):
            reasons.append(f"{label}: missing paired attempts")
            continue

        outcomes: dict[str, bool] = {}
        pair_valid = True
        for policy in _POLICIES:
            attempt = pair_attempts.get(policy)
            if not isinstance(attempt, Mapping):
                reasons.append(f"{label}: missing {policy} attempt")
                pair_valid = False
                continue
            spec = attempt.get("policy")
            observed_version = spec.get("version") if isinstance(spec, Mapping) else None
            if str(observed_version) != policy_version:
                reasons.append(
                    f"{label}: {policy} attempt policy version "
                    f"{observed_version!r}, expected {policy_version!r}"
                )
                pair_valid = False
            verification = attempt.get("verification")
            outcome = verification.get("success") if isinstance(verification, Mapping) else None
            if not isinstance(outcome, bool):
                reasons.append(f"{label}: {policy} attempt lacks a boolean outcome")
                pair_valid = False
                continue
            outcomes[policy] = outcome

        if not pair_valid or len(outcomes) != 2:
            continue
        paired += 1
        for policy, outcome in outcomes.items():
            attempts[policy] += 1
            successes[policy] += int(outcome)
        direct, deliberate = outcomes["direct"], outcomes["deliberate"]
        if direct and deliberate:
            matrix["both_ok"] += 1
        elif direct:
            matrix["direct_only"] += 1
        elif deliberate:
            matrix["deliberate_only"] += 1
        else:
            matrix["both_fail"] += 1

    if len(records) != expected_jobs:
        reasons.append(f"found {len(records)} jobs, expected {expected_jobs}")
    if paired != expected_jobs:
        reasons.append(f"found {paired} usable pairs, expected {expected_jobs}")

    disagreement_count = matrix["direct_only"] + matrix["deliberate_only"]
    disagreement_rate = disagreement_count / paired if paired else None
    if disagreement_rate is None or disagreement_rate < min_disagreement:
        rendered = "none" if disagreement_rate is None else f"{disagreement_rate:.3f}"
        reasons.append(
            f"paired disagreement {rendered} is below required {min_disagreement:.3f}"
        )

    policy_rates: dict[str, float | None] = {}
    for policy in _POLICIES:
        count = attempts[policy]
        rate = successes[policy] / count if count else None
        policy_rates[policy] = rate
        if rate is None or rate <= 0 or rate >= 1:
            reasons.append(f"{policy} success rate is degenerate: {rate!r}")

    return {
        "gate": "terminal-poc-v1",
        "passed": not reasons,
        "batch_basename": Path(batch_path).expanduser().resolve().name,
        "expected_jobs": expected_jobs,
        "records": len(records),
        "usable_pairs": paired,
        "policy_version": policy_version,
        "min_disagreement": min_disagreement,
        "disagreement_count": disagreement_count,
        "disagreement_rate": disagreement_rate,
        "matrix": {
            name: matrix[name]
            for name in ("both_ok", "direct_only", "deliberate_only", "both_fail")
        },
        "policy_success_rate": policy_rates,
        "total_wall_seconds": round(total_wall_seconds, 3),
        "reasons": reasons,
        "warning": (
            "Descriptive proof-of-concept gate only; passing does not establish "
            "allocator effectiveness or statistical significance."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyreplab-harness-poc-gate",
        description="Validate completion and outcome diversity for a paired POC batch.",
    )
    parser.add_argument("batch", help="batch JSONL output")
    parser.add_argument("--expected-jobs", type=int, default=24)
    parser.add_argument("--policy-version", default="4")
    parser.add_argument("--min-disagreement", type=float, default=0.15)
    parser.add_argument("--output", help="optional report JSON path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = evaluate_gate(
            args.batch,
            expected_jobs=args.expected_jobs,
            policy_version=args.policy_version,
            min_disagreement=args.min_disagreement,
        )
        payload = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        if args.output:
            target = Path(args.output).expanduser().resolve()
            write_json(target, report)
        print(payload, end="")
        return 0 if report["passed"] else 2
    except (OSError, ValueError) as error:
        print(f"gate error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "evaluate_gate", "main"]
