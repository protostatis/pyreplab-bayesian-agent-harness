"""Expected-failure/expected-success runner for the live Pi + Unbrowser smoke."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .orchestrator import build_parser, run_smoke


NEGATIVE_POLICY_ID = "unbrowser-wrong-selector"
POSITIVE_POLICY_ID = "unbrowser-correct-selector"
EXPECTED_SELECTORS = {
    NEGATIVE_POLICY_ID: "p",
    POSITIVE_POLICY_ID: "h1",
}
EXPECTED_VERIFICATION = {
    NEGATIVE_POLICY_ID: False,
    POSITIVE_POLICY_ID: True,
}


def _evaluate_smoke(result: dict[str, Any]) -> dict[str, Any]:
    """Check polarity and exact read-only tool calls for both controls."""

    attempts = result.get("attempts")
    if not isinstance(attempts, dict):
        return {"success": False, "problems": ["result omitted treatment attempts"]}

    by_policy: dict[str, dict[str, Any]] = {}
    for attempt in attempts.values():
        if not isinstance(attempt, dict):
            continue
        policy = attempt.get("policy")
        if isinstance(policy, dict) and isinstance(policy.get("id"), str):
            by_policy[policy["id"]] = attempt

    problems: list[str] = []
    controls: dict[str, Any] = {}
    for policy_id, expected_selector in EXPECTED_SELECTORS.items():
        attempt = by_policy.get(policy_id)
        if attempt is None:
            problems.append(f"missing attempt for {policy_id}")
            continue
        verification = attempt.get("verification")
        actual_success = (
            verification.get("success") if isinstance(verification, dict) else None
        )
        expected_success = EXPECTED_VERIFICATION[policy_id]
        if actual_success is not expected_success:
            problems.append(
                f"{policy_id} verification was {actual_success!r}; "
                f"expected {expected_success!r}"
            )
        if attempt.get("pi_return_code") != 0:
            problems.append(
                f"{policy_id} Pi exited with {attempt.get('pi_return_code')!r}"
            )

        trajectory = attempt.get("trajectory")
        trace = trajectory.get("tool_trace") if isinstance(trajectory, dict) else None
        unbrowser_calls = []
        if isinstance(trace, list):
            for item in trace:
                if not isinstance(item, dict) or item.get("tool_name") != "unbrowser":
                    continue
                details = item.get("details")
                unbrowser_calls.append(details if isinstance(details, dict) else {})
        observed = [
            (call.get("action"), call.get("selector")) for call in unbrowser_calls
        ]
        expected_calls = [("navigate", None), ("text", expected_selector)]
        if observed != expected_calls:
            problems.append(
                f"{policy_id} Unbrowser calls were {observed!r}; "
                f"expected {expected_calls!r}"
            )
        if any(call.get("error") for call in unbrowser_calls):
            problems.append(f"{policy_id} reported an Unbrowser adapter error")
        versions = {
            call.get("runtime_version")
            for call in unbrowser_calls
            if call.get("runtime_version")
        }
        if not versions:
            problems.append(f"{policy_id} omitted the Unbrowser runtime version")

        controls[policy_id] = {
            "expected_verification": expected_success,
            "actual_verification": actual_success,
            "unbrowser_calls": observed,
            "runtime_versions": sorted(str(version) for version in versions),
            "failure_code": (
                verification.get("failure_code")
                if isinstance(verification, dict)
                else None
            ),
        }

    return {
        "success": not problems,
        "controls": controls,
        "problems": problems,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[2]
    args.family = "unbrowser"
    args.difficulty = "easy"
    args.pair = False
    args.attempt_id = None
    args.treatment_registry = str(
        project_root / "policies" / "unbrowser-smoke-treatments.json"
    )
    args.treatments = "all"

    try:
        result = run_smoke(args)
        assessment = _evaluate_smoke(result)
    except (OSError, RuntimeError, subprocess.TimeoutExpired, ValueError) as error:
        print(
            json.dumps({"ok": False, "error": str(error)}, sort_keys=True),
            file=sys.stderr,
        )
        return 1

    print(
        json.dumps(
            {"ok": assessment["success"], "assessment": assessment, **result},
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if assessment["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
