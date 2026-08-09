"""Python repair task family for the Pyreplab harness gym.

Each task is a deterministic, seeded "repair the buggy pure function" puzzle
inside a small package-shaped workspace:

- public workspace: ``TASK.md``, a buggy ``solution.py``, and an optional
  public smoke test ``test_public.py`` (API-only; may pass while the
  implementation is still wrong);
- private bundle: the reference implementation, hidden cases, and a generated
  verifier runner (never mounted into the agent's workspace).

Bug categories seeded across templates: boundary conditions (off-by-one),
wrong boolean condition, and wrong key/aggregation.

Safety and determinism contract
-------------------------------
- Submitted code is **never** imported in the harness or test process.
- Verification runs the generated runner inside the Bubblewrap sandbox
  (``sandbox.py``) under a systemd user scope with resource limits. The
  submitted module therefore executes with:
  - the agent workspace mounted **read-only** at ``/workspace``;
  - the private verifier bundle mounted **read-only** at a separate in-sandbox
    path ``/private`` (never into the agent workspace);
  - a throwaway output directory mounted **writable** at ``/output``;
  - unshared network, mount, PID, IPC, UTS and user namespaces;
  - no ``/home`` mount, a fresh ``/tmp``/``/dev``/``/proc`` and a cleared
    environment rebuilt from an explicit allowlist.
- The runner is generated at task-build time, so the private bundle is
  self-contained and auditable. It runs with in-sandbox paths only; the
  submitted module never receives host filesystem paths (the only unavoidable
  leak is the ``bwrap`` argv visible under the sandbox's own ``/proc``, which
  contains mount sources, not private data contents).
- The submitted module necessarily sees the hidden case data because the
  verifier imports it in the same process as the runner; instead, every piece
  of diagnostics returned after verification is capped (stdout/stderr tails,
  failure entries, detail strings) and the report file is size-limited, so a
  misbehaving module cannot exfiltrate or flood results.
- Memory, task, CPU and wall-clock limits are enforced through
  ``systemd-run --user`` properties. On timeout the process group and the
  transient unit are killed; no process survives verification.
- There is **no silent unsafe fallback**: if ``bwrap`` or a systemd user
  session is unavailable, the verifier returns a distinct ``sandbox_unavailable``
  result instead of executing submitted code on the host.
- ``verify_python_repair_attempt`` writes ``verification.json`` into the
  attempt directory and updates the attempt metadata exactly like the other
  gym families.

Verifier failure taxonomy
-------------------------
``missing_file``       the required module is not present in the workspace
``syntax_error``       the module does not compile
``runtime_error``      the module fails while being imported, the verifier
                       itself cannot complete, or the sandbox run produced no
                       usable report
``test_failure``       the function ran but produced wrong results and/or
                       raised while evaluating hidden cases
``timeout``            the sandboxed subprocess exceeded the wall-clock limit
``sandbox_unavailable`` bwrap and/or a systemd user session are missing (or no
                       interpreter reachable inside the sandbox); no
                       host-execution fallback is attempted
``None``               success (all hidden cases passed)
"""

from __future__ import annotations

import random
import re
import shlex
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from .artifact_gym import (  # noqa: F401  (generic helpers shared with the artifact gym)
    load_attempt,
    load_task,
    prepare_attempt,
)
from .contracts import TaskSpec, VerificationResult
from .io_utils import read_json, write_json
from .sandbox import (
    ISOLATED_RUNTIME_PATHS,
    BubblewrapSandbox,
    SandboxBind,
    SandboxLimits,
    SandboxUnavailableError,
    bwrap_available,
    sandbox_available,
    sandbox_python_interpreter,
    systemd_user_available,
)

GENERATOR_VERSION = "python-repair-v1"
VERIFIER_ID = "python-repair-hidden-tests"
VERIFIER_VERSION = "1"
DIFFICULTIES = {"easy", "medium", "hard"}
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

MODULE_NAME = "solution"
RUNNER_FILE = "verify_runner.py"
HIDDEN_CASES_FILE = "hidden_cases.json"
REFERENCE_FILE = "reference.py"
REPORT_FILE = "verification.out.json"

_DIAGNOSTIC_TAIL = 2000
_DETAIL_LIMIT = 2000
_REPORT_MAX_BYTES = 2_000_000

_CASE_COUNTS = {"easy": 4, "medium": 6, "hard": 8}


def _safe_id(value: str, label: str) -> str:
    if not SAFE_ID.fullmatch(value):
        raise ValueError(f"invalid {label}: {value!r}")
    return value


def _root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _task_dir(root: Path, task_id: str) -> Path:
    return root / "tasks" / _safe_id(task_id, "task id")


def _attempt_dir(root: Path, attempt_id: str) -> Path:
    return root / "attempts" / _safe_id(attempt_id, "attempt id")


# --------------------------------------------------------------------------
# Template sources.  The buggy file keeps the correct-sounding docstring; the
# flaw is in the code only.  `{min_age}` is interpolated for eligibility.
# --------------------------------------------------------------------------

_RANGE_REFERENCE = '''\
"""Repair task module.

Contract: count_in_range(values, low, high) returns the number of integers v
in values with low <= v <= high. Both endpoints are inclusive.
"""


def count_in_range(values, low, high):
    """Return how many integers in values lie in [low, high], inclusive."""
    return sum(1 for v in values if low <= v <= high)
'''

_RANGE_BUGGY = '''\
"""Repair task module.

Contract: count_in_range(values, low, high) returns the number of integers v
in values with low <= v <= high. Both endpoints are inclusive.
"""


def count_in_range(values, low, high):
    """Return how many integers in values lie in [low, high], inclusive."""
    return sum(1 for v in values if low < v < high)
'''

_RANGE_BUGGY_HARD = '''\
"""Repair task module.

Contract: count_in_range(values, low, high) returns the number of integers v
in values with low <= v <= high. Both endpoints are inclusive.
"""


def count_in_range(values, low, high):
    """Return how many integers in values lie in [low, high], inclusive."""
    return sum(1 for v in values if v < low or v > high)
'''

_ELIGIBILITY_REFERENCE = '''\
"""Repair task module.

Contract: is_eligible(age, has_id) returns True only when age >= MIN_AGE and
the caller holds a valid id.
"""

MIN_AGE = {min_age}


def is_eligible(age, has_id):
    """Return True when age is at least MIN_AGE and the caller has a valid id."""
    return age >= MIN_AGE and has_id
'''

_ELIGIBILITY_BUGGY = '''\
"""Repair task module.

Contract: is_eligible(age, has_id) returns True only when age >= MIN_AGE and
the caller holds a valid id.
"""

MIN_AGE = {min_age}


def is_eligible(age, has_id):
    """Return True when age is at least MIN_AGE and the caller has a valid id."""
    return age >= MIN_AGE or has_id
'''

_ELIGIBILITY_BUGGY_HARD = '''\
"""Repair task module.

Contract: is_eligible(age, has_id) returns True only when age >= MIN_AGE and
the caller holds a valid id.
"""

MIN_AGE = {min_age}


def is_eligible(age, has_id):
    """Return True when age is at least MIN_AGE and the caller has a valid id."""
    return age > MIN_AGE and has_id
'''

_TOTAL_BILL_REFERENCE = '''\
"""Repair task module.

Contract: total_bill(orders) returns the total cost in cents across all
orders, where every order is a dict with keys "qty" and "unit_price_cents".
"""


def total_bill(orders):
    """Return the total price in cents, summing qty * unit_price_cents."""
    return sum(order["qty"] * order["unit_price_cents"] for order in orders)
'''

_TOTAL_BILL_BUGGY = '''\
"""Repair task module.

Contract: total_bill(orders) returns the total cost in cents across all
orders, where every order is a dict with keys "qty" and "unit_price_cents".
"""


def total_bill(orders):
    """Return the total price in cents, summing qty * unit_price_cents."""
    return sum(order["unit_price_cents"] for order in orders)
'''

_TOTAL_BILL_BUGGY_HARD = '''\
"""Repair task module.

Contract: total_bill(orders) returns the total cost in cents across all
orders, where every order is a dict with keys "qty" and "unit_price_cents".
"""


def total_bill(orders):
    """Return the total price in cents, summing qty * unit_price_cents."""
    return sum(order["qty"] * order["price_cents"] for order in orders)
'''

_RANGE_PUBLIC_TEST = '''\
import unittest

import solution


class PublicSmokeTest(unittest.TestCase):
    def test_mid_range_values(self):
        self.assertEqual(solution.count_in_range([1, 2, 3], 0, 10), 3)

    def test_empty_list(self):
        self.assertEqual(solution.count_in_range([], 0, 10), 0)


if __name__ == "__main__":
    unittest.main()
'''

_ELIGIBILITY_PUBLIC_TEST = '''\
import unittest

import solution


class PublicSmokeTest(unittest.TestCase):
    def test_old_enough_with_id(self):
        self.assertTrue(solution.is_eligible(30, True))


if __name__ == "__main__":
    unittest.main()
'''

_TOTAL_BILL_PUBLIC_TEST = '''\
import unittest

import solution


class PublicSmokeTest(unittest.TestCase):
    def test_single_order_quantity_one(self):
        orders = [{"item": "a", "qty": 1, "unit_price_cents": 100}]
        self.assertEqual(solution.total_bill(orders), 100)


if __name__ == "__main__":
    unittest.main()
'''

_RUNNER_SOURCE = '''\
"""Private verifier runner for Python repair tasks.

Executed in a subprocess as ``python -I -S -B verify_runner.py <workspace>
<module_name> <function_name> <cases_json> <output_json>`` inside the
Bubblewrap sandbox, where ``<workspace>`` is the read-only in-sandbox path
``/workspace``, the cases come from the read-only ``/private`` bundle and the
report is written into the writable ``/output`` directory.

Submitted code is imported only here, never in the harness or test process.
The runner writes a JSON report and exits; the harness maps the report status
to a VerificationResult.
"""

from __future__ import annotations

import importlib
import json
import sys


def _write_report(output_path, report):
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, sort_keys=True)


def main() -> int:
    workspace, module_name, function_name, cases_path, output_path = sys.argv[1:6]
    sys.path.insert(0, workspace)

    try:
        with open(cases_path, "r", encoding="utf-8") as handle:
            cases = json.load(handle)
    except Exception as error:
        _write_report(
            output_path,
            {
                "status": "runtime_error",
                "detail": f"verifier could not read hidden cases: {type(error).__name__}: {error}",
            },
        )
        return 0

    try:
        module = importlib.import_module(module_name)
    except SyntaxError as error:
        filename = getattr(error, "filename", None) or f"{module_name}.py"
        lineno = getattr(error, "lineno", None)
        location = f"{filename}:{lineno}" if lineno is not None else filename
        _write_report(output_path, {"status": "syntax_error", "detail": f"{location}: {error.msg}"})
        return 0
    except Exception as error:
        _write_report(
            output_path,
            {
                "status": "runtime_error",
                "detail": f"import failed: {type(error).__name__}: {error}",
            },
        )
        return 0

    function = getattr(module, function_name, None)
    if not callable(function):
        _write_report(
            output_path,
            {
                "status": "runtime_error",
                "detail": f"module {module_name!r} has no callable {function_name!r}",
            },
        )
        return 0

    failures = []
    for index, case in enumerate(cases):
        args = list(case["args"])
        expected = case["expected"]
        try:
            actual = function(*args)
        except Exception as error:
            failures.append(
                {"case": index, "args": args, "error": f"{type(error).__name__}: {error}"}
            )
            continue
        if actual != expected:
            failures.append({"case": index, "args": args, "expected": expected, "actual": actual})

    if failures:
        report = {
            "status": "test_failure",
            "cases_run": len(cases),
            "cases_passed": len(cases) - len(failures),
            "failures": failures,
        }
    else:
        report = {
            "status": "success",
            "cases_run": len(cases),
            "cases_passed": len(cases),
            "failures": [],
        }
    _write_report(output_path, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


# --------------------------------------------------------------------------
# Seeded per-template builders.
# --------------------------------------------------------------------------


def _range_bounds(rng: random.Random, difficulty: str) -> tuple[int, int]:
    if difficulty == "easy":
        low = rng.randint(0, 5)
        span = rng.randint(5, 15)
    elif difficulty == "medium":
        low = rng.randint(-20, 20)
        span = rng.randint(10, 40)
    else:
        low = rng.randint(-80, 80)
        span = rng.randint(20, 120)
    return low, low + span


def _range_cases(rng: random.Random, difficulty: str, low: int, high: int) -> list[dict[str, Any]]:
    del rng  # Bounds were sampled before; cases are deterministic from (low, high).
    mid = low + (high - low) // 2  # Strictly inside because span >= 5.
    ordered = [
        {"args": [[low], low, high], "expected": 1},
        {"args": [[high], low, high], "expected": 1},
        {"args": [[low - 1], low, high], "expected": 0},
        {"args": [[high + 1], low, high], "expected": 0},
        {"args": [[], low, high], "expected": 0},
        {"args": [[low, high, low - 1, high + 1, mid], low, high], "expected": 3},
        {"args": [[mid, mid, low, high], low, high], "expected": 4},
        {"args": [[low - 10, high + 10], low, high], "expected": 0},
    ]
    return ordered[:_CASE_COUNTS[difficulty]]


def _build_range(rng: random.Random, difficulty: str) -> dict[str, Any]:
    low, high = _range_bounds(rng, difficulty)
    return {
        "template_id": "range-boundary-v1",
        "bug_category": "boundary",
        "function_name": "count_in_range",
        "signature": "count_in_range(values, low, high) -> int",
        "description": (
            "`count_in_range(values, low, high)` must return the number of integers `v` in "
            "`values` with `low <= v <= high`. Both endpoints are inclusive and the list "
            "may be empty."
        ),
        "reference_source": _RANGE_REFERENCE,
        "buggy_source": _RANGE_BUGGY,
        "buggy_hard_source": _RANGE_BUGGY_HARD,
        "public_test_source": _RANGE_PUBLIC_TEST,
        "cases": _range_cases(rng, difficulty, low, high),
    }


def _min_age(rng: random.Random, difficulty: str) -> int:
    if difficulty == "easy":
        return rng.randint(16, 18)
    if difficulty == "medium":
        return rng.randint(16, 21)
    return rng.randint(16, 24)


def _eligibility_cases(
    rng: random.Random, difficulty: str, min_age: int
) -> list[dict[str, Any]]:
    del rng  # MIN_AGE was sampled before; cases are deterministic from it.
    ordered = [
        {"args": [min_age, True], "expected": True},
        {"args": [min_age - 1, True], "expected": False},
        {"args": [min_age, False], "expected": False},
        {"args": [min_age + 10, True], "expected": True},
        {"args": [min_age + 10, False], "expected": False},
        {"args": [0, False], "expected": False},
        {"args": [min_age + 1, True], "expected": True},
        {"args": [min_age - 1, False], "expected": False},
    ]
    return ordered[:_CASE_COUNTS[difficulty]]


def _build_eligibility(rng: random.Random, difficulty: str) -> dict[str, Any]:
    min_age = _min_age(rng, difficulty)
    return {
        "template_id": "eligibility-v1",
        "bug_category": "wrong_condition",
        "function_name": "is_eligible",
        "signature": "is_eligible(age, has_id) -> bool",
        "description": (
            "`is_eligible(age, has_id)` must return `True` only when `age >= MIN_AGE` and "
            "the caller holds a valid id. `MIN_AGE` is a module constant you must not change."
        ),
        "reference_source": _ELIGIBILITY_REFERENCE.format(min_age=min_age),
        "buggy_source": _ELIGIBILITY_BUGGY.format(min_age=min_age),
        "buggy_hard_source": _ELIGIBILITY_BUGGY_HARD.format(min_age=min_age),
        "public_test_source": _ELIGIBILITY_PUBLIC_TEST,
        "cases": _eligibility_cases(rng, difficulty, min_age),
    }


def _total_bill_cases(rng: random.Random, difficulty: str) -> list[dict[str, Any]]:
    per_case = {"easy": 2, "medium": 3, "hard": 4}[difficulty]
    cases: list[dict[str, Any]] = []
    for index in range(_CASE_COUNTS[difficulty]):
        orders = []
        for slot in range(per_case):
            qty = rng.randint(1, 5)
            price = rng.randint(50, 2000)
            orders.append({"item": f"sku-{index}-{slot}", "qty": qty, "unit_price_cents": price})
        if index == 0:
            orders[0]["qty"] = 2  # Guarantee the aggregation bug is visible.
        expected = sum(order["qty"] * order["unit_price_cents"] for order in orders)
        cases.append({"args": [orders], "expected": expected})
    return cases


def _build_total_bill(rng: random.Random, difficulty: str) -> dict[str, Any]:
    return {
        "template_id": "total-bill-v1",
        "bug_category": "wrong_aggregation",
        "function_name": "total_bill",
        "signature": "total_bill(orders) -> int",
        "description": (
            "`total_bill(orders)` must return the total cost in cents: the sum over every "
            "order of `qty * unit_price_cents`. Each order is a dict with exactly the keys "
            "`item`, `qty`, and `unit_price_cents`."
        ),
        "reference_source": _TOTAL_BILL_REFERENCE,
        "buggy_source": _TOTAL_BILL_BUGGY,
        "buggy_hard_source": _TOTAL_BILL_BUGGY_HARD,
        "public_test_source": _TOTAL_BILL_PUBLIC_TEST,
        "cases": _total_bill_cases(rng, difficulty),
    }


_TEMPLATE_BUILDERS = {
    "range-boundary-v1": _build_range,
    "eligibility-v1": _build_eligibility,
    "total-bill-v1": _build_total_bill,
}


def _build_task_md(template: dict[str, Any]) -> str:
    lines = [
        "# Python Repair Task",
        "",
        f"Repair the Python module `{MODULE_NAME}.py` in the isolated /workspace directory.",
        "",
        "The module must expose exactly one public function:",
        "",
        f"    def {template['signature']}",
        "",
        "Required behavior:",
        "",
        f"- {template['description']}",
        "- Do not rename the module, the file, or the public function, and do not change its signature.",
        "- Use only the Python standard library; no third-party packages.",
        "- Handle empty inputs and boundary values correctly.",
        "",
        "A public smoke test `test_public.py` is included. It only checks that the API is",
        "present and may pass even while the implementation is still wrong. Final verification",
        "runs additional hidden cases that the repaired module must satisfy.",
        "",
        "The task is complete only when every hidden case passes.",
        "",
    ]
    return "\n".join(lines)


def generate_python_repair_task(
    root: str | Path, seed: int, difficulty: str = "medium"
) -> TaskSpec:
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {sorted(DIFFICULTIES)}")
    root_path = _root(root)
    rng = random.Random(seed)
    builder = _TEMPLATE_BUILDERS[rng.choice(tuple(_TEMPLATE_BUILDERS))]
    template = builder(rng, difficulty)
    template_id = template["template_id"]
    task_id = f"python-repair-{template_id}-{difficulty}-{seed}"
    task_path = _task_dir(root_path, task_id)
    manifest_path = task_path / "task.json"
    if manifest_path.exists():
        return TaskSpec.from_dict(read_json(manifest_path))

    initial = task_path / "initial"
    private = task_path / "private"
    initial.mkdir(parents=True, exist_ok=False)
    private.mkdir(parents=True, exist_ok=False)

    function_name = template["function_name"]
    buggy_source = template["buggy_hard_source"] if difficulty == "hard" else template["buggy_source"]
    task_md = _build_task_md(template) + "\n"

    (initial / "TASK.md").write_text(task_md, encoding="utf-8")
    (initial / f"{MODULE_NAME}.py").write_text(buggy_source, encoding="utf-8")
    (initial / "test_public.py").write_text(template["public_test_source"], encoding="utf-8")

    (private / REFERENCE_FILE).write_text(template["reference_source"], encoding="utf-8")
    write_json(private / HIDDEN_CASES_FILE, template["cases"])
    (private / RUNNER_FILE).write_text(_RUNNER_SOURCE, encoding="utf-8")

    contract = (
        f"Repair {MODULE_NAME}.py so it exposes the single public function "
        f"{function_name} with its declared signature.",
        template["description"],
        "Handle empty inputs and boundary values correctly.",
        "Use only the Python standard library; do not rename the module, file, or function.",
        "All hidden verification cases must pass; the public smoke test alone is not sufficient.",
    )
    spec = TaskSpec(
        id=task_id,
        family="python_repair",
        template_id=template_id,
        generator_version=GENERATOR_VERSION,
        seed=seed,
        difficulty=difficulty,
        prompt=task_md.strip(),
        contract=contract,
        public_metadata={
            "module_name": MODULE_NAME,
            "function_name": function_name,
            "template_id": template_id,
            "bug_category": template["bug_category"],
            "hidden_case_count": len(template["cases"]),
            "files": ["TASK.md", f"{MODULE_NAME}.py", "test_public.py"],
            "private_files": [REFERENCE_FILE, HIDDEN_CASES_FILE, RUNNER_FILE],
        },
        workspace_ref=str(initial),
        verifier_ref=str(private),
    )
    write_json(manifest_path, spec.to_dict())
    return spec


# --------------------------------------------------------------------------
# Verification.
# --------------------------------------------------------------------------


def _cap_failures(failures: list[dict[str, Any]], limit: int = 3, entry_limit: int = 400) -> list[dict[str, Any]]:
    capped: list[dict[str, Any]] = []
    for failure in (failures or [])[:limit]:
        entry = dict(failure)
        for key in ("args", "expected", "actual", "error"):
            if key in entry:
                entry[key] = repr(entry[key])[:entry_limit]
        capped.append(entry)
    return capped


def _capped(value: Any, limit: int = _DETAIL_LIMIT) -> str | None:
    """Cap an arbitrary diagnostic value so a misbehaving module cannot flood
    or exfiltrate through the returned report."""
    if value is None:
        return None
    return str(value)[:limit]


def _result_from_report(report: Mapping[str, Any]) -> VerificationResult:
    """Map the private runner's JSON report to a VerificationResult.

    Kept as a pure function so the taxonomy mapping is unit-testable without
    a sandbox. All diagnostics are capped before being returned.
    """
    status = report.get("status")
    if status == "success":
        return VerificationResult(
            success=True,
            verifier_id=VERIFIER_ID,
            verifier_version=VERIFIER_VERSION,
            failure_code=None,
            diagnostics={
                "cases_run": report.get("cases_run"),
                "cases_passed": report.get("cases_passed"),
            },
        )
    if status == "syntax_error":
        return VerificationResult(
            success=False,
            verifier_id=VERIFIER_ID,
            verifier_version=VERIFIER_VERSION,
            failure_code="syntax_error",
            diagnostics={"detail": _capped(report.get("detail"))},
        )
    if status == "runtime_error":
        return VerificationResult(
            success=False,
            verifier_id=VERIFIER_ID,
            verifier_version=VERIFIER_VERSION,
            failure_code="runtime_error",
            diagnostics={"detail": _capped(report.get("detail"))},
        )
    if status == "test_failure":
        return VerificationResult(
            success=False,
            verifier_id=VERIFIER_ID,
            verifier_version=VERIFIER_VERSION,
            failure_code="test_failure",
            diagnostics={
                "cases_run": report.get("cases_run"),
                "cases_passed": report.get("cases_passed"),
                "failures": _cap_failures(report.get("failures") or []),
            },
        )
    return VerificationResult(
        success=False,
        verifier_id=VERIFIER_ID,
        verifier_version=VERIFIER_VERSION,
        failure_code="runtime_error",
        diagnostics={"error": f"unknown verifier status: {status!r}"},
    )


def _verifier_command(python: str, module_name: str, function_name: str) -> str:
    """Shell command run inside the sandbox; only in-sandbox paths are used so
    the submitted module never sees host paths in its argv."""
    return (
        f"{shlex.quote(python)} -I -S -B "
        f"{shlex.quote('/private/' + RUNNER_FILE)} "
        f"{shlex.quote('/workspace')} "
        f"{shlex.quote(module_name)} "
        f"{shlex.quote(function_name)} "
        f"{shlex.quote('/private/' + HIDDEN_CASES_FILE)} "
        f"{shlex.quote('/output/' + REPORT_FILE)}"
    )


def _sandbox_python() -> str | None:
    """A Python interpreter that runs inside the sandbox.

    Prefers a system interpreter under the read-only runtime mounts that
    matches the running interpreter's major.minor version (determinism); the
    running interpreter itself is the fallback when it already lives under a
    runtime mount. Never returns a host path the sandbox cannot reach.
    """
    target = f"{sys.version_info.major}.{sys.version_info.minor}"
    found = sandbox_python_interpreter(preferred_version=target)
    if found is not None:
        return found
    resolved = Path(sys.executable).resolve()
    if str(resolved).startswith(ISOLATED_RUNTIME_PATHS):
        return str(resolved)
    return None


def _write_verification(
    root_path: Path, attempt: Any, result: VerificationResult
) -> None:
    attempt_path = _attempt_dir(root_path, attempt.attempt_id)
    verification_path = attempt_path / "verification.json"
    write_json(verification_path, result.to_dict())
    updated = replace(
        attempt,
        status="verified",
        verification_ref=str(verification_path),
    )
    write_json(attempt_path / "attempt.json", updated.to_dict())


def verify_python_repair_attempt(
    root: str | Path,
    task_id: str,
    attempt_id: str,
    timeout_seconds: int = 15,
) -> VerificationResult:
    root_path = _root(root)
    spec = load_task(root_path, task_id)
    attempt = load_attempt(root_path, attempt_id)
    if attempt.task_id != spec.id:
        raise ValueError("attempt does not belong to task")

    module_name = str(spec.public_metadata["module_name"])
    function_name = str(spec.public_metadata["function_name"])
    private = Path(spec.verifier_ref)
    workspace = Path(attempt.workspace_ref)
    submitted = workspace / f"{module_name}.py"

    if not submitted.exists():
        result = VerificationResult(
            success=False,
            verifier_id=VERIFIER_ID,
            verifier_version=VERIFIER_VERSION,
            failure_code="missing_file",
            diagnostics={"required_module": f"{module_name}.py"},
        )
        _write_verification(root_path, attempt, result)
        return result

    timeout = max(1, int(timeout_seconds))

    # No host-execution fallback: without the sandbox backend we fail loudly.
    if not sandbox_available():
        result = VerificationResult(
            success=False,
            verifier_id=VERIFIER_ID,
            verifier_version=VERIFIER_VERSION,
            failure_code="sandbox_unavailable",
            diagnostics={
                "reason": (
                    "bwrap and a systemd user session are required for sandboxed "
                    "verification; submitted code is never executed on the host"
                ),
                "bwrap": bwrap_available(),
                "systemd_user": systemd_user_available(),
            },
        )
        _write_verification(root_path, attempt, result)
        return result

    python = _sandbox_python()
    if python is None:
        result = VerificationResult(
            success=False,
            verifier_id=VERIFIER_ID,
            verifier_version=VERIFIER_VERSION,
            failure_code="sandbox_unavailable",
            diagnostics={
                "reason": (
                    f"no Python {sys.version_info.major}.{sys.version_info.minor} "
                    "interpreter is reachable inside the sandbox"
                ),
            },
        )
        _write_verification(root_path, attempt, result)
        return result

    limits = SandboxLimits(max_timeout_seconds=min(600, max(30, timeout)))
    sandbox = BubblewrapSandbox(root_path, workspace, limits)
    with tempfile.TemporaryDirectory(prefix="pyreplab-python-repair-") as tmp:
        tmp_path = Path(tmp)
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        result_file = output_dir / REPORT_FILE
        command = _verifier_command(python, module_name, function_name)
        try:
            run = sandbox.execute_isolated(
                command,
                timeout,
                read_only=[
                    SandboxBind(workspace, "/workspace"),
                    SandboxBind(private, "/private"),
                ],
                writable=[SandboxBind(output_dir, "/output")],
                extra_env={"PYTHONDONTWRITEBYTECODE": "1"},
            )
        except SandboxUnavailableError as error:
            result = VerificationResult(
                success=False,
                verifier_id=VERIFIER_ID,
                verifier_version=VERIFIER_VERSION,
                failure_code="sandbox_unavailable",
                diagnostics={"error": _capped(str(error))},
            )
            _write_verification(root_path, attempt, result)
            return result

        if run.timed_out:
            result = VerificationResult(
                success=False,
                verifier_id=VERIFIER_ID,
                verifier_version=VERIFIER_VERSION,
                failure_code="timeout",
                diagnostics={
                    "timeout_seconds": timeout,
                    "stderr_tail": run.stderr[-_DIAGNOSTIC_TAIL:],
                },
            )
        elif run.exit_code != 0 or not result_file.exists():
            result = VerificationResult(
                success=False,
                verifier_id=VERIFIER_ID,
                verifier_version=VERIFIER_VERSION,
                failure_code="runtime_error",
                diagnostics={
                    "exit_code": run.exit_code,
                    "stdout_tail": run.stdout[-_DIAGNOSTIC_TAIL:],
                    "stderr_tail": run.stderr[-_DIAGNOSTIC_TAIL:],
                    "truncated": run.truncated,
                },
            )
        else:
            try:
                if result_file.stat().st_size > _REPORT_MAX_BYTES:
                    raise ValueError("verifier report exceeds size limit")
                report = read_json(result_file)
                result = _result_from_report(report)
            except Exception as error:
                result = VerificationResult(
                    success=False,
                    verifier_id=VERIFIER_ID,
                    verifier_version=VERIFIER_VERSION,
                    failure_code="runtime_error",
                    diagnostics={"error": f"verifier report could not be read: {error}"},
                )
        _write_verification(root_path, attempt, result)
        return result
