"""Public registry and dispatch for the verifiable gym families.

Canonical family names are ``artifact``, ``sqlite``, ``shell`` and
``python_repair``, plus the narrow live ``unbrowser`` smoke family and the
interactive ``unbrowser_interactive`` plumbing spike. Generation and
verification delegate to the per-family modules; attempts are prepared with
the generic ``artifact_gym.prepare_attempt`` helper that every family already
shares.

Unknown family names fail with a ``ValueError`` that lists the valid names.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .artifact_gym import (
    generate_artifact_task,
    prepare_attempt as _prepare_attempt_impl,
    verify_artifact_attempt,
)
from .contracts import AttemptRecord, TaskSpec, VerificationResult
from .python_repair_gym import generate_python_repair_task, verify_python_repair_attempt
from .shell_gym import generate_shell_task, verify_shell_attempt
from .sqlite_gym import generate_sqlite_task, verify_sqlite_attempt
from .unbrowser_gym import generate_unbrowser_task, verify_unbrowser_attempt
from .unbrowser_interactive_gym import (
    generate_unbrowser_interactive_task,
    verify_unbrowser_interactive_attempt,
)

#: Canonical family names, in a stable display order.
FAMILIES: tuple[str, ...] = (
    "artifact",
    "sqlite",
    "shell",
    "python_repair",
    "unbrowser",
    "unbrowser_interactive",
)

_REGISTRY: dict[str, dict[str, Callable[..., Any]]] = {
    "artifact": {
        "generate": generate_artifact_task,
        "verify": verify_artifact_attempt,
    },
    "sqlite": {
        "generate": generate_sqlite_task,
        "verify": verify_sqlite_attempt,
    },
    "shell": {
        "generate": generate_shell_task,
        "verify": verify_shell_attempt,
    },
    "python_repair": {
        "generate": generate_python_repair_task,
        "verify": verify_python_repair_attempt,
    },
    "unbrowser": {
        "generate": generate_unbrowser_task,
        "verify": verify_unbrowser_attempt,
    },
    "unbrowser_interactive": {
        "generate": generate_unbrowser_interactive_task,
        "verify": verify_unbrowser_interactive_attempt,
    },
}


def _lookup(family: str) -> dict[str, Callable[..., Any]]:
    try:
        return _REGISTRY[family]
    except KeyError as error:
        valid = ", ".join(FAMILIES)
        raise ValueError(f"unknown family: {family!r}; expected one of: {valid}") from error


def generate_task(
    family: str, root: str | Path, seed: int, difficulty: str = "medium"
) -> TaskSpec:
    """Generate a deterministic task of ``family`` under ``root``."""
    return _lookup(family)["generate"](root, seed, difficulty)


def verify_attempt(
    family: str, root: str | Path, task_id: str, attempt_id: str
) -> VerificationResult:
    """Verify ``attempt_id`` against ``task_id`` with the family verifier."""
    return _lookup(family)["verify"](root, task_id, attempt_id)


def prepare_attempt(
    root: str | Path,
    task_id: str,
    attempt_id: str,
    policy_id: str,
    policy_version: str = "1",
) -> AttemptRecord:
    """Generic attempt preparation shared by every family."""
    return _prepare_attempt_impl(root, task_id, attempt_id, policy_id, policy_version)


__all__ = [
    "FAMILIES",
    "generate_task",
    "prepare_attempt",
    "verify_attempt",
]
