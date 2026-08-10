"""Resumable sequential batch runner for pilot evaluations.

``run_batch`` drives the existing orchestrator functions ``run_pair`` and
``run_single`` one job at a time, appending exactly one JSONL record per job
with timestamps, duration, job coordinates and either the measured ``result``
or a structured ``error``.

Execution is strictly sequential on purpose: the shared Gemma server is pinned
to ``parallelism=1`` (a single model slot), so concurrent pilot requests would
either be serialized server-side or rejected. Never run a batch concurrently
with any other Gemma workload. This module touches no cron schedule and never
unloads or switches the loaded Gemma model beyond what the orchestrator
already does.

Resume semantics
----------------
``run_batch(..., resume=True)`` parses the output JSONL and treats every
complete line whose record has ``status: "completed"`` as an already-finished
job. Those jobs are skipped; a verification failure is a completed, measured
outcome (``status: "completed"``, ``ok: false``) and is therefore never
re-run. Malformed or truncated lines are tolerated and simply ignored, so
their jobs are re-run. Records are appended atomically (write to a sibling
temp file, ``fsync``, then ``os.replace``), so a completed job is never left
partially recorded in the output file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .gym_registry import FAMILIES
from .orchestrator import (
    RemoteConfig,
    run_pair,
    run_registered_treatments,
    run_single,
    validate_remote_config,
)
from .treatments import TreatmentRegistry

#: Canonical fixture template names recognised by the unbrowser fixture family.
_FIXTURE_TEMPLATES: tuple[str, ...] = (
    "single_page_extraction",
    "table_filter_sort",
    "multi_page_navigation",
    "search_filter_controls",
    "form_entry_validation",
    "cross_page_comparison",
    "stateful_workflow",
    "distractor_recovery",
)

_DEFAULT_FIXTURE_TEMPLATE = "single_page_extraction"
_DEFAULT_CONFINE_UNBROWSER = True

_POLICIES: tuple[str, ...] = ("direct", "deliberate")
_DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard")

_DEFAULT_REMOTE_PROJECT = os.environ.get("PYREPLAB_REMOTE_PROJECT") or None
_DEFAULT_REMOTE_RUN_ROOT = os.environ.get("PYREPLAB_REMOTE_RUN_ROOT") or (
    f"{_DEFAULT_REMOTE_PROJECT}/.runs" if _DEFAULT_REMOTE_PROJECT else None
)

#: Mirrors ``orchestrator.build_parser`` defaults so ``run_batch`` works with a
#: minimal mapping and the CLI stays drop-in compatible with orchestrator flags.
_ORCHESTRATOR_DEFAULTS: dict[str, Any] = {
    "host": os.environ.get("PYREPLAB_HARNESS_HOST", "ubuntu-local"),
    "remote_project": _DEFAULT_REMOTE_PROJECT,
    "remote_run_root": _DEFAULT_REMOTE_RUN_ROOT,
    "remote_python": "python3",
    "pi": os.environ.get("PYREPLAB_PI", "pi"),
    "provider": os.environ.get("PYREPLAB_PI_PROVIDER", "ubuntu-gemma"),
    "model": os.environ.get("PYREPLAB_PI_MODEL", "gemma-4-26b-a4b"),
    "thinking": os.environ.get("PYREPLAB_PI_THINKING", "off"),
    "model_switch_extension": os.environ.get("PYREPLAB_MODEL_SWITCH_EXTENSION")
    or None,
    "policy": "direct",
    "policy_version": "1",
    "attempt_id": None,
    "fixture_template": _DEFAULT_FIXTURE_TEMPLATE,
    "confine_unbrowser": _DEFAULT_CONFINE_UNBROWSER,
}


@dataclass(frozen=True)
class BatchSpec:
    """Serializable description of the job matrix to expand and run.

    Jobs are expanded deterministically as (family, difficulty, seed), with
    ``family`` varying slowest. ``pair`` selects the two-policy runner by
    default; ``single_policy`` selects the single-policy runner.
    """

    families: tuple[str, ...]
    difficulties: tuple[str, ...]
    seeds: tuple[int, ...]
    pair: bool = True
    single_policy: str | None = None
    treatment_refs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "families": list(self.families),
            "difficulties": list(self.difficulties),
            "seeds": list(self.seeds),
            "pair": self.pair,
            "single_policy": self.single_policy,
            "treatment_refs": list(self.treatment_refs),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BatchSpec":
        return cls(
            families=tuple(value["families"]),
            difficulties=tuple(value["difficulties"]),
            seeds=tuple(int(seed) for seed in value["seeds"]),
            pair=bool(value.get("pair", True)),
            single_policy=value.get("single_policy"),
            treatment_refs=tuple(str(item) for item in value.get("treatment_refs", [])),
        )


@dataclass(frozen=True)
class JobSpec:
    """A single expanded job: one family/difficulty/seed in one runner mode."""

    family: str
    difficulty: str
    seed: int
    mode: str
    policy: str | None = None

    @property
    def key(self) -> str:
        if self.mode == "treatment_set":
            signature = hashlib.sha256(str(self.policy).encode("utf-8")).hexdigest()[:12]
            return (
                f"treatment-set/{signature}/{self.family}/{self.difficulty}/"
                f"seed={self.seed}"
            )
        if self.mode == "pair":
            return f"pair/{self.family}/{self.difficulty}/seed={self.seed}"
        return f"single/{self.policy}/{self.family}/{self.difficulty}/seed={self.seed}"


def expand_jobs(spec: BatchSpec) -> tuple[JobSpec, ...]:
    """Deterministically expand a spec into its job list.

    Iteration order is family-major, then difficulty, then seed, and is stable
    across calls, so two runs with the same spec execute the same sequence.
    """
    jobs: list[JobSpec] = []
    for family in spec.families:
        for difficulty in spec.difficulties:
            for seed in spec.seeds:
                if spec.treatment_refs:
                    jobs.append(
                        JobSpec(
                            family,
                            difficulty,
                            seed,
                            mode="treatment_set",
                            policy=",".join(spec.treatment_refs),
                        )
                    )
                elif spec.pair:
                    jobs.append(JobSpec(family, difficulty, seed, mode="pair"))
                else:
                    jobs.append(
                        JobSpec(
                            family,
                            difficulty,
                            seed,
                            mode="single",
                            policy=spec.single_policy,
                        )
                    )
    return tuple(jobs)


def _parse_csv_list(text: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in text.split(",") if item.strip())


def parse_families(text: str) -> tuple[str, ...]:
    families = _parse_csv_list(text)
    unknown = [family for family in families if family not in FAMILIES]
    if unknown:
        raise ValueError(
            f"unknown family: {unknown[0]!r}; expected one of: {', '.join(FAMILIES)}"
        )
    return families


def parse_difficulties(text: str) -> tuple[str, ...]:
    difficulties = _parse_csv_list(text)
    unknown = [difficulty for difficulty in difficulties if difficulty not in _DIFFICULTIES]
    if unknown:
        raise ValueError(
            f"unknown difficulty: {unknown[0]!r}; "
            f"expected one of: {', '.join(_DIFFICULTIES)}"
        )
    return difficulties


def parse_seeds(text: str) -> tuple[int, ...]:
    """Parse comma-separated seeds with ``start-end`` range support.

    ``"1-3"`` expands to ``(1, 2, 3)``; ``"2-4,7,3"`` expands to
    ``(2, 3, 4, 7)`` (first-occurrence order, duplicates removed).
    """
    seeds: list[int] = []
    seen: set[int] = set()
    for part in _parse_csv_list(text):
        if "-" in part:
            start_text, _, end_text = part.partition("-")
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"invalid seed range: {part!r}")
            values: Any = range(start, end + 1)
        else:
            values = (int(part),)
        for seed in values:
            if seed not in seen:
                seen.add(seed)
                seeds.append(seed)
    return tuple(seeds)


def validate_spec(spec: BatchSpec) -> list[str]:
    """Return a list of human-readable problems with ``spec`` (empty if valid)."""
    problems: list[str] = []
    if not spec.families:
        problems.append("families must not be empty")
    if not spec.difficulties:
        problems.append("difficulties must not be empty")
    if not spec.seeds:
        problems.append("seeds must not be empty")
    for family in spec.families:
        if family not in FAMILIES:
            problems.append(
                f"unknown family: {family!r}; expected one of: {', '.join(FAMILIES)}"
            )
    for difficulty in spec.difficulties:
        if difficulty not in _DIFFICULTIES:
            problems.append(
                f"unknown difficulty: {difficulty!r}; "
                f"expected one of: {', '.join(_DIFFICULTIES)}"
            )
    if not spec.treatment_refs and not spec.pair and spec.single_policy not in _POLICIES:
        problems.append(
            f"single mode requires single_policy to be one of: {', '.join(_POLICIES)}"
        )
    if spec.treatment_refs and spec.single_policy is not None:
        problems.append("treatment-set mode cannot be combined with single_policy")
    return problems


def default_preflight(
    spec: BatchSpec,
    orchestrator_args: Mapping[str, Any] | argparse.Namespace,
    output_path: str | Path,
) -> None:
    """Default preflight check: validate the spec and create the output directory.

    This is intentionally side-effect free with respect to the model: it never
    unloads or switches Gemma, and never touches cron schedules.
    """
    problems = validate_spec(spec)
    if problems:
        raise ValueError("invalid batch spec: " + "; ".join(problems))
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    # Resume files are treatment-specific.  Refuse to silently skip v1 jobs
    # when a caller points a v2 run at an older output (or vice versa).
    base_args = _as_dict(orchestrator_args)
    validate_remote_config(_build_config(_base_args(base_args)))
    expected_version = str(base_args.get("policy_version", "1"))
    expected_registry_hash: str | None = None
    if spec.treatment_refs:
        registry_path = base_args.get("treatment_registry")
        if not registry_path:
            raise ValueError("treatment-set batch requires treatment_registry")
        registry = TreatmentRegistry.load(registry_path)
        expected_registry_hash = registry.registry_hash
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if not isinstance(record, Mapping) or record.get("status") != "completed":
                continue
            if expected_registry_hash is not None:
                observed_registry = record.get("treatment_registry_hash")
                if str(observed_registry) != expected_registry_hash:
                    raise ValueError(
                        "output contains treatment registry hash "
                        f"{observed_registry!r}, expected {expected_registry_hash!r}; "
                        "use a separate --output file"
                    )
                continue
            observed = record.get("policy_version")
            if observed is None and expected_version == "1":
                continue  # Backward-compatible legacy v1 batch record.
            if str(observed) != expected_version:
                raise ValueError(
                    f"output contains policy version {observed!r}, expected "
                    f"{expected_version!r}; use a separate --output file"
                )


def _as_dict(value: Mapping[str, Any] | argparse.Namespace) -> dict[str, Any]:
    if isinstance(value, argparse.Namespace):
        return vars(value)
    return dict(value)


def _base_args(value: Mapping[str, Any] | argparse.Namespace) -> dict[str, Any]:
    merged = dict(_ORCHESTRATOR_DEFAULTS)
    merged.update(_as_dict(value))
    return merged


def _build_config(base: Mapping[str, Any]) -> RemoteConfig:
    return RemoteConfig(
        host=str(base.get("host") or ""),
        project=str(base.get("remote_project") or ""),
        run_root=str(base.get("remote_run_root") or ""),
        python=str(base.get("remote_python") or "python3"),
    )


def _invoke_runner(
    project_root: Path, base: Mapping[str, Any], job: JobSpec
) -> dict[str, Any]:
    """Build per-job orchestrator args and call the matching orchestrator runner.

    ``run_pair`` and ``run_single`` are resolved at call time so tests can
    mock them by patching this module's globals.
    """
    args = argparse.Namespace(**dict(base))
    args.family = job.family
    args.difficulty = job.difficulty
    args.seed = job.seed
    config = _build_config(base)
    if job.mode == "pair":
        args.pair = True
        return run_pair(project_root, config, args)
    if job.mode == "treatment_set":
        args.pair = False
        args.treatments = job.policy
        return run_registered_treatments(project_root, config, args)
    args.pair = False
    args.policy = job.policy
    return run_single(project_root, config, args)


def _verification_ok(result: Mapping[str, Any]) -> bool:
    """True when every attempt in a runner result passed verification."""
    verification = result.get("verification")
    if isinstance(verification, Mapping):
        return bool(verification.get("success"))
    attempts = result.get("attempts") or {}
    return bool(attempts) and all(
        bool(item.get("verification", {}).get("success"))
        for item in attempts.values()
    )


def _fsync_dir(directory: Path) -> None:
    try:
        descriptor = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _append_result(output_path: Path, record: Mapping[str, Any]) -> None:
    """Durably append one JSONL line for ``record``.

    The whole file is rewritten via a sibling temp file followed by
    ``os.replace``, so the output always contains only complete lines: a crash
    leaves either the old file or the new file, never a partially recorded
    completed job.
    """
    line = json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
    existing = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
    tmp = Path(f"{output_path}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(existing)
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, output_path)
        _fsync_dir(output_path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def _load_done_keys(output_path: Path) -> set[str]:
    """Return keys of jobs durably recorded with ``status: "completed"``.

    Malformed, truncated or non-object lines are tolerated and ignored, so a
    corrupt previous JSONL never aborts a resumed batch.
    """
    if not output_path.exists():
        return set()
    keys: set[str] = set()
    for line in output_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, Mapping):
            continue
        if record.get("status") == "completed" and isinstance(record.get("key"), str):
            keys.add(record["key"])
    return keys


@dataclass(frozen=True)
class BatchRunSummary:
    jobs_total: int
    completed: int
    error: int
    skipped: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "jobs_total": self.jobs_total,
            "completed": self.completed,
            "error": self.error,
            "skipped": self.skipped,
        }


def _run_job(
    project_root: Path, base: Mapping[str, Any], job: JobSpec
) -> dict[str, Any]:
    started = time.monotonic()
    record: dict[str, Any] = {
        "key": job.key,
        "family": job.family,
        "difficulty": job.difficulty,
        "seed": job.seed,
        "mode": job.mode,
        "policy": job.policy,
        "policy_version": str(base.get("policy_version", "1")),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    if job.mode == "treatment_set":
        record["treatment_refs"] = str(job.policy).split(",")
        record["treatment_registry_hash"] = base.get("treatment_registry_hash")
    if job.family == "unbrowser_fixture":
        record["fixture_template"] = str(base.get("fixture_template", _DEFAULT_FIXTURE_TEMPLATE))
        record["confine_unbrowser"] = bool(base.get("confine_unbrowser", _DEFAULT_CONFINE_UNBROWSER))
    try:
        result = _invoke_runner(project_root, base, job)
    except Exception as error:  # noqa: BLE001 - per-job failures are data.
        record["status"] = "error"
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        record["duration_seconds"] = round(time.monotonic() - started, 3)
        record["error"] = {"type": type(error).__name__, "message": str(error)}
    else:
        record["status"] = "completed"
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        record["duration_seconds"] = round(time.monotonic() - started, 3)
        record["ok"] = _verification_ok(result)
        record["result"] = result
    return record


def run_batch(
    spec: BatchSpec,
    orchestrator_args: Mapping[str, Any] | argparse.Namespace,
    output_path: str | Path,
    resume: bool = True,
    *,
    preflight: Callable[..., None] | None = default_preflight,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> BatchRunSummary:
    """Run every expanded job in ``spec`` strictly one at a time.

    Each job invokes the existing orchestrator runner (``run_pair`` or
    ``run_single``), then appends exactly one JSONL record atomically. Per-job
    exceptions are caught and recorded as structured errors; the batch
    continues. With ``resume=True``, jobs durably recorded as ``completed`` in
    ``output_path`` are skipped and never re-run.
    """
    output = Path(output_path).expanduser()
    base = _base_args(orchestrator_args)
    if spec.treatment_refs and not base.get("treatment_registry_hash"):
        registry_path = base.get("treatment_registry")
        if not registry_path:
            raise ValueError("treatment-set batch requires treatment_registry")
        base["treatment_registry_hash"] = TreatmentRegistry.load(
            registry_path
        ).registry_hash
    if preflight is not None:
        preflight(spec, base, output)
    project_root = Path(__file__).resolve().parents[2]
    done_keys = _load_done_keys(output) if resume else set()
    jobs = expand_jobs(spec)
    completed = error = skipped = 0
    for job in jobs:
        if job.key in done_keys:
            skipped += 1
            continue
        record = _run_job(project_root, base, job)
        _append_result(output, record)
        done_keys.add(job.key)
        if record["status"] == "completed":
            completed += 1
        else:
            error += 1
        if progress is not None:
            progress(record)
    return BatchRunSummary(
        jobs_total=len(jobs),
        completed=completed,
        error=error,
        skipped=skipped,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyreplab-harness-batch",
        description=(
            "Run a resumable batch of pilot evaluations, strictly one job at a "
            "time. Jobs are never run concurrently: the shared Gemma server is "
            "pinned to parallelism=1 (a single model slot), so concurrent "
            "requests would serialize or be rejected server-side. This command "
            "touches no cron schedule and never unloads or switches the loaded "
            "model beyond what the orchestrator already does."
        ),
    )
    parser.add_argument("--host", default=_ORCHESTRATOR_DEFAULTS["host"])
    parser.add_argument(
        "--remote-project", default=_ORCHESTRATOR_DEFAULTS["remote_project"]
    )
    parser.add_argument(
        "--remote-run-root", default=_ORCHESTRATOR_DEFAULTS["remote_run_root"]
    )
    parser.add_argument("--remote-python", default=_ORCHESTRATOR_DEFAULTS["remote_python"])
    parser.add_argument("--pi", default=_ORCHESTRATOR_DEFAULTS["pi"])
    parser.add_argument("--provider", default=_ORCHESTRATOR_DEFAULTS["provider"])
    parser.add_argument("--model", default=_ORCHESTRATOR_DEFAULTS["model"])
    parser.add_argument("--thinking", default=_ORCHESTRATOR_DEFAULTS["thinking"])
    parser.add_argument(
        "--model-switch-extension",
        default=_ORCHESTRATOR_DEFAULTS["model_switch_extension"],
    )
    parser.add_argument(
        "--policy-version",
        choices=("1", "2", "3", "4"),
        default=_ORCHESTRATOR_DEFAULTS["policy_version"],
        help="immutable Direct/Deliberate policy version; use a separate output per version",
    )
    parser.add_argument(
        "--families",
        required=True,
        help="comma-separated gym families, e.g. artifact,sqlite",
    )
    parser.add_argument(
        "--difficulties",
        default="medium",
        help="comma-separated difficulties: easy,medium,hard (default: medium)",
    )
    parser.add_argument(
        "--seeds",
        required=True,
        help="comma-separated seeds; ranges like 1-3 expand to 1,2,3",
    )
    parser.add_argument(
        "--single-policy",
        choices=_POLICIES,
        default=None,
        help="run a single policy (direct or deliberate) instead of the default pair",
    )
    parser.add_argument(
        "--treatment-registry",
        default=None,
        help="immutable registry for generalized treatment-set batches",
    )
    parser.add_argument(
        "--treatments",
        default=None,
        help="comma-separated registry treatment references; mutually exclusive with --single-policy",
    )
    parser.add_argument(
        "--fixture-template",
        choices=_FIXTURE_TEMPLATES,
        default=_DEFAULT_FIXTURE_TEMPLATE,
        help="fixture page template for the unbrowser_fixture family "
        "(default: single_page_extraction)",
    )
    parser.add_argument(
        "--confine-unbrowser",
        action="store_true",
        default=_DEFAULT_CONFINE_UNBROWSER,
        help="confine Unbrowser to only the fixture page URL (default: True)",
    )
    parser.add_argument(
        "--no-confine-unbrowser",
        action="store_false",
        dest="confine_unbrowser",
        help="disable Unbrowser confinement (allow any URL)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="path to the JSONL results file; one record is appended per job",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore previously recorded jobs in --output and rerun every job",
    )
    return parser


def _log_progress(record: Mapping[str, Any]) -> None:
    print(f"job {record['key']}: {record['status']}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        treatment_refs = _parse_csv_list(args.treatments or "")
        if treatment_refs and args.single_policy is not None:
            raise ValueError("--treatments cannot be combined with --single-policy")
        registry_hash = None
        if treatment_refs:
            if not args.treatment_registry:
                raise ValueError("--treatments requires --treatment-registry")
            registry_hash = TreatmentRegistry.load(args.treatment_registry).registry_hash
        spec = BatchSpec(
            families=parse_families(args.families),
            difficulties=parse_difficulties(args.difficulties),
            seeds=parse_seeds(args.seeds),
            pair=args.single_policy is None,
            single_policy=args.single_policy,
            treatment_refs=treatment_refs,
        )
        orchestrator_args = {
            "host": args.host,
            "remote_project": args.remote_project,
            "remote_run_root": args.remote_run_root,
            "remote_python": args.remote_python,
            "pi": args.pi,
            "provider": args.provider,
            "model": args.model,
            "thinking": args.thinking,
            "model_switch_extension": args.model_switch_extension,
            "policy_version": args.policy_version,
            "treatment_registry": args.treatment_registry,
            "treatments": args.treatments,
            "treatment_registry_hash": registry_hash,
            "fixture_template": args.fixture_template,
            "confine_unbrowser": args.confine_unbrowser,
        }
        summary = run_batch(
            spec,
            orchestrator_args,
            args.output,
            resume=not args.no_resume,
            progress=_log_progress,
        )
    except ValueError as error:
        print(f"batch error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary.to_dict(), sort_keys=True, indent=2))
    # Verification failures are completed, measured outcomes and do not affect
    # the exit code; only infrastructure-level job errors do.
    return 1 if summary.error else 0


__all__ = [
    "BatchRunSummary",
    "BatchSpec",
    "JobSpec",
    "build_parser",
    "default_preflight",
    "expand_jobs",
    "main",
    "parse_difficulties",
    "parse_families",
    "parse_seeds",
    "run_batch",
    "validate_spec",
    "_FIXTURE_TEMPLATES",
    "_DEFAULT_FIXTURE_TEMPLATE",
    "_DEFAULT_CONFINE_UNBROWSER",
]


if __name__ == "__main__":
    raise SystemExit(main())
