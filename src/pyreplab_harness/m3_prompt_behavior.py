"""Standalone, treatment-blind behavior classification for M3 attempts.

This module is deliberately isolated from the rest of the harness. It does not
import treatment, verifier, dataset, or controller code. Two roles live here:

1. A **producer/adapter** (``build_restricted_evidence``) that converts the REAL
   normalized ``trajectory.tool_trace`` details-nested shape plus raw Pi JSONL
   events into a strict, flattened, bounded *restricted-evidence* schema. It
   inspects raw ``toolCall`` arguments only transiently to compute canonical
   request hashes (joining on ``toolCall.id``/``tool_call_id``, which is dropped
   from the emitted evidence) and never returns or persists raw arguments,
   free-form error or detail text, keys, treatments, outcomes, templates, or
   verifier fields. The producer raises :class:`RestrictedEvidenceError` on
   malformed input (malformed raw JSON, duplicate/conflicting toolCall IDs,
   malformed trajectory entries, or missing/invalid ``tool_name``) so callers can
   never obtain apparently-valid evidence from malformed input.

2. **Classifiers** that consume only restricted evidence (and an optional
   controller result-write receipt) and never accept treatment identity,
   verifier outcome, ``verified_success``, a private oracle/answer/key, or the
   task template.

Guarantees
----------

* Strict, versioned, deterministic decision tables (no model, no randomness).
* Fail-closed: malformed, ambiguous, contradictory, or privacy-violating
  evidence classifies as ``unknown`` (classifiers never raise for bad evidence);
  the producer raises a documented :class:`RestrictedEvidenceError` for bad
  *source* input.
* Treatment-blind API: public signatures accept only ``evidence``/``trajectory``
  plus optional ``raw_events``/``result_write_receipt``.
* Privacy: raw request arguments are never accepted by a classifier (only their
  canonical SHA-256 via ``request_args_hash``); the recursive ``privacy_scan``
  rejects forbidden field names without false positives on benign source keys
  and without duplicate violations.
* Self-hashed receipts: ``analyze_attempt`` returns a canonical, self-hashed
  behavior receipt containing schema/classifier source identity, bounded
  labels/counters/fingerprints only — never raw text, args, keys, outcomes,
  treatments, templates, or the result-write receipt's content/hash.
* ITT neutrality: recovery/opportunity diagnostics are purely descriptive and
  never alter inclusion (the receipt carries ``itt_inclusion: "unconditional"``).

Restricted evidence schema
--------------------------

``pyreplab-behavior-restricted-evidence-v1``::

    {
      "schema_version": "pyreplab-behavior-restricted-evidence-v1",
      "provider_turn_count": int | null,
      "tool_trace": [
        {
          "tool_name": str,                 # exactly "bash" or "unbrowser"
          "is_error": bool,                 # raw Pi isError (bounded)
          "budget_rejected": bool,
          "operation_aborted": bool,
          "pre_execution_rejected": bool,
          "result_submission": bool,
          "infrastructure_error": bool,
          "error_class": str | null,        # bounded enum, see ERROR_CLASSES
          "status": int | null,             # bash exit code / browser HTTP status
          "request_args_hash": str | null,  # sha256(canonical_json(args))
        }, ...
      ]
    }

``tool_name`` is validated against the exact pilot vocabulary ``{bash,
unbrowser}``. ``tool_call_id`` is used only transiently for the raw-event hash
join and is never emitted. No arbitrary ``details`` or raw ``error`` strings are
accepted; the producer emits bounded error classes only. Status semantics are
tool-aware: bash exit code 0 succeeds, browser HTTP status 200 succeeds;
otherwise classification relies on ``is_error`` and the bounded ``error_class``.

Error-class derivation (producer)
---------------------------------

For each tool call the producer derives a bounded ``error_class`` with this
priority: a pre-execution-rejection flag yields ``pre_execution_rejection``; a
budget-rejection flag or budget text marker yields ``budget_limit``; an explicit
infrastructure flag or infra text marker yields ``infrastructure``; ``is_error``
yields ``tool_error``; a tool-aware *successful* status dominates any remaining
benign free-form warning/error text (so it yields ``None``); a non-success
status yields ``tool_error``; finally a non-empty error text yields
``tool_error``. Bash reads ``exit_code`` first and falls back to integer
``status``.

Completion labels
-----------------

``submitted_before_budget_block``   exactly one valid-shape write, no prior
                                    budget block, no later tool attempt, no
                                    prior eligible tool error.
``submitted_after_prior_error``     exactly one valid write with a prior budget
                                    block and/or a prior eligible ordinary tool
                                    error, and no later tool attempt.
``no_submission``                   no valid result write in the trace.
``multiple_submissions``            more than one valid result write.
``post_submission_tool_activity``   a valid write followed by a later tool
                                    attempt (including budget-blocked ones).
``unknown``                         malformed/ambiguous/privacy-violating input.

The separate ``intended_behavior`` flag is true when there is exactly one
valid-shape write, no prior budget block, and no later tool attempt — including
a successful submission after an ordinary recoverable error (which still gets
the ``submitted_after_prior_error`` label). A submission after a budget block is
not intended behavior.

Recovery labels
---------------

Opportunities are built only from **eligible ordinary tool errors/rejections**:
a tool call whose bounded ``error_class``/tool-aware ``status``/``is_error``
marks an executed failure, excluding budget-limit errors, infrastructure
failures, and never-executed pre-execution rejections. Retries are compared by
canonical fingerprint ``(tool_name, args_sha256)`` and raw arguments are never
emitted. When request hashes needed for retry comparison are unavailable the
classifier fails closed to ``unknown``.

``no_opportunity``                     no eligible ordinary tool error.
``corrected_once_success``             one changed retry that succeeded.
``corrected_once_failed_then_stopped`` one changed retry that failed and then
                                       the agent made no further tool call.
``unchanged_repeat``                   a single retry with the identical
                                       canonical request fingerprint.
``retry_loop``                         two or more same-tool retries.
``abandoned_after_error``              no tool call at all after the first
                                       eligible error.
``unknown``                            ambiguous (e.g., retries without
                                       request hashes, retried then moved on).

Result-write receipt (ephemeral)
--------------------------------

``classify_completion`` and ``analyze_attempt`` validate the optional
controller-produced result-write receipt **in memory only**. Its
``content_sha256`` and any hash of the receipt itself are never copied into the
behavior receipt. The receipt must not be persisted in safe ledgers or exported
datasets: the V3 fixture ``verification_key`` is a short, low-entropy nonce, so
even a content digest could enable offline nonce recovery.

R2 integration API
------------------

A production caller (the execution module) should::

    from pyreplab_harness.m3_prompt_behavior import (
        build_restricted_evidence, analyze_attempt, RestrictedEvidenceError,
    )

    try:
        evidence = build_restricted_evidence(trajectory, raw_events)
    except RestrictedEvidenceError:
        # source input was malformed; no apparently-valid evidence is produced
        ...
    receipt = analyze_attempt(evidence, result_write_receipt)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

RESTRICTED_EVIDENCE_SCHEMA_VERSION = "pyreplab-behavior-restricted-evidence-v1"
RESULT_WRITE_RECEIPT_SCHEMA_VERSION = "pyreplab-result-write-receipt-v1"
BEHAVIOR_RECEIPT_SCHEMA_VERSION = "pyreplab-behavior-receipt-v1"
CLASSIFIER_SOURCE = "pyreplab_harness.m3_prompt_behavior"

RESULT_JSON_PATH = "/workspace/result.json"
# The result-write receipt is scoped to the M3 prompt-only (E/C/R) pilot over the
# V3 outcome-only fixture generator. Only that pilot may produce a receipt with
# this scope.
RESULT_WRITE_PILOT_SCOPE = "m3-prompt-only-pilot"

# Exact pilot tool vocabulary.
PILOT_TOOLS = frozenset({"bash", "unbrowser"})
# Tools whose integer ``status`` is an HTTP status (200 = success). Every other
# tool's ``status`` is a process exit code (0 = success).
_BROWSER_TOOLS = frozenset({"unbrowser"})

# Bounded error classes. ``None`` means no error.
ERROR_CLASSES = (
    "tool_error",
    "budget_limit",
    "infrastructure",
    "pre_execution_rejection",
)

COMPLETION_LABELS = (
    "submitted_before_budget_block",
    "submitted_after_prior_error",
    "no_submission",
    "multiple_submissions",
    "post_submission_tool_activity",
    "unknown",
)

RECOVERY_LABELS = (
    "no_opportunity",
    "corrected_once_success",
    "corrected_once_failed_then_stopped",
    "unchanged_repeat",
    "retry_loop",
    "abandoned_after_error",
    "unknown",
)

# Error text markers the producer inspects transiently (never persisted) to
# classify budget-limit and infrastructure failures into bounded error classes.
_BUDGET_ERROR_MARKERS = (
    "tool call limit",
    "unbrowser call limit",
    "tool_limit",
    "shared_tool_limit",
)
_INFRA_ERROR_MARKERS = (
    "connection",
    "network error",
    "timed out",
    "timeout",
    "protocol",
    "spawn",
    "socket",
    "unreachable",
    "crash",
    "disabled",
    "not enabled",
)

_EVIDENCE_TOP_LEVEL_KEYS = frozenset(
    {"schema_version", "provider_turn_count", "tool_trace"}
)
_EVIDENCE_ENTRY_KEYS = frozenset(
    {
        "tool_name",
        "is_error",
        "budget_rejected",
        "operation_aborted",
        "pre_execution_rejected",
        "result_submission",
        "infrastructure_error",
        "error_class",
        "status",
        "request_args_hash",
    }
)
_BOOL_ENTRY_FLAGS = (
    "is_error",
    "budget_rejected",
    "operation_aborted",
    "pre_execution_rejected",
    "result_submission",
    "infrastructure_error",
)
_RESULT_WRITE_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "pilot_scope",
        "path",
        "operation",
        "content_sha256",
        "shape",
        "verification_key",
    }
)
_BEHAVIOR_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "classifier_source",
        "classifier_source_sha256",
        "itt_inclusion",
        "provider_turn_count",
        "completion",
        "recovery",
        "receipt_hash",
    }
)


class RestrictedEvidenceError(ValueError):
    """Raised by the producer when source input cannot build valid evidence.

    The producer never silently skips or coerces malformed input, so callers
    cannot obtain apparently-valid restricted evidence from malformed raw JSON,
    duplicate/conflicting toolCall IDs, malformed trajectory entries, or a
    missing/invalid ``tool_name``.
    """


# ---------------------------------------------------------------------------
# Privacy validator
# ---------------------------------------------------------------------------

# Exact key names that must never appear in classifier input (except the
# special-cased ``verification_key`` descriptor inside result-write receipts).
_PRIVACY_EXACT_FORBIDDEN_KEYS = frozenset(
    {
        "success",
        "verification",
        "verification_key",  # special-cased: only the bounded descriptor shape
        "verifier",
        "verifier_id",
        "verifier_version",
        "verified_success",
        "failure_code",
        "oracle",
        "oracle_snapshot",
        "nonce",
        "answer",
        "final_answer",
        "expected_key",
        "api_key",
        "secret",
        "token",
        "password",
        "credential",
        "private",
        "task_id",
        "template_id",
        "template",
        "task_template",
        "task_prompt",
        "policy_id",
        "policy_version",
        "treatment_id",
        "treatment_version",
        "bundle_id",
        "bundle_hash",
        "registry_hash",
        "prompt",
        "system_prompt",
        "request_args",  # raw request arguments must never reach a classifier
    }
)

# Narrow substring patterns checked against lowercased key names. They are
# chosen to be distinctive so they never false-positive on benign source keys
# (e.g. ``prompt_tokens``) and never collide with the module's own restricted
# vocabulary (``request_args_hash``, ``later_success``, ``result_write_count``).
_PRIVACY_SUBSTRING_PATTERNS = (
    "verification_key",
    "oracle",
    "nonce",
    "verifier",
    "verified_success",
    "treatment",
    "policy",
    "bundle",
    "template",
    "system_prompt",
    "task_id",
    "failure_code",
)


def _is_verification_key_descriptor(value: Any) -> bool:
    """Return whether ``value`` is the bounded receipt descriptor for
    ``verification_key`` (presence + type only, never the key value)."""
    return (
        isinstance(value, Mapping)
        and set(value) == {"present", "type"}
        and value.get("present") is True
        and value.get("type") == "string"
    )


def privacy_scan(value: Any, prefix: str = "<root>") -> list[str]:
    """Recursively report forbidden fields that leak treatment identity,
    verifier outcomes, private oracles/answers/keys, task templates, or raw
    request arguments.

    The only permitted representation of ``verification_key`` is the bounded
    ``{"present": true, "type": "string"}`` descriptor used inside a valid
    result-write receipt. Any other occurrence (in particular a literal key
    value) is reported.

    Exact key matches short-circuit substring matching so a single key never
    yields duplicate violations. Substring patterns are deliberately narrow to
    avoid false positives on benign source keys (e.g. ``prompt_tokens``).

    Returns a list of violation descriptions; an empty list means clean.
    """
    violations: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            label = f"{prefix}.{key}"
            if isinstance(key, str):
                lowered = key.casefold()
                if lowered in _PRIVACY_EXACT_FORBIDDEN_KEYS:
                    if lowered == "verification_key":
                        if not _is_verification_key_descriptor(item):
                            violations.append(
                                f"forbidden field {key!r} at {label}: "
                                "verification_key must not carry a key value"
                            )
                    else:
                        violations.append(f"forbidden field {key!r} at {label}")
                else:
                    for pattern in _PRIVACY_SUBSTRING_PATTERNS:
                        if pattern in lowered:
                            violations.append(
                                f"forbidden field pattern {pattern!r} "
                                f"in {key!r} at {label}"
                            )
                            break
            violations.extend(privacy_scan(item, label))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            violations.extend(privacy_scan(item, f"{prefix}[{index}]"))
    return violations


# ---------------------------------------------------------------------------
# Canonical hashing
# ---------------------------------------------------------------------------


def _canonical_sha256(value: Any) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def canonical_args_hash(args: Mapping[str, Any]) -> str:
    """Return the canonical SHA-256 of a request-arguments object.

    The producer hashes raw request arguments with this function and stores only
    the digest in restricted evidence; the classifier itself never accepts or
    emits raw arguments.
    """
    if not isinstance(args, Mapping):
        raise ValueError("request args must be a JSON object")
    return _canonical_sha256(dict(args))


def _canonical_args_hash_any(args: Any) -> str | None:
    """Tolerant request-argument hashing used only by the producer.

    Accepts a mapping or a JSON string (the two shapes Pi emits for
    ``toolCall.arguments``). Returns ``None`` when no arguments are present.
    """
    if args is None:
        return None
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, Mapping):
            args = parsed
        elif parsed is not None:
            args = parsed
        else:
            args = args
    if isinstance(args, Mapping):
        return _canonical_sha256(dict(args))
    return _canonical_sha256(args)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_sha256_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def module_source_sha256() -> str:
    """Return the SHA-256 of this module's source file (classifier identity)."""
    path = Path(__file__).resolve()
    if not path.is_file() and path.suffix == ".pyc":
        path = path.with_suffix(".py")
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Producer / adapter: real nested trajectory + raw JSONL -> restricted evidence
# ---------------------------------------------------------------------------


def _parse_raw_events(raw_events: Any) -> Iterable[Any]:
    """Yield raw Pi event objects, failing closed on malformed input.

    Raises :class:`RestrictedEvidenceError` on malformed JSON lines and on
    non-object event values; the producer must never silently skip source input.
    """
    if isinstance(raw_events, str):
        for line_number, line in enumerate(raw_events.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise RestrictedEvidenceError(
                    f"malformed raw JSON on line {line_number}: {error}"
                ) from error
            if not isinstance(event, dict):
                raise RestrictedEvidenceError(
                    f"raw event on line {line_number} is not an object"
                )
            yield event
        return
    if isinstance(raw_events, Mapping):
        yield raw_events
        return
    for event in raw_events:
        if not isinstance(event, dict):
            raise RestrictedEvidenceError("raw event is not an object")
        yield event


def extract_request_args_hashes(raw_events: Any) -> dict[str, str]:
    """Map toolCall IDs to canonical request-argument hashes from raw events.

    Scans ``message_end`` assistant content for ``toolCall``/``tool_use``/
    ``tool-call`` items and hashes their ``arguments`` transiently. Returns only
    ``{tool_call_id: sha256}`` — no raw arguments, text, keys, treatments,
    outcomes, templates, or verifier fields.

    Raises :class:`RestrictedEvidenceError` on malformed raw JSON and on
    duplicate/conflicting toolCall IDs.
    """
    hashes: dict[str, str] = {}
    seen: dict[str, str | None] = {}
    for event in _parse_raw_events(raw_events):
        if event.get("type") != "message_end":
            continue
        message = event.get("message")
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, Mapping):
                continue
            if item.get("type") not in {"toolCall", "tool_use", "tool-call"}:
                continue
            tool_call_id = item.get("id")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                continue
            digest = _canonical_args_hash_any(item.get("arguments"))
            if tool_call_id in seen:
                if seen[tool_call_id] != digest:
                    raise RestrictedEvidenceError(
                        f"conflicting toolCall id {tool_call_id!r}"
                    )
                raise RestrictedEvidenceError(
                    f"duplicate toolCall id {tool_call_id!r}"
                )
            seen[tool_call_id] = digest
            if digest is not None:
                hashes[tool_call_id] = digest
    return hashes


def _status_value_is_success(tool_name: str, status: Any) -> bool | None:
    """Return tool-aware success for a bounded status value.

    ``unbrowser`` succeeds on HTTP 200; other tools succeed on exit code 0.
    Returns ``None`` when the status is not a usable integer.
    """
    if _is_int(status):
        return status == 200 if tool_name in _BROWSER_TOOLS else status == 0
    return None


def _compute_error_class(
    *,
    tool_name: str,
    status: int | None,
    is_error: bool,
    budget_rejected: bool,
    operation_aborted: bool,
    pre_execution_rejected: bool,
    infrastructure_error: bool,
    error_text: str,
) -> str | None:
    """Derive a bounded error class from flags + tool-aware status + text.

    Priority: pre-execution rejection flag; budget flag or text marker;
    explicit infrastructure flag or text marker; ``is_error``; a successful
    tool-aware status dominates remaining benign free-form text; non-success
    status -> ``tool_error``; non-empty text -> ``tool_error``.
    """
    if pre_execution_rejected:
        return "pre_execution_rejection"
    lowered = error_text.casefold()
    if budget_rejected or operation_aborted:
        return "budget_limit"
    if any(marker in lowered for marker in _BUDGET_ERROR_MARKERS):
        return "budget_limit"
    if infrastructure_error:
        return "infrastructure"
    if any(marker in lowered for marker in _INFRA_ERROR_MARKERS):
        return "infrastructure"
    if is_error:
        return "tool_error"
    success = _status_value_is_success(tool_name, status)
    if success is True:
        return None
    if success is False:
        return "tool_error"
    if error_text:
        return "tool_error"
    return None


def _flatten_details(
    tool_name: str, details: Any
) -> tuple[int | None, bool, bool, str]:
    """Flatten raw details into bounded (status, result_submission,
    infrastructure_error, error_text). No free-form strings are emitted (the
    error_text is returned transiently for classification only)."""
    if not isinstance(details, Mapping):
        return (None, False, False, "")
    status: int | None = None
    result_submission = False
    if tool_name in _BROWSER_TOOLS:
        http_status = details.get("status")
        if _is_int(http_status):
            status = http_status
    else:
        # bash / non-browser: read exit_code first, fall back to integer status.
        code = details.get("exit_code")
        if not _is_int(code):
            code = details.get("status")
        if _is_int(code):
            status = code
        result_submission = details.get("result_submission") is True
    infrastructure_error = details.get("infrastructure_error") is True
    error_text = details.get("error")
    error_text = error_text if isinstance(error_text, str) else ""
    return (status, result_submission, infrastructure_error, error_text)


def build_restricted_evidence(
    trajectory: Mapping[str, Any],
    raw_events: Any = None,
) -> dict[str, Any]:
    """Convert a normalized trajectory (details-nested ``tool_trace``) plus raw
    Pi JSONL events into strict restricted evidence.

    Treatment-blind: accepts only a trajectory and optional raw events; no
    treatment, outcome, template, verifier, oracle, answer, or key inputs.
    Raw ``toolCall`` arguments and detail/error strings are inspected only
    transiently (to compute canonical request hashes and bounded error classes)
    and are never returned or persisted. ``tool_call_id`` is used only for the
    raw-event hash join and is not emitted.

    Raises :class:`RestrictedEvidenceError` on malformed raw JSON,
    duplicate/conflicting toolCall IDs, malformed trajectory entries, or a
    missing/out-of-vocabulary ``tool_name``.
    """
    if not isinstance(trajectory, Mapping):
        raise RestrictedEvidenceError("trajectory must be an object")

    args_by_id: dict[str, str] = {}
    if raw_events is not None:
        args_by_id = extract_request_args_hashes(raw_events)

    provider_turn_count = trajectory.get("provider_turn_count")
    if provider_turn_count is not None and not _is_int(provider_turn_count):
        raise RestrictedEvidenceError(
            "trajectory.provider_turn_count must be an integer or null"
        )
    if _is_int(provider_turn_count) and provider_turn_count < 0:
        raise RestrictedEvidenceError(
            "trajectory.provider_turn_count must be non-negative"
        )

    raw_trace = trajectory.get("tool_trace")
    if raw_trace is None:
        raw_trace = []
    if not isinstance(raw_trace, list):
        raise RestrictedEvidenceError("trajectory.tool_trace must be a list")

    entries: list[dict[str, Any]] = []
    for entry in raw_trace:
        if not isinstance(entry, Mapping):
            raise RestrictedEvidenceError(
                "trajectory.tool_trace entries must be objects"
            )
        tool_name = entry.get("tool_name")
        if not isinstance(tool_name, str) or not tool_name:
            raise RestrictedEvidenceError(
                "trajectory.tool_trace entry is missing a tool_name"
            )
        if tool_name not in PILOT_TOOLS:
            raise RestrictedEvidenceError(
                f"unsupported tool_name {tool_name!r}; expected one of "
                f"{sorted(PILOT_TOOLS)!r}"
            )
        tool_call_id = entry.get("tool_call_id")
        if tool_call_id is not None and not isinstance(tool_call_id, str):
            raise RestrictedEvidenceError(
                "trajectory.tool_trace tool_call_id must be a string"
            )
        status, result_submission, infrastructure_error, error_text = (
            _flatten_details(tool_name, entry.get("details"))
        )
        request_args_hash = args_by_id.get(tool_call_id) if tool_call_id else None
        if request_args_hash is not None and not _is_sha256_hex(request_args_hash):
            request_args_hash = None
        error_class = _compute_error_class(
            tool_name=tool_name,
            status=status,
            is_error=entry.get("is_error") is True,
            budget_rejected=entry.get("budget_rejected") is True,
            operation_aborted=entry.get("operation_aborted") is True,
            pre_execution_rejected=entry.get("pre_execution_rejected") is True,
            infrastructure_error=infrastructure_error,
            error_text=error_text,
        )
        entries.append(
            {
                "tool_name": tool_name,
                "is_error": entry.get("is_error") is True,
                "budget_rejected": entry.get("budget_rejected") is True,
                "operation_aborted": entry.get("operation_aborted") is True,
                "pre_execution_rejected": entry.get("pre_execution_rejected")
                is True,
                "result_submission": result_submission,
                "infrastructure_error": infrastructure_error,
                "error_class": error_class,
                "status": status,
                "request_args_hash": request_args_hash,
            }
        )

    return {
        "schema_version": RESTRICTED_EVIDENCE_SCHEMA_VERSION,
        "provider_turn_count": provider_turn_count,
        "tool_trace": entries,
    }


# ---------------------------------------------------------------------------
# Evidence and receipt validation (fail-closed inputs)
# ---------------------------------------------------------------------------


def validate_evidence(evidence: Any) -> list[str]:
    """Return structural violations for restricted attempt evidence.

    An empty list means the evidence matches the restricted schema. Evidence
    violating the schema (including any arbitrary ``details`` or raw ``error``
    strings, ``tool_call_id``, or an out-of-vocabulary ``tool_name``) must be
    classified as ``unknown`` by the classifiers.
    """
    violations: list[str] = []
    if not isinstance(evidence, Mapping):
        return ["evidence must be a JSON object"]
    for key in evidence:
        if key not in _EVIDENCE_TOP_LEVEL_KEYS:
            violations.append(f"evidence: unknown top-level key {key!r}")
    if evidence.get("schema_version") != RESTRICTED_EVIDENCE_SCHEMA_VERSION:
        violations.append(
            f"evidence: unsupported schema_version {evidence.get('schema_version')!r}"
        )
    provider_turn_count = evidence.get("provider_turn_count")
    if provider_turn_count is not None and not _is_int(provider_turn_count):
        violations.append(
            "evidence: provider_turn_count must be an integer or null"
        )
    elif _is_int(provider_turn_count) and provider_turn_count < 0:
        violations.append("evidence: provider_turn_count must be non-negative")
    trace = evidence.get("tool_trace")
    if trace is None:
        return violations + ["evidence: missing tool_trace"]
    if not isinstance(trace, list):
        return violations + ["evidence: tool_trace must be a list"]
    for index, entry in enumerate(trace):
        if not isinstance(entry, Mapping):
            violations.append(f"evidence: tool_trace[{index}] must be an object")
            continue
        for key in entry:
            if key not in _EVIDENCE_ENTRY_KEYS:
                violations.append(f"evidence: tool_trace[{index}] unknown key {key!r}")
        tool_name = entry.get("tool_name")
        if tool_name not in PILOT_TOOLS:
            violations.append(
                f"evidence: tool_trace[{index}] tool_name must be one of "
                f"{sorted(PILOT_TOOLS)!r}"
            )
        for flag in _BOOL_ENTRY_FLAGS:
            value = entry.get(flag, False)
            if not isinstance(value, bool):
                violations.append(
                    f"evidence: tool_trace[{index}] {flag} must be a boolean"
                )
        status = entry.get("status")
        if status is not None and not _is_int(status):
            violations.append(
                f"evidence: tool_trace[{index}] status must be an integer or null"
            )
        error_class = entry.get("error_class")
        if error_class is not None and error_class not in ERROR_CLASSES:
            violations.append(
                f"evidence: tool_trace[{index}] error_class must be one of "
                f"{ERROR_CLASSES!r} or null"
            )
        request_args_hash = entry.get("request_args_hash")
        if request_args_hash is not None and not _is_sha256_hex(request_args_hash):
            violations.append(
                f"evidence: tool_trace[{index}] request_args_hash must be a "
                "64-character hex digest or null"
            )
    return violations


def validate_result_write_receipt(receipt: Any) -> list[str]:
    """Return structural violations for a controller result-write receipt.

    A valid-shape receipt confirms only: the ``m3-prompt-only-pilot`` scope,
    path ``/workspace/result.json``, creation/replacement, content SHA-256,
    JSON-object shape, and presence/type (string) of ``verification_key``. It
    must never carry or compare the key value.
    """
    violations: list[str] = []
    if not isinstance(receipt, Mapping):
        return ["receipt must be a JSON object"]
    for key in receipt:
        if key not in _RESULT_WRITE_RECEIPT_KEYS:
            violations.append(f"receipt: unknown key {key!r}")
    if receipt.get("schema_version") != RESULT_WRITE_RECEIPT_SCHEMA_VERSION:
        violations.append(
            f"receipt: unsupported schema_version {receipt.get('schema_version')!r}"
        )
    if receipt.get("pilot_scope") != RESULT_WRITE_PILOT_SCOPE:
        violations.append(
            f"receipt: pilot_scope must be {RESULT_WRITE_PILOT_SCOPE!r}"
        )
    if receipt.get("path") != RESULT_JSON_PATH:
        violations.append(f"receipt: path must be {RESULT_JSON_PATH!r}")
    if receipt.get("operation") not in {"created", "replaced"}:
        violations.append("receipt: operation must be 'created' or 'replaced'")
    if not _is_sha256_hex(receipt.get("content_sha256")):
        violations.append("receipt: content_sha256 must be a 64-character hex digest")
    if receipt.get("shape") != "json_object":
        violations.append("receipt: shape must be 'json_object'")
    key_shape = receipt.get("verification_key")
    if not isinstance(key_shape, Mapping) or set(key_shape) != {"present", "type"}:
        violations.append(
            "receipt: verification_key must be an object with exactly "
            "present and type"
        )
    else:
        if key_shape.get("present") is not True:
            violations.append("receipt: verification_key.present must be true")
        if key_shape.get("type") != "string":
            violations.append("receipt: verification_key.type must be 'string'")
    return violations


# ---------------------------------------------------------------------------
# Entry semantics (tool-aware)
# ---------------------------------------------------------------------------


def _budget_blocked(entry: Mapping[str, Any]) -> bool:
    return bool(entry.get("budget_rejected")) or bool(entry.get("operation_aborted"))


def _pre_rejected(entry: Mapping[str, Any]) -> bool:
    return bool(entry.get("pre_execution_rejected"))


def _infra(entry: Mapping[str, Any]) -> bool:
    return bool(entry.get("infrastructure_error"))


def _error_class(entry: Mapping[str, Any]) -> str | None:
    value = entry.get("error_class")
    return value if value in ERROR_CLASSES else None


def _is_tool_error(entry: Mapping[str, Any]) -> bool:
    """Return whether an executed tool call failed (for retry outcomes)."""
    if _budget_blocked(entry) or _pre_rejected(entry) or _infra(entry):
        return True
    error_class = _error_class(entry)
    if error_class is not None:
        return True
    if entry.get("is_error") is True:
        return True
    if _status_value_is_success(entry.get("tool_name"), entry.get("status")) is False:
        return True
    return False


def _is_eligible_ordinary_error(entry: Mapping[str, Any]) -> bool:
    """Return whether an executed tool call is an eligible recovery opportunity.

    Budget-limit errors, infrastructure failures, and never-executed
    pre-execution rejections are excluded; ordinary tool errors are eligible.
    """
    if _budget_blocked(entry) or _pre_rejected(entry) or _infra(entry):
        return False
    if not isinstance(entry.get("tool_name"), str) or not entry["tool_name"]:
        return False
    error_class = _error_class(entry)
    if error_class in ("budget_limit", "infrastructure", "pre_execution_rejection"):
        return False
    if error_class == "tool_error":
        return True
    if entry.get("is_error") is True:
        return True
    if _status_value_is_success(entry.get("tool_name"), entry.get("status")) is False:
        return True
    return False


def _succeeded(entry: Mapping[str, Any]) -> bool:
    return not _is_tool_error(entry)


def _result_write(entry: Mapping[str, Any]) -> bool:
    """Return whether the entry records a valid-shape result write.

    Errored, budget-blocked, pre-rejected, or infrastructure-failed submissions
    never count as valid writes.
    """
    if entry.get("result_submission") is not True:
        return False
    if _budget_blocked(entry) or _pre_rejected(entry) or _infra(entry):
        return False
    if _is_tool_error(entry):
        return False
    return True


def _fingerprint(entry: Mapping[str, Any]) -> tuple[Any, Any]:
    return (entry.get("tool_name"), entry.get("request_args_hash"))


# ---------------------------------------------------------------------------
# Completion classification
# ---------------------------------------------------------------------------


def _unknown_completion(diagnostics: list[str]) -> dict[str, Any]:
    return {
        "label": "unknown",
        "intended_behavior": None,
        "result_write_count": None,
        "trace_entry_count": None,
        "prior_budget_block": None,
        "prior_eligible_error": None,
        "post_submission_tool_attempts": None,
        "diagnostics": list(diagnostics),
    }


def classify_completion(
    evidence: Any,
    result_write_receipt: Any = None,
) -> dict[str, Any]:
    """Classify how an attempt ended, from restricted evidence alone.

    Receives only restricted attempt evidence and an optional controller
    result-write receipt. Never accepts treatment identity, verifier outcome,
    ``verified_success``, a private oracle/answer/key, or the task template.
    Fails closed to ``unknown`` on privacy violations, schema violations, or an
    invalid/inconsistent receipt. The receipt is validated in memory only and
    its content/hash are never copied into results.

    ``intended_behavior`` is true for exactly one valid-shape write, no prior
    budget block, and no later tool attempt — including a successful submission
    after an ordinary recoverable error.
    """
    privacy_violations = list(privacy_scan(evidence))
    if result_write_receipt is not None:
        privacy_violations.extend(privacy_scan(result_write_receipt))
    if privacy_violations:
        return _unknown_completion(
            [f"privacy violation: {violation}" for violation in privacy_violations]
        )

    schema_violations = validate_evidence(evidence)
    if schema_violations:
        return _unknown_completion(schema_violations)
    trace = list(evidence["tool_trace"])  # type: ignore[index]

    if result_write_receipt is not None:
        receipt_violations = validate_result_write_receipt(result_write_receipt)
        if receipt_violations:
            return _unknown_completion(
                ["invalid result-write receipt: " + v for v in receipt_violations]
            )

    writes: list[int] = [
        index for index, entry in enumerate(trace) if _result_write(entry)
    ]

    if result_write_receipt is not None and len(writes) != 1:
        return _unknown_completion(
            [
                "result-write receipt confirms one valid write but the trace "
                f"shows {len(writes)} valid write(s)"
            ]
        )

    if not writes:
        return {
            "label": "no_submission",
            "intended_behavior": False,
            "result_write_count": 0,
            "trace_entry_count": len(trace),
            "prior_budget_block": False,
            "prior_eligible_error": False,
            "post_submission_tool_attempts": 0,
            "diagnostics": [],
        }
    if len(writes) > 1:
        return {
            "label": "multiple_submissions",
            "intended_behavior": False,
            "result_write_count": len(writes),
            "trace_entry_count": len(trace),
            "prior_budget_block": False,
            "prior_eligible_error": False,
            "post_submission_tool_attempts": 0,
            "diagnostics": [],
        }

    write_index = writes[0]
    before = trace[:write_index]
    after = trace[write_index + 1 :]
    prior_budget_block = any(_budget_blocked(entry) for entry in before)
    prior_eligible_error = any(_is_eligible_ordinary_error(entry) for entry in before)
    post_submission_tool_attempts = len(after)
    intended_behavior = (
        not prior_budget_block and post_submission_tool_attempts == 0
    )

    if post_submission_tool_attempts > 0:
        return {
            "label": "post_submission_tool_activity",
            "intended_behavior": False,
            "result_write_count": 1,
            "trace_entry_count": len(trace),
            "prior_budget_block": prior_budget_block,
            "prior_eligible_error": prior_eligible_error,
            "post_submission_tool_attempts": post_submission_tool_attempts,
            "diagnostics": [],
        }
    if prior_budget_block or prior_eligible_error:
        return {
            "label": "submitted_after_prior_error",
            "intended_behavior": intended_behavior,
            "result_write_count": 1,
            "trace_entry_count": len(trace),
            "prior_budget_block": prior_budget_block,
            "prior_eligible_error": prior_eligible_error,
            "post_submission_tool_attempts": 0,
            "diagnostics": [],
        }
    return {
        "label": "submitted_before_budget_block",
        "intended_behavior": True,
        "result_write_count": 1,
        "trace_entry_count": len(trace),
        "prior_budget_block": False,
        "prior_eligible_error": False,
        "post_submission_tool_attempts": 0,
        "diagnostics": [],
    }


# ---------------------------------------------------------------------------
# Recovery classification
# ---------------------------------------------------------------------------


def _unknown_recovery(
    diagnostics: list[str],
    *,
    opportunity_count: int | None = None,
    retry_count: int = 0,
    changed_retry_count: int = 0,
    unchanged_repeat_count: int = 0,
    later_success: bool = False,
) -> dict[str, Any]:
    return {
        "label": "unknown",
        "opportunity_count": opportunity_count,
        "retry_count": retry_count,
        "changed_retry_count": changed_retry_count,
        "unchanged_repeat_count": unchanged_repeat_count,
        "later_success": later_success,
        "diagnostics": list(diagnostics),
    }


def classify_recovery(evidence: Any) -> dict[str, Any]:
    """Classify the response to eligible ordinary tool errors.

    Opportunities are built only from eligible ordinary tool errors/rejections
    (budget-limit errors, infrastructure failures, and never-executed
    pre-execution rejections are excluded). Retry comparison uses canonical
    request fingerprints ``(tool_name, args_sha256)`` and never emits raw
    arguments. Fails closed to ``unknown`` on privacy/schema violations and
    when fingerprints cannot be verified.
    """
    privacy_violations = privacy_scan(evidence)
    if privacy_violations:
        return _unknown_recovery(
            [f"privacy violation: {violation}" for violation in privacy_violations]
        )
    schema_violations = validate_evidence(evidence)
    if schema_violations:
        return _unknown_recovery(schema_violations)
    trace = list(evidence["tool_trace"])  # type: ignore[index]

    opportunity_indices = [
        index
        for index, entry in enumerate(trace)
        if _is_eligible_ordinary_error(entry)
    ]
    if not opportunity_indices:
        return {
            "label": "no_opportunity",
            "opportunity_count": 0,
            "retry_count": 0,
            "changed_retry_count": 0,
            "unchanged_repeat_count": 0,
            "later_success": False,
            "diagnostics": [],
        }

    first_index = opportunity_indices[0]
    error_entry = trace[first_index]
    rest = trace[first_index + 1 :]
    if not rest:
        return {
            "label": "abandoned_after_error",
            "opportunity_count": len(opportunity_indices),
            "retry_count": 0,
            "changed_retry_count": 0,
            "unchanged_repeat_count": 0,
            "later_success": False,
            "diagnostics": [],
        }

    later_success = any(not _is_tool_error(entry) for entry in rest)
    error_tool = error_entry.get("tool_name")
    # A result submission is a terminal act, not a retry of the failed request,
    # so submission markers are excluded from retry comparison.
    retries = [
        (index, entry)
        for index, entry in enumerate(rest, start=first_index + 1)
        if entry.get("tool_name") == error_tool
        and entry.get("result_submission") is not True
    ]
    retry_count = len(retries)
    if retry_count == 0:
        return _unknown_recovery(
            ["no same-tool retry after the first eligible error"],
            opportunity_count=len(opportunity_indices),
            later_success=later_success,
        )

    comparison_entries = [error_entry] + [entry for _, entry in retries]
    if any(entry.get("request_args_hash") is None for entry in comparison_entries):
        return _unknown_recovery(
            ["request_args_hash missing: cannot verify unchanged vs changed retry"],
            opportunity_count=len(opportunity_indices),
            retry_count=retry_count,
            later_success=later_success,
        )

    error_fp = _fingerprint(error_entry)
    unchanged = [
        (index, entry) for index, entry in retries if _fingerprint(entry) == error_fp
    ]
    changed = [
        (index, entry) for index, entry in retries if _fingerprint(entry) != error_fp
    ]
    changed_count = len(changed)
    unchanged_count = len(unchanged)

    base = {
        "opportunity_count": len(opportunity_indices),
        "retry_count": retry_count,
        "changed_retry_count": changed_count,
        "unchanged_repeat_count": unchanged_count,
        "later_success": later_success,
        "diagnostics": [],
    }

    if retry_count >= 2:
        return {"label": "retry_loop", **base}

    retry_index, retry_entry = retries[0]
    if changed_count == 1:
        if _is_tool_error(retry_entry):
            after_retry = trace[retry_index + 1 :]
            if not after_retry:
                return {"label": "corrected_once_failed_then_stopped", **base}
            return {
                "label": "unknown",
                **base,
                "diagnostics": ["changed retry failed but the agent continued"],
            }
        return {"label": "corrected_once_success", **base}

    return {"label": "unchanged_repeat", **base}


# ---------------------------------------------------------------------------
# Self-hashed behavior receipt
# ---------------------------------------------------------------------------


def analyze_attempt(
    evidence: Any,
    result_write_receipt: Any = None,
) -> dict[str, Any]:
    """Return a canonical self-hashed behavior receipt for one attempt.

    The receipt carries schema/classifier source identity, bounded
    labels/counters only, and the unconditional ITT-inclusion marker. It never
    contains raw text, arguments, request-argument hashes/fingerprints, keys,
    outcomes, treatment identity, the task template, or the result-write
    receipt's ``content_sha256`` (the result-write receipt is validated in
    memory only and is never persisted into the receipt). ``receipt_hash`` is
    the canonical SHA-256 of the rest of the payload.
    """
    completion = classify_completion(evidence, result_write_receipt)
    recovery = classify_recovery(evidence)
    provider_turn_count = None
    if isinstance(evidence, Mapping):
        value = evidence.get("provider_turn_count")
        if _is_int(value):
            provider_turn_count = value
    payload = {
        "schema_version": BEHAVIOR_RECEIPT_SCHEMA_VERSION,
        "classifier_source": CLASSIFIER_SOURCE,
        "classifier_source_sha256": module_source_sha256(),
        "itt_inclusion": "unconditional",
        "provider_turn_count": provider_turn_count,
        "completion": {
            "label": completion["label"],
            "intended_behavior": completion["intended_behavior"],
            "result_write_count": completion["result_write_count"],
            "prior_budget_block": completion["prior_budget_block"],
            "prior_eligible_error": completion["prior_eligible_error"],
            "post_submission_tool_attempts": completion[
                "post_submission_tool_attempts"
            ],
        },
        "recovery": {
            "label": recovery["label"],
            "opportunity_count": recovery["opportunity_count"],
            "retry_count": recovery["retry_count"],
            "changed_retry_count": recovery["changed_retry_count"],
            "unchanged_repeat_count": recovery["unchanged_repeat_count"],
            "later_success": recovery["later_success"],
        },
    }
    return {**payload, "receipt_hash": _canonical_sha256(payload)}


def validate_behavior_receipt(receipt: Any) -> list[str]:
    """Return violations for a self-hashed behavior receipt (tamper detection).

    Checks schema identity, classifier source identity, embedded hash
    consistency, bounded label/counter types, and privacy cleanliness.
    """
    violations: list[str] = []
    if not isinstance(receipt, Mapping):
        return ["behavior receipt must be a JSON object"]
    for key in receipt:
        if key not in _BEHAVIOR_RECEIPT_KEYS:
            violations.append(f"behavior receipt: unknown key {key!r}")
    if receipt.get("schema_version") != BEHAVIOR_RECEIPT_SCHEMA_VERSION:
        violations.append("behavior receipt: unsupported schema_version")
    if receipt.get("classifier_source") != CLASSIFIER_SOURCE:
        violations.append("behavior receipt: classifier_source mismatch")
    if receipt.get("classifier_source_sha256") != module_source_sha256():
        violations.append("behavior receipt: classifier_source_sha256 mismatch")
    if receipt.get("itt_inclusion") != "unconditional":
        violations.append("behavior receipt: itt_inclusion must be 'unconditional'")

    embedded = dict(receipt)
    embedded_hash = embedded.pop("receipt_hash", None)
    if not _is_sha256_hex(embedded_hash):
        violations.append("behavior receipt: missing or malformed receipt_hash")
    elif embedded_hash != _canonical_sha256(embedded):
        violations.append(
            "behavior receipt: embedded receipt_hash does not match canonical payload"
        )

    provider_turn_count = receipt.get("provider_turn_count")
    if provider_turn_count is not None and not _is_int(provider_turn_count):
        violations.append(
            "behavior receipt: provider_turn_count must be an integer or null"
        )

    completion = receipt.get("completion")
    if not isinstance(completion, Mapping):
        violations.append("behavior receipt: completion must be an object")
    elif completion.get("label") not in COMPLETION_LABELS:
        violations.append(
            f"behavior receipt: unknown completion label {completion.get('label')!r}"
        )

    recovery = receipt.get("recovery")
    if not isinstance(recovery, Mapping):
        violations.append("behavior receipt: recovery must be an object")
    elif recovery.get("label") not in RECOVERY_LABELS:
        violations.append(
            f"behavior receipt: unknown recovery label {recovery.get('label')!r}"
        )

    for counter_name in (
        "result_write_count",
        "post_submission_tool_attempts",
    ):
        value = (completion or {}).get(counter_name)
        if value is not None and not _is_int(value):
            violations.append(
                f"behavior receipt: completion.{counter_name} must be an integer or null"
            )
    for counter_name in (
        "opportunity_count",
        "retry_count",
        "changed_retry_count",
        "unchanged_repeat_count",
    ):
        value = (recovery or {}).get(counter_name)
        if value is not None and not _is_int(value):
            violations.append(
                f"behavior receipt: recovery.{counter_name} must be an integer or null"
            )

    for label, obj in (("completion", completion), ("recovery", recovery)):
        if isinstance(obj, Mapping):
            for flag in (
                "intended_behavior",
                "prior_budget_block",
                "prior_eligible_error",
                "later_success",
            ):
                value = obj.get(flag)
                if value is not None and not isinstance(value, bool):
                    violations.append(
                        f"behavior receipt: {label}.{flag} must be a boolean or null"
                    )

    violations.extend(privacy_scan(receipt))
    return violations


__all__ = [
    "BEHAVIOR_RECEIPT_SCHEMA_VERSION",
    "CLASSIFIER_SOURCE",
    "COMPLETION_LABELS",
    "ERROR_CLASSES",
    "PILOT_TOOLS",
    "RECOVERY_LABELS",
    "RESULT_JSON_PATH",
    "RESULT_WRITE_PILOT_SCOPE",
    "RESULT_WRITE_RECEIPT_SCHEMA_VERSION",
    "RESTRICTED_EVIDENCE_SCHEMA_VERSION",
    "RestrictedEvidenceError",
    "analyze_attempt",
    "build_restricted_evidence",
    "canonical_args_hash",
    "classify_completion",
    "classify_recovery",
    "extract_request_args_hashes",
    "module_source_sha256",
    "privacy_scan",
    "validate_behavior_receipt",
    "validate_evidence",
    "validate_result_write_receipt",
]
