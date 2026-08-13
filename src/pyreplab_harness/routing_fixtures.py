"""Deterministic utility-routing fixture family (Stage A + Stage B backend).

This module implements the versioned routing fixture family described in
Section 4 of ``notes/m3-utility-routing-smoke-plan.md`` and is the design
backend consumed by ``m3_routing_probe_gate``:

* ``build_stage_a_design()`` -> list of 32 *private* design coordinates
  (8 per stratum), each carrying an opaque ``fixture_id``, ``stratum``,
  ``first_bottleneck``, ``operation_flags``, ``difficulty``, ``seed``, and
  private verifier metadata (nonce + oracle) that must never reach a model.
* ``build_stage_b_design()`` -> list of 24 *private* Stage-B design
  coordinates in two independently seeded blocks (12 per block, 3 per stratum,
  one easy/medium/hard per block-stratum, 6 table / 6 form preferred per
  block, mixed/ambiguous 2/1 capability splits reversed across blocks).
  Coordinates additionally carry a ``generator_version``, ``block``, and
  ``state`` transition metadata for server/gym integration.
* ``generate_routing_fixture(coord)`` -> a *public* fixture record with the
  initial HTML (which always embeds both a bounded table cue and a bounded
  form cue), a declared task prompt, an opaque id, and the audit-only source
  hash.  It deliberately omits the stratum, difficulty, seed, first bottleneck,
  template id, nonce, and oracle.

Mixed dependency semantics (corrected for Stage B, retrofitted into Stage A):

* ``table_first``: the directory table exposes the target row's access code
  (the nonce); the form requires that access code and, on success, echoes the
  nonce as a confirmation key.
* ``form_first``: the initial form requires no table-derived access code; a
  successful submission returns a deterministic *record reference*; the
  directory table then requires that reference (via ``record_reference``) to
  unlock and reveal the final nonce.  The nonce and reference never appear in
  the initial HTML for form-first tasks.

All forms are rendered with ``method="get"`` because the semantic browser
capability core only supports GET submissions.

Anti-leakage contract
---------------------
The private design object is the list returned by ``build_stage_a_design``; the
public, model-visible surface is what ``generate_routing_fixture`` returns plus
``model_visible(coord)``.  ``model_visible`` exposes only ``fixture_id`` and the
declared ``operation_flags`` -- never stratum, difficulty, seed, template id,
bottleneck label, nonce, or oracle.

Determinism
-----------
Every artifact is derived from SHA-256 digests over explicit identity parts.
The content RNG is seeded by ``(seed, stratum, difficulty, preferred_capability)``
(not the text variant), so a text variant changes only visible text while
preserving DOM structure and the answer placement.

Only the standard library is used.
"""

from __future__ import annotations

import hashlib
import html
import random
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Mapping, Sequence

GENERATOR_VERSION = "routing-fixtures-v1"
"""Version of the Stage-A generator family (frozen; do not change)."""

STAGE_B_GENERATOR_VERSION = "routing-fixtures-v2"
"""Version of the Stage-B generator family (additive; Stage A is untouched)."""

STAGE_B_SCHEMA = "m3-routing-stage-b-design-v1"
"""Schema label for the Stage-B design coordinates."""

STRATA: tuple[str, ...] = ("pure_table", "pure_form", "mixed", "ambiguous")
"""The four task strata."""

DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard")
"""Structural-complexity levels."""

PREFERRED_CAPABILITIES: tuple[str, ...] = ("table", "form")
"""Preferred-specialist capabilities."""

TEXT_VARIANTS: tuple[int, ...] = (0, 1)
"""Supported text-variant indices (0 = primary, 1 = alternate)."""

DEFAULT_STAGE_A_SEED = 20260813
"""Default master seed for the Stage-A design."""

DEFAULT_STAGE_B_SEED = 20260814
"""Default master seed for the Stage-B design (distinct from Stage A)."""

BLOCK_COUNT = 2
"""Number of independently seeded Stage-B blocks."""

TASKS_PER_BLOCK = 12
"""Stage-B coordinates per block (3 strata-free tasks x 4 strata)."""

TASKS_PER_STRATUM_PER_BLOCK = 3
"""Stage-B coordinates per stratum per block (one per difficulty)."""

RECORD_REFERENCE_PARAM = "record_reference"
"""Query-parameter key that unlocks the form-first table (final nonce)."""

# Opaque specialist labels used by the frozen router/gate.
_SPECIALIST_BY_CAPABILITY = {"table": "table_specialist", "form": "form_specialist"}


# ---------------------------------------------------------------------------
# deterministic hashing helpers
# ---------------------------------------------------------------------------


def _digest(*parts: object) -> str:
    """Deterministic SHA-256 hex digest over stringified identity parts."""
    raw = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _seed_int(*parts: object) -> int:
    """Deterministic non-negative integer derived from identity parts."""
    return int(_digest("seed", *parts)[:16], 16)


def _rng(*parts: object) -> random.Random:
    """Deterministic stdlib RNG seeded from identity parts."""
    return random.Random(_seed_int("rng", *parts))


def _make_nonce(
    seed: int,
    stratum: str,
    difficulty: str,
    capability: str,
    *,
    generator_version: str = GENERATOR_VERSION,
) -> str:
    """Deterministic per-design nonce (text-variant independent).

    The Stage-B generator uses a distinct digest namespace so its nonces can
    never collide with Stage-A nonces even for equal identity parts.
    """
    if generator_version == STAGE_B_GENERATOR_VERSION:
        return "RF-" + _digest(
            "nonce", "stage-b", seed, stratum, difficulty, capability
        )[:12].upper()
    return "RF-" + _digest("nonce", seed, stratum, difficulty, capability)[:12].upper()


def _make_fixture_id(
    seed: int,
    stratum: str,
    difficulty: str,
    capability: str,
    *,
    generator_version: str = GENERATOR_VERSION,
) -> str:
    """Opaque deterministic fixture id (reveals nothing about the design)."""
    if generator_version == STAGE_B_GENERATOR_VERSION:
        return "rf-" + _digest(
            "fixture", "stage-b", seed, stratum, difficulty, capability
        )[:16]
    return "rf-" + _digest("fixture", seed, stratum, difficulty, capability)[:16]


def _make_reference(
    seed: int, stratum: str, difficulty: str, capability: str,
) -> str:
    """Deterministic record reference returned by a form-first submission."""
    return "REF-" + _digest(
        "reference", "stage-b", seed, stratum, difficulty, capability
    )[:10].upper()


def sealed_label(coord: Mapping[str, Any]) -> str:
    """Deterministic seal over the private design identity (excludes variant)."""
    return _digest(
        "sealed-design-v1",
        coord.get("generator_version", GENERATOR_VERSION),
        coord["template_id"],
        coord["stratum"],
        coord["preferred_capability"],
        coord["first_bottleneck"],
        coord["difficulty"],
        coord["seed"],
        coord["nonce"],
    )


# ---------------------------------------------------------------------------
# private design geometry
# ---------------------------------------------------------------------------

# Per-stratum (capability, difficulty) layout: 8 coords, balanced 4/4 capability
# and 3/3/2 complexity.
_STRATUM_LAYOUT: dict[str, tuple[tuple[str, str], ...]] = {
    "pure_table": (
        ("table", "easy"), ("table", "easy"), ("table", "easy"),
        ("table", "medium"), ("table", "medium"), ("table", "medium"),
        ("table", "hard"), ("table", "hard"),
    ),
    "pure_form": (
        ("form", "easy"), ("form", "easy"), ("form", "easy"),
        ("form", "medium"), ("form", "medium"), ("form", "medium"),
        ("form", "hard"), ("form", "hard"),
    ),
    "mixed": (
        ("table", "easy"), ("form", "easy"), ("table", "easy"),
        ("form", "medium"), ("table", "medium"), ("form", "medium"),
        ("table", "hard"), ("form", "hard"),
    ),
    "ambiguous": (
        ("table", "easy"), ("form", "easy"), ("table", "easy"),
        ("form", "medium"), ("table", "medium"), ("form", "medium"),
        ("table", "hard"), ("form", "hard"),
    ),
}

_TEMPLATE_IDS: dict[str, str] = {
    "pure_table": "routing_pure_table_v1",
    "pure_form": "routing_pure_form_v1",
    "mixed": "routing_mixed_v1",
    "ambiguous": "routing_ambiguous_v1",
}

_STAGE_B_TEMPLATE_IDS: dict[str, str] = {
    "pure_table": "routing_pure_table_v2",
    "pure_form": "routing_pure_form_v2",
    "mixed": "routing_mixed_v2",
    "ambiguous": "routing_ambiguous_v2",
}


def _operation_flags(stratum: str, capability: str) -> dict[str, Any]:
    """Declared-operation flags conforming to the gate's per-stratum contract."""
    if stratum == "pure_table":
        return {"table_operation": True, "form_operation": False, "first_operation": None}
    if stratum == "pure_form":
        return {"table_operation": False, "form_operation": True, "first_operation": None}
    if stratum == "mixed":
        return {
            "table_operation": True,
            "form_operation": True,
            "first_operation": "table" if capability == "table" else "form",
        }
    # ambiguous: the declared operation is the (opposite) requested bottleneck.
    return {
        "table_operation": capability == "table",
        "form_operation": capability == "form",
        "first_operation": None,
    }


# ---------------------------------------------------------------------------
# content model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Field:
    name: str
    label: str
    kind: str
    hint: str
    required: bool
    pattern: str | None = None
    options: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Content:
    nonce: str
    stratum: str
    difficulty: str
    capability: str
    dependency_order: str | None
    # table
    table_headers: tuple[str, ...]
    table_rows: tuple[tuple[str, ...], ...]
    code_columns: tuple[int, ...]
    target_row_index: int | None
    target_column: int | None
    target_row_label: str | None
    target_row_department: str | None
    # form
    form_fields: tuple[_Field, ...]
    form_method: str
    correct_values: tuple[tuple[str, str], ...]
    # semantics
    table_relevant: bool
    form_relevant: bool
    # form-first reference unlock (Stage B / repaired mixed form-first)
    reference: str | None = None
    reference_column: int | None = None


_RECORDS_COLUMNS: tuple[str, ...] = ("Name", "Department", "Serial Number", "Access Code")
_FORM_FIRST_COLUMNS: tuple[str, ...] = (
    "Name", "Department", "Reference Number", "Access Code",
)
_IRRELEVANT_COLUMNS: tuple[str, ...] = ("Item", "Status", "Last Updated")
_CODE_COLUMNS: tuple[int, ...] = (2, 3)

_TABLE_ROWS = {"easy": 4, "medium": 8, "hard": 14}
_FORM_FIELDS = {"easy": 2, "medium": 4, "hard": 6}
# ambiguous distractor sizes: the irrelevant structure is the larger one.
_AMBIG_TABLE_ROWS = {"easy": 2, "medium": 3, "hard": 4}
_AMBIG_FORM_FIELDS = {"easy": 2, "medium": 3, "hard": 4}
_AMBIG_DISTRACTOR_TABLE_ROWS = {"easy": 10, "medium": 16, "hard": 22}
_AMBIG_DISTRACTOR_FORM_FIELDS = {"easy": 8, "medium": 10, "hard": 12}

_FIRST_NAMES = (
    "Avery", "Blake", "Cameron", "Dana", "Ellis", "Finley", "Gray",
    "Harper", "Ira", "Jordan", "Kai", "Lee", "Morgan", "Noel", "Parker", "Quinn",
)
_LAST_NAMES = (
    "Chen", "Davis", "Edwards", "Fisher", "Garcia", "Hughes", "Ito", "Jensen",
    "Kim", "Liu", "Martinez", "Nguyen", "Okafor", "Patel", "Rivera", "Singh",
)
_DEPARTMENTS = ("Engineering", "Marketing", "Finance", "Operations", "Research", "Support")
_EQUIPMENT = ("Printer", "Scanner", "Terminal", "Monitor", "Hub", "Node", "Console", "Relay")

_FORM_FIELD_SPECS: tuple[dict[str, Any], ...] = (
    {"name": "full_name", "label": "Full Name", "hint": "Your legal name",
     "required": True, "pattern": None},
    {"name": "email", "label": "Email Address", "hint": "e.g. user@example.com",
     "required": True, "pattern": r"^[^@\s]+@[^@\s]+\.[^@\s]+$"},
    {"name": "reference", "label": "Reference Number", "hint": "Format: REF-12345",
     "required": True, "pattern": r"^REF-\d{5}$"},
    {"name": "department", "label": "Department", "hint": "Select your department",
     "required": True, "pattern": None,
     "options": ("Engineering", "Marketing", "Finance", "Operations")},
    {"name": "clearance", "label": "Security Clearance",
     "hint": "Format: LVL-A, LVL-B, or LVL-C", "required": True,
     "pattern": r"^LVL-[ABC]$"},
    {"name": "project_code", "label": "Project Code", "hint": "Optional; format PRJ-123",
     "required": False, "pattern": r"^PRJ-\d{3}$"},
)

_ACCESS_CODE_FIELD = _Field(
    name="access_code", label="Access Code", kind="text",
    hint="From the directory table", required=True, pattern=None,
)

# Form-first mixed forms must not prompt for a user-supplied reference number
# (the reference is *returned* by the server), so the "reference" input field
# is excluded from the form-first field pool.
_FORM_FIRST_FIELD_SPECS: tuple[dict[str, Any], ...] = tuple(
    spec for spec in _FORM_FIELD_SPECS if spec["name"] != "reference"
)


def _fields_from_specs(specs: Sequence[Mapping[str, Any]]) -> tuple[_Field, ...]:
    fields = []
    for spec in specs:
        fields.append(
            _Field(
                name=str(spec["name"]),
                label=str(spec["label"]),
                kind="select" if spec.get("options") else "text",
                hint=str(spec["hint"]),
                required=bool(spec.get("required", True)),
                pattern=spec.get("pattern"),
                options=tuple(spec["options"]) if spec.get("options") else (),
            )
        )
    return tuple(fields)


def _generic_fields(count: int) -> tuple[_Field, ...]:
    return tuple(
        _Field(
            name=f"field_{i + 1}", label=f"Field {i + 1}", kind="text",
            hint="Optional response", required=False, pattern=None,
        )
        for i in range(count)
    )


def _records_table(
    rng: random.Random, nrows: int, nonce: str,
) -> tuple[tuple[tuple[str, ...], ...], int]:
    rows: list[list[str]] = []
    for _ in range(nrows):
        name = f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}"
        dept = rng.choice(_DEPARTMENTS)
        serial = f"EMP-{rng.randint(10000, 99999)}"
        code = f"CODE_{rng.randint(10**7, 10**8 - 1):08x}"
        rows.append([name, dept, serial, code])
    target_idx = rng.randrange(nrows)
    rows[target_idx][3] = nonce
    return tuple(tuple(row) for row in rows), target_idx


def _records_table_form_first(
    rng: random.Random, nrows: int,
) -> tuple[tuple[tuple[str, ...], ...], int]:
    """Directory table for a form-first mixed task.

    The target row's Reference Number is initially ``PENDING`` and its Access
    Code is ``LOCKED``: neither the returned reference nor the final nonce may
    be visible before the form is submitted and the reference is supplied back.
    """
    rows: list[list[str]] = []
    for _ in range(nrows):
        name = f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}"
        dept = rng.choice(_DEPARTMENTS)
        ref = f"REF-{rng.randint(10000, 99999)}"
        code = f"CODE_{rng.randint(10**7, 10**8 - 1):08x}"
        rows.append([name, dept, ref, code])
    target_idx = rng.randrange(nrows)
    rows[target_idx][2] = "PENDING"
    rows[target_idx][3] = "LOCKED"
    return tuple(tuple(row) for row in rows), target_idx


def _distractor_rows(rng: random.Random, nrows: int) -> tuple[tuple[str, ...], ...]:
    rows = []
    for _ in range(nrows):
        item = f"{rng.choice(_EQUIPMENT)}-{rng.randint(10, 99)}"
        status = rng.choice(("Online", "Offline", "Maintenance", "Standby"))
        updated = f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        rows.append((item, status, updated))
    return tuple(rows)


def _correct_values_for(
    rng: random.Random,
    fields: tuple[_Field, ...],
    access_code: str | None,
) -> tuple[tuple[str, str], ...]:
    pairs = []
    for field in fields:
        if field.name == "access_code":
            value = access_code
        elif field.options:
            value = field.options[0]
        elif field.name == "full_name":
            value = "Jordan Lee"
        elif field.name == "email":
            value = f"user{rng.randint(1000, 9999)}@example.com"
        elif field.name == "reference":
            value = f"REF-{rng.randint(10000, 99999)}"
        elif field.name == "clearance":
            value = rng.choice(("LVL-A", "LVL-B", "LVL-C"))
        elif field.name == "project_code":
            value = f"PRJ-{rng.randint(100, 999)}"
        else:
            value = f"value-{rng.randint(1000, 9999)}"
        pairs.append((field.name, value))
    return tuple(pairs)


def _compute_content(
    seed: int,
    stratum: str,
    difficulty: str,
    capability: str,
    *,
    generator_version: str = GENERATOR_VERSION,
) -> _Content:
    nonce = _make_nonce(
        seed, stratum, difficulty, capability, generator_version=generator_version
    )
    rng = _rng("content", seed, stratum, difficulty, capability)

    table_headers = _RECORDS_COLUMNS
    code_columns = _CODE_COLUMNS
    table_rows: tuple[tuple[str, ...], ...] = ()
    target_row_index: int | None = None
    target_column: int | None = None
    target_row_label: str | None = None
    target_row_department: str | None = None
    form_fields: tuple[_Field, ...] = ()
    correct_values: tuple[tuple[str, str], ...] = ()
    table_relevant = False
    form_relevant = False
    dependency_order: str | None = None
    form_method = "get"
    reference: str | None = None
    reference_column: int | None = None

    if stratum == "pure_table":
        table_relevant = True
        table_rows, target_row_index = _records_table(rng, _TABLE_ROWS[difficulty], nonce)
        target_column = 3
        form_fields = _fields_from_specs(_FORM_FIELD_SPECS[:2])
    elif stratum == "pure_form":
        form_relevant = True
        table_rows = _distractor_rows(rng, 3)
        table_headers = _IRRELEVANT_COLUMNS
        code_columns = ()
        form_fields = _fields_from_specs(_FORM_FIELD_SPECS[:_FORM_FIELDS[difficulty]])
        correct_values = _correct_values_for(rng, form_fields, None)
        form_method = "get"
    elif stratum == "mixed":
        table_relevant = True
        form_relevant = True
        dependency_order = "table_first" if capability == "table" else "form_first"
        form_method = "get"
        if capability == "table":
            # table-first: the access code is visible in the table; the form
            # requires it and echoes the nonce on success.
            table_rows, target_row_index = _records_table(rng, _TABLE_ROWS[difficulty], nonce)
            target_column = 3
            form_fields = _fields_from_specs(_FORM_FIELD_SPECS[:_FORM_FIELDS[difficulty]]) + (
                _ACCESS_CODE_FIELD,
            )
            correct_values = _correct_values_for(rng, form_fields, nonce)
        else:
            # form-first: the form returns a reference; the table stays locked
            # until that reference is supplied back to reveal the nonce.
            reference = _make_reference(seed, stratum, difficulty, capability)
            table_headers = _FORM_FIRST_COLUMNS
            table_rows, target_row_index = _records_table_form_first(
                rng, _TABLE_ROWS[difficulty]
            )
            target_column = 3
            reference_column = 2
            form_fields = _fields_from_specs(_FORM_FIRST_FIELD_SPECS[:_FORM_FIELDS[difficulty]])
            correct_values = _correct_values_for(rng, form_fields, None)
    else:  # ambiguous
        if capability == "table":
            table_relevant = True
            table_rows, target_row_index = _records_table(rng, _AMBIG_TABLE_ROWS[difficulty], nonce)
            target_column = 3
            form_fields = _generic_fields(_AMBIG_DISTRACTOR_FORM_FIELDS[difficulty])
        else:
            form_relevant = True
            table_rows = _distractor_rows(rng, _AMBIG_DISTRACTOR_TABLE_ROWS[difficulty])
            table_headers = _IRRELEVANT_COLUMNS
            code_columns = ()
            form_fields = _fields_from_specs(_FORM_FIELD_SPECS[:_AMBIG_FORM_FIELDS[difficulty]])
            correct_values = _correct_values_for(rng, form_fields, None)
            form_method = "get"

    if target_row_index is not None:
        target_row_label = table_rows[target_row_index][0]
        target_row_department = table_rows[target_row_index][1]

    return _Content(
        nonce=nonce,
        stratum=stratum,
        difficulty=difficulty,
        capability=capability,
        dependency_order=dependency_order,
        table_headers=table_headers,
        table_rows=table_rows,
        code_columns=code_columns,
        target_row_index=target_row_index,
        target_column=target_column,
        target_row_label=target_row_label,
        target_row_department=target_row_department,
        form_fields=form_fields,
        form_method=form_method,
        correct_values=correct_values,
        table_relevant=table_relevant,
        form_relevant=form_relevant,
        reference=reference,
        reference_column=reference_column,
    )


# ---------------------------------------------------------------------------
# surface text (text variants preserve DOM structure)
# ---------------------------------------------------------------------------

_VARIANT_TEXT: dict[int, dict[str, str]] = {
    0: {
        "org": "Delta Corp",
        "portal": "Operations Portal",
        "intro": "Internal records &mdash; authorized personnel only.",
        "table_heading": "Staff Directory",
        "form_heading": "Request Form",
        "submit_label": "Submit",
    },
    1: {
        "org": "Nimbus LLC",
        "portal": "Staff Console",
        "intro": "Confidential systems &mdash; restricted access.",
        "table_heading": "Personnel Records",
        "form_heading": "Submission Form",
        "submit_label": "Send",
    },
}


def _prompt_text(content: _Content) -> str:
    target = content.target_row_label
    if content.stratum == "pure_table":
        return (
            f"Locate the record for {target} in the staff directory table "
            f"and report its Access Code."
        )
    if content.stratum == "pure_form":
        return (
            "Complete the request form with valid values and submit it. "
            "Report the verification key shown on the success page."
        )
    if content.stratum == "mixed":
        if content.dependency_order == "table_first":
            return (
                f"Look up the Access Code for {target} in the directory table, "
                f"then enter it into the verification form and submit it to "
                f"receive your confirmation key."
            )
        return (
            "Complete and submit the verification form with valid values to "
            "receive a record reference, then use that reference to locate the "
            "confirmation key in the directory table and report it."
        )
    # ambiguous
    if content.capability == "table":
        return f"Report the Access Code for {target} shown in the records table."
    return (
        "Complete the request form with valid values and submit it. "
        "Report the verification key shown on the success page."
    )


_HTML_TEMPLATE = (
    "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
    "<title>{title}</title>\n</head>\n<body>\n{body}</body>\n</html>\n"
)


def _table_html(content: _Content) -> str:
    thead = "".join(f"<th>{html.escape(h)}</th>" for h in content.table_headers)
    tbody = ""
    for row in content.table_rows:
        cells = ""
        for index, cell in enumerate(row):
            text = html.escape(cell)
            if index in content.code_columns:
                cells += f"<td><code>{text}</code></td>"
            else:
                cells += f"<td>{text}</td>"
        tbody += f"<tr>{cells}</tr>\n"
    return f"<table>\n<thead><tr>{thead}</tr></thead>\n<tbody>\n{tbody}</tbody>\n</table>\n"


def _form_html(content: _Content, submit_label: str) -> str:
    out = f'<form method="{content.form_method}" action="">\n'
    for field in content.form_fields:
        out += f'<label for="{html.escape(field.name)}">{html.escape(field.label)}</label>\n'
        if field.kind == "select":
            opts = "".join(
                f'<option value="{html.escape(o)}">{html.escape(o)}</option>'
                for o in field.options
            )
            req = " required" if field.required else ""
            out += (
                f'<select id="{html.escape(field.name)}" name="{html.escape(field.name)}"{req}>\n'
                f'<option value="">-- Select --</option>\n{opts}</select>\n'
            )
        else:
            req = " required" if field.required else ""
            out += (
                f'<input type="text" id="{html.escape(field.name)}" '
                f'name="{html.escape(field.name)}"{req} '
                f'placeholder="{html.escape(field.hint)}">\n'
            )
        out += f'<span class="note">{html.escape(field.hint)}</span>\n'
    out += f'<button type="submit">{html.escape(submit_label)}</button>\n</form>\n'
    return out


def _build_html(content: _Content, text_variant: int) -> str:
    text = _VARIANT_TEXT[text_variant]
    title = f"{text['org']} {text['portal']}"
    prompt = _prompt_text(content)
    body = (
        f'<h1>{html.escape(text["org"])} {html.escape(text["portal"])}</h1>\n'
        f'<p class="note">{text["intro"]}</p>\n'
        f'<h2>{html.escape(text["table_heading"])}</h2>\n'
        f"{_table_html(content)}\n"
        f'<h2>{html.escape(text["form_heading"])}</h2>\n'
        f"{_form_html(content, text['submit_label'])}\n"
        f'<p class="task">Task: {html.escape(prompt)}</p>\n'
    )
    return _HTML_TEMPLATE.format(title=html.escape(title), body=body), title, prompt


# ---------------------------------------------------------------------------
# public API consumed by the gate
# ---------------------------------------------------------------------------


def build_stage_a_design(seed: int = DEFAULT_STAGE_A_SEED) -> list[dict[str, Any]]:
    """Return the fixed 32-coordinate private Stage-A design.

    Exactly eight coordinates per stratum; balanced preferred capability
    (4 table / 4 form for mixed and ambiguous) and balanced complexity
    (3 easy / 3 medium / 2 hard per stratum).  Each coordinate is a private
    mapping carrying the opaque ``fixture_id``, ``stratum``, ``first_bottleneck``,
    ``operation_flags``, ``difficulty``, ``seed``, and private verifier metadata.
    """
    coordinates: list[dict[str, Any]] = []
    ordinal = 0
    for stratum in STRATA:
        for capability, difficulty in _STRATUM_LAYOUT[stratum]:
            fixture_seed = _seed_int("stage-a", seed, ordinal)
            nonce = _make_nonce(fixture_seed, stratum, difficulty, capability)
            content = _compute_content(fixture_seed, stratum, difficulty, capability)
            first_bottleneck = _SPECIALIST_BY_CAPABILITY[capability]
            coordinates.append(
                {
                    "fixture_id": _make_fixture_id(
                        fixture_seed, stratum, difficulty, capability
                    ),
                    "stratum": stratum,
                    "preferred_capability": capability,
                    "first_bottleneck": first_bottleneck,
                    "operation_flags": _operation_flags(stratum, capability),
                    "difficulty": difficulty,
                    "seed": fixture_seed,
                    "template_id": _TEMPLATE_IDS[stratum],
                    "nonce": nonce,
                    "dependency_order": content.dependency_order,
                    "oracle": _oracle_dict(content),
                }
            )
            ordinal += 1
    return coordinates


def _stage_b_block_layout(block_index: int) -> dict[str, tuple[tuple[str, str], ...]]:
    """Per-block (capability, difficulty) layout for the Stage-B design.

    Each stratum contributes three tasks (one easy / one medium / one hard).
    Mixed and ambiguous strata carry a 2/1 capability split that is reversed
    across the two blocks, so each block totals 6 table / 6 form preferred.
    """
    if block_index % 2 == 0:
        mixed = (("table", "easy"), ("table", "medium"), ("form", "hard"))
        ambiguous = (("form", "easy"), ("form", "medium"), ("table", "hard"))
    else:
        mixed = (("form", "easy"), ("form", "medium"), ("table", "hard"))
        ambiguous = (("table", "easy"), ("table", "medium"), ("form", "hard"))
    return {
        "pure_table": (
            ("table", "easy"), ("table", "medium"), ("table", "hard"),
        ),
        "pure_form": (
            ("form", "easy"), ("form", "medium"), ("form", "hard"),
        ),
        "mixed": mixed,
        "ambiguous": ambiguous,
    }


def _build_stage_b_block(
    seed: int, block_seed: int, block_index: int,
) -> list[dict[str, Any]]:
    layout = _stage_b_block_layout(block_index)
    coordinates: list[dict[str, Any]] = []
    ordinal = 0
    for stratum in STRATA:
        for capability, difficulty in layout[stratum]:
            fixture_seed = _seed_int("stage-b", block_seed, ordinal)
            nonce = _make_nonce(
                fixture_seed,
                stratum,
                difficulty,
                capability,
                generator_version=STAGE_B_GENERATOR_VERSION,
            )
            content = _compute_content(
                fixture_seed,
                stratum,
                difficulty,
                capability,
                generator_version=STAGE_B_GENERATOR_VERSION,
            )
            first_bottleneck = _SPECIALIST_BY_CAPABILITY[capability]
            coordinates.append(
                {
                    "fixture_id": _make_fixture_id(
                        fixture_seed,
                        stratum,
                        difficulty,
                        capability,
                        generator_version=STAGE_B_GENERATOR_VERSION,
                    ),
                    "stratum": stratum,
                    "preferred_capability": capability,
                    "first_bottleneck": first_bottleneck,
                    "operation_flags": _operation_flags(stratum, capability),
                    "difficulty": difficulty,
                    "seed": fixture_seed,
                    "template_id": _STAGE_B_TEMPLATE_IDS[stratum],
                    "nonce": nonce,
                    "dependency_order": content.dependency_order,
                    "oracle": _oracle_dict(content),
                    "generator_version": STAGE_B_GENERATOR_VERSION,
                    "block": block_index,
                    "state": _state_metadata(content),
                }
            )
            ordinal += 1
    return coordinates


def build_stage_b_design(
    seed: int = DEFAULT_STAGE_B_SEED,
    block_seeds: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Return the fixed 24-coordinate private Stage-B design in two blocks.

    Exactly 12 coordinates per block, three per stratum (one easy / one medium
    / one hard per block-stratum), balanced 6 table / 6 form preferred
    capability per block, with mixed and ambiguous 2/1 capability splits
    reversed across blocks.  Every coordinate carries a ``generator_version``,
    ``block`` index, and private ``state`` transition metadata in addition to
    the Stage-A coordinate fields.  Fixture ids, seeds, and nonces are derived
    from a distinct Stage-B namespace and can never overlap Stage A.
    """
    if block_seeds is None:
        block_seeds = tuple(
            _seed_int("stage-b", seed, "block", index) for index in range(BLOCK_COUNT)
        )
    if (
        not isinstance(block_seeds, Sequence)
        or isinstance(block_seeds, (str, bytes))
        or len(block_seeds) != BLOCK_COUNT
    ):
        raise ValueError(
            f"block_seeds must be a sequence of {BLOCK_COUNT} integers"
        )
    for value in block_seeds:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("each block seed must be an integer")

    coordinates: list[dict[str, Any]] = []
    for block_index, block_seed in enumerate(block_seeds):
        coordinates.extend(_build_stage_b_block(seed, block_seed, block_index))
    return coordinates


def _oracle_dict(content: _Content) -> dict[str, Any]:
    oracle: dict[str, Any] = {
        "expected_answer": content.nonce,
        "nonce": content.nonce,
        "verification_type": "exact_match",
        "correct_form_values": dict(content.correct_values),
        "dependency_order": content.dependency_order,
    }
    if content.reference is not None:
        oracle["reference"] = content.reference
        oracle["reference_column"] = "Reference Number"
        oracle["unlock_query_param"] = RECORD_REFERENCE_PARAM
    if content.target_row_label is not None:
        oracle["target_row_label"] = content.target_row_label
        oracle["target_column"] = "Access Code"
        oracle["target_row_department"] = content.target_row_department
    return oracle


def _state_metadata(content: _Content) -> dict[str, Any]:
    """Private transition metadata for server/gym integration.

    Describes the reachable render states and what each transition reveals,
    without exposing any secret value.  Consumed by a future Stage-B server,
    never by a model.
    """
    state: dict[str, Any] = {
        "initial": {
            "table_locked": content.reference is not None,
            "form_fields": [field.name for field in content.form_fields],
        },
        "transitions": [],
    }
    if content.form_relevant:
        state["transitions"].append(
            {
                "trigger": "form_submit",
                "query": "form field names",
                "reveals": (
                    "reference"
                    if content.dependency_order == "form_first"
                    else "nonce"
                ),
            }
        )
    if content.reference is not None:
        state["transitions"].append(
            {
                "trigger": "record_reference",
                "query_param": RECORD_REFERENCE_PARAM,
                "reveals": "nonce",
                "requires_reference_match": True,
            }
        )
    if content.table_relevant:
        state["transitions"].append(
            {
                "trigger": "department_filter",
                "query_param": "filter",
                "reveals_nonce": (
                    content.reference is None
                    and content.dependency_order != "form_first"
                ),
            }
        )
    return state


def generate_routing_fixture(
    coord: Mapping[str, Any], text_variant: int = 0,
) -> dict[str, Any]:
    """Return the public fixture record for a private design coordinate.

    The returned mapping contains only the opaque ``fixture_id``, ``title``,
    declared ``prompt``, the initial ``html`` (which embeds both a table cue and
    a form cue), and the audit-only ``source_sha256``.  It never includes the
    stratum, difficulty, seed, template id, first bottleneck, nonce, or oracle.
    """
    if text_variant not in TEXT_VARIANTS:
        raise ValueError(
            f"text_variant must be one of {list(TEXT_VARIANTS)}, got {text_variant!r}"
        )
    stratum = _require(coord, "stratum")
    if stratum not in STRATA:
        raise ValueError(f"unknown stratum {stratum!r}")
    difficulty = _require(coord, "difficulty")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"unknown difficulty {difficulty!r}")
    capability = coord.get("preferred_capability", coord.get("first_bottleneck"))
    if capability in _SPECIALIST_BY_CAPABILITY:
        capability = _SPECIALIST_BY_CAPABILITY[capability]
    capability = capability.rsplit("_specialist", 1)[0]
    if capability not in PREFERRED_CAPABILITIES:
        raise ValueError(f"unknown preferred capability {capability!r}")

    seed = _require(coord, "seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("coord.seed must be an integer")

    generator_version = coord.get("generator_version", GENERATOR_VERSION)
    if generator_version not in (GENERATOR_VERSION, STAGE_B_GENERATOR_VERSION):
        raise ValueError(f"unknown generator_version {generator_version!r}")

    content = _compute_content(
        seed, stratum, difficulty, capability, generator_version=generator_version
    )
    html_str, title, prompt = _build_html(content, text_variant)
    return {
        "fixture_id": str(coord["fixture_id"]),
        "title": title,
        "prompt": prompt,
        "html": html_str,
        "source_sha256": hashlib.sha256(html_str.encode("utf-8")).hexdigest(),
    }


def _require(coord: Mapping[str, Any], key: str) -> Any:
    value = coord.get(key)
    if value is None:
        raise ValueError(f"coord missing required field {key!r}")
    return value


def model_visible(coord: Mapping[str, Any]) -> dict[str, Any]:
    """Explicit model-visible extraction.

    Returns only ``fixture_id`` and the declared ``operation_flags``; it
    deliberately excludes stratum, difficulty, seed, template id, first
    bottleneck, nonce, and oracle.
    """
    return {
        "fixture_id": str(coord["fixture_id"]),
        "operation_flags": dict(coord["operation_flags"]),
    }


# ---------------------------------------------------------------------------
# text-only mutation (invariance testing)
# ---------------------------------------------------------------------------


class _TextMutator(HTMLParser):
    """Rewrite visible text only, preserving tags, attributes, and nesting."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self._out: list[str] = []

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        self._out.append(self.get_starttag_text())

    def handle_startendtag(self, tag: str, attrs: Any) -> None:
        self._out.append(self.get_starttag_text())

    def handle_endtag(self, tag: str) -> None:
        self._out.append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        if data and data.strip():
            self._out.append(re.sub(r"\S", "X", data))
        else:
            self._out.append(data)

    def handle_entityref(self, name: str) -> None:
        self._out.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self._out.append(f"&#{name};")

    def handle_comment(self, data: str) -> None:
        self._out.append(f"<!--{data}-->")

    def handle_decl(self, decl: str) -> None:
        self._out.append(f"<!{decl}>")

    def handle_pi(self, data: str) -> None:
        self._out.append(f"<?{data}>")

    def unknown_decl(self, data: str) -> None:
        self._out.append(f"<![{data}]>")

    def text(self) -> str:
        return "".join(self._out)


def mutate_text(html_str: str) -> str:
    """Return a text-only rewrite of ``html_str`` with identical DOM structure."""
    parser = _TextMutator()
    parser.feed(html_str)
    parser.close()
    return parser.text()


class _StructureParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tokens: list[tuple[str, tuple[str, ...]]] = []

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        names = tuple(sorted(name for name, _ in attrs))
        self.tokens.append((tag, names))

    def handle_startendtag(self, tag: str, attrs: Any) -> None:
        names = tuple(sorted(name for name, _ in attrs))
        self.tokens.append((tag, names))

    def handle_endtag(self, tag: str) -> None:
        self.tokens.append((f"/{tag}", ()))


def structure_tokens(html_str: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return the tag/attribute-name topology of ``html_str`` (text ignored)."""
    parser = _StructureParser()
    parser.feed(html_str)
    parser.close()
    return tuple(parser.tokens)


def structural_signature(html_str: str) -> str:
    """Deterministic digest of the structural topology (text invariant)."""
    tokens = structure_tokens(html_str)
    return _digest("structure", *(f"{tag}{''.join(names)}" for tag, names in tokens))


# ---------------------------------------------------------------------------
# subsequent state handling (form submission / table query)
# ---------------------------------------------------------------------------


def render_state(
    coord: Mapping[str, Any],
    query_params: Mapping[str, str] | None = None,
) -> str:
    """Return subsequent HTML for the same task.

    The transition contract is:

    * form submission (any rendered field name present in ``query_params``)
      validates and returns the success or error page.  For form-first mixed
      tasks the success page reveals the record *reference*; for every other
      form-bearing task it reveals the *nonce*.
    * ``record_reference`` lookup (form-first mixed only) unlocks the
      directory table: a matching reference reveals the target row's access
      code (the final nonce); a wrong or missing reference leaves it locked.
    * ``filter`` returns the department-filtered directory (the nonce is only
      reachable there for tasks whose table is not reference-locked).

    Otherwise the initial HTML is returned.
    """
    qp = dict(query_params or {})
    seed = coord["seed"]
    stratum = coord["stratum"]
    difficulty = coord["difficulty"]
    capability = coord.get("preferred_capability")
    if capability in _SPECIALIST_BY_CAPABILITY:
        capability = _SPECIALIST_BY_CAPABILITY[capability]
    capability = capability.rsplit("_specialist", 1)[0]
    generator_version = coord.get("generator_version", GENERATOR_VERSION)
    content = _compute_content(
        seed, stratum, difficulty, capability, generator_version=generator_version
    )

    if content.reference is not None and RECORD_REFERENCE_PARAM in qp:
        return _reference_table_html(content, qp[RECORD_REFERENCE_PARAM])

    field_names = {field.name for field in content.form_fields}
    if content.form_relevant and field_names & set(qp):
        return _form_result_html(content, qp)

    if content.table_relevant and "filter" in qp:
        return _filtered_table_html(content, qp["filter"])

    return generate_routing_fixture(coord)["html"]


def _form_result_html(content: _Content, qp: Mapping[str, str]) -> str:
    errors: list[str] = []
    for field in content.form_fields:
        value = qp.get(field.name, "").strip()
        if field.name == "access_code":
            if value != content.nonce:
                errors.append(f"{field.label} does not match the directory record.")
            continue
        if field.required and not value:
            errors.append(f"{field.label} is required.")
            continue
        if field.options and value not in field.options:
            errors.append(f"{field.label} must be one of the listed options.")
            continue
        if field.pattern and not re.match(field.pattern, value):
            errors.append(f"{field.label} must match: {field.hint}")

    text = _VARIANT_TEXT[0]
    title = f"{text['org']} {text['portal']}"
    if errors:
        body = (
            f"<h1>Form Validation Error</h1>\n"
            f"<ul>\n" + "".join(f"<li>{html.escape(e)}</li>\n" for e in errors) + "</ul>\n"
            f"<p>Please correct the fields and submit again.</p>\n"
        )
    elif content.dependency_order == "form_first":
        # Form-first: the form returns a record reference, not the nonce.
        body = (
            f"<h1>Submission Successful</h1>\n"
            f"<p>Your request has been recorded.</p>\n"
            f'<p>Record reference number: <code>{html.escape(content.reference or "")}</code></p>\n'
            f"<p>Use this reference number to retrieve your confirmation key "
            f"from the directory.</p>\n"
            f'<form method="get" action="">\n'
            f'<label for="{RECORD_REFERENCE_PARAM}">Record Reference</label>\n'
            f'<input type="text" id="{RECORD_REFERENCE_PARAM}" '
            f'name="{RECORD_REFERENCE_PARAM}" required>\n'
            f'<button type="submit">Retrieve Directory Record</button>\n'
            f'</form>\n'
        )
    else:
        body = (
            f"<h1>Submission Successful</h1>\n"
            f"<p>Your request has been recorded.</p>\n"
            f'<p>Verification key: <code>{html.escape(content.nonce)}</code></p>\n'
        )
    return _HTML_TEMPLATE.format(title=html.escape(title), body=body)


def _reference_table_html(content: _Content, query: str) -> str:
    """Unlock the form-first directory table when ``query`` matches the record
    reference, revealing the final nonce in the target row's access code."""
    match = (query or "").strip()
    correct = content.reference is not None and match == content.reference
    rows: list[tuple[str, ...]] = []
    for index, row in enumerate(content.table_rows):
        cells = list(row)
        if correct and index == content.target_row_index:
            cells[content.target_column or 3] = content.nonce
        rows.append(tuple(cells))
    unlocked = _Content(
        nonce=content.nonce,
        stratum=content.stratum,
        difficulty=content.difficulty,
        capability=content.capability,
        dependency_order=content.dependency_order,
        table_headers=content.table_headers,
        table_rows=tuple(rows),
        code_columns=content.code_columns,
        target_row_index=content.target_row_index,
        target_column=content.target_column,
        target_row_label=content.target_row_label,
        target_row_department=content.target_row_department,
        form_fields=content.form_fields,
        form_method=content.form_method,
        correct_values=content.correct_values,
        table_relevant=content.table_relevant,
        form_relevant=content.form_relevant,
        reference=content.reference,
        reference_column=content.reference_column,
    )
    text = _VARIANT_TEXT[0]
    title = f"{text['org']} {text['portal']}"
    note = ""
    if not correct:
        note = (
            "<p class=\"note\">No directory record matches the supplied "
            "reference. The access code remains locked.</p>\n"
        )
    body = (
        f'<h1>{html.escape(text["org"])} {html.escape(text["portal"])}</h1>\n'
        f'<h2>{html.escape(text["table_heading"])}</h2>\n'
        f"{_table_html(unlocked)}\n"
        f"{note}"
    )
    return _HTML_TEMPLATE.format(title=html.escape(title), body=body)


def _filtered_table_html(content: _Content, query: str) -> str:
    match = query.strip().casefold()
    rows = tuple(
        row for row in content.table_rows if row[1].casefold() == match
    ) if match else content.table_rows
    filtered = _Content(
        nonce=content.nonce,
        stratum=content.stratum,
        difficulty=content.difficulty,
        capability=content.capability,
        dependency_order=content.dependency_order,
        table_headers=content.table_headers,
        table_rows=rows,
        code_columns=content.code_columns,
        target_row_index=content.target_row_index,
        target_column=content.target_column,
        target_row_label=content.target_row_label,
        target_row_department=content.target_row_department,
        form_fields=content.form_fields,
        form_method=content.form_method,
        correct_values=content.correct_values,
        table_relevant=content.table_relevant,
        form_relevant=content.form_relevant,
        reference=content.reference,
        reference_column=content.reference_column,
    )
    text = _VARIANT_TEXT[0]
    title = f"{text['org']} {text['portal']}"
    body = (
        f'<h1>{html.escape(text["org"])} {html.escape(text["portal"])}</h1>\n'
        f'<h2>{html.escape(text["table_heading"])}</h2>\n'
        f"{_table_html(filtered)}\n"
    )
    return _HTML_TEMPLATE.format(title=html.escape(title), body=body)


__all__ = [
    "GENERATOR_VERSION",
    "STAGE_B_GENERATOR_VERSION",
    "STAGE_B_SCHEMA",
    "STRATA",
    "DIFFICULTIES",
    "PREFERRED_CAPABILITIES",
    "TEXT_VARIANTS",
    "DEFAULT_STAGE_A_SEED",
    "DEFAULT_STAGE_B_SEED",
    "BLOCK_COUNT",
    "TASKS_PER_BLOCK",
    "TASKS_PER_STRATUM_PER_BLOCK",
    "RECORD_REFERENCE_PARAM",
    "build_stage_a_design",
    "build_stage_b_design",
    "generate_routing_fixture",
    "model_visible",
    "render_state",
    "mutate_text",
    "structure_tokens",
    "structural_signature",
    "sealed_label",
]
