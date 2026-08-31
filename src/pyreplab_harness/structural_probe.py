"""Controller-owned public-HTML structural probe.

Implements the neutral probe contract from notes/m3-utility-routing-smoke-plan.md
Section 3.

The controller obtains the initial public HTML before starting Pi or assigning a
treatment. This module parses those bytes with a deterministic stdlib parser
(``html.parser.HTMLParser``) and exposes only a fixed allowlist of bounded
non-negative integers:

``element_count``, ``max_dom_depth``, ``table_count``, ``table_row_count``,
``table_cell_count``, ``max_table_columns``, ``form_count``, ``control_count``,
``required_control_count``, ``get_form_count``, ``post_form_count``,
``text_input_count``, ``select_count``, ``textarea_count``, ``button_count``,
``anchor_count``.

The model-visible object never contains page text, attribute values, URLs,
selectors, or any private task information. An audit-only receipt records the
probe schema/mechanism, source byte count, source HTML SHA-256, canonical
feature SHA-256, and delivery status.

Design rules enforced here:

* All feature values are capped to a frozen per-feature bound before they are
  ever exposed.
* Oversized input is rejected (never silently truncated).
* Input is accepted as ``str`` or ``bytes``; the two forms are equivalent for
  UTF-8-decodable text and produce an identical source hash.
* Canonicalization is deterministic: features are serialized as sorted-key,
  compact, ASCII-safe JSON before hashing.
"""

from __future__ import annotations

import hashlib
import json
from html.parser import HTMLParser
from typing import Any, Mapping

__all__ = [
    "StructuralProbeError",
    "FEATURE_KEYS",
    "FEATURE_CAPS",
    "FORBIDDEN_KEY_NAMES",
    "MAX_SOURCE_BYTES",
    "SCHEMA_VERSION",
    "MECHANISM",
    "structural_probe",
    "parse_features",
    "cap_features",
    "canonical_feature_bytes",
    "canonical_feature_sha256",
    "audit_features",
    "audit_receipt",
    "audit_result",
    "is_forbidden_key",
]

# ---------------------------------------------------------------------------
# Frozen schema constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "pyreplab-public-html-structural-probe-v1"
MECHANISM = "controller_owned_public_html_structural_probe"

# Frozen, ordered feature allowlist. This exact ordering is also the canonical
# serialization order.
FEATURE_KEYS: tuple[str, ...] = (
    "element_count",
    "max_dom_depth",
    "table_count",
    "table_row_count",
    "table_cell_count",
    "max_table_columns",
    "form_count",
    "control_count",
    "required_control_count",
    "get_form_count",
    "post_form_count",
    "text_input_count",
    "select_count",
    "textarea_count",
    "button_count",
    "anchor_count",
)

# Frozen per-feature caps. Every count is clamped to its bound before exposure.
FEATURE_CAPS: Mapping[str, int] = {
    "element_count": 2_000,
    "max_dom_depth": 64,
    "table_count": 50,
    "table_row_count": 500,
    "table_cell_count": 2_000,
    "max_table_columns": 64,
    "form_count": 50,
    "control_count": 500,
    "required_control_count": 500,
    "get_form_count": 50,
    "post_form_count": 50,
    "text_input_count": 500,
    "select_count": 500,
    "textarea_count": 500,
    "button_count": 500,
    "anchor_count": 500,
}

# Frozen maximum accepted source size in bytes. Larger inputs are rejected.
MAX_SOURCE_BYTES: int = 4 * 1024 * 1024

# Known content-leaking key names. Used by the privacy audit to give a more
# specific "forbidden key" message than the generic "extra key" message.
FORBIDDEN_KEY_NAMES = frozenset(
    {
        "url",
        "href",
        "src",
        "text",
        "text_content",
        "label",
        "name",
        "value",
        "id",
        "class",
        "selector",
        "domain",
        "action",
        "seed",
        "template",
        "template_id",
        "difficulty",
        "route",
        "route_label",
        "oracle",
        "answer",
        "expected_answer",
        "verifier",
        "verifier_data",
        "treatment",
        "treatment_id",
        "header",
        "headers",
        "title",
        "source_sha256",
        "canonical_feature_sha256",
        "page",
        "content",
        "html",
        "attributes",
        "attrs",
        "links",
    }
)

# HTML void elements: a bare start tag for these never opens a nesting level.
_VOID_ELEMENTS = frozenset(
    {
        "area",
        "base",
        "basefont",
        "bgsound",
        "br",
        "col",
        "command",
        "embed",
        "frame",
        "hr",
        "img",
        "input",
        "keygen",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)

# ``<input type>`` values treated as free-form text entry.
_TEXT_INPUT_TYPES = frozenset(
    {
        "text",
        "search",
        "tel",
        "url",
        "email",
        "password",
        "number",
        "date",
        "time",
        "datetime-local",
        "month",
        "week",
        "color",
        "range",
    }
)

# ``<input type>`` values treated as buttons (contribute to ``button_count``).
_BUTTON_INPUT_TYPES = frozenset({"submit", "reset", "button", "image"})


class StructuralProbeError(ValueError):
    """Raised when the probe cannot be produced for a deterministic reason.

    ``code`` is one of ``input_oversized``, ``undecodable_utf8``,
    ``invalid_input_type``, ``invalid_limit``, or ``invalid_features``.
    """

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class _StructuralParser(HTMLParser):
    """Deterministic structural parser producing raw (uncapped) counts."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.element_count = 0
        self.max_dom_depth = 0
        self._depth = 0
        self.table_count = 0
        self.table_row_count = 0
        self.table_cell_count = 0
        self.max_table_columns = 0
        self._row_cell_stack: list[int] = []
        self.form_count = 0
        self.control_count = 0
        self.required_control_count = 0
        self.get_form_count = 0
        self.post_form_count = 0
        self.text_input_count = 0
        self.select_count = 0
        self.textarea_count = 0
        self.button_count = 0
        self.anchor_count = 0

    # -- HTMLParser callbacks ------------------------------------------------

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._enter(tag, attrs, self_closing=False)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._enter(tag, attrs, self_closing=True)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in ("tr", "thead", "tbody", "tfoot", "table"):
            self._finalize_row()
        if tag not in _VOID_ELEMENTS and self._depth > 0:
            self._depth -= 1

    # -- internals -----------------------------------------------------------

    def _enter(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
        *,
        self_closing: bool,
    ) -> None:
        tag = tag.lower()
        self.element_count += 1

        if tag == "table":
            self.table_count += 1
        elif tag == "tr":
            self.table_row_count += 1
            self._row_cell_stack.append(0)
        elif tag in ("td", "th"):
            self.table_cell_count += 1
            if self._row_cell_stack:
                self._row_cell_stack[-1] += 1
        elif tag == "form":
            self.form_count += 1
            method = (self._attr(attrs, "method") or "").lower()
            if method in ("", "get"):
                self.get_form_count += 1
            elif method == "post":
                self.post_form_count += 1
        elif tag == "input":
            self.control_count += 1
            if self._has_attr(attrs, "required"):
                self.required_control_count += 1
            input_type = (self._attr(attrs, "type") or "text").lower()
            if input_type in _BUTTON_INPUT_TYPES:
                self.button_count += 1
            elif input_type in _TEXT_INPUT_TYPES:
                self.text_input_count += 1
        elif tag == "select":
            self.control_count += 1
            self.select_count += 1
            if self._has_attr(attrs, "required"):
                self.required_control_count += 1
        elif tag == "textarea":
            self.control_count += 1
            self.textarea_count += 1
            if self._has_attr(attrs, "required"):
                self.required_control_count += 1
        elif tag == "button":
            self.control_count += 1
            self.button_count += 1
        elif tag == "a":
            self.anchor_count += 1

        if self_closing or tag in _VOID_ELEMENTS:
            return
        self._depth += 1
        if self._depth > self.max_dom_depth:
            self.max_dom_depth = self._depth

    def _finalize_row(self) -> None:
        if self._row_cell_stack:
            cells = self._row_cell_stack.pop()
            if cells > self.max_table_columns:
                self.max_table_columns = cells

    @staticmethod
    def _attr(attrs: list[tuple[str, str | None]], name: str) -> str | None:
        for key, value in attrs:
            if key == name:
                return value
        return None

    @staticmethod
    def _has_attr(attrs: list[tuple[str, str | None]], name: str) -> bool:
        return any(key == name for key, _ in attrs)


def _raw_features(html_text: str) -> dict[str, int]:
    parser = _StructuralParser()
    parser.feed(html_text)
    parser.close()
    return {
        "element_count": parser.element_count,
        "max_dom_depth": parser.max_dom_depth,
        "table_count": parser.table_count,
        "table_row_count": parser.table_row_count,
        "table_cell_count": parser.table_cell_count,
        "max_table_columns": parser.max_table_columns,
        "form_count": parser.form_count,
        "control_count": parser.control_count,
        "required_control_count": parser.required_control_count,
        "get_form_count": parser.get_form_count,
        "post_form_count": parser.post_form_count,
        "text_input_count": parser.text_input_count,
        "select_count": parser.select_count,
        "textarea_count": parser.textarea_count,
        "button_count": parser.button_count,
        "anchor_count": parser.anchor_count,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _coerce_source(html: str | bytes) -> bytes:
    if isinstance(html, str):
        return html.encode("utf-8")
    if isinstance(html, (bytes, bytearray)):
        return bytes(html)
    raise StructuralProbeError(
        "html must be str or bytes, got %s" % type(html).__name__,
        "invalid_input_type",
    )


def cap_features(
    raw: Mapping[str, int],
    caps: Mapping[str, int] | None = None,
) -> dict[str, int]:
    """Clamp every allowlisted feature to its frozen bound.

    Emits exactly ``FEATURE_KEYS`` in canonical order. Unknown raw keys are
    dropped, and missing raw keys are treated as zero. A *caps* override only
    replaces the listed bounds; all other keys keep their frozen bound. This is
    the single choke point through which counts enter the model-visible object.
    """
    if not isinstance(raw, Mapping):
        raise StructuralProbeError(
            "raw features must be a mapping", "invalid_features"
        )
    bounds = dict(FEATURE_CAPS)
    if caps is not None:
        bounds.update(caps)
    return {
        key: max(0, min(int(raw.get(key, 0)), int(bounds[key])))
        for key in FEATURE_KEYS
    }


def parse_features(html: str | bytes) -> dict[str, int]:
    """Parse HTML and return capped, allowlisted features.

    Convenience wrapper over :func:`structural_probe` that drops the receipt.
    """
    return structural_probe(html)["features"]


def canonical_feature_bytes(features: Mapping[str, int]) -> bytes:
    """Deterministic canonical serialization of a feature mapping.

    Sorted keys, compact separators, and ASCII-safe output make the encoding
    independent of insertion order and of non-ASCII page content.
    """
    payload = json.dumps(
        dict(features),
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return payload.encode("utf-8")


def canonical_feature_sha256(features: Mapping[str, int]) -> str:
    """SHA-256 hex digest of the canonical feature encoding."""
    return hashlib.sha256(canonical_feature_bytes(features)).hexdigest()


def structural_probe(
    html: str | bytes,
    *,
    max_source_bytes: int = MAX_SOURCE_BYTES,
    feature_caps: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Probe public HTML and return ``{"features": ..., "receipt": ...}``.

    ``features`` holds only the frozen bounded-integer allowlist. ``receipt``
    holds audit-only provenance: schema, mechanism, source byte count, source
    SHA-256, canonical feature SHA-256, and delivery status. No page text,
    attribute values, URLs, or selectors ever reach the caller.
    """
    source = _coerce_source(html)
    if max_source_bytes <= 0:
        raise StructuralProbeError(
            "max_source_bytes must be positive", "invalid_limit"
        )
    if len(source) > max_source_bytes:
        raise StructuralProbeError(
            "source is %d bytes, exceeding the %d byte limit"
            % (len(source), max_source_bytes),
            "input_oversized",
        )
    try:
        html_text = source.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise StructuralProbeError(
            "source is not valid UTF-8: %s" % exc,
            "undecodable_utf8",
        ) from exc

    features = cap_features(_raw_features(html_text), caps=feature_caps)
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "mechanism": MECHANISM,
        "source_bytes": len(source),
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "canonical_feature_sha256": canonical_feature_sha256(features),
        "delivered": True,
    }
    return {"features": features, "receipt": receipt}


# ---------------------------------------------------------------------------
# Privacy / provenance audit
# ---------------------------------------------------------------------------


def is_forbidden_key(key: str) -> bool:
    """True if *key* is a known content-leaking key name."""
    return key in FORBIDDEN_KEY_NAMES


def audit_features(features: Any) -> list[str]:
    """Return a list of privacy violations; empty means clean.

    A clean feature mapping has exactly the frozen allowlist keys and every
    value is a non-negative integer within its frozen cap. Extra keys (which
    include any content-leaking key) and out-of-bound values are violations.
    """
    violations: list[str] = []
    if not isinstance(features, Mapping):
        return ["features is not a mapping"]

    present = set(features)
    allowed = set(FEATURE_KEYS)
    for key in sorted(present - allowed):
        if is_forbidden_key(key):
            violations.append("forbidden key leaked: %r" % key)
        else:
            violations.append("extra key not in allowlist: %r" % key)
    for key in sorted(allowed - present):
        violations.append("missing feature key: %r" % key)

    for key in FEATURE_KEYS:
        if key not in features:
            continue
        value = features[key]
        if isinstance(value, bool) or not isinstance(value, int):
            violations.append(
                "%s: value must be a bounded integer, got %s"
                % (key, type(value).__name__)
            )
            continue
        if value < 0:
            violations.append("%s: negative value %d" % (key, value))
        if key in FEATURE_CAPS and value > FEATURE_CAPS[key]:
            violations.append(
                "%s: value %d exceeds cap %d"
                % (key, value, FEATURE_CAPS[key])
            )
    return violations


def _is_sha256_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def audit_receipt(receipt: Any, features: Mapping[str, int]) -> list[str]:
    """Return a list of receipt-integrity violations; empty means valid."""
    violations: list[str] = []
    if not isinstance(receipt, Mapping):
        return ["receipt is not a mapping"]

    required = (
        "schema_version",
        "mechanism",
        "source_bytes",
        "source_sha256",
        "canonical_feature_sha256",
        "delivered",
    )
    for field in required:
        if field not in receipt:
            violations.append("missing receipt field: %r" % field)

    if receipt.get("schema_version") != SCHEMA_VERSION:
        violations.append("unexpected schema_version: %r" % receipt.get("schema_version"))
    if receipt.get("mechanism") != MECHANISM:
        violations.append("unexpected mechanism: %r" % receipt.get("mechanism"))
    if receipt.get("delivered") is not True:
        violations.append("delivered is not True")

    source_bytes = receipt.get("source_bytes")
    if isinstance(source_bytes, bool) or not isinstance(source_bytes, int) or source_bytes < 0:
        violations.append("source_bytes must be a non-negative integer")
    if not _is_sha256_hex(receipt.get("source_sha256")):
        violations.append("source_sha256 is not a valid sha256 hex digest")

    canonical = receipt.get("canonical_feature_sha256")
    if canonical != canonical_feature_sha256(features):
        violations.append(
            "canonical_feature_sha256 does not match canonical feature encoding"
        )
    return violations


def audit_result(result: Any) -> list[str]:
    """Audit a full ``structural_probe`` result (features + receipt)."""
    violations: list[str] = []
    if not isinstance(result, Mapping):
        return ["result is not a mapping"]
    if set(result) != {"features", "receipt"}:
        for key in sorted(set(result) - {"features", "receipt"}):
            violations.append("unexpected result key: %r" % key)
    features = result.get("features")
    violations.extend(audit_features(features))
    violations.extend(audit_receipt(result.get("receipt"), features))
    return violations
