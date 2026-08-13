"""Pure deterministic semantic browser capability core.

Parses public HTML bytes/text using stdlib ``html.parser.HTMLParser``.
No template id, seed, or private files.

APIs:
  semantic_table_query(html_text, request)
  semantic_form_describe(html_text, form_index=0)
  semantic_form_submission(html_text, current_url, values, form_index=0)
"""

from __future__ import annotations

import hashlib
import html.parser
import json
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode, urljoin, urlparse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1.0"
_MAX_HTML_BYTES = 4 * 1024 * 1024  # 4 MiB
_MAX_RESULT_ROWS = 10_000
_MAX_FORM_CONTROLS = 200


# ---------------------------------------------------------------------------
# Public: Table API helpers (pure functions)
# ---------------------------------------------------------------------------


class _TableExtractor(html.parser.HTMLParser):
    """Extract table headers and rows from HTML into lists of dicts."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[dict[str, Any]] = []
        self._state: str = "root"  # root | table | thead | tbody | tr
        self._current_table: dict[str, Any] | None = None
        self._current_headers: list[str] = []
        self._current_row: list[str] = []
        self._current_cell: str = ""
        self._cell_depth: int = 0  # >0 when inside a th/td (handles nesting)
        self._collecting: bool = False
        self._tag_stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_l = tag.lower()
        if tag_l == "table":
            self._state = "table"
            self._current_table = {"headers": [], "rows": []}
            self._current_headers = []
        elif tag_l == "thead" and self._state in ("table", "thead"):
            self._state = "thead"
        elif tag_l == "tbody" and self._state in ("table", "thead", "tbody"):
            self._state = "tbody"
        elif tag_l in ("tfoot", "caption", "colgroup"):
            pass  # skip these silently
        elif tag_l == "tr" and self._state in ("thead", "tbody", "table"):
            self._current_row = []
            self._collecting = True
        elif tag_l in ("th", "td") and self._collecting:
            self._current_cell = ""
            self._cell_depth += 1
        elif self._cell_depth > 0:
            self._cell_depth += 1
        self._tag_stack.append(tag_l)

    def handle_endtag(self, tag: str) -> None:
        tag_l = tag.lower()
        while self._tag_stack and self._tag_stack[-1] != tag_l:
            self._tag_stack.pop()
        if self._tag_stack:
            self._tag_stack.pop()

        if self._current_table is None:
            return

        # Decrement cell depth for any tag within a cell
        if self._cell_depth > 0:
            self._cell_depth -= 1

        if tag_l in ("th", "td") and self._collecting:
            if self._cell_depth == 0:
                value = self._current_cell.strip()
                self._current_row.append(value)
            return
        elif tag_l == "tr" and self._collecting:
            self._collecting = False
            if self._state == "thead":
                # Treat any tr in thead as header row; keep the widest one
                if len(self._current_row) > len(self._current_headers):
                    self._current_headers = list(self._current_row)
            elif self._state in ("tbody", "table"):
                # First tr without thead becomes headers; subsequent become data
                if not self._current_headers:
                    self._current_headers = list(self._current_row)
                    self._state = "tbody"
                elif len(self._current_row) >= 1:
                    self._current_table["rows"].append(list(self._current_row))
        elif tag_l in ("thead", "tbody"):
            if self._current_headers:
                self._current_table["headers"] = list(self._current_headers)
            self._state = "table"
        elif tag_l == "table":
            if self._current_table is not None:
                self._current_table["headers"] = list(self._current_headers)
                self.tables.append(self._current_table)
                self._current_table = None
                self._current_headers = []
                self._state = "root"

    def handle_data(self, data: str) -> None:
        if self._collecting and self._cell_depth > 0:
            self._current_cell += data


def _extract_tables(html_text: str) -> list[dict[str, Any]]:
    """Parse all tables from HTML text."""
    parser = _TableExtractor()
    parser.feed(html_text)
    return parser.tables


# ---------------------------------------------------------------------------
# Numeric / sorting helpers
# ---------------------------------------------------------------------------


def _try_parse_number(raw: str) -> float | None:
    """Parse currency/comma numbers deterministically. Returns float or None."""
    cleaned = raw.strip()
    if not cleaned:
        return None
    currency = "$€£¥₹¢"
    if cleaned[0] in currency:
        cleaned = cleaned[1:].strip()
    sign = -1.0 if cleaned.startswith("-") else 1.0
    if cleaned[:1] in {"-", "+"}:
        cleaned = cleaned[1:]
    if not cleaned or any(
        not (character.isdigit() or character in {",", "."})
        for character in cleaned
    ):
        return None
    digits = cleaned.replace(",", "")
    # Count dots - if more than one, it's not a simple number
    if digits.count(".") > 1:
        return None
    try:
        return sign * float(digits)
    except ValueError:
        return None


def _sort_key(value: str, numeric: bool) -> tuple[int, float | str]:
    """Deterministic sort key: (priority, value).

    Priority 0 = numeric, 1 = text.  Ensures numbers sort before text.
    """
    if numeric:
        num = _try_parse_number(value)
        if num is not None:
            return (0, num)
    return (1, value.casefold())


# ---------------------------------------------------------------------------
# Deterministic canonical JSON
# ---------------------------------------------------------------------------


def _canonical_json(obj: Any) -> str:
    """Deterministic canonical JSON: sorted keys, no trailing whitespace."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


def _sha256_hex(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _make_receipt(
    source_html: str,
    canonical_request: str,
    result_content: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source_html_sha256": _sha256_hex(source_html),
        "canonical_request_sha256": _sha256_hex(canonical_request),
        "result_content_sha256": _sha256_hex(result_content),
    }


# ---------------------------------------------------------------------------
# Table query
# ---------------------------------------------------------------------------


def semantic_table_query(
    html_text: str | bytes,
    request: dict[str, Any],
) -> dict[str, Any]:
    """Parse HTML tables, apply exact filters/sort/projection, return results.

    ``request`` shape (all optional):
      table_index  : int (default 0; required if >1 table present)
      filters      : [{"column": str, "value": str}, ...]
      sort         : {"column": str, "direction": "asc"|"desc"}
      offset       : int (zero-based, default 0)
      limit        : int (positive, default infinite)
      projection   : [str, ...]  (column names to return)

    Returns:
      source_table_count : int
      selected_headers   : [str, ...]
      rows               : [[str, ...], ...]
      total_matched      : int
      matched            : int (after offset/limit)
      receipt            : {schema_version, source_html_sha256,
                            canonical_request_sha256, result_content_sha256}
    """
    if isinstance(html_text, bytes):
        html_text = html_text.decode("utf-8", errors="replace")
    if len(html_text) > _MAX_HTML_BYTES:
        raise ValueError(f"HTML text exceeds maximum size of {_MAX_HTML_BYTES} bytes")

    tables = _extract_tables(html_text)
    table_count = len(tables)
    if table_count == 0:
        raise ValueError("No tables found in HTML")

    # Determine table index
    table_index = request.get("table_index", 0)
    if not isinstance(table_index, int) or table_index < 0:
        raise ValueError(f"table_index must be a non-negative integer, got {table_index!r}")
    if table_count > 1 and "table_index" not in request:
        raise ValueError(
            f"Multiple tables ({table_count}) found; "
            f"request must include explicit table_index"
        )
    if table_index >= table_count:
        raise ValueError(
            f"table_index {table_index} out of range (found {table_count} tables)"
        )

    table = tables[table_index]
    headers: list[str] = table["headers"]
    rows: list[list[str]] = table["rows"]

    # Build column name -> index mapping
    col_map: dict[str, int] = {}
    for idx, h in enumerate(headers):
        col_map[h] = idx

    # Normalize headers for lookup (casefold)
    cf_to_orig: dict[str, str] = {}
    for h in headers:
        cf = h.casefold()
        if cf not in cf_to_orig:
            cf_to_orig[cf] = h

    def _resolve_column(col_name: str) -> int:
        """Resolve column name to index, case-insensitive."""
        if col_name in col_map:
            return col_map[col_name]
        cf = col_name.casefold()
        if cf in cf_to_orig:
            return col_map[cf_to_orig[cf]]
        raise ValueError(f"Column {col_name!r} not found in headers {headers}")

    # Apply filters (exact match, casefold for text)
    filters: list[dict[str, str]] = request.get("filters", [])
    if not isinstance(filters, list):
        raise ValueError("filters must be a list of {column, value} dicts")

    filtered_rows = list(enumerate(rows))  # (original_index, row)
    for f_spec in filters:
        if not isinstance(f_spec, dict):
            raise ValueError(f"each filter must be a dict, got {f_spec!r}")
        col_name = f_spec.get("column")
        val = f_spec.get("value")
        if not isinstance(col_name, str) or not isinstance(val, str):
            raise ValueError(
                f"filter must have string 'column' and 'value', got {f_spec!r}"
            )
        col_idx = _resolve_column(col_name)
        filtered_rows = [
            (orig_i, row)
            for (orig_i, row) in filtered_rows
            if col_idx < len(row) and row[col_idx].casefold() == val.casefold()
        ]

    total_matched = len(filtered_rows)

    # Apply sort
    sort_spec: dict[str, str] | None = request.get("sort", None)
    if sort_spec is not None:
        if not isinstance(sort_spec, dict):
            raise ValueError(f"sort must be a dict, got {sort_spec!r}")
        sort_col = sort_spec.get("column")
        sort_dir = sort_spec.get("direction", "asc").casefold()
        if not isinstance(sort_col, str):
            raise ValueError("sort.column must be a string")
        if sort_dir not in ("asc", "desc"):
            raise ValueError(f"sort.direction must be 'asc' or 'desc', got {sort_dir!r}")

        col_idx = _resolve_column(sort_col)
        reverse = sort_dir == "desc"

        # Deterministic sort: try numeric first, fall back to casefold text
        # We use a key that pairs (numeric_or_text, original_index) for stability
        def _make_sort_key(item: tuple[int, list[str]]) -> Any:
            _orig_i, row = item
            val = row[col_idx] if col_idx < len(row) else ""
            return _sort_key(val, numeric=True)

        filtered_rows.sort(key=_make_sort_key, reverse=reverse)

    # Apply offset/limit
    offset = request.get("offset", 0)
    if not isinstance(offset, int) or offset < 0:
        raise ValueError(f"offset must be a non-negative integer, got {offset!r}")

    limit = request.get("limit")
    if limit is not None:
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError(f"limit must be a positive integer, got {limit!r}")
        end = offset + limit
    else:
        end = None

    sliced = filtered_rows[offset:end]

    if len(sliced) > _MAX_RESULT_ROWS:
        raise ValueError(
            f"Result set exceeds maximum of {_MAX_RESULT_ROWS} rows"
        )

    # Apply projection
    projection: list[str] | None = request.get("projection", None)
    if projection is not None:
        if not isinstance(projection, list):
            raise ValueError(f"projection must be a list of column names")
        proj_indices: list[int] = [_resolve_column(c) for c in projection]
        output_headers = [headers[i] for i in proj_indices]
        output_rows = [
            [row[i] for i in proj_indices] if all(i < len(row) for i in proj_indices)
            else [row[i] if i < len(row) else "" for i in proj_indices]
            for (_orig_i, row) in sliced
        ]
    else:
        output_headers = list(headers)
        output_rows = [row for (_orig_i, row) in sliced]

    matched = len(output_rows)

    # Build result
    result = {
        "source_table_count": table_count,
        "selected_headers": output_headers,
        "rows": output_rows,
        "total_matched": total_matched,
        "matched": matched,
    }

    # Build receipt
    canonical_request_json = _canonical_json(request)
    result_content_json = _canonical_json(result)
    receipt = _make_receipt(html_text, canonical_request_json, result_content_json)
    result["receipt"] = receipt

    return result


# ---------------------------------------------------------------------------
# Form helpers
# ---------------------------------------------------------------------------


@dataclass
class FormDescription:
    method: str
    action: str
    controls: list[dict[str, Any]] = field(default_factory=list)


class _FormExtractor(html.parser.HTMLParser):
    """Extract form descriptions from HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.forms: list[FormDescription] = []
        self._current_form: FormDescription | None = None
        self._collecting: bool = False
        # Track label text by for-id
        self._labels: dict[str, str] = {}
        self._pending_label: str = ""
        self._pending_label_for: str | None = None
        self._in_label: bool = False
        self._tag_stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_l = tag.lower()
        attrs_d = dict(attrs)

        if tag_l == "form":
            method = attrs_d.get("method", "GET").upper()
            action = attrs_d.get("action", "")
            self._current_form = FormDescription(method=method, action=action)
            self._collecting = True
        elif tag_l == "label" and self._collecting:
            self._in_label = True
            self._pending_label = ""
            self._pending_label_for = attrs_d.get("for")
        elif tag_l == "input" and self._collecting and self._current_form is not None:
            ctrl = self._extract_input(attrs_d)
            if ctrl:
                self._current_form.controls.append(ctrl)
        elif tag_l == "select" and self._collecting and self._current_form is not None:
            ctrl = self._extract_select(attrs_d)
            if ctrl:
                self._current_form.controls.append(ctrl)
                # We'll populate options as we parse option children
        elif tag_l == "option" and self._collecting and self._current_form is not None:
            if self._current_form.controls:
                last_ctrl = self._current_form.controls[-1]
                if last_ctrl.get("type") == "select":
                    val = attrs_d.get("value", "")
                    last_ctrl.setdefault("options", []).append(val)

        self._tag_stack.append(tag_l)

    def handle_endtag(self, tag: str) -> None:
        tag_l = tag.lower()
        while self._tag_stack and self._tag_stack[-1] != tag_l:
            self._tag_stack.pop()
        if self._tag_stack:
            self._tag_stack.pop()

        if tag_l == "label" and self._in_label:
            if self._pending_label_for:
                self._labels[self._pending_label_for] = " ".join(
                    self._pending_label.split()
                )
            self._in_label = False
            self._pending_label_for = None
        elif tag_l == "form":
            if self._current_form is not None:
                self.forms.append(self._current_form)
                self._current_form = None
            self._collecting = False

    def handle_data(self, data: str) -> None:
        if self._in_label:
            self._pending_label += data

    def _extract_input(self, attrs_d: dict[str, str | None]) -> dict[str, Any] | None:
        inp_type = (attrs_d.get("type") or "text").casefold()
        if inp_type in ("submit", "reset", "button", "image", "hidden", "file"):
            return None  # Not a value-holding control for our purposes
        name = attrs_d.get("name")
        if not name:
            return None
        required = "required" in attrs_d or attrs_d.get("required") is not None
        pattern = attrs_d.get("pattern", None)
        placeholder = attrs_d.get("placeholder", None)

        # Look up associated label
        ctrl_id = attrs_d.get("id")
        label = self._labels.get(ctrl_id, "") if ctrl_id else ""

        return {
            "label": label,
            "name": name,
            "type": inp_type,
            "required": required,
            "pattern": pattern,
            "placeholder": placeholder,
            "options": None,
        }

    def _extract_select(self, attrs_d: dict[str, str | None]) -> dict[str, Any] | None:
        name = attrs_d.get("name")
        if not name:
            return None
        required = "required" in attrs_d or attrs_d.get("required") is not None

        ctrl_id = attrs_d.get("id")
        label = self._labels.get(ctrl_id, "") if ctrl_id else ""

        return {
            "label": label,
            "name": name,
            "type": "select",
            "required": required,
            "pattern": None,
            "placeholder": None,
            "options": [],
        }

    def error(self, message: str) -> None:
        pass  # Silently ignore parse errors; HTML is messy


def _extract_forms(html_text: str) -> list[FormDescription]:
    """Parse all forms from HTML text."""
    parser = _FormExtractor()
    parser.feed(html_text)
    return parser.forms


# ---------------------------------------------------------------------------
# Form describe
# ---------------------------------------------------------------------------


def semantic_form_describe(
    html_text: str | bytes,
    form_index: int = 0,
) -> dict[str, Any]:
    """Describe a form's method, action, and controls from HTML.

    Returns:
      method           : str ("GET" or "POST")
      action           : str (URL)
      controls         : [{label, name, type, required, pattern, placeholder, options}, ...]
      form_count       : int (total forms found)
      receipt          : {schema_version, source_html_sha256, ...}
    """
    if isinstance(html_text, bytes):
        html_text = html_text.decode("utf-8", errors="replace")
    if len(html_text) > _MAX_HTML_BYTES:
        raise ValueError(f"HTML text exceeds maximum size of {_MAX_HTML_BYTES} bytes")

    forms = _extract_forms(html_text)
    form_count = len(forms)

    if not isinstance(form_index, int) or form_index < 0:
        raise ValueError(f"form_index must be a non-negative integer, got {form_index!r}")
    if form_index >= form_count:
        raise ValueError(
            f"form_index {form_index} out of range (found {form_count} forms)"
        )

    form = forms[form_index]

    result: dict[str, Any] = {
        "method": form.method,
        "action": form.action,
        "controls": [],
        "form_count": form_count,
    }

    for ctrl in form.controls[: _MAX_FORM_CONTROLS]:
        entry: dict[str, Any] = {
            "label": ctrl.get("label", ""),
            "name": ctrl["name"],
            "type": ctrl["type"],
            "required": ctrl.get("required", False),
        }
        if ctrl.get("pattern"):
            entry["pattern"] = ctrl["pattern"]
        if ctrl.get("placeholder"):
            entry["placeholder"] = ctrl["placeholder"]
        if ctrl.get("options") is not None:
            entry["options"] = list(ctrl["options"])
        result["controls"].append(entry)

    result["receipt"] = _make_receipt(
        html_text,
        _canonical_json({"form_index": form_index}),
        _canonical_json(result),
    )

    return result


# ---------------------------------------------------------------------------
# Form submission
# ---------------------------------------------------------------------------


def semantic_form_submission(
    html_text: str | bytes,
    current_url: str,
    values: dict[str, Any],
    form_index: int = 0,
) -> dict[str, Any]:
    """Build a same-origin GET URL from form description and user-supplied values.

    Validates against public constraints (required, pattern, options).
    Rejects unknown/duplicate/non-string values. Does NOT perform network I/O.

    Returns:
      url              : str (same-origin GET URL with query parameters)
      action           : str
      method           : str
      validated_fields : {field_name: value, ...}
      receipt          : {...}
    """
    if isinstance(html_text, bytes):
        html_text = html_text.decode("utf-8", errors="replace")
    if len(html_text) > _MAX_HTML_BYTES:
        raise ValueError(f"HTML text exceeds maximum size of {_MAX_HTML_BYTES} bytes")

    # Describe the form to get control definitions
    description = semantic_form_describe(html_text, form_index=form_index)
    controls: dict[str, dict[str, Any]] = {}
    duplicate_names: set[str] = set()
    for control in description["controls"]:
        name = str(control["name"])
        if name in controls:
            duplicate_names.add(name)
        controls[name] = control
    if duplicate_names:
        raise ValueError(
            f"Form has duplicate field names: {sorted(duplicate_names)}"
        )
    action_raw = description["action"]
    method = description["method"]

    # Validate values
    if not isinstance(values, dict):
        raise ValueError("values must be a dict mapping field names to string values")

    # Check for unknown fields
    supplied_names = set(values.keys())
    known_names = set(controls.keys())
    unknown = supplied_names - known_names
    if unknown:
        raise ValueError(
            f"Unknown field names: {sorted(unknown)}. "
            f"Known fields: {sorted(known_names)}"
        )

    missing_required = sorted(
        name
        for name, control in controls.items()
        if control["required"] and name not in supplied_names
    )
    if missing_required:
        raise ValueError(f"Required fields are missing: {missing_required}")

    # Check for non-string values
    validated: dict[str, str] = {}
    for name, val in values.items():
        if not isinstance(val, str):
            raise ValueError(
                f"Value for field {name!r} must be a string, got {type(val).__name__}"
            )

        ctrl = controls[name]
        val_stripped = val.strip()

        # Required check
        if ctrl["required"] and not val_stripped:
            raise ValueError(f"Field {name!r} is required but received empty value")

        # Pattern check (only if non-empty; empty non-required is fine)
        if ctrl.get("pattern") and val_stripped:
            if not re.match(ctrl["pattern"], val_stripped):
                raise ValueError(
                    f"Field {name!r} value {val_stripped!r} does not match "
                    f"pattern {ctrl['pattern']!r}"
                )

        # Options check (only if non-empty)
        if ctrl.get("options") and val_stripped:
            if val_stripped not in ctrl["options"]:
                raise ValueError(
                    f"Field {name!r} value {val_stripped!r} is not one of "
                    f"the allowed options: {ctrl['options']!r}"
                )

        validated[name] = val_stripped

    # Build same-origin GET URL
    parsed_origin = urlparse(current_url)
    origin = f"{parsed_origin.scheme}://{parsed_origin.netloc}"

    # Resolve action URL
    if action_raw:
        # Resolve relative to origin
        if action_raw.startswith("http://") or action_raw.startswith("https://"):
            action_url = action_raw
        else:
            action_url = urljoin(origin + "/", action_raw)
    else:
        action_url = current_url

    parsed_action = urlparse(action_url)
    action_origin = f"{parsed_action.scheme}://{parsed_action.netloc}"
    if action_origin.casefold() != origin.casefold():
        raise ValueError(
            f"Form action origin {action_origin} does not match "
            f"current page origin {origin}"
        )

    if method != "GET":
        raise ValueError(f"Only GET form submissions are supported, got {method!r}")

    # Build query parameters from validated values
    query_string = urlencode(sorted(validated.items()))
    # Preserve any existing query params in the action URL
    existing_qs = parsed_action.query
    if existing_qs:
        full_qs = f"{existing_qs}&{query_string}" if query_string else existing_qs
    else:
        full_qs = query_string

    parsed_action = parsed_action._replace(
        query=full_qs,
        fragment="",
    )
    built_url = parsed_action.geturl()

    result: dict[str, Any] = {
        "url": built_url,
        "action": action_raw or current_url,
        "method": method,
        "validated_fields": validated,
    }

    canonical_input = _canonical_json(
        {"form_index": form_index, "current_url": current_url, "values": values}
    )
    result["receipt"] = _make_receipt(
        html_text,
        canonical_input,
        _canonical_json(result),
    )

    return result
