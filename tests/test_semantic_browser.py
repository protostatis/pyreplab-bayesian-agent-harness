"""Tests for semantic_browser -- deterministic semantic capability core.

Tests table query, form describe, form submission, edge cases, and
anti-leak enforcement against generated public fixture HTML.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import sys
import unittest
from unittest.mock import patch

# Production module under test
from pyreplab_harness import semantic_browser as sut

# Fixture templates are used ONLY in test code to generate HTML input.
# The production module does NOT import fixture_templates.
from pyreplab_harness.fixture_templates import DIFFICULTIES, generate_page


# ---------------------------------------------------------------------------
# HTML fixtures: hand-crafted public HTML that exercises edge cases
# ---------------------------------------------------------------------------

MULTI_TABLE_HTML = """\
<!DOCTYPE html>
<html><body>
<table>
  <thead><tr><th>A</th><th>B</th></tr></thead>
  <tbody>
    <tr><td>1</td><td>x</td></tr>
    <tr><td>2</td><td>y</td></tr>
  </tbody>
</table>
<table>
  <thead><tr><th>C</th><th>D</th></tr></thead>
  <tbody>
    <tr><td>10</td><td>alpha</td></tr>
    <tr><td>20</td><td>beta</td></tr>
  </tbody>
</table>
</body></html>"""

EMPTY_HTML = "<html><body><p>No tables or forms here.</p></body></html>"

MALFORMED_TABLE_HTML = """\
<!DOCTYPE html>
<html><body>
<table>
  <thead><tr><th>Product</th><th>Price</th></tr></thead>
  <tbody>
    <tr><td>Widget</td><td>$100</td></tr>
    <tr><td>Gadget</td></tr>  <!-- missing cell -->
    <tr><td>Doohickey</td><td>$50</td><td>extra</td></tr>
  </tbody>
</table>
</body></html>"""

CURRENCY_TABLE_HTML = """\
<!DOCTYPE html>
<html><body>
<table>
  <thead><tr><th>Item</th><th>Cost</th></tr></thead>
  <tbody>
    <tr><td>Basic</td><td>$1,234.56</td></tr>
    <tr><td>Pro</td><td>$9,999.00</td></tr>
    <tr><td>Enterprise</td><td>$100.00</td></tr>
    <tr><td>Free</td><td>$0.00</td></tr>
  </tbody>
</table>
</body></html>"""

SIMPLE_FORM_HTML = """\
<!DOCTYPE html>
<html><body>
<h1>Sign Up</h1>
<form method="POST" action="/submit">
  <label for="username">Username:</label>
  <input type="text" id="username" name="username" required
         placeholder="Enter username" pattern="^[a-z0-9_]{3,20}$">

  <label for="email">Email:</label>
  <input type="text" id="email" name="email" required
         placeholder="user@example.com"
         pattern="^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$">

  <label for="role">Role:</label>
  <select id="role" name="role" required>
    <option value="">--Select--</option>
    <option value="admin">Administrator</option>
    <option value="user">User</option>
    <option value="guest">Guest</option>
  </select>

  <input type="submit" value="Submit">
</form>
</body></html>"""

KEY_TABLE_HTML = """\
<!DOCTYPE html>
<html><body>
<table>
  <thead><tr><th>Name</th><th>Code</th></tr></thead>
  <tbody>
    <tr><td>Alpha</td><td>KEY_abc12345</td></tr>
    <tr><td>Beta</td><td>KEY_def67890</td></tr>
    <tr><td>Gamma</td><td>REG_xyx11111</td></tr>
  </tbody>
</table>
</body></html>"""


# ---------------------------------------------------------------------------
# Table API tests
# ---------------------------------------------------------------------------


class TestTableQueryBasic(unittest.TestCase):
    """Core table query functionality."""

    def test_single_table_no_args(self) -> None:
        result = sut.semantic_table_query(MULTI_TABLE_HTML, {"table_index": 0})
        self.assertEqual(result["source_table_count"], 2)
        self.assertEqual(result["selected_headers"], ["A", "B"])
        self.assertEqual(len(result["rows"]), 2)
        self.assertEqual(result["total_matched"], 2)
        self.assertEqual(result["matched"], 2)

    def test_explicit_table_index_required_for_multiple(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            sut.semantic_table_query(MULTI_TABLE_HTML, {})
        self.assertIn("Multiple tables", str(ctx.exception))
        self.assertIn("table_index", str(ctx.exception))

    def test_explicit_table_index_second_table(self) -> None:
        result = sut.semantic_table_query(MULTI_TABLE_HTML, {"table_index": 1})
        self.assertEqual(result["selected_headers"], ["C", "D"])
        self.assertEqual(result["rows"][0], ["10", "alpha"])

    def test_exact_filter(self) -> None:
        html = """<!DOCTYPE html><table>
<thead><tr><th>Name</th><th>Dept</th></tr></thead>
<tbody>
<tr><td>Alice</td><td>Engineering</td></tr>
<tr><td>Bob</td><td>Marketing</td></tr>
<tr><td>Carol</td><td>engineering</td></tr>
</tbody></table>"""
        result = sut.semantic_table_query(
            html, {"filters": [{"column": "Dept", "value": "Engineering"}]}
        )
        # Casefold -- "Engineering" and "engineering" both match
        self.assertEqual(result["total_matched"], 2)
        self.assertEqual(result["matched"], 2)

    def test_filter_no_match(self) -> None:
        html = """<!DOCTYPE html><table>
<thead><tr><th>Name</th></tr></thead>
<tbody><tr><td>Alice</td></tr></tbody></table>"""
        result = sut.semantic_table_query(
            html, {"filters": [{"column": "Name", "value": "Nobody"}]}
        )
        self.assertEqual(result["total_matched"], 0)
        self.assertEqual(result["matched"], 0)

    def test_filter_missing_column(self) -> None:
        html = """<!DOCTYPE html><table>
<thead><tr><th>Name</th></tr></thead><tbody><tr><td>X</td></tr></tbody></table>"""
        with self.assertRaises(ValueError) as ctx:
            sut.semantic_table_query(
                html, {"filters": [{"column": "Nonexistent", "value": "X"}]}
            )
        self.assertIn("Nonexistent", str(ctx.exception))

    def test_sort_asc_numeric(self) -> None:
        result = sut.semantic_table_query(
            CURRENCY_TABLE_HTML,
            {"sort": {"column": "Cost", "direction": "asc"}},
        )
        rows = result["rows"]
        costs = [r[1] for r in rows]
        self.assertEqual(costs, ["$0.00", "$100.00", "$1,234.56", "$9,999.00"])

    def test_sort_desc_numeric(self) -> None:
        result = sut.semantic_table_query(
            CURRENCY_TABLE_HTML,
            {"sort": {"column": "Cost", "direction": "desc"}},
        )
        rows = result["rows"]
        costs = [r[1] for r in rows]
        self.assertEqual(costs, ["$9,999.00", "$1,234.56", "$100.00", "$0.00"])

    def test_sort_text_fallback(self) -> None:
        html = """<!DOCTYPE html><table>
<thead><tr><th>Name</th></tr></thead>
<tbody>
<tr><td>Zebra</td></tr>
<tr><td>apple</td></tr>
<tr><td>Banana</td></tr>
</tbody></table>"""
        result = sut.semantic_table_query(
            html, {"sort": {"column": "Name", "direction": "asc"}}
        )
        names = [r[0] for r in result["rows"]]
        # casefold sort: apple, Banana, Zebra
        self.assertEqual(names, ["apple", "Banana", "Zebra"])

    def test_alphanumeric_values_are_not_coerced_to_numbers(self) -> None:
        html = """<table><tr><th>Name</th></tr>
<tr><td>X20</td></tr><tr><td>X3</td></tr><tr><td>X100</td></tr></table>"""
        result = sut.semantic_table_query(
            html, {"sort": {"column": "Name", "direction": "asc"}}
        )
        self.assertEqual(result["rows"], [["X100"], ["X20"], ["X3"]])

    def test_sort_desc_text(self) -> None:
        html = """<!DOCTYPE html><table>
<thead><tr><th>Name</th></tr></thead>
<tbody>
<tr><td>Zebra</td></tr>
<tr><td>apple</td></tr>
<tr><td>Banana</td></tr>
</tbody></table>"""
        result = sut.semantic_table_query(
            html, {"sort": {"column": "Name", "direction": "desc"}}
        )
        names = [r[0] for r in result["rows"]]
        self.assertEqual(names, ["Zebra", "Banana", "apple"])

    def test_sort_invalid_direction(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            sut.semantic_table_query(
                CURRENCY_TABLE_HTML,
                {"sort": {"column": "Cost", "direction": "sideways"}},
            )
        self.assertIn("direction", str(ctx.exception))

    def test_offset_and_limit(self) -> None:
        result = sut.semantic_table_query(
            CURRENCY_TABLE_HTML,
            {
                "sort": {"column": "Cost", "direction": "asc"},
                "offset": 1,
                "limit": 2,
            },
        )
        self.assertEqual(result["total_matched"], 4)
        self.assertEqual(result["matched"], 2)
        rows = result["rows"]
        self.assertEqual([r[0] for r in rows], ["Enterprise", "Basic"])

    def test_projection(self) -> None:
        result = sut.semantic_table_query(
            CURRENCY_TABLE_HTML,
            {
                "sort": {"column": "Cost", "direction": "asc"},
                "projection": ["Item"],
                "limit": 2,
            },
        )
        self.assertEqual(result["selected_headers"], ["Item"])
        self.assertEqual([r[0] for r in result["rows"]], ["Free", "Enterprise"])

    def test_no_tables_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            sut.semantic_table_query(EMPTY_HTML, {})
        self.assertIn("No tables", str(ctx.exception))

    def test_oversized_html_raises(self) -> None:
        big = "x" * (sut._MAX_HTML_BYTES + 1)
        with self.assertRaises(ValueError) as ctx:
            sut.semantic_table_query(big, {})
        self.assertIn("exceeds maximum", str(ctx.exception))

    def test_bytes_input_accepted(self) -> None:
        html = "<!DOCTYPE html><table><thead><tr><th>X</th></tr></thead><tbody><tr><td>1</td></tr></tbody></table>"
        result = sut.semantic_table_query(html.encode(), {})
        self.assertEqual(result["rows"], [["1"]])

    def test_malformed_rows(self) -> None:
        """Rows with missing or extra cells should still be handled."""
        result = sut.semantic_table_query(MALFORMED_TABLE_HTML, {})
        rows = result["rows"]
        self.assertGreaterEqual(len(rows), 3)
        # Row with missing cell should have fewer cells
        self.assertEqual(result["selected_headers"], ["Product", "Price"])

    def test_key_table_not_prioritized(self) -> None:
        """KEY_ values should never be specially prioritized in sorting."""
        result = sut.semantic_table_query(
            KEY_TABLE_HTML,
            {"sort": {"column": "Name", "direction": "asc"}},
        )
        names = [r[0] for r in result["rows"]]
        self.assertEqual(names, ["Alpha", "Beta", "Gamma"])

    def test_column_case_insensitive(self) -> None:
        html = """<!DOCTYPE html><table>
<thead><tr><th>Full Name</th></tr></thead>
<tbody><tr><td>Alice</td></tr></tbody></table>"""
        result = sut.semantic_table_query(
            html, {"filters": [{"column": "FULL NAME", "value": "Alice"}]}
        )
        self.assertEqual(result["total_matched"], 1)


class TestTableQueryDeterminism(unittest.TestCase):
    """Determinism and idempotency checks."""

    def test_deterministic_output(self) -> None:
        runs = []
        for _ in range(5):
            r = sut.semantic_table_query(CURRENCY_TABLE_HTML, {
                "sort": {"column": "Cost", "direction": "asc"},
                "projection": ["Item", "Cost"],
            })
            runs.append(sut._canonical_json(r))
        self.assertEqual(len(set(runs)), 1)

    def test_receipt_present_and_consistent(self) -> None:
        result = sut.semantic_table_query(CURRENCY_TABLE_HTML, {
            "sort": {"column": "Cost", "direction": "asc"},
        })
        receipt = result["receipt"]
        self.assertEqual(receipt["schema_version"], "1.0")
        self.assertEqual(len(receipt["source_html_sha256"]), 64)
        self.assertEqual(len(receipt["canonical_request_sha256"]), 64)
        self.assertEqual(len(receipt["result_content_sha256"]), 64)
        # Same input -> same receipt hashes
        result2 = sut.semantic_table_query(CURRENCY_TABLE_HTML, {
            "sort": {"column": "Cost", "direction": "asc"},
        })
        self.assertEqual(receipt, result2["receipt"])


# ---------------------------------------------------------------------------
# Form API tests
# ---------------------------------------------------------------------------


class TestFormDescribe(unittest.TestCase):
    """Form description extraction."""

    def test_describe_simple_form(self) -> None:
        desc = sut.semantic_form_describe(SIMPLE_FORM_HTML)
        self.assertEqual(desc["method"], "POST")
        self.assertEqual(desc["action"], "/submit")
        self.assertEqual(desc["form_count"], 1)
        self.assertEqual(len(desc["controls"]), 3)

        names = {c["name"] for c in desc["controls"]}
        self.assertEqual(names, {"username", "email", "role"})

        username = next(c for c in desc["controls"] if c["name"] == "username")
        self.assertEqual(username["type"], "text")
        self.assertTrue(username["required"])
        self.assertEqual(username["pattern"], "^[a-z0-9_]{3,20}$")
        self.assertEqual(username["placeholder"], "Enter username")

        role = next(c for c in desc["controls"] if c["name"] == "role")
        self.assertEqual(role["type"], "select")
        self.assertEqual(role["options"], ["", "admin", "user", "guest"])

    def test_describe_no_forms_raises(self) -> None:
        with self.assertRaises(ValueError):
            sut.semantic_form_describe(EMPTY_HTML)

    def test_describe_form_index_out_of_range(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            sut.semantic_form_describe(SIMPLE_FORM_HTML, form_index=5)
        self.assertIn("out of range", str(ctx.exception))

    def test_describe_receipt(self) -> None:
        desc = sut.semantic_form_describe(SIMPLE_FORM_HTML)
        receipt = desc["receipt"]
        self.assertEqual(receipt["schema_version"], "1.0")
        self.assertEqual(len(receipt["source_html_sha256"]), 64)

    def test_form_with_get_method(self) -> None:
        html = """<!DOCTYPE html><form method="GET" action="/search">
<input name="q" type="text"></form>"""
        desc = sut.semantic_form_describe(html)
        self.assertEqual(desc["method"], "GET")
        self.assertEqual(desc["action"], "/search")

    def test_form_default_method(self) -> None:
        html = """<!DOCTYPE html><form action="/x">
<input name="a" type="text"></form>"""
        desc = sut.semantic_form_describe(html)
        self.assertEqual(desc["method"], "GET")

    def test_form_empty_action(self) -> None:
        html = """<!DOCTYPE html><form>
<input name="a" type="text"></form>"""
        desc = sut.semantic_form_describe(html)
        self.assertEqual(desc["action"], "")

    def test_form_skips_submit_buttons(self) -> None:
        html = """<!DOCTYPE html><form>
<input name="name" type="text">
<input name="ok" type="submit" value="Go">
<input type="hidden" name="secret" value="xyz">
<input type="file" name="attachment">
</form>"""
        desc = sut.semantic_form_describe(html)
        names = {c["name"] for c in desc["controls"]}
        self.assertEqual(names, {"name"})

    def test_form_label_association(self) -> None:
        html = """<!DOCTYPE html><form>
<label for="n">Handle</label>
<input type="text" id="n" name="username">
</form>"""
        desc = sut.semantic_form_describe(html)
        ctrl = desc["controls"][0]
        self.assertEqual(ctrl["name"], "username")
        self.assertEqual(ctrl["label"], "Handle")


class TestFormSubmission(unittest.TestCase):
    """Form submission URL builder and validation."""

    def test_valid_submission(self) -> None:
        html = SIMPLE_FORM_HTML.replace('method="POST"', 'method="GET"')
        result = sut.semantic_form_submission(
            html,
            "https://example.com/register",
            {
                "username": "test_user1",
                "email": "user@example.com",
                "role": "admin",
            },
        )
        self.assertIn("url", result)
        self.assertEqual(result["method"], "GET")
        self.assertIn("test_user1", result["url"])
        self.assertIn("user%40example.com", result["url"])
        self.assertIn("role=admin", result["url"])

    def test_same_origin_enforcement(self) -> None:
        """Action origin must match current page origin."""
        # Same-origin relative action should succeed (no error raised here)
        sut.semantic_form_submission(
            SIMPLE_FORM_HTML.replace('method="POST"', 'method="GET"'),
            "https://example.com/page",
            {"username": "test", "email": "x@y.com", "role": "admin"},
        )
        # Cross-origin action should be rejected
        cross_html = """<!DOCTYPE html><form method="GET" action="https://evil.com/steal">
<input name="username" type="text"></form>"""
        with self.assertRaises(ValueError) as ctx2:
            sut.semantic_form_submission(
                cross_html,
                "https://example.com/page",
                {"username": "test"},
            )
        self.assertIn("origin", str(ctx2.exception).casefold())

    def test_required_field_missing(self) -> None:
        html = SIMPLE_FORM_HTML.replace('method="POST"', 'method="GET"')
        with self.assertRaises(ValueError) as ctx:
            sut.semantic_form_submission(
                html,
                "https://example.com/",
                {
                    "username": "",
                    "email": "x@y.com",
                    "role": "user",
                },
            )
        self.assertIn("required", str(ctx.exception).casefold())

    def test_required_field_omitted(self) -> None:
        html = SIMPLE_FORM_HTML.replace('method="POST"', 'method="GET"')
        with self.assertRaisesRegex(ValueError, "Required fields are missing"):
            sut.semantic_form_submission(
                html,
                "https://example.com/",
                {"username": "test", "role": "user"},
            )

    def test_post_form_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Only GET"):
            sut.semantic_form_submission(
                SIMPLE_FORM_HTML,
                "https://example.com/",
                {
                    "username": "test",
                    "email": "x@y.com",
                    "role": "user",
                },
            )

    def test_pattern_mismatch(self) -> None:
        html = SIMPLE_FORM_HTML.replace('method="POST"', 'method="GET"')
        with self.assertRaises(ValueError) as ctx:
            sut.semantic_form_submission(
                html,
                "https://example.com/",
                {
                    "username": "invalid username with spaces!",
                    "email": "user@example.com",
                    "role": "user",
                },
            )
        self.assertIn("does not match pattern", str(ctx.exception))

    def test_option_not_in_list(self) -> None:
        html = SIMPLE_FORM_HTML.replace('method="POST"', 'method="GET"')
        with self.assertRaises(ValueError) as ctx:
            sut.semantic_form_submission(
                html,
                "https://example.com/",
                {
                    "username": "test_user",
                    "email": "user@example.com",
                    "role": "superadmin",
                },
            )
        self.assertIn("not one of the allowed options", str(ctx.exception))

    def test_unknown_field_rejected(self) -> None:
        html = SIMPLE_FORM_HTML.replace('method="POST"', 'method="GET"')
        with self.assertRaises(ValueError) as ctx:
            sut.semantic_form_submission(
                html,
                "https://example.com/",
                {
                    "username": "test",
                    "email": "x@y.com",
                    "role": "user",
                    "injected_field": "malicious",
                },
            )
        self.assertIn("Unknown field names", str(ctx.exception))

    def test_non_string_value_rejected(self) -> None:
        html = SIMPLE_FORM_HTML.replace('method="POST"', 'method="GET"')
        with self.assertRaises(ValueError) as ctx:
            sut.semantic_form_submission(
                html,
                "https://example.com/",
                {
                    "username": 12345,  # type: ignore
                    "email": "x@y.com",
                    "role": "user",
                },
            )
        self.assertIn("must be a string", str(ctx.exception))
        self.assertIn("username", str(ctx.exception))

    def test_non_dict_values_rejected(self) -> None:
        html = SIMPLE_FORM_HTML.replace('method="POST"', 'method="GET"')
        with self.assertRaises(ValueError):
            sut.semantic_form_submission(
                html,
                "https://example.com/",
                "not-a-dict",  # type: ignore
            )

    def test_submission_receipt(self) -> None:
        html = SIMPLE_FORM_HTML.replace('method="POST"', 'method="GET"')
        result = sut.semantic_form_submission(
            html,
            "https://example.com/",
            {
                "username": "test_user",
                "email": "user@example.com",
                "role": "admin",
            },
        )
        receipt = result["receipt"]
        self.assertEqual(receipt["schema_version"], "1.0")
        self.assertIn("source_html_sha256", receipt)
        self.assertIn("canonical_request_sha256", receipt)
        self.assertIn("result_content_sha256", receipt)

    def test_submission_deterministic(self) -> None:
        runs = []
        html = SIMPLE_FORM_HTML.replace('method="POST"', 'method="GET"')
        vals = {"username": "test_user", "email": "u@ex.com", "role": "admin"}
        for _ in range(5):
            r = sut.semantic_form_submission(
                html, "https://example.com/", vals
            )
            runs.append(sut._canonical_json(r))
        self.assertEqual(len(set(runs)), 1)

    def test_relative_action_resolved(self) -> None:
        html = """<!DOCTYPE html><form action="/results">
<input name="q" type="text"></form>"""
        result = sut.semantic_form_submission(
            html, "https://example.com/search", {"q": "hello"}
        )
        parsed = __import__("urllib.parse", fromlist=["urlparse"]).urlparse(
            result["url"]
        )
        self.assertEqual(parsed.scheme, "https")
        self.assertEqual(parsed.netloc, "example.com")
        self.assertIn("/results", parsed.path)

    def test_optional_field_empty_allowed(self) -> None:
        html = """<!DOCTYPE html><form action="/x">
<input name="required_f" type="text" required>
<input name="optional_f" type="text">
</form>"""
        result = sut.semantic_form_submission(
            html, "https://example.com/", {"required_f": "ok", "optional_f": ""}
        )
        self.assertIn("required_f=ok", result["url"])


# ---------------------------------------------------------------------------
# Fixture template integration tests
# ---------------------------------------------------------------------------


class TestTableFilterSortFixture(unittest.TestCase):
    """Test semantic_table_query against table_filter_sort fixture pages.

    Uses the fixture template to generate HTML (test-only).
    The production module cannot import fixture_templates.
    The nonce/oracle are used only by test assertions, not by the module.
    """

    def _run_table_filter_sort(
        self, seed: int, difficulty: str
    ) -> tuple[dict[str, object], object]:
        page = generate_page("table_filter_sort", seed, difficulty)
        oracle = page.oracle

        # Parse the task description from the HTML to extract params
        # (simulating what an LLM would infer):
        # "Filter the table to only products in the X category,
        # then sort by Price ascending. Report the Reference Code
        # of the Nth item."
        target_cat = oracle["target_category"]
        target_rank = oracle["target_rank"]

        result = sut.semantic_table_query(
            page.html,
            {
                "filters": [{"column": "Category", "value": target_cat}],
                "sort": {"column": "Price (USD)", "direction": "asc"},
                "offset": target_rank - 1,
                "limit": 1,
                "projection": ["Reference Code"],
            },
        )
        return result, oracle

    def test_easy_seed_0(self) -> None:
        result, oracle = self._run_table_filter_sort(0, "easy")
        # total_matched varies per seed regardless of target_rank
        self.assertGreaterEqual(result["total_matched"], oracle["target_rank"])
        self.assertEqual(result["matched"], 1)
        self.assertEqual(result["rows"][0][0], oracle["expected_answer"])

    def test_easy_seed_7(self) -> None:
        result, oracle = self._run_table_filter_sort(7, "easy")
        self.assertEqual(result["matched"], 1)
        self.assertEqual(result["rows"][0][0], oracle["expected_answer"])

    def test_medium_seed_0(self) -> None:
        result, oracle = self._run_table_filter_sort(0, "medium")
        self.assertEqual(result["matched"], 1)
        self.assertEqual(result["rows"][0][0], oracle["expected_answer"])

    def test_medium_seed_5(self) -> None:
        result, oracle = self._run_table_filter_sort(5, "medium")
        self.assertEqual(result["matched"], 1)
        self.assertEqual(result["rows"][0][0], oracle["expected_answer"])

    def test_hard_seed_0(self) -> None:
        result, oracle = self._run_table_filter_sort(0, "hard")
        self.assertEqual(result["matched"], 1)
        self.assertEqual(result["rows"][0][0], oracle["expected_answer"])

    def test_hard_seed_3(self) -> None:
        result, oracle = self._run_table_filter_sort(3, "hard")
        self.assertEqual(result["matched"], 1)
        self.assertEqual(result["rows"][0][0], oracle["expected_answer"])


class TestSearchFilterControlsFixture(unittest.TestCase):
    """Test semantic_table_query against search_filter_controls fixture pages."""

    def _run_search(
        self, seed: int, difficulty: str
    ) -> tuple[dict[str, object], object]:
        # First get the hub page to extract the search term from oracle
        hub_page = generate_page("search_filter_controls", seed, difficulty)
        search_term = hub_page.oracle["search_term"]
        # Then get the search results page
        page = generate_page(
            "search_filter_controls", seed, difficulty,
            query_params={"q": search_term},
        )
        oracle = page.oracle

        result = sut.semantic_table_query(
            page.html,
            {
                "filters": [{"column": "Reference", "value": oracle["expected_answer"]}],
                "projection": ["Name", "Category", "Reference"],
            },
        )
        return result, oracle

    def test_easy_seed_0(self) -> None:
        result, oracle = self._run_search(0, "easy")
        self.assertGreaterEqual(result["total_matched"], 1)
        self.assertEqual(result["rows"][0][2], oracle["expected_answer"])

    def test_easy_seed_5(self) -> None:
        result, oracle = self._run_search(5, "easy")
        self.assertGreaterEqual(result["total_matched"], 1)
        self.assertEqual(result["rows"][0][2], oracle["expected_answer"])

    def test_medium_seed_0(self) -> None:
        result, oracle = self._run_search(0, "medium")
        self.assertGreaterEqual(result["total_matched"], 1)
        self.assertEqual(result["rows"][0][2], oracle["expected_answer"])


class TestFormEntryValidationFixture(unittest.TestCase):
    """Test semantic_form_submission against form_entry_validation fixture pages."""

    def _run_form(
        self, seed: int, difficulty: str
    ) -> tuple[dict[str, object], object]:
        """Load the fixture form page (no query params = form display)."""
        page = generate_page("form_entry_validation", seed, difficulty)
        oracle = page.oracle
        correct_values = oracle["correct_values"]

        # Build submission from correct values
        result = sut.semantic_form_submission(
            page.html,
            "https://example.com/page",
            correct_values,
        )
        return result, oracle

    def test_easy_form_seed_0(self) -> None:
        result, oracle = self._run_form(0, "easy")
        self.assertIn("url", result)
        self.assertEqual(
            set(result["validated_fields"].keys()),
            set(oracle["correct_values"].keys()),
        )

    def test_medium_form_seed_0(self) -> None:
        result, oracle = self._run_form(0, "medium")
        self.assertIn("url", result)
        self.assertEqual(
            set(result["validated_fields"].keys()),
            set(oracle["correct_values"].keys()),
        )

    def test_hard_form_seed_0(self) -> None:
        result, oracle = self._run_form(0, "hard")
        self.assertIn("url", result)
        self.assertEqual(
            set(result["validated_fields"].keys()),
            set(oracle["correct_values"].keys()),
        )

    def test_invalid_form_value_raises(self) -> None:
        """Verify that validation catches bad values on public form constraints."""
        with self.assertRaises(ValueError):
            sut.semantic_form_submission(
                SIMPLE_FORM_HTML.replace('method="POST"', 'method="GET"'),
                "https://example.com/",
                {"username": "test", "email": "not-an-email", "role": "user"},
            )


# ---------------------------------------------------------------------------
# Anti-leak tests
# ---------------------------------------------------------------------------


class TestAntiLeak(unittest.TestCase):
    """Verify the production module contains no leakage of template/oracle/seeds.

    The production module must not:
    - Import fixture_templates or any private file
    - Contain references to seed, oracle, expected_answer, KEY_ nonce patterns
    - Prioritize KEY_ values in any heuristic
    - Access private files or treatment metadata
    """

    def test_no_fixture_template_import(self) -> None:
        source = inspect.getsource(sut)
        self.assertNotIn("fixture_templates", source)
        self.assertNotIn("fixture", source.casefold())

    def test_no_oracle_or_seed_references(self) -> None:
        source = inspect.getsource(sut)
        source_lower = source.casefold()
        # "oracle" might appear in docstrings/comments; be precise
        self.assertNotIn("oracle", source_lower)
        self.assertNotIn("expected_answer", source_lower)
        self.assertNotIn('"seed"', source)
        self.assertNotIn('seed=', source.casefold())

    def test_no_nonce_prioritization(self) -> None:
        """The module must not contain any KEY_ pattern matching or ranking."""
        source = inspect.getsource(sut)
        self.assertNotIn("KEY_", source)
        self.assertNotIn("nonce", source.casefold())
        self.assertNotIn("verification_key", source.casefold())

    def test_no_private_file_access(self) -> None:
        source = inspect.getsource(sut)
        self.assertNotIn("__file__", source)
        self.assertNotIn("open(", source)

    def test_no_treatment_metadata(self) -> None:
        source = inspect.getsource(sut)
        self.assertNotIn("treatment", source.casefold())
        self.assertNotIn("policy", source.casefold())

    def test_no_regex_target_heuristics(self) -> None:
        """The core logic must not use regex to find target/nonce values in tables."""
        source = inspect.getsource(sut)
        # We allow re.match in form pattern validation only.
        # Check that re is only imported and used for form pattern validation.
        import_lines = [
            l for l in source.split("\n") if l.strip().startswith("import re")
        ]
        # re is used for pattern validation in form submission only
        # There should be no re.search/re.findall/re.sub for content targeting
        self.assertNotIn("re.search", source)
        self.assertNotIn("re.findall", source)
        self.assertNotIn("re.sub", source)

    def test_receipt_has_no_oracle_fields(self) -> None:
        """Receipt dicts must not contain oracle/template/seed/nonce fields."""
        result = sut.semantic_table_query(CURRENCY_TABLE_HTML, {
            "sort": {"column": "Cost", "direction": "asc"},
        })
        receipt = result["receipt"]
        receipt_keys = {
            k.casefold() for k in receipt.keys()
        }
        forbidden = {"oracle", "seed", "template", "nonce", "expected_answer",
                     "target", "key_", "verification_key"}
        self.assertFalse(
            receipt_keys & forbidden,
            f"Receipt contains forbidden keys: {receipt_keys & forbidden}",
        )

    def test_receipt_keys_are_stable(self) -> None:
        """Receipt schema must have exactly the expected keys."""
        result = sut.semantic_table_query(CURRENCY_TABLE_HTML, {
            "sort": {"column": "Cost", "direction": "asc"},
        })
        receipt = result["receipt"]
        expected = {"schema_version", "source_html_sha256",
                    "canonical_request_sha256", "result_content_sha256"}
        self.assertEqual(set(receipt.keys()), expected)

    def test_result_has_no_oracle_fields(self) -> None:
        """Result dicts must not contain oracle/template/seed/nonce fields."""
        result = sut.semantic_table_query(CURRENCY_TABLE_HTML, {
            "sort": {"column": "Cost", "direction": "asc"},
        })
        result_keys = {k.casefold() for k in result.keys()}
        forbidden = {"oracle", "seed", "template", "nonce", "expected_answer",
                     "target", "key_", "verification_key"}
        self.assertFalse(
            result_keys & forbidden,
            f"Result contains forbidden keys: {result_keys & forbidden}",
        )

    def test_stdlib_only_imports(self) -> None:
        """Module must only import from stdlib."""
        source = inspect.getsource(sut)
        third_party = {"numpy", "pandas", "requests", "beautifulsoup",
                       "bs4", "lxml", "selenium", "playwright", "scrapy",
                       "pyreplab_harness.fixture_templates",
                       "pyreplab_harness.treatments"}
        for pkg in third_party:
            self.assertNotIn(f"import {pkg}", source)
            self.assertNotIn(f"from {pkg}", source)

    def test_form_result_has_no_oracle_fields(self) -> None:
        """Form submission result must not contain oracle fields."""
        result = sut.semantic_form_submission(
            SIMPLE_FORM_HTML.replace('method="POST"', 'method="GET"'),
            "https://example.com/",
            {"username": "test", "email": "x@y.com", "role": "admin"},
        )
        result_keys = {k.casefold() for k in result.keys()}
        forbidden = {"oracle", "seed", "template", "nonce", "expected_answer",
                     "target", "key_", "verification_key"}
        self.assertFalse(
            result_keys & forbidden,
            f"Form result contains forbidden keys: {result_keys & forbidden}",
        )


# ---------------------------------------------------------------------------
# Regression tests for edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases(unittest.TestCase):
    """Edge cases and boundaries."""

    def test_empty_table_no_rows(self) -> None:
        html = """<!DOCTYPE html><table>
<thead><tr><th>Col</th></tr></thead>
<tbody></tbody></table>"""
        result = sut.semantic_table_query(html, {})
        self.assertEqual(result["rows"], [])

    def test_table_without_thead(self) -> None:
        html = """<!DOCTYPE html><table>
<tr><td>A</td><td>B</td></tr>
<tr><td>1</td><td>2</td></tr>
</table>"""
        result = sut.semantic_table_query(html, {})
        # First row becomes header
        self.assertEqual(result["selected_headers"], ["A", "B"])
        self.assertEqual(result["rows"], [["1", "2"]])

    def test_offset_beyond_results(self) -> None:
        result = sut.semantic_table_query(
            CURRENCY_TABLE_HTML, {"offset": 100}
        )
        self.assertEqual(result["rows"], [])
        self.assertEqual(result["matched"], 0)

    def test_invalid_filter_spec(self) -> None:
        with self.assertRaises(ValueError):
            sut.semantic_table_query(
                CURRENCY_TABLE_HTML,
                {"filters": "not-a-list"},
            )

    def test_filter_missing_column_key(self) -> None:
        with self.assertRaises(ValueError):
            sut.semantic_table_query(
                CURRENCY_TABLE_HTML,
                {"filters": [{"wrong_key": "X", "value": "Y"}]},
            )

    def test_negative_offset_raises(self) -> None:
        with self.assertRaises(ValueError):
            sut.semantic_table_query(CURRENCY_TABLE_HTML, {"offset": -1})

    def test_zero_limit_raises(self) -> None:
        with self.assertRaises(ValueError):
            sut.semantic_table_query(CURRENCY_TABLE_HTML, {"limit": 0})

    def test_negative_limit_raises(self) -> None:
        with self.assertRaises(ValueError):
            sut.semantic_table_query(CURRENCY_TABLE_HTML, {"limit": -5})

    def test_multiple_tables_auto_index_0_ok(self) -> None:
        """Single table: table_index can be omitted (defaults to 0)."""
        html = """<!DOCTYPE html><table>
<thead><tr><th>X</th></tr></thead><tbody><tr><td>1</td></tr></tbody></table>"""
        result = sut.semantic_table_query(html, {})
        self.assertEqual(result["source_table_count"], 1)
        self.assertEqual(result["rows"], [["1"]])

    def test_form_with_no_name_input_skipped(self) -> None:
        html = """<!DOCTYPE html><form>
<input type="text">
<input name="valid" type="text">
</form>"""
        desc = sut.semantic_form_describe(html)
        self.assertEqual(len(desc["controls"]), 1)
        self.assertEqual(desc["controls"][0]["name"], "valid")

    def test_form_duplicate_names_rejected_for_submission(self) -> None:
        html = """<!DOCTYPE html><form>
<input name="x" type="text" value="first">
<input name="x" type="text" value="second">
</form>"""
        with self.assertRaisesRegex(ValueError, "duplicate field names"):
            sut.semantic_form_submission(
                html, "https://example.com/", {"x": "value"}
            )

    def test_same_origin_empty_action_uses_current(self) -> None:
        html = """<!DOCTYPE html><form>
<input name="q" type="text">
</form>"""
        result = sut.semantic_form_submission(
            html, "https://mysite.com/path", {"q": "hello"}
        )
        self.assertIn("https://mysite.com/path", result["url"])

    def test_form_existing_query_params_preserved(self) -> None:
        html = """<!DOCTYPE html><form action="/search?page=1">
<input name="q" type="text">
</form>"""
        result = sut.semantic_form_submission(
            html, "https://example.com/", {"q": "hello"}
        )
        self.assertIn("page=1", result["url"])
        self.assertIn("q=hello", result["url"])


if __name__ == "__main__":
    unittest.main()
