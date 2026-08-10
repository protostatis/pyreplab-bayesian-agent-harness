"""Deterministic fixture template tests.

Verifies all 8 templates generate valid, deterministic Output with
embedded nonces, proper HTML, and difficulty scaling.
"""

from __future__ import annotations

import html.parser
import json
import unittest

from pyreplab_harness.fixture_templates import (
    DIFFICULTIES,
    TEMPLATES,
    FixturePage,
    generate_nonce,
    generate_page,
)


class HtmlWellFormedTestMixin:
    """Shared helpers for HTML well-formedness checks."""

    @staticmethod
    def _parse_html(html_str: str) -> html.parser.HTMLParser:
        parser = _CheckParser()
        parser.feed(html_str)
        return parser

    @staticmethod
    def _has_table(html_str: str) -> bool:
        return "<table" in html_str and "</table>" in html_str

    @staticmethod
    def _has_form(html_str: str) -> bool:
        return "<form" in html_str and "</form>" in html_str

    @staticmethod
    def _has_links(html_str: str) -> bool:
        return "<a " in html_str and "href=" in html_str

    @staticmethod
    def _nonce_in_content(html_str: str, nonce: str) -> bool:
        """Nonce must be in visible content, not comments/meta."""
        return nonce in html_str


class _CheckParser(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.errors: list[str] = []
        self.tag_stack: list[str] = []

    def handle_starttag(self, tag, attrs):
        # void elements don't need closing
        if tag not in (
            "area", "base", "br", "col", "embed", "hr", "img", "input",
            "link", "meta", "param", "source", "track", "wbr",
        ):
            self.tag_stack.append(tag)

    def handle_endtag(self, tag):
        if tag in (
            "area", "base", "br", "col", "embed", "hr", "img", "input",
            "link", "meta", "param", "source", "track", "wbr",
        ):
            return
        if self.tag_stack and self.tag_stack[-1] == tag:
            self.tag_stack.pop()
        elif tag in self.tag_stack:
            # unclosed intermediate, pop until match
            while self.tag_stack and self.tag_stack[-1] != tag:
                self.errors.append(f"unclosed <{self.tag_stack.pop()}>")
            if self.tag_stack:
                self.tag_stack.pop()
        else:
            self.errors.append(f"unexpected </{tag}>")


class FixtureTemplatesTest(unittest.TestCase, HtmlWellFormedTestMixin):
    """Core template property tests across all 8 templates."""

    # ------------------------------------------------------------------
    # Determinism
    # ------------------------------------------------------------------

    def test_determinism_same_seed_same_output(self) -> None:
        for template in TEMPLATES:
            for difficulty in DIFFICULTIES:
                with self.subTest(template=template, difficulty=difficulty):
                    a = generate_page(template, 42, difficulty)
                    b = generate_page(template, 42, difficulty)
                    self.assertEqual(a.html, b.html,
                                     f"{template}/{difficulty}: html differs")
                    self.assertEqual(a.nonce, b.nonce)
                    self.assertEqual(a.oracle, b.oracle)

    def test_different_seeds_produce_different_output(self) -> None:
        for template in TEMPLATES:
            for difficulty in DIFFICULTIES:
                with self.subTest(template=template, difficulty=difficulty):
                    a = generate_page(template, 42, difficulty)
                    b = generate_page(template, 99, difficulty)
                    self.assertNotEqual(a.html, b.html,
                                        f"{template}/{difficulty}: html identical across seeds")
                    self.assertNotEqual(a.nonce, b.nonce)
                    self.assertNotEqual(a.oracle, b.oracle)

    def test_different_difficulties_produce_different_output(self) -> None:
        for template in TEMPLATES:
            with self.subTest(template=template):
                easy = generate_page(template, 42, "easy")
                medium = generate_page(template, 42, "medium")
                hard = generate_page(template, 42, "hard")
                self.assertNotEqual(easy.html, medium.html,
                                    f"{template}: easy == medium")
                self.assertNotEqual(medium.html, hard.html,
                                    f"{template}: medium == hard")
                self.assertNotEqual(easy.html, hard.html,
                                    f"{template}: easy == hard")

    # ------------------------------------------------------------------
    # Nonce
    # ------------------------------------------------------------------

    def test_nonce_present_in_html(self) -> None:
        for template in TEMPLATES:
            for difficulty in DIFFICULTIES:
                with self.subTest(template=template, difficulty=difficulty):
                    page = generate_page(template, 42, difficulty)
                    nonce_page = self._find_nonce_page(template, 42, difficulty, page)
                    self.assertIsNotNone(
                        nonce_page,
                        f"{template}/{difficulty}: could not locate nonce page",
                    )
                    self.assertIn(
                        page.nonce, nonce_page.html,
                        f"{template}/{difficulty}: nonce not in target page html",
                    )
                    # Ensure nonce is in visible content area, not in a comment
                    stripped = nonce_page.html.replace("<!--", "").replace("-->", "")
                    self.assertIn(
                        page.nonce, stripped,
                        f"{template}/{difficulty}: nonce only in comments",
                    )

    @staticmethod
    def _find_nonce_page(
        template: str, seed: int, difficulty: str, main_page: FixturePage,
    ) -> FixturePage | None:
        """Return the page in the fixture that should contain the nonce."""
        nonce = main_page.nonce
        oracle = main_page.oracle

        # Templates where nonce is on the main page
        if template in ("single_page_extraction", "table_filter_sort"):
            if nonce in main_page.html:
                return main_page
            return None

        # multi_page_navigation: nonce is on the target sub-page
        if template == "multi_page_navigation":
            target_page = oracle.get("target_page")
            if target_page:
                return generate_page(template, seed, difficulty, page=target_page)
            return None

        # search_filter_controls: nonce is on the search results page
        if template == "search_filter_controls":
            search_term = oracle.get("search_term", "quantum")
            return generate_page(
                template, seed, difficulty,
                query_params={"q": search_term},
            )

        # form_entry_validation: nonce is on the success page
        if template == "form_entry_validation":
            correct = oracle.get("correct_values", {})
            if correct:
                return generate_page(
                    template, seed, difficulty,
                    query_params=correct,
                )
            return None

        # cross_page_comparison: nonce is on the target sub-page
        if template == "cross_page_comparison":
            target_loc = oracle.get("target_location")
            num_pages = oracle.get("num_pages", 3)
            if target_loc:
                # Try each sub-page until we find one with the nonce
                for i in range(num_pages):
                    sub = generate_page(
                        template, seed, difficulty, page=f"page_{i}",
                    )
                    if nonce in sub.html:
                        return sub
            return None

        # stateful_workflow: nonce is on the final success step
        if template == "stateful_workflow":
            correct_path = oracle.get("correct_path", [])
            state_tokens = oracle.get("state_tokens", [])
            num_steps = oracle.get("num_steps", 3)
            if correct_path and len(state_tokens) == num_steps:
                last_step = num_steps - 1
                return generate_page(
                    template, seed, difficulty,
                    page=f"step/{last_step}",
                    query_params={
                        "choice": str(correct_path[-1]),
                        "state": str(state_tokens[-1]),
                    },
                )
            return None

        # distractor_recovery: nonce is on the correct sub-page
        if template == "distractor_recovery":
            correct_page = oracle.get("correct_page")
            if correct_page:
                return generate_page(
                    template, seed, difficulty, page=correct_page,
                )
            return None

        return None

    def test_oracle_matches_page_nonce(self) -> None:
        for template in TEMPLATES:
            for difficulty in DIFFICULTIES:
                with self.subTest(template=template, difficulty=difficulty):
                    page = generate_page(template, 42, difficulty)
                    self.assertEqual(
                        page.nonce,
                        page.oracle["nonce"],
                        f"{template}/{difficulty}: oracle nonce mismatch",
                    )
                    self.assertIn("expected_answer", page.oracle)
                    self.assertIn("verification_type", page.oracle)

    def test_nonce_deterministic_from_helper(self) -> None:
        for template in TEMPLATES:
            for difficulty in DIFFICULTIES:
                with self.subTest(template=template, difficulty=difficulty):
                    n1 = generate_nonce(template, 42, difficulty)
                    n2 = generate_nonce(template, 42, difficulty)
                    self.assertEqual(n1, n2)
                    self.assertTrue(n1.startswith("KEY_"), f"bad nonce format: {n1}")

    # ------------------------------------------------------------------
    # FixturePage structure
    # ------------------------------------------------------------------

    def test_all_templates_produce_valid_fixture_page(self) -> None:
        for template in TEMPLATES:
            for difficulty in DIFFICULTIES:
                with self.subTest(template=template, difficulty=difficulty):
                    page = generate_page(template, 42, difficulty)
                    self.assertIsInstance(page, FixturePage)
                    self.assertIsInstance(page.html, str)
                    self.assertTrue(len(page.html) > 200,
                                    f"{template}/{difficulty}: html too short")
                    self.assertIsInstance(page.title, str)
                    self.assertTrue(len(page.title) > 0)
                    self.assertIsInstance(page.oracle, dict)

    def test_html_well_formed(self) -> None:
        for template in TEMPLATES:
            for difficulty in DIFFICULTIES:
                with self.subTest(template=template, difficulty=difficulty):
                    page = generate_page(template, 42, difficulty)
                    parser = self._parse_html(page.html)
                    self.assertEqual(
                        parser.errors, [],
                        f"{template}/{difficulty}: html parse errors: {parser.errors}",
                    )

    # ------------------------------------------------------------------
    # Difficulty scaling (content should grow)
    # ------------------------------------------------------------------

    def test_difficulty_scaling_increases_size(self) -> None:
        for template in TEMPLATES:
            with self.subTest(template=template):
                easy = generate_page(template, 7, "easy")
                medium = generate_page(template, 7, "medium")
                hard = generate_page(template, 7, "hard")
                self.assertLess(
                    len(easy.html), len(medium.html),
                    f"{template}: easy html not smaller than medium",
                )
                self.assertLess(
                    len(medium.html), len(hard.html),
                    f"{template}: medium html not smaller than hard",
                )

    # ------------------------------------------------------------------
    # Template-specific checks
    # ------------------------------------------------------------------

    def test_template_1_has_table_with_rows(self) -> None:
        page = generate_page("single_page_extraction", 42, "medium")
        self._parse_html(page.html)
        self._has_table(page.html)
        # Count <tr> elements (excluding thead tr)
        tr_count = page.html.count("<tr>")
        self.assertGreater(tr_count, 5, "too few table rows")

    def test_template_2_has_filter_sort_instructions(self) -> None:
        page = generate_page("table_filter_sort", 42, "medium")
        self._has_table(page.html)
        self.assertIn("category", page.html.lower())
        self.assertIn("price", page.html.lower())

    def test_template_3_has_navigation_links(self) -> None:
        page = generate_page("multi_page_navigation", 42, "medium")
        self._has_links(page.html)
        # Sub-pages exist
        sub = generate_page("multi_page_navigation", 42, "medium", page="page_0")
        self.assertIn(sub.title, sub.html)
        self.assertIsInstance(sub, FixturePage)

    def test_template_4_search_form_and_results(self) -> None:
        page = generate_page("search_filter_controls", 42, "medium")
        self._has_form(page.html)
        # Results page
        results = generate_page(
            "search_filter_controls", 42, "medium",
            query_params={"q": page.oracle.get("search_term", "quantum")},
        )
        self.assertIn("Search Results", results.title)

    def test_template_5_form_validation(self) -> None:
        page = generate_page("form_entry_validation", 42, "medium")
        self._has_form(page.html)
        # Success page
        correct = page.oracle.get("correct_values", {})
        if correct:
            success = generate_page(
                "form_entry_validation", 42, "medium",
                query_params=correct,
            )
            self.assertIn("successful", success.html.lower())
            self.assertIn(success.nonce, success.html)

    def test_template_6_cross_page_comparison(self) -> None:
        page = generate_page("cross_page_comparison", 42, "medium")
        self._has_links(page.html)
        self.assertIn("revenue", page.html.lower())
        # Sub-page
        sub = generate_page("cross_page_comparison", 42, "medium", page="page_0")
        self._has_table(sub.html)

    def test_template_7_stateful_workflow(self) -> None:
        page = generate_page("stateful_workflow", 42, "medium")
        self._has_links(page.html)
        # A later step cannot be opened without prior-state proof.
        step = generate_page(
            "stateful_workflow", 42, "medium",
            page="step/0", query_params={"choice": "0"},
        )
        self.assertIn("Workflow State Error", step.html)
        self.assertNotIn(page.nonce, step.html)

    def test_stateful_workflow_requires_correct_order(self) -> None:
        page = generate_page("stateful_workflow", 42, "medium")
        correct_path = page.oracle["correct_path"]
        state_tokens = page.oracle["state_tokens"]
        for step_num, choice in enumerate(correct_path):
            result = generate_page(
                "stateful_workflow",
                42,
                "medium",
                page=f"step/{step_num}",
                query_params={
                    "choice": str(choice),
                    "state": state_tokens[step_num],
                },
            )
        self.assertIn(page.nonce, result.html)

    def test_semantic_templates_do_not_repeat_nonce_as_generic_footer(self) -> None:
        for template in ("single_page_extraction", "table_filter_sort"):
            page = generate_page(template, 42, "medium")
            self.assertEqual(page.html.count(page.nonce), 1, template)

        search = generate_page("search_filter_controls", 42, "medium")
        wrong_results = generate_page(
            "search_filter_controls",
            42,
            "medium",
            query_params={"q": "not-the-target-category"},
        )
        self.assertNotIn(search.nonce, wrong_results.html)

        comparison = generate_page("cross_page_comparison", 42, "medium")
        for index in range(comparison.oracle["num_pages"]):
            branch = generate_page(
                "cross_page_comparison", 42, "medium", page=f"page_{index}"
            )
            self.assertNotIn("Top Performer", branch.html)
            self.assertEqual(branch.html.count("BRANCH VERIFICATION KEY:"), 1)

    def test_template_8_distractor_recovery(self) -> None:
        page = generate_page("distractor_recovery", 42, "medium")
        self._has_links(page.html)
        # Decoy page exists and returns error
        decoy = generate_page("distractor_recovery", 42, "medium", page="page_0")
        self.assertIn("Error", decoy.title)

    # ------------------------------------------------------------------
    # Prompt-injection safety (template 8)
    # ------------------------------------------------------------------

    def test_prompt_injection_does_not_change_correct_answer(self) -> None:
        """Prompt injection decoys must not alter the oracle's expected answer."""
        for difficulty in DIFFICULTIES:
            with self.subTest(difficulty=difficulty):
                page = generate_page("distractor_recovery", 42, difficulty)
                # The expected answer must still be the deterministic nonce
                self.assertEqual(
                    page.oracle["expected_answer"],
                    page.nonce,
                    f"distractor_recovery/{difficulty}: injection altered answer",
                )
                # Injection strings must be present in the HTML
                self.assertIn(
                    "Ignore", page.html,
                    f"distractor_recovery/{difficulty}: no injection text found",
                )
                # But the nonce must also be in the correct page, not just the main
                correct_page_id = page.oracle.get("correct_page")
                if correct_page_id:
                    correct = generate_page(
                        "distractor_recovery", 42, difficulty, page=correct_page_id,
                    )
                    self.assertIn(correct.nonce, correct.html)

    # ------------------------------------------------------------------
    # No real-world data leakage
    # ------------------------------------------------------------------

    def test_no_real_person_names(self) -> None:
        """Generated pages should not contain real-world sensitive names."""
        real_names = [
            "Barack Obama", "Donald Trump", "Elon Musk",
            "Jeff Bezos", "Bill Gates", "Taylor Swift",
        ]
        for template in TEMPLATES:
            for difficulty in DIFFICULTIES:
                with self.subTest(template=template, difficulty=difficulty):
                    page = generate_page(template, 42, difficulty)
                    for name in real_names:
                        self.assertNotIn(
                            name, page.html,
                            f"{template}/{difficulty}: contains real name {name!r}",
                        )

    def test_no_real_companies_or_tickers(self) -> None:
        """Should not have real company names that could be memorized."""
        real_companies = [
            "Apple Inc.", "Google LLC", "Microsoft Corporation",
            "Amazon", "Meta Platforms", "Tesla",
        ]
        for template in TEMPLATES:
            for difficulty in DIFFICULTIES:
                with self.subTest(template=template, difficulty=difficulty):
                    page = generate_page(template, 42, difficulty)
                    for name in real_companies:
                        self.assertNotIn(
                            name, page.html,
                            f"{template}/{difficulty}: contains {name!r}",
                        )

    # ------------------------------------------------------------------
    # Unknown template / difficulty raises
    # ------------------------------------------------------------------

    def test_unknown_template_raises_valueerror(self) -> None:
        with self.assertRaises(ValueError):
            generate_page("nonexistent", 42, "easy")

    def test_unknown_difficulty_raises_valueerror(self) -> None:
        with self.assertRaises(ValueError):
            generate_page("single_page_extraction", 42, "impossible")

    # ------------------------------------------------------------------
    # Oracle is JSON-serializable
    # ------------------------------------------------------------------

    def test_oracle_is_json_serializable(self) -> None:
        for template in TEMPLATES:
            for difficulty in DIFFICULTIES:
                with self.subTest(template=template, difficulty=difficulty):
                    page = generate_page(template, 42, difficulty)
                    try:
                        json.dumps(page.oracle)
                    except (TypeError, ValueError) as exc:
                        self.fail(
                            f"{template}/{difficulty}: oracle not JSON-serializable: {exc}"
                        )


if __name__ == "__main__":
    unittest.main()
