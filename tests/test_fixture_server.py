"""Fixture server integration tests.

Starts a live FixtureServer, makes HTTP requests, and verifies behaviour.
"""

from __future__ import annotations

import json
import threading
import unittest
import urllib.parse
import urllib.request
import urllib.error

from pyreplab_harness.fixture_server import FixtureServer
from pyreplab_harness.fixture_templates import (
    DIFFICULTIES,
    TEMPLATES,
    generate_nonce,
)


class FixtureServerTest(unittest.TestCase):
    """Integration tests for FixtureServer."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._server = FixtureServer(host="127.0.0.1", port=0)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._server.stop()

    # ------------------------------------------------------------------
    # Basic serving
    # ------------------------------------------------------------------

    def test_server_starts_and_serves_page(self) -> None:
        url = self._server.url_for("single_page_extraction", 42, "easy")
        code, body = self._get(url)
        self.assertEqual(code, 200)
        self.assertIn("<!DOCTYPE html>", body)
        self.assertIn("Delta Corp", body)

    def test_all_templates_serve_200(self) -> None:
        for template in TEMPLATES:
            for difficulty in DIFFICULTIES:
                with self.subTest(template=template, difficulty=difficulty):
                    url = self._server.url_for(template, 42, difficulty)
                    code, body = self._get(url)
                    self.assertEqual(
                        code, 200,
                        f"{template}/{difficulty}: got {code}",
                    )
                    self.assertIn("<!DOCTYPE html>", body)

    # ------------------------------------------------------------------
    # URL routing
    # ------------------------------------------------------------------

    def test_valid_url_returns_200(self) -> None:
        url = self._server.url_for("single_page_extraction", 42, "easy")
        code, _ = self._get(url)
        self.assertEqual(code, 200)

    def test_unknown_template_returns_404(self) -> None:
        url = f"{self._server.base_url}/nonexistent/42/easy"
        code, body = self._get(url)
        self.assertEqual(code, 404)
        self.assertIn("Unknown template", body)

    def test_invalid_seed_returns_400(self) -> None:
        url = f"{self._server.base_url}/single_page_extraction/notanumber/easy"
        code, body = self._get(url)
        self.assertEqual(code, 400)
        self.assertIn("Seed must be an integer", body)

    def test_unknown_difficulty_returns_400(self) -> None:
        url = f"{self._server.base_url}/single_page_extraction/42/impossible"
        code, body = self._get(url)
        self.assertEqual(code, 400)
        self.assertIn("Unknown difficulty", body)

    def test_malformed_path_returns_400(self) -> None:
        url = f"{self._server.base_url}/only_one_segment"
        code, body = self._get(url)
        self.assertEqual(code, 400)

    # ------------------------------------------------------------------
    # Determinism via HTTP
    # ------------------------------------------------------------------

    def test_same_url_returns_same_content(self) -> None:
        url = self._server.url_for("single_page_extraction", 42, "easy")
        _, body1 = self._get(url)
        _, body2 = self._get(url)
        self.assertEqual(body1, body2)

    def test_different_seeds_different_content(self) -> None:
        url_a = self._server.url_for("table_filter_sort", 42, "medium")
        url_b = self._server.url_for("table_filter_sort", 99, "medium")
        _, body_a = self._get(url_a)
        _, body_b = self._get(url_b)
        self.assertNotEqual(body_a, body_b)

    # ------------------------------------------------------------------
    # Nonce in served content
    # ------------------------------------------------------------------

    def test_page_contains_nonce(self) -> None:
        template = "single_page_extraction"
        difficulty = "medium"
        seed = 42
        url = self._server.url_for(template, seed, difficulty)
        nonce = generate_nonce(template, seed, difficulty)
        _, body = self._get(url)
        self.assertIn(nonce, body,
                       f"nonce {nonce} missing from page at {url}")

    # ------------------------------------------------------------------
    # Oracle
    # ------------------------------------------------------------------

    def test_oracle_contains_expected_fields(self) -> None:
        oracle = self._server.oracle_for("single_page_extraction", 42, "easy")
        self.assertIn("expected_answer", oracle)
        self.assertIn("nonce", oracle)
        self.assertIn("verification_type", oracle)

    def test_oracle_nonce_matches_page_nonce(self) -> None:
        for template in TEMPLATES[:3]:  # sample first 3
            for difficulty in DIFFICULTIES:
                with self.subTest(template=template, difficulty=difficulty):
                    oracle = self._server.oracle_for(template, 42, difficulty)
                    server_nonce = self._server.nonce_for(template, 42, difficulty)
                    self.assertEqual(oracle["nonce"], server_nonce)

    def test_oracle_is_deterministic(self) -> None:
        a = self._server.oracle_for("single_page_extraction", 42, "easy")
        b = self._server.oracle_for("single_page_extraction", 42, "easy")
        self.assertEqual(a, b)

    # ------------------------------------------------------------------
    # Content-Type header
    # ------------------------------------------------------------------

    def test_content_type_is_html(self) -> None:
        url = self._server.url_for("single_page_extraction", 42, "easy")
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            content_type = resp.headers.get("Content-Type", "")
            self.assertIn("text/html", content_type)

    # ------------------------------------------------------------------
    # Sub-page routing
    # ------------------------------------------------------------------

    def test_multi_page_navigation_sub_pages(self) -> None:
        url = self._server.url_for_page("multi_page_navigation", 42, "medium", "page_0")
        code, body = self._get(url)
        self.assertEqual(code, 200)
        self.assertIn("Report", body)

    def test_cross_page_comparison_sub_pages(self) -> None:
        url = self._server.url_for_page("cross_page_comparison", 42, "medium", "page_0")
        code, body = self._get(url)
        self.assertEqual(code, 200)
        self.assertIn("Revenue", body)

    # ------------------------------------------------------------------
    # Search query (template 4)
    # ------------------------------------------------------------------

    def test_search_form_with_query(self) -> None:
        base = self._server.url_for("search_filter_controls", 42, "medium")
        url = f"{base}?q=quantum"
        code, body = self._get(url)
        self.assertEqual(code, 200)
        # Could be results or "no items" — both are valid 200 responses
        self.assertIn("Search Results", body)

    def test_recovery_probe_returns_non_200(self) -> None:
        url = self._server.url_for_page(
            "distractor_recovery", 42, "medium", "page_0"
        )
        code, body = self._get(url)
        self.assertEqual(code, 503)
        self.assertIn("Error", body)

    # ------------------------------------------------------------------
    # Form validation (template 5)
    # ------------------------------------------------------------------

    def test_form_validation_success_route(self) -> None:
        oracle = self._server.oracle_for("form_entry_validation", 42, "easy")
        correct = oracle.get("correct_values", {})
        if correct:
            params = urllib.parse.urlencode(correct)
            base = self._server.url_for("form_entry_validation", 42, "easy")
            url = f"{base}?{params}"
            code, body = self._get(url)
            self.assertEqual(code, 200)
            self.assertIn("successful", body.lower())

    # ------------------------------------------------------------------
    # Thread safety: concurrent requests
    # ------------------------------------------------------------------

    def test_concurrent_requests(self) -> None:
        errors = []
        lock = threading.Lock()

        def fetch_one(template, seed, difficulty):
            try:
                url = self._server.url_for(template, seed, difficulty)
                code, body = self._get(url)
                if code != 200:
                    with lock:
                        errors.append(f"{template}/{seed}/{difficulty}: HTTP {code}")
                    return
                nonce = generate_nonce(template, seed, difficulty)
                # For single-page templates, nonce is on the main page
                if nonce in body:
                    return
                # For multi-page/interactive templates, find the nonce page
                sub_urls = self._resolve_nonce_urls(
                    template, seed, difficulty,
                )
                found = False
                if sub_urls:
                    for sub_url in sub_urls:
                        code2, body2 = self._get(sub_url)
                        if code2 == 200 and nonce in body2:
                            found = True
                            break
                if not found and template != "form_entry_validation":
                    # form_entry_validation requires correct form submission,
                    # which is impractical to test concurrently
                    with lock:
                        errors.append(
                            f"{template}/{seed}/{difficulty}: nonce missing"
                        )
            except Exception as exc:
                with lock:
                    errors.append(f"{template}/{seed}/{difficulty}: {exc}")

        threads = []
        for i, template in enumerate(TEMPLATES):
            for difficulty in DIFFICULTIES:
                t = threading.Thread(
                    target=fetch_one,
                    args=(template, i * 10 + 1, difficulty),
                    daemon=True,
                )
                threads.append(t)
                t.start()

        for t in threads:
            t.join(timeout=10)

        self.assertEqual(errors, [], f"concurrent errors: {errors}")

    def _resolve_nonce_urls(
        self, template: str, seed: int, difficulty: str,
    ) -> list[str]:
        """For multi-page templates, return candidate URLs that may contain the nonce."""
        oracle = self._server.oracle_for(template, seed, difficulty)

        # multi_page_navigation
        if template == "multi_page_navigation":
            target_page = oracle.get("target_page")
            if target_page:
                return [self._server.url_for_page(
                    template, seed, difficulty, target_page,
                )]

        # search_filter_controls
        if template == "search_filter_controls":
            search_term = oracle.get("search_term", "")
            if search_term:
                return [f"{self._server.url_for(template, seed, difficulty)}?q={search_term}"]

        # form_entry_validation: skip (requires correct form submission)
        if template == "form_entry_validation":
            return []

        # cross_page_comparison: probe all pages
        if template == "cross_page_comparison":
            num_pages = oracle.get("num_pages", 3)
            return [
                self._server.url_for_page(template, seed, difficulty, f"page_{i}")
                for i in range(num_pages)
            ]

        # stateful_workflow
        if template == "stateful_workflow":
            correct_path = oracle.get("correct_path", [])
            state_tokens = oracle.get("state_tokens", [])
            ns = oracle.get("num_steps", 3)
            if correct_path and len(state_tokens) == ns:
                last = ns - 1
                params = urllib.parse.urlencode(
                    {"choice": correct_path[-1], "state": state_tokens[-1]}
                )
                return [(
                    f"{self._server.url_for(template, seed, difficulty)}"
                    f"/step/{last}?{params}"
                )]

        # distractor_recovery
        if template == "distractor_recovery":
            correct_page = oracle.get("correct_page")
            if correct_page:
                return [self._server.url_for_page(
                    template, seed, difficulty, correct_page,
                )]

        return []

    def test_concurrent_oracle_lookups(self) -> None:
        """Oracle lookups from multiple threads return consistent values."""
        results: dict[int, dict] = {}
        lock = threading.Lock()

        def lookup(idx):
            oracle = self._server.oracle_for(
                "single_page_extraction", 42, "easy",
            )
            with lock:
                results[idx] = oracle

        threads = [
            threading.Thread(target=lookup, args=(i,), daemon=True)
            for i in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        # All should be identical
        values = list(results.values())
        for v in values[1:]:
            self.assertEqual(values[0], v)

    # ------------------------------------------------------------------
    # Server stop / cleanup
    # ------------------------------------------------------------------

    def test_server_stops_cleanly(self) -> None:
        # Create a temporary server and stop it
        srv = FixtureServer(host="127.0.0.1", port=0)
        url = srv.url_for("single_page_extraction", 42, "easy")
        code, _ = self._get(url)
        self.assertEqual(code, 200)
        srv.stop()
        # After stop, requests should fail
        with self.assertRaises((urllib.error.URLError, ConnectionRefusedError,
                                OSError, TimeoutError)):
            urllib.request.urlopen(url, timeout=2)

    def test_stopping_twice_is_safe(self) -> None:
        srv = FixtureServer(host="127.0.0.1", port=0)
        srv.stop()
        # Should not raise
        srv.stop()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get(url: str, timeout: int = 5) -> tuple[int, str]:
        req = urllib.request.Request(url)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8", errors="replace")


if __name__ == "__main__":
    unittest.main()
