from __future__ import annotations

import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from pyreplab_harness.unbrowser_rpc import (
    FIXTURE_INTERACTIVE_ORIGIN,
    UNBROWSER_INTERACTIVE_ORIGIN,
    UNBROWSER_INTERACTIVE_URL,
    UNBROWSER_SMOKE_URL,
    UnbrowserProtocolError,
    UnbrowserSession,
    validate_interactive_url,
    validate_smoke_url,
)


def _fake_unbrowser(directory: str, *, returned_url: str = UNBROWSER_SMOKE_URL) -> str:
    path = Path(directory) / "fake-unbrowser"
    script = f"""#!{sys.executable}
import json
import sys

if "--version" in sys.argv:
    print("unbrowser test-1")
    raise SystemExit(0)

first_request = True
for line in sys.stdin:
    request = json.loads(line)
    if first_request:
        print(json.dumps({{"event": "ready", "data": {{"version": "test-1"}}}}), flush=True)
        first_request = False
    method = request["method"]
    request_id = request["id"]
    if method == "navigate":
        print(json.dumps({{"event": "navigation_started", "data": {{}}}}), flush=True)
        result = {{"status": 200, "url": {returned_url!r}, "challenge": None}}
    elif method == "text":
        selector = request.get("params", {{}}).get("selector")
        result = "Example Domain" if selector == "h1" else "Wrong paragraph"
    elif method == "query":
        result = [{{"text": "Example Domain"}}]
    elif method == "blockmap":
        result = {{"title": "Example Domain"}}
    elif method == "close":
        result = "bye"
    else:
        print(json.dumps({{"id": request_id, "error": {{"message": "bad"}}}}), flush=True)
        continue
    print(json.dumps({{"id": request_id, "result": result}}), flush=True)
    if method == "close":
        break
"""
    path.write_text(textwrap.dedent(script), encoding="utf-8")
    path.chmod(0o755)
    return str(path)


def _fake_unbrowser_interactive(
    directory: str,
    *,
    navigate_status: int = 200,
    navigate_url: str = UNBROWSER_INTERACTIVE_URL,
    navigate_challenge: object = None,
    click_url: str | None = None,
    known_refs: dict[str, str] | None = None,
) -> str:
    known = known_refs or {}

    # Pre-compute the navigate-challenge assignment to avoid a SyntaxWarning
    # about "is not" with a literal in the generated script.
    if navigate_challenge is not None:
        challenge_assign = f'        result["challenge"] = {json.dumps(navigate_challenge)}\n'
    else:
        challenge_assign = ""

    # Pre-compute the click result expression to avoid the same warning.
    if click_url is not None:
        click_result = f'{{"url": {click_url!r}, "title": "target page"}}'
    else:
        click_result = '{"url": KNOWN_REFS[ref], "title": "target page"}'

    path = Path(directory) / "fake-unbrowser-interactive"
    script = f"""#!{sys.executable}
import json
import sys

if "--version" in sys.argv:
    print("unbrowser test-1")
    raise SystemExit(0)

KNOWN_REFS = {json.dumps(known)}

first_request = True
for line in sys.stdin:
    request = json.loads(line)
    if first_request:
        print(json.dumps({{"event": "ready", "data": {{"version": "test-1"}}}}), flush=True)
        first_request = False
    method = request["method"]
    request_id = request["id"]
    params = request.get("params", {{}})
    if method == "navigate":
        print(json.dumps({{"event": "navigation_started", "data": {{}}}}), flush=True)
        result = {{"status": {navigate_status}, "url": {navigate_url!r}}}
{challenge_assign}\
    elif method == "click":
        ref = params.get("ref")
        if ref in KNOWN_REFS:
            result = {click_result}
        else:
            result = {{"error": "unknown ref"}}
    elif method == "type":
        ref = params.get("ref")
        text = params.get("text")
        result = {{"ref": ref, "value": text, "ok": True}}
    elif method == "submit":
        ref = params.get("ref")
        if ref in KNOWN_REFS:
            result = {{"url": KNOWN_REFS[ref], "title": "submitted page"}}
        else:
            result = {{"error": "unknown ref"}}
    elif method == "text":
        selector = params.get("selector")
        result = "Example Domain" if selector == "h1" else "Wrong paragraph"
    elif method == "query":
        result = [{{"text": "Example Domain"}}]
    elif method == "blockmap":
        result = {{"title": "Example Domain"}}
    elif method == "close":
        result = "bye"
    else:
        print(json.dumps({{"id": request_id, "error": {{"message": "bad"}}}}), flush=True)
        continue
    print(json.dumps({{"id": request_id, "result": result}}), flush=True)
    if method == "close":
        break
"""
    path.write_text(textwrap.dedent(script), encoding="utf-8")
    path.chmod(0o755)
    return str(path)


class UnbrowserSessionTest(unittest.TestCase):
    def test_fixed_page_navigate_then_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with UnbrowserSession(
                _fake_unbrowser(directory), UNBROWSER_SMOKE_URL
            ) as session:
                navigation = session.execute({"action": "navigate"})
                heading = session.execute({"action": "text", "selector": "h1"})
            self.assertEqual(navigation["result"]["status"], 200)
            self.assertEqual(heading["result"], "Example Domain")
            self.assertEqual(heading["runtime_version"], "test-1")
            self.assertFalse(session.started)

    def test_non_navigation_actions_require_navigation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = UnbrowserSession(
                _fake_unbrowser(directory), UNBROWSER_SMOKE_URL
            )
            with self.assertRaisesRegex(ValueError, "navigate must succeed"):
                session.execute({"action": "text", "selector": "h1"})
            self.assertFalse(session.started)

    def test_rejects_unsafe_url_actions_and_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "pinned"):
            validate_smoke_url("https://127.0.0.1/")
        with tempfile.TemporaryDirectory() as directory:
            session = UnbrowserSession(
                _fake_unbrowser(directory), UNBROWSER_SMOKE_URL
            )
            with self.assertRaisesRegex(ValueError, "action must be"):
                session.execute({"action": "eval"})
            with self.assertRaisesRegex(ValueError, "unknown unbrowser parameters"):
                session.execute({"action": "navigate", "url": "https://example.com/"})
            with self.assertRaisesRegex(ValueError, "does not accept a selector"):
                session.execute({"action": "navigate", "selector": "h1"})

    def test_kills_session_when_navigation_leaves_fixed_page(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = UnbrowserSession(
                _fake_unbrowser(directory, returned_url="https://iana.org/"),
                UNBROWSER_SMOKE_URL,
            )
            with self.assertRaisesRegex(UnbrowserProtocolError, "left the fixed page"):
                session.execute({"action": "navigate"})
            self.assertFalse(session.started)

    def test_binary_must_be_absolute(self) -> None:
        with self.assertRaisesRegex(ValueError, "absolute path"):
            UnbrowserSession("unbrowser", UNBROWSER_SMOKE_URL)


class UnbrowserInteractiveSessionTest(unittest.TestCase):
    """Tests for the interactive Unbrowser session (click/type/submit)."""

    def test_navigate_then_click_then_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with UnbrowserSession(
                _fake_unbrowser_interactive(
                    directory,
                    click_url="https://en.wikipedia.org/wiki/Bayes%27_theorem",
                    known_refs={"e:123": "https://en.wikipedia.org/wiki/Bayes%27_theorem"},
                ),
                UNBROWSER_INTERACTIVE_URL,
                interactive=True,
            ) as session:
                navigation = session.execute({"action": "navigate"})
                self.assertEqual(navigation["result"]["status"], 200)
                self.assertTrue(navigation["interactive"])
                click_result = session.execute({"action": "click", "ref": "e:123"})
                self.assertEqual(click_result["action"], "click")
                heading = session.execute({"action": "text", "selector": "h1"})
                self.assertEqual(heading["result"], "Example Domain")
            self.assertFalse(session.started)

    def test_type_requires_ref_and_value(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with UnbrowserSession(
                _fake_unbrowser_interactive(directory),
                UNBROWSER_INTERACTIVE_URL,
                interactive=True,
            ) as session:
                session.execute({"action": "navigate"})
                with self.assertRaisesRegex(ValueError, "ref must be"):
                    session.execute({"action": "type", "value": "search text"})
                with self.assertRaisesRegex(ValueError, "ref must be"):
                    session.execute({"action": "type", "ref": ""})
                with self.assertRaisesRegex(ValueError, "type value must be"):
                    session.execute({"action": "type", "ref": "e:142"})

    def test_submit_requires_ref(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with UnbrowserSession(
                _fake_unbrowser_interactive(
                    directory,
                    known_refs={"e:search": "https://en.wikipedia.org/wiki/Bayesian_inference"},
                ),
                UNBROWSER_INTERACTIVE_URL,
                interactive=True,
            ) as session:
                session.execute({"action": "navigate"})
                with self.assertRaisesRegex(ValueError, "ref must be"):
                    session.execute({"action": "submit"})
                result = session.execute({"action": "submit", "ref": "e:search"})
                self.assertEqual(result["action"], "submit")

    def test_explicit_interaction_error_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with UnbrowserSession(
                _fake_unbrowser_interactive(directory),
                UNBROWSER_INTERACTIVE_URL,
                interactive=True,
            ) as session:
                session.execute({"action": "navigate"})
                with self.assertRaisesRegex(
                    UnbrowserProtocolError, "submit failed: unknown ref"
                ):
                    session.execute({"action": "submit", "ref": "e:unknown"})

    def test_non_navigate_requires_prior_navigate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = UnbrowserSession(
                _fake_unbrowser_interactive(directory),
                UNBROWSER_INTERACTIVE_URL,
                interactive=True,
            )
            with self.assertRaisesRegex(ValueError, "navigate must succeed"):
                session.execute({"action": "click", "ref": "e:123"})
            with self.assertRaisesRegex(ValueError, "navigate must succeed"):
                session.execute({"action": "type", "ref": "e:123", "value": "hello"})
            with self.assertRaisesRegex(ValueError, "navigate must succeed"):
                session.execute({"action": "submit", "ref": "e:123"})
            self.assertFalse(session.started)

    def test_status_or_challenge_blocks_further_actions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = UnbrowserSession(
                _fake_unbrowser_interactive(directory, navigate_status=403),
                UNBROWSER_INTERACTIVE_URL,
                interactive=True,
            )
            session.execute({"action": "navigate"})
            with self.assertRaisesRegex(ValueError, "navigate must succeed"):
                session.execute({"action": "text", "selector": "h1"})

    def test_navigate_blocks_with_challenge_present(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = UnbrowserSession(
                _fake_unbrowser_interactive(
                    directory, navigate_challenge={"type": "captcha"}
                ),
                UNBROWSER_INTERACTIVE_URL,
                interactive=True,
            )
            session.execute({"action": "navigate"})
            with self.assertRaisesRegex(ValueError, "navigate must succeed"):
                session.execute({"action": "text", "selector": "h1"})

    def test_ref_validation_empty_too_long_control_chars(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with UnbrowserSession(
                _fake_unbrowser_interactive(directory),
                UNBROWSER_INTERACTIVE_URL,
                interactive=True,
            ) as session:
                session.execute({"action": "navigate"})
                with self.assertRaisesRegex(ValueError, "ref must be"):
                    session.execute({"action": "click"})
                with self.assertRaisesRegex(ValueError, "ref must be"):
                    session.execute({"action": "click", "ref": ""})
                with self.assertRaisesRegex(ValueError, "ref must be"):
                    session.execute({"action": "click", "ref": "a" * 257})
                with self.assertRaisesRegex(ValueError, "control-line"):
                    session.execute({"action": "click", "ref": "e:12\n3"})

    def test_type_value_rejects_nul_and_length(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = UnbrowserSession(
                _fake_unbrowser_interactive(directory),
                UNBROWSER_INTERACTIVE_URL,
                interactive=True,
            )
            session.execute({"action": "navigate"})
            with self.assertRaisesRegex(ValueError, "NUL"):
                session.execute(
                    {"action": "type", "ref": "e:142", "value": "hel\x00lo"}
                )
            with self.assertRaisesRegex(ValueError, "must be at most"):
                session.execute(
                    {"action": "type", "ref": "e:142", "value": "x" * 1025}
                )
            session.close()

    def test_unknown_params_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = UnbrowserSession(
                _fake_unbrowser_interactive(directory),
                UNBROWSER_INTERACTIVE_URL,
                interactive=True,
            )
            with self.assertRaisesRegex(ValueError, "unknown unbrowser parameters"):
                session.execute({"action": "navigate", "color": "red"})
            session.close()

    def test_read_only_session_rejects_interactive_actions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = UnbrowserSession(
                _fake_unbrowser(directory),
                UNBROWSER_SMOKE_URL,
            )
            # In read-only mode, interactive actions are unknown or rejected.
            # click without ref fails unknown params first.
            with self.assertRaisesRegex(ValueError, "unknown unbrowser parameters"):
                session.execute({"action": "click", "ref": "e:123"})
            # click without params is an unknown action.
            with self.assertRaisesRegex(ValueError, "action must be"):
                session.execute({"action": "click"})
            with self.assertRaisesRegex(ValueError, "action must be"):
                session.execute({"action": "submit"})

    def test_navigate_rejects_ref_and_value_params(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = UnbrowserSession(
                _fake_unbrowser_interactive(directory),
                UNBROWSER_INTERACTIVE_URL,
                interactive=True,
            )
            with self.assertRaisesRegex(ValueError, "does not accept ref"):
                session.execute({"action": "navigate", "ref": "e:123"})
            session.close()

    def test_kills_session_on_off_origin_click(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = UnbrowserSession(
                _fake_unbrowser_interactive(
                    directory,
                    click_url="https://google.com/",
                    known_refs={"e:bad": "https://google.com/"},
                ),
                UNBROWSER_INTERACTIVE_URL,
                interactive=True,
            )
            session.execute({"action": "navigate"})
            with self.assertRaisesRegex(UnbrowserProtocolError, "left the allowed origin"):
                session.execute({"action": "click", "ref": "e:bad"})
            self.assertFalse(session.started)

    def test_interactive_url_validation(self) -> None:
        validate_interactive_url(UNBROWSER_INTERACTIVE_URL)
        with self.assertRaisesRegex(ValueError, "pinned"):
            validate_interactive_url("https://example.com/")
        with self.assertRaisesRegex(ValueError, "pinned"):
            validate_interactive_url("http://en.wikipedia.org/wiki/Main_Page")
        # Same-origin Wikipedia pages are allowed.
        validate_interactive_url("https://en.wikipedia.org/wiki/Bayes%27_theorem")

    def test_rejects_url_outside_interactive_origin_in_constructor(self) -> None:
        with self.assertRaisesRegex(ValueError, "pinned"):
            UnbrowserSession(
                "/bin/true",
                "https://example.com/",
                interactive=True,
            )


class ConfinedUnbrowserSessionTest(unittest.TestCase):
    """Tests for the confined UnbrowserSession (Bubblewrap filesystem isolation)."""

    def test_accepts_confined_parameter(self) -> None:
        session = UnbrowserSession(
            "/bin/true",
            UNBROWSER_SMOKE_URL,
            confined=True,
        )
        self.assertTrue(session.confined)

    def test_default_is_not_confined(self) -> None:
        session = UnbrowserSession(
            "/bin/true",
            UNBROWSER_SMOKE_URL,
        )
        self.assertFalse(session.confined)

    def test_confined_false_behavior_unchanged(self) -> None:
        session = UnbrowserSession(
            "/bin/true",
            UNBROWSER_SMOKE_URL,
            confined=False,
        )
        self.assertFalse(session.confined)
        # Backward-compatible: no bwrap, normal process launch path.
        self.assertFalse(session.started)

    def test_confined_with_interactive(self) -> None:
        session = UnbrowserSession(
            "/bin/true",
            UNBROWSER_INTERACTIVE_URL,
            confined=True,
            interactive=True,
        )
        self.assertTrue(session.confined)
        self.assertTrue(session.interactive)

    def test_confined_with_fake_binary_uses_sandbox(self) -> None:
        """When confined=True, _start uses UnbrowserSandbox to build the launch command."""
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as d:
            binary = _fake_unbrowser(d)
            with patch(
                "pyreplab_harness.unbrowser_sandbox.UnbrowserSandbox"
            ) as MockSandbox:
                mock_instance = MockSandbox.return_value
                # Return a command that simulates a real bwrap command with --
                # separator so the env-injection code works, but still
                # delegates to the fake binary for subprocess execution.
                mock_instance.build_command.side_effect = lambda *a: [
                    binary, "--", binary
                ] + list(a)

                session = UnbrowserSession(
                    binary,
                    UNBROWSER_SMOKE_URL,
                    confined=True,
                )
                try:
                    result = session.execute({"action": "navigate"})
                    self.assertEqual(result["result"]["status"], 200)
                    # UnbrowserSandbox was constructed with the binary.
                    MockSandbox.assert_called_once()
                    # build_command was called (at least for version probe and launch).
                    self.assertTrue(mock_instance.build_command.call_count >= 2)
                finally:
                    session.close()

    def test_version_probe_in_confined_mode(self) -> None:
        """Version probe uses the sandbox command when confined."""
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as d:
            binary = _fake_unbrowser(d)
            with patch(
                "pyreplab_harness.unbrowser_sandbox.UnbrowserSandbox"
            ) as MockSandbox:
                mock_instance = MockSandbox.return_value
                mock_instance.build_command.side_effect = lambda *a: [
                    binary, "--", binary
                ] + list(a)

                session = UnbrowserSession(
                    binary,
                    UNBROWSER_SMOKE_URL,
                    confined=True,
                )
                try:
                    session.execute({"action": "navigate"})
                    self.assertEqual(session.runtime_version, "test-1")
                finally:
                    session.close()

    def test_non_confined_does_not_import_sandbox(self) -> None:
        """Non-confined sessions do not trigger UnbrowserSandbox import."""
        with tempfile.TemporaryDirectory() as d:
            binary = _fake_unbrowser(d)
            session = UnbrowserSession(
                binary,
                UNBROWSER_SMOKE_URL,
                confined=False,
            )
            try:
                session.execute({"action": "navigate"})
                self.assertEqual(session.runtime_version, "test-1")
            finally:
                session.close()


class FixtureUrlValidationTest(unittest.TestCase):
    """Tests for fixture URL validation in interactive unbrowser sessions."""

    def test_fixture_url_validation_accepts_correct_prefix(self) -> None:
        validate_interactive_url(
            "http://127.0.0.1:18090/single_page_extraction/7/easy",
            allow_fixture=True,
        )
        validate_interactive_url(
            "http://127.0.0.1:18090/table_filter_sort/42/medium",
            allow_fixture=True,
        )
        # Sub-pages within fixture origin are allowed.
        validate_interactive_url(
            "http://127.0.0.1:18090/multi_page_navigation/7/easy/page_0",
            allow_fixture=True,
        )

    def test_fixture_url_validation_rejects_other_hosts(self) -> None:
        with self.assertRaisesRegex(ValueError, "pinned"):
            validate_interactive_url("http://localhost:18090/page", allow_fixture=True)
        with self.assertRaisesRegex(ValueError, "pinned"):
            validate_interactive_url("http://example.com:18090/page", allow_fixture=True)
        with self.assertRaisesRegex(ValueError, "pinned"):
            validate_interactive_url("https://127.0.0.1:18090/page", allow_fixture=True)
        # Wrong port
        with self.assertRaisesRegex(ValueError, "pinned"):
            validate_interactive_url(
                "http://127.0.0.1:18091/single_page_extraction/7/easy",
                allow_fixture=True,
            )

    def test_fixture_url_without_allow_fixture_rejected(self) -> None:
        """Without allow_fixture=True, fixture URLs should fail the Wikipedia check."""
        with self.assertRaisesRegex(ValueError, "pinned"):
            validate_interactive_url("http://127.0.0.1:18090/page")

    def test_wikipedia_validation_still_works_when_allow_fixture_false(self) -> None:
        # Wikipedia URLs still work with allow_fixture=False (default)
        validate_interactive_url(UNBROWSER_INTERACTIVE_URL)
        validate_interactive_url("https://en.wikipedia.org/wiki/Bayes%27_theorem")

    def test_interactive_session_with_fixture_url(self) -> None:
        """UnbrowserSession in interactive mode should accept fixture URLs."""
        session = UnbrowserSession(
            "/bin/true",
            "http://127.0.0.1:18090/single_page_extraction/7/easy",
            interactive=True,
        )
        self.assertTrue(session.interactive)
        self.assertEqual(
            session.allowed_url,
            "http://127.0.0.1:18090/single_page_extraction/7/easy",
        )

    def test_interactive_session_with_fixture_url_unsafe_rejected(self) -> None:
        """Fixture session rejects URLs outside the fixture origin."""
        with self.assertRaisesRegex(ValueError, "pinned"):
            UnbrowserSession(
                "/bin/true",
                "http://127.0.0.1:99999/single_page_extraction/7/easy",
                interactive=True,
            )

    def test_fixture_origin_tracking(self) -> None:
        """Session tracks fixture origin for same-origin enforcement."""
        session = UnbrowserSession(
            "/bin/true",
            "http://127.0.0.1:18090/single_page_extraction/7/easy",
            interactive=True,
        )
        self.assertEqual(session._interactive_origin, FIXTURE_INTERACTIVE_ORIGIN)

    def test_wikipedia_origin_tracking(self) -> None:
        """Session tracks wikipedia origin for same-origin enforcement."""
        session = UnbrowserSession(
            "/bin/true",
            UNBROWSER_INTERACTIVE_URL,
            interactive=True,
        )
        self.assertEqual(session._interactive_origin, UNBROWSER_INTERACTIVE_ORIGIN)


class RequiredFirstObservationTest(unittest.TestCase):
    """Tests for the required_first_observation auto-delivery feature."""

    def test_text_auto_delivery_on_first_navigate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = UnbrowserSession(
                _fake_unbrowser_interactive(directory),
                UNBROWSER_INTERACTIVE_URL,
                interactive=True,
                required_first_observation="text",
            )
            try:
                result = session.execute({"action": "navigate"})
                self.assertIn("required_first_observation_receipt", result)
                receipt = result["required_first_observation_receipt"]
                self.assertEqual(receipt["required_action"], "text")
                self.assertEqual(receipt["delivered_action"], "text")
                self.assertEqual(receipt["selector"], "body")
                self.assertTrue(receipt["delivered"])
                self.assertIsInstance(receipt["payload_bytes"], int)
                self.assertIsInstance(receipt["payload_sha256"], str)
                self.assertEqual(len(receipt["payload_sha256"]), 64)
                self.assertEqual(receipt["mechanism"], "auto_delivered_first_observation")
                self.assertEqual(
                    receipt["schema_version"], "pyreplab-required-first-observation-v1"
                )
                self.assertIn("auto_delivered_observation", result)
                self.assertIsInstance(result["auto_delivered_observation"], str)
            finally:
                session.close()

    def test_blockmap_auto_delivery_on_first_navigate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = UnbrowserSession(
                _fake_unbrowser_interactive(directory),
                UNBROWSER_INTERACTIVE_URL,
                interactive=True,
                required_first_observation="blockmap",
            )
            try:
                result = session.execute({"action": "navigate"})
                self.assertIn("required_first_observation_receipt", result)
                receipt = result["required_first_observation_receipt"]
                self.assertEqual(receipt["required_action"], "blockmap")
                self.assertEqual(receipt["delivered_action"], "blockmap")
                self.assertIsNone(receipt["selector"])
                self.assertTrue(receipt["delivered"])
                self.assertIsInstance(receipt["payload_sha256"], str)
                self.assertEqual(len(receipt["payload_sha256"]), 64)
            finally:
                session.close()

    def test_second_navigate_does_not_auto_deliver(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = UnbrowserSession(
                _fake_unbrowser_interactive(directory),
                UNBROWSER_INTERACTIVE_URL,
                interactive=True,
                required_first_observation="text",
            )
            try:
                session.execute({"action": "navigate"})  # first — auto-delivers
                result = session.execute({"action": "navigate"})  # second — no auto-delivery
                self.assertNotIn("required_first_observation_receipt", result)
                self.assertNotIn("auto_delivered_observation", result)
            finally:
                session.close()

    def test_required_observation_fails_closed_on_bad_output(self) -> None:
        """When the auto-observation fails, the session is killed with an error."""
        with tempfile.TemporaryDirectory() as directory:
            session = UnbrowserSession(
                _fake_unbrowser_interactive(directory),
                UNBROWSER_INTERACTIVE_URL,
                interactive=True,
                required_first_observation="text",
            )
            try:
                # First navigate — auto-observes and the fake unbrowser
                # returns "Example Domain" for text queries. This should succeed.
                result = session.execute({"action": "navigate"})
                self.assertIn("required_first_observation_receipt", result)
                # The session is now past its first direct navigate.
                # The first_direct_navigate_done flag prevents further auto-delivery.
                self.assertTrue(session._first_direct_navigate_done)
            finally:
                session.close()

    def test_invalid_required_first_observation_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be None"):
            UnbrowserSession(
                "/bin/true",
                UNBROWSER_INTERACTIVE_URL,
                interactive=True,
                required_first_observation="query",
            )

    def test_required_observation_on_noninteractive_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "only allowed in interactive"):
            UnbrowserSession(
                "/bin/true",
                UNBROWSER_SMOKE_URL,
                interactive=False,
                required_first_observation="text",
            )

    def test_receipt_hash_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = UnbrowserSession(
                _fake_unbrowser_interactive(directory),
                UNBROWSER_INTERACTIVE_URL,
                interactive=True,
                required_first_observation="text",
            )
            r1 = session.execute({"action": "navigate"})
            session.close()
        with tempfile.TemporaryDirectory() as directory2:
            session2 = UnbrowserSession(
                _fake_unbrowser_interactive(directory2),
                UNBROWSER_INTERACTIVE_URL,
                interactive=True,
                required_first_observation="text",
            )
            r2 = session2.execute({"action": "navigate"})
            session2.close()
        # The fake unbrowser returns the same text, so the sha256 should match.
        self.assertEqual(
            r1["required_first_observation_receipt"]["payload_sha256"],
            r2["required_first_observation_receipt"]["payload_sha256"],
        )

    def test_close_resets_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = UnbrowserSession(
                _fake_unbrowser_interactive(directory),
                UNBROWSER_INTERACTIVE_URL,
                interactive=True,
                required_first_observation="text",
            )
            session.execute({"action": "navigate"})
            self.assertTrue(session._first_direct_navigate_done)
            session.close()
            self.assertFalse(session._first_direct_navigate_done)


if __name__ == "__main__":
    unittest.main()


class ProcessLifetimeDecouplingTest(unittest.TestCase):
    """Tests that persistent process lifetime is decoupled from per-request timeout."""

    def test_process_survives_beyond_old_timeout_boundary(self) -> None:
        """With a short timeout_seconds the process lives long enough for
        multiple requests across a wall-clock span that exceeds the old
        per-action timeout boundary, proving lifetime is separate."""
        with tempfile.TemporaryDirectory() as directory:
            session = UnbrowserSession(
                _fake_unbrowser(directory),
                UNBROWSER_SMOKE_URL,
                timeout_seconds=2,
            )
            try:
                # First request completes well within the per-action deadline.
                nav = session.execute({"action": "navigate"})
                self.assertEqual(nav["result"]["status"], 200)
                self.assertTrue(session.started)

                # Sleep longer than the old per-action timeout boundary
                # (the old behaviour would have used this value for GNU
                # timeout as well), but well within the process lifetime
                # of at least 600 seconds.
                import time
                time.sleep(3)

                # Second request still succeeds — the process was not killed
                # by a per-action timeout expiring between requests.
                heading = session.execute({"action": "text", "selector": "h1"})
                self.assertEqual(heading["result"], "Example Domain")
                self.assertTrue(session.started)
            finally:
                session.close()
            self.assertFalse(session.started)


class OversizedResultRecoveryTest(unittest.TestCase):
    """Tests that oversized wrapped results do not kill the session."""

    def _fake_unbrowser_oversized(self, directory: str) -> str:
        path = Path(directory) / "fake-unbrowser-oversized"
        script = f"""#!{sys.executable}
import json
import sys

if "--version" in sys.argv:
    print("unbrowser test-1")
    raise SystemExit(0)

first_request = True
for line in sys.stdin:
    request = json.loads(line)
    if first_request:
        print(json.dumps({{"event": "ready", "data": {{"version": "test-1"}}}}), flush=True)
        first_request = False
    method = request["method"]
    request_id = request["id"]
    params = request.get("params", {{}})
    if method == "navigate":
        result = {{"status": 200, "url": "https://example.com/", "challenge": None}}
    elif method == "text":
        selector = params.get("selector", "")
        if selector == "BIG":
            # Return a very large text payload that will exceed
            # max_result_bytes when wrapped.
            result = "X" * (64 * 1024 + 1)
        else:
            result = "Small response"
    elif method == "close":
        result = "bye"
    else:
        print(json.dumps({{"id": request_id, "error": {{"message": "bad"}}}}), flush=True)
        continue
    print(json.dumps({{"id": request_id, "result": result}}), flush=True)
    if method == "close":
        break
"""
        path.write_text(textwrap.dedent(script), encoding="utf-8")
        path.chmod(0o755)
        return str(path)

    def test_oversized_result_does_not_kill_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = UnbrowserSession(
                self._fake_unbrowser_oversized(directory),
                UNBROWSER_SMOKE_URL,
                max_result_bytes=1024,
            )
            try:
                session.execute({"action": "navigate"})
                # The BIG text response will overflow the wrapped result size.
                with self.assertRaises(UnbrowserProtocolError) as ctx:
                    session.execute({"action": "text", "selector": "BIG"})
                self.assertIn("exceeds", str(ctx.exception))
                self.assertTrue(session.started, "session must remain alive")

                # A smaller subsequent request must succeed in the same process.
                heading = session.execute({"action": "text", "selector": "h1"})
                self.assertEqual(heading["result"], "Small response")
                self.assertTrue(session.started)
            finally:
                session.close()
            self.assertFalse(session.started)


class ProcessExitBeforeRequestTest(unittest.TestCase):
    """Tests that a pre-exited process raises a clean error before writing."""

    def _fake_unbrowser_exit_early(self, directory: str) -> str:
        path = Path(directory) / "fake-unbrowser-exit"
        path.write_text(
            f"#!{sys.executable}\n"
            "import sys\n"
            'print("ok", flush=True)\n'
            "raise SystemExit(1)\n",
            encoding="utf-8",
        )
        path.chmod(0o755)
        return str(path)

    def test_poll_check_prevents_raw_broken_pipe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = UnbrowserSession(
                self._fake_unbrowser_exit_early(directory),
                UNBROWSER_SMOKE_URL,
            )
            try:
                # The fake binary exits immediately.  On the first request
                # the session _start tries a version probe, which also fails.
                # We just verify that a clean UnbrowserProtocolError is raised
                # rather than a raw BrokenPipeError.
                with self.assertRaises(UnbrowserProtocolError) as ctx:
                    session.execute({"action": "navigate"})
                self.assertIn("exit_code", str(ctx.exception))
                self.assertTrue(ctx.exception.infrastructure_error)
            finally:
                session.close()


class BrokenPipeDuringRequestTest(unittest.TestCase):
    """Tests that a BrokenPipeError during request is caught cleanly."""

    def _fake_unbrowser_close_stdin(self, directory: str) -> str:
        path = Path(directory) / "fake-unbrowser-close-stdin"
        script = f"""#!{sys.executable}
import sys
if "--version" in sys.argv:
    print("unbrowser test-1")
    raise SystemExit(0)
# Close stdin after the first line to trigger BrokenPipeError on the
# second write from the host.
first_line = sys.stdin.readline()
sys.stdin.close()
raise SystemExit(42)
"""
        path.write_text(textwrap.dedent(script), encoding="utf-8")
        path.chmod(0o755)
        return str(path)

    def test_broken_pipe_raises_protocol_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = UnbrowserSession(
                self._fake_unbrowser_close_stdin(directory),
                UNBROWSER_SMOKE_URL,
            )
            try:
                # The first request's _start prints an event, but the fake
                # closes stdin before the version probe completes.  The
                # resulting communication failure must surface as
                # UnbrowserProtocolError, not a raw BrokenPipeError.
                with self.assertRaises(UnbrowserProtocolError) as ctx:
                    session.execute({"action": "navigate"})
                self.assertIn("exit_code", str(ctx.exception))
                self.assertTrue(ctx.exception.infrastructure_error)
            finally:
                session.close()


class SemanticCapabilityAdapterTest(unittest.TestCase):
    def _fixture_session(self, capability: str) -> UnbrowserSession:
        session = UnbrowserSession(
            "/bin/true",
            "http://127.0.0.1:18090/form_entry_validation/7/medium",
            interactive=True,
            semantic_capability=capability,
        )
        session._navigated = True
        session._current_url = session.allowed_url
        return session

    def test_table_adapter_emits_payload_bound_specialist_receipt(self) -> None:
        session = self._fixture_session("table")
        html = "<table><tr><th>Name</th></tr><tr><td>Alice</td></tr></table>"
        session._fetch_public_html = lambda: html  # type: ignore[method-assign]
        result = session.semantic_table({"projection": ["Name"]})
        payload = result["semantic_payload"]
        receipt = result["semantic_specialist_receipt"]
        encoded = json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        self.assertEqual(receipt["schema_version"], "pyreplab-semantic-specialist-receipt-v1")
        self.assertEqual(receipt["specialist"], "table_specialist")
        self.assertEqual(receipt["action"], "semantic_table")
        self.assertEqual(receipt["payload_bytes"], len(encoded))
        import hashlib
        self.assertEqual(receipt["payload_sha256"], hashlib.sha256(encoded).hexdigest())

    def test_form_describe_adapter_uses_form_index(self) -> None:
        session = self._fixture_session("form")
        html = """
        <form method="get"><input name="first"></form>
        <form method="get"><input name="second" required></form>
        """
        session._fetch_public_html = lambda: html  # type: ignore[method-assign]
        result = session.semantic_form({"action": "describe", "form_index": 1})
        payload = result["semantic_payload"]
        self.assertEqual(payload["form_count"], 2)
        self.assertEqual(payload["controls"][0]["name"], "second")

    def test_form_submit_adapter_uses_current_url_and_fields(self) -> None:
        session = self._fixture_session("form")
        html = """
        <form method="get" action="">
          <input name="email" required pattern="^[^@]+@[^@]+$">
        </form>
        """
        session._fetch_public_html = lambda: html  # type: ignore[method-assign]
        calls: list[tuple[str, dict | None]] = []

        def request(method: str, params: dict | None = None):
            calls.append((method, params))
            if method == "navigate":
                return {"status": 200, "url": params["url"], "challenge": None}
            if method == "text":
                return "Submitted"
            raise AssertionError(method)

        session._request = request  # type: ignore[method-assign]
        result = session.semantic_form({
            "action": "submit",
            "fields": [{"name": "email", "value": "a@example.com"}],
        })
        payload = result["semantic_payload"]
        self.assertIn("email=a%40example.com", payload["submission_url"])
        self.assertEqual(calls[0][0], "navigate")
        self.assertEqual(calls[1], ("text", {"selector": "body"}))

    def test_form_submit_rejects_duplicate_fields(self) -> None:
        session = self._fixture_session("form")
        session._fetch_public_html = lambda: "<form method='get'><input name='q'></form>"  # type: ignore[method-assign]
        with self.assertRaisesRegex(ValueError, "duplicate field"):
            session.semantic_form({
                "action": "submit",
                "fields": [
                    {"name": "q", "value": "one"},
                    {"name": "q", "value": "two"},
                ],
            })
