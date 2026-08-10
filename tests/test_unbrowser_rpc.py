from __future__ import annotations

import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from pyreplab_harness.unbrowser_rpc import (
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
            with self.assertRaisesRegex(UnbrowserProtocolError, "left wikipedia"):
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


if __name__ == "__main__":
    unittest.main()
