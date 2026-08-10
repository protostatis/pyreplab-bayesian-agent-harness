from __future__ import annotations

import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from pyreplab_harness.unbrowser_rpc import (
    UNBROWSER_SMOKE_URL,
    UnbrowserProtocolError,
    UnbrowserSession,
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


if __name__ == "__main__":
    unittest.main()
