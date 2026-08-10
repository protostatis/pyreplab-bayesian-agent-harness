"""Deterministic fixture web server for the Unbrowser interactive harness.

Stdlib-only HTTP server that serves seeded fixture pages at configurable
URLs.  Runs in a background thread.  Thread-safe oracle lookups.
"""

from __future__ import annotations

import threading
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

from .fixture_templates import (
    DIFFICULTIES,
    TEMPLATES,
    FixturePage,
    generate_page,
    generate_nonce,
)


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------

class _FixtureHandler(BaseHTTPRequestHandler):
    """Serves deterministic fixture pages."""

    # Silence stdout logging (stderr only for real errors)
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query_params = dict(urllib.parse.parse_qsl(parsed.query))

        # Path: /<template>/<seed>/<difficulty>[/<remainder>...]
        parts = [p for p in path.split("/") if p]

        if len(parts) < 3:
            self._send_error(400, "URL must be /<template>/<seed>/<difficulty>[/<page>]")
            return

        template = parts[0]
        if template not in TEMPLATES:
            self._send_error(404, f"Unknown template: {template!r}")
            return

        try:
            seed = int(parts[1])
        except ValueError:
            self._send_error(400, f"Seed must be an integer, got: {parts[1]!r}")
            return

        difficulty = parts[2]
        if difficulty not in DIFFICULTIES:
            self._send_error(400, f"Unknown difficulty: {difficulty!r}")
            return

        # Remainder after /<template>/<seed>/<difficulty> becomes the page id
        page = "/".join(parts[3:]) if len(parts) > 3 else None

        try:
            fixture = generate_page(
                template=template,
                seed=seed,
                difficulty=difficulty,
                page=page,
                query_params=query_params,
            )
        except Exception as exc:
            self._send_error(500, f"Page generation failed: {exc}")
            return

        body = fixture.html.encode("utf-8")
        self.send_response(fixture.status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, code: int, message: str) -> None:
        body = (
            f"<!DOCTYPE html>\n<html><head><title>Error {code}</title></head>\n"
            f"<body><h1>Error {code}</h1><p>{message}</p></body></html>\n"
        ).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# FixtureServer
# ---------------------------------------------------------------------------


class FixtureServer:
    """HTTP server that serves deterministic fixture pages.

    Runs in a background thread on a configurable host and port.
    Start with ``port=0`` for auto-assignment (OS picks a free port).
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self._host = host
        self._server = HTTPServer((host, port), _FixtureHandler)
        self._port = self._server.server_port
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="fixture-server",
            daemon=True,
        )
        self._lock = threading.Lock()
        self._thread.start()

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    @property
    def base_url(self) -> str:
        """Base URL like ``http://127.0.0.1:PORT``."""
        return f"http://{self._host}:{self._port}"

    def url_for(self, template: str, seed: int, difficulty: str) -> str:
        """Return the URL for a fixture page's main/hub page."""
        return f"{self.base_url}/{template}/{seed}/{difficulty}"

    def url_for_page(
        self, template: str, seed: int, difficulty: str, page: str
    ) -> str:
        """Return the URL for a specific sub-page of a fixture."""
        return f"{self.base_url}/{template}/{seed}/{difficulty}/{page}"

    def oracle_for(
        self, template: str, seed: int, difficulty: str
    ) -> dict[str, Any]:
        """Return the hidden oracle for a fixture.

        The oracle is deterministic and thread-safe (calling this from
        multiple threads concurrently returns the same value for the same
        arguments).
        """
        # Oracle is deterministic; no lock needed for the computation.
        # We use a lock only to serialize access to the RNG-less generation
        # (though generate_page is already deterministic when called with
        # identical args).  The lock protects against any potential
        # non-thread-safe code paths in the template generators.
        with self._lock:
            fixture = generate_page(
                template=template,
                seed=seed,
                difficulty=difficulty,
                page=None,
                query_params=None,
            )
        return dict(fixture.oracle)

    def nonce_for(
        self, template: str, seed: int, difficulty: str
    ) -> str:
        """Return the deterministic nonce for a fixture."""
        return generate_nonce(template, seed, difficulty)

    def stop(self) -> None:
        """Shut down the server gracefully."""
        self._server.shutdown()
        self._server.server_close()
