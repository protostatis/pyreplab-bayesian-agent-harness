"""Strict read-only and interactive adapters for the live Unbrowser smoke test.

The model never supplies a URL or raw JSON-RPC payload.  This adapter owns one
fresh ``unbrowser`` process, pins navigation to the fixed public smoke page
(read-only) or Wikipedia (interactive), and exposes non-mutating actions plus
click/type/submit for the interactive path.  It intentionally runs outside the
Bubblewrap command sandbox, so this module is the network security boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import select
import signal
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Mapping


UNBROWSER_SMOKE_URL = "https://example.com/"
UNBROWSER_INTERACTIVE_URL = "https://en.wikipedia.org/wiki/Main_Page"
UNBROWSER_INTERACTIVE_ORIGIN = "https://en.wikipedia.org/"
FIXTURE_PORT = 18090
FIXTURE_URL_PREFIX = f"http://127.0.0.1:{FIXTURE_PORT}"
FIXTURE_INTERACTIVE_ORIGIN = f"http://127.0.0.1:{FIXTURE_PORT}/"
READ_ONLY_ACTIONS = frozenset({"navigate", "query", "text", "blockmap"})
INTERACTIVE_ACTIONS = READ_ONLY_ACTIONS | {"click", "type", "submit"}
SEMANTIC_CAPABILITIES = frozenset({"table", "form"})
MAX_SELECTOR_CHARS = 256
MAX_REF_CHARS = 256
MAX_TYPE_TEXT_CHARS = 1024
MAX_RPC_LINE_BYTES = 256 * 1024
DEFAULT_MAX_RESULT_BYTES = 64 * 1024
DEFAULT_SEMANTIC_FETCH_MAX_BYTES = 4 * 1024 * 1024  # 4 MiB

RECEIPT_SCHEMA_VERSION = "pyreplab-required-first-observation-v1"
SEMANTIC_RECEIPT_SCHEMA_VERSION = "pyreplab-semantic-specialist-receipt-v1"
_REQUIRED_OBSERVATION_CHOICES = frozenset({None, "text", "blockmap"})


class UnbrowserProtocolError(RuntimeError):
    """Raised when the child or its bounded JSON-RPC transport fails."""

    def __init__(self, message: str, *, infrastructure_error: bool = False) -> None:
        super().__init__(message)
        self.infrastructure_error = infrastructure_error


# ---------------------------------------------------------------------------
# Semantic capability stubs — delegated to semantic_browser.py at runtime
# ---------------------------------------------------------------------------

_DESCRIBE_ACTIONS = frozenset({"describe", "submit"})


def semantic_table_query(
    public_html: str, request: Mapping[str, Any]
) -> dict[str, Any]:
    """Analyse a public HTML page and return a bounded table view.

    This is a pure controller-side function that does NOT call the Unbrowser
    process.  It operates on the raw public HTML fetched via urllib.

    Parameters
    ----------
    public_html : str
        Current page HTML (public, no credentials).
    request : Mapping
        Fields: ``table_index`` (int), ``filters`` (list of {column,value}),
        ``sort`` ({column,direction}), ``offset``, ``limit``, ``projection``.

    Returns
    -------
    dict with ``columns``, ``rows``, ``total_row_count``, ``receipt``.
    """
    from .semantic_browser import semantic_table_query as _impl

    return _impl(public_html, dict(request))


def semantic_form_describe(
    public_html: str, request: Mapping[str, Any]
) -> dict[str, Any]:
    """Describe forms found in a public HTML page.

    This is a pure controller-side function.  It returns a bounded
    description of each form: method, action, fields and constraints.

    Parameters
    ----------
    public_html : str
        Current page HTML (public, no credentials).
    request : Mapping
        Fields: ``action`` ("describe"), ``form_index`` (int).

    Returns
    -------
    dict with ``forms`` list containing ``method``, ``action_url``,
    ``fields`` (list of {name, type, required, constraints}).
    """
    from .semantic_browser import semantic_form_describe as _impl

    return _impl(public_html, form_index=request.get("form_index", 0))


def semantic_form_submission(
    public_html: str,
    current_url: str,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and construct a GET form submission URL from public HTML.

    This is a pure controller-side function.  It validates the supplied
    fields against the form definition in public_html and builds the
    query-string URL.  Only GET forms are supported.

    Parameters
    ----------
    public_html : str
        Current page HTML (public, no credentials).
    request : Mapping
        Fields: ``action`` ("submit"), ``form_index`` (int),
        ``fields`` (list of {name, value}).

    Returns
    -------
    dict with ``submission_url``, ``validated_fields``, ``warnings``.
    """
    from .semantic_browser import semantic_form_submission as _impl

    fields = request.get("fields")
    if not isinstance(fields, list):
        raise ValueError("semantic_form submit requires 'fields' as a list")
    values: dict[str, str] = {}
    for field in fields:
        if not isinstance(field, Mapping):
            raise ValueError("each field in 'fields' must be an object")
        name = field.get("name")
        value = field.get("value")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("each field must have a non-empty 'name' string")
        if name in values:
            raise ValueError(f"duplicate field name: {name!r}")
        if not isinstance(value, str):
            raise ValueError(f"field {name!r} must have a string 'value'")
        values[name] = value
    return _impl(
        public_html,
        current_url,
        values,
        form_index=request.get("form_index", 0),
    )


def validate_smoke_url(url: str) -> str:
    """Accept only the exact, predeclared HTTPS smoke page."""

    if url != UNBROWSER_SMOKE_URL:
        raise ValueError(
            f"unbrowser is pinned to {UNBROWSER_SMOKE_URL!r}; got {url!r}"
        )
    return url


def validate_interactive_url(url: str, *, allow_fixture: bool = False) -> str:
    """Accept only a fixed Wikipedia URL with same-origin enforcement.

    The initial URL is controller-fixed to the Wikipedia main page.
    After click/submit, the final URL may change within the same origin.
    When ``allow_fixture=True``, fixture URLs on 127.0.0.1:18090 are
    also accepted and the origin check uses the fixture origin.
    This is explicitly NOT an SSRF defence.
    """

    if allow_fixture:
        if not url.startswith(FIXTURE_INTERACTIVE_ORIGIN):
            raise ValueError(
                f"unbrowser interactive fixture is pinned to {FIXTURE_INTERACTIVE_ORIGIN}; got {url!r}"
            )
        return url
    if not url.startswith(UNBROWSER_INTERACTIVE_ORIGIN):
        raise ValueError(
            f"unbrowser interactive is pinned to {UNBROWSER_INTERACTIVE_ORIGIN}; got {url!r}"
        )
    return url


class UnbrowserSession:
    """One isolated, fixed-page Unbrowser JSON-RPC session.

    In read-only mode (default), navigation is pinned to an exact URL and only
    non-mutating actions are allowed.  In interactive mode, click/type/submit
    are additionally available and same-origin Wikipedia URL changes are
    permitted after navigation.

    When ``confined=True`` the Unbrowser binary runs inside a Bubblewrap
    sandbox that retains network access but restricts filesystem visibility.
    """

    def __init__(
        self,
        binary: str,
        allowed_url: str,
        *,
        timeout_seconds: int = 30,
        max_result_bytes: int = DEFAULT_MAX_RESULT_BYTES,
        interactive: bool = False,
        confined: bool = False,
        required_first_observation: str | None = None,
        semantic_capability: str | None = None,
        semantic_fetch_timeout_seconds: int = 30,
    ) -> None:
        binary_path = Path(binary)
        if not binary_path.is_absolute():
            raise ValueError("unbrowser binary must be an absolute path")
        if timeout_seconds <= 0:
            raise ValueError("unbrowser timeout must be positive")
        if max_result_bytes <= 0:
            raise ValueError("unbrowser max result bytes must be positive")
        if required_first_observation not in _REQUIRED_OBSERVATION_CHOICES:
            raise ValueError(
                "required_first_observation must be None, 'text', or 'blockmap'"
            )
        if required_first_observation is not None and not interactive:
            raise ValueError(
                "required_first_observation is only allowed in interactive sessions"
            )
        if semantic_capability is not None and semantic_capability not in SEMANTIC_CAPABILITIES:
            raise ValueError(
                f"semantic_capability must be None or one of {sorted(SEMANTIC_CAPABILITIES)!r}"
            )
        if semantic_capability is not None:
            if not interactive:
                raise ValueError(
                    "semantic_capability requires an interactive session"
                )
            if not allowed_url.startswith(FIXTURE_URL_PREFIX):
                raise ValueError(
                    "semantic_capability requires a fixture interactive URL"
                )

        self.binary = str(binary_path)
        self.timeout_seconds = int(timeout_seconds)
        self.max_result_bytes = int(max_result_bytes)
        self.runtime_version: str | None = None
        self.interactive = bool(interactive)
        self.confined = bool(confined)
        self.required_first_observation: str | None = required_first_observation
        self.semantic_capability: str | None = semantic_capability
        self.semantic_fetch_timeout_seconds = int(semantic_fetch_timeout_seconds)

        if self.interactive:
            if allowed_url.startswith(FIXTURE_URL_PREFIX):
                self.allowed_url = validate_interactive_url(allowed_url, allow_fixture=True)
                self._interactive_origin = FIXTURE_INTERACTIVE_ORIGIN
            else:
                self.allowed_url = validate_interactive_url(allowed_url)
                self._interactive_origin = UNBROWSER_INTERACTIVE_ORIGIN
        else:
            self.allowed_url = validate_smoke_url(allowed_url)
            self._interactive_origin = ""  # Not used in read-only mode

        self._process: subprocess.Popen[bytes] | None = None
        self._temporary_home: tempfile.TemporaryDirectory[str] | None = None
        self._next_id = 0
        self._navigated = False
        self._first_direct_navigate_done = False
        self._stdout_buffer = bytearray()
        self._current_url: str | None = None

    @property
    def started(self) -> bool:
        return self._process is not None

    def _minimal_environment(self, temporary_home: str) -> dict[str, str]:
        binary_dir = str(Path(self.binary).parent)
        return {
            "HOME": temporary_home,
            "TMPDIR": temporary_home,
            "PATH": f"{binary_dir}:/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "UNBROWSER_TIMEOUT_MS": str(self.timeout_seconds * 1000),
        }

    def _start(self) -> None:
        if self._process is not None:
            return
        binary_path = Path(self.binary)
        if not binary_path.is_file() or not os.access(binary_path, os.X_OK):
            raise FileNotFoundError(f"unbrowser binary is not executable: {self.binary}")

        if self.confined:
            self._start_confined()
        else:
            self._start_unconfined()

    def _start_unconfined(self) -> None:
        """Launch the Unbrowser binary as a direct child process (current behaviour)."""
        temporary_home = tempfile.TemporaryDirectory(prefix="pyreplab-unbrowser-")
        environment = self._minimal_environment(temporary_home.name)
        try:
            version_check = subprocess.run(
                [self.binary, "--version"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=environment,
                check=False,
                timeout=min(self.timeout_seconds, 5),
            )
            version_text = version_check.stdout.decode(
                "utf-8", errors="replace"
            ).strip()
            if version_check.returncode != 0 or not version_text:
                raise UnbrowserProtocolError(
                    f"unbrowser --version failed (exit_code={version_check.returncode})",
                    infrastructure_error=True,
                )
            if len(version_text) > 128:
                raise UnbrowserProtocolError(
                    "unbrowser version output is oversized",
                    infrastructure_error=True,
                )
            self.runtime_version = version_text.removeprefix("unbrowser ")
            process = subprocess.Popen(
                [self.binary],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=environment,
                start_new_session=True,
            )
        except Exception:
            temporary_home.cleanup()
            raise

        self._temporary_home = temporary_home
        self._process = process

    def _start_confined(self) -> None:
        """Launch the Unbrowser binary inside a Bubblewrap sandbox.

        The sandbox retains network access but restricts filesystem visibility
        so the browser cannot read project source, run artifacts, SSH keys,
        model files, or the host user home directory.
        """
        from .unbrowser_sandbox import UnbrowserSandbox

        sandbox = UnbrowserSandbox(self.binary)

        # --- version probe inside the sandbox ---
        version_cmd = sandbox.build_command("--version")
        version_check = subprocess.run(
            version_cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=min(self.timeout_seconds + 5, 15),
        )
        version_text = version_check.stdout.decode("utf-8", errors="replace").strip()
        if version_check.returncode != 0 or not version_text:
            raise UnbrowserProtocolError(
                "unbrowser --version failed "
                f"(exit_code={version_check.returncode}, output={version_text[:128]!r})",
                infrastructure_error=True,
            )
        if len(version_text) > 128:
            raise UnbrowserProtocolError(
                "unbrowser version output is oversized",
                infrastructure_error=True,
            )
        self.runtime_version = version_text.removeprefix("unbrowser ")

        # --- launch unbrowser inside the sandbox ---
        launch_cmd = list(sandbox.build_command())
        # Inject UNBROWSER_TIMEOUT_MS before the final "--" separator.
        # build_command() always appends "--" followed by the binary + args.
        try:
            sep_index = launch_cmd.index("--")
        except ValueError:
            sep_index = -1
        launch_cmd[sep_index:sep_index] = [
            "--setenv",
            "UNBROWSER_TIMEOUT_MS",
            str(self.timeout_seconds * 1000),
        ]

        process = subprocess.Popen(
            launch_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        # No host-side _temporary_home is needed: the sandbox creates its own.
        self._process = process

    def _read_message(self, deadline: float) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdout is None:
            raise UnbrowserProtocolError("unbrowser stdout is unavailable")

        while b"\n" not in self._stdout_buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise UnbrowserProtocolError(
                    f"unbrowser response timed out after {self.timeout_seconds}s",
                    infrastructure_error=True,
                )
            readable, _, _ = select.select(
                [process.stdout.fileno()], [], [], remaining
            )
            if not readable:
                raise UnbrowserProtocolError(
                    f"unbrowser response timed out after {self.timeout_seconds}s",
                    infrastructure_error=True,
                )
            chunk = os.read(process.stdout.fileno(), 64 * 1024)
            if not chunk:
                code = process.poll()
                raise UnbrowserProtocolError(
                    f"unbrowser exited before replying (exit_code={code})",
                    infrastructure_error=True,
                )
            self._stdout_buffer.extend(chunk)
            if (
                len(self._stdout_buffer) > MAX_RPC_LINE_BYTES
                and b"\n" not in self._stdout_buffer
            ):
                raise UnbrowserProtocolError(
                    "unbrowser emitted an oversized JSON line"
                )

        line, _, remainder = self._stdout_buffer.partition(b"\n")
        self._stdout_buffer = bytearray(remainder)
        if len(line) > MAX_RPC_LINE_BYTES:
            raise UnbrowserProtocolError("unbrowser emitted an oversized JSON line")
        try:
            message = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise UnbrowserProtocolError("unbrowser emitted malformed JSON") from error
        if not isinstance(message, dict):
            raise UnbrowserProtocolError("unbrowser response must be a JSON object")
        return message

    def _request(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        try:
            self._start()
        except UnbrowserProtocolError:
            raise
        except (OSError, subprocess.SubprocessError) as error:
            raise UnbrowserProtocolError(
                f"unbrowser startup failed: {type(error).__name__}: {error}",
                infrastructure_error=True,
            ) from error
        process = self._process
        if process is None or process.stdin is None:
            raise UnbrowserProtocolError("unbrowser stdin is unavailable")

        self._next_id += 1
        request_id = self._next_id
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params is not None:
            payload["params"] = dict(params)

        # Check process health before writing — a pre-exited process would
        # otherwise surface as a raw BrokenPipeError.
        exit_code = process.poll()
        if exit_code is not None:
            raise UnbrowserProtocolError(
                f"unbrowser process exited before request (exit_code={exit_code})",
                infrastructure_error=True,
            )

        try:
            encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"
            try:
                process.stdin.write(encoded)
                process.stdin.flush()
            except (BrokenPipeError, ConnectionResetError) as exc:
                code = process.poll()
                raise UnbrowserProtocolError(
                    f"unbrowser process connection broken (exit_code={code})",
                    infrastructure_error=True,
                ) from exc
            deadline = time.monotonic() + self.timeout_seconds
            while True:
                message = self._read_message(deadline)
                if message.get("event") == "ready":
                    data = message.get("data")
                    if isinstance(data, dict) and isinstance(data.get("version"), str):
                        self.runtime_version = data["version"]
                    continue
                if message.get("id") != request_id:
                    # Unbrowser emits bounded lifecycle/diagnostic events on
                    # stdout before the matching JSON-RPC response.
                    continue
                if message.get("error") is not None:
                    raise UnbrowserProtocolError(
                        f"unbrowser {method} failed: {message['error']}"
                    )
                if "result" not in message:
                    raise UnbrowserProtocolError(
                        f"unbrowser {method} response omitted result"
                    )
                return message["result"]
        except Exception:
            self._kill()
            raise

    @staticmethod
    def _validate_selector(value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("selector must be a non-empty string")
        if len(value) > MAX_SELECTOR_CHARS:
            raise ValueError(
                f"selector must be at most {MAX_SELECTOR_CHARS} characters"
            )
        if "\x00" in value or "\n" in value or "\r" in value:
            raise ValueError("selector must not contain control-line characters")
        return value

    @staticmethod
    def _validate_ref(value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("ref must be a non-empty string")
        if len(value) > MAX_REF_CHARS:
            raise ValueError(f"ref must be at most {MAX_REF_CHARS} characters")
        if "\x00" in value or "\n" in value or "\r" in value:
            raise ValueError("ref must not contain control-line characters")
        return value

    @staticmethod
    def _validate_type_text(value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("type value must be a string")
        if len(value) > MAX_TYPE_TEXT_CHARS:
            raise ValueError(
                f"type value must be at most {MAX_TYPE_TEXT_CHARS} characters"
            )
        if "\x00" in value:
            raise ValueError("type value must not contain NUL bytes")
        return value

    def _check_navigate_result(self, result: Any) -> bool:
        """Validate the navigate result and enforce URL/origin rules.

        Returns ``True`` when navigation succeeded and further actions are
        allowed.  Returns ``False`` when status is non-200 or a challenge is
        present (interactive mode only).  Raises when the returned URL leaves
        the allowed origin.
        """
        if not isinstance(result, dict):
            raise UnbrowserProtocolError("navigate result must be an object")

        if self.interactive:
            status = result.get("status")
            challenge = result.get("challenge")
            if status != 200 or (challenge is not None and challenge):
                return False
            returned_url = result.get("url")
            origin = self._interactive_origin
            if not isinstance(returned_url, str) or not returned_url.startswith(origin):
                self._kill()
                raise UnbrowserProtocolError(
                    f"unbrowser left the allowed origin: {returned_url!r}"
                )
            return True

        returned_url = result.get("url")
        if returned_url != self.allowed_url:
            self._kill()
            raise UnbrowserProtocolError(
                f"unbrowser left the fixed page: {returned_url!r}"
            )
        return True

    def _check_interaction_result(self, action: str, result: Any) -> None:
        """Fail closed on explicit interaction errors and validate transitions."""

        if not isinstance(result, dict):
            return
        error = result.get("error")
        if result.get("ok") is False or error:
            detail = error if isinstance(error, str) and error else "operation failed"
            raise UnbrowserProtocolError(f"unbrowser {action} failed: {detail}")

        # Navigation-producing interactions return status/challenge metadata.
        # Non-navigation clicks may return only an element result, so retain the
        # current page state when those fields are absent.
        if "status" in result or "challenge" in result:
            self._navigated = self._check_navigate_result(result)
            if self._navigated and isinstance(result.get("url"), str):
                self._current_url = result["url"]
            return

        returned_url = result.get("url")
        origin = self._interactive_origin
        if isinstance(returned_url, str) and not returned_url.startswith(origin):
            self._kill()
            raise UnbrowserProtocolError(
                f"unbrowser left the allowed origin: {returned_url!r}"
            )
        if isinstance(returned_url, str) and returned_url:
            self._current_url = returned_url

    def _build_receipt(
        self, observation_payload: bytes, required_action: str, selector: str | None
    ) -> dict[str, Any]:
        """Build a deterministic receipt for an auto-delivered observation."""
        payload_sha256 = hashlib.sha256(observation_payload).hexdigest()
        return {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "mechanism": "auto_delivered_first_observation",
            "required_action": required_action,
            "delivered_action": required_action,
            "selector": selector,
            "delivered": True,
            "payload_bytes": len(observation_payload),
            "payload_sha256": payload_sha256,
        }

    def execute(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and execute one model-requested action.

        In read-only mode only ``navigate``, ``query``, ``text``, and
        ``blockmap`` are available.  In interactive mode ``click``, ``type``,
        and ``submit`` are additionally available.

        When ``required_first_observation`` is configured, the first successful
        direct navigate automatically runs the assigned observation before
        returning and includes a deterministic receipt.
        """

        receipt = None
        observation_result = None

        allowed_actions = INTERACTIVE_ACTIONS if self.interactive else READ_ONLY_ACTIONS
        interact_params = {"ref", "value"} if self.interactive else set()
        known = {"action", "selector"} | interact_params
        unknown = set(params) - known
        if unknown:
            raise ValueError(f"unknown unbrowser parameters: {sorted(unknown)!r}")
        action = params.get("action")
        if not isinstance(action, str) or action not in allowed_actions:
            raise ValueError(
                f"unbrowser action must be one of {sorted(allowed_actions)!r}"
            )

        selector = params.get("selector")
        if action == "navigate":
            if selector is not None:
                raise ValueError("navigate does not accept a selector")
            if self.interactive:
                if "ref" in params or "value" in params:
                    raise ValueError("navigate does not accept ref or value")
            result = self._request("navigate", {"url": self.allowed_url})
            navigated_ok = self._check_navigate_result(result)
            self._navigated = navigated_ok
            if navigated_ok:
                self._current_url = result.get("url")

            # ---- required-first-observation: auto-deliver on first direct navigate ----
            observation_result = None
            receipt = None
            is_first_direct = (
                not self._first_direct_navigate_done
                and self.required_first_observation is not None
                and navigated_ok
            )
            if navigated_ok:
                self._first_direct_navigate_done = True
            if is_first_direct:
                required_action = self.required_first_observation  # text or blockmap
                try:
                    if required_action == "text":
                        obs_selector = "body"
                        observation_result = self._request(
                            "text", {"selector": obs_selector}
                        )
                    else:  # blockmap
                        obs_selector = None
                        observation_result = self._request("blockmap")
                    obs_payload = json.dumps(
                        observation_result,
                        sort_keys=True,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    receipt = self._build_receipt(
                        obs_payload, required_action, obs_selector
                    )
                except Exception:
                    self._kill()
                    raise UnbrowserProtocolError(
                        f"required first observation ({required_action}) "
                        f"failed on initial navigate"
                    ) from None
            # --------------------------------------------------------------------------
        elif action in {"click", "submit"}:
            if not self._navigated:
                raise ValueError("navigate must succeed before other unbrowser actions")
            ref = params.get("ref")
            self._validate_ref(ref)
            if action == "click":
                result = self._request("click", {"ref": ref})
            else:
                result = self._request("submit", {"ref": ref})
            self._check_interaction_result(action, result)
        elif action == "type":
            if not self._navigated:
                raise ValueError("navigate must succeed before other unbrowser actions")
            ref = params.get("ref")
            value = params.get("value")
            self._validate_ref(ref)
            self._validate_type_text(value)
            result = self._request("type", {"ref": ref, "text": value})
            self._check_interaction_result(action, result)
        else:
            if not self._navigated:
                raise ValueError("navigate must succeed before other unbrowser actions")
            if action in {"query", "text"}:
                result = self._request(
                    action, {"selector": self._validate_selector(selector)}
                )
            else:  # blockmap
                if selector is not None:
                    raise ValueError("blockmap does not accept a selector")
                result = self._request("blockmap")

        wrapped: dict[str, Any] = {
            "action": action,
            "allowed_url": self.allowed_url,
            "runtime_version": self.runtime_version,
            "result": result,
        }
        if self.interactive:
            wrapped["interactive"] = True
        if receipt is not None:
            wrapped["required_first_observation_receipt"] = receipt
            wrapped["auto_delivered_observation"] = observation_result
        encoded = json.dumps(
            wrapped, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        if len(encoded) > self.max_result_bytes:
            raise UnbrowserProtocolError(
                f"unbrowser result exceeds {self.max_result_bytes} bytes"
            )
        return wrapped

    # ------------------------------------------------------------------
    # Semantic capability methods (controller-side, fixture-only)
    # ------------------------------------------------------------------

    def _fetch_public_html(self) -> str:
        """Fetch the current page's public HTML via urllib.

        No credentials, cookies, or private data are transmitted. The fetch
        uses the currently-tracked URL from the last successful navigation
        or navigation-producing click/submit.

        Returns the raw HTML as a string, capped at
        ``DEFAULT_SEMANTIC_FETCH_MAX_BYTES``.
        """
        if self._current_url is None:
            raise UnbrowserProtocolError(
                "semantic capability requires a successful navigation first"
            )
        try:
            request = urllib.request.Request(
                self._current_url,
                headers={"Accept": "text/html"},
                method="GET",
            )
            with urllib.request.urlopen(
                request, timeout=self.semantic_fetch_timeout_seconds
            ) as response:
                final_url = response.geturl()
                if not final_url.startswith(self._interactive_origin):
                    raise UnbrowserProtocolError(
                        f"fetch redirected off origin: {final_url!r}",
                        infrastructure_error=True,
                    )
                raw = response.read(DEFAULT_SEMANTIC_FETCH_MAX_BYTES + 1)
                if len(raw) > DEFAULT_SEMANTIC_FETCH_MAX_BYTES:
                    raise UnbrowserProtocolError(
                        f"fetched HTML exceeds {DEFAULT_SEMANTIC_FETCH_MAX_BYTES} bytes",
                        infrastructure_error=True,
                    )
                return raw.decode(response.headers.get_content_charset("utf-8"))
        except UnbrowserProtocolError:
            raise
        except Exception as error:
            raise UnbrowserProtocolError(
                f"controller fetch failed: {type(error).__name__}: {error}",
                infrastructure_error=True,
            ) from error

    def _semantic_capability_allowed(self) -> None:
        """Fail closed if semantic capability is not enabled."""
        if self.semantic_capability is None:
            raise UnbrowserProtocolError(
                "semantic capability is not enabled for this session"
            )
        if not self._navigated:
            raise UnbrowserProtocolError(
                "semantic capability requires a successful navigation first"
            )

    def _build_semantic_receipt(
        self, capability: str, action: str, payload: Any
    ) -> dict[str, Any]:
        """Build a deterministic receipt for semantic capability results."""
        encoded = json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        return {
            "schema_version": SEMANTIC_RECEIPT_SCHEMA_VERSION,
            "capability": capability,
            "specialist": f"{capability}_specialist",
            "action": action,
            "delivered": True,
            "payload_bytes": len(encoded),
            "payload_sha256": hashlib.sha256(encoded).hexdigest(),
        }

    def _wrap_semantic_payload(
        self, capability: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        action = f"semantic_{capability}"
        receipt = self._build_semantic_receipt(capability, action, payload)
        wrapped = {
            "semantic_payload": dict(payload),
            "semantic_specialist_receipt": receipt,
            "infrastructure_error": False,
        }
        encoded = json.dumps(
            wrapped, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        if len(encoded) > self.max_result_bytes:
            raise UnbrowserProtocolError(
                f"semantic {capability} result exceeds {self.max_result_bytes} bytes",
                infrastructure_error=True,
            )
        return wrapped

    def semantic_table(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Controller-side semantic table query on the current public HTML.

        Parameters
        ----------
        params : Mapping
            ``table_index`` (int, optional), ``filters`` (list),
            ``sort`` ({column,direction}), ``offset`` (int), ``limit`` (int),
            ``projection`` (list of str).

        Returns
        -------
        dict with ``columns``, ``rows``, ``total_row_count``,
        ``receipt`` (deterministic), and ``infrastructure_error`` marker.
        """
        if self.semantic_capability != "table":
            raise UnbrowserProtocolError(
                "semantic_table is not enabled; current capability is "
                f"{self.semantic_capability!r}"
            )
        self._semantic_capability_allowed()
        public_html = self._fetch_public_html()
        result = semantic_table_query(public_html, params)
        return self._wrap_semantic_payload("table", result)

    def semantic_form(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Controller-side semantic form describe or submit.

        Parameters
        ----------
        params : Mapping
            ``action`` ("describe" or "submit"), ``form_index`` (int, optional),
            ``fields`` (list of {name,value}, required for submit).

        Returns
        -------
        dict with describe/submit results, ``receipt``, navigation metadata,
        ``infrastructure_error`` marker.

        For submit, the controller constructs the GET query-string URL and
        navigates the existing Unbrowser process to it, then reads text.
        """
        if self.semantic_capability != "form":
            raise UnbrowserProtocolError(
                "semantic_form is not enabled; current capability is "
                f"{self.semantic_capability!r}"
            )
        self._semantic_capability_allowed()
        action = params.get("action")
        if action not in _DESCRIBE_ACTIONS:
            raise ValueError(
                f"semantic_form action must be 'describe' or 'submit'; got {action!r}"
            )
        public_html = self._fetch_public_html()
        if action == "describe":
            result = semantic_form_describe(public_html, params)
            return self._wrap_semantic_payload("form", result)
        return self._semantic_form_submit_impl(public_html, params)

    def _semantic_form_submit_impl(
        self, public_html: str, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Implement submit: validate, build URL, navigate Unbrowser, read body."""
        current_url = self._current_url
        if current_url is None:
            raise UnbrowserProtocolError(
                "semantic form submission requires a current URL",
                infrastructure_error=True,
            )
        submission_result = semantic_form_submission(
            public_html, current_url, params
        )
        submission_url = submission_result.get("url")
        if not isinstance(submission_url, str) or not submission_url:
            raise UnbrowserProtocolError(
                "semantic form submission did not produce a valid URL"
            )
        # Only same-origin GET URLs are accepted.
        if not submission_url.startswith(self._interactive_origin):
            raise UnbrowserProtocolError(
                f"submission URL must stay within origin: {submission_url!r}"
            )
        if "?" not in submission_url:
            raise UnbrowserProtocolError(
                "semantic form submission requires a GET query-string URL"
            )

        # Navigate the Unbrowser process to the constructed URL.
        navigate_result = self._request("navigate", {"url": submission_url})
        navigated_ok = self._check_navigate_result(navigate_result)
        if not navigated_ok:
            raise UnbrowserProtocolError(
                "semantic form submission navigate failed"
            )
        self._navigated = True
        self._current_url = navigate_result.get("url", submission_url)

        # Auto-read body text after navigation.
        body_text = self._request("text", {"selector": "body"})

        payload: dict[str, Any] = {
            "operation": "submit",
            "submission": submission_result,
            "submission_url": submission_url,
            "navigate_result": {
                "status": navigate_result.get("status"),
                "url": navigate_result.get("url"),
            },
            "body_text": body_text,
        }
        return self._wrap_semantic_payload("form", payload)

    def _kill(self) -> None:
        process = self._process
        self._process = None
        self._navigated = False
        self._first_direct_navigate_done = False
        self._current_url = None
        self._stdout_buffer.clear()
        if process is not None:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=1)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        pass
            if process.stdin is not None:
                process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()

        temporary_home = self._temporary_home
        self._temporary_home = None
        if temporary_home is not None:
            temporary_home.cleanup()

    def close(self) -> None:
        """Close the child gracefully when possible, then kill its process group."""

        if self._process is None:
            return
        try:
            self._request("close")
        except Exception:
            # _request already killed the process on protocol/timeout failures.
            pass
        finally:
            self._kill()

    def __enter__(self) -> "UnbrowserSession":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
