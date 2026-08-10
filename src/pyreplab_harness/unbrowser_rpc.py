"""Strict read-only and interactive adapters for the live Unbrowser smoke test.

The model never supplies a URL or raw JSON-RPC payload.  This adapter owns one
fresh ``unbrowser`` process, pins navigation to the fixed public smoke page
(read-only) or Wikipedia (interactive), and exposes non-mutating actions plus
click/type/submit for the interactive path.  It intentionally runs outside the
Bubblewrap command sandbox, so this module is the network security boundary.
"""

from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import tempfile
import time
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
MAX_SELECTOR_CHARS = 256
MAX_REF_CHARS = 256
MAX_TYPE_TEXT_CHARS = 1024
MAX_RPC_LINE_BYTES = 256 * 1024
DEFAULT_MAX_RESULT_BYTES = 64 * 1024


class UnbrowserProtocolError(RuntimeError):
    """Raised when the child violates the bounded JSON-RPC protocol."""


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
    ) -> None:
        binary_path = Path(binary)
        if not binary_path.is_absolute():
            raise ValueError("unbrowser binary must be an absolute path")
        if timeout_seconds <= 0:
            raise ValueError("unbrowser timeout must be positive")
        if max_result_bytes <= 0:
            raise ValueError("unbrowser max result bytes must be positive")

        self.binary = str(binary_path)
        self.timeout_seconds = int(timeout_seconds)
        self.max_result_bytes = int(max_result_bytes)
        self.runtime_version: str | None = None
        self.interactive = bool(interactive)
        self.confined = bool(confined)

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
        self._stdout_buffer = bytearray()

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
                    f"unbrowser --version failed (exit_code={version_check.returncode})"
                )
            if len(version_text) > 128:
                raise UnbrowserProtocolError("unbrowser version output is oversized")
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

        sandbox = UnbrowserSandbox(self.binary, command_timeout=self.timeout_seconds)

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
                f"(exit_code={version_check.returncode}, output={version_text[:128]!r})"
            )
        if len(version_text) > 128:
            raise UnbrowserProtocolError("unbrowser version output is oversized")
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
                raise TimeoutError(
                    f"unbrowser response timed out after {self.timeout_seconds}s"
                )
            readable, _, _ = select.select(
                [process.stdout.fileno()], [], [], remaining
            )
            if not readable:
                raise TimeoutError(
                    f"unbrowser response timed out after {self.timeout_seconds}s"
                )
            chunk = os.read(process.stdout.fileno(), 64 * 1024)
            if not chunk:
                code = process.poll()
                raise UnbrowserProtocolError(
                    f"unbrowser exited before replying (exit_code={code})"
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
        self._start()
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

        try:
            encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"
            process.stdin.write(encoded)
            process.stdin.flush()
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
            return

        returned_url = result.get("url")
        origin = self._interactive_origin
        if isinstance(returned_url, str) and not returned_url.startswith(origin):
            self._kill()
            raise UnbrowserProtocolError(
                f"unbrowser left the allowed origin: {returned_url!r}"
            )

    def execute(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and execute one model-requested action.

        In read-only mode only ``navigate``, ``query``, ``text``, and
        ``blockmap`` are available.  In interactive mode ``click``, ``type``,
        and ``submit`` are additionally available.
        """

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
            self._navigated = self._check_navigate_result(result)
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
        encoded = json.dumps(
            wrapped, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        if len(encoded) > self.max_result_bytes:
            self._kill()
            raise UnbrowserProtocolError(
                f"unbrowser result exceeds {self.max_result_bytes} bytes"
            )
        return wrapped

    def _kill(self) -> None:
        process = self._process
        self._process = None
        self._navigated = False
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
