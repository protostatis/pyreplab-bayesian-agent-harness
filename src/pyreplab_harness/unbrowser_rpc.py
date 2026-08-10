"""Strict read-only adapter for the live Unbrowser smoke test.

The model never supplies a URL or raw JSON-RPC payload.  This adapter owns one
fresh ``unbrowser`` process, pins navigation to the fixed public smoke page,
and exposes only four non-mutating actions.  It intentionally runs outside the
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
READ_ONLY_ACTIONS = frozenset({"navigate", "query", "text", "blockmap"})
MAX_SELECTOR_CHARS = 256
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


class UnbrowserSession:
    """One isolated, fixed-page Unbrowser JSON-RPC session."""

    def __init__(
        self,
        binary: str,
        allowed_url: str,
        *,
        timeout_seconds: int = 30,
        max_result_bytes: int = DEFAULT_MAX_RESULT_BYTES,
    ) -> None:
        binary_path = Path(binary)
        if not binary_path.is_absolute():
            raise ValueError("unbrowser binary must be an absolute path")
        if timeout_seconds <= 0:
            raise ValueError("unbrowser timeout must be positive")
        if max_result_bytes <= 0:
            raise ValueError("unbrowser max result bytes must be positive")

        self.binary = str(binary_path)
        self.allowed_url = validate_smoke_url(allowed_url)
        self.timeout_seconds = int(timeout_seconds)
        self.max_result_bytes = int(max_result_bytes)
        self.runtime_version: str | None = None

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

    def execute(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and execute one model-requested read-only action."""

        unknown = set(params) - {"action", "selector"}
        if unknown:
            raise ValueError(f"unknown unbrowser parameters: {sorted(unknown)!r}")
        action = params.get("action")
        if not isinstance(action, str) or action not in READ_ONLY_ACTIONS:
            raise ValueError(
                f"unbrowser action must be one of {sorted(READ_ONLY_ACTIONS)!r}"
            )

        selector = params.get("selector")
        if action == "navigate":
            if selector is not None:
                raise ValueError("navigate does not accept a selector")
            result = self._request("navigate", {"url": self.allowed_url})
            if not isinstance(result, dict):
                raise UnbrowserProtocolError("navigate result must be an object")
            returned_url = result.get("url")
            if returned_url != self.allowed_url:
                self._kill()
                raise UnbrowserProtocolError(
                    f"unbrowser left the fixed page: {returned_url!r}"
                )
            self._navigated = True
        else:
            if not self._navigated:
                raise ValueError("navigate must succeed before other unbrowser actions")
            if action in {"query", "text"}:
                result = self._request(
                    action, {"selector": self._validate_selector(selector)}
                )
            else:
                if selector is not None:
                    raise ValueError("blockmap does not accept a selector")
                result = self._request("blockmap")

        wrapped = {
            "action": action,
            "allowed_url": self.allowed_url,
            "runtime_version": self.runtime_version,
            "result": result,
        }
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
