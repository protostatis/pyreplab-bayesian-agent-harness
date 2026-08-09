from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

_RUNTIME_PATHS = ("/usr", "/bin", "/lib", "/lib64")
# On some hosts /usr/local is a separate mount that a ``--ro-bind /usr`` does
# not cover, so the verifier (isolated) layout binds it explicitly. This is
# also the set of roots a sandbox-reachable interpreter may live under.
ISOLATED_RUNTIME_PATHS: tuple[str, ...] = _RUNTIME_PATHS + ("/usr/local",)
_UNIT_RE = re.compile(rb"Running as unit: (\S+)")


class SandboxUnavailableError(RuntimeError):
    """Raised when the Bubblewrap/systemd-run backend cannot be started.

    Callers that must never fall back to host execution (for example the
    python-repair verifier) should map this to a distinct failure code instead
    of running the submitted code without OS-level containment.
    """


@dataclass(frozen=True)
class SandboxLimits:
    max_timeout_seconds: int = 30
    memory_max: str = "1G"
    tasks_max: int = 64
    cpu_quota: str = "200%"
    output_limit_bytes: int = 64 * 1024

    def __post_init__(self) -> None:
        if not 1 <= self.max_timeout_seconds <= 600:
            raise ValueError("max_timeout_seconds must be between 1 and 600")
        if not 1 <= self.tasks_max <= 1024:
            raise ValueError("tasks_max must be between 1 and 1024")
        if not re.fullmatch(r"[1-9][0-9]*(?:[KMG])?", self.memory_max):
            raise ValueError("memory_max must look like 512M or 1G")
        if not re.fullmatch(r"[1-9][0-9]*%", self.cpu_quota):
            raise ValueError("cpu_quota must look like 100% or 200%")


@dataclass(frozen=True)
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool
    truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class SandboxBind:
    """A host directory mounted into the sandbox.

    ``target`` is the absolute in-sandbox path. ``writable=False`` mounts the
    source read-only, which is what the verifier wants for the submitted
    workspace and the private bundle.
    """

    source: Path | str
    target: str
    writable: bool = False


def resolve_workspace(root: str | Path, workspace: str | Path) -> tuple[Path, Path]:
    root_path = Path(root).expanduser().resolve(strict=True)
    workspace_path = Path(workspace).expanduser().resolve(strict=True)
    if not root_path.is_dir() or not workspace_path.is_dir():
        raise ValueError("root and workspace must be directories")
    if workspace_path == root_path or root_path not in workspace_path.parents:
        raise ValueError("workspace must be a strict descendant of root")
    if workspace_path.name != "workspace" or workspace_path.parent.parent.name != "attempts":
        raise ValueError("workspace must use the expected attempts/<id>/workspace layout")
    return root_path, workspace_path


def bwrap_available() -> bool:
    """True when ``bwrap`` exists *and* can actually create an isolated sandbox
    (i.e. unprivileged user namespaces work on this host)."""
    if shutil.which("bwrap") is None:
        return False
    probe = [
        "bwrap",
        "--unshare-all",
        "--new-session",
        "--die-with-parent",
        "--clearenv",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
    ]
    for path in ISOLATED_RUNTIME_PATHS:
        if os.path.exists(path):
            probe.extend(["--ro-bind", path, path])
    probe.append("/bin/true")
    try:
        proc = subprocess.run(
            probe,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def systemd_user_available() -> bool:
    """True when ``systemd-run --user`` can create transient units (the user
    systemd manager is running)."""
    if shutil.which("systemd-run") is None or shutil.which("systemctl") is None:
        return False
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    if Path(runtime_dir, "systemd", "private").exists():
        return True
    try:
        proc = subprocess.run(
            ["systemctl", "--user", "is-system-running"],
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    # "running" is 0, "degraded" is 1; both still accept new units.
    return proc.returncode in (0, 1)


def sandbox_available() -> bool:
    return bwrap_available() and systemd_user_available()


def _probe_python_version(binary: str) -> str | None:
    try:
        proc = subprocess.run(
            [binary, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def sandbox_python_interpreter(preferred_version: str | None = None) -> str | None:
    """Return a canonical absolute host Python interpreter path that is
    reachable inside the isolated sandbox (under one of the read-only runtime
    mounts).

    When ``preferred_version`` (e.g. ``"3.11"``) is given, only candidates
    whose probed major.minor version matches are considered, so the sandboxed
    interpreter matches the harness interpreter. The returned path is the
    fully resolved ``realpath``: a symlink such as ``/usr/bin/python3`` that
    hops through ``/etc/alternatives/python3`` would not resolve inside the
    sandbox because ``/etc`` is not mounted, so the canonical target (e.g.
    ``/usr/bin/python3.10``) is returned instead. The path is verified to
    exist and be executable, and its real target must stay inside the runtime
    mounts.
    """
    candidates = ["/usr/bin/python3", "/usr/local/bin/python3"]
    if preferred_version:
        candidates = [
            f"/usr/bin/python{preferred_version}",
            f"/usr/local/bin/python{preferred_version}",
            *candidates,
        ]
    for candidate in candidates:
        if not os.path.exists(candidate):
            continue
        if preferred_version is not None and _probe_python_version(candidate) != preferred_version:
            continue
        resolved = os.path.realpath(candidate)
        if resolved.startswith(ISOLATED_RUNTIME_PATHS) and os.access(resolved, os.X_OK):
            return resolved
    return None


def _clamp_timeout(timeout_seconds: int | None, limits: SandboxLimits) -> int:
    return max(
        1,
        min(int(timeout_seconds or limits.max_timeout_seconds), limits.max_timeout_seconds),
    )


def _truncate(value: bytes, limit: int) -> tuple[bytes, bool]:
    if len(value) <= limit:
        return value, False
    marker = b"\n...[output truncated]...\n"
    side = max(1, (limit - len(marker)) // 2)
    return value[:side] + marker + value[-side:], True


def _parse_unit_name(stderr: bytes) -> str | None:
    match = _UNIT_RE.search(stderr)
    return match.group(1).decode("utf-8", "replace") if match else None


def _strip_unit_line(stderr: bytes) -> bytes:
    return b"\n".join(line for line in stderr.split(b"\n") if not _UNIT_RE.match(line))


def _kill_after_timeout(process: subprocess.Popen[Any], unit_name: str | None) -> None:
    """Kill the sandbox tree on timeout: the process group, the systemd-run
    client, and the transient unit's cgroup (``systemctl --user kill``). The
    unit's ``RuntimeMaxSec`` property is the final backstop."""
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass
    try:
        os.kill(process.pid, signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass
    if unit_name:
        try:
            subprocess.run(
                ["systemctl", "--user", "kill", unit_name, "--signal=SIGKILL"],
                capture_output=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass


def _run(
    argv: list[str], timeout_seconds: int
) -> tuple[bytes, bytes, int | None, bool]:
    """Run ``systemd-run ... bwrap ...`` capturing stdout/stderr.

    Returns ``(stdout, stderr, exit_code, timed_out)``. On timeout the process
    group, the systemd-run client and the transient unit are killed so no
    sandboxed process survives verification.
    """
    try:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as error:
        raise SandboxUnavailableError(f"cannot start sandbox subprocess: {error}") from error
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        timed_out = True
        partial_out = error.stdout or b""
        partial_err = error.stderr or b""
        unit_name = _parse_unit_name(partial_err)
        _kill_after_timeout(process, unit_name)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            stdout, stderr = partial_out, partial_err
    stderr = _strip_unit_line(stderr)
    return stdout, stderr, (None if timed_out else process.returncode), timed_out


def _finalize(
    stdout: bytes,
    stderr: bytes,
    exit_code: int | None,
    timed_out: bool,
    output_limit: int,
) -> SandboxResult:
    stdout, stdout_truncated = _truncate(stdout, output_limit)
    stderr, stderr_truncated = _truncate(stderr, output_limit)
    return SandboxResult(
        stdout=stdout.decode("utf-8", errors="replace"),
        stderr=stderr.decode("utf-8", errors="replace"),
        exit_code=exit_code,
        timed_out=timed_out,
        truncated=stdout_truncated or stderr_truncated,
    )


class BubblewrapSandbox:
    def __init__(
        self,
        root: str | Path,
        workspace: str | Path,
        limits: SandboxLimits | None = None,
    ) -> None:
        self.root, self.workspace = resolve_workspace(root, workspace)
        self.limits = limits or SandboxLimits()

    def _systemd_run(self, timeout: int, bwrap_args: list[str], *, quiet: bool) -> list[str]:
        argv = ["systemd-run", "--user"]
        if quiet:
            argv.append("--quiet")
        argv.extend(
            [
                "--wait",
                "--collect",
                "--pipe",
                f"--property=MemoryMax={self.limits.memory_max}",
                f"--property=TasksMax={self.limits.tasks_max}",
                f"--property=CPUQuota={self.limits.cpu_quota}",
                f"--property=RuntimeMaxSec={timeout + 2}s",
                *bwrap_args,
            ]
        )
        return argv

    def build_command(self, command: str, timeout_seconds: int) -> list[str]:
        """Worker-style layout: the workspace is writable at ``/workspace``.

        Output (option order, defaults, behavior) is unchanged from the
        original remote-worker command so the existing protocol keeps working.
        """
        timeout = _clamp_timeout(timeout_seconds, self.limits)
        bwrap = [
            "bwrap",
            "--unshare-all",
            "--new-session",
            "--die-with-parent",
            "--clearenv",
        ]
        for path in _RUNTIME_PATHS:
            if os.path.exists(path):
                bwrap.extend(["--ro-bind", path, path])
        bwrap.extend(
            [
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--tmpfs",
                "/tmp",
                "--dir",
                "/workspace",
                "--bind",
                str(self.workspace),
                "/workspace",
                "--chdir",
                "/workspace",
                "--setenv",
                "HOME",
                "/tmp",
                "--setenv",
                "PATH",
                "/usr/local/bin:/usr/bin:/bin",
                "--setenv",
                "LANG",
                "C.UTF-8",
                "--setenv",
                "PYTHONNOUSERSITE",
                "1",
                "/bin/bash",
                "--noprofile",
                "--norc",
                "-lc",
                command,
            ]
        )
        return self._systemd_run(timeout, bwrap, quiet=True)

    def build_isolated_command(
        self,
        command: str,
        timeout_seconds: int,
        *,
        read_only: Sequence[SandboxBind] = (),
        writable: Sequence[SandboxBind] = (),
        chdir: str = "/workspace",
        extra_env: dict[str, str] | None = None,
    ) -> list[str]:
        """Verifier-style layout: only the explicitly listed directories exist
        besides the runtime paths, ``/proc``, ``/dev`` and a fresh ``/tmp``.

        ``read_only`` binds are mounted read-only (submitted workspace, private
        bundle); ``writable`` binds are the throwaway output directory. There is
        no ``/home`` mount and the environment is cleared then rebuilt from an
        explicit allowlist.
        """
        timeout = _clamp_timeout(timeout_seconds, self.limits)
        bwrap = [
            "bwrap",
            "--unshare-all",
            "--new-session",
            "--die-with-parent",
            "--clearenv",
        ]
        for path in ISOLATED_RUNTIME_PATHS:
            if os.path.exists(path):
                bwrap.extend(["--ro-bind", path, path])
        for bind in read_only:
            source = str(Path(bind.source).expanduser().resolve())
            bwrap.extend(["--ro-bind", source, bind.target])
        for bind in writable:
            source = str(Path(bind.source).expanduser().resolve())
            bwrap.extend(["--bind", source, bind.target])
        bwrap.extend(["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"])
        env = {
            "HOME": "/tmp",
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "PYTHONNOUSERSITE": "1",
        }
        env.update(extra_env or {})
        for key, value in env.items():
            bwrap.extend(["--setenv", key, value])
        bwrap.extend(["--chdir", chdir, "--", "/bin/bash", "--noprofile", "--norc", "-lc", command])
        # quiet=False so the transient unit name can be parsed for a prompt
        # ``systemctl --user kill`` on timeout; the line is stripped again on
        # the way out by ``_run``.
        return self._systemd_run(timeout, bwrap, quiet=False)

    def execute(self, command: str, timeout_seconds: int | None = None) -> SandboxResult:
        if not isinstance(command, str) or not command.strip():
            raise ValueError("command must be a non-empty string")
        timeout = _clamp_timeout(timeout_seconds, self.limits)
        stdout, stderr, exit_code, timed_out = _run(
            self.build_command(command, timeout), timeout + 5
        )
        return _finalize(
            stdout, stderr, exit_code, timed_out, self.limits.output_limit_bytes
        )

    def execute_isolated(
        self,
        command: str,
        timeout_seconds: int,
        *,
        read_only: Sequence[SandboxBind] = (),
        writable: Sequence[SandboxBind] = (),
        chdir: str = "/workspace",
        extra_env: dict[str, str] | None = None,
    ) -> SandboxResult:
        timeout = _clamp_timeout(timeout_seconds, self.limits)
        argv = self.build_isolated_command(
            command,
            timeout,
            read_only=read_only,
            writable=writable,
            chdir=chdir,
            extra_env=extra_env,
        )
        stdout, stderr, exit_code, timed_out = _run(argv, timeout)
        return _finalize(
            stdout, stderr, exit_code, timed_out, self.limits.output_limit_bytes
        )
