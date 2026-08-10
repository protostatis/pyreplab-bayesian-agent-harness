"""Bubblewrap confinement profile for the Unbrowser child process.

Retains outbound network access but restricts filesystem so the browser
cannot read project source, run artifacts, SSH keys, model files, or the
host user home directory.
"""

from __future__ import annotations

import os
from pathlib import Path

# Only library and runtime directories — not entire /usr which would pull in
# /usr/local with the Unbrowser binary, /usr/local/bin/pi, etc.
_UNBROWSER_LIB_PATHS = ("/usr/lib", "/lib", "/lib64")

# DNS and TLS — the minimum needed for HTTPS outbound.
_UNBROWSER_CONFIG_PATHS = (
    "/etc/resolv.conf",
    "/etc/ssl",
)


class UnbrowserSandbox:
    """Bubblewrap confinement profile for the Unbrowser child process.

    Retains network access but restricts filesystem to:
    - the Unbrowser binary itself (read-only bind)
    - minimal runtime libraries (read-only)
    - a temporary HOME directory (writable, ephemeral)
    - /tmp (writable, ephemeral)
    - /dev/null, /dev/urandom (via fresh /dev)
    - /etc/resolv.conf (read-only, for DNS)

    Does NOT mount:
    - project source code
    - run artifacts or datasets
    - SSH keys or agent sockets
    - model files or Pi installation
    - user home directory
    - /home
    """

    def __init__(self, binary: str, *, command_timeout: int = 30) -> None:
        """Configure the confinement.

        Args:
            binary: Absolute path to the Unbrowser executable.  Must exist and
                be executable at construction time.
            command_timeout: Maximum wall-clock seconds the confined process may
                live (defence-in-depth via GNU ``timeout``).
        """
        binary_path = Path(binary)
        if not binary_path.is_absolute():
            raise ValueError("unbrowser binary must be an absolute path")
        if not binary_path.is_file() or not os.access(binary_path, os.X_OK):
            raise FileNotFoundError(
                f"unbrowser binary is not executable: {binary}"
            )
        if command_timeout <= 0:
            raise ValueError("command_timeout must be positive")

        self.binary = str(binary_path)
        self.command_timeout = int(command_timeout)

    def build_command(self, *args: str) -> list[str]:
        """Return the full ``bwrap`` command line that launches unbrowser.

        The returned list is suitable for ``subprocess.Popen`` /
        ``subprocess.run``.  The caller should add any extra environment
        variables (e.g. ``UNBROWSER_TIMEOUT_MS``) by inserting additional
        ``--setenv`` flags before the ``--`` separator.

        Network namespace is **not** unshared so the browser can reach fixture
        pages and Wikipedia over HTTPS.
        """
        bwrap_cmd = [
            "bwrap",
            # New namespaces — but NOT --unshare-net.
            "--unshare-user",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-mount",
            "--die-with-parent",
            "--new-session",
            "--clearenv",
            # Fresh kernel interfaces.
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            # Temporary writable HOME inside the sandbox (not the host).
            "--dir",
            "/home/unbrowser",
            # Minimal environment.
            "--setenv",
            "HOME",
            "/home/unbrowser",
            "--setenv",
            "TMPDIR",
            "/tmp",
            "--setenv",
            "PATH",
            "/usr/bin:/bin",
            "--setenv",
            "LANG",
            "C.UTF-8",
        ]

        # Bind the Unbrowser binary itself read-only.
        bwrap_cmd.extend(["--ro-bind", self.binary, self.binary])

        # Bind runtime library directories (read-only).
        for path in _UNBROWSER_LIB_PATHS:
            if os.path.exists(path):
                bwrap_cmd.extend(["--ro-bind", path, path])

        # Bind DNS and TLS configuration (read-only, only when present).
        for path in _UNBROWSER_CONFIG_PATHS:
            if os.path.exists(path):
                bwrap_cmd.extend(["--ro-bind", path, path])

        # The binary to execute inside the sandbox.
        bwrap_cmd.append("--")
        bwrap_cmd.append(self.binary)
        bwrap_cmd.extend(args)

        # Wrap with GNU timeout for defence-in-depth: if the sandbox hangs the
        # kernel will eventually deliver SIGKILL even when --die-with-parent
        # races with a stuck mount namespace.
        return [
            "timeout",
            "--kill-after=5s",
            f"{self.command_timeout}s",
            *bwrap_cmd,
        ]

    def canary_paths(self) -> list[str]:
        """Return paths the confined process must NOT be able to read.

        Used for filesystem canary tests that verify the sandbox denies access
        to sensitive host directories.
        """
        home = os.path.expanduser("~")
        paths = [
            home,
            "/home",
            "/root",
            os.getcwd(),
        ]
        # Include typical SSH and model directories if they exist.
        ssh_dir = os.path.join(home, ".ssh")
        if os.path.exists(ssh_dir):
            paths.append(ssh_dir)
        # Exclude paths that overlap with the runtime library mounts so the
        # canary list stays meaningful: a path under /usr/lib, /lib, or /lib64
        # might be reachable through the read-only library binds and would
        # produce a false positive.
        return [
            p
            for p in paths
            if not any(
                p.startswith(prefix.rstrip("/") + "/") or p == prefix.rstrip("/")
                for prefix in _UNBROWSER_LIB_PATHS
            )
        ]
