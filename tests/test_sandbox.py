from __future__ import annotations

import json
import os
import socket
import tempfile
import unittest
from pathlib import Path

from pyreplab_harness.sandbox import (
    BubblewrapSandbox,
    SandboxBind,
    SandboxLimits,
    bwrap_available,
    resolve_workspace,
    sandbox_available,
    sandbox_python_interpreter,
    systemd_user_available,
)


class SandboxTest(unittest.TestCase):
    def _workspace(self, root: Path) -> Path:
        workspace = root / "attempts" / "a1" / "workspace"
        workspace.mkdir(parents=True)
        return workspace

    def test_workspace_must_follow_attempt_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wrong = root / "workspace"
            wrong.mkdir()
            with self.assertRaisesRegex(ValueError, "expected"):
                resolve_workspace(root, wrong)

    def test_workspace_cannot_escape_root_through_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            attempt = root / "attempts" / "a1"
            attempt.mkdir(parents=True)
            (attempt / "workspace").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "descendant"):
                resolve_workspace(root, attempt / "workspace")

    def test_command_uses_network_and_mount_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = self._workspace(root)
            sandbox = BubblewrapSandbox(root, workspace, SandboxLimits(max_timeout_seconds=12))
            command = sandbox.build_command("pwd", 50)
            joined = " ".join(command)
            self.assertIn("--unshare-all", command)
            self.assertIn("--clearenv", command)
            self.assertIn(str(workspace.resolve()), command)
            self.assertIn("/workspace", command)
            self.assertNotIn(str(Path.home()), joined)
            self.assertIn("--property=RuntimeMaxSec=14s", command)

    def test_isolated_command_mounts_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as extra:
            root = Path(directory)
            workspace = self._workspace(root)
            private = Path(extra) / "private"
            output = Path(extra) / "output"
            private.mkdir()
            output.mkdir()
            sandbox = BubblewrapSandbox(root, workspace, SandboxLimits(max_timeout_seconds=12))
            command = sandbox.build_isolated_command(
                "pwd",
                50,
                read_only=[SandboxBind(workspace, "/workspace"), SandboxBind(private, "/private")],
                writable=[SandboxBind(output, "/output")],
                extra_env={"PYTHONDONTWRITEBYTECODE": "1"},
            )
            joined = " ".join(command)
            # Namespace + environment isolation.
            self.assertIn("--unshare-all", command)
            self.assertIn("--clearenv", command)
            self.assertIn("--tmpfs /tmp", joined)
            # Workspace and private bundle are read-only.
            self.assertIn(f"--ro-bind {workspace.resolve()} /workspace", joined)
            self.assertIn(f"--ro-bind {private.resolve()} /private", joined)
            self.assertNotIn(f"--bind {workspace.resolve()} /workspace", joined)
            # Output directory is the only writable bind.
            self.assertIn(f"--bind {output.resolve()} /output", joined)
            # No /home mount and HOME points at the throwaway tmpfs.
            self.assertNotIn("--ro-bind /home", joined)
            self.assertNotIn("--bind /home", joined)
            self.assertIn("--setenv HOME /tmp", joined)
            # systemd resource limits are applied.
            self.assertIn("--property=RuntimeMaxSec=14s", joined)
            self.assertIn("--property=MemoryMax=", joined)
            self.assertIn("--property=TasksMax=", joined)
            self.assertIn("--property=CPUQuota=", joined)
            # Extra verifier env is passed through.
            self.assertIn("PYTHONDONTWRITEBYTECODE", joined)

    def test_isolated_command_can_make_workspace_writable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = self._workspace(root)
            sandbox = BubblewrapSandbox(root, workspace)
            command = sandbox.build_isolated_command(
                "pwd",
                10,
                read_only=[],
                writable=[SandboxBind(workspace, "/workspace")],
            )
            joined = " ".join(command)
            self.assertIn(f"--bind {workspace.resolve()} /workspace", joined)
            self.assertNotIn(f"--ro-bind {workspace.resolve()} /workspace", joined)

    def test_availability_helpers_are_boolean(self) -> None:
        self.assertIsInstance(bwrap_available(), bool)
        self.assertIsInstance(systemd_user_available(), bool)
        self.assertIsInstance(sandbox_available(), bool)
        # A sandbox is only available when both layers are present.
        self.assertEqual(sandbox_available(), bwrap_available() and systemd_user_available())

    def test_sandbox_python_interpreter_is_reachable_when_present(self) -> None:
        # The resolver must never return a path the sandbox cannot reach.
        interpreter = sandbox_python_interpreter()
        if interpreter is not None:
            self.assertTrue(os.path.exists(interpreter), interpreter)
            self.assertTrue(
                interpreter.startswith(("/usr/", "/bin/", "/lib/", "/lib64/")),
                interpreter,
            )
            # The returned path must be canonical: a symlink like
            # /usr/bin/python3 -> /etc/alternatives/python3 -> ... would hop
            # through an unmounted directory inside the sandbox, so the
            # resolved path has to equal its realpath (no symlink hops).
            self.assertEqual(os.path.realpath(interpreter), interpreter, interpreter)
        # A bogus version resolves to nothing rather than a wrong interpreter.
        self.assertIsNone(sandbox_python_interpreter(preferred_version="9.9"))

    @unittest.skipUnless(sandbox_available(), "requires bwrap and a systemd user session")
    def test_isolated_execute_blocks_host_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as extra:
            root = Path(directory)
            workspace = self._workspace(root)
            private = Path(extra) / "private"
            output = Path(extra) / "output"
            private.mkdir()
            output.mkdir()
            python = sandbox_python_interpreter()
            if python is None:
                self.skipTest("no sandbox-visible python3 interpreter")
            # Host-side listener simulating a local model endpoint.
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            # Canary in the real home proving /home is not readable in-sandbox.
            canary = Path.home() / f".pyreplab-sandbox-canary-{os.getpid()}.txt"
            canary.write_text("secret", encoding="utf-8")
            probe = private / "probe.py"
            probe.write_text(
                (
                    "import json, os, socket\n"
                    "probe = {}\n"
                    "canary = os.environ['CANARY']\n"
                    "port = int(os.environ['PORT'])\n"
                    "try:\n"
                    "    with open(canary, 'r', encoding='utf-8') as handle:\n"
                    "        probe['home'] = 'read:' + handle.read()\n"
                    "except Exception as error:\n"
                    "    probe['home'] = type(error).__name__\n"
                    "try:\n"
                    "    conn = socket.create_connection(('127.0.0.1', port), timeout=3)\n"
                    "    conn.close()\n"
                    "    probe['network'] = 'connected'\n"
                    "except Exception as error:\n"
                    "    probe['network'] = type(error).__name__\n"
                    "for label, target in (('workspace', '/workspace/evil'), ('etc', '/etc/pyreplab-evil')):\n"
                    "    try:\n"
                    "        with open(target, 'w', encoding='utf-8') as handle:\n"
                    "            handle.write('x')\n"
                    "        probe['write_' + label] = 'ok'\n"
                    "    except Exception as error:\n"
                    "        probe['write_' + label] = type(error).__name__\n"
                    "try:\n"
                    "    with open('/output/probe.json', 'w', encoding='utf-8') as handle:\n"
                    "        probe['write_output'] = 'ok'\n"
                    "        json.dump(probe, handle)\n"
                    "except Exception as error:\n"
                    "    probe['write_output'] = type(error).__name__\n"
                ),
                encoding="utf-8",
            )
            try:
                sandbox = BubblewrapSandbox(root, workspace)
                result = sandbox.execute_isolated(
                    f"{python} /private/probe.py",
                    30,
                    read_only=[SandboxBind(workspace, "/workspace"), SandboxBind(private, "/private")],
                    writable=[SandboxBind(output, "/output")],
                    extra_env={"CANARY": str(canary), "PORT": str(port)},
                )
                self.assertEqual(result.exit_code, 0, result.stderr)
                probe_report = json.loads((output / "probe.json").read_text(encoding="utf-8"))
                # Cannot read the /home canary: on the host this open() succeeds.
                self.assertNotEqual(probe_report["home"], "read:secret")
                self.assertTrue(probe_report["home"].endswith("Error"))
                # Cannot reach the host loopback model endpoint (host connect succeeds).
                self.assertNotEqual(probe_report["network"], "connected")
                self.assertTrue(probe_report["network"].endswith("Error"))
                # Cannot write outside the allowed mounts.
                self.assertTrue(probe_report["write_workspace"].endswith("Error"))
                self.assertTrue(probe_report["write_etc"].endswith("Error"))
                # The writable output mount works as intended.
                self.assertEqual(probe_report["write_output"], "ok")
            finally:
                listener.close()
                try:
                    canary.unlink()
                except OSError:
                    pass


if __name__ == "__main__":
    unittest.main()
