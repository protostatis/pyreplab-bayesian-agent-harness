from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from pyreplab_harness.unbrowser_sandbox import UnbrowserSandbox


class UnbrowserSandboxTest(unittest.TestCase):
    def _fake_binary(self, directory: str) -> str:
        path = Path(directory) / "fake-unbrowser"
        path.write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
        path.chmod(0o755)
        return str(path)

    # -- constructor validation --

    def test_rejects_relative_binary(self) -> None:
        with self.assertRaisesRegex(ValueError, "absolute path"):
            UnbrowserSandbox("unbrowser")

    def test_rejects_missing_binary(self) -> None:
        with self.assertRaises(FileNotFoundError):
            UnbrowserSandbox("/nonexistent/unbrowser-binary")

    def test_rejects_nonpositive_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            binary = self._fake_binary(d)
            with self.assertRaisesRegex(ValueError, "positive"):
                UnbrowserSandbox(binary, command_timeout=0)
            with self.assertRaisesRegex(ValueError, "positive"):
                UnbrowserSandbox(binary, command_timeout=-5)

    # -- build_command shape --

    def test_build_command_starts_with_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            binary = self._fake_binary(d)
            sandbox = UnbrowserSandbox(binary)
            cmd = sandbox.build_command()
            self.assertEqual(cmd[0], "timeout")
            # timeout --kill-after=5s <seconds>s bwrap ...
            self.assertTrue(cmd[1].startswith("--kill-after"))

    def test_build_command_includes_bwrap(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            binary = self._fake_binary(d)
            sandbox = UnbrowserSandbox(binary)
            cmd = sandbox.build_command()
            self.assertIn("bwrap", cmd)

    def test_binary_is_ro_bound(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            binary = self._fake_binary(d)
            sandbox = UnbrowserSandbox(binary)
            cmd = sandbox.build_command()
            # Find the --ro-bind for the binary
            found = False
            for i, token in enumerate(cmd):
                if token == "--ro-bind" and cmd[i + 1] == binary:
                    found = True
                    break
            self.assertTrue(found, f"binary {binary!r} not found with --ro-bind in {cmd}")

    def test_network_namespace_not_unshared(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            binary = self._fake_binary(d)
            sandbox = UnbrowserSandbox(binary)
            cmd = sandbox.build_command()
            self.assertNotIn("--unshare-net", cmd)

    def test_mount_namespace_is_unshared(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            binary = self._fake_binary(d)
            sandbox = UnbrowserSandbox(binary)
            cmd = sandbox.build_command()
            self.assertIn("--unshare-mount", cmd)

    def test_user_namespace_is_unshared(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            binary = self._fake_binary(d)
            sandbox = UnbrowserSandbox(binary)
            cmd = sandbox.build_command()
            self.assertIn("--unshare-user", cmd)

    def test_temporary_home_is_inside_sandbox(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            binary = self._fake_binary(d)
            sandbox = UnbrowserSandbox(binary)
            cmd = sandbox.build_command()
            # The HOME is set to /home/unbrowser (inside the sandbox, not host).
            home_setenv = False
            for i, token in enumerate(cmd):
                if token == "--setenv" and cmd[i + 1] == "HOME":
                    self.assertEqual(cmd[i + 2], "/home/unbrowser")
                    home_setenv = True
                    break
            self.assertTrue(home_setenv, "HOME not set to /home/unbrowser")
            # The directory is created inside the sandbox.
            self.assertIn("--dir", cmd)

    def test_no_host_home_mount(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            binary = self._fake_binary(d)
            sandbox = UnbrowserSandbox(binary)
            cmd = sandbox.build_command()
            cmd_str = " ".join(cmd)
            host_home = os.path.expanduser("~")
            self.assertNotIn(host_home, cmd_str)

    def test_no_all_unshare(self) -> None:
        """--unshare-all would unshare net; we use individual flags instead."""
        with tempfile.TemporaryDirectory() as d:
            binary = self._fake_binary(d)
            sandbox = UnbrowserSandbox(binary)
            cmd = sandbox.build_command()
            self.assertNotIn("--unshare-all", cmd)

    def test_different_timeouts_produce_different_timeout_values(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            binary = self._fake_binary(d)
            s1 = UnbrowserSandbox(binary, command_timeout=30)
            s2 = UnbrowserSandbox(binary, command_timeout=60)
            cmd1 = s1.build_command()
            cmd2 = s2.build_command()
            self.assertNotEqual(cmd1, cmd2)
            # The timeout value appears in the command after --kill-after=5s.
            self.assertIn("30s", cmd1)
            self.assertIn("60s", cmd2)

    def test_fresh_dev_and_proc(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            binary = self._fake_binary(d)
            sandbox = UnbrowserSandbox(binary)
            cmd = sandbox.build_command()
            self.assertIn("--proc", cmd)
            self.assertIn("--dev", cmd)
            # /tmp is a fresh tmpfs.
            self.assertIn("--tmpfs", cmd)
            self.assertIn("/tmp", cmd)

    def test_dns_and_tls_binds_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            binary = self._fake_binary(d)
            sandbox = UnbrowserSandbox(binary)
            cmd = sandbox.build_command()
            cmd_str = " ".join(cmd)
            if os.path.exists("/etc/resolv.conf"):
                self.assertIn("/etc/resolv.conf", cmd_str)
            if os.path.exists("/etc/ssl"):
                self.assertIn("/etc/ssl", cmd_str)

    def test_clearenv_present(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            binary = self._fake_binary(d)
            sandbox = UnbrowserSandbox(binary)
            cmd = sandbox.build_command()
            self.assertIn("--clearenv", cmd)

    def test_args_are_appended_after_double_dash(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            binary = self._fake_binary(d)
            sandbox = UnbrowserSandbox(binary)
            cmd = sandbox.build_command("--version")
            # The -- separator must be present, and args come after it.
            self.assertIn("--", cmd)
            dash_index = cmd.index("--")
            self.assertIn("--version", cmd[dash_index + 1:])

    def test_binary_appears_after_separator(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            binary = self._fake_binary(d)
            sandbox = UnbrowserSandbox(binary)
            cmd = sandbox.build_command()
            dash_index = cmd.index("--")
            after = cmd[dash_index + 1:]
            self.assertEqual(after[0], binary)

    # -- canary_paths --

    def test_canary_paths_returns_list_of_strings(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            binary = self._fake_binary(d)
            sandbox = UnbrowserSandbox(binary)
            paths = sandbox.canary_paths()
            self.assertIsInstance(paths, list)
            self.assertTrue(all(isinstance(p, str) for p in paths))

    def test_canary_paths_includes_home(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            binary = self._fake_binary(d)
            sandbox = UnbrowserSandbox(binary)
            paths = sandbox.canary_paths()
            self.assertIn(os.path.expanduser("~"), paths)
            self.assertIn("/home", paths)

    def test_canary_paths_excludes_runtime_libs(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            binary = self._fake_binary(d)
            sandbox = UnbrowserSandbox(binary)
            paths = sandbox.canary_paths()
            for p in paths:
                self.assertFalse(
                    p.startswith("/usr/lib/") or p == "/usr/lib",
                    f"canary path {p!r} should not be in runtime libs",
                )
                self.assertFalse(
                    p.startswith("/lib/") or p == "/lib",
                    f"canary path {p!r} should not be in runtime libs",
                )
                self.assertFalse(
                    p.startswith("/lib64/") or p == "/lib64",
                    f"canary path {p!r} should not be in runtime libs",
                )

    def test_no_project_or_run_artifacts_in_mounts(self) -> None:
        """Verify the mount list does not contain project or user paths."""
        with tempfile.TemporaryDirectory() as d:
            binary = self._fake_binary(d)
            sandbox = UnbrowserSandbox(binary)
            cmd = sandbox.build_command()
            cmd_str = " ".join(cmd)

            canary_paths = sandbox.canary_paths()
            # None of the canary paths should appear in the mount list.
            for cp in canary_paths:
                self.assertNotIn(
                    f"--ro-bind {cp}",
                    cmd_str,
                    f"canary path {cp!r} found in mount list",
                )
                self.assertNotIn(
                    f"--bind {cp}",
                    cmd_str,
                    f"canary path {cp!r} found as writable mount",
                )

    def test_stores_command_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            binary = self._fake_binary(d)
            sandbox = UnbrowserSandbox(binary, command_timeout=45)
            self.assertEqual(sandbox.command_timeout, 45)


if __name__ == "__main__":
    unittest.main()
