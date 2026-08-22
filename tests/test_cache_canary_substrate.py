from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pyreplab_harness.cache_canary_substrate import (
    build_cache_canary_substrate_manifest,
    preflight_cache_canary_substrate,
    validate_cache_canary_substrate_manifest,
)
from pyreplab_harness.cache_mechanics import build_cache_runtime_receipt, canonical_receipt_hash


SERVER = "/usr/local/lib/ollama/llama-server"
MODEL = "/models/gemma.gguf"
HELP = " ".join(
    (
        "--model --alias --host --port --ctx-size --flash-attn --n-cpu-moe",
        "--n-gpu-layers --parallel --reasoning --threads --cache-type-k",
        "--cache-type-v --cache-ram --ctx-checkpoints --checkpoint-min-step",
        "--cache-idle-slots --no-cache-idle-slots --cache-reuse --kv-unified",
        "--no-kv-unified --metrics --slots --no-slots --slot-save-path",
        "--sleep-idle-seconds --perf --no-context-shift --no-cont-batching",
        "--warmup --no-webui --timeout --sse-ping-interval",
        "--cache-prompt --no-cache-prompt",
    )
)


def _runtime_probe() -> dict:
    return build_cache_runtime_receipt(
        {
            "checked_at": "2026-08-15T00:00:00+00:00",
            "pi_version": "0.84.1",
            "pi_sha256": "a" * 64,
            "llama_server_version": "version: 1",
            "llama_server_sha256": "b" * 64,
            "model_sha256": "c" * 64,
            "model_state": "sleeping",
            "server_argv": [
                SERVER,
                "--model",
                MODEL,
                "--ctx-size",
                "65536",
                "--parallel",
                "1",
                "--sleep-idle-seconds",
                "300",
            ],
            "server_help": HELP,
            "metrics_endpoint": {"http_status": 400},
            "slots_endpoint": {"http_status": 400},
        }
    )


class CacheCanarySubstrateTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.manifest = build_cache_canary_substrate_manifest(
            project_root=self.root,
            runtime_probe=_runtime_probe(),
            server_binary=SERVER,
            model_artifact=MODEL,
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _rehash(self, manifest: dict) -> None:
        manifest["manifest_hash"] = canonical_receipt_hash(
            {key: value for key, value in manifest.items() if key != "manifest_hash"}
        )

    def test_manifest_is_non_authorizing_and_cells_have_one_delta(self) -> None:
        validate_cache_canary_substrate_manifest(self.manifest)
        self.assertFalse(self.manifest["live_model_execution_authorized"])
        common = self.manifest["common_configuration"][
            "server_argv_without_cache_mode"
        ]
        off, on = self.manifest["cells"]
        self.assertEqual(off["server_argv"], [*common, "--no-cache-prompt"])
        self.assertEqual(on["server_argv"], [*common, "--cache-prompt"])
        self.assertNotIn("--slot-save-path", common)
        self.assertFalse(
            self.manifest["telemetry_contract"]["raw_request_persistence"]
        )

    def test_extra_cell_delta_is_rejected(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["cells"][1]["server_argv"].extend(["--threads", "9"])
        manifest["cells"][1]["server_argv_hash"] = canonical_receipt_hash(
            manifest["cells"][1]["server_argv"]
        )
        self._rehash(manifest)
        with self.assertRaisesRegex(ValueError, "beyond cache mode"):
            validate_cache_canary_substrate_manifest(manifest)

    def test_slot_save_path_is_forbidden(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["common_configuration"]["server_argv_without_cache_mode"].extend(
            ["--slot-save-path", "/tmp/cache"]
        )
        manifest["common_config_hash"] = canonical_receipt_hash(
            manifest["common_configuration"]
        )
        self._rehash(manifest)
        with self.assertRaisesRegex(ValueError, "slot persistence"):
            validate_cache_canary_substrate_manifest(manifest)

    def test_preflight_is_identity_only_and_does_not_launch_server(self) -> None:
        commands: list[list[str]] = []

        def capture(host: str, command: list[str], **kwargs) -> str:
            self.assertEqual(host, "ubuntu-local")
            commands.append(command)
            if command == [SERVER, "--version"]:
                return "version: 1"
            if command == ["sha256sum", SERVER]:
                return f"{'b' * 64}  {SERVER}"
            if command == ["test", "-r", MODEL]:
                return ""
            if command == ["sha256sum", MODEL]:
                return f"{'c' * 64}  {MODEL}"
            if command == [SERVER, "--help"]:
                return HELP
            if command == ["ss", "-H", "-ltn"]:
                return "LISTEN 0 128 127.0.0.1:8081 0.0.0.0:*"
            if command[0:4] == [
                "systemctl",
                "--user",
                "show",
                "gemma.service",
            ]:
                return "ActiveState=active\nFragmentPath=/gemma.service\nExecStart=/server"
            raise AssertionError(command)

        with mock.patch(
            "pyreplab_harness.cache_canary_substrate._ssh_capture",
            side_effect=capture,
        ), mock.patch(
            "pyreplab_harness.cache_canary_substrate._local_port_available",
            return_value=True,
        ):
            receipt = preflight_cache_canary_substrate(
                self.manifest, project_root=self.root, host="ubuntu-local"
            )

        self.assertFalse(receipt["model_loaded_or_invoked"])
        self.assertFalse(receipt["active_service_mutated"])
        self.assertFalse(receipt["live_model_execution_authorized"])
        flattened = " ".join(" ".join(command) for command in commands)
        self.assertNotIn("--cache-prompt", flattened)
        self.assertNotIn("/v1/completions", flattened)
        self.assertNotIn("curl", flattened)

    def test_preflight_rejects_remote_port_collision(self) -> None:
        def capture(host: str, command: list[str], **kwargs) -> str:
            if command == [SERVER, "--version"]:
                return "version: 1"
            if command == ["sha256sum", SERVER]:
                return f"{'b' * 64}  {SERVER}"
            if command == ["test", "-r", MODEL]:
                return ""
            if command == ["sha256sum", MODEL]:
                return f"{'c' * 64}  {MODEL}"
            if command == [SERVER, "--help"]:
                return HELP
            if command == ["ss", "-H", "-ltn"]:
                return "LISTEN 0 128 127.0.0.1:18082 0.0.0.0:*"
            raise AssertionError(command)

        with mock.patch(
            "pyreplab_harness.cache_canary_substrate._ssh_capture",
            side_effect=capture,
        ):
            with self.assertRaisesRegex(RuntimeError, "already in use"):
                preflight_cache_canary_substrate(
                    self.manifest, project_root=self.root, host="ubuntu-local"
                )

    def test_preflight_rejects_model_hash_drift(self) -> None:
        def capture(host: str, command: list[str], **kwargs) -> str:
            if command == [SERVER, "--version"]:
                return "version: 1"
            if command == ["sha256sum", SERVER]:
                return f"{'b' * 64}  {SERVER}"
            if command == ["test", "-r", MODEL]:
                return ""
            if command == ["sha256sum", MODEL]:
                return f"{'d' * 64}  {MODEL}"
            raise AssertionError(command)

        with mock.patch(
            "pyreplab_harness.cache_canary_substrate._ssh_capture",
            side_effect=capture,
        ):
            with self.assertRaisesRegex(RuntimeError, "model artifact hash drift"):
                preflight_cache_canary_substrate(
                    self.manifest, project_root=self.root, host="ubuntu-local"
                )


if __name__ == "__main__":
    unittest.main()
