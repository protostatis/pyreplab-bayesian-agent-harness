from __future__ import annotations

import hashlib
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pyreplab_harness.orchestrator import RemoteConfig
from pyreplab_harness.m3_pilot import (
    HELD_TEMPLATES,
    KNOWN_TEMPLATES,
    _contains_argument_sequence,
    _argument_value,
    _pi_provider_identity,
    _run_checked,
    build_headroom_manifest,
    freeze_headroom_manifest,
    run_headroom_pilot,
    source_tree_hash,
    validate_headroom_manifest,
)
from pyreplab_harness.treatments import TreatmentRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "policies" / "m3-unbrowser-treatments.json"
SPLIT_PATH = PROJECT_ROOT / "policies" / "m3-unbrowser-policy-split.json"
PILOT_PATH = PROJECT_ROOT / "policies" / "m3-unbrowser-headroom-pilot.json"


class M3PilotManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = TreatmentRegistry.load(REGISTRY_PATH)
        self.policy_split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
        self.manifest = build_headroom_manifest(
            self.registry,
            self.policy_split,
            registry_file=REGISTRY_PATH.name,
            policy_split_file=SPLIT_PATH.name,
        )

    def test_manifest_is_valid_and_uses_only_meta_train_policies(self) -> None:
        validate_headroom_manifest(self.manifest, self.registry, self.policy_split)
        meta_train = set(self.policy_split["splits"]["meta_train"])
        self.assertEqual(len(self.manifest["policy_labels"]), 4)
        self.assertTrue(set(self.manifest["policy_labels"].values()) <= meta_train)
        self.assertEqual(
            self.manifest["selection"]["pairwise_hamming_distances"],
            [3, 4, 4, 3, 4, 4],
        )
        self.assertEqual(self.manifest["selection"]["total_pairwise_distance"], 22)

    def test_task_matrix_is_balanced_and_uses_every_order_once(self) -> None:
        tasks = self.manifest["tasks"]
        panels = self.manifest["panels"]
        self.assertEqual(len(tasks), 12)
        self.assertEqual(len(panels), 24)
        self.assertEqual({task["template"] for task in tasks}, set(KNOWN_TEMPLATES))
        self.assertFalse({task["template"] for task in tasks} & set(HELD_TEMPLATES))
        for template in KNOWN_TEMPLATES:
            self.assertEqual(sum(task["template"] == template for task in tasks), 2)
        for difficulty in ("easy", "medium", "hard"):
            self.assertEqual(
                sum(task["difficulty"] == difficulty for task in tasks), 4
            )
        self.assertEqual(
            {tuple(panel["execution_order"]) for panel in panels},
            set(itertools.permutations(("A", "B", "C", "D"))),
        )
        for position in range(4):
            for label in ("A", "B", "C", "D"):
                self.assertEqual(
                    sum(panel["execution_order"][position] == label for panel in panels),
                    6,
                )
        task_by_id = {task["task_id"]: task for task in tasks}
        for template in KNOWN_TEMPLATES:
            template_panels = [
                panel
                for panel in panels
                if task_by_id[panel["task_id"]]["template"] == template
            ]
            for position in range(4):
                self.assertEqual(
                    {panel["execution_order"][position] for panel in template_panels},
                    {"A", "B", "C", "D"},
                )
        self.assertEqual(len({panel["sampling_seed"] for panel in panels}), 24)
        first_replica_counts = {0: 0, 1: 0}
        for task in tasks:
            task_panels = [
                panel for panel in panels if panel["task_id"] == task["task_id"]
            ]
            first_panel = min(task_panels, key=panels.index)
            first_replica_counts[first_panel["rollout_replica"]] += 1
        self.assertEqual(first_replica_counts, {0: 6, 1: 6})

    def test_manifest_tampering_fails_closed(self) -> None:
        tampered = json.loads(json.dumps(self.manifest))
        tampered["tasks"][0]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "manifest_hash mismatch"):
            validate_headroom_manifest(tampered, self.registry, self.policy_split)

    def test_freeze_is_idempotent_and_committed_file_matches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pilot.json"
            first = freeze_headroom_manifest(output, REGISTRY_PATH, SPLIT_PATH)
            second = freeze_headroom_manifest(output, REGISTRY_PATH, SPLIT_PATH)
            self.assertEqual(first, second)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")), self.manifest
            )
        self.assertEqual(
            json.loads(PILOT_PATH.read_text(encoding="utf-8")), self.manifest
        )

    def test_source_tree_hash_is_content_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            path = root / "src" / "module.py"
            path.write_text("value = 1\n", encoding="utf-8")
            first = source_tree_hash(root)
            self.assertEqual(first, source_tree_hash(root))
            path.write_text("value = 2\n", encoding="utf-8")
            self.assertNotEqual(first, source_tree_hash(root))

    def test_pi_provider_identity_excludes_secrets_and_pins_sampling_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.json"
            path.write_text(
                json.dumps(
                    {
                        "providers": {
                            "provider": {
                                "api": "openai-completions",
                                "baseUrl": "http://127.0.0.1:1234/v1",
                                "apiKey": "secret",
                                "models": [
                                    {
                                        "id": "model",
                                        "contextWindow": 1024,
                                        "maxTokens": 128,
                                        "reasoning": False,
                                    }
                                ],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                _pi_provider_identity(path, "provider", "model"),
                {
                    "api": "openai-completions",
                    "base_url": "http://127.0.0.1:1234/v1",
                    "context_window": 1024,
                    "max_tokens": 128,
                    "reasoning": False,
                    "sampling_params": None,
                },
            )

    def test_required_server_arguments_match_exact_token_pairs(self) -> None:
        arguments = ["llama-server", "--parallel", "10", "--threads", "8"]
        self.assertFalse(_contains_argument_sequence(arguments, "--parallel 1"))
        self.assertTrue(_contains_argument_sequence(arguments, "--threads 8"))
        self.assertEqual(_argument_value(arguments, "--threads"), "8")
        self.assertIsNone(_argument_value(arguments, "--model", "-m"))

    def test_preflight_cli_parses_without_running_pilot(self) -> None:
        from pyreplab_harness.m3_pilot import _build_parser

        args = _build_parser().parse_args(
            [
                "preflight",
                "--root",
                str(PROJECT_ROOT),
                "--remote-project",
                "/remote/project",
                "--remote-run-root",
                "/remote/runs",
                "--unbrowser-binary",
                "/remote/unbrowser",
                "--model-artifact",
                "/remote/model.gguf",
            ]
        )
        self.assertEqual(args.command, "preflight")

    def test_checked_command_can_read_successful_version_from_stderr(self) -> None:
        with patch("pyreplab_harness.m3_pilot.subprocess.run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = ""
            run.return_value.stderr = "version: 1 (abc)\n"
            self.assertEqual(
                _run_checked(["server", "--version"], stderr_fallback=True),
                "version: 1 (abc)",
            )

    def test_runtime_preflight_require_clean_default_raises_on_dirty(self) -> None:
        from pyreplab_harness.m3_pilot import runtime_preflight
        from unittest.mock import patch, MagicMock

        with patch(
            "pyreplab_harness.m3_pilot._run_checked",
            return_value=" M dirty.py\n",
        ), patch("pyreplab_harness.m3_pilot.validate_remote_config"), patch(
            "pyreplab_harness.m3_pilot._sha256_file",
            return_value="a" * 64,
        ), patch(
            "pyreplab_harness.m3_pilot._pi_provider_identity",
            return_value={},
        ), patch(
            "pyreplab_harness.m3_pilot._model_endpoint_entry",
            return_value={"status": {"args": ["/tmp/llama-server", "--parallel", "1", "--threads", "8"]}},
        ), patch(
            "pyreplab_harness.m3_pilot._ssh_capture",
        ) as ssh:
            ssh.side_effect = lambda *a, **kw: {
                "pi_version": "0.84.1",
                "code_revision": "a" * 40,
                "source_tree_hash": "b" * 64,
            }.get(kw.get("stderr_fallback", False) and "version" or "", "ok")

            config = RemoteConfig("host", "/p", "/r", "python3")
            with self.assertRaises(RuntimeError):
                runtime_preflight(
                    Path("/tmp/fake-root"),
                    config,
                    pi_binary="/tmp/pi",
                    thinking="off",
                    unbrowser_binary="/tmp/unbrowser",
                    model_artifact="/tmp/model.gguf",
                    llama_server_binary="/tmp/llama-server",
                )  # require_clean=True by default

    def test_runtime_preflight_require_clean_false_returns_worktree_fields(
        self,
    ) -> None:
        from pyreplab_harness.m3_pilot import _RUNTIME_PINS, runtime_preflight

        dirty = " M dirty.py\n"
        server = _RUNTIME_PINS["llama_server_path"]
        model = _RUNTIME_PINS["model_artifact_path"]
        server_args = [server, "--model", model]
        for required in _RUNTIME_PINS["llama_server_required_args"]:
            server_args.extend(required.split())

        def run_checked(command, **_kwargs):
            if command[:3] == ["git", "status", "--porcelain"]:
                return dirty
            if command[:3] == ["git", "rev-parse", "HEAD"]:
                return "abc123"
            if command[-1] == "--version":
                return _RUNTIME_PINS["pi_version"]
            raise AssertionError(command)

        def ssh_capture(_host, command, **_kwargs):
            joined = " ".join(command)
            if "source-hash" in command:
                return "source-hash"
            if "confined-unbrowser-check" in command:
                return f"unbrowser {_RUNTIME_PINS['unbrowser_version']}"
            if command[0] == "sha256sum":
                hashes = {
                    _RUNTIME_PINS["unbrowser_path"]: _RUNTIME_PINS["unbrowser_sha256"],
                    model: _RUNTIME_PINS["model_artifact_sha256"],
                    server: _RUNTIME_PINS["llama_server_sha256"],
                }
                return f"{hashes[command[1]]}  {command[1]}"
            if command == [server, "--version"]:
                return _RUNTIME_PINS["llama_server_version"]
            if command == ["bwrap", "--version"]:
                return _RUNTIME_PINS["bubblewrap_version"]
            if "model-endpoint-entry" in command:
                return json.dumps({"status": {"args": server_args}})
            if "fixture-port-check" in command:
                return "available"
            raise AssertionError(joined)

        pi_path = "/tmp/node_modules/@earendil-works/pi-coding-agent/pi"
        with patch(
            "pyreplab_harness.m3_pilot._run_checked", side_effect=run_checked
        ), patch(
            "pyreplab_harness.m3_pilot.validate_remote_config"
        ), patch(
            "pyreplab_harness.m3_pilot.shutil.which", return_value=pi_path
        ), patch(
            "pyreplab_harness.m3_pilot._sha256_file",
            return_value=_RUNTIME_PINS["pi_cli_sha256"],
        ), patch(
            "pyreplab_harness.m3_pilot._pi_provider_identity",
            return_value=_RUNTIME_PINS["pi_provider_config"],
        ), patch(
            "pyreplab_harness.m3_pilot.source_tree_hash",
            return_value="source-hash",
        ), patch(
            "pyreplab_harness.m3_pilot._model_endpoint_entry",
            return_value={"status": {"args": server_args}},
        ), patch(
            "pyreplab_harness.m3_pilot._ssh_capture", side_effect=ssh_capture
        ):
            report = runtime_preflight(
                Path("/tmp/fake-root"),
                RemoteConfig("host", "/p", "/r", "python3"),
                pi_binary="pi",
                thinking="off",
                unbrowser_binary=_RUNTIME_PINS["unbrowser_path"],
                model_artifact=model,
                llama_server_binary=server,
                require_clean=False,
            )

        self.assertFalse(report["worktree_clean"])
        self.assertEqual(
            report["worktree_status_hash"],
            hashlib.sha256(dirty.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(report["source_tree_hash"], "source-hash")
        self.assertEqual(report["runtime_pins"], _RUNTIME_PINS)

    def test_interrupted_panel_marker_blocks_resume(self) -> None:
        runtime = {
            "checked_at": "2026-08-10T00:00:00+00:00",
            "code_revision": "a" * 40,
            "source_tree_hash": "b" * 64,
            "worktree_clean": True,
            "worktree_status_hash": hashlib.sha256(
                b"PYREPLAB_GIT_WORKTREE_CLEAN_MARKER_V1"
            ).hexdigest(),
            "runtime_pins": self.manifest["runtime_pins"],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pilot.jsonl"
            arguments = (
                PILOT_PATH,
                REGISTRY_PATH,
                SPLIT_PATH,
                output,
                RemoteConfig("host", "/remote/project", "/remote/runs", "python3"),
            )
            keywords = {
                "pi_binary": "pi",
                "provider": "ubuntu-gemma",
                "model": "gemma-4-26b-a4b",
                "thinking": "off",
                "unbrowser_binary": "/remote/unbrowser",
                "model_artifact": "/remote/model.gguf",
                "llama_server_binary": "/remote/llama-server",
            }
            with patch(
                "pyreplab_harness.m3_pilot.runtime_preflight",
                return_value=runtime,
            ), patch(
                "pyreplab_harness.m3_pilot.run_registered_treatments",
                side_effect=KeyboardInterrupt,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    run_headroom_pilot(*arguments, **keywords)
            active = output.with_suffix(".jsonl.active.json")
            self.assertTrue(active.is_file())
            with self.assertRaisesRegex(RuntimeError, "unfinished pilot panel"):
                run_headroom_pilot(*arguments, **keywords)


if __name__ == "__main__":
    unittest.main()
