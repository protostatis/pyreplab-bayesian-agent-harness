from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pyreplab_harness.events import (
    BUDGET_RECEIPT_SCHEMA_VERSION,
    NORMALIZED_EVENT_SCHEMA_VERSION,
    PROVIDER_TURN_SEMANTICS,
)
from pyreplab_harness.m3_empty_overlay_baseline import (
    EXPECTED_ATTEMPTS,
    EXPECTED_TASKS,
    SAMPLING_SEED_START,
    SCREEN_ID,
    TASK_SEED_START,
    _validate_lifecycle_receipt,
    build_baseline_manifest,
    build_command_template_receipt,
    build_empty_overlay_registry,
    build_local_preflight,
    freeze_baseline_artifacts,
    run_lifecycle_stress,
    run_remote_preflight,
    validate_baseline_manifest,
    validate_local_preflight,
    validate_remote_preflight,
)
from pyreplab_harness.m3_pilot import (
    HELD_TEMPLATES,
    KNOWN_TEMPLATES,
    _RUNTIME_PINS,
    _canonical_hash,
)
from pyreplab_harness.orchestrator import (
    RemoteConfig,
    run_registered_treatments,
)
from pyreplab_harness.treatments import TreatmentRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_IDENTITY = {
    "host": "ubuntu-local",
    "project": "/remote/project",
    "run_root": "/remote/project/.runs/empty-overlay",
    "python": "python3",
}


class EmptyOverlayBaselineManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = build_empty_overlay_registry()
        self.manifest = build_baseline_manifest(
            self.registry,
            REMOTE_IDENTITY,
            registry_file="registry.json",
        )

    def test_registry_contains_one_exact_empty_overlay(self) -> None:
        self.assertEqual(len(self.registry), 1)
        treatment = self.registry.treatments[0]
        self.assertEqual(treatment.system_prompt, "")
        self.assertEqual(treatment.allowed_tools, ("bash", "unbrowser"))
        self.assertEqual(treatment.tool_call_limit, 12)
        self.assertEqual(treatment.max_output_tokens, 4096)

    def test_manifest_is_deterministic_and_excludes_holdouts(self) -> None:
        second = build_baseline_manifest(
            self.registry,
            REMOTE_IDENTITY,
            registry_file="registry.json",
        )
        self.assertEqual(self.manifest, second)
        self.assertEqual(len(self.manifest["tasks"]), EXPECTED_TASKS)
        self.assertEqual(len(self.manifest["panels"]), EXPECTED_ATTEMPTS)
        self.assertEqual(
            {task["template"] for task in self.manifest["tasks"]},
            set(KNOWN_TEMPLATES),
        )
        self.assertFalse(
            set(HELD_TEMPLATES)
            & {task["template"] for task in self.manifest["tasks"]}
        )
        self.assertEqual(
            len({panel["sampling_seed"] for panel in self.manifest["panels"]}),
            EXPECTED_ATTEMPTS,
        )
        self.assertEqual(SCREEN_ID, "m3-empty-overlay-baseline-20260815-v5")
        self.assertEqual(TASK_SEED_START, 2026091001)
        self.assertEqual(SAMPLING_SEED_START, 1900009001)
        self.assertEqual(
            self.manifest["task_generator_version"], "unbrowser-fixture-v3"
        )
        self.assertEqual(
            self.manifest["runtime_pins"]["fixture_generator_version"],
            _RUNTIME_PINS["fixture_generator_version"],
        )
        self.assertEqual(
            self.manifest["event_accounting"],
            {
                "normalizer_schema_version": NORMALIZED_EVENT_SCHEMA_VERSION,
                "provider_turn_semantics": PROVIDER_TURN_SEMANTICS,
                "budget_receipt_schema_version": BUDGET_RECEIPT_SCHEMA_VERSION,
            },
        )
        self.assertTrue(
            {task["seed"] for task in self.manifest["tasks"]}.isdisjoint(
                range(2026088001, 2026088001 + EXPECTED_TASKS)
            )
        )
        self.assertTrue(
            {panel["sampling_seed"] for panel in self.manifest["panels"]}.isdisjoint(
                range(1900006001, 1900006001 + EXPECTED_ATTEMPTS)
            )
        )
        self.assertTrue(
            {task["seed"] for task in self.manifest["tasks"]}.isdisjoint(
                range(2026089001, 2026089001 + EXPECTED_TASKS)
            )
        )
        self.assertTrue(
            {panel["sampling_seed"] for panel in self.manifest["panels"]}.isdisjoint(
                range(1900007001, 1900007001 + EXPECTED_ATTEMPTS)
            )
        )
        self.assertFalse(
            self.manifest["authorization_boundary"][
                "live_model_execution_authorized"
            ]
        )
        validate_baseline_manifest(self.manifest, self.registry)

    def test_manifest_tampering_is_rejected(self) -> None:
        tampered = dict(self.manifest)
        tampered["schedule_seed"] = 1
        with self.assertRaisesRegex(ValueError, "manifest_hash"):
            validate_baseline_manifest(tampered, self.registry)

    def test_command_receipt_proves_no_appended_prompt(self) -> None:
        receipt = build_command_template_receipt(
            self.manifest,
            self.registry,
            PROJECT_ROOT,
        )
        self.assertNotIn("--append-system-prompt", receipt["argv"])
        self.assertEqual(receipt["argv"][-1], "__PYREPLAB_TASK_PROMPT__")
        self.assertIn(str(PROJECT_ROOT / "pi_extensions"), " ".join(receipt["argv"]))
        self.assertIn("gym-budget-v3.ts", " ".join(receipt["argv"]))
        self.assertEqual(
            receipt["argv"][receipt["argv"].index("--gym-provider-turn-limit") + 1],
            "13",
        )
        first_panel = self.manifest["panels"][0]
        first_task = next(
            task
            for task in self.manifest["tasks"]
            if task["task_id"] == first_panel["task_id"]
        )
        expected_path = (
            f"/{first_task['template']}/{first_task['seed']}/"
            f"{first_task['difficulty']}"
        )
        self.assertTrue(
            any(expected_path in argument for argument in receipt["argv"]),
        )
        self.assertTrue(all(receipt["checks"].values()))

    def test_local_preflight_generates_all_outcome_only_tasks(self) -> None:
        preflight = build_local_preflight(
            self.manifest,
            self.registry,
            PROJECT_ROOT,
        )
        self.assertEqual(preflight["generated_task_count"], EXPECTED_TASKS)
        self.assertEqual(len(preflight["generated_tasks"]), EXPECTED_TASKS)
        self.assertTrue(
            all(
                len(item["commitment_hash"]) == 64
                for item in preflight["generated_tasks"]
            )
        )
        self.assertEqual(preflight["policy_leakage_markers_found"], 0)
        self.assertFalse(preflight["held_templates_consumed"])
        self.assertFalse(preflight["live_model_execution_authorized"])
        validate_local_preflight(
            preflight,
            self.manifest,
            self.registry,
            PROJECT_ROOT,
        )

    def test_local_preflight_rejects_current_source_drift(self) -> None:
        preflight = build_local_preflight(
            self.manifest,
            self.registry,
            PROJECT_ROOT,
        )
        with mock.patch(
            "pyreplab_harness.m3_empty_overlay_baseline.source_tree_hash",
            return_value="f" * 64,
        ):
            with self.assertRaisesRegex(ValueError, "source tree hash"):
                validate_local_preflight(
                    preflight,
                    self.manifest,
                    self.registry,
                    PROJECT_ROOT,
                )

    def test_local_preflight_rejects_rehashed_task_commitment_drift(self) -> None:
        preflight = build_local_preflight(
            self.manifest,
            self.registry,
            PROJECT_ROOT,
        )
        tampered = json.loads(json.dumps(preflight))
        tampered["generated_tasks"][0]["oracle_sha256"] = "f" * 64
        tampered["generated_tasks_sha256"] = _canonical_hash(
            {"tasks": tampered["generated_tasks"]}
        )
        payload = {
            key: value for key, value in tampered.items() if key != "preflight_hash"
        }
        tampered["preflight_hash"] = _canonical_hash(payload)
        with self.assertRaisesRegex(ValueError, "commitments drifted"):
            validate_local_preflight(
                tampered,
                self.manifest,
                self.registry,
                PROJECT_ROOT,
            )

    def test_generic_runner_cannot_bypass_authorization_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry_path = Path(directory) / "registry.json"
            self.registry.save(registry_path)
            args = argparse.Namespace(
                treatment_registry=str(registry_path),
                treatments="all",
                family="unbrowser_fixture",
            )
            with self.assertRaisesRegex(ValueError, "dedicated.*authorized"):
                run_registered_treatments(
                    PROJECT_ROOT,
                    RemoteConfig("host", "/project", "/runs"),
                    args,
                )

    def test_freeze_is_immutable_and_roundtrips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry_path = Path(directory) / "registry.json"
            manifest_path = Path(directory) / "manifest.json"
            first = freeze_baseline_artifacts(
                registry_path,
                manifest_path,
                REMOTE_IDENTITY,
            )
            second = freeze_baseline_artifacts(
                registry_path,
                manifest_path,
                REMOTE_IDENTITY,
            )
            self.assertEqual(first, second)
            restored = TreatmentRegistry.load(registry_path)
            self.assertEqual(restored.registry_hash, self.registry.registry_hash)

    def test_remote_preflight_is_hash_bound_and_does_not_authorize_execution(self) -> None:
        local_preflight = build_local_preflight(
            self.manifest,
            self.registry,
            PROJECT_ROOT,
        )
        lifecycle_payload = {
            "schema_version": "m3-unbrowser-lifecycle-stress-v1",
            "checked_at": "2026-08-14T00:00:00+00:00",
            "wait_seconds": 36.0,
            "elapsed_seconds": 36.1,
            "fixture_url": "http://127.0.0.1:18090/single_page_extraction/2026091001/easy",
            "navigation_status": 200,
            "post_wait_observation_sha256": "a" * 64,
            "runtime_version": "0.0.19",
            "confined": True,
            "same_session": True,
            "passed": True,
        }
        lifecycle = {
            **lifecycle_payload,
            "receipt_hash": _canonical_hash(lifecycle_payload),
        }
        runtime = {
            "source_tree_hash": local_preflight["source_tree_hash"],
            "runtime_pins": json.loads(json.dumps(self.manifest["runtime_pins"])),
        }
        with tempfile.TemporaryDirectory() as directory:
            registry_path = Path(directory) / "registry.json"
            manifest_path = Path(directory) / "manifest.json"
            local_path = Path(directory) / "local.json"
            output_path = Path(directory) / "remote.json"
            self.registry.save(registry_path)
            manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
            local_path.write_text(json.dumps(local_preflight), encoding="utf-8")
            with mock.patch(
                "pyreplab_harness.m3_empty_overlay_baseline.runtime_preflight",
                return_value=runtime,
            ), mock.patch(
                "pyreplab_harness.m3_empty_overlay_baseline._ssh_capture",
                return_value=json.dumps(lifecycle),
            ):
                report = run_remote_preflight(
                    output_path,
                    manifest_path,
                    registry_path,
                    local_path,
                    PROJECT_ROOT,
                    RemoteConfig(**REMOTE_IDENTITY),
                    pi_binary="pi",
                    thinking="off",
                    unbrowser_binary=str(_RUNTIME_PINS["unbrowser_path"]),
                    model_artifact=str(_RUNTIME_PINS["model_artifact_path"]),
                    llama_server_binary=str(_RUNTIME_PINS["llama_server_path"]),
                )
            self.assertTrue(report["ready_for_authorization"])
            self.assertFalse(report["live_model_execution_authorized"])
            remote_preflight = json.loads(output_path.read_text(encoding="utf-8"))
            validate_remote_preflight(
                remote_preflight,
                self.manifest,
                self.registry,
                local_preflight,
                PROJECT_ROOT,
            )
            tampered_local = dict(local_preflight)
            tampered_local["generated_task_count"] = 0
            with self.assertRaisesRegex(ValueError, "preflight_hash"):
                validate_remote_preflight(
                    remote_preflight,
                    self.manifest,
                    self.registry,
                    tampered_local,
                    PROJECT_ROOT,
                )


class LifecycleStressTest(unittest.TestCase):
    def test_rejects_wait_below_historical_boundary(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 35"):
            run_lifecycle_stress("/unbrowser", wait_seconds=34.9)

    def test_uses_one_confined_session_across_wait(self) -> None:
        state = {"closed": False, "server_stopped": False}

        class FakeServer:
            def __init__(self, port: int) -> None:
                self.port = port

            def url_for(self, template: str, seed: int, difficulty: str) -> str:
                return f"http://127.0.0.1:{self.port}/{template}/{seed}/{difficulty}"

            def stop(self) -> None:
                state["server_stopped"] = True

        class FakeSession:
            runtime_version = "0.0.19"

            def __init__(self, *_args, **kwargs) -> None:
                self.kwargs = kwargs
                self.calls = 0
                self.confined = kwargs["confined"]

            def execute(self, request):
                self.calls += 1
                if request["action"] == "navigate":
                    return {"result": {"status": 200}}
                return {"result": "Employee Directory"}

            def close(self) -> None:
                state["closed"] = True

        with mock.patch(
            "pyreplab_harness.m3_empty_overlay_baseline.FixtureServer",
            FakeServer,
        ), mock.patch(
            "pyreplab_harness.m3_empty_overlay_baseline.UnbrowserSession",
            FakeSession,
        ), mock.patch(
            "pyreplab_harness.m3_empty_overlay_baseline.time.sleep"
        ) as sleeper, mock.patch(
            "pyreplab_harness.m3_empty_overlay_baseline.time.monotonic",
            side_effect=[100.0, 136.0],
        ):
            receipt = run_lifecycle_stress("/unbrowser", wait_seconds=36)

        sleeper.assert_called_once_with(36)
        self.assertTrue(receipt["same_session"])
        self.assertTrue(receipt["confined"])
        self.assertTrue(state["closed"])
        self.assertTrue(state["server_stopped"])
        _validate_lifecycle_receipt(receipt)

    def test_receipt_rejects_declared_wait_without_measured_elapsed_time(self) -> None:
        payload = {
            "schema_version": "m3-unbrowser-lifecycle-stress-v1",
            "checked_at": "2026-08-14T00:00:00+00:00",
            "wait_seconds": 36.0,
            "elapsed_seconds": 0.1,
            "fixture_url": "http://127.0.0.1:18090/example",
            "navigation_status": 200,
            "post_wait_observation_sha256": "a" * 64,
            "runtime_version": "0.0.19",
            "confined": True,
            "same_session": True,
            "passed": True,
        }
        receipt = {**payload, "receipt_hash": _canonical_hash(payload)}
        with self.assertRaisesRegex(ValueError, "elapsed"):
            _validate_lifecycle_receipt(receipt)


if __name__ == "__main__":
    unittest.main()
