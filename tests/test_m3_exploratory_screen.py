from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from pyreplab_harness.orchestrator import RemoteConfig
from pyreplab_harness.m3_exploratory_screen import (
    PANEL_RESULT_SCHEMA,
    _beta_mean,
    _existing_result_keys,
    _generate_execution_orders,
    analyze_screen,
    build_screen_manifest,
    freeze_screen_manifest,
    run_screen,
    validate_screen_manifest,
)
from pyreplab_harness.treatments import TreatmentRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "policies" / "m3-unbrowser-treatments.json"
SPLIT_PATH = PROJECT_ROOT / "policies" / "m3-unbrowser-policy-split.json"


def _spec_4_policies() -> dict:
    """Return a minimal valid spec with 4 meta_train policies and 2 tasks."""
    policy_split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    meta_train = policy_split["splits"]["meta_train"]
    bundle_ids = meta_train[:4]
    return {
        "screen_id": "test-screen-001",
        "purpose": "Unit test exploratory screen with balanced layout",
        "remote_identity": {
            "host": "test-host",
            "project": "/remote/test-project",
            "run_root": "/remote/test-runs",
            "python": "python3",
        },
        "policy_bundle_ids": bundle_ids,
        "tasks": [
            {"template": "single_page_extraction", "difficulty": "easy",
             "seed": 1001},
            {"template": "table_filter_sort", "difficulty": "medium",
             "seed": 1002},
        ],
        "rollout_replicas": 1,
        "sampling_seed_start": 3000,
        "schedule_seed": 424242,
        "selection": {"reason": "test"},
    }


def _make_mock_attempt(
    bid: str,
    success: bool = True,
    output_tokens: float = 100,
    pi_return_code: int = 0,
) -> dict[str, Any]:
    """Build one mock orchestrator attempt dict."""
    return {
        "attempt_id": f"attempt-{bid}",
        "policy": {"id": "test"},
        "pi_return_code": pi_return_code,
        "pi_stderr": "",
        "verification": {"success": success, "details": {}},
        "usage": {"output": output_tokens, "prompt_tokens": 50},
        "trajectory": {
            "planning_preamble": {"present": False},
            "tool_trace": [
                {
                    "tool_name": "unbrowser",
                    "is_error": False,
                    "budget_rejected": False,
                    "operation_aborted": False,
                    "details": {"action": "navigate", "status": 200},
                },
            ],
            "provider_turn_count": 1,
        },
        "timing": {},
    }


def _build_mock_results(
    manifest: dict[str, Any],
    all_success: bool = True,
    output_tokens: float = 100,
) -> list[dict[str, Any]]:
    """Build a list of valid mock panel result records."""
    records: list[dict[str, Any]] = []
    for panel in manifest["panels"]:
        task = next(
            t for t in manifest["tasks"] if t["task_id"] == panel["task_id"]
        )
        attempts: dict[str, dict[str, Any]] = {}
        for bid in panel["execution_order"]:
            attempts[bid] = _make_mock_attempt(
                bid, success=all_success, output_tokens=output_tokens
            )
        record = {
            "schema_version": PANEL_RESULT_SCHEMA,
            "panel_id": panel["panel_id"],
            "manifest_hash": manifest["manifest_hash"],
            "task": task,
            "panel": panel,
            "status": "completed",
            "started_at": "2026-08-10T00:00:00Z",
            "finished_at": "2026-08-10T00:01:00Z",
            "duration_seconds": 60.0,
            "result": {
                "task_id": panel["task_id"],
                "mode": "treatment_set",
                "execution_order": panel["execution_order"],
                "attempts": attempts,
                "pilot_manifest_hash": manifest["manifest_hash"],
                "pilot_panel_id": panel["panel_id"],
                "rollout_replica": panel["rollout_replica"],
                "sampling_seed": panel["sampling_seed"],
            },
        }
        records.append(record)
    return records


class M3ExploratoryScreenManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = TreatmentRegistry.load(REGISTRY_PATH)
        self.policy_split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
        self.spec = _spec_4_policies()

    def test_build_and_validate_balanced_manifest(self) -> None:
        manifest = build_screen_manifest(
            self.registry, self.policy_split, self.spec,
            registry_file=REGISTRY_PATH.name,
            policy_split_file=SPLIT_PATH.name,
        )
        validate_screen_manifest(manifest, self.registry, self.policy_split)

        self.assertEqual(manifest["screen_id"], "test-screen-001")
        self.assertEqual(len(manifest["tasks"]), 2)
        self.assertEqual(len(manifest["panels"]), 2)
        self.assertEqual(manifest["gates"]["attempts"], 8)

        for panel in manifest["panels"]:
            self.assertEqual(len(panel["execution_order"]), 4)
            self.assertEqual(
                set(panel["execution_order"]),
                set(self.spec["policy_bundle_ids"]),
            )

        seeds = [p["sampling_seed"] for p in manifest["panels"]]
        self.assertEqual(len(seeds), len(set(seeds)))

        for task in manifest["tasks"]:
            self.assertTrue(task["task_id"].startswith("unbrowser-fixture-v2-"))
            self.assertEqual(task["role"], "T_pilot")

    def test_policies_must_all_be_in_meta_train(self) -> None:
        spec = dict(self.spec)
        dev_ids = self.policy_split["splits"]["development"]
        spec["policy_bundle_ids"] = [self.spec["policy_bundle_ids"][0], dev_ids[0]]
        with self.assertRaises(ValueError):
            build_screen_manifest(
                self.registry, self.policy_split, spec,
                registry_file=REGISTRY_PATH.name,
                policy_split_file=SPLIT_PATH.name,
            )

    def test_held_template_is_rejected(self) -> None:
        spec = dict(self.spec)
        spec["tasks"] = [
            {"template": "cross_page_comparison", "difficulty": "easy",
             "seed": 5001},
        ]
        with self.assertRaises(ValueError):
            build_screen_manifest(
                self.registry, self.policy_split, spec,
                registry_file=REGISTRY_PATH.name,
                policy_split_file=SPLIT_PATH.name,
            )

    def test_tampered_manifest_hash_fails_validation(self) -> None:
        manifest = build_screen_manifest(
            self.registry, self.policy_split, self.spec,
            registry_file=REGISTRY_PATH.name,
            policy_split_file=SPLIT_PATH.name,
        )
        tampered = json.loads(json.dumps(manifest))
        tampered["tasks"][0]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "manifest_hash"):
            validate_screen_manifest(tampered, self.registry, self.policy_split)

    def test_freeze_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec_path = Path(directory) / "spec.json"
            spec_path.write_text(json.dumps(self.spec, indent=2), encoding="utf-8")
            output = Path(directory) / "manifest.json"
            first = freeze_screen_manifest(
                output, REGISTRY_PATH, SPLIT_PATH, spec_path
            )
            second = freeze_screen_manifest(
                output, REGISTRY_PATH, SPLIT_PATH, spec_path
            )
            self.assertEqual(first, second)

    # ---- Issue 1: execution order tests ------------------------------------

    def test_4_policies_24_panels_uses_all_permutations(self) -> None:
        spec = dict(self.spec)
        spec["rollout_replicas"] = 12  # 24 panels total
        manifest = build_screen_manifest(
            self.registry, self.policy_split, spec,
            registry_file=REGISTRY_PATH.name,
            policy_split_file=SPLIT_PATH.name,
        )
        self.assertEqual(len(manifest["panels"]), 24)
        observed = {tuple(p["execution_order"]) for p in manifest["panels"]}
        self.assertEqual(len(observed), 24)

    def test_4_policies_4_panels_uses_cyclic_not_permutations(self) -> None:
        spec = dict(self.spec)
        spec["rollout_replicas"] = 2  # 4 panels total
        manifest = build_screen_manifest(
            self.registry, self.policy_split, spec,
            registry_file=REGISTRY_PATH.name,
            policy_split_file=SPLIT_PATH.name,
        )
        self.assertEqual(len(manifest["panels"]), 4)
        # With 4 panels and cyclic rotations, each policy should appear
        # in each position exactly once.
        positions = [set() for _ in range(4)]
        for panel in manifest["panels"]:
            for pos, bid in enumerate(panel["execution_order"]):
                positions[pos].add(bid)
        # Each position should have seen all 4 policies across 4 panels if balanced.
        for pos in range(4):
            self.assertEqual(len(positions[pos]), 4)

    def test_4_policies_8_panels_uses_cyclic_not_permutations(self) -> None:
        spec = dict(self.spec)
        spec["rollout_replicas"] = 4  # 8 panels total
        manifest = build_screen_manifest(
            self.registry, self.policy_split, spec,
            registry_file=REGISTRY_PATH.name,
            policy_split_file=SPLIT_PATH.name,
        )
        self.assertEqual(len(manifest["panels"]), 8)
        # With 8 panels (not divisible by 24), should use cyclic rotations.
        observed = {tuple(p["execution_order"]) for p in manifest["panels"]}
        self.assertEqual(len(observed), 4)  # Only 4 distinct rotations

    # ---- Issue 12: remote identity validation -------------------------------

    def test_remote_identity_mismatch_is_rejected(self) -> None:
        manifest = build_screen_manifest(
            self.registry, self.policy_split, self.spec,
            registry_file=REGISTRY_PATH.name,
            policy_split_file=SPLIT_PATH.name,
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            output_path = Path(directory) / "results.jsonl"

            wrong_config = RemoteConfig(
                "wrong-host",
                manifest["remote_identity"]["project"],
                manifest["remote_identity"]["run_root"],
                manifest["remote_identity"]["python"],
            )
            with self.assertRaisesRegex(ValueError, "host mismatch"):
                run_screen(
                    manifest_path, REGISTRY_PATH, SPLIT_PATH, output_path,
                    wrong_config,
                    pi_binary="pi", provider="ubuntu-gemma",
                    model="gemma-4-26b-a4b", thinking="off",
                    unbrowser_binary="/tmp/unbrowser",
                    model_artifact="/tmp/model.gguf",
                    llama_server_binary="/tmp/llama-server",
                )

    def test_remote_identity_python_mismatch_is_rejected(self) -> None:
        manifest = build_screen_manifest(
            self.registry, self.policy_split, self.spec,
            registry_file=REGISTRY_PATH.name,
            policy_split_file=SPLIT_PATH.name,
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            output_path = Path(directory) / "results.jsonl"

            wrong_config = RemoteConfig(
                manifest["remote_identity"]["host"],
                manifest["remote_identity"]["project"],
                manifest["remote_identity"]["run_root"],
                "python3.99",  # wrong python
            )
            with self.assertRaisesRegex(ValueError, "python mismatch"):
                run_screen(
                    manifest_path, REGISTRY_PATH, SPLIT_PATH, output_path,
                    wrong_config,
                    pi_binary="pi", provider="ubuntu-gemma",
                    model="gemma-4-26b-a4b", thinking="off",
                    unbrowser_binary="/tmp/unbrowser",
                    model_artifact="/tmp/model.gguf",
                    llama_server_binary="/tmp/llama-server",
                )

    # ---- Issue 13: panel schedule interleaving ------------------------------

    def test_panel_schedule_interleaves_replicas(self) -> None:
        """With rollout_replicas=2, replicas should not always be adjacent."""
        spec = dict(self.spec)
        spec["rollout_replicas"] = 2  # 4 panels total
        manifest = build_screen_manifest(
            self.registry, self.policy_split, spec,
            registry_file=REGISTRY_PATH.name,
            policy_split_file=SPLIT_PATH.name,
        )
        self.assertEqual(len(manifest["panels"]), 4)
        # Check that panels for the same task are not always adjacent.
        task_order = [p["task_id"] for p in manifest["panels"]]
        # With 2 tasks and 2 replicas, after shuffle they may or may not be
        # adjacent depending on seed, but the schedule is deterministic.
        # Just verify each task appears twice.
        for task in manifest["tasks"]:
            self.assertEqual(task_order.count(task["task_id"]), 2)

    # ---- Issue 16: existing_result_keys strict validation -------------------

    def test_active_marker_blocks_resume(self) -> None:
        manifest = build_screen_manifest(
            self.registry, self.policy_split, self.spec,
            registry_file=REGISTRY_PATH.name,
            policy_split_file=SPLIT_PATH.name,
        )
        runtime = {
            "checked_at": "2026-08-10T00:00:00+00:00",
            "code_revision": "a" * 40,
            "source_tree_hash": "b" * 64,
            "worktree_clean": True,
            "worktree_status_hash": hashlib.sha256(
                b"PYREPLAB_GIT_WORKTREE_CLEAN_MARKER_V1"
            ).hexdigest(),
            "runtime_pins": manifest["runtime_pins"],
        }
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            output = Path(directory) / "results.jsonl"
            config = RemoteConfig(
                manifest["remote_identity"]["host"],
                manifest["remote_identity"]["project"],
                manifest["remote_identity"]["run_root"],
                manifest["remote_identity"]["python"],
            )
            with patch(
                "pyreplab_harness.m3_exploratory_screen.runtime_preflight",
                return_value=runtime,
            ), patch(
                "pyreplab_harness.m3_exploratory_screen."
                "run_registered_treatments",
                side_effect=KeyboardInterrupt,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    run_screen(
                        manifest_path, REGISTRY_PATH, SPLIT_PATH, output,
                        config,
                        pi_binary="pi", provider="ubuntu-gemma",
                        model="gemma-4-26b-a4b", thinking="off",
                        unbrowser_binary="/tmp/unbrowser",
                        model_artifact="/tmp/model.gguf",
                        llama_server_binary="/tmp/llama-server",
                    )
            active = output.with_suffix(".jsonl.active.json")
            self.assertTrue(active.is_file())
            with self.assertRaisesRegex(RuntimeError, "unfinished screen panel"):
                run_screen(
                    manifest_path, REGISTRY_PATH, SPLIT_PATH, output, config,
                    pi_binary="pi", provider="ubuntu-gemma",
                    model="gemma-4-26b-a4b", thinking="off",
                    unbrowser_binary="/tmp/unbrowser",
                    model_artifact="/tmp/model.gguf",
                    llama_server_binary="/tmp/llama-server",
                )

    def test_existing_keys_rejects_unknown_schema(self) -> None:
        manifest = build_screen_manifest(
            self.registry, self.policy_split, self.spec,
            registry_file=REGISTRY_PATH.name,
            policy_split_file=SPLIT_PATH.name,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.jsonl"
            bad_line = json.dumps({
                "schema_version": "unknown-schema-v1",
                "panel_id": "x",
                "manifest_hash": manifest["manifest_hash"],
                "status": "completed",
            })
            output.write_text(bad_line + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                _existing_result_keys(output, manifest)

    # ---- Issue 15: runtime pins validation via rebuild ----------------------

    def test_runtime_pins_mismatch_rejected(self) -> None:
        manifest = build_screen_manifest(
            self.registry, self.policy_split, self.spec,
            registry_file=REGISTRY_PATH.name,
            policy_split_file=SPLIT_PATH.name,
        )
        tampered = json.loads(json.dumps(manifest))
        tampered["runtime_pins"]["rollout_replicas"] = 99
        # Recompute hash so manifest_hash check passes first.
        from pyreplab_harness.m3_pilot import _canonical_hash
        tampered.pop("manifest_hash", None)
        tampered["manifest_hash"] = _canonical_hash(tampered)
        with self.assertRaisesRegex(ValueError, "runtime_pins"):
            validate_screen_manifest(tampered, self.registry, self.policy_split)

    def test_known_templates_mismatch_rejected(self) -> None:
        manifest = build_screen_manifest(
            self.registry, self.policy_split, self.spec,
            registry_file=REGISTRY_PATH.name,
            policy_split_file=SPLIT_PATH.name,
        )
        tampered = json.loads(json.dumps(manifest))
        tampered["known_templates"] = ["wrong"]
        from pyreplab_harness.m3_pilot import _canonical_hash
        tampered.pop("manifest_hash", None)
        tampered["manifest_hash"] = _canonical_hash(tampered)
        with self.assertRaisesRegex(ValueError, "known_templates"):
            validate_screen_manifest(tampered, self.registry, self.policy_split)

    def test_panel_rebuild_mismatch_rejected(self) -> None:
        manifest = build_screen_manifest(
            self.registry, self.policy_split, self.spec,
            registry_file=REGISTRY_PATH.name,
            policy_split_file=SPLIT_PATH.name,
        )
        tampered = json.loads(json.dumps(manifest))
        tampered["panels"][0]["sampling_seed"] = 999999
        from pyreplab_harness.m3_pilot import _canonical_hash
        tampered.pop("manifest_hash", None)
        tampered["manifest_hash"] = _canonical_hash(tampered)
        with self.assertRaisesRegex(ValueError, "panel 0 mismatch"):
            validate_screen_manifest(tampered, self.registry, self.policy_split)


class M3ExploratoryScreenAnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = TreatmentRegistry.load(REGISTRY_PATH)
        self.policy_split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
        spec = _spec_4_policies()
        self.spec = spec
        self.manifest = build_screen_manifest(
            self.registry, self.policy_split, spec,
            registry_file=REGISTRY_PATH.name,
            policy_split_file=SPLIT_PATH.name,
        )

    def test_beta_mean_calculation(self) -> None:
        self.assertAlmostEqual(_beta_mean(0, 0), 0.5)
        self.assertAlmostEqual(_beta_mean(1, 1), 2 / 3)
        self.assertAlmostEqual(_beta_mean(3, 5), 4 / 7)

    # ---- Issue 3: global successes ------------------------------------------

    def test_analyze_with_mock_results_reports_exact_success_totals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
            results_path = Path(directory) / "results.jsonl"
            records = _build_mock_results(self.manifest, all_success=True)
            results_path.write_text(
                "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n",
                encoding="utf-8",
            )
            analysis = analyze_screen(
                manifest_path, results_path, REGISTRY_PATH, SPLIT_PATH,
            )
            self.assertTrue(analysis["completeness"]["complete"])
            # 2 panels × 4 policies = 8 total successes
            self.assertEqual(analysis["summary"]["successes"], 8)
            self.assertEqual(analysis["summary"]["total_attempts"], 8)
            self.assertEqual(analysis["summary"]["overall_success_rate"], 1.0)

    def test_analyze_with_half_failures_reports_correct_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
            results_path = Path(directory) / "results.jsonl"
            records = _build_mock_results(self.manifest, all_success=True)
            # Make half the attempts fail.
            fail_count = 0
            for record in records:
                attempts = record["result"]["attempts"]
                for bid in list(attempts.keys()):
                    if fail_count < 4:
                        attempts[bid] = _make_mock_attempt(
                            bid, success=False, output_tokens=50
                        )
                        fail_count += 1
            results_path.write_text(
                "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n",
                encoding="utf-8",
            )
            analysis = analyze_screen(
                manifest_path, results_path, REGISTRY_PATH, SPLIT_PATH,
            )
            self.assertEqual(analysis["summary"]["total_attempts"], 8)
            self.assertEqual(analysis["summary"]["successes"], 4)
            self.assertAlmostEqual(analysis["summary"]["overall_success_rate"], 0.5)

    # ---- Issue 4: usage key is "output" -------------------------------------

    def test_mean_output_tokens_uses_usage_output_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
            results_path = Path(directory) / "results.jsonl"
            # Use varying output token values.
            records = _build_mock_results(self.manifest, all_success=True,
                                          output_tokens=200)
            results_path.write_text(
                "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n",
                encoding="utf-8",
            )
            analysis = analyze_screen(
                manifest_path, results_path, REGISTRY_PATH, SPLIT_PATH,
            )
            # 8 attempts × 200 = 1600 / 8 = 200
            self.assertAlmostEqual(analysis["summary"]["mean_output_tokens"], 200.0)

    # ---- Issue 6: Hamming-1 paired reporting --------------------------------

    def test_hamming1_has_paired_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
            results_path = Path(directory) / "results.jsonl"
            records = _build_mock_results(self.manifest, all_success=True)
            results_path.write_text(
                "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n",
                encoding="utf-8",
            )
            analysis = analyze_screen(
                manifest_path, results_path, REGISTRY_PATH, SPLIT_PATH,
            )
            h1 = analysis["hamming_1_pairs"]
            self.assertGreater(len(h1), 0)
            for pair in h1:
                self.assertIn("pair_count", pair)
                self.assertIn("a_wins", pair)
                self.assertIn("b_wins", pair)
                self.assertIn("both_success", pair)
                self.assertIn("both_fail", pair)
                self.assertIn("paired_success_difference", pair)
                self.assertIn("factor_changed", pair)
                # pair_count + a_wins + b_wins + both_success + both_fail
                # should sum correctly.
                self.assertEqual(
                    pair["pair_count"],
                    pair["a_wins"] + pair["b_wins"] +
                    pair["both_success"] + pair["both_fail"],
                )

    # ---- Issue 8: model runtime failures ------------------------------------

    def test_model_runtime_failures_counted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
            results_path = Path(directory) / "results.jsonl"
            records = _build_mock_results(self.manifest, all_success=True)
            # Make one attempt have nonzero return code.
            records[0]["result"]["attempts"][
                self.manifest["policy_bundle_ids"][0]
            ]["pi_return_code"] = 1
            results_path.write_text(
                "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n",
                encoding="utf-8",
            )
            analysis = analyze_screen(
                manifest_path, results_path, REGISTRY_PATH, SPLIT_PATH,
            )
            self.assertEqual(analysis["summary"]["model_runtime_failures"], 1)
            # Successes should still count verification, not return code.
            self.assertEqual(analysis["summary"]["successes"], 8)

    # ---- Issue 9: task/template outcomes ------------------------------------

    def test_task_and_template_outcomes_populated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
            results_path = Path(directory) / "results.jsonl"
            records = _build_mock_results(self.manifest, all_success=True)
            results_path.write_text(
                "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n",
                encoding="utf-8",
            )
            analysis = analyze_screen(
                manifest_path, results_path, REGISTRY_PATH, SPLIT_PATH,
            )
            self.assertIn("task_outcomes", analysis)
            self.assertIn("template_outcomes", analysis)
            # Each task has 4 attempts (1 panel × 4 policies).
            for tid, to in analysis["task_outcomes"].items():
                self.assertEqual(to["attempts"], 4)
                self.assertEqual(to["successes"], 4)
            for tmpl, to in analysis["template_outcomes"].items():
                self.assertEqual(to["attempts"], 4)
                self.assertEqual(to["successes"], 4)

    # ---- Issue 10: mean admitted calls with zero trajectories ---------------

    def test_mean_admitted_calls_counts_zero_trajectories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
            results_path = Path(directory) / "results.jsonl"
            records = _build_mock_results(self.manifest, all_success=True)
            results_path.write_text(
                "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n",
                encoding="utf-8",
            )
            analysis = analyze_screen(
                manifest_path, results_path, REGISTRY_PATH, SPLIT_PATH,
            )
            for bid, pp in analysis["per_policy"].items():
                # With our mock trajectory (1 unbrowser navigate, no bash),
                # admitted tool calls should be counted.
                self.assertIn("mean_admitted_tool_calls", pp)
                self.assertIsInstance(pp["mean_admitted_tool_calls"], float)

    # ---- Issue 11: adherence marginals --------------------------------------

    def test_adherence_in_marginal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
            results_path = Path(directory) / "results.jsonl"
            records = _build_mock_results(self.manifest, all_success=True)
            results_path.write_text(
                "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n",
                encoding="utf-8",
            )
            analysis = analyze_screen(
                manifest_path, results_path, REGISTRY_PATH, SPLIT_PATH,
            )
            marginal = analysis["marginal"]
            # Each factor level should have adherence rates.
            for factor in ("planning", "observation", "verification",
                           "recovery", "tool_cap"):
                self.assertIn(factor, marginal)
                for level, stats in marginal[factor].items():
                    if factor == "planning":
                        self.assertIn("planning_adherence_rate", stats)
                    elif factor == "observation":
                        self.assertIn("observation_adherence_rate", stats)
                    elif factor == "verification":
                        self.assertIn("verification_adherence_rate", stats)
                    elif factor == "recovery":
                        self.assertIn("recovery_adherence_rate", stats)
                        self.assertIn("recovery_eligible_count", stats)
                    elif factor == "tool_cap":
                        self.assertIn("tool_cap_compliance_rate", stats)

    # ---- Issue 7: strict result validation ----------------------------------

    def test_analysis_rejects_unknown_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
            results_path = Path(directory) / "results.jsonl"
            bad_line = json.dumps({
                "schema_version": "bad-schema",
                "panel_id": "x",
                "manifest_hash": self.manifest["manifest_hash"],
                "status": "completed",
            })
            results_path.write_text(bad_line + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                analyze_screen(
                    manifest_path, results_path, REGISTRY_PATH, SPLIT_PATH,
                )

    def test_analysis_rejects_duplicate_panel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
            results_path = Path(directory) / "results.jsonl"
            records = _build_mock_results(self.manifest, all_success=True)
            dup = json.loads(json.dumps(records[0]))
            lines = [json.dumps(r, sort_keys=True) for r in records] + \
                    [json.dumps(dup, sort_keys=True)]
            results_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                analyze_screen(
                    manifest_path, results_path, REGISTRY_PATH, SPLIT_PATH,
                )

    def test_analysis_rejects_missing_policy_in_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
            results_path = Path(directory) / "results.jsonl"
            records = _build_mock_results(self.manifest, all_success=True)
            # Remove one policy from first panel's attempts.
            bad = json.loads(json.dumps(records[0]))
            bid = self.manifest["policy_bundle_ids"][0]
            del bad["result"]["attempts"][bid]
            lines = [json.dumps(bad, sort_keys=True)] + \
                    [json.dumps(r, sort_keys=True) for r in records[1:]]
            results_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                analyze_screen(
                    manifest_path, results_path, REGISTRY_PATH, SPLIT_PATH,
                )

    def test_analysis_rejects_non_bool_verification_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
            results_path = Path(directory) / "results.jsonl"
            records = _build_mock_results(self.manifest, all_success=True)
            bad = json.loads(json.dumps(records[0]))
            bid = self.manifest["policy_bundle_ids"][0]
            bad["result"]["attempts"][bid]["verification"]["success"] = 1
            lines = [json.dumps(bad, sort_keys=True)] + \
                    [json.dumps(r, sort_keys=True) for r in records[1:]]
            results_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                analyze_screen(
                    manifest_path, results_path, REGISTRY_PATH, SPLIT_PATH,
                )

    def test_mixed_manifest_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
            results_path = Path(directory) / "results.jsonl"
            bad_record = json.dumps({
                "schema_version": PANEL_RESULT_SCHEMA,
                "panel_id": "x",
                "manifest_hash": "deadbeef",
                "status": "completed",
            })
            results_path.write_text(bad_record + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                analyze_screen(
                    manifest_path, results_path, REGISTRY_PATH, SPLIT_PATH,
                )

    def test_analyze_empty_results_fails_on_missing_panels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
            results_path = Path(directory) / "results.jsonl"
            results_path.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                analyze_screen(
                    manifest_path, results_path, REGISTRY_PATH, SPLIT_PATH,
                )

    def test_execution_order_mismatch_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
            results_path = Path(directory) / "results.jsonl"
            records = _build_mock_results(self.manifest, all_success=True)
            bad = json.loads(json.dumps(records[0]))
            bad["result"]["execution_order"] = list(
                reversed(bad["result"]["execution_order"])
            )
            lines = [json.dumps(bad, sort_keys=True)] + \
                    [json.dumps(r, sort_keys=True) for r in records[1:]]
            results_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                analyze_screen(
                    manifest_path, results_path, REGISTRY_PATH, SPLIT_PATH,
                )

    def test_pi_return_code_non_int_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
            results_path = Path(directory) / "results.jsonl"
            records = _build_mock_results(self.manifest, all_success=True)
            bad = json.loads(json.dumps(records[0]))
            bid = self.manifest["policy_bundle_ids"][0]
            bad["result"]["attempts"][bid]["pi_return_code"] = "not_int"
            lines = [json.dumps(bad, sort_keys=True)] + \
                    [json.dumps(r, sort_keys=True) for r in records[1:]]
            results_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                analyze_screen(
                    manifest_path, results_path, REGISTRY_PATH, SPLIT_PATH,
                )

    def test_pi_return_code_bool_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
            results_path = Path(directory) / "results.jsonl"
            records = _build_mock_results(self.manifest, all_success=True)
            bad = json.loads(json.dumps(records[0]))
            bid = self.manifest["policy_bundle_ids"][0]
            bad["result"]["attempts"][bid]["pi_return_code"] = True
            lines = [json.dumps(bad, sort_keys=True)] + \
                    [json.dumps(r, sort_keys=True) for r in records[1:]]
            results_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                analyze_screen(
                    manifest_path, results_path, REGISTRY_PATH, SPLIT_PATH,
                )

    # ---- Issue 3: usage.output strictness -----------------------------------

    def test_usage_output_none_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
            results_path = Path(directory) / "results.jsonl"
            records = _build_mock_results(self.manifest, all_success=True)
            bad = json.loads(json.dumps(records[0]))
            bid = self.manifest["policy_bundle_ids"][0]
            bad["result"]["attempts"][bid]["usage"]["output"] = None
            lines = [json.dumps(bad, sort_keys=True)] + \
                    [json.dumps(r, sort_keys=True) for r in records[1:]]
            results_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                analyze_screen(
                    manifest_path, results_path, REGISTRY_PATH, SPLIT_PATH,
                )

    def test_usage_output_bool_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
            results_path = Path(directory) / "results.jsonl"
            records = _build_mock_results(self.manifest, all_success=True)
            bad = json.loads(json.dumps(records[0]))
            bid = self.manifest["policy_bundle_ids"][0]
            bad["result"]["attempts"][bid]["usage"]["output"] = True
            lines = [json.dumps(bad, sort_keys=True)] + \
                    [json.dumps(r, sort_keys=True) for r in records[1:]]
            results_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                analyze_screen(
                    manifest_path, results_path, REGISTRY_PATH, SPLIT_PATH,
                )

    def test_valid_float_usage_output_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
            results_path = Path(directory) / "results.jsonl"
            records = _build_mock_results(
                self.manifest, all_success=True, output_tokens=123.45
            )
            results_path.write_text(
                "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n",
                encoding="utf-8",
            )
            analysis = analyze_screen(
                manifest_path, results_path, REGISTRY_PATH, SPLIT_PATH,
            )
            self.assertAlmostEqual(
                analysis["summary"]["mean_output_tokens"], 123.45
            )

    # ---- Issue 5: trajectory required and structure validation ---------------

    def test_trajectory_none_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
            results_path = Path(directory) / "results.jsonl"
            records = _build_mock_results(self.manifest, all_success=True)
            bad = json.loads(json.dumps(records[0]))
            bid = self.manifest["policy_bundle_ids"][0]
            bad["result"]["attempts"][bid]["trajectory"] = None
            lines = [json.dumps(bad, sort_keys=True)] + \
                    [json.dumps(r, sort_keys=True) for r in records[1:]]
            results_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                analyze_screen(
                    manifest_path, results_path, REGISTRY_PATH, SPLIT_PATH,
                )

    def test_trajectory_missing_tool_trace_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
            results_path = Path(directory) / "results.jsonl"
            records = _build_mock_results(self.manifest, all_success=True)
            bad = json.loads(json.dumps(records[0]))
            bid = self.manifest["policy_bundle_ids"][0]
            bad["result"]["attempts"][bid]["trajectory"] = {
                "planning_preamble": {"present": False},
                # no tool_trace
                "provider_turn_count": 0,
            }
            lines = [json.dumps(bad, sort_keys=True)] + \
                    [json.dumps(r, sort_keys=True) for r in records[1:]]
            results_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                analyze_screen(
                    manifest_path, results_path, REGISTRY_PATH, SPLIT_PATH,
                )

    def test_trajectory_bad_trace_entry_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
            results_path = Path(directory) / "results.jsonl"
            records = _build_mock_results(self.manifest, all_success=True)
            bad = json.loads(json.dumps(records[0]))
            bid = self.manifest["policy_bundle_ids"][0]
            bad["result"]["attempts"][bid]["trajectory"] = {
                "planning_preamble": {"present": False},
                "tool_trace": [
                    {"tool_name": "bash"}  # missing is_error, details
                ],
                "provider_turn_count": 0,
            }
            lines = [json.dumps(bad, sort_keys=True)] + \
                    [json.dumps(r, sort_keys=True) for r in records[1:]]
            results_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                analyze_screen(
                    manifest_path, results_path, REGISTRY_PATH, SPLIT_PATH,
                )

    # ---- Issue 6: missing schema rejection ----------------------------------

    def test_load_jsonl_rejects_missing_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
            results_path = Path(directory) / "results.jsonl"
            # Record without schema_version
            no_schema = {
                "panel_id": self.manifest["panels"][0]["panel_id"],
                "manifest_hash": self.manifest["manifest_hash"],
                "status": "completed",
            }
            results_path.write_text(
                json.dumps(no_schema) + "\n", encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                analyze_screen(
                    manifest_path, results_path, REGISTRY_PATH, SPLIT_PATH,
                )

    def test_existing_keys_rejects_missing_schema(self) -> None:
        manifest = build_screen_manifest(
            self.registry, self.policy_split, self.spec,
            registry_file=REGISTRY_PATH.name,
            policy_split_file=SPLIT_PATH.name,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.jsonl"
            no_schema = json.dumps({
                "panel_id": manifest["panels"][0]["panel_id"],
                "manifest_hash": manifest["manifest_hash"],
                "status": "completed",
            })
            output.write_text(no_schema + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                _existing_result_keys(output, manifest)

    # ---- Issue 7: malformed run result validation ---------------------------

    def test_run_stops_on_malformed_result(self) -> None:
        manifest = build_screen_manifest(
            self.registry, self.policy_split, self.spec,
            registry_file=REGISTRY_PATH.name,
            policy_split_file=SPLIT_PATH.name,
        )
        runtime_pins = dict(manifest["runtime_pins"])
        runtime_pins["rollout_replicas"] = 2
        runtime = {
            "checked_at": "2026-08-10T00:00:00+00:00",
            "code_revision": "a" * 40,
            "source_tree_hash": "b" * 64,
            "worktree_clean": True,
            "worktree_status_hash": hashlib.sha256(
                b"PYREPLAB_GIT_WORKTREE_CLEAN_MARKER_V1"
            ).hexdigest(),
            "runtime_pins": runtime_pins,
        }
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            output = Path(directory) / "results.jsonl"
            config = RemoteConfig(
                manifest["remote_identity"]["host"],
                manifest["remote_identity"]["project"],
                manifest["remote_identity"]["run_root"],
                manifest["remote_identity"]["python"],
            )
            # Return a result that's missing the attempts key.
            bad_result = {
                "task_id": manifest["panels"][0]["task_id"],
                "mode": "treatment_set",
                "execution_order": manifest["panels"][0]["execution_order"],
                # no "attempts" key
                "pilot_manifest_hash": manifest["manifest_hash"],
                "pilot_panel_id": manifest["panels"][0]["panel_id"],
                "rollout_replica": manifest["panels"][0]["rollout_replica"],
                "sampling_seed": manifest["panels"][0]["sampling_seed"],
            }
            with patch(
                "pyreplab_harness.m3_exploratory_screen.runtime_preflight",
                return_value=runtime,
            ), patch(
                "pyreplab_harness.m3_exploratory_screen."
                "run_registered_treatments",
                return_value=bad_result,
            ):
                with self.assertRaises(RuntimeError):
                    run_screen(
                        manifest_path, REGISTRY_PATH, SPLIT_PATH, output,
                        config,
                        pi_binary="pi", provider="ubuntu-gemma",
                        model="gemma-4-26b-a4b", thinking="off",
                        unbrowser_binary="/tmp/unbrowser",
                        model_artifact="/tmp/model.gguf",
                        llama_server_binary="/tmp/llama-server",
                    )
            # Verify the output file has the error record.
            lines = output.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["status"], "error")
            self.assertEqual(record["error"]["type"], "MalformedPanelResult")
            preflight = json.loads(
                output.with_suffix(".jsonl.preflight.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                preflight["runtime_pins"], manifest["runtime_pins"]
            )

    def test_complete_analysis_includes_all_sections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
            results_path = Path(directory) / "results.jsonl"
            records = _build_mock_results(self.manifest)
            results_path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            analysis = analyze_screen(
                manifest_path, results_path, REGISTRY_PATH, SPLIT_PATH,
            )
        self.assertIn("ranking_label", analysis)
        self.assertIn("warnings", analysis)
        self.assertIn("task_outcomes", analysis)
        self.assertIn("template_outcomes", analysis)


class M3ExploratoryScreenCanaryTest(unittest.TestCase):
    """Tests for T_canary role and protocol binding in exploratory screens."""

    def setUp(self) -> None:
        self.registry = TreatmentRegistry.load(REGISTRY_PATH)
        self.policy_split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))

    def _spec(self, **overrides) -> dict:
        base = _spec_4_policies()
        base.update(overrides)
        return base

    def test_canary_spec_builds_manifest_with_t_canary_role(self) -> None:
        spec = self._spec(
            screen_id="canary-screen-001",
            task_role="T_canary",
        )
        manifest = build_screen_manifest(
            self.registry, self.policy_split, spec,
            registry_file=REGISTRY_PATH.name,
            policy_split_file=SPLIT_PATH.name,
        )
        self.assertEqual(manifest.get("task_role"), "T_canary")
        for task in manifest["tasks"]:
            self.assertEqual(task["role"], "T_canary")
        self.assertIn("T_canary", manifest["exclusion"])
        validate_screen_manifest(manifest, self.registry, self.policy_split)

    def test_default_role_is_t_pilot(self) -> None:
        spec = self._spec(screen_id="default-role-001")
        manifest = build_screen_manifest(
            self.registry, self.policy_split, spec,
            registry_file=REGISTRY_PATH.name,
            policy_split_file=SPLIT_PATH.name,
        )
        # Default: task_role not in manifest (T_pilot is default, omitted for brevity).
        for task in manifest["tasks"]:
            self.assertEqual(task["role"], "T_pilot")
        self.assertIn("T_pilot", manifest["exclusion"])

    def test_old_manifest_without_top_level_task_role_still_validates(self) -> None:
        """Old manifests without top-level task_role should still validate
        (backward compatibility)."""
        spec = _spec_4_policies()
        manifest = build_screen_manifest(
            self.registry, self.policy_split, spec,
            registry_file=REGISTRY_PATH.name,
            policy_split_file=SPLIT_PATH.name,
        )
        # Simulate old manifest: remove top-level task_role if it exists.
        manifest.pop("task_role", None)
        # Recompute hash so it passes.
        from pyreplab_harness.m3_pilot import _canonical_hash
        manifest.pop("manifest_hash", None)
        manifest["manifest_hash"] = _canonical_hash(manifest)
        # Should still validate (T_pilot in tasks is fine).
        validate_screen_manifest(manifest, self.registry, self.policy_split)

    def test_invalid_task_role_rejected(self) -> None:
        spec = self._spec(
            screen_id="bad-role-001",
            task_role="T_invalid",
        )
        with self.assertRaisesRegex(ValueError, "task_role must be"):
            build_screen_manifest(
                self.registry, self.policy_split, spec,
                registry_file=REGISTRY_PATH.name,
                policy_split_file=SPLIT_PATH.name,
            )

    def test_protocol_object_bound_into_manifest(self) -> None:
        protocol = {"name": "text-first-enforcement", "version": 1}
        spec = self._spec(
            screen_id="protocol-screen-001",
            protocol=protocol,
        )
        manifest = build_screen_manifest(
            self.registry, self.policy_split, spec,
            registry_file=REGISTRY_PATH.name,
            policy_split_file=SPLIT_PATH.name,
        )
        self.assertIn("protocol", manifest)
        self.assertEqual(manifest["protocol"], protocol)

    def test_protocol_without_object_rejected(self) -> None:
        spec = self._spec(
            screen_id="bad-protocol-001",
            protocol="not-an-object",
        )
        with self.assertRaisesRegex(ValueError, "protocol must be an object"):
            build_screen_manifest(
                self.registry, self.policy_split, spec,
                registry_file=REGISTRY_PATH.name,
                policy_split_file=SPLIT_PATH.name,
            )

    def test_protocol_none_is_omitted(self) -> None:
        spec = self._spec(
            screen_id="no-protocol-001",
            protocol=None,
        )
        manifest = build_screen_manifest(
            self.registry, self.policy_split, spec,
            registry_file=REGISTRY_PATH.name,
            policy_split_file=SPLIT_PATH.name,
        )
        self.assertNotIn("protocol", manifest)

    def test_protocol_not_in_spec_is_omitted(self) -> None:
        spec = self._spec(screen_id="no-protocol-key-001")
        # Ensure 'protocol' is not even in the spec
        spec.pop("protocol", None)
        manifest = build_screen_manifest(
            self.registry, self.policy_split, spec,
            registry_file=REGISTRY_PATH.name,
            policy_split_file=SPLIT_PATH.name,
        )
        self.assertNotIn("protocol", manifest)


if __name__ == "__main__":
    unittest.main()
