from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pyreplab_harness.m3_pilot import (
    HELD_TEMPLATES,
    KNOWN_TEMPLATES,
    _RUNTIME_PINS,
    _canonical_hash,
    _write_immutable_json,
)
from pyreplab_harness.m3_prompt_only_pilot import (
    ADMITTED_TOOL_CALLS_PER_CELL,
    AGGREGATE_WALL_SECONDS,
    ARMS,
    ARM_PERMUTATIONS,
    ARM_PROMPTS,
    DUMMY_PROVIDER_API_KEY,
    EXECUTION_DISCIPLINE_PROMPT,
    EXPECTED_CELLS,
    EXPECTED_PANELS,
    EXPECTED_TASKS,
    FREEZE_FALSE_ADVANCE_MAX,
    FREEZE_FALSE_INTERACTION_MAX,
    MANIFEST_SCHEMA_VERSION,
    MAX_SCAN_FILE_BYTES,
    MIN_FREEZE_BANKS,
    NULL_SCENARIOS,
    PER_CELL_WALL_SECONDS,
    PI_CONFORMANCE_SCHEMA_VERSION,
    PREFLIGHT_SCHEMA_VERSION,
    PROMPT_TEMPLATES,
    PROVIDER_BACKED_TURNS_PER_CELL,
    RECOVERY_DISCIPLINE_PROMPT,
    REGISTERED_SCENARIOS,
    RUN_LOCAL_PROXY_PORT,
    RUN_LOCAL_TUNNEL_PORT,
    RUN_MODEL_ALIAS,
    RUN_PI_BASE_URL,
    RUN_PROVIDER,
    RUN_PROXY_UPSTREAM,
    RUN_TUNNEL_REMOTE_TARGET,
    SAMPLING_SEED_START,
    SCHEDULE_SEED,
    SIMULATOR_SEED,
    SCREEN_ID,
    SIMULATOR_REPORT_SCHEMA_VERSION,
    SOURCE_BUNDLE_SCHEMA_VERSION,
    SUBSTRATE_RECEIPT_SCHEMA_VERSION,
    TASK_SEED_START,
    TOOL_ATTEMPTS_PER_CELL,
    V8_FAILURE_HASH,
    V8_TRANSPORT_TOTAL_SECONDS,
    V8_TURN_LATENCIES_SECONDS,
    V10_COMPLETED_CELL_MODEL_WALL_SECONDS,
    V10_COMPLETED_CELL_RECORD_HASH,
    V10_FAILURE_HASH,
    WALL_BUDGET_AMENDMENT_SCHEMA_VERSION,
    _conformance_failures,
    _derive_bound_artifact_exclusions,
    _list_models_warnings,
    _parse_list_models_rows,
    _verify_conformance_stub_observations,
    analyze_ledger,
    analyze_ledger_test_only_valid_substrate,
    assert_no_collisions,
    build_cache_off_server_binding,
    build_command_arm_receipt,
    build_frozen_models_json,
    build_local_preflight,
    build_manifest,
    build_pi_conformance_receipt,
    build_prompt_only_registry,
    build_schedule,
    build_source_bundle_manifest,
    build_wall_budget_amendment,
    check_freeze_requirements,
    content_addressed_project_path,
    dummy_api_key_binding,
    evaluate_decision,
    evaluate_fixed_gate,
    evaluate_interaction_gate,
    freeze_prompt_only_artifacts,
    models_json_sha256,
    project_is_content_addressed,
    run_pi_conformance,
    run_simulator,
    scan_collisions,
    select_fixed_arm,
    source_bundle_manifest_hash,
    validate_frozen_models_json_config,
    validate_local_preflight,
    validate_manifest,
    validate_pi_conformance_receipt,
    validate_schedule,
    validate_simulator_report,
    validate_substrate_receipt,
    write_frozen_models_json,
)
from pyreplab_harness.orchestrator import (
    UNBROWSER_INTERACTIVE_TOOL_INTERFACE,
    run_registered_treatments,
)
from pyreplab_harness.treatments import TreatmentRegistry
from pyreplab_harness.unbrowser_fixture_gym import OUTCOME_ONLY_GENERATOR_VERSION


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _remote_identity() -> dict:
    """A content-addressed remote identity bound to the current source bundle."""
    bundle_hash = source_bundle_manifest_hash(PROJECT_ROOT)
    project = content_addressed_project_path("/remote/project", bundle_hash)
    return {
        "host": "ubuntu-local",
        "project": project,
        "run_root": f"{project}/.runs/prompt-only",
        "python": "python3",
    }


REMOTE_IDENTITY = _remote_identity()

_EXPECTED_EXECUTION = (
    "Execution discipline: Reserve one tool call for the required "
    "/workspace/result.json write. As soon as the requested verification key "
    "is visible, write the exact required JSON and stop. Do not make another "
    "browser call after the key is found."
)
_EXPECTED_RECOVERY = (
    "Recovery discipline: When a tool call fails or is rejected, inspect the "
    "returned error before acting. Never repeat an unchanged failed request. "
    "Make at most one corrected retry for that failure; if it fails again, "
    "stop retrying that operation and choose a different route or stop."
)


def _registry() -> TreatmentRegistry:
    return build_prompt_only_registry()


def _manifest(registry: TreatmentRegistry | None = None) -> dict:
    return build_manifest(
        registry or _registry(), REMOTE_IDENTITY, registry_file="registry.json"
    )


def _counts(form=(0, 0, 0), distractor=(0, 0, 0)) -> dict:
    return {
        "form": {"E": form[0], "C": form[1], "R": form[2]},
        "distractor": {"E": distractor[0], "C": distractor[1], "R": distractor[2]},
    }


def _substrate_receipt(manifest: dict) -> dict:
    authorization_hash = "9" * 64
    payload = {
        "schema_version": SUBSTRATE_RECEIPT_SCHEMA_VERSION,
        "authorization_hash": authorization_hash,
        "manifest_hash": manifest["manifest_hash"],
        "isolated_no_cache_server_identity": manifest[
            "isolated_no_cache_server_identity"
        ],
        "server_argv_hash_match": True,
        "substrate_valid": True,
        "live_model_execution_authorized": False,
        "evidence": {
            "authorization_hash": authorization_hash,
            "server_receipt_hash": "a" * 64,
            "tunnel_receipt_hash": "b" * 64,
            "active_service_receipt_hash": "c" * 64,
            "slot_clear_receipt_hashes": ["d" * 64] * EXPECTED_CELLS,
            "proxy_receipt_hashes": ["e" * 64] * EXPECTED_CELLS,
            "off_server_argv_hash": manifest[
                "isolated_no_cache_server_identity"
            ]["server_argv_hash"],
            "server_alias": RUN_MODEL_ALIAS,
            "server_readiness_verified": True,
            "tunnel_topology": {
                "local_port": RUN_LOCAL_TUNNEL_PORT,
                "remote_target": RUN_TUNNEL_REMOTE_TARGET,
            },
            "proxy_topology": {
                "local_port": RUN_LOCAL_PROXY_PORT,
                "upstream": RUN_PROXY_UPSTREAM,
            },
            "active_service_unchanged": True,
            "cache_invalidation_free": True,
            "teardown_verified": True,
            "slot_action_dir_preparation_receipt": {"path": "p", "mode": "555"},
            "slot_action_dir_removal_receipt": {"path": "p", "mode": "555"},
            "slot_action_dir_removed": True,
            "slot_action_dir_absence_verified": True,
            "generation_lease_acquire_receipt": {"path": "g", "mode": "555"},
            "generation_lease_release_receipt": {"path": "g", "released": True},
            "generation_lease_local_acquire_receipt": {
                "path": "local-g",
                "mode": "600",
            },
            "generation_lease_local_release_receipt": {
                "path": "local-g",
                "released": True,
            },
            "infrastructure_invalid_cells": 0,
            "source_commit": "abc123def456",
            "source_bundle_hash": source_bundle_manifest_hash(PROJECT_ROOT),
        },
    }
    return {**payload, "receipt_hash": _canonical_hash(payload)}


def _fake_simulator_report(
    manifest: dict, *, draws: int, overshoot: bool = False
) -> dict:
    """Build a self-consistent simulator report whose Wilson bounds/rates are
    recomputable from the raw decision counts (for freeze tests)."""
    from pyreplab_harness.m3_prompt_only_pilot import _wilson_upper_95

    scenarios = []
    for name in REGISTERED_SCENARIOS:
        kind = "null" if name in NULL_SCENARIOS else "alternative"
        if overshoot and kind == "null":
            # Push both false-advance and false-interaction bounds over the
            # freeze thresholds using a maximal interaction decision count.
            interaction = draws
            fixed = 0
            stop = 0
            invalid = 0
        else:
            fixed = 0
            interaction = 0
            stop = draws
            invalid = 0
        advance = fixed + interaction
        scenarios.append(
            {
                "scenario": name,
                "kind": kind,
                "draws": draws,
                "seed": 1,
                "decision_counts": {
                    "stop": stop,
                    "independent_fixed_policy_replication": fixed,
                    "independent_interaction_replication": interaction,
                    "invalid": invalid,
                },
                "advance_rate": advance / draws,
                "interaction_rate": interaction / draws,
                "advance_upper_95": _wilson_upper_95(advance, draws),
                "interaction_upper_95": _wilson_upper_95(interaction, draws),
            }
        )
    payload = {
        "schema_version": SIMULATOR_REPORT_SCHEMA_VERSION,
        "manifest_hash": manifest["manifest_hash"],
        "seed": 1,
        "draws_per_scenario": draws,
        "min_freeze_banks": MIN_FREEZE_BANKS,
        "freeze_requirements_met": check_freeze_requirements(scenarios),
        "freeze_thresholds": {
            "false_advance_max": FREEZE_FALSE_ADVANCE_MAX,
            "false_interaction_max": FREEZE_FALSE_INTERACTION_MAX,
            "screening_limits_are_choices_not_guarantees": True,
        },
        "scenarios": scenarios,
    }
    return {**payload, "report_hash": _canonical_hash(payload)}


class PromptOnlyRegistryTest(unittest.TestCase):
    def test_three_arms_exact_prompt_bytes(self) -> None:
        registry = _registry()
        self.assertEqual(len(registry), 3)
        self.assertEqual(ARM_PROMPTS["E"], "")
        self.assertEqual(EXECUTION_DISCIPLINE_PROMPT, _EXPECTED_EXECUTION)
        self.assertEqual(RECOVERY_DISCIPLINE_PROMPT, _EXPECTED_RECOVERY)
        self.assertEqual(
            ARM_PROMPTS["R"],
            _EXPECTED_EXECUTION + "\n\n" + _EXPECTED_RECOVERY,
        )
        for arm in ARMS:
            treatment = registry.by_id(arm)
            self.assertEqual(treatment.system_prompt, ARM_PROMPTS[arm])
            self.assertEqual(treatment.allowed_tools, ("bash", "unbrowser"))
            self.assertEqual(treatment.tool_interface, UNBROWSER_INTERACTIVE_TOOL_INTERFACE)
            self.assertEqual(treatment.max_output_tokens, 4096)
            self.assertEqual(treatment.tool_call_limit, 12)
            self.assertEqual(treatment.command_timeout_seconds, 60)
            self.assertEqual(treatment.wall_time_limit_seconds, 3300)
            self.assertIn("pilot-excluded", treatment.version)

    def test_arms_identical_except_system_prompt_bytes(self) -> None:
        registry = _registry()
        dicts = {arm: registry.by_id(arm).to_dict() for arm in ARMS}
        for arm in ARMS:
            self.assertEqual(dicts[arm]["version"], dicts["E"]["version"])
            self.assertEqual(dicts[arm]["allowed_tools"], dicts["E"]["allowed_tools"])
            self.assertEqual(dicts[arm]["max_output_tokens"], dicts["E"]["max_output_tokens"])
            self.assertEqual(dicts[arm]["tool_call_limit"], dicts["E"]["tool_call_limit"])
            self.assertEqual(dicts[arm]["command_timeout_seconds"], dicts["E"]["command_timeout_seconds"])
            self.assertEqual(dicts[arm]["wall_time_limit_seconds"], dicts["E"]["wall_time_limit_seconds"])
            self.assertEqual(dicts[arm]["tool_interface"], dicts["E"]["tool_interface"])
            self.assertEqual(dicts[arm]["generator_metadata"], dicts["E"]["generator_metadata"])
        self.assertEqual(
            len({dicts[arm]["system_prompt"] for arm in ARMS}), 3
        )
        self.assertEqual(len({dicts[arm]["bundle_hash"] for arm in ARMS}), 3)


class PromptOnlyManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = _registry()
        self.manifest = _manifest(self.registry)

    def test_manifest_deterministic_and_self_hashed(self) -> None:
        second = _manifest(self.registry)
        self.assertEqual(self.manifest, second)
        validate_manifest(self.manifest, self.registry)
        self.assertEqual(self.manifest["screen_id"], SCREEN_ID)
        self.assertEqual(SCREEN_ID, "m3-prompt-only-pilot-20260816-v11")
        self.assertEqual(len(self.manifest["tasks"]), EXPECTED_TASKS)
        self.assertEqual(len(self.manifest["panels"]), EXPECTED_PANELS)
        self.assertEqual(len(self.manifest["cells"]), EXPECTED_CELLS)

    def test_manifest_binds_v3_generator_and_cache_off_identity(self) -> None:
        self.assertEqual(self.manifest["task_generator_version"], "unbrowser-fixture-v3")
        self.assertEqual(self.manifest["task_generator_version"], OUTCOME_ONLY_GENERATOR_VERSION)
        self.assertEqual(
            self.manifest["runtime_pins"]["fixture_generator_version"],
            OUTCOME_ONLY_GENERATOR_VERSION,
        )
        cache = self.manifest["isolated_no_cache_server_identity"]
        expected = build_cache_off_server_binding(
            str(_RUNTIME_PINS["llama_server_path"]),
            str(_RUNTIME_PINS["model_artifact_path"]),
        )
        self.assertEqual(cache["server_argv_hash"], expected["server_argv_hash"])
        self.assertEqual(cache["mode"], "off")
        self.assertFalse(cache["cache_canary_implied_passed"])

    def test_manifest_tampering_rejected(self) -> None:
        tampered = dict(self.manifest)
        tampered["schedule_seed"] = 1
        with self.assertRaisesRegex(ValueError, "manifest_hash"):
            validate_manifest(tampered, self.registry)

    def test_template_identity_fail_closed_default(self) -> None:
        self.assertFalse(self.manifest["estimands"]["template_identity_available_pre_action"])
        self.assertTrue(self.manifest["estimands"]["lookup_value_is_diagnostic_only"])
        self.assertEqual(
            self.manifest["estimands"]["legal_lookup"],
            {"form_entry_validation": "C", "distractor_recovery": "R"},
        )

    def test_declared_template_identity_roundtrips(self) -> None:
        declared = build_manifest(
            self.registry,
            REMOTE_IDENTITY,
            registry_file="registry.json",
            declare_template_identity_available=True,
        )
        self.assertTrue(declared["estimands"]["template_identity_available_pre_action"])
        self.assertFalse(declared["estimands"]["lookup_value_is_diagnostic_only"])
        validate_manifest(declared, self.registry)

    def test_no_live_authorization_flags(self) -> None:
        self.assertFalse(
            self.manifest["authorization_boundary"]["live_model_execution_authorized"]
        )
        self.assertFalse(
            self.manifest["authorization_boundary"]["cache_canary_implied_passed"]
        )

    def test_behavior_diagnostic_is_diagnostic_only(self) -> None:
        diagnostic = self.manifest["diagnostics"]["behavior_classification"]
        self.assertEqual(diagnostic["role"], "diagnostic_only")
        self.assertIsNone(diagnostic["advancement_gate"])
        self.assertIsNone(diagnostic["scientific_threshold"])
        self.assertIn("diagnostic", diagnostic["note"])

    def test_untested_templates_distinct_from_held_templates(self) -> None:
        self.assertEqual(
            self.manifest["held_templates"], list(HELD_TEMPLATES)
        )
        untested = self.manifest["untested_templates"]
        self.assertEqual(
            set(untested),
            set(KNOWN_TEMPLATES) - set(PROMPT_TEMPLATES),
        )
        # Untested templates are not described as held-out, and vice versa.
        self.assertTrue(set(untested).isdisjoint(set(HELD_TEMPLATES)))
        self.assertNotIn("cross_page_comparison", untested)
        self.assertNotIn("stateful_workflow", untested)


class ScheduleBalanceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.schedule = build_schedule()
        self.tasks = self.schedule["tasks"]
        self.panels = self.schedule["panels"]
        self.cells = self.schedule["cells"]

    def test_schedule_counts_and_validate(self) -> None:
        validate_schedule(self.schedule)
        self.assertEqual(len(self.tasks), EXPECTED_TASKS)
        self.assertEqual(len(self.panels), EXPECTED_PANELS)
        self.assertEqual(len(self.cells), EXPECTED_CELLS)

    def test_task_seeds_consecutive_and_form_first(self) -> None:
        self.assertEqual(
            [task["seed"] for task in self.tasks],
            list(range(TASK_SEED_START, TASK_SEED_START + EXPECTED_TASKS)),
        )
        self.assertEqual(
            [task["template"] for task in self.tasks[:6]], ["form_entry_validation"] * 6
        )
        self.assertEqual(
            [task["template"] for task in self.tasks[6:]], ["distractor_recovery"] * 6
        )
        for template in ("form_entry_validation", "distractor_recovery"):
            difficulties = [
                task["difficulty"] for task in self.tasks if task["template"] == template
            ]
            self.assertEqual(difficulties.count("easy"), 2)
            self.assertEqual(difficulties.count("medium"), 2)
            self.assertEqual(difficulties.count("hard"), 2)

    def test_permutation_balance_within_template(self) -> None:
        for template in ("form_entry_validation", "distractor_recovery"):
            panels = [p for p in self.panels if p["template"] == template]
            orders = [tuple(p["execution_order"]) for p in panels]
            counts = {order: orders.count(order) for order in set(orders)}
            self.assertEqual(set(counts), set(ARM_PERMUTATIONS))
            self.assertEqual(set(counts.values()), {2})
            for position in range(3):
                arms = [order[position] for order in orders]
                self.assertEqual(
                    {arm: arms.count(arm) for arm in ARMS},
                    {"E": 4, "C": 4, "R": 4},
                )

    def test_sampling_seeds_consecutive_and_unique(self) -> None:
        self.assertEqual(
            [p["sampling_seed"] for p in self.panels],
            list(range(SAMPLING_SEED_START, SAMPLING_SEED_START + EXPECTED_PANELS)),
        )
        self.assertEqual(len({p["sampling_seed"] for p in self.panels}), EXPECTED_PANELS)

    def test_cell_ids_unique_and_expected(self) -> None:
        self.assertEqual(len({c["cell_id"] for c in self.cells}), EXPECTED_CELLS)
        for cell in self.cells:
            self.assertEqual(
                cell["cell_id"], f"{cell['panel_id']}/arm={cell['arm']}"
            )

    def test_schedule_seed_value(self) -> None:
        self.assertEqual(SCHEDULE_SEED, 1608262501)

    def test_schedule_seed_genuinely_determines_ordering(self) -> None:
        from pyreplab_harness.m3_prompt_only_pilot import (
            _build_panels,
            _build_tasks,
            _seeded_permutation_order,
        )

        order_a = _seeded_permutation_order(SCHEDULE_SEED)
        order_b = _seeded_permutation_order(SCHEDULE_SEED)
        order_c = _seeded_permutation_order(SCHEDULE_SEED + 999999)
        # Reproducible and still a permutation of the six arm permutations.
        self.assertEqual(order_a, order_b)
        self.assertEqual(set(order_a), set(ARM_PERMUTATIONS))
        # A different seed genuinely changes the derived ordering.
        self.assertNotEqual(order_a, order_c)

        tasks = _build_tasks()
        with mock.patch("pyreplab_harness.m3_prompt_only_pilot.SCHEDULE_SEED", 1):
            panels_a = _build_panels(tasks)
        with mock.patch("pyreplab_harness.m3_prompt_only_pilot.SCHEDULE_SEED", 2):
            panels_b = _build_panels(tasks)
        orders_a = [tuple(p["execution_order"]) for p in panels_a]
        orders_b = [tuple(p["execution_order"]) for p in panels_b]
        self.assertNotEqual(orders_a, orders_b)

    def test_replica_chronology_counterbalanced(self) -> None:
        first_replica_counts = {0: 0, 1: 0}
        for task in self.tasks:
            task_panels = [
                panel for panel in self.panels if panel["task_id"] == task["task_id"]
            ]
            self.assertEqual(
                {panel["rollout_replica"] for panel in task_panels}, {0, 1}
            )
            first = min(task_panels, key=self.panels.index)
            first_replica_counts[int(first["rollout_replica"])] += 1
        self.assertEqual(first_replica_counts, {0: 6, 1: 6})

    def test_fresh_seeds_disjoint_from_prior_modules(self) -> None:
        task_seeds = {task["seed"] for task in self.tasks}
        sampling_seeds = {panel["sampling_seed"] for panel in self.panels}
        for start, count in (
            (2026081001, 12),
            (2026088001, EXPECTED_TASKS),
            (2026089001, EXPECTED_TASKS),
            (2026091001, 24),
        ):
            self.assertTrue(task_seeds.isdisjoint(range(start, start + count)))
        for start, count in ((1900006001, 24), (1900007001, 24), (1900009001, 24)):
            self.assertTrue(sampling_seeds.isdisjoint(range(start, start + count)))

    def test_v11_seeds_fresh_and_no_v1_through_v10_reuse(self) -> None:
        # V11 uses fresh task/sampling/schedule/simulator seeds after v10's
        # infrastructure-invalid generation: v10 completed five cells (slowest
        # arm C at 2423.536s model wall < 3300s, record e93e2802a90a0c4d635d
        # 72cf3286c6ceafaeae3869a14e614e9398666fafc2d4) and was then consumed
        # by a single-shot slot-clear transport timeout while the OFF server
        # was busy prompt-evaluating the previous cell's final completion
        # request (12,213 tokens ≈ 305 s at ~40 tokens/s), root-caused and
        # fixed with bounded wait-for-idle slot-clear polling. v10's failure
        # hash is b4a318a72f12c5cbd9af921b9deac6ef24fdefcdcb910f92818b5e508d30969f.
        self.assertEqual(TASK_SEED_START, 2026093001)
        self.assertEqual(SAMPLING_SEED_START, 1900011001)
        self.assertEqual(SCHEDULE_SEED, 1608262501)
        self.assertEqual(SIMULATOR_SEED, 1608262502)
        self.assertEqual(SCREEN_ID, "m3-prompt-only-pilot-20260816-v11")
        task_seeds = {task["seed"] for task in self.tasks}
        sampling_seeds = {panel["sampling_seed"] for panel in self.panels}
        self.assertEqual(
            task_seeds, set(range(2026093001, 2026093001 + EXPECTED_TASKS))
        )
        self.assertEqual(
            sampling_seeds, set(range(1900011001, 1900011001 + EXPECTED_PANELS))
        )
        # Every aborted or infrastructure-invalid v1-v10 prompt-only seed bank.
        v1_task = set(range(2026092001, 2026092001 + EXPECTED_TASKS))
        v1_sampling = set(range(1900010001, 1900010001 + EXPECTED_PANELS))
        v2_task = set(range(2026092101, 2026092101 + EXPECTED_TASKS))
        v2_sampling = set(range(1900010101, 1900010101 + EXPECTED_PANELS))
        v3_task = set(range(2026092201, 2026092201 + EXPECTED_TASKS))
        v3_sampling = set(range(1900010201, 1900010201 + EXPECTED_PANELS))
        v4_task = set(range(2026092301, 2026092301 + EXPECTED_TASKS))
        v4_sampling = set(range(1900010301, 1900010301 + EXPECTED_PANELS))
        v5_task = set(range(2026092401, 2026092401 + EXPECTED_TASKS))
        v5_sampling = set(range(1900010401, 1900010401 + EXPECTED_PANELS))
        v6_task = set(range(2026092501, 2026092501 + EXPECTED_TASKS))
        v6_sampling = set(range(1900010501, 1900010501 + EXPECTED_PANELS))
        v7_task = set(range(2026092601, 2026092601 + EXPECTED_TASKS))
        v7_sampling = set(range(1900010601, 1900010601 + EXPECTED_PANELS))
        v8_task = set(range(2026092701, 2026092701 + EXPECTED_TASKS))
        v8_sampling = set(range(1900010701, 1900010701 + EXPECTED_PANELS))
        v9_task = set(range(2026092801, 2026092801 + EXPECTED_TASKS))
        v9_sampling = set(range(1900010801, 1900010801 + EXPECTED_PANELS))
        v10_task = set(range(2026092901, 2026092901 + EXPECTED_TASKS))
        v10_sampling = set(range(1900010901, 1900010901 + EXPECTED_PANELS))
        all_v11 = task_seeds | sampling_seeds | {SCHEDULE_SEED, SIMULATOR_SEED}
        for bank in (
            v1_task,
            v1_sampling,
            v2_task,
            v2_sampling,
            v3_task,
            v3_sampling,
            v4_task,
            v4_sampling,
            v5_task,
            v5_sampling,
            v6_task,
            v6_sampling,
            v7_task,
            v7_sampling,
            v8_task,
            v8_sampling,
            v9_task,
            v9_sampling,
            v10_task,
            v10_sampling,
        ):
            self.assertTrue(all_v11.isdisjoint(bank))
        for old in (
            2026081601,
            2026081602,
            1608261601,
            1608261602,
            1608261701,
            1608261702,
            1608261801,
            1608261802,
            1608261901,
            1608261902,
            1608262001,
            1608262002,
            1608262101,
            1608262102,
            1608262201,
            1608262202,
            1608262301,
            1608262302,
            1608262401,
            1608262402,
        ):
            self.assertNotIn(old, all_v11)

    def test_v11_seed_values_absent_from_runs(self) -> None:
        # Every v11 task/sampling/schedule/simulator seed must be absent from
        # all current JSON/JSONL run artifacts before freeze (v8/v9/v10 seeds
        # ARE present, in the consumed v8/v9/v10 artifacts).
        #
        # Excluded: the v11 generation's OWN closed-run artifacts. After the
        # interrupted-terminal closure (2026-08-23), the immutable 43-cell
        # ledger and its bound artifacts legitimately contain v11 identifiers
        # and seeds — they ARE the v11 audit record, not a leakage vector.
        # Only exact file paths are excluded (never directories/wildcards),
        # matching _normalized_exclude_paths policy. The closed generation's
        # satellite artifacts (proxy-N.jsonl, .active/.launch/.claim markers)
        # are enumerated as exact paths at setup time.
        runs_dir = PROJECT_ROOT / ".runs"
        exclude = sorted(
            str(p) for p in runs_dir.glob("m3-prompt-only-pilot-20260816-v11*")
            if p.is_file()
        )
        # Analysis-layer raw-event cache: verbatim copies of the closed
        # generation's pi-events.jsonl fetched for post-hoc label recovery.
        exclude += sorted(
            str(p) for p in (runs_dir / "raw_cache").glob("*.jsonl")
        )
        # Derived analysis artifacts of the closed generation (feature table
        # built from the 43-cell ledger).
        v11_features = runs_dir / "v11-cell-features.json"
        if v11_features.is_file():
            exclude.append(str(v11_features))
        collisions = scan_collisions(
            _manifest(), runs_dir, exclude_paths=exclude
        )
        self.assertEqual(collisions, [])
        # v10 seeds must not appear in the v11 schedule (tasks/panels/cells/
        # schedule/simulator seeds). The amendment's provenance fields
        # legitimately cite the v10 failure hash and the v10 completed cell id
        # (which embeds the v10 task seed 2026092907), so the amendment is
        # excluded from this string scan.
        manifest_without_amendment = dict(_manifest())
        manifest_without_amendment.pop("wall_budget_amendment")
        for value in (
            2026092901,
            2026092912,
            1900010901,
            1900010924,
            1608262401,
            1608262402,
        ):
            self.assertNotIn(
                str(value), json.dumps(manifest_without_amendment, sort_keys=True)
            )


class WallBudgetAmendmentTest(unittest.TestCase):
    """Wall-budget amendment: exact 3300 per cell / 237600 aggregate."""

    def setUp(self) -> None:
        self.registry = _registry()
        self.manifest = _manifest(self.registry)
        self.amendment = build_wall_budget_amendment()

    def test_amendment_exact_values_and_stable_schema(self) -> None:
        self.assertEqual(WALL_BUDGET_AMENDMENT_SCHEMA_VERSION, "m3-prompt-only-wall-budget-amendment-v1")
        self.assertEqual(self.amendment["schema_version"], WALL_BUDGET_AMENDMENT_SCHEMA_VERSION)
        self.assertEqual(PER_CELL_WALL_SECONDS, 3300)
        self.assertEqual(AGGREGATE_WALL_SECONDS, 237600)
        self.assertEqual(self.amendment["per_cell_wall_seconds"], 3300)
        self.assertEqual(self.amendment["aggregate_wall_seconds"], 237600)
        self.assertEqual(AGGREGATE_WALL_SECONDS, EXPECTED_CELLS * PER_CELL_WALL_SECONDS)

    def test_frozen_derivation_is_exact(self) -> None:
        # ceil_to_300(1.25 * 587.955304 * ((13*14)/(6*7))) = 3300
        raw = 1.25 * V8_TRANSPORT_TOTAL_SECONDS * ((13 * 14) / (6 * 7))
        derived = int(math.ceil(raw / 300.0)) * 300
        self.assertEqual(derived, 3300)
        self.assertEqual(self.amendment["derivation"], "ceil_to_300(1.25 * 587.955304 * ((13*14)/(6*7))) = 3300")
        self.assertEqual(self.amendment["derivation_exact"], 3300)
        self.assertEqual(self.amendment["headroom_factor"], 1.25)
        self.assertEqual(self.amendment["rounding_step_seconds"], 300)

    def test_amendment_binds_v8_failure_evidence(self) -> None:
        self.assertEqual(
            V8_FAILURE_HASH,
            "a87334c276bc910de651324e80bf3fe4458818395ee65f550861fcaf93283a7b",
        )
        self.assertEqual(self.amendment["source_generation"], "v8")
        self.assertEqual(self.amendment["source_failure_hash"], V8_FAILURE_HASH)
        self.assertEqual(self.amendment["observed_transport_total_seconds"], 587.955304)
        self.assertEqual(
            self.amendment["observed_turn_latencies_seconds"],
            list(V8_TURN_LATENCIES_SECONDS),
        )
        self.assertEqual(
            tuple(self.amendment["observed_turn_latencies_seconds"]),
            (52.694562, 89.258063, 85.543581, 91.294580, 129.586812, 139.577706),
        )
        self.assertEqual(self.amendment["observed_turn_count"], 6)
        self.assertEqual(self.amendment["observed_gate_check_count"], 7)

    def test_amendment_binds_v10_failure_and_validation_evidence(self) -> None:
        self.assertEqual(
            V10_FAILURE_HASH,
            "b4a318a72f12c5cbd9af921b9deac6ef24fdefcdcb910f92818b5e508d30969f",
        )
        self.assertEqual(self.amendment["generation_failure_hash"], V10_FAILURE_HASH)
        validation = self.amendment["generation_validated_by_completed_cell"]
        self.assertEqual(validation["generation"], "v10")
        self.assertEqual(validation["record_hash"], V10_COMPLETED_CELL_RECORD_HASH)
        self.assertEqual(
            validation["record_hash"],
            "e93e2802a90a0c4d635d72cf3286c6ceafaeae3869a14e614e9398666fafc2d4",
        )
        self.assertEqual(validation["model_wall_seconds"], 2423.536)
        self.assertEqual(validation["model_wall_seconds"], V10_COMPLETED_CELL_MODEL_WALL_SECONDS)
        self.assertEqual(
            validation["cell_id"],
            "unbrowser-fixture-v3-distractor_recovery-easy-2026092907"
            "/replica=0/arm=C",
        )
        self.assertEqual(validation["status"], "completed")
        self.assertLess(validation["model_wall_seconds"], PER_CELL_WALL_SECONDS)

    def test_turn_tool_limits_unchanged(self) -> None:
        self.assertEqual(PROVIDER_BACKED_TURNS_PER_CELL, 13)
        self.assertEqual(TOOL_ATTEMPTS_PER_CELL, 13)
        self.assertEqual(ADMITTED_TOOL_CALLS_PER_CELL, 12)
        self.assertEqual(
            self.amendment["turn_limits_unchanged"],
            {
                "provider_backed_turns_per_cell": 13,
                "tool_attempts_per_cell": 13,
                "admitted_tool_calls_per_cell": 12,
            },
        )
        for arm in ARMS:
            treatment = self.registry.by_id(arm)
            self.assertEqual(treatment.tool_call_limit, 12)
            self.assertEqual(treatment.command_timeout_seconds, 60)
            self.assertEqual(treatment.wall_time_limit_seconds, PER_CELL_WALL_SECONDS)

    def test_amendment_is_conservative_envelope_not_confidence_bound(self) -> None:
        self.assertEqual(self.amendment["kind"], "conservative_engineering_envelope")
        self.assertIs(self.amendment["not_a_statistical_confidence_bound"], True)
        self.assertIs(self.amendment["reduces_arm_informative_censoring"], True)
        self.assertIn("not a statistical", self.amendment["purpose"])
        self.assertIn("censoring", self.amendment["purpose"])

    def test_manifest_carries_exact_amendment_and_validate_passes(self) -> None:
        self.assertEqual(self.manifest["wall_budget_amendment"], self.amendment)
        validate_manifest(self.manifest, self.registry)

    def test_amendment_drift_rejected_even_when_hash_consistent(self) -> None:
        # The explicit amendment check fires before the full manifest
        # recompute: a hash-consistent manifest carrying a stale amendment is
        # refused as soon as the frozen amendment builder output drifts.
        drifted = dict(self.amendment)
        drifted["per_cell_wall_seconds"] = 3000
        with mock.patch(
            "pyreplab_harness.m3_prompt_only_pilot.build_wall_budget_amendment",
            return_value=drifted,
        ):
            with self.assertRaisesRegex(ValueError, "wall budget amendment drifted"):
                validate_manifest(self.manifest, self.registry)

    def test_amendment_deterministic(self) -> None:
        self.assertEqual(build_wall_budget_amendment(), build_wall_budget_amendment())

    def test_amendment_binds_into_preflight_via_manifest_hash(self) -> None:
        # The local preflight binds the manifest hash; a manifest whose
        # amendment drifted changes the hash and is rejected end to end.
        drifted = json.loads(json.dumps(self.manifest))
        amendment = dict(drifted["wall_budget_amendment"])
        amendment["per_cell_wall_seconds"] = 3000
        drifted["wall_budget_amendment"] = amendment
        with self.assertRaisesRegex(ValueError, "manifest_hash"):
            validate_manifest(drifted, self.registry)


class GenerationSchemaRejectionTest(unittest.TestCase):
    """V8 artifacts are rejected: schema/screen identity is generation-bound."""

    def setUp(self) -> None:
        self.registry = _registry()
        self.manifest = _manifest(self.registry)

    def _v8_manifest(self) -> dict:
        payload = {key: value for key, value in self.manifest.items() if key != "manifest_hash"}
        payload["schema_version"] = "m3-prompt-only-pilot-manifest-v8"
        payload["screen_id"] = "m3-prompt-only-pilot-20260816-v8"
        return {**payload, "manifest_hash": _canonical_hash(payload)}

    def test_v8_manifest_rejected(self) -> None:
        v8 = self._v8_manifest()
        with self.assertRaisesRegex(ValueError, "unsupported prompt-only manifest schema"):
            validate_manifest(v8, self.registry)
        # The v8 schema string is not any current schema constant.
        self.assertNotEqual(v8["schema_version"], MANIFEST_SCHEMA_VERSION)
        self.assertNotEqual(v8["screen_id"], SCREEN_ID)

    def test_v8_preflight_schema_rejected(self) -> None:
        # A self-hashed preflight carrying the v8 schema is refused before any
        # artifact comparison.
        payload = {
            "schema_version": "m3-prompt-only-pilot-local-preflight-v8",
            "manifest_hash": self.manifest["manifest_hash"],
        }
        preflight = {**payload, "preflight_hash": _canonical_hash(payload)}
        with self.assertRaisesRegex(ValueError, "unsupported prompt-only preflight schema"):
            validate_local_preflight(
                preflight,
                self.manifest,
                self.registry,
                PROJECT_ROOT,
                PROJECT_ROOT / ".runs",
            )
        self.assertNotEqual(payload["schema_version"], PREFLIGHT_SCHEMA_VERSION)

    def test_v8_substrate_receipt_schema_rejected(self) -> None:
        receipt = _substrate_receipt(self.manifest)
        payload = {key: value for key, value in receipt.items() if key != "receipt_hash"}
        payload["schema_version"] = "m3-prompt-only-pilot-substrate-receipt-v8"
        v8_receipt = {**payload, "receipt_hash": _canonical_hash(payload)}
        with self.assertRaisesRegex(ValueError, "unsupported substrate receipt schema"):
            validate_substrate_receipt(v8_receipt, self.manifest)
        self.assertNotEqual(payload["schema_version"], SUBSTRATE_RECEIPT_SCHEMA_VERSION)


class SourceBundleTest(unittest.TestCase):
    def test_bundle_manifest_deterministic_and_self_hashed(self) -> None:
        manifest = build_source_bundle_manifest(PROJECT_ROOT)
        self.assertEqual(manifest["schema_version"], SOURCE_BUNDLE_SCHEMA_VERSION)
        self.assertEqual(manifest["bundle_hash"], source_bundle_manifest_hash(PROJECT_ROOT))
        self.assertEqual(len(manifest["bundle_hash"]), 64)
        paths = [entry["path"] for entry in manifest["files"]]
        self.assertEqual(paths, sorted(paths))
        for entry in manifest["files"]:
            self.assertIsInstance(entry["size"], int)
            self.assertEqual(len(entry["sha256"]), 64)
        self.assertEqual(build_source_bundle_manifest(PROJECT_ROOT), manifest)

    def test_bundle_manifest_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "src" / "real.py").write_text("x = 1\n", encoding="utf-8")
            (root / "src" / "link.py").symlink_to(root / "src" / "real.py")
            with self.assertRaisesRegex(ValueError, "symlink"):
                build_source_bundle_manifest(root)

    def test_bundle_extra_missing_and_byte_drift_change_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
            h0 = source_bundle_manifest_hash(root)
            # Extra file changes identity.
            (root / "src" / "b.py").write_text("y = 2\n", encoding="utf-8")
            h1 = source_bundle_manifest_hash(root)
            self.assertNotEqual(h0, h1)
            # Byte drift changes identity.
            (root / "src" / "a.py").write_text("x = 2\n", encoding="utf-8")
            h2 = source_bundle_manifest_hash(root)
            self.assertNotEqual(h1, h2)
            # Missing file changes identity.
            (root / "src" / "b.py").unlink()
            h3 = source_bundle_manifest_hash(root)
            self.assertNotEqual(h2, h3)

    def test_content_addressed_project_path_helpers(self) -> None:
        bundle_hash = "a" * 64
        self.assertEqual(
            content_addressed_project_path("/remote/project", bundle_hash),
            f"/remote/project-{bundle_hash}",
        )
        self.assertTrue(project_is_content_addressed(f"/remote/project-{bundle_hash}", bundle_hash))
        self.assertFalse(project_is_content_addressed("/remote/project", bundle_hash))
        self.assertFalse(project_is_content_addressed(f"/remote/project-{'b' * 64}", bundle_hash))

    def test_local_preflight_rejects_non_content_addressed_project(self) -> None:
        registry = _registry()
        non_addressed = build_manifest(
            registry,
            {
                "host": "ubuntu-local",
                "project": "/remote/project",
                "run_root": "/remote/project/.runs/prompt-only",
                "python": "python3",
            },
            registry_file="registry.json",
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "not content-addressed"):
                build_local_preflight(
                    non_addressed, registry, PROJECT_ROOT, directory, simulator_draws=20
                )

    def test_ancestor_named_excluded_dir_does_not_drop_files(self) -> None:
        # A checkout whose ancestor is named like an excluded dir must still
        # produce the full manifest (exclusions are namespace-relative only).
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / ".git" / "venv" / "checkout"
            (parent / "src").mkdir(parents=True)
            (parent / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
            manifest = build_source_bundle_manifest(parent)
            self.assertEqual(
                [entry["path"] for entry in manifest["files"]], ["src/a.py"]
            )

    def test_namespace_root_symlink_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real-src"
            real.mkdir()
            (real / "a.py").write_text("x = 1\n", encoding="utf-8")
            (root / "src").symlink_to(real)
            with self.assertRaisesRegex(ValueError, "symlink namespace root"):
                build_source_bundle_manifest(root)

    def test_namespace_root_non_directory_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").write_text("not a directory\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-directory namespace root"):
                build_source_bundle_manifest(root)

    def test_egg_info_does_not_change_hash_but_extra_source_does(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
            h0 = source_bundle_manifest_hash(root)
            # Editable-install build junk is excluded deterministically.
            egg = root / "src" / "pyreplab.egg-info"
            egg.mkdir()
            (egg / "PKG-INFO").write_text("Metadata-Version: 2.1\n", encoding="utf-8")
            self.assertEqual(source_bundle_manifest_hash(root), h0)
            # A real extra source/config file still changes identity.
            (root / "src" / "b.py").write_text("y = 2\n", encoding="utf-8")
            self.assertNotEqual(source_bundle_manifest_hash(root), h0)


class CollisionScanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = _manifest()

    def test_empty_run_root_has_no_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = assert_no_collisions(self.manifest, directory)
            self.assertEqual(result["collisions"], 0)

    def test_collision_on_task_seed_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prior = Path(directory) / "prior.json"
            prior.write_text(
                json.dumps({"seed": self.manifest["tasks"][0]["seed"]}), encoding="utf-8"
            )
            collisions = scan_collisions(self.manifest, directory)
            self.assertTrue(any(item["kind"] == "seed" for item in collisions))
            with self.assertRaisesRegex(ValueError, "collides"):
                assert_no_collisions(self.manifest, directory)

    def test_collision_on_sampling_seed_and_task_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prior = Path(directory) / "prior.jsonl"
            lines = [
                json.dumps({"sampling_seed": self.manifest["panels"][0]["sampling_seed"]}),
                json.dumps({"task_id": self.manifest["tasks"][6]["task_id"]}),
            ]
            prior.write_text("\n".join(lines) + "\n", encoding="utf-8")
            collisions = scan_collisions(self.manifest, directory)
            kinds = {item["kind"] for item in collisions}
            self.assertIn("seed", kinds)
            self.assertIn("identifier", kinds)

    def test_collision_on_simulator_seed_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prior = Path(directory) / "prior.json"
            prior.write_text(
                json.dumps({"simulator_seed": self.manifest["simulator_seed"]}),
                encoding="utf-8",
            )
            collisions = scan_collisions(self.manifest, directory)
            self.assertTrue(any(item["kind"] == "seed" for item in collisions))
            with self.assertRaisesRegex(ValueError, "collides"):
                assert_no_collisions(self.manifest, directory)

    def test_nested_directories_are_scanned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            nested = Path(directory) / "deep" / "nested"
            nested.mkdir(parents=True)
            (nested / "prior.json").write_text(
                json.dumps({"seed": self.manifest["tasks"][0]["seed"]}), encoding="utf-8"
            )
            collisions = scan_collisions(self.manifest, directory)
            self.assertTrue(any(item["kind"] == "seed" for item in collisions))
            with self.assertRaisesRegex(ValueError, "collides"):
                assert_no_collisions(self.manifest, directory)

    def test_embedded_long_digit_run_is_ambiguous_and_fails(self) -> None:
        # Seed embedded inside a longer digit run must not be skipped.
        seed = self.manifest["tasks"][0]["seed"]
        embedded = f"9{seed}123"  # 9 + 10-digit seed + 123 -> 14 digits
        with tempfile.TemporaryDirectory() as directory:
            prior = Path(directory) / "prior.jsonl"
            prior.write_text(
                json.dumps({"value": embedded}), encoding="utf-8"
            )
            collisions = scan_collisions(self.manifest, directory)
            seed_collisions = [c for c in collisions if c["kind"] == "seed"]
            self.assertTrue(seed_collisions)
            self.assertTrue(
                any(c.get("ambiguous") for c in seed_collisions)
            )
            with self.assertRaisesRegex(ValueError, "collides"):
                assert_no_collisions(self.manifest, directory)

    def test_unreadable_file_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prior = Path(directory) / "binary.jsonl"
            prior.write_bytes(b"\xff\xfe\x00invalid-utf8-\x80")
            with self.assertRaisesRegex(ValueError, "collision scan cannot"):
                scan_collisions(self.manifest, directory)

    def test_oversized_file_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prior = Path(directory) / "huge.json"
            with mock.patch(
                "pyreplab_harness.m3_prompt_only_pilot.MAX_SCAN_FILE_BYTES", 10
            ):
                prior.write_text(json.dumps({"a": "b"}) * 100, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "oversized"):
                    scan_collisions(self.manifest, directory)


class CommandArmReceiptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = _registry()
        self.manifest = _manifest(self.registry)

    def test_argv_equal_after_removing_append_prompt(self) -> None:
        receipt = build_command_arm_receipt(self.manifest, self.registry, PROJECT_ROOT)
        self.assertEqual(
            receipt["checks"]["append_system_prompt_present"],
            {"E": False, "C": True, "R": True},
        )
        self.assertTrue(receipt["checks"]["arm_argv_equal_after_stripping_append_prompt"])
        self.assertTrue(receipt["checks"]["tools_equal"])
        self.assertTrue(receipt["checks"]["tool_limit_frozen"])
        self.assertTrue(receipt["checks"]["command_timeout_frozen"])
        self.assertTrue(receipt["checks"]["output_limit_frozen"])
        self.assertTrue(receipt["checks"]["provider_turn_limit_frozen"])
        self.assertTrue(receipt["checks"]["budget_v3_extension_loaded"])
        self.assertTrue(receipt["checks"]["runtime_identity_equal"])
        self.assertTrue(receipt["checks"]["dummy_keyless_api_key_present"])
        self.assertEqual(receipt["common_argv"][-1], "__PYREPLAB_TASK_PROMPT__")

    def test_dummy_key_binding_and_command_threading(self) -> None:
        # The keyless loopback provider needs the fixed non-secret dummy key on
        # the exact production Pi command, while artifacts bind only its
        # mode/hash (never the literal).
        binding = dummy_api_key_binding()
        self.assertEqual(binding["mode"], "fixed_dummy_non_secret")
        self.assertEqual(
            binding["key_sha256"],
            hashlib.sha256(DUMMY_PROVIDER_API_KEY.encode("utf-8")).hexdigest(),
        )
        receipt = build_command_arm_receipt(self.manifest, self.registry, PROJECT_ROOT)
        self.assertEqual(receipt["api_key_binding"], binding)
        self.assertIn("--api-key", receipt["common_argv"])
        self.assertEqual(
            receipt["common_argv"][receipt["common_argv"].index("--api-key") + 1],
            "<bound-dummy-api-key>",
        )
        self.assertNotIn(DUMMY_PROVIDER_API_KEY, json.dumps(receipt, sort_keys=True))
        # The manifest binds the same mode/hash identity.
        self.assertEqual(self.manifest["runtime_pins"]["pi_api_key"], binding)

    def test_receipt_is_hash_bound(self) -> None:
        receipt = build_command_arm_receipt(self.manifest, self.registry, PROJECT_ROOT)
        tampered = json.loads(json.dumps(receipt))
        tampered["common_argv"][0] = "evil"
        from pyreplab_harness.m3_prompt_only_pilot import _verify_embedded_hash

        with self.assertRaisesRegex(ValueError, "receipt_hash"):
            _verify_embedded_hash(tampered, "receipt_hash")


def _valid_conformance_receipt_fixture(
    streaming_stub: dict | None = None,
) -> dict:
    """A structurally valid conformance receipt bound to the pinned Pi identity.

    Built with the pure receipt builder (no Pi invocation) so model-free tests
    can exercise preflight/authorization flows hermetically.
    """
    if streaming_stub is None:
        streaming_stub = _valid_conformance_stub_fixture()
    return build_pi_conformance_receipt(
        pi_identity={
            "path": "/opt/homebrew/bin/pi",
            "sha256": _RUNTIME_PINS["pi_cli_sha256"],
            "version": _RUNTIME_PINS["pi_version"],
        },
        list_models_rc=0,
        list_models_stdout=(
            "provider            model                         context  max-out  thinking  images\n"
            f"{RUN_PROVIDER:<20} {RUN_MODEL_ALIAS:<30}  65.5K    8.2K     no        no\n"
        ),
        list_models_stderr="",
        streaming_stub=streaming_stub,
    )


def _valid_conformance_stub_fixture() -> dict:
    """Valid loopback stub observations for the pure receipt builder."""
    return {
        "requests": [
            {
                "path": "/v1/chat/completions",
                "auth": f"Bearer {DUMMY_PROVIDER_API_KEY}",
                "model": RUN_MODEL_ALIAS,
                "stream": True,
            }
        ],
        "rc": 0,
        "stdout": "PYREPLAB-PROMPT-ONLY-CONFORMANCE-SENTINEL",
        "stderr": "",
        "config_sha256": "a" * 64,
    }


class PiConformanceGateTest(unittest.TestCase):
    def test_frozen_models_json_omits_sampling_params(self) -> None:
        # Regression: Pi 0.84.1 rejects ``samplingParams: null`` ("must be
        # object") and then reports no models available. The frozen config must
        # omit the key entirely and stay credential-free.
        content = build_frozen_models_json()
        serialized = json.dumps(content, sort_keys=True)
        self.assertNotIn("samplingParams", serialized)
        for secret in ("apiKey", "api_key", "secret", "password", "token"):
            self.assertNotIn(secret, serialized)
        model = content["providers"][RUN_PROVIDER]["models"][0]
        self.assertEqual(model["id"], RUN_MODEL_ALIAS)
        self.assertEqual(model["api"], "openai-completions")
        self.assertEqual(content["providers"][RUN_PROVIDER]["baseUrl"], RUN_PI_BASE_URL)

    def test_validate_frozen_config_rejects_null_sampling_params(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_dir = Path(directory) / "config"
            receipt = write_frozen_models_json(config_dir)
            self.assertEqual(receipt["models_json_sha256"], models_json_sha256())
            self.assertEqual((config_dir / "models.json").stat().st_mode & 0o777, 0o444)
            validate_frozen_models_json_config(config_dir)
            # The exact v7 failure shape must now fail closed on disk: adding
            # ``samplingParams: null`` drifts the content hash, so the final
            # config validation rejects it before any authorization consume.
            drifted = json.loads(
                (config_dir / "models.json").read_text(encoding="utf-8")
            )
            drifted["providers"][RUN_PROVIDER]["models"][0]["samplingParams"] = None
            (config_dir / "models.json").chmod(0o644)
            (config_dir / "models.json").write_text(
                json.dumps(drifted), encoding="utf-8"
            )
            (config_dir / "models.json").chmod(0o444)
            with self.assertRaisesRegex(ValueError, "drifted"):
                validate_frozen_models_json_config(config_dir)

    def test_list_models_row_parsing(self) -> None:
        rows, unrecognized = _parse_list_models_rows(
            "provider            model                         context  max-out  thinking  images\n"
            f"{RUN_PROVIDER:<20} {RUN_MODEL_ALIAS:<30}  65.5K    8.2K     no        no\n"
        )
        self.assertEqual(rows, [{"provider": RUN_PROVIDER, "model": RUN_MODEL_ALIAS}])
        self.assertEqual(unrecognized, [])
        # "No models available." is a warning, never a parseable row.
        rows, _ = _parse_list_models_rows(
            "No models available. Use /login to log into a provider via OAuth "
            "or API key. See:\n  /opt/pi/docs/models.md\n"
        )
        self.assertEqual(rows, [])
        self.assertIn(
            "no models available",
            " | ".join(_list_models_warnings("", "No models available.")),
        )

    def test_conformance_failure_schema_warning(self) -> None:
        failures = _conformance_failures(
            pi_identity={
                "sha256": _RUNTIME_PINS["pi_cli_sha256"],
                "version": _RUNTIME_PINS["pi_version"],
            },
            list_models_rc=0,
            list_models_stdout="No models available.\n",
            list_models_stderr=(
                "Warning: errors loading models.json:\n"
                "Invalid models.json schema:\n"
                "  - providers.prompt-pilot-gemma.models.0.samplingParams: "
                "must be object\n"
            ),
            stub_observations=None,
        )
        self.assertTrue(any("config warnings" in item for item in failures))
        self.assertTrue(any("expected exactly one" in item for item in failures))

    def test_conformance_failure_no_model_and_wrong_model(self) -> None:
        no_model = _conformance_failures(
            pi_identity={
                "sha256": _RUNTIME_PINS["pi_cli_sha256"],
                "version": _RUNTIME_PINS["pi_version"],
            },
            list_models_rc=0,
            list_models_stdout="provider model\n",
            list_models_stderr="",
            stub_observations=None,
        )
        self.assertTrue(any("expected exactly one" in item for item in no_model))
        wrong_model = _conformance_failures(
            pi_identity={
                "sha256": _RUNTIME_PINS["pi_cli_sha256"],
                "version": _RUNTIME_PINS["pi_version"],
            },
            list_models_rc=0,
            list_models_stdout=(
                "provider model\n"
                "prompt-pilot-gemma other-model-alias\n"
            ),
            list_models_stderr="",
            stub_observations=None,
        )
        self.assertTrue(any("expected exactly one" in item for item in wrong_model))

    def test_conformance_failure_wrong_request_auth_and_multiple(self) -> None:
        base = _valid_conformance_stub_fixture()
        wrong_path = dict(base)
        wrong_path["requests"] = [dict(base["requests"][0], path="/v1/completions")]
        failures = _verify_conformance_stub_observations(wrong_path)
        self.assertTrue(any("request path" in item for item in failures))

        wrong_model = dict(base)
        wrong_model["requests"] = [dict(base["requests"][0], model="other-model")]
        failures = _verify_conformance_stub_observations(wrong_model)
        self.assertTrue(any("model is wrong" in item for item in failures))

        wrong_auth = dict(base)
        wrong_auth["requests"] = [dict(base["requests"][0], auth="Bearer wrong-key")]
        failures = _verify_conformance_stub_observations(wrong_auth)
        self.assertTrue(any("bearer header" in item for item in failures))

        multiple = dict(base)
        multiple["requests"] = [base["requests"][0], base["requests"][0]]
        failures = _verify_conformance_stub_observations(multiple)
        self.assertTrue(any("exactly one" in item for item in failures))

    def test_conformance_receipt_roundtrip_and_tamper(self) -> None:
        receipt = _valid_conformance_receipt_fixture(
            streaming_stub=_valid_conformance_stub_fixture()
        )
        self.assertEqual(receipt["schema_version"], PI_CONFORMANCE_SCHEMA_VERSION)
        validate_pi_conformance_receipt(receipt)
        # Tampering with the verdict breaks the embedded normalized hash.
        tampered = json.loads(json.dumps(receipt))
        tampered["list_models"]["rows"] = []
        with self.assertRaisesRegex(ValueError, "receipt_hash"):
            validate_pi_conformance_receipt(tampered)
        # A re-hashed tamper that drifts the row is also rejected structurally.
        tampered["receipt_hash"] = _canonical_hash(
            {k: v for k, v in tampered.items() if k != "receipt_hash"}
        )
        with self.assertRaisesRegex(ValueError, "exactly the expected"):
            validate_pi_conformance_receipt(tampered)

    def test_conformance_receipt_requires_streaming_stub(self) -> None:
        receipt = build_pi_conformance_receipt(
            pi_identity={
                "path": "/opt/homebrew/bin/pi",
                "sha256": _RUNTIME_PINS["pi_cli_sha256"],
                "version": _RUNTIME_PINS["pi_version"],
            },
            list_models_rc=0,
            list_models_stdout=(
                "provider model\n"
                f"{RUN_PROVIDER} {RUN_MODEL_ALIAS}\n"
            ),
            list_models_stderr="",
            streaming_stub=None,
        )
        with self.assertRaisesRegex(ValueError, "requires the streaming stub"):
            validate_pi_conformance_receipt(receipt)

    def test_conformance_receipt_rejects_unpinned_executable(self) -> None:
        receipt = _valid_conformance_receipt_fixture()
        tampered = json.loads(json.dumps(receipt))
        tampered["pi_sha256"] = "f" * 64
        tampered["receipt_hash"] = _canonical_hash(
            {k: v for k, v in tampered.items() if k != "receipt_hash"}
        )
        with self.assertRaisesRegex(ValueError, "pinned Pi executable"):
            validate_pi_conformance_receipt(tampered)

    def test_conformance_receipt_rejects_wrong_auth_hash(self) -> None:
        receipt = _valid_conformance_receipt_fixture(
            streaming_stub=_valid_conformance_stub_fixture()
        )
        tampered = json.loads(json.dumps(receipt))
        tampered["streaming_stub"]["request_auth_sha256"] = "0" * 64
        tampered["receipt_hash"] = _canonical_hash(
            {k: v for k, v in tampered.items() if k != "receipt_hash"}
        )
        with self.assertRaisesRegex(ValueError, "bearer header"):
            validate_pi_conformance_receipt(tampered)

    def test_conformance_receipt_rejects_multiple_requests(self) -> None:
        stub = _valid_conformance_stub_fixture()
        stub["requests"] = [stub["requests"][0], stub["requests"][0]]
        receipt = _valid_conformance_receipt_fixture(streaming_stub=stub)
        with self.assertRaisesRegex(ValueError, "exactly one stub request"):
            validate_pi_conformance_receipt(receipt)

    @unittest.skipUnless(shutil.which("pi"), "pinned Pi CLI is unavailable")
    def test_run_pi_conformance_end_to_end(self) -> None:
        # Real gate: pinned Pi binary, isolated PI_CODING_AGENT_DIR, sanitized
        # PI_OFFLINE=1 environment, list-models + loopback streaming stub.
        receipt = run_pi_conformance("pi", include_streaming_stub=True)
        validate_pi_conformance_receipt(receipt)
        self.assertEqual(receipt["list_models"]["rc"], 0)
        self.assertEqual(receipt["list_models"]["warnings_count"], 0)
        self.assertEqual(receipt["streaming_stub"]["request_count"], 1)
        self.assertIs(receipt["streaming_stub"]["sentinel_present"], True)

    def test_preflight_embeds_and_revalidates_conformance(self) -> None:
        registry = _registry()
        manifest = _manifest(registry)
        receipt = _valid_conformance_receipt_fixture()
        with tempfile.TemporaryDirectory() as directory:
            preflight = build_local_preflight(
                manifest,
                registry,
                PROJECT_ROOT,
                directory,
                simulator_draws=20,
                pi_conformance_receipt=receipt,
            )
            self.assertEqual(preflight["pi_conformance"], receipt)
            validate_local_preflight(
                preflight,
                manifest,
                registry,
                PROJECT_ROOT,
                directory,
                simulator_draws=20,
                require_pi_conformance=True,
            )
            # A preflight without the receipt fails closed when required.
            bare = build_local_preflight(
                manifest, registry, PROJECT_ROOT, directory, simulator_draws=20
            )
            self.assertIsNone(bare["pi_conformance"])
            with self.assertRaisesRegex(ValueError, "missing the pi conformance"):
                validate_local_preflight(
                    bare,
                    manifest,
                    registry,
                    PROJECT_ROOT,
                    directory,
                    simulator_draws=20,
                    require_pi_conformance=True,
                )
            # A tampered embedded receipt breaks the preflight hash.
            tampered = dict(preflight)
            tampered["pi_conformance"] = json.loads(json.dumps(receipt))
            tampered["pi_conformance"]["list_models"]["rc"] = 1
            with self.assertRaisesRegex(ValueError, "preflight_hash"):
                validate_local_preflight(
                    tampered,
                    manifest,
                    registry,
                    PROJECT_ROOT,
                    directory,
                    simulator_draws=20,
                )


class GateTest(unittest.TestCase):
    def test_select_fixed_arm_pooled_then_contrast_then_lexical(self) -> None:
        # C has larger pooled difference.
        self.assertEqual(select_fixed_arm(_counts(form=(0, 8, 0), distractor=(0, 8, 0))), "C")
        # R has larger pooled difference.
        self.assertEqual(select_fixed_arm(_counts(form=(0, 0, 8), distractor=(0, 0, 8))), "R")
        # Equal pooled, C has larger minimum contrast.
        self.assertEqual(select_fixed_arm(_counts(form=(0, 6, 8), distractor=(0, 6, 4))), "C")
        # Equal pooled and contrast -> C lexical.
        self.assertEqual(select_fixed_arm(_counts(form=(0, 6, 6), distractor=(0, 6, 6))), "C")

    def test_fixed_gate_boundary(self) -> None:
        # Exact boundary: pooled 6, each template 2.
        passed = evaluate_fixed_gate(_counts(form=(0, 2, 0), distractor=(0, 4, 0)))
        self.assertTrue(passed["passed"])
        # One template below boundary.
        failed = evaluate_fixed_gate(_counts(form=(0, 1, 0), distractor=(0, 5, 0)))
        self.assertFalse(failed["passed"])
        # Pooled below boundary.
        failed = evaluate_fixed_gate(_counts(form=(0, 2, 0), distractor=(0, 3, 0)))
        self.assertFalse(failed["passed"])

    def test_fixed_gate_severe_veto_blocks(self) -> None:
        passed = evaluate_fixed_gate(
            _counts(form=(0, 6, 0), distractor=(0, 6, 0)),
            severe_vetos=["infrastructure_error"],
        )
        self.assertFalse(passed["passed"])
        self.assertEqual(passed["severe_vetos"], ["infrastructure_error"])

    def test_interaction_gate_boundary(self) -> None:
        # Exact boundary: lookup 6, fc_fr 3, dr_dc 3, fc_fe 2, dr_de 2.
        passed = evaluate_interaction_gate(_counts(form=(0, 3, 0), distractor=(0, 0, 3)))
        self.assertTrue(passed["passed"])
        # Lookup below boundary.
        failed = evaluate_interaction_gate(_counts(form=(1, 3, 0), distractor=(0, 0, 3)))
        self.assertFalse(failed["passed"])

    def test_interaction_veto_when_lookup_not_above_competing_arm(self) -> None:
        result = evaluate_interaction_gate(_counts(form=(0, 2, 5), distractor=(0, 5, 2)))
        self.assertIn("form_lookup_below_recovery", result["vetos"])
        self.assertIn("distractor_lookup_below_execution", result["vetos"])
        self.assertFalse(result["passed"])

    def test_decision_precedence(self) -> None:
        # Interaction passing -> interaction decision.
        result = evaluate_decision(_counts(form=(0, 3, 0), distractor=(0, 0, 3)), substrate_valid=True)
        self.assertEqual(result["decision"], "independent_interaction_replication")
        # Fixed passing (interaction not met) -> fixed decision.
        result = evaluate_decision(_counts(form=(0, 6, 6), distractor=(0, 6, 6)), substrate_valid=True)
        self.assertEqual(result["decision"], "independent_fixed_policy_replication")
        # Neither -> stop.
        result = evaluate_decision(_counts(form=(0, 0, 0), distractor=(0, 0, 0)), substrate_valid=True)
        self.assertEqual(result["decision"], "stop")
        # Invalid substrate -> invalid.
        result = evaluate_decision(_counts(form=(0, 6, 0), distractor=(0, 6, 0)), substrate_valid=False)
        self.assertEqual(result["decision"], "invalid")

    def test_select_fixed_arm_disqualifies_vetoed_arm(self) -> None:
        counts = _counts(form=(0, 8, 0), distractor=(0, 8, 0))
        self.assertEqual(select_fixed_arm(counts), "C")
        self.assertEqual(select_fixed_arm(counts, disqualified_arms=("C",)), "R")
        self.assertIsNone(select_fixed_arm(counts, disqualified_arms=("C", "R")))

    def test_fixed_gate_disqualified_candidate_blocks(self) -> None:
        result = evaluate_fixed_gate(
            _counts(form=(0, 8, 0), distractor=(0, 8, 0)),
            disqualified_arms=("C",),
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["candidate_arm"], "R")
        self.assertEqual(result["disqualified_arms"], ["C"])

    def test_fixed_gate_both_candidates_disqualified_blocks(self) -> None:
        result = evaluate_fixed_gate(
            _counts(form=(0, 8, 0), distractor=(0, 8, 0)),
            disqualified_arms=("C", "R"),
        )
        self.assertFalse(result["passed"])
        self.assertIsNone(result["candidate_arm"])

    def test_interaction_gate_disqualified_lookup_vetoes(self) -> None:
        result = evaluate_interaction_gate(
            _counts(form=(0, 3, 0), distractor=(0, 0, 3)),
            disqualified_arms=("C",),
        )
        self.assertIn("form_lookup_disqualified", result["vetos"])
        self.assertFalse(result["passed"])


class AnalyzerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = _manifest()

    def _ledger(self, success_by_arm=None) -> list:
        cells = []
        for cell in self.manifest["cells"]:
            success = True
            if success_by_arm is not None:
                success = success_by_arm.get(cell["arm"], True)
            cells.append(
                {
                    "cell_id": cell["cell_id"],
                    "panel_id": cell["panel_id"],
                    "task_id": cell["task_id"],
                    "template": cell["template"],
                    "difficulty": cell["difficulty"],
                    "arm": cell["arm"],
                    "success": success,
                    "tool_calls": 3,
                    "wall_seconds": 10.0,
                }
            )
        return cells

    def test_complete_ledger_analyzes_without_router(self) -> None:
        ledger = self._ledger({"E": False, "C": True, "R": True})
        analysis = analyze_ledger_test_only_valid_substrate(self.manifest, ledger)
        self.assertFalse(analysis["fitted_router"])
        self.assertEqual(analysis["decision_rule_fitted_model"], "none")
        self.assertTrue(analysis["gate_is_screen_not_efficacy_claim"])
        self.assertEqual(analysis["decision"], "independent_fixed_policy_replication")
        self.assertEqual(analysis["counts"]["form"]["E"], 0)
        self.assertEqual(analysis["counts"]["form"]["C"], 12)
        self.assertEqual(analysis["finite_bank"]["form_entry_validation"]["C"], 1.0)
        self.assertEqual(analysis["finite_bank"]["distractor_recovery"]["E"], 0.0)
        self.assertEqual(analysis["cost_failures"]["tool_calls_total"], 72 * 3)

    def test_analyzer_no_task_winner_output(self) -> None:
        analysis = analyze_ledger_test_only_valid_substrate(self.manifest, self._ledger())
        serialized = json.dumps(analysis["raw_task_vectors"])
        self.assertNotIn("winner", serialized.lower())
        self.assertNotIn("argmax", serialized.lower())
        self.assertNotIn("best_arm", serialized.lower())
        for vector in analysis["raw_task_vectors"]:
            self.assertEqual(set(vector["arm_outcomes"]), set(ARMS))

    def test_incomplete_ledger_rejected(self) -> None:
        ledger = self._ledger()
        with self.assertRaisesRegex(ValueError, "not a complete"):
            analyze_ledger_test_only_valid_substrate(self.manifest, ledger[:-1])

    def test_duplicate_cell_rejected(self) -> None:
        ledger = self._ledger()
        ledger[5] = dict(ledger[0])
        with self.assertRaisesRegex(ValueError, "not a complete"):
            analyze_ledger_test_only_valid_substrate(self.manifest, ledger)

    def test_extra_cell_rejected(self) -> None:
        ledger = self._ledger()
        extra = dict(ledger[0])
        extra["cell_id"] = "unbrowser-fixture-v3-extra/arm=E"
        extra["panel_id"] = "unbrowser-fixture-v3-extra/replica=0"
        ledger.append(extra)
        with self.assertRaisesRegex(ValueError, "not a complete"):
            analyze_ledger_test_only_valid_substrate(self.manifest, ledger)

    def test_lookup_diagnostics_present(self) -> None:
        analysis = analyze_ledger_test_only_valid_substrate(
            self.manifest, self._ledger({"E": False, "C": True, "R": True})
        )
        diagnostics = analysis["lookup_diagnostics"]
        self.assertFalse(diagnostics["template_identity_available_pre_action"])
        self.assertTrue(diagnostics["lookup_value_is_diagnostic_only"])
        self.assertEqual(
            diagnostics["legal_lookup"],
            {"form_entry_validation": "C", "distractor_recovery": "R"},
        )
        self.assertEqual(diagnostics["lookup_arm_values"]["form_entry_validation"], 1.0)

    def test_default_substrate_invalid_but_counts_reported(self) -> None:
        analysis = analyze_ledger(
            self.manifest, self._ledger({"E": False, "C": True, "R": True})
        )
        self.assertEqual(analysis["decision"], "invalid")
        self.assertFalse(analysis["substrate"]["substrate_valid"])
        self.assertIsNone(analysis["substrate"]["substrate_receipt_hash"])
        # Scientific counts are still reported.
        self.assertEqual(analysis["counts"]["form"]["C"], 12)
        self.assertTrue(analysis["gates"]["fixed"]["passed"])

    def test_substrate_receipt_grants_valid_decision(self) -> None:
        receipt = _substrate_receipt(self.manifest)
        analysis = analyze_ledger(
            self.manifest,
            self._ledger({"E": False, "C": True, "R": True}),
            substrate_receipt=receipt,
        )
        self.assertTrue(analysis["substrate"]["substrate_valid"])
        self.assertEqual(analysis["substrate"]["substrate_receipt_hash"], receipt["receipt_hash"])
        self.assertEqual(analysis["decision"], "independent_fixed_policy_replication")

    def test_invalid_substrate_receipt_rejected(self) -> None:
        receipt = _substrate_receipt(self.manifest)
        tampered = dict(receipt)
        tampered["substrate_valid"] = False
        with self.assertRaisesRegex(ValueError, "receipt_hash"):
            analyze_ledger(
                self.manifest,
                self._ledger(),
                substrate_receipt=tampered,
            )

    def test_wrong_manifest_schema_rejected_before_analysis(self) -> None:
        bad = dict(self.manifest)
        bad["schema_version"] = "wrong-schema"
        bad["manifest_hash"] = _canonical_hash(
            {k: v for k, v in bad.items() if k != "manifest_hash"}
        )
        with self.assertRaisesRegex(ValueError, "schema"):
            analyze_ledger_test_only_valid_substrate(bad, self._ledger())

    def test_wrong_manifest_screen_rejected_before_analysis(self) -> None:
        bad = dict(self.manifest)
        bad["screen_id"] = "other-screen"
        bad["manifest_hash"] = _canonical_hash(
            {k: v for k, v in bad.items() if k != "manifest_hash"}
        )
        with self.assertRaisesRegex(ValueError, "screen"):
            analyze_ledger_test_only_valid_substrate(bad, self._ledger())

    def test_mismatched_registry_rejected_before_analysis(self) -> None:
        from pyreplab_harness.m3_empty_overlay_baseline import (
            build_empty_overlay_registry,
        )

        other_registry = build_empty_overlay_registry()
        with self.assertRaisesRegex(ValueError, "registry hash"):
            analyze_ledger_test_only_valid_substrate(
                self.manifest, self._ledger(), registry=other_registry
            )

    def test_malformed_row_missing_arm_raises_value_error(self) -> None:
        ledger = self._ledger()
        del ledger[0]["arm"]
        with self.assertRaisesRegex(ValueError, "not a complete"):
            analyze_ledger_test_only_valid_substrate(self.manifest, ledger)

    def test_malformed_row_bad_success_type_raises_value_error(self) -> None:
        ledger = self._ledger()
        ledger[0]["success"] = "yes"
        with self.assertRaisesRegex(ValueError, "not a complete"):
            analyze_ledger_test_only_valid_substrate(self.manifest, ledger)

    def test_non_mapping_row_raises_value_error(self) -> None:
        ledger = self._ledger()
        ledger[0] = "not-an-object"
        with self.assertRaisesRegex(ValueError, "JSON object"):
            analyze_ledger_test_only_valid_substrate(self.manifest, ledger)

    def _ledger_with_veto(self, arm_to_code=None, success_by_arm=None) -> list:
        ledger = self._ledger(success_by_arm=success_by_arm)
        if arm_to_code:
            for row in ledger:
                code = arm_to_code.get(row["arm"])
                if code is not None:
                    row["severe_veto"] = code
        return ledger

    def test_generation_invalid_veto_forces_invalid_decision(self) -> None:
        ledger = self._ledger_with_veto(
            arm_to_code={"C": "cross_arm_task_contamination"},
            success_by_arm={"E": False, "C": True, "R": True},
        )
        analysis = analyze_ledger_test_only_valid_substrate(self.manifest, ledger)
        self.assertEqual(analysis["decision"], "invalid")
        self.assertTrue(analysis["severe_vetos"]["generation_invalid"])
        self.assertTrue(analysis["severe_vetos"]["itt_rows_retained"])

    def test_e_arm_severe_veto_forces_invalid_decision(self) -> None:
        ledger = self._ledger_with_veto(
            arm_to_code={"E": "shell_network_attempt"},
            success_by_arm={"E": False, "C": True, "R": True},
        )
        analysis = analyze_ledger_test_only_valid_substrate(self.manifest, ledger)
        self.assertEqual(analysis["decision"], "invalid")
        self.assertTrue(analysis["severe_vetos"]["generation_invalid"])

    def test_c_arm_veto_disqualifies_fixed_candidate(self) -> None:
        # C dominates the pooled difference, but a C-arm severe veto forces the
        # fixed gate to select R instead.
        ledger = self._ledger_with_veto(
            arm_to_code={"C": "shell_network_attempt"},
            success_by_arm={"E": False, "C": True, "R": True},
        )
        analysis = analyze_ledger_test_only_valid_substrate(self.manifest, ledger)
        self.assertFalse(analysis["severe_vetos"]["generation_invalid"])
        self.assertEqual(analysis["severe_vetos"]["disqualified_arms"], ["C"])
        self.assertEqual(analysis["gates"]["fixed"]["candidate_arm"], "R")

    def test_unknown_veto_code_rejected(self) -> None:
        ledger = self._ledger_with_veto(arm_to_code={"C": "not_a_real_code"})
        with self.assertRaisesRegex(ValueError, "unknown severe veto code"):
            analyze_ledger_test_only_valid_substrate(self.manifest, ledger)


class SimulatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = _manifest()

    def test_simulator_deterministic(self) -> None:
        first = run_simulator(self.manifest, draws=100, seed=SIMULATOR_SEED)
        second = run_simulator(self.manifest, draws=100, seed=SIMULATOR_SEED)
        self.assertEqual(first, second)

    def test_registered_null_and_alternative_scenarios(self) -> None:
        from pyreplab_harness.m3_prompt_only_pilot import (
            NULL_SCENARIOS,
            REGISTERED_SCENARIOS,
        )

        self.assertIn("null_flat", NULL_SCENARIOS)
        self.assertIn("null_heterogeneous", NULL_SCENARIOS)
        self.assertIn("alt_fixed", REGISTERED_SCENARIOS)
        self.assertIn("alt_interaction", REGISTERED_SCENARIOS)

    def test_null_scenarios_pass_freeze_bounds(self) -> None:
        report = run_simulator(
            self.manifest,
            draws=5000,
            seed=SIMULATOR_SEED,
            scenarios=("null_flat", "null_heterogeneous"),
        )
        self.assertTrue(check_freeze_requirements(report["scenarios"]))
        for scenario in report["scenarios"]:
            self.assertLessEqual(scenario["advance_upper_95"], FREEZE_FALSE_ADVANCE_MAX)
            self.assertLessEqual(
                scenario["interaction_upper_95"], FREEZE_FALSE_INTERACTION_MAX
            )

    def test_freeze_check_rejects_overshooting_null(self) -> None:
        failing = [
            {
                "kind": "null",
                "advance_upper_95": FREEZE_FALSE_ADVANCE_MAX + 0.01,
                "interaction_upper_95": 0.0,
            }
        ]
        self.assertFalse(check_freeze_requirements(failing))
        passing = [
            {
                "kind": "null",
                "advance_upper_95": FREEZE_FALSE_ADVANCE_MAX - 0.01,
                "interaction_upper_95": FREEZE_FALSE_INTERACTION_MAX - 0.01,
            }
        ]
        self.assertTrue(check_freeze_requirements(passing))

    def test_alternative_scenarios_have_power(self) -> None:
        report = run_simulator(
            self.manifest,
            draws=2000,
            seed=SIMULATOR_SEED,
            scenarios=("alt_fixed", "alt_interaction"),
        )
        by_name = {scenario["scenario"]: scenario for scenario in report["scenarios"]}
        self.assertGreater(by_name["alt_fixed"]["advance_rate"], 0.0)
        self.assertGreater(by_name["alt_interaction"]["interaction_rate"], 0.0)

    def test_operating_characteristic_reports_screening_choices(self) -> None:
        report = run_simulator(self.manifest, draws=200, seed=SIMULATOR_SEED)
        thresholds = report["freeze_thresholds"]
        self.assertTrue(thresholds["screening_limits_are_choices_not_guarantees"])
        self.assertEqual(thresholds["false_advance_max"], FREEZE_FALSE_ADVANCE_MAX)
        self.assertEqual(thresholds["false_interaction_max"], FREEZE_FALSE_INTERACTION_MAX)
        oc = report["operating_characteristic"]
        self.assertTrue(oc["screening_limits_are_choices_not_guarantees"])
        self.assertEqual(
            {scenario["scenario"] for scenario in oc["null_scenarios"]},
            set(NULL_SCENARIOS),
        )
        for scenario in oc["null_scenarios"]:
            self.assertIn("false_advance_rate", scenario)
            self.assertIn("false_interaction_rate", scenario)

    def test_validate_simulator_report_structural(self) -> None:
        report = run_simulator(self.manifest, draws=200, seed=SIMULATOR_SEED)
        validate_simulator_report(report, self.manifest, expected_draws=200)
        # Tampered manifest hash is rejected.
        tampered = json.loads(json.dumps(report))
        tampered["manifest_hash"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "report_hash"):
            validate_simulator_report(tampered, self.manifest, expected_draws=200)
        # Missing a scenario is rejected.
        short = json.loads(json.dumps(report))
        short["scenarios"] = short["scenarios"][:-1]
        short["report_hash"] = _canonical_hash(
            {k: v for k, v in short.items() if k != "report_hash"}
        )
        with self.assertRaisesRegex(ValueError, "scenarios mismatch"):
            validate_simulator_report(short, self.manifest, expected_draws=200)

    def test_validate_simulator_report_requires_freeze_met(self) -> None:
        report = run_simulator(self.manifest, draws=200, seed=SIMULATOR_SEED)
        with self.assertRaisesRegex(ValueError, "at least"):
            validate_simulator_report(
                report, self.manifest, expected_draws=200, require_freeze_met=True
            )


class FreezeImmutableTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = _manifest()

    def test_freeze_requires_minimum_banks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "at least"):
                freeze_prompt_only_artifacts(
                    Path(directory) / "registry.json",
                    Path(directory) / "manifest.json",
                    REMOTE_IDENTITY,
                    project_root=PROJECT_ROOT,
                    run_root=directory,
                    simulator_draws=MIN_FREEZE_BANKS - 1,
                )

    def test_freeze_writes_immutable_and_non_authorizing(self) -> None:
        passing_report = _fake_simulator_report(self.manifest, draws=200_000)
        with tempfile.TemporaryDirectory() as directory:
            registry_path = Path(directory) / "registry.json"
            manifest_path = Path(directory) / "manifest.json"
            with mock.patch(
                "pyreplab_harness.m3_prompt_only_pilot.run_simulator",
                return_value=passing_report,
            ), mock.patch(
                "pyreplab_harness.m3_prompt_only_pilot.MIN_FREEZE_BANKS", 10
            ):
                first = freeze_prompt_only_artifacts(
                    registry_path,
                    manifest_path,
                    REMOTE_IDENTITY,
                    project_root=PROJECT_ROOT,
                    run_root=directory,
                    simulator_draws=200_000,
                )
                second = freeze_prompt_only_artifacts(
                    registry_path,
                    manifest_path,
                    REMOTE_IDENTITY,
                    project_root=PROJECT_ROOT,
                    run_root=directory,
                    simulator_draws=200_000,
                )
            self.assertEqual(first, second)
            self.assertFalse(first["live_model_execution_authorized"])
            self.assertFalse(first["cache_canary_implied_passed"])
            self.assertTrue(first["freeze_requirements_met"])
            restored = TreatmentRegistry.load(registry_path)
            self.assertEqual(restored.registry_hash, _registry().registry_hash)

    def test_freeze_rejects_failing_simulator_report(self) -> None:
        failing_report = _fake_simulator_report(
            self.manifest, draws=200_000, overshoot=True
        )
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch(
                "pyreplab_harness.m3_prompt_only_pilot.run_simulator",
                return_value=failing_report,
            ), mock.patch(
                "pyreplab_harness.m3_prompt_only_pilot.MIN_FREEZE_BANKS", 10
            ):
                with self.assertRaisesRegex(ValueError, "fails freeze"):
                    freeze_prompt_only_artifacts(
                        Path(directory) / "registry.json",
                        Path(directory) / "manifest.json",
                        REMOTE_IDENTITY,
                        project_root=PROJECT_ROOT,
                        run_root=directory,
                        simulator_draws=200_000,
                    )

    def test_immutable_write_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.json"
            _write_immutable_json(path, {"a": 1})
            with self.assertRaises(FileExistsError):
                _write_immutable_json(path, {"a": 2})


class PreflightTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = _registry()
        self.manifest = _manifest(self.registry)

    def test_local_preflight_roundtrips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            preflight = build_local_preflight(
                self.manifest, self.registry, PROJECT_ROOT, directory, simulator_draws=200
            )
            self.assertTrue(preflight["no_model_invoked"])
            self.assertFalse(preflight["live_model_execution_authorized"])
            validate_local_preflight(
                preflight,
                self.manifest,
                self.registry,
                PROJECT_ROOT,
                directory,
                simulator_draws=200,
            )

    def test_preflight_tampering_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            preflight = build_local_preflight(
                self.manifest, self.registry, PROJECT_ROOT, directory, simulator_draws=200
            )
            tampered = dict(preflight)
            tampered["schedule"]["tasks"] = 0
            with self.assertRaisesRegex(ValueError, "preflight_hash"):
                validate_local_preflight(
                    tampered,
                    self.manifest,
                    self.registry,
                    PROJECT_ROOT,
                    directory,
                    simulator_draws=200,
                )

    def test_exclude_paths_flow_through_local_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            # The manifest itself contains every task/sampling seed; placing it
            # under the run root would self-collide unless excluded.
            manifest_path = run_root / "manifest.json"
            manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
            preflight = build_local_preflight(
                self.manifest,
                self.registry,
                PROJECT_ROOT,
                run_root,
                simulator_draws=200,
                exclude_paths=[manifest_path],
            )
            self.assertEqual(preflight["collision_scan"]["collisions"], 0)
            validate_local_preflight(
                preflight,
                self.manifest,
                self.registry,
                PROJECT_ROOT,
                run_root,
                simulator_draws=200,
                exclude_paths=[manifest_path],
            )
            # Without the exclusion the same run root collides on the manifest.
            with self.assertRaises(ValueError):
                assert_no_collisions(self.manifest, run_root)

    def test_own_bound_artifacts_under_run_root_are_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "runs"
            run_root.mkdir()
            manifest_path = run_root / "manifest.json"
            registry_path = run_root / "registry.json"
            preflight_path = run_root / "preflight.json"
            manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
            self.registry.save(registry_path)
            # The freeze-time exclusion is derived from the exact artifact paths.
            exclude = _derive_bound_artifact_exclusions(
                run_root, registry_path, manifest_path, preflight_path
            )
            preflight = build_local_preflight(
                self.manifest,
                self.registry,
                PROJECT_ROOT,
                run_root,
                simulator_draws=200,
                exclude_paths=exclude,
            )
            # The exact exclusion contract is persisted, sorted + normalized.
            self.assertEqual(
                preflight["collision_scan"]["excluded_paths"],
                sorted(
                    str(p.resolve())
                    for p in (manifest_path, registry_path, preflight_path)
                ),
            )
            # Freeze and execution scans agree on the same contract.
            validate_local_preflight(
                preflight,
                self.manifest,
                self.registry,
                PROJECT_ROOT,
                run_root,
                simulator_draws=200,
                exclude_paths=exclude,
            )

    def test_unrelated_colliding_artifact_still_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            manifest_path = run_root / "manifest.json"
            manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
            # A prior (historical) artifact carrying a task seed must still fail.
            prior = run_root / "prior.json"
            prior.write_text(
                json.dumps({"seed": self.manifest["tasks"][0]["seed"]}),
                encoding="utf-8",
            )
            exclude = _derive_bound_artifact_exclusions(run_root, manifest_path)
            with self.assertRaisesRegex(ValueError, "collides"):
                assert_no_collisions(self.manifest, run_root, exclude_paths=exclude)

    def test_exclusion_contract_tampering_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            manifest_path = run_root / "manifest.json"
            manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
            exclude = _derive_bound_artifact_exclusions(run_root, manifest_path)
            preflight = build_local_preflight(
                self.manifest,
                self.registry,
                PROJECT_ROOT,
                run_root,
                simulator_draws=200,
                exclude_paths=exclude,
            )
            # Tamper the persisted contract to gain an unrelated exclusion.
            tampered = dict(preflight)
            tampered_collision = dict(tampered["collision_scan"])
            tampered_collision["excluded_paths"] = sorted(
                tampered_collision["excluded_paths"] + [str(run_root / "prior.json")]
            )
            tampered["collision_scan"] = tampered_collision
            tampered["preflight_hash"] = _canonical_hash(
                {k: v for k, v in tampered.items() if k != "preflight_hash"}
            )
            with self.assertRaisesRegex(ValueError, "collision receipt drifted"):
                validate_local_preflight(
                    tampered,
                    self.manifest,
                    self.registry,
                    PROJECT_ROOT,
                    run_root,
                    simulator_draws=200,
                    exclude_paths=exclude,
                )

    def test_path_substitution_cannot_gain_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            manifest_path = run_root / "manifest.json"
            manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
            exclude = _derive_bound_artifact_exclusions(run_root, manifest_path)
            preflight = build_local_preflight(
                self.manifest,
                self.registry,
                PROJECT_ROOT,
                run_root,
                simulator_draws=200,
                exclude_paths=exclude,
            )
            # Substituting a different artifact path at validation time cannot
            # gain exclusion: the re-derived exclusion set no longer matches the
            # persisted contract, so the re-scan re-exposes the real manifest
            # (which collides) rather than silently excluding it.
            substituted = run_root / "other-manifest.json"
            with self.assertRaises(ValueError):
                validate_local_preflight(
                    preflight,
                    self.manifest,
                    self.registry,
                    PROJECT_ROOT,
                    run_root,
                    simulator_draws=200,
                    exclude_paths=_derive_bound_artifact_exclusions(run_root, substituted),
                )

    def test_generic_runner_cannot_bypass_authorization_boundary(self) -> None:
        import argparse

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
                    __import__("pyreplab_harness.orchestrator", fromlist=["RemoteConfig"]).RemoteConfig(
                        "host", "/project", "/runs"
                    ),
                    args,
                )


class CLISmokeTest(unittest.TestCase):
    def test_simulate_validate_analyze_commands(self) -> None:
        from pyreplab_harness.m3_prompt_only_pilot import main

        manifest = _manifest()
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            registry_path = Path(directory) / "registry.json"
            _registry().save(registry_path)
            self.assertEqual(
                main(["simulate", "--manifest", str(manifest_path), "--draws", "20"]), 0
            )
            self.assertEqual(
                main(
                    [
                        "validate",
                        "--manifest",
                        str(manifest_path),
                        "--registry",
                        str(registry_path),
                    ]
                ),
                0,
            )
            cells = []
            for cell in manifest["cells"]:
                cells.append(
                    {
                        "cell_id": cell["cell_id"],
                        "panel_id": cell["panel_id"],
                        "task_id": cell["task_id"],
                        "template": cell["template"],
                        "difficulty": cell["difficulty"],
                        "arm": cell["arm"],
                        "success": True,
                    }
                )
            ledger_path = Path(directory) / "ledger.json"
            ledger_path.write_text(
                json.dumps(
                    {
                        "schema_version": "m3-prompt-only-pilot-ledger-v1",
                        "cells": cells,
                    }
                ),
                encoding="utf-8",
            )
            receipt_path = Path(directory) / "substrate-receipt.json"
            receipt_path.write_text(
                json.dumps(_substrate_receipt(manifest)), encoding="utf-8"
            )
            self.assertEqual(
                main(
                    [
                        "analyze",
                        "--manifest",
                        str(manifest_path),
                        "--ledger",
                        str(ledger_path),
                        "--registry",
                        str(registry_path),
                        "--substrate-receipt",
                        str(receipt_path),
                    ]
                ),
                0,
            )


if __name__ == "__main__":
    unittest.main()
