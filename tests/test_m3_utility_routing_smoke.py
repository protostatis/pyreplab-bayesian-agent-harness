from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from pyreplab_harness import m3_routing_probe_gate as gate
from pyreplab_harness import m3_utility_routing_smoke as smoke
from pyreplab_harness.m3_utility_routing_smoke import (
    ATTEMPTS_PER_SCHEDULE,
    ATTEMPT_IDENTITY_FIELDS,
    AUTHORIZATION_SCHEMA,
    BLOCKS,
    CANARY_EXCLUSION,
    CANARY_ROW_STATUS,
    DECISION_INVALID,
    DECISION_NO_GO,
    DECISION_PASS,
    EXECUTION_RECEIPT_SCHEMA,
    GATE_REPORT_SCHEMA,
    MANIFEST_SCHEMA,
    MAX_SAMPLING_SEED,
    PANELS_PER_SCHEDULE,
    REPLICAS,
    SAFE_ATTEMPT_SCHEMA,
    SAFE_EXPORT_SCHEMA,
    SPECIALIST_CAPABILITIES,
    TASKS,
    TASKS_PER_BLOCK,
    TASKS_PER_STRATUM_PER_BLOCK,
    THRESHOLDS,
    analyze_stage_b,
    build_authorization,
    build_execution_receipt,
    build_manifest,
    canonical_hash,
    canonical_json,
    craft_attempt_row,
    craft_runtime_rows,
    default_spec,
    immutable_write,
    immutable_write_jsonl,
    load_safe_attempts_jsonl,
    main,
    export_safe_attempts,
    run_stage_b,
    validate_authorization,
    validate_execution_receipt,
    validate_gate_report,
    validate_manifest,
    validate_runtime_preflight,
)
from pyreplab_harness.orchestrator import (
    AttemptExecutionError,
    policy_spec_from_treatment,
)
from pyreplab_harness.treatments import TreatmentRegistry

REGISTRY_PATH = (
    Path(__file__).resolve().parents[1] / "policies" / "m3-unbrowser-semantic-canary-v1.json"
)

# Deterministic, cacheable production Stage-A evidence (probe_pass).
_STAGE_A_MANIFEST = gate.build_manifest(gate.default_spec(seed=20260813))
_STAGE_A_REPORT = gate.run_stage_a(_STAGE_A_MANIFEST)
assert _STAGE_A_REPORT["decision"] == "probe_pass", _STAGE_A_REPORT["reasons"]


# ---------------------------------------------------------------------------
# production-compatible mock design that flips ambiguous declared operations
# ---------------------------------------------------------------------------


def _prompt_from_operation_flags(operation_flags: dict) -> str:
    table_operation = operation_flags.get("table_operation") is True
    form_operation = operation_flags.get("form_operation") is True
    first_operation = operation_flags.get("first_operation")
    if table_operation and not form_operation:
        return "Locate the access code for this task in the directory table."
    if form_operation and not table_operation:
        return "Complete the request form and submit it for processing."
    if table_operation and form_operation:
        if first_operation == "form":
            return (
                "Complete and submit the verification form first, then use the "
                "returned code to locate the confirmation key in the directory table."
            )
        return (
            "Locate the access code in the directory table, then complete and "
            "submit the verification form."
        )
    return "Review the task and carry out the required step."


class FlippedRoutingDesign:
    """32-coordinate Stage-A design whose ambiguous fixtures may be flipped so
    the frozen router disagrees with the sealed label (a probe_no_go gate)."""

    GENERATOR_VERSION = "mock-routing-design-v1"

    def __init__(self, flip_ambiguous: set | None = None):
        self._flip = flip_ambiguous or set()
        self._coords = self._build()

    def _build(self) -> list[dict]:
        coords: list[dict] = []
        for stratum in gate.STRATA:
            for i in range(8):
                index = len(coords)
                coord: dict = {
                    "fixture_id": f"sf-{index:08d}",
                    "stratum": stratum,
                    "difficulty": ("easy", "medium", "hard")[i % 3],
                    "seed": 1000 + index,
                }
                if stratum == "pure_table":
                    coord["operation_flags"] = {
                        "table_operation": True, "form_operation": False,
                        "first_operation": None,
                    }
                    coord["first_bottleneck"] = "table_specialist"
                elif stratum == "pure_form":
                    coord["operation_flags"] = {
                        "table_operation": False, "form_operation": True,
                        "first_operation": None,
                    }
                    coord["first_bottleneck"] = "form_specialist"
                elif stratum == "mixed":
                    first = "table" if i % 2 == 0 else "form"
                    coord["operation_flags"] = {
                        "table_operation": True, "form_operation": True,
                        "first_operation": first,
                    }
                    coord["first_bottleneck"] = (
                        "table_specialist" if first == "table" else "form_specialist"
                    )
                else:  # ambiguous
                    bottleneck = "table" if i < 4 else "form"
                    declared = bottleneck
                    if index in self._flip:
                        declared = "form" if bottleneck == "table" else "table"
                    coord["operation_flags"] = {
                        "table_operation": declared == "table",
                        "form_operation": declared == "form",
                        "first_operation": None,
                    }
                    coord["first_bottleneck"] = (
                        "table_specialist" if bottleneck == "table" else "form_specialist"
                    )
                coords.append(coord)
        return coords

    def build_stage_a_design(self, seed: int | None = None) -> list[dict]:
        return [dict(coord) for coord in self._coords]

    def generate_routing_fixture(self, coord: dict) -> dict:
        html = "<html><body>fixture</body></html>"
        return {
            "fixture_id": coord["fixture_id"],
            "title": f"Fixture {coord['fixture_id']}",
            "prompt": _prompt_from_operation_flags(coord["operation_flags"]),
            "html": html,
            "source_sha256": hashlib.sha256(html.encode("utf-8")).hexdigest(),
        }


# ---------------------------------------------------------------------------
# test harness
# ---------------------------------------------------------------------------


class StageBSmokeTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.sa_manifest_path = self.root / "stage-a-manifest.json"
        self.sa_report_path = self.root / "stage-a-report.json"
        immutable_write(self.sa_manifest_path, _STAGE_A_MANIFEST)
        immutable_write(self.sa_report_path, _STAGE_A_REPORT)
        self.registry = TreatmentRegistry.load(REGISTRY_PATH)
        self.spec = default_spec(REGISTRY_PATH, self.sa_manifest_path, self.sa_report_path)
        self.manifest = build_manifest(self.spec, self.registry, _STAGE_A_MANIFEST)

    def tearDown(self) -> None:
        self._tmp.cleanup()


# ---------------------------------------------------------------------------
# manifest build / determinism
# ---------------------------------------------------------------------------


class ManifestBuildTest(StageBSmokeTestBase):
    def test_deterministic_build(self) -> None:
        first = canonical_json(self.manifest)
        second = canonical_json(
            build_manifest(self.spec, self.registry, _STAGE_A_MANIFEST)
        )
        self.assertEqual(first, second)
        other_spec = default_spec(
            REGISTRY_PATH,
            self.sa_manifest_path,
            self.sa_report_path,
            seed=20260901,
        )
        other = build_manifest(other_spec, self.registry, _STAGE_A_MANIFEST)
        self.assertNotEqual(first, canonical_json(other))

    def test_manifest_self_hash(self) -> None:
        self.assertEqual(self.manifest["schema_version"], MANIFEST_SCHEMA)
        self.assertEqual(len(self.manifest["manifest_hash"]), 64)
        recomputed = canonical_hash(
            {k: v for k, v in self.manifest.items() if k != "manifest_hash"}
        )
        self.assertEqual(self.manifest["manifest_hash"], recomputed)

    def test_manifest_validates(self) -> None:
        self.assertEqual(validate_manifest(self.manifest), [])

    def test_counts_and_balance(self) -> None:
        tasks = self.manifest["tasks"]
        self.assertEqual(len(tasks), TASKS)
        per_block = {0: 0, 1: 0}
        per_block_stratum_difficulty: dict[tuple[int, str], set[str]] = {}
        per_block_capability: dict[tuple[int, str], int] = {}
        for task in tasks:
            per_block[task["block"]] += 1
            key = (task["block"], task["stratum"])
            per_block_stratum_difficulty.setdefault(key, set()).add(task["difficulty"])
            cap_key = (task["block"], task["preferred_capability"])
            per_block_capability[cap_key] = per_block_capability.get(cap_key, 0) + 1
        self.assertEqual(per_block, {0: TASKS_PER_BLOCK, 1: TASKS_PER_BLOCK})
        for block in (0, 1):
            for stratum in gate.STRATA:
                self.assertEqual(
                    per_block_stratum_difficulty[(block, stratum)], {"easy", "medium", "hard"}
                )
            for capability in ("table", "form"):
                self.assertEqual(per_block_capability[(block, capability)], 6)

        for schedule_name in ("primary", "contingency"):
            schedule = self.manifest["schedule"][schedule_name]
            self.assertEqual(len(schedule["panels"]), PANELS_PER_SCHEDULE)
            self.assertEqual(len(schedule["attempts"]), ATTEMPTS_PER_SCHEDULE)
        self.assertEqual(
            len(self.manifest["schedule"]["primary"]["attempts"])
            + len(self.manifest["schedule"]["contingency"]["attempts"]),
            192,
        )

    def test_route_receipts_bound(self) -> None:
        receipts = self.manifest["route_receipts"]
        self.assertEqual(len(receipts), TASKS)
        for task in self.manifest["tasks"]:
            receipt = receipts[task["task_id"]]
            self.assertEqual(receipt["route_receipt_sha256"], task["route_receipt_sha256"])
            self.assertIn(receipt["combined_route"], SPECIALIST_CAPABILITIES)
            self.assertIn(receipt["prompt_only_route"], SPECIALIST_CAPABILITIES)
            recomputed = canonical_hash(
                {k: v for k, v in receipt.items() if k != "route_receipt_sha256"}
            )
            self.assertEqual(receipt["route_receipt_sha256"], recomputed)
            self.assertEqual(task["probe_features_sha256"], canonical_hash(receipt["probe_features"]))
            self.assertEqual(task["request_features_sha256"], canonical_hash(receipt["request_features"]))

    def test_attempts_bind_task_hashes(self) -> None:
        task_by_id = {t["task_id"]: t for t in self.manifest["tasks"]}
        for schedule_name in ("primary", "contingency"):
            for attempt in self.manifest["schedule"][schedule_name]["attempts"]:
                task = task_by_id[attempt["task_id"]]
                self.assertEqual(attempt["source_sha256"], task["source_sha256"])
                self.assertEqual(attempt["probe_receipt_sha256"], task["probe_receipt_sha256"])
                self.assertEqual(attempt["route_receipt_sha256"], task["route_receipt_sha256"])
                self.assertEqual(attempt["schema_version"], SAFE_ATTEMPT_SCHEMA)
                self.assertEqual(attempt["canary_row_status"], CANARY_ROW_STATUS)
                self.assertEqual(attempt["canary_exclusion"], CANARY_EXCLUSION)

    def test_policy_binding(self) -> None:
        registry = self.manifest["registry"]
        self.assertEqual(registry["registry_hash"], self.registry.registry_hash)
        policies = registry["policies"]
        self.assertEqual(set(policies), set(SPECIALIST_CAPABILITIES))
        table = policies["table_specialist"]
        form = policies["form_specialist"]
        self.assertEqual(table["capability"], "table_specialist")
        self.assertEqual(form["capability"], "form_specialist")
        self.assertEqual(len(table["bundle_hash"]), 64)
        self.assertEqual(len(form["bundle_hash"]), 64)
        self.assertIsInstance(table["tool_interface"], str)
        self.assertIsInstance(form["tool_interface"], str)
        self.assertEqual(
            registry["policy_order"],
            [table["bundle_id"], form["bundle_id"]],
        )


# ---------------------------------------------------------------------------
# schedule identity
# ---------------------------------------------------------------------------


class ScheduleTest(StageBSmokeTestBase):
    def test_arm_crossover_within_task(self) -> None:
        panels = self.manifest["schedule"]["primary"]["panels"]
        by_task: dict[str, dict[int, dict]] = {}
        for panel in panels:
            by_task.setdefault(panel["task_id"], {})[panel["replica"]] = panel
        for task_id, replicas in by_task.items():
            self.assertEqual(
                list(reversed(replicas[0]["execution_order"])),
                replicas[1]["execution_order"],
                f"task {task_id}: replica 1 must reverse replica 0 arm positions",
            )

    def test_arm_position_block_balance(self) -> None:
        for schedule_name in ("primary", "contingency"):
            for block in (0, 1):
                for replica in (0, 1):
                    counts: dict[str, int] = {}
                    for attempt in self.manifest["schedule"][schedule_name]["attempts"]:
                        if (
                            attempt["block"] == block
                            and attempt["replica"] == replica
                            and attempt["arm_position"] == 0
                        ):
                            counts[attempt["policy_capability"]] = (
                                counts.get(attempt["policy_capability"], 0) + 1
                            )
                    self.assertEqual(
                        counts,
                        {"table_specialist": 6, "form_specialist": 6},
                        f"{schedule_name} block {block} replica {replica}",
                    )

    def test_schedule_uniqueness(self) -> None:
        all_attempt_ids: list[str] = []
        for schedule_name in ("primary", "contingency"):
            schedule = self.manifest["schedule"][schedule_name]
            attempt_ids = [a["attempt_id"] for a in schedule["attempts"]]
            all_attempt_ids.extend(attempt_ids)
            self.assertEqual(len(set(attempt_ids)), ATTEMPTS_PER_SCHEDULE)
            panel_ids = [p["panel_id"] for p in schedule["panels"]]
            self.assertEqual(len(set(panel_ids)), PANELS_PER_SCHEDULE)
            panel_seeds = [p["sampling_seed"] for p in schedule["panels"]]
            self.assertEqual(len(set(panel_seeds)), PANELS_PER_SCHEDULE)
            self.assertTrue(
                all(0 <= seed <= MAX_SAMPLING_SEED for seed in panel_seeds)
            )
            attempt_seeds = [a["sampling_seed"] for a in schedule["attempts"]]
            self.assertEqual(len(set(attempt_seeds)), PANELS_PER_SCHEDULE)
            self.assertTrue(
                all(attempt_seeds.count(seed) == 2 for seed in set(attempt_seeds))
            )
            positions = sorted(a["execution_order"] for a in schedule["attempts"])
            self.assertEqual(positions, list(range(ATTEMPTS_PER_SCHEDULE)))
        self.assertEqual(len(set(all_attempt_ids)), 192)

    def test_contingency_freshness(self) -> None:
        primary = self.manifest["schedule"]["primary"]
        contingency = self.manifest["schedule"]["contingency"]
        self.assertNotEqual(primary["chronology"], contingency["chronology"])
        primary_ids = {a["attempt_id"] for a in primary["attempts"]}
        contingency_ids = {a["attempt_id"] for a in contingency["attempts"]}
        self.assertTrue(primary_ids.isdisjoint(contingency_ids))
        primary_seeds = {a["sampling_seed"] for a in primary["attempts"]}
        contingency_seeds = {a["sampling_seed"] for a in contingency["attempts"]}
        self.assertTrue(primary_seeds.isdisjoint(contingency_seeds))
        self.assertTrue(all(seed <= MAX_SAMPLING_SEED for seed in primary_seeds))
        self.assertTrue(all(seed <= MAX_SAMPLING_SEED for seed in contingency_seeds))
        primary_tasks = {a["task_id"] for a in primary["attempts"]}
        contingency_tasks = {a["task_id"] for a in contingency["attempts"]}
        self.assertEqual(primary_tasks, contingency_tasks)
        primary_routes = {
            a["task_id"]: a["route_receipt_sha256"] for a in primary["attempts"]
        }
        contingency_routes = {
            a["task_id"]: a["route_receipt_sha256"] for a in contingency["attempts"]
        }
        self.assertEqual(primary_routes, contingency_routes)
        primary_orders = {
            (panel["task_id"], panel["replica"]): panel["execution_order"]
            for panel in primary["panels"]
        }
        contingency_orders = {
            (panel["task_id"], panel["replica"]): panel["execution_order"]
            for panel in contingency["panels"]
        }
        for key, order in primary_orders.items():
            self.assertEqual(contingency_orders[key], list(reversed(order)))

    def test_frozen_analysis_constants(self) -> None:
        analysis = self.manifest["analysis"]
        self.assertEqual(analysis["lambda_grid"], [0.0, 0.25, 0.5, 1.0, 2.0])
        self.assertEqual(analysis["primary_lambda"], 1.0)
        self.assertEqual(analysis["bootstrap"]["draws"], 100000)
        self.assertEqual(analysis["bootstrap"]["seed"], 2026081302)
        self.assertEqual(analysis["bootstrap"]["quantile"], "nearest-rank empirical 10th percentile")
        self.assertEqual(analysis["thresholds"], THRESHOLDS)
        self.assertEqual(
            analysis["run_policy"]["canary"],
            {"row_status": "T_canary", "exclusion": "canary_excluded", "training_impact": analysis["run_policy"]["canary"]["training_impact"]},
        )
        self.assertIn("intention_to_treat_failure", analysis["error_taxonomy"])


# ---------------------------------------------------------------------------
# tamper rejection and leakage
# ---------------------------------------------------------------------------


class IntegrityTest(StageBSmokeTestBase):
    def test_tampered_field_detected(self) -> None:
        tampered = json.loads(json.dumps(self.manifest))
        tampered["gates"]["tasks"] = 99
        errors = validate_manifest(tampered)
        self.assertTrue(any("counts.tasks" in e or "gates" in e for e in errors))

    def test_tampered_hash_detected(self) -> None:
        tampered = json.loads(json.dumps(self.manifest))
        tampered["manifest_hash"] = "0" * 64
        errors = validate_manifest(tampered)
        self.assertTrue(any("manifest_hash mismatch" in e for e in errors))

    def test_forged_manifest_detected_by_rebuild(self) -> None:
        """Even a well-rehashed forgery is caught by the deterministic rebuild."""
        forged = json.loads(json.dumps(self.manifest))
        forged["tasks"][0]["source_sha256"] = "0" * 64
        forged["manifest_hash"] = canonical_hash(
            {k: v for k, v in forged.items() if k != "manifest_hash"}
        )
        errors = validate_manifest(forged)
        self.assertTrue(any("rebuild" in e or "tasks do not match" in e for e in errors))

    def test_forged_schedule_detected_by_rebuild(self) -> None:
        forged = json.loads(json.dumps(self.manifest))
        schedule = forged["schedule"]["primary"]
        schedule["attempts"][0]["sampling_seed"] = schedule["attempts"][0]["sampling_seed"] + 1
        forged["manifest_hash"] = canonical_hash(
            {k: v for k, v in forged.items() if k != "manifest_hash"}
        )
        errors = validate_manifest(forged)
        self.assertTrue(any("schedule" in e for e in errors))

    def test_forged_boolean_chronology_replica_detected(self) -> None:
        forged = json.loads(json.dumps(self.manifest))
        coordinate = next(
            item
            for item in forged["schedule"]["primary"]["chronology"]
            if item["replica"] == 1
        )
        coordinate["replica"] = True
        forged["manifest_hash"] = canonical_hash(
            {k: v for k, v in forged.items() if k != "manifest_hash"}
        )
        errors = validate_manifest(forged)
        self.assertTrue(any("schedule.primary" in error for error in errors), errors)

    def test_forged_block_seeds_detected(self) -> None:
        forged = json.loads(json.dumps(self.manifest))
        forged["block_seeds"][0] += 1
        forged["manifest_hash"] = canonical_hash(
            {k: v for k, v in forged.items() if k != "manifest_hash"}
        )
        errors = validate_manifest(forged)
        self.assertTrue(any("block_seeds" in error for error in errors), errors)

    def test_malformed_task_entries_fail_closed(self) -> None:
        for replacement in ([], {"ordinal": 0}):
            malformed = json.loads(json.dumps(self.manifest))
            malformed["tasks"][0] = replacement
            malformed["manifest_hash"] = canonical_hash(
                {k: v for k, v in malformed.items() if k != "manifest_hash"}
            )
            errors = validate_manifest(malformed)
            self.assertTrue(errors)

    def test_no_secret_leakage(self) -> None:
        serialized = canonical_json(self.manifest)
        self.assertNotIn("<html", serialized)
        self.assertNotIn("RF-", serialized)
        self.assertNotIn("REF-", serialized)
        self.assertNotIn("PENDING", serialized)
        self.assertNotIn("LOCKED", serialized)
        self.assertNotIn("unlock_query_param", serialized)
        self.assertNotIn("expected_answer", serialized)

    def test_stage_b_rejects_nonzero_text_variant(self) -> None:
        spec = json.loads(json.dumps(self.spec))
        spec["text_variant"] = 1
        with self.assertRaisesRegex(ValueError, "text_variant"):
            build_manifest(spec, self.registry, _STAGE_A_MANIFEST)


# ---------------------------------------------------------------------------
# authorization
# ---------------------------------------------------------------------------


class AuthorizationTest(StageBSmokeTestBase):
    def test_validate_with_external_bindings(self) -> None:
        errors = validate_manifest(
            self.manifest,
            registry=self.registry,
            stage_a_manifest=_STAGE_A_MANIFEST,
            stage_a_report=_STAGE_A_REPORT,
        )
        self.assertEqual(errors, [])

    def test_authorization_pass(self) -> None:
        authorization = build_authorization(
            self.sa_manifest_path, self.sa_report_path, self.manifest
        )
        self.assertEqual(authorization["schema_version"], AUTHORIZATION_SCHEMA)
        self.assertEqual(authorization["manifest_hash"], self.manifest["manifest_hash"])
        self.assertEqual(authorization["decision"], "probe_pass")
        self.assertEqual(
            validate_authorization(
                authorization, self.manifest, self.sa_manifest_path, self.sa_report_path
            ),
            [],
        )
        recomputed = canonical_hash(
            {k: v for k, v in authorization.items() if k != "authorization_hash"}
        )
        self.assertEqual(authorization["authorization_hash"], recomputed)

    def test_authorization_rejects_no_go(self) -> None:
        design = FlippedRoutingDesign(flip_ambiguous={24, 25, 26})
        no_go_manifest = gate.build_manifest(
            gate.default_spec(seed=20260813), design_adapter=design
        )
        no_go_report = gate.run_stage_a(no_go_manifest, design_adapter=design)
        self.assertEqual(no_go_report["decision"], "probe_no_go")
        no_go_manifest_path = self.root / "no-go-manifest.json"
        no_go_report_path = self.root / "no-go-report.json"
        immutable_write(no_go_manifest_path, no_go_manifest)
        immutable_write(no_go_report_path, no_go_report)
        spec = default_spec(REGISTRY_PATH, no_go_manifest_path, no_go_report_path)
        stage_b = build_manifest(spec, self.registry, no_go_manifest)
        self.assertEqual(validate_manifest(stage_b), [])
        with self.assertRaises(ValueError):
            build_authorization(no_go_manifest_path, no_go_report_path, stage_b)

    def test_authorization_rejects_tampered_report_bytes(self) -> None:
        authorization = build_authorization(
            self.sa_manifest_path, self.sa_report_path, self.manifest
        )
        with open(self.sa_report_path, "a", encoding="utf-8") as handle:
            handle.write(" ")
        errors = validate_authorization(
            authorization, self.manifest, self.sa_manifest_path, self.sa_report_path
        )
        self.assertTrue(any("byte SHA-256 mismatch" in e for e in errors))

    def test_authorization_rejects_tampered_self_hash(self) -> None:
        authorization = build_authorization(
            self.sa_manifest_path, self.sa_report_path, self.manifest
        )
        tampered = json.loads(json.dumps(authorization))
        tampered["authorization_hash"] = "0" * 64
        errors = validate_authorization(
            tampered, self.manifest, self.sa_manifest_path, self.sa_report_path
        )
        self.assertTrue(any("authorization_hash mismatch" in e for e in errors))

    def test_authorization_rejects_wrong_manifest(self) -> None:
        authorization = build_authorization(
            self.sa_manifest_path, self.sa_report_path, self.manifest
        )
        other_spec = default_spec(
            REGISTRY_PATH,
            self.sa_manifest_path,
            self.sa_report_path,
            seed=20260901,
        )
        other_manifest = build_manifest(other_spec, self.registry, _STAGE_A_MANIFEST)
        errors = validate_authorization(
            authorization, other_manifest, self.sa_manifest_path, self.sa_report_path
        )
        self.assertTrue(any("manifest_hash" in e for e in errors))


# ---------------------------------------------------------------------------
# immutable writes and CLI
# ---------------------------------------------------------------------------


class WriteAndCliTest(StageBSmokeTestBase):
    def test_immutable_write_idempotent_and_refuses_different(self) -> None:
        path = self.root / "artifact.json"
        immutable_write(path, self.manifest)
        immutable_write(path, self.manifest)  # byte-identical no-op
        with self.assertRaises(FileExistsError):
            immutable_write(path, {"different": True})

    def test_cli_freeze_validate_authorize(self) -> None:
        registry = REGISTRY_PATH
        spec_path = self.root / "spec.json"
        manifest_path = self.root / "manifest.json"
        auth_path = self.root / "authorization.json"
        immutable_write(spec_path, self.spec)
        with redirect_stdout(StringIO()):
            rc = main(["freeze", str(spec_path), str(registry),
                       str(self.sa_manifest_path), str(manifest_path)])
        self.assertEqual(rc, 0)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(validate_manifest(manifest), [])
        with redirect_stdout(StringIO()):
            self.assertEqual(main(["validate", str(manifest_path)]), 0)
        with redirect_stdout(StringIO()):
            rc = main(["authorize", str(self.sa_manifest_path), str(self.sa_report_path),
                       str(manifest_path), str(auth_path)])
        self.assertEqual(rc, 0)
        with redirect_stdout(StringIO()):
            rc = main(["validate", str(manifest_path), "--authorization", str(auth_path),
                       "--stage-a-manifest", str(self.sa_manifest_path),
                       "--stage-a-report", str(self.sa_report_path)])
        self.assertEqual(rc, 0)
        # tampered manifest must fail validation
        tampered = json.loads(json.dumps(manifest))
        tampered["manifest_hash"] = "0" * 64
        tampered_path = self.root / "tampered.json"
        immutable_write(tampered_path, tampered)
        with redirect_stdout(StringIO()):
            self.assertEqual(main(["validate", str(tampered_path)]), 1)

    def test_cli_freeze_refuses_overwrite(self) -> None:
        spec_path = self.root / "spec.json"
        manifest_path = self.root / "manifest.json"
        immutable_write(spec_path, self.spec)
        with redirect_stdout(StringIO()):
            self.assertEqual(
                main(["freeze", str(spec_path), str(REGISTRY_PATH),
                      str(self.sa_manifest_path), str(manifest_path)]),
                0,
            )
        with redirect_stdout(StringIO()):
            self.assertEqual(
                main(["freeze", str(spec_path), str(REGISTRY_PATH),
                      str(self.sa_manifest_path), str(manifest_path)]),
                0,  # byte-identical freeze is idempotent
            )
        other_spec = default_spec(
            REGISTRY_PATH,
            self.sa_manifest_path,
            self.sa_report_path,
            seed=20260901,
        )
        other_spec_path = self.root / "other-spec.json"
        immutable_write(other_spec_path, other_spec)
        with redirect_stdout(StringIO()):
            rc = main(["freeze", str(other_spec_path), str(REGISTRY_PATH),
                       str(self.sa_manifest_path), str(manifest_path)])
        self.assertEqual(rc, 1)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["seed"], 20260814)

    def test_default_spec_requires_files(self) -> None:
        with self.assertRaises(ValueError):
            default_spec(self.root / "missing.json", self.sa_manifest_path, self.sa_report_path)
        with self.assertRaises(ValueError):
            default_spec(REGISTRY_PATH, self.root / "missing.json", self.sa_report_path)
        with self.assertRaises(ValueError):
            default_spec(REGISTRY_PATH, self.sa_manifest_path, self.root / "missing.json")


# ---------------------------------------------------------------------------
# Stage-B analyzer, execution receipt, and gate report
# ---------------------------------------------------------------------------


class StageBAnalyzerTestBase(StageBSmokeTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.authorization = build_authorization(
            self.sa_manifest_path, self.sa_report_path, self.manifest
        )

    def route_of(self, attempt: dict) -> str:
        return self.manifest["route_receipts"][attempt["task_id"]]["combined_route"]

    def routed_rows(
        self, *, block_schedules: dict | None = None, output_tokens=0
    ) -> list[dict]:
        """Rows where the frozen routed arm always succeeds and the other arm fails."""
        return craft_runtime_rows(
            self.manifest,
            self.authorization,
            block_schedules=block_schedules,
            success=lambda a: a["policy_capability"] == self.route_of(a),
            output_tokens=output_tokens,
        )

    def write_attempts_jsonl(self, rows: list[dict]) -> Path:
        path = self.root / "attempts.jsonl"
        path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )
        return path


def _runtime_preflight(manifest: dict, authorization: dict) -> dict:
    payload = {
        "schema_version": smoke.RUNTIME_PREFLIGHT_SCHEMA,
        "manifest_hash": manifest["manifest_hash"],
        "authorization_hash": authorization["authorization_hash"],
        "stage_b_id": manifest["stage_b_id"],
        "checked_at": "2026-08-13T00:00:00+00:00",
        "code_revision": "a" * 40,
        "source_tree_hash": manifest["implementation"]["source_tree_hash"],
        "worktree_clean": True,
        "worktree_status_hash": "b" * 64,
        "runtime_pins": manifest["runtime"]["pins"],
        "remote_identity": manifest["runtime"]["remote_identity"],
    }
    payload["preflight_hash"] = canonical_hash(payload)
    return payload


def _raw_result(manifest: dict, schedule_name: str, panel: dict) -> dict:
    planned = {
        attempt["policy_bundle_id"]: attempt
        for attempt in manifest["schedule"][schedule_name]["attempts"]
        if attempt["panel_id"] == panel["panel_id"]
    }
    attempts = {}
    for bundle_id in panel["execution_order"]:
        attempt = planned[bundle_id]
        binding = manifest["registry"]["policies"][attempt["policy_capability"]]
        treatment = TreatmentRegistry.load(REGISTRY_PATH).by_bundle_id(bundle_id)
        policy = policy_spec_from_treatment(treatment).to_dict()
        specialist_tool = (
            "semantic_table"
            if attempt["policy_capability"] == "table_specialist"
            else "semantic_form"
        )
        payload = {"rows": []}
        encoded = smoke.canonical_json(payload).encode("utf-8")
        attempts[bundle_id] = {
            "attempt_id": attempt["attempt_id"],
            "policy": policy,
            "pi_return_code": 0,
            "pi_stderr": "",
            "sampling_receipt": {
                "seed": panel["sampling_seed"],
                "parameters": manifest["runtime"]["sampling"]["parameters"],
            },
            "verification": {
                "success": True,
                "verifier_id": manifest["implementation"]["verifier_id"],
                "verifier_version": manifest["implementation"]["verifier_version"],
                "failure_code": None,
            },
            "usage": {"output": 10},
            "trajectory": {
                "provider_turn_count": 1,
                "tool_trace": [
                    {
                        "tool_name": specialist_tool,
                        "is_error": False,
                        "budget_rejected": False,
                        "operation_aborted": False,
                        "pre_execution_rejected": False,
                        "details": {
                            "semantic_payload": payload,
                            "semantic_specialist_receipt": {
                                "schema_version": "pyreplab-semantic-specialist-receipt-v1",
                                "specialist": attempt["policy_capability"],
                                "action": specialist_tool,
                                "delivered": True,
                                "payload_bytes": len(encoded),
                                "payload_sha256": hashlib.sha256(encoded).hexdigest(),
                            },
                        },
                    }
                ]
            },
            "timing": {},
        }
    return {
        "task_id": panel["task_id"],
        "mode": "treatment_set",
        "execution_order": panel["execution_order"],
        "attempts": attempts,
        "treatment_registry_hash": manifest["registry"]["registry_hash"],
        "rollout_replica": panel["replica"],
        "sampling_seed": panel["sampling_seed"],
        "pilot_manifest_hash": manifest["manifest_hash"],
        "pilot_panel_id": panel["panel_id"],
        "task_commitments": {
            "source_sha256": next(
                task["source_sha256"]
                for task in manifest["tasks"]
                if task["task_id"] == panel["task_id"]
            ),
            "probe_features_sha256": next(
                task["probe_features_sha256"]
                for task in manifest["tasks"]
                if task["task_id"] == panel["task_id"]
            ),
            "probe_receipt_sha256": next(
                task["probe_receipt_sha256"]
                for task in manifest["tasks"]
                if task["task_id"] == panel["task_id"]
            ),
        },
    }


def _raw_panel_record(
    manifest: dict,
    authorization: dict,
    preflight: dict,
    schedule_name: str,
    panel: dict,
    *,
    result: dict | None = None,
    infrastructure: bool = False,
) -> dict:
    record = {
        "schema_version": smoke.RAW_PANEL_SCHEMA,
        "manifest_hash": manifest["manifest_hash"],
        "authorization_hash": authorization["authorization_hash"],
        "preflight_hash": preflight["preflight_hash"],
        "stage_b_id": manifest["stage_b_id"],
        "schedule": schedule_name,
        "panel_id": panel["panel_id"],
        "block": panel["block"],
        "task_id": panel["task_id"],
        "replica": panel["replica"],
        "started_at": "2026-08-13T00:00:00+00:00",
        "finished_at": "2026-08-13T00:00:01+00:00",
        "duration_seconds": 1.0,
        "status": "infrastructure_invalid" if infrastructure else "completed",
        "result": None if infrastructure else result,
        "failure": (
            {
                "error_class": "infrastructure_invalid",
                "error_code": "browser_transport_error",
                "phase": "pi_controller",
                "attempt_id": next(
                    attempt["attempt_id"]
                    for attempt in manifest["schedule"][schedule_name]["attempts"]
                    if attempt["panel_id"] == panel["panel_id"]
                ),
                "type": "AttemptExecutionError",
            }
            if infrastructure
            else None
        ),
    }
    record["raw_record_hash"] = smoke._raw_record_hash(record)
    return record


class AnalyzeStageBTest(StageBAnalyzerTestBase):
    def test_analyze_pass(self) -> None:
        rows = self.routed_rows()
        report = analyze_stage_b(self.manifest, self.authorization, rows)
        self.assertTrue(report["valid"])
        self.assertTrue(report["gates_passed"])
        self.assertEqual(report["decision"], DECISION_PASS)
        self.assertEqual(
            validate_gate_report(report, self.manifest, self.authorization), []
        )
        analysis = report["analysis"]
        self.assertEqual(analysis["attempts"]["rows"], ATTEMPTS_PER_SCHEDULE)
        self.assertEqual(analysis["success"]["routed"], 1.0)
        self.assertEqual(analysis["success"]["best_fixed"], 0.5)
        self.assertEqual(analysis["success"]["lift"], 0.5)
        self.assertEqual(analysis["utility"]["pooled"]["lift"], 0.5)
        self.assertEqual(
            analysis["discordant_cells"]["overall"]["count"], 0
        )
        # prompt-only secondary report with each fixed arm
        self.assertIn("routed", analysis["prompt_only"])
        self.assertIn("form_specialist", analysis["prompt_only"]["fixed"])
        self.assertIn("table_specialist", analysis["prompt_only"]["fixed"])
        self.assertEqual(report["configured_draws"], 100000)
        self.assertEqual(analysis["utility"]["bootstrap"]["draws"], 100000)

    def test_analyze_no_go(self) -> None:
        # mechanically valid (all intention-to-treat failures) but below thresholds
        rows = craft_runtime_rows(self.manifest, self.authorization, success=False)
        report = analyze_stage_b(self.manifest, self.authorization, rows)
        self.assertTrue(report["valid"])
        self.assertFalse(report["gates_passed"])
        self.assertEqual(report["decision"], DECISION_NO_GO)
        self.assertEqual(
            validate_gate_report(report, self.manifest, self.authorization), []
        )

    def test_analyze_invalid_missing_row(self) -> None:
        report = analyze_stage_b(
            self.manifest, self.authorization, self.routed_rows()[:-1]
        )
        self.assertFalse(report["valid"])
        self.assertEqual(report["decision"], DECISION_INVALID)
        self.assertTrue(any("missing planned attempt" in e for e in report["errors"]))

    def test_analyze_invalid_duplicate(self) -> None:
        rows = self.routed_rows()
        rows[0]["attempt_id"] = rows[1]["attempt_id"]
        report = analyze_stage_b(self.manifest, self.authorization, rows)
        self.assertFalse(report["valid"])
        self.assertEqual(report["decision"], DECISION_INVALID)
        self.assertTrue(any("duplicate attempt_id" in e for e in report["errors"]))

    def test_analyze_invalid_identity(self) -> None:
        rows = self.routed_rows()
        rows[0]["fixture_id"] = "tampered-fixture"
        report = analyze_stage_b(self.manifest, self.authorization, rows)
        self.assertFalse(report["valid"])
        self.assertTrue(any("identity field" in e for e in report["errors"]))

    def test_analyze_invalid_governance(self) -> None:
        rows = self.routed_rows()
        rows[0]["canary_row_status"] = "T_train"
        report = analyze_stage_b(self.manifest, self.authorization, rows)
        self.assertFalse(report["valid"])
        self.assertTrue(any("identity field" in e for e in report["errors"]))

    def test_analyze_invalid_sampling(self) -> None:
        rows = self.routed_rows()
        rows[0]["sampling_receipt"]["seed"] += 1
        report = analyze_stage_b(self.manifest, self.authorization, rows)
        self.assertFalse(report["valid"])
        self.assertTrue(any("sampling_receipt" in e for e in report["errors"]))

        rows = self.routed_rows()
        rows[0]["sampling_receipt"]["parameters"]["repeat_penalty"] = True
        report = analyze_stage_b(self.manifest, self.authorization, rows)
        self.assertFalse(report["valid"])
        self.assertTrue(any("sampling_receipt" in e for e in report["errors"]))

    def test_analyze_invalid_cost(self) -> None:
        rows = self.routed_rows()
        rows[0]["usage"]["output"] = -1
        report = analyze_stage_b(self.manifest, self.authorization, rows)
        self.assertFalse(report["valid"])
        self.assertTrue(any("usage.output" in e for e in report["errors"]))

        rows = self.routed_rows()
        rows[0]["usage"]["output"] = True  # boolean, not a non-negative integer
        report = analyze_stage_b(self.manifest, self.authorization, rows)
        self.assertFalse(report["valid"])
        self.assertTrue(any("usage.output" in e for e in report["errors"]))

    def test_analyze_invalid_verifier(self) -> None:
        rows = self.routed_rows()
        rows[0]["verification"]["verifier_id"] = "tampered-verifier"
        report = analyze_stage_b(self.manifest, self.authorization, rows)
        self.assertFalse(report["valid"])
        self.assertTrue(any("verifier_id" in e for e in report["errors"]))

    def test_analyze_invalid_mechanism(self) -> None:
        for mutation in (
            {"infrastructure_errors": 1},
            {"unavailable_specialist_found": True},
            {"tool_cap_compliant": False},
            {"specialist_receipt_valid": False},
            {"specialist_receipt_valid": True, "specialist_action_match": False},
        ):
            rows = self.routed_rows()
            rows[0]["mechanism"].update(mutation)
            report = analyze_stage_b(self.manifest, self.authorization, rows)
            self.assertFalse(report["valid"], mutation)
            self.assertEqual(report["decision"], DECISION_INVALID, mutation)

    def test_itt_failure_remains_valid(self) -> None:
        target = None
        for attempt in self.manifest["schedule"]["primary"]["attempts"]:
            if attempt["policy_capability"] == self.route_of(attempt):
                target = attempt["attempt_id"]
                break
        self.assertIsNotNone(target)
        rows = craft_runtime_rows(
            self.manifest,
            self.authorization,
            success=lambda a: (
                a["policy_capability"] == self.route_of(a)
                if a["attempt_id"] != target
                else False
            ),
            error_class=lambda a: (
                "intention_to_treat_failure" if a["attempt_id"] == target else None
            ),
            error_code=lambda a: "refusal" if a["attempt_id"] == target else None,
        )
        report = analyze_stage_b(self.manifest, self.authorization, rows)
        self.assertTrue(report["valid"])
        self.assertEqual(
            validate_gate_report(report, self.manifest, self.authorization), []
        )
        self.assertEqual(report["analysis"]["discordant_cells"]["overall"]["count"], 1)

    def test_non_itt_error_class_makes_invalid(self) -> None:
        target = self.manifest["schedule"]["primary"]["attempts"][0]["attempt_id"]
        rows = craft_runtime_rows(
            self.manifest,
            self.authorization,
            success=lambda a: (
                a["policy_capability"] == self.route_of(a)
                if a["attempt_id"] != target
                else False
            ),
            error_class=lambda a: (
                "infrastructure_invalid" if a["attempt_id"] == target else None
            ),
            error_code=lambda a: "controller_error" if a["attempt_id"] == target else None,
        )
        report = analyze_stage_b(self.manifest, self.authorization, rows)
        self.assertFalse(report["valid"])
        self.assertEqual(report["decision"], DECISION_INVALID)
        self.assertTrue(any("intention_to_treat_failure" in e for e in report["errors"]))

    def test_global_not_per_task_comparator(self) -> None:
        # table succeeds on 14 tasks, form on 12, but a per-task hindsight
        # oracle would succeed on all 24; the comparator must be a single
        # global fixed specialist, never per-task.
        table_routed = set()
        form_routed = set()
        for task in self.manifest["tasks"]:
            route = self.manifest["route_receipts"][task["task_id"]]["combined_route"]
            if route == "table_specialist":
                table_routed.add(task["task_id"])
            else:
                form_routed.add(task["task_id"])
        extra_table = sorted(form_routed)[:2]

        def success(attempt: dict) -> bool:
            task_id = attempt["task_id"]
            if attempt["policy_capability"] == "table_specialist":
                return task_id in table_routed or task_id in extra_table
            return task_id in form_routed

        report = analyze_stage_b(
            self.manifest, self.authorization,
            craft_runtime_rows(self.manifest, self.authorization, success=success),
        )
        self.assertTrue(report["valid"])
        self.assertEqual(
            report["analysis"]["success"]["best_fixed_policy"], "table_specialist"
        )
        self.assertEqual(report["analysis"]["success"]["best_fixed"], 14 / 24)
        self.assertEqual(
            report["analysis"]["success"]["fixed"]["form_specialist"], 12 / 24
        )
        self.assertEqual(report["analysis"]["success"]["routed"], 1.0)

    def test_deterministic_bootstrap_and_report(self) -> None:
        rows = self.routed_rows()
        first = analyze_stage_b(self.manifest, self.authorization, rows)
        second = analyze_stage_b(self.manifest, self.authorization, rows)
        self.assertEqual(canonical_json(first), canonical_json(second))
        self.assertEqual(
            validate_gate_report(first, self.manifest, self.authorization), []
        )
        lower_bound = first["analysis"]["utility"]["bootstrap"]["lower_bound"]
        self.assertGreater(lower_bound, 0.0)
        self.assertLess(lower_bound, 0.5)
        self.assertEqual(first["analysis"]["utility"]["bootstrap"]["seed"], 2026081302)

    def test_nonpositive_bootstrap_draws_rejected(self) -> None:
        rows = self.routed_rows()
        for draws in (0, -1, True):
            with self.subTest(draws=draws):
                with self.assertRaises(ValueError):
                    analyze_stage_b(
                        self.manifest,
                        self.authorization,
                        rows,
                        _bootstrap_draws=draws,
                    )

    def test_tamper_report_detected(self) -> None:
        report = analyze_stage_b(self.manifest, self.authorization, self.routed_rows())

        tampered = json.loads(json.dumps(report))
        tampered["report_hash"] = "0" * 64
        self.assertTrue(
            any("report_hash" in e for e in validate_gate_report(tampered, self.manifest, self.authorization))
        )

        tampered = json.loads(json.dumps(report))
        tampered["decision"] = DECISION_NO_GO
        self.assertTrue(
            any("incoherent" in e for e in validate_gate_report(tampered, self.manifest, self.authorization))
        )

        tampered = json.loads(json.dumps(report))
        tampered["configured_draws"] = 50
        self.assertTrue(
            any("configured_draws" in e for e in validate_gate_report(tampered, self.manifest, self.authorization))
        )

        tampered = json.loads(json.dumps(report))
        tampered["checks"][0]["passed"] = False
        self.assertTrue(
            any("valid flag" in e for e in validate_gate_report(tampered, self.manifest, self.authorization))
        )

        tampered = json.loads(json.dumps(report))
        tampered["checks"][0]["passed"] = 1
        tampered["report_hash"] = canonical_hash(
            {k: v for k, v in tampered.items() if k != "report_hash"}
        )
        self.assertTrue(
            any("must be a boolean" in e for e in validate_gate_report(tampered, self.manifest, self.authorization))
        )


class ExecutionReceiptTest(StageBAnalyzerTestBase):
    def test_build_and_validate(self) -> None:
        receipt = build_execution_receipt(
            self.manifest, self.authorization, {0: "primary", 1: "primary"}
        )
        self.assertEqual(receipt["schema_version"], EXECUTION_RECEIPT_SCHEMA)
        self.assertEqual(
            validate_execution_receipt(receipt, self.manifest, self.authorization), []
        )
        self.assertEqual(receipt["manifest_hash"], self.manifest["manifest_hash"])
        self.assertEqual(
            receipt["authorization_hash"], self.authorization["authorization_hash"]
        )
        self.assertEqual(len(receipt["selected_attempt_ids"]), ATTEMPTS_PER_SCHEDULE)
        self.assertEqual(receipt["quarantined_primary_attempt_ids"], {})
        self.assertEqual(receipt["replacement_triggers"], {})

    def test_contingency_quarantines_full_primary_block(self) -> None:
        receipt = build_execution_receipt(
            self.manifest,
            self.authorization,
            {0: "contingency", 1: "primary"},
            {
                0: {
                    "error_class": "infrastructure_invalid",
                    "error_code": "browser_transport_error",
                    "attempt_id": "primary-trigger-attempt",
                    "phase": "pi_controller",
                    "raw_record_hash": "a" * 64,
                }
            },
        )
        expected = [
            a["attempt_id"]
            for a in self.manifest["schedule"]["primary"]["attempts"]
            if a["block"] == 0
        ]
        self.assertEqual(
            receipt["quarantined_primary_attempt_ids"]["0"], expected
        )
        self.assertEqual(len(expected), ATTEMPTS_PER_SCHEDULE // 2)
        self.assertEqual(
            receipt["selected_attempt_ids_by_block"]["0"],
            [
                a["attempt_id"]
                for a in self.manifest["schedule"]["contingency"]["attempts"]
                if a["block"] == 0
            ],
        )

    def test_replacement_trigger_must_be_infrastructure(self) -> None:
        with self.assertRaises(ValueError):
            build_execution_receipt(
                self.manifest,
                self.authorization,
                {0: "contingency", 1: "primary"},
                {0: {"error_class": "probe_invalid", "error_code": "probe_failure"}},
            )

    def test_primary_block_cannot_have_trigger(self) -> None:
        with self.assertRaises(ValueError):
            build_execution_receipt(
                self.manifest,
                self.authorization,
                {0: "primary", 1: "primary"},
                {1: {"error_class": "infrastructure_invalid", "error_code": "controller_error"}},
            )

    def test_contingency_requires_trigger(self) -> None:
        with self.assertRaises(ValueError):
            build_execution_receipt(
                self.manifest, self.authorization, {0: "contingency", 1: "primary"}
            )

    def test_tampered_receipt_detected(self) -> None:
        receipt = build_execution_receipt(
            self.manifest, self.authorization, {0: "primary", 1: "primary"}
        )
        tampered = json.loads(json.dumps(receipt))
        tampered["block_schedules"] = {"0": "contingency", "1": "primary"}
        self.assertTrue(
            validate_execution_receipt(tampered, self.manifest, self.authorization)
        )


class SafeExportTest(StageBAnalyzerTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.preflight = _runtime_preflight(self.manifest, self.authorization)
        self.assertEqual(
            validate_runtime_preflight(
                self.preflight, self.manifest, self.authorization
            ),
            [],
        )

    def records(self, block_schedules: dict[str, str]) -> list[dict]:
        records = []
        for block in (0, 1):
            schedule_name = block_schedules[str(block)]
            for panel in self.manifest["schedule"][schedule_name]["panels"]:
                if panel["block"] != block:
                    continue
                records.append(
                    _raw_panel_record(
                        self.manifest,
                        self.authorization,
                        self.preflight,
                        schedule_name,
                        panel,
                        result=_raw_result(self.manifest, schedule_name, panel),
                    )
                )
        return records

    def test_safe_export_whitelists_attempt_rows(self) -> None:
        execution = build_execution_receipt(
            self.manifest, self.authorization, {0: "primary", 1: "primary"}
        )
        rows, receipt = export_safe_attempts(
            self.manifest,
            self.authorization,
            self.preflight,
            self.records({"0": "primary", "1": "primary"}),
            execution,
        )
        self.assertEqual(len(rows), ATTEMPTS_PER_SCHEDULE)
        self.assertEqual(receipt["schema_version"], SAFE_EXPORT_SCHEMA)
        self.assertNotIn("pi_stderr", canonical_json(rows))
        self.assertNotIn("semantic_payload", canonical_json(rows))
        self.assertTrue(
            all(row["provenance"].get("raw_record_hash") for row in rows)
        )
        report = analyze_stage_b(
            self.manifest,
            self.authorization,
            rows,
            execution_receipt=execution,
            _bootstrap_draws=10,
        )
        self.assertTrue(report["valid"])

    def test_safe_export_rejects_raw_tamper(self) -> None:
        execution = build_execution_receipt(
            self.manifest, self.authorization, {0: "primary", 1: "primary"}
        )
        records = self.records({"0": "primary", "1": "primary"})
        records[0]["result"]["sampling_seed"] += 1
        records[0]["raw_record_hash"] = smoke._raw_record_hash(records[0])
        with self.assertRaisesRegex(ValueError, "sampling_seed"):
            export_safe_attempts(
                self.manifest,
                self.authorization,
                self.preflight,
                records,
                execution,
            )

    def test_safe_export_rejects_runtime_commitment_drift(self) -> None:
        execution = build_execution_receipt(
            self.manifest, self.authorization, {0: "primary", 1: "primary"}
        )
        records = self.records({"0": "primary", "1": "primary"})
        records[0]["result"]["task_commitments"]["source_sha256"] = "0" * 64
        records[0]["raw_record_hash"] = smoke._raw_record_hash(records[0])
        with self.assertRaisesRegex(ValueError, "commitments"):
            export_safe_attempts(
                self.manifest,
                self.authorization,
                self.preflight,
                records,
                execution,
            )

    def test_safe_export_rejects_harness_failure_code(self) -> None:
        execution = build_execution_receipt(
            self.manifest, self.authorization, {0: "primary", 1: "primary"}
        )
        records = self.records({"0": "primary", "1": "primary"})
        item = next(iter(records[0]["result"]["attempts"].values()))
        item["verification"]["success"] = False
        item["verification"]["failure_code"] = "oracle_unreadable"
        records[0]["raw_record_hash"] = smoke._raw_record_hash(records[0])
        with self.assertRaisesRegex(ValueError, "not a frozen intention-to-treat"):
            export_safe_attempts(
                self.manifest,
                self.authorization,
                self.preflight,
                records,
                execution,
            )

    def test_safe_export_rejects_malformed_specialist_receipt(self) -> None:
        execution = build_execution_receipt(
            self.manifest, self.authorization, {0: "primary", 1: "primary"}
        )
        records = self.records({"0": "primary", "1": "primary"})
        item = next(iter(records[0]["result"]["attempts"].values()))
        item["trajectory"]["tool_trace"][0]["details"].pop(
            "semantic_specialist_receipt"
        )
        records[0]["raw_record_hash"] = smoke._raw_record_hash(records[0])
        with self.assertRaisesRegex(ValueError, "invalid specialist receipt"):
            export_safe_attempts(
                self.manifest,
                self.authorization,
                self.preflight,
                records,
                execution,
            )

    def test_raw_resume_rejects_out_of_order_ledger(self) -> None:
        records = self.records({"0": "primary", "1": "primary"})
        raw = self.root / "raw.jsonl"
        immutable_write_jsonl(raw, list(reversed(records[:2])))
        with self.assertRaisesRegex(ValueError, "chronology"):
            smoke._existing_raw_state(
                raw, self.manifest, self.authorization, self.preflight
            )


class LiveRunnerTest(StageBAnalyzerTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.manifest_path = self.root / "manifest.json"
        self.authorization_path = self.root / "authorization.json"
        immutable_write(self.manifest_path, self.manifest)
        immutable_write(self.authorization_path, self.authorization)
        self.preflight = _runtime_preflight(self.manifest, self.authorization)

    def paths(self) -> dict[str, Path]:
        return {
            "raw": self.root / "raw.jsonl",
            "safe": self.root / "safe.jsonl",
            "execution": self.root / "execution.json",
            "safe_receipt": self.root / "safe-receipt.json",
            "preflight": self.root / "preflight.json",
        }

    def invoke(self, runner) -> tuple[dict, dict[str, Path]]:
        paths = self.paths()
        with mock.patch.object(
            smoke, "build_runtime_preflight", return_value=self.preflight
        ), mock.patch.object(
            smoke, "run_registered_treatments", side_effect=runner
        ):
            result = run_stage_b(
                self.manifest_path,
                self.authorization_path,
                REGISTRY_PATH,
                self.sa_manifest_path,
                self.sa_report_path,
                paths["raw"],
                paths["safe"],
                paths["execution"],
                paths["safe_receipt"],
                paths["preflight"],
                pi_binary="pi",
                provider=self.manifest["runtime"]["pins"]["provider"],
                model=self.manifest["runtime"]["pins"]["model_alias"],
                thinking=self.manifest["runtime"]["pins"]["thinking"],
                unbrowser_binary=self.manifest["runtime"]["pins"]["unbrowser_path"],
                model_artifact=self.manifest["runtime"]["pins"]["model_artifact_path"],
                llama_server_binary=self.manifest["runtime"]["pins"]["llama_server_path"],
            )
        return result, paths

    def test_live_runner_primary_end_to_end(self) -> None:
        def runner(_root, _config, args):
            panel = next(
                panel
                for panel in self.manifest["schedule"]["primary"]["panels"]
                if panel["panel_id"] == args.pilot_panel_id
            )
            return _raw_result(self.manifest, "primary", panel)

        result, paths = self.invoke(runner)
        self.assertEqual(result["safe_attempts"], ATTEMPTS_PER_SCHEDULE)
        self.assertEqual(result["block_schedules"], {"0": "primary", "1": "primary"})
        self.assertEqual(len(paths["raw"].read_text().splitlines()), PANELS_PER_SCHEDULE)
        self.assertEqual(len(load_safe_attempts_jsonl(paths["safe"])), ATTEMPTS_PER_SCHEDULE)

    def test_live_runner_replaces_complete_block_once(self) -> None:
        failed = False

        def runner(_root, _config, args):
            nonlocal failed
            if not failed:
                panel = next(
                    panel
                    for panel in self.manifest["schedule"]["primary"]["panels"]
                    if panel["panel_id"] == args.pilot_panel_id
                )
                if panel["block"] == 0:
                    failed = True
                    raise AttemptExecutionError(
                        "browser died",
                        error_class="infrastructure_invalid",
                        error_code="browser_transport_error",
                        phase="pi_controller",
                    )
            for schedule_name in ("primary", "contingency"):
                match = next(
                    (
                        panel
                        for panel in self.manifest["schedule"][schedule_name]["panels"]
                        if panel["panel_id"] == args.pilot_panel_id
                    ),
                    None,
                )
                if match is not None:
                    return _raw_result(self.manifest, schedule_name, match)
            self.fail("unknown panel")

        result, paths = self.invoke(runner)
        self.assertEqual(result["block_schedules"], {"0": "contingency", "1": "primary"})
        receipt = json.loads(paths["execution"].read_text())
        self.assertEqual(len(receipt["quarantined_primary_attempt_ids"]["0"]), 48)
        safe_rows = load_safe_attempts_jsonl(paths["safe"])
        self.assertEqual(len(safe_rows), ATTEMPTS_PER_SCHEDULE)
        self.assertTrue(
            all(row["schedule"] == "contingency" for row in safe_rows if row["block"] == 0)
        )

    def test_live_runner_replaces_on_recorded_browser_transport_error(self) -> None:
        injected = False

        def runner(_root, _config, args):
            nonlocal injected
            for schedule_name in ("primary", "contingency"):
                panel = next(
                    (
                        candidate
                        for candidate in self.manifest["schedule"][schedule_name]["panels"]
                        if candidate["panel_id"] == args.pilot_panel_id
                    ),
                    None,
                )
                if panel is None:
                    continue
                result = _raw_result(self.manifest, schedule_name, panel)
                if schedule_name == "primary" and panel["block"] == 0 and not injected:
                    injected = True
                    item = next(iter(result["attempts"].values()))
                    item["trajectory"]["tool_trace"][0]["details"] = {
                        "error": "browser process exited",
                        "infrastructure_error": True,
                    }
                return result
            self.fail("unknown panel")

        result, _paths = self.invoke(runner)
        self.assertEqual(
            result["block_schedules"], {"0": "contingency", "1": "primary"}
        )

    def test_live_runner_replaces_on_provider_transport_error(self) -> None:
        injected = False

        def runner(_root, _config, args):
            nonlocal injected
            for schedule_name in ("primary", "contingency"):
                panel = next(
                    (
                        candidate
                        for candidate in self.manifest["schedule"][schedule_name]["panels"]
                        if candidate["panel_id"] == args.pilot_panel_id
                    ),
                    None,
                )
                if panel is None:
                    continue
                result = _raw_result(self.manifest, schedule_name, panel)
                if schedule_name == "primary" and panel["block"] == 0 and not injected:
                    injected = True
                    item = next(iter(result["attempts"].values()))
                    item["pi_return_code"] = 1
                    item["trajectory"]["provider_turn_count"] = 0
                    item["trajectory"]["tool_trace"] = []
                return result
            self.fail("unknown panel")

        result, _paths = self.invoke(runner)
        self.assertEqual(
            result["block_schedules"], {"0": "contingency", "1": "primary"}
        )


class ContingencyAnalysisTest(StageBAnalyzerTestBase):
    def _contingency_receipt(self) -> dict:
        return build_execution_receipt(
            self.manifest,
            self.authorization,
            {0: "contingency", 1: "primary"},
            {
                0: {
                    "error_class": "infrastructure_invalid",
                    "error_code": "browser_transport_error",
                    "attempt_id": "primary-trigger-attempt",
                    "phase": "pi_controller",
                    "raw_record_hash": "a" * 64,
                }
            },
        )

    def test_contingency_valid(self) -> None:
        receipt = self._contingency_receipt()
        rows = self.routed_rows(block_schedules={0: "contingency", 1: "primary"})
        report = analyze_stage_b(
            self.manifest, self.authorization, rows, execution_receipt=receipt
        )
        self.assertTrue(report["valid"])
        self.assertEqual(report["decision"], DECISION_PASS)
        self.assertEqual(
            validate_gate_report(report, self.manifest, self.authorization), []
        )
        self.assertEqual(report["analysis"]["schedules"], {"0": "contingency", "1": "primary"})

    def test_contingency_rows_without_receipt_invalid(self) -> None:
        rows = self.routed_rows(block_schedules={0: "contingency", 1: "primary"})
        report = analyze_stage_b(self.manifest, self.authorization, rows)
        self.assertFalse(report["valid"])
        self.assertEqual(report["decision"], DECISION_INVALID)

    def test_contingency_schedule_mismatch_invalid(self) -> None:
        receipt = self._contingency_receipt()
        rows = self.routed_rows()  # primary rows for both blocks
        report = analyze_stage_b(
            self.manifest, self.authorization, rows, execution_receipt=receipt
        )
        self.assertFalse(report["valid"])
        self.assertEqual(report["decision"], DECISION_INVALID)


class LoadAttemptsTest(StageBAnalyzerTestBase):
    def test_load_safe_attempts_jsonl(self) -> None:
        rows = self.routed_rows()
        path = self.write_attempts_jsonl(rows)
        loaded = load_safe_attempts_jsonl(path)
        self.assertEqual(loaded, rows)
        self.assertEqual(len(loaded), ATTEMPTS_PER_SCHEDULE)

    def test_load_rejects_non_object_line(self) -> None:
        path = self.root / "bad.jsonl"
        path.write_text('"just a string"\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            load_safe_attempts_jsonl(path)


class AnalyzeCliTest(StageBAnalyzerTestBase):
    def _write_artifacts(self, rows: list[dict] | None = None) -> tuple:
        manifest_path = self.root / "manifest.json"
        auth_path = self.root / "authorization.json"
        report_path = self.root / "report.json"
        immutable_write(manifest_path, self.manifest)
        immutable_write(auth_path, self.authorization)
        attempts_path = self.write_attempts_jsonl(
            rows if rows is not None else self.routed_rows()
        )
        return manifest_path, auth_path, attempts_path, report_path

    def test_cli_analyze_and_validate_gate(self) -> None:
        manifest_path, auth_path, attempts_path, report_path = self._write_artifacts()
        with redirect_stdout(StringIO()):
            self.assertEqual(
                main(["analyze", str(manifest_path), str(auth_path),
                      str(attempts_path), str(report_path)]),
                0,
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["decision"], DECISION_PASS)
        with redirect_stdout(StringIO()):
            self.assertEqual(
                main(["validate-gate", str(report_path), str(manifest_path),
                      str(auth_path)]),
                0,
            )

    def test_cli_analyze_no_go_exit_2(self) -> None:
        rows = craft_runtime_rows(self.manifest, self.authorization, success=False)
        manifest_path, auth_path, attempts_path, report_path = self._write_artifacts(rows)
        with redirect_stdout(StringIO()):
            self.assertEqual(
                main(["analyze", str(manifest_path), str(auth_path),
                      str(attempts_path), str(report_path)]),
                2,
            )

    def test_cli_analyze_invalid_exit_1(self) -> None:
        rows = self.routed_rows()[:-1]
        manifest_path, auth_path, attempts_path, report_path = self._write_artifacts(rows)
        with redirect_stdout(StringIO()):
            self.assertEqual(
                main(["analyze", str(manifest_path), str(auth_path),
                      str(attempts_path), str(report_path)]),
                1,
            )

    def test_cli_analyze_with_execution_receipt(self) -> None:
        receipt = build_execution_receipt(
            self.manifest, self.authorization, {0: "primary", 1: "primary"}
        )
        receipt_path = self.root / "receipt.json"
        immutable_write(receipt_path, receipt)
        manifest_path, auth_path, attempts_path, report_path = self._write_artifacts()
        with redirect_stdout(StringIO()):
            self.assertEqual(
                main(["analyze", str(manifest_path), str(auth_path),
                      str(attempts_path), str(report_path),
                      "--execution-receipt", str(receipt_path)]),
                0,
            )


if __name__ == "__main__":
    unittest.main()
