"""Tests for the stdlib-only m3 semantic dataset package builder.

All fixtures are synthetic: two specialist treatments (table/form), a
self-hashed policy split, a ``T_canary`` mechanics manifest, complete or
partial panel records, a synthetic preflight and authoritative gate report,
and a minimal raw run root.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from pyreplab_harness.m3_exploratory_screen import (
    PANEL_RESULT_SCHEMA,
    build_screen_manifest,
)
from pyreplab_harness.m3_semantic_dataset import (
    ATTEMPT_SCHEMA,
    CONTRACT_SCHEMA,
    PACKAGE_SCHEMA,
    build_package,
    load_contract,
    privacy_scan,
    verify_package,
)
from pyreplab_harness.treatments import TreatmentRegistry, TreatmentSpec

_TABLE_PAYLOAD = {"rows": [{"a": 1, "b": 2}], "columns": ["a", "b"], "shape": [1, 2]}
_FORM_PAYLOAD = {"fields": [{"name": "email", "type": "text"}], "action": "/submit"}


def _canonical_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _encode_payload(payload: dict[str, Any]) -> tuple[int, str]:
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return len(encoded), hashlib.sha256(encoded).hexdigest()


def _make_treatment(
    bundle_id: str,
    hash_prefix: str,
    capability: str,
    tool_interface: str,
    allowed_tools: tuple[str, ...],
) -> TreatmentSpec:
    bundle_hash = hash_prefix + "a" * (64 - len(hash_prefix))
    return TreatmentSpec(
        id=bundle_id,
        version="1-canary-excluded",
        system_prompt=(
            "Planning: direct\nCapability: specialist_assigned\n"
            "Verification: submit_directly\nRecovery: fail_fast\n"
            "Safety: Workspace only.\n"
        ),
        allowed_tools=allowed_tools,
        max_output_tokens=4096,
        tool_call_limit=6,
        command_timeout_seconds=60,
        wall_time_limit_seconds=600,
        tool_interface=tool_interface,
        generator_metadata={
            "grammar_name": "unbrowser_semantic_capability_canary",
            "grammar_version": "m3-semantic-canary-v1",
            "planning": "direct",
            "capability": capability,
            "verification": "submit_directly",
            "recovery": "fail_fast",
            "tool_cap": "lean",
            "parent_bundle_id": "parent-common-001",
            "substrate": "public_html",
            "observation_mechanism": "controller_owned_public_html_semantic_operation",
        },
    )


def _make_registry() -> tuple[TreatmentRegistry, TreatmentSpec, TreatmentSpec]:
    table = _make_treatment(
        "semantic-table-specialist",
        "tb",
        "table_specialist",
        "native_bash_unbrowser_semantic_table_v1",
        ("bash", "unbrowser", "semantic_table"),
    )
    form = _make_treatment(
        "semantic-form-specialist",
        "fm",
        "form_specialist",
        "native_bash_unbrowser_semantic_form_v1",
        ("bash", "unbrowser", "semantic_form"),
    )
    return TreatmentRegistry((table, form)), table, form


def _make_policy_split(registry: TreatmentRegistry) -> dict[str, Any]:
    bundle_ids = [t.bundle_id for t in registry]
    payload: dict[str, Any] = {
        "grammar_name": "unbrowser_semantic_capability_canary",
        "grammar_version": "m3-semantic-canary-v1",
        "policy_version": "1-canary-excluded",
        "registry_file": "treatments.json",
        "registry_hash": registry.registry_hash,
        "schema_version": "m3-policy-split-v1",
        "split_algorithm": "semantic-canary-isolated-v1",
        "split_seed": 20260812,
        "splits": {"development": [], "final_held_out": [], "meta_train": list(bundle_ids)},
    }
    payload["manifest_hash"] = _canonical_hash(payload)
    return payload


def _make_manifest(
    registry: TreatmentRegistry, policy_split: dict[str, Any]
) -> dict[str, Any]:
    table_t, form_t = registry.treatments
    treatments_by_cap = {
        str(t.generator_metadata["capability"]): t for t in registry
    }
    protocol: dict[str, Any] = {
        "schema_version": "m3-semantic-capability-protocol-v1",
        "stage": "mechanics_dry_run",
        "claim_boundary": "screening_futility_only",
        "mechanism": {
            "name": "controller_owned_public_html_semantic_operation",
            "receipt_schema_version": "pyreplab-semantic-specialist-receipt-v1",
        },
        "parent_bundle_ids": {
            level: str(t.generator_metadata["parent_bundle_id"])
            for level, t in treatments_by_cap.items()
        },
        "decision_rule": {"all_attempts_mechanically_valid": True},
    }
    spec: dict[str, Any] = {
        "screen_id": "test-semantic-canary-001",
        "purpose": "Synthetic semantic capability canary mechanics",
        "remote_identity": {
            "host": "test-host",
            "project": "/remote/test-project",
            "run_root": "/remote/test-runs",
            "python": "python3",
        },
        "policy_bundle_ids": [t.bundle_id for t in registry],
        "tasks": [
            {"template": "table_filter_sort", "difficulty": "easy", "seed": 1001},
            {"template": "form_entry_validation", "difficulty": "easy", "seed": 1002},
        ],
        "rollout_replicas": 1,
        "sampling_seed_start": 5000,
        "schedule_seed": 42,
        "task_role": "T_canary",
        "protocol": protocol,
        "selection": {"reason": "synthetic semantic capability canary test"},
    }
    manifest = build_screen_manifest(
        registry,
        policy_split,
        spec,
        registry_file="treatments.json",
        policy_split_file="split.json",
    )
    # build_screen_manifest already embeds protocol and manifest_hash.
    assert manifest["task_role"] == "T_canary"
    assert table_t.bundle_id in manifest["policy_bundle_ids"]
    assert form_t.bundle_id in manifest["policy_bundle_ids"]
    return manifest


def _make_contract(manifest: dict[str, Any], registry: TreatmentRegistry, policy_split: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": CONTRACT_SCHEMA,
        "dataset_id": "test-semantic-canary-001",
        "task_role": "T_canary",
        "registry_hash": registry.registry_hash,
        "policy_split_manifest_hash": policy_split["manifest_hash"],
        "planned": {
            "tasks": [str(t["task_id"]) for t in manifest["tasks"]],
            "bundle_ids": [str(bid) for bid in manifest["policy_bundle_ids"]],
            "rollout_replicas": int(manifest["rollout_replicas"]),
            "attempts": int(manifest["gates"]["attempts"]),
        },
        "purpose": "synthetic test contract",
    }
    payload["contract_hash"] = _canonical_hash(payload)
    return payload


def _make_preflight(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "manifest_hash": manifest["manifest_hash"],
        "screen_preflight": True,
        "checked_at": "2026-01-01T00:00:00Z",
        "code_revision": "a" * 40,
        "source_tree_hash": "b" * 64,
        "worktree_clean": True,
        "worktree_status_hash": hashlib.sha256(
            b"PYREPLAB_GIT_WORKTREE_CLEAN_MARKER_V1"
        ).hexdigest(),
        "runtime_pins": manifest["runtime_pins"],
        "remote_identity": manifest["remote_identity"],
    }


def _make_gate_report(
    manifest: dict[str, Any], decision: str = "mechanics_pass"
) -> dict[str, Any]:
    return {
        "gate": "m3-semantic-capability-gate-v1",
        "schema_version": "m3-semantic-capability-gate-v1",
        "manifest_hash": manifest["manifest_hash"],
        "stage": "mechanics_dry_run",
        "passed": decision in ("mechanics_pass", "screen_pass"),
        "decision": decision,
        "checks": {},
        "reasons": [],
        "completeness": {},
        "mechanism": {"structural_errors": [], "mechanisms_errors": []},
        "stability": {},
        "descriptive_outcomes": {},
        "warning": "screening only",
    }


def _specialist_trace(specialist: str) -> dict[str, Any]:
    payload = _TABLE_PAYLOAD if specialist == "table_specialist" else _FORM_PAYLOAD
    payload_bytes, payload_sha256 = _encode_payload(payload)
    action = "semantic_table" if specialist == "table_specialist" else "semantic_form"
    entry = {
        "tool_name": action,
        "is_error": False,
        "budget_rejected": False,
        "operation_aborted": False,
        "details": {
            "status": 200,
            "url": "http://127.0.0.1:PORT/test/page_0",
            "semantic_specialist_receipt": {
                "schema_version": "pyreplab-semantic-specialist-receipt-v1",
                "specialist": specialist,
                "action": action,
                "delivered": True,
                "payload_bytes": payload_bytes,
                "payload_sha256": payload_sha256,
            },
            "semantic_payload": payload,
        },
    }
    submit = {
        "tool_name": "bash",
        "is_error": False,
        "budget_rejected": False,
        "operation_aborted": False,
        "details": {"result_submission": True, "status": 0},
    }
    return {
        "planning_preamble": {"present": False},
        "tool_trace": [entry, submit],
        "provider_turn_count": 2,
        "tool_call_count": 2,
        "tool_limit_rejection_count": 0,
        "length_stop_count": 0,
    }


def _make_records(
    manifest: dict[str, Any],
    registry: TreatmentRegistry,
    *,
    skip_panel_ids: tuple[str, ...] = (),
    success: bool = True,
) -> list[dict[str, Any]]:
    """Build one completed panel record per manifest panel (unless skipped)."""
    runtime_pins = manifest["runtime_pins"]
    table_t, form_t = registry.treatments
    treatment_by_bundle = {t.bundle_id: t for t in registry}
    records: list[dict[str, Any]] = []
    cell_index = 0
    for panel in manifest["panels"]:
        if str(panel["panel_id"]) in skip_panel_ids:
            cell_index += len(panel["execution_order"])
            continue
        task = next(t for t in manifest["tasks"] if t["task_id"] == panel["task_id"])
        attempts: dict[str, dict[str, Any]] = {}
        for bid in panel["execution_order"]:
            treatment = treatment_by_bundle[bid]
            specialist = str(treatment.generator_metadata["capability"])
            attempts[bid] = {
                "attempt_id": f"attempt-{cell_index:04d}",
                "pi_return_code": 0,
                "pi_stderr": "PRIVATE_STDERR_SHOULD_NOT_APPEAR",
                "verification": {
                    "success": success,
                    "details": {},
                    "verifier_id": runtime_pins["fixture_verifier_id"],
                    "verifier_version": runtime_pins["fixture_verifier_version"],
                },
                "usage": {"output": 100, "prompt_tokens": 50},
                "trajectory": _specialist_trace(specialist),
                "sampling_receipt": {
                    "seed": panel["sampling_seed"],
                    "parameters": runtime_pins["sampling"]["parameters"],
                },
            }
            cell_index += 1
        record = {
            "schema_version": PANEL_RESULT_SCHEMA,
            "panel_id": str(panel["panel_id"]),
            "manifest_hash": manifest["manifest_hash"],
            "task": task,
            "panel": panel,
            "status": "completed",
            "started_at": "2026-01-01T00:00:00Z",
            "finished_at": "2026-01-01T00:01:00Z",
            "duration_seconds": 60.0,
            "result": {
                "task_id": panel["task_id"],
                "mode": "treatment_set",
                "execution_order": panel["execution_order"],
                "attempts": attempts,
                "pilot_manifest_hash": manifest["manifest_hash"],
                "pilot_panel_id": str(panel["panel_id"]),
                "rollout_replica": panel["rollout_replica"],
                "sampling_seed": panel["sampling_seed"],
                "treatment_registry_hash": registry.registry_hash,
            },
        }
        records.append(record)
    return records


def _make_raw_root(
    manifest: dict[str, Any], registry: TreatmentRegistry, raw_root: Path
) -> list[str]:
    """Write one raw file per planned cell into ``raw_root``; return attempt ids."""
    raw_root.mkdir(parents=True, exist_ok=True)
    attempts_dir = raw_root / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    attempt_ids: list[str] = []
    safe_rows: list[dict[str, Any]] = []
    treatment_by_bundle = {t.bundle_id: t for t in registry}
    task_by_id = {str(task["task_id"]): task for task in manifest["tasks"]}
    cell_index = 0
    for panel in manifest["panels"]:
        for bid in panel["execution_order"]:
            aid = f"attempt-{cell_index:04d}"
            attempt_ids.append(aid)
            treatment = treatment_by_bundle[bid]
            task = task_by_id[str(panel["task_id"])]
            attempt_dir = attempts_dir / aid
            attempt_dir.mkdir()
            (attempt_dir / "pi-events.jsonl").write_text(
                json.dumps({"attempt_id": aid, "bundle_id": bid}, sort_keys=True),
                encoding="utf-8",
            )
            safe_rows.append({
                "attempt_id": aid,
                "task_id": task["task_id"],
                "family": "unbrowser_fixture",
                "template_id": task["template"],
                "generator_version": "unbrowser-fixture-v2",
                "seed": task["seed"],
                "difficulty": task["difficulty"],
                "prompt": "Synthetic fixture prompt.",
                "contract": ["Write result.json."],
                "public_metadata": {
                    "difficulty": task["difficulty"],
                    "fixture_url": "http://127.0.0.1:18090/test",
                    "network_mode": "fixed-page-interactive-fixture",
                    "page_description": "Synthetic fixture.",
                    "required_output": "result.json",
                    "task_role": "T_canary",
                    "template": task["template"],
                },
                "policy_id": treatment.id,
                "policy_version": treatment.version,
                "treatment_bundle_id": treatment.bundle_id,
                "treatment_bundle_hash": treatment.bundle_hash,
                "treatment_registry_hash": registry.registry_hash,
                "rollout_replica": panel["rollout_replica"],
                "sampling_seed": panel["sampling_seed"],
                "pilot_manifest_hash": manifest["manifest_hash"],
                "pilot_panel_id": panel["panel_id"],
                "verified_success": True,
                "failure_code": None,
                "verifier_id": manifest["runtime_pins"]["fixture_verifier_id"],
                "verifier_version": manifest["runtime_pins"]["fixture_verifier_version"],
                "usage": {"input": 50, "output": 100, "total_tokens": 150},
                "output_token_cost": 100,
                "assistant_message_count": 2,
                "provider_turn_count": 2,
                "tool_call_count": 2,
                "tool_limit_rejection_count": 0,
                "length_stop_count": 0,
                "final_text_length": 10,
                "termination_class": "normal_completion",
                "task_role": "T_canary",
                "split": "canary_excluded",
                "governance_role": "canary_excluded",
                "eligibility": {
                    "training": False,
                    "calibration": False,
                    "development": False,
                    "final": False,
                },
            })
            cell_index += 1
    (raw_root / "attempts.safe.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in safe_rows
        ),
        encoding="utf-8",
    )
    return attempt_ids


class _Fixture:
    """Bundles all synthetic inputs for a build in one temp directory."""

    def __init__(self, tmp: str) -> None:
        self.root = Path(tmp)
        self.registry, self.table_t, self.form_t = _make_registry()
        self.policy_split = _make_policy_split(self.registry)
        self.manifest = _make_manifest(self.registry, self.policy_split)
        self.contract = _make_contract(self.manifest, self.registry, self.policy_split)
        self.manifest = dict(self.manifest)
        self.manifest.pop("manifest_hash", None)
        self.manifest["protocol"] = dict(self.manifest["protocol"])
        self.manifest["protocol"]["dataset_contract"] = {
            "schema_version": CONTRACT_SCHEMA,
            "contract_hash": self.contract["contract_hash"],
            "governance_role": "canary_excluded",
        }
        self.manifest["manifest_hash"] = _canonical_hash(self.manifest)
        self.preflight = _make_preflight(self.manifest)
        self.gate = _make_gate_report(self.manifest)
        self.raw_root = self.root / "raw-root"
        _make_raw_root(self.manifest, self.registry, self.raw_root)

    def write_inputs(
        self, records: list[dict[str, Any]]
    ) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        paths["contract"] = self._write_json(self.root / "contract.json", self.contract)
        paths["manifest"] = self._write_json(self.root / "manifest.json", self.manifest)
        paths["preflight"] = self._write_json(self.root / "preflight.json", self.preflight)
        paths["gate"] = self._write_json(self.root / "gate.json", self.gate)
        paths["registry"] = self._write_json(self.root / "registry.json", self.registry.to_dict())
        paths["policy_split"] = self._write_json(self.root / "split.json", self.policy_split)
        paths["results"] = self.root / "results.jsonl"
        paths["results"].write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in records),
            encoding="utf-8",
        )
        paths["raw_root"] = self.raw_root
        return paths

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> Path:
        path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
        return path


class BuildAndVerifyTest(unittest.TestCase):
    def _build(self, fx: _Fixture, records: list[dict[str, Any]], output: Path) -> dict:
        paths = fx.write_inputs(records)
        return build_package(
            contract_path=paths["contract"],
            manifest_path=paths["manifest"],
            results_path=paths["results"],
            preflight_path=paths["preflight"],
            gate_path=paths["gate"],
            registry_path=paths["registry"],
            policy_split_path=paths["policy_split"],
            raw_root=paths["raw_root"],
            output=output,
        )

    def test_build_emits_all_files_and_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = _Fixture(tmp)
            records = _make_records(fx.manifest, fx.registry)
            output = Path(tmp) / "package"
            audit = self._build(fx, records, output)
            self.assertTrue(audit["passed"], audit["checks"])
            for rel in (
                "data/attempts.jsonl",
                "raw/inventory.jsonl",
                "analysis/inclusion-ledger.jsonl",
                "analysis/gate-report.json",
                "QUALITY_AUDIT.json",
                "DATASET_CARD.md",
                "MANIFEST.json",
            ):
                self.assertTrue((output / rel).is_file(), rel)

            verify_audit = verify_package(output)
            self.assertTrue(verify_audit["passed"], verify_audit["checks"])

    def test_one_row_per_cell_with_canary_governance_and_false_eligibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = _Fixture(tmp)
            records = _make_records(fx.manifest, fx.registry)
            output = Path(tmp) / "package"
            self._build(fx, records, output)
            rows = [
                json.loads(line)
                for line in (output / "data" / "attempts.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            expected_cells = len(fx.manifest["panels"]) * len(fx.manifest["policy_bundle_ids"])
            self.assertEqual(len(rows), expected_cells)
            self.assertEqual(len(rows), fx.manifest["gates"]["attempts"])
            for row in rows:
                self.assertEqual(row["schema_version"], ATTEMPT_SCHEMA)
                self.assertEqual(row["task_role"], "T_canary")
                self.assertEqual(row["governance_role"], "canary_excluded")
                self.assertEqual(row["split"], "canary_excluded")
                self.assertEqual(
                    row["eligibility"],
                    {"training": False, "calibration": False, "development": False, "final": False},
                )
                self.assertEqual(row["execution"]["status"], "completed")
                self.assertEqual(row["mechanism"]["specialist_receipt_valid"], True)
                self.assertEqual(row["outcome"]["success"], True)
                self.assertEqual(row["outcome"]["failure_code"], None)
                self.assertEqual(row["task"]["prompt"], "Synthetic fixture prompt.")
                self.assertEqual(row["execution"]["usage"]["total_tokens"], 150)
                self.assertIn(row["panel"]["execution_position"], (0, 1))
                self.assertIsNotNone(row["raw"])
                self.assertTrue(row["raw"]["path"].startswith("raw/"))
                self.assertFalse(row["raw"]["path"].startswith("/"))
                self.assertEqual(len(row["raw"]["sha256"]), 64)

    def test_partial_results_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = _Fixture(tmp)
            all_panels = [str(p["panel_id"]) for p in fx.manifest["panels"]]
            skip = (all_panels[0],)
            records = _make_records(fx.manifest, fx.registry, skip_panel_ids=skip)
            with self.assertRaisesRegex(ValueError, "every planned panel"):
                self._build(fx, records, Path(tmp) / "package")

    def test_v2_confirmation_gate_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = _Fixture(tmp)
            fx.gate["gate"] = "m3-semantic-capability-gate-v2"
            fx.gate["schema_version"] = "m3-semantic-capability-gate-v2"
            fx.gate["stage"] = "replication_screen"
            fx.gate["decision"] = "confirmation_pass"
            fx.gate["passed"] = True
            audit = self._build(
                fx, _make_records(fx.manifest, fx.registry), Path(tmp) / "package"
            )
            self.assertTrue(audit["passed"])

    def test_deterministic_rebuild_is_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = _Fixture(tmp)
            records = _make_records(fx.manifest, fx.registry)
            out_a = Path(tmp) / "package-a"
            out_b = Path(tmp) / "package-b"
            self._build(fx, records, out_a)
            self._build(fx, records, out_b)
            rel_files = []
            for base in (out_a, out_b):
                rel_files.append(
                    {
                        str(p.relative_to(base)): p.read_bytes()
                        for p in sorted(base.rglob("*"))
                        if p.is_file()
                    }
                )
            self.assertEqual(set(rel_files[0]), set(rel_files[1]))
            for rel in rel_files[0]:
                self.assertEqual(rel_files[0][rel], rel_files[1][rel], rel)

    def test_manifest_has_schedule_and_file_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = _Fixture(tmp)
            records = _make_records(fx.manifest, fx.registry)
            output = Path(tmp) / "package"
            self._build(fx, records, output)
            pkg = json.loads((output / "MANIFEST.json").read_text(encoding="utf-8"))
            self.assertEqual(pkg["schema_version"], PACKAGE_SCHEMA)
            self.assertEqual(pkg["contract_hash"], fx.contract["contract_hash"])
            self.assertEqual(pkg["task_role"], "T_canary")
            self.assertEqual(pkg["governance_role"], "canary_excluded")
            self.assertEqual(len(pkg["schedule"]), len(fx.manifest["panels"]))
            self.assertIn("data/attempts.jsonl", pkg["files"])
            self.assertNotIn("MANIFEST.json", pkg["files"])
            self.assertIn("safe_export_file_sha256", pkg["identities"])
            self.assertIn("builder_source_sha256", pkg["identities"])


class ValidationFailureTest(unittest.TestCase):
    def _paths(self, fx: _Fixture, records: list[dict[str, Any]]) -> dict[str, Path]:
        return fx.write_inputs(records)

    def test_contract_hash_tamper_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = _Fixture(tmp)
            records = _make_records(fx.manifest, fx.registry)
            fx.contract["planned"]["attempts"] = 999
            # hash no longer matches
            paths = self._paths(fx, records)
            with self.assertRaisesRegex(ValueError, "contract_hash mismatch"):
                build_package(
                    contract_path=paths["contract"],
                    manifest_path=paths["manifest"],
                    results_path=paths["results"],
                    preflight_path=paths["preflight"],
                    gate_path=paths["gate"],
                    registry_path=paths["registry"],
                    policy_split_path=paths["policy_split"],
                    raw_root=paths["raw_root"],
                    output=Path(tmp) / "package",
                )

    def test_invalid_gate_decision_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = _Fixture(tmp)
            records = _make_records(fx.manifest, fx.registry)
            fx.gate["decision"] = "invalid"
            fx.gate["passed"] = False
            paths = self._paths(fx, records)
            with self.assertRaisesRegex(ValueError, "decision"):
                build_package(
                    contract_path=paths["contract"],
                    manifest_path=paths["manifest"],
                    results_path=paths["results"],
                    preflight_path=paths["preflight"],
                    gate_path=paths["gate"],
                    registry_path=paths["registry"],
                    policy_split_path=paths["policy_split"],
                    raw_root=paths["raw_root"],
                    output=Path(tmp) / "package",
                )

    def test_unknown_panel_in_results_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = _Fixture(tmp)
            records = _make_records(fx.manifest, fx.registry)
            bad = dict(records[0])
            bad["panel_id"] = "unknown-panel"
            records.append(bad)
            paths = self._paths(fx, records)
            with self.assertRaisesRegex(ValueError, "unknown panel_id"):
                build_package(
                    contract_path=paths["contract"],
                    manifest_path=paths["manifest"],
                    results_path=paths["results"],
                    preflight_path=paths["preflight"],
                    gate_path=paths["gate"],
                    registry_path=paths["registry"],
                    policy_split_path=paths["policy_split"],
                    raw_root=paths["raw_root"],
                    output=Path(tmp) / "package",
                )

    def test_forbidden_gate_content_fails_privacy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = _Fixture(tmp)
            records = _make_records(fx.manifest, fx.registry)
            fx.gate["system_prompt"] = "SECRET_PROMPT_TEXT"
            paths = self._paths(fx, records)
            with self.assertRaisesRegex(ValueError, "privacy"):
                build_package(
                    contract_path=paths["contract"],
                    manifest_path=paths["manifest"],
                    results_path=paths["results"],
                    preflight_path=paths["preflight"],
                    gate_path=paths["gate"],
                    registry_path=paths["registry"],
                    policy_split_path=paths["policy_split"],
                    raw_root=paths["raw_root"],
                    output=Path(tmp) / "package",
                )

    def test_absolute_path_in_gate_fails_privacy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = _Fixture(tmp)
            records = _make_records(fx.manifest, fx.registry)
            fx.gate["checks"] = {"artifact_dir": "/home/user/secrets"}
            paths = self._paths(fx, records)
            with self.assertRaisesRegex(ValueError, "privacy"):
                build_package(
                    contract_path=paths["contract"],
                    manifest_path=paths["manifest"],
                    results_path=paths["results"],
                    preflight_path=paths["preflight"],
                    gate_path=paths["gate"],
                    registry_path=paths["registry"],
                    policy_split_path=paths["policy_split"],
                    raw_root=paths["raw_root"],
                    output=Path(tmp) / "package",
                )


class PrivacyScanTest(unittest.TestCase):
    def test_flags_forbidden_keys_and_absolute_paths(self) -> None:
        violations = privacy_scan(
            {"ok": "fine", "system_prompt": "x", "nested": {"stderr": "y"}}
        )
        self.assertTrue(any("system_prompt" in v for v in violations))
        self.assertTrue(any("stderr" in v for v in violations))

    def test_flags_absolute_path_values(self) -> None:
        violations = privacy_scan({"a": "/home/user/x", "b": "relative/path", "c": "C:\\tmp\\x"})
        self.assertTrue(any("absolute path" in v for v in violations))
        # The relative path itself is not flagged.
        self.assertFalse(any("relative/path" in v for v in violations))

    def test_clean_payload_has_no_violations(self) -> None:
        self.assertEqual(privacy_scan({"path": "raw/a.json", "bytes": 1, "sha256": "a" * 64}), [])


class VerifyTamperTest(unittest.TestCase):
    def test_verify_fails_on_modified_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = _Fixture(tmp)
            records = _make_records(fx.manifest, fx.registry)
            paths = fx.write_inputs(records)
            output = Path(tmp) / "package"
            build_package(
                contract_path=paths["contract"],
                manifest_path=paths["manifest"],
                results_path=paths["results"],
                preflight_path=paths["preflight"],
                gate_path=paths["gate"],
                registry_path=paths["registry"],
                policy_split_path=paths["policy_split"],
                raw_root=paths["raw_root"],
                output=output,
            )
            self.assertTrue(verify_package(output)["passed"])
            # Tamper a derived file.
            attempts = output / "data" / "attempts.jsonl"
            attempts.write_text(attempts.read_text(encoding="utf-8").replace('"split":"canary_excluded"', '"split":"train"', 1), encoding="utf-8")
            audit = verify_package(output)
            self.assertFalse(audit["passed"])
            names = {c["name"] for c in audit["checks"] if not c["passed"]}
            self.assertIn("file_hashes", names)


class ContractLoaderTest(unittest.TestCase):
    def test_load_contract_verifies_embedded_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = _Fixture(tmp)
            path = Path(tmp) / "contract.json"
            path.write_text(json.dumps(fx.contract, sort_keys=True), encoding="utf-8")
            self.assertEqual(load_contract(path)["contract_hash"], fx.contract["contract_hash"])


if __name__ == "__main__":
    unittest.main()
