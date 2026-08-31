from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from pyreplab_harness import m3_routing_probe_gate as gate
from pyreplab_harness.m3_routing_probe_gate import (
    MANIFEST_SCHEMA,
    PROBE_FEATURE_NAMES,
    canonical_hash,
    SPECIALISTS,
    STRATA,
    analyze_stage_a,
    build_manifest,
    default_spec,
    frozen_heuristic,
    main,
    privacy_scan,
    probe_only_heuristic,
    prompt_only_heuristic,
    run_stage_a,
    run_synthetic_utility_smoke,
    select_policy,
    utility,
    validate_gate_report,
    validate_manifest,
)
from pyreplab_harness.structural_probe import structural_probe as _structural_probe


def _prompt_from_operation_flags(operation_flags: dict) -> str:
    """Generate request text that drives ``derive_request_features`` in gates."""
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
                "Complete and submit the verification form first, then use the returned "
                "code to locate the confirmation key in the directory table."
            )
        return (
            "Locate the access code in the directory table, then complete and "
            "submit the verification form."
        )
    return "Review the task and carry out the required step."


class MockStructuralProbe:
    def structural_probe(self, html: str) -> dict:
        return _structural_probe(html)


# ---------------------------------------------------------------------------
# mock routing design (32 fixtures, 8 per stratum)
# ---------------------------------------------------------------------------


def _table_html(rows: int, cols: int, label: str = "t") -> str:
    rows_html = ""
    for r in range(rows):
        cells = "".join(f"<td>{label}{r}{c}</td>" for c in range(cols))
        rows_html += f"<tr>{cells}</tr>"
    return f"<table><tbody>{rows_html}</tbody></table>"


def _form_html(n_inputs: int, method: str = "post", *, required: int = 0) -> str:
    parts = []
    for i in range(n_inputs):
        req = " required" if i < required else ""
        parts.append(f'<input type="text" name="f{i}"{req}>')
    return f'<form method="{method}">{"".join(parts)}</form>'


def _page(*blocks: str) -> str:
    return "<html><body>" + "".join(blocks) + "</body></html>"


def build_fixture_html(coord: dict) -> str:
    stratum = coord["stratum"]
    if stratum == "pure_table":
        return _page(_table_html(5, 4), _form_html(1))
    if stratum == "pure_form":
        return _page(_form_html(4), _table_html(1, 2))
    if stratum == "mixed":
        return _page(_table_html(4, 3), _form_html(3))
    # ambiguous: the larger visible structure is opposite the bottleneck.
    if coord["first_bottleneck"] == "table_specialist":
        return _page(_form_html(6), _table_html(1, 2))
    return _page(_table_html(6, 4), _form_html(1, method="get"))


class MockRoutingDesign:
    """Deterministic 32-fixture design.

    ``flip_ambiguous``: set of global fixture indices (0..31) whose ambiguous
    declared operation is inverted relative to the sealed label, producing a
    per-stratum router disagreement.  ``extra_flags``: mapping merged into every
    fixture's public operation_flags (used to inject privacy leaks).
    """

    GENERATOR_VERSION = "mock-routing-design-v1"

    def __init__(self, *, flip_ambiguous: set | None = None, extra_flags: dict | None = None):
        self._flip = flip_ambiguous or set()
        self._extra = extra_flags or {}
        self._coords = self._build()
        self.last_seed: int | None = None

    def _build(self) -> list[dict]:
        coords: list[dict] = []
        index = 0
        for stratum in STRATA:
            for i in range(8):
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
                for key, value in self._extra.items():
                    coord["operation_flags"][key] = value
                coords.append(coord)
                index += 1
        return coords

    def build_stage_a_design(self, seed: int | None = None) -> list[dict]:
        self.last_seed = seed
        return [dict(coord) for coord in self._coords]

    def generate_routing_fixture(self, coord: dict) -> dict:
        html = build_fixture_html(coord)
        return {
            "fixture_id": coord["fixture_id"],
            "title": f"Fixture {coord['fixture_id']}",
            "prompt": _prompt_from_operation_flags(coord["operation_flags"]),
            "html": html,
            "source_sha256": hashlib.sha256(html.encode("utf-8")).hexdigest(),
        }


class MockUtilityValidator:
    def __call__(self, smoke_report: dict) -> dict:
        return {"passed": bool(smoke_report.get("passed")), "backend": "mock"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _freeze_spec(*, seed: int = 1, design_adapter=None) -> dict:
    return build_manifest(default_spec(seed=seed), design_adapter=design_adapter)


def _dumps(report: dict) -> str:
    return json.dumps(report, sort_keys=True, ensure_ascii=False)


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


class ManifestTest(unittest.TestCase):
    def test_build_and_validate(self) -> None:
        manifest = build_manifest(default_spec())
        self.assertEqual(manifest["schema_version"], MANIFEST_SCHEMA)
        self.assertEqual(validate_manifest(manifest), [])
        self.assertEqual(len(manifest["manifest_hash"]), 64)
        # self-hash is over everything except the hash field itself
        recomputed = canonical_hash(
            {k: v for k, v in manifest.items() if k != "manifest_hash"}
        )
        self.assertEqual(manifest["manifest_hash"], recomputed)

    def test_deterministic_build(self) -> None:
        a = _dumps(build_manifest(default_spec(seed=42)))
        b = _dumps(build_manifest(default_spec(seed=42)))
        self.assertEqual(a, b)
        c = _dumps(build_manifest(default_spec(seed=43)))
        self.assertNotEqual(a, c)

    def test_hash_tamper_detected(self) -> None:
        manifest = build_manifest(default_spec())
        manifest["gates"]["router_agreement_min"] = 20
        errors = validate_manifest(manifest)
        self.assertTrue(any("router_agreement_min" in e for e in errors))

    def test_hash_field_tamper_detected(self) -> None:
        manifest = build_manifest(default_spec())
        manifest["manifest_hash"] = "0" * 64
        errors = validate_manifest(manifest)
        self.assertTrue(any("manifest_hash mismatch" in e for e in errors))

    def test_invalid_manifest_rejected_by_run(self) -> None:
        manifest = build_manifest(default_spec())
        manifest["gates"]["privacy_violations_max"] = 5
        with self.assertRaises(ValueError):
            run_stage_a(
                manifest,
                design_adapter=MockRoutingDesign(),
                probe_adapter=MockStructuralProbe(),
            )

    def test_build_manifest_propagates_seed(self) -> None:
        design = MockRoutingDesign()
        build_manifest(default_spec(seed=77), design_adapter=design)
        self.assertEqual(design.last_seed, 77)

    def test_manifest_binds_design_commitments(self) -> None:
        manifest = build_manifest(default_spec(seed=1), design_adapter=MockRoutingDesign())
        design = manifest["design"]
        self.assertEqual(design["generator_version"], MockRoutingDesign.GENERATOR_VERSION)
        self.assertEqual(len(design["fixtures"]), 32)
        ids = {entry["fixture_id"] for entry in design["fixtures"]}
        self.assertEqual(len(ids), 32)
        for entry in design["fixtures"]:
            self.assertEqual(len(entry["private_seal"]), 64)
            self.assertEqual(len(entry["source_sha256"]), 64)
            self.assertEqual(len(entry["prompt_sha256"]), 64)

    def test_manifest_design_tamper_detected(self) -> None:
        manifest = build_manifest(default_spec(seed=1), design_adapter=MockRoutingDesign())
        manifest["design"]["fixtures"][0]["private_seal"] = "not-a-hex-digest"
        errors = validate_manifest(manifest)
        self.assertTrue(any("private_seal" in e for e in errors))
        # tampering a bound entry to a different valid digest breaks the self-hash
        manifest2 = build_manifest(default_spec(seed=1), design_adapter=MockRoutingDesign())
        manifest2["design"]["fixtures"][0]["source_sha256"] = "0" * 64
        self.assertTrue(any("manifest_hash mismatch" in e for e in validate_manifest(manifest2)))


# ---------------------------------------------------------------------------
# heuristic and utility
# ---------------------------------------------------------------------------


class HeuristicTest(unittest.TestCase):
    def _router(self) -> dict:
        return build_manifest(default_spec())["router"]

    def test_combined_heuristic_transparent(self) -> None:
        router = self._router()
        flags = {"table_operation": True, "form_operation": False, "first_operation": None}
        features = {name: 0 for name in PROBE_FEATURE_NAMES}
        features["table_count"] = 3
        breakdown = frozen_heuristic(flags, features, router)
        self.assertEqual(breakdown["choice"], "table_specialist")
        for key in ("score_table", "score_form", "prompt_table", "probe_table"):
            self.assertIn(key, breakdown)

    def test_heuristic_deterministic(self) -> None:
        router = self._router()
        flags = {"table_operation": True, "form_operation": True, "first_operation": "form"}
        features = {name: 0 for name in PROBE_FEATURE_NAMES}
        a = frozen_heuristic(flags, features, router)
        b = frozen_heuristic(flags, features, router)
        self.assertEqual(a, b)
        self.assertEqual(a["choice"], "form_specialist")

    def test_ambiguous_prompt_overrides_structure(self) -> None:
        router = self._router()
        flags = {"table_operation": True, "form_operation": False, "first_operation": None}
        features = {name: 0 for name in PROBE_FEATURE_NAMES}
        features["form_count"] = 10
        combined = frozen_heuristic(flags, features, router)["choice"]
        prompt_only = prompt_only_heuristic(flags, features, router)["choice"]
        probe_only = probe_only_heuristic(flags, features, router)["choice"]
        self.assertEqual(combined, "table_specialist")
        self.assertEqual(prompt_only, "table_specialist")
        self.assertEqual(probe_only, "form_specialist")

    def test_utility_and_tie_break(self) -> None:
        self.assertAlmostEqual(utility(0.8, 5000, 1.0), 0.8 - 0.5)
        # dominance
        self.assertEqual(
            select_policy(
                ["A", "B"],
                {"A": {"success": 0.8, "cost_tokens": 5000},
                 "B": {"success": 0.5, "cost_tokens": 9000}},
                1.0,
            ),
            "A",
        )
        # equal utility -> higher success, then lower cost, then registry order
        self.assertEqual(
            select_policy(
                ["A", "B"],
                {"A": {"success": 0.7, "cost_tokens": 10000},
                 "B": {"success": 0.7, "cost_tokens": 10000}},
                1.0,
            ),
            "A",
        )
        self.assertEqual(
            select_policy(
                ["A", "B"],
                {"A": {"success": 0.7, "cost_tokens": 20000},
                 "B": {"success": 0.7, "cost_tokens": 10000}},
                1.0,
            ),
            "B",
        )

    def test_select_policy_rejects_bad_predictions(self) -> None:
        for bad in (
            {"A": {"success": None, "cost_tokens": 1}, "B": {"success": 0.5, "cost_tokens": 1}},
            {"A": {"success": 0.5, "cost_tokens": None}, "B": {"success": 0.5, "cost_tokens": 1}},
            {"A": {"success": float("nan"), "cost_tokens": 1}, "B": {"success": 0.5, "cost_tokens": 1}},
            {"A": {"success": 0.5, "cost_tokens": float("inf")}, "B": {"success": 0.5, "cost_tokens": 1}},
            {"A": {"success": 0.5, "cost_tokens": -1}, "B": {"success": 0.5, "cost_tokens": 1}},
        ):
            with self.assertRaises(ValueError):
                select_policy(["A", "B"], bad, 1.0)

    def test_synthetic_utility_smoke_passes(self) -> None:
        smoke = run_synthetic_utility_smoke()
        self.assertTrue(smoke["passed"], smoke["reasons"])
        self.assertIn("dominance", smoke["cases"])
        self.assertIn("success_cost_tradeoff", smoke["cases"])
        self.assertIn("missing_nonfinite_detected", smoke["cases"])


# ---------------------------------------------------------------------------
# privacy audit
# ---------------------------------------------------------------------------


class PrivacyTest(unittest.TestCase):
    def test_privacy_scan_flags_forbidden_key(self) -> None:
        violations = privacy_scan({"operation_flags": {"label": "table_specialist"}})
        self.assertTrue(any("forbidden key 'label'" in v for v in violations))

    def test_privacy_scan_flags_hex_digest_and_url(self) -> None:
        violations = privacy_scan(
            {"a": "0" * 64, "b": "http://example.com/x", "c": "/etc/passwd"}
        )
        self.assertEqual(len(violations), 3)


# ---------------------------------------------------------------------------
# Stage A end-to-end
# ---------------------------------------------------------------------------


class StageATest(unittest.TestCase):
    def test_stage_a_pass(self) -> None:
        manifest = _freeze_spec(design_adapter=MockRoutingDesign())
        report = run_stage_a(
            manifest,
            design_adapter=MockRoutingDesign(),
            probe_adapter=MockStructuralProbe(),
            utility_validator=MockUtilityValidator(),
        )
        self.assertEqual(report["decision"], "probe_pass", report["reasons"])
        self.assertEqual(report["exit_code"], 0)
        self.assertTrue(report["passed"])
        self.assertEqual(report["completeness"]["valid_receipts"], 96)
        routing = report["routing"]
        self.assertEqual(routing["combined_agreement"], 32)
        for stratum in STRATA:
            self.assertEqual(routing["combined_agreement_by_stratum"][stratum], 8)
        # all three baselines reported
        self.assertIsInstance(routing["prompt_only_agreement"], int)
        self.assertIsInstance(routing["probe_only_agreement"], int)
        self.assertEqual(report["privacy"]["violation_count"], 0)

    def test_stage_a_per_stratum_fail(self) -> None:
        manifest = _freeze_spec(
            design_adapter=MockRoutingDesign(flip_ambiguous={24, 25, 26})
        )
        # flip three ambiguous fixtures: overall 29/32 but ambiguous 5/8.
        flip = {24, 25, 26}
        report = run_stage_a(
            manifest,
            design_adapter=MockRoutingDesign(flip_ambiguous=flip),
            probe_adapter=MockStructuralProbe(),
        )
        self.assertEqual(report["decision"], "probe_no_go", report["reasons"])
        self.assertEqual(report["exit_code"], 2)
        routing = report["routing"]
        self.assertEqual(routing["combined_agreement"], 29)
        self.assertEqual(routing["combined_agreement_by_stratum"]["ambiguous"], 5)
        self.assertFalse(report["checks"]["router_agreement"])
        self.assertTrue(any("router agreement" in r for r in report["reasons"]))

    def test_stage_a_privacy_fail(self) -> None:
        manifest = _freeze_spec(design_adapter=MockRoutingDesign())
        original_derive = gate.derive_request_features

        def leaking_request_features(prompt: str) -> dict:
            features = dict(original_derive(prompt))
            features["label"] = "table_specialist"
            return features

        with patch.object(gate, "derive_request_features", side_effect=leaking_request_features):
            report = run_stage_a(
                manifest,
                design_adapter=MockRoutingDesign(),
                probe_adapter=MockStructuralProbe(),
            )
        self.assertEqual(report["decision"], "probe_no_go", report["reasons"])
        self.assertEqual(report["exit_code"], 2)
        self.assertFalse(report["checks"]["privacy_zero"])
        self.assertGreater(report["privacy"]["violation_count"], 0)

    def test_stage_a_invalid_design_raises(self) -> None:
        manifest = _freeze_spec(design_adapter=MockRoutingDesign())
        design = MockRoutingDesign()
        coords = design.build_stage_a_design()[:-1]  # only 31 -> mechanical invalid
        broken = type(
            "BrokenDesign",
            (),
            {
                "GENERATOR_VERSION": "mock-routing-design-v1",
                "build_stage_a_design": staticmethod(lambda seed=None: coords),
                "generate_routing_fixture": staticmethod(design.generate_routing_fixture),
            },
        )()
        with self.assertRaises(ValueError):
            run_stage_a(manifest, design_adapter=broken, probe_adapter=MockStructuralProbe())

    def test_stage_a_invalid_receipt(self) -> None:
        manifest = _freeze_spec(design_adapter=MockRoutingDesign())
        design = MockRoutingDesign()
        probe = MockStructuralProbe()

        class TamperedProbe:
            def structural_probe(self, html: str) -> dict:
                out = probe.structural_probe(html)
                out["receipt"]["schema_version"] = "forged"
                return out

        report = run_stage_a(
            manifest,
            design_adapter=design,
            probe_adapter=TamperedProbe(),
        )
        self.assertEqual(report["decision"], "invalid", report["reasons"])
        self.assertEqual(report["exit_code"], 1)
        self.assertFalse(report["checks"]["receipts_valid"])

    def test_deterministic_report(self) -> None:
        manifest = _freeze_spec(design_adapter=MockRoutingDesign())
        first = run_stage_a(
            manifest,
            design_adapter=MockRoutingDesign(),
            probe_adapter=MockStructuralProbe(),
        )
        second = run_stage_a(
            manifest,
            design_adapter=MockRoutingDesign(),
            probe_adapter=MockStructuralProbe(),
        )
        self.assertEqual(_dumps(first), _dumps(second))

    def test_analyze_rejects_bad_manifest(self) -> None:
        manifest = build_manifest(default_spec())
        manifest["gates"]["router_agreement_min"] = 99
        with self.assertRaises(ValueError):
            analyze_stage_a(manifest, [])

    def test_seed_propagation_to_design(self) -> None:
        manifest = _freeze_spec(seed=12345, design_adapter=MockRoutingDesign())
        self.assertEqual(manifest["seed"], 12345)
        design = MockRoutingDesign()
        run_stage_a(
            manifest,
            design_adapter=design,
            probe_adapter=MockStructuralProbe(),
        )
        self.assertEqual(design.last_seed, 12345)

    def test_unauthorized_design_rejected(self) -> None:
        manifest = _freeze_spec(design_adapter=MockRoutingDesign())

        class RenamedDesign(MockRoutingDesign):
            def build_stage_a_design(self, seed: int | None = None) -> list[dict]:
                coords = super().build_stage_a_design(seed=seed)
                for coord in coords:
                    coord["fixture_id"] = "renamed-" + coord["fixture_id"]
                return coords

        with self.assertRaises(ValueError):
            run_stage_a(
                manifest,
                design_adapter=RenamedDesign(),
                probe_adapter=MockStructuralProbe(),
            )

    def test_unauthorized_prompt_change_rejected(self) -> None:
        manifest = _freeze_spec(design_adapter=MockRoutingDesign())

        class PromptDivergentDesign(MockRoutingDesign):
            def generate_routing_fixture(self, coord: dict) -> dict:
                out = super().generate_routing_fixture(coord)
                out["prompt"] = out["prompt"] + " "  # trailing space changes hash only
                return out

        with self.assertRaises(ValueError):
            run_stage_a(
                manifest,
                design_adapter=PromptDivergentDesign(),
                probe_adapter=MockStructuralProbe(),
            )

    def test_report_hash_embedded(self) -> None:
        manifest = _freeze_spec(design_adapter=MockRoutingDesign())
        report = run_stage_a(
            manifest,
            design_adapter=MockRoutingDesign(),
            probe_adapter=MockStructuralProbe(),
        )
        self.assertEqual(len(report["report_hash"]), 64)
        recomputed = canonical_hash(
            {k: v for k, v in report.items() if k != "report_hash"}
        )
        self.assertEqual(report["report_hash"], recomputed)

    def test_report_fixture_commitments(self) -> None:
        manifest = _freeze_spec(design_adapter=MockRoutingDesign())
        report = run_stage_a(
            manifest,
            design_adapter=MockRoutingDesign(),
            probe_adapter=MockStructuralProbe(),
        )
        commits = report["fixture_commitments"]
        self.assertEqual(len(commits), 32)
        allowed = {
            "fixture_id",
            "source_sha256",
            "canonical_feature_sha256",
            "combined_route",
            "prompt_only_route",
        }
        for entry in commits:
            self.assertEqual(set(entry), allowed)
            self.assertEqual(len(entry["source_sha256"]), 64)
            self.assertEqual(len(entry["canonical_feature_sha256"]), 64)
            self.assertIn(entry["combined_route"], SPECIALISTS)
            self.assertIn(entry["prompt_only_route"], SPECIALISTS)
        serialized = _dumps(report)
        for forbidden in ("<html", "oracle", "nonce"):
            self.assertNotIn(forbidden, serialized)

    def test_phase_order_receipt(self) -> None:
        manifest = _freeze_spec(design_adapter=MockRoutingDesign())
        report = run_stage_a(
            manifest,
            design_adapter=MockRoutingDesign(),
            probe_adapter=MockStructuralProbe(),
        )
        phases = report["phases"]
        self.assertEqual(
            phases["order"], ["probe_collection", "router_decision", "label_reveal"]
        )
        self.assertTrue(phases["probe_precedes_treatment"])
        self.assertTrue(phases["zero_outcome"])
        self.assertIsNone(phases["treatment_assignment"])
        self.assertTrue(report["checks"]["probe_precedes_treatment"])

    def test_validate_gate_report_accepts(self) -> None:
        manifest = _freeze_spec(design_adapter=MockRoutingDesign())
        report = run_stage_a(
            manifest,
            design_adapter=MockRoutingDesign(),
            probe_adapter=MockStructuralProbe(),
        )
        self.assertEqual(validate_gate_report(report, manifest), [])

    def test_validate_gate_report_rejects_tampered_hash(self) -> None:
        manifest = _freeze_spec(design_adapter=MockRoutingDesign())
        report = run_stage_a(
            manifest,
            design_adapter=MockRoutingDesign(),
            probe_adapter=MockStructuralProbe(),
        )
        report["report_hash"] = "0" * 64
        errors = validate_gate_report(report, manifest)
        self.assertTrue(any("report_hash" in e for e in errors))

    def test_validate_gate_report_rejects_wrong_manifest(self) -> None:
        manifest = _freeze_spec(design_adapter=MockRoutingDesign())
        report = run_stage_a(
            manifest,
            design_adapter=MockRoutingDesign(),
            probe_adapter=MockStructuralProbe(),
        )
        other = _freeze_spec(seed=2, design_adapter=MockRoutingDesign())
        errors = validate_gate_report(report, other)
        self.assertTrue(any("manifest_hash" in e for e in errors))

    def test_validate_gate_report_checks_pass_semantics(self) -> None:
        manifest = _freeze_spec(design_adapter=MockRoutingDesign())
        report = run_stage_a(
            manifest,
            design_adapter=MockRoutingDesign(),
            probe_adapter=MockStructuralProbe(),
        )
        report["decision"] = "probe_no_go"
        errors = validate_gate_report(report, manifest)
        self.assertTrue(any("probe_no_go" in e for e in errors))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class CliTest(unittest.TestCase):
    def _write(self, path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")

    def test_freeze_validates_and_writes_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec_path = root / "spec.json"
            manifest_path = root / "manifest.json"
            self._write(spec_path, default_spec(seed=7))
            with redirect_stdout(StringIO()):
                rc = main(["freeze", str(spec_path), str(manifest_path)])
            self.assertEqual(rc, 0)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(validate_manifest(manifest), [])

    def test_validate_exit_codes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            self._write(manifest_path, build_manifest(default_spec()))
            with redirect_stdout(StringIO()):
                self.assertEqual(main(["validate", str(manifest_path)]), 0)
            tampered = build_manifest(default_spec())
            tampered["gates"]["valid_receipts"] = 1
            self._write(manifest_path, tampered)
            with redirect_stdout(StringIO()):
                self.assertEqual(main(["validate", str(manifest_path)]), 1)

    def test_run_pass_exit_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            output = root / "gate.json"
            design = MockRoutingDesign()
            self._write(manifest_path, build_manifest(default_spec(), design_adapter=design))
            with patch.object(gate, "_lazy_design_backend", return_value=design), \
                 patch.object(gate, "_lazy_probe_backend", return_value=MockStructuralProbe()), \
                 redirect_stdout(StringIO()):
                rc = main(["run", str(manifest_path), "--output", str(output)])
            self.assertEqual(rc, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["decision"], "probe_pass")

    def test_run_no_go_exit_two(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            output = root / "gate.json"
            design = MockRoutingDesign(flip_ambiguous={24, 25, 26})
            self._write(manifest_path, build_manifest(default_spec(), design_adapter=design))
            with patch.object(gate, "_lazy_design_backend", return_value=design), \
                 patch.object(gate, "_lazy_probe_backend", return_value=MockStructuralProbe()), \
                 redirect_stdout(StringIO()):
                rc = main(["run", str(manifest_path), "--output", str(output)])
            self.assertEqual(rc, 2)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["decision"], "probe_no_go")

    def test_run_invalid_exit_one_with_invalid_backends(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            self._write(manifest_path, build_manifest(default_spec()))
            missing = root / "missing_backend.py"
            with redirect_stdout(StringIO()):
                rc = main([
                    "run",
                    str(manifest_path),
                    "--design-module",
                    str(missing),
                ])
            self.assertEqual(rc, 1)

    def test_freeze_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec_a = root / "spec_a.json"
            spec_b = root / "spec_b.json"
            manifest_path = root / "manifest.json"
            self._write(spec_a, default_spec(seed=7))
            self._write(spec_b, default_spec(seed=8))
            with redirect_stdout(StringIO()):
                self.assertEqual(main(["freeze", str(spec_a), str(manifest_path)]), 0)
            # byte-identical freeze is idempotent (no overwrite, still succeeds)
            with redirect_stdout(StringIO()):
                self.assertEqual(main(["freeze", str(spec_a), str(manifest_path)]), 0)
            # different content must refuse to overwrite
            with redirect_stdout(StringIO()):
                self.assertEqual(main(["freeze", str(spec_b), str(manifest_path)]), 1)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["seed"], 7)

    def test_run_refuses_overwrite_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            output = root / "gate.json"
            design = MockRoutingDesign()
            self._write(manifest_path, build_manifest(default_spec(), design_adapter=design))
            output.write_text("garbage", encoding="utf-8")
            with patch.object(gate, "_lazy_design_backend", return_value=design), \
                 patch.object(gate, "_lazy_probe_backend", return_value=MockStructuralProbe()), \
                 redirect_stdout(StringIO()):
                rc = main(["run", str(manifest_path), "--output", str(output)])
            self.assertEqual(rc, 1)
            self.assertEqual(output.read_text(encoding="utf-8"), "garbage")


if __name__ == "__main__":
    unittest.main()
