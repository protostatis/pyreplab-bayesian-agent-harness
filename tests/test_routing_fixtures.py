"""Deterministic utility-routing fixture family tests.

Covers determinism, the 32-coordinate Stage-A balance, sealed design labels,
presence of both table and form cues, text-variant structural invariance,
distinct seeds/nonces, and the anti-leakage model-visible contract.  Also runs
an end-to-end ``m3_routing_probe_gate`` Stage-A integration to prove the
backend satisfies the gate's expectations.
"""

from __future__ import annotations

import hashlib
import json
import unittest

from pyreplab_harness import m3_routing_probe_gate as gate
from pyreplab_harness import routing_fixtures as rf
from pyreplab_harness.m3_routing_probe_gate import (
    PROBE_FEATURE_NAMES,
)
from pyreplab_harness.structural_probe import structural_probe as _structural_probe


# ---------------------------------------------------------------------------
# structural probe adapter (production-compatible)
# ---------------------------------------------------------------------------


class _ProbeAdapter:
    def structural_probe(self, html: str) -> dict:
        return _structural_probe(html)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _design() -> list[dict]:
    return rf.build_stage_a_design()


def _stage_b_design() -> list[dict]:
    return rf.build_stage_b_design()


class DesignBalanceTest(unittest.TestCase):
    def test_stage_a_has_exactly_32_coordinates(self) -> None:
        self.assertEqual(len(_design()), 32)

    def test_eight_per_stratum(self) -> None:
        by_stratum: dict[str, int] = {}
        for coord in _design():
            by_stratum[coord["stratum"]] = by_stratum.get(coord["stratum"], 0) + 1
        for stratum in rf.STRATA:
            self.assertEqual(by_stratum[stratum], 8, stratum)

    def test_capability_balance(self) -> None:
        # pure strata are single-capability; mixed/ambiguous are 4/4 balanced.
        for stratum in rf.STRATA:
            caps = [
                c["preferred_capability"]
                for c in _design() if c["stratum"] == stratum
            ]
            if stratum == "pure_table":
                self.assertEqual(caps.count("table"), 8, stratum)
                self.assertEqual(caps.count("form"), 0, stratum)
            elif stratum == "pure_form":
                self.assertEqual(caps.count("table"), 0, stratum)
                self.assertEqual(caps.count("form"), 8, stratum)
            else:
                self.assertEqual(caps.count("table"), 4, stratum)
                self.assertEqual(caps.count("form"), 4, stratum)
        # overall balanced preferred capabilities 16/16
        all_caps = [c["preferred_capability"] for c in _design()]
        self.assertEqual(all_caps.count("table"), 16)
        self.assertEqual(all_caps.count("form"), 16)

    def test_complexity_balance(self) -> None:
        for stratum in rf.STRATA:
            diffs = [c["difficulty"] for c in _design() if c["stratum"] == stratum]
            self.assertEqual(diffs.count("easy"), 3, stratum)
            self.assertEqual(diffs.count("medium"), 3, stratum)
            self.assertEqual(diffs.count("hard"), 2, stratum)

    def test_coordinate_fields(self) -> None:
        for coord in _design():
            self.assertIsInstance(coord["fixture_id"], str)
            self.assertTrue(coord["fixture_id"])
            self.assertIn(coord["stratum"], rf.STRATA)
            self.assertIn(
                coord["first_bottleneck"], ("table_specialist", "form_specialist")
            )
            self.assertIn(coord["difficulty"], rf.DIFFICULTIES)
            self.assertIsInstance(coord["seed"], int)
            self.assertIsInstance(coord["operation_flags"], dict)
            self.assertIn("oracle", coord)
            self.assertIn("nonce", coord)

    def test_fixture_ids_unique_and_opaque(self) -> None:
        ids = [c["fixture_id"] for c in _design()]
        self.assertEqual(len(ids), len(set(ids)))
        for fixture_id in ids:
            self.assertFalse(fixture_id.startswith("/"))
            self.assertFalse(fixture_id.startswith("http"))
            self.assertNotEqual(len(fixture_id), 64)


class DeterminismTest(unittest.TestCase):
    def test_design_is_deterministic(self) -> None:
        self.assertEqual(_design(), _design())

    def test_fixture_html_is_deterministic(self) -> None:
        for coord in _design():
            a = rf.generate_routing_fixture(coord)
            b = rf.generate_routing_fixture(coord)
            self.assertEqual(a["html"], b["html"], coord["fixture_id"])
            self.assertEqual(a["source_sha256"], b["source_sha256"])

    def test_seed_changes_design(self) -> None:
        self.assertNotEqual(
            rf.build_stage_a_design(seed=1), rf.build_stage_a_design(seed=2)
        )

    def test_source_sha256_matches_html(self) -> None:
        for coord in _design():
            fixture = rf.generate_routing_fixture(coord)
            self.assertEqual(
                fixture["source_sha256"],
                hashlib.sha256(fixture["html"].encode("utf-8")).hexdigest(),
            )


class SealedLabelTest(unittest.TestCase):
    def test_sealed_label_deterministic(self) -> None:
        for coord in _design():
            self.assertEqual(rf.sealed_label(coord), rf.sealed_label(coord))

    def test_sealed_labels_distinct(self) -> None:
        labels = [rf.sealed_label(c) for c in _design()]
        self.assertEqual(len(labels), len(set(labels)))

    def test_sealed_label_ignores_text_variant(self) -> None:
        coord = _design()[0]
        self.assertEqual(rf.sealed_label(coord), rf.sealed_label(dict(coord)))


class StructurePresenceTest(unittest.TestCase):
    def test_every_initial_html_has_table_and_form(self) -> None:
        for coord in _design():
            html = rf.generate_routing_fixture(coord)["html"]
            self.assertIn("<table", html, coord["fixture_id"])
            self.assertIn("</table>", html, coord["fixture_id"])
            self.assertIn("<form", html, coord["fixture_id"])
            self.assertIn("</form>", html, coord["fixture_id"])


class TextVariantTest(unittest.TestCase):
    def test_variants_preserve_structural_topology(self) -> None:
        for coord in _design():
            primary = rf.generate_routing_fixture(coord, text_variant=0)["html"]
            alternate = rf.generate_routing_fixture(coord, text_variant=1)["html"]
            self.assertEqual(
                rf.structure_tokens(primary),
                rf.structure_tokens(alternate),
                coord["fixture_id"],
            )
            self.assertEqual(
                rf.structural_signature(primary),
                rf.structural_signature(alternate),
                coord["fixture_id"],
            )
            self.assertNotEqual(primary, alternate, coord["fixture_id"])

    def test_mutate_text_preserves_structure_changes_hash(self) -> None:
        for coord in _design():
            html = rf.generate_routing_fixture(coord)["html"]
            mutated = rf.mutate_text(html)
            self.assertNotEqual(html, mutated)
            self.assertEqual(rf.structure_tokens(html), rf.structure_tokens(mutated))

    def test_variants_share_answer(self) -> None:
        for coord in _design():
            primary = rf.generate_routing_fixture(coord, text_variant=0)["html"]
            alternate = rf.generate_routing_fixture(coord, text_variant=1)["html"]
            self.assertEqual(
                coord["nonce"] in primary,
                coord["nonce"] in alternate,
                coord["fixture_id"],
            )


class DistinctSeedNonceTest(unittest.TestCase):
    def test_nonces_distinct_across_design(self) -> None:
        nonces = [c["nonce"] for c in _design()]
        self.assertEqual(len(nonces), len(set(nonces)))

    def test_different_seeds_produce_different_nonce(self) -> None:
        a = rf.build_stage_a_design(seed=10)[0]["nonce"]
        b = rf.build_stage_a_design(seed=11)[0]["nonce"]
        self.assertNotEqual(a, b)

    def test_nonce_reachable_in_relevant_table(self) -> None:
        # For table-relevant strata the nonce must appear in the directory table,
        # except form-first mixed where it stays locked until the reference is
        # supplied back (and neither the nonce nor the reference may leak).
        for coord in _design():
            html = rf.generate_routing_fixture(coord)["html"]
            if coord["stratum"] == "mixed" and coord["dependency_order"] == "form_first":
                self.assertNotIn(coord["nonce"], html, coord["fixture_id"])
                self.assertNotIn(
                    coord["oracle"]["reference"], html, coord["fixture_id"]
                )
            elif coord["stratum"] in ("pure_table", "mixed"):
                self.assertIn(coord["nonce"], html, coord["fixture_id"])
            elif coord["stratum"] == "ambiguous":
                if coord["preferred_capability"] == "table":
                    self.assertIn(coord["nonce"], html, coord["fixture_id"])
                else:
                    self.assertNotIn(coord["nonce"], html, coord["fixture_id"])
            else:  # pure_form: nonce revealed only after submission
                self.assertNotIn(coord["nonce"], html, coord["fixture_id"])


class AntiLeakageTest(unittest.TestCase):
    def test_generated_fixture_excludes_secret_fields(self) -> None:
        forbidden = {
            "stratum", "difficulty", "seed", "template_id", "first_bottleneck",
            "nonce", "oracle", "preferred_capability", "dependency_order",
        }
        for coord in _design():
            fixture = rf.generate_routing_fixture(coord)
            self.assertEqual(
                set(fixture), {"fixture_id", "title", "prompt", "html", "source_sha256"}
            )
            self.assertTrue(forbidden.isdisjoint(fixture.keys()), coord["fixture_id"])
            # the nonce must not be a structured field of the fixture record
            self.assertNotIn("nonce", fixture)

    def test_model_visible_excludes_labels_and_seeds(self) -> None:
        for coord in _design():
            visible = rf.model_visible(coord)
            self.assertEqual(set(visible), {"fixture_id", "operation_flags"})
            self.assertNotIn("stratum", visible)
            self.assertNotIn("difficulty", visible)
            self.assertNotIn("seed", visible)
            self.assertNotIn("first_bottleneck", visible)
            self.assertNotIn("nonce", visible)
            self.assertNotIn("oracle", visible)

    def test_model_visible_passes_gate_privacy_scan(self) -> None:
        for coord in _design():
            violations = gate.privacy_scan(rf.model_visible(coord))
            self.assertEqual(violations, [], coord["fixture_id"])

    def test_oracle_is_json_serializable(self) -> None:
        for coord in _design():
            json.dumps(coord["oracle"])


class RenderStateTest(unittest.TestCase):
    def test_form_submission_reveals_nonce(self) -> None:
        for coord in _design():
            if coord["stratum"] not in ("pure_form", "mixed"):
                continue
            if coord["stratum"] == "ambiguous":
                continue
            oracle = coord["oracle"]
            values = dict(oracle["correct_form_values"])
            result = rf.render_state(coord, query_params=values)
            if coord["stratum"] == "mixed" and coord["dependency_order"] == "form_first":
                # form-first: success returns the record reference, not the nonce.
                self.assertIn(oracle["reference"], result, coord["fixture_id"])
                self.assertNotIn(coord["nonce"], result, coord["fixture_id"])
                self.assertIn("successful", result.lower())
            else:
                self.assertIn(coord["nonce"], result, coord["fixture_id"])
                self.assertIn("successful", result.lower())

    def test_form_submission_rejects_bad_values(self) -> None:
        for coord in _design():
            if coord["stratum"] != "pure_form":
                continue
            oracle = coord["oracle"]
            bad = {name: "INVALID" for name in oracle["correct_form_values"]}
            result = rf.render_state(coord, query_params=bad)
            self.assertNotIn(coord["nonce"], result)
            self.assertIn("error", result.lower())

    def test_table_query_keeps_nonce(self) -> None:
        for coord in _design():
            if coord["stratum"] not in ("pure_table", "mixed"):
                continue
            dept = coord["oracle"]["target_row_department"]
            result = rf.render_state(coord, query_params={"filter": dept})
            if coord["stratum"] == "mixed" and coord["dependency_order"] == "form_first":
                # form-first: a department filter alone must not unlock the nonce.
                self.assertNotIn(coord["nonce"], result, coord["fixture_id"])
            else:
                self.assertIn(coord["nonce"], result, coord["fixture_id"])


class GateIntegrationTest(unittest.TestCase):
    def test_stage_a_runs_to_probe_pass(self) -> None:
        manifest = gate.build_manifest(gate.default_spec())
        report = gate.run_stage_a(
            manifest,
            design_adapter=rf,
            probe_adapter=_ProbeAdapter(),
        )
        self.assertEqual(report["decision"], "probe_pass", report["reasons"])
        self.assertTrue(report["passed"])
        self.assertEqual(report["completeness"]["valid_receipts"], 96)
        self.assertEqual(report["privacy"]["violation_count"], 0)
        routing = report["routing"]
        self.assertGreaterEqual(routing["combined_agreement"], 28)
        for stratum in rf.STRATA:
            self.assertGreaterEqual(
                routing["combined_agreement_by_stratum"][stratum], 6, stratum
            )
        # all three baselines reported
        self.assertIsInstance(routing["prompt_only_agreement"], int)
        self.assertIsInstance(routing["probe_only_agreement"], int)
        self.assertTrue(report["checks"]["deterministic_features"])
        self.assertTrue(report["checks"]["text_invariance"])

    def test_design_passes_gate_coordinate_validation(self) -> None:
        manifest = gate.build_manifest(gate.default_spec())
        self.assertEqual(
            gate._validate_design_coordinates(manifest, rf.build_stage_a_design()),
            [],
        )

    def test_ambiguous_probe_only_baseline_is_misled(self) -> None:
        # Faithfulness check: for ambiguous fixtures the larger visible structure
        # is irrelevant, so the probe-only baseline should route to the opposite
        # specialist of the sealed bottleneck at least sometimes.
        manifest = gate.build_manifest(gate.default_spec())
        router = manifest["router"]
        misled = 0
        ambiguous = 0
        probe = _ProbeAdapter()
        for coord in rf.build_stage_a_design():
            if coord["stratum"] != "ambiguous":
                continue
            ambiguous += 1
            fixture = rf.generate_routing_fixture(coord)
            features = probe.structural_probe(fixture["html"])["features"]
            choice = gate.probe_only_heuristic(
                coord["operation_flags"], features, router
            )["choice"]
            if choice != coord["first_bottleneck"]:
                misled += 1
        self.assertEqual(ambiguous, 8)
        self.assertGreater(misled, 0)


class ValidationTest(unittest.TestCase):
    def test_bad_text_variant_raises(self) -> None:
        coord = _design()[0]
        with self.assertRaises(ValueError):
            rf.generate_routing_fixture(coord, text_variant=7)

    def test_bad_stratum_raises(self) -> None:
        coord = dict(_design()[0])
        coord["stratum"] = "nonsense"
        with self.assertRaises(ValueError):
            rf.generate_routing_fixture(coord)

    def test_bad_difficulty_raises(self) -> None:
        coord = dict(_design()[0])
        coord["difficulty"] = "impossible"
        with self.assertRaises(ValueError):
            rf.generate_routing_fixture(coord)


# ---------------------------------------------------------------------------
# Stage B: excluded crossed-outcome design
# ---------------------------------------------------------------------------


def _form_first_coords(design: list[dict]) -> list[dict]:
    return [
        c for c in design
        if c["stratum"] == "mixed" and c["dependency_order"] == "form_first"
    ]


class StageBDesignTest(unittest.TestCase):
    def test_stage_b_has_exactly_24_coordinates_in_2_blocks(self) -> None:
        design = _stage_b_design()
        self.assertEqual(len(design), 24)
        blocks = {c["block"] for c in design}
        self.assertEqual(blocks, {0, 1})
        self.assertEqual(len([c for c in design if c["block"] == 0]), 12)
        self.assertEqual(len([c for c in design if c["block"] == 1]), 12)

    def test_three_per_stratum_per_block(self) -> None:
        design = _stage_b_design()
        for block in (0, 1):
            for stratum in rf.STRATA:
                self.assertEqual(
                    sum(
                        1 for c in design
                        if c["block"] == block and c["stratum"] == stratum
                    ),
                    3,
                    (block, stratum),
                )

    def test_one_easy_medium_hard_per_block_stratum(self) -> None:
        design = _stage_b_design()
        for block in (0, 1):
            for stratum in rf.STRATA:
                diffs = [
                    c["difficulty"] for c in design
                    if c["block"] == block and c["stratum"] == stratum
                ]
                self.assertEqual(sorted(diffs), ["easy", "hard", "medium"], (block, stratum))

    def test_block_capability_balance_6_6(self) -> None:
        design = _stage_b_design()
        for block in (0, 1):
            caps = [c["preferred_capability"] for c in design if c["block"] == block]
            self.assertEqual(caps.count("table"), 6, block)
            self.assertEqual(caps.count("form"), 6, block)

    def test_mixed_ambiguous_splits_reversed_across_blocks(self) -> None:
        design = _stage_b_design()
        for stratum in ("mixed", "ambiguous"):
            per_block = []
            for block in (0, 1):
                caps = [
                    c["preferred_capability"] for c in design
                    if c["block"] == block and c["stratum"] == stratum
                ]
                self.assertEqual(len(caps), 3, (stratum, block))
                per_block.append((caps.count("table"), caps.count("form")))
            # 2/1 in one block, 1/2 in the other (reversed).
            self.assertEqual(per_block[0], (per_block[1][1], per_block[1][0]), stratum)
            self.assertIn(per_block[0], [(2, 1), (1, 2)], stratum)

    def test_stage_b_ids_seeds_nonces_unique_no_stage_a_overlap(self) -> None:
        stage_a = _design()
        stage_b = _stage_b_design()
        a_ids = {c["fixture_id"] for c in stage_a}
        a_nonces = {c["nonce"] for c in stage_a}
        a_seeds = {c["seed"] for c in stage_a}
        b_ids = [c["fixture_id"] for c in stage_b]
        b_nonces = [c["nonce"] for c in stage_b]
        b_seeds = [c["seed"] for c in stage_b]
        self.assertEqual(len(b_ids), len(set(b_ids)))
        self.assertEqual(len(b_nonces), len(set(b_nonces)))
        self.assertEqual(len(b_seeds), len(set(b_seeds)))
        self.assertTrue(a_ids.isdisjoint(b_ids))
        self.assertTrue(a_nonces.isdisjoint(b_nonces))
        self.assertTrue(a_seeds.isdisjoint(b_seeds))

    def test_stage_b_coordinates_carry_version_block_and_state(self) -> None:
        for coord in _stage_b_design():
            self.assertEqual(coord["generator_version"], rf.STAGE_B_GENERATOR_VERSION)
            self.assertIn(coord["block"], (0, 1))
            self.assertTrue(coord["template_id"].endswith("_v2"), coord["template_id"])
            self.assertIn("state", coord)
            self.assertIn("transitions", coord["state"])
            self.assertIn("initial", coord["state"])
            self.assertIn("oracle", coord)

    def test_stage_a_unchanged_generator_version(self) -> None:
        # Stage A must keep the frozen v1 contract (no generator_version key).
        self.assertEqual(rf.GENERATOR_VERSION, "routing-fixtures-v1")
        for coord in _design():
            self.assertNotIn("generator_version", coord)
            self.assertTrue(coord["template_id"].endswith("_v1"), coord["template_id"])

    def test_stage_b_deterministic(self) -> None:
        self.assertEqual(_stage_b_design(), _stage_b_design())

    def test_stage_b_seed_changes_design(self) -> None:
        self.assertNotEqual(
            rf.build_stage_b_design(seed=1), rf.build_stage_b_design(seed=2)
        )

    def test_stage_b_block_seeds_validation(self) -> None:
        with self.assertRaises(ValueError):
            rf.build_stage_b_design(block_seeds=(1,))
        with self.assertRaises(ValueError):
            rf.build_stage_b_design(block_seeds=(1, 2, 3))
        with self.assertRaises(ValueError):
            rf.build_stage_b_design(block_seeds=(1, "two"))

    def test_stage_b_fixtures_and_oracle_are_serializable(self) -> None:
        for coord in _stage_b_design():
            json.dumps(coord["oracle"])
            json.dumps(coord["state"])
            fixture = rf.generate_routing_fixture(coord)
            self.assertEqual(
                set(fixture), {"fixture_id", "title", "prompt", "html", "source_sha256"}
            )
            # the private reference and nonce must never leak into the public
            # fixture record or the model-visible extraction.
            self.assertNotIn("nonce", fixture)
            self.assertNotIn("reference", fixture)
            visible = rf.model_visible(coord)
            self.assertEqual(set(visible), {"fixture_id", "operation_flags"})

    def test_stage_b_fixture_html_has_table_and_form(self) -> None:
        for coord in _stage_b_design():
            html = rf.generate_routing_fixture(coord)["html"]
            self.assertIn("<table", html, coord["fixture_id"])
            self.assertIn("<form", html, coord["fixture_id"])


class StageBTransitionTest(unittest.TestCase):
    def _form_first(self) -> list[dict]:
        return _form_first_coords(_stage_b_design())

    def test_form_first_initial_form_has_no_access_code(self) -> None:
        for coord in self._form_first():
            html = rf.generate_routing_fixture(coord)["html"]
            self.assertNotIn('name="access_code"', html, coord["fixture_id"])
            self.assertNotIn("Access Code</label>", html, coord["fixture_id"])

    def test_form_first_initial_html_locks_nonce_and_reference(self) -> None:
        for coord in self._form_first():
            html = rf.generate_routing_fixture(coord)["html"]
            self.assertNotIn(coord["nonce"], html, coord["fixture_id"])
            self.assertNotIn(coord["oracle"]["reference"], html, coord["fixture_id"])
            self.assertIn("LOCKED", html, coord["fixture_id"])
            self.assertIn("PENDING", html, coord["fixture_id"])

    def test_form_first_valid_form_returns_reference(self) -> None:
        for coord in self._form_first():
            oracle = coord["oracle"]
            values = dict(oracle["correct_form_values"])
            self.assertNotIn("access_code", values)
            result = rf.render_state(coord, query_params=values)
            self.assertIn(oracle["reference"], result, coord["fixture_id"])
            self.assertNotIn(coord["nonce"], result, coord["fixture_id"])
            self.assertIn("successful", result.lower())
            self.assertIn(f'name="{rf.RECORD_REFERENCE_PARAM}"', result)
            self.assertIn("Retrieve Directory Record", result)

    def test_form_first_reference_unlocks_final_nonce(self) -> None:
        for coord in self._form_first():
            result = rf.render_state(
                coord,
                query_params={rf.RECORD_REFERENCE_PARAM: coord["oracle"]["reference"]},
            )
            self.assertIn(coord["nonce"], result, coord["fixture_id"])

    def test_form_first_cannot_finish_table_before_valid_reference(self) -> None:
        for coord in self._form_first():
            initial = rf.generate_routing_fixture(coord)["html"]
            self.assertNotIn(coord["nonce"], initial, coord["fixture_id"])
            # wrong reference -> still locked
            wrong = rf.render_state(
                coord, query_params={rf.RECORD_REFERENCE_PARAM: "REF-WRONG"}
            )
            self.assertNotIn(coord["nonce"], wrong, coord["fixture_id"])
            # department filter alone -> still locked
            filtered = rf.render_state(
                coord, query_params={"filter": coord["oracle"]["target_row_department"]}
            )
            self.assertNotIn(coord["nonce"], filtered, coord["fixture_id"])

    def test_form_first_reference_is_deterministic(self) -> None:
        for coord in self._form_first():
            self.assertEqual(
                coord["oracle"]["reference"], coord["oracle"]["reference"]
            )
            self.assertTrue(
                coord["oracle"]["reference"].startswith("REF-"),
                coord["oracle"]["reference"],
            )

    def test_table_first_remains_table_then_form(self) -> None:
        for coord in _stage_b_design():
            if coord["stratum"] != "mixed" or coord["dependency_order"] != "table_first":
                continue
            html = rf.generate_routing_fixture(coord)["html"]
            # nonce visible in the table and the form requires the access code.
            self.assertIn(coord["nonce"], html, coord["fixture_id"])
            self.assertIn('name="access_code"', html, coord["fixture_id"])
            oracle = coord["oracle"]
            values = dict(oracle["correct_form_values"])
            self.assertIn("access_code", values)
            result = rf.render_state(coord, query_params=values)
            self.assertIn(coord["nonce"], result, coord["fixture_id"])

    def test_all_forms_use_get(self) -> None:
        for coord in _stage_b_design():
            html = rf.generate_routing_fixture(coord)["html"]
            if "<form" not in html:
                continue
            # every rendered form is method="get" (semantic_browser GET-only).
            for form in html.split("<form")[1:]:
                method = form.split(">", 1)[0]
                self.assertIn('method="get"', method, coord["fixture_id"])
                self.assertNotIn('method="post"', method, coord["fixture_id"])

    def test_form_first_state_metadata_describes_unlock(self) -> None:
        for coord in self._form_first():
            transitions = coord["state"]["transitions"]
            triggers = {t["trigger"] for t in transitions}
            self.assertIn("form_submit", triggers)
            self.assertIn("record_reference", triggers)
            self.assertTrue(coord["state"]["initial"]["table_locked"])
            record = next(
                t for t in transitions if t["trigger"] == "record_reference"
            )
            self.assertTrue(record["requires_reference_match"])
            self.assertEqual(record["reveals"], "nonce")


class StageBPromptTest(unittest.TestCase):
    def test_prompts_truthfully_describe_dependency(self) -> None:
        for coord in _stage_b_design():
            prompt = rf.generate_routing_fixture(coord)["prompt"].casefold()
            if coord["stratum"] == "mixed":
                if coord["dependency_order"] == "form_first":
                    self.assertIn("reference", prompt, coord["fixture_id"])
                    self.assertNotIn("access code for", prompt, coord["fixture_id"])
                else:
                    self.assertIn("access code", prompt, coord["fixture_id"])
                    self.assertIn("verification form", prompt, coord["fixture_id"])


if __name__ == "__main__":
    unittest.main()
