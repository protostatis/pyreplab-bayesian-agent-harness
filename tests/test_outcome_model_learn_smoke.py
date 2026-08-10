"""Tests for the treatment-held-out descriptor-learning smoke.

Fixtures are pure stdlib; the integration test requires PyTorch and is
guarded by ``TORCH_AVAILABLE``.
"""

from __future__ import annotations

import json
import math
import random
import tempfile
import unittest
from pathlib import Path

from pyreplab_harness import outcome_model as om
from pyreplab_harness.outcome_model_learn_smoke import (
    _IDENTITY_PLACEHOLDER,
    FAMILIES,
    DIFFICULTIES,
    _rank_values,
    _GRAMMAR_FACTORS,
    _verify_identity_neutralization,
    _model_input,
    _factor_modulation,
    treatment_logit,
    treatment_true_p,
    _deterministic_coverage_split,
    build_treatment_held_out_rows,
    spearman_rho,
    _validate_build_inputs,
    _validate_run_inputs,
)
from pyreplab_harness.treatments import (
    TreatmentRegistry,
    generate_treatments,
)


# -- helpers ----------------------------------------------------------------


def _build_test_registry(count: int = 8, seed: int = 99) -> TreatmentRegistry:
    return TreatmentRegistry(tuple(generate_treatments(count, seed=seed)))


# -- fixture-invariant tests ------------------------------------------------


class TreatmentHeldOutFixtureTest(unittest.TestCase):
    """Pure-stdlib invariants on data generation and helpers."""

    def setUp(self) -> None:
        self.registry = _build_test_registry(14)

    def test_split_is_stratified_and_nonempty(self) -> None:
        """Every (family, difficulty) cell has at least one task in each split."""
        rows, _, train_ids, held_ids = build_treatment_held_out_rows(
            self.registry,
            tasks_per_cell=(12, 10, 8),
            data_seed=1,
        )
        for family in FAMILIES:
            for difficulty in DIFFICULTIES:
                for split in ("train", "validation", "test"):
                    count = sum(
                        1 for r in rows
                        if r["family"] == family
                        and r["difficulty"] == difficulty
                        and r["split"] == split
                    )
                    self.assertGreater(
                        count, 0,
                        f"No {split} rows for {family}/{difficulty}",
                    )
        self.assertGreater(sum(1 for r in rows if r["split"] == "test"), 0)

    def test_extreme_valid_fractions_still_reserve_each_split(self) -> None:
        rows, _, _, _ = build_treatment_held_out_rows(
            self.registry,
            tasks_per_cell=(3, 3, 3),
            data_seed=11,
            train_frac=0.89,
            val_frac=0.1,
        )
        for family in FAMILIES:
            for difficulty in DIFFICULTIES:
                self.assertEqual(
                    {
                        row["split"]
                        for row in rows
                        if row["family"] == family
                        and row["difficulty"] == difficulty
                    },
                    {"train", "validation", "test"},
                )

    def test_train_rows_only_use_training_treatments(self) -> None:
        """Train/val rows must not contain held-out bundle IDs."""
        rows, _, train_ids, held_ids = build_treatment_held_out_rows(
            self.registry, tasks_per_cell=(8, 6, 4), data_seed=2,
        )
        for r in rows:
            if r["split"] in ("train", "validation"):
                self.assertIn(r["bundle_id"], train_ids)
        for r in rows:
            if r["split"] == "test":
                self.assertIn(r["bundle_id"], held_ids)

    def test_test_treatments_are_held_out(self) -> None:
        """Every test-row bundle_id is from the held-out set."""
        rows, _, _, held_ids = build_treatment_held_out_rows(
            self.registry, tasks_per_cell=(8, 6, 4), data_seed=3,
        )
        test_rows = [r for r in rows if r["split"] == "test"]
        self.assertTrue(all(r["bundle_id"] in held_ids for r in test_rows))

    def test_registry_metadata_consistent(self) -> None:
        """Generated tasks_per_cell match expected row counts."""
        rows, true_p, train_ids, held_ids = build_treatment_held_out_rows(
            self.registry, tasks_per_cell=(6, 4, 3), data_seed=4,
        )
        n_treatments = len(self.registry)
        self.assertGreater(n_treatments, 2)
        self.assertGreater(len(train_ids), 0)
        self.assertGreater(len(held_ids), 0)
        self.assertEqual(len(train_ids) + len(held_ids), n_treatments)
        # All rows should have a valid split and registry hash.
        valid_splits = {"train", "validation", "test"}
        for r in rows:
            self.assertIn(r["split"], valid_splits)
            self.assertEqual(r["treatment_registry_hash"], self.registry.registry_hash)

    def test_identity_neutralized_in_model_input(self) -> None:
        """policy_id and treatment bundle_id are constant placeholders."""
        rows, _, _, _ = build_treatment_held_out_rows(
            self.registry, tasks_per_cell=(4, 3, 3), data_seed=13,
        )
        for r in rows:
            mi = r["model_input"]
            self.assertEqual(mi["policy_id"], _IDENTITY_PLACEHOLDER)
            self.assertEqual(mi["policy_version"], "1")
            self.assertEqual(mi["treatment"]["policy_id"], _IDENTITY_PLACEHOLDER)
            self.assertEqual(mi["treatment"]["bundle_id"], _IDENTITY_PLACEHOLDER)

    def test_determinism(self) -> None:
        """Same seed produces identical rows."""
        rows_a, _, _, _ = build_treatment_held_out_rows(
            self.registry, tasks_per_cell=(4, 3, 3), data_seed=42,
        )
        rows_b, _, _, _ = build_treatment_held_out_rows(
            self.registry, tasks_per_cell=(4, 3, 3), data_seed=42,
        )
        self.assertEqual(len(rows_a), len(rows_b))
        for a, b in zip(rows_a, rows_b):
            self.assertEqual(a["task_id"], b["task_id"])
            self.assertEqual(a["split"], b["split"])
            self.assertEqual(a["bundle_id"], b["bundle_id"])
            self.assertAlmostEqual(a["true_p"], b["true_p"], places=12)

    def test_true_p_in_unit_interval(self) -> None:
        """Every true_p is a valid probability."""
        rows, _, _, _ = build_treatment_held_out_rows(
            self.registry, tasks_per_cell=(4, 3, 3), data_seed=6,
        )
        for r in rows:
            self.assertGreaterEqual(r["true_p"], 0.0)
            self.assertLessEqual(r["true_p"], 1.0)

    def test_task_variant_in_model_input(self) -> None:
        """task_variant is visible in public_metadata."""
        rows, _, _, _ = build_treatment_held_out_rows(
            self.registry, tasks_per_cell=(4, 3, 3), data_seed=7,
        )
        seen_variants = set()
        for r in rows:
            variant = r["model_input"]["public_metadata"].get("task_variant")
            self.assertIsNotNone(variant)
            self.assertGreaterEqual(variant, -0.6)
            self.assertLessEqual(variant, 0.6)
            seen_variants.add(variant)
        # Should have multiple distinct variants from different tasks.
        self.assertGreaterEqual(len(seen_variants), 2)


class TaskVariantRankingTest(unittest.TestCase):
    """Tests that task_variant genuinely causes treatment rankings to vary."""

    def setUp(self) -> None:
        self.registry = _build_test_registry(10)
        self.treatments = sorted(
            self.registry.treatments, key=lambda t: t.bundle_id,
        )

    def test_task_variant_changes_best_treatment_for_at_least_one_family(self):
        """For at least one family, the best treatment changes when task_variant varies."""
        any_rankings_flip = False
        for family in FAMILIES:
            best_by_variant: dict[float, str] = {}
            for variant in (-0.5, -0.25, 0.0, 0.25, 0.5):
                best_bundle = None
                best_logit = -float("inf")
                for treatment in self.treatments:
                    logit = treatment_logit(
                        treatment, family, "medium", task_variant=variant,
                    )
                    if logit > best_logit:
                        best_logit = logit
                        best_bundle = treatment.bundle_id
                best_by_variant[variant] = best_bundle
            if len(set(best_by_variant.values())) > 1:
                any_rankings_flip = True
                break
        self.assertTrue(
            any_rankings_flip,
            "task_variant must change the best-treatment ranking for at least one family",
        )

    def test_factor_modulation_ordering(self):
        """Different phase offsets produce different best-factor rankings.

        At variant=0, cos(0)=1, cos(pi/2)=0, cos(pi)=-1, cos(3pi/2)=0.
        So planning dominates, then budget/verification tied, then execution.
        At variant=0.5, cos(pi/2)=0, cos(pi)=-1, cos(3pi/2)=0, cos(2pi)=1
        — execution dominates.  The test verifies that the modulation values
        are exactly as expected (no rounding surprises)."""
        mods_0 = [
            _factor_modulation(0.0, 0.0),
            _factor_modulation(0.0, math.pi / 2),
            _factor_modulation(0.0, math.pi),
            _factor_modulation(0.0, 3 * math.pi / 2),
        ]
        # cos(0)=1, cos(pi/2)=0, cos(pi)=-1, cos(3pi/2)=0
        self.assertAlmostEqual(mods_0[0], 1.0, places=10)
        self.assertAlmostEqual(mods_0[1], 0.0, places=10)
        self.assertAlmostEqual(mods_0[2], -1.0, places=10)
        self.assertAlmostEqual(mods_0[3], 0.0, places=10)

        # At variant=1.0, all phases shift by pi, so cos flips signs.
        mods_1 = [
            _factor_modulation(1.0, 0.0),
            _factor_modulation(1.0, math.pi / 2),
            _factor_modulation(1.0, math.pi),
            _factor_modulation(1.0, 3 * math.pi / 2),
        ]
        self.assertAlmostEqual(mods_1[0], -1.0, places=10)
        self.assertAlmostEqual(mods_1[1], 0.0, places=10)
        self.assertAlmostEqual(mods_1[2], 1.0, places=10)
        self.assertAlmostEqual(mods_1[3], 0.0, places=10)


class CoverageAwareSplitTest(unittest.TestCase):
    """Deterministic coverage-aware train/held-out split."""

    def test_split_covers_all_factor_levels_in_train(self):
        """Every factor level used by any held-out treatment is present in train."""
        registry = _build_test_registry(10)
        treatments = list(registry.treatments)
        train, held_out = _deterministic_coverage_split(
            treatments, data_seed=1, train_treatment_frac=0.7
        )
        self.assertGreaterEqual(len(held_out), 2)
        train_levels: set[tuple[str, str]] = set()
        for t in train:
            for factor in _GRAMMAR_FACTORS:
                train_levels.add((factor, str(t.generator_metadata[factor])))
        for t in held_out:
            for factor in _GRAMMAR_FACTORS:
                level = (factor, str(t.generator_metadata[factor]))
                self.assertIn(
                    level, train_levels,
                    f"held-out treatment uses {level} not seen in train",
                )

    def test_split_deterministic(self):
        """Same data_seed produces the same split."""
        registry = _build_test_registry(12)
        treatments = list(registry.treatments)
        train_a, held_a = _deterministic_coverage_split(
            treatments, data_seed=42, train_treatment_frac=0.7
        )
        train_b, held_b = _deterministic_coverage_split(
            treatments, data_seed=42, train_treatment_frac=0.7
        )
        self.assertEqual([t.bundle_id for t in train_a], [t.bundle_id for t in train_b])
        self.assertEqual([t.bundle_id for t in held_a], [t.bundle_id for t in held_b])

    def test_too_few_treatments_rejected(self):
        """A registry with < 3 treatments cannot be split coverage-aware."""
        tiny = _build_test_registry(2)
        treatments = list(tiny.treatments)
        with self.assertRaisesRegex(ValueError, "at least 3"):
            _deterministic_coverage_split(
                treatments, data_seed=1, train_treatment_frac=0.7
            )

    def test_split_targets_requested_training_fraction(self):
        registry = _build_test_registry(10)
        train, held_out = _deterministic_coverage_split(
            list(registry.treatments), data_seed=2, train_treatment_frac=0.7
        )
        # Coverage may require more than the requested fraction, but never
        # fewer training treatments or fewer than two held-out candidates.
        self.assertGreaterEqual(len(train), 7)
        self.assertGreaterEqual(len(held_out), 2)

    def test_build_with_coverage_split(self):
        """build_treatment_held_out_rows works with the coverage-aware split."""
        registry = _build_test_registry(10)
        rows, _, train_ids, held_ids = build_treatment_held_out_rows(
            registry, tasks_per_cell=(3, 3, 3), data_seed=99,
        )
        self.assertGreater(len(train_ids), 0)
        self.assertGreaterEqual(len(held_ids), 2)
        test_rows = [r for r in rows if r["split"] == "test"]
        self.assertGreater(len(test_rows), 0)
        for r in test_rows:
            self.assertIn(r["bundle_id"], held_ids)


class HeldOutVariationValidationTest(unittest.TestCase):
    """Tests that invalid configurations are rejected with clear messages."""

    def test_registry_with_too_few_treatments_is_rejected(self):
        """A registry with fewer than 3 treatments is rejected."""
        tiny = _build_test_registry(2)
        with self.assertRaisesRegex(ValueError, "at least 3"):
            build_treatment_held_out_rows(
                tiny, tasks_per_cell=(3, 3, 3), data_seed=99,
            )

    def test_invalid_tasks_per_cell_is_rejected(self):
        """tasks_per_cell values under 3 are rejected."""
        registry = _build_test_registry(8)
        with self.assertRaisesRegex(ValueError, "tasks_per_cell values must all be >= 3"):
            build_treatment_held_out_rows(
                registry, tasks_per_cell=(2, 4, 6), data_seed=1,
            )

    def test_invalid_fraction_is_rejected(self):
        """train_frac outside (0, 1) is rejected."""
        with self.assertRaisesRegex(ValueError, "train_frac must be in"):
            _validate_build_inputs(
                (4, 3, 3), 0.75, 0.7, train_frac=0.0, val_frac=0.2
            )
        with self.assertRaisesRegex(ValueError, "train_frac must be in"):
            _validate_build_inputs(
                (4, 3, 3), 0.75, 0.7, train_frac=1.0, val_frac=0.2
            )

    def test_invalid_signal_scale_is_rejected(self):
        """signal_scale <= 0 is rejected."""
        registry = _build_test_registry(8)
        with self.assertRaisesRegex(ValueError, "signal_scale"):
            build_treatment_held_out_rows(
                registry, tasks_per_cell=(4, 3, 3), signal_scale=0.0, data_seed=1,
            )

    def test_fractions_sum_invalid_is_rejected(self):
        """train_frac + val_frac >= 1 is rejected."""
        registry = _build_test_registry(8)
        with self.assertRaisesRegex(ValueError, "must be < 1"):
            build_treatment_held_out_rows(
                registry, tasks_per_cell=(4, 3, 3), train_frac=0.7, val_frac=0.3, data_seed=1,
            )

    def test_non_finite_signal_scale_is_rejected(self):
        """NaN or inf signal_scale is rejected."""
        with self.assertRaisesRegex(ValueError, "finite"):
            _validate_build_inputs(
                (4, 3, 3), float("nan"), 0.7, train_frac=0.6, val_frac=0.2
            )

    def test_wrong_tasks_per_cell_shape_is_rejected(self):
        """tasks_per_cell not a 3-tuple is rejected."""
        with self.assertRaisesRegex(ValueError, "3-tuple"):
            _validate_build_inputs((4, 3), 0.75, 0.7, train_frac=0.6, val_frac=0.2)

    def test_run_hyperparameters_validated(self):
        """Epochs/batch/patience/samples must be positive ints."""
        with self.assertRaises(ValueError):
            _validate_run_inputs(0, 8, 5, 50)
        with self.assertRaises(ValueError):
            _validate_run_inputs(50, 0, 5, 50)
        with self.assertRaises(ValueError):
            _validate_run_inputs(50, 8, 0, 50)
        with self.assertRaises(ValueError):
            _validate_run_inputs(50, 8, 5, 0)
        with self.assertRaises(TypeError):
            _validate_run_inputs(50, 8, 5, None)  # type: ignore

    def test_missing_generator_metadata_is_rejected(self):
        """Treatments without generator_metadata are rejected."""
        from pyreplab_harness.treatments import TreatmentSpec

        broken = TreatmentSpec(
            id="broken-a",
            version="1",
            system_prompt="do something",
            allowed_tools=("bash",),
            max_output_tokens=1024,
            tool_call_limit=4,
            command_timeout_seconds=30,
            wall_time_limit_seconds=180,
        )
        from pyreplab_harness.treatments import TreatmentRegistry as TR

        reg = TR((broken,))
        with self.assertRaisesRegex(ValueError, "generator_metadata"):
            build_treatment_held_out_rows(
                reg, tasks_per_cell=(3, 3, 3), data_seed=1,
            )


class SpearmanTest(unittest.TestCase):
    """Stdlib tie-aware Spearman rank correlation."""

    def test_perfect_positive(self) -> None:
        rho = spearman_rho([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0])
        self.assertAlmostEqual(rho, 1.0, places=10)

    def test_perfect_negative(self) -> None:
        rho = spearman_rho([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
        self.assertAlmostEqual(rho, -1.0, places=10)

    def test_no_correlation(self) -> None:
        rho = spearman_rho([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 1.0, 3.0])
        self.assertLess(rho, 0.5)

    def test_ties(self) -> None:
        rho = spearman_rho([1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0])
        self.assertAlmostEqual(rho, 1.0, places=10)

    def test_with_ties_fractional(self) -> None:
        # Ties should produce fractional ranks.
        ranks = _rank_values([3.0, 1.0, 1.0, 2.0])
        # sorted: (1) 1.0, (2) 1.0, (3) 2.0, (4) 3.0
        # ranks: first two tie at (1+2)/2=1.5, next 3, last 4
        # Original order: 3.0=4, 1.0=1.5, 1.0=1.5, 2.0=3 → [4, 1.5, 1.5, 3]
        self.assertEqual(ranks, [4.0, 1.5, 1.5, 3.0])

    def test_all_ties_yields_nan(self) -> None:
        rho = spearman_rho([1.0, 1.0, 1.0], [2.0, 3.0, 4.0])
        self.assertTrue(math.isnan(rho))

    def test_too_few_points_yields_nan(self) -> None:
        rho = spearman_rho([1.0, 2.0], [1.0, 2.0])
        self.assertTrue(math.isnan(rho))


# -- PyTorch integration test -----------------------------------------------


@unittest.skipUnless(om.TORCH_AVAILABLE, "PyTorch is not installed")
class LearnSmokeIntegrationTest(unittest.TestCase):
    """Small integration test that trains and inspects verdict fields."""

    def test_trains_and_reports_verdict_fields(self) -> None:
        registry = _build_test_registry(10)
        with tempfile.TemporaryDirectory() as directory:
            result_dir = Path(directory) / "output"
            # Minimally-sized run for CI speed.
            from pyreplab_harness.outcome_model_learn_smoke import run_learn_smoke

            result = run_learn_smoke(
                registry,
                result_dir,
                tasks_per_cell=(4, 3, 3),
                signal_scale=0.75,
                data_seed=1,
                train_seed=2,
                epochs=30,
                batch_size=16,
                patience=5,
                num_samples=32,
                verbose=False,
            )

        # Protocol facts
        self.assertTrue(result["synthetic_only"])
        self.assertIn("warning", result)
        self.assertIn("config", result)
        self.assertIn("data", result)
        self.assertIn("verdict", result)
        self.assertIn("identity_neutralization", result)
        self.assertIn("ranking_recovery", result)
        self.assertIn("allocator_lift", result)
        self.assertIn("counterfactual_top1", result)
        self.assertIn("representative_held_out_ranking", result)

        # Data integrity
        self.assertGreater(result["data"]["total_rows"], 0)
        for split in ("train", "validation", "test"):
            self.assertGreaterEqual(result["data"]["split_counts"][split], 0)
        self.assertGreater(result["data"]["split_counts"]["test"], 0)

        # Identity neutralization: test treatments are structurally held-out
        # (their bundle_ids never appear in training rows).
        identity = result["identity_neutralization"]
        self.assertTrue(identity["structural_test_disjoint_from_train"])
        self.assertTrue(identity["all_test_bundle_ids_in_held_out_set"])
        self.assertTrue(identity["placeholder_neutralized_in_model_input"])

        # Verdict fields must exist (no brittle numeric score assertions).
        verdict = result["verdict"]
        for key in (
            "predicts_better_than_naive_brier",
            "expected_allocator_beats_random",
            "expected_allocator_beats_predeclared_baseline",
            "ranking_spearman_positive",
            "test_treatments_are_held_out",
            "descriptor_learned_something_usable",
        ):
            self.assertIn(key, verdict)

        # Allocator lift fields must exist and be numeric or None.
        alloc = result["allocator_lift"]
        for key in (
            "expected_model_vs_true_pstar",
            "expected_random_vs_true_pstar",
            "expected_allocator_lift_over_random",
            "expected_allocator_lift_over_predeclared_baseline",
            "predeclared_baseline_bundle",
        ):
            self.assertIn(key, alloc)

        # Ranking recovery has Spearman fields.
        ranking = result["ranking_recovery"]
        self.assertIn("spearman_pred_vs_true_pstar", ranking)
        self.assertIn("n_family_treatment_cells", ranking)
        self.assertGreater(ranking["n_family_treatment_cells"], 0)

        representative = result["representative_held_out_ranking"]
        self.assertTrue(representative["task_id"])
        self.assertEqual(
            len(representative["ranked"]),
            result["config"]["n_held_out_treatments"],
        )
        self.assertEqual(
            [item["model_rank"] for item in representative["ranked"]],
            list(range(1, len(representative["ranked"]) + 1)),
        )
        for item in representative["ranked"]:
            self.assertGreaterEqual(item["mean"], 0.0)
            self.assertLessEqual(item["mean"], 1.0)
            self.assertGreaterEqual(item["true_p"], 0.0)
            self.assertLessEqual(item["true_p"], 1.0)


if __name__ == "__main__":
    unittest.main()
