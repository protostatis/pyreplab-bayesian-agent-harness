"""Tests for the calibration context builder and split utilities."""

from __future__ import annotations

import math
import unittest

from pyreplab_harness import calibration


class LeakageAuditTest(unittest.TestCase):
    def test_clean_context_passes(self) -> None:
        context = {
            "success": 1.0,
            "cost": 100.0,
            "task_features": {"template": "extraction", "difficulty": "easy"},
        }
        violations = calibration.audit_context_leakage(context)
        self.assertEqual(len(violations), 0,
                         f"Clean context should have no violations, got: {violations}")

    def test_forbidden_page_text_detected(self) -> None:
        context = {
            "page_text": "some html content",
            "success": 1.0,
        }
        violations = calibration.audit_context_leakage(context)
        self.assertGreater(len(violations), 0,
                           "Should detect 'page_text' forbidden field")

    def test_forbidden_url_detected(self) -> None:
        context = {
            "url": "https://example.com",
        }
        violations = calibration.audit_context_leakage(context)
        self.assertGreater(len(violations), 0,
                           "Should detect 'url' forbidden field")

    def test_forbidden_policy_id_detected(self) -> None:
        context = {
            "policy_id": "ub-direct-text_first-submit_directly-fail_fast-lean",
        }
        violations = calibration.audit_context_leakage(context)
        self.assertGreater(len(violations), 0,
                           "Should detect 'policy_id' forbidden field")

    def test_forbidden_bundle_id_detected(self) -> None:
        context = {
            "bundle_id": "some-bundle-id@1-abc12345",
        }
        violations = calibration.audit_context_leakage(context)
        self.assertGreater(len(violations), 0,
                           "Should detect 'bundle_id' forbidden field")

    def test_forbidden_selector_detected(self) -> None:
        context = {
            "model_input": {"selector": "#content > div"},
        }
        violations = calibration.audit_context_leakage(context)
        self.assertGreater(len(violations), 0,
                           "Should detect 'selector' forbidden field")

    def test_nested_forbidden_detected(self) -> None:
        context = {
            "model_input": {
                "task": {
                    "answer": "42",
                    "difficulty": "easy",
                },
            },
        }
        violations = calibration.audit_context_leakage(context)
        self.assertGreater(len(violations), 0,
                           f"Should detect nested 'answer' forbidden field, got {violations}")

    def test_clean_includes_path(self) -> None:
        context = {
            "model_input": {
                "task": {
                    "template": "extraction",
                    "difficulty": "easy",
                },
                "treatment": {
                    "numeric": {"enforced_tool_call_cap": 6},
                },
            },
            "outcome": {
                "verified_success": True,
                "output_token_cost": 150,
            },
        }
        violations = calibration.audit_context_leakage(context)
        self.assertEqual(len(violations), 0,
                         f"Clean context should have no violations, got: {violations}")


class BuildCalibrationContextTest(unittest.TestCase):
    def _make_row(self, task_id: str, success: bool, cost: float,
                  term_class: str = "normal_completion") -> dict:
        return {
            "task_id": task_id,
            "verified_success": success,
            "cost": cost,
            "output_token_cost": cost,
            "termination_class": term_class,
            "task_embedding": [0.1, 0.2, 0.3],
        }

    def test_k0_returns_empty(self) -> None:
        rows = [self._make_row("t1", True, 100.0) for _ in range(8)]
        ctx = calibration.build_calibration_context(rows, k=0)
        self.assertEqual(ctx["k_actual"], 0)
        self.assertEqual(len(ctx["success"]), 0)

    def test_k4_uses_first_4(self) -> None:
        rows = [self._make_row(f"t{i}", i % 2 == 0, 100.0 * i) for i in range(8)]
        ctx = calibration.build_calibration_context(rows, k=4)
        self.assertEqual(ctx["k_actual"], 4)
        self.assertEqual(len(ctx["success"]), 4)
        self.assertEqual(len(ctx["cost"]), 4)
        self.assertEqual(len(ctx["mask"]), 4)
        # First row success.
        self.assertAlmostEqual(ctx["success"][0], 1.0)  # t0: even -> True

    def test_k_gt_len_pads_with_invalid(self) -> None:
        rows = [self._make_row("t1", True, 100.0)]
        ctx = calibration.build_calibration_context(rows, k=4)
        self.assertEqual(ctx["k_actual"], 1)  # only 1 actual
        self.assertEqual(len(ctx["success"]), 4)
        # Padding has mask=0.
        self.assertEqual(ctx["mask"][0], 1.0)
        self.assertEqual(ctx["mask"][1], 0.0)

    def test_normalization_applied(self) -> None:
        rows = [self._make_row(f"t{i}", True, 200.0 + i * 10) for i in range(5)]
        norm_stats = {"cost_mean": math.log1p(200.0), "cost_std": 1.0}
        ctx = calibration.build_calibration_context(rows, k=5, normalization_stats=norm_stats)
        # First row log1p(cost) equals the fitted mean.
        self.assertAlmostEqual(ctx["cost"][0], 0.0)

    def test_negative_cost_raises(self) -> None:
        rows = [self._make_row("t1", True, -1.0)]
        with self.assertRaisesRegex(ValueError, "invalid cost"):
            calibration.build_calibration_context(rows, k=1)

    def test_conflicting_cost_fields_raise(self) -> None:
        row = self._make_row("t1", True, 10.0)
        row["output_token_cost"] = 11.0
        with self.assertRaisesRegex(ValueError, "conflicting cost fields"):
            calibration.build_calibration_context([row], k=1)

    def test_termination_encoding(self) -> None:
        rows = [
            self._make_row("t1", True, 100.0, "normal_completion"),
            self._make_row("t2", False, 50.0, "tool_call_limit"),
        ]
        ctx = calibration.build_calibration_context(rows, k=2)
        self.assertEqual(len(ctx["term_onehot"]), 2)
        self.assertEqual(len(ctx["term_onehot"][0]), 6)
        # normal_completion -> index 0.
        self.assertEqual(ctx["term_onehot"][0][0], 1.0)
        # tool_call_limit -> index 1.
        self.assertEqual(ctx["term_onehot"][1][1], 1.0)

    def test_consumes_nested_cnp_embedding(self) -> None:
        """build_calibration_context finds task_embedding in
        model_input.task.task_embedding.vector."""
        nested_emb = [0.1, 0.2, 0.3, 0.4]
        row = {
            "task_id": "t-nested",
            "verified_success": True,
            "cost": 100.0,
            "output_token_cost": 100,
            "termination_class": "normal_completion",
            "model_input": {
                "task": {
                    "task_embedding": {
                        "encoder": "sha256_ascii_projection_v1",
                        "version": 1,
                        "vector": list(nested_emb),
                    },
                    "template": "extraction",
                },
            },
        }
        ctx = calibration.build_calibration_context([row], k=1)
        self.assertEqual(len(ctx["task_feature_vectors"]), 1)
        self.assertEqual(ctx["task_feature_vectors"][0], list(nested_emb))

    def test_legacy_top_level_embedding_still_works(self) -> None:
        """Legacy rows with top-level task_embedding are still consumed."""
        legacy_emb = [0.5, 0.6, 0.7]
        row = {
            "task_id": "t-legacy",
            "verified_success": True,
            "cost": 50.0,
            "output_token_cost": 50,
            "termination_class": "normal_completion",
            "task_embedding": list(legacy_emb),
        }
        ctx = calibration.build_calibration_context([row], k=1)
        self.assertEqual(ctx["task_feature_vectors"][0], list(legacy_emb))


class FrozenCalibrationSplitTest(unittest.TestCase):
    def test_nested_prefixes(self) -> None:
        tasks = [f"task-{i}" for i in range(24)]
        result = calibration.frozen_calibration_split(
            tasks, k_sizes=(0, 4, 8, 16), seed=42,
        )
        self.assertEqual(len(result["ordered_tasks"]), 16)
        # k=4 is prefix of k=8.
        k4 = result["k_4"]
        k8 = result["k_8"]
        self.assertEqual(k4, k8[:4])

    def test_all_k_sizes_returned(self) -> None:
        tasks = [f"task-{i}" for i in range(20)]
        result = calibration.frozen_calibration_split(
            tasks, k_sizes=(4, 8, 16), seed=42,
        )
        self.assertEqual(len(result["k_4"]), 4)
        self.assertEqual(len(result["k_8"]), 8)
        self.assertEqual(len(result["k_16"]), 16)

    def test_deterministic_given_seed(self) -> None:
        tasks = [f"task-{i}" for i in range(20)]
        r1 = calibration.frozen_calibration_split(tasks, k_sizes=(4, 8), seed=42)
        r2 = calibration.frozen_calibration_split(tasks, k_sizes=(4, 8), seed=42)
        self.assertEqual(r1["k_4"], r2["k_4"])
        self.assertEqual(r1["k_8"], r2["k_8"])

    def test_non_increasing_raises(self) -> None:
        tasks = [f"task-{i}" for i in range(10)]
        with self.assertRaises(ValueError):
            calibration.frozen_calibration_split(tasks, k_sizes=(8, 4), seed=42)

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            calibration.frozen_calibration_split([], seed=42)

    def test_insufficient_unique_tasks_raises(self) -> None:
        tasks = [f"task-{i}" for i in range(8)]
        with self.assertRaisesRegex(ValueError, "at least 16 unique"):
            calibration.frozen_calibration_split(tasks, seed=42)

    def test_duplicate_tasks_raise(self) -> None:
        tasks = [f"task-{i}" for i in range(16)] + ["task-0"]
        with self.assertRaisesRegex(ValueError, "must be unique"):
            calibration.frozen_calibration_split(tasks, seed=42)


class PolicyTaskSplitTest(unittest.TestCase):
    def test_no_overlap(self) -> None:
        tasks = [f"task-{i}" for i in range(100)]
        policies = [f"policy-{j}" for j in range(60)]

        result = calibration.policy_task_split(
            tasks, policies,
            ratios={
                "task_meta_train": 0.4, "task_dev_cal": 0.1, "task_dev_target": 0.1,
                "task_final_cal": 0.1, "task_final_known": 0.2, "task_final_held": 0.1,
                "policy_meta_train": 0.6, "policy_development": 0.2, "policy_final": 0.2,
            },
            seed=42,
        )

        # Task splits should not overlap.
        all_task_assignments: list[str] = []
        for key in result["tasks"]:
            all_task_assignments.extend(result["tasks"][key])
        self.assertEqual(len(all_task_assignments), len(set(all_task_assignments)))

    def test_returns_all_keys(self) -> None:
        tasks = [f"task-{i}" for i in range(50)]
        policies = [f"policy-{j}" for j in range(30)]

        result = calibration.policy_task_split(
            tasks, policies,
            ratios={
                "task_meta_train": 0.4, "task_dev_cal": 0.1, "task_dev_target": 0.1,
                "task_final_cal": 0.1, "task_final_known": 0.2, "task_final_held": 0.1,
                "policy_meta_train": 0.6, "policy_development": 0.2, "policy_final": 0.2,
            },
            seed=42,
        )

        for key in ("task_meta_train", "task_dev_cal", "task_dev_target",
                     "task_final_cal", "task_final_known", "task_final_held"):
            self.assertIn(key, result["tasks"])
        for key in ("policy_meta_train", "policy_development", "policy_final"):
            self.assertIn(key, result["policies"])

    def test_deterministic(self) -> None:
        tasks = [f"task-{i}" for i in range(30)]
        policies = [f"policy-{j}" for j in range(20)]
        ratios = {
            "task_meta_train": 0.4, "task_dev_cal": 0.1, "task_dev_target": 0.1,
            "task_final_cal": 0.1, "task_final_known": 0.2, "task_final_held": 0.1,
            "policy_meta_train": 0.6, "policy_development": 0.2, "policy_final": 0.2,
        }
        r1 = calibration.policy_task_split(tasks, policies, ratios, seed=42)
        r2 = calibration.policy_task_split(tasks, policies, ratios, seed=42)
        for key in r1["tasks"]:
            self.assertEqual(r1["tasks"][key], r2["tasks"][key])

    def test_missing_ratio_keys_raises(self) -> None:
        with self.assertRaises(ValueError):
            calibration.policy_task_split(
                ["a", "b"], ["x", "y"],
                ratios={"task_meta_train": 0.5},  # missing keys
                seed=42,
            )

    def test_insufficient_inputs_raise_instead_of_empty_splits(self) -> None:
        ratios = {
            "task_meta_train": 0.4, "task_dev_cal": 0.1, "task_dev_target": 0.1,
            "task_final_cal": 0.1, "task_final_known": 0.2, "task_final_held": 0.1,
            "policy_meta_train": 0.6, "policy_development": 0.2, "policy_final": 0.2,
        }
        with self.assertRaisesRegex(ValueError, "at least 6 tasks"):
            calibration.policy_task_split(
                ["t1", "t2"], ["p1", "p2", "p3"], ratios, seed=42
            )
        with self.assertRaisesRegex(ValueError, "at least 3 policies"):
            calibration.policy_task_split(
                [f"t{i}" for i in range(6)], ["p1", "p2"], ratios, seed=42
            )

    def test_duplicate_inputs_raise(self) -> None:
        ratios = {
            "task_meta_train": 0.4, "task_dev_cal": 0.1, "task_dev_target": 0.1,
            "task_final_cal": 0.1, "task_final_known": 0.2, "task_final_held": 0.1,
            "policy_meta_train": 0.6, "policy_development": 0.2, "policy_final": 0.2,
        }
        with self.assertRaisesRegex(ValueError, "tasks must be unique"):
            calibration.policy_task_split(
                ["t1"] * 6, ["p1", "p2", "p3"], ratios, seed=42
            )
        with self.assertRaisesRegex(ValueError, "policies must be unique"):
            calibration.policy_task_split(
                [f"t{i}" for i in range(6)], ["p1"] * 3, ratios, seed=42
            )


class FitNormalizationStatsTest(unittest.TestCase):
    def test_non_empty_data(self) -> None:
        rows = [
            {"cost": 100.0},
            {"cost": 200.0},
            {"cost": 150.0},
        ]
        stats = calibration.fit_normalization_stats(rows)
        expected = sum(math.log1p(value) for value in (100.0, 200.0, 150.0)) / 3
        self.assertAlmostEqual(stats["cost_mean"], expected)
        self.assertEqual(stats["cost_transform"], "log1p_zscore")
        self.assertEqual(stats["n_rows"], 3)
        self.assertGreater(stats["cost_std"], 0)

    def test_empty_data(self) -> None:
        stats = calibration.fit_normalization_stats([])
        self.assertEqual(stats["cost_mean"], 0.0)
        self.assertEqual(stats["cost_std"], 1.0)
        self.assertEqual(stats["n_rows"], 0)

    def test_uses_output_token_cost_fallback(self) -> None:
        rows = [
            {"output_token_cost": 50.0},
            {"output_token_cost": 100.0},
        ]
        stats = calibration.fit_normalization_stats(rows)
        self.assertAlmostEqual(
            stats["cost_mean"],
            (math.log1p(50.0) + math.log1p(100.0)) / 2,
        )

    def test_conflicting_cost_fields_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "conflicting cost fields"):
            calibration.fit_normalization_stats(
                [{"cost": 50.0, "output_token_cost": 51.0}]
            )

    def test_ignores_non_finite(self) -> None:
        rows = [
            {"cost": 100.0},
            {"cost": float("inf")},
            {"cost": 200.0},
            {"cost": float("nan")},
        ]
        stats = calibration.fit_normalization_stats(rows)
        self.assertAlmostEqual(
            stats["cost_mean"],
            (math.log1p(100.0) + math.log1p(200.0)) / 2,
        )
        self.assertEqual(stats["n_rows"], 2)

    def test_single_row(self) -> None:
        rows = [{"cost": 100.0}]
        stats = calibration.fit_normalization_stats(rows)
        self.assertAlmostEqual(stats["cost_mean"], math.log1p(100.0))
        self.assertEqual(stats["cost_std"], 1.0)  # fallback for single point


if __name__ == "__main__":
    unittest.main()
