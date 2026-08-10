"""Tests for the success-cost frontier evaluator."""

from __future__ import annotations

import unittest

from pyreplab_harness import frontier_eval


class ComputeFrontierTest(unittest.TestCase):
    def _make_predictions(self, tasks: int, policies: int) -> list:
        return [
            [
                {"success_prob": 0.5 + (j - policies / 2) * 0.1, "cost_mean": 100.0 + j * 20}
                for j in range(policies)
            ]
            for _ in range(tasks)
        ]

    def _make_outcomes(self, tasks: int, policies: int) -> list:
        return [
            [
                {"verified_success": (i + j) % 3 == 0, "cost": 80.0 + j * 30}
                for j in range(policies)
            ]
            for i in range(tasks)
        ]

    def test_frontier_area_positive(self) -> None:
        predictions = self._make_predictions(10, 4)
        outcomes = self._make_outcomes(10, 4)
        result = frontier_eval.compute_frontier(
            predictions, outcomes, lambda_grid=[0.0, 0.5, 1.0]
        )
        self.assertGreaterEqual(result["frontier_area"], 0.0)
        self.assertEqual(result["n_tasks"], 10)
        self.assertEqual(result["n_policies"], 4)

    def test_lambda_zero_included(self) -> None:
        predictions = self._make_predictions(5, 3)
        outcomes = self._make_outcomes(5, 3)
        lambda_grid = [0.0, 0.5, 1.0, 2.0]
        result = frontier_eval.compute_frontier(
            predictions, outcomes, lambda_grid=lambda_grid,
        )
        self.assertEqual(result["lambda_grid"], lambda_grid)
        # Lambda=0 result should be present.
        lambda0 = next((r for r in result["lambda_results"] if r["lambda"] == 0.0), None)
        self.assertIsNotNone(lambda0, "lambda=0 must be in results")

    def test_pure_success_rate_in_range(self) -> None:
        predictions = self._make_predictions(8, 3)
        outcomes = self._make_outcomes(8, 3)
        result = frontier_eval.compute_frontier(
            predictions, outcomes, lambda_grid=[0.0],
        )
        self.assertTrue(0.0 <= result["pure_success_rate"] <= 1.0)

    def test_oracle_frontier_not_less_than_selected(self) -> None:
        predictions = self._make_predictions(12, 5)
        outcomes = self._make_outcomes(12, 5)
        result = frontier_eval.compute_frontier(
            predictions, outcomes, lambda_grid=[0.0, 1.0],
        )
        self.assertGreaterEqual(
            result["oracle_pure_success_rate"],
            result["pure_success_rate"],
        )

    def test_mismatched_lengths_raises(self) -> None:
        predictions = self._make_predictions(5, 3)
        outcomes = self._make_outcomes(3, 3)
        with self.assertRaises(ValueError):
            frontier_eval.compute_frontier(predictions, outcomes, [0.0])

    def test_different_policies_raises(self) -> None:
        predictions = self._make_predictions(5, 3)
        outcomes = self._make_outcomes(5, 4)
        with self.assertRaises(ValueError):
            frontier_eval.compute_frontier(predictions, outcomes, [0.0])

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            frontier_eval.compute_frontier([], [], [0.0])

    def test_result_structure(self) -> None:
        predictions = self._make_predictions(4, 2)
        outcomes = self._make_outcomes(4, 2)
        result = frontier_eval.compute_frontier(
            predictions, outcomes, lambda_grid=[0.0, 1.0],
        )
        self.assertIn("frontier_area", result)
        self.assertIn("pure_success_rate", result)
        self.assertIn("oracle_frontier_area", result)
        self.assertIn("oracle_pure_success_rate", result)
        self.assertIn("regret_vs_oracle", result)
        self.assertIn("lambda_results", result)


class GlobalRankingTest(unittest.TestCase):
    def test_perfect_correlation(self) -> None:
        result = frontier_eval.compute_global_ranking_metrics(
            [0.9, 0.7, 0.5, 0.3],
            [0.8, 0.6, 0.4, 0.2],
        )
        self.assertAlmostEqual(result["spearman_rho"], 1.0, places=1)
        self.assertAlmostEqual(result["pairwise_accuracy"], 1.0, places=1)
        self.assertTrue(result["top1_correct"])

    def test_anti_correlation(self) -> None:
        result = frontier_eval.compute_global_ranking_metrics(
            [0.9, 0.7, 0.5, 0.3],
            [0.2, 0.4, 0.6, 0.8],
        )
        self.assertAlmostEqual(result["spearman_rho"], -1.0, places=1)

    def test_small_n_returns_nan(self) -> None:
        result = frontier_eval.compute_global_ranking_metrics(
            [0.5], [0.5],
        )
        self.assertTrue(result["spearman_rho"] != result["spearman_rho"])  # NaN check

    def test_top3_includes_top1(self) -> None:
        result = frontier_eval.compute_global_ranking_metrics(
            [0.9, 0.7, 0.5, 0.3],
            [0.8, 0.6, 0.4, 0.2],
        )
        if result["top1_correct"]:
            self.assertTrue(result["top3_correct"])


class EvaluateAllocatorTest(unittest.TestCase):
    def _make_predictions(self, tasks: int, policies: int) -> list:
        return [
            [
                {"success_prob": 0.5 + (j - policies / 2) * 0.1, "cost_mean": 50.0 + j * 10}
                for j in range(policies)
            ]
            for _ in range(tasks)
        ]

    def _make_outcomes(self, tasks: int, policies: int) -> list:
        return [
            [
                {"verified_success": (i + j) % 3 == 0, "cost": 60.0 + j * 10}
                for j in range(policies)
            ]
            for i in range(tasks)
        ]

    def test_evaluate_structure(self) -> None:
        predictions = self._make_predictions(10, 4)
        outcomes = self._make_outcomes(10, 4)
        result = frontier_eval.evaluate_allocator(
            predictions, outcomes,
            policies=["p0", "p1", "p2", "p3"],
            lambda_grid=[0.0, 0.5, 1.0],
        )
        self.assertIn("frontier", result)
        self.assertIn("global_ranking", result)
        self.assertIn("n_tasks", result)
        self.assertIn("n_policies", result)

    def test_complete_panel_rejects_missing_cells(self) -> None:
        predictions = self._make_predictions(5, 3)
        predictions[0] = predictions[0][:2]  # missing one policy
        outcomes = self._make_outcomes(5, 3)

        with self.assertRaises(ValueError):
            frontier_eval.evaluate_allocator(
                predictions, outcomes,
                policies=["a", "b", "c"],
                lambda_grid=[0.0],
            )

    def test_with_task_labels(self) -> None:
        predictions = self._make_predictions(6, 3)
        outcomes = self._make_outcomes(6, 3)
        task_labels = [
            {"template": "extraction", "difficulty": "easy"},
            {"template": "extraction", "difficulty": "hard"},
            {"template": "navigation", "difficulty": "easy"},
            {"template": "navigation", "difficulty": "hard"},
            {"template": "search", "difficulty": "easy"},
            {"template": "search", "difficulty": "hard"},
        ]
        result = frontier_eval.evaluate_allocator(
            predictions, outcomes,
            policies=["a", "b", "c"],
            lambda_grid=[0.0, 0.5],
            task_labels=task_labels,
        )
        self.assertIn("per_template", result)
        self.assertIn("per_difficulty", result)
        self.assertIn("extraction", result["per_template"])

    def test_with_bootstrap(self) -> None:
        predictions = self._make_predictions(8, 3)
        outcomes = self._make_outcomes(8, 3)
        result = frontier_eval.evaluate_allocator(
            predictions, outcomes,
            policies=["a", "b", "c"],
            lambda_grid=[0.0],
            bootstrap_config={"seed": 42, "num_trials": 50},
        )
        self.assertIn("bootstrap", result)
        self.assertIn("frontier_area", result["bootstrap"])

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            frontier_eval.evaluate_allocator(
                [], [], policies=[], lambda_grid=[0.0],
            )


class CompareToBaselinesTest(unittest.TestCase):
    def test_comparison_structure(self) -> None:
        cnp_eval = {
            "frontier": {"frontier_area": 0.5, "pure_success_rate": 0.7},
            "global_ranking": {"pairwise_accuracy": 0.65},
        }
        baselines = {
            "random": {
                "frontier": {"frontier_area": 0.3, "pure_success_rate": 0.5},
                "global_ranking": {"pairwise_accuracy": 0.5},
            },
            "best_fixed": {
                "frontier": {"frontier_area": 0.45, "pure_success_rate": 0.65},
                "global_ranking": {"pairwise_accuracy": 0.55},
            },
        }
        result = frontier_eval.compare_to_baselines(cnp_eval, baselines)
        self.assertIn("comparisons", result)
        self.assertIn("random", result["comparisons"])
        self.assertIn("best_fixed", result["comparisons"])
        self.assertIn("frontier_area_delta", result["comparisons"]["random"])
        self.assertTrue(result["cnp_beats_all"])


class PairwiseRankingAccuracyTest(unittest.TestCase):
    def test_perfect(self) -> None:
        acc = frontier_eval.pairwise_ranking_accuracy(
            [1.0, 0.8, 0.6, 0.4],
            [0.9, 0.7, 0.5, 0.3],
        )
        self.assertAlmostEqual(acc, 1.0)

    def test_random(self) -> None:
        acc = frontier_eval.pairwise_ranking_accuracy(
            [1.0, 0.8, 0.6, 0.4],
            [0.4, 0.6, 0.8, 1.0],
        )
        self.assertAlmostEqual(acc, 0.0)

    def test_small_n(self) -> None:
        acc = frontier_eval.pairwise_ranking_accuracy([0.5], [0.5])
        self.assertTrue(acc != acc)  # NaN


if __name__ == "__main__":
    unittest.main()
