from __future__ import annotations

import unittest

from pyreplab_harness import routing_utility as ru


class ScoreCandidatesTest(unittest.TestCase):
    def test_utility_formula_and_output_units(self) -> None:
        candidates = (
            {"candidate_id": "table", "predicted_success": 0.84, "predicted_output_tokens": 12345},
            {"candidate_id": "form", "predicted_success": 0.77, "predicted_output_tokens": 6789},
        )
        scored = ru.score_candidates(
            candidates,
            1.25,
            candidate_order=("table", "form"),
        )
        self.assertEqual(len(scored), 2)
        self.assertEqual(scored[0]["candidate_id"], "table")
        self.assertAlmostEqual(scored[0]["predicted_output_cost_units"], 1.2345)
        self.assertAlmostEqual(
            scored[0]["utility"],
            0.84 - 1.25 * 1.2345,
        )
        self.assertAlmostEqual(scored[1]["utility"], 0.77 - 1.25 * 0.6789)

    def test_return_type_and_registry_position(self) -> None:
        candidates = (
            {"candidate_id": "a", "predicted_success": 0.1, "predicted_output_tokens": 0},
            {"candidate_id": "b", "predicted_success": 0.2, "predicted_output_tokens": 1},
        )
        scored = ru.score_candidates(candidates, 0.0, candidate_order=("a", "b"))
        self.assertIsInstance(scored, tuple)
        self.assertEqual(scored[0]["registry_position"], 0)
        self.assertEqual(scored[1]["registry_position"], 1)

    def test_tie_break_higher_success_before_lower_cost(self) -> None:
        # Both utilities equal to 0.8 under lambda=1.0; higher success wins.
        candidates = (
            {"candidate_id": "a", "predicted_success": 0.9, "predicted_output_tokens": 1000},
            {"candidate_id": "b", "predicted_success": 0.8, "predicted_output_tokens": 0},
        )
        result = ru.select_candidate(candidates, 1.0, candidate_order=("a", "b"))
        self.assertEqual(result["selected_candidate_id"], "a")

    def test_tie_break_cost_when_success_tied(self) -> None:
        # Both utilities equal to 0.8; lower cost wins.
        candidates = (
            {"candidate_id": "a", "predicted_success": 0.9, "predicted_output_tokens": 1000},
            {"candidate_id": "b", "predicted_success": 0.9, "predicted_output_tokens": 1200},
        )
        result = ru.select_candidate(candidates, 1.0, candidate_order=("a", "b"))
        self.assertEqual(result["selected_candidate_id"], "a")

    def test_tie_break_registry_order_when_full_tie(self) -> None:
        candidates = (
            {"candidate_id": "zzz", "predicted_success": 0.7, "predicted_output_tokens": 1000},
            {"candidate_id": "aaa", "predicted_success": 0.7, "predicted_output_tokens": 1000},
        )
        result = ru.select_candidate(candidates, 1.0, candidate_order=("zzz", "aaa"))
        self.assertEqual(result["selected_candidate_id"], "zzz")

    def test_select_receipt_is_aggregate_safe(self) -> None:
        candidates = (
            {"candidate_id": "a", "predicted_success": 0.6, "predicted_output_tokens": 10},
            {"candidate_id": "b", "predicted_success": 0.7, "predicted_output_tokens": 20},
        )
        result = ru.select_candidate(candidates, 0.5, candidate_order=("a", "b"))
        self.assertIn("candidates", result)
        self.assertIn("selected_candidate_id", result)
        self.assertIn("selected_candidate_utility", result)
        forbidden = {
            "observed_success",
            "observed_output_tokens",
            "verified_success",
            "outcome",
        }
        blob = repr(result)
        for token in forbidden:
            self.assertNotIn(token, blob)


class ValidationFailureTest(unittest.TestCase):
    def test_candidate_order_must_be_immutable_tuple(self) -> None:
        candidates = ({"candidate_id": "a", "predicted_success": 0.5, "predicted_output_tokens": 0},)
        with self.assertRaises(TypeError):
            ru.select_candidate(candidates, 1.0, candidate_order=["a"])  # type: ignore[arg-type]

    def test_candidate_order_ids_must_be_non_empty_unique(self) -> None:
        candidates = (
            {"candidate_id": "a", "predicted_success": 0.5, "predicted_output_tokens": 0},
            {"candidate_id": "b", "predicted_success": 0.6, "predicted_output_tokens": 0},
        )
        with self.assertRaises(ValueError):
            ru.select_candidate(candidates, 1.0, candidate_order=("", "b"))
        with self.assertRaises(ValueError):
            ru.select_candidate(candidates, 1.0, candidate_order=("a", "a"))

    def test_missing_and_extra_candidates_are_fail_closed(self) -> None:
        candidates = (
            {"candidate_id": "a", "predicted_success": 0.5, "predicted_output_tokens": 0},
            {"candidate_id": "b", "predicted_success": 0.6, "predicted_output_tokens": 0},
        )
        with self.assertRaises(ValueError):
            ru.select_candidate(candidates, 1.0, candidate_order=("a",))
        with self.assertRaises(ValueError):
            ru.select_candidate(candidates, 1.0, candidate_order=("a", "b", "c"))

    def test_duplicate_candidate_ids_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ru.select_candidate(
                (
                    {"candidate_id": "dup", "predicted_success": 0.5, "predicted_output_tokens": 0},
                    {"candidate_id": "dup", "predicted_success": 0.4, "predicted_output_tokens": 10},
                ),
                1.0,
                candidate_order=("dup",),
            )

    def test_non_finite_or_out_of_range_success_rejected(self) -> None:
        base = {"candidate_id": "a", "predicted_output_tokens": 10}
        with self.assertRaises(ValueError):
            ru.select_candidate(({**base, "predicted_success": -0.01},), 1.0, candidate_order=("a",))
        with self.assertRaises(ValueError):
            ru.select_candidate(({**base, "predicted_success": 1.01},), 1.0, candidate_order=("a",))
        with self.assertRaises(ValueError):
            ru.select_candidate(({**base, "predicted_success": float("nan")},), 1.0, candidate_order=("a",))
        with self.assertRaises(ValueError):
            ru.select_candidate(({**base, "predicted_success": float("inf")},), 1.0, candidate_order=("a",))
        with self.assertRaises(ValueError):
            ru.select_candidate(({**base, "predicted_success": True},), 1.0, candidate_order=("a",))

    def test_non_finite_or_negative_cost_rejected(self) -> None:
        base = {"candidate_id": "a", "predicted_success": 0.5}
        with self.assertRaises(ValueError):
            ru.select_candidate(({**base, "predicted_output_tokens": -1},), 1.0, candidate_order=("a",))
        with self.assertRaises(ValueError):
            ru.select_candidate(({**base, "predicted_output_tokens": float("nan")},), 1.0, candidate_order=("a",))
        with self.assertRaises(ValueError):
            ru.select_candidate(({**base, "predicted_output_tokens": float("inf")},), 1.0, candidate_order=("a",))
        with self.assertRaises(ValueError):
            ru.select_candidate(({**base, "predicted_output_tokens": True},), 1.0, candidate_order=("a",))

    def test_lambda_must_be_finite_and_non_negative(self) -> None:
        candidate = {"candidate_id": "a", "predicted_success": 0.5, "predicted_output_tokens": 10}
        with self.assertRaises(ValueError):
            ru.select_candidate((candidate,), -0.1, candidate_order=("a",))
        with self.assertRaises(ValueError):
            ru.select_candidate((candidate,), float("inf"), candidate_order=("a",))
        with self.assertRaises(ValueError):
            ru.select_candidate((candidate,), float("nan"), candidate_order=("a",))


class SyntheticSmokeMatrixTest(unittest.TestCase):
    def test_default_smoke_matrix_and_lambda_grid(self) -> None:
        report = ru.run_utility_scoring_smoke_matrix()
        self.assertTrue(report["passed"])
        self.assertEqual(report["lambda_grid"], list(ru.FROZEN_LAMBDA_GRID))
        self.assertEqual(report["primary_lambda"], ru.PRIMARY_LAMBDA)
        self.assertIn("dominance", report["cases"])
        self.assertIn("tradeoff", report["cases"])
        self.assertIn("tie_by_order", report["cases"])
        self.assertIn("tie_by_success", report["cases"])
        for name in ("dominance", "tradeoff", "tie_by_order", "tie_by_success"):
            self.assertEqual(len(report["cases"][name]), len(ru.FROZEN_LAMBDA_GRID))

        self.assertTrue(all(report["invalid_cases"].values()))

    def test_custom_grid_is_exercised(self) -> None:
        custom_grid = (0.0, 0.4, 1.6)
        report = ru.run_utility_scoring_smoke_matrix(lambda_grid=custom_grid)
        self.assertEqual(report["lambda_grid"], list(custom_grid))
        for name in ("dominance", "tradeoff", "tie_by_order", "tie_by_success"):
            self.assertEqual(len(report["cases"][name]), len(custom_grid))


class LambdaTradeoffTest(unittest.TestCase):
    def test_tradeoff_switches_with_lambda_grid(self) -> None:
        report = ru.run_utility_scoring_smoke_matrix()
        tradeoff = report["cases"]["tradeoff"]
        selected = [entry["selected_candidate_id"] for entry in tradeoff]
        # Expected: expensive-high-succ option only at lambda=0.0, then economy.
        self.assertEqual(selected[0], "latency")
        self.assertTrue(all(item == "economy" for item in selected[1:]))


if __name__ == "__main__":
    unittest.main()
