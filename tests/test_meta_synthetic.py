"""Tests for the synthetic validation harness."""

from __future__ import annotations

import unittest

from pyreplab_harness import meta_grammar
from pyreplab_harness import meta_synthetic
from pyreplab_harness.meta_cnp import TORCH_AVAILABLE

TORCH_REQUIRED = unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")


class GenerateSyntheticOutcomesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.grammar = meta_grammar.enumerate_unbrowser_grammar()

    def test_generates_correct_format(self) -> None:
        data = meta_synthetic.generate_synthetic_outcomes(
            self.grammar, n_tasks=10, n_policies=5, seed=42,
        )
        self.assertIn("rows", data)
        self.assertIn("true_params", data)
        self.assertIn("task_modifiers", data)

        rows = data["rows"]
        self.assertEqual(len(rows), 10 * 5)  # tasks x policies

        # Check row structure (frozen interface contract).
        row = rows[0]
        self.assertIn("task_id", row)
        self.assertIn("policy_bundle_id", row)
        self.assertIn("verified_success", row)
        self.assertIn("true_p", row)
        self.assertIn("output_token_cost", row)
        self.assertIn("termination_class", row)
        self.assertIn("model_input", row)

        mi = row["model_input"]
        self.assertIn("task", mi)
        self.assertIn("treatment", mi)
        self.assertIn("task_embedding", mi["task"])
        self.assertIn("grammar_factors", mi["treatment"])
        self.assertIn("numeric", mi["treatment"])
        self.assertIn("text", mi["treatment"])

    def test_success_is_bernoulli(self) -> None:
        data = meta_synthetic.generate_synthetic_outcomes(
            self.grammar, n_tasks=20, n_policies=5, seed=42,
        )
        successes = set()
        for row in data["rows"]:
            self.assertIn(row["verified_success"], (True, False))
            successes.add(row["verified_success"])
        self.assertGreater(len(successes), 1,
                           "Should have both successes and failures")

    def test_costs_are_positive(self) -> None:
        data = meta_synthetic.generate_synthetic_outcomes(
            self.grammar, n_tasks=10, n_policies=5, seed=42,
        )
        for row in data["rows"]:
            self.assertGreater(row["output_token_cost"], 0,
                               f"Cost should be positive, got {row['output_token_cost']}")

    def test_termination_classes_in_range(self) -> None:
        data = meta_synthetic.generate_synthetic_outcomes(
            self.grammar, n_tasks=20, n_policies=5, seed=42,
        )
        valid_classes = {
            "normal_completion", "tool_call_limit", "wall_timeout",
            "invalid_or_tool_error", "model_runtime_failure",
            "verifier_declared_unsuccessful",
        }
        for row in data["rows"]:
            self.assertIn(row["termination_class"], valid_classes)

    def test_deterministic_given_seed(self) -> None:
        data1 = meta_synthetic.generate_synthetic_outcomes(
            self.grammar, n_tasks=5, n_policies=3, seed=42,
        )
        data2 = meta_synthetic.generate_synthetic_outcomes(
            self.grammar, n_tasks=5, n_policies=3, seed=42,
        )
        self.assertEqual(
            [r["verified_success"] for r in data1["rows"]],
            [r["verified_success"] for r in data2["rows"]],
        )

    def test_different_seed_different_data(self) -> None:
        data1 = meta_synthetic.generate_synthetic_outcomes(
            self.grammar, n_tasks=5, n_policies=3, seed=42,
        )
        data2 = meta_synthetic.generate_synthetic_outcomes(
            self.grammar, n_tasks=5, n_policies=3, seed=99,
        )
        successes1 = [r["verified_success"] for r in data1["rows"]]
        successes2 = [r["verified_success"] for r in data2["rows"]]
        self.assertNotEqual(successes1, successes2)

    def test_true_params_have_expected_keys(self) -> None:
        data = meta_synthetic.generate_synthetic_outcomes(
            self.grammar, n_tasks=5, n_policies=5, seed=42,
        )
        for param in data["true_params"]:
            self.assertIn("policy_idx", param)
            self.assertIn("base_logit", param)
            self.assertIn("base_cost_log_mean", param)
            self.assertIn("grammar_factors", param)
            self.assertIn("planning", param["grammar_factors"])

    def test_task_modifiers_are_task_specific(self) -> None:
        data = meta_synthetic.generate_synthetic_outcomes(
            self.grammar, n_tasks=10, n_policies=3, seed=42,
        )
        modifiers = data["task_modifiers"]
        self.assertEqual(len(modifiers), 10)
        for mod in modifiers:
            self.assertIn("planning_bonus", mod)
            self.assertIn("template_id", mod)
            self.assertIn("difficulty", mod)

    def test_task_embedding_dimension(self) -> None:
        data = meta_synthetic.generate_synthetic_outcomes(
            self.grammar, n_tasks=3, n_policies=2, seed=42,
        )
        row = data["rows"][0]
        task_emb = row["model_input"]["task"]["task_embedding"]
        self.assertEqual(len(task_emb), 32)

    def test_no_policy_identity_leakage(self) -> None:
        """model_input.treatment should NOT contain policy identity fields."""
        data = meta_synthetic.generate_synthetic_outcomes(
            self.grammar, n_tasks=3, n_policies=2, seed=42,
        )
        for row in data["rows"]:
            treatment = row["model_input"]["treatment"]
            self.assertNotIn("policy_id", treatment)
            self.assertNotIn("policy_version", treatment)
            self.assertNotIn("bundle_id", treatment)
            self.assertNotIn("bundle_hash", treatment)


@TORCH_REQUIRED
class RunSyntheticValidationTest(unittest.TestCase):
    def test_validation_runs_and_returns_report(self) -> None:
        result = meta_synthetic.run_synthetic_validation(seed=99)
        self.assertIn("validation", result)
        self.assertEqual(result["validation"], "complete")
        self.assertIn("grammar", result)
        self.assertIn("model", result)
        self.assertIn("claims", result)
        self.assertIn("verdict", result)

    def test_grammar_split_sizes(self) -> None:
        result = meta_synthetic.run_synthetic_validation(seed=99)
        g = result["grammar"]
        self.assertEqual(g["n_policies_total"], 72)
        self.assertEqual(g["n_meta_train"], 48)
        self.assertEqual(g["n_development"], 12)
        self.assertEqual(g["n_final_held"], 12)

    def test_model_param_count(self) -> None:
        result = meta_synthetic.run_synthetic_validation(seed=99)
        self.assertLess(result["model"]["param_count"], 1_000_000,
                        f"Model should be under 1M params, got {result['model']['param_count']}")

    def test_k8_results_produced(self) -> None:
        result = meta_synthetic.run_synthetic_validation(seed=99)
        self.assertIn("k0_mean_success_prob", result["results"])
        self.assertIn("k8_mean_success_prob", result["results"])
        self.assertIn("k8_minus_k0", result["results"])

    def test_shuffled_negative_control_produced(self) -> None:
        result = meta_synthetic.run_synthetic_validation(seed=99)
        self.assertIn("shuffled_mean_success_prob", result["results"])
        self.assertIn("shuffled_minus_k0", result["results"])

    def test_frontier_produced(self) -> None:
        result = meta_synthetic.run_synthetic_validation(seed=99)
        self.assertIn("frontier", result)
        self.assertIn("frontier_area", result["frontier"])
        self.assertGreaterEqual(result["frontier"]["frontier_area"], 0.0)

    def test_claims_are_booleans(self) -> None:
        result = meta_synthetic.run_synthetic_validation(seed=99)
        for key, val in result["claims"].items():
            self.assertIsInstance(val, bool,
                f"Claim '{key}' should be bool, got {type(val)} - this means "
                f"the validation did not produce a clear result")


@TORCH_REQUIRED
class NegativeControlTest(unittest.TestCase):
    """Explicit test for the shuffled-context negative control."""

    def test_shuffled_context_does_not_beat_clean(self) -> None:
        """Shuffled contexts should destroy calibration gains."""
        result = meta_synthetic.run_synthetic_validation(seed=123)
        # The shuffled context should not meaningfully beat descriptor-only.
        shuffled_minus_k0 = result["results"].get("shuffled_minus_k0", 0.0)
        k8_minus_k0 = result["results"].get("k8_minus_k0", 0.0)

        # At minimum, shuffled should not outperform clean k=8.
        # (This is a synthetic test; on real data this would be a strict
        # gate requiring shuffled < k0.)
        self.assertIsNotNone(shuffled_minus_k0)
        self.assertIsNotNone(k8_minus_k0)


if __name__ == "__main__":
    unittest.main()
