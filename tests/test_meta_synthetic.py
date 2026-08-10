"""Tests for the synthetic validation harness."""

from __future__ import annotations

import math
import subprocess
import sys
import unittest

from pyreplab_harness import meta_grammar
from pyreplab_harness import meta_synthetic
from pyreplab_harness.meta_cnp import TORCH_AVAILABLE

TORCH_REQUIRED = unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")


# ---------------------------------------------------------------------------
# Non-torch tests: latent sharing, panel structure, output format
# ---------------------------------------------------------------------------


class LatentResidualSharingTest(unittest.TestCase):
    """Stable policy latent via SHA-256, shared across pools."""

    def test_same_bundle_same_latent_seed_gives_same_residual(self) -> None:
        a1, a2 = meta_synthetic._compute_policy_latent("bundle-abc", latent_seed=42)
        b1, b2 = meta_synthetic._compute_policy_latent("bundle-abc", latent_seed=42)
        self.assertEqual(a1, b1)
        self.assertEqual(a2, b2)

    def test_different_bundle_gives_different_residual(self) -> None:
        a1, a2 = meta_synthetic._compute_policy_latent("bundle-A", latent_seed=42)
        b1, b2 = meta_synthetic._compute_policy_latent("bundle-B", latent_seed=42)
        self.assertNotEqual(a1, b1)

    def test_different_latent_seed_gives_different_residual(self) -> None:
        a1, a2 = meta_synthetic._compute_policy_latent("bundle-abc", latent_seed=42)
        b1, b2 = meta_synthetic._compute_policy_latent("bundle-abc", latent_seed=99)
        self.assertNotEqual(a1, b1)

    def test_residual_in_expected_range(self) -> None:
        for i in range(200):
            a, b = meta_synthetic._compute_policy_latent(f"bundle-{i}", latent_seed=meta_synthetic._LATENT_SEED)
            self.assertGreaterEqual(a, -1.0)
            self.assertLessEqual(a, 1.0)
            self.assertGreaterEqual(b, -1.0)
            self.assertLessEqual(b, 1.0)

    def test_latent_applied_to_outcomes(self) -> None:
        """DGP with latent produces outcomes NOT explainable by model_input alone."""
        grammar = meta_grammar.enumerate_unbrowser_grammar()
        data_with = meta_synthetic.generate_synthetic_outcomes(
            grammar, n_tasks=20, n_policies=5, seed=42, latent_seed=meta_synthetic._LATENT_SEED,
        )
        data_without = meta_synthetic.generate_synthetic_outcomes(
            grammar, n_tasks=20, n_policies=5, seed=42, latent_seed=None,
        )
        # Same seed, same model_input, but different outcomes due to latent.
        for r1, r2 in zip(data_with["rows"], data_without["rows"]):
            if r1["true_p"] != r2["true_p"]:
                # At least one row differs in true_p, confirming latent effect.
                return
        self.fail("Latent residual had no effect on true_p")

    def test_latent_not_in_model_input(self) -> None:
        """model_input.task must NOT contain latent residual leakage."""
        grammar = meta_grammar.enumerate_unbrowser_grammar()
        data = meta_synthetic.generate_synthetic_outcomes(
            grammar, n_tasks=5, n_policies=3, seed=42, latent_seed=meta_synthetic._LATENT_SEED,
        )
        for row in data["rows"]:
            task = row["model_input"]["task"]
            self.assertNotIn("latent", task)
            self.assertNotIn("residual", task)
            self.assertNotIn("policy_latent", task)


class PanelStructureTest(unittest.TestCase):
    """Calibration/target split: disjoint, frozen, nested prefixes."""

    def setUp(self) -> None:
        self.grammar = meta_grammar.enumerate_unbrowser_grammar()
        self.policies = self.grammar[:12]

    def test_split_creates_cal_and_target_roles(self) -> None:
        data = meta_synthetic.generate_synthetic_outcomes(
            self.policies, n_tasks=32, n_policies=12, seed=42,
        )
        cal, tgt = meta_synthetic._split_calibration_target(
            data["rows"], 32, 12, 16,
        )
        self.assertEqual(len(cal), 16 * 12)
        self.assertEqual(len(tgt), 16 * 12)
        for r in cal:
            self.assertEqual(r["task_pool_role"], "calibration")
        for r in tgt:
            self.assertEqual(r["task_pool_role"], "target")

    def test_cal_target_disjoint_task_ids(self) -> None:
        data = meta_synthetic.generate_synthetic_outcomes(
            self.policies, n_tasks=32, n_policies=12, seed=42,
        )
        cal, tgt = meta_synthetic._split_calibration_target(
            data["rows"], 32, 12, 16,
        )
        cal_ids = {r["task_id"] for r in cal}
        tgt_ids = {r["task_id"] for r in tgt}
        self.assertTrue(len(cal_ids & tgt_ids) == 0,
                        "Calibration and target task IDs overlap")

    def test_cal_rows_ordered_by_task_then_policy(self) -> None:
        """cal_rows indexed by (cal_task_idx, policy_idx) in that order."""
        data = meta_synthetic.generate_synthetic_outcomes(
            self.policies, n_tasks=32, n_policies=12, seed=42,
        )
        cal, _ = meta_synthetic._split_calibration_target(
            data["rows"], 32, 12, 16,
        )
        # First 12 rows: cal_task_0 x policy_0..11
        for p in range(12):
            self.assertEqual(cal[p]["policy_idx"], p)
            self.assertEqual(cal[p]["task_pool_index"], 0)
        # Next 12 rows: cal_task_1 x policy_0..11
        for p in range(12):
            self.assertEqual(cal[12 + p]["policy_idx"], p)
            self.assertEqual(cal[12 + p]["task_pool_index"], 1)

    def test_build_derangement_no_fixed_points(self) -> None:
        d = meta_synthetic._build_derangement(12, seed=99)
        self.assertEqual(len(d), 12)
        for i in range(12):
            self.assertNotEqual(d[i], i, f"Fixed point at {i} -> {d[i]}")
        # Verify it's a permutation (bijection).
        self.assertEqual(set(d.keys()), set(range(12)))
        self.assertEqual(set(d.values()), set(range(12)))

    def test_task_pool_prefixes_make_independent_pools_disjoint(self) -> None:
        dev = meta_synthetic.generate_synthetic_outcomes(
            self.policies,
            n_tasks=2,
            n_policies=12,
            seed=42,
            task_id_prefix="dev",
        )
        final = meta_synthetic.generate_synthetic_outcomes(
            self.policies,
            n_tasks=2,
            n_policies=12,
            seed=43,
            task_id_prefix="final",
        )
        self.assertTrue(meta_synthetic._all_task_pools_disjoint(
            dev["rows"], final["rows"],
        ))


class SyntheticMetricTest(unittest.TestCase):
    def test_ranking_accuracy_uses_global_policy_means(self) -> None:
        predictions = [
            [0.9, 0.8, 0.1],
            [0.1, 0.2, 0.9],
            [0.9, 0.4, 0.1],
        ]
        true_probabilities = [
            [0.8, 0.7, 0.1],
            [0.8, 0.7, 0.1],
            [0.8, 0.7, 0.1],
        ]
        # Global predicted means preserve policy order 0 > 1 > 2 even though
        # the second task's within-task ordering is reversed.
        accuracy = meta_synthetic._global_pairwise_ranking_accuracy(
            predictions, true_probabilities,
        )
        self.assertEqual(accuracy, 1.0)


class GenerateSyntheticOutcomesTest(unittest.TestCase):
    """Output format, Bernoulli costs, determinism, no identity leakage."""

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
        self.assertEqual(len(rows), 10 * 5)
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
        successes = {row["verified_success"] for row in data["rows"]}
        self.assertGreater(len(successes), 1, "Should have both successes and failures")

    def test_costs_are_positive(self) -> None:
        data = meta_synthetic.generate_synthetic_outcomes(
            self.grammar, n_tasks=10, n_policies=5, seed=42,
        )
        for row in data["rows"]:
            self.assertGreater(row["output_token_cost"], 0)

    def test_termination_classes_in_range(self) -> None:
        data = meta_synthetic.generate_synthetic_outcomes(
            self.grammar, n_tasks=20, n_policies=5, seed=42,
        )
        valid = {"normal_completion", "tool_call_limit", "wall_timeout",
                 "invalid_or_tool_error", "model_runtime_failure",
                 "verifier_declared_unsuccessful"}
        for row in data["rows"]:
            self.assertIn(row["termination_class"], valid)

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
        data = meta_synthetic.generate_synthetic_outcomes(
            self.grammar, n_tasks=3, n_policies=2, seed=42,
        )
        for row in data["rows"]:
            treatment = row["model_input"]["treatment"]
            self.assertNotIn("policy_id", treatment)
            self.assertNotIn("policy_version", treatment)
            self.assertNotIn("bundle_id", treatment)
            self.assertNotIn("bundle_hash", treatment)

    def test_task_embed_reproducible_across_processes(self) -> None:
        import json as _json

        modifier = {
            "planning_bonus": 0.15,
            "observation_bonus": -0.22,
            "verification_bonus": 0.0,
            "recovery_bonus": 0.08,
            "cost_shift": -0.1,
            "template_id": "extraction",
            "difficulty": "medium",
        }
        emb1 = meta_synthetic._make_task_embed(0, modifier, seed=42, dim=32)
        modifier_json = _json.dumps(modifier, sort_keys=True)
        code = (
            "import json\n"
            "from pyreplab_harness.meta_synthetic import _make_task_embed\n"
            "modifier = json.loads('''" + modifier_json + "''')\n"
            "emb = _make_task_embed(0, modifier, seed=42, dim=32)\n"
            "print(json.dumps(emb))\n"
        )
        env = {**__import__("os").environ, "PYTHONHASHSEED": "999"}
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=15, env=env,
        )
        self.assertEqual(proc.returncode, 0, f"Subprocess failed: {proc.stderr}")
        emb2 = _json.loads(proc.stdout.strip())
        self.assertEqual(len(emb1), len(emb2))
        self.assertEqual(emb1, emb2)


# ---------------------------------------------------------------------------
# Torch tests: metric fields, best-state/model-mode, cached runs
# ---------------------------------------------------------------------------


@TORCH_REQUIRED
class RunSyntheticValidationTest(unittest.TestCase):
    """Full validation pipeline tests with result caching."""

    _cached: dict[int, dict] = {}

    @classmethod
    def _get_result(cls, seed: int = 99) -> dict:
        if seed not in cls._cached:
            cls._cached[seed] = meta_synthetic.run_synthetic_validation(seed=seed)
        return cls._cached[seed]

    def test_validation_runs_and_returns_report(self) -> None:
        result = self._get_result(99)
        self.assertIn("validation", result)
        self.assertEqual(result["validation"], "complete")
        self.assertIn("grammar", result)
        self.assertIn("model", result)
        self.assertIn("claims", result)
        self.assertIn("verdict", result)
        self.assertIn("protocol_meta", result)

    def test_grammar_split_sizes(self) -> None:
        result = self._get_result(99)
        g = result["grammar"]
        self.assertEqual(g["n_policies_total"], 72)
        self.assertEqual(g["n_meta_train"], 48)
        self.assertEqual(g["n_development"], 12)
        self.assertEqual(g["n_final_held"], 12)

    def test_model_param_count(self) -> None:
        result = self._get_result(99)
        self.assertLess(result["model"]["param_count"], 1_000_000)

    def test_all_k_values_evaluated(self) -> None:
        result = self._get_result(99)
        by_k = result["results"]["by_k"]
        for k_str in ["0", "4", "8", "16"]:
            self.assertIn(k_str, by_k)
            entry = by_k[k_str]
            for field in ("log_loss_binary", "brier", "mean_predicted_prob", "ranking_accuracy"):
                self.assertIn(field, entry)
                self.assertIsInstance(entry[field], (int, float))

    def test_primary_k8_has_metric_fields(self) -> None:
        result = self._get_result(99)
        primary = result["results"]["primary"]
        self.assertEqual(primary["k"], 8)
        for field in ("log_loss_binary", "brier", "mean_predicted_prob", "ranking_accuracy"):
            self.assertIn(field, primary)
            self.assertIsInstance(primary[field], (int, float))
            self.assertTrue(math.isfinite(primary[field]),
                            f"primary.{field} = {primary[field]} is not finite")

    def test_dev_loss_tracked(self) -> None:
        result = self._get_result(99)
        model = result["model"]
        self.assertIn("best_dev_loss", model)
        self.assertIn("last_dev_loss", model)
        self.assertIn("best_epoch", model)
        self.assertEqual(model["dev_selection_metric"], "true_p_cross_entropy_k8")
        self.assertTrue(model["best_dev_loss"] < float("inf"))

    def test_best_epoch_is_valid(self) -> None:
        result = self._get_result(99)
        model = result["model"]
        self.assertGreaterEqual(model["best_epoch"], 0)
        self.assertLess(model["best_epoch"], model["n_epochs_trained"])

    def test_deranged_negative_control_produced(self) -> None:
        result = self._get_result(99)
        deranged = result["results"]["deranged_k8"]
        self.assertIn("log_loss_binary", deranged)
        self.assertIn("brier", deranged)
        self.assertIn("ranking_accuracy", deranged)
        self.assertIn("delta_log_loss_vs_k0", deranged)

    def test_k8_minus_k0_deltas_reported(self) -> None:
        result = self._get_result(99)
        deltas = result["results"]["k8_minus_k0"]
        for key in ("delta_log_loss", "delta_brier", "delta_ranking_accuracy",
                     "delta_true_p_ce", "k8_improves_over_k0"):
            self.assertIn(key, deltas)
        self.assertIsInstance(deltas["k8_improves_over_k0"], bool)

    def test_claims_are_booleans(self) -> None:
        result = self._get_result(99)
        for key, val in result["claims"].items():
            self.assertIsInstance(val, bool,
                f"Claim '{key}' should be bool, got {type(val)}")

    def test_frontier_produced(self) -> None:
        result = self._get_result(99)
        self.assertIn("frontier", result)
        self.assertIn("frontier_area", result["frontier"])
        self.assertGreaterEqual(result["frontier"]["frontier_area"], 0.0)

    def test_frontier_policy_task_counts(self) -> None:
        result = self._get_result(99)
        frontier = result["frontier"]
        self.assertIn("n_tasks", frontier)
        self.assertIn("n_policies", frontier)
        self.assertEqual(frontier["n_policies"], 12)
        # 64 target tasks for final held-out.
        self.assertEqual(frontier["n_tasks"], 64)

    def test_protocol_meta_task_pool_disjoint(self) -> None:
        result = self._get_result(99)
        pm = result["protocol_meta"]
        self.assertTrue(pm["cal_target_disjoint_dev"])
        self.assertTrue(pm["cal_target_disjoint_final"])
        self.assertEqual(pm["n_cal_tasks"], 16)
        self.assertEqual(len(pm["final_cal_task_ids"]), 16)
        self.assertEqual(len(pm["final_target_task_ids"]), 64)
        self.assertTrue(pm["all_task_pools_globally_disjoint"])
        self.assertEqual(pm["success_training_target"], "synthetic_true_probability")

    def test_protocol_meta_derangement_is_permutation(self) -> None:
        result = self._get_result(99)
        d = result["protocol_meta"]["derangement_mapping"]
        keys = {int(k) for k in d}
        vals = {int(v) for v in d.values()}
        self.assertEqual(keys, set(range(12)))
        self.assertEqual(vals, set(range(12)))
        for k, v in d.items():
            self.assertNotEqual(int(k), int(v), f"derangement fixed point at {k}")


@TORCH_REQUIRED
class NegativeControlDerangementTest(unittest.TestCase):
    """Derangement must remove the clean calibration improvement."""

    _cached: dict[int, dict] = {}

    @classmethod
    def _get_result(cls, seed: int = 123) -> dict:
        if seed not in cls._cached:
            cls._cached[seed] = meta_synthetic.run_synthetic_validation(seed=seed)
        return cls._cached[seed]

    def test_deranged_context_removes_calibration_gain(self) -> None:
        result = self._get_result(123)
        deranged = result["results"]["deranged_k8"]
        k0 = result["results"]["by_k"]["0"]
        # Deranged k=8 should NOT show log-loss improvement over k=0.
        self.assertTrue(
            deranged["log_loss_binary"] >= k0["log_loss_binary"]
            or deranged["ranking_accuracy"] < k0["ranking_accuracy"],
            f"Deranged k=8 ({deranged['log_loss_binary']:.4f}) unexpectedly "
            f"improved over k=0 ({k0['log_loss_binary']:.4f})",
        )


if __name__ == "__main__":
    unittest.main()
