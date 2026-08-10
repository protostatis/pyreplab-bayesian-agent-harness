from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pyreplab_harness import outcome_model as om
from pyreplab_harness.outcome_model_smoke import (
    CORRECT_POLICY_ID,
    SMOKE_TASK_PROMPT,
    WRONG_POLICY_ID,
    build_smoke_rows,
    proposed_treatments,
    run_smoke,
)


class OutcomeModelSmokeFixtureTest(unittest.TestCase):
    def test_complete_panels_are_split_by_whole_task(self) -> None:
        rows, registry = build_smoke_rows(SMOKE_TASK_PROMPT)
        self.assertEqual(len(rows), 80)
        self.assertEqual(len(registry), 2)
        by_task: dict[str, list[dict]] = {}
        for row in rows:
            by_task.setdefault(row["task_id"], []).append(row)
        self.assertEqual(len(by_task), 40)
        for panel in by_task.values():
            self.assertEqual({row["split"] for row in panel}, {panel[0]["split"]})
            self.assertEqual(
                {row["model_input"]["policy_id"] for row in panel},
                {CORRECT_POLICY_ID, WRONG_POLICY_ID},
            )
            labels = {row["model_input"]["policy_id"]: row["verified_success"] for row in panel}
            self.assertTrue(labels[CORRECT_POLICY_ID])
            self.assertFalse(labels[WRONG_POLICY_ID])

    def test_order_collision_has_the_same_token_multiset(self) -> None:
        candidates = {treatment.id: treatment for treatment in proposed_treatments()}
        first = om.tokenize_text(candidates["candidate-order-h1-not-p"].system_prompt)
        second = om.tokenize_text(candidates["candidate-order-p-not-h1"].system_prompt)
        self.assertEqual(sorted(first), sorted(second))


@unittest.skipUnless(om.TORCH_AVAILABLE, "PyTorch is not installed")
class OutcomeModelSmokeIntegrationTest(unittest.TestCase):
    def test_trains_and_reports_the_unseen_descriptor_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = run_smoke(SMOKE_TASK_PROMPT, Path(directory) / "output")
        self.assertTrue(result["synthetic_only"])
        self.assertTrue(result["ranking_matches_synthetic_labels"])
        self.assertEqual(result["observed_top_policy"], CORRECT_POLICY_ID)
        self.assertEqual(len(result["counterfactuals"]), 2)
        # The two training bundles are cleanly separable, but unseen IDs and
        # bundle IDs collapse to UNK. The current tiny synthetic corpus is
        # intentionally insufficient for substantial descriptor generalization.
        self.assertEqual(result["descriptor_probe"]["status"], "inconclusive")
        self.assertGreater(result["descriptor_probe"]["exact_clone_margin"], 0.0)
        self.assertLess(result["descriptor_probe"]["exact_clone_margin"], 0.05)
        self.assertGreater(result["descriptor_probe"]["paraphrase_margin"], 0.0)
        self.assertLess(result["descriptor_probe"]["paraphrase_margin"], 0.05)
        self.assertEqual(result["descriptor_probe"]["order_collision_gap"], 0.0)
        self.assertEqual(len(result["candidate_ranking"]), 6)
        for encoding in result["descriptor_probe"]["identity_encoding"].values():
            self.assertTrue(encoding["policy_id_is_unk"])
            self.assertTrue(encoding["bundle_id_is_unk"])
        for item in result["counterfactuals"]:
            self.assertGreaterEqual(item["mean"], 0.0)
            self.assertLessEqual(item["mean"], 1.0)
            self.assertGreaterEqual(item["std"], 0.0)


if __name__ == "__main__":
    unittest.main()
