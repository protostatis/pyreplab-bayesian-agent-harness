from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pyreplab_harness import repeated_outcome_smoke as ros


def _ordered_trial_rows(
    task_name: str,
    arm: str,
    rewards_by_position: list[int],
    seed: int,
    model: str,
) -> list[dict[str, object]]:
    trial_names = [f"t{idx}" for idx in range(len(rewards_by_position))]
    ordered = sorted(
        trial_names,
        key=lambda trial_name: ros.trial_hash(seed, task_name, arm, trial_name),
    )
    return [
        {
            "task_name": task_name,
            "agent": arm,
            "model": model,
            "reward": rewards_by_position[trial_names.index(name)],
            "trial_name": name,
        }
        for name in ordered
    ]


class RepeatedOutcomeSmokeTest(unittest.TestCase):
    arm_a = "openhands"
    arm_b = "terminus-2"
    model = "model"

    def test_stable_heterogeneity_reports_positive_lift(self) -> None:
        seed = 20260811
        rows: list[dict[str, object]] = []
        for index in range(20):
            task = f"task-{index}"
            if index % 2:
                rewards_a, rewards_b = [1, 1, 1], [0, 0, 0]
            else:
                rewards_a, rewards_b = [0, 0, 0], [1, 1, 1]
            rows.extend(
                _ordered_trial_rows(task, self.arm_a, rewards_a, seed, self.model)
            )
            rows.extend(
                _ordered_trial_rows(task, self.arm_b, rewards_b, seed, self.model)
            )

        repeats = 3
        with tempfile.TemporaryDirectory() as directory:
            inp = Path(directory) / "input.jsonl"
            inp.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
            out = Path(directory) / "report.json"
            report = ros.run_smoke(
                inp,
                out,
                arm_a=self.arm_a,
                arm_b=self.arm_b,
                model=self.model,
                repeats=repeats,
                seed=seed,
                bootstrap_trials=100,
            )
            self.assertIn("learning_curve", report)
            first = report["learning_curve"]["1"]["lift"]["selector_minus_fixed"]
            second = report["learning_curve"]["2"]["lift"]["selector_minus_fixed"]
            self.assertGreater(first, 0.0)
            self.assertGreater(second, 0.0)
            self.assertGreaterEqual(report["learning_curve"]["1"]["allocation"]["tie_rate"], 0.0)
            self.assertEqual(report["decision"]["verdict"], "supports_repeats")

    def test_fixed_arm_dominance_gives_zero_lift(self) -> None:
        seed = 7
        rows: list[dict[str, object]] = []
        for task in ["task-a", "task-b", "task-c"]:
            rows.extend(_ordered_trial_rows(task, self.arm_a, [1, 1, 1], seed, self.model))
            rows.extend(_ordered_trial_rows(task, self.arm_b, [0, 0, 0], seed, self.model))

        with tempfile.TemporaryDirectory() as directory:
            inp = Path(directory) / "input.jsonl"
            inp.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
            out = Path(directory) / "report.json"
            report = ros.run_smoke(
                inp,
                out,
                arm_a=self.arm_a,
                arm_b=self.arm_b,
                model=self.model,
                repeats=3,
                seed=seed,
                bootstrap_trials=10,
            )
            for k in ["1", "2"]:
                lift = report["learning_curve"][k]["lift"]["selector_minus_fixed"]
                self.assertAlmostEqual(lift, 0.0, places=10)
            self.assertEqual(
                report["decision"]["verdict"], "does_not_support_repeats"
            )

    def test_fractional_ties_recorded(self) -> None:
        seed = 11
        rows: list[dict[str, object]] = []
        rows.extend(_ordered_trial_rows("task-1", self.arm_a, [1, 0, 1], seed, self.model))
        rows.extend(_ordered_trial_rows("task-1", self.arm_b, [1, 0, 1], seed, self.model))
        rows.extend(_ordered_trial_rows("task-2", self.arm_a, [0, 1, 0], seed, self.model))
        rows.extend(_ordered_trial_rows("task-2", self.arm_b, [0, 1, 0], seed, self.model))

        with tempfile.TemporaryDirectory() as directory:
            inp = Path(directory) / "input.jsonl"
            inp.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
            out = Path(directory) / "report.json"
            report = ros.run_smoke(
                inp,
                out,
                arm_a=self.arm_a,
                arm_b=self.arm_b,
                model=self.model,
                repeats=3,
                seed=seed,
                bootstrap_trials=5,
            )
            tie_rates = [
                report["learning_curve"][key]["allocation"]["tie_rate"] for key in report["learning_curve"]
            ]
            self.assertTrue(any(tie_rate > 0 for tie_rate in tie_rates))

    def test_duplicate_consistency_and_conflict(self) -> None:
        seed = 13
        base_rows = _ordered_trial_rows("task-1", self.arm_a, [1, 0, 1], seed, self.model)
        base_rows.extend(_ordered_trial_rows("task-1", self.arm_b, [0, 1, 0], seed, self.model))

        # Duplicate row with matching reward is tolerated.
        rows_consistent = base_rows + [dict(base_rows[0]), dict(base_rows[1])]
        filtered = ros.filter_rows(rows_consistent, model=self.model, arm_a=self.arm_a, arm_b=self.arm_b)
        deduped, dup_count = ros.deduplicate_trials(filtered, arm_a=self.arm_a, arm_b=self.arm_b)
        self.assertEqual(dup_count, 2)
        self.assertEqual(len(deduped), len(base_rows))

        # Conflicting duplicate reward is a hard error.
        conflict_row = dict(base_rows[0])
        conflict_row["reward"] = 0 if base_rows[0]["reward"] == 1 else 1
        with self.assertRaises(ValueError):
            ros.deduplicate_trials(base_rows + [conflict_row], arm_a=self.arm_a, arm_b=self.arm_b)

    def test_incomplete_tasks_are_excluded(self) -> None:
        seed = 17
        rows: list[dict[str, object]] = []
        # Complete task.
        rows.extend(_ordered_trial_rows("task-complete", self.arm_a, [1, 1, 1], seed, self.model))
        rows.extend(_ordered_trial_rows("task-complete", self.arm_b, [0, 0, 0], seed, self.model))

        # Incomplete task (only 2 repeats for arm b).
        rows.extend(_ordered_trial_rows("task-incomplete", self.arm_a, [1, 1], seed, self.model))
        rows.extend(_ordered_trial_rows("task-incomplete", self.arm_b, [0, 0], seed, self.model))

        with tempfile.TemporaryDirectory() as directory:
            inp = Path(directory) / "input.jsonl"
            inp.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
            out = Path(directory) / "report.json"
            report = ros.run_smoke(
                inp,
                out,
                arm_a=self.arm_a,
                arm_b=self.arm_b,
                model=self.model,
                repeats=3,
                seed=seed,
                bootstrap_trials=3,
            )
            counts = report["counts"]
            self.assertEqual(counts["completeness"]["tasks_retained"], 1)
            self.assertEqual(counts["completeness"]["tasks_excluded"], 1)
            self.assertEqual(counts["exclusion"]["wrong_repeat_count"], 1)

    def test_deterministic_output_is_repeatable(self) -> None:
        seed = 19
        rows: list[dict[str, object]] = []
        rows.extend(_ordered_trial_rows("task-1", self.arm_a, [1, 1, 0], seed, self.model))
        rows.extend(_ordered_trial_rows("task-1", self.arm_b, [0, 1, 1], seed, self.model))
        rows.extend(_ordered_trial_rows("task-2", self.arm_a, [1, 0, 1], seed, self.model))
        rows.extend(_ordered_trial_rows("task-2", self.arm_b, [0, 0, 1], seed, self.model))

        with tempfile.TemporaryDirectory() as directory:
            inp = Path(directory) / "input.jsonl"
            inp.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
            out1 = Path(directory) / "report1.json"
            out2 = Path(directory) / "report2.json"
            report1 = ros.run_smoke(
                inp,
                out1,
                arm_a=self.arm_a,
                arm_b=self.arm_b,
                model=self.model,
                repeats=3,
                seed=seed,
                bootstrap_trials=6,
            )
            report2 = ros.run_smoke(
                inp,
                out2,
                arm_a=self.arm_a,
                arm_b=self.arm_b,
                model=self.model,
                repeats=3,
                seed=seed,
                bootstrap_trials=6,
            )
            self.assertEqual(report1["learning_curve"], report2["learning_curve"])
            self.assertEqual(report1["counts"], report2["counts"])

    def test_position_combination_counts_are_squared_for_independent_arms(self) -> None:
        """For R=3, k in {1,2}: C(3,1)^2 = 9 and C(3,2)^2 = 9 (not 3)."""
        seed = 42
        rows: list[dict[str, object]] = []
        rows.extend(_ordered_trial_rows("task-1", self.arm_a, [1, 0, 0], seed, self.model))
        rows.extend(_ordered_trial_rows("task-1", self.arm_b, [0, 1, 0], seed, self.model))

        filtered = ros.filter_rows(rows, model=self.model, arm_a=self.arm_a, arm_b=self.arm_b)
        deduped, _dup = ros.deduplicate_trials(filtered, arm_a=self.arm_a, arm_b=self.arm_b)
        panels, _ex = ros.build_complete_task_panels(
            deduped, repeats=3, seed=seed, arm_a=self.arm_a, arm_b=self.arm_b,
        )

        for k in (1, 2):
            avg, splits = ros.analyze_k(panels, repeats=3, k=k)
            self.assertEqual(avg["position_combinations"], 9,
                             f"Expected 9 (C(3,{k})^2) position combos for k={k}")
            self.assertEqual(len(splits), 9,
                             f"Expected 9 split results for k={k}")

            # Verify each split has independent arm-A and arm-B calibration positions
            for split_result in splits:
                self.assertIn("calibration_positions_a", split_result)
                self.assertIn("calibration_positions_b", split_result)

    def test_permuted_arm_outcomes_yield_same_averaged_metrics(self) -> None:
        """Independently permuting one arm's ordered outcomes must not change
        averaged learning-curve point metrics (Cartesian product is invariant
        to within-arm position permutation)."""
        seed = 99
        rewards_a_orig = [1, 1, 0, 0, 0]  # R=5
        rewards_b = [0, 1, 0, 0, 1]        # R=5

        rows_orig: list[dict[str, object]] = []
        rows_orig.extend(_ordered_trial_rows("task", self.arm_a, rewards_a_orig, seed, self.model))
        rows_orig.extend(_ordered_trial_rows("task", self.arm_b, rewards_b, seed, self.model))

        # Permuted arm A (cyclic shift of positions)
        rewards_a_perm = [0, 0, 0, 1, 1]  # same multiset

        rows_perm: list[dict[str, object]] = []
        rows_perm.extend(_ordered_trial_rows("task", self.arm_a, rewards_a_perm, seed, self.model))
        rows_perm.extend(_ordered_trial_rows("task", self.arm_b, rewards_b, seed, self.model))

        filtered_orig = ros.filter_rows(rows_orig, model=self.model, arm_a=self.arm_a, arm_b=self.arm_b)
        deduped_orig, _ = ros.deduplicate_trials(filtered_orig, arm_a=self.arm_a, arm_b=self.arm_b)
        panels_orig, _ = ros.build_complete_task_panels(
            deduped_orig, repeats=5, seed=seed, arm_a=self.arm_a, arm_b=self.arm_b,
        )

        filtered_perm = ros.filter_rows(rows_perm, model=self.model, arm_a=self.arm_a, arm_b=self.arm_b)
        deduped_perm, _ = ros.deduplicate_trials(filtered_perm, arm_a=self.arm_a, arm_b=self.arm_b)
        panels_perm, _ = ros.build_complete_task_panels(
            deduped_perm, repeats=5, seed=seed, arm_a=self.arm_a, arm_b=self.arm_b,
        )

        for k in range(1, 5):
            avg_orig, _ = ros.analyze_k(panels_orig, repeats=5, k=k)
            avg_perm, _ = ros.analyze_k(panels_perm, repeats=5, k=k)

            # Averaged supervised metrics must be identical
            self.assertEqual(avg_orig["position_combinations"], avg_perm["position_combinations"])
            self.assertAlmostEqual(
                avg_orig["selector"]["held_success_rate"],
                avg_perm["selector"]["held_success_rate"], places=12,
            )
            self.assertAlmostEqual(
                avg_orig["calibrated_fixed"]["held_success_rate"],
                avg_perm["calibrated_fixed"]["held_success_rate"], places=12,
            )
            self.assertAlmostEqual(
                avg_orig["lift"]["selector_minus_fixed"],
                avg_perm["lift"]["selector_minus_fixed"], places=12,
            )
            self.assertAlmostEqual(
                avg_orig["always_arm_a"]["held_success_rate"],
                avg_perm["always_arm_a"]["held_success_rate"], places=12,
            )


if __name__ == "__main__":
    unittest.main()
