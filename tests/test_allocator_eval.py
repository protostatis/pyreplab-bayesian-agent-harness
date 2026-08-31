from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from pyreplab_harness import allocator_eval as ae
from pyreplab_harness import dashboard as db
from pyreplab_harness import outcome_model as om
from pyreplab_harness.treatments import (
    TreatmentRegistry,
    TreatmentSpec,
    generate_treatments,
    treatment_model_input_descriptor,
)

TORCH_REQUIRED = unittest.skipUnless(om.TORCH_AVAILABLE, "PyTorch is not installed")


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------
def make_row(
    task_id: str,
    policy: str,
    split: str,
    success: bool,
    *,
    attempt_id: str | None = None,
    tokens: int | None = None,
    usage: dict | None = None,
    tool_calls: int = 2,
    messages: int = 1,
    policy_version: str = "1",
    family: str = "artifact",
    difficulty: str = "easy",
) -> dict:
    if usage is None:
        usage = (
            {"input": tokens // 3, "output": tokens - tokens // 3, "total_tokens": tokens}
            if tokens is not None
            else {"input": 100, "output": 50, "total_tokens": 150}
        )
    attempt_id = attempt_id or f"att-{task_id}-{policy}"
    return {
        "task_id": task_id,
        "family": family,
        "template_id": "template-v1",
        "generator_version": "1",
        "seed": 1,
        "difficulty": difficulty,
        "prompt": f"prompt {task_id}",
        "contract": ["produce result.json"],
        "public_metadata": {"rows": 5},
        "attempt_id": attempt_id,
        "policy_id": policy,
        "policy_version": policy_version,
        "split": split,
        "verified_success": success,
        "failure_code": None if success else "boom",
        "verifier_id": "verifier-v1",
        "verifier_version": "1",
        "usage": usage,
        "assistant_message_count": messages,
        "tool_call_count": tool_calls,
        "final_text_length": 12,
        "model_input": {
            "text": f"Predecision text for {task_id}",
            "family": family,
            "template_id": "template-v1",
            "difficulty": difficulty,
            "public_metadata": {"rows": 5},
            "policy_id": policy,
            "policy_version": policy_version,
        },
    }


def paired_rows(
    outcomes: list[tuple[str, bool, bool]],
    *,
    split: str = "test",
    direct_tokens: int = 100,
    deliberate_tokens: int = 200,
    direct_tools: int = 1,
    deliberate_tools: int = 3,
    direct_messages: int = 1,
    deliberate_messages: int = 2,
) -> list[dict]:
    rows = []
    for task_id, direct_ok, deliberate_ok in outcomes:
        rows.append(
            make_row(
                task_id,
                "direct",
                split,
                direct_ok,
                tokens=direct_tokens,
                tool_calls=direct_tools,
                messages=direct_messages,
            )
        )
        rows.append(
            make_row(
                task_id,
                "deliberate",
                split,
                deliberate_ok,
                tokens=deliberate_tokens,
                tool_calls=deliberate_tools,
                messages=deliberate_messages,
            )
        )
    return rows


def make_treatment_row(
    task_id: str,
    treatment: TreatmentSpec,
    split: str,
    success: bool,
    *,
    registry_hash: str | None = None,
    attempt_id: str | None = None,
    tokens: int = 150,
) -> dict:
    row = make_row(
        task_id,
        treatment.id,
        split,
        success,
        attempt_id=attempt_id,
        tokens=tokens,
        policy_version=treatment.version,
    )
    row["treatment_bundle_id"] = treatment.bundle_id
    row["treatment_bundle_hash"] = treatment.bundle_hash
    if registry_hash is not None:
        row["treatment_registry_hash"] = registry_hash
    row["model_input"]["treatment"] = treatment_model_input_descriptor(treatment)
    return row


def treatment_panel_rows(
    registry: TreatmentRegistry,
    outcomes: list[tuple[str, list[bool]]],
    *,
    split: str = "test",
) -> list[dict]:
    rows: list[dict] = []
    for task_id, task_outcomes in outcomes:
        if len(task_outcomes) != len(registry.treatments):
            raise ValueError("one outcome is required per registry treatment")
        for index, (treatment, success) in enumerate(
            zip(registry.treatments, task_outcomes)
        ):
            rows.append(
                make_treatment_row(
                    task_id,
                    treatment,
                    split,
                    success,
                    registry_hash=registry.registry_hash,
                    tokens=100 + index * 100,
                )
            )
    return rows


def sample_registry(count: int = 3, seed: int = 91) -> TreatmentRegistry:
    treatments = sorted(generate_treatments(count, seed=seed), key=lambda item: item.bundle_id)
    return TreatmentRegistry(tuple(treatments))


def write_dataset(root: Path, rows: list[dict], name: str = "dataset.jsonl") -> Path:
    path = root / name
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    return path


def build_result_like(rows: list[dict], split: str = "test") -> dict:
    """Assemble a full evaluation dict the way evaluate_allocator does, using
    the pure pieces and neutral injected predictions (no torch needed)."""
    pairs, exclusions = ae.build_task_pairs(rows, split)
    uplift = [0.0] * len(pairs)
    selected_ids, k = ae.select_deliberate_tasks(pairs, uplift, 0.5)
    strategies = ae.compare_strategies(
        pairs, selected_ids, seed=42, random_trials=60, bootstrap_trials=60
    )
    cost_model = ae.fit_train_cost(rows)
    n = len(pairs)
    metadata = {
        "evaluator": "allocator_eval",
        "version": 1,
        "dataset_path": "/tmp/fake/dataset.jsonl",
        "artifact_dir": "/tmp/fake/artifacts",
        "split": split,
        "seed": 42,
        "deliberate_fraction": 0.5,
        "k": k,
        "n_tasks": n,
        "posterior_samples": 50,
        "random_trials": 60,
        "bootstrap_trials": 60,
        "duplicate_attempts": "reject",
        "cost_adjusted_uplift": False,
        "selection_rule": "rank tasks by predicted success uplift; ties by task_id",
        "observed_cost_role": "reporting only; observed cost never influences selection",
        "warnings": [],
        "statistical_warning": None,
    }
    model_meta = {
        "artifact_dir": "/tmp/fake/artifacts",
        "policies": ["direct", "deliberate"],
        "posterior_samples": 50,
        "config": {},
        "training": {},
    }
    budget = {
        "deliberate_fraction": 0.5,
        "k": k,
        "n": n,
        "rule": "exactly k tasks assigned to deliberate; all others direct",
    }
    return ae.assemble_output(
        strategies=strategies,
        metadata=metadata,
        exclusions=exclusions,
        model=model_meta,
        split=split,
        budget=budget,
        cost_model=cost_model,
    )


# ---------------------------------------------------------------------------
# Pair grouping and same-split handling
# ---------------------------------------------------------------------------
class PairingTest(unittest.TestCase):
    def test_exactly_one_row_per_policy_per_task(self) -> None:
        rows = paired_rows([("t1", True, True), ("t2", True, False)])
        pairs, exclusions = ae.build_task_pairs(rows, "test")
        self.assertEqual([pair["task_id"] for pair in pairs], ["t1", "t2"])
        self.assertEqual(exclusions["evaluated"], 2)
        self.assertEqual(exclusions["missing_pair"]["count"], 0)
        self.assertEqual(exclusions["duplicate_attempts"]["count"], 0)
        self.assertEqual(pairs[0]["split"], "test")

    def test_missing_policy_row_excluded_and_reported(self) -> None:
        rows = paired_rows([("t1", True, True)])
        rows.append(make_row("t2", "direct", "test", True))
        pairs, exclusions = ae.build_task_pairs(rows, "test")
        self.assertEqual([pair["task_id"] for pair in pairs], ["t1"])
        self.assertEqual(exclusions["missing_pair"], {"count": 1, "task_ids": ["t2"]})

    def test_duplicate_attempts_rejected_by_default(self) -> None:
        rows = paired_rows([("t1", True, True)])
        rows.append(make_row("t1", "direct", "test", False, attempt_id="z-late"))
        pairs, exclusions = ae.build_task_pairs(rows, "test")
        self.assertEqual(pairs, [])
        self.assertEqual(exclusions["duplicate_attempts"], {"count": 1, "task_ids": ["t1"]})

    def test_duplicate_attempts_first_rule_is_deterministic(self) -> None:
        rows = paired_rows([("t1", True, True)])
        rows.append(make_row("t1", "direct", "test", True, attempt_id="a-first"))
        rows.append(make_row("t1", "deliberate", "test", False, attempt_id="a-first-b"))
        pairs, exclusions = ae.build_task_pairs(rows, "test", duplicate_attempts="first")
        self.assertEqual(len(pairs), 1)
        # Lexicographically-first attempt id per policy is selected.
        self.assertEqual(pairs[0]["direct"]["attempt_id"], "a-first")
        self.assertEqual(pairs[0]["deliberate"]["attempt_id"], "a-first-b")
        self.assertEqual(exclusions["duplicate_attempts"]["count"], 0)

    def test_only_rows_from_the_requested_split_are_paired(self) -> None:
        rows = paired_rows([("t1", True, True)], split="train")
        rows += paired_rows([("t2", True, True)], split="validation")
        rows += paired_rows([("t3", True, True)], split="test")
        pairs, _ = ae.build_task_pairs(rows, "validation")
        self.assertEqual([pair["task_id"] for pair in pairs], ["t2"])
        self.assertEqual(pairs[0]["direct"]["split"], "validation")
        self.assertEqual(pairs[0]["deliberate"]["split"], "validation")

    def test_unknown_policy_rows_are_ignored(self) -> None:
        rows = paired_rows([("t1", True, True)])
        rows.append(make_row("t1", "other", "test", True))
        pairs, _ = ae.build_task_pairs(rows, "test")
        self.assertEqual([pair["task_id"] for pair in pairs], ["t1"])
        self.assertEqual(pairs[0]["direct"]["policy_id"], "direct")

    def test_invalid_duplicate_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            ae.build_task_pairs([], "test", duplicate_attempts="bogus")


class TreatmentPanelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = sample_registry()
        self.candidates = ae.resolve_treatment_candidates(self.registry)

    def _rows(self) -> list[dict]:
        return treatment_panel_rows(
            self.registry,
            [("t1", [True, False, True]), ("t2", [False, True, False])],
        )

    def test_candidate_resolution_is_canonical_and_committed(self) -> None:
        self.assertEqual(
            [item.bundle_id for item in self.candidates],
            sorted(item.bundle_id for item in self.registry),
        )
        selected = ae.resolve_treatment_candidates(
            self.registry,
            f"{self.candidates[1].id}@{self.candidates[1].version},{self.candidates[0].bundle_id}",
        )
        self.assertEqual(
            [item.bundle_id for item in selected],
            sorted([self.candidates[0].bundle_id, self.candidates[1].bundle_id]),
        )
        self.assertEqual(len(ae.treatment_candidate_set_hash(selected)), 64)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ae.resolve_treatment_candidates(
                self.registry,
                [self.candidates[0].id, self.candidates[0].bundle_id],
            )
        with self.assertRaisesRegex(ValueError, "unknown"):
            ae.resolve_treatment_candidates(self.registry, "not-registered")

    def test_builds_strict_complete_panels(self) -> None:
        panels, exclusions = ae.build_treatment_panels(
            self._rows(), "test", self.candidates, self.registry.registry_hash
        )
        self.assertEqual([panel["task_id"] for panel in panels], ["t1", "t2"])
        self.assertEqual(exclusions["evaluated"], 2)
        self.assertEqual(
            set(panels[0]["rows"]),
            {treatment.bundle_id for treatment in self.candidates},
        )

    def test_missing_cell_is_a_hard_error(self) -> None:
        rows = self._rows()
        rows.pop()
        with self.assertRaisesRegex(ValueError, "incomplete treatment panel"):
            ae.build_treatment_panels(
                rows, "test", self.candidates, self.registry.registry_hash
            )

    def test_duplicate_cell_is_a_hard_error(self) -> None:
        rows = self._rows()
        duplicate = dict(rows[0])
        duplicate["attempt_id"] = "duplicate-attempt"
        rows.append(duplicate)
        with self.assertRaisesRegex(ValueError, "duplicate treatment attempts"):
            ae.build_treatment_panels(
                rows, "test", self.candidates, self.registry.registry_hash
            )

    def test_bundle_and_registry_drift_are_rejected(self) -> None:
        rows = self._rows()
        rows[0]["treatment_bundle_hash"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "bundle_hash mismatch"):
            ae.build_treatment_panels(
                rows, "test", self.candidates, self.registry.registry_hash
            )
        rows = self._rows()
        rows[0]["treatment_registry_hash"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "registry_hash mismatch"):
            ae.build_treatment_panels(
                rows, "test", self.candidates, self.registry.registry_hash
            )
        rows = self._rows()
        rows[0].pop("treatment_registry_hash")
        with self.assertRaisesRegex(ValueError, "registry_hash mismatch"):
            ae.build_treatment_panels(
                rows, "test", self.candidates, self.registry.registry_hash
            )

    def test_descriptor_and_task_input_drift_are_rejected(self) -> None:
        rows = self._rows()
        rows[0]["model_input"]["treatment"]["tool_call_limit"] += 1
        with self.assertRaisesRegex(ValueError, "descriptor mismatch"):
            ae.build_treatment_panels(
                rows, "test", self.candidates, self.registry.registry_hash
            )
        rows = self._rows()
        rows[0]["model_input"]["text"] = "different task text"
        with self.assertRaisesRegex(ValueError, "differs across treatments"):
            ae.build_treatment_panels(
                rows, "test", self.candidates, self.registry.registry_hash
            )

    def test_cross_split_task_id_is_rejected(self) -> None:
        rows = self._rows()
        leaked = make_treatment_row(
            "t1",
            self.registry.treatments[0],
            "train",
            True,
            registry_hash=self.registry.registry_hash,
        )
        rows.append(leaked)
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            ae.build_treatment_panels(
                rows, "test", self.candidates, self.registry.registry_hash
            )


class TreatmentAllocationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = sample_registry()
        self.candidates = ae.resolve_treatment_candidates(self.registry)
        rows = treatment_panel_rows(
            self.registry,
            [
                ("t1", [True, False, False]),
                ("t2", [False, True, False]),
                ("t3", [False, False, True]),
                ("t4", [False, False, False]),
            ],
        )
        self.panels, _ = ae.build_treatment_panels(
            rows, "test", self.candidates, self.registry.registry_hash
        )

    def test_argmax_uses_predictions_and_lexical_ties(self) -> None:
        ids = [item.bundle_id for item in self.candidates]
        predictions = [
            {bundle_id: {"mean": 0.5, "std": 0.1} for bundle_id in ids},
            {
                ids[0]: {"mean": 0.1, "std": 0.1},
                ids[1]: {"mean": 0.8, "std": 0.1},
                ids[2]: {"mean": 0.2, "std": 0.1},
            },
            {
                ids[0]: {"mean": 0.1, "std": 0.1},
                ids[1]: {"mean": 0.2, "std": 0.1},
                ids[2]: {"mean": 0.9, "std": 0.1},
            },
            {bundle_id: {"mean": 0.0, "std": 0.1} for bundle_id in ids},
        ]
        assignment = ae.select_treatment_argmax(
            self.panels, predictions, self.candidates
        )
        self.assertEqual(assignment, [ids[0], ids[1], ids[2], ids[0]])

    def test_strategy_values_and_oracle_label(self) -> None:
        ids = [item.bundle_id for item in self.candidates]
        # Deliberately choose the wrong treatment on every winnable task.
        assignment = [ids[1], ids[2], ids[0], ids[0]]
        strategies = ae.compare_treatment_strategies(
            self.panels,
            assignment,
            self.candidates,
            seed=7,
            random_trials=2000,
            bootstrap_trials=100,
        )
        self.assertAlmostEqual(strategies["neural_argmax"]["success_rate"], 0.0)
        oracle = strategies["hindsight_realized_oracle"]
        self.assertAlmostEqual(oracle["success_rate"], 0.75)
        self.assertTrue(oracle["uses_heldout_outcomes_for_selection"])
        self.assertEqual(oracle["strategy_type"], "hindsight_realized_oracle")
        self.assertEqual(
            len([name for name in strategies if name.startswith("always::")]), 3
        )

    def test_uniform_random_reports_exact_expected_success(self) -> None:
        result = ae.uniform_random_treatment_aggregate(
            self.panels,
            self.candidates,
            seed=5,
            random_trials=1000,
        )
        # Three successes across 4 tasks x 3 candidate cells.
        self.assertAlmostEqual(result["success_rate"], 3 / 12)
        self.assertAlmostEqual(result["successes"], 1.0)
        self.assertIn("randomization_quantiles_95", result)
        self.assertEqual(result["random_trials"], 1000)

    def test_generalized_strategy_output_contains_no_task_payload(self) -> None:
        ids = [item.bundle_id for item in self.candidates]
        strategies = ae.compare_treatment_strategies(
            self.panels,
            [ids[0]] * len(self.panels),
            self.candidates,
            random_trials=20,
            bootstrap_trials=20,
        )
        blob = json.dumps(strategies, sort_keys=True)
        for forbidden in ("model_input", "verified_success", "usage", "task_id"):
            self.assertNotIn(f'"{forbidden}"', blob)


# ---------------------------------------------------------------------------
# Fail-closed split handling (excluded rows must never enter cost fitting)
# ---------------------------------------------------------------------------
class FailClosedSplitTest(unittest.TestCase):
    def test_row_split_rejects_excluded_and_unknown_splits(self) -> None:
        for split in ("canary_excluded", "pilot_excluded", "bogus"):
            with self.subTest(split=split):
                with self.assertRaises(ValueError):
                    ae._row_split(make_row("t1", "direct", split, True))

    def test_row_split_rejects_missing_split(self) -> None:
        row = make_row("t1", "direct", "train", True)
        del row["split"]
        with self.assertRaises(ValueError):
            ae._row_split(row)

    def test_cost_fit_rejects_excluded_rows(self) -> None:
        rows = [make_row("t1", "direct", "train", True, tokens=100)]
        rows.append(make_row("t2", "direct", "canary_excluded", True, tokens=999999))
        with self.assertRaisesRegex(ValueError, "split"):
            ae.fit_train_cost(rows)

    def test_task_pairs_reject_excluded_rows(self) -> None:
        rows = paired_rows([("t1", True, True)])
        rows.append(make_row("t2", "direct", "pilot_excluded", True))
        with self.assertRaisesRegex(ValueError, "split"):
            ae.build_task_pairs(rows, "test")


# ---------------------------------------------------------------------------
# Cost extraction and train-only fitting
# ---------------------------------------------------------------------------
class CostFitTest(unittest.TestCase):
    def test_total_tokens_preferred_over_fallback(self) -> None:
        row = make_row("t1", "direct", "test", True, usage={"total_tokens": 150, "input": 999})
        self.assertEqual(ae.task_tokens(row), 150)

    def test_fallback_sums_output_input_cache(self) -> None:
        row = make_row(
            "t1",
            "direct",
            "test",
            True,
            usage={"output": 40, "input": 60, "cache": 10},
        )
        self.assertEqual(ae.task_tokens(row), 110)
        row2 = make_row("t1", "direct", "test", True, usage={"output": 40, "input": 60})
        self.assertEqual(ae.task_tokens(row2), 100)

    def test_missing_cost_is_none(self) -> None:
        self.assertIsNone(ae.task_tokens(make_row("t1", "direct", "test", True, usage={})))
        row = make_row("t1", "direct", "test", True)
        row["usage"] = {"output": "forty"}
        self.assertIsNone(ae.task_tokens(row))

    def test_cost_fit_uses_train_rows_only(self) -> None:
        rows = [
            make_row("t1", "direct", "train", True, tokens=100),
            make_row("t2", "direct", "train", True, tokens=120),
            make_row("t3", "direct", "train", True, tokens=90),
            make_row("t4", "deliberate", "train", True, tokens=200),
            make_row("t5", "deliberate", "train", True, tokens=220),
            make_row("t6", "deliberate", "train", True, tokens=210),
            # Test/validation rows must never enter the fit, even with absurd costs.
            make_row("t7", "direct", "test", True, tokens=999999),
            make_row("t8", "deliberate", "test", True, tokens=999999),
            make_row("t9", "direct", "validation", True, tokens=888888),
        ]
        cost = ae.fit_train_cost(rows)
        self.assertEqual(cost["fit_split"], "train")
        self.assertEqual(cost["n_direct"], 3)
        self.assertEqual(cost["n_deliberate"], 3)
        self.assertAlmostEqual(cost["mean_tokens_direct"], 310.0 / 3.0)
        self.assertAlmostEqual(cost["mean_tokens_deliberate"], 210.0)
        self.assertAlmostEqual(cost["incremental_tokens_per_task"], 210.0 - 310.0 / 3.0)

    def test_cost_fit_with_missing_costs_and_unknown_policies(self) -> None:
        rows = [
            make_row("t1", "direct", "train", True, usage={}),
            make_row("t2", "direct", "train", True, tokens=100),
            make_row("t3", "other", "train", True, tokens=50),
        ]
        cost = ae.fit_train_cost(rows)
        self.assertEqual(cost["n_direct"], 1)
        self.assertAlmostEqual(cost["mean_tokens_direct"], 100.0)
        self.assertIsNone(cost["mean_tokens_deliberate"])
        self.assertIsNone(cost["incremental_tokens_per_task"])


# ---------------------------------------------------------------------------
# Selection: exact k, ties, no leakage
# ---------------------------------------------------------------------------
class SelectionTest(unittest.TestCase):
    def test_exact_k_with_deterministic_ties_by_task_id(self) -> None:
        rows = paired_rows(
            [
                ("t1", True, False),
                ("t2", False, True),
                ("t3", True, True),
                ("t4", False, False),
            ]
        )
        pairs, _ = ae.build_task_pairs(rows, "test")
        # Every uplift identical -> ties broken by task_id -> first two ids win.
        uplift = [0.5, 0.5, 0.5, 0.5]
        selected, k = ae.select_deliberate_tasks(pairs, uplift, 0.5)
        self.assertEqual(k, 2)
        self.assertEqual(selected, ["t1", "t2"])

    def test_rank_by_uplift_descending(self) -> None:
        rows = paired_rows([("t1", True, True), ("t2", True, True), ("t3", True, True)])
        pairs, _ = ae.build_task_pairs(rows, "test")
        uplift = [0.1, 0.9, 0.5]
        selected, k = ae.select_deliberate_tasks(pairs, uplift, 2 / 3)
        self.assertEqual(k, 2)
        self.assertEqual(selected, ["t2", "t3"])

    def test_budget_clamped_to_full_range(self) -> None:
        rows = paired_rows([("t1", True, True), ("t2", True, True)])
        pairs, _ = ae.build_task_pairs(rows, "test")
        selected, k = ae.select_deliberate_tasks(pairs, [0.0, 0.0], 0.0)
        self.assertEqual((selected, k), ([], 0))
        selected, k = ae.select_deliberate_tasks(pairs, [0.0, 0.0], 1.0)
        self.assertEqual((selected, k), (["t1", "t2"], 2))

    def test_fraction_validation(self) -> None:
        rows = paired_rows([("t1", True, True)])
        pairs, _ = ae.build_task_pairs(rows, "test")
        for bad in (-0.1, 1.1, "half"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    ae.select_deliberate_tasks(pairs, [0.0], bad)

    def test_uplift_length_mismatch_raises(self) -> None:
        rows = paired_rows([("t1", True, True), ("t2", True, True)])
        pairs, _ = ae.build_task_pairs(rows, "test")
        with self.assertRaises(ValueError):
            ae.select_deliberate_tasks(pairs, [0.0], 0.5)

    def test_selection_uses_predictions_not_observed_outcomes(self) -> None:
        # The model prefers deliberate on t1 (wrong), while the observed data
        # says deliberate wins only on t2.  Selection must follow predictions.
        rows = paired_rows(
            [
                ("t1", True, False),
                ("t2", False, True),
                ("t3", True, True),
            ]
        )
        pairs, _ = ae.build_task_pairs(rows, "test")
        uplift = [0.5, -0.5, 0.0]
        selected, k = ae.select_deliberate_tasks(pairs, uplift, 1 / 3)
        self.assertEqual(k, 1)
        self.assertEqual(selected, ["t1"])
        strategies = ae.compare_strategies(
            pairs, selected, seed=42, random_trials=50, bootstrap_trials=50
        )
        neural = strategies["neural_allocator"]
        # t1 deliberate(F), t2 direct(F), t3 direct(T) -> 1/3.  If selection had
        # peeked at outcomes it would have picked t2 and scored 1.0.
        self.assertAlmostEqual(neural["success_rate"], 1 / 3)
        oracle = strategies["oracle_upper_bound"]
        self.assertAlmostEqual(oracle["success_rate"], 1.0)
        # Neural allocator never returns the observed selection.
        self.assertNotEqual(neural["success_rate"], oracle["success_rate"])


# ---------------------------------------------------------------------------
# Strategy aggregates on a hand dataset
# ---------------------------------------------------------------------------
class StrategiesTest(unittest.TestCase):
    def _hand_dataset(self):
        rows = paired_rows(
            [
                ("t1", True, True),
                ("t2", True, False),
                ("t3", False, True),
                ("t4", False, False),
            ]
        )
        pairs, _ = ae.build_task_pairs(rows, "test")
        uplift = [0.01, 0.99, 0.98, -0.5]
        selected, k = ae.select_deliberate_tasks(pairs, uplift, 0.5)
        self.assertEqual((selected, k), (["t2", "t3"], 2))
        return pairs, selected

    def test_expected_baselines_on_hand_dataset(self) -> None:
        pairs, selected = self._hand_dataset()
        strategies = ae.compare_strategies(
            pairs, selected, seed=7, random_trials=2000, bootstrap_trials=200
        )
        # Always Direct: t1 T, t2 T, t3 F, t4 F -> 0.5
        self.assertAlmostEqual(strategies["always_direct"]["success_rate"], 0.5)
        self.assertEqual(strategies["always_direct"]["selected_deliberate"], 0)
        self.assertEqual(strategies["always_direct"]["n"], 4)
        self.assertAlmostEqual(strategies["always_direct"]["mean_tokens"], 100.0)
        self.assertAlmostEqual(strategies["always_direct"]["mean_tool_calls"], 1.0)
        # Always Deliberate: t1 T, t2 F, t3 T, t4 F -> 0.5
        self.assertAlmostEqual(strategies["always_deliberate"]["success_rate"], 0.5)
        self.assertEqual(strategies["always_deliberate"]["selected_deliberate"], 4)
        self.assertAlmostEqual(strategies["always_deliberate"]["mean_tokens"], 200.0)
        self.assertAlmostEqual(strategies["always_deliberate"]["mean_messages"], 2.0)
        # Neural: t1 direct(T), t2 deliberate(F), t3 deliberate(T), t4 direct(F)
        neural = strategies["neural_allocator"]
        self.assertAlmostEqual(neural["success_rate"], 0.5)
        self.assertEqual(neural["selected_deliberate"], 2)
        self.assertAlmostEqual(neural["mean_tokens"], 150.0)
        self.assertAlmostEqual(neural["mean_tool_calls"], 2.0)
        self.assertAlmostEqual(neural["mean_messages"], 1.5)
        # Oracle upper bound at same k: observed uplift t3=+1, t1/t4=0, t2=-1
        # -> select t3, t1 (ties by id) -> 3 successes.
        oracle = strategies["oracle_upper_bound"]
        self.assertAlmostEqual(oracle["success_rate"], 0.75)
        self.assertEqual(oracle["selected_deliberate"], 2)
        self.assertAlmostEqual(oracle["mean_messages"], 1.5)
        # Per-task oracle = t1,t2,t3 = 3 successes.
        for name in ("always_direct", "always_deliberate", "neural_allocator"):
            self.assertEqual(strategies[name]["regret"], 1.0)
        self.assertEqual(oracle["regret"], 0.0)

    def test_random_mix_mean_and_quantiles(self) -> None:
        pairs, selected = self._hand_dataset()
        strategies = ae.compare_strategies(
            pairs, selected, seed=7, random_trials=2000, bootstrap_trials=200
        )
        mix = strategies["random_mix"]
        self.assertEqual(mix["selected_deliberate"], 2)
        self.assertEqual(mix["n"], 4)
        # Analytic mean over the six exact-2 subsets is 0.5.
        self.assertAlmostEqual(mix["success_rate"], 0.5, delta=0.05)
        self.assertAlmostEqual(mix["mean_tokens"], 150.0, delta=3.0)
        lo, hi = mix["quantiles"]["success_rate"]
        self.assertLessEqual(lo, mix["success_rate"])
        self.assertLessEqual(mix["success_rate"], hi)
        self.assertLessEqual(lo, hi)
        self.assertAlmostEqual(mix["regret"], 1.0, delta=0.05)

    def test_missing_costs_excluded_from_cost_but_not_success(self) -> None:
        rows = [
            make_row("t1", "direct", "test", True, tokens=100),
            make_row("t1", "deliberate", "test", True, tokens=200),
            make_row("t2", "direct", "test", True, tokens=120),
            make_row("t2", "deliberate", "test", False, usage={}),
            make_row("t3", "direct", "test", False, tokens=90),
            make_row("t3", "deliberate", "test", True, tokens=180),
        ]
        pairs, _ = ae.build_task_pairs(rows, "test")
        uplift = [0.1, 0.2, -0.3]
        selected, k = ae.select_deliberate_tasks(pairs, uplift, 0.5)
        self.assertEqual((selected, k), (["t1", "t2"], 2))
        strategies = ae.compare_strategies(
            pairs, selected, seed=3, random_trials=100, bootstrap_trials=100
        )
        neural = strategies["neural_allocator"]
        # Assignment: t1 deliberate(T), t2 deliberate(F), t3 direct(F).
        self.assertEqual(neural["n"], 3)
        self.assertAlmostEqual(neural["success_rate"], 1 / 3)
        # t2's deliberate cost is missing -> excluded from the mean, not the n.
        self.assertEqual(neural["cost_n"], 2)
        self.assertAlmostEqual(neural["mean_tokens"], (200.0 + 90.0) / 2.0)
        # Always Direct is unaffected by the missing deliberate cost.
        self.assertAlmostEqual(strategies["always_direct"]["mean_tokens"], 310.0 / 3.0)
        self.assertEqual(strategies["always_direct"]["cost_n"], 3)

    def test_bootstrap_ci_all_success_collapses_to_one(self) -> None:
        rows = paired_rows([("t1", True, True), ("t2", True, True), ("t3", True, True)])
        pairs, _ = ae.build_task_pairs(rows, "test")
        assignment = ["direct"] * len(pairs)
        first = ae.task_bootstrap_ci(pairs, assignment, seed=9, bootstrap_trials=300)
        second = ae.task_bootstrap_ci(pairs, assignment, seed=9, bootstrap_trials=300)
        self.assertEqual(first, [1.0, 1.0])
        self.assertEqual(first, second)

    def test_bootstrap_ci_deterministic_and_ordered(self) -> None:
        rows = paired_rows([("t1", True, True), ("t2", True, False), ("t3", False, True)])
        pairs, _ = ae.build_task_pairs(rows, "test")
        assignment = ["direct", "direct", "direct"]
        first = ae.task_bootstrap_ci(pairs, assignment, seed=11, bootstrap_trials=500)
        second = ae.task_bootstrap_ci(pairs, assignment, seed=11, bootstrap_trials=500)
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], first[1])
        self.assertGreaterEqual(first[0], 0.0)
        self.assertLessEqual(first[1], 1.0)

    def test_bootstrap_ci_none_for_tiny_n(self) -> None:
        rows = paired_rows([("t1", True, True)])
        pairs, _ = ae.build_task_pairs(rows, "test")
        self.assertIsNone(
            ae.task_bootstrap_ci(pairs, ["direct"], seed=1, bootstrap_trials=100)
        )

    def test_random_mix_reproducible_under_same_seed(self) -> None:
        rows = paired_rows([("t1", True, True), ("t2", True, False), ("t3", False, True)])
        pairs, _ = ae.build_task_pairs(rows, "test")
        first = ae.random_mix_aggregate(pairs, 2, seed=5, random_trials=400)
        second = ae.random_mix_aggregate(pairs, 2, seed=5, random_trials=400)
        self.assertEqual(first, second)
        other = ae.random_mix_aggregate(pairs, 2, seed=6, random_trials=400)
        self.assertNotEqual(first["success_rate"], other["success_rate"])

    def test_strategy_output_is_fully_json_serializable(self) -> None:
        rows = paired_rows([("t1", True, True), ("t2", True, False)])
        pairs, _ = ae.build_task_pairs(rows, "test")
        selected, _ = ae.select_deliberate_tasks(pairs, [0.1, 0.2], 0.5)
        strategies = ae.compare_strategies(
            pairs, selected, seed=1, random_trials=20, bootstrap_trials=20
        )
        blob = json.dumps(strategies, sort_keys=True)
        self.assertIsInstance(json.loads(blob), dict)
        self.assertEqual(
            list(strategies),
            ["always_direct", "always_deliberate", "random_mix", "neural_allocator", "oracle_upper_bound"],
        )


# ---------------------------------------------------------------------------
# Privacy: no per-task predictions/outcomes in the output
# ---------------------------------------------------------------------------
class PrivacyTest(unittest.TestCase):
    def _secret_rows(self) -> list[dict]:
        rows = paired_rows([("t1", True, False), ("t2", False, True)])
        for row in rows:
            row["model_input"]["text"] = "SECRET_TEXT_" + row["policy_id"]
            row["prompt"] = "SECRET_PROMPT_" + row["task_id"]
            row["failure_code"] = "SECRET_FAILURE"
            row["usage"]["secret_key"] = "SECRET_USAGE_KEY"
            row["public_metadata"]["secret_name"] = "SECRET_METADATA"
        return rows

    def test_output_contains_no_per_task_fields(self) -> None:
        result = build_result_like(self._secret_rows())
        blob = json.dumps(result, sort_keys=True)
        for sentinel in (
            "SECRET_TEXT_",
            "SECRET_PROMPT_",
            "SECRET_FAILURE",
            "SECRET_USAGE_KEY",
            "SECRET_METADATA",
        ):
            with self.subTest(sentinel=sentinel):
                self.assertNotIn(sentinel, blob)
        for forbidden_key in (
            "model_input",
            "verified_success",
            "usage",
            "prompt",
            "prediction",
            "per_task",
            "public_metadata",
            "outcome",
        ):
            with self.subTest(forbidden_key=forbidden_key):
                self.assertNotIn(f'"{forbidden_key}"', blob)

    def test_strategy_entries_are_aggregates_only(self) -> None:
        result = build_result_like(self._secret_rows())
        allowed = {
            "success_rate",
            "n",
            "mean_tokens",
            "mean_tool_calls",
            "mean_messages",
            "selected_deliberate",
            "regret",
            "ci_95",
            "cost_n",
            "quantiles",
            "random_trials",
        }
        for name, entry in result["strategies"].items():
            with self.subTest(strategy=name):
                self.assertTrue(set(entry).issubset(allowed), set(entry) - allowed)

    def test_top_level_layout_matches_contract(self) -> None:
        result = build_result_like(self._secret_rows())
        self.assertEqual(
            set(result), {"strategies", "metadata", "exclusions", "model", "cost_model", "split", "budget"}
        )
        self.assertEqual(result["split"], "test")
        self.assertIn("budget", result)
        self.assertIn("exclusions", result)
        self.assertIn("k", result["budget"])


# ---------------------------------------------------------------------------
# Dashboard compatibility
# ---------------------------------------------------------------------------
class DashboardCompatibilityTest(unittest.TestCase):
    def test_dashboard_parser_reads_evaluation_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = paired_rows(
                [("t1", True, True), ("t2", True, False), ("t3", False, True), ("t4", False, False)]
            )
            result = build_result_like(rows)
            output = root / "evaluation.json"
            ae.write_evaluation(result, output)
            parsed = db._parse_baselines(output)
            self.assertEqual(parsed["status"], "present")
            by_name = {row["strategy"]: row for row in parsed["rows"]}
            self.assertEqual(set(by_name), set(result["strategies"]))
            for name, entry in result["strategies"].items():
                row = by_name[name]
                self.assertAlmostEqual(row["success_rate"], entry["success_rate"])
                self.assertEqual(row["n"], entry["n"])
                self.assertAlmostEqual(row["mean_tokens"], entry["mean_tokens"])
                self.assertAlmostEqual(row["mean_tool_calls"], entry["mean_tool_calls"])
                self.assertAlmostEqual(row["mean_messages"], entry["mean_messages"])


# ---------------------------------------------------------------------------
# Write atomicity and CLI
# ---------------------------------------------------------------------------
class WriteAndCliTest(unittest.TestCase):
    def test_write_evaluation_is_atomic_and_summarizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = build_result_like(paired_rows([("t1", True, True)]))
            output = root / "nested" / "dir" / "evaluation.json"
            summary = ae.write_evaluation(result, output)
            self.assertTrue(output.is_file())
            self.assertEqual(summary["n_tasks"], 1)
            self.assertEqual(summary["strategies"], sorted(result["strategies"]))
            self.assertIn("output_path", summary)
            leftovers = list(output.parent.glob(".*evaluation.json.*"))
            self.assertEqual(leftovers, [])
            reloaded = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(reloaded, json.loads(json.dumps(result, sort_keys=True)))

    def test_write_evaluation_rejects_non_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                ae.write_evaluation({"not": "strategies"}, root / "out.json")
            self.assertFalse((root / "out.json").exists())

    def test_cli_error_returns_nonzero_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = write_dataset(root, paired_rows([("t1", True, True)]))
            output = root / "out.json"
            stderr = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
                code = ae.main([str(dataset), str(root / "missing-artifacts"), str(output)])
            self.assertEqual(code, 1)
            self.assertIn("error:", stderr.getvalue())
            self.assertFalse(output.exists())

    def test_cli_missing_dataset_returns_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stderr = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
                code = ae.main([str(root / "missing.jsonl"), str(root / "artifacts")])
            self.assertEqual(code, 1)
            self.assertIn("error:", stderr.getvalue())

    def test_cli_parser_options(self) -> None:
        parser = ae.build_parser()
        args = parser.parse_args(
            [
                "ds.jsonl",
                "artifacts",
                "out.json",
                "--split",
                "validation",
                "--deliberate-fraction",
                "0.3",
                "--posterior-samples",
                "25",
                "--seed",
                "7",
                "--random-trials",
                "50",
                "--bootstrap-trials",
                "40",
                "--duplicate-attempts",
                "first",
                "--cost-adjusted-uplift",
            ]
        )
        self.assertEqual(args.split, "validation")
        self.assertAlmostEqual(args.deliberate_fraction, 0.3)
        self.assertEqual(args.posterior_samples, 25)
        self.assertEqual(args.seed, 7)
        self.assertEqual(args.random_trials, 50)
        self.assertEqual(args.bootstrap_trials, 40)
        self.assertEqual(args.duplicate_attempts, "first")
        self.assertTrue(args.cost_adjusted_uplift)

    def test_cli_parser_generalized_options(self) -> None:
        parser = ae.build_parser()
        args = parser.parse_args(
            [
                "ds.jsonl",
                "artifacts",
                "--treatment-registry",
                "registry.json",
                "--treatments",
                "all",
            ]
        )
        self.assertEqual(args.treatment_registry, "registry.json")
        self.assertEqual(args.treatments, "all")

    def test_cli_treatments_requires_registry(self) -> None:
        stderr = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
            code = ae.main(["missing.jsonl", "artifacts", "--treatments", "a"])
        self.assertEqual(code, 1)
        self.assertIn("requires --treatment-registry", stderr.getvalue())

    def test_generalized_mode_rejects_legacy_budget_options(self) -> None:
        stderr = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
            code = ae.main(
                [
                    "missing.jsonl",
                    "artifacts",
                    "--treatment-registry",
                    "registry.json",
                    "--deliberate-fraction",
                    "0.5",
                ]
            )
        self.assertEqual(code, 1)
        self.assertIn("does not accept legacy options", stderr.getvalue())

    def test_invalid_fraction_cli_returns_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = write_dataset(root, paired_rows([("t1", True, True)]))
            stderr = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
                code = ae.main(
                    [str(dataset), str(root / "artifacts"), "--deliberate-fraction", "1.5"]
                )
            self.assertEqual(code, 1)
            self.assertIn("error:", stderr.getvalue())


# ---------------------------------------------------------------------------
# Torch integration (skipped when PyTorch is not installed)
# ---------------------------------------------------------------------------
@TORCH_REQUIRED
class TreatmentTorchIntegrationTest(unittest.TestCase):
    def test_descriptor_model_evaluates_complete_registry_panels(self) -> None:
        registry = sample_registry()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_path = root / "registry.json"
            registry.save(registry_path)
            rows: list[dict] = []
            for split, count in (("train", 9), ("validation", 3), ("test", 3)):
                for task_index in range(count):
                    outcomes = [
                        (task_index + treatment_index) % 3 == 0
                        for treatment_index in range(len(registry.treatments))
                    ]
                    rows.extend(
                        treatment_panel_rows(
                            registry,
                            [(f"{split}-{task_index}", outcomes)],
                            split=split,
                        )
                    )
            dataset = write_dataset(root, rows)
            artifact_dir = root / "artifacts"
            om.train_model(
                dataset,
                artifact_dir,
                epochs=2,
                batch_size=8,
                seed=42,
                max_vocab=200,
                max_tokens=32,
                text_dim=8,
                cat_dim=4,
                numeric_hidden=6,
                fusion_hidden=8,
                num_samples=10,
                verbose=False,
            )
            result = ae.evaluate_treatment_allocator(
                dataset,
                artifact_dir,
                treatment_registry=registry_path,
                split="test",
                posterior_samples=10,
                random_trials=40,
                bootstrap_trials=40,
                seed=42,
            )
        self.assertEqual(result["metadata"]["mode"], "generalized_treatments")
        self.assertEqual(result["metadata"]["n_tasks"], 3)
        self.assertEqual(
            result["treatment_set"]["registry_hash"], registry.registry_hash
        )
        self.assertEqual(len(result["treatment_set"]["candidates"]), 3)
        self.assertTrue(
            all(item["seen_in_training"] for item in result["treatment_set"]["candidates"])
        )
        self.assertIn("neural_argmax", result["strategies"])
        self.assertIn("hindsight_realized_oracle", result["strategies"])
        blob = json.dumps(result, sort_keys=True)
        for forbidden in ('"model_input"', '"verified_success"', '"usage"', '"per_task"'):
            self.assertNotIn(forbidden, blob)


@TORCH_REQUIRED
class TorchIntegrationTest(unittest.TestCase):
    def _write_synthetic_dataset(self, root: Path) -> Path:
        rows = []
        # Train split: 12 paired tasks, both policies on each task.
        for index in range(12):
            task_id = f"train-{index}"
            direct_ok = index % 2 == 0
            deliberate_ok = index % 3 == 0
            rows.append(
                make_row(
                    task_id,
                    "direct",
                    "train",
                    direct_ok,
                    tokens=120,
                    tool_calls=2,
                    messages=1,
                )
            )
            rows.append(
                make_row(
                    task_id,
                    "deliberate",
                    "train",
                    deliberate_ok,
                    tokens=240,
                    tool_calls=5,
                    messages=2,
                )
            )
        for index in range(4):
            task_id = f"val-{index}"
            rows.append(
                make_row(
                    task_id,
                    "direct",
                    "validation",
                    index % 2 == 0,
                    tokens=130,
                    tool_calls=2,
                    messages=1,
                )
            )
            rows.append(
                make_row(
                    task_id,
                    "deliberate",
                    "validation",
                    index % 3 == 0,
                    tokens=250,
                    tool_calls=5,
                    messages=2,
                )
            )
        for index in range(4):
            task_id = f"test-{index}"
            rows.append(
                make_row(
                    task_id,
                    "direct",
                    "test",
                    index % 2 == 1,
                    tokens=140,
                    tool_calls=2,
                    messages=1,
                )
            )
            rows.append(
                make_row(
                    task_id,
                    "deliberate",
                    "test",
                    index % 3 == 1,
                    tokens=260,
                    tool_calls=5,
                    messages=2,
                )
            )
        return write_dataset(root, rows)

    def _train(self, root: Path, dataset: Path, artifact_dir: Path) -> None:
        om.train_model(
            dataset,
            artifact_dir,
            epochs=2,
            batch_size=8,
            seed=42,
            max_vocab=200,
            max_tokens=32,
            text_dim=8,
            cat_dim=4,
            numeric_hidden=6,
            fusion_hidden=8,
            num_samples=20,
            verbose=False,
        )

    def test_end_to_end_validation_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self._write_synthetic_dataset(root)
            artifact_dir = root / "artifacts"
            self._train(root, dataset, artifact_dir)
            result = ae.evaluate_allocator(
                dataset,
                artifact_dir,
                split="validation",
                deliberate_fraction=0.5,
                posterior_samples=10,
                seed=42,
                random_trials=50,
                bootstrap_trials=50,
            )
            self.assertEqual(result["split"], "validation")
            self.assertEqual(result["budget"]["n"], 4)
            self.assertEqual(result["budget"]["k"], 2)
            self.assertEqual(
                set(result["strategies"]),
                {"always_direct", "always_deliberate", "random_mix", "neural_allocator", "oracle_upper_bound"},
            )
            neural = result["strategies"]["neural_allocator"]
            self.assertEqual(neural["selected_deliberate"], 2)
            self.assertEqual(neural["n"], 4)
            self.assertGreaterEqual(neural["success_rate"], 0.0)
            self.assertLessEqual(neural["success_rate"], 1.0)
            self.assertIn("statistical_warning", result["metadata"])
            self.assertIsNotNone(result["metadata"]["statistical_warning"])
            self.assertEqual(result["cost_model"]["fit_split"], "train")
            # Privacy: no per-task payloads anywhere in the result.
            blob = json.dumps(result, sort_keys=True)
            for forbidden in ('"model_input"', '"verified_success"', '"usage"', '"per_task"'):
                self.assertNotIn(forbidden, blob)

    def test_evaluate_allocator_reproducible_under_same_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self._write_synthetic_dataset(root)
            artifact_dir = root / "artifacts"
            self._train(root, dataset, artifact_dir)
            first = ae.evaluate_allocator(
                dataset,
                artifact_dir,
                split="validation",
                posterior_samples=10,
                seed=42,
                random_trials=60,
                bootstrap_trials=60,
            )
            second = ae.evaluate_allocator(
                dataset,
                artifact_dir,
                split="validation",
                posterior_samples=10,
                seed=42,
                random_trials=60,
                bootstrap_trials=60,
            )
            self.assertEqual(first, second)

    def test_evaluate_allocator_raises_without_paired_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self._write_synthetic_dataset(root)
            artifact_dir = root / "artifacts"
            self._train(root, dataset, artifact_dir)
            rows = om.load_dataset_rows(dataset)
            for row in rows:
                if row["split"] == "validation" and row["policy_id"] == "deliberate":
                    # Destroy the validation pairs in the feature path too
                    # (pairing reads model_input.policy_id).
                    row["policy_id"] = "direct"
                    row["model_input"]["policy_id"] = "direct"
            path = write_dataset(root, rows, name="broken.jsonl")
            with self.assertRaises(ValueError):
                ae.evaluate_allocator(path, artifact_dir, split="validation")

    def test_cli_success_writes_atomic_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self._write_synthetic_dataset(root)
            artifact_dir = root / "artifacts"
            self._train(root, dataset, artifact_dir)
            output = root / "cli-evaluation.json"
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = ae.main(
                    [
                        str(dataset),
                        str(artifact_dir),
                        str(output),
                        "--split",
                        "validation",
                        "--posterior-samples",
                        "10",
                        "--random-trials",
                        "40",
                        "--bootstrap-trials",
                        "40",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertTrue(output.is_file())
            summary = json.loads(buffer.getvalue())
            self.assertEqual(summary["n_tasks"], 4)
            self.assertIn("neural_allocator", summary["strategies"])
            reloaded = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("strategies", reloaded)


if __name__ == "__main__":
    unittest.main()
