from __future__ import annotations

import json
import unittest
from pathlib import Path

from pyreplab_harness.m3_headroom_gate import (
    _uniform_tie_score,
    evaluate_headroom_gate,
)
from pyreplab_harness.orchestrator import policy_spec_from_treatment
from pyreplab_harness.treatments import TreatmentRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "policies" / "m3-unbrowser-treatments.json"
SPLIT_PATH = PROJECT_ROOT / "policies" / "m3-unbrowser-policy-split.json"
PILOT_PATH = PROJECT_ROOT / "policies" / "m3-unbrowser-headroom-pilot.json"


def _entry(tool: str, **details):
    return {
        "tool_name": tool,
        "is_error": False,
        "budget_rejected": False,
        "details": details,
    }


class M3HeadroomGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = TreatmentRegistry.load(REGISTRY_PATH)
        self.policy_split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
        self.manifest = json.loads(PILOT_PATH.read_text(encoding="utf-8"))
        self.preflight = {
            "pilot_manifest_hash": self.manifest["manifest_hash"],
            "code_revision": "a" * 40,
            "source_tree_hash": "b" * 64,
            "runtime_pins": self.manifest["runtime_pins"],
        }

    @staticmethod
    def _success(task_index: int, label: str) -> bool:
        if task_index < 6:
            return True
        if task_index == 6:
            return label == "A"
        if task_index == 7:
            return label == "B"
        if task_index == 8:
            return label == "C"
        return False

    def _trajectory(self, bundle_id: str, task: dict) -> dict:
        treatment = self.registry.by_bundle_id(bundle_id)
        metadata = treatment.generator_metadata
        planning = metadata["planning"]
        planning_shape = {
            "present": planning != "direct",
            "line_count": 1 if planning == "brief_plan" else (2 if planning == "decompose" else 0),
            "plan_marker": planning == "brief_plan",
            "step_marker_count": 2 if planning == "decompose" else 0,
        }
        first_action = {
            "text_first": "text",
            "structure_first": "blockmap",
            "targeted_query_first": "query",
        }[metadata["observation"]]
        selector = "#target" if first_action != "blockmap" else None
        first_read = _entry("unbrowser", action=first_action, selector=selector)
        trace = [_entry("unbrowser", action="navigate"), first_read]
        if task["template"] == "distractor_recovery":
            trace.append(
                _entry(
                    "unbrowser",
                    action="click",
                    status=task["recovery_probe_status"],
                    url=task["recovery_probe_url"],
                )
            )
            if metadata["recovery"] == "diagnose_retry_once":
                trace.append(_entry("unbrowser", action="navigate", status=200))
        else:
            if metadata["verification"] == "final_reobserve":
                trace.append(_entry("unbrowser", action=first_action, selector=selector))
            trace.append(_entry("bash", exit_code=0, result_submission=True))
        return {
            "provider_turn_count": 1,
            "planning_preamble": planning_shape,
            "tool_trace": trace,
        }

    def _records(self) -> list[dict]:
        costs = {"A": 100, "B": 125, "C": 150, "D": 200}
        records = []
        task_by_id = {task["task_id"]: task for task in self.manifest["tasks"]}
        task_indices = {
            task["task_id"]: index for index, task in enumerate(self.manifest["tasks"])
        }
        for panel in self.manifest["panels"]:
            task = task_by_id[panel["task_id"]]
            task_index = task_indices[task["task_id"]]
            replica = panel["rollout_replica"]
            attempts = {}
            execution_order = []
            for label in panel["execution_order"]:
                bundle_id = self.manifest["policy_labels"][label]
                execution_order.append(bundle_id)
                attempts[bundle_id] = {
                    "attempt_id": f"attempt-{task_index:02d}-{replica}-{label}",
                    "policy": policy_spec_from_treatment(
                        self.registry.by_bundle_id(bundle_id)
                    ).to_dict(),
                    "pi_return_code": 0,
                    "sampling_receipt": {
                        "seed": panel["sampling_seed"],
                        "parameters": self.manifest["runtime_pins"]["sampling"][
                            "parameters"
                        ],
                    },
                    "verification": {
                        "success": self._success(task_index, label),
                        "verifier_id": self.manifest["runtime_pins"][
                            "fixture_verifier_id"
                        ],
                        "verifier_version": self.manifest["runtime_pins"][
                            "fixture_verifier_version"
                        ],
                    },
                    "usage": {"output": costs[label]},
                    "trajectory": self._trajectory(bundle_id, task),
                }
            records.append(
                {
                    "schema_version": "m3-headroom-task-result-v1",
                    "key": panel["panel_id"],
                    "pilot_manifest_hash": self.manifest["manifest_hash"],
                    "task": task,
                    "panel": panel,
                    "status": "completed",
                    "result": {
                        "task_id": task["task_id"],
                        "treatment_registry_hash": self.registry.registry_hash,
                        "execution_order": execution_order,
                        "rollout_replica": replica,
                        "sampling_seed": panel["sampling_seed"],
                        "pilot_manifest_hash": self.manifest["manifest_hash"],
                        "pilot_panel_id": panel["panel_id"],
                        "attempts": attempts,
                    },
                }
            )
        return records

    def test_complete_passing_matrix_passes_all_frozen_gates(self) -> None:
        report = evaluate_headroom_gate(
            self.manifest,
            self.registry,
            self.policy_split,
            self._records(),
            preflight=self.preflight,
        )
        self.assertTrue(report["passed"], report["reasons"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(report["headroom"]["stable_disagreement_tasks"], 3)
        self.assertEqual(report["headroom"]["cross_replica_lift_successes"], 2)
        self.assertEqual(report["headroom"]["repeat_discordance_rate"], 0)
        self.assertEqual(
            report["manipulation"]["recovery_eligible_counts"],
            {"fail_fast": 8, "diagnose_retry_once": 8},
        )
        self.assertEqual(report["headroom"]["headroom_task_count"], 10)
        self.assertEqual(
            report["manipulation"]["verification_opportunity_counts"],
            {"submit_directly": 40, "final_reobserve": 40},
        )

    def test_missing_task_is_no_go(self) -> None:
        records = self._records()[:-1]
        report = evaluate_headroom_gate(
            self.manifest,
            self.registry,
            self.policy_split,
            records,
            preflight=self.preflight,
        )
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["complete"])
        self.assertEqual(len(report["completeness"]["missing_panels"]), 1)

    def test_missing_runtime_preflight_is_no_go(self) -> None:
        report = evaluate_headroom_gate(
            self.manifest,
            self.registry,
            self.policy_split,
            self._records(),
            preflight=None,
        )
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["complete"])

    def test_policy_verifier_and_trajectory_drift_are_no_go(self) -> None:
        mutations = (
            lambda item: item.update(policy={}),
            lambda item: item["verification"].update(verifier_version="drifted"),
            lambda item: item.update(trajectory=None),
            lambda item: item.update(sampling_receipt=None),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                records = self._records()
                first_attempt = next(iter(records[0]["result"]["attempts"].values()))
                mutate(first_attempt)
                report = evaluate_headroom_gate(
                    self.manifest,
                    self.registry,
                    self.policy_split,
                    records,
                    preflight=self.preflight,
                )
                self.assertFalse(report["passed"])
                self.assertFalse(report["checks"]["complete"])

    def test_uniform_tie_score_is_not_label_order_dependent(self) -> None:
        policies = {
            "A": {0: True, 1: False},
            "B": {0: True, 1: True},
            "C": {0: False, 1: False},
            "D": {0: False, 1: False},
        }
        self.assertEqual(
            _uniform_tie_score(policies, ["A", "B", "C", "D"], 0, 1),
            0.5,
        )
        self.assertEqual(
            _uniform_tie_score(policies, ["B", "A", "D", "C"], 0, 1),
            0.5,
        )

    def test_wrong_recovery_probe_is_no_go(self) -> None:
        records = self._records()
        recovery_record = next(
            record
            for record in records
            if record["task"]["template"] == "distractor_recovery"
        )
        for attempt in recovery_record["result"]["attempts"].values():
            attempt["trajectory"]["tool_trace"][2]["details"]["url"] = "wrong"
        report = evaluate_headroom_gate(
            self.manifest,
            self.registry,
            self.policy_split,
            records,
            preflight=self.preflight,
        )
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["recovery_separation"])


if __name__ == "__main__":
    unittest.main()
