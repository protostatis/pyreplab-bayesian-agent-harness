from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pyreplab_harness.orchestrator import (
    RemoteConfig,
    _pair_order,
    _run_pi,
    _run_policy_set,
    build_parser,
    policy_spec,
    policy_spec_from_treatment,
    run_registered_treatments,
    validate_remote_config,
)
from pyreplab_harness.treatments import TreatmentRegistry, TreatmentSpec, generate_treatments


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PolicyVersionTest(unittest.TestCase):
    def test_v1_policy_is_preserved(self) -> None:
        direct = policy_spec(PROJECT_ROOT, "direct", "1")
        self.assertEqual(direct.version, "1")
        self.assertEqual(direct.max_output_tokens, 1536)
        self.assertEqual(direct.tool_call_limit, 6)

    def test_v2_policy_is_lean_and_separately_versioned(self) -> None:
        direct = policy_spec(PROJECT_ROOT, "direct", "2")
        deliberate = policy_spec(PROJECT_ROOT, "deliberate", "2")
        self.assertEqual(direct.version, "2")
        self.assertEqual(deliberate.version, "2")
        self.assertLess(direct.max_output_tokens, deliberate.max_output_tokens)
        self.assertLess(direct.tool_call_limit, deliberate.tool_call_limit)
        self.assertIn("Remove temporary or helper files", deliberate.system_prompt)

    def test_v3_restores_working_headroom_with_hard_stop_versioning(self) -> None:
        direct = policy_spec(PROJECT_ROOT, "direct", "3")
        deliberate = policy_spec(PROJECT_ROOT, "deliberate", "3")
        self.assertEqual(direct.version, "3")
        self.assertEqual(deliberate.version, "3")
        self.assertEqual(direct.max_output_tokens, 1536)
        self.assertEqual(deliberate.max_output_tokens, 2560)
        self.assertEqual(direct.tool_call_limit, 7)
        self.assertEqual(deliberate.tool_call_limit, 8)
        self.assertIn("probe tool versions", direct.system_prompt)

    def test_v4_pins_corrected_hard_stop_treatment(self) -> None:
        direct = policy_spec(PROJECT_ROOT, "direct", "4")
        deliberate = policy_spec(PROJECT_ROOT, "deliberate", "4")
        self.assertEqual(direct.version, "4")
        self.assertEqual(deliberate.version, "4")
        self.assertEqual(direct.tool_call_limit, 7)
        self.assertEqual(deliberate.tool_call_limit, 8)

    def test_rejects_unknown_policy_version(self) -> None:
        with self.assertRaisesRegex(ValueError, "policy version"):
            policy_spec(PROJECT_ROOT, "direct", "99")

    def test_pair_order_is_deterministic_for_task_key(self) -> None:
        first = _pair_order("artifact-hard-21", ["direct", "deliberate"])
        second = _pair_order("artifact-hard-21", ["direct", "deliberate"])
        self.assertEqual(first, second)
        self.assertCountEqual(first, ["direct", "deliberate"])

    def test_parser_accepts_policy_version(self) -> None:
        args = build_parser().parse_args(["--policy-version", "4"])
        self.assertEqual(args.policy_version, "4")

    def test_remote_paths_must_be_explicit_and_absolute(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit absolute remote path"):
            validate_remote_config(RemoteConfig("host", "", ""))
        with self.assertRaisesRegex(ValueError, "other than '/'"):
            validate_remote_config(RemoteConfig("host", "/project", "/"))
        validate_remote_config(RemoteConfig("host", "/project", "/runs"))

    def test_pi_provider_model_are_configurable_and_switch_is_optional(self) -> None:
        policy = policy_spec(PROJECT_ROOT, "direct", "1")
        completed = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch(
            "pyreplab_harness.orchestrator.subprocess.run", return_value=completed
        ) as runner:
            result = _run_pi(
                PROJECT_ROOT,
                RemoteConfig("host", "/project", "/runs"),
                "/runs/workspace",
                "task prompt",
                policy,
                "pi",
                None,
                "custom-provider",
                "custom-model",
                "off",
            )
        self.assertIs(result, completed)
        command = runner.call_args.args[0]
        self.assertEqual(command[command.index("--provider") + 1], "custom-provider")
        self.assertEqual(command[command.index("--model") + 1], "custom-model")
        self.assertNotIn("model-switch.ts", " ".join(command))


class GeneralizedTreatmentOrchestratorTest(unittest.TestCase):
    def test_registry_treatment_converts_to_budget_enforced_policy(self) -> None:
        treatment = generate_treatments(1, seed=7)[0]
        policy = policy_spec_from_treatment(treatment)
        self.assertEqual(policy.id, treatment.id)
        self.assertEqual(policy.system_prompt, treatment.system_prompt)
        self.assertEqual(policy.bundle_hash, treatment.bundle_hash)
        self.assertEqual(policy.tool_interface, "native_bash")
        self.assertTrue(policy.enforce_budget)

    def test_unsupported_interface_or_tools_fail_closed(self) -> None:
        base = dict(
            id="unsupported",
            version="1",
            system_prompt="Prompt",
            max_output_tokens=100,
            tool_call_limit=2,
            command_timeout_seconds=10,
            wall_time_limit_seconds=30,
        )
        with self.assertRaisesRegex(ValueError, "tool interface"):
            policy_spec_from_treatment(
                TreatmentSpec(
                    **base,
                    allowed_tools=("bash",),
                    tool_interface="text_protocol",
                )
            )
        with self.assertRaisesRegex(ValueError, "only allowed_tools"):
            policy_spec_from_treatment(
                TreatmentSpec(
                    **base,
                    allowed_tools=("bash", "read"),
                )
            )

    def test_parser_accepts_registry_treatment_menu(self) -> None:
        args = build_parser().parse_args(
            [
                "--treatment-registry",
                "registry.json",
                "--treatments",
                "a,b@2",
            ]
        )
        self.assertEqual(args.treatment_registry, "registry.json")
        self.assertEqual(args.treatments, "a,b@2")

    def test_registered_treatments_resolve_and_forward_registry_hash(self) -> None:
        treatments = generate_treatments(3, seed=13)
        registry = TreatmentRegistry(tuple(treatments))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            registry.save(path)
            args = argparse.Namespace(
                treatment_registry=str(path),
                treatments=f"{treatments[0].id},{treatments[1].bundle_id}",
            )
            config = RemoteConfig("host", "/project", "/runs")
            with mock.patch(
                "pyreplab_harness.orchestrator._run_policy_set",
                return_value={"mode": "treatment_set"},
            ) as runner:
                result = run_registered_treatments(PROJECT_ROOT, config, args)
            self.assertEqual(result["mode"], "treatment_set")
            selected = runner.call_args.args[3]
            self.assertEqual(
                set(selected), {treatments[0].bundle_id, treatments[1].bundle_id}
            )
            self.assertEqual(
                runner.call_args.kwargs["registry_hash"], registry.registry_hash
            )

    def test_unknown_or_duplicate_treatment_selection_is_rejected(self) -> None:
        treatment = generate_treatments(1, seed=15)[0]
        registry = TreatmentRegistry((treatment,))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            registry.save(path)
            config = RemoteConfig("host", "/project", "/runs")
            unknown = argparse.Namespace(
                treatment_registry=str(path), treatments="not-present"
            )
            with self.assertRaisesRegex(ValueError, "unknown treatment"):
                run_registered_treatments(PROJECT_ROOT, config, unknown)
            duplicate = argparse.Namespace(
                treatment_registry=str(path),
                treatments=f"{treatment.id},{treatment.bundle_id}",
            )
            with self.assertRaisesRegex(ValueError, "duplicate treatment"):
                run_registered_treatments(PROJECT_ROOT, config, duplicate)

    def test_policy_set_runs_every_bundle_on_one_task(self) -> None:
        treatments = generate_treatments(2, seed=21)
        policies = {
            treatment.bundle_id: policy_spec_from_treatment(treatment)
            for treatment in treatments
        }
        args = argparse.Namespace(seed=3, family="artifact")
        config = RemoteConfig("host", "/project", "/runs")

        def fake_attempt(_root, _config, task, policy, attempt_id, _args, **_kwargs):
            return {
                "attempt_id": attempt_id,
                "policy": policy.to_dict(),
                "pi_return_code": 0,
                "pi_stderr": "",
                "verification": {"success": True},
                "usage": {},
                "trajectory": {},
                "timing": {},
                "task_id": task["id"],
            }

        with mock.patch(
            "pyreplab_harness.orchestrator._task_json",
            return_value={"id": "artifact-task-3", "prompt": "task"},
        ), mock.patch(
            "pyreplab_harness.orchestrator._run_attempt", side_effect=fake_attempt
        ) as attempt:
            result = _run_policy_set(
                PROJECT_ROOT,
                config,
                args,
                policies,
                mode="treatment_set",
                registry_hash="registry-hash",
            )
        self.assertEqual(attempt.call_count, 2)
        self.assertEqual(result["task_id"], "artifact-task-3")
        self.assertEqual(set(result["attempts"]), set(policies))
        self.assertEqual(result["treatment_registry_hash"], "registry-hash")


if __name__ == "__main__":
    unittest.main()
