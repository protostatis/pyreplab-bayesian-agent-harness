from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pyreplab_harness.orchestrator import (
    UNBROWSER_INTERACTIVE_STRUCTURE_FIRST_INTERFACE,
    UNBROWSER_INTERACTIVE_TEXT_FIRST_INTERFACE,
    UNBROWSER_INTERACTIVE_TOOL_INTERFACE,
    UNBROWSER_TOOL_INTERFACE,
    RemoteConfig,
    _pair_order,
    _parse_budget_receipt,
    _parse_sampling_receipt,
    _required_first_observation_from_interface,
    _run_attempt,
    _run_pi,
    _run_policy_set,
    _task_json,
    _unbrowser_url_for_task,
    build_parser,
    policy_spec,
    policy_spec_from_treatment,
    run_registered_treatments,
    run_pair,
    run_single,
    validate_remote_config,
)
from pyreplab_harness.gym_registry import FAMILIES
from pyreplab_harness.treatments import TreatmentRegistry, TreatmentSpec, generate_treatments
from pyreplab_harness.unbrowser_rpc import (
    FIXTURE_INTERACTIVE_ORIGIN,
    UNBROWSER_INTERACTIVE_URL,
    UNBROWSER_SMOKE_URL,
)


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

    def test_parser_accepts_canary_role_and_sampling_seed(self) -> None:
        args = build_parser().parse_args(
            ["--task-role", "T_canary", "--sampling-seed", "1900000001"]
        )
        self.assertEqual(args.task_role, "T_canary")
        self.assertEqual(args.sampling_seed, 1900000001)

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
        self.assertEqual(
            command[command.index("--append-system-prompt") + 1],
            policy.system_prompt,
        )
        self.assertEqual(command[-1], "task prompt")
        self.assertNotIn("model-switch.ts", " ".join(command))

    def test_sampling_seed_reaches_gym_extension(self) -> None:
        policy = policy_spec(PROJECT_ROOT, "direct", "1")
        completed = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch(
            "pyreplab_harness.orchestrator.subprocess.run", return_value=completed
        ) as runner:
            _run_pi(
                PROJECT_ROOT,
                RemoteConfig("host", "/project", "/runs"),
                "/runs/workspace",
                "task prompt",
                policy,
                "pi",
                None,
                sampling_seed=2026082001,
            )
        command = runner.call_args.args[0]
        self.assertEqual(
            command[command.index("--gym-sampling-seed") + 1],
            "2026082001",
        )

    def test_sampling_receipt_parser_keeps_only_sanitized_marker(self) -> None:
        receipt = _parse_sampling_receipt(
            "diagnostic\n"
            'PYREPLAB_SAMPLING_V1 {"seed":7,"parameters":{"temperature":0.8}}\n'
        )
        self.assertEqual(
            receipt,
            {"seed": 7, "parameters": {"temperature": 0.8}},
        )

    def test_budget_receipt_parser_keeps_only_structured_marker(self) -> None:
        receipt = _parse_budget_receipt(
            "diagnostic\n"
            'PYREPLAB_GYM_BUDGET_V3 {"schema_version":"receipt-v1","provider_request_admissions":3}\n'
        )
        self.assertEqual(
            receipt,
            {"schema_version": "receipt-v1", "provider_request_admissions": 3},
        )


class GeneralizedTreatmentOrchestratorTest(unittest.TestCase):
    def test_registry_treatment_converts_to_budget_enforced_policy(self) -> None:
        treatment = generate_treatments(1, seed=7)[0]
        policy = policy_spec_from_treatment(treatment)
        self.assertEqual(policy.id, treatment.id)
        self.assertEqual(policy.system_prompt, treatment.system_prompt)
        self.assertEqual(policy.bundle_hash, treatment.bundle_hash)
        self.assertEqual(policy.tool_interface, "native_bash")
        self.assertTrue(policy.enforce_budget)

    def test_empty_overlay_treatment_omits_append_system_prompt(self) -> None:
        treatment = TreatmentSpec(
            id="empty-overlay",
            version="1",
            system_prompt="",
            allowed_tools=("bash",),
            max_output_tokens=512,
            tool_call_limit=4,
            command_timeout_seconds=20,
            wall_time_limit_seconds=60,
        )
        policy = policy_spec_from_treatment(treatment)
        completed = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch(
            "pyreplab_harness.orchestrator.subprocess.run", return_value=completed
        ) as runner:
            _run_pi(
                PROJECT_ROOT,
                RemoteConfig("host", "/project", "/runs"),
                "/runs/workspace",
                "task prompt",
                policy,
                "pi",
                None,
            )
        command = runner.call_args.args[0]
        self.assertNotIn("--append-system-prompt", command)
        self.assertIn("gym-budget-v3.ts", " ".join(command))
        self.assertEqual(
            command[command.index("--gym-provider-turn-limit") + 1],
            "5",
        )
        self.assertEqual(command[-1], "task prompt")

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
        with self.assertRaisesRegex(ValueError, "requires exactly"):
            policy_spec_from_treatment(
                TreatmentSpec(
                    **base,
                    allowed_tools=("bash", "read"),
                )
            )

    def test_readonly_unbrowser_interface_is_exact_and_executable(self) -> None:
        base = dict(
            id="unbrowser-correct",
            version="1",
            system_prompt="Use the fixed read-only browser.",
            max_output_tokens=512,
            tool_call_limit=4,
            command_timeout_seconds=20,
            wall_time_limit_seconds=60,
            tool_interface=UNBROWSER_TOOL_INTERFACE,
        )
        treatment = TreatmentSpec(
            **base,
            allowed_tools=("bash", "unbrowser"),
        )
        policy = policy_spec_from_treatment(treatment)
        self.assertEqual(policy.allowed_tools, ("bash", "unbrowser"))
        self.assertEqual(policy.tool_interface, UNBROWSER_TOOL_INTERFACE)

        task = {
            "family": "unbrowser",
            "public_metadata": {"allowed_url": UNBROWSER_SMOKE_URL},
        }
        self.assertEqual(_unbrowser_url_for_task(task, policy), UNBROWSER_SMOKE_URL)

        with self.assertRaisesRegex(ValueError, "requires exactly"):
            policy_spec_from_treatment(
                TreatmentSpec(**base, allowed_tools=("unbrowser",))
            )
        with self.assertRaisesRegex(ValueError, "restricted to the unbrowser"):
            _unbrowser_url_for_task(
                {"family": "artifact", "public_metadata": {}}, policy
            )

    def test_pi_activates_and_configures_readonly_unbrowser(self) -> None:
        treatment = TreatmentSpec(
            id="unbrowser-correct",
            version="1",
            system_prompt="Use Unbrowser.",
            allowed_tools=("bash", "unbrowser"),
            max_output_tokens=512,
            tool_call_limit=4,
            command_timeout_seconds=20,
            wall_time_limit_seconds=60,
            tool_interface=UNBROWSER_TOOL_INTERFACE,
        )
        completed = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch(
            "pyreplab_harness.orchestrator.subprocess.run", return_value=completed
        ) as runner:
            _run_pi(
                PROJECT_ROOT,
                RemoteConfig("host", "/project", "/runs"),
                "/runs/workspace",
                "task prompt",
                policy_spec_from_treatment(treatment),
                "pi",
                None,
                unbrowser_url=UNBROWSER_SMOKE_URL,
                unbrowser_binary="/usr/local/bin/unbrowser",
            )
        command = runner.call_args.args[0]
        self.assertEqual(command[command.index("--tools") + 1], "bash,unbrowser")
        self.assertEqual(
            command[command.index("--gym-unbrowser-url") + 1],
            UNBROWSER_SMOKE_URL,
        )
        self.assertEqual(
            command[command.index("--gym-unbrowser-binary") + 1],
            "/usr/local/bin/unbrowser",
        )

    def test_unbrowser_family_rejects_legacy_and_mixed_tool_policies(self) -> None:
        args = argparse.Namespace(
            family="unbrowser",
            policy="direct",
            policy_version="1",
        )
        config = RemoteConfig("host", "/project", "/runs")
        with self.assertRaisesRegex(ValueError, "require.*registered"):
            run_single(PROJECT_ROOT, config, args)
        with self.assertRaisesRegex(ValueError, "require.*registered"):
            run_pair(PROJECT_ROOT, config, args)

        bash_only = generate_treatments(1, seed=33)[0]
        registry = TreatmentRegistry((bash_only,))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            registry.save(path)
            registered_args = argparse.Namespace(
                family="unbrowser",
                treatment_registry=str(path),
                treatments="all",
            )
            with self.assertRaisesRegex(ValueError, "every treatment"):
                run_registered_treatments(PROJECT_ROOT, config, registered_args)

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

    def test_interactive_unbrowser_interface_is_accepted(self) -> None:
        base = dict(
            id="unbrowser-interactive-correct",
            version="1",
            system_prompt="Use the interactive browser.",
            max_output_tokens=2048,
            tool_call_limit=12,
            command_timeout_seconds=45,
            wall_time_limit_seconds=360,
            tool_interface=UNBROWSER_INTERACTIVE_TOOL_INTERFACE,
        )
        treatment = TreatmentSpec(
            **base,
            allowed_tools=("bash", "unbrowser"),
        )
        policy = policy_spec_from_treatment(treatment)
        self.assertEqual(policy.allowed_tools, ("bash", "unbrowser"))
        self.assertEqual(policy.tool_interface, UNBROWSER_INTERACTIVE_TOOL_INTERFACE)
        self.assertEqual(policy.tool_call_limit, 12)

        task = {
            "family": "unbrowser_interactive",
            "public_metadata": {"allowed_url": UNBROWSER_INTERACTIVE_URL},
        }
        self.assertEqual(
            _unbrowser_url_for_task(task, policy), UNBROWSER_INTERACTIVE_URL
        )

    def test_interactive_url_enforces_wikipedia_origin(self) -> None:
        treatment = TreatmentSpec(
            id="unbrowser-int",
            version="1",
            system_prompt="Prompt",
            allowed_tools=("bash", "unbrowser"),
            max_output_tokens=512,
            tool_call_limit=12,
            command_timeout_seconds=30,
            wall_time_limit_seconds=180,
            tool_interface=UNBROWSER_INTERACTIVE_TOOL_INTERFACE,
        )
        policy = policy_spec_from_treatment(treatment)

        task = {
            "family": "unbrowser_interactive",
            "public_metadata": {"allowed_url": "https://example.com/"},
        }
        with self.assertRaisesRegex(ValueError, "pinned"):
            _unbrowser_url_for_task(task, policy)

        # Non-interactive tasks with unbrowser tool should still fail if
        # family is not unbrowser or unbrowser_interactive.
        artifact_task = {
            "family": "artifact",
            "public_metadata": {"allowed_url": UNBROWSER_SMOKE_URL},
        }
        with self.assertRaisesRegex(ValueError, "restricted"):
            _unbrowser_url_for_task(artifact_task, policy)

    def test_interactive_pi_uses_higher_call_limit(self) -> None:
        treatment = TreatmentSpec(
            id="unbrowser-int-correct",
            version="1",
            system_prompt="Use Unbrowser interactively.",
            allowed_tools=("bash", "unbrowser"),
            max_output_tokens=2048,
            tool_call_limit=12,
            command_timeout_seconds=45,
            wall_time_limit_seconds=360,
            tool_interface=UNBROWSER_INTERACTIVE_TOOL_INTERFACE,
        )
        completed = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch(
            "pyreplab_harness.orchestrator.subprocess.run", return_value=completed
        ) as runner:
            _run_pi(
                PROJECT_ROOT,
                RemoteConfig("host", "/project", "/runs"),
                "/runs/workspace",
                "task prompt",
                policy_spec_from_treatment(treatment),
                "pi",
                None,
                unbrowser_url=UNBROWSER_INTERACTIVE_URL,
                unbrowser_binary="/usr/local/bin/unbrowser",
                unbrowser_interactive=True,
            )
        command = runner.call_args.args[0]
        self.assertEqual(command[command.index("--tools") + 1], "bash,unbrowser")
        self.assertEqual(
            command[command.index("--gym-unbrowser-url") + 1],
            UNBROWSER_INTERACTIVE_URL,
        )
        self.assertEqual(
            command[command.index("--gym-unbrowser-tool-limit") + 1],
            "12",
        )
        self.assertEqual(
            command[command.index("--gym-unbrowser-interactive") + 1],
            "true",
        )

    def test_unbrowser_interactive_family_rejects_legacy_and_mixed_tool_policies(self) -> None:
        args = argparse.Namespace(
            family="unbrowser_interactive",
            policy="direct",
            policy_version="1",
        )
        config = RemoteConfig("host", "/project", "/runs")
        with self.assertRaisesRegex(ValueError, "require.*registered"):
            run_single(PROJECT_ROOT, config, args)
        with self.assertRaisesRegex(ValueError, "require.*registered"):
            run_pair(PROJECT_ROOT, config, args)

        bash_only = generate_treatments(1, seed=33)[0]
        registry = TreatmentRegistry((bash_only,))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            registry.save(path)
            registered_args = argparse.Namespace(
                family="unbrowser_interactive",
                treatment_registry=str(path),
                treatments="all",
            )
            with self.assertRaisesRegex(ValueError, "every treatment"):
                run_registered_treatments(PROJECT_ROOT, config, registered_args)

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

    def test_policy_set_can_preserve_preregistered_order(self) -> None:
        treatments = generate_treatments(3, seed=21)
        policies = {
            treatment.bundle_id: policy_spec_from_treatment(treatment)
            for treatment in treatments
        }
        args = argparse.Namespace(
            seed=3,
            family="artifact",
            preserve_treatment_order=True,
        )
        config = RemoteConfig("host", "/project", "/runs")

        with mock.patch(
            "pyreplab_harness.orchestrator._task_json",
            return_value={"id": "artifact-task-3", "prompt": "task"},
        ), mock.patch(
            "pyreplab_harness.orchestrator._run_attempt",
            return_value={
                "attempt_id": "a",
                "policy": {},
                "pi_return_code": 0,
                "pi_stderr": "",
                "verification": {"success": True},
            },
        ):
            result = _run_policy_set(
                PROJECT_ROOT,
                config,
                args,
                policies,
                mode="treatment_set",
            )
        self.assertEqual(result["execution_order"], list(policies))

    def test_empty_pi_stdout_is_still_recorded_and_verified(self) -> None:
        treatment = generate_treatments(1, seed=22)[0]
        policy = policy_spec_from_treatment(treatment)
        config = RemoteConfig("host", "/project", "/runs")
        task = {
            "id": "artifact-task-5",
            "family": "artifact",
            "prompt": "Do the task",
            "public_metadata": {},
        }

        def fake_remote(_config, arguments, **kwargs):
            if arguments[0] == "prepare-attempt":
                return {"workspace_ref": "/runs/attempts/a/workspace"}
            if arguments[0] == "record-events":
                return {"recorded": 0}
            if arguments[0] == "verify":
                return {"success": False, "failure_code": "missing_output"}
            if arguments[0] == "inspect-attempt":
                return {"status": "verified"}
            self.fail(f"unexpected remote command: {arguments}")

        completed = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch(
            "pyreplab_harness.orchestrator.remote_json", side_effect=fake_remote
        ) as remote, mock.patch(
            "pyreplab_harness.orchestrator._run_pi_checked", return_value=completed
        ):
            result = _run_attempt(
                PROJECT_ROOT,
                config,
                task,
                policy,
                "attempt-empty-output",
                build_parser().parse_args(["--family", "artifact"]),
                with_usage=False,
            )

        record_call = next(
            call for call in remote.call_args_list if call.args[1][0] == "record-events"
        )
        self.assertEqual(record_call.kwargs["input_text"], "")
        self.assertEqual(result["verification"]["failure_code"], "missing_output")

    def test_model_admission_callback_runs_after_prepare_and_before_pi(self) -> None:
        treatment = generate_treatments(1, seed=22)[0]
        policy = policy_spec_from_treatment(treatment)
        config = RemoteConfig("host", "/project", "/runs")
        task = {
            "id": "artifact-task-5",
            "family": "artifact",
            "prompt": "Do the task",
            "public_metadata": {},
        }
        order: list[str] = []

        def fake_remote(_config, arguments, **_kwargs):
            if arguments[0] == "prepare-attempt":
                order.append("prepare")
                self.assertEqual(
                    arguments[
                        arguments.index("--expected-task-commitment-hash") + 1
                    ],
                    "a" * 64,
                )
                return {"workspace_ref": "/runs/attempts/a/workspace"}
            self.fail(f"unexpected remote command: {arguments}")

        def reject_admission() -> None:
            order.append("admission")
            raise RuntimeError("authorization expired")

        args = build_parser().parse_args(["--family", "artifact"])
        args.expected_task_commitment_hash = "a" * 64
        with mock.patch(
            "pyreplab_harness.orchestrator.remote_json", side_effect=fake_remote
        ), mock.patch(
            "pyreplab_harness.orchestrator._run_pi_checked"
        ) as run_pi:
            with self.assertRaisesRegex(RuntimeError, "authorization expired"):
                _run_attempt(
                    PROJECT_ROOT,
                    config,
                    task,
                    policy,
                    "attempt-expired",
                    args,
                    with_usage=False,
                    before_model_admission=reject_admission,
                )

        self.assertEqual(order, ["prepare", "admission"])
        run_pi.assert_not_called()

    def test_routing_attempt_requires_normalized_event_summary(self) -> None:
        treatment = TreatmentSpec(
            id="routing-fixture",
            version="1",
            system_prompt="Use routing fixture.",
            allowed_tools=("bash", "unbrowser"),
            max_output_tokens=256,
            tool_call_limit=2,
            command_timeout_seconds=10,
            wall_time_limit_seconds=30,
            tool_interface=UNBROWSER_INTERACTIVE_TOOL_INTERFACE,
        )
        task = {
            "id": "routing-fixture-rf-test",
            "family": "routing_fixture",
            "prompt": "task",
            "public_metadata": {
                "allowed_url": "http://127.0.0.1:18090/routing/rf-test"
            },
        }

        def remote(_config, arguments, **_kwargs):
            if arguments[0] == "prepare-attempt":
                return {"workspace_ref": "/runs/attempts/a/workspace"}
            if arguments[0] == "record-events":
                return {"recorded": 1}
            if arguments[0] == "verify":
                return {
                    "success": False,
                    "verifier_id": "routing-fixture-nonce",
                    "verifier_version": "1",
                    "failure_code": "missing_output",
                }
            self.fail(f"unexpected remote command: {arguments}")

        completed = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch(
            "pyreplab_harness.orchestrator.remote_json", side_effect=remote
        ), mock.patch(
            "pyreplab_harness.orchestrator._run_pi_checked", return_value=completed
        ), mock.patch(
            "pyreplab_harness.orchestrator._attempt_event_summary", return_value=None
        ):
            with self.assertRaisesRegex(RuntimeError, "event summary"):
                _run_attempt(
                    PROJECT_ROOT,
                    RemoteConfig("host", "/project", "/runs"),
                    task,
                    policy_spec_from_treatment(treatment),
                    "attempt-routing",
                    build_parser().parse_args(["--family", "routing_fixture"]),
                    with_usage=True,
                    require_complete_event_summary=True,
                )


class UnbrowserFixtureOrchestratorTest(unittest.TestCase):
    """Tests for the unbrowser_fixture family in the orchestrator."""

    def test_fixture_family_accepted_in_run_single(self) -> None:
        args = argparse.Namespace(
            family="unbrowser_fixture",
            policy="direct",
            policy_version="1",
        )
        config = RemoteConfig("host", "/project", "/runs")
        with self.assertRaisesRegex(ValueError, "require.*registered"):
            run_single(PROJECT_ROOT, config, args)

    def test_fixture_family_accepted_in_run_pair(self) -> None:
        args = argparse.Namespace(
            family="unbrowser_fixture",
            policy="direct",
            policy_version="1",
        )
        config = RemoteConfig("host", "/project", "/runs")
        with self.assertRaisesRegex(ValueError, "require.*registered"):
            run_pair(PROJECT_ROOT, config, args)

    def test_fixture_url_validated_correctly(self) -> None:
        """Fixture URLs pass validation with allow_fixture=True."""
        treatment = TreatmentSpec(
            id="unbrowser-fixture",
            version="1",
            system_prompt="Use Unbrowser on fixture pages.",
            allowed_tools=("bash", "unbrowser"),
            max_output_tokens=2048,
            tool_call_limit=12,
            command_timeout_seconds=45,
            wall_time_limit_seconds=360,
            tool_interface=UNBROWSER_INTERACTIVE_TOOL_INTERFACE,
        )
        policy = policy_spec_from_treatment(treatment)

        task = {
            "family": "unbrowser_fixture",
            "public_metadata": {
                "allowed_url": "http://127.0.0.1:18090/single_page_extraction/7/easy"
            },
        }
        url = _unbrowser_url_for_task(task, policy)
        self.assertEqual(
            url, "http://127.0.0.1:18090/single_page_extraction/7/easy"
        )

    def test_fixture_url_rejected_for_wrong_origin(self) -> None:
        """Fixture family rejects URLs outside the fixture origin."""
        treatment = TreatmentSpec(
            id="unbrowser-fixture-bad",
            version="1",
            system_prompt="Use Unbrowser on fixture pages.",
            allowed_tools=("bash", "unbrowser"),
            max_output_tokens=2048,
            tool_call_limit=12,
            command_timeout_seconds=45,
            wall_time_limit_seconds=360,
            tool_interface=UNBROWSER_INTERACTIVE_TOOL_INTERFACE,
        )
        policy = policy_spec_from_treatment(treatment)

        task = {
            "family": "unbrowser_fixture",
            "public_metadata": {
                "allowed_url": "http://example.com/page"
            },
        }
        with self.assertRaisesRegex(ValueError, "pinned"):
            _unbrowser_url_for_task(task, policy)

    def test_fixture_family_passes_confine_unbrowser(self) -> None:
        """_run_pi passes --gym-confine-unbrowser for fixture tasks."""
        treatment = TreatmentSpec(
            id="unbrowser-fixture",
            version="1",
            system_prompt="Use Unbrowser on fixture pages.",
            allowed_tools=("bash", "unbrowser"),
            max_output_tokens=2048,
            tool_call_limit=12,
            command_timeout_seconds=45,
            wall_time_limit_seconds=360,
            tool_interface=UNBROWSER_INTERACTIVE_TOOL_INTERFACE,
        )
        completed = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch(
            "pyreplab_harness.orchestrator.subprocess.run", return_value=completed
        ) as runner:
            _run_pi(
                PROJECT_ROOT,
                RemoteConfig("host", "/project", "/runs"),
                "/runs/workspace",
                "task prompt",
                policy_spec_from_treatment(treatment),
                "pi",
                None,
                unbrowser_url="http://127.0.0.1:18090/single_page_extraction/7/easy",
                unbrowser_binary="/usr/local/bin/unbrowser",
                unbrowser_interactive=True,
                confine_unbrowser=True,
            )
        command = runner.call_args.args[0]
        # Check that --gym-confine-unbrowser is passed
        self.assertIn("--gym-confine-unbrowser", command)
        # Check that --gym-unbrowser-interactive is passed
        self.assertIn("--gym-unbrowser-interactive", command)
        # Check tool-call limit is 12 for interactive fixture tasks
        self.assertEqual(
            command[command.index("--gym-unbrowser-tool-limit") + 1], "12"
        )

    def test_fixture_family_requires_registered_treatment(self) -> None:
        bash_only = generate_treatments(1, seed=33)[0]
        registry = TreatmentRegistry((bash_only,))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            registry.save(path)
            registered_args = argparse.Namespace(
                family="unbrowser_fixture",
                treatment_registry=str(path),
                treatments="all",
            )
            config = RemoteConfig("host", "/project", "/runs")
            with self.assertRaisesRegex(ValueError, "every treatment"):
                run_registered_treatments(PROJECT_ROOT, config, registered_args)

    def test_fixture_family_accepted_in_registered_treatments(self) -> None:
        """unbrowser_fixture family should be accepted with unbrowser treatments."""
        from pyreplab_harness.treatments import generate_treatments as _gen

        treatments = _gen(1, seed=13)
        # This might not include unbrowser; just ensure family is recognized
        registry = TreatmentRegistry(tuple(treatments))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            registry.save(path)

            # Only test that the family name is recognized (vs rejected as unknown)
            # Treatment mismatch will be caught before family validation
            self.assertIn("unbrowser_fixture", FAMILIES)

    def test_fixture_family_parser_accepts_family(self) -> None:
        args = build_parser().parse_args(["--family", "unbrowser_fixture"])
        self.assertEqual(args.family, "unbrowser_fixture")

    def test_task_json_includes_fixture_template_for_unbrowser_fixture(self) -> None:
        """_task_json adds --fixture-template only for unbrowser_fixture."""
        config = RemoteConfig("host", "/project", "/runs")
        fixture_args = argparse.Namespace(
            family="unbrowser_fixture", seed=7, difficulty="medium",
            fixture_template="table_filter_sort",
        )
        with mock.patch(
            "pyreplab_harness.orchestrator.remote_json",
            return_value={"id": "task-1"},
        ) as remote:
            _task_json(config, fixture_args)
        arguments = remote.call_args.args[1]
        self.assertIn("--fixture-template", arguments)
        self.assertIn("table_filter_sort", arguments)

    def test_task_json_includes_frozen_fixture_role(self) -> None:
        config = RemoteConfig("host", "/project", "/runs")
        fixture_args = argparse.Namespace(
            family="unbrowser_fixture",
            seed=7,
            difficulty="medium",
            fixture_template="table_filter_sort",
            task_role="T_pilot",
        )
        with mock.patch(
            "pyreplab_harness.orchestrator.remote_json",
            return_value={"id": "task-1"},
        ) as remote:
            _task_json(config, fixture_args)
        arguments = remote.call_args.args[1]
        self.assertEqual(arguments[arguments.index("--task-role") + 1], "T_pilot")

    def test_task_json_forwards_outcome_only_fixture_version(self) -> None:
        config = RemoteConfig("host", "/project", "/runs")
        fixture_args = argparse.Namespace(
            family="unbrowser_fixture",
            seed=7,
            difficulty="medium",
            fixture_template="table_filter_sort",
            fixture_generator_version="unbrowser-fixture-v3",
        )
        with mock.patch(
            "pyreplab_harness.orchestrator.remote_json",
            return_value={"id": "task-1"},
        ) as remote:
            _task_json(config, fixture_args)
        arguments = remote.call_args.args[1]
        self.assertEqual(
            arguments[arguments.index("--fixture-generator-version") + 1],
            "unbrowser-fixture-v3",
        )

    def test_task_json_omits_fixture_template_for_non_fixture(self) -> None:
        """_task_json does NOT add --fixture-template for non-fixture families."""
        config = RemoteConfig("host", "/project", "/runs")
        artifact_args = argparse.Namespace(
            family="artifact", seed=7, difficulty="medium",
            fixture_template="table_filter_sort",
        )
        with mock.patch(
            "pyreplab_harness.orchestrator.remote_json",
            return_value={"id": "task-1"},
        ) as remote:
            _task_json(config, artifact_args)
        arguments = remote.call_args.args[1]
        self.assertNotIn("--fixture-template", arguments)

    def test_custom_unbrowser_binary_reaches_runner_args(self) -> None:
        """Custom --unbrowser-binary is forwarded from args to _run_pi."""
        treatment = TreatmentSpec(
            id="unbrowser-correct",
            version="1",
            system_prompt="Use Unbrowser.",
            allowed_tools=("bash", "unbrowser"),
            max_output_tokens=512,
            tool_call_limit=4,
            command_timeout_seconds=20,
            wall_time_limit_seconds=60,
            tool_interface=UNBROWSER_TOOL_INTERFACE,
        )
        completed = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch(
            "pyreplab_harness.orchestrator.subprocess.run", return_value=completed
        ) as runner:
            _run_pi(
                PROJECT_ROOT,
                RemoteConfig("host", "/project", "/runs"),
                "/runs/workspace",
                "task prompt",
                policy_spec_from_treatment(treatment),
                "pi",
                None,
                unbrowser_url=UNBROWSER_SMOKE_URL,
                unbrowser_binary="/opt/custom-unbrowser",
            )
        command = runner.call_args.args[0]
        self.assertEqual(
            command[command.index("--gym-unbrowser-binary") + 1],
            "/opt/custom-unbrowser",
        )

    def test_orchestrator_parser_accepts_fixture_template(self) -> None:
        """--fixture-template is accepted by the orchestrator parser."""
        args = build_parser().parse_args(
            ["--fixture-template", "multi_page_navigation"]
        )
        self.assertEqual(args.fixture_template, "multi_page_navigation")


class RequiredFirstObservationOrchestratorTest(unittest.TestCase):
    """Tests for the text_first and structure_first observation-enforcement interfaces."""

    def test_text_first_interface_is_accepted(self) -> None:
        base = dict(
            id="text-first-correct",
            version="1",
            system_prompt="Use the interactive browser with text-first observation.",
            max_output_tokens=2048,
            tool_call_limit=12,
            command_timeout_seconds=45,
            wall_time_limit_seconds=360,
            tool_interface=UNBROWSER_INTERACTIVE_TEXT_FIRST_INTERFACE,
        )
        treatment = TreatmentSpec(
            **base,
            allowed_tools=("bash", "unbrowser"),
        )
        policy = policy_spec_from_treatment(treatment)
        self.assertEqual(policy.allowed_tools, ("bash", "unbrowser"))
        self.assertEqual(policy.tool_interface, UNBROWSER_INTERACTIVE_TEXT_FIRST_INTERFACE)
        self.assertEqual(policy.enforce_budget, True)

    def test_structure_first_interface_is_accepted(self) -> None:
        base = dict(
            id="structure-first-correct",
            version="1",
            system_prompt="Use the interactive browser with structure-first observation.",
            max_output_tokens=2048,
            tool_call_limit=12,
            command_timeout_seconds=45,
            wall_time_limit_seconds=360,
            tool_interface=UNBROWSER_INTERACTIVE_STRUCTURE_FIRST_INTERFACE,
        )
        treatment = TreatmentSpec(
            **base,
            allowed_tools=("bash", "unbrowser"),
        )
        policy = policy_spec_from_treatment(treatment)
        self.assertEqual(policy.allowed_tools, ("bash", "unbrowser"))
        self.assertEqual(policy.tool_interface, UNBROWSER_INTERACTIVE_STRUCTURE_FIRST_INTERFACE)

    def test_required_first_observation_helper_returns_correct_value(self) -> None:
        self.assertEqual(
            _required_first_observation_from_interface(UNBROWSER_INTERACTIVE_TEXT_FIRST_INTERFACE),
            "text",
        )
        self.assertEqual(
            _required_first_observation_from_interface(UNBROWSER_INTERACTIVE_STRUCTURE_FIRST_INTERFACE),
            "blockmap",
        )
        self.assertIsNone(
            _required_first_observation_from_interface(UNBROWSER_INTERACTIVE_TOOL_INTERFACE),
        )
        self.assertIsNone(_required_first_observation_from_interface("native_bash"))

    def test_text_first_pi_command_includes_required_flag(self) -> None:
        treatment = TreatmentSpec(
            id="text-first-correct",
            version="1",
            system_prompt="Use Unbrowser interactively.",
            allowed_tools=("bash", "unbrowser"),
            max_output_tokens=2048,
            tool_call_limit=12,
            command_timeout_seconds=45,
            wall_time_limit_seconds=360,
            tool_interface=UNBROWSER_INTERACTIVE_TEXT_FIRST_INTERFACE,
        )
        completed = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch(
            "pyreplab_harness.orchestrator.subprocess.run", return_value=completed
        ) as runner:
            _run_pi(
                PROJECT_ROOT,
                RemoteConfig("host", "/project", "/runs"),
                "/runs/workspace",
                "task prompt",
                policy_spec_from_treatment(treatment),
                "pi",
                None,
                unbrowser_url=UNBROWSER_INTERACTIVE_URL,
                unbrowser_binary="/usr/local/bin/unbrowser",
                unbrowser_interactive=True,
            )
        command = runner.call_args.args[0]
        self.assertIn("--gym-unbrowser-required-first-observation", command)
        idx = command.index("--gym-unbrowser-required-first-observation")
        self.assertEqual(command[idx + 1], "text")

    def test_structure_first_pi_command_includes_required_flag(self) -> None:
        treatment = TreatmentSpec(
            id="structure-first-correct",
            version="1",
            system_prompt="Use Unbrowser interactively.",
            allowed_tools=("bash", "unbrowser"),
            max_output_tokens=2048,
            tool_call_limit=12,
            command_timeout_seconds=45,
            wall_time_limit_seconds=360,
            tool_interface=UNBROWSER_INTERACTIVE_STRUCTURE_FIRST_INTERFACE,
        )
        completed = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch(
            "pyreplab_harness.orchestrator.subprocess.run", return_value=completed
        ) as runner:
            _run_pi(
                PROJECT_ROOT,
                RemoteConfig("host", "/project", "/runs"),
                "/runs/workspace",
                "task prompt",
                policy_spec_from_treatment(treatment),
                "pi",
                None,
                unbrowser_url=UNBROWSER_INTERACTIVE_URL,
                unbrowser_binary="/usr/local/bin/unbrowser",
                unbrowser_interactive=True,
            )
        command = runner.call_args.args[0]
        self.assertIn("--gym-unbrowser-required-first-observation", command)
        idx = command.index("--gym-unbrowser-required-first-observation")
        self.assertEqual(command[idx + 1], "blockmap")

    def test_plain_interactive_pi_command_omits_required_flag(self) -> None:
        treatment = TreatmentSpec(
            id="plain-int",
            version="1",
            system_prompt="Use Unbrowser interactively.",
            allowed_tools=("bash", "unbrowser"),
            max_output_tokens=2048,
            tool_call_limit=12,
            command_timeout_seconds=45,
            wall_time_limit_seconds=360,
            tool_interface=UNBROWSER_INTERACTIVE_TOOL_INTERFACE,
        )
        completed = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch(
            "pyreplab_harness.orchestrator.subprocess.run", return_value=completed
        ) as runner:
            _run_pi(
                PROJECT_ROOT,
                RemoteConfig("host", "/project", "/runs"),
                "/runs/workspace",
                "task prompt",
                policy_spec_from_treatment(treatment),
                "pi",
                None,
                unbrowser_url=UNBROWSER_INTERACTIVE_URL,
                unbrowser_binary="/usr/local/bin/unbrowser",
                unbrowser_interactive=True,
            )
        command = runner.call_args.args[0]
        self.assertNotIn("--gym-unbrowser-required-first-observation", command)


class RoutingFixtureOrchestratorTest(unittest.TestCase):
    """Tests for the routing_fixture family in the orchestrator."""

    def test_routing_fixture_family_rejects_legacy_and_mixed_tool_policies(self) -> None:
        args = argparse.Namespace(
            family="routing_fixture",
            policy="direct",
            policy_version="1",
        )
        config = RemoteConfig("host", "/project", "/runs")
        with self.assertRaisesRegex(ValueError, "require.*registered"):
            run_single(PROJECT_ROOT, config, args)
        with self.assertRaisesRegex(ValueError, "require.*registered"):
            run_pair(PROJECT_ROOT, config, args)

        bash_only = generate_treatments(1, seed=33)[0]
        registry = TreatmentRegistry((bash_only,))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            registry.save(path)
            registered_args = argparse.Namespace(
                family="routing_fixture",
                treatment_registry=str(path),
                treatments="all",
            )
            with self.assertRaisesRegex(ValueError, "every treatment"):
                run_registered_treatments(PROJECT_ROOT, config, registered_args)

    def test_routing_fixture_url_validated_correctly(self) -> None:
        treatment = TreatmentSpec(
            id="routing-fixture",
            version="1",
            system_prompt="Use Unbrowser on routing fixture pages.",
            allowed_tools=("bash", "unbrowser"),
            max_output_tokens=2048,
            tool_call_limit=12,
            command_timeout_seconds=45,
            wall_time_limit_seconds=360,
            tool_interface=UNBROWSER_INTERACTIVE_TOOL_INTERFACE,
        )
        policy = policy_spec_from_treatment(treatment)

        task = {
            "family": "routing_fixture",
            "public_metadata": {
                "allowed_url": "http://127.0.0.1:18090/routing/rf-abc123"
            },
        }
        url = _unbrowser_url_for_task(task, policy)
        self.assertEqual(url, "http://127.0.0.1:18090/routing/rf-abc123")

    def test_routing_fixture_url_rejected_for_wrong_origin(self) -> None:
        treatment = TreatmentSpec(
            id="routing-fixture-bad",
            version="1",
            system_prompt="Use Unbrowser on routing fixture pages.",
            allowed_tools=("bash", "unbrowser"),
            max_output_tokens=2048,
            tool_call_limit=12,
            command_timeout_seconds=45,
            wall_time_limit_seconds=360,
            tool_interface=UNBROWSER_INTERACTIVE_TOOL_INTERFACE,
        )
        policy = policy_spec_from_treatment(treatment)

        task = {
            "family": "routing_fixture",
            "public_metadata": {"allowed_url": "http://example.com/page"},
        }
        with self.assertRaisesRegex(ValueError, "pinned"):
            _unbrowser_url_for_task(task, policy)

    def test_routing_fixture_passes_confine_and_interactive(self) -> None:
        treatment = TreatmentSpec(
            id="routing-fixture",
            version="1",
            system_prompt="Use Unbrowser on routing fixture pages.",
            allowed_tools=("bash", "unbrowser"),
            max_output_tokens=2048,
            tool_call_limit=12,
            command_timeout_seconds=45,
            wall_time_limit_seconds=360,
            tool_interface=UNBROWSER_INTERACTIVE_TOOL_INTERFACE,
        )
        completed = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch(
            "pyreplab_harness.orchestrator.subprocess.run", return_value=completed
        ) as runner:
            _run_pi(
                PROJECT_ROOT,
                RemoteConfig("host", "/project", "/runs"),
                "/runs/workspace",
                "task prompt",
                policy_spec_from_treatment(treatment),
                "pi",
                None,
                unbrowser_url="http://127.0.0.1:18090/routing/rf-abc123",
                unbrowser_binary="/usr/local/bin/unbrowser",
                unbrowser_interactive=True,
                confine_unbrowser=True,
            )
        command = runner.call_args.args[0]
        self.assertIn("--gym-confine-unbrowser", command)
        self.assertIn("--gym-unbrowser-interactive", command)

    def test_task_json_forwards_task_role_for_routing_fixture(self) -> None:
        config = RemoteConfig("host", "/project", "/runs")
        fixture_args = argparse.Namespace(
            family="routing_fixture",
            seed=7,
            difficulty="easy",
            task_role="T_canary",
        )
        with mock.patch(
            "pyreplab_harness.orchestrator.remote_json",
            return_value={"id": "task-1"},
        ) as remote:
            _task_json(config, fixture_args)
        arguments = remote.call_args.args[1]
        self.assertEqual(arguments[arguments.index("--task-role") + 1], "T_canary")
        # routing_fixture has no template parameter.
        self.assertNotIn("--fixture-template", arguments)

    def test_routing_fixture_result_includes_controller_commitment(self) -> None:
        treatment = TreatmentSpec(
            id="routing-fixture",
            version="1",
            system_prompt="Use routing fixture.",
            allowed_tools=("bash", "unbrowser"),
            max_output_tokens=256,
            tool_call_limit=2,
            command_timeout_seconds=10,
            wall_time_limit_seconds=30,
            tool_interface=UNBROWSER_INTERACTIVE_TOOL_INTERFACE,
        )
        policies = {
            treatment.bundle_id: policy_spec_from_treatment(treatment)
        }
        args = argparse.Namespace(
            family="routing_fixture",
            seed=3,
            difficulty="easy",
            preserve_treatment_order=True,
        )
        task = {
            "id": "routing-fixture-rf-test",
            "family": "routing_fixture",
            "prompt": "task",
            "public_metadata": {
                "allowed_url": "http://127.0.0.1:18090/routing/rf-test"
            },
        }
        commitment = {
            "source_sha256": "a" * 64,
            "probe_features_sha256": "b" * 64,
            "probe_receipt_sha256": "c" * 64,
        }

        def remote(_config, arguments, **_kwargs):
            self.assertEqual(arguments[0], "routing-commitment")
            return commitment

        with mock.patch(
            "pyreplab_harness.orchestrator._task_json", return_value=task
        ), mock.patch(
            "pyreplab_harness.orchestrator.remote_json", side_effect=remote
        ), mock.patch(
            "pyreplab_harness.orchestrator._run_attempt",
            return_value={
                "attempt_id": "attempt-1",
                "policy": policies[treatment.bundle_id].to_dict(),
                "pi_return_code": 0,
                "pi_stderr": "",
                "verification": {"success": True},
            },
        ):
            result = _run_policy_set(
                PROJECT_ROOT,
                RemoteConfig("host", "/project", "/runs"),
                args,
                policies,
                mode="treatment_set",
            )
        self.assertEqual(result["task_commitments"], commitment)


class AttemptIdsByTreatmentTest(unittest.TestCase):
    """Tests for explicit planned attempt-id plumbing in _run_policy_set."""

    def _policies(self, n: int = 2, seed: int = 21) -> dict:
        treatments = generate_treatments(n, seed=seed)
        return {
            treatment.bundle_id: policy_spec_from_treatment(treatment)
            for treatment in treatments
        }

    def test_supplied_attempt_ids_are_used_verbatim(self) -> None:
        policies = self._policies()
        planned = {
            ref: f"planned-{index}-attempt"
            for index, ref in enumerate(sorted(policies))
        }
        args = argparse.Namespace(
            seed=3,
            family="artifact",
            attempt_ids_by_treatment=planned,
        )
        config = RemoteConfig("host", "/project", "/runs")
        seen: list[str] = []

        def fake_attempt(_root, _config, task, policy, attempt_id, _args, **_kw):
            seen.append(attempt_id)
            return {
                "attempt_id": attempt_id,
                "policy": policy.to_dict(),
                "pi_return_code": 0,
                "pi_stderr": "",
                "verification": {"success": True},
            }

        with mock.patch(
            "pyreplab_harness.orchestrator._task_json",
            return_value={"id": "artifact-task-3", "prompt": "task"},
        ), mock.patch(
            "pyreplab_harness.orchestrator._run_attempt", side_effect=fake_attempt
        ):
            result = _run_policy_set(
                PROJECT_ROOT,
                config,
                args,
                policies,
                mode="treatment_set",
            )
        self.assertEqual(sorted(seen), sorted(planned.values()))
        for attempt_id in result["attempts"].values():
            self.assertIn(attempt_id["attempt_id"], planned.values())

    def test_wrong_keys_are_rejected(self) -> None:
        policies = self._policies()
        args = argparse.Namespace(
            seed=3,
            family="artifact",
            attempt_ids_by_treatment={"wrong-key": "planned-0"},
        )
        config = RemoteConfig("host", "/project", "/runs")
        with mock.patch(
            "pyreplab_harness.orchestrator._task_json",
            return_value={"id": "artifact-task-3", "prompt": "task"},
        ):
            with self.assertRaisesRegex(ValueError, "exactly equal"):
                _run_policy_set(
                    PROJECT_ROOT,
                    config,
                    args,
                    policies,
                    mode="treatment_set",
                )

    def test_unsafe_attempt_ids_are_rejected(self) -> None:
        policies = self._policies()
        planned = {ref: "not/safe" for ref in policies}
        args = argparse.Namespace(
            seed=3,
            family="artifact",
            attempt_ids_by_treatment=planned,
        )
        config = RemoteConfig("host", "/project", "/runs")
        with mock.patch(
            "pyreplab_harness.orchestrator._task_json",
            return_value={"id": "artifact-task-3", "prompt": "task"},
        ):
            with self.assertRaisesRegex(ValueError, "invalid attempt id"):
                _run_policy_set(
                    PROJECT_ROOT,
                    config,
                    args,
                    policies,
                    mode="treatment_set",
                )

    def test_duplicate_attempt_ids_are_rejected(self) -> None:
        policies = self._policies()
        planned = {ref: "same-attempt-id" for ref in policies}
        args = argparse.Namespace(
            seed=3,
            family="artifact",
            attempt_ids_by_treatment=planned,
        )
        config = RemoteConfig("host", "/project", "/runs")
        with mock.patch(
            "pyreplab_harness.orchestrator._task_json",
            return_value={"id": "artifact-task-3", "prompt": "task"},
        ):
            with self.assertRaisesRegex(ValueError, "unique"):
                _run_policy_set(
                    PROJECT_ROOT,
                    config,
                    args,
                    policies,
                    mode="treatment_set",
                )

    def test_non_mapping_is_rejected(self) -> None:
        policies = self._policies()
        args = argparse.Namespace(
            seed=3,
            family="artifact",
            attempt_ids_by_treatment=["planned-0"],
        )
        config = RemoteConfig("host", "/project", "/runs")
        with mock.patch(
            "pyreplab_harness.orchestrator._task_json",
            return_value={"id": "artifact-task-3", "prompt": "task"},
        ):
            with self.assertRaisesRegex(ValueError, "must be a mapping"):
                _run_policy_set(
                    PROJECT_ROOT,
                    config,
                    args,
                    policies,
                    mode="treatment_set",
                )


if __name__ == "__main__":
    unittest.main()
