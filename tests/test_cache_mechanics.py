from __future__ import annotations

import unittest
from unittest import mock

from pyreplab_harness.cache_mechanics import (
    CACHE_CANARY_CELL_SCHEMA_VERSION,
    build_cache_runtime_receipt,
    build_provider_turn_cache_receipt,
    canonical_receipt_hash,
    compare_cache_cells,
    parse_cache_launch_configuration,
    probe_cache_runtime,
    validate_cache_runtime_receipt,
    validate_provider_turn_cache_receipt,
)


HELP_TEXT = " ".join(
    (
        "--cache-prompt --no-cache-prompt --cache-type-k --cache-type-v",
        "--cache-ram --ctx-checkpoints --checkpoint-min-step",
        "--cache-idle-slots --no-cache-idle-slots --cache-reuse",
        "--kv-unified --no-kv-unified --metrics --slots --no-slots",
        "--slot-save-path --sleep-idle-seconds --parallel --ctx-size",
    )
)


def _argv(mode: str) -> list[str]:
    return [
        "/llama-server",
        "--cache-prompt" if mode == "on" else "--no-cache-prompt",
        "--cache-type-k",
        "f16",
        "--cache-type-v",
        "f16",
        "--cache-ram",
        "8192",
        "--ctx-checkpoints",
        "32",
        "--checkpoint-min-step",
        "8192",
        "--cache-idle-slots",
        "--cache-reuse",
        "0",
        "--kv-unified",
        "--metrics",
        "--slots",
        "--slot-save-path",
        "/controlled/cache",
        "--sleep-idle-seconds",
        "-1",
        "--parallel",
        "1",
        "--ctx-size",
        "65536",
    ]


def _runtime(mode: str, *, eligible: bool = True) -> dict:
    receipt = build_cache_runtime_receipt(
        {
            "checked_at": "2026-08-15T00:00:00+00:00",
            "pi_version": "0.84.1",
            "pi_sha256": "a" * 64,
            "llama_server_version": "version: 1",
            "llama_server_sha256": "b" * 64,
            "model_sha256": "c" * 64,
            "model_state": "ready" if eligible else "sleeping",
            "server_argv": _argv(mode),
            "server_help": HELP_TEXT,
            "metrics_endpoint": {
                "http_status": 200 if eligible else 400,
                "response_bytes": 10,
                "response_sha256": "d" * 64,
            },
            "slots_endpoint": {
                "http_status": 200 if eligible else 400,
                "response_bytes": 10,
                "response_sha256": "e" * 64,
            },
        }
    )
    return receipt


def _turn(mode: str = "on") -> dict:
    cache_read = 0 if mode == "off" else 90
    input_tokens = 100 if mode == "off" else 10
    return {
        "turn_index": 1,
        "provider": "ubuntu-gemma",
        "model": "gemma-4-26b-a4b",
        "usage": {
            "input": input_tokens,
            "output": 2,
            "cache_read": cache_read,
            "cache_write": 0,
            "reasoning": 0,
            "total_tokens": 102,
            "logical_prompt_tokens": 100,
            "complete": True,
            "missing_fields": [],
        },
        "assistant_content_sha256": "f" * 64,
    }


def _turn_receipt(runtime: dict, mode: str, *, reused: int | None = None) -> dict:
    mechanics = {
        "exact_serialized_request_sha256": "1" * 64,
        "reused_prefix_tokens": 0 if mode == "off" else 90,
        "prompt_evaluation_seconds": 1.0 if mode == "off" else 0.4,
        "generation_seconds": 0.2,
        "slot_identity": "slot-0",
        "server_prompt_tokens": 100 if mode == "off" else 10,
        "server_predicted_tokens": 2,
    }
    if reused is not None:
        mechanics["reused_prefix_tokens"] = reused
    return build_provider_turn_cache_receipt(
        _turn(mode),
        attempt_id=f"attempt-{mode}",
        panel_id="panel-1",
        pair_id="pair-1",
        sampling_seed=7,
        cache_runtime_receipt=runtime,
        mechanics=mechanics,
    )


def _cell(mode: str, *, eligible: bool = True, reused: int | None = None) -> dict:
    runtime = _runtime(mode, eligible=eligible)
    turn = _turn_receipt(runtime, mode, reused=reused)
    payload = {
        "schema_version": CACHE_CANARY_CELL_SCHEMA_VERSION,
        "cell_id": f"cache-{mode}",
        "cache_mode": mode,
        "runtime_receipt": runtime,
        "attempts": [
            {
                "pair_id": "pair-1",
                "order_index": 0,
                "sampling_receipt": {"seed": 7, "temperature": 0.8},
                "provider_turn_receipts": [turn],
                "final_output_sha256": "2" * 64,
                "tool_trajectory_sha256": "3" * 64,
                "verifier_result_sha256": "4" * 64,
            }
        ],
    }
    return {**payload, "cell_hash": canonical_receipt_hash(payload)}


class CacheRuntimeReceiptTest(unittest.TestCase):
    def test_current_implicit_configuration_is_ineligible(self) -> None:
        receipt = build_cache_runtime_receipt(
            {
                "pi_version": "0.84.1",
                "pi_sha256": "a" * 64,
                "llama_server_version": "version: 1",
                "llama_server_sha256": "b" * 64,
                "model_sha256": "c" * 64,
                "model_state": "sleeping",
                "server_argv": [
                    "/llama-server",
                    "--ctx-size",
                    "65536",
                    "--parallel",
                    "1",
                    "--sleep-idle-seconds",
                    "300",
                ],
                "server_help": HELP_TEXT,
                "metrics_endpoint": {"http_status": 400},
                "slots_endpoint": {"http_status": 400},
            }
        )
        validate_cache_runtime_receipt(receipt)
        self.assertEqual(receipt["cache_mode"], "unresolved")
        self.assertFalse(receipt["acceptance"]["eligible_for_canary"])
        self.assertIn(
            "cache_configuration_uses_implicit_defaults",
            receipt["acceptance"]["invalidation_codes"],
        )
        self.assertEqual(
            receipt["endpoints"]["metrics"]["status"],
            "blocked_while_sleeping",
        )

    def test_prompt_mode_is_only_common_hash_exclusion(self) -> None:
        off = _runtime("off")
        on = _runtime("on")
        self.assertEqual(off["common_config_hash"], on["common_config_hash"])
        self.assertNotEqual(off["cell_config_hash"], on["cell_config_hash"])

    def test_duplicate_and_conflicting_cache_arguments_fail_closed(self) -> None:
        parsed = parse_cache_launch_configuration(
            ["server", "--cache-prompt", "--no-cache-prompt"], HELP_TEXT
        )
        self.assertIn(
            "duplicate_cache_argument:prompt_cache",
            parsed["invalidation_codes"],
        )
        self.assertIn(
            "conflicting_cache_argument:prompt_cache",
            parsed["invalidation_codes"],
        )

    def test_help_support_requires_a_complete_flag_token(self) -> None:
        parsed = parse_cache_launch_configuration(
            ["server", "--ctx-size", "65536"],
            "--cache-prompt --cache-ram",
        )
        self.assertFalse(parsed["fields"]["ctx_size"]["supported_by_help"])
        self.assertIn(
            "cache_argument_not_in_help:ctx_size",
            parsed["invalidation_codes"],
        )

    def test_tampered_runtime_receipt_is_rejected(self) -> None:
        receipt = _runtime("off")
        receipt["runtime"]["model_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "receipt_hash mismatch"):
            validate_cache_runtime_receipt(receipt)

    def test_probe_uses_only_identity_commands_and_get_endpoints(self) -> None:
        model_entry = {
            "status": {
                "value": "sleeping",
                "args": [*_argv("on"), "--model", "/model.gguf"],
            }
        }
        endpoint_calls: list[str] = []

        def endpoint(host: str, url: str) -> dict:
            self.assertEqual(host, "ubuntu-local")
            endpoint_calls.append(url)
            return {"http_status": 400, "response_bytes": 0, "response_sha256": "0" * 64}

        with mock.patch(
            "pyreplab_harness.cache_mechanics.shutil.which", return_value="/pi"
        ), mock.patch(
            "pyreplab_harness.cache_mechanics._sha256_file", return_value="a" * 64
        ), mock.patch(
            "pyreplab_harness.cache_mechanics._ssh_free_pi_version",
            return_value="0.84.1",
        ), mock.patch(
            "pyreplab_harness.cache_mechanics._remote_model_endpoint_entry",
            return_value=model_entry,
        ), mock.patch(
            "pyreplab_harness.cache_mechanics._remote_http_get_observation",
            side_effect=endpoint,
        ), mock.patch(
            "pyreplab_harness.cache_mechanics._ssh_capture",
            side_effect=["version: 1", HELP_TEXT, "b" * 64, "c" * 64],
        ) as ssh_capture:
            probe_cache_runtime(
                host="ubuntu-local",
                provider_base_url="http://127.0.0.1:18081/v1",
                remote_provider_base_url="http://127.0.0.1:8081/v1",
                model_alias="gemma",
                pi_binary="pi",
                llama_server_binary="/llama-server",
                model_artifact="/model.gguf",
            )

        self.assertEqual(
            endpoint_calls,
            ["http://127.0.0.1:8081/metrics", "http://127.0.0.1:8081/slots"],
        )
        commands = [call.args[1] for call in ssh_capture.call_args_list]
        self.assertEqual(
            commands,
            [
                ["/llama-server", "--version"],
                ["/llama-server", "--help"],
                ["sha256sum", "/llama-server"],
                ["sha256sum", "/model.gguf"],
            ],
        )

    def test_probe_rejects_serving_binary_path_mismatch(self) -> None:
        with mock.patch(
            "pyreplab_harness.cache_mechanics.shutil.which", return_value="/pi"
        ), mock.patch(
            "pyreplab_harness.cache_mechanics._remote_model_endpoint_entry",
            return_value={
                "status": {
                    "value": "sleeping",
                    "args": ["/other-server", "--model", "/model.gguf"],
                }
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "llama-server path mismatch"):
                probe_cache_runtime(
                    host="ubuntu-local",
                    provider_base_url="http://127.0.0.1:18081/v1",
                    remote_provider_base_url="http://127.0.0.1:8081/v1",
                    model_alias="gemma",
                    pi_binary="pi",
                    llama_server_binary="/llama-server",
                    model_artifact="/model.gguf",
                )

    def test_probe_rejects_serving_model_path_mismatch(self) -> None:
        with mock.patch(
            "pyreplab_harness.cache_mechanics.shutil.which", return_value="/pi"
        ), mock.patch(
            "pyreplab_harness.cache_mechanics._remote_model_endpoint_entry",
            return_value={
                "status": {
                    "value": "sleeping",
                    "args": ["/llama-server", "--model", "/other.gguf"],
                }
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "model artifact path mismatch"):
                probe_cache_runtime(
                    host="ubuntu-local",
                    provider_base_url="http://127.0.0.1:18081/v1",
                    remote_provider_base_url="http://127.0.0.1:8081/v1",
                    model_alias="gemma",
                    pi_binary="pi",
                    llama_server_binary="/llama-server",
                    model_artifact="/model.gguf",
                )


class ProviderTurnCacheReceiptTest(unittest.TestCase):
    def test_provider_cache_read_is_not_promoted_to_server_reuse(self) -> None:
        runtime = _runtime("on", eligible=False)
        receipt = build_provider_turn_cache_receipt(
            _turn(),
            attempt_id="attempt",
            panel_id="panel",
            pair_id="pair",
            sampling_seed=7,
            cache_runtime_receipt=runtime,
        )
        validate_provider_turn_cache_receipt(receipt)
        self.assertEqual(
            receipt["provider_reported_cache_read_tokens"],
            {"status": "observed", "value": 90},
        )
        self.assertEqual(receipt["reused_prefix_tokens"]["status"], "unobservable")
        self.assertFalse(receipt["mechanics_valid"])

    def test_complete_instrumentation_produces_valid_receipt(self) -> None:
        receipt = _turn_receipt(_runtime("on"), "on")
        validate_provider_turn_cache_receipt(receipt)
        self.assertTrue(receipt["mechanics_valid"])


class CacheCanaryComparatorTest(unittest.TestCase):
    def _rehash_turn_and_cell(self, cell: dict) -> None:
        turn = cell["attempts"][0]["provider_turn_receipts"][0]
        turn["receipt_hash"] = canonical_receipt_hash(
            {key: value for key, value in turn.items() if key != "receipt_hash"}
        )
        cell["cell_hash"] = canonical_receipt_hash(
            {key: value for key, value in cell.items() if key != "cell_hash"}
        )

    def test_equivalent_cells_with_savings_pass(self) -> None:
        report = compare_cache_cells(_cell("off"), _cell("on"), ["pair-1"])
        self.assertTrue(report["stage1_acceptance"]["passed"])
        self.assertEqual(report["stage1_acceptance"]["decision"], "retain_transparent_cache")
        self.assertAlmostEqual(
            report["performance_summary"]["prompt_evaluation_savings_fraction"],
            0.6,
        )

    def test_request_hash_mismatch_is_no_go(self) -> None:
        on = _cell("on")
        turn = on["attempts"][0]["provider_turn_receipts"][0]
        turn["exact_serialized_request_sha256"]["value"] = "9" * 64
        turn["receipt_hash"] = canonical_receipt_hash(
            {key: value for key, value in turn.items() if key != "receipt_hash"}
        )
        on["cell_hash"] = canonical_receipt_hash(
            {key: value for key, value in on.items() if key != "cell_hash"}
        )
        report = compare_cache_cells(_cell("off"), on, ["pair-1"])
        self.assertFalse(report["input_equivalence"]["passed"])
        self.assertFalse(report["stage1_acceptance"]["passed"])

    def test_behavior_divergence_is_no_go(self) -> None:
        on = _cell("on")
        on["attempts"][0]["final_output_sha256"] = "9" * 64
        on["cell_hash"] = canonical_receipt_hash(
            {key: value for key, value in on.items() if key != "cell_hash"}
        )
        report = compare_cache_cells(_cell("off"), on, ["pair-1"])
        self.assertFalse(report["behavior_invariance"]["passed"])

    def test_server_predicted_token_divergence_is_no_go(self) -> None:
        on = _cell("on")
        turn = on["attempts"][0]["provider_turn_receipts"][0]
        turn["server_predicted_tokens"]["value"] = 3
        self._rehash_turn_and_cell(on)
        report = compare_cache_cells(_cell("off"), on, ["pair-1"])
        self.assertFalse(report["behavior_invariance"]["passed"])

    def test_server_logical_prompt_divergence_is_no_go(self) -> None:
        on = _cell("on")
        turn = on["attempts"][0]["provider_turn_receipts"][0]
        turn["server_prompt_tokens"]["value"] = 11
        self._rehash_turn_and_cell(on)
        report = compare_cache_cells(_cell("off"), on, ["pair-1"])
        self.assertFalse(report["input_equivalence"]["passed"])

    def test_provider_server_accounting_mismatch_is_mechanics_no_go(self) -> None:
        on = _cell("on")
        turn = on["attempts"][0]["provider_turn_receipts"][0]
        turn["provider_usage"]["input"] = 11
        self._rehash_turn_and_cell(on)
        report = compare_cache_cells(_cell("off"), on, ["pair-1"])
        self.assertFalse(report["mechanics_observability"]["passed"])
        self.assertIn(
            "cache_on_provider_server_input_mismatch:pair-1:1",
            report["mechanics_observability"]["errors"],
        )

    def test_cache_off_reuse_is_no_go(self) -> None:
        report = compare_cache_cells(
            _cell("off", reused=1), _cell("on"), ["pair-1"]
        )
        self.assertFalse(report["mechanics_observability"]["passed"])

    def test_reordered_or_missing_pair_is_no_go(self) -> None:
        off = _cell("off")
        off["attempts"][0]["pair_id"] = "wrong"
        off["cell_hash"] = canonical_receipt_hash(
            {key: value for key, value in off.items() if key != "cell_hash"}
        )
        report = compare_cache_cells(off, _cell("on"), ["pair-1"])
        self.assertFalse(report["input_equivalence"]["passed"])

    def test_missing_mechanics_cannot_claim_savings(self) -> None:
        report = compare_cache_cells(
            _cell("off", eligible=False),
            _cell("on", eligible=False),
            ["pair-1"],
        )
        self.assertFalse(report["mechanics_observability"]["passed"])
        self.assertFalse(report["performance_evidence"]["passed"])
        self.assertIsNone(
            report["performance_summary"]["prompt_evaluation_savings_fraction"]
        )


if __name__ == "__main__":
    unittest.main()
