from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pyreplab_harness.artifact_gym import prepare_attempt
from pyreplab_harness.io_utils import write_json
from pyreplab_harness.routing_fixture_gym import (
    FIXTURE_BASE_URL,
    GENERATOR_VERSION,
    VERIFIER_ID,
    VERIFIER_VERSION,
    generate_routing_fixture_task,
    verify_routing_fixture_attempt,
)
from pyreplab_harness.routing_fixtures import build_stage_b_design


def _coord(difficulty: str = "easy") -> dict:
    for coord in build_stage_b_design():
        if coord["difficulty"] == difficulty:
            return coord
    raise AssertionError(f"no Stage-B coordinate with difficulty {difficulty!r}")


class RoutingFixtureGymTest(unittest.TestCase):
    def test_generation_is_deterministic_and_url_is_opaque_fixed(self) -> None:
        coord = _coord("easy")
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            task_a = generate_routing_fixture_task(first, coord["seed"], "easy")
            task_b = generate_routing_fixture_task(second, coord["seed"], "easy")
            self.assertEqual(task_a.id, task_b.id)
            self.assertEqual(task_a.prompt, task_b.prompt)
            self.assertEqual(task_a.family, "routing_fixture")
            allowed_url = task_a.public_metadata["allowed_url"]
            self.assertTrue(allowed_url.startswith(FIXTURE_BASE_URL))
            self.assertIn(f"/routing/{coord['fixture_id']}", allowed_url)
            # Opaque: no template/seed/difficulty/stratum in the route.
            self.assertNotIn(str(coord["seed"]), allowed_url)
            self.assertNotIn(coord["stratum"], allowed_url)

    def test_task_id_uses_opaque_fixture_id(self) -> None:
        coord = _coord("easy")
        with tempfile.TemporaryDirectory() as directory:
            task = generate_routing_fixture_task(directory, coord["seed"], "easy")
            self.assertEqual(task.id, f"routing-fixture-{coord['fixture_id']}")

    def test_private_oracle_matches_design(self) -> None:
        coord = _coord("medium")
        with tempfile.TemporaryDirectory() as directory:
            task = generate_routing_fixture_task(directory, coord["seed"], "medium")
            oracle_path = Path(task.verifier_ref)
            self.assertTrue(oracle_path.exists())
            oracle = json.loads(oracle_path.read_text())
            self.assertEqual(oracle["nonce"], coord["nonce"])
            self.assertEqual(oracle["expected_answer"], coord["nonce"])
            self.assertEqual(oracle["verification_type"], "exact_match")

    def test_prompt_and_contract_are_truthful_and_not_leaky(self) -> None:
        coord = _coord("easy")
        with tempfile.TemporaryDirectory() as directory:
            task = generate_routing_fixture_task(directory, coord["seed"], "easy")
            # The prompt describes the true per-task operation and never
            # reveals the stratum, nonce, reference, or private labels.
            for secret in (coord["nonce"], coord["stratum"]):
                self.assertNotIn(secret, task.prompt)
            for line in task.contract:
                self.assertNotIn(coord["nonce"], line)
            # The contract mentions the fixed URL and result.json.
            joined = "\n".join(task.contract)
            self.assertIn(FIXTURE_BASE_URL, joined)
            self.assertIn("result.json", joined)

    def test_verifier_accepts_correct_nonce(self) -> None:
        coord = _coord("easy")
        with tempfile.TemporaryDirectory() as directory:
            task = generate_routing_fixture_task(directory, coord["seed"], "easy")
            attempt = prepare_attempt(directory, task.id, "rf-pass", "correct")
            write_json(
                Path(attempt.workspace_ref) / "result.json",
                {"verification_key": coord["nonce"]},
            )
            result = verify_routing_fixture_attempt(
                directory, task.id, attempt.attempt_id
            )
            self.assertTrue(result.success)
            self.assertIsNone(result.failure_code)
            self.assertEqual(result.verifier_id, VERIFIER_ID)
            self.assertEqual(result.verifier_version, VERIFIER_VERSION)

    def test_verifier_rejects_wrong_nonce(self) -> None:
        coord = _coord("easy")
        with tempfile.TemporaryDirectory() as directory:
            task = generate_routing_fixture_task(directory, coord["seed"], "easy")
            attempt = prepare_attempt(directory, task.id, "rf-wrong", "wrong")
            write_json(
                Path(attempt.workspace_ref) / "result.json",
                {"verification_key": "RF-WRONG"},
            )
            result = verify_routing_fixture_attempt(
                directory, task.id, attempt.attempt_id
            )
            self.assertFalse(result.success)
            self.assertEqual(result.failure_code, "nonce_mismatch")

    def test_verifier_rejects_missing_result(self) -> None:
        coord = _coord("easy")
        with tempfile.TemporaryDirectory() as directory:
            task = generate_routing_fixture_task(directory, coord["seed"], "easy")
            attempt = prepare_attempt(directory, task.id, "rf-missing", "wrong")
            result = verify_routing_fixture_attempt(
                directory, task.id, attempt.attempt_id
            )
            self.assertFalse(result.success)
            self.assertEqual(result.failure_code, "missing_output")

    def test_verifier_rejects_malformed_json(self) -> None:
        coord = _coord("easy")
        with tempfile.TemporaryDirectory() as directory:
            task = generate_routing_fixture_task(directory, coord["seed"], "easy")
            attempt = prepare_attempt(directory, task.id, "rf-malformed", "wrong")
            (Path(attempt.workspace_ref) / "result.json").write_text(
                "not valid json {{", encoding="utf-8"
            )
            result = verify_routing_fixture_attempt(
                directory, task.id, attempt.attempt_id
            )
            self.assertFalse(result.success)
            self.assertEqual(result.failure_code, "invalid_json")

    def test_rejects_seed_not_in_frozen_design(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "no frozen Stage-B"):
                generate_routing_fixture_task(directory, 42, "easy")

    def test_rejects_wrong_difficulty(self) -> None:
        coord = _coord("easy")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "difficulty"):
                generate_routing_fixture_task(directory, coord["seed"], "hard")

    def test_rejects_invalid_difficulty_label(self) -> None:
        coord = _coord("easy")
        with self.assertRaisesRegex(ValueError, "difficulty must be"):
            generate_routing_fixture_task(tempfile.mkdtemp(), coord["seed"], "nope")

    def test_task_role_is_supported_and_cached_drift_rejected(self) -> None:
        coord = _coord("easy")
        with tempfile.TemporaryDirectory() as directory:
            task = generate_routing_fixture_task(
                directory, coord["seed"], "easy", task_role="T_canary"
            )
            self.assertEqual(task.public_metadata["task_role"], "T_canary")
            with self.assertRaisesRegex(ValueError, "cached task role mismatch"):
                generate_routing_fixture_task(directory, coord["seed"], "easy")

    def test_generator_version_constant(self) -> None:
        coord = _coord("easy")
        with tempfile.TemporaryDirectory() as directory:
            task = generate_routing_fixture_task(directory, coord["seed"], "easy")
            self.assertEqual(task.generator_version, GENERATOR_VERSION)
            self.assertEqual(task.generator_version, "routing-fixture-gym-v1")

    def test_subsequent_call_returns_cached_task(self) -> None:
        coord = _coord("hard")
        with tempfile.TemporaryDirectory() as directory:
            first = generate_routing_fixture_task(directory, coord["seed"], "hard")
            second = generate_routing_fixture_task(directory, coord["seed"], "hard")
            self.assertIsNot(first, second)
            self.assertEqual(first.id, second.id)
            self.assertEqual(first.to_dict(), second.to_dict())


if __name__ == "__main__":
    unittest.main()
