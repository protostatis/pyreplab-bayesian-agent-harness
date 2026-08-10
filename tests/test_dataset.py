from __future__ import annotations

import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from pyreplab_harness.artifact_gym import prepare_attempt, record_pi_events
from pyreplab_harness.dataset import (
    _grammar_factors_from_treatment,
    _UNBROWSER_GRAMMAR_INTERFACE,
    build_dataset,
    build_model_input,
    flatten_public_metadata,
    iter_dataset_rows,
    main as dataset_main,
    task_split,
    write_dataset,
)
from pyreplab_harness.events import normalize_pi_events
from pyreplab_harness.gym_registry import generate_task, verify_attempt
from pyreplab_harness.io_utils import read_json, write_json
from pyreplab_harness.meta_grammar import (
    enumerate_unbrowser_grammar,
    export_grammar_factors,
)
from pyreplab_harness.treatments import (
    TreatmentRegistry,
    TreatmentSpec,
    generate_treatments,
)


def _synthetic_events(
    *,
    assistant_count: int = 2,
    tool_calls: int = 3,
    input_per: int = 100,
    output_per: int = 40,
    text: str = "DONE",
) -> list[dict]:
    events = [{"type": "session", "version": 3, "id": "s1", "cwd": "/tmp"}]
    for _ in range(assistant_count):
        events.append(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "provider": "ubuntu-gemma",
                    "model": "gemma-4-26b-a4b",
                    "content": [{"type": "text", "text": text}],
                    "usage": {
                        "input": input_per,
                        "output": output_per,
                        "totalTokens": input_per + output_per,
                    },
                },
            }
        )
    for index in range(tool_calls):
        events.append(
            {
                "type": "tool_execution_end",
                "toolCallId": f"t{index}",
                "toolName": "bash",
                "result": {"content": [{"type": "text", "text": "ok"}]},
            }
        )
    return events


def _make_attempt(
    root: str,
    task,
    attempt_id: str,
    policy_id: str,
    *,
    events: list[dict] | None = None,
    success: bool | None = None,
    policy_version: str = "1",
):
    """Prepare an attempt, optionally record events, optionally submit a
    correct artifact, and always run the family verifier."""
    record = prepare_attempt(root, task.id, attempt_id, policy_id, policy_version)
    if events is not None:
        raw = "\n".join(json.dumps(event) for event in events)
        record_pi_events(root, attempt_id, raw, normalize_pi_events(raw))
    if success is True:
        expected = read_json(Path(task.verifier_ref))["expected"]
        write_json(Path(record.workspace_ref) / "result.json", expected)
    return verify_attempt(task.family, root, task.id, attempt_id)


class DatasetJoinTest(unittest.TestCase):
    def test_joins_task_attempt_and_verification_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_task("artifact", directory, 7, "easy")
            events = _synthetic_events(assistant_count=2, tool_calls=3, input_per=100, output_per=40)
            result = _make_attempt(
                directory, task, "attempt-1", "direct", events=events, success=True
            )
            self.assertTrue(result.success)

            rows = build_dataset(directory)
            self.assertEqual(len(rows), 1)
            row = rows[0]

            self.assertEqual(row["task_id"], task.id)
            self.assertEqual(row["family"], "artifact")
            self.assertEqual(row["template_id"], task.template_id)
            self.assertEqual(row["generator_version"], task.generator_version)
            self.assertEqual(row["seed"], 7)
            self.assertEqual(row["difficulty"], "easy")
            self.assertEqual(row["prompt"], task.prompt)
            self.assertEqual(row["contract"], list(task.contract))
            self.assertEqual(row["public_metadata"], task.public_metadata)

            self.assertEqual(row["attempt_id"], "attempt-1")
            self.assertEqual(row["policy_id"], "direct")
            self.assertEqual(row["policy_version"], "1")

            self.assertTrue(row["verified_success"])
            self.assertIsNone(row["failure_code"])
            self.assertEqual(row["verifier_id"], result.verifier_id)
            self.assertEqual(row["verifier_version"], result.verifier_version)

            self.assertEqual(
                row["usage"],
                {
                    "input": 200,
                    "output": 80,
                    "cache_read": 0,
                    "cache_write": 0,
                    "reasoning": 0,
                    "total_tokens": 280,
                },
            )
            self.assertEqual(row["assistant_message_count"], 2)
            self.assertEqual(row["provider_turn_count"], 2)
            self.assertEqual(row["tool_call_count"], 3)
            self.assertEqual(row["tool_limit_rejection_count"], 0)
            self.assertEqual(row["length_stop_count"], 0)
            self.assertEqual(row["final_text_length"], 4)

            self.assertEqual(row["split"], task_split(task.template_id, task.seed))
            self.assertIn(row["split"], {"train", "validation", "test"})

    def test_failed_verification_is_kept_with_failure_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_task("artifact", directory, 9, "medium")
            _make_attempt(
                directory, task, "attempt-fail", "deliberate",
                events=_synthetic_events(), success=None,
            )
            rows = build_dataset(directory)
            self.assertEqual(len(rows), 1)
            self.assertFalse(rows[0]["verified_success"])
            self.assertEqual(rows[0]["failure_code"], "missing_output")
            self.assertEqual(rows[0]["policy_id"], "deliberate")


class LeakageTest(unittest.TestCase):
    FORBIDDEN_KEYS = {
        "verifier_ref",
        "workspace_ref",
        "pi_events_ref",
        "normalized_events_ref",
        "verification_ref",
        "diagnostics",
        "final_text",
        "raw_trajectory",
        "trajectory",
        "pi_events",
        "oracle",
        "private_metadata",
    }

    def test_rows_expose_no_private_paths_trajectory_or_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for family in ("artifact", "sqlite", "shell"):
                with self.subTest(family=family):
                    task = generate_task(family, directory, 21, "easy")
                    _make_attempt(
                        directory, task, f"attempt-{family}", "direct",
                        events=_synthetic_events(),
                    )
            for row in build_dataset(directory):
                for forbidden in self.FORBIDDEN_KEYS:
                    self.assertNotIn(forbidden, row)

    def test_model_input_contains_only_predecision_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_task("artifact", directory, 3, "hard")
            events = _synthetic_events(assistant_count=4, tool_calls=5)
            _make_attempt(directory, task, "attempt-mi", "deliberate", events=events, success=True)
            row = build_dataset(directory)[0]

            model_input = row["model_input"]
            self.assertEqual(
                set(model_input),
                {"text", "family", "template_id", "difficulty", "public_metadata", "policy_id", "policy_version"},
            )
            self.assertEqual(model_input["family"], "artifact")
            self.assertEqual(model_input["template_id"], task.template_id)
            self.assertEqual(model_input["difficulty"], "hard")
            self.assertEqual(model_input["policy_id"], "deliberate")
            self.assertEqual(model_input["policy_version"], "1")
            self.assertEqual(
                model_input["text"],
                task.prompt + "\n\n" + "\n".join(task.contract),
            )
            # Post-action fields must never enter the predecision input.
            for post_action in ("usage", "verified_success", "failure_code", "assistant_message_count", "tool_call_count", "final_text_length", "split", "attempt_id"):
                self.assertNotIn(post_action, model_input)

    def test_model_input_metadata_is_numeric_or_bool_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for family in ("artifact", "sqlite", "shell"):
                with self.subTest(family=family):
                    task = generate_task(family, directory, 5, "medium")
                    _make_attempt(
                        directory, task, f"attempt-flat-{family}", "direct",
                        events=_synthetic_events(),
                    )
            for row in build_dataset(directory):
                for key, value in row["model_input"]["public_metadata"].items():
                    self.assertTrue(
                        isinstance(value, (bool, int, float)),
                        f"{key!r} has non-numeric value {value!r}",
                    )
                    if isinstance(value, float):
                        self.assertTrue(math.isfinite(value))


class TreatmentRegistryDatasetTest(unittest.TestCase):
    @staticmethod
    def _registry(policy_id: str = "direct", version: str = "1") -> TreatmentRegistry:
        return TreatmentRegistry(
            (
                TreatmentSpec(
                    id=policy_id,
                    version=version,
                    system_prompt="Plan briefly, execute, and verify the final artifact.",
                    allowed_tools=("bash",),
                    max_output_tokens=1536,
                    tool_call_limit=7,
                    command_timeout_seconds=30,
                    wall_time_limit_seconds=300,
                ),
            )
        )

    def test_registry_enriches_predecision_treatment_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_task("artifact", directory, 17, "easy")
            _make_attempt(
                directory,
                task,
                "attempt-treatment",
                "direct",
                events=_synthetic_events(),
                success=True,
            )
            registry = self._registry()
            rows = build_dataset(directory, treatment_registry=registry)
            self.assertEqual(len(rows), 1)
            row = rows[0]
            treatment = row["model_input"]["treatment"]
            self.assertEqual(treatment["text"], registry.treatments[0].system_prompt)
            self.assertNotIn(task.prompt, treatment["text"])
            self.assertEqual(treatment["bundle_id"], registry.treatments[0].bundle_id)
            self.assertEqual(treatment["tool_call_limit"], 7)
            self.assertEqual(row["treatment_bundle_hash"], registry.treatments[0].bundle_hash)
            self.assertEqual(row["treatment_registry_hash"], registry.registry_hash)
            self.assertEqual(row["model_input"]["text"], task.prompt + "\n\n" + "\n".join(task.contract))

    def test_registry_hash_is_reported_and_path_loading_works(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_task("artifact", directory, 18, "easy")
            _make_attempt(
                directory,
                task,
                "attempt-treatment",
                "direct",
                events=_synthetic_events(),
            )
            registry = self._registry()
            registry_path = Path(directory) / "treatments.json"
            registry.save(registry_path)
            output = Path(directory) / "dataset.jsonl"
            summary = write_dataset(
                directory, output, treatment_registry=registry_path
            )
            self.assertEqual(summary["treatment_registry_hash"], registry.registry_hash)
            written_row = json.loads(output.read_text())
            self.assertIn("treatment", written_row["model_input"])
            self.assertEqual(
                written_row["treatment_registry_hash"], registry.registry_hash
            )

    def test_registry_missing_attempt_treatment_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_task("artifact", directory, 19, "easy")
            _make_attempt(
                directory,
                task,
                "attempt-treatment",
                "direct",
                events=_synthetic_events(),
            )
            registry = self._registry(policy_id="different")
            with self.assertRaisesRegex(ValueError, "missing from the supplied registry"):
                build_dataset(directory, treatment_registry=registry)

    def test_build_model_input_rejects_identity_mismatch(self) -> None:
        treatment = self._registry(policy_id="different").treatments[0]
        task = {
            "prompt": "task",
            "contract": [],
            "family": "artifact",
            "template_id": "template",
            "difficulty": "easy",
            "public_metadata": {},
        }
        with self.assertRaisesRegex(ValueError, "does not match"):
            build_model_input(task, "direct", "1", treatment=treatment)


class NumericFlatteningTest(unittest.TestCase):
    def test_flatten_public_metadata_keeps_only_finite_numeric_bools(self) -> None:
        metadata = {
            "customer_count": 5,
            "active": True,
            "ratio": 0.5,
            "name": "orders",
            "tags": ["a", "b"],
            "nested": {"rows": 3, "flag": False, "label": "x", "depth": {"leaf": 1}},
            "not_a_number": None,
            "inf": float("inf"),
            "nan": float("nan"),
        }
        self.assertEqual(
            flatten_public_metadata(metadata),
            {
                "customer_count": 5,
                "active": True,
                "ratio": 0.5,
                "nested.rows": 3,
                "nested.flag": False,
                "nested.depth.leaf": 1,
            },
        )

    def test_row_model_input_uses_flattened_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_task("artifact", directory, 11, "easy")
            manifest = read_json(Path(directory) / "tasks" / task.id / "task.json")
            manifest["public_metadata"] = {
                "customer_count": 5,
                "active": True,
                "ratio": 0.5,
                "name": "orders",
                "tags": ["a", "b"],
                "nested": {"rows": 3, "flag": False, "label": "x", "depth": {"leaf": 1}},
                "not_a_number": None,
                "inf": float("inf"),
                "nan": float("nan"),
            }
            write_json(Path(directory) / "tasks" / task.id / "task.json", manifest)
            _make_attempt(
                directory, task, "attempt-flat", "direct",
                events=_synthetic_events(), success=True,
            )
            row = build_dataset(directory)[0]
            self.assertEqual(
                row["model_input"]["public_metadata"],
                {
                    "customer_count": 5,
                    "active": True,
                    "ratio": 0.5,
                    "nested.rows": 3,
                    "nested.flag": False,
                    "nested.depth.leaf": 1,
                },
            )
            # The raw audit copy is untouched.
            self.assertEqual(row["public_metadata"]["tags"], ["a", "b"])


class SplitTest(unittest.TestCase):
    def test_split_distribution_is_approximately_70_15_15(self) -> None:
        counts = {"train": 0, "validation": 0, "test": 0}
        for seed in range(3000):
            counts[task_split("some-template-v1", seed)] += 1
        total = sum(counts.values())
        self.assertEqual(total, 3000)
        self.assertAlmostEqual(counts["train"] / total, 0.70, delta=0.05)
        self.assertAlmostEqual(counts["validation"] / total, 0.15, delta=0.04)
        self.assertAlmostEqual(counts["test"] / total, 0.15, delta=0.04)

    def test_paired_attempts_share_split_and_dataset_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            for root in (first, second):
                for seed in (1, 2, 3, 4):
                    task = generate_task("artifact", root, seed, "easy")
                    _make_attempt(
                        root, task, f"z-att-{seed}", "deliberate",
                        events=_synthetic_events(), success=True,
                    )
                    _make_attempt(
                        root, task, f"a-att-{seed}", "direct",
                        events=_synthetic_events(), success=True,
                    )
            rows_a = build_dataset(first)
            rows_b = build_dataset(second)
            self.assertEqual(rows_a, rows_b)

            by_task: dict[str, set[str]] = {}
            for row in rows_a:
                by_task.setdefault(row["task_id"], set()).add(row["split"])
            self.assertTrue(by_task)
            for splits in by_task.values():
                self.assertEqual(len(splits), 1)  # Pair always shares the split.
            for row in rows_a:
                self.assertIn(row["split"], {"train", "validation", "test"})

    def test_rows_are_sorted_by_task_policy_then_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task_b = generate_task("artifact", directory, 2, "easy")
            task_a = generate_task("artifact", directory, 1, "easy")
            _make_attempt(directory, task_b, "z-last", "direct", events=_synthetic_events())
            _make_attempt(directory, task_a, "a-first", "direct", events=_synthetic_events())
            _make_attempt(directory, task_a, "m-mid", "deliberate", events=_synthetic_events())

            rows = build_dataset(directory)
            keys = [(row["task_id"], row["policy_id"], row["attempt_id"]) for row in rows]
            self.assertEqual(keys, sorted(keys))
            self.assertEqual(
                keys[0],
                (task_a.id, "deliberate", "m-mid"),
            )

    def test_split_ignores_stored_manifest_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_task("artifact", directory, 8, "easy")
            manifest = read_json(Path(directory) / "tasks" / task.id / "task.json")
            manifest["split"] = "bogus-stored-value"
            write_json(Path(directory) / "tasks" / task.id / "task.json", manifest)
            _make_attempt(directory, task, "attempt-split", "direct", events=_synthetic_events())
            row = build_dataset(directory)[0]
            self.assertEqual(row["split"], task_split(task.template_id, task.seed))
            self.assertNotEqual(row["split"], "bogus-stored-value")


class IncompleteSkipTest(unittest.TestCase):
    def test_unverified_and_missing_events_are_skipped_with_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_task("artifact", directory, 4, "easy")
            # Fully recorded and verified -> row.
            _make_attempt(directory, task, "complete", "direct", events=_synthetic_events(), success=True)
            # Prepared but never executed or verified.
            prepare_attempt(directory, task.id, "prepared-only", "direct")
            # Recorded events but never verified.
            record = prepare_attempt(directory, task.id, "executed-only", "deliberate")
            raw = "\n".join(json.dumps(event) for event in _synthetic_events())
            record_pi_events(directory, record.attempt_id, raw, normalize_pi_events(raw))
            # Verified but the normalized events file was removed.
            _make_attempt(directory, task, "no-events", "deliberate", events=_synthetic_events())
            (Path(directory) / "attempts" / "no-events" / "pi-events.normalized.json").unlink()

            summary = write_dataset(directory, Path(directory) / "dataset.jsonl")
            rows = build_dataset(directory)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["attempt_id"], "complete")
            self.assertEqual(summary["attempts_found"], 4)
            self.assertEqual(summary["rows_written"], 1)
            self.assertEqual(
                summary["skipped"],
                {"missing_events": 1, "unverified": 2},
            )

    def test_verified_failure_still_counts_as_a_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_task("artifact", directory, 6, "easy")
            _make_attempt(directory, task, "failed-verified", "direct", events=_synthetic_events())
            summary = write_dataset(directory, Path(directory) / "dataset.jsonl")
            self.assertEqual(summary["rows"], 1)
            self.assertEqual(summary["skipped"], {})


class MalformedRecordTest(unittest.TestCase):
    def test_invalid_task_json_raises_with_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_task("artifact", directory, 2, "easy")
            prepare_attempt(directory, task.id, "attempt-mal", "direct")
            manifest = Path(directory) / "tasks" / task.id / "task.json"
            manifest.write_text("{\nnot json", encoding="utf-8")
            with self.assertRaises(ValueError) as context:
                build_dataset(directory)
            self.assertIn(str(manifest), str(context.exception))

    def test_missing_task_field_raises_with_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_task("artifact", directory, 2, "easy")
            prepare_attempt(directory, task.id, "attempt-mal", "direct")
            manifest = read_json(Path(directory) / "tasks" / task.id / "task.json")
            del manifest["seed"]
            write_json(Path(directory) / "tasks" / task.id / "task.json", manifest)
            with self.assertRaises(ValueError) as context:
                build_dataset(directory)
            message = str(context.exception)
            self.assertIn("task manifest", message)
            self.assertIn("seed", message)

    def test_malformed_attempt_manifest_raises_with_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_task("artifact", directory, 2, "easy")
            prepare_attempt(directory, task.id, "attempt-mal", "direct")
            attempt_path = Path(directory) / "attempts" / "attempt-mal" / "attempt.json"
            manifest = read_json(attempt_path)
            del manifest["policy_id"]
            write_json(attempt_path, manifest)
            with self.assertRaises(ValueError) as context:
                build_dataset(directory)
            message = str(context.exception)
            self.assertIn(str(attempt_path), message)
            self.assertIn("policy_id", message)

    def test_verified_status_without_verification_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_task("artifact", directory, 2, "easy")
            prepare_attempt(directory, task.id, "attempt-mal", "direct")
            attempt_path = Path(directory) / "attempts" / "attempt-mal" / "attempt.json"
            manifest = read_json(attempt_path)
            manifest["status"] = "verified"
            write_json(attempt_path, manifest)
            with self.assertRaises(ValueError) as context:
                build_dataset(directory)
            self.assertIn("verified", str(context.exception))

    def test_missing_task_manifest_for_attempt_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_task("artifact", directory, 2, "easy")
            prepare_attempt(directory, task.id, "attempt-orphan", "direct")
            import shutil

            shutil.rmtree(Path(directory) / "tasks" / task.id)
            with self.assertRaises(OSError) as context:
                build_dataset(directory)
            self.assertIn("task.json", str(context.exception))

    def test_invalid_verification_json_raises_with_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_task("artifact", directory, 2, "easy")
            _make_attempt(directory, task, "attempt-bad-ver", "direct", events=_synthetic_events())
            verification = Path(directory) / "attempts" / "attempt-bad-ver" / "verification.json"
            verification.write_text("not-json", encoding="utf-8")
            with self.assertRaises(ValueError) as context:
                build_dataset(directory)
            self.assertIn(str(verification), str(context.exception))


class DeterministicJsonlTest(unittest.TestCase):
    def test_write_dataset_is_byte_deterministic_and_parseable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for seed in (1, 2, 3):
                task = generate_task("artifact", directory, seed, "easy")
                _make_attempt(directory, task, f"att-{seed}", "direct", events=_synthetic_events())
                _make_attempt(directory, task, f"att-{seed}-d", "deliberate", events=_synthetic_events())

            first = Path(directory) / "dataset-a.jsonl"
            second = Path(directory) / "dataset-b.jsonl"
            summary_a = write_dataset(directory, first)
            summary_b = write_dataset(directory, second)

            self.assertEqual(first.read_bytes(), second.read_bytes())
            expected = dict(summary_a)
            expected["output_path"] = str(second.resolve())
            self.assertEqual(expected, summary_b)

            lines = first.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), summary_a["rows_written"])
            for line in lines:
                payload = json.loads(line)
                self.assertEqual(payload, json.loads(json.dumps(payload, sort_keys=True)))
            self.assertEqual(summary_a["output_path"], str(first.resolve()))

    def test_iter_dataset_rows_and_build_agree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_task("artifact", directory, 10, "easy")
            _make_attempt(directory, task, "attempt-iter", "direct", events=_synthetic_events())
            self.assertEqual(
                list(iter_dataset_rows(directory)),
                build_dataset(directory),
            )

    def test_empty_root_builds_empty_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(build_dataset(directory), [])

    def test_missing_root_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            build_dataset(Path(tempfile.gettempdir()) / "does-not-exist-pyreplab")


class CliTest(unittest.TestCase):
    def test_cli_writes_dataset_and_prints_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_task("artifact", directory, 13, "easy")
            _make_attempt(directory, task, "attempt-cli", "direct", events=_synthetic_events())
            output = Path(directory) / "out.jsonl"
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = dataset_main([directory, str(output)])
            self.assertEqual(code, 0)
            summary = json.loads(buffer.getvalue())
            self.assertEqual(summary["rows_written"], 1)
            self.assertEqual(summary["attempts_found"], 1)
            self.assertTrue(output.exists())
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 1)

    def test_cli_without_output_prints_summary_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = generate_task("artifact", directory, 14, "easy")
            _make_attempt(directory, task, "attempt-cli2", "direct", events=_synthetic_events())
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = dataset_main([directory])
            self.assertEqual(code, 0)
            summary = json.loads(buffer.getvalue())
            self.assertEqual(summary["rows"], 1)
            self.assertNotIn("rows_written", summary)


class GrammarFactorExportTest(unittest.TestCase):
    """Tests that grammar factor labels from the 72-cell Unbrowser policy
    grammar are correctly included in ``model_input.treatment`` for dataset
    rows generated from grammar treatments, and that non-grammar treatments
    are unaffected."""

    @staticmethod
    def _unbrowser_grammar_treatment(index: int = 0) -> TreatmentSpec:
        """Return a single Unbrowser grammar treatment from the full 72."""
        return enumerate_unbrowser_grammar()[index]

    @staticmethod
    def _legacy_bash_treatment() -> TreatmentSpec:
        """Return a non-grammar treatment (standard bash policy)."""
        return TreatmentSpec(
            id="test-legacy",
            version="1",
            system_prompt="Plan briefly, execute, verify.",
            allowed_tools=("bash",),
            max_output_tokens=1024,
            tool_call_limit=4,
            command_timeout_seconds=30,
            wall_time_limit_seconds=300,
        )

    def test_grammar_factors_extracted_for_unbrowser_treatment(self) -> None:
        treatment = self._unbrowser_grammar_treatment(0)
        factors = _grammar_factors_from_treatment(treatment)
        self.assertIsNotNone(factors)
        self.assertEqual(
            set(factors),
            {"planning", "observation", "verification", "recovery", "tool_cap"},
        )
        self.assertEqual(factors["planning"], treatment.generator_metadata["planning"])

    def test_grammar_factors_none_for_non_grammar_treatment(self) -> None:
        treatment = self._legacy_bash_treatment()
        self.assertIsNone(_grammar_factors_from_treatment(treatment))

    def test_grammar_factors_none_for_empty_metadata(self) -> None:
        treatment = TreatmentSpec(
            id="test-empty-meta",
            version="1",
            system_prompt="No grammar.",
            allowed_tools=("bash", "unbrowser"),
            max_output_tokens=1024,
            tool_call_limit=4,
            command_timeout_seconds=30,
            wall_time_limit_seconds=300,
            tool_interface=_UNBROWSER_GRAMMAR_INTERFACE,
            generator_metadata={},  # No grammar factors
        )
        self.assertIsNone(_grammar_factors_from_treatment(treatment))

    def test_grammar_factors_excluded_from_legacy_treatment_model_input(
        self,
    ) -> None:
        """model_input.treatment must NOT have grammar_factors for non-
        grammar treatments (backward compatibility)."""
        task = {
            "prompt": "task",
            "contract": [],
            "family": "artifact",
            "template_id": "template",
            "difficulty": "easy",
            "public_metadata": {},
        }
        treatment = self._legacy_bash_treatment()
        model_input = build_model_input(
            task, treatment.id, treatment.version, treatment=treatment
        )
        self.assertIn("treatment", model_input)
        self.assertNotIn("grammar_factors", model_input["treatment"])

    def test_grammar_factors_included_in_unbrowser_model_input(self) -> None:
        task = {
            "prompt": "Extract verification key from fixture page.",
            "contract": ["Navigate to the fixture page.", "Extract the code."],
            "family": "unbrowser_fixture",
            "template_id": "single_page_extraction",
            "difficulty": "easy",
            "public_metadata": {"fixture_url": "http://127.0.0.1:18090/fixture/7/easy"},
        }
        treatment = self._unbrowser_grammar_treatment(0)
        model_input = build_model_input(
            task, treatment.id, treatment.version, treatment=treatment
        )
        self.assertIn("treatment", model_input)
        tr = model_input["treatment"]
        self.assertIn("grammar_factors", tr)

        factors = tr["grammar_factors"]
        self.assertEqual(
            set(factors),
            {"planning", "observation", "verification", "recovery", "tool_cap"},
        )
        # All factor values are strings, not empty.
        for key in ("planning", "observation", "verification", "recovery", "tool_cap"):
            self.assertIsInstance(factors[key], str)
            self.assertTrue(len(factors[key]) > 0)

    def test_grammar_factors_only_behavioral_levels(self) -> None:
        """Grammar factors must contain ONLY the five behavioural factor
        levels — no identity, hash, registry metadata, or version strings."""
        FORBIDDEN_IN_GRAMMAR = {
            "policy_id",
            "policy_version",
            "bundle_id",
            "bundle_hash",
            "grammar_version",
            "grammar_size",
            "grammar_name",
            "index",
            "registry_position",
        }
        treatment = self._unbrowser_grammar_treatment(10)
        factors = _grammar_factors_from_treatment(treatment)
        self.assertIsNotNone(factors)
        for forbidden in FORBIDDEN_IN_GRAMMAR:
            self.assertNotIn(
                forbidden,
                factors,
                f"grammar_factors must not contain {forbidden!r}",
            )

    def test_exported_fixture_row_matches_cnp_schema(self) -> None:
        """Frozen interface contract: an exported fixture task row must
        have the fields that calibration.py and meta_cnp.py expect.

        * model_input.treatment contains ``grammar_factors`` (five string
          labels) and the numeric budget keys.
        * model_input.task (via the parent model_input) contains
          predecision features (text, family, template_id, difficulty,
          public_metadata, policy_id, policy_version).
        * The row has verified_success, termination-like failure_code,
          and usage cost counters.
        * NO forbidden fields (page text excerpts, URLs in grammar_factors,
          verifier diagnostics, policy_id or bundle_id inside
          grammar_factors) appear where they should not.
        """
        treatment = self._unbrowser_grammar_treatment(0)
        task = {
            "prompt": "Extract verification key from fixture page.",
            "contract": ["Navigate to the fixture page.", "Extract the code."],
            "family": "unbrowser_fixture",
            "template_id": "single_page_extraction",
            "difficulty": "easy",
            "public_metadata": {
                "fixture_url": "http://127.0.0.1:18090/fixture/7/easy",
            },
        }
        model_input = build_model_input(
            task, treatment.id, treatment.version, treatment=treatment
        )

        # -- Predecision task fields exist -----------------------------------
        for required in (
            "text",
            "family",
            "template_id",
            "difficulty",
            "public_metadata",
            "policy_id",
            "policy_version",
        ):
            self.assertIn(required, model_input, f"missing predecision key {required!r}")

        # -- Treatment with grammar factors ----------------------------------
        tr = model_input["treatment"]
        self.assertIn("grammar_factors", tr)
        factors = tr["grammar_factors"]
        self.assertEqual(
            set(factors),
            {"planning", "observation", "verification", "recovery", "tool_cap"},
        )

        # Numeric budget keys — required for CNP features.
        for budget_key in (
            "max_output_tokens",
            "tool_call_limit",
            "command_timeout_seconds",
            "wall_time_limit_seconds",
        ):
            self.assertIn(budget_key, tr)
            self.assertIsInstance(tr[budget_key], int)

        # -- No forbidden fields in grammar_factors --------------------------
        FORBIDDEN = {
            "policy_id",
            "bundle_id",
            "bundle_hash",
            "grammar_version",
            "grammar_name",
            "index",
            "page_text",
            "url",
            "selector",
            "answer",
        }
        for forbidden in FORBIDDEN:
            self.assertNotIn(forbidden, factors)

        # -- model_input must NOT contain post-action fields -----------------
        for post in ("usage", "verified_success", "failure_code"):
            self.assertNotIn(post, model_input)

    def test_all_72_grammar_treatments_export_valid_factors(self) -> None:
        """Every cell in the 72-cell grammar must yield valid grammar_factors."""
        treatments = enumerate_unbrowser_grammar()
        self.assertEqual(len(treatments), 72)

        seen_combinations: set[tuple[str, ...]] = set()
        for treatment in treatments:
            factors = _grammar_factors_from_treatment(treatment)
            self.assertIsNotNone(factors)
            combo = (
                factors["planning"],
                factors["observation"],
                factors["verification"],
                factors["recovery"],
                factors["tool_cap"],
            )
            seen_combinations.add(combo)

        # The 72-cell grammar should produce 72 unique factor combinations.
        self.assertEqual(len(seen_combinations), 72)

    def test_one_hot_from_meta_grammar_export_matches_factors(self) -> None:
        """Verify consistency between raw factor labels and the one-hot
        encoding produced by ``export_grammar_factors()`` from meta_grammar.

        The dataset exports *raw* factor labels; the model computes one-hot
        from them. This test ensures the two representations agree.
        """
        treatment = self._unbrowser_grammar_treatment(42)
        raw = _grammar_factors_from_treatment(treatment)
        onehot = export_grammar_factors(treatment)

        self.assertEqual(raw["planning"], onehot["factor_labels"]["planning"])
        self.assertEqual(raw["observation"], onehot["factor_labels"]["observation"])
        self.assertEqual(raw["verification"], onehot["factor_labels"]["verification"])
        self.assertEqual(raw["recovery"], onehot["factor_labels"]["recovery"])
        self.assertEqual(raw["tool_cap"], onehot["factor_labels"]["tool_cap"])

        # The one-hot vectors have the right dimensions.
        self.assertEqual(len(onehot["one_hot"]["planning"]), 3)
        self.assertEqual(len(onehot["one_hot"]["observation"]), 3)
        self.assertEqual(len(onehot["one_hot"]["verification"]), 2)
        self.assertEqual(len(onehot["one_hot"]["recovery"]), 2)
        self.assertEqual(len(onehot["one_hot"]["tool_cap"]), 2)

    def test_deterministic_export_with_grammar_treatments(self) -> None:
        """Multiple exports with the same grammar treatment produce the
        same model_input.treatment (dataset determinism)."""
        treatment = self._unbrowser_grammar_treatment(0)
        task = {
            "prompt": "task",
            "contract": [],
            "family": "unbrowser_fixture",
            "template_id": "single_page_extraction",
            "difficulty": "easy",
            "public_metadata": {},
        }
        first = build_model_input(
            task, treatment.id, treatment.version, treatment=treatment
        )
        second = build_model_input(
            task, treatment.id, treatment.version, treatment=treatment
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
