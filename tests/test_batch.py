from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pyreplab_harness.batch import (
    BatchRunSummary,
    BatchSpec,
    build_parser,
    default_preflight,
    expand_jobs,
    main as batch_main,
    parse_difficulties,
    parse_families,
    parse_seeds,
    run_batch,
    validate_spec,
)
from pyreplab_harness.treatments import TreatmentRegistry, generate_treatments


def pair_result(ok: bool = True) -> dict:
    return {
        "task_id": "task-1",
        "mode": "pair",
        "execution_order": ["direct", "deliberate"],
        "attempts": {
            "direct": {
                "attempt_id": "a1",
                "policy": {"id": "direct", "version": "1"},
                "pi_return_code": 0,
                "pi_stderr": "",
                "verification": {
                    "success": ok,
                    "verifier_id": "artifact",
                    "verifier_version": "1",
                },
                "usage": None,
            },
            "deliberate": {
                "attempt_id": "a2",
                "policy": {"id": "deliberate", "version": "1"},
                "pi_return_code": 0,
                "pi_stderr": "",
                "verification": {
                    "success": ok,
                    "verifier_id": "artifact",
                    "verifier_version": "1",
                },
                "usage": None,
            },
        },
    }


def single_result(ok: bool = True) -> dict:
    return {
        "task_id": "task-1",
        "attempt_id": "a1",
        "policy": {"id": "direct", "version": "1"},
        "pi_return_code": 0,
        "pi_stderr": "",
        "verification": {
            "success": ok,
            "verifier_id": "artifact",
            "verifier_version": "1",
        },
    }


def base_args(**overrides: object) -> dict:
    args = {
        "host": "ubuntu-local",
        "remote_project": "/remote/project",
        "remote_run_root": "/remote/runs",
        "remote_python": "python3",
        "pi": "pi",
        "model_switch_extension": "~/.pi/switch.ts",
    }
    args.update(overrides)
    return args


def read_records(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            continue
    return records


class SeedParsingTest(unittest.TestCase):
    def test_single_seed(self) -> None:
        self.assertEqual(parse_seeds("7"), (7,))

    def test_range(self) -> None:
        self.assertEqual(parse_seeds("1-3"), (1, 2, 3))

    def test_range_single_value(self) -> None:
        self.assertEqual(parse_seeds("5-5"), (5,))

    def test_csv(self) -> None:
        self.assertEqual(parse_seeds("1,3,5"), (1, 3, 5))

    def test_mixed_with_dedupe_and_order(self) -> None:
        self.assertEqual(parse_seeds("2-4,7,3"), (2, 3, 4, 7))

    def test_whitespace_tolerated(self) -> None:
        self.assertEqual(parse_seeds("1 , 3"), (1, 3))

    def test_invalid_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "seed range"):
            parse_seeds("3-1")

    def test_invalid_multi_dash(self) -> None:
        with self.assertRaises(ValueError):
            parse_seeds("1-2-3")


class FamilyDifficultyParsingTest(unittest.TestCase):
    def test_families(self) -> None:
        self.assertEqual(parse_families("artifact,sqlite"), ("artifact", "sqlite"))

    def test_families_unknown(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown family"):
            parse_families("bogus")

    def test_difficulties(self) -> None:
        self.assertEqual(parse_difficulties("easy, hard"), ("easy", "hard"))

    def test_difficulties_unknown(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown difficulty"):
            parse_difficulties("expert")


class SpecAndExpansionTest(unittest.TestCase):
    def test_spec_round_trip(self) -> None:
        spec = BatchSpec(
            families=("artifact", "sqlite"),
            difficulties=("easy", "hard"),
            seeds=(1, 2),
            pair=True,
        )
        self.assertEqual(BatchSpec.from_dict(spec.to_dict()), spec)

    def test_spec_single_policy_round_trip(self) -> None:
        spec = BatchSpec(
            families=("artifact",),
            difficulties=("medium",),
            seeds=(5,),
            pair=False,
            single_policy="direct",
        )
        decoded = BatchSpec.from_dict(spec.to_dict())
        self.assertEqual(decoded, spec)
        self.assertFalse(decoded.pair)

    def test_spec_default_pair(self) -> None:
        spec = BatchSpec(families=("artifact",), difficulties=("easy",), seeds=(1,))
        self.assertTrue(spec.pair)

    def test_expansion_is_deterministic_and_ordered(self) -> None:
        spec = BatchSpec(
            families=("artifact", "sqlite"),
            difficulties=("easy", "hard"),
            seeds=(1, 2),
        )
        jobs = expand_jobs(spec)
        self.assertEqual(
            [job.key for job in jobs],
            [
                "pair/artifact/easy/seed=1",
                "pair/artifact/easy/seed=2",
                "pair/artifact/hard/seed=1",
                "pair/artifact/hard/seed=2",
                "pair/sqlite/easy/seed=1",
                "pair/sqlite/easy/seed=2",
                "pair/sqlite/hard/seed=1",
                "pair/sqlite/hard/seed=2",
            ],
        )
        self.assertEqual(expand_jobs(spec), jobs)

    def test_expansion_single_mode(self) -> None:
        spec = BatchSpec(
            families=("artifact",),
            difficulties=("medium",),
            seeds=(5,),
            pair=False,
            single_policy="direct",
        )
        jobs = expand_jobs(spec)
        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        self.assertEqual(job.mode, "single")
        self.assertEqual(job.policy, "direct")
        self.assertEqual(job.key, "single/direct/artifact/medium/seed=5")

    def test_expansion_treatment_set_mode(self) -> None:
        spec = BatchSpec(
            families=("artifact",),
            difficulties=("easy",),
            seeds=(1, 2),
            treatment_refs=("policy-a", "policy-b@2"),
        )
        jobs = expand_jobs(spec)
        self.assertEqual([job.mode for job in jobs], ["treatment_set", "treatment_set"])
        self.assertEqual(jobs[0].policy, "policy-a,policy-b@2")
        self.assertTrue(jobs[0].key.startswith("treatment-set/"))
        self.assertNotEqual(jobs[0].key, jobs[1].key)


class ValidateSpecTest(unittest.TestCase):
    def test_empty_seeds(self) -> None:
        problems = validate_spec(BatchSpec(families=("artifact",), difficulties=("easy",), seeds=()))
        self.assertTrue(any("seeds" in problem for problem in problems))

    def test_unknown_family(self) -> None:
        problems = validate_spec(BatchSpec(families=("bogus",), difficulties=("easy",), seeds=(1,)))
        self.assertTrue(any("unknown family" in problem for problem in problems))

    def test_single_mode_requires_policy(self) -> None:
        problems = validate_spec(
            BatchSpec(
                families=("artifact",),
                difficulties=("easy",),
                seeds=(1,),
                pair=False,
                single_policy=None,
            )
        )
        self.assertTrue(any("single mode" in problem for problem in problems))

    def test_valid_spec_has_no_problems(self) -> None:
        spec = BatchSpec(families=("artifact",), difficulties=("easy",), seeds=(1,))
        self.assertEqual(validate_spec(spec), [])

    def test_treatment_set_does_not_require_legacy_single_policy(self) -> None:
        spec = BatchSpec(
            families=("artifact",),
            difficulties=("easy",),
            seeds=(1,),
            pair=False,
            treatment_refs=("policy-a",),
        )
        self.assertEqual(validate_spec(spec), [])

    def test_default_preflight_raises_on_invalid(self) -> None:
        bad = BatchSpec(families=("bogus",), difficulties=("easy",), seeds=(1,))
        with self.assertRaisesRegex(ValueError, "unknown family"):
            default_preflight(bad, base_args(), Path("unused.jsonl"))

    def test_default_preflight_creates_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "nested" / "runs.jsonl"
            spec = BatchSpec(families=("artifact",), difficulties=("easy",), seeds=(1,))
            default_preflight(spec, base_args(), out)
            self.assertTrue(out.parent.exists())

    def test_default_preflight_rejects_mixed_policy_versions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "runs.jsonl"
            out.write_text(
                json.dumps(
                    {
                        "key": "pair/artifact/easy/seed=1",
                        "status": "completed",
                        "policy_version": "1",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            spec = BatchSpec(families=("artifact",), difficulties=("easy",), seeds=(1,))
            with self.assertRaisesRegex(ValueError, "separate --output"):
                default_preflight(spec, base_args(policy_version="2"), out)


class RunBatchTest(unittest.TestCase):
    def test_pair_jobs_run_sequentially_with_expected_args(self) -> None:
        spec = BatchSpec(
            families=("artifact", "sqlite"),
            difficulties=("easy",),
            seeds=(1, 2),
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "runs.jsonl"
            calls: list[tuple[str, str, int, bool]] = []

            def fake_run_pair(project_root, config, args):
                calls.append((args.family, args.difficulty, args.seed, args.pair))
                return pair_result()

            with mock.patch(
                "pyreplab_harness.batch.run_pair", side_effect=fake_run_pair
            ) as fake_pair, mock.patch("pyreplab_harness.batch.run_single") as fake_single:
                summary = run_batch(spec, base_args(), out)

            self.assertEqual(
                calls,
                [
                    ("artifact", "easy", 1, True),
                    ("artifact", "easy", 2, True),
                    ("sqlite", "easy", 1, True),
                    ("sqlite", "easy", 2, True),
                ],
            )
            fake_single.assert_not_called()
            self.assertEqual(
                summary, BatchRunSummary(jobs_total=4, completed=4, error=0, skipped=0)
            )
            records = read_records(out)
            self.assertEqual(len(records), 4)
            self.assertTrue(all(record["status"] == "completed" for record in records))

    def test_single_mode_invokes_run_single(self) -> None:
        spec = BatchSpec(
            families=("artifact",),
            difficulties=("medium",),
            seeds=(3,),
            pair=False,
            single_policy="deliberate",
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "runs.jsonl"
            with mock.patch(
                "pyreplab_harness.batch.run_single", return_value=single_result()
            ) as fake_single, mock.patch("pyreplab_harness.batch.run_pair") as fake_pair:
                run_batch(spec, base_args(), out)

            fake_single.assert_called_once()
            runner_args = fake_single.call_args.args[2]
            self.assertEqual(runner_args.policy, "deliberate")
            self.assertFalse(runner_args.pair)
            fake_pair.assert_not_called()

    def test_treatment_set_invokes_registered_runner_and_records_hash(self) -> None:
        treatments = generate_treatments(2, seed=71)
        registry = TreatmentRegistry(tuple(treatments))
        spec = BatchSpec(
            families=("artifact",),
            difficulties=("easy",),
            seeds=(1,),
            treatment_refs=(treatments[0].id, treatments[1].id),
        )
        result = {
            "task_id": "task-1",
            "mode": "treatment_set",
            "attempts": {
                treatment.bundle_id: {"verification": {"success": True}}
                for treatment in treatments
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "registry.json"
            registry.save(registry_path)
            out = Path(tmp) / "runs.jsonl"
            args = base_args(treatment_registry=str(registry_path))
            with mock.patch(
                "pyreplab_harness.batch.run_registered_treatments",
                return_value=result,
            ) as registered, mock.patch(
                "pyreplab_harness.batch.run_pair"
            ) as pair, mock.patch("pyreplab_harness.batch.run_single") as single:
                summary = run_batch(spec, args, out)
            record = read_records(out)[0]
        self.assertEqual(summary.completed, 1)
        registered.assert_called_once()
        pair.assert_not_called()
        single.assert_not_called()
        runner_args = registered.call_args.args[2]
        self.assertEqual(
            runner_args.treatments,
            f"{treatments[0].id},{treatments[1].id}",
        )
        self.assertEqual(record["mode"], "treatment_set")
        self.assertEqual(record["treatment_registry_hash"], registry.registry_hash)

    def test_treatment_resume_rejects_registry_change(self) -> None:
        first = TreatmentRegistry(tuple(generate_treatments(2, seed=72)))
        second = TreatmentRegistry(tuple(generate_treatments(2, seed=73)))
        with tempfile.TemporaryDirectory() as tmp:
            first_path = Path(tmp) / "first.json"
            second_path = Path(tmp) / "second.json"
            first.save(first_path)
            second.save(second_path)
            output = Path(tmp) / "runs.jsonl"
            output.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "key": "treatment-set/x/artifact/easy/seed=1",
                        "treatment_registry_hash": first.registry_hash,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            spec = BatchSpec(
                families=("artifact",),
                difficulties=("easy",),
                seeds=(1,),
                treatment_refs=(second.treatments[0].id,),
            )
            with self.assertRaisesRegex(ValueError, "separate --output"):
                default_preflight(
                    spec,
                    base_args(treatment_registry=str(second_path)),
                    output,
                )

    def test_record_schema(self) -> None:
        spec = BatchSpec(families=("artifact",), difficulties=("easy",), seeds=(7,))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "runs.jsonl"
            with mock.patch(
                "pyreplab_harness.batch.run_pair", return_value=pair_result()
            ):
                run_batch(spec, base_args(), out)
            record = read_records(out)[0]
        for key in (
            "key",
            "family",
            "difficulty",
            "seed",
            "mode",
            "policy",
            "policy_version",
            "started_at",
            "finished_at",
            "duration_seconds",
            "status",
            "ok",
            "result",
        ):
            self.assertIn(key, record)
        self.assertEqual(record["key"], "pair/artifact/easy/seed=7")
        self.assertEqual(record["family"], "artifact")
        self.assertEqual(record["difficulty"], "easy")
        self.assertEqual(record["seed"], 7)
        self.assertEqual(record["mode"], "pair")
        self.assertIsNone(record["policy"])
        self.assertEqual(record["policy_version"], "1")
        self.assertEqual(record["status"], "completed")
        self.assertIs(record["ok"], True)
        self.assertGreaterEqual(record["duration_seconds"], 0)
        self.assertLessEqual(record["started_at"], record["finished_at"])

    def test_verification_failure_is_completed_outcome(self) -> None:
        spec = BatchSpec(families=("artifact",), difficulties=("easy",), seeds=(1,))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "runs.jsonl"
            with mock.patch(
                "pyreplab_harness.batch.run_pair", return_value=pair_result(ok=False)
            ):
                summary = run_batch(spec, base_args(), out)
            record = read_records(out)[0]
        self.assertEqual(summary.error, 0)
        self.assertEqual(summary.completed, 1)
        self.assertEqual(record["status"], "completed")
        self.assertIs(record["ok"], False)
        self.assertIn("result", record)

    def test_exception_is_recorded_and_batch_continues(self) -> None:
        spec = BatchSpec(families=("artifact",), difficulties=("easy",), seeds=(1, 2))

        def fake_run_pair(project_root, config, args):
            if args.seed == 1:
                raise RuntimeError("boom")
            return pair_result()

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "runs.jsonl"
            with mock.patch("pyreplab_harness.batch.run_pair", side_effect=fake_run_pair):
                summary = run_batch(spec, base_args(), out)
            records = read_records(out)
        self.assertEqual(summary.completed, 1)
        self.assertEqual(summary.error, 1)
        error_record = next(record for record in records if record["status"] == "error")
        self.assertEqual(error_record["key"], "pair/artifact/easy/seed=1")
        self.assertEqual(
            error_record["error"], {"type": "RuntimeError", "message": "boom"}
        )
        self.assertIn("started_at", error_record)
        self.assertIn("finished_at", error_record)
        self.assertIn("duration_seconds", error_record)
        self.assertNotIn("result", error_record)
        completed_record = next(
            record for record in records if record["status"] == "completed"
        )
        self.assertEqual(completed_record["key"], "pair/artifact/easy/seed=2")

    def test_resume_skips_completed_jobs(self) -> None:
        spec = BatchSpec(families=("artifact",), difficulties=("easy",), seeds=(1, 2))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "runs.jsonl"
            with mock.patch(
                "pyreplab_harness.batch.run_pair", return_value=pair_result()
            ) as fake:
                first = run_batch(spec, base_args(), out)
                self.assertEqual(fake.call_count, 2)
                second = run_batch(spec, base_args(), out, resume=True)
            self.assertEqual(fake.call_count, 2)
            self.assertEqual(first.completed, 2)
            self.assertEqual(second.skipped, 2)
            self.assertEqual(second.completed, 0)
            self.assertEqual(len(read_records(out)), 2)

    def test_no_resume_reruns_everything(self) -> None:
        spec = BatchSpec(families=("artifact",), difficulties=("easy",), seeds=(1, 2))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "runs.jsonl"
            with mock.patch(
                "pyreplab_harness.batch.run_pair", return_value=pair_result()
            ) as fake:
                run_batch(spec, base_args(), out)
                run_batch(spec, base_args(), out, resume=False)
            self.assertEqual(fake.call_count, 4)
            self.assertEqual(len(read_records(out)), 4)

    def test_resume_reruns_error_records(self) -> None:
        spec = BatchSpec(families=("artifact",), difficulties=("easy",), seeds=(1, 2))

        def fake_run_pair(project_root, config, args):
            if args.seed == 1:
                raise RuntimeError("boom")
            return pair_result()

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "runs.jsonl"
            with mock.patch("pyreplab_harness.batch.run_pair", side_effect=fake_run_pair):
                run_batch(spec, base_args(), out)
            # Second pass: the error job is retried; the completed job is skipped.
            with mock.patch("pyreplab_harness.batch.run_pair", side_effect=fake_run_pair) as fake:
                summary = run_batch(spec, base_args(), out, resume=True)
            self.assertEqual(fake.call_count, 1)
            self.assertEqual(summary.skipped, 1)
            self.assertEqual(summary.error, 1)

    def test_malformed_previous_jsonl_tolerated(self) -> None:
        spec = BatchSpec(families=("artifact",), difficulties=("easy",), seeds=(1, 2))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "runs.jsonl"
            out.write_text(
                '{"key": "pair/artifact/easy/seed=1", "status": "completed", "ok": true}\n'
                "this is not json\n"
                '{"key": "pair/artifact/easy/seed=2", \n',
                encoding="utf-8",
            )
            with mock.patch(
                "pyreplab_harness.batch.run_pair", return_value=pair_result()
            ) as fake:
                summary = run_batch(spec, base_args(), out, resume=True)
            self.assertEqual(fake.call_count, 1)
            self.assertEqual(summary.skipped, 1)
            self.assertEqual(summary.completed, 1)
            records = read_records(out)
            self.assertEqual(len(records), 2)  # seed=1 (pre-existing) + new seed=2 record
            self.assertEqual(len(out.read_text(encoding="utf-8").splitlines()), 4)

    def test_preflight_hook_called_with_args(self) -> None:
        spec = BatchSpec(families=("artifact",), difficulties=("easy",), seeds=(1,))
        seen = []

        def my_preflight(spec_, args_, output_path_):
            seen.append((spec_, args_, output_path_))

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "runs.jsonl"
            with mock.patch(
                "pyreplab_harness.batch.run_pair", return_value=pair_result()
            ) as fake:
                run_batch(spec, base_args(), out, preflight=my_preflight)
            self.assertEqual(len(seen), 1)
            self.assertIs(seen[0][0], spec)
            self.assertEqual(seen[0][1]["host"], "ubuntu-local")
            self.assertEqual(Path(seen[0][2]), out)
            self.assertEqual(fake.call_count, 1)

    def test_preflight_can_abort_batch_before_any_job(self) -> None:
        spec = BatchSpec(families=("artifact",), difficulties=("easy",), seeds=(1, 2))

        def raising_preflight(spec_, args_, output_path_):
            raise RuntimeError("abort")

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "runs.jsonl"
            with mock.patch("pyreplab_harness.batch.run_pair", return_value=pair_result()) as fake:
                with self.assertRaisesRegex(RuntimeError, "abort"):
                    run_batch(spec, base_args(), out, preflight=raising_preflight)
            fake.assert_not_called()
            self.assertFalse(out.exists())

    def test_progress_callback_receives_each_record(self) -> None:
        spec = BatchSpec(families=("artifact",), difficulties=("easy",), seeds=(1, 2))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "runs.jsonl"
            progress: list[dict] = []
            with mock.patch(
                "pyreplab_harness.batch.run_pair", return_value=pair_result()
            ):
                run_batch(spec, base_args(), out, progress=progress.append)
            self.assertEqual(len(progress), 2)
            self.assertEqual(
                [record["key"] for record in progress],
                ["pair/artifact/easy/seed=1", "pair/artifact/easy/seed=2"],
            )

    def test_minimal_mapping_requires_explicit_remote_paths(self) -> None:
        spec = BatchSpec(families=("artifact",), difficulties=("easy",), seeds=(1,))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "runs.jsonl"
            with mock.patch("pyreplab_harness.batch.run_pair", return_value=pair_result()) as fake:
                with self.assertRaisesRegex(ValueError, "explicit absolute remote path"):
                    run_batch(spec, {"pi": "custom-pi"}, out)
            fake.assert_not_called()


class CliTest(unittest.TestCase):
    def test_parser_accepts_batch_flags(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--families",
                "artifact,sqlite",
                "--difficulties",
                "easy,medium",
                "--seeds",
                "1-3,7",
                "--host",
                "myhost",
                "--output",
                "out.jsonl",
            ]
        )
        self.assertEqual(args.families, "artifact,sqlite")
        self.assertEqual(args.difficulties, "easy,medium")
        self.assertEqual(args.seeds, "1-3,7")
        self.assertEqual(args.host, "myhost")
        self.assertEqual(args.policy_version, "1")
        self.assertIsNone(args.single_policy)
        self.assertFalse(args.no_resume)

    def test_single_policy_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["--families", "artifact", "--seeds", "3", "--single-policy", "direct", "--output", "out.jsonl"]
        )
        self.assertEqual(args.single_policy, "direct")

    def test_treatment_registry_flags(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--families",
                "artifact",
                "--seeds",
                "3",
                "--treatment-registry",
                "registry.json",
                "--treatments",
                "a,b",
                "--output",
                "out.jsonl",
            ]
        )
        self.assertEqual(args.treatment_registry, "registry.json")
        self.assertEqual(args.treatments, "a,b")

    def test_policy_version_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--families",
                "artifact",
                "--seeds",
                "3",
                "--policy-version",
                "2",
                "--output",
                "out.jsonl",
            ]
        )
        self.assertEqual(args.policy_version, "2")

    def test_help_documents_sequential_execution(self) -> None:
        help_text = build_parser().format_help().lower()
        self.assertIn("strictly one job at a", help_text)
        self.assertIn("parallelism", help_text)
        self.assertIn("no cron", help_text)

    def test_main_runs_batch_and_prints_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "runs.jsonl"
            with mock.patch("pyreplab_harness.batch.run_batch") as fake:
                fake.return_value = BatchRunSummary(jobs_total=1, completed=1, error=0, skipped=0)
                rc = batch_main(["--families", "artifact", "--seeds", "3", "--output", str(out)])
            self.assertEqual(rc, 0)
            spec = fake.call_args.args[0]
            orchestrator_args = fake.call_args.args[1]
            self.assertEqual(spec.families, ("artifact",))
            self.assertEqual(spec.seeds, (3,))
            self.assertTrue(spec.pair)
            self.assertIsNone(spec.single_policy)
            self.assertEqual(orchestrator_args["policy_version"], "1")

    def test_main_forwards_policy_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "runs.jsonl"
            with mock.patch("pyreplab_harness.batch.run_batch") as fake:
                fake.return_value = BatchRunSummary(
                    jobs_total=1, completed=1, error=0, skipped=0
                )
                rc = batch_main(
                    [
                        "--families",
                        "artifact",
                        "--seeds",
                        "3",
                        "--policy-version",
                        "2",
                        "--output",
                        str(out),
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertEqual(fake.call_args.args[1]["policy_version"], "2")

    def test_main_single_policy_builds_single_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "runs.jsonl"
            with mock.patch("pyreplab_harness.batch.run_batch") as fake:
                fake.return_value = BatchRunSummary(jobs_total=1, completed=1, error=0, skipped=0)
                rc = batch_main(
                    [
                        "--families",
                        "artifact",
                        "--seeds",
                        "3",
                        "--single-policy",
                        "deliberate",
                        "--output",
                        str(out),
                    ]
                )
            self.assertEqual(rc, 0)
            spec = fake.call_args.args[0]
            self.assertFalse(spec.pair)
            self.assertEqual(spec.single_policy, "deliberate")

    def test_main_no_resume_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "runs.jsonl"
            with mock.patch("pyreplab_harness.batch.run_batch") as fake:
                fake.return_value = BatchRunSummary(jobs_total=1, completed=1, error=0, skipped=0)
                batch_main(
                    ["--families", "artifact", "--seeds", "3", "--output", str(out), "--no-resume"]
                )
            self.assertFalse(fake.call_args.kwargs["resume"])

    def test_main_exit_code_reflects_infra_errors_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "runs.jsonl"
            with mock.patch("pyreplab_harness.batch.run_batch") as fake:
                fake.return_value = BatchRunSummary(jobs_total=1, completed=0, error=1, skipped=0)
                rc = batch_main(["--families", "artifact", "--seeds", "3", "--output", str(out)])
            self.assertEqual(rc, 1)

    def test_main_rejects_unknown_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "runs.jsonl"
            rc = batch_main(["--families", "bogus", "--seeds", "3", "--output", str(out)])
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
