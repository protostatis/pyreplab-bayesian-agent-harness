from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from pyreplab_harness import outcome_model as om
from pyreplab_harness.treatments import (
    TreatmentSpec,
    generate_treatments,
    treatment_model_input_descriptor,
)

TORCH_REQUIRED = unittest.skipUnless(om.TORCH_AVAILABLE, "PyTorch is not installed")


def make_model_input(seed: int, policy_id: str, policy_version: str = "1") -> dict:
    return {
        "text": (
            f"repair the computation for batch {seed} ensuring totals match the "
            f"contract rows across files with medium difficulty constraints"
        ),
        "family": "artifact",
        "template_id": "template-v1",
        "difficulty": "medium",
        "public_metadata": {
            "rows": 10 + (seed % 7),
            "files": 2,
            "nested": {"depth": seed % 3},
            "verbose": (seed % 2 == 0),
        },
        "policy_id": policy_id,
        "policy_version": policy_version,
    }


def make_row(seed: int, policy_id: str, split: str, success: bool | None = None) -> dict:
    if success is None:
        success = bool((seed * 7 + len(policy_id)) % 3)
    return {
        "task_id": f"task-{seed}-{policy_id}",
        "attempt_id": f"att-{seed}-{policy_id}",
        "split": split,
        "verified_success": success,
        "usage": {"input": 100 * seed, "output": 40 * seed},
        "assistant_message_count": seed,
        "tool_call_count": seed % 5,
        "final_text_length": 10 + seed,
        "failure_code": None if success else "boom",
        "model_input": make_model_input(seed, policy_id),
    }


def make_treatment_model_input(
    seed: int, treatment: TreatmentSpec
) -> dict:
    value = make_model_input(seed, treatment.id, treatment.version)
    value["treatment"] = treatment_model_input_descriptor(treatment)
    return value


def write_synthetic_dataset(path: Path, n_train: int = 40, n_val: int = 6, n_test: int = 6) -> None:
    rows = []
    for index in range(n_train):
        rows.append(make_row(index, "direct" if index % 2 == 0 else "deliberate", "train"))
    for index in range(n_train, n_train + n_val):
        rows.append(make_row(index, "direct" if index % 2 == 0 else "deliberate", "validation"))
    for index in range(n_train + n_val, n_train + n_val + n_test):
        rows.append(make_row(index, "direct" if index % 2 == 0 else "deliberate", "test"))
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


class TokenizerTest(unittest.TestCase):
    def test_lowercase_regex_tokenization_is_deterministic(self) -> None:
        text = "Fix BATCH_42 and the Rows.FILE2 (case, sensitive, 1.5!)"
        first = om.tokenize_text(text)
        second = om.tokenize_text(text)
        self.assertEqual(first, second)
        self.assertIn("batch", first)
        self.assertIn("42", first)
        self.assertIn("rows", first)
        self.assertIn("file2", first)
        for token in first:
            self.assertEqual(token, token.lower())

    def test_empty_and_none_text(self) -> None:
        self.assertEqual(om.tokenize_text(""), [])
        self.assertEqual(om.tokenize_text(None), [])


class PreprocessorTest(unittest.TestCase):
    def _train_inputs(self) -> list[dict]:
        return [
            {
                "text": "join the orders table with customers then aggregate totals",
                "family": "artifact",
                "template_id": "join-v1",
                "difficulty": "easy",
                "public_metadata": {"rows": 3, "files": 1, "depth": 0},
                "policy_id": "direct",
                "policy_version": "1",
            },
            {
                "text": "repair the batch aggregation totals across files medium difficulty",
                "family": "sqlite",
                "template_id": "aggregate-v1",
                "difficulty": "medium",
                "public_metadata": {"rows": 10, "files": 2, "depth": 1},
                "policy_id": "deliberate",
                "policy_version": "1",
            },
            {
                "text": "migrate schema while preserving invariants hard constraints",
                "family": "python_repair",
                "template_id": "migrate-v2",
                "difficulty": "hard",
                "public_metadata": {"rows": 2, "files": 4},
                "policy_id": "direct",
                "policy_version": "2",
            },
        ]

    def test_vocab_and_encoding_are_deterministic_with_pad_unk(self) -> None:
        pre = om.Preprocessor(max_vocab=50, max_tokens=16).fit(self._train_inputs())
        again = om.Preprocessor(max_vocab=50, max_tokens=16).fit(self._train_inputs())
        self.assertEqual(pre.to_dict(), again.to_dict())
        self.assertEqual(pre.token_to_id[om._PAD], 0)
        self.assertEqual(pre.token_to_id[om._UNK], 1)

        x = pre.transform(self._train_inputs()[0])
        self.assertEqual(len(x["token_ids"]), 16)
        self.assertEqual(len(x["token_mask"]), 16)
        self.assertEqual(x["token_length"], sum(x["token_mask"]))
        self.assertEqual(x["token_ids"][0], pre.token_to_id["join"])
        self.assertEqual(x["token_mask"][0], 1)
        # Padding slots are PAD id with mask 0.
        self.assertEqual(x["token_ids"][-1], 0)
        self.assertEqual(x["token_mask"][-1], 0)

    def test_truncation_to_max_tokens(self) -> None:
        pre = om.Preprocessor(max_vocab=100, max_tokens=4).fit(self._train_inputs())
        x = pre.transform(self._train_inputs()[0])
        self.assertEqual(len(x["token_ids"]), 4)
        self.assertLessEqual(x["token_length"], 4)

    def test_train_only_unknown_handling(self) -> None:
        pre = om.Preprocessor(max_vocab=100, max_tokens=32).fit(self._train_inputs())
        # An unknown test-only numeric key is ignored entirely; known keys are
        # still standardized against the train moments.
        unknown = {
            "text": "join the frobnicated orders table with customers then aggregate totals",
            "family": "artifact",
            "template_id": "join-v1",
            "difficulty": "easy",
            "public_metadata": {
                "rows": 3,
                "files": 1,
                "depth": 0,
                "test_only_key": 999,
                "rows_new": 7,
            },
            "policy_id": "never-seen-policy",
            "policy_version": "v99",
        }
        x = pre.transform(unknown)
        self.assertNotIn("test_only_key", pre.numeric_keys)
        self.assertNotIn("rows_new", pre.numeric_keys)
        self.assertEqual(len(x["numeric"]), len(pre.numeric_keys))
        rows_index = pre.numeric_keys.index("rows")
        mean, std = pre.numeric_mean["rows"], pre.numeric_std["rows"]
        self.assertAlmostEqual(x["numeric"][rows_index], (3.0 - mean) / std)
        self.assertEqual(x["numeric_mask"][rows_index], 1)
        # Unknown category -> UNK id 0.
        self.assertEqual(x["policy_id"], 0)
        self.assertEqual(x["policy_version"], 0)
        # Unknown tokens -> UNK id 1.
        self.assertIn(1, x["token_ids"])
        self.assertNotIn("never-seen-policy", pre.cat_vocab["policy_id"])

    def test_numeric_missingness_and_standardization(self) -> None:
        pre = om.Preprocessor(max_vocab=10, max_tokens=8).fit(self._train_inputs())
        # rows appears as 3, 10, 2 -> mean 5, population std sqrt(38/3).
        mean = 5.0
        std = math.sqrt(38.0 / 3.0)
        x = pre.transform(self._train_inputs()[0])
        key = "rows"
        index = pre.numeric_keys.index(key)
        self.assertAlmostEqual(x["numeric"][index], (3.0 - mean) / std)
        self.assertEqual(x["numeric_mask"][index], 1)
        # "depth" is missing in the third input -> imputed 0.0, mask 0.
        missing = pre.transform(self._train_inputs()[2])
        depth_index = pre.numeric_keys.index("depth")
        self.assertEqual(missing["numeric"][depth_index], 0.0)
        self.assertEqual(missing["numeric_mask"][depth_index], 0)

    def test_bool_metadata_is_finite_numeric(self) -> None:
        pre = om.Preprocessor(max_vocab=10, max_tokens=8).fit(
            [
                {
                    "text": "a",
                    "family": "artifact",
                    "template_id": "t",
                    "difficulty": "easy",
                    "public_metadata": {"flag": True, "count": 5},
                    "policy_id": "direct",
                    "policy_version": "1",
                },
                {
                    "text": "b",
                    "family": "artifact",
                    "template_id": "t",
                    "difficulty": "easy",
                    "public_metadata": {"flag": False, "count": 3},
                    "policy_id": "direct",
                    "policy_version": "1",
                },
            ]
        )
        x = pre.transform(
            {
                "text": "b",
                "family": "artifact",
                "template_id": "t",
                "difficulty": "easy",
                "public_metadata": {"flag": True, "count": 9},
                "policy_id": "direct",
                "policy_version": "1",
            }
        )
        flag_index = pre.numeric_keys.index("flag")
        # Standardized True against train mean 0.5 / std 0.5.
        self.assertAlmostEqual(x["numeric"][flag_index], 1.0)
        self.assertEqual(x["numeric_mask"][flag_index], 1)

    def test_serialization_roundtrip(self) -> None:
        pre = om.Preprocessor(max_vocab=40, max_tokens=12).fit(self._train_inputs())
        serialized = json.dumps(pre.to_dict(), sort_keys=True)
        restored = om.Preprocessor.from_dict(json.loads(serialized))
        self.assertEqual(pre.to_dict(), restored.to_dict())
        for model_input in self._train_inputs():
            self.assertEqual(pre.transform(model_input), restored.transform(model_input))

    def test_leakage_guard_top_level_fields_cannot_change_x(self) -> None:
        pre = om.Preprocessor(max_vocab=100, max_tokens=32).fit(
            [make_row(i, "direct", "train")["model_input"] for i in range(10)]
        )
        row = make_row(3, "deliberate", "validation", success=True)
        x_before = pre.transform(row["model_input"])
        # Mutate every top-level post-action / audit field; the features must
        # stay identical and only the label may change.
        row["verified_success"] = False
        row["usage"] = {"input": 99999, "output": 99999}
        row["assistant_message_count"] = 9876
        row["tool_call_count"] = 1234
        row["final_text_length"] = 0
        row["failure_code"] = "leaked"
        row["task_id"] = "leaked-task"
        row["attempt_id"] = "leaked-attempt"
        x_after = pre.transform(row["model_input"])
        self.assertEqual(x_before, x_after)
        self.assertFalse(row["verified_success"])


class TreatmentPreprocessorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.treatments = generate_treatments(4, seed=91)
        self.inputs = [
            make_treatment_model_input(index, treatment)
            for index, treatment in enumerate(self.treatments)
        ]

    def test_treatment_descriptor_enables_separate_feature_path(self) -> None:
        pre = om.Preprocessor(max_vocab=500, max_tokens=64).fit(self.inputs)
        self.assertTrue(pre.treatment_enabled)
        transformed = pre.transform(self.inputs[0])
        self.assertEqual(len(transformed["treatment_token_ids"]), 64)
        self.assertEqual(
            len(transformed["treatment_numeric"]),
            len(om.TREATMENT_NUMERIC_FIELDS),
        )
        for field in om.TREATMENT_CATEGORICAL_FIELDS:
            self.assertIn(f"treatment_{field}", transformed)
        self.assertEqual(
            transformed["treatment_token_length"],
            sum(transformed["treatment_token_mask"]),
        )

    def test_unseen_ids_still_differ_through_descriptors(self) -> None:
        pre = om.Preprocessor(max_vocab=500, max_tokens=64).fit(self.inputs)
        first = TreatmentSpec(
            id="unseen-a",
            version="1",
            system_prompt="Plan carefully and verify every intermediate result.",
            allowed_tools=("bash",),
            max_output_tokens=1024,
            tool_call_limit=4,
            command_timeout_seconds=30,
            wall_time_limit_seconds=180,
        )
        second = TreatmentSpec(
            id="unseen-b",
            version="1",
            system_prompt="Execute directly in one pass without verification.",
            allowed_tools=("bash",),
            max_output_tokens=4096,
            tool_call_limit=12,
            command_timeout_seconds=60,
            wall_time_limit_seconds=600,
        )
        a = pre.transform(make_treatment_model_input(99, first))
        b = pre.transform(make_treatment_model_input(99, second))
        self.assertEqual(a["policy_id"], 0)
        self.assertEqual(b["policy_id"], 0)
        self.assertEqual(a["treatment_bundle_id"], 0)
        self.assertEqual(b["treatment_bundle_id"], 0)
        self.assertNotEqual(a["treatment_token_ids"], b["treatment_token_ids"])
        self.assertNotEqual(a["treatment_numeric"], b["treatment_numeric"])
        # Task features remain identical for the counterfactual pair.
        for key in ("token_ids", "token_mask", "token_length", "numeric", "numeric_mask"):
            self.assertEqual(a[key], b[key])

    def test_treatment_preprocessor_roundtrip(self) -> None:
        pre = om.Preprocessor(max_vocab=500, max_tokens=64).fit(self.inputs)
        payload = pre.to_dict()
        self.assertEqual(payload["version"], 2)
        restored = om.Preprocessor.from_dict(json.loads(json.dumps(payload)))
        self.assertTrue(restored.treatment_enabled)
        self.assertEqual(restored.to_dict(), payload)
        for model_input in self.inputs:
            self.assertEqual(pre.transform(model_input), restored.transform(model_input))

    def test_legacy_preprocessor_remains_treatment_disabled(self) -> None:
        legacy_inputs = [make_model_input(index, "direct") for index in range(3)]
        pre = om.Preprocessor(max_vocab=100, max_tokens=16).fit(legacy_inputs)
        self.assertFalse(pre.treatment_enabled)
        payload = pre.to_dict()
        self.assertEqual(payload["version"], 1)
        restored = om.Preprocessor.from_dict(payload)
        self.assertFalse(restored.treatment_enabled)
        self.assertNotIn("treatment_token_ids", restored.transform(legacy_inputs[0]))

    def test_id_only_counterfactual_is_rejected_for_descriptor_model(self) -> None:
        pre = om.Preprocessor(max_vocab=100, max_tokens=16).fit(self.inputs)
        with self.assertRaisesRegex(ValueError, "treatment descriptors"):
            om.score_policy_counterfactuals(None, pre, self.inputs[0])


class MetricsTest(unittest.TestCase):
    def test_perfect_predictions(self) -> None:
        y = [0.0, 0.0, 1.0, 1.0]
        p = [0.0, 0.0, 1.0, 1.0]
        metrics = om.compute_metrics(y, p, posterior_std=[0.05, 0.05, 0.05, 0.05])
        self.assertEqual(metrics["n"], 4)
        self.assertAlmostEqual(metrics["brier"], 0.0, places=6)
        self.assertAlmostEqual(metrics["log_loss"], 0.0, places=6)
        self.assertAlmostEqual(metrics["accuracy_05"], 1.0)
        self.assertAlmostEqual(metrics["ece"], 0.0, places=6)
        self.assertAlmostEqual(metrics["mean_posterior_std"], 0.05)

    def test_log_loss_is_finite_after_clipping(self) -> None:
        y = [1.0, 0.0]
        p = [0.0, 1.0]  # wrong and overconfident -> clipped, finite.
        value = om.log_loss(y, p)
        self.assertTrue(math.isfinite(value))
        self.assertGreater(value, 0.0)

    def test_empty_metrics_are_explicit_nulls(self) -> None:
        metrics = om.compute_metrics([], [])
        self.assertEqual(metrics["n"], 0)
        for key in ("log_loss", "brier", "accuracy_05", "ece", "mean_posterior_std"):
            self.assertIsNone(metrics[key])

    def test_ece_known_value(self) -> None:
        y = [1.0, 0.0, 0.0, 1.0]
        p = [0.9, 0.1, 0.1, 0.9]
        ece = om.expected_calibration_error(y, p, num_bins=2)
        # Bin [0, 0.5): p=0.1,0.1 with y=0,0 -> conf 0.1, acc 0, weight 0.5.
        # Bin [0.5, 1): p=0.9,0.9 with y=1,1 -> conf 0.9, acc 1, weight 0.5.
        self.assertAlmostEqual(ece, 0.1)


@TORCH_REQUIRED
class BayesianModelTest(unittest.TestCase):
    def _tensor_batch(self, n: int = 8) -> dict:
        pre = om.Preprocessor(max_vocab=50, max_tokens=16).fit(
            [make_model_input(i, "direct") for i in range(4)]
            + [make_model_input(i, "deliberate") for i in range(4)]
        )
        config = om.build_model_config(pre, text_dim=8, cat_dim=4, numeric_hidden=6, fusion_hidden=8)
        model = om.OutcomeModel(config)
        x = om.collate_transform(
            [pre.transform(make_model_input(i, "direct" if i % 2 else "deliberate")) for i in range(n)]
        )
        return model, x

    def test_bayesian_kl_is_positive_and_finite(self) -> None:
        model, _ = self._tensor_batch()
        kl = model.head.kl()
        value = float(kl)
        self.assertTrue(math.isfinite(value))
        self.assertGreater(value, 0.0)

    def test_forward_shapes(self) -> None:
        model, x = self._tensor_batch()
        output = model(x)
        self.assertEqual(tuple(output["logits"].shape), (8,))
        self.assertEqual(tuple(output["kl"].shape), ())
        self.assertTrue(bool(torch_is_finite(output["logits"])))

    def test_posterior_predict_shape_and_range(self) -> None:
        model, x = self._tensor_batch()
        posterior = model.posterior_predict(x, num_samples=50, seed=7)
        self.assertEqual(tuple(posterior["mean"].shape), (8,))
        self.assertEqual(tuple(posterior["std"].shape), (8,))
        self.assertEqual(tuple(posterior["quantiles"].shape), (5, 8))
        mean = posterior["mean"].tolist()
        std = posterior["std"].tolist()
        for value in mean:
            self.assertTrue(0.0 <= value <= 1.0)
        for value in std:
            self.assertGreaterEqual(value, 0.0)
        quantiles = posterior["quantiles"].tolist()
        for row in range(8):
            column = [quantiles[q][row] for q in range(5)]
            self.assertEqual(column, sorted(column))

    def test_tiny_synthetic_train_save_load_predict(self) -> None:
        import torch  # noqa: F401  (guaranteed present under this class)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "dataset.jsonl"
            artifact_dir = root / "artifacts"
            write_synthetic_dataset(dataset_path, n_train=40, n_val=6, n_test=6)

            result = om.train_model(
                dataset_path,
                artifact_dir,
                epochs=3,
                batch_size=8,
                seed=7,
                max_vocab=200,
                max_tokens=32,
                text_dim=8,
                cat_dim=4,
                numeric_hidden=6,
                fusion_hidden=8,
                num_samples=25,
                verbose=False,
            )
            for name in ("config.json", "preprocessor.json", "model.pt", "metrics.json"):
                self.assertTrue((artifact_dir / name).is_file())
            metrics = result["metrics"]
            self.assertIn("train", metrics)
            self.assertIn("validation", metrics)
            self.assertIn("test", metrics)
            self.assertGreater(metrics["train"]["n"], 0)
            self.assertIn("per_policy", metrics["train"])
            self.assertGreaterEqual(metrics["train"]["per_policy"].get("direct", {}).get("n", 0), 0)

            config, pre, model = om.load_artifacts(artifact_dir, device="cpu")
            self.assertEqual(config["model"]["vocab_size"], pre.vocab_size)
            model_input = make_model_input(99, "deliberate")
            prediction = om.predict_single(model, pre, model_input, num_samples=25, seed=11)
            self.assertTrue(0.0 <= prediction["mean"] <= 1.0)
            self.assertGreaterEqual(prediction["std"], 0.0)
            self.assertEqual(prediction["num_samples"], 25)
            self.assertEqual(sorted(prediction["quantiles"]), ["0.05", "0.25", "0.5", "0.75", "0.95"])
            quantile_values = list(prediction["quantiles"].values())
            self.assertEqual(quantile_values, sorted(quantile_values))

    def test_policy_counterfactual_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "dataset.jsonl"
            artifact_dir = root / "artifacts"
            write_synthetic_dataset(dataset_path, n_train=40, n_val=6, n_test=6)
            om.train_model(
                dataset_path,
                artifact_dir,
                epochs=2,
                batch_size=16,
                seed=3,
                max_vocab=200,
                max_tokens=32,
                text_dim=8,
                cat_dim=4,
                numeric_hidden=6,
                fusion_hidden=8,
                num_samples=25,
                verbose=False,
            )
            config, pre, model = om.load_artifacts(artifact_dir)
            model_input = make_model_input(123, "direct")
            scored = om.score_policy_counterfactuals(model, pre, model_input, num_samples=25, seed=5)
            self.assertEqual({entry["policy_id"] for entry in scored}, {"direct", "deliberate"})
            for entry in scored:
                self.assertIn("mean", entry)
                self.assertIn("std", entry)
                self.assertIn("quantiles", entry)
                self.assertTrue(0.0 <= entry["mean"] <= 1.0)

    def test_treatment_encoder_maps_unseen_descriptors_to_distinct_vectors(self) -> None:
        treatments = generate_treatments(6, seed=33)
        train_inputs = [
            make_treatment_model_input(index, treatment)
            for index, treatment in enumerate(treatments[:4])
        ]
        pre = om.Preprocessor(max_vocab=500, max_tokens=64).fit(train_inputs)
        config = om.build_model_config(
            pre, text_dim=8, cat_dim=4, numeric_hidden=6, fusion_hidden=12
        )
        model = om.OutcomeModel(config)
        unseen = [
            pre.transform(make_treatment_model_input(99, treatment))
            for treatment in treatments[4:]
        ]
        self.assertEqual(unseen[0]["policy_id"], 0)
        self.assertEqual(unseen[1]["policy_id"], 0)
        batch = om.collate_transform(unseen)
        encoded = model.encode(batch)
        self.assertEqual(tuple(encoded.shape), (2, 12))
        self.assertFalse(bool((encoded[0] == encoded[1]).all()))

    def test_treatment_model_train_save_load_and_full_counterfactuals(self) -> None:
        treatments = generate_treatments(5, seed=44)
        rows = []
        for index in range(50):
            treatment = treatments[index % 4]
            split = "train" if index < 36 else ("validation" if index < 44 else "test")
            rows.append(
                {
                    "task_id": f"task-{index // 4}",
                    "attempt_id": f"attempt-{index}",
                    "split": split,
                    "verified_success": bool((index + treatment.tool_call_limit) % 3),
                    "model_input": make_treatment_model_input(index, treatment),
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "dataset.jsonl"
            artifact_dir = root / "model"
            dataset_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            om.train_model(
                dataset_path,
                artifact_dir,
                epochs=2,
                batch_size=12,
                seed=8,
                max_vocab=500,
                max_tokens=64,
                text_dim=8,
                cat_dim=4,
                numeric_hidden=6,
                fusion_hidden=12,
                num_samples=16,
                verbose=False,
            )
            _config, pre, model = om.load_artifacts(artifact_dir)
            self.assertTrue(pre.treatment_enabled)
            scored = om.score_treatment_counterfactuals(
                model,
                pre,
                make_treatment_model_input(100, treatments[0]),
                [treatments[0], treatments[4]],
                num_samples=16,
                seed=9,
            )
            self.assertEqual(len(scored), 2)
            self.assertEqual(scored[1]["bundle_id"], treatments[4].bundle_id)
            for result in scored:
                self.assertTrue(0.0 <= result["mean"] <= 1.0)

    def test_deterministic_seeded_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "dataset.jsonl"
            artifact_dir = root / "artifacts"
            write_synthetic_dataset(dataset_path, n_train=40, n_val=6, n_test=6)
            om.train_model(
                dataset_path,
                artifact_dir,
                epochs=2,
                batch_size=16,
                seed=42,
                max_vocab=200,
                max_tokens=32,
                text_dim=8,
                cat_dim=4,
                numeric_hidden=6,
                fusion_hidden=8,
                num_samples=25,
                verbose=False,
            )
            config, pre, model = om.load_artifacts(artifact_dir)
            model_input = make_model_input(55, "deliberate")
            first = om.predict_single(model, pre, model_input, num_samples=25, seed=99)
            second = om.predict_single(model, pre, model_input, num_samples=25, seed=99)
            self.assertEqual(first, second)
            # Retraining with the same seed reproduces the same metrics.
            other_dir = root / "artifacts-again"
            om.train_model(
                dataset_path,
                other_dir,
                epochs=2,
                batch_size=16,
                seed=42,
                max_vocab=200,
                max_tokens=32,
                text_dim=8,
                cat_dim=4,
                numeric_hidden=6,
                fusion_hidden=8,
                num_samples=25,
                verbose=False,
            )
            metrics_a = json.loads((artifact_dir / "metrics.json").read_text())
            metrics_b = json.loads((other_dir / "metrics.json").read_text())
            for split in ("train", "validation", "test"):
                for key in ("n", "log_loss", "brier", "accuracy_05", "ece"):
                    self.assertAlmostEqual(metrics_a[split][key], metrics_b[split][key], places=5)

    def test_empty_splits_produce_null_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "dataset.jsonl"
            artifact_dir = root / "artifacts"
            rows = [make_row(i, "direct", "train") for i in range(30)]
            with dataset_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            om.train_model(
                dataset_path,
                artifact_dir,
                epochs=2,
                batch_size=16,
                seed=9,
                max_vocab=100,
                max_tokens=32,
                text_dim=8,
                cat_dim=4,
                numeric_hidden=6,
                fusion_hidden=8,
                num_samples=20,
                verbose=False,
            )
            metrics = json.loads((artifact_dir / "metrics.json").read_text())
            for split in ("validation", "test"):
                self.assertEqual(metrics[split]["n"], 0)
                for key in ("log_loss", "brier", "accuracy_05", "ece", "mean_posterior_std"):
                    self.assertIsNone(metrics[split][key])
                self.assertEqual(metrics[split]["per_policy"], {})

    def test_evaluate_cli_and_predict_cli(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "dataset.jsonl"
            artifact_dir = root / "artifacts"
            write_synthetic_dataset(dataset_path, n_train=40, n_val=6, n_test=6)
            code = om.main(["train", str(dataset_path), str(artifact_dir), "--epochs", "2", "--batch-size", "16", "--max-tokens", "32", "--max-vocab", "200", "--text-dim", "8", "--cat-dim", "4", "--numeric-hidden", "6", "--fusion-hidden", "8", "--num-samples", "20"])
            self.assertEqual(code, 0)
            code = om.main(["evaluate", str(dataset_path), str(artifact_dir), "--num-samples", "20"])
            self.assertEqual(code, 0)
            model_input_path = root / "input.json"
            model_input_path.write_text(json.dumps(make_model_input(7, "direct")))
            code = om.main(["predict", str(model_input_path), str(artifact_dir), "--num-samples", "20"])
            self.assertEqual(code, 0)
            code = om.main(
                [
                    "inspect",
                    str(artifact_dir),
                    "--dataset",
                    str(dataset_path),
                    "--posterior-samples",
                    "16",
                    "--prior-samples",
                    "16",
                    "--max-rows",
                    "10",
                ]
            )
            self.assertEqual(code, 0)


def torch_is_finite(tensor) -> bool:
    import torch

    return bool(torch.isfinite(tensor).all())


if __name__ == "__main__":
    unittest.main()
