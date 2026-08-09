from __future__ import annotations

import io
import json
import math
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from pyreplab_harness import dashboard as db


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------
def make_dataset_row(
    task_id: str,
    attempt_id: str,
    policy: str,
    success: bool,
    *,
    family: str = "artifact",
    difficulty: str = "easy",
    split: str = "train",
    seed: int = 1,
    template_id: str = "template-v1",
    generator_version: str = "1",
    policy_version: str = "1",
    failure_code: str | None = None,
    verifier_id: str = "verifier-v1",
    verifier_version: str = "1",
    usage: dict | None = None,
    tool_calls: int = 3,
    messages: int = 2,
    text_len: int = 12,
    prompt: str = "Solve the task and write result.json to the workspace.",
    contract: list[str] | None = None,
    public_metadata: dict | None = None,
    model_input_metadata: dict | None = None,
    **extra: Any,
) -> dict:
    """A full dataset.py-shaped row (leakage-safe JSONL schema)."""
    row = {
        "task_id": task_id,
        "family": family,
        "template_id": template_id,
        "generator_version": generator_version,
        "seed": seed,
        "difficulty": difficulty,
        "prompt": prompt,
        "contract": contract if contract is not None else ["produce result.json"],
        "public_metadata": (
            public_metadata if public_metadata is not None else {"rows": 5}
        ),
        "attempt_id": attempt_id,
        "policy_id": policy,
        "policy_version": policy_version,
        "split": split,
        "verified_success": success,
        "failure_code": failure_code,
        "verifier_id": verifier_id,
        "verifier_version": verifier_version,
        "usage": usage if usage is not None else {"input": 100, "output": 40, "total_tokens": 140},
        "assistant_message_count": messages,
        "tool_call_count": tool_calls,
        "final_text_length": text_len,
        "model_input": {
            "text": f"Predecision text for {task_id} {policy}",
            "family": family,
            "template_id": template_id,
            "difficulty": difficulty,
            "public_metadata": (
                model_input_metadata if model_input_metadata is not None else {"rows": 5}
            ),
            "policy_id": policy,
            "policy_version": policy_version,
        },
    }
    row.update(extra)
    return row


def write_dataset(root: Path, rows: list[dict], name: str = "dataset.jsonl") -> Path:
    path = root / name
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    return path


def build_data(root: Path, metrics: Path | None = None, baselines: Path | None = None) -> dict:
    return db.build_dashboard_data(root / "dataset.jsonl", metrics, baselines)


def render_for(root: Path, metrics: Path | None = None, baselines: Path | None = None) -> str:
    return db.render_dashboard(build_data(root, metrics, baselines))


def extract_embedded_json(html_doc: str) -> dict:
    match = re.search(r"var DATA = (.*?);\nvar CAP", html_doc, re.S)
    assert match, "embedded DATA payload not found"
    return json.loads(match.group(1))


def _metric_file(root: Path, payload: Any, name: str = "metrics.json") -> Path:
    path = root / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _baseline_file(root: Path, payload: Any) -> Path:
    path = root / "baselines.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Privacy whitelist (ship-blocking)
# ---------------------------------------------------------------------------
class PrivacyWhitelistTest(unittest.TestCase):
    # Every forbidden field carries a unique secret sentinel string.
    FORBIDDEN = {
        "workspace_ref": "PATH_SECRET_WORKSPACE",
        "verifier_ref": "PATH_SECRET_VERIFIER",
        "pi_events_ref": "PATH_SECRET_EVENTS",
        "normalized_events_ref": "PATH_SECRET_NORMALIZED",
        "verification_ref": "PATH_SECRET_VERIFICATION",
        "oracle": "ORACLE_SECRET_42",
        "private_metadata": "PRIVATE_METADATA_SECRET",
        "expected": "EXPECTED_SECRET_VALUE",
        "diagnostics": "DIAGNOSTICS_SECRET_TRACEBACK",
        "events": "EVENTS_SECRET_RAW",
        "trajectory": "TRAJECTORY_SECRET_RAW",
        "raw_trajectory": "RAW_TRAJECTORY_SECRET",
        "final_text": "FINAL_TEXT_SECRET",
        "model": "MODEL_ID_SECRET",
        "provider": "PROVIDER_SECRET",
        "system_prompt": "SYSTEM_PROMPT_SECRET",
        "host": "HOST_SECRET_NAME",
        "model_config": "MODEL_CONFIG_SECRET",
        "posterior": "POSTERIOR_SAMPLE_SECRET",
        "posterior_samples": "POSTERIOR_SAMPLES_SECRET",
        "per_task_std": "PER_TASK_STD_SECRET",
        "samples": "SAMPLES_SECRET",
    }

    def _secret_rows(self) -> list[dict]:
        long_prompt = ("word " * 80) + "PROMPT_SECRET_BEYOND_PREVIEW_120"
        contract_secrets = ["CONTRACT_SECRET_FIRST", "CONTRACT_SECRET_SECOND"]
        raw_metadata = {
            "customer_name": "PM_SECRET_NAME",
            "tags": ["PM_SECRET_TAG"],
            "notes": {"inner": "PM_SECRET_INNER"},
            "inf": float("inf"),
            "nan": float("nan"),
            "safe_count": 7,
        }
        row = make_dataset_row(
            "t-secret",
            "a-secret",
            "direct",
            True,
            prompt=long_prompt,
            contract=contract_secrets,
            public_metadata=raw_metadata,
            model_input_metadata={"safe_count": 7},
            **self.FORBIDDEN,
        )
        return [row]

    def test_forbidden_sentinels_absent_from_html(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dataset(root, self._secret_rows())
            html_doc = render_for(root)
            for name, sentinel in self.FORBIDDEN.items():
                with self.subTest(field=name):
                    self.assertNotIn(sentinel, html_doc)
            for sentinel in ("PROMPT_SECRET_BEYOND_PREVIEW_120",):
                self.assertNotIn(sentinel, html_doc)
            for sentinel in ("CONTRACT_SECRET_FIRST", "CONTRACT_SECRET_SECOND"):
                self.assertNotIn(sentinel, html_doc)
            for sentinel in ("PM_SECRET_NAME", "PM_SECRET_TAG", "PM_SECRET_INNER"):
                self.assertNotIn(sentinel, html_doc)
            self.assertNotIn("Predecision text for t-secret", html_doc)

    def test_forbidden_sentinels_absent_from_embedded_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dataset(root, self._secret_rows())
            html_doc = render_for(root)
            payload = extract_embedded_json(html_doc)
            blob = json.dumps(payload, sort_keys=True)
            for name, sentinel in self.FORBIDDEN.items():
                with self.subTest(field=name):
                    self.assertNotIn(sentinel, blob)

    def test_public_row_contains_only_whitelist_keys(self) -> None:
        row = make_dataset_row("t1", "a1", "direct", True, **self.FORBIDDEN)
        public = db._row_to_public(row)
        allowed = {
            "task_id", "family", "template_id", "generator_version", "seed",
            "difficulty", "prompt_preview", "attempt_id", "policy_id",
            "policy_version", "split", "verified_success", "failure_code",
            "verifier_id", "verifier_version", "usage",
            "assistant_message_count", "tool_call_count", "final_text_length",
            "metadata",
        }
        self.assertEqual(set(public), allowed)
        blob = json.dumps(public, sort_keys=True)
        for name, sentinel in self.FORBIDDEN.items():
            with self.subTest(field=name):
                self.assertNotIn(sentinel, blob)

    def test_prompt_preview_is_capped_and_sanitized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            long_prompt = "  ".join(["alpha"] * 100)  # 100 words, >120 chars after collapse
            write_dataset(root, [make_dataset_row("t1", "a1", "direct", True, prompt=long_prompt)])
            data = build_data(root)
            preview = data["rows"][0]["prompt_preview"]
            self.assertLessEqual(len(preview), db.PROMPT_PREVIEW_LIMIT + 1)
            self.assertFalse("\n" in preview)
            self.assertTrue(preview.endswith("…"))
            # Whitespace is collapsed; the raw multi-space prompt is never embedded.
            self.assertNotIn("  ", preview)

    def test_metrics_and_baselines_drop_forbidden_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dataset(root, [make_dataset_row("t1", "a1", "direct", True)])
            metrics = _metric_file(
                root,
                {
                    "train": {
                        "n": 1,
                        "log_loss": 0.5,
                        "brier": 0.2,
                        "accuracy_05": 0.8,
                        "ece": 0.1,
                        "mean_posterior_std": 0.05,
                        "per_policy": {},
                    },
                    "posterior_samples": "METRICS_POSTERIOR_SECRET",
                    "per_task_std": "METRICS_PER_TASK_SECRET",
                    "raw": {"host": "METRICS_HOST_SECRET"},
                    "notes": "METRICS_NOTES_SECRET",
                    "paths": ["/tmp/METRICS_PATH_SECRET"],
                },
            )
            baselines = _baseline_file(
                root,
                {
                    "always-direct": {
                        "verified_success_rate": 0.7,
                        "n": 10,
                        "notes": "BASELINES_NOTES_SECRET",
                        "host": {"name": "BASELINES_HOST_SECRET"},
                        "paths": ["/tmp/BASELINES_PATH_SECRET"],
                    }
                },
            )
            data = build_data(root, metrics, baselines)
            html_doc = db.render_dashboard(data)
            for sentinel in (
                "METRICS_POSTERIOR_SECRET",
                "METRICS_PER_TASK_SECRET",
                "METRICS_HOST_SECRET",
                "METRICS_NOTES_SECRET",
                "METRICS_PATH_SECRET",
                "BASELINES_NOTES_SECRET",
                "BASELINES_HOST_SECRET",
                "BASELINES_PATH_SECRET",
            ):
                with self.subTest(sentinel=sentinel):
                    self.assertNotIn(sentinel, html_doc)

    def test_absolute_paths_and_username_never_embedded(self) -> None:
        """Header/export metadata must carry basenames, never absolute paths.

        An absolute path may expose the owning username or the host layout, so
        it must be absent from both the static HTML *and* the embedded DATA
        JSON payload, not merely hidden by rendering.
        """
        sentinel = "USER_SECRET_PATHxyz"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            secret_dir = root / sentinel
            secret_dir.mkdir()
            dataset_path = write_dataset(
                secret_dir,
                [make_dataset_row("t1", "a", "direct", True)],
                name="run-dataset.jsonl",
            )
            metrics = _metric_file(
                secret_dir,
                {
                    "train": {
                        "n": 1, "log_loss": 0.5, "brier": 0.2,
                        "accuracy_05": 0.8, "ece": 0.1,
                        "mean_posterior_std": 0.05, "per_policy": {},
                    }
                },
                name="run-metrics.json",
            )
            baselines = _baseline_file(secret_dir, {"direct": {"success_rate": 1.0, "n": 1}})
            data = db.build_dashboard_data(dataset_path, metrics, baselines)
            meta = data["meta"]
            # No absolute-path keys anywhere in the metadata dict.
            for key in ("dataset_path", "metrics_path", "baselines_path"):
                self.assertNotIn(key, meta)
            self.assertEqual(meta["dataset_basename"], "run-dataset.jsonl")
            self.assertEqual(meta["metrics_basename"], "run-metrics.json")
            self.assertEqual(meta["baselines_basename"], "baselines.json")
            self.assertNotIn(sentinel, json.dumps(meta, sort_keys=True))
            html_doc = db.render_dashboard(data)
            self.assertNotIn(sentinel, html_doc)
            for key in ("dataset_path", "metrics_path", "baselines_path"):
                self.assertNotIn(key, html_doc)
            payload = extract_embedded_json(html_doc)
            self.assertNotIn(sentinel, json.dumps(payload, sort_keys=True))
            for key in ("dataset_path", "metrics_path", "baselines_path"):
                self.assertNotIn(key, json.dumps(payload))
            # Basenames are shown as the source labels.
            self.assertIn("run-dataset.jsonl", html_doc)
            self.assertIn("run-metrics.json", html_doc)
            self.assertIn("baselines.json", html_doc)


# ---------------------------------------------------------------------------
# HTML escaping / script breakout
# ---------------------------------------------------------------------------
class EscapingTest(unittest.TestCase):
    def test_script_breakout_is_impossible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sneaky = "</script><script>alert(1)</script>"
            rows = [
                make_dataset_row(
                    "t1",
                    "a1",
                    "direct",
                    True,
                    family='<img src=x onerror=alert(2)>',
                    difficulty=sneaky,
                    prompt=f"prefix {sneaky} suffix",
                    template_id=sneaky,
                    verifier_id=sneaky,
                    model_input_metadata={"key": f"v{sneaky}"},
                    public_metadata={"x": sneaky},
                )
            ]
            write_dataset(root, rows)
            html_doc = render_for(root)
            # Exactly one real closing script tag (the legitimate one): any
            # embedded </script> would add another.
            self.assertEqual(html_doc.count("</script>"), 1)
            self.assertIn("\\u003c", html_doc)   # JSON escaping of '<'
            self.assertIn("&lt;", html_doc)      # static-markup escaping of '<'
            self.assertNotIn("<script>alert(1)", html_doc)
            self.assertNotIn("<img", html_doc)
            # The payload round-trips unchanged, proving the value survived
            # escaping rather than being stripped or mangled.
            payload = extract_embedded_json(html_doc)
            self.assertEqual(
                payload["rows"][0]["difficulty"], "</script><script>alert(1)</script>"
            )

    def test_static_markup_html_escapes_untrusted_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                make_dataset_row(
                    "t<1>&", "a<&>1", "direct", True, family="fam&<",
                    failure_code=None, verifier_id="v<&>",
                )
            ]
            write_dataset(root, rows)
            html_doc = render_for(root)
            # Raw untrusted values must never appear unescaped in the document.
            for raw in ("t<1>&", "a<&>1", "fam&<", "v<&>"):
                self.assertNotIn(raw, html_doc)
            # The escaped forms are used in static markup.
            self.assertIn("t&lt;1&gt;&amp;", html_doc)
            self.assertIn("fam&amp;&lt;", html_doc)
            # And the embedded payload carries the escaped JSON forms.
            self.assertIn("a\\u003c\\u0026\\u003e1", html_doc)

    def test_js_never_uses_inner_html(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dataset(root, [make_dataset_row("t1", "a1", "direct", True)])
            html_doc = render_for(root)
            self.assertNotIn("innerHTML", html_doc)
            self.assertIn("textContent", html_doc)


# ---------------------------------------------------------------------------
# Aggregate correctness
# ---------------------------------------------------------------------------
class AggregateCorrectnessTest(unittest.TestCase):
    def _primary_dataset(self) -> list[dict]:
        tokens_direct = {"input": 60, "output": 40, "total_tokens": 100}
        tokens_deliberate = {"input": 120, "output": 80, "total_tokens": 200}
        return [
            # both succeed
            make_dataset_row("t1", "t1-d", "direct", True, usage=tokens_direct, tool_calls=2, messages=1),
            make_dataset_row("t1", "t1-b", "deliberate", True, usage=tokens_deliberate, tool_calls=4, messages=2),
            # direct-only win
            make_dataset_row("t2", "t2-d", "direct", True, usage=tokens_direct, tool_calls=2, messages=1),
            make_dataset_row("t2", "t2-b", "deliberate", False, failure_code="timeout", usage=tokens_deliberate, tool_calls=4, messages=2),
            # deliberate-only win
            make_dataset_row("t3", "t3-d", "direct", False, failure_code="missing_output", usage=tokens_direct, tool_calls=2, messages=1),
            make_dataset_row("t3", "t3-b", "deliberate", True, usage=tokens_deliberate, tool_calls=4, messages=2),
            # both fail
            make_dataset_row("t4", "t4-d", "direct", False, failure_code="missing_output", usage=tokens_direct, tool_calls=2, messages=1),
            make_dataset_row("t4", "t4-b", "deliberate", False, failure_code="missing_output", usage=tokens_deliberate, tool_calls=4, messages=2),
            # unpaired direct attempt
            make_dataset_row("t5", "t5-d", "direct", True, usage=tokens_direct, tool_calls=2, messages=1),
        ]

    def test_gym_health_aggregates(self) -> None:
        rows = self._primary_dataset()
        public = [db._row_to_public(row) for row in rows]
        gym = db.aggregate_gym(public)
        self.assertEqual(gym["total_rows"], 9)
        self.assertEqual(gym["total_tasks"], 5)
        self.assertEqual(gym["paired_tasks"], 4)
        self.assertAlmostEqual(gym["task_coverage"], 0.8)
        self.assertEqual(gym["success_count"], 5)
        self.assertAlmostEqual(gym["overall_success_rate"], 5 / 9)
        self.assertAlmostEqual(gym["avg_total_tokens"], 1300 / 9, places=6)
        artifact = [c for c in gym["family_policy"] if c["family"] == "artifact"]
        direct = next(c for c in artifact if c["policy"] == "direct")
        self.assertEqual(direct["n"], 5)
        self.assertEqual(direct["success"], 3)
        self.assertAlmostEqual(direct["success_rate"], 0.6)

    def test_paired_matrix_by_task_id(self) -> None:
        rows = self._primary_dataset()
        public = [db._row_to_public(row) for row in rows]
        dis = db.policy_disagreement(public)
        self.assertEqual(dis["policy_a"], "direct")
        self.assertEqual(dis["policy_b"], "deliberate")
        self.assertEqual(dis["paired_tasks"], 4)
        self.assertEqual(dis["unpaired_tasks"], 1)
        self.assertEqual(
            dis["matrix"], {"both_ok": 1, "a_only": 1, "b_only": 1, "both_fail": 1}
        )
        self.assertEqual(dis["disagreement_count"], 2)
        self.assertAlmostEqual(dis["disagreement_rate"], 0.5)
        family_row = dis["by_family"][0]
        self.assertEqual(family_row["family"], "artifact")
        self.assertEqual(family_row["paired"], 4)
        self.assertAlmostEqual(family_row["disagreement_rate"], 0.5)

    def test_failure_analysis(self) -> None:
        rows = self._primary_dataset()
        public = [db._row_to_public(row) for row in rows]
        failures = db.failure_analysis(public)
        self.assertEqual(failures["failure_count"], 4)
        self.assertAlmostEqual(failures["failure_rate"], 4 / 9)
        family = failures["by_family"][0]
        self.assertEqual(family["family"], "artifact")
        self.assertEqual(family["total"], 9)
        self.assertEqual(family["failures"], 4)
        by_code = {c["code"]: c for c in failures["by_code"]}
        self.assertEqual(by_code["missing_output"]["count"], 3)
        self.assertEqual(by_code["timeout"]["count"], 1)
        self.assertAlmostEqual(by_code["missing_output"]["share_of_failures"], 0.75)
        self.assertAlmostEqual(by_code["timeout"]["failure_rate"], 1 / 9)

    def test_cost_statistics(self) -> None:
        rows = [
            make_dataset_row("t1", "a", "direct", True, usage={"total_tokens": 100}, tool_calls=2, messages=1),
            make_dataset_row("t2", "b", "direct", True, usage={"total_tokens": 200}, tool_calls=4, messages=1),
            make_dataset_row("t3", "c", "direct", True, usage={"total_tokens": 300}, tool_calls=6, messages=1),
            make_dataset_row("t4", "d", "deliberate", True, usage={"total_tokens": 150}, tool_calls=3, messages=2),
            make_dataset_row("t5", "e", "deliberate", True, usage={"total_tokens": 250}, tool_calls=5, messages=2),
        ]
        public = [db._row_to_public(row) for row in rows]
        costs = db.cost_stats(public)
        by_key = {(c["family"], c["policy"]): c for c in costs["rows"]}
        direct = by_key[("artifact", "direct")]
        self.assertEqual(direct["n"], 3)
        self.assertAlmostEqual(direct["total_tokens"]["mean"], 200.0)
        self.assertAlmostEqual(direct["total_tokens"]["std"], math.sqrt(20000 / 3))
        self.assertEqual(direct["total_tokens"]["min"], 100.0)
        self.assertEqual(direct["total_tokens"]["max"], 300.0)
        self.assertAlmostEqual(direct["tool_calls"]["mean"], 4.0)
        self.assertAlmostEqual(direct["tool_calls"]["std"], math.sqrt(8 / 3))
        self.assertAlmostEqual(direct["assistant_messages"]["std"], 0.0)
        deliberate = by_key[("artifact", "deliberate")]
        self.assertEqual(deliberate["n"], 2)
        self.assertAlmostEqual(deliberate["total_tokens"]["mean"], 200.0)
        self.assertAlmostEqual(deliberate["total_tokens"]["std"], 50.0)
        self.assertEqual(deliberate["total_tokens"]["min"], 150.0)
        self.assertEqual(deliberate["total_tokens"]["max"], 250.0)

    def test_pair_representative_attempt_first_by_attempt_id(self) -> None:
        rows = [
            # Two direct attempts on the same task: "a-first" succeeds, "z-late" fails.
            make_dataset_row("tx", "z-late", "direct", False),
            make_dataset_row("tx", "a-first", "direct", True),
            make_dataset_row("tx", "only", "deliberate", True),
        ]
        public = [db._row_to_public(row) for row in rows]
        dis = db.policy_disagreement(public)
        self.assertEqual(dis["paired_tasks"], 1)
        self.assertEqual(dis["matrix"]["both_ok"], 1)
        self.assertEqual(dis["matrix"]["a_only"], 0)

    def test_static_html_contains_computed_aggregates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dataset(root, self._primary_dataset())
            html_doc = render_for(root)
            self.assertIn("80.0%", html_doc)  # task coverage
            self.assertIn("50.0%", html_doc)  # paired disagreement
            self.assertIn("55.6%", html_doc)  # overall success 5/9
            self.assertIn(">144<", html_doc)  # avg tokens 144.4 -> 144
            self.assertIn("missing_output", html_doc)
            self.assertIn("timeout", html_doc)
            # matrix cells (paired outcome matrix rendered statically)
            self.assertIn('>1</td>', html_doc)

    def test_static_disagreement_kpi_is_real_not_dash(self) -> None:
        """The Gym Health disagreement KPI must be a real value, not '—'."""
        rows = self._primary_dataset()
        public = [db._row_to_public(row) for row in rows]
        gym = db.aggregate_gym(public)
        self.assertAlmostEqual(gym["disagreement_rate"], 0.5)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dataset(root, rows)
            html_doc = render_for(root)
            self.assertIn(
                '<span class="kpi-label">Paired disagreement</span>'
                '<span class="kpi-value">50.0%</span>',
                html_doc,
            )
            self.assertNotIn(
                '<span class="kpi-label">Paired disagreement</span>'
                '<span class="kpi-value">—</span>',
                html_doc,
            )

    def test_cost_single_attempt_cells_suppressed(self) -> None:
        """n<=1 cost cells suppress std and min-max in static and JS output."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                # deliberate: a single attempt -> n=1 -> std/range suppressed.
                make_dataset_row("t1", "t1-b", "deliberate", True, usage={"total_tokens": 200}, tool_calls=4, messages=3),
                # direct: two attempts with distinct tokens -> n=2 -> std/range shown.
                make_dataset_row("t2", "t2-a", "direct", True, usage={"total_tokens": 100}, tool_calls=2, messages=1),
                make_dataset_row("t2", "t2-a2", "direct", True, usage={"total_tokens": 300}, tool_calls=6, messages=2),
            ]
            write_dataset(root, rows)
            html_doc = render_for(root)
            # n=1 row: mean only, then "—" for std and min-max on every metric.
            self.assertIn(
                "200.0</td><td>—</td><td>4.0</td><td>—</td><td>3.0</td><td>—</td>",
                html_doc,
            )
            # n=2 row: std and range are computed.
            self.assertIn("200.0 ± 100.0", html_doc)
            self.assertIn("100–300", html_doc)
            # No fabricated zero-spread / collapsed range for the n=1 cell.
            self.assertNotIn("200–200", html_doc)
            self.assertNotIn("200.0 ± 0.0", html_doc)
            # The JS renderer mirrors the same n<=1 guard.
            self.assertIn("hasRange = r.n > 1", html_doc)


# ---------------------------------------------------------------------------
# Optional input states (missing / malformed) and calibration gating
# ---------------------------------------------------------------------------
class OptionalInputStatesTest(unittest.TestCase):
    def _basic_rows(self) -> list[dict]:
        return [
            make_dataset_row("t1", "a", "direct", True),
            make_dataset_row("t1", "b", "deliberate", True),
        ]

    def test_missing_model_metrics_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dataset(root, self._basic_rows())
            data = build_data(root)
            self.assertEqual(data["meta"]["model_status"], "missing")
            self.assertEqual(data["model"]["status"], "missing")
            html_doc = db.render_dashboard(data)
            self.assertIn("Model evaluation is not available", html_doc)
            self.assertIn("no metrics file was provided", html_doc)

    def test_missing_metrics_file_state_with_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dataset(root, self._basic_rows())
            missing = root / "nope-metrics.json"
            data = build_data(root, missing)
            self.assertEqual(data["model"]["status"], "missing")
            self.assertIn("metrics file not found", data["model"]["warning"])
            self.assertIn("metrics file not found", db.render_dashboard(data))

    def test_malformed_metrics_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dataset(root, self._basic_rows())
            bad = _metric_file(root, {"not-a-split": [1, 2, 3]})
            data = build_data(root, bad)
            self.assertEqual(data["model"]["status"], "malformed")
            self.assertEqual(data["meta"]["model_status"], "malformed")
            html_doc = db.render_dashboard(data)
            self.assertIn("Model evaluation is not available", html_doc)
            self.assertIn("no recognized split bundles", html_doc)

    def test_malformed_metrics_json_syntax(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dataset(root, self._basic_rows())
            bad = root / "metrics.json"
            bad.write_text("{ this is not json", encoding="utf-8")
            data = build_data(root, bad)
            self.assertEqual(data["model"]["status"], "malformed")
            self.assertIn("unreadable", data["model"]["warning"])
            self.assertIn("Model evaluation is not available", db.render_dashboard(data))

    def test_missing_baselines_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dataset(root, self._basic_rows())
            data = build_data(root)
            self.assertEqual(data["baselines"]["status"], "missing")
            html_doc = db.render_dashboard(data)
            self.assertIn("Allocator baselines are not available", html_doc)
            self.assertIn("no baselines file was provided", html_doc)

    def test_malformed_baselines_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dataset(root, self._basic_rows())
            bad = _baseline_file(root, {"not": [1], "usable": "x"})
            data = build_data(root, baselines=bad)
            self.assertEqual(data["baselines"]["status"], "malformed")
            html_doc = db.render_dashboard(data)
            self.assertIn("Allocator baselines are not available", html_doc)

    def test_present_model_and_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dataset(root, self._basic_rows())
            metrics = _metric_file(
                root,
                {
                    "train": {
                        "n": 2,
                        "log_loss": 0.61,
                        "brier": 0.21,
                        "accuracy_05": 0.75,
                        "ece": 0.04,
                        "mean_posterior_std": 0.09,
                        "per_policy": {
                            "direct": {"n": 1, "log_loss": 0.6, "brier": 0.2, "accuracy_05": 0.8, "ece": 0.05, "mean_posterior_std": 0.1}
                        },
                    },
                    "test": {"n": 0, "log_loss": None, "brier": None, "accuracy_05": None, "ece": None, "mean_posterior_std": None, "per_policy": {}},
                },
            )
            baselines = _baseline_file(
                root,
                {
                    "always-direct": {"verified_success_rate": 0.5, "n": 4, "mean_tokens": 120.0, "mean_tool_calls": 2.0, "mean_messages": 1.0},
                    "always-deliberate": {"success_rate": 0.75, "attempts": 4, "mean_tokens": 260.0},
                },
            )
            data = build_data(root, metrics, baselines)
            self.assertEqual(data["meta"]["model_status"], "present")
            self.assertEqual(data["meta"]["baselines_status"], "present")
            self.assertEqual(data["model"]["splits"][0]["split"], "train")
            self.assertEqual(data["model"]["splits"][0]["n"], 2)
            self.assertEqual(data["model"]["per_policy"][0]["policy"], "direct")
            self.assertEqual(data["baselines"]["rows"][0]["strategy"], "always-deliberate")
            html_doc = db.render_dashboard(data)
            self.assertIn("✓ Model metrics: available", html_doc)
            self.assertIn("✓ Allocator baselines: available", html_doc)
            self.assertIn("Metrics by split", html_doc)
            self.assertIn("Allocator baseline strategies", html_doc)
            self.assertNotIn("Model evaluation is not available", html_doc)
            self.assertNotIn("Allocator baselines are not available", html_doc)

    def test_calibration_only_drawn_when_binned_data_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dataset(root, self._basic_rows())
            # Metrics with ECE but NO calibration data -> chart must not be drawn.
            metrics = _metric_file(
                root,
                {
                    "train": {
                        "n": 2, "log_loss": 0.6, "brier": 0.2,
                        "accuracy_05": 0.8, "ece": 0.05,
                        "mean_posterior_std": 0.1, "per_policy": {},
                    }
                },
            )
            html_doc = render_for(root, metrics)
            self.assertIn("calibration plot is not drawn", html_doc)
            self.assertIn("cannot be fabricated from aggregate metrics", html_doc)
            self.assertNotIn("Reliability diagram", html_doc)
            self.assertNotIn("<svg", html_doc)
            # Now add real binned data.
            metrics2 = _metric_file(
                root,
                {
                    "train": {
                        "n": 2, "log_loss": 0.6, "brier": 0.2,
                        "accuracy_05": 0.8, "ece": 0.05,
                        "mean_posterior_std": 0.1, "per_policy": {},
                    },
                    "calibration": {
                        "train": [
                            {"lower": 0.0, "upper": 0.5, "confidence": 0.2, "accuracy": 0.25, "count": 8},
                            {"lower": 0.5, "upper": 1.0, "confidence": 0.9, "accuracy": 0.85, "count": 12},
                        ]
                    },
                },
            )
            html_doc2 = render_for(root, metrics2)
            self.assertIn("Reliability diagram", html_doc2)
            self.assertIn("<svg", html_doc2)
            data2 = build_data(root, metrics2)
            self.assertTrue(data2["model"]["has_calibration"])
            self.assertEqual(len(data2["model"]["calibration"][0]["bins"]), 2)

    def test_percentage_style_baseline_success_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dataset(root, self._basic_rows())
            baselines = _baseline_file(root, {"direct": {"success_rate": 73.2, "n": 100}})
            data = build_data(root, baselines=baselines)
            self.assertAlmostEqual(data["baselines"]["rows"][0]["success_rate"], 0.732)

    def test_static_per_policy_table_includes_split(self) -> None:
        """The static per-policy metrics table must match the JS columns."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dataset(root, self._basic_rows())
            metrics = _metric_file(
                root,
                {
                    "train": {
                        "n": 2, "log_loss": 0.6, "brier": 0.2,
                        "accuracy_05": 0.8, "ece": 0.05,
                        "mean_posterior_std": 0.1,
                        "per_policy": {
                            "direct": {
                                "n": 1, "log_loss": 0.6, "brier": 0.2,
                                "accuracy_05": 0.8, "ece": 0.05,
                                "mean_posterior_std": 0.1,
                            },
                            "deliberate": {
                                "n": 1, "log_loss": 0.5, "brier": 0.18,
                                "accuracy_05": 0.9, "ece": 0.04,
                                "mean_posterior_std": 0.08,
                            },
                        },
                    },
                },
            )
            html_doc = render_for(root, metrics)
            # The per-policy table leads with Policy then Split (matching JS).
            self.assertIn('<th scope="col">Policy</th><th scope="col">Split</th>', html_doc)
            # The split-level table leads with Split.
            self.assertIn('<th scope="col">Split</th><th scope="col">n</th>', html_doc)
            # The per-policy row is Policy then Split (matching the JS order).
            self.assertIn(">direct</td><td>train</td>", html_doc)
            self.assertIn(">deliberate</td><td>train</td>", html_doc)
            # JS renders the same column set for the per-policy table.
            self.assertIn(
                '["Policy","Split","n","Log loss","Brier","Accuracy","ECE","Posterior std"]',
                html_doc,
            )


# ---------------------------------------------------------------------------
# Empty dataset / one policy / filter robustness
# ---------------------------------------------------------------------------
class EdgeStateTest(unittest.TestCase):
    def test_empty_dataset_renders_empty_states(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dataset(root, [])
            data = build_data(root)
            self.assertEqual(data["meta"]["rows"], 0)
            self.assertEqual(data["gym"]["total_rows"], 0)
            self.assertEqual(data["disagreement"]["paired_tasks"], 0)
            html_doc = db.render_dashboard(data)
            self.assertIn("No verified attempts to aggregate.", html_doc)
            self.assertIn("No attempts match the current filters.", html_doc)
            self.assertIn("No paired tasks to compare", html_doc)
            # filters show "none present"
            self.assertIn("none present", html_doc)
            self.assertIn("Showing 0 of 0 attempts", html_doc)

    def test_single_policy_pairless_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                make_dataset_row("t1", "a", "direct", True),
                make_dataset_row("t2", "b", "direct", False),
            ]
            write_dataset(root, rows)
            data = build_data(root)
            self.assertEqual(data["disagreement"]["policy_a"], "direct")
            self.assertIsNone(data["disagreement"]["policy_b"])
            self.assertEqual(data["disagreement"]["paired_tasks"], 0)
            html_doc = db.render_dashboard(data)
            self.assertIn("No paired tasks to compare", html_doc)

    def test_rows_missing_optional_fields_are_sanitized(self) -> None:
        row = make_dataset_row("t1", "a1", "direct", True)
        row.pop("usage", None)
        row.pop("model_input", None)
        row.pop("difficulty", None)
        row.pop("verifier_id", None)
        row["public_metadata"] = {"name": "secret-name", "rows": 3}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dataset(root, [row])
            data = build_data(root)
            public = data["rows"][0]
            self.assertEqual(public["difficulty"], None)
            self.assertIsNone(public["verifier_id"])
            self.assertEqual(db.total_tokens(public), 0)
            self.assertEqual(public["metadata"], {"rows": 3})
            html_doc = db.render_dashboard(data)
            self.assertNotIn("secret-name", html_doc)

    def test_filtered_zero_guard_present_in_js(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dataset(root, [make_dataset_row("t1", "a", "direct", True)])
            html_doc = render_for(root)
            # The JS must guard empty filtered sets in every section.
            for message in (
                "No verified attempts to aggregate.",
                "No attempts match the current filters.",
                "No paired tasks to compare",
                "No cost data to aggregate.",
                "No verified failures recorded.",
            ):
                self.assertIn(message, html_doc)

    def test_skipped_unusable_rows_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = make_dataset_row("t1", "a", "direct", True)
            bad_no_label = dict(good, attempt_id="b", verified_success="yes")
            bad_no_identity = dict(good, attempt_id="c", task_id=None)
            write_dataset(root, [good, bad_no_label, bad_no_identity])
            data = build_data(root)
            self.assertEqual(data["meta"]["rows"], 1)
            self.assertTrue(any("skipped" in w for w in data["meta"]["warnings"]))

    def test_malformed_dataset_line_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dataset(root, [make_dataset_row("t1", "a", "direct", True)])
            path = root / "dataset.jsonl"
            path.write_text(path.read_text(encoding="utf-8") + "{bad json\n", encoding="utf-8")
            with self.assertRaises(ValueError) as context:
                build_data(root)
            self.assertIn("dataset.jsonl", str(context.exception))

    def test_missing_dataset_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                build_data(root)

    def test_small_sample_banner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                make_dataset_row(f"t{i}", f"t{i}-a", "direct", True) for i in range(5)
            ]
            rows += [
                make_dataset_row(f"t{i}", f"t{i}-b", "deliberate", True) for i in range(4)
            ]
            write_dataset(root, rows)
            html_doc = render_for(root)
            self.assertIn("Small sample — only 9 verified attempts", html_doc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for index in range(15):
                rows.append(make_dataset_row(f"t{index}", f"t{index}-a", "direct", True))
                rows.append(make_dataset_row(f"t{index}", f"t{index}-b", "deliberate", True))
            write_dataset(root, rows)
            html_doc = render_for(root)
            self.assertNotIn("Small sample", html_doc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dataset(root, [])
            html_doc = render_for(root)
            self.assertNotIn("Small sample", html_doc)


# ---------------------------------------------------------------------------
# Attempt browser table cap
# ---------------------------------------------------------------------------
class BrowserCapTest(unittest.TestCase):
    def test_browser_cap_is_500_and_static_rows_are_capped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for index in range(300):
                tid = f"task-{index}"
                rows.append(make_dataset_row(tid, f"{tid}-a", "direct", index % 2 == 0))
                rows.append(make_dataset_row(tid, f"{tid}-b", "deliberate", index % 3 == 0))
            write_dataset(root, rows)
            data = build_data(root)
            self.assertEqual(len(data["rows"]), 600)
            self.assertEqual(data["config"]["browser_cap"], db.BROWSER_ROW_CAP)
            self.assertEqual(db.BROWSER_ROW_CAP, 500)
            html_doc = db.render_dashboard(data)
            self.assertIn("Showing 500 of 600 attempts (capped at 500 rows).", html_doc)
            # JS caps the rendered rows dynamically as well.
            self.assertIn("rows.slice(0,CAP)", html_doc)
            payload = extract_embedded_json(html_doc)
            self.assertEqual(payload["config"]["browser_cap"], 500)
            self.assertEqual(len(payload["rows"]), 600)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
class DeterminismTest(unittest.TestCase):
    def test_render_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                make_dataset_row(f"t{i}", f"t{i}-a", "direct", i % 2 == 0, family="sqlite" if i % 2 else "artifact")
                for i in range(12)
            ]
            write_dataset(root, rows)
            first = render_for(root)
            second = render_for(root)
            self.assertEqual(first, second)
            self.assertEqual(first.encode("utf-8"), second.encode("utf-8"))

    def test_build_data_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dataset(root, [make_dataset_row("t1", "a", "direct", True)])
            self.assertEqual(build_data(root), build_data(root))


# ---------------------------------------------------------------------------
# Accessibility / landmarks / filter controls
# ---------------------------------------------------------------------------
class AccessibilityTest(unittest.TestCase):
    def _full_html(self) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                make_dataset_row("t1", "a", "direct", True),
                make_dataset_row("t1", "b", "deliberate", False, failure_code="timeout"),
                make_dataset_row("t2", "c", "direct", False, failure_code="missing_output"),
            ]
            write_dataset(root, rows)
            metrics = _metric_file(
                root,
                {
                    "train": {"n": 2, "log_loss": 0.6, "brier": 0.2, "accuracy_05": 0.8, "ece": 0.05, "mean_posterior_std": 0.1, "per_policy": {}},
                    "calibration": {"train": [{"confidence": 0.6, "accuracy": 0.5, "count": 3}]},
                },
            )
            baselines = _baseline_file(root, {"always-direct": {"success_rate": 0.5, "n": 2}})
            return render_for(root, metrics, baselines)

    def test_document_structure(self) -> None:
        html_doc = self._full_html()
        self.assertTrue(html_doc.startswith("<!DOCTYPE html>"))
        self.assertIn("<html lang=\"en\">", html_doc)
        self.assertIn("<meta charset=\"utf-8\">", html_doc)
        self.assertIn("<header", html_doc)
        self.assertIn("<main", html_doc)
        self.assertIn("<footer", html_doc)

    def test_sections_have_aria_labelledby_landmarks(self) -> None:
        html_doc = self._full_html()
        sections = ["gym", "disagreement", "failures", "costs", "model", "baselines", "browser"]
        for section in sections:
            with self.subTest(section=section):
                self.assertIn(f'<section aria-labelledby="heading-{section}" id="{section}">', html_doc)
                self.assertIn(f'id="heading-{section}"', html_doc)
        self.assertIn('<nav aria-label="Sections">', html_doc)
        self.assertEqual(html_doc.count("<section"), 7)

    def test_filter_controls(self) -> None:
        html_doc = self._full_html()
        self.assertIn('<form id="filters" aria-label="Dashboard filters"', html_doc)
        for label in ("Family", "Difficulty", "Policy"):
            self.assertIn(f"<fieldset><legend>{label}</legend>", html_doc)
        self.assertIn('<button type="button" id="filter-reset" class="btn">Reset</button>', html_doc)
        # native checkboxes with labels
        self.assertGreaterEqual(html_doc.count('type="checkbox"'), 3)
        self.assertIn('data-filter="family"', html_doc)
        self.assertIn('data-filter="difficulty"', html_doc)
        self.assertIn('data-filter="policy"', html_doc)
        self.assertIn("<label", html_doc)
        # live regions
        self.assertGreaterEqual(html_doc.count('aria-live="polite"'), 2)

    def test_tables_use_scope_and_captions(self) -> None:
        html_doc = self._full_html()
        self.assertGreaterEqual(html_doc.count('scope="col"'), 20)
        self.assertGreaterEqual(html_doc.count("<caption"), 4)

    def test_color_and_non_color_cues(self) -> None:
        html_doc = self._full_html()
        self.assertIn("#1565C0", html_doc)  # direct blue
        self.assertIn("#E65100", html_doc)  # deliberate orange
        self.assertIn("✓", html_doc)
        self.assertIn("✗", html_doc)
        # success/failure are never conveyed by color alone (icons/text present)
        self.assertIn("success", html_doc)
        self.assertIn("fail", html_doc)

    def test_focus_and_motion_and_layout_css(self) -> None:
        html_doc = self._full_html()
        self.assertIn(":focus-visible{outline:3px solid var(--focus)", html_doc)
        self.assertIn("@media (prefers-reduced-motion: reduce)", html_doc)
        self.assertIn("max-width:1280px", html_doc)
        self.assertIn("-apple-system,BlinkMacSystemFont", html_doc)
        self.assertIn("font-family:var(--font)", html_doc)

    def test_keyboard_accessible_detail_toggles(self) -> None:
        html_doc = self._full_html()
        self.assertIn("data-detail-toggle", html_doc)
        self.assertIn('aria-expanded="false"', html_doc)
        self.assertIn("aria-controls=\"detail-0\"", html_doc)
        # no modal dialogs
        self.assertNotIn('<dialog', html_doc)

    def test_state_notes_use_role_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dataset(root, [])
            html_doc = render_for(root)
            self.assertIn('role="status"', html_doc)

    def test_skip_link_and_min_font_sizes(self) -> None:
        html_doc = self._full_html()
        # Keyboard-visible skip link targets the main landmark.
        self.assertIn('class="skip-link" href="#main"', html_doc)
        self.assertIn('<main id="main">', html_doc)
        self.assertIn(".skip-link{", html_doc)
        self.assertIn(".skip-link:focus", html_doc)
        # Body font is >= 16px and headings sit visually above the body text.
        self.assertIn("font-size:16px", html_doc)
        self.assertIn("h3{font-size:18px", html_doc)
        self.assertIn("h4{font-size:16px", html_doc)


# ---------------------------------------------------------------------------
# CLI and write atomicity
# ---------------------------------------------------------------------------
class CliAndAtomicityTest(unittest.TestCase):
    def test_write_dashboard_is_atomic_and_summarizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dataset(root, [make_dataset_row("t1", "a", "direct", True)])
            output = root / "nested" / "dir" / "dashboard.html"
            summary = db.write_dashboard(root / "dataset.jsonl", output)
            self.assertTrue(output.is_file())
            self.assertEqual(summary["rows"], 1)
            self.assertIn("output_path", summary)
            self.assertGreater(summary["bytes"], 1000)
            # no leftover temp files
            leftovers = list(output.parent.glob(".*dashboard.html.*"))
            self.assertEqual(leftovers, [])
            self.assertTrue(output.read_text(encoding="utf-8").startswith("<!DOCTYPE html>"))

    def test_write_failure_leaves_no_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dataset(root, [make_dataset_row("t1", "a", "direct", True)])
            output = root / "out.html"
            with self.assertRaises(FileNotFoundError):
                db.write_dashboard(root / "missing.jsonl", output)
            self.assertFalse(output.exists())

    def test_cli_writes_dashboard_and_prints_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dataset(root, [make_dataset_row("t1", "a", "direct", True)])
            metrics = _metric_file(root, {"train": {"n": 1, "log_loss": 0.5, "brier": 0.2, "accuracy_05": 0.8, "ece": 0.1, "mean_posterior_std": 0.05, "per_policy": {}}})
            baselines = _baseline_file(root, {"direct": {"success_rate": 1.0, "n": 1}})
            output = root / "cli-dashboard.html"
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = db.main([str(root / "dataset.jsonl"), str(output), "--metrics", str(metrics), "--baselines", str(baselines)])
            self.assertEqual(code, 0)
            summary = json.loads(buffer.getvalue())
            self.assertEqual(summary["rows"], 1)
            self.assertEqual(summary["model_status"], "present")
            self.assertEqual(summary["baselines_status"], "present")
            self.assertTrue(output.is_file())

    def test_cli_error_returns_nonzero(self) -> None:
        import sys
        from contextlib import redirect_stderr

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stderr = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
                code = db.main([str(root / "missing.jsonl"), str(root / "out.html")])
            self.assertEqual(code, 1)
            self.assertIn("error:", stderr.getvalue())
            self.assertFalse((root / "out.html").exists())

    def test_module_cli_parser(self) -> None:
        parser = db.build_parser()
        args = parser.parse_args(["ds.jsonl", "out.html", "--metrics", "m.json", "--baselines", "b.json"])
        self.assertEqual(args.dataset, "ds.jsonl")
        self.assertEqual(args.metrics, "m.json")
        self.assertEqual(args.baselines, "b.json")

    def test_module_cli_runs_as_subprocess(self) -> None:
        """Regression: `python -m pyreplab_harness.dashboard` must work.

        Executing the module as ``__main__`` runs its body top to bottom, so
        the entry-point guard must only fire after every definition (notably
        the HTML ``_TEMPLATE``) is bound.  Calling ``main()`` after import
        would never catch a misplaced guard, so this test runs the module as a
        real subprocess.
        """
        import os
        import subprocess
        import sys

        src_dir = Path(db.__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dataset(root, [make_dataset_row("t1", "a", "direct", True)])
            metrics = _metric_file(
                root,
                {
                    "train": {
                        "n": 1, "log_loss": 0.5, "brier": 0.2,
                        "accuracy_05": 0.8, "ece": 0.1,
                        "mean_posterior_std": 0.05, "per_policy": {},
                    }
                },
            )
            baselines = _baseline_file(root, {"direct": {"success_rate": 1.0, "n": 1}})
            output = root / "subprocess-dashboard.html"
            env = dict(os.environ)
            env["PYTHONPATH"] = str(src_dir)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pyreplab_harness.dashboard",
                    str(root / "dataset.jsonl"),
                    str(output),
                    "--metrics",
                    str(metrics),
                    "--baselines",
                    str(baselines),
                ],
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            summary = json.loads(result.stdout)
            self.assertEqual(summary["rows"], 1)
            self.assertEqual(summary["model_status"], "present")
            self.assertEqual(summary["baselines_status"], "present")
            self.assertTrue(output.is_file())
            self.assertTrue(output.read_text(encoding="utf-8").startswith("<!DOCTYPE html>"))


class RenderGuardTest(unittest.TestCase):
    def test_render_tolerates_minimal_data_dict(self) -> None:
        # Guards in render_dashboard must handle missing optional keys.
        html_doc = db.render_dashboard(
            {
                "meta": {"title": "Minimal", "rows": 0, "model_status": "missing", "baselines_status": "missing", "warnings": []},
                "filters": {"family": [], "difficulty": [], "policy": []},
                "rows": [],
            }
        )
        self.assertIn("Minimal", html_doc)
        self.assertIn("No verified attempts to aggregate.", html_doc)


if __name__ == "__main__":
    unittest.main()
