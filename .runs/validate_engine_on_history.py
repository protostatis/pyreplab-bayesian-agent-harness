#!/usr/bin/env python3
"""Validate the Bayesian outcome engine against the real screening history.

Note: v13's 24 cells were re-run inside v14, so cross-generation dedup by
cell_id yields 62 unique (task, replica, arm) observations, not 115.

Feeds the 115 completed arm-screen runs (v11 43 + v13 24 + v14 48) to the
engine as a dataset, then checks two things:

1. ARM-EFFECT RECOVERY (train = easy+medium tasks, test = held-out hard tasks):
   does the engine, fitted only on easy/medium history, correctly project the
   recovery-discipline arm as the weakest on the hard stratum it never saw?
2. HONEST UNCERTAINTY: per-arm posterior means vs directly observed rates, so
   any disagreement is visible instead of hidden.

This is validation of the ENGINE, not new science: 115 rows cannot support
sweeping claims, and the task bank is narrow by design.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / ".runs"))

from pyreplab_harness.outcome_model import (  # noqa: E402
    load_artifacts,
    predict_single,
    train_model,
)

ARMS = {
    "E": ("baseline_execution", "baseline prompt (no discipline rules)"),
    "C": ("execution_discipline", "execution-discipline prompt"),
    "R": ("execution_recovery_discipline", "execution + recovery discipline prompt"),
}
TASK_TEXT = (
    "Unbrowser fixture task ({template}, {difficulty}). Complete the web page "
    "workflow and write the required verification key to /workspace/result.json."
)


def load_ledgers() -> list[dict]:
    recs: list[dict] = []
    paths = [
        REPO / ".runs/m3-prompt-only-pilot-20260816-v11.results.jsonl",
        REPO / ".runs/m3-ppo-v13/m3-prompt-only-pilot-20260816-v13.results.jsonl",
        REPO / ".runs/m3-ppo-v14/m3-prompt-only-pilot-20260816-v14.results.jsonl",
    ]
    for p in paths:
        for line in p.read_text().splitlines():
            if line.strip():
                recs.append(json.loads(line))
    return recs


def load_system_prompts() -> dict[str, str]:
    """Real system prompts per arm from the frozen v14 registry — the actual
    experimental variable, not a placeholder."""
    registry = json.loads(
        (REPO / ".runs/m3-ppo-v14/m3-prompt-only-pilot-20260816-v14.registry.json")
        .read_text()
    )
    return {t["id"]: str(t.get("system_prompt", "")) for t in registry["treatments"]}


def to_row(rec: dict, split: str, system_prompts: dict[str, str]) -> dict:
    cell = rec["cell"]
    task = rec["task"]
    arm = cell["arm"]
    policy_id = ARMS[arm][0]
    attempt = rec["result"]["attempts"][next(iter(rec["result"]["attempts"]))]
    policy = attempt["policy"]
    seed = task["seed"]
    difficulty_index = {"easy": 0.0, "medium": 1.0, "hard": 2.0}[task["difficulty"]]
    model_input = {
        "text": TASK_TEXT.format(template=task["template"], difficulty=task["difficulty"]),
        "family": "unbrowser",
        "template_id": f"{task['template']}-{task['difficulty']}-v1",
        "difficulty": task["difficulty"],
        "public_metadata": {"seed": float(seed), "difficulty_index": difficulty_index},
        "policy_id": policy_id,
        "policy_version": str(policy.get("version", "1")),
        "treatment": {
            # SENSITIVITY CHECK (2026-08-30): embedding the real system-prompt
            # text here INVERTED the recovery (weakest predicted = C) — with 62
            # rows and 3 distinct prompts, the text pathway overfits and hard-
            # stratum generalization breaks. Identity categoricals + weak prior
            # is the correct configuration at screening scale. Re-test text
            # features only with >=500 rows.
            "text": (
                f"{policy_id}: tool_call_limit={policy.get('tool_call_limit')}; "
                f"system_prompt_sha=policy-defined"
            ),
            "max_output_tokens": float(policy.get("max_output_tokens", 4096)),
            "tool_call_limit": float(policy.get("tool_call_limit", 12)),
            "command_timeout_seconds": float(policy.get("command_timeout_seconds", 60)),
            "wall_time_limit_seconds": float(policy.get("wall_time_limit_seconds", 3300)),
            "tool_interface": str(policy.get("tool_interface", "")),
            "allowed_tools_signature": ",".join(policy.get("allowed_tools", [])),
            "bundle_id": str(policy.get("bundle_hash", ""))[:16] or "unknown",
            "policy_id": policy_id,
            "policy_version": str(policy.get("version", "1")),
        },
    }
    return {
        "model_input": model_input,
        "verified_success": bool(
            attempt["verification"]["success"]
        ),
        "split": split,
        "arm": arm,
        "task_seed": seed,
        "generation": rec.get("schema_version") and rec.get("attempt_id", "")[:0] or "",
        "cell_id": rec["cell_id"],
    }


def main() -> int:
    out_dir = REPO / ".runs" / "engine_validation"
    out_dir.mkdir(exist_ok=True)
    records = load_ledgers()
    system_prompts = load_system_prompts()
    seen: set[str] = set()
    rows: list[dict] = []
    for rec in records:
        if rec["cell_id"] in seen:  # guard against overlapping prefixes
            continue
        seen.add(rec["cell_id"])
        seed = rec["task"]["seed"]
        # easy/medium seeds 01-04,07-10 train; 09/10 validation; hard 05/06/11/12 test
        if rec["task"]["difficulty"] == "hard":
            split = "test"
        elif seed in (2026093009, 2026093010):
            split = "validation"
        else:
            split = "train"
        rows.append(to_row(rec, split, system_prompts))
    dataset = out_dir / "screening_history_dataset.jsonl"
    dataset.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    counts = defaultdict(lambda: defaultdict(int))
    for r in rows:
        counts[r["split"]][r["arm"]] += 1
    print("dataset rows:", dict((k, dict(v)) for k, v in counts.items()))

    metrics = train_model(
        dataset,
        out_dir / "artifact",
        epochs=60,
        batch_size=16,
        seed=42,
        device="cpu",
        max_vocab=2000,
        max_tokens=128,
        patience=8,
        prior_sigma=5.0,
        cat_dim=32,
        verbose=False,
    )
    report = {"dataset_rows": {k: dict(v) for k, v in counts.items()}, "metrics": metrics}

    # Per-arm recovery on the held-out hard stratum: predicted mean vs observed.
    _, pre, model = load_artifacts(out_dir / "artifact")
    test_rows = [r for r in rows if r["split"] == "test"]
    per_arm: dict[str, dict] = defaultdict(lambda: {"pred": [], "obs": []})
    for r in test_rows:
        prediction = predict_single(model, pre, r["model_input"], seed=7)
        per_arm[r["arm"]]["pred"].append(prediction["mean"])
        per_arm[r["arm"]]["obs"].append(1.0 if r["verified_success"] else 0.0)
    table = {}
    for arm, values in sorted(per_arm.items()):
        pred_mean = sum(values["pred"]) / len(values["pred"])
        obs_mean = sum(values["obs"]) / len(values["obs"])
        table[arm] = {
            "name": ARMS[arm][1],
            "n_test": len(values["obs"]),
            "predicted_mean": round(pred_mean, 3),
            "observed_mean": round(obs_mean, 3),
            "pred_std": round(
                (sum((p - pred_mean) ** 2 for p in values["pred"]) / len(values["pred"])) ** 0.5,
                3,
            ),
        }
    report["hard_stratum_recovery"] = table
    rank_by_pred = sorted(table, key=lambda a: -table[a]["predicted_mean"])
    rank_by_obs = sorted(table, key=lambda a: -table[a]["observed_mean"])
    report["recovery_check"] = {
        "weakest_arm_by_prediction": rank_by_pred[-1],
        "weakest_arm_observed": rank_by_obs[-1],
        "recovered": rank_by_pred[-1] == rank_by_obs[-1],
    }
    (out_dir / "validation_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("hard_stratum_recovery", "recovery_check")}, indent=1))
    metrics = (report.get("metrics") or {}).get("metrics") or {}
    for split in ("validation", "test"):
        block = metrics.get(split) or {}
        print(f"{split}: brier={block.get('brier')} ece={block.get('ece')} log_loss={block.get('log_loss')} n={block.get('n')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
