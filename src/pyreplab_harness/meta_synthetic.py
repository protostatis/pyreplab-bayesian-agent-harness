"""Synthetic validation harness for the M3 meta-policy-learner.

Generates synthetic policy outcomes with a policy-specific latent adaptation
signal (SHA-256 keyed by bundle identity and a fixed latent seed). The latent
residual is shared across calibration and target task pools but is never
exposed in model_input, forcing the CNP to learn it from calibration context.

This validates the full pipeline (grammar, CNP model, frontier evaluator,
calibration) before real Gemma rollouts. All data is synthetic; no claims
about real effectiveness.
"""

from __future__ import annotations

import copy
import hashlib
import math
import random
from typing import Any

from . import meta_grammar
from .meta_grammar import enumerate_unbrowser_grammar, grammar_factor_vector

try:
    from .meta_cnp import TORCH_AVAILABLE
except ImportError:
    TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Base seed for independent latent worlds. Within a world, the same latent
# residual is shared by calibration and target outcomes.
_LATENT_SEED: int = 8239471

# Task pool sizes.
_N_META_TASKS: int = 64       # per latent world
_N_META_WORLDS: int = 16      # prevents descriptor-to-latent memorization
_N_DEV_WORLDS: int = 4        # stabilizes synthetic checkpoint selection
_N_DEV_TASKS: int = 80        # 16 calibration + 64 target
_N_FINAL_TASKS: int = 80      # 16 calibration + 64 target
_N_CAL_TASKS: int = 16        # calibration tasks per dev/final split
_LATENT_LOGIT_SCALE: float = 1.5

# Training constants.
_N_EPOCHS: int = 40
_N_BATCH_SIZE: int = 32
_N_BATCHES_PER_EPOCH: int = 32
_EARLY_STOP_PATIENCE: int = 8

# ---------------------------------------------------------------------------
# SHA-256 stable policy latent
# ---------------------------------------------------------------------------


def _compute_policy_latent(
    bundle_id: str,
    latent_seed: int = _LATENT_SEED,
) -> tuple[float, float]:
    """Compute stable policy-specific latent residuals via SHA-256.

    Returns (logit_residual, cost_log_residual), each in [-1.0, 1.0].
    Deterministic for a given (bundle_id, latent_seed) pair and
    independent of Python hash or random seeds.
    """
    payload = f"latent:{bundle_id}:{latent_seed}".encode("utf-8")
    hash_bytes = hashlib.sha256(payload).digest()
    hash_int = int.from_bytes(hash_bytes[:8], "big")
    logit_res = (hash_int % 20000) / 10000.0 - 1.0
    cost_res = ((hash_int >> 32) % 20000) / 10000.0 - 1.0
    return logit_res, cost_res


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _binary_log_loss(pred_probs: list[float], outcomes: list[float]) -> float:
    """Bernoulli log loss (binary cross-entropy)."""
    n = len(pred_probs)
    if n == 0:
        return float("nan")
    eps = 1e-12
    total = 0.0
    for p, y in zip(pred_probs, outcomes):
        p_clipped = max(eps, min(1.0 - eps, p))
        total += -(y * math.log(p_clipped) + (1.0 - y) * math.log(1.0 - p_clipped))
    return total / n


def _brier_score(pred_probs: list[float], outcomes: list[float]) -> float:
    """Brier score (mean squared error)."""
    n = len(pred_probs)
    if n == 0:
        return float("nan")
    return sum((p - y) ** 2 for p, y in zip(pred_probs, outcomes)) / n


def _cross_entropy_vs_true(
    pred_probs: list[float], true_probs: list[float],
) -> float:
    """Cross entropy of predicted probs relative to true DGP probabilities."""
    n = len(pred_probs)
    if n == 0:
        return float("nan")
    eps = 1e-12
    total = 0.0
    for p, t in zip(pred_probs, true_probs):
        p_clipped = max(eps, min(1.0 - eps, p))
        t_clipped = max(eps, min(1.0 - eps, t))
        total += -(t_clipped * math.log(p_clipped) + (1.0 - t_clipped) * math.log(1.0 - p_clipped))
    return total / n


def _brier_vs_true(pred_probs: list[float], true_probs: list[float]) -> float:
    """Brier score of predicted probs relative to true DGP probabilities."""
    n = len(pred_probs)
    if n == 0:
        return float("nan")
    return sum((p - t) ** 2 for p, t in zip(pred_probs, true_probs)) / n


def _normalize_context_cost(cost: float, stats: dict[str, Any]) -> float:
    """Apply the frozen meta-train log1p cost standardization."""
    if not math.isfinite(cost) or cost < 0.0:
        raise ValueError(f"context cost must be finite and nonnegative, got {cost!r}")
    mean = float(stats["cost_mean"])
    std = float(stats["cost_std"])
    if std <= 0.0 or not math.isfinite(std):
        raise ValueError(f"cost_std must be finite and positive, got {std!r}")
    return (math.log1p(cost) - mean) / std


def _mean_evaluation_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    """Average scalar evaluation metrics across independent latent worlds."""
    if not results:
        raise ValueError("results must not be empty")
    metric_names = (
        "log_loss_binary",
        "brier",
        "mean_predicted_prob",
        "ranking_accuracy",
        "true_p_cross_entropy",
        "true_p_brier",
    )
    return {
        name: sum(float(result[name]) for result in results) / len(results)
        for name in metric_names
    }


def _global_pairwise_ranking_accuracy(
    pred_probs_by_task: list[list[float]],
    true_probs_by_task: list[list[float]],
) -> float:
    """Pairwise accuracy after averaging each policy over target tasks."""
    if not pred_probs_by_task or not true_probs_by_task:
        return float("nan")
    n_policies = len(pred_probs_by_task[0])
    if n_policies < 2:
        return float("nan")
    if any(len(row) != n_policies for row in pred_probs_by_task):
        raise ValueError("prediction panel has inconsistent policy counts")
    if any(len(row) != n_policies for row in true_probs_by_task):
        raise ValueError("true-probability panel has inconsistent policy counts")

    pred_means = [
        sum(task_probs[j] for task_probs in pred_probs_by_task)
        / len(pred_probs_by_task)
        for j in range(n_policies)
    ]
    true_means = [
        sum(task_probs[j] for task_probs in true_probs_by_task)
        / len(true_probs_by_task)
        for j in range(n_policies)
    ]

    correct = 0
    total = 0
    for i in range(n_policies):
        for j in range(i + 1, n_policies):
            pred_diff = pred_means[i] - pred_means[j]
            true_diff = true_means[i] - true_means[j]
            total += 1
            if (
                (pred_diff > 0 and true_diff > 0)
                or (pred_diff < 0 and true_diff < 0)
                or (pred_diff == 0 and true_diff == 0)
            ):
                correct += 1
    return correct / total if total > 0 else float("nan")


def _split_calibration_target(
    rows: list[dict[str, Any]],
    n_tasks: int,
    n_policies: int,
    n_cal_tasks: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split rows into calibration (first n_cal_tasks) and target pools.

    Calibration rows: tasks [0, n_cal_tasks) x all policies.
    Target rows: tasks [n_cal_tasks, n_tasks) x all policies.

    Returns (calibration_rows, target_rows).  Each row is tagged with
    ``task_pool_role`` so disjointness is testable.
    """
    cal_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    for task_idx in range(n_tasks):
        role = "calibration" if task_idx < n_cal_tasks else "target"
        for policy_idx in range(n_policies):
            row_idx = task_idx * n_policies + policy_idx
            row = dict(rows[row_idx])
            row["task_pool_role"] = role
            row["task_pool_index"] = (
                task_idx if role == "calibration" else task_idx - n_cal_tasks
            )
            if role == "calibration":
                cal_rows.append(row)
            else:
                target_rows.append(row)
    return cal_rows, target_rows


def _build_derangement(n: int, seed: int = 424242) -> dict[int, int]:
    """Build a deterministic derangement (permutation with no fixed points).

    Returns a dict mapping i -> j where j != i for all i in [0, n).
    """
    if n < 2:
        raise ValueError("a derangement requires at least two policies")
    rng = random.Random(seed)
    cycle = list(range(n))
    rng.shuffle(cycle)
    return {
        cycle[i]: cycle[(i + 1) % n]
        for i in range(n)
    }


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


def generate_synthetic_outcomes(
    grammar: list[Any],
    n_tasks: int,
    n_policies: int,
    seed: int,
    *,
    latent_seed: int | None = None,
    latent_logit_scale: float = 1.0,
    task_id_prefix: str = "synthetic-task",
) -> dict[str, Any]:
    """Generate synthetic policy outcomes with factor-dependent probabilities
    and optional policy-specific latent residuals.

    DGP (data-generating process):
        logit = base_logit + task_interaction + difficulty_penalty
              + latent_residual(bundle_id, latent_seed)

    The latent residual is a stable, SHA-256-derived per-policy signal that
    is shared across independently generated calibration and target pools
    (same ``latent_seed``) but NEVER exposed in ``model_input``.  The CNP
    must learn it from calibration context.

    Args:
        grammar: list of TreatmentSpec policy objects.
        n_tasks: number of synthetic task scenarios.
        n_policies: number of policies to use (<= len(grammar)).
        seed: main random seed for Bernoulli outcomes and task modifiers.
        latent_seed: if not None, add per-policy latent residuals via
            :func:`_compute_policy_latent`.
        latent_logit_scale: multiplier for the hidden success-logit residual.
        task_id_prefix: namespace used to keep independently generated task
            pools globally disjoint.

    Returns:
        dict with "rows", "true_params", "task_modifiers", "grammar",
        "n_tasks", "n_policies".
    """
    if not math.isfinite(latent_logit_scale) or latent_logit_scale <= 0.0:
        raise ValueError("latent_logit_scale must be finite and positive")
    rng = random.Random(seed)
    n_policies = min(n_policies, len(grammar))
    policies = list(grammar)[:n_policies]

    # Factor baseline effects (success logits).
    planning_effects = {"direct": 0.0, "brief_plan": 0.2, "decompose": -0.1}
    observation_effects = {"text_first": 0.1, "structure_first": 0.0, "targeted_query_first": -0.05}
    verification_effects = {"submit_directly": -0.1, "final_reobserve": 0.15}
    recovery_effects = {"fail_fast": -0.05, "diagnose_retry_once": 0.1}
    tool_cap_effects = {"lean": -0.2, "expanded": 0.3}

    # Cost factor effects (log-scale).
    planning_cost = {"direct": -0.2, "brief_plan": 0.0, "decompose": 0.3}
    observation_cost = {"text_first": 0.2, "structure_first": 0.0, "targeted_query_first": -0.1}
    tool_cap_cost = {"lean": -0.5, "expanded": 0.0}

    # Pre-compute per-policy latent residuals (shared across pools).
    policy_latents: dict[str, tuple[float, float]] = {}
    if latent_seed is not None:
        for idx in range(n_policies):
            bundle_id = policies[idx].bundle_id
            if bundle_id not in policy_latents:
                policy_latents[bundle_id] = _compute_policy_latent(bundle_id, latent_seed)

    # Compute per-policy true parameters.
    true_params: list[dict[str, Any]] = []
    for idx, treatment in enumerate(policies):
        meta = treatment.generator_metadata
        pl = str(meta.get("planning", "direct"))
        ob = str(meta.get("observation", "text_first"))
        ve = str(meta.get("verification", "submit_directly"))
        re = str(meta.get("recovery", "fail_fast"))
        tc = str(meta.get("tool_cap", "lean"))

        base_logit = (
            0.0
            + planning_effects.get(pl, 0.0)
            + observation_effects.get(ob, 0.0)
            + verification_effects.get(ve, 0.0)
            + recovery_effects.get(re, 0.0)
            + tool_cap_effects.get(tc, 0.0)
        )

        base_cost_log = (
            3.0
            + planning_cost.get(pl, 0.0)
            + observation_cost.get(ob, 0.0)
            + tool_cap_cost.get(tc, 0.0)
        )

        lat_logit, lat_cost = (0.0, 0.0)
        if latent_seed is not None:
            lat_logit, lat_cost = policy_latents.get(treatment.bundle_id, (0.0, 0.0))
            lat_logit *= latent_logit_scale

        true_params.append({
            "policy_idx": idx,
            "treatment": treatment,
            "base_logit": base_logit,
            "latent_logit": lat_logit,
            "base_cost_log_mean": base_cost_log + lat_cost * 0.3,
            "cost_log_sigma": 0.5,
            "grammar_factors": {
                "planning": pl, "observation": ob,
                "verification": ve, "recovery": re, "tool_cap": tc,
            },
        })

    # Generate task-dependent interaction effects.
    task_modifiers: list[dict[str, float]] = []
    for i in range(n_tasks):
        task_modifiers.append({
            "planning_bonus": rng.uniform(-0.3, 0.3),
            "observation_bonus": rng.uniform(-0.3, 0.3),
            "verification_bonus": rng.uniform(-0.3, 0.3),
            "recovery_bonus": rng.uniform(-0.3, 0.3),
            "cost_shift": rng.uniform(-0.3, 0.3),
            "template_id": rng.choice(["extraction", "table_filter", "navigation", "search",
                                       "form_entry", "comparison", "workflow", "recovery"]),
            "difficulty": rng.choice(["easy", "medium", "hard"]),
        })

    # Generate rows.
    rows: list[dict[str, Any]] = []
    for task_idx in range(n_tasks):
        mod = task_modifiers[task_idx]
        task_embed = _make_task_embed(task_idx, mod, seed)

        for policy_idx in range(n_policies):
            param = true_params[policy_idx]
            pl = param["grammar_factors"]["planning"]
            ob = param["grammar_factors"]["observation"]
            ve = param["grammar_factors"]["verification"]

            # Task interaction from factor-difficulty cross.
            task_interaction = (
                planning_effects.get(pl, 0.0) * mod["planning_bonus"]
                + observation_effects.get(ob, 0.0) * mod["observation_bonus"]
                + verification_effects.get(ve, 0.0) * mod["verification_bonus"]
                + recovery_effects.get(param["grammar_factors"]["recovery"], 0.0) * mod["recovery_bonus"]
            )

            difficulty_penalty = {"easy": 0.2, "medium": 0.0, "hard": -0.4}

            # Build logit: main effects + task interaction + difficulty + latent.
            logit = (
                param["base_logit"]
                + task_interaction
                + difficulty_penalty.get(mod["difficulty"], 0.0)
                + param["latent_logit"]
            )
            true_p = _sigmoid(logit)

            # Bernoulli outcome.
            success = rng.random() < true_p

            # Log-normal cost with latent cost residual folded into mean.
            cost_log_mean = param["base_cost_log_mean"] + mod["cost_shift"]
            cost = math.exp(rng.gauss(cost_log_mean, param["cost_log_sigma"]))

            # Termination class.
            if success:
                term_class = "normal_completion"
            elif cost < 50:
                term_class = "invalid_or_tool_error"
            elif rng.random() < 0.5:
                term_class = "tool_call_limit"
            else:
                term_class = "verifier_declared_unsuccessful"

            # model_input NEVER includes the latent residual.
            row = {
                "task_id": f"{task_id_prefix}-{task_idx:04d}",
                "policy_idx": policy_idx,
                "policy_bundle_id": param["treatment"].bundle_id,
                "verified_success": success,
                "true_p": true_p,
                "output_token_cost": cost,
                "termination_class": term_class,
                "model_input": {
                    "task": {
                        "task_embedding": task_embed,
                        "template": mod["template_id"],
                        "difficulty": mod["difficulty"],
                        "task_modifier": mod,
                    },
                    "treatment": {
                        "grammar_factors": param["grammar_factors"],
                        "numeric": {
                            "enforced_tool_call_cap": param["treatment"].tool_call_limit,
                        },
                        "text": param["treatment"].system_prompt,
                    },
                },
            }
            rows.append(row)

    return {
        "rows": rows,
        "true_params": true_params,
        "task_modifiers": task_modifiers,
        "grammar": policies,
        "n_tasks": n_tasks,
        "n_policies": n_policies,
    }


def _make_task_embed(
    task_idx: int,
    modifier: dict[str, Any],
    seed: int,
    dim: int = 32,
) -> list[float]:
    """Create a deterministic embedding with task-factor geometry.

    Synthetic embeddings should transfer across tasks rather than assign an
    unrelated random vector to every row. The first coordinates encode the
    template, difficulty, and continuous task modifiers; remaining dimensions
    are zero-padded to match the frozen-encoder interface.
    """
    del task_idx, seed
    templates = [
        "extraction", "table_filter", "navigation", "search",
        "form_entry", "comparison", "workflow", "recovery",
    ]
    difficulties = ["easy", "medium", "hard"]
    vector = [
        1.0 if modifier.get("template_id") == template else 0.0
        for template in templates
    ]
    vector.extend(
        1.0 if modifier.get("difficulty") == difficulty else 0.0
        for difficulty in difficulties
    )
    vector.extend([
        float(modifier.get("planning_bonus", 0.0)),
        float(modifier.get("observation_bonus", 0.0)),
        float(modifier.get("verification_bonus", 0.0)),
        float(modifier.get("recovery_bonus", 0.0)),
        float(modifier.get("cost_shift", 0.0)),
    ])
    return (vector + [0.0] * dim)[:dim]


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------


def _extract_task_structured(row: dict[str, Any]) -> list[float]:
    """Extract structured task features from a row."""
    task = row["model_input"]["task"]
    mod = task.get("task_modifier", {})

    templates = ["extraction", "table_filter", "navigation", "search",
                 "form_entry", "comparison", "workflow", "recovery"]
    difficulties = ["easy", "medium", "hard"]

    template_idx = templates.index(task["template"]) if task["template"] in templates else 0
    difficulty_idx = difficulties.index(task["difficulty"]) if task["difficulty"] in difficulties else 0

    return [
        float(template_idx) / len(templates),
        float(difficulty_idx) / len(difficulties),
        mod.get("planning_bonus", 0.0),
        mod.get("observation_bonus", 0.0),
        mod.get("verification_bonus", 0.0),
        mod.get("recovery_bonus", 0.0),
        mod.get("cost_shift", 0.0),
    ]


def _term_onehot(row: dict[str, Any]) -> list[float]:
    """Encode termination class as 6-dim one-hot."""
    onehot = [0.0] * 6
    term = str(row.get("termination_class", "normal_completion"))
    mapping = {
        "normal_completion": 0,
        "tool_call_limit": 1,
        "wall_timeout": 2,
        "invalid_or_tool_error": 3,
        "model_runtime_failure": 4,
        "verifier_declared_unsuccessful": 5,
    }
    idx = mapping.get(term, 0)
    onehot[idx] = 1.0
    return onehot


# ---------------------------------------------------------------------------
# Synthetic validation
# ---------------------------------------------------------------------------


def run_synthetic_validation(seed: int = 42) -> dict[str, Any]:
    """Run the full synthetic validation pipeline with proper split design.

    Architecture:
      1. Grammar split: 48 meta-train / 12 dev / 12 final policies.
      2. Multiple meta-training and development latent worlds; each dev/final
         panel uses 16 calibration + 64 target tasks.
      3. DGP with SHA-256 latent residuals shared within each latent world.
      4. Frozen ordered calibration panels with nested k=0/4/8/16 prefixes.
      5. Paired episodic training at nested k=0/4/8/16 and deterministic
         dev-panel validation.
      6. Best-weight checkpointing with deep copy; restore before final eval.
      7. Per-k evaluation reporting bin-log-loss, Brier, ranking accuracy,
         true-p cross-entropy, and explicit metric deltas.
      8. Negative control via deterministic policy-panel derangement.
      9. Frontier evaluation on k=8 predictions.

    Returns:
        Validation report dict with protocol metadata, metrics, and claims.
    """
    if not TORCH_AVAILABLE:
        return {
            "validation": "skipped",
            "reason": "PyTorch is not installed. Synthetic validation requires PyTorch.",
            "claims": {
                "cnp_trains_stably": None,
                "k8_improves_over_k0": None,
                "shuffled_negative_control_passes": None,
                "frontier_sensible": None,
            },
        }

    import torch
    from .meta_cnp import MetaCNPModel, compute_loss, set_seed as cnp_set_seed, count_parameters
    from .calibration import fit_normalization_stats
    from .frontier_eval import compute_frontier

    cnp_set_seed(seed)

    # ---- 1. Grammar and policy splits ----------------------------------------
    grammar = enumerate_unbrowser_grammar()
    assert len(grammar) == 72, f"Expected 72 policies, got {len(grammar)}"

    meta_train, development, final_held = meta_grammar.split_policies(grammar, seed=seed)
    assert len(meta_train) == 48
    assert len(development) == 12
    assert len(final_held) == 12

    n_meta_policies = len(meta_train)
    n_dev_policies = len(development)
    n_final_policies = len(final_held)

    # ---- 2. Generate independent latent worlds -------------------------------
    # Reusing one latent per training policy would let the unique descriptor
    # memorize that latent. Multiple worlds force the model to infer the hidden
    # residual from context outcomes.
    meta_worlds = [
        generate_synthetic_outcomes(
            meta_train,
            _N_META_TASKS,
            n_meta_policies,
            seed=seed + world_idx * 10_000,
            latent_seed=_LATENT_SEED + world_idx * 104_729,
            latent_logit_scale=_LATENT_LOGIT_SCALE,
            task_id_prefix=f"synthetic-meta-w{world_idx}",
        )
        for world_idx in range(_N_META_WORLDS)
    ]
    dev_worlds = [
        generate_synthetic_outcomes(
            development,
            _N_DEV_TASKS,
            n_dev_policies,
            seed=seed + 1000 + world_idx * 10_000,
            latent_seed=_LATENT_SEED + 1_000_003 + world_idx * 104_729,
            latent_logit_scale=_LATENT_LOGIT_SCALE,
            task_id_prefix=f"synthetic-dev-w{world_idx}",
        )
        for world_idx in range(_N_DEV_WORLDS)
    ]
    final_data = generate_synthetic_outcomes(
        final_held, _N_FINAL_TASKS, n_final_policies, seed=seed + 2000,
        latent_seed=_LATENT_SEED + 2_000_003,
        latent_logit_scale=_LATENT_LOGIT_SCALE,
        task_id_prefix="synthetic-final",
    )

    # ---- 3. Split calibration / target pools ---------------------------------
    dev_panels = [
        _split_calibration_target(
            world["rows"], _N_DEV_TASKS, n_dev_policies, _N_CAL_TASKS,
        )
        for world in dev_worlds
    ]
    final_cal_rows, final_target_rows = _split_calibration_target(
        final_data["rows"], _N_FINAL_TASKS, n_final_policies, _N_CAL_TASKS,
    )

    meta_rows = [
        row
        for world in meta_worlds
        for row in world["rows"]
    ]
    normalization_stats = fit_normalization_stats(meta_rows)

    # ---- 4. Build CNP model (under 1M params) --------------------------------
    model = MetaCNPModel(
        structured_task_dim=7,
        text_embed_dim=32,
        hx_dim=96,
        hp_dim=64,
        ei_dim=128,
        context_hidden=256,
        decoder_hidden=128,
        dropout=0.15,
        num_heads=2,
    )

    param_count = count_parameters(model)
    assert param_count < 1_000_000, f"Model has {param_count} params, exceeding 1M limit"

    device = "cpu"
    model.to(device)

    # ---- 5. Episodic meta-training on meta-train pool ------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_rng = random.Random(seed + 7777)

    losses: list[float] = []
    success_losses: list[float] = []
    dev_losses: list[float] = []
    best_epoch = 0
    best_dev_loss = float("inf")
    best_state_dict: dict[str, Any] | None = None

    for epoch in range(_N_EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        epoch_success_loss = 0.0
        for _ in range(_N_BATCHES_PER_EPOCH):
            batch_structured: list[list[float]] = []
            batch_text_emb: list[list[float]] = []
            batch_policy_desc: list[list[float]] = []
            batch_target_success: list[float] = []
            batch_target_cost: list[float] = []
            batch_target_term: list[int] = []

            k_max = 16
            batch_ctx_structured: list[list[list[float]]] = []
            batch_ctx_text_emb: list[list[list[float]]] = []
            batch_ctx_success: list[list[float]] = []
            batch_ctx_cost: list[list[float]] = []
            batch_ctx_term: list[list[list[float]]] = []
            batch_ctx_mask: list[list[float]] = []

            # Eight independent (world, policy, target) episodes, each expanded
            # to the nested k={0,4,8,16} conditions in the same batch.
            for _base_episode in range(_N_BATCH_SIZE // 4):
                world_idx = train_rng.randrange(_N_META_WORLDS)
                world_rows = meta_worlds[world_idx]["rows"]
                task_idx = train_rng.randrange(_N_META_TASKS)
                policy_idx = train_rng.randrange(n_meta_policies)
                target_idx = task_idx * n_meta_policies + policy_idx
                target_row = world_rows[target_idx]

                term_map = {
                    "normal_completion": 0, "tool_call_limit": 1, "wall_timeout": 2,
                    "invalid_or_tool_error": 3, "model_runtime_failure": 4,
                    "verifier_declared_unsuccessful": 5,
                }
                other_tasks = [t for t in range(_N_META_TASKS) if t != task_idx]
                train_rng.shuffle(other_tasks)
                ordered_context_tasks = other_tasks[:k_max]

                for k in (0, 4, 8, 16):
                    batch_structured.append(_extract_task_structured(target_row))
                    batch_text_emb.append(
                        target_row["model_input"]["task"]["task_embedding"]
                    )
                    batch_policy_desc.append(grammar_factor_vector(meta_train[policy_idx]))
                    batch_target_success.append(
                        float(target_row["true_p"])
                    )
                    batch_target_cost.append(target_row["output_token_cost"])
                    batch_target_term.append(
                        term_map.get(target_row["termination_class"], 0)
                    )

                    ctx_structured: list[list[float]] = []
                    ctx_text_emb: list[list[float]] = []
                    ctx_success: list[float] = []
                    ctx_cost: list[float] = []
                    ctx_term: list[list[float]] = []
                    ctx_mask: list[float] = []
                    for context_task_idx in ordered_context_tasks[:k]:
                        ctx_idx = context_task_idx * n_meta_policies + policy_idx
                        ctx_row = world_rows[ctx_idx]
                        ctx_structured.append(_extract_task_structured(ctx_row))
                        ctx_text_emb.append(
                            ctx_row["model_input"]["task"]["task_embedding"]
                        )
                        ctx_success.append(
                            1.0 if ctx_row["verified_success"] else 0.0
                        )
                        ctx_cost.append(_normalize_context_cost(
                            float(ctx_row["output_token_cost"]), normalization_stats,
                        ))
                        ctx_term.append(_term_onehot(ctx_row))
                        ctx_mask.append(1.0)

                    while len(ctx_structured) < k_max:
                        ctx_structured.append([0.0] * len(batch_structured[-1]))
                        ctx_text_emb.append([0.0] * len(batch_text_emb[-1]))
                        ctx_success.append(0.0)
                        ctx_cost.append(0.0)
                        ctx_term.append([0.0] * 6)
                        ctx_mask.append(0.0)

                    batch_ctx_structured.append(ctx_structured)
                    batch_ctx_text_emb.append(ctx_text_emb)
                    batch_ctx_success.append(ctx_success)
                    batch_ctx_cost.append(ctx_cost)
                    batch_ctx_term.append(ctx_term)
                    batch_ctx_mask.append(ctx_mask)

            # Convert to tensors.
            target_structured_t = torch.tensor(batch_structured, dtype=torch.float32)
            target_text_t = torch.tensor(batch_text_emb, dtype=torch.float32)
            policy_desc_t = torch.tensor(batch_policy_desc, dtype=torch.float32)

            ctx_structured_t = torch.tensor(batch_ctx_structured, dtype=torch.float32)
            ctx_text_t = torch.tensor(batch_ctx_text_emb, dtype=torch.float32)
            ctx_success_t = torch.tensor(batch_ctx_success, dtype=torch.float32).unsqueeze(-1)
            ctx_cost_t = torch.tensor(batch_ctx_cost, dtype=torch.float32).unsqueeze(-1)
            ctx_term_t = torch.tensor(batch_ctx_term, dtype=torch.float32)
            ctx_mask_t = torch.tensor(batch_ctx_mask, dtype=torch.float32).unsqueeze(-1)

            target_success_t = torch.tensor(batch_target_success, dtype=torch.float32)
            target_cost_t = torch.tensor(batch_target_cost, dtype=torch.float32)
            target_term_t = torch.tensor(batch_target_term, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(
                target_structured_t, target_text_t, policy_desc_t,
                ctx_structured_t, ctx_text_t,
                ctx_success_t, ctx_term_t, ctx_cost_t, ctx_mask_t,
            )
            loss_dict = compute_loss(
                outputs,
                target_success_t,
                target_cost_t,
                target_term_t,
                cost_loss_weight=0.1,
                term_loss_weight=0.05,
            )
            loss_dict["total"].backward()
            optimizer.step()

            epoch_loss += loss_dict["total"].item()
            epoch_success_loss += loss_dict["success"].item()
            n_batches += 1

        losses.append(epoch_loss / max(1, n_batches))
        success_losses.append(epoch_success_loss / max(1, n_batches))

        # Development evaluation: fixed deterministic full-panel, NOT random.
        model.eval()
        dev_evals = [
            _evaluate_model_on_heldout(
                model,
                dev_cal_rows,
                dev_target_rows,
                development,
                n_dev_policies,
                _N_DEV_TASKS - _N_CAL_TASKS,
                _N_CAL_TASKS,
                k=8,
                normalization_stats=normalization_stats,
            )
            for dev_cal_rows, dev_target_rows in dev_panels
        ]
        # Known DGP probabilities provide a low-noise synthetic checkpoint
        # selector. Binary log loss remains the reported external metric.
        dev_loss = sum(
            evaluation["true_p_cross_entropy"] for evaluation in dev_evals
        ) / len(dev_evals)
        dev_losses.append(dev_loss)

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_epoch = epoch
            # Deep-copy best weights.
            best_state_dict = copy.deepcopy({k: v.cpu().clone() for k, v in model.state_dict().items()})

        if epoch - best_epoch > _EARLY_STOP_PATIENCE:
            break

    # ---- 6. Restore best weights before final evaluation ---------------------
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    model.eval()

    selected_dev_by_k = {
        k: _mean_evaluation_metrics([
            _evaluate_model_on_heldout(
                model,
                dev_cal_rows,
                dev_target_rows,
                development,
                n_dev_policies,
                _N_DEV_TASKS - _N_CAL_TASKS,
                _N_CAL_TASKS,
                k=k,
                normalization_stats=normalization_stats,
            )
            for dev_cal_rows, dev_target_rows in dev_panels
        ])
        for k in (0, 8)
    }

    # ---- 7. Evaluate on final held-out at all k ------------------------------
    k_values = [0, 4, 8, 16]
    results_by_k: dict[int, dict[str, Any]] = {}
    for k in k_values:
        results_by_k[k] = _evaluate_model_on_heldout(
            model, final_cal_rows, final_target_rows, final_held,
            n_final_policies, _N_FINAL_TASKS - _N_CAL_TASKS, _N_CAL_TASKS,
            k=k, normalization_stats=normalization_stats,
        )

    # ---- 8. Negative control: policy-panel derangement -----------------------
    derangement = _build_derangement(n_final_policies)
    results_deranged = _evaluate_model_on_heldout(
        model, final_cal_rows, final_target_rows, final_held,
        n_final_policies, _N_FINAL_TASKS - _N_CAL_TASKS, _N_CAL_TASKS,
        k=8, normalization_stats=normalization_stats,
        derangement_mapping=derangement,
    )

    # ---- 9. Frontier evaluation (k=8 primary) --------------------------------
    predictions_k8 = results_by_k[8]["predictions_by_task"]
    outcomes_k8 = results_by_k[8]["outcomes_by_task"]
    lambda_grid = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    frontier = compute_frontier(predictions_k8, outcomes_k8, lambda_grid)

    # ---- 10. Assemble validation report --------------------------------------
    k0 = results_by_k[0]
    k8 = results_by_k[8]

    cnp_trains_stably = (
        len(success_losses) > 0
        and success_losses[best_epoch] < success_losses[0] * 0.98
        and math.isfinite(best_dev_loss)
        and param_count < 1_000_000
    )

    # k=8-vs-k=0 claim: requires strict log-loss improvement AND non-worse
    # ranking accuracy, with explicit deltas.
    delta_log_loss = k0["log_loss_binary"] - k8["log_loss_binary"]
    delta_ranking = k8["ranking_accuracy"] - k0["ranking_accuracy"]
    k8_improves_over_k0 = bool(
        k8["log_loss_binary"] < k0["log_loss_binary"]
        and k8["ranking_accuracy"] >= k0["ranking_accuracy"]
    )

    # Negative control: derangement must remove most of the clean calibration
    # gain under the same held-out log-loss metric.
    deranged_k8 = results_deranged
    delta_deranged_log_loss = k0["log_loss_binary"] - deranged_k8["log_loss_binary"]
    shuffled_negative_control_passes = bool(
        k8_improves_over_k0
        and delta_deranged_log_loss <= max(0.0, delta_log_loss * 0.25)
    )

    frontier_sensible = (
        frontier["frontier_area"] >= 0.0
        and 0.0 <= frontier["pure_success_rate"] <= 1.0
        and len(frontier.get("lambda_results", [])) == len(lambda_grid)
    )

    # Protocol metadata for reproducibility audit.
    dev_row_pools = [rows for panel in dev_panels for rows in panel]
    protocol_meta = {
        "latent_seed": _LATENT_SEED,
        "latent_logit_scale": _LATENT_LOGIT_SCALE,
        "n_meta_worlds": _N_META_WORLDS,
        "n_dev_worlds": _N_DEV_WORLDS,
        "n_meta_tasks": _N_META_TASKS,
        "n_dev_tasks": _N_DEV_TASKS,
        "n_final_tasks": _N_FINAL_TASKS,
        "n_cal_tasks": _N_CAL_TASKS,
        "dev_cal_task_ids": sorted({
            row["task_id"]
            for dev_cal_rows, _ in dev_panels
            for row in dev_cal_rows
        }),
        "dev_target_task_ids": sorted({
            row["task_id"]
            for _, dev_target_rows in dev_panels
            for row in dev_target_rows
        }),
        "final_cal_task_ids": sorted({r["task_id"] for r in final_cal_rows}),
        "final_target_task_ids": sorted({r["task_id"] for r in final_target_rows}),
        "cal_target_disjoint_dev": all(
            _check_disjoint(dev_cal_rows, dev_target_rows)
            for dev_cal_rows, dev_target_rows in dev_panels
        ),
        "cal_target_disjoint_final": _check_disjoint(final_cal_rows, final_target_rows),
        "all_task_pools_globally_disjoint": _all_task_pools_disjoint(
            meta_rows, *dev_row_pools, final_cal_rows, final_target_rows,
        ),
        "dev_panel_frozen": True,  # cal rows indexed by (cal_task, policy)
        "cal_prefix_nested": True,  # k=4 subset of k=8 subset of k=16
        "derangement_mapping": {str(k): str(v) for k, v in derangement.items()},
        "cost_normalization": normalization_stats,
        "success_training_target": "synthetic_true_probability",
    }

    return {
        "validation": "complete",
        "warning": (
            "All data is synthetic. This validates plumbing only. "
            "No claims about real effectiveness."
        ),
        "grammar": {
            "n_policies_total": len(grammar),
            "n_meta_train": n_meta_policies,
            "n_development": n_dev_policies,
            "n_final_held": n_final_policies,
        },
        "data": {
            "n_meta_tasks_per_world": _N_META_TASKS,
            "n_meta_worlds": _N_META_WORLDS,
            "n_meta_rows": len(meta_rows),
            "n_dev_tasks_per_world": _N_DEV_TASKS,
            "n_dev_worlds": _N_DEV_WORLDS,
            "n_dev_cal_rows": sum(len(panel[0]) for panel in dev_panels),
            "n_dev_target_rows": sum(len(panel[1]) for panel in dev_panels),
            "n_final_tasks": _N_FINAL_TASKS,
            "n_final_cal_rows": len(final_cal_rows),
            "n_final_target_rows": len(final_target_rows),
        },
        "model": {
            "param_count": param_count,
            "under_1M": param_count < 1_000_000,
            "n_epochs_trained": len(losses),
            "best_epoch": best_epoch,
            "last_train_loss": losses[-1] if losses else float("nan"),
            "initial_train_loss": losses[0] if losses else float("nan"),
            "last_success_loss": success_losses[-1] if success_losses else float("nan"),
            "initial_success_loss": success_losses[0] if success_losses else float("nan"),
            "selected_epoch_train_loss": losses[best_epoch] if losses else float("nan"),
            "best_dev_loss": best_dev_loss,
            "last_dev_loss": dev_losses[-1] if dev_losses else float("nan"),
            "dev_selection_metric": "true_p_cross_entropy_k8",
        },
        "results": {
            "development": {
                "k0_log_loss_binary": selected_dev_by_k[0]["log_loss_binary"],
                "k8_log_loss_binary": selected_dev_by_k[8]["log_loss_binary"],
                "k0_true_p_cross_entropy": selected_dev_by_k[0]["true_p_cross_entropy"],
                "k8_true_p_cross_entropy": selected_dev_by_k[8]["true_p_cross_entropy"],
                "k0_ranking_accuracy": selected_dev_by_k[0]["ranking_accuracy"],
                "k8_ranking_accuracy": selected_dev_by_k[8]["ranking_accuracy"],
            },
            "primary": {
                "k": 8,
                "log_loss_binary": k8["log_loss_binary"],
                "brier": k8["brier"],
                "mean_predicted_prob": k8["mean_predicted_prob"],
                "ranking_accuracy": k8["ranking_accuracy"],
                "true_p_cross_entropy": k8["true_p_cross_entropy"],
                "true_p_brier": k8["true_p_brier"],
            },
            "by_k": {
                str(k): {
                    "log_loss_binary": results_by_k[k]["log_loss_binary"],
                    "brier": results_by_k[k]["brier"],
                    "mean_predicted_prob": results_by_k[k]["mean_predicted_prob"],
                    "ranking_accuracy": results_by_k[k]["ranking_accuracy"],
                    "true_p_cross_entropy": results_by_k[k]["true_p_cross_entropy"],
                    "true_p_brier": results_by_k[k]["true_p_brier"],
                }
                for k in k_values
            },
            "k8_minus_k0": {
                "delta_log_loss": delta_log_loss,
                "delta_brier": k0["brier"] - k8["brier"],
                "delta_ranking_accuracy": delta_ranking,
                "delta_true_p_ce": k0["true_p_cross_entropy"] - k8["true_p_cross_entropy"],
                "k8_improves_over_k0": k8_improves_over_k0,
            },
            "deranged_k8": {
                "log_loss_binary": deranged_k8["log_loss_binary"],
                "brier": deranged_k8["brier"],
                "ranking_accuracy": deranged_k8["ranking_accuracy"],
                "true_p_cross_entropy": deranged_k8["true_p_cross_entropy"],
                "delta_log_loss_vs_k0": delta_deranged_log_loss,
            },
        },
        "frontier": frontier,
        "protocol_meta": protocol_meta,
        "claims": {
            "cnp_trains_stably": cnp_trains_stably,
            "k8_improves_over_k0": k8_improves_over_k0,
            "shuffled_negative_control_passes": shuffled_negative_control_passes,
            "frontier_sensible": frontier_sensible,
        },
        "verdict": {
            "all_checks_pass": (
                cnp_trains_stably
                and k8_improves_over_k0
                and shuffled_negative_control_passes
                and frontier_sensible
            ),
        },
    }


def _check_disjoint(
    cal_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
) -> bool:
    """Verify calibration and target task_id sets are disjoint."""
    cal_ids = {r["task_id"] for r in cal_rows}
    target_ids = {r["task_id"] for r in target_rows}
    return len(cal_ids & target_ids) == 0


def _all_task_pools_disjoint(*row_pools: list[dict[str, Any]]) -> bool:
    """Return whether every task ID belongs to exactly one task pool."""
    seen: set[str] = set()
    for rows in row_pools:
        task_ids = {str(row["task_id"]) for row in rows}
        if seen & task_ids:
            return False
        seen.update(task_ids)
    return True


# ---------------------------------------------------------------------------
# Held-out evaluation with proper cal/target separation
# ---------------------------------------------------------------------------


def _evaluate_model_on_heldout(
    model: Any,
    calibration_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    policies: list[Any],
    n_policies: int,
    n_target_tasks: int,
    n_cal_tasks: int,
    k: int,
    *,
    normalization_stats: dict[str, Any],
    derangement_mapping: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Evaluate model on held-out (calibration, target) split.

    Context is built SOLELY from ``calibration_rows`` using the first ``k``
    calibration tasks in frozen order.  Target predictions use ``target_rows``
    and NEVER contribute to context.

    When ``derangement_mapping`` is provided (negative control), each policy's
    calibration panel is replaced with the panel from a different policy,
    destroying the policy-outcome correspondence.

    Returns:
        dict with:
        - ``log_loss_binary``, ``brier``: vs observed binary outcomes
        - ``mean_predicted_prob``: average predicted success probability
        - ``ranking_accuracy``: pairwise ranking vs true (DGP) probabilities
        - ``true_p_cross_entropy``, ``true_p_brier``: vs DGP true probs
        - ``predictions_by_task``, ``outcomes_by_task``: for frontier eval
    """
    import torch
    import torch.nn.functional as functional

    if model.training:
        raise RuntimeError("held-out evaluation requires model.eval()")
    if len(policies) != n_policies:
        raise ValueError("n_policies does not match policies")
    if len(calibration_rows) != n_cal_tasks * n_policies:
        raise ValueError("calibration panel is incomplete")
    if len(target_rows) != n_target_tasks * n_policies:
        raise ValueError("target panel is incomplete")
    if k < 0:
        raise ValueError("k must be nonnegative")

    k_actual = min(k, n_cal_tasks)
    context_width = max(k_actual, 1)
    device = next(model.parameters()).device

    target_structured_batch: list[list[float]] = []
    target_text_batch: list[list[float]] = []
    policy_desc_batch: list[list[float]] = []
    context_structured_batch: list[list[list[float]]] = []
    context_text_batch: list[list[list[float]]] = []
    context_success_batch: list[list[float]] = []
    context_cost_batch: list[list[float]] = []
    context_term_batch: list[list[list[float]]] = []
    context_mask_batch: list[list[float]] = []

    for target_task_idx in range(n_target_tasks):
        for policy_idx in range(n_policies):
            target_row = target_rows[target_task_idx * n_policies + policy_idx]
            target_structured_batch.append(_extract_task_structured(target_row))
            target_text_batch.append(
                target_row["model_input"]["task"]["task_embedding"]
            )
            policy_desc_batch.append(grammar_factor_vector(policies[policy_idx]))

            ctx_structured: list[list[float]] = []
            ctx_text: list[list[float]] = []
            ctx_success: list[float] = []
            ctx_cost: list[float] = []
            ctx_term: list[list[float]] = []
            ctx_mask: list[float] = []
            source_policy = (
                derangement_mapping[policy_idx]
                if derangement_mapping is not None
                else policy_idx
            )
            for cal_idx in range(k_actual):
                cal_row = calibration_rows[cal_idx * n_policies + source_policy]
                ctx_structured.append(_extract_task_structured(cal_row))
                ctx_text.append(cal_row["model_input"]["task"]["task_embedding"])
                ctx_success.append(1.0 if cal_row["verified_success"] else 0.0)
                ctx_cost.append(_normalize_context_cost(
                    float(cal_row["output_token_cost"]), normalization_stats,
                ))
                ctx_term.append(_term_onehot(cal_row))
                ctx_mask.append(1.0)

            if k_actual == 0:
                ctx_structured.append([0.0] * len(target_structured_batch[-1]))
                ctx_text.append([0.0] * len(target_text_batch[-1]))
                ctx_success.append(0.0)
                ctx_cost.append(0.0)
                ctx_term.append([0.0] * 6)
                ctx_mask.append(0.0)

            if len(ctx_structured) != context_width:
                raise AssertionError("context width does not match k")
            context_structured_batch.append(ctx_structured)
            context_text_batch.append(ctx_text)
            context_success_batch.append(ctx_success)
            context_cost_batch.append(ctx_cost)
            context_term_batch.append(ctx_term)
            context_mask_batch.append(ctx_mask)

    with torch.no_grad():
        outputs = model(
            torch.tensor(target_structured_batch, dtype=torch.float32, device=device),
            torch.tensor(target_text_batch, dtype=torch.float32, device=device),
            torch.tensor(policy_desc_batch, dtype=torch.float32, device=device),
            torch.tensor(context_structured_batch, dtype=torch.float32, device=device),
            torch.tensor(context_text_batch, dtype=torch.float32, device=device),
            torch.tensor(context_success_batch, dtype=torch.float32, device=device).unsqueeze(-1),
            torch.tensor(context_term_batch, dtype=torch.float32, device=device),
            torch.tensor(context_cost_batch, dtype=torch.float32, device=device).unsqueeze(-1),
            torch.tensor(context_mask_batch, dtype=torch.float32, device=device).unsqueeze(-1),
        )
        success_probs = torch.sigmoid(outputs["logit_success"])
        mu_log = outputs["cost_params"][:, 0]
        raw_scale = outputs["cost_params"][:, 1]
        sigma_log = functional.softplus(raw_scale)
        sigma_sq = torch.clamp(sigma_log.square(), max=50.0)
        expected_log_cost = torch.clamp(mu_log + sigma_sq / 2.0, max=50.0)
        cost_means = torch.clamp(torch.exp(expected_log_cost) - 1.0, min=0.0)
        log_cost_variance = (
            torch.log(torch.expm1(sigma_sq).clamp(min=1e-12))
            + 2.0 * mu_log
            + sigma_sq
        )
        cost_stds = torch.exp(torch.clamp(log_cost_variance, max=50.0) / 2.0)
        termination_probs = functional.softmax(outputs["term_logits"], dim=-1)

    success_values = success_probs.cpu().tolist()
    cost_mean_values = cost_means.cpu().tolist()
    cost_std_values = cost_stds.cpu().tolist()
    term_values = termination_probs.cpu().tolist()
    raw_logit_values = outputs["logit_success"].cpu().tolist()
    raw_mu_values = mu_log.cpu().tolist()
    raw_scale_values = raw_scale.cpu().tolist()

    all_pred_probs: list[float] = []
    all_outcomes: list[float] = []
    all_true_probs: list[float] = []
    pred_probs_by_task: list[list[float]] = []
    true_probs_by_task: list[list[float]] = []
    outcomes_by_task: list[list[dict[str, Any]]] = []
    predictions_by_task: list[list[dict[str, Any]]] = []

    offset = 0
    for target_task_idx in range(n_target_tasks):
        task_pred_probs: list[float] = []
        task_true_probs: list[float] = []
        task_preds: list[dict[str, Any]] = []
        task_outcomes: list[dict[str, Any]] = []
        for policy_idx in range(n_policies):
            target_row = target_rows[target_task_idx * n_policies + policy_idx]
            probability = float(success_values[offset])
            prediction = {
                "success_prob": probability,
                "cost_mean": float(cost_mean_values[offset]),
                "cost_std": float(cost_std_values[offset]),
                "termination_probs": term_values[offset],
                "raw_logit_success": float(raw_logit_values[offset]),
                "raw_cost_mu_log": float(raw_mu_values[offset]),
                "raw_cost_log_sigma": float(raw_scale_values[offset]),
            }
            outcome = {
                "verified_success": target_row["verified_success"],
                "cost": target_row["output_token_cost"],
                "termination_class": target_row["termination_class"],
            }
            true_probability = float(target_row["true_p"])
            task_pred_probs.append(probability)
            task_true_probs.append(true_probability)
            task_preds.append(prediction)
            task_outcomes.append(outcome)
            all_pred_probs.append(probability)
            all_outcomes.append(1.0 if target_row["verified_success"] else 0.0)
            all_true_probs.append(true_probability)
            offset += 1
        pred_probs_by_task.append(task_pred_probs)
        true_probs_by_task.append(task_true_probs)
        predictions_by_task.append(task_preds)
        outcomes_by_task.append(task_outcomes)

    log_loss = _binary_log_loss(all_pred_probs, all_outcomes)
    brier = _brier_score(all_pred_probs, all_outcomes)
    mean_pred_prob = sum(all_pred_probs) / len(all_pred_probs) if all_pred_probs else float("nan")
    ranking_acc = _global_pairwise_ranking_accuracy(pred_probs_by_task, true_probs_by_task)
    true_p_ce = _cross_entropy_vs_true(all_pred_probs, all_true_probs)
    true_p_brier = _brier_vs_true(all_pred_probs, all_true_probs)
    predicted_policy_means = [
        sum(task_probs[j] for task_probs in pred_probs_by_task) / n_target_tasks
        for j in range(n_policies)
    ]
    true_policy_means = [
        sum(task_probs[j] for task_probs in true_probs_by_task) / n_target_tasks
        for j in range(n_policies)
    ]

    return {
        "log_loss_binary": log_loss,
        "brier": brier,
        "mean_predicted_prob": mean_pred_prob,
        "ranking_accuracy": ranking_acc,
        "true_p_cross_entropy": true_p_ce,
        "true_p_brier": true_p_brier,
        "predictions_by_task": predictions_by_task,
        "outcomes_by_task": outcomes_by_task,
        "predicted_policy_means": predicted_policy_means,
        "true_policy_means": true_policy_means,
        "k_actual": k_actual,
    }


__all__ = [
    "generate_synthetic_outcomes",
    "run_synthetic_validation",
    "_compute_policy_latent",
    "_split_calibration_target",
    "_build_derangement",
    "_LATENT_SEED",
]
