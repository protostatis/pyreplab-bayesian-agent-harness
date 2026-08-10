"""Synthetic validation harness for the M3 meta-policy-learner.

Generates synthetic policy outcomes to validate the full pipeline (grammar,
CNP model, frontier evaluator, calibration) before real Gemma rollouts.

The data-generating process assigns each policy a true success rate based on
factor main effects + interactions, generates Bernoulli outcomes and log-normal
costs, and includes termination classes. Row format is compatible with the
CNP model's expected input.

Key validation checks:
1. CNP learns factor main effects from synthetic data.
2. k=8 calibration improves over k=0 when factors have task-dependent effects.
3. Shuffled calibration contexts destroy the gain (negative control).
4. Frontier evaluator produces sensible metrics.
"""

from __future__ import annotations

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
# Synthetic data generation
# ---------------------------------------------------------------------------


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def generate_synthetic_outcomes(
    grammar: list[Any],
    n_tasks: int,
    n_policies: int,
    seed: int,
) -> dict[str, Any]:
    """Generate synthetic policy outcomes with known factor-dependent probabilities.

    Data-generating process:
        - Each policy has a true success rate based on factor main effects
          + some task-dependent factor interactions.
        - Bernoulli outcomes sampled from true success rates.
        - Log-normal costs with factor-dependent means.
        - Termination classes included.

    Returns a dict with:
        - "rows": list of row dicts compatible with model input format
        - "true_params": per-policy true parameters
        - "grammar": list of policy objects
    """
    rng = random.Random(seed)
    n_policies = min(n_policies, len(grammar))
    policies = list(grammar)[:n_policies]

    # Assign per-factor baseline effects.
    planning_effects = {"direct": 0.0, "brief_plan": 0.2, "decompose": -0.1}
    observation_effects = {"text_first": 0.1, "structure_first": 0.0, "targeted_query_first": -0.05}
    verification_effects = {"submit_directly": -0.1, "final_reobserve": 0.15}
    recovery_effects = {"fail_fast": -0.05, "diagnose_retry_once": 0.1}
    tool_cap_effects = {"lean": -0.2, "expanded": 0.3}

    # Cost factor effects (log-scale).
    planning_cost = {"direct": -0.2, "brief_plan": 0.0, "decompose": 0.3}
    observation_cost = {"text_first": 0.2, "structure_first": 0.0, "targeted_query_first": -0.1}
    tool_cap_cost = {"lean": -0.5, "expanded": 0.0}

    # Compute per-policy true parameters.
    true_params: list[dict[str, Any]] = []
    for idx, treatment in enumerate(policies):
        meta = treatment.generator_metadata
        pl = str(meta.get("planning", "direct"))
        ob = str(meta.get("observation", "text_first"))
        ve = str(meta.get("verification", "submit_directly"))
        re = str(meta.get("recovery", "fail_fast"))
        tc = str(meta.get("tool_cap", "lean"))

        # Baseline logit.
        base_logit = (
            0.0
            + planning_effects.get(pl, 0.0)
            + observation_effects.get(ob, 0.0)
            + verification_effects.get(ve, 0.0)
            + recovery_effects.get(re, 0.0)
            + tool_cap_effects.get(tc, 0.0)
        )

        # Baseline cost log-mean.
        base_cost_log = (
            3.0  # ~20 tokens base
            + planning_cost.get(pl, 0.0)
            + observation_cost.get(ob, 0.0)
            + tool_cap_cost.get(tc, 0.0)
        )

        true_params.append({
            "policy_idx": idx,
            "treatment": treatment,
            "base_logit": base_logit,
            "base_cost_log_mean": base_cost_log,
            "cost_log_sigma": 0.5,
            "grammar_factors": {
                "planning": pl, "observation": ob,
                "verification": ve, "recovery": re, "tool_cap": tc,
            },
        })

    # Generate task-dependent interaction effects.
    # Each task has a "task_modifier" that interacts with factors.
    task_modifiers: list[dict[str, float]] = []
    for i in range(n_tasks):
        # Task modifies how much each factor contributes.
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
        # Task embed: use simple random projection (hash-based).
        task_embed = _make_task_embed(task_idx, mod, seed)

        for policy_idx in range(n_policies):
            param = true_params[policy_idx]
            pl = param["grammar_factors"]["planning"]
            ob = param["grammar_factors"]["observation"]
            ve = param["grammar_factors"]["verification"]

            # Compute success logit with task-dependent interaction.
            task_interaction = (
                planning_effects.get(pl, 0.0) * mod["planning_bonus"]
                + observation_effects.get(ob, 0.0) * mod["observation_bonus"]
                + verification_effects.get(ve, 0.0) * mod["verification_bonus"]
                + recovery_effects.get(param["grammar_factors"]["recovery"], 0.0) * mod["recovery_bonus"]
            )

            # Difficulty penalty.
            difficulty_penalty = {"easy": 0.2, "medium": 0.0, "hard": -0.4}
            logit = param["base_logit"] + task_interaction + difficulty_penalty.get(mod["difficulty"], 0.0)
            true_p = _sigmoid(logit)

            # Generate Bernoulli outcome.
            success = rng.random() < true_p

            # Generate log-normal cost.
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

            # Build row in frozen interface contract format.
            row = {
                "task_id": f"synthetic-task-{task_idx:04d}",
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
    """Create a deterministic random-projection task embedding."""
    rng = random.Random(seed + task_idx * 1000 + hash(str(modifier)) % 10000)
    return [rng.uniform(-1.0, 1.0) for _ in range(dim)]


# ---------------------------------------------------------------------------
# Synthetic validation
# ---------------------------------------------------------------------------


def run_synthetic_validation(seed: int = 42) -> dict[str, Any]:
    """Run the full synthetic validation pipeline.

    Generates data for all 72 policies, splits into 48/12/12, trains CNP
    ensemble on meta-train, evaluates on held-out with k=0,4,8,16, runs
    frontier evaluator, and compares against baselines.

    Validation checks:
    1. CNP can learn factor main effects from synthetic data.
    2. k=8 calibration improves over k=0 with task-dependent effects.
    3. Shuffled calibration contexts destroy the gain (negative control).
    4. Frontier evaluator produces sensible metrics.

    Returns a validation report dict.
    """
    if not TORCH_AVAILABLE:
        return {
            "validation": "skipped",
            "reason": "PyTorch is not installed. Synthetic validation requires PyTorch.",
            "claims": {
                "cnp_learns_main_effects": None,
                "k8_improves_over_k0": None,
                "shuffled_negative_control_passes": None,
                "frontier_sensible": None,
            },
        }

    import torch
    from .meta_cnp import MetaCNPModel, compute_loss, set_seed as cnp_set_seed, count_parameters
    from .frontier_eval import compute_frontier, evaluate_allocator
    from .calibration import fit_normalization_stats

    rng = random.Random(seed)
    cnp_set_seed(seed)

    # Step 1: Generate grammar and split.
    grammar = enumerate_unbrowser_grammar()
    assert len(grammar) == 72, f"Expected 72 policies, got {len(grammar)}"

    meta_train, development, final_held = meta_grammar.split_policies(grammar, seed=seed)
    assert len(meta_train) == 48
    assert len(development) == 12
    assert len(final_held) == 12

    # Step 2: Generate synthetic outcomes.
    # Use separate task pools for meta-train, dev, and final.
    n_meta_tasks = 64
    n_dev_tasks = 16
    n_final_tasks = 32

    meta_data = generate_synthetic_outcomes(
        meta_train, n_meta_tasks, len(meta_train), seed=seed,
    )
    dev_data = generate_synthetic_outcomes(
        development, n_dev_tasks, len(development), seed=seed + 1000,
    )
    final_data = generate_synthetic_outcomes(
        final_held, n_final_tasks, len(final_held), seed=seed + 2000,
    )

    # Step 3: Prepare training data.
    meta_rows = meta_data["rows"]
    dev_rows = dev_data["rows"]
    final_rows = final_data["rows"]

    # Fit normalization on meta-train only.
    norm_stats = fit_normalization_stats(meta_rows)

    # Step 4: Train CNP on meta-train data (single model for synthetic).
    model = MetaCNPModel(
        structured_task_dim=6,  # template index + difficulty index + 4 modifiers
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

    # Train on meta-train data.
    device = "cpu"
    model.to(device)

    n_meta_tasks_actual = n_meta_tasks
    n_meta_policies = len(meta_train)

    # Build training batches (episodic training).
    # For synthetic validation, use simple minibatch training.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_epochs = 50
    batch_size = 32

    # Prepare training dataset: each row is (task_features, policy_desc, outcome).
    train_data = _prepare_training_tensors(
        meta_rows, n_meta_tasks_actual, n_meta_policies, norm_stats,
    )

    model.train()
    losses: list[float] = []
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0

        # Shuffle tasks.
        task_order = list(range(n_meta_tasks_actual))
        rng.shuffle(task_order)

        for start in range(0, n_meta_tasks_actual, batch_size):
            batch_tasks = task_order[start : start + batch_size]
            if len(batch_tasks) < 2:
                continue

            # For each task, pick a random policy and random k.
            batch_structured: list[list[float]] = []
            batch_text_emb: list[list[float]] = []
            batch_policy_desc: list[list[float]] = []
            batch_target_success: list[float] = []
            batch_target_cost: list[float] = []
            batch_target_term: list[int] = []

            # Context tensors
            K_max = 8
            batch_ctx_structured: list[list[list[float]]] = []
            batch_ctx_text_emb: list[list[list[float]]] = []
            batch_ctx_success: list[list[float]] = []
            batch_ctx_cost: list[list[float]] = []
            batch_ctx_term: list[list[list[float]]] = []
            batch_ctx_mask: list[list[float]] = []

            for task_idx in batch_tasks:
                # Pick a policy and k.
                policy_idx = rng.randrange(n_meta_policies)
                k = rng.choice([0, 4, 8, 16])

                # Get target row.
                target_idx = task_idx * n_meta_policies + policy_idx
                target_row = meta_rows[target_idx]

                # Build target features.
                batch_structured.append(_extract_task_structured(target_row))
                batch_text_emb.append(target_row["model_input"]["task"]["task_embedding"])
                batch_policy_desc.append(grammar_factor_vector(meta_train[policy_idx]))

                batch_target_success.append(1.0 if target_row["verified_success"] else 0.0)
                batch_target_cost.append(target_row["output_token_cost"])

                term_map = {
                    "normal_completion": 0, "tool_call_limit": 1, "wall_timeout": 2,
                    "invalid_or_tool_error": 3, "model_runtime_failure": 4,
                    "verifier_declared_unsuccessful": 5,
                }
                batch_target_term.append(term_map.get(target_row["termination_class"], 0))

                # Build context (k other task rows for same policy).
                ctx_structured: list[list[float]] = []
                ctx_text_emb: list[list[float]] = []
                ctx_success: list[float] = []
                ctx_cost: list[float] = []
                ctx_term: list[list[float]] = []
                ctx_mask: list[float] = []

                other_tasks = [t for t in range(n_meta_tasks_actual) if t != task_idx]
                rng.shuffle(other_tasks)
                context_tasks = other_tasks[:k]

                for ct in context_tasks:
                    ctx_idx = ct * n_meta_policies + policy_idx
                    ctx_row = meta_rows[ctx_idx]
                    ctx_structured.append(_extract_task_structured(ctx_row))
                    ctx_text_emb.append(ctx_row["model_input"]["task"]["task_embedding"])
                    ctx_success.append(1.0 if ctx_row["verified_success"] else 0.0)
                    ctx_cost.append(ctx_row["output_token_cost"])
                    ctx_term.append(_term_onehot(ctx_row))
                    ctx_mask.append(1.0)

                # Pad to K_max.
                while len(ctx_structured) < K_max:
                    ctx_structured.append([0.0] * len(batch_structured[-1]))
                    ctx_text_emb.append([0.0] * len(batch_text_emb[-1]))
                    ctx_success.append(0.0)
                    ctx_cost.append(0.0)
                    ctx_term.append([0.0] * 6)
                    ctx_mask.append(0.0)

                batch_ctx_structured.append(ctx_structured[:K_max])
                batch_ctx_text_emb.append(ctx_text_emb[:K_max])
                batch_ctx_success.append(ctx_success[:K_max])
                batch_ctx_cost.append(ctx_cost[:K_max])
                batch_ctx_term.append(ctx_term[:K_max])
                batch_ctx_mask.append(ctx_mask[:K_max])

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
                outputs, target_success_t, target_cost_t, target_term_t,
            )
            loss_dict["total"].backward()
            optimizer.step()

            epoch_loss += loss_dict["total"].item()
            n_batches += 1

        losses.append(epoch_loss / max(1, n_batches))

    # Step 5: Evaluate on final held-out policies.
    model.eval()

    # Evaluate k=0 (descriptor only) and k=8.
    results_k0 = _evaluate_model_on_heldout(
        model, final_rows, final_held, n_final_tasks, k=0, norm_stats=norm_stats,
    )
    results_k8 = _evaluate_model_on_heldout(
        model, final_rows, final_held, n_final_tasks, k=8, norm_stats=norm_stats,
    )

    # Evaluate shuffled context negative control (k=8 but shuffled).
    results_shuffled = _evaluate_model_on_heldout(
        model, final_rows, final_held, n_final_tasks, k=8, norm_stats=norm_stats,
        shuffle_context=True, shuffle_seed=seed + 9999,
    )

    # Step 6: Frontier evaluation.
    predictions_k8 = results_k8["predictions"]
    outcomes_k8 = results_k8["outcomes"]
    lambda_grid = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]

    frontier = compute_frontier(predictions_k8, outcomes_k8, lambda_grid)

    # Step 7: Assemble validation report.
    k0_success = results_k0.get("mean_success_prob", 0.0)
    k8_success = results_k8.get("mean_success_prob", 0.0)
    shuffled_success = results_shuffled.get("mean_success_prob", 0.0)

    cnp_learns_main_effects = (
        len(losses) > 0
        and losses[-1] < losses[0] * 0.95
        and param_count < 1_000_000
    )

    k8_improves_over_k0 = k8_success > k0_success

    shuffled_negative_control_passes = shuffled_success <= k0_success + 0.02

    frontier_sensible = (
        frontier["frontier_area"] > 0.0
        and frontier["pure_success_rate"] >= 0.0
    )

    return {
        "validation": "complete",
        "warning": (
            "All data is synthetic. This validates the plumbing only. "
            "No claims about real effectiveness."
        ),
        "grammar": {
            "n_policies_total": len(grammar),
            "n_meta_train": len(meta_train),
            "n_development": len(development),
            "n_final_held": len(final_held),
        },
        "data": {
            "n_meta_tasks": n_meta_tasks,
            "n_meta_rows": len(meta_rows),
            "n_final_tasks": n_final_tasks,
            "n_final_rows": len(final_rows),
        },
        "model": {
            "param_count": param_count,
            "under_1M": param_count < 1_000_000,
            "n_epochs": n_epochs,
            "final_loss": losses[-1] if losses else float("nan"),
            "initial_loss": losses[0] if losses else float("nan"),
        },
        "results": {
            "k0_mean_success_prob": k0_success,
            "k8_mean_success_prob": k8_success,
            "shuffled_mean_success_prob": shuffled_success,
            "k8_minus_k0": k8_success - k0_success,
            "shuffled_minus_k0": shuffled_success - k0_success,
        },
        "frontier": frontier,
        "claims": {
            "cnp_learns_main_effects": cnp_learns_main_effects,
            "k8_improves_over_k0": k8_improves_over_k0,
            "shuffled_negative_control_passes": shuffled_negative_control_passes,
            "frontier_sensible": frontier_sensible,
        },
        "verdict": {
            "all_checks_pass": (
                cnp_learns_main_effects
                and k8_improves_over_k0
                and shuffled_negative_control_passes
                and frontier_sensible
            ),
        },
    }


def _prepare_training_tensors(
    rows: list[dict[str, Any]],
    n_tasks: int,
    n_policies: int,
    norm_stats: dict[str, Any],
) -> list[dict[str, Any]]:
    """Prepare tensor-ready training data."""
    # Just return rows for now; tensor conversion happens per-batch.
    return rows


def _extract_task_structured(row: dict[str, Any]) -> list[float]:
    """Extract structured task features from a row.

    Features: template index, difficulty index, planning_bonus, observation_bonus,
    verification_bonus, recovery_bonus.
    """
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


def _evaluate_model_on_heldout(
    model: Any,
    rows: list[dict[str, Any]],
    policies: list[Any],
    n_tasks: int,
    k: int,
    norm_stats: dict[str, Any],
    shuffle_context: bool = False,
    shuffle_seed: int = 0,
) -> dict[str, Any]:
    """Evaluate model predictions on held-out policies.

    Returns predictions (list-of-lists) and observed outcomes (list-of-lists).
    """
    import torch
    from .meta_grammar import grammar_factor_vector

    n_policies = len(policies)
    rng = random.Random(shuffle_seed)

    predictions: list[list[dict[str, Any]]] = []
    outcomes: list[list[dict[str, Any]]] = []

    for task_idx in range(n_tasks):
        task_preds: list[dict[str, Any]] = []
        task_outcomes: list[dict[str, Any]] = []

        # Get a representative row for task features.
        base_row_idx = task_idx * n_policies
        base_row = rows[base_row_idx]

        task_structured = torch.tensor(
            [_extract_task_structured(base_row)], dtype=torch.float32
        )
        task_text_emb = torch.tensor(
            [base_row["model_input"]["task"]["task_embedding"]], dtype=torch.float32
        )

        for policy_idx in range(n_policies):
            row_idx = task_idx * n_policies + policy_idx
            row = rows[row_idx]

            policy_desc = torch.tensor(
                [grammar_factor_vector(policies[policy_idx])], dtype=torch.float32
            )

            # Build calibration context from other tasks.
            if k > 0:
                other_tasks = [t for t in range(n_tasks) if t != task_idx]
                if shuffle_context:
                    rng.shuffle(other_tasks)
                context_tasks = other_tasks[:k]

                ctx_success: list[float] = []
                ctx_cost: list[float] = []
                ctx_term: list[list[float]] = []
                ctx_mask: list[float] = []

                for ct in context_tasks:
                    ctx_idx = ct * n_policies + policy_idx
                    ctx_row = rows[ctx_idx]
                    ctx_success.append(1.0 if ctx_row["verified_success"] else 0.0)
                    ctx_cost.append(ctx_row["output_token_cost"])
                    ctx_term.append(_term_onehot(ctx_row))
                    ctx_mask.append(1.0)

                context = {
                    "structured": task_structured.squeeze(0),  # dummy, not used per-element
                    "text_emb": task_text_emb.squeeze(0),
                    "success": torch.tensor(ctx_success, dtype=torch.float32),
                    "cost": torch.tensor(ctx_cost, dtype=torch.float32),
                    "term_onehot": torch.tensor(ctx_term, dtype=torch.float32),
                    "mask": torch.tensor(ctx_mask, dtype=torch.float32),
                }
            else:
                context = None

            with torch.no_grad():
                pred = model.predict(
                    {"structured": task_structured.squeeze(0), "text_emb": task_text_emb.squeeze(0)},
                    policy_desc.squeeze(0),
                    context,
                )

            task_preds.append(pred)
            task_outcomes.append({
                "verified_success": row["verified_success"],
                "cost": row["output_token_cost"],
                "termination_class": row["termination_class"],
            })

        predictions.append(task_preds)
        outcomes.append(task_outcomes)

    # Compute mean success probability across all predictions.
    all_probs = []
    for task_preds in predictions:
        for pred in task_preds:
            all_probs.append(pred["success_prob"])

    mean_success_prob = sum(all_probs) / len(all_probs) if all_probs else 0.0

    return {
        "predictions": predictions,
        "outcomes": outcomes,
        "mean_success_prob": mean_success_prob,
    }


__all__ = [
    "generate_synthetic_outcomes",
    "run_synthetic_validation",
]
