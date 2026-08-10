"""Attentive Conditional Neural Process model for meta-policy learning.

Architecture (from M3 preregistration Section 8.1):

    Task encoder:
        structured features (template, difficulty, etc.) through small MLP
        -> h_x (96-128 dim)

    Policy descriptor encoder:
        one-hot grammar factors + numeric tool_cap
        -> 2-layer MLP -> h_p (64 dim)

    Context element encoder (per calibration row):
        phi(h_x_i, h_p, y_i, onehot(term_i), zlogcost_i, mask_i) -> e_i
        2-layer MLP (256->128)

    Global DeepSets summary:
        r_global = rho(h_p, mean(e_i), variance(e_i), outcome moments,
                       log(1+k))

    Target-conditioned attention:
        r_local = Attention(q=h_x_target, k=h_x_i, v=e_i)

    Decoder:
        input: [h_x_target, h_p, r_global, r_local, h_x_target * h_p]
        -> MLP (256->128) with dropout
        -> three heads:
            success:     bounded context residual around a descriptor prior
            cost:        log-normal (mu_c, log_sigma_c)
            termination: categorical logits (6 classes, auxiliary)

PyTorch conventions follow :mod:`pyreplab_harness.outcome_model` for device
handling, train/eval mode, and seed management.
"""

from __future__ import annotations

import math
import random
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - only exercised on torch-less hosts
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Stub base class for when torch is absent (follows outcome_model.py pattern)
# ---------------------------------------------------------------------------
if nn is None:

    class _TorchStubModule:
        """Base class placeholder when torch is absent; instantiation raises."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "PyTorch is required for the meta-CNP model but is not installed."
            )

    _Module = _TorchStubModule
else:
    _Module = nn.Module

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NUM_TERMINATION_CLASSES = 6  # coarse termination classes from spec
_EPS = 1e-7

# Task feature dimensions (use small random-projection for synthetic text emb).
_DEFAULT_TASK_EMBED_DIM = 32
_DEFAULT_STRUCTURED_TASK_DIM = 4  # template_id, difficulty, family, interaction-type

# Grammar factor dimensions
_NUM_GRAMMAR_ONE_HOT = 12  # 3+3+2+2+2
_NUM_GRAMMAR_NUMERIC = 1   # tool_call_limit (normalized)
_POLICY_DESC_DIM = 13      # total flat policy descriptor

# Model dimensions
_DEFAULT_HX_DIM = 96
_DEFAULT_HP_DIM = 64
_DEFAULT_EI_DIM = 128
_DEFAULT_CONTEXT_HIDDEN = 256
_DEFAULT_DECODER_HIDDEN = 128
_DEFAULT_DROPOUT = 0.15
_DEFAULT_NUM_HEADS = 2


def _require_torch():
    if torch is None:
        raise RuntimeError(
            "This operation requires PyTorch, which is not installed."
        )
    return torch


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------


class TaskEncoder(_Module):
    """Encode task features into a fixed-dimension representation h_x.

    Inputs: structured task features (template, difficulty as one-hot/numeric)
    plus an optional text embedding vector (frozen/projected).
    """

    def __init__(
        self,
        structured_dim: int = _DEFAULT_STRUCTURED_TASK_DIM,
        text_embed_dim: int = _DEFAULT_TASK_EMBED_DIM,
        output_dim: int = _DEFAULT_HX_DIM,
    ) -> None:
        super().__init__()
        input_dim = structured_dim + text_embed_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, structured: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """Encode task features -> h_x.

        Args:
            structured: (B, structured_dim) tensor of one-hot/numeric task features
            text_emb: (B, text_embed_dim) tensor of text embeddings
        Returns:
            h_x: (B, output_dim)
        """
        x = torch.cat([structured, text_emb], dim=-1)
        return self.net(x)


class PolicyDescriptorEncoder(_Module):
    """Encode grammar factors into policy descriptor h_p.

    Input: one-hot grammar factors (12 dim) + numeric tool_cap (1 dim).
    Output: h_p (64 dim). No policy identity, version, bundle_id, or hash.
    """

    def __init__(
        self,
        input_dim: int = _POLICY_DESC_DIM,
        hidden_dim: int = 48,
        output_dim: int = _DEFAULT_HP_DIM,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, policy_desc: torch.Tensor) -> torch.Tensor:
        """Encode policy descriptor -> h_p.

        Args:
            policy_desc: (B, 13) tensor (12 one-hot + 1 numeric)
        Returns:
            h_p: (B, output_dim)
        """
        return self.net(policy_desc)


class ContextElementEncoder(_Module):
    """Encode one calibration row into context element e_i.

    phi(h_x_i, h_p, y_i, onehot(term_i), zlogcost_i, mask_i) -> e_i
    """

    def __init__(
        self,
        hx_dim: int = _DEFAULT_HX_DIM,
        hp_dim: int = _DEFAULT_HP_DIM,
        output_dim: int = _DEFAULT_EI_DIM,
        hidden_dim: int = _DEFAULT_CONTEXT_HIDDEN,
    ) -> None:
        super().__init__()
        # Input: h_x_i (96) + h_p (64) + success (1) + termination one-hot (6) +
        #        standardized log1p(cost) (1) + mask (1) = 169
        input_dim = hx_dim + hp_dim + 1 + _NUM_TERMINATION_CLASSES + 1 + 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        hx: torch.Tensor,
        hp: torch.Tensor,
        success: torch.Tensor,
        termination_onehot: torch.Tensor,
        log_cost: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode calibration row -> e_i.

        Args:
            hx: (B, hx_dim) task features
            hp: (B, hp_dim) policy descriptor (same for context rows of same policy)
            success: (B, 1) binary success
            termination_onehot: (B, 6) one-hot termination class
            log_cost: (B, 1) meta-train-standardized log1p(cost)
            mask: (B, 1) validity mask (1 = valid, 0 = padding)
        Returns:
            e_i: (B, output_dim)
        """
        x = torch.cat([hx, hp, success, termination_onehot, log_cost, mask], dim=-1)
        e = self.net(x)
        # Zero out padding elements so they don't affect aggregation.
        return e * mask


class MultiheadAttention(_Module):
    """Simple multi-head attention for target-conditioned local summary."""

    def __init__(
        self,
        q_dim: int = _DEFAULT_HX_DIM,
        k_dim: int = _DEFAULT_HX_DIM,
        v_dim: int = _DEFAULT_EI_DIM,
        num_heads: int = _DEFAULT_NUM_HEADS,
        output_dim: int = _DEFAULT_EI_DIM,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(q_dim, output_dim)
        self.k_proj = nn.Linear(k_dim, output_dim)
        self.v_proj = nn.Linear(v_dim, output_dim)
        self.out_proj = nn.Linear(output_dim, output_dim)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Target-conditioned attention.

        Args:
            query: (B, q_dim) target task features
            keys: (B, K, k_dim) context task features
            values: (B, K, v_dim) context elements
            mask: (B, K, 1) optional mask (1=valid, 0=padding)
        Returns:
            r_local: (B, output_dim)
        """
        B, K, _ = keys.shape

        q = self.q_proj(query).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 1, D)
        k = self.k_proj(keys).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)    # (B, H, K, D)
        v = self.v_proj(values).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, K, D)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, 1, K)
        has_valid_context = None
        if mask is not None:
            # mask: (B, K, 1) -> (B, 1, 1, K)
            mask = mask.squeeze(-1).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, K)
            has_valid_context = mask.any(dim=-1, keepdim=True)
            attn = attn.masked_fill(mask == 0, float("-inf"))
            # Softmax over an all-masked row is undefined. Empty rows are
            # replaced with learned vectors by MetaCNPModel.forward().
            attn = torch.where(has_valid_context, attn, torch.zeros_like(attn))
        attn = F.softmax(attn, dim=-1)
        if has_valid_context is not None:
            attn = attn * has_valid_context
        out = attn @ v  # (B, H, 1, D)
        out = out.transpose(1, 2).contiguous().view(B, -1)  # (B, output_dim)
        return self.out_proj(out)


class Decoder(_Module):
    """Decoder MLP with three heads: success, cost, termination."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = _DEFAULT_DECODER_HIDDEN,
        dropout: float = _DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.success_head = nn.Linear(hidden_dim, 1)  # Bernoulli logit
        self.cost_head = nn.Linear(hidden_dim, 2)      # mu_log, log_sigma_log
        self.term_head = nn.Linear(hidden_dim, _NUM_TERMINATION_CLASSES)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decode into three head outputs.

        Args:
            x: (B, input_dim) concatenated decoder input
        Returns:
            dict with "logit_success", "cost_params", "term_logits"
        """
        h = self.shared(x)
        return {
            "logit_success": self.success_head(h).squeeze(-1),  # (B,)
            "cost_params": self.cost_head(h),                    # (B, 2)
            "term_logits": self.term_head(h),                    # (B, 6)
        }


# ---------------------------------------------------------------------------
# Main CNP model
# ---------------------------------------------------------------------------


class MetaCNPModel(_Module):
    """Attentive Conditional Neural Process for meta-policy learning.

    Predicts success, cost, and termination for a target (task, policy) pair
    given a calibration context of k observed outcomes for that policy.

    Usage::

        model = MetaCNPModel()
        output = model.predict(target_config, policy_descriptor, context)
        # or batch:
        panel_output = model.predict_panel(tasks, policies, contexts)
    """

    def __init__(
        self,
        structured_task_dim: int = _DEFAULT_STRUCTURED_TASK_DIM,
        text_embed_dim: int = _DEFAULT_TASK_EMBED_DIM,
        hx_dim: int = _DEFAULT_HX_DIM,
        hp_dim: int = _DEFAULT_HP_DIM,
        ei_dim: int = _DEFAULT_EI_DIM,
        context_hidden: int = _DEFAULT_CONTEXT_HIDDEN,
        decoder_hidden: int = _DEFAULT_DECODER_HIDDEN,
        dropout: float = _DEFAULT_DROPOUT,
        num_heads: int = _DEFAULT_NUM_HEADS,
    ) -> None:
        super().__init__()
        self.hx_dim = hx_dim
        self.hp_dim = hp_dim
        self.ei_dim = ei_dim

        self.task_encoder = TaskEncoder(
            structured_dim=structured_task_dim,
            text_embed_dim=text_embed_dim,
            output_dim=hx_dim,
        )
        self.policy_encoder = PolicyDescriptorEncoder(
            input_dim=_POLICY_DESC_DIM,
            output_dim=hp_dim,
        )
        self.context_encoder = ContextElementEncoder(
            hx_dim=hx_dim,
            hp_dim=hp_dim,
            output_dim=ei_dim,
            hidden_dim=context_hidden,
        )
        self.attention = MultiheadAttention(
            q_dim=hx_dim,
            k_dim=hx_dim,
            v_dim=ei_dim,
            num_heads=num_heads,
            output_dim=ei_dim,
        )

        # r_global input: policy descriptor, learned element moments, explicit
        # outcome moments, and log(1+k). The explicit moments make the global
        # random-intercept signal available without requiring the element MLP
        # to rediscover simple sufficient statistics.
        r_global_input_dim = (
            hp_dim + ei_dim + ei_dim + 1 + 1 + _NUM_TERMINATION_CLASSES + 1
        )
        self.r_global_net = nn.Sequential(
            nn.Linear(r_global_input_dim, context_hidden),
            nn.ReLU(),
            nn.Linear(context_hidden, ei_dim),
        )

        # Learned empty-context vectors for k=0
        self.empty_r_global = nn.Parameter(torch.zeros(ei_dim))
        self.empty_r_local = nn.Parameter(torch.zeros(ei_dim))

        # Project the policy representation to the task latent dimension before
        # forming the element-wise interaction.
        self.hp_to_hx_proj = nn.Linear(hp_dim, hx_dim, bias=False)

        # Descriptor-only prior for success. Non-empty contexts update this
        # prior through a shrinkage path in forward(); the attentive decoder
        # contributes only a bounded task-specific residual.
        success_prior_input_dim = hx_dim + hp_dim + hx_dim
        self.success_prior_net = nn.Sequential(
            nn.Linear(success_prior_input_dim, decoder_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden, 1),
        )
        self.calibration_strength_logit = nn.Parameter(torch.tensor(1.3862944))

        # Decoder input: h_x_target + h_p + r_global + r_local + h_x_target * h_p
        decoder_input_dim = hx_dim + hp_dim + ei_dim + ei_dim + hx_dim
        self.decoder = Decoder(
            input_dim=decoder_input_dim,
            hidden_dim=decoder_hidden,
            dropout=dropout,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if "empty_r" in name:
                nn.init.normal_(param, std=0.01)
            elif "weight" in name and param.ndim >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def _encode_context(
        self,
        context_hx: torch.Tensor,
        hp: torch.Tensor,
        context_success: torch.Tensor,
        context_term_onehot: torch.Tensor,
        context_log_cost: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode all calibration rows into e_i elements.

        Args:
            context_hx: (B, K, hx_dim)
            hp: (B, hp_dim) - expanded or broadcast
            context_success: (B, K, 1)
            context_term_onehot: (B, K, 6)
            context_log_cost: (B, K, 1)
            context_mask: (B, K, 1)
        Returns:
            e_i: (B, K, ei_dim)
        """
        B, K, _ = context_hx.shape
        # Expand hp to match (B, K, hp_dim)
        hp_expanded = hp.unsqueeze(1).expand(-1, K, -1)

        # Flatten to (B*K, *) for element encoder
        hx_flat = context_hx.reshape(B * K, -1)
        hp_flat = hp_expanded.reshape(B * K, -1)
        success_flat = context_success.reshape(B * K, 1)
        term_flat = context_term_onehot.reshape(B * K, _NUM_TERMINATION_CLASSES)
        cost_flat = context_log_cost.reshape(B * K, 1)
        mask_flat = context_mask.reshape(B * K, 1)

        e_flat = self.context_encoder(
            hx_flat, hp_flat, success_flat, term_flat, cost_flat, mask_flat
        )
        return e_flat.reshape(B, K, self.ei_dim)

    def forward(
        self,
        target_structured: torch.Tensor,
        target_text_emb: torch.Tensor,
        policy_desc: torch.Tensor,
        context_structured: torch.Tensor,
        context_text_emb: torch.Tensor,
        context_success: torch.Tensor,
        context_term_onehot: torch.Tensor,
        context_cost: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with variable-k calibration context.

        Args:
            target_structured: (B, structured_dim)
            target_text_emb: (B, text_embed_dim)
            policy_desc: (B, 13) grammar factor vector
            context_structured: (B, K, structured_dim)
            context_text_emb: (B, K, text_embed_dim)
            context_success: (B, K, 1) binary success
            context_term_onehot: (B, K, 6) termination one-hot
        context_cost: (B, K, 1) meta-train-standardized log1p cost values
            context_mask: (B, K, 1) validity mask

        Returns:
            dict with "logit_success", "cost_params", "term_logits"
        """
        B = target_structured.shape[0]
        K = context_structured.shape[1]

        # Encode target task.
        hx_target = self.task_encoder(target_structured, target_text_emb)  # (B, hx_dim)

        # Encode policy descriptor.
        hp = self.policy_encoder(policy_desc)  # (B, hp_dim)

        # Encode context tasks.
        context_hx = self.task_encoder(
            context_structured.reshape(B * K, -1),
            context_text_emb.reshape(B * K, -1),
        ).reshape(B, K, self.hx_dim)

        # Encode context elements.
        e_i = self._encode_context(
            context_hx, hp,
            context_success, context_term_onehot,
            context_cost, context_mask,
        )  # (B, K, ei_dim)

        # Global DeepSets summary.
        # Mean of valid elements.
        e_sum = (e_i * context_mask).sum(dim=1)  # (B, ei_dim)
        valid_count_raw = context_mask.sum(dim=1)  # (B, 1) -- unclamped
        valid_count_safe = valid_count_raw.clamp(min=1.0)  # (B, 1)
        e_mean = e_sum / valid_count_safe  # (B, ei_dim)

        # Variance of valid elements.
        e_var = ((e_i - e_mean.unsqueeze(1)) ** 2 * context_mask).sum(dim=1) / valid_count_safe  # (B, ei_dim)

        # log(1+k) feature -- use actual k (unclamped).
        k_val = valid_count_raw.squeeze(-1)  # (B,)
        log1pk = torch.log1p(k_val).unsqueeze(-1)  # (B, 1)

        context_success_mean = (
            context_success * context_mask
        ).sum(dim=1) / valid_count_safe
        context_cost_mean = (
            context_cost * context_mask
        ).sum(dim=1) / valid_count_safe
        context_term_mean = (
            context_term_onehot * context_mask
        ).sum(dim=1) / valid_count_safe

        r_global = self.r_global_net(
            torch.cat([
                hp,
                e_mean,
                e_var,
                context_success_mean,
                context_cost_mean,
                context_term_mean,
                log1pk,
            ], dim=-1)
        )  # (B, ei_dim)

        # Target-conditioned attention for local summary.
        r_local = self.attention(
            query=hx_target,
            keys=context_hx,
            values=e_i,
            mask=context_mask,
        )  # (B, ei_dim)

        # For k=0 (all context_mask zero), use learned empty-context vectors.
        is_empty = (k_val < 1.0)  # (B,) -- unclamped check
        if is_empty.any():
            empty_mask = is_empty.view(-1, 1)
            r_global = torch.where(empty_mask, self.empty_r_global.unsqueeze(0).expand(B, -1), r_global)
            r_local = torch.where(empty_mask, self.empty_r_local.unsqueeze(0).expand(B, -1), r_local)

        # Decoder input: project hp to hx_dim for element-wise interaction.
        interaction = hx_target * self.hp_to_hx_proj(hp)  # (B, hx_dim)
        decoder_input = torch.cat([hx_target, hp, r_global, r_local, interaction], dim=-1)

        outputs = self.decoder(decoder_input)

        prior_logit = self.success_prior_net(
            torch.cat([hx_target, hp, interaction], dim=-1)
        ).squeeze(-1)
        success_count = (context_success * context_mask).sum(dim=1).squeeze(-1)
        prior_count = 2.0
        smoothed_success = (
            success_count + prior_count
        ) / (k_val + 2.0 * prior_count)
        context_evidence_logit = torch.logit(
            smoothed_success.clamp(min=_EPS, max=1.0 - _EPS)
        )
        shrinkage = k_val / (k_val + 4.0)
        calibration_strength = torch.sigmoid(self.calibration_strength_logit)
        attentive_residual = 0.1 * torch.tanh(outputs["logit_success"])
        has_context = (k_val > 0.0).to(prior_logit.dtype)
        outputs["logit_success"] = prior_logit + has_context * (
            calibration_strength
            * shrinkage
            * (context_evidence_logit - prior_logit)
            + attentive_residual
        )
        return outputs

    def predict(
        self,
        target_task_features: dict[str, torch.Tensor],
        policy_descriptor: torch.Tensor,
        calibration_context: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        """Predict outcomes for a single target task-policy pair.

        Args:
            target_task_features: dict with keys "structured" (structured_dim,),
                "text_emb" (text_embed_dim,)
            policy_descriptor: (13,) grammar factor vector
            calibration_context: dict with keys "structured", "text_emb",
                "success", "term_onehot", "cost", "mask", each of shape
                (K, ...), or None for k=0.

        Returns:
            dict with "success_prob", "cost_mean", "cost_std", "termination_probs"
        """
        if self.training:
            raise RuntimeError("predict() should be called in eval mode")

        target_structured = target_task_features["structured"].unsqueeze(0)  # (1, D)
        target_text_emb = target_task_features["text_emb"].unsqueeze(0)       # (1, D)
        policy_desc = policy_descriptor.unsqueeze(0)                           # (1, 13)

        if calibration_context is None:
            # k=0: empty context
            K = 1
            device = target_structured.device
            context_structured = torch.zeros(1, K, target_structured.shape[-1], device=device)
            context_text_emb = torch.zeros(1, K, target_text_emb.shape[-1], device=device)
            context_success = torch.zeros(1, K, 1, device=device)
            context_term = torch.zeros(1, K, _NUM_TERMINATION_CLASSES, device=device)
            context_cost = torch.zeros(1, K, 1, device=device)
            context_mask = torch.zeros(1, K, 1, device=device)
        else:
            K = calibration_context["success"].shape[0]
            context_structured = calibration_context["structured"].unsqueeze(0)  # (K,D) -> (1,K,D)
            context_text_emb = calibration_context["text_emb"].unsqueeze(0)       # (K,D) -> (1,K,D)
            context_term = calibration_context["term_onehot"].unsqueeze(0)        # (K,6) -> (1,K,6)

            # Normalize scalar fields: accept (K,) or (K,1), produce (1,K,1).
            def _to_batch_k_1(name: str, t: torch.Tensor) -> torch.Tensor:
                if t.ndim == 1:
                    normalized = t.unsqueeze(-1)
                elif t.ndim == 2 and t.shape[1] == 1:
                    normalized = t
                else:
                    raise ValueError(
                        f"calibration_context[{name!r}] must have shape (K,) "
                        f"or (K, 1); got {tuple(t.shape)}"
                    )
                if normalized.shape[0] != K:
                    raise ValueError(
                        f"calibration_context[{name!r}] has K={normalized.shape[0]}, "
                        f"expected {K}"
                    )
                return normalized.unsqueeze(0)

            context_success = _to_batch_k_1("success", calibration_context["success"])
            context_cost = _to_batch_k_1("cost", calibration_context["cost"])
            context_mask = _to_batch_k_1("mask", calibration_context["mask"])

        with torch.no_grad():
            outputs = self.forward(
                target_structured, target_text_emb, policy_desc,
                context_structured, context_text_emb,
                context_success, context_term, context_cost, context_mask,
            )

        success_prob = torch.sigmoid(outputs["logit_success"]).squeeze(0).item()
        mu_log = outputs["cost_params"][0, 0].item()
        log_sigma = outputs["cost_params"][0, 1].item()
        sigma_log = F.softplus(torch.tensor(log_sigma)).item()

        # Cost expectation: E[C] = exp(mu + sigma^2/2) - 1
        sigma_sq = min(sigma_log ** 2, 50.0)
        shifted_cost_mean = math.exp(min(mu_log + sigma_sq / 2.0, 50.0))
        cost_mean = max(0.0, shifted_cost_mean - 1.0)
        if sigma_sq == 0.0:
            cost_std = 0.0
        else:
            log_cost_variance = (
                math.log(math.expm1(sigma_sq)) + 2.0 * mu_log + sigma_sq
            )
            cost_std = math.exp(min(log_cost_variance, 50.0) / 2.0)

        term_probs = F.softmax(outputs["term_logits"], dim=-1).squeeze(0).tolist()

        return {
            "success_prob": success_prob,
            "cost_mean": cost_mean,
            "cost_std": cost_std,
            "termination_probs": term_probs,
            "raw_logit_success": float(outputs["logit_success"].squeeze(0).item()),
            "raw_cost_mu_log": mu_log,
            "raw_cost_log_sigma": log_sigma,
        }

    def predict_panel(
        self,
        tasks: list[dict[str, torch.Tensor]],
        policies: list[torch.Tensor],
        calibration_contexts: list[dict[str, torch.Tensor] | None],
    ) -> list[list[dict[str, Any]]]:
        """Batch predict for all task-policy combinations.

        Args:
            tasks: list of task feature dicts (length N_tasks)
            policies: list of policy descriptor tensors (length N_policies)
            calibration_contexts: list of calibration contexts per policy
                (length N_policies)

        Returns:
            predictions[i][j] for task i, policy j
        """
        if self.training:
            raise RuntimeError("predict_panel() should be called in eval mode")

        panel: list[list[dict[str, Any]]] = []
        for task_features in tasks:
            task_preds: list[dict[str, Any]] = []
            for j, policy_desc in enumerate(policies):
                context = calibration_contexts[j] if j < len(calibration_contexts) else None
                pred = self.predict(task_features, policy_desc, context)
                task_preds.append(pred)
            panel.append(task_preds)
        return panel


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


def compute_loss(
    outputs: dict[str, torch.Tensor],
    target_success: torch.Tensor,
    target_cost: torch.Tensor,
    target_term: torch.Tensor,
    *,
    cost_loss_weight: float = 1.0,
    term_loss_weight: float = 0.2,
) -> dict[str, torch.Tensor]:
    """Compute episodic loss.

    Loss = Bernoulli NLL(success) + log-normal NLL(cost) + weight * CE(term).

    Args:
        outputs: dict from model.forward()
        target_success: (B,) binary success
        target_cost: (B,) raw cost values (>= 0)
        target_term: (B,) termination class indices (long)
        cost_loss_weight: weight for the cost likelihood
        term_loss_weight: weight for termination auxiliary loss

    Returns:
        dict with "total", "success", "cost", "term"
    """
    if not math.isfinite(cost_loss_weight) or cost_loss_weight < 0.0:
        raise ValueError("cost_loss_weight must be finite and nonnegative")
    if not math.isfinite(term_loss_weight) or term_loss_weight < 0.0:
        raise ValueError("term_loss_weight must be finite and nonnegative")

    # Bernoulli NLL for success.
    success_loss = F.binary_cross_entropy_with_logits(
        outputs["logit_success"], target_success
    )

    # Log-normal NLL for cost: log(1+C) ~ N(mu, sigma^2).
    log_cost = torch.log1p(target_cost.clamp(min=0.0))
    mu_log = outputs["cost_params"][:, 0]
    raw_scale = outputs["cost_params"][:, 1]  # unconstrained log-sigma
    sigma = F.softplus(raw_scale) + _EPS
    cost_nll = (
        0.5 * math.log(2.0 * math.pi)
        + torch.log(sigma)
        + 0.5 * ((log_cost - mu_log) / sigma) ** 2
    ).mean()

    # Categorical CE for termination.
    term_loss = F.cross_entropy(outputs["term_logits"], target_term.long())

    total = (
        success_loss
        + cost_loss_weight * cost_nll
        + term_loss_weight * term_loss
    )

    return {
        "total": total,
        "success": success_loss,
        "cost": cost_nll,
        "term": term_loss,
    }


def episode_sample(
    policy_indices: list[int],
    task_pool: list[dict[str, Any]],
    policy_task_map: dict[int, list[int]],
    k: int,
    rng: random.Random,
) -> tuple[int, list[int], list[int]]:
    """Sample one episodic training instance.

    Returns (policy_idx, calibration_task_indices, target_task_indices).
    Calibration and target tasks are disjoint.
    """
    policy_idx = rng.choice(policy_indices)
    policy_tasks = policy_task_map[policy_idx]

    if len(policy_tasks) < k + 1:
        # Not enough tasks for this policy; sample with replacement.
        k_eff = min(k, len(policy_tasks))
        shuffled = rng.sample(policy_tasks, len(policy_tasks))
        cal_tasks = shuffled[:k_eff] + rng.choices(policy_tasks, k=k - k_eff)
        target_tasks = rng.choices([t for t in policy_tasks if t not in cal_tasks[:k_eff]], k=1)
    else:
        shuffled = rng.sample(policy_tasks, len(policy_tasks))
        cal_tasks = shuffled[:k]
        target_tasks = shuffled[k : k + 1]

    return policy_idx, cal_tasks, target_tasks


def set_seed(seed: int) -> None:
    """Seed Python/torch RNGs for deterministic training."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "MetaCNPModel",
    "TaskEncoder",
    "PolicyDescriptorEncoder",
    "ContextElementEncoder",
    "MultiheadAttention",
    "Decoder",
    "compute_loss",
    "episode_sample",
    "set_seed",
    "count_parameters",
    "TORCH_AVAILABLE",
    "_NUM_TERMINATION_CLASSES",
]
