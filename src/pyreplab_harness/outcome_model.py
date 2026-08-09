"""Policy-conditioned multimodal Bayesian neural success model.

This module implements the first policy-conditioned neural success model for
the pyreplab harness experiment:

    Y | x, pi ~ Bernoulli(sigmoid(g_phi(F_psi(x_text, x_num, x_cat), E(pi))))

Features come exclusively from ``row["model_input"]`` (predecision text,
family/template/difficulty, flattened numeric public metadata and the policy
identity) and labels exclusively from ``row["verified_success"]``.  Post-action
fields such as usage, message/tool counts or failure codes never enter the
feature path.

PyTorch is optional.  The tokenizer, :class:`Preprocessor` and the pure metric
functions run on the standard library alone; every neural path raises a clear
``RuntimeError`` when torch is not importable.

Architecture
------------
* trainable token embedding with masked mean over the text,
* learned embeddings for every categorical input,
* an MLP over standardized numeric values plus a missingness mask,
* a fusion MLP with dropout, and
* a genuine variational Bayesian final linear layer: trainable weight/bias
  ``mu``/``rho`` (softplus sigma), standard-normal prior, reparameterized
  samples and a closed-form KL.  The neural representation stays trainable;
  the Bayesian scope is initially the outcome head only.

CLI
---
``python -m pyreplab_harness.outcome_model train DATASET ARTIFACT_DIR ...``
``python -m pyreplab_harness.outcome_model evaluate DATASET ARTIFACT_DIR``
``python -m pyreplab_harness.outcome_model predict MODEL_INPUT.json ARTIFACT_DIR``
``python -m pyreplab_harness.outcome_model inspect ARTIFACT_DIR [--dataset DATASET]``
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from .dataset import flatten_public_metadata
from .io_utils import write_json
from .treatments import (
    TreatmentRegistry,
    TreatmentSpec,
    treatment_model_input_descriptor,
)

# ---------------------------------------------------------------------------
# Optional PyTorch import.  The rest of this module (tokenizer, Preprocessor,
# metrics) works without torch; neural paths raise a clear error instead.
# ---------------------------------------------------------------------------
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

_PAD = "<pad>"
_UNK = "<unk>"
#: Deterministic lowercase ASCII regex tokenizer: runs of letters/digits.
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_EPS = 1e-7
#: Categorical model_input fields with a train vocabulary plus UNK.
CATEGORICAL_FIELDS = ("family", "template_id", "difficulty", "policy_id", "policy_version")
TREATMENT_NUMERIC_FIELDS = (
    "max_output_tokens",
    "tool_call_limit",
    "command_timeout_seconds",
    "wall_time_limit_seconds",
)
TREATMENT_CATEGORICAL_FIELDS = (
    "tool_interface",
    "allowed_tools_signature",
    "bundle_id",
)
_SPLIT_NAMES = ("train", "validation", "test")
_INSPECT_METRIC_NAMES = (
    "log_loss",
    "brier",
    "accuracy_05",
    "ece",
    "average_precision",
    "precision",
    "recall",
    "f1",
)


def _require_torch():
    if torch is None:
        raise RuntimeError(
            "This operation requires PyTorch, which is not installed. "
            "Install a CPU or CUDA wheel of torch to train or evaluate the model."
        )
    return torch


def tokenize_text(text: Any) -> list[str]:
    """Deterministic lowercase regex tokenization of ``text``."""
    if not text:
        return []
    return _TOKEN_RE.findall(str(text).lower())


def _as_finite_float(value: Any) -> float | None:
    """Convert a JSON scalar to a finite float, or ``None`` if not usable."""
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return None


# ===========================================================================
# Preprocessor (pure stdlib, serializable, fit on the train split only)
# ===========================================================================
class Preprocessor:
    """Text/numeric/categorical featurizer fitted on the train split only.

    ``fit`` learns (in this order): the top-``max_vocab`` token vocabulary
    (``<pad>``=0, ``<unk>``=1), the sorted train numeric key vocabulary with
    per-key mean/std over *present* values, and sorted categorical
    vocabularies (``<unk>``=0).  ``transform`` encodes one ``model_input``
    dict into fixed-shape lists: padded token ids plus a mask and length,
    standardized numeric values plus a missingness mask (missing values are
    mean-imputed in standardized space), and categorical ids.  Numeric keys
    unseen during fit are ignored; unknown tokens and categories map to UNK.
    """

    def __init__(self, max_vocab: int = 5000, max_tokens: int = 256) -> None:
        max_vocab = int(max_vocab)
        max_tokens = int(max_tokens)
        if max_vocab < 0:
            raise ValueError("max_vocab must be >= 0")
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        self.max_vocab = max_vocab
        self.max_tokens = max_tokens
        self.token_to_id: dict[str, int] = {}
        self.numeric_keys: list[str] = []
        self.numeric_mean: dict[str, float] = {}
        self.numeric_std: dict[str, float] = {}
        self.cat_vocab: dict[str, dict[str, int]] = {}
        self.treatment_enabled = False
        self.treatment_numeric_keys: list[str] = []
        self.treatment_numeric_mean: dict[str, float] = {}
        self.treatment_numeric_std: dict[str, float] = {}
        self.treatment_cat_vocab: dict[str, dict[str, int]] = {}

    # -- fitting -------------------------------------------------------------
    def fit(self, model_inputs: list[dict[str, Any]]) -> "Preprocessor":
        counter: Counter[str] = Counter()
        self.treatment_enabled = any(
            isinstance(model_input.get("treatment"), dict)
            for model_input in model_inputs
        )
        for model_input in model_inputs:
            counter.update(tokenize_text(model_input.get("text", "")))
            if self.treatment_enabled:
                treatment = model_input.get("treatment")
                if isinstance(treatment, dict):
                    counter.update(tokenize_text(treatment.get("text", "")))
        # Deterministic: most frequent first, ties broken by token string.
        ordered = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[: self.max_vocab]
        self.token_to_id = {_PAD: 0, _UNK: 1}
        for token, _count in ordered:
            self.token_to_id[token] = len(self.token_to_id)

        key_values: dict[str, list[float]] = {}
        for model_input in model_inputs:
            flat = flatten_public_metadata(model_input.get("public_metadata") or {})
            for key, value in flat.items():
                value_f = _as_finite_float(value)
                if value_f is None:
                    continue
                key_values.setdefault(key, []).append(value_f)
        self.numeric_keys = sorted(key_values)
        for key in self.numeric_keys:
            values = key_values[key]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            self.numeric_mean[key] = mean
            self.numeric_std[key] = math.sqrt(variance)

        self.cat_vocab = {}
        for field in CATEGORICAL_FIELDS:
            vocab: dict[str, int] = {_UNK: 0}
            values = sorted(
                {
                    str(model_input[field])
                    for model_input in model_inputs
                    if model_input.get(field) is not None
                }
            )
            for value in values:
                vocab[value] = len(vocab)
            self.cat_vocab[field] = vocab

        self.treatment_numeric_keys = []
        self.treatment_numeric_mean = {}
        self.treatment_numeric_std = {}
        self.treatment_cat_vocab = {}
        if self.treatment_enabled:
            self.treatment_numeric_keys = list(TREATMENT_NUMERIC_FIELDS)
            treatment_values: dict[str, list[float]] = {
                key: [] for key in self.treatment_numeric_keys
            }
            for model_input in model_inputs:
                treatment = model_input.get("treatment")
                if not isinstance(treatment, dict):
                    continue
                for key in self.treatment_numeric_keys:
                    value = _as_finite_float(treatment.get(key))
                    if value is not None:
                        treatment_values[key].append(value)
            for key in self.treatment_numeric_keys:
                values = treatment_values[key]
                if not values:
                    self.treatment_numeric_mean[key] = 0.0
                    self.treatment_numeric_std[key] = 0.0
                    continue
                mean = sum(values) / len(values)
                variance = sum((value - mean) ** 2 for value in values) / len(values)
                self.treatment_numeric_mean[key] = mean
                self.treatment_numeric_std[key] = math.sqrt(variance)

            for field in TREATMENT_CATEGORICAL_FIELDS:
                vocab = {_UNK: 0}
                values = sorted(
                    {
                        str(treatment[field])
                        for model_input in model_inputs
                        if isinstance((treatment := model_input.get("treatment")), dict)
                        and treatment.get(field) is not None
                    }
                )
                for value in values:
                    vocab[value] = len(vocab)
                self.treatment_cat_vocab[field] = vocab
        return self

    # -- transforming --------------------------------------------------------
    def transform(self, model_input: dict[str, Any]) -> dict[str, Any]:
        if not self.token_to_id or not self.cat_vocab:
            raise ValueError("Preprocessor must be fit() before transform()")
        tokens = tokenize_text(model_input.get("text", ""))[: self.max_tokens]
        ids = [self.token_to_id.get(token, 1) for token in tokens]
        padding = [0] * (self.max_tokens - len(ids))
        token_ids = ids + padding
        token_mask = [1] * len(ids) + [0] * len(padding)

        flat = flatten_public_metadata(model_input.get("public_metadata") or {})
        numeric: list[float] = []
        numeric_mask: list[int] = []
        for key in self.numeric_keys:
            value = _as_finite_float(flat.get(key))
            if value is None:
                numeric.append(0.0)  # mean-imputation in standardized space
                numeric_mask.append(0)
            else:
                std = self.numeric_std.get(key, 0.0)
                numeric.append((value - self.numeric_mean.get(key, 0.0)) / std if std > 0 else 0.0)
                numeric_mask.append(1)

        out: dict[str, Any] = {
            "token_ids": token_ids,
            "token_mask": token_mask,
            "token_length": len(ids),
            "numeric": numeric,
            "numeric_mask": numeric_mask,
        }
        for field in CATEGORICAL_FIELDS:
            raw = model_input.get(field)
            out[field] = self.cat_vocab[field].get(str(raw) if raw is not None else "", 0)

        if self.treatment_enabled:
            treatment = model_input.get("treatment")
            if not isinstance(treatment, dict):
                treatment = {
                    "text": "",
                    "bundle_id": (
                        f"{model_input.get('policy_id', '<unknown>')}@"
                        f"{model_input.get('policy_version', '<unknown>')}"
                    ),
                    "tool_interface": "unknown",
                    "allowed_tools_signature": "",
                }
            treatment_tokens = tokenize_text(treatment.get("text", ""))[: self.max_tokens]
            treatment_ids = [self.token_to_id.get(token, 1) for token in treatment_tokens]
            treatment_padding = [0] * (self.max_tokens - len(treatment_ids))
            out["treatment_token_ids"] = treatment_ids + treatment_padding
            out["treatment_token_mask"] = [1] * len(treatment_ids) + [0] * len(
                treatment_padding
            )
            out["treatment_token_length"] = len(treatment_ids)

            treatment_numeric: list[float] = []
            treatment_numeric_mask: list[int] = []
            for key in self.treatment_numeric_keys:
                value = _as_finite_float(treatment.get(key))
                if value is None:
                    treatment_numeric.append(0.0)
                    treatment_numeric_mask.append(0)
                else:
                    std = self.treatment_numeric_std.get(key, 0.0)
                    mean = self.treatment_numeric_mean.get(key, 0.0)
                    treatment_numeric.append((value - mean) / std if std > 0 else 0.0)
                    treatment_numeric_mask.append(1)
            out["treatment_numeric"] = treatment_numeric
            out["treatment_numeric_mask"] = treatment_numeric_mask
            for field in TREATMENT_CATEGORICAL_FIELDS:
                raw = treatment.get(field)
                out[f"treatment_{field}"] = self.treatment_cat_vocab[field].get(
                    str(raw) if raw is not None else "", 0
                )
        return out

    # -- serialization -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        value = {
            "version": 2 if self.treatment_enabled else 1,
            "max_vocab": self.max_vocab,
            "max_tokens": self.max_tokens,
            "token_to_id": dict(self.token_to_id),
            "numeric_keys": list(self.numeric_keys),
            "numeric_mean": {key: float(value) for key, value in self.numeric_mean.items()},
            "numeric_std": {key: float(value) for key, value in self.numeric_std.items()},
            "cat_vocab": {field: dict(vocab) for field, vocab in self.cat_vocab.items()},
        }
        if self.treatment_enabled:
            value.update(
                {
                    "treatment_enabled": True,
                    "treatment_numeric_keys": list(self.treatment_numeric_keys),
                    "treatment_numeric_mean": dict(self.treatment_numeric_mean),
                    "treatment_numeric_std": dict(self.treatment_numeric_std),
                    "treatment_cat_vocab": {
                        field: dict(vocab)
                        for field, vocab in self.treatment_cat_vocab.items()
                    },
                }
            )
        return value

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Preprocessor":
        pre = cls(max_vocab=data["max_vocab"], max_tokens=data["max_tokens"])
        pre.token_to_id = {str(key): int(value) for key, value in data["token_to_id"].items()}
        pre.numeric_keys = [str(key) for key in data["numeric_keys"]]
        pre.numeric_mean = {str(key): float(value) for key, value in data["numeric_mean"].items()}
        pre.numeric_std = {str(key): float(value) for key, value in data["numeric_std"].items()}
        pre.cat_vocab = {
            str(field): {str(token): int(value) for token, value in vocab.items()}
            for field, vocab in data["cat_vocab"].items()
        }
        pre.treatment_enabled = bool(data.get("treatment_enabled", False))
        pre.treatment_numeric_keys = [
            str(key) for key in data.get("treatment_numeric_keys", [])
        ]
        pre.treatment_numeric_mean = {
            str(key): float(value)
            for key, value in data.get("treatment_numeric_mean", {}).items()
        }
        pre.treatment_numeric_std = {
            str(key): float(value)
            for key, value in data.get("treatment_numeric_std", {}).items()
        }
        pre.treatment_cat_vocab = {
            str(field): {str(token): int(value) for token, value in vocab.items()}
            for field, vocab in data.get("treatment_cat_vocab", {}).items()
        }
        return pre

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)


# ===========================================================================
# Pure metric functions (no sklearn)
# ===========================================================================
def _clip_probability(probability: float) -> float:
    return max(min(float(probability), 1.0 - _EPS), _EPS)


def log_loss(y_true: list[float], y_pred: list[float]) -> float | None:
    """Mean negative Bernoulli log likelihood, probabilities clipped to ``_EPS``."""
    if not y_true:
        return None
    y_true = [float(value) for value in y_true]
    y_pred = [_clip_probability(value) for value in y_pred]
    return -sum(
        y * math.log(p) + (1.0 - y) * math.log(1.0 - p) for y, p in zip(y_true, y_pred)
    ) / len(y_true)


def brier_score(y_true: list[float], y_pred: list[float]) -> float | None:
    """Mean squared error between predicted success probability and label."""
    if not y_true:
        return None
    return sum((float(p) - float(y)) ** 2 for y, p in zip(y_true, y_pred)) / len(y_true)


def accuracy_at_05(y_true: list[float], y_pred: list[float]) -> float | None:
    """Accuracy with the 0.5 decision threshold."""
    if not y_true:
        return None
    hits = sum(1 for y, p in zip(y_true, y_pred) if (float(p) >= 0.5) == (float(y) >= 0.5))
    return hits / len(y_true)


def expected_calibration_error(y_true: list[float], y_pred: list[float], num_bins: int = 10) -> float | None:
    """Expected calibration error over equal-width bins in ``[0, 1]``."""
    if not y_true:
        return None
    y_true = [float(value) for value in y_true]
    y_pred = [_clip_probability(value) for value in y_pred]
    total = 0.0
    for bin_index in range(num_bins):
        low = bin_index / num_bins
        high = (bin_index + 1) / num_bins
        members = [
            (y, p)
            for y, p in zip(y_true, y_pred)
            if low <= p < high or (bin_index == num_bins - 1 and p == 1.0)
        ]
        if not members:
            continue
        confidence = sum(p for _, p in members) / len(members)
        accuracy = sum(y for y, _ in members) / len(members)
        total += (len(members) / len(y_true)) * abs(accuracy - confidence)
    return total


def mean_posterior_std(posterior_std: list[float]) -> float | None:
    if not posterior_std:
        return None
    return sum(float(value) for value in posterior_std) / len(posterior_std)


def average_precision_score(y_true: list[float], y_score: list[float]) -> float | None:
    """Compute uncalibrated Average Precision (area under precision-recall curve).

    Implemented without sklearn to keep this module dependency-light.
    """
    if not y_true:
        return None
    y_true = [1.0 if float(value) >= 0.5 else 0.0 for value in y_true]
    y_score = [float(value) for value in y_score]
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length")

    positives = sum(y_true)
    if positives == 0:
        return 0.0
    if positives == len(y_true):
        return 1.0

    pairs = sorted(zip(y_true, y_score), key=lambda item: item[1], reverse=True)
    tp = 0.0
    fp = 0.0
    prev_recall = 0.0
    ap = 0.0

    index = 0
    total = len(pairs)
    while index < total:
        score = pairs[index][1]
        while index < total and pairs[index][1] == score:
            if pairs[index][0] >= 0.5:
                tp += 1.0
            else:
                fp += 1.0
            index += 1
        recall = tp / positives
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        ap += precision * (recall - prev_recall)
        prev_recall = recall
    return ap


def binary_counts_at_threshold(y_true: list[float], y_score: list[float], threshold: float = 0.5) -> dict[str, int]:
    """Return confusion-matrix counts at a fixed threshold."""
    y_true = [1 if float(value) >= 0.5 else 0 for value in y_true]
    y_score = [float(value) for value in y_score]
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length")

    tn = fp = fn = tp = 0
    for label, score in zip(y_true, y_score):
        pred = 1 if score >= threshold else 0
        if label == 1 and pred == 1:
            tp += 1
        elif label == 1 and pred == 0:
            fn += 1
        elif label == 0 and pred == 1:
            fp += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def compute_metrics(
    y_true: list[float],
    y_pred: list[float],
    posterior_std: list[float] | None = None,
    *,
    include_classification_metrics: bool = False,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Per-split / per-policy metric bundle, with explicit nulls for empty data."""
    y_true = [float(value) for value in y_true]
    y_pred = [float(value) for value in y_pred]
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        return {
            "n": 0,
            "log_loss": None,
            "brier": None,
            "accuracy_05": None,
            "ece": None,
            "mean_posterior_std": None,
            "average_precision": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
        }

    metrics: dict[str, Any] = {
        "n": len(y_true),
        "log_loss": log_loss(y_true, y_pred),
        "brier": brier_score(y_true, y_pred),
        "accuracy_05": accuracy_at_05(y_true, y_pred),
        "ece": expected_calibration_error(y_true, y_pred),
        "mean_posterior_std": mean_posterior_std(posterior_std) if posterior_std else None,
        "average_precision": average_precision_score(y_true, y_pred),
    }
    if include_classification_metrics:
        counts = binary_counts_at_threshold(y_true, y_pred, threshold=threshold)
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        tn = counts["tn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics.update({
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        })
    else:
        metrics.update({"precision": None, "recall": None, "f1": None, "tp": 0, "fp": 0, "fn": 0, "tn": 0})
    return metrics


# ===========================================================================
# Data loading
# ===========================================================================
def load_dataset_rows(dataset_path: str | Path) -> list[dict[str, Any]]:
    """Load a deterministic JSONL dataset produced by ``dataset.py``.

    Every row must carry ``model_input`` (features) and ``verified_success``
    (label); other top-level fields are never consumed as features.
    """
    path = Path(dataset_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"dataset does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"malformed JSON at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict) or "model_input" not in row:
                raise ValueError(f"row {line_number} at {path} is missing 'model_input'")
            if "verified_success" not in row:
                raise ValueError(f"row {line_number} at {path} is missing 'verified_success'")
            rows.append(row)
    return rows


def group_by_split(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Preserve the whole-task splits already assigned in the dataset."""
    groups: dict[str, list[dict[str, Any]]] = {name: [] for name in _SPLIT_NAMES}
    for row in rows:
        split = row.get("split", "train")
        if split not in groups:
            split = "train"
        groups[split].append(row)
    return groups


# ===========================================================================
# PyTorch model (lazily guarded; only usable when torch is installed)
# ===========================================================================
if nn is None:

    class _TorchStubModule:
        """Base class placeholder when torch is absent; instantiation raises."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "PyTorch is required for the neural model classes but is not installed."
            )

    _Module = _TorchStubModule
else:
    _Module = nn.Module


def build_model_config(
    pre: Preprocessor,
    *,
    text_dim: int = 64,
    cat_dim: int = 16,
    numeric_hidden: int = 32,
    fusion_hidden: int = 64,
    dropout: float = 0.2,
    prior_sigma: float = 1.0,
    rho_init: float = -3.0,
    weight_init_std: float = 0.05,
) -> dict[str, Any]:
    """Assemble the model config from a fitted preprocessor plus hyperparameters."""
    return {
        "vocab_size": pre.vocab_size,
        "max_tokens": pre.max_tokens,
        "num_numeric": len(pre.numeric_keys),
        "cat_sizes": {field: len(pre.cat_vocab[field]) for field in CATEGORICAL_FIELDS},
        "treatment_enabled": pre.treatment_enabled,
        "num_treatment_numeric": len(pre.treatment_numeric_keys),
        "treatment_cat_sizes": {
            field: len(pre.treatment_cat_vocab.get(field, {_UNK: 0}))
            for field in TREATMENT_CATEGORICAL_FIELDS
        },
        "text_dim": int(text_dim),
        "cat_dim": int(cat_dim),
        "numeric_hidden": int(numeric_hidden),
        "fusion_hidden": int(fusion_hidden),
        "dropout": float(dropout),
        "prior_sigma": float(prior_sigma),
        "rho_init": float(rho_init),
        "weight_init_std": float(weight_init_std),
    }


class BayesianLinear(_Module):
    """Variational Bayesian linear layer with a standard-normal prior.

    ``mu``/``rho`` are trainable; ``sigma = softplus(rho) + eps``.  Forward
    returns one reparameterized sample; :meth:`kl` returns the closed-form
    KL divergence between the per-coordinate posterior ``N(mu, sigma**2)``
    and the prior ``N(0, prior_sigma**2)`` summed over all coordinates.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        *,
        prior_sigma: float = 1.0,
        rho_init: float = -3.0,
        init_std: float = 0.05,
    ) -> None:
        super().__init__()
        if in_features < 1:
            raise ValueError("in_features must be >= 1")
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = float(prior_sigma)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_rho = nn.Parameter(torch.empty(out_features))
        nn.init.normal_(self.weight_mu, std=init_std)
        nn.init.normal_(self.bias_mu, std=init_std)
        nn.init.constant_(self.weight_rho, rho_init)
        nn.init.constant_(self.bias_rho, rho_init)

    def _sigma(self, rho: torch.Tensor) -> torch.Tensor:
        return F.softplus(rho) + 1e-6

    def _sample(self, mu: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mu)
        return mu + self._sigma(rho) * eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self._sample(self.weight_mu, self.weight_rho), self._sample(self.bias_mu, self.bias_rho))

    def _kl_term(self, mu: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        sigma = self._sigma(rho)
        prior = self.prior_sigma
        return torch.log(torch.tensor(prior, dtype=mu.dtype, device=mu.device)) - torch.log(sigma) + (
            sigma**2 + mu**2 - prior**2
        ) / (2 * prior**2)

    def kl(self) -> torch.Tensor:
        return self._kl_term(self.weight_mu, self.weight_rho).sum() + self._kl_term(self.bias_mu, self.bias_rho).sum()

    def sample_parameters(self, num_samples: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            (self._sample(self.weight_mu, self.weight_rho), self._sample(self.bias_mu, self.bias_rho))
            for _ in range(max(num_samples, 1))
        ]


class OutcomeModel(_Module):
    """Policy-conditioned multimodal neural success model with a Bayesian head."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = dict(config)
        vocab_size = config["vocab_size"]
        text_dim = config["text_dim"]
        cat_dim = config["cat_dim"]
        numeric_hidden = config["numeric_hidden"]
        fusion_hidden = config["fusion_hidden"]
        dropout = config["dropout"]
        if vocab_size < 2:
            raise ValueError("vocab_size must be >= 2 (PAD and UNK)")
        self.text_embedding = nn.Embedding(vocab_size, text_dim, padding_idx=0)
        self.cat_embeddings = nn.ModuleDict(
            {
                field: nn.Embedding(config["cat_sizes"][field], cat_dim)
                for field in CATEGORICAL_FIELDS
            }
        )
        self.numeric_mlp = nn.Sequential(
            nn.Linear(config["num_numeric"] * 2, numeric_hidden),
            nn.ReLU(),
        )
        self.treatment_enabled = bool(config.get("treatment_enabled", False))
        treatment_hidden = max(int(fusion_hidden) // 2, 8)
        if self.treatment_enabled:
            self.treatment_text_embedding = nn.Embedding(
                vocab_size, text_dim, padding_idx=0
            )
            self.treatment_cat_embeddings = nn.ModuleDict(
                {
                    field: nn.Embedding(config["treatment_cat_sizes"][field], cat_dim)
                    for field in TREATMENT_CATEGORICAL_FIELDS
                }
            )
            treatment_numeric_dim = int(config.get("num_treatment_numeric", 0))
            self.treatment_numeric_mlp = nn.Sequential(
                nn.Linear(treatment_numeric_dim * 2, numeric_hidden),
                nn.ReLU(),
            )
            treatment_in = (
                text_dim
                + len(TREATMENT_CATEGORICAL_FIELDS) * cat_dim
                + numeric_hidden
                + 1
            )
            self.treatment_fusion = nn.Sequential(
                nn.Linear(treatment_in, treatment_hidden),
                nn.ReLU(),
            )
        fusion_in = text_dim + len(CATEGORICAL_FIELDS) * cat_dim + numeric_hidden + 1
        if self.treatment_enabled:
            fusion_in += treatment_hidden
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = BayesianLinear(
            fusion_hidden,
            1,
            prior_sigma=config["prior_sigma"],
            rho_init=config["rho_init"],
            init_std=config["weight_init_std"],
        )

    def encode(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Deterministic multimodal representation; the Bayesian scope is the head."""
        text = self.text_embedding(x["token_ids"])  # (B, T, text_dim)
        mask = x["token_mask"].unsqueeze(-1)
        masked_sum = (text * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1.0)
        text_vec = masked_sum / count  # masked mean

        cat_vecs = [self.cat_embeddings[field](x[field]) for field in CATEGORICAL_FIELDS]
        numeric_in = torch.cat([x["numeric"], x["numeric_mask"]], dim=1)
        numeric_vec = self.numeric_mlp(numeric_in)
        length_feature = torch.log1p(x["token_length"].float()).unsqueeze(1)
        parts = [text_vec, *cat_vecs, numeric_vec, length_feature]
        if self.treatment_enabled:
            treatment_text = self.treatment_text_embedding(x["treatment_token_ids"])
            treatment_mask = x["treatment_token_mask"].unsqueeze(-1)
            treatment_sum = (treatment_text * treatment_mask).sum(dim=1)
            treatment_count = treatment_mask.sum(dim=1).clamp(min=1.0)
            treatment_text_vec = treatment_sum / treatment_count
            treatment_cat_vecs = [
                self.treatment_cat_embeddings[field](x[f"treatment_{field}"])
                for field in TREATMENT_CATEGORICAL_FIELDS
            ]
            treatment_numeric_in = torch.cat(
                [x["treatment_numeric"], x["treatment_numeric_mask"]], dim=1
            )
            treatment_numeric_vec = self.treatment_numeric_mlp(treatment_numeric_in)
            treatment_length = torch.log1p(
                x["treatment_token_length"].float()
            ).unsqueeze(1)
            treatment_vec = self.treatment_fusion(
                torch.cat(
                    [
                        treatment_text_vec,
                        *treatment_cat_vecs,
                        treatment_numeric_vec,
                        treatment_length,
                    ],
                    dim=1,
                )
            )
            parts.append(treatment_vec)
        return self.fusion(torch.cat(parts, dim=1))

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        logits = self.head(self.encode(x)).squeeze(-1)  # (B,)
        return {"logits": logits, "kl": self.head.kl()}

    def posterior_logits(self, x: dict[str, torch.Tensor], num_samples: int = 50) -> torch.Tensor:
        """One reparameterized logit sample per row per posterior draw -> (S, B)."""
        features = self.encode(x)
        samples = self.head.sample_parameters(num_samples)
        return torch.stack(
            [F.linear(features, weight, bias).squeeze(-1) for weight, bias in samples], dim=0
        )

    def posterior_predict(
        self, x: dict[str, torch.Tensor], num_samples: int = 50, seed: int | None = None
    ) -> dict[str, torch.Tensor]:
        """Posterior predictive mean/std/quantiles of P(success) over the head."""
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        logits = self.posterior_logits(x, num_samples)
        probs = torch.sigmoid(logits)  # (S, B)
        mean = probs.mean(dim=0)
        std = probs.std(dim=0, unbiased=True) if num_samples > 1 else torch.zeros_like(mean)
        quantiles = torch.quantile(
            probs,
            torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], dtype=probs.dtype, device=probs.device),
            dim=0,
        )  # (5, B)
        return {"mean": mean, "std": std, "quantiles": quantiles}


def collate_transform(transformed: list[dict[str, Any]], device: str = "cpu") -> dict[str, torch.Tensor]:
    """Stack preprocessor output rows into tensors for the model."""
    _require_torch()
    stacked = {
        "token_ids": torch.tensor([x["token_ids"] for x in transformed], dtype=torch.long, device=device),
        "token_mask": torch.tensor([x["token_mask"] for x in transformed], dtype=torch.float32, device=device),
        "token_length": torch.tensor([x["token_length"] for x in transformed], dtype=torch.long, device=device),
        "numeric": torch.tensor([x["numeric"] for x in transformed], dtype=torch.float32, device=device),
        "numeric_mask": torch.tensor([x["numeric_mask"] for x in transformed], dtype=torch.float32, device=device),
    }
    for field in CATEGORICAL_FIELDS:
        stacked[field] = torch.tensor([x[field] for x in transformed], dtype=torch.long, device=device)
    if transformed and "treatment_token_ids" in transformed[0]:
        stacked.update(
            {
                "treatment_token_ids": torch.tensor(
                    [x["treatment_token_ids"] for x in transformed],
                    dtype=torch.long,
                    device=device,
                ),
                "treatment_token_mask": torch.tensor(
                    [x["treatment_token_mask"] for x in transformed],
                    dtype=torch.float32,
                    device=device,
                ),
                "treatment_token_length": torch.tensor(
                    [x["treatment_token_length"] for x in transformed],
                    dtype=torch.long,
                    device=device,
                ),
                "treatment_numeric": torch.tensor(
                    [x["treatment_numeric"] for x in transformed],
                    dtype=torch.float32,
                    device=device,
                ),
                "treatment_numeric_mask": torch.tensor(
                    [x["treatment_numeric_mask"] for x in transformed],
                    dtype=torch.float32,
                    device=device,
                ),
            }
        )
        for field in TREATMENT_CATEGORICAL_FIELDS:
            stacked[f"treatment_{field}"] = torch.tensor(
                [x[f"treatment_{field}"] for x in transformed],
                dtype=torch.long,
                device=device,
            )
    return stacked


# ===========================================================================
# Determinism helpers
# ===========================================================================
def set_seed(seed: int, torch_mod: Any = None) -> None:
    """Seed Python/torch RNGs for deterministic training and prediction."""
    torch_mod = torch_mod or _require_torch()
    random.seed(seed)
    torch_mod.manual_seed(seed)
    if torch_mod.cuda.is_available():
        torch_mod.cuda.manual_seed_all(seed)
    try:
        torch_mod.backends.cudnn.deterministic = True
        torch_mod.backends.cudnn.benchmark = False
    except Exception:
        pass


def resolve_device(device: str | None = None, torch_mod: Any = None) -> str:
    torch_mod = torch_mod or _require_torch()
    device = device or "cpu"
    if str(device).startswith("cuda") and not torch_mod.cuda.is_available():
        print("warning: CUDA requested but unavailable; falling back to CPU", file=sys.stderr)
        return "cpu"
    return device


# ===========================================================================
# Evaluation / prediction
# ===========================================================================
def _split_metrics(
    model: "OutcomeModel",
    pre: Preprocessor,
    rows: list[dict[str, Any]],
    device: str,
    num_samples: int = 50,
    seed: int | None = None,
) -> dict[str, Any]:
    """Posterior-predict every row in ``rows`` and compute metric bundles."""
    if not rows:
        return {
            "n": 0,
            "log_loss": None,
            "brier": None,
            "accuracy_05": None,
            "ece": None,
            "mean_posterior_std": None,
            "average_precision": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "per_policy": {},
        }
    x = collate_transform([pre.transform(row["model_input"]) for row in rows], device)
    y_true = [float(bool(row["verified_success"])) for row in rows]
    with torch.no_grad():
        posterior = model.posterior_predict(x, num_samples=num_samples, seed=seed)
    mean = posterior["mean"].tolist()
    std = posterior["std"].tolist()
    metrics = compute_metrics(y_true, mean, std, include_classification_metrics=True)
    per_policy: dict[str, Any] = {}
    for policy_id in sorted({str(row["model_input"].get("policy_id")) for row in rows}):
        index = [i for i, row in enumerate(rows) if str(row["model_input"].get("policy_id")) == policy_id]
        per_policy[policy_id] = compute_metrics(
            [y_true[i] for i in index], [mean[i] for i in index], [std[i] for i in index], include_classification_metrics=True
        )
    metrics["per_policy"] = per_policy
    return metrics


def _quantile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    left = int(math.floor(pos))
    right = int(math.ceil(pos))
    if right == left:
        return sorted_values[left]
    weight = pos - left
    return (1.0 - weight) * sorted_values[left] + weight * sorted_values[right]


def _summarize_values(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p05": None,
            "p50": None,
            "p95": None,
        }
    values = [float(value) for value in values]
    n = len(values)
    mu = sum(values) / n
    sorted_values = sorted(values)
    std = math.sqrt(sum((value - mu) ** 2 for value in values) / (n - 1)) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mu,
        "std": std,
        "min": sorted_values[0],
        "max": sorted_values[-1],
        "p05": _quantile(sorted_values, 0.05),
        "p50": _quantile(sorted_values, 0.50),
        "p95": _quantile(sorted_values, 0.95),
    }


def _safe_fraction_zero_credible(weight_mu: Any, weight_sigma: Any, z: float = 1.959963984540054) -> float:
    mu = weight_mu.view(-1).detach().float()
    sigma = weight_sigma.view(-1).detach().float()
    if sigma.numel() == 0:
        return 0.0
    lo = mu - z * sigma
    hi = mu + z * sigma
    return float(((lo <= 0.0) & (hi >= 0.0)).float().mean().item())


def summarize_head_distribution(head: BayesianLinear, prior_sigma: float) -> dict[str, Any]:
    weight_mu = head.weight_mu.detach().cpu()
    weight_rho = head.weight_rho.detach().cpu()
    weight_sigma = torch.nn.functional.softplus(weight_rho)
    bias_mu = head.bias_mu.detach().cpu()
    bias_rho = head.bias_rho.detach().cpu()
    bias_sigma = torch.nn.functional.softplus(bias_rho)

    return {
        "weight": {
            "count": int(weight_mu.numel()),
            "mu_mean": float(weight_mu.mean().item()),
            "mu_std": float(weight_mu.std(unbiased=False).item()),
            "mu_min": float(weight_mu.min().item()),
            "mu_max": float(weight_mu.max().item()),
            "sigma_mean": float(weight_sigma.mean().item()),
            "sigma_std": float(weight_sigma.std(unbiased=False).item()),
            "sigma_min": float(weight_sigma.min().item()),
            "sigma_max": float(weight_sigma.max().item()),
            "sigma_ratio_to_prior": float(weight_sigma.mean().item() / prior_sigma),
            "zero_in_95ci_frac": _safe_fraction_zero_credible(weight_mu, weight_sigma),
        },
        "bias": {
            "count": int(bias_mu.numel()),
            "mu_mean": float(bias_mu.mean().item()),
            "mu_std": float(bias_mu.std(unbiased=False).item()),
            "mu_min": float(bias_mu.min().item()),
            "mu_max": float(bias_mu.max().item()),
            "sigma_mean": float(bias_sigma.mean().item()),
            "sigma_std": float(bias_sigma.std(unbiased=False).item()),
            "sigma_min": float(bias_sigma.min().item()),
            "sigma_max": float(bias_sigma.max().item()),
            "sigma_ratio_to_prior": float(bias_sigma.mean().item() / prior_sigma),
            "zero_in_95ci_frac": _safe_fraction_zero_credible(bias_mu, bias_sigma),
        },
        "kl": float(head.kl().item()),
    }


def _sample_head_weights(
    head: BayesianLinear,
    num_samples: int,
    *,
    from_prior: bool,
    prior_sigma: float,
    seed: int,
) -> tuple[Any, Any]:
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if from_prior:
        weight_shape = head.weight_mu.shape
        bias_shape = head.bias_mu.shape
        w = torch.randn(
            (num_samples, *weight_shape), dtype=head.weight_mu.dtype, device=head.weight_mu.device
        )
        b = torch.randn((num_samples, *bias_shape), dtype=head.bias_mu.dtype, device=head.bias_mu.device)
        w = (w * prior_sigma).view(num_samples, -1)
        b = (b * prior_sigma).view(num_samples, -1)
    else:
        samples = head.sample_parameters(num_samples)
        w = torch.stack([ws.view(-1) for ws, _ in samples], dim=0)
        b = torch.stack([bs.view(-1) for _, bs in samples], dim=0)
    return w, b


def _logits_from_features(features: torch.Tensor, weight_samples: torch.Tensor, bias_samples: torch.Tensor) -> torch.Tensor:
    logits = torch.einsum("sf,bf->sb", weight_samples, features)
    logits = logits + bias_samples.squeeze(-1).unsqueeze(1)
    return logits


def _sample_metric_summary(y_true: list[float], probs_samples: torch.Tensor) -> tuple[dict[str, Any], dict[str, Any]]:
    split_metric_samples: dict[str, list[float]] = {name: [] for name in _INSPECT_METRIC_NAMES}
    split_rate_stats: list[float] = []
    for sample_index in range(probs_samples.shape[0]):
        probs = probs_samples[sample_index].tolist()
        metrics = compute_metrics(y_true, probs, include_classification_metrics=True)
        for name in _INSPECT_METRIC_NAMES:
            split_metric_samples[name].append(metrics[name])
        split_rate = sum(1 for value in probs if value >= 0.5) / len(probs)
        split_rate_stats.append(float(split_rate))

    metric_summary = {name: _summarize_values(values) for name, values in split_metric_samples.items()}
    return metric_summary, _summarize_values(split_rate_stats)


def _downsample_rows(rows: list[dict[str, Any]], limit: int | None, seed: int) -> list[dict[str, Any]]:
    if limit is None or limit <= 0 or len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    rows_copy = list(rows)
    rng.shuffle(rows_copy)
    return rows_copy[:limit]


def run_split_bayesian_diagnostics(
    label_rows: list[dict[str, Any]],
    model: "OutcomeModel",
    pre_transform_rows: list[dict[str, Any]],
    device: str,
    prior_sigma: float,
    posterior_samples: int,
    prior_samples: int,
    seed: int,
) -> dict[str, Any]:
    if not label_rows:
        return {
            "n": 0,
            "labels_true_rate": None,
            "posterior": {
                "point_metrics": None,
                "mc_metrics": None,
                "pred_mean_summary": None,
                "pred_std_summary": None,
                "rate_summary": None,
            },
            "prior": {
                "mc_metrics": None,
                "rate_summary": None,
            },
        }

    y_true = [1.0 if bool(row.get("verified_success", False)) else 0.0 for row in label_rows]
    y_rate = sum(y_true) / len(y_true)

    x = collate_transform(pre_transform_rows, device=device)
    with torch.no_grad():
        features = model.encode(x)

    posterior_w, posterior_b = _sample_head_weights(
        model.head,
        posterior_samples,
        from_prior=False,
        prior_sigma=prior_sigma,
        seed=seed,
    )
    prior_w, prior_b = _sample_head_weights(
        model.head,
        prior_samples,
        from_prior=True,
        prior_sigma=prior_sigma,
        seed=seed + 1,
    )

    posterior_w = posterior_w.to(features.device)
    posterior_b = posterior_b.to(features.device)
    prior_w = prior_w.to(features.device)
    prior_b = prior_b.to(features.device)

    with torch.no_grad():
        posterior_logits = _logits_from_features(features, posterior_w, posterior_b)
        posterior_probs = torch.sigmoid(posterior_logits)
        prior_logits = _logits_from_features(features, prior_w, prior_b)
        prior_probs = torch.sigmoid(prior_logits)

        posterior_mean = posterior_probs.mean(dim=0).tolist()
        posterior_std = (
            posterior_probs.std(dim=0, unbiased=True).tolist() if posterior_samples > 1 else [0.0] * len(y_true)
        )
        point_metrics = compute_metrics(
            y_true,
            posterior_mean,
            posterior_std,
            include_classification_metrics=True,
        )

    posterior_mc_metrics, posterior_rate_summary = _sample_metric_summary(y_true, posterior_probs)
    prior_mc_metrics, prior_rate_summary = _sample_metric_summary(y_true, prior_probs)

    return {
        "n": len(y_true),
        "labels_true_rate": y_rate,
        "posterior": {
            "point_metrics": {name: point_metrics[name] for name in _INSPECT_METRIC_NAMES},
            "mc_metrics": posterior_mc_metrics,
            "pred_mean_summary": _summarize_values(posterior_mean),
            "pred_std_summary": _summarize_values([float(v) for v in posterior_std]),
            "rate_summary": posterior_rate_summary,
        },
        "prior": {
            "mc_metrics": prior_mc_metrics,
            "rate_summary": prior_rate_summary,
        },
    }


def inspect_artifact_fit(
    artifact_dir: str | Path,
    *,
    dataset: str | Path | None = None,
    device: str = "cpu",
    seed: int = 42,
    posterior_samples: int = 64,
    prior_samples: int = 64,
    max_rows: int | None = None,
) -> dict[str, Any]:
    set_seed(seed)
    config, pre, model = load_artifacts(artifact_dir, device=device)
    model.eval()
    prior_sigma = float(config["model"].get("prior_sigma", 1.0))

    posterior_summary = summarize_head_distribution(model.head, prior_sigma)
    initial_model = OutcomeModel(config["model"])
    initial_summary = summarize_head_distribution(initial_model.head, prior_sigma)

    payload: dict[str, Any] = {
        "artifact_dir": str(Path(artifact_dir).expanduser().resolve()),
        "device": device,
        "seed": seed,
        "prior_sigma": prior_sigma,
        "prior_hyperprior": {"mean": 0.0, "sigma": prior_sigma, "variance": prior_sigma ** 2},
        "head": {
            "trained": posterior_summary,
            "initialized": initial_summary,
            "kl_reduction": initial_summary["kl"] - posterior_summary["kl"],
        },
        "splits": {},
    }

    if dataset is not None:
        rows = load_dataset_rows(dataset)
        split_rows = group_by_split(rows)
        for split_name in _SPLIT_NAMES:
            subset_rows = _downsample_rows(split_rows.get(split_name, []), max_rows, seed + 100 + len(split_name))
            transformed = [pre.transform(row["model_input"]) for row in subset_rows]
            payload["splits"][split_name] = run_split_bayesian_diagnostics(
                label_rows=subset_rows,
                model=model,
                pre_transform_rows=transformed,
                device=device,
                prior_sigma=prior_sigma,
                posterior_samples=posterior_samples,
                prior_samples=prior_samples,
                seed=seed + 1000,
            )
        payload["dataset_rows"] = len(rows)
        payload["dataset_tasks"] = len({row.get("task_id") for row in rows})
    return payload


def predict_single(
    model: "OutcomeModel",
    pre: Preprocessor,
    model_input: dict[str, Any],
    *,
    num_samples: int = 50,
    seed: int | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Posterior predictive summary for one JSON ``model_input`` dict."""
    x = collate_transform([pre.transform(model_input)], device)
    with torch.no_grad():
        posterior = model.posterior_predict(x, num_samples=num_samples, seed=seed)
    quantiles = posterior["quantiles"].squeeze(1).tolist()
    return {
        "mean": posterior["mean"].item(),
        "std": posterior["std"].item(),
        "quantiles": {
            "0.05": quantiles[0],
            "0.25": quantiles[1],
            "0.5": quantiles[2],
            "0.75": quantiles[3],
            "0.95": quantiles[4],
        },
        "num_samples": num_samples,
    }


def score_policy_counterfactuals(
    model: "OutcomeModel",
    pre: Preprocessor,
    model_input: dict[str, Any],
    *,
    policies: list[tuple[str, str]] | None = None,
    num_samples: int = 50,
    seed: int | None = None,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Score candidate ``(policy_id, policy_version)`` pairs on the same task.

    Only ``model_input.policy_id`` / ``model_input.policy_version`` are
    replaced; every other predecision feature stays identical.
    """
    if pre.treatment_enabled:
        raise ValueError(
            "this artifact uses treatment descriptors; policy ID/version alone "
            "cannot define a counterfactual. Use score_treatment_counterfactuals()."
        )
    if policies is None:
        policy_ids = [token for token in pre.cat_vocab["policy_id"] if token != _UNK]
        versions = [token for token in pre.cat_vocab["policy_version"] if token != _UNK]
        version = versions[0] if versions else str(model_input.get("policy_version", "1"))
        policies = [(policy_id, version) for policy_id in policy_ids]
    results: list[dict[str, Any]] = []
    for policy_id, policy_version in policies:
        candidate = dict(model_input)
        candidate["policy_id"] = policy_id
        candidate["policy_version"] = policy_version
        results.append(
            {
                "policy_id": policy_id,
                "policy_version": policy_version,
                **predict_single(
                    model, pre, candidate, num_samples=num_samples, seed=seed, device=device
                ),
            }
        )
    return results


def score_treatment_counterfactuals(
    model: "OutcomeModel",
    pre: Preprocessor,
    model_input: dict[str, Any],
    treatments: list[TreatmentSpec],
    *,
    num_samples: int = 50,
    seed: int | None = None,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Score complete treatment bundles while holding task features fixed.

    Unlike :func:`score_policy_counterfactuals`, this replaces the categorical
    identity *and* the prompt/tool/budget descriptor.  That is required for
    novel policy bundles whose IDs map to UNK but whose descriptors can still
    map to different treatment representations.
    """

    if not pre.treatment_enabled:
        raise ValueError(
            "this artifact has no treatment encoder; use score_policy_counterfactuals()"
        )
    if not treatments:
        raise ValueError("at least one treatment is required")
    results: list[dict[str, Any]] = []
    for treatment in treatments:
        candidate = dict(model_input)
        candidate["policy_id"] = treatment.id
        candidate["policy_version"] = treatment.version
        candidate["treatment"] = treatment_model_input_descriptor(treatment)
        results.append(
            {
                "policy_id": treatment.id,
                "policy_version": treatment.version,
                "bundle_id": treatment.bundle_id,
                "bundle_hash": treatment.bundle_hash,
                **predict_single(
                    model,
                    pre,
                    candidate,
                    num_samples=num_samples,
                    seed=seed,
                    device=device,
                ),
            }
        )
    return results


# ===========================================================================
# Training
# ===========================================================================
def train_model(
    dataset_path: str | Path,
    artifact_dir: str | Path,
    *,
    epochs: int = 50,
    batch_size: int = 16,
    seed: int = 42,
    device: str = "cpu",
    max_vocab: int = 5000,
    max_tokens: int = 256,
    text_dim: int = 64,
    cat_dim: int = 16,
    numeric_hidden: int = 32,
    fusion_hidden: int = 64,
    dropout: float = 0.2,
    prior_sigma: float = 1.0,
    lr: float = 1e-3,
    patience: int = 5,
    min_delta: float = 1e-4,
    num_samples: int = 50,
    verbose: bool = True,
) -> dict[str, Any]:
    """Fit the model on the train split, early-stop on validation Brier, save artifacts.

    The preprocessor is fitted on the train split only.  Validation/test splits
    are scored with the same trained model; empty splits produce explicit null
    metrics rather than falling back to training data.
    """
    torch_mod = _require_torch()
    device = resolve_device(device, torch_mod)
    set_seed(seed, torch_mod)

    rows = load_dataset_rows(dataset_path)
    if not rows:
        raise ValueError(f"dataset contains no rows: {dataset_path}")
    splits = group_by_split(rows)
    train_rows = splits["train"]
    if not train_rows:
        raise ValueError(f"dataset contains no train rows: {dataset_path}")
    validation_rows = splits["validation"]
    test_rows = splits["test"]

    pre = Preprocessor(max_vocab=max_vocab, max_tokens=max_tokens)
    pre.fit([row["model_input"] for row in train_rows])

    config = build_model_config(
        pre,
        text_dim=text_dim,
        cat_dim=cat_dim,
        numeric_hidden=numeric_hidden,
        fusion_hidden=fusion_hidden,
        dropout=dropout,
        prior_sigma=prior_sigma,
    )
    model = OutcomeModel(config).to(device)
    optimizer = torch_mod.optim.Adam(model.parameters(), lr=lr)

    train_x = collate_transform([pre.transform(row["model_input"]) for row in train_rows], device)
    train_y = torch_mod.tensor(
        [float(bool(row["verified_success"])) for row in train_rows], dtype=torch.float32, device=device
    )
    shuffle = torch_mod.Generator(device=device).manual_seed(seed)

    best_brier: float | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    patience_left = patience
    stop_epoch = epochs

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch_mod.randperm(len(train_rows), generator=shuffle)
        total_loss = 0.0
        num_batches = 0
        for start in range(0, len(permutation), batch_size):
            index = permutation[start : start + batch_size]
            batch_x = {key: value[index] for key, value in train_x.items()}
            batch_y = train_y[index]
            optimizer.zero_grad(set_to_none=True)
            output = model(batch_x)
            bce = F.binary_cross_entropy_with_logits(output["logits"], batch_y)
            # ELBO per example: negative log-likelihood + KL / N.
            loss = bce + output["kl"] / len(train_rows)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            num_batches += 1
        train_loss = total_loss / max(num_batches, 1)

        if validation_rows:
            monitor_brier = _split_metrics(
                model, pre, validation_rows, device, num_samples=num_samples
            )["brier"]
        else:
            monitor_brier = _split_metrics(
                model, pre, train_rows, device, num_samples=num_samples
            )["brier"]

        if best_brier is None or monitor_brier < best_brier - min_delta:
            best_brier = monitor_brier
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            patience_left = patience
        else:
            patience_left -= 1

        if verbose:
            print(
                f"epoch {epoch:03d} loss={train_loss:.4f} brier={monitor_brier:.4f} "
                f"best={best_brier:.4f} best_epoch={best_epoch}",
                flush=True,
            )
        if validation_rows and patience_left <= 0:
            stop_epoch = epoch
            if verbose:
                print(f"early stopping at epoch {epoch} (best epoch {best_epoch})", flush=True)
            break

    if best_state is None:
        best_state = {key: value.clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()

    metrics: dict[str, Any] = {}
    for name in _SPLIT_NAMES:
        metrics[name] = _split_metrics(
            model, pre, splits[name], device, num_samples=num_samples, seed=seed + 1000
        )

    training_meta = {
        "seed": seed,
        "device": device,
        "epochs_configured": epochs,
        "epochs_run": stop_epoch,
        "best_epoch": best_epoch,
        "patience": patience,
        "batch_size": batch_size,
        "num_samples": num_samples,
        "lr": lr,
        "dataset": str(Path(dataset_path).expanduser().resolve()),
        "torch_version": getattr(torch_mod, "__version__", "unknown"),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "test_rows": len(test_rows),
        "feature_schema_version": 2 if pre.treatment_enabled else 1,
        "treatment_encoder_enabled": pre.treatment_enabled,
        "seen_treatment_bundle_ids": sorted(
            {
                str(treatment["bundle_id"])
                for row in train_rows
                if isinstance(
                    (treatment := row["model_input"].get("treatment")), dict
                )
                and treatment.get("bundle_id") is not None
            }
        ),
    }
    save_artifacts(artifact_dir, config, pre, model, metrics, training_meta)
    return {"metrics": metrics, "config": config, "training": training_meta}


# ===========================================================================
# Artifacts
# ===========================================================================
def _read_json_checked(path: Path, label: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"missing artifact {label} at {path}")
    with path.open("r", encoding="utf-8") as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError as error:
            raise ValueError(f"malformed {label} at {path}: {error}") from error


def save_artifacts(
    artifact_dir: str | Path,
    config: dict[str, Any],
    pre: Preprocessor,
    model: "OutcomeModel",
    metrics: dict[str, Any],
    training_meta: dict[str, Any],
) -> None:
    """Write ``config.json``, ``preprocessor.json``, ``model.pt``, ``metrics.json``."""
    _require_torch()
    directory = Path(artifact_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "config.json", {"model": config, "training": training_meta})
    write_json(directory / "preprocessor.json", pre.to_dict())
    write_json(directory / "metrics.json", metrics)
    model_path = directory / "model.pt"
    with tempfile.NamedTemporaryFile(dir=directory, prefix=".model.", suffix=".pt", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        torch.save(model.state_dict(), temp_path)
        os.replace(temp_path, model_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _load_state_dict_safe(path: Path, device: str, torch_mod: Any = None) -> dict[str, Any]:
    """torch.load with ``weights_only=True`` when the installed torch supports it."""
    torch_mod = torch_mod or _require_torch()
    try:
        return torch_mod.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch_mod.load(path, map_location=device)


def load_artifacts(artifact_dir: str | Path, device: str = "cpu") -> tuple[dict[str, Any], Preprocessor, "OutcomeModel"]:
    """Load a saved artifact directory back into ``(config, pre, model)``."""
    torch_mod = _require_torch()
    directory = Path(artifact_dir).expanduser().resolve()
    config = _read_json_checked(directory / "config.json", "config.json")
    pre = Preprocessor.from_dict(_read_json_checked(directory / "preprocessor.json", "preprocessor.json"))
    model = OutcomeModel(config["model"]).to(device)
    model.load_state_dict(_load_state_dict_safe(directory / "model.pt", device, torch_mod))
    model.eval()
    return config, pre, model


def evaluate_model(
    dataset_path: str | Path,
    artifact_dir: str | Path,
    *,
    device: str = "cpu",
    num_samples: int = 50,
    seed: int | None = None,
) -> dict[str, Any]:
    """Score a saved model on a dataset; refresh ``metrics.json`` in the artifact dir."""
    config, pre, model = load_artifacts(artifact_dir, device=device)
    if seed is None:
        seed = int(config.get("training", {}).get("seed", 0)) + 1000
    rows = load_dataset_rows(dataset_path)
    if not rows:
        raise ValueError(f"dataset contains no rows: {dataset_path}")
    splits = group_by_split(rows)
    metrics: dict[str, Any] = {}
    for name in _SPLIT_NAMES:
        metrics[name] = _split_metrics(model, pre, splits[name], device, num_samples=num_samples, seed=seed)
    directory = Path(artifact_dir).expanduser().resolve()
    write_json(directory / "metrics.json", metrics)
    return {
        "metrics": metrics,
        "dataset_rows": len(rows),
        "artifact_dir": str(directory),
    }


# ===========================================================================
# CLI
# ===========================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyreplab-harness-outcome-model",
        description="Train/evaluate a policy-conditioned Bayesian neural success model.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="fit the model and save artifacts")
    train.add_argument("dataset", help="deterministic JSONL dataset from dataset.py")
    train.add_argument("artifact_dir", help="directory for config/preprocessor/model/metrics")
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="cpu")
    train.add_argument("--max-vocab", type=int, default=5000)
    train.add_argument("--max-tokens", type=int, default=256)
    train.add_argument("--text-dim", type=int, default=64)
    train.add_argument("--cat-dim", type=int, default=16)
    train.add_argument("--numeric-hidden", type=int, default=32)
    train.add_argument("--fusion-hidden", type=int, default=64)
    train.add_argument("--dropout", type=float, default=0.2)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--patience", type=int, default=5)
    train.add_argument("--num-samples", type=int, default=50)

    evaluate = subparsers.add_parser("evaluate", help="score a saved model on a dataset")
    evaluate.add_argument("dataset", help="deterministic JSONL dataset from dataset.py")
    evaluate.add_argument("artifact_dir", help="artifact directory produced by train")
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--num-samples", type=int, default=50)
    evaluate.add_argument("--seed", type=int, default=None)

    predict = subparsers.add_parser(
        "predict", help="predict P(success) for one JSON model_input and score policy counterfactuals"
    )
    predict.add_argument("model_input", help="path to a JSON file containing one model_input dict")
    predict.add_argument("artifact_dir", help="artifact directory produced by train")
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--num-samples", type=int, default=50)
    predict.add_argument("--seed", type=int, default=None)
    predict.add_argument(
        "--treatment-registry",
        default=None,
        help="registry containing complete candidate treatment descriptors",
    )
    predict.add_argument(
        "--treatments",
        default=None,
        help="comma-separated treatment references (default: every registry entry)",
    )

    inspect = subparsers.add_parser("inspect", help="diagnose Bayesian fit and prior/posterior predictive diagnostics")
    inspect.add_argument("artifact_dir", help="artifact directory produced by train")
    inspect.add_argument("--dataset", default=None, help="optional JSONL dataset for split-level diagnostics")
    inspect.add_argument("--device", default="cpu")
    inspect.add_argument("--seed", type=int, default=42)
    inspect.add_argument("--posterior-samples", type=int, default=64)
    inspect.add_argument("--prior-samples", type=int, default=64)
    inspect.add_argument("--max-rows", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train":
            result = train_model(
                args.dataset,
                args.artifact_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=args.seed,
                device=args.device,
                max_vocab=args.max_vocab,
                max_tokens=args.max_tokens,
                text_dim=args.text_dim,
                cat_dim=args.cat_dim,
                numeric_hidden=args.numeric_hidden,
                fusion_hidden=args.fusion_hidden,
                dropout=args.dropout,
                lr=args.lr,
                patience=args.patience,
                num_samples=args.num_samples,
            )
            print(json.dumps({"metrics": result["metrics"]}, indent=2, sort_keys=True))
        elif args.command == "evaluate":
            result = evaluate_model(
                args.dataset,
                args.artifact_dir,
                device=args.device,
                num_samples=args.num_samples,
                seed=args.seed,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
        elif args.command == "predict":
            data = _read_json_checked(Path(args.model_input), "model_input")
            model_input = data["model_input"] if isinstance(data, dict) and "model_input" in data else data
            config, pre, model = load_artifacts(args.artifact_dir, device=args.device)
            prediction = predict_single(
                model, pre, model_input, num_samples=args.num_samples, seed=args.seed, device=args.device
            )
            if args.treatment_registry:
                registry = TreatmentRegistry.load(args.treatment_registry)
                if args.treatments:
                    references = [
                        value.strip()
                        for value in args.treatments.split(",")
                        if value.strip()
                    ]
                    treatments: list[TreatmentSpec] = []
                    for reference in references:
                        try:
                            treatments.append(registry.by_bundle_id(reference))
                        except KeyError:
                            if "@" in reference:
                                treatment_id, version = reference.rsplit("@", 1)
                                treatments.append(
                                    registry.by_id_version(treatment_id, version)
                                )
                            else:
                                treatments.append(registry.by_id(reference))
                else:
                    treatments = list(registry.treatments)
                counterfactuals = score_treatment_counterfactuals(
                    model,
                    pre,
                    model_input,
                    treatments,
                    num_samples=args.num_samples,
                    seed=args.seed,
                    device=args.device,
                )
            else:
                counterfactuals = score_policy_counterfactuals(
                    model,
                    pre,
                    model_input,
                    num_samples=args.num_samples,
                    seed=args.seed,
                    device=args.device,
                )
            print(
                json.dumps(
                    {"prediction": prediction, "counterfactuals": counterfactuals},
                    indent=2,
                    sort_keys=True,
                )
            )
        elif args.command == "inspect":
            set_seed(args.seed)
            result = inspect_artifact_fit(
                args.artifact_dir,
                dataset=args.dataset,
                device=args.device,
                seed=args.seed,
                posterior_samples=args.posterior_samples,
                prior_samples=args.prior_samples,
                max_rows=args.max_rows,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BayesianLinear",
    "CATEGORICAL_FIELDS",
    "TREATMENT_CATEGORICAL_FIELDS",
    "TREATMENT_NUMERIC_FIELDS",
    "OutcomeModel",
    "Preprocessor",
    "TORCH_AVAILABLE",
    "accuracy_at_05",
    "brier_score",
    "build_model_config",
    "build_parser",
    "collate_transform",
    "average_precision_score",
    "compute_metrics",
    "evaluate_model",
    "expected_calibration_error",
    "group_by_split",
    "load_artifacts",
    "load_dataset_rows",
    "log_loss",
    "main",
    "mean_posterior_std",
    "predict_single",
    "resolve_device",
    "run_split_bayesian_diagnostics",
    "save_artifacts",
    "score_policy_counterfactuals",
    "score_treatment_counterfactuals",
    "set_seed",
    "summarize_head_distribution",
    "tokenize_text",
    "train_model",
    "inspect_artifact_fit",
]
