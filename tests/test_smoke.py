"""
End-to-end smoke test — generate synthetic panel → tiny model →
ZINB NLL loss → verify shapes, finiteness, and gradient flow.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from civicsafe.synthetic.distributions import generate_spatiotemporal_panel

# ---------------------------------------------------------------------------
# Tiny model that maps features → (pi, mu, r)
# ---------------------------------------------------------------------------


class _TinyZINBHead(nn.Module):
    """Minimal model: Linear → split into zero-inflation, mean, dispersion."""

    def __init__(self, num_features: int, num_categories: int) -> None:
        super().__init__()
        # Output 3 values per category: pi, mu, r
        self.linear = nn.Linear(num_features, num_categories * 3)
        self.num_categories = num_categories

    def forward(self, feature_input: Tensor) -> Tensor:
        """Return (batch, num_categories, 3) with columns [pi, mu, r]."""
        raw_output = self.linear(feature_input)  # (batch, categories*3)
        reshaped = raw_output.view(-1, self.num_categories, 3)

        # Apply activations to keep parameters in valid ranges
        pi_logit = torch.sigmoid(reshaped[..., 0])  # (0, 1)
        mu_positive = torch.exp(reshaped[..., 1])  # (0, ∞)
        r_positive = torch.exp(reshaped[..., 2])  # (0, ∞)

        return torch.stack([pi_logit, mu_positive, r_positive], dim=-1)


# ---------------------------------------------------------------------------
# ZINB negative log-likelihood (inline, for smoke-test independence)
# ---------------------------------------------------------------------------


def _zinb_nll(
    counts: Tensor,
    pi: Tensor,
    mu: Tensor,
    r: Tensor,
) -> Tensor:
    """Compute ZINB negative log-likelihood.

    Uses the log-sum-exp trick for numerical stability at zero counts.
    """
    eps = 1e-8
    counts_float = counts.float()

    # Negative-binomial log-prob
    log_nb = (
        torch.lgamma(counts_float + r)
        - torch.lgamma(r)
        - torch.lgamma(counts_float + 1)
        + r * torch.log(r / (r + mu) + eps)
        + counts_float * torch.log(mu / (r + mu) + eps)
    )

    # Zero-inflated mixture
    is_zero = (counts_float == 0).float()
    log_pi = torch.log(pi + eps)
    log_one_minus_pi = torch.log(1 - pi + eps)

    # For zero counts: log(pi + (1-pi)*NB(0))
    # For non-zero counts: log(1-pi) + log_nb
    log_likelihood_zero = torch.logsumexp(
        torch.stack([log_pi, log_one_minus_pi + log_nb], dim=-1),
        dim=-1,
    )
    log_likelihood_nonzero = log_one_minus_pi + log_nb

    log_likelihood = (
        is_zero * log_likelihood_zero + (1 - is_zero) * log_likelihood_nonzero
    )

    return -log_likelihood.mean()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

_SMOKE_SPATIAL: int = 5
_SMOKE_TIME: int = 10
_SMOKE_CATEGORIES: int = 2
_SMOKE_FEATURES: int = 3


def test_smoke_synthetic_pipeline() -> None:
    """Full pipeline: synthetic data → model → ZINB NLL → backward.

    Verifies:
    1. Panel generation produces correctly shaped tensors
    2. Model output has shape (spatial * time, categories, 3)
    3. ZINB NLL loss is finite
    4. Gradients flow back to model parameters
    """
    # ── Step 1: Generate synthetic panel ──────────────────────────
    panel: dict[str, Tensor] = generate_spatiotemporal_panel(
        num_spatial_units=_SMOKE_SPATIAL,
        num_time_steps=_SMOKE_TIME,
        num_categories=_SMOKE_CATEGORIES,
        num_features=_SMOKE_FEATURES,
        seed=42,
    )

    observed_counts = panel["counts"]  # (S, T, C)
    feature_matrix = panel["features"]  # (S, T, F)

    assert observed_counts.shape == (
        _SMOKE_SPATIAL,
        _SMOKE_TIME,
        _SMOKE_CATEGORIES,
    ), f"Unexpected counts shape: {observed_counts.shape}"
    assert feature_matrix.shape == (
        _SMOKE_SPATIAL,
        _SMOKE_TIME,
        _SMOKE_FEATURES,
    ), f"Unexpected features shape: {feature_matrix.shape}"

    # ── Step 2: Forward pass through tiny model ───────────────────
    model = _TinyZINBHead(
        num_features=_SMOKE_FEATURES,
        num_categories=_SMOKE_CATEGORIES,
    )

    # Flatten spatial × time into a single batch dimension
    batch_size = _SMOKE_SPATIAL * _SMOKE_TIME
    flat_features = feature_matrix.reshape(batch_size, _SMOKE_FEATURES)

    model_output = model(flat_features)  # (batch, C, 3)

    assert model_output.shape == (batch_size, _SMOKE_CATEGORIES, 3), (
        f"Model output shape {model_output.shape} != "
        f"expected ({batch_size}, {_SMOKE_CATEGORIES}, 3)"
    )

    predicted_pi = model_output[..., 0]  # (batch, C)
    predicted_mu = model_output[..., 1]
    predicted_r = model_output[..., 2]

    # ── Step 3: Compute ZINB NLL loss ─────────────────────────────
    flat_counts = observed_counts.reshape(batch_size, _SMOKE_CATEGORIES)

    nll_loss = _zinb_nll(flat_counts, predicted_pi, predicted_mu, predicted_r)

    assert torch.isfinite(nll_loss), f"ZINB NLL loss is non-finite: {nll_loss.item()}"

    # ── Step 4: Backward pass — gradient flow ─────────────────────
    nll_loss.backward()

    for param_name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for parameter '{param_name}'"
        assert torch.isfinite(
            param.grad
        ).all(), f"Non-finite gradient for parameter '{param_name}'"
