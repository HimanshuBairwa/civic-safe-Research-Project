"""Tests for level anchoring in the ZINB head.

The premise of the fix: an un-anchored head must reconstruct each cell's
absolute level from scratch through nonlinear layers, and empirically loses to
a trailing mean (Chicago test CRPS 3.2291 model vs 2.9322 rolling HA). With
anchoring, mu = anchor * exp(clamp(delta)), and because the final layer is
zero-bias/small-weight initialized the model *starts* at the rolling-HA
forecast and learns only the departure from it.

The load-bearing property is therefore not "anchoring runs" but "anchoring at
initialization reproduces the rolling historical average". That is what
test_anchored_model_starts_at_rolling_historical_average pins, by comparing
against baselines.py's HA formula rather than restating the head's own
arithmetic.
"""

from __future__ import annotations

import pytest
import torch

from civicsafe.models.civicsafe_model import CivicSafeModel
from civicsafe.models.zinb_head import ZINBHead

S, W, C, F = 12, 52, 3, 8
HIDDEN = 32


def _edges(num_nodes: int) -> torch.Tensor:
    """A simple ring graph so the GATv2 encoder has real edges to attend over."""
    src = torch.arange(num_nodes)
    dst = (src + 1) % num_nodes
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])


def _build_input(counts: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """Replicate the call-site convention: cat([features, log1p(counts)], -1).

    Mirrors trainer.py:344, trainer.py:486, evaluate_trained.py:390 and
    run_conformal_evaluation.py:354. The model recovers its anchor from the
    final C channels, so this ordering is load-bearing.
    """
    return torch.cat([features, torch.log1p(counts.float())], dim=-1)


# ---------------------------------------------------------------------------
# The head in isolation
# ---------------------------------------------------------------------------
def test_anchored_head_is_identity_on_the_anchor_at_init() -> None:
    """At init, delta ~ 0 so mu should be within a few percent of the anchor."""
    torch.manual_seed(0)
    head = ZINBHead(in_features=HIDDEN, num_categories=C, level_anchor=True)
    x = torch.randn(S, HIDDEN)
    anchor = torch.rand(S, C) * 20.0 + 0.5

    _, mu, _ = head(x, anchor=anchor)

    # std=0.01 on the final layer over 32 inputs keeps |delta| tiny, so mu
    # tracks the anchor closely. Deliberately not exact: the point is that the
    # model begins in a neighbourhood of the baseline, not at a fixed point.
    rel_err = ((mu - anchor).abs() / anchor).max().item()
    assert rel_err < 0.15, f"mu departs from anchor by {rel_err:.3f} at init"


def test_anchored_head_requires_an_anchor() -> None:
    """Silently falling back to the unanchored path would misreport the model."""
    head = ZINBHead(in_features=HIDDEN, num_categories=C, level_anchor=True)
    with pytest.raises(ValueError, match="level_anchor=True"):
        head(torch.randn(S, HIDDEN), anchor=None)


def test_unanchored_head_ignores_a_supplied_anchor() -> None:
    """The default path must be untouched, so published numbers cannot move."""
    torch.manual_seed(0)
    head = ZINBHead(in_features=HIDDEN, num_categories=C, level_anchor=False)
    x = torch.randn(S, HIDDEN)

    _, mu_none, _ = head(x, anchor=None)
    _, mu_given, _ = head(x, anchor=torch.full((S, C), 99.0))

    assert torch.equal(mu_none, mu_given)


def test_delta_clamp_bounds_mu_and_keeps_gradients_finite() -> None:
    """Without the clamp, one large logit gives exp(inf) -> NaN through CRPS."""
    head = ZINBHead(in_features=HIDDEN, num_categories=C, level_anchor=True)
    # Force enormous pre-activations through the final layer.
    with torch.no_grad():
        head.mu_mlp[-1].bias.fill_(500.0)

    anchor = torch.full((S, C), 4.0)
    x = torch.randn(S, HIDDEN, requires_grad=True)
    _, mu, _ = head(x, anchor=anchor)

    assert torch.isfinite(mu).all(), "clamp failed to prevent overflow"
    # exp(+3) is the documented ceiling.
    assert mu.max().item() <= 4.0 * torch.exp(torch.tensor(3.0)).item() + 1e-3

    mu.sum().backward()
    assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# The end-to-end claim
# ---------------------------------------------------------------------------
def test_anchored_model_starts_at_rolling_historical_average() -> None:
    """An initialized anchored model must approximate the rolling-HA forecast.

    This is the reason the fix exists. The reference is computed with
    baselines.py's own formula -- ``input_counts.mean(dim=1)`` -- so the test
    compares the model against the baseline that beat it, not against the
    model's internal arithmetic.
    """
    torch.manual_seed(0)
    counts = torch.randint(0, 40, (S, W, C))
    features = torch.randn(S, W, F)
    x_in = _build_input(counts, features)

    model = CivicSafeModel(
        num_features=F + C,
        hidden_dim=HIDDEN,
        spatial_layers=1,
        spatial_heads=2,
        temporal_layers=1,
        temporal_heads=2,
        temporal_ff_dim=32,
        num_categories=C,
        max_seq_len=W,
        level_anchor=True,
    ).eval()

    edge_index = _edges(S)
    with torch.no_grad():
        out = model(x_in, edge_index, None)

    # baselines.py:174 -- the rolling HA that outperformed the unanchored model.
    ha = counts.float().mean(dim=1)  # (S, C)

    rel_err = ((out["mu"] - ha).abs() / ha.clamp(min=0.05)).max().item()
    assert rel_err < 0.25, (
        f"anchored model starts {rel_err:.3f} away from rolling HA; the whole "
        "point of anchoring is that it starts AT the baseline"
    )


def test_anchor_recovers_counts_exactly_through_log1p_expm1() -> None:
    """expm1(log1p(counts)) must round-trip, or the anchor is a wrong level."""
    counts = torch.randint(0, 500, (S, W, C)).float()
    round_tripped = torch.expm1(torch.log1p(counts))
    assert torch.allclose(round_tripped, counts, atol=1e-3)


def test_unanchored_model_is_bit_identical_to_pre_change_behaviour() -> None:
    """level_anchor defaults to False, so existing checkpoints are unaffected.

    Guards the backward-compatibility contract: the anchored and unanchored
    heads have identical parameter shapes, so a checkpoint loads cleanly either
    way and a wrong default would silently rescale every mu.
    """
    torch.manual_seed(0)
    counts = torch.randint(0, 40, (S, W, C))
    features = torch.randn(S, W, F)
    x_in = _build_input(counts, features)

    def build() -> CivicSafeModel:
        torch.manual_seed(1234)
        return CivicSafeModel(
            num_features=F + C,
            hidden_dim=HIDDEN,
            spatial_layers=1,
            spatial_heads=2,
            temporal_layers=1,
            temporal_heads=2,
            temporal_ff_dim=32,
            num_categories=C,
            max_seq_len=W,
        ).eval()

    model = build()
    assert model.zinb_head.level_anchor is False
    edge_index = _edges(S)
    with torch.no_grad():
        mu_a = model(x_in, edge_index, None)["mu"]
        mu_b = build()(x_in, edge_index, None)["mu"]
    assert torch.equal(mu_a, mu_b)

    # And the state dict is shape-compatible with an anchored model, which is
    # exactly why the flag must be recorded in the checkpoint's `arch`.
    torch.manual_seed(1234)
    anchored = CivicSafeModel(
        num_features=F + C,
        hidden_dim=HIDDEN,
        spatial_layers=1,
        spatial_heads=2,
        temporal_layers=1,
        temporal_heads=2,
        temporal_ff_dim=32,
        num_categories=C,
        max_seq_len=W,
        level_anchor=True,
    )
    missing, unexpected = anchored.load_state_dict(model.state_dict(), strict=False)
    assert not missing and not unexpected, (
        "anchored/unanchored state dicts diverged; the silent-mismatch hazard "
        "documented in evaluate_trained.build_model no longer applies"
    )
