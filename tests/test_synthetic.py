"""
Tests for civicsafe.synthetic.distributions — ZINB sampling statistics,
Poisson special case, parameter validation, and panel data properties.
"""
from __future__ import annotations

import pytest
import torch
from torch import Tensor

from civicsafe.synthetic.distributions import (
    generate_poisson_samples,
    generate_spatiotemporal_panel,
    generate_zinb_samples,
)

# Number of draws large enough for the law of large numbers to kick in
_LARGE_SAMPLE_SIZE: int = 100_000

# Relative tolerance for statistical assertions
_REL_TOL: float = 0.05


# ===================================================================
# ZINB — Zero-Inflated Negative Binomial
# ===================================================================

class TestZINBSampling:
    """Statistical and correctness tests for the ZINB sampler."""

    # Known parameters for the test distribution
    ZERO_INFLATION_PROB: float = 0.3
    NB_MEAN: float = 5.0
    NB_DISPERSION: float = 2.0

    def test_zinb_mean(self) -> None:
        """Sample mean must approximate (1 - pi) * mu within 5 %."""
        zinb_samples, _true_params = generate_zinb_samples(
            num_samples=_LARGE_SAMPLE_SIZE,
            pi=self.ZERO_INFLATION_PROB,
            mu=self.NB_MEAN,
            r=self.NB_DISPERSION,
            seed=42,
        )
        empirical_mean = zinb_samples.float().mean().item()
        theoretical_mean = (1 - self.ZERO_INFLATION_PROB) * self.NB_MEAN

        assert empirical_mean == pytest.approx(theoretical_mean, rel=_REL_TOL), (
            f"ZINB sample mean {empirical_mean:.4f} deviates from "
            f"theoretical {theoretical_mean:.4f} by more than {_REL_TOL*100}%"
        )

    def test_zinb_zeros(self) -> None:
        """Fraction of zeros ≈ pi + (1-pi)*(r/(r+mu))^r within 5 %."""
        zinb_samples, _true_params = generate_zinb_samples(
            num_samples=_LARGE_SAMPLE_SIZE,
            pi=self.ZERO_INFLATION_PROB,
            mu=self.NB_MEAN,
            r=self.NB_DISPERSION,
            seed=42,
        )
        empirical_zero_fraction = (zinb_samples == 0).float().mean().item()

        pi = self.ZERO_INFLATION_PROB
        mu = self.NB_MEAN
        r = self.NB_DISPERSION
        nb_zero_prob = (r / (r + mu)) ** r
        theoretical_zero_fraction = pi + (1 - pi) * nb_zero_prob

        assert empirical_zero_fraction == pytest.approx(
            theoretical_zero_fraction, rel=_REL_TOL
        ), (
            f"Zero fraction {empirical_zero_fraction:.4f} deviates from "
            f"theoretical {theoretical_zero_fraction:.4f}"
        )

    def test_zinb_deterministic(self) -> None:
        """Identical seeds must produce identical ZINB sample tensors."""
        shared_kwargs = dict(
            num_samples=500,
            pi=self.ZERO_INFLATION_PROB,
            mu=self.NB_MEAN,
            r=self.NB_DISPERSION,
        )
        zinb_draw_first, _ = generate_zinb_samples(**shared_kwargs, seed=42)
        zinb_draw_second, _ = generate_zinb_samples(**shared_kwargs, seed=42)

        assert torch.equal(zinb_draw_first, zinb_draw_second), (
            "ZINB sampler is not deterministic for the same seed"
        )

    @pytest.mark.parametrize(
        "invalid_pi, invalid_mu, invalid_r",
        [
            (-0.1, 5.0, 2.0),
            (0.3, -1.0, 2.0),
            (0.3, 5.0, 0.0),
        ],
        ids=["negative_pi", "negative_mu", "zero_r"],
    )
    def test_zinb_parameter_validation(
        self,
        invalid_pi: float,
        invalid_mu: float,
        invalid_r: float,
    ) -> None:
        """Out-of-range ZINB parameters must raise AssertionError."""
        with pytest.raises(AssertionError):
            generate_zinb_samples(
                num_samples=10,
                pi=invalid_pi,
                mu=invalid_mu,
                r=invalid_r,
                seed=42,
            )


# ===================================================================
# Poisson — a special case of ZINB with pi=0, r→∞
# ===================================================================

class TestPoissonSampling:
    """Verify the Poisson sampler recovers known rate statistics."""

    POISSON_RATE: float = 7.0

    def test_poisson_mean(self) -> None:
        """Sample mean must approximate the Poisson rate within 5 %."""
        poisson_samples, _true_params = generate_poisson_samples(
            num_samples=_LARGE_SAMPLE_SIZE,
            rate=self.POISSON_RATE,
            seed=42,
        )
        empirical_mean = poisson_samples.float().mean().item()

        assert empirical_mean == pytest.approx(self.POISSON_RATE, rel=_REL_TOL), (
            f"Poisson sample mean {empirical_mean:.4f} deviates from "
            f"rate {self.POISSON_RATE} by more than {_REL_TOL*100}%"
        )


# ===================================================================
# Spatiotemporal panel — shape, symmetry, non-negativity
# ===================================================================

_PANEL_SPATIAL: int = 5
_PANEL_TIME: int = 10
_PANEL_CATEGORIES: int = 2
_PANEL_FEATURES: int = 3


class TestPanelData:
    """Structural tests on the generated spatiotemporal panel."""

    def test_panel_shapes(self, tiny_panel: dict[str, Tensor]) -> None:
        """All tensors in the panel must have the documented shapes."""
        counts = tiny_panel["counts"]
        features = tiny_panel["features"]
        adjacency = tiny_panel["adjacency"]

        assert counts.shape == (
            _PANEL_SPATIAL,
            _PANEL_TIME,
            _PANEL_CATEGORIES,
        ), f"counts shape {counts.shape} != expected ({_PANEL_SPATIAL}, {_PANEL_TIME}, {_PANEL_CATEGORIES})"

        assert features.shape == (
            _PANEL_SPATIAL,
            _PANEL_TIME,
            _PANEL_FEATURES,
        ), f"features shape {features.shape} != expected ({_PANEL_SPATIAL}, {_PANEL_TIME}, {_PANEL_FEATURES})"

        assert adjacency.shape == (
            _PANEL_SPATIAL,
            _PANEL_SPATIAL,
        ), f"adjacency shape {adjacency.shape} != expected ({_PANEL_SPATIAL}, {_PANEL_SPATIAL})"

    def test_panel_adjacency_symmetric(self, tiny_panel: dict[str, Tensor]) -> None:
        """The spatial adjacency matrix must be symmetric."""
        adjacency = tiny_panel["adjacency"]
        assert torch.equal(adjacency, adjacency.T), (
            "Adjacency matrix is not symmetric"
        )

    def test_panel_counts_nonnegative(self, tiny_panel: dict[str, Tensor]) -> None:
        """All count values must be non-negative (crime counts ≥ 0)."""
        counts = tiny_panel["counts"]
        assert (counts >= 0).all(), (
            f"Negative count values found: min = {counts.min().item()}"
        )
