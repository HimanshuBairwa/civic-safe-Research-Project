"""Tests for the conformal calibration module.

Covers:
    - ZINB distribution utilities (CDF, PPF, quantile pairs)
    - All 5 calibration methods (Split CP, Weighted CP, Mondrian, Equalized, ECRC)
    - Calibration evaluation metrics (PICP, AIW, Winkler Score)
    - Coverage guarantees (PICP ≥ 1-α on synthetic data)
    - Edge cases (all-zero observations, extreme parameters)
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from civicsafe.calibration.conformal import (
    ECRCCalibrator,
    EqualizedCoverageCalibrator,
    MondrianConformalCalibrator,
    SplitConformalCalibrator,
    WeightedConformalCalibrator,
    compute_cqr_scores,
    create_calibrator,
)
from civicsafe.calibration.metrics import (
    average_interval_width,
    compute_all_calibration_metrics,
    conditional_coverage,
    coverage_gap,
    picp,
    winkler_score,
)
from civicsafe.calibration.zinb_distribution import (
    zinb_cdf,
    zinb_cdf_full,
    zinb_ppf,
    zinb_ppf_pair,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def synthetic_zinb_data() -> dict[str, Tensor]:
    """Generate synthetic ZINB calibration data.

    Creates 1000 samples from known ZINB parameters, then uses those
    same parameters as the "predicted" parameters. This way we know the
    model is perfectly specified, and coverage should be ≥ 1-α.
    """
    torch.manual_seed(42)
    N = 1000
    pi = torch.full((N,), 0.3)
    mu = torch.full((N,), 10.0)
    r = torch.full((N,), 5.0)

    # Sample from ZINB: Y = 0 w.p. π, else Y ~ NB(r, r/(r+μ))
    is_zero = torch.bernoulli(pi).bool()
    p = r / (r + mu)
    nb_samples = torch.distributions.NegativeBinomial(
        total_count=r, probs=p
    ).sample()
    y = torch.where(is_zero, torch.zeros_like(nb_samples), nb_samples)

    return {"y": y, "pi": pi, "mu": mu, "r": r}


@pytest.fixture
def grouped_data(synthetic_zinb_data: dict[str, Tensor]) -> dict[str, Tensor]:
    """Add group labels to synthetic data (4 groups)."""
    N = synthetic_zinb_data["y"].shape[0]
    groups = torch.arange(N) % 4  # 4 groups, cycling
    return {**synthetic_zinb_data, "groups": groups}


# ============================================================
# ZINB Distribution Tests
# ============================================================

class TestZINBDistribution:
    """Tests for zinb_distribution.py utilities."""

    def test_zinb_cdf_monotonic(self) -> None:
        """CDF must be non-decreasing."""
        pi = torch.tensor([0.3, 0.0, 0.5])
        mu = torch.tensor([10.0, 5.0, 20.0])
        r = torch.tensor([5.0, 2.0, 10.0])

        _, F = zinb_cdf_full(pi, mu, r)

        # Check monotonicity: F[:, k+1] >= F[:, k]
        diffs = F[:, 1:] - F[:, :-1]
        assert (diffs >= -1e-6).all(), "CDF must be non-decreasing"

    def test_zinb_cdf_range(self) -> None:
        """CDF must be in [0, 1]."""
        pi = torch.tensor([0.3])
        mu = torch.tensor([10.0])
        r = torch.tensor([5.0])

        _, F = zinb_cdf_full(pi, mu, r)
        assert (F >= -1e-6).all() and (F <= 1.0 + 1e-6).all()

    def test_zinb_cdf_at_zero(self) -> None:
        """F_ZINB(0) = π + (1-π)·F_NB(0)."""
        pi = torch.tensor([0.3])
        mu = torch.tensor([10.0])
        r = torch.tensor([5.0])

        F_at_0 = zinb_cdf(0, pi, mu, r)

        # F_NB(0) = P(X=0) = (r/(r+mu))^r
        p = r / (r + mu)
        F_nb_0 = p ** r
        expected = pi + (1.0 - pi) * F_nb_0

        assert torch.allclose(F_at_0, expected, atol=1e-4)

    def test_zinb_cdf_zero_inflation_effect(self) -> None:
        """Higher π should increase CDF at all points."""
        mu = torch.tensor([10.0, 10.0])
        r = torch.tensor([5.0, 5.0])
        pi_low = torch.tensor([0.1, 0.1])
        pi_high = torch.tensor([0.5, 0.5])

        F_low = zinb_cdf(5, pi_low, mu, r)
        F_high = zinb_cdf(5, pi_high, mu, r)

        assert (F_high >= F_low).all(), "Higher π → higher CDF"

    def test_zinb_ppf_inverse_of_cdf(self) -> None:
        """PPF(F(k)) should return k (approximately, due to discreteness)."""
        pi = torch.tensor([0.3])
        mu = torch.tensor([10.0])
        r = torch.tensor([5.0])

        k = 7
        F_at_k = zinb_cdf(k, pi, mu, r)

        # PPF at CDF(k) should return k (or k-1 due to step function)
        recovered_k = zinb_ppf(F_at_k, pi, mu, r)
        assert abs(recovered_k.item() - k) <= 1, (
            f"PPF(CDF({k})) = {recovered_k.item()}, expected within 1 of {k}"
        )

    def test_zinb_ppf_nonnegative(self) -> None:
        """Quantiles must be ≥ 0 (crime counts are non-negative)."""
        pi = torch.tensor([0.5, 0.0, 0.9])
        mu = torch.tensor([1.0, 50.0, 0.1])
        r = torch.tensor([1.0, 10.0, 0.5])

        q = torch.tensor([0.01, 0.5, 0.99])
        ppf = zinb_ppf(q, pi, mu, r)
        assert (ppf >= 0).all()

    def test_zinb_ppf_pair_ordering(self) -> None:
        """q_low ≤ q_high always."""
        pi = torch.rand(100)
        mu = torch.rand(100) * 30 + 1.0
        r = torch.rand(100) * 10 + 0.5

        q_low, q_high = zinb_ppf_pair(0.1, pi, mu, r)
        assert (q_high >= q_low).all(), "Upper quantile must be ≥ lower"

    def test_zinb_ppf_pair_wider_at_lower_alpha(self) -> None:
        """Smaller α → wider intervals."""
        pi = torch.tensor([0.3])
        mu = torch.tensor([10.0])
        r = torch.tensor([5.0])

        q_low_90, q_high_90 = zinb_ppf_pair(0.1, pi, mu, r)  # 90%
        q_low_80, q_high_80 = zinb_ppf_pair(0.2, pi, mu, r)  # 80%

        width_90 = q_high_90 - q_low_90
        width_80 = q_high_80 - q_low_80

        assert width_90 >= width_80, "90% interval should be wider than 80%"


# ============================================================
# Non-Conformity Score Tests
# ============================================================

class TestCQRScores:
    """Tests for CQR non-conformity score computation."""

    def test_scores_shape(self, synthetic_zinb_data: dict[str, Tensor]) -> None:
        """Scores should have same shape as y."""
        d = synthetic_zinb_data
        scores = compute_cqr_scores(d["y"], d["pi"], d["mu"], d["r"])
        assert scores.shape == d["y"].shape

    def test_scores_finite(self, synthetic_zinb_data: dict[str, Tensor]) -> None:
        """All scores should be finite."""
        d = synthetic_zinb_data
        scores = compute_cqr_scores(d["y"], d["pi"], d["mu"], d["r"])
        assert torch.isfinite(scores).all()

    def test_inside_observations_have_negative_scores(self) -> None:
        """Points inside [q_low, q_high] should have negative scores."""
        # Use a point that's exactly the mean — should be inside
        pi = torch.tensor([0.0])
        mu = torch.tensor([10.0])
        r = torch.tensor([100.0])  # Tight distribution around 10
        y = torch.tensor([10.0])

        scores = compute_cqr_scores(y, pi, mu, r, alpha=0.1)
        # With tight distribution, mean should be well inside
        assert scores.item() <= 0, "Mean should be inside heuristic interval"


# ============================================================
# Split Conformal Calibrator Tests
# ============================================================

class TestSplitConformal:
    """Tests for standard split conformal prediction."""

    def test_fit_sets_threshold(
        self, synthetic_zinb_data: dict[str, Tensor]
    ) -> None:
        """After fit(), threshold should be set."""
        d = synthetic_zinb_data
        cal = SplitConformalCalibrator(alpha=0.1)
        cal.fit(d["y"], d["pi"], d["mu"], d["r"])
        assert cal._fitted
        assert isinstance(cal.threshold, float)

    def test_predict_shapes(
        self, synthetic_zinb_data: dict[str, Tensor]
    ) -> None:
        """Predict should return lower, upper, point with correct shapes."""
        d = synthetic_zinb_data
        cal = SplitConformalCalibrator(alpha=0.1)
        cal.fit(d["y"], d["pi"], d["mu"], d["r"])
        result = cal.predict(d["pi"], d["mu"], d["r"])

        assert result["lower"].shape == d["pi"].shape
        assert result["upper"].shape == d["pi"].shape
        assert result["point"].shape == d["pi"].shape

    def test_lower_le_upper(
        self, synthetic_zinb_data: dict[str, Tensor]
    ) -> None:
        """Lower bound ≤ upper bound always."""
        d = synthetic_zinb_data
        cal = SplitConformalCalibrator(alpha=0.1)
        cal.fit(d["y"], d["pi"], d["mu"], d["r"])
        result = cal.predict(d["pi"], d["mu"], d["r"])

        assert (result["upper"] >= result["lower"]).all()

    def test_lower_nonnegative(
        self, synthetic_zinb_data: dict[str, Tensor]
    ) -> None:
        """Lower bound ≥ 0 always (crime counts cannot be negative)."""
        d = synthetic_zinb_data
        cal = SplitConformalCalibrator(alpha=0.1)
        cal.fit(d["y"], d["pi"], d["mu"], d["r"])
        result = cal.predict(d["pi"], d["mu"], d["r"])

        assert (result["lower"] >= 0).all()

    def test_coverage_guarantee(
        self, synthetic_zinb_data: dict[str, Tensor]
    ) -> None:
        """PICP should be ≥ 1-α on well-specified data.

        Since the model is perfectly specified (we use the true parameters),
        coverage should easily exceed the target.
        """
        d = synthetic_zinb_data
        alpha = 0.1

        # Split data: first half for calibration, second for test
        N = d["y"].shape[0]
        mid = N // 2

        cal = SplitConformalCalibrator(alpha=alpha)
        cal.fit(
            d["y"][:mid], d["pi"][:mid], d["mu"][:mid], d["r"][:mid]
        )
        result = cal.predict(d["pi"][mid:], d["mu"][mid:], d["r"][mid:])

        coverage = picp(d["y"][mid:], result["lower"], result["upper"])
        assert coverage >= 1.0 - alpha - 0.05, (
            f"Coverage {coverage:.3f} < {1 - alpha - 0.05:.3f}"
        )

    def test_predict_before_fit_raises(self) -> None:
        """Predict without fit() should raise RuntimeError."""
        cal = SplitConformalCalibrator(alpha=0.1)
        pi = torch.tensor([0.3])
        mu = torch.tensor([10.0])
        r = torch.tensor([5.0])

        with pytest.raises(RuntimeError, match="not been fitted"):
            cal.predict(pi, mu, r)

    def test_invalid_alpha_raises(self) -> None:
        """Invalid alpha values should raise ValueError."""
        with pytest.raises(ValueError):
            SplitConformalCalibrator(alpha=0.0)
        with pytest.raises(ValueError):
            SplitConformalCalibrator(alpha=0.6)


# ============================================================
# Weighted Conformal Tests
# ============================================================

class TestWeightedConformal:
    """Tests for temporally-weighted conformal prediction."""

    def test_fit_and_predict(
        self, synthetic_zinb_data: dict[str, Tensor]
    ) -> None:
        """Fit + predict should work without error."""
        d = synthetic_zinb_data
        cal = WeightedConformalCalibrator(alpha=0.1, decay_rate=0.05)
        cal.fit(d["y"], d["pi"], d["mu"], d["r"])
        result = cal.predict(d["pi"], d["mu"], d["r"])

        assert result["lower"].shape == d["y"].shape
        assert (result["lower"] >= 0).all()
        assert (result["upper"] >= result["lower"]).all()

    def test_coverage_with_decay(
        self, synthetic_zinb_data: dict[str, Tensor]
    ) -> None:
        """Weighted CP should produce valid coverage on well-specified data."""
        d = synthetic_zinb_data
        N = d["y"].shape[0]
        mid = N // 2

        cal = WeightedConformalCalibrator(alpha=0.1, decay_rate=0.05)
        cal.fit(d["y"][:mid], d["pi"][:mid], d["mu"][:mid], d["r"][:mid])

        result = cal.predict(d["pi"][mid:], d["mu"][mid:], d["r"][mid:])
        coverage = picp(d["y"][mid:], result["lower"], result["upper"])

        assert coverage >= 0.85, f"Coverage {coverage:.3f} too low"


# ============================================================
# Mondrian Conformal Tests
# ============================================================

class TestMondrianConformal:
    """Tests for group-conditional conformal prediction."""

    def test_fit_and_predict(self, grouped_data: dict[str, Tensor]) -> None:
        """Fit + predict with groups should work."""
        d = grouped_data
        cal = MondrianConformalCalibrator(alpha=0.1, min_group_size=20)
        cal.fit(d["y"], d["pi"], d["mu"], d["r"], groups=d["groups"])
        result = cal.predict(
            d["pi"], d["mu"], d["r"], groups=d["groups"]
        )

        assert result["lower"].shape == d["y"].shape
        assert (result["lower"] >= 0).all()

    def test_per_group_thresholds_stored(
        self, grouped_data: dict[str, Tensor]
    ) -> None:
        """Each group should have its own threshold."""
        d = grouped_data
        cal = MondrianConformalCalibrator(alpha=0.1, min_group_size=20)
        cal.fit(d["y"], d["pi"], d["mu"], d["r"], groups=d["groups"])

        # Should have 4 groups
        assert len(cal._group_thresholds) == 4

    def test_small_groups_fallback_to_global(self) -> None:
        """Groups smaller than min_group_size should use global threshold."""
        N = 100
        y = torch.randint(0, 20, (N,)).float()
        pi = torch.full((N,), 0.2)
        mu = torch.full((N,), 10.0)
        r = torch.full((N,), 5.0)
        # Make one group very small (only 5 samples)
        groups = torch.zeros(N, dtype=torch.long)
        groups[:5] = 99  # Tiny group

        cal = MondrianConformalCalibrator(alpha=0.1, min_group_size=40)
        cal.fit(y, pi, mu, r, groups=groups)

        # The tiny group should fallback to global threshold
        assert cal._group_thresholds[99] == cal._global_threshold


# ============================================================
# Equalized Coverage Tests
# ============================================================

class TestEqualizedCoverage:
    """Tests for equalized coverage conformal prediction."""

    def test_fit_and_predict(self, grouped_data: dict[str, Tensor]) -> None:
        """Fit + predict should work."""
        d = grouped_data
        cal = EqualizedCoverageCalibrator(alpha=0.1, lambda_eq=1.0)
        cal.fit(
            d["y"], d["pi"], d["mu"], d["r"], groups=d["groups"]
        )
        result = cal.predict(d["pi"], d["mu"], d["r"])

        assert result["lower"].shape == d["y"].shape
        assert (result["lower"] >= 0).all()


# ============================================================
# ECRC Tests
# ============================================================

class TestECRC:
    """Tests for Equalized Conditional Risk Control."""

    def test_fit_and_predict(self, grouped_data: dict[str, Tensor]) -> None:
        """ECRC fit + predict should work."""
        d = grouped_data
        cal = ECRCCalibrator(alpha=0.1, delta=0.05)
        cal.fit(d["y"], d["pi"], d["mu"], d["r"], groups=d["groups"])
        result = cal.predict(
            d["pi"], d["mu"], d["r"], groups=d["groups"]
        )

        assert result["lower"].shape == d["y"].shape
        assert (result["lower"] >= 0).all()

    def test_epsilon_computed(self, grouped_data: dict[str, Tensor]) -> None:
        """Hoeffding epsilon should be > 0."""
        d = grouped_data
        cal = ECRCCalibrator(alpha=0.1, delta=0.05)
        cal.fit(d["y"], d["pi"], d["mu"], d["r"], groups=d["groups"])

        assert cal.epsilon > 0

    def test_smaller_delta_gives_larger_epsilon(
        self, grouped_data: dict[str, Tensor]
    ) -> None:
        """Smaller δ → larger ε (stronger guarantee → wider intervals)."""
        d = grouped_data

        cal_loose = ECRCCalibrator(alpha=0.1, delta=0.1)
        cal_tight = ECRCCalibrator(alpha=0.1, delta=0.01)

        cal_loose.fit(
            d["y"], d["pi"], d["mu"], d["r"], groups=d["groups"]
        )
        cal_tight.fit(
            d["y"], d["pi"], d["mu"], d["r"], groups=d["groups"]
        )

        assert cal_tight.epsilon > cal_loose.epsilon


# ============================================================
# Factory Tests
# ============================================================

class TestCreateCalibrator:
    """Tests for the config → calibrator factory."""

    def test_split_cp(self) -> None:
        """Factory should create SplitConformalCalibrator."""
        cfg = {"calibration": {"method": "split_cp", "alpha": 0.1}}
        cal = create_calibrator(cfg)
        assert isinstance(cal, SplitConformalCalibrator)

    def test_weighted_cp(self) -> None:
        """Factory should create WeightedConformalCalibrator."""
        cfg = {
            "calibration": {
                "method": "weighted_cp",
                "alpha": 0.1,
                "decay_rate": 0.05,
            }
        }
        cal = create_calibrator(cfg)
        assert isinstance(cal, WeightedConformalCalibrator)

    def test_mondrian(self) -> None:
        """Factory should create MondrianConformalCalibrator."""
        cfg = {
            "calibration": {"method": "mondrian", "alpha": 0.1, "min_group_size": 40}
        }
        cal = create_calibrator(cfg)
        assert isinstance(cal, MondrianConformalCalibrator)

    def test_equalized(self) -> None:
        """Factory should create EqualizedCoverageCalibrator."""
        cfg = {
            "calibration": {
                "method": "equalized_coverage",
                "alpha": 0.1,
                "lambda_eq": 1.0,
            }
        }
        cal = create_calibrator(cfg)
        assert isinstance(cal, EqualizedCoverageCalibrator)

    def test_ecrc(self) -> None:
        """Factory should create ECRCCalibrator."""
        cfg = {
            "calibration": {
                "method": "ecrc",
                "alpha": 0.1,
                "delta": 0.05,
            }
        }
        cal = create_calibrator(cfg)
        assert isinstance(cal, ECRCCalibrator)

    def test_unknown_method_raises(self) -> None:
        """Unknown method should raise ValueError."""
        cfg = {"calibration": {"method": "nonexistent"}}
        with pytest.raises(ValueError, match="Unknown calibration method"):
            create_calibrator(cfg)


# ============================================================
# Calibration Metrics Tests
# ============================================================

class TestCalibrationMetrics:
    """Tests for PICP, AIW, and Winkler Score."""

    def test_picp_perfect_coverage(self) -> None:
        """100% coverage when interval contains all observations."""
        y = torch.tensor([3.0, 5.0, 7.0, 10.0])
        lower = torch.tensor([0.0, 0.0, 0.0, 0.0])
        upper = torch.tensor([20.0, 20.0, 20.0, 20.0])

        assert picp(y, lower, upper) == 1.0

    def test_picp_zero_coverage(self) -> None:
        """0% coverage when all observations are outside."""
        y = torch.tensor([10.0, 20.0, 30.0])
        lower = torch.tensor([0.0, 0.0, 0.0])
        upper = torch.tensor([1.0, 1.0, 1.0])

        assert picp(y, lower, upper) == 0.0

    def test_picp_partial_coverage(self) -> None:
        """Partial coverage should be exact."""
        y = torch.tensor([5.0, 15.0, 25.0, 35.0])
        lower = torch.tensor([0.0, 0.0, 0.0, 0.0])
        upper = torch.tensor([10.0, 10.0, 10.0, 10.0])

        # Only y=5 is in [0, 10]
        assert picp(y, lower, upper) == 0.25

    def test_aiw_correct(self) -> None:
        """AIW = mean(upper - lower)."""
        lower = torch.tensor([0.0, 5.0, 10.0])
        upper = torch.tensor([10.0, 15.0, 20.0])

        assert average_interval_width(lower, upper) == 10.0

    def test_winkler_no_penalty(self) -> None:
        """Winkler = width when all observations are covered."""
        y = torch.tensor([5.0])
        lower = torch.tensor([0.0])
        upper = torch.tensor([10.0])

        score = winkler_score(y, lower, upper, alpha=0.1)
        assert abs(score - 10.0) < 1e-6  # Just the width

    def test_winkler_penalty_below(self) -> None:
        """Winkler adds 2/α × (lower - y) penalty when y < lower."""
        y = torch.tensor([0.0])
        lower = torch.tensor([5.0])
        upper = torch.tensor([10.0])
        alpha = 0.1

        expected = (10.0 - 5.0) + (2.0 / 0.1) * (5.0 - 0.0)
        score = winkler_score(y, lower, upper, alpha=alpha)
        assert abs(score - expected) < 1e-6

    def test_winkler_penalty_above(self) -> None:
        """Winkler adds 2/α × (y - upper) penalty when y > upper."""
        y = torch.tensor([20.0])
        lower = torch.tensor([5.0])
        upper = torch.tensor([10.0])
        alpha = 0.1

        expected = (10.0 - 5.0) + (2.0 / 0.1) * (20.0 - 10.0)
        score = winkler_score(y, lower, upper, alpha=alpha)
        assert abs(score - expected) < 1e-6

    def test_conditional_coverage(self) -> None:
        """Per-group coverage should be computed correctly."""
        y = torch.tensor([5.0, 5.0, 100.0, 100.0])
        lower = torch.tensor([0.0, 0.0, 0.0, 0.0])
        upper = torch.tensor([10.0, 10.0, 10.0, 10.0])
        groups = torch.tensor([0, 0, 1, 1])

        cov = conditional_coverage(y, lower, upper, groups)
        assert cov[0] == 1.0  # Group 0: both covered
        assert cov[1] == 0.0  # Group 1: none covered

    def test_coverage_gap(self) -> None:
        """Coverage gap should be max deviation from target."""
        y = torch.tensor([5.0, 5.0, 100.0, 100.0])
        lower = torch.tensor([0.0, 0.0, 0.0, 0.0])
        upper = torch.tensor([10.0, 10.0, 10.0, 10.0])
        groups = torch.tensor([0, 0, 1, 1])

        gap = coverage_gap(y, lower, upper, groups, alpha=0.1)
        # Group 0: cov=1.0, dev=|1.0-0.9|=0.1
        # Group 1: cov=0.0, dev=|0.0-0.9|=0.9
        assert abs(gap - 0.9) < 1e-6

    def test_compute_all_returns_all_keys(self) -> None:
        """compute_all should return all expected metric keys."""
        y = torch.randint(0, 20, (50,)).float()
        lower = torch.zeros(50)
        upper = torch.full((50,), 25.0)

        result = compute_all_calibration_metrics(y, lower, upper, alpha=0.1)
        assert "picp" in result
        assert "aiw" in result
        assert "winkler" in result
        assert "coverage_valid" in result

    def test_compute_all_with_groups(self) -> None:
        """compute_all with groups should include fairness metrics."""
        y = torch.randint(0, 20, (50,)).float()
        lower = torch.zeros(50)
        upper = torch.full((50,), 25.0)
        groups = torch.arange(50) % 3

        result = compute_all_calibration_metrics(
            y, lower, upper, alpha=0.1, groups=groups
        )
        assert "coverage_gap" in result
        assert "conditional_coverage" in result


# ============================================================
# Integration Tests
# ============================================================

class TestEndToEnd:
    """End-to-end tests: fit → predict → evaluate."""

    def test_split_cp_end_to_end(self) -> None:
        """Full pipeline: generate data → calibrate → verify coverage."""
        torch.manual_seed(123)
        N = 2000
        pi = torch.full((N,), 0.25)
        mu = torch.rand(N) * 20 + 1.0
        r = torch.rand(N) * 8 + 1.0

        # Sample from true ZINB
        is_zero = torch.bernoulli(pi).bool()
        p = r / (r + mu)
        nb_samples = torch.distributions.NegativeBinomial(
            total_count=r, probs=p
        ).sample()
        y = torch.where(is_zero, torch.zeros_like(nb_samples), nb_samples)

        # Split: 50% cal, 50% test
        mid = N // 2
        cal = SplitConformalCalibrator(alpha=0.1)
        cal.fit(y[:mid], pi[:mid], mu[:mid], r[:mid])

        result = cal.predict(pi[mid:], mu[mid:], r[mid:])
        metrics = compute_all_calibration_metrics(
            y[mid:], result["lower"], result["upper"], alpha=0.1
        )

        # Coverage must be valid
        assert metrics["picp"] >= 0.85, (
            f"Coverage {metrics['picp']:.3f} too low"
        )
        # Width should be finite and reasonable
        assert metrics["aiw"] > 0 and metrics["aiw"] < 200
        # Winkler should be finite
        assert metrics["winkler"] > 0 and metrics["winkler"] < 1000

    def test_ecrc_end_to_end(self) -> None:
        """ECRC pipeline with group fairness verification."""
        torch.manual_seed(456)
        N = 2000
        pi = torch.full((N,), 0.2)
        mu = torch.rand(N) * 15 + 1.0
        r = torch.rand(N) * 5 + 1.0
        groups = torch.arange(N) % 4

        is_zero = torch.bernoulli(pi).bool()
        p = r / (r + mu)
        nb_samples = torch.distributions.NegativeBinomial(
            total_count=r, probs=p
        ).sample()
        y = torch.where(is_zero, torch.zeros_like(nb_samples), nb_samples)

        mid = N // 2
        cal = ECRCCalibrator(alpha=0.1, delta=0.05)
        cal.fit(
            y[:mid], pi[:mid], mu[:mid], r[:mid],
            groups=groups[:mid],
        )
        result = cal.predict(
            pi[mid:], mu[mid:], r[mid:],
            groups=groups[mid:],
        )

        metrics = compute_all_calibration_metrics(
            y[mid:], result["lower"], result["upper"],
            alpha=0.1, groups=groups[mid:],
        )

        assert metrics["picp"] >= 0.85
        assert metrics["aiw"] > 0
