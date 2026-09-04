"""Conformal calibration for ZINB crime-count predictions.

Implements five conformal prediction strategies, each matching a config
in ``configs/calibration/``:

1. **Split CP** (``split_cp``): Standard split conformal with CQR scores.
1b. **Randomized Split CP** (``randomized_split_cp``): Split conformal on the
   Dunn-Smyth randomized PIT. The integer CQR score is degenerate on sparse
   count panels -- >90% of scores are <= 0, so the empirical quantile pins to
   0.0 and the correction becomes the identity. Conformalizing the randomized
   PIT restores an exact finite-sample guarantee ON THE PIT SCALE. The
   delivered integer intervals still overcover, because integer endpoints on a
   discrete law always do; both numbers are reported.
2. **Weighted CP** (``weighted_cp``): Temporally-weighted conformal for
   non-stationary crime data.
3. **Mondrian CP** (``mondrian``): Group-conditional calibration with
   per-group coverage guarantees.
4. **Equalized Coverage** (``equalized_coverage``): Regularised threshold
   selection encouraging equal coverage across protected groups.
5. **ECRC** (``ecrc``): Equalized Conditional Risk Control using Hoeffding
   bounds for PAC-style per-group coverage.

All methods produce discrete prediction intervals ``[L, U]`` where
``L >= 0`` and ``L <= U``, guaranteed to achieve at least ``1 - α``
marginal (or per-group) coverage under the respective exchangeability
assumptions.

Core non-conformity score (Conformalized Quantile Regression — Romano et al., 2019):
    s_i = max(q_low(i) - y_i, y_i - q_high(i))
where q_low, q_high are ZINB quantiles at α/2 and 1-α/2.

References:
    - Romano, Patterson, Candès (2019): "Conformalized Quantile Regression"
    - Tibshirani et al. (2019): "Conformal Prediction Under Covariate Shift"
    - Vovk (2005): "Algorithmic Learning in a Random World" (Mondrian)
    - Romano et al. (2020): "Achieving Equalized Coverage"
    - Feldman et al. (2021): risk-control framework (ECRC)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Literal

import torch
from scipy.stats import beta as beta_distribution
from torch import Tensor

from civicsafe.calibration.zinb_distribution import (
    zinb_cdf_full,
    zinb_ppf,
    zinb_ppf_pair,
)

logger = logging.getLogger(__name__)

ScoreType = Literal["raw", "variance_scaled"]
ECRCBound = Literal["exact_binomial", "empirical_bernstein", "hoeffding"]


# ===================================================================
# Randomized PIT (for discrete-count conformal calibration)
# ===================================================================


def randomized_pit(
    y: Tensor,
    pi: Tensor,
    mu: Tensor,
    r: Tensor,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Dunn-Smyth randomized probability integral transform.

    For a DISCRETE response, ``F(y)`` is not uniform even under a perfectly
    specified model -- it can only take the countably many values the CDF
    attains. Drawing

        u ~ Uniform(F(y - 1), F(y))

    restores exact uniformity on [0, 1]. This is the standard device for
    residual diagnostics and conformal calibration of count models
    (Dunn & Smyth, 1996, "Randomized quantile residuals").

    Why this matters here
    ---------------------
    The integer CQR score ``max(q_low - y, y - q_high)`` is what the shipped
    calibrators consume, and it is degenerate on this panel: because the ZINB
    zero-mass exceeds alpha/2 for nearly every cell, ``q_low = 0``, so the
    interval is one-sided ``[0, q_high]``, and because the smallest integer k
    with ``F(k) >= 1 - alpha/2`` typically has ``F(k)`` STRICTLY greater than
    ``1 - alpha/2``, the raw interval already overcovers. Measured on
    ZINB(pi=0.05, r=2) draws: base coverage 0.9739 at mu=0.5 falling to 0.9514
    at mu=40, against a 0.90 target.

    The consequence is not a small bias. More than 90% of nonconformity scores
    are <= 0, so the (1-alpha) empirical quantile pins to exactly 0.0 and the
    conformal correction becomes the identity -- calibration is a NO-OP and the
    reported "calibrated" coverage is just the raw ZINB interval. Conformalizing
    the randomized PIT restores the exact finite-sample guarantee on the PIT
    scale: measured 0.9000 against a 0.9000 target on a mixed-scale synthetic
    panel.

    Inverting that band back to integer endpoints reimposes the lattice
    ceiling (0.9623 measured), so randomization repairs the calibration step
    without making the delivered intervals nominal. See
    :meth:`RandomizedSplitConformalCalibrator.predict`.

    Args:
        y: Observed counts. Shape: (N,)
        pi, mu, r: ZINB parameters. Shape: (N,)
        generator: Optional RNG for reproducibility. The randomization is
            genuine auxiliary noise, so a run is only reproducible if this is
            seeded; the caller is responsible for that.

    Returns:
        Randomized PIT values in [0, 1]. Shape: (N,)
    """
    y = y.reshape(-1).float()
    pi = pi.reshape(-1).float().clamp(0.0, 1.0)
    mu = mu.reshape(-1).float().clamp(min=1e-6)
    r = r.reshape(-1).float().clamp(min=0.1)

    _, F = zinb_cdf_full(pi, mu, r)  # (N, K)
    K = F.shape[1]

    idx = y.long().clamp(min=0, max=K - 1)
    F_y = F.gather(1, idx.unsqueeze(-1)).squeeze(-1)
    # F(-1) = 0 by definition, so a zero count draws from (0, F(0)).
    F_prev = torch.where(
        idx > 0,
        F.gather(1, (idx - 1).clamp(min=0).unsqueeze(-1)).squeeze(-1),
        torch.zeros_like(F_y),
    )

    if generator is None:
        u = torch.rand_like(F_y)
    else:
        u = torch.rand(F_y.shape, generator=generator, device=F_y.device)

    return F_prev + u * (F_y - F_prev).clamp(min=0.0)


# ===================================================================
# Non-conformity scores (shared across all methods)
# ===================================================================

def compute_cqr_scores(
    y: Tensor,
    pi: Tensor,
    mu: Tensor,
    r: Tensor,
    alpha: float = 0.1,
    *,
    score_type: ScoreType = "raw",
    continuity_correction: float = 0.5,
) -> Tensor:
    """Compute CQR non-conformity scores.

    s_i = max(q_low_i - y_i, y_i - q_high_i)

    Negative scores mean the observation was inside the heuristic interval.
    Positive scores mean it was outside.

    Args:
        y: Observed counts. Shape: (N,)
        pi, mu, r: Predicted ZINB parameters. Shape: (N,)
        alpha: Nominal miscoverage level.

    Returns:
        Non-conformity scores. Shape: (N,)
    """
    if score_type not in ("raw", "variance_scaled"):
        raise ValueError(f"Unknown score_type: {score_type!r}")
    if continuity_correction < 0.0:
        raise ValueError("continuity_correction must be non-negative")
    y = y.float()
    q_low, q_high = zinb_ppf_pair(alpha, pi, mu, r)
    if score_type == "raw":
        return torch.max(q_low - y, y - q_high)
    scale = zinb_predictive_scale(pi, mu, r)
    return torch.max(
        (q_low - y - continuity_correction) / scale,
        (y - q_high - continuity_correction) / scale,
    )


def zinb_predictive_scale(pi: Tensor, mu: Tensor, r: Tensor) -> Tensor:
    """Return ``sqrt(Var[Y]) + 1`` for a ZINB predictive distribution."""
    pi_f = pi.float().clamp(0.0, 1.0)
    mu_f = mu.float().clamp(min=1e-6)
    r_f = r.float().clamp(min=0.1)
    variance = (1.0 - pi_f) * mu_f * (
        1.0 + mu_f / r_f + pi_f * mu_f
    )
    return variance.clamp_min(0.0).sqrt() + 1.0


def compute_variance_scaled_cqr_scores(
    y: Tensor,
    pi: Tensor,
    mu: Tensor,
    r: Tensor,
    alpha: float = 0.1,
    *,
    continuity_correction: float = 0.5,
) -> Tensor:
    """Compute locally normalized, lattice-aware CQR nonconformity scores."""
    return compute_cqr_scores(
        y,
        pi,
        mu,
        r,
        alpha,
        score_type="variance_scaled",
        continuity_correction=continuity_correction,
    )


def _apply_cqr_threshold(
    alpha: float,
    pi: Tensor,
    mu: Tensor,
    r: Tensor,
    thresholds: Tensor,
    *,
    score_type: ScoreType,
    continuity_correction: float,
) -> tuple[Tensor, Tensor]:
    """Invert a raw or variance-scaled CQR threshold into count intervals."""
    q_low, q_high = zinb_ppf_pair(alpha, pi, mu, r)
    if score_type == "raw":
        lower = (q_low - thresholds).clamp(min=0.0).floor()
        upper = (q_high + thresholds).ceil()
    else:
        local_correction = thresholds * zinb_predictive_scale(pi, mu, r)
        lower = (
            q_low - local_correction - continuity_correction
        ).ceil().clamp(min=0.0)
        upper = (q_high + local_correction + continuity_correction).floor()
    return lower, torch.max(upper, lower)


def _ecrc_quantile_level(
    scores: Tensor,
    *,
    alpha: float,
    delta_group: float,
    bound: ECRCBound,
) -> tuple[float, float]:
    """Return the ECRC calibration quantile and its effective coverage slack."""
    n = int(scores.numel())
    if n < 1:
        raise ValueError("ECRC group must contain at least one score")
    target = 1.0 - alpha
    if bound == "exact_binomial":
        selected_rank = n
        for rank in range(max(1, math.ceil(target * n)), n + 1):
            lower_coverage = beta_distribution.ppf(
                delta_group, rank, n + 1 - rank
            )
            if math.isfinite(lower_coverage) and lower_coverage >= target:
                selected_rank = rank
                break
        level = selected_rank / n
        return min(level, 1.0), max(level - target, 0.0)
    if bound == "hoeffding":
        epsilon = math.sqrt(math.log(2.0 / delta_group) / (2.0 * n))
    elif bound == "empirical_bernstein":
        nominal_rank = min(max(math.ceil(target * n), 1), n)
        nominal_threshold = torch.kthvalue(scores, nominal_rank).values
        covered = (scores <= nominal_threshold).float()
        variance = float(covered.var(unbiased=True).item()) if n > 1 else 0.0
        log_term = math.log(2.0 / delta_group)
        epsilon = math.sqrt(2.0 * variance * log_term / n) + (
            7.0 * log_term / (3.0 * max(n - 1, 1))
        )
    else:
        raise ValueError(f"Unknown ECRC bound: {bound!r}")
    adjusted_alpha = max(alpha - epsilon, 0.0)
    return min(1.0 - adjusted_alpha, 1.0), epsilon


def _quantile_threshold(scores: Tensor, level: float) -> float:
    """Select an empirical order statistic without interpolation."""
    rank = min(max(math.ceil(level * scores.numel()), 1), scores.numel())
    return float(torch.kthvalue(scores, rank).values.item())


def _warn_if_degenerate(
    name: str,
    scores: Tensor,
    threshold: float,
    alpha: float,
) -> None:
    """Log loudly when the conformal correction has collapsed to a no-op.

    On sparse count panels the CQR score is bounded above by 0 for every
    observation that falls inside the raw ZINB interval, and the raw interval
    already overcovers because of discreteness (the smallest integer k with
    F(k) >= 1-alpha/2 usually has F(k) strictly greater). When more than
    (1-alpha) of the scores are <= 0, the empirical (1-alpha) quantile IS 0 and
    ``predict`` returns the uncalibrated quantiles unchanged.

    This shipped silently once: Chicago reported 0.9278 "calibrated" marginal
    coverage at a 0.90 target, which was simply the raw ZINB interval. Nothing
    in the output distinguished that from real calibration, so the warning
    exists to make the failure visible in the run log rather than inferrable
    only by re-deriving the threshold by hand.
    """
    frac_inside = float((scores <= 0).float().mean().item())
    if abs(threshold) < 1e-9 and frac_inside > (1.0 - alpha):
        logger.warning(
            f"  {name}: DEGENERATE CALIBRATION — threshold is exactly 0.0 and "
            f"{frac_inside:.1%} of calibration scores are <= 0 (needs "
            f"<= {1 - alpha:.0%} for the quantile to bind). The conformal "
            f"correction is the IDENTITY: predict() returns raw ZINB quantiles, "
            f"so any coverage you report is uncalibrated. This is caused by "
            f"count discreteness, not by a bad model. Use "
            f"RandomizedSplitConformalCalibrator for an exact guarantee."
        )


# ===================================================================
# Base Calibrator
# ===================================================================

class _BaseCalibrator:
    """Base class for conformal calibrators.

    Subclasses implement ``_compute_threshold`` to find the calibration
    correction from non-conformity scores. All methods share the same
    ``predict`` logic: inflate heuristic quantiles by the threshold.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        *,
        score_type: ScoreType = "raw",
        continuity_correction: float = 0.5,
    ) -> None:
        if not 0.01 <= alpha <= 0.5:
            raise ValueError(f"alpha must be in [0.01, 0.5], got {alpha}")
        self.alpha = alpha
        self.score_type = score_type
        self.continuity_correction = continuity_correction
        self._threshold: float | None = None
        self._fitted = False

    @property
    def threshold(self) -> float:
        """The calibration correction \\hat{q}_s."""
        if self._threshold is None:
            raise RuntimeError("Calibrator has not been fitted. Call fit() first.")
        return self._threshold

    def fit(
        self,
        y: Tensor,
        pi: Tensor,
        mu: Tensor,
        r: Tensor,
        **kwargs: Any,
    ) -> None:
        """Fit the calibrator on a calibration (held-out validation) set.

        Args:
            y: Observed counts on calibration set. Shape: (N,)
            pi, mu, r: Model-predicted ZINB parameters. Shape: (N,)
            **kwargs: Method-specific arguments (e.g., weights, groups).
        """
        y = y.reshape(-1).float()
        pi = pi.reshape(-1).float().clamp(0.0, 1.0)
        mu = mu.reshape(-1).float().clamp(min=1e-6)
        r = r.reshape(-1).float().clamp(min=0.1)

        scores = compute_cqr_scores(
            y,
            pi,
            mu,
            r,
            alpha=self.alpha,
            score_type=self.score_type,
            continuity_correction=self.continuity_correction,
        )
        self._threshold = self._compute_threshold(scores, **kwargs)
        self._fitted = True

        logger.info(
            f"  {self.__class__.__name__} fitted: threshold = {self._threshold:.4f}, "
            f"n_cal = {y.shape[0]}"
        )
        _warn_if_degenerate(
            self.__class__.__name__, scores, self._threshold, self.alpha
        )

    def _compute_threshold(self, scores: Tensor, **kwargs: Any) -> float:
        """Compute the calibration threshold from scores. Override in subclass."""
        raise NotImplementedError

    def predict(
        self,
        pi: Tensor,
        mu: Tensor,
        r: Tensor,
    ) -> dict[str, Tensor]:
        """Produce calibrated prediction intervals.

        Args:
            pi, mu, r: Predicted ZINB parameters. Shape: (N,) or (S, C).

        Returns:
            Dictionary with keys:
                lower: Lower bound of interval. Shape: same as input.
                upper: Upper bound of interval. Shape: same as input.
                point: Point estimate E[Y] = (1-π)·μ. Shape: same as input.
        """
        if not self._fitted:
            raise RuntimeError("Calibrator has not been fitted. Call fit() first.")

        orig_shape = pi.shape
        pi_f = pi.reshape(-1).float().clamp(0.0, 1.0)
        mu_f = mu.reshape(-1).float().clamp(min=1e-6)
        r_f = r.reshape(-1).float().clamp(min=0.1)

        thresholds = torch.full_like(pi_f, self.threshold)
        lower, upper = _apply_cqr_threshold(
            self.alpha,
            pi_f,
            mu_f,
            r_f,
            thresholds,
            score_type=self.score_type,
            continuity_correction=self.continuity_correction,
        )

        point = (1.0 - pi_f) * mu_f

        return {
            "lower": lower.reshape(orig_shape),
            "upper": upper.reshape(orig_shape),
            "point": point.reshape(orig_shape),
        }


# ===================================================================
# 1. Split Conformal Prediction
# ===================================================================

class SplitConformalCalibrator(_BaseCalibrator):
    """Standard split conformal prediction (Romano et al., 2019).

    The simplest method: takes the ⌈(1-α)(1+1/n)⌉-th empirical quantile
    of the non-conformity scores as the threshold.

    Guarantee: P(Y ∈ [L, U]) ≥ 1 - α (marginal, finite-sample, exact).
    """

    def _compute_threshold(self, scores: Tensor, **kwargs: Any) -> float:
        n = scores.shape[0]
        # Finite-sample correction: ⌈(1-α)(1+1/n)⌉
        quantile_level = min((1.0 - self.alpha) * (1.0 + 1.0 / n), 1.0)
        return torch.quantile(scores, quantile_level).item()


class VarianceScaledConformalCalibrator(SplitConformalCalibrator):
    """Split conformal with local ZINB predictive-scale normalization."""

    def __init__(
        self,
        alpha: float = 0.1,
        *,
        continuity_correction: float = 0.5,
    ) -> None:
        super().__init__(
            alpha,
            score_type="variance_scaled",
            continuity_correction=continuity_correction,
        )


# ===================================================================
# 1b. Randomized (smoothed) Split Conformal — exact for discrete counts
# ===================================================================


class RandomizedSplitConformalCalibrator:
    """Split conformal on the randomized PIT, giving exact discrete coverage.

    Every other calibrator in this module conformalizes the integer CQR score
    and, on this panel, degenerates: >90% of scores are <= 0, the empirical
    (1-alpha) quantile pins to 0.0, and the correction is the identity. See
    :func:`randomized_pit` for the measurement.

    This calibrator instead conformalizes ``u = randomized_pit(y, ...)``, which
    is exactly Uniform(0,1) under a correct model regardless of how discrete
    the counts are. Calibration finds the empirical ``[alpha/2, 1-alpha/2]``
    band of ``u`` on the calibration split, then inverts it back through the
    ZINB quantile function to produce integer bounds.

    What this does and does not fix
    ------------------------------
    It fixes CALIBRATION: the band is now selected by a procedure with an exact
    finite-sample guarantee (measured 0.9000 in PIT space) instead of by a
    degenerate all-zero score where the correction was the identity.

    It does NOT eliminate overcoverage of the delivered integer intervals.
    Inverting the band through ``zinb_ppf`` reimposes the lattice ceiling, so
    :meth:`predict` measures 0.9623 -- close to split_cp's 0.9632. That is not
    a defect in this class; it is a fact about integer-endpoint intervals on a
    discrete law, and it means the honest claim in the paper is "coverage is
    conservative by a quantifiable amount driven by count discreteness", NOT
    "we achieve nominal 90% coverage".

    Report both numbers: :meth:`predict` for what an analyst acts on, and
    :meth:`coverage_in_pit_space` for what the theory certifies.

    The randomization is real auxiliary noise: two runs with different seeds
    give slightly different bands. Pass ``seed`` to pin it, and report the
    seed. This is inherent to exact conformal inference on discrete data, not
    an implementation shortcut.

    Args:
        alpha: Miscoverage level.
        seed: RNG seed for the PIT randomization. ``None`` uses global RNG.
    """

    def __init__(self, alpha: float = 0.1, seed: int | None = None) -> None:
        if not 0.01 <= alpha <= 0.5:
            raise ValueError(f"alpha must be in [0.01, 0.5], got {alpha}")
        self.alpha = alpha
        self.seed = seed
        self._lo_level: float | None = None
        self._hi_level: float | None = None
        self._fitted = False

    def _generator(self, device: torch.device) -> torch.Generator | None:
        if self.seed is None:
            return None
        g = torch.Generator(device=device)
        g.manual_seed(self.seed)
        return g

    def fit(
        self,
        y: Tensor,
        pi: Tensor,
        mu: Tensor,
        r: Tensor,
        **kwargs: Any,
    ) -> None:
        """Find the empirical PIT band on the calibration split."""
        u = randomized_pit(y, pi, mu, r, generator=self._generator(mu.device))
        n = u.shape[0]

        # Finite-sample corrected levels, same spirit as split CP's
        # ceil((1-alpha)(n+1))-th order statistic, applied two-sided.
        lo_level = (self.alpha / 2.0) * (1.0 - 1.0 / (n + 1))
        hi_level = min((1.0 - self.alpha / 2.0) * (1.0 + 1.0 / n), 1.0)

        self._lo_level = torch.quantile(u, lo_level).item()
        self._hi_level = torch.quantile(u, hi_level).item()
        self._fitted = True

        logger.info(
            f"  RandomizedSplitConformalCalibrator fitted: PIT band = "
            f"[{self._lo_level:.4f}, {self._hi_level:.4f}] "
            f"(ideal [{self.alpha / 2:.4f}, {1 - self.alpha / 2:.4f}]), "
            f"n_cal = {n}"
        )

    def predict(
        self,
        pi: Tensor,
        mu: Tensor,
        r: Tensor,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        """Invert the calibrated PIT band through the ZINB quantile function.

        NOTE ON WHAT THIS DOES AND DOES NOT FIX
        ---------------------------------------
        The returned bounds are integers, and any integer-endpoint interval
        necessarily overcovers a discrete distribution -- you cannot cover
        exactly 90% of a lattice when the atoms have mass. Measured: this
        predict() lands at 0.9623 on a mixed-scale panel where the PIT band
        itself is exactly 0.9000. Inverting through ``zinb_ppf`` reimposes the
        ceiling that :func:`randomized_pit` removed.

        So this method fixes the CALIBRATION (the band is now chosen by an
        exact procedure rather than by a degenerate all-zero score) but not the
        REPORTING (integer endpoints still overcover). Use
        :meth:`coverage_in_pit_space` for the number that carries the exact
        finite-sample guarantee, and report both: the integer interval is what
        an analyst acts on, the PIT-space figure is what the theory certifies.
        """
        if not self._fitted:
            raise RuntimeError("Calibrator has not been fitted. Call fit() first.")
        assert self._lo_level is not None and self._hi_level is not None

        orig_shape = pi.shape
        pi_f = pi.reshape(-1).float().clamp(0.0, 1.0)
        mu_f = mu.reshape(-1).float().clamp(min=1e-6)
        r_f = r.reshape(-1).float().clamp(min=0.1)

        n = pi_f.shape[0]
        lo_lv = torch.full((n,), self._lo_level, device=pi_f.device)
        hi_lv = torch.full((n,), self._hi_level, device=pi_f.device)

        lower = zinb_ppf(lo_lv, pi_f, mu_f, r_f)
        upper = zinb_ppf(hi_lv, pi_f, mu_f, r_f)
        upper = torch.max(upper, lower)
        point = (1.0 - pi_f) * mu_f

        return {
            "lower": lower.reshape(orig_shape),
            "upper": upper.reshape(orig_shape),
            "point": point.reshape(orig_shape),
        }

    def coverage_in_pit_space(
        self,
        y: Tensor,
        pi: Tensor,
        mu: Tensor,
        r: Tensor,
    ) -> float:
        """Coverage on the randomized-PIT scale — the exact guarantee.

        This is the quantity split conformal actually certifies. It avoids the
        integer-endpoint overshoot entirely because the randomized PIT is
        continuous, so an exchangeable calibration/test split gives coverage in
        ``[1-alpha, 1-alpha + 1/(n+1)]``.

        Report this ALONGSIDE the integer-interval coverage from
        :meth:`predict`, never instead of it: an analyst acts on the integer
        interval, and quoting only the PIT figure would overstate how tight the
        delivered intervals are.
        """
        if not self._fitted:
            raise RuntimeError("Calibrator has not been fitted. Call fit() first.")
        assert self._lo_level is not None and self._hi_level is not None

        u = randomized_pit(y, pi, mu, r, generator=self._generator(mu.device))
        inside = (u >= self._lo_level) & (u <= self._hi_level)
        return float(inside.float().mean().item())


# ===================================================================
# 2. Weighted Conformal Prediction (temporal decay)
# ===================================================================

class WeightedConformalCalibrator(_BaseCalibrator):
    """Weighted conformal prediction for non-stationary data.

    Assigns exponentially decaying weights to calibration points, giving
    more influence to recent observations. Produces tighter intervals when
    the data distribution shifts over time (e.g., seasonal crime patterns).

    Reference: Barber, Candes, Ramdas & Tibshirani (2023), "Conformal
    prediction beyond exchangeability", Annals of Statistics -- fixed
    non-uniform weights on calibration points, which is what recency decay
    is. NOT Tibshirani et al. (2019), "Conformal Prediction Under Covariate
    Shift", which this docstring previously cited: that construction weights
    by a covariate likelihood ratio dQ_X/dP_X to correct a shift in the
    covariate marginal under an invariant P(Y|X). The two are different
    estimators with different assumptions, and citing the covariate-shift
    paper for a temporal-decay method is a misattribution.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        decay_rate: float = 0.05,
        min_weight: float = 1e-4,
    ) -> None:
        super().__init__(alpha)
        self.decay_rate = decay_rate
        self.min_weight = min_weight

    def fit(
        self,
        y: Tensor,
        pi: Tensor,
        mu: Tensor,
        r: Tensor,
        *,
        time_deltas: Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        """Fit with temporal weights.

        Args:
            y, pi, mu, r: Standard calibration data.
            time_deltas: Time difference from the most recent calibration
                point. Shape: (N,). If None, assumes uniform spacing
                (i.e., indices as time deltas).
        """
        y = y.reshape(-1).float()
        pi = pi.reshape(-1).float().clamp(0.0, 1.0)
        mu = mu.reshape(-1).float().clamp(min=1e-6)
        r = r.reshape(-1).float().clamp(min=0.1)

        scores = compute_cqr_scores(y, pi, mu, r, alpha=self.alpha)

        if time_deltas is None:
            # Assume uniform spacing: most recent = index N-1
            n = scores.shape[0]
            time_deltas = torch.arange(n, 0, -1, device=scores.device).float()

        self._threshold = self._compute_threshold(
            scores, time_deltas=time_deltas
        )
        self._fitted = True
        logger.info(
            f"  WeightedConformalCalibrator fitted: threshold = {self._threshold:.4f}"
        )

    def _compute_threshold(
        self, scores: Tensor, **kwargs: Any
    ) -> float:
        time_deltas = kwargs.get("time_deltas")
        if time_deltas is None:
            n = scores.shape[0]
            time_deltas = torch.arange(n, 0, -1, device=scores.device).float()

        # Exponential decay weights
        weights = torch.exp(-self.decay_rate * time_deltas).clamp(min=self.min_weight)
        weights = weights / weights.sum()  # Normalise to 1

        # Weighted quantile: sort scores, compute cumulative weights
        sorted_idx = torch.argsort(scores)
        sorted_scores = scores[sorted_idx]
        sorted_weights = weights[sorted_idx]

        cum_weights = sorted_weights.cumsum(dim=0)
        target = 1.0 - self.alpha

        # Find the first index where cumulative weight >= target
        mask = cum_weights >= target
        idx = (
            mask.float().argmax().item()
            if mask.any()
            else len(sorted_scores) - 1
        )

        return sorted_scores[int(idx)].item()


# ===================================================================
# 3. Mondrian Conformal Prediction (group-conditional)
# ===================================================================

class MondrianConformalCalibrator:
    """Mondrian (group-conditional) conformal prediction.

    Runs independent Split CP within each group to provide per-group
    coverage guarantees: P(Y ∈ C(X) | G=g) ≥ 1-α for every group g.

    Groups with fewer than ``min_group_size`` calibration points fall
    back to the global (pooled) quantile.

    Reference: Vovk (2005), "Algorithmic Learning in a Random World", §4.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        min_group_size: int = 40,
    ) -> None:
        if not 0.01 <= alpha <= 0.5:
            raise ValueError(f"alpha must be in [0.01, 0.5], got {alpha}")
        self.alpha = alpha
        self.min_group_size = min_group_size
        self._group_thresholds: dict[int, float] = {}
        self._global_threshold: float = 0.0
        self._fitted = False

    def fit(
        self,
        y: Tensor,
        pi: Tensor,
        mu: Tensor,
        r: Tensor,
        *,
        groups: Tensor,
        **kwargs: Any,
    ) -> None:
        """Fit per-group calibrators.

        Args:
            y, pi, mu, r: Calibration data. Shape: (N,)
            groups: Integer group labels. Shape: (N,)
        """
        y = y.reshape(-1).float()
        pi = pi.reshape(-1).float().clamp(0.0, 1.0)
        mu = mu.reshape(-1).float().clamp(min=1e-6)
        r = r.reshape(-1).float().clamp(min=0.1)
        groups = groups.reshape(-1)

        scores = compute_cqr_scores(y, pi, mu, r, alpha=self.alpha)

        # Global fallback threshold
        n = scores.shape[0]
        q_level = min((1.0 - self.alpha) * (1.0 + 1.0 / n), 1.0)
        self._global_threshold = torch.quantile(scores, q_level).item()

        # Per-group thresholds
        unique_groups = groups.unique().tolist()  # type: ignore[no-untyped-call]
        for g in unique_groups:
            mask = groups == g
            group_scores = scores[mask]
            n_g = group_scores.shape[0]

            if n_g >= self.min_group_size:
                q_level_g = min((1.0 - self.alpha) * (1.0 + 1.0 / n_g), 1.0)
                self._group_thresholds[int(g)] = torch.quantile(
                    group_scores, q_level_g
                ).item()
            else:
                self._group_thresholds[int(g)] = self._global_threshold

        self._fitted = True
        n_specific = sum(
            1 for g in unique_groups
            if (groups == g).sum() >= self.min_group_size
        )
        logger.info(
            f"  MondrianCP fitted: {n_specific}/{len(unique_groups)} groups "
            f"have ≥{self.min_group_size} calibration points "
            f"(global threshold = {self._global_threshold:.4f})"
        )

    def predict(
        self,
        pi: Tensor,
        mu: Tensor,
        r: Tensor,
        *,
        groups: Tensor,
    ) -> dict[str, Tensor]:
        """Predict with per-group calibration.

        Args:
            pi, mu, r: ZINB parameters. Shape: (N,) or (S, C).
            groups: Integer group labels. Shape: same as pi.

        Returns:
            dict with "lower", "upper", "point" tensors.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")

        orig_shape = pi.shape
        pi_f = pi.reshape(-1).float().clamp(0.0, 1.0)
        mu_f = mu.reshape(-1).float().clamp(min=1e-6)
        r_f = r.reshape(-1).float().clamp(min=0.1)
        groups_f = groups.reshape(-1)

        q_low, q_high = zinb_ppf_pair(self.alpha, pi_f, mu_f, r_f)

        # Build per-element threshold tensor
        thresholds = torch.full_like(pi_f, self._global_threshold)
        for g, t in self._group_thresholds.items():
            mask = groups_f == g
            thresholds[mask] = t

        lower = (q_low - thresholds).clamp(min=0.0).floor()
        upper = (q_high + thresholds).ceil()
        upper = torch.max(upper, lower)
        point = (1.0 - pi_f) * mu_f

        return {
            "lower": lower.reshape(orig_shape),
            "upper": upper.reshape(orig_shape),
            "point": point.reshape(orig_shape),
        }


# ===================================================================
# 4. Equalized Coverage Conformal Prediction
# ===================================================================

class EqualizedCoverageCalibrator:
    """Equalized coverage conformal prediction.

    Chooses the threshold ``q`` that minimises a regularised objective
    balancing marginal coverage and cross-group coverage variance:

        L(q) = |{i : s_i > q}|/n  +  λ_eq × max_g |coverage(g) - (1-α)|

    This encourages equal coverage across protected groups (e.g., income
    quartiles) at the cost of slightly wider intervals overall.

    Reference: Romano et al. (2020), "Achieving Equalized Coverage."
    """

    def __init__(
        self,
        alpha: float = 0.1,
        lambda_eq: float = 1.0,
    ) -> None:
        if not 0.01 <= alpha <= 0.5:
            raise ValueError(f"alpha must be in [0.01, 0.5], got {alpha}")
        self.alpha = alpha
        self.lambda_eq = lambda_eq
        self._threshold: float = 0.0
        self._fitted = False

    def fit(
        self,
        y: Tensor,
        pi: Tensor,
        mu: Tensor,
        r: Tensor,
        *,
        groups: Tensor,
        **kwargs: Any,
    ) -> None:
        """Fit via grid search over candidate thresholds.

        Args:
            y, pi, mu, r: Calibration data. Shape: (N,)
            groups: Protected group labels. Shape: (N,)
        """
        y = y.reshape(-1).float()
        pi = pi.reshape(-1).float().clamp(0.0, 1.0)
        mu = mu.reshape(-1).float().clamp(min=1e-6)
        r = r.reshape(-1).float().clamp(min=0.1)
        groups = groups.reshape(-1)

        scores = compute_cqr_scores(y, pi, mu, r, alpha=self.alpha)
        q_low, q_high = zinb_ppf_pair(self.alpha, pi, mu, r)

        # Candidate thresholds: unique sorted score values
        candidates = torch.unique(scores)
        target_cov = 1.0 - self.alpha
        unique_groups = groups.unique()  # type: ignore[no-untyped-call]

        best_loss = float("inf")
        best_q = candidates[-1].item()  # Conservative default

        for q_candidate in candidates:
            q_val = q_candidate.item()

            # Compute interval for each calibration point
            lo = (q_low - q_val).clamp(min=0.0).floor()
            hi = (q_high + q_val).ceil()

            # Overall coverage
            covered = ((y >= lo) & (y <= hi)).float()
            marginal_cov = covered.mean().item()

            # Per-group coverage deviation
            max_dev = 0.0
            for g in unique_groups:
                mask = groups == g
                if mask.sum() > 0:
                    group_cov = covered[mask].mean().item()
                    dev = abs(group_cov - target_cov)
                    max_dev = max(max_dev, dev)

            # Penalise undercoverage
            undercoverage_penalty = max(0.0, target_cov - marginal_cov)

            loss = undercoverage_penalty + self.lambda_eq * max_dev
            if loss < best_loss:
                best_loss = loss
                best_q = q_val

        self._threshold = best_q
        self._fitted = True
        logger.info(
            f"  EqualizedCoverageCalibrator fitted: threshold = {self._threshold:.4f}"
        )

    def predict(
        self,
        pi: Tensor,
        mu: Tensor,
        r: Tensor,
    ) -> dict[str, Tensor]:
        """Produce calibrated prediction intervals."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")

        orig_shape = pi.shape
        pi_f = pi.reshape(-1).float().clamp(0.0, 1.0)
        mu_f = mu.reshape(-1).float().clamp(min=1e-6)
        r_f = r.reshape(-1).float().clamp(min=0.1)

        q_low, q_high = zinb_ppf_pair(self.alpha, pi_f, mu_f, r_f)

        lower = (q_low - self._threshold).clamp(min=0.0).floor()
        upper = (q_high + self._threshold).ceil()
        upper = torch.max(upper, lower)
        point = (1.0 - pi_f) * mu_f

        return {
            "lower": lower.reshape(orig_shape),
            "upper": upper.reshape(orig_shape),
            "point": point.reshape(orig_shape),
        }


# ===================================================================
# 5. ECRC — Equalized Conditional Risk Control
# ===================================================================

class ECRCCalibrator:
    """Equalized Conditional Risk Control (ECRC).

    Provides high-probability per-group coverage guarantees using
    Hoeffding's inequality. For each group g:

        P(coverage(g) ≥ 1 - α - ε) ≥ 1 - δ

    where ε = √(ln(2·G/δ) / (2·n_cal/G)) is the Hoeffding slack.

    This is the primary calibration method for CIVIC-SAFE because it
    provides the strongest fairness guarantees with a principled
    statistical foundation.

    Reference: Feldman et al. (2021), risk-control framework.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        delta: float = 0.05,
        group_type: str = "geographic",
        bound: ECRCBound = "exact_binomial",
        score_type: ScoreType = "variance_scaled",
        continuity_correction: float = 0.5,
    ) -> None:
        if not 0.01 <= alpha <= 0.5:
            raise ValueError(f"alpha must be in [0.01, 0.5], got {alpha}")
        self.alpha = alpha
        self.delta = delta
        self.group_type = group_type
        self.bound = bound
        self.score_type = score_type
        self.continuity_correction = continuity_correction
        self._group_thresholds: dict[int, float] = {}
        self._group_epsilons: dict[int, float] = {}
        self._group_quantile_levels: dict[int, float] = {}
        self._epsilon: float = 0.0
        self._fitted = False

    def fit(
        self,
        y: Tensor,
        pi: Tensor,
        mu: Tensor,
        r: Tensor,
        *,
        groups: Tensor,
        **kwargs: Any,
    ) -> None:
        """Fit ECRC calibrator.

        Args:
            y, pi, mu, r: Calibration data. Shape: (N,)
            groups: Group labels. Shape: (N,)
        """
        y = y.reshape(-1).float()
        pi = pi.reshape(-1).float().clamp(0.0, 1.0)
        mu = mu.reshape(-1).float().clamp(min=1e-6)
        r = r.reshape(-1).float().clamp(min=0.1)
        groups = groups.reshape(-1)

        scores = compute_cqr_scores(
            y,
            pi,
            mu,
            r,
            alpha=self.alpha,
            score_type=self.score_type,
            continuity_correction=self.continuity_correction,
        )

        unique_groups = groups.unique()  # type: ignore[no-untyped-call]
        G = len(unique_groups)
        n_cal = scores.shape[0]

        # Bonferroni-corrected per-group delta
        delta_g = self.delta / G

        max_epsilon = 0.0
        for g in unique_groups:
            mask = groups == g
            group_scores = scores[mask]
            q_level, epsilon_g = _ecrc_quantile_level(
                group_scores,
                alpha=self.alpha,
                delta_group=delta_g,
                bound=self.bound,
            )
            g_idx = int(g.item())
            self._group_quantile_levels[g_idx] = q_level
            self._group_epsilons[g_idx] = epsilon_g
            max_epsilon = max(max_epsilon, epsilon_g)
        self._epsilon = max_epsilon

        # Per-group calibration at the bound-selected order statistic.
        for g in unique_groups:
            mask = groups == g
            group_scores = scores[mask]
            n_g = group_scores.shape[0]

            if n_g > 0:
                g_idx = int(g.item())
                self._group_thresholds[g_idx] = _quantile_threshold(
                    group_scores, self._group_quantile_levels[g_idx]
                )

        self._fitted = True
        logger.info(
            f"  ECRCCalibrator fitted: bound={self.bound}, "
            f"max_slack = {self._epsilon:.4f}, "
            f"G = {G} groups, n_cal = {n_cal}"
        )

    def predict(
        self,
        pi: Tensor,
        mu: Tensor,
        r: Tensor,
        *,
        groups: Tensor,
    ) -> dict[str, Tensor]:
        """Produce intervals with Hoeffding-guaranteed per-group coverage."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")

        orig_shape = pi.shape
        pi_f = pi.reshape(-1).float().clamp(0.0, 1.0)
        mu_f = mu.reshape(-1).float().clamp(min=1e-6)
        r_f = r.reshape(-1).float().clamp(min=0.1)
        groups_f = groups.reshape(-1)

        # Build per-element threshold
        # Default to conservative global threshold
        all_thresholds = list(self._group_thresholds.values())
        fallback = max(all_thresholds) if all_thresholds else 0.0

        thresholds = torch.full_like(pi_f, fallback)
        for g, t in self._group_thresholds.items():
            mask = groups_f == g
            thresholds[mask] = t

        lower, upper = _apply_cqr_threshold(
            self.alpha,
            pi_f,
            mu_f,
            r_f,
            thresholds,
            score_type=self.score_type,
            continuity_correction=self.continuity_correction,
        )
        point = (1.0 - pi_f) * mu_f

        return {
            "lower": lower.reshape(orig_shape),
            "upper": upper.reshape(orig_shape),
            "point": point.reshape(orig_shape),
        }

    @property
    def epsilon(self) -> float:
        """Largest effective group-wise bound slack."""
        return self._epsilon


# ===================================================================
# 6. Adaptive Temporal ECRC (Phase 5)
# ===================================================================

class AdaptiveTemporalECRCCalibrator:
    """Adaptive Temporal Conformal Calibration with Demographic Stratification.

    Combines Adaptive Conformal Inference (ACI, Gibbs & Candes 2021) with
    Equalized Conditional Risk Control (ECRC, Feldman et al. 2021).
    Corrects for temporal non-exchangeability by dynamically adjusting the
    target alpha level per demographic group based on recent empirical coverage.

    For each group g at time t, the effective alpha is updated:
        alpha_{t,g} = alpha_{t-1,g} + gamma * (err_{t-1,g} - alpha)

    Where err_{t-1,g} is the empirical miscoverage for group g at time t-1.
    """

    # Bound on the PID integral accumulator (anti-windup). With k_i = 1e-3 this
    # caps the integral contribution to alpha at +/- 0.002 per step, so a long
    # run of one-sided error can no longer silently wind up and then overshoot.
    _INTEGRAL_CLIP: float = 2.0

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.005,
        delta: float = 0.1,
        group_type: str = "income",
        k_i: float = 0.001,
        k_d: float = 0.0005,
        max_width: float = float("inf"),
        max_width_ratio: float | None = None,
        bound: ECRCBound = "exact_binomial",
        score_type: ScoreType = "variance_scaled",
        continuity_correction: float = 0.5,
    ) -> None:
        if not 0.01 <= alpha <= 0.5:
            raise ValueError(f"alpha must be in [0.01, 0.5], got {alpha}")
        self.nominal_alpha = alpha
        self.gamma = gamma
        self.delta = delta
        self.group_type = group_type

        # PID Constants
        self.k_p = gamma
        self.k_i = k_i
        self.k_d = k_d
        # Abstention thresholds. `max_width` is an ABSOLUTE count width and
        # defaults to off, because an absolute threshold cannot be correct on a
        # panel spanning three orders of magnitude in mu: the previous default
        # of 100.0 never fires for a mu=0.5 drug cell (where a width of 100 is
        # absurd) yet fires on EVERY mu=200 property cell, whose correct 90%
        # ZINB interval is ~470 wide. Measured at mu=200: 100% of cells
        # abstained, and because abstention emits NaN that downstream coverage
        # counted as a miss, marginal coverage read 0.0000 and the ACI loop
        # drove alpha_t to its 0.01 floor -- widening intervals, causing more
        # abstention. Level anchoring raises fitted mu into exactly this range,
        # so the old default would have silently destroyed the anchored re-run.
        # Use `max_width_ratio` for a scale-free rule instead.
        self.max_width = max_width
        self.max_width_ratio = max_width_ratio
        self.bound = bound
        self.score_type = score_type
        self.continuity_correction = continuity_correction

        # State tracking per group
        self._alpha_t: dict[int, float] = {}
        self._integral_err: dict[int, float] = {}
        self._prev_err: dict[int, float] = {}
        self._calibration_scores: dict[int, torch.Tensor] = {}
        self._initial_quantile_levels: dict[int, float] = {}
        self._epsilon: float = 0.0
        self._fitted = False
        # Counts update() calls. If predict() runs while this is 0, every
        # alpha_t is still the fit-time constant max(alpha - epsilon, 0.01),
        # which is byte-identical to what ECRCCalibrator produces -- the
        # "adaptive" part of this class has done nothing. That silent
        # degradation shipped a duplicate method into the results table once
        # already; predict() now warns about it.
        self._n_updates: int = 0
        self._warned_never_updated = False

    def fit(
        self,
        y: Tensor,
        pi: Tensor,
        mu: Tensor,
        r: Tensor,
        *,
        groups: Tensor,
        **kwargs: Any,
    ) -> None:
        y = y.reshape(-1).float()
        pi = pi.reshape(-1).float().clamp(0.0, 1.0)
        mu = mu.reshape(-1).float().clamp(min=1e-6)
        r = r.reshape(-1).float().clamp(min=0.1)
        groups = groups.reshape(-1)

        scores = compute_cqr_scores(
            y,
            pi,
            mu,
            r,
            alpha=self.nominal_alpha,
            score_type=self.score_type,
            continuity_correction=self.continuity_correction,
        )

        unique_groups = groups.unique()  # type: ignore[no-untyped-call]
        G = len(unique_groups)

        delta_g = self.delta / G
        max_epsilon = 0.0
        for g in unique_groups:
            group_scores = scores[groups == g]
            q_level, epsilon_g = _ecrc_quantile_level(
                group_scores,
                alpha=self.nominal_alpha,
                delta_group=delta_g,
                bound=self.bound,
            )
            self._initial_quantile_levels[int(g.item())] = q_level
            max_epsilon = max(max_epsilon, epsilon_g)
        self._epsilon = max_epsilon
        base_alpha = max(self.nominal_alpha - self._epsilon, 0.01)

        for g in unique_groups:
            g_idx = int(g.item())
            mask = groups == g
            self._calibration_scores[g_idx] = scores[mask].clone()
            self._alpha_t[g_idx] = base_alpha

        self._fitted = True
        logger.info(
            f"  AdaptiveTemporalECRC fitted: base_alpha = {base_alpha:.4f}, "
            f"gamma = {self.gamma:.3f}, G = {G} groups"
        )

    def predict(
        self,
        pi: Tensor,
        mu: Tensor,
        r: Tensor,
        *,
        groups: Tensor,
    ) -> dict[str, Tensor]:
        if not self._fitted:
            raise RuntimeError("Call fit() first.")

        if self._n_updates == 0 and not self._warned_never_updated:
            self._warned_never_updated = True
            logger.warning(
                "  AdaptiveTemporalECRC: predict() called after 0 update() "
                "calls. Every alpha_t is still the fit-time constant "
                f"{max(self.nominal_alpha - self._epsilon, 0.01):.4f}, so these "
                "intervals are IDENTICAL to plain ECRC -- the ACI adaptation "
                "has not run. Call update() on held-out weeks in a rolling "
                "loop, or report this as 'ecrc' rather than as an adaptive "
                "method."
            )

        orig_shape = pi.shape
        pi_f = pi.reshape(-1).float().clamp(0.0, 1.0)
        mu_f = mu.reshape(-1).float().clamp(min=1e-6)
        r_f = r.reshape(-1).float().clamp(min=0.1)
        groups_f = groups.reshape(-1)

        thresholds = torch.zeros_like(pi_f)
        for g, alpha_t in self._alpha_t.items():
            mask = groups_f == g
            if mask.sum() == 0:
                continue

            cal_scores = self._calibration_scores.get(g)
            if cal_scores is None or len(cal_scores) == 0:
                t_val = 0.0
            else:
                q_level = (
                    self._initial_quantile_levels[g]
                    if self._n_updates == 0
                    else min(1.0 - alpha_t, 1.0)
                )
                t_val = _quantile_threshold(cal_scores, q_level)

            thresholds[mask] = t_val

        lower, upper = _apply_cqr_threshold(
            self.nominal_alpha,
            pi_f,
            mu_f,
            r_f,
            thresholds,
            score_type=self.score_type,
            continuity_correction=self.continuity_correction,
        )
        point = (1.0 - pi_f) * mu_f

        # Abstention: emit NaN where the interval is too wide to stand behind.
        # NaN is the deliberate sentinel -- it propagates rather than silently
        # reading as a valid bound -- but every consumer MUST mask it out before
        # computing coverage, because `y >= nan` is False and an unmasked
        # abstention therefore reads as a miscoverage. See
        # compute_coverage_metrics() in scripts/run_conformal_evaluation.py.
        width = upper - lower
        abstain_mask = width > self.max_width
        if self.max_width_ratio is not None:
            # Scale-free rule: abstain when the interval is wider than
            # `ratio` times the predicted level. The +1 keeps the rule finite
            # for near-zero cells instead of abstaining on all of them.
            abstain_mask = abstain_mask | (width > self.max_width_ratio * (point + 1.0))
        lower[abstain_mask] = float('nan')
        upper[abstain_mask] = float('nan')

        return {
            "lower": lower.reshape(orig_shape),
            "upper": upper.reshape(orig_shape),
            "point": point.reshape(orig_shape),
        }

    def update(
        self,
        y_true: Tensor,
        pi: Tensor,
        mu: Tensor,
        r: Tensor,
        *,
        groups: Tensor,
    ) -> None:
        """Update the adaptive alpha_t based on observed coverage at time t."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")

        # Incremented before the internal predict() below so that call does not
        # trip the "never updated" warning on the very first update.
        self._n_updates += 1

        y_f = y_true.reshape(-1).float()
        pi_f = pi.reshape(-1).float().clamp(0.0, 1.0)
        mu_f = mu.reshape(-1).float().clamp(min=1e-6)
        r_f = r.reshape(-1).float().clamp(min=0.1)
        groups_f = groups.reshape(-1)

        # We need the intervals we *would have* predicted
        intervals = self.predict(pi_f, mu_f, r_f, groups=groups_f)
        lower = intervals["lower"].reshape(-1)
        upper = intervals["upper"].reshape(-1)

        covered = ((y_f >= lower) & (y_f <= upper)).float()
        # Abstained cells issued no interval, so they are neither covered nor
        # missed and must not enter the ACI error signal. Counting them as
        # misses creates a positive feedback loop: NaN reads as miscoverage ->
        # err_t rises -> alpha_t falls -> intervals widen -> more cells exceed
        # the width threshold -> more abstention. Measured with the old absolute
        # default at mu=200, alpha_t collapsed 0.0315 -> 0.0100 (the floor) in
        # four updates and stayed pinned.
        issued = ~(torch.isnan(lower) | torch.isnan(upper))

        # Also compute scores to add to the calibration set (sliding window / growing)
        scores = compute_cqr_scores(
            y_f,
            pi_f,
            mu_f,
            r_f,
            alpha=self.nominal_alpha,
            score_type=self.score_type,
            continuity_correction=self.continuity_correction,
        )

        unique_groups = groups_f.unique()  # type: ignore[no-untyped-call]
        for g in unique_groups:
            g_idx = int(g.item())
            mask = groups_f == g

            if mask.sum() > 0:
                # 1. Update alpha_t using ACI, on issued intervals only.
                scored = mask & issued
                if scored.sum() == 0:
                    # Every cell in this group abstained. There is no coverage
                    # evidence, so hold alpha_t rather than inventing an error.
                    # The calibration-set append below still runs: the CQR
                    # scores do not depend on whether an interval was issued.
                    logger.warning(
                        f"  AdaptiveTemporalECRC: group {g_idx} abstained on all "
                        f"{int(mask.sum().item())} cells this step; holding "
                        f"alpha_t at {self._alpha_t.get(g_idx, float('nan')):.4f}. "
                        "Check max_width / max_width_ratio -- the threshold may "
                        "be tighter than the panel's natural interval widths."
                    )
                else:
                    empirical_cov = covered[scored].mean().item()
                    err_t = 1.0 - empirical_cov

                    if g_idx not in self._alpha_t:
                        self._alpha_t[g_idx] = max(
                            self.nominal_alpha - self._epsilon, 0.01
                        )

                    # PID update rule
                    e_t = self.nominal_alpha - err_t  # Negative feedback error term

                    if g_idx not in self._integral_err:
                        self._integral_err[g_idx] = 0.0
                        self._prev_err[g_idx] = e_t

                    p_term = self.k_p * e_t
                    d_term = self.k_d * (e_t - self._prev_err[g_idx])
                    self._prev_err[g_idx] = e_t

                    # Anti-windup on the integral term: stop integrating while
                    # the output is saturated, and bound the accumulator.
                    # Without this the accumulator grows every week even when
                    # alpha_t is pinned at a clamp and cannot respond.
                    #
                    # Scope note (measured, not assumed): anti-windup alone
                    # barely moves the observed coverage drift (0.0805 -> 0.0799
                    # in simulation) because the proportional term dominates.
                    # The drift from ~0.99 coverage in early test weeks to ~0.81
                    # in late ones is driven by base_alpha starting pinned at the
                    # 0.01 floor, which happens when one demographic group is so
                    # small that the Hoeffding slack eats the whole alpha budget.
                    # Balancing the groups moves simulated drift from +0.0799 to
                    # -0.0151. Fix the group sizes; this clamp is hygiene, not
                    # the cure.
                    prev_alpha = self._alpha_t[g_idx]
                    saturated = prev_alpha <= 0.01 or prev_alpha >= 0.99
                    if not saturated:
                        self._integral_err[g_idx] += e_t
                    self._integral_err[g_idx] = max(
                        min(self._integral_err[g_idx], self._INTEGRAL_CLIP),
                        -self._INTEGRAL_CLIP,
                    )
                    i_term = self.k_i * self._integral_err[g_idx]

                    new_alpha = prev_alpha + p_term + i_term + d_term
                    self._alpha_t[g_idx] = max(min(new_alpha, 0.99), 0.01)

                # 2. Add to calibration set (EnbPI style)
                # For memory bounds, keep only last N scores (e.g. 500)
                if g_idx not in self._calibration_scores:
                    self._calibration_scores[g_idx] = scores[mask]
                else:
                    self._calibration_scores[g_idx] = torch.cat([
                        self._calibration_scores[g_idx], scores[mask]
                    ])[-500:] # Keep last 500 elements per group to maintain adaptation speed


# ===================================================================
# Factory: config → calibrator
# ===================================================================

def create_calibrator(config: dict[str, Any]) -> (
    _BaseCalibrator | MondrianConformalCalibrator
    | EqualizedCoverageCalibrator | ECRCCalibrator
    | RandomizedSplitConformalCalibrator | AdaptiveTemporalECRCCalibrator
):
    """Create a calibrator from a Hydra config dictionary.

    Args:
        config: Must contain a ``calibration`` key with ``method`` and
            ``alpha`` at minimum. Method-specific keys are passed through.

    Returns:
        An unfitted calibrator instance.

    Example::

        cfg = yaml.safe_load(open("configs/calibration/ecrc.yaml"))
        calibrator = create_calibrator(cfg)
        calibrator.fit(y_cal, pi_cal, mu_cal, r_cal, groups=groups_cal)
        intervals = calibrator.predict(pi_test, mu_test, r_test, groups=groups_test)
    """
    cal_cfg = config.get("calibration", config)
    method = cal_cfg["method"]
    alpha = cal_cfg.get("alpha", 0.1)

    if method == "split_cp":
        return SplitConformalCalibrator(alpha=alpha)

    elif method in ("variance_scaled", "variance_scaled_split_cp"):
        return VarianceScaledConformalCalibrator(
            alpha=alpha,
            continuity_correction=cal_cfg.get("continuity_correction", 0.5),
        )

    elif method == "randomized_split_cp":
        return RandomizedSplitConformalCalibrator(
            alpha=alpha,
            seed=cal_cfg.get("seed", 0),
        )

    elif method == "weighted_cp":
        return WeightedConformalCalibrator(
            alpha=alpha,
            decay_rate=cal_cfg.get("decay_rate", 0.05),
            min_weight=cal_cfg.get("min_weight", 1e-4),
        )

    elif method == "mondrian":
        return MondrianConformalCalibrator(
            alpha=alpha,
            min_group_size=cal_cfg.get("min_group_size", 40),
        )

    elif method == "equalized_coverage":
        return EqualizedCoverageCalibrator(
            alpha=alpha,
            lambda_eq=cal_cfg.get("lambda_eq", 1.0),
        )

    elif method == "ecrc":
        return ECRCCalibrator(
            alpha=alpha,
            delta=cal_cfg.get("delta", 0.05),
            group_type=cal_cfg.get("group_type", "geographic"),
            bound=cal_cfg.get("bound", "exact_binomial"),
            score_type=cal_cfg.get("score_type", "variance_scaled"),
            continuity_correction=cal_cfg.get("continuity_correction", 0.5),
        )

    elif method == "adaptive_ecrc":
        return AdaptiveTemporalECRCCalibrator(
            alpha=alpha,
            gamma=cal_cfg.get("gamma", 0.05),
            delta=cal_cfg.get("delta", 0.05),
            group_type=cal_cfg.get("group_type", "geographic"),
            bound=cal_cfg.get("bound", "exact_binomial"),
            score_type=cal_cfg.get("score_type", "variance_scaled"),
            continuity_correction=cal_cfg.get("continuity_correction", 0.5),
        )

    else:
        raise ValueError(
            f"Unknown calibration method: '{method}'. "
            f"Valid: split_cp, randomized_split_cp, weighted_cp, mondrian, "
            f"equalized_coverage, ecrc, adaptive_ecrc"
        )
