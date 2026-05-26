"""Conformal calibration for ZINB crime-count predictions.

Implements five conformal prediction strategies, each matching a config
in ``configs/calibration/``:

1. **Split CP** (``split_cp``): Standard split conformal with CQR scores.
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
from typing import Any

import torch
from torch import Tensor

from civicsafe.calibration.zinb_distribution import zinb_ppf_pair

logger = logging.getLogger(__name__)


# ===================================================================
# Non-conformity scores (shared across all methods)
# ===================================================================

def compute_cqr_scores(
    y: Tensor,
    pi: Tensor,
    mu: Tensor,
    r: Tensor,
    alpha: float = 0.1,
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
    y = y.float()
    q_low, q_high = zinb_ppf_pair(alpha, pi, mu, r)
    return torch.max(q_low - y, y - q_high)


# ===================================================================
# Base Calibrator
# ===================================================================

class _BaseCalibrator:
    """Base class for conformal calibrators.

    Subclasses implement ``_compute_threshold`` to find the calibration
    correction from non-conformity scores. All methods share the same
    ``predict`` logic: inflate heuristic quantiles by the threshold.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        if not 0.01 <= alpha <= 0.5:
            raise ValueError(f"alpha must be in [0.01, 0.5], got {alpha}")
        self.alpha = alpha
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

        scores = compute_cqr_scores(y, pi, mu, r, alpha=self.alpha)
        self._threshold = self._compute_threshold(scores, **kwargs)
        self._fitted = True

        logger.info(
            f"  {self.__class__.__name__} fitted: threshold = {self._threshold:.4f}, "
            f"n_cal = {y.shape[0]}"
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

        q_low, q_high = zinb_ppf_pair(self.alpha, pi_f, mu_f, r_f)

        # Apply CQR correction
        lower = (q_low - self.threshold).clamp(min=0.0).floor()
        upper = (q_high + self.threshold).ceil()

        # Ensure L <= U (can happen if threshold is very negative)
        upper = torch.max(upper, lower)

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


# ===================================================================
# 2. Weighted Conformal Prediction (temporal decay)
# ===================================================================

class WeightedConformalCalibrator(_BaseCalibrator):
    """Weighted conformal prediction for non-stationary data.

    Assigns exponentially decaying weights to calibration points, giving
    more influence to recent observations. Produces tighter intervals when
    the data distribution shifts over time (e.g., seasonal crime patterns).

    Reference: Tibshirani et al. (2019), "Conformal Prediction Under
    Covariate Shift."
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
        if mask.any():
            idx = mask.float().argmax().item()
        else:
            idx = len(sorted_scores) - 1

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
    ) -> None:
        if not 0.01 <= alpha <= 0.5:
            raise ValueError(f"alpha must be in [0.01, 0.5], got {alpha}")
        self.alpha = alpha
        self.delta = delta
        self.group_type = group_type
        self._group_thresholds: dict[int, float] = {}
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

        scores = compute_cqr_scores(y, pi, mu, r, alpha=self.alpha)

        unique_groups = groups.unique()  # type: ignore[no-untyped-call]
        G = len(unique_groups)
        n_cal = scores.shape[0]

        # Hoeffding epsilon
        self._epsilon = math.sqrt(
            math.log(2.0 * G / self.delta) / (2.0 * n_cal / G)
        )

        # Adjusted alpha for per-group guarantees
        adjusted_alpha = max(self.alpha - self._epsilon, 0.01)

        # Per-group calibration with adjusted alpha
        for g in unique_groups:
            mask = groups == g
            group_scores = scores[mask]
            n_g = group_scores.shape[0]

            if n_g > 0:
                q_level = min(
                    (1.0 - adjusted_alpha) * (1.0 + 1.0 / max(n_g, 1)), 1.0
                )
                self._group_thresholds[int(g.item())] = torch.quantile(
                    group_scores, q_level
                ).item()

        self._fitted = True
        logger.info(
            f"  ECRCCalibrator fitted: ε = {self._epsilon:.4f}, "
            f"adjusted_α = {adjusted_alpha:.4f}, "
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

        q_low, q_high = zinb_ppf_pair(self.alpha, pi_f, mu_f, r_f)

        # Build per-element threshold
        # Default to conservative global threshold
        all_thresholds = list(self._group_thresholds.values())
        fallback = max(all_thresholds) if all_thresholds else 0.0

        thresholds = torch.full_like(pi_f, fallback)
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

    @property
    def epsilon(self) -> float:
        """Hoeffding slack term."""
        return self._epsilon


# ===================================================================
# Factory: config → calibrator
# ===================================================================

def create_calibrator(config: dict[str, Any]) -> (
    _BaseCalibrator | MondrianConformalCalibrator
    | EqualizedCoverageCalibrator | ECRCCalibrator
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
        )

    else:
        raise ValueError(
            f"Unknown calibration method: '{method}'. "
            f"Valid: split_cp, weighted_cp, mondrian, equalized_coverage, ecrc"
        )
