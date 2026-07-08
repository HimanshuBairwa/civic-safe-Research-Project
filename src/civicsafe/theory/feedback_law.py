"""Feedback Amplification Law — the theoretical core of CIVIC-SAFE.

This module formalises *why* a forecaster embedded in an allocation loop can
become **confidently wrong**: its coverage of recorded data stays valid while
its coverage of the *true* latent process collapses. The results here are the
genuinely novel contribution of the project; the ZINB/GNN machinery elsewhere
is prior art (Zhuang et al., KDD 2022, STZINB-GNN; STMGNN-ZINB 2024) and serves
as the base predictor, not the contribution.

Model (Allocation under Observation-Biased Feedback, AOBF)
----------------------------------------------------------
* Latent incidence intensity ``lambda_s > 0`` per spatial cell ``s`` — never observed.
* Policy allocates attention ``a_s = pi(mu_s)`` from the model's recorded-rate
  estimate ``mu_s`` (any smooth increasing policy; e.g. patrol proportional to
  a predicted upper quantile).
* Observation-biased recording (the Ensign et al. 2018 mechanism):
  ``y_s ~ Poisson(lambda_s * g(a_s))`` where detection gain ``g`` is smooth and
  increasing — more attention inflates what gets recorded.
* A consistent online learner tracks the recorded mean, so the feedback fixed
  point satisfies ``mu_s = lambda_s * g(pi(mu_s))``.

Feedback gain
-------------
``kappa = (d log a / d log mu) * (d log g / d log a)`` — the product of the
policy elasticity and the detection elasticity, evaluated at the fixed point.

Main results (all verified numerically in ``tests/test_feedback_law.py``)
-------------------------------------------------------------------------
1. **Universal Amplification Law** (:func:`amplification_exponent`):
   ``d log mu_s / d log lambda_s = 1 / (1 - kappa)``.
   Recorded disparity is true disparity raised to ``1/(1-kappa)``. The exponent
   has a pole at ``kappa* = 1`` — the runaway threshold. The result is
   coordinate-free (holds for any smooth increasing ``pi``, ``g``), not a
   power-law artefact.

2. **Runaway-Discrimination Corollary** (:func:`disparity_ratio`):
   two groups with identical true incidence but initial recording-bias ratio
   ``b`` reach fixed-point recorded disparity ``b ** (1 / (1 - kappa))`` — an
   exact functional form for Ensign et al.'s runaway feedback.

3. **Passive impossibility / Active identification duality**
   (:func:`identify_kappa_did`): the confidently-wrong state is *not*
   detectable from passively observed data (a biased world and an honest world
   induce identical observables), but ``kappa`` is point-identified by a
   difference-in-differences on log recorded rates after an exogenous shock to
   detection sensitivity (a staggered ShotSpotter / patrol-policy change).

References
----------
* Ensign, Friedler, Neville, Scheidegger, Venkatasubramanian (2018),
  "Runaway Feedback Loops in Predictive Policing", FAT*.
* Perdomo, Zrnic, Mendler-Dünner, Hardt (2020), "Performative Prediction", ICML.
* Gibbs & Candès (2021), "Adaptive Conformal Inference Under Distribution Shift".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

__all__ = [
    "amplification_exponent",
    "disparity_ratio",
    "power_law_fixed_point",
    "general_fixed_point",
    "local_feedback_gain",
    "identify_kappa_did",
    "FeedbackLawResult",
]

# Runaway threshold: the pole of the amplification exponent.
KAPPA_STAR: float = 1.0


def amplification_exponent(kappa: float) -> float:
    """Return the amplification exponent ``1 / (1 - kappa)``.

    ``mu_s / mu_r = (lambda_s / lambda_r) ** amplification_exponent(kappa)``.

    Args:
        kappa: Feedback gain in ``[0, 1)``. Values ``>= 1`` are the runaway
            regime with no finite stable amplification.

    Returns:
        The exponent ``1 / (1 - kappa)``; ``inf`` at ``kappa == 1``.

    Raises:
        ValueError: if ``kappa`` is negative.
    """
    if kappa < 0:
        raise ValueError(f"kappa must be >= 0, got {kappa}")
    if kappa >= KAPPA_STAR:
        return float("inf")
    return 1.0 / (1.0 - kappa)


def disparity_ratio(initial_bias: float, kappa: float) -> float:
    """Fixed-point recorded disparity for equal-truth groups (Corollary).

    Two groups with identical latent incidence but an initial recording-bias
    ratio ``initial_bias`` diverge to ``initial_bias ** (1 / (1 - kappa))``.

    Args:
        initial_bias: Initial ratio of recorded rates between the two groups
            (``> 0``; ``1.0`` means no initial bias).
        kappa: Feedback gain in ``[0, 1)``.

    Returns:
        The amplified disparity ratio; ``inf`` in the runaway regime.
    """
    if initial_bias <= 0:
        raise ValueError(f"initial_bias must be > 0, got {initial_bias}")
    exponent = amplification_exponent(kappa)
    if exponent == float("inf"):
        return float("inf") if initial_bias > 1.0 else (1.0 if initial_bias == 1.0 else 0.0)
    return float(initial_bias**exponent)


def power_law_fixed_point(
    lam: np.ndarray,
    beta: float,
    rho: float,
    iters: int = 8000,
    tol: float = 1e-11,
) -> np.ndarray | None:
    """Solve the power-law feedback fixed point ``mu = lam * (mu/mean(mu))**(beta*rho)``.

    Args:
        lam: Latent intensities, shape ``(S,)``, all positive.
        beta: Policy elasticity (``a = (mu/M)**beta``).
        rho: Detection elasticity (``g(a) = a**rho``).
        iters: Maximum fixed-point iterations.
        tol: Relative convergence tolerance.

    Returns:
        The fixed-point recorded rates ``mu``, or ``None`` if the iteration
        diverges (runaway regime, ``kappa = beta * rho >= 1``).
    """
    lam = np.asarray(lam, dtype=float)
    mu = lam.copy()
    kappa = beta * rho
    for _ in range(iters):
        m = mu.mean()
        nxt = lam * (mu / m) ** kappa
        if not np.all(np.isfinite(nxt)) or nxt.max() > 1e12:
            return None
        if np.max(np.abs(nxt - mu)) < tol * m:
            return nxt
        mu = nxt
    return mu


def general_fixed_point(
    lam: np.ndarray,
    policy: Callable[[np.ndarray, float], np.ndarray],
    detection: Callable[[np.ndarray], np.ndarray],
    iters: int = 20000,
    tol: float = 1e-13,
    damping: float = 0.5,
) -> np.ndarray:
    """Solve the *general* feedback fixed point ``mu = lam * g(pi(mu, M))``.

    Works for any smooth increasing ``policy`` and ``detection`` (not just the
    power law), demonstrating the coordinate-free nature of the pole.

    Args:
        lam: Latent intensities, shape ``(S,)``.
        policy: ``pi(mu, M) -> a`` attention, given rates and their mean ``M``.
        detection: ``g(a) -> gain`` recording amplification.
        iters: Maximum iterations.
        tol: Relative convergence tolerance.
        damping: Damping factor in ``(0, 1]`` for stability of general maps.

    Returns:
        Fixed-point recorded rates ``mu``.
    """
    lam = np.asarray(lam, dtype=float)
    mu = lam.copy()
    for _ in range(iters):
        m = mu.mean()
        nxt = lam * detection(policy(mu, m))
        if np.max(np.abs(nxt - mu)) < tol * m:
            return nxt
        mu = (1.0 - damping) * mu + damping * nxt
    return mu


def local_feedback_gain(
    mu: np.ndarray,
    policy: Callable[[np.ndarray, float], np.ndarray],
    detection: Callable[[np.ndarray], np.ndarray],
    h: float = 1e-5,
) -> np.ndarray:
    """Estimate the local feedback gain ``kappa`` at a fixed point via elasticities.

    ``kappa = (d log a / d log mu) * (d log g / d log a)`` computed by central
    differences. This is the quantity whose value relative to 1 governs the
    amplification exponent for arbitrary ``policy``/``detection``.

    Args:
        mu: Fixed-point recorded rates, shape ``(S,)``.
        policy: ``pi(mu, M) -> a``.
        detection: ``g(a) -> gain``.
        h: Relative step for finite differences.

    Returns:
        Per-cell local feedback gain, shape ``(S,)``.
    """
    mu = np.asarray(mu, dtype=float)
    m = mu.mean()
    a = policy(mu, m)
    a_up, a_dn = policy(mu * (1 + h), m), policy(mu * (1 - h), m)
    beta_eff = (np.log(a_up) - np.log(a_dn)) / (2 * h)
    g_up, g_dn = detection(a * (1 + h)), detection(a * (1 - h))
    rho_eff = (np.log(g_up) - np.log(g_dn)) / (2 * h)
    return beta_eff * rho_eff


@dataclass
class FeedbackLawResult:
    """Result of estimating the feedback gain from a natural experiment.

    Attributes:
        kappa_hat: Point estimate of the feedback gain.
        did_estimate: The observed difference-in-differences on log recorded rates.
        recovered: Whether a finite estimate in ``[0, 1)`` was recovered.
    """

    kappa_hat: float
    did_estimate: float
    recovered: bool


def identify_kappa_did(
    mu_pre_treated: np.ndarray,
    mu_pre_control: np.ndarray,
    mu_post_treated: np.ndarray,
    mu_post_control: np.ndarray,
    delta: float,
    beta: float = 1.0,
    lam: np.ndarray | None = None,
    treated_mask: np.ndarray | None = None,
    grid: np.ndarray | None = None,
) -> FeedbackLawResult:
    """Identify the feedback gain ``kappa`` from a staggered detection shock.

    Active-identification half of the passive/active duality: although the
    confidently-wrong state is invisible to passive observation, an exogenous
    shock ``delta`` to detection sensitivity in treated cells identifies
    ``kappa`` via a difference-in-differences on *log recorded rates* — without
    ever observing the latent intensity.

    Two usage modes:

    * **Structural inversion** (preferred, exact): pass ``lam`` and
      ``treated_mask``; the estimator re-solves the fixed point on a ``kappa``
      grid and matches the observed DiD. This is what the field design targets.
    * **Reduced-form fallback**: if ``lam`` is not supplied, returns the raw DiD
      as a monotone summary (``recovered=False``) for diagnostics only.

    Args:
        mu_pre_treated: Recorded rates, treated cells, pre-shock.
        mu_pre_control: Recorded rates, control cells, pre-shock.
        mu_post_treated: Recorded rates, treated cells, post-shock.
        mu_post_control: Recorded rates, control cells, post-shock.
        delta: Known multiplicative shock to the detection elasticity in
            treated cells (``rho -> rho * (1 + delta)``).
        beta: Known policy elasticity (an operator design parameter).
        lam: Latent intensities (for structural inversion, simulation/anchor only).
        treated_mask: Boolean mask of treated cells (structural inversion).
        grid: Optional ``kappa`` search grid; defaults to ``linspace(0.02, 0.98, 481)``.

    Returns:
        A :class:`FeedbackLawResult`.
    """

    def _dlog(post: np.ndarray, pre: np.ndarray) -> float:
        post = np.asarray(post, dtype=float)
        pre = np.asarray(pre, dtype=float)
        return float(np.log(post).mean() - np.log(pre).mean())

    did = (_dlog(mu_post_treated, mu_pre_treated)
           - _dlog(mu_post_control, mu_pre_control))

    if lam is None or treated_mask is None:
        return FeedbackLawResult(kappa_hat=float("nan"), did_estimate=did, recovered=False)

    treated_mask = np.asarray(treated_mask, dtype=bool)
    if grid is None:
        grid = np.linspace(0.02, 0.98, 481)

    def predicted_did(rho: float) -> float:
        rho_vec = np.where(treated_mask, rho * (1 + delta), rho)
        m0 = power_law_fixed_point(lam, beta, rho)
        mp = _hetero_fixed_point(lam, beta, rho_vec)
        if m0 is None or mp is None:
            return float("inf")
        dt = float(np.log(mp[treated_mask]).mean() - np.log(m0[treated_mask]).mean())
        dc = float(np.log(mp[~treated_mask]).mean() - np.log(m0[~treated_mask]).mean())
        return dt - dc

    best_rho = min(grid, key=lambda rr: abs(predicted_did(rr) - did))
    kappa_hat = float(beta * best_rho)
    return FeedbackLawResult(
        kappa_hat=kappa_hat,
        did_estimate=did,
        recovered=bool(0.0 <= kappa_hat < 1.0),
    )


def _hetero_fixed_point(
    lam: np.ndarray,
    beta: float,
    rho_vec: np.ndarray,
    iters: int = 8000,
    tol: float = 1e-11,
) -> np.ndarray | None:
    """Fixed point with heterogeneous detection elasticity ``rho_vec`` per cell."""
    lam = np.asarray(lam, dtype=float)
    rho_vec = np.asarray(rho_vec, dtype=float)
    mu = lam.copy()
    for _ in range(iters):
        m = mu.mean()
        a = (mu / m) ** beta
        nxt = lam * a**rho_vec
        if not np.all(np.isfinite(nxt)) or nxt.max() > 1e12:
            return None
        if np.max(np.abs(nxt - mu)) < tol * m:
            return nxt
        mu = nxt
    return mu
