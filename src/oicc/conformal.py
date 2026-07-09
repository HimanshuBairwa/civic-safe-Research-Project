"""Leave-pivot-out conformal prediction for the LATENT target (OICC core, C1).

The problem the standard toolbox cannot solve.  Conformal prediction and
prediction-powered inference both need to *observe* the target on a calibration
sample. Here the target theta_i (true latent log-rate) is NEVER observed on any
sample. So there is no ordinary nonconformity score |theta - theta_hat|.

The OICC move.  Split the channels into the pivot (channel 0) and the rest.
Build theta_hat_i = m(Y^{2:K}_i) using ONLY the non-pivot channels. Then the
residual against the held-out pivot channel,

    R_i = Y^0_i - theta_hat_i,

is fully COMPUTABLE, and under the model equals  S_i + eps^0_i  where
S_i = (theta_i - theta_hat_i) is the (infeasible) latent-recovery error. Because
eps^0 is independent of everything else, the law of R is the convolution of the
law of S with the pivot-noise law; the quantiles of S are recovered by
subtracting the pivot-noise spread, giving a prediction interval for theta_i:

    C(i) = theta_hat_i + [ q_{alpha/2}(S) - infl,  q_{1-alpha/2}(S) + infl ].

We estimate the S-quantiles by a conservative moment/deconvolution step:
Var(S) = Var(R) - Var(eps^0), then use either a Gaussian or an empirical
quantile of a deconvolved residual sample. `infl` = f(delta_perp_hat) +
g(gamma_cm) is the sensitivity inflation: a DATA-DRIVEN part from the
over-identification test (detectable violations) plus a USER-KNOB part for the
irreducible common-mode direction.

Coverage claim (honest).  Under the maintained model (A1-A3) with correct
moments, C(i) covers theta_i at level >= 1 - alpha up to deconvolution error.
The interval WIDENS visibly where the spec test fires (delta_perp_hat large) and
where the user sets a larger gamma_cm. It is NOT claimed valid against
common-mode violations of unbounded size.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from oicc.measurement import _as_2d_channels
from oicc.deconvolve import deconvolve_blup, blup_from_subset
from oicc.moments import estimate_factor_moments
from oicc.spec_test import overid_wald_test

ArrayF = np.ndarray

_VAR_FLOOR = 1e-6


@dataclass
class ConformalResult:
    """Latent prediction intervals and their provenance.

    theta_hat : (n,) latent point estimate (from non-pivot channels).
    lower, upper : (n,) latent-scale interval bounds.
    var_s : float, estimated variance of the latent-recovery error S.
    infl_data : float, data-driven inflation f(delta_perp_hat).
    infl_knob : float, common-mode inflation g(gamma_cm).
    delta_perp_hat : float, detectable-violation magnitude from the spec test.
    gamma_cm : float, the common-mode user knob used.
    pivot : int.
    """

    theta_hat: ArrayF
    lower: ArrayF
    upper: ArrayF
    var_s: float
    infl_data: float
    infl_knob: float
    delta_perp_hat: float
    gamma_cm: float
    pivot: int


def leave_pivot_out_conformal(
    log_channels: ArrayF,
    alpha: float = 0.1,
    pivot: int = 0,
    gamma_cm: float = 0.0,
    *,
    use_spec_test: bool = True,
    spec_seed: int = 0,
    empirical_quantiles: bool = True,
) -> ConformalResult:
    """Build leave-pivot-out latent prediction intervals.

    Parameters
    ----------
    log_channels : (K, n) array, K >= 3 required (need pivot + >=2 others).
    alpha : float in (0, 1), miscoverage (coverage 1 - alpha).
    pivot : int, the held-out channel used to form the computable residual.
    gamma_cm : float >= 0, common-mode sensitivity knob (extra half-width in
        latent SD units for the irreducible Delta-parallel direction).
    use_spec_test : bool, if True add a data-driven inflation from the over-ID
        test's detectable-violation magnitude.
    spec_seed : int, RNG seed for the spec-test bootstrap.
    empirical_quantiles : bool, if True use deconvolved empirical residual
        quantiles; else Gaussian quantiles from Var(S).

    Returns
    -------
    ConformalResult
    """
    Y = _as_2d_channels(log_channels)
    K, n = Y.shape
    if K < 3:
        raise ValueError(
            f"leave-pivot-out needs K >= 3 (pivot + >=2 others); got K={K}"
        )
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1); got {alpha}")
    if not (0 <= pivot < K):
        raise ValueError(f"pivot must be in [0,{K}); got {pivot}")
    if gamma_cm < 0:
        raise ValueError(f"gamma_cm must be >= 0; got {gamma_cm}")

    others = [c for c in range(K) if c != pivot]

    # Full-model moments (loadings on the PIVOT scale: beta_pivot = 1).
    full = estimate_factor_moments(Y, pivot=pivot)

    # theta_hat from the NON-pivot channels, on the pivot scale, anchored to the
    # held-out pivot channel's mean (which estimates E[theta] since alpha_pivot=0
    # and beta_pivot=1). This keeps theta_hat on the SAME scale as Y_pivot, so the
    # residual below is a clean S + eps_pivot.
    Y_pivot = Y[pivot]
    anchor = float(np.mean(Y_pivot))
    est = blup_from_subset(Y, full, others, anchor_mean=anchor)
    theta_hat = est.theta_hat

    # Computable residual against the held-out pivot channel: R = S + eps_pivot.
    R = Y_pivot - theta_hat

    # Pivot-noise variance from the full-model moments.
    pivot_noise_var = float(np.clip(full.noise_var[pivot], _VAR_FLOOR, None))

    var_R = float(np.var(R))
    var_s = max(var_R - pivot_noise_var, _VAR_FLOOR)

    # Recover S-quantiles.
    if empirical_quantiles:
        # Deconvolve R -> S by moment-matched Gaussian widening/narrowing of the
        # centered residual sample: rescale centered R to have Var = var_s.
        Rc = R - np.median(R)
        scale = np.sqrt(var_s / max(var_R, _VAR_FLOOR))
        s_sample = Rc * scale
        q_lo = float(np.quantile(s_sample, alpha / 2.0))
        q_hi = float(np.quantile(s_sample, 1.0 - alpha / 2.0))
    else:
        z = stats.norm.ppf(1.0 - alpha / 2.0)
        q_hi = z * np.sqrt(var_s)
        q_lo = -q_hi

    # Sensitivity inflation.
    delta_perp_hat = 0.0
    infl_data = 0.0
    if use_spec_test:
        spec = overid_wald_test(Y, seed=spec_seed)
        delta_perp_hat = spec.delta_perp_hat
        # f(delta): extra half-width proportional to detectable-violation size,
        # in latent-SD units. Monotone, 0 under H0.
        infl_data = float(delta_perp_hat * np.sqrt(max(full.var_theta, _VAR_FLOOR)))

    # g(gamma_cm): common-mode half-width in latent-SD units.
    infl_knob = float(gamma_cm * np.sqrt(max(full.var_theta, _VAR_FLOOR)))

    infl = infl_data + infl_knob
    lower = theta_hat + q_lo - infl
    upper = theta_hat + q_hi + infl

    return ConformalResult(
        theta_hat=theta_hat,
        lower=lower,
        upper=upper,
        var_s=var_s,
        infl_data=infl_data,
        infl_knob=infl_knob,
        delta_perp_hat=delta_perp_hat,
        gamma_cm=gamma_cm,
        pivot=pivot,
    )
