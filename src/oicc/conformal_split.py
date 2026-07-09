"""Split-conformal calibration for OICC, with an HONEST two-interval design.

The leave-pivot-out residual  R_i = Y^pivot_i - theta_hat_i  splits, under the
one-factor model, as  R_i = S_i + eps^pivot_i  with  S_i = theta_i - theta_hat_i
INDEPENDENT of eps^pivot_i (theta_hat uses only the non-pivot channels).

We return TWO clearly-separated intervals:

  (1) EXACT, finite-sample, distribution-free interval for the OBSERVED pivot
      channel value  Y^pivot_{n+1}.  Pure split conformal on {R_i} over a
      held-out calibration fold: with the (1 + 1/n) quantile, coverage is
      >= 1 - alpha for exchangeable units.  This is a REAL guarantee and needs
      no model beyond exchangeability.

  (2) MODEL-ASSISTED interval for the LATENT target theta_{n+1}.  We deconvolve
      eps^pivot out of the calibration residuals to recover the law of S, then
      form theta_hat_{n+1} + [q_{alpha/2}(S), q_{1-alpha/2}(S)] (plus sensitivity
      inflation).  Coverage of the *latent* theta is ASYMPTOTIC and relies on the
      measurement model (error-CF identification via K>=3 channels).  We label it
      as such.  No distribution-free finite-sample latent interval can exist
      (impossibility: the common-mode direction is unidentified) -- so this is the
      honest best.

Sample splitting (crucial for the exact guarantee): moments/loadings are fit on
a TRAIN fold; the residuals used for the conformal quantile come from a disjoint
CALIBRATION fold, so they are exchangeable with a fresh test residual.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from oicc.measurement import _as_2d_channels
from oicc.moments import estimate_factor_moments
from oicc.deconvolve import blup_from_subset
from oicc.spec_test import overid_wald_test
from oicc.cf_deconv import deconvolve_error_law

ArrayF = np.ndarray

_VAR_FLOOR = 1e-6


@dataclass
class SplitConformalResult:
    """Two-interval split-conformal output.

    theta_hat : (n_test,) latent point estimate for test units.
    obs_lower, obs_upper : (n_test,) EXACT distribution-free interval for the
        observed pivot channel value (finite-sample >= 1-alpha coverage).
    lat_lower, lat_upper : (n_test,) model-assisted interval for the LATENT theta.
    method : "cf" or "gaussian" (how the S-law was recovered).
    infl_data, infl_knob : sensitivity inflations applied to the latent interval.
    delta_perp_hat, gamma_cm : provenance of the inflations.
    n_cal : int, calibration-fold size (drives the finite-sample slack).
    """

    theta_hat: ArrayF
    obs_lower: ArrayF
    obs_upper: ArrayF
    lat_lower: ArrayF
    lat_upper: ArrayF
    method: str
    infl_data: float
    infl_knob: float
    delta_perp_hat: float
    gamma_cm: float
    n_cal: int


def split_conformal_latent(
    log_channels: ArrayF,
    alpha: float = 0.1,
    pivot: int = 0,
    gamma_cm: float = 0.0,
    *,
    cal_frac: float = 0.5,
    seed: int = 0,
    use_spec_test: bool = True,
    latent_method: str = "gaussian",
) -> SplitConformalResult:
    """Split-conformal latent + observed intervals with a real finite-sample half.

    Parameters
    ----------
    log_channels : (K, n) array, K >= 3.
    alpha : miscoverage in (0, 1).
    pivot : held-out channel index.
    gamma_cm : common-mode sensitivity knob (>= 0).
    cal_frac : fraction of the half-sample used for the conformal calibration.
    seed : RNG seed for the train/calibration/test split.
    use_spec_test : add data-driven inflation from the over-ID statistic.
    latent_method : "gaussian" (default) uses moment-exact Gaussian quantiles for
        the recovery error S -- justified because S = -(1/(K-1)) sum eps_k is a
        near-Gaussian AVERAGE of channel errors with variance known exactly from
        the moment identity Var(S)=Var(R)-Var(eps_pivot); "cf" uses nonparametric
        characteristic-function deconvolution (for heavy-tailed channel noise).

    Returns
    -------
    SplitConformalResult
    """
    Y = _as_2d_channels(log_channels)
    K, n = Y.shape
    if K < 3:
        raise ValueError(f"need K >= 3; got {K}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1); got {alpha}")
    if not (0 <= pivot < K):
        raise ValueError(f"pivot in [0,{K}); got {pivot}")
    if not (0.1 <= cal_frac <= 0.9):
        raise ValueError(f"cal_frac must be in [0.1, 0.9]; got {cal_frac}")
    if gamma_cm < 0:
        raise ValueError(f"gamma_cm must be >= 0; got {gamma_cm}")
    if n < 24:
        raise ValueError(
            f"split_conformal_latent needs n >= 24 for three viable folds "
            f"(train >= 8, cal >= 2, test >= 1); got n={n}. Use "
            f"leave_pivot_out_conformal for small samples."
        )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    # three disjoint folds: train (fit loadings), cal (conformal quantile),
    # test (produce/score intervals). This makes cal and test residuals
    # exchangeable given the train-estimated loadings -> the exact guarantee holds.
    n_cal = int(round(cal_frac * (n // 2)))
    n_cal = min(max(n_cal, 4), (n // 2) - 2)
    n_train = n - 2 * n_cal
    if n_train < 8:
        # fall back: shrink cal folds so training has enough for stable moments
        n_cal = max(4, (n - 8) // 2)
        n_train = n - 2 * n_cal
    tr_idx = perm[:n_train]
    cal_idx = perm[n_train:n_train + n_cal]
    test_idx = perm[n_train + n_cal:]
    # invariant guaranteed by the n>=24 check above; assert to fail loudly (never
    # silently) if the fold arithmetic is ever changed.
    if not (n_train >= 8 and n_cal >= 2 and test_idx.size >= 1):
        raise ValueError(
            f"fold sizing failed for n={n} (train={n_train}, cal={n_cal}, "
            f"test={test_idx.size}); increase n or adjust cal_frac"
        )
    others = [c for c in range(K) if c != pivot]

    # --- fit moments on TRAIN fold only (keeps calibration residuals honest) ---
    full_tr = estimate_factor_moments(Y[:, tr_idx], pivot=pivot)
    pivot_noise_var = float(np.clip(full_tr.noise_var[pivot], _VAR_FLOOR, None))

    # helper: theta_hat for a set of columns using TRAIN-fold loadings
    def theta_hat_for(cols: ArrayF) -> ArrayF:
        anchor = float(np.mean(Y[pivot, tr_idx]))   # E[theta] anchor from train
        est = blup_from_subset(Y[:, cols], full_tr, others, anchor_mean=anchor)
        return est.theta_hat

    # calibration residuals R = Y_pivot - theta_hat (computable, = S + eps_pivot)
    theta_cal = theta_hat_for(cal_idx)
    R_cal = Y[pivot, cal_idx] - theta_cal
    n_cal = R_cal.size

    # === (1) EXACT split-conformal interval for the OBSERVED pivot value ===
    # symmetric absolute-residual score => two-sided interval around theta_hat.
    scores = np.abs(R_cal - np.median(R_cal))
    level = min(1.0, np.ceil((n_cal + 1) * (1 - alpha)) / n_cal)  # (1+1/n) corr.
    q_abs = float(np.quantile(scores, level, method="higher"))
    med_R = float(np.median(R_cal))

    # === (2) LATENT interval: recover the law of S = theta - theta_hat ===
    # S is an AVERAGE of channel errors, hence near-Gaussian with variance known
    # exactly from the moment identity; Gaussian quantiles are the justified
    # default. CF deconvolution is available for heavy-tailed channel noise.
    if latent_method == "gaussian":
        from scipy import stats as _stats
        var_s = max(float(np.var(R_cal)) - pivot_noise_var, _VAR_FLOOR)
        med_S = float(np.median(R_cal))
        z = float(_stats.norm.ppf(1.0 - alpha / 2.0))
        q_lo = med_S - z * np.sqrt(var_s)
        q_hi = med_S + z * np.sqrt(var_s)
        method = "gaussian"
    elif latent_method == "cf":
        dd = deconvolve_error_law(R_cal, pivot_noise_var)
        q_lo = float(dd.quantile(alpha / 2.0))
        q_hi = float(dd.quantile(1.0 - alpha / 2.0))
        method = dd.method
    else:
        raise ValueError(f"latent_method must be 'gaussian' or 'cf'; got {latent_method}")

    # sensitivity inflation (latent interval only)
    delta_perp_hat = 0.0
    infl_data = 0.0
    if use_spec_test:
        spec = overid_wald_test(Y, seed=seed)
        delta_perp_hat = spec.delta_perp_hat
        infl_data = float(delta_perp_hat * np.sqrt(max(full_tr.var_theta,
                                                       _VAR_FLOOR)))
    infl_knob = float(gamma_cm * np.sqrt(max(full_tr.var_theta, _VAR_FLOOR)))
    infl = infl_data + infl_knob

    # === evaluate on the disjoint TEST fold ===
    theta_test = theta_hat_for(test_idx)

    obs_lower = theta_test + med_R - q_abs
    obs_upper = theta_test + med_R + q_abs
    lat_lower = theta_test + q_lo - infl
    lat_upper = theta_test + q_hi + infl

    return SplitConformalResult(
        theta_hat=theta_test,
        obs_lower=obs_lower,
        obs_upper=obs_upper,
        lat_lower=lat_lower,
        lat_upper=lat_upper,
        method=method,
        infl_data=infl_data,
        infl_knob=infl_knob,
        delta_perp_hat=delta_perp_hat,
        gamma_cm=gamma_cm,
        n_cal=n_cal,
    )
