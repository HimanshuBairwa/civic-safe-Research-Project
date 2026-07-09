"""Latent recovery (BLUP / empirical-Bayes posterior mean) for OICC.

Given the one-factor moments (beta, var_theta, noise_var), the best linear
unbiased predictor of theta_i from the channel vector Y_i is the Gaussian
posterior mean under

    Y_i | theta_i ~ N(alpha + beta*theta_i, diag(noise_var)),
    theta_i       ~ N(mu_theta, var_theta).

Posterior mean (Bayesian linear Gaussian):

    prec = 1/var_theta + sum_c beta_c^2 / noise_var_c
    theta_hat_i = ( mu_theta/var_theta
                    + sum_c beta_c*(Y^c_i - alpha_c)/noise_var_c ) / prec

Intercepts alpha are not separately identified from mu_theta without a location
normalization, so we estimate the combined location by anchoring the pivot
channel's mean (alpha_pivot = 0 by convention) and centering.  Concretely we
work with de-meaned channels and add back the pivot mean, which is exactly the
identified location.  This yields per-unit theta_hat whose SHAPE is BLUP-optimal
and whose LEVEL is anchored to the pivot channel.

The posterior variance (same for all i in the Gaussian model) is returned so the
conformal layer can, if desired, widen for recovery uncertainty.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from oicc.measurement import _as_2d_channels
from oicc.moments import FactorMoments, estimate_factor_moments

ArrayF = np.ndarray

_VAR_FLOOR = 1e-6


@dataclass
class LatentEstimate:
    """Recovered latent log-rate and its posterior precision.

    theta_hat : (n,) posterior-mean estimate of the latent log-rate.
    post_var : float, Gaussian posterior variance of theta given a channel vector.
    moments : the FactorMoments used.
    weights : (K,) BLUP weights on the de-meaned channels (for interpretability).
    """

    theta_hat: ArrayF
    post_var: float
    moments: FactorMoments
    weights: ArrayF


def deconvolve_blup(
    log_channels: ArrayF,
    moments: FactorMoments | None = None,
    pivot: int = 0,
) -> LatentEstimate:
    """Empirical-Bayes BLUP recovery of the latent log-rate.

    Parameters
    ----------
    log_channels : (K, n) array
    moments : FactorMoments, optional
        Precomputed moments; estimated from the data if omitted.
    pivot : int
        Pivot channel for the location anchor (defaults to 0).

    Returns
    -------
    LatentEstimate
    """
    Y = _as_2d_channels(log_channels)
    K, n = Y.shape
    if moments is None:
        moments = estimate_factor_moments(Y, pivot=pivot)

    beta = moments.beta
    noise_var = np.clip(moments.noise_var, _VAR_FLOOR, None)
    var_theta = max(moments.var_theta, _VAR_FLOOR)

    # De-mean each channel (removes alpha_c up to the shared latent mean).
    means = Y.mean(axis=1, keepdims=True)
    Yc = Y - means

    # BLUP weights on the de-meaned, loading-scaled channels.
    prec = 1.0 / var_theta + np.sum(beta**2 / noise_var)
    # numerator contribution per channel: beta_c / noise_var_c
    w = beta / noise_var / prec  # (K,)
    theta_centered = w @ Yc  # (n,)

    # Anchor the level to the pivot channel's mean (alpha_pivot = 0, beta_pivot=1
    # so E[Y_pivot] = E[theta]); this is the identified location.
    theta_hat = theta_centered + means[pivot, 0]

    post_var = float(1.0 / prec)

    return LatentEstimate(
        theta_hat=theta_hat,
        post_var=post_var,
        moments=moments,
        weights=w,
    )


def blup_from_subset(
    log_channels: ArrayF,
    moments: FactorMoments,
    subset: list[int],
    anchor_mean: float,
) -> LatentEstimate:
    """BLUP of theta from a SUBSET of channels, using loadings on the pivot scale.

    Unlike `deconvolve_blup`, this does NOT re-estimate loadings on the subset
    (which would re-normalize theta to a different scale). It uses the supplied
    full-model `moments` (whose loadings are on the pivot's scale, beta_pivot=1)
    restricted to `subset`, and anchors the level to `anchor_mean` (typically the
    held-out pivot channel's mean, which estimates E[theta] since alpha_pivot=0,
    beta_pivot=1).

    This is the scale-correct estimator required by the leave-pivot-out conformal
    construction: theta_hat is on the SAME scale as the held-out pivot channel.
    """
    Y = _as_2d_channels(log_channels)
    K, n = Y.shape
    subset = list(subset)
    if len(subset) < 2:
        raise ValueError(f"subset needs >= 2 channels; got {subset}")
    if any(not (0 <= c < K) for c in subset):
        raise ValueError(f"subset indices out of range for K={K}: {subset}")

    beta = moments.beta[subset]
    noise_var = np.clip(moments.noise_var[subset], _VAR_FLOOR, None)
    var_theta = max(moments.var_theta, _VAR_FLOOR)

    Ysub = Y[subset]
    means = Ysub.mean(axis=1, keepdims=True)
    Yc = Ysub - means  # deviations: beta_c*(theta-E theta) + eps_c

    prec = 1.0 / var_theta + np.sum(beta**2 / noise_var)
    w = beta / noise_var / prec  # weights on the deviations
    theta_dev = w @ Yc
    theta_hat = anchor_mean + theta_dev

    return LatentEstimate(
        theta_hat=theta_hat,
        post_var=float(1.0 / prec),
        moments=moments,
        weights=w,
    )
