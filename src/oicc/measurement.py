"""Measurement model and synthetic data generators for OICC.

Model (log per-capita scale).  Latent log-rate theta_i in area/period i.
Each of K channels is a noisy, biased measurement:

    Y^c_i = alpha_c + beta_c * theta_i + eps^c_i,      c = 1..K

with, under the maintained assumptions,

    (A1) eps^1, ..., eps^K mutually independent given theta,
    (A3) location/scale normalization beta_1 = 1, alpha_1 = 0.

The generators below let us inject controlled VIOLATIONS so tests can prove
the specification test detects them:

    confound_pair : a shock shared by channels (0, 1) only  -> DETECTABLE
                    (Delta-perp): it makes those two covary MORE than the
                    one-factor model allows.
    common_mode   : a shock loaded on ALL channels along beta -> UNDETECTABLE
                    (Delta-parallel): absorbed into the latent factor, invisible
                    to any over-identification test.  This is the honest,
                    irreducible limitation and the generator proves it is
                    invisible (tests assert the spec test does NOT fire on it).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ArrayF = np.ndarray


@dataclass
class Channels:
    """A generated multi-channel dataset with known ground truth.

    Attributes
    ----------
    log_channels : (K, n) array
        Channel measurements Y on the log scale, one row per channel.
    theta : (n,) array
        The TRUE latent log-rate (never observed by the estimator; used only
        to score coverage/RMSE in controlled experiments).
    alpha, beta : (K,) arrays
        True channel intercepts and loadings.
    noise_sd : (K,) array
        True per-channel idiosyncratic noise standard deviations.
    x : (n,) array
        An area covariate (e.g. income) available to the estimator.
    """

    log_channels: ArrayF
    theta: ArrayF
    alpha: ArrayF
    beta: ArrayF
    noise_sd: ArrayF
    x: ArrayF

    @property
    def K(self) -> int:
        return int(self.log_channels.shape[0])

    @property
    def n(self) -> int:
        return int(self.log_channels.shape[1])


def _as_2d_channels(log_channels: ArrayF) -> ArrayF:
    """Validate and coerce channel input to a (K, n) float array."""
    arr = np.asarray(log_channels, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            f"log_channels must be 2-D (K, n); got shape {arr.shape}"
        )
    if arr.shape[0] < 2:
        raise ValueError(
            f"need at least 2 channels; got K={arr.shape[0]}"
        )
    if arr.shape[1] < 8:
        raise ValueError(
            f"need at least 8 observations for stable moments; got n={arr.shape[1]}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("log_channels contains non-finite values")
    return arr


def generate(
    n: int = 4000,
    seed: int = 0,
    *,
    K: int = 4,
    alpha: ArrayF | None = None,
    beta: ArrayF | None = None,
    noise_sd: ArrayF | None = None,
    theta_mean: float = 1.2,
    theta_x_coef: float = 0.5,
    theta_sd: float = 0.6,
    confound_pair: float = 0.0,
    common_mode: float = 0.0,
    xbias: float = 0.0,
    xbias_coefs: ArrayF | None = None,
) -> Channels:
    """Generate a controlled multi-channel dataset with known latent ground truth.

    Parameters
    ----------
    n : int
        Number of areas/periods (observations).
    seed : int
        RNG seed (deterministic; uses `np.random.default_rng`).
    K : int
        Number of channels (>= 2).
    alpha, beta, noise_sd : arrays of length K, optional
        True intercepts, loadings, noise SDs. Sensible defaults if omitted.
        `beta[0]` is forced to 1 and `alpha[0]` to 0 (the pivot normalization),
        matching the estimator's identification convention.
    theta_mean, theta_x_coef, theta_sd : float
        Latent log-rate is theta = theta_mean + theta_x_coef*x + N(0, theta_sd^2).
    confound_pair : float >= 0
        Strength of a DETECTABLE common shock injected into channels (0, 1) only.
        0 => assumptions hold.
    common_mode : float >= 0
        Strength of an UNDETECTABLE shock loaded on ALL channels along beta.
        This is Delta-parallel; the spec test provably cannot see it.
    xbias : float >= 0
        Scales a covariate-dependent (non-additive) component of channel bias,
        which VIOLATES the additive-bias assumption. 0 => additive bias only.
    xbias_coefs : (K,) array, optional
        Per-channel covariate-bias slopes (used only when xbias > 0).

    Returns
    -------
    Channels
    """
    if K < 2:
        raise ValueError(f"K must be >= 2; got {K}")
    if n < 8:
        raise ValueError(f"n must be >= 8; got {n}")
    if confound_pair < 0 or common_mode < 0 or xbias < 0:
        raise ValueError("shock strengths must be non-negative")

    rng = np.random.default_rng(seed)

    if alpha is None:
        alpha = np.linspace(-0.8, 0.2, K)
    if beta is None:
        beta = np.linspace(1.0, 1.4, K)
    if noise_sd is None:
        noise_sd = np.linspace(0.35, 0.60, K)
    if xbias_coefs is None:
        xbias_coefs = np.linspace(-0.4, 0.1, K)

    alpha = np.array(alpha, dtype=float)
    beta = np.array(beta, dtype=float)
    noise_sd = np.array(noise_sd, dtype=float)
    xbias_coefs = np.array(xbias_coefs, dtype=float)
    for name, v in [("alpha", alpha), ("beta", beta), ("noise_sd", noise_sd)]:
        if v.shape != (K,):
            raise ValueError(f"{name} must have shape ({K},); got {v.shape}")
    if np.any(noise_sd <= 0):
        raise ValueError("noise_sd entries must be strictly positive")

    # pivot normalization (channel 0 is the pivot)
    beta = beta.copy()
    alpha = alpha.copy()
    beta[0] = 1.0
    alpha[0] = 0.0

    x = rng.normal(0.0, 1.0, n)
    theta = theta_mean + theta_x_coef * x + rng.normal(0.0, theta_sd, n)

    # shared shocks
    pair_shock = rng.normal(0.0, 1.0, n) * confound_pair
    cm_shock = rng.normal(0.0, 1.0, n) * common_mode

    Y = np.empty((K, n), dtype=float)
    for c in range(K):
        bias = alpha[c]
        if xbias > 0.0:
            bias = bias + xbias * xbias_coefs[c] * x
        shared = beta[c] * cm_shock  # common-mode loads ALONG beta (invisible)
        if c in (0, 1):
            shared = shared + pair_shock  # detectable extra covariation
        Y[c] = bias + beta[c] * theta + shared + rng.normal(0.0, noise_sd[c], n)

    return Channels(
        log_channels=Y,
        theta=theta,
        alpha=alpha,
        beta=beta,
        noise_sd=noise_sd,
        x=x,
    )


def to_log_rate(counts: ArrayF, population: ArrayF, per: float = 1e5) -> ArrayF:
    """Convert integer counts + population into a stabilized log per-capita rate.

    Uses log1p on the per-`per` rate so that structural zeros map to 0 rather
    than -inf.  Returns an array the same shape as `counts`.
    """
    counts = np.asarray(counts, dtype=float)
    population = np.asarray(population, dtype=float)
    if counts.shape != population.shape:
        raise ValueError(
            f"counts {counts.shape} and population {population.shape} must match"
        )
    if np.any(population <= 0):
        raise ValueError("population must be strictly positive")
    rate = counts / population * per
    return np.log1p(rate)


@dataclass
class ProximalChannels:
    """Generated data with a KNOWN common-mode confounder and negative controls.

    signal_channels : (K, n) primary channels  Y^c = a_c + b_c*theta + l_c*W + eps.
    controls : (Q, n) NEGATIVE-CONTROL channels  N^q = a_q + m_q*W + eps  (no theta:
        they carry ZERO latent signal but DO load on the common-mode confounder W).
    theta : (n,) true latent log-rate.
    W : (n,) true common-mode confounder (loads on ALL signal channels along the
        factor direction, hence invisible to the over-identification test).
    beta, cm_load : (K,) signal loadings on theta and on W.
    ctrl_load : (Q,) control loadings on W.
    """

    signal_channels: ArrayF
    controls: ArrayF
    theta: ArrayF
    W: ArrayF
    beta: ArrayF
    cm_load: ArrayF
    ctrl_load: ArrayF

    @property
    def K(self) -> int:
        return int(self.signal_channels.shape[0])

    @property
    def Q(self) -> int:
        return int(self.controls.shape[0])

    @property
    def n(self) -> int:
        return int(self.signal_channels.shape[1])


def generate_proximal(
    n: int = 4000,
    seed: int = 0,
    *,
    K: int = 4,
    Q: int = 2,
    cm_strength: float = 1.0,
    theta_sd: float = 0.6,
    theta_x_coef: float = 0.5,
) -> ProximalChannels:
    """Generate signal channels with a common-mode confounder W + negative controls.

    Signal:   Y^c = alpha_c + beta_c*theta + cm_load_c*W + eps^c   (c = 1..K)
    Controls: N^q = a_q + ctrl_load_q*W + nu^q                     (q = 1..Q)

    W loads on the signal channels ALONG a direction correlated with beta (so the
    over-identification test is blind to it). The controls give an INDEPENDENT
    handle on W: they respond to W but NOT to theta. `cm_strength` scales Var(W).

    Q >= 2 controls point-identify the W-contamination (see proximal.py);
    Q == 1 gives detection + partial correction only.
    """
    if K < 3:
        raise ValueError(f"K must be >= 3; got {K}")
    if Q < 1:
        raise ValueError(f"Q must be >= 1; got {Q}")
    if cm_strength < 0:
        raise ValueError(f"cm_strength must be >= 0; got {cm_strength}")

    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, n)
    theta = 1.2 + theta_x_coef * x + rng.normal(0.0, theta_sd, n)
    W = rng.normal(0.0, 1.0, n) * cm_strength

    beta = np.linspace(1.0, 1.4, K)
    beta[0] = 1.0
    # common-mode loads roughly along beta (the invisible direction), with a
    # little heterogeneity so it is a realistic near-parallel confounder.
    cm_load = beta * (0.8 + 0.1 * rng.standard_normal(K))
    noise_sd = np.linspace(0.35, 0.55, K)
    alpha = np.linspace(-0.5, 0.3, K)
    alpha[0] = 0.0

    Y = np.empty((K, n))
    for c in range(K):
        Y[c] = (alpha[c] + beta[c] * theta + cm_load[c] * W
                + rng.normal(0.0, noise_sd[c], n))

    ctrl_load = np.linspace(0.9, 1.2, Q)
    ctrl_noise = np.linspace(0.30, 0.45, Q)
    ctrl_alpha = np.linspace(-0.2, 0.2, Q)
    N = np.empty((Q, n))
    for q in range(Q):
        N[q] = ctrl_alpha[q] + ctrl_load[q] * W + rng.normal(0.0, ctrl_noise[q], n)

    return ProximalChannels(
        signal_channels=Y,
        controls=N,
        theta=theta,
        W=W,
        beta=beta,
        cm_load=cm_load,
        ctrl_load=ctrl_load,
    )
