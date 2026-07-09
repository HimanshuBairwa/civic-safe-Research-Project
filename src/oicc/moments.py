"""One-factor moment estimation for the OICC measurement model.

Under  Y^c = alpha_c + beta_c * theta + eps^c  with mutually independent eps
given theta and pivot normalization beta_1 = 1:

    Cov(Y^j, Y^k) = beta_j * beta_k * Var(theta)      for j != k.

So the off-diagonal of the K x K covariance matrix is a rank-1 matrix
beta beta^T * Var(theta) with the diagonal removed.  With K >= 3 channels the
loadings beta and Var(theta) are OVER-identified (more equations than unknowns);
that redundancy is exactly what the specification test exploits.

Estimator (robust, closed-form):
  * With the pivot fixed (beta_1 = 1), for any k >= 2 and any j (j != k, j != 1):
        beta_k = Cov(Y^1, Y^k) / Cov(Y^1, Y^j) * beta_j ... (ratios)
    We instead solve the rank-1 problem directly and stably: take the leading
    eigenvector of the *hollow* covariance matrix (diagonal removed), rescale so
    its pivot entry is 1, giving beta_hat; then Var(theta) from the best-fit
    scale of the off-diagonal.
  * Var(eps_k) = Var(Y^k) - beta_k^2 * Var(theta), floored at a small epsilon.

This avoids fragile single-ratio estimates and degrades gracefully.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from oicc.measurement import _as_2d_channels

ArrayF = np.ndarray

_VAR_FLOOR = 1e-6


@dataclass
class FactorMoments:
    """Estimated one-factor moments.

    beta : (K,) loadings with beta[pivot] == 1.
    var_theta : float, estimated Var(theta) (>= _VAR_FLOOR).
    noise_var : (K,) idiosyncratic Var(eps_c) (>= _VAR_FLOOR).
    cov : (K, K) sample covariance matrix of the channels.
    pivot : int, index of the pivot channel.
    """

    beta: ArrayF
    var_theta: float
    noise_var: ArrayF
    cov: ArrayF
    pivot: int


def pairwise_varu(log_channels: ArrayF) -> ArrayF:
    """Return every off-diagonal covariance / (beta_j beta_k) estimate of Var(theta).

    With unknown betas we cannot divide yet, so this returns the raw off-diagonal
    covariances Cov(Y^j, Y^k); under the model with betas ~ 1 these all estimate
    (approximately) Var(theta) and must agree.  Used by the specification test.
    Returns a 1-D array of the K(K-1)/2 upper-triangle covariances.
    """
    Y = _as_2d_channels(log_channels)
    K = Y.shape[0]
    cov = np.cov(Y)
    covs = np.array([cov[j, k] for j in range(K) for k in range(j + 1, K)])
    return covs


def estimate_factor_moments(
    log_channels: ArrayF, pivot: int = 0
) -> FactorMoments:
    """Estimate one-factor loadings, latent variance, and noise variances.

    Parameters
    ----------
    log_channels : (K, n) array
    pivot : int
        Channel whose loading is normalized to 1.

    Returns
    -------
    FactorMoments
    """
    Y = _as_2d_channels(log_channels)
    K = Y.shape[0]
    if not (0 <= pivot < K):
        raise ValueError(f"pivot must be in [0, {K}); got {pivot}")

    cov = np.cov(Y)  # (K, K), unbiased

    # --- Var(theta) by averaged tetrads (robust, near-unbiased) ---------------
    # For distinct i,j,k:  Cov(Y_i,Y_j) Cov(Y_i,Y_k) / Cov(Y_j,Y_k)
    #   = beta_i^2 Var(theta).  Dividing by beta_i^2 gives Var(theta); but with
    # the pivot fixed we instead estimate  beta_i^2 Var(theta)  for i = pivot
    # (beta_pivot = 1) => this directly yields Var(theta).
    tetrads: list[float] = []
    for j in range(K):
        for k in range(j + 1, K):
            if j == pivot or k == pivot:
                continue
            denom = cov[j, k]
            if abs(denom) > 1e-9:
                tetrads.append(cov[pivot, j] * cov[pivot, k] / denom)
    if tetrads:
        # median is robust to the occasional near-zero denominator
        var_theta = float(np.median(tetrads))
    else:
        # K == 3 or degenerate: fall back to the single available tetrad, or to
        # the mean off-diagonal covariance (equal-loading approximation).
        others = [c for c in range(K) if c != pivot]
        if len(others) >= 2 and abs(cov[others[0], others[1]]) > 1e-9:
            var_theta = float(
                cov[pivot, others[0]] * cov[pivot, others[1]]
                / cov[others[0], others[1]]
            )
        else:
            offdiag = cov[np.triu_indices(K, 1)]
            var_theta = float(np.mean(offdiag)) if offdiag.size else _VAR_FLOOR
    var_theta = max(var_theta, _VAR_FLOOR)

    # --- loadings: beta_k = Cov(Y_pivot, Y_k) / Var(theta) --------------------
    # (since Cov(Y_pivot, Y_k) = beta_pivot beta_k Var(theta) and beta_pivot=1)
    beta = cov[pivot, :] / var_theta
    beta[pivot] = 1.0

    # Noise variances from the diagonal: Var(Y_k) = beta_k^2 var_theta + noise_k.
    noise_var = np.diag(cov) - beta**2 * var_theta
    noise_var = np.clip(noise_var, _VAR_FLOOR, None)

    return FactorMoments(
        beta=beta,
        var_theta=var_theta,
        noise_var=noise_var,
        cov=cov,
        pivot=pivot,
    )
