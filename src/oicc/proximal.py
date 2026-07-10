"""Proximal / negative-control correction of the common-mode confounder (OICC).

THE PROBLEM.  A common-mode confounder W that loads on ALL signal channels along
the factor direction,  Y^c = alpha_c + beta_c*theta + l_c*W + eps^c, is absorbed
into the estimated factor and is INVISIBLE to the over-identification test at any
number of channels (proved: the observable law is exactly one-factor in the
composite F = theta + kappa*W).  This is the irreducible blind spot.

THE ESCAPE (negative controls / proximal causal inference; Miao-Geng-Tchetgen
Tchetgen 2018; Kuroki-Pearl 2014).  Add NEGATIVE-CONTROL channels N^q that carry
ZERO true latent signal (beta=0) but DO load on W:  N^q = a_q + m_q*W + nu^q.
They are noisy readouts of the confounder.  Regressing each signal channel on the
controls estimates and removes its W-component, leaving a DECONFOUNDED channel on
which the ordinary OICC factor model recovers theta (not the composite F).

HONESTY (the assumptions are untestable, by construction):
  * (NC-excl) controls carry NO latent signal (beta_N = 0);
  * (NC-relevance/completeness) controls are W-relevant, rank(Cov(controls)) >=
    dim(W); with Q>=2 independent relevant controls W is point-identified, with
    Q==1 only detection + partial (bounded) correction is available;
  * (proportional loading) the control W-loading is comparable to the signal
    W-loading direction.
None of these can be checked from the channels alone (the same blindness that
hides W hides an invalid control).  What the module BUYS is a principled way to
IMPORT an external identifying assumption that pierces the factor direction; what
it CANNOT buy is data-driven validation of that assumption.  We therefore report
the correction AND a diagnostic of how much variance the controls explain.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from oicc.measurement import _as_2d_channels

ArrayF = np.ndarray

_VAR_FLOOR = 1e-6


@dataclass
class ProximalCorrection:
    """Result of the negative-control common-mode correction.

    deconfounded : (K, n) signal channels with their control-explained
        (common-mode) component removed.
    what_explained : (K,) fraction of each channel's variance explained by the
        controls (a diagnostic: large => strong common-mode contamination).
    n_controls : int, number of negative-control channels used.
    identified : bool, True if Q >= 2 (point-identification regime), else partial.
    """

    deconfounded: ArrayF
    what_explained: ArrayF
    n_controls: int
    identified: bool


def _ols_residualize(y: ArrayF, X: ArrayF) -> tuple[ArrayF, float]:
    """Regress y on [1, X]; return (residual, R^2)."""
    n = y.shape[0]
    A = np.column_stack([np.ones(n), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    fit = A @ coef
    resid = y - fit
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ss_res = float(np.sum(resid ** 2))
    r2 = 0.0 if ss_tot <= _VAR_FLOOR else max(0.0, 1.0 - ss_res / ss_tot)
    return resid, r2


def proximal_deconfound(
    signal_channels: ArrayF,
    controls: ArrayF,
) -> ProximalCorrection:
    """Remove the common-mode (control-explained) component from each channel.

    Parameters
    ----------
    signal_channels : (K, n) primary channels.
    controls : (Q, n) negative-control channels (carry W, not theta).

    Returns
    -------
    ProximalCorrection

    Notes
    -----
    Each signal channel is residualized against the controls: the fitted part is
    its estimated common-mode component, the residual is the deconfounded channel
    (retaining theta + idiosyncratic noise).  The deconfounded channels can be fed
    straight into `estimate_factor_moments` / the conformal layer to recover theta
    rather than the confounded composite F = theta + kappa*W.
    """
    Y = _as_2d_channels(signal_channels)
    N = np.asarray(controls, dtype=float)
    if N.ndim != 2:
        raise ValueError(f"controls must be 2-D (Q, n); got {N.shape}")
    if N.shape[1] != Y.shape[1]:
        raise ValueError(
            f"controls have n={N.shape[1]} but channels have n={Y.shape[1]}"
        )
    if not np.all(np.isfinite(N)):
        raise ValueError("controls contain non-finite values")

    K, n = Y.shape
    Q = N.shape[0]
    Xctrl = N.T  # (n, Q)

    deconf = np.empty_like(Y)
    explained = np.empty(K)
    for c in range(K):
        resid, r2 = _ols_residualize(Y[c], Xctrl)
        # add back the channel mean so the deconfounded channel keeps its level
        deconf[c] = resid + Y[c].mean()
        explained[c] = r2

    return ProximalCorrection(
        deconfounded=deconf,
        what_explained=explained,
        n_controls=Q,
        identified=(Q >= 2),
    )


@dataclass
class PointIDResult:
    """Point-identification of the latent variance under a common-mode confounder.

    var_theta_clean : float, Var(theta) with the common-mode W removed
        (point-identified with Q >= 2 valid controls).
    var_theta_naive : float, the CONFOUNDED estimate Var(theta + kappa W) you get
        by ignoring the controls (over-states the truth per Theorem 3).
    var_W : float, estimated Var(W) of the common-mode confounder.
    beta_clean : (K,) deconfounded signal loadings on theta.
    signal_cm_load : (K,) signal loadings l_c on the confounder W.
    ctrl_load : (Q,) control loadings m_q on W (m_0 normalized to 1).
    identified : bool, True iff Q >= 2.
    """

    var_theta_clean: float
    var_theta_naive: float
    var_W: float
    beta_clean: ArrayF
    signal_cm_load: ArrayF
    ctrl_load: ArrayF
    identified: bool


def point_identify(
    signal_channels: ArrayF,
    controls: ArrayF,
    pivot: int = 0,
) -> PointIDResult:
    """Point-identify Var(theta) free of a common-mode confounder using controls.

    Uses the two-factor anchored covariance identities (controls load on W only):

        Cov(Y^c, N^q) = l_c * m_q * Var(W)            (theta indep W)
        Cov(N^q, N^r) = m_q * m_r * Var(W)   (q != r)
        Cov(Y^c, Y^d) = beta_c beta_d Var(theta) + l_c l_d Var(W)   (c != d)

    With the control-scale normalization m_0 = 1:
        m_q  = mean_c Cov(Y^c, N^q) / Cov(Y^c, N^0)
        VarW = Cov(N^0, N^1) / m_1        (averaged over control pairs)
        l_c  = Cov(Y^c, N^0) / VarW
        Cov_clean(Y^c, Y^d) = Cov(Y^c, Y^d) - l_c l_d VarW
    then the clean one-factor tetrad estimator gives beta and Var(theta).

    Requires Q >= 2 controls for point identification; with Q == 1 it returns the
    partial (attenuated) result and `identified = False`.
    """
    from oicc.moments import estimate_factor_moments  # local import (avoid cycle)

    Y = _as_2d_channels(signal_channels)
    N = np.asarray(controls, dtype=float)
    if N.ndim != 2 or N.shape[1] != Y.shape[1]:
        raise ValueError("controls must be (Q, n) with n matching the channels")
    K, n = Y.shape
    Q = N.shape[0]

    # naive (confounded) latent variance: ignore controls.
    naive_fm = estimate_factor_moments(Y, pivot=pivot)
    var_naive = float(naive_fm.var_theta)

    # cross-covariances Cov(Y^c, N^q) and control covariances.
    stacked = np.vstack([Y, N])
    C = np.cov(stacked)
    Cyn = C[:K, K:]          # (K, Q)
    Cnn = C[K:, K:]          # (Q, Q)

    # DETECTION GATE: only correct if the controls are significantly correlated
    # with the signal channels (a common mode is actually present). Without this,
    # cm=0 data (controls ~ pure noise) would trigger a spurious subtraction.
    # Test max |corr(Y^c, N^q)| against a permutation-style threshold ~ 3/sqrt(n).
    corr_yn = np.array([
        [C[c, K + q] / np.sqrt(max(C[c, c] * C[K + q, K + q], 1e-12))
         for q in range(Q)] for c in range(K)
    ])
    max_abs_corr = float(np.max(np.abs(corr_yn)))
    detect_thresh = 3.0 / np.sqrt(max(n, 1))   # ~3 SE of a null correlation
    common_mode_detected = max_abs_corr > detect_thresh

    if not common_mode_detected:
        # No detectable common mode -> the naive one-factor estimate IS the clean
        # one; do not subtract noise. (Matches Theorem 1 when no confounder.)
        return PointIDResult(
            var_theta_clean=var_naive,
            var_theta_naive=var_naive,
            var_W=0.0,
            beta_clean=naive_fm.beta,
            signal_cm_load=np.zeros(K),
            ctrl_load=np.ones(min(Q, 1)) if Q >= 1 else np.array([]),
            identified=(Q >= 2),
        )

    if Q < 2:
        # partial: cannot separate Var(W); fall back to the residualization
        # correction and report the naive variance as an upper bound.
        pc = proximal_deconfound(Y, N)
        clean_fm = estimate_factor_moments(pc.deconfounded, pivot=pivot)
        return PointIDResult(
            var_theta_clean=float(clean_fm.var_theta),
            var_theta_naive=var_naive,
            var_W=float("nan"),
            beta_clean=clean_fm.beta,
            signal_cm_load=np.full(K, np.nan),
            ctrl_load=np.array([1.0]),
            identified=False,
        )

    # control loadings m_q (m_0 = 1) from cross-cov ratios, averaged over signals.
    ref = Cyn[:, 0]
    safe_ref = np.where(np.abs(ref) > 1e-9, ref, np.nan)
    m = np.ones(Q)
    for q in range(1, Q):
        ratios = Cyn[:, q] / safe_ref
        if np.any(np.isfinite(ratios)):
            m[q] = float(np.nanmedian(ratios))
        else:
            m[q] = 1.0  # degenerate cross-cov: fall back rather than emit NaN
    # never let a NaN loading escape to the caller
    m = np.where(np.isfinite(m), m, 1.0)

    # Var(W) from every control pair, averaged: Cov(N^q,N^r) = m_q m_r VarW.
    w_ests = []
    for q in range(Q):
        for r in range(q + 1, Q):
            denom = m[q] * m[r]
            if abs(denom) > 1e-9:
                w_ests.append(Cnn[q, r] / denom)
    var_W = float(np.median(w_ests)) if w_ests else _VAR_FLOOR
    var_W = max(var_W, _VAR_FLOOR)

    # signal confounder loadings l_c = Cov(Y^c, N^0)/VarW  (m_0 = 1).
    l = Cyn[:, 0] / var_W
    l = np.where(np.isfinite(l), l, 0.0)  # defensive: no NaN to the caller

    # remove the W-contamination from the signal covariance off-diagonals.
    C_yy = C[:K, :K].copy()
    C_clean = C_yy - var_W * np.outer(l, l)
    # keep the (unidentified) diagonal as-is; the tetrad estimator uses only
    # off-diagonals, so overwrite the diagonal with the cleaned-consistent value.
    np.fill_diagonal(C_clean, np.diag(C_yy))

    # clean one-factor tetrad estimate on C_clean (reuse the moments logic).
    # build averaged tetrads for Var(theta):
    tetr = []
    for j in range(K):
        for k in range(j + 1, K):
            if j == pivot or k == pivot:
                continue
            denom = C_clean[j, k]
            if abs(denom) > 1e-9:
                tetr.append(C_clean[pivot, j] * C_clean[pivot, k] / denom)
    if tetr:
        var_clean = float(np.median(tetr))
    else:
        # K == 3: single tetrad
        others = [c for c in range(K) if c != pivot]
        d = C_clean[others[0], others[1]]
        var_clean = float(C_clean[pivot, others[0]] * C_clean[pivot, others[1]]
                          / d) if abs(d) > 1e-9 else var_naive
    var_clean = max(var_clean, _VAR_FLOOR)
    beta_clean = C_clean[pivot, :] / var_clean
    beta_clean[pivot] = 1.0

    return PointIDResult(
        var_theta_clean=var_clean,
        var_theta_naive=var_naive,
        var_W=var_W,
        beta_clean=beta_clean,
        signal_cm_load=l,
        ctrl_load=m,
        identified=True,
    )


def _var_theta_from_blocks(
    Cyy: ArrayF, Cyn: ArrayF, Cnn: ArrayF, pivot: int
) -> float:
    """Naive proximal point-ID of Var(theta) from covariance BLOCKS (no gate).

    Implements the deconfounding identities (m_q via cross-cov ratios, Var(W) via
    control pairs, l_c, clean covariance, triad) directly on supplied blocks.
    Shared by `point_identify` and the exclusion-sensitivity forward simulation.
    """
    K = Cyy.shape[0]
    Q = Cnn.shape[0]
    ref = Cyn[:, 0]
    safe_ref = np.where(np.abs(ref) > 1e-9, ref, np.nan)
    m = np.ones(Q)
    for q in range(1, Q):
        ratios = Cyn[:, q] / safe_ref
        m[q] = float(np.nanmedian(ratios)) if np.any(np.isfinite(ratios)) else 1.0
    m = np.where(np.isfinite(m), m, 1.0)
    w_ests = [Cnn[q, r] / (m[q] * m[r])
              for q in range(Q) for r in range(q + 1, Q)
              if abs(m[q] * m[r]) > 1e-9]
    var_W = max(float(np.median(w_ests)) if w_ests else _VAR_FLOOR, _VAR_FLOOR)
    l = Cyn[:, 0] / var_W
    l = np.where(np.isfinite(l), l, 0.0)
    C_clean = Cyy - var_W * np.outer(l, l)
    np.fill_diagonal(C_clean, np.diag(Cyy))
    tetr = [C_clean[pivot, j] * C_clean[pivot, k] / C_clean[j, k]
            for j in range(K) for k in range(j + 1, K)
            if j != pivot and k != pivot and abs(C_clean[j, k]) > 1e-9]
    if tetr:
        return max(float(np.median(tetr)), _VAR_FLOOR)
    others = [c for c in range(K) if c != pivot]
    d = C_clean[others[0], others[1]]
    if abs(d) > 1e-9:
        return max(float(C_clean[pivot, others[0]] * C_clean[pivot, others[1]] / d),
                   _VAR_FLOOR)
    return _VAR_FLOOR


@dataclass
class ExclusionSensitivity:
    """Sensitivity of the point-identified Var(theta) to an exclusion violation.

    eps_grid : sensitivity levels swept (fraction of a control's systematic
        variance explained by theta; 0 = valid exclusion).
    var_theta_lo, var_theta_hi : (len eps_grid,) lower/upper implied Var(theta)
        over both signs of the violation, at each eps.
    var_theta_ref : Var(theta) under the exclusion (eps=0) -- the headline point.
    robustness_eps : smallest eps at which the band deviates from the reference by
        more than `rel_tol` (a Cinelli--Hazlett-style robustness value); 1.0 if
        the conclusion never flips within the grid.
    rel_tol : the relative-deviation threshold used.
    """

    eps_grid: ArrayF
    var_theta_lo: ArrayF
    var_theta_hi: ArrayF
    var_theta_ref: float
    robustness_eps: float
    rel_tol: float


def exclusion_sensitivity(
    signal_channels: ArrayF,
    controls: ArrayF,
    pivot: int = 0,
    eps_max: float = 0.3,
    n_grid: int = 13,
    rel_tol: float = 0.2,
) -> ExclusionSensitivity:
    """How much could Var(theta) move if the controls secretly carry theta?

    The negative-control exclusion (controls carry the confounder W but NOT the
    latent theta) is UNTESTABLE. This sweeps a bounded violation, parameterized by
    eps = fraction of each control's systematic variance explained by theta
    (Cinelli--Hazlett flavored), and reports the band of implied Var(theta) over
    eps in [0, eps_max] and both violation signs -- a quantitative robustness
    statement for the reviewer's sharpest objection.

    Method (forward simulation from the eps=0 structural fit): fit (beta, l, m,
    Var theta, Var W) assuming exclusion; for each (eps, sign) map to a control
    theta-loading delta_q = sign * m_q * sqrt(VarW/VarTheta * eps/(1-eps)),
    forward-simulate the covariance the analyst would then observe, and re-run the
    naive point-ID to read the Var(theta) it would report.

    HONEST SCOPE: eps_max is an expert prior, NOT estimable from data; the sign of
    the violation is unidentified (hence both are swept). This is a bounded bias
    analysis, not a repair -- it says "the estimate is robust up to eps_max," or,
    if the band is wide, "the estimate is fragile to plausible exclusion
    violations." It does not identify the true Var(theta).
    """
    Y = _as_2d_channels(signal_channels)
    N = np.asarray(controls, dtype=float)
    if N.ndim != 2 or N.shape[1] != Y.shape[1]:
        raise ValueError("controls must be (Q, n) with n matching the channels")
    K, n = Y.shape
    Q = N.shape[0]
    if Q < 2:
        raise ValueError("exclusion_sensitivity needs Q >= 2 controls")
    if not (0.0 <= eps_max < 1.0):
        raise ValueError(f"eps_max must be in [0,1); got {eps_max}")

    # reference structural fit under the exclusion (eps = 0)
    ref = point_identify(Y, N, pivot=pivot)
    var_theta_ref = float(ref.var_theta_clean)
    var_W = max(float(ref.var_W), _VAR_FLOOR)
    beta = np.asarray(ref.beta_clean, dtype=float)
    l = np.asarray(ref.signal_cm_load, dtype=float)
    m = np.asarray(ref.ctrl_load, dtype=float)
    if not np.all(np.isfinite(m)) or m.size != Q:
        m = np.ones(Q)

    ratio = var_W / max(var_theta_ref, _VAR_FLOOR)
    signs = list(_sign_combos(Q))
    eps_grid = np.linspace(0.0, eps_max, n_grid)

    lo = np.empty(n_grid)
    hi = np.empty(n_grid)
    for i, eps in enumerate(eps_grid):
        reported = []
        r_mag = np.sqrt(ratio * eps / (1.0 - eps)) if eps < 1.0 else np.inf
        for sg in signs:
            delta = np.array(sg, dtype=float) * m * r_mag  # (Q,)
            # forward-simulate the observed covariance blocks under this delta
            Cyy = beta[:, None] * beta[None, :] * var_theta_ref \
                + l[:, None] * l[None, :] * var_W
            np.fill_diagonal(Cyy, np.diag(Cyy) + 1.0)
            Cyn = (l[:, None] * m[None, :] * var_W
                   + beta[:, None] * delta[None, :] * var_theta_ref)  # (K,Q)
            Cnn = m[:, None] * m[None, :] * var_W \
                + delta[:, None] * delta[None, :] * var_theta_ref
            np.fill_diagonal(Cnn, np.diag(Cnn) + 1.0)
            reported.append(_var_theta_from_blocks(Cyy, Cyn, Cnn, pivot))
        lo[i] = min(reported)
        hi[i] = max(reported)

    # robustness eps: first grid point whose band deviates from ref by > rel_tol
    rob = 1.0
    for i, eps in enumerate(eps_grid):
        dev = max(abs(lo[i] - var_theta_ref), abs(hi[i] - var_theta_ref))
        if dev > rel_tol * max(abs(var_theta_ref), _VAR_FLOOR):
            rob = float(eps)
            break

    return ExclusionSensitivity(
        eps_grid=eps_grid,
        var_theta_lo=lo,
        var_theta_hi=hi,
        var_theta_ref=var_theta_ref,
        robustness_eps=rob,
        rel_tol=rel_tol,
    )


def _sign_combos(Q: int):
    """Yield all +/-1 sign vectors of length Q (Q small)."""
    from itertools import product
    for combo in product((1.0, -1.0), repeat=Q):
        yield combo
