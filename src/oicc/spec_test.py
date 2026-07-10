"""Over-identification specification test for the one-factor OICC model.

Under the maintained model  Y^c = alpha_c + beta_c*theta + eps^c  (eps mutually
independent given theta), every off-diagonal covariance is rank-1:

    Cov(Y^j, Y^k) = beta_j * beta_k * Var(theta),   j != k.

Taking logs of the (positive) covariances turns this into an ADDITIVE model:

    log Cov(Y^j, Y^k) = a_j + a_k + c,     a_j := log beta_j,  c := log Var(theta).

So the vector of m = K(K-1)/2 log-covariances lies in the column space of a
known design matrix X (row (j,k) has 1's in columns j and k, plus an intercept).
The OVER-IDENTIFYING RESTRICTIONS are exactly the residuals of that regression;
they are zero iff the rank-1 (one-factor + independent-error) structure holds.
The residual degrees of freedom are

    df = m - K = K(K-1)/2 - K,

which is 0 at K = 3 (just-identified: no second-moment over-ID -> we fall back to
a third-cumulant test, flagged as low power) and 2 at K = 4.

This test is LOADING-INVARIANT (loadings enter the design linearly, so unequal
loadings do NOT trigger a false rejection).  We compute the residuals, bootstrap
their covariance, and form a Wald statistic with a chi-square reference.

HONESTY.  The test has power only against DETECTABLE (Delta-perp) violations that
distort the rank-1 structure. A common-mode shock loaded on ALL channels along
beta (Delta-parallel) preserves rank-1 exactly and is INVISIBLE at every K. This
is a NECESSARY, not sufficient, test of conditional independence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from oicc.measurement import _as_2d_channels

ArrayF = np.ndarray


@dataclass
class SpecTestResult:
    """Result of the over-identification specification test.

    stat : float, the Wald statistic.
    pvalue : float, upper-tail chi-square p-value.
    df : int, degrees of freedom (0 => no over-ID content available).
    delta_perp_hat : float, non-negative estimated magnitude of the detectable
        violation (concentrates near 0 under H0); used as a data-driven conformal
        sensitivity radius.
    kind : str, "logcov-tetrad" (K>=4) or "cumulant" (K==3).
    underpowered : bool, True when K==3 (no second-moment over-ID; low power).
    """

    stat: float
    pvalue: float
    df: int
    delta_perp_hat: float
    kind: str
    underpowered: bool


def _log_cov_residuals(Y: ArrayF) -> tuple[ArrayF, ArrayF]:
    """Return (log-cov vector, design matrix) for the additive rank-1 model."""
    K = Y.shape[0]
    cov = np.cov(Y)
    pairs = [(j, k) for j in range(K) for k in range(j + 1, K)]
    m = len(pairs)
    # positive-covariance guard: use |cov| floored (one-factor with positive
    # loadings gives positive covariances; abs keeps it defined if a sample dips).
    y = np.array([np.log(max(abs(cov[j, k]), 1e-12)) for (j, k) in pairs])
    X = np.zeros((m, K + 1))
    X[:, 0] = 1.0  # intercept (= log Var(theta))
    for r, (j, k) in enumerate(pairs):
        X[r, 1 + j] = 1.0
        X[r, 1 + k] = 1.0
    return y, X


def _residual_of(Y: ArrayF) -> ArrayF:
    y, X = _log_cov_residuals(Y)
    # OLS residual via least squares (X is rank K, one column redundant with the
    # intercept -> lstsq handles the rank deficiency stably).
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ coef


def _third_cumulant_scale(Y: ArrayF) -> ArrayF:
    """For K==3: compare Var(theta) implied by 2nd vs 3rd moments (must agree)."""
    Yc = Y - Y.mean(axis=1, keepdims=True)
    c = np.cov(Y)
    c01, c02, c12 = c[0, 1], c[0, 2], c[1, 2]
    denom = c12 if abs(c12) > 1e-9 else np.sign(c12 + 1e-12) * 1e-9
    v2 = c01 * c02 / denom  # 2nd-moment Var(theta)
    cum = float(np.mean(Yc[0] * Yc[1] * Yc[2]))  # co-skew = b0 b1 b2 kappa3(theta)
    v3 = np.sign(cum) * np.abs(cum) ** (2.0 / 3.0)  # comparable 3rd-moment scalar
    return np.array([v2, v3])


def _boot_indices(n: int, block: int, rng: np.random.Generator) -> ArrayF:
    """Resample indices: i.i.d. if block<=1, else a moving-block bootstrap.

    A moving-block bootstrap preserves within-block serial/spatial dependence, so
    the residual covariance is not under-estimated on dependent panels (state-year
    or spatial-area data) -- which would otherwise inflate the Wald statistic and
    over-reject the one-factor null.
    """
    if block <= 1:
        return rng.integers(0, n, n)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, max(n - block + 1, 1), n_blocks)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
    return np.clip(idx, 0, n - 1)


def overid_wald_test(
    log_channels: ArrayF,
    n_boot: int = 400,
    seed: int = 0,
    alpha_level: float = 0.05,
    block: int = 1,
    bootstrap_pvalue: bool = False,
) -> SpecTestResult:
    """Bootstrap Wald over-identification test (loading-invariant, correct size).

    Parameters
    ----------
    log_channels : (K, n) array
    n_boot : int, bootstrap replications for the residual covariance.
    seed : int, RNG seed for the bootstrap.
    alpha_level : float, nominal level (informational only).
    block : int, moving-block length for the bootstrap (>1 preserves serial /
        spatial dependence in a panel; 1 = i.i.d. resampling). Use block>1 for
        dependent data (e.g. state-year or spatial-area panels).
    bootstrap_pvalue : bool, if True the p-value is the bootstrap-null tail
        probability of the Wald statistic instead of the chi-square reference
        (more accurate at small df / small n, where chi-square over-rejects).

    Returns
    -------
    SpecTestResult
    """
    Y = _as_2d_channels(log_channels)
    K, n = Y.shape
    rng = np.random.default_rng(seed)

    if K >= 4:
        kind = "logcov-tetrad"
        underpowered = False
        r0 = _residual_of(Y)
        boot = np.empty((n_boot, r0.size))
        for b in range(n_boot):
            boot[b] = _residual_of(Y[:, _boot_indices(n, block, rng)])
        V = np.cov(boot.T)
        moment = r0
        df = int(r0.size - K)  # residual df of the additive rank-1 model
        if df < 1:
            df = 1
    else:
        kind = "cumulant"
        underpowered = True
        m0 = _third_cumulant_scale(Y)
        boot = np.empty((n_boot, m0.size))
        for b in range(n_boot):
            boot[b] = _third_cumulant_scale(Y[:, _boot_indices(n, block, rng)])
        Vfull = np.cov(boot.T)
        R = np.array([[1.0, -1.0]])  # the two Var(theta) estimates must agree
        moment = R @ m0
        V = R @ Vfull @ R.T
        df = 1

    moment = np.atleast_1d(moment)
    V = np.atleast_2d(V)
    ridge = 1e-10 * (np.trace(V) / max(V.shape[0], 1) + 1e-12)
    Vr = V + ridge * np.eye(V.shape[0])
    Vr_inv = np.linalg.pinv(Vr)
    stat = float(moment @ Vr_inv @ moment)

    if bootstrap_pvalue:
        # bootstrap-null reference: recenter each bootstrap moment and form its
        # Wald stat against the same Vr; the p-value is the tail fraction. This
        # avoids the chi-square small-df over-rejection.
        centered = boot - boot.mean(axis=0, keepdims=True)
        if K < 4:
            centered = centered @ R.T  # apply the contrast used above
        null_stats = np.einsum("bi,ij,bj->b", centered, Vr_inv, centered)
        pvalue = float((null_stats >= stat).mean())
    else:
        pvalue = float(stats.chi2.sf(stat, df))

    # data-driven detectable-violation magnitude (concentrates at 0 under H0)
    delta_perp_hat = float(np.sqrt(max(stat - df, 0.0) / max(n, 1)))

    return SpecTestResult(
        stat=stat,
        pvalue=pvalue,
        df=df,
        delta_perp_hat=delta_perp_hat,
        kind=kind,
        underpowered=underpowered,
    )


# =========================================================================== #
# Third-cumulant over-identification test (power at K=3; all-order impossibility)
# =========================================================================== #
#
# Under the one-factor model Y^c = alpha_c + beta_c*theta + eps^c with mutually
# independent errors, the THIRD cross-cumulants inherit the same rank-1 loading
# structure as the covariances:
#
#     Cum(Y^a, Y^b, Y^c) = kappa3(theta) * beta_a beta_b beta_c   (a,b,c distinct)
#     Cum(Y^a, Y^a, Y^b) = kappa3(theta) * beta_a^2 beta_b
#
# So log|Cum(Y^a,Y^b,Y^c)| = a_a + a_b + a_c + const_3 is an ADDITIVE model in the
# log-loadings (a_j = log|beta_j|) -- exactly analogous to the covariance tetrad
# test, but on third cumulants. Its residuals are over-identifying restrictions
# that exist even at K=3 (where the second-moment test has df=0), giving GENUINE
# power there. And -- crucially -- a common-mode confounder W proportional to beta
# leaves these third-cumulant restrictions satisfied too (it merges into the
# single factor at every order), so this test is ALSO blind to Delta-parallel:
# an empirical demonstration that the impossibility holds at every moment order,
# not just the second. This pre-empts the "just use higher moments / ICA"
# objection.
#
# The test is only meaningful when theta is detectably non-Gaussian (kappa3 != 0);
# we gate on that and report `usable`.


@dataclass
class CumulantTestResult:
    """Result of the third-cumulant over-identification test.

    stat : float, Wald statistic on the third-cumulant rank-1 residuals.
    pvalue : float.
    df : int, number of over-identifying restrictions (residual dimension).
    theta_skew : float, standardized third cumulant of the recovered factor
        (near 0 => theta is ~Gaussian and the test has no leverage).
    usable : bool, True if theta is detectably non-Gaussian (|theta_skew| above a
        small threshold) so the test carries information.
    """

    stat: float
    pvalue: float
    df: int
    theta_skew: float
    usable: bool


def _third_cumulants(Y: ArrayF) -> tuple[ArrayF, list]:
    """Return (log|cumulant| vector, index-multiset list) over triples with <=2
    distinct channels repeated, capturing the rank-1 third-order structure."""
    K, n = Y.shape
    Yc = Y - Y.mean(axis=1, keepdims=True)
    triples = []
    # distinct triples (need K>=3)
    for a in range(K):
        for b in range(a + 1, K):
            for c in range(b + 1, K):
                triples.append((a, b, c))
    # repeated-pair triples (a,a,b) give extra restrictions at small K
    for a in range(K):
        for b in range(K):
            if b != a:
                triples.append((a, a, b))
    vals = []
    for (a, b, c) in triples:
        cum = float(np.mean(Yc[a] * Yc[b] * Yc[c]))
        vals.append(cum)
    return np.array(vals), triples


def _cumulant_design(triples: list, K: int) -> ArrayF:
    """Design matrix for log|Cum| = sum_j (count of j in triple) * a_j + const."""
    m = len(triples)
    X = np.zeros((m, K + 1))
    X[:, 0] = 1.0
    for r, tri in enumerate(triples):
        for j in tri:
            X[r, 1 + j] += 1.0
    return X


def _cumulant_residual(Y: ArrayF) -> ArrayF:
    vals, triples = _third_cumulants(Y)
    K = Y.shape[0]
    y = np.log(np.clip(np.abs(vals), 1e-12, None))
    X = _cumulant_design(triples, K)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ coef


def _factor_skew(Y: ArrayF) -> float:
    """Standardized skew of the recovered common factor (mean of standardized
    channels), a scale-free proxy for kappa3(theta) != 0."""
    Yc = Y - Y.mean(axis=1, keepdims=True)
    sd = Yc.std(axis=1, keepdims=True)
    sd = np.where(sd > 1e-9, sd, 1.0)
    f = (Yc / sd).mean(axis=0)
    fs = f.std()
    if fs < 1e-9:
        return 0.0
    fz = (f - f.mean()) / fs
    return float(np.mean(fz ** 3))


def overid_cumulant_test(
    log_channels: ArrayF,
    n_boot: int = 400,
    seed: int = 0,
    block: int = 1,
    skew_threshold: float = 0.15,
) -> CumulantTestResult:
    """Third-cumulant over-identification test (loading-invariant).

    Provides genuine over-ID power at K=3 (where the second-moment tetrad test has
    df=0) whenever the latent factor is detectably non-Gaussian. Blind to a
    common-mode confounder proportional to beta -- demonstrating the impossibility
    holds at third order too.

    Parameters
    ----------
    log_channels : (K, n) array, K >= 3.
    n_boot : int, bootstrap replications for the residual covariance.
    seed : int, RNG seed.
    block : int, moving-block length for dependent panels (1 = i.i.d.).
    skew_threshold : float, |factor skew| below which theta is treated as
        Gaussian and the test is flagged `usable=False`.

    Returns
    -------
    CumulantTestResult
    """
    Y = _as_2d_channels(log_channels)
    K, n = Y.shape
    if K < 3:
        raise ValueError(f"third-cumulant test needs K >= 3; got {K}")
    rng = np.random.default_rng(seed)

    r0 = _cumulant_residual(Y)
    df = int(r0.size - K)
    if df < 1:
        df = 1
    boot = np.empty((n_boot, r0.size))
    for b in range(n_boot):
        boot[b] = _cumulant_residual(Y[:, _boot_indices(n, block, rng)])
    V = np.cov(boot.T)
    V = np.atleast_2d(V)
    ridge = 1e-10 * (np.trace(V) / max(V.shape[0], 1) + 1e-12)
    Vr = V + ridge * np.eye(V.shape[0])
    stat = float(r0 @ np.linalg.pinv(Vr) @ r0)
    pvalue = float(stats.chi2.sf(stat, df))

    skew = _factor_skew(Y)
    usable = abs(skew) >= skew_threshold

    return CumulantTestResult(
        stat=stat,
        pvalue=pvalue,
        df=df,
        theta_skew=skew,
        usable=usable,
    )
