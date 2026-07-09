"""
Proof-of-concept v2 — corrected estimator + correctly-calibrated over-ID test.

v1 finding (honest): a naive precision-weighted combiner does NOT beat the best
single channel when per-channel bias is COVARIATE-DEPENDENT, and an ad-hoc
"spread vs synthetic null" test had broken size (rejected 100% under H0).

v2 fixes:
  (A) Correct latent recovery as a factor-model BLUP for the model
        lc_k = u + b_k + eps_k,  eps_k indep across k, eps indep of u.
      With constant b_k the demeaned optimal estimator has variance
      1/sum_k(1/Var eps_k), provably below the best single channel.
  (B) Correct over-identification test as a WALD test on the K(K-1)/2 pairwise
      covariances (all equal Var(u) under H0), with a bootstrap covariance of
      the covariance-vector -> chi-square reference. This has correct size.
  (C) Two regimes reported honestly:
        - constant per-channel bias  (assumptions HOLD)      -> should win + size ok
        - covariate-dependent bias   (assumption VIOLATED)   -> should LOSE / test fires
      This surfaces the real limitation instead of hiding it.
"""
from __future__ import annotations

import numpy as np


def generate(n=4000, seed=0, confound=0.0, xbias=0.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    u = 1.2 + 0.5 * x + rng.normal(0, 0.6, n)          # latent log-rate (unobserved)
    common = rng.normal(0, 1.0, n) * confound          # confounder shared by c1,c2

    # constant bias baseline; xbias scales the covariate-dependent part
    lc1 = u + (-0.8 - xbias * 0.4 * x) + 0.9 * common + rng.normal(0, 0.35, n)
    lc2 = u + (-0.2 + xbias * 0.1 * x) + 0.9 * common + rng.normal(0, 0.40, n)
    lc3 = u + (0.0)                    + 0.0 * common + rng.normal(0, 0.55, n)
    lc4 = u + (0.15)                   + 0.0 * common + rng.normal(0, 0.60, n)
    return dict(x=x, u=u, c=np.array([lc1, lc2, lc3, lc4]))


def pairwise_covs(L):
    K = L.shape[0]
    idx = [(j, k) for j in range(K) for k in range(j + 1, K)]
    covs = np.array([np.cov(L[j], L[k])[0, 1] for j, k in idx])
    return covs, idx


def deconvolve_blup(L, unbiased_idx=2):
    """Factor-model BLUP of u under lc_k = u + b_k + eps_k.

    Var(u) from median pairwise covariance; Var(eps_k)=Var(lc_k)-Var(u).
    Demeaned optimal combiner weights ∝ 1/Var(eps_k); level anchored on the
    near-unbiased channel (survey).
    """
    covs, _ = pairwise_covs(L)
    varu = max(np.median(covs), 1e-3)
    noise = np.clip(np.var(L, axis=1) - varu, 1e-3, None)
    w = (1.0 / noise)
    w = w / w.sum()
    demeaned = L - L.mean(axis=1, keepdims=True)
    uhat = (w[:, None] * demeaned).sum(axis=0) + L[unbiased_idx].mean()
    return uhat, varu, noise


def overid_wald(L, n_boot=300, seed=1):
    """Wald test that all pairwise covariances are equal (over-ID restriction).

    c = vector of K(K-1)/2 pairwise covs; under H0 all equal a common Var(u).
    Contrast R c = 0 where R differences each cov from the first.
    Stat = (R c)' (R V R')^{-1} (R c),  V = bootstrap cov of c.  ~ chi^2(m-1).
    """
    from numpy.linalg import pinv
    rng = np.random.default_rng(seed)
    c, _ = pairwise_covs(L)
    m = len(c)
    n = L.shape[1]
    boot = np.empty((n_boot, m))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[b], _ = pairwise_covs(L[:, idx])
    V = np.cov(boot.T)
    R = np.zeros((m - 1, m))
    for i in range(m - 1):
        R[i, 0] = 1.0
        R[i, i + 1] = -1.0
    Rc = R @ c
    stat = float(Rc @ pinv(R @ V @ R.T) @ Rc)
    # chi-square survival with df = m-1
    from math import erfc, sqrt
    df = m - 1
    # use a simple gamma-based sf via numpy (avoid scipy dependency)
    # p = P(chi2_df >= stat) approximated by Wilson-Hilferty
    if stat <= 0:
        p = 1.0
    else:
        wh = ((stat / df) ** (1.0 / 3.0) - (1 - 2.0 / (9 * df))) / sqrt(2.0 / (9 * df))
        p = 0.5 * erfc(wh / sqrt(2))
    return dict(stat=stat, pval=float(p), df=df)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    print("=" * 72)
    print("MULTI-CHANNEL LATENT RECOVERY + OVER-ID WALD TEST  (v2, corrected)")
    print("=" * 72)

    for regime, xbias in [("CONSTANT bias (assumptions HOLD)", 0.0),
                          ("COVARIATE-DEPENDENT bias (assumption partially violated)", 1.0)]:
        print(f"\n########## REGIME: {regime} ##########")

        # ---- recovery ----
        errs = {k: [] for k in ["records", "calls", "survey", "deconv"]}
        for seed in range(15):
            d = generate(seed=seed, confound=0.0, xbias=xbias)
            u = d["u"]
            L = d["c"]
            for name, k in [("records", 0), ("calls", 1), ("survey", 2)]:
                g = L[k] - L[k].mean() + L[2].mean()
                errs[name].append(rmse(g, u))
            uhat, _, _ = deconvolve_blup(L)
            errs["deconv"].append(rmse(uhat, u))
        print("  latent-recovery RMSE(log-rate), lower=better:")
        for k in ["records", "calls", "survey", "deconv"]:
            tag = "  <== 3-channel" if k == "deconv" else ""
            print(f"    {k:8s}: {np.mean(errs[k]):.3f}{tag}")
        best_single = min(np.mean(errs['records']), np.mean(errs['calls']), np.mean(errs['survey']))
        verdict = "WINS" if np.mean(errs['deconv']) < best_single else "does NOT win"
        print(f"    -> deconvolution {verdict} vs best single channel")

        # ---- test size (H0, no confounder) ----
        rej = sum(overid_wald(generate(seed=s, confound=0.0, xbias=xbias)["c"], seed=s)["pval"] < 0.05
                  for s in range(15)) / 15
        print(f"  over-ID Wald test rejection @H0 (no confounder), want ~0.05: {rej:.2f}")

        # ---- test power (H1, confounder in c1,c2) ----
        print("  over-ID Wald test power vs confounder strength:")
        for cf in [0.0, 0.3, 0.6, 1.0]:
            rej = sum(overid_wald(generate(seed=s, confound=cf, xbias=xbias)["c"], seed=s)["pval"] < 0.05
                      for s in range(15)) / 15
            print(f"    confound={cf:.1f} -> reject {rej:.2f}")

    print("\n" + "=" * 72)
    print("HONEST READ:")
    print(" - Under constant bias, 3-channel deconvolution should beat every single")
    print("   channel AND the Wald test should hold ~0.05 size and gain power vs a")
    print("   confounder -> the over-ID restriction makes independence TESTABLE.")
    print(" - Under covariate-dependent bias, recovery degrades: a real, reportable")
    print("   limitation (the model needs the covariate, or the bias must be additive).")
    print(" - INVISIBLE to any over-ID test: a confounder shared EQUALLY by ALL")
    print("   channels (shifts every covariance together). This is the residual")
    print("   untestable core; it must be stated as a named assumption, not hidden.")


if __name__ == "__main__":
    main()
