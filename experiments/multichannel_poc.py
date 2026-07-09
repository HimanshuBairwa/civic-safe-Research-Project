"""
Proof-of-concept for the ceiling-break claim:
Can >=3 noisy channels of a latent crime-rate field (a) recover the latent
better than any single biased channel, and (b) provide an OVER-IDENTIFYING
specification test that DETECTS violations of conditional independence?

This is a controlled, fully-synthetic study where the latent theta is KNOWN,
so every claim is falsifiable against ground truth. No real-data leakage; the
generator is deliberately DIFFERENT in form from the estimator (log-normal
latent + multiplicative reporting), so recovery is not tautological.

Channels (mimicking the real program):
  c1 = police records         (under-reports, bias depends on covariate x)
  c2 = 911 calls-for-service   (different bias direction)
  c3 = victimization survey    (noisy, near-unbiased, small sample)
  [+ optional c4 to test the "how many channels to test the confounder" claim]

Estimator: method-of-moments repeated-measurement deconvolution on log scale
(Kotlarski-style: with >=2 channels sharing a latent, the latent mean/variance
are identified from pairwise covariances; with >=3, the system is OVER-identified
=> a testable restriction).

Run:  python experiments/multichannel_poc.py
"""
from __future__ import annotations

import numpy as np


def generate(n=4000, seed=0, confound=0.0):
    """Generate latent log-rates and 3-4 noisy channels.

    confound = strength of a COMMON shock injected into c1 and c2 only
    (violates conditional independence). confound=0 => assumptions hold.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)                       # area covariate (e.g. income)
    # latent log victimization rate (UNOBSERVED ground truth)
    u = 1.2 + 0.5 * x + rng.normal(0, 0.6, n)      # log theta
    theta = np.exp(u)

    # channel-specific log biases (b_c) and noises (eps_c)
    # records: strong x-dependent under-report; calls: mild opposite; survey: ~unbiased noisy
    common = rng.normal(0, 1.0, n) * confound      # shared confounder (media/policing intensity)

    lc1 = u + (-0.8 - 0.4 * x) + 0.9 * common + rng.normal(0, 0.35, n)   # records
    lc2 = u + (-0.2 + 0.1 * x) + 0.9 * common + rng.normal(0, 0.40, n)   # calls
    lc3 = u + (0.0)              + 0.0 * common + rng.normal(0, 0.55, n)   # survey (indep.)
    lc4 = u + (0.15)            + 0.0 * common + rng.normal(0, 0.60, n)   # aux indep. channel

    return dict(x=x, u=u, theta=theta,
                c=np.array([lc1, lc2, lc3, lc4]))  # log-scale channels


def deconvolve(logchannels, use=(0, 1, 2)):
    """Method-of-moments latent recovery on the log scale.

    Model: lc_k = u + b_k + eps_k, with E[eps_k]=0 (after bias centering unknown),
    Cov(eps_j, eps_k)=0 for j!=k, eps indep of u.
    Then for j!=k:  Cov(lc_j, lc_k) = Var(u)   (bias constants drop out in covariance).
    => Var(u) identified by ANY off-diagonal covariance.
    With >=3 channels we get multiple estimates of Var(u) that must AGREE
    (the over-identifying restriction).

    Latent-level per-unit estimate: we cannot recover per-unit u without a
    reference bias; we anchor bias using the (near-unbiased) survey channel's
    mean, then form a precision-weighted latent estimate.
    """
    L = logchannels[list(use)]
    K = L.shape[0]
    # pairwise covariance estimates of Var(u)
    varu_hats = []
    for j in range(K):
        for k in range(j + 1, K):
            varu_hats.append(np.cov(L[j], L[k])[0, 1])
    varu_hats = np.array(varu_hats)
    varu = varu_hats.mean()

    # noise variance per channel: Var(lc_k) - Var(u)
    noise = np.array([np.var(L[k]) - varu for k in range(K)])
    noise = np.clip(noise, 1e-3, None)

    # precision-weighted combination of (bias-centered) channels as latent estimate.
    # center each channel to its own mean (removes b_k up to the shared E[u]);
    # add back a common offset from the least-biased channel (last in `use`).
    centered = L - L.mean(axis=1, keepdims=True)
    w = (1.0 / noise)
    w = w / w.sum()
    uhat_centered = (w[:, None] * centered).sum(axis=0)
    uhat = uhat_centered + L[-1].mean()   # anchor level to survey-like channel
    return dict(varu=varu, varu_hats=varu_hats, noise=noise, uhat=uhat)


def overid_test(logchannels, use=(0, 1, 2, 3), n_boot=150, seed=1):
    """Over-identification specification test.

    H0: conditional independence holds => all pairwise Cov(lc_j,lc_k) equal Var(u).
    Statistic: spread (max-min) of pairwise covariance estimates, standardized by
    a bootstrap null. Rejection => the channels do NOT share a single latent under
    conditional independence (a confounder or dependent errors exist).
    """
    rng = np.random.default_rng(seed)
    L = logchannels[list(use)]
    K, n = L.shape

    def spread(mat):
        covs = []
        for j in range(K):
            for k in range(j + 1, K):
                covs.append(np.cov(mat[j], mat[k])[0, 1])
        covs = np.array(covs)
        return covs.std(), covs

    stat, covs = spread(L)
    # bootstrap null: resample units, recompute spread under the SAME data
    # (captures sampling variability of the covariance spread)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[b], _ = spread(L[:, idx])
    # p-value: how extreme is 0-spread world? We test whether the observed spread
    # is larger than what pure sampling noise of a TRUE single-factor model gives.
    # Build that null by projecting onto a rank-1 (single latent) model then adding
    # independent noise, and bootstrapping its spread.
    varu = np.median([np.cov(L[j], L[k])[0, 1]
                      for j in range(K) for k in range(j + 1, K)])
    varu = max(varu, 1e-3)
    u_proxy = L.mean(axis=0)
    u_proxy = (u_proxy - u_proxy.mean()) / (u_proxy.std() + 1e-9) * np.sqrt(varu)
    null_spreads = np.empty(n_boot)
    for b in range(n_boot):
        synth = np.empty_like(L)
        for k in range(K):
            nk = max(np.var(L[k]) - varu, 1e-3)
            synth[k] = u_proxy + rng.normal(0, np.sqrt(nk), n)
        idx = rng.integers(0, n, n)
        null_spreads[b], _ = spread(synth[:, idx])
    pval = float((null_spreads >= stat).mean())
    return dict(stat=float(stat), pval=pval, covs=covs, varu=varu)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    print("=" * 70)
    print("MULTI-CHANNEL LATENT RECOVERY + OVER-ID SPECIFICATION TEST — POC")
    print("=" * 70)

    # ---- Part 1: recovery under valid assumptions (confound=0) ----
    print("\n[Part 1] Latent recovery vs single-channel baselines (assumptions HOLD)")
    rec_err, call_err, surv_err, deconv_err = [], [], [], []
    for seed in range(10):
        d = generate(seed=seed, confound=0.0)
        u = d["u"]
        est = deconvolve(d["c"], use=(0, 1, 2))
        # each single channel, bias-centered then re-anchored, as a latent guess
        def chan_guess(k):
            g = d["c"][k] - d["c"][k].mean() + d["c"][2].mean()
            return g
        rec_err.append(rmse(chan_guess(0), u))
        call_err.append(rmse(chan_guess(1), u))
        surv_err.append(rmse(chan_guess(2), u))
        deconv_err.append(rmse(est["uhat"], u))
    print(f"  records-only   RMSE(logT): {np.mean(rec_err):.3f}")
    print(f"  calls-only     RMSE(logT): {np.mean(call_err):.3f}")
    print(f"  survey-only    RMSE(logT): {np.mean(surv_err):.3f}")
    print(f"  3-CH DECONVOLVE RMSE(logT): {np.mean(deconv_err):.3f}  <-- should be lowest")

    # ---- Part 2: does the over-ID test hold its size under H0? ----
    print("\n[Part 2] Over-ID specification test — size under H0 (no confounder)")
    rej = 0
    for seed in range(15):
        d = generate(seed=seed, confound=0.0)
        t = overid_test(d["c"], use=(0, 1, 2, 3), seed=seed)
        rej += (t["pval"] < 0.05)
    print(f"  rejection rate under H0 (should be ~0.05): {rej/15:.3f}")

    # ---- Part 3: does the test DETECT a confounder (power)? ----
    print("\n[Part 3] Over-ID test POWER — c1,c2 share a confounder (H1)")
    for cf in [0.0, 0.3, 0.6, 1.0]:
        rej = 0
        for seed in range(15):
            d = generate(seed=seed, confound=cf)
            t = overid_test(d["c"], use=(0, 1, 2, 3), seed=seed)
            rej += (t["pval"] < 0.05)
        print(f"  confound={cf:.1f}  ->  rejection rate: {rej/15:.3f}")

    print("\n[Interpretation]")
    print("  If Part1 deconvolve < all singles  => multi-channel recovery works.")
    print("  If Part2 ~0.05 and Part3 rises with confound => the over-ID test")
    print("  MAKES conditional-independence TESTABLE (the ceiling-break claim).")
    print("  The honest caveat the test CANNOT catch: a confounder shared by")
    print("  ALL channels equally (moves every covariance together) -> invisible.")


if __name__ == "__main__":
    main()
