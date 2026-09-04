"""Q2 evidence: what happens when the recording model is genuinely misspecified?

The correction deflates by an ASSUMED power-law multiplier m_hat. This builds a
world where the TRUE multiplier differs from it by a bounded factor, then measures
(a) the plain corrected interval's latent coverage, which should degrade, and
(b) the Gamma-inflated interval's latent coverage, which should not.

That is the exact objection a referee raises about Theorem 2(i), whose exactness
claim assumes the power-law fixed point holds.
"""

from __future__ import annotations

import numpy as np

from civicsafe.theory import _poisson as poisson
from civicsafe.theory.correction_robustness import robust_latent_interval
from civicsafe.theory.latent_correction import recording_multiplier

ALPHA = 0.10
KAPPA = 0.6
S = 6000
TRIALS = 8


def run() -> None:
    rng = np.random.default_rng(11)
    print(f"true kappa = {KAPPA}, target coverage = {1-ALPHA:.2f}, "
          f"{S} cells x {TRIALS} trials")
    print()
    print(f"{'misspec':>8}{'plain corrected':>17}{'Gamma-inflated':>16}"
          f"{'width ratio':>13}")
    print("-" * 56)

    for factor in (1.0, 1.3, 1.6, 2.0, 3.0):
        plain, infl, wr = [], [], []
        for _ in range(TRIALS):
            lam = rng.gamma(2.0, 2.0, S) + 0.3
            # Recorded rate under the ASSUMED power law...
            mu = lam * (lam / lam.mean()) ** (KAPPA / (1 - KAPPA))
            m_hat = recording_multiplier(mu, KAPPA)
            # ...but the TRUE multiplier is off by a per-cell factor inside
            # [1/factor, factor], which is precisely the sensitivity model's band.
            log_dev = rng.uniform(-np.log(factor), np.log(factor), S) if factor > 1 else np.zeros(S)
            m_true = m_hat * np.exp(log_dev)
            lam_true = np.clip(mu / m_true, 1e-6, None)
            y = poisson.rvs(lam_true, random_state=rng)

            iv1 = robust_latent_interval(mu, KAPPA, gamma=1.0, alpha=ALPHA)
            ivg = robust_latent_interval(mu, KAPPA, gamma=factor, alpha=ALPHA)
            plain.append(np.mean((y >= iv1["lower"]) & (y <= iv1["upper"])))
            infl.append(np.mean((y >= ivg["lower"]) & (y <= ivg["upper"])))
            wr.append(np.mean(ivg["upper"] - ivg["lower"] + 1.0)
                      / np.mean(iv1["upper"] - iv1["lower"] + 1.0))

        flag = "" if np.mean(plain) >= 1 - ALPHA else "  <- below target"
        print(f"{factor:>8.1f}{np.mean(plain):>17.4f}{np.mean(infl):>16.4f}"
              f"{np.mean(wr):>13.2f}{flag}")

    print()
    print("Reading: the plain corrected interval loses latent coverage as the true")
    print("recording model departs from the assumed one; the Gamma-inflated interval")
    print("holds the nominal rate for every model in the band, at a bounded and")
    print("smoothly growing width cost. Gamma = 1 recovers the plain interval.")


if __name__ == "__main__":
    run()
