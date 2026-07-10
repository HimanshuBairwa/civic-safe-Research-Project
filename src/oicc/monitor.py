"""Anytime-valid deployment monitor for OICC (e-process / testing by betting).

In deployment the measurement structure can DRIFT: channels that were
conditionally independent given the latent start sharing new dependence (a
detectable, Delta-perp change). We monitor a stream of over-identification
p-values {p_1, p_2, ...} (one per time window from `overid_wald_test`) and raise
an alarm the moment cumulative evidence against the one-factor structure is
strong -- with a TIME-UNIFORM false-alarm guarantee that holds at every stopping
time (no multiple-testing correction needed, unlike repeated fixed-level tests).

Construction (Vovk-Wang p-to-e calibrators + Ville's inequality).
Under H0 (structure holds) each p_t is (super-)uniform on [0,1]. For kappa in
(0,1) the calibrator

    f_kappa(p) = kappa * p^(kappa - 1),      E_{p~U[0,1]}[f_kappa(p)] = 1,

turns a p-value into an e-value (expected value <= 1 under H0). The running
product

    M_t = prod_{s<=t} bar_f(p_s),     bar_f = average of f_kappa over a grid,

is a non-negative test supermartingale with M_0 = 1. By Ville's inequality,

    P( exists t : M_t >= 1/alpha  |  H0 )  <=  alpha,

so declaring drift when M_t >= 1/alpha controls the ANYTIME false-alarm rate at
alpha. When the structure breaks, p-values concentrate near 0, f_kappa(p) blows
up, M_t grows and the alarm fires (with a detection delay that shrinks as the
violation strengthens).

This monitors DETECTABLE (Delta-perp) drift only -- consistent with the rest of
OICC, a common-mode (Delta-parallel) drift stays invisible (Theorem 3).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ArrayF = np.ndarray


def _calibrate_p_to_e(p: float, kappas: ArrayF) -> float:
    """Average Vovk-Wang calibrator: mean over kappa of kappa * p^(kappa-1)."""
    p = float(min(max(p, 1e-12), 1.0))  # guard log/pow at 0
    vals = kappas * p ** (kappas - 1.0)
    return float(np.mean(vals))


@dataclass
class EProcessMonitor:
    """Anytime-valid drift monitor via a product e-process.

    Parameters
    ----------
    alpha : float, target anytime false-alarm level (alarm threshold 1/alpha).
    kappas : array of calibrator exponents in (0,1); a mixture for robustness.

    State
    -----
    wealth : float, current M_t (starts at 1).
    log_wealth : float, log M_t (numerically stable).
    history : list of (p, e, wealth) per update.
    alarm : bool, whether M_t has ever crossed 1/alpha.
    alarm_time : int or None, the 1-indexed step at which the alarm first fired.
    """

    alpha: float = 0.05
    kappas: ArrayF = field(
        default_factory=lambda: np.array([0.2, 0.35, 0.5, 0.65, 0.8])
    )
    log_wealth: float = 0.0
    step: int = 0
    alarm: bool = False
    alarm_time: int | None = None
    history: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must be in (0,1); got {self.alpha}")
        self.kappas = np.asarray(self.kappas, dtype=float)
        if np.any((self.kappas <= 0.0) | (self.kappas >= 1.0)):
            raise ValueError("kappas must lie strictly in (0,1)")

    @property
    def wealth(self) -> float:
        # cap to avoid float overflow once the alarm has fired and wealth explodes;
        # the exact magnitude past the threshold is irrelevant to the decision.
        return float(np.exp(min(self.log_wealth, 700.0)))

    @property
    def threshold(self) -> float:
        return 1.0 / self.alpha

    def update(self, pvalue: float) -> float:
        """Feed one over-ID p-value; return the current wealth M_t."""
        if not (0.0 <= pvalue <= 1.0):
            raise ValueError(f"pvalue must be in [0,1]; got {pvalue}")
        e = _calibrate_p_to_e(pvalue, self.kappas)
        self.log_wealth += np.log(max(e, 1e-300))
        self.step += 1
        w = self.wealth
        if (not self.alarm) and self.log_wealth >= np.log(self.threshold):
            self.alarm = True
            self.alarm_time = self.step
        self.history.append((float(pvalue), float(e), w))
        return w

    def run(self, pvalues: ArrayF) -> "EProcessMonitor":
        """Feed a whole stream of p-values in order."""
        for p in np.asarray(pvalues, dtype=float).ravel():
            self.update(float(p))
        return self
