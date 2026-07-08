"""Anytime-valid feedback tripwire — a live monitor for the confidently-wrong regime.

The correction (:mod:`civicsafe.theory.latent_correction`) fixes intervals once
the feedback gain ``kappa`` is known. But a *deployed* system needs to know,
continuously and with a formal guarantee, when it is entering the regime where
its coverage of the latent process is failing — before harm accumulates.

This module provides a **test supermartingale / betting confidence sequence**
(Waudby-Smith & Ramdas 2024; Vovk's conformal test martingales) over the stream
of coverage indicators. Under the null "the intervals are calibrated" the wealth
process is a non-negative supermartingale, so by Ville's inequality
``P(sup_t W_t >= 1/alpha) <= alpha`` for all stopping times simultaneously —
a *time-uniform* false-alarm guarantee, valid under continuous monitoring and
optional stopping, with no multiple-testing correction and nothing to tune.

The monitor bets against the null using a mixture of constant bets (method of
mixtures), so it is parameter-free and adapts its aggressiveness automatically.
When wealth crosses ``1/alpha`` the tripwire fires: the observed miscoverage has
drifted from nominal in a way that is not explainable by chance — the live
signature of the feedback loop crossing toward the runaway threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["FeedbackTripwire", "TripwireState"]


@dataclass
class TripwireState:
    """Snapshot of the tripwire after processing the stream so far.

    Attributes:
        log_wealth: Natural log of the current wealth process value.
        fired: Whether the alarm has fired (wealth crossed ``1/alpha``).
        fired_at: Step index at which it first fired, or ``-1``.
        steps: Number of coverage observations processed.
        running_miscoverage: Empirical miscoverage rate so far.
    """

    log_wealth: float = 0.0
    fired: bool = False
    fired_at: int = -1
    steps: int = 0
    running_miscoverage: float = 0.0
    _miss: int = field(default=0, repr=False)


class FeedbackTripwire:
    """Time-uniform monitor that fires when coverage drifts from nominal.

    The monitor consumes a stream of coverage indicators ``covered_t in {0, 1}``
    for intervals issued at nominal level ``1 - alpha_nominal``. Under the null
    ``E[covered_t] >= 1 - alpha_nominal`` (calibrated or over-covering), the
    wealth built by betting on *miscoverage* is a non-negative supermartingale;
    Ville's inequality then gives a false-alarm probability ``<= alarm_level``
    uniformly over time.

    We mix a grid of constant betting fractions (method of mixtures) so no single
    bet size must be chosen in advance; the mixture wealth is the average wealth
    across the grid and remains a supermartingale under the null.

    Args:
        alpha_nominal: Target miscoverage of the monitored intervals (e.g. 0.10
            for 90% intervals). The null is ``miscoverage <= alpha_nominal``.
        alarm_level: Time-uniform false-alarm probability (e.g. 0.01). The
            tripwire fires when wealth ``>= 1 / alarm_level``.
        bet_grid: Betting fractions in ``(0, 1)`` mixed over. Defaults to a
            log-spaced grid.
    """

    def __init__(
        self,
        alpha_nominal: float = 0.10,
        alarm_level: float = 0.01,
        bet_grid: np.ndarray | None = None,
    ) -> None:
        if not 0.0 < alpha_nominal < 1.0:
            raise ValueError("alpha_nominal must be in (0, 1)")
        if not 0.0 < alarm_level < 1.0:
            raise ValueError("alarm_level must be in (0, 1)")
        self.alpha_nominal = alpha_nominal
        self.alarm_level = alarm_level
        if bet_grid is None:
            bet_grid = np.concatenate([
                np.linspace(0.05, 0.9, 12),
            ])
        self.bet_grid = np.asarray(bet_grid, dtype=float)
        # Per-bet log-wealth (mixture is averaged in wealth space).
        self._log_w = np.zeros_like(self.bet_grid)
        self._log_threshold = float(np.log(1.0 / alarm_level))
        self.state = TripwireState()

    def update(self, covered: int | bool) -> TripwireState:
        """Process one coverage indicator and return the updated state.

        Args:
            covered: ``1``/``True`` if the realized value fell inside the issued
                interval at this step, else ``0``/``False``.

        Returns:
            The updated :class:`TripwireState`.
        """
        miss = 0 if covered else 1
        a = self.alpha_nominal

        # Bet on the event {miss}. Under the null E[miss] <= a. Each constant bet
        # f multiplies wealth by (1 + f * (miss - a) / scale) with a payoff that
        # is a supermartingale under the null: gain when miss occurs more than a.
        # Normalize the per-step multiplier to stay non-negative for f in (0, 1).
        # payoff_t = 1 + f * ((miss - a) / max(a, 1 - a))
        increment = (miss - a) / max(a, 1.0 - a)
        multipliers = 1.0 + self.bet_grid * increment
        multipliers = np.clip(multipliers, 1e-12, None)
        self._log_w += np.log(multipliers)

        # Mixture wealth = mean over bets (log-sum-exp for stability).
        m = self._log_w.max()
        log_mixture = m + np.log(np.mean(np.exp(self._log_w - m)))

        self.state.steps += 1
        self.state._miss += miss
        self.state.running_miscoverage = self.state._miss / self.state.steps
        self.state.log_wealth = float(log_mixture)

        if not self.state.fired and log_mixture >= self._log_threshold:
            self.state.fired = True
            self.state.fired_at = self.state.steps

        return self.state

    def run(self, covered_stream: np.ndarray) -> TripwireState:
        """Process an entire stream of coverage indicators.

        Args:
            covered_stream: Array of ``{0, 1}`` coverage indicators.

        Returns:
            The final :class:`TripwireState`.
        """
        for c in np.asarray(covered_stream).reshape(-1):
            self.update(int(c))
        return self.state
