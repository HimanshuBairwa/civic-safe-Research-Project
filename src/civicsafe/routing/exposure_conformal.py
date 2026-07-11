"""Conformal exposure certificates for advisory safe routing.

The genuinely novel routing primitive: a **finite-sample, distribution-free
upper bound on the realized risk-exposure of the route a policy returns.**

Motivation
----------
A risk-aware router plans on a *predicted* risk field, but what matters to a
civilian is the *realized* exposure of the route they are actually sent on. Those
differ because the predicted field is imperfect. Existing robust-routing methods
(min-max regret over interval edge costs) are NP-hard and over-conservative
(Bertsimas--Sim; Kouvelis--Yu); conformal navigation work calibrates obstacle
sets for robots, not exposure over a latent crime field. This module fills that
gap with a clean guarantee.

Method (honest and rigorous)
----------------------------
Fix a routing *policy* pi (a map from a predicted risk field to a route). On n
held-out calibration scenarios, each with a predicted field and the *realized*
node-risk field that materialized, score the realized exposure

    E_i = sum_{s in pi(predicted_i)} realized_i[s].

For a new scenario, split conformal gives, with

    k = ceil((n + 1) * (1 - alpha)),   Q = k-th smallest of {E_i},

the guarantee  P( E_{n+1} <= Q ) >= 1 - alpha,  under exchangeability of the
scenarios. **No monotonicity assumption** is required -- which matters here
because realized exposure is NOT monotone in the router's risk-aversion (planning
on a mis-calibrated field can reroute a civilian through a truly-high-risk node).
That is exactly why split conformal on the scalar exposure functional is the
correct tool, and why a naive conformal-risk-control monotone-threshold argument
would be unsound here. We cite CRC (Angelopoulos et al., ICLR 2024) and conformal
decision theory (Lekeufack et al., ICRA 2024) as the conceptual lineage, but the
guarantee below stands on split conformal alone.

Why it ties to OICC
-------------------
The "realized" field in the crime setting is the *latent* rate that OICC recovers
(records are a biased view of it). Certifying a policy that routes on the
OICC-debiased field versus one that routes on the raw record yields two honest
certificates -- and the debiased policy's realized exposure is the fair one to
report. This makes routing the deployed consequence of the measurement
contribution, not a bolt-on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "RouteExposureCertificate",
    "route_exposure",
    "conformal_upper_quantile",
    "certify_route_exposure",
    "select_risk_budget",
    "RoutePolicy",
    "Scenario",
]

# A routing policy maps a predicted node-risk field (N,) to an ordered node path.
RoutePolicy = Callable[[np.ndarray], Sequence[int]]


@dataclass(frozen=True)
class Scenario:
    """One calibration/test scenario for exposure certification.

    Attributes:
        predicted: (N,) node-risk field the router plans on (e.g. OICC upper
            bound, or a raw-record risk for the biased baseline).
        realized: (N,) node-risk field that actually materialized (the latent
            truth in synthetic studies; a held-out realized proxy in practice).
    """

    predicted: np.ndarray
    realized: np.ndarray


@dataclass(frozen=True)
class RouteExposureCertificate:
    """A finite-sample upper bound on realized route exposure.

    Attributes:
        q_upper: The (1 - alpha) conformal upper bound on realized exposure.
        alpha: Target miscoverage (coverage 1 - alpha).
        n_cal: Number of calibration scenarios used.
        rank: Order statistic k used (k-th smallest calibration exposure).
        finite: False if n is too small for a finite bound at this alpha
            (k > n), in which case q_upper is +inf (the honest answer).
        mean_exposure: Mean calibration exposure (for context, not guaranteed).
    """

    q_upper: float
    alpha: float
    n_cal: int
    rank: int
    finite: bool
    mean_exposure: float


def route_exposure(path: Sequence[int], realized_risk: np.ndarray) -> float:
    """Realized risk-exposure of a route = sum of realized node risks on it.

    Args:
        path: Ordered node indices of the route.
        realized_risk: (N,) realized (true) node-risk field.

    Returns:
        Scalar realized exposure. Empty path -> 0.0.
    """
    r = np.asarray(realized_risk, dtype=float)
    if len(path) == 0:
        return 0.0
    idx = np.asarray(list(path), dtype=int)
    return float(r[idx].sum())


def conformal_upper_quantile(scores: np.ndarray, alpha: float) -> tuple[float, int, bool]:
    """Split-conformal (1 - alpha) upper bound from calibration scores.

    Returns the k-th smallest score with k = ceil((n + 1) * (1 - alpha)). If
    k > n the finite-sample bound is +inf (honest: not enough data to certify
    at this alpha).

    Args:
        scores: (n,) calibration nonconformity scores (here, realized exposures).
        alpha: Miscoverage in (0, 1).

    Returns:
        (q_upper, rank_k, finite).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    s = np.sort(np.asarray(scores, dtype=float))
    n = s.size
    if n == 0:
        return math.inf, 0, False
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return math.inf, k, False
    # k-th smallest is index k-1 (1-indexed order statistic).
    return float(s[k - 1]), k, True


def certify_route_exposure(
    policy: RoutePolicy,
    scenarios: Sequence[Scenario],
    alpha: float = 0.1,
) -> RouteExposureCertificate:
    """Certify a routing policy's realized exposure at level 1 - alpha.

    For each calibration scenario, run the policy on the predicted field, then
    score the route's exposure under the realized field. The conformal upper
    quantile of those scores is a distribution-free (1 - alpha) bound on the
    realized exposure of the policy's route on an exchangeable new scenario.

    Args:
        policy: Maps a predicted (N,) risk field to an ordered node path.
        scenarios: Calibration scenarios (predicted + realized fields).
        alpha: Miscoverage (coverage 1 - alpha).

    Returns:
        A :class:`RouteExposureCertificate`.
    """
    exposures = np.array(
        [route_exposure(policy(sc.predicted), sc.realized) for sc in scenarios],
        dtype=float,
    )
    q, k, finite = conformal_upper_quantile(exposures, alpha)
    return RouteExposureCertificate(
        q_upper=q,
        alpha=alpha,
        n_cal=exposures.size,
        rank=k,
        finite=finite,
        mean_exposure=float(exposures.mean()) if exposures.size else float("nan"),
    )


def select_risk_budget(
    policy_family: Callable[[float], RoutePolicy],
    knobs: Sequence[float],
    select_scenarios: Sequence[Scenario],
    certify_scenarios: Sequence[Scenario],
    target_exposure: float,
    alpha: float = 0.1,
) -> dict:
    """Pick the least-aversion policy whose certified exposure meets a budget.

    Honest, multiplicity-free protocol using a data split:

      1. On the SELECTION fold, certify each candidate policy pi(knob).
      2. Choose the smallest knob whose certified q_upper <= target_exposure
         (least risk-aversion that still meets the safety budget -- so we do not
         over-avoid and inflate distance needlessly). Fall back to the most
         risk-averse knob if none meet it.
      3. Re-certify the chosen policy on the DISJOINT certification fold, so the
         reported guarantee is valid with no selection bias.

    Args:
        policy_family: Maps a risk-aversion knob to a routing policy.
        knobs: Increasing sequence of risk-aversion knobs to consider.
        select_scenarios: Fold used to choose the knob.
        certify_scenarios: Disjoint fold used for the final valid certificate.
        target_exposure: Safety budget the certified exposure must not exceed.
        alpha: Miscoverage.

    Returns:
        Dict with ``chosen_knob``, ``certificate`` (valid on the certify fold),
        ``met_budget`` (bool), and ``selection_curve`` ({knob: q_upper}).
    """
    curve: dict[float, float] = {}
    chosen = None
    for knob in knobs:
        cert = certify_route_exposure(policy_family(knob), select_scenarios, alpha)
        curve[float(knob)] = cert.q_upper
        if chosen is None and cert.finite and cert.q_upper <= target_exposure:
            chosen = knob
    met = chosen is not None
    if chosen is None:
        # none met the budget on the selection fold -> use most risk-averse knob
        chosen = knobs[-1]
    final_cert = certify_route_exposure(
        policy_family(chosen), certify_scenarios, alpha
    )
    return {
        "chosen_knob": float(chosen),
        "certificate": final_cert,
        "met_budget": bool(met and final_cert.finite
                           and final_cert.q_upper <= target_exposure),
        "selection_curve": curve,
    }
