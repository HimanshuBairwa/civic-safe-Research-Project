"""Feedback-aware safe routing — routing over the *latent* risk, not the record.

The default routing costs (``civicsafe.routing.cost``) consume conformal
intervals computed from the **recorded** crime process. Under observation-biased
feedback, that record is inflated exactly where attention has historically
concentrated — so naive risk-aware routing would faithfully steer civilians
around *over-policed* areas rather than *genuinely high-risk* ones, laundering
enforcement bias into navigation. This is the routing analogue of the
"confidently wrong" phenomenon.

This module closes that gap by routing on a **debiased latent risk field**. The
honest, preferred source of that field is **OICC** (:mod:`oicc`), which recovers
a latent rate from >=3 mechanism-independent channels *without* any unidentified
feedback-gain assumption and ships genuine leave-pivot-out conformal intervals.
Use :func:`oicc_routing_field` to build routing bounds directly from an OICC
``ConformalResult``.

It also retains a **sensitivity-analysis** correction path (``correct_node_risk``
/ ``correct_node_intervals``) parameterized by a feedback gain ``kappa``. IMPORTANT
HONESTY NOTE: ``kappa`` is **NOT point-identified from passive data** (that was an
earlier over-claim, now retracted — see ``docs/AUDIT_2026-07.md``). Treat it as a
*sensitivity knob*: sweep it to see how conclusions move, do not report a single
"identified" value. The OICC path above is the identified one.

Provided here:

* :func:`oicc_routing_field` — turn an OICC latent conformal band into
  ``(risk_upper, interval_width)`` routing weights (the identified path).
* :func:`correct_node_risk` / :func:`correct_node_intervals` — kappa-sensitivity
  deflation of a recorded field (NOT identified; for robustness sweeps only).
* :class:`LatentCVaRCost` — a tail-risk (CVaR) edge cost over the routing
  interval, the robust-routing best practice.
* :class:`ExposureDisparityAudit` — measures whether risk-aware routes
  systematically divert exposure away from (or toward) a demographic group, and
  quantifies how much debiasing reduces that disparity.

Together these make routing the deployed consequence of the measurement work:
*recover honest risk (OICC), route on it, and show it shrinks navigational
redlining.*
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from civicsafe.routing.graph import Edge
from civicsafe.theory.latent_correction import recording_multiplier, should_abstain

__all__ = [
    "oicc_routing_field",
    "correct_node_risk",
    "correct_node_intervals",
    "LatentCVaRCost",
    "ExposureDisparityAudit",
    "ExposureDisparityResult",
]


def oicc_routing_field(
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build routing weights from an OICC latent conformal band (identified path).

    This is the HONEST source of the debiased risk field: ``lower``/``upper`` come
    from :func:`oicc.leave_pivot_out_conformal` (a ``ConformalResult``), which
    recovers the latent rate from mechanism-independent channels with no
    unidentified feedback-gain assumption. The router plans on the conformal
    ``upper`` bound (conservative) with ``interval_width = upper - lower`` as the
    uncertainty signal.

    Args:
        lower: (N,) OICC latent lower bounds (log-rate or rate scale).
        upper: (N,) OICC latent upper bounds, same scale.

    Returns:
        ``{"risk_upper": (N,), "interval_width": (N,)}`` ready to inject into a
        :class:`~civicsafe.routing.graph.RoutingGraph`.
    """
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if lo.shape != hi.shape:
        raise ValueError("lower and upper must have the same shape")
    return {"risk_upper": hi, "interval_width": np.clip(hi - lo, 0.0, None)}


def correct_node_risk(mu: np.ndarray, kappa: float) -> np.ndarray:
    """Deflate recorded node risk ``mu`` to a latent scale (kappa SENSITIVITY).

    NOTE: ``kappa`` is a sensitivity parameter, NOT point-identified from passive
    data. Use this to sweep how routing conclusions move with the assumed feedback
    gain; for the identified field use :func:`oicc_routing_field` instead.

    ``lambda_hat_s = mu_s / (mu_s / mean(mu)) ** kappa``. With ``kappa == 0``
    (no feedback) this is the identity; as ``kappa`` grows, cells recorded far
    above the mean are deflated more, undoing the amplification.

    Args:
        mu: Recorded node risk (e.g. point crime rate), shape ``(N,)``, positive.
        kappa: Assumed feedback gain in ``[0, 1)`` (sensitivity knob).

    Returns:
        Latent-scale node risk, shape ``(N,)``.
    """
    mu = np.asarray(mu, dtype=float)
    return mu / recording_multiplier(mu, kappa)


def correct_node_intervals(
    lower: np.ndarray,
    upper: np.ndarray,
    mu: np.ndarray,
    kappa: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Deflate recorded conformal node intervals to the latent scale.

    The recorded interval ``[lower, upper]`` centred on ``mu`` is divided by the
    per-node recording multiplier ``m_s = (mu_s / mean(mu)) ** kappa``, mapping
    both endpoints to the latent scale on which the router should plan.

    Args:
        lower: Recorded lower bounds, shape ``(N,)``.
        upper: Recorded upper bounds, shape ``(N,)``.
        mu: Recorded point risk used to form the multiplier, shape ``(N,)``.
        kappa: Identified feedback gain in ``[0, 1)``.

    Returns:
        ``(lower_latent, upper_latent)`` deflated bounds.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    m = recording_multiplier(mu, kappa)
    return lower / m, upper / m


@dataclass(frozen=True)
class LatentCVaRCost:
    """Tail-risk (CVaR) edge cost over the feedback-corrected risk interval.

    Rather than planning against a single conformal quantile, CVaR plans against
    the mean of the worst ``(1 - beta)`` tail of the risk interval — the robust
    choice for safety-critical routing. Assuming the corrected node risk is
    (approximately) uniform on its interval ``[lo, hi]`` (the distribution-free
    stance consistent with conformal bounds), the ``beta``-CVaR of the upper tail
    is ``hi - 0.5 * (1 - beta) * (hi - lo)``.

    The edge cost combines distance with the CVaR of its endpoint risks. Nodes
    flagged un-correctable (feedback gain near runaway) contribute an
    ``abstain_penalty`` so the router routes *around* them when it can and the
    abstention monitor triggers when it cannot.

    Attributes:
        w_dist: Weight on physical distance.
        w_risk: Weight on the CVaR tail-risk term.
        beta: CVaR confidence (e.g. 0.9 = average of worst 10% of the interval).
        abstain_penalty: Additive cost for traversing an un-correctable node.
    """

    w_dist: float = 0.3
    w_risk: float = 0.7
    beta: float = 0.9
    abstain_penalty: float = 1e6

    def _cvar(self, lo: float, hi: float) -> float:
        """Upper-tail CVaR of a uniform risk on ``[lo, hi]`` at level ``beta``."""
        tail = 1.0 - self.beta
        return hi - 0.5 * tail * max(hi - lo, 0.0)

    def __call__(self, edge: Edge) -> float:
        """Compute the tail-risk edge cost.

        Uses ``edge.risk_upper`` as ``hi`` and ``hi - edge.interval_width`` as
        ``lo`` (both already latent-corrected if the graph was injected with
        corrected bounds). An edge whose width is non-finite (an abstained node)
        incurs ``abstain_penalty``.
        """
        hi = float(edge.risk_upper)
        width = float(edge.interval_width)
        if not np.isfinite(hi) or not np.isfinite(width):
            return self.w_dist * float(edge.distance) + self.abstain_penalty
        lo = hi - width
        return self.w_dist * float(edge.distance) + self.w_risk * self._cvar(lo, hi)


@dataclass
class ExposureDisparityResult:
    """Result of an exposure-disparity audit.

    Attributes:
        exposure_share: ``{group_id: fraction of routed risk-exposure}``.
        true_share: ``{group_id: fraction of latent incidence}``.
        disparity: ``{group_id: exposure_share / true_share - 1}`` (0 = fair).
        max_abs_disparity: Worst-group absolute disparity (headline number).
    """

    exposure_share: dict[str, float]
    true_share: dict[str, float]
    disparity: dict[str, float]
    max_abs_disparity: float


class ExposureDisparityAudit:
    """Measure whether risk-aware routing over/under-exposes a demographic group.

    Given the risk field the router *plans on* (``routed_risk``) and the true
    latent incidence (``latent_risk``), each group's share of total routed
    exposure is compared to its share of true incidence. A group whose routed
    exposure share falls below its true-incidence share is being systematically
    *routed around* — the navigational analogue of redlining. Running the audit
    on the biased vs. the feedback-corrected risk field quantifies how much the
    correction reduces that disparity.
    """

    def audit(
        self,
        routed_risk: np.ndarray,
        latent_risk: np.ndarray,
        groups: np.ndarray,
    ) -> ExposureDisparityResult:
        """Compute per-group exposure disparity.

        Args:
            routed_risk: Node risk the router plans on (recorded or corrected),
                shape ``(N,)``, non-negative.
            latent_risk: True latent incidence per node, shape ``(N,)``.
            groups: Integer group label per node, shape ``(N,)``.

        Returns:
            An :class:`ExposureDisparityResult`.
        """
        routed = np.clip(np.asarray(routed_risk, dtype=float), 0, None)
        latent = np.clip(np.asarray(latent_risk, dtype=float), 0, None)
        groups = np.asarray(groups)

        tot_routed = routed.sum()
        tot_latent = latent.sum()
        exposure_share: dict[str, float] = {}
        true_share: dict[str, float] = {}
        disparity: dict[str, float] = {}

        for g in np.unique(groups):
            key = str(int(g))
            mask = groups == g
            es = float(routed[mask].sum() / tot_routed) if tot_routed > 0 else float("nan")
            ts = float(latent[mask].sum() / tot_latent) if tot_latent > 0 else float("nan")
            exposure_share[key] = es
            true_share[key] = ts
            disparity[key] = (es / ts - 1.0) if ts > 1e-12 else float("nan")

        valid = [abs(v) for v in disparity.values() if np.isfinite(v)]
        return ExposureDisparityResult(
            exposure_share=exposure_share,
            true_share=true_share,
            disparity=disparity,
            max_abs_disparity=float(np.max(valid)) if valid else float("nan"),
        )

    def correction_reduces_disparity(
        self,
        recorded_risk: np.ndarray,
        latent_risk: np.ndarray,
        groups: np.ndarray,
        kappa: float,
    ) -> dict[str, float]:
        """Show debiasing shrinks exposure disparity (kappa-sensitivity variant).

        Audits the biased (recorded) risk field and a debiased field against the
        same latent truth and reports both headline disparities. Here the
        debiased field is formed by the ``kappa``-sensitivity deflation; for the
        identified analysis, pass an OICC latent field to :meth:`audit` directly
        and compare to the recorded audit (see
        ``experiments/oicc_runs`` feedback-loop demo).

        Args:
            recorded_risk: Observation-biased node risk, shape ``(N,)``.
            latent_risk: True latent incidence per node, shape ``(N,)``.
            groups: Integer group labels, shape ``(N,)``.
            kappa: Assumed feedback gain (sensitivity knob, NOT identified).

        Returns:
            ``{"biased_max_disparity", "corrected_max_disparity",
            "reduction"}``. ``reduction`` is the absolute drop in worst-group
            disparity (positive = debiasing helped).
        """
        biased = self.audit(recorded_risk, latent_risk, groups).max_abs_disparity
        corrected_field = correct_node_risk(np.asarray(recorded_risk, dtype=float), kappa)
        corrected = self.audit(corrected_field, latent_risk, groups).max_abs_disparity
        return {
            "biased_max_disparity": biased,
            "corrected_max_disparity": corrected,
            "reduction": biased - corrected,
        }
