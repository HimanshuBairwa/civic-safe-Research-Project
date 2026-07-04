"""Anomaly Skill Coefficient (ASC) and Bias Amplification Score (BAS).

This module introduces two **novel** audit metrics designed to quantify the
risk that deploying a predictive model will *reinforce* existing patterns of
over- or under-policing across demographic groups.  Traditional fairness
metrics (equalized odds, calibration) measure static parity at a single
snapshot; they cannot detect whether a model's predictions will **amplify**
historical trends when fed back into resource-allocation decisions.  The ASC
and BAS fill this gap.

Why this matters
----------------
Crime-prediction models are typically trained on *reported* crime data, which
conflates true incidence with enforcement intensity.  If a model's forecasts
correlate with the *direction of change* in a group's crime trend, deploying
that model may reinforce the trajectory — allocating more patrols to areas
already trending upward, starving resources from areas trending downward.
The Anomaly Skill Coefficient makes this dynamic measurable *before* deployment.

Mathematical formulations are given in each class docstring.

Classes
-------
AnomalySkillCoefficient
    Per-group metric capturing trend-amplification risk.
BiasAmplificationScore
    Per-group metric capturing systematic over/under-prediction ratios.

Functions
---------
compute_all_feedback_metrics
    Convenience function returning a JSON-serialisable dict of both metrics
    plus aggregate disparity statistics.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

_MIN_GROUP_SIZE: int = 5
"""Minimum number of samples in a group for reliable statistics."""


# ===================================================================
# Anomaly Skill Coefficient
# ===================================================================


class AnomalySkillCoefficient:
    r"""Quantify trend-amplification risk per demographic group.

    For each demographic group :math:`g` the **Anomaly Skill Coefficient** is

    .. math::

        \mathrm{ASC}_g
        = \rho\!\bigl(\hat{y}_g,\;\Delta_g\bigr)
          \;\cdot\;
          \frac{\operatorname{Var}(\hat{y}_g)}
               {\operatorname{Var}(y_g)}

    where

    * :math:`\hat{y}_g` — model predictions for areas belonging to group
      :math:`g`,
    * :math:`y_g` — observed (ground-truth) counts for group :math:`g`,
    * :math:`\Delta_g = y_g - \bar{h}_g` — deviation of current observations
      from the historical mean :math:`\bar{h}_g` (the *trend signal*),
    * :math:`\rho` — Pearson correlation coefficient,
    * :math:`\operatorname{Var}` — sample variance.

    **Interpretation**

    ======  ====================================================
    Range   Meaning
    ======  ====================================================
    > 0     Predictions *amplify* the historical trend (risk).
    ≈ 0     Predictions are neutral w.r.t. the trend.
    < 0     Predictions *counteract* the trend (corrective).
    ======  ====================================================

    The variance-ratio term scales the correlation by the model's
    *decisiveness*: a model that concentrates its probability mass
    (high :math:`\operatorname{Var}(\hat{y})`) relative to the
    ground truth contributes more to the skill disparity, all else equal.
    """

    def __init__(self) -> None:
        """Initialise AnomalySkillCoefficient (stateless — no fitted parameters)."""

    # ------------------------------------------------------------------
    def compute(
        self,
        y_pred: Tensor,
        y_true: Tensor,
        groups: Tensor,
        historical_trend: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        r"""Compute per-group ASC values.

        Parameters
        ----------
        y_pred : Tensor, shape ``(N,)``
            Model predictions (continuous scores or expected counts).
        y_true : Tensor, shape ``(N,)``
            Ground-truth observed values.
        groups : Tensor, shape ``(N,)``
            Integer group labels (e.g. census-tract demographic codes).
        historical_trend : Tensor, shape ``(N,)``, optional
            Historical mean for each observation's area.  When *None*,
            the within-group mean of ``y_true`` is used as a proxy, which
            yields :math:`\Delta_g = y_{g} - \bar{y}_g`.

        Returns
        -------
        dict
            ``per_group``  — ``{group_id: asc_value}`` mapping.
            ``aggregate``  — ``{"mean_asc", "max_asc", "min_asc"}``.
        """
        y_pred = y_pred.detach().float()
        y_true = y_true.detach().float()
        groups = groups.detach().long()

        unique_groups = torch.unique(groups)
        per_group: Dict[str, float] = {}

        for g in unique_groups:
            g_label = str(g.item())
            mask = groups == g
            n_g = int(mask.sum().item())

            if n_g < _MIN_GROUP_SIZE:
                logger.warning(
                    "Group %s has %d samples (< %d); ASC set to NaN.",
                    g_label, n_g, _MIN_GROUP_SIZE,
                )
                per_group[g_label] = float("nan")
                continue

            yp_g = y_pred[mask]
            yt_g = y_true[mask]

            # Trend signal
            if historical_trend is not None:
                h_g = historical_trend[mask].float()
            else:
                h_g = yt_g.mean().expand_as(yt_g)
            delta_g = yt_g - h_g

            var_yp = torch.var(yp_g, correction=1)
            var_yt = torch.var(yt_g, correction=1)

            if var_yt.item() < 1e-12 or var_yp.item() < 1e-12:
                logger.info(
                    "Group %s has near-zero variance; ASC set to 0.0.",
                    g_label,
                )
                per_group[g_label] = 0.0
                continue

            # Pearson correlation via numpy (robust, handles edge cases)
            corr = float(
                np.corrcoef(
                    yp_g.cpu().numpy(), delta_g.cpu().numpy()
                )[0, 1]
            )
            if np.isnan(corr):
                corr = 0.0

            asc = corr
            per_group[g_label] = float(asc)

        # Aggregate statistics (ignoring NaN entries)
        valid = [v for v in per_group.values() if not np.isnan(v)]
        aggregate: Dict[str, float] = {
            "mean_asc": float(np.mean(valid)) if valid else float("nan"),
            "max_asc": float(np.max(valid)) if valid else float("nan"),
            "min_asc": float(np.min(valid)) if valid else float("nan"),
        }

        return {"per_group": per_group, "aggregate": aggregate}


# ===================================================================
# Bias Amplification Score
# ===================================================================


class BiasAmplificationScore:
    r"""Measure systematic over- or under-prediction per group.

    For group :math:`g` the **Bias Amplification Score** is

    .. math::

        \mathrm{BAS}_g
        = \frac{\mathbb{E}[\hat{y} \mid g]}
               {\mathbb{E}[y \mid g]}
          - 1

    * :math:`\mathrm{BAS}_g = 0` — predictions perfectly mirror the
      observed rate for group :math:`g`.
    * :math:`\mathrm{BAS}_g > 0` — predictions **over-estimate** for
      group :math:`g` (amplification).
    * :math:`\mathrm{BAS}_g < 0` — predictions **under-estimate** for
      group :math:`g` (dampening).

    Unlike calibration error, BAS is a *signed* quantity that reveals the
    direction of bias, enabling auditors to detect whether a model
    systematically inflates risk for historically over-policed communities.
    """

    def __init__(self) -> None:
        """Initialise BiasAmplificationScore (stateless)."""

    def compute(
        self,
        y_pred: Tensor,
        y_true: Tensor,
        groups: Tensor,
    ) -> Dict[str, Any]:
        r"""Compute per-group BAS values.

        Parameters
        ----------
        y_pred : Tensor, shape ``(N,)``
            Model predictions.
        y_true : Tensor, shape ``(N,)``
            Ground-truth observed values.
        groups : Tensor, shape ``(N,)``
            Integer group labels.

        Returns
        -------
        dict
            ``per_group``  — ``{group_id: bas_value}``.
            ``aggregate``  — ``{"max_abs_bas", "mean_abs_bas"}``.
        """
        y_pred = y_pred.detach().float()
        y_true = y_true.detach().float()
        groups = groups.detach().long()

        unique_groups = torch.unique(groups)
        per_group: Dict[str, float] = {}

        for g in unique_groups:
            g_label = str(g.item())
            mask = groups == g
            n_g = int(mask.sum().item())

            if n_g < _MIN_GROUP_SIZE:
                logger.warning(
                    "Group %s has %d samples (< %d); BAS set to NaN.",
                    g_label, n_g, _MIN_GROUP_SIZE,
                )
                per_group[g_label] = float("nan")
                continue

            mean_pred = y_pred[mask].mean().item()
            mean_true = y_true[mask].mean().item()

            if abs(mean_true) < 1e-12:
                logger.info(
                    "Group %s has near-zero true mean; BAS set to NaN.",
                    g_label,
                )
                per_group[g_label] = float("nan")
                continue

            bas = (mean_pred / mean_true) - 1.0
            per_group[g_label] = float(bas)

        valid = [v for v in per_group.values() if not np.isnan(v)]
        aggregate: Dict[str, float] = {
            "max_abs_bas": float(np.max(np.abs(valid))) if valid else float("nan"),
            "mean_abs_bas": float(np.mean(np.abs(valid))) if valid else float("nan"),
        }

        return {"per_group": per_group, "aggregate": aggregate}


# ===================================================================
# Downstream Allocation Disparity
# ===================================================================

class DownstreamAllocationDisparity:
    r"""Measure downstream patrol allocation disparity per group.
    
    Allocations are typically proportional to predicted upper bounds or means.
    This metric compares the share of total resources allocated to a group
    versus the share of total true incidents experienced by that group.
    
    .. math::
    
        \mathrm{DAD}_g = \frac{\text{Allocation Share}_g}{\text{True Share}_g} - 1
        
    * > 0 means the group is over-allocated relative to its true incidents.
    * < 0 means the group is under-allocated relative to its true incidents.
    """
    def __init__(self) -> None:
        """Initialise DownstreamAllocationDisparity (stateless)."""

    def compute(self, y_pred: Tensor, y_true: Tensor, groups: Tensor) -> Dict[str, Any]:
        y_pred = y_pred.detach().float().clamp(min=0)
        y_true = y_true.detach().float().clamp(min=0)
        
        total_alloc = y_pred.sum().item()
        total_true = y_true.sum().item()
        
        unique_groups = torch.unique(groups)
        per_group: Dict[str, float] = {}
        for g in unique_groups:
            g_label = str(g.item())
            mask = groups == g
            group_alloc = y_pred[mask].sum().item()
            group_true = y_true[mask].sum().item()
            
            if total_alloc < 1e-9 or total_true < 1e-9 or group_true < 1e-9:
                per_group[g_label] = float('nan')
                continue
                
            alloc_share = group_alloc / total_alloc
            true_share = group_true / total_true
            
            disparity = (alloc_share / true_share) - 1.0
            per_group[g_label] = float(disparity)
            
        valid = [v for v in per_group.values() if not np.isnan(v)]
        aggregate: Dict[str, float] = {
            "max_alloc_disparity": float(np.max(np.abs(valid))) if valid else float("nan"),
            "mean_alloc_disparity": float(np.mean(np.abs(valid))) if valid else float("nan"),
        }
        return {"per_group": per_group, "aggregate": aggregate}

# ===================================================================
# Convenience wrapper
# ===================================================================


def compute_all_feedback_metrics(
    y_pred: Tensor,
    y_true: Tensor,
    groups: Tensor,
    counts_historical: Optional[Tensor] = None,
) -> Dict[str, Any]:
    """Compute ASC and BAS in one call and return a JSON-ready summary.

    Parameters
    ----------
    y_pred : Tensor, shape ``(N,)``
        Model predictions.
    y_true : Tensor, shape ``(N,)``
        Ground-truth observed values.
    groups : Tensor, shape ``(N,)``
        Integer group labels.
    counts_historical : Tensor, shape ``(N,)``, optional
        Historical baseline counts per observation.  Forwarded to
        :pyclass:`AnomalySkillCoefficient` as ``historical_trend``.

    Returns
    -------
    dict
        Top-level keys: ``"asc"``, ``"bas"``, ``"disparity"``.
        All values are plain Python scalars (JSON-serialisable).
    """
    asc_result = AnomalySkillCoefficient().compute(
        y_pred, y_true, groups, historical_trend=counts_historical,
    )
    bas_result = BiasAmplificationScore().compute(y_pred, y_true, groups)
    dad_result = DownstreamAllocationDisparity().compute(y_pred, y_true, groups)

    # Cross-metric disparity: max |ASC| spread and max |BAS| spread
    asc_vals = [
        v for v in asc_result["per_group"].values() if not np.isnan(v)
    ]
    bas_vals = [
        v for v in bas_result["per_group"].values() if not np.isnan(v)
    ]

    disparity: Dict[str, float] = {
        "asc_range": (max(asc_vals) - min(asc_vals)) if len(asc_vals) >= 2 else 0.0,
        "bas_range": (max(bas_vals) - min(bas_vals)) if len(bas_vals) >= 2 else 0.0,
        "max_abs_asc": float(np.max(np.abs(asc_vals))) if asc_vals else float("nan"),
        "max_abs_bas": float(np.max(np.abs(bas_vals))) if bas_vals else float("nan"),
    }

    return {
        "asc": asc_result,
        "bas": bas_result,
        "dad": dad_result,
        "disparity": disparity,
    }
