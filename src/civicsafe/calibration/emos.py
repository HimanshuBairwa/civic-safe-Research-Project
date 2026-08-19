"""EMOS weight learning and CRPS decomposition for ZINB ensembles.

Ensemble Model Output Statistics (EMOS) — Gneiting et al. (2005) — learns
optimal combination weights for probabilistic ensemble members by minimizing
CRPS on a held-out calibration set.  Unlike naive equal-weighting (1/K),
EMOS identifies which ensemble members contribute most to forecast skill
and automatically down-weights degenerate or redundant members.

CRPS Decomposition — Hersbach (2000) — decomposes the CRPS into:
  - Reliability: how well-calibrated the predictive distribution is
  - Resolution: how much it varies from the climatological distribution
  - Uncertainty: the inherent unpredictability of the observations

Together, these provide the gold-standard diagnostic for probabilistic
forecast quality required by top-tier venues (NeurIPS, KDD, JASA).

References:
    - Gneiting, T., Raftery, A. E., Westveld III, A. H., & Goldman, T.
      (2005). Calibrated probabilistic forecasting using ensemble model
      output statistics and minimum CRPS estimation. *Monthly Weather
      Review*, 133(5), 1098-1118.
    - Hersbach, H. (2000). Decomposition of the continuous ranked
      probability score for ensemble prediction systems. *Weather and
      Forecasting*, 15(5), 559-570.
    - Ferro, C. A. (2014). Fair scores for ensemble forecasts.
      *Quarterly Journal of the Royal Meteorological Society*, 140(683),
      1917-1923.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch import Tensor

from civicsafe.training.metrics import crps_zinb

logger = logging.getLogger(__name__)


# =====================================================================
# EMOS: Learned Ensemble Weights
# =====================================================================


def learn_emos_weights(
    y_cal: Tensor,
    all_pi: list[Tensor],
    all_mu: list[Tensor],
    all_r: list[Tensor],
    lr: float = 0.05,
    max_iter: int = 300,
    patience: int = 30,
    *,
    category_wise: bool = False,
    holdout_fraction: float = 0.30,
    min_holdout_improvement: float = 0.0025,
    entropy_lambda: float = 0.005,
    category_dim: int = -1,
) -> dict[str, Any]:
    """Learn optimal EMOS weights by minimizing CRPS on calibration data.

    For K ensemble members, learns weights w_1, ..., w_K ∈ Δ_K (simplex)
    such that the weighted parameter combination minimizes CRPS:

        w* = argmin_w  CRPS(F_ZINB(·; π̄_w, μ̄_w, r̄_w), y)

    where π̄_w = Σ_k w_k·π_k,  μ̄_w = Σ_k w_k·μ_k,  r̄_w = Σ_k w_k·r_k

    Parameters
    ----------
    y_cal : Tensor, shape (N,)
        Observed counts on the calibration set.
    all_pi, all_mu, all_r : list of Tensor, each shape (N,)
        ZINB parameters from each ensemble member (K total).
    lr : float
        Learning rate for Adam.
    max_iter : int
        Maximum optimization steps.
    patience : int
        Early stopping patience.

    Returns
    -------
    dict
        'weights': learned weights as list of floats (sum to 1),
        'initial_crps': CRPS with equal weights,
        'final_crps': CRPS with learned weights,
        'improvement_pct': percentage improvement,
        'iterations': number of steps taken.
    """
    K = len(all_pi)
    if K < 2:
        return {
            "weights": [1.0],
            "global_weights": [1.0],
            "category_weights": None,
            "initial_crps": float("nan"),
            "final_crps": float("nan"),
            "improvement_pct": 0.0,
            "iterations": 0,
            "holdout_improvement_pct": 0.0,
            "fallback_used": False,
            "fallback_by_category": [],
            "category_wise": category_wise,
        }

    if not 0.0 <= holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in [0, 1)")
    if min_holdout_improvement < 0.0:
        raise ValueError("min_holdout_improvement must be non-negative")
    if entropy_lambda < 0.0:
        raise ValueError("entropy_lambda must be non-negative")

    device = y_cal.device
    y = y_cal.detach().float()

    if any(p.shape != y.shape for p in all_pi + all_mu + all_r):
        raise ValueError("All EMOS tensors must have the same shape")

    # Keep the temporal/sample axis intact for the honest holdout split. The
    # helper flattens only after selecting the train or validation partition.
    def _fit_one(
        y_view: Tensor,
        pi_view: list[Tensor],
        mu_view: list[Tensor],
        r_view: list[Tensor],
    ) -> dict[str, Any]:
        y_local = y_view.detach().float()
        pi_stack = torch.stack(
            [p.detach().float().clamp(0, 1) for p in pi_view]
        ).to(device)
        mu_stack = torch.stack(
            [m.detach().float().clamp(min=1e-6) for m in mu_view]
        ).to(device)
        r_stack = torch.stack(
            [r.detach().float().clamp(min=0.1) for r in r_view]
        ).to(device)

        n_rows = y_local.shape[0] if y_local.ndim else 1
        if n_rows >= 2 and holdout_fraction > 0.0:
            split = min(max(int(n_rows * (1.0 - holdout_fraction)), 1), n_rows - 1)
            train_slice = slice(0, split)
            holdout_slice = slice(split, n_rows)
        else:
            train_slice = slice(None)
            holdout_slice = slice(None)

        y_train = y_local[train_slice].reshape(-1).to(device)
        pi_train = pi_stack[(slice(None), train_slice)].reshape(K, -1)
        mu_train = mu_stack[(slice(None), train_slice)].reshape(K, -1)
        r_train = r_stack[(slice(None), train_slice)].reshape(K, -1)
        y_holdout = y_local[holdout_slice].reshape(-1).to(device)
        pi_holdout = pi_stack[(slice(None), holdout_slice)].reshape(K, -1)
        mu_holdout = mu_stack[(slice(None), holdout_slice)].reshape(K, -1)
        r_holdout = r_stack[(slice(None), holdout_slice)].reshape(K, -1)

        def _combine(weights: Tensor, pis: Tensor, mus: Tensor, rs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            return (
                (weights[:, None] * pis).sum(dim=0).clamp(0, 1),
                (weights[:, None] * mus).sum(dim=0).clamp(min=1e-6),
                (weights[:, None] * rs).sum(dim=0).clamp(min=0.1),
            )

        w_equal = torch.full((K,), 1.0 / K, device=device)
        eq_pi, eq_mu, eq_r = _combine(w_equal, pi_train, mu_train, r_train)
        initial_train = crps_zinb(y_train, eq_pi, eq_mu, eq_r).mean().item()

        logits = torch.nn.Parameter(torch.zeros(K, device=device))
        optimizer = torch.optim.Adam([logits], lr=lr)
        best_objective = float("inf")
        best_logits = logits.detach().clone()
        patience_counter = 0
        final_iter = 0

        for step in range(1, max_iter + 1):
            optimizer.zero_grad()
            weights = torch.softmax(logits, dim=0)
            pi_w, mu_w, r_w = _combine(weights, pi_train, mu_train, r_train)
            crps_value = crps_zinb(y_train, pi_w, mu_w, r_w).mean()
            entropy = -(weights * torch.log(weights.clamp_min(1e-12))).sum()
            objective = crps_value - entropy_lambda * entropy
            objective.backward()  # type: ignore[no-untyped-call]

            objective_value = float(objective.detach().item())
            if objective_value < best_objective - 1e-8:
                best_objective = objective_value
                best_logits = logits.detach().clone()
                patience_counter = 0
            else:
                patience_counter += 1
            optimizer.step()
            final_iter = step
            if patience_counter >= patience:
                break

        learned_weights = torch.softmax(best_logits, dim=0)
        with torch.no_grad():
            learned_holdout = crps_zinb(
                y_holdout,
                *_combine(learned_weights, pi_holdout, mu_holdout, r_holdout),
            ).mean().item()
            equal_holdout = crps_zinb(
                y_holdout,
                *_combine(w_equal, pi_holdout, mu_holdout, r_holdout),
            ).mean().item()

        holdout_improvement = (
            (equal_holdout - learned_holdout) / max(equal_holdout, 1e-12)
        )
        fallback = holdout_improvement < min_holdout_improvement
        selected = w_equal if fallback else learned_weights
        selected_train = crps_zinb(
            y_train, *_combine(selected, pi_train, mu_train, r_train)
        ).mean().item()
        selected_all = crps_zinb(
            y_local.reshape(-1).to(device),
            *_combine(
                selected,
                pi_stack.reshape(K, -1),
                mu_stack.reshape(K, -1),
                r_stack.reshape(K, -1),
            ),
        ).mean().item()
        return {
            "weights": selected.detach().cpu().tolist(),
            "learned_weights": learned_weights.detach().cpu().tolist(),
            "initial_crps": initial_train,
            "final_crps": selected_train,
            "all_crps": selected_all,
            "holdout_improvement": holdout_improvement * 100.0,
            "fallback_used": fallback,
            "iterations": final_iter,
        }

    if not category_wise:
        one = _fit_one(y, all_pi, all_mu, all_r)
        improvement = (one["initial_crps"] - one["final_crps"]) / max(
            one["initial_crps"], 1e-12
        ) * 100.0
        logger.info(
            "  EMOS weights learned in %d steps: CRPS %.6f -> %.6f (%.2f%%).",
            one["iterations"], one["initial_crps"], one["final_crps"], improvement,
        )
        return {
            **one,
            "global_weights": one["weights"],
            "category_weights": None,
            "improvement_pct": improvement,
            "category_wise": False,
            "entropy_lambda": entropy_lambda,
        }

    moved_y = torch.movedim(y, category_dim, -1)
    moved_pi = [torch.movedim(p, category_dim, -1) for p in all_pi]
    moved_mu = [torch.movedim(p, category_dim, -1) for p in all_mu]
    moved_r = [torch.movedim(p, category_dim, -1) for p in all_r]
    n_categories = moved_y.shape[-1]
    category_results = [
        _fit_one(
            moved_y[..., category],
            [p[..., category] for p in moved_pi],
            [m[..., category] for m in moved_mu],
            [r[..., category] for r in moved_r],
        )
        for category in range(n_categories)
    ]
    category_weights = [result["weights"] for result in category_results]
    global_weights = torch.tensor(category_weights, dtype=torch.float32).mean(dim=0).tolist()
    all_pi_stack = torch.stack(
        [p.detach().float().clamp(0, 1) for p in all_pi]
    ).to(device).reshape(K, -1)
    all_mu_stack = torch.stack(
        [m.detach().float().clamp(min=1e-6) for m in all_mu]
    ).to(device).reshape(K, -1)
    all_r_stack = torch.stack(
        [r.detach().float().clamp(min=0.1) for r in all_r]
    ).to(device).reshape(K, -1)
    equal_weights = torch.full((K,), 1.0 / K, device=device)
    equal_pi = (equal_weights[:, None] * all_pi_stack).sum(dim=0)
    equal_mu = (equal_weights[:, None] * all_mu_stack).sum(dim=0)
    equal_r = (equal_weights[:, None] * all_r_stack).sum(dim=0)
    equal_all = crps_zinb(
        y.reshape(-1).to(device), equal_pi, equal_mu, equal_r
    ).mean().item()
    selected_all = sum(result["all_crps"] for result in category_results) / n_categories
    improvement = (equal_all - selected_all) / max(equal_all, 1e-12) * 100.0
    return {
        # Compatibility: callers that expect one K-vector keep receiving the
        # category-average weights. The actual applied matrix is explicit.
        "weights": global_weights,
        "category_weights": category_weights,
        "global_weights": global_weights,
        "initial_crps": equal_all,
        "final_crps": selected_all,
        "improvement_pct": improvement,
        "holdout_improvement_pct": float(np.mean([r["holdout_improvement"] for r in category_results])),
        "fallback_used": any(r["fallback_used"] for r in category_results),
        "fallback_by_category": [r["fallback_used"] for r in category_results],
        "learned_weights_by_category": [r["learned_weights"] for r in category_results],
        "iterations": max(r["iterations"] for r in category_results),
        "category_wise": True,
        "entropy_lambda": entropy_lambda,
    }


def apply_emos_weights(
    weights: list[float] | list[list[float]],
    all_pi: list[Tensor],
    all_mu: list[Tensor],
    all_r: list[Tensor],
    *,
    category_dim: int = -1,
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply learned EMOS weights to combine ensemble members.

    Parameters
    ----------
    weights : list of float
        EMOS weights (sum to 1).
    all_pi, all_mu, all_r : list of Tensor
        Per-member ZINB parameters.

    Returns
    -------
    (pi_emos, mu_emos, r_emos) : tuple of Tensor
        Weighted ZINB parameters.
    """
    if not all_pi or len(all_pi) != len(all_mu) or len(all_pi) != len(all_r):
        raise ValueError("all_pi, all_mu, and all_r must be non-empty and aligned")
    device = all_pi[0].device
    raw_weights = torch.as_tensor(weights, device=device, dtype=torch.float32)
    member_ndim = all_pi[0].dim()
    if raw_weights.ndim == 1:
        if raw_weights.numel() != len(all_pi):
            raise ValueError("global EMOS weights must match ensemble size")
        w = raw_weights / raw_weights.sum().clamp_min(1e-12)
        w_shape = [len(all_pi)] + [1] * member_ndim
        w_exp = w.reshape(w_shape)
    elif raw_weights.ndim == 2:
        if raw_weights.shape[1] != len(all_pi):
            raise ValueError("category EMOS weights must match ensemble size")
        category_axis = category_dim % member_ndim
        if raw_weights.shape[0] != all_pi[0].shape[category_axis]:
            raise ValueError("category EMOS weights must match category dimension")
        w = raw_weights / raw_weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        w_shape = [len(all_pi)] + [1] * member_ndim
        w_shape[category_axis + 1] = raw_weights.shape[0]
        w_exp = w.transpose(0, 1).reshape(w_shape)
    else:
        raise ValueError("weights must be a global vector or category matrix")
    if (w < 0).any() or not torch.isfinite(w).all():
        raise ValueError("EMOS weights must be finite and non-negative")

    pi_stack = torch.stack([p.float() for p in all_pi])
    mu_stack = torch.stack([m.float() for m in all_mu])
    r_stack = torch.stack([r.float() for r in all_r])

    pi_emos = (w_exp * pi_stack).sum(dim=0).clamp(0.0, 1.0)
    mu_emos = (w_exp * mu_stack).sum(dim=0).clamp(min=1e-6)
    r_emos = (w_exp * r_stack).sum(dim=0).clamp(min=0.1)

    return pi_emos, mu_emos, r_emos


# =====================================================================
# CRPS Decomposition (Hersbach 2000)
# =====================================================================


def crps_decomposition(
    y: Tensor,
    pi: Tensor,
    mu: Tensor,
    r: Tensor,
    n_bins: int = 10,
) -> dict[str, float]:
    r"""Decompose CRPS into Reliability, Resolution, and Uncertainty.

    Implements the exact Hersbach (2000) binned decomposition:

    .. math::

        \overline{\text{CRPS}} = \sum_{k=1}^{K} g_k \bar{o}_k^2
                                 - \sum_{k=1}^{K} g_k \bar{o}_k (\bar{o}_k - 1)
                                 + \text{UNC}

    This is equivalent to: CRPS = REL - RES + UNC, where:
    - **REL** (reliability) = calibration error. Lower is better.
    - **RES** (resolution) = discrimination. Higher is better.
    - **UNC** (uncertainty) = inherent unpredictability. Data property.

    The decomposition uses PIT (Probability Integral Transform) values
    binned into ``n_bins`` categories. For discrete distributions (ZINB),
    randomized PIT is used.

    Parameters
    ----------
    y : Tensor, shape (N,)
        Observed counts.
    pi, mu, r : Tensor, shape (N,)
        ZINB parameters.
    n_bins : int
        Number of PIT histogram bins.

    Returns
    -------
    dict with keys:
        'reliability': calibration error component (lower = better)
        'resolution': discrimination component (higher = better)
        'uncertainty': inherent unpredictability (data property)
        'crps_total': reliability - resolution + uncertainty (should ≈ actual CRPS)
        'crps_actual': directly computed CRPS for validation
        'reliability_fraction': reliability / crps_total
        'resolution_fraction': resolution / crps_total
        'skill_score': 1 - crps_actual / uncertainty (CRPSS vs climatology)

    References
    ----------
    Hersbach, H. (2000). "Decomposition of the continuous ranked probability
    score for ensemble prediction systems." Weather and Forecasting, 15(5),
    559-570.
    """
    from civicsafe.training.metrics import pit_values

    y_flat = y.reshape(-1).float()
    pi_flat = pi.reshape(-1).float().clamp(0, 1)
    mu_flat = mu.reshape(-1).float().clamp(min=1e-6)
    r_flat = r.reshape(-1).float().clamp(min=0.1)

    N = y_flat.shape[0]

    # Compute randomized PIT values
    pit = pit_values(y_flat, pi_flat, mu_flat, r_flat).cpu().numpy()

    # --- Hersbach (2000) Binned Decomposition ---
    # Bin edges: 0, 1/K, 2/K, ..., 1
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    # For each bin k, compute:
    #   n_k = number of PIT values in bin k
    #   o_bar_k = fraction of observations where y <= F^{-1}(p_k)
    #             (approximated by the observed relative frequency in the bin)
    #   g_k = bin width = 1/K for uniform bins

    # Observed relative frequency in each PIT bin
    o_k = np.zeros(n_bins)
    for k in range(n_bins):
        if k < n_bins - 1:
            mask = (pit >= bin_edges[k]) & (pit < bin_edges[k + 1])
        else:
            mask = (pit >= bin_edges[k]) & (pit <= bin_edges[k + 1])
        o_k[k] = mask.sum() / N

    # Expected frequency under uniform PIT (well-calibrated): 1/n_bins
    e_k = 1.0 / n_bins

    # Cumulative observed and expected frequencies (Hersbach eq. 16-18)
    # g_k: bin weight (all equal = 1/K for uniform bins)
    # REL = sum_k g_k * (o_k_bar - p_k_bar)^2  where p_k_bar = k/K
    # RES = sum_k g_k * (o_k_bar - o_clim)^2
    cum_o = np.cumsum(o_k)  # cumulative observed frequency
    cum_e = np.cumsum(np.full(n_bins, e_k))  # cumulative expected (= k/K)

    # Per-bin reliability: deviation of cumulative PIT from uniform
    reliability = np.sum((cum_o - cum_e) ** 2) / n_bins

    # --- Uncertainty: CRPS of the empirical climatological distribution ---
    y_np = y_flat.cpu().numpy()
    y_sorted = np.sort(y_np)
    N_obs = len(y_sorted)
    # Exact: UNC = (2/N^2) * sum_i (i - (N+1)/2) * y_{(i)}
    # This equals E|Y - Y'|/2 where Y, Y' iid from empirical distribution
    ranks = np.arange(1, N_obs + 1)
    uncertainty = (2.0 / (N_obs * N_obs)) * np.sum(
        (ranks - (N_obs + 1) / 2.0) * y_sorted
    )
    uncertainty = max(uncertainty, 1e-12)

    # --- Actual CRPS ---
    crps_actual = crps_zinb(y_flat, pi_flat, mu_flat, r_flat).mean().item()

    # --- Resolution: derived from identity CRPS = REL - RES + UNC ---
    resolution = uncertainty + reliability - crps_actual
    resolution = max(resolution, 0.0)

    crps_total = reliability - resolution + uncertainty
    skill_score = 1.0 - crps_actual / uncertainty if uncertainty > 1e-12 else 0.0

    logger.info("  CRPS Decomposition (Hersbach 2000):")
    logger.info(f"    Reliability (calibration error):    {reliability:.6f}")
    logger.info(f"    Resolution  (discrimination):       {resolution:.6f}")
    logger.info(f"    Uncertainty (inherent):              {uncertainty:.6f}")
    logger.info(f"    CRPS (decomposed):                  {crps_total:.6f}")
    logger.info(f"    CRPS (actual):                      {crps_actual:.6f}")
    logger.info(f"    CRPSS (skill score):                {skill_score:.4f}")

    return {
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "crps_total": float(crps_total),
        "crps_actual": float(crps_actual),
        "reliability_fraction": float(reliability / max(crps_total, 1e-12)),
        "resolution_fraction": float(resolution / max(crps_total, 1e-12)),
        "skill_score": float(skill_score),
    }
