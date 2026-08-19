"""Tail-focused proper scoring rules for non-negative count forecasts.

The threshold-weighted CRPS (twCRPS) of Gneiting & Ranjan (2011) applies a
weight function to the CRPS integral.  For a count distribution the integral
has a natural unit-width representation, so the implementation below sums
the CDF loss over the integer support at or above the requested threshold.

The functions accept either a full predictive CDF or ZINB parameters.  They
are deliberately evaluation-only: no model parameters are fitted or changed.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from civicsafe.calibration.zinb_distribution import zinb_cdf_full
from civicsafe.training.metrics import crps_zinb


def _as_float_tensor(value: Tensor | float, *, device: torch.device) -> Tensor:
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def threshold_weighted_crps(
    y: Tensor,
    cdf: Tensor,
    *,
    threshold: float | Tensor,
    grid: Tensor | None = None,
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> Tensor:
    """Compute threshold-weighted CRPS from a predictive CDF.

    Args:
        y: Observations, shape ``(N,)`` or any shape broadcastable to rows of
            ``cdf``.
        cdf: CDF evaluated on a common increasing grid, shape ``(N, K)``.
        threshold: Lower support point receiving non-zero weight.
        grid: Optional support grid.  Integer unit spacing is used by default.
            Non-unit grids are integrated with a right-endpoint rectangle rule.
        reduction: ``"none"`` returns one score per observation.

    Returns:
        Tensor of scores, or a scalar for ``mean``/``sum``.
    """
    cdf = torch.as_tensor(cdf, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32, device=cdf.device)
    if cdf.ndim != 2:
        raise ValueError(f"cdf must have shape (N, K), got {tuple(cdf.shape)}")
    y_flat = torch.as_tensor(y, dtype=torch.float32).reshape(-1)
    if y_flat.numel() != cdf.shape[0]:
        raise ValueError("y and cdf must contain the same number of observations")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be one of 'none', 'mean', or 'sum'")

    cdf = cdf.float().clamp(0.0, 1.0)
    if grid is None:
        support = torch.arange(cdf.shape[1], device=cdf.device, dtype=torch.float32)
        widths = torch.ones_like(support)
    else:
        support = grid.reshape(-1).to(device=cdf.device, dtype=torch.float32)
        if support.numel() != cdf.shape[1]:
            raise ValueError("grid length must equal cdf.shape[1]")
        if support.numel() > 1 and not bool(torch.all(support[1:] > support[:-1])):
            raise ValueError("grid must be strictly increasing")
        widths = torch.ones_like(support)
        if support.numel() > 1:
            widths[:-1] = (support[1:] - support[:-1]).clamp_min(0.0)

    threshold_t = _as_float_tensor(threshold, device=cdf.device)
    if threshold_t.numel() not in (1, y_flat.numel()):
        raise ValueError("threshold must be scalar or have one value per observation")
    threshold_t = threshold_t.reshape(-1)
    if threshold_t.numel() == 1:
        threshold_t = threshold_t.expand(y_flat.numel())

    mask = support.unsqueeze(0) >= threshold_t.unsqueeze(1)
    indicator = (y_flat.unsqueeze(1) <= support.unsqueeze(0)).float()
    scores = (((cdf - indicator) ** 2) * mask * widths.unsqueeze(0)).sum(dim=1)
    if reduction == "none":
        return scores
    return scores.mean() if reduction == "mean" else scores.sum()


def twcrps_zinb(
    y: Tensor,
    pi: Tensor,
    mu: Tensor,
    r: Tensor,
    *,
    threshold: float | Tensor | None = None,
    tail_quantile: float = 0.90,
    k_max: int | None = None,
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> Tensor:
    """Threshold-weighted CRPS for ZINB forecasts.

    If ``threshold`` is omitted it is the empirical ``tail_quantile`` of the
    supplied observations.  The returned score is therefore directly usable
    for the dossier's top-10%-week tail comparison.
    """
    y_flat = torch.as_tensor(y, dtype=torch.float32).reshape(-1)
    if not 0.0 < tail_quantile < 1.0:
        raise ValueError("tail_quantile must lie strictly between 0 and 1")
    pi_flat = torch.as_tensor(pi, dtype=torch.float32).reshape(-1)
    mu_flat = torch.as_tensor(mu, dtype=torch.float32).reshape(-1)
    r_flat = torch.as_tensor(r, dtype=torch.float32).reshape(-1)
    if not (y_flat.numel() == pi_flat.numel() == mu_flat.numel() == r_flat.numel()):
        raise ValueError("y, pi, mu, and r must have identical sizes")

    if threshold is None:
        threshold = torch.quantile(y_flat, tail_quantile)
    support, cdf = zinb_cdf_full(pi_flat, mu_flat, r_flat, k_max=k_max)
    # CRPS is defined on the whole count support.  The distribution helper's
    # saturation grid is usually sufficient, but an observed extreme can lie
    # beyond it; extend the grid so a tail spike is never silently truncated.
    needed = int(torch.ceil(y_flat.max()).item()) if y_flat.numel() else 0
    if needed > int(support[-1].item()):
        support, cdf = zinb_cdf_full(
            pi_flat, mu_flat, r_flat, k_max=max(needed, int(support[-1].item()))
        )
    return threshold_weighted_crps(
        y_flat,
        cdf,
        threshold=threshold,
        grid=support,
        reduction=reduction,
    )


def tail_crps_summary(
    y: Tensor,
    pi: Tensor,
    mu: Tensor,
    r: Tensor,
    *,
    tail_quantile: float = 0.90,
    k_max: int | None = None,
) -> dict[str, float | int]:
    """Return overall and extreme-event CRPS diagnostics for one forecast."""
    y_flat = torch.as_tensor(y, dtype=torch.float32).reshape(-1)
    pi = torch.as_tensor(pi, dtype=torch.float32).reshape(-1)
    mu = torch.as_tensor(mu, dtype=torch.float32).reshape(-1)
    r = torch.as_tensor(r, dtype=torch.float32).reshape(-1)
    threshold = float(torch.quantile(y_flat, tail_quantile).item())
    scores = twcrps_zinb(
        y_flat, pi, mu, r, threshold=threshold, k_max=k_max, reduction="none"
    )
    tail_mask = y_flat >= threshold
    base = crps_zinb(y_flat, pi, mu, r, k_max=k_max)
    return {
        "threshold": threshold,
        "tail_quantile": float(tail_quantile),
        "tail_n": int(tail_mask.sum().item()),
        "tail_fraction": float(tail_mask.float().mean().item()),
        "twcrps": float(scores.mean().item()),
        "twcrps_tail": float(scores[tail_mask].mean().item()) if tail_mask.any() else 0.0,
        "crps": float(base.mean().item()),
    }


def twcrps_deterministic(
    y: Tensor,
    forecast: Tensor,
    *,
    threshold: float | Tensor | None = None,
    tail_quantile: float = 0.90,
    k_max: int | None = None,
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> Tensor:
    """twCRPS for a deterministic point forecast (HA/XGBoost baseline)."""
    y_t = torch.as_tensor(y, dtype=torch.float32).reshape(-1)
    f_t = torch.as_tensor(forecast, dtype=torch.float32).reshape(-1)
    if y_t.numel() != f_t.numel():
        raise ValueError("y and forecast must have identical sizes")
    if threshold is None:
        threshold = torch.quantile(y_t, tail_quantile)
    max_y = int(torch.ceil(torch.maximum(y_t.max(), f_t.max())).item()) if y_t.numel() else 0
    support = torch.arange(0, max(k_max or 0, max_y) + 1, dtype=torch.float32)
    cdf = (support.unsqueeze(0) >= f_t.unsqueeze(1)).float()
    return threshold_weighted_crps(y_t, cdf, threshold=threshold, grid=support, reduction=reduction)


def compare_tail_forecasts(
    y: Tensor,
    forecasts: dict[str, dict[str, Tensor] | Tensor],
    *,
    tail_quantile: float = 0.90,
    k_max: int | None = None,
) -> dict[str, dict[str, float]]:
    """Compute comparable tail scores for ZINB and deterministic forecasts."""
    threshold = torch.quantile(torch.as_tensor(y, dtype=torch.float32).reshape(-1), tail_quantile)
    result: dict[str, dict[str, float]] = {}
    for name, forecast in forecasts.items():
        if isinstance(forecast, dict):
            score = twcrps_zinb(
                y,
                forecast["pi"],
                forecast["mu"],
                forecast["r"],
                threshold=threshold,
                k_max=k_max,
            )
        else:
            score = twcrps_deterministic(y, forecast, threshold=threshold, k_max=k_max)
        result[name] = {"twcrps": float(score.item()), "threshold": float(threshold.item())}
    return result


# Common spelling used in papers and downstream notebooks.
twCRPS = twcrps_zinb  # noqa: N816 - preserve paper/API spelling
threshold_weighted_crps_zinb = twcrps_zinb
compute_twcrps = twcrps_zinb
twcrps = twcrps_zinb


__all__ = [
    "compare_tail_forecasts",
    "compute_twcrps",
    "tail_crps_summary",
    "threshold_weighted_crps",
    "threshold_weighted_crps_zinb",
    "twCRPS",
    "twcrps",
    "twcrps_deterministic",
    "twcrps_zinb",
]
