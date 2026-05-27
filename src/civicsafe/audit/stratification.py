"""Dynamic stratification engine for fairness analysis.

Bins continuous socioeconomic features (poverty rate, income, etc.) into
categorical groups for per-group equity evaluation.  Follows best practices
from fairness literature: quantile binning as default (handles skew),
with equal-width and threshold options for sensitivity analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor


@dataclass
class StratConfig:
    """Configuration for a single stratification feature.

    Attributes:
        method: Binning strategy — ``"quantile"``, ``"equal_width"``, or
            ``"threshold"``.
        n_bins: Number of bins (ignored for threshold method).
        threshold: Split point for the threshold method.  If ``None``,
            the median is used automatically.
    """

    method: Literal["quantile", "equal_width", "threshold"]
    n_bins: int = 5
    threshold: float | None = None


class StratificationEngine:
    """Static methods for binning continuous tensors into group labels."""

    @staticmethod
    def quantile_bins(values: Tensor, n_bins: int = 5) -> Tensor:
        """Assign each value to a quantile-based bin (0 … n_bins−1).

        Handles ties by ranking with ``torch.argsort`` so that each bin
        contains approximately ``N / n_bins`` elements.

        Args:
            values: 1-D tensor of continuous values.
            n_bins: Number of quantile bins (e.g. 5 for quintiles).

        Returns:
            Integer tensor of bin labels in ``[0, n_bins)``.
        """
        n = values.shape[0]
        if n == 0:
            return torch.zeros(0, dtype=torch.long, device=values.device)

        # Rank-based binning guarantees equal-sized groups
        order = torch.argsort(values)
        ranks = torch.zeros_like(order)
        ranks[order] = torch.arange(n, device=values.device)

        bins = (ranks.float() * n_bins / n).long().clamp(max=n_bins - 1)
        return bins

    @staticmethod
    def equal_width_bins(values: Tensor, n_bins: int = 5) -> Tensor:
        """Assign each value to an equal-width bin.

        Args:
            values: 1-D tensor of continuous values.
            n_bins: Number of bins.

        Returns:
            Integer tensor of bin labels in ``[0, n_bins)``.
        """
        n = values.shape[0]
        if n == 0:
            return torch.zeros(0, dtype=torch.long, device=values.device)

        vmin = values.min()
        vmax = values.max()
        span = vmax - vmin

        if span == 0:
            return torch.zeros(n, dtype=torch.long, device=values.device)

        normalised = (values - vmin) / span  # [0, 1]
        bins = (normalised * n_bins).long().clamp(max=n_bins - 1)
        return bins

    @staticmethod
    def threshold_bins(values: Tensor, threshold: float | None = None) -> Tensor:
        """Binary split: 0 (below threshold) or 1 (at or above).

        Args:
            values: 1-D tensor of continuous values.
            threshold: Split point.  If ``None``, the median is used.

        Returns:
            Integer tensor of labels in ``{0, 1}``.
        """
        if threshold is None:
            threshold = float(values.median().item())

        return (values >= threshold).long()

    @classmethod
    def auto_stratify(
        cls,
        features: dict[str, Tensor],
        configs: dict[str, StratConfig],
    ) -> dict[str, Tensor]:
        """Apply configured binning to multiple features at once.

        Args:
            features: ``{feature_name: (N,) continuous values}``.
            configs: ``{feature_name: StratConfig}``.  Only features
                present in both dicts are processed.

        Returns:
            ``{feature_name: (N,) integer group labels}``.
        """
        result: dict[str, Tensor] = {}
        for name, cfg in configs.items():
            if name not in features:
                continue
            vals = features[name]
            if cfg.method == "quantile":
                result[name] = cls.quantile_bins(vals, cfg.n_bins)
            elif cfg.method == "equal_width":
                result[name] = cls.equal_width_bins(vals, cfg.n_bins)
            elif cfg.method == "threshold":
                result[name] = cls.threshold_bins(vals, cfg.threshold)
            else:
                msg = f"Unknown stratification method: {cfg.method}"
                raise ValueError(msg)
        return result
