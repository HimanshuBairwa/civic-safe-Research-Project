"""Immutable data container for audit components.

``AuditBundle`` is the 'currency' passed between audit components —
analogous to AIF360's Dataset but tailored for CIVIC-SAFE's
regression + prediction-interval setting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class AuditBundle:
    """Immutable container for all data needed by audit components.

    Attributes:
        y_true: Ground-truth crime counts.  Shape ``(N,)``.
        y_pred: Point predictions ``E[Y] = (1-π)·μ``.  Shape ``(N,)``.
        lower: Prediction-interval lower bounds.  Shape ``(N,)``.
        upper: Prediction-interval upper bounds.  Shape ``(N,)``.
        pi: ZINB zero-inflation parameter.  Shape ``(N,)``.
        mu: ZINB mean parameter.  Shape ``(N,)``.
        r: ZINB dispersion parameter.  Shape ``(N,)``.
        strata: Stratification features ``{name: (N,) group-label tensor}``.
        spatial_units: Spatial-unit IDs.  Shape ``(N,)``.
        alpha: Nominal mis-coverage level (e.g. 0.1 for 90 % coverage).
        metadata: Free-form metadata (city, period, model version …).
    """

    y_true: Tensor
    y_pred: Tensor
    lower: Tensor
    upper: Tensor
    pi: Tensor
    mu: Tensor
    r: Tensor
    strata: dict[str, Tensor]
    spatial_units: Tensor
    alpha: float
    metadata: dict[str, Any]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_samples(self) -> int:
        """Number of data points in this bundle."""
        return int(self.y_true.shape[0])

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Assert that all tensors share the same leading dimension.

        Raises:
            ValueError: If any tensor has a mismatched first dimension.
        """
        n = self.num_samples
        fields = {
            "y_pred": self.y_pred,
            "lower": self.lower,
            "upper": self.upper,
            "pi": self.pi,
            "mu": self.mu,
            "r": self.r,
            "spatial_units": self.spatial_units,
        }
        for name, tensor in fields.items():
            if tensor.shape[0] != n:
                msg = (
                    f"{name} has shape {tensor.shape} but y_true has "
                    f"shape {self.y_true.shape}"
                )
                raise ValueError(msg)

        for feat_name, feat_tensor in self.strata.items():
            if feat_tensor.shape[0] != n:
                msg = (
                    f"strata['{feat_name}'] has shape {feat_tensor.shape} "
                    f"but y_true has shape {self.y_true.shape}"
                )
                raise ValueError(msg)

    # ------------------------------------------------------------------
    # Reporting-bias sensitivity helper
    # ------------------------------------------------------------------

    def with_thinned_targets(
        self,
        p_report: float,
        seed: int = 42,
    ) -> "AuditBundle":
        """Return a new bundle with binomial-thinned ``y_true``.

        For each count ``y_true[i]``, the thinned count is drawn as
        ``Y_obs[i] ~ Binomial(y_true[i], p_report)``.

        Args:
            p_report: Reporting probability in ``(0, 1]``.
            seed: RNG seed for reproducibility.

        Returns:
            A new ``AuditBundle`` with thinned ``y_true`` (other fields
            are shared, not copied).
        """
        if p_report >= 1.0:
            return self

        gen = torch.Generator(device=self.y_true.device).manual_seed(seed)
        y_long = self.y_true.long()
        max_count = int(y_long.max().item())

        if max_count == 0:
            return self

        probs = torch.full(
            (y_long.shape[0], max_count),
            p_report,
            device=self.y_true.device,
        )
        bernoulli_draws = torch.bernoulli(probs, generator=gen)
        indices = torch.arange(max_count, device=self.y_true.device).unsqueeze(0)
        valid_mask = indices < y_long.unsqueeze(-1)
        thinned = (bernoulli_draws * valid_mask.float()).sum(dim=-1).long()

        # frozen dataclass — use object.__setattr__ workaround via constructor
        return AuditBundle(
            y_true=thinned.float(),
            y_pred=self.y_pred,
            lower=self.lower,
            upper=self.upper,
            pi=self.pi,
            mu=self.mu,
            r=self.r,
            strata=self.strata,
            spatial_units=self.spatial_units,
            alpha=self.alpha,
            metadata={**self.metadata, "p_report": p_report, "thinning_seed": seed},
        )
