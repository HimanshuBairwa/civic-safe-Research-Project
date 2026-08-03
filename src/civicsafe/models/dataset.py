"""Sliding-window dataset for spatiotemporal crime forecasting.

Implements strict chronological splitting to prevent data leakage:
  Train: 2018–2021 (208 weeks)
  Val:   2022      (52 weeks)
  Test:  2023      (52 weeks)

The dataset slides a window of length W over the time axis, producing
(input_window, target) pairs where the model sees W weeks of history
and predicts the next week's counts.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class CrimeWindowDataset(Dataset):  # type: ignore[type-arg]
    """Rolling-window dataset for autoregressive crime forecasting.

    Args:
        counts: Crime count tensor. Shape: (S, T, C)
        features: Covariate tensor. Shape: (S, T, F)
        window_size: Number of history weeks per sample.
        start_idx: First valid target index (inclusive).
        end_idx: Last valid target index (exclusive).
    """

    def __init__(
        self,
        counts: Tensor,
        features: Tensor,
        groups: Tensor | None = None,
        window_size: int = 52,
        start_idx: int | None = None,
        end_idx: int | None = None,
        exclude_crime_history: bool = False,
    ) -> None:
        super().__init__()
        _S, T, _C = counts.shape
        self.counts = counts
        self.features = features
        self.groups = groups
        self.window_size = window_size
        self.exclude_crime_history = exclude_crime_history

        # Default: use all valid windows
        if start_idx is None:
            start_idx = window_size
        if end_idx is None:
            end_idx = T

        self.start_idx = max(start_idx, window_size)
        self.end_idx = min(end_idx, T)

        self.valid_targets = list(range(self.start_idx, self.end_idx))

        logger.info(
            f"  Dataset: {len(self.valid_targets)} windows, "
            f"window_size={window_size}, "
            f"target range [{self.start_idx}, {self.end_idx})"
        )

    def __len__(self) -> int:
        return len(self.valid_targets)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Get a single (input, target) pair.

        Returns:
            Dictionary with:
              input_counts:   (S, W, C) — history window of counts
              input_features: (S, W, F) — history window of features
              target_counts:  (S, C)    — next-step ground truth counts
        """
        t = self.valid_targets[idx]
        w = self.window_size

        # Counts arrive as int64 (panel.py builds them with dtype=torch.long,
        # correctly -- crime counts are integers). Emit float32 instead, because
        # every consumer is doing float arithmetic on them and torch will not
        # infer a float output dtype from an integer input:
        #
        #   input_counts.mean(dim=1)
        #   RuntimeError: mean(): could not infer output dtype.
        #                 Input dtype must be either a floating point ... Got: Long
        #
        # The main model never hit this because it reaches the counts through
        # log1p, which promotes to float32 on its own. Baselines that average or
        # regress on the raw window hit it immediately. Casting here fixes every
        # consumer at once rather than sprinkling .float() at each call site, and
        # it is provably neutral for the main path: log1p(long) and
        # log1p(float32) are bit-identical (verified over random counts to 800).
        # float32 also halves the memory of these windows versus int64.
        input_counts = self.counts[:, t - w : t, :].float()
        if self.exclude_crime_history:
            input_counts = torch.zeros_like(input_counts)

        res = {
            "input_counts": input_counts,  # (S, W, C) float32
            "input_features": self.features[:, t - w : t, :],  # (S, W, F)
            "target_counts": self.counts[:, t, :].float(),  # (S, C) float32
        }
        if self.groups is not None:
            res["groups"] = self.groups  # (S,)
            
        return res


def create_chronological_splits(
    counts: Tensor,
    features: Tensor,
    groups: Tensor | None = None,
    start_year: int = 2018,
    end_year: int = 2023,
    val_year: int = 2022,
    test_year: int = 2023,
    weeks_per_year: int = 52,
    window_size: int = 52,
    exclude_crime_history: bool = False,
) -> dict[str, CrimeWindowDataset]:
    """Create train/val/cal/test splits with strict chronological separation.

    Ensures NO temporal leakage: train data never overlaps with val/cal/test.
    The validation year is split in half: first 26 weeks for validation (early
    stopping), second 26 weeks for conformal calibration.

    Args:
        counts: (S, T, C) crime counts tensor.
        features: (S, T, F) covariate tensor.
        groups: (S,) optional demographic groups tensor.
        start_year: First year in the dataset.
        end_year: Last year in the dataset.
        val_year: Validation year.
        test_year: Test year.
        weeks_per_year: Weeks per year (52).
        window_size: History window size.
        exclude_crime_history: If True, zeroes out crime history inputs.

    Returns:
        Dictionary with 'train', 'val', 'cal', 'test' CrimeWindowDataset instances.
    """
    _total_years = end_year - start_year + 1
    total_weeks = counts.shape[1]

    # Calculate week indices for each split
    val_start_week = (val_year - start_year) * weeks_per_year
    cal_start_week = val_start_week + (weeks_per_year // 2)  # Mid-year split
    test_start_week = (test_year - start_year) * weeks_per_year

    logger.info("  Chronological split:")
    logger.info(f"    Train: weeks [0, {val_start_week}) = {start_year}–{val_year - 1}")
    logger.info(f"    Val:   weeks [{val_start_week}, {cal_start_week}) = {val_year} H1")
    logger.info(f"    Cal:   weeks [{cal_start_week}, {test_start_week}) = {val_year} H2")
    logger.info(f"    Test:  weeks [{test_start_week}, {total_weeks}) = {test_year}")

    train_ds = CrimeWindowDataset(
        counts,
        features,
        groups=groups,
        window_size=window_size,
        start_idx=window_size,
        end_idx=val_start_week,
        exclude_crime_history=exclude_crime_history,
    )
    val_ds = CrimeWindowDataset(
        counts,
        features,
        groups=groups,
        window_size=window_size,
        start_idx=val_start_week,
        end_idx=cal_start_week,
        exclude_crime_history=exclude_crime_history,
    )
    cal_ds = CrimeWindowDataset(
        counts,
        features,
        groups=groups,
        window_size=window_size,
        start_idx=cal_start_week,
        end_idx=test_start_week,
        exclude_crime_history=exclude_crime_history,
    )
    test_ds = CrimeWindowDataset(
        counts,
        features,
        groups=groups,
        window_size=window_size,
        start_idx=test_start_week,
        end_idx=total_weeks,
        exclude_crime_history=exclude_crime_history,
    )

    return {"train": train_ds, "val": val_ds, "cal": cal_ds, "test": test_ds}
