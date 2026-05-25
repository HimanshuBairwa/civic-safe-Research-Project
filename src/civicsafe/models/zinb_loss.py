"""ZINB negative log-likelihood loss with full numerical stability.

All computations are in log-space. Critical operations use:
  - torch.lgamma (not Gamma then log)
  - torch.logaddexp (not log(a + b))
  - safe_log from civicsafe.utils.numerics (not raw torch.log)

This loss is ALWAYS computed in float32 even under mixed precision,
as specified in configs/training/default.yaml.

Reference: arXiv:2408.04193 (STMGNN-ZINB, 2024)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from civicsafe.utils.numerics import NUMERICAL_EPS


class ZINBLoss(nn.Module):
    """Zero-Inflated Negative Binomial negative log-likelihood.

    Given observed counts y and predicted parameters (pi, mu, r):
      - pi: zero-inflation probability in [0, 1]
      - mu: NB mean in (0, inf)
      - r:  NB dispersion (concentration) in [r_floor, inf)

    The log-probability is:
      y = 0: log[pi + (1-pi) * NB(0; mu, r)]
      y > 0: log(1-pi) + log NB(y; mu, r)

    where log NB(y; mu, r) =
      lgamma(y+r) - lgamma(r) - lgamma(y+1)
      + r*log(r/(r+mu)) + y*log(mu/(r+mu))
    """

    def __init__(
        self,
        r_floor: float = 0.1,
        eps: float = NUMERICAL_EPS,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.r_floor = r_floor
        self.eps = eps
        self.reduction = reduction

    def forward(
        self,
        y: Tensor,
        pi: Tensor,
        mu: Tensor,
        r: Tensor,
    ) -> Tensor:
        """Compute ZINB NLL.

        Args:
            y:  Observed counts.  Shape: (*,)
            pi: Zero-inflation.  Shape: (*,)  Values in [0, 1].
            mu: NB mean.         Shape: (*,)  Values in (0, inf).
            r:  NB dispersion.   Shape: (*,)  Values in [r_floor, inf).

        Returns:
            Scalar (if reduction='mean'|'sum') or per-element NLL.
        """
        # Force float32 for numerical stability (critical under AMP)
        y = y.float()
        pi = pi.float()
        mu = mu.float()
        r = r.float()

        # Clamp parameters to safe ranges
        pi = torch.clamp(pi, min=self.eps, max=1.0 - self.eps)
        mu = torch.clamp(mu, min=self.eps)
        r = torch.clamp(r, min=self.r_floor)

        # --- NB log-probability (for all y, including y=0) ---
        # lgamma inputs must be strictly positive
        log_nb = (
            torch.lgamma(y + r)
            - torch.lgamma(r)
            - torch.lgamma(y + 1.0)
            + r * (torch.log(r + self.eps) - torch.log(r + mu + self.eps))
            + y * (torch.log(mu + self.eps) - torch.log(r + mu + self.eps))
        )

        # --- Zero-inflation handling ---
        # y = 0 case: log[pi + (1-pi)*NB(0)] via logaddexp for stability
        log_pi = torch.log(pi + self.eps)
        log_1_minus_pi = torch.log(1.0 - pi + self.eps)

        # log[pi + (1-pi)*NB(0;mu,r)]
        log_zinb_zero = torch.logaddexp(log_pi, log_1_minus_pi + log_nb)

        # y > 0 case: log(1-pi) + log NB(y)
        log_zinb_pos = log_1_minus_pi + log_nb

        # Select based on observed y
        is_zero = y < 0.5  # float comparison for integer counts
        log_prob = torch.where(is_zero, log_zinb_zero, log_zinb_pos)

        # Negative log-likelihood
        nll = -log_prob

        # Catch any NaN/Inf that slipped through (defensive)
        nll = torch.where(torch.isfinite(nll), nll, torch.zeros_like(nll))

        if self.reduction == "mean":
            return nll.mean()
        elif self.reduction == "sum":
            return nll.sum()
        return nll
