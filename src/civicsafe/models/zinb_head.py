"""ZINB 3-parameter projection head.

Three independent 2-layer MLPs project the fused spatiotemporal embedding
into the ZINB distribution parameters:
  pi: zero-inflation probability  (Sigmoid → [0, 1])
  mu: NB mean                     (Softplus → (0, inf))
  r:  NB dispersion               (Softplus + floor → [r_floor, inf))

Weight initialization: final layers use small variance (0.01) to produce
moderate initial predictions, preventing NaN in the ZINB loss during the
first few training steps.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ZINBHead(nn.Module):
    """Three-parameter ZINB projection head.

    Args:
        in_features: Input dimension (from temporal encoder / feature mixer).
        pi_hidden: Hidden dim for the pi MLP.
        mu_hidden: Hidden dim for the mu MLP.
        r_hidden: Hidden dim for the r MLP.
        num_categories: Number of crime categories to predict.
        r_floor: Minimum dispersion value to prevent NaN.
    """

    def __init__(
        self,
        in_features: int = 128,
        pi_hidden: int = 64,
        mu_hidden: int = 64,
        r_hidden: int = 64,
        num_categories: int = 3,
        r_floor: float = 0.1,
        zero_inflation: bool = True,
    ) -> None:
        super().__init__()
        self.r_floor = r_floor
        self.num_categories = num_categories
        self.zero_inflation = zero_inflation

        # Pi MLP: → Sigmoid → [0, 1]
        self.pi_mlp = nn.Sequential(
            nn.Linear(in_features, pi_hidden),
            nn.ReLU(),
            nn.Linear(pi_hidden, num_categories),
        )

        # Mu MLP: → Softplus → (0, inf)
        self.mu_mlp = nn.Sequential(
            nn.Linear(in_features, mu_hidden),
            nn.ReLU(),
            nn.Linear(mu_hidden, num_categories),
        )

        # R MLP: → Softplus + r_floor → [r_floor, inf)
        self.r_mlp = nn.Sequential(
            nn.Linear(in_features, r_hidden),
            nn.ReLU(),
            nn.Linear(r_hidden, num_categories),
        )

        # Initialize final layers with small weights for stable early training
        self._init_weights()

    def _init_weights(self) -> None:
        """Small-variance initialization for final projection layers."""
        for mlp in [self.pi_mlp, self.mu_mlp, self.r_mlp]:
            final_layer = mlp[-1]
            nn.init.normal_(final_layer.weight, mean=0.0, std=0.01)  # type: ignore[arg-type]
            nn.init.zeros_(final_layer.bias)  # type: ignore[arg-type]

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Project embedding to ZINB parameters.

        Args:
            x: Fused embedding. Shape: (B, D)

        Returns:
            Tuple of (pi, mu, r), each of shape (B, num_categories):
              pi: zero-inflation probability in [0, 1]
              mu: NB mean in (0, inf)
              r:  NB dispersion in [r_floor, inf)
        """
        if self.zero_inflation:
            pi = torch.sigmoid(self.pi_mlp(x))  # (B, C)
        else:
            pi = torch.zeros(x.shape[0], self.num_categories, device=x.device)
            
        mu = F.softplus(self.mu_mlp(x))  # (B, C)
        r = F.softplus(self.r_mlp(x)) + self.r_floor  # (B, C)

        return pi, mu, r
