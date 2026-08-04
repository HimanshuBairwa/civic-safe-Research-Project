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
        level_anchor: If True, ``mu`` is a multiplicative correction on a
            caller-supplied level anchor instead of an absolute prediction.
            See :meth:`forward` for why this matters.
        delta_clamp: Symmetric bound on the log-correction when
            ``level_anchor`` is on. 3.0 permits a 0.05x-20x adjustment of the
            anchor, which spans every week-over-week move in the panel while
            keeping ``exp`` out of overflow range under fp16 autocast.
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
        level_anchor: bool = False,
        delta_clamp: float = 3.0,
    ) -> None:
        super().__init__()
        self.r_floor = r_floor
        self.num_categories = num_categories
        self.zero_inflation = zero_inflation
        self.level_anchor = level_anchor
        self.delta_clamp = delta_clamp

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

    def forward(
        self, x: Tensor, anchor: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Project embedding to ZINB parameters.

        Args:
            x: Fused embedding. Shape: (B, D)
            anchor: Optional per-cell level anchor, shape (B, num_categories),
                in COUNT space (not log space). Required when
                ``level_anchor`` is True; ignored otherwise.

        Returns:
            Tuple of (pi, mu, r), each of shape (B, num_categories):
              pi: zero-inflation probability in [0, 1]
              mu: NB mean in (0, inf)
              r:  NB dispersion in [r_floor, inf)

        On the mu parameterization
        --------------------------
        With ``level_anchor=False`` (default), ``mu = softplus(mu_mlp(x))`` --
        an absolute prediction built from scratch. Because the final layer is
        zero-bias/small-weight initialized, every cell starts at
        ``softplus(0) ~ 0.693`` and the network must learn the absolute level
        of each (unit, category) cell through nonlinear layers from normalized
        log1p inputs. Those levels span a wide range across the panel
        (per-category test CRPS: Property 5.70 vs Drug 0.76), so most of the
        head's capacity goes into re-deriving a level that a trailing mean
        already knows for free. Measured consequence on Chicago: a rolling
        52-week historical average scores CRPS 2.9322 while this model scores
        3.2291 -- the 689K-parameter network loses to an arithmetic mean.

        With ``level_anchor=True``, mu becomes a multiplicative correction:

            mu = anchor * exp(clamp(mu_mlp(x), -delta_clamp, +delta_clamp))

        At initialization mu_mlp(x) ~ 0, so exp(0) = 1 and ``mu == anchor``
        exactly: the model *starts* as the rolling-mean baseline and learns
        only the residual departure from it. The network is then spending its
        capacity on the part a mean cannot express (spatial spillover,
        seasonality, covariate shocks) instead of on level reconstruction.
        This is the DeepAR scale-handling argument (Salinas et al., 2020,
        Sec. 3.3) expressed multiplicatively rather than by input rescaling,
        which suits count data where the anchor can legitimately be zero.

        The clamp is load-bearing: without it a single large logit produces
        ``exp(large) = inf``, and inf*0 in the CRPS CDF sum yields NaN that
        silently poisons the epoch. Clamping bounds mu to
        [anchor*0.05, anchor*20] and keeps the gradient finite.
        """
        if self.zero_inflation:
            pi = torch.sigmoid(self.pi_mlp(x))  # (B, C)
        else:
            pi = torch.zeros(x.shape[0], self.num_categories, device=x.device)

        if self.level_anchor:
            if anchor is None:
                raise ValueError(
                    "ZINBHead was constructed with level_anchor=True but "
                    "forward() received anchor=None. The caller must supply "
                    "a (B, num_categories) level anchor in count space; "
                    "CivicSafeModel derives it from the crime-history "
                    "channels of its input. Silently falling back to the "
                    "unanchored path would produce a model whose predictions "
                    "do not match the architecture recorded in the "
                    "checkpoint, so this is an error rather than a warning."
                )
            delta = torch.clamp(
                self.mu_mlp(x), min=-self.delta_clamp, max=self.delta_clamp
            )
            # Anchor is floored below by the caller, so mu stays strictly
            # positive even for all-zero history.
            mu = anchor * torch.exp(delta)  # (B, C)
        else:
            mu = F.softplus(self.mu_mlp(x))  # (B, C)

        r = F.softplus(self.r_mlp(x)) + self.r_floor  # (B, C)

        return pi, mu, r
