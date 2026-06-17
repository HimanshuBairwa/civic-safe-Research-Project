"""CIVIC-SAFE V2: Unified Spatiotemporal Graph Transformer → MFFM → ZINB Head.

Unlike V1 (sequential Spatial→Temporal), V2 uses a single Spatiotemporal
Graph Transformer that captures joint space-time dependencies via structured
attention masks. This enables:
  - Cross-spatial temporal patterns (city-wide events)
  - Temporal patterns informed by spatial context at each step
  - Unified positional encoding (node ID + timestep)

For ablation studies, both V1 and V2 can be compared to quantify the
benefit of joint spatiotemporal attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from civicsafe.models.feature_mixer import FeatureMixer
from civicsafe.models.st_transformer import SpatiotemporalTransformer
from civicsafe.models.zinb_head import ZINBHead


class CivicSafeModelV2(nn.Module):
    """CIVIC-SAFE V2 with unified Spatiotemporal Graph Transformer.

    Args:
        num_features: Number of input features per node per timestep.
        hidden_dim: Hidden embedding dimension.
        st_layers: Number of spatiotemporal transformer layers.
        st_heads: Number of attention heads.
        st_ff_dim: FFN intermediate dimension.
        max_nodes: Maximum number of spatial nodes.
        max_seq_len: Maximum sequence length (weeks).
        mixer_heads: Number of MFFM factor heads.
        mixer_temperature: MFFM softmax temperature.
        mixer_collapse_threshold: MFFM JSD collapse threshold.
        num_categories: Number of crime categories to predict.
        pi_hidden: ZINB pi MLP hidden dim.
        mu_hidden: ZINB mu MLP hidden dim.
        r_hidden: ZINB r MLP hidden dim.
        r_floor: Minimum dispersion value.
        dropout: Global dropout rate.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        st_layers: int = 3,
        st_heads: int = 4,
        st_ff_dim: int = 512,
        max_nodes: int = 100,
        max_seq_len: int = 52,
        mixer_heads: int = 3,
        mixer_temperature: float = 1.0,
        mixer_collapse_threshold: float = 0.1,
        num_categories: int = 3,
        pi_hidden: int = 64,
        mu_hidden: int = 64,
        r_hidden: int = 64,
        r_floor: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input projection: F_in → hidden_dim
        self.input_proj = nn.Linear(num_features, hidden_dim)

        # Unified spatiotemporal transformer
        self.st_transformer = SpatiotemporalTransformer(
            d_model=hidden_dim,
            num_layers=st_layers,
            num_heads=st_heads,
            dim_feedforward=st_ff_dim,
            max_nodes=max_nodes,
            max_time=max_seq_len,
            dropout=dropout,
        )

        # Feature mixer (MFFM)
        self.feature_mixer = FeatureMixer(
            d_model=hidden_dim,
            num_heads=mixer_heads,
            temperature=mixer_temperature,
            collapse_threshold=mixer_collapse_threshold,
        )

        # ZINB projection head
        self.zinb_head = ZINBHead(
            in_features=hidden_dim,
            pi_hidden=pi_hidden,
            mu_hidden=mu_hidden,
            r_hidden=r_hidden,
            num_categories=num_categories,
            r_floor=r_floor,
        )

    def forward(
        self,
        features: Tensor,
        edge_index_queen: Tensor,
        edge_index_knn: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Full forward pass with unified spatiotemporal encoding.

        Args:
            features: Input features. Shape: (S, T, F)
            edge_index_queen: Queen contiguity edges. Shape: (2, E_q)
            edge_index_knn: K-NN edges. Shape: (2, E_k) or None.

        Returns:
            Dictionary with keys:
              pi: (S, C) zero-inflation probabilities
              mu: (S, C) NB means
              r:  (S, C) NB dispersions
              diversity_loss: scalar MFFM regularization term
        """
        S, T, F = features.shape

        # Project input features to hidden dim
        x = self.input_proj(features)  # (S, T, hidden_dim)

        # Combine edge indices for the unified transformer
        if edge_index_knn is not None:
            edge_index = torch.cat([edge_index_queen, edge_index_knn], dim=1)
        else:
            edge_index = edge_index_queen

        # --- Unified spatiotemporal encoding ---
        st_out = self.st_transformer(x, edge_index)  # (S, T, hidden_dim)

        # --- Feature mixing ---
        mixed, diversity_loss = self.feature_mixer(st_out)  # (S, T, hidden_dim)

        # --- ZINB prediction from the last timestep ---
        last_hidden = mixed[:, -1, :]  # (S, hidden_dim)
        pi, mu, r = self.zinb_head(last_hidden)  # each (S, C)

        return {
            "pi": pi,
            "mu": mu,
            "r": r,
            "diversity_loss": diversity_loss,
        }
