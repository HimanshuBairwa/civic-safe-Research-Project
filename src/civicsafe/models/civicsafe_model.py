"""CIVIC-SAFE Master Model: GATv2 → Causal Transformer → MFFM → ZINB Head.

This is the complete spatiotemporal graph neural network that outputs
full probabilistic forecasts for crime counts as ZINB parameters.

Architecture:
  1. SpatialEncoder (GATv2, dual adjacency) processes each timestep independently
  2. TemporalEncoder (Causal Transformer) processes the sequence autoregressively
  3. FeatureMixer (MFFM) decomposes into interpretable factors
  4. ZINBHead projects to (pi, mu, r) per spatial unit per category

Supports:
  - torch.cuda.amp.autocast for mixed precision
  - torch.utils.checkpoint for gradient checkpointing
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch import Tensor

from civicsafe.models.feature_mixer import FeatureMixer
from civicsafe.models.spatial import SpatialEncoder
from civicsafe.models.temporal import TemporalEncoder
from civicsafe.models.zinb_head import ZINBHead
from civicsafe.models.adversarial_head import AdversarialDiscriminator


class CivicSafeModel(nn.Module):
    """Complete CIVIC-SAFE spatiotemporal ZINB forecasting model.

    Args:
        num_features: Number of input features per node per timestep.
        hidden_dim: Hidden embedding dimension throughout the model.
        spatial_layers: Number of GATv2 layers.
        spatial_heads: Number of GAT attention heads.
        temporal_layers: Number of transformer layers.
        temporal_heads: Number of transformer attention heads.
        temporal_ff_dim: Transformer FFN intermediate dimension.
        mixer_heads: Number of MFFM factor heads.
        mixer_temperature: MFFM softmax temperature.
        mixer_collapse_threshold: MFFM JSD collapse threshold.
        num_categories: Number of crime categories to predict.
        pi_hidden: ZINB pi MLP hidden dim.
        mu_hidden: ZINB mu MLP hidden dim.
        r_hidden: ZINB r MLP hidden dim.
        r_floor: Minimum dispersion value.
        max_seq_len: Maximum sequence length (weeks).
        dropout: Global dropout rate.
        use_gradient_checkpointing: Whether to use gradient checkpointing.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        spatial_layers: int = 2,
        spatial_heads: int = 4,
        temporal_layers: int = 2,
        temporal_heads: int = 4,
        temporal_ff_dim: int = 512,
        mixer_heads: int = 3,
        mixer_temperature: float = 1.0,
        mixer_collapse_threshold: float = 0.1,
        num_categories: int = 3,
        pi_hidden: int = 64,
        mu_hidden: int = 64,
        r_hidden: int = 64,
        r_floor: float = 0.1,
        max_seq_len: int = 52,
        dropout: float = 0.1,
        num_adv_classes: int = 0,
        adv_lambda: float = 1.0,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Input projection: F_in → hidden_dim
        self.input_proj = nn.Linear(num_features, hidden_dim)

        # Spatial encoder (GATv2 with dual adjacency)
        self.spatial_encoder = SpatialEncoder(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            num_layers=spatial_layers,
            num_heads=spatial_heads,
            dropout=dropout,
        )

        # Temporal encoder (Causal Transformer)
        self.temporal_encoder = TemporalEncoder(
            d_model=hidden_dim,
            num_heads=temporal_heads,
            num_layers=temporal_layers,
            dim_feedforward=temporal_ff_dim,
            dropout=dropout,
            max_seq_len=max_seq_len,
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

        # Adversarial Head (GRL)
        self.adv_head = None
        if num_adv_classes > 0:
            self.adv_head = AdversarialDiscriminator(
                in_features=hidden_dim,
                hidden_dim=hidden_dim,
                num_classes=num_adv_classes,
                lambda_=adv_lambda,
            )

    def forward(
        self,
        features: Tensor,
        edge_index_queen: Tensor,
        edge_index_knn: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Full forward pass.

        Args:
            features: Input features. Shape: (S, T, F)
                S = spatial units, T = time steps, F = features
            edge_index_queen: Queen contiguity edges. Shape: (2, E_q)
            edge_index_knn: K-NN edges. Shape: (2, E_k) or None.

        Returns:
            Dictionary with keys:
              pi: (S, C) zero-inflation probabilities
              mu: (S, C) NB means
              r:  (S, C) NB dispersions
              diversity_loss: scalar MFFM regularization term
              adv_logits: (S, num_adv_classes) demographic predictions (if enabled)
        """
        S, T, F = features.shape

        # Project input features to hidden dim
        x = self.input_proj(features)  # (S, T, hidden_dim)

        # --- Spatial encoding: process each timestep ---
        spatial_outputs = []
        for t in range(T):
            x_t = x[:, t, :]  # (S, hidden_dim)

            if self.use_gradient_checkpointing and self.training:
                h_t = cp.checkpoint(
                    self._spatial_forward,
                    x_t,
                    edge_index_queen,
                    edge_index_knn,
                    use_reentrant=False,
                )
            else:
                h_t = self._spatial_forward(x_t, edge_index_queen, edge_index_knn)

            spatial_outputs.append(h_t)

        # Stack: (S, T, hidden_dim)
        spatial_seq = torch.stack(spatial_outputs, dim=1)

        # --- Temporal encoding: process the sequence ---
        temporal_out = self.temporal_encoder(spatial_seq)  # (S, T, hidden_dim)

        # --- Feature mixing ---
        mixed, diversity_loss = self.feature_mixer(temporal_out)  # (S, T, hidden_dim)

        # --- ZINB prediction from the last timestep ---
        last_hidden = mixed[:, -1, :]  # (S, hidden_dim)
        pi, mu, r = self.zinb_head(last_hidden)  # each (S, C)

        # --- Adversarial demographic prediction ---
        adv_logits = None
        if self.adv_head is not None:
            adv_logits = self.adv_head(last_hidden)

        return {
            "pi": pi,
            "mu": mu,
            "r": r,
            "diversity_loss": diversity_loss,
            "adv_logits": adv_logits,
        }

    def _spatial_forward(
        self,
        x_t: Tensor,
        edge_index_queen: Tensor,
        edge_index_knn: Tensor | None,
    ) -> Tensor:
        """Spatial encoding for a single timestep (checkpointable)."""
        return self.spatial_encoder(x_t, edge_index_queen, edge_index_knn)  # type: ignore[no-any-return]
