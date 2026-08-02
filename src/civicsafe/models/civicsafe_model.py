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


class NoGraphSpatialBypass(nn.Module):
    """Per-node MLP replacing GATv2 — ablates spatial message passing.

    Deliberately keeps the ``SpatialEncoder`` call signature (and accepts the
    edge indices, ignoring them) so the surrounding forward path is unchanged.
    Each node is encoded from its own features alone.

    Matching depth, width, and nonlinearity to the encoder it replaces is what
    makes this a test of *message passing* specifically. A bypass that also
    changed capacity would confound "graph structure doesn't help" with
    "the replacement was simply smaller".

    Args:
        hidden_dim: Embedding width, matched to the GATv2 encoder.
        num_layers: Number of MLP blocks, matched to ``spatial_layers``.
        dropout: Dropout rate, matched to the encoder.
    """

    def __init__(
        self, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(num_layers):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        x: Tensor,
        edge_index_queen: Tensor | None = None,
        edge_index_knn: Tensor | None = None,
    ) -> Tensor:
        """Encode nodes independently, ignoring graph structure.

        Args:
            x: Node features, either (S, D) for one timestep or (S, T, D).
            edge_index_queen: Accepted for signature parity; unused.
            edge_index_knn: Accepted for signature parity; unused.

        Returns:
            Same shape as ``x``.
        """
        return self.mlp(x)  # type: ignore[no-any-return]


class NoAttentionTemporalBypass(nn.Module):
    """Per-timestep MLP + causal running mean, replacing the causal Transformer.

    Position ``t`` receives the uniform average of encoded steps ``0..t``, so
    the receptive field and the causality constraint are identical to the
    Transformer's. The only thing removed is *learned* attention weighting.

    That makes this a test of whether attention beats uniform pooling, rather
    than a test of whether the model can see history at all — a bypass that
    only saw the current step would collapse the comparison into "history
    helps", which is not in question.

    Args:
        d_model: Model width, matched to the Transformer it replaces.
        dropout: Dropout rate, matched to the Transformer.
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Encode each step, then causally average.

        Args:
            x: Sequence of spatial embeddings. Shape: (S, T, D)

        Returns:
            (S, T, D) — position ``t`` is the mean of encoded steps ``0..t``.
        """
        h = self.mlp(x)
        # Running mean over time: cumsum / (1, 2, ..., T). Strictly causal.
        csum = h.cumsum(dim=1)
        counts = torch.arange(1, h.shape[1] + 1, device=h.device, dtype=h.dtype)
        return self.norm(csum / counts.view(1, -1, 1))  # type: ignore[no-any-return]


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
        use_gnn: If False, replace GATv2 with a per-node MLP (spatial ablation).
        use_transformer: If False, replace the causal Transformer with a
            per-step MLP plus causal mean pooling (temporal ablation).
        zero_inflation: If False, the ZINB head emits pi=0 (pure NB ablation).
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
        use_gnn: bool = True,
        use_transformer: bool = True,
        zero_inflation: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gnn = use_gnn
        self.use_transformer = use_transformer

        # Input projection: F_in → hidden_dim
        self.input_proj = nn.Linear(num_features, hidden_dim)

        # Spatial encoder (GATv2 with dual adjacency), or MLP bypass if ablated
        if use_gnn:
            self.spatial_encoder: nn.Module = SpatialEncoder(
                in_channels=hidden_dim,
                hidden_channels=hidden_dim,
                num_layers=spatial_layers,
                num_heads=spatial_heads,
                dropout=dropout,
            )
        else:
            self.spatial_encoder = NoGraphSpatialBypass(
                hidden_dim=hidden_dim,
                num_layers=spatial_layers,
                dropout=dropout,
            )

        # Temporal encoder (Causal Transformer), or causal-mean bypass if ablated
        if use_transformer:
            self.temporal_encoder: nn.Module = TemporalEncoder(
                d_model=hidden_dim,
                num_heads=temporal_heads,
                num_layers=temporal_layers,
                dim_feedforward=temporal_ff_dim,
                dropout=dropout,
                max_seq_len=max_seq_len,
            )
        else:
            self.temporal_encoder = NoAttentionTemporalBypass(
                d_model=hidden_dim, dropout=dropout
            )

        # Feature mixer (MFFM)
        self.feature_mixer = FeatureMixer(
            d_model=hidden_dim,
            num_heads=mixer_heads,
            temperature=mixer_temperature,
            collapse_threshold=mixer_collapse_threshold,
        )

        # ZINB projection head (pi head is disabled when zero_inflation=False)
        self.zinb_head = ZINBHead(
            in_features=hidden_dim,
            pi_hidden=pi_hidden,
            mu_hidden=mu_hidden,
            r_hidden=r_hidden,
            num_categories=num_categories,
            r_floor=r_floor,
            zero_inflation=zero_inflation,
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
        """Spatial encoding for a single timestep (checkpointable).

        When ``use_gnn=False``, the spatial encoder is a per-node MLP that
        ignores the edge tensors — this forward path passes them anyway for
        signature parity, keeping the caller simple.
        """
        return self.spatial_encoder(x_t, edge_index_queen, edge_index_knn)  # type: ignore[no-any-return]
