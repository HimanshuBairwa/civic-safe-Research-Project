#!/usr/bin/env python
"""Deep learning baselines for CIVIC-SAFE spatiotemporal crime forecasting.

Implements three deep baselines on the SAME data pipeline as CIVIC-SAFE,
enabling fair head-to-head comparison for NeurIPS/KDD submission:

  1. LSTM-NB       — 2-layer LSTM → Negative Binomial output head
  2. SimplifiedTFT — Self-attention + Variable Selection → ZINB output + CRPS loss
  3. GraphWaveNet  — Dilated causal convolution + adaptive adjacency → ZINB + CRPS

All models use:
  - CrimeWindowDataset / create_chronological_splits (same splits as CIVIC-SAFE)
  - Training-period-only normalisation (no data leakage)
  - Validation-based early stopping
  - compute_all_metrics / crps_zinb for evaluation (same metrics as main pipeline)

Usage:
    python scripts/deep_baselines.py data=chicago
    python scripts/deep_baselines.py data=nyc
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from civicsafe.models.dataset import CrimeWindowDataset, create_chronological_splits
from civicsafe.training.metrics import compute_all_metrics, crps_zinb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: collate function for CrimeWindowDataset
# ---------------------------------------------------------------------------
# Each dataset item is a dict with tensors shaped (S, W, C/F) and (S, C).
# DataLoader default_collate would add a batch dim → (B, S, W, C).
# For our per-node models we flatten S into the batch → (B*S, W, C).
# For the graph model we keep S separate and stack time windows → (B, S, W, C).

def _collate_flat(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate by concatenating spatial units into the batch dimension.

    Input items:  input_counts (S,W,C), input_features (S,W,F), target_counts (S,C)
    Output:       input_counts (B*S,W,C), input_features (B*S,W,F), target_counts (B*S,C)
    """
    ic = torch.cat([b["input_counts"] for b in batch], dim=0)
    iff = torch.cat([b["input_features"] for b in batch], dim=0)
    tc = torch.cat([b["target_counts"] for b in batch], dim=0)
    return {"input_counts": ic, "input_features": iff, "target_counts": tc}


def _collate_graph(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate by stacking into (B, S, W, C/F) — preserves spatial topology."""
    ic = torch.stack([b["input_counts"] for b in batch], dim=0)
    iff = torch.stack([b["input_features"] for b in batch], dim=0)
    tc = torch.stack([b["target_counts"] for b in batch], dim=0)
    return {"input_counts": ic, "input_features": iff, "target_counts": tc}


# ============================================================================
# Baseline 1: LSTM with Negative Binomial Output Head
# ============================================================================

class LSTMNBModel(nn.Module):
    """2-layer LSTM with Negative Binomial output head for crime count forecasting.

    Operates independently per spatial unit (no explicit graph structure).
    Input:  concatenated [counts, features] → (batch, window, C+F)
    Output: NB parameters (mu, r) per crime category

    Since there is no zero-inflation term, pi is fixed to 0 for evaluation
    with ZINB metrics (NB is a special case of ZINB with pi=0).

    Args:
        input_dim: Total input feature dimension (C + F).
        hidden_dim: LSTM hidden state dimension.
        num_layers: Number of stacked LSTM layers.
        num_categories: Number of crime categories (C).
        dropout: Dropout between LSTM layers.
        r_floor: Minimum dispersion to prevent numerical issues.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_categories: int = 3,
        dropout: float = 0.1,
        r_floor: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_categories = num_categories
        self.r_floor = r_floor

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Mu head: Softplus → (0, ∞)
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_categories),
        )
        # R (dispersion) head: Softplus + floor → [r_floor, ∞)
        self.r_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_categories),
        )
        self._init_output_heads()

    def _init_output_heads(self) -> None:
        """Small-variance init on final layers for stable early training."""
        for head in [self.mu_head, self.r_head]:
            final = head[-1]
            nn.init.normal_(final.weight, 0.0, 0.01)
            nn.init.zeros_(final.bias)

    def forward(
        self, counts: Tensor, features: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass.

        Args:
            counts:   (B, W, C) — historical crime counts window.
            features: (B, W, F) — historical covariate window.

        Returns:
            (pi, mu, r) each of shape (B, C).
            pi is always zero (pure NB, no zero-inflation).
        """
        # Concatenate counts and features along the feature dimension
        x = torch.cat([counts, features], dim=-1)  # (B, W, C+F)

        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # (B, W, hidden)
        h_last = lstm_out[:, -1, :]  # (B, hidden) — use final timestep
        h_last = self.layer_norm(h_last)

        # Output distribution parameters
        mu = F.softplus(self.mu_head(h_last))  # (B, C)
        r = F.softplus(self.r_head(h_last)) + self.r_floor  # (B, C)
        pi = torch.zeros_like(mu)  # NB, not ZINB → pi=0

        return pi, mu, r


def nb_nll_loss(y: Tensor, mu: Tensor, r: Tensor) -> Tensor:
    """Negative Binomial NLL (no zero-inflation).

    NLL = -[lgamma(y+r) - lgamma(r) - lgamma(y+1)
            + r*log(r/(r+mu)) + y*log(mu/(r+mu))]

    Args:
        y:  Observed counts (B, C).
        mu: NB mean (B, C), strictly positive.
        r:  NB dispersion (B, C), strictly positive.

    Returns:
        Scalar mean NLL.
    """
    eps = 1e-8
    y = y.float()
    mu = mu.clamp(min=eps)
    r = r.clamp(min=0.1)

    log_nb = (
        torch.lgamma(y + r)
        - torch.lgamma(r)
        - torch.lgamma(y + 1.0)
        + r * (torch.log(r + eps) - torch.log(r + mu + eps))
        + y * (torch.log(mu + eps) - torch.log(r + mu + eps))
    )
    return -log_nb.mean()


# ============================================================================
# Baseline 2: Simplified Temporal Fusion Transformer (TFT)
# ============================================================================

class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) — core building block of TFT.

    Implements: GRN(a) = LayerNorm(a + GLU(W1·ELU(W2·a)))

    Args:
        d_model: Input and output dimension.
        d_hidden: Hidden dimension inside the GRN.
        dropout: Dropout rate.
    """

    def __init__(self, d_model: int, d_hidden: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model * 2)  # *2 for GLU
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        h = F.elu(self.fc1(x))
        h = self.dropout(self.fc2(h))
        # GLU: split into two halves, sigmoid-gate the second
        h1, h2 = h.chunk(2, dim=-1)
        h = h1 * torch.sigmoid(h2)
        return self.layer_norm(x + h)


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network (VSN) — learns which features matter.

    Produces per-variable importance weights via a softmax gate, then
    applies a GRN to each selected variable.

    Args:
        num_vars: Number of input variables (features).
        d_model: Embedding dimension per variable.
        dropout: Dropout rate.
    """

    def __init__(self, num_vars: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_vars = num_vars
        self.d_model = d_model

        # Per-variable GRNs
        self.var_grns = nn.ModuleList(
            [GatedResidualNetwork(d_model, d_model, dropout) for _ in range(num_vars)]
        )
        # Gate: flattened input → softmax weights over variables
        self.gate_grn = GatedResidualNetwork(num_vars * d_model, num_vars * d_model, dropout)
        self.gate_proj = nn.Linear(num_vars * d_model, num_vars)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: (B, T, num_vars * d_model) — concatenated variable embeddings.

        Returns:
            (B, T, d_model) — weighted sum of transformed variables.
        """
        B, T, _ = x.shape

        # Split into per-variable chunks
        chunks = x.view(B, T, self.num_vars, self.d_model)  # (B, T, V, D)

        # Apply per-variable GRN
        transformed = []
        for i in range(self.num_vars):
            transformed.append(self.var_grns[i](chunks[:, :, i, :]))
        transformed = torch.stack(transformed, dim=2)  # (B, T, V, D)

        # Compute variable importance weights
        flat = x.view(B * T, -1)
        gate_input = self.gate_grn(flat)
        weights = torch.softmax(self.gate_proj(gate_input), dim=-1)  # (B*T, V)
        weights = weights.view(B, T, self.num_vars, 1)  # (B, T, V, 1)

        # Weighted combination
        out = (weights * transformed).sum(dim=2)  # (B, T, D)
        return out


class SimplifiedTFTModel(nn.Module):
    """Simplified Temporal Fusion Transformer for crime count forecasting.

    Architecture:
      1. Input projection to d_model per variable group (counts, features)
      2. Variable Selection Network to gate which features matter
      3. Sinusoidal positional encoding
      4. Causal multi-head self-attention over the temporal dimension
      5. Feed-forward GRN
      6. ZINB output head (pi, mu, r) — same distribution as CIVIC-SAFE

    Trained with CRPS loss for FAIR comparison with CIVIC-SAFE.

    Args:
        count_dim: Number of crime categories (C).
        feature_dim: Number of covariate features (F).
        d_model: Transformer model dimension.
        nhead: Number of attention heads.
        num_layers: Number of attention blocks.
        dropout: Dropout rate.
        r_floor: Minimum NB dispersion.
    """

    def __init__(
        self,
        count_dim: int,
        feature_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        r_floor: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.r_floor = r_floor

        # Project count and feature groups to d_model each
        num_var_groups = 2  # counts, features
        self.count_proj = nn.Linear(count_dim, d_model)
        self.feature_proj = nn.Linear(feature_dim, d_model)

        # Variable selection: decides importance of counts vs. features
        self.vsn = VariableSelectionNetwork(
            num_vars=num_var_groups, d_model=d_model, dropout=dropout
        )

        # Positional encoding (sinusoidal, same as CIVIC-SAFE temporal encoder)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 1, d_model), requires_grad=False)
        self._init_sinusoidal_pe(max_len=104)

        # Causal self-attention layers
        self.attn_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attn_layers.append(
                nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=nhead,
                    dropout=dropout,
                    batch_first=True,
                )
            )
            self.ffn_layers.append(GatedResidualNetwork(d_model, d_model * 2, dropout))

        self.final_norm = nn.LayerNorm(d_model)

        # ZINB output head
        self.pi_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, count_dim),
        )
        self.mu_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, count_dim),
        )
        self.r_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, count_dim),
        )
        self._init_output_heads()

    def _init_sinusoidal_pe(self, max_len: int = 104) -> None:
        """Pre-compute sinusoidal positional encodings."""
        pe = torch.zeros(max_len, self.d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("sinusoidal_pe", pe.unsqueeze(0))  # (1, max_len, D)

    def _init_output_heads(self) -> None:
        for head in [self.pi_head, self.mu_head, self.r_head]:
            final = head[-1]
            nn.init.normal_(final.weight, 0.0, 0.01)
            nn.init.zeros_(final.bias)

    def forward(
        self, counts: Tensor, features: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass.

        Args:
            counts:   (B, W, C) — historical crime counts.
            features: (B, W, F) — historical covariates.

        Returns:
            (pi, mu, r) each of shape (B, C).
        """
        B, W, _C = counts.shape

        # Project variable groups
        c_emb = self.count_proj(counts)    # (B, W, D)
        f_emb = self.feature_proj(features)  # (B, W, D)

        # Concatenate for VSN: (B, W, 2*D)
        vsn_input = torch.cat([c_emb, f_emb], dim=-1)
        x = self.vsn(vsn_input)  # (B, W, D)

        # Add positional encoding
        x = x + self.sinusoidal_pe[:, :W, :]

        # Causal self-attention (mask prevents attending to future)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(W, device=x.device)
        for attn, ffn in zip(self.attn_layers, self.ffn_layers):
            residual = x
            x_attn, _ = attn(x, x, x, attn_mask=causal_mask, is_causal=True)
            x = residual + x_attn
            x = ffn(x)

        x = self.final_norm(x)

        # Take the last timestep for prediction
        h = x[:, -1, :]  # (B, D)

        # ZINB parameters
        pi = torch.sigmoid(self.pi_head(h))  # (B, C)
        mu = F.softplus(self.mu_head(h))      # (B, C)
        r = F.softplus(self.r_head(h)) + self.r_floor  # (B, C)

        return pi, mu, r


# ============================================================================
# Baseline 3: Graph WaveNet
# ============================================================================

class DilatedCausalConv(nn.Module):
    """Single dilated causal 1D convolution with gated activation.

    Implements: tanh(Wf * x) ⊙ σ(Wg * x), where * denotes causal conv.
    Causal padding ensures no information from the future leaks in.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        dilation: Dilation factor.
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 2, dilation: int = 1
    ) -> None:
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        # Padding for causal conv: (kernel_size - 1) * dilation on the left
        self.padding = (kernel_size - 1) * dilation

        self.filter_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation
        )
        self.gate_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: (B, C, T) — input sequence in channels-first format.

        Returns:
            (B, out_channels, T) — gated output (same temporal length).
        """
        # Causal padding: pad only on the left
        x_pad = F.pad(x, (self.padding, 0))
        f = torch.tanh(self.filter_conv(x_pad))
        g = torch.sigmoid(self.gate_conv(x_pad))
        return f * g


class GraphConvLayer(nn.Module):
    """Graph convolution using an adjacency matrix (dense, not sparse).

    Implements: X' = A_hat @ X @ W, where A_hat is row-normalised adjacency.

    Args:
        in_features: Input node feature dimension.
        out_features: Output node feature dimension.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x:   (B, S, D) — node features.
            adj: (S, S)    — adjacency matrix (row-normalised).

        Returns:
            (B, S, D_out)
        """
        # Spatial message passing: aggregate neighbour features
        h = torch.matmul(adj, x)  # (B, S, D)
        return self.linear(h)


class GraphWaveNetModel(nn.Module):
    """Graph WaveNet for spatiotemporal crime forecasting.

    Combines:
      - Dilated causal convolutions (WaveNet-style) for temporal modelling
      - Adaptive + learned adjacency matrix for spatial message passing
      - ZINB output head (pi, mu, r) for probabilistic forecasting

    The adaptive adjacency matrix is learned as A = softmax(E1 @ E2^T),
    where E1, E2 are learnable node embeddings. This allows the model to
    discover spatial relationships beyond queen contiguity.

    Args:
        num_nodes: Number of spatial units (S).
        input_dim: Per-node input feature dimension (C + F).
        channels: Internal channel dimension for WaveNet blocks.
        num_layers: Number of dilated causal conv layers.
        kernel_size: Temporal convolution kernel size.
        num_categories: Number of crime categories.
        embed_dim: Dimension of adaptive adjacency node embeddings.
        dropout: Dropout rate.
        r_floor: Minimum NB dispersion.
    """

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        channels: int = 32,
        num_layers: int = 4,
        kernel_size: int = 2,
        num_categories: int = 3,
        embed_dim: int = 16,
        dropout: float = 0.1,
        r_floor: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.num_categories = num_categories
        self.r_floor = r_floor
        self.channels = channels

        # Input projection: (C+F) → channels
        self.input_proj = nn.Conv1d(input_dim, channels, kernel_size=1)

        # Adaptive adjacency: learnable node embeddings
        self.node_emb1 = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.1)
        self.node_emb2 = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.1)

        # WaveNet blocks with skip connections
        self.temporal_convs = nn.ModuleList()
        self.graph_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        for i in range(num_layers):
            dilation = 2 ** i
            self.temporal_convs.append(
                DilatedCausalConv(channels, channels, kernel_size, dilation)
            )
            self.graph_convs.append(GraphConvLayer(channels, channels))
            self.residual_convs.append(nn.Conv1d(channels, channels, 1))
            self.skip_convs.append(nn.Conv1d(channels, channels, 1))
            self.bn_layers.append(nn.BatchNorm1d(channels))

        self.dropout = nn.Dropout(dropout)

        # Output MLP: skip_sum → ZINB parameters
        self.output_fc1 = nn.Linear(channels, channels)
        self.pi_head = nn.Linear(channels, num_categories)
        self.mu_head = nn.Linear(channels, num_categories)
        self.r_head = nn.Linear(channels, num_categories)
        self._init_output_heads()

    def _init_output_heads(self) -> None:
        for head in [self.pi_head, self.mu_head, self.r_head]:
            nn.init.normal_(head.weight, 0.0, 0.01)
            nn.init.zeros_(head.bias)

    def _get_adaptive_adj(self) -> Tensor:
        """Compute adaptive adjacency from learned node embeddings.

        Returns:
            (S, S) row-normalised adjacency matrix.
        """
        adj = F.softmax(F.relu(self.node_emb1 @ self.node_emb2.T), dim=1)
        return adj

    def forward(
        self, counts: Tensor, features: Tensor, static_adj: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass.

        Args:
            counts:     (B, S, W, C) — historical crime counts.
            features:   (B, S, W, F) — historical covariates.
            static_adj: (S, S) — optional fixed adjacency (queen contiguity).

        Returns:
            (pi, mu, r) each of shape (B, S, C).
        """
        B, S, W, C = counts.shape
        F_dim = features.shape[-1]

        # Concatenate counts and features: (B, S, W, C+F)
        x = torch.cat([counts, features], dim=-1)

        # Reshape for per-node temporal conv: (B*S, C+F, W)
        x = x.reshape(B * S, W, C + F_dim).permute(0, 2, 1)
        x = self.input_proj(x)  # (B*S, channels, W)

        # Compute adaptive adjacency
        adap_adj = self._get_adaptive_adj()  # (S, S)

        # Combine with static adjacency if provided
        if static_adj is not None:
            static_adj = static_adj.to(adap_adj.device)
            adj = 0.5 * adap_adj + 0.5 * static_adj
        else:
            adj = adap_adj

        # WaveNet blocks with graph convolution
        skip_sum = 0
        for tc, gc, rc, sc, bn in zip(
            self.temporal_convs, self.graph_convs,
            self.residual_convs, self.skip_convs, self.bn_layers
        ):
            residual = x
            # Temporal: dilated causal conv with gated activation
            h = tc(x)  # (B*S, channels, W)
            h = self.dropout(h)

            # Spatial: graph conv at each timestep
            # Reshape: (B*S, channels, W) → (B, S, channels, W)
            h_spatial = h.reshape(B, S, self.channels, W)
            # For each timestep, apply graph conv
            h_graph_list = []
            for t_idx in range(W):
                h_t = h_spatial[:, :, :, t_idx]  # (B, S, channels)
                h_t = gc(h_t, adj)  # (B, S, channels)
                h_graph_list.append(h_t)
            h_graph = torch.stack(h_graph_list, dim=-1)  # (B, S, channels, W)
            h = h_graph.reshape(B * S, self.channels, W)

            # Skip connection
            skip_sum = skip_sum + sc(h)

            # Residual connection
            h = rc(h) + residual
            h = bn(h)
            x = h

        # Aggregate temporal: take the last timestep from skip connections
        out = F.relu(skip_sum[:, :, -1])  # (B*S, channels)
        out = self.dropout(F.relu(self.output_fc1(out)))

        # ZINB parameters
        pi = torch.sigmoid(self.pi_head(out))       # (B*S, C)
        mu = F.softplus(self.mu_head(out))            # (B*S, C)
        r = F.softplus(self.r_head(out)) + self.r_floor  # (B*S, C)

        # Reshape back to (B, S, C)
        pi = pi.reshape(B, S, self.num_categories)
        mu = mu.reshape(B, S, self.num_categories)
        r = r.reshape(B, S, self.num_categories)

        return pi, mu, r


# ============================================================================
# Baseline 4: STZINB-GNN (Zhuang et al.)
# ============================================================================

class STZINBGNNModel(nn.Module):
    """Spatiotemporal GNN with ZINB output head (Zhuang et al. 2022 baseline)."""
    
    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        hidden_dim: int = 64,
        num_categories: int = 3,
        r_floor: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.num_categories = num_categories
        self.r_floor = r_floor
        
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.spatial_conv = nn.Linear(hidden_dim, hidden_dim)
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        self.pi_head = nn.Linear(hidden_dim, num_categories)
        self.mu_head = nn.Linear(hidden_dim, num_categories)
        self.r_head = nn.Linear(hidden_dim, num_categories)
        self._init_output_heads()

    def _init_output_heads(self) -> None:
        for head in [self.pi_head, self.mu_head, self.r_head]:
            nn.init.normal_(head.weight, 0.0, 0.01)
            nn.init.zeros_(head.bias)

    def forward(
        self, counts: Tensor, features: Tensor, static_adj: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        B, S, W, _ = counts.shape
        x = torch.cat([counts, features], dim=-1)  # (B, S, W, C+F)
        
        x = self.fc_in(x)  # (B, S, W, H)
        
        adj_norm = static_adj + torch.eye(S, device=static_adj.device)
        d = adj_norm.sum(1, keepdim=True)
        adj_norm = adj_norm / d
        
        x_sp = torch.einsum('ij,bsjh->bsih', adj_norm, x)
        x_sp = F.relu(self.spatial_conv(x_sp))
        
        x_flat = x_sp.reshape(B * S, W, -1)
        lstm_out, _ = self.lstm(x_flat)
        h_last = lstm_out[:, -1, :]  # (B*S, H)
        
        pi = torch.sigmoid(self.pi_head(h_last))
        mu = F.softplus(self.mu_head(h_last))
        r = F.softplus(self.r_head(h_last)) + self.r_floor
        
        pi = pi.reshape(B, S, self.num_categories)
        mu = mu.reshape(B, S, self.num_categories)
        r = r.reshape(B, S, self.num_categories)
        return pi, mu, r


# ============================================================================
# Training and Evaluation Infrastructure
# ============================================================================

def train_model(
    model: nn.Module,
    train_ds: CrimeWindowDataset,
    val_ds: CrimeWindowDataset,
    loss_fn,
    model_name: str,
    device: torch.device,
    lr: float = 1e-3,
    epochs: int = 50,
    patience: int = 10,
    batch_size: int = 4,
    is_graph_model: bool = False,
) -> nn.Module:
    """Train a deep baseline with validation-based early stopping.

    Args:
        model: The PyTorch model to train.
        train_ds: Training CrimeWindowDataset.
        val_ds: Validation CrimeWindowDataset.
        loss_fn: Callable (y, pi, mu, r) → scalar loss, or (y, mu, r) for NB.
        model_name: Name for logging.
        device: CUDA or CPU device.
        lr: Learning rate.
        epochs: Maximum number of training epochs.
        patience: Early stopping patience (epochs without improvement).
        batch_size: DataLoader batch size.
        is_graph_model: If True, use graph collation (preserves spatial dim).

    Returns:
        Trained model with best validation weights restored.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    collate = _collate_graph if is_graph_model else _collate_flat
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    logger.info(f"Training {model_name} | epochs={epochs}, lr={lr}, device={device}")
    logger.info(f"  Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss_sum = 0.0
        train_steps = 0

        for batch in train_loader:
            ic = batch["input_counts"].to(device)
            iff = batch["input_features"].to(device)
            tc = batch["target_counts"].to(device)

            optimizer.zero_grad()

            if is_graph_model:
                # GraphWaveNet: (B, S, W, C/F) → (B, S, C) outputs
                pi, mu, r = model(ic, iff)
                loss = loss_fn(tc, pi, mu, r)
            else:
                # LSTM / TFT: (B*S, W, C/F) → (B*S, C) outputs
                pi, mu, r = model(ic, iff)
                if model_name == "LSTM_NB":
                    loss = nb_nll_loss(tc, mu, r)
                else:
                    loss = loss_fn(tc, pi, mu, r)

            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss_sum += loss.item()
            train_steps += 1

        scheduler.step()
        avg_train_loss = train_loss_sum / max(train_steps, 1)

        # --- Validation ---
        model.eval()
        val_loss_sum = 0.0
        val_steps = 0

        with torch.no_grad():
            for batch in val_loader:
                ic = batch["input_counts"].to(device)
                iff = batch["input_features"].to(device)
                tc = batch["target_counts"].to(device)

                if is_graph_model:
                    pi, mu, r = model(ic, iff)
                    # Always validate on CRPS for fair comparison
                    val_crps = crps_zinb(tc, pi, mu, r).mean()
                else:
                    pi, mu, r = model(ic, iff)
                    val_crps = crps_zinb(tc, pi, mu, r).mean()

                if torch.isfinite(val_crps):
                    val_loss_sum += val_crps.item()
                val_steps += 1

        avg_val_loss = val_loss_sum / max(val_steps, 1)

        # --- Early stopping ---
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = " ★"
        else:
            patience_counter += 1
            marker = ""

        if (epoch + 1) % 5 == 0 or epoch == 0 or marker:
            logger.info(
                f"  [{model_name}] Epoch {epoch+1:3d}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val CRPS: {avg_val_loss:.4f}{marker}"
            )

        if patience_counter >= patience:
            logger.info(f"  [{model_name}] Early stopping at epoch {epoch+1} (patience={patience})")
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"  [{model_name}] Restored best weights (Val CRPS = {best_val_loss:.4f})")
    model = model.to(device)
    return model


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_ds: CrimeWindowDataset,
    model_name: str,
    device: torch.device,
    batch_size: int = 4,
    is_graph_model: bool = False,
) -> dict[str, float]:
    """Evaluate a trained model on the test set using CIVIC-SAFE metrics.

    Collects all predictions and targets, then computes:
      - CRPS (primary metric)
      - MAE, RMSE (point forecast accuracy)
      - Brier score (zero-inflation calibration)

    Args:
        model: Trained model.
        test_ds: Test CrimeWindowDataset.
        model_name: Name for logging.
        device: Device.
        batch_size: DataLoader batch size.
        is_graph_model: Whether to use graph collation.

    Returns:
        Dictionary of metric name → value.
    """
    model.eval()
    collate = _collate_graph if is_graph_model else _collate_flat
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    all_y, all_pi, all_mu, all_r = [], [], [], []

    for batch in test_loader:
        ic = batch["input_counts"].to(device)
        iff = batch["input_features"].to(device)
        tc = batch["target_counts"].to(device)

        if is_graph_model:
            pi, mu, r = model(ic, iff)
            # Flatten (B, S, C) → (B*S, C) for metric computation
            B, S, C = pi.shape
            pi = pi.reshape(-1, C)
            mu = mu.reshape(-1, C)
            r = r.reshape(-1, C)
            tc = tc.reshape(-1, C)
        else:
            pi, mu, r = model(ic, iff)

        all_y.append(tc.cpu())
        all_pi.append(pi.cpu())
        all_mu.append(mu.cpu())
        all_r.append(r.cpu())

    y = torch.cat(all_y, dim=0)
    pi = torch.cat(all_pi, dim=0)
    mu = torch.cat(all_mu, dim=0)
    r = torch.cat(all_r, dim=0)

    metrics = compute_all_metrics(y, pi, mu, r)
    logger.info(f"  [{model_name}] Test metrics: {metrics}")
    return metrics


def get_adjacency_matrix(edge_index: Tensor, num_nodes: int) -> Tensor:
    """Convert edge_index (COO) to row-normalised dense adjacency matrix.

    Args:
        edge_index: (2, E) tensor of edge source/target indices.
        num_nodes: Number of nodes in the graph.

    Returns:
        (num_nodes, num_nodes) row-normalised adjacency matrix.
    """
    adj = torch.zeros((num_nodes, num_nodes))
    adj[edge_index[0], edge_index[1]] = 1.0
    # Add self-loops
    adj = adj + torch.eye(num_nodes)
    # Row-normalise
    deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
    return adj / deg


# ============================================================================
# Main Entry Point
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deep Learning Baselines for CIVIC-SAFE"
    )
    parser.add_argument("args", nargs="*", help="Override configs, e.g. data=nyc")
    parsed = parser.parse_args()

    # Parse data=<city> argument (same convention as baselines.py)
    data_name = "chicago"
    for arg in parsed.args:
        if arg.startswith("data="):
            data_name = arg.split("=", 1)[1]

    project_root = Path(__file__).resolve().parent.parent
    panel_path = project_root / "data" / "processed" / f"{data_name}_panel.pt"
    graph_path = project_root / "data" / "processed" / f"{data_name}_graph.pt"

    if not panel_path.exists():
        logger.error(
            f"Panel data not found: {panel_path}. "
            f"Run `python scripts/fetch_data.py` first."
        )
        sys.exit(1)

    # --- Reproducibility ---
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True

    # --- Device selection ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")

    # --- Load data ---
    logger.info(f"Loading {data_name} dataset from {panel_path}...")
    panel = torch.load(panel_path, weights_only=False)
    counts = panel["counts"]   # (S, T, C)
    features = panel["features"]  # (S, T, F)

    S, T, C = counts.shape
    F_dim = features.shape[-1]
    logger.info(f"  Shape: S={S} spatial units, T={T} weeks, C={C} categories, F={F_dim} features")

    # --- Normalise features (training period only — no data leakage) ---
    norm_stats_path = project_root / "data" / "processed" / f"{data_name}_norm_stats.pt"
    if norm_stats_path.exists():
        norm_stats = torch.load(norm_stats_path, weights_only=False)
        feat_mean = norm_stats["mean"]
        feat_std = norm_stats["std"]
        logger.info("  Loaded normalisation stats from training period")
    else:
        train_end_idx = 208  # 4 years × 52 weeks
        train_features = features[:, :train_end_idx, :]
        feat_mean = train_features.mean(dim=(0, 1), keepdim=True)
        feat_std = train_features.std(dim=(0, 1), keepdim=True).clamp(min=1e-6)
        logger.info(f"  Computed normalisation from training period (first {train_end_idx} weeks)")
    features = (features - feat_mean) / feat_std

    # --- Chronological splits (identical to main pipeline) ---
    logger.info("Creating chronological splits...")
    splits = create_chronological_splits(
        counts, features,
        start_year=2018, end_year=2023,
        val_year=2022, test_year=2023,
        window_size=52,
    )

    # --- Load graph for Graph WaveNet ---
    if graph_path.exists():
        graph = torch.load(graph_path, weights_only=False)
        adj = get_adjacency_matrix(graph["queen"], S)
        logger.info(f"  Loaded graph adjacency ({adj.shape})")
    else:
        logger.warning("Graph not found — using identity adjacency for GraphWaveNet.")
        adj = torch.eye(S)

    # --- Loss functions ---
    def crps_loss_fn(y: Tensor, pi: Tensor, mu: Tensor, r: Tensor) -> Tensor:
        """CRPS loss for ZINB outputs (same loss used by CIVIC-SAFE)."""
        return crps_zinb(y, pi, mu, r).mean()

    results: dict[str, dict[str, float]] = {}

    # =====================================================================
    # Baseline 1: LSTM with NB output head
    # =====================================================================
    logger.info("=" * 70)
    logger.info("Baseline 1: LSTM with Negative Binomial Output Head")
    logger.info("=" * 70)

    lstm_model = LSTMNBModel(
        input_dim=C + F_dim,
        hidden_dim=64,
        num_layers=2,
        num_categories=C,
        dropout=0.1,
    )
    total_params = sum(p.numel() for p in lstm_model.parameters())
    logger.info(f"  Parameters: {total_params:,}")

    t0 = time.time()
    lstm_model = train_model(
        lstm_model, splits["train"], splits["val"],
        loss_fn=nb_nll_loss,  # NB NLL (not used directly — handled inside train_model)
        model_name="LSTM_NB",
        device=device,
        lr=1e-3, epochs=50, patience=10,
    )
    lstm_time = time.time() - t0

    lstm_metrics = evaluate_model(
        lstm_model, splits["test"], "LSTM_NB", device
    )
    lstm_metrics["train_time_s"] = round(lstm_time, 1)
    results["LSTM_NB"] = lstm_metrics

    # =====================================================================
    # Baseline 2: Simplified TFT
    # =====================================================================
    logger.info("=" * 70)
    logger.info("Baseline 2: Simplified Temporal Fusion Transformer")
    logger.info("=" * 70)

    tft_model = SimplifiedTFTModel(
        count_dim=C,
        feature_dim=F_dim,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.1,
    )
    total_params = sum(p.numel() for p in tft_model.parameters())
    logger.info(f"  Parameters: {total_params:,}")

    t0 = time.time()
    tft_model = train_model(
        tft_model, splits["train"], splits["val"],
        loss_fn=crps_loss_fn,  # CRPS loss — same as CIVIC-SAFE
        model_name="TFT_ZINB",
        device=device,
        lr=1e-3, epochs=50, patience=10,
    )
    tft_time = time.time() - t0

    tft_metrics = evaluate_model(
        tft_model, splits["test"], "TFT_ZINB", device
    )
    tft_metrics["train_time_s"] = round(tft_time, 1)
    results["TFT_ZINB"] = tft_metrics

    # =====================================================================
    # Baseline 3: Graph WaveNet
    # =====================================================================
    logger.info("=" * 70)
    logger.info("Baseline 3: Graph WaveNet")
    logger.info("=" * 70)

    gwnet_model = GraphWaveNetModel(
        num_nodes=S,
        input_dim=C + F_dim,
        channels=32,
        num_layers=4,
        kernel_size=2,
        num_categories=C,
        embed_dim=16,
        dropout=0.1,
    )
    total_params = sum(p.numel() for p in gwnet_model.parameters())
    logger.info(f"  Parameters: {total_params:,}")

    # GraphWaveNet needs static adjacency passed through the forward call
    adj_device = adj.to(device)
    original_forward = gwnet_model.forward

    def gwnet_forward_with_adj(counts: Tensor, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return original_forward(counts, features, static_adj=adj_device)

    gwnet_model.forward = gwnet_forward_with_adj  # type: ignore[assignment]

    t0 = time.time()
    gwnet_model = train_model(
        gwnet_model, splits["train"], splits["val"],
        loss_fn=crps_loss_fn,
        model_name="GraphWaveNet",
        device=device,
        lr=1e-3, epochs=50, patience=10,
        batch_size=2,  # Smaller batch — graph model uses more memory
        is_graph_model=True,
    )
    gwnet_time = time.time() - t0

    gwnet_metrics = evaluate_model(
        gwnet_model, splits["test"], "GraphWaveNet", device,
        batch_size=2, is_graph_model=True,
    )
    gwnet_metrics["train_time_s"] = round(gwnet_time, 1)
    results["GraphWaveNet"] = gwnet_metrics

    # =====================================================================
    # Baseline 4: STZINB-GNN
    # =====================================================================
    logger.info("=" * 70)
    logger.info("Baseline 4: STZINB-GNN (Zhuang et al. 2022)")
    logger.info("=" * 70)

    stzinb_model = STZINBGNNModel(
        num_nodes=S,
        input_dim=C + F_dim,
        hidden_dim=64,
        num_categories=C,
    )
    total_params = sum(p.numel() for p in stzinb_model.parameters())
    logger.info(f"  Parameters: {total_params:,}")

    stzinb_original_forward = stzinb_model.forward

    def stzinb_forward_with_adj(counts: Tensor, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return stzinb_original_forward(counts, features, static_adj=adj_device)

    stzinb_model.forward = stzinb_forward_with_adj  # type: ignore[assignment]

    t0 = time.time()
    stzinb_model = train_model(
        stzinb_model, splits["train"], splits["val"],
        loss_fn=crps_loss_fn,
        model_name="STZINB_GNN",
        device=device,
        lr=1e-3, epochs=50, patience=10,
        batch_size=2,
        is_graph_model=True,
    )
    stzinb_time = time.time() - t0

    stzinb_metrics = evaluate_model(
        stzinb_model, splits["test"], "STZINB_GNN", device,
        batch_size=2, is_graph_model=True,
    )
    stzinb_metrics["train_time_s"] = round(stzinb_time, 1)
    results["STZINB_GNN"] = stzinb_metrics

    # =====================================================================
    # Save results and print comparison table
    # =====================================================================
    output_dir = project_root / "outputs" / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{data_name}_deep_baselines.json"

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_file}")

    # --- Pretty-print comparison table ---
    print("\n" + "=" * 78)
    print(f"  DEEP BASELINE RESULTS — {data_name.upper()} — TEST SET (2023)")
    print("=" * 78)
    header = f"{'Model':<20} {'CRPS':>10} {'MAE':>10} {'RMSE':>10} {'Brier':>10} {'Time(s)':>10}"
    print(header)
    print("-" * 78)
    for name, m in results.items():
        row = (
            f"{name:<20} "
            f"{m.get('crps', float('nan')):>10.4f} "
            f"{m.get('mae', float('nan')):>10.4f} "
            f"{m.get('rmse', float('nan')):>10.4f} "
            f"{m.get('brier_zero', float('nan')):>10.4f} "
            f"{m.get('train_time_s', 0):>10.1f}"
        )
        print(row)
    print("=" * 78)
    print(f"\n  Lower is better for ALL metrics. Primary metric: CRPS.")
    print(f"  Results saved to: {out_file}\n")


if __name__ == "__main__":
    main()
