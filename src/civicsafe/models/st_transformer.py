"""Spatiotemporal Graph Transformer with structured attention masks.

Fixes the critical architectural weakness in the sequential
Spatial→Temporal pipeline: the standard approach processes each
spatial unit's time series independently in the temporal encoder,
meaning NO cross-spatial information flows during temporal attention.

This module implements a unified Graph Transformer where each token
represents a (node, timestep) pair, and a structured attention mask
control which tokens can attend to which:
  - Temporal self-attention: each node attends to its own past (causal)
  - Spatial cross-attention: each node attends to graph neighbors at
    the same timestep
  - NO future leakage: strict causal masking across time

Complexity: O(S²T + ST²) per layer (sparse attention), vs O(S²T²)
for dense attention. Feasible for S=77, T=52.

References:
  - Ying et al. (2021): "Do Transformers Really Perform Bad for Graph
    Representation?" (Graphormer)
  - Xu et al. (2020): "Spatial-Temporal Transformer Networks" (STTN)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SpatiotemporalPositionalEncoding(nn.Module):
    """Combined spatial + temporal positional encoding.
    
    Spatial PE: learnable embeddings per node ID (since graph is fixed).
    Temporal PE: sinusoidal encoding (generalizes to unseen lengths).
    Combined via addition: PE(s,t) = PE_spatial(s) + PE_temporal(t).
    """
    
    def __init__(
        self,
        d_model: int,
        max_nodes: int = 100,
        max_time: int = 52,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        # Learnable spatial PE (each node gets a unique embedding)
        self.spatial_pe = nn.Embedding(max_nodes, d_model)
        
        # Fixed sinusoidal temporal PE
        pe = torch.zeros(max_time, d_model)
        position = torch.arange(0, max_time, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("temporal_pe", pe)  # (max_time, d_model)
    
    def forward(self, x: Tensor, node_ids: Tensor, timesteps: Tensor) -> Tensor:
        """Add combined positional encoding.
        
        Args:
            x: Token embeddings. Shape: (B, L, D) where L = S*T
            node_ids: Node ID for each token. Shape: (B, L)
            timesteps: Timestep for each token. Shape: (B, L)
        
        Returns:
            x + PE, same shape.
        """
        spatial = self.spatial_pe(node_ids)  # (B, L, D)
        temporal = self.temporal_pe[timesteps]  # (B, L, D)
        return self.dropout(x + spatial + temporal)


def build_st_attention_mask(
    num_nodes: int,
    num_timesteps: int,
    edge_index: Tensor,
    device: torch.device,
) -> Tensor:
    """Build structured spatiotemporal attention mask.
    
    Token ordering: flatten (S, T) → (S*T,) in row-major order.
    Token index for node s at time t: s * T + t
    
    Mask rules:
      1. Same node, past/current time: ATTEND (temporal self-attention)
      2. Neighbor node, same time: ATTEND (spatial cross-attention)  
      3. All other pairs: BLOCK
      4. Future time for any node: BLOCK (causal constraint)
    
    Args:
        num_nodes: S = number of spatial units
        num_timesteps: T = number of timesteps  
        edge_index: Graph edges. Shape: (2, E)
        device: Target device
    
    Returns:
        Attention mask. Shape: (S*T, S*T)
        True = BLOCK, False = ATTEND (PyTorch convention for additive masks)
    """
    L = num_nodes * num_timesteps
    # Start with everything blocked
    mask = torch.ones(L, L, dtype=torch.bool, device=device)
    
    # Rule 1: Same node, causal temporal attention
    for s in range(num_nodes):
        for t1 in range(num_timesteps):
            for t2 in range(t1 + 1):  # t2 <= t1 (causal)
                idx1 = s * num_timesteps + t1
                idx2 = s * num_timesteps + t2
                mask[idx1, idx2] = False
    
    # Rule 2: Neighbor nodes, same timestep
    src_nodes = edge_index[0].tolist()
    dst_nodes = edge_index[1].tolist()
    for src, dst in zip(src_nodes, dst_nodes):
        for t in range(num_timesteps):
            idx_src = src * num_timesteps + t
            idx_dst = dst * num_timesteps + t
            mask[idx_src, idx_dst] = False  # src can attend to dst at same time
    
    return mask


class SpatiotemporalTransformerLayer(nn.Module):
    """Single layer of the Spatiotemporal Graph Transformer.
    
    Uses Pre-LN architecture for training stability.
    """
    
    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        """Forward with structured attention mask.
        
        Args:
            x: Token sequence. Shape: (B, L, D)
            attn_mask: Boolean mask. Shape: (L, L)
        
        Returns:
            Updated tokens. Shape: (B, L, D)
        """
        # Pre-LN self-attention with structured mask
        x_norm = self.norm1(x)
        # Convert boolean mask to float mask for MultiheadAttention
        float_mask = attn_mask.float().masked_fill(attn_mask, float('-inf'))
        attn_out, _ = self.self_attn(
            x_norm, x_norm, x_norm,
            attn_mask=float_mask,
        )
        x = x + self.dropout(attn_out)
        
        # Pre-LN FFN
        x = x + self.ffn(self.norm2(x))
        
        return x


class SpatiotemporalTransformer(nn.Module):
    """Complete Spatiotemporal Graph Transformer encoder.
    
    Replaces the sequential SpatialEncoder → TemporalEncoder pipeline
    with a unified transformer that captures joint space-time dependencies.
    
    Args:
        d_model: Model dimension.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        dim_feedforward: FFN intermediate dimension.
        max_nodes: Maximum number of spatial nodes.
        max_time: Maximum sequence length.
        dropout: Dropout rate.
    """
    
    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dim_feedforward: int = 512,
        max_nodes: int = 100,
        max_time: int = 52,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        
        self.pos_encoder = SpatiotemporalPositionalEncoding(
            d_model=d_model,
            max_nodes=max_nodes,
            max_time=max_time,
            dropout=dropout,
        )
        
        self.layers = nn.ModuleList([
            SpatiotemporalTransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        self._cached_mask: Tensor | None = None
        self._cached_mask_key: tuple[int, int, str] | None = None
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """Forward pass through the spatiotemporal transformer.
        
        Args:
            x: Node features over time. Shape: (S, T, D)
            edge_index: Graph edges (union of queen + knn). Shape: (2, E)
        
        Returns:
            Encoded features. Shape: (S, T, D)
        """
        S, T, D = x.shape
        
        # Flatten to token sequence: (1, S*T, D)
        x_flat = x.reshape(1, S * T, D)
        
        # Build node_ids and timesteps for PE
        node_ids = torch.arange(S, device=x.device).unsqueeze(1).expand(S, T).reshape(1, S * T)
        timesteps = torch.arange(T, device=x.device).unsqueeze(0).expand(S, T).reshape(1, S * T)
        
        # Add positional encodings
        x_flat = self.pos_encoder(x_flat, node_ids, timesteps)
        
        # Build or retrieve cached attention mask
        mask_key = (S, T, str(x.device))
        if self._cached_mask_key != mask_key:
            self._cached_mask = build_st_attention_mask(S, T, edge_index, x.device)
            self._cached_mask_key = mask_key
        
        # Apply transformer layers
        for layer in self.layers:
            x_flat = layer(x_flat, self._cached_mask)
        
        x_flat = self.final_norm(x_flat)
        
        # Reshape back: (1, S*T, D) → (S, T, D)
        return x_flat.reshape(S, T, D)
