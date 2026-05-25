"""GATv2 spatial encoder with dual adjacency support.

Uses GATv2Conv (Brody, Alon & Yahav, ICLR 2022) which has strictly
more expressive dynamic attention than GATv1, at the same cost.
It is a drop-in replacement for GATConv.

Design:
  - Dual adjacency: Queen contiguity + K-NN edges run through the
    same GAT layers, outputs summed. This lets the model learn
    border-diffusion and socioeconomic-similarity effects simultaneously.
  - LayerNorm after each layer prevents oversmoothing on small graphs (N=77).
  - ELU activation (standard for GAT).
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GATv2Conv


class SpatialEncoder(nn.Module):
    """Multi-layer GATv2 encoder with dual adjacency support.

    Args:
        in_channels: Number of input features per node.
        hidden_channels: Hidden dimension per layer.
        num_layers: Number of GATv2 layers.
        num_heads: Number of attention heads per layer.
        dropout: Dropout on attention coefficients and features.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            # Last layer: concat=False (average heads) → output = hidden_channels
            # Hidden layers: concat=True → output = hidden_channels
            # To keep concat=True, out_channels = hidden_channels // num_heads
            if i < num_layers - 1:
                out_ch = hidden_channels // num_heads
                concat = True
            else:
                out_ch = hidden_channels
                concat = False

            self.convs.append(
                GATv2Conv(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    heads=num_heads,
                    concat=concat,
                    dropout=dropout,
                    add_self_loops=True,
                    share_weights=False,
                )
            )
            # Norm dimension: hidden_channels for all layers
            self.norms.append(nn.LayerNorm(hidden_channels))

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        edge_index_queen: Tensor,
        edge_index_knn: Tensor | None = None,
    ) -> Tensor:
        """Forward pass through the spatial encoder.

        Args:
            x: Node features. Shape: (N, F_in)
            edge_index_queen: Queen contiguity edges. Shape: (2, E_q)
            edge_index_knn: K-NN edges. Shape: (2, E_k) or None.

        Returns:
            Spatial embeddings. Shape: (N, hidden_channels)
        """
        for i in range(self.num_layers):
            # Run GAT on queen adjacency
            h_queen = self.convs[i](x, edge_index_queen)

            # Run GAT on KNN adjacency (if provided)
            if edge_index_knn is not None:
                h_knn = self.convs[i](x, edge_index_knn)
                h = h_queen + h_knn  # Sum dual adjacency outputs
            else:
                h = h_queen

            h = self.norms[i](h)
            h = self.activation(h)
            h = self.dropout(h)
            x = h

        return x
