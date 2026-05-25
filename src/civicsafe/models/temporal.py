"""Causal Transformer temporal encoder for autoregressive forecasting.

Uses nn.TransformerEncoder with is_causal=True (PyTorch 2.0+ native).
This is architecturally identical to a decoder-only transformer but avoids
the unnecessary cross-attention sublayer of nn.TransformerDecoder.

Sinusoidal positional encoding is used because:
  - It generalizes to unseen sequence lengths (important for inference)
  - It adds zero trainable parameters (prevents overfitting on T=52 weeks)
  - It is the recommended baseline per 2024 literature for short sequences

Reference: Vaswani et al. "Attention Is All You Need" (2017)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding.

    Args:
        d_model: Embedding dimension.
        max_len: Maximum sequence length supported.
        dropout: Dropout applied after adding PE.
    """

    def __init__(self, d_model: int, max_len: int = 52, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor. Shape: (B, T, D)

        Returns:
            x + PE, same shape.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TemporalEncoder(nn.Module):
    """Causal Transformer encoder for time-series of spatial embeddings.

    Uses nn.TransformerEncoder with is_causal=True. The causal mask
    is generated automatically by PyTorch, ensuring position t can
    only attend to positions <= t. This mathematically guarantees
    zero future information leakage.

    Args:
        d_model: Model dimension (must match spatial encoder output).
        num_heads: Number of attention heads. Must divide d_model.
        num_layers: Number of transformer layers.
        dim_feedforward: FFN intermediate dimension.
        dropout: Dropout on attention and FFN.
        max_seq_len: Maximum sequence length for positional encoding.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 52,
    ) -> None:
        super().__init__()

        self.pos_encoder = SinusoidalPositionalEncoding(
            d_model=d_model, max_len=max_seq_len, dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with causal masking.

        Args:
            x: Sequence of spatial embeddings. Shape: (B, T, D)

        Returns:
            Temporally-encoded sequence. Shape: (B, T, D)
        """
        x = self.pos_encoder(x)
        # PyTorch 2.1 requires an explicit mask even when is_causal=True
        seq_len = x.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        x = self.transformer(x, mask=mask, is_causal=True)
        return x
