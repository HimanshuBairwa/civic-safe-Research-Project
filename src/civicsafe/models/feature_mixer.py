"""Multi-Factor Feature Mixer (MFFM) with diversity regularization.

Soft-attention module that decomposes the fused spatiotemporal embedding
into interpretable factor heads. Includes a Jensen-Shannon Divergence
collapse penalty to prevent degenerate solutions where all heads
learn identical attention patterns.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FeatureMixer(nn.Module):
    """Multi-Factor Feature Mixer with diversity regularization.

    Args:
        d_model: Input/output embedding dimension.
        num_heads: Number of soft-attention factor heads.
        temperature: Softmax temperature (lower = sharper attention).
        collapse_threshold: Minimum JSD between any two heads before
            the diversity penalty activates.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 3,
        temperature: float = 1.0,
        collapse_threshold: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.temperature = temperature
        self.collapse_threshold = collapse_threshold

        # Each head projects to attention logits over d_model features
        self.attention_heads = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(num_heads)]
        )
        # Output projection to combine heads
        self.output_proj = nn.Linear(d_model * num_heads, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Apply multi-factor attention mixing.

        Args:
            x: Input embeddings. Shape: (B, T, D)

        Returns:
            Tuple of:
              - Mixed output. Shape: (B, T, D)
              - Diversity loss scalar (add to total loss for regularization)
        """
        head_outputs = []
        attention_dists = []

        for head in self.attention_heads:
            # Compute attention weights over feature dimensions
            logits = head(x) / self.temperature  # (B, T, D)
            attn = F.softmax(logits, dim=-1)  # (B, T, D)
            attention_dists.append(attn)
            # Apply attention (element-wise weighting)
            head_outputs.append(x * attn)  # (B, T, D)

        # Concatenate and project
        concat = torch.cat(head_outputs, dim=-1)  # (B, T, D*num_heads)
        mixed = self.output_proj(concat)  # (B, T, D)
        mixed = self.norm(mixed + x)  # Residual + LayerNorm

        # Compute diversity loss (JSD between all head pairs)
        div_loss = self._diversity_loss(attention_dists)

        return mixed, div_loss

    def _diversity_loss(self, attention_dists: list[Tensor]) -> Tensor:
        """Jensen-Shannon Divergence penalty for head collapse prevention.

        If any two heads have JSD below the threshold, a penalty term
        encourages them to diversify.

        Args:
            attention_dists: List of (B, T, D) attention weight tensors.

        Returns:
            Scalar penalty term (0 if all heads are diverse enough).
        """
        eps = 1e-8
        penalty = torch.tensor(0.0, device=attention_dists[0].device)

        for i in range(len(attention_dists)):
            for j in range(i + 1, len(attention_dists)):
                p = attention_dists[i].mean(dim=(0, 1))  # (D,)
                q = attention_dists[j].mean(dim=(0, 1))  # (D,)

                # Clamp for log stability
                p = torch.clamp(p, min=eps)
                q = torch.clamp(q, min=eps)

                m = 0.5 * (p + q)
                jsd = 0.5 * (
                    (p * (torch.log(p + eps) - torch.log(m + eps))).sum()
                    + (q * (torch.log(q + eps) - torch.log(m + eps))).sum()
                )

                # Differentiable penalty: activates smoothly when JSD < threshold
                # Uses F.relu instead of Python `if` for torch.compile compatibility
                penalty = penalty + F.relu(self.collapse_threshold - jsd)

        return penalty
