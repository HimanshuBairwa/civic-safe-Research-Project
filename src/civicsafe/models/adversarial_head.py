"""Adversarial head for demographic invariance via Gradient Reversal.

This module implements a Gradient Reversal Layer (GRL) and an adversarial
discriminator. The goal is to force the main spatiotemporal encoder to learn
representations that are invariant to protected demographic attributes
(e.g., income, race), mitigating proxy bias.

During the forward pass, the GRL acts as an identity function.
During the backward pass, the GRL multiplies the gradient by a negative scalar
(-lambda). This sets up a minimax game where the discriminator tries to
predict the demographic group from the representation, while the encoder
tries to fool the discriminator.

References:
  - Ganin et al. (2015): "Unsupervised Domain Adaptation by Backpropagation"
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


class GradientReversalFunction(torch.autograd.Function):
    """Autograd function that reverses gradients in the backward pass."""

    @staticmethod
    def forward(ctx: Any, x: Tensor, lambda_: float) -> Tensor:
        """Forward pass is an identity operation.
        
        Args:
            ctx: Autograd context.
            x: Input tensor.
            lambda_: Scaling factor for the reversed gradient.
            
        Returns:
            The unchanged input tensor.
        """
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor | None, None]:
        """Backward pass negates and scales the gradient.
        
        Args:
            ctx: Autograd context.
            grad_output: Upstream gradient.
            
        Returns:
            Tuple of (reversed_gradient, None) corresponding to the forward args.
        """
        lambda_ = ctx.lambda_
        grad_input = grad_output.clone() * -lambda_
        return grad_input, None


class GradientReversalLayer(nn.Module):
    """Module wrapper for Gradient Reversal Function.
    
    Args:
        lambda_: Initial scalar for the reversed gradient. Usually ramped up
                 over training to stabilize the early stages.
    """

    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: Tensor) -> Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)  # type: ignore[no-any-return]


class AdversarialDiscriminator(nn.Module):
    """Adversarial MLP to predict demographic groups from latent representations.
    
    Includes the Gradient Reversal Layer at the input so that gradients flowing
    back to the encoder are reversed, encouraging demographic invariance.
    
    Args:
        in_features: Dimension of the latent representation.
        hidden_dim: Hidden layer dimension.
        num_classes: Number of demographic groups to discriminate.
        lambda_: Gradient reversal scaling factor.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_classes: int,
        lambda_: float = 1.0,
    ) -> None:
        super().__init__()
        
        self.grl = GradientReversalLayer(lambda_=lambda_)
        
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Predict demographic logits from the latent state.
        
        Args:
            x: Latent representation. Shape: (B, S, F) or (S, F).
               Typically applied only to the final timestep.
               
        Returns:
            Logits for demographic classes. Shape: (..., num_classes).
        """
        x_rev = self.grl(x)
        logits = self.mlp(x_rev)
        return logits
