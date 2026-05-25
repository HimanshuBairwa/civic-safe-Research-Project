"""Numerical safety utilities for tensor operations.

Every function preserves the input tensor's dtype and applies explicit
epsilon floors to avoid NaN/Inf propagation.
"""

from typing import Final

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------

NUMERICAL_EPS: Final[float] = 1e-8
"""General-purpose epsilon for denominator floors. Range: (0, 1e-4]."""

LOG_FLOOR: Final[float] = 1e-38
"""Minimum clamp value before torch.log to avoid -inf. Range: (0, 1e-30]."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def safe_log(tensor: Tensor, eps: float = LOG_FLOOR) -> Tensor:
    """Element-wise log with a lower clamp to prevent -inf.

    Args:
        tensor: Input tensor of any shape.  # (*,)
        eps: Floor value applied before log. Default: :data:`LOG_FLOOR`.

    Returns:
        Tensor of same shape and dtype as *tensor*.
    """
    return torch.log(torch.clamp(tensor, min=eps)).to(tensor.dtype)


def safe_divide(
    numerator: Tensor,
    denominator: Tensor,
    eps: float = NUMERICAL_EPS,
) -> Tensor:
    """Element-wise division with epsilon floor on |denominator|.

    The sign of the denominator is preserved; only its absolute value
    is floored to *eps*.

    Args:
        numerator: Dividend tensor.   # (*,)
        denominator: Divisor tensor.   # (*,) — broadcastable with numerator
        eps: Minimum absolute value for denominator. Default: :data:`NUMERICAL_EPS`.

    Returns:
        Tensor of same shape and dtype as *numerator*.
    """
    safe_denom: Tensor = torch.sign(denominator) * torch.clamp(  # (*,)
        torch.abs(denominator), min=eps
    )
    return (numerator / safe_denom).to(numerator.dtype)


def log_sum_exp(tensor: Tensor, dim: int) -> Tensor:
    """Numerically stable log-sum-exp via the max-subtract trick.

    Args:
        tensor: Input tensor.  # (..., D, ...) where D is the reduction dim
        dim: Dimension along which to reduce.

    Returns:
        Tensor with *dim* squeezed out.  # (..., ...)
    """
    max_val: Tensor = tensor.max(dim, keepdim=True).values  # (..., 1, ...)
    shifted_exp: Tensor = torch.exp(tensor - max_val)  # (..., D, ...)
    sum_exp: Tensor = torch.sum(shifted_exp, dim=dim)  # (..., ...)
    result: Tensor = max_val.squeeze(dim) + torch.log(sum_exp)  # (..., ...)
    return result


def clamp_probabilities(
    tensor: Tensor,
    eps: float = NUMERICAL_EPS,
) -> Tensor:
    """Clamp values into the open interval (eps, 1 − eps).

    Useful before operations like log on probability tensors.

    Args:
        tensor: Probability tensor.  # (*,)
        eps: Margin from 0 and 1. Default: :data:`NUMERICAL_EPS`.

    Returns:
        Clamped tensor of same shape and dtype.
    """
    return torch.clamp(tensor, min=eps, max=1.0 - eps).to(tensor.dtype)
