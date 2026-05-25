"""
Tests for civicsafe.utils.numerics — safe_log, safe_divide, log_sum_exp,
clamp_probabilities, and dtype preservation.
"""

from __future__ import annotations

import pytest
import torch

from civicsafe.utils.numerics import (
    clamp_probabilities,
    log_sum_exp,
    safe_divide,
    safe_log,
)

# ---------------------------------------------------------------------------
# safe_log
# ---------------------------------------------------------------------------


def test_safe_log_positive() -> None:
    """safe_log on strictly positive inputs must match natural log."""
    positive_inputs = torch.tensor([1.0, 2.718281828])
    log_values = safe_log(positive_inputs)

    expected = torch.tensor([0.0, 1.0])
    assert torch.allclose(log_values, expected, atol=1e-3), (
        f"safe_log({positive_inputs.tolist()}) = {log_values.tolist()}, "
        f"expected ≈ {expected.tolist()}"
    )


def test_safe_log_zero() -> None:
    """safe_log(0) must return a finite value, not -inf."""
    zero_input = torch.tensor([0.0])
    log_of_zero = safe_log(zero_input)

    assert torch.isfinite(
        log_of_zero
    ).all(), f"safe_log(0) produced non-finite value: {log_of_zero.item()}"


def test_safe_log_negative_clamped() -> None:
    """safe_log on a negative input must return a finite value (clamped)."""
    negative_input = torch.tensor([-1.0])
    log_of_negative = safe_log(negative_input)

    assert torch.isfinite(
        log_of_negative
    ).all(), f"safe_log(-1) produced non-finite value: {log_of_negative.item()}"


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float64],
    ids=["float32", "float64"],
)
def test_safe_log_preserves_dtype(dtype: torch.dtype) -> None:
    """safe_log must preserve the input tensor's floating-point dtype."""
    typed_input = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
    log_output = safe_log(typed_input)

    assert (
        log_output.dtype == dtype
    ), f"Input dtype {dtype} was changed to {log_output.dtype}"


# ---------------------------------------------------------------------------
# safe_divide
# ---------------------------------------------------------------------------


def test_safe_divide_normal() -> None:
    """safe_divide with a non-zero denominator must return the exact quotient."""
    quotient = safe_divide(
        torch.tensor(6.0),
        torch.tensor(3.0),
    )
    assert quotient.item() == pytest.approx(
        2.0
    ), f"safe_divide(6, 3) = {quotient.item()}, expected 2.0"


def test_safe_divide_by_zero() -> None:
    """safe_divide by zero must return a finite value (not inf or nan)."""
    quotient = safe_divide(
        torch.tensor(1.0),
        torch.tensor(0.0),
    )
    assert torch.isfinite(
        quotient
    ).all(), f"safe_divide(1, 0) produced non-finite value: {quotient.item()}"


# ---------------------------------------------------------------------------
# log_sum_exp
# ---------------------------------------------------------------------------


def test_log_sum_exp_correctness() -> None:
    """log_sum_exp must agree with torch.logsumexp on a random tensor."""
    torch.manual_seed(42)
    random_tensor = torch.randn(5, 10)

    custom_lse = log_sum_exp(random_tensor, dim=1)
    reference_lse = torch.logsumexp(random_tensor, dim=1)

    assert torch.allclose(custom_lse, reference_lse, atol=1e-5), (
        f"log_sum_exp deviates from torch.logsumexp:\n"
        f"  custom:    {custom_lse.tolist()}\n"
        f"  reference: {reference_lse.tolist()}"
    )


def test_log_sum_exp_numerical_stability() -> None:
    """log_sum_exp must handle values > 500 without overflow (exp(500) is
    inf in float32, but the log-sum-exp trick avoids this).
    """
    large_values = torch.tensor([500.0, 501.0, 502.0])
    stable_result = log_sum_exp(large_values, dim=0)

    assert torch.isfinite(
        stable_result
    ).all(), f"log_sum_exp overflowed on large inputs: {stable_result.item()}"
    # Reference: log(exp(500) + exp(501) + exp(502))
    # = 502 + log(exp(-2) + exp(-1) + 1) ≈ 502.408
    reference = torch.logsumexp(large_values, dim=0)
    assert torch.allclose(stable_result, reference, atol=1e-4), (
        f"Numerically stable result {stable_result.item()} deviates "
        f"from reference {reference.item()}"
    )


# ---------------------------------------------------------------------------
# clamp_probabilities
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_probabilities",
    [
        torch.tensor([-0.5, 0.0, 0.5, 1.0, 1.5]),
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([1.0, 1.0, 1.0]),
        torch.tensor([-100.0, 200.0]),
    ],
    ids=["mixed", "all_zero", "all_one", "extreme"],
)
def test_clamp_probabilities_range(raw_probabilities: torch.Tensor) -> None:
    """clamp_probabilities must map every element into [eps, 1 - eps]."""
    eps = 1e-6
    clamped = clamp_probabilities(raw_probabilities, eps=eps)

    assert (
        clamped >= eps
    ).all(), f"Values below eps found: {clamped[clamped < eps].tolist()}"
    assert (
        clamped <= 1.0 - eps
    ).all(), f"Values above 1-eps found: {clamped[clamped > 1.0 - eps].tolist()}"
