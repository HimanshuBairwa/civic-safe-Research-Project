"""
Tests for civicsafe.utils.seeding — determinism, cross-framework sync,
state round-trip, and input validation.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from civicsafe.utils.seeding import (
    get_seed_state,
    seed_everything,
    set_seed_state,
)

# ---------------------------------------------------------------------------
# Determinism — same seed yields identical draws
# ---------------------------------------------------------------------------


def test_seed_deterministic_torch(default_seed: int) -> None:
    """Same seed must produce identical torch.randn sequences."""
    seed_everything(default_seed)
    torch_draw_first = torch.randn(100)

    seed_everything(default_seed)
    torch_draw_second = torch.randn(100)

    assert torch.allclose(
        torch_draw_first, torch_draw_second
    ), "torch.randn produced different values for the same seed"


def test_seed_deterministic_numpy(default_seed: int) -> None:
    """Same seed must produce identical numpy.random.randn sequences."""
    seed_everything(default_seed)
    numpy_draw_first = np.random.randn(100)

    seed_everything(default_seed)
    numpy_draw_second = np.random.randn(100)

    np.testing.assert_array_equal(
        numpy_draw_first,
        numpy_draw_second,
        err_msg="numpy.random.randn produced different values for the same seed",
    )


def test_seed_deterministic_python(default_seed: int) -> None:
    """Same seed must produce identical random.random() sequences."""
    seed_everything(default_seed)
    python_draw_first = [random.random() for _ in range(100)]

    seed_everything(default_seed)
    python_draw_second = [random.random() for _ in range(100)]

    assert (
        python_draw_first == python_draw_second
    ), "random.random() produced different values for the same seed"


# ---------------------------------------------------------------------------
# Cross-framework synchronisation
# ---------------------------------------------------------------------------


def test_seed_cross_framework_sync(default_seed: int) -> None:
    """After seed_everything(42), the first draws from torch AND numpy are
    individually reproducible across invocations (though they differ from
    each other because the generators are independent).
    """
    seed_everything(default_seed)
    torch_first_value = torch.randn(1).item()
    numpy_first_value = np.random.randn(1).item()

    seed_everything(default_seed)
    torch_first_value_again = torch.randn(1).item()
    numpy_first_value_again = np.random.randn(1).item()

    assert torch_first_value == pytest.approx(
        torch_first_value_again
    ), "Torch first draw is not reproducible"
    assert numpy_first_value == pytest.approx(
        numpy_first_value_again
    ), "Numpy first draw is not reproducible"
    # Torch and numpy generators are independent — draws will almost
    # certainly differ.
    assert torch_first_value != pytest.approx(
        numpy_first_value, abs=1e-6
    ), "Torch and numpy first draws are unexpectedly identical"


# ---------------------------------------------------------------------------
# Different seeds produce different sequences
# ---------------------------------------------------------------------------


def test_different_seeds_differ() -> None:
    """Seeds 42 and 43 must produce divergent torch.randn sequences."""
    seed_everything(42)
    torch_draw_seed42 = torch.randn(100)

    seed_everything(43)
    torch_draw_seed43 = torch.randn(100)

    assert not torch.allclose(
        torch_draw_seed42, torch_draw_seed43
    ), "Different seeds produced identical torch sequences"


# ---------------------------------------------------------------------------
# State round-trip — save / restore RNG state
# ---------------------------------------------------------------------------


def test_seed_state_roundtrip(default_seed: int) -> None:
    """get_seed_state → mutate generators → set_seed_state → verify restored."""
    seed_everything(default_seed)

    # Capture state *before* any draws
    saved_state = get_seed_state()

    # Consume random numbers to advance every generator
    _ = torch.randn(50)
    _ = np.random.randn(50)
    _ = [random.random() for _ in range(50)]

    # Record the *next* values after advancing
    torch_after_advance = torch.randn(10)
    numpy_after_advance = np.random.randn(10)
    python_after_advance = [random.random() for _ in range(10)]

    # Restore state to the saved snapshot (before the 50 draws)
    set_seed_state(saved_state)

    # Re-consume the same 50 values
    _ = torch.randn(50)
    _ = np.random.randn(50)
    _ = [random.random() for _ in range(50)]

    # The *next* values must match what we recorded earlier
    torch_after_restore = torch.randn(10)
    numpy_after_restore = np.random.randn(10)
    python_after_restore = [random.random() for _ in range(10)]

    assert torch.allclose(
        torch_after_advance, torch_after_restore
    ), "Torch state was not correctly restored"
    np.testing.assert_array_equal(
        numpy_after_advance,
        numpy_after_restore,
        err_msg="Numpy state was not correctly restored",
    )
    assert (
        python_after_advance == python_after_restore
    ), "Python random state was not correctly restored"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_seed_negative_raises() -> None:
    """seed_everything(-1) must raise AssertionError."""
    with pytest.raises(AssertionError):
        seed_everything(-1)
