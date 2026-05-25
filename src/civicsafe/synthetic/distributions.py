"""Synthetic distribution generators with known ground-truth parameters.

Each generator returns both samples and the true parameters used, enabling
exact validation of parameter-recovery tests, coverage checks, and
pipeline integration tests.
"""

from __future__ import annotations

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_SEED: int = 42
MIN_ZERO_INFLATION: float = 0.0
MAX_ZERO_INFLATION: float = 1.0  # exclusive upper bound for pi
MIN_POSITIVE: float = 0.0  # exclusive lower bound for mu and r
DEFAULT_DISTANCE_THRESHOLD: float = 0.3  # geometric-graph edge threshold
FEATURE_SIGNAL_WEIGHT: float = 0.3  # weight of mu-correlated signal in features
FEATURE_NOISE_WEIGHT: float = 0.7  # weight of random noise in features


# ---------------------------------------------------------------------------
# ZINB Sampler
# ---------------------------------------------------------------------------
def generate_zinb_samples(
    num_samples: int,
    pi: float,
    mu: float,
    r: float,
    seed: int = DEFAULT_SEED,
) -> tuple[Tensor, dict[str, float]]:
    """Generate samples from a Zero-Inflated Negative Binomial distribution.

    ZINB model:
        With probability ``pi``, Y = 0  (structural zero).
        With probability ``1 - pi``, Y ~ NegBin(r, p=r/(r+mu)).

    Args:
        num_samples: Number of i.i.d. draws. Must be ≥ 1.
        pi: Zero-inflation probability. Range [0, 1).
        mu: Mean of the NegBin component. Must be > 0.
        r: Dispersion (concentration) of the NegBin component. Must be > 0.
        seed: RNG seed for reproducibility.

    Returns:
        Tuple of (samples, true_params) where:
            samples: integer tensor of shape ``(num_samples,)``
            true_params: dict ``{"pi": pi, "mu": mu, "r": r}``
    """
    _validate_zinb_params(pi, mu, r)
    assert num_samples >= 1, f"num_samples must be ≥ 1, got {num_samples}"

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)

        zero_mask = _sample_zero_mask(num_samples, pi, generator)
        nb_counts = _sample_negative_binomial(num_samples, mu, r, generator)

        # Apply zero-inflation: structural zeros override NB draws
        samples = nb_counts * (~zero_mask).long()  # (num_samples,)

    true_params = {"pi": pi, "mu": mu, "r": r}
    return samples, true_params


def _validate_zinb_params(pi: float, mu: float, r: float) -> None:
    """Assert ZINB parameter constraints."""
    assert (
        MIN_ZERO_INFLATION <= pi < MAX_ZERO_INFLATION
    ), f"pi must be in [0, 1), got {pi}"
    assert mu > MIN_POSITIVE, f"mu must be > 0, got {mu}"
    assert r > MIN_POSITIVE, f"r must be > 0, got {r}"


def _sample_zero_mask(n: int, pi: float, generator: torch.Generator) -> Tensor:
    """Sample a boolean mask for structural zeros.

    Returns:
        Boolean tensor of shape ``(n,)`` — True where the draw is a
        structural zero.
    """
    uniform = torch.rand(n, generator=generator)  # (n,)
    return uniform < pi  # (n,)


def _sample_negative_binomial(
    n: int, mu: float, r: float, generator: torch.Generator
) -> Tensor:
    """Sample from NegBin(r, p=r/(r+mu)) via the Gamma–Poisson mixture.

    The Gamma–Poisson representation avoids needing a discrete NB sampler:
        λ ~ Gamma(concentration=r, rate=r/mu)
        Y | λ ~ Poisson(λ)

    Returns:
        Integer tensor of shape ``(n,)``
    """
    # Gamma parameterised by concentration (shape) and rate
    gamma_concentration = torch.full((n,), r)  # (n,)
    gamma_rate = torch.full((n,), r / mu)  # (n,)
    # PyTorch Gamma uses (concentration, rate) parameterisation
    gamma_dist = torch.distributions.Gamma(gamma_concentration, gamma_rate)
    lambdas = gamma_dist.sample()  # (n,)

    # Poisson draws conditioned on the Gamma rates
    counts = torch.poisson(lambdas, generator=generator)  # (n,)
    return counts.long()  # (n,)


# ---------------------------------------------------------------------------
# Poisson Sampler (ZINB special case)
# ---------------------------------------------------------------------------
def generate_poisson_samples(
    num_samples: int,
    rate: float,
    seed: int = DEFAULT_SEED,
) -> tuple[Tensor, dict[str, float]]:
    """Generate samples from a Poisson distribution.

    This is a special case of ZINB with ``pi = 0`` and ``r → ∞``.
    Uses ``torch.poisson`` directly for efficiency.

    Args:
        num_samples: Number of i.i.d. draws. Must be ≥ 1.
        rate: Poisson rate parameter (λ). Must be > 0.
        seed: RNG seed for reproducibility.

    Returns:
        Tuple of (samples, true_params) where:
            samples: integer tensor of shape ``(num_samples,)``
            true_params: dict ``{"rate": rate}``
    """
    assert num_samples >= 1, f"num_samples must be ≥ 1, got {num_samples}"
    assert rate > MIN_POSITIVE, f"rate must be > 0, got {rate}"

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)

        rate_tensor = torch.full((num_samples,), rate)  # (num_samples,)
        samples = torch.poisson(
            rate_tensor, generator=generator
        ).long()  # (num_samples,)

    true_params = {"rate": rate}
    return samples, true_params


# ---------------------------------------------------------------------------
# Spatiotemporal Panel Generator
# ---------------------------------------------------------------------------
def generate_spatiotemporal_panel(
    num_spatial_units: int = 10,
    num_time_steps: int = 52,
    num_categories: int = 3,
    num_features: int = 5,
    seed: int = DEFAULT_SEED,
) -> dict[str, Tensor]:
    """Generate a synthetic spatiotemporal panel with known ZINB parameters.

    Creates a complete dataset mimicking weekly crime counts across spatial
    units and crime categories, with correlated covariates and a random
    geometric adjacency graph.

    Args:
        num_spatial_units: Number of spatial units (e.g. community areas).
            Range [2, 1000]. Default 10.
        num_time_steps: Number of time steps (e.g. weeks). Range [1, 520].
            Default 52 (one year).
        num_categories: Number of crime categories. Range [1, 20].
            Default 3 (violent, property, drug).
        num_features: Number of covariate features per (unit, time).
            Range [1, 100]. Default 5.
        seed: RNG seed for reproducibility.

    Returns:
        Dictionary with keys:
            ``counts``:    (spatial, time, category) int64
            ``features``:  (spatial, time, features) float32
            ``adjacency``: (spatial, spatial) float32 — binary symmetric
            ``true_pi``:   (spatial, category) float32
            ``true_mu``:   (spatial, category) float32
            ``true_r``:    (spatial, category) float32
    """
    _validate_panel_params(
        num_spatial_units, num_time_steps, num_categories, num_features
    )

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)

        true_pi, true_mu, true_r = _sample_ground_truth_params(
            num_spatial_units, num_categories, generator
        )

        counts = _generate_panel_counts(
            true_pi, true_mu, true_r, num_time_steps, generator
        )

        features = _generate_correlated_features(
            true_mu, num_time_steps, num_features, generator
        )

        adjacency = _build_random_geometric_graph(num_spatial_units, generator)

    return {
        "counts": counts,  # (spatial, time, category)
        "features": features,  # (spatial, time, features)
        "adjacency": adjacency,  # (spatial, spatial)
        "true_pi": true_pi,  # (spatial, category)
        "true_mu": true_mu,  # (spatial, category)
        "true_r": true_r,  # (spatial, category)
    }


def _validate_panel_params(
    num_spatial_units: int,
    num_time_steps: int,
    num_categories: int,
    num_features: int,
) -> None:
    """Assert panel dimension constraints."""
    assert (
        2 <= num_spatial_units <= 1000
    ), f"num_spatial_units must be in [2, 1000], got {num_spatial_units}"
    assert (
        1 <= num_time_steps <= 520
    ), f"num_time_steps must be in [1, 520], got {num_time_steps}"
    assert (
        1 <= num_categories <= 20
    ), f"num_categories must be in [1, 20], got {num_categories}"
    assert (
        1 <= num_features <= 100
    ), f"num_features must be in [1, 100], got {num_features}"


def _sample_ground_truth_params(
    num_spatial_units: int,
    num_categories: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sample ground-truth ZINB parameters per (spatial unit, category).

    pi ~ Uniform(0.05, 0.6)  — realistic zero-inflation range
    mu ~ Uniform(1, 50)      — realistic weekly crime count range
    r  ~ Uniform(0.5, 10)    — realistic dispersion range

    Returns:
        (true_pi, true_mu, true_r), each of shape ``(spatial, category)``.
    """
    shape = (num_spatial_units, num_categories)

    pi_low, pi_high = 0.05, 0.6
    mu_low, mu_high = 1.0, 50.0
    r_low, r_high = 0.5, 10.0

    true_pi = torch.rand(shape, generator=generator) * (pi_high - pi_low) + pi_low
    true_mu = torch.rand(shape, generator=generator) * (mu_high - mu_low) + mu_low
    true_r = torch.rand(shape, generator=generator) * (r_high - r_low) + r_low

    return true_pi, true_mu, true_r  # each (spatial, category)


def _generate_panel_counts(
    true_pi: Tensor,
    true_mu: Tensor,
    true_r: Tensor,
    num_time_steps: int,
    generator: torch.Generator,
) -> Tensor:
    """Generate ZINB counts for each (spatial unit, time step, category).

    Args:
        true_pi: (spatial, category) zero-inflation probabilities.
        true_mu: (spatial, category) NB means.
        true_r:  (spatial, category) NB dispersions.
        num_time_steps: number of time steps to generate.
        generator: torch random generator.

    Returns:
        Integer tensor of shape ``(spatial, time, category)``.
    """
    num_spatial, num_categories = true_pi.shape
    total = num_spatial * num_time_steps * num_categories

    # Expand parameters to (spatial, time, category)
    pi_expanded = true_pi.unsqueeze(1).expand(
        num_spatial, num_time_steps, num_categories
    )  # (spatial, time, category)
    mu_expanded = true_mu.unsqueeze(1).expand_as(pi_expanded)
    r_expanded = true_r.unsqueeze(1).expand_as(pi_expanded)

    # Flatten for vectorised sampling
    pi_flat = pi_expanded.reshape(total)  # (total,)
    mu_flat = mu_expanded.reshape(total)  # (total,)
    r_flat = r_expanded.reshape(total)  # (total,)

    # Structural zeros
    zero_mask = torch.rand(total, generator=generator) < pi_flat  # (total,)

    # Gamma–Poisson mixture for NB component
    gamma_dist = torch.distributions.Gamma(r_flat, r_flat / mu_flat)
    lambdas = gamma_dist.sample()  # (total,)
    nb_counts = torch.poisson(lambdas, generator=generator)  # (total,)

    counts_flat = nb_counts * (~zero_mask).float()  # (total,)
    counts = counts_flat.reshape(
        num_spatial, num_time_steps, num_categories
    ).long()  # (spatial, time, category)

    return counts


def _generate_correlated_features(
    true_mu: Tensor,
    num_time_steps: int,
    num_features: int,
    generator: torch.Generator,
) -> Tensor:
    """Generate covariate features weakly correlated with true_mu.

    The first feature dimension has a linear relationship with the log of
    the mean across categories (so a model can learn something).
    Remaining features are standard normal noise.

    Args:
        true_mu: (spatial, category) NB means.
        num_time_steps: number of time steps.
        num_features: total number of features.
        generator: torch random generator.

    Returns:
        Float tensor of shape ``(spatial, time, features)``.
    """
    num_spatial = true_mu.shape[0]

    # Signal: log-mean across categories, normalised to N(0,1) scale
    log_mu_mean = true_mu.log().mean(dim=1)  # (spatial,)
    signal = (log_mu_mean - log_mu_mean.mean()) / (
        log_mu_mean.std() + 1e-8
    )  # (spatial,)

    # Expand signal to (spatial, time, 1)
    signal_expanded = (
        signal.unsqueeze(1).unsqueeze(2).expand(num_spatial, num_time_steps, 1)
    )  # (spatial, time, 1)

    # Random noise for all feature dims
    noise = torch.randn(
        num_spatial, num_time_steps, num_features, generator=generator
    )  # (spatial, time, features)

    # Inject signal into the first feature column
    noise[:, :, :1] = (
        FEATURE_SIGNAL_WEIGHT * signal_expanded + FEATURE_NOISE_WEIGHT * noise[:, :, :1]
    )  # (spatial, time, 1) blended

    return noise  # (spatial, time, features)


def _build_random_geometric_graph(
    num_spatial_units: int,
    generator: torch.Generator,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
) -> Tensor:
    """Build a symmetric binary adjacency matrix from a random geometric graph.

    Units are placed uniformly at random in the 2D unit square. An edge
    exists between units i and j if their Euclidean distance is below
    ``distance_threshold``.

    Args:
        num_spatial_units: number of nodes.
        generator: torch random generator.
        distance_threshold: maximum distance for edge creation.
            Range (0, √2]. Default 0.3 gives moderate connectivity.

    Returns:
        Binary float tensor of shape ``(spatial, spatial)`` with zero diagonal.
    """
    assert (
        distance_threshold > 0
    ), f"distance_threshold must be > 0, got {distance_threshold}"

    # Random 2D positions in the unit square
    positions = torch.rand(num_spatial_units, 2, generator=generator)  # (spatial, 2)

    # Pairwise Euclidean distances
    diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # (spatial, spatial, 2)
    distances = diff.norm(dim=2)  # (spatial, spatial)

    # Threshold to get adjacency (no self-loops)
    adjacency = (distances < distance_threshold).float()  # (spatial, spatial)
    adjacency.fill_diagonal_(0.0)

    return adjacency  # (spatial, spatial)
