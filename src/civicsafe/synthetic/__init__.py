"""Synthetic data generators for CIVIC-SAFE testing and development.

This package provides generators that produce data with known ground-truth
parameters, enabling deterministic unit testing of the ZINB model, conformal
calibration, and spatiotemporal pipeline components.

Exports:
    generate_zinb_samples: ZINB draws with known (pi, mu, r).
    generate_poisson_samples: Poisson draws (ZINB special case).
    generate_spatiotemporal_panel: Full (spatial × time × category) panel.
"""

from civicsafe.synthetic.distributions import (
    generate_poisson_samples,
    generate_spatiotemporal_panel,
    generate_zinb_samples,
)

__all__ = [
    "generate_poisson_samples",
    "generate_spatiotemporal_panel",
    "generate_zinb_samples",
]
