"""OICC -- Over-Identification-Calibrated Conformal Deconvolution.

A small, dependency-light (numpy + scipy) implementation of the honest
multi-channel latent-rate estimation method:

  * measurement model + synthetic generators           (`generate`)
  * one-factor moment estimation of the latent law     (`moments`)
  * robust latent recovery (BLUP / empirical Bayes)     (`deconvolve`)
  * over-identification specification test              (`spec_test`)
  * leave-pivot-out conformal set for the latent target (`conformal`)
  * Delta-perp / Delta-parallel sensitivity inflation    (`sensitivity`)

Design rules (so nothing breaks downstream):
  - every public function validates shapes and dtypes,
  - every matrix inverse is a pseudo-inverse or ridged,
  - every variance is floored at a small positive epsilon,
  - no reliance on optional/unstable libraries.

The scientific claims are deliberately *honest*: the method identifies the
latent-rate distribution and makes conditional independence falsifiable in the
DETECTABLE (Delta-perp) subspace, while the common-mode (Delta-parallel)
subspace is quarantined into a single user knob `gamma_cm`. See docstrings.
"""
from __future__ import annotations

from oicc.measurement import (
    Channels,
    ProximalChannels,
    generate,
    generate_proximal,
    to_log_rate,
)
from oicc.moments import (
    FactorMoments,
    estimate_factor_moments,
    pairwise_varu,
)
from oicc.deconvolve import (
    LatentEstimate,
    deconvolve_blup,
    blup_from_subset,
)
from oicc.spec_test import (
    SpecTestResult,
    overid_wald_test,
    CumulantTestResult,
    overid_cumulant_test,
)
from oicc.cf_deconv import (
    DeconvDensity,
    deconvolve_error_law,
)
from oicc.conformal import (
    ConformalResult,
    leave_pivot_out_conformal,
)
from oicc.conformal_split import (
    SplitConformalResult,
    split_conformal_latent,
)
from oicc.proximal import (
    ProximalCorrection,
    proximal_deconfound,
    PointIDResult,
    point_identify,
    ExclusionSensitivity,
    exclusion_sensitivity,
)
from oicc.monitor import (
    EProcessMonitor,
)
from oicc.uncertainty import (
    BootstrapCI,
    bootstrap_moments,
    bootstrap_point_id,
)
from oicc.baselines import (
    BaselineComparison,
    compare_baselines,
    compare_baselines_confounded,
)

__all__ = [
    "Channels",
    "ProximalChannels",
    "generate",
    "generate_proximal",
    "to_log_rate",
    "FactorMoments",
    "estimate_factor_moments",
    "pairwise_varu",
    "LatentEstimate",
    "deconvolve_blup",
    "blup_from_subset",
    "SpecTestResult",
    "overid_wald_test",
    "CumulantTestResult",
    "overid_cumulant_test",
    "DeconvDensity",
    "deconvolve_error_law",
    "ConformalResult",
    "leave_pivot_out_conformal",
    "SplitConformalResult",
    "split_conformal_latent",
    "ProximalCorrection",
    "proximal_deconfound",
    "PointIDResult",
    "point_identify",
    "ExclusionSensitivity",
    "exclusion_sensitivity",
    "EProcessMonitor",
    "BootstrapCI",
    "bootstrap_moments",
    "bootstrap_point_id",
    "BaselineComparison",
    "compare_baselines",
    "compare_baselines_confounded",
]

__version__ = "0.6.0"
