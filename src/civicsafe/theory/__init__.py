"""Theoretical core of CIVIC-SAFE: the Feedback Amplification Law.

See :mod:`civicsafe.theory.feedback_law` for the AOBF model, the universal
amplification law, the runaway-discrimination corollary, and the
passive/active identification duality; :mod:`civicsafe.theory.latent_correction`
for the feedback-corrected latent predictor; :mod:`civicsafe.theory.feedback_tripwire`
for the anytime-valid live monitor; and :mod:`civicsafe.theory.sensitivity` for
the correction's robustness envelope under gain misspecification.
"""

from civicsafe.theory.correction_robustness import (
    GammaRobustnessResult,
    robust_latent_interval,
    robustness_gamma,
)
from civicsafe.theory.feedback_law import (
    KAPPA_STAR,
    FeedbackLawResult,
    amplification_exponent,
    disparity_ratio,
    general_fixed_point,
    identify_kappa_did,
    local_feedback_gain,
    power_law_fixed_point,
)
from civicsafe.theory.feedback_tripwire import FeedbackTripwire, TripwireState
from civicsafe.theory.latent_correction import (
    deflate_latent_rate,
    latent_prediction_interval,
    recording_multiplier,
    should_abstain,
)
from civicsafe.theory.sensitivity import (
    RobustnessResult,
    robustness_value,
    sensitivity_curve,
)

__all__ = [
    "KAPPA_STAR",
    "FeedbackLawResult",
    "FeedbackTripwire",
    "GammaRobustnessResult",
    "RobustnessResult",
    "TripwireState",
    "amplification_exponent",
    "deflate_latent_rate",
    "disparity_ratio",
    "general_fixed_point",
    "identify_kappa_did",
    "latent_prediction_interval",
    "local_feedback_gain",
    "power_law_fixed_point",
    "recording_multiplier",
    "robust_latent_interval",
    "robustness_gamma",
    "robustness_value",
    "sensitivity_curve",
    "should_abstain",
]
