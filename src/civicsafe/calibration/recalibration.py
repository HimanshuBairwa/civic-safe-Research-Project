"""Post-hoc distributional recalibration for ZINB predictive models.

Analogous to Platt scaling for classifiers, this module learns a small set of
post-hoc correction parameters on a held-out calibration set to improve the
distributional quality of Zero-Inflated Negative Binomial (ZINB) predictions.

Two recalibration methods are supported:

**Affine recalibration** (4 parameters):

.. math::

    \\mu_{\\text{recal}} = \\text{softplus}(a) \\cdot \\mu + b, \\quad
    r_{\\text{recal}}   = \\text{softplus}(c) \\cdot r   + d

where :math:`a, c` are initialized so that :math:`\\text{softplus}(a)=1` and
:math:`b = d = 0`, recovering the identity map at initialisation.

**Temperature scaling** (1 parameter):

.. math::

    \\mu_{\\text{recal}} = \\mu \\cdot T, \\quad
    r_{\\text{recal}}   = r \\,/\\, T

Increasing *T* widens the predictive distribution (higher variance);
decreasing *T* sharpens it.  Both methods leave :math:`\\pi` unchanged
because it is already well-calibrated by the upstream sigmoid.

Parameters are optimised with Adam by minimising the mean CRPS on the
calibration set, which is a strictly proper scoring rule for distributional
forecasts (Gneiting & Raftery, 2007).

References:
    - Gneiting, T. & Raftery, A. E. (2007). Strictly Proper Scoring Rules,
      Prediction, and Estimation. *JASA*, 102(477), 359–378.
    - Kuleshov, V., Fenner, N. & Ermon, S. (2018). Accurate Uncertainties for
      Deep Learning Using Calibrated Regression. *ICML*.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Literal, Tuple

import torch
from torch import Tensor

from civicsafe.training.metrics import crps_zinb

logger = logging.getLogger(__name__)

# softplus⁻¹(1) ≈ 0.5413 — initialises softplus(a)=1 for identity map.
_SOFTPLUS_INV_ONE: float = 0.5413248546129181


class ZINBRecalibrator:
    """Post-hoc distributional recalibrator for ZINB predictions.

    Learns a lightweight parameter correction on a held-out calibration set
    by minimising the Continuous Ranked Probability Score (CRPS), then applies
    the learned correction to new predictions at inference time.

    Parameters
    ----------
    method : ``'affine'`` | ``'temperature'``
        Recalibration strategy.  ``'affine'`` learns four free parameters
        ``(a, b, c, d)``; ``'temperature'`` learns a single scalar ``T``.

    Examples
    --------
    >>> recal = ZINBRecalibrator(method="affine")
    >>> info = recal.fit(y_cal, pi_cal, mu_cal, r_cal)
    >>> pi_new, mu_new, r_new = recal.transform(pi_test, mu_test, r_test)
    """

    _VALID_METHODS = ("affine", "temperature")

    def __init__(self, method: Literal["affine", "temperature"] = "affine") -> None:
        if method not in self._VALID_METHODS:
            raise ValueError(
                f"Unknown recalibration method '{method}'. "
                f"Choose from {self._VALID_METHODS}."
            )
        self.method: str = method
        self._fitted: bool = False
        self._params: Dict[str, torch.nn.Parameter] = {}
        self._init_params()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def _init_params(self) -> None:
        """Create learnable parameters with identity-map initialisation."""
        if self.method == "affine":
            self._params = {
                "a": torch.nn.Parameter(torch.tensor(_SOFTPLUS_INV_ONE)),
                "b": torch.nn.Parameter(torch.tensor(0.0)),
                "c": torch.nn.Parameter(torch.tensor(_SOFTPLUS_INV_ONE)),
                "d": torch.nn.Parameter(torch.tensor(0.0)),
            }
        else:  # temperature
            self._params = {
                "T": torch.nn.Parameter(torch.tensor(1.0)),
            }

    # ------------------------------------------------------------------
    # Core transform
    # ------------------------------------------------------------------
    def _apply(
        self, pi: Tensor, mu: Tensor, r: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Apply the current parameterisation (differentiable)."""
        if self.method == "affine":
            scale_mu = torch.nn.functional.softplus(self._params["a"])
            scale_r = torch.nn.functional.softplus(self._params["c"])
            mu_out = scale_mu * mu + self._params["b"]
            r_out = scale_r * r + self._params["d"]
        else:  # temperature
            T = self._params["T"].clamp(min=1e-4)  # prevent division by zero
            mu_out = mu * T
            r_out = r / T

        # Safety clamps — keep values in valid ZINB domain
        mu_out = mu_out.clamp(min=1e-6)
        r_out = r_out.clamp(min=0.1)
        pi_out = pi.clamp(0.0, 1.0)
        return pi_out, mu_out, r_out

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(
        self,
        y_cal: Tensor,
        pi_cal: Tensor,
        mu_cal: Tensor,
        r_cal: Tensor,
        lr: float = 0.01,
        max_iter: int = 500,
    ) -> Dict[str, Any]:
        """Learn recalibration parameters by minimising CRPS on calibration data.

        Uses Adam to minimise:

        .. math::

            \\mathcal{L} = \\frac{1}{N}\\sum_{i=1}^{N}
                \\text{CRPS}\\bigl(F_{\\theta}(\\cdot; \\hat\\pi_i,
                \\hat\\mu_i, \\hat r_i),\\; y_i\\bigr)

        where :math:`\\theta` are the recalibration parameters.

        Parameters
        ----------
        y_cal : Tensor
            Observed counts on the calibration split.
        pi_cal, mu_cal, r_cal : Tensor
            ZINB parameter predictions on the calibration split.
        lr : float
            Adam learning rate.
        max_iter : int
            Maximum optimisation steps.

        Returns
        -------
        dict
            Training summary with keys ``initial_crps``, ``final_crps``,
            ``improvement_pct``, and ``iterations``.
        """
        if y_cal.numel() == 0:
            raise ValueError("Calibration set is empty.")

        # Re-initialise to identity so fit() is idempotent
        self._init_params()

        # Move parameters to same device as data
        device = y_cal.device
        for name in self._params:
            self._params[name] = torch.nn.Parameter(
                self._params[name].data.to(device)
            )

        optimizer = torch.optim.Adam(list(self._params.values()), lr=lr)

        # Detach inputs — they are frozen model outputs, not part of the graph
        y = y_cal.detach().float()
        pi = pi_cal.detach().float()
        mu = mu_cal.detach().float()
        r = r_cal.detach().float()

        initial_crps: float | None = None
        best_crps: float = float("inf")
        patience_counter: int = 0
        final_iter: int = 0

        for step in range(1, max_iter + 1):
            optimizer.zero_grad()
            pi_t, mu_t, r_t = self._apply(pi, mu, r)
            loss = crps_zinb(y, pi_t, mu_t, r_t).mean()
            loss.backward()
            optimizer.step()

            crps_val = loss.item()
            if initial_crps is None:
                initial_crps = crps_val

            if crps_val < best_crps - 1e-6:
                best_crps = crps_val
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping with generous patience
            if patience_counter >= 50:
                logger.info(
                    "Recalibration converged at step %d (CRPS=%.6f).", step, crps_val
                )
                final_iter = step
                break

            if step % 100 == 0:
                logger.debug(
                    "Recalibration step %d/%d — CRPS: %.6f", step, max_iter, crps_val
                )
        else:
            final_iter = max_iter

        self._fitted = True
        assert initial_crps is not None  # guaranteed by at least 1 iteration

        improvement_pct = (
            (initial_crps - best_crps) / max(initial_crps, 1e-12) * 100.0
        )
        summary = {
            "initial_crps": initial_crps,
            "final_crps": best_crps,
            "improvement_pct": improvement_pct,
            "iterations": final_iter,
        }
        logger.info(
            "Recalibration complete — CRPS %.6f → %.6f (%.2f%% improvement, %d iters).",
            initial_crps,
            best_crps,
            improvement_pct,
            final_iter,
        )
        return summary

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------
    @torch.no_grad()
    def transform(
        self, pi: Tensor, mu: Tensor, r: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Apply learned recalibration to new ZINB predictions.

        Parameters
        ----------
        pi, mu, r : Tensor
            Raw model outputs.

        Returns
        -------
        tuple of Tensor
            ``(pi_recal, mu_recal, r_recal)`` with corrected distributional
            parameters.

        Raises
        ------
        RuntimeError
            If :py:meth:`fit` has not been called.
        """
        if not self._fitted:
            raise RuntimeError(
                "Recalibrator has not been fitted. Call fit() first."
            )
        return self._apply(pi, mu, r)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------
    def get_params(self) -> Dict[str, float]:
        """Return the learned recalibration parameters as plain floats.

        Returns
        -------
        dict
            Mapping from parameter name to its current value.  For the
            ``'affine'`` method the effective multiplicative scales are
            returned (i.e. ``softplus(a)`` and ``softplus(c)``).
        """
        if self.method == "affine":
            return {
                "scale_mu": torch.nn.functional.softplus(self._params["a"]).item(),
                "shift_mu": self._params["b"].item(),
                "scale_r": torch.nn.functional.softplus(self._params["c"]).item(),
                "shift_r": self._params["d"].item(),
            }
        # temperature
        return {"T": self._params["T"].item()}

    def __repr__(self) -> str:  # pragma: no cover
        status = "fitted" if self._fitted else "unfitted"
        return f"ZINBRecalibrator(method={self.method!r}, {status})"


# ======================================================================
# Convenience function
# ======================================================================
def recalibrate_and_evaluate(
    y_cal: Tensor,
    pi_cal: Tensor,
    mu_cal: Tensor,
    r_cal: Tensor,
    y_test: Tensor,
    pi_test: Tensor,
    mu_test: Tensor,
    r_test: Tensor,
    method: Literal["affine", "temperature"] = "affine",
    lr: float = 0.01,
    max_iter: int = 500,
) -> Tuple[Tuple[Tensor, Tensor, Tensor], Dict[str, Any]]:
    """Fit a recalibrator on calibration data and evaluate on a test set.

    This is a convenience wrapper that:

    1. Fits a :class:`ZINBRecalibrator` on ``(y_cal, pi_cal, mu_cal, r_cal)``.
    2. Applies the learned correction to ``(pi_test, mu_test, r_test)``.
    3. Computes before/after CRPS on the test set.

    Parameters
    ----------
    y_cal, pi_cal, mu_cal, r_cal : Tensor
        Calibration split (observations + model predictions).
    y_test, pi_test, mu_test, r_test : Tensor
        Test split (observations + model predictions).
    method : str
        Recalibration method (``'affine'`` or ``'temperature'``).
    lr : float
        Learning rate for Adam.
    max_iter : int
        Maximum optimisation steps.

    Returns
    -------
    (pi_recal, mu_recal, r_recal) : tuple of Tensor
        Recalibrated ZINB parameters for the test set.
    metrics : dict
        Contains ``'cal_initial_crps'``, ``'cal_final_crps'``,
        ``'cal_improvement_pct'``, ``'test_crps_before'``,
        ``'test_crps_after'``, ``'test_improvement_pct'``,
        ``'learned_params'``, and ``'iterations'``.
    """
    recalibrator = ZINBRecalibrator(method=method)
    fit_info = recalibrator.fit(y_cal, pi_cal, mu_cal, r_cal, lr=lr, max_iter=max_iter)

    # Test CRPS before recalibration
    with torch.no_grad():
        test_crps_before = crps_zinb(y_test, pi_test, mu_test, r_test).mean().item()

    # Apply recalibration
    pi_recal, mu_recal, r_recal = recalibrator.transform(pi_test, mu_test, r_test)

    # Test CRPS after recalibration
    with torch.no_grad():
        test_crps_after = crps_zinb(y_test, pi_recal, mu_recal, r_recal).mean().item()

    test_improvement = (
        (test_crps_before - test_crps_after) / max(test_crps_before, 1e-12) * 100.0
    )

    metrics: Dict[str, Any] = {
        "cal_initial_crps": fit_info["initial_crps"],
        "cal_final_crps": fit_info["final_crps"],
        "cal_improvement_pct": fit_info["improvement_pct"],
        "test_crps_before": test_crps_before,
        "test_crps_after": test_crps_after,
        "test_improvement_pct": test_improvement,
        "learned_params": recalibrator.get_params(),
        "iterations": fit_info["iterations"],
    }

    logger.info(
        "Test CRPS: %.6f → %.6f (%.2f%% improvement).",
        test_crps_before,
        test_crps_after,
        test_improvement,
    )

    return (pi_recal, mu_recal, r_recal), metrics
