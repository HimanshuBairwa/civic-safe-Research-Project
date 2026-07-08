"""Tests for the ShotSpotter DiD field-identification estimator.

The estimator's correctness is established on synthetic panels with a known,
injected recording shock: it must recover the shock magnitude, be insignificant
at zero shock, and show flat pre-trends. (Real-data identification additionally
requires official deployment records; see docs/RESULTS_field_identification.md.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from civicsafe.theory.field_identification import (
    ShotSpotterRollout,
    estimate_did,
    event_study,
    implied_kappa,
)


def _synthetic(true_shock_log: float, seed: int = 1) -> tuple[pd.DataFrame, ShotSpotterRollout]:
    rng = np.random.default_rng(seed)
    n_units, n_months, activate_idx = 60, 60, 30
    units = np.arange(1, n_units + 1)
    months = pd.period_range("2015-01", periods=n_months, freq="M").to_timestamp()
    unit_fx = rng.normal(0, 0.5, n_units)
    month_fx = np.cumsum(rng.normal(0, 0.02, n_months))
    treated = set(units[: n_units // 3])
    activate = months[activate_idx]
    rows = []
    for i, u in enumerate(units):
        for t, m in enumerate(months):
            lr = unit_fx[i] + month_fx[t] + rng.normal(0, 0.1) + 2.0
            if u in treated and m >= activate:
                lr += true_shock_log
            rows.append({"spatial_unit": u, "month": m, "log_rate": lr})
    return pd.DataFrame(rows), ShotSpotterRollout(sorted(treated), f"{activate:%Y-%m}")


def test_recovers_known_shock() -> None:
    """DiD recovers an injected recording shock to within tolerance."""
    for true_shock in [0.15, 0.30, 0.50]:
        panel, rollout = _synthetic(true_shock)
        did = estimate_did(panel, rollout)
        assert abs(did.tau - true_shock) < 0.03
        assert did.pvalue < 0.01


def test_no_false_positive_at_zero_shock() -> None:
    """With no injected shock, the DiD is insignificant (no spurious effect)."""
    panel, rollout = _synthetic(0.0)
    did = estimate_did(panel, rollout)
    assert did.pvalue > 0.05
    assert abs(did.tau) < 0.03


def test_flat_pretrends_under_true_model() -> None:
    """Event-study pre-period coefficients are near zero under the true model."""
    panel, rollout = _synthetic(0.30)
    es = event_study(panel, rollout, max_lead=6, max_lag=10)
    pre = es[es["rel_month"] < 0]
    assert pre["coef"].abs().mean() < 0.05


def test_implied_kappa_scales_with_beta() -> None:
    """Implied kappa scales linearly with the assumed policy elasticity beta."""
    panel, rollout = _synthetic(0.30)
    did = estimate_did(panel, rollout)
    tab = implied_kappa(did, beta_grid=(1.0, 2.0))
    k1 = tab[tab["beta"] == 1.0]["kappa_hat"].iloc[0]
    k2 = tab[tab["beta"] == 2.0]["kappa_hat"].iloc[0]
    assert abs(k2 - 2 * k1) < 1e-9
