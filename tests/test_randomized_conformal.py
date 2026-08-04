"""Tests for randomized-PIT conformal calibration on discrete counts.

The defect these tests pin: the shipped CQR pipeline conformalizes an INTEGER
nonconformity score. On a sparse count panel the raw ZINB interval already
overcovers -- because the smallest integer k with F(k) >= 1-alpha/2 almost
always has F(k) strictly greater, and because q_low collapses to 0 whenever the
zero-mass exceeds alpha/2 -- so more than (1-alpha) of scores land at or below
zero, the empirical quantile pins to exactly 0.0, and the conformal correction
becomes the IDENTITY. Chicago's reported 0.9278 "calibrated" coverage at a 0.90
target was the uncalibrated ZINB interval.

The exact mechanism, measured: q_low = 0 for 100% of cells, so any observed
y = 0 scores max(0 - 0, 0 - q_high) = 0 exactly. Every structural zero piles
onto one atom at score 0 (~21% of cells), on top of the ~75% scoring strictly
negative. The (1-alpha) quantile therefore falls inside that atom and returns
exactly 0.0. Because the correction is then added to integer quantiles and
re-rounded, the effective control is quantized: t = 0 already overcovers and
t = 1 overcovers more, each step adding ~2 to the interval width, so NO
admissible threshold attains 0.90. The same atom flattens the quantile function
over roughly [0.80, 0.955], which is a deadband the ACI controller in
AdaptiveTemporalECRC must escape before any of its alpha_t updates change a
single interval -- adaptation can appear to run while changing nothing.

The randomized PIT (Dunn & Smyth, 1996) is exactly Uniform(0,1) for discrete y
under a correct model, so conformalizing it restores the exact finite-sample
guarantee. It does NOT make the delivered integer intervals nominal -- integer
endpoints on a lattice always overcover -- and the tests below pin BOTH facts,
so nobody later "fixes" the conservative integer coverage by reintroducing the
degenerate path.
"""

from __future__ import annotations

import logging

import pytest
import torch

from civicsafe.calibration.conformal import (
    AdaptiveTemporalECRCCalibrator,
    ECRCCalibrator,
    RandomizedSplitConformalCalibrator,
    SplitConformalCalibrator,
    compute_cqr_scores,
    randomized_pit,
)

ALPHA = 0.1
# Scales chosen to span the real panel's range: Drug is near-zero-inflated,
# Property runs an order of magnitude higher.
SCALES = [0.5, 2.0, 4.0, 10.0, 40.0]
PER_SCALE = 4000


def _draw(n: int, p: float, mv: float, rv: float) -> tuple[torch.Tensor, ...]:
    pi = torch.full((n,), p)
    mu = torch.full((n,), mv)
    r = torch.full((n,), rv)
    z = (torch.rand(n) < p).float()
    nb = torch.distributions.NegativeBinomial(
        total_count=r, probs=mu / (mu + r)
    ).sample()
    return (1 - z) * nb, pi, mu, r


def _mixed_panel() -> tuple[torch.Tensor, ...]:
    """A mixed-scale ZINB panel, drawn from the model that is being calibrated.

    Exchangeability holds by construction, so split conformal's guarantee is
    exact here and any departure is attributable to the method, not to drift.
    """
    ys, pis, mus, rs = [], [], [], []
    for mv in SCALES:
        y, pi, mu, r = _draw(PER_SCALE, 0.05, mv, 2.0)
        ys.append(y)
        pis.append(pi)
        mus.append(mu)
        rs.append(r)
    return (
        torch.cat(ys),
        torch.cat(pis),
        torch.cat(mus),
        torch.cat(rs),
    )


def _split(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    perm = torch.randperm(n)
    return perm[: n // 2], perm[n // 2 :]


# ---------------------------------------------------------------------------
# The defect itself
# ---------------------------------------------------------------------------
def test_integer_cqr_score_is_degenerate_on_sparse_counts() -> None:
    """The shipped score pins the threshold to 0, making calibration a no-op.

    This is the root-cause test. If it ever fails, the discreteness problem has
    changed character and the randomized path may no longer be necessary.
    """
    torch.manual_seed(0)
    y, pi, mu, r = _mixed_panel()

    scores = compute_cqr_scores(y, pi, mu, r, alpha=ALPHA)
    frac_inside = (scores <= 0).float().mean().item()

    assert frac_inside > 1.0 - ALPHA, (
        f"only {frac_inside:.1%} of scores are <= 0; the degeneracy that "
        "motivates randomized conformal is not present"
    )

    cal = SplitConformalCalibrator(alpha=ALPHA)
    cal.fit(y, pi, mu, r)
    assert cal.threshold == pytest.approx(0.0, abs=1e-9), (
        f"threshold {cal.threshold} is nonzero; expected the empirical "
        "quantile to pin to 0 given the score distribution above"
    )


def test_degenerate_calibration_emits_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A silent no-op is what let 0.9278 ship as 'calibrated'. It must be loud."""
    torch.manual_seed(0)
    y, pi, mu, r = _mixed_panel()

    with caplog.at_level(logging.WARNING, logger="civicsafe.calibration.conformal"):
        SplitConformalCalibrator(alpha=ALPHA).fit(y, pi, mu, r)

    assert any("DEGENERATE CALIBRATION" in rec.message for rec in caplog.records), (
        "no degeneracy warning was logged, so a no-op calibration would again "
        "be indistinguishable from a real one in the run log"
    )


def test_raw_zinb_interval_overcovers_because_of_discreteness() -> None:
    """Establishes that the overcoverage is intrinsic, not a coding error.

    Pins the direction and rough size so a future change to the quantile
    function that accidentally narrows intervals gets caught.
    """
    torch.manual_seed(0)
    y, pi, mu, r = _mixed_panel()
    from civicsafe.calibration.zinb_distribution import zinb_ppf_pair

    q_low, q_high = zinb_ppf_pair(ALPHA, pi, mu, r)
    coverage = ((y >= q_low) & (y <= q_high)).float().mean().item()

    assert coverage > 1.0 - ALPHA + 0.02, (
        f"raw interval coverage {coverage:.4f} is not meaningfully above the "
        f"{1 - ALPHA:.2f} target; the discreteness argument would not hold"
    )
    # q_low collapsing to 0 is half the story: the interval is one-sided.
    assert (q_low == 0).float().mean().item() > 0.5, (
        "q_low did not collapse to 0 for most cells, so the one-sided-interval "
        "part of the diagnosis no longer applies"
    )


# ---------------------------------------------------------------------------
# The randomized PIT
# ---------------------------------------------------------------------------
def test_randomized_pit_is_uniform() -> None:
    """The whole method rests on this: u ~ U(0,1) exactly, despite discreteness."""
    torch.manual_seed(0)
    y, pi, mu, r = _mixed_panel()

    u = randomized_pit(y, pi, mu, r)

    assert u.min().item() >= 0.0 and u.max().item() <= 1.0
    # Uniform(0,1) has mean 1/2 and sd 1/sqrt(12) = 0.2887.
    assert u.mean().item() == pytest.approx(0.5, abs=0.01)
    assert u.std().item() == pytest.approx(0.28868, abs=0.01)

    # Decile occupancy: a discrete PIT would clump; this must not.
    hist = torch.histc(u, bins=10, min=0.0, max=1.0) / u.numel()
    assert hist.max().item() < 0.13, f"PIT deciles are not flat: {hist.tolist()}"
    assert hist.min().item() > 0.07, f"PIT deciles are not flat: {hist.tolist()}"


def test_randomized_pit_is_reproducible_with_a_seed() -> None:
    """Randomization is real auxiliary noise, so the seed must pin the result."""
    torch.manual_seed(0)
    y, pi, mu, r = _mixed_panel()
    n = y.numel()
    cal, test = _split(n)

    def band(seed: int) -> tuple[float, float]:
        c = RandomizedSplitConformalCalibrator(alpha=ALPHA, seed=seed)
        c.fit(y[cal], pi[cal], mu[cal], r[cal])
        return c._lo_level, c._hi_level  # type: ignore[return-value]

    assert band(0) == band(0)
    assert band(0) != band(12345), (
        "different seeds produced an identical band; the PIT randomization is "
        "not actually being driven by the seed"
    )


def test_randomized_pit_recovers_counts_at_the_cdf_boundaries() -> None:
    """u must lie in (F(y-1), F(y)] — the interval whose width is y's own mass."""
    torch.manual_seed(0)
    from civicsafe.calibration.zinb_distribution import zinb_cdf_full

    y, pi, mu, r = _draw(2000, 0.05, 3.0, 2.0)
    u = randomized_pit(y, pi, mu, r)

    _, F = zinb_cdf_full(pi, mu, r)
    idx = y.long()
    F_y = F.gather(1, idx.unsqueeze(-1)).squeeze(-1)
    F_prev = torch.where(
        idx > 0,
        F.gather(1, (idx - 1).clamp(min=0).unsqueeze(-1)).squeeze(-1),
        torch.zeros_like(F_y),
    )

    assert (u >= F_prev - 1e-6).all(), "PIT fell below F(y-1)"
    assert (u <= F_y + 1e-6).all(), "PIT exceeded F(y)"


# ---------------------------------------------------------------------------
# The guarantee, and its honest limit
# ---------------------------------------------------------------------------
def test_randomized_conformal_is_exact_in_pit_space() -> None:
    """The claim the paper can actually make: exact finite-sample coverage.

    Split conformal on a continuous score gives coverage in
    [1-alpha, 1-alpha + 1/(n+1)]. With n ~ 10000 that band is essentially a
    point, so a tolerance of 0.01 is generous and still discriminating.
    """
    torch.manual_seed(0)
    y, pi, mu, r = _mixed_panel()
    n = y.numel()
    cal, test = _split(n)

    c = RandomizedSplitConformalCalibrator(alpha=ALPHA, seed=0)
    c.fit(y[cal], pi[cal], mu[cal], r[cal])
    cov = c.coverage_in_pit_space(y[test], pi[test], mu[test], r[test])

    assert cov == pytest.approx(1.0 - ALPHA, abs=0.01), (
        f"PIT-space coverage {cov:.4f} missed the {1 - ALPHA:.2f} target; the "
        "exactness claim does not hold"
    )


def test_integer_intervals_still_overcover_and_that_is_documented() -> None:
    """Guards the honest limit: randomization does NOT make integers nominal.

    Inverting the calibrated PIT band through zinb_ppf reimposes the lattice
    ceiling. Asserting the conservatism is deliberate -- if someone later makes
    predict() hit 0.90 exactly, either they found something genuinely better or
    they reintroduced a bug, and either way this test should force the
    conversation.
    """
    torch.manual_seed(0)
    y, pi, mu, r = _mixed_panel()
    n = y.numel()
    cal, test = _split(n)

    c = RandomizedSplitConformalCalibrator(alpha=ALPHA, seed=0)
    c.fit(y[cal], pi[cal], mu[cal], r[cal])
    iv = c.predict(pi[test], mu[test], r[test])

    cov = ((y[test] >= iv["lower"]) & (y[test] <= iv["upper"])).float().mean().item()
    assert cov > 1.0 - ALPHA, (
        f"integer-interval coverage {cov:.4f} dropped to or below nominal; "
        "on a discrete law that suggests the bounds are too narrow"
    )

    pit_cov = c.coverage_in_pit_space(y[test], pi[test], mu[test], r[test])
    assert cov > pit_cov, (
        "integer coverage is not above PIT coverage, so the lattice-ceiling "
        "effect this test exists to document is absent"
    )


def test_intervals_are_valid_and_ordered() -> None:
    """Basic contract: 0 <= L <= U, integer-valued, no NaN."""
    torch.manual_seed(0)
    y, pi, mu, r = _mixed_panel()
    n = y.numel()
    cal, test = _split(n)

    c = RandomizedSplitConformalCalibrator(alpha=ALPHA, seed=0)
    c.fit(y[cal], pi[cal], mu[cal], r[cal])
    iv = c.predict(pi[test], mu[test], r[test])

    assert torch.isfinite(iv["lower"]).all() and torch.isfinite(iv["upper"]).all()
    assert (iv["lower"] >= 0).all()
    assert (iv["upper"] >= iv["lower"]).all()
    assert torch.equal(iv["lower"], iv["lower"].round())
    assert torch.equal(iv["upper"], iv["upper"].round())
    # point estimate is E[Y] = (1-pi)*mu, independent of the calibration band
    assert iv["point"].shape == iv["lower"].shape


def test_predict_before_fit_raises() -> None:
    c = RandomizedSplitConformalCalibrator(alpha=ALPHA)
    with pytest.raises(RuntimeError, match="fit"):
        c.predict(torch.rand(4), torch.rand(4) + 1, torch.rand(4) + 1)


def test_alpha_is_validated() -> None:
    with pytest.raises(ValueError, match="alpha"):
        RandomizedSplitConformalCalibrator(alpha=0.9)


# ---------------------------------------------------------------------------
# Root cause: zero-inflation manufactures an atom at score == 0
# ---------------------------------------------------------------------------
def test_every_zero_observation_scores_exactly_zero() -> None:
    """The precise mechanism behind the degeneracy, isolated.

    The CQR score is max(q_low - y, y - q_high). Zero-inflation drives q_low to
    0 for essentially every cell, so for an observed y = 0 the score is
    max(0 - 0, 0 - q_high) = max(0, -q_high) = 0 exactly. Every structural zero
    therefore lands on the same atom. That atom -- not any coding error -- is
    what the empirical quantile falls into, pinning the threshold to 0.

    This is the load-bearing fact for the whole module. If it changes, the
    degeneracy analysis in these docstrings needs redoing.
    """
    torch.manual_seed(0)
    y, pi, mu, r = _mixed_panel()
    from civicsafe.calibration.zinb_distribution import zinb_ppf_pair

    q_low, _ = zinb_ppf_pair(ALPHA, pi, mu, r)
    scores = compute_cqr_scores(y, pi, mu, r, alpha=ALPHA)

    assert (q_low == 0).all(), (
        "q_low is no longer identically 0, so the mechanism below does not apply"
    )
    zero_obs = y == 0
    assert zero_obs.any(), "panel has no zeros; it cannot exercise this path"
    assert (scores[zero_obs].abs() < 1e-9).all(), (
        "some y == 0 cell did not score exactly 0, contradicting the "
        "max(0, -q_high) = 0 derivation"
    )


def test_no_integer_threshold_achieves_nominal_coverage() -> None:
    """Why every method in this module overcovers: the control is quantized.

    The conformal correction is added to integer ZINB quantiles and re-rounded,
    so the delivered threshold is effectively an integer. t = 0 already
    overcovers and t = 1 overcovers by more, with each step adding ~2 to the
    width. No admissible t lands on 0.90, so conservatism is a property of the
    lattice rather than a tuning failure.
    """
    torch.manual_seed(0)
    y, pi, mu, r = _mixed_panel()
    from civicsafe.calibration.zinb_distribution import zinb_ppf_pair

    q_low, q_high = zinb_ppf_pair(ALPHA, pi, mu, r)

    coverages = []
    for t in range(4):
        lower = (q_low - t).clamp(min=0.0).floor()
        upper = (q_high + t).ceil()
        coverages.append(((y >= lower) & (y <= upper)).float().mean().item())

    assert coverages[0] > 1.0 - ALPHA, (
        f"t=0 coverage {coverages[0]:.4f} is at or below nominal, so widening "
        "would be the correct response and the argument here does not hold"
    )
    assert all(
        b > a for a, b in zip(coverages, coverages[1:])
    ), f"coverage is not monotone in the threshold: {coverages}"


# ---------------------------------------------------------------------------
# The same degeneracy disables the adaptive controller
# ---------------------------------------------------------------------------
def test_adaptive_ecrc_warns_when_update_was_never_called(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """fit()->predict() makes AdaptiveTemporalECRC a byte-identical clone of ECRC.

    That is exactly how a duplicate row reached the Chicago results table while
    being labelled 'adaptive'. Both halves are asserted: the warning fires, and
    the outputs really are identical.
    """
    torch.manual_seed(0)
    y, pi, mu, r = _draw(4000, 0.05, 3.0, 2.0)
    groups = torch.randint(0, 4, (4000,))

    with caplog.at_level(logging.WARNING, logger="civicsafe.calibration.conformal"):
        adaptive = AdaptiveTemporalECRCCalibrator(alpha=ALPHA, group_type="demographic")
        adaptive.fit(y, pi, mu, r, groups=groups)
        iv_adaptive = adaptive.predict(pi, mu, r, groups=groups)

    assert any("0 update() calls" in rec.message for rec in caplog.records), (
        "no warning fired, so a static 'adaptive' run would again be "
        "indistinguishable from a genuine one in the log"
    )

    plain = ECRCCalibrator(alpha=ALPHA, group_type="demographic")
    plain.fit(y, pi, mu, r, groups=groups)
    iv_plain = plain.predict(pi, mu, r, groups=groups)

    assert torch.equal(iv_adaptive["lower"], iv_plain["lower"])
    assert torch.equal(iv_adaptive["upper"], iv_plain["upper"])


def test_adaptive_ecrc_does_not_warn_once_update_has_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The rolling loop is the supported path; it must stay warning-free."""
    torch.manual_seed(0)
    y, pi, mu, r = _draw(4000, 0.05, 3.0, 2.0)
    groups = torch.randint(0, 4, (4000,))

    c = AdaptiveTemporalECRCCalibrator(alpha=ALPHA, group_type="demographic")
    c.fit(y, pi, mu, r, groups=groups)
    with caplog.at_level(logging.WARNING, logger="civicsafe.calibration.conformal"):
        for w in range(4):
            s = slice(w * 500, (w + 1) * 500)
            c.update(y[s], pi[s], mu[s], r[s], groups=groups[s])
        c.predict(pi, mu, r, groups=groups)

    assert c._n_updates == 4
    assert not any("0 update() calls" in rec.message for rec in caplog.records), (
        "the never-updated warning fired on the rolling path, which would train "
        "readers to ignore it"
    )

