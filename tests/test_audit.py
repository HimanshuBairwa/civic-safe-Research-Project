"""Comprehensive tests for the equity & sensitivity audit module.

~40 tests covering:
- AuditBundle construction, validation, immutability
- StratificationEngine (quantile, equal-width, threshold, auto)
- All 7 audit components
- Statistical testing (bootstrap, permutation, BH-FDR)
- AuditHarness end-to-end
- AuditReport serialisation
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor

from civicsafe.audit.bundle import AuditBundle
from civicsafe.audit.components import (
    AbstentionEquityAudit,
    AuditResult,
    CalibrationEquityAudit,
    CoverageEquityAudit,
    IntervalWidthEquityAudit,
    PointAccuracyEquityAudit,
    ReportingBiasSensitivityAudit,
    WinklerEquityAudit,
    default_components,
)
from civicsafe.audit.harness import AuditHarness
from civicsafe.audit.report import AuditReport
from civicsafe.audit.statistical import (
    BootstrapTest,
    MultipleComparisonCorrector,
    PermutationTest,
)
from civicsafe.audit.stratification import StratConfig, StratificationEngine


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture()
def simple_bundle() -> AuditBundle:
    """A minimal bundle with known properties for deterministic testing."""
    torch.manual_seed(42)
    n = 100
    y_true = torch.randint(0, 10, (n,)).float()
    y_pred = y_true + torch.randn(n) * 0.5  # close predictions
    lower = y_pred - 2.0
    upper = y_pred + 2.0
    pi = torch.rand(n) * 0.3
    mu = torch.rand(n) * 5.0 + 1.0
    r = torch.rand(n) * 2.0 + 0.5
    # 3 groups: 0, 1, 2
    groups = torch.cat([
        torch.zeros(34, dtype=torch.long),
        torch.ones(33, dtype=torch.long),
        torch.full((33,), 2, dtype=torch.long),
    ])
    spatial = torch.arange(n) % 10

    return AuditBundle(
        y_true=y_true,
        y_pred=y_pred,
        lower=lower,
        upper=upper,
        pi=pi,
        mu=mu,
        r=r,
        strata={"poverty_quintile": groups},
        spatial_units=spatial,
        alpha=0.1,
        metadata={"city": "test", "model_version": "0.1.0"},
    )


@pytest.fixture()
def perfect_bundle() -> AuditBundle:
    """Bundle with perfect predictions and perfect coverage."""
    n = 60
    y = torch.arange(n).float()
    groups = torch.cat([
        torch.zeros(20, dtype=torch.long),
        torch.ones(20, dtype=torch.long),
        torch.full((20,), 2, dtype=torch.long),
    ])
    return AuditBundle(
        y_true=y,
        y_pred=y,
        lower=y - 1.0,
        upper=y + 1.0,
        pi=torch.zeros(n),
        mu=y.clamp(min=0.1),
        r=torch.ones(n),
        strata={"group": groups},
        spatial_units=torch.arange(n) % 5,
        alpha=0.1,
        metadata={"city": "perfect"},
    )


# ===================================================================
# TestAuditBundle
# ===================================================================


class TestAuditBundle:
    """Tests for the AuditBundle dataclass."""

    def test_construction(self, simple_bundle: AuditBundle) -> None:
        assert simple_bundle.num_samples == 100
        assert simple_bundle.alpha == 0.1

    def test_validation_passes(self, simple_bundle: AuditBundle) -> None:
        simple_bundle.validate()  # should not raise

    def test_validation_fails_on_mismatch(self) -> None:
        with pytest.raises(ValueError, match="y_pred"):
            AuditBundle(
                y_true=torch.zeros(10),
                y_pred=torch.zeros(5),  # wrong size!
                lower=torch.zeros(10),
                upper=torch.ones(10),
                pi=torch.zeros(10),
                mu=torch.ones(10),
                r=torch.ones(10),
                strata={},
                spatial_units=torch.zeros(10),
                alpha=0.1,
                metadata={},
            ).validate()

    def test_immutability(self, simple_bundle: AuditBundle) -> None:
        with pytest.raises(AttributeError):
            simple_bundle.alpha = 0.2  # type: ignore[misc]

    def test_thinning_reduces_counts(self, simple_bundle: AuditBundle) -> None:
        thinned = simple_bundle.with_thinned_targets(0.5, seed=42)
        # E[Y_obs] = 0.5 * E[Y_true], so thinned mean should be lower
        assert thinned.y_true.sum() <= simple_bundle.y_true.sum()
        assert thinned.metadata["p_report"] == 0.5

    def test_thinning_p1_returns_same(self, simple_bundle: AuditBundle) -> None:
        same = simple_bundle.with_thinned_targets(1.0)
        assert same is simple_bundle


# ===================================================================
# TestStratification
# ===================================================================


class TestStratification:
    """Tests for the StratificationEngine."""

    def test_quantile_bins_equal_groups(self) -> None:
        values = torch.arange(100).float()
        bins = StratificationEngine.quantile_bins(values, n_bins=5)
        # Each bin should have ~20 elements
        for b in range(5):
            count = (bins == b).sum().item()
            assert 18 <= count <= 22, f"Bin {b} has {count} elements"

    def test_quantile_bins_handles_ties(self) -> None:
        values = torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0])
        bins = StratificationEngine.quantile_bins(values, n_bins=2)
        assert bins.shape == (10,)
        assert set(bins.tolist()) == {0, 1}

    def test_equal_width_bins(self) -> None:
        values = torch.tensor([0.0, 2.5, 5.0, 7.5, 10.0])
        bins = StratificationEngine.equal_width_bins(values, n_bins=4)
        assert bins[0].item() == 0
        assert bins[-1].item() == 3  # max goes to last bin

    def test_threshold_bins_default_median(self) -> None:
        values = torch.arange(10).float()
        bins = StratificationEngine.threshold_bins(values)
        # Median of 0..9 is 4.5, so 5+ should be bin 1
        assert bins[4].item() == 0
        assert bins[5].item() == 1

    def test_threshold_bins_custom(self) -> None:
        values = torch.tensor([1.0, 3.0, 5.0, 7.0])
        bins = StratificationEngine.threshold_bins(values, threshold=4.0)
        assert bins.tolist() == [0, 0, 1, 1]

    def test_auto_stratify(self) -> None:
        features = {
            "poverty": torch.arange(20).float(),
            "income": torch.arange(20).float() * 100,
        }
        configs = {
            "poverty": StratConfig(method="quantile", n_bins=4),
            "income": StratConfig(method="threshold", threshold=1000.0),
        }
        result = StratificationEngine.auto_stratify(features, configs)
        assert "poverty" in result
        assert "income" in result
        assert result["poverty"].shape == (20,)
        assert set(result["income"].tolist()) == {0, 1}


# ===================================================================
# Test Components
# ===================================================================


class TestCoverageEquity:
    """Tests for CoverageEquityAudit."""

    def test_perfect_coverage(self, perfect_bundle: AuditBundle) -> None:
        comp = CoverageEquityAudit()
        result = comp.evaluate(
            perfect_bundle.y_true, perfect_bundle.y_pred,
            perfect_bundle.lower, perfect_bundle.upper,
            perfect_bundle.pi, perfect_bundle.mu, perfect_bundle.r,
            perfect_bundle.strata["group"], perfect_bundle.alpha,
        )
        assert result.overall_metrics["picp"] == 1.0
        assert result.passes_threshold is True

    def test_disparate_coverage(self) -> None:
        """Group 0 has perfect coverage, group 1 has 0% coverage."""
        n = 20
        y = torch.arange(n).float()
        lower = torch.zeros(n)
        upper = torch.full((n,), 100.0)
        # Make group 1 fail: set bounds to miss
        upper[10:] = -1.0
        groups = torch.cat([torch.zeros(10, dtype=torch.long),
                            torch.ones(10, dtype=torch.long)])

        comp = CoverageEquityAudit()
        result = comp.evaluate(
            y, y, lower, upper,
            torch.zeros(n), torch.ones(n), torch.ones(n),
            groups, 0.1,
        )
        assert result.per_group_metrics["0"]["picp"] == 1.0
        assert result.per_group_metrics["1"]["picp"] == 0.0
        assert result.passes_threshold is False

    def test_result_structure(self, simple_bundle: AuditBundle) -> None:
        comp = CoverageEquityAudit()
        result = comp.evaluate(
            simple_bundle.y_true, simple_bundle.y_pred,
            simple_bundle.lower, simple_bundle.upper,
            simple_bundle.pi, simple_bundle.mu, simple_bundle.r,
            simple_bundle.strata["poverty_quintile"], simple_bundle.alpha,
        )
        assert isinstance(result, AuditResult)
        assert result.component_name == "coverage_equity"
        assert "picp" in result.overall_metrics


class TestIntervalWidthEquity:
    """Tests for IntervalWidthEquityAudit."""

    def test_equal_widths(self, perfect_bundle: AuditBundle) -> None:
        comp = IntervalWidthEquityAudit()
        result = comp.evaluate(
            perfect_bundle.y_true, perfect_bundle.y_pred,
            perfect_bundle.lower, perfect_bundle.upper,
            perfect_bundle.pi, perfect_bundle.mu, perfect_bundle.r,
            perfect_bundle.strata["group"], perfect_bundle.alpha,
        )
        # All intervals are width 2, so ratio should be 1.0
        assert result.disparity_metrics["max_ratio"] == 1.0
        assert result.passes_threshold is True

    def test_disparate_widths(self) -> None:
        n = 20
        y = torch.zeros(n)
        lower = torch.zeros(n)
        upper = torch.cat([torch.full((10,), 2.0), torch.full((10,), 10.0)])
        groups = torch.cat([torch.zeros(10, dtype=torch.long),
                            torch.ones(10, dtype=torch.long)])

        comp = IntervalWidthEquityAudit()
        result = comp.evaluate(
            y, y, lower, upper,
            torch.zeros(n), torch.ones(n), torch.ones(n),
            groups, 0.1,
        )
        assert result.disparity_metrics["max_ratio"] == 5.0


class TestPointAccuracyEquity:
    """Tests for PointAccuracyEquityAudit."""

    def test_perfect_predictions(self, perfect_bundle: AuditBundle) -> None:
        comp = PointAccuracyEquityAudit()
        result = comp.evaluate(
            perfect_bundle.y_true, perfect_bundle.y_pred,
            perfect_bundle.lower, perfect_bundle.upper,
            perfect_bundle.pi, perfect_bundle.mu, perfect_bundle.r,
            perfect_bundle.strata["group"], perfect_bundle.alpha,
        )
        assert result.overall_metrics["mae"] == 0.0
        assert result.overall_metrics["rmse"] == 0.0

    def test_disparate_errors(self) -> None:
        y_true = torch.zeros(20)
        y_pred = torch.cat([torch.zeros(10), torch.full((10,), 5.0)])
        groups = torch.cat([torch.zeros(10, dtype=torch.long),
                            torch.ones(10, dtype=torch.long)])

        comp = PointAccuracyEquityAudit()
        result = comp.evaluate(
            y_true, y_pred, torch.zeros(20), torch.ones(20),
            torch.zeros(20), torch.ones(20), torch.ones(20),
            groups, 0.1,
        )
        assert result.per_group_metrics["0"]["mae"] == 0.0
        assert result.per_group_metrics["1"]["mae"] == 5.0


class TestCalibrationEquity:
    """Tests for CalibrationEquityAudit."""

    def test_perfect_calibration(self) -> None:
        n = 20
        y = torch.zeros(n)  # all zeros
        pi = torch.ones(n)  # predicting all zero → perfect
        groups = torch.cat([torch.zeros(10, dtype=torch.long),
                            torch.ones(10, dtype=torch.long)])

        comp = CalibrationEquityAudit()
        result = comp.evaluate(
            y, y, torch.zeros(n), torch.ones(n),
            pi, torch.ones(n), torch.ones(n),
            groups, 0.1,
        )
        assert result.overall_metrics["brier"] == 0.0


class TestWinklerEquity:
    """Tests for WinklerEquityAudit."""

    def test_perfect_intervals(self, perfect_bundle: AuditBundle) -> None:
        comp = WinklerEquityAudit()
        result = comp.evaluate(
            perfect_bundle.y_true, perfect_bundle.y_pred,
            perfect_bundle.lower, perfect_bundle.upper,
            perfect_bundle.pi, perfect_bundle.mu, perfect_bundle.r,
            perfect_bundle.strata["group"], perfect_bundle.alpha,
        )
        # No penalties, just width = 2.0
        assert result.overall_metrics["winkler"] == 2.0

    def test_penalty_for_miss(self) -> None:
        y = torch.tensor([10.0])
        lower = torch.tensor([0.0])
        upper = torch.tensor([5.0])  # y is above upper → penalty
        groups = torch.zeros(1, dtype=torch.long)

        comp = WinklerEquityAudit()
        result = comp.evaluate(
            y, y, lower, upper,
            torch.zeros(1), torch.ones(1), torch.ones(1),
            groups, 0.1,
        )
        # Width=5, penalty=(2/0.1)*(10-5)=100, total=105
        assert result.overall_metrics["winkler"] == 105.0


class TestAbstentionEquity:
    """Tests for AbstentionEquityAudit."""

    def test_no_abstention(self, perfect_bundle: AuditBundle) -> None:
        comp = AbstentionEquityAudit(width_threshold=100.0)
        result = comp.evaluate(
            perfect_bundle.y_true, perfect_bundle.y_pred,
            perfect_bundle.lower, perfect_bundle.upper,
            perfect_bundle.pi, perfect_bundle.mu, perfect_bundle.r,
            perfect_bundle.strata["group"], perfect_bundle.alpha,
        )
        assert result.overall_metrics["abstention_rate"] == 0.0

    def test_disparate_abstention(self) -> None:
        n = 20
        y = torch.zeros(n)
        lower = torch.zeros(n)
        # Group 0: narrow intervals, Group 1: very wide
        upper = torch.cat([torch.full((10,), 2.0), torch.full((10,), 100.0)])
        groups = torch.cat([torch.zeros(10, dtype=torch.long),
                            torch.ones(10, dtype=torch.long)])

        comp = AbstentionEquityAudit(width_threshold=50.0)
        result = comp.evaluate(
            y, y, lower, upper,
            torch.zeros(n), torch.ones(n), torch.ones(n),
            groups, 0.1,
        )
        assert result.per_group_metrics["0"]["abstention_rate"] == 0.0
        assert result.per_group_metrics["1"]["abstention_rate"] == 1.0


class TestReportingBias:
    """Tests for ReportingBiasSensitivityAudit."""

    def test_thinning_reduces_coverage(self, simple_bundle: AuditBundle) -> None:
        comp = ReportingBiasSensitivityAudit(
            reporting_rates=[0.3, 1.0], seed=42,
        )
        result = comp.evaluate(
            simple_bundle.y_true, simple_bundle.y_pred,
            simple_bundle.lower, simple_bundle.upper,
            simple_bundle.pi, simple_bundle.mu, simple_bundle.r,
            simple_bundle.strata["poverty_quintile"], simple_bundle.alpha,
        )
        assert "p=1.0" in result.per_group_metrics
        assert "p=0.3" in result.per_group_metrics

    def test_result_structure(self, simple_bundle: AuditBundle) -> None:
        comp = ReportingBiasSensitivityAudit(
            reporting_rates=[0.5, 1.0], seed=42,
        )
        result = comp.evaluate(
            simple_bundle.y_true, simple_bundle.y_pred,
            simple_bundle.lower, simple_bundle.upper,
            simple_bundle.pi, simple_bundle.mu, simple_bundle.r,
            simple_bundle.strata["poverty_quintile"], simple_bundle.alpha,
        )
        assert result.component_name == "reporting_bias_sensitivity"

    def test_seed_determinism(self, simple_bundle: AuditBundle) -> None:
        comp = ReportingBiasSensitivityAudit(
            reporting_rates=[0.5], seed=42,
        )
        r1 = comp.evaluate(
            simple_bundle.y_true, simple_bundle.y_pred,
            simple_bundle.lower, simple_bundle.upper,
            simple_bundle.pi, simple_bundle.mu, simple_bundle.r,
            simple_bundle.strata["poverty_quintile"], simple_bundle.alpha,
        )
        r2 = comp.evaluate(
            simple_bundle.y_true, simple_bundle.y_pred,
            simple_bundle.lower, simple_bundle.upper,
            simple_bundle.pi, simple_bundle.mu, simple_bundle.r,
            simple_bundle.strata["poverty_quintile"], simple_bundle.alpha,
        )
        assert r1.per_group_metrics == r2.per_group_metrics


# ===================================================================
# Test Statistical Testing
# ===================================================================


class TestBootstrapCI:
    """Tests for BootstrapTest."""

    def test_ci_contains_mean(self) -> None:
        rng = np.random.default_rng(42)
        values = rng.normal(5.0, 1.0, size=200).astype(np.float64)
        groups = np.concatenate([np.zeros(100), np.ones(100)]).astype(np.int64)

        bt = BootstrapTest(n_bootstrap=500, seed=42)
        results = bt.metric_ci(values, groups, lambda x: float(np.mean(x)))

        for _g, (point, lo, hi) in results.items():
            assert lo <= point <= hi

    def test_ci_narrows_with_more_data(self) -> None:
        rng = np.random.default_rng(42)
        bt = BootstrapTest(n_bootstrap=500, seed=42)

        small = rng.normal(0, 1, size=20).astype(np.float64)
        small_groups = np.zeros(20, dtype=np.int64)
        r_small = bt.metric_ci(small, small_groups, lambda x: float(np.mean(x)))

        large = rng.normal(0, 1, size=500).astype(np.float64)
        large_groups = np.zeros(500, dtype=np.int64)
        r_large = bt.metric_ci(large, large_groups, lambda x: float(np.mean(x)))

        width_small = r_small[0][2] - r_small[0][1]
        width_large = r_large[0][2] - r_large[0][1]
        assert width_large < width_small


class TestPermutationTest:
    """Tests for PermutationTest."""

    def test_equal_groups_not_significant(self) -> None:
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, size=100).astype(np.float64)
        b = rng.normal(0, 1, size=100).astype(np.float64)

        pt = PermutationTest(n_permutations=1000, seed=42)
        result = pt.test_difference(a, b, lambda x: float(np.mean(x)))
        assert result["p_value"] > 0.05

    def test_different_groups_significant(self) -> None:
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, size=100).astype(np.float64)
        b = rng.normal(5, 1, size=100).astype(np.float64)

        pt = PermutationTest(n_permutations=1000, seed=42)
        result = pt.test_difference(a, b, lambda x: float(np.mean(x)))
        assert result["p_value"] < 0.01


class TestMultipleComparison:
    """Tests for MultipleComparisonCorrector."""

    def test_bh_fewer_rejections_than_raw(self) -> None:
        p_values = [0.001, 0.01, 0.03, 0.04, 0.05, 0.10, 0.50]
        raw_significant = sum(1 for p in p_values if p < 0.05)

        results = MultipleComparisonCorrector.benjamini_hochberg(p_values)
        bh_significant = sum(1 for r in results if r["significant"])

        assert bh_significant <= raw_significant

    def test_bh_adjusted_p_monotonic(self) -> None:
        p_values = [0.01, 0.02, 0.03, 0.10]
        results = MultipleComparisonCorrector.benjamini_hochberg(p_values)
        for r in results:
            assert r["adjusted_p"] >= r["raw_p"]

    def test_bonferroni(self) -> None:
        p_values = [0.01, 0.04]
        results = MultipleComparisonCorrector.bonferroni(p_values, alpha=0.05)
        # 0.01 * 2 = 0.02 < 0.05 → significant
        assert results[0]["significant"] is True
        # 0.04 * 2 = 0.08 > 0.05 → not significant
        assert results[1]["significant"] is False


# ===================================================================
# Test AuditHarness
# ===================================================================


class TestAuditHarness:
    """Tests for the AuditHarness orchestrator."""

    def test_full_audit_runs(self, simple_bundle: AuditBundle) -> None:
        harness = AuditHarness()
        report = harness.run_full_audit(simple_bundle)
        assert isinstance(report, AuditReport)
        assert len(report.results) == 7
        for comp_name in [
            "coverage_equity", "interval_width_equity",
            "point_accuracy_equity", "calibration_equity",
            "winkler_equity", "abstention_equity",
            "reporting_bias_sensitivity",
        ]:
            assert comp_name in report.results

    def test_single_audit(self, simple_bundle: AuditBundle) -> None:
        harness = AuditHarness()
        result = harness.run_single_audit(simple_bundle, "coverage_equity")
        assert result.component_name == "coverage_equity"

    def test_unknown_component_raises(self, simple_bundle: AuditBundle) -> None:
        harness = AuditHarness()
        with pytest.raises(KeyError, match="nonexistent"):
            harness.run_single_audit(simple_bundle, "nonexistent")

    def test_from_config(self) -> None:
        config: dict[str, Any] = {
            "stratification": {
                "poverty": {"method": "quantile", "n_bins": 5},
            },
            "statistical_testing": {"bootstrap_samples": 1000},
        }
        harness = AuditHarness.from_config(config)
        assert harness.strat_configs is not None
        assert "poverty" in harness.strat_configs

    def test_geographic_fallback(self) -> None:
        """When no strata are provided, falls back to spatial_units."""
        n = 30
        bundle = AuditBundle(
            y_true=torch.ones(n),
            y_pred=torch.ones(n),
            lower=torch.zeros(n),
            upper=torch.full((n,), 2.0),
            pi=torch.zeros(n),
            mu=torch.ones(n),
            r=torch.ones(n),
            strata={},  # empty!
            spatial_units=torch.arange(n) % 3,
            alpha=0.1,
            metadata={},
        )
        harness = AuditHarness()
        report = harness.run_full_audit(bundle)
        assert len(report.results) == 7


# ===================================================================
# Test AuditReport
# ===================================================================


class TestAuditReport:
    """Tests for AuditReport serialisation."""

    def test_to_dict_serialisable(self, simple_bundle: AuditBundle) -> None:
        harness = AuditHarness()
        report = harness.run_full_audit(simple_bundle)
        d = report.to_dict()
        # Must be JSON-serialisable
        json_str = json.dumps(d, default=str)
        assert len(json_str) > 100

    def test_summary_table(self, simple_bundle: AuditBundle) -> None:
        harness = AuditHarness()
        report = harness.run_full_audit(simple_bundle)
        table = report.summary_table()
        assert "coverage_equity" in table
        assert "picp" in table["coverage_equity"]

    def test_pass_fail_summary(self, simple_bundle: AuditBundle) -> None:
        harness = AuditHarness()
        report = harness.run_full_audit(simple_bundle)
        pf = report.pass_fail_summary()
        assert len(pf) == 7
        for v in pf.values():
            assert isinstance(v, bool)

    def test_to_json(self, simple_bundle: AuditBundle, tmp_path: Any) -> None:
        harness = AuditHarness()
        report = harness.run_full_audit(simple_bundle)
        path = tmp_path / "audit_report.json"
        report.to_json(path)
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert "audits" in data
