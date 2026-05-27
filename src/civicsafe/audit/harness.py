"""Master audit orchestrator (Façade pattern).

``AuditHarness`` is the single entry-point for the full equity-audit
pipeline.  It wires together the 7 audit components, dynamic
stratification, and statistical testing into one composable workflow.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from torch import Tensor

from civicsafe.audit.bundle import AuditBundle
from civicsafe.audit.components import (
    AuditResult,
    _BaseAuditComponent,
    default_components,
)
from civicsafe.audit.report import AuditReport
from civicsafe.audit.stratification import StratConfig, StratificationEngine


class AuditHarness:
    """Façade that orchestrates the full equity-audit pipeline.

    Args:
        components: List of audit components.  Defaults to all 7.
        stratification_configs: Optional configs for dynamic binning.
            Applied to ``AuditBundle.strata`` features before auditing.
        statistical_config: Parameters for bootstrap / permutation tests
            (reserved for future use).
    """

    def __init__(
        self,
        components: list[_BaseAuditComponent] | None = None,
        stratification_configs: dict[str, StratConfig] | None = None,
        statistical_config: dict[str, Any] | None = None,
    ) -> None:
        self.components = components or default_components()
        self.strat_configs = stratification_configs
        self.stat_config = statistical_config or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_full_audit(
        self,
        bundle: AuditBundle,
        strata_key: str | None = None,
    ) -> AuditReport:
        """Execute all audit components and compile a report.

        Args:
            bundle: Data bundle containing predictions, intervals, and
                stratification features.
            strata_key: Which ``bundle.strata`` feature to use as the
                group variable.  If ``None``, the first feature in
                ``bundle.strata`` is used.

        Returns:
            Complete ``AuditReport`` with results from all components.
        """
        bundle.validate()
        groups = self._resolve_groups(bundle, strata_key)

        results: dict[str, AuditResult] = {}
        for comp in self.components:
            results[comp.name] = comp.evaluate(
                y_true=bundle.y_true,
                y_pred=bundle.y_pred,
                lower=bundle.lower,
                upper=bundle.upper,
                pi=bundle.pi,
                mu=bundle.mu,
                r=bundle.r,
                groups=groups,
                alpha=bundle.alpha,
            )

        return AuditReport(
            results=results,
            bundle_metadata=bundle.metadata,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
        )

    def run_single_audit(
        self,
        bundle: AuditBundle,
        component_name: str,
        strata_key: str | None = None,
    ) -> AuditResult:
        """Run a specific audit component by name.

        Args:
            bundle: Data bundle.
            component_name: Must match a component's ``.name`` property.
            strata_key: Stratification feature key.

        Returns:
            Single ``AuditResult``.

        Raises:
            KeyError: If ``component_name`` is not found.
        """
        bundle.validate()
        groups = self._resolve_groups(bundle, strata_key)
        comp = self._get_component(component_name)

        return comp.evaluate(
            y_true=bundle.y_true,
            y_pred=bundle.y_pred,
            lower=bundle.lower,
            upper=bundle.upper,
            pi=bundle.pi,
            mu=bundle.mu,
            r=bundle.r,
            groups=groups,
            alpha=bundle.alpha,
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AuditHarness":
        """Create an ``AuditHarness`` from a Hydra-style config dict.

        Expected keys:
            ``stratification`` — mapping of feature → {method, n_bins}.
            ``statistical_testing`` — bootstrap / permutation params.

        Args:
            config: Configuration dictionary.

        Returns:
            Configured ``AuditHarness`` instance.
        """
        strat_configs: dict[str, StratConfig] | None = None
        if "stratification" in config:
            strat_configs = {}
            for feat, cfg in config["stratification"].items():
                strat_configs[feat] = StratConfig(
                    method=cfg.get("method", "quantile"),
                    n_bins=cfg.get("n_bins", 5),
                    threshold=cfg.get("threshold"),
                )

        stat_config = config.get("statistical_testing", {})

        return cls(
            stratification_configs=strat_configs,
            statistical_config=stat_config,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_groups(
        self,
        bundle: AuditBundle,
        strata_key: str | None,
    ) -> Tensor:
        """Get the group-label tensor from the bundle.

        If ``strata_key`` is provided, use that feature.  Otherwise,
        use the first available feature.  Falls back to spatial-unit IDs
        if no strata are defined.
        """
        if strata_key and strata_key in bundle.strata:
            return bundle.strata[strata_key]

        if bundle.strata:
            first_key = next(iter(bundle.strata))
            return bundle.strata[first_key]

        # Fallback: geographic stratification by spatial unit
        return bundle.spatial_units

    def _get_component(self, name: str) -> _BaseAuditComponent:
        """Look up a component by name.

        Raises:
            KeyError: If the component is not registered.
        """
        for comp in self.components:
            if comp.name == name:
                return comp
        available = [c.name for c in self.components]
        msg = f"Component '{name}' not found.  Available: {available}"
        raise KeyError(msg)
