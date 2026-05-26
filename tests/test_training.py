"""Tests for Phase 3 Training Infrastructure.

Covers:
  - CRPS correctness (12 tests): known values, edge cases, consistency
  - Point metrics (4 tests): MAE, RMSE, Brier Score
  - EarlyStopping (6 tests): patience, restoration, min_delta, reset
  - Scheduler (4 tests): warmup, cosine decay, state dict
  - Trainer smoke test (2 tests): 2-epoch train + val on synthetic data
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from civicsafe.training.early_stopping import EarlyStopping
from civicsafe.training.metrics import (
    brier_zero_inflation,
    compute_all_metrics,
    crps_zinb,
    mae_zinb,
    pit_values,
    rmse_zinb,
)
from civicsafe.training.scheduler import CosineWarmupScheduler, create_cosine_warmup_scheduler


# ===================================================================
# CRPS Tests (12 tests)
# ===================================================================
class TestCRPS:

    def test_crps_perfect_prediction(self) -> None:
        """CRPS = 0 when prediction is a point mass at the observed value."""
        # When pi=0, mu matches y exactly, and r→∞ (high concentration),
        # the ZINB collapses to a point mass → CRPS should be near 0.
        y = torch.tensor([5.0])
        pi = torch.tensor([0.0])
        mu = torch.tensor([5.0])
        r = torch.tensor([1000.0])  # Very concentrated
        crps = crps_zinb(y, pi, mu, r, k_max=50)
        # Should be small (not exactly 0 due to discrete CDF steps)
        assert crps.item() < 1.0

    def test_crps_zero_observation_with_zero_inflation(self) -> None:
        """CRPS for y=0 with high zero-inflation should be small."""
        y = torch.tensor([0.0])
        pi = torch.tensor([0.9])  # High zero-inflation
        mu = torch.tensor([1.0])
        r = torch.tensor([2.0])
        crps = crps_zinb(y, pi, mu, r, k_max=50)
        assert torch.isfinite(crps)
        assert crps.item() >= 0

    def test_crps_always_nonnegative(self) -> None:
        """CRPS is always ≥ 0 for any valid parameters."""
        B = 100
        y = torch.randint(0, 20, (B,)).float()
        pi = torch.rand(B) * 0.5
        mu = torch.rand(B) * 10 + 1
        r = torch.rand(B) * 5 + 0.5
        crps = crps_zinb(y, pi, mu, r)
        assert (crps >= 0).all()

    def test_crps_finite_for_edge_cases(self) -> None:
        """CRPS should be finite for extreme parameter values."""
        cases = [
            (0.0, 0.0, 0.001, 0.1),   # Very small mu
            (0.0, 0.99, 10.0, 0.1),    # High zero-inflation
            (100.0, 0.0, 100.0, 10.0), # Large count
            (0.0, 0.5, 5.0, 100.0),    # Large dispersion
        ]
        for y_val, pi_val, mu_val, r_val in cases:
            y = torch.tensor([y_val])
            pi = torch.tensor([pi_val])
            mu = torch.tensor([mu_val])
            r = torch.tensor([r_val])
            crps = crps_zinb(y, pi, mu, r)
            assert torch.isfinite(crps), f"CRPS not finite for y={y_val}, pi={pi_val}, mu={mu_val}, r={r_val}"

    def test_crps_batch_consistency(self) -> None:
        """Batch computation should match element-wise computation."""
        B = 10
        y = torch.randint(0, 10, (B,)).float()
        pi = torch.rand(B) * 0.5
        mu = torch.rand(B) * 5 + 1
        r = torch.rand(B) * 3 + 0.5

        # Batch
        crps_batch = crps_zinb(y, pi, mu, r, k_max=100)

        # Element-wise
        crps_single = torch.stack([
            crps_zinb(y[i:i+1], pi[i:i+1], mu[i:i+1], r[i:i+1], k_max=100)
            for i in range(B)
        ]).squeeze()

        assert torch.allclose(crps_batch, crps_single, atol=1e-4)

    def test_crps_increases_with_worse_prediction(self) -> None:
        """CRPS should increase when prediction is farther from truth."""
        y = torch.tensor([5.0])
        pi = torch.tensor([0.0])
        r = torch.tensor([5.0])

        crps_good = crps_zinb(y, pi, torch.tensor([5.0]), r, k_max=100)
        crps_bad = crps_zinb(y, pi, torch.tensor([20.0]), r, k_max=100)
        assert crps_bad > crps_good

    def test_crps_multidimensional(self) -> None:
        """CRPS should work with (B, C) shaped inputs."""
        B, C = 8, 3
        y = torch.randint(0, 10, (B, C)).float()
        pi = torch.rand(B, C) * 0.5
        mu = torch.rand(B, C) * 5 + 1
        r = torch.rand(B, C) * 3 + 0.5
        crps = crps_zinb(y, pi, mu, r, k_max=50)
        assert crps.shape == (B, C)

    def test_crps_zero_inflation_effect(self) -> None:
        """Higher pi should reduce CRPS for y=0."""
        y = torch.tensor([0.0])
        mu = torch.tensor([5.0])
        r = torch.tensor([2.0])

        crps_low_pi = crps_zinb(y, torch.tensor([0.1]), mu, r, k_max=100)
        crps_high_pi = crps_zinb(y, torch.tensor([0.8]), mu, r, k_max=100)
        assert crps_high_pi < crps_low_pi

    def test_crps_auto_k_max(self) -> None:
        """Auto k_max selection should produce valid results."""
        y = torch.tensor([10.0])
        pi = torch.tensor([0.1])
        mu = torch.tensor([10.0])
        r = torch.tensor([2.0])
        crps = crps_zinb(y, pi, mu, r)  # No k_max specified
        assert torch.isfinite(crps)
        assert crps.item() > 0

    def test_crps_deterministic(self) -> None:
        """CRPS should be deterministic (no randomness)."""
        y = torch.tensor([3.0])
        pi = torch.tensor([0.3])
        mu = torch.tensor([5.0])
        r = torch.tensor([2.0])
        crps1 = crps_zinb(y, pi, mu, r, k_max=50)
        crps2 = crps_zinb(y, pi, mu, r, k_max=50)
        assert torch.allclose(crps1, crps2)

    def test_crps_higher_k_max_more_accurate(self) -> None:
        """Increasing k_max should converge (larger is at least as accurate)."""
        y = torch.tensor([10.0])
        pi = torch.tensor([0.2])
        mu = torch.tensor([10.0])
        r = torch.tensor([2.0])
        crps_50 = crps_zinb(y, pi, mu, r, k_max=50)
        crps_200 = crps_zinb(y, pi, mu, r, k_max=200)
        # Values should be close (the CDF truncation matters less at higher k)
        assert abs(crps_50.item() - crps_200.item()) < 1.0

    def test_crps_gpu_cpu_consistency(self) -> None:
        """CRPS on CPU and GPU (if available) should match."""
        y = torch.tensor([5.0])
        pi = torch.tensor([0.3])
        mu = torch.tensor([5.0])
        r = torch.tensor([2.0])
        crps_cpu = crps_zinb(y, pi, mu, r, k_max=50)

        if torch.cuda.is_available():
            crps_gpu = crps_zinb(
                y.cuda(), pi.cuda(), mu.cuda(), r.cuda(), k_max=50
            )
            assert torch.allclose(crps_cpu, crps_gpu.cpu(), atol=1e-4)


# ===================================================================
# Point Metrics Tests (4 tests)
# ===================================================================
class TestPointMetrics:

    def test_mae_perfect(self) -> None:
        """MAE = 0 when E[Y] = y exactly."""
        y = torch.tensor([5.0, 10.0, 0.0])
        pi = torch.tensor([0.0, 0.0, 1.0])  # pi=1 → E[Y]=0
        mu = torch.tensor([5.0, 10.0, 999.0])  # mu doesn't matter when pi=1
        mae = mae_zinb(y, pi, mu)
        assert mae.item() < 1e-5

    def test_rmse_always_nonnegative(self) -> None:
        """RMSE ≥ 0 always."""
        B = 50
        rmse = rmse_zinb(
            torch.randint(0, 10, (B,)).float(),
            torch.rand(B) * 0.5,
            torch.rand(B) * 10 + 1,
        )
        assert rmse.item() >= 0

    def test_brier_perfect_calibration(self) -> None:
        """Brier = 0 when pi perfectly predicts zero/nonzero."""
        y = torch.tensor([0.0, 5.0, 0.0, 3.0])
        pi = torch.tensor([1.0, 0.0, 1.0, 0.0])
        brier = brier_zero_inflation(y, pi)
        assert brier.item() < 1e-5

    def test_brier_worst_case(self) -> None:
        """Brier = 1 when predictions are maximally wrong."""
        y = torch.tensor([0.0, 5.0])
        pi = torch.tensor([0.0, 1.0])  # Opposite of truth
        brier = brier_zero_inflation(y, pi)
        assert abs(brier.item() - 1.0) < 1e-5


# ===================================================================
# PIT Tests (2 tests)
# ===================================================================
class TestPIT:

    def test_pit_in_unit_interval(self) -> None:
        """PIT values should be in [0, 1]."""
        B = 50
        pit = pit_values(
            torch.randint(0, 10, (B,)).float(),
            torch.rand(B) * 0.5,
            torch.rand(B) * 5 + 1,
            torch.rand(B) * 3 + 0.5,
        )
        assert (pit >= 0).all()
        assert (pit <= 1).all()

    def test_pit_shape(self) -> None:
        """PIT output shape matches input."""
        B = 20
        y = torch.randint(0, 5, (B,)).float()
        pit = pit_values(y, torch.rand(B)*0.3, torch.rand(B)*5+1, torch.rand(B)*2+0.5)
        assert pit.shape == (B,)


# ===================================================================
# EarlyStopping Tests (6 tests)
# ===================================================================
class TestEarlyStopping:

    def _dummy_model(self) -> nn.Module:
        return nn.Linear(10, 5)

    def test_no_stop_while_improving(self) -> None:
        """Should not stop if metric keeps improving."""
        es = EarlyStopping(patience=3, mode="min")
        model = self._dummy_model()
        for i in range(10):
            stopped = es.step(10.0 - i, epoch=i, model=model)
            assert not stopped

    def test_stop_after_patience(self) -> None:
        """Should stop after patience epochs without improvement."""
        es = EarlyStopping(patience=3, mode="min")
        model = self._dummy_model()
        es.step(1.0, epoch=0, model=model)  # New best
        es.step(1.5, epoch=1, model=model)  # Worse
        es.step(1.5, epoch=2, model=model)  # Worse
        stopped = es.step(1.5, epoch=3, model=model)  # 3rd failure → stop
        assert stopped

    def test_best_epoch_tracked(self) -> None:
        """Should track which epoch had the best metric."""
        es = EarlyStopping(patience=5, mode="min")
        model = self._dummy_model()
        es.step(5.0, epoch=0, model=model)
        es.step(3.0, epoch=1, model=model)
        es.step(4.0, epoch=2, model=model)
        assert es.best_epoch == 1
        assert abs(es.best_score - 3.0) < 1e-6

    def test_restore_best_weights(self) -> None:
        """Should restore the model to its best-epoch state."""
        es = EarlyStopping(patience=2, mode="min")
        model = nn.Linear(5, 3)

        # Save initial weights
        with torch.no_grad():
            model.weight.fill_(1.0)
        es.step(1.0, epoch=0, model=model)

        # Change weights
        with torch.no_grad():
            model.weight.fill_(99.0)
        es.step(5.0, epoch=1, model=model)
        es.step(5.0, epoch=2, model=model)

        # Restore should bring back the epoch-0 weights
        es.restore_best_weights(model)
        assert torch.allclose(model.weight, torch.ones_like(model.weight))

    def test_min_delta_respected(self) -> None:
        """Tiny improvements below min_delta should not count."""
        es = EarlyStopping(patience=2, min_delta=0.1, mode="min")
        model = self._dummy_model()
        es.step(1.0, epoch=0, model=model)
        es.step(0.999, epoch=1, model=model)  # Improvement < 0.1 → not counted
        es.step(0.998, epoch=2, model=model)  # Still not enough
        assert es.should_stop

    def test_reset_clears_state(self) -> None:
        """Reset should allow reuse for a new seed."""
        es = EarlyStopping(patience=2, mode="min")
        model = self._dummy_model()
        es.step(1.0, epoch=0, model=model)
        es.step(5.0, epoch=1, model=model)
        es.step(5.0, epoch=2, model=model)
        assert es.should_stop

        es.reset()
        assert not es.should_stop
        assert es.best_score is None
        assert es.counter == 0


# ===================================================================
# Scheduler Tests (4 tests)
# ===================================================================
class TestScheduler:

    def test_warmup_starts_at_zero(self) -> None:
        """LR should start near 0 during warmup."""
        model = nn.Linear(10, 5)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = create_cosine_warmup_scheduler(opt, warmup_steps=10, total_steps=100)
        scheduler.step()
        lr = opt.param_groups[0]["lr"]
        assert lr < 2e-4  # Should be ~1e-4 at step 1 (1/10 * 1e-3)

    def test_warmup_reaches_peak(self) -> None:
        """LR should reach peak at end of warmup."""
        model = nn.Linear(10, 5)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = create_cosine_warmup_scheduler(opt, warmup_steps=10, total_steps=100)
        for _ in range(10):
            scheduler.step()
        lr = opt.param_groups[0]["lr"]
        assert abs(lr - 1e-3) < 1e-5

    def test_cosine_decays(self) -> None:
        """LR should decay after warmup."""
        model = nn.Linear(10, 5)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = create_cosine_warmup_scheduler(
            opt, warmup_steps=10, total_steps=100, min_lr=1e-6
        )
        for _ in range(50):
            scheduler.step()
        lr_mid = opt.param_groups[0]["lr"]
        assert lr_mid < 1e-3  # Should be decayed
        assert lr_mid > 1e-6  # But not at minimum yet

    def test_scheduler_state_dict(self) -> None:
        """Scheduler state should be serializable."""
        model = nn.Linear(10, 5)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sched = CosineWarmupScheduler(
            opt, warmup_epochs=2, total_epochs=10, steps_per_epoch=5
        )
        for _ in range(15):
            sched.step()
        state = sched.state_dict()
        assert isinstance(state, dict)


# ===================================================================
# Trainer Smoke Tests (2 tests)
# ===================================================================
class TestTrainerSmoke:

    @pytest.fixture
    def tiny_setup(self):
        """Create a tiny model + data for trainer smoke tests."""
        from civicsafe.models.civicsafe_model import CivicSafeModel
        from civicsafe.models.dataset import CrimeWindowDataset
        from civicsafe.models.graph import build_adjacency_from_synthetic

        S, T, C, F = 10, 20, 3, 5
        counts = torch.randint(0, 10, (S, T, C))
        features = torch.randn(S, T, F)
        graph = build_adjacency_from_synthetic(num_nodes=S, seed=42, knn_k=4)

        ds = CrimeWindowDataset(counts, features, window_size=5)

        edge_queen = graph["queen"]
        edge_knn = graph.get("knn")

        def collate_fn(batch):
            return {
                "input_features": torch.stack([b["input_features"] for b in batch]),
                "input_counts": torch.stack([b["input_counts"] for b in batch]),
                "target_counts": torch.stack([b["target_counts"] for b in batch]),
                "edge_queen": edge_queen,
                "edge_knn": edge_knn,
            }

        loader = torch.utils.data.DataLoader(
            ds, batch_size=4, collate_fn=collate_fn, drop_last=True
        )

        model = CivicSafeModel(
            num_features=F,
            hidden_dim=32,
            spatial_layers=1,
            spatial_heads=4,
            temporal_layers=1,
            temporal_heads=4,
            temporal_ff_dim=64,
            num_categories=C,
            max_seq_len=10,
        )

        config = {
            "training": {
                "epochs": 2,
                "optimizer": {"lr": 1e-3, "weight_decay": 1e-2, "betas": [0.9, 0.999]},
                "scheduler": {"warmup_epochs": 1, "min_lr": 1e-6},
                "gradient_clip_norm": 1.0,
                "mixed_precision": False,  # CPU doesn't support bf16
                "early_stopping": {"patience": 5, "min_delta": 1e-4, "mode": "min"},
                "diversity_lambda": 0.1,
            }
        }

        return model, loader, config

    def test_trainer_runs_without_error(self, tiny_setup) -> None:
        """Trainer should complete 2 epochs without crashing."""
        from civicsafe.training.trainer import Trainer

        model, loader, config = tiny_setup
        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            config=config,
            device="cpu",
        )
        results = trainer.fit()
        assert "history" in results
        assert "best_metrics" in results
        assert len(results["history"]["train_loss"]) == 2

    def test_trainer_loss_is_finite(self, tiny_setup) -> None:
        """Training loss should be finite at every epoch."""
        from civicsafe.training.trainer import Trainer

        model, loader, config = tiny_setup
        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            config=config,
            device="cpu",
        )
        results = trainer.fit()
        for loss in results["history"]["train_loss"]:
            assert loss > 0, "Loss should be positive (ZINB NLL)"
            assert loss < 1e6, "Loss should not explode"


# ===================================================================
# compute_all_metrics Tests (1 test)
# ===================================================================
class TestComputeAllMetrics:

    def test_returns_all_keys(self) -> None:
        """compute_all_metrics should return all expected metric keys."""
        B = 20
        metrics = compute_all_metrics(
            y=torch.randint(0, 10, (B,)).float(),
            pi=torch.rand(B) * 0.5,
            mu=torch.rand(B) * 5 + 1,
            r=torch.rand(B) * 3 + 0.5,
        )
        assert "crps" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "brier_zero" in metrics
        for v in metrics.values():
            assert isinstance(v, float)
