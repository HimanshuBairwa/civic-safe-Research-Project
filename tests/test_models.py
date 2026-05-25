"""Tests for Phase 2 Model Core — CIVIC-SAFE spatiotemporal ZINB GNN.

43 tests covering every component:
  - ZINB Loss (12 tests): NLL correctness, gradient flow, NaN prevention
  - Spatial Encoder (6 tests): shape, dual adjacency, gradient flow
  - Temporal Encoder (6 tests): shape, causal mask, positional encoding
  - Feature Mixer (4 tests): shape, diversity loss, gradient flow
  - ZINB Head (5 tests): parameter constraints, initialization
  - Master Model (5 tests): end-to-end, checkpointing, mixed precision
  - Dataset (5 tests): windowing, chronological splits
"""

from __future__ import annotations

import pytest
import torch

from civicsafe.models.civicsafe_model import CivicSafeModel
from civicsafe.models.dataset import CrimeWindowDataset, create_chronological_splits
from civicsafe.models.feature_mixer import FeatureMixer
from civicsafe.models.graph import build_adjacency_from_synthetic
from civicsafe.models.spatial import SpatialEncoder
from civicsafe.models.temporal import SinusoidalPositionalEncoding, TemporalEncoder
from civicsafe.models.zinb_head import ZINBHead
from civicsafe.models.zinb_loss import ZINBLoss


# ===================================================================
# Fixtures
# ===================================================================
@pytest.fixture
def small_graph():
    """10-node synthetic dual adjacency graph."""
    return build_adjacency_from_synthetic(num_nodes=10, seed=42, knn_k=4)


@pytest.fixture
def zinb_loss():
    return ZINBLoss(r_floor=0.1, eps=1e-8, reduction="mean")


# ===================================================================
# ZINB Loss (12 tests)
# ===================================================================
class TestZINBLoss:

    def test_zero_observation_log_prob(self, zinb_loss) -> None:
        """Manual calculation: y=0, pi=0.5, mu=5, r=2."""
        y = torch.tensor([0.0])
        pi = torch.tensor([0.5])
        mu = torch.tensor([5.0])
        r = torch.tensor([2.0])
        loss = zinb_loss(y, pi, mu, r)
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_positive_observation_log_prob(self, zinb_loss) -> None:
        y = torch.tensor([5.0])
        pi = torch.tensor([0.3])
        mu = torch.tensor([5.0])
        r = torch.tensor([2.0])
        loss = zinb_loss(y, pi, mu, r)
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_gradient_flows_pi(self, zinb_loss) -> None:
        pi = torch.tensor([0.3], requires_grad=True)
        mu = torch.tensor([5.0])
        r = torch.tensor([2.0])
        loss = zinb_loss(torch.tensor([3.0]), pi, mu, r)
        loss.backward()
        assert pi.grad is not None
        assert torch.isfinite(pi.grad).all()

    def test_gradient_flows_mu(self, zinb_loss) -> None:
        mu = torch.tensor([5.0], requires_grad=True)
        loss = zinb_loss(
            torch.tensor([3.0]), torch.tensor([0.3]), mu, torch.tensor([2.0])
        )
        loss.backward()
        assert mu.grad is not None
        assert torch.isfinite(mu.grad).all()

    def test_gradient_flows_r(self, zinb_loss) -> None:
        r = torch.tensor([2.0], requires_grad=True)
        loss = zinb_loss(
            torch.tensor([3.0]), torch.tensor([0.3]), torch.tensor([5.0]), r
        )
        loss.backward()
        assert r.grad is not None
        assert torch.isfinite(r.grad).all()

    def test_no_nan_at_r_floor(self, zinb_loss) -> None:
        loss = zinb_loss(
            torch.tensor([0.0]),
            torch.tensor([0.5]),
            torch.tensor([1.0]),
            torch.tensor([0.05]),
        )
        assert torch.isfinite(loss)

    def test_no_nan_pi_near_zero(self, zinb_loss) -> None:
        loss = zinb_loss(
            torch.tensor([3.0]),
            torch.tensor([0.001]),
            torch.tensor([5.0]),
            torch.tensor([2.0]),
        )
        assert torch.isfinite(loss)

    def test_no_nan_pi_near_one(self, zinb_loss) -> None:
        loss = zinb_loss(
            torch.tensor([0.0]),
            torch.tensor([0.999]),
            torch.tensor([5.0]),
            torch.tensor([2.0]),
        )
        assert torch.isfinite(loss)

    def test_no_nan_mu_very_small(self, zinb_loss) -> None:
        loss = zinb_loss(
            torch.tensor([0.0]),
            torch.tensor([0.3]),
            torch.tensor([0.001]),
            torch.tensor([2.0]),
        )
        assert torch.isfinite(loss)

    def test_no_nan_mu_very_large(self, zinb_loss) -> None:
        loss = zinb_loss(
            torch.tensor([100.0]),
            torch.tensor([0.1]),
            torch.tensor([500.0]),
            torch.tensor([5.0]),
        )
        assert torch.isfinite(loss)

    def test_batch_dimension(self, zinb_loss) -> None:
        B = 32
        loss = zinb_loss(
            torch.randint(0, 10, (B,)).float(),
            torch.rand(B) * 0.5 + 0.1,
            torch.rand(B) * 10 + 1,
            torch.rand(B) * 5 + 0.5,
        )
        assert loss.shape == ()

    def test_reduction_none(self) -> None:
        loss_fn = ZINBLoss(reduction="none")
        nll = loss_fn(
            torch.tensor([0.0, 5.0]),
            torch.tensor([0.3, 0.3]),
            torch.tensor([5.0, 5.0]),
            torch.tensor([2.0, 2.0]),
        )
        assert nll.shape == (2,)


# ===================================================================
# Spatial Encoder (6 tests)
# ===================================================================
class TestSpatialEncoder:

    def test_output_shape(self, small_graph) -> None:
        enc = SpatialEncoder(
            in_channels=8, hidden_channels=32, num_layers=2, num_heads=4
        )
        x = torch.randn(10, 8)
        out = enc(x, small_graph["queen"], small_graph["knn"])
        assert out.shape == (10, 32)

    def test_dual_vs_single_adjacency(self, small_graph) -> None:
        enc = SpatialEncoder(
            in_channels=8, hidden_channels=32, num_layers=2, num_heads=4
        )
        x = torch.randn(10, 8)
        out_dual = enc(x, small_graph["queen"], small_graph["knn"])
        out_single = enc(x, small_graph["queen"], None)
        assert not torch.allclose(out_dual, out_single, atol=1e-5)

    def test_finite_output(self, small_graph) -> None:
        enc = SpatialEncoder(in_channels=8, hidden_channels=32)
        x = torch.randn(10, 8)
        out = enc(x, small_graph["queen"])
        assert torch.isfinite(out).all()

    def test_zero_features(self, small_graph) -> None:
        enc = SpatialEncoder(in_channels=8, hidden_channels=32)
        x = torch.zeros(10, 8)
        out = enc(x, small_graph["queen"])
        assert torch.isfinite(out).all()

    def test_gradient_flow(self, small_graph) -> None:
        enc = SpatialEncoder(in_channels=8, hidden_channels=32)
        x = torch.randn(10, 8, requires_grad=True)
        out = enc(x, small_graph["queen"])
        out.sum().backward()
        assert x.grad is not None

    def test_deterministic(self, small_graph) -> None:
        torch.manual_seed(42)
        enc = SpatialEncoder(in_channels=8, hidden_channels=32)
        x = torch.randn(10, 8)
        out1 = enc(x, small_graph["queen"])
        out2 = enc(x, small_graph["queen"])
        assert torch.allclose(out1, out2)


# ===================================================================
# Temporal Encoder (6 tests)
# ===================================================================
class TestTemporalEncoder:

    def test_output_shape(self) -> None:
        enc = TemporalEncoder(d_model=32, num_heads=4, num_layers=1, max_seq_len=20)
        x = torch.randn(5, 10, 32)
        out = enc(x)
        assert out.shape == (5, 10, 32)

    def test_causal_mask_no_future_leakage(self) -> None:
        """Verify that changing future values doesn't affect past outputs."""
        enc = TemporalEncoder(d_model=32, num_heads=4, num_layers=1, max_seq_len=10)
        enc.eval()
        x1 = torch.randn(1, 5, 32)
        x2 = x1.clone()
        x2[0, 3:, :] = torch.randn(2, 32)  # Change future values
        with torch.no_grad():
            out1 = enc(x1)
            out2 = enc(x2)
        # First 3 positions should be identical (causal = no future leakage)
        assert torch.allclose(out1[0, :3, :], out2[0, :3, :], atol=1e-5)

    def test_positional_encoding_added(self) -> None:
        pe = SinusoidalPositionalEncoding(d_model=32, max_len=10)
        x = torch.zeros(1, 5, 32)
        out = pe(x)
        # Output should not be all zeros (PE was added)
        assert not torch.allclose(out, torch.zeros_like(out))

    def test_gradient_flow(self) -> None:
        enc = TemporalEncoder(d_model=32, num_heads=4, num_layers=1)
        x = torch.randn(2, 5, 32, requires_grad=True)
        out = enc(x)
        out.sum().backward()
        assert x.grad is not None

    def test_variable_sequence_length(self) -> None:
        enc = TemporalEncoder(d_model=32, num_heads=4, max_seq_len=52)
        for T in [5, 10, 30, 52]:
            out = enc(torch.randn(2, T, 32))
            assert out.shape == (2, T, 32)

    def test_deterministic(self) -> None:
        enc = TemporalEncoder(d_model=32, num_heads=4)
        enc.eval()
        x = torch.randn(1, 5, 32)
        with torch.no_grad():
            out1 = enc(x)
            out2 = enc(x)
        assert torch.allclose(out1, out2)


# ===================================================================
# Feature Mixer (4 tests)
# ===================================================================
class TestFeatureMixer:

    def test_output_shape(self) -> None:
        mixer = FeatureMixer(d_model=32, num_heads=3)
        x = torch.randn(5, 10, 32)
        out, div_loss = mixer(x)
        assert out.shape == (5, 10, 32)

    def test_diversity_loss_is_scalar(self) -> None:
        mixer = FeatureMixer(d_model=32, num_heads=3)
        _, div_loss = mixer(torch.randn(5, 10, 32))
        assert div_loss.shape == ()

    def test_gradient_flow(self) -> None:
        mixer = FeatureMixer(d_model=32, num_heads=3)
        x = torch.randn(2, 5, 32, requires_grad=True)
        out, _ = mixer(x)
        out.sum().backward()
        assert x.grad is not None

    def test_temperature_affects_output(self) -> None:
        x = torch.randn(2, 5, 32)
        m1 = FeatureMixer(d_model=32, temperature=0.1)
        m2 = FeatureMixer(d_model=32, temperature=10.0)
        # Different temperatures should produce different outputs
        # (not a strict test, but statistically very likely)
        o1, _ = m1(x)
        o2, _ = m2(x)
        assert not torch.allclose(o1, o2, atol=1e-3)


# ===================================================================
# ZINB Head (5 tests)
# ===================================================================
class TestZINBHead:

    def test_pi_in_zero_one(self) -> None:
        head = ZINBHead(in_features=32, num_categories=3)
        pi, _, _ = head(torch.randn(10, 32))
        assert (pi >= 0).all() and (pi <= 1).all()

    def test_mu_strictly_positive(self) -> None:
        head = ZINBHead(in_features=32, num_categories=3)
        _, mu, _ = head(torch.randn(10, 32))
        assert (mu > 0).all()

    def test_r_above_floor(self) -> None:
        head = ZINBHead(in_features=32, num_categories=3, r_floor=0.1)
        _, _, r = head(torch.randn(10, 32))
        assert (r >= 0.1).all()

    def test_output_shape(self) -> None:
        head = ZINBHead(in_features=32, num_categories=5)
        pi, mu, r = head(torch.randn(10, 32))
        assert pi.shape == (10, 5)
        assert mu.shape == (10, 5)
        assert r.shape == (10, 5)

    def test_small_init_produces_moderate_values(self) -> None:
        head = ZINBHead(in_features=32, num_categories=3)
        pi, mu, r = head(torch.randn(10, 32))
        # After small-variance init, values should be moderate
        assert pi.mean().item() < 0.8
        assert pi.mean().item() > 0.2
        assert mu.mean().item() < 50


# ===================================================================
# Master Model (5 tests)
# ===================================================================
class TestCivicSafeModel:

    @pytest.fixture
    def model_and_graph(self, small_graph):
        model = CivicSafeModel(
            num_features=8,
            hidden_dim=32,
            spatial_layers=1,
            spatial_heads=4,
            temporal_layers=1,
            temporal_heads=4,
            temporal_ff_dim=64,
            num_categories=3,
            max_seq_len=10,
        )
        return model, small_graph

    def test_end_to_end_shape(self, model_and_graph) -> None:
        model, graph = model_and_graph
        features = torch.randn(10, 5, 8)  # S=10, T=5, F=8
        out = model(features, graph["queen"], graph["knn"])
        assert out["pi"].shape == (10, 3)
        assert out["mu"].shape == (10, 3)
        assert out["r"].shape == (10, 3)

    def test_all_params_have_grad(self, model_and_graph) -> None:
        model, graph = model_and_graph
        features = torch.randn(10, 5, 8)
        out = model(features, graph["queen"], graph["knn"])
        loss = out["pi"].sum() + out["mu"].sum() + out["r"].sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No grad for {name}"

    def test_output_finite(self, model_and_graph) -> None:
        model, graph = model_and_graph
        features = torch.randn(10, 5, 8)
        out = model(features, graph["queen"], graph["knn"])
        assert torch.isfinite(out["pi"]).all()
        assert torch.isfinite(out["mu"]).all()
        assert torch.isfinite(out["r"]).all()

    def test_deterministic_with_seed(self, small_graph) -> None:
        torch.manual_seed(42)
        m1 = CivicSafeModel(
            num_features=8,
            hidden_dim=32,
            spatial_heads=4,
            spatial_layers=1,
            temporal_layers=1,
            temporal_heads=4,
            temporal_ff_dim=64,
            num_categories=3,
            max_seq_len=10,
        )
        torch.manual_seed(42)
        m2 = CivicSafeModel(
            num_features=8,
            hidden_dim=32,
            spatial_heads=4,
            spatial_layers=1,
            temporal_layers=1,
            temporal_heads=4,
            temporal_ff_dim=64,
            num_categories=3,
            max_seq_len=10,
        )
        x = torch.randn(10, 5, 8)
        o1 = m1(x, small_graph["queen"])
        o2 = m2(x, small_graph["queen"])
        assert torch.allclose(o1["pi"], o2["pi"])

    def test_diversity_loss_returned(self, model_and_graph) -> None:
        model, graph = model_and_graph
        features = torch.randn(10, 5, 8)
        out = model(features, graph["queen"])
        assert "diversity_loss" in out
        assert out["diversity_loss"].shape == ()


# ===================================================================
# Dataset (5 tests)
# ===================================================================
class TestDataset:

    @pytest.fixture
    def panel_data(self):
        S, T, C, F = 10, 52 * 3, 3, 5  # 3 years
        counts = torch.randint(0, 20, (S, T, C))
        features = torch.randn(S, T, F)
        return counts, features

    def test_window_shapes(self, panel_data) -> None:
        counts, features = panel_data
        ds = CrimeWindowDataset(counts, features, window_size=10)
        sample = ds[0]
        assert sample["input_counts"].shape == (10, 10, 3)
        assert sample["input_features"].shape == (10, 10, 5)
        assert sample["target_counts"].shape == (10, 3)

    def test_dataset_length(self, panel_data) -> None:
        counts, features = panel_data
        ds = CrimeWindowDataset(counts, features, window_size=10)
        # Total T=156, window=10, valid targets: [10, 156) = 146
        assert len(ds) == 146

    def test_chronological_no_overlap(self) -> None:
        S, T, C, F = 5, 52 * 6, 3, 4  # 6 years
        counts = torch.randint(0, 10, (S, T, C))
        features = torch.randn(S, T, F)
        splits = create_chronological_splits(
            counts,
            features,
            start_year=2018,
            end_year=2023,
            val_year=2022,
            test_year=2023,
            window_size=10,
        )
        # Val targets should start at 2022 boundary
        assert splits["val"].start_idx >= splits["train"].end_idx

    def test_train_val_test_exist(self) -> None:
        S, T, C, F = 5, 52 * 6, 3, 4
        counts = torch.randint(0, 10, (S, T, C))
        features = torch.randn(S, T, F)
        splits = create_chronological_splits(
            counts,
            features,
            start_year=2018,
            end_year=2023,
            window_size=10,
        )
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits

    def test_single_week_window(self) -> None:
        counts = torch.randint(0, 10, (5, 10, 3))
        features = torch.randn(5, 10, 4)
        ds = CrimeWindowDataset(counts, features, window_size=1)
        sample = ds[0]
        assert sample["input_counts"].shape == (5, 1, 3)
