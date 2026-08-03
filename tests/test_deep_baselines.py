"""Regression tests for the deep-learning baselines in scripts/deep_baselines.py.

These are competitor models (LSTM-NB, TFT-ZINB, GraphWaveNet, STZINB-GNN) used
for the head-to-head comparison table. They are not part of the installed
package, so they are loaded from scripts/ by path.

The einsum test below exists because a wrong-axis contraction in STZINB-GNN's
spatial diffusion crashed three full training runs (~1000s each) *after* the
other three baselines had already trained, and the exception escaped main() so
their results were discarded too.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


@pytest.fixture(scope="module")
def db():
    """Load scripts/deep_baselines.py as a module."""
    spec = importlib.util.spec_from_file_location(
        "deep_baselines", SCRIPTS_DIR / "deep_baselines.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# The spatial-diffusion contraction
# ---------------------------------------------------------------------------
def test_spatial_diffusion_contracts_the_node_axis() -> None:
    """'ij,bjwh->biwh' must diffuse over nodes, matching an explicit spatial sum.

    This is the correctness half of the bug. The shipped form was
    'ij,bsjh->bsih', which binds j to the WINDOW axis and therefore diffuses
    across time using a spatial adjacency. Compare against a hand-rolled sum
    over neighbours so the assertion does not just restate the einsum.
    """
    B, S, W, H = 2, 6, 4, 3
    torch.manual_seed(0)
    x = torch.randn(B, S, W, H)
    adj = torch.rand(S, S)
    adj = adj / adj.sum(1, keepdim=True)

    got = torch.einsum("ij,bjwh->biwh", adj, x)

    expected = torch.zeros_like(got)
    for b in range(B):
        for i in range(S):
            for j in range(S):
                expected[b, i] += adj[i, j] * x[b, j]

    assert torch.allclose(got, expected, atol=1e-6)


def test_wrong_axis_contraction_is_silently_different_when_S_equals_W() -> None:
    """Negative control: at S == W the old form ran and produced wrong numbers.

    This is why the bug is worse than a crash. With the default window_size of
    52 and a node count near 52, the buggy einsum type-checks, trains, and
    reports plausible metrics that are simply wrong.
    """
    B, S, W, H = 2, 5, 5, 3  # S == W: both forms are shape-legal
    torch.manual_seed(0)
    x = torch.randn(B, S, W, H)
    adj = torch.rand(S, S)
    adj = adj / adj.sum(1, keepdim=True)

    correct = torch.einsum("ij,bjwh->biwh", adj, x)
    buggy = torch.einsum("ij,bsjh->bsih", adj, x)

    assert correct.shape == buggy.shape  # indistinguishable by shape alone
    assert not torch.allclose(correct, buggy, atol=1e-4)


def test_stzinb_forward_survives_S_not_equal_W(db) -> None:
    """The exact configuration that crashed: 77 nodes, 52-week window."""
    B, S, W, C, F = 2, 77, 52, 3, 8
    model = db.STZINBGNNModel(num_nodes=S, input_dim=C + F, num_categories=C)
    counts = torch.log1p(torch.randint(0, 60, (B, S, W, C)).float())
    features = torch.randn(B, S, W, F)
    adj = torch.rand(S, S)
    adj = adj / adj.sum(1, keepdim=True)

    pi, mu, r = model(counts, features, static_adj=adj)

    assert pi.shape == mu.shape == r.shape == (B, S, C)
    assert torch.isfinite(mu).all()
    assert (pi >= 0).all() and (pi <= 1).all()
    assert (mu > 0).all()
    assert (r > 0).all()


# ---------------------------------------------------------------------------
# All four baselines: shapes, finite loss, real gradients
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("name", "is_graph"),
    [
        ("LSTM_NB", False),
        ("TFT_ZINB", False),
        ("GraphWaveNet", True),
        ("STZINB_GNN", True),
    ],
)
def test_baseline_forward_backward_at_real_shapes(db, name: str, is_graph: bool) -> None:
    """Every baseline must emit head-shaped params and produce usable gradients.

    Mirrors the pre-flight check main() now runs before training, at the real
    Chicago panel shape with S != W so a wrong-axis contraction cannot hide.
    """
    from civicsafe.training.metrics import crps_zinb

    S, W, C, F = 77, 52, 3, 8
    B = 2
    builders = {
        "LSTM_NB": lambda: db.LSTMNBModel(input_dim=C + F, num_categories=C),
        "TFT_ZINB": lambda: db.SimplifiedTFTModel(count_dim=C, feature_dim=F),
        "GraphWaveNet": lambda: db.GraphWaveNetModel(
            num_nodes=S, input_dim=C + F, num_categories=C
        ),
        "STZINB_GNN": lambda: db.STZINBGNNModel(
            num_nodes=S, input_dim=C + F, num_categories=C
        ),
    }
    model = builders[name]()

    if is_graph:
        counts = torch.log1p(torch.randint(0, 60, (B, S, W, C)).float())
        features = torch.randn(B, S, W, F)
        target = torch.randint(0, 60, (B, S, C)).float()
        adj = torch.rand(S, S)
        adj = adj / adj.sum(1, keepdim=True)
        pi, mu, r = model(counts, features, static_adj=adj)
    else:
        counts = torch.log1p(torch.randint(0, 60, (B * S, W, C)).float())
        features = torch.randn(B * S, W, F)
        target = torch.randint(0, 60, (B * S, C)).float()
        pi, mu, r = model(counts, features)

    assert pi.shape == mu.shape == r.shape == target.shape

    loss = (
        db.nb_nll_loss(target, mu, r)
        if name == "LSTM_NB"
        else crps_zinb(target, pi, mu, r).mean()
    )
    assert torch.isfinite(loss), f"{name} produced a non-finite loss"

    loss.backward()
    with_grad = [
        p
        for p in model.parameters()
        if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    ]
    assert with_grad, f"{name} produced no usable gradients -- it would not train"
