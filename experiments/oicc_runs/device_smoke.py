"""Device-agnostic end-to-end TRAINING smoke test (A100-critical path).

The existing tests exercise model math on CPU. This test exercises the exact
GPU training path that would run on an A100 -- moving the model and data to the
detected device, a bfloat16 autocast forward, a float32 ZINB loss, backward, and
an optimizer step -- on whatever device is available. On this box it runs on CPU;
on an A100 it runs on cuda with bf16 autocast, proving there is no device/dtype/
shape error in the real training loop before a long run is launched.

Run:
    python experiments/oicc_runs/device_smoke.py
It prints the device it used and asserts finiteness + gradient flow. Exits 0 on
success, non-zero on any error -- so it is a safe pre-flight check on the A100.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from civicsafe.models.civicsafe_model import CivicSafeModel  # noqa: E402
from civicsafe.models.zinb_loss import ZINBLoss              # noqa: E402


def _tiny_graph(S: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """A small ring + a couple of chords, as (2, E) edge_index tensors."""
    src = list(range(S))
    dst = [(i + 1) % S for i in range(S)]
    # symmetric ring
    ei = torch.tensor([src + dst, dst + src], dtype=torch.long, device=device)
    # a knn-like second graph = ring skip-1
    dst2 = [(i + 2) % S for i in range(S)]
    ek = torch.tensor([src + dst2, dst2 + src], dtype=torch.long, device=device)
    return ei, ek


def run(verbose: bool = True) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32

    torch.manual_seed(0)
    S, T, F, C = 12, 6, 8, 3

    model = CivicSafeModel(
        num_features=F, hidden_dim=32, spatial_layers=1, spatial_heads=4,
        temporal_layers=1, temporal_heads=4, temporal_ff_dim=64,
        num_categories=C, max_seq_len=T,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    features = torch.randn(S, T, F, device=device)
    counts = torch.randint(0, 5, (S, C), device=device).float()
    edge_queen, edge_knn = _tiny_graph(S, device)

    model.train()
    opt.zero_grad()
    # exact A100 path: bf16 autocast forward, float32 loss
    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=amp_enabled):
        out = model(features, edge_queen, edge_knn)
    pi = out["pi"].float()
    mu = out["mu"].float()
    r = out["r"].float()
    loss = ZINBLoss(reduction="mean")(counts, pi, mu, r)
    loss.backward()
    grad_ok = all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters() if p.requires_grad
    )
    opt.step()

    finite = bool(torch.isfinite(loss) and torch.isfinite(pi).all()
                  and torch.isfinite(mu).all() and torch.isfinite(r).all())
    result = {
        "device": str(device),
        "amp_dtype": str(amp_dtype),
        "loss": float(loss.detach()),
        "outputs_finite": finite,
        "grads_finite": bool(grad_ok),
        "cuda_available": torch.cuda.is_available(),
    }
    if verbose:
        print("device-agnostic training smoke:")
        for k, v in result.items():
            print(f"  {k}: {v}")
    assert finite, "non-finite model outputs/loss"
    assert grad_ok, "missing or non-finite gradients"
    if verbose:
        print("PASS: full training path runs on", device)
    return result


if __name__ == "__main__":
    run()
