"""Pytest wrapper for the device-agnostic training smoke test.

Runs the full GNN training path (forward + bf16 autocast + ZINB loss + backward
+ optimizer step) on whatever device is present. On CPU here; on an A100 pytest
run it exercises the real cuda + bfloat16 path, catching any device/dtype/shape
error before a long training run.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "experiments" / "oicc_runs"))

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")


def test_full_training_path_on_device():
    from device_smoke import run
    res = run(verbose=False)
    assert res["outputs_finite"] is True
    assert res["grads_finite"] is True
    # loss must be a finite positive-ish NLL
    assert res["loss"] == res["loss"]  # not NaN


def test_runs_on_cuda_if_available():
    """If a GPU is present (A100), the smoke path must use it without error."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device on this machine")
    from device_smoke import run
    res = run(verbose=False)
    assert res["device"].startswith("cuda")
    assert res["amp_dtype"] == "torch.bfloat16"
    assert res["outputs_finite"] and res["grads_finite"]
