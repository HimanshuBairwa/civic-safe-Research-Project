#!/usr/bin/env python
"""One-command preflight + run entry point for CIVIC-SAFE + OICC on any box.

Runs a deterministic sequence and reports a single PASS/FAIL, so on a fresh A100
you can do `python run_all.py` and know in one shot whether the whole codebase is
healthy on that machine:

  1. environment report (python, numpy, torch, CUDA, PyG)
  2. device-agnostic training smoke (the real GPU forward/backward path)
  3. OICC test suite (fast, no data/GPU needed)
  4. OICC headline reproduction (asserts 13 numbers)
  5. (optional) the full civicsafe test suite with --full

Usage:
    python run_all.py            # preflight: env + smoke + oicc tests + reproduce
    python run_all.py --full     # also run the entire civicsafe test suite
    python run_all.py --env-only # just print the environment report
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
ENV = {"PYTHONPATH": str(SRC)}


def _run(cmd: list[str], label: str) -> bool:
    print(f"\n{'=' * 70}\n>>> {label}\n{'=' * 70}")
    import os
    env = {**os.environ, **ENV, "MPLBACKEND": "Agg", "WANDB_MODE": "disabled"}
    r = subprocess.run(cmd, cwd=str(ROOT), env=env)
    ok = r.returncode == 0
    print(f"<<< {label}: {'PASS' if ok else 'FAIL (rc=%d)' % r.returncode}")
    return ok


def env_report() -> None:
    print("=" * 70)
    print("ENVIRONMENT")
    print("=" * 70)
    print("python:", sys.version.split()[0])
    for mod in ("numpy", "scipy", "pandas"):
        try:
            m = __import__(mod)
            print(f"{mod}:", m.__version__)
        except Exception as e:  # pragma: no cover
            print(f"{mod}: NOT INSTALLED ({e})")
    try:
        import torch
        print("torch:", torch.__version__, "| CUDA build:", torch.version.cuda,
              "| CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
    except Exception as e:  # pragma: no cover
        print("torch: NOT INSTALLED (", e, ")")
    try:
        import torch_geometric as g
        print("torch_geometric:", g.__version__)
    except Exception as e:  # pragma: no cover
        print("torch_geometric: NOT INSTALLED (", e, ")")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="also run the full civicsafe test suite")
    ap.add_argument("--env-only", action="store_true")
    args = ap.parse_args()

    env_report()
    if args.env_only:
        return 0

    results: list[tuple[str, bool]] = []
    py = sys.executable

    results.append((
        "device-agnostic training smoke",
        _run([py, "experiments/oicc_runs/device_smoke.py"],
             "training smoke (real forward/backward on the detected device)"),
    ))
    results.append((
        "OICC test suite",
        _run([py, "-m", "pytest", "tests_oicc/", "-q", "--no-header",
              "-p", "no:cacheprovider"], "OICC test suite"),
    ))
    results.append((
        "OICC reproduction",
        _run([py, "experiments/oicc_runs/reproduce_all.py", "--quick"],
             "OICC headline reproduction (machine-checked assertions)"),
    ))
    if args.full:
        results.append((
            "full civicsafe suite",
            _run([py, "-m", "pytest", "tests/", "-q", "--no-header",
                  "-p", "no:cacheprovider"], "full civicsafe test suite"),
        ))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    all_ok = all(ok for _, ok in results)
    print("=" * 70)
    print("ALL GREEN" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
