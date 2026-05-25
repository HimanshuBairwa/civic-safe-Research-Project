"""
CIVIC-SAFE Jupyter Server Setup & Verification Script.
Checks Python version, detects A100 CUDA capabilities, installs dependencies,
runs the pytest suite programmatically, and runs a GPU tensor benchmark.
"""
from __future__ import annotations

import os
import sys
import subprocess
import shutil

def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(f" {title.upper()} ".center(80, "="))
    print("=" * 80)

def print_success(msg: str) -> None:
    print(f"[+] SUCCESS: {msg}")

def print_warning(msg: str) -> None:
    print(f"[!] WARNING: {msg}")

def print_info(msg: str) -> None:
    print(f"[*] INFO: {msg}")

def main() -> None:
    print_header("CIVIC-SAFE environment diagnostics")

    # 1. Check Python Version
    print_info(f"Python Executable: {sys.executable}")
    print_info(f"Python Version: {sys.version}")
    if sys.version_info < (3, 11):
        print_warning("Python 3.11+ is recommended for CIVIC-SAFE. Proceeding anyway.")
    else:
        print_success("Python version satisfies requirement (>=3.11).")

    # 2. Check if running inside project root
    current_dir = os.getcwd()
    print_info(f"Current Directory: {current_dir}")
    pyproject_exists = os.path.exists("pyproject.toml")
    if not pyproject_exists:
        print_warning("Could not find pyproject.toml in the current directory.")
        print_warning("Please make sure you run this script from the 'PCC project' root directory.")
        return
    print_success("Located pyproject.toml. Running inside project root.")

    # 3. Dynamic Dependency Installation
    print_header("checking and installing dependencies")
    print_info("Installing CIVIC-SAFE dependencies in editable dev mode...")
    try:
        # Run pip install from within Python
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-e", ".[dev]"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )
        print_success("All project dependencies installed successfully.")
    except Exception as e:
        print_warning(f"Editable pip install failed: {e}")
        print_info("Attempting fallback direct installation of core packages...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "torch", "pandas", "geopandas", "hydra-core", "pytest", "rich"],
                stdout=subprocess.DEVNULL
            )
            print_success("Fallback package installation succeeded.")
        except Exception as fallback_err:
            print_warning(f"Fallback installation failed: {fallback_err}")
            print_warning("Please manually run: !pip install -e .[dev] in a notebook cell.")

    # 4. Check PyTorch and A100 GPU Acceleration
    print_header("cuda & gpu diagnostics")
    try:
        import torch
        print_success(f"PyTorch imported successfully (Version: {torch.__version__})")
        
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            device_cap = torch.cuda.get_device_capability(0)
            print_success(f"CUDA is AVAILABLE.")
            print_success(f"GPU Device 0: {device_name}")
            print_success(f"CUDA Compute Capability: {device_cap[0]}.{device_cap[1]}")
            
            # Simple GPU Speed Warmup Test
            print_info("Running float32 matrix multiplication warmup on GPU...")
            x = torch.randn(5000, 5000, device="cuda")
            y = torch.randn(5000, 5000, device="cuda")
            
            # Record time for multiplication
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            z = torch.matmul(x, y)
            end_event.record()
            
            torch.cuda.synchronize()
            elapsed_time_ms = start_event.elapsed_time(end_event)
            print_success(f"GPU MatMul (5000x5000) took: {elapsed_time_ms:.2f} ms")
        else:
            print_warning("CUDA is NOT available to PyTorch. Training will run on CPU.")
            print_info("Note: For high-speed A100 usage, ensure PyTorch matches the CUDA drivers on your server.")
    except ImportError:
        print_warning("PyTorch is not installed. Skipping GPU diagnostics.")

    # 5. Programmatic Pytest Runner
    print_header("running test suite")
    try:
        import pytest
        print_info("Launching 30-test suite via pytest...")
        
        # We run pytest programmatically pointing to the tests/ directory
        exit_code = pytest.main(["tests/", "-v", "--tb=short"])
        
        if exit_code == 0:
            print_success("All 30 tests PASSED successfully!")
        else:
            print_warning(f"Some tests failed (Exit code: {exit_code}). See output above for details.")
    except ImportError:
        print_warning("pytest is not installed. Cannot run tests programmatically.")

    print_header("diagnostics complete")
    print_info("If the tests passed and CUDA was detected, your Jupyter server is 100% ready!")
    print_info("We are ready to proceed with Phase 1 Data Ingestion and Harmonization.")

if __name__ == "__main__":
    main()
