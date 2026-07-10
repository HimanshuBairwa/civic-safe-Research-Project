# =============================================================================
# CIVIC-SAFE + OICC -- reproducible A100 (CUDA 12.1) image.
#
# Uses the official PyTorch CUDA runtime image so torch + CUDA are guaranteed
# compatible with an A100 (sm_80). The model needs only GATv2Conv from PyG, so
# no compiled torch-scatter/torch-sparse extensions are installed (the usual GPU
# pain point is avoided by construction).
#
# Build:  docker build -t civicsafe-oicc .
# Test:   docker run --gpus all civicsafe-oicc            # runs the test suite
# Shell:  docker run --gpus all -it civicsafe-oicc bash
# Repro:  docker run --gpus all civicsafe-oicc \
#             python experiments/oicc_runs/reproduce_all.py
# =============================================================================
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# system deps for geopandas/shapely (GEOS/PROJ) -- optional but keeps the full
# repo importable; harmless if unused.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git libgeos-dev libproj-dev proj-data proj-bin \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# install python deps first (better layer caching)
COPY requirements-a100.txt .
RUN pip install --no-cache-dir -r requirements-a100.txt

# copy the project
COPY . .

# make both packages importable without PYTHONPATH gymnastics
ENV PYTHONPATH=/workspace/src
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg
# headless / offline-friendly defaults
ENV WANDB_MODE=disabled

# default: run the full test suite (proves the whole codebase runs on this box).
# GPU tests auto-detect the device; data-dependent tests skip gracefully.
CMD ["python", "-m", "pytest", "tests/", "tests_oicc/", "-q", \
     "--no-header", "-p", "no:cacheprovider"]
