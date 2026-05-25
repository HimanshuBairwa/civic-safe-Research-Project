# CIVIC-SAFE: Audited Conformal Crime Forecasting

[![CI](https://github.com/civic-safe/civic-safe/actions/workflows/ci.yml/badge.svg)](https://github.com/civic-safe/civic-safe/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

CIVIC-SAFE is an audited spatiotemporal crime forecasting benchmark with:

- **Uncertainty-aware prediction intervals** via 5 conformal calibration procedures
- **Dual-granularity fairness audit** (geographic + demographic equity)
- **Reporting-bias sensitivity analysis** via binomial deconvolution
- **Advisory safe-route reference** with Pareto-optimal routing and abstention

## Ethics Commitments

1. **Civilian-facing only** — no police dashboards, no patrol allocation
2. **No person-level prediction** — spatial-temporal aggregates only
3. **No protected attributes as inputs** — used only as evaluation strata
4. **Predicts reported crime, not committed crime** — reporting-bias sensitivity is mandatory
5. **Advisory only** — never autonomous
6. **Abstains under diagnostic failure** — with audited equity of abstention

## Installation

```bash
# Clone the repository
git clone https://github.com/civic-safe/civic-safe.git
cd civic-safe

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

## Project Structure

```
src/civicsafe/
├── data/           # Data loading, harmonization, crosswalks
├── models/         # Spatiotemporal ZINB forecaster
├── calibration/    # 5 conformal calibration procedures
├── audit/          # 7 audit components + harness
├── routing/        # 4D Pareto routing reference
├── utils/          # Seeding, logging, checkpointing, numerics
└── synthetic/      # Synthetic data generators for testing
```

## Quick Start

```bash
# Run the test suite
pytest

# Run smoke tests only
pytest -m smoke

# Run with coverage
pytest --cov=src/civicsafe
```

## Configuration

CIVIC-SAFE uses [Hydra](https://hydra.cc/) for configuration management. Configs are in `configs/`.

```bash
# Run an experiment with default config
python -m civicsafe.train

# Override parameters
python -m civicsafe.train model.spatial.hidden_dim=256 training.lr=0.0005
```

## License

MIT License. See [LICENSE](LICENSE) for details.
