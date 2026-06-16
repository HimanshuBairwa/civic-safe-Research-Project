# CIVIC-SAFE Analysis Log — June 13, 2026

## 1. Codebase Re-Audit Findings

### Files Audited

| Module | Status | Discrepancies Found |
|--------|:------:|---------------------|
| `src/civicsafe/models/zinb_loss.py` | ✓ Correct | r_floor=0.1 via clamp, no regularization (now fixed in trainer) |
| `src/civicsafe/models/spatial.py` | ✓ Correct | GATv2Conv, dual adjacency, LayerNorm. No degree normalization issue — GATv2 handles normalization internally via attention |
| `src/civicsafe/training/trainer.py` | **BUG FOUND** | `fit()` never called `save_checkpoint()` — root cause of conformal eval crash. **FIXED.** |
| `src/civicsafe/training/early_stopping.py` | ✓ Correct | Saves weights to RAM, restores on completion. Now paired with disk save in trainer. |
| `configs/training/default.yaml` | **Config Gap** | patience=10 was too low for 200-epoch cosine schedule. **FIXED: patience=30.** |
| `scripts/baselines.py` | **Missing Baseline** | No seasonal-naive baseline. **FIXED: Added seasonal-naive + lag-1 persistence.** |
| `scripts/run_conformal_evaluation.py` | **Single Baseline** | CRPSS computed only vs HA. **FIXED: Now computes vs both HA and seasonal-naive.** |
| `src/civicsafe/utils/seeding.py` | ✓ Correct | Seeds torch, numpy, python random. |
| `configs/model/spatiotemporal_zinb.yaml` | ✓ Correct | 128 hidden, 2 layers, 4 heads. |

---

## 2. Deep Analysis of Training Output

### 2.1 Training Completion Summary

| City | Seeds | Best Epoch | Early Stop Epoch | Total Epochs Used |
|------|:-----:|:----------:|:----------------:|:-----------------:|
| NYC (seed 512) | ✓ | 42 | 52 | 52/200 |
| NYC (seed 1024) | ✓ | 42 | 52 | 52/200 |
| Chicago | ✓ (prior run) | 42 | 52 | 52/200 |

**Critical observation:** ALL seeds across BOTH cities converge to epoch 42 as best. This is NOT coincidence — it's a structural artifact of the LR schedule + patience interaction (see §A3 in deep_analysis).

### 2.2 Final Metrics (NYC, 5-seed aggregate)

| Metric | Mean | Std |
|--------|:----:|:---:|
| **CRPS** | 16.5602 | 0.0982 |
| **MAE** | 21.5676 | 0.0709 |
| **RMSE** | 35.1220 | 0.1009 |
| **Brier Zero** | 0.1289 | 0.0046 |

### 2.3 r-Collapse Pattern (Bug 1)

**Still occurring.** Evidence from NYC seed 512:
- Epoch 42 (best): CRPS=16.539, MAE=21.560
- Epoch 52 (stopped): CRPS=17.392 (+0.853), MAE=20.695 (−0.865)

The CRPS/MAE divergence after epoch 42 is **the exact signature of r-collapse**:
- MAE improves because the model sharpens the point prediction (μ parameter)
- CRPS degrades because the model destroys distributional calibration (r→∞)
- The loss function has no penalty for r-collapse

**Fix applied:** Added per-cell r-floor regularization to `trainer.py` (Opus formula).

### 2.4 Conformal Evaluation Results

| City | Status | Error |
|------|:------:|-------|
| Chicago | **CRASHED** | `FileNotFoundError: No checkpoint files found` |
| NYC | **CRASHED** | `FileNotFoundError: No checkpoint files found` |

**Root cause:** `trainer.fit()` saves weights to RAM via EarlyStopping but NEVER writes `.pt` files to disk. The conformal evaluation script searches for `best.pt` files and finds none.

**Fix applied:** Added `self.save_checkpoint(ckpt_path, epoch, val_metrics)` call in `fit()` on every EarlyStopping improvement.

### 2.5 CRPSS Assessment

Not yet computable against real trained model (checkpoints didn't exist). Quick-train test showed:
- Model CRPS: 15.48 (2-epoch model)
- HA Baseline CRPS: 3.88
- CRPSS vs HA: −2.99 (expected — model barely trained)

**Gap identified:** Need to also compute CRPSS against seasonal-naive baseline.
**Fix applied:** `compute_baseline_crps()` now returns both HA and seasonal-naive CRPS.

### 2.6 New Errors and Warnings

| Error | Severity | Fix |
|-------|:--------:|-----|
| Flash Attention non-determinism warning | LOW | Add `torch.use_deterministic_algorithms(True)` for publication |
| ACS API fallback to synthetic demographics | MEDIUM | User has Census API key; will work with real key set |
| No checkpoint saved to disk | **CRITICAL** | **FIXED** |

### 2.7 Publication Floor Asserts

| Criterion | Target | Current Status |
|-----------|:------:|:--------------:|
| CRPSS ≥ 0.10 (vs seasonal-naive) | ≥ 0.10 | **UNKNOWN** (need retrain with r-reg) |
| Coverage within ±1pp of nominal | 0.89–0.91 | ✓ 0.9003 (from quick-train test) |
| Disparity < 3pp | < 0.03 | ✓ 0.0142 (ECRC method) |

---

## 3. Conflict Resolutions

### Conflict A: r-floor regularization formula
- **Fable:** `relu(r_floor - r.mean())` — penalizes batch mean only
- **Opus:** `relu(r_floor - r).mean()` — penalizes each cell individually
- **Resolution:** Opus (implemented). Fable's version lets 10% of cells collapse while batch mean stays safe.
- **Implementation:** `trainer.py` lines 312-316

### Conflict B: Degree normalization
- **Fable:** 1/sqrt(degree_i) target-node only
- **Opus:** 1/sqrt(degree_i * degree_j) symmetric GCN-style
- **Resolution:** NOT APPLICABLE. We use GATv2Conv which has its own attention-based normalization. Neither manual normalization is needed. GATv2's attention mechanism (Brody et al., 2022) computes per-edge attention weights that implicitly normalize message aggregation.

### Conflict C: Root cause of negative CRPSS
- **Fable:** Single bug (r-collapse)
- **Opus:** Triple structural mismatch: (1) NLL≠CRPS, (2) 19:1 param ratio, (3) seasonality dominates
- **Resolution:** Opus's framing is correct. r-collapse is necessary-but-not-sufficient. The r-regularization fix addresses (1) partially. Full fix requires CRPS-blended objective (future work) and beating seasonal-naive specifically.

### Conflict D: Baseline strength
- **Fable:** Only Historical Average
- **Opus:** Must include Seasonal Naive
- **Resolution:** Opus (implemented). Added seasonal-naive and lag-1 persistence to `baselines.py`.

---

## 4. Fixes Applied in This Session

| Fix | File | Lines Changed | Bug Addressed |
|-----|------|:------------:|---------------|
| Checkpoint saving | `trainer.py` | +4 | Conformal eval crash |
| r-floor regularization | `trainer.py` | +8 | CRPS degradation post-epoch 42 |
| r-reg config | `default.yaml` | +7 | Config for r-reg |
| Patience 10→30 | `default.yaml` | +2 | Premature early stopping |
| Seasonal-naive baseline | `baselines.py` | +45 | Missing strongest baseline |
| Lag-1 persistence baseline | `baselines.py` | +16 | Missing persistence baseline |
| Dual-baseline CRPSS | `run_conformal_evaluation.py` | +30 | CRPSS only vs HA |
| Dual-baseline audit report | `run_conformal_evaluation.py` | +5 | Report only showed HA |
