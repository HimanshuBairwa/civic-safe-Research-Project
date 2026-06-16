#!/usr/bin/env python
"""Baselines for CivicSafe spatiotemporal crime forecasting.

Evaluates:
1. Historical Average (HA)
2. STARIMA (Space-Time ARIMA via Statsmodels)
3. Zero-Inflated Negative Binomial Regression (ZINB via Statsmodels)
4. Spatiotemporal XGBoost

Usage:
    python scripts/baselines.py data=chicago
    python scripts/baselines.py data=nyc
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from xgboost import XGBRegressor

import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning

try:
    from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
except ImportError:
    ZeroInflatedNegativeBinomialP = None

from civicsafe.models.dataset import create_chronological_splits

warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', UserWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, and CRPS (equals MAE for deterministic point forecasts)."""
    mae = np.abs(y_true - y_pred).mean()
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    crps = mae  # Point forecast CRPS matches MAE
    return {"crps": float(crps), "mae": float(mae), "rmse": float(rmse)}


def get_adjacency_matrix(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Convert edge_index to row-normalized adjacency matrix."""
    adj = torch.zeros((num_nodes, num_nodes))
    adj[edge_index[0], edge_index[1]] = 1.0
    deg = adj.sum(dim=1, keepdim=True)
    adj = adj / deg.clamp(min=1.0)
    return adj


def extract_features(ds, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract tabular features from CrimeWindowDataset for baselines.
    
    Returns:
        X: (T, S, C, F) tensor of features.
        y: (T, S, C) tensor of targets.
    """
    X_list = []
    y_list = []
    
    for idx in range(len(ds)):
        item = ds[idx]
        ic = item["input_counts"]
        feat = item["input_features"]
        tc = item["target_counts"]
        
        S, W, C = ic.shape
        ar_1 = ic[:, -1, :]
        ar_2 = ic[:, -2, :] if W >= 2 else torch.zeros_like(ar_1)
        ar_3 = ic[:, -3, :] if W >= 3 else torch.zeros_like(ar_1)
        ar_mean = ic.mean(dim=1)
        
        sp_1 = torch.matmul(adj, ar_1)
        
        target_features = feat[:, -1, :]
        
        X_c = []
        for c in range(C):
            X_cat = torch.cat([
                ar_1[:, c:c+1], ar_2[:, c:c+1], ar_3[:, c:c+1], ar_mean[:, c:c+1],
                sp_1[:, c:c+1],
                target_features
            ], dim=-1)
            X_c.append(X_cat)
            
        X_list.append(torch.stack(X_c, dim=1))
        y_list.append(tc)
        
    return torch.stack(X_list, dim=0), torch.stack(y_list, dim=0)


def run_historical_average(test_ds) -> dict[str, float]:
    """1. Historical Average (HA) baseline."""
    preds = []
    targets = []
    for idx in range(len(test_ds)):
        item = test_ds[idx]
        pred = item["input_counts"].mean(dim=1).numpy()
        preds.append(pred)
        targets.append(item["target_counts"].numpy())
        
    preds = np.stack(preds)
    targets = np.stack(targets)
    return compute_metrics(targets, preds)


def run_seasonal_naive(test_ds) -> dict[str, float]:
    """Seasonal Naive: predict same week from last year (lag=52).
    
    This is the strongest naive baseline for weekly crime counts because
    ~80% of the signal is seasonal + persistence (Opus analysis).
    Y_hat(t) = Y(t-52) — the crime count from exactly 52 weeks ago.
    Requires window_size >= 52.
    
    A model that cannot beat seasonal-naive is an instant desk-reject.
    """
    preds = []
    targets = []
    for idx in range(len(test_ds)):
        item = test_ds[idx]
        # input_counts has shape (S, W, C) where W is the window size
        # The first timestep in the window is exactly W weeks ago
        # For W=52, input_counts[:, 0, :] is the same week last year
        pred = item["input_counts"][:, 0, :].numpy()  # (S, C) = lag-52
        preds.append(pred)
        targets.append(item["target_counts"].numpy())

    preds = np.stack(preds)
    targets = np.stack(targets)
    return compute_metrics(targets, preds)


def run_lag1_persistence(test_ds) -> dict[str, float]:
    """Lag-1 Persistence: predict last week's count.
    
    Y_hat(t) = Y(t-1) — the simplest autoregressive baseline.
    """
    preds = []
    targets = []
    for idx in range(len(test_ds)):
        item = test_ds[idx]
        # Last timestep in the window = lag-1
        pred = item["input_counts"][:, -1, :].numpy()  # (S, C)
        preds.append(pred)
        targets.append(item["target_counts"].numpy())

    preds = np.stack(preds)
    targets = np.stack(targets)
    return compute_metrics(targets, preds)


def run_starima(train_ds, test_ds, adj) -> dict[str, float]:
    """2. STARIMA baseline.
    
    Fits a Space-Time AR model (via OLS) per node and category.
    This exactly models an AR(1) with a spatial lag term, mapping perfectly 
    to the sliding window evaluation paradigm.
    """
    X_train, y_train = extract_features(train_ds, adj)
    X_test, y_test = extract_features(test_ds, adj)
    
    T_te, S, C, _ = X_test.shape
    preds = np.zeros((T_te, S, C))
    targets = y_test.numpy()
    
    # Indices: 0 = AR(1), 4 = Spatial Lag(1)
    features_idx = [0, 4]
    
    for c in range(C):
        for s in tqdm(range(S), desc=f"STARIMA Cat {c}", leave=False):
            y_tr_cs = y_train[:, s, c].numpy()
            exog_tr_cs = X_train[:, s, c, :].numpy()[:, features_idx]
            exog_te_cs = X_test[:, s, c, :].numpy()[:, features_idx]
            
            exog_tr_cs = sm.add_constant(exog_tr_cs, has_constant='add')
            exog_te_cs = sm.add_constant(exog_te_cs, has_constant='add')
            
            try:
                model = sm.OLS(y_tr_cs, exog_tr_cs)
                result = model.fit()
                pred_cs = result.predict(exog_te_cs)
            except Exception:
                pred_cs = np.full(T_te, y_tr_cs.mean())
                
            preds[:, s, c] = pred_cs
            
    preds = np.clip(preds, a_min=0, a_max=None)
    return compute_metrics(targets, preds)


def run_zinb(train_ds, test_ds, adj) -> dict[str, float] | None:
    """3. Zero-Inflated Negative Binomial Regression (ZINB) baseline."""
    if ZeroInflatedNegativeBinomialP is None:
        logger.warning("statsmodels ZeroInflatedNegativeBinomialP unavailable.")
        return None
        
    X_train, y_train = extract_features(train_ds, adj)
    X_test, y_test = extract_features(test_ds, adj)
    
    T_te, S, C, F_new = X_test.shape
    preds = np.zeros((T_te, S, C))
    targets = y_test.numpy()
    
    for c in range(C):
        logger.info(f"  Training ZINB for category {c}...")
        X_tr_c = X_train[:, :, c, :].reshape(-1, F_new).numpy()
        y_tr_c = y_train[:, :, c].reshape(-1).numpy()
        
        X_tr_c = sm.add_constant(X_tr_c, has_constant='add')
        X_te_c = X_test[:, :, c, :].reshape(-1, F_new).numpy()
        X_te_c = sm.add_constant(X_te_c, has_constant='add')
        
        try:
            model = ZeroInflatedNegativeBinomialP(y_tr_c, X_tr_c, exog_infl=X_tr_c)
            result = model.fit(method='bfgs', maxiter=30, disp=False)
            pred_c = result.predict(X_te_c, exog_infl=X_te_c)
        except Exception as e:
            logger.warning(f"  ZINB fallback for category {c} (failed to converge): {e}")
            try:
                model = sm.GLM(y_tr_c, X_tr_c, family=sm.families.NegativeBinomial())
                result = model.fit()
                pred_c = result.predict(X_te_c)
            except Exception:
                pred_c = np.full(X_te_c.shape[0], y_tr_c.mean())
                
        preds[:, :, c] = pred_c.reshape(T_te, S)
        
    preds = np.clip(preds, a_min=0, a_max=None)
    return compute_metrics(targets, preds)


def run_xgboost(train_ds, test_ds, adj) -> dict[str, float]:
    """4. Spatiotemporal XGBoost baseline."""
    X_train, y_train = extract_features(train_ds, adj)
    X_test, y_test = extract_features(test_ds, adj)
    
    T_te, S, C, _ = X_test.shape
    preds = np.zeros((T_te, S, C))
    targets = y_test.numpy()
    
    for c in range(C):
        logger.info(f"  Training XGBoost for category {c}...")
        model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42)
        X_tr_c = X_train[:, :, c, :].reshape(-1, X_train.shape[-1]).numpy()
        y_tr_c = y_train[:, :, c].reshape(-1).numpy()
        
        model.fit(X_tr_c, y_tr_c)
        
        X_te_c = X_test[:, :, c, :].reshape(-1, X_test.shape[-1]).numpy()
        pred_c = model.predict(X_te_c)
        preds[:, :, c] = pred_c.reshape(T_te, S)
        
    preds = np.clip(preds, a_min=0, a_max=None)
    return compute_metrics(targets, preds)


def main():
    parser = argparse.ArgumentParser(description="CIVIC-SAFE Baselines")
    parser.add_argument("args", nargs="*", help="Override configs like data=nyc")
    parsed = parser.parse_args()

    data_name = "chicago"
    for arg in parsed.args:
        if arg.startswith("data="):
            data_name = arg.split("=", 1)[1]
            
    project_root = Path(__file__).resolve().parent.parent
    panel_path = project_root / "data" / "processed" / f"{data_name}_panel.pt"
    graph_path = project_root / "data" / "processed" / f"{data_name}_graph.pt"
    
    if not panel_path.exists():
        logger.error(f"Panel data not found: {panel_path}. Run fetch_data.py first.")
        sys.exit(1)
        
    logger.info(f"Loading {data_name} dataset...")
    panel = torch.load(panel_path, weights_only=False)
    counts = panel["counts"]
    features = panel["features"]
    
    # Normalize features
    feat_mean = features.mean(dim=(0, 1), keepdim=True)
    feat_std = features.std(dim=(0, 1), keepdim=True).clamp(min=1e-6)
    features = (features - feat_mean) / feat_std
    
    # Splits (match main model exactly)
    splits = create_chronological_splits(
        counts, features,
        start_year=2018, end_year=2023,
        val_year=2022, test_year=2023,
        window_size=52
    )
    
    # Adjacency matrix for spatial lags
    if graph_path.exists():
        graph = torch.load(graph_path, weights_only=False)
        adj = get_adjacency_matrix(graph["queen"], counts.shape[0])
    else:
        logger.warning("Graph not found. Using identity matrix.")
        adj = torch.eye(counts.shape[0])
        
    results = {}
    
    # 1. Historical Average
    logger.info("Running Historical Average (HA)...")
    res_ha = run_historical_average(splits["test"])
    results["HA"] = res_ha
    logger.info(f"  HA Results: {res_ha}")

    # 2. Seasonal Naive (same week last year — the strongest naive baseline)
    logger.info("Running Seasonal Naive (lag-52)...")
    res_sn = run_seasonal_naive(splits["test"])
    results["Seasonal_Naive"] = res_sn
    logger.info(f"  Seasonal Naive Results: {res_sn}")

    # 3. Lag-1 Persistence
    logger.info("Running Lag-1 Persistence...")
    res_l1 = run_lag1_persistence(splits["test"])
    results["Lag1_Persistence"] = res_l1
    logger.info(f"  Lag-1 Results: {res_l1}")
    
    # 4. STARIMA
    logger.info("Running STARIMA...")
    res_starima = run_starima(splits["train"], splits["test"], adj)
    results["STARIMA"] = res_starima
    logger.info(f"  STARIMA Results: {res_starima}")
    
    # 5. ZINB
    logger.info("Running ZINB Regression...")
    res_zinb = run_zinb(splits["train"], splits["test"], adj)
    if res_zinb:
        results["ZINB"] = res_zinb
        logger.info(f"  ZINB Results: {res_zinb}")
        
    # 6. XGBoost
    logger.info("Running Spatiotemporal XGBoost...")
    res_xgb = run_xgboost(splits["train"], splits["test"], adj)
    results["XGBoost"] = res_xgb
    logger.info(f"  XGBoost Results: {res_xgb}")
    
    # Compile and save results
    output_dir = project_root / "outputs" / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(results).T
    df.index.name = "Model"
    out_file = output_dir / f"{data_name}_baselines.csv"
    df.to_csv(out_file)
    logger.info(f"Results saved to {out_file}")
    
    print("\n" + "=" * 60)
    print(f"  BASELINE RESULTS ({data_name.upper()}) - TEST SET (2023)")
    print("=" * 60)
    print(df.to_string())


if __name__ == "__main__":
    main()
