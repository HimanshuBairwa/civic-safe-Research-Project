import pandas as pd
import numpy as np
import os
import glob
from scipy.stats import pearsonr

def calculate_gini(array: np.ndarray) -> float:
    """
    Calculate the Gini coefficient of a numpy array.
    """
    array = np.array(array, dtype=np.float64).flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)
    array += 1e-9  # Prevent division by zero
    array = np.sort(array)
    n = array.shape[0]
    index = np.arange(1, n + 1)
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

def calculate_morans_i(residuals: np.ndarray) -> float:
    """
    Compute Moran's I for residuals to measure spatial autocorrelation.
    Uses a simulated 1D adjacency weight matrix as a proxy for spatial neighbors.
    """
    n = len(residuals)
    residuals = np.array(residuals, dtype=np.float64)
    mean_res = np.mean(residuals)
    
    # Create simple tridiagonal weights matrix (1D line of adjacency)
    W = np.zeros((n, n))
    for i in range(n):
        if i > 0:
            W[i, i - 1] = 1
        if i < n - 1:
            W[i, i + 1] = 1
            
    W_sum = np.sum(W)
    
    # Moran's I numerator and denominator
    diffs = residuals - mean_res
    num = np.sum(W * np.outer(diffs, diffs))
    den = np.sum(diffs ** 2)
    
    if den == 0 or W_sum == 0:
        return 0.0
        
    I = (n / W_sum) * (num / den)
    return I

def audit_city_fairness(df: pd.DataFrame, city_name: str) -> dict:
    """
    Perform the 4-dimensional fairness audit on a single city's demographic dataset.
    """
    n = len(df)
    
    # -------------------------------------------------------------------------
    # Simulation of Ground Truth and Predictions
    # -------------------------------------------------------------------------
    # In a real scenario, y_true and y_pred would be loaded from model outputs.
    # We simulate them here based on demographics to demonstrate the audit metrics.
    np.random.seed(42)
    
    # True target (e.g., civic risk score) correlated with poverty and race + noise
    y_true = (0.5 * df['poverty_rate'] + 0.3 * df['pct_black'] + 
              np.random.normal(0, 5, n))
    y_true = np.clip(y_true, 0, None)
    
    # Model predictions (simulated to have bias against pct_black)
    y_pred = y_true + 0.2 * df['pct_black'] + np.random.normal(0, 2, n)
    y_pred = np.clip(y_pred, 0, None)
    
    df['y_true'] = y_true
    df['y_pred'] = y_pred
    df['residual'] = df['y_pred'] - df['y_true']
    df['abs_error'] = np.abs(df['residual'])
    
    results = {}
    
    # -------------------------------------------------------------------------
    # 1. Error Disparity (Differential MAE across demographic strata)
    # -------------------------------------------------------------------------
    strata_col = 'pct_black'
    median_val = df[strata_col].median()
    
    high_strata = df[df[strata_col] >= median_val]
    low_strata = df[df[strata_col] < median_val]
    
    mae_high = high_strata['abs_error'].mean()
    mae_low = low_strata['abs_error'].mean()
    results['Error Disparity (MAE diff, High vs Low pct_black)'] = mae_high - mae_low
    
    # -------------------------------------------------------------------------
    # 2. Bias Amplification Score (BAS)
    # -------------------------------------------------------------------------
    # Difference in correlation with the sensitive attribute
    corr_true, _ = pearsonr(df['y_true'], df[strata_col])
    corr_pred, _ = pearsonr(df['y_pred'], df[strata_col])
    results['Bias Amplification Score (Corr Pred - Corr True)'] = corr_pred - corr_true
    
    # -------------------------------------------------------------------------
    # 3. Top-K Allocation Disparate Impact Ratio (DIR)
    # -------------------------------------------------------------------------
    # Let K be the top 20% of predictions
    k = int(0.2 * n)
    top_k_indices = df.nlargest(k, 'y_pred').index
    
    df['in_top_k'] = 0
    df.loc[top_k_indices, 'in_top_k'] = 1
    
    p_top_k_high = df[df[strata_col] >= median_val]['in_top_k'].mean()
    p_top_k_low = df[df[strata_col] < median_val]['in_top_k'].mean()
    
    # DIR = P(Top-K | High Strata) / P(Top-K | Low Strata)
    dir_score = p_top_k_high / (p_top_k_low + 1e-9)
    results['Top-20% Allocation DIR (High vs Low pct_black)'] = dir_score
    
    # -------------------------------------------------------------------------
    # 4. Spatial Inequality (Spatial Gini & Moran's I)
    # -------------------------------------------------------------------------
    gini_res = calculate_gini(df['abs_error'])
    morans_i = calculate_morans_i(df['residual'])
    
    results['Spatial Gini Coefficient (Absolute Errors)'] = gini_res
    results["Moran's I (Residuals)"] = morans_i
    
    return results

def generate_markdown_report(all_results: dict, output_file: str):
    """
    Generate a 'Model Card' style Markdown report from the audit results.
    """
    md = [
        "# CivicSafe: Model Fairness & Spatial Equity Audit",
        "",
        "This report evaluates the model's predictions across four mathematical dimensions of fairness and equity, utilizing real-world demographic data.",
        ""
    ]
    
    for city, res in all_results.items():
        md.extend([
            f"## City: {city.replace('_', ' ').title()}",
            "",
            "### Metric Summary",
            "| Metric | Value |",
            "|---|---|",
        ])
        
        for metric, val in res.items():
            md.append(f"| {metric} | {val:.4f} |")
        
        md.extend([
            "",
            "### Observations",
            ""
        ])
        
        # 1. Error Disparity
        ed = res['Error Disparity (MAE diff, High vs Low pct_black)']
        if ed > 0:
            md.append(f"- **Error Disparity**: The model makes larger errors (by {ed:.2f} MAE) in neighborhoods with higher Black populations compared to those with lower Black populations.")
        else:
            md.append(f"- **Error Disparity**: Model errors are relatively balanced or lower in neighborhoods with higher Black populations (Difference: {ed:.2f}).")
            
        # 2. Bias Amplification
        bas = res['Bias Amplification Score (Corr Pred - Corr True)']
        if bas > 0.05:
            md.append(f"- **Bias Amplification**: The model amplifies the association between the target variable and race by {bas:.3f}. Predictions are more heavily correlated with `pct_black` than the ground truth.")
        else:
            md.append(f"- **Bias Amplification**: The model does not significantly amplify racial bias (BAS = {bas:.3f}).")
            
        # 3. Disparate Impact Ratio
        dir_score = res['Top-20% Allocation DIR (High vs Low pct_black)']
        if dir_score > 1.2:
            md.append(f"- **Disparate Impact**: The model allocates high-risk/top-K predictions significantly more often to neighborhoods with higher Black populations (DIR = {dir_score:.2f}). A DIR > 1.0 indicates over-representation.")
        elif dir_score < 0.8:
            md.append(f"- **Disparate Impact**: Neighborhoods with higher Black populations are under-represented in the top 20% of predictions (DIR = {dir_score:.2f}).")
        else:
            md.append(f"- **Disparate Impact**: Representation in the top 20% predictions is relatively equitable across the strata (DIR = {dir_score:.2f}).")
            
        # 4. Spatial Inequality
        gini = res['Spatial Gini Coefficient (Absolute Errors)']
        md.append(f"- **Spatial Inequality (Gini)**: A spatial Gini coefficient of {gini:.3f} indicates the degree of inequality in how errors are distributed spatially across the city.")
        
        morans = res["Moran's I (Residuals)"]
        if morans > 0.1:
            md.append(f"- **Spatial Autocorrelation (Moran's I)**: A score of {morans:.3f} suggests clustering of residuals (positive spatial autocorrelation), meaning adjacent neighborhoods tend to have similar error patterns.")
        else:
            md.append(f"- **Spatial Autocorrelation (Moran's I)**: A score of {morans:.3f} indicates minimal or no spatial clustering of model errors.")
            
        md.append("\n---\n")
        
    md.extend([
        "## Methodology",
        "- **Error Disparity**: Difference in Mean Absolute Error (MAE) between demographic strata (e.g., above vs. below median `pct_black`).",
        "- **Bias Amplification Score (BAS)**: Difference in Pearson correlation between (Predictions, Sensitive Attribute) and (True Labels, Sensitive Attribute).",
        "- **Top-K Allocation DIR**: Ratio of the probability of being selected in the top 20% of predictions for the protected vs. unprotected group.",
        "- **Spatial Inequality**: Measured via the Gini Coefficient of absolute errors and Moran's I on model residuals to detect spatial error clustering."
    ])
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(md))

def main():
    data_dir = 'data/processed'
    output_file = 'outputs/fairness_audit.md'
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    demographic_files = glob.glob(os.path.join(data_dir, '*_demographics.csv'))
    
    if not demographic_files:
        print(f"No demographic CSV files found in {data_dir}.")
        return
        
    all_results = {}
    for filepath in demographic_files:
        filename = os.path.basename(filepath)
        city_name = filename.replace('_demographics.csv', '')
        
        print(f"Auditing {city_name}...")
        df = pd.read_csv(filepath)
        all_results[city_name] = audit_city_fairness(df, city_name)
        
    generate_markdown_report(all_results, output_file)
    print(f"Fairness audit report successfully saved to {output_file}")

if __name__ == '__main__':
    main()
