# CivicSafe: Model Fairness & Spatial Equity Audit

This report evaluates the model's predictions across four mathematical dimensions of fairness and equity, utilizing real-world demographic data.

## City: Chicago

### Metric Summary
| Metric | Value |
|---|---|
| Error Disparity (MAE diff, High vs Low pct_black) | 0.8415 |
| Bias Amplification Score (Corr Pred - Corr True) | 0.1234 |
| Top-20% Allocation DIR (High vs Low pct_black) | 1.4502 |
| Spatial Gini Coefficient (Absolute Errors) | 0.3120 |
| Moran's I (Residuals) | 0.0521 |

### Observations

- **Error Disparity**: The model makes larger errors (by 0.84 MAE) in neighborhoods with higher Black populations compared to those with lower Black populations.
- **Bias Amplification**: The model amplifies the association between the target variable and race by 0.123. Predictions are more heavily correlated with `pct_black` than the ground truth.
- **Disparate Impact**: The model allocates high-risk/top-K predictions significantly more often to neighborhoods with higher Black populations (DIR = 1.45). A DIR > 1.0 indicates over-representation.
- **Spatial Inequality (Gini)**: A spatial Gini coefficient of 0.312 indicates the degree of inequality in how errors are distributed spatially across the city.
- **Spatial Autocorrelation (Moran's I)**: A score of 0.052 indicates minimal or no spatial clustering of model errors.

---
## City: Nyc

### Metric Summary
| Metric | Value |
|---|---|
| Error Disparity (MAE diff, High vs Low pct_black) | 0.7321 |
| Bias Amplification Score (Corr Pred - Corr True) | 0.1089 |
| Top-20% Allocation DIR (High vs Low pct_black) | 1.3400 |
| Spatial Gini Coefficient (Absolute Errors) | 0.2980 |
| Moran's I (Residuals) | 0.0450 |

### Observations

- **Error Disparity**: The model makes larger errors (by 0.73 MAE) in neighborhoods with higher Black populations compared to those with lower Black populations.
- **Bias Amplification**: The model amplifies the association between the target variable and race by 0.109. Predictions are more heavily correlated with `pct_black` than the ground truth.
- **Disparate Impact**: The model allocates high-risk/top-K predictions significantly more often to neighborhoods with higher Black populations (DIR = 1.34). A DIR > 1.0 indicates over-representation.
- **Spatial Inequality (Gini)**: A spatial Gini coefficient of 0.298 indicates the degree of inequality in how errors are distributed spatially across the city.
- **Spatial Autocorrelation (Moran's I)**: A score of 0.045 indicates minimal or no spatial clustering of model errors.

---
## Methodology
- **Error Disparity**: Difference in Mean Absolute Error (MAE) between demographic strata (e.g., above vs. below median `pct_black`).
- **Bias Amplification Score (BAS)**: Difference in Pearson correlation between (Predictions, Sensitive Attribute) and (True Labels, Sensitive Attribute).
- **Top-K Allocation DIR**: Ratio of the probability of being selected in the top 20% of predictions for the protected vs. unprotected group.
- **Spatial Inequality**: Measured via the Gini Coefficient of absolute errors and Moran's I on model residuals to detect spatial error clustering.
