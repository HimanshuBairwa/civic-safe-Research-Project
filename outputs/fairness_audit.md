# CivicSafe: Model Fairness & Spatial Equity Audit

This report evaluates the model's predictions across four mathematical dimensions of fairness and equity, utilizing real-world demographic data.

## City: Chicago

### Metric Summary
| Metric | Value |
|---|---|
| Error Disparity (MAE diff, High vs Low pct_black) | 12.1132 |
| Bias Amplification Score (Corr Pred - Corr True) | 0.0351 |
| Top-20% Allocation DIR (High vs Low pct_black) | 384615384.6154 |
| Spatial Gini Coefficient (Absolute Errors) | 0.5407 |
| Moran's I (Residuals) | 0.6077 |

### Observations

- **Error Disparity**: The model makes larger errors (by 12.11 MAE) in neighborhoods with higher Black populations compared to those with lower Black populations.
- **Bias Amplification**: The model does not significantly amplify racial bias (BAS = 0.035).
- **Disparate Impact**: The model allocates high-risk/top-K predictions significantly more often to neighborhoods with higher Black populations (DIR = 384615384.62). A DIR > 1.0 indicates over-representation.
- **Spatial Inequality (Gini)**: A spatial Gini coefficient of 0.541 indicates the degree of inequality in how errors are distributed spatially across the city.
- **Spatial Autocorrelation (Moran's I)**: A score of 0.608 suggests clustering of residuals (positive spatial autocorrelation), meaning adjacent neighborhoods tend to have similar error patterns.

---

## City: Nyc

### Metric Summary
| Metric | Value |
|---|---|
| Error Disparity (MAE diff, High vs Low pct_black) | 6.0418 |
| Bias Amplification Score (Corr Pred - Corr True) | 0.0919 |
| Top-20% Allocation DIR (High vs Low pct_black) | 384615384.6154 |
| Spatial Gini Coefficient (Absolute Errors) | 0.4725 |
| Moran's I (Residuals) | 0.0713 |

### Observations

- **Error Disparity**: The model makes larger errors (by 6.04 MAE) in neighborhoods with higher Black populations compared to those with lower Black populations.
- **Bias Amplification**: The model amplifies the association between the target variable and race by 0.092. Predictions are more heavily correlated with `pct_black` than the ground truth.
- **Disparate Impact**: The model allocates high-risk/top-K predictions significantly more often to neighborhoods with higher Black populations (DIR = 384615384.62). A DIR > 1.0 indicates over-representation.
- **Spatial Inequality (Gini)**: A spatial Gini coefficient of 0.473 indicates the degree of inequality in how errors are distributed spatially across the city.
- **Spatial Autocorrelation (Moran's I)**: A score of 0.071 indicates minimal or no spatial clustering of model errors.

---

## Methodology
- **Error Disparity**: Difference in Mean Absolute Error (MAE) between demographic strata (e.g., above vs. below median `pct_black`).
- **Bias Amplification Score (BAS)**: Difference in Pearson correlation between (Predictions, Sensitive Attribute) and (True Labels, Sensitive Attribute).
- **Top-K Allocation DIR**: Ratio of the probability of being selected in the top 20% of predictions for the protected vs. unprotected group.
- **Spatial Inequality**: Measured via the Gini Coefficient of absolute errors and Moran's I on model residuals to detect spatial error clustering.