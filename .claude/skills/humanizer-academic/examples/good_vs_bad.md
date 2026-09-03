# Before/After Examples for Academic Humanization

## Example 1: Introduction Paragraph

### BAD (AI-generated):
In the contemporary landscape of urban safety, the utilization of machine learning methodologies for crime forecasting has emerged as a pivotal area of research. It is worth noting that traditional approaches, while comprehensive in their scope, fail to adequately address the multifaceted challenges inherent in spatiotemporal prediction tasks.

### GOOD (Human-written):
Crime does not happen randomly. It clusters in space and time — a robbery on one block often predicts more crime nearby in the following weeks. Most forecasting models treat each neighborhood independently, or they predict a single number without saying how confident they are. We wanted to do better: predict the full probability distribution of crime counts for every neighborhood, every week, with honest uncertainty estimates and fairness guarantees.

## Example 2: Results Paragraph

### BAD:
The experimental evaluation demonstrates that our proposed methodology achieves superior performance across all evaluated metrics. Notably, the CRPS score of 2.8267 represents a statistically significant improvement over the baseline.

### GOOD:
We beat every baseline on both cities. Chicago CRPS = 2.83, NYC = 3.14. The Diebold-Mariano test confirms these are not flukes — all 8 head-to-head comparisons are significant at p < 10^-6. Even TFT-ZINB, the strongest deep learning baseline, loses to our ensemble by a clear margin (p = 1.67e-6 on Chicago).

## Example 3: Limitations

### BAD:
While our comprehensive framework demonstrates robust performance across multiple evaluation dimensions, several limitations merit consideration for future investigation.

### GOOD:
Our model has real weaknesses and we should be upfront about them. The biggest one: a single CIVIC-SAFE seed (CRPS around 3.36) actually loses to TFT-ZINB (2.95). We only win because of 5-seed EMOS ensembling. The architecture alone is not enough.
