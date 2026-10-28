# Stage 3 — MMA3001 Week 5 model comparison

The final regression models map directly to MMA3001: **Linear Regression (5.1)**, **Decision Tree Regression (5.2)** and **Neural Network Regression (5.4)**. Mean-RUL and cycle-only linear regression are reference baselines.

**Inputs:** operating cycle plus 14 sensor channels. **Output:** uncapped remaining useful life in cycles. Development uses 80 engine-disjoint training histories (16,561 rows) and 20 artificially truncated validation engine endpoints (4,070 history rows). Feature scaling is fit on training engines only.

The Decision Tree is a *single tree*, not a Random Forest. We used `max_depth=8`, `min_samples_leaf=5`, `random_state=42` to constrain the depth and avoid extremely small leaf partitions. The MLP retains its two hidden layers (48, 24), ReLU, Adam, regularisation and training-internal early stopping. No exhaustive hyperparameter search is claimed.

| Model | Validation MAE (cycles) | Validation RMSE (cycles) | R² |
|---|---:|---:|---:|
| Linear Regression | 25.24 | 29.80 | 0.179 |
| Decision Tree | 21.96 | 28.77 | 0.235 |
| Neural Network | **18.91** | **25.90** | **0.380** |

The neural model has the lowest observed validation RMSE, although 20 engine endpoints provide limited precision.

**Transparency:** This change was made for curriculum alignment **after results from an earlier Random Forest-based official test run had been viewed**. The revised official-test comparison is **exploratory**, not an entirely blind external evaluation. Historical code and results remain recoverable through Git history.
