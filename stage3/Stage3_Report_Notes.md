# MMA3001 Stage 3 — Report Methodology and Initial Results

## Research objective

We investigated whether engine sensor readings improve prediction of remaining useful life (RUL) beyond knowledge of elapsed operating cycles, and whether nonlinear models offer an advantage over linear regression.

## Dataset and evaluation design

The FD001 simulator supplies 100 complete run-to-failure training engines. Stage 2 allocated 80 independent engine IDs (16,561 cycle-level observations) to development training and 20 independent engine IDs (4,070 cycle-level observations) to validation. Primary validation uses **one preselected artificial partial-history endpoint per held-out engine** (20 predictions total, one per engine). The approach tests generalisation across engine IDs and avoids leakage from random cycle-level splitting. RUL labels use original uncapped cycles remaining to the known training failure event. The original NASA test set is **not touched** in Stage 3.

## Engineering and methodological justification

**Mean baseline:** Tests if the models improve on a constant train-set average RUL.

**Cycle-only baseline:** Tests whether accumulated operating time alone explains RUL. It also exposes the limitations of assuming all engines age identically.

**Multivariable Linear Regression:** Tests a simple interpretable additive relationship between elapsed cycle, sensor measurements and remaining life. It is a sensible baseline but cannot represent complex interactions or strongly nonlinear degradation responses.

**Random Forest Regression:** Multiple bagged decision trees capture nonlinear relations and interactions without requiring feature scaling or prespecifying physical equations. It may be less stable at extrapolating into underrepresented life stages.

**MLP Neural Network:** A feedforward multilayer perceptron (hidden layers 48 and 24, ReLU activation, Adam optimizer) learns a flexible nonlinear mapping from the same sensor-and-cycle predictors to RUL. A pipeline standardises features, and training-only internal early stopping limits needless iteration. It uses only present-cycle features and therefore does not explicitly model temporal history.

The predictor set is consistent across the three full models (cycle plus 14 sensors), while the cycle-only baseline is deliberately restricted. The Random Forest uses raw inputs; the linear model and MLP fit StandardScaler on development-training rows only. Predictions are clipped to nonnegative cycles for all models.

## Results

| Model | MAE (cycles) | RMSE (cycles) | R² |
|---|---:|---:|---:|
| Mean baseline | 37.53 | 45.74 | -0.933 |
| Cycle-only baseline | 33.68 | 39.45 | -0.438 |
| Linear Regression | 25.24 | 29.80 | 0.179 |
| Random Forest | 21.84 | 28.93 | 0.227 |
| Neural Network | **18.91** | **25.90** | **0.380** |

The MLP has the lowest MAE and RMSE on the 20 engineered held-out validation endpoints and is the **provisional** best-performing candidate. The fact that the sensor-based models outperform cycle-only regression suggests that sensor data provide additional information about remaining life beyond elapsed cycles for this validation split.

Using a simple resampling bootstrap across the 20 held-out engines, the MLP MAE interval is approximately **11.6–26.7 cycles**, and the Random Forest interval **14.4–30.6 cycles**. These large overlapping intervals indicate that this validation exercise is insufficient to establish a reliable superiority ranking. The original NASA FD001 test data will provide a larger independent endpoint evaluation in Stage 4; it must not be used to tune models.

## Sensor contributions

Impurity-based Random Forest feature importances place `cycle` (~0.542 relative importance) and `sensor_11` (~0.187) highest in this particular fitted model, followed by `sensor_4`, `sensor_9`, `sensor_12`, `sensor_14`, `sensor_7` and `sensor_15`. These are predictive associations, not proof that a sensor causally drives compressor degradation. Shared sensor information, correlations and decision tree splitting mechanics influence these figures.

## Important limitations and risk discussion

- **Simulator vs real engines:** FD001 contains simulated, not field-collected, engine data; its failure modes and operating regimes do not cover the complexity of in-service engines.
- **Limited validation sample:** Results reflect only 20 engines, each with one artificially selected cut-off at 30–90% of its full lifetime. This differs from the unknown NASA official test censoring process.
- **Static predictor limitation:** These models use the current cycle and current sensor values, not explicit historical windows, temporal derivatives or recurrent models.
- **MLP early stopping:** Its internal early-stop split is by row, taken entirely from the training engine pool; it is not an independent engine-group evaluation.
- **Prediction errors matter unequally:** Overestimating RUL may delay maintenance, while underestimating it may cause unnecessary servicing. Symmetric MAE/RMSE alone do not represent the operational cost asymmetry.
- **R² is modest:** Even the best Stage 3 model explains only around 38% of RUL variation relative to predicting the held-out endpoints' mean. It is not ready to support actual maintenance decisions.
- **Training and validation MAE not directly comparable:** Training scores aggregate every cycle; primary validation scores a single artificial truncation per engine, changing the target and input distributions. A smaller validation score does not imply absence of overfitting.
- **Computational comparison:** The measured fitting times are local notebook runtime, not rigorous comparable engineering benchmarks.

## Suggested report conclusion (Stage 3 only)

Machine learning regression models were developed and compared using NASA's C-MAPSS FD001 simulation dataset. Three sensor-informed regressors were assessed against simple constant and cycle-based baselines using 20 held-out engine snapshots. The neural network achieved the lowest observed prediction error (MAE 18.91 and RMSE 25.90 cycles), while Random Forest also performed better than the linear model on this sample. The performance improvement over the cycle-only baseline suggests that the selected engine sensor measurements are useful indicators of remaining lifetime. However, substantial prediction errors and the limited number of validation engines prevent a strong generalisation claim. The next stage will evaluate fixed models on NASA's official, independent test engines after refitting on all original training engines.
