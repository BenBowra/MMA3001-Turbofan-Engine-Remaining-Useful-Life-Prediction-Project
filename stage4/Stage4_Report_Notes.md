# Stage 4 — Final Evaluation: Findings for the MMA3001 report

## Methodology
The NASA C-MAPSS FD001 simulated turbofan dataset contains 100 run-to-failure training engines (20,631 observations) and 100 truncated testing engines (13,096 observations). Only the last observation for each test engine is scored. The uncapped training target was calculated as full engine lifespan minus current cycle. The final feature list was frozen in Stage 3: operating cycle and 14 selected sensor channels (S2, S3, S4, S7, S8, S9, S11, S12, S13, S14, S15, S17, S20, S21). Seven low-variance/constant sensors and the three operational settings were excluded before official test evaluation.

Three regression methods — Linear Regression (with StandardScaler), Random Forest and a multilayer perceptron Neural Network (with StandardScaler) — were refitted on all 100 training engines using configurations fixed during Stage 3. A mean baseline and an operating-cycle-only linear baseline were included. The official NASA RUL labels were opened only after these fixed models generated predictions for the 100 test-engine endpoints. Predicted negative RUL values were clipped to zero, while labels remained uncapped.

## Final official test performance

| Model | MAE (cycles) | RMSE (cycles) | R² | Mean signed error (cycles) |
|---|---:|---:|---:|---:|
| Neural Network | 18.24 | 24.96 | 0.639 | +9.94 |
| Random Forest | 19.58 | 26.41 | 0.596 | +9.64 |
| Linear Regression | 25.58 | 30.84 | 0.449 | +16.19 |
| Cycle-only baseline | 33.46 | 39.31 | 0.105 | +16.33 |
| Mean baseline | 40.88 | 52.62 | -0.604 | +32.29 |

**Interpretation.** The preselected Neural Network achieved the lowest official test MAE (18.24 cycles) and RMSE (24.96 cycles). Its MAE was 45.5% lower than the cycle-only baseline, suggesting that sensor measurements improved RUL predictions beyond age information alone. Relative to the Random Forest (MAE 19.58), the Neural Network advantage was small (1.35 cycles). A paired engine bootstrap of the MAE advantage (Random Forest minus Neural Network) gave an approximate 95% interval of -0.15 to +2.90 cycles. Therefore, the test data do not establish a clear practical advantage of the Neural Network over Random Forest; additionally, comparing numerous models is inherently exploratory.

## Maintenance-relevant errors
The Neural Network overestimated RUL for 64 of 100 test engines, and underestimated it for 36. Its mean signed error was +9.94 cycles; 30 engines were overestimated by more than 20 cycles. Overestimation can delay scheduled maintenance, so MAE alone does not establish readiness for safety-critical application. The maximum absolute error was approximately 69.08 cycles. The largest risks must be evaluated in the context of operational failure thresholds, which the simulated data alone do not supply.

When grouped by true RUL, the Neural Network MAE was approximately 9.17 cycles for 25 engines with 0–30 actual cycles remaining; 8.68 cycles for 14 engines with 31–60; 25.42 cycles for 15 engines with 61–90; and 23.73 cycles for 46 engines with over 90. These are retrospective descriptive groups, not labels available at prediction time.

## Limitations and uncertainty
* All results come from simulated FD001 high-pressure compressor degradation under one operating condition, not actual airline flight/maintenance histories or a specific Airbus A320 engine.
* The 100 test-engine endpoints do not evaluate performance for all stages of every engine's life.
* Stage 3's holdout validation covered only 20 artificially censored engines, so its ranking had uncertainty and differed in censoring distribution from NASA's official test.
* The MLP's internal early-stopping split is based on sample rows, not entire engines, but all rows belong to the training fleet; it is not used as an external generalisation score.
* Engine-life predictions are point estimates, with no calibrated prediction intervals or certified decision thresholds.
* The models use current sensor values and cycle number, not explicit recurrent histories or sequence learning. The Random Forest feature importance from Stage 3 is not causal evidence.
* Bootstrap intervals are descriptive uncertainty over finite simulated test-engine samples and cannot certify field reliability.

## Conclusion
Among the predefined models, the Neural Network had the best overall official FD001 test accuracy, while Random Forest achieved similar results and may offer different deployment trade-offs. All three sensor-informed models outperformed the cycle-only baseline. However, systematic RUL overestimation and the limitations of simulated data mean that this study provides a proof-of-concept for predictive-maintenance modelling, **not a production-ready aircraft maintenance decision tool**. No model adjustments should be made on the official test set; any further model development requires a new untouched benchmark or resampling entirely within the original training engines.

## Figures
1. `figures/01_official_test_MAE_RMSE.png` — model error comparison
2. `figures/02_official_test_actual_vs_predicted.png` — parity plots
3. `figures/03_official_test_signed_errors.png` — prediction over-/underestimation
4. `figures/04_test_MAE_by_RUL_band.png` — error by true RUL band (retrospective diagnostic)

## Sources
NASA Ames Prognostics Center of Excellence C-MAPSS Turbofan Engine Degradation Simulation dataset; included original `data/readme.txt`. Model choices are from the Stage 3 `stage3_model_decision.json` included in `data/`.
