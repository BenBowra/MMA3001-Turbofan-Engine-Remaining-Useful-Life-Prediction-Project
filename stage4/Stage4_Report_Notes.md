# Stage 4 — FD001 official test evaluation (revised/exploratory)

All five approaches were refitted on 100 run-to-failure NASA FD001 training engines (20,631 cycles) and used to predict RUL at the final recorded observation of 100 truncated test-engine histories. The current run loads test labels after generating predictions. However, this project had **previously examined NASA test results** before the curriculum-driven switch from Random Forest to a single Decision Tree. Hence the revised scores must not be characterised as a newly untouched blind test.

| Method | MAE (cycles) | RMSE (cycles) | R² | Mean signed error |
|---|---:|---:|---:|---:|
| Mean baseline | 40.88 | 52.62 | −0.604 | +32.29 |
| Cycle-only baseline | 33.46 | 39.31 | 0.105 | +16.33 |
| Linear Regression | 25.58 | 30.84 | 0.449 | +16.19 |
| **Decision Tree** | **22.68** | **29.49** | **0.496** | **+11.74** |
| Neural Network | **18.24** | **24.96** | **0.639** | **+9.94** |

The decision tree used `max_depth=8`, `min_samples_leaf=5` and `random_state=42`. The Neural Network remains the lowest-error model in this benchmark. It showed an observed MAE advantage of about **4.44 cycles** over the single tree (paired bootstrap 95% interval approximately **+1.83 to +7.00 cycles**). Such intervals describe variation between the simulated FD001 test engines, not predictive reliability on aircraft.

Positive RUL prediction errors can delay maintenance. The Neural Network overestimated RUL on **64/100** test engines; the Decision Tree overestimated on **62/100**. The FD001 dataset covers one simulated operating regime and one fault mode, not all real turbofan conditions.

**Engineering conclusion:** sensor inputs add predictive information beyond cycle number, but simulation-only validation and optimistic bias prevent safety-critical deployment claims.
