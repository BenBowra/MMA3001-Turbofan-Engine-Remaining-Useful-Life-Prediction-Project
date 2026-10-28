# MMA3001 — Stage 2 report-ready methodology notes

## Objective
Prepare the NASA C-MAPSS FD001 run-to-failure records for reliable supervised regression of turbofan remaining useful life (RUL), while preserving engine-level independence between development and validation data.

## Target construction
For each *training* engine i at cycle t, the target was defined as RUL_i(t) = T_failure,i − t, where T_failure,i is the last recorded cycle of that training engine. These are **uncapped RUL targets** measured in operating cycles. The test engines end before failure, so their actual RUL cannot be determined from the truncated test records and must instead be taken from NASA's separate RUL test labels at the final evaluation stage.

## Train/validation split
The 100 complete engine trajectories were divided into 80 development-training engines (16,561 rows) and 20 validation engines (4,070 rows), with the split performed at the engine level using a reproducible random seed (42). This avoids leakage from successive, highly correlated sensor records of the same engine into both sets.

## Inputs and feature selection
Seven sensors—S1, S5, S6, S10, S16, S18 and S19—were excluded because each had no more than two unique values in the development-training fold. The initial input set contains 14 sensor channels (S2, S3, S4, S7, S8, S9, S11, S12, S13, S14, S15, S17, S20, S21) plus operating-cycle count. `engine_id` is retained for grouping only and excluded from the regression inputs. The three operational setting columns are omitted as an initial assumption because FD001 has one main operating condition and their variations are small; this assumption may be assessed in a future validation ablation. Correlation between sensors and RUL supports preliminary interpretation but does not establish causation or justify feature selection by itself.

## Scaling
A StandardScaler was fitted only on the development-training data (16,561 rows) and then applied without refitting to the validation rows. Standardisation is appropriate for the future linear and neural-network regressors; a random forest can use raw features. Keeping the fitted preprocessing independent of validation reduces information leakage.

## Representative validation
Because NASA evaluates one RUL prediction per truncated test engine, one artificial, pre-failure truncation snapshot was selected per validation engine (20 evaluation rows). Truncation occurred at a seeded random fraction between 30% and 90% of the known full validation life. Complete histories were used to derive true RUL and ensure a pre-failure cutoff, but not to provide future sensor information to the predictor. This simulated censoring is not identical to NASA's test distribution, so validation scores will be considered approximate, especially with only 20 snapshots.

## Outputs and next step
The workflow saved prepared training and validation CSV datasets, 20 validation snapshots, the 100 unlabeled NASA test endpoints, model-ready NumPy arrays, a fitted development-only scaler, metadata and preprocessing graphs. Stage 3 will implement cycle-only baseline, Linear Regression, Random Forest Regression and a feedforward Neural Network; compare MAE, RMSE and R² on the same 20 validation snapshots; and avoid using NASA's official test RUL labels for model selection.
