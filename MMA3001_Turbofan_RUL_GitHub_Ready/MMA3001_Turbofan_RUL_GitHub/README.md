# Turbofan Engine Remaining Useful Life Prediction | NASA C-MAPSS FD001

**MMA3001 numerical methods and machine learning project**  
**Task:** Predict the remaining useful life (RUL) of simulated turbofan engines in **operating cycles** from sensor observations.

![Actual vs predicted RUL on 100 NASA FD001 test engines](stage4/figures/02_official_test_actual_vs_predicted.png)

## Overview

This project investigates whether sensor-informed regression models predict simulated engine degradation more accurately than a cycle-only model. The NASA C-MAPSS **FD001** subset contains one operating condition and one fault mode (high-pressure compressor degradation).

**Inputs:** current operating cycle and 14 selected sensor channels (after excluding seven near-constant/constant channels). Engine ID is used for grouping, **not** as a predictor.  
**Output:** remaining useful life in cycles. For complete training histories, `RUL = failure_cycle - current_cycle`.

The modelling pipeline compares five methods:

1. Mean-RUL baseline
2. Cycle-only linear regression baseline
3. Multivariate Linear Regression
4. Random Forest Regression
5. Feedforward Neural Network (MLP)

## Official FD001 test results

The models were fixed using an engine-disjoint development/validation split, then refitted on all 100 training engines and evaluated once on the 100 official test-engine endpoints. RUL targets are **uncapped** and predictions are clipped to nonnegative values.

| Model | MAE (cycles) ↓ | RMSE (cycles) ↓ | R² ↑ |
|:--|--:|--:|--:|
| **Neural Network** | **18.24** | **24.96** | **0.639** |
| Random Forest | 19.58 | 26.41 | 0.596 |
| Linear Regression | 25.58 | 30.84 | 0.449 |
| Cycle-only baseline | 33.46 | 39.31 | 0.105 |
| Mean baseline | 40.88 | 52.63 | -0.604 |

The Neural Network improved MAE by approximately **45.5%** relative to the cycle-only baseline. Its advantage over Random Forest was much smaller (**1.35 cycles MAE**), and a paired bootstrap interval included zero; do not interpret this result as decisive superiority over Random Forest.

**Maintenance caveat:** The Neural Network overestimated RUL on **64/100** engines and had a mean signed error of **+9.94 cycles**. These are simulated benchmark results, **not** a certified real-aircraft prognostics system.

![Model comparison, MAE and RMSE](stage4/figures/01_official_test_MAE_RMSE.png)

## Project organisation

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── setup_dataset.py            # Extract only FD001 from NASA archive
│   └── sync_stage3_data.py         # Move Stage 2 outputs into Stage 3
├── stage1/
│   ├── 01_Data_Exploration.ipynb
│   └── figures/                    # Exploratory plots
├── stage2/
│   ├── 02_Data_Preprocessing.ipynb
│   ├── prepare_fd001.py
│   └── Stage2_Report_Notes.md
├── stage3/
│   ├── 03_Model_Training.ipynb
│   ├── 03_Model_Training.py
│   ├── data/                       # Frozen decision metadata (not raw records)
│   ├── results/                    # Validation metrics and predictions
│   └── figures/
├── stage4/
│   ├── 04_Final_Evaluation.ipynb
│   ├── 04_Final_Evaluation.py
│   ├── data/                       # Frozen Stage 3 decision (not raw records)
│   ├── results/                    # Official test metrics and predictions
│   └── figures/
└── docs/                           # Report publication notes
```

The raw NASA dataset, large serialized model binaries, and generated intermediate training rows are intentionally **not committed**. Scripts can recreate them locally. The full report draft is excluded while it is under revision.

## Reproduce the analysis (Windows PowerShell)

Clone or download this repository, open a terminal in its root, and run:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**1. Obtain the NASA C-MAPSS archive:** Visit the [NASA Prognostics Center of Excellence dataset repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) and download **6. Turbofan Engine Degradation Simulation**. Either the outer NASA ZIP or the enclosed `CMAPSSData.zip` works:

```powershell
python scripts/setup_dataset.py "C:\path\to\NASA_CMAPSS_archive.zip"
```

**2. Explore the dataset:**

```powershell
cd stage1
jupyter notebook 01_Data_Exploration.ipynb
cd ..
```

**3. Prepare data and generate validation split:**

```powershell
cd stage2
python prepare_fd001.py
cd ..
python scripts/sync_stage3_data.py
```

**4. Fit and evaluate the development models:**

```powershell
cd stage3
python 03_Model_Training.py
cd ..
```

**5. Refit and perform independent NASA test evaluation:**

```powershell
cd stage4
python 04_Final_Evaluation.py
cd ..
```

You can also open the corresponding Stage 2–4 Jupyter notebooks from their own directories. Model training involves random seeds and computational dependencies; numerical results may vary slightly across package versions or hardware. Keep test labels untouched during model selection.

### Experimental design and data leakage controls

- Split the 100 complete training engines into **80 development engines and 20 independent validation engines**; never randomly split rows from the same engine across groups.
- Score model selection on one artificially censored observation per validation engine (20 validation predictions).
- Fit any feature standardisation on training engines only.
- Fix model configurations before using the 100 official test-engine RUL labels.
- Refit each model on all 100 training engines; evaluate one final available sensor observation per official test engine.
- Examine overestimation separately because high RUL forecasts can be problematic for maintenance planning.

## Limitations

This is synthetic simulator data, not operational telemetry from a particular Airbus A320 engine. The predictions concern a single FD001 fault scenario, not all engine failures. Validation covers only a small number of held-out engines and there is substantial uncertainty, especially for a safety-critical maintenance decision.

## Dataset source and attribution

Saxena, A. and Goebel, K. (2008), **Turbofan Engine Degradation Simulation Data Set**, NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA. [Original NASA dataset page](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).

The dataset and NASA software are not produced by this repository. The original dataset's redistribution terms have not been independently verified, so the repository provides download/preparation instructions instead of republishing the raw files. Repository analysis and explanations are for academic demonstration; do not use them for flight-critical decisions.
