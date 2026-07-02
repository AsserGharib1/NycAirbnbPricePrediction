# NYC Airbnb Price Prediction — Regression Benchmark

Price modeling for **49,000 NYC Airbnb listings** (AB_NYC_2019), comparing tuned non-linear regressors against a linear baseline.

## Results

| Model | Notes |
|---|---|
| **KNN (k = 27, tuned)** | Beats linear baseline on RMSE and R² |
| **Decision Tree (depth = 20, tuned)** | Beats linear baseline on RMSE and R² |
| Linear Regression | Baseline |

Model selection used validation-curve sweeps over k and tree depth; learning curves diagnose bias/variance for each model. Final metrics (RMSE, R²) are computed on a held-out test set — see the last notebook cells.

## Pipeline

- Cleaning + feature preparation: imputation, one-hot encoding, scaling (fit on train only)
- Hyperparameter sweeps for KNN and Decision Tree
- Learning-curve analysis and test-set comparison

## Repository contents

- `nyc_airbnb_price_regression.ipynb` — full workflow with preserved outputs.
- `AB_NYC_2019.csv` — dataset (source: *New York City Airbnb Open Data*, Kaggle).

## Running

```bash
pip install -r requirements.txt
jupyter notebook nyc_airbnb_price_regression.ipynb
```
