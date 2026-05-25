# Stock Index Forecasting with Deep Learning

This repository contains the reproducibility package for the empirical forecasting analysis associated with the submitted manuscript, **Feature Complexity and Architecture Performance in Deep Learning Financial Forecasting: An Empirical Systematic Literature Review**.

The repository preserves the updated May 23 analysis notebook, the Optuna-selected hyperparameters, the aggregated ten-seed performance table, and the refreshed Diebold-Mariano test results used for the paper revision.

## Repository structure

```text
.
|-- data/
|   |-- raw/                  # Raw downloads are not committed
|   `-- processed/            # Leakage-safe prepared datasets should be placed here
|-- notebooks/
|   `-- Close_paper_May23.ipynb
|-- results/
|   |-- hyperparameters/
|   |   `-- best_hyperparameters.json
|   `-- tables/
|       |-- aggregated_summary.csv
|       `-- dm_tests_refreshed.csv
|-- scripts/
|   `-- run_full_pipeline.py  # Plain-Python extraction of the notebook workflow
|-- src/
|   `-- reports.py            # Utilities for validating and exporting result tables
|-- requirements.txt
`-- environment.yml
```

## Included result files

- `results/tables/aggregated_summary.csv`: 72 rows and 21 columns of validation/test metrics across architectures, indices, scenarios, and seeds. The reported metrics are MAE, RMSE, MAPE, $R^2$, and MedAE.
- `results/tables/dm_tests_refreshed.csv`: 24 rows and 17 columns for the planned Diebold-Mariano comparisons. The file contains A-vs-B and A-vs-C comparisons only; B-vs-C comparisons are intentionally excluded.
- `results/hyperparameters/best_hyperparameters.json`: 36 Optuna-selected architecture-index-scenario configurations.

## Models and scenarios

The supplied results cover four architectures: LSTM, BiLSTM, TCN, and Transformer. They evaluate three equity indices: S&P 500, EURO STOXX 50, and Nikkei 225. The stored tables include scenarios A, B, and C, consistent with the current empirical paper outputs.

## Setup

Create a Python environment with either `pip` or `conda`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

or

```bash
conda env create -f environment.yml
conda activate stock-index-forecasting
```

## Quick checks

Validate and inspect the supplied result tables:

```bash
python -m src.reports
```

This check verifies that the aggregated summary follows the current MedAE-based metric schema and that the DM file contains only A-vs-B and A-vs-C comparisons.

Export LaTeX result tables:

```bash
python - <<'PY'
from src.reports import to_latex_tables
to_latex_tables()
PY
```

This writes `.tex` files to `results/latex/`.

## Reproducing the full workflow

The updated workflow is preserved in `notebooks/Close_paper_May23.ipynb`. A plain-Python version is provided in `scripts/run_full_pipeline.py` for version control and easier inspection.

To rerun the full empirical pipeline, place the prepared scenario CSV files in `data/processed/` using the naming conventions expected by the notebook/script, then execute the relevant phases:

1. download market data and construct technical indicators;
2. apply leakage-safe rolling normalization and chronological train/validation/test splits;
3. run Optuna tuning with seed 42;
4. evaluate the selected hyperparameters across the ten non-tuning seeds;
5. compute the refreshed Diebold-Mariano tests for A-vs-B and A-vs-C only.

Raw market data and intermediate prediction files are intentionally not committed. They can be regenerated through the notebook using Yahoo Finance.

## Version-control notes

Large generated artifacts, local environments, caches, model checkpoints, and intermediate predictions are excluded through `.gitignore`. Commit the source code, notebooks, configuration files, and final result tables needed for the paper.

## Citation

The associated manuscript has been submitted to *Artificial Intelligence Review*. Formal citation metadata will be added after publication details are available.
