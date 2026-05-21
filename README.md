# Stock Index Forecasting with Deep Learning

This repository contains the reproducibility package for the paper's empirical forecasting analysis. It includes the original analysis notebook, the Optuna-selected hyperparameters, the aggregated multi-seed performance table, and the refreshed Diebold-Mariano test results.

## Repository structure

```text
.
|-- data/
|   |-- raw/                  # Raw downloads are not committed
|   `-- processed/            # Leakage-safe prepared datasets should be placed here
|-- notebooks/
|   `-- Close_paper_May21.ipynb
|-- results/
|   |-- hyperparameters/
|   |   `-- best_hyperparameters.json
|   `-- tables/
|       |-- aggregated_summary.csv
|       `-- dm_tests_refreshed.csv
|-- scripts/
|   `-- run_full_pipeline.py  # Plain-Python extraction of the notebook workflow
|-- src/
|   `-- reports.py            # Utilities for loading/exporting result tables
|-- requirements.txt
`-- environment.yml
```

## Included result files

- `results/tables/aggregated_summary.csv`: 72 rows and 21 columns of validation/test metrics across architectures, indices, scenarios, and seeds.
- `results/tables/dm_tests_refreshed.csv`: 36 rows and 16 columns for refreshed Diebold-Mariano comparisons.
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

Inspect the supplied result tables:

```bash
python -m src.reports
```

Export LaTeX result tables:

```bash
python - <<'PY'
from src.reports import to_latex_tables
to_latex_tables()
PY
```

This writes `.tex` files to `results/latex/`.

## Reproducing the full workflow

The original workflow is preserved in `notebooks/Close_paper_May21.ipynb`. A plain-Python version is provided in `scripts/run_full_pipeline.py` for version control and easier inspection.

To rerun the full empirical pipeline, place the prepared scenario CSV files in `data/processed/` using the naming conventions expected by the notebook/script, then execute the relevant phases:

1. download market data and construct technical indicators;
2. apply leakage-safe rolling normalization and chronological train/validation/test splits;
3. run Optuna tuning with seed 42;
4. evaluate the selected hyperparameters across the ten non-tuning seeds;
5. compute the refreshed Diebold-Mariano tests.

Raw market data and intermediate prediction files are intentionally not committed. They can be regenerated through the notebook using Yahoo Finance.

## Version-control notes

Large generated artifacts, local environments, caches, model checkpoints, and intermediate predictions are excluded through `.gitignore`. Commit the source code, notebooks, configuration files, and final result tables needed for the paper.

## Citation

When using this repository, cite the associated paper. Update `CITATION.cff` with the final author list, title, venue, DOI, and year before public release.
