# Manifest

This repository contains the reproducibility files for the empirical forecasting analysis.

## Included

- `notebooks/Close_paper_May23.ipynb`: updated analysis notebook.
- `scripts/run_full_pipeline.py`: plain-Python extraction of the notebook workflow.
- `results/tables/aggregated_summary.csv`: aggregated ten-seed validation/test metrics with MedAE.
- `results/tables/dm_tests_refreshed.csv`: refreshed Diebold-Mariano results for A-vs-B and A-vs-C only.
- `results/hyperparameters/best_hyperparameters.json`: Optuna-selected hyperparameters.
- `src/reports.py`: validation and export utilities.

## Not included

Raw data, intermediate generated datasets, prediction arrays, local caches, model checkpoints, and environment directories are excluded from version control.
