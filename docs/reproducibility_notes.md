# Reproducibility notes

- Hyperparameter tuning uses Optuna with tuning seed 42.
- Multi-seed evaluation uses ten non-tuning seeds: 123, 456, 789, 1024, 2025, 3141, 7777, 9001, 31337, and 65537.
- Reported metrics include MAE, RMSE, MAPE, R2, and MASE.
- Diebold-Mariano tests use median predictions across seeds and include multiple-comparison adjusted p-values.
- The notebook was originally written for Google Colab; the extracted script replaces the Colab Drive paths with repository-relative paths where possible.
