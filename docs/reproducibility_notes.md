# Reproducibility notes

- Hyperparameter tuning uses Optuna with tuning seed 42.
- Multi-seed evaluation uses ten non-tuning seeds: 123, 456, 789, 1024, 2025, 3141, 7777, 9001, 31337, and 65537.
- Reported metrics include MAE, RMSE, MAPE, R2, and MedAE.
- Diebold-Mariano tests use median predictions across seeds and report planned A-vs-B and A-vs-C comparisons only.
- B-vs-C comparisons are intentionally excluded from the refreshed DM output because they are not part of the planned comparisons against the Close-only baseline.
- The notebook was originally written for Google Colab; the extracted script preserves the workflow for inspection and should be adapted to local paths before a full rerun.
