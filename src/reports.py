"""Helpers for validating, inspecting, and exporting paper result tables."""
from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

REQUIRED_AGG_COLUMNS = {
    "Arch", "Index", "Scenario", "Lookback", "Split", "MAE_mean", "MAE_std",
    "RMSE_mean", "RMSE_std", "MAPE_mean", "MAPE_std", "R2_mean", "R2_std",
    "MedAE_mean", "MedAE_std", "n_seeds", "MAE_ci95", "RMSE_ci95",
    "MAPE_ci95", "R2_ci95", "MedAE_ci95",
}

ALLOWED_DM_PAIRS = {"A vs B", "A vs C"}


def load_aggregated_summary(path: str | Path | None = None, validate: bool = True) -> pd.DataFrame:
    """Load the aggregated multi-seed performance summary."""
    df = pd.read_csv(path or RESULTS / "tables" / "aggregated_summary.csv")
    if validate:
        validate_aggregated_summary(df)
    return df


def load_dm_tests(path: str | Path | None = None, validate: bool = True) -> pd.DataFrame:
    """Load the refreshed Diebold-Mariano test table."""
    df = pd.read_csv(path or RESULTS / "tables" / "dm_tests_refreshed.csv")
    if validate:
        validate_dm_tests(df)
    return df


def load_best_hyperparameters(path: str | Path | None = None) -> dict:
    """Load Optuna-selected hyperparameters."""
    with open(path or RESULTS / "hyperparameters" / "best_hyperparameters.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


def validate_aggregated_summary(df: pd.DataFrame) -> None:
    """Validate the current MedAE-based metric schema."""
    missing = REQUIRED_AGG_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"aggregated_summary.csv is missing columns: {sorted(missing)}")
    if len(df) != 72:
        raise ValueError(f"Expected 72 rows in aggregated_summary.csv, found {len(df)}")


def validate_dm_tests(df: pd.DataFrame) -> None:
    """Validate that only planned A-vs-B and A-vs-C DM comparisons are reported."""
    if "Pair" not in df.columns:
        raise ValueError("dm_tests_refreshed.csv is missing the Pair column")
    observed_pairs = set(df["Pair"].dropna().unique())
    unexpected = observed_pairs.difference(ALLOWED_DM_PAIRS)
    if unexpected:
        raise ValueError(f"Unexpected DM comparison pairs found: {sorted(unexpected)}")
    if len(df) != 24:
        raise ValueError(f"Expected 24 rows in dm_tests_refreshed.csv, found {len(df)}")


def best_test_rows(metric: str = "MAE_mean") -> pd.DataFrame:
    """Return the best test-row per index and scenario for the requested metric."""
    df = load_aggregated_summary()
    test = df[df["Split"].eq("test")].copy()
    idx = test.groupby(["Index", "Scenario"])[metric].idxmin()
    return test.loc[idx].sort_values(["Index", "Scenario"]).reset_index(drop=True)


def to_latex_tables(output_dir: str | Path = RESULTS / "latex") -> None:
    """Export compact LaTeX tables for the supplied result CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    agg = load_aggregated_summary()
    dm = load_dm_tests()
    best_test_rows().to_latex(output_dir / "best_test_rows.tex", index=False, float_format="%.4f")
    agg.to_latex(output_dir / "aggregated_summary.tex", index=False, float_format="%.4f")
    dm.to_latex(output_dir / "dm_tests_refreshed.tex", index=False, float_format="%.4f")


if __name__ == "__main__":
    agg = load_aggregated_summary()
    dm = load_dm_tests()
    hparams = load_best_hyperparameters()
    print(f"aggregated_summary.csv: {agg.shape[0]} rows x {agg.shape[1]} columns")
    print(f"dm_tests_refreshed.csv: {dm.shape[0]} rows x {dm.shape[1]} columns")
    print(f"best_hyperparameters.json: {len(hparams)} configurations")
    print("\nBest test rows by MAE:")
    print(best_test_rows().to_string(index=False))
