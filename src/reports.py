"""Helpers for inspecting and exporting paper result tables."""
from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load_aggregated_summary(path: str | Path | None = None) -> pd.DataFrame:
    """Load the aggregated multi-seed performance summary."""
    return pd.read_csv(path or RESULTS / "tables" / "aggregated_summary.csv")


def load_dm_tests(path: str | Path | None = None) -> pd.DataFrame:
    """Load the refreshed Diebold-Mariano test table."""
    return pd.read_csv(path or RESULTS / "tables" / "dm_tests_refreshed.csv")


def load_best_hyperparameters(path: str | Path | None = None) -> dict:
    """Load Optuna-selected hyperparameters."""
    with open(path or RESULTS / "hyperparameters" / "best_hyperparameters.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


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
    print("Best test rows by MAE:")
    print(best_test_rows().to_string(index=False))
