"""Plain-Python extraction of notebooks/Close_paper_May23.ipynb.

The original notebook was written for Google Colab. Colab shell commands and Drive-mount commands are commented out below. Review repository-relative paths before using this script for a full local rerun.
"""

# %% Cell 0
# Colab-only: !pip -q install yfinance ta scikit-learn pandas numpy

# %% Cell 1
# Colab-only: from google.colab import drive
import os


# Colab-only: drive.mount('/content/drive', force_remount=True)
data_dir = '/content/drive/MyDrive/Close_res/'
folder_path=data_dir
# Colab-only: os.chdir('/content/drive/MyDrive/Close_res/')

# %% Cell 2
# =========================
# 1) Install dependencies
# =========================

# =========================
# 2) Imports
# =========================
import itertools
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

# technical indicators
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

# selection
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

# =========================
# 3) Config
# =========================
START = "2010-01-01"
END = None  # to today

# Robust ticker candidates for each index (Yahoo Finance)
INDEX_TICKERS = {
    "SP500": ["^GSPC"],
    "EUROSTOXX50": ["^STOXX50E", "^SX5E"],
    "NIKKEI225": ["^N225"],
}

OHLCV = ["Open", "High", "Low", "Close", "Volume"]

# =========================
# 4) Helpers
# =========================

def download_first_working(candidates, start=None, end=None):
    """
    Try tickers in order; return (ticker, cleaned OHLCV DataFrame).
    Uses auto_adjust=True so Close always exists (Adj Close is dropped).
    """
    for t in candidates:
        df = yf.download(
            t, start=start, end=end,
            auto_adjust=True,            # <-- key change
            progress=False,
            group_by="column",
            multi_level_index=False,     # <-- keep columns flat
            threads=False,
            actions=False
        )
        if df is None or df.empty:
            continue

        # Ensure standard columns exist; Volume may be NaN/0 for some indices
        for col in ["Open","High","Low","Close","Volume"]:
            if col not in df.columns:
                df[col] = np.nan

        df = df[["Open","High","Low","Close","Volume"]].copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index().dropna(subset=["Close"])

        if not df.empty:
            return t, df

    raise ValueError(f"No working ticker among: {candidates}")

def has_usable_volume(df: pd.DataFrame) -> bool:
    if "Volume" not in df.columns:
        return False
    v = df["Volume"]
    return not (v.isna().all() or (v.fillna(0) == 0).all())

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add indicators using standard parameters.
    SMA(5,20), EMA(12,26), MACD(12,26,9), RSI(14), Stoch(14,3),
    Boll(20,2), ADX(14), OBV (if volume), ROC(10), ATR(14)
    """
    out = df.copy()

    # Moving averages: keep windows small as requested
    out["sma_5"]  = SMAIndicator(close=out["Close"], window=5, fillna=False).sma_indicator()
    out["sma_20"] = SMAIndicator(close=out["Close"], window=20, fillna=False).sma_indicator()

    out["ema_12"] = EMAIndicator(close=out["Close"], window=12, fillna=False).ema_indicator()
    out["ema_26"] = EMAIndicator(close=out["Close"], window=26, fillna=False).ema_indicator()

    macd = MACD(close=out["Close"], window_slow=26, window_fast=12, window_sign=9, fillna=False)
    out["macd"]        = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_hist"]   = macd.macd_diff()

    out["rsi_14"] = RSIIndicator(close=out["Close"], window=14, fillna=False).rsi()

    stoch = StochasticOscillator(
        high=out["High"], low=out["Low"], close=out["Close"],
        window=14, smooth_window=3, fillna=False
    )
    out["stoch_k_14_3"] = stoch.stoch()
    out["stoch_d_14_3"] = stoch.stoch_signal()

    bb = BollingerBands(close=out["Close"], window=20, window_dev=2, fillna=False)
    out["bb_m_20_2"] = bb.bollinger_mavg()
    out["bb_h_20_2"] = bb.bollinger_hband()
    out["bb_l_20_2"] = bb.bollinger_lband()
    # Optional: bandwidth and %B
    out["bb_bw_20_2"] = (out["bb_h_20_2"] - out["bb_l_20_2"]) / out["bb_m_20_2"]
    # PercentB: position within bands
    out["bb_pctb_20_2"] = (out["Close"] - out["bb_l_20_2"]) / (out["bb_h_20_2"] - out["bb_l_20_2"])

    out["adx_14"] = ADXIndicator(
        high=out["High"], low=out["Low"], close=out["Close"], window=14, fillna=False
    ).adx()

    # OBV only if volume looks usable
    if has_usable_volume(out):
        out["obv"] = OnBalanceVolumeIndicator(close=out["Close"], volume=out["Volume"], fillna=False).on_balance_volume()
    else:
        out["obv"] = np.nan  # will be dropped later if all-NaN

    out["roc_10"] = ROCIndicator(close=out["Close"], window=10, fillna=False).roc()
    out["atr_14"] = AverageTrueRange(
        high=out["High"], low=out["Low"], close=out["Close"], window=14, fillna=False
    ).average_true_range()

    # Drop pure-NaN columns (e.g., OBV when volume unusable)
    out = out.loc[:, ~out.isna().all()]

    return out

def make_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add next-day targets (aligned so that features at t map to target at t+1)
    """
    out = df.copy()
    out["y_close_t+1"] = out["Close"].shift(-1)
    out["y_logret_t+1"] = np.log(out["Close"].shift(-1) / out["Close"])
    return out

def time_series_train_test_split(df: pd.DataFrame, train_frac=0.8):
    n = len(df)
    cut = int(n * train_frac)
    train = df.iloc[:cut].copy()
    test  = df.iloc[cut:].copy()
    return train, test

def corr_and_lasso_select(train_df: pd.DataFrame, base_cols, cand_cols, target_col,
                          top_k_corr=12, corr_threshold=0.02, collinear_threshold=0.95,
                          n_splits=5, random_state=0):
    """
    1) keep indicators with |corr(y)| above threshold,
       take top_k by absolute correlation (cap),
       and drop collinear ones (|corr|>collinear_threshold).
    2) run LassoCV (with TimeSeriesSplit) on standardized features to prune further.
    Always keep base_cols (OHLCV).
    Returns: list of selected indicator columns (not including base_cols).
    """
    # --- 1) correlation screen ---
    corr_vals = (
        train_df[cand_cols + [target_col]]
        .dropna()
        .corr()[target_col]
        .drop(labels=[target_col])
        .abs()
        .sort_values(ascending=False)
    )
    corr_keep = corr_vals[corr_vals >= corr_threshold].index.tolist()
    corr_keep = corr_keep[:top_k_corr] if top_k_corr is not None else corr_keep
    if not corr_keep:
        return []  # nothing passes minimum signal

    # drop collinear features among corr_keep
    subset = train_df[corr_keep].dropna()
    to_drop = set()
    if subset.shape[1] > 1:
        cm = subset.corr().abs()
        # upper triangle indices
        for i in range(len(cm.columns)):
            for j in range(i+1, len(cm.columns)):
                if cm.iloc[i, j] > collinear_threshold:
                    # drop less correlated-to-target one
                    ci, cj = cm.columns[i], cm.columns[j]
                    if corr_vals[ci] >= corr_vals[cj]:
                        to_drop.add(cj)
                    else:
                        to_drop.add(ci)
    pruned = [c for c in corr_keep if c not in to_drop]

    # --- 2) LassoCV on pruned set ---
    if not pruned:
        return []

    X = train_df[base_cols + pruned]
    y = train_df[target_col]
    tmp = pd.concat([X, y], axis=1).dropna()
    X, y = tmp[base_cols + pruned], tmp[target_col]

    if len(tmp) < 100 or len(pruned) == 0:
        # not enough data for CV; accept pruned list
        return pruned

    tscv = TimeSeriesSplit(n_splits=min(n_splits, len(tmp)//50 if len(tmp)//50 >= 3 else 3))
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lasso",  LassoCV(alphas=np.logspace(-4, 2, 50), cv=tscv, max_iter=20000, random_state=random_state)),
    ])
    pipe.fit(X, y)

    # coefficients for indicators only (exclude base OHLCV from selection)
    coef = pipe.named_steps["lasso"].coef_
    feat_names = X.columns.tolist()
    # map feature -> coef
    keep = []
    for name, b in zip(feat_names, coef):
        if name in pruned and np.abs(b) > 1e-8:
            keep.append(name)

    # fall back to pruned if Lasso zeros everything
    return keep if keep else pruned

def build_and_save_scenarios(index_name, df_raw):
    """
    For one index:
      - compute indicators
      - add targets
      - build Scenario A/B/C/D dataframes (drop initial NaNs from rolling indicators)
      - save to CSV
    """
    df = compute_indicators(df_raw)
    df = make_targets(df)

    # --- define columns ---
    base_ohlcv = [c for c in OHLCV if c in df.columns]  # Volume may be present or not
    indicator_cols = [c for c in df.columns if c not in (["Adj Close"] + base_ohlcv + ["y_close_t+1","y_logret_t+1"])]

    # Drop rows that have NaNs produced by indicator warmups
    df_clean = df.dropna(subset=["y_close_t+1","y_logret_t+1"]).copy()
    # Be conservative: drop rows where any selected indicator NaN
    df_clean = df_clean.dropna(subset=list(set(indicator_cols)))  # if all volume-based features are NaN, they were removed upstream

    # --- Scenario A: Close only ---
    A = df_clean[["Close", "y_close_t+1", "y_logret_t+1"]].copy()
    A.to_csv(f"{index_name}_scenario_A.csv")

    # --- Scenario B: OHLCV ---
    B_cols = list(dict.fromkeys(base_ohlcv + ["y_close_t+1", "y_logret_t+1"]))
    B = df_clean[B_cols].copy()
    B.to_csv(f"{index_name}_scenario_B.csv")

    # --- Scenario C: OHLCV + ALL indicators ---
    C_cols = list(dict.fromkeys(base_ohlcv + indicator_cols + ["y_close_t+1", "y_logret_t+1"]))
    C = df_clean[C_cols].copy()
    C.to_csv(f"{index_name}_scenario_C.csv")

    # --- Scenario D: OHLCV + FILTERED indicators (train-only selection) ---
    # Split train/test chronologically for selection (80/20)
    train_df, test_df = time_series_train_test_split(df_clean, train_frac=0.8)

    selected_inds = corr_and_lasso_select(
        train_df=train_df,
        base_cols=base_ohlcv,
        cand_cols=indicator_cols,
        target_col="y_logret_t+1",     # selection on next-day log return (stationary-ish)
        top_k_corr=12,
        corr_threshold=0.02,
        collinear_threshold=0.95,
        n_splits=5,
        random_state=0
    )
    D_cols = list(dict.fromkeys(base_ohlcv + selected_inds + ["y_close_t+1", "y_logret_t+1"]))
    D = df_clean[D_cols].copy()
    D.to_csv(f"{index_name}_scenario_D.csv")

    print(f"\n[{index_name}] usable rows: {len(df_clean):,}")
    print(f"[{index_name}] OHLCV columns used: {base_ohlcv}")
    print(f"[{index_name}] Indicators computed ({len(indicator_cols)}): {indicator_cols}")
    print(f"[{index_name}] FILTERED indicators kept for Scenario D ({len(selected_inds)}): {selected_inds}")
    print(f"[{index_name}] Saved CSVs -> {index_name}_scenario_[A|B|C|D].csv")

# =========================
# 5) Run for all indices
# =========================
results = {}
for name, candidates in INDEX_TICKERS.items():
    ticker, df_raw = download_first_working(candidates, start=START, end=END)
    print(f"{name}: using ticker {ticker}, rows={len(df_raw):,}, date range {df_raw.index.min().date()} → {df_raw.index.max().date()}")
    build_and_save_scenarios(name, df_raw)

# %% Cell 3
# =========================================
# Leakage-safe normalization + time splits
# =========================================
# Colab-only: !pip -q install pandas numpy scikit-learn

import os, json, math
import numpy as np
import pandas as pd

# --------- Config ---------
BASE_DIR = ""                    # where your scenario CSVs are
OUT_DIR  = "/content/drive/MyDrive/Close_res/prepared"
INDEXES  = ["SP500", "EUROSTOXX50", "NIKKEI225"]
SCENARIOS = ["A", "B", "C", "D"]

WINDOW = 60                              # rolling window length for μ/σ
TRAIN_FRAC = 0.70                        # chronological split
VAL_FRAC   = 0.10                        # test implicitly = 1 - TRAIN - VAL

TARGET_COLS = ["y_close_t+1", "y_logret_t+1"]   # kept raw (not normalized)

os.makedirs(OUT_DIR, exist_ok=True)

def _ensure_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing input: {path} — run the earlier feature script first.")

def _rolling_norm_no_leak(df, feature_cols, window=60):
    """
    Leakage-safe rolling z-score:
      z_t = (x_t - mean_{t-1..t-W}) / std_{t-1..t-W}
    Returns:
      z_df  : normalized dataframe with original target columns preserved
      mu_df : rolling means aligned to z_df.index (columns 'mu_<feature>')
      sd_df : rolling stds   aligned to z_df.index (columns 'sd_<feature>')
    """
    # compute μ and σ from strictly prior data
    mu = df[feature_cols].shift(1).rolling(window=window, min_periods=window).mean()
    sd = df[feature_cols].shift(1).rolling(window=window, min_periods=window).std(ddof=0)

    # avoid divide-by-zero
    sd_repl = sd.replace(0, np.nan)

    z = (df[feature_cols] - mu) / sd_repl

    # Align to dates where all features have valid z AND targets exist
    valid_idx = z.dropna().index
    if TARGET_COLS:
        valid_idx = valid_idx.intersection(df.dropna(subset=TARGET_COLS).index)

    z = z.loc[valid_idx]
    mu = mu.loc[valid_idx]
    sd = sd.loc[valid_idx]

    # Rename μ/σ columns for saving
    mu.columns = [f"mu_{c}" for c in mu.columns]
    sd.columns = [f"sd_{c}" for c in sd.columns]

    # Reattach targets (unscaled)
    out = pd.concat([z, df.loc[valid_idx, TARGET_COLS]], axis=1)

    return out, mu, sd

def _chronological_splits(df, train_frac=0.7, val_frac=0.1):
    n = len(df)
    assert 0 < train_frac < 1 and 0 <= val_frac < 1 and train_frac + val_frac < 1, "Bad split fractions."
    i_train_end = int(math.floor(n * train_frac))
    i_val_end   = int(math.floor(n * (train_frac + val_frac)))
    train = df.iloc[:i_train_end].copy()
    val   = df.iloc[i_train_end:i_val_end].copy()
    test  = df.iloc[i_val_end:].copy()
    return train, val, test

def _save_split_frames(base_path, df_full, mu_df, sd_df, feature_cols, target_cols):
    os.makedirs(base_path, exist_ok=True)

    # Save normalized full dataset
    df_full.to_csv(os.path.join(base_path, "normalized_full.csv.gz"), index=True)

    # Save rolling μ/σ per timestamp
    roll_stats = pd.concat([mu_df, sd_df], axis=1)
    roll_stats.to_csv(os.path.join(base_path, "rolling_stats.csv.gz"), index=True)

    # Chronological splits
    train, val, test = _chronological_splits(df_full, TRAIN_FRAC, VAL_FRAC)

    # Feature/target lists
    meta = {
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "window": WINDOW,
        "splits": {
            "train": {"rows": len(train), "start": str(train.index.min()), "end": str(train.index.max()) if len(train) else None},
            "val":   {"rows": len(val), "start": str(val.index.min()),     "end": str(val.index.max()) if len(val) else None},
            "test":  {"rows": len(test), "start": str(test.index.min()),    "end": str(test.index.max()) if len(test) else None},
        }
    }

    # Train-only global μ/σ snapshot (not used for z_t, but handy to keep)
    if len(train):
        train_mu = train[feature_cols].mean(numeric_only=True)
        train_sd = train[feature_cols].std(ddof=0, numeric_only=True).replace(0, np.nan)
        meta["train_global_mu"] = {c: (None if pd.isna(v) else float(v)) for c, v in train_mu.items()}
        meta["train_global_sd"] = {c: (None if pd.isna(v) else float(v)) for c, v in train_sd.items()}

        # Also save as a separate JSON for quick reuse
        with open(os.path.join(base_path, "train_scaler.json"), "w") as f:
            json.dump({
                "mu": meta["train_global_mu"],
                "sd": meta["train_global_sd"],
                "window": WINDOW
            }, f, indent=2)

    # Save splits
    train.to_csv(os.path.join(base_path, "train.csv.gz"))
    val.to_csv(os.path.join(base_path, "val.csv.gz"))
    test.to_csv(os.path.join(base_path, "test.csv.gz"))

    # Save meta
    with open(os.path.join(base_path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return len(train), len(val), len(test)

# --------------- Run ---------------
for idx in INDEXES:
    for sc in SCENARIOS:
        in_path = f"{idx}_scenario_{sc}.csv"
        _ensure_exists(in_path)
        df = pd.read_csv(in_path, parse_dates=[0], index_col=0)

        # Identify features to normalize (everything except targets)
        cols = df.columns.tolist()
        feature_cols = [c for c in cols if c not in TARGET_COLS]

        # Apply leakage-safe rolling normalization
        z_df, mu_df, sd_df = _rolling_norm_no_leak(df, feature_cols, window=WINDOW)

        # Save normalized full + splits + rolling stats + meta
        base_out = f"{OUT_DIR}/{idx}/scenario_{sc}"
        ntr, nv, nts = _save_split_frames(base_out, z_df, mu_df, sd_df, feature_cols, TARGET_COLS)

        print(f"[{idx} - {sc}] rows after warm-up drop: {len(z_df):,}  | train={ntr:,} val={nv:,} test={nts:,}")
        print(f"  -> Saved to: {base_out}")

# %% Cell 4
# ================================================================
# PHASE 1: Hyperparameter tuning with Optuna (seed=42 only)
# Tunes 4 architectures × 3 scenarios × 3 indices = 36 configs
# Saves best hyperparameters as JSON
# ================================================================
# Colab-only: !pip -q install torch torchvision torchaudio numpy pandas scipy optuna

import os, json, gc, random
import numpy as np
import pandas as pd
from typing import List, Dict
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# -----------------------------
# Config
# -----------------------------
PREP_DIR   = "/content/drive/MyDrive/Close_res/prepared"
OUT_DIR    = "/content/drive/MyDrive/Close_res/optuna_results"
INDEXES    = ["SP500", "EUROSTOXX50", "NIKKEI225"]
SCENARIOS  = ["A", "B", "C"]
ARCHS      = ["LSTM", "BiLSTM", "TCN", "Transformer"]
TARGET_COL = "y_close_t+1"
N_TRIALS   = 30          # Optuna trials per config
TUNING_SEED = 42
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MAX_EPOCHS_TUNE = 30     # shorter during tuning for speed; final eval uses 40

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Reproducibility
# -----------------------------
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------
# IO helpers (same as original)
# -----------------------------
def load_split(idx, sc, split):
    return pd.read_csv(f"{PREP_DIR}/{idx}/scenario_{sc}/{split}.csv.gz",
                       parse_dates=[0], index_col=0).sort_index()

def load_roll(idx, sc):
    return pd.read_csv(f"{PREP_DIR}/{idx}/scenario_{sc}/rolling_stats.csv.gz",
                       parse_dates=[0], index_col=0).sort_index()

def reconstruct_raw_close(df, roll):
    return df["Close"] * roll.loc[df.index, "sd_Close"] + roll.loc[df.index, "mu_Close"]

def feature_columns(df):
    return [c for c in df.columns if c not in ["y_close_t+1", "y_logret_t+1"]]

# -----------------------------
# Dataset (z-target, same as original)
# -----------------------------
class SeqDataset(Dataset):
    def __init__(self, df, roll, lookback, target_col):
        self.df = df.copy()
        self.roll = roll.loc[df.index]
        self.L = lookback
        self.target_col = target_col
        self.features = feature_columns(df)

        self.mu = self.roll["mu_Close"].astype(np.float32).values
        self.sd = self.roll["sd_Close"].astype(np.float32).values
        self.last_close_raw = reconstruct_raw_close(self.df, self.roll).astype(np.float32).values
        self.y_raw = self.df[self.target_col].astype(np.float32).values

        self.X, self.y_z, self.mu_t, self.sd_t, self.last_t, self.y_raw_t1 = self._build()

    def _build(self):
        Xs, yz, mu, sd, lastc, yraw = [], [], [], [], [], []
        vals = self.df[self.features].values.astype(np.float32)
        sd_safe = np.where(np.isfinite(self.sd) & (self.sd != 0), self.sd, np.nan)

        for i in range(self.L - 1, len(self.df)):
            mu_i = self.mu[i]; sd_i = sd_safe[i]
            if not np.isfinite(sd_i):
                continue
            Xs.append(vals[i - self.L + 1:i + 1, :])
            yz.append((self.y_raw[i] - mu_i) / sd_i)
            mu.append(mu_i); sd.append(sd_i)
            lastc.append(self.last_close_raw[i])
            yraw.append(self.y_raw[i])

        return (np.array(Xs, dtype=np.float32),
                np.array(yz, dtype=np.float32).reshape(-1, 1),
                np.array(mu, dtype=np.float32).reshape(-1, 1),
                np.array(sd, dtype=np.float32).reshape(-1, 1),
                np.array(lastc, dtype=np.float32).reshape(-1, 1),
                np.array(yraw, dtype=np.float32).reshape(-1, 1))

    def __len__(self): return len(self.y_z)
    def __getitem__(self, i):
        return (self.X[i], self.y_z[i], self.mu_t[i], self.sd_t[i], self.last_t[i], self.y_raw_t1[i])

# -----------------------------
# Models — original 3 + vanilla Transformer
# -----------------------------
class LSTMModel(nn.Module):
    def __init__(self, n_feats, h1=96, h2=64, drop=0.2):
        super().__init__()
        self.l1 = nn.LSTM(n_feats, h1, batch_first=True)
        self.do1 = nn.Dropout(drop)
        self.l2 = nn.LSTM(h1, h2, batch_first=True)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(h2, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        x, _ = self.l1(x); x = self.do1(x)
        x, _ = self.l2(x)
        return self.head(x[:, -1, :])

class BiLSTMModel(nn.Module):
    def __init__(self, n_feats, h1=96, h2=64, drop=0.2):
        super().__init__()
        self.bi1 = nn.LSTM(n_feats, h1, batch_first=True, bidirectional=True)
        self.do1 = nn.Dropout(drop)
        self.bi2 = nn.LSTM(2 * h1, h2, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(2 * h2, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        x, _ = self.bi1(x); x = self.do1(x)
        x, _ = self.bi2(x)
        return self.head(x[:, -1, :])

class TCNBlock(nn.Module):
    def __init__(self, ch, dil, drop=0.2):
        super().__init__()
        pad = dil
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=pad, dilation=dil), nn.ReLU(), nn.Dropout(drop),
            nn.Conv1d(ch, ch, 3, padding=pad, dilation=dil), nn.ReLU(), nn.Dropout(drop),
        )
    def forward(self, x): return x + self.net(x)

class TCN(nn.Module):
    def __init__(self, n_feats, channels=64, drop=0.2):
        super().__init__()
        self.proj = nn.Conv1d(n_feats, channels, 1)
        self.stack = nn.Sequential(*[TCNBlock(channels, d, drop) for d in [1, 2, 4, 8, 16, 32]])
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                   nn.Linear(channels, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = self.stack(x)
        return self.head(x)

class VanillaTransformer(nn.Module):
    """
    Vanilla encoder-only Transformer for short-horizon forecasting.
    Input projection -> learned positional encoding -> N encoder layers -> last-token head.
    """
    def __init__(self, n_feats, d_model=64, n_heads=4, n_layers=2, drop=0.2, max_len=20):
        super().__init__()
        # Ensure d_model is divisible by n_heads
        assert d_model % n_heads == 0, f"d_model={d_model} must be divisible by n_heads={n_heads}"
        self.input_proj = nn.Linear(n_feats, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=4 * d_model, dropout=drop,
            batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1)
        )
    def forward(self, x):  # [B, L, F]
        x = self.input_proj(x)
        x = x + self.pos_enc[:, :x.size(1), :]
        x = self.encoder(x)
        return self.head(x[:, -1, :])

def build_model(name, n_feats, hp):
    """Build a model from architecture name and hyperparameter dict."""
    if name == "LSTM":
        return LSTMModel(n_feats, h1=hp["h1"], h2=hp["h2"], drop=hp["dropout"]).to(DEVICE)
    if name == "BiLSTM":
        return BiLSTMModel(n_feats, h1=hp["h1"], h2=hp["h2"], drop=hp["dropout"]).to(DEVICE)
    if name == "TCN":
        return TCN(n_feats, channels=hp["channels"], drop=hp["dropout"]).to(DEVICE)
    if name == "Transformer":
        return VanillaTransformer(
            n_feats, d_model=hp["d_model"], n_heads=hp["n_heads"],
            n_layers=hp["n_layers"], drop=hp["dropout"], max_len=20
        ).to(DEVICE)
    raise ValueError(name)

# -----------------------------
# Training (with optional pruning)
# -----------------------------
def train_with_pruning(model, loaders, hp, trial=None, max_epochs=MAX_EPOCHS_TUNE):
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    loss_fn = nn.SmoothL1Loss()
    best_val = np.inf; bad = 0; best_state = None
    patience = hp.get("patience", 10)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    for ep in range(1, max_epochs + 1):
        model.train()
        for xb, yb_z, _, _, _, _ in loaders["train"]:
            xb, yb_z = xb.to(DEVICE), yb_z.to(DEVICE)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                pred_z = model(xb)
                loss = loss_fn(pred_z, yb_z)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()

        # Validation
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for xb, yb_z, _, _, _, _ in loaders["val"]:
                xb, yb_z = xb.to(DEVICE), yb_z.to(DEVICE)
                pred_z = model(xb)
                val_loss += loss_fn(pred_z, yb_z).item() * len(xb)
        val_loss /= max(1, len(loaders["val"].dataset))

        # Optuna pruning hook
        if trial is not None:
            trial.report(val_loss, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if val_loss + 1e-9 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience: break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val

def evaluate_val_mae(model, loader):
    """Compute val MAE in raw (un-normalized) space."""
    model.eval()
    y_all, yhat_all = [], []
    with torch.no_grad():
        for xb, _, mu, sd, _, yb_raw in loader:
            xb = xb.to(DEVICE)
            pred_z = model(xb).cpu().numpy().reshape(-1, 1)
            yhat = mu.numpy() + sd.numpy() * pred_z
            y_all.append(yb_raw.numpy().reshape(-1))
            yhat_all.append(yhat.reshape(-1))
    y = np.concatenate(y_all); yhat = np.concatenate(yhat_all)
    return float(np.mean(np.abs(y - yhat)))

# -----------------------------
# Optuna objective
# -----------------------------
def make_loaders(df_tr, df_va, roll, L, target_col, batch_size, seed):
    ds_tr = SeqDataset(df_tr, roll, L, target_col)
    ds_va = SeqDataset(df_va, roll, L, target_col)
    g = torch.Generator(); g.manual_seed(seed)
    return ds_tr, ds_va, {
        "train": DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=False, generator=g),
        "val":   DataLoader(ds_va, batch_size=batch_size, shuffle=False, drop_last=False),
    }

def make_objective(arch, idx, sc):
    """Build an Optuna objective for a specific (arch, index, scenario)."""
    df_tr = load_split(idx, sc, "train")
    df_va = load_split(idx, sc, "val")
    roll  = load_roll(idx, sc)

    def objective(trial):
        # Common hyperparameters
        L          = trial.suggest_categorical("lookback", [2, 3, 5, 10, 20])
        lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        dropout    = trial.suggest_categorical("dropout", [0.0, 0.1, 0.2, 0.3])
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

        # Architecture-specific hyperparameters
        hp = {"lr": lr, "dropout": dropout, "batch_size": batch_size, "patience": 10}
        if arch in ["LSTM", "BiLSTM"]:
            hp["h1"] = trial.suggest_categorical("h1", [64, 96, 128])
            hp["h2"] = trial.suggest_categorical("h2", [32, 64, 96])
        elif arch == "TCN":
            hp["channels"] = trial.suggest_categorical("channels", [32, 64, 128])
        elif arch == "Transformer":
            hp["d_model"]  = trial.suggest_categorical("d_model", [32, 64, 128])
            hp["n_heads"]  = trial.suggest_categorical("n_heads", [2, 4, 8])
            hp["n_layers"] = trial.suggest_categorical("n_layers", [1, 2, 3])
            # Constraint: d_model must be divisible by n_heads
            if hp["d_model"] % hp["n_heads"] != 0:
                raise optuna.TrialPruned()

        set_all_seeds(TUNING_SEED)
        ds_tr, ds_va, loaders = make_loaders(df_tr, df_va, roll, L, TARGET_COL, batch_size, TUNING_SEED)
        if len(ds_va) == 0:
            raise optuna.TrialPruned()

        n_feats = len(feature_columns(df_tr))

        try:
            model = build_model(arch, n_feats, hp)
            model, _ = train_with_pruning(model, loaders, hp, trial=trial, max_epochs=MAX_EPOCHS_TUNE)
            val_mae = evaluate_val_mae(model, loaders["val"])
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"  Trial failed: {e}")
            raise optuna.TrialPruned()
        finally:
            if 'model' in locals():
                del model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        return val_mae

    return objective

# -----------------------------
# Run all studies
# -----------------------------
all_best = {}
log_file = open(f"{OUT_DIR}/tuning_log.txt", "w")

def log(msg):
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()

start_time = datetime.now()
log(f"Starting Optuna tuning at {start_time}")

for arch in ARCHS:
    for idx in INDEXES:
        for sc in SCENARIOS:
            config_id = f"{arch}_{idx}_{sc}"
            log(f"\n{'='*70}\nTuning: {config_id}\n{'='*70}")

            sampler = TPESampler(seed=TUNING_SEED)
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)
            study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

            try:
                study.optimize(
                    make_objective(arch, idx, sc),
                    n_trials=N_TRIALS,
                    show_progress_bar=False
                )
            except Exception as e:
                log(f"  Study failed: {e}")
                continue

            best = study.best_params
            best_val = study.best_value
            log(f"  Best val MAE: {best_val:.4f}")
            log(f"  Best params: {best}")

            all_best[config_id] = {
                "arch": arch, "index": idx, "scenario": sc,
                "best_val_mae": best_val,
                "best_params": best,
                "n_trials_completed": len(study.trials),
                "n_trials_pruned": sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED),
            }

            # Save incrementally so a crash doesn't lose progress
            with open(f"{OUT_DIR}/best_hyperparameters.json", "w") as f:
                json.dump(all_best, f, indent=2)

end_time = datetime.now()
log(f"\nFinished at {end_time}, total time: {end_time - start_time}")
log_file.close()

print(f"\nBest hyperparameters saved to: {OUT_DIR}/best_hyperparameters.json")
print(f"Tuning log: {OUT_DIR}/tuning_log.txt")

# %% Cell 5
# ================================================================
# PHASE 2: Multi-seed evaluation using best hyperparameters from Phase 1
# Runs 10 seeds (none = 42) for each of 36 configs
# Saves per-seed predictions and aggregated mean/std summary
# ================================================================
# Colab-only: !pip -q install torch torchvision torchaudio numpy pandas scipy

import os, json, gc, random
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Config
# -----------------------------
PREP_DIR     = "/content/drive/MyDrive/Close_res/prepared"
OPTUNA_DIR   = "/content/drive/MyDrive/Close_res/optuna_results"
OUT_DIR      = "/content/drive/MyDrive/Close_res/seeded_results"
INDEXES      = ["SP500", "EUROSTOXX50", "NIKKEI225"]
SCENARIOS    = ["A", "B", "C"]
ARCHS        = ["LSTM", "BiLSTM", "TCN", "Transformer"]
EVAL_SEEDS   = [123, 456, 789, 1024, 2025, 3141, 7777, 9001, 31337, 65537]  # ≠ 42
TARGET_COL   = "y_close_t+1"
MAX_EPOCHS   = 40
PATIENCE     = 10
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)

# Load best hyperparameters from Phase 1
with open(f"{OPTUNA_DIR}/best_hyperparameters.json") as f:
    best_hp = json.load(f)

print(f"Loaded {len(best_hp)} tuned configurations")

# -----------------------------
# Reproducibility (same as before)
# -----------------------------
def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------
# IO helpers
# -----------------------------
def load_split(idx, sc, split):
    return pd.read_csv(f"{PREP_DIR}/{idx}/scenario_{sc}/{split}.csv.gz",
                       parse_dates=[0], index_col=0).sort_index()

def load_roll(idx, sc):
    return pd.read_csv(f"{PREP_DIR}/{idx}/scenario_{sc}/rolling_stats.csv.gz",
                       parse_dates=[0], index_col=0).sort_index()

def reconstruct_raw_close(df, roll):
    return df["Close"] * roll.loc[df.index, "sd_Close"] + roll.loc[df.index, "mu_Close"]

def feature_columns(df):
    return [c for c in df.columns if c not in ["y_close_t+1", "y_logret_t+1"]]

def metrics(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    mape = float(100.0 * np.mean(np.abs((y_true - y_pred) / y_true)))
    # R²
    ss_res = float(np.sum((y_true - y_pred)**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

def medae_metric(y_true, y_pred):
    """MedAE: median absolute error (robust to outliers)."""
    return float(np.median(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

# -----------------------------
# Dataset, Models, Training (paste from Notebook 1)
# -----------------------------
class SeqDataset(Dataset):
    def __init__(self, df, roll, lookback, target_col):
        self.df = df.copy()
        self.roll = roll.loc[df.index]
        self.L = lookback
        self.target_col = target_col
        self.features = feature_columns(df)

        self.mu = self.roll["mu_Close"].astype(np.float32).values
        self.sd = self.roll["sd_Close"].astype(np.float32).values
        self.last_close_raw = reconstruct_raw_close(self.df, self.roll).astype(np.float32).values
        self.y_raw = self.df[self.target_col].astype(np.float32).values

        self.X, self.y_z, self.mu_t, self.sd_t, self.last_t, self.y_raw_t1 = self._build()

    def _build(self):
        Xs, yz, mu, sd, lastc, yraw = [], [], [], [], [], []
        vals = self.df[self.features].values.astype(np.float32)
        sd_safe = np.where(np.isfinite(self.sd) & (self.sd != 0), self.sd, np.nan)

        for i in range(self.L - 1, len(self.df)):
            mu_i = self.mu[i]; sd_i = sd_safe[i]
            if not np.isfinite(sd_i): continue
            Xs.append(vals[i - self.L + 1:i + 1, :])
            yz.append((self.y_raw[i] - mu_i) / sd_i)
            mu.append(mu_i); sd.append(sd_i)
            lastc.append(self.last_close_raw[i])
            yraw.append(self.y_raw[i])

        return (np.array(Xs, dtype=np.float32),
                np.array(yz, dtype=np.float32).reshape(-1, 1),
                np.array(mu, dtype=np.float32).reshape(-1, 1),
                np.array(sd, dtype=np.float32).reshape(-1, 1),
                np.array(lastc, dtype=np.float32).reshape(-1, 1),
                np.array(yraw, dtype=np.float32).reshape(-1, 1))

    def __len__(self): return len(self.y_z)
    def __getitem__(self, i):
        return (self.X[i], self.y_z[i], self.mu_t[i], self.sd_t[i], self.last_t[i], self.y_raw_t1[i])

class LSTMModel(nn.Module):
    def __init__(self, n_feats, h1=96, h2=64, drop=0.2):
        super().__init__()
        self.l1 = nn.LSTM(n_feats, h1, batch_first=True)
        self.do1 = nn.Dropout(drop)
        self.l2 = nn.LSTM(h1, h2, batch_first=True)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(h2, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        x, _ = self.l1(x); x = self.do1(x)
        x, _ = self.l2(x)
        return self.head(x[:, -1, :])

class BiLSTMModel(nn.Module):
    def __init__(self, n_feats, h1=96, h2=64, drop=0.2):
        super().__init__()
        self.bi1 = nn.LSTM(n_feats, h1, batch_first=True, bidirectional=True)
        self.do1 = nn.Dropout(drop)
        self.bi2 = nn.LSTM(2 * h1, h2, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(2 * h2, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        x, _ = self.bi1(x); x = self.do1(x)
        x, _ = self.bi2(x)
        return self.head(x[:, -1, :])

class TCNBlock(nn.Module):
    def __init__(self, ch, dil, drop=0.2):
        super().__init__()
        pad = dil
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=pad, dilation=dil), nn.ReLU(), nn.Dropout(drop),
            nn.Conv1d(ch, ch, 3, padding=pad, dilation=dil), nn.ReLU(), nn.Dropout(drop),
        )
    def forward(self, x): return x + self.net(x)

class TCN(nn.Module):
    def __init__(self, n_feats, channels=64, drop=0.2):
        super().__init__()
        self.proj = nn.Conv1d(n_feats, channels, 1)
        self.stack = nn.Sequential(*[TCNBlock(channels, d, drop) for d in [1, 2, 4, 8, 16, 32]])
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                   nn.Linear(channels, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = self.stack(x)
        return self.head(x)

class VanillaTransformer(nn.Module):
    def __init__(self, n_feats, d_model=64, n_heads=4, n_layers=2, drop=0.2, max_len=20):
        super().__init__()
        assert d_model % n_heads == 0
        self.input_proj = nn.Linear(n_feats, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=drop, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pos_enc[:, :x.size(1), :]
        x = self.encoder(x)
        return self.head(x[:, -1, :])

def build_model(name, n_feats, hp):
    if name == "LSTM":
        return LSTMModel(n_feats, h1=hp["h1"], h2=hp["h2"], drop=hp["dropout"]).to(DEVICE)
    if name == "BiLSTM":
        return BiLSTMModel(n_feats, h1=hp["h1"], h2=hp["h2"], drop=hp["dropout"]).to(DEVICE)
    if name == "TCN":
        return TCN(n_feats, channels=hp["channels"], drop=hp["dropout"]).to(DEVICE)
    if name == "Transformer":
        return VanillaTransformer(
            n_feats, d_model=hp["d_model"], n_heads=hp["n_heads"],
            n_layers=hp["n_layers"], drop=hp["dropout"], max_len=20
        ).to(DEVICE)
    raise ValueError(name)

def make_loaders(df_tr, df_va, df_te, roll, L, target_col, batch_size, seed):
    ds_tr = SeqDataset(df_tr, roll, L, target_col)
    ds_va = SeqDataset(df_va, roll, L, target_col)
    ds_te = SeqDataset(df_te, roll, L, target_col)
    g = torch.Generator(); g.manual_seed(seed)
    return ds_tr, ds_va, ds_te, {
        "train": DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=False, generator=g),
        "val":   DataLoader(ds_va, batch_size=batch_size, shuffle=False, drop_last=False),
        "test":  DataLoader(ds_te, batch_size=batch_size, shuffle=False, drop_last=False),
    }

def train_model(model, loaders, hp, max_epochs=MAX_EPOCHS):
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    loss_fn = nn.SmoothL1Loss()
    best_val = np.inf; bad = 0; best_state = None
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    for ep in range(1, max_epochs + 1):
        model.train()
        for xb, yb_z, _, _, _, _ in loaders["train"]:
            xb, yb_z = xb.to(DEVICE), yb_z.to(DEVICE)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                pred_z = model(xb); loss = loss_fn(pred_z, yb_z)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()

        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for xb, yb_z, _, _, _, _ in loaders["val"]:
                xb, yb_z = xb.to(DEVICE), yb_z.to(DEVICE)
                pred_z = model(xb)
                val_loss += loss_fn(pred_z, yb_z).item() * len(xb)
        val_loss /= max(1, len(loaders["val"].dataset))

        if val_loss + 1e-9 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE: break

    if best_state is not None: model.load_state_dict(best_state)
    return model

def predict_raw(model, loader):
    model.eval()
    y_all, yhat_all, last_all = [], [], []
    with torch.no_grad():
        for xb, _, mu, sd, lastc, yb_raw in loader:
            xb = xb.to(DEVICE)
            pred_z = model(xb).cpu().numpy().reshape(-1, 1)
            yhat = mu.numpy() + sd.numpy() * pred_z
            y_all.append(yb_raw.numpy().reshape(-1))
            yhat_all.append(yhat.reshape(-1))
            last_all.append(lastc.numpy().reshape(-1))
    return np.concatenate(y_all), np.concatenate(yhat_all), np.concatenate(last_all)

# -----------------------------
# Main loop: 10 seeds × 36 configs
# -----------------------------
results_rows = []
start_time = datetime.now()
print(f"Starting multi-seed evaluation at {start_time}")

for arch in ARCHS:
    for idx in INDEXES:
        for sc in SCENARIOS:
            config_id = f"{arch}_{idx}_{sc}"
            if config_id not in best_hp:
                print(f"⚠ Skipping {config_id} — no best hyperparameters found")
                continue

            hp = best_hp[config_id]["best_params"]
            L = hp["lookback"]
            batch_size = hp["batch_size"]

            print(f"\n--- {config_id} | best params: {hp} ---")

            df_tr = load_split(idx, sc, "train")
            df_va = load_split(idx, sc, "val")
            df_te = load_split(idx, sc, "test")
            roll  = load_roll(idx, sc)
            n_feats = len(feature_columns(df_tr))



            for seed in EVAL_SEEDS:
                set_all_seeds(seed)
                ds_tr, ds_va, ds_te, loaders = make_loaders(
                    df_tr, df_va, df_te, roll, L, TARGET_COL, batch_size, seed
                )
                if len(ds_va) == 0 or len(ds_te) == 0:
                    print(f"  seed={seed}: insufficient rows; skipping")
                    continue

                set_all_seeds(seed)  # re-seed for model init
                model = build_model(arch, n_feats, hp)
                model = train_model(model, loaders, hp)

                yv, pv, lv = predict_raw(model, loaders["val"])
                yt, pt, lt = predict_raw(model, loaders["test"])

                mv = metrics(yv, pv); mt = metrics(yt, pt)
                mv["MedAE"] = medae_metric(yv, pv)
                mt["MedAE"] = medae_metric(yt, pt)

                # Save predictions
                save_dir = os.path.join(OUT_DIR, arch, idx, sc, f"seed_{seed}")
                os.makedirs(save_dir, exist_ok=True)
                np.save(f"{save_dir}/y_val.npy", yv)
                np.save(f"{save_dir}/yhat_val.npy", pv)
                np.save(f"{save_dir}/y_test.npy", yt)
                np.save(f"{save_dir}/yhat_test.npy", pt)
                np.save(f"{save_dir}/last_close_test.npy", lt)

                results_rows.append({
                    "Arch": arch, "Index": idx, "Scenario": sc, "Seed": seed,
                    "Lookback": L, "Split": "val", **mv
                })
                results_rows.append({
                    "Arch": arch, "Index": idx, "Scenario": sc, "Seed": seed,
                    "Lookback": L, "Split": "test", **mt
                })

                print(f"  seed={seed} | val MAE={mv['MAE']:.2f} | test MAE={mt['MAE']:.2f} | test MedAE={mt['MedAE']:.3f}")

                del model
                if DEVICE == "cuda": torch.cuda.empty_cache()
                gc.collect()

# Save raw per-seed
all_results = pd.DataFrame(results_rows)
all_results.to_csv(f"{OUT_DIR}/per_seed_results.csv", index=False)
print(f"\nSaved per-seed: {OUT_DIR}/per_seed_results.csv")

# Aggregate mean ± std + 95% CI
def aggregate(df):
    grouped = df.groupby(["Arch", "Index", "Scenario", "Lookback", "Split"])
    agg = grouped.agg(
        MAE_mean=("MAE", "mean"),   MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        MAPE_mean=("MAPE", "mean"), MAPE_std=("MAPE", "std"),
        R2_mean=("R2", "mean"),     R2_std=("R2", "std"),
        MedAE_mean=("MedAE", "mean"), MedAE_std=("MedAE", "std"),
        n_seeds=("MAE", "count")
    ).reset_index()
    t_crit = stats.t.ppf(0.975, df=agg["n_seeds"] - 1)
    for m in ["MAE", "RMSE", "MAPE", "R2", "MedAE"]:
        agg[f"{m}_ci95"] = t_crit * agg[f"{m}_std"] / np.sqrt(agg["n_seeds"])
    return agg

agg = aggregate(all_results)
agg.to_csv(f"{OUT_DIR}/aggregated_summary.csv", index=False)
print(f"Saved aggregated: {OUT_DIR}/aggregated_summary.csv")

print(f"\nTotal time: {datetime.now() - start_time}")

# %% Cell 6
# ================================================================
# Diebold-Mariano refresh: 4 architectures × 3 indices × 2 pair-comparisons
# Uses median predictions across 10 seeds for robustness
# Date-aligns scenarios by intersection of sample dates (handles
# different lookbacks producing different test-period start rows)
# Includes HAC variance and Holm correction
# ================================================================
# Colab-only: !pip -q install numpy pandas scipy

import os, json
import numpy as np
import pandas as pd
from scipy.stats import norm

# -----------------------------
# Config
# -----------------------------
SEEDED_DIR  = "/content/drive/MyDrive/Close_res/seeded_results"
OPTUNA_DIR  = "/content/drive/MyDrive/Close_res/optuna_results"
PREP_DIR    = "/content/drive/MyDrive/Close_res/prepared"
OUT_DIR     = "/content/drive/MyDrive/Close_res/dm_results"
INDEXES     = ["SP500", "EUROSTOXX50", "NIKKEI225"]
SCENARIOS   = ["A", "B", "C"]
ARCHS       = ["LSTM", "BiLSTM", "TCN", "Transformer"]
SEEDS       = [123, 456, 789, 1024, 2025, 3141, 7777, 9001, 31337, 65537]
TARGET_COL  = "y_close_t+1"

os.makedirs(OUT_DIR, exist_ok=True)

# Load best hyperparameters (for lookback values)
with open(f"{OPTUNA_DIR}/best_hyperparameters.json") as f:
    best_hp = json.load(f)

# -----------------------------
# DM test with HAC (Newey-West) variance
# -----------------------------
def dm_test(y, yhat1, yhat2, loss="ae", nw_lags="auto"):
    """
    Diebold-Mariano test with Newey-West (Bartlett) variance.
    Returns (statistic, p-value, T, dbar).
    Negative statistic + small p-value => model 1 (yhat1) significantly better than model 2 (yhat2).
    """
    y = np.asarray(y, float)
    e1 = y - np.asarray(yhat1, float)
    e2 = y - np.asarray(yhat2, float)
    if loss == "ae":
        L1, L2 = np.abs(e1), np.abs(e2)
    elif loss == "se":
        L1, L2 = e1**2, e2**2
    else:
        raise ValueError("loss must be 'ae' or 'se'")
    d = L1 - L2
    T = len(d)
    if T < 5:
        return np.nan, np.nan, T, np.nan

    dbar = np.mean(d)
    if nw_lags == "auto":
        L = int(np.floor(1.5 * (T ** (1/3))))
    else:
        L = int(nw_lags)

    d_center = d - dbar
    gamma0 = np.dot(d_center, d_center) / T
    s = gamma0
    for k in range(1, min(L, T-1) + 1):
        cov = np.dot(d_center[k:], d_center[:-k]) / T
        w = 1.0 - k/(L+1.0)
        s += 2.0 * w * cov
    var_dbar = s / T
    if var_dbar <= 0 or not np.isfinite(var_dbar):
        return np.nan, np.nan, T, dbar

    stat = dbar / np.sqrt(var_dbar)
    pval = 2.0 * (1.0 - norm.cdf(np.abs(stat)))
    return float(stat), float(pval), int(T), float(dbar)

# -----------------------------
# Date reconstruction helpers
# -----------------------------
def load_test_split(idx, sc):
    return pd.read_csv(f"{PREP_DIR}/{idx}/scenario_{sc}/test.csv.gz",
                       parse_dates=[0], index_col=0).sort_index()

def load_rolling_stats(idx, sc):
    return pd.read_csv(f"{PREP_DIR}/{idx}/scenario_{sc}/rolling_stats.csv.gz",
                       parse_dates=[0], index_col=0).sort_index()

def dataset_sample_dates(df_test, roll_test, L):
    """
    Reconstruct the dates at which a SeqDataset with lookback L emits predictions.
    Mirrors SeqDataset._build: starts at row L-1, skips rows where sd is invalid.
    """
    sd = roll_test.loc[df_test.index, "sd_Close"].values.astype(float)
    valid = np.isfinite(sd) & (sd != 0)
    return pd.Index([df_test.index[i] for i in range(L - 1, len(df_test)) if valid[i]])

# -----------------------------
# Load and median-aggregate predictions across 10 seeds, with dates
# -----------------------------
def load_median_preds_dated(arch, idx, sc):
    """
    Load test predictions across 10 seeds, median them, and attach the
    correct date index (reconstructed from prepared test CSV + the
    Optuna-selected lookback for this configuration).
    Returns a DataFrame indexed by date with columns ['y', 'yhat'], or None.
    """
    config_id = f"{arch}_{idx}_{sc}"
    if config_id not in best_hp:
        return None
    L = best_hp[config_id]["best_params"]["lookback"]

    preds_per_seed = []
    y_true = None
    for seed in SEEDS:
        path = f"{SEEDED_DIR}/{arch}/{idx}/{sc}/seed_{seed}"
        if not os.path.exists(f"{path}/yhat_test.npy"):
            continue
        yh = np.load(f"{path}/yhat_test.npy")
        preds_per_seed.append(yh)
        if y_true is None:
            y_true = np.load(f"{path}/y_test.npy")
    if not preds_per_seed:
        return None

    preds_stack = np.stack(preds_per_seed, axis=0)
    median_preds = np.median(preds_stack, axis=0)

    # Reconstruct dates for this (arch, idx, sc, L) configuration
    df_test = load_test_split(idx, sc)
    roll_test = load_rolling_stats(idx, sc)
    dates = dataset_sample_dates(df_test, roll_test, L)

    # Align lengths defensively (in case of off-by-one in warm-up)
    n = min(len(dates), len(y_true), len(median_preds))
    return pd.DataFrame(
        {"y": y_true[:n].reshape(-1), "yhat": median_preds[:n].reshape(-1)},
        index=dates[:n]
    )

# -----------------------------
# Run DM tests: A vs B and A vs C, for each architecture × index
# -----------------------------
rows = []
pairs = [("A", "B"), ("A", "C")]

for arch in ARCHS:
    for idx in INDEXES:
        for scA, scB in pairs:
            A = load_median_preds_dated(arch, idx, scA)
            B = load_median_preds_dated(arch, idx, scB)
            if A is None or B is None:
                print(f"⚠ Missing preds for {arch} {idx} {scA}/{scB} — skipping")
                continue

            # Date-based intersection (handles different lookbacks correctly)
            common = A.index.intersection(B.index)
            if len(common) < 20:
                print(f"⚠ Too few aligned dates for {arch} {idx} {scA}/{scB} ({len(common)}) — skipping")
                continue
            A_aligned = A.loc[common]
            B_aligned = B.loc[common]

            # Sanity check: y_true should now match exactly across scenarios
            if not np.allclose(A_aligned["y"].values, B_aligned["y"].values):
                print(f"⚠ y_true mismatch for {arch} {idx} {scA}/{scB} after date alignment — skipping")
                continue

            yA  = A_aligned["y"].values
            yhA = A_aligned["yhat"].values
            yhB = B_aligned["yhat"].values

            stat_ae, p_ae, T, dbar_ae = dm_test(yA, yhA, yhB, loss="ae")
            stat_se, p_se, _,  dbar_se = dm_test(yA, yhA, yhB, loss="se")

            mae_A = float(np.mean(np.abs(yA - yhA)))
            mae_B = float(np.mean(np.abs(yA - yhB)))
            winner = scA if mae_A < mae_B else scB

            rows.append({
                "Arch": arch, "Index": idx,
                "Pair": f"{scA} vs {scB}",
                "Aligned_T": T,
                "MAE_A": mae_A, "MAE_B": mae_B,
                "DM_stat_AE": stat_ae, "p_AE_raw": p_ae, "dbar_AE": dbar_ae,
                "DM_stat_SE": stat_se, "p_SE_raw": p_se, "dbar_SE": dbar_se,
                "Winner_by_MAE": winner,
                "n_seeds_used": len(SEEDS),
            })

dm_df = pd.DataFrame(rows)

# -----------------------------
# Multiple-comparison correction
# -----------------------------
def holm_bonferroni(pvals):
    """Apply Holm-Bonferroni step-down correction. Returns adjusted p-values."""
    pvals = np.asarray(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(n)
    prev = 0
    for rank, i in enumerate(order):
        adj = pvals[i] * (n - rank)
        adj = min(adj, 1.0)
        adj = max(adj, prev)  # monotonic
        adjusted[i] = adj
        prev = adj
    return adjusted

if len(dm_df):
    dm_df["p_AE_holm"]       = holm_bonferroni(dm_df["p_AE_raw"].values)
    dm_df["p_SE_holm"]       = holm_bonferroni(dm_df["p_SE_raw"].values)
    dm_df["p_AE_bonferroni"] = np.minimum(dm_df["p_AE_raw"] * len(dm_df), 1.0)

# -----------------------------
# Save & print summary
# -----------------------------
dm_path = f"{OUT_DIR}/dm_tests_refreshed.csv"
dm_df.to_csv(dm_path, index=False)
print(f"\nSaved: {dm_path}\n")

print("=" * 80)
print("DM TEST SUMMARY")
print("=" * 80)
total       = len(dm_df)
sig_raw_05  = (dm_df["p_AE_raw"]  < 0.05).sum()
sig_holm_05 = (dm_df["p_AE_holm"] < 0.05).sum()
print(f"Total comparisons: {total}")
print(f"Significant at p<0.05 (raw):  {sig_raw_05} / {total}")
print(f"Significant at p<0.05 (Holm): {sig_holm_05} / {total}")
print()

print("Full results table (sorted by raw p-value):")
display_cols = ["Arch", "Index", "Pair", "MAE_A", "MAE_B", "Winner_by_MAE",
                "DM_stat_AE", "p_AE_raw", "p_AE_holm"]
print(dm_df.sort_values("p_AE_raw")[display_cols].to_string(index=False))

# %% Cell 7
import torch
import json
import torch
import torch.nn as nn
import json
import pandas as pd
# Load best hyperparameters from Optuna
with open("/content/drive/MyDrive/Close_res/optuna_results/best_hyperparameters.json") as f:
    best_hp = json.load(f)

# -----------------------------
# Models — original 3 + vanilla Transformer
# -----------------------------
class LSTMModel(nn.Module):
    def __init__(self, n_feats, h1=96, h2=64, drop=0.2):
        super().__init__()
        self.l1 = nn.LSTM(n_feats, h1, batch_first=True)
        self.do1 = nn.Dropout(drop)
        self.l2 = nn.LSTM(h1, h2, batch_first=True)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(h2, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        x, _ = self.l1(x); x = self.do1(x)
        x, _ = self.l2(x)
        return self.head(x[:, -1, :])

class BiLSTMModel(nn.Module):
    def __init__(self, n_feats, h1=96, h2=64, drop=0.2):
        super().__init__()
        self.bi1 = nn.LSTM(n_feats, h1, batch_first=True, bidirectional=True)
        self.do1 = nn.Dropout(drop)
        self.bi2 = nn.LSTM(2 * h1, h2, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(2 * h2, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        x, _ = self.bi1(x); x = self.do1(x)
        x, _ = self.bi2(x)
        return self.head(x[:, -1, :])

class TCNBlock(nn.Module):
    def __init__(self, ch, dil, drop=0.2):
        super().__init__()
        pad = dil
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=pad, dilation=dil), nn.ReLU(), nn.Dropout(drop),
            nn.Conv1d(ch, ch, 3, padding=pad, dilation=dil), nn.ReLU(), nn.Dropout(drop),
        )
    def forward(self, x): return x + self.net(x)

class TCN(nn.Module):
    def __init__(self, n_feats, channels=64, drop=0.2):
        super().__init__()
        self.proj = nn.Conv1d(n_feats, channels, 1)
        self.stack = nn.Sequential(*[TCNBlock(channels, d, drop) for d in [1, 2, 4, 8, 16, 32]])
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                   nn.Linear(channels, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = self.stack(x)
        return self.head(x)

class VanillaTransformer(nn.Module):
    """
    Vanilla encoder-only Transformer for short-horizon forecasting.
    Input projection -> learned positional encoding -> N encoder layers -> last-token head.
    """
    def __init__(self, n_feats, d_model=64, n_heads=4, n_layers=2, drop=0.2, max_len=20):
        super().__init__()
        # Ensure d_model is divisible by n_heads
        assert d_model % n_heads == 0, f"d_model={d_model} must be divisible by n_heads={n_heads}"
        self.input_proj = nn.Linear(n_feats, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=4 * d_model, dropout=drop,
            batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1)
        )
    def forward(self, x):  # [B, L, F]
        x = self.input_proj(x)
        x = x + self.pos_enc[:, :x.size(1), :]
        x = self.encoder(x)
        return self.head(x[:, -1, :])


def count_params(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Map scenario to feature count
SCENARIO_FEATS = {"A": 1, "B": 5, "C": 25}

print(f"{'Config':<40} {'Params':>12}")
print("-" * 54)

results = []
for config_id, info in sorted(best_hp.items()):
    arch = info["arch"]
    sc = info["scenario"]
    hp = info["best_params"]
    n_feats = SCENARIO_FEATS[sc]

    if arch == "LSTM":
        m = LSTMModel(n_feats, h1=hp["h1"], h2=hp["h2"], drop=hp["dropout"])
    elif arch == "BiLSTM":
        m = BiLSTMModel(n_feats, h1=hp["h1"], h2=hp["h2"], drop=hp["dropout"])
    elif arch == "TCN":
        m = TCN(n_feats, channels=hp["channels"], drop=hp["dropout"])
    elif arch == "Transformer":
        m = VanillaTransformer(
            n_feats,
            d_model=hp["d_model"],
            n_heads=hp["n_heads"],
            n_layers=hp["n_layers"],
            drop=hp["dropout"],
            max_len=20
        )
    else:
        continue

    n_params = count_params(m)
    results.append({"config": config_id, "arch": arch, "scenario": sc, "params": n_params})
    print(f"{config_id:<40} {n_params:>12,}")

# Save and summarize
import pandas as pd
df = pd.DataFrame(results)
df.to_csv("/content/drive/MyDrive/Close_res/optuna_results/parameter_counts.csv", index=False)

print("\n" + "=" * 54)
print("RANGE BY ARCHITECTURE:")
for arch in ["LSTM", "BiLSTM", "TCN", "Transformer"]:
    sub = df[df["arch"] == arch]
    if len(sub):
        print(f"  {arch:<12} min={sub['params'].min():>10,}  max={sub['params'].max():>10,}  mean={int(sub['params'].mean()):>10,}")
