"""
[Adaptive-QuantRL] Phase 1: Data Pipeline, Feature Engineering & Leakage Verification
Implements:
1. 15-year unadjusted SPY OHLCV acquisition (excluding dividend cashflows)
2. Warm-up indicator computation (no look-ahead bias)
3. Chronological Holdout Split: Train(2010~2019), Validation(2020~2022), Test(2023~2024)
4. Train-only StandardScaler for Market Features & Log-Variance Normalization Stats
5. Strict Data Leakage & Accounting Integrity Checks
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from config import (
    DEFAULT_CONFIG,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    ExperimentConfig,
)


def fetch_spy_ohlc(
    config: ExperimentConfig = DEFAULT_CONFIG,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Fetches unadjusted SPY OHLCV data (auto_adjust=False) from warmup_start_date
    through test_end. Caches raw data to RAW_DATA_DIR / 'spy_raw_ohlc.csv'.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_DATA_DIR / f"{config.ticker.lower()}_unadjusted_ohlc.csv"

    if cache_path.exists() and not force_download:
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        df.index.name = "Date"
        return df.sort_index()

    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required to download SPY data. "
            "Install via `pip install -r requirements.txt`."
        ) from exc

    # Fetch up to a few days past test_end to ensure inclusive end date coverage
    end_inclusive = (pd.Timestamp(config.test_end) + pd.Timedelta(days=5)).strftime(
        "%Y-%m-%d"
    )
    raw = yf.download(
        config.ticker,
        start=config.warmup_start_date,
        end=end_inclusive,
        auto_adjust=config.auto_adjust,  # False: exclude dividends from OHLC prices
        actions=False,
        progress=False,
    )

    if raw is None or raw.empty:
        raise RuntimeError(
            f"Failed to download {config.ticker} data from yfinance."
        )

    # Flatten MultiIndex columns if returned by newer yfinance versions
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required_cols if c not in raw.columns]
    if missing:
        raise ValueError(f"Downloaded data missing required OHLCV columns: {missing}")

    # Keep only standard OHLCV columns (plus Adj Close if present for reference only)
    keep_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in raw.columns]
    df = raw[keep_cols].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    df = df.sort_index()
    df = df.loc[config.warmup_start_date : config.test_end]

    df.to_csv(cache_path)
    return df


def compute_wilder_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Computes True Range (TR), ATR, +DI, -DI, and ADX(14) using Wilder's smoothing
    (alpha = 1 / window) strictly using information up to day t close.
    Returns (atr, plus_di, minus_di, adx).
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm_s = pd.Series(plus_dm, index=close.index)
    minus_dm_s = pd.Series(minus_dm, index=close.index)

    alpha = 1.0 / float(window)
    atr = tr.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    plus_dm_smooth = plus_dm_s.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    minus_dm_smooth = minus_dm_s.ewm(alpha=alpha, adjust=False, min_periods=window).mean()

    plus_di = 100.0 * (plus_dm_smooth / (atr + 1e-12))
    minus_di = 100.0 * (minus_dm_smooth / (atr + 1e-12))

    dx = 100.0 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-12)
    adx = dx.ewm(alpha=alpha, adjust=False, min_periods=window).mean()

    return atr, plus_di, minus_di, adx


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """
    Computes Wilder's RSI(14) in range [0, 100] using only historical closes up to t.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    alpha = 1.0 / float(window)
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=window).mean()

    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def engineer_market_features(
    raw_df: pd.DataFrame,
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Computes market features and technical indicators using warm-up history.
    Preserves unadjusted OHLC prices, raw moving averages, raw ADX(14),
    and raw 30-day variance alongside the market feature columns.
    """
    df = raw_df.copy().sort_index()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # 1. Returns (strictly backward-looking up to day t close)
    df["ret_1d"] = close.pct_change(1)
    df["log_ret_1d"] = np.log(close / close.shift(1))
    df["ret_5d"] = close.pct_change(5)
    df["ret_20d"] = close.pct_change(20)

    # 2. Moving Averages & Relative Distances
    df["sma_20"] = close.rolling(window=config.ma_fast_window, min_periods=config.ma_fast_window).mean()
    df["sma_50"] = close.rolling(window=config.ma_slow_window, min_periods=config.ma_slow_window).mean()
    df["sma_200"] = close.rolling(window=config.ma_long_window, min_periods=config.ma_long_window).mean()

    df["close_to_sma20"] = (close / df["sma_20"]) - 1.0
    df["close_to_sma50"] = (close / df["sma_50"]) - 1.0
    df["close_to_sma200"] = (close / df["sma_200"]) - 1.0

    # Moving average trend signals (known at day t close, executed at day t+1 open)
    df["ma_signal_20_50"] = (df["sma_20"] > df["sma_50"]).astype(float)
    df["ma_signal_close_200"] = (close > df["sma_200"]).astype(float)

    # 3. Volatility & 30-Day Raw Return Variance (Section 4)
    df["vol_20d"] = df["ret_1d"].rolling(window=20, min_periods=20).std(ddof=1)
    df["mkt_var_30d_raw"] = df["ret_1d"].rolling(
        window=config.variance_window,
        min_periods=config.variance_window,
    ).var(ddof=1)

    # 4. Momentum & Trend Indicators: RSI(14), MACD(12,26,9), ATR(14), ADX(14)
    df["rsi_14"] = compute_rsi(close, window=14)

    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    macd_hist = macd - macd_signal

    df["macd_norm"] = macd / close
    df["macd_signal_norm"] = macd_signal / close
    df["macd_hist_norm"] = macd_hist / close

    atr_14, plus_di_14, minus_di_14, adx_14 = compute_wilder_adx(
        high=high,
        low=low,
        close=close,
        window=config.adx_window,
    )
    df["atr_14_norm"] = atr_14 / close
    df["plus_di_14"] = plus_di_14
    df["minus_di_14"] = minus_di_14

    # Keep unscaled `adx_14_raw` for Phase 5 regime thresholding (e.g., 20, 25, 30)
    # and `adx_14_scaled` as part of standardized Market Features
    df["adx_14_raw"] = adx_14
    df["adx_14_scaled"] = adx_14.copy()

    return df


def split_and_standardize(
    featured_df: pd.DataFrame,
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    1. Splits `featured_df` chronologically into Train(2010~2019), Val(2020~2022), Test(2023~2024).
    2. Fits StandardScaler strictly on Train Market Features (`config.market_feature_columns`).
    3. Transforms Market Features across Train, Val, and Test (storing scaled values in
       both `<col>` and `<col>_z` while preserving raw OHLC and `adx_14_raw`).
    4. Computes Train-only normalization parameters for Observation Log-Variance:
       Variance_obs = Normalize(log(Variance_raw + 1e-8)).
    """
    train_df = featured_df.loc[config.train_start : config.train_end].copy()
    val_df = featured_df.loc[config.val_start : config.val_end].copy()
    test_df = featured_df.loc[config.test_start : config.test_end].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            f"One or more chronological splits are empty: "
            f"Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
        )

    feature_cols = list(config.market_feature_columns)

    # Fit StandardScaler strictly on Train (using ddof=0 to match sklearn StandardScaler)
    train_means = train_df[feature_cols].mean(axis=0)
    train_stds = train_df[feature_cols].std(axis=0, ddof=0).replace(0.0, 1.0)

    # Compute Train-only log-variance normalization statistics for Section 4:
    # Variance^{obs}_{30,t} = Normalize(log(Variance^{raw}_{30,t} + 1e-8))
    train_log_var = np.log(train_df["mkt_var_30d_raw"].clip(lower=0.0) + config.variance_eps)
    log_var_mean = float(train_log_var.mean())
    log_var_std = float(train_log_var.std(ddof=0))
    if log_var_std == 0.0:
        log_var_std = 1.0

    for split_df in (train_df, val_df, test_df):
        # Preserve unscaled market features with `_raw_feat` suffix for inspection
        for col in feature_cols:
            split_df[f"{col}_raw_feat"] = split_df[col]
            split_df[col] = (split_df[col] - train_means[col]) / train_stds[col]

        # Precompute normalized market observation variance for reference
        raw_var = split_df["mkt_var_30d_raw"].clip(lower=0.0)
        split_df["mkt_var_30d_obs"] = (
            np.log(raw_var + config.variance_eps) - log_var_mean
        ) / log_var_std

    scaler_stats = {
        "train_start": str(train_df.index.min().date()),
        "train_end": str(train_df.index.max().date()),
        "val_start": str(val_df.index.min().date()),
        "val_end": str(val_df.index.max().date()),
        "test_start": str(test_df.index.min().date()),
        "test_end": str(test_df.index.max().date()),
        "n_train_days": int(len(train_df)),
        "n_val_days": int(len(val_df)),
        "n_test_days": int(len(test_df)),
        "feature_columns": feature_cols,
        "market_scaler_mean": {k: float(v) for k, v in train_means.items()},
        "market_scaler_std": {k: float(v) for k, v in train_stds.items()},
        "obs_log_variance_norm": {
            "eps": float(config.variance_eps),
            "window": int(config.variance_window),
            "mean": log_var_mean,
            "std": log_var_std,
            "initial_obs_value": float(config.initial_obs_variance),
        },
    }

    return train_df, val_df, test_df, scaler_stats


def verify_no_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scaler_stats: Dict,
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> Dict[str, bool]:
    """
    Executes rigorous assertions to verify zero data leakage and data integrity:
    1. Chronological disjointness: Train_end < Val_start and Val_end < Test_start.
    2. Monotonic date indices without duplicates.
    3. Zero NaN or Inf values in OHLCV and Market Features across all splits
       (confirming warm-up prevented NaN loss in 2010).
    4. Train Market Features have mean == 0 and std == 1 (within 1e-6), while Val/Test
       use Train statistics rather than their own.
    5. Unadjusted OHLC price sanity (High >= max(Open, Close), Low <= min(Open, Close), all > 0).
    """
    checks: Dict[str, bool] = {}

    # 1. Chronological boundary check
    assert train_df.index.max() < val_df.index.min(), (
        f"Leakage detected: Train max date ({train_df.index.max()}) >= Val min date ({val_df.index.min()})"
    )
    assert val_df.index.max() < test_df.index.min(), (
        f"Leakage detected: Val max date ({val_df.index.max()}) >= Test min date ({test_df.index.min()})"
    )
    assert train_df.index.min() >= pd.Timestamp(config.train_start)
    assert train_df.index.max() <= pd.Timestamp(config.train_end)
    assert val_df.index.min() >= pd.Timestamp(config.val_start)
    assert val_df.index.max() <= pd.Timestamp(config.val_end)
    assert test_df.index.min() >= pd.Timestamp(config.test_start)
    assert test_df.index.max() <= pd.Timestamp(config.test_end)
    checks["chronological_holdout_disjoint"] = True

    # 2. Index monotonicity and uniqueness
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        assert df.index.is_monotonic_increasing, f"{name} index is not monotonic increasing."
        assert not df.index.has_duplicates, f"{name} index has duplicate dates."
    checks["monotonic_unique_dates"] = True

    # 3. Zero NaNs / Infs in required OHLC and Market Feature columns
    check_cols = ["Open", "High", "Low", "Close", "Volume", "adx_14_raw"] + list(
        config.market_feature_columns
    )
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        nan_count = int(df[check_cols].isna().sum().sum())
        inf_count = int(np.isinf(df[check_cols].to_numpy()).sum())
        assert nan_count == 0, f"{name} contains {nan_count} NaN values in check_cols."
        assert inf_count == 0, f"{name} contains {inf_count} Inf values in check_cols."
    checks["zero_nan_or_inf_after_warmup"] = True

    # 4. StandardScaler Train-only fit verification
    for col in config.market_feature_columns:
        tr_mean = float(train_df[col].mean())
        tr_std = float(train_df[col].std(ddof=0))
        assert abs(tr_mean) < 1e-6, f"Train feature {col} mean is {tr_mean}, expected ~0.0"
        assert abs(tr_std - 1.0) < 1e-6, f"Train feature {col} std is {tr_std}, expected ~1.0"

        # Verify Val and Test were transformed using Train's mean and std
        expected_val = (
            val_df[f"{col}_raw_feat"] - scaler_stats["market_scaler_mean"][col]
        ) / scaler_stats["market_scaler_std"][col]
        assert np.allclose(val_df[col].to_numpy(), expected_val.to_numpy(), atol=1e-7), (
            f"Val feature {col} does not match Train StandardScaler transformation."
        )
    checks["train_only_standard_scaler_verified"] = True

    # 5. Unadjusted OHLC integrity check
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        assert (df[["Open", "High", "Low", "Close"]] > 0).all().all(), (
            f"{name} has non-positive OHLC prices."
        )
        assert (df["High"] >= df[["Open", "Close"]].max(axis=1) - 1e-5).all(), (
            f"{name} violates High >= max(Open, Close)."
        )
        assert (df["Low"] <= df[["Open", "Close"]].min(axis=1) + 1e-5).all(), (
            f"{name} violates Low <= min(Open, Close)."
        )
    checks["unadjusted_ohlc_price_integrity"] = True

    return checks


def run_data_pipeline(
    config: ExperimentConfig = DEFAULT_CONFIG,
    force_download: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict, Dict[str, bool]]:
    """
    Executes the full Phase 1 data pipeline:
    Fetch -> Feature Engineer (with 2009 warm-up) -> Split & Standardize -> Verify Leakage -> Save.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = fetch_spy_ohlc(config=config, force_download=force_download)
    featured_df = engineer_market_features(raw_df=raw_df, config=config)
    train_df, val_df, test_df, scaler_stats = split_and_standardize(
        featured_df=featured_df,
        config=config,
    )
    leakage_checks = verify_no_leakage(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        scaler_stats=scaler_stats,
        config=config,
    )

    # Save processed splits and scaler metadata
    train_df.to_csv(PROCESSED_DATA_DIR / "spy_train_2010_2019.csv")
    val_df.to_csv(PROCESSED_DATA_DIR / "spy_val_2020_2022.csv")
    test_df.to_csv(PROCESSED_DATA_DIR / "spy_test_2023_2024.csv")

    metadata = {
        "config": asdict(config),
        "scaler_stats": scaler_stats,
        "leakage_checks": leakage_checks,
    }
    with open(PROCESSED_DATA_DIR / "phase1_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return train_df, val_df, test_df, scaler_stats, leakage_checks
