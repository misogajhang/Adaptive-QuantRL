"""
[Adaptive-QuantRL] Phase 1 Runner: Data Pipeline & 4 Baselines Pre-computation
Executes:
1. Unadjusted SPY 15-Year OHLC acquisition (with 2009 warm-up)
2. Chronological Holdout Split (Train: 2010~2019, Val: 2020~2022, Test: 2023~2024)
3. Train-only StandardScaler & Data Leakage Verification
4. Pre-computation of the 4 Baseline Strategies (B&H, Cash 100%, Fixed 50:50, Moving Average)
   with 0.1% one-way transaction cost and dividend cashflows excluded.
"""

import argparse
from pathlib import Path

import pandas as pd

from baseline_models import run_all_baselines
from config import DEFAULT_CONFIG, RESULTS_DIR
from data_pipeline import run_data_pipeline


def main(force_download: bool = False) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("[Adaptive-QuantRL] Phase 1: Data Pipeline & Baseline Construction")
    print("=" * 80)

    # 1. Run Data Pipeline & Leakage Verification
    train_df, val_df, test_df, scaler_stats, leakage_checks = run_data_pipeline(
        config=DEFAULT_CONFIG,
        force_download=force_download,
    )

    print("\n[1] Chronological Holdout Split Summary (Unadjusted SPY OHLC):")
    print(
        f"  - Train      : {scaler_stats['train_start']} ~ {scaler_stats['train_end']} "
        f"({scaler_stats['n_train_days']} trading days)"
    )
    print(
        f"  - Validation : {scaler_stats['val_start']} ~ {scaler_stats['val_end']} "
        f"({scaler_stats['n_val_days']} trading days)"
    )
    print(
        f"  - Test       : {scaler_stats['test_start']} ~ {scaler_stats['test_end']} "
        f"({scaler_stats['n_test_days']} trading days)"
    )

    print("\n[2] Data Leakage & Integrity Verification:")
    for check_name, passed in leakage_checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  - {check_name:40s}: [{status}]")

    # 2. Evaluate 4 Baselines across Train, Validation, and Test
    obs_log_var_stats = scaler_stats["obs_log_variance_norm"]
    all_summaries = []

    splits = [
        ("Train (2010-2019)", "train", train_df),
        ("Validation (2020-2022)", "val", val_df),
        ("Test (2023-2024)", "test", test_df),
    ]

    for split_label, split_key, split_df in splits:
        summary_df, trajectories = run_all_baselines(
            split_df=split_df,
            split_name=split_label,
            config=DEFAULT_CONFIG,
            obs_log_var_stats=obs_log_var_stats,
        )
        all_summaries.append(summary_df)

        # Save daily equity curves for each split
        combined_eq = pd.DataFrame(
            {
                strat: traj["equity_close"]
                for strat, traj in trajectories.items()
            }
        )
        combined_eq.to_csv(RESULTS_DIR / f"baseline_equity_curves_{split_key}.csv")

    full_summary_df = pd.concat(all_summaries, ignore_index=True)
    summary_csv_path = RESULTS_DIR / "phase1_baseline_metrics.csv"
    full_summary_df.to_csv(summary_csv_path, index=False)

    print("\n[3] 4 Baseline Strategies Performance (One-way Cost = 0.1%, Dividends Excluded):")
    display_cols = [
        "Split",
        "Strategy",
        "Final_Equity($)",
        "Cumulative_Return(%)",
        "CAGR(%)",
        "Ann_Volatility(%)",
        "Sharpe_Ratio",
        "MDD(%)",
        "Avg_Stock_Weight(%)",
        "Total_Cost($)",
    ]
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    print(full_summary_df[display_cols].to_string(index=False))

    print("\n[4] Artifacts Saved:")
    print(f"  - Processed Datasets & Metadata : data/processed/")
    print(f"  - Baseline Metrics Table        : {summary_csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Adaptive-QuantRL Phase 1")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of SPY OHLC data from yfinance",
    )
    args = parser.parse_args()
    main(force_download=args.force_download)
