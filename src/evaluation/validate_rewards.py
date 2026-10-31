"""
[Adaptive-QuantRL] Phase 4: Reward Ablation Validation & Model Freezing
Implements:
1. Deterministic evaluation of trained checkpoints from `models/checkpoints/` on Validation (2020~2022)
2. Per-seed trajectory & performance metrics computation (Sharpe, MDD, CAGR, Volatility, Turnover, Cost)
3. Strict 3-tier lexicographic selection criterion:
   - Tier 1: Higher Seed Mean Sharpe (`mean_sharpe` DESC)
   - Tier 2: Lower Seed Mean MDD (`mean_mdd` ASC)
   - Tier 3: Lower Inter-Seed Sharpe Std (`std_sharpe` ASC)
4. Freezes optimal single model & Reward A/B/C representative models for Phase 7 Test evaluation.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from baseline_models import compute_performance_metrics
from config import CHECKPOINTS_DIR, DEFAULT_CONFIG, PROJECT_ROOT, ExperimentConfig
from envs.trading_env import TradingEnv
from training.train_models import load_trained_model


PHASE4_RESULTS_DIR = PROJECT_ROOT / "results" / "phase4"


def evaluate_single_checkpoint_on_env(
    algorithm: str,
    reward_type: str,
    seed: int,
    mode: str = "val",
    one_way_cost_rate: float = DEFAULT_CONFIG.one_way_cost_rate,
    checkpoints_root: Path = CHECKPOINTS_DIR,
    config: ExperimentConfig = DEFAULT_CONFIG,
    preloaded_model_and_meta: Optional[Tuple[Any, Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Loads a single trained checkpoint (`{algorithm}_{reward}_{seed}`) and runs a
    deterministic rollout over the full Validation or Test split in `TradingEnv`.
    Returns `(metrics_dict, trajectory_df)`.
    """
    if preloaded_model_and_meta is not None:
        model, meta = preloaded_model_and_meta
    else:
        model, meta = load_trained_model(
            algorithm=algorithm,
            reward_type=reward_type,
            seed=seed,
            checkpoints_root=checkpoints_root,
        )

    env = TradingEnv(
        mode=mode,
        reward_type=reward_type,
        lambda_d=float(meta.get("configured_lambda_d", config.default_lambda_d)),
        lambda_v=float(meta.get("configured_lambda_v", config.default_lambda_v)),
        one_way_cost_rate=one_way_cost_rate,
        config=config,
    )

    obs, info0 = env.reset(seed=seed)
    records: List[Dict[str, Any]] = [
        {
            "Date": env.df.index[0],
            "Open": float(env.df["Open"].iloc[0]),
            "Close": float(env.df["Close"].iloc[0]),
            "adx_14_raw": float(env.df["adx_14_raw"].iloc[0])
            if "adx_14_raw" in env.df.columns
            else 0.0,
            "equity_close": float(info0["equity"]),
            "cash": float(info0["cash"]),
            "shares": float(info0["shares"]),
            "target_weight": 0.0,
            "close_weight": 0.0,
            "turnover_weight": 0.0,
            "transaction_cost": 0.0,
            "peak_equity": float(info0["peak_equity"]),
            "drawdown": float(info0["drawdown"]),
            "port_ret_simple": 0.0,
            "port_ret_log": 0.0,
            "raw_variance": 0.0,
        }
    ]

    done = False
    while not done:
        step_t = env.current_step
        adx_t = (
            float(env.df["adx_14_raw"].iloc[step_t])
            if "adx_14_raw" in env.df.columns
            else 0.0
        )
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated

        records.append(
            {
                "Date": env.df.index[env.current_step],
                "Open": float(env.df["Open"].iloc[env.current_step]),
                "Close": float(env.df["Close"].iloc[env.current_step]),
                "adx_14_raw": adx_t,  # ADX(14) at day t close that governed action_t
                "equity_close": float(step_info["equity"]),
                "cash": float(step_info["cash"]),
                "shares": float(step_info["shares"]),
                "target_weight": float(step_info["target_weight"]),
                "close_weight": float(step_info["realized_weight"]),
                "turnover_weight": float(step_info["turnover_weight"]),
                "transaction_cost": float(step_info["cost"]),
                "peak_equity": float(step_info["peak_equity"]),
                "drawdown": float(step_info["drawdown"]),
                "port_ret_simple": float(step_info["port_ret_simple"]),
                "port_ret_log": float(step_info["port_ret_log"]),
                "raw_variance": float(step_info["raw_variance"]),
            }
        )

    traj_df = pd.DataFrame(records).set_index("Date")
    perf = compute_performance_metrics(sim_df=traj_df, config=config)

    metrics_row = {
        "checkpoint_id": meta["checkpoint_id"],
        "track": meta.get("track", ""),
        "algorithm": algorithm.strip().upper(),
        "reward_type": reward_type.strip().upper(),
        "seed": int(seed),
        "lambda_d": float(meta.get("lambda_d", 0.0)),
        "lambda_v": float(meta.get("lambda_v", 0.0)),
        "one_way_cost_rate": float(one_way_cost_rate),
        **perf,
    }
    return metrics_row, traj_df


def rank_model_groups(agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the fixed 3-tier lexicographic ranking criterion:
      1st: Seed Mean Sharpe (`mean_sharpe` DESC)
      2nd: Seed Mean MDD (`mean_mdd` ASC)
      3rd: Inter-seed Sharpe Std (`std_sharpe` ASC)
    """
    ranked = agg_df.sort_values(
        by=["mean_sharpe", "mean_mdd", "std_sharpe"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def run_phase4_validation(
    seeds: Optional[Tuple[int, ...]] = None,
    algorithms: Tuple[str, ...] = ("ppo", "sac"),
    reward_types: Tuple[str, ...] = ("A", "B", "C"),
    checkpoints_root: Path = CHECKPOINTS_DIR,
    output_dir: Path = PHASE4_RESULTS_DIR,
    config: ExperimentConfig = DEFAULT_CONFIG,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Executes Phase 4 Validation (2020~2022) across all trained models in `checkpoints_root`:
    - Computes per-seed and seed-aggregated Validation metrics.
    - Ranks model groups by 1st: Mean Sharpe (DESC), 2nd: Mean MDD (ASC), 3rd: Std Sharpe (ASC).
    - Freezes representative models for Phase 7 Test evaluation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    eff_seeds = tuple(seeds if seeds is not None else config.random_seeds)

    per_seed_rows: List[Dict[str, Any]] = []
    for algo in algorithms:
        for rew in reward_types:
            for seed in eff_seeds:
                metrics_row, _ = evaluate_single_checkpoint_on_env(
                    algorithm=algo,
                    reward_type=rew,
                    seed=seed,
                    mode="val",
                    one_way_cost_rate=config.one_way_cost_rate,
                    checkpoints_root=checkpoints_root,
                    config=config,
                )
                per_seed_rows.append(metrics_row)

    per_seed_df = pd.DataFrame(per_seed_rows)

    # Aggregate across seeds for each (algorithm, reward_type)
    agg_rows: List[Dict[str, Any]] = []
    for (algo, rew), grp in per_seed_df.groupby(
        ["algorithm", "reward_type"], sort=False
    ):
        n_s = len(grp)
        sharpe_std = (
            float(grp["Sharpe_Ratio"].std(ddof=1)) if n_s > 1 else 0.0
        )
        mdd_std = float(grp["MDD(%)"].std(ddof=1)) if n_s > 1 else 0.0
        cagr_std = float(grp["CAGR(%)"].std(ddof=1)) if n_s > 1 else 0.0

        agg_rows.append(
            {
                "algorithm": algo,
                "reward_type": rew,
                "n_seeds": int(n_s),
                "lambda_d": float(grp["lambda_d"].iloc[0]),
                "lambda_v": float(grp["lambda_v"].iloc[0]),
                "mean_sharpe": round(float(grp["Sharpe_Ratio"].mean()), 4),
                "std_sharpe": round(sharpe_std, 4),
                "mean_mdd": round(float(grp["MDD(%)"].mean()), 2),
                "std_mdd": round(mdd_std, 2),
                "mean_cagr": round(float(grp["CAGR(%)"].mean()), 2),
                "std_cagr": round(cagr_std, 2),
                "mean_cum_ret": round(float(grp["Cumulative_Return(%)"].mean()), 2),
                "mean_vol": round(float(grp["Ann_Volatility(%)"].mean()), 2),
                "mean_weight": round(float(grp["Avg_Stock_Weight(%)"].mean()), 2),
                "mean_cost": round(float(grp["Total_Cost($)"].mean()), 2),
            }
        )

    agg_df = pd.DataFrame(agg_rows)
    ranked_df = rank_model_groups(agg_df)

    # Select best overall single model group, best per algorithm, and best per reward type
    best_overall = ranked_df.iloc[0].to_dict()

    best_by_algorithm: Dict[str, Any] = {}
    for algo in algorithms:
        algo_upper = algo.upper()
        sub = ranked_df[ranked_df["algorithm"] == algo_upper]
        if not sub.empty:
            best_by_algorithm[algo_upper] = sub.iloc[0].to_dict()

    best_by_reward: Dict[str, Any] = {}
    for rew in reward_types:
        rew_upper = rew.upper()
        sub = ranked_df[ranked_df["reward_type"] == rew_upper]
        if not sub.empty:
            best_by_reward[rew_upper] = sub.iloc[0].to_dict()

    # Freeze all Reward A/B/C checkpoints for both PPO and SAC so Test evaluation cannot modify them
    frozen_ablation_models: Dict[str, Dict[str, Any]] = {}
    for _, row in ranked_df.iterrows():
        key = f"{row['algorithm']}_{row['reward_type']}"
        frozen_ablation_models[key] = {
            "algorithm": row["algorithm"].lower(),
            "reward_type": row["reward_type"],
            "seeds": list(eff_seeds),
            "lambda_d": float(row["lambda_d"]),
            "lambda_v": float(row["lambda_v"]),
            "val_mean_sharpe": float(row["mean_sharpe"]),
            "val_mean_mdd": float(row["mean_mdd"]),
            "val_std_sharpe": float(row["std_sharpe"]),
            "val_rank": int(row["rank"]),
            "checkpoint_ids": [
                f"{row['algorithm'].lower()}_{row['reward_type']}_{s}"
                for s in eff_seeds
            ],
        }

    frozen_payload = {
        "selection_criteria": [
            "1_mean_sharpe_desc",
            "2_mean_mdd_asc",
            "3_std_sharpe_asc",
        ],
        "validation_period": f"{config.val_start} ~ {config.val_end}",
        "seeds": list(eff_seeds),
        "best_overall_single_group": best_overall,
        "best_by_algorithm": best_by_algorithm,
        "best_by_reward_type": best_by_reward,
        "frozen_ablation_groups": frozen_ablation_models,
    }

    per_seed_df.to_csv(output_dir / "phase4_val_per_seed_metrics.csv", index=False)
    ranked_df.to_csv(output_dir / "phase4_val_aggregated_ranking.csv", index=False)
    with open(output_dir / "phase4_frozen_models.json", "w", encoding="utf-8") as f:
        json.dump(frozen_payload, f, indent=2, ensure_ascii=False)

    if verbose:
        print("\n[Phase 4] Validation (2020~2022) Aggregated Model Ranking:")
        print(ranked_df.to_string(index=False))
        print(
            f"\n  -> Best Overall Single Model Group: "
            f"{best_overall['algorithm']} (Reward {best_overall['reward_type']}) | "
            f"Mean Sharpe={best_overall['mean_sharpe']:.4f}, "
            f"Mean MDD={best_overall['mean_mdd']:.2f}%, "
            f"Std Sharpe={best_overall['std_sharpe']:.4f}"
        )

    return {
        "per_seed_df": per_seed_df,
        "ranked_df": ranked_df,
        "frozen_payload": frozen_payload,
    }


if __name__ == "__main__":
    run_phase4_validation(verbose=True)
