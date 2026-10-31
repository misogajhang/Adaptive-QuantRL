"""
[Adaptive-QuantRL] Phase 3: Pre-Training Reward Scale Measurement Module
Executes Buy & Hold, Fixed 50:50, and Random Uniform policies in the Train TradingEnv
to measure empirical distributions (Mean, Std, Min, 25%, Median, 75%, Max) of:
  1. Log Return: R_{A,t} = ln(Equity_{t+1}^{close} / Equity_t^{close}) and |R_{A,t}|
  2. Drawdown: Drawdown_t in [0, 1]
  3. Raw 30-day Portfolio Variance: Variance^{raw}_{30,t} >= 0
Derives and validates lambda_d and lambda_v candidates so risk penalties do not
overwhelm the log-return signal.
"""

import json
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from config import DEFAULT_CONFIG, PHASE3_RESULTS_DIR, ExperimentConfig
from envs.trading_env import TradingEnv


def _compute_quantiles(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "Mean": 0.0,
            "Std": 0.0,
            "Min": 0.0,
            "25%": 0.0,
            "Median": 0.0,
            "75%": 0.0,
            "Max": 0.0,
        }
    q25, q50, q75 = np.percentile(arr, [25.0, 50.0, 75.0])
    return {
        "Mean": float(np.mean(arr)),
        "Std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "Min": float(np.min(arr)),
        "25%": float(q25),
        "Median": float(q50),
        "75%": float(q75),
        "Max": float(np.max(arr)),
    }


def _collect_policy_rollouts(
    policy_name: str,
    action_fn: Callable[[np.random.Generator], float],
    num_episodes: int = 10,
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> Dict[str, np.ndarray]:
    """
    Runs `num_episodes` of 252-day Train episodes across controlled seeds and
    collects step-level log_return, abs_log_return, drawdown, and raw_variance.
    """
    env = TradingEnv(mode="train", reward_type="A", config=config)

    log_returns: List[float] = []
    drawdowns: List[float] = []
    raw_variances_all: List[float] = []
    raw_variances_active: List[float] = []

    for ep_idx in range(num_episodes):
        ep_seed = int(config.random_seeds[ep_idx % len(config.random_seeds)] + ep_idx * 100)
        rng = np.random.default_rng(ep_seed)
        env.reset(seed=ep_seed)
        done = False
        step_idx = 0
        while not done:
            w = float(action_fn(rng))
            _, _, terminated, truncated, info = env.step(
                np.array([w], dtype=np.float32)
            )
            step_idx += 1
            log_returns.append(float(info["reward_a"]))
            drawdowns.append(float(info["drawdown"]))
            raw_variances_all.append(float(info["raw_variance"]))
            if step_idx >= config.variance_window:
                raw_variances_active.append(float(info["raw_variance"]))
            done = terminated or truncated

    log_ret_arr = np.array(log_returns, dtype=np.float64)
    return {
        "policy": policy_name,
        "log_return": log_ret_arr,
        "abs_log_return": np.abs(log_ret_arr),
        "drawdown": np.array(drawdowns, dtype=np.float64),
        "raw_variance_all": np.array(raw_variances_all, dtype=np.float64),
        "raw_variance_active": np.array(raw_variances_active, dtype=np.float64),
    }


def measure_reward_scales(
    num_episodes_per_policy: int = 10,
    config: ExperimentConfig = DEFAULT_CONFIG,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes Buy & Hold, Fixed 50:50, and Random Policy in the Train environment,
    computes quantile tables for log_return, |log_return|, drawdown, and raw_variance,
    and evaluates candidate penalty scales (lambda_d, lambda_v).
    """
    PHASE3_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    policies: List[Tuple[str, Callable[[np.random.Generator], float]]] = [
        ("Buy_and_Hold", lambda rng: 1.0),
        ("Fixed_50_50", lambda rng: 0.5),
        ("Random_Policy", lambda rng: float(rng.uniform(0.0, 1.0))),
    ]

    rows: List[Dict[str, Any]] = []
    pooled_abs_log_ret: List[np.ndarray] = []
    pooled_dd: List[np.ndarray] = []
    pooled_var_active: List[np.ndarray] = []

    for policy_name, action_fn in policies:
        rollout = _collect_policy_rollouts(
            policy_name=policy_name,
            action_fn=action_fn,
            num_episodes=num_episodes_per_policy,
            config=config,
        )
        pooled_abs_log_ret.append(rollout["abs_log_return"])
        pooled_dd.append(rollout["drawdown"])
        pooled_var_active.append(rollout["raw_variance_active"])

        metrics_map = [
            ("log_return (R_A)", rollout["log_return"]),
            ("|log_return| (|R_A|)", rollout["abs_log_return"]),
            ("drawdown (DD_t)", rollout["drawdown"]),
            ("raw_variance_all (Var_30)", rollout["raw_variance_all"]),
            ("raw_variance_active (t>=30)", rollout["raw_variance_active"]),
        ]
        for metric_label, arr in metrics_map:
            q = _compute_quantiles(arr)
            rows.append({"Policy": policy_name, "Metric": metric_label, **q})

    summary_df = pd.DataFrame(rows)

    # Pooled statistics across all 3 reference policies for scale calibration
    all_abs_ret = np.concatenate(pooled_abs_log_ret)
    all_dd = np.concatenate(pooled_dd)
    all_var = np.concatenate(pooled_var_active)

    mean_abs_ret = float(np.mean(all_abs_ret))
    median_abs_ret = float(np.median(all_abs_ret))
    mean_dd = float(np.mean(all_dd))
    median_dd = float(np.median(all_dd))
    mean_var = float(np.mean(all_var))
    median_var = float(np.median(all_var))

    # Evaluate penalty-to-signal ratios for config candidates
    lambda_d_analysis = []
    for ld in config.lambda_d_candidates:
        mean_pen = float(ld * mean_dd)
        med_pen = float(ld * median_dd)
        lambda_d_analysis.append(
            {
                "lambda_d": float(ld),
                "mean_penalty": mean_pen,
                "median_penalty": med_pen,
                "ratio_to_mean_abs_log_ret(%)": round((mean_pen / mean_abs_ret) * 100.0, 2),
                "ratio_to_median_abs_log_ret(%)": round((med_pen / median_abs_ret) * 100.0, 2),
            }
        )

    lambda_v_analysis = []
    for lv in config.lambda_v_candidates:
        mean_pen = float(lv * mean_var)
        med_pen = float(lv * median_var)
        lambda_v_analysis.append(
            {
                "lambda_v": float(lv),
                "mean_penalty": mean_pen,
                "median_penalty": med_pen,
                "ratio_to_mean_abs_log_ret(%)": round((mean_pen / mean_abs_ret) * 100.0, 2),
                "ratio_to_median_abs_log_ret(%)": round((med_pen / median_abs_ret) * 100.0, 2),
            }
        )

    report = {
        "num_episodes_per_policy": int(num_episodes_per_policy),
        "steps_per_episode": int(config.train_episode_length),
        "pooled_reference_scales": {
            "mean_abs_log_return": mean_abs_ret,
            "median_abs_log_return": median_abs_ret,
            "mean_drawdown": mean_dd,
            "median_drawdown": median_dd,
            "mean_raw_variance_active": mean_var,
            "median_raw_variance_active": median_var,
        },
        "lambda_d_candidates": list(config.lambda_d_candidates),
        "lambda_v_candidates": list(config.lambda_v_candidates),
        "selected_default_lambda_d": float(config.default_lambda_d),
        "selected_default_lambda_v": float(config.default_lambda_v),
        "lambda_d_scale_ratios": lambda_d_analysis,
        "lambda_v_scale_ratios": lambda_v_analysis,
    }

    summary_csv_path = PHASE3_RESULTS_DIR / "reward_scale_quantiles.csv"
    report_json_path = PHASE3_RESULTS_DIR / "reward_scale_report.json"
    summary_df.to_csv(summary_csv_path, index=False)
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if verbose:
        print("\n[Phase 3 - Step 1] Reward Component Scale & Quantile Distribution (Train Env):")
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 150)
        pd.set_option("display.float_format", lambda x: f"{x:.6f}")
        print(summary_df.to_string(index=False))
        pd.reset_option("display.float_format")

        print("\n[Calibrated Penalty Candidate Ratios vs |Log Return|]:")
        print(
            f"  - Pooled Mean |R_A| = {mean_abs_ret:.6f}, Median |R_A| = {median_abs_ret:.6f}"
        )
        print(
            f"  - Pooled Mean DD    = {mean_dd:.6f}, Mean Var_30 (active) = {mean_var:.8f}"
        )
        print("  - lambda_d candidates:", pd.DataFrame(lambda_d_analysis).to_dict(orient="records"))
        print("  - lambda_v candidates:", pd.DataFrame(lambda_v_analysis).to_dict(orient="records"))
        print(
            f"  -> Representative Phase 3 Defaults: lambda_d = {config.default_lambda_d}, "
            f"lambda_v = {config.default_lambda_v}"
        )

    return summary_df, report


if __name__ == "__main__":
    measure_reward_scales(verbose=True)
