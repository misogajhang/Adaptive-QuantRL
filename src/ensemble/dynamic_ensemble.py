"""
[Adaptive-QuantRL] Phase 5 & Phase 6: ADX Regime Analysis & Shared-Env Dynamic Ensemble
Implements:
1. Phase 5: ADX(14) pre-defined threshold candidates (20, 25, 30) regime classification
   at day t close, recording regime sample counts (N), PPO/SAC mean weights, and action correlations.
2. Phase 6: Direct Validation backtest of all 27 dynamic ensemble rules
   (3 ADX thresholds x 3 strong-trend weights [0.25, 0.50, 0.75] x 3 weak-trend weights [0.25, 0.50, 0.75])
   inside the shared `TradingEnv`.
3. Selects & freezes the single best ensemble rule using the fixed 3-tier criterion:
   1st: Seed Mean Sharpe (DESC), 2nd: Seed Mean MDD (ASC), 3rd: Inter-seed Sharpe Std (ASC).
4. Extracts and freezes each seed's Validation average stock weight as the
   `Exposure-Matched Baseline` for Phase 7 Test evaluation.
"""

import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from baseline_models import compute_performance_metrics
from config import CHECKPOINTS_DIR, DEFAULT_CONFIG, PROJECT_ROOT, ExperimentConfig
from envs.trading_env import TradingEnv
from training.train_models import load_trained_model


PHASE5_6_RESULTS_DIR = PROJECT_ROOT / "results" / "phase5_6"
WEIGHT_CANDIDATES: Tuple[float, ...] = (0.25, 0.50, 0.75)


def classify_adx_regime(adx_value: float, threshold: float) -> str:
    """Classifies day t close ADX(14) into 'Strong_Trend' (>= threshold) or 'Weak_Trend' (< threshold)."""
    return "Strong_Trend" if float(adx_value) >= float(threshold) else "Weak_Trend"


def compute_regime_sample_counts(
    df: pd.DataFrame,
    thresholds: Tuple[float, ...] = DEFAULT_CONFIG.adx_threshold_candidates,
) -> pd.DataFrame:
    """
    Computes trading-day sample counts (N) and proportions (%) for Strong_Trend vs Weak_Trend
    across candidate ADX(14) thresholds on a given split DataFrame.
     Note: Step transitions use t = 0 .. len(df)-2 (the N-1 decision days).
    """
    decision_adx = df["adx_14_raw"].iloc[:-1].to_numpy(dtype=np.float64)
    total_days = len(decision_adx)
    rows: List[Dict[str, Any]] = []

    for th in thresholds:
        strong_mask = decision_adx >= float(th)
        n_strong = int(np.sum(strong_mask))
        n_weak = int(total_days - n_strong)
        rows.append(
            {
                "adx_threshold": float(th),
                "regime": "Strong_Trend",
                "n_days": n_strong,
                "proportion(%)": round((n_strong / total_days) * 100.0, 2)
                if total_days > 0
                else 0.0,
                "sufficient_sample(N>=30)": bool(n_strong >= 30),
            }
        )
        rows.append(
            {
                "adx_threshold": float(th),
                "regime": "Weak_Trend",
                "n_days": n_weak,
                "proportion(%)": round((n_weak / total_days) * 100.0, 2)
                if total_days > 0
                else 0.0,
                "sufficient_sample(N>=30)": bool(n_weak >= 30),
            }
        )

    return pd.DataFrame(rows)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    c = float(np.corrcoef(x, y)[0, 1])
    return 0.0 if np.isnan(c) else c


def _fast_predict_weight(model: Any, obs: np.ndarray) -> float:
    """
    Fast deterministic scalar action prediction in [0.0, 1.0] without SB3's per-step
    observation space validation overhead. Matches `model.predict(obs, deterministic=True)`.
    """
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=model.device).unsqueeze(0)
        raw_act = model.policy._predict(obs_t, deterministic=True).cpu().numpy().reshape(-1)[0]
    if getattr(model.policy, "squash_output", False):
        # SAC squashes output to [-1, 1]; unscale to [0, 1]
        val = 0.5 * (float(raw_act) + 1.0)
    else:
        val = float(raw_act)
    return float(np.clip(val, 0.0, 1.0))


def run_phase5_regime_behavior_analysis(
    seeds: Optional[Tuple[int, ...]] = None,
    reward_type: str = "A",
    thresholds: Tuple[float, ...] = DEFAULT_CONFIG.adx_threshold_candidates,
    checkpoints_root: Path = CHECKPOINTS_DIR,
    output_dir: Path = PHASE5_6_RESULTS_DIR,
    config: ExperimentConfig = DEFAULT_CONFIG,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Phase 5: Evaluates PPO and SAC models on Validation (2020~2022) and analyzes
    regime-wise observation counts (N), average stock weights, and PPO/SAC action
    correlations for each pre-defined ADX(14) threshold (20, 25, 30).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    eff_seeds = tuple(seeds if seeds is not None else config.random_seeds)

    # Collect daily decision actions from independent PPO and SAC Validation rollouts per seed
    seed_action_records: List[pd.DataFrame] = []

    for seed in eff_seeds:
        ppo_model, _ = load_trained_model(
            algorithm="ppo",
            reward_type=reward_type,
            seed=seed,
            checkpoints_root=checkpoints_root,
        )
        sac_model, _ = load_trained_model(
            algorithm="sac",
            reward_type=reward_type,
            seed=seed,
            checkpoints_root=checkpoints_root,
        )

        # Rollout PPO
        env_ppo = TradingEnv(mode="val", reward_type=reward_type, config=config)
        obs_p, _ = env_ppo.reset(seed=seed)
        ppo_actions: List[float] = []
        adx_vals: List[float] = []
        dates: List[Any] = []
        done = False
        while not done:
            t_idx = env_ppo.current_step
            adx_vals.append(float(env_ppo.df["adx_14_raw"].iloc[t_idx]))
            dates.append(env_ppo.df.index[t_idx])
            w_p = _fast_predict_weight(ppo_model, obs_p)
            ppo_actions.append(w_p)
            obs_p, _, term, trunc, _ = env_ppo.step(np.array([w_p], dtype=np.float32))
            done = term or trunc

        # Rollout SAC
        env_sac = TradingEnv(mode="val", reward_type=reward_type, config=config)
        obs_s, _ = env_sac.reset(seed=seed)
        sac_actions: List[float] = []
        done = False
        while not done:
            w_s = _fast_predict_weight(sac_model, obs_s)
            sac_actions.append(w_s)
            obs_s, _, term, trunc, _ = env_sac.step(np.array([w_s], dtype=np.float32))
            done = term or trunc

        seed_action_records.append(
            pd.DataFrame(
                {
                    "Date": dates,
                    "seed": int(seed),
                    "adx_14_raw": adx_vals,
                    "ppo_weight": ppo_actions,
                    "sac_weight": sac_actions,
                }
            )
        )

    all_actions_df = pd.concat(seed_action_records, ignore_index=True)

    rows: List[Dict[str, Any]] = []
    for th in thresholds:
        for regime_label in ("Strong_Trend", "Weak_Trend", "All"):
            if regime_label == "Strong_Trend":
                sub = all_actions_df[all_actions_df["adx_14_raw"] >= float(th)]
            elif regime_label == "Weak_Trend":
                sub = all_actions_df[all_actions_df["adx_14_raw"] < float(th)]
            else:
                sub = all_actions_df

            # Compute per-seed stats then average across seeds
            n_days_per_seed = int(len(sub) // len(eff_seeds)) if len(eff_seeds) > 0 else 0
            ppo_mean = float(sub["ppo_weight"].mean()) if not sub.empty else 0.0
            sac_mean = float(sub["sac_weight"].mean()) if not sub.empty else 0.0

            seed_corrs = []
            for s in eff_seeds:
                s_sub = sub[sub["seed"] == s]
                seed_corrs.append(
                    _safe_corr(
                        s_sub["ppo_weight"].to_numpy(),
                        s_sub["sac_weight"].to_numpy(),
                    )
                )
            mean_corr = float(np.mean(seed_corrs)) if seed_corrs else 0.0

            rows.append(
                {
                    "adx_threshold": float(th),
                    "regime": regime_label,
                    "n_days_per_seed": n_days_per_seed,
                    "ppo_mean_weight(%)": round(ppo_mean * 100.0, 2),
                    "sac_mean_weight(%)": round(sac_mean * 100.0, 2),
                    "weight_diff_ppo_minus_sac(%)": round((ppo_mean - sac_mean) * 100.0, 2),
                    "ppo_sac_action_corr": round(mean_corr, 4),
                }
            )

    regime_df = pd.DataFrame(rows)
    regime_df.to_csv(output_dir / "phase5_regime_behavior_val.csv", index=False)

    if verbose:
        print("\n[Phase 5] Validation ADX(14) Regime Classification & PPO/SAC Behavioral Analysis:")
        print(regime_df.to_string(index=False))

    return regime_df


def simulate_dynamic_ensemble_on_shared_env(
    ppo_model: Any,
    sac_model: Any,
    env: TradingEnv,
    adx_threshold: float,
    w_strong_ppo: float,
    w_weak_ppo: float,
    seed: int = 11,
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Runs a single Dynamic Ensemble rollout directly inside a shared `TradingEnv`.
    At each day t close:
    - Both `ppo_model` and `sac_model` observe the SAME shared `obs` from `env`.
    - Day t close `adx_14_raw` determines whether the market is in `Strong_Trend` (>= adx_threshold)
      or `Weak_Trend` (< adx_threshold).
    - Action-level blending:
        w_target = w_ppo * a_ppo + (1 - w_ppo) * a_sac
      where w_ppo = w_strong_ppo in Strong_Trend else w_weak_ppo.
    """
    obs, info0 = env.reset(seed=seed)
    adx_arr = env.df["adx_14_raw"].to_numpy(dtype=np.float64)

    records: List[Dict[str, Any]] = [
        {
            "Date": env.df.index[0],
            "Open": float(env.df["Open"].iloc[0]),
            "Close": float(env.df["Close"].iloc[0]),
            "adx_14_raw": float(adx_arr[0]),
            "regime": classify_adx_regime(adx_arr[0], adx_threshold),
            "ppo_action": 0.0,
            "sac_action": 0.0,
            "w_ppo_blend": 0.0,
            "target_weight": 0.0,
            "close_weight": 0.0,
            "equity_close": float(info0["equity"]),
            "cash": float(info0["cash"]),
            "shares": float(info0["shares"]),
            "turnover_weight": 0.0,
            "transaction_cost": 0.0,
            "peak_equity": float(info0["peak_equity"]),
            "drawdown": float(info0["drawdown"]),
            "port_ret_simple": 0.0,
            "port_ret_log": 0.0,
        }
    ]

    done = False
    while not done:
        t_idx = env.current_step
        adx_t = float(adx_arr[t_idx])
        regime_t = classify_adx_regime(adx_t, adx_threshold)
        w_ppo = float(w_strong_ppo) if regime_t == "Strong_Trend" else float(w_weak_ppo)

        a_ppo = _fast_predict_weight(ppo_model, obs)
        a_sac = _fast_predict_weight(sac_model, obs)

        w_ens = float(np.clip(w_ppo * a_ppo + (1.0 - w_ppo) * a_sac, 0.0, 1.0))
        obs, _, term, trunc, step_info = env.step(np.array([w_ens], dtype=np.float32))
        done = term or trunc

        records.append(
            {
                "Date": env.df.index[env.current_step],
                "Open": float(env.df["Open"].iloc[env.current_step]),
                "Close": float(env.df["Close"].iloc[env.current_step]),
                "adx_14_raw": adx_t,
                "regime": regime_t,
                "ppo_action": a_ppo,
                "sac_action": a_sac,
                "w_ppo_blend": w_ppo,
                "target_weight": float(step_info["target_weight"]),
                "close_weight": float(step_info["realized_weight"]),
                "equity_close": float(step_info["equity"]),
                "cash": float(step_info["cash"]),
                "shares": float(step_info["shares"]),
                "turnover_weight": float(step_info["turnover_weight"]),
                "transaction_cost": float(step_info["cost"]),
                "peak_equity": float(step_info["peak_equity"]),
                "drawdown": float(step_info["drawdown"]),
                "port_ret_simple": float(step_info["port_ret_simple"]),
                "port_ret_log": float(step_info["port_ret_log"]),
            }
        )

    traj_df = pd.DataFrame(records).set_index("Date")
    perf = compute_performance_metrics(sim_df=traj_df, config=config)
    return perf, traj_df


def run_phase6_ensemble_optimization(
    seeds: Optional[Tuple[int, ...]] = None,
    reward_type: str = "A",
    adx_thresholds: Tuple[float, ...] = DEFAULT_CONFIG.adx_threshold_candidates,
    strong_weights: Tuple[float, ...] = WEIGHT_CANDIDATES,
    weak_weights: Tuple[float, ...] = WEIGHT_CANDIDATES,
    checkpoints_root: Path = CHECKPOINTS_DIR,
    output_dir: Path = PHASE5_6_RESULTS_DIR,
    config: ExperimentConfig = DEFAULT_CONFIG,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Phase 6:
    1. Evaluates all 27 Dynamic Ensemble combinations (3 ADX thresholds x 3 strong weights x 3 weak weights)
       directly in the shared Validation `TradingEnv` across all seeds.
    2. Ranks the 27 rules using the fixed pre-defined criterion:
       1st: Seed Mean Sharpe (DESC)
       2nd: Seed Mean MDD (ASC)
       3rd: Inter-seed Sharpe Std (ASC)
    3. Freezes the #1 ensemble rule AND extracts each seed's Validation average stock weight
       to freeze the `Exposure-Matched Baseline` for Phase 7 Test evaluation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    eff_seeds = tuple(seeds if seeds is not None else config.random_seeds)

    # Pre-load PPO & SAC models per seed once to avoid redundant disk I/O across 27 rules
    models_by_seed: Dict[int, Tuple[Any, Any]] = {}
    for seed in eff_seeds:
        ppo_m, _ = load_trained_model(
            algorithm="ppo",
            reward_type=reward_type,
            seed=seed,
            checkpoints_root=checkpoints_root,
        )
        sac_m, _ = load_trained_model(
            algorithm="sac",
            reward_type=reward_type,
            seed=seed,
            checkpoints_root=checkpoints_root,
        )
        models_by_seed[int(seed)] = (ppo_m, sac_m)

    shared_val_env = TradingEnv(
        mode="val",
        reward_type=reward_type,
        one_way_cost_rate=config.one_way_cost_rate,
        config=config,
    )

    combinations = list(
        itertools.product(adx_thresholds, strong_weights, weak_weights)
    )
    assert len(combinations) == 27 or len(combinations) == (
        len(adx_thresholds) * len(strong_weights) * len(weak_weights)
    )

    per_seed_rows: List[Dict[str, Any]] = []

    for rule_idx, (adx_th, w_str, w_wk) in enumerate(combinations, start=1):
        rule_id = f"ADX{int(adx_th)}_S{int(w_str*100)}_W{int(w_wk*100)}"
        for seed in eff_seeds:
            ppo_m, sac_m = models_by_seed[int(seed)]
            perf, _ = simulate_dynamic_ensemble_on_shared_env(
                ppo_model=ppo_m,
                sac_model=sac_m,
                env=shared_val_env,
                adx_threshold=float(adx_th),
                w_strong_ppo=float(w_str),
                w_weak_ppo=float(w_wk),
                seed=int(seed),
                config=config,
            )
            per_seed_rows.append(
                {
                    "rule_idx": rule_idx,
                    "rule_id": rule_id,
                    "adx_threshold": float(adx_th),
                    "w_strong_ppo": float(w_str),
                    "w_weak_ppo": float(w_wk),
                    "seed": int(seed),
                    **perf,
                }
            )

    per_seed_df = pd.DataFrame(per_seed_rows)

    # Aggregate across seeds for each of the 27 rules
    agg_rows: List[Dict[str, Any]] = []
    for (rule_id, adx_th, w_str, w_wk), grp in per_seed_df.groupby(
        ["rule_id", "adx_threshold", "w_strong_ppo", "w_weak_ppo"], sort=False
    ):
        n_s = len(grp)
        sharpe_std = float(grp["Sharpe_Ratio"].std(ddof=1)) if n_s > 1 else 0.0
        mdd_std = float(grp["MDD(%)"].std(ddof=1)) if n_s > 1 else 0.0
        cagr_std = float(grp["CAGR(%)"].std(ddof=1)) if n_s > 1 else 0.0

        agg_rows.append(
            {
                "rule_id": rule_id,
                "adx_threshold": float(adx_th),
                "w_strong_ppo": float(w_str),
                "w_strong_sac": round(1.0 - float(w_str), 2),
                "w_weak_ppo": float(w_wk),
                "w_weak_sac": round(1.0 - float(w_wk), 2),
                "n_seeds": int(n_s),
                "mean_sharpe": round(float(grp["Sharpe_Ratio"].mean()), 4),
                "std_sharpe": round(sharpe_std, 4),
                "mean_mdd": round(float(grp["MDD(%)"].mean()), 2),
                "std_mdd": round(mdd_std, 2),
                "mean_cagr": round(float(grp["CAGR(%)"].mean()), 2),
                "std_cagr": round(cagr_std, 2),
                "mean_cum_ret": round(float(grp["Cumulative_Return(%)"].mean()), 2),
                "mean_vol": round(float(grp["Ann_Volatility(%)"].mean()), 2),
                "mean_stock_weight(%)": round(float(grp["Avg_Stock_Weight(%)"].mean()), 2),
                "mean_cost($)": round(float(grp["Total_Cost($)"].mean()), 2),
            }
        )

    agg_df = pd.DataFrame(agg_rows)
    ranked_27_df = agg_df.sort_values(
        by=["mean_sharpe", "mean_mdd", "std_sharpe"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    ranked_27_df["rank"] = np.arange(1, len(ranked_27_df) + 1)

    best_rule = ranked_27_df.iloc[0].to_dict()
    best_rule_id = best_rule["rule_id"]

    # Extract per-seed Validation average stock weight (in [0, 1]) for Exposure-Matched Baseline
    best_seed_rows = per_seed_df[per_seed_df["rule_id"] == best_rule_id]
    seed_exposure_weights: Dict[str, float] = {}
    for _, srow in best_seed_rows.iterrows():
        seed_exposure_weights[str(int(srow["seed"]))] = round(
            float(srow["Avg_Stock_Weight(%)"]) / 100.0, 6
        )

    mean_exposure_weight = round(
        float(np.mean(list(seed_exposure_weights.values()))), 6
    )

    frozen_ensemble_payload = {
        "selection_criteria": [
            "1_mean_sharpe_desc",
            "2_mean_mdd_asc",
            "3_std_sharpe_asc",
        ],
        "num_rules_evaluated": int(len(ranked_27_df)),
        "reward_type_used": reward_type,
        "seeds": list(eff_seeds),
        "frozen_best_rule": {
            "rule_id": best_rule["rule_id"],
            "adx_threshold": float(best_rule["adx_threshold"]),
            "w_strong_ppo": float(best_rule["w_strong_ppo"]),
            "w_strong_sac": float(best_rule["w_strong_sac"]),
            "w_weak_ppo": float(best_rule["w_weak_ppo"]),
            "w_weak_sac": float(best_rule["w_weak_sac"]),
            "val_mean_sharpe": float(best_rule["mean_sharpe"]),
            "val_mean_mdd": float(best_rule["mean_mdd"]),
            "val_std_sharpe": float(best_rule["std_sharpe"]),
        },
        "exposure_matched_baseline": {
            "description": "Frozen Validation average stock weight per seed from the selected ensemble rule",
            "seed_weights": seed_exposure_weights,
            "mean_weight": mean_exposure_weight,
        },
    }

    per_seed_df.to_csv(output_dir / "phase6_27rules_per_seed_val.csv", index=False)
    ranked_27_df.to_csv(output_dir / "phase6_27rules_ranking_val.csv", index=False)
    with open(output_dir / "phase6_frozen_ensemble.json", "w", encoding="utf-8") as f:
        json.dump(frozen_ensemble_payload, f, indent=2, ensure_ascii=False)

    if verbose:
        print("\n[Phase 6] Top 5 of 27 Dynamic Ensemble Rules on Shared Validation TradingEnv:")
        print(ranked_27_df.head(5).to_string(index=False))
        print(
            f"\n  -> Frozen #1 Ensemble Rule: {best_rule['rule_id']} "
            f"(ADX >= {best_rule['adx_threshold']}: PPO={best_rule['w_strong_ppo']}, "
            f"ADX < {best_rule['adx_threshold']}: PPO={best_rule['w_weak_ppo']}) | "
            f"Val Mean Sharpe={best_rule['mean_sharpe']:.4f}, Mean MDD={best_rule['mean_mdd']:.2f}%"
        )
        print(
            f"  -> Frozen Exposure-Matched Baseline Weights (by seed): {seed_exposure_weights} "
            f"(Mean={mean_exposure_weight:.4f})"
        )

    return {
        "per_seed_df": per_seed_df,
        "ranked_27_df": ranked_27_df,
        "frozen_ensemble_payload": frozen_ensemble_payload,
    }


if __name__ == "__main__":
    run_phase5_regime_behavior_analysis(verbose=True)
    run_phase6_ensemble_optimization(verbose=True)
