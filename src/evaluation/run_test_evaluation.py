"""
[Adaptive-QuantRL] Phase 7 & Phase 8: Independent Test Evaluation, Cost Sensitivity & 4 RQ Report
Implements:
1. Phase 7: Deterministic evaluation on Test (2023~2024) of:
   - 4 Traditional Benchmarks (Buy & Hold, Cash 100%, Fixed 50:50, Moving Average 20/50)
   - Frozen Exposure-Matched Baseline (seed-specific Validation average stock weights)
   - Frozen Single RL Models (PPO, SAC)
   - Frozen #1 Dynamic Ensemble Rule in shared Test TradingEnv
   - Frozen Reward Ablation Models (Reward A, B, C for PPO & SAC)
2. Transaction Cost Sensitivity Analysis:
   - Policy parameters fixed at 0.1% training cost; evaluated under 0.1%, 0.3%, and 0.5% one-way cost.
   - Identifies threshold cost interval where excess performance vs benchmark turns <= 0.
3. Phase 8:
   - Reports Test SPY market statistics & ADX regime sample sizes (N trading days).
   - Generates structured CSV tables and Markdown report (`phase8_rq_verification_report.md`)
     verifying all 4 Research Questions (RQ1 ~ RQ4).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from baseline_models import compute_performance_metrics, run_all_baselines, simulate_strategy
from config import CHECKPOINTS_DIR, DEFAULT_CONFIG, PROJECT_ROOT, ExperimentConfig
from envs.trading_env import TradingEnv
from src.ensemble.dynamic_ensemble import (
    PHASE5_6_RESULTS_DIR,
    compute_regime_sample_counts,
    simulate_dynamic_ensemble_on_shared_env,
)
from src.evaluation.validate_rewards import (
    PHASE4_RESULTS_DIR,
    evaluate_single_checkpoint_on_env,
)
from training.train_models import load_trained_model


PHASE7_8_RESULTS_DIR = PROJECT_ROOT / "results" / "phase7_8"
COST_SENSITIVITY_RATES: Tuple[float, ...] = (0.001, 0.003, 0.005)


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Formats a DataFrame as a GitHub-Flavored Markdown table without requiring `tabulate`."""
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body_lines = []
    for _, row in df.iterrows():
        vals = [str(row[c]) for c in df.columns]
        body_lines.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *body_lines])


def _aggregate_seed_metrics(
    df: pd.DataFrame,
    group_cols: List[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for keys, grp in df.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_dict = dict(zip(group_cols, keys))
        n_s = len(grp)
        rows.append(
            {
                **key_dict,
                "n_seeds": int(n_s),
                "mean_final_equity($)": round(float(grp["Final_Equity($)"].mean()), 2),
                "mean_cum_return(%)": round(float(grp["Cumulative_Return(%)"].mean()), 2),
                "std_cum_return(%)": round(
                    float(grp["Cumulative_Return(%)"].std(ddof=1)) if n_s > 1 else 0.0, 2
                ),
                "mean_cagr(%)": round(float(grp["CAGR(%)"].mean()), 2),
                "std_cagr(%)": round(
                    float(grp["CAGR(%)"].std(ddof=1)) if n_s > 1 else 0.0, 2
                ),
                "mean_vol(%)": round(float(grp["Ann_Volatility(%)"].mean()), 2),
                "mean_sharpe": round(float(grp["Sharpe_Ratio"].mean()), 4),
                "std_sharpe": round(
                    float(grp["Sharpe_Ratio"].std(ddof=1)) if n_s > 1 else 0.0, 4
                ),
                "mean_mdd(%)": round(float(grp["MDD(%)"].mean()), 2),
                "std_mdd(%)": round(
                    float(grp["MDD(%)"].std(ddof=1)) if n_s > 1 else 0.0, 2
                ),
                "mean_calmar": round(float(grp["Calmar_Ratio"].mean()), 4),
                "mean_stock_weight(%)": round(float(grp["Avg_Stock_Weight(%)"].mean()), 2),
                "mean_turnover": round(float(grp["Total_Turnover"].mean()), 4),
                "mean_cost($)": round(float(grp["Total_Cost($)"].mean()), 2),
            }
        )
    return pd.DataFrame(rows)


def evaluate_exposure_matched_baseline_on_test(
    test_df: pd.DataFrame,
    seed_weights: Dict[str, float],
    one_way_cost_rate: float = DEFAULT_CONFIG.one_way_cost_rate,
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Evaluates the frozen Exposure-Matched Baseline on Test for each seed's
    frozen Validation average stock weight.
    """
    from dataclasses import replace

    cost_cfg = replace(config, one_way_cost_rate=float(one_way_cost_rate))
    rows: List[Dict[str, Any]] = []
    for seed_str, target_w in seed_weights.items():
        w_fixed = float(np.clip(target_w, 0.0, 1.0))
        sim_df = simulate_strategy(
            df=test_df,
            weight_fn=lambda df, t, cur_w, w=w_fixed: w,
            strategy_name=f"Exposure_Matched_Seed_{seed_str}",
            config=cost_cfg,
        )
        perf = compute_performance_metrics(sim_df=sim_df, config=cost_cfg)
        rows.append(
            {
                "Category": "Exposure_Matched_Baseline",
                "Model": "Exposure_Matched_Baseline",
                "algorithm": "BASELINE",
                "reward_type": "-",
                "seed": int(seed_str),
                "one_way_cost_rate": float(one_way_cost_rate),
                **perf,
            }
        )
    return pd.DataFrame(rows)


def run_phase7_and_8_evaluation(
    seeds: Optional[Tuple[int, ...]] = None,
    cost_rates: Tuple[float, ...] = COST_SENSITIVITY_RATES,
    checkpoints_root: Path = CHECKPOINTS_DIR,
    phase4_dir: Path = PHASE4_RESULTS_DIR,
    phase6_dir: Path = PHASE5_6_RESULTS_DIR,
    output_dir: Path = PHASE7_8_RESULTS_DIR,
    config: ExperimentConfig = DEFAULT_CONFIG,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Executes Phase 7 (Independent Test Evaluation & Transaction Cost Sensitivity Analysis)
    and Phase 8 (4 Research Questions Verification & Automated CSV/Markdown Reporting).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    eff_seeds = tuple(seeds if seeds is not None else config.random_seeds)

    # Load frozen ensemble specification from Phase 6
    frozen_ens_path = phase6_dir / "phase6_frozen_ensemble.json"
    if not frozen_ens_path.exists():
        raise FileNotFoundError(
            f"Frozen ensemble file not found at {frozen_ens_path}. Run Phase 6 first."
        )
    with open(frozen_ens_path, "r", encoding="utf-8") as f:
        frozen_ens = json.load(f)

    best_rule = frozen_ens["frozen_best_rule"]
    adx_th = float(best_rule["adx_threshold"])
    w_str_ppo = float(best_rule["w_strong_ppo"])
    w_wk_ppo = float(best_rule["w_weak_ppo"])
    seed_exp_weights: Dict[str, float] = frozen_ens["exposure_matched_baseline"][
        "seed_weights"
    ]

    # Reference Test environment to access Test DataFrame & regime sample sizes
    base_test_env = TradingEnv(mode="test", reward_type="A", config=config)
    test_df = base_test_env.df

    # Phase 8: Test Market Context & Regime Sample Counts (N)
    test_regime_counts_df = compute_regime_sample_counts(
        df=test_df,
        thresholds=config.adx_threshold_candidates,
    )
    test_regime_counts_df.to_csv(
        output_dir / "phase8_test_regime_sample_counts.csv", index=False
    )

    # Evaluate across cost sensitivity rates (0.1%, 0.3%, 0.5%)
    from dataclasses import replace

    all_cost_seed_rows: List[Dict[str, Any]] = []
    all_cost_bench_rows: List[Dict[str, Any]] = []

    # Pre-load models and metadata once
    loaded_models: Dict[Tuple[str, str, int], Tuple[Any, Dict[str, Any]]] = {}
    for algo in ("ppo", "sac"):
        for rew in ("A", "B", "C"):
            for seed in eff_seeds:
                m, meta = load_trained_model(
                    algorithm=algo,
                    reward_type=rew,
                    seed=seed,
                    checkpoints_root=checkpoints_root,
                )
                loaded_models[(algo, rew, int(seed))] = (m, meta)

    for cost_rate in cost_rates:
        cost_cfg = replace(config, one_way_cost_rate=float(cost_rate))

        # 1. 4 Traditional Benchmarks at `cost_rate`
        bench_summary_df, _ = run_all_baselines(
            split_df=test_df,
            split_name="Test (2023-2024)",
            config=cost_cfg,
        )
        for _, brow in bench_summary_df.iterrows():
            all_cost_bench_rows.append(
                {
                    "Category": "Traditional_Benchmark",
                    "Model": brow["Strategy"],
                    "algorithm": "BENCHMARK",
                    "reward_type": "-",
                    "one_way_cost_rate": float(cost_rate),
                    "n_seeds": 1,
                    "mean_final_equity($)": float(brow["Final_Equity($)"]),
                    "mean_cum_return(%)": float(brow["Cumulative_Return(%)"]),
                    "std_cum_return(%)": 0.0,
                    "mean_cagr(%)": float(brow["CAGR(%)"]),
                    "std_cagr(%)": 0.0,
                    "mean_vol(%)": float(brow["Ann_Volatility(%)"]),
                    "mean_sharpe": float(brow["Sharpe_Ratio"]),
                    "std_sharpe": 0.0,
                    "mean_mdd(%)": float(brow["MDD(%)"]),
                    "std_mdd(%)": 0.0,
                    "mean_calmar": float(brow["Calmar_Ratio"]),
                    "mean_stock_weight(%)": float(brow["Avg_Stock_Weight(%)"]),
                    "mean_turnover": float(brow["Total_Turnover"]),
                    "mean_cost($)": float(brow["Total_Cost($)"]),
                }
            )

        # 2. Exposure-Matched Baseline at `cost_rate`
        exp_df = evaluate_exposure_matched_baseline_on_test(
            test_df=test_df,
            seed_weights=seed_exp_weights,
            one_way_cost_rate=float(cost_rate),
            config=cost_cfg,
        )
        all_cost_seed_rows.extend(exp_df.to_dict(orient="records"))

        # 3. Single RL & Reward Ablation Models (PPO/SAC x A/B/C) at `cost_rate`
        for algo in ("ppo", "sac"):
            for rew in ("A", "B", "C"):
                for seed in eff_seeds:
                    mrow, _ = evaluate_single_checkpoint_on_env(
                        algorithm=algo,
                        reward_type=rew,
                        seed=seed,
                        mode="test",
                        one_way_cost_rate=float(cost_rate),
                        checkpoints_root=checkpoints_root,
                        config=cost_cfg,
                        preloaded_model_and_meta=loaded_models[(algo, rew, int(seed))],
                    )
                    cat = (
                        "Single_RL_Model"
                        if rew == "A"
                        else "Reward_Ablation_Model"
                    )
                    all_cost_seed_rows.append(
                        {
                            "Category": cat,
                            "Model": f"{algo.upper()}_Reward_{rew}",
                            "algorithm": algo.upper(),
                            "reward_type": rew,
                            "seed": int(seed),
                            "one_way_cost_rate": float(cost_rate),
                            **{
                                k: mrow[k]
                                for k in [
                                    "Final_Equity($)",
                                    "Cumulative_Return(%)",
                                    "CAGR(%)",
                                    "Ann_Volatility(%)",
                                    "Sharpe_Ratio",
                                    "MDD(%)",
                                    "Calmar_Ratio",
                                    "Avg_Stock_Weight(%)",
                                    "Total_Turnover",
                                    "Total_Cost($)",
                                ]
                            },
                        }
                    )

        # 4. Frozen Dynamic Ensemble in Shared Test TradingEnv at `cost_rate`
        shared_test_env = TradingEnv(
            mode="test",
            reward_type="A",
            one_way_cost_rate=float(cost_rate),
            config=cost_cfg,
        )
        for seed in eff_seeds:
            ppo_m, _ = loaded_models[("ppo", "A", int(seed))]
            sac_m, _ = loaded_models[("sac", "A", int(seed))]
            ens_perf, _ = simulate_dynamic_ensemble_on_shared_env(
                ppo_model=ppo_m,
                sac_model=sac_m,
                env=shared_test_env,
                adx_threshold=adx_th,
                w_strong_ppo=w_str_ppo,
                w_weak_ppo=w_wk_ppo,
                seed=int(seed),
                config=cost_cfg,
            )
            all_cost_seed_rows.append(
                {
                    "Category": "Dynamic_Ensemble",
                    "Model": f"Dynamic_Ensemble_{best_rule['rule_id']}",
                    "algorithm": "ENSEMBLE",
                    "reward_type": "A",
                    "seed": int(seed),
                    "one_way_cost_rate": float(cost_rate),
                    **{
                        k: ens_perf[k]
                        for k in [
                            "Final_Equity($)",
                            "Cumulative_Return(%)",
                            "CAGR(%)",
                            "Ann_Volatility(%)",
                            "Sharpe_Ratio",
                            "MDD(%)",
                            "Calmar_Ratio",
                            "Avg_Stock_Weight(%)",
                            "Total_Turnover",
                            "Total_Cost($)",
                        ]
                    },
                }
            )

    per_seed_all_costs_df = pd.DataFrame(all_cost_seed_rows)
    agg_rl_all_costs_df = _aggregate_seed_metrics(
        per_seed_all_costs_df,
        group_cols=["Category", "Model", "algorithm", "reward_type", "one_way_cost_rate"],
    )
    bench_all_costs_df = pd.DataFrame(all_cost_bench_rows)

    full_cost_table_df = pd.concat(
        [bench_all_costs_df, agg_rl_all_costs_df], ignore_index=True
    )

    # Extract baseline 0.1% Test summary table
    base_cost = float(config.one_way_cost_rate)
    test_summary_01_df = full_cost_table_df[
        np.isclose(full_cost_table_df["one_way_cost_rate"], base_cost)
    ].copy()
    per_seed_01_df = per_seed_all_costs_df[
        np.isclose(per_seed_all_costs_df["one_way_cost_rate"], base_cost)
    ].copy()

    # Build Cost Sensitivity Analysis Table (0.1%, 0.3%, 0.5%) with Excess vs Fixed_50_50 & Exposure_Matched
    cost_sensitivity_rows: List[Dict[str, Any]] = []
    for cost_rate in cost_rates:
        sub_c = full_cost_table_df[
            np.isclose(full_cost_table_df["one_way_cost_rate"], float(cost_rate))
        ]
        f50_row = sub_c[sub_c["Model"] == "3_Fixed_50_50"].iloc[0]
        exp_row = sub_c[sub_c["Model"] == "Exposure_Matched_Baseline"].iloc[0]
        ma_row = sub_c[sub_c["Model"] == "4_Moving_Average_20_50"].iloc[0]

        for _, r in sub_c.iterrows():
            excess_sharpe_vs_f50 = float(r["mean_sharpe"] - f50_row["mean_sharpe"])
            excess_cagr_vs_f50 = float(r["mean_cagr(%)"] - f50_row["mean_cagr(%)"])
            excess_sharpe_vs_exp = float(r["mean_sharpe"] - exp_row["mean_sharpe"])
            excess_cagr_vs_exp = float(r["mean_cagr(%)"] - exp_row["mean_cagr(%)"])
            excess_sharpe_vs_ma = float(r["mean_sharpe"] - ma_row["mean_sharpe"])

            cost_sensitivity_rows.append(
                {
                    "one_way_cost(%)": round(float(cost_rate) * 100.0, 2),
                    "Category": r["Category"],
                    "Model": r["Model"],
                    "mean_cagr(%)": r["mean_cagr(%)"],
                    "mean_sharpe": r["mean_sharpe"],
                    "mean_mdd(%)": r["mean_mdd(%)"],
                    "mean_cost($)": r["mean_cost($)"],
                    "excess_cagr_vs_Fixed5050(%)": round(excess_cagr_vs_f50, 2),
                    "excess_sharpe_vs_Fixed5050": round(excess_sharpe_vs_f50, 4),
                    "excess_cagr_vs_ExpMatched(%)": round(excess_cagr_vs_exp, 2),
                    "excess_sharpe_vs_ExpMatched": round(excess_sharpe_vs_exp, 4),
                    "excess_sharpe_vs_MA20_50": round(excess_sharpe_vs_ma, 4),
                    "outperforms_Fixed5050(Sharpe>0)": bool(excess_sharpe_vs_f50 > 0.0),
                    "outperforms_ExpMatched(Sharpe>0)": bool(excess_sharpe_vs_exp > 0.0),
                }
            )

    cost_sensitivity_df = pd.DataFrame(cost_sensitivity_rows)

    # Save CSV artifacts
    test_summary_01_df.to_csv(output_dir / "phase7_test_summary_table.csv", index=False)
    per_seed_01_df.to_csv(output_dir / "phase7_test_per_seed_metrics.csv", index=False)
    cost_sensitivity_df.to_csv(output_dir / "phase7_cost_sensitivity.csv", index=False)

    # Generate Phase 8 Markdown Report answering RQ1 ~ RQ4
    report_md = _build_phase8_markdown_report(
        test_summary_01_df=test_summary_01_df,
        per_seed_01_df=per_seed_01_df,
        cost_sensitivity_df=cost_sensitivity_df,
        test_regime_counts_df=test_regime_counts_df,
        frozen_ens=frozen_ens,
        config=config,
    )
    report_path = output_dir / "phase8_rq_verification_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    if verbose:
        print("\n[Phase 7] Test (2023~2024) Evaluation Summary (One-way Cost = 0.1%):")
        display_cols = [
            "Category",
            "Model",
            "mean_cum_return(%)",
            "mean_cagr(%)",
            "mean_vol(%)",
            "mean_sharpe",
            "std_sharpe",
            "mean_mdd(%)",
            "mean_stock_weight(%)",
            "mean_cost($)",
        ]
        print(test_summary_01_df[display_cols].to_string(index=False))
        print(f"\n[Phase 8] 4 RQ Verification Report generated at: {report_path}")

    return {
        "test_summary_01_df": test_summary_01_df,
        "per_seed_01_df": per_seed_01_df,
        "cost_sensitivity_df": cost_sensitivity_df,
        "test_regime_counts_df": test_regime_counts_df,
        "report_path": report_path,
    }


def _build_phase8_markdown_report(
    test_summary_01_df: pd.DataFrame,
    per_seed_01_df: pd.DataFrame,
    cost_sensitivity_df: pd.DataFrame,
    test_regime_counts_df: pd.DataFrame,
    frozen_ens: Dict[str, Any],
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> str:
    """
    Constructs a transparent, structured Markdown report addressing all 4 Research Questions
    and explicitly stating the POMDP, dividend-exclusion, and market timing findings.
    """
    best_rule = frozen_ens["frozen_best_rule"]
    adx_th = float(best_rule["adx_threshold"])

    bh_row = test_summary_01_df[test_summary_01_df["Model"] == "1_Buy_and_Hold"].iloc[0]
    exp_row = test_summary_01_df[
        test_summary_01_df["Model"] == "Exposure_Matched_Baseline"
    ].iloc[0]
    ens_row = test_summary_01_df[
        test_summary_01_df["Category"] == "Dynamic_Ensemble"
    ].iloc[0]

    # Check Ensemble vs Exposure-Matched Baseline market timing significance (RQ2)
    ens_sharpe_diff = float(ens_row["mean_sharpe"] - exp_row["mean_sharpe"])
    ens_cagr_diff = float(ens_row["mean_cagr(%)"] - exp_row["mean_cagr(%)"])
    if ens_sharpe_diff > 0.0 and ens_cagr_diff > 0.0:
        rq2_timing_conclusion = (
            f"동적 앙상블(`{ens_row['Model']}`)이 Exposure-Matched Baseline 대비 "
            f"평균 Sharpe `+{ens_sharpe_diff:.4f}`, 평균 CAGR `+{ens_cagr_diff:.2f}%p`의 양(+)의 초과 성과를 기록하여 "
            f"노출 통제 하에서도 유효한 시장 타이밍 효과를 보였습니다."
        )
    else:
        rq2_timing_conclusion = (
            f"동적 앙상블(`{ens_row['Model']}`)이 Exposure-Matched Baseline 대비 "
            f"평균 Sharpe `{ens_sharpe_diff:+.4f}`, 평균 CAGR `{ens_cagr_diff:+.2f}%p`를 기록하여, "
            f"평균 주식 노출 비중을 통제한 조건에서는 유의미한 초과 성과(시장 타이밍 효과)가 부족함을 투명하게 보고합니다."
        )

    # Check Reward Ablation A -> B (Drawdown penalty) and B -> C (Variance penalty) consistency across seeds
    ablation_lines: List[str] = []
    for algo in ("PPO", "SAC"):
        sub_a = per_seed_01_df[
            (per_seed_01_df["algorithm"] == algo) & (per_seed_01_df["reward_type"] == "A")
        ].set_index("seed")
        sub_b = per_seed_01_df[
            (per_seed_01_df["algorithm"] == algo) & (per_seed_01_df["reward_type"] == "B")
        ].set_index("seed")
        sub_c = per_seed_01_df[
            (per_seed_01_df["algorithm"] == algo) & (per_seed_01_df["reward_type"] == "C")
        ].set_index("seed")

        common_seeds = sorted(list(set(sub_a.index) & set(sub_b.index) & set(sub_c.index)))
        n_total = len(common_seeds)
        b_improves_mdd = sum(
            1 for s in common_seeds if sub_b.loc[s, "MDD(%)"] <= sub_a.loc[s, "MDD(%)"]
        )
        c_improves_vol = sum(
            1
            for s in common_seeds
            if sub_c.loc[s, "Ann_Volatility(%)"] <= sub_b.loc[s, "Ann_Volatility(%)"]
        )
        mdd_diff_ab = float(sub_b["MDD(%)"].mean() - sub_a["MDD(%)"].mean())
        vol_diff_bc = float(
            sub_c["Ann_Volatility(%)"].mean() - sub_b["Ann_Volatility(%)"].mean()
        )
        ablation_lines.append(
            f"- **{algo}**: Reward A $\\to$ B 평균 MDD 변화 `{mdd_diff_ab:+.2f}%p` "
            f"(일관성: `{b_improves_mdd}/{n_total}` Seeds MDD 개선/유지) | "
            f"Reward B $\\to$ C 평균 변동성 변화 `{vol_diff_bc:+.2f}%p` "
            f"(일관성: `{c_improves_vol}/{n_total}` Seeds 변동성 개선/유지)"
        )

    lines = [
        "# [Adaptive-QuantRL] Phase 7 & 8: 최종 Test 검증 및 4대 연구 질문(RQ) 보고서",
        "",
        "## 1. 연구 범위, 한계 및 Test 구간(2023~2024) 시장 환경",
        f"- **평가 구간**: `{config.test_start} ~ {config.test_end}` (배당 현금흐름 제외 비수정 SPY 가격 기준, 편도 기본 비용 `0.1%`)",
        f"- **SPY (Buy & Hold) 기준 성과**: 누적 수익률 `{bh_row['mean_cum_return(%)']:.2f}%`, "
        f"CAGR `{bh_row['mean_cagr(%)']:.2f}%`, 연 변동성 `{bh_row['mean_vol(%)']:.2f}%`, "
        f"Sharpe `{bh_row['mean_sharpe']:.4f}`, MDD `{bh_row['mean_mdd(%)']:.2f}%`",
        "- **한계 명시**: 본 결과는 배당을 제외한 가격 수익률 기준이며, 부분 관측 환경(POMDP) 및 단일 자산(SPY) 2023~2024 상승장 국면에 기반하므로 타 자산이나 표본이 부족한 국면에 무분별하게 일반화하지 않는다.",
        "",
        "### Test 구간 ADX(14) 국면별 표본 크기 (거래일 수 $N$)",
        _df_to_markdown(test_regime_counts_df),
        "",
        "---",
        "",
        "## 2.RQ별 핵심 검증 결과",
        "",
        "### RQ 1: 전략적 차이 (강화학습 동적 비중 조절 vs 4대 전통적 벤치마크)",
        _df_to_markdown(
            test_summary_01_df[
                [
                    "Category",
                    "Model",
                    "mean_cum_return(%)",
                    "mean_cagr(%)",
                    "mean_vol(%)",
                    "mean_sharpe",
                    "std_sharpe",
                    "mean_mdd(%)",
                    "mean_stock_weight(%)",
                    "mean_cost($)",
                ]
            ]
        ),
        "",
        "### RQ 2: 행동 특성 및 동적 앙상블 실효성 (Exposure-Matched Baseline 비교)",
        f"- **동결된 최적 앙상블 규칙**: `{best_rule['rule_id']}` "
        f"(ADX $\\ge {adx_th}$: PPO 가중치 `{best_rule['w_strong_ppo']}`, "
        f"ADX $< {adx_th}$: PPO 가중치 `{best_rule['w_weak_ppo']}`)",
        f"- **Exposure-Matched Baseline 평균 목표 비중**: `{frozen_ens['exposure_matched_baseline']['mean_weight']*100:.2f}%`",
        f"- **검증 결론**: {rq2_timing_conclusion}",
        "",
        "### RQ 3: 위험 페널티의 일관성 (Reward Ablation: A vs B vs C)",
        *ablation_lines,
        "",
        "### RQ 4: 재현성 및 거래 비용 민감도 분석 (편도 0.1%, 0.3%, 0.5%)",
        _df_to_markdown(cost_sensitivity_df),
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    run_phase7_and_8_evaluation(verbose=True)
