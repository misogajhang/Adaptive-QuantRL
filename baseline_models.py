"""
[Adaptive-QuantRL] Phase 1: Accounting Engine & 4 Baseline Strategies
Strictly implements Section 3, Section 4, and Phase 1 of 초안.md:
- Accounting sequence: Day t Close -> Day t+1 Open (overnight gap) -> Rebalance at Open
  (0.1% one-way cost = 0.05% commission + 0.05% slippage) -> Day t+1 Close.
- Dividend cashflows excluded (unadjusted OHLC price returns).
- 4 Baselines:
  1. Buy & Hold (B&H)
  2. Cash 100% (현금 100%)
  3. Fixed 50:50 (고정 50:50)
  4. Moving Average Crossover (이동평균 전략)
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import DEFAULT_CONFIG, ExperimentConfig


def execute_portfolio_step(
    cash_t: float,
    shares_t: float,
    close_t: float,
    open_t1: float,
    close_t1: float,
    target_weight: float,
    peak_equity_t: float,
    one_way_cost_rate: float = DEFAULT_CONFIG.one_way_cost_rate,
) -> Dict[str, float]:
    """
    Executes a single step of the portfolio accounting specification (Section 4):
    1. Equity_t^{close} = Cash_t + Shares_t * Close_t
    2. Equity_{t+1}^{open, pre} = Cash_t + Shares_t * Open_{t+1} (overnight gap reflected first)
    3. Rebalance to `target_weight` in [0, 1] at Open_{t+1} and deduct one-way cost (0.1%):
       - w_{t+1}^{pre} = (Shares_t * Open_{t+1}) / Equity_{t+1}^{open, pre}
       - turnover_weight = |target_weight - w_{t+1}^{pre}|
       - cost = turnover_weight * Equity_{t+1}^{open, pre} * one_way_cost_rate
       - Equity_{t+1}^{open, post} = Equity_{t+1}^{open, pre} - cost
       - Shares_{t+1} = (target_weight * Equity_{t+1}^{open, post}) / Open_{t+1}
       - Cash_{t+1} = (1.0 - target_weight) * Equity_{t+1}^{open, post}
    4. Equity_{t+1}^{close} = Cash_{t+1} + Shares_{t+1} * Close_{t+1}
    5. End-of-Day Peak Equity & Drawdown update.
    """
    w_target = float(np.clip(target_weight, 0.0, 1.0))

    # Step 1: Day t close equity
    equity_t_close = float(cash_t + shares_t * close_t)

    # Step 2: Day t+1 open pre-trade equity (overnight gap reflected)
    stock_val_open_pre = float(shares_t * open_t1)
    equity_t1_open_pre = float(cash_t + stock_val_open_pre)

    if equity_t1_open_pre <= 0.0:
        raise ValueError(f"Non-positive pre-trade open equity: {equity_t1_open_pre}")

    w_open_pre = stock_val_open_pre / equity_t1_open_pre

    # Step 3: Rebalance at Open_{t+1} and deduct one-way transaction cost (0.1%)
    turnover_weight = abs(w_target - w_open_pre)
    traded_notional = turnover_weight * equity_t1_open_pre
    transaction_cost = traded_notional * float(one_way_cost_rate)

    equity_t1_open_post = equity_t1_open_pre - transaction_cost
    stock_val_open_post = w_target * equity_t1_open_post
    cash_t1 = (1.0 - w_target) * equity_t1_open_post
    shares_t1 = stock_val_open_post / float(open_t1)

    # Accounting identity assertion at t+1 Open
    assert np.isclose(
        cash_t1 + shares_t1 * open_t1, equity_t1_open_post, rtol=1e-10, atol=1e-8
    ), "Open post-trade accounting identity violated!"

    # Step 4: Day t+1 close equity
    stock_val_t1_close = float(shares_t1 * close_t1)
    equity_t1_close = float(cash_t1 + stock_val_t1_close)

    # Actual end-of-day stock weight at t+1 close
    weight_t1_close = (
        stock_val_t1_close / equity_t1_close if equity_t1_close > 0.0 else 0.0
    )

    # Step 5: End-of-Day Peak Equity & Drawdown
    peak_equity_t1 = max(float(peak_equity_t), equity_t1_close)
    drawdown_t1 = (
        (peak_equity_t1 - equity_t1_close) / peak_equity_t1
        if peak_equity_t1 > 0.0
        else 0.0
    )

    # Step 6: Daily portfolio simple and log returns (Close_t -> Close_{t+1})
    port_ret_simple = (equity_t1_close / equity_t_close) - 1.0
    port_ret_log = float(np.log(equity_t1_close / equity_t_close))

    return {
        "equity_t_close": equity_t_close,
        "equity_t1_open_pre": equity_t1_open_pre,
        "equity_t1_open_post": equity_t1_open_post,
        "equity_t1_close": equity_t1_close,
        "cash_t1": cash_t1,
        "shares_t1": shares_t1,
        "target_weight": w_target,
        "w_open_pre": w_open_pre,
        "weight_t1_close": weight_t1_close,
        "turnover_weight": turnover_weight,
        "traded_notional": traded_notional,
        "transaction_cost": transaction_cost,
        "peak_equity_t1": peak_equity_t1,
        "drawdown_t1": drawdown_t1,
        "port_ret_simple": port_ret_simple,
        "port_ret_log": port_ret_log,
    }


def simulate_strategy(
    df: pd.DataFrame,
    weight_fn: Callable[[pd.DataFrame, int, float], float],
    strategy_name: str,
    config: ExperimentConfig = DEFAULT_CONFIG,
    obs_log_var_stats: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Runs a deterministic backtest over `df` following Section 3 & Section 4 rules:
    - Initial state at index 0 (first trading day close of the evaluation split):
      Cash = $10,000, Shares = 0, Drawdown = 0, Peak Equity = $10,000,
      portfolio return history = empty.
    - For t = 0, 1, ..., N-2:
      Decides target weight w_t using data up to day t close, executes at day t+1 open,
      and evaluates at day t+1 close.
    """
    n = len(df)
    if n < 2:
        raise ValueError("DataFrame must have at least 2 trading days to simulate.")

    cash = float(config.initial_cash)
    shares = float(config.initial_shares)
    peak_equity = float(config.initial_peak_equity)
    drawdown = float(config.initial_drawdown)
    current_close_weight = 0.0

    # Portfolio return history for 30-day variance (reset at start of episode/split)
    port_ret_history: List[float] = []

    records: List[Dict] = [
        {
            "Date": df.index[0],
            "strategy": strategy_name,
            "Open": float(df["Open"].iloc[0]),
            "Close": float(df["Close"].iloc[0]),
            "cash": cash,
            "shares": shares,
            "equity_open_pre": cash,
            "equity_open_post": cash,
            "equity_close": cash,
            "target_weight": 0.0,
            "close_weight": 0.0,
            "turnover_weight": 0.0,
            "transaction_cost": 0.0,
            "peak_equity": peak_equity,
            "drawdown": drawdown,
            "port_ret_simple": 0.0,
            "port_ret_log": 0.0,
            "var_30d_raw": 0.0,
            "var_30d_obs": float(config.initial_obs_variance),
        }
    ]

    for t in range(n - 1):
        close_t = float(df["Close"].iloc[t])
        open_t1 = float(df["Open"].iloc[t + 1])
        close_t1 = float(df["Close"].iloc[t + 1])

        # Target weight determined at day t close
        target_w = float(weight_fn(df, t, current_close_weight))

        step_res = execute_portfolio_step(
            cash_t=cash,
            shares_t=shares,
            close_t=close_t,
            open_t1=open_t1,
            close_t1=close_t1,
            target_weight=target_w,
            peak_equity_t=peak_equity,
            one_way_cost_rate=config.one_way_cost_rate,
        )

        cash = step_res["cash_t1"]
        shares = step_res["shares_t1"]
        peak_equity = step_res["peak_equity_t1"]
        drawdown = step_res["drawdown_t1"]
        current_close_weight = step_res["weight_t1_close"]

        port_ret_history.append(step_res["port_ret_simple"])

        # Section 4: Initial <30 days raw variance is 0.0, obs variance is fixed initial value
        if len(port_ret_history) < config.variance_window:
            var_30d_raw = 0.0
            var_30d_obs = float(config.initial_obs_variance)
        else:
            window_slice = port_ret_history[-config.variance_window :]
            var_30d_raw = float(np.var(window_slice, ddof=1))
            log_v = float(np.log(max(var_30d_raw, 0.0) + config.variance_eps))
            if obs_log_var_stats is not None:
                var_30d_obs = (log_v - obs_log_var_stats["mean"]) / obs_log_var_stats["std"]
            else:
                var_30d_obs = log_v

        records.append(
            {
                "Date": df.index[t + 1],
                "strategy": strategy_name,
                "Open": open_t1,
                "Close": close_t1,
                "cash": cash,
                "shares": shares,
                "equity_open_pre": step_res["equity_t1_open_pre"],
                "equity_open_post": step_res["equity_t1_open_post"],
                "equity_close": step_res["equity_t1_close"],
                "target_weight": step_res["target_weight"],
                "close_weight": current_close_weight,
                "turnover_weight": step_res["turnover_weight"],
                "transaction_cost": step_res["transaction_cost"],
                "peak_equity": peak_equity,
                "drawdown": drawdown,
                "port_ret_simple": step_res["port_ret_simple"],
                "port_ret_log": step_res["port_ret_log"],
                "var_30d_raw": var_30d_raw,
                "var_30d_obs": var_30d_obs,
            }
        )

    sim_df = pd.DataFrame(records).set_index("Date")
    return sim_df


def compute_performance_metrics(
    sim_df: pd.DataFrame,
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> Dict[str, float]:
    """
    Computes standard evaluation metrics from a simulated portfolio DataFrame:
    - Initial Equity ($), Final Equity ($)
    - Cumulative Return (%)
    - CAGR (%)
    - Annualized Volatility (%)
    - Sharpe Ratio (Rf = config.risk_free_rate)
    - Maximum Drawdown (MDD, %)
    - Calmar Ratio (CAGR / MDD)
    - Total Transaction Cost ($) & Total Turnover
    - Average Stock Exposure (%)
    """
    initial_eq = float(sim_df["equity_close"].iloc[0])
    final_eq = float(sim_df["equity_close"].iloc[-1])
    cum_return = (final_eq / initial_eq) - 1.0

    # Daily returns excluding the initial t=0 setup row
    daily_rets = sim_df["port_ret_simple"].iloc[1:].to_numpy(dtype=float)
    n_periods = len(daily_rets)
    years = n_periods / float(config.trading_days_per_year)

    cagr = (final_eq / initial_eq) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    daily_std = float(np.std(daily_rets, ddof=1)) if n_periods > 1 else 0.0
    ann_vol = daily_std * np.sqrt(config.trading_days_per_year)

    daily_rf = config.risk_free_rate / float(config.trading_days_per_year)
    excess_daily = daily_rets - daily_rf
    mean_excess = float(np.mean(excess_daily)) if n_periods > 0 else 0.0

    if daily_std > 1e-12:
        sharpe = (mean_excess / daily_std) * np.sqrt(config.trading_days_per_year)
    else:
        sharpe = 0.0

    max_dd = float(sim_df["drawdown"].max())
    calmar = (cagr / max_dd) if max_dd > 1e-12 else 0.0

    total_cost = float(sim_df["transaction_cost"].sum())
    total_turnover = float(sim_df["turnover_weight"].sum())
    avg_weight = float(sim_df["target_weight"].iloc[1:].mean())

    return {
        "Initial_Equity($)": round(initial_eq, 2),
        "Final_Equity($)": round(final_eq, 2),
        "Cumulative_Return(%)": round(cum_return * 100.0, 2),
        "CAGR(%)": round(cagr * 100.0, 2),
        "Ann_Volatility(%)": round(ann_vol * 100.0, 2),
        "Sharpe_Ratio": round(float(sharpe), 4),
        "MDD(%)": round(max_dd * 100.0, 2),
        "Calmar_Ratio": round(float(calmar), 4),
        "Avg_Stock_Weight(%)": round(avg_weight * 100.0, 2),
        "Total_Turnover": round(total_turnover, 4),
        "Total_Cost($)": round(total_cost, 2),
        "Trading_Days": int(n_periods),
    }


def run_all_baselines(
    split_df: pd.DataFrame,
    split_name: str,
    config: ExperimentConfig = DEFAULT_CONFIG,
    obs_log_var_stats: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Evaluates the 4 core Baseline strategies on a given split (Train / Val / Test):
    1. Buy & Hold (B&H): Target weight = 1.0 (buys 100% SPY at first open and holds)
    2. Cash 100%: Target weight = 0.0 (holds 100% cash)
    3. Fixed 50:50: Target weight = 0.5 (rebalances to 50% stock / 50% cash)
    4. Moving Average (SMA 20/50 Crossover): Target weight = 1.0 if SMA_20 > SMA_50 at t close else 0.0
    """
    strategies: Dict[str, Callable[[pd.DataFrame, int, float], float]] = {
        "1_Buy_and_Hold": lambda df, t, cur_w: 1.0,
        "2_Cash_100": lambda df, t, cur_w: 0.0,
        "3_Fixed_50_50": lambda df, t, cur_w: 0.5,
        "4_Moving_Average_20_50": lambda df, t, cur_w: float(
            df["ma_signal_20_50"].iloc[t]
        ),
    }

    metrics_rows: List[Dict] = []
    trajectories: Dict[str, pd.DataFrame] = {}

    for strat_name, weight_fn in strategies.items():
        sim_df = simulate_strategy(
            df=split_df,
            weight_fn=weight_fn,
            strategy_name=strat_name,
            config=config,
            obs_log_var_stats=obs_log_var_stats,
        )
        trajectories[strat_name] = sim_df
        m = compute_performance_metrics(sim_df=sim_df, config=config)
        row = {"Split": split_name, "Strategy": strat_name, **m}
        metrics_rows.append(row)

    summary_df = pd.DataFrame(metrics_rows)
    return summary_df, trajectories
