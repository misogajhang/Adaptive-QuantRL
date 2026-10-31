"""
[Adaptive-QuantRL] Phase 2 Unit Tests: TradingEnv, Accounting & Reward Separation
Tests:
1. Stable-Baselines3 `check_env` compatibility & Space/Info contracts
2. Overnight gap PnL accuracy & temporal alignment (t close -> t+1 open -> t+1 close)
3. One-way 0.1% cost deduction, zero double-deduction, non-negative cash, and Phase 1 baseline equivalence
4. Strict separation of Observation transformed variance vs Reward raw variance (including <30d zero rule)
5. Episode management protocol (Train random 252-day slicing vs Val/Test full deterministic traversal)
"""

import numpy as np
import pandas as pd
import pytest
from stable_baselines3.common.env_checker import check_env

from baseline_models import run_all_baselines
from config import DEFAULT_CONFIG
from envs.trading_env import RewardCalculator, TradingEnv, compute_reward


def _make_synthetic_df(
    open_prices: list[float],
    close_prices: list[float],
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Helper to create a synthetic DataFrame with required OHLC and market features."""
    assert len(open_prices) == len(close_prices)
    n = len(open_prices)
    if feature_cols is None:
        feature_cols = list(DEFAULT_CONFIG.market_feature_columns)

    dates = pd.bdate_range("2020-01-01", periods=n)
    data = {
        "Open": np.array(open_prices, dtype=np.float64),
        "High": np.maximum(open_prices, close_prices) * 1.01,
        "Low": np.minimum(open_prices, close_prices) * 0.99,
        "Close": np.array(close_prices, dtype=np.float64),
        "Volume": np.full(n, 1_000_000.0),
    }
    rng = np.random.default_rng(42)
    for col in feature_cols:
        data[col] = rng.normal(0.0, 1.0, size=n).astype(np.float32)

    return pd.DataFrame(data, index=dates)


def test_sb3_check_env_and_spaces_contract() -> None:
    """
    1. Verifies Stable-Baselines3 `check_env` passes without error across Train/Val/Test
       and Reward A/B/C configurations.
    2. Verifies Action Space, Observation Space, and Info dictionary keys.
    """
    for mode in ("train", "val", "test"):
        for rtype in ("A", "B", "C"):
            env = TradingEnv(mode=mode, reward_type=rtype, lambda_d=0.1, lambda_v=10.0)
            check_env(env, warn=True)

            obs, info = env.reset(seed=11)
            expected_obs_dim = (30 * len(DEFAULT_CONFIG.market_feature_columns)) + 3
            assert env.action_space.shape == (1,)
            assert env.action_space.dtype == np.float32
            assert float(env.action_space.low[0]) == 0.0
            assert float(env.action_space.high[0]) == 1.0

            assert env.observation_space.shape == (expected_obs_dim,)
            assert env.observation_space.dtype == np.float32
            assert obs.shape == (expected_obs_dim,)
            assert obs.dtype == np.float32

            obs_next, reward, terminated, truncated, step_info = env.step(
                np.array([0.6], dtype=np.float32)
            )
            assert obs_next.shape == (expected_obs_dim,)
            assert obs_next.dtype == np.float32
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)

            required_info_keys = {
                "equity",
                "cash",
                "shares",
                "realized_weight",
                "drawdown",
                "cost",
                "raw_variance",
            }
            assert required_info_keys.issubset(step_info.keys())


def test_overnight_gap_and_temporal_alignment() -> None:
    """
    Verifies exact temporal alignment:
    t close observation/action -> t+1 open overnight gap reflected first ->
    t+1 open rebalancing at Open_{t+1} with 0.1% cost -> t+1 close valuation.
    """
    # Day 0: Open=100, Close=100 (Initial state: Cash=$10,000, Shares=0)
    # Day 1: Open=110 (+10% gap while holding 0 shares -> no overnight gain), Close=121 (+10% intraday)
    # Day 2: Open=133.1 (+10% overnight gap while holding shares -> gap gain!), Close=120
    df = _make_synthetic_df(
        open_prices=[100.0, 110.0, 133.1],
        close_prices=[100.0, 121.0, 120.0],
    )
    env = TradingEnv(
        df=df,
        mode="test",
        reward_type="A",
        initial_cash=10_000.0,
        one_way_cost_rate=0.001,
    )
    obs0, info0 = env.reset()
    assert info0["equity"] == pytest.approx(10_000.0)
    assert info0["shares"] == pytest.approx(0.0)

    # Step 0 (Day 0 Close -> Day 1 Open -> Day 1 Close):
    # Action at Day 0 Close: target_weight = 1.0
    # At Day 1 Open ($110): pre-trade equity is still $10,000 (since shares_0 == 0).
    # One-way cost = 0.1% of $10,000 = $10.00.
    # Post-trade open equity = $9,990.00 -> buys 9990 / 110 = 90.818181818... shares at $110.
    # At Day 1 Close ($121): equity_close = 90.818181818 * 121 = $10,989.00.
    obs1, reward1, term1, trunc1, info1 = env.step(np.array([1.0], dtype=np.float32))
    assert not term1
    assert info1["equity_open_pre"] == pytest.approx(10_000.0, rel=1e-9)
    assert info1["cost"] == pytest.approx(10.0, rel=1e-9)
    assert info1["equity_open_post"] == pytest.approx(9_990.0, rel=1e-9)
    assert info1["cash"] == pytest.approx(0.0, abs=1e-9)
    assert info1["shares"] == pytest.approx(9_990.0 / 110.0, rel=1e-9)
    assert info1["equity"] == pytest.approx(10_989.0, rel=1e-9)
    assert info1["realized_weight"] == pytest.approx(1.0, rel=1e-9)
    assert reward1 == pytest.approx(np.log(10_989.0 / 10_000.0), rel=1e-9)

    # Step 1 (Day 1 Close -> Day 2 Open -> Day 2 Close):
    # Action at Day 1 Close: target_weight = 0.0 (liquidate all shares at Day 2 Open $133.1)
    # Overnight gap from Day 1 Close ($121) to Day 2 Open ($133.1) is +10%!
    # Pre-trade open equity at Day 2 Open = shares * 133.1 = 10,989.0 * 1.1 = $12,087.90.
    # Liquidating 100% stock to 0% incurs 0.1% cost on $12,087.90 = $12.0879.
    # Post-trade cash = 12,087.90 - 12.0879 = $12,075.8121, shares = 0.0.
    # At Day 2 Close ($120): since shares == 0, intraday drop from 133.1 to 120 has 0 impact!
    obs2, reward2, term2, trunc2, info2 = env.step(np.array([0.0], dtype=np.float32))
    assert term2
    expected_open_pre_2 = (9_990.0 / 110.0) * 133.1  # 12,087.90
    expected_cost_2 = expected_open_pre_2 * 0.001  # 12.0879
    expected_equity_2 = expected_open_pre_2 - expected_cost_2  # 12,075.8121

    assert info2["equity_open_pre"] == pytest.approx(expected_open_pre_2, rel=1e-9)
    assert info2["cost"] == pytest.approx(expected_cost_2, rel=1e-9)
    assert info2["shares"] == pytest.approx(0.0, abs=1e-9)
    assert info2["cash"] == pytest.approx(expected_equity_2, rel=1e-9)
    assert info2["equity"] == pytest.approx(expected_equity_2, rel=1e-9)
    assert info2["drawdown"] == pytest.approx(0.0, abs=1e-9)


def test_rebalancing_cost_non_negative_cash_and_baseline_equivalence() -> None:
    """
    1. Verifies one-way 0.1% transaction cost is deducted exactly once (no double deduction).
    2. Verifies cash never becomes negative even at 100% target weight (w=1.0).
    3. Verifies holding 100% stock across consecutive steps incurs 0.0 additional cost.
    4. Verifies TradingEnv on Validation split matches Phase 1 Baseline simulator exactly.
    """
    env = TradingEnv(mode="val", reward_type="A")
    obs, info = env.reset()

    # Step 1: Buy 100% stock -> cost = $10.00, cash >= 0, open post identity holds
    obs, _, _, _, info1 = env.step(np.array([1.0], dtype=np.float32))
    assert info1["cash"] >= 0.0
    assert info1["cost"] == pytest.approx(10.0, rel=1e-9)
    assert info1["equity_open_post"] == pytest.approx(
        info1["equity_open_pre"] - info1["cost"], rel=1e-10
    )

    # Step 2: Keep target_weight = 1.0 -> already 100% in stock at open, so turnover = 0 and cost = 0!
    obs, _, _, _, info2 = env.step(np.array([1.0], dtype=np.float32))
    assert info2["turnover_weight"] == pytest.approx(0.0, abs=1e-12)
    assert info2["cost"] == pytest.approx(0.0, abs=1e-12)
    assert info2["cash"] >= 0.0

    # Full trajectory comparison against Phase 1 `run_all_baselines` on Test split
    test_env = TradingEnv(mode="test", reward_type="A")
    summary_df, trajectories = run_all_baselines(
        split_df=test_env.df,
        split_name="Test",
        config=DEFAULT_CONFIG,
    )

    for strat_key, constant_w in [("1_Buy_and_Hold", 1.0), ("3_Fixed_50_50", 0.5)]:
        test_env.reset()
        done = False
        env_equities = [test_env.equity]
        env_costs = [0.0]
        while not done:
            _, _, terminated, truncated, step_info = test_env.step(
                np.array([constant_w], dtype=np.float32)
            )
            env_equities.append(step_info["equity"])
            env_costs.append(step_info["cost"])
            done = terminated or truncated

        traj = trajectories[strat_key]
        assert np.allclose(env_equities, traj["equity_close"].to_numpy(), rtol=1e-10, atol=1e-8)
        assert np.allclose(env_costs, traj["transaction_cost"].to_numpy(), rtol=1e-10, atol=1e-8)


def test_observation_vs_reward_variance_separation() -> None:
    """
    Verifies Section 4 requirement:
    - Observation uses transformed log-variance: Normalize(log(Variance_raw + 1e-8)),
      fixed at 0.0 for the first <30 steps.
    - Reward C uses raw non-negative variance (Variance_raw >= 0.0), with 0.0 penalty
      for the first <30 steps.
    - Ensures Reward C never uses the negative log-normalized observation variance.
    """
    lambda_d = 0.2
    lambda_v = 50.0
    env = TradingEnv(
        mode="val",
        reward_type="C",
        lambda_d=lambda_d,
        lambda_v=lambda_v,
    )
    obs, _ = env.reset()
    assert obs[-3] == pytest.approx(0.0)  # realized_weight
    assert obs[-2] == pytest.approx(0.0)  # drawdown
    assert obs[-1] == pytest.approx(0.0)  # initial obs_variance

    for step_num in range(1, 45):
        obs, reward_c, terminated, truncated, info = env.step(
            np.array([0.8], dtype=np.float32)
        )
        obs_weight = float(obs[-3])
        obs_dd = float(obs[-2])
        obs_var = float(obs[-1])

        raw_var = float(info["raw_variance"])
        dd = float(info["drawdown"])
        r_a = float(info["reward_a"])
        r_b = float(info["reward_b"])

        assert obs_weight == pytest.approx(info["realized_weight"], abs=1e-6)
        assert obs_dd == pytest.approx(dd, abs=1e-6)
        assert r_b == pytest.approx(r_a - lambda_d * dd, abs=1e-10)

        if step_num < 30:
            # Under 30 days: both raw_variance and obs_variance must be 0.0, and Reward C == Reward B
            assert raw_var == 0.0
            assert obs_var == 0.0
            assert reward_c == pytest.approx(r_b, abs=1e-12)
        else:
            # 30+ days: raw_variance > 0, while obs_var is log-normalized (distinct from raw_variance)
            assert raw_var > 0.0
            expected_obs_var = (
                np.log(raw_var + env.variance_eps) - env.log_var_mean
            ) / env.log_var_std
            assert obs_var == pytest.approx(expected_obs_var, rel=1e-6)
            assert not np.isclose(obs_var, raw_var)

            # Reward C must penalize using raw_var >= 0 (so reward_c < r_b)
            expected_r_c = r_b - (lambda_v * raw_var)
            assert reward_c == pytest.approx(expected_r_c, abs=1e-10)
            assert reward_c < r_b


def test_episode_management_train_vs_eval() -> None:
    """
    Verifies Section 3 Episode Protocol:
    - Train mode: Random start index within Train split and fixed 252 trading-day length.
    - Val/Test mode: Fixed start index (0) with $10,000 reset and full split traversal.
    """
    train_env = TradingEnv(mode="train", reward_type="A")
    start_steps = set()
    for seed in (11, 22, 33, 44, 55):
        train_env.reset(seed=seed)
        start_steps.add(train_env.start_step)
        # Ensure at least 29 prior market feature days exist for warm-up
        assert train_env.start_step >= 29

    # Random start should vary across seeds
    assert len(start_steps) > 1

    # Run a full Train episode and verify it lasts exactly 252 steps
    train_env.reset(seed=11)
    steps = 0
    done = False
    while not done:
        _, _, terminated, truncated, _ = train_env.step(np.array([0.5], dtype=np.float32))
        steps += 1
        done = terminated or truncated
    assert steps == 252

    # Run Test episode and verify deterministic start_step == 0 and full traversal
    test_env = TradingEnv(mode="test", reward_type="A")
    _, info_reset = test_env.reset(seed=99)
    assert test_env.start_step == 0
    assert info_reset["equity"] == 10_000.0
    assert info_reset["shares"] == 0.0
    assert info_reset["drawdown"] == 0.0
    assert test_env.max_episode_steps == len(test_env.df) - 1
