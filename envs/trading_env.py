"""
[Adaptive-QuantRL] Phase 2: Production Gymnasium TradingEnv & Modular Reward Functions
Strictly implements Section 3, Section 4, and Phase 2 of 초안.md:
- Action Space: Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
- Observation Space: Box(low=-inf, high=inf, shape=((30 * num_market_features) + 3,), dtype=np.float32)
  * Market: 30-day standardized market features (Train StandardScaler applied)
  * Portfolio: [realized_weight, drawdown, variance_obs] (float32)
  * Variance_obs: Normalize(log(Variance_raw + 1e-8)), fixed to 0.0 for <30 days
- Accounting Alignment:
  * Day t close obs/action -> Day t+1 open overnight gap -> Open rebalancing (fractional shares,
    one-way 0.1% cost deducted, non-negative cash) -> Day t+1 close evaluation & EoD MDD update
- Modular Reward (A / B / C):
  * Reward A: ln(Equity_{t+1}^{close} / Equity_t^{close})
  * Reward B: Reward A - (lambda_d * Drawdown)
  * Reward C: Reward B - (lambda_v * Variance_{30}^{raw}) (raw variance >= 0; 0 penalty for <30 days)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from baseline_models import execute_portfolio_step
from config import DEFAULT_CONFIG, PROCESSED_DATA_DIR, ExperimentConfig


def compute_reward(
    equity_prev_close: float,
    equity_curr_close: float,
    drawdown: float,
    raw_variance_30d: float,
    reward_type: str = "A",
    lambda_d: float = 0.1,
    lambda_v: float = 10.0,
    num_return_steps: int = 30,
    variance_window: int = DEFAULT_CONFIG.variance_window,
) -> Tuple[float, Dict[str, float]]:
    """
    Computes modular Reward A, B, or C according to Section 4 of 초안.md:
      R_{A,t} = ln(Equity_{t+1}^{close} / Equity_t^{close})
      R_{B,t} = R_{A,t} - (lambda_d * Drawdown_t)
      R_{C,t} = R_{B,t} - (lambda_v * Variance^{raw}_{30,t})

    Note:
    - Uses non-negative raw variance (Variance^{raw}_{30,t} >= 0) for reward penalty,
      strictly separated from the log-normalized observation variance.
    - When `num_return_steps < variance_window` (< 30 days), raw variance and its
      penalty are strictly 0.0.
    """
    if equity_prev_close <= 0.0 or equity_curr_close <= 0.0:
        raise ValueError(
            f"Equities must be positive: prev={equity_prev_close}, curr={equity_curr_close}"
        )

    dd = float(max(0.0, drawdown))
    if num_return_steps < variance_window:
        raw_var = 0.0
    else:
        raw_var = float(max(0.0, raw_variance_30d))

    r_a = float(np.log(equity_curr_close / equity_prev_close))
    dd_penalty = float(lambda_d * dd)
    var_penalty = float(lambda_v * raw_var)

    r_b = r_a - dd_penalty
    r_c = r_b - var_penalty

    rtype = reward_type.strip().upper()
    if rtype in ("A", "REWARD_A"):
        selected = r_a
    elif rtype in ("B", "REWARD_B"):
        selected = r_b
    elif rtype in ("C", "REWARD_C"):
        selected = r_c
    else:
        raise ValueError(f"Unsupported reward_type: {reward_type}. Expected 'A', 'B', or 'C'.")

    breakdown = {
        "reward_a": r_a,
        "reward_b": r_b,
        "reward_c": r_c,
        "dd_penalty": dd_penalty,
        "var_penalty": var_penalty,
        "effective_raw_variance": raw_var,
    }
    return float(selected), breakdown


class RewardCalculator:
    """
    Stateful/configurable wrapper around `compute_reward` for Reward A/B/C ablation.
    """

    def __init__(
        self,
        reward_type: str = "A",
        lambda_d: float = 0.1,
        lambda_v: float = 10.0,
        variance_window: int = DEFAULT_CONFIG.variance_window,
    ) -> None:
        self.reward_type = reward_type.strip().upper()
        if self.reward_type not in ("A", "B", "C", "REWARD_A", "REWARD_B", "REWARD_C"):
            raise ValueError(f"Invalid reward_type: {reward_type}")
        self.lambda_d = float(lambda_d)
        self.lambda_v = float(lambda_v)
        self.variance_window = int(variance_window)

    def __call__(
        self,
        equity_prev_close: float,
        equity_curr_close: float,
        drawdown: float,
        raw_variance_30d: float,
        num_return_steps: int,
    ) -> Tuple[float, Dict[str, float]]:
        return compute_reward(
            equity_prev_close=equity_prev_close,
            equity_curr_close=equity_curr_close,
            drawdown=drawdown,
            raw_variance_30d=raw_variance_30d,
            reward_type=self.reward_type,
            lambda_d=self.lambda_d,
            lambda_v=self.lambda_v,
            num_return_steps=num_return_steps,
            variance_window=self.variance_window,
        )


def _load_phase1_split_and_metadata(
    mode: str,
    processed_dir: Path = PROCESSED_DATA_DIR,
    window_size: int = 30,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Loads the Phase 1 processed split CSV (`train`, `val`, or `test`),
    the preceding warm-up rows (for `val` and `test`), and `phase1_metadata.json`.
    """
    meta_path = processed_dir / "phase1_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found at {meta_path}. Run `python run_phase1.py` first."
        )

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    mode_key = mode.strip().lower()
    train_path = processed_dir / "spy_train_2010_2019.csv"
    val_path = processed_dir / "spy_val_2020_2022.csv"
    test_path = processed_dir / "spy_test_2023_2024.csv"

    if mode_key == "train":
        df = pd.read_csv(train_path, index_col=0, parse_dates=True)
        warmup_df = None
    elif mode_key in ("val", "validation"):
        df = pd.read_csv(val_path, index_col=0, parse_dates=True)
        prev_df = pd.read_csv(train_path, index_col=0, parse_dates=True)
        warmup_df = prev_df.iloc[-(window_size - 1) :].copy() if window_size > 1 else None
    elif mode_key == "test":
        df = pd.read_csv(test_path, index_col=0, parse_dates=True)
        prev_df = pd.read_csv(val_path, index_col=0, parse_dates=True)
        warmup_df = prev_df.iloc[-(window_size - 1) :].copy() if window_size > 1 else None
    else:
        raise ValueError(f"Unknown mode '{mode}'. Expected 'train', 'val', or 'test'.")

    return df, warmup_df, metadata


class TradingEnv(gym.Env):
    """
    Gymnasium-compatible Custom Trading Environment for Adaptive-QuantRL.

    - Action Space:
        Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        Represents target stock weight w_t in [0, 1] decided at day t close.
    - Observation Space:
        Box(low=-inf, high=inf, shape=((30 * num_market_features) + 3,), dtype=np.float32)
        Concatenation of:
          1. Flattened 30-day standardized market features up to day t close: (30 * F,)
          2. Portfolio state at day t close: [realized_weight, drawdown, variance_obs] (3,)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        mode: str = "train",
        reward_type: str = "A",
        lambda_d: float = 0.1,
        lambda_v: float = 10.0,
        window_size: int = 30,
        episode_length: int = DEFAULT_CONFIG.train_episode_length,
        initial_cash: float = DEFAULT_CONFIG.initial_cash,
        one_way_cost_rate: float = DEFAULT_CONFIG.one_way_cost_rate,
        feature_columns: Optional[List[str]] = None,
        scaler_metadata: Optional[Dict[str, Any]] = None,
        warmup_df: Optional[pd.DataFrame] = None,
        random_start: Optional[bool] = None,
        config: ExperimentConfig = DEFAULT_CONFIG,
    ) -> None:
        super().__init__()

        self.config = config
        self.mode = mode.strip().lower()
        if self.mode == "validation":
            self.mode = "val"
        if self.mode not in ("train", "val", "test"):
            raise ValueError(f"mode must be 'train', 'val', or 'test', got '{mode}'")

        self.window_size = int(window_size)
        self.episode_length = int(episode_length)
        self.initial_cash = float(initial_cash)
        self.one_way_cost_rate = float(one_way_cost_rate)
        self.variance_window = int(config.variance_window)
        self.variance_eps = float(config.variance_eps)
        self.initial_obs_variance = float(config.initial_obs_variance)

        # Load Phase 1 data & metadata if df not explicitly supplied
        if df is None:
            loaded_df, loaded_warmup, loaded_meta = _load_phase1_split_and_metadata(
                mode=self.mode,
                window_size=self.window_size,
            )
            self.df = loaded_df.copy()
            if warmup_df is None:
                warmup_df = loaded_warmup
            if scaler_metadata is None:
                scaler_metadata = loaded_meta
        else:
            self.df = df.copy()
            if scaler_metadata is None:
                meta_path = PROCESSED_DATA_DIR / "phase1_metadata.json"
                if meta_path.exists():
                    with open(meta_path, "r", encoding="utf-8") as f:
                        scaler_metadata = json.load(f)

        # Determine feature columns
        if feature_columns is not None:
            self.feature_columns = list(feature_columns)
        elif (
            scaler_metadata is not None
            and "scaler_stats" in scaler_metadata
            and "feature_columns" in scaler_metadata["scaler_stats"]
        ):
            self.feature_columns = list(
                scaler_metadata["scaler_stats"]["feature_columns"]
            )
        else:
            self.feature_columns = list(config.market_feature_columns)

        missing_cols = [
            c for c in (["Open", "Close"] + self.feature_columns) if c not in self.df.columns
        ]
        if missing_cols:
            raise ValueError(f"DataFrame is missing required columns: {missing_cols}")

        if len(self.df) < 2:
            raise ValueError("DataFrame must contain at least 2 rows for step transitions.")

        # Extract log-variance normalization stats from Train metadata
        if (
            scaler_metadata is not None
            and "scaler_stats" in scaler_metadata
            and "obs_log_variance_norm" in scaler_metadata["scaler_stats"]
        ):
            obs_norm = scaler_metadata["scaler_stats"]["obs_log_variance_norm"]
            self.log_var_mean = float(obs_norm["mean"])
            self.log_var_std = float(obs_norm["std"])
            self.variance_eps = float(obs_norm.get("eps", self.variance_eps))
        else:
            self.log_var_mean = -9.5
            self.log_var_std = 1.0

        if self.log_var_std == 0.0:
            self.log_var_std = 1.0

        # Pre-extract numpy arrays for fast stepping
        self.open_prices = self.df["Open"].to_numpy(dtype=np.float64)
        self.close_prices = self.df["Close"].to_numpy(dtype=np.float64)
        self.feature_matrix = self.df[self.feature_columns].to_numpy(dtype=np.float32)

        if warmup_df is not None and len(warmup_df) > 0:
            self.warmup_features = warmup_df[self.feature_columns].to_numpy(
                dtype=np.float32
            )
        else:
            self.warmup_features = None

        self.num_market_features = len(self.feature_columns)
        self.obs_dim = (self.window_size * self.num_market_features) + 3

        # Whether episodes start at a random index (default True for train, False for val/test)
        if random_start is None:
            self.random_start = self.mode == "train"
        else:
            self.random_start = bool(random_start)

        # Reward calculator
        self.reward_calculator = RewardCalculator(
            reward_type=reward_type,
            lambda_d=lambda_d,
            lambda_v=lambda_v,
            variance_window=self.variance_window,
        )

        # Spaces
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        # Internal portfolio state variables (initialized in reset)
        self.start_step: int = 0
        self.current_step: int = 0
        self.episode_steps: int = 0
        self.max_episode_steps: int = 0

        self.cash: float = self.initial_cash
        self.shares: float = 0.0
        self.equity: float = self.initial_cash
        self.peak_equity: float = self.initial_cash
        self.drawdown: float = 0.0
        self.realized_weight: float = 0.0
        self.last_cost: float = 0.0
        self.raw_variance: float = 0.0
        self.obs_variance: float = self.initial_obs_variance
        self.port_ret_history: List[float] = []

    def convert_raw_to_obs_variance(
        self, raw_variance: float, num_return_steps: int
    ) -> float:
        """
        Converts raw 30-day portfolio variance to normalized log-variance for Observation:
          Variance^{obs}_{30,t} = Normalize(log(Variance^{raw}_{30,t} + 1e-8))
        For the initial <30 days (`num_return_steps < variance_window`), returns 0.0 fixed.
        """
        if num_return_steps < self.variance_window:
            return float(self.initial_obs_variance)
        clipped_raw = max(0.0, float(raw_variance))
        log_val = float(np.log(clipped_raw + self.variance_eps))
        return float((log_val - self.log_var_mean) / self.log_var_std)

    def _get_market_window(self, step_idx: int) -> np.ndarray:
        """
        Returns a (window_size, num_market_features) float32 array of standardized
        market features ending at `step_idx` (inclusive), using only historical data <= step_idx.
        """
        if step_idx >= self.window_size - 1:
            start_i = step_idx - (self.window_size - 1)
            window = self.feature_matrix[start_i : step_idx + 1]
            return window.astype(np.float32, copy=False)

        # Need `missing` rows prior to index 0
        available = self.feature_matrix[0 : step_idx + 1]
        missing = self.window_size - len(available)

        if self.warmup_features is not None and len(self.warmup_features) >= missing:
            prefix = self.warmup_features[-missing:]
        elif self.warmup_features is not None and len(self.warmup_features) > 0:
            rem = missing - len(self.warmup_features)
            pad = np.repeat(self.warmup_features[0:1], rem, axis=0)
            prefix = np.concatenate([pad, self.warmup_features], axis=0)
        else:
            prefix = np.repeat(self.feature_matrix[0:1], missing, axis=0)

        window = np.concatenate([prefix, available], axis=0)
        return window.astype(np.float32, copy=False)

    def _get_obs(self) -> np.ndarray:
        """
        Constructs the 1D float32 observation vector:
          [flattened_30d_market_features, realized_weight, drawdown, obs_variance]
        """
        mkt_window = self._get_market_window(self.current_step).reshape(-1)
        portfolio_features = np.array(
            [
                float(self.realized_weight),
                float(self.drawdown),
                float(self.obs_variance),
            ],
            dtype=np.float32,
        )
        obs = np.concatenate([mkt_window, portfolio_features], axis=0).astype(
            np.float32, copy=False
        )
        return obs

    def _get_info(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "step": int(self.current_step),
            "episode_step": int(self.episode_steps),
            "date": str(self.df.index[self.current_step]),
            "equity": float(self.equity),
            "cash": float(self.cash),
            "shares": float(self.shares),
            "realized_weight": float(self.realized_weight),
            "drawdown": float(self.drawdown),
            "peak_equity": float(self.peak_equity),
            "cost": float(self.last_cost),
            "raw_variance": float(self.raw_variance),
            "obs_variance": float(self.obs_variance),
        }
        if extra:
            info.update(extra)
        return info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        n_rows = len(self.df)

        if options is not None and "start_step" in options:
            self.start_step = int(options["start_step"])
            if self.mode == "train":
                self.max_episode_steps = min(
                    self.episode_length, (n_rows - 1) - self.start_step
                )
            else:
                self.max_episode_steps = (n_rows - 1) - self.start_step
        elif self.random_start and self.mode == "train":
            # Ensure 252 trading steps fit within the DataFrame and prefer full 30-day warm-up
            min_start = (
                self.window_size - 1
                if (self.warmup_features is None and n_rows > self.episode_length + self.window_size)
                else 0
            )
            max_start = max(min_start, (n_rows - 1) - self.episode_length)
            if max_start > min_start:
                self.start_step = int(
                    self.np_random.integers(low=min_start, high=max_start + 1)
                )
            else:
                self.start_step = int(min_start)
            self.max_episode_steps = min(
                self.episode_length, (n_rows - 1) - self.start_step
            )
        else:
            # Validation / Test or deterministic start: start at index 0 and traverse entire split
            self.start_step = 0
            self.max_episode_steps = (
                min(self.episode_length, n_rows - 1)
                if self.mode == "train"
                else (n_rows - 1)
            )

        self.current_step = self.start_step
        self.episode_steps = 0

        # Section 3 initialization protocol:
        # Cash = $10,000, Shares = 0, Drawdown = 0, Peak Equity = $10,000, return history = []
        self.cash = float(self.initial_cash)
        self.shares = float(self.config.initial_shares)
        self.equity = float(self.initial_cash)
        self.peak_equity = float(self.initial_cash)
        self.drawdown = float(self.config.initial_drawdown)
        self.realized_weight = 0.0
        self.last_cost = 0.0
        self.raw_variance = 0.0
        self.obs_variance = float(self.initial_obs_variance)
        self.port_ret_history = []

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: Union[np.ndarray, float, int, List[float]]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.current_step >= len(self.df) - 1:
            raise RuntimeError("Cannot call step() on a terminated episode. Call reset() first.")

        # Parse and clip action to [0.0, 1.0]
        arr_action = np.asarray(action, dtype=np.float64).reshape(-1)
        if arr_action.size == 0 or np.isnan(arr_action[0]):
            raise ValueError(f"Invalid action received: {action}")
        target_weight = float(np.clip(arr_action[0], 0.0, 1.0))

        close_t = float(self.close_prices[self.current_step])
        open_t1 = float(self.open_prices[self.current_step + 1])
        close_t1 = float(self.close_prices[self.current_step + 1])

        # Execute exact accounting step: Day t close -> Day t+1 open -> Day t+1 close
        step_res = execute_portfolio_step(
            cash_t=self.cash,
            shares_t=self.shares,
            close_t=close_t,
            open_t1=open_t1,
            close_t1=close_t1,
            target_weight=target_weight,
            peak_equity_t=self.peak_equity,
            one_way_cost_rate=self.one_way_cost_rate,
        )

        equity_prev_close = step_res["equity_t_close"]
        self.cash = step_res["cash_t1"]
        self.shares = step_res["shares_t1"]
        self.equity = step_res["equity_t1_close"]
        self.peak_equity = step_res["peak_equity_t1"]
        self.drawdown = step_res["drawdown_t1"]
        self.realized_weight = step_res["weight_t1_close"]
        self.last_cost = step_res["transaction_cost"]

        # Update portfolio daily return history & 30-day variance (raw vs obs separated)
        self.port_ret_history.append(step_res["port_ret_simple"])
        num_ret_steps = len(self.port_ret_history)

        if num_ret_steps < self.variance_window:
            self.raw_variance = 0.0
            self.obs_variance = float(self.initial_obs_variance)
        else:
            recent_30 = self.port_ret_history[-self.variance_window :]
            self.raw_variance = float(max(0.0, np.var(recent_30, ddof=1)))
            self.obs_variance = self.convert_raw_to_obs_variance(
                raw_variance=self.raw_variance,
                num_return_steps=num_ret_steps,
            )

        # Compute modular reward (A / B / C) using raw_variance (NOT obs_variance)
        reward, reward_breakdown = self.reward_calculator(
            equity_prev_close=equity_prev_close,
            equity_curr_close=self.equity,
            drawdown=self.drawdown,
            raw_variance_30d=self.raw_variance,
            num_return_steps=num_ret_steps,
        )

        # Advance time pointers
        self.current_step += 1
        self.episode_steps += 1

        terminated = bool(
            (self.episode_steps >= self.max_episode_steps)
            or (self.current_step >= len(self.df) - 1)
        )
        truncated = False

        obs = self._get_obs()
        info = self._get_info(
            extra={
                "target_weight": target_weight,
                "equity_open_pre": step_res["equity_t1_open_pre"],
                "equity_open_post": step_res["equity_t1_open_post"],
                "turnover_weight": step_res["turnover_weight"],
                "port_ret_simple": step_res["port_ret_simple"],
                "port_ret_log": step_res["port_ret_log"],
                **reward_breakdown,
            }
        )

        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        print(
            f"[Step {self.current_step} | Date {self.df.index[self.current_step].date()}] "
            f"Equity: ${self.equity:,.2f} | Cash: ${self.cash:,.2f} | "
            f"Shares: {self.shares:.4f} | Weight: {self.realized_weight:.2%} | "
            f"MDD: {self.drawdown:.2%} | Cost: ${self.last_cost:.2f}"
        )
