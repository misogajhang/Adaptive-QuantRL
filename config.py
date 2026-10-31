"""
[Adaptive-QuantRL] Central Configuration for Phase 1 ~ Phase 8
Strictly follows the Project Specification in 초안.md.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase1"
PHASE3_RESULTS_DIR = PROJECT_ROOT / "results" / "phase3"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"


@dataclass(frozen=True)
class ExperimentConfig:
    # 1. Asset & Data Fetching (Unadjusted OHLC, Dividend Excluded)
    ticker: str = "SPY"
    auto_adjust: bool = False  # Explicitly exclude dividend cashflows / adjustments
    warmup_start_date: str = "2009-01-01"  # Warm-up buffer prior to 2010 Train start

    # 2. Chronological Holdout Split (Section 3)
    train_start: str = "2010-01-01"
    train_end: str = "2019-12-31"
    val_start: str = "2020-01-01"
    val_end: str = "2022-12-31"
    test_start: str = "2023-01-01"
    test_end: str = "2024-12-31"

    # 3. Portfolio & Accounting Protocol (Section 3 & 4)
    initial_cash: float = 10_000.0
    initial_shares: float = 0.0
    initial_drawdown: float = 0.0
    initial_peak_equity: float = 10_000.0
    train_episode_length: int = 252  # 252 trading days fixed for Train episodes

    # Transaction cost: one-way 0.1% (commission 0.05% + slippage 0.05%)
    commission_rate: float = 0.0005
    slippage_rate: float = 0.0005
    one_way_cost_rate: float = 0.0010  # Total one-way cost = 0.1%

    # 4. Feature & Variance Parameters (Section 4 & Phase 5)
    variance_window: int = 30
    variance_eps: float = 1e-8
    initial_obs_variance: float = 0.0  # Pre-defined fixed observation value for <30 days
    adx_window: int = 14
    adx_threshold_candidates: Tuple[float, ...] = (20.0, 25.0, 30.0)

    # Moving Average Baseline Parameters
    ma_fast_window: int = 20
    ma_slow_window: int = 50
    ma_long_window: int = 200

    # 5. Phase 3 ~ 7 Controlled RL Training & Reward Ablation Parameters
    random_seeds: Tuple[int, ...] = (11, 22, 33, 44, 55)
    total_timesteps: int = 10_080  # 40 episodes * 252 trading days (configurable)

    # Calibrated penalty coefficient candidates & representative defaults (Phase 3 & 4)
    # Scaled so penalty terms remain ~5% to ~35% of typical |daily log return| (~0.0055)
    lambda_d_candidates: Tuple[float, ...] = (0.01, 0.02, 0.05)
    lambda_v_candidates: Tuple[float, ...] = (5.0, 10.0, 25.0)
    default_lambda_d: float = 0.02
    default_lambda_v: float = 10.0

    # Controlled PPO Hyperparameters
    ppo_hyperparams: Dict[str, Any] = field(
        default_factory=lambda: {
            "learning_rate": 3e-4,
            "n_steps": 252,
            "batch_size": 63,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.005,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "policy_kwargs": {"net_arch": [128, 128]},
        }
    )

    # Controlled SAC Hyperparameters
    sac_hyperparams: Dict[str, Any] = field(
        default_factory=lambda: {
            "learning_rate": 3e-4,
            "buffer_size": 50_000,
            "learning_starts": 252,
            "batch_size": 64,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
            "ent_coef": "auto",
            "policy_kwargs": {"net_arch": [128, 128]},
        }
    )

    # Annualization factor
    trading_days_per_year: int = 252
    risk_free_rate: float = 0.0

    # Market features to be standardized using Train-only StandardScaler
    market_feature_columns: List[str] = field(
        default_factory=lambda: [
            "ret_1d",
            "log_ret_1d",
            "ret_5d",
            "ret_20d",
            "close_to_sma20",
            "close_to_sma50",
            "close_to_sma200",
            "vol_20d",
            "rsi_14",
            "macd_norm",
            "macd_signal_norm",
            "macd_hist_norm",
            "atr_14_norm",
            "adx_14_scaled",
            "plus_di_14",
            "minus_di_14",
        ]
    )


DEFAULT_CONFIG = ExperimentConfig()
