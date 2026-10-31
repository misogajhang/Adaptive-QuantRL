"""
[Adaptive-QuantRL] Phase 3 Unit Tests:
- Reward scale measurement & quantile table output
- Track-separated PPO & SAC checkpoint saving (`models/checkpoints/{algorithm}_{reward}_{seed}/`)
- Checkpoint loading & metadata integrity verification
"""

from pathlib import Path

import numpy as np
import pytest

from config import DEFAULT_CONFIG
from training.measure_reward_scale import measure_reward_scales
from training.train_models import (
    load_trained_model,
    train_all_phase3_models,
    verify_phase3_checkpoints,
)


def test_measure_reward_scales_output() -> None:
    df, report = measure_reward_scales(
        num_episodes_per_policy=2,
        config=DEFAULT_CONFIG,
        verbose=False,
    )
    assert not df.empty
    expected_policies = {"Buy_and_Hold", "Fixed_50_50", "Random_Policy"}
    assert set(df["Policy"].unique()) == expected_policies
    for col in ("Mean", "Std", "Min", "25%", "Median", "75%", "Max"):
        assert col in df.columns

    # Ensure penalty candidates do not overwhelm |log_return| (< 50% of mean |R_A|)
    for item in report["lambda_d_scale_ratios"]:
        assert 0.0 < item["ratio_to_mean_abs_log_ret(%)"] < 50.0
    for item in report["lambda_v_scale_ratios"]:
        assert 0.0 < item["ratio_to_mean_abs_log_ret(%)"] < 50.0


def test_train_and_verify_checkpoints_isolated(tmp_path: Path) -> None:
    """
    Tests end-to-end training, checkpoint saving (`model.zip` + `metadata.json`),
    and integrity verification on a single seed in an isolated temporary directory.
    """
    test_seeds = (11,)
    reg_df = train_all_phase3_models(
        total_timesteps=252,
        seeds=test_seeds,
        config=DEFAULT_CONFIG,
        checkpoints_root=tmp_path,
        overwrite=True,
        verbose=False,
    )
    # 2 algorithms (PPO, SAC) * 3 rewards (A, B, C) * 1 seed = 6 models
    assert len(reg_df) == 6

    res = verify_phase3_checkpoints(
        seeds=test_seeds,
        config=DEFAULT_CONFIG,
        checkpoints_root=tmp_path,
        verbose=False,
    )
    assert res["all_passed"] is True
    assert res["verified_checkpoints"] == 6

    # Check individual checkpoint files
    for algo in ("ppo", "sac"):
        for rew in ("A", "B", "C"):
            ckpt_dir = tmp_path / f"{algo}_{rew}_11"
            assert (ckpt_dir / "model.zip").exists()
            assert (ckpt_dir / "metadata.json").exists()
            model, meta = load_trained_model(
                algorithm=algo,
                reward_type=rew,
                seed=11,
                checkpoints_root=tmp_path,
            )
            assert meta["checkpoint_id"] == f"{algo}_{rew}_11"
            assert meta["one_way_cost_rate"] == pytest.approx(0.001)
