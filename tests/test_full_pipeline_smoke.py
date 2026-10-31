"""
[Adaptive-QuantRL] End-to-End (E2E) Pipeline Smoke Test (Phase 3 -> Phase 4 -> Phase 5/6 -> Phase 7/8)
Verifies that with 1 Seed (seed=11) and ultra-short dummy training (<= 1,000 steps),
the entire pipeline from model training -> Validation reward ablation -> ADX regime analysis &
27-rule shared-env ensemble optimization -> Test evaluation, cost sensitivity (0.1%, 0.3%, 0.5%),
and 4 RQ CSV/Markdown report generation completes without error in under 10 seconds.
"""

import time
from pathlib import Path

import pytest

from config import DEFAULT_CONFIG
from src.ensemble.dynamic_ensemble import (
    run_phase5_regime_behavior_analysis,
    run_phase6_ensemble_optimization,
)
from src.evaluation.run_test_evaluation import run_phase7_and_8_evaluation
from src.evaluation.validate_rewards import run_phase4_validation
from training.train_models import train_all_phase3_models, verify_phase3_checkpoints


def test_full_pipeline_e2e_smoke(tmp_path: Path) -> None:
    t0 = time.perf_counter()

    smoke_seeds = (11,)
    smoke_timesteps = 128  # <= 1,000 steps ultra-short dummy training

    ckpt_dir = tmp_path / "checkpoints"
    p4_dir = tmp_path / "results_phase4"
    p56_dir = tmp_path / "results_phase5_6"
    p78_dir = tmp_path / "results_phase7_8"

    # 1. Phase 3: Train 6 dummy models (PPO/SAC x Reward A/B/C for seed=11) & verify checkpoints
    reg_df = train_all_phase3_models(
        total_timesteps=smoke_timesteps,
        seeds=smoke_seeds,
        config=DEFAULT_CONFIG,
        checkpoints_root=ckpt_dir,
        overwrite=True,
        verbose=False,
    )
    assert len(reg_df) == 6

    ckpt_check = verify_phase3_checkpoints(
        seeds=smoke_seeds,
        config=DEFAULT_CONFIG,
        checkpoints_root=ckpt_dir,
        verbose=False,
    )
    assert ckpt_check["all_passed"] is True

    # 2. Phase 4: Validation evaluation & 3-tier ranking / model freezing
    p4_res = run_phase4_validation(
        seeds=smoke_seeds,
        checkpoints_root=ckpt_dir,
        output_dir=p4_dir,
        config=DEFAULT_CONFIG,
        verbose=False,
    )
    assert (p4_dir / "phase4_val_per_seed_metrics.csv").exists()
    assert (p4_dir / "phase4_val_aggregated_ranking.csv").exists()
    assert (p4_dir / "phase4_frozen_models.json").exists()
    assert len(p4_res["ranked_df"]) == 6

    # 3. Phase 5 & 6: ADX(14) regime analysis & 27-rule Dynamic Ensemble optimization in shared Val Env
    p5_df = run_phase5_regime_behavior_analysis(
        seeds=smoke_seeds,
        checkpoints_root=ckpt_dir,
        output_dir=p56_dir,
        config=DEFAULT_CONFIG,
        verbose=False,
    )
    assert not p5_df.empty
    assert (p56_dir / "phase5_regime_behavior_val.csv").exists()

    p6_res = run_phase6_ensemble_optimization(
        seeds=smoke_seeds,
        checkpoints_root=ckpt_dir,
        output_dir=p56_dir,
        config=DEFAULT_CONFIG,
        verbose=False,
    )
    assert len(p6_res["ranked_27_df"]) == 27
    assert (p56_dir / "phase6_27rules_ranking_val.csv").exists()
    assert (p56_dir / "phase6_frozen_ensemble.json").exists()
    assert "11" in p6_res["frozen_ensemble_payload"]["exposure_matched_baseline"]["seed_weights"]

    # 4. Phase 7 & 8: Test evaluation, cost sensitivity (0.1%, 0.3%, 0.5%) & 4 RQ Report
    p78_res = run_phase7_and_8_evaluation(
        seeds=smoke_seeds,
        checkpoints_root=ckpt_dir,
        phase4_dir=p4_dir,
        phase6_dir=p56_dir,
        output_dir=p78_dir,
        config=DEFAULT_CONFIG,
        verbose=False,
    )
    assert (p78_dir / "phase7_test_summary_table.csv").exists()
    assert (p78_dir / "phase7_cost_sensitivity.csv").exists()
    assert (p78_dir / "phase8_test_regime_sample_counts.csv").exists()
    assert (p78_dir / "phase8_rq_verification_report.md").exists()

    report_text = (p78_dir / "phase8_rq_verification_report.md").read_text(encoding="utf-8")
    assert "RQ 1" in report_text
    assert "RQ 2" in report_text
    assert "RQ 3" in report_text
    assert "RQ 4" in report_text
    assert "Exposure_Matched_Baseline" in report_text

    elapsed = time.perf_counter() - t0
    print(f"\n[Smoke Test Completed in {elapsed:.2f}s]")
    assert elapsed < 10.0, f"Smoke test took {elapsed:.2f}s, expected < 10.0s."
