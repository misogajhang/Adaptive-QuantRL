"""
[Adaptive-QuantRL] Phase 4 ~ Phase 8 Master Evaluation & Reporting Runner
Executes sequentially:
  - Phase 4: Validation (2020~2022) Reward Ablation & Model Freezing
  - Phase 5: Validation ADX(14) Regime Classification & PPO/SAC Behavioral Analysis
  - Phase 6: Shared-Env 27 Dynamic Ensemble Optimization & Exposure-Matched Baseline Freezing
  - Phase 7: Deterministic Test (2023~2024) Evaluation & Transaction Cost Sensitivity (0.1%, 0.3%, 0.5%)
  - Phase 8: 4 Research Questions (RQ1 ~ RQ4) CSV & Markdown Report Generation
"""

from config import DEFAULT_CONFIG
from src.ensemble.dynamic_ensemble import (
    run_phase5_regime_behavior_analysis,
    run_phase6_ensemble_optimization,
)
from src.evaluation.run_test_evaluation import run_phase7_and_8_evaluation
from src.evaluation.validate_rewards import run_phase4_validation


def main() -> None:
    print("=" * 90)
    print("[Adaptive-QuantRL] Executing Phase 4 -> Phase 5 -> Phase 6 -> Phase 7 -> Phase 8")
    print("=" * 90)

    run_phase4_validation(config=DEFAULT_CONFIG, verbose=True)
    run_phase5_regime_behavior_analysis(config=DEFAULT_CONFIG, verbose=True)
    run_phase6_ensemble_optimization(config=DEFAULT_CONFIG, verbose=True)
    run_phase7_and_8_evaluation(config=DEFAULT_CONFIG, verbose=True)

    print("\n" + "=" * 90)
    print("[Adaptive-QuantRL] Phase 4 ~ Phase 8 Pipeline Complete!")
    print("  - Phase 4 Frozen Models     : results/phase4/phase4_frozen_models.json")
    print("  - Phase 5 Regime Behavior   : results/phase5_6/phase5_regime_behavior_val.csv")
    print("  - Phase 6 Frozen Ensemble   : results/phase5_6/phase6_frozen_ensemble.json")
    print("  - Phase 7 Test Summary      : results/phase7_8/phase7_test_summary_table.csv")
    print("  - Phase 7 Cost Sensitivity  : results/phase7_8/phase7_cost_sensitivity.csv")
    print("  - Phase 8 RQ Final Report   : results/phase7_8/phase8_rq_verification_report.md")
    print("=" * 90)


if __name__ == "__main__":
    main()
