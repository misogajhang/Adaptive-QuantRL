"""
[Adaptive-QuantRL] Phase 3 Runner:
1. Pre-Training Reward Scale Measurement (Buy & Hold, Fixed 50:50, Random Policy)
2. Independent PPO / SAC Training across Track 1 (Ensemble: Reward A) & Track 2 (Ablation: Reward B/C)
   for 5 Random Seeds (11, 22, 33, 44, 55)
3. Checkpoint Storage & Integrity Verification
"""

import argparse

from config import DEFAULT_CONFIG
from training.measure_reward_scale import measure_reward_scales
from training.train_models import train_all_phase3_models, verify_phase3_checkpoints


def main(
    total_timesteps: int | None = None,
    overwrite: bool = False,
    num_scale_episodes: int = 10,
) -> None:
    print("=" * 90)
    print("[Adaptive-QuantRL] Phase 3: Reward Scale Calibration & Track-Separated RL Training")
    print("=" * 90)

    # Step 1: Measure reward component scales & quantiles in Train TradingEnv
    scale_df, scale_report = measure_reward_scales(
        num_episodes_per_policy=num_scale_episodes,
        config=DEFAULT_CONFIG,
        verbose=True,
    )

    # Step 2: Train Track 1 (Ensemble: Reward A) & Track 2 (Ablation: Reward B & C)
    eff_timesteps = (
        total_timesteps if total_timesteps is not None else DEFAULT_CONFIG.total_timesteps
    )
    print(
        f"\n[Phase 3 - Step 2] Training PPO & SAC Models "
        f"(Seeds={DEFAULT_CONFIG.random_seeds}, Timesteps={eff_timesteps}):"
    )
    registry_df = train_all_phase3_models(
        total_timesteps=eff_timesteps,
        seeds=DEFAULT_CONFIG.random_seeds,
        lambda_d=scale_report["selected_default_lambda_d"],
        lambda_v=scale_report["selected_default_lambda_v"],
        config=DEFAULT_CONFIG,
        overwrite=overwrite,
        verbose=True,
    )

    # Step 3: Verify all saved checkpoints and normalization metadata
    verification = verify_phase3_checkpoints(
        seeds=DEFAULT_CONFIG.random_seeds,
        config=DEFAULT_CONFIG,
        verbose=True,
    )

    print("\n[Phase 3 Summary]:")
    print(f"  - Total Models Trained & Verified : {len(registry_df)} / 30")
    print(f"  - Scale Quantiles Saved           : results/phase3/reward_scale_quantiles.csv")
    print(f"  - Model Registry Saved            : results/phase3/phase3_model_registry.csv")
    print(f"  - Checkpoints Root                : models/checkpoints/{{algorithm}}_{{reward}}_{{seed}}/")
    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Adaptive-QuantRL Phase 3 Pipeline")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total training timesteps per model (default from config.py)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force re-training even if checkpoint already exists",
    )
    parser.add_argument(
        "--scale-episodes",
        type=int,
        default=10,
        help="Number of 252-day Train episodes per policy for reward scale measurement",
    )
    args = parser.parse_args()
    main(
        total_timesteps=args.timesteps,
        overwrite=args.overwrite,
        num_scale_episodes=args.scale_episodes,
    )
