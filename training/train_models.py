"""
[Adaptive-QuantRL] Phase 3: Independent PPO / SAC Training & Track Separation Pipeline
Implements:
- Track 1 (Ensemble Track): Reward A with PPO & SAC across 5 Seeds (11, 22, 33, 44, 55)
- Track 2 (Ablation Track): Reward B & Reward C with PPO & SAC across 5 Seeds (11, 22, 33, 44, 55)
- Controlled Conditions: Fixed base hyperparameters, timesteps, Observation/Action spaces,
  and 0.1% one-way transaction cost.
- Saves model weights (`model.zip`) and normalization/training metadata (`metadata.json`) under:
    models/checkpoints/{algorithm}_{reward}_{seed}/
"""

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.utils import set_random_seed

from config import (
    CHECKPOINTS_DIR,
    DEFAULT_CONFIG,
    PHASE3_RESULTS_DIR,
    PROCESSED_DATA_DIR,
    ExperimentConfig,
)
from envs.trading_env import TradingEnv


ALGO_MAP = {
    "ppo": PPO,
    "sac": SAC,
}


def set_all_seeds(seed: int) -> None:
    """Sets random seeds across random, numpy, torch, and SB3 for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_random_seed(seed)


def get_checkpoint_dir(
    algorithm: str,
    reward_type: str,
    seed: int,
    checkpoints_root: Path = CHECKPOINTS_DIR,
) -> Path:
    algo_key = algorithm.strip().lower()
    rew_key = reward_type.strip().upper()
    return checkpoints_root / f"{algo_key}_{rew_key}_{int(seed)}"


def _load_scaler_stats(processed_dir: Path = PROCESSED_DATA_DIR) -> Dict[str, Any]:
    meta_path = processed_dir / "phase1_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Phase 1 metadata not found at {meta_path}. Run `python run_phase1.py` first."
        )
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta.get("scaler_stats", {})


def train_single_model(
    algorithm: str,
    reward_type: str,
    seed: int,
    track_name: str,
    total_timesteps: Optional[int] = None,
    lambda_d: Optional[float] = None,
    lambda_v: Optional[float] = None,
    config: ExperimentConfig = DEFAULT_CONFIG,
    checkpoints_root: Path = CHECKPOINTS_DIR,
    overwrite: bool = False,
    verbose: int = 0,
) -> Dict[str, Any]:
    """
    Trains a single PPO or SAC agent on `TradingEnv(mode='train')` under strictly
    controlled conditions and saves both `model.zip` and `metadata.json` to:
      models/checkpoints/{algorithm}_{reward}_{seed}/
    """
    algo_key = algorithm.strip().lower()
    if algo_key not in ALGO_MAP:
        raise ValueError(f"Unsupported algorithm '{algorithm}'. Expected 'ppo' or 'sac'.")

    rew_key = reward_type.strip().upper()
    if rew_key not in ("A", "B", "C"):
        raise ValueError(f"Unsupported reward_type '{reward_type}'. Expected 'A', 'B', or 'C'.")

    eff_timesteps = int(total_timesteps if total_timesteps is not None else config.total_timesteps)
    eff_lambda_d = float(lambda_d if lambda_d is not None else config.default_lambda_d)
    eff_lambda_v = float(lambda_v if lambda_v is not None else config.default_lambda_v)

    ckpt_dir = get_checkpoint_dir(
        algorithm=algo_key,
        reward_type=rew_key,
        seed=seed,
        checkpoints_root=checkpoints_root,
    )
    model_zip_path = ckpt_dir / "model.zip"
    meta_json_path = ckpt_dir / "metadata.json"

    if model_zip_path.exists() and meta_json_path.exists() and not overwrite:
        with open(meta_json_path, "r", encoding="utf-8") as f:
            existing_meta = json.load(f)
        if existing_meta.get("total_timesteps") == eff_timesteps:
            return existing_meta

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    set_all_seeds(seed)

    # Create controlled Train environment
    env = TradingEnv(
        mode="train",
        reward_type=rew_key,
        lambda_d=eff_lambda_d,
        lambda_v=eff_lambda_v,
        one_way_cost_rate=config.one_way_cost_rate,
        config=config,
    )
    env.reset(seed=seed)

    # Select controlled algorithm hyperparameters
    if algo_key == "ppo":
        hyperparams = dict(config.ppo_hyperparams)
        # Ensure n_steps is compatible with small test timesteps if overridden
        if eff_timesteps < hyperparams["n_steps"]:
            hyperparams["n_steps"] = max(64, eff_timesteps)
            hyperparams["batch_size"] = min(64, hyperparams["n_steps"])
        model: Union[PPO, SAC] = PPO(
            policy="MlpPolicy",
            env=env,
            seed=seed,
            verbose=verbose,
            device="cpu",
            **hyperparams,
        )
    else:
        hyperparams = dict(config.sac_hyperparams)
        if eff_timesteps <= hyperparams["learning_starts"]:
            hyperparams["learning_starts"] = max(16, eff_timesteps // 4)
        model = SAC(
            policy="MlpPolicy",
            env=env,
            seed=seed,
            verbose=verbose,
            device="cpu",
            **hyperparams,
        )

    t0 = time.time()
    model.learn(total_timesteps=eff_timesteps, progress_bar=False)
    elapsed_sec = round(time.time() - t0, 3)

    # Save SB3 model weights to models/checkpoints/{algorithm}_{reward}_{seed}/model.zip
    model.save(str(ckpt_dir / "model"))

    scaler_stats = _load_scaler_stats()

    metadata: Dict[str, Any] = {
        "checkpoint_id": f"{algo_key}_{rew_key}_{int(seed)}",
        "track": track_name,
        "algorithm": algo_key,
        "reward_type": rew_key,
        "seed": int(seed),
        "lambda_d": eff_lambda_d if rew_key in ("B", "C") else 0.0,
        "lambda_v": eff_lambda_v if rew_key == "C" else 0.0,
        "configured_lambda_d": eff_lambda_d,
        "configured_lambda_v": eff_lambda_v,
        "total_timesteps": eff_timesteps,
        "train_episode_length": int(config.train_episode_length),
        "one_way_cost_rate": float(config.one_way_cost_rate),
        "observation_shape": list(env.observation_space.shape),
        "action_shape": list(env.action_space.shape),
        "hyperparameters": hyperparams,
        "scaler_stats": {
            "train_start": scaler_stats.get("train_start"),
            "train_end": scaler_stats.get("train_end"),
            "feature_columns": scaler_stats.get("feature_columns"),
            "market_scaler_mean": scaler_stats.get("market_scaler_mean"),
            "market_scaler_std": scaler_stats.get("market_scaler_std"),
            "obs_log_variance_norm": scaler_stats.get("obs_log_variance_norm"),
        },
        "model_path": str(model_zip_path),
        "training_time_sec": elapsed_sec,
    }

    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata


def load_trained_model(
    algorithm: str,
    reward_type: str,
    seed: int,
    checkpoints_root: Path = CHECKPOINTS_DIR,
) -> Tuple[Union[PPO, SAC], Dict[str, Any]]:
    """
    Loads a saved PPO or SAC model and its metadata from
    `models/checkpoints/{algorithm}_{reward}_{seed}/`.
    """
    algo_key = algorithm.strip().lower()
    rew_key = reward_type.strip().upper()
    ckpt_dir = get_checkpoint_dir(
        algorithm=algo_key,
        reward_type=rew_key,
        seed=seed,
        checkpoints_root=checkpoints_root,
    )
    model_zip_path = ckpt_dir / "model.zip"
    meta_json_path = ckpt_dir / "metadata.json"

    if not model_zip_path.exists() or not meta_json_path.exists():
        raise FileNotFoundError(
            f"Checkpoint incomplete or missing at {ckpt_dir}"
        )

    with open(meta_json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    model_cls = ALGO_MAP[algo_key]
    model = model_cls.load(str(model_zip_path), device="cpu")
    return model, metadata


def train_all_phase3_models(
    total_timesteps: Optional[int] = None,
    seeds: Optional[Tuple[int, ...]] = None,
    lambda_d: Optional[float] = None,
    lambda_v: Optional[float] = None,
    config: ExperimentConfig = DEFAULT_CONFIG,
    checkpoints_root: Path = CHECKPOINTS_DIR,
    overwrite: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Executes both experimental tracks under controlled conditions:
    - Track 1 (Ensemble Track):
        Reward A x {PPO, SAC} x 5 Seeds (11, 22, 33, 44, 55) = 10 models
    - Track 2 (Reward Ablation Track):
        {Reward B, Reward C} x {PPO, SAC} x 5 Seeds (11, 22, 33, 44, 55) = 20 models
    Total = 30 independent models.
    """
    PHASE3_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoints_root.mkdir(parents=True, exist_ok=True)

    eff_seeds = tuple(seeds if seeds is not None else config.random_seeds)
    eff_timesteps = int(total_timesteps if total_timesteps is not None else config.total_timesteps)
    eff_lambda_d = float(lambda_d if lambda_d is not None else config.default_lambda_d)
    eff_lambda_v = float(lambda_v if lambda_v is not None else config.default_lambda_v)

    track_specs = [
        ("Track_1_Ensemble", ("A",)),
        ("Track_2_Reward_Ablation", ("B", "C")),
    ]
    algorithms = ("ppo", "sac")

    registry_rows: List[Dict[str, Any]] = []
    total_jobs = sum(len(rews) for _, rews in track_specs) * len(algorithms) * len(eff_seeds)
    job_idx = 0

    for track_name, reward_types in track_specs:
        if verbose:
            print(f"\n--- Executing {track_name} (Rewards: {reward_types}) ---")
        for rew in reward_types:
            for algo in algorithms:
                for seed in eff_seeds:
                    job_idx += 1
                    meta = train_single_model(
                        algorithm=algo,
                        reward_type=rew,
                        seed=seed,
                        track_name=track_name,
                        total_timesteps=eff_timesteps,
                        lambda_d=eff_lambda_d,
                        lambda_v=eff_lambda_v,
                        config=config,
                        checkpoints_root=checkpoints_root,
                        overwrite=overwrite,
                        verbose=0,
                    )
                    if verbose:
                        model_label = f"{meta['algorithm'].upper()}_Reward{meta['reward_type']}_seed{meta['seed']}"
                        print(
                            f"  [{job_idx}/{total_jobs}] {model_label} 완료 | "
                            f"Saved {meta['checkpoint_id']:10s} | "
                            f"Track: {track_name:23s} | Steps: {meta['total_timesteps']:6d} | "
                            f"lambda_d={meta['lambda_d']:.3f}, lambda_v={meta['lambda_v']:.2f} | "
                            f"Time: {meta['training_time_sec']:.2f}s",
                            flush=True,
                        )
                    registry_rows.append(
                        {
                            "checkpoint_id": meta["checkpoint_id"],
                            "track": meta["track"],
                            "algorithm": meta["algorithm"].upper(),
                            "reward_type": meta["reward_type"],
                            "seed": meta["seed"],
                            "lambda_d": meta["lambda_d"],
                            "lambda_v": meta["lambda_v"],
                            "total_timesteps": meta["total_timesteps"],
                            "one_way_cost_rate": meta["one_way_cost_rate"],
                            "training_time_sec": meta["training_time_sec"],
                            "model_path": meta["model_path"],
                        }
                    )

    registry_df = pd.DataFrame(registry_rows)
    registry_df.to_csv(PHASE3_RESULTS_DIR / "phase3_model_registry.csv", index=False)
    return registry_df


def verify_phase3_checkpoints(
    seeds: Optional[Tuple[int, ...]] = None,
    config: ExperimentConfig = DEFAULT_CONFIG,
    checkpoints_root: Path = CHECKPOINTS_DIR,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Verifies the integrity of all saved Phase 3 checkpoints:
    1. All expected checkpoint directories (`{algorithm}_{reward}_{seed}`) exist.
    2. Both `model.zip` and `metadata.json` exist and contain complete normalization stats.
    3. Controlled experimental conditions (cost rate, obs shape, action shape, timesteps) are identical.
    4. Every saved model loads cleanly and outputs valid deterministic actions in [0.0, 1.0].
    """
    eff_seeds = tuple(seeds if seeds is not None else config.random_seeds)
    algorithms = ("ppo", "sac")
    reward_types = ("A", "B", "C")

    eval_env = TradingEnv(mode="val", reward_type="A", config=config)
    obs_sample, _ = eval_env.reset(seed=11)

    verified_count = 0
    expected_count = len(algorithms) * len(reward_types) * len(eff_seeds)
    timesteps_set = set()
    cost_rates_set = set()
    obs_shapes_set = set()

    for algo in algorithms:
        for rew in reward_types:
            for seed in eff_seeds:
                model, meta = load_trained_model(
                    algorithm=algo,
                    reward_type=rew,
                    seed=seed,
                    checkpoints_root=checkpoints_root,
                )
                assert meta["algorithm"] == algo
                assert meta["reward_type"] == rew
                assert meta["seed"] == int(seed)

                # Verify track assignment
                expected_track = (
                    "Track_1_Ensemble" if rew == "A" else "Track_2_Reward_Ablation"
                )
                assert meta["track"] == expected_track

                # Verify normalization stats saved in metadata
                scaler_stats = meta.get("scaler_stats", {})
                assert "market_scaler_mean" in scaler_stats and scaler_stats["market_scaler_mean"]
                assert "market_scaler_std" in scaler_stats and scaler_stats["market_scaler_std"]
                assert "obs_log_variance_norm" in scaler_stats and scaler_stats["obs_log_variance_norm"]

                timesteps_set.add(int(meta["total_timesteps"]))
                cost_rates_set.add(float(meta["one_way_cost_rate"]))
                obs_shapes_set.add(tuple(meta["observation_shape"]))

                # Test deterministic inference action in [0.0, 1.0]
                action, _ = model.predict(obs_sample, deterministic=True)
                act_val = float(np.asarray(action).reshape(-1)[0])
                assert np.isfinite(act_val), f"Non-finite action from {meta['checkpoint_id']}"
                assert 0.0 - 1e-6 <= act_val <= 1.0 + 1e-6, (
                    f"Action {act_val} out of bounds [0, 1] for {meta['checkpoint_id']}"
                )

                verified_count += 1

    assert verified_count == expected_count
    assert len(timesteps_set) == 1, f"Inconsistent timesteps across models: {timesteps_set}"
    assert cost_rates_set == {float(config.one_way_cost_rate)}
    assert obs_shapes_set == {(eval_env.obs_dim,)}

    result = {
        "all_passed": True,
        "verified_checkpoints": verified_count,
        "expected_checkpoints": expected_count,
        "controlled_timesteps": list(timesteps_set)[0],
        "controlled_one_way_cost_rate": float(config.one_way_cost_rate),
        "controlled_observation_shape": list(eval_env.observation_space.shape),
        "scaler_metadata_persisted": True,
        "deterministic_inference_verified": True,
    }

    if verbose:
        print("\n[Phase 3 - Step 3] Checkpoint & Control Integrity Verification:")
        for k, v in result.items():
            print(f"  - {k:36s}: {v}")

    return result
