#!/usr/bin/env python3
"""Hyperparameter tuning on validation set."""

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.data.split import load_splits
from src.env import PortfolioEnv
from src.eval.metrics import compute_metrics
from src.eval.run_episode import run_episode


def evaluate_on_val(model, val_data, n_episodes: int = 10, seed: int = 42) -> float:
    """Run model on val set, return mean Sharpe (higher is better)."""
    env = PortfolioEnv(
        val_data,
        history_window=50,
        fee_rate=0.001,
        slippage_sigma=0.05,
        episode_length=252,
        max_weight_per_asset=0.35,
        seed=seed,
    )

    def get_action(obs, info):
        action, _ = model.predict(obs, deterministic=True)
        return action

    all_rewards = []
    for i in range(n_episodes):
        rewards, _, _ = run_episode(env, get_action, seed=seed + i)
        all_rewards.append(rewards)

    concat = np.concatenate(all_rewards)
    metrics = compute_metrics(concat, periods_per_year=252 * 24)
    return metrics["sharpe"]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Tune PPO hyperparameters on validation set")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--timesteps-per-trial", type=int, default=50_000)
    parser.add_argument("--n-trials", type=int, default=6, help="Max configs to try")
    parser.add_argument("--n-envs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("models/ppo_best"))
    args = parser.parse_args()

    train, val, test = load_splits(args.data_dir)
    if not train or not val:
        print("Need both train and val data. Run data pipeline with 2+ years.")
        return 1

    # Grid: (turnover_penalty, learning_rate, reward_scale)
    # Prioritize turnover_penalty to test regularization
    grid = [
        (0.05, 1e-4, 10.0),   # high turnover penalty
        (0.01, 1e-4, 10.0),   # medium
        (0.0, 1e-4, 10.0),    # none
        (0.05, 1e-4, 5.0),
        (0.01, 3e-4, 10.0),
        (0.05, 3e-4, 10.0),
    ][: args.n_trials]

    print(f"Tuning {len(grid)} configs, {args.timesteps_per_trial} steps each...")
    best_sharpe = -np.inf
    best_config = None
    best_model_path = None

    for i, (turnover_penalty, lr, reward_scale) in enumerate(grid):
        print(f"\n--- Trial {i + 1}/{len(grid)}: turnover_penalty={turnover_penalty}, lr={lr}, reward_scale={reward_scale} ---")

        def env_fn():
            return PortfolioEnv(
                train,
                history_window=50,
                fee_rate=0.001,
                slippage_sigma=0.05,
                episode_length=252,
                reward_scale=reward_scale,
                turnover_penalty=turnover_penalty,
                max_weight_per_asset=0.35,
                seed=None,
            )

        vec_env = DummyVecEnv([env_fn for _ in range(args.n_envs)])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=0.99)

        model = PPO(
            "MlpPolicy",
            vec_env,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            learning_rate=lr,
            n_steps=1024,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.05,
            verbose=0,
            seed=args.seed,
        )

        model.learn(total_timesteps=args.timesteps_per_trial)

        # Eval on val: use val data, wrap with same obs normalization
        def val_env_fn():
            return PortfolioEnv(
                val,
                history_window=50,
                fee_rate=0.001,
                slippage_sigma=0.05,
                episode_length=252,
                reward_scale=reward_scale,
                turnover_penalty=turnover_penalty,
                max_weight_per_asset=0.35,
                seed=None,
            )

        vec_eval = DummyVecEnv([val_env_fn])
        vec_eval = VecNormalize(vec_eval, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=0.99)
        vec_eval.obs_rms = vec_env.obs_rms
        vec_eval.training = False

        from src.env.vec_env_wrapper import VecEnvToGymWrapper
        eval_env = VecEnvToGymWrapper(vec_eval)

        def get_action(obs, info):
            action, _ = model.predict(obs, deterministic=True)
            return action

        all_rewards = []
        for j in range(10):
            rewards, _, _ = run_episode(eval_env, get_action, seed=args.seed + j)
            all_rewards.append(rewards)
        concat = np.concatenate(all_rewards)
        metrics = compute_metrics(concat, periods_per_year=252 * 24)
        sharpe = metrics["sharpe"]

        print(f"  Val Sharpe: {sharpe:.4f}  Total ret: {metrics['total_return']:.2%}")

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_config = dict(
                turnover_penalty=turnover_penalty,
                learning_rate=lr,
                reward_scale=reward_scale,
            )
            args.out.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(args.out))
            vec_env.save(str(args.out.parent / "vec_normalize_best.pkl"))
            best_model_path = args.out
            print(f"  -> New best! Saved to {args.out}")

    print(f"\n--- Best config: {best_config} (Val Sharpe: {best_sharpe:.4f}) ---")
    print(f"Model: {best_model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
