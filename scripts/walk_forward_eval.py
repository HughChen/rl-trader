#!/usr/bin/env python3
"""
Walk-forward evaluation: train on rolling windows, evaluate on out-of-sample periods.

Reports mean/median Sharpe and consistency across windows for more robust performance estimate.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.data.split import load_full_data, split_by_date
from src.env import PortfolioEnv
from src.env.vec_env_wrapper import VecEnvToGymWrapper
from src.eval.metrics import compute_metrics
from src.eval.run_episode import run_equal_weight, run_episode


# Rolling windows: (train_end, test_end) -> train on data <= train_end, test on (train_end, test_end]
WALK_FORWARD_WINDOWS = [
    ("2024-06-30", "2024-09-30"),   # Q1-Q2 train, Q3 test
    ("2024-09-30", "2024-12-31"),   # Q1-Q3 train, Q4 test
    ("2024-12-31", "2025-03-31"),   # full 2024 train, Q1 2025 test
    ("2025-03-31", "2025-06-30"),   # through Q1 2025 train, Q2 2025 test
]


def main():
    parser = argparse.ArgumentParser(description="Walk-forward evaluation")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--timesteps-per-window", type=int, default=100_000)
    parser.add_argument("--episode-length", type=int, default=100)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--n-eval-episodes", type=int, default=5)
    parser.add_argument("--turnover-penalty", type=float, default=0.05)
    parser.add_argument("--max-weight", type=float, default=0.35)
    parser.add_argument("--notional", type=float, default=1e6)
    parser.add_argument("--models-dir", type=Path, default=Path("models/walk_forward"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    full_data = load_full_data(args.data_dir)
    if not full_data:
        print("No data. Run data pipeline first.")
        return 1

    # Infer date range from data
    all_dates = []
    for df in full_data.values():
        all_dates.extend(df.index.tolist())
    min_ts = min(all_dates)
    max_ts = max(all_dates)

    # Filter windows to those we have data for
    windows = []
    for train_end, test_end in WALK_FORWARD_WINDOWS:
        train_end_ts = pd.Timestamp(train_end, tz="UTC")
        test_end_ts = pd.Timestamp(test_end, tz="UTC")
        if train_end_ts >= max_ts or test_end_ts > max_ts:
            continue
        if train_end_ts <= min_ts:
            continue
        windows.append((train_end, test_end))

    if not windows:
        print("No valid windows in date range. Need more data.")
        return 1

    print(f"Walk-forward: {len(windows)} windows, {args.timesteps_per_window} steps/window")
    print()

    period_sharpes_agent = []
    period_sharpes_baseline = []
    periods_per_year = 252 * 24

    for i, (train_end, test_end) in enumerate(windows):
        print(f"--- Window {i + 1}/{len(windows)}: train <= {train_end}, test ({train_end}, {test_end}] ---")

        train, val, test = split_by_date(full_data, train_end=train_end, val_end=test_end)
        # val holds our test period (between train_end and test_end)
        eval_data = val
        if not eval_data or sum(len(df) for df in eval_data.values()) == 0:
            print("  No eval data, skipping")
            continue

        # Train
        def env_fn():
            return PortfolioEnv(
                train,
                history_window=50,
                fee_rate=0.001,
                slippage_sigma=0.05,
                notional_usd=args.notional,
                episode_length=args.episode_length,
                reward_scale=10.0,
                turnover_penalty=args.turnover_penalty,
                max_weight_per_asset=args.max_weight,
                seed=None,
            )

        vec_env = DummyVecEnv([env_fn for _ in range(args.n_envs)])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=0.99)

        model = PPO(
            "MlpPolicy",
            vec_env,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            learning_rate=1e-4,
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
        model.learn(total_timesteps=args.timesteps_per_window)

        # Save per-window model
        out_path = args.models_dir / f"window_{i + 1}"
        out_path.mkdir(parents=True, exist_ok=True)
        model.save(str(out_path / "model.zip"))
        vec_env.save(str(out_path / "vec_normalize.pkl"))

        # Evaluate on test period
        def eval_env_fn():
            return PortfolioEnv(
                eval_data,
                history_window=50,
                fee_rate=0.001,
                slippage_sigma=0.1,
                notional_usd=args.notional,
                episode_length=None,  # full period
                max_weight_per_asset=args.max_weight,
                seed=args.seed,
            )

        eval_vec = DummyVecEnv([eval_env_fn])
        eval_vec = VecNormalize.load(str(out_path / "vec_normalize.pkl"), eval_vec)
        eval_vec.training = False
        eval_vec.norm_reward = False
        eval_env = VecEnvToGymWrapper(eval_vec)

        # Baseline
        baseline_env = PortfolioEnv(
            eval_data,
            history_window=50,
            fee_rate=0.001,
            slippage_sigma=0.1,
            notional_usd=args.notional,
            episode_length=None,
            max_weight_per_asset=args.max_weight,
            seed=args.seed,
        )
        rewards_baseline, _, _ = run_equal_weight(baseline_env, seed=args.seed)
        metrics_baseline = compute_metrics(rewards_baseline, periods_per_year=periods_per_year)

        # Agent
        def get_action(obs, info):
            action, _ = model.predict(obs, deterministic=True)
            return action

        all_rewards = []
        for j in range(args.n_eval_episodes):
            rewards, _, _ = run_episode(eval_env, get_action, seed=args.seed + j)
            all_rewards.append(rewards)
        concat = np.concatenate(all_rewards)
        metrics_agent = compute_metrics(concat, periods_per_year=periods_per_year)

        period_sharpes_agent.append(metrics_agent["sharpe"])
        period_sharpes_baseline.append(metrics_baseline["sharpe"])

        print(f"  Agent:   Sharpe {metrics_agent['sharpe']:.4f}  Ret {metrics_agent['total_return']:.2%}")
        print(f"  Baseline: Sharpe {metrics_baseline['sharpe']:.4f}  Ret {metrics_baseline['total_return']:.2%}")
        print()

    # Summary
    print("=" * 50)
    print("Walk-forward summary")
    print("=" * 50)
    agent_mean = np.mean(period_sharpes_agent)
    agent_median = np.median(period_sharpes_agent)
    baseline_mean = np.mean(period_sharpes_baseline)
    wins = sum(1 for a, b in zip(period_sharpes_agent, period_sharpes_baseline) if a > b)
    print(f"Agent   Sharpe: mean={agent_mean:.4f}  median={agent_median:.4f}")
    print(f"Baseline Sharpe: mean={baseline_mean:.4f}")
    print(f"Agent beats baseline: {wins}/{len(windows)} windows")
    print(f"Models saved to {args.models_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
