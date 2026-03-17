#!/usr/bin/env python3
"""Evaluate trained agent vs baseline."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.data.split import load_splits
from src.env import PortfolioEnv
from src.env.vec_env_wrapper import VecEnvToGymWrapper
from src.eval.metrics import compute_metrics
from src.eval.run_episode import run_equal_weight, run_episode


def main():
    parser = argparse.ArgumentParser(description="Evaluate agent vs baseline")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--model", type=Path, default=Path("models/ppo_portfolio.zip"))
    parser.add_argument("--vec-normalize", type=Path, default=Path("models/vec_normalize.pkl"), help="Path to VecNormalize stats; use '' to skip")
    parser.add_argument("--episode-length", type=int, default=None)
    parser.add_argument("--n-episodes", type=int, default=5, help="For agent (random starts)")
    parser.add_argument("--slippage-sigma", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train, val, test = load_splits(args.data_dir)
    data = {"train": train, "val": val, "test": test}[args.split]
    if not data:
        print(f"No data in {args.data_dir}/{args.split}")
        return 1

    def make_env():
        return PortfolioEnv(
            data,
            history_window=50,
            fee_rate=0.001,
            slippage_sigma=args.slippage_sigma,
            episode_length=args.episode_length,
            seed=args.seed,
        )

    env = make_env()
    periods_per_year = 252 * 24  # 1h bars

    # Baseline
    print("\n--- Equal-weight baseline ---")
    rewards_baseline, _, _ = run_equal_weight(env, seed=args.seed)
    metrics_baseline = compute_metrics(rewards_baseline, periods_per_year=periods_per_year)
    print(f"  Sharpe: {metrics_baseline['sharpe']:.4f}  Sortino: {metrics_baseline['sortino']:.4f}")
    print(f"  Max DD: {metrics_baseline['max_drawdown']:.2%}  Total ret: {metrics_baseline['total_return']:.2%}")

    # Agent (if model exists)
    if args.model.exists():
        print("\n--- PPO agent ---")
        model = PPO.load(str(args.model))

        if args.vec_normalize and str(args.vec_normalize) and args.vec_normalize.exists():
            eval_vec_env = DummyVecEnv([make_env])
            eval_vec_env = VecNormalize.load(str(args.vec_normalize), eval_vec_env)
            eval_vec_env.training = False
            eval_vec_env.norm_reward = False
            agent_env = VecEnvToGymWrapper(eval_vec_env)
        else:
            agent_env = make_env()

        def get_action(obs, info):
            action, _ = model.predict(obs, deterministic=True)
            return action

        all_rewards = []
        for i in range(args.n_episodes):
            rewards, _, _ = run_episode(agent_env, get_action, seed=args.seed + i)
            all_rewards.append(rewards)

        concat = np.concatenate(all_rewards)
        metrics_agent = compute_metrics(concat, periods_per_year=periods_per_year)
        print(f"  Sharpe: {metrics_agent['sharpe']:.4f}  Sortino: {metrics_agent['sortino']:.4f}")
        print(f"  Max DD: {metrics_agent['max_drawdown']:.2%}  Total ret: {metrics_agent['total_return']:.2%}")

        print("\n--- Comparison ---")
        sharpe_diff = metrics_agent["sharpe"] - metrics_baseline["sharpe"]
        print(f"  Sharpe diff (agent - baseline): {sharpe_diff:+.4f}")
    else:
        print(f"\nModel not found: {args.model}. Run: python scripts/train_agent.py")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
