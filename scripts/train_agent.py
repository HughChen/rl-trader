#!/usr/bin/env python3
"""Train PPO agent on portfolio environment."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.data.split import load_splits
from src.env import PortfolioEnv


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--episode-length", type=int, default=252, help="Steps per episode")
    parser.add_argument("--n-envs", type=int, default=4, help="Parallel envs for vec env")
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--out", type=Path, default=Path("models/ppo_portfolio"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train, val, test = load_splits(args.data_dir)
    data = {"train": train, "val": val, "test": test}[args.split]
    if not data:
        print(f"No data in {args.data_dir}/{args.split}")
        return 1

    env_fn = lambda: PortfolioEnv(
        data,
        history_window=50,
        fee_rate=0.001,
        slippage_sigma=0.1,
        episode_length=args.episode_length,
        seed=None,
    )

    vec_env = DummyVecEnv([env_fn for _ in range(args.n_envs)])

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=args.learning_rate,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        seed=args.seed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.learn(total_timesteps=args.total_timesteps)
    model.save(str(args.out))
    print(f"Model saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
