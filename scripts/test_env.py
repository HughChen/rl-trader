#!/usr/bin/env python3
"""Quick test of the portfolio environment."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.split import load_splits
from src.env.portfolio_env import PortfolioEnv


def main():
    data_dir = Path("data/processed")
    train, val, test = load_splits(data_dir)

    # Use test split (we have data there)
    data = test if test else (val if val else train)
    if not data:
        print("No data found. Run: python scripts/run_data_pipeline.py")
        return 1

    print(f"Using {len(data)} symbols: {list(data.keys())[:3]}...")

    env = PortfolioEnv(
        data,
        history_window=20,
        fee_rate=0.001,
        slippage_sigma=0.05,
        episode_length=100,
        seed=42,
    )

    obs, info = env.reset()
    print(f"Obs shape: {obs.shape}, n_assets: {env.n_assets}")

    total_reward = 0.0
    n_steps = 0
    # Equal-weight baseline
    action = np.full(env.n_assets, 1.0 / env.n_assets, dtype=np.float32)

    done = False
    while not done:
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        n_steps += 1

    print(f"Equal-weight: {n_steps} steps, total reward {total_reward:.6f}")

    # Random agent (sample actions)
    obs, info = env.reset(seed=123)
    total_reward = 0.0
    n_steps = 0
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        n_steps += 1

    print(f"Random agent: {n_steps} steps, total reward {total_reward:.6f}")
    print(f"Final portfolio value: {info['portfolio_value']:.4f}")
    print("Env test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
