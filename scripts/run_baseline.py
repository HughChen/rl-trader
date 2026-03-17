#!/usr/bin/env python3
"""Run equal-weight baseline and report metrics."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.split import load_splits
from src.env import PortfolioEnv
from src.eval.metrics import compute_metrics
from src.eval.run_episode import run_equal_weight


def main():
    parser = argparse.ArgumentParser(description="Run equal-weight baseline")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--episode-length", type=int, default=None, help="None = full dataset")
    parser.add_argument("--n-episodes", type=int, default=1, help="For random sub-episodes")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train, val, test = load_splits(args.data_dir)
    data = {"train": train, "val": val, "test": test}[args.split]
    if not data:
        print(f"No data in {args.data_dir}/{args.split}")
        return 1

    env = PortfolioEnv(
        data,
        history_window=50,
        fee_rate=0.001,
        slippage_sigma=0.1,
        episode_length=args.episode_length,
        seed=args.seed,
    )

    if args.episode_length and args.n_episodes > 1:
        import numpy as np
        from src.eval.run_episode import run_episodes

        action = np.full(env.n_assets, 1.0 / env.n_assets, dtype=np.float32)

        def get_action(obs, info):
            return action

        _, _, metrics = run_episodes(env, get_action, n_episodes=args.n_episodes, seed=args.seed)
    else:
        rewards, final_val, _ = run_equal_weight(env, seed=args.seed)
        metrics = compute_metrics(rewards, periods_per_year=252 * 24)  # 1h bars

    print(f"\nEqual-weight baseline ({args.split})")
    print(f"  Sharpe:      {metrics['sharpe']:.4f}")
    print(f"  Sortino:     {metrics['sortino']:.4f}")
    print(f"  Max DD:      {metrics['max_drawdown']:.2%}")
    print(f"  Total ret:   {metrics['total_return']:.2%}")
    print(f"  Volatility:  {metrics['volatility']:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
