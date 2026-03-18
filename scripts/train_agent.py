#!/usr/bin/env python3
"""Train PPO agent on portfolio environment."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.data.split import load_splits
from src.env import PortfolioEnv
from src.policy import PortfolioDictExtractor


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--episode-length", type=int, default=100, help="Steps per episode; 0 = full dataset")
    parser.add_argument("--n-envs", type=int, default=4, help="Parallel envs for vec env")
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--reward-scale", type=float, default=10.0)
    parser.add_argument("--slippage-sigma", type=float, default=0.05)
    parser.add_argument("--notional", type=float, default=1e6, help="Portfolio size (USD) for slippage scaling")
    parser.add_argument("--turnover-penalty", type=float, default=0.05)
    parser.add_argument("--max-weight", type=float, default=0.35, help="Max weight per asset (concentration limit)")
    parser.add_argument("--reward-type", choices=["log_return", "sharpe"], default="log_return")
    parser.add_argument("--policy", choices=["mlp", "cnn"], default="mlp", help="mlp=returns, cnn=pgportfolio")
    parser.add_argument("--sharpe-window", type=int, default=20, help="Rolling window for Sharpe reward")
    parser.add_argument("--diversification-lambda", type=float, default=1e-4, help="PGPortfolio concentration penalty")
    parser.add_argument("--out", type=Path, default=Path("models/ppo_portfolio"))
    parser.add_argument("--save-vec-normalize", type=Path, default=Path("models/vec_normalize.pkl"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train, val, test = load_splits(args.data_dir)
    data = {"train": train, "val": val, "test": test}[args.split]
    if not data:
        print(f"No data in {args.data_dir}/{args.split}")
        return 1

    episode_len = None if args.episode_length <= 0 else args.episode_length
    obs_type = "pgportfolio" if args.policy == "cnn" else "returns"

    def env_fn():
        return PortfolioEnv(
            data,
            history_window=50,
            fee_rate=0.001,
            slippage_sigma=args.slippage_sigma,
            notional_usd=args.notional,
            episode_length=episode_len,
            reward_scale=args.reward_scale,
            turnover_penalty=args.turnover_penalty,
            max_weight_per_asset=args.max_weight,
            reward_type=args.reward_type,
            sharpe_window=args.sharpe_window,
            observation_type=obs_type,
            diversification_lambda=args.diversification_lambda if args.policy == "cnn" else 0.0,
            seed=None,
        )

    vec_env = DummyVecEnv([env_fn for _ in range(args.n_envs)])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=(args.policy == "mlp"),  # Don't norm CNN input (already normalized)
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.99,
    )

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    if args.policy == "cnn":
        policy_kwargs["features_extractor_class"] = PortfolioDictExtractor
        policy_kwargs["features_extractor_kwargs"] = dict(features_dim=128)

    policy_cls = "MultiInputPolicy" if args.policy == "cnn" else "MlpPolicy"
    model = PPO(
        policy_cls,
        vec_env,
        policy_kwargs=policy_kwargs,
        learning_rate=args.learning_rate,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        verbose=1,
        seed=args.seed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.learn(total_timesteps=args.total_timesteps)
    model.save(str(args.out))

    args.save_vec_normalize.parent.mkdir(parents=True, exist_ok=True)
    vec_env.save(str(args.save_vec_normalize))
    print(f"VecNormalize stats saved to {args.save_vec_normalize}")

    print(f"Model saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
