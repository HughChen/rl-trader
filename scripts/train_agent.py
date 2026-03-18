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
    parser.add_argument("--cost-model", choices=["full", "simple"], default="full", help="full=fee+slippage (match eval), simple=fee only")
    parser.add_argument("--sample-bias", type=float, default=0.3, help="Bias episode starts toward recent (0=uniform, 0.3=recent). 0 to disable")
    parser.add_argument("--min-rebalance-interval", type=int, default=6, help="Min steps between rebalances (0=no constraint)")
    parser.add_argument("--max-turnover", type=float, default=0.2, help="Max turnover per step (0=no cap)")
    parser.add_argument("--terminal-reward-scale", type=float, default=1.0, help="Terminal Sharpe bonus at episode end (0=disabled)")
    parser.add_argument("--lr-decay", action="store_true", help="Linear LR decay (PGPortfolio uses decay)")
    parser.add_argument("--no-high-impact", action="store_true", help="Disable min-rebalance-interval, max-turnover, terminal-reward")
    parser.add_argument("--curriculum", action="store_true", help="Phase 1: short episodes, Phase 2: long episodes (uses 2x total_timesteps)")
    parser.add_argument("--regime-bias", type=float, default=0.0, help="Oversample bear periods (0=disabled, 1.0=2x weight for bear)")
    parser.add_argument("--load", type=Path, default=None, help="Load model to continue training")
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

    fee_rate = 0.0025 if args.cost_model == "simple" else 0.001  # PGPortfolio uses 0.25%
    sample_bias = args.sample_bias if args.policy == "cnn" and args.sample_bias > 0 else 0.0
    # Regime bias: when set, use uniform-ish sampling (override sample_bias for regime)
    regime_bias = args.regime_bias if args.policy == "cnn" else 0.0
    if regime_bias > 0:
        sample_bias = 0.0  # Regime uses its own sampling

    min_rebal = 0 if args.no_high_impact else args.min_rebalance_interval
    max_turnover = None if args.no_high_impact or args.max_turnover <= 0 else args.max_turnover
    terminal_scale = 0.0 if args.no_high_impact else args.terminal_reward_scale

    def make_env_fn(ep_len: int | None):
        def env_fn():
            return PortfolioEnv(
            data,
            history_window=50,
            fee_rate=fee_rate,
            slippage_sigma=args.slippage_sigma,
            notional_usd=args.notional,
            episode_length=ep_len,
            reward_scale=args.reward_scale,
            turnover_penalty=args.turnover_penalty,
            max_weight_per_asset=args.max_weight,
            reward_type=args.reward_type,
            sharpe_window=args.sharpe_window,
            observation_type=obs_type,
            diversification_lambda=args.diversification_lambda if args.policy == "cnn" else 0.0,
            cost_model=args.cost_model,
            sample_bias=sample_bias,
            min_rebalance_interval=min_rebal,
            max_turnover_per_step=max_turnover,
            terminal_reward_scale=terminal_scale,
            regime_bias=regime_bias,
            regime_lookback=24,
            seed=None,
        )
        return env_fn

    env_fn = make_env_fn(episode_len)

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

    # Linear LR decay: lr goes from initial to 0 as progress goes 1->0
    lr = args.learning_rate
    if args.lr_decay:
        lr = lambda progress: progress * args.learning_rate

    policy_cls = "MultiInputPolicy" if args.policy == "cnn" else "MlpPolicy"
    timesteps_per_phase = args.total_timesteps // 2 if args.curriculum else args.total_timesteps

    model = PPO(
        policy_cls,
        vec_env,
        policy_kwargs=policy_kwargs,
        learning_rate=lr,
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

    if args.load is not None:
        load_path = args.load.with_suffix(".zip") if args.load.suffix != ".zip" else args.load
        if load_path.exists():
            model = PPO.load(str(load_path), env=vec_env)
            print(f"Loaded model from {load_path}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.curriculum:
        # Phase 1: short episodes (allocation learning)
        print("Curriculum Phase 1: short episodes (50 steps)")
        vec_env_short = DummyVecEnv([make_env_fn(50) for _ in range(args.n_envs)])
        vec_env_short = VecNormalize(
            vec_env_short,
            norm_obs=(args.policy == "mlp"),
            norm_reward=False,
            clip_obs=10.0,
            gamma=0.99,
        )
        model.set_env(vec_env_short)
        model.learn(total_timesteps=timesteps_per_phase)
        # Phase 2: long episodes (cost awareness)
        print("Curriculum Phase 2: long episodes (200 steps)")
        vec_env_long = DummyVecEnv([make_env_fn(200) for _ in range(args.n_envs)])
        vec_env_long = VecNormalize(
            vec_env_long,
            norm_obs=(args.policy == "mlp"),
            norm_reward=False,
            clip_obs=10.0,
            gamma=0.99,
        )
        model.set_env(vec_env_long)
        model.learn(total_timesteps=timesteps_per_phase)
        vec_env = vec_env_long
    else:
        model.learn(total_timesteps=args.total_timesteps)

    model.save(str(args.out))

    args.save_vec_normalize.parent.mkdir(parents=True, exist_ok=True)
    vec_env.save(str(args.save_vec_normalize))
    print(f"VecNormalize stats saved to {args.save_vec_normalize}")

    print(f"Model saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
