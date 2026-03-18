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
from src.policy import PortfolioDictExtractor
from src.env.vec_env_wrapper import VecEnvToGymWrapper
from src.eval.cost_analysis import replay_from_data
from src.eval.metrics import compute_metrics
from src.eval.run_episode import run_equal_weight, run_equal_weight_buy_and_hold, run_episode


def main():
    parser = argparse.ArgumentParser(description="Evaluate agent vs baseline")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--model", type=Path, default=Path("models/ppo_portfolio.zip"))
    parser.add_argument("--vec-normalize", type=Path, default=Path("models/vec_normalize.pkl"), help="Path to VecNormalize stats; use '' to skip")
    parser.add_argument("--episode-length", type=int, default=None)
    parser.add_argument("--n-episodes", type=int, default=5, help="For agent (random starts)")
    parser.add_argument("--slippage-sigma", type=float, default=0.1)
    parser.add_argument("--notional", type=float, default=1e6, help="Portfolio size (USD) for slippage")
    parser.add_argument("--max-weight", type=float, default=0.35)
    parser.add_argument("--cost-analysis", action="store_true", help="Break down fee vs slippage impact")
    parser.add_argument("--cost-model", choices=["full", "simple"], default="full", help="full=fee+slippage, simple=fee only (PGPortfolio, match training)")
    parser.add_argument("--policy", choices=["mlp", "cnn"], default="mlp", help="Must match trained model")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train, val, test = load_splits(args.data_dir)
    data = {"train": train, "val": val, "test": test}[args.split]
    if not data:
        print(f"No data in {args.data_dir}/{args.split}")
        return 1

    obs_type = "pgportfolio" if args.policy == "cnn" else "returns"
    fee_rate = 0.0025 if args.cost_model == "simple" else 0.001

    def make_env():
        return PortfolioEnv(
            data,
            history_window=50,
            fee_rate=fee_rate,
            slippage_sigma=args.slippage_sigma,
            notional_usd=args.notional,
            episode_length=args.episode_length,
            max_weight_per_asset=args.max_weight,
            observation_type=obs_type,
            cost_model=args.cost_model,
            seed=args.seed,
        )

    env = make_env()
    periods_per_year = 252 * 24  # 1h bars

    # Baselines
    print("\n--- Equal-weight rebalancing (with friction) ---")
    rewards_rebal, _, _ = run_equal_weight(env, seed=args.seed)
    metrics_rebal = compute_metrics(rewards_rebal, periods_per_year=periods_per_year)
    print(f"  Sharpe: {metrics_rebal['sharpe']:.4f}  Sortino: {metrics_rebal['sortino']:.4f}")
    print(f"  Max DD: {metrics_rebal['max_drawdown']:.2%}  Total ret: {metrics_rebal['total_return']:.2%}")

    print("\n--- Equal-weight buy-and-hold (no friction) ---")
    rewards_bh, _, _ = run_equal_weight_buy_and_hold(env, seed=args.seed)
    metrics_bh = compute_metrics(rewards_bh, periods_per_year=periods_per_year)
    print(f"  Sharpe: {metrics_bh['sharpe']:.4f}  Sortino: {metrics_bh['sortino']:.4f}")
    print(f"  Max DD: {metrics_bh['max_drawdown']:.2%}  Total ret: {metrics_bh['total_return']:.2%}")

    # Agent (if model exists)
    if args.model.exists():
        print("\n--- PPO agent ---")
        custom_objects = {}
        if args.policy == "cnn":
            custom_objects["policy_kwargs"] = dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                features_extractor_class=PortfolioDictExtractor,
                features_extractor_kwargs=dict(features_dim=128),
            )
        model = PPO.load(str(args.model), custom_objects=custom_objects or None)

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
        sharpe_diff_rebal = metrics_agent["sharpe"] - metrics_rebal["sharpe"]
        sharpe_diff_bh = metrics_agent["sharpe"] - metrics_bh["sharpe"]
        print(f"  Sharpe diff vs rebalancing: {sharpe_diff_rebal:+.4f}")
        print(f"  Sharpe diff vs buy-and-hold: {sharpe_diff_bh:+.4f}")

        # Cost attribution: replay with different fee/slippage to isolate impact
        if args.cost_analysis:
            print("\n--- Cost analysis (agent, single full episode) ---")
            def make_env_full():
                return PortfolioEnv(
                    data,
                    history_window=50,
                    fee_rate=0.001,
                    slippage_sigma=args.slippage_sigma,
                    notional_usd=args.notional,
                    episode_length=None,  # Full period for cost analysis
                    max_weight_per_asset=args.max_weight,
                    observation_type=obs_type,
                    seed=args.seed,
                )
            if args.vec_normalize and str(args.vec_normalize) and args.vec_normalize.exists():
                eval_vec_full = DummyVecEnv([make_env_full])
                eval_vec_full = VecNormalize.load(str(args.vec_normalize), eval_vec_full)
                eval_vec_full.training = False
                eval_vec_full.norm_reward = False
                agent_env_full = VecEnvToGymWrapper(eval_vec_full)
            else:
                agent_env_full = make_env_full()
            rewards, _, _, actions, start_t = run_episode(
                agent_env_full, get_action, seed=args.seed, return_actions=True
            )
            # Replay with 4 cost configs
            r_full = replay_from_data(
                data, start_t, actions,
                fee_rate=0.001, slippage_sigma=args.slippage_sigma,
                notional_usd=args.notional,
                max_weight_per_asset=args.max_weight,
            )
            r_no_fee = replay_from_data(
                data, start_t, actions,
                fee_rate=0, slippage_sigma=args.slippage_sigma,
                notional_usd=args.notional,
                max_weight_per_asset=args.max_weight,
            )
            r_no_slip = replay_from_data(
                data, start_t, actions,
                fee_rate=0.001, slippage_sigma=0,
                notional_usd=args.notional,
                max_weight_per_asset=args.max_weight,
            )
            r_none = replay_from_data(
                data, start_t, actions,
                fee_rate=0, slippage_sigma=0,
                notional_usd=args.notional,
                max_weight_per_asset=args.max_weight,
            )
            print(f"  Frictionless (no costs):  ret={r_none.total_return:+.2%}  final={r_none.final_value:.4f}")
            print(f"  Fees only (no slippage):  ret={r_no_slip.total_return:+.2%}  cum_fee={r_no_slip.cumulative_fee_pct:.2%}")
            print(f"  Slippage only (no fees):  ret={r_no_fee.total_return:+.2%}  cum_slip={r_no_fee.cumulative_slippage_pct:.2%}")
            print(f"  Full costs:               ret={r_full.total_return:+.2%}  cum_fee={r_full.cumulative_fee_pct:.2%}  cum_slip={r_full.cumulative_slippage_pct:.2%}")
            # Fee impact = value lost when adding fees (vs frictionless)
            fee_impact = r_none.final_value - r_no_slip.final_value
            # Slippage impact = value lost when adding slippage (vs frictionless)
            slip_impact = r_none.final_value - r_no_fee.final_value
            v0 = max(r_none.final_value, 1e-10)
            print(f"  Fee impact (frictionless - fees_only):     {fee_impact:.4f} ({fee_impact / v0 * 100:.1f}% of frictionless)")
            print(f"  Slippage impact (frictionless - slip_only): {slip_impact:.4f} ({slip_impact / v0 * 100:.1f}% of frictionless)")
    else:
        print(f"\nModel not found: {args.model}. Run: python scripts/train_agent.py")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
