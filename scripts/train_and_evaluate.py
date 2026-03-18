#!/usr/bin/env python3
"""Train PPO agent and evaluate on val/test. Single pipeline for full workflow."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO agent, then evaluate on val and test splits"
    )
    # Training args (forwarded to train_agent.py)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--episode-length", type=int, default=100)
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--policy", choices=["mlp", "cnn"], default="cnn")
    parser.add_argument("--turnover-penalty", type=float, default=0.05)
    parser.add_argument("--notional", type=float, default=100_000)
    parser.add_argument("--slippage-sigma", type=float, default=0.05)
    parser.add_argument("--max-weight", type=float, default=0.35)
    parser.add_argument("--out", type=Path, default=Path("models/ppo_portfolio"))
    parser.add_argument("--save-vec-normalize", type=Path, default=Path("models/vec_normalize.pkl"))
    parser.add_argument("--seed", type=int, default=42)
    # Pipeline control
    parser.add_argument("--skip-train", action="store_true", help="Skip training, only evaluate")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation, only train")
    parser.add_argument("--eval-splits", nargs="+", default=["val", "test"], help="Splits to evaluate on")
    parser.add_argument("--cost-analysis", action="store_true", help="Run cost attribution on test")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    train_script = root / "scripts" / "train_agent.py"
    eval_script = root / "scripts" / "evaluate.py"
    python = sys.executable

    if not args.skip_train:
        print("=" * 60)
        print("TRAINING")
        print("=" * 60)
        cmd = [
            python,
            str(train_script),
            "--data-dir", str(args.data_dir),
            "--split", "train",
            "--episode-length", str(args.episode_length),
            "--total-timesteps", str(args.total_timesteps),
            "--policy", args.policy,
            "--turnover-penalty", str(args.turnover_penalty),
            "--notional", str(args.notional),
            "--slippage-sigma", str(args.slippage_sigma),
            "--max-weight", str(args.max_weight),
            "--out", str(args.out),
            "--save-vec-normalize", str(args.save_vec_normalize),
            "--seed", str(args.seed),
        ]
        result = subprocess.run(cmd, cwd=str(root))
        if result.returncode != 0:
            print(f"Training failed with exit code {result.returncode}")
            return result.returncode

    if not args.skip_eval:
        model_path = args.out.with_suffix(".zip") if args.out.suffix != ".zip" else args.out
        if not model_path.exists():
            print(f"Model not found: {model_path}. Cannot evaluate.")
            return 1

        for split in args.eval_splits:
            print("\n" + "=" * 60)
            print(f"EVALUATION ({split})")
            print("=" * 60)
            cmd = [
                python,
                str(eval_script),
                "--data-dir", str(args.data_dir),
                "--split", split,
                "--model", str(model_path),
                "--vec-normalize", str(args.save_vec_normalize),
                "--policy", args.policy,
                "--notional", str(args.notional),
                "--slippage-sigma", str(args.slippage_sigma),
                "--max-weight", str(args.max_weight),
                "--seed", str(args.seed),
            ]
            if args.cost_analysis and split == "test":
                cmd.append("--cost-analysis")
            result = subprocess.run(cmd, cwd=str(root))
            if result.returncode != 0:
                print(f"Evaluation on {split} failed with exit code {result.returncode}")
                return result.returncode

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
