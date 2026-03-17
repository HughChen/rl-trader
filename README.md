# rl-trader

RL-based cryptocurrency portfolio allocation for offline backtesting.

- [PLANNING.md](PLANNING.md) — Design and roadmap
- [experiments/](experiments/) — Experimental setup and findings

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

## Data Pipeline

Fetch spot OHLCV, compute technical indicators, and split into train/val/test:

```bash
python scripts/run_data_pipeline.py
```

**Exchange restrictions** (important for historical depth):

| Exchange | 1h history | Notes |
|----------|------------|-------|
| **KuCoin** (default) | ~2 years | ✅ Use this. Paginates correctly. |
| **Kraken** | ~1 month | 720-candle API limit; `since` ignored beyond that. |
| **Binance** | Years | Geo-restricted (451) in US and some regions. |

Options:
- `--data-dir DATA_DIR` — Output directory (default: `data/`)
- `--years YEARS` — Years of history (default: 2)
- `--skip-fetch` — Use existing raw data, recompute indicators and split
- `--exchange EXCHANGE` — CCXT exchange (kucoin, kraken, binance, bybit)

Output structure:
```
data/
├── raw/           # OHLCV parquet files per symbol
└── processed/
    ├── train/
    ├── val/
    └── test/
```

## Environment (Phase 2)

Gymnasium-compatible portfolio allocation environment:

```python
from src.data.split import load_splits
from src.env.portfolio_env import PortfolioEnv

train, val, test = load_splits("data/processed")
env = PortfolioEnv(train, history_window=50, fee_rate=0.001, slippage_sigma=0.1)
obs, info = env.reset()
obs, reward, done, truncated, info = env.step(env.action_space.sample())
```

Test the environment:
```bash
python scripts/test_env.py
```

## Training & Evaluation (Phase 3–4)

**Baseline** (equal-weight buy-and-hold):
```bash
python scripts/run_baseline.py --split test
```

**Train PPO agent**:
```bash
python scripts/train_agent.py --split train --total-timesteps 100000 --episode-length 252
```

PPO improvements: larger network (256×256), VecNormalize for obs, reward scaling, lower learning rate. Options: `--reward-scale`, `--slippage-sigma`, `--learning-rate`.

**Evaluate** (agent vs baseline):
```bash
python scripts/evaluate.py --split test --model models/ppo_portfolio.zip --vec-normalize models/vec_normalize.pkl
```

**Hyperparameter tuning** (on validation set):
```bash
python scripts/tune_hyperparams.py --timesteps-per-trial 40000 --n-trials 6
```

**Turnover penalty** (discourages churning): `--turnover-penalty 0.05` in `train_agent.py`
