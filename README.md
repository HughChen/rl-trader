# rl-trader

RL-based cryptocurrency portfolio allocation for offline backtesting. See [PLANNING.md](PLANNING.md) for the full design.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

## Data Pipeline

Fetch spot OHLCV (Kraken by default; Binance may be geo-restricted), compute technical indicators, and split into train/val/test:

```bash
python scripts/run_data_pipeline.py
```

Options:
- `--data-dir DATA_DIR` — Output directory (default: `data/`)
- `--years YEARS` — Years of history (default: 2)
- `--skip-fetch` — Use existing raw data, recompute indicators and split
- `--exchange EXCHANGE` — CCXT exchange (kraken, binance, kucoin, bybit, etc.)

Output structure:
```
data/
├── raw/           # OHLCV parquet files per symbol
└── processed/
    ├── train/
    ├── val/
    └── test/
```
