#!/usr/bin/env python3
"""
Run the full data pipeline: fetch -> indicators -> split -> save.

Usage:
    python scripts/run_data_pipeline.py [--data-dir DATA_DIR] [--years YEARS]
"""

import argparse
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.config import EXCHANGE_ID, SYMBOLS
from src.data.fetch import fetch_all_history, load_raw, save_raw
from src.data.indicators import add_indicators_batch
from src.data.split import save_splits, split_by_date


def main():
    parser = argparse.ArgumentParser(description="RL Trader data pipeline")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory for raw and processed data",
    )
    parser.add_argument(
        "--years",
        type=float,
        default=2,
        help="Years of history to fetch",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip fetch; use existing raw data",
    )
    parser.add_argument(
        "--exchange",
        type=str,
        default=None,
        help="CCXT exchange (default: from config; use kraken if binance is geo-restricted)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"

    # 1. Fetch
    first_symbol_safe = SYMBOLS[0].replace("/", "_") if SYMBOLS else "BTC_USD"
    if args.skip_fetch and (raw_dir / f"{first_symbol_safe}.parquet").exists():
        print("Loading existing raw data...")
        data = load_raw(raw_dir, SYMBOLS)
    else:
        exchange = args.exchange or EXCHANGE_ID
        print(f"Fetching OHLCV from {exchange}...")
        data = fetch_all_history(symbols=SYMBOLS, years=args.years, exchange_id=exchange)
        if not data:
            print("No data fetched. Exiting.")
            return 1
        save_raw(data, raw_dir)

    # 2. Indicators
    print("Computing technical indicators...")
    data = add_indicators_batch(data)

    # 3. Split
    print("Splitting train/val/test...")
    train, val, test = split_by_date(data)
    save_splits(train, val, test, processed_dir)

    # Summary
    n_train = min(len(df) for df in train.values()) if train else 0
    n_val = min(len(df) for df in val.values()) if val else 0
    n_test = min(len(df) for df in test.values()) if test else 0
    print(f"\nDone. Train: {n_train} bars, Val: {n_val} bars, Test: {n_test} bars")
    print(f"Data saved to {processed_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
