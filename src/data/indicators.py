"""Compute technical indicators for RL observation space."""

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange, BollingerBands


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add RSI, MACD, Bollinger Bands, and ATR to OHLCV DataFrame.

    Expects columns: open, high, low, close, volume
    """
    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]

    # RSI (14)
    out["rsi"] = RSIIndicator(close=close, window=14).rsi()

    # MACD (12, 26, 9)
    macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_diff"] = macd.macd_diff()

    # Bollinger Bands (20, 2)
    bb = BollingerBands(close=close, window=20, window_dev=2)
    out["bb_upper"] = bb.bollinger_hband()
    out["bb_middle"] = bb.bollinger_mavg()
    out["bb_lower"] = bb.bollinger_lband()

    # ATR (14)
    out["atr"] = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()

    # Relative volume: volume / rolling mean of volume
    out["rel_volume"] = out["volume"] / out["volume"].rolling(24).mean()

    return out


def add_indicators_batch(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Add indicators to each symbol's DataFrame."""
    return {symbol: add_indicators(df) for symbol, df in data.items()}
