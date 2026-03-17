"""Aligned market data for portfolio environment."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


# Indicator columns from our data pipeline (order matters for observation space)
INDICATOR_COLS = [
    "rsi",
    "macd",
    "macd_signal",
    "macd_diff",
    "bb_position",  # (close - lower) / (upper - lower)
    "atr",
    "rel_volume",
]


@dataclass
class AlignedMarketData:
    """
    Aligned market data indexed by step.

    Attributes:
        prices: (T, n_assets) close prices
        volumes: (T, n_assets) volume in quote currency
        returns: (T, n_assets) simple return = (close_t / close_{t-1}) - 1
        indicators: (T, n_assets, n_indicators) normalized indicator values
        symbols: list of symbol names (asset order)
        timestamps: (T,) timestamps for each step
    """

    prices: np.ndarray  # (T, n_assets)
    volumes: np.ndarray  # (T, n_assets)
    returns: np.ndarray  # (T, n_assets)
    indicators: np.ndarray  # (T, n_assets, n_indicators)
    symbols: list[str]
    timestamps: np.ndarray  # (T,)

    @property
    def n_steps(self) -> int:
        return self.prices.shape[0]

    @property
    def n_assets(self) -> int:
        return self.prices.shape[1]

    @property
    def n_indicators(self) -> int:
        return self.indicators.shape[2]

    def get_prices(self, t: int) -> np.ndarray:
        return self.prices[t]

    def get_volumes(self, t: int) -> np.ndarray:
        return self.volumes[t]

    def get_returns(self, t: int) -> np.ndarray:
        return self.returns[t]

    def get_indicators(self, t: int) -> np.ndarray:
        return self.indicators[t]


def _normalize_indicator(arr: np.ndarray, col: str) -> np.ndarray:
    """Clip and normalize indicator to roughly [0, 1] for stability."""
    out = np.nan_to_num(arr, nan=0.5, posinf=1.0, neginf=0.0)
    if col == "rsi":
        out = np.clip(out / 100.0, 0, 1)
    elif col in ("macd", "macd_signal", "macd_diff", "atr"):
        p99 = np.nanpercentile(np.abs(arr[arr != 0]), 99) if np.any(arr != 0) else 1.0
        scale = p99 + 1e-8
        out = np.clip(arr / scale, -1, 1)
        out = (out + 1) / 2  # to [0, 1]
    elif col == "rel_volume":
        out = np.clip(out, 0, 3) / 3  # cap at 3x avg
    return out


def align_data(
    data: dict[str, pd.DataFrame],
    indicator_cols: list[str] | None = None,
    warmup: int = 26,
) -> AlignedMarketData:
    """
    Align per-symbol DataFrames into step-indexed arrays.

    Args:
        data: symbol -> DataFrame with OHLCV + indicator columns
        indicator_cols: columns to use as features (default: INDICATOR_COLS)
        warmup: drop first N rows (indicator warmup)

    Returns:
        AlignedMarketData with prices, volumes, returns, indicators
    """
    indicator_cols = indicator_cols or INDICATOR_COLS
    symbols = sorted(data.keys())

    # Align on common timestamps
    dfs = [data[s].sort_index() for s in symbols]
    common_idx = dfs[0].index
    for df in dfs[1:]:
        common_idx = common_idx.intersection(df.index)

    if len(common_idx) < warmup + 2:
        raise ValueError(
            f"Not enough aligned rows: {len(common_idx)} after alignment, need {warmup + 2}"
        )

    # Build aligned DataFrames
    n_steps = len(common_idx)
    n_assets = len(symbols)

    prices = np.zeros((n_steps, n_assets), dtype=np.float64)
    volumes = np.zeros((n_steps, n_assets), dtype=np.float64)

    for i, sym in enumerate(symbols):
        df = data[sym].loc[common_idx]
        prices[:, i] = df["close"].values
        volumes[:, i] = df["volume"].values

    # Returns: r_t = (close_t / close_{t-1}) - 1
    returns = np.zeros_like(prices)
    returns[1:] = (prices[1:] / prices[:-1]) - 1
    returns[0] = 0

    # Drop warmup
    prices = prices[warmup:]
    volumes = volumes[warmup:]
    returns = returns[warmup:]
    timestamps = common_idx[warmup:].values

    n_steps = len(timestamps)

    # Indicators: (T, n_assets, n_indicators)
    n_ind = len(indicator_cols)
    indicators = np.zeros((n_steps, n_assets, n_ind), dtype=np.float32)

    for i, sym in enumerate(symbols):
        df = data[sym].loc[common_idx].iloc[warmup:]
        for j, col in enumerate(indicator_cols):
            if col == "bb_position":
                close = df["close"].values
                if "bb_upper" in df.columns and "bb_lower" in df.columns:
                    upper = df["bb_upper"].values
                    lower = df["bb_lower"].values
                else:
                    indicators[:, i, j] = 0.5
                    continue
                span = upper - lower
                span = np.where(span > 1e-8, span, 1e-8)
                bb_pos = (close - lower) / span
                indicators[:, i, j] = np.clip(bb_pos, 0, 1).astype(np.float32)
            elif col not in df.columns:
                indicators[:, i, j] = 0.5  # neutral
            else:
                arr = _normalize_indicator(df[col].values, col)
                indicators[:, i, j] = arr.astype(np.float32)

    return AlignedMarketData(
        prices=prices,
        volumes=volumes,
        returns=returns,
        indicators=indicators,
        symbols=symbols,
        timestamps=timestamps,
    )
