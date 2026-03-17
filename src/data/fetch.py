"""Fetch OHLCV data via CCXT."""

from pathlib import Path

import ccxt
import pandas as pd

from .config import EXCHANGE_ID, SYMBOLS, TIMEFRAME


def _get_exchange(exchange_id: str | None = None):
    """Create CCXT exchange instance."""
    exchange_id = exchange_id or EXCHANGE_ID
    cls = getattr(ccxt, exchange_id)
    return cls({"enableRateLimit": True})


def fetch_ohlcv(
    symbols: list[str] | None = None,
    timeframe: str | None = None,
    since: int | None = None,
    limit: int = 1000,
    exchange_id: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV candles for each symbol.

    Returns:
        Dict mapping symbol (e.g. "BTC/USDT") to DataFrame with columns:
        timestamp, open, high, low, close, volume
    """
    symbols = symbols or SYMBOLS
    timeframe = timeframe or TIMEFRAME
    exchange = _get_exchange(exchange_id)

    result: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        except Exception as e:
            print(f"Failed to fetch {symbol}: {e}")
            continue

        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        result[symbol] = df

    return result


def fetch_all_history(
    symbols: list[str] | None = None,
    timeframe: str | None = None,
    years: float = 2,
    exchange_id: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Fetch full history by paginating through CCXT (1000 candles per request).
    """
    symbols = symbols or SYMBOLS
    timeframe = timeframe or TIMEFRAME
    exchange = _get_exchange(exchange_id)

    # Approximate ms per candle
    tf_ms = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "1d": 86_400_000}
    ms_per_candle = tf_ms.get(timeframe, 3_600_000)

    result: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        all_ohlcv = []
        since = exchange.parse8601(
            (pd.Timestamp.utcnow() - pd.Timedelta(days=int(365 * years))).isoformat()
        )

        while True:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            except Exception as e:
                print(f"Failed to fetch {symbol} at {since}: {e}")
                break

            if not ohlcv:
                break

            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + ms_per_candle

            if len(ohlcv) < 1000:
                break

        if not all_ohlcv:
            continue

        df = pd.DataFrame(
            all_ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        df.set_index("timestamp", inplace=True)
        result[symbol] = df
        print(f"Fetched {symbol}: {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    return result


def save_raw(data: dict[str, pd.DataFrame], out_dir: Path | str) -> None:
    """Save raw OHLCV to Parquet files (one per symbol)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for symbol, df in data.items():
        safe_name = symbol.replace("/", "_")
        path = out_dir / f"{safe_name}.parquet"
        df.to_parquet(path, index=True)
        print(f"Saved {path}")


def load_raw(data_dir: Path | str, symbols: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Load raw OHLCV from Parquet files."""
    data_dir = Path(data_dir)
    symbols = symbols or SYMBOLS

    result: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        safe_name = symbol.replace("/", "_")
        path = data_dir / f"{safe_name}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index, utc=True)
            result[symbol] = df
    return result
