"""Train/validation/test split by date."""

from pathlib import Path

import pandas as pd

from .config import SYMBOLS, TRAIN_END, VAL_END


def split_by_date(
    data: dict[str, pd.DataFrame],
    train_end: str = TRAIN_END,
    val_end: str = VAL_END,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """
    Split data into train, validation, and test sets by date.

    Returns:
        (train, validation, test) - each a dict of symbol -> DataFrame
    """
    train_end = pd.Timestamp(train_end, tz="UTC")
    val_end = pd.Timestamp(val_end, tz="UTC")

    train = {}
    val = {}
    test = {}

    for symbol, df in data.items():
        df = df.sort_index()
        train[symbol] = df[df.index <= train_end].copy()
        val[symbol] = df[(df.index > train_end) & (df.index <= val_end)].copy()
        test[symbol] = df[df.index > val_end].copy()

    return train, val, test


def save_splits(
    train: dict[str, pd.DataFrame],
    val: dict[str, pd.DataFrame],
    test: dict[str, pd.DataFrame],
    out_dir: Path | str,
) -> None:
    """Save train/val/test splits to Parquet files."""
    out_dir = Path(out_dir)
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for symbol, df in split_data.items():
            safe_name = symbol.replace("/", "_")
            df.to_parquet(split_dir / f"{safe_name}.parquet", index=True)


def load_full_data(
    data_dir: Path | str,
    symbols: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Load and merge train/val/test into full per-symbol DataFrames.
    Used for walk-forward evaluation with custom date splits.
    """
    train, val, test = load_splits(data_dir, symbols=symbols)
    full = {}
    for symbol in set(train) | set(val) | set(test):
        dfs = [d[symbol] for d in (train, val, test) if symbol in d and len(d[symbol]) > 0]
        if dfs:
            full[symbol] = pd.concat(dfs).sort_index().drop_duplicates()
    return full


def load_splits(
    data_dir: Path | str,
    symbols: list[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Load train/val/test splits from disk. Uses config SYMBOLS by default."""
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    symbols = symbols or SYMBOLS
    # Fallback: discover from disk if config symbols not found
    if not symbols and train_dir.exists():
        symbols = [p.stem.replace("_", "/") for p in train_dir.glob("*.parquet")]

    def load_split(name: str) -> dict[str, pd.DataFrame]:
        result = {}
        split_dir = data_dir / name
        if not split_dir.exists():
            return result
        for symbol in symbols:
            safe_name = symbol.replace("/", "_")
            path = split_dir / f"{safe_name}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                df.index = pd.to_datetime(df.index, utc=True)
                result[symbol] = df
        return result

    return load_split("train"), load_split("val"), load_split("test")
