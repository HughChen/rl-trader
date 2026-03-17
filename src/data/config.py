"""Data pipeline configuration."""

# Exchange: binance, kraken, kucoin, bybit, etc. (Binance may be geo-restricted)
EXCHANGE_ID = "kraken"

# Top liquid spot pairs (quote: USDT on most exchanges; Kraken uses USD)
# Kraken uses USD pairs; most others use USDT
SYMBOLS_BY_EXCHANGE = {
    "binance": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT"],
    "kraken": ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "ADA/USD", "DOGE/USD", "AVAX/USD", "LINK/USD", "DOT/USD"],  # BNB not on Kraken
    "kucoin": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT"],
    "bybit": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT"],
}
SYMBOLS = SYMBOLS_BY_EXCHANGE.get(EXCHANGE_ID, SYMBOLS_BY_EXCHANGE["kraken"])

# Bar frequency: 1h for flexibility (can resample to 30m if needed)
TIMEFRAME = "1h"

# Train/validation/test split (approximate dates)
TRAIN_END = "2024-12-31"
VAL_END = "2025-06-30"
# Test: after VAL_END
