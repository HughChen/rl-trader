# Experiment 005: Exchange Restrictions (CCXT)

## Setup

| Parameter | Value |
|-----------|-------|
| **Tool** | CCXT fetch_ohlcv |
| **Tested** | Kraken, Binance, KuCoin |
| **Requested** | 2 years, 1h bars |
| **Command** | `python scripts/run_data_pipeline.py --exchange {kraken,binance,kucoin} --years 2` |

---

## Findings

| Exchange | 1h history | Notes |
|----------|------------|-------|
| **KuCoin** | ~2 years | ✅ Paginates correctly. Use for RL. |
| **Kraken** | ~1 month | 720-candle API limit; `since` ignored beyond that. |
| **Binance** | Years | Geo-restricted (451) in US and some regions. |

**Recommendation**: Use KuCoin for 2+ years of data.
