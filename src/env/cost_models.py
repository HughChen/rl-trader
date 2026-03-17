"""Transaction cost models: fees and slippage."""

import numpy as np


def fee_cost(
    w_prev: np.ndarray,
    w_target: np.ndarray,
    fee_rate: float,
) -> float:
    """
    Transaction fee as fraction of portfolio value.

    cost = fee_rate * sum_i |w_target[i] - w_prev[i]|

    Args:
        w_prev: Previous weights (n_assets,)
        w_target: Target weights (n_assets,)
        fee_rate: Fee per unit traded (e.g. 0.001 for 0.1%)

    Returns:
        Fraction of portfolio lost to fees
    """
    turnover = np.sum(np.abs(w_target - w_prev))
    return fee_rate * turnover


def slippage_cost(
    w_prev: np.ndarray,
    w_target: np.ndarray,
    portfolio_value: float,
    prices: np.ndarray,
    volumes: np.ndarray,
    sigma: float,
    min_volume: float = 1e-8,
) -> float:
    """
    Square-root market impact slippage (Almgren-Chriss style).

    Per asset i: impact_i = sigma * sqrt(trade_value_i / volume_quote_i)
    trade_value_i = |Δw_i| * portfolio_value
    volume_quote_i = volumes[i] * prices[i] (quote currency)

    Total cost = sum over assets of (impact_i * |Δw_i|) as fraction of portfolio.

    Args:
        w_prev: Previous weights
        w_target: Target weights
        portfolio_value: Current portfolio value
        prices: Asset prices (n_assets,)
        volumes: Volume in base currency (n_assets,) — we use volume*price for quote
        sigma: Impact coefficient
        min_volume: Floor to avoid div by zero

    Returns:
        Fraction of portfolio lost to slippage
    """
    dw = np.abs(w_target - w_prev)
    turnover = np.sum(dw)
    if turnover < 1e-12:
        return 0.0

    trade_values = dw * portfolio_value
    volume_quote = np.maximum(volumes * prices, min_volume)

    impact = sigma * np.sqrt(trade_values / volume_quote)
    impact = np.minimum(impact, 0.1)  # Cap per-asset impact at 10%
    total_cost = np.sum(impact * dw) / turnover
    return min(total_cost, 0.5)  # Cap total slippage at 50%
