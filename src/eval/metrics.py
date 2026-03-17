"""Portfolio evaluation metrics."""

import numpy as np


def equity_curve(rewards: np.ndarray) -> np.ndarray:
    """
    Cumulative portfolio value from log-return rewards.

    reward_t = log(p_t / p_{t-1}) => p_t = p_0 * exp(sum(rewards[:t]))
    """
    return np.exp(np.cumsum(rewards))


def compute_metrics(
    rewards: np.ndarray,
    periods_per_year: float = 252.0,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """
    Compute Sharpe, Sortino, max drawdown from log-return rewards.

    Args:
        rewards: Log returns per step
        periods_per_year: For annualization (252 for daily, 8760 for hourly)
        risk_free_rate: Annual risk-free rate (e.g. 0.05 for 5%)

    Returns:
        Dict with sharpe, sortino, max_drawdown, total_return, volatility
    """
    if len(rewards) < 2:
        return {
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "total_return": 0.0,
            "volatility": 0.0,
        }

    # Simple returns for volatility
    returns = np.exp(rewards) - 1
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    excess_ret = mean_ret - rf_per_period

    # Sharpe (annualized)
    sharpe = (
        np.sqrt(periods_per_year) * excess_ret / std_ret
        if std_ret > 1e-12
        else 0.0
    )

    # Sortino (downside deviation)
    downside = returns[returns < 0]
    downside_std = np.std(downside) if len(downside) > 0 else 1e-12
    sortino = (
        np.sqrt(periods_per_year) * excess_ret / downside_std
        if downside_std > 1e-12
        else 0.0
    )

    # Max drawdown
    curve = equity_curve(rewards)
    running_max = np.maximum.accumulate(curve)
    drawdowns = (curve - running_max) / running_max
    max_drawdown = np.min(drawdowns)

    total_return = curve[-1] - 1.0 if len(curve) > 0 else 0.0
    volatility = np.sqrt(periods_per_year) * std_ret if std_ret > 0 else 0.0

    return {
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_drawdown),
        "total_return": float(total_return),
        "volatility": float(volatility),
    }
