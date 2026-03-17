"""Cost attribution: replay trajectories with different fee/slippage to isolate impact."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.env.aligned_data import AlignedMarketData, align_data

if TYPE_CHECKING:
    pass
from src.env.portfolio_env import _normalize_weights
from src.env.cost_models import fee_cost, slippage_cost


@dataclass
class ReplayResult:
    """Result of replaying a trajectory with given cost params."""

    final_value: float
    total_return: float
    cumulative_fee_pct: float  # Sum of fee fractions (not compounded)
    cumulative_slippage_pct: float
    n_steps: int


def replay_trajectory(
    aligned: AlignedMarketData,
    start_t: int,
    actions: list[np.ndarray],
    *,
    fee_rate: float = 0.001,
    slippage_sigma: float = 0.1,
    notional_usd: float = 1e6,
    max_weight_per_asset: float | None = 0.35,
) -> ReplayResult:
    """
    Replay a recorded action sequence with given cost parameters.

    Args:
        aligned: Aligned market data
        start_t: Starting step index
        actions: List of target weight arrays (one per step)
        fee_rate: Transaction fee rate (0 = no fees)
        slippage_sigma: Slippage coefficient (0 = no slippage)
        notional_usd: Reference portfolio size for slippage scaling
        max_weight_per_asset: Weight cap for normalization

    Returns:
        ReplayResult with final value, return, and cumulative cost breakdown
    """
    n_assets = aligned.n_assets
    T = aligned.n_steps

    w = np.ones(n_assets) / n_assets
    p = 1.0
    cum_fee = 0.0
    cum_slippage = 0.0

    for i, action in enumerate(actions):
        t = start_t + i
        if t >= T - 1:
            break

        w_target = _normalize_weights(
            np.asarray(action, dtype=np.float64),
            max_weight_per_asset=max_weight_per_asset,
        )

        prices_prev = aligned.get_prices(t)
        prices_now = aligned.get_prices(t + 1)
        volumes_now = aligned.get_volumes(t + 1)

        returns = (prices_now / prices_prev) - 1
        p_before = p * (1 + np.dot(w, returns))

        fee = fee_cost(w, w_target, fee_rate) if fee_rate > 0 else 0.0
        slippage = (
            slippage_cost(
                w, w_target, p_before, prices_now, volumes_now, slippage_sigma,
                notional_usd=notional_usd,
            )
            if slippage_sigma > 0
            else 0.0
        )

        cost_factor = 1.0 - fee - slippage
        p_after = p_before * cost_factor

        cum_fee += fee
        cum_slippage += slippage
        p = p_after
        w = w_target

    total_return = p - 1.0
    return ReplayResult(
        final_value=p,
        total_return=total_return,
        cumulative_fee_pct=cum_fee,
        cumulative_slippage_pct=cum_slippage,
        n_steps=len(actions),
    )


def replay_from_data(
    data: dict[str, pd.DataFrame],
    start_t: int,
    actions: list[np.ndarray],
    *,
    fee_rate: float = 0.001,
    slippage_sigma: float = 0.1,
    notional_usd: float = 1e6,
    max_weight_per_asset: float | None = 0.35,
    warmup: int = 26,
) -> ReplayResult:
    """Replay trajectory from data dict (aligns internally)."""
    aligned = align_data(data, warmup=warmup)
    return replay_trajectory(
        aligned,
        start_t,
        actions,
        fee_rate=fee_rate,
        slippage_sigma=slippage_sigma,
        notional_usd=notional_usd,
        max_weight_per_asset=max_weight_per_asset,
    )
