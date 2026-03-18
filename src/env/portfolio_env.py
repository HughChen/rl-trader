"""Gymnasium portfolio allocation environment."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np

from .aligned_data import INDICATOR_COLS, align_data
from .cost_models import fee_cost, slippage_cost


def _normalize_weights(
    x: np.ndarray,
    max_weight_per_asset: float | None = None,
) -> np.ndarray:
    """Clip to [0, 1] (or [0, max_weight_per_asset]) and normalize to sum to 1."""
    x = np.clip(x, 0, 1).astype(np.float64)
    if max_weight_per_asset is not None and max_weight_per_asset < 1.0:
        x = np.clip(x, 0, max_weight_per_asset)
    s = np.sum(x)
    if s < 1e-12:
        return np.ones_like(x) / len(x)
    return x / s


class PortfolioEnv(gym.Env):
    """
    Portfolio allocation environment for multi-asset spot trading.

    Observation: price returns (history_window × n_assets) + indicators (n_assets × n_ind) + prev weights (n_assets)
    Action: target portfolio weights (n_assets), env normalizes to sum=1
    Reward: log portfolio return minus transaction costs
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data: dict[str, pd.DataFrame],
        *,
        history_window: int = 50,
        fee_rate: float = 0.001,
        slippage_sigma: float = 0.1,
        notional_usd: float = 1e6,
        episode_length: int | None = None,
        warmup: int = 26,
        seed: int | None = None,
        reward_scale: float = 100.0,
        turnover_penalty: float = 0.0,
        max_weight_per_asset: float | None = 0.35,
        reward_type: str = "log_return",
        sharpe_window: int = 20,
        observation_type: str = "returns",
        diversification_lambda: float = 0.0,
    ):
        """
        Args:
            data: symbol -> DataFrame with OHLCV + indicators
            history_window: number of past return steps per asset
            fee_rate: transaction fee (e.g. 0.001 = 0.1%)
            slippage_sigma: market impact coefficient
            notional_usd: reference portfolio size (USD) for slippage scaling
            episode_length: if set, random sub-episodes; if None, full dataset
            warmup: rows to drop for indicator warmup
            seed: random seed
            reward_scale: scale factor for reward (stronger learning signal)
            turnover_penalty: extra penalty per unit turnover (discourages churning)
            max_weight_per_asset: cap per-asset weight (e.g. 0.35); None = no cap
            reward_type: "log_return" or "sharpe" (rolling Sharpe-like reward)
            sharpe_window: window size for rolling Sharpe when reward_type="sharpe"
            observation_type: "returns" (flat) or "pgportfolio" (Dict: price matrix + prev_w)
            diversification_lambda: PGPortfolio-style penalty for concentrated portfolios (default 0)
        """
        super().__init__()
        self.observation_type = observation_type
        self._aligned = align_data(data, warmup=warmup)
        self.history_window = history_window
        self.fee_rate = fee_rate
        self.slippage_sigma = slippage_sigma
        self.notional_usd = notional_usd
        self.episode_length = episode_length
        self.warmup = warmup

        n_assets = self._aligned.n_assets
        n_indicators = self._aligned.n_indicators

        if observation_type == "pgportfolio":
            # Dict: price (3, n_assets, H) close/high/low + prev_w (n_assets)
            self.observation_space = spaces.Dict({
                "price": spaces.Box(
                    low=0.0,
                    high=2.0,
                    shape=(3, n_assets, history_window),
                    dtype=np.float32,
                ),
                "prev_w": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(n_assets,),
                    dtype=np.float32,
                ),
            })
        else:
            obs_dim = (
                history_window * n_assets
                + n_assets * n_indicators
                + n_assets
            )
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32,
            )
        self.action_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(n_assets,),
            dtype=np.float32,
        )

        self.reward_scale = reward_scale
        self.turnover_penalty = turnover_penalty
        self.max_weight_per_asset = max_weight_per_asset
        self.reward_type = reward_type
        self.sharpe_window = sharpe_window
        self.diversification_lambda = diversification_lambda
        self._rng = np.random.default_rng(seed)
        self._reward_buffer: deque[float] = deque(maxlen=sharpe_window)
        self._t: int = 0
        self._start_t: int = 0
        self._w: np.ndarray = np.zeros(n_assets)
        self._portfolio_value: float = 1.0

    def _build_obs(self, t: int) -> np.ndarray:
        """Build observation for step t."""
        H = self.history_window
        n = self._aligned.n_assets

        if self.observation_type == "pgportfolio":
            # PGPortfolio: 3-channel (close, high, low) normalized by close_t
            start = max(0, t - H + 1)
            close = self._aligned.prices[start : t + 1]  # (H, n)
            high = self._aligned.prices_high[start : t + 1]
            low = self._aligned.prices_low[start : t + 1]
            if len(close) < H:
                pad_c = np.tile(close[0:1], (H - len(close), 1))
                pad_h = np.tile(high[0:1], (H - len(high), 1))
                pad_l = np.tile(low[0:1], (H - len(low), 1))
                close = np.concatenate([pad_c, close], axis=0)
                high = np.concatenate([pad_h, high], axis=0)
                low = np.concatenate([pad_l, low], axis=0)
            close_t = close[-1]
            close_t = np.where(close_t > 1e-12, close_t, 1.0)
            norm_close = (close / close_t).T.astype(np.float32)  # (n, H)
            norm_high = (high / close_t).T.astype(np.float32)
            norm_low = (low / close_t).T.astype(np.float32)
            price_obs = np.stack([norm_close, norm_high, norm_low], axis=0)  # (3, n, H)
            return {"price": price_obs, "prev_w": self._w.astype(np.float32)}

        # Default: returns + indicators + prev_weights
        n_ind = self._aligned.n_indicators
        start = max(0, t - H)
        returns = self._aligned.returns[start:t]
        if len(returns) < H:
            pad = np.zeros((H - len(returns), n), dtype=np.float32)
            returns = np.concatenate([pad, returns], axis=0)
        price_part = returns.flatten()
        ind_part = self._aligned.get_indicators(t).flatten()
        w_part = self._w.astype(np.float32)
        return np.concatenate([price_part, ind_part, w_part]).astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        T = self._aligned.n_steps
        H = self.history_window

        if self.episode_length is None:
            self._start_t = H  # Need at least H steps for observation
            self._t = self._start_t
            max_len = T - self._start_t - 1
        else:
            max_start = T - self.episode_length - H - 1
            max_start = max(max_start, H)
            self._start_t = int(self._rng.integers(H, max_start + 1))
            self._t = self._start_t
            max_len = self.episode_length

        # Equal-weight initial portfolio
        n = self._aligned.n_assets
        self._w = np.ones(n) / n
        self._portfolio_value = 1.0
        self._reward_buffer.clear()

        obs = self._build_obs(self._t)
        info = {
            "t": self._t,
            "portfolio_value": self._portfolio_value,
            "max_steps": max_len,
        }
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        w_target = _normalize_weights(
            np.asarray(action, dtype=np.float64),
            max_weight_per_asset=self.max_weight_per_asset,
        )
        t = self._t
        T = self._aligned.n_steps

        if t >= T - 1:
            obs = self._build_obs(t)
            return obs, 0.0, True, False, {"portfolio_value": self._portfolio_value}

        prices_prev = self._aligned.get_prices(t)
        prices_now = self._aligned.get_prices(t + 1)
        volumes_now = self._aligned.get_volumes(t + 1)

        # Portfolio value change from price move (before rebalancing)
        p_prev = self._portfolio_value
        returns = (prices_now / prices_prev) - 1
        p_before = p_prev * (1 + np.dot(self._w, returns))

        # Transaction costs
        fee = fee_cost(self._w, w_target, self.fee_rate)
        slippage = slippage_cost(
            self._w,
            w_target,
            p_before,
            prices_now,
            volumes_now,
            self.slippage_sigma,
            notional_usd=self.notional_usd,
        )
        cost_factor = 1.0 - fee - slippage
        p_after = p_before * cost_factor

        raw_reward = np.log(p_after / p_prev)
        turnover = np.sum(np.abs(w_target - self._w))

        div_penalty = 0.0
        if self.diversification_lambda > 0:
            # PGPortfolio: -log(1+1e-6 - w) penalizes concentration
            div_penalty = self.diversification_lambda * np.sum(
                -np.log(1e-6 + 1.0 - np.clip(w_target, 0, 1 - 1e-6))
            )

        if self.reward_type == "sharpe":
            self._reward_buffer.append(raw_reward)
            if len(self._reward_buffer) >= 2:
                arr = np.array(self._reward_buffer)
                mean_r = np.mean(arr)
                std_r = np.std(arr) + 1e-8
                sharpe_like = mean_r / std_r
            else:
                sharpe_like = raw_reward
            reward = (sharpe_like - self.turnover_penalty * turnover - div_penalty) * self.reward_scale
        else:
            reward = (raw_reward - self.turnover_penalty * turnover - div_penalty) * self.reward_scale
        self._portfolio_value = p_after
        self._w = w_target
        self._t = t + 1

        obs = self._build_obs(self._t)
        info = {
            "portfolio_value": self._portfolio_value,
            "fee": fee,
            "slippage": slippage,
            "raw_reward": raw_reward,
        }

        # Done when we've reached episode end
        if self.episode_length is None:
            done = self._t >= T - 1
        else:
            done = (self._t - self._start_t) >= self.episode_length or self._t >= T - 1

        truncated = False
        return obs, float(reward), done, truncated, info

    @property
    def n_assets(self) -> int:
        return self._aligned.n_assets

    @property
    def symbols(self) -> list[str]:
        return self._aligned.symbols


def make_env(
    data_dir: Path | str = "data/processed",
    split: str = "train",
    **kwargs,
) -> PortfolioEnv:
    """Create env from processed data directory."""
    from ..data.split import load_splits

    train, val, test = load_splits(data_dir)
    data = {"train": train, "val": val, "test": test}[split]
    if not data:
        raise FileNotFoundError(f"No data in {data_dir}/{split}")
    return PortfolioEnv(data, **kwargs)
