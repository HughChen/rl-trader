"""PGPortfolio-style CNN feature extractor for Dict observation (price + prev_w)."""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PortfolioDictExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for PGPortfolio-style Dict observation.

    Dict keys:
      - price: (3, n_assets, history_window) - close, high, low normalized by close_t
      - prev_w: (n_assets,) - previous portfolio weights

    Uses CNN on price matrix, concatenates prev_w, outputs combined features.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 128,
    ):
        if not isinstance(observation_space, spaces.Dict):
            raise ValueError("Expected Dict observation space")

        price_space = observation_space.spaces["price"]
        prev_w_space = observation_space.spaces["prev_w"]

        if len(price_space.shape) != 3:
            raise ValueError("Expected price shape (C, H, W)")

        C, H, W = price_space.shape
        n_prev_w = prev_w_space.shape[0]

        # Compute CNN output size
        super().__init__(observation_space, features_dim)
        self._price_cnn = nn.Sequential(
            nn.Conv2d(C, 16, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.ReLU(),
        )

        with torch.no_grad():
            sample = torch.as_tensor(price_space.sample()[None]).float()
            cnn_out = self._price_cnn(sample)
            cnn_flatten_dim = cnn_out.numel()

        # CNN features + prev_w -> features_dim
        combined_dim = cnn_flatten_dim + n_prev_w
        self._fc = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
        )
        self._cnn_flatten_dim = cnn_flatten_dim

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        price = observations["price"]
        prev_w = observations["prev_w"]

        cnn_out = self._price_cnn(price)
        cnn_flat = cnn_out.view(cnn_out.size(0), -1)
        combined = torch.cat([cnn_flat, prev_w], dim=1)
        return self._fc(combined)


# Backward compatibility alias
PortfolioCNNExtractor = PortfolioDictExtractor
