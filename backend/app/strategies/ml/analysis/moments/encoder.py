from typing import Tuple

import torch
import torch.nn as nn


class DualStreamEncoder(nn.Module):
    """
    Encoder that processes two streams: mean and volatility
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(0.1))

        # Mean-specific stream
        self.mean_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )

        # Volatility-specific stream
        self.vol_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )

        # Interaction layer (captures mean-vol correlation)
        self.interaction = nn.Sequential(nn.Linear(hidden_dim // 2, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, hidden_dim // 4))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Shared features
        shared = self.shared(x)

        # Separate streams
        mean_features = self.mean_stream(shared)
        vol_features = self.vol_stream(shared)

        # Interaction features (concatenate and process)
        combined = torch.cat([mean_features, vol_features], dim=-1)
        interaction = self.interaction(combined)

        return mean_features, vol_features, interaction
