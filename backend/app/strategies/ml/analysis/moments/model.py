from typing import Tuple

import torch
import torch.nn as nn

from .encoder import DualStreamEncoder


class IntegratedMomentsModel(nn.Module):
    """
    Neural network that jointly predicts returns and volatility
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, lstm_layers: int = 2, dropout: float = 0.2):
        super().__init__()

        # LSTM for temporal dynamics
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=lstm_layers, batch_first=True, dropout=dropout, bidirectional=True
        )

        # Dual-stream encoder
        self.encoder = DualStreamEncoder(hidden_dim * 2, hidden_dim)

        # Mean prediction head
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim // 4 * 3, hidden_dim // 2),  # 3 streams combined
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Volatility prediction head (positive output via softplus)
        self.vol_head = nn.Sequential(
            nn.Linear(hidden_dim // 4 * 3, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # Ensures positive volatility
        )

        # Uncertainty estimation (Bayesian dropout approximation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            mean_pred: Predicted returns
            vol_pred: Predicted volatility
            uncertainty: Prediction uncertainty
        """
        # LSTM encoding
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # Take last timestep

        # Dual-stream encoding
        mean_feat, vol_feat, interaction = self.encoder(last_hidden)

        # Combine all features
        combined = torch.cat([mean_feat, vol_feat, interaction], dim=-1)

        # Predictions
        mean_pred = self.mean_head(combined)
        vol_pred = self.vol_head(combined)

        # Uncertainty via Monte Carlo dropout
        self.train()  # Enable dropout
        mc_predictions = []
        for _ in range(50):
            # Apply dropout manually
            dropped = self.dropout(combined)
            mc_mean = self.mean_head(dropped)
            mc_predictions.append(mc_mean.detach())

        mc_stack = torch.stack(mc_predictions)
        uncertainty = mc_stack.std(dim=0)

        self.eval()  # Back to eval mode

        return mean_pred, vol_pred, uncertainty
