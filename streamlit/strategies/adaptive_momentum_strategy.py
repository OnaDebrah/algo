from abc import ABC

import pandas as pd

from backend.app.strategies.cs_momentum_strategy import CrossSectionalMomentumStrategy


class AdaptiveMomentumStrategy(CrossSectionalMomentumStrategy, ABC):
    """
    Adaptive Momentum Strategy

    Dynamically adjusts momentum parameters based on market conditions
    """

    def __init__(self, regime_detector, **kwargs):
        super().__init__(**kwargs)
        self.regime_detector = regime_detector

    def detect_market_regime(self, market_data: pd.Series) -> str:
        """
        Detect current market regime
        """
        return self.regime_detector.detect_regime(market_data)

    def adapt_parameters(self, regime: str):
        """
        Adjust momentum parameters based on regime
        """
        regime_params = {
            "trending": {
                "formation_period": 126,  # Shorter in trends
                "skip_period": 0,  # No skip in trends
                "top_quantile": 0.2,  # More concentrated
                "holding_period": 10,  # More frequent rebalancing
            },
            "mean_reverting": {
                "formation_period": 252,  # Longer lookback
                "skip_period": 21,  # Skip to avoid reversal
                "top_quantile": 0.3,  # Standard
                "holding_period": 21,  # Monthly
            },
            "high_vol": {
                "formation_period": 63,  # Shorter in high vol
                "skip_period": 5,  # Small skip
                "top_quantile": 0.4,  # Broader selection
                "holding_period": 5,  # Very frequent
            },
            "low_vol": {
                "formation_period": 252,  # Full year
                "skip_period": 21,  # Standard skip
                "top_quantile": 0.3,  # Standard
                "holding_period": 21,  # Monthly
            },
        }

        if regime in regime_params:
            for param, value in regime_params[regime].items():
                setattr(self, param, value)
