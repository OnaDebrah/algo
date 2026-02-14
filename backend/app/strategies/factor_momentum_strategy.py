from typing import Dict

import pandas as pd

from backend.app.strategies.cs_momentum_strategy import CrossSectionalMomentumStrategy


class FactorMomentumStrategy(CrossSectionalMomentumStrategy):
    """
    Factor Momentum Strategy

    Applies cross-sectional momentum to factor portfolios
    rather than individual stocks
    """

    def __init__(self, factor_model, **kwargs):
        universe = kwargs.get("universe", [])
        super().__init__(universe=universe, **kwargs)
        self.factor_model = factor_model

    def generate_factor_momentum_signals(self, stock_data: pd.DataFrame) -> Dict:
        """
        Generate momentum signals for factors
        """
        # Extract factor returns
        factor_returns = self.factor_model.calculate_factor_returns(stock_data)

        # Apply cross-sectional momentum to factors
        factor_signals = self.generate_signal(factor_returns)

        # Map factor signals back to stocks
        stock_signals = self._map_factors_to_stocks(factor_signals, stock_data)

        return stock_signals

    def _map_factors_to_stocks(self, factor_signals: Dict, stock_data: pd.DataFrame) -> Dict:
        """
        Map factor momentum signals to individual stocks
        """
        # Get factor exposures
        factor_exposures = self.factor_model.get_factor_exposures(stock_data)

        # Weight stocks by their factor exposures and factor momentum
        stock_weights = {}

        for stock in stock_data.columns:
            weight = 0.0
            for factor, factor_signal in factor_signals["signals"].items():
                exposure = factor_exposures.get(stock, {}).get(factor, 0.0)
                weight += exposure * factor_signal

            stock_weights[stock] = weight

        # Normalize
        total_weight = sum(abs(w) for w in stock_weights.values())
        if total_weight > 0:
            stock_weights = {k: v / total_weight for k, v in stock_weights.items()}

        return stock_weights
