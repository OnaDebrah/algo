from typing import Dict

from .greek_calculator import GreekCalculator


class HedgeManager:
    """Manages hedging of Greek exposures"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.greek_calculator = GreekCalculator()
        self.hedge_positions = {}

    def calculate_hedges(self, position: Dict, market_data: Dict) -> Dict:
        """Calculate hedge ratios for a position"""
        if not self.enabled:
            return {}

        S = market_data.get("spot", 0)
        K = position.get("strike", 0)
        T = max(position.get("dte", 30) / 365, 0.001)
        sigma = position.get("iv", 0.3)

        option_type = position.get("option_type", "call")

        greeks = self.greek_calculator.calculate_all(S, K, T, sigma, option_type)
        position_size = position.get("size", 0)

        hedges = {
            "delta_hedge": -greeks["delta"] * position_size,
            "gamma_hedge": -greeks["gamma"] * position_size,
            "vega_hedge": -greeks["vega"] * position_size,
            "theta_hedge": -greeks["theta"] * position_size,
        }

        return hedges

    def aggregate_exposure(self, positions: Dict[str, Dict]) -> Dict[str, float]:
        """Aggregate Greek exposure across all positions"""
        total = {"delta": 0, "gamma": 0, "vega": 0, "theta": 0, "rho": 0}

        for position in positions.values():
            if position.get("status") == "open":
                greeks = position.get("greeks", {})
                size = position.get("size", 0) * position.get("direction", 1)

                for greek in total.keys():
                    total[greek] += greeks.get(greek, 0) * size

        return total
