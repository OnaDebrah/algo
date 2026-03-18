from typing import Dict, List

import numpy as np
import pandas as pd

from ....strategies.arbitrage.derivative.arbitrage_detector import ArbitrageDetector
from ....strategies.arbitrage.derivative.arbitrage_strategy import ArbitrageStrategy
from ....strategies.arbitrage.derivative.greek_calculator import GreekCalculator
from ....strategies.arbitrage.derivative.hedge_manager import HedgeManager
from ....strategies.arbitrage.derivative.position_sizer import PositionSizer
from ....strategies.arbitrage.derivative.volatility_forecaster import VolatilityForecaster
from ....strategies.arbitrage.derivative.volatility_surface import VolatilitySurface


class DerivativesArbitrage(ArbitrageStrategy):
    """
    Derivatives Arbitrage Strategy with modular components
    """

    def __init__(
        self,
        universe: List[str],
        arb_types: List[str] = None,
        volatility_model: str = "garch",
        risk_per_trade: float = 0.02,
        enable_hedging: bool = True,
        **kwargs,
    ):
        """
        Initialize Derivatives Arbitrage Strategy
        """
        super().__init__("Derivatives Arbitrage Strategy", kwargs, universe)

        self.capital = None
        if arb_types is None:
            arb_types = ["volatility", "put_call", "term_structure", "skew"]

        self.vol_surface = VolatilitySurface(
            min_moneyness=kwargs.get("min_moneyness", 0.8),
            max_moneyness=kwargs.get("max_moneyness", 1.2),
            min_dte=kwargs.get("min_dte", 7),
            max_dte=kwargs.get("max_dte", 180),
        )

        self.vol_forecaster = VolatilityForecaster(model=volatility_model, lookback_period=kwargs.get("lookback_period", 60))

        self.greek_calculator = GreekCalculator(risk_free_rate=kwargs.get("risk_free_rate", 0.03))

        self.arb_detector = ArbitrageDetector(entry_threshold=kwargs.get("entry_threshold", 2.0), min_liquidity=kwargs.get("min_liquidity", 1.0))

        self.position_sizer = PositionSizer(
            risk_per_trade=risk_per_trade, max_position_size=kwargs.get("max_position_size", 0.1), use_vega_sizing=kwargs.get("use_vega_sizing", True)
        )

        self.hedge_manager = HedgeManager(enabled=enable_hedging)

        self.arb_types = arb_types

        self.opportunities = []
        self.greek_exposure = {}

    def generate_signal(self, market_data: Dict) -> Dict[str, Dict]:
        """
        Generate trading signals using modular components
        """
        signals = {}

        spots = market_data.get("spots", pd.Series())
        option_chains = market_data.get("options", {})
        historical = market_data.get("historical", {})

        for asset in self.universe:
            if asset not in option_chains:
                continue

            if asset in historical and not historical[asset].empty:
                self.vol_forecaster.forecast(historical[asset])
            else:
                continue

            option_chain = option_chains[asset]
            if option_chain.empty:
                continue

            spot = spots.get(asset, 0)
            if spot <= 0:
                continue

            # Build volatility surface
            self.vol_surface.build_surface(asset, option_chain, spot)

            # Detect arbitrage opportunities
            opportunities = self.arb_detector.detect_all(asset, option_chain, spot, self.vol_surface, self.vol_forecaster, self.greek_calculator)

            # Filter by arbitrage type
            opportunities = [o for o in opportunities if o["type"] in self.arb_types]

            for opp in opportunities:
                trade_id = f"{asset}_{opp['type']}_{pd.Timestamp.now().timestamp()}"

                greeks = self.greek_calculator.calculate_all(
                    spot, opp["strike"], opp["dte"] / 365, opp.get("current_iv", 0.3), opp.get("option_type", "call")
                )

                # Calculate position size
                size = self.position_sizer.calculate_size(opp, self.capital, greeks)

                signal = {
                    "asset": asset,
                    "opportunity": opp,
                    "signal": opp["direction"],
                    "confidence": opp["confidence"],
                    "size": size,
                    "greeks": greeks,
                    "entry_price": opp.get("entry_price", opp.get("call_price", opp.get("put_price", 0))),
                }

                signals[trade_id] = signal
                self.opportunities.append(opp)

        return signals

    def _update_single_position(self, position: Dict, market_data: Dict) -> Dict:
        """Update a single position with new market data"""

        # Update current price (simplified - would get from market)
        position["current_price"] = position["entry_price"] * (1 + np.random.normal(0, 0.01))

        # Calculate P&L
        position["pnl"] = (position["current_price"] - position["entry_price"]) * position["size"] * position["signal"]

        # Check exit conditions
        self._check_exit_conditions(position)

        return position

    def _check_exit_conditions(self, position: Dict):
        """Check exit conditions for a position"""
        if position["status"] != "open":
            return

        signal = position["signal"]
        current_price = position["current_price"]
        entry_price = position["entry_price"]

        # Simple stop loss / take profit
        if signal > 0:  # Long
            if current_price <= entry_price * 0.9:
                position["status"] = "closed_stop_loss"
            elif current_price >= entry_price * 1.2:
                position["status"] = "closed_take_profit"
        else:  # Short
            if current_price >= entry_price * 1.1:
                position["status"] = "closed_stop_loss"
            elif current_price <= entry_price * 0.8:
                position["status"] = "closed_take_profit"

        # Check expiration
        if position.get("dte", 30) <= 1:
            position["status"] = "closed_expired"

        if position["status"] != "open":
            position["exit_time"] = pd.Timestamp.now()

    def get_greek_exposure(self) -> Dict[str, float]:
        """Get aggregate Greek exposure"""
        self.greek_exposure = self.hedge_manager.aggregate_exposure(self.active_positions)
        return self.greek_exposure
