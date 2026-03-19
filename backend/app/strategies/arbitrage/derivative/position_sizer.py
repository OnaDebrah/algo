from typing import Dict


class PositionSizer:
    """Calculates position sizes based on risk parameters"""

    def __init__(self, risk_per_trade: float = 0.02, max_position_size: float = 0.1, use_vega_sizing: bool = True):
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.use_vega_sizing = use_vega_sizing

    def calculate_size(self, opportunity: Dict, capital: float, greeks: Dict = None) -> float:
        """Calculate position size"""

        if self.use_vega_sizing and greeks and "vega" in greeks:
            # Vega-based sizing
            confidence = opportunity.get("confidence", 1.0)
            vega = abs(greeks["vega"])

            if vega > 0:
                target_risk = capital * self.risk_per_trade * confidence
                size = target_risk / vega
            else:
                size = self._notional_sizing(opportunity, capital)
        else:
            size = self._notional_sizing(opportunity, capital)

        # Apply maximum constraint
        max_notional = capital * self.max_position_size
        option_price = opportunity.get("entry_price", opportunity.get("call_price", opportunity.get("put_price", 1)))

        if option_price > 0:
            max_contracts = max_notional / (option_price * 100)
            size = min(size, max_contracts)

        return max(size, 0)

    def _notional_sizing(self, opportunity: Dict, capital: float) -> float:
        """Simple notional-based sizing"""
        confidence = opportunity.get("confidence", 1.0)
        option_price = opportunity.get("entry_price", opportunity.get("call_price", opportunity.get("put_price", 1)))

        target_notional = capital * self.risk_per_trade * confidence * 10
        return target_notional / (option_price * 100)
