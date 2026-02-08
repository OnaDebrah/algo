from typing import Any, Dict, Optional

from core.trade.order.order import Order, OrderSide, OrderType, TimeInForce


class TrailingStopOrder(Order):
    """Trailing stop - stop price trails market price by specified amount"""

    def __init__(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        trail_amount: Optional[float] = None,
        trail_percent: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
    ):
        super().__init__(symbol, side, quantity, OrderType.TRAILING_STOP, time_in_force)

        if trail_amount is None and trail_percent is None:
            raise ValueError("Must specify either trail_amount or trail_percent")

        self.trail_amount = trail_amount
        self.trail_percent = trail_percent
        self.stop_price: Optional[float] = None
        self.high_water_mark: Optional[float] = None  # For sell
        self.low_water_mark: Optional[float] = None  # For buy
        self.triggered = False

    def update(self, current_price: float):
        """Update trailing stop based on current price"""
        if self.side == OrderSide.SELL:
            # Sell trailing stop - trail below high
            if self.high_water_mark is None or current_price > self.high_water_mark:
                self.high_water_mark = current_price

                # Update stop price
                if self.trail_amount:
                    self.stop_price = self.high_water_mark - self.trail_amount
                else:
                    self.stop_price = self.high_water_mark * (1 - self.trail_percent / 100)

            # Check if triggered
            if self.stop_price and current_price <= self.stop_price:
                self.triggered = True

        else:  # BUY
            # Buy trailing stop - trail above low
            if self.low_water_mark is None or current_price < self.low_water_mark:
                self.low_water_mark = current_price

                # Update stop price
                if self.trail_amount:
                    self.stop_price = self.low_water_mark + self.trail_amount
                else:
                    self.stop_price = self.low_water_mark * (1 + self.trail_percent / 100)

            # Check if triggered
            if self.stop_price and current_price >= self.stop_price:
                self.triggered = True

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["trail_amount"] = self.trail_amount
        data["trail_percent"] = self.trail_percent
        data["stop_price"] = self.stop_price
        data["high_water_mark"] = self.high_water_mark
        data["low_water_mark"] = self.low_water_mark
        data["triggered"] = self.triggered
        return data
