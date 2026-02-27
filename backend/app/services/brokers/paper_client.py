import logging
import random
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional

import pytz
from sqlalchemy.ext.asyncio import AsyncSession

from ...models import User, UserSettings
from ...services.brokers.base_client import BrokerClient

logger = logging.getLogger(__name__)


class PaperTradingClient(BrokerClient):
    """
    Paper trading implementation
    Uses real market data but simulates trades
    """

    def __init__(self):
        self.connected = False
        self.cash = 100000.0  # Starting cash
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.orders: Dict[str, Dict[str, Any]] = {}
        self.order_counter = 0

        # Mock market data cache
        self.market_data_cache: Dict[str, Dict[str, List[float]]] = {}

    async def connect(self, settings: UserSettings) -> bool:
        """Connect to paper trading"""
        logger.info("Connecting to paper trading")
        self.connected = True

        if settings.initial_capital is not None:
            self.cash = float(settings.initial_capital)

        return True

    async def disconnect(self):
        """Disconnect from paper trading"""
        logger.info("Disconnecting from paper trading")
        self.connected = False

    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        positions_value = sum(pos["quantity"] * pos["current_price"] for pos in self.positions.values())

        # Calculate unrealized P&L
        unrealized_pnl = sum((pos["current_price"] - pos["entry_price"]) * pos["quantity"] for pos in self.positions.values())

        equity = self.cash + positions_value

        # Simple margin calculation (positions value as margin used)
        margin_used = positions_value * 0.5  # Assume 50% margin requirement

        return {
            "cash": self.cash,
            "equity": equity,
            "buying_power": self.cash * 4,  # 4x leverage for paper trading
            "portfolio_value": equity,
            "positions_value": positions_value,
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": 0.0,  # Track this separately if needed
            "margin_used": margin_used,
            "total_cash": self.cash,
        }

    async def get_latest_bars(self, symbol: str, limit: int = 100) -> Optional[Dict[str, List[float]]]:
        """
        Get latest price bars

        For paper trading, we generate realistic mock data
        """
        # Check cache first
        if symbol in self.market_data_cache:
            cached_data = self.market_data_cache[symbol]
            # If cache is recent (last 60 seconds), return it
            if len(cached_data["timestamp"]) > 0:
                last_time = cached_data["timestamp"][-1]
                if (datetime.now() - last_time).seconds < 60:
                    return cached_data

        # Generate mock data
        data = self._generate_mock_bars(symbol, limit)
        self.market_data_cache[symbol] = data

        return data

    def _generate_mock_bars(self, symbol: str, limit: int) -> Dict[str, List[float]]:
        """
        Generate realistic mock price data

        Uses random walk with realistic volatility
        """
        # Base price (different for each symbol)
        base_price = hash(symbol) % 500 + 50  # Price between 50-550

        # Generate bars
        close_prices = []
        open_prices = []
        high_prices = []
        low_prices = []
        volumes = []
        timestamps = []

        current_price = base_price
        now = datetime.now()

        for i in range(limit):
            # Random walk with drift
            change_pct = random.gauss(0.0001, 0.002)  # 0.2% daily volatility
            current_price *= 1 + change_pct

            # Generate OHLC for the bar
            bar_volatility = current_price * 0.001

            open_price = current_price + random.gauss(0, bar_volatility)
            close_price = current_price + random.gauss(0, bar_volatility)
            high_price = max(open_price, close_price) + abs(random.gauss(0, bar_volatility))
            low_price = min(open_price, close_price) - abs(random.gauss(0, bar_volatility))

            volume = random.randint(1000000, 10000000)

            open_prices.append(open_price)
            high_prices.append(high_price)
            low_prices.append(low_price)
            close_prices.append(close_price)
            volumes.append(volume)
            timestamps.append(now - timedelta(minutes=limit - i))

        return {"open": open_prices, "high": high_prices, "low": low_prices, "close": close_prices, "volume": volumes, "timestamp": timestamps}

    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        user: Optional[User] = None,
        db: Optional[AsyncSession] = None,
        broker_type: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Place a simulated order"""

        if not self.connected:
            logger.error("Not connected to broker")
            return None

        # Get current price
        bars = await self.get_latest_bars(symbol, limit=1)
        if not bars:
            logger.error(f"No market data for {symbol}")
            return None

        current_price = bars["close"][-1]

        # For market orders, fill immediately at current price
        if order_type == "market":
            filled_price = current_price

            # Add slippage (0.05% average)
            slippage = current_price * 0.0005 * random.choice([-1, 1])
            filled_price += slippage

            # Execute trade
            if side.lower() == "buy":
                total_cost = filled_price * quantity

                if total_cost > self.cash:
                    logger.error(f"Insufficient cash: need ${total_cost:.2f}, have ${self.cash:.2f}")
                    return {"order_id": None, "status": "rejected", "reason": "insufficient_funds"}

                # Deduct cash
                self.cash -= total_cost

                # Add position
                if symbol in self.positions:
                    # Average down
                    pos = self.positions[symbol]
                    total_quantity = pos["quantity"] + quantity
                    avg_price = ((pos["entry_price"] * pos["quantity"]) + (filled_price * quantity)) / total_quantity

                    self.positions[symbol] = {"symbol": symbol, "quantity": total_quantity, "entry_price": avg_price, "current_price": filled_price}
                else:
                    self.positions[symbol] = {"symbol": symbol, "quantity": quantity, "entry_price": filled_price, "current_price": filled_price}

            else:  # sell
                if symbol not in self.positions:
                    logger.error(f"No position to sell for {symbol}")
                    return {"order_id": None, "status": "rejected", "reason": "no_position"}

                pos = self.positions[symbol]
                if pos["quantity"] < quantity:
                    logger.error(f"Insufficient shares: need {quantity}, have {pos['quantity']}")
                    return {"order_id": None, "status": "rejected", "reason": "insufficient_shares"}

                # Add proceeds to cash
                self.cash += filled_price * quantity

                # Update or remove position
                pos["quantity"] -= quantity
                if pos["quantity"] <= 0:
                    del self.positions[symbol]

            # Create order record
            self.order_counter += 1
            order_id = f"PAPER_{self.order_counter:06d}"

            order = {
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "order_type": order_type,
                "status": "filled",
                "filled_price": filled_price,
                "filled_quantity": quantity,
                "timestamp": datetime.now(),
            }

            self.orders[order_id] = order

            logger.info(f"Paper trade executed: {side.upper()} {quantity} {symbol} @ ${filled_price:.2f}")

            return order

        else:
            # Limit orders not yet implemented
            logger.warning("Limit orders not yet implemented in paper trading")
            return None

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if order_id in self.orders:
            order = self.orders[order_id]
            if order["status"] != "filled":
                order["status"] = "cancelled"
                return True
        return False

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions"""
        # Update current prices
        positions_list = []

        for symbol, pos in self.positions.items():
            # Get latest price
            bars = await self.get_latest_bars(symbol, limit=1)
            if bars:
                current_price = bars["close"][-1]
                pos["current_price"] = current_price

                market_value = pos["quantity"] * current_price
                unrealized_pnl = (current_price - pos["entry_price"]) * pos["quantity"]

                positions_list.append(
                    {
                        "symbol": symbol,
                        "quantity": pos["quantity"],
                        "entry_price": pos["entry_price"],
                        "current_price": current_price,
                        "market_value": market_value,
                        "unrealized_pnl": unrealized_pnl,
                    }
                )

        return positions_list

    async def is_market_open(self) -> bool:
        """
        Check if market is open for trading

        For paper trading, simulates NYSE/NASDAQ market hours:
        - Monday-Friday (excluding holidays)
        - 9:30 AM - 4:00 PM ET
        - Pre-market: 4:00 AM - 9:30 AM ET
        - After-hours: 4:00 PM - 8:00 PM ET
        """
        try:
            # Get current time in Eastern Time
            eastern = pytz.timezone("US/Eastern")
            now_et = datetime.now(eastern)

            # Check if weekend
            if now_et.weekday() >= 5:  # Saturday=5, Sunday=6
                logger.debug(f"Weekend: {now_et.strftime('%A')}")
                return False

            # Check if holiday
            if self._is_market_holiday(now_et.date()):
                logger.debug(f"Market holiday: {now_et.date()}")
                return False

            # Check regular trading hours (RTH)
            market_open = time(9, 30)
            market_close = time(16, 0)
            current_time = now_et.time()

            # Early market close for certain holidays
            if self._is_early_close_day(now_et.date()):
                market_close = time(13, 0)  # 1:00 PM ET
                logger.debug(f"Early market close at {market_close}")

            # Check if within trading hours
            is_open = market_open <= current_time < market_close

            # Log for debugging
            logger.debug(
                f"Market status at {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}: "
                f"{'OPEN' if is_open else 'CLOSED'} "
                f"(Hours: {market_open.strftime('%H:%M')}-{market_close.strftime('%H:%M')})"
            )

            return is_open

        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            return False  # Default to closed on error

    def _is_market_holiday(self, check_date: date) -> bool:
        """
        Check if date is a US stock market holiday
        """
        # NYSE observed holidays
        market_holidays = {
            # New Year's Day
            date(check_date.year, 1, 1),
            # Martin Luther King Jr. Day (3rd Monday in January)
            self._nth_weekday_of_month(check_date.year, 1, 0, 3),  # Monday=0
            # Washington's Birthday/Presidents Day (3rd Monday in February)
            self._nth_weekday_of_month(check_date.year, 2, 0, 3),
            # Good Friday (varies by year - need calculation)
            # This is a simplified placeholder
            # Memorial Day (last Monday in May)
            self._last_weekday_of_month(check_date.year, 5, 0),
            # Juneteenth (June 19th)
            date(check_date.year, 6, 19),
            # Independence Day (July 4th)
            date(check_date.year, 7, 4),
            # Labor Day (1st Monday in September)
            self._nth_weekday_of_month(check_date.year, 9, 0, 1),
            # Thanksgiving Day (4th Thursday in November)
            self._nth_weekday_of_month(check_date.year, 11, 3, 4),  # Thursday=3
            # Christmas Day (December 25th)
            date(check_date.year, 12, 25),
        }

        # Add observed holidays (if holiday falls on weekend)
        for holiday_date in list(market_holidays):
            if holiday_date.weekday() == 5:  # Saturday
                market_holidays.add(holiday_date - timedelta(days=1))  # Friday before
            elif holiday_date.weekday() == 6:  # Sunday
                market_holidays.add(holiday_date + timedelta(days=1))  # Monday after

        # Remove the original weekend dates
        market_holidays = {d for d in market_holidays if d.weekday() < 5}

        return check_date in market_holidays

    def _is_early_close_day(self, check_date: date) -> bool:
        """
        Check if market closes early (1:00 PM ET)
        """
        early_close_days = {
            # Day after Thanksgiving (Black Friday)
            self._nth_weekday_of_month(check_date.year, 11, 3, 4) + timedelta(days=1),
            # Christmas Eve (if not on weekend)
            date(check_date.year, 12, 24),
            # New Year's Eve (if not on weekend)
            date(check_date.year, 12, 31),
        }

        # Only if it's a weekday
        early_close_days = {d for d in early_close_days if d.weekday() < 5}

        return check_date in early_close_days

    @staticmethod
    def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
        """
        Get the nth weekday of a month (e.g., 3rd Monday)
        weekday: 0=Monday, 1=Tuesday, ..., 6=Sunday
        n: which occurrence (1=first, 2=second, etc.)
        """
        # First day of the month
        first_day = date(year, month, 1)

        # Find first occurrence of the weekday
        days_to_add = (weekday - first_day.weekday()) % 7
        first_occurrence = first_day + timedelta(days=days_to_add)

        # Add weeks to get nth occurrence
        return first_occurrence + timedelta(weeks=n - 1)

    @staticmethod
    def _last_weekday_of_month(year: int, month: int, weekday: int) -> date:
        """
        Get the last weekday of a month
        """
        # First day of next month
        if month == 12:
            next_month = date(year + 1, 1, 1)
        else:
            next_month = date(year, month + 1, 1)

        # Go back one day at a time until we find the right weekday
        last_day = next_month - timedelta(days=1)
        while last_day.weekday() != weekday:
            last_day -= timedelta(days=1)

        return last_day

    async def get_market_hours(self) -> Dict[str, Any]:
        """
        Get detailed market hours information

        Returns: {
            'is_open': bool,
            'next_open': datetime,
            'next_close': datetime,
            'regular_hours': {'open': time, 'close': time},
            'early_close': bool,
            'holiday': bool,
            'current_session': 'pre' | 'regular' | 'post' | 'closed'
        }
        """
        eastern = pytz.timezone("US/Eastern")
        now_et = datetime.now(eastern)
        today = now_et.date()

        # Default regular hours
        regular_open = time(9, 30)
        regular_close = time(16, 0)

        # Check for early close
        early_close = self._is_early_close_day(today)
        if early_close:
            regular_close = time(13, 0)

        # Check if holiday
        is_holiday = self._is_market_holiday(today)

        # Determine current session
        current_time = now_et.time()

        if is_holiday or now_et.weekday() >= 5:
            session = "closed"
            is_open = False
        elif regular_open <= current_time < regular_close:
            session = "regular"
            is_open = True
        elif time(4, 0) <= current_time < regular_open:  # Pre-market
            session = "pre"
            is_open = False
        elif regular_close <= current_time < time(20, 0):  # After-hours
            session = "post"
            is_open = False
        else:
            session = "closed"
            is_open = False

        # Calculate next open/close
        next_open, next_close = self._calculate_next_market_hours(now_et)

        return {
            "is_open": is_open,
            "next_open": next_open,
            "next_close": next_close,
            "regular_hours": {"open": regular_open.strftime("%H:%M"), "close": regular_close.strftime("%H:%M")},
            "early_close": early_close,
            "holiday": is_holiday,
            "current_session": session,
            "current_time": now_et.strftime("%Y-%m-%d %H:%M:%S %Z"),
        }

    def _calculate_next_market_hours(self, current_dt: datetime) -> tuple:
        """
        Calculate next market open and close times
        """
        eastern = pytz.timezone("US/Eastern")
        check_date = current_dt.date()

        # Keep looking forward until we find a trading day
        days_ahead = 0
        while True:
            next_date = check_date + timedelta(days=days_ahead)

            # Skip weekends and holidays
            if next_date.weekday() >= 5 or self._is_market_holiday(next_date):
                days_ahead += 1
                continue

            # Calculate open and close times for this day
            open_time = time(9, 30)
            close_time = time(16, 0)

            if self._is_early_close_day(next_date):
                close_time = time(13, 0)

            # Create datetime objects
            next_open = eastern.localize(datetime.combine(next_date, open_time))
            next_close = eastern.localize(datetime.combine(next_date, close_time))

            # If market is already open today, next close is today's close
            if days_ahead == 0 and current_dt.time() < open_time:
                # Market hasn't opened yet today
                return next_open, next_close
            elif days_ahead == 0 and current_dt.time() < close_time:
                # Market is currently open, next close is today's close
                return None, next_close
            else:
                # Next trading day
                return next_open, next_close

    async def place_market_order(
        self, symbol: str, qty: float, side: str, time_in_force: str = "day", extended_hours: bool = False
    ) -> Dict[str, Any]:
        pass

    async def place_limit_order(
        self, symbol: str, qty: float, side: str, limit_price: float, time_in_force: str = "day", extended_hours: bool = False
    ) -> Dict[str, Any]:
        pass

    async def place_stop_order(
        self, symbol: str, qty: float, side: str, stop_price: float, time_in_force: str = "day", extended_hours: bool = False
    ) -> Dict[str, Any]:
        pass

    async def place_stop_limit_order(
        self, symbol: str, qty: float, side: str, stop_price: float, limit_price: float, time_in_force: str = "day", extended_hours: bool = False
    ) -> Dict[str, Any]:
        pass

    async def place_option_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        option_type: str,
        strike: float,
        expiration: str,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        time_in_force: str = "day",
    ) -> Dict[str, Any]:
        pass

    async def get_option_positions(self) -> List[Dict[str, Any]]:
        pass

    async def get_orders(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        pass

    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        pass

    async def replace_order(
        self,
        order_id: str,
        qty: Optional[float] = None,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: Optional[str] = None,
    ) -> Dict[str, Any]:
        pass

    async def get_bars(self, symbol: str, timeframe: str, start: str, end: str, adjustment: str = "raw") -> List[Dict[str, Any]]:
        pass

    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        pass
