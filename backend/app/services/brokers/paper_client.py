import logging
import random
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from brokers.base_client import BrokerClient

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

    async def connect(self, credentials: Dict[str, str]) -> bool:
        """Connect to paper trading"""
        logger.info("Connecting to paper trading")
        self.connected = True

        # Set initial capital if provided
        if 'initial_capital' in credentials:
            self.cash = float(credentials['initial_capital'])

        return True

    async def disconnect(self):
        """Disconnect from paper trading"""
        logger.info("Disconnecting from paper trading")
        self.connected = False

    async def is_market_open(self) -> bool:
        """
        Check if market is open

        For paper trading, we'll simulate market hours:
        - Monday-Friday
        - 9:30 AM - 4:00 PM ET
        """
        now = datetime.now()

        # Check if weekend
        if now.weekday() >= 5:  # Saturday=5, Sunday=6
            return False

        # Check market hours (simplified - doesn't account for holidays)
        hour = now.hour
        if hour < 9 or hour >= 16:
            return False
        if hour == 9 and now.minute < 30:
            return False

        return True

    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        positions_value = sum(
            pos['quantity'] * pos['current_price']
            for pos in self.positions.values()
        )

        equity = self.cash + positions_value

        return {
            'cash': self.cash,
            'equity': equity,
            'buying_power': self.cash,  # Simplified - no margin
            'positions_value': positions_value
        }

    async def get_latest_bars(
            self,
            symbol: str,
            limit: int = 100
    ) -> Optional[Dict[str, List[float]]]:
        """
        Get latest price bars

        For paper trading, we generate realistic mock data
        """
        # Check cache first
        if symbol in self.market_data_cache:
            cached_data = self.market_data_cache[symbol]
            # If cache is recent (last 60 seconds), return it
            if len(cached_data['timestamp']) > 0:
                last_time = cached_data['timestamp'][-1]
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
            current_price *= (1 + change_pct)

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

        return {
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes,
            'timestamp': timestamps
        }

    async def place_order(
            self,
            symbol: str,
            side: str,
            quantity: float,
            order_type: str = 'market',
            limit_price: Optional[float] = None
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

        current_price = bars['close'][-1]

        # For market orders, fill immediately at current price
        if order_type == 'market':
            filled_price = current_price

            # Add slippage (0.05% average)
            slippage = current_price * 0.0005 * random.choice([-1, 1])
            filled_price += slippage

            # Execute trade
            if side.lower() == 'buy':
                total_cost = filled_price * quantity

                if total_cost > self.cash:
                    logger.error(f"Insufficient cash: need ${total_cost:.2f}, have ${self.cash:.2f}")
                    return {
                        'order_id': None,
                        'status': 'rejected',
                        'reason': 'insufficient_funds'
                    }

                # Deduct cash
                self.cash -= total_cost

                # Add position
                if symbol in self.positions:
                    # Average down
                    pos = self.positions[symbol]
                    total_quantity = pos['quantity'] + quantity
                    avg_price = (
                                        (pos['entry_price'] * pos['quantity']) +
                                        (filled_price * quantity)
                                ) / total_quantity

                    self.positions[symbol] = {
                        'symbol': symbol,
                        'quantity': total_quantity,
                        'entry_price': avg_price,
                        'current_price': filled_price
                    }
                else:
                    self.positions[symbol] = {
                        'symbol': symbol,
                        'quantity': quantity,
                        'entry_price': filled_price,
                        'current_price': filled_price
                    }

            else:  # sell
                if symbol not in self.positions:
                    logger.error(f"No position to sell for {symbol}")
                    return {
                        'order_id': None,
                        'status': 'rejected',
                        'reason': 'no_position'
                    }

                pos = self.positions[symbol]
                if pos['quantity'] < quantity:
                    logger.error(f"Insufficient shares: need {quantity}, have {pos['quantity']}")
                    return {
                        'order_id': None,
                        'status': 'rejected',
                        'reason': 'insufficient_shares'
                    }

                # Add proceeds to cash
                self.cash += filled_price * quantity

                # Update or remove position
                pos['quantity'] -= quantity
                if pos['quantity'] <= 0:
                    del self.positions[symbol]

            # Create order record
            self.order_counter += 1
            order_id = f"PAPER_{self.order_counter:06d}"

            order = {
                'order_id': order_id,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'order_type': order_type,
                'status': 'filled',
                'filled_price': filled_price,
                'filled_quantity': quantity,
                'timestamp': datetime.now()
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
            if order['status'] != 'filled':
                order['status'] = 'cancelled'
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
                current_price = bars['close'][-1]
                pos['current_price'] = current_price

                market_value = pos['quantity'] * current_price
                unrealized_pnl = (current_price - pos['entry_price']) * pos['quantity']

                positions_list.append({
                    'symbol': symbol,
                    'quantity': pos['quantity'],
                    'entry_price': pos['entry_price'],
                    'current_price': current_price,
                    'market_value': market_value,
                    'unrealized_pnl': unrealized_pnl
                })

        return positions_list
