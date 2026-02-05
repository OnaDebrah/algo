from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List


class ComplianceManager:
    """
    Compliance and regulatory reporting

    Features:
    - Transaction reporting
    - Position limits
    - Pattern day trader detection
    - Wash sale detection
    """

    def __init__(self):
        self.trade_history: List[Dict] = []

    async def check_pattern_day_trader(self, user_id: int, account_value: float) -> Dict[str, Any]:
        """
        Check if user qualifies as pattern day trader (PDT)

        PDT Rule: 4 or more day trades in 5 business days
        with account < $25,000
        """
        # Get trades from last 5 business days
        recent_trades = self._get_recent_trades(user_id, days=5)

        # Count day trades (open and close same day)
        day_trades = self._count_day_trades(recent_trades)

        # Check PDT rule
        is_pdt = day_trades >= 4 and account_value < 25000

        return {
            "is_pattern_day_trader": is_pdt,
            "day_trades_count": day_trades,
            "account_value": account_value,
            "warning": "Pattern Day Trader restrictions may apply" if is_pdt else None,
        }

    async def detect_wash_sales(self, user_id: int, symbol: str) -> List[Dict[str, Any]]:
        """
        Detect potential wash sales

        Wash Sale: Selling at loss and repurchasing within 30 days
        """
        wash_sales = []

        # Get all trades for symbol
        trades = [t for t in self.trade_history if t["symbol"] == symbol]

        # Look for loss sales followed by repurchase within 30 days
        for i, trade in enumerate(trades):
            if trade["profit"] < 0:  # Loss
                # Check for repurchase within 30 days
                repurchase_window = timedelta(days=30)

                for j in range(i + 1, len(trades)):
                    next_trade = trades[j]
                    time_diff = next_trade["timestamp"] - trade["timestamp"]

                    if time_diff <= repurchase_window and next_trade["side"] == "BUY":
                        wash_sales.append({"original_trade": trade, "repurchase_trade": next_trade, "disallowed_loss": abs(trade["profit"])})
                        break

        return wash_sales

    def _get_recent_trades(self, user_id: int, days: int) -> List[Dict]:
        """Get recent trades for user"""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        return [t for t in self.trade_history if t["user_id"] == user_id and t["timestamp"] > cutoff]

    def _count_day_trades(self, trades: List[Dict]) -> int:
        """Count day trades in trade list"""
        day_trades = 0

        # Group trades by date
        trades_by_date = {}
        for trade in trades:
            date = trade["timestamp"].date()
            if date not in trades_by_date:
                trades_by_date[date] = []
            trades_by_date[date].append(trade)

        # Check each date for day trades
        for date, date_trades in trades_by_date.items():
            # Look for open and close on same day
            buys = [t for t in date_trades if t["side"] == "BUY"]
            sells = [t for t in date_trades if t["side"] == "SELL"]

            # Each matching buy-sell pair is a day trade
            day_trades += min(len(buys), len(sells))

        return day_trades
