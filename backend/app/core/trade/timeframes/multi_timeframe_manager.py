import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from core.trade.timeframes.timeframe_aggregator import TimeframeAggregator
from core.trade.timeframes.timeframe_bar import TimeframeBar

logger = logging.getLogger(__name__)


class MultiTimeframeManager:
    """
    Manages multiple timeframes for a symbol

    Automatically aggregates 1-minute data into higher timeframes
    """

    def __init__(self, symbol: str, timeframes: List[str]):
        self.symbol = symbol
        self.timeframes = timeframes

        # Create aggregator for each timeframe
        self.aggregators: Dict[str, TimeframeAggregator] = {tf: TimeframeAggregator(tf) for tf in timeframes}

        logger.info(f"Initialized multi-timeframe manager for {symbol}: {timeframes}")

    def update(self, timestamp: datetime, price: float, volume: float = 0):
        """
        Update all timeframes with new price tick
        """
        for tf, aggregator in self.aggregators.items():
            aggregator.add_tick(timestamp, price, volume)

    def update_from_bar(self, bar_data: Dict[str, Any]):
        """
        Update from a complete bar

        Args:
            bar_data: {
                'timestamp': datetime,
                'open': float,
                'high': float,
                'low': float,
                'close': float,
                'volume': float
            }
        """
        bar = TimeframeBar(**bar_data)

        for tf, aggregator in self.aggregators.items():
            aggregator.add_bar(bar)

    def get_data(self, timeframe: str, count: Optional[int] = None) -> pd.DataFrame:
        """
        Get data for a specific timeframe

        Args:
            timeframe: '1min', '5min', '15min', '1hour', etc.
            count: Number of bars to return (None = all)

        Returns:
            DataFrame with OHLCV data
        """
        if timeframe not in self.aggregators:
            raise ValueError(f"Timeframe {timeframe} not available")

        return self.aggregators[timeframe].get_dataframe(count)

    def get_all_data(self, count: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """Get data for all timeframes"""
        return {tf: aggregator.get_dataframe(count) for tf, aggregator in self.aggregators.items()}

    def get_latest_price(self, timeframe: str) -> Optional[float]:
        """Get latest close price for a timeframe"""
        bars = self.aggregators[timeframe].get_bars(1)
        if bars:
            return bars[-1].close
        return None
