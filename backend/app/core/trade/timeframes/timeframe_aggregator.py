from collections import deque
from datetime import datetime
from typing import List, Optional

import pandas as pd

from core.trade.timeframes.timeframe_bar import TimeframeBar


class TimeframeAggregator:
    """
    Aggregates 1-minute bars into higher timeframes

    Supports: 1min, 5min, 15min, 30min, 1hour, 4hour, 1day
    """

    TIMEFRAMES = {"1min": 1, "5min": 5, "15min": 15, "30min": 30, "1hour": 60, "4hour": 240, "1day": 1440}

    def __init__(self, timeframe: str, max_bars: int = 500):
        if timeframe not in self.TIMEFRAMES:
            raise ValueError(f"Invalid timeframe: {timeframe}")

        self.timeframe = timeframe
        self.minutes = self.TIMEFRAMES[timeframe]
        self.max_bars = max_bars

        # Store aggregated bars
        self.bars: deque[TimeframeBar] = deque(maxlen=max_bars)

        # Current bar being built
        self.current_bar: Optional[TimeframeBar] = None
        self.current_bar_start: Optional[datetime] = None

    def add_tick(self, timestamp: datetime, price: float, volume: float = 0):
        """
        Add a price tick and update aggregation

        For 1-minute resolution, this would be called every minute
        """
        # Determine bar period for this tick
        bar_start = self._get_bar_start_time(timestamp)

        # Check if we need to start a new bar
        if self.current_bar_start != bar_start:
            # Close previous bar
            if self.current_bar:
                self.bars.append(self.current_bar)

            # Start new bar
            self.current_bar = TimeframeBar(timestamp=bar_start, open=price, high=price, low=price, close=price, volume=volume)
            self.current_bar_start = bar_start

        else:
            # Update current bar
            self.current_bar.high = max(self.current_bar.high, price)
            self.current_bar.low = min(self.current_bar.low, price)
            self.current_bar.close = price
            self.current_bar.volume += volume

    def add_bar(self, bar: TimeframeBar):
        """
        Add a complete bar (for initializing from historical data)
        """
        bar_start = self._get_bar_start_time(bar.timestamp)

        if self.current_bar_start != bar_start:
            if self.current_bar:
                self.bars.append(self.current_bar)

            self.current_bar = bar
            self.current_bar_start = bar_start
        else:
            # Merge with current bar
            self.current_bar.high = max(self.current_bar.high, bar.high)
            self.current_bar.low = min(self.current_bar.low, bar.low)
            self.current_bar.close = bar.close
            self.current_bar.volume += bar.volume

    def get_bars(self, count: Optional[int] = None) -> List[TimeframeBar]:
        """Get historical bars (excluding incomplete current bar)"""
        bars_list = list(self.bars)
        if count:
            return bars_list[-count:]
        return bars_list

    def get_dataframe(self, count: Optional[int] = None) -> pd.DataFrame:
        """Get bars as pandas DataFrame"""
        bars = self.get_bars(count)

        if not bars:
            return pd.DataFrame()

        return pd.DataFrame([bar.to_dict() for bar in bars])

    def _get_bar_start_time(self, timestamp: datetime) -> datetime:
        """Calculate bar start time for a given timestamp"""
        if self.timeframe == "1day":
            # Start of day
            return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)

        # For intraday timeframes
        minute = (timestamp.minute // self.minutes) * self.minutes
        return timestamp.replace(minute=minute, second=0, microsecond=0)
