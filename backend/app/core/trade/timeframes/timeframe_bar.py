from datetime import datetime
from typing import Any, Dict


class TimeframeBar:
    """Represents a single bar (OHLCV)"""

    def __init__(self, timestamp: datetime, open: float, high: float, low: float, close: float, volume: float):
        self.timestamp = timestamp
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

    def to_dict(self) -> Dict[str, Any]:
        return {"timestamp": self.timestamp, "open": self.open, "high": self.high, "low": self.low, "close": self.close, "volume": self.volume}
