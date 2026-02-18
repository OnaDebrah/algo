from abc import ABC, abstractmethod
from typing import Any, Optional
import pandas as pd

class DataProvider(ABC):
    """
    Abstract base class for all data providers.
    Ensures a consistent interface for fetching market data.
    """

    @abstractmethod
    def fetch_data(
        self, 
        symbol: str, 
        period: str, 
        interval: str, 
        start: Optional[Any] = None, 
        end: Optional[Any] = None
    ) -> pd.DataFrame:
        """
        Fetch market data for a given symbol.
        
        Args:
            symbol: Ticker symbol.
            period: Time period (e.g., '1y', 'max').
            interval: Data interval (e.g., '1d', '1h').
            start: Optional start date.
            end: Optional end date.
            
        Returns:
            DataFrame with OHLCV data.
        """
        pass
