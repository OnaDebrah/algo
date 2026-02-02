"""
Base strategy class for all trading strategies
"""

from abc import ABC, abstractmethod
from typing import Dict

import pandas as pd


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""

    def __init__(self, name: str, params: Dict):
        """
        Initialize strategy

        Args:
            name: Strategy name
            params: Strategy parameters
        """
        self.name = name
        self.params = params

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> int:
        """
        Generate trading signal based on market data

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Signal: 1 (buy), -1 (sell), 0 (hold)
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', params={self.params})"
