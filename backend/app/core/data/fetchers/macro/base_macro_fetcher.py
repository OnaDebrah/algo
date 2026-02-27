from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd


class BaseMacroFetcher(ABC):
    """Abstract base class for all macro data providers"""

    @abstractmethod
    async def get_indicators(
        self,
        indicators: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        frequency: str = "monthly",
        country: str = None,
    ) -> pd.DataFrame:
        """
        Fetch multiple indicators

        Returns:
            DataFrame with dates as index, indicators as columns
        """
        pass

    @abstractmethod
    async def get_indicator(
        self,
        indicator_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        frequency: str = "monthly",
        country: str = None,
    ) -> pd.Series:
        """Fetch a single indicator"""
        pass

    @abstractmethod
    def get_available_indicators(self) -> Dict[str, str]:
        """Return mapping of indicator codes to descriptions"""
        pass
