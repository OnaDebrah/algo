from typing import Any, Dict, List, Optional

import pandas as pd

from core.trade.timeframes.multi_timeframe_manager import MultiTimeframeManager


class StrategyMultiTimeframeAdapter:
    """
    Adapter to provide multi-timeframe data to strategies

    Usage:
        # In strategy
        def generate_signal(self, data):
            # data is now a dict of DataFrames
            df_1min = data['1min']
            df_5min = data['5min']
            df_1hour = data['1hour']

            # Use multiple timeframes for decision
            trend_1hour = self._get_trend(df_1hour)
            signal_5min = self._get_signal(df_5min)
            entry_1min = self._get_entry(df_1min)
    """

    def __init__(self, symbols: List[str], timeframes: List[str]):
        # {symbol: MultiTimeframeManager}
        self.managers: Dict[str, MultiTimeframeManager] = {symbol: MultiTimeframeManager(symbol, timeframes) for symbol in symbols}

    def update(self, symbol: str, bar_data: Dict[str, Any]):
        """Update data for a symbol"""
        if symbol in self.managers:
            self.managers[symbol].update_from_bar(bar_data)

    def get_strategy_data(self, symbol: str, count: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Get all timeframe data for a symbol in strategy-friendly format

        Returns:
            {
                '1min': DataFrame,
                '5min': DataFrame,
                '1hour': DataFrame
            }
        """
        if symbol not in self.managers:
            return {}

        return self.managers[symbol].get_all_data(count)

    def get_multi_symbol_data(self, symbols: List[str], timeframe: str, count: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Get specific timeframe data for multiple symbols

        Returns:
            {
                'AAPL': DataFrame,
                'MSFT': DataFrame
            }
        """
        return {symbol: self.managers[symbol].get_data(timeframe, count) for symbol in symbols if symbol in self.managers}
