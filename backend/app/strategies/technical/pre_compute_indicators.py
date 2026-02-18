# backend/app/core/precomputed_indicators.py

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from backend.app.core.data_fetcher import fetch_stock_data
from backend.app.strategies.technical.bb_mean_reversion import BollingerMeanReversionStrategy
from backend.app.strategies.technical.macd_strategy import MACDStrategy
from backend.app.strategies.technical.rsi_strategy import RSIStrategy

logger = logging.getLogger(__name__)


class PreComputeIndicators:
    def __init__(self):
        self._cache = {}
        self._indicator_cache = {}

    async def get_data(self, symbol: str, interval: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        """Fetch and cache data"""
        cache_key = f"{symbol}_{interval}"

        if cache_key not in self._cache:
            logger.info(f"Fetching data for {symbol}")
            data = await fetch_stock_data(symbol, "max", interval)
            self._cache[cache_key] = data

        data = self._cache[cache_key].copy()

        if start:
            try:
                # Find start index with bfill
                target_idx = data.index.get_indexer([pd.Timestamp(start)], method="bfill")[0]
                # Go back 252 bars for 1-year warm-up
                warm_up_idx = max(0, target_idx - 250)
                data = data.iloc[warm_up_idx:].loc[:end]
            except Exception as e:
                logger.warning(f"Optimization data slicing failed: {e}")
                data = data.loc[:end]

        return data

    def precalculate_indicators(self, data: pd.DataFrame, indicator_config: Dict[str, Any]) -> pd.DataFrame:
        """
        Pre-calculate technical indicators using existing strategy classes

        Args:
            data: Raw OHLCV data
            indicator_config: Configuration for which indicators to calculate

        Returns:
            DataFrame with added indicator columns
        """
        # Key must include the data boundaries, not just config, since
        # different WFA folds pass different data slices.
        data_sig = (len(data), str(data.index[0]), str(data.index[-1]))
        cache_key = hash((frozenset(indicator_config.items()), data_sig))

        if cache_key in self._indicator_cache:
            return self._indicator_cache[cache_key]

        df = data.copy()

        # Returns (always calculated as they're simple)
        if indicator_config.get("returns", True):
            df["returns"] = df["Close"].pct_change()
            df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))

        # Volatility
        if indicator_config.get("volatility", False):
            for period in indicator_config.get("volatility_periods", [10, 20, 30]):
                df[f"volatility_{period}"] = df["returns"].rolling(period).std() * np.sqrt(252)

        # Moving Averages - using SMA strategy logic
        if indicator_config.get("moving_averages", False):
            for period in indicator_config.get("ma_periods", [10, 20, 50, 200]):
                # Create a temporary SMA strategy to calculate MA
                # Note: SMA strategy calculates MAs internally, but we can also calculate directly
                df[f"ma_{period}"] = df["Close"].rolling(window=period).mean()
                df[f"ma_{period}_slope"] = df[f"ma_{period}"].pct_change(5)

        # RSI - using existing RSI strategy logic
        if indicator_config.get("rsi", False):
            for period in indicator_config.get("rsi_periods", [14]):
                # Create RSI strategy instance to use its calculation
                rsi_strategy = RSIStrategy(period=period)
                df[f"rsi_{period}"] = rsi_strategy.calculate_rsi(df)

                # Also add overbought/oversold levels as metadata
                df[f"rsi_overbought_{period}"] = 70  # Default, could be configurable
                df[f"rsi_oversold_{period}"] = 30  # Default, could be configurable

        # MACD - using existing MACD strategy logic
        if indicator_config.get("macd", False):
            # Get MACD parameters from config or use defaults
            fast = indicator_config.get("macd_fast", 12)
            slow = indicator_config.get("macd_slow", 26)
            signal = indicator_config.get("macd_signal", 9)

            # Create MACD strategy instance
            macd_strategy = MACDStrategy(fast=fast, slow=slow, signal=signal)
            macd_line, signal_line, histogram = macd_strategy.calculate_macd(df)

            df["macd"] = macd_line
            df["macd_signal"] = signal_line
            df["macd_hist"] = histogram

        # Bollinger Bands - using existing Bollinger strategy logic
        if indicator_config.get("bollinger_bands", False):
            for period in indicator_config.get("bb_periods", [20]):
                std_dev = indicator_config.get("bb_std_dev", 2.0)

                # Create Bollinger strategy instance
                bb_strategy = BollingerMeanReversionStrategy(period=period, std_dev=std_dev)
                sma, upper_band, lower_band = bb_strategy.calculate_bollinger_bands(df)

                # Store results with consistent naming
                df[f"bb_ma_{period}"] = sma
                df[f"bb_upper_{period}"] = upper_band
                df[f"bb_lower_{period}"] = lower_band
                df[f"bb_width_{period}"] = (upper_band - lower_band) / sma

        self._indicator_cache[cache_key] = df
        return df
