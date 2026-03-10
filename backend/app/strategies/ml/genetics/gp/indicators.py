import pandas as pd


class TechnicalIndicators:
    """
    Technical indicators for feature engineering.
    Serves as the terminal set for genetic programming [citation:2][citation:8].
    """

    @staticmethod
    def SMA(data: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=window).mean()

    @staticmethod
    def EMA(data: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=window, adjust=False).mean()

    @staticmethod
    def RSI(data: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def MACD(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """MACD line"""
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        return ema_fast - ema_slow

    @staticmethod
    def BB_position(data: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
        """Position within Bollinger Bands (0 to 1)"""
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper = sma + num_std * std
        lower = sma - num_std * std
        return (data - lower) / (upper - lower)

    @staticmethod
    def ATR(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range"""
        high_low = high - low
        high_close = (high - close.shift()).abs()
        low_close = (low - close.shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()

    @staticmethod
    def volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
        """Volume relative to its average"""
        return volume / volume.rolling(window=window).mean()

    @staticmethod
    def price_position(data: pd.Series, window: int = 20) -> pd.Series:
        """Position within price range (0 to 1)"""
        rolling_min = data.rolling(window=window).min()
        rolling_max = data.rolling(window=window).max()
        return (data - rolling_min) / (rolling_max - rolling_min)

    @staticmethod
    def volatility(data: pd.Series, window: int = 20) -> pd.Series:
        """Rolling volatility"""
        return data.pct_change().rolling(window=window).std()

    @staticmethod
    def directional_change(data: pd.Series, threshold: float = 0.01) -> pd.Series:
        """
        Directional Change indicator [citation:1]
        Returns 1 for uptrend, -1 for downtrend, 0 for no change
        """
        # Simplified implementation - full DC framework is more complex
        # This identifies significant turning points
        rolling_high = data.expanding().max()
        rolling_low = data.expanding().min()

        # Check if we've moved up enough from a low
        up_signal = (data / rolling_low - 1) > threshold

        # Check if we've moved down enough from a high
        down_signal = (1 - data / rolling_high) > threshold

        result = pd.Series(0, index=data.index)
        result[up_signal] = 1
        result[down_signal] = -1

        return result
