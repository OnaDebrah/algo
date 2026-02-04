"""
Base strategy class for all trading strategies
Enhanced to support both simple signals and complex signal dictionaries
"""

from abc import ABC, abstractmethod
from typing import Dict, Union

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
    def generate_signal(self, data: pd.DataFrame) -> Union[int, Dict]:
        """
        Generate trading signal based on market data

        Args:
            data: DataFrame with OHLCV data (or multi-asset data for pairs strategies)

        Returns:
            For simple strategies:
                int: Signal: 1 (buy), -1 (sell), 0 (hold)

            For advanced strategies (pairs trading, ML, etc.):
                Dict: {
                    "signal": int,              # 1 (buy/long), -1 (sell/short), 0 (hold/close)
                    "position_size": float,     # Position size multiplier (0.0 to 1.0)
                    "metadata": dict           # Additional strategy-specific data
                }

        Examples:
            Simple strategy:
                return 1  # Buy signal

            Advanced strategy:
                return {
                    "signal": 1,
                    "position_size": 0.8,  # 80% position
                    "metadata": {
                        "confidence": 0.85,
                        "z_score": -2.3,
                        "hedge_ratio": 1.5
                    }
                }
        """
        pass

    def reset(self):
        """
        Reset strategy state

        Override this method if your strategy maintains internal state
        (e.g., positions, counters, Kalman filters, etc.)
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', params={self.params})"


# Utility function for backward compatibility
def normalize_signal(signal: Union[int, Dict]) -> Dict:
    """
    Normalize strategy signal to standard dictionary format

    This allows backward compatibility with strategies that return just an int.

    Args:
        signal: Raw signal from generate_signal (int or dict)

    Returns:
        Normalized signal dictionary with keys: signal, position_size, metadata

    Examples:
        >>> normalize_signal(1)
        {'signal': 1, 'position_size': 1.0, 'metadata': {}}

        >>> normalize_signal({'signal': -1, 'position_size': 0.5, 'metadata': {'confidence': 0.8}})
        {'signal': -1, 'position_size': 0.5, 'metadata': {'confidence': 0.8}}
    """
    if isinstance(signal, dict):
        # Already in dictionary format
        return {"signal": signal.get("signal", 0), "position_size": signal.get("position_size", 1.0), "metadata": signal.get("metadata", {})}
    else:
        # Simple integer signal - normalize to dict
        return {"signal": int(signal), "position_size": 1.0, "metadata": {}}
