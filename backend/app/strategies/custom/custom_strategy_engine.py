"""
Custom Strategy Engine - Execute user-defined strategies with safety checks
"""

import logging
from typing import Any, Dict, Union

import pandas as pd

from ...strategies.base_strategy import BaseStrategy
from ...strategies.custom.safe_exe_env import SafeExecutionEnvironment

logger = logging.getLogger(__name__)


class CustomStrategyAdapter(BaseStrategy):
    """
    Adapter that wraps a standalone generate_signals(data) function
    inside a BaseStrategy-compatible interface for TradingEngine.
    """

    def __init__(self, name: str, code: str, params: Dict = None):
        super().__init__(name=name, params=params or {})
        self.code = code
        self._env = SafeExecutionEnvironment()
        self._compiled_func = None
        self._compile()

    def _compile(self):
        """Pre-compile the code and extract the generate_signals function."""
        is_valid, error = self._env.validate_code(self.code)
        if not is_valid:
            raise ValueError(f"Invalid strategy code: {error}")
        safe_globals = self._env.create_safe_globals()
        safe_locals: Dict[str, Any] = {}
        exec(self.code, safe_globals, safe_locals)  # noqa: S102 - sandboxed
        if "generate_signals" not in safe_locals:
            raise ValueError("Strategy code must define 'generate_signals(data)' function")
        self._compiled_func = safe_locals["generate_signals"]

    def generate_signal(self, data: pd.DataFrame) -> Union[int, Dict]:
        """Loop-based: call generate_signals and return the last signal."""
        signals = self._compiled_func(data.copy())
        if isinstance(signals, pd.Series) and len(signals) > 0:
            return int(signals.iloc[-1])
        return 0

    def generate_signals_vectorized(self, data: pd.DataFrame) -> pd.Series:
        """Vectorized: return the full signal series from user code."""
        result = self._compiled_func(data.copy())
        if isinstance(result, pd.Series):
            return result
        return pd.Series(0, index=data.index)
