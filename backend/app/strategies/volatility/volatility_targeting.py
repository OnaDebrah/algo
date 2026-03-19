from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from ...config import DEFAULT_ANNUAL_LOOKBACK
from ...strategies.volatility.base_volatility import BaseVolatilityStrategy


class VolatilityTargetingStrategy(BaseVolatilityStrategy):
    """
    Volatility Targeting Strategy

    Dynamically adjusts portfolio exposure to maintain target volatility.
    Standalone mode: uses momentum direction scaled by target_vol / realized_vol.
    Meta-strategy mode: wraps a base_strategy and scales its signals.
    """

    def __init__(
        self,
        base_strategy: callable = None,  # Underlying strategy to scale
        lookforward_vol: bool = True,  # Use forward-looking vol estimates
        volatility_buffer: float = 0.1,  # 10% buffer around target
        max_drawdown_adjustment: bool = True,
        trend_lookback: int = 20,  # Momentum lookback for standalone direction
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_strategy = base_strategy
        self.lookforward_vol = lookforward_vol
        self.volatility_buffer = volatility_buffer
        self.max_drawdown_adjustment = max_drawdown_adjustment
        self.trend_lookback = trend_lookback

        # Portfolio tracking
        self.portfolio_returns = []
        self.drawdown_level = 0.0
        self.scale_factor = 1.0

    def generate_signal(self, data: Union[pd.Series, pd.DataFrame]) -> Dict:
        """
        Generate volatility-targeted signal for standalone backtesting.

        Uses momentum direction scaled by vol-targeting leverage.
        When base_strategy is provided, delegates direction to it.
        """
        if isinstance(data, pd.DataFrame):
            if "Close" in data.columns:
                prices = data["Close"]
            elif "close" in data.columns:
                prices = data["close"]
            else:
                prices = data.iloc[:, 0]
        else:
            prices = data

        min_required = max(self.vol_lookback, self.trend_lookback)
        if len(prices) < min_required:
            return {"signal": 0, "position_size": 0.0, "metadata": {"strategy": "volatility_targeting"}}

        returns = self.calculate_returns(prices)
        self.update_volatility_state(returns)

        # Direction: delegate to base_strategy or use momentum
        if self.base_strategy is not None:
            base_signal = self.base_strategy.generate_signal(data)
            signal = base_signal.get("signal", 0) if isinstance(base_signal, dict) else int(base_signal)
        else:
            momentum = prices.iloc[-1] / prices.iloc[-self.trend_lookback] - 1
            signal = 1 if momentum > 0 else (-1 if momentum < 0 else 0)

        position_size = self.current_leverage * abs(signal)

        return {
            "signal": signal,
            "position_size": position_size,
            "leverage": self.current_leverage,
            "current_volatility": self.volatility_history[-1] if self.volatility_history else 0.0,
            "target_volatility": self.target_volatility,
            "metadata": {"strategy": "volatility_targeting", "vol_estimator": self.vol_estimator},
        }

    def generate_signals_vectorized(self, data: pd.DataFrame) -> pd.Series:
        """Vectorized signal generation using momentum direction."""
        close = data["Close"] if "Close" in data.columns else data.get("close", data.iloc[:, 0])
        signals = pd.Series(0, index=data.index)

        momentum = close / close.shift(self.trend_lookback) - 1
        direction = pd.Series(0, index=data.index)
        direction[momentum > 0] = 1
        direction[momentum < 0] = -1

        # Emit signals at direction transitions
        direction_change = direction != direction.shift(1)
        signals[(direction == 1) & direction_change] = 1
        signals[(direction != 1) & direction_change & (direction.shift(1) == 1)] = -1

        return signals

    def calculate_required_leverage(self, strategy_returns: pd.Series, target_vol: Optional[float] = None) -> float:
        """
        Calculate required leverage to hit target volatility

        Args:
            strategy_returns: Returns of underlying strategy
            target_vol: Target volatility (if None, use self.target_volatility)

        Returns:
            Required leverage factor
        """
        if target_vol is None:
            target_vol = self.target_volatility

        if len(strategy_returns) < 20:
            return 1.0

        # Calculate strategy volatility
        strategy_vol = strategy_returns.std() * np.sqrt(DEFAULT_ANNUAL_LOOKBACK)

        if strategy_vol <= 0:
            return 1.0

        # Required leverage to hit target
        required_leverage = target_vol / strategy_vol

        # Apply adjustments
        if self.max_drawdown_adjustment:
            # Reduce leverage during drawdowns
            cumulative = (1 + strategy_returns).cumprod()
            current_dd = (cumulative.max() - cumulative.iloc[-1]) / cumulative.max()

            if current_dd > 0.1:  # If in >10% drawdown
                dd_adjustment = 1.0 - min(0.5, current_dd / 0.2)  # Reduce up to 50%
                required_leverage *= dd_adjustment

        # Apply bounds
        required_leverage = np.clip(required_leverage, 0.1, 3.0)

        return required_leverage

    def scale_strategy_signals(self, base_signals: Dict, strategy_returns: pd.Series) -> Dict:
        """
        Scale strategy signals based on volatility target

        Args:
            base_signals: Signals from underlying strategy
            strategy_returns: Historical returns of strategy

        Returns:
            Scaled signals with volatility targeting
        """
        # Calculate required leverage
        required_leverage = self.calculate_required_leverage(strategy_returns)

        # Scale signals
        scaled_signals = {}

        if isinstance(base_signals, dict):
            for key, signal_info in base_signals.items():
                if "position_size" in signal_info:
                    scaled_position = signal_info["position_size"] * required_leverage

                    # Create scaled signal
                    scaled_signal = signal_info.copy()
                    scaled_signal["position_size"] = scaled_position
                    scaled_signal["vol_targeting"] = {
                        "required_leverage": required_leverage,
                        "target_volatility": self.target_volatility,
                        "strategy_volatility": (strategy_returns.std() * np.sqrt(DEFAULT_ANNUAL_LOOKBACK) if len(strategy_returns) > 0 else 0),
                    }

                    scaled_signals[key] = scaled_signal

        return scaled_signals

    def generate_portfolio_signal(self, portfolio_returns: pd.Series, current_exposure: float = 1.0) -> Dict:
        """
        Generate portfolio-level volatility targeting signal

        Args:
            portfolio_returns: Portfolio return series
            current_exposure: Current portfolio exposure

        Returns:
            Adjustment signal to maintain target volatility
        """
        if len(portfolio_returns) < 20:
            return {"adjustment": 0, "new_exposure": current_exposure}

        # Calculate portfolio volatility
        portfolio_vol = portfolio_returns.std() * np.sqrt(DEFAULT_ANNUAL_LOOKBACK)

        # Calculate required adjustment
        if portfolio_vol > 0:
            target_ratio = self.target_volatility / portfolio_vol
            required_adjustment = target_ratio - 1.0

            # Apply buffer
            if abs(required_adjustment) < self.volatility_buffer:
                required_adjustment = 0.0

            # Smooth adjustment
            smooth_adjustment = np.clip(required_adjustment, -0.2, 0.2)  # Max 20% adjustment

            new_exposure = current_exposure * (1 + smooth_adjustment)
            new_exposure = np.clip(new_exposure, 0.1, 3.0)  # 10% to 300% exposure
        else:
            smooth_adjustment = 0.0
            new_exposure = current_exposure

        return {
            "adjustment": smooth_adjustment,
            "new_exposure": new_exposure,
            "current_volatility": portfolio_vol,
            "target_volatility": self.target_volatility,
            "volatility_ratio": (self.target_volatility / portfolio_vol if portfolio_vol > 0 else 1.0),
            "metadata": {
                "strategy": "portfolio_vol_targeting",
                "lookforward_vol": self.lookforward_vol,
            },
        }
