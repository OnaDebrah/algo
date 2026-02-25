from typing import Dict, Optional

import numpy as np
import pandas as pd

from ...strategies.volatility.base_volatility import BaseVolatilityStrategy


class VolatilityTargetingStrategy(BaseVolatilityStrategy):
    """
    Volatility Targeting Strategy

    Dynamically adjusts portfolio exposure to maintain target volatility
    Used for portfolio-level risk control
    """

    def __init__(
        self,
        base_strategy: callable = None,  # Underlying strategy to scale
        lookforward_vol: bool = True,  # Use forward-looking vol estimates
        volatility_buffer: float = 0.1,  # 10% buffer around target
        max_drawdown_adjustment: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_strategy = base_strategy
        self.lookforward_vol = lookforward_vol
        self.volatility_buffer = volatility_buffer
        self.max_drawdown_adjustment = max_drawdown_adjustment

        # Portfolio tracking
        self.portfolio_returns = []
        self.drawdown_level = 0.0
        self.scale_factor = 1.0

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
        strategy_vol = strategy_returns.std() * np.sqrt(252)

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
                        "strategy_volatility": (strategy_returns.std() * np.sqrt(252) if len(strategy_returns) > 0 else 0),
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
        portfolio_vol = portfolio_returns.std() * np.sqrt(252)

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
