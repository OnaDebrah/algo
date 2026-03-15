"""
Reusable trading environments for RL strategies.
Refactored from drl_portfolio.py with configurable state/reward/action spaces.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SingleAssetTradingEnv:
    """
    Single-asset trading environment for RL agents.

    State:  configurable feature vector (technical indicators, sentiment, regime, etc.)
    Action: continuous position size in [-1, 1] (short to long)
    Reward: configurable reward function (default: risk-adjusted return)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        lookback: int = 20,
        transaction_cost: float = 0.001,
        reward_fn: Optional[Callable] = None,
        initial_capital: float = 100_000,
    ):
        self.data = data
        self.lookback = lookback
        self.transaction_cost = transaction_cost
        self.initial_capital = initial_capital
        self.reward_fn = reward_fn or self._default_reward

        # Compute returns
        self.returns = data["Close"].pct_change().fillna(0).values

        # Build feature matrix
        if feature_columns and all(c in data.columns for c in feature_columns):
            self.features = data[feature_columns].fillna(0).values
        else:
            self.features = self._build_default_features(data)

        self.n_features = self.features.shape[1]
        self.n_steps = len(data)

        # State dimension: lookback * features + position + drawdown + step_frac
        self.state_dim = lookback * self.n_features + 3

        # Episode state
        self.current_step = 0
        self.position = 0.0
        self.portfolio_value = initial_capital
        self.peak_value = initial_capital
        self.trades: List[Dict] = []

    def _build_default_features(self, data: pd.DataFrame) -> np.ndarray:
        """Build default technical feature matrix from OHLCV data."""
        close = data["Close"].values.astype(float)
        high = data["High"].values.astype(float)
        low = data["Low"].values.astype(float)
        volume = data["Volume"].values.astype(float)

        features = []

        # Returns (1-day, 5-day, 20-day)
        ret_1 = np.diff(close, prepend=close[0]) / np.maximum(close, 1e-8)
        features.append(ret_1)

        ret_5 = np.zeros_like(close)
        ret_5[5:] = (close[5:] - close[:-5]) / np.maximum(close[:-5], 1e-8)
        features.append(ret_5)

        # Volatility (20-day rolling std of returns)
        vol = pd.Series(ret_1).rolling(20).std().fillna(0).values
        features.append(vol)

        # RSI (14-period)
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14).mean().fillna(0).values
        avg_loss = pd.Series(loss).rolling(14).mean().fillna(0).values
        rs = avg_gain / np.maximum(avg_loss, 1e-8)
        rsi = 100 - (100 / (1 + rs))
        features.append(rsi / 100)  # Normalize to [0,1]

        # Bollinger Band position (where price sits relative to bands)
        sma_20 = pd.Series(close).rolling(20).mean().fillna(close[0]).values
        std_20 = pd.Series(close).rolling(20).std().fillna(1).values
        bb_pos = (close - sma_20) / np.maximum(2 * std_20, 1e-8)
        features.append(np.clip(bb_pos, -2, 2))

        # Volume ratio
        vol_sma = pd.Series(volume).rolling(20).mean().fillna(volume.mean()).values
        vol_ratio = volume / np.maximum(vol_sma, 1e-8)
        features.append(np.clip(vol_ratio, 0, 5))

        # High-Low range normalized
        hl_range = (high - low) / np.maximum(close, 1e-8)
        features.append(hl_range)

        return np.column_stack(features)

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_step = self.lookback
        self.position = 0.0
        self.portfolio_value = self.initial_capital
        self.peak_value = self.initial_capital
        self.trades = []
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Build state vector: feature window + position + drawdown + progress."""
        window = self.features[self.current_step - self.lookback : self.current_step]
        flat_features = window.flatten()

        drawdown = (self.peak_value - self.portfolio_value) / max(self.peak_value, 1e-8)
        step_frac = self.current_step / max(self.n_steps - 1, 1)

        return np.concatenate([flat_features, [self.position, drawdown, step_frac]])

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step.

        Args:
            action: target position in [-1, 1]

        Returns:
            (next_state, reward, done, info)
        """
        action = float(np.clip(action, -1.0, 1.0))

        # Transaction cost for position change
        turnover = abs(action - self.position)
        cost = turnover * self.transaction_cost * self.portfolio_value

        # Record trade if position changed
        if abs(action - self.position) > 0.01:
            self.trades.append({
                "step": self.current_step,
                "old_pos": self.position,
                "new_pos": action,
                "turnover": turnover,
            })

        # Update position
        old_position = self.position
        self.position = action

        # Advance step
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1

        if not done:
            # Portfolio return
            step_return = self.returns[self.current_step]
            portfolio_return = self.position * step_return
            self.portfolio_value *= (1 + portfolio_return)
            self.portfolio_value -= cost
            self.peak_value = max(self.peak_value, self.portfolio_value)

            # Compute reward
            drawdown = (self.peak_value - self.portfolio_value) / max(self.peak_value, 1e-8)
            reward = self.reward_fn(portfolio_return, drawdown, turnover, cost)
        else:
            reward = 0.0
            portfolio_return = 0.0
            drawdown = 0.0

        info = {
            "portfolio_value": self.portfolio_value,
            "position": self.position,
            "drawdown": drawdown,
            "turnover": turnover,
            "cost": cost,
            "step_return": portfolio_return if not done else 0.0,
        }

        next_state = self._get_state() if not done else np.zeros(self.state_dim)
        return next_state, reward, done, info

    @staticmethod
    def _default_reward(portfolio_return: float, drawdown: float, turnover: float, cost: float) -> float:
        """Default reward: return minus risk penalty."""
        risk_free = 0.05 / 252  # Daily risk-free rate
        return (portfolio_return - risk_free) - 0.5 * drawdown - 0.1 * turnover


class MultiAssetTradingEnv:
    """
    Multi-asset trading environment for portfolio allocation RL agents.
    Wraps the pattern from drl_portfolio.py's PortfolioEnvironment.

    State:  price windows + returns + current weights
    Action: target portfolio weights (softmax, sum to 1)
    Reward: portfolio return minus costs
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        lookback: int = 20,
        transaction_cost: float = 0.001,
        initial_capital: float = 100_000,
    ):
        self.prices = prices
        self.n_assets = len(prices.columns)
        self.lookback = lookback
        self.transaction_cost = transaction_cost
        self.initial_capital = initial_capital

        self.returns = prices.pct_change().fillna(0)

        # State dim: lookback * n_assets (prices) + lookback * n_assets (returns) + n_assets (weights)
        self.state_dim = 2 * lookback * self.n_assets + self.n_assets

        self.current_step = 0
        self.weights = np.zeros(self.n_assets)
        self.portfolio_value = initial_capital
        self.peak_value = initial_capital
        self.trades: List[Dict] = []

    def reset(self) -> np.ndarray:
        """Reset to initial state."""
        self.current_step = self.lookback
        self.weights = np.zeros(self.n_assets)
        self.portfolio_value = self.initial_capital
        self.peak_value = self.initial_capital
        self.trades = []
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Build state: normalized price window + return window + current weights."""
        price_window = self.prices.iloc[self.current_step - self.lookback : self.current_step].values
        # Normalize prices by first row in window
        price_norm = price_window / np.maximum(price_window[0], 1e-8)

        return_window = self.returns.iloc[self.current_step - self.lookback : self.current_step].values

        return np.concatenate([price_norm.flatten(), return_window.flatten(), self.weights])

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step with target portfolio weights.

        Args:
            action: target weights array (will be normalized to sum to 1)
        """
        # Normalize weights
        action = np.clip(action, 0, 1)
        weight_sum = action.sum()
        if weight_sum > 1e-8:
            action = action / weight_sum
        else:
            action = np.ones(self.n_assets) / self.n_assets

        # Transaction cost
        turnover = np.sum(np.abs(action - self.weights))
        cost = turnover * self.transaction_cost * self.portfolio_value

        self.weights = action
        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1

        if not done:
            step_returns = self.returns.iloc[self.current_step].values
            portfolio_return = float(np.dot(self.weights, step_returns))
            self.portfolio_value *= (1 + portfolio_return)
            self.portfolio_value -= cost
            self.peak_value = max(self.peak_value, self.portfolio_value)

            reward = portfolio_return - 0.02 / 252  # Sharpe contribution
        else:
            reward = 0.0
            portfolio_return = 0.0

        info = {
            "portfolio_value": self.portfolio_value,
            "weights": self.weights.copy(),
            "turnover": turnover,
            "cost": cost,
        }

        next_state = self._get_state() if not done else np.zeros(self.state_dim)
        return next_state, reward, done, info
