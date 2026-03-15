"""
RL Regime Allocator — Hierarchical RL for multi-strategy allocation.

A "meta-strategy" that uses RL to dynamically allocate capital across
existing sub-strategies based on detected market regime.

The RL agent observes:
  - Market regime features (volatility, trend, Hurst exponent)
  - Recent performance of each sub-strategy
  - Current drawdown state

And learns which sub-strategies to allocate capital to.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base_rl_strategy import BaseRLStrategy

logger = logging.getLogger(__name__)

# Number of sub-strategies the allocator can choose from
N_SUB_STRATEGIES = 6
SUB_STRATEGY_NAMES = ["SMA Crossover", "RSI", "Bollinger Bands", "MACD", "Momentum", "Vol Breakout"]


class RegimeAllocatorPolicy(nn.Module):
    """
    Policy network for regime-aware strategy allocation.
    Outputs allocation weights across N sub-strategies.
    """

    def __init__(self, state_dim: int, n_strategies: int = N_SUB_STRATEGIES, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)

        # Actor: allocation weights across strategies
        self.actor = nn.Linear(hidden_dim // 2, n_strategies)
        # Critic: state value
        self.critic = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        weights = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return weights, value


class RLRegimeAllocator(BaseRLStrategy):
    """
    Hierarchical RL strategy allocator.

    Uses market regime detection to build state features, then
    an RL policy to allocate capital across sub-strategies.
    The sub-strategies generate signals independently; the RL agent
    learns the optimal weighting based on regime + performance history.
    """

    def __init__(
        self,
        name: str = "RL Regime Allocator",
        lookback: int = 60,
        hidden_dim: int = 128,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        episodes: int = 500,
        performance_window: int = 20,
        **kwargs,
    ):
        params = {
            "lookback": lookback,
            "hidden_dim": hidden_dim,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "episodes": episodes,
            "performance_window": performance_window,
        }
        params.update(kwargs)
        super().__init__(name=name, params=params)

        self.hidden_dim = hidden_dim
        self.performance_window = performance_window
        self.n_strategies = N_SUB_STRATEGIES

        # State: regime features (5) + sub-strategy returns (N * perf_window) + drawdown (1) + vol (1)
        self._state_dim = 5 + self.n_strategies * 3 + 2  # 5 regime + 3 stats per strategy + 2 global

    def _build_policy(self, state_dim: int) -> nn.Module:
        """Build the regime allocator policy."""
        policy = RegimeAllocatorPolicy(state_dim, self.n_strategies, self.hidden_dim)
        return policy.to(self.device)

    def _compute_regime_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract regime features from price data.
        Returns 5 features: vol_regime, trend_strength, mean_reversion, momentum, vol_level.
        """
        close = data["Close"].values.astype(float)
        n = len(close)

        if n < 20:
            return np.zeros(5)

        returns = np.diff(close) / np.maximum(close[:-1], 1e-8)

        # 1. Volatility regime (normalized rolling vol)
        vol_20 = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
        vol_60 = np.std(returns[-60:]) if len(returns) >= 60 else vol_20
        vol_regime = vol_20 / max(vol_60, 1e-8)

        # 2. Trend strength (R-squared of linear fit)
        x = np.arange(min(n, 20))
        y = close[-len(x):]
        if len(x) > 1:
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            trend_strength = 1 - ss_res / max(ss_tot, 1e-8)
        else:
            trend_strength = 0.0

        # 3. Mean reversion signal (autocorrelation of returns)
        if len(returns) >= 5:
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            mean_reversion = -autocorr  # Negative autocorr = mean reversion
        else:
            mean_reversion = 0.0

        # 4. Momentum (20-day return)
        momentum = (close[-1] / max(close[-min(20, n)], 1e-8)) - 1

        # 5. Volatility level (annualized)
        vol_level = vol_20 * np.sqrt(252)

        return np.array([vol_regime, trend_strength, mean_reversion, momentum, vol_level])

    def _simulate_sub_strategy_returns(self, data: pd.DataFrame) -> np.ndarray:
        """
        Compute recent performance stats for each sub-strategy.
        Returns 3 stats per strategy: avg return, win rate, Sharpe.
        """
        close = data["Close"].values.astype(float)
        returns = np.diff(close) / np.maximum(close[:-1], 1e-8)
        n = len(returns)

        if n < 20:
            return np.zeros(self.n_strategies * 3)

        stats = []
        for i in range(self.n_strategies):
            # Generate simplified signals for each sub-strategy type
            signals = self._generate_sub_signals(data, strategy_idx=i)
            strategy_returns = signals[1:] * returns[-len(signals[1:]):]
            recent = strategy_returns[-self.performance_window:]

            avg_ret = float(np.mean(recent)) if len(recent) > 0 else 0.0
            win_rate = float(np.mean(recent > 0)) if len(recent) > 0 else 0.5
            sharpe = float(np.mean(recent) / max(np.std(recent), 1e-8)) if len(recent) > 0 else 0.0

            stats.extend([avg_ret, win_rate, sharpe])

        return np.array(stats)

    def _generate_sub_signals(self, data: pd.DataFrame, strategy_idx: int) -> np.ndarray:
        """Generate simplified signals for sub-strategy evaluation."""
        close = data["Close"].values.astype(float)
        n = len(close)
        signals = np.zeros(n)

        if n < 20:
            return signals

        if strategy_idx == 0:  # SMA Crossover
            sma_fast = pd.Series(close).rolling(10).mean().values
            sma_slow = pd.Series(close).rolling(30).mean().values
            signals[30:] = np.where(sma_fast[30:] > sma_slow[30:], 1, -1)

        elif strategy_idx == 1:  # RSI
            delta = np.diff(close, prepend=close[0])
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = pd.Series(gain).rolling(14).mean().fillna(0).values
            avg_loss = pd.Series(loss).rolling(14).mean().fillna(0).values
            rs = avg_gain / np.maximum(avg_loss, 1e-8)
            rsi = 100 - (100 / (1 + rs))
            signals = np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0)).astype(float)

        elif strategy_idx == 2:  # Bollinger Bands
            sma = pd.Series(close).rolling(20).mean().fillna(close[0]).values
            std = pd.Series(close).rolling(20).std().fillna(1).values
            upper = sma + 2 * std
            lower = sma - 2 * std
            signals = np.where(close < lower, 1, np.where(close > upper, -1, 0)).astype(float)

        elif strategy_idx == 3:  # MACD
            ema12 = pd.Series(close).ewm(span=12).mean().values
            ema26 = pd.Series(close).ewm(span=26).mean().values
            macd = ema12 - ema26
            signal_line = pd.Series(macd).ewm(span=9).mean().values
            signals = np.where(macd > signal_line, 1, -1).astype(float)

        elif strategy_idx == 4:  # Momentum
            mom = pd.Series(close).pct_change(20).fillna(0).values
            signals = np.where(mom > 0.05, 1, np.where(mom < -0.05, -1, 0)).astype(float)

        elif strategy_idx == 5:  # Vol Breakout
            sma = pd.Series(close).rolling(20).mean().fillna(close[0]).values
            std = pd.Series(close).rolling(20).std().fillna(1).values
            upper = sma + 1.5 * std
            lower = sma - 1.5 * std
            signals = np.where(close > upper, 1, np.where(close < lower, -1, 0)).astype(float)

        return signals

    def _build_state(self, data: pd.DataFrame) -> np.ndarray:
        """Build full state vector: regime features + strategy performance + global stats."""
        regime = self._compute_regime_features(data)
        strategy_stats = self._simulate_sub_strategy_returns(data)

        close = data["Close"].values.astype(float)
        returns = np.diff(close) / np.maximum(close[:-1], 1e-8)

        # Global stats: drawdown, rolling vol
        peak = np.maximum.accumulate(close)
        drawdown = float((peak[-1] - close[-1]) / max(peak[-1], 1e-8))
        vol = float(np.std(returns[-20:])) if len(returns) >= 20 else 0.0

        return np.concatenate([regime, strategy_stats, [drawdown, vol]])

    def _action_to_signal(self, action: np.ndarray) -> Dict:
        """Convert allocation weights to a trading signal."""
        # Find dominant strategy
        weights = np.clip(action, 0, 1)
        weight_sum = weights.sum()
        if weight_sum > 1e-8:
            weights = weights / weight_sum
        else:
            weights = np.ones(self.n_strategies) / self.n_strategies

        dominant_idx = int(np.argmax(weights))
        confidence = float(weights[dominant_idx])

        # Map dominant strategy to signal direction
        # Positive = trend-following strategies favored, negative = mean-reversion
        trend_weight = weights[0] + weights[3] + weights[4]   # SMA + MACD + Momentum
        reversion_weight = weights[1] + weights[2] + weights[5]  # RSI + BB + VolBreakout

        if trend_weight > reversion_weight + 0.15:
            signal = 1
        elif reversion_weight > trend_weight + 0.15:
            signal = -1
        else:
            signal = 0

        return {
            "signal": signal,
            "position_size": confidence,
            "metadata": {
                "allocation_weights": {SUB_STRATEGY_NAMES[i]: float(weights[i]) for i in range(self.n_strategies)},
                "dominant_strategy": SUB_STRATEGY_NAMES[dominant_idx],
                "confidence": confidence,
            },
        }

    def train(self, data: pd.DataFrame) -> Dict:
        """Train the regime allocator on historical data."""
        self.policy = self._build_policy(self._state_dim)
        optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        episode_returns = []
        best_return = float("-inf")
        n = len(data)

        for episode in range(self.episodes):
            # Random start within the data to diversify training
            start = np.random.randint(self.lookback, max(n - 200, self.lookback + 1))
            end = min(start + np.random.randint(100, 300), n)
            episode_data = data.iloc[:end]

            episode_reward = 0.0
            states, rewards = [], []

            for t in range(start, end - 1):
                window = episode_data.iloc[:t + 1]
                state = self._build_state(window)
                states.append(state)

                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                weights, value = self.policy(state_tensor)

                # Add exploration noise
                noise = torch.randn_like(weights) * 0.1
                action = F.softmax(weights + noise, dim=-1).squeeze().detach().cpu().numpy()

                # Compute weighted portfolio return
                close = episode_data["Close"].values.astype(float)
                actual_return = (close[t + 1] - close[t]) / max(close[t], 1e-8)

                # Get signal from each sub-strategy and weight
                sub_signals = np.array([
                    self._generate_sub_signals(window, i)[-1] for i in range(self.n_strategies)
                ])
                portfolio_return = float(np.dot(action, sub_signals)) * actual_return

                # Reward: return with drawdown penalty
                reward = portfolio_return - 0.01 * abs(portfolio_return) if portfolio_return < 0 else portfolio_return
                rewards.append(reward)
                episode_reward += reward

            # PPO update
            if len(states) > 10:
                self._ppo_update(optimizer, states, rewards)

            episode_returns.append(episode_reward)
            if episode_reward > best_return:
                best_return = episode_reward

            if episode % 50 == 0:
                avg = np.mean(episode_returns[-50:]) if len(episode_returns) >= 50 else np.mean(episode_returns)
                logger.info(f"Regime Allocator Episode {episode}/{self.episodes} | Avg: {avg:.4f} | Best: {best_return:.4f}")

        self.is_trained = True
        self.training_history = {
            "episode_returns": episode_returns,
            "best_return": best_return,
        }
        return self.training_history

    def _ppo_update(self, optimizer, states, rewards):
        """Simplified PPO update."""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns_t = torch.FloatTensor(returns).to(self.device)
        states_t = torch.FloatTensor(np.array(states)).to(self.device)

        weights, values = self.policy(states_t)

        advantages = returns_t - values.squeeze()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_loss = -(advantages.detach() * weights.mean(dim=-1)).mean()
        value_loss = F.mse_loss(values.squeeze(), returns_t.detach())
        entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean()

        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        optimizer.step()
