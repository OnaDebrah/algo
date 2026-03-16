"""
RL Portfolio Allocator — wraps the existing DRLPortfolioOptimizer
into a proper BaseStrategy for catalog integration.

Multi-asset strategy: learns optimal portfolio weight allocation
using PPO actor-critic with continuous action space.
"""

import logging
from typing import Dict, Optional, Tuple, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base_rl_strategy import BaseRLStrategy
from .trading_environment import MultiAssetTradingEnv

logger = logging.getLogger(__name__)


class PortfolioPolicyNetwork(nn.Module):
    """Actor-Critic policy for portfolio weight allocation."""

    def __init__(self, state_dim: int, n_assets: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)

        # Actor: portfolio weights (softmax → sum to 1)
        self.actor = nn.Linear(hidden_dim // 2, n_assets)
        # Critic: state value
        self.critic = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        weights = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return weights, value


class RLPortfolioAllocator(BaseRLStrategy):
    """
    Deep RL portfolio allocator using PPO.

    Learns to dynamically allocate capital across multiple assets
    by maximizing risk-adjusted returns net of transaction costs.

    This wraps and extends the existing drl_portfolio.py infrastructure
    into a proper BaseStrategy for catalog and backtest integration.
    """

    def __init__(
        self,
        name: str = "RL Portfolio Allocator",
        lookback: int = 20,
        hidden_dim: int = 128,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        episodes: int = 500,
        transaction_cost: float = 0.001,
        **kwargs,
    ):
        params = {
            "lookback": lookback,
            "hidden_dim": hidden_dim,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "episodes": episodes,
            "transaction_cost": transaction_cost,
        }
        params.update(kwargs)
        super().__init__(name=name, params=params)

        self.hidden_dim = hidden_dim
        self.transaction_cost = transaction_cost
        self.n_assets: Optional[int] = None

    def _build_policy(self, state_dim: int) -> nn.Module:
        """Build portfolio allocation policy network."""
        if self.n_assets is None:
            raise ValueError("n_assets must be set before building policy")
        policy = PortfolioPolicyNetwork(state_dim, self.n_assets, self.hidden_dim)
        return policy.to(self.device)

    def _build_state(self, data: pd.DataFrame) -> np.ndarray:
        """Build state from multi-asset price data."""
        if len(data) < self.lookback:
            return np.zeros(self.policy.fc1.in_features if self.policy else 1)

        prices = cast(pd.DataFrame, cast(object, data.iloc[-self.lookback :]))
        returns = cast(pd.DataFrame, cast(object, data.pct_change().fillna(0).iloc[-self.lookback :]))

        # Normalize prices by first row
        price_norm = prices.values / np.maximum(prices.values[0], 1e-8)

        weights = np.zeros(self.n_assets or prices.shape[1])
        return np.concatenate([price_norm.flatten(), returns.values.flatten(), weights])

    def _action_to_signal(self, action: np.ndarray) -> Dict:
        """Convert portfolio weights to a signal dict."""
        # Normalize to sum to 1
        action = np.clip(action, 0, 1)
        weight_sum = action.sum()
        if weight_sum > 1e-8:
            action = action / weight_sum
        else:
            action = np.ones_like(action) / len(action)

        # Net signal: >0.5 total long weight = buy, <0.5 = sell
        net_position = float(action.sum())
        signal = 1 if net_position > 0.6 else (-1 if net_position < 0.4 else 0)

        return {
            "signal": signal,
            "position_size": float(np.max(action)),
            "metadata": {
                "weights": action.tolist(),
                "confidence": float(np.max(action)),
            },
        }

    def train(self, data: pd.DataFrame) -> Dict:
        """
        Train the RL agent on multi-asset historical price data.

        Args:
            data: DataFrame with multiple asset columns (each column = asset prices)
        """
        self.n_assets = data.shape[1]
        env = MultiAssetTradingEnv(
            prices=data,
            lookback=self.lookback,
            transaction_cost=self.transaction_cost,
        )

        # Build policy
        self.policy = self._build_policy(env.state_dim)
        optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        episode_returns = []
        best_return = float("-inf")

        for episode in range(self.episodes):
            state = env.reset()
            episode_reward = 0.0

            states, actions, rewards, values, log_probs = [], [], [], [], []

            while True:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                weights, value = self.policy(state_tensor)

                # Add exploration noise
                noise = torch.randn_like(weights) * 0.05
                action = F.softmax(weights + noise, dim=-1).squeeze().detach().cpu().numpy()

                next_state, reward, done, info = env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                values.append(value.item())
                log_probs.append(-0.5 * float(np.sum((noise.squeeze().numpy()) ** 2) / 0.05**2))

                state = next_state
                episode_reward += reward

                if done:
                    break

            # PPO update
            if len(states) > 0:
                self._ppo_update(optimizer, states, actions, rewards, values, log_probs)

            episode_returns.append(episode_reward)

            if episode_reward > best_return:
                best_return = episode_reward

            if episode % 50 == 0:
                avg_return = np.mean(episode_returns[-50:]) if len(episode_returns) >= 50 else np.mean(episode_returns)
                logger.info(f"Episode {episode}/{self.episodes} | Avg Return: {avg_return:.4f} | Best: {best_return:.4f}")

        self.is_trained = True
        self.training_history = {
            "episode_returns": episode_returns,
            "best_return": best_return,
            "episodes_trained": len(episode_returns),
        }

        logger.info(f"Training complete: {len(episode_returns)} episodes, best return: {best_return:.4f}")
        return self.training_history

    def _ppo_update(self, optimizer, states, actions, rewards, values, log_probs):
        """Single PPO update step."""
        # Compute returns and advantages
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns_t = torch.FloatTensor(returns).to(self.device)
        values_t = torch.FloatTensor(values).to(self.device)
        advantages = returns_t - values_t
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t = torch.FloatTensor(np.array(states)).to(self.device)

        # Forward pass
        new_weights, new_values = self.policy(states_t)

        # Policy loss (simplified PPO)
        policy_loss = -(advantages.detach() * new_weights.mean(dim=-1)).mean()

        # Value loss
        value_loss = F.mse_loss(new_values.squeeze(), returns_t.detach())

        # Entropy bonus
        entropy = -(new_weights * (new_weights + 1e-8).log()).sum(dim=-1).mean()

        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        optimizer.step()
