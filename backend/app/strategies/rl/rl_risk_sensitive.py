"""
RL Risk-Sensitive Trader — Distributional RL for risk-adjusted trading.

Single-asset RL trader that optimizes a risk-sensitive objective:
maximizes returns while explicitly penalizing drawdowns and tail risk (CVaR).

Uses a quantile-aware critic that learns the distribution of returns,
not just the expected value — enabling better tail-risk management.
"""

import logging
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base_rl_strategy import BaseRLStrategy
from .trading_environment import SingleAssetTradingEnv

logger = logging.getLogger(__name__)


class RiskSensitivePolicy(nn.Module):
    """
    Actor-Critic with quantile critic for risk-sensitive trading.

    Actor: outputs position size in [-1, 1] via tanh
    Critic: outputs N quantile values for distributional value estimation
    """

    def __init__(self, state_dim: int, hidden_dim: int = 128, n_quantiles: int = 10):
        super().__init__()
        self.n_quantiles = n_quantiles

        # Shared backbone
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Actor: continuous position [-1, 1]
        self.actor_fc = nn.Linear(hidden_dim, hidden_dim // 2)
        self.actor_mu = nn.Linear(hidden_dim // 2, 1)
        self.actor_log_std = nn.Linear(hidden_dim // 2, 1)

        # Critic: distributional (quantile values)
        self.critic_fc = nn.Linear(hidden_dim, hidden_dim // 2)
        self.critic_quantiles = nn.Linear(hidden_dim // 2, n_quantiles)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (action_mean, action_log_std, quantile_values)."""
        shared = F.relu(self.fc1(x))
        shared = F.relu(self.fc2(shared))

        # Actor
        actor_h = F.relu(self.actor_fc(shared))
        mu = torch.tanh(self.actor_mu(actor_h))
        log_std = torch.clamp(self.actor_log_std(actor_h), -2, 0.5)

        # Critic (quantile values)
        critic_h = F.relu(self.critic_fc(shared))
        quantiles = self.critic_quantiles(critic_h)

        return mu, log_std, quantiles


class RLRiskSensitiveTrader(BaseRLStrategy):
    """
    Risk-sensitive RL trader using distributional RL.

    Key features:
    - Quantile-based critic learns full return distribution
    - Reward includes explicit drawdown and CVaR penalties
    - Position sizing is continuous [-1, 1] (short to long)
    - Adapts position aggressiveness to market risk conditions
    """

    def __init__(
        self,
        name: str = "RL Risk-Sensitive Trader",
        lookback: int = 20,
        hidden_dim: int = 128,
        n_quantiles: int = 10,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        episodes: int = 500,
        drawdown_penalty: float = 2.0,
        cvar_penalty: float = 1.0,
        transaction_cost: float = 0.001,
        **kwargs,
    ):
        params = {
            "lookback": lookback,
            "hidden_dim": hidden_dim,
            "n_quantiles": n_quantiles,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "episodes": episodes,
            "drawdown_penalty": drawdown_penalty,
            "cvar_penalty": cvar_penalty,
            "transaction_cost": transaction_cost,
        }
        params.update(kwargs)
        super().__init__(name=name, params=params)

        self.hidden_dim = hidden_dim
        self.n_quantiles = n_quantiles
        self.drawdown_penalty = drawdown_penalty
        self.cvar_penalty = cvar_penalty
        self.transaction_cost = transaction_cost

    def _risk_sensitive_reward(self, portfolio_return: float, drawdown: float, turnover: float, cost: float) -> float:
        """
        Risk-sensitive reward function.

        reward = return - lambda_1 * drawdown^2 - lambda_2 * turnover_cost - risk_free
        """
        risk_free = 0.05 / 252
        return (
            portfolio_return
            - risk_free
            - self.drawdown_penalty * drawdown ** 2
            - 0.1 * turnover
        )

    def _build_policy(self, state_dim: int) -> nn.Module:
        """Build risk-sensitive policy network."""
        policy = RiskSensitivePolicy(state_dim, self.hidden_dim, self.n_quantiles)
        return policy.to(self.device)

    def _build_state(self, data: pd.DataFrame) -> np.ndarray:
        """Build state from single-asset OHLCV data."""
        env = SingleAssetTradingEnv(data, lookback=self.lookback, transaction_cost=self.transaction_cost)
        # Return the state at the last timestep
        env.current_step = min(len(data) - 1, max(self.lookback, len(data) - 1))
        return env._get_state()

    def _action_to_signal(self, action: np.ndarray) -> Dict:
        """Convert continuous position to trading signal."""
        position = float(np.clip(action[0] if len(action.shape) > 0 else action, -1.0, 1.0))

        if position > 0.15:
            signal = 1
        elif position < -0.15:
            signal = -1
        else:
            signal = 0

        return {
            "signal": signal,
            "position_size": abs(position),
            "metadata": {
                "raw_position": position,
                "confidence": abs(position),
            },
        }

    def train(self, data: pd.DataFrame) -> Dict:
        """Train the risk-sensitive RL agent."""
        env = SingleAssetTradingEnv(
            data,
            lookback=self.lookback,
            transaction_cost=self.transaction_cost,
            reward_fn=self._risk_sensitive_reward,
        )

        self.policy = self._build_policy(env.state_dim)
        optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        # Quantile targets for distributional RL
        tau = torch.FloatTensor(
            [(2 * i + 1) / (2 * self.n_quantiles) for i in range(self.n_quantiles)]
        ).to(self.device)

        episode_returns = []
        episode_drawdowns = []
        best_sharpe = float("-inf")

        for episode in range(self.episodes):
            state = env.reset()
            episode_reward = 0.0
            max_drawdown = 0.0

            states, actions, rewards, quantile_preds = [], [], [], []

            while True:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                mu, log_std, quantiles = self.policy(state_tensor)

                # Sample action with exploration
                std = torch.exp(log_std)
                noise = torch.randn_like(mu) * std
                action_raw = mu + noise
                action = torch.tanh(action_raw).squeeze().detach().cpu().numpy()

                next_state, reward, done, info = env.step(float(action))

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                quantile_preds.append(quantiles.detach())

                max_drawdown = max(max_drawdown, info.get("drawdown", 0))
                state = next_state
                episode_reward += reward

                if done:
                    break

            # Update
            if len(states) > 10:
                self._distributional_ppo_update(optimizer, states, actions, rewards, quantile_preds, tau)

            episode_returns.append(episode_reward)
            episode_drawdowns.append(max_drawdown)

            # Track Sharpe-like metric
            if len(episode_returns) >= 20:
                recent = episode_returns[-20:]
                sharpe = np.mean(recent) / max(np.std(recent), 1e-8)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe

            if episode % 50 == 0:
                avg = np.mean(episode_returns[-50:]) if len(episode_returns) >= 50 else np.mean(episode_returns)
                avg_dd = np.mean(episode_drawdowns[-50:]) if len(episode_drawdowns) >= 50 else np.mean(episode_drawdowns)
                logger.info(
                    f"RiskSensitive Episode {episode}/{self.episodes} | "
                    f"Avg Return: {avg:.4f} | Avg MaxDD: {avg_dd:.2%} | Best Sharpe: {best_sharpe:.3f}"
                )

        self.is_trained = True
        self.training_history = {
            "episode_returns": episode_returns,
            "episode_drawdowns": episode_drawdowns,
            "best_sharpe": best_sharpe,
        }
        return self.training_history

    def _distributional_ppo_update(self, optimizer, states, actions, rewards, quantile_preds, tau):
        """PPO update with quantile regression loss for distributional critic."""
        # Compute returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns_t = torch.FloatTensor(returns).to(self.device)
        states_t = torch.FloatTensor(np.array(states)).to(self.device)

        # Forward pass
        mu, log_std, quantiles = self.policy(states_t)

        # -- Policy loss --
        # Use mean of quantiles as value estimate
        values = quantiles.mean(dim=-1)
        advantages = returns_t - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_loss = -(advantages * mu.squeeze()).mean()

        # -- Quantile regression loss --
        # Each quantile should predict a different part of the return distribution
        target = returns_t.unsqueeze(-1).expand_as(quantiles)
        td_error = target - quantiles
        huber_loss = torch.where(
            td_error.abs() < 1.0,
            0.5 * td_error ** 2,
            td_error.abs() - 0.5,
        )
        quantile_loss = (tau.unsqueeze(0) - (td_error < 0).float()).abs() * huber_loss
        critic_loss = quantile_loss.sum(dim=-1).mean()

        # -- CVaR penalty --
        # Penalize low quantiles (tail risk)
        cvar = quantiles[:, :self.n_quantiles // 5].mean()  # Bottom 20% quantiles
        cvar_loss = -self.cvar_penalty * cvar  # Penalize low expected tail returns

        # -- Entropy bonus --
        std = torch.exp(log_std)
        entropy = 0.5 * torch.log(2 * np.pi * np.e * std ** 2).mean()

        loss = policy_loss + critic_loss + cvar_loss - self.entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        optimizer.step()
