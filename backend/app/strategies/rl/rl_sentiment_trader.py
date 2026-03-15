"""
RL Sentiment Trader — Sentiment-augmented RL for trading.

Fuses technical indicators with sentiment scores to make trading decisions.
The RL agent learns complex non-linear mappings between sentiment + technicals
and optimal position sizing.

State space includes:
  - Standard technical features (RSI, MACD, BB, vol)
  - Sentiment score and momentum
  - Market regime indicator
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


class SentimentAwarePolicy(nn.Module):
    """
    Policy network with dedicated sentiment processing branch.

    Architecture:
      - Technical branch: processes price/volume features
      - Sentiment branch: processes sentiment features
      - Fusion layer: combines both branches
      - Actor + Critic heads
    """

    def __init__(self, tech_dim: int, sentiment_dim: int, hidden_dim: int = 128):
        super().__init__()

        # Technical feature branch
        self.tech_fc1 = nn.Linear(tech_dim, hidden_dim)
        self.tech_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)

        # Sentiment feature branch
        self.sent_fc1 = nn.Linear(sentiment_dim, hidden_dim // 2)
        self.sent_fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)

        # Fusion
        fusion_dim = hidden_dim // 2 + hidden_dim // 4
        self.fusion = nn.Linear(fusion_dim, hidden_dim // 2)

        # Actor
        self.actor = nn.Linear(hidden_dim // 2, 1)

        # Critic
        self.critic = nn.Linear(hidden_dim // 2, 1)

    def forward(self, tech_features: torch.Tensor, sent_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Process technical features
        tech = F.relu(self.tech_fc1(tech_features))
        tech = F.relu(self.tech_fc2(tech))

        # Process sentiment features
        sent = F.relu(self.sent_fc1(sent_features))
        sent = F.relu(self.sent_fc2(sent))

        # Fuse
        combined = torch.cat([tech, sent], dim=-1)
        fused = F.relu(self.fusion(combined))

        # Actor: position in [-1, 1]
        action = torch.tanh(self.actor(fused))

        # Critic: state value
        value = self.critic(fused)

        return action, value


class RLSentimentTrader(BaseRLStrategy):
    """
    Sentiment-augmented RL trader.

    Combines technical indicators with sentiment analysis to learn
    an optimal trading policy. The dual-branch architecture lets the
    agent weight technical vs. sentiment signals dynamically.

    When sentiment data is unavailable, falls back to technical-only trading
    with a neutral sentiment signal.
    """

    def __init__(
        self,
        name: str = "RL Sentiment Trader",
        lookback: int = 20,
        hidden_dim: int = 128,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        episodes: int = 500,
        transaction_cost: float = 0.001,
        sentiment_weight: float = 0.3,
        **kwargs,
    ):
        params = {
            "lookback": lookback,
            "hidden_dim": hidden_dim,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "episodes": episodes,
            "transaction_cost": transaction_cost,
            "sentiment_weight": sentiment_weight,
        }
        params.update(kwargs)
        super().__init__(name=name, params=params)

        self.hidden_dim = hidden_dim
        self.transaction_cost = transaction_cost
        self.sentiment_weight = sentiment_weight

        # Sentiment feature dim: score + momentum + std + regime
        self.sentiment_dim = 4
        self._tech_dim: int = 0

    def _build_policy(self, state_dim: int) -> nn.Module:
        """Build dual-branch policy network."""
        # state_dim here is the tech_dim (from trading env)
        self._tech_dim = state_dim
        policy = SentimentAwarePolicy(state_dim, self.sentiment_dim, self.hidden_dim)
        return policy.to(self.device)

    def _compute_sentiment_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract sentiment features from data.

        If a 'Sentiment' column exists in the data, uses it directly.
        Otherwise generates a synthetic neutral sentiment.

        Returns 4 features:
          1. Current sentiment score [-1, 1]
          2. Sentiment momentum (5-day change)
          3. Sentiment volatility (5-day std)
          4. Regime indicator (bullish/bearish/neutral from sentiment trend)
        """
        if "Sentiment" in data.columns:
            sentiment = data["Sentiment"].fillna(0).values.astype(float)
        else:
            # Synthetic neutral sentiment with slight momentum alignment
            close = data["Close"].values.astype(float)
            returns = np.diff(close, prepend=close[0]) / np.maximum(close, 1e-8)
            # Use smoothed returns as proxy sentiment
            sentiment = pd.Series(returns).rolling(5).mean().fillna(0).values
            sentiment = np.clip(sentiment * 10, -1, 1)  # Scale to [-1, 1]

        n = len(sentiment)

        # 1. Current sentiment
        current = float(sentiment[-1]) if n > 0 else 0.0

        # 2. Sentiment momentum (5-day change)
        if n >= 6:
            momentum = float(sentiment[-1] - sentiment[-6])
        else:
            momentum = 0.0

        # 3. Sentiment volatility
        if n >= 5:
            vol = float(np.std(sentiment[-5:]))
        else:
            vol = 0.0

        # 4. Regime from sentiment trend
        if n >= 10:
            recent_avg = float(np.mean(sentiment[-5:]))
            older_avg = float(np.mean(sentiment[-10:-5]))
            regime = np.clip(recent_avg - older_avg, -1, 1)
        else:
            regime = 0.0

        return np.array([current, momentum, vol, regime])

    def _build_state(self, data: pd.DataFrame) -> np.ndarray:
        """Build combined technical + sentiment state."""
        # Technical state from trading environment
        env = SingleAssetTradingEnv(data, lookback=self.lookback, transaction_cost=self.transaction_cost)
        env.current_step = min(len(data) - 1, max(self.lookback, len(data) - 1))
        tech_state = env._get_state()

        # Sentiment state
        sent_state = self._compute_sentiment_features(data)

        return np.concatenate([tech_state, sent_state])

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
                "sentiment_driven": abs(position) > 0.5,
            },
        }

    def generate_signal(self, data: pd.DataFrame) -> Union[int, Dict]:
        """Override to use dual-branch forward pass."""
        if self.policy is None or not self.is_trained:
            return 0

        # Build separate feature vectors
        env = SingleAssetTradingEnv(data, lookback=self.lookback, transaction_cost=self.transaction_cost)
        env.current_step = min(len(data) - 1, max(self.lookback, len(data) - 1))
        tech_state = env._get_state()

        sent_state = self._compute_sentiment_features(data)

        tech_tensor = torch.FloatTensor(tech_state).unsqueeze(0).to(self.device)
        sent_tensor = torch.FloatTensor(sent_state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            self.policy.eval()
            action, _ = self.policy(tech_tensor, sent_tensor)

        return self._action_to_signal(action.squeeze().cpu().numpy())

    def train(self, data: pd.DataFrame) -> Dict:
        """Train the sentiment-aware RL agent."""
        env = SingleAssetTradingEnv(
            data,
            lookback=self.lookback,
            transaction_cost=self.transaction_cost,
        )

        self.policy = self._build_policy(env.state_dim)
        optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        episode_returns = []
        best_return = float("-inf")

        for episode in range(self.episodes):
            state = env.reset()
            episode_reward = 0.0

            tech_states, sent_states, rewards = [], [], []

            while True:
                # Build sentiment features for current step
                window = data.iloc[:env.current_step + 1] if env.current_step < len(data) else data
                sent_features = self._compute_sentiment_features(window)

                tech_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                sent_tensor = torch.FloatTensor(sent_features).unsqueeze(0).to(self.device)

                action, value = self.policy(tech_tensor, sent_tensor)

                # Add exploration noise
                noise = torch.randn_like(action) * 0.1
                action_noisy = torch.tanh(action + noise).squeeze().detach().cpu().numpy()

                next_state, reward, done, info = env.step(float(action_noisy))

                # Sentiment accuracy bonus: if sentiment aligns with outcome
                step_return = info.get("step_return", 0)
                sentiment_score = sent_features[0]
                if (sentiment_score > 0 and step_return > 0) or (sentiment_score < 0 and step_return < 0):
                    reward += 0.01 * abs(sentiment_score)

                tech_states.append(state)
                sent_states.append(sent_features)
                rewards.append(reward)

                state = next_state
                episode_reward += reward

                if done:
                    break

            # Update
            if len(tech_states) > 10:
                self._sentiment_ppo_update(optimizer, tech_states, sent_states, rewards)

            episode_returns.append(episode_reward)
            if episode_reward > best_return:
                best_return = episode_reward

            if episode % 50 == 0:
                avg = np.mean(episode_returns[-50:]) if len(episode_returns) >= 50 else np.mean(episode_returns)
                logger.info(f"Sentiment Trader Episode {episode}/{self.episodes} | Avg: {avg:.4f} | Best: {best_return:.4f}")

        self.is_trained = True
        self.training_history = {
            "episode_returns": episode_returns,
            "best_return": best_return,
        }
        return self.training_history

    def _sentiment_ppo_update(self, optimizer, tech_states, sent_states, rewards):
        """PPO update with dual-branch inputs."""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns_t = torch.FloatTensor(returns).to(self.device)
        tech_t = torch.FloatTensor(np.array(tech_states)).to(self.device)
        sent_t = torch.FloatTensor(np.array(sent_states)).to(self.device)

        actions, values = self.policy(tech_t, sent_t)

        advantages = returns_t - values.squeeze()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_loss = -(advantages.detach() * actions.squeeze()).mean()
        value_loss = F.mse_loss(values.squeeze(), returns_t.detach())

        loss = policy_loss + self.value_coef * value_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        optimizer.step()
