"""
Deep Reinforcement Learning for optimal portfolio allocation [citation:5][citation:10]
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

logger = logging.getLogger(__name__)


class PortfolioEnvironment:
    """
    Trading environment for portfolio optimization
    """

    def __init__(self, prices: pd.DataFrame, initial_capital: float = 100000, transaction_cost: float = 0.001, lookback: int = 20):
        self.weights = None
        self.trades = None
        self.capital = None
        self.current_step = None
        self.holdings = None
        self.portfolio_value = None
        self.prices = prices
        self.n_assets = len(prices.columns)
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.lookback = lookback

        self.returns = prices.pct_change().dropna()

        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = self.lookback
        self.capital = self.initial_capital
        self.holdings = np.zeros(self.n_assets)
        self.portfolio_value = self.initial_capital
        self.weights = np.zeros(self.n_assets)
        self.trades = []

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Get current state"""
        price_window = self.prices.iloc[self.current_step - self.lookback : self.current_step].values

        returns_window = self.returns.iloc[self.current_step - self.lookback : self.current_step].values

        weights = self.weights

        state = np.concatenate([price_window.flatten(), returns_window.flatten(), weights])

        return state

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take action in environment

        Args:
            action: Target portfolio weights (sum to 1)
        """
        # Ensure weights sum to 1
        action = action / (action.sum() + 1e-8)

        # Calculate transaction costs
        turnover = np.sum(np.abs(action - self.weights))
        cost = turnover * self.transaction_cost * self.portfolio_value

        # Execute trades
        self.weights = action

        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1

        if not done:
            # Calculate returns for this step
            step_returns = self.returns.iloc[self.current_step].values
            portfolio_return = np.dot(self.weights, step_returns)

            # Update portfolio value
            self.portfolio_value *= 1 + portfolio_return
            self.portfolio_value -= cost

            # Calculate reward (Sharpe ratio contribution)
            reward = portfolio_return - 0.02 / 252  # Risk-free rate adjustment
        else:
            reward = 0

        return self._get_state(), reward, done, {"portfolio_value": self.portfolio_value, "weights": self.weights, "turnover": turnover, "cost": cost}


class PolicyNetwork(nn.Module):
    """Policy network for DRL agent"""

    def __init__(self, input_dim: int, n_assets: int, hidden_dim: int = 128):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        # Actor: outputs portfolio weights
        self.actor = nn.Linear(hidden_dim, n_assets)

        # Critic: outputs state value
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # Actor output with softmax for weights
        weights = F.softmax(self.actor(x), dim=-1)

        # Critic output
        value = self.critic(x)

        return weights, value


class DRLPortfolioOptimizer:
    """
    Deep Reinforcement Learning portfolio optimizer [citation:5]

    Uses actor-critic architecture with:
    - Proximal Policy Optimization (PPO)
    - Dynamic factor integration [citation:10]
    """

    def __init__(
        self,
        n_assets: int,
        state_dim: int,
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
    ):
        self.training_history = None
        self.n_assets = n_assets
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Networks
        self.policy = PolicyNetwork(state_dim, n_assets)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Experience buffer
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

        self.training_step = 0

    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """Get action from policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            weights, value = self.policy(state_tensor)

        if deterministic:
            action = weights.squeeze().numpy()
            log_prob = 0
        else:
            # Add exploration noise
            noise = np.random.normal(0, 0.05, self.n_assets)
            action = weights.squeeze().numpy() + noise
            action = np.clip(action, 0, 1)
            action = action / action.sum()  # Renormalize

            # Approximate log prob
            log_prob = -0.5 * np.sum(noise**2) / (0.05**2)

        return action, log_prob, value.item()

    def store_transition(self, state: np.ndarray, action: np.ndarray, log_prob: float, reward: float, value: float, done: bool):
        """Store transition in buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def _compute_gae(self, last_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation"""
        values = self.values + [last_value]
        advantages = []
        returns = []
        gae = 0

        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            delta = self.rewards[t] + self.gamma * next_value * (1 - self.dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        return np.array(advantages), np.array(returns)

    def train(self, env: PortfolioEnvironment, episodes: int = 1000, max_steps: int = 1000) -> List[float]:
        """Train the agent"""
        episode_returns = []

        # Store training history for analysis
        self.training_history = {
            "episode_returns": [],
            "policy_losses": [],
            "value_losses": [],
            "entropy_losses": [],
            "gradient_norms": [],
            "update_info": [],  # Store update info for analysis
        }

        # For tracking best model
        best_return = float("-inf")
        patience_counter = 0
        patience = 50  # Early stopping patience

        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            step = 0
            episode_updates = []  # Track updates in this episode

            while step < max_steps:
                # Get action
                action, log_prob, value = self.get_action(state)

                # Take step
                next_state, reward, done, info = env.step(action)

                # Store transition
                self.store_transition(state, action, log_prob, reward, value, done)

                state = next_state
                episode_reward += reward
                step += 1

                if done or step % 128 == 0:  # Update every 128 steps
                    # Get last value
                    _, _, last_value = self.get_action(state)
                    update_info = self.update(last_value)

                    # NOW USING update_info!
                    if update_info:
                        episode_updates.append(update_info)

                        # Store key metrics for monitoring
                        self.training_history["policy_losses"].append(update_info.get("policy_loss", 0))
                        self.training_history["value_losses"].append(update_info.get("value_loss", 0))
                        self.training_history["entropy_losses"].append(update_info.get("entropy", 0))
                        self.training_history["gradient_norms"].append(update_info.get("grad_norm", 0))

                        # Log if significant update
                        if update_info.get("policy_loss", 0) > 1.0:
                            logger.debug(f"Large policy loss at step {step}: {update_info['policy_loss']:.4f}")

                    if done:
                        break

            # Store episode results
            episode_returns.append(episode_reward)
            self.training_history["episode_returns"].append(episode_reward)

            # Calculate episode statistics from updates
            if episode_updates:
                avg_policy_loss = np.mean([u.get("policy_loss", 0) for u in episode_updates])
                avg_value_loss = np.mean([u.get("value_loss", 0) for u in episode_updates])
                avg_entropy = np.mean([u.get("entropy", 0) for u in episode_updates])
            else:
                avg_policy_loss = avg_value_loss = avg_entropy = 0

            # Track best model
            if episode_reward > best_return:
                best_return = episode_reward
                patience_counter = 0
                self.save_checkpoint("best_model.pth")
            else:
                patience_counter += 1

            # Logging
            if episode % 10 == 0:
                logger.info(
                    f"Episode {episode} | "
                    f"Return: {episode_reward:.2f} | "
                    f"Best: {best_return:.2f} | "
                    f"Policy Loss: {avg_policy_loss:.4f} | "
                    f"Value Loss: {avg_value_loss:.4f} | "
                    f"Entropy: {avg_entropy:.4f} | "
                    f"Updates: {len(episode_updates)}"
                )

                # Log update info summary
                if episode_updates:
                    last_update = episode_updates[-1]
                    if "kl_divergence" in last_update:
                        logger.debug(f"KL Divergence: {last_update['kl_divergence']:.4f}")
                    if "explained_variance" in last_update:
                        logger.debug(f"Explained Variance: {last_update['explained_variance']:.4f}")

            # Early stopping check
            if patience_counter >= patience:
                logger.info(f"Early stopping at episode {episode}")
                break

        # Final training summary
        self._log_training_summary(episode_returns)

        return episode_returns

    def update(self, last_value: float) -> Dict[str, float]:
        """
        Update policy and value networks using collected transitions

        Returns:
            Dictionary with update information for monitoring
        """
        if len(self.memory) < self.batch_size:
            return {}

        # Sample batch
        states, actions, old_log_probs, rewards, values, dones = self.memory.sample(self.batch_size)

        # Calculate advantages
        returns, advantages = self._compute_gae(rewards, values, dones, last_value)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Update policy multiple epochs
        policy_losses = []
        value_losses = []
        entropies = []
        kl_divs = []

        for _ in range(self.ppo_epochs):
            # Get current policy outputs
            action_probs, state_values, dist_entropy = self.evaluate_actions(states, actions)

            # Compute ratios
            ratios = torch.exp(action_probs - old_log_probs.detach())

            # Surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(state_values.squeeze(), returns)

            # Entropy bonus
            entropy_loss = -self.entropy_coef * dist_entropy.mean()

            # Total loss
            loss = policy_loss + self.value_coef * value_loss + entropy_loss

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)

            self.optimizer.step()

            # Store losses
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(dist_entropy.mean().item())

            # Calculate KL divergence
            with torch.no_grad():
                new_probs, _, _ = self.evaluate_actions(states, actions)
                kl_div = (old_log_probs - new_probs).mean().item()
                kl_divs.append(kl_div)

        # Clear memory
        self.memory.clear()

        update_info = {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy": np.mean(entropies),
            "kl_divergence": np.mean(kl_divs),
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "explained_variance": self._compute_explained_variance(returns, values),
            "advantages_mean": advantages.mean().item(),
            "advantages_std": advantages.std().item(),
            "clip_fraction": self._compute_clip_fraction(ratios).item(),
            "update_step": self.update_counter,
        }

        self.update_counter += 1

        return update_info

    def _compute_explained_variance(self, returns: torch.Tensor, values: torch.Tensor) -> float:
        """Compute explained variance of value predictions"""
        returns = returns.detach().cpu().numpy()
        values = values.detach().cpu().numpy()

        if returns.std() == 0:
            return 1.0

        return 1 - ((returns - values) ** 2).mean() / returns.var()

    def _compute_clip_fraction(self, ratios: torch.Tensor) -> torch.Tensor:
        """Compute fraction of clipped ratios"""
        clipped = ((ratios < 1 - self.clip_epsilon) | (ratios > 1 + self.clip_epsilon)).float()
        return clipped.mean()

    def _log_training_summary(self, episode_returns: List[float]):
        """Log training summary statistics"""
        if not episode_returns:
            return

        returns_array = np.array(episode_returns)

        logger.info("=" * 50)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total episodes: {len(episode_returns)}")
        logger.info(f"Best return: {np.max(episode_returns):.2f}")
        logger.info(f"Average return: {np.mean(episode_returns):.2f}")
        logger.info(f"Median return: {np.median(episode_returns):.2f}")
        logger.info(f"Std return: {np.std(episode_returns):.2f}")
        logger.info(f"Min return: {np.min(episode_returns):.2f}")

        # Last 100 episodes
        if len(episode_returns) >= 100:
            last_100 = returns_array[-100:]
            logger.info(f"Last 100 avg: {np.mean(last_100):.2f}")
            logger.info(f"Last 100 trend: {np.polyfit(range(100), last_100, 1)[0]:.4f}")

        # Policy statistics
        if self.training_history["policy_losses"]:
            logger.info(f"Avg policy loss: {np.mean(self.training_history['policy_losses'][-100:]):.4f}")
            logger.info(f"Avg value loss: {np.mean(self.training_history['value_losses'][-100:]):.4f}")
            logger.info(f"Avg entropy: {np.mean(self.training_history['entropy_losses'][-100:]):.4f}")

        logger.info("=" * 50)

    def plot_training_progress(self, save_path: Optional[str] = None):
        """Plot training progress using collected history"""
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # Episode returns
            axes[0, 0].plot(self.training_history["episode_returns"])
            axes[0, 0].set_title("Episode Returns")
            axes[0, 0].set_xlabel("Episode")
            axes[0, 0].set_ylabel("Return")

            # Policy loss
            if self.training_history["policy_losses"]:
                axes[0, 1].plot(self.training_history["policy_losses"])
                axes[0, 1].set_title("Policy Loss")
                axes[0, 1].set_xlabel("Update Step")

            # Value loss
            if self.training_history["value_losses"]:
                axes[0, 2].plot(self.training_history["value_losses"])
                axes[0, 2].set_title("Value Loss")
                axes[0, 2].set_xlabel("Update Step")

            # Entropy
            if self.training_history["entropy_losses"]:
                axes[1, 0].plot(self.training_history["entropy_losses"])
                axes[1, 0].set_title("Entropy")
                axes[1, 0].set_xlabel("Update Step")

            # Gradient norms
            if self.training_history["gradient_norms"]:
                axes[1, 1].plot(self.training_history["gradient_norms"])
                axes[1, 1].set_title("Gradient Norm")
                axes[1, 1].set_xlabel("Update Step")

            # Moving average of returns
            if len(self.training_history["episode_returns"]) > 50:
                returns = self.training_history["episode_returns"]
                ma = pd.Series(returns).rolling(50).mean()
                axes[1, 2].plot(returns, alpha=0.5, label="Raw")
                axes[1, 2].plot(ma, "r-", linewidth=2, label="50-ep MA")
                axes[1, 2].set_title("Returns with Moving Average")
                axes[1, 2].legend()

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)
                logger.info(f"Training plot saved to {save_path}")
            else:
                plt.show()

        except ImportError:
            logger.warning("matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Error plotting training progress: {e}")

    def optimize_portfolio(self, current_state: np.ndarray) -> np.ndarray:
        """Get optimal portfolio weights for current state"""
        action, _, _ = self.get_action(current_state, deterministic=True)
        return action

    def save_checkpoint(self, filepath: str):
        """Save model checkpoint with training history"""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_history": self.training_history,
            "update_counter": self.update_counter,
            "hyperparameters": {
                "gamma": self.gamma,
                "tau": self.tau,
                "clip_epsilon": self.clip_epsilon,
                "ppo_epochs": self.ppo_epochs,
                "batch_size": self.batch_size,
                "entropy_coef": self.entropy_coef,
                "value_coef": self.value_coef,
                "max_grad_norm": self.max_grad_norm,
            },
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_history = checkpoint.get("training_history", {})
        self.update_counter = checkpoint.get("update_counter", 0)
        logger.info(f"Checkpoint loaded from {filepath}")


class DynamicFactorDRL(DRLPortfolioOptimizer):
    """
    Dynamic Factor-informed Reinforcement Learning [citation:10]

    Integrates five fundamental factors:
    - Size, Value, Beta, Investment, Quality
    """

    def __init__(self, n_assets: int, state_dim: int, factor_dim: int = 5):
        super().__init__(n_assets, state_dim + factor_dim)

        self.factor_dim = factor_dim
        self.factor_scores = np.ones(factor_dim) / factor_dim

    def get_action(self, state: np.ndarray, factors: Optional[np.ndarray] = None, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """
        Get action with factor information [citation:10]
        """
        if factors is None:
            factors = self.factor_scores

        augmented_state = np.concatenate([state, factors])

        return super().get_action(augmented_state, deterministic)

    def update_factor_scores(self, new_scores: np.ndarray):
        """Update factor importance scores"""
        self.factor_scores = new_scores / new_scores.sum()

    def explain_decision(self, state: np.ndarray, factors: np.ndarray) -> Dict:
        """
        Explain which factors influenced the decision [citation:10]
        """
        # Get action with and without factors
        action_with, _, _ = self.get_action(state, factors, deterministic=True)
        action_without, _, _ = self.get_action(state, np.ones(self.factor_dim) / self.factor_dim, deterministic=True)

        # Calculate factor influence
        influence = {}
        for i in range(self.factor_dim):
            # Perturb factor i
            factors_perturbed = factors.copy()
            factors_perturbed[i] = 1 - factors_perturbed[i]
            factors_perturbed = factors_perturbed / factors_perturbed.sum()

            action_perturbed, _, _ = self.get_action(state, factors_perturbed, deterministic=True)

            # Measure change in allocation
            influence[f"factor_{i}"] = np.mean(np.abs(action_perturbed - action_with))

        return {
            "action_with_factors": action_with.tolist(),
            "action_without_factors": action_without.tolist(),
            "factor_influence": influence,
            "most_influential": max(influence, key=influence.get),
        }
