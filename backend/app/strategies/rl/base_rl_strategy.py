"""
Base class for all RL-based trading strategies.
Handles model persistence, train/inference modes, and BaseStrategy interface compliance.
"""

import logging
import os
from abc import abstractmethod
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

# Default directory for saving RL model checkpoints
RL_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "ml_models", "rl")


class BaseRLStrategy(BaseStrategy):
    """
    Abstract base class for RL trading strategies.

    Subclasses must implement:
        - _build_policy()       -> nn.Module
        - _build_state(data)    -> np.ndarray
        - _action_to_signal(action) -> int | Dict
        - train(data)           -> Dict  (training metrics)
    """

    def __init__(self, name: str, params: Dict):
        super().__init__(name=name, params=params)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy: Optional[nn.Module] = None
        self.is_trained = False
        self.training_history: Dict = {}

        # RL hyperparameters (overridable by subclass)
        self.learning_rate = params.get("learning_rate", 3e-4)
        self.gamma = params.get("gamma", 0.99)
        self.gae_lambda = params.get("gae_lambda", 0.95)
        self.clip_epsilon = params.get("clip_epsilon", 0.2)
        self.entropy_coef = params.get("entropy_coef", 0.01)
        self.value_coef = params.get("value_coef", 0.5)
        self.max_grad_norm = params.get("max_grad_norm", 0.5)
        self.episodes = params.get("episodes", 500)
        self.lookback = params.get("lookback", 20)

        # Internal state
        self._current_position = 0.0

    @abstractmethod
    def _build_policy(self, state_dim: int) -> nn.Module:
        """Build the policy network. Called lazily on first use."""
        ...

    @abstractmethod
    def _build_state(self, data: pd.DataFrame) -> np.ndarray:
        """Convert market data to RL state vector for the latest timestep."""
        ...

    @abstractmethod
    def _action_to_signal(self, action: np.ndarray) -> Union[int, Dict]:
        """Convert raw policy output to a trading signal (int or Dict)."""
        ...

    @abstractmethod
    def train(self, data: pd.DataFrame) -> Dict:
        """Train the RL agent on historical data. Returns training metrics."""
        ...

    # ────────────────────────────────────────────────────────────────
    # BaseStrategy interface
    # ────────────────────────────────────────────────────────────────

    def generate_signal(self, data: pd.DataFrame) -> Union[int, Dict]:
        """Generate a single trading signal using the trained policy."""
        if self.policy is None or not self.is_trained:
            return 0  # Hold when untrained

        state = self._build_state(data)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self._get_deterministic_action(state_tensor)

        return self._action_to_signal(action)

    def generate_signals_vectorized(self, data: pd.DataFrame) -> pd.Series:
        """Batch inference over full history."""
        signals = pd.Series(0, index=data.index)

        if self.policy is None or not self.is_trained:
            return signals

        # Slide through data generating signals
        for i in range(self.lookback, len(data)):
            window = data.iloc[: i + 1]
            try:
                state = self._build_state(window)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = self._get_deterministic_action(state_tensor)
                signal = self._action_to_signal(action)
                if isinstance(signal, dict):
                    signals.iloc[i] = signal.get("signal", 0)
                else:
                    signals.iloc[i] = int(signal)
            except Exception:
                continue

        return signals

    def reset(self):
        """Reset episode state."""
        self._current_position = 0.0

    # ────────────────────────────────────────────────────────────────
    # Policy helpers
    # ────────────────────────────────────────────────────────────────

    def _get_deterministic_action(self, state_tensor: torch.Tensor) -> np.ndarray:
        """Get deterministic action from policy (inference mode)."""
        self.policy.eval()
        output = self.policy(state_tensor)

        # Handle actor-critic (tuple) or single output
        if isinstance(output, tuple):
            action_output = output[0]
        else:
            action_output = output

        return action_output.squeeze().cpu().numpy()

    # ────────────────────────────────────────────────────────────────
    # Model persistence
    # ────────────────────────────────────────────────────────────────

    def save_model(self, path: Optional[str] = None) -> str:
        """Save the trained policy to disk."""
        if self.policy is None:
            raise ValueError("No policy to save — train first")

        if path is None:
            os.makedirs(RL_MODELS_DIR, exist_ok=True)
            path = os.path.join(RL_MODELS_DIR, f"{self.name}.pth")

        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "params": self.params,
            "is_trained": self.is_trained,
            "training_history": self.training_history,
        }, path)

        logger.info(f"RL model saved to {path}")
        return path

    def load_model(self, path: Optional[str] = None) -> bool:
        """Load a trained policy from disk."""
        if path is None:
            path = os.path.join(RL_MODELS_DIR, f"{self.name}.pth")

        if not os.path.exists(path):
            logger.warning(f"No model found at {path}")
            return False

        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

            # Rebuild policy if needed
            if self.policy is None:
                # Caller must initialize policy with correct state_dim before loading
                logger.warning("Policy not initialized — call _build_policy() first")
                return False

            self.policy.load_state_dict(checkpoint["policy_state_dict"])
            self.is_trained = checkpoint.get("is_trained", True)
            self.training_history = checkpoint.get("training_history", {})
            self.policy.to(self.device)
            self.policy.eval()

            logger.info(f"RL model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
