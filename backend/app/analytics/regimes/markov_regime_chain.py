import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class MarkovRegimeChain:
    """Markov chain model for regime transitions"""

    def __init__(self, regimes: List[str] = None):
        """
        Initialize Markov chain for regime analysis

        Args:
            regimes: List of possible regime states
        """
        self.regimes = regimes or ["bull", "neutral", "bear"]
        self.n_states = len(self.regimes)
        self.regime_to_idx = {r: i for i, r in enumerate(self.regimes)}
        self.idx_to_regime = {i: r for i, r in enumerate(self.regimes)}

        # Transition matrix (from row to column)
        self.transition_matrix = None
        self.stationary_distribution = None
        self.regime_history = []

    def fit(self, regime_sequence: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Fit Markov chain to regime sequence

        Args:
            regime_sequence: List of regime labels in chronological order

        Returns:
            Transition probability matrix as dictionary
        """
        if len(regime_sequence) < 2:
            logger.warning("Insufficient data for Markov chain")
            return self._empty_transition_matrix()

        # Convert regimes to indices
        indices = []
        for regime in regime_sequence:
            if regime in self.regime_to_idx:
                indices.append(self.regime_to_idx[regime])
            else:
                # Handle unknown regime
                logger.warning(f"Unknown regime: {regime}, skipping")
                continue

        if len(indices) < 2:
            return self._empty_transition_matrix()

        # Count transitions
        transition_counts = np.zeros((self.n_states, self.n_states))

        for i in range(len(indices) - 1):
            from_idx = indices[i]
            to_idx = indices[i + 1]
            transition_counts[from_idx, to_idx] += 1

        # Convert to probabilities
        self.transition_matrix = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            row_sum = transition_counts[i].sum()
            if row_sum > 0:
                self.transition_matrix[i] = transition_counts[i] / row_sum
            else:
                # If no observations, assume equal probability or self-transition
                self.transition_matrix[i, i] = 1.0

        # Calculate stationary distribution
        self._calculate_stationary_distribution()

        # Store history
        self.regime_history = regime_sequence

        # Convert to dictionary format
        return self.matrix_to_dict()

    def _calculate_stationary_distribution(self, max_iter=1000, tol=1e-8):
        """Calculate stationary distribution of Markov chain"""
        if self.transition_matrix is None:
            return

        # Power method for stationary distribution
        n = self.n_states
        pi = np.ones(n) / n

        for _ in range(max_iter):
            pi_new = pi @ self.transition_matrix
            if np.max(np.abs(pi_new - pi)) < tol:
                break
            pi = pi_new

        self.stationary_distribution = pi

    def predict_next_regime(self, current_regime: str) -> Dict[str, float]:
        """Predict probabilities for next regime"""
        if self.transition_matrix is None or current_regime not in self.regime_to_idx:
            return {r: 1 / self.n_states for r in self.regimes}

        current_idx = self.regime_to_idx[current_regime]
        probs = self.transition_matrix[current_idx]

        return {self.idx_to_regime[i]: float(probs[i]) for i in range(self.n_states)}

    def get_expected_duration(self, regime: str) -> float:
        """Get expected duration of regime in days"""
        if self.transition_matrix is None or regime not in self.regime_to_idx:
            return 5  # Default

        idx = self.regime_to_idx[regime]
        # Expected duration = 1 / (1 - self-transition probability)
        self_transition = self.transition_matrix[idx, idx]
        if self_transition >= 1.0:
            return float("inf")

        return 1.0 / (1.0 - self_transition)

    def get_transition_probabilities(self) -> Dict[str, Dict[str, float]]:
        """Get transition matrix as nested dictionary"""
        return self.matrix_to_dict()

    def matrix_to_dict(self) -> Dict[str, Dict[str, float]]:
        """Convert numpy transition matrix to dictionary format"""
        if self.transition_matrix is None:
            return self._empty_transition_matrix()

        result = {}
        for i, from_regime in enumerate(self.regimes):
            result[from_regime] = {}
            for j, to_regime in enumerate(self.regimes):
                result[from_regime][to_regime] = float(self.transition_matrix[i, j])

        return result

    def _empty_transition_matrix(self) -> Dict[str, Dict[str, float]]:
        """Create empty transition matrix with equal probabilities"""
        result = {}
        equal_prob = 1.0 / self.n_states

        for from_regime in self.regimes:
            result[from_regime] = {}
            for to_regime in self.regimes:
                result[from_regime][to_regime] = equal_prob

        return result

    def get_regime_persistence(self) -> Dict[str, float]:
        """Get persistence (self-transition probability) for each regime"""
        if self.transition_matrix is None:
            return {r: 0.5 for r in self.regimes}

        return {self.idx_to_regime[i]: float(self.transition_matrix[i, i]) for i in range(self.n_states)}
