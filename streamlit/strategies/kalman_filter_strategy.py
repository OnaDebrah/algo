from typing import Dict, Tuple

import numpy as np
import pandas as pd
from pykalman import KalmanFilter

from streamlit.strategies import BaseStrategy


class KalmanFilterStrategy(BaseStrategy):
    """
    Pairs trading with Kalman Filter for dynamic hedge ratio
    """

    def __init__(
        self,
        asset_1: str,
        asset_2: str,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_loss_z: float = 3.0,
        min_obs: int = 20,
        transitory_std: float = 0.01,  # How much beta can change daily
        observation_std: float = 0.1,  # Observation noise
        decay_factor: float = 0.99,  # Forget old observations
    ):
        params = {
            "asset_1": asset_1,
            "asset_2": asset_2,
            "entry_z": entry_z,
            "exit_z": exit_z,
            "stop_loss_z": stop_loss_z,
            "min_obs": min_obs,
            "transitory_std": transitory_std,
            "observation_std": observation_std,
            "decay_factor": decay_factor,
        }
        super().__init__("Kalman Pairs Trading", params)

        # Kalman Filter state
        self.kf = None
        self.state_means = None
        self.state_covs = None

        # Trading state
        self.in_position = False
        self.position_direction = 0  # 1: long spread, -1: short spread
        self.entry_spread = 0

    def _initialize_kalman(self, initial_prices: pd.DataFrame):
        """
        Initialize Kalman Filter with initial data
        """
        asset_1 = self.params["asset_1"]
        asset_2 = self.params["asset_2"]

        # Initial hedge ratio from OLS
        y = np.log(initial_prices[asset_1].values)
        x = np.log(initial_prices[asset_2].values)
        initial_beta = np.polyfit(x, y, 1)[0]

        # State: [hedge_ratio, intercept]
        # Observation: price_1 = hedge_ratio * price_2 + intercept

        # Transition matrix: identity (state persists)
        transition_matrix = np.eye(2)

        # Observation matrix: [price_2, 1]
        # We'll update this with each observation

        # Process noise (how much state can change)
        transition_covariance = np.eye(2) * self.params["transitory_std"] ** 2

        # Observation noise
        observation_covariance = np.array([[self.params["observation_std"] ** 2]])

        # Initial state
        initial_state_mean = np.array([initial_beta, 0.0])
        initial_state_covariance = np.eye(2) * 0.1

        self.kf = KalmanFilter(
            transition_matrices=transition_matrix,
            observation_matrices=None,  # Will be dynamic
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
        )

        # Apply forgetting factor (discount old observations)
        self.kf.transition_covariance /= self.params["decay_factor"]

    def _update_kalman(self, price_1: float, price_2: float) -> Tuple[float, float]:
        """
        Update Kalman Filter with new prices
        Returns: (hedge_ratio, intercept)
        """
        # Observation matrix for this timestep
        observation_matrix = np.array([[np.log(price_2), 1.0]])

        # Observation (log price of asset 1)
        observation = np.array([[np.log(price_1)]])

        if self.state_means is None:
            # First update
            self.state_means, self.state_covs = self.kf.filter_update(
                filtered_state_mean=self.kf.initial_state_mean,
                filtered_state_covariance=self.kf.initial_state_covariance,
                observation=observation,
                observation_matrix=observation_matrix,
            )
        else:
            # Subsequent updates
            self.state_means, self.state_covs = self.kf.filter_update(
                filtered_state_mean=self.state_means,
                filtered_state_covariance=self.state_covs,
                observation=observation,
                observation_matrix=observation_matrix,
            )

        hedge_ratio = self.state_means[0, 0]
        intercept = self.state_means[0, 1]

        return hedge_ratio, intercept

    def _calculate_spread_history(self, prices_1: pd.Series, prices_2: pd.Series) -> pd.Series:
        """
        Calculate spread using Kalman-filtered hedge ratio
        """
        spreads = []
        hedge_ratios = []

        # Recalculate full history with Kalman updates
        for i in range(len(prices_1)):
            if i < self.params["min_obs"]:
                # Use initial OLS estimate
                if i == self.params["min_obs"] - 1:
                    # Initialize Kalman
                    self._initialize_kalman(
                        pd.DataFrame(
                            {
                                self.params["asset_1"]: prices_1[: self.params["min_obs"]],
                                self.params["asset_2"]: prices_2[: self.params["min_obs"]],
                            }
                        )
                    )
                hedge_ratio, _ = self._initial_hedge_ratio(prices_1[: i + 1], prices_2[: i + 1])
            else:
                # Update Kalman
                hedge_ratio, _ = self._update_kalman(prices_1.iloc[i], prices_2.iloc[i])

            hedge_ratios.append(hedge_ratio)
            spread = np.log(prices_1.iloc[i]) - hedge_ratio * np.log(prices_2.iloc[i])
            spreads.append(spread)

        return pd.Series(spreads, index=prices_1.index), pd.Series(hedge_ratios, index=prices_1.index)

    def _initial_hedge_ratio(self, prices_1: pd.Series, prices_2: pd.Series) -> float:
        """Initial OLS estimate before Kalman has enough data"""
        y = np.log(prices_1.values)
        x = np.log(prices_2.values)
        return np.polyfit(x, y, 1)[0], 0.0

    def _calculate_zscore(self, spread: pd.Series) -> pd.Series:
        """
        Calculate rolling z-score with exponential weighting
        """
        # Exponential moving statistics (more responsive)
        ema = spread.ewm(span=self.params["min_obs"] * 2).mean()
        emstd = spread.ewm(span=self.params["min_obs"] * 2).std()

        return (spread - ema) / emstd

    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """
        Generate trading signal using Kalman Filter
        """
        asset_1 = self.params["asset_1"]
        asset_2 = self.params["asset_2"]

        if len(data) < self.params["min_obs"]:
            return {
                "signal": 0,
                "position_size": 0,
                "metadata": {"reason": "insufficient_data"},
            }

        prices_1 = data[asset_1]
        prices_2 = data[asset_2]

        # Calculate spread with Kalman-filtered hedge ratio
        spread_series, hedge_ratios = self._calculate_spread_history(prices_1, prices_2)
        current_spread = spread_series.iloc[-1]
        current_hedge_ratio = hedge_ratios.iloc[-1]

        # Calculate z-score
        zscore_series = self._calculate_zscore(spread_series)
        current_z = zscore_series.iloc[-1]

        # Get Kalman uncertainty
        if self.state_covs is not None:
            hedge_ratio_variance = self.state_covs[0, 0, -1]
            confidence = 1 / (1 + np.sqrt(hedge_ratio_variance))
        else:
            confidence = 0.5

        # Signal generation with confidence weighting
        signal_info = self._generate_signal_logic(current_z, current_spread, confidence)

        # Add metadata
        signal_info["metadata"].update(
            {
                "hedge_ratio": float(current_hedge_ratio),
                "hedge_ratio_confidence": float(confidence),
                "spread": float(current_spread),
                "z_score": float(current_z),
                "kalman_initialized": self.kf is not None,
            }
        )

        return signal_info

    def _generate_signal_logic(self, current_z: float, current_spread: float, confidence: float) -> Dict:
        """
        Core signal logic with confidence weighting
        """
        entry_z = self.params["entry_z"]
        exit_z = self.params["exit_z"]
        stop_loss_z = self.params["stop_loss_z"]

        signal = 0
        position_size = 0

        if not self.in_position:
            # Entry logic weighted by confidence
            if current_z > entry_z and confidence > 0.7:
                # Spread too wide: short asset_1, long asset_2
                signal = -1
                position_size = min(1.0, confidence)
                self.in_position = True
                self.position_direction = -1
                self.entry_spread = current_spread

            elif current_z < -entry_z and confidence > 0.7:
                # Spread too narrow: long asset_1, short asset_2
                signal = 1
                position_size = min(1.0, confidence)
                self.in_position = True
                self.position_direction = 1
                self.entry_spread = current_spread

        else:
            # Exit logic
            exit_signal = False

            # Stop loss
            if abs(current_z) > stop_loss_z:
                exit_signal = True

            # Mean reversion exit
            elif abs(current_z) < exit_z:
                exit_signal = True

            # Confidence dropped too low
            elif confidence < 0.5:
                exit_signal = True

            if exit_signal:
                signal = -self.position_direction  # Close position
                position_size = 1.0
                self.in_position = False
                self.position_direction = 0

        return {
            "signal": signal,
            "position_size": position_size,
            "metadata": {
                "in_position": self.in_position,
                "confidence": float(confidence),
            },
        }

    def reset(self):
        """Reset strategy state"""
        self.kf = None
        self.state_means = None
        self.state_covs = None
        self.in_position = False
        self.position_direction = 0
        self.entry_spread = 0
