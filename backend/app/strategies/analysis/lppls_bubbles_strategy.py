from typing import Dict, Union

import numpy as np
import pandas as pd

from ...strategies import BaseStrategy
from ...strategies.analysis.lppls_strategy import LPPLSModel


class LPPLSBubbleStrategy(BaseStrategy):
    """
    LPPLS-based bubble detection strategy with confidence scoring

    This strategy implements the AI-enhanced LPPLS methodology from 2025 research:
    - Detects endogenous bubble formation
    - Provides reliability scores for predictions
    - Calculates Distance-to-Crash with AI (DTCAI) metric
    """

    def __init__(self, name: str = "LPPLS_Bubble", params: Dict = None):
        """
        Initialize LPPLS strategy

        Default params:
            lookback_window: 252  # 1 year of trading days
            min_window: 60  # Minimum window for bubble detection
            max_search_days: 30  # Maximum days ahead to search for tc
            confidence_threshold: 0.6  # Minimum confidence for signal
            crash_prob_threshold: 0.3  # Probability threshold for crash alert
            use_ensemble: True  # Use multiple windows for ensemble prediction
        """
        default_params = {
            "lookback_window": 252,
            "min_window": 60,
            "max_search_days": 30,
            "confidence_threshold": 0.6,
            "crash_prob_threshold": 0.3,
            "use_ensemble": True,
            "ensemble_windows": [126, 252, 378],  # 6mo, 1yr, 1.5yr
            "position_size": 1.0,  # Full position when signal triggered
        }

        if params:
            default_params.update(params)

        super().__init__(name, default_params)
        self.models = []
        self.current_bubble_status = {}
        self.dtcai_history = []  # Distance-to-Crash with AI history

    def generate_signal(self, data: pd.DataFrame) -> Union[int, Dict]:
        """
        Generate trading signal based on LPPLS bubble detection

        Args:
            data: DataFrame with OHLCV data (must have 'Close' column)

        Returns:
            Dict with signal information
        """
        if data.empty or len(data) < self.params["min_window"]:
            return {"signal": 0, "position_size": 0, "metadata": {"error": "Insufficient data", "bubble_detected": False}}

        # Use closing prices
        prices = data["Close"].values

        # Create time array (trading days)
        times = np.arange(len(prices))

        # Detect bubble using primary window
        bubble_status = self._detect_bubble(prices, times, window=self.params["lookback_window"])

        # Ensemble prediction if enabled
        if self.params["use_ensemble"]:
            ensemble_votes = []
            ensemble_confidences = []

            for window in self.params["ensemble_windows"]:
                if len(prices) >= window:
                    status = self._detect_bubble(prices[-window:], times[-window:], window=window)
                    ensemble_votes.append(1 if status["is_bubble"] else 0)
                    ensemble_confidences.append(status["confidence"])

            # Weighted ensemble vote
            if ensemble_votes:
                # Guard against all-zero weights (np.average raises ZeroDivisionError)
                if sum(ensemble_confidences) > 0:
                    weighted_vote = np.average(ensemble_votes, weights=ensemble_confidences)
                else:
                    weighted_vote = np.mean(ensemble_votes)
                ensemble_confidence = np.mean(ensemble_confidences)

                # Override bubble status with ensemble if confidence is higher
                if ensemble_confidence > bubble_status.get("confidence", 0):
                    bubble_status["is_bubble"] = weighted_vote > 0.5
                    bubble_status["confidence"] = ensemble_confidence
                    bubble_status["ensemble_agreement"] = weighted_vote

        # Store current status
        self.current_bubble_status = bubble_status

        # Track DTCAI history
        if "crash_probability" in bubble_status:
            self.dtcai_history.append(
                {"date": data.index[-1], "dtcai": bubble_status.get("confidence", 0), "crash_prob": bubble_status["crash_probability"]}
            )

        # Generate signal
        signal = 0
        metadata = bubble_status.copy()

        # Signal conditions based on bubble detection
        if bubble_status["is_bubble"]:
            if bubble_status["confidence"] >= self.params["confidence_threshold"]:
                if bubble_status.get("crash_probability", 0) >= self.params["crash_prob_threshold"]:
                    # High confidence bubble with significant crash probability -> SELL/SHORT
                    signal = -1
                    metadata["signal_strength"] = bubble_status["crash_probability"]
                    metadata["action"] = "SHORT"
                else:
                    # Bubble detected but low crash probability -> HOLD/REDUCE
                    signal = 0
                    metadata["action"] = "CAUTION"
                    metadata["message"] = "Bubble detected but crash probability low"
            else:
                # Low confidence bubble -> HOLD
                signal = 0
                metadata["action"] = "MONITOR"
                metadata["message"] = "Potential bubble with low confidence"
        else:
            # No bubble detected
            signal = 0
            metadata["action"] = "NO_ACTION"

        # Determine position size based on confidence
        position_size = self.params["position_size"] * bubble_status.get("confidence", 0)

        return {"signal": signal, "position_size": position_size if signal != 0 else 0, "metadata": metadata}

    def _detect_bubble(self, prices: np.ndarray, times: np.ndarray, window: int) -> Dict:
        """
        Detect bubble in price series using LPPLS
        """
        if len(prices) < self.params["min_window"]:
            return {"is_bubble": False, "confidence": 0, "reasons": ["Insufficient data"], "crash_probability": 0}

        # Use only the most recent window
        if len(prices) > window:
            prices = prices[-window:]
            times = times[-window:]

        # Normalize times
        times_norm = (times - times[0]) / (times[-1] - times[0])

        # Fit LPPLS model
        model = LPPLSModel()
        params = model.fit(prices, times_norm, max_search_days=self.params["max_search_days"])

        if not params:
            return {"is_bubble": False, "confidence": 0, "reasons": ["Model fitting failed"], "crash_probability": 0}

        # Detect bubble regime
        bubble_status = model.detect_bubble_regime()

        # Add parameter information
        bubble_status["parameters"] = {
            "m": params.get("m", 0),
            "omega": params.get("omega", 0),
            "tc_days_ahead": params.get("tc_days_ahead", 0),
            "amplitude": params.get("amplitude", 0),
        }

        # Store model for reference
        self.models.append(model)

        return bubble_status

    def generate_signals_vectorized(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate signals for backtesting using rolling window
        """
        signals = pd.Series(index=data.index, data=0)

        for i in range(self.params["min_window"], len(data)):
            window_data = data.iloc[: i + 1]
            signal_dict = self.generate_signal(window_data)
            signals.iloc[i] = signal_dict["signal"]

        return signals

    def get_dtcai_timeseries(self) -> pd.DataFrame:
        """Get historical DTCAI (Distance-to-Crash with AI) values"""
        if not self.dtcai_history:
            return pd.DataFrame()

        df = pd.DataFrame(self.dtcai_history)
        df.set_index("date", inplace=True)
        return df

    def reset(self):
        """Reset strategy state"""
        self.models = []
        self.current_bubble_status = {}
        self.dtcai_history = []
