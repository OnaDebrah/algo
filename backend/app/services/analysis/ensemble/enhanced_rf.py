"""
Enhanced Random Forest with features from LPPLS and LSTM models
"""

import logging
import warnings
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from core.data.providers.providers import ProviderFactory
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from strategies.analysis.lppls_bubbles_strategy import LPPLSBubbleStrategy
from strategies.analysis.lstm_stress_strategy import LSTMStressStrategy

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class EnhancedRandomForest:
    """
    Random Forest enhanced with LPPLS and LSTM features
    """

    def __init__(self, crash_threshold: float = 0.20, lead_time: int = 60):
        self.crash_threshold = crash_threshold
        self.lead_time = lead_time
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.lppls_strategy = LPPLSBubbleStrategy()
        self.lstm_strategy = LSTMStressStrategy()
        self.provider_factory = ProviderFactory()

    def generate_enhanced_features(self, data: pd.DataFrame, symbol: str, include_ml_features: bool = True) -> pd.DataFrame:
        """
        Generate enhanced feature set including LPPLS and LSTM outputs
        """
        features = pd.DataFrame(index=data.index)

        # 1. Basic technical features (from original)
        returns = data["Close"].pct_change()
        features["returns_1d"] = returns
        features["returns_5d"] = returns.rolling(5).sum()
        features["returns_20d"] = returns.rolling(20).sum()

        # Volatility features
        features["volatility_5d"] = returns.rolling(5).std()
        features["volatility_20d"] = returns.rolling(20).std()
        features["volatility_60d"] = returns.rolling(60).std()
        features["vol_expansion"] = features["volatility_20d"] / features["volatility_60d"]

        # Moving averages
        features["ma_20"] = data["Close"].rolling(20).mean()
        features["ma_50"] = data["Close"].rolling(50).mean()
        features["ma_200"] = data["Close"].rolling(200).mean()
        features["price_vs_ma20"] = data["Close"] / features["ma_20"] - 1
        features["price_vs_ma50"] = data["Close"] / features["ma_50"] - 1

        # RSI
        delta = data["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features["rsi"] = 100 - (100 / (1 + rs))

        # 2. LPPLS features (if enabled)
        if include_ml_features:
            lppls_features = self._generate_lppls_features(data, symbol)
            for col, values in lppls_features.items():
                features[col] = values

            # 3. LSTM features
            lstm_features = self._generate_lstm_features(data, symbol)
            for col, values in lstm_features.items():
                features[col] = values

            # 4. Interaction features
            features["bubble_stress_interaction"] = features.get("lppls_crash_prob", 0) * features.get("lstm_stress_index", 0.5)

            features["ml_consensus"] = 0.6 * features.get("lppls_crash_prob", 0) + 0.4 * features.get("lstm_stress_index", 0.5)

        return features

    def _generate_lppls_features(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Generate LPPLS-derived features for each time point"""
        features = {
            "lppls_bubble": [],
            "lppls_confidence": [],
            "lppls_crash_prob": [],
            "lppls_days_to_critical": [],
            "lppls_m_param": [],
            "lppls_omega": [],
        }

        # Need at least 60 days for LPPLS
        min_window = 60

        for i in range(len(data)):
            if i < min_window:
                # Not enough data
                features["lppls_bubble"].append(0)
                features["lppls_confidence"].append(0)
                features["lppls_crash_prob"].append(0)
                features["lppls_days_to_critical"].append(0)
                features["lppls_m_param"].append(0.5)
                features["lppls_omega"].append(6.0)
                continue

            # Get window of data
            window_data = data.iloc[: i + 1].copy()

            # Generate signal
            signal = self.lppls_strategy.generate_signal(window_data)
            metadata = signal["metadata"]

            features["lppls_bubble"].append(1 if metadata.get("is_bubble", False) else 0)
            features["lppls_confidence"].append(metadata.get("confidence", 0))
            features["lppls_crash_prob"].append(metadata.get("crash_probability", 0))

            params = metadata.get("parameters", {})
            features["lppls_days_to_critical"].append(params.get("tc_days_ahead", 0))
            features["lppls_m_param"].append(params.get("m", 0.5))
            features["lppls_omega"].append(params.get("omega", 6.0))

        return features

    def _generate_lstm_features(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Generate LSTM-derived features for each time point"""
        features = {"lstm_stress_index": [], "lstm_confidence": [], "lstm_trend": [], "lstm_tap_deviation": []}

        min_window = 60

        for i in range(len(data)):
            if i < min_window:
                features["lstm_stress_index"].append(0.5)
                features["lstm_confidence"].append(0)
                features["lstm_trend"].append(0)
                features["lstm_tap_deviation"].append(0)
                continue

            window_data = data.iloc[: i + 1].copy()
            signal = self.lstm_strategy.generate_signal(window_data)
            metadata = signal["metadata"]

            features["lstm_stress_index"].append(metadata.get("stress_index", 0.5))
            features["lstm_confidence"].append(metadata.get("confidence", 0))

            trend = metadata.get("stress_trend", "stable")
            features["lstm_trend"].append(1 if trend == "increasing" else (-1 if trend == "decreasing" else 0))
            features["lstm_tap_deviation"].append(metadata.get("tap_deviation", 0))

        return features

    def prepare_training_data(self, data: pd.DataFrame, symbol: str, lookahead: int = 60) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data with enhanced features and crash labels
        """
        # Generate enhanced features
        features = self.generate_enhanced_features(data, symbol)

        # Calculate forward drawdown for crash labeling
        rolling_max = data["Close"].rolling(window=lookahead, min_periods=1).max().shift(-lookahead)
        forward_return = (data["Close"].shift(-lookahead) - rolling_max) / rolling_max

        # Crash if forward return <= -threshold
        crash_labels = (forward_return <= -self.crash_threshold).astype(int)

        # Align features and labels
        combined = pd.concat([features, crash_labels.rename("crash_label")], axis=1)
        combined = combined.dropna()

        return combined.drop("crash_label", axis=1), combined["crash_label"]

    def train(self, X: pd.DataFrame, y: pd.Series, cv_splits: int = 5, optimize_threshold: bool = True):
        """
        Train enhanced Random Forest with calibrated probabilities
        """
        # Store feature names
        self.feature_names = X.columns.tolist()

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_splits)

        # Base Random Forest
        base_rf = RandomForestClassifier(
            n_estimators=300, max_depth=15, min_samples_split=20, min_samples_leaf=10, class_weight="balanced", random_state=42, n_jobs=-1
        )

        # Calibrated classifier
        self.model = CalibratedClassifierCV(base_rf, method="sigmoid", cv=tscv)

        # Train
        self.model.fit(X, y)

        if optimize_threshold:
            self.optimal_threshold = self._find_optimal_threshold(X, y)

        # Feature importance
        self.feature_importance = self._get_feature_importance()

        logger.info(f"Model trained with {len(X)} samples, {len(self.feature_names)} features")

        return self.model

    def _find_optimal_threshold(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Find optimal probability threshold for crash detection"""
        from sklearn.metrics import f1_score

        proba = self.model.predict_proba(X)[:, 1]

        thresholds = np.arange(0.1, 0.6, 0.02)
        f1_scores = []

        for threshold in thresholds:
            pred = (proba >= threshold).astype(int)
            f1_scores.append(f1_score(y, pred, zero_division=0))

        optimal_idx = np.argmax(f1_scores)
        return thresholds[optimal_idx]

    def _get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from base estimator"""
        base_estimator = self.model.calibrated_classifiers_[0].base_estimator

        importance = pd.DataFrame({"feature": self.feature_names, "importance": base_estimator.feature_importances_}).sort_values(
            "importance", ascending=False
        )

        return importance

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get calibrated crash probabilities"""
        if self.model is None:
            raise ValueError("Model not trained")

        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """Get binary predictions using optimal threshold"""
        if threshold is None:
            threshold = getattr(self, "optimal_threshold", 0.33)

        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def save(self, path: str):
        """Save model and metadata"""
        joblib.dump(
            {
                "model": self.model,
                "feature_names": self.feature_names,
                "optimal_threshold": self.optimal_threshold,
                "feature_importance": self.feature_importance,
                "crash_threshold": self.crash_threshold,
                "lead_time": self.lead_time,
            },
            path,
        )
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model and metadata"""
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.optimal_threshold = data["optimal_threshold"]
        self.feature_importance = data["feature_importance"]
        self.crash_threshold = data["crash_threshold"]
        self.lead_time = data["lead_time"]
        logger.info(f"Model loaded from {path}")
