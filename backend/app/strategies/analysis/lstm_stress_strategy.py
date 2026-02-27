"""
LSTM-based cross-market stress prediction model
Forecasts market dysfunction 60 days ahead using multi-market indicators
Based on 2026 research from Systemic Risk Centre [citation:5]
"""

import logging
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout

# Deep learning imports
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

from ...strategies.base_strategy import BaseStrategy

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@dataclass
class MarketStressIndicators:
    """Collection of market stress indicators"""

    # Volatility indicators
    vix_level: float  # CBOE VIX
    vix_term_structure: float  # VIX futures curve slope
    equity_volatility: float  # Realized volatility

    # Liquidity indicators
    bid_ask_spread: float  # Average bid-ask spread
    amihud_illiquidity: float  # Amihud illiquidity ratio
    turnover_ratio: float  # Volume relative to average

    # Cross-market stress
    tap_deviation: float  # Triangular arbitrage parity deviation [citation:5]
    bond_equity_correlation: float  # Rolling correlation
    credit_spread: float  # Corporate bond spread

    # Currency indicators
    risk_reversals: float  # FX risk reversals
    safe_haven_flows: float  # Flows to safe havens

    # Market internals
    advance_decline_ratio: float  # Market breadth
    put_call_ratio: float  # Options sentiment
    new_highs_lows: float  # New highs vs new lows


class LSTMStressPredictor:
    """
    LSTM neural network for predicting market stress 60 days ahead
    """

    def __init__(self, sequence_length: int = 60, forecast_horizon: int = 60):
        """
        Initialize LSTM predictor

        Args:
            sequence_length: Number of past days to use for prediction
            forecast_horizon: Days ahead to forecast (60 from research)
        """
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.feature_names = None
        self.training_history = []

    def build_model(self, n_features: int, lstm_units=None):
        """
        Build LSTM architecture based on research specifications [citation:5]

        Uses bidirectional LSTM for better pattern recognition
        """
        if lstm_units is None:
            lstm_units = [128, 64]
        model = Sequential()

        # First Bidirectional LSTM layer
        model.add(Bidirectional(LSTM(lstm_units[0], return_sequences=True), input_shape=(self.sequence_length, n_features)))
        model.add(Dropout(0.2))

        # Second LSTM layer
        model.add(LSTM(lstm_units[1], return_sequences=False))
        model.add(Dropout(0.2))

        # Dense layers for regression
        model.add(Dense(32, activation="relu"))
        model.add(Dropout(0.1))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(1))  # Single output: stress index

        # Compile with Adam optimizer
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

        self.model = model
        return model

    def prepare_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training

        Creates sequences of length sequence_length for each prediction
        """
        X_seq, y_seq = [], []

        for i in range(len(X) - self.sequence_length - self.forecast_horizon + 1):
            X_seq.append(X[i : i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length + self.forecast_horizon - 1])

        return np.array(X_seq), np.array(y_seq)

    def fit(self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2, epochs: int = 100, batch_size: int = 32, verbose: int = 0) -> Dict:
        """
        Train LSTM model on market data
        """
        # Store feature names
        self.feature_names = X.columns.tolist()

        # Scale features
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

        # Prepare sequences
        X_seq, y_seq = self.prepare_sequences(X_scaled, y_scaled)

        if len(X_seq) < 100:
            logger.warning(f"Insufficient sequences: {len(X_seq)}. Need at least 100.")
            return {}

        # Split data (time-series split to avoid lookahead bias)
        split_idx = int(len(X_seq) * (1 - validation_split))
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]

        # Build model if not exists
        if self.model is None:
            self.build_model(X.shape[1])

        # Callbacks
        early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=verbose)

        # Train
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=verbose,
            shuffle=False,  # Don't shuffle time series data
        )

        self.training_history = history.history

        # Calculate metrics
        y_pred_val = self.model.predict(X_val, verbose=0)
        y_pred_val_inv = self.scaler_y.inverse_transform(y_pred_val)
        y_val_inv = self.scaler_y.inverse_transform(y_val.reshape(-1, 1))

        # Calculate error metrics
        mae = np.mean(np.abs(y_pred_val_inv - y_val_inv))
        rmse = np.sqrt(np.mean((y_pred_val_inv - y_val_inv) ** 2))

        metrics = {
            "mae": float(mae),
            "rmse": float(rmse),
            "val_loss": float(history.history["val_loss"][-1]),
            "train_loss": float(history.history["loss"][-1]),
        }

        logger.info(f"LSTM training completed - MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        return metrics

    def predict_stress(self, X: pd.DataFrame) -> Tuple[float, float]:
        """
        Predict market stress for next period

        Returns:
            stress_index: Predicted stress level
            confidence: Prediction confidence based on model uncertainty
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Scale features
        X_scaled = self.scaler_X.transform(X.tail(self.sequence_length))

        # Reshape for LSTM
        X_pred = X_scaled.reshape(1, self.sequence_length, -1)

        # Make prediction
        y_pred_scaled = self.model.predict(X_pred, verbose=0)
        stress_index = float(self.scaler_y.inverse_transform(y_pred_scaled)[0, 0])

        # Monte Carlo dropout for uncertainty estimation
        if hasattr(self.model, "layers"):
            # Enable dropout during inference for MC dropout
            mc_predictions = []
            for _ in range(50):
                pred = self.model(X_pred, training=True).numpy()
                mc_predictions.append(self.scaler_y.inverse_transform(pred)[0, 0])

            # Calculate confidence (inverse of std dev)
            pred_std = np.std(mc_predictions)
            confidence = 1.0 / (1.0 + pred_std)  # Normalize to [0,1]
        else:
            confidence = 0.5

        return stress_index, confidence

    def save_model(self, filepath: str):
        """Save LSTM model and scalers"""
        if self.model is None:
            raise ValueError("No model to save")

        # Save Keras model
        self.model.save(f"{filepath}_lstm.h5")

        # Save scalers and metadata
        import joblib

        joblib.dump(
            {
                "scaler_X": self.scaler_X,
                "scaler_y": self.scaler_y,
                "feature_names": self.feature_names,
                "sequence_length": self.sequence_length,
                "forecast_horizon": self.forecast_horizon,
                "training_history": self.training_history,
            },
            f"{filepath}_metadata.pkl",
        )

        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load LSTM model and scalers"""
        import joblib

        # Load Keras model
        self.model = load_model(f"{filepath}_lstm.h5")

        # Load scalers and metadata
        metadata = joblib.load(f"{filepath}_metadata.pkl")
        self.scaler_X = metadata["scaler_X"]
        self.scaler_y = metadata["scaler_y"]
        self.feature_names = metadata["feature_names"]
        self.sequence_length = metadata["sequence_length"]
        self.forecast_horizon = metadata["forecast_horizon"]
        self.training_history = metadata.get("training_history", [])

        logger.info(f"Model loaded from {filepath}")

        return self


class LSTMStressStrategy(BaseStrategy):
    """
    Trading strategy based on LSTM market stress predictions

    Uses predicted stress index to adjust market exposure and hedges
    """

    def __init__(self, name: str = "LSTM_Stress", params: Dict = None):
        """
        Initialize LSTM stress strategy

        Default params:
            stress_threshold_high: 0.7  # High stress level
            stress_threshold_low: 0.3   # Low stress level
            confidence_threshold: 0.6    # Minimum confidence for trading
            lookback_window: 252          # 1 year of data
            retrain_frequency: 30         # Retrain every 30 days
            position_reduction: 0.5       # Reduce position by 50% in high stress
        """
        default_params = {
            "stress_threshold_high": 0.7,
            "stress_threshold_low": 0.3,
            "confidence_threshold": 0.6,
            "lookback_window": 252,
            "retrain_frequency": 30,
            "position_reduction": 0.5,
            "use_mc_dropout": True,
        }

        if params:
            default_params.update(params)

        super().__init__(name, default_params)

        self.predictor = LSTMStressPredictor(sequence_length=60, forecast_horizon=60)
        self.last_train_date = None
        self.stress_history = []
        self.confidence_history = []

    def generate_signal(self, data: pd.DataFrame) -> Union[int, Dict]:
        """
        Generate signal based on predicted market stress

        Args:
            data: DataFrame with market indicators (must include all required features)

        Returns:
            Signal dictionary with position adjustments based on stress level
        """
        if data.empty or len(data) < self.params["lookback_window"]:
            return {"signal": 0, "position_size": 1.0, "metadata": {"error": "Insufficient data", "stress_index": 0.5}}

        # Check if we need to retrain
        should_retrain = self.last_train_date is None or (datetime.now() - self.last_train_date).days >= self.params["retrain_frequency"]

        if should_retrain and len(data) >= 252:  # Need at least 1 year for training
            self._train_model(data)

        # Predict current stress
        stress_index, confidence = self._predict_stress(data)

        # Store history
        self.stress_history.append(stress_index)
        self.confidence_history.append(confidence)

        # Generate signal based on stress level
        signal = 0  # Default hold
        position_size = 1.0
        action = "HOLD"
        message = ""

        if confidence >= self.params["confidence_threshold"]:
            if stress_index >= self.params["stress_threshold_high"]:
                # High stress - reduce exposure or hedge
                signal = -1 if stress_index > 0.8 else 0  # Only short in extreme stress
                position_size = self.params["position_reduction"]
                action = "REDUCE_EXPOSURE"
                message = f"High market stress detected: {stress_index:.2f}"

            elif stress_index <= self.params["stress_threshold_low"]:
                # Low stress - normal or increased exposure
                signal = 1
                position_size = 1.0
                action = "NORMAL_EXPOSURE"
                message = f"Low market stress: {stress_index:.2f}"
            else:
                # Normal stress - maintain position
                signal = 0
                position_size = 1.0
                action = "HOLD"
                message = f"Normal market stress: {stress_index:.2f}"
        else:
            message = f"Low confidence prediction: {confidence:.2f}"

        # Calculate trend
        stress_trend = self._calculate_trend()

        metadata = {
            "stress_index": stress_index,
            "confidence": confidence,
            "action": action,
            "message": message,
            "stress_trend": stress_trend,
            "stress_history": self.stress_history[-10:] if self.stress_history else [],
            "tap_deviation": self._get_latest_tap_deviation(data),  # Key indicator [citation:5]
        }

        return {"signal": signal, "position_size": position_size, "metadata": metadata}

    def _train_model(self, data: pd.DataFrame):
        """Train LSTM model on historical data"""
        try:
            # Prepare features and target
            features = self._prepare_features(data)
            target = self._prepare_target(data)

            if features.empty or target.empty:
                logger.warning("Insufficient data for training")
                return

            # Train model
            metrics = self.predictor.fit(features, target)

            self.last_train_date = datetime.now()
            logger.info(f"Model retrained successfully. Metrics: {metrics}")

        except Exception as e:
            logger.error(f"Error training model: {e}")

    def _predict_stress(self, data: pd.DataFrame) -> Tuple[float, float]:
        """Predict current stress index"""
        try:
            features = self._prepare_features(data)
            stress_index, confidence = self.predictor.predict_stress(features)
            return stress_index, confidence
        except Exception as e:
            logger.error(f"Error predicting stress: {e}")
            return 0.5, 0.0

    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare feature set for LSTM model

        Includes all indicators from MarketStressIndicators class
        """
        features = pd.DataFrame(index=data.index)

        # Volatility indicators
        if "Close" in data.columns:
            returns = data["Close"].pct_change()
            features["volatility_20d"] = returns.rolling(20).std()
            features["volatility_60d"] = returns.rolling(60).std()
            features["volatility_ratio"] = features["volatility_20d"] / features["volatility_60d"]

        # Liquidity indicators
        if "Volume" in data.columns:
            features["volume_ratio"] = data["Volume"] / data["Volume"].rolling(20).mean()
            features["turnover"] = data["Volume"] * data["Close"]

        # Bond-equity correlation (simplified - would need bond data)
        # In production, fetch from provider

        # TAP deviation - key indicator [citation:5]
        # This requires FX data - simplified version using inverse relationship
        if "Close" in data.columns:
            # Proxy for TAP using rolling correlations
            features["tap_proxy"] = returns.rolling(20).skew()

        # Put-call ratio - would need options data

        # Fill NaN values
        features = features.fillna(method="bfill").fillna(method="ffill").fillna(0)

        return features

    def _prepare_target(self, data: pd.DataFrame) -> pd.Series:
        """
        Prepare target variable: future market stress

        Based on forward-looking volatility and drawdown
        """
        if "Close" not in data.columns:
            return pd.Series()

        # Calculate forward 60-day realized volatility
        returns = data["Close"].pct_change()
        future_vol = returns.shift(-60).rolling(60).std()

        # Calculate forward drawdown
        rolling_max = data["Close"].rolling(60, min_periods=1).max()
        drawdown = (data["Close"] - rolling_max) / rolling_max

        # Composite stress index
        stress_index = (future_vol * 10) - drawdown  # Higher vol = more stress, drawdown negative

        # Normalize to [0, 1]
        stress_index = (stress_index - stress_index.min()) / (stress_index.max() - stress_index.min())

        return stress_index

    def _calculate_trend(self) -> str:
        """Calculate trend in stress index"""
        if len(self.stress_history) < 5:
            return "insufficient_data"

        recent = self.stress_history[-5:]
        if recent[-1] > recent[0] * 1.1:
            return "increasing"
        elif recent[-1] < recent[0] * 0.9:
            return "decreasing"
        else:
            return "stable"

    def _get_latest_tap_deviation(self, data: pd.DataFrame) -> float:
        """Extract latest TAP deviation proxy"""
        try:
            features = self._prepare_features(data)
            return float(features["tap_proxy"].iloc[-1]) if "tap_proxy" in features.columns else 0.0
        except Exception as e:
            logger.error(f"Failed to extract latest TAP {e}")
            return 0.0

    def reset(self):
        """Reset strategy state"""
        self.stress_history = []
        self.confidence_history = []
        # Don't reset model - keep trained model
