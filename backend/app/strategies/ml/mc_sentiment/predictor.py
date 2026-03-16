import logging
import warnings
from typing import Tuple

import nltk
import numpy as np
import pandas as pd

# ML imports
try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

warnings.filterwarnings("ignore")

nltk.download("vader_lexicon", quiet=True)

logger = logging.getLogger(__name__)


class MLPredictor:
    """
    Machine learning model for price prediction using sentiment + technical features
    """

    def __init__(self, model_type: str = "gradient_boosting"):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []

        if not ML_AVAILABLE:
            raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")

    def create_features(self, data: pd.DataFrame, sentiment_features: pd.DataFrame) -> pd.DataFrame:
        """
        Create feature set combining technical indicators and sentiment
        """
        features = pd.DataFrame(index=data.index)

        # Technical features
        features["returns"] = data["close"].pct_change()
        features["returns_5d"] = data["close"].pct_change(5)
        features["returns_20d"] = data["close"].pct_change(20)

        # Volatility features
        features["volatility_20d"] = data["close"].pct_change().rolling(20).std()
        features["volatility_60d"] = data["close"].pct_change().rolling(60).std()

        # Volume features
        if "volume" in data.columns:
            features["volume_ma_ratio"] = data["volume"] / data["volume"].rolling(20).mean()

        # Moving averages
        features["sma_20"] = data["close"].rolling(20).mean() / data["close"] - 1
        features["sma_50"] = data["close"].rolling(50).mean() / data["close"] - 1

        # RSI
        delta = data["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features["rsi"] = 100 - (100 / (1 + rs))

        # Sentiment features
        for col in sentiment_features.columns:
            features[col] = sentiment_features[col]

        # Lag features
        for lag in [1, 2, 3, 5]:
            features[f"returns_lag_{lag}"] = features["returns"].shift(lag)

        return features

    def train(self, features: pd.DataFrame, targets: pd.Series, test_size: float = 0.2):
        """
        Train the ML model
        """
        # Remove NaN values
        valid_idx = features.dropna().index.intersection(targets.dropna().index)
        X = features.loc[valid_idx]
        y = targets.loc[valid_idx]

        if len(X) < 100:
            raise ValueError(f"Insufficient training data: {len(X)} samples")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        if self.model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        elif self.model_type == "random_forest":
            self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.model.fit(X_train_scaled, y_train)
        self.feature_names = X.columns.tolist()
        self.is_trained = True

        # Calculate metrics
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)

        return {"train_r2": train_score, "test_r2": test_score, "train_samples": len(X_train), "test_samples": len(X_test)}

    def predict(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict returns and uncertainty

        Returns:
            predictions: Expected returns
            uncertainty: Standard deviation of predictions (ensemble std or constant)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        X = features[self.feature_names]
        nan_count = X.isna().sum().sum()
        if nan_count > 0:
            logger.debug(f"Imputing {nan_count} NaN values in prediction features")
            X = X.fillna(0)
        X_scaled = self.scaler.transform(X)

        predictions = self.model.predict(X_scaled)

        # Estimate uncertainty (simplified)
        if hasattr(self.model, "estimators_"):
            # For ensemble methods, use prediction variance
            # GradientBoosting: estimators_ is 2D array (n_estimators, 1) of DecisionTreeRegressors
            # RandomForest: estimators_ is a flat list of DecisionTreeRegressors
            estimators = self.model.estimators_
            if hasattr(estimators, "ravel"):
                # GradientBoosting: flatten 2D array to get individual trees
                estimators = estimators.ravel()
            all_predictions = np.array([tree.predict(X_scaled) for tree in estimators])
            uncertainty = np.std(all_predictions, axis=0)
        else:
            # Fixed uncertainty for non-ensemble methods
            uncertainty = np.ones_like(predictions) * 0.02

        return predictions, uncertainty
