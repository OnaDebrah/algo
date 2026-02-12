"""
Machine Learning based trading strategy
"""

import logging

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from backend.app.strategies import BaseStrategy
from config import DEFAULT_ML_TEST_SIZE, DEFAULT_ML_THRESHOLD

logger = logging.getLogger(__name__)


class MLStrategy(BaseStrategy):
    """Machine Learning based trading strategy"""

    def __init__(
        self,
        name: str = "ML Strategy",
        model_type: str = "random_forest",
        n_estimators: int = 100,
        max_depth: int = 10,
        test_size: float = DEFAULT_ML_TEST_SIZE,
        learning_rate: float = 0.1,
    ):
        """
        Initialize ML strategy

        Args:
            name: Strategy name
            model_type: 'random_forest' or 'gradient_boosting'
            n_estimators: Number of trees/boosting stages
            max_depth: Maximum tree depth
            test_size: Fraction of data for testing
            learning_rate: Learning rate (for gradient boosting)
        """
        self.name = name
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.test_size = test_size
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_cols = []

        params = {
            "name": name,
            "model_type": model_type,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "test_size": test_size,
            "learning_rate": learning_rate,
        }
        super().__init__(model_type, params)

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare technical indicators as features"""
        df = data.copy()

        # Price features
        df["returns"] = df["Close"].pct_change()

        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f"sma_{window}"] = df["Close"].rolling(window=window).mean()
            df[f"ema_{window}"] = df["Close"].ewm(span=window, adjust=False).mean()

        # RSI
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        df["bb_middle"] = df["Close"].rolling(window=20).mean()
        bb_std = df["Close"].rolling(window=20).std()
        df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
        df["bb_lower"] = df["bb_middle"] - (bb_std * 2)

        # Volatility
        df["volatility"] = df["returns"].rolling(window=20).std()

        return df

    def create_labels(self, data: pd.DataFrame, threshold: float = DEFAULT_ML_THRESHOLD) -> pd.Series:
        """
        Create labels for training

        Args:
            data: Price data
            threshold: Return threshold for buy/sell signals

        Returns:
            Series with labels: 1 (buy), -1 (sell), 0 (hold)
        """
        future_returns = data["Close"].pct_change(1).shift(-1)

        labels = pd.Series(0, index=data.index)
        labels[future_returns > threshold] = 1
        labels[future_returns < -threshold] = -1

        return labels

    def train(self, data: pd.DataFrame, test_size: float = DEFAULT_ML_TEST_SIZE) -> tuple:
        """
        Train the ML model

        Args:
            data: Historical price data
            test_size: Fraction of data for testing

        Returns:
            Tuple of (train_score, test_score)
        """
        df = self.prepare_features(data)
        labels = self.create_labels(df)

        if test_size is None:
            test_size = self.test_size

        feature_cols = [
            col
            for col in df.columns
            if col
            not in [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Dividends",
                "Stock Splits",
            ]
        ]

        df = df.dropna()
        labels = labels[df.index]

        X = df[feature_cols]
        y = labels

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=42)
        elif self.model_type == "svm":
            self.model = SVC(probability=True, kernel="rbf", random_state=42)
        elif self.model_type == "logistic_regression":
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=self.n_estimators, max_depth=self.max_depth, learning_rate=self.learning_rate, random_state=42
            )

        self.model.fit(X_train_scaled, y_train)

        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)

        self.is_trained = True
        self.feature_cols = feature_cols

        logger.info(f"Model trained - Train: {train_score:.2%}, Test: {test_score:.2%}")

        # Build training history for the frontend chart
        training_history = self._build_training_history(X_train_scaled, y_train, X_test_scaled, y_test)

        return train_score, test_score, training_history

    def _build_training_history(self, X_train, y_train, X_test, y_test):
        """Build per-stage training curves for the frontend chart.

        GradientBoosting: uses staged_predict for per-stage accuracy curves.
        Others (RF, SVM, Logistic): single-point history with final scores.
        """
        training_history = []

        if isinstance(self.model, GradientBoostingClassifier):
            # Collect all staged predictions in one pass each (cheap, uses already-trained trees)
            train_staged = list(self.model.staged_predict(X_train))
            test_staged = list(self.model.staged_predict(X_test))

            for stage in range(len(train_staged)):
                train_acc = accuracy_score(y_train, train_staged[stage])
                test_acc = accuracy_score(y_test, test_staged[stage])
                training_history.append(
                    {
                        "epoch": stage + 1,
                        "loss": round(1.0 - train_acc, 6),
                        "accuracy": round(train_acc, 6),
                        "val_loss": round(1.0 - test_acc, 6),
                        "val_accuracy": round(test_acc, 6),
                    }
                )
        else:
            # Single-shot models (RF, SVM, Logistic) — single data point
            train_acc = self.model.score(X_train, y_train)
            test_acc = self.model.score(X_test, y_test)
            training_history.append(
                {
                    "epoch": 1,
                    "loss": round(1.0 - train_acc, 6),
                    "accuracy": round(train_acc, 6),
                    "val_loss": round(1.0 - test_acc, 6),
                    "val_accuracy": round(test_acc, 6),
                }
            )

        return training_history

    def generate_signal(self, data: pd.DataFrame) -> int:
        """
        Generate trading signal using ML model

        Args:
            data: Current market data

        Returns:
            Signal: 1 (buy), -1 (sell), 0 (hold)
        """
        if not self.is_trained:
            return 0

        df = self.prepare_features(data)
        df = df.dropna()

        if len(df) == 0:
            return 0

        X = df[self.feature_cols].iloc[-1:].values
        X_scaled = self.scaler.transform(X)

        prediction = self.model.predict(X_scaled)[0]

        return int(prediction)

    def generate_signals_vectorized(self, data: pd.DataFrame) -> pd.Series:
        """Vectorized version of ML prediction"""
        if not self.is_trained:
            logger.warning(f"ML Model {self.name} not trained. Vectorized signals will be 0.")
            return pd.Series(0, index=data.index)

        df = self.prepare_features(data)
        df_clean = df.dropna()

        if len(df_clean) == 0:
            return pd.Series(0, index=data.index)

        X = df_clean[self.feature_cols].values
        X_scaled = self.scaler.transform(X)

        predictions = self.model.predict(X_scaled)

        signals = pd.Series(0, index=data.index)
        signals.loc[df_clean.index] = predictions

        return signals

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from trained model"""
        if not self.is_trained:
            return pd.DataFrame()

        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importance = abs(self.model.coef_[0])
        else:
            return pd.DataFrame()

        importance_df = pd.DataFrame(
            {
                "feature": self.feature_cols,
                "importance": importance,
            }
        ).sort_values("importance", ascending=False)

        return importance_df
