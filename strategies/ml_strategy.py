"""
Machine Learning based trading strategy
"""

import logging

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import DEFAULT_ML_TEST_SIZE, DEFAULT_ML_THRESHOLD

logger = logging.getLogger(__name__)


class MLStrategy:
    """Machine Learning based trading strategy"""

    def __init__(self, name: str, model_type: str = "random_forest"):
        """
        Initialize ML strategy

        Args:
            name: Strategy name
            model_type: 'random_forest' or 'gradient_boosting'
        """
        self.name = name
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_cols = []

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

    def create_labels(
        self, data: pd.DataFrame, threshold: float = DEFAULT_ML_THRESHOLD
    ) -> pd.Series:
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

    def train(
        self, data: pd.DataFrame, test_size: float = DEFAULT_ML_TEST_SIZE
    ) -> tuple:
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

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=100, max_depth=5, random_state=42
            )

        self.model.fit(X_train_scaled, y_train)

        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)

        self.is_trained = True
        self.feature_cols = feature_cols

        logger.info(f"Model trained - Train: {train_score:.2%}, Test: {test_score:.2%}")

        return train_score, test_score

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

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from trained model"""
        if not self.is_trained or not hasattr(self.model, "feature_importances_"):
            return pd.DataFrame()

        importance_df = pd.DataFrame(
            {
                "feature": self.feature_cols,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        return importance_df
