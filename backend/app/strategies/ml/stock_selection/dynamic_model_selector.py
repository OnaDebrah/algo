import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class DynamicModelSelector:
    """
    Dynamically selects best model for current market environment
    """

    MODELS = {"linear": LinearRegression, "ridge": Ridge, "random_forest": RandomForestRegressor, "gradient_boosting": GradientBoostingRegressor}

    def __init__(self, model_configs: Dict = None):
        self.model_configs = model_configs or {
            "linear": {},
            "ridge": {"alpha": 1.0},
            "random_forest": {"n_estimators": 100, "max_depth": 10},
            "gradient_boosting": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5},
        }
        self.model_performance = {name: [] for name in self.MODELS.keys()}
        self.current_best_model = None
        self.scaler = StandardScaler()

    def select_best_model(self, X: pd.DataFrame, y: pd.Series, cv_splits: int = 5) -> Tuple[str, float]:
        """
        Select best model based on cross-validation performance

        Returns:
            Tuple of (model_name, best_score)
        """
        tscv = TimeSeriesSplit(n_splits=cv_splits)

        best_score = float("inf")
        best_model_name = None

        for model_name, model_class in self.MODELS.items():
            cv_scores = []

            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Scale features
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_val_scaled = self.scaler.transform(X_val)

                # Train model
                model = model_class(**self.model_configs.get(model_name, {}))
                model.fit(X_train_scaled, y_train)

                # Evaluate
                y_pred = model.predict(X_val_scaled)
                mse = mean_squared_error(y_val, y_pred)
                cv_scores.append(mse)

            avg_mse = np.mean(cv_scores)
            self.model_performance[model_name].append(avg_mse)

            if avg_mse < best_score:
                best_score = avg_mse
                best_model_name = model_name

        self.current_best_model = best_model_name
        logger.info(f"Selected model: {best_model_name} with MSE: {best_score:.6f}")

        return best_model_name, best_score

    def get_performance_trend(self, window: int = 5) -> Dict:
        """Get performance trend for each model"""
        trends = {}

        for model_name, perf in self.model_performance.items():
            if len(perf) >= window:
                recent = perf[-window:]
                trend = np.polyfit(range(window), recent, 1)[0]
                trends[model_name] = {"current": perf[-1], "trend": float(trend), "improving": trend < 0}
            else:
                trends[model_name] = {"current": perf[-1] if perf else None, "trend": 0, "improving": None}

        return trends
