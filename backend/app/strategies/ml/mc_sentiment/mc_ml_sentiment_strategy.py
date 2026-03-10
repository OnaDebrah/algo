"""
Monte Carlo ML Sentiment Strategy Implementation
File: strategies/mc_ml_sentiment_strategy.py

A complete implementation combining sentiment analysis, machine learning,
and Monte Carlo simulation for probabilistic price forecasting.
"""

import logging
import warnings
from datetime import datetime
from typing import Any, Dict, List

import nltk
import numpy as np
import pandas as pd

from ....core.trade.sizing.position_sizer import PositionSizer
from ....strategies import BaseStrategy
from ....strategies.ml.mc_sentiment.analyzer import SentimentAnalyzer
from ....strategies.ml.mc_sentiment.mc_engine import MonteCarloEngine
from ....strategies.ml.mc_sentiment.predictor import MLPredictor

warnings.filterwarnings("ignore")

nltk.download("vader_lexicon", quiet=True)

logger = logging.getLogger(__name__)


class MonteCarloMLSentimentStrategy(BaseStrategy):
    """
    Monte Carlo ML Sentiment Strategy

    Combines sentiment analysis, ML predictions, and Monte Carlo simulation
    for probabilistic price forecasting and risk-aware position sizing.
    """

    def __init__(
        self,
        sentiment_sources: List[str] = None,
        ml_model_type: str = "gradient_boosting",
        lookback_period: int = 252,
        forecast_horizon: int = 20,
        num_simulations: int = 10000,
        backtest_num_simulations: int = 500,
        confidence_level: float = 0.95,
        retraining_frequency: str = "weekly",
        sentiment_weight: float = 0.3,
        min_sentiment_quality: float = 0.7,
        **kwargs,
    ):
        params = {
            "ml_model_type": ml_model_type,
            "lookback_period": lookback_period,
            "forecast_horizon": forecast_horizon,
            "num_simulations": num_simulations,
            "backtest_num_simulations": backtest_num_simulations,
            "confidence_level": confidence_level,
            "retraining_frequency": retraining_frequency,
            "sentiment_weight": sentiment_weight,
            "min_sentiment_quality": min_sentiment_quality,
        }
        super().__init__("Monte Carlo ML Sentiment Strategy", params)

        self.sentiment_sources = sentiment_sources or ["news", "twitter", "stocktwits", "options"]
        self.ml_model_type = ml_model_type
        self.lookback_period = lookback_period
        self.forecast_horizon = forecast_horizon
        self.num_simulations = num_simulations
        self.backtest_num_simulations = backtest_num_simulations
        self.confidence_level = confidence_level
        self.retraining_frequency = retraining_frequency
        self.sentiment_weight = sentiment_weight
        self.min_sentiment_quality = min_sentiment_quality

        # Initialize components (backtest_mode=True for deterministic historical sentiment)
        self.sentiment_analyzer = SentimentAnalyzer(self.sentiment_sources, backtest_mode=True)
        self.ml_predictor = MLPredictor(self.ml_model_type)
        self.mc_engine = MonteCarloEngine(self.num_simulations)
        self.position_sizer = PositionSizer(risk_per_trade=0.02)

        # State tracking
        self.last_training_date = None
        self.simulation_results = {}
        self.sentiment_history = pd.DataFrame()
        self.symbol = None  # Set by engine/service before backtesting

    def _normalize_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to lowercase for consistent access."""
        df = data.copy()
        col_map = {}
        for col in df.columns:
            lower = col.lower()
            if lower != col and lower not in df.columns:
                col_map[col] = lower
        if col_map:
            df = df.rename(columns=col_map)
        return df

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators and gather sentiment data
        """
        df = self._normalize_columns(data)

        # Collect sentiment features
        sentiment_data = []
        for date in df.index:
            # Get sentiment for current symbol (assuming single symbol for now)
            sentiment_features = self.sentiment_analyzer.get_sentiment_features(self.symbol or "SPY", date)
            sentiment_features["date"] = date
            sentiment_data.append(sentiment_features)

        sentiment_df = pd.DataFrame(sentiment_data).set_index("date")
        self.sentiment_history = sentiment_df

        # Create ML features
        features = self.ml_predictor.create_features(df, sentiment_df)

        # Store features in dataframe
        for col in features.columns:
            df[col] = features[col]

        return df

    def should_retrain(self, current_date: datetime) -> bool:
        """
        Determine if model should be retrained
        """
        if self.last_training_date is None:
            return True

        days_since_training = (current_date - self.last_training_date).days

        if self.retraining_frequency == "daily":
            return days_since_training >= 1
        elif self.retraining_frequency == "weekly":
            return days_since_training >= 7
        elif self.retraining_frequency == "monthly":
            return days_since_training >= 30

        return False

    def train_model(self, data: pd.DataFrame):
        """
        Train or retrain the ML model
        """
        # Prepare features
        df = self.calculate_indicators(data)

        # Get feature columns (exclude price data and output columns)
        feature_cols = [
            col
            for col in df.columns
            if col not in ["open", "high", "low", "close", "volume", "signal", "position_size", "expected_return", "confidence"]
        ]
        features = df[feature_cols]

        targets = df["close"].pct_change().shift(-1)

        metrics = self.ml_predictor.train(features, targets)
        self.last_training_date = df.index[-1]

        return metrics

    def generate_signal(self, data: pd.DataFrame) -> int:
        """
        Generate a single trading signal (required by BaseStrategy).

        Delegates to generate_signals() and returns the last signal value.
        """
        try:
            result = self.generate_signals(data)
            signal_val = result["signal"].iloc[-1]
            return int(signal_val) if not pd.isna(signal_val) else 0
        except Exception as e:
            logger.error(f"MC ML Sentiment generate_signal error: {e}")
            return 0

    def generate_signals_vectorized(self, data: pd.DataFrame) -> pd.Series:
        """
        Vectorized signal generation for backtest engine compatibility.
        Runs full signal generation and returns the signal column as pd.Series.
        """
        try:
            result = self.generate_signals(data)
            return result["signal"].fillna(0).astype(int)
        except Exception as e:
            logger.error(f"MC ML Sentiment vectorized signal error: {e}")
            return pd.Series(0, index=data.index)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals across the full time series for backtesting.

        Iterates bar-by-bar after the lookback period:
        1. Pre-computes sentiment and technical features in one pass
        2. Retrains the ML model on schedule (weekly by default)
        3. Runs Monte Carlo simulation per bar (reduced sims for performance)
        4. Sets signal based on prob_profit and expected_return thresholds
        """
        df = self._normalize_columns(data)
        df["signal"] = 0
        df["position_size"] = 0.0
        df["expected_return"] = 0.0
        df["confidence"] = 0.0

        # Need sufficient data for rolling-window warmup (~65 bars) + training (100 valid samples)
        min_required = max(200, self.lookback_period)
        if len(df) < min_required:
            logger.warning(f"Insufficient data: {len(df)} bars, need {min_required}. Returning zeros.")
            return df

        # --- Step 1: Pre-compute sentiment for all dates in one pass ---
        symbol = getattr(self, "symbol", None) or "SPY"
        sentiment_data = []
        for date in df.index:
            sentiment_features = self.sentiment_analyzer.get_sentiment_features(symbol, date)
            sentiment_features["date"] = date
            sentiment_data.append(sentiment_features)

        sentiment_df = pd.DataFrame(sentiment_data).set_index("date")
        self.sentiment_history = sentiment_df

        # --- Step 2: Pre-compute all technical + sentiment features once ---
        all_features = self.ml_predictor.create_features(df, sentiment_df)
        feature_cols = [
            col
            for col in all_features.columns
            if col
            not in [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "signal",
                "position_size",
                "expected_return",
                "confidence",
            ]
        ]

        # Target: next-day return (for training)
        targets = df["close"].pct_change().shift(-1)

        # --- Step 3: Use reduced MC simulations for backtest performance ---
        original_sims = self.mc_engine.num_simulations
        self.mc_engine.num_simulations = self.backtest_num_simulations

        # --- Step 4: Iterate bar-by-bar from lookback_period onward ---
        start_idx = self.lookback_period
        last_stats = {}

        for i in range(start_idx, len(df)):
            current_date = df.index[i]

            # Retrain ML model on schedule (weekly by default)
            if not self.ml_predictor.is_trained or self.should_retrain(current_date):
                train_features = all_features[feature_cols].iloc[:i]
                train_targets = targets.iloc[:i]

                try:
                    self.ml_predictor.train(train_features, train_targets)
                    self.last_training_date = current_date
                except Exception as e:
                    logger.warning(f"Training failed at bar {i}: {e}")
                    if not self.ml_predictor.is_trained:
                        continue  # No model exists yet, can't predict
                    # else: fall through to prediction with existing model

            # Predict return + uncertainty for current bar
            current_features = all_features[feature_cols].iloc[i : i + 1]
            try:
                predictions, uncertainty = self.ml_predictor.predict(current_features)
                predicted_return = predictions[0]
                predicted_volatility = uncertainty[0]
            except Exception as e:
                logger.warning(f"Prediction failed at bar {i}: {e}")
                continue

            # Annualize daily predictions for GBM simulation (dt=1/252 expects annualized inputs)
            annual_return = predicted_return * 252
            annual_volatility = predicted_volatility * np.sqrt(252)

            current_price = df["close"].iloc[i]
            paths = self.mc_engine.simulate_paths(
                current_price=current_price,
                predicted_return=annual_return,
                predicted_volatility=annual_volatility,
                forecast_horizon=self.forecast_horizon,
            )

            stats = self.mc_engine.calculate_statistics(paths, self.confidence_level)
            last_stats = stats

            # Generate signal based on probability and expected return
            if stats["prob_profit"] > 0.55 and stats["expected_return"] > 0.01:
                df.iloc[i, df.columns.get_loc("signal")] = 1
            elif stats["prob_profit"] < 0.45 and stats["expected_return"] < -0.01:
                df.iloc[i, df.columns.get_loc("signal")] = -1

            df.iloc[i, df.columns.get_loc("expected_return")] = stats["expected_return"]
            df.iloc[i, df.columns.get_loc("confidence")] = stats["prob_profit"]

        # Restore original MC simulation count (for live use)
        self.mc_engine.num_simulations = original_sims
        self.simulation_results = last_stats

        # Diagnostic summary
        buy_count = int((df["signal"] == 1).sum())
        sell_count = int((df["signal"] == -1).sum())
        total_bars = len(df) - start_idx
        logger.info(
            f"MC ML Sentiment signals: {buy_count} buy, {sell_count} sell, {total_bars - buy_count - sell_count} neutral out of {total_bars} bars"
        )

        return df

    def get_simulation_summary(self) -> Dict[str, Any]:
        """
        Get summary of latest Monte Carlo simulation results
        """
        return self.simulation_results

    def get_sentiment_summary(self) -> pd.DataFrame:
        """
        Get summary of recent sentiment data
        """
        if self.sentiment_history.empty:
            return pd.DataFrame()

        return self.sentiment_history.tail(20)
