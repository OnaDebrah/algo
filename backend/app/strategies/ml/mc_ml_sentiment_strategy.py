"""
Monte Carlo ML Sentiment Strategy Implementation
File: strategies/mc_ml_sentiment_strategy.py

A complete implementation combining sentiment analysis, machine learning,
and Monte Carlo simulation for probabilistic price forecasting.
"""

import logging
import warnings
from datetime import datetime
from typing import Any, Dict, List, Tuple

import nltk
import numpy as np
import pandas as pd
import praw
import requests
import tweepy
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from ...strategies import BaseStrategy

# ML imports
try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    ML_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    ML_AVAILABLE = False

warnings.filterwarnings("ignore")

nltk.download("vader_lexicon", quiet=True)

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Extracts and processes sentiment from various sources
    """

    def __init__(self, sources: List[str] = None):
        self.sources = sources or ["news", "twitter", "options"]
        self.sentiment_cache = {}

    def get_sentiment_features(self, symbol: str, date: datetime) -> Dict[str, float]:
        """
        Extract sentiment features for a given symbol and date

        Returns:
            Dictionary with sentiment scores [-1, 1] for each source
        """
        features = {}

        for source in self.sources:
            if source == "news":
                features["news_sentiment"] = self._get_news_sentiment(symbol, date)
            elif source == "twitter":
                features["twitter_sentiment"] = self._get_twitter_sentiment(symbol, date)
            elif source == "stocktwits":
                features["stocktwits_sentiment"] = self._get_stocktwits_sentiment(symbol, date)
            elif source == "reddit":
                features["reddit_sentiment"] = self._get_reddit_sentiment(symbol, date)
            elif source == "options":
                features["options_sentiment"] = self._get_options_sentiment(symbol, date)

        # Add aggregate sentiment
        if features:
            features["sentiment_aggregate"] = np.mean(list(features.values()))
            features["sentiment_volatility"] = np.std(list(features.values()))

        return features

    def _get_news_sentiment(self, symbol: str, date: datetime) -> float:
        """
        Get news sentiment score
        In production: Use APIs like RavenPack, Bloomberg, or NewsAPI
        """
        # Placeholder: Random walk with mean reversion for demo
        np.random.seed(hash(f"{symbol}{date}") % (2**32))
        return np.clip(np.random.normal(0, 0.3), -1, 1)

    def _get_stocktwits_sentiment(self, symbol: str, date: datetime) -> float:
        """
        Fetches Stocktwits messages and applies VADER sentiment analysis
        weighted by user 'bullish/bearish' labels and user reputation.
        """
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
        analyzer = SentimentIntensityAnalyzer()

        weighted_scores = []
        total_weight = 0

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            messages = data.get("messages", [])
            if not messages:
                return 0.0

            for msg in messages:
                text = msg.get("body", "")
                user = msg.get("user", {})

                # 1. Base VADER Score
                vader_score = analyzer.polarity_scores(text)["compound"]

                # 2. Extract Explicit Sentiment (Bullish/Bearish tag)
                # Users can manually tag their posts. This is a very strong signal.
                explicit_sentiment = msg.get("entities", {}).get("sentiment", {}).get("basic")
                if explicit_sentiment == "Bullish":
                    vader_score = (vader_score + 1.0) / 2  # Nudge score higher
                elif explicit_sentiment == "Bearish":
                    vader_score = (vader_score - 1.0) / 2  # Nudge score lower

                # 3. Calculate Weight (Institutional Grade)
                # Use followers + 'ideas' (total posts) to gauge authority
                followers = user.get("followers", 0)
                experience = user.get("ideas", 0)
                is_official = 10 if user.get("official", False) else 1

                # Logarithmic weighting to avoid one whale skewing everything
                weight = (np.log1p(followers) * 0.7) + (np.log1p(experience) * 0.3) + is_official

                weighted_scores.append(vader_score * weight)
                total_weight += weight

        except Exception as e:
            # In production, log this to your monitoring service
            logger.error(f"Stocktwits API Error for {symbol}: {e}")
            return 0.0

        # 4. Final Weighted Average
        if total_weight == 0:
            return 0.0

        return float(np.sum(weighted_scores) / total_weight)

    def _get_twitter_sentiment(self, symbol: str, date: datetime) -> float:
        """
        Institutional grade sentiment analysis using VADER and Weighted Influence.
        """
        BEARER_TOKEN = "YOUR_TWITTER_BEARER_TOKEN"
        query = f"({symbol} OR #{symbol}) -is:retweet lang:en"

        analyzer = SentimentIntensityAnalyzer()
        weighted_sentiments = []
        total_weight = 0

        try:
            client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True)

            # We request user fields to calculate influence/weight
            tweets = client.search_recent_tweets(
                query=query,
                max_results=100,
                tweet_fields=["text", "public_metrics"],
                expansions="author_id",
                user_fields=["public_metrics", "verified"],
            )

            if not tweets.data:
                return 0.0

            # Create a user lookup map for weights
            users = {u["id"]: u for u in tweets.includes["users"]}

            for tweet in tweets.data:
                author = users.get(tweet.author_id)

                # 1. Calculate Sentiment using VADER
                # 'compound' score ranges from -1 (Extremely Negative) to 1 (Extremely Positive)
                score = analyzer.polarity_scores(tweet.text)["compound"]

                # 2. Calculate Influence Weight
                # Institutional logic: Verified accounts and high follower counts have more "alpha"
                follower_count = author["public_metrics"]["followers_count"] if author else 0
                is_verified = author["verified"] if author else False

                # Weight formula: Logarithmic scale for followers + bonus for verification
                weight = np.log1p(follower_count) + (10 if is_verified else 0)

                weighted_sentiments.append(score * weight)
                total_weight += weight

        except tweepy.TooManyRequests:
            # Production fallback: Log the rate limit and return a neutral/cached value
            return self._get_cached_sentiment(symbol)
        except Exception as e:
            logger.error(f"Sentiment Pipeline Error: {e}")
            return 0.0

        # 3. Final Aggregation
        if total_weight == 0:
            return 0.0

        return float(np.sum(weighted_sentiments) / total_weight)

    def _get_reddit_sentiment(self, symbol: str, date: datetime) -> float:
        """
        Scrapes high-signal subreddits using PRAW and calculates
        a weighted sentiment score based on post engagement.
        """
        # 1. Authentication (Use environment variables for these)
        reddit = praw.Reddit(client_id="YOUR_CLIENT_ID", client_secret="YOUR_CLIENT_SECRET", user_agent="FinancialAnalystBot/1.0 by /u/YourUsername")

        analyzer = SentimentIntensityAnalyzer()
        subreddits = ["wallstreetbets", "stocks", "investing", "options"]

        weighted_scores = []
        total_weight = 0

        # Create a search query for the ticker
        query = f"{symbol}"

        try:
            # Search across the combined subreddits
            target_subs = reddit.subreddit("+".join(subreddits))

            # Fetch top/relevant posts from the last 24-48 hours
            # 'cloud' search is more effective for symbols
            for submission in target_subs.search(query, sort="relevance", time_filter="week", limit=25):
                # 2. VADER Analysis on Title + Body
                content = f"{submission.title} {submission.selftext}"
                vader_score = analyzer.polarity_scores(content)["compound"]

                # 3. Calculate Weight based on "Social Proof"
                # Logic: More upvotes + higher upvote ratio = more market-moving sentiment
                score = submission.score
                ratio = submission.upvote_ratio  # Ranges from 0 to 1
                num_comments = submission.num_comments

                # Institutional Weighting Formula
                # We use log1p for score to handle viral posts without over-skewing
                weight = (np.log1p(max(0, score)) * ratio) + (np.log1p(num_comments) * 0.5)

                weighted_scores.append(vader_score * weight)
                total_weight += weight

                # Optional: Sample the top 3 comments for even deeper sentiment
                submission.comment_sort = "top"
                submission.comments.replace_more(limit=0)  # Flatten comment tree
                for comment in submission.comments[:3]:
                    c_score = analyzer.polarity_scores(comment.body)["compound"]
                    c_weight = np.log1p(max(0, comment.score)) * 0.5  # Comments have less weight than posts
                    weighted_scores.append(c_score * c_weight)
                    total_weight += c_weight

        except Exception as e:
            logger.error(f"Reddit API Error for {symbol}: {e}")
            return 0.0

        # 4. Final Aggregation
        if total_weight == 0:
            return 0.0

        return float(np.sum(weighted_scores) / total_weight)

    def _get_options_sentiment(self, symbol: str, date: datetime) -> float:
        """
        Get options market sentiment (put/call ratio, IV skew)
        In production: Calculate from actual options data
        """
        np.random.seed(hash(f"options{symbol}{date}") % (2**32))
        # Put/call ratio: >1 = bearish, <1 = bullish
        put_call_ratio = np.random.uniform(0.5, 1.5)
        sentiment = (1.0 - put_call_ratio) / 0.5  # Normalize to [-1, 1]
        return np.clip(sentiment, -1, 1)


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

        X = features[self.feature_names].dropna()
        X_scaled = self.scaler.transform(X)

        predictions = self.model.predict(X_scaled)

        # Estimate uncertainty (simplified)
        if hasattr(self.model, "estimators_"):
            # For ensemble methods, use prediction variance
            all_predictions = np.array([tree.predict(X_scaled) for tree in self.model.estimators_])
            uncertainty = np.std(all_predictions, axis=0)
        else:
            # Fixed uncertainty for non-ensemble methods
            uncertainty = np.ones_like(predictions) * 0.02

        return predictions, uncertainty


class MonteCarloEngine:
    """
    Generates Monte Carlo simulations of future price paths
    """

    def __init__(self, num_simulations: int = 10000):
        self.num_simulations = num_simulations

    def simulate_paths(
        self, current_price: float, predicted_return: float, predicted_volatility: float, forecast_horizon: int, dt: float = 1 / 252
    ) -> np.ndarray:
        """
        Generate Monte Carlo price paths using Geometric Brownian Motion

        Args:
            current_price: Starting price
            predicted_return: Expected daily return from ML model
            predicted_volatility: Expected volatility from ML model
            forecast_horizon: Number of days to simulate
            dt: Time step (1/252 for daily)

        Returns:
            Array of shape (num_simulations, forecast_horizon) with price paths
        """
        # Initialize paths
        paths = np.zeros((self.num_simulations, forecast_horizon))
        paths[:, 0] = current_price

        # Generate random shocks
        random_shocks = np.random.normal(0, 1, (self.num_simulations, forecast_horizon - 1))

        # Simulate GBM paths
        for t in range(1, forecast_horizon):
            drift = (predicted_return - 0.5 * predicted_volatility**2) * dt
            diffusion = predicted_volatility * np.sqrt(dt) * random_shocks[:, t - 1]
            paths[:, t] = paths[:, t - 1] * np.exp(drift + diffusion)

        return paths

    def calculate_statistics(self, paths: np.ndarray, confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Calculate statistics from simulated paths
        """
        final_prices = paths[:, -1]

        # Confidence intervals
        lower_percentile = (1 - confidence_level) / 2
        upper_percentile = 1 - lower_percentile

        stats = {
            "mean_price": np.mean(final_prices),
            "median_price": np.median(final_prices),
            "std_price": np.std(final_prices),
            "lower_bound": np.percentile(final_prices, lower_percentile * 100),
            "upper_bound": np.percentile(final_prices, upper_percentile * 100),
            "prob_profit": np.mean(final_prices > paths[0, 0]),
            "expected_return": np.mean((final_prices - paths[0, 0]) / paths[0, 0]),
            "var_95": np.percentile(final_prices - paths[0, 0], 5),
            "cvar_95": np.mean((final_prices - paths[0, 0])[final_prices - paths[0, 0] <= np.percentile(final_prices - paths[0, 0], 5)]),
        }

        return stats


class PositionSizer:
    """
    Determines position sizes based on Monte Carlo simulation results
    """

    def __init__(self, risk_per_trade: float = 0.02):
        self.risk_per_trade = risk_per_trade

    def calculate_position_size(self, portfolio_value: float, simulation_stats: Dict[str, Any], current_price: float) -> float:
        """
        Calculate position size based on simulation results

        Uses Kelly Criterion adjusted for risk constraints
        """
        # Probability of profit from simulations
        win_prob = simulation_stats["prob_profit"]

        # Expected return
        expected_return = simulation_stats["expected_return"]

        # Risk (distance to lower bound)
        risk = abs((simulation_stats["lower_bound"] - current_price) / current_price)

        if risk == 0 or expected_return <= 0:
            return 0.0

        # Kelly Criterion: f = (p*b - q) / b
        # where p = win probability, q = loss probability, b = win/loss ratio
        win_loss_ratio = abs(expected_return / risk) if risk > 0 else 1
        kelly_fraction = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio

        # Apply safety factor (half Kelly)
        kelly_fraction = max(0, kelly_fraction * 0.5)

        # Cap at risk per trade limit
        position_fraction = min(kelly_fraction, self.risk_per_trade)

        # Calculate position size
        position_value = portfolio_value * position_fraction
        position_size = position_value / current_price

        return position_size


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
            "confidence_level": confidence_level,
            "retraining_frequency": retraining_frequency,
            "sentiment_weight": sentiment_weight,
            "min_sentiment_quality": min_sentiment_quality,
        }
        super().__init__("Monte Carlo ML Sentiment Strategy", params)

        self.sentiment_sources = sentiment_sources or ["news", "twitter", "options"]
        self.ml_model_type = ml_model_type
        self.lookback_period = lookback_period
        self.forecast_horizon = forecast_horizon
        self.num_simulations = num_simulations
        self.confidence_level = confidence_level
        self.retraining_frequency = retraining_frequency
        self.sentiment_weight = sentiment_weight
        self.min_sentiment_quality = min_sentiment_quality

        # Initialize components
        self.sentiment_analyzer = SentimentAnalyzer(self.sentiment_sources)
        self.ml_predictor = MLPredictor(self.ml_model_type)
        self.mc_engine = MonteCarloEngine(self.num_simulations)
        self.position_sizer = PositionSizer(risk_per_trade=0.02)

        # State tracking
        self.last_training_date = None
        self.simulation_results = {}
        self.sentiment_history = pd.DataFrame()

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
            sentiment_features = self.sentiment_analyzer.get_sentiment_features("SYMBOL", date)
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

        # Get feature columns (exclude price data)
        feature_cols = [col for col in df.columns if col not in ["open", "high", "low", "close", "volume"]]
        features = df[feature_cols]

        # Target: next day return
        targets = df["close"].pct_change().shift(-1)

        # Train model
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
        Generate trading signals using Monte Carlo simulation
        """
        df = self._normalize_columns(data)
        df["signal"] = 0
        df["position_size"] = 0.0
        df["expected_return"] = 0.0
        df["confidence"] = 0.0

        # Need sufficient data for training
        if len(df) < self.lookback_period:
            return df

        # Train or retrain model
        if not self.ml_predictor.is_trained or self.should_retrain(df.index[-1]):
            train_data = df.iloc[-self.lookback_period :]
            try:
                self.train_model(train_data)
            except Exception as e:
                logger.error(f"Training failed: {e}")
                return df

        # Generate features for prediction
        df_with_features = self.calculate_indicators(df)

        # Get feature columns
        feature_cols = [
            col
            for col in df_with_features.columns
            if col not in ["open", "high", "low", "close", "volume", "signal", "position_size", "expected_return", "confidence"]
        ]

        # Make predictions on recent data
        recent_features = df_with_features[feature_cols].iloc[-1:]

        try:
            predictions, uncertainty = self.ml_predictor.predict(recent_features)
            predicted_return = predictions[0]
            predicted_volatility = uncertainty[0]
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return df

        # Run Monte Carlo simulation
        current_price = df["close"].iloc[-1]
        paths = self.mc_engine.simulate_paths(
            current_price=current_price,
            predicted_return=predicted_return,
            predicted_volatility=predicted_volatility,
            forecast_horizon=self.forecast_horizon,
        )

        # Calculate simulation statistics
        stats = self.mc_engine.calculate_statistics(paths, self.confidence_level)
        self.simulation_results = stats

        # Generate signal based on probability and expected return
        if stats["prob_profit"] > 0.55 and stats["expected_return"] > 0.01:
            df.loc[df.index[-1], "signal"] = 1
        elif stats["prob_profit"] < 0.45 and stats["expected_return"] < -0.01:
            df.loc[df.index[-1], "signal"] = -1
        else:
            df.loc[df.index[-1], "signal"] = 0

        # Calculate position size
        if df.loc[df.index[-1], "signal"] != 0:
            position_size = self.position_sizer.calculate_position_size(
                portfolio_value=100000,  # Placeholder
                simulation_stats=stats,
                current_price=current_price,
            )
            df.loc[df.index[-1], "position_size"] = position_size

        # Store additional info
        df.loc[df.index[-1], "expected_return"] = stats["expected_return"]
        df.loc[df.index[-1], "confidence"] = stats["prob_profit"]

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
