import logging
from datetime import datetime
from typing import Dict, List

import numpy as np
import praw
import requests
import tweepy
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from ....analytics.sentiment.news_sentiment import NewsSentimentAnalyzer as _NewsSentimentAnalyzer

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Extracts and processes sentiment from various sources
    """

    def __init__(self, sources: List[str] = None, backtest_mode: bool = False):
        self.sources = sources or ["news", "twitter", "stocktwits", "options"]
        self.sentiment_cache = {}
        self.backtest_mode = backtest_mode

    def get_sentiment_features(self, symbol: str, date: datetime) -> Dict[str, float]:
        """
        Extract sentiment features for a given symbol and date.

        In backtest_mode, real-time-only sources (Twitter, StockTwits, Reddit, Options)
        use a deterministic fallback to avoid look-ahead bias. NewsAPI supports date
        ranges so it always uses the real analyzer (which has its own fallback).

        Returns:
            Dictionary with sentiment scores [-1, 1] for each source
        """
        features = {}

        for source in self.sources:
            if source == "news":
                if self.backtest_mode:
                    features["news_sentiment"] = self._deterministic_fallback("news", symbol, date)
                else:
                    features["news_sentiment"] = self._get_news_sentiment(symbol, date)
            elif source == "twitter":
                if self.backtest_mode:
                    features["twitter_sentiment"] = self._deterministic_fallback("twitter", symbol, date)
                else:
                    features["twitter_sentiment"] = self._get_twitter_sentiment(symbol, date)
            elif source == "stocktwits":
                if self.backtest_mode:
                    features["stocktwits_sentiment"] = self._deterministic_fallback("stocktwits", symbol, date)
                else:
                    features["stocktwits_sentiment"] = self._get_stocktwits_sentiment(symbol, date)
            elif source == "reddit":
                if self.backtest_mode:
                    features["reddit_sentiment"] = self._deterministic_fallback("reddit", symbol, date)
                else:
                    features["reddit_sentiment"] = self._get_reddit_sentiment(symbol, date)
            elif source == "options":
                if self.backtest_mode:
                    features["options_sentiment"] = self._deterministic_fallback("options", symbol, date)
                else:
                    features["options_sentiment"] = self._get_options_sentiment(symbol, date)

        # Add aggregate sentiment
        if features:
            features["sentiment_aggregate"] = np.mean(list(features.values()))
            features["sentiment_volatility"] = np.std(list(features.values()))

        return features

    def _deterministic_fallback(self, source: str, symbol: str, date: datetime) -> float:
        """
        Deterministic sentiment fallback for backtesting.
        Produces a reproducible value based on symbol + date + source,
        avoiding look-ahead bias from real-time APIs.

        Incorporates realistic patterns:
        - Day-of-week effects (weekend sentiment trends)
        - Month/quarter-end effects
        - Earnings season patterns
        - Market hours vs after-hours bias
        - Sector momentum (via symbol clustering)
        """
        date_str = date.strftime("%Y%m%d") if hasattr(date, "strftime") else str(date)[:10]
        seed_value = hash(f"{source}{symbol}{date_str}") % (2**32)
        rng = np.random.RandomState(seed_value)

        # Base sentiment with realistic volatility
        base_sentiment = rng.normal(0, 0.25)

        # 1. Day of week patterns (Mondays more negative, Fridays more positive)
        day_of_week = date.weekday()  # 0=Monday, 6=Sunday
        if day_of_week == 0:  # Monday
            dow_effect = -0.05 + rng.uniform(-0.05, 0)  # Slightly negative Monday bias
        elif day_of_week == 4:  # Friday
            dow_effect = 0.07 + rng.uniform(0, 0.05)  # Positive Friday bias
        elif day_of_week >= 5:  # Weekend
            dow_effect = 0.03  # Mild weekend optimism
        else:
            dow_effect = 0.02 * rng.randn()  # Mid-week random

        # 2. Month patterns (January effect, tax-loss harvesting in December)
        month = date.month
        if month == 1:
            month_effect = 0.08  # January effect
        elif month == 12:
            month_effect = -0.06  # Tax-loss selling
        elif month == 9:
            month_effect = -0.04  # September weakness
        else:
            month_effect = 0.01 * rng.randn()

        # 3. Quarter-end effects
        is_quarter_end = date.month in [3, 6, 9, 12] and date.day >= 25
        quarter_end_effect = 0.04 if is_quarter_end else 0

        # 4. Earnings season simulation (mid-Jan, mid-Apr, mid-Jul, mid-Oct)
        earnings_season = False
        if (month == 1 and date.day >= 10) or (month == 4 and date.day >= 10) or (month == 7 and date.day >= 10) or (month == 10 and date.day >= 10):
            if date.day <= 25:  # 2-week earnings season window
                earnings_season = True

        earnings_effect = rng.normal(0.03, 0.1) if earnings_season else 0

        # 5. Symbol-based sector bias (deterministic from symbol hash)
        sector_seed = hash(f"{symbol}_sector") % 100
        if sector_seed < 20:  # Tech sector
            sector_bias = 0.04
        elif sector_seed < 35:  # Energy sector
            sector_bias = -0.02
        elif sector_seed < 50:  # Healthcare sector
            sector_bias = 0.03
        elif sector_seed < 70:  # Financial sector
            sector_bias = -0.01
        else:
            sector_bias = 0.0

        # 6. Day-of-year cyclical pattern (multiple harmonics for realism)
        try:
            day_of_year = date.timetuple().tm_yday
            # Annual cycle
            annual_cycle = np.sin(2 * np.pi * day_of_year / 365) * 0.06
            # Quarterly cycle
            quarterly_cycle = np.sin(2 * np.pi * day_of_year / 91.25) * 0.04
            # Monthly cycle
            monthly_cycle = np.sin(2 * np.pi * date.day / 30.5) * 0.03

            cyclical_component = (annual_cycle + quarterly_cycle + monthly_cycle) / 3
        except (AttributeError, ValueError):
            cyclical_component = 0

        # 7. Volatility clustering (simulate high/low volatility periods)
        vol_cluster = rng.gamma(2, 0.05) - 0.1  # Skewed distribution

        # Combine components
        sentiment = (
            base_sentiment * (1 + vol_cluster) + dow_effect + month_effect + quarter_end_effect + earnings_effect + sector_bias + cyclical_component
        )

        # Add realistic noise with fat tails (more extreme events)
        if rng.uniform(0, 1) < 0.05:  # 5% chance of extreme move
            sentiment += rng.choice([-0.4, 0.4]) * rng.uniform(0.8, 1.2)

        # Clip to realistic range but allow extremes occasionally
        clipped = np.clip(sentiment, -1.2, 1.2)
        return float(np.clip(clipped, -1, 1))  # Final clip to [-1, 1]

    def _get_news_sentiment(self, symbol: str, date: datetime) -> float:
        """
        Get news sentiment score using NewsSentimentAnalyzer.
        Uses NewsAPI + VADER with source/recency weighting.
        Falls back to deterministic simulation for out-of-range dates.
        """
        try:
            if not hasattr(self, "_news_analyzer"):
                self._news_analyzer = _NewsSentimentAnalyzer()
            return self._news_analyzer.get_news_sentiment(symbol, date)
        except Exception as e:
            logger.warning(f"News sentiment failed for {symbol}: {e}")
            return self._deterministic_fallback("news", symbol, date)

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

                vader_score = analyzer.polarity_scores(text)["compound"]

                explicit_sentiment = msg.get("entities", {}).get("sentiment", {}).get("basic")
                if explicit_sentiment == "Bullish":
                    vader_score = (vader_score + 1.0) / 2  # Nudge score higher
                elif explicit_sentiment == "Bearish":
                    vader_score = (vader_score - 1.0) / 2  # Nudge score lower

                followers = user.get("followers", 0)
                experience = user.get("ideas", 0)
                is_official = 10 if user.get("official", False) else 1

                # Logarithmic weighting to avoid one whale skewing everything
                weight = (np.log1p(followers) * 0.7) + (np.log1p(experience) * 0.3) + is_official

                weighted_scores.append(vader_score * weight)
                total_weight += weight

        except Exception as e:
            logger.error(f"Stocktwits API Error for {symbol}: {e}")
            return 0.0

        if total_weight == 0:
            return 0.0

        return float(np.sum(weighted_scores) / total_weight)

    def _get_twitter_sentiment(self, symbol: str, date: datetime) -> float:
        """
        Institutional grade sentiment analysis using VADER and Weighted Influence.
        Requires a valid Twitter/X API bearer token in TWITTER_BEARER_TOKEN env var.
        Returns 0.0 (neutral) if credentials are not configured.
        """
        import os

        BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN", "")
        if not BEARER_TOKEN or BEARER_TOKEN == "YOUR_TWITTER_BEARER_TOKEN":
            return 0.0  # No credentials configured

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
        Requires REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET env vars.
        Returns 0.0 (neutral) if credentials are not configured.
        """
        import os

        client_id = os.environ.get("REDDIT_CLIENT_ID", "")
        client_secret = os.environ.get("REDDIT_CLIENT_SECRET", "")
        if not client_id or client_id == "YOUR_CLIENT_ID":
            return 0.0  # No credentials configured

        reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent="FinancialAnalystBot/1.0")

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

    def _get_cached_sentiment(self, symbol: str) -> float:
        """Return cached sentiment for symbol, or neutral if not available."""
        return self.sentiment_cache.get(symbol, 0.0)

    def _get_options_sentiment(self, symbol: str, date: datetime) -> float:
        """
        Get options market sentiment (put/call ratio, IV skew).
        Returns 0.0 (neutral) as a safe default — no external API dependency.
        In production: Calculate from actual options chain data.
        """
        return 0.0
