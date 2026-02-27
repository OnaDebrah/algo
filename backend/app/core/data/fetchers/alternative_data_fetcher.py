import logging
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np

from ...data_fetcher import fetch_news, fetch_quote, fetch_recommendations

logger = logging.getLogger(__name__)


class AlternativeDataSource:
    """
    Unified interface for alternative data sources
    """

    def __init__(self, cache_ttl_hours: int = 24):
        self.cache = {}
        self.cache_ttl = timedelta(hours=cache_ttl_hours)

    async def get_sentiment_score(self, symbol: str, days_back: int = 7) -> Dict[str, float]:
        """
        Get sentiment score from news and social media
        Returns: {
            'overall_sentiment': float (-1 to 1),
            'news_sentiment': float,
            'social_sentiment': float,
            'analyst_sentiment': float,
            'volume': int,
            'momentum': float
        }
        """
        cache_key = f"sentiment_{symbol}_{days_back}"

        # Check cache
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if datetime.now() - cached_time < self.cache_ttl:
                return cached_data

        try:
            # Fetch news sentiment
            news_items = await fetch_news(symbol, limit=50)
            news_sentiment = self._calculate_news_sentiment(news_items)

            # Fetch analyst recommendations
            recommendations = await fetch_recommendations(symbol)
            analyst_sentiment = self._calculate_analyst_sentiment(recommendations)

            # Fetch quote for momentum
            quote = await fetch_quote(symbol)

            # Calculate momentum
            momentum = self._calculate_momentum(quote)

            # Combine sentiments
            overall = news_sentiment * 0.4 + analyst_sentiment * 0.4 + momentum * 0.2

            result = {
                "overall_sentiment": overall,
                "news_sentiment": news_sentiment,
                "analyst_sentiment": analyst_sentiment,
                "momentum_score": momentum,
                "news_volume": len(news_items),
                "timestamp": datetime.now().isoformat(),
            }

            # Cache result
            self.cache[cache_key] = (datetime.now(), result)

            return result

        except Exception as e:
            logger.error(f"Error fetching sentiment for {symbol}: {e}")
            return self._default_sentiment()

    async def get_sector_sentiment(self, sector: str, stocks: List[str], days_back: int = 7) -> Dict[str, float]:
        """
        Aggregate sentiment for an entire sector
        """
        sentiments = []

        # Get sentiment for top stocks in sector
        for symbol in stocks[:10]:  # Limit for performance
            try:
                sent = await self.get_sentiment_score(symbol, days_back)
                sentiments.append(sent["overall_sentiment"])
            except Exception as e:
                logger.debug(f"Error getting sentiment for {symbol}: {e}")
                continue

        if not sentiments:
            return {"sector_sentiment": 0, "sentiment_std": 0, "positive_ratio": 0, "sample_size": 0}

        return {
            "sector_sentiment": float(np.mean(sentiments)),
            "sentiment_std": float(np.std(sentiments)),
            "positive_ratio": float(np.sum(np.array(sentiments) > 0) / len(sentiments)),
            "sample_size": len(sentiments),
        }

    async def get_market_sentiment(self) -> Dict[str, float]:
        """
        Get overall market sentiment
        """
        # Key market symbols for sentiment
        market_symbols = ["SPY", "QQQ", "DIA", "VIX"]

        sentiments = []
        for symbol in market_symbols:
            try:
                sent = await self.get_sentiment_score(symbol, days_back=1)
                sentiments.append(sent["overall_sentiment"])
            except Exception:
                continue

        return {
            "market_sentiment": float(np.mean(sentiments)) if sentiments else 0,
            "fear_greed_index": self._calculate_fear_greed_index(sentiments) if sentiments else 50,
        }

    def _calculate_news_sentiment(self, news_items: List[Dict]) -> float:
        """
        Calculate sentiment from news items using NLP
        Simplified version - in production, use transformer models
        """
        if not news_items:
            return 0

        # Simple keyword-based sentiment
        positive_words = {"beat", "raise", "upgrade", "growth", "profit", "gain", "positive", "bull", "outperform"}
        negative_words = {"miss", "lower", "downgrade", "loss", "decline", "risk", "negative", "bear", "underperform"}

        sentiments = []
        for item in news_items[:20]:  # Limit to recent
            title = item.get("title", "").lower()

            pos_count = sum(1 for word in positive_words if word in title)
            neg_count = sum(1 for word in negative_words if word in title)

            if pos_count + neg_count > 0:
                sentiment = (pos_count - neg_count) / (pos_count + neg_count)
                sentiments.append(sentiment)

        return float(np.mean(sentiments)) if sentiments else 0

    def _calculate_analyst_sentiment(self, recommendations: List[Dict]) -> float:
        """
        Calculate sentiment from analyst recommendations
        """
        if not recommendations:
            return 0

        # Map ratings to numeric values
        rating_map = {"strong_buy": 1.0, "buy": 0.5, "hold": 0, "underperform": -0.5, "sell": -1.0}

        ratings = []
        for rec in recommendations[:10]:
            rating = rec.get("recommendation", "").lower().replace(" ", "_")
            ratings.append(rating_map.get(rating, 0))

        return float(np.mean(ratings)) if ratings else 0

    def _calculate_momentum(self, quote: Dict) -> float:
        """
        Calculate momentum score from quote data
        """
        try:
            change_pct = quote.get("regularMarketChangePercent", 0)
            if change_pct:
                # Normalize to -1 to 1 range
                return float(np.clip(change_pct / 5, -1, 1))
        except Exception:
            pass
        return 0

    def _calculate_fear_greed_index(self, sentiments: List[float]) -> float:
        """
        Calculate fear & greed index from sentiments
        0 = extreme fear, 100 = extreme greed
        """
        if not sentiments:
            return 50

        avg_sentiment = np.mean(sentiments)
        # Map -1..1 to 0..100
        return float((avg_sentiment + 1) * 50)

    def _default_sentiment(self) -> Dict:
        """Return default sentiment values"""
        return {
            "overall_sentiment": 0,
            "news_sentiment": 0,
            "analyst_sentiment": 0,
            "momentum_score": 0,
            "news_volume": 0,
            "timestamp": datetime.now().isoformat(),
        }
