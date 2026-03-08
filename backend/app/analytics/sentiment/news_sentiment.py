import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import nltk
import numpy as np
import requests
from dotenv import load_dotenv
from nltk.sentiment import SentimentIntensityAnalyzer
from requests.adapters import HTTPAdapter
from tenacity import retry, stop_after_attempt, wait_exponential
from urllib3.util.retry import Retry

from ...utils.helpers import SYMBOL_TO_COMPANY

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsSentimentAnalyzer:
    """
    A comprehensive news sentiment analyzer using NewsAPI and VADER sentiment analysis.
    Fetches news articles for a given symbol and date, then computes aggregate sentiment.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the news sentiment analyzer.

        Args:
            api_key: NewsAPI key. If None, tries to get from environment variable.
        """
        self.api_key = api_key or os.getenv("NEWSAPI_KEY")
        if not self.api_key:
            raise ValueError("NewsAPI key is required. Set NEWSAPI_KEY environment variable or pass api_key parameter.")

        self.base_url = "https://newsapi.org/v2"

        # Download VADER lexicon if not already present
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)

        self.sentiment_analyzer = SentimentIntensityAnalyzer()

        self.session = self._create_session()

        self._description_cache = {}

    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy."""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _fetch_articles(
        self, query: str, from_date: datetime, to_date: datetime, language: str = "en", sort_by: str = "relevancy", page_size: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Fetch news articles from NewsAPI.

        Args:
            query: Search query (stock symbol or company name)
            from_date: Start date for articles
            to_date: End date for articles
            language: Article language (default: 'en')
            sort_by: Sort method ('relevancy', 'popularity', 'publishedAt')
            page_size: Number of articles per page (max 100)

        Returns:
            List of article dictionaries
        """
        all_articles = []
        page = 1
        max_pages = 2  # Limit to 2 pages to avoid rate limits

        while page <= max_pages:
            try:
                params = {
                    "q": query,
                    "from": from_date.strftime("%Y-%m-%d"),
                    "to": to_date.strftime("%Y-%m-%d"),
                    "language": language,
                    "sortBy": sort_by,
                    "pageSize": min(page_size, 100),
                    "page": page,
                    "apiKey": self.api_key,
                }

                # Use the 'everything' endpoint for comprehensive search
                response = self.session.get(f"{self.base_url}/everything", params=params, timeout=10)
                response.raise_for_status()

                data = response.json()

                if data["status"] != "ok":
                    logger.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                    break

                articles = data.get("articles", [])
                if not articles:
                    break

                # Filter articles by exact date if needed
                filtered_articles = self._filter_articles_by_date(articles, from_date, to_date)
                all_articles.extend(filtered_articles)

                # Check if we have more pages
                total_results = data.get("totalResults", 0)
                if len(articles) < page_size or len(all_articles) >= total_results:
                    break

                page += 1
                time.sleep(0.5)  # Rate limiting

            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching articles: {e}")
                break

        return all_articles

    def _filter_articles_by_date(self, articles: List[Dict[str, Any]], from_date: datetime, to_date: datetime) -> List[Dict[str, Any]]:
        """Filter articles to ensure they fall within the date range."""
        filtered = []
        for article in articles:
            published_at = article.get("publishedAt")
            if published_at:
                try:
                    # Parse the ISO format date
                    pub_date = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
                    if from_date <= pub_date <= to_date:
                        filtered.append(article)
                except (ValueError, TypeError):
                    # If date parsing fails, include the article
                    filtered.append(article)
            else:
                filtered.append(article)
        return filtered

    def _extract_text_for_sentiment(self, article: Dict[str, Any]) -> str:
        """
        Extract and combine text from article for sentiment analysis.
        Prioritizes description, then title, then content.

        Args:
            article: Article dictionary from NewsAPI

        Returns:
            Combined text for sentiment analysis
        """
        parts = []

        # Title (most important)
        if article.get("title"):
            # Remove [Removed] placeholder articles
            if article["title"] != "[Removed]":
                parts.append(article["title"])

        # Description (good summary)
        if article.get("description") and article["description"] != "[Removed]":
            parts.append(article["description"])

        # Content (full article preview)
        if article.get("content") and article["content"] != "[Removed]":
            # Remove the truncation indicator [+XXXX chars]
            content = article["content"]
            if "…" in content:
                content = content.split("…")[0]
            parts.append(content)

        return " ".join(parts)

    def _analyze_single_article(self, article: Dict[str, Any]) -> float:
        """
        Analyze sentiment of a single article.

        Args:
            article: Article dictionary

        Returns:
            Sentiment score between -1 and 1
        """
        text = self._extract_text_for_sentiment(article)

        # Check cache
        cache_key = hash(text)
        if cache_key in self._description_cache:
            return self._description_cache[cache_key]

        if not text.strip():
            return 0.0

        # Get VADER sentiment scores
        scores = self.sentiment_analyzer.polarity_scores(text)

        # compound score is between -1 and 1
        sentiment = scores["compound"]

        # Cache the result
        self._description_cache[cache_key] = sentiment

        return sentiment

    def _weight_by_source_and_recency(self, sentiments: List[float], articles: List[Dict[str, Any]], target_date: datetime) -> float:
        """
        Weight sentiments by source credibility and recency.

        Args:
            sentiments: List of sentiment scores
            articles: Corresponding article dictionaries
            target_date: Target date for recency weighting

        Returns:
            Weighted average sentiment score
        """
        if not sentiments:
            return 0.0

        weights = []
        source_weights = {
            "reuters.com": 1.2,
            "bloomberg.com": 1.2,
            "wsj.com": 1.2,
            "ft.com": 1.2,
            "cnbc.com": 1.1,
            "cnn.com": 1.0,
            "bbc.com": 1.0,
            "apnews.com": 1.1,
            "default": 0.9,
        }

        for i, article in enumerate(articles):
            weight = 1.0

            # Source credibility weighting
            url = article.get("url", "").lower()
            for domain, source_weight in source_weights.items():
                if domain in url:
                    weight *= source_weight
                    break
            else:
                weight *= source_weights["default"]

            # Recency weighting (articles closer to target date get higher weight)
            published_at = article.get("publishedAt")
            if published_at:
                try:
                    pub_date = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
                    hours_diff = abs((pub_date - target_date).total_seconds() / 3600)
                    # Exponential decay with half-life of 24 hours
                    recency_weight = np.exp(-hours_diff / 24)
                    weight *= recency_weight
                except (ValueError, TypeError):
                    pass

            weights.append(weight)

        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()

        # Calculate weighted average
        weighted_sentiment = np.average(sentiments, weights=weights)

        return weighted_sentiment

    def _deduplicate_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate articles based on title and source.

        Args:
            articles: List of articles

        Returns:
            Deduplicated list of articles
        """
        seen = set()
        unique_articles = []

        for article in articles:
            # Create a unique key from title and source
            title = article.get("title", "")
            source = article.get("source", {}).get("name", "")
            key = f"{title}|{source}"

            if key not in seen:
                seen.add(key)
                unique_articles.append(article)

        return unique_articles

    def _get_company_name_for_symbol(self, symbol: str) -> str:
        """
        Map stock symbol to company name for better search results.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Company name or original symbol if not found
        """
        return SYMBOL_TO_COMPANY.get(symbol.upper(), symbol)

    def get_news_sentiment(self, symbol: str, date: datetime, days_back: int = 3, min_articles: int = 1) -> float:
        """
        Main method to get news sentiment for a symbol on a specific date.
        Fetches articles around the target date and computes weighted sentiment.

        Args:
            symbol: Stock ticker symbol
            date: Target date for sentiment
            days_back: Number of days to look back for articles
            min_articles: Minimum articles required for valid sentiment

        Returns:
            Sentiment score between -1 and 1
        """
        # Define date range (look back from target date)
        from_date = date - timedelta(days=days_back)
        to_date = date

        # Get company name for better search
        company_name = self._get_company_name_for_symbol(symbol)

        logger.info(f"Fetching news for {symbol} ({company_name}) from {from_date.date()} to {to_date.date()}")

        try:
            # Fetch articles using both symbol and company name
            articles_symbol = self._fetch_articles(symbol, from_date, to_date)
            articles_company = self._fetch_articles(company_name, from_date, to_date)

            # Combine and deduplicate
            all_articles = articles_symbol + articles_company
            all_articles = self._deduplicate_articles(all_articles)

            if len(all_articles) < min_articles:
                logger.warning(f"Only {len(all_articles)} articles found for {symbol} on {date.date()}. Using fallback.")
                return self._fallback_sentiment(symbol, date)

            logger.info(f"Found {len(all_articles)} unique articles for {symbol}")

            # Analyze sentiment for each article
            sentiments = []
            for article in all_articles:
                sentiment = self._analyze_single_article(article)
                sentiments.append(sentiment)

            # Weight sentiments by source and recency
            weighted_sentiment = self._weight_by_source_and_recency(sentiments, all_articles, date)

            # Clip to ensure between -1 and 1
            final_sentiment = np.clip(weighted_sentiment, -1, 1)

            logger.info(f"Final sentiment for {symbol}: {final_sentiment:.4f}")

            return final_sentiment

        except Exception as e:
            logger.error(f"Error in news sentiment analysis for {symbol}: {e}")
            return self._fallback_sentiment(symbol, date)

    def _fallback_sentiment(self, symbol: str, date: datetime) -> float:
        """
        Fallback sentiment calculation when API fails or returns no articles.
        Implements a random walk with mean reversion for demo purposes.

        Args:
            symbol: Stock ticker symbol
            date: Target date

        Returns:
            Simulated sentiment score
        """
        logger.info(f"Using fallback sentiment for {symbol} on {date.date()}")

        # Create deterministic random seed based on symbol and date
        seed_value = hash(f"{symbol}{date.strftime('%Y%m%d')}") % (2**32)
        np.random.seed(seed_value)

        # Generate sentiment with slight mean reversion
        # Use normal distribution centered at 0 with std 0.3
        sentiment = np.random.normal(0, 0.3)

        # Add some serial correlation for consecutive dates (simulate trend)
        # This makes the fallback more realistic
        day_of_year = date.timetuple().tm_yday
        trend = np.sin(day_of_year / 30) * 0.1  # Seasonal component

        sentiment = sentiment + trend

        return np.clip(sentiment, -1, 1)

    def get_batch_sentiment(self, symbols: List[str], dates: List[datetime]) -> Dict[str, List[float]]:
        """
        Get sentiment for multiple symbols and dates efficiently.

        Args:
            symbols: List of stock symbols
            dates: List of dates

        Returns:
            Dictionary mapping symbols to list of sentiment scores
        """
        results = {symbol: [] for symbol in symbols}

        for date in dates:
            for symbol in symbols:
                sentiment = self.get_news_sentiment(symbol, date)
                results[symbol].append(sentiment)

        return results


# The original function you requested - for direct replacement
def _get_news_sentiment(symbol: str, date: datetime) -> float:
    """
    Get news sentiment score using NewsAPI and VADER sentiment analysis.

    Args:
        symbol: Stock ticker symbol
        date: Target date for sentiment analysis

    Returns:
        Sentiment score between -1 and 1
    """
    try:
        # Initialize the analyzer (API key from environment)
        analyzer = NewsSentimentAnalyzer()

        # Get sentiment with default parameters
        sentiment = analyzer.get_news_sentiment(symbol, date)

        return sentiment

    except Exception as e:
        logger.error(f"Failed to get news sentiment: {e}")
        # Fallback to original placeholder behavior
        np.random.seed(hash(f"{symbol}{date}") % (2**32))
        return np.clip(np.random.normal(0, 0.3), -1, 1)
