import logging
import json
from typing import Dict, List, Optional
import yfinance as yf
from backend.app.config import settings

logger = logging.getLogger(__name__)

class SentimentService:
    """Service for analyzing market sentiment using LLM and news data"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.ANTHROPIC_API_KEY
        self.client = None
        if self.api_key:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)

    async def get_sentiment(self, ticker: str) -> Dict:
        """
        Get sentiment analysis for a ticker
        
        Returns:
            Dict containing:
                score: float (-1.0 to 1.0)
                label: str (Bullish, Bearish, Neutral)
                summary: str (Brief explanation)
                headlines: List[str] (Headlines analyzed)
        """
        logger.info(f"Fetching sentiment for {ticker}")
        
        try:
            # 1. Fetch news from yfinance
            stock = yf.Ticker(ticker)
            news = stock.news
            
            if not news:
                return {
                    "score": 0.0,
                    "label": "Neutral",
                    "summary": "No recent news found for this ticker.",
                    "headlines": []
                }
            
            headlines = [item.get('title', '') for item in news[:10]]
            
            # 2. Analyze with LLM if available
            if self.client:
                sentiment_data = await self._analyze_headlines_with_llm(ticker, headlines)
                return {
                    "score": sentiment_data.get("score", 0.0),
                    "label": sentiment_data.get("label", "Neutral"),
                    "summary": sentiment_data.get("summary", "Analysis completed."),
                    "headlines": headlines
                }
            
            # 3. Fallback to simple keyword analysis
            return self._fallback_sentiment(headlines)
            
        except Exception as e:
            logger.error(f"Error in SentimentService: {e}")
            return {
                "score": 0.0,
                "label": "Error",
                "summary": f"Failed to analyze sentiment: {str(e)}",
                "headlines": []
            }

    async def _analyze_headlines_with_llm(self, ticker: str, headlines: List[str]) -> Dict:
        """Use Claude to analyze headlines and return structured sentiment"""
        
        prompt = f"""Analyze the market sentiment for {ticker} based on these recent headlines:
        {json.dumps(headlines, indent=2)}
        
        Respond ONLY with a JSON object in this format:
        {{
          "score": float (-1.0 for very bearish to 1.0 for very bullish),
          "label": "Bullish" | "Bearish" | "Neutral",
          "summary": "One sentence summary of why"
        }}
        """
        
        try:
            # Note: Using synchronous client library in an async wrapper for simplicity
            # In production, a truly async client or threadpool would be used.
            message = self.client.messages.create(
                model="claude-3-haiku-20240307", # Use faster model for sentiment
                max_tokens=200,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = message.content[0].text
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            
            return json.loads(response_text)
        except Exception as e:
            logger.warning(f"LLM Sentiment analysis failed: {e}")
            return {"score": 0.0, "label": "Neutral", "summary": "LLM analysis unavailable."}

    def _fallback_sentiment(self, headlines: List[str]) -> Dict:
        """Simple keyword-based sentiment analysis"""
        bullish_words = ["bullish", "upgrade", "buy", "gain", "profit", "beat", "positive", "growth", "high", "success"]
        bearish_words = ["bearish", "downgrade", "sell", "loss", "miss", "negative", "drop", "fall", "risk", "failure"]
        
        score = 0
        text = " ".join(headlines).lower()
        
        for word in bullish_words:
            score += text.count(word)
        for word in bearish_words:
            score -= text.count(word)
        
        # Normalize
        norm_score = max(-1.0, min(1.0, score / 10.0))
        label = "Bullish" if norm_score > 0.1 else "Bearish" if norm_score < -0.1 else "Neutral"
        
        return {
            "score": norm_score,
            "label": label,
            "summary": "Determined via keyword analysis of headlines."
        }
