import json
import logging
import warnings
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from anthropic.types import MessageParam
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..config import ANTHROPIC_MODEL_HAIKU_3, settings
from ..core.data.providers.providers import ProviderFactory

warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


class SentimentService:
    """Service for analyzing market sentiment using LLM and news data"""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "ProsusAI/finbert"):
        self.api_key = api_key or settings.ANTHROPIC_API_KEY
        self.client = None
        if self.api_key:
            import anthropic

            self.client = anthropic.Anthropic(api_key=self.api_key)

        # FINBERT
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_safetensors=True, clean_up_tokenization_spaces=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Move to GPU if available for faster batch processing
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

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
            # 1. Fetch news via provider layer
            provider = ProviderFactory()
            news = await provider.get_news(ticker, limit=10)

            if not news:
                return {"score": 0.0, "label": "Neutral", "summary": "No recent news found for this ticker.", "headlines": []}

            headlines = [item.get("title", "") for item in news[:10]]

            # 2. Analyze with LLM if available
            if self.client:
                sentiment_data = await self._analyze_headlines_with_llm(ticker, headlines)
                return {
                    "score": sentiment_data.get("score", 0.0),
                    "label": sentiment_data.get("label", "Neutral"),
                    "summary": sentiment_data.get("summary", "Analysis completed."),
                    "headlines": headlines,
                }

            # 3. Fallback to simple keyword analysis
            return self.analyze_with_finbert(headlines)

        except Exception as e:
            logger.error(f"Error in SentimentService: {e}")
            return {"score": 0.0, "label": "Error", "summary": f"Failed to analyze sentiment: {str(e)}", "headlines": []}

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
            messages: list[MessageParam] = [{"role": "user", "content": prompt}]
            # Note: Using synchronous client library in an async wrapper for simplicity
            # In production, a truly async client or threadpool would be used.
            message = self.client.messages.create(
                model=ANTHROPIC_MODEL_HAIKU_3,  # Use faster model for sentiment
                max_tokens=200,
                temperature=0,
                messages=messages,
            )

            response_text = message.content[0].text
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()

            return json.loads(response_text)
        except Exception as e:
            logger.warning(f"LLM Sentiment analysis failed: {e}")
            return {"score": 0.0, "label": "Neutral", "summary": "LLM analysis unavailable."}

    def analyze_with_finbert(self, headlines: List[str]) -> Dict[str, Union[str, float]]:
        """
        Standalone method to analyze text sentiment.
        Returns a dictionary with sentiment, confidence, and score.
        """
        full_text = " ".join(headlines)

        if not full_text or not full_text.strip():
            return {"sentiment": "neutral", "confidence": 0.0, "score": 0.0}
        try:
            # 1. Tokenize input
            inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = F.softmax(outputs.logits, dim=-1)

            # FinBERT Labels: 0 -> positive, 1 -> negative, 2 -> neutral
            probs = probabilities[0].tolist()

            # Map probabilities to a -1 to 1 score
            # (Positive Probability - Negative Probability)
            sentiment_score = probs[0] - probs[1]

            # Determine winning label
            labels = ["positive", "negative", "neutral"]
            max_index = torch.argmax(probabilities).item()

            return {"sentiment": labels[max_index], "confidence": round(probs[max_index], 4), "score": round(sentiment_score, 4)}
        except Exception as e:
            logger.error(f"FinBERT API Unavailable, using fallback: {e}")

            return self._fallback_sentiment(headlines)

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

        return {"score": norm_score, "label": label, "summary": "Determined via keyword analysis of headlines."}
