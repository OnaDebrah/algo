"""
Service for LPPLS bubble detection using Yahoo Finance provider
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.data.providers.providers import ProviderFactory
from ...models.user import User
from ...strategies.analysis.lppls_bubbles_strategy import LPPLSBubbleStrategy

logger = logging.getLogger(__name__)


class LPPLSService:
    """
    Service for LPPLS-based bubble detection
    Uses ProviderFactory to fetch data through user's configured provider
    """

    def __init__(self):
        self.provider_factory = ProviderFactory()
        self.strategies = {}

    async def analyze_symbol(
        self, symbol: str, user: Optional[User] = None, db: Optional[AsyncSession] = None, lookback_days: int = 365, params: Optional[Dict] = None
    ) -> Dict:
        """
        Analyze a single symbol for bubble detection

        Args:
            symbol: Stock/crypto symbol
            user: User for provider selection
            db: Database session
            lookback_days: Days of historical data to analyze
            params: Strategy parameters

        Returns:
            Dictionary with bubble analysis results
        """
        try:
            # Fetch data using provider factory
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)

            data = await self.provider_factory.fetch_data(
                symbol=symbol,
                period="1y",  # Yahoo Finance period
                interval="1d",  # Daily data
                start=start_date,
                end=end_date,
                user=user,
                db=db,
            )

            if data.empty:
                return {"symbol": symbol, "error": "No data available", "bubble_detected": False}

            # Create or get strategy
            strategy_key = f"{symbol}_{hash(frozenset(params.items())) if params else 'default'}"
            if strategy_key not in self.strategies:
                self.strategies[strategy_key] = LPPLSBubbleStrategy(name=f"LPPLS_{symbol}", params=params or {})

            strategy = self.strategies[strategy_key]

            # Generate signal
            signal = strategy.generate_signal(data)

            # Get DTCAI history
            dtcai_history = strategy.get_dtcai_timeseries()

            # Get fundamental data for context
            quote = await self.provider_factory.get_quote(symbol, user, db)
            info = await self.provider_factory.get_ticker_info(symbol, user, db)

            # Prepare response
            result = {
                "symbol": symbol,
                "analysis_date": datetime.now().isoformat(),
                "current_price": quote.get("price", 0),
                "bubble_detected": signal["metadata"].get("is_bubble", False),
                "confidence": signal["metadata"].get("confidence", 0),
                "crash_probability": signal["metadata"].get("crash_probability", 0),
                "critical_date": signal["metadata"].get("critical_date"),
                "action": signal["metadata"].get("action", "NO_ACTION"),
                "signal_strength": signal["metadata"].get("signal_strength", 0),
                "reasons": signal["metadata"].get("reasons", []),
                "parameters": signal["metadata"].get("parameters", {}),
                "dtcai_trend": self._calculate_trend(dtcai_history) if not dtcai_history.empty else None,
                "fundamental_context": {
                    "sector": info.get("sector"),
                    "industry": info.get("industry"),
                    "market_cap": quote.get("marketCap"),
                    "pe_ratio": info.get("trailingPE"),
                    "volume_ratio": quote.get("volume", 0) / quote.get("avgVolume", 1) if quote.get("avgVolume") else 1,
                },
            }

            # Add alert if needed
            if result["bubble_detected"] and result["confidence"] >= 0.6:
                result["alert"] = {
                    "level": "HIGH" if result["crash_probability"] >= 0.5 else "MODERATE",
                    "message": f"Bubble detected with {result['confidence']:.1%} confidence. "
                    f"Crash probability: {result['crash_probability']:.1%}",
                    "recommendation": "Consider hedging or reducing exposure",
                }

            return result

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return {"symbol": symbol, "error": str(e), "bubble_detected": False}

    async def analyze_multiple(self, symbols: List[str], user: Optional[User] = None, db: Optional[AsyncSession] = None) -> Dict[str, Dict]:
        """
        Analyze multiple symbols for bubble detection
        """
        results = {}
        for symbol in symbols:
            results[symbol] = await self.analyze_symbol(symbol, user, db)

        # Sort by crash probability
        sorted_results = dict(sorted(results.items(), key=lambda x: x[1].get("crash_probability", 0), reverse=True))

        return sorted_results

    def _calculate_trend(self, dtcai_history: pd.DataFrame) -> str:
        """Calculate trend in DTCAI values"""
        if len(dtcai_history) < 5:
            return "insufficient_data"

        recent = dtcai_history["dtcai"].iloc[-5:].values
        if len(recent) < 2:
            return "stable"

        slope = np.polyfit(range(len(recent)), recent, 1)[0]

        if slope > 0.05:
            return "increasing"
        elif slope < -0.05:
            return "decreasing"
        else:
            return "stable"
