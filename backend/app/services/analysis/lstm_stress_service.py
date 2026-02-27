"""
Service for LSTM-based market stress prediction
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.data.providers.providers import ProviderFactory
from ...models.user import User
from ...strategies.analysis.lstm_stress_strategy import LSTMStressStrategy

logger = logging.getLogger(__name__)


class LSTMStressService:
    """
    Service for LSTM-based market stress prediction
    Uses multiple market indices to forecast systemic risk
    """

    def __init__(self):
        self.provider_factory = ProviderFactory()
        self.strategy = LSTMStressStrategy()
        self.last_analysis = None
        self.market_stress_level = 0.5
        self.alert_triggers = []

    async def analyze_market_stress(
        self, symbols: List[str], user: Optional[User] = None, db: Optional[AsyncSession] = None, period: str = "2y"
    ) -> Dict:
        """
        Analyze overall market stress using multiple symbols

        Args:
            symbols: List of market indices to analyze
            user: User for provider selection
            db: Database session
            period: Days of historical data

        Returns:
            Market stress analysis with predictions
        """
        try:
            # Fetch data for all symbols
            combined_data = await self._fetch_multi_symbol_data(symbols, user, db, period)

            if combined_data.empty:
                return {"error": "Insufficient data", "stress_index": 0.5}

            # Generate signal using strategy
            signal = self.strategy.generate_signal(combined_data)

            # Extract stress metrics
            stress_index = signal["metadata"].get("stress_index", 0.5)
            confidence = signal["metadata"].get("confidence", 0.0)
            action = signal["metadata"].get("action", "HOLD")
            message = signal["metadata"].get("message", "")
            stress_trend = signal["metadata"].get("stress_trend", "stable")

            # Store current stress level
            self.market_stress_level = stress_index
            self.last_analysis = datetime.now()

            # Check for alert conditions
            alert = self._check_alert_conditions(stress_index, confidence, stress_trend)
            if alert:
                self.alert_triggers.append({"timestamp": datetime.now(), "alert": alert})

            result = {
                "timestamp": datetime.now().isoformat(),
                "stress_index": stress_index,
                "confidence": confidence,
                "action": action,
                "message": message,
                "trend": stress_trend,
                "position_size": signal["position_size"],
                "tap_deviation": signal["metadata"].get("tap_deviation", 0),
                "stress_history": signal["metadata"].get("stress_history", []),
                "alert": alert,
            }

            # Add 60-day forecast
            result["forecast"] = await self._generate_forecast(stress_index, confidence, stress_trend)

            return result

        except Exception as e:
            logger.error(f"Error analyzing market stress: {e}")
            return {"error": str(e), "stress_index": 0.5}

    async def _fetch_multi_symbol_data(self, symbols: List[str], user: Optional[User], db: Optional[AsyncSession], period: str) -> pd.DataFrame:
        """Fetch and combine data from multiple symbols"""
        # end_date = datetime.now()
        # start_date = end_date - timedelta(days=period)

        all_data = []

        for symbol in symbols:
            try:
                data = await self.provider_factory.fetch_data(symbol=symbol, period=period, interval="1d", user=user, db=db)

                if not data.empty:
                    # Prefix columns with symbol
                    data.columns = [f"{symbol}_{col}" for col in data.columns]
                    all_data.append(data)

            except Exception as e:
                logger.warning(f"Error fetching {symbol}: {e}")

        if not all_data:
            return pd.DataFrame()

        # Combine all data
        combined = pd.concat(all_data, axis=1)

        # Add market-wide indicators
        if len(symbols) >= 2:
            # Calculate cross-market correlations
            combined["cross_market_corr"] = self._calculate_cross_correlation(combined, symbols)

        return combined

    def _calculate_cross_correlation(self, df: pd.DataFrame, symbols: List[str]) -> pd.Series:
        """Calculate rolling correlation between markets"""
        returns = pd.DataFrame()

        for symbol in symbols:
            col = f"{symbol}_Close"
            if col in df.columns:
                returns[symbol] = df[col].pct_change()

        if len(returns.columns) >= 2:
            # Rolling average correlation
            corr_matrix = returns.rolling(60).corr()
            n = len(returns.columns)
            avg_corr = pd.Series(index=df.index, dtype=float)

            # rolling(60).corr() returns a MultiIndex DataFrame — use .xs() to
            # extract the n×n sub-matrix for each timestamp
            for idx in df.index[59:]:
                try:
                    corr_slice = corr_matrix.xs(idx, level=0)
                    upper_tri = corr_slice.values[np.triu_indices(n, k=1)]
                    avg_corr.loc[idx] = np.mean(upper_tri)
                except (KeyError, ValueError):
                    continue

            return avg_corr

        return pd.Series(index=df.index, data=0)

    def _check_alert_conditions(self, stress_index: float, confidence: float, trend: str) -> Optional[Dict]:
        """Check if alert conditions are met"""
        if confidence < 0.5:
            return None

        if stress_index >= 0.8:
            return {
                "level": "CRITICAL",
                "message": f"Extreme market stress detected: {stress_index:.1%}",
                "recommendation": "Implement full tail-risk hedge immediately",
            }
        elif stress_index >= 0.7:
            return {"level": "HIGH", "message": f"High market stress: {stress_index:.1%}", "recommendation": "Consider hedging and reducing exposure"}
        elif stress_index >= 0.6 and trend == "increasing":
            return {"level": "MODERATE", "message": f"Rising market stress: {stress_index:.1%}", "recommendation": "Monitor closely, prepare hedges"}

        return None

    async def _generate_forecast(self, stress_index: float, confidence: float, trend: str) -> Dict:
        """Generate 60-day forward-looking forecast"""

        if confidence < 0.3:
            forecast_quality = "low"
            forecast_range = "uncertain"
        elif confidence < 0.6:
            forecast_quality = "medium"
        else:
            forecast_quality = "high"

        if trend == "increasing":
            projected_stress = min(1.0, stress_index * 1.2)
            outlook = "deteriorating"
        elif trend == "decreasing":
            projected_stress = max(0, stress_index * 0.8)
            outlook = "improving"
        else:
            projected_stress = stress_index
            outlook = "stable"

        return {
            "horizon_days": 60,
            "current_stress": stress_index,
            "projected_stress_60d": projected_stress,
            "outlook": outlook,
            "forecast_range": forecast_range,
            "forecast_quality": forecast_quality,
            "key_indicators": {
                "volatility_regime": "high" if stress_index > 0.6 else "normal",
                "liquidity_conditions": "tight" if stress_index > 0.7 else "normal",
                "correlation_regime": "high" if stress_index > 0.65 else "normal",
            },
        }

    def get_stress_history(self, days: int = 30) -> List[float]:
        """Get historical stress index values"""
        return self.strategy.stress_history[-days:] if self.strategy.stress_history else []

    def get_confidence_history(self, days: int = 30) -> List[float]:
        """Get historical confidence values"""
        return self.strategy.confidence_history[-days:] if self.strategy.confidence_history else []
