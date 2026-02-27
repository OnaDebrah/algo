"""
Hedging service that integrates with ML crash predictions
"""

import logging
from datetime import date, datetime, timedelta
from typing import Dict, List

from sqlalchemy.ext.asyncio import AsyncSession

from ...core.data.providers.providers import ProviderFactory
from ...core.quantlib_hedge import QuantLibHedgeEngine
from ...models.user import User
from ...strategies.analysis.lppls_bubbles_strategy import LPPLSBubbleStrategy
from ...strategies.analysis.lstm_stress_strategy import LSTMStressStrategy

logger = logging.getLogger(__name__)


class HedgeRecommendationService:
    """
    Service that combines ML crash predictions with options hedging
    """

    def __init__(self):
        self.provider_factory = ProviderFactory()
        self.hedge_engine = QuantLibHedgeEngine()
        self.lppls_strategy = LPPLSBubbleStrategy()
        self.lstm_strategy = LSTMStressStrategy()

    async def get_hedge_recommendation(
        self, user: User, db: AsyncSession, portfolio_value: float, portfolio_beta: float = 1.0, primary_index: str = "SPY"
    ) -> Dict:
        """
        Get comprehensive hedge recommendation based on ML predictions

        Args:
            user: User for provider selection
            db: Database session
            portfolio_value: Total portfolio value
            portfolio_beta: Portfolio beta (default 1.0)
            primary_index: Index to hedge against
        """
        # Step 1: Get ML predictions
        crash_analysis = await self._get_crash_predictions(user, db, primary_index)

        # Step 2: Get market data
        market_data = await self._get_market_data(user, db, primary_index)

        # Step 3: Setup hedge engine
        self.hedge_engine.setup_market(
            spot_price=market_data["index_price"], risk_free_rate=market_data["risk_free_rate"], volatility=market_data["implied_volatility"]
        )

        # Step 4: Get hedge recommendation based on ML
        recommendation = self.hedge_engine.suggest_hedge_structure(
            portfolio_value=portfolio_value,
            portfolio_beta=portfolio_beta,
            index_price=market_data["index_price"],
            crash_probability=crash_analysis["combined_probability"],
            crash_intensity=crash_analysis["intensity"],
        )

        # Step 5: Add implementation details
        recommendation["ml_signals"] = {
            "lppls_bubble": crash_analysis["lppls"].get("is_bubble", False),
            "lppls_confidence": crash_analysis["lppls"].get("confidence", 0),
            "lppls_crash_prob": crash_analysis["lppls"].get("crash_probability", 0),
            "lstm_stress": crash_analysis["lstm"].get("stress_index", 0.5),
            "lstm_confidence": crash_analysis["lstm"].get("confidence", 0),
            "combined_probability": crash_analysis["combined_probability"],
        }

        # Step 6: Add monitoring instructions
        recommendation["monitoring"] = {
            "rebalance_frequency": "Daily" if crash_analysis["intensity"] == "severe" else "Weekly",
            "alert_triggers": self._get_alert_triggers(crash_analysis),
            "stop_loss_levels": self._get_stop_loss_levels(crash_analysis),
        }

        return recommendation

    async def _get_crash_predictions(self, user: User, db: AsyncSession, symbol: str) -> Dict:
        """Get combined predictions from all ML models"""

        # Get data for analysis
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)

        data = await self.provider_factory.fetch_data(symbol=symbol, period=None, interval="1d", start=start_date, end=end_date, user=user, db=db)

        # LPPLS analysis
        lppls_signal = self.lppls_strategy.generate_signal(data)
        lppls_data = lppls_signal["metadata"]

        # LSTM stress analysis
        lstm_signal = self.lstm_strategy.generate_signal(data)
        lstm_data = lstm_signal["metadata"]

        # Combine probabilities
        lppls_prob = lppls_data.get("crash_probability") or 0.0
        lstm_stress = lstm_data.get("stress_index") or 0.5

        # Weighted combination
        combined_prob = 0.6 * lppls_prob + 0.4 * lstm_stress

        # Determine intensity
        if combined_prob >= 0.7:
            intensity = "severe"
        elif combined_prob >= 0.4:
            intensity = "moderate"
        else:
            intensity = "mild"

        return {
            "lppls": lppls_data,
            "lstm": lstm_data,
            "combined_probability": combined_prob,
            "intensity": intensity,
            "timestamp": datetime.now().isoformat(),
        }

    async def _get_market_data(self, user: User, db: AsyncSession, symbol: str) -> Dict:
        """Get current market data for hedging calculations"""

        # Get quote
        quote = await self.provider_factory.get_quote(symbol, user, db)

        # Get options data for implied vol
        expirations = await self.provider_factory.get_option_expirations(symbol, user, db)

        # Get risk-free rate (simplified - would fetch from Treasury data)
        risk_free_rate = 0.05  # 5% placeholder

        # Get implied volatility from ATM options
        implied_vol = 0.20  # 20% placeholder
        if expirations:
            try:
                chain = await self.provider_factory.get_option_chain(symbol, expirations[0], user, db)
                if not chain["calls"].empty:
                    # Approximate ATM implied vol
                    atm_calls = chain["calls"][abs(chain["calls"]["strike"] - quote["price"]) < quote["price"] * 0.05]
                    if not atm_calls.empty:
                        # Would need proper IV calculation
                        implied_vol = 0.20
            except Exception:
                pass

        return {
            "index_price": quote.get("price", 100),
            "risk_free_rate": risk_free_rate,
            "implied_volatility": implied_vol,
            "dividend_yield": 0.02,  # 2% placeholder
            "date": date.today(),
        }

    def _get_alert_triggers(self, crash_analysis: Dict) -> List[Dict]:
        """Get alert triggers based on ML predictions"""
        triggers = []

        if crash_analysis["lppls"].get("is_bubble", False):
            triggers.append(
                {
                    "condition": "Bubble detected",
                    "action": "Review put hedge",
                    "threshold": f"Confidence: {crash_analysis['lppls'].get('confidence', 0):.1%}",
                }
            )

        lstm_stress = crash_analysis["lstm"].get("stress_index", 0.5)
        if lstm_stress > 0.7:
            triggers.append({"condition": "High market stress", "action": "Consider increasing hedge", "threshold": f"Stress: {lstm_stress:.1%}"})

        if crash_analysis["lstm"].get("stress_trend", "stable") == "increasing":
            triggers.append({"condition": "Rising stress", "action": "Monitor daily", "threshold": "Trend increasing"})

        return triggers

    def _get_stop_loss_levels(self, crash_analysis: Dict) -> Dict:
        """Get stop-loss recommendations based on crash intensity"""
        intensity = crash_analysis["intensity"]

        if intensity == "severe":
            return {"portfolio_stop": "5% drawdown", "hedge_trigger": "Immediate full hedge", "cash_level": "Increase to 30%"}
        elif intensity == "moderate":
            return {"portfolio_stop": "10% drawdown", "hedge_trigger": "Add to hedge at 5% drop", "cash_level": "Maintain 15-20% cash"}
        else:
            return {"portfolio_stop": "15% drawdown", "hedge_trigger": "Monitor only", "cash_level": "Normal cash position"}
