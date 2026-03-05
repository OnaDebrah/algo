"""
Hedging service that integrates with ML crash predictions
"""

import logging
import math
from datetime import date, datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.data.providers.providers import ProviderFactory
from ...core.quantlib_hedge import QuantLibHedgeEngine
from ...models.user import User

logger = logging.getLogger(__name__)


class HedgeRecommendationService:
    """
    Service that combines ML crash predictions with options hedging.

    Uses fast proxy methods for LPPLS and LSTM analysis instead of full models:
    - LPPLS proxy: feature-based bubble detection (~0.05s vs 90+ sec)
    - LSTM proxy: feature-based stress estimation (~0.01s vs 30+ sec)
    """

    def __init__(self):
        self.provider_factory = ProviderFactory()
        self.hedge_engine = QuantLibHedgeEngine()

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
        """Get combined predictions from all ML models.

        Uses fast proxy methods for both LPPLS and LSTM stress analysis.
        - LPPLS full model (differential_evolution) takes 90+ sec and blocks event loop
        - LSTM full model (Keras training) takes 30+ sec and produces NaN with market data
        Both proxies give instant results using the same market features the models detect.
        """
        data = await self.provider_factory.fetch_data(symbol=symbol, period="1y", interval="1d", user=user, db=db)

        # LPPLS analysis — fast proxy (~0.05s vs 90+ sec for full model)
        lppls_prob = self._run_lppls_proxy(data)
        lppls_data = {
            "crash_probability": lppls_prob,
            "confidence": min(lppls_prob * 1.5, 1.0),
            "is_bubble": lppls_prob > 0.3,
            "method": "proxy",
            "action": "SHORT" if lppls_prob > 0.3 else "NO_ACTION",
        }

        # LSTM stress analysis — fast proxy (~0.01s vs 30+ sec for full Keras training)
        lstm_stress = self._run_lstm_proxy(data)
        # Determine trend from recent stress
        stress_trend = "stable"
        if len(data) >= 40:
            close = data["Close"]
            returns_recent = close.tail(20).pct_change().dropna().std()
            returns_prior = close.iloc[-40:-20].pct_change().dropna().std()
            if returns_recent > returns_prior * 1.2:
                stress_trend = "increasing"
            elif returns_recent < returns_prior * 0.8:
                stress_trend = "decreasing"

        lstm_data = {
            "stress_index": lstm_stress,
            "confidence": min(lstm_stress * 1.5, 1.0) if lstm_stress > 0.3 else 0.5,
            "stress_trend": stress_trend,
            "method": "proxy",
        }

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

    def _run_lstm_proxy(self, data: pd.DataFrame) -> float:
        """
        Fast LSTM stress proxy using market data features.

        Combines: realized volatility, drawdown severity, volume spike,
        and momentum breakdown — the same features the LSTM would learn.
        ~0.01s vs 30+ sec for full Keras model training.
        """
        try:
            if "Close" not in data.columns or len(data) < 60:
                return 0.5

            close = data["Close"]
            returns = close.pct_change().dropna()

            # 1. Realized volatility (20-day, annualized)
            vol_20 = returns.tail(20).std() * np.sqrt(252)
            vol_norm = min(float(vol_20) / 0.4, 1.0)  # 40% vol = max stress

            # 2. Drawdown severity
            peak = close.rolling(60, min_periods=1).max()
            drawdown = float(((close - peak) / peak).iloc[-1])
            dd_stress = min(abs(drawdown) / 0.2, 1.0)  # 20% drawdown = max stress

            # 3. Volume spike (if available)
            vol_stress = 0.0
            if "Volume" in data.columns:
                vol_mean = data["Volume"].rolling(20).mean().iloc[-1]
                if vol_mean > 0:
                    vol_ratio = float(data["Volume"].iloc[-1]) / float(vol_mean)
                    vol_stress = min(max(vol_ratio - 1, 0) / 3, 1.0)

            # 4. Momentum breakdown (price below 50-day SMA)
            sma_50 = close.rolling(50, min_periods=1).mean().iloc[-1]
            momentum_stress = 1.0 if float(close.iloc[-1]) < float(sma_50) else 0.0

            # Weighted combination
            stress = 0.35 * vol_norm + 0.30 * dd_stress + 0.15 * vol_stress + 0.20 * momentum_stress
            return float(min(max(stress, 0.0), 1.0))

        except Exception as e:
            logger.debug(f"LSTM proxy failed: {e}")
            return 0.5

    def _sanitize_dict(self, d: Dict) -> Dict:
        """Recursively replace NaN/Inf with safe defaults in a dict."""
        sanitized = {}
        for k, v in d.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                sanitized[k] = 0.0
            elif isinstance(v, dict):
                sanitized[k] = self._sanitize_dict(v)
            elif isinstance(v, list):
                sanitized[k] = [
                    0.0
                    if isinstance(item, float) and (math.isnan(item) or math.isinf(item))
                    else (self._sanitize_dict(item) if isinstance(item, dict) else item)
                    for item in v
                ]
            elif isinstance(v, (np.floating,)):
                fv = float(v)
                sanitized[k] = 0.0 if math.isnan(fv) or math.isinf(fv) else fv
            elif isinstance(v, (np.integer,)):
                sanitized[k] = int(v)
            elif isinstance(v, (pd.Timestamp, datetime)):
                sanitized[k] = str(v)
            else:
                sanitized[k] = v
        return sanitized

    def _run_lppls_proxy(self, data: pd.DataFrame) -> float:
        """
        Fast LPPLS bubble proxy using market features instead of differential_evolution.

        Approximates bubble probability in [0, 1] — ~0.05s vs ~10s for full model.
        Detects: super-exponential growth, trend deviation, volatility clustering,
        return acceleration, and rising vol with rising price.
        """
        try:
            if "Close" not in data.columns or len(data) < 60:
                return 0.0

            close = data["Close"].values.astype(float)
            log_price = np.log(close)
            n = len(close)

            # 1. Super-exponential growth detection (weight: 0.30)
            t = np.arange(n, dtype=float)
            coeffs = np.polyfit(t, log_price, 2)
            curvature = coeffs[0]
            super_exp_score = float(min(max(curvature / 5e-5, 0.0), 1.0))

            # 2. Price deviation from trend (weight: 0.25)
            if n >= 200:
                ema = pd.Series(close).ewm(span=200, min_periods=100).mean().values
                rolling_std = pd.Series(close).rolling(60, min_periods=30).std().values
            else:
                ema_span = max(n // 2, 30)
                ema = pd.Series(close).ewm(span=ema_span, min_periods=20).mean().values
                rolling_std = pd.Series(close).rolling(30, min_periods=15).std().values

            std_val = rolling_std[-1] if rolling_std[-1] > 0 else 1e-10
            z_score = (close[-1] - ema[-1]) / std_val
            deviation_score = float(min(max(z_score / 2.0, 0.0), 1.0))

            # 3. Log-periodic oscillation proxy (weight: 0.20)
            returns = np.diff(log_price)
            if len(returns) >= 60:
                vol_20 = np.std(returns[-20:])
                vol_60 = np.std(returns[-60:])
                vol_ratio = vol_20 / (vol_60 + 1e-10)
                oscillation_score = float(min(max((vol_ratio - 1.0) / 0.5, 0.0), 1.0))
            else:
                oscillation_score = 0.0

            # 4. Return acceleration (weight: 0.15)
            if n >= 60:
                ret_20 = (close[-1] / close[-20] - 1) * (252 / 20)
                ret_60 = (close[-1] / close[-60] - 1) * (252 / 60)
                if ret_60 > 0 and ret_20 > ret_60:
                    accel = (ret_20 - ret_60) / (ret_60 + 1e-10)
                    acceleration_score = float(min(max(accel, 0.0), 1.0))
                else:
                    acceleration_score = 0.0
            else:
                acceleration_score = 0.0

            # 5. Volatility regime (weight: 0.10)
            if n >= 40:
                vol_recent = np.std(returns[-20:]) * np.sqrt(252)
                vol_prior = np.std(returns[-40:-20]) * np.sqrt(252)
                price_up = close[-1] > close[-20]
                vol_rising = vol_recent > vol_prior * 1.1
                vol_regime_score = 1.0 if (price_up and vol_rising) else 0.0
            else:
                vol_regime_score = 0.0

            # Weighted combination
            proxy_prob = (
                0.30 * super_exp_score + 0.25 * deviation_score + 0.20 * oscillation_score + 0.15 * acceleration_score + 0.10 * vol_regime_score
            )
            return float(min(max(proxy_prob, 0.0), 1.0))

        except Exception as e:
            logger.debug(f"LPPLS proxy failed: {e}")
            return 0.0

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
