import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..core.data.fetchers.fundamentals_fetcher import FundamentalsFetcher
from ..core.data.providers.providers import ProviderFactory
from ..strategies.ml.stock_selection.factor_engine import FactorEngine
from ..strategies.ml.stock_selection.ranking_pipeline import (
    RankingConfig,
    RankingMethod,
    RankingPipeline,
)
from ..strategies.strategy_catalog import get_catalog

logger = logging.getLogger(__name__)


# Regime → strategy mapping
REGIME_STRATEGY_MAP: Dict[str, List[str]] = {
    "bull": ["macd", "parabolic_sar", "ts_momentum", "sma_crossover", "donchian_channel"],
    "neutral": ["bollinger_mean_reversion", "rsi", "sma_crossover", "kama"],
    "bear": ["rsi", "bollinger_mean_reversion", "volatility_targeting", "dynamic_vol_scaling"],
    "high_volatility": ["volatility_breakout", "donchian_atr", "dynamic_vol_scaling", "variance_risk_premium"],
    "low_volatility": ["sma_crossover", "bollinger_mean_reversion", "kama", "macd"],
}


class SectorScannerService:
    """Orchestrates sector scanning, stock ranking, and strategy recommendation."""

    def __init__(self):
        self.provider = ProviderFactory()
        self.fundamentals = FundamentalsFetcher()
        self.factor_engine = FactorEngine()

    def get_sectors(self) -> List[Dict[str, Any]]:
        """Return list of sectors with metadata."""
        sectors = []
        for name, stocks in FundamentalsFetcher.SECTOR_MAPPINGS.items():
            etf = FundamentalsFetcher.SECTOR_ETFS.get(name, "")
            sectors.append(
                {
                    "name": name,
                    "etf": etf,
                    "stock_count": len(stocks),
                    "top_stocks": stocks[:5],
                }
            )
        return sectors

    async def scan_sectors(self, period: str = "6mo") -> Dict[str, Any]:
        """
        Scan all sectors using ETF data to rank by momentum and risk-adjusted returns.

        Returns dict with ranked sectors and metadata.
        """
        etf_map = FundamentalsFetcher.SECTOR_ETFS
        sector_results: List[Dict[str, Any]] = []

        # Fetch all ETF data in parallel
        async def fetch_etf(sector: str, etf: str) -> Optional[Dict[str, Any]]:
            try:
                data = await self.provider.fetch_data(etf, period, "1d")
                if data is None or data.empty or len(data) < 20:
                    return None

                prices = data["Close"]
                momentum = FactorEngine.momentum_factor(prices)
                volatility = FactorEngine.volatility_factor(prices)

                # Compute simple return over period
                total_return = float(prices.iloc[-1] / prices.iloc[0] - 1)

                # Composite score: momentum-weighted + inverse vol
                mom_score = momentum.get("momentum_weighted", momentum.get("momentum_21d", 0))
                vol_21d = volatility.get("volatility_21d", 0.2)
                risk_adj = mom_score / vol_21d if vol_21d > 0 else 0

                return {
                    "name": sector,
                    "etf": etf,
                    "stock_count": len(FundamentalsFetcher.SECTOR_MAPPINGS.get(sector, [])),
                    "top_stocks": FundamentalsFetcher.SECTOR_MAPPINGS.get(sector, [])[:5],
                    "momentum_score": round(mom_score, 4),
                    "volatility": round(vol_21d, 4),
                    "total_return": round(total_return, 4),
                    "composite_score": round(risk_adj, 4),
                    "momentum_factors": {k: round(v, 4) for k, v in momentum.items()},
                    "volatility_factors": {k: round(v, 4) for k, v in volatility.items()},
                }
            except Exception as e:
                logger.error(f"Error scanning sector {sector} ({etf}): {e}")
                return None

        tasks = [fetch_etf(sector, etf) for sector, etf in etf_map.items()]
        results = await asyncio.gather(*tasks)

        sector_results = [r for r in results if r is not None]

        # Rank by composite score
        sector_results.sort(key=lambda x: x["composite_score"], reverse=True)
        for i, s in enumerate(sector_results):
            s["rank"] = i + 1

        # Detect overall market regime from SPY
        market_regime = "neutral"
        try:
            spy_data = await self.provider.fetch_data("SPY", period, "1d")
            if spy_data is not None and not spy_data.empty:
                spy_mom = FactorEngine.momentum_factor(spy_data["Close"])
                spy_vol = FactorEngine.volatility_factor(spy_data["Close"])
                mom_w = spy_mom.get("momentum_weighted", 0)
                vol_ratio = spy_vol.get("volatility_ratio", 1.0)

                if mom_w > 0.5:
                    market_regime = "bull"
                elif mom_w < -0.5:
                    market_regime = "bear"
                elif vol_ratio > 1.3:
                    market_regime = "high_volatility"
                elif vol_ratio < 0.7:
                    market_regime = "low_volatility"
        except Exception as e:
            logger.warning(f"Could not detect market regime: {e}")

        return {
            "sectors": sector_results,
            "scan_date": datetime.now().isoformat(),
            "market_regime": market_regime,
        }

    async def rank_stocks(self, sector: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Rank stocks in a sector using multi-factor analysis.

        Uses FactorEngine for momentum, volatility, quality, valuation, growth factors
        then feeds into RankingPipeline for ensemble scoring.
        """
        symbols = FundamentalsFetcher.SECTOR_MAPPINGS.get(sector, [])
        if not symbols:
            raise ValueError(f"Unknown sector: {sector}")

        # Fetch price data + fundamentals for all stocks in parallel
        async def fetch_stock_data(symbol: str) -> Optional[Dict[str, Any]]:
            try:
                # Fetch 1y daily prices
                price_data = await self.provider.fetch_data(symbol, "1y", "1d")
                if price_data is None or price_data.empty:
                    return None

                # Fetch ticker info for fundamentals
                info = await self.provider.get_ticker_info(symbol)

                # Extract fundamentals dict for FactorEngine
                fundamentals = {}
                if info:
                    fundamentals["roe"] = info.get("returnOnEquity", 0) or 0
                    fundamentals["operating_margin"] = info.get("operatingMargins", 0) or 0
                    fundamentals["debt_to_equity"] = info.get("debtToEquity", 0) or 0
                    fundamentals["pe_ratio"] = info.get("trailingPE", 0) or 0
                    fundamentals["forward_pe"] = info.get("forwardPE", 0) or 0
                    fundamentals["peg_ratio"] = info.get("pegRatio", 0) or 0
                    fundamentals["pb_ratio"] = info.get("priceToBook", 0) or 0
                    fundamentals["revenue_growth"] = info.get("revenueGrowth", 0) or 0
                    fundamentals["eps_growth"] = info.get("earningsGrowth", 0) or 0

                market_cap = (info.get("marketCap", 0) or 0) if info else 0

                return {
                    "prices": price_data["Close"],
                    "volume": price_data["Volume"],
                    "fundamentals": fundamentals,
                    "market_cap": market_cap,
                    "sector": sector,
                }
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                return None

        tasks = [fetch_stock_data(s) for s in symbols]
        results = await asyncio.gather(*tasks)

        # Build stocks_data dict for RankingPipeline
        stocks_data: Dict[str, Dict[str, Any]] = {}
        for symbol, result in zip(symbols, results):
            if result is not None:
                stocks_data[symbol] = result

        if not stocks_data:
            return []

        # Run ranking pipeline
        config = RankingConfig(
            method=RankingMethod.ENSEMBLE,
            max_stocks=top_n,
            use_factor_neutralization=False,  # all same sector
            use_winsorization=True,
        )
        pipeline = RankingPipeline(config=config, factor_engine=self.factor_engine)
        ranked = await pipeline.rank_stocks(stocks_data, sector_neutral=False)

        return [stock.to_dict() for stock in ranked]

    async def recommend_strategies(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Recommend trading strategies for a given symbol based on its current regime.

        Uses price data to detect regime, then maps to best-fit strategies from catalog.
        """
        # Fetch price data
        price_data = await self.provider.fetch_data(symbol, "1y", "1d")
        if price_data is None or price_data.empty:
            raise ValueError(f"No price data available for {symbol}")

        # Detect regime using simple heuristics (avoids hmmlearn dependency issues)
        regime = self._detect_simple_regime(price_data)

        # Get strategy recommendations for this regime
        catalog = get_catalog()
        strategy_keys = REGIME_STRATEGY_MAP.get(regime, REGIME_STRATEGY_MAP["neutral"])

        recommendations = []
        for key in strategy_keys:
            info = catalog.get_info(key)
            if info is None:
                continue

            # Compute suitability score based on how well the strategy fits the regime
            suitability = self._compute_suitability(info, regime, price_data)

            recommendations.append(
                {
                    "strategy_name": info.name,
                    "strategy_key": key,
                    "category": info.category.value,
                    "suitability_score": round(suitability, 2),
                    "regime": regime,
                    "reason": self._get_recommendation_reason(info, regime),
                    "backtest_mode": info.backtest_mode,
                    "complexity": info.complexity,
                    "time_horizon": info.time_horizon,
                }
            )

        # Sort by suitability
        recommendations.sort(key=lambda x: x["suitability_score"], reverse=True)
        return recommendations

    def _detect_simple_regime(self, price_data: pd.DataFrame) -> str:
        """Simple regime detection from price data."""
        closes = price_data["Close"]
        returns = closes.pct_change().dropna()

        # 21-day momentum
        if len(closes) >= 21:
            mom_21d = closes.iloc[-1] / closes.iloc[-21] - 1
        else:
            mom_21d = 0

        # Volatility (annualised)
        vol_21d = returns.iloc[-21:].std() * np.sqrt(252) if len(returns) >= 21 else 0.2

        # Trend: price vs 50-day SMA
        sma_50 = closes.rolling(50).mean().iloc[-1] if len(closes) >= 50 else closes.mean()
        price_vs_sma = closes.iloc[-1] / sma_50 - 1

        # Vol regime
        vol_ratio = 1.0
        if len(returns) >= 63:
            vol_short = returns.iloc[-21:].std()
            vol_long = returns.iloc[-63:].std()
            vol_ratio = vol_short / vol_long if vol_long > 0 else 1.0

        if vol_ratio > 1.5 or vol_21d > 0.35:
            return "high_volatility"
        elif vol_ratio < 0.6 and vol_21d < 0.12:
            return "low_volatility"
        elif mom_21d > 0.03 and price_vs_sma > 0.02:
            return "bull"
        elif mom_21d < -0.03 and price_vs_sma < -0.02:
            return "bear"
        else:
            return "neutral"

    def _compute_suitability(self, info: Any, regime: str, price_data: pd.DataFrame) -> float:
        """Compute suitability score (0-1) for a strategy given the regime."""
        score = 0.5  # base

        # Regime fit bonuses
        regime_fit = {
            "bull": {"Trend Following": 0.3, "Momentum": 0.3, "Technical Indicators": 0.2},
            "bear": {"Mean Reversion": 0.3, "Volatility": 0.2, "Technical Indicators": 0.2},
            "high_volatility": {"Volatility": 0.3, "Price Action": 0.2, "Adaptive Strategies": 0.2},
            "low_volatility": {"Mean Reversion": 0.3, "Technical Indicators": 0.2, "Trend Following": 0.1},
            "neutral": {"Mean Reversion": 0.2, "Technical Indicators": 0.2, "Adaptive Strategies": 0.15},
        }

        fits = regime_fit.get(regime, {})
        score += fits.get(info.category.value, 0)

        # Complexity bonus for simpler strategies
        if info.complexity == "Beginner":
            score += 0.05
        elif info.complexity == "Advanced":
            score -= 0.05

        return min(max(score, 0.1), 0.99)

    def _get_recommendation_reason(self, info: Any, regime: str) -> str:
        """Generate a short reason for why this strategy is recommended."""
        reasons = {
            "bull": f"{info.name} works well in bullish/trending markets",
            "bear": f"{info.name} is suited for bearish conditions and can profit from mean-reversion",
            "high_volatility": f"{info.name} is designed to capitalise on high volatility",
            "low_volatility": f"{info.name} performs well in calm, range-bound markets",
            "neutral": f"{info.name} is versatile and works in mixed market conditions",
        }
        base = reasons.get(regime, f"{info.name} is recommended for current conditions")
        if info.best_for:
            base += f". Best for: {', '.join(info.best_for[:2])}"
        return base
