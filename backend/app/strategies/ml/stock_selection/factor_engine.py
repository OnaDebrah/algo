from typing import Dict, Optional

import numpy as np
import pandas as pd


class FactorEngine:
    """Computes various ranking factors from data"""

    @staticmethod
    def momentum_factor(prices: pd.Series, lookbacks=None) -> Dict[str, float]:
        """Calculate momentum factors for different lookbacks"""
        if lookbacks is None:
            lookbacks = [21, 63, 126, 252]
        factors = {}

        for lookback in lookbacks:
            if len(prices) > lookback:
                ret = prices.iloc[-1] / prices.iloc[-lookback] - 1

                # Risk-adjusted momentum
                volatility = prices.pct_change().rolling(lookback).std().iloc[-1]
                if volatility > 0 and not np.isnan(volatility):
                    sharpe_ratio = ret / (volatility * np.sqrt(252))
                    factors[f"momentum_{lookback}d"] = float(sharpe_ratio)
                else:
                    factors[f"momentum_{lookback}d"] = float(ret)

        # Add weighted combination
        if len(factors) > 1:
            weights = {21: 0.5, 63: 0.3, 126: 0.15, 252: 0.05}
            weighted_sum = sum(factors.get(f"momentum_{k}d", 0) * w for k, w in weights.items() if f"momentum_{k}d" in factors)
            factors["momentum_weighted"] = weighted_sum

        return factors

    @staticmethod
    def volatility_factor(prices: pd.Series, lookbacks=None) -> Dict[str, float]:
        """Calculate volatility factors"""
        if lookbacks is None:
            lookbacks = [21, 63]
        factors = {}

        returns = prices.pct_change().dropna()

        for lookback in lookbacks:
            if len(returns) > lookback:
                vol = returns.iloc[-lookback:].std() * np.sqrt(252)

                # Volatility percentile rank within history
                if len(returns) > lookback * 2:
                    hist_vol = returns.rolling(lookback).std() * np.sqrt(252)
                    vol_percentile = (hist_vol < vol).sum() / len(hist_vol)
                    factors[f"volatility_{lookback}d_percentile"] = float(vol_percentile)

                factors[f"volatility_{lookback}d"] = float(vol)

        # Volatility regime
        if len(returns) > 63:
            vol_short = returns.iloc[-21:].std() * np.sqrt(252)
            vol_long = returns.iloc[-63:].std() * np.sqrt(252)
            factors["volatility_ratio"] = float(vol_short / vol_long) if vol_long > 0 else 1.0

        return factors

    @staticmethod
    def quality_factor(fundamentals: Dict[str, float]) -> Dict[str, float]:
        """Calculate quality factors from fundamentals"""
        factors = {}

        # Profitability metrics
        if "roe" in fundamentals:
            # Normalize ROE (typical range 0-30%)
            factors["roe_score"] = float(np.clip(fundamentals["roe"] / 0.3, 0, 1))

        if "operating_margin" in fundamentals:
            factors["margin_score"] = float(np.clip(fundamentals["operating_margin"] / 0.4, 0, 1))

        # Leverage metrics (lower is better)
        if "debt_to_equity" in fundamentals:
            dte = fundamentals["debt_to_equity"]
            # Inverse relationship - lower debt is better
            factors["leverage_score"] = float(np.clip(1 - (dte / 200), 0, 1)) if dte > 0 else 1.0

        # Earnings quality
        if "earnings_volatility" in fundamentals:
            # Lower volatility is better
            ev = fundamentals["earnings_volatility"]
            factors["earnings_stability"] = float(np.clip(1 - (ev / 0.5), 0, 1))

        # Composite quality score
        quality_scores = [v for v in factors.values()]
        if quality_scores:
            factors["quality_composite"] = float(np.mean(quality_scores))

        return factors

    @staticmethod
    def valuation_factor(fundamentals: Dict[str, float], market_data: Optional[pd.Series] = None) -> Dict[str, float]:
        """Calculate valuation factors"""
        factors = {}

        # P/E ratio (lower is better, but handle negative)
        if "pe_ratio" in fundamentals:
            pe = fundamentals["pe_ratio"]
            if pe > 0:
                # Invert and normalize (typical PE range 5-30)
                pe_score = np.clip(1 - ((pe - 10) / 30), 0, 1)
                factors["pe_score"] = float(pe_score)

        # P/B ratio
        if "pb_ratio" in fundamentals:
            pb = fundamentals["pb_ratio"]
            if pb > 0:
                pb_score = np.clip(1 - ((pb - 1) / 5), 0, 1)
                factors["pb_score"] = float(pb_score)

        # Forward P/E if available
        if "forward_pe" in fundamentals:
            fpe = fundamentals["forward_pe"]
            if fpe > 0:
                fpe_score = np.clip(1 - ((fpe - 10) / 30), 0, 1)
                factors["forward_pe_score"] = float(fpe_score)

        # PEG ratio (growth-adjusted)
        if "peg_ratio" in fundamentals:
            peg = fundamentals["peg_ratio"]
            if peg > 0:
                peg_score = np.clip(1 - ((peg - 1) / 3), 0, 1)
                factors["peg_score"] = float(peg_score)

        # Composite valuation score
        valuation_scores = [v for v in factors.values()]
        if valuation_scores:
            factors["valuation_composite"] = float(np.mean(valuation_scores))

        return factors

    @staticmethod
    def growth_factor(fundamentals: Dict[str, float]) -> Dict[str, float]:
        """Calculate growth factors"""
        factors = {}

        # Revenue growth
        if "revenue_growth" in fundamentals:
            rg = fundamentals["revenue_growth"]
            factors["revenue_growth_score"] = float(np.clip(rg / 0.3, 0, 1))

        # EPS growth
        if "eps_growth" in fundamentals:
            eg = fundamentals["eps_growth"]
            factors["eps_growth_score"] = float(np.clip(eg / 0.3, 0, 1))

        # Analyst growth estimates
        if "growth_estimate" in fundamentals:
            ge = fundamentals["growth_estimate"]
            factors["analyst_growth_score"] = float(np.clip(ge / 0.25, 0, 1))

        # Historical growth consistency
        if "growth_consistency" in fundamentals:
            factors["growth_consistency"] = fundamentals["growth_consistency"]

        # Composite growth score
        growth_scores = [v for v in factors.values()]
        if growth_scores:
            factors["growth_composite"] = float(np.mean(growth_scores))

        return factors

    @staticmethod
    def sentiment_factor(sentiment_data: Dict[str, float]) -> Dict[str, float]:
        """Calculate sentiment factors"""
        factors = {}

        if "overall_sentiment" in sentiment_data:
            factors["sentiment_score"] = float(
                (sentiment_data["overall_sentiment"] + 1) / 2  # Map -1..1 to 0..1
            )

        if "news_sentiment" in sentiment_data:
            factors["news_sentiment"] = float((sentiment_data["news_sentiment"] + 1) / 2)

        if "analyst_sentiment" in sentiment_data:
            factors["analyst_sentiment"] = float((sentiment_data["analyst_sentiment"] + 1) / 2)

        if "social_momentum" in sentiment_data:
            factors["social_momentum"] = float(np.clip(sentiment_data["social_momentum"] / 100, 0, 1))

        # Sentiment trend
        if "sentiment_change" in sentiment_data:
            factors["sentiment_trend"] = float(np.clip((sentiment_data["sentiment_change"] + 0.5) / 1, 0, 1))

        return factors

    @staticmethod
    def liquidity_factor(volume_data: pd.Series, market_cap: Optional[float] = None) -> Dict[str, float]:
        """Calculate liquidity factors"""
        factors = {}

        if len(volume_data) > 20:
            # Average daily volume
            avg_volume = volume_data.iloc[-20:].mean()
            factors["avg_daily_volume"] = float(avg_volume)

            # Volume trend
            vol_trend = volume_data.iloc[-5:].mean() / volume_data.iloc[-20:-5].mean()
            factors["volume_trend"] = float(np.clip(vol_trend / 2, 0, 1))

            # Volume percentile
            if len(volume_data) > 63:
                hist_vol = volume_data.rolling(20).mean()
                vol_percentile = (hist_vol < avg_volume).sum() / len(hist_vol)
                factors["volume_percentile"] = float(vol_percentile)

        if market_cap:
            # Log market cap for normalization
            factors["market_cap_log"] = float(np.log(market_cap))

        return factors
