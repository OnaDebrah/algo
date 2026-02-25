import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ....strategies.ml.stock_selection.factor_engine import FactorEngine

logger = logging.getLogger(__name__)


class RankingMethod(Enum):
    """Available ranking methods"""

    ENSEMBLE = "ensemble"
    WEIGHTED = "weighted"
    ML_RANKING = "ml_ranking"
    FACTOR_BASED = "factor_based"
    QUANTILE = "quantile"


@dataclass
class RankingConfig:
    """Configuration for ranking pipeline"""

    method: RankingMethod = RankingMethod.ENSEMBLE
    top_percentile: float = 0.2
    min_stocks: int = 10
    max_stocks: int = 50
    use_factor_neutralization: bool = True
    use_winsorization: bool = True
    winsorization_limits: Tuple[float, float] = (0.01, 0.99)
    score_combination_method: str = "average"  # average, weighted, min, max
    factor_weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class RankedStock:
    """Ranked stock with all metadata"""

    symbol: str
    rank: int
    composite_score: float
    factor_scores: Dict[str, float]
    factor_contributions: Dict[str, float]
    percentile: float
    expected_return: Optional[float] = None
    confidence: float = 0.5
    sector: Optional[str] = None
    market_cap: Optional[float] = None
    volume: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "symbol": self.symbol,
            "rank": self.rank,
            "composite_score": self.composite_score,
            "factor_scores": self.factor_scores,
            "factor_contributions": self.factor_contributions,
            "percentile": self.percentile,
            "expected_return": self.expected_return,
            "confidence": self.confidence,
            "sector": self.sector,
        }


class RankingPipeline:
    """
    Flexible ranking pipeline for stocks
    Combines multiple factors with customizable weighting and neutralization
    """

    def __init__(self, config: Optional[RankingConfig] = None, factor_engine: Optional[FactorEngine] = None):
        self.config = config or RankingConfig()
        self.factor_engine = factor_engine or FactorEngine()
        self.ranked_stocks: List[RankedStock] = []
        self.factor_correlation_matrix: Optional[pd.DataFrame] = None

    async def rank_stocks(
        self, stocks_data: Dict[str, Dict[str, Any]], market_data: Optional[pd.DataFrame] = None, sector_neutral: bool = True
    ) -> List[RankedStock]:
        """
        Rank stocks based on multiple factors

        Args:
            stocks_data: Dictionary with symbol -> {prices, fundamentals, sentiment, volume, market_cap}
            market_data: Overall market data for context
            sector_neutral: Whether to neutralize sector bias

        Returns:
            List of RankedStock objects sorted by composite score
        """
        # Step 1: Calculate factor scores for each stock
        factor_scores = {}
        for symbol, data in stocks_data.items():
            scores = await self._calculate_all_factors(symbol, data)
            factor_scores[symbol] = scores

        # Step 2: Neutralize factors if requested
        if sector_neutral and self.config.use_factor_neutralization:
            factor_scores = self._neutralize_factors(factor_scores, stocks_data)

        # Step 3: Winsorize outliers
        if self.config.use_winsorization:
            factor_scores = self._winsorize_scores(factor_scores, self.config.winsorization_limits)

        # Step 4: Calculate composite scores
        composite_scores = self._calculate_composite_scores(factor_scores)

        # Step 5: Sort and create ranked objects
        ranked = self._create_rankings(composite_scores, factor_scores, stocks_data)

        self.ranked_stocks = ranked
        return ranked

    async def _calculate_all_factors(self, symbol: str, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate all factors for a single stock"""
        scores = {}

        # Extract data components
        prices = data.get("prices", pd.Series())
        fundamentals = data.get("fundamentals", {})
        sentiment = data.get("sentiment", {})
        volume = data.get("volume", pd.Series())
        market_cap = data.get("market_cap")

        # Calculate each factor group
        if not prices.empty:
            scores.update(self.factor_engine.momentum_factor(prices))
            scores.update(self.factor_engine.volatility_factor(prices))

        if fundamentals:
            scores.update(self.factor_engine.quality_factor(fundamentals))
            scores.update(self.factor_engine.valuation_factor(fundamentals))
            scores.update(self.factor_engine.growth_factor(fundamentals))

        if sentiment:
            scores.update(self.factor_engine.sentiment_factor(sentiment))

        if not volume.empty:
            scores.update(self.factor_engine.liquidity_factor(volume, market_cap))

        # Add metadata
        scores["_sector"] = data.get("sector", "Unknown")
        scores["_market_cap"] = market_cap or 0

        return scores

    def _neutralize_factors(self, factor_scores: Dict[str, Dict[str, float]], stocks_data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Neutralize factors by sector
        Subtract sector mean from each factor
        """
        # Group by sector
        sector_groups = {}
        for symbol, scores in factor_scores.items():
            sector = scores.get("_sector", "Unknown")
            if sector not in sector_groups:
                sector_groups[sector] = []
            sector_groups[sector].append(symbol)

        # Calculate sector means for each factor
        factor_names = set()
        for scores in factor_scores.values():
            factor_names.update([k for k in scores.keys() if not k.startswith("_")])

        sector_means = {}
        for sector, symbols in sector_groups.items():
            sector_means[sector] = {}
            for factor in factor_names:
                values = [factor_scores[s][factor] for s in symbols if factor in factor_scores[s]]
                if values:
                    sector_means[sector][factor] = np.mean(values)

        # Neutralize
        neutralized = {}
        for symbol, scores in factor_scores.items():
            sector = scores.get("_sector", "Unknown")
            neutralized[symbol] = {}

            for factor, value in scores.items():
                if factor.startswith("_"):
                    neutralized[symbol][factor] = value
                elif factor in sector_means.get(sector, {}):
                    neutralized[symbol][factor] = value - sector_means[sector][factor]
                else:
                    neutralized[symbol][factor] = value

        return neutralized

    def _winsorize_scores(self, factor_scores: Dict[str, Dict[str, float]], limits: Tuple[float, float]) -> Dict[str, Dict[str, float]]:
        """
        Winsorize outliers to limits
        """
        # Identify all factor values
        factor_values = {}
        for scores in factor_scores.values():
            for factor, value in scores.items():
                if not factor.startswith("_"):
                    if factor not in factor_values:
                        factor_values[factor] = []
                    factor_values[factor].append(value)

        # Calculate percentiles
        percentiles = {}
        for factor, values in factor_values.items():
            percentiles[factor] = (np.percentile(values, limits[0] * 100), np.percentile(values, limits[1] * 100))

        # Winsorize
        winsorized = {}
        for symbol, scores in factor_scores.items():
            winsorized[symbol] = {}
            for factor, value in scores.items():
                if factor.startswith("_"):
                    winsorized[symbol][factor] = value
                elif factor in percentiles:
                    low, high = percentiles[factor]
                    winsorized[symbol][factor] = float(np.clip(value, low, high))
                else:
                    winsorized[symbol][factor] = value

        return winsorized

    def _calculate_composite_scores(self, factor_scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate composite scores from individual factors
        """
        composite_scores = {}

        for symbol, scores in factor_scores.items():
            # Get all non-metadata factors
            factor_values = [v for k, v in scores.items() if not k.startswith("_") and not np.isnan(v)]

            if not factor_values:
                composite_scores[symbol] = 0
                continue

            if self.config.score_combination_method == "average":
                score = np.mean(factor_values)
            elif self.config.score_combination_method == "weighted":
                # Use config weights if provided
                weights = self.config.factor_weights
                weighted_sum = 0
                total_weight = 0

                for factor, value in scores.items():
                    if not factor.startswith("_") and factor in weights:
                        weighted_sum += value * weights[factor]
                        total_weight += weights[factor]

                score = weighted_sum / total_weight if total_weight > 0 else np.mean(factor_values)
            elif self.config.score_combination_method == "min":
                score = np.min(factor_values)
            elif self.config.score_combination_method == "max":
                score = np.max(factor_values)
            else:
                score = np.mean(factor_values)

            composite_scores[symbol] = float(score)

        return composite_scores

    def _create_rankings(
        self, composite_scores: Dict[str, float], factor_scores: Dict[str, Dict[str, float]], stocks_data: Dict[str, Dict[str, Any]]
    ) -> List[RankedStock]:
        """
        Create ranked stock objects
        """
        # Sort by composite score
        sorted_symbols = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)

        # Calculate percentiles
        n_stocks = len(sorted_symbols)
        ranked_stocks = []

        for rank, (symbol, score) in enumerate(sorted_symbols):
            # Limit number of stocks
            if rank >= self.config.max_stocks:
                break

            # Calculate percentile
            percentile = (n_stocks - rank) / n_stocks

            # Get factor scores
            scores = factor_scores.get(symbol, {})

            # Calculate factor contributions
            total = sum(abs(v) for k, v in scores.items() if not k.startswith("_"))
            contributions = {}
            if total > 0:
                contributions = {k: abs(v) / total for k, v in scores.items() if not k.startswith("_")}

            # Create ranked stock
            ranked = RankedStock(
                symbol=symbol,
                rank=rank + 1,
                composite_score=score,
                factor_scores={k: v for k, v in scores.items() if not k.startswith("_")},
                factor_contributions=contributions,
                percentile=percentile,
                expected_return=stocks_data.get(symbol, {}).get("expected_return"),
                confidence=min(percentile * 1.2, 0.95),  # Higher rank = higher confidence
                sector=scores.get("_sector"),
                market_cap=scores.get("_market_cap"),
                volume=scores.get("avg_daily_volume"),
            )

            ranked_stocks.append(ranked)

        return ranked_stocks

    def get_top_stocks(self, n: Optional[int] = None, percentile: Optional[float] = None, min_confidence: float = 0.0) -> List[RankedStock]:
        """
        Get top-ranked stocks
        """
        if not self.ranked_stocks:
            return []

        # Filter by confidence
        filtered = [s for s in self.ranked_stocks if s.confidence >= min_confidence]

        if n:
            return filtered[:n]
        elif percentile:
            cutoff = int(len(filtered) * percentile)
            return filtered[:cutoff]
        else:
            return filtered

    def get_factor_correlation(self) -> pd.DataFrame:
        """
        Calculate correlation between factors
        """
        if not self.ranked_stocks:
            return pd.DataFrame()

        # Build factor matrix
        factor_names = set()
        for stock in self.ranked_stocks:
            factor_names.update(stock.factor_scores.keys())

        factor_names = sorted(factor_names)
        matrix = []

        for stock in self.ranked_stocks:
            row = [stock.factor_scores.get(f, np.nan) for f in factor_names]
            matrix.append(row)

        df = pd.DataFrame(matrix, columns=factor_names)
        self.factor_correlation_matrix = df.corr()

        return self.factor_correlation_matrix

    def get_ranking_summary(self) -> Dict:
        """
        Get summary of current rankings
        """
        if not self.ranked_stocks:
            return {}

        top_5 = self.ranked_stocks[:5]

        return {
            "total_ranked": len(self.ranked_stocks),
            "top_performer": {
                "symbol": top_5[0].symbol,
                "score": top_5[0].composite_score,
                "top_factors": sorted(top_5[0].factor_contributions.items(), key=lambda x: x[1], reverse=True)[:3],
            },
            "score_distribution": {
                "mean": np.mean([s.composite_score for s in self.ranked_stocks]),
                "std": np.std([s.composite_score for s in self.ranked_stocks]),
                "min": min(s.composite_score for s in self.ranked_stocks),
                "max": max(s.composite_score for s in self.ranked_stocks),
            },
            "sector_breakdown": self._get_sector_breakdown(),
        }

    def _get_sector_breakdown(self) -> Dict[str, int]:
        """Get count of stocks by sector in top rankings"""
        breakdown = {}
        for stock in self.ranked_stocks[:50]:  # Top 50
            if stock.sector:
                breakdown[stock.sector] = breakdown.get(stock.sector, 0) + 1
        return breakdown
