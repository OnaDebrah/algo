"""
Strategy Catalog and Categories
Organize all trading strategies by type - Expanded Version
"""

import logging
from typing import Dict, List, Optional

import pandas as pd

from ...strategies import (
    BaseStrategy,
)
from ...strategies.catelog.adaptive import Adaptive
from ...strategies.catelog.category import StrategyCategory
from ...strategies.catelog.deep_learning import DeepLearning
from ...strategies.catelog.mean_reversion import MeanReversion
from ...strategies.catelog.ml import ML
from ...strategies.catelog.momentum import Momentum
from ...strategies.catelog.pairs_trading import PairsTrading
from ...strategies.catelog.statistical_arbitrage import StatisticalArbitrage
from ...strategies.catelog.strategy_info import StrategyInfo
from ...strategies.catelog.trend_following import TrendFollowing
from ...strategies.catelog.volatility import Volatility

logger = logging.getLogger(__name__)


class StrategyCatalog:
    """
    Catalog of all available trading strategies
    Aggregates strategies from all individual category catalogs
    """

    def __init__(self):
        self.category_catalogs = {
            "trend_following": TrendFollowing(),
            "momentum": Momentum(),
            "mean_reversion": MeanReversion(),
            "volatility": Volatility(),
            "statistical_arbitrage": StatisticalArbitrage(),
            "pairs_trading": PairsTrading(),
            "adaptive": Adaptive(),
            "machine_learning": ML(),
            "deep_learning": DeepLearning(),
        }

        self.strategies = self._build_catalog()

    def _build_catalog(self) -> Dict[str, StrategyInfo]:
        """
        Build the complete strategy catalog by merging all individual catalogs

        Returns:
            Dictionary mapping strategy IDs to StrategyInfo objects
        """
        combined_catalog = {}

        # Merge all catalogs
        for catalog_name, catalog_instance in self.category_catalogs.items():
            if hasattr(catalog_instance, "strategies"):
                combined_catalog.update(catalog_instance.strategies)
                logger.info(f"✅ Loaded {len(catalog_instance.strategies)} strategies from {catalog_name}")
            else:
                logger.critical(f"⚠️  Catalog {catalog_name} has no 'strategies' attribute")

        logger.info(f"Total strategies loaded: {len(combined_catalog)}")

        return combined_catalog

    def get_by_mode(self, mode: str) -> Dict[str, StrategyInfo]:
        """Get strategies compatible with a backtest mode ('single' or 'multi')"""
        return {key: info for key, info in self.strategies.items() if info.backtest_mode == mode or info.backtest_mode == "both"}

    def get_by_category(self, category: StrategyCategory) -> Dict[str, StrategyInfo]:
        """Get all strategies in a category"""
        return {key: info for key, info in self.strategies.items() if info.category == category}

    def get_by_complexity(self, complexity: str) -> Dict[str, StrategyInfo]:
        """Get strategies by complexity level"""
        return {key: info for key, info in self.strategies.items() if info.complexity == complexity}

    def get_categories(self) -> List[StrategyCategory]:
        """Get list of all categories"""
        return list(set(info.category for info in self.strategies.values()))

    def get_strategy_names(self) -> List[str]:
        """Get list of all strategy display names"""
        return [info.name for info in self.strategies.values()]

    def get_info(self, strategy_key: str) -> StrategyInfo:
        """Get information about a specific strategy"""
        return self.strategies.get(strategy_key)

    def create_strategy(self, strategy_key: str, **kwargs) -> BaseStrategy:
        """
        Create a strategy instance

        Args:
            strategy_key: Strategy identifier
            **kwargs: Strategy parameters

        Returns:
            Instantiated strategy
        """
        info = self.strategies.get(strategy_key)
        if not info:
            raise ValueError(f"Unknown strategy: {strategy_key}")

        # Use defaults for missing parameters and sanitize types
        params = {}
        for param_name, param_info in info.parameters.items():
            val = kwargs.get(param_name, param_info.get("default"))

            # Robust type conversion: if default is int, ensure val is int
            # This handles cases like Bayesian optimization returning floats for windows
            default_val = param_info.get("default")
            if isinstance(default_val, int) and not isinstance(val, int) and val is not None:
                try:
                    # Capture float strings or direct floats
                    val = int(float(val))
                except (ValueError, TypeError):
                    logger.warning(f"Failed to cast parameter {param_name} to int: {val}")

            params[param_name] = val

        return info.class_type(**params)

    def format_for_ui(self) -> Dict[str, List[Dict]]:
        """Format strategies for UI display, grouped by category"""

        result = {}

        for category in self.get_categories():
            strategies = self.get_by_category(category)

            result[category.value] = [
                {
                    "key": key,
                    "name": info.name,
                    "description": info.description,
                    "complexity": info.complexity,
                    "time_horizon": info.time_horizon,
                    "best_for": info.best_for,
                }
                for key, info in strategies.items()
            ]

        return result

    def get_comparison_matrix(self) -> Dict:
        """Get comparison matrix of all strategies"""

        matrix = []

        for key, info in self.strategies.items():
            matrix.append(
                {
                    "Strategy": info.name,
                    "Category": info.category.value,
                    "Complexity": info.complexity,
                    "Time Horizon": info.time_horizon,
                    "Best For": ", ".join(info.best_for[:2]),
                    "Pros": len(info.pros),
                    "Cons": len(info.cons),
                }
            )

        return matrix

    def get_strategy_count_by_category(self) -> Dict[str, int]:
        """Get count of strategies in each category"""
        counts = {}
        for category in self.get_categories():
            counts[category.value] = len(self.get_by_category(category))
        return counts

    def search_strategies(self, query: str) -> List[Dict]:
        """Search strategies by name, description, or tags"""
        results = []
        query_lower = query.lower()

        for key, info in self.strategies.items():
            if (
                query_lower in info.name.lower()
                or query_lower in info.description.lower()
                or any(query_lower in tag.lower() for tag in info.best_for)
            ):
                results.append({"key": key, "name": info.name, "category": info.category.value, "description": info.description})

        return results


# Global catalog instance
strategy_catalog = StrategyCatalog()


def get_catalog() -> StrategyCatalog:
    """Get the global strategy catalog"""
    return strategy_catalog


def get_strategies_by_category() -> Dict[str, List[str]]:
    """Get strategies organized by category for UI"""
    catalog = get_catalog()

    result = {}
    for category in catalog.get_categories():
        strategies = catalog.get_by_category(category)
        result[category.value] = [info.name for info in strategies.values()]

    return result


def get_strategy_description(strategy_name: str) -> str:
    """Get strategy description by display name"""
    catalog = get_catalog()

    for info in catalog.strategies.values():
        if info.name == strategy_name:
            return info.description

    return "No description available"


def get_recommended_strategies(level: str = "Beginner") -> List[str]:
    """Get recommended strategies for a skill level"""
    catalog = get_catalog()
    strategies = catalog.get_by_complexity(level)

    return [info.name for info in strategies.values()]


def get_strategy_summary() -> Dict:
    """Get summary statistics of the strategy catalog"""
    catalog = get_catalog()

    return {
        "total_strategies": len(catalog.strategies),
        "by_category": catalog.get_strategy_count_by_category(),
        "by_complexity": {
            "Beginner": len(catalog.get_by_complexity("Beginner")),
            "Intermediate": len(catalog.get_by_complexity("Intermediate")),
            "Advanced": len(catalog.get_by_complexity("Advanced")),
        },
        "categories": [cat.value for cat in catalog.get_categories()],
    }


def get_strategy(self, strategy_id: str) -> Optional[StrategyInfo]:
    """Get strategy info by ID"""
    return self.strategies.get(strategy_id)


def get_strategy_by_name(self, name: str) -> Optional[StrategyInfo]:
    """Get strategy info by name (case-insensitive partial match)"""
    name_lower = name.lower()
    for strategy_id, info in self.strategies.items():
        if name_lower in info.name.lower():
            return info
    return None

    # ========== LISTING METHODS ==========


def list_strategies(
    self,
    category: Optional[StrategyCategory] = None,
    complexity: Optional[str] = None,
    tags: Optional[List[str]] = None,
    requires_ml: Optional[bool] = None,
    time_horizon: Optional[str] = None,
    search_term: Optional[str] = None,
) -> List[Dict]:
    """
    List strategies with optional filters

    Args:
        category: Filter by strategy category
        complexity: Filter by complexity level
        tags: Filter by tags (must have all specified tags)
        requires_ml: Filter by ML requirement
        time_horizon: Filter by time horizon
        search_term: Search in name and description

    Returns:
        List of strategy summaries
    """
    results = []

    for strategy_id, info in self.strategies.items():
        # Apply filters
        if category and info.category != category:
            continue
        if complexity and info.complexity != complexity:
            continue
        if requires_ml is not None and info.requires_ml_training != requires_ml:
            continue
        if time_horizon and info.time_horizon != time_horizon:
            continue
        if tags and not all(tag in info.tags for tag in tags):
            continue
        if search_term:
            search_term_lower = search_term.lower()
            if search_term_lower not in info.name.lower() and search_term_lower not in info.description.lower():
                continue

        results.append(
            {
                "id": strategy_id,
                "name": info.name,
                "category": info.category.value,
                "complexity": info.complexity,
                "description": info.description[:100] + "..." if len(info.description) > 100 else info.description,
                "time_horizon": info.time_horizon,
                "requires_ml": info.requires_ml_training,
                "min_data_days": info.min_data_days,
                "tags": info.tags[:3],  # Show top 3 tags
                "best_for": info.best_for[:2],  # Show top 2 use cases
            }
        )

    return results


def list_all_strategies_simple(self) -> List[Dict]:
    """Simple list of all strategies without filters"""
    return [{"id": sid, "name": info.name, "category": info.category.value, "complexity": info.complexity} for sid, info in self.strategies.items()]

    # ========== DETAILED INFO METHODS ==========


def get_strategy_details(self, strategy_id: str) -> Optional[Dict]:
    """Get detailed information about a strategy"""
    info = self.get_strategy(strategy_id)
    if not info:
        return None

    return {
        "id": strategy_id,
        "name": info.name,
        "category": info.category.value,
        "description": info.description.strip(),
        "complexity": info.complexity,
        "time_horizon": info.time_horizon,
        "best_for": info.best_for,
        "parameters": info.parameters,
        "pros": info.pros,
        "cons": info.cons,
        "backtest_mode": info.backtest_mode,
        "tags": info.tags,
        "requires_ml_training": info.requires_ml_training,
        "min_data_days": info.min_data_days,
    }


def get_strategies_by_complexity(self) -> Dict[str, List[Dict]]:
    """Group strategies by complexity level"""
    grouped = {"Beginner": [], "Intermediate": [], "Advanced": [], "Expert": []}

    for strategy_id, info in self.strategies.items():
        if info.complexity in grouped:
            grouped[info.complexity].append(
                {"id": strategy_id, "name": info.name, "category": info.category.value, "description": info.description[:80] + "..."}
            )

    # Remove empty groups
    return {k: v for k, v in grouped.items() if v}


def get_strategies_by_tag(self, tag: str) -> List[Dict]:
    """Get all strategies with a specific tag"""
    results = []
    for strategy_id, info in self.strategies.items():
        if tag in info.tags:
            results.append({"id": strategy_id, "name": info.name, "category": info.category.value, "complexity": info.complexity, "tags": info.tags})
    return results

    # ========== COMPARISON METHODS ==========


def compare_strategies(self, strategy_ids: List[str]) -> pd.DataFrame:
    """Compare multiple strategies side by side"""

    comparison = []
    for sid in strategy_ids:
        info = self.get_strategy(sid)
        if info:
            comparison.append(
                {
                    "Strategy": info.name,
                    "Category": info.category.value,
                    "Complexity": info.complexity,
                    "Time Horizon": info.time_horizon,
                    "ML Required": "Yes" if info.requires_ml_training else "No",
                    "Min Data Days": info.min_data_days,
                    "Tags": ", ".join(info.tags[:3]),
                    "Best For": ", ".join(info.best_for[:2]),
                }
            )

    return pd.DataFrame(comparison)

    # ========== SEARCH AND SUGGESTION METHODS ==========


def search_strategies(self, query: str) -> List[Dict]:
    """Search strategies by keyword in name, description, or tags"""
    query_lower = query.lower()
    results = []

    for strategy_id, info in self.strategies.items():
        # Search in name
        if query_lower in info.name.lower():
            score = 10
        # Search in description
        elif query_lower in info.description.lower():
            score = 5
        # Search in tags
        elif any(query_lower in tag.lower() for tag in info.tags):
            score = 3
        # Search in best_for
        elif any(query_lower in bf.lower() for bf in info.best_for):
            score = 2
        else:
            continue

        results.append(
            {
                "id": strategy_id,
                "name": info.name,
                "category": info.category.value,
                "complexity": info.complexity,
                "score": score,
                "description": info.description[:100] + "...",
            }
        )

    # Sort by relevance score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def suggest_strategy(self, market_regime: str, risk_tolerance: str, experience: str, asset_class: Optional[str] = None) -> List[Dict]:
    """
    Suggest strategies based on market conditions and user preferences

    Args:
        market_regime: "trending", "ranging", "volatile", "unknown"
        risk_tolerance: "low", "medium", "high"
        experience: "beginner", "intermediate", "advanced", "expert"
        asset_class: Optional filter for specific asset class

    Returns:
        List of suggested strategies with scores
    """
    suggestions = []

    for strategy_id, info in self.strategies.items():
        score = 0

        # Score based on market regime
        if market_regime == "trending":
            if info.category in [StrategyCategory.TREND_FOLLOWING, StrategyCategory.MOMENTUM]:
                score += 3
        elif market_regime == "ranging":
            if info.category in [StrategyCategory.MEAN_REVERSION, StrategyCategory.STATISTICAL_ARBITRAGE]:
                score += 3
        elif market_regime == "volatile":
            if info.category in [StrategyCategory.VOLATILITY, StrategyCategory.RISK_MANAGEMENT]:
                score += 3
            # Deep learning strategies good for volatility
            if "volatility" in info.tags:
                score += 2

        # Score based on risk tolerance
        if risk_tolerance == "low":
            if "risk-management" in info.tags:
                score += 3
            # Conservative position sizing
            if hasattr(info, "max_position") and info.max_position < 1.0:
                score += 1
        elif risk_tolerance == "high":
            if "adaptive" in info.tags or "aggressive" in info.tags:
                score += 2
            if hasattr(info, "max_position") and info.max_position > 1.0:
                score += 2

        # Score based on experience
        complexity_scores = {"beginner": 3, "intermediate": 2, "advanced": 1, "expert": 0}

        complexity_map = {"Beginner": 3, "Intermediate": 2, "Advanced": 1, "Expert": 0}

        if info.complexity in complexity_map:
            exp_level = complexity_scores.get(experience, 0)
            strategy_level = complexity_map[info.complexity]
            if exp_level >= strategy_level:
                score += 2

        # Bonus for recent research
        if "2025" in info.description or "2024" in info.description:
            score += 1

        if score > 0:
            suggestions.append(
                {
                    "id": strategy_id,
                    "name": info.name,
                    "category": info.category.value,
                    "complexity": info.complexity,
                    "score": score,
                    "description": info.description[:100] + "...",
                    "requires_ml": info.requires_ml_training,
                }
            )

    # Sort by score
    suggestions.sort(key=lambda x: x["score"], reverse=True)
    return suggestions[:10]  # Return top 10

    # ========== STATISTICS AND METADATA ==========


def get_catalog_stats(self) -> Dict:
    """Get statistics about the catalog"""
    total = len(self.strategies)
    by_category = {}
    by_complexity = {}
    ml_count = 0

    for info in self.strategies.values():
        cat = info.category.value
        by_category[cat] = by_category.get(cat, 0) + 1

        comp = info.complexity
        by_complexity[comp] = by_complexity.get(comp, 0) + 1

        if info.requires_ml_training:
            ml_count += 1

    return {
        "total_strategies": total,
        "strategies_by_category": by_category,
        "strategies_by_complexity": by_complexity,
        "ml_strategies": ml_count,
        "traditional_strategies": total - ml_count,
        "categories_count": len(by_category),
    }


def get_popular_tags(self, limit: int = 10) -> List[Dict]:
    """Get most common tags across strategies"""
    tag_counts = {}
    for info in self.strategies.values():
        for tag in info.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
    return [{"tag": tag, "count": count} for tag, count in sorted_tags[:limit]]

    # ========== UTILITY METHODS ==========


def validate_strategy_id(self, strategy_id: str) -> bool:
    """Check if a strategy ID exists"""
    return strategy_id in self.strategies


def get_strategy_class(self, strategy_id: str) -> Optional[type]:
    """Get the strategy class for instantiation"""
    info = self.get_strategy(strategy_id)
    return info.class_type if info else None


def get_parameter_info(self, strategy_id: str) -> Optional[Dict]:
    """Get parameter information for a strategy"""
    info = self.get_strategy(strategy_id)
    return info.parameters if info else None


def refresh_catalog(self):
    """Rebuild the catalog (useful if strategies are added dynamically)"""
    self.strategies = self._build_catalog()
    logger.info(f"Catalog refreshed: {len(self.strategies)} strategies loaded")
