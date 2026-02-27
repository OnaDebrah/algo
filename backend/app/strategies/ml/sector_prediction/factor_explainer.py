import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


@dataclass
class FactorExplanation:
    """Comprehensive factor explanation for a prediction"""

    symbol: str
    prediction: float
    confidence: float
    top_factors: List[Dict[str, Any]]
    all_factors: List[Dict[str, Any]]
    interaction_effects: List[Dict[str, Any]]
    factor_correlations: Dict[str, float]
    base_value: float
    explanation_method: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "symbol": self.symbol,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "top_factors": self.top_factors,
            "interaction_effects": self.interaction_effects,
            "base_value": self.base_value,
            "explanation_method": self.explanation_method,
            "timestamp": self.timestamp.isoformat(),
        }


class FactorExplainer:
    """
    Factor explanation engine using SHAP and multiple methods
    """

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        model_type: str = "auto",
        background_data: Optional[np.ndarray] = None,
        use_interactions: bool = True,
        cache_explanations: bool = True,
        n_background_samples: int = 100,
    ):
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self.use_interactions = use_interactions
        self.cache_explanations = cache_explanations
        self.n_background_samples = n_background_samples

        # Caches
        self.explanation_cache = {}  # symbol -> FactorExplanation
        self.shap_values_cache = {}  # feature_hash -> shap_values

        self.explainer, self.background_data = self._initialize_explainer(background_data)

        self.feature_metadata = self._initialize_feature_metadata()

    def _initialize_explainer(self, background_data: Optional[np.ndarray]) -> Tuple[Optional[shap.Explainer], Optional[np.ndarray]]:
        """Initialize appropriate SHAP explainer using current API"""
        try:
            if self.model_type == "auto":
                # Auto-detect model type
                if hasattr(self.model, "estimators_") or "sklearn.ensemble" in str(type(self.model)):
                    # Tree-based model
                    self.model_type = "tree"
                    logger.info("Using TreeExplainer for tree-based model")

                    # For tree models, we don't need background data
                    explainer = shap.TreeExplainer(self.model)
                    return explainer, None

                elif hasattr(self.model, "coef_") or "sklearn.linear_model" in str(type(self.model)):
                    # Linear model
                    self.model_type = "linear"
                    logger.info("Using LinearExplainer for linear model")

                    if background_data is not None:
                        # Use subset for efficiency
                        if len(background_data) > self.n_background_samples:
                            background_data = background_data[np.random.choice(len(background_data), self.n_background_samples, replace=False)]
                        explainer = shap.LinearExplainer(self.model, background_data)
                        return explainer, background_data
                    else:
                        explainer = shap.Explainer(self.model.predict, shap.sample(background_data) if background_data is not None else None)
                        return explainer, background_data
                else:
                    # Fallback to KernelExplainer for black-box models
                    self.model_type = "kernel"
                    logger.info("Using KernelExplainer for black-box model")

                    if background_data is not None:
                        # Use subset for efficiency
                        if len(background_data) > self.n_background_samples:
                            background_data = background_data[np.random.choice(len(background_data), self.n_background_samples, replace=False)]
                        explainer = shap.KernelExplainer(self.model.predict, background_data)
                        return explainer, background_data

            elif self.model_type == "tree" and (hasattr(self.model, "estimators_") or hasattr(self.model, "feature_importances_")):
                explainer = shap.TreeExplainer(self.model)
                return explainer, None

            elif self.model_type == "linear" and hasattr(self.model, "coef_"):
                if background_data is not None:
                    if len(background_data) > self.n_background_samples:
                        background_data = background_data[np.random.choice(len(background_data), self.n_background_samples, replace=False)]
                    explainer = shap.LinearExplainer(self.model, background_data)
                    return explainer, background_data

            elif self.model_type == "kernel" and background_data is not None:
                if len(background_data) > self.n_background_samples:
                    background_data = background_data[np.random.choice(len(background_data), self.n_background_samples, replace=False)]
                explainer = shap.KernelExplainer(self.model.predict, background_data)
                return explainer, background_data

        except Exception as e:
            logger.error(f"Error initializing SHAP explainer: {e}")

        return None, background_data

    def _initialize_feature_metadata(self) -> Dict[str, Dict]:
        """Initialize feature metadata including categories and typical ranges"""
        metadata = {}

        # Define feature categories
        for feature in self.feature_names:
            feature_lower = feature.lower()

            if any(x in feature_lower for x in ["momentum", "trend", "ma_", "moving_avg"]):
                category = "momentum"
            elif any(x in feature_lower for x in ["volatility", "vol_", "vix", "std"]):
                category = "volatility"
            elif any(x in feature_lower for x in ["pe", "pb", "ps", "valuation", "price_to"]):
                category = "valuation"
            elif any(x in feature_lower for x in ["growth", "revenue", "eps", "earnings"]):
                category = "growth"
            elif any(x in feature_lower for x in ["roe", "roa", "margin", "quality", "profit"]):
                category = "quality"
            elif any(x in feature_lower for x in ["sentiment", "news", "analyst", "social"]):
                category = "sentiment"
            elif any(x in feature_lower for x in ["volume", "liquidity", "market_cap", "turnover"]):
                category = "liquidity"
            else:
                category = "other"

            metadata[feature] = {
                "category": category,
                "typical_range": self._get_typical_range(feature),
                "description": self._get_feature_description(feature),
            }

        return metadata

    def _get_typical_range(self, feature: str) -> Tuple[float, float]:
        """Get typical range for a feature"""
        feature_lower = feature.lower()

        if "pe_ratio" in feature_lower:
            return (5, 30)
        elif "growth" in feature_lower:
            return (-0.2, 0.3)
        elif "margin" in feature_lower:
            return (0, 0.4)
        elif "roe" in feature_lower:
            return (0, 0.3)
        elif "volume" in feature_lower:
            return (1e6, 1e8)
        elif "volatility" in feature_lower:
            return (0.1, 0.4)
        else:
            return (-1, 1)

    def _get_feature_description(self, feature: str) -> str:
        """Get human-readable feature description"""
        descriptions = {
            "pe_ratio": "Price to Earnings ratio - valuation metric",
            "pb_ratio": "Price to Book ratio - valuation metric",
            "ps_ratio": "Price to Sales ratio - valuation metric",
            "roe": "Return on Equity - profitability metric",
            "roa": "Return on Assets - profitability metric",
            "revenue_growth": "Year-over-year revenue growth",
            "eps_growth": "Earnings per share growth",
            "operating_margin": "Operating profit margin",
            "net_margin": "Net profit margin",
            "debt_to_equity": "Leverage ratio",
            "current_ratio": "Liquidity ratio",
            "momentum_weighted": "Multi-period momentum score",
            "volatility_21d": "21-day annualized volatility",
            "sentiment_score": "News and social media sentiment",
            "analyst_rating": "Average analyst recommendation",
            "market_cap": "Market capitalization",
            "avg_daily_volume": "Average daily trading volume",
        }

        # Try exact match
        if feature in descriptions:
            return descriptions[feature]

        # Try partial match
        for key, desc in descriptions.items():
            if key in feature.lower():
                return desc

        return f"Feature: {feature}"

    async def get_top_factors(
        self,
        symbol: str,
        features: np.ndarray,
        feature_values: Optional[pd.Series] = None,
        n_factors: int = 5,
        include_interactions: bool = True,
        use_cache: bool = True,
    ) -> List[Dict]:
        """
        Get top contributing factors with full explanations

        Args:
            symbol: Stock symbol
            features: Feature array for this prediction
            feature_values: Actual feature values (for context)
            n_factors: Number of top factors to return
            include_interactions: Whether to include interaction effects
            use_cache: Whether to use cached explanations

        Returns:
            List of factor dictionaries with importance, impact, and context
        """
        # Check cache
        if use_cache and symbol in self.explanation_cache:
            cached = self.explanation_cache[symbol]
            return cached.top_factors[:n_factors]

        try:
            # Method 1: SHAP values (if available)
            if self.explainer is not None:
                factors, base_value = await self._get_shap_factors(features, feature_values, include_interactions)
            else:
                # Method 2: Fallback to feature importance
                factors, base_value = self._get_importance_factors(features)

            # Method 3: Add interaction effects if requested
            if include_interactions and len(factors) > 1:
                interactions = self._get_interaction_effects(features, factors)
                factors = self._merge_interactions(factors, interactions)

            # Enhance factors with metadata
            factors = self._enhance_factors_with_metadata(factors, feature_values)

            # Get prediction
            prediction = self._get_prediction(features)

            # Calculate confidence
            confidence = self._calculate_confidence(factors)

            # Cache the full explanation
            if self.cache_explanations:
                explanation = FactorExplanation(
                    symbol=symbol,
                    prediction=prediction,
                    confidence=confidence,
                    top_factors=factors[:n_factors],
                    all_factors=factors,
                    interaction_effects=[f for f in factors if f.get("is_interaction", False)],
                    factor_correlations=self._get_factor_correlations(factors),
                    base_value=base_value,
                    explanation_method="shap" if self.explainer else "importance",
                )
                self.explanation_cache[symbol] = explanation

            return factors[:n_factors]

        except Exception as e:
            logger.error(f"Error getting top factors for {symbol}: {e}")
            return self._get_fallback_factors()

    async def _get_shap_factors(
        self, features: np.ndarray, feature_values: Optional[pd.Series], include_interactions: bool
    ) -> Tuple[List[Dict], float]:
        """Get factors using current SHAP API"""

        # Ensure features are 2D
        if features.ndim == 1:
            features_2d = features.reshape(1, -1)
        else:
            features_2d = features

        # Calculate SHAP values using the current API
        shap_values_obj = self.explainer(features_2d)

        # Extract values based on SHAP version and output type
        if hasattr(shap_values_obj, "values"):
            # New SHAP API returns an Explanation object
            shap_values = shap_values_obj.values

            # Get base value
            if hasattr(shap_values_obj, "base_values"):
                if isinstance(shap_values_obj.base_values, (list, np.ndarray)):
                    base_value = float(shap_values_obj.base_values[0]) if len(shap_values_obj.base_values) > 0 else 0
                else:
                    base_value = float(shap_values_obj.base_values)
            else:
                base_value = 0
        else:
            # Older API or different format
            shap_values = shap_values_obj
            base_value = 0
            if hasattr(self.explainer, "expected_value"):
                if isinstance(self.explainer.expected_value, (list, np.ndarray)):
                    base_value = float(self.explainer.expected_value[0])
                else:
                    base_value = float(self.explainer.expected_value)

        # Handle different shapes
        if isinstance(shap_values, list):
            # Multi-output case
            shap_values = shap_values[0]

        if shap_values.ndim > 1:
            if shap_values.shape[0] == 1:
                shap_values = shap_values[0]
            else:
                shap_values = shap_values.mean(axis=0)

        shap_values = np.array(shap_values).flatten()

        # Ensure we have the right length
        if len(shap_values) != len(self.feature_names):
            logger.warning(f"SHAP values length {len(shap_values)} doesn't match feature names {len(self.feature_names)}")
            # Pad or truncate
            if len(shap_values) > len(self.feature_names):
                shap_values = shap_values[: len(self.feature_names)]
            else:
                shap_values = np.pad(shap_values, (0, len(self.feature_names) - len(shap_values)))

        # Create factor list
        factors = []
        total_impact = np.sum(np.abs(shap_values))

        for i, (name, shap_val) in enumerate(zip(self.feature_names, shap_values)):
            # Get actual feature value
            actual_value = None
            if feature_values is not None and name in feature_values.index:
                actual_value = float(feature_values[name])
            elif i < len(features):
                actual_value = float(features[i])

            # Calculate contribution percentage
            contribution_pct = (abs(shap_val) / total_impact * 100) if total_impact > 0 else 0

            factors.append(
                {
                    "factor": name,
                    "shap_value": float(shap_val),
                    "impact": "positive" if shap_val > 0 else "negative",
                    "magnitude": float(abs(shap_val)),
                    "contribution_pct": float(contribution_pct),
                    "actual_value": actual_value,
                    "typical_range": self.feature_metadata.get(name, {}).get("typical_range", (None, None)),
                    "category": self.feature_metadata.get(name, {}).get("category", "other"),
                    "description": self.feature_metadata.get(name, {}).get("description", ""),
                    "is_interaction": False,
                }
            )

        # Sort by magnitude
        factors.sort(key=lambda x: x["magnitude"], reverse=True)

        # Get interaction values if available and requested
        if include_interactions and hasattr(shap_values_obj, "interaction_values"):
            try:
                interaction_values = shap_values_obj.interaction_values
                factors = self._add_interaction_factors(factors, interaction_values, features)
            except Exception as e:
                logger.debug(f"Could not calculate interaction values: {e}")

        return factors, base_value

    def _add_interaction_factors(self, factors: List[Dict], interaction_values: Any, features: np.ndarray) -> List[Dict]:
        """Add interaction effects to factors"""

        # Handle different interaction value formats
        if hasattr(interaction_values, "values"):
            interaction_values = interaction_values.values

        if isinstance(interaction_values, list):
            interaction_values = interaction_values[0]

        if hasattr(interaction_values, "shape") and len(interaction_values.shape) == 3:
            # For tree models, interaction_values[i,j] is interaction between features i and j
            n_features = len(self.feature_names)

            # Get the matrix for the first sample
            if interaction_values.shape[0] > 0:
                inter_matrix = interaction_values[0]

                for i in range(min(n_features, inter_matrix.shape[0])):
                    for j in range(i + 1, min(n_features, inter_matrix.shape[1])):
                        interaction_val = inter_matrix[i, j]

                        if abs(interaction_val) > 0.01:  # Filter small interactions
                            factors.append(
                                {
                                    "factor": f"{self.feature_names[i]} × {self.feature_names[j]}",
                                    "shap_value": float(interaction_val),
                                    "impact": "positive" if interaction_val > 0 else "negative",
                                    "magnitude": float(abs(interaction_val)),
                                    "contribution_pct": 0,  # Will be recalculated
                                    "actual_value": float(features[i] * features[j]) if i < len(features) and j < len(features) else None,
                                    "category": "interaction",
                                    "description": f"Interaction between {self.feature_names[i]} and {self.feature_names[j]}",
                                    "is_interaction": True,
                                    "factors": [self.feature_names[i], self.feature_names[j]],
                                }
                            )

        return factors

    def _get_importance_factors(self, features: np.ndarray) -> Tuple[List[Dict], float]:
        """Fallback method using feature importance"""
        factors = []
        base_value = 0

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_

            for i, (name, imp) in enumerate(zip(self.feature_names, importances)):
                factors.append(
                    {
                        "factor": name,
                        "importance": float(imp),
                        "magnitude": float(imp),
                        "contribution_pct": float(imp * 100),
                        "impact": "positive" if imp > np.mean(importances) else "negative",
                        "actual_value": float(features[i]) if i < len(features) else None,
                        "category": self.feature_metadata.get(name, {}).get("category", "other"),
                        "description": self.feature_metadata.get(name, {}).get("description", ""),
                        "is_interaction": False,
                    }
                )

            factors.sort(key=lambda x: x["magnitude"], reverse=True)
        elif hasattr(self.model, "coef_"):
            # Linear model coefficients
            coefficients = self.model.coef_
            if coefficients.ndim > 1:
                coefficients = coefficients[0]

            for i, (name, coefficient) in enumerate(zip(self.feature_names, coefficients)):
                factors.append(
                    {
                        "factor": name,
                        "coefficient": float(coefficient),
                        "magnitude": float(abs(coefficient)),
                        "contribution_pct": float(abs(coefficient) / np.sum(np.abs(coefficients)) * 100) if np.sum(np.abs(coefficients)) > 0 else 0,
                        "impact": "positive" if coefficient > 0 else "negative",
                        "actual_value": float(features[i]) if i < len(features) else None,
                        "category": self.feature_metadata.get(name, {}).get("category", "other"),
                        "description": self.feature_metadata.get(name, {}).get("description", ""),
                        "is_interaction": False,
                    }
                )

            factors.sort(key=lambda x: x["magnitude"], reverse=True)

        return factors, base_value

    def _get_interaction_effects(self, features: np.ndarray, factors: List[Dict]) -> List[Dict]:
        """Identify important interaction effects between factors"""
        interactions = []

        # Check for correlated factors that might have interaction effects
        for i, factor1 in enumerate(factors[:5]):  # Only top 5
            factor1_name = factor1["factor"]
            factor1_category = factor1.get("category", "other")

            for factor2 in factors[i + 1 : 10]:  # Next 5
                factor2_name = factor2["factor"]
                factor2_category = factor2.get("category", "other")

                # Stronger interaction if same category
                if factor1_category == factor2_category:
                    strength = 0.4
                    description = f"Strong interaction within {factor1_category} category"
                else:
                    strength = 0.2
                    description = f"Cross-category interaction between {factor1_category} and {factor2_category}"

                interactions.append(
                    {
                        "factor": f"{factor1_name} × {factor2_name}",
                        "type": "within_category" if factor1_category == factor2_category else "cross_category",
                        "strength": strength,
                        "description": description,
                        "category": "interaction",
                        "factors": [factor1_name, factor2_name],
                    }
                )

        return interactions

    def _merge_interactions(self, factors: List[Dict], interactions: List[Dict]) -> List[Dict]:
        """Merge interaction effects into factor list"""
        # Add interactions as additional factors
        for interaction in interactions:
            factors.append(
                {
                    "factor": interaction["factor"],
                    "importance": interaction["strength"],
                    "magnitude": interaction["strength"],
                    "contribution_pct": interaction["strength"] * 50,  # Scale for visibility
                    "impact": "positive" if interaction["strength"] > 0.3 else "negative",
                    "category": "interaction",
                    "description": interaction["description"],
                    "is_interaction": True,
                    "factors": interaction.get("factors", []),
                }
            )

        # Re-sort
        factors.sort(key=lambda x: x["magnitude"], reverse=True)

        return factors

    def _enhance_factors_with_metadata(self, factors: List[Dict], feature_values: Optional[pd.Series]) -> List[Dict]:
        """Add contextual information to factors"""

        for factor in factors:
            if factor.get("is_interaction"):
                # Handle interaction factors
                factor_name = factor["factor"]
                factor["interpretation"] = f'Interaction effect: {factor["description"]}'
                factor["action"] = "interaction"
                continue

            factor_name = factor["factor"]

            # Add contextual interpretation based on factor type
            if factor["impact"] == "positive":
                if "pe_ratio" in factor_name.lower():
                    factor["interpretation"] = "Higher P/E ratio than average, suggesting growth expectations"
                elif "growth" in factor_name.lower():
                    factor["interpretation"] = "Strong growth contributing positively to outlook"
                elif "sentiment" in factor_name.lower():
                    factor["interpretation"] = "Positive market sentiment boosting outlook"
                elif "momentum" in factor_name.lower():
                    factor["interpretation"] = "Positive price momentum supporting uptrend"
                elif "margin" in factor_name.lower():
                    factor["interpretation"] = "Strong profitability margins"
                elif "roe" in factor_name.lower():
                    factor["interpretation"] = "High return on equity indicates efficient capital use"
                else:
                    factor["interpretation"] = f"Positive contribution from {factor_name}"
            else:
                if "pe_ratio" in factor_name.lower():
                    factor["interpretation"] = "Lower P/E ratio than average, suggesting value opportunity"
                elif "volatility" in factor_name.lower():
                    factor["interpretation"] = "High volatility creating uncertainty and risk"
                elif "debt" in factor_name.lower():
                    factor["interpretation"] = "Higher leverage increasing financial risk"
                elif "valuation" in factor_name.lower():
                    factor["interpretation"] = "Valuation concerns weighing on outlook"
                else:
                    factor["interpretation"] = f"Negative contribution from {factor_name}"

            # Add recommendation based on contribution percentage
            if factor["contribution_pct"] > 20:
                factor["action"] = "key_driver"
            elif factor["contribution_pct"] > 10:
                factor["action"] = "significant"
            else:
                factor["action"] = "minor"

        return factors

    def _get_prediction(self, features: np.ndarray) -> float:
        """Get model prediction"""
        try:
            if hasattr(self.model, "predict"):
                features_2d = features.reshape(1, -1) if features.ndim == 1 else features
                pred = self.model.predict(features_2d)
                return float(pred[0]) if isinstance(pred, (list, np.ndarray)) else float(pred)
        except Exception as e:
            logger.debug(f"Error getting prediction: {e}")
        return 0.0

    def _calculate_confidence(self, factors: List[Dict]) -> float:
        """Calculate prediction confidence based on factor agreement"""
        if not factors:
            return 0.5

        # Filter out interactions for confidence calculation
        main_factors = [f for f in factors if not f.get("is_interaction", False)]

        if not main_factors:
            return 0.5

        # 1. Magnitude of top factors (higher = more confident)
        top_magnitude = main_factors[0]["magnitude"] if main_factors else 0
        magnitude_conf = min(top_magnitude * 2, 0.8)

        # 2. Consistency of factor directions
        positive_count = sum(1 for f in main_factors if f["impact"] == "positive")
        negative_count = sum(1 for f in main_factors if f["impact"] == "negative")

        if positive_count > negative_count:
            direction_consistency = positive_count / len(main_factors)
        else:
            direction_consistency = negative_count / len(main_factors)

        # 3. Spread of contributions (less spread = more focused = higher confidence)
        if len(main_factors) > 1:
            contributions = [f["contribution_pct"] for f in main_factors[:5]]
            if np.mean(contributions) > 0:
                contribution_spread = np.std(contributions) / np.mean(contributions)
                focus_conf = max(0, 1 - contribution_spread / 2)
            else:
                focus_conf = 0.5
        else:
            focus_conf = 1.0

        # Combine metrics
        confidence = 0.4 * magnitude_conf + 0.3 * direction_consistency + 0.3 * focus_conf

        return float(np.clip(confidence, 0.3, 0.95))

    def _get_factor_correlations(self, factors: List[Dict]) -> Dict[str, float]:
        """Calculate correlations between top factors"""
        # This would use historical factor data
        # Placeholder implementation for demonstration
        correlations = {}

        main_factors = [f for f in factors[:5] if not f.get("is_interaction", False)]

        for i, f1 in enumerate(main_factors):
            for f2 in main_factors[i + 1 :]:
                if f1.get("category") == f2.get("category"):
                    correlations[f"{f1['factor']} vs {f2['factor']}"] = 0.7
                else:
                    correlations[f"{f1['factor']} vs {f2['factor']}"] = 0.2

        return correlations

    def _get_fallback_factors(self) -> List[Dict]:
        """Return fallback factors when explanation fails"""
        return [
            {
                "factor": "revenue_growth",
                "importance": 0.3,
                "magnitude": 0.3,
                "contribution_pct": 30.0,
                "impact": "positive",
                "category": "growth",
                "description": "Year-over-year revenue growth",
                "interpretation": "Strong revenue growth driving positive outlook",
                "action": "key_driver",
                "is_interaction": False,
            },
            {
                "factor": "pe_ratio",
                "importance": 0.25,
                "magnitude": 0.25,
                "contribution_pct": 25.0,
                "impact": "negative",
                "category": "valuation",
                "description": "Price to Earnings ratio",
                "interpretation": "Higher valuation requiring strong growth",
                "action": "significant",
                "is_interaction": False,
            },
            {
                "factor": "roe",
                "importance": 0.2,
                "magnitude": 0.2,
                "contribution_pct": 20.0,
                "impact": "positive",
                "category": "quality",
                "description": "Return on Equity",
                "interpretation": "Strong profitability metrics",
                "action": "significant",
                "is_interaction": False,
            },
        ]

    def get_explanation_summary(self, symbol: str) -> Optional[Dict]:
        """Get comprehensive explanation summary for a symbol"""
        if symbol in self.explanation_cache:
            return self.explanation_cache[symbol].to_dict()
        return None

    def get_feature_categories(self) -> Dict[str, List[str]]:
        """Get features grouped by category"""
        categories = {}
        for feature, metadata in self.feature_metadata.items():
            cat = metadata["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(feature)
        return categories

    def clear_cache(self):
        """Clear explanation cache"""
        self.explanation_cache.clear()
        self.shap_values_cache.clear()

    # Backward compatibility function
    async def _get_top_factors(self, symbol: str, features: np.ndarray, feature_values: Optional[pd.Series] = None, n_factors: int = 5) -> List[Dict]:
        """
        Get top contributing factors for prediction

        This is the method that would be called from your strategy classes
        """
        # Initialize explainer if not already done
        if not hasattr(self, "_factor_explainer"):
            # Get feature names from somewhere (you'd need to store these)
            feature_names = getattr(self, "feature_names", [f"feature_{i}" for i in range(len(features))])

            # Get model
            model = getattr(self, "model", None)
            if model is None:
                model = getattr(self, "sector_models", {}).get("default", None)

            self._factor_explainer = FactorExplainer(
                model=model, feature_names=feature_names, model_type="auto", use_interactions=True, n_background_samples=100
            )

        # Get explanations
        return await self._factor_explainer.get_top_factors(
            symbol=symbol, features=features, feature_values=feature_values, n_factors=n_factors, include_interactions=True
        )
