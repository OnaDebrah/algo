from typing import Dict

from ...strategies.catelog.category import StrategyCategory
from ...strategies.catelog.strategy_info import StrategyInfo
from ...strategies.ml.analysis.moments.integrated_moments import IntegratedMomentsStrategy


class DeepLearning:
    def __init__(self):
        self.strategies = self._build_catalog()

    def _build_catalog(self) -> Dict[str, StrategyInfo]:
        """Build the strategy catalog"""

        catalog = {
            "integrated_moments": StrategyInfo(
                name="Integrated First-Second Moments",
                class_type=IntegratedMomentsStrategy,
                category=StrategyCategory.DEEP_LEARNING,
                description="""
                Jointly predicts returns (first moment) and volatility (second moment) for optimal position sizing.
                Based on 2025 research showing >200% improvement over standalone models.

                Key innovation: Uses risk-adjusted returns to scale positions dynamically.
                """,
                complexity="Advanced",
                time_horizon="Medium-term",
                best_for=["Risk-aware trading", "Volatile markets", "Position sizing optimization", "Adaptive strategies"],
                parameters={
                    "sequence_length": {"default": 60, "range": (30, 120), "description": "Number of past days used for prediction"},
                    "forecast_horizon": {"default": 5, "range": (1, 20), "description": "Days ahead to forecast"},
                    "min_risk_adjusted": {"default": 0.5, "range": (0.1, 2.0), "description": "Minimum Sharpe-like ratio for entry"},
                    "max_position": {"default": 1.0, "range": (0.1, 2.0), "description": "Maximum position size (can exceed 1.0 with leverage)"},
                    "confidence_threshold": {"default": 0.3, "range": (0.1, 0.95), "description": "Minimum confidence for trading"},
                    "use_adaptive_sizing": {"default": True, "type": "boolean", "description": "Use Kelly-inspired dynamic position sizing"},
                    "retrain_frequency": {"default": 30, "range": (7, 90), "description": "Days between model retraining"},
                },
                pros=[
                    "Jointly models returns and risk",
                    "Dynamic position sizing",
                    "Uncertainty quantification via MC dropout",
                    "Adapts to changing market conditions",
                    "Research-backed >200% improvement",
                    "Rich feature set including technicals and moments",
                    "Auto-retraining capability",
                    "Kelly-optimal position sizing",
                ],
                cons=[
                    "Requires significant training data (2+ years)",
                    "Computationally intensive",
                    "Complex to understand",
                    "Needs GPU for training",
                    "Black box model with many parameters",
                    "Risk of overfitting without proper validation",
                ],
                backtest_mode="both",
                tags=["deep-learning", "risk-management", "volatility", "position-sizing", "adaptive", "first-moment", "second-moment", "integrated"],
                requires_ml_training=True,
                min_data_days=500,
            ),
        }
        return catalog
