from typing import Dict

from ...strategies import MLStrategy
from ...strategies.catelog.category import StrategyCategory
from ...strategies.catelog.strategy_info import StrategyInfo
from ...strategies.lstm_strategy import LSTMStrategy
from ..ml.mc_ml_sentiment_strategy import MonteCarloMLSentimentStrategy
from ..ml.sector_prediction.sector_rotation_alt_strategy import SectorRotationAltStrategy
from ..ml.sector_prediction.sector_rotation_strategy import SectorRotationStrategy


class ML:
    def __init__(self):
        self.strategies = self._build_catalog()

    def _build_catalog(self) -> Dict[str, StrategyInfo]:
        """Build the strategy catalog"""

        catalog = {
            "ml_random_forest": StrategyInfo(
                name="ML Random Forest",
                class_type=MLStrategy,
                category=StrategyCategory.MACHINE_LEARNING,
                description="Uses Random Forest classifier trained on technical indicators to predict market direction.",
                complexity="Advanced",
                time_horizon="Adaptable",
                best_for=[
                    "Complex pattern recognition",
                    "Multi-factor analysis",
                    "Data-rich environments",
                ],
                parameters={
                    "strategy_type": {"default": "random_forest", "range": None, "description": "Model type"},
                    "n_estimators": {
                        "default": 100,
                        "range": (50, 500),
                        "description": "Number of trees",
                    },
                    "max_depth": {
                        "default": 10,
                        "range": (5, 30),
                        "description": "Maximum tree depth",
                    },
                    "test_size": {
                        "default": 0.2,
                        "range": (0.1, 0.4),
                        "description": "Test set size",
                    },
                },
                pros=[
                    "Learns complex patterns",
                    "Adapts to market conditions",
                    "Multi-indicator integration",
                    "Non-linear relationships",
                ],
                cons=[
                    "Requires substantial training data",
                    "Black box (hard to interpret)",
                    "Risk of overfitting",
                    "Computationally expensive",
                ],
                backtest_mode="single",
            ),
            "ml_gradient_boosting": StrategyInfo(
                name="ML Gradient Boosting",
                class_type=MLStrategy,
                category=StrategyCategory.MACHINE_LEARNING,
                description="Uses Gradient Boosting classifier for sequential learning and improved predictions.",
                complexity="Advanced",
                time_horizon="Adaptable",
                best_for=[
                    "Complex patterns",
                    "Incremental learning",
                    "High accuracy needs",
                ],
                parameters={
                    "strategy_type": {"default": "gradient_boosting", "range": None, "description": "Model type"},
                    "n_estimators": {
                        "default": 100,
                        "range": (50, 500),
                        "description": "Number of boosting stages",
                    },
                    "learning_rate": {
                        "default": 0.1,
                        "range": (0.01, 0.3),
                        "description": "Learning rate",
                    },
                    "max_depth": {
                        "default": 5,
                        "range": (3, 15),
                        "description": "Tree depth",
                    },
                },
                pros=[
                    "Often more accurate than Random Forest",
                    "Handles complex patterns well",
                    "Sequential learning",
                    "Feature importance available",
                ],
                cons=[
                    "Even more prone to overfitting",
                    "Slower to train",
                    "Requires careful tuning",
                    "Computationally intensive",
                ],
                backtest_mode="single",
            ),
            "ml_svm": StrategyInfo(
                name="ML SVM Classifier",
                class_type=MLStrategy,
                category=StrategyCategory.MACHINE_LEARNING,
                description="Uses Support Vector Machine (SVM) to classify market regimes and predict direction.",
                complexity="Advanced",
                time_horizon="Adaptable",
                best_for=[
                    "Regime classification",
                    "Non-linear boundaries",
                    "Small datasets",
                ],
                parameters={
                    "strategy_type": {"default": "svm", "range": None, "description": "Model type"},
                    "test_size": {
                        "default": 0.2,
                        "range": (0.1, 0.4),
                        "description": "Test set size",
                    },
                },
                pros=[
                    "Effective in high dimensions",
                    "Robust to overfitting",
                    "Good for regime detection",
                ],
                cons=[
                    "Slow on large datasets",
                    "Sensitive to noise",
                    "Hard to interpret probability",
                ],
                backtest_mode="single",
            ),
            "ml_logistic": StrategyInfo(
                name="ML Logistic Regression",
                class_type=MLStrategy,
                category=StrategyCategory.MACHINE_LEARNING,
                description="Uses Logistic Regression for simple, interpretable market direction prediction.",
                complexity="Intermediate",
                time_horizon="Adaptable",
                best_for=[
                    "Baseline models",
                    "Interpretability",
                    "Linear relationships",
                ],
                parameters={
                    "strategy_type": {"default": "logistic_regression", "range": None, "description": "Model type"},
                    "test_size": {
                        "default": 0.2,
                        "range": (0.1, 0.4),
                        "description": "Test set size",
                    },
                },
                pros=[
                    "Highly interpretable",
                    "Fast training",
                    "Less overfitting risk",
                ],
                cons=[
                    "Linear boundaries only",
                    "Underperforms on complex data",
                    "Requires feature engineering",
                ],
                backtest_mode="single",
            ),
            "ml_lstm": StrategyInfo(
                name="ML LSTM (Deep Learning)",
                class_type=LSTMStrategy,
                category=StrategyCategory.MACHINE_LEARNING,
                description="Uses Long Short-Term Memory (LSTM) neural network for time-series forecasting.",
                complexity="Expert",
                time_horizon="Short to Medium-term",
                best_for=[
                    "Time-series forecasting",
                    "Sequence patterns",
                    "Complex temporal dependencies",
                ],
                parameters={
                    "lookback": {
                        "default": 10,
                        "range": (5, 60),
                        "description": "Sequence lookback length",
                    },
                    "classes": {
                        "default": 2,
                        "range": (2, 3),
                        "description": "Number of classes (Up/Down)",
                    },
                    "epochs": {
                        "default": 20,
                        "range": (10, 100),
                        "description": "Training epochs",
                    },
                },
                pros=[
                    "Captures temporal dependencies",
                    "State-of-the-art for sequences",
                    "Non-linear mapping",
                ],
                cons=[
                    "Requires large data",
                    "Computationally expensive",
                    "Hard to train (vanishing gradients)",
                    "Black box",
                ],
                backtest_mode="single",
            ),
            "mc_ml_sentiment": StrategyInfo(
                name="Monte Carlo ML Sentiment",
                class_type=MonteCarloMLSentimentStrategy,
                category=StrategyCategory.MACHINE_LEARNING,
                description="Combines sentiment analysis, machine learning predictions, "
                "and Monte Carlo simulation for probabilistic price forecasting. "
                "Uses Kelly Criterion for risk-aware position sizing.",
                complexity="Expert",
                time_horizon="Short to Medium-term",
                best_for=[
                    "Sentiment-driven trading",
                    "Probabilistic forecasting",
                    "Risk-aware position sizing",
                    "Multi-factor alpha generation",
                ],
                parameters={
                    "ml_model_type": {
                        "default": "gradient_boosting",
                        "range": ["gradient_boosting", "random_forest"],
                        "description": "ML model for return prediction",
                    },
                    "lookback_period": {
                        "default": 252,
                        "range": (60, 504),
                        "description": "Training data lookback (days)",
                    },
                    "forecast_horizon": {
                        "default": 20,
                        "range": (5, 60),
                        "description": "Forecast horizon (days)",
                    },
                    "num_simulations": {
                        "default": 10000,
                        "range": (1000, 50000),
                        "description": "Number of Monte Carlo paths",
                    },
                    "confidence_level": {
                        "default": 0.95,
                        "range": (0.90, 0.99),
                        "description": "Confidence level for VaR/bounds",
                    },
                    "sentiment_weight": {
                        "default": 0.3,
                        "range": (0.0, 1.0),
                        "description": "Weight of sentiment in combined signal",
                    },
                },
                pros=[
                    "Combines multiple alpha sources (sentiment + technical + ML)",
                    "Probabilistic risk assessment via Monte Carlo",
                    "Kelly Criterion position sizing",
                    "Automatic model retraining",
                    "Confidence-weighted signals",
                ],
                cons=[
                    "Requires sentiment data API access (Twitter, news)",
                    "Computationally intensive (Monte Carlo sims)",
                    "Complex pipeline with multiple failure points",
                    "Sentiment data quality varies",
                    "ML model overfitting risk",
                ],
                backtest_mode="single",
            ),
            # ============================================================
            # ML SECTOR ROTATION & REGIME-AWARE
            # ============================================================
            "sector_rotation": StrategyInfo(
                name="Sector Rotation",
                class_type=SectorRotationStrategy,
                category=StrategyCategory.MACHINE_LEARNING,
                description="ML-based sector rotation using macro ETF data and fundamental analysis. Predicts top-performing sectors and selects best stocks within them.",
                complexity="Advanced",
                time_horizon="Medium-term",
                best_for=["Sector allocation", "Macro-driven trading", "Portfolio rotation"],
                parameters={
                    "lookback_years": {"default": 5, "range": (2, 10), "description": "Years of historical data for training"},
                    "forecast_horizon_days": {"default": 60, "range": (20, 120), "description": "Forward return prediction horizon (days)"},
                    "top_sectors": {"default": 3, "range": (1, 5), "description": "Number of sectors to allocate to"},
                    "stocks_per_sector": {"default": 5, "range": (3, 10), "description": "Number of stocks per sector"},
                    "model_type": {
                        "default": "random_forest",
                        "range": ["random_forest", "gradient_boosting", "ensemble"],
                        "description": "ML model type",
                    },
                    "rebalance_frequency_days": {"default": 30, "range": (7, 90), "description": "Rebalancing frequency (days)"},
                },
                pros=[
                    "Combines macro and fundamental analysis",
                    "Dynamic sector allocation",
                    "Regime-aware predictions",
                    "Confidence-weighted positions",
                ],
                cons=[
                    "Requires multiple data sources",
                    "Computationally intensive",
                    "Model retraining needed",
                    "Sector ETF data quality dependent",
                ],
                backtest_mode="multi",
            ),
            "sector_rotation_alt": StrategyInfo(
                name="Enhanced Sector Rotation",
                class_type=SectorRotationAltStrategy,
                category=StrategyCategory.MACHINE_LEARNING,
                description="Sector rotation with SHAP explanations, alternative data (sentiment/news), and progressive ML complexity.",
                complexity="Advanced",
                time_horizon="Medium-term",
                best_for=["Explainable ML", "Alternative data", "Sentiment-driven rotation"],
                parameters={
                    "lookback_years": {"default": 5, "range": (2, 10), "description": "Years of historical data"},
                    "forecast_horizon_days": {"default": 60, "range": (20, 120), "description": "Forecast horizon (days)"},
                    "top_sectors": {"default": 3, "range": (1, 5), "description": "Number of sectors to select"},
                    "model_type": {
                        "default": "random_forest",
                        "range": ["random_forest", "gradient_boosting", "ensemble"],
                        "description": "ML model type",
                    },
                    "sentiment_weight": {"default": 0.2, "range": (0.0, 0.5), "description": "Weight of sentiment in scoring"},
                    "use_shap_explanations": {"default": True, "range": [True, False], "description": "Enable SHAP-based explanations"},
                },
                pros=[
                    "SHAP-based explainability",
                    "Alternative data integration",
                    "Sentiment-aware scoring",
                    "Transparent decision process",
                ],
                cons=[
                    "Requires sentiment API access",
                    "SHAP computation overhead",
                    "Complex pipeline",
                    "Sentiment data quality varies",
                ],
                backtest_mode="multi",
            ),
        }
        return catalog
