"""
Reinforcement Learning strategy catalog.
Registers all RL-based trading strategies.
"""

from typing import Dict

from ...strategies.catelog.category import StrategyCategory
from ...strategies.catelog.strategy_info import StrategyInfo
from ...strategies.rl.rl_portfolio_allocator import RLPortfolioAllocator
from ...strategies.rl.rl_regime_allocator import RLRegimeAllocator
from ...strategies.rl.rl_risk_sensitive import RLRiskSensitiveTrader
from ...strategies.rl.rl_sentiment_trader import RLSentimentTrader


class ReinforcementLearning:
    def __init__(self):
        self.strategies = self._build_catalog()

    def _build_catalog(self) -> Dict[str, StrategyInfo]:
        """Build the RL strategy catalog."""

        catalog = {
            "rl_portfolio_allocator": StrategyInfo(
                name="RL Portfolio Allocator",
                class_type=RLPortfolioAllocator,
                category=StrategyCategory.REINFORCEMENT_LEARNING,
                description="Deep RL portfolio optimizer using PPO actor-critic. "
                "Learns optimal weight allocation across multiple assets by maximizing "
                "risk-adjusted returns net of transaction costs.",
                complexity="Expert",
                time_horizon="Medium-Long",
                best_for=[
                    "Multi-asset portfolio optimization",
                    "Dynamic asset allocation",
                    "Adaptive capital management",
                ],
                parameters={
                    "lookback": {
                        "default": 20,
                        "range": (10, 60),
                        "description": "State observation window (days)",
                    },
                    "hidden_dim": {
                        "default": 128,
                        "range": (64, 256),
                        "description": "Policy network hidden layer size",
                    },
                    "learning_rate": {
                        "default": 0.0003,
                        "range": (0.00001, 0.001),
                        "description": "PPO learning rate",
                    },
                    "gamma": {
                        "default": 0.99,
                        "range": (0.9, 0.999),
                        "description": "Discount factor",
                    },
                    "episodes": {
                        "default": 500,
                        "range": (100, 2000),
                        "description": "Training episodes",
                    },
                    "transaction_cost": {
                        "default": 0.001,
                        "range": (0.0, 0.01),
                        "description": "Transaction cost per unit turnover",
                    },
                },
                pros=[
                    "Adapts to changing market conditions",
                    "Considers transaction costs in optimization",
                    "Learns complex cross-asset relationships",
                    "Continuously improvable with more data",
                ],
                cons=[
                    "Requires significant training data",
                    "Computationally expensive to train",
                    "Risk of overfitting to training period",
                    "Black-box decision making",
                ],
                backtest_mode="multi",
                tags=["reinforcement-learning", "portfolio", "deep-learning", "ppo", "multi-asset"],
                requires_ml_training=True,
                min_data_days=252,
            ),
            "rl_regime_allocator": StrategyInfo(
                name="RL Regime Allocator",
                class_type=RLRegimeAllocator,
                category=StrategyCategory.REINFORCEMENT_LEARNING,
                description="Hierarchical RL meta-strategy that dynamically allocates capital across "
                "6 sub-strategies (SMA, RSI, Bollinger, MACD, Momentum, Vol Breakout) based on "
                "detected market regime. Learns when to deploy which specialist strategy.",
                complexity="Expert",
                time_horizon="Adaptable",
                best_for=[
                    "All-weather portfolio management",
                    "Regime-adaptive trading",
                    "Multi-strategy ensemble",
                ],
                parameters={
                    "lookback": {
                        "default": 60,
                        "range": (20, 120),
                        "description": "Regime detection window (days)",
                    },
                    "hidden_dim": {
                        "default": 128,
                        "range": (64, 256),
                        "description": "Policy network hidden layer size",
                    },
                    "learning_rate": {
                        "default": 0.0003,
                        "range": (0.00001, 0.001),
                        "description": "PPO learning rate",
                    },
                    "episodes": {
                        "default": 500,
                        "range": (100, 2000),
                        "description": "Training episodes",
                    },
                    "performance_window": {
                        "default": 20,
                        "range": (5, 60),
                        "description": "Sub-strategy performance evaluation window",
                    },
                },
                pros=[
                    "Adapts to regime changes automatically",
                    "Combines best of multiple strategy types",
                    "Robust across market conditions",
                    "Learns optimal strategy timing",
                ],
                cons=[
                    "Complex to train and validate",
                    "Many hyperparameters",
                    "Requires diverse market data for training",
                    "Higher latency than single strategies",
                ],
                backtest_mode="single",
                tags=["reinforcement-learning", "regime", "hierarchical", "ensemble", "adaptive"],
                requires_ml_training=True,
                min_data_days=252,
            ),
            "rl_risk_sensitive": StrategyInfo(
                name="RL Risk-Sensitive Trader",
                class_type=RLRiskSensitiveTrader,
                category=StrategyCategory.REINFORCEMENT_LEARNING,
                description="Distributional RL trader that maximizes risk-adjusted returns using "
                "quantile regression. Learns the full distribution of returns (not just expected value) "
                "and explicitly penalizes drawdowns and tail risk (CVaR).",
                complexity="Expert",
                time_horizon="Medium",
                best_for=[
                    "Risk-conscious trading",
                    "Drawdown minimization",
                    "Tail risk management",
                ],
                parameters={
                    "lookback": {
                        "default": 20,
                        "range": (10, 60),
                        "description": "State observation window (days)",
                    },
                    "hidden_dim": {
                        "default": 128,
                        "range": (64, 256),
                        "description": "Policy network hidden layer size",
                    },
                    "n_quantiles": {
                        "default": 10,
                        "range": (5, 50),
                        "description": "Number of return distribution quantiles",
                    },
                    "drawdown_penalty": {
                        "default": 2.0,
                        "range": (0.5, 10.0),
                        "description": "Drawdown penalty weight in reward",
                    },
                    "cvar_penalty": {
                        "default": 1.0,
                        "range": (0.1, 5.0),
                        "description": "CVaR (tail risk) penalty weight",
                    },
                    "episodes": {
                        "default": 500,
                        "range": (100, 2000),
                        "description": "Training episodes",
                    },
                },
                pros=[
                    "Explicit tail-risk management",
                    "Learns full return distribution",
                    "Adaptive position sizing based on risk",
                    "Outperforms in crash-prone markets",
                ],
                cons=[
                    "Slower training (distributional critic)",
                    "Sensitive to penalty hyperparameters",
                    "May be overly conservative in bull markets",
                    "Complex reward function tuning",
                ],
                backtest_mode="single",
                tags=["reinforcement-learning", "risk-management", "distributional", "cvar", "quantile"],
                requires_ml_training=True,
                min_data_days=252,
            ),
            "rl_sentiment_trader": StrategyInfo(
                name="RL Sentiment Trader",
                class_type=RLSentimentTrader,
                category=StrategyCategory.REINFORCEMENT_LEARNING,
                description="Sentiment-augmented RL agent with dual-branch architecture. "
                "Fuses technical indicators with sentiment scores (FinBERT/NewsAPI) to learn "
                "optimal trading decisions. Adapts sentiment weighting dynamically.",
                complexity="Expert",
                time_horizon="Short-Medium",
                best_for=[
                    "News-driven markets",
                    "Earnings season trading",
                    "Sentiment-driven momentum",
                ],
                parameters={
                    "lookback": {
                        "default": 20,
                        "range": (10, 60),
                        "description": "Technical observation window (days)",
                    },
                    "hidden_dim": {
                        "default": 128,
                        "range": (64, 256),
                        "description": "Policy network hidden layer size",
                    },
                    "sentiment_weight": {
                        "default": 0.3,
                        "range": (0.1, 0.8),
                        "description": "Initial sentiment influence weight",
                    },
                    "episodes": {
                        "default": 500,
                        "range": (100, 2000),
                        "description": "Training episodes",
                    },
                },
                pros=[
                    "Captures sentiment-driven moves",
                    "Dual-branch lets model weight signals dynamically",
                    "Learns non-linear sentiment→position mappings",
                    "Works with or without real sentiment data",
                ],
                cons=[
                    "Sentiment data quality varies",
                    "Requires sentiment API (FinBERT/NewsAPI)",
                    "Synthetic sentiment fallback is weak",
                    "Overfitting risk with noisy sentiment",
                ],
                backtest_mode="single",
                tags=["reinforcement-learning", "sentiment", "nlp", "dual-branch", "finbert"],
                requires_ml_training=True,
                min_data_days=120,
            ),
        }

        return catalog
