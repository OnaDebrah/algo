from typing import Dict

from ...strategies.catelog.category import StrategyCategory
from ...strategies.catelog.strategy_info import StrategyInfo
from ...strategies.ml.genetics.gp.genetic_programming import GeneticProgrammingStrategy


class Genetic:
    def __init__(self):
        self.strategies = self._build_catalog()

    def _build_catalog(self) -> Dict[str, StrategyInfo]:
        """Build the genetic strategies catalog"""

        catalog = {
            "genetic_programming": StrategyInfo(
                name="Genetic Programming",
                class_type=GeneticProgrammingStrategy,
                category=StrategyCategory.MACHINE_LEARNING,
                description="Evolves trading rules using strongly-typed genetic programming. "
                "Automatically discovers optimal combinations of technical indicators "
                "and logical operators through evolutionary optimization. "
                "Supports both Physical Time and Directional Change frameworks.",
                complexity="Expert",
                time_horizon="Medium to Long-term",
                best_for=[
                    "Automated rule discovery",
                    "Complex pattern recognition",
                    "Strategy evolution",
                    "Non-linear trading rules",
                ],
                parameters={
                    "population_size": {
                        "default": 100,
                        "range": (20, 500),
                        "description": "Number of GP trees in population",
                    },
                    "generations": {
                        "default": 50,
                        "range": (10, 200),
                        "description": "Number of evolution generations",
                    },
                    "tournament_size": {
                        "default": 5,
                        "range": (2, 10),
                        "description": "Tournament selection size",
                    },
                    "elitism_count": {
                        "default": 2,
                        "range": (1, 10),
                        "description": "Number of elites to preserve",
                    },
                    "crossover_rate": {
                        "default": 0.7,
                        "range": (0.5, 0.95),
                        "description": "Crossover probability",
                    },
                    "mutation_rate": {
                        "default": 0.1,
                        "range": (0.01, 0.3),
                        "description": "Mutation probability",
                    },
                    "max_depth": {
                        "default": 5,
                        "range": (3, 10),
                        "description": "Maximum tree depth",
                    },
                    "init_method": {
                        "default": "ramped_half",
                        "range": ["grow", "full", "ramped_half"],
                        "description": "Population initialization method",
                    },
                    "fitness_metric": {
                        "default": "sharpe",
                        "range": ["sharpe", "total_return", "calmar", "sortino"],
                        "description": "Fitness evaluation metric",
                    },
                    "retrain_frequency": {
                        "default": 90,
                        "range": (30, 365),
                        "description": "Retraining frequency (days)",
                    },
                    "validation_split": {
                        "default": 0.3,
                        "range": (0.1, 0.5),
                        "description": "Fraction of data for validation",
                    },
                    "use_dc_framework": {
                        "default": False,
                        "range": [True, False],
                        "description": "Use Directional Change event framework",
                    },
                    "dc_threshold": {
                        "default": 0.01,
                        "range": (0.005, 0.05),
                        "description": "Directional Change threshold",
                    },
                    "random_seed": {
                        "default": 42,
                        "range": (1, 99999),
                        "description": "Random seed for reproducibility",
                    },
                },
                pros=[
                    "Automatically discovers trading rules",
                    "No assumption about strategy structure",
                    "Evolves interpretable tree-based rules",
                    "Supports multiple fitness metrics",
                    "Directional Change framework option",
                    "Built-in validation split",
                ],
                cons=[
                    "Computationally expensive evolution",
                    "Risk of overfitting to training data",
                    "Results vary with random seed",
                    "Requires substantial historical data",
                    "Complex to debug evolved rules",
                ],
                backtest_mode="single",
                tags=["genetic-programming", "evolutionary", "rule-discovery", "machine-learning", "self-training"],
                requires_ml_training=True,
                min_data_days=252,
            ),
        }
        return catalog
