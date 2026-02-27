"""
Registry for all ML models with unified interface and auto-registration
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from ...services.analysis.ensemble.enhanced_rf import EnhancedRandomForest
from ..analysis.lppls_bubbles_strategy import LPPLSBubbleStrategy
from ..analysis.lstm_stress_strategy import LSTMStressStrategy
from .drl_portfolio import DRLPortfolioOptimizer, DynamicFactorDRL
from .temporal_gat import TemporalGATService
from .timegan import TimeGAN

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Central registry for all ML models
    """

    _instance = None

    def __new__(cls):
        """Singleton pattern to ensure one registry"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.models = {}
        self.model_performance = {}
        self._initialized = True

        # Auto-register model classes (not instances)
        self.model_classes = {
            "temporal_gat": TemporalGATService,
            "drl_optimizer": DRLPortfolioOptimizer,
            "dynamic_factor_drl": DynamicFactorDRL,
            "timegan": TimeGAN,
            "enhanced_rf": EnhancedRandomForest,
            "lppls_bubble": LPPLSBubbleStrategy,
            "lstm_stress": LSTMStressStrategy,
        }

        logger.info(f"ModelRegistry initialized with {len(self.model_classes)} model types")

    def create_model(self, model_type: str, **kwargs) -> Any:
        """
        Factory method to create model instances
        """
        if model_type not in self.model_classes:
            raise ValueError(f"Unknown model type: {model_type}")

        model_class = self.model_classes[model_type]
        model = model_class(**kwargs)

        # Auto-register with default name
        name = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.register_model(name, model, model_type)

        return model

    def register_model(self, name: str, model: Any, model_type: str):
        """Register a model instance"""
        self.models[name] = {
            "model": model,
            "type": model_type,
            "registered_at": datetime.now(),
            "performance": {},
            "class": model.__class__.__name__,
        }
        logger.info(f"Registered model: {name} ({model_type})")

    def get_model(self, name: str) -> Optional[Any]:
        """Get model by name"""
        if name in self.models:
            return self.models[name]["model"]
        return None

    def get_all_models_of_type(self, model_type: str) -> List[Dict]:
        """Get all models of a specific type"""
        return [info for name, info in self.models.items() if info["type"] == model_type]

    def update_performance(self, name: str, metrics: Dict):
        """Update model performance metrics"""
        if name in self.models:
            self.models[name]["performance"] = metrics
            self.model_performance[name] = metrics

    def get_best_model(self, metric: str = "f1", model_type: Optional[str] = None) -> str:
        """
        Get best performing model by metric, optionally filtered by type
        """
        best_name = None
        best_score = -float("inf")

        for name, info in self.models.items():
            if model_type and info["type"] != model_type:
                continue

            score = info["performance"].get(metric, -float("inf"))
            if score > best_score:
                best_score = score
                best_name = name

        return best_name

    def ensemble_predict(self, X, method: str = "weighted", model_type: Optional[str] = None) -> np.ndarray:
        """
        Ensemble predictions from all models or specific type
        """
        predictions = []
        weights = []
        model_names = []

        for name, info in self.models.items():
            if model_type and info["type"] != model_type:
                continue

            model = info["model"]

            try:
                if hasattr(model, "predict_proba"):
                    pred = model.predict_proba(X)
                    # Handle both binary and multi-class
                    if len(pred.shape) > 1 and pred.shape[1] > 1:
                        pred = pred[:, 1]  # Take positive class probability
                    else:
                        pred = pred.flatten()
                elif hasattr(model, "predict"):
                    pred = model.predict(X)
                    if len(pred.shape) > 1:
                        pred = pred.flatten()
                else:
                    continue

                predictions.append(pred)
                model_names.append(name)

                # Weight by performance
                if method == "weighted":
                    perf = info["performance"].get("f1", 0.5)
                    weights.append(perf)
                else:
                    weights.append(1.0)

            except Exception as e:
                logger.warning(f"Error getting prediction from {name}: {e}")
                continue

        if not predictions:
            logger.warning("No predictions available for ensemble")
            return np.zeros(len(X) if hasattr(X, "__len__") else 1)

        # Weighted average
        predictions = np.array(predictions)
        weights = np.array(weights)

        if weights.sum() > 0:
            weights = weights / weights.sum()
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
        else:
            ensemble_pred = np.mean(predictions, axis=0)

        # Log ensemble composition
        logger.info(f"Ensemble prediction using {len(predictions)} models: {model_names}")

        return ensemble_pred

    def get_model_explanations(self, name: str, X_sample) -> Dict:
        """Get explanations for model prediction"""
        model_info = self.models.get(name)
        if not model_info:
            return {"error": f"Model {name} not found"}

        model = model_info["model"]

        # Try different explanation methods
        if hasattr(model, "explain_prediction"):
            return model.explain_prediction(X_sample)
        elif hasattr(model, "explain_decision"):
            return model.explain_decision(X_sample)
        elif hasattr(model, "get_feature_importance"):
            return {"feature_importance": model.get_feature_importance()}
        else:
            return {"explanation": "No explanation method available"}

    def list_models(self) -> List[Dict]:
        """List all registered models"""
        return [
            {
                "name": name,
                "type": info["type"],
                "class": info["class"],
                "registered_at": info["registered_at"].isoformat(),
                "performance": info["performance"],
            }
            for name, info in self.models.items()
        ]

    def summary(self) -> Dict:
        """Get summary statistics of registered models"""
        total = len(self.models)
        by_type = {}

        for info in self.models.values():
            model_type = info["type"]
            by_type[model_type] = by_type.get(model_type, 0) + 1

        return {
            "total_models": total,
            "models_by_type": by_type,
            "best_model_by_f1": self.get_best_model("f1"),
            "best_model_by_recall": self.get_best_model("recall"),
            "best_model_by_precision": self.get_best_model("precision"),
        }


registry = ModelRegistry()


# Example usage function
def initialize_default_models(symbols=None):
    """
    Initialize and register default model instances
    """
    # Create TemporalGAT
    if symbols is None:
        symbols = ["SPY", "QQQ", "TLT"]
    registry.create_model("temporal_gat", symbols=symbols)

    # Create DRL Optimizer (requires state_dim - would need actual data)
    # drl = registry.create_model('drl_optimizer', n_assets=3, state_dim=100)

    # Create LPPLS Strategy
    registry.create_model("lppls_bubble", name="default_lppls", params={"lookback_window": 252})

    # Create LSTM Stress
    registry.create_model("lstm_stress", name="default_lstm")

    logger.info(f"Initialized {len(registry.models)} default models")
    return registry
