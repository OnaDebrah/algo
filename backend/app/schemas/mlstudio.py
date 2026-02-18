from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class TrainingEpoch(BaseModel):
    epoch: int
    loss: float
    accuracy: float
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None


class MLModel(BaseModel):
    id: str
    name: str
    type: str
    symbol: str
    accuracy: float
    test_accuracy: float
    overfit_score: float
    features: int
    training_time: float
    created: datetime
    status: str
    feature_importance: List[FeatureImportance]
    hyperparams: Dict[str, Any]
    training_history: List[TrainingEpoch] = []


class TrainingConfig(BaseModel):
    model_config = {"protected_namespaces": ()}

    symbol: str
    model_type: str
    training_period: str
    test_size: int
    epochs: int
    batch_size: int
    learning_rate: float
    threshold: float
    use_feature_engineering: bool
    use_cross_validation: bool
