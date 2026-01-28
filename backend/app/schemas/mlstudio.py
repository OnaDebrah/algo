from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime

class FeatureImportance(BaseModel):
    feature: str
    importance: float

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

class TrainingConfig(BaseModel):
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
