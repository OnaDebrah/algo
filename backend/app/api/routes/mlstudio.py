import os
import pickle
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException

from backend.app.api.deps import get_current_active_user
from backend.app.config import settings
from backend.app.core import fetch_stock_data
from backend.app.models import User
from backend.app.schemas.mlstudio import MLModel, TrainingConfig
from backend.app.strategies import MLStrategy

router = APIRouter(prefix="/mlstudio", tags=["ML Studio"])

# Directory to store trained models
MODELS_DIR = os.path.join(os.path.dirname(settings.DATABASE_PATH), "ml_models")
os.makedirs(MODELS_DIR, exist_ok=True)

# In-memory model registry (in production, use database)
model_registry: Dict[str, Dict[str, Any]] = {}


def save_model(model_id: str, ml_strategy: MLStrategy, metadata: Dict[str, Any]):
    """Save trained model to disk"""
    model_path = os.path.join(MODELS_DIR, f"{model_id}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(ml_strategy, f)

    model_registry[model_id] = {**metadata, "model_path": model_path}


def load_model(model_id: str) -> MLStrategy:
    """Load trained model from disk"""
    if model_id not in model_registry:
        raise ValueError(f"Model {model_id} not found")

    model_path = model_registry[model_id]["model_path"]
    with open(model_path, "rb") as f:
        return pickle.load(f)


def get_model_metadata(model_id: str) -> Dict[str, Any]:
    """Get model metadata without loading the full model"""
    if model_id not in model_registry:
        raise ValueError(f"Model {model_id} not found")
    return model_registry[model_id]


@router.get("/models", response_model=List[MLModel])
async def get_models(current_user: User = Depends(get_current_active_user)):
    """
    Get all trained ML models for the current user
    """
    # Filter models by user (in production, query from database)
    user_models = [MLModel(**metadata) for model_id, metadata in model_registry.items() if metadata.get("user_id") == current_user.id]

    # If no models, return empty list
    return user_models


@router.post("/train", response_model=MLModel)
async def train_model(config: TrainingConfig, current_user: User = Depends(get_current_active_user)):
    """
    Train a new ML model with the specified configuration
    """
    try:
        # Fetch historical data
        data = fetch_stock_data(config.symbol, period=config.period if hasattr(config, "period") else "2y", interval="1d")

        if data.empty:
            raise HTTPException(status_code=400, detail=f"No data available for {config.symbol}")

        # Create and train ML strategy
        ml_strategy = MLStrategy(
            name=f"{config.symbol}_{config.model_type}",
            model_type=config.model_type,
            n_estimators=config.n_estimators if hasattr(config, "n_estimators") else 100,
            max_depth=config.max_depth if hasattr(config, "max_depth") else 10,
            test_size=0.2,
            learning_rate=config.learning_rate if hasattr(config, "learning_rate") else 0.1,
        )

        # Train the model
        train_start = datetime.now()
        train_score, test_score = ml_strategy.train(data)
        training_time = (datetime.now() - train_start).total_seconds()

        # Get feature importance
        feature_importance_df = ml_strategy.get_feature_importance()
        feature_importance = [
            {"feature": row["feature"], "importance": float(row["importance"])} for _, row in feature_importance_df.head(10).iterrows()
        ]

        # Generate model ID
        model_id = f"model-{config.symbol}-{int(datetime.now().timestamp())}"

        # Calculate overfit score
        overfit_score = abs(train_score - test_score)

        # Create model metadata
        metadata = {
            "id": model_id,
            "name": f"{config.symbol}_{config.model_type}_v1",
            "type": config.model_type,
            "symbol": config.symbol,
            "accuracy": round(train_score, 3),
            "test_accuracy": round(test_score, 3),
            "overfit_score": round(overfit_score, 3),
            "features": len(ml_strategy.feature_cols),
            "training_time": int(training_time),
            "created": datetime.now(),
            "status": "trained",
            "feature_importance": feature_importance,
            "hyperparams": {
                "n_estimators": ml_strategy.n_estimators,
                "max_depth": ml_strategy.max_depth,
                "learning_rate": ml_strategy.learning_rate,
            },
            "user_id": current_user.id,
        }

        # Save model
        save_model(model_id, ml_strategy, metadata)

        return MLModel(**metadata)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to train model: {str(e)}")


@router.post("/deploy/{model_id}")
async def deploy_model(model_id: str, current_user: User = Depends(get_current_active_user)):
    """
    Deploy a trained model for live trading
    """
    try:
        metadata = get_model_metadata(model_id)

        # Verify ownership
        if metadata.get("user_id") != current_user.id:
            raise HTTPException(status_code=403, detail="Unauthorized")

        # Undeploy other models for the same symbol
        symbol = metadata["symbol"]
        for mid, meta in model_registry.items():
            if meta["symbol"] == symbol and meta["status"] == "deployed":
                meta["status"] = "trained"

        # Deploy this model
        model_registry[model_id]["status"] = "deployed"

        return {"status": "success", "model_id": model_id, "message": f"Model deployed for {symbol}"}

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deployment failed: {str(e)}")


@router.delete("/models/{model_id}")
async def delete_model(model_id: str, current_user: User = Depends(get_current_active_user)):
    """
    Delete a trained model
    """
    try:
        metadata = get_model_metadata(model_id)

        # Verify ownership
        if metadata.get("user_id") != current_user.id:
            raise HTTPException(status_code=403, detail="Unauthorized")

        # Delete model file
        model_path = metadata["model_path"]
        if os.path.exists(model_path):
            os.remove(model_path)

        # Remove from registry
        del model_registry[model_id]

        return {"status": "success", "message": f"Model {model_id} deleted"}

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")


@router.get("/models/{model_id}/predict")
async def predict_with_model(model_id: str, current_user: User = Depends(get_current_active_user)):
    """
    Generate prediction using a trained model
    """
    try:
        metadata = get_model_metadata(model_id)

        # Verify ownership
        if metadata.get("user_id") != current_user.id:
            raise HTTPException(status_code=403, detail="Unauthorized")

        # Load model
        ml_strategy = load_model(model_id)

        # Fetch recent data
        symbol = metadata["symbol"]
        data = fetch_stock_data(symbol, period="1mo", interval="1d")

        if data.empty:
            raise HTTPException(status_code=400, detail=f"No data available for {symbol}")

        # Generate signal
        signal = ml_strategy.generate_signal(data)

        signal_map = {1: "BUY", -1: "SELL", 0: "HOLD"}

        return {
            "model_id": model_id,
            "symbol": symbol,
            "signal": signal_map.get(signal, "HOLD"),
            "signal_value": signal,
            "timestamp": datetime.now().isoformat(),
            "current_price": float(data["Close"].iloc[-1]),
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
