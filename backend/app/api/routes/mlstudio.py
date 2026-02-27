import json
import logging
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException

from ...api.deps import get_current_active_user
from ...config import settings
from ...core import fetch_stock_data
from ...models import User
from ...schemas.mlstudio import MLModel, TrainingConfig
from ...strategies import MLStrategy

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mlstudio", tags=["ML Studio"])

# Directory to store trained models
MODELS_DIR = os.path.join(os.path.dirname(settings.DATABASE_PATH), "ml_models")
os.makedirs(MODELS_DIR, exist_ok=True)

# In-memory model registry — restored from disk on startup
model_registry: Dict[str, Dict[str, Any]] = {}


def _save_metadata_to_disk(model_id: str, metadata: Dict[str, Any]):
    """Save model metadata as a JSON sidecar file alongside the pickle"""
    meta_path = os.path.join(MODELS_DIR, f"{model_id}.json")
    # Convert datetime to ISO string for JSON serialization
    serializable = {}
    for k, v in metadata.items():
        if isinstance(v, datetime):
            serializable[k] = v.isoformat()
        else:
            serializable[k] = v
    with open(meta_path, "w") as f:
        json.dump(serializable, f, indent=2)


def _restore_registry_from_disk():
    """Restore model registry from JSON sidecar files on startup"""
    restored = 0
    for filename in os.listdir(MODELS_DIR):
        if filename.endswith(".json"):
            meta_path = os.path.join(MODELS_DIR, filename)
            try:
                with open(meta_path, "r") as f:
                    metadata = json.load(f)
                mid = metadata.get("id")
                if mid and mid not in model_registry:
                    # Ensure model_path is set
                    pkl_path = os.path.join(MODELS_DIR, f"{mid}.pkl")
                    if os.path.exists(pkl_path):
                        metadata["model_path"] = pkl_path
                        # Convert created back to datetime if it's a string
                        if isinstance(metadata.get("created"), str):
                            try:
                                metadata["created"] = datetime.fromisoformat(metadata["created"])
                            except (ValueError, TypeError):
                                metadata["created"] = datetime.now()
                        model_registry[mid] = metadata
                        restored += 1
            except Exception as e:
                logger.warning(f"Failed to restore model metadata from {filename}: {e}")
    if restored > 0:
        logger.info(f"Restored {restored} ML models from disk into registry")


# Restore registry on module load (server startup)
_restore_registry_from_disk()


def save_model(model_id: str, ml_strategy: MLStrategy, metadata: Dict[str, Any]):
    """Save trained model to disk with metadata sidecar"""
    model_path = os.path.join(MODELS_DIR, f"{model_id}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(ml_strategy, f)

    full_metadata = {**metadata, "model_path": model_path}
    model_registry[model_id] = full_metadata
    _save_metadata_to_disk(model_id, full_metadata)


def load_model(model_id: str) -> MLStrategy:
    """Load trained model from disk"""
    if model_id not in model_registry:
        # Try to find the pickle file directly as a fallback
        pkl_path = os.path.join(MODELS_DIR, f"{model_id}.pkl")
        if os.path.exists(pkl_path):
            logger.info(f"Model {model_id} not in registry but found on disk, loading directly")
            with open(pkl_path, "rb") as f:
                return pickle.load(f)
        raise ValueError(f"Model {model_id} not found in registry or on disk")

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


@router.get("/models/deployed")
async def get_deployed_models(current_user: User = Depends(get_current_active_user)):
    """
    Get all deployed ML models for the current user.
    Used by the backtest UI to populate the model selector when an ML strategy is selected.
    """
    deployed = []
    for model_id, metadata in model_registry.items():
        if metadata.get("user_id") == current_user.id and metadata.get("status") == "deployed":
            deployed.append(
                {
                    "id": metadata["id"],
                    "name": metadata.get("name", model_id),
                    "type": metadata.get("type", "unknown"),
                    "symbol": metadata.get("symbol", ""),
                    "accuracy": metadata.get("accuracy", 0),
                    "test_accuracy": metadata.get("test_accuracy", 0),
                    "status": metadata.get("status", "deployed"),
                }
            )
    return deployed


# Map display names from the frontend UI to internal model_type slugs
_MODEL_TYPE_MAP = {
    "Random Forest": "random_forest",
    "Gradient Boosting": "gradient_boosting",
    "LSTM": "lstm",
    "XGBoost": "gradient_boosting",  # XGBoost falls back to gradient boosting
    "SVM": "svm",
    "Logistic Regression": "logistic_regression",
    # Also accept already-correct slugs
    "random_forest": "random_forest",
    "gradient_boosting": "gradient_boosting",
    "svm": "svm",
    "logistic_regression": "logistic_regression",
    "lstm": "lstm",
}


@router.post("/train", response_model=MLModel)
async def train_model(config: TrainingConfig, current_user: User = Depends(get_current_active_user)):
    """
    Train a new ML model with the specified configuration
    """
    try:
        # Normalize model_type from display name to internal slug
        model_type_slug = _MODEL_TYPE_MAP.get(config.model_type, config.model_type.lower().replace(" ", "_"))
        logger.info(f"Training model: type='{config.model_type}' -> slug='{model_type_slug}' for {config.symbol}")

        # Fetch historical data
        # TrainingConfig uses 'training_period' (e.g. "1Y", "2Y") — map to fetch_stock_data's 'period'
        training_period = getattr(config, "training_period", "2y").lower()
        data = await fetch_stock_data(config.symbol, period=training_period, interval="1d")

        if data.empty:
            raise HTTPException(status_code=400, detail=f"No data available for {config.symbol}")

        # Handle LSTM separately — it uses LSTMStrategy, not MLStrategy
        if model_type_slug == "lstm":
            from ...strategies.lstm_strategy import LSTMStrategy

            ml_strategy = LSTMStrategy(
                name=f"{config.symbol}_lstm",
                lookback=10,
                epochs=config.epochs if hasattr(config, "epochs") else 20,
                learning_rate=config.learning_rate if hasattr(config, "learning_rate") else 0.01,
            )
        else:
            # Create and train ML strategy (sklearn-based)
            ml_strategy = MLStrategy(
                name=f"{config.symbol}_{model_type_slug}",
                strategy_type=model_type_slug,
                n_estimators=config.n_estimators if hasattr(config, "n_estimators") else 100,
                max_depth=config.max_depth if hasattr(config, "max_depth") else 10,
                test_size=0.2,
                learning_rate=config.learning_rate if hasattr(config, "learning_rate") else 0.1,
            )

        # Train the model — returns (train_score, test_score, training_history)
        train_start = datetime.now()
        result = ml_strategy.train(data)
        training_time = (datetime.now() - train_start).total_seconds()

        # Unpack: strategies now return 3 values (backward compat for 2-value returns)
        if len(result) == 3:
            train_score, test_score, training_history = result
        else:
            train_score, test_score = result
            training_history = []

        # Get feature importance (not available for all model types, e.g. LSTM)
        feature_importance = []
        if hasattr(ml_strategy, "get_feature_importance"):
            feature_importance_df = ml_strategy.get_feature_importance()
            if not feature_importance_df.empty:
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
            "name": f"{config.symbol}_{model_type_slug}_v1",
            "type": model_type_slug,
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
                "n_estimators": getattr(ml_strategy, "n_estimators", 0),
                "max_depth": getattr(ml_strategy, "max_depth", 0),
                "learning_rate": getattr(ml_strategy, "learning_rate", 0),
                "lookback": getattr(ml_strategy, "lookback", 0),
                "epochs": getattr(ml_strategy, "epochs", 0),
            },
            "training_history": training_history,
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
                _save_metadata_to_disk(mid, meta)

        # Deploy this model
        model_registry[model_id]["status"] = "deployed"
        _save_metadata_to_disk(model_id, model_registry[model_id])

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

        # Delete model file and metadata sidecar
        model_path = metadata["model_path"]
        if os.path.exists(model_path):
            os.remove(model_path)
        meta_path = os.path.join(MODELS_DIR, f"{model_id}.json")
        if os.path.exists(meta_path):
            os.remove(meta_path)

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
