"""
Market regime routes - FIXED VERSION
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException

from backend.app.analytics.market_regime_detector import MarketRegimeDetector
from backend.app.api.deps import get_current_active_user
from backend.app.models.user import User
from core.data_fetcher import fetch_stock_data

router = APIRouter(prefix="/regime", tags=["Market Regime"])

# Initialize detector (consider making this a singleton or dependency)
_detector_cache = {}


def get_detector(symbol: str) -> MarketRegimeDetector:
    """Get or create a MarketRegimeDetector instance for a symbol"""
    if symbol not in _detector_cache:
        _detector_cache[symbol] = MarketRegimeDetector(lookback_period=252, primary_index=symbol, use_ml=True, confidence_threshold=0.7)
    return _detector_cache[symbol]


@router.get("/detect/{symbol}")
async def detect_market_regime(symbol: str, period: str = "2y", current_user: User = Depends(get_current_active_user)):
    """Detect current market regime for a symbol"""
    try:
        # Fetch data
        data = fetch_stock_data(symbol, period=period, interval="1d")

        if data.empty:
            raise HTTPException(status_code=404, detail="No data available")

        # Get detector instance
        detector = get_detector(symbol)

        # Detect regime - returns a comprehensive dict
        regime_info = detector.detect_current_regime(
            price_data=data, volume_data=data.get("Volume") if "Volume" in data.columns else None, update_history=True
        )

        return {
            "symbol": symbol,
            "regime": regime_info["regime"],
            "confidence": regime_info["confidence"],
            "strategy_allocation": regime_info.get("strategy_allocation", {}),
            "regime_strength": regime_info.get("regime_strength", 0),
            "change_warning": regime_info.get("change_warning", {}),
            "duration_prediction": regime_info.get("duration_prediction"),
            "scores": regime_info.get("scores", {}),
            "method": regime_info.get("method", "unknown"),
            "timestamp": data.index[-1].isoformat(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting regime: {str(e)}")


@router.get("/history/{symbol}")
async def get_regime_history_data(symbol: str, period: str = "2y", current_user: User = Depends(get_current_active_user)):
    """Get historical regime changes"""
    try:
        # Fetch data
        data = fetch_stock_data(symbol, period=period, interval="1d")

        if data.empty:
            raise HTTPException(status_code=404, detail="No data available")

        # Get detector instance
        detector = get_detector(symbol)

        # Process historical data to build regime history
        detector.detect_current_regime(price_data=data, volume_data=data.get("Volume") if "Volume" in data.columns else None, update_history=True)

        # Format history for API response
        history = [
            {
                "timestamp": entry["timestamp"].isoformat() if hasattr(entry["timestamp"], "isoformat") else str(entry["timestamp"]),
                "regime": entry["regime"],
                "confidence": entry["confidence"],
                "strength": entry.get("strength", 0),
                "allocation": entry.get("allocation", {}),
            }
            for entry in detector.regime_history[-100:]  # Last 100 entries
        ]

        return {"symbol": symbol, "history": history, "total_entries": len(detector.regime_history)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching regime history: {str(e)}")


@router.get("/report/{symbol}")
async def get_regime_report(symbol: str, period: str = "2y", current_user: User = Depends(get_current_active_user)):
    """Get comprehensive regime analysis report"""
    try:
        # Fetch data
        data = fetch_stock_data(symbol, period=period, interval="1d")

        if data.empty:
            raise HTTPException(status_code=404, detail="No data available")

        # Get detector instance
        detector = get_detector(symbol)

        # Ensure regime history is populated
        detector.detect_current_regime(price_data=data, volume_data=data.get("Volume") if "Volume" in data.columns else None, update_history=True)

        # Generate comprehensive report
        report = detector.generate_regime_report()

        return {"symbol": symbol, "report": report, "timestamp": data.index[-1].isoformat()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")


@router.post("/batch")
async def detect_batch_regimes(symbols: List[str], period: str = "2y", current_user: User = Depends(get_current_active_user)):
    """Detect regimes for multiple symbols"""
    results = []
    errors = []

    for symbol in symbols:
        try:
            data = fetch_stock_data(symbol, period=period, interval="1d")

            if not data.empty:
                detector = get_detector(symbol)

                regime_info = detector.detect_current_regime(
                    price_data=data,
                    volume_data=data.get("Volume") if "Volume" in data.columns else None,
                    update_history=False,  # Don't update history for batch operations
                )

                results.append(
                    {
                        "symbol": symbol,
                        "regime": regime_info["regime"],
                        "confidence": regime_info["confidence"],
                        "strategy_allocation": regime_info.get("strategy_allocation", {}),
                        "regime_strength": regime_info.get("regime_strength", 0),
                        "method": regime_info.get("method", "unknown"),
                    }
                )
            else:
                errors.append({"symbol": symbol, "error": "No data available"})

        except Exception as e:
            errors.append({"symbol": symbol, "error": str(e)})
            continue

    return {"results": results, "errors": errors, "total_requested": len(symbols), "successful": len(results), "failed": len(errors)}


@router.post("/train/{symbol}")
async def train_ml_model(symbol: str, period: str = "5y", current_user: User = Depends(get_current_active_user)):
    """Train ML model for regime detection on historical data"""
    try:
        # Fetch extended historical data for training
        data = fetch_stock_data(symbol, period=period, interval="1d")

        if data.empty or len(data) < 500:
            raise HTTPException(status_code=400, detail="Insufficient data for training (need at least 500 days)")

        # Get detector instance
        detector = get_detector(symbol)

        # Train the model
        detector.train_ml_model(historical_data=data, volume_data=data.get("Volume") if "Volume" in data.columns else None)

        return {
            "symbol": symbol,
            "status": "trained",
            "training_samples": len(data),
            "training_date": data.index[-1].isoformat(),
            "message": "ML model trained successfully",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")


@router.get("/warning/{symbol}")
async def get_regime_change_warning(symbol: str, period: str = "2y", current_user: User = Depends(get_current_active_user)):
    """Get early warning signals for potential regime changes"""
    try:
        # Fetch data
        data = fetch_stock_data(symbol, period=period, interval="1d")

        if data.empty:
            raise HTTPException(status_code=404, detail="No data available")

        # Get detector instance
        detector = get_detector(symbol)

        # Calculate features
        features = detector.calculate_features(price_data=data, volume_data=data.get("Volume") if "Volume" in data.columns else None)

        # Get current regime
        current_regime = detector.detect_current_regime(
            price_data=data, volume_data=data.get("Volume") if "Volume" in data.columns else None, update_history=True
        )

        # Get warning signals
        warning = detector.detect_regime_change_warning(features)

        return {"symbol": symbol, "current_regime": current_regime["regime"], "warning": warning, "timestamp": data.index[-1].isoformat()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking warnings: {str(e)}")


@router.delete("/cache/{symbol}")
async def clear_detector_cache(symbol: str, current_user: User = Depends(get_current_active_user)):
    """Clear cached detector for a symbol (useful for resetting)"""
    if symbol in _detector_cache:
        del _detector_cache[symbol]
        return {"status": "cleared", "symbol": symbol}
    return {"status": "not_found", "symbol": symbol}


@router.delete("/cache")
async def clear_all_cache(current_user: User = Depends(get_current_active_user)):
    """Clear all cached detectors"""
    count = len(_detector_cache)
    _detector_cache.clear()
    return {"status": "cleared", "count": count}
