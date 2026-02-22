"""
Market regime routes - FIXED VERSION
"""

from typing import List

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.analytics.market_regime_detector import MarketRegimeDetector
from backend.app.api.deps import get_current_active_user
from backend.app.core import fetch_stock_data
from backend.app.database import get_db
from backend.app.models.user import User
from backend.app.schemas.regime import (
    AllocationResponse,
    CurrentRegimeResponse,
    FeatureImportance,
    FeaturesResponse,
    RegimeData,
    RegimeMetrics,
    RegimeStrengthResponse,
    StrategyAllocation,
    TransitionProbability,
    TransitionResponse,
)
from backend.app.services.auth_service import AuthService

router = APIRouter(prefix="/regime", tags=["Market Regime"])

# Initialize detector (consider making this a singleton or dependency)
_detector_cache = {}


def get_detector(symbol: str) -> MarketRegimeDetector:
    """Get or create a MarketRegimeDetector instance for a symbol"""
    if symbol not in _detector_cache:
        _detector_cache[symbol] = MarketRegimeDetector(lookback_period=252, primary_index=symbol, use_ml=True, confidence_threshold=0.7)
    return _detector_cache[symbol]


@router.get("/detect/{symbol}", response_model=CurrentRegimeResponse)
async def detect_market_regime(
    symbol: str, period: str = "2y", current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)
):
    """Detect current market regime for a symbol"""
    await AuthService.track_usage(db, current_user.id, "detect_market_regime", {"symbol": symbol})

    try:
        # Fetch data
        data = await fetch_stock_data(symbol, period=period, interval="1d")

        if data.empty:
            raise HTTPException(status_code=404, detail="No data available")

        # Get detector instance
        detector = get_detector(symbol)

        # Detect regime - returns a comprehensive dict
        volume_data = data.get("Volume") if "Volume" in data.columns else None

        regime_info = detector.detect_current_regime(price_data=data, volume_data=volume_data, update_history=True)

        # Compute real metrics from features (cached, no extra cost)
        features = detector.calculate_features(price_data=data, volume_data=volume_data)
        latest = features.iloc[-1] if len(features) > 0 else pd.Series()

        # NaN-safe accessor
        def _safe(key, default=0.0):
            val = latest.get(key, default)
            return float(val) if pd.notna(val) else float(default)

        # Liquidity: normalise volume_ma_ratio (1.0 = average volume)
        raw_vol_ratio = _safe("volume_ma_ratio", 1.0)
        liquidity_score = round(min(max(raw_vol_ratio / 2.0, 0.0), 1.0), 3)

        # Correlation: use avg_correlation if multi-asset, else proxy from vol + trend
        raw_corr = latest.get("avg_correlation", None)
        if raw_corr is not None and pd.notna(raw_corr):
            correlation_index = round(float(min(max(raw_corr, 0.0), 1.0)), 3)
        else:
            vol = _safe("volatility_21d", 0.2)
            trend = abs(_safe("trend_strength", 0))
            correlation_index = round(min(max((vol + trend) / 2, 0.0), 1.0), 3)

        metrics = RegimeMetrics(
            volatility=_safe("volatility_21d", 0.0),
            trend_strength=_safe("trend_strength", 0.0),
            liquidity_score=liquidity_score,
            correlation_index=correlation_index,
        )

        current_regime = RegimeData(
            id=regime_info["regime"].lower().replace(" ", "_"),
            name=regime_info["regime"],
            description=regime_info.get("description", f"Market is in {regime_info['regime']} state"),
            start_date=data.index[-1],  # Simplified
            end_date=None,
            confidence=regime_info["confidence"],
            metrics=metrics,
        )

        return CurrentRegimeResponse(
            symbol=symbol,
            current_regime=current_regime,
            historical_regimes=[],  # Populate if needed
            market_health_score=regime_info.get("regime_strength", 0) * 100,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting regime: {str(e)}")


@router.get("/history/{symbol}")
async def get_regime_history_data(
    symbol: str, period: str = "2y", current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)
):
    """Get historical regime changes"""
    await AuthService.track_usage(db, current_user.id, "get_regime_history_data", {"symbol": symbol})

    try:
        # Fetch data
        data = await fetch_stock_data(symbol, period=period, interval="1d")

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
async def get_regime_report(
    symbol: str, period: str = "2y", current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)
):
    """Get comprehensive regime analysis report"""
    await AuthService.track_usage(db, current_user.id, "get_regime_report", {"symbol": symbol})

    try:
        # Fetch data
        data = await fetch_stock_data(symbol, period=period, interval="1d")

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
async def detect_batch_regimes(
    symbols: List[str], period: str = "2y", current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)
):
    await AuthService.track_usage(db, current_user.id, "detect_batch_regimes", {"symbols": symbols})

    """Detect regimes for multiple symbols"""
    results = []
    errors = []

    for symbol in symbols:
        try:
            data = await fetch_stock_data(symbol, period=period, interval="1d")

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
async def train_ml_model(symbol: str, period: str = "5y", current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)):
    """Train ML model for regime detection on historical data"""
    await AuthService.track_usage(db, current_user.id, "train_ml_model", {"symbol": symbol})

    try:
        # Fetch extended historical data for training
        data = await fetch_stock_data(symbol, period=period, interval="1d")

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
async def get_regime_change_warning(
    symbol: str, period: str = "2y", current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)
):
    """Get early warning signals for potential regime changes"""
    await AuthService.track_usage(db, current_user.id, "get_regime_change_warning", {"symbol": symbol})

    try:
        # Fetch data
        data = await fetch_stock_data(symbol, period=period, interval="1d")

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
async def clear_detector_cache(symbol: str, current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)):
    """Clear cached detector for a symbol (useful for resetting)"""
    await AuthService.track_usage(db, current_user.id, "clear_detector_cache", {"symbol": symbol})

    if symbol in _detector_cache:
        del _detector_cache[symbol]
        return {"status": "cleared", "symbol": symbol}
    return {"status": "not_found", "symbol": symbol}


@router.delete("/cache")
async def clear_all_cache(current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)):
    """Clear all cached detectors"""
    await AuthService.track_usage(db, current_user.id, "clear_all_cache")

    count = len(_detector_cache)
    _detector_cache.clear()
    return {"status": "cleared", "count": count}


# ============================================================
# ENHANCED ENDPOINTS
# ============================================================
@router.get("/allocation/{symbol}", response_model=AllocationResponse)
async def get_strategy_allocation(
    symbol: str, period: str = "2y", current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)
):
    """
    Get recommended strategy allocation based on current market regime
    """
    await AuthService.track_usage(db, current_user.id, "get_strategy_allocation", {"symbol": symbol})

    try:
        # Fetch data
        data = await fetch_stock_data(symbol, period=period, interval="1d")

        if data.empty:
            raise HTTPException(status_code=404, detail="No data available")

        # Get detector instance
        detector = get_detector(symbol)

        # Detect current regime
        regime_info = detector.detect_current_regime(
            price_data=data, volume_data=data.get("Volume") if "Volume" in data.columns else None, update_history=True
        )

        # Get recommended allocation
        allocation = detector.get_strategy_allocation(regime_info["regime"], regime_info["confidence"])

        return AllocationResponse(
            symbol=symbol,
            current_regime=regime_info["regime"],
            confidence=regime_info["confidence"],
            allocation=StrategyAllocation(**allocation),
            timestamp=data.index[-1].isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting allocation: {str(e)}")


@router.get("/strength/{symbol}", response_model=RegimeStrengthResponse)
async def get_regime_strength(
    symbol: str, period: str = "2y", current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)
):
    """
    Calculate how strongly the current regime is presenting itself
    """
    await AuthService.track_usage(db, current_user.id, "get_regime_strength", {"symbol": symbol})

    try:
        # Fetch data
        data = await fetch_stock_data(symbol, period=period, interval="1d")

        if data.empty:
            raise HTTPException(status_code=404, detail="No data available")

        # Get detector instance
        detector = get_detector(symbol)

        # Calculate features
        features = detector.calculate_features(price_data=data, volume_data=data.get("Volume") if "Volume" in data.columns else None)

        # Get current regime
        regime_info = detector.detect_current_regime(
            price_data=data, volume_data=data.get("Volume") if "Volume" in data.columns else None, update_history=True
        )

        # Calculate regime strength
        strength = detector.calculate_regime_strength(features)

        # Count confirming signals
        scores = regime_info.get("scores", {})
        current_regime = regime_info["regime"]
        confirming = sum(1 for r, score in scores.items() if r == current_regime and score > 0.1)
        total = len(scores)

        # Description based on strength
        if strength > 0.8:
            description = "Very strong regime - high conviction positioning recommended"
        elif strength > 0.6:
            description = "Strong regime - normal positioning appropriate"
        elif strength > 0.4:
            description = "Moderate regime - consider reduced position sizes"
        else:
            description = "Weak regime - caution advised, possible transition"

        return RegimeStrengthResponse(
            symbol=symbol,
            current_regime=current_regime,
            strength=strength,
            confirming_signals=confirming,
            total_signals=total,
            description=description,
            timestamp=data.index[-1].isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating strength: {str(e)}")


@router.get("/transitions/{symbol}", response_model=TransitionResponse)
async def get_transition_probabilities(symbol: str, period: str = "2y", current_user: User = Depends(get_current_active_user), db=Depends(get_db)):
    """
    Get regime transition probabilities and expected duration
    """
    await AuthService.track_usage(db, current_user.id, "get_transition_probabilities", {"symbol": symbol})

    try:
        data = await fetch_stock_data(symbol, period=period, interval="1d")

        if data.empty:
            raise HTTPException(status_code=404, detail="No data available")

        detector = get_detector(symbol)

        # Ensure history is populated
        detector.detect_current_regime(price_data=data, volume_data=data.get("Volume") if "Volume" in data.columns else None, update_history=True)

        transition_matrix = detector.get_transition_probabilities()

        # Get expected duration
        duration_info = detector.predict_regime_duration()

        # Extract likely transitions for current regime
        likely_transitions = []
        if transition_matrix and len(detector.regime_history) > 0:
            current_regime = detector.regime_history[-1]["regime"]
            if current_regime in transition_matrix:
                for to_regime, prob in transition_matrix[current_regime].items():
                    if prob > 0.1:  # Only include significant probabilities
                        likely_transitions.append(TransitionProbability(from_regime=current_regime, to_regime=to_regime, probability=prob))

        likely_transitions.sort(key=lambda x: x.probability, reverse=True)

        return TransitionResponse(
            symbol=symbol,
            current_regime=detector.regime_history[-1]["regime"] if detector.regime_history else "unknown",
            expected_duration=round(duration_info["expected_duration"]) if duration_info else 0,
            median_duration=duration_info["median_duration"] if duration_info else 0,
            probability_end_next_week=duration_info["probability_end_next_week"] if duration_info else 0,
            likely_transitions=likely_transitions[:5],  # Top 5
            timestamp=data.index[-1].isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating transitions: {str(e)}")


@router.get("/features/{symbol}", response_model=FeaturesResponse)
async def get_feature_analysis(symbol: str, period: str = "2y", current_user: User = Depends(get_current_active_user), db=Depends(get_db)):
    """
    Get feature importance analysis - what's driving the current regime
    """
    await AuthService.track_usage(db, current_user.id, "get_feature_analysis", {"symbol": symbol})

    try:
        # Fetch data
        data = await fetch_stock_data(symbol, period=period, interval="1d")

        if data.empty:
            raise HTTPException(status_code=404, detail="No data available")

        detector = get_detector(symbol)

        features = detector.calculate_features(price_data=data, volume_data=data.get("Volume") if "Volume" in data.columns else None)

        regime_info = detector.detect_current_regime(
            price_data=data, volume_data=data.get("Volume") if "Volume" in data.columns else None, update_history=True
        )

        if len(features) > 0:
            latest_features = features.iloc[-1]

            key_features = [
                "volatility_21d",
                "trend_strength",
                "hurst_exponent",
                "rsi",
                "macd",
                "half_life",
                "z_score",
                "advance_decline_ratio",
                "volume_ma_ratio",
            ]

            top_features = []
            for feat in key_features:
                if feat in latest_features.index:
                    value = latest_features[feat]
                    if pd.notna(value):
                        # Simple importance based on deviation from neutral
                        if feat == "volatility_21d":
                            importance = min(abs(value - 0.2) / 0.3, 1.0)
                        elif feat == "trend_strength":
                            importance = abs(value)
                        elif feat == "hurst_exponent":
                            importance = abs(value - 0.5) * 2
                        elif feat == "rsi":
                            importance = abs(value - 50) / 50
                        else:
                            importance = min(abs(value), 1.0)

                        top_features.append(FeatureImportance(feature=feat, importance=float(importance), current_value=float(value)))

            # Sort by importance
            top_features.sort(key=lambda x: x.importance, reverse=True)
        else:
            top_features = []

        return FeaturesResponse(
            symbol=symbol,
            current_regime=regime_info["regime"],
            top_features=top_features[:10],  # Top 10
            timestamp=data.index[-1].isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing features: {str(e)}")
