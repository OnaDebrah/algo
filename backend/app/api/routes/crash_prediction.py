"""
Crash Prediction API routes

Exposes ML-based crash prediction, market stress analysis,
hedge recommendations, and alert configuration.
"""

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...database import get_db
from ...models.crash_prediction import CrashPrediction
from ...models.user import User
from ...services.analysis.hedge_service import HedgeRecommendationService
from ...services.analysis.historical_accuracy_service import HistoricalAccuracyService
from ...services.analysis.lstm_stress_service import LSTMStressService
from ..deps import get_current_active_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/crash", tags=["Crash Prediction"])

# Singleton services
_hedge_service = None
_stress_service = None
_accuracy_service = None


def _get_hedge_service() -> HedgeRecommendationService:
    global _hedge_service
    if _hedge_service is None:
        _hedge_service = HedgeRecommendationService()
    return _hedge_service


def _get_stress_service() -> LSTMStressService:
    global _stress_service
    if _stress_service is None:
        _stress_service = LSTMStressService()
    return _stress_service


def _get_accuracy_service() -> HistoricalAccuracyService:
    global _accuracy_service
    if _accuracy_service is None:
        _accuracy_service = HistoricalAccuracyService()
    return _accuracy_service


@router.get("/predict/{symbol}")
async def predict_crash(
    symbol: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Run crash prediction for a symbol using the ML ensemble
    (LPPLS bubble detection + LSTM stress + combined probability)

    Returns crash probability, intensity classification, and individual model signals.
    """
    try:
        service = _get_hedge_service()
        crash_analysis = await service._get_crash_predictions(current_user, db, symbol)

        # Persist prediction to DB
        try:
            prediction = CrashPrediction(
                user_id=current_user.id,
                symbol=symbol.upper(),
                crash_probability=crash_analysis["combined_probability"],
                intensity=crash_analysis["intensity"],
                confidence=max(
                    crash_analysis["lppls"].get("confidence") or 0,
                    crash_analysis["lstm"].get("confidence") or 0,
                ),
                lppls_confidence=crash_analysis["lppls"].get("confidence") or 0,
                lppls_crash_probability=crash_analysis["lppls"].get("crash_probability") or 0,
                lppls_bubble_detected=crash_analysis["lppls"].get("is_bubble", False),
                lstm_stress_index=crash_analysis["lstm"].get("stress_index") or 0.5,
                lstm_confidence=crash_analysis["lstm"].get("confidence") or 0,
                lstm_stress_trend=crash_analysis["lstm"].get("stress_trend", "stable"),
                combined_score=crash_analysis["combined_probability"],
                meta_data=json.loads(json.dumps(crash_analysis, default=str)),
            )
            db.add(prediction)
            await db.commit()
        except Exception as e:
            logger.warning(f"Failed to persist crash prediction: {e}")
            try:
                await db.rollback()
            except Exception:
                pass

        return {
            "symbol": symbol.upper(),
            "crash_probability": crash_analysis["combined_probability"],
            "intensity": crash_analysis["intensity"],
            "confidence": max(
                crash_analysis["lppls"].get("confidence") or 0,
                crash_analysis["lstm"].get("confidence") or 0,
            ),
            "timestamp": crash_analysis["timestamp"],
            "lppls": crash_analysis["lppls"],
            "lstm": crash_analysis["lstm"],
            "combined_score": crash_analysis["combined_probability"],
        }

    except Exception as e:
        logger.error(f"Crash prediction failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Crash prediction failed: {str(e)}")


@router.get("/stress")
async def get_market_stress(
    symbols: str = Query(
        "SPY,QQQ,IWM",
        description="Comma-separated list of symbols to analyze",
    ),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Analyze market stress using LSTM model across multiple indices.

    Returns stress index, confidence, trend, 60-day forecast.
    """
    try:
        symbol_list = list(dict.fromkeys(s.strip().upper() for s in symbols.split(",") if s.strip()))
        if not symbol_list:
            symbol_list = ["SPY", "QQQ", "IWM"]

        service = _get_stress_service()
        result = await service.analyze_market_stress(
            symbols=symbol_list,
            user=current_user,
            db=db,
        )

        if "error" in result and result.get("stress_index") == 0.5:
            raise HTTPException(status_code=400, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Market stress analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Market stress analysis failed: {str(e)}")


@router.get("/hedge-recommendation")
async def get_hedge_recommendation(
    portfolio_value: float = Query(..., description="Total portfolio value", gt=0),
    portfolio_beta: float = Query(1.0, description="Portfolio beta"),
    primary_index: str = Query("SPY", description="Index to hedge against"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get hedge recommendation based on ML crash predictions.

    Returns recommended strategy, cost, protection level, ML signals, and monitoring instructions.
    """
    try:
        service = _get_hedge_service()
        recommendation = await service.get_hedge_recommendation(
            user=current_user,
            db=db,
            portfolio_value=portfolio_value,
            portfolio_beta=portfolio_beta,
            primary_index=primary_index.upper(),
        )
        return recommendation

    except Exception as e:
        logger.error(f"Hedge recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Hedge recommendation failed: {str(e)}")


@router.get("/history")
async def get_prediction_history(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get historical crash predictions for the current user.
    """
    try:
        query = (
            select(CrashPrediction)
            .where(CrashPrediction.user_id == current_user.id)
            .order_by(CrashPrediction.timestamp.desc())
            .limit(limit)
            .offset(offset)
        )

        if symbol:
            query = query.where(CrashPrediction.symbol == symbol.upper())

        result = await db.execute(query)
        predictions = result.scalars().all()

        return {
            "total": len(predictions),
            "offset": offset,
            "limit": limit,
            "predictions": [p.to_dict() for p in predictions],
        }

    except Exception as e:
        logger.error(f"Failed to fetch prediction history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")


@router.get("/dashboard/{symbol}")
async def get_crash_dashboard(
    symbol: str,
    portfolio_value: float = Query(100000, description="Portfolio value for hedge calc", gt=0),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get comprehensive crash prediction dashboard data.

    Aggregates crash prediction, stress analysis, and hedge recommendation
    in a single call for the frontend dashboard.
    """
    try:
        hedge_service = _get_hedge_service()
        stress_service = _get_stress_service()

        # Run prediction and stress analysis in parallel
        prediction_task = hedge_service._get_crash_predictions(current_user, db, symbol.upper())
        # Deduplicate symbols to avoid duplicate column names in combined DataFrame
        stress_symbols = list(dict.fromkeys([symbol.upper(), "SPY", "QQQ"]))
        stress_task = stress_service.analyze_market_stress(
            symbols=stress_symbols,
            user=current_user,
            db=db,
        )

        prediction_result, stress_result = await asyncio.gather(prediction_task, stress_task, return_exceptions=True)

        # Handle prediction result
        prediction_data = {}
        if isinstance(prediction_result, Exception):
            logger.warning(f"Crash prediction failed in dashboard: {prediction_result}")
            prediction_data = {
                "crash_probability": 0,
                "intensity": "unknown",
                "lppls": {},
                "lstm": {},
                "combined_probability": 0,
                "error": str(prediction_result),
            }
        else:
            prediction_data = prediction_result

        # Handle stress result
        stress_data = {}
        if isinstance(stress_result, Exception):
            logger.warning(f"Stress analysis failed in dashboard: {stress_result}")
            stress_data = {
                "stress_index": 0.5,
                "confidence": 0,
                "trend": "unknown",
                "error": str(stress_result),
            }
        else:
            stress_data = stress_result

        hedge_data = {}
        try:
            hedge_data = await hedge_service.get_hedge_recommendation(
                user=current_user,
                db=db,
                portfolio_value=portfolio_value,
                primary_index=symbol.upper() if symbol.upper() in ("SPY", "QQQ", "IWM", "DIA") else "SPY",
            )
        except Exception as e:
            logger.warning(f"Hedge recommendation failed in dashboard: {e}")
            hedge_data = {"strategy": "none", "error": str(e)}

        history = []
        try:
            history_query = (
                select(CrashPrediction)
                .where(
                    CrashPrediction.user_id == current_user.id,
                    CrashPrediction.symbol == symbol.upper(),
                )
                .order_by(CrashPrediction.timestamp.desc())
                .limit(30)
            )
            result = await db.execute(history_query)
            predictions = result.scalars().all()
            history = [p.to_dict() for p in predictions]
        except Exception as e:
            logger.warning(f"Failed to fetch history for dashboard: {e}")

        try:
            crash_prob = prediction_data.get("combined_probability", 0)
            prediction = CrashPrediction(
                user_id=current_user.id,
                symbol=symbol.upper(),
                crash_probability=crash_prob,
                intensity=prediction_data.get("intensity", "unknown"),
                confidence=max(
                    prediction_data.get("lppls", {}).get("confidence") or 0,
                    prediction_data.get("lstm", {}).get("confidence") or 0,
                ),
                lppls_confidence=prediction_data.get("lppls", {}).get("confidence") or 0,
                lppls_crash_probability=prediction_data.get("lppls", {}).get("crash_probability") or 0,
                lppls_bubble_detected=prediction_data.get("lppls", {}).get("is_bubble", False),
                lstm_stress_index=prediction_data.get("lstm", {}).get("stress_index") or 0.5,
                lstm_confidence=prediction_data.get("lstm", {}).get("confidence") or 0,
                lstm_stress_trend=prediction_data.get("lstm", {}).get("stress_trend", "stable"),
                combined_score=crash_prob,
                hedge_strategy=hedge_data.get("strategy"),
                hedge_cost=hedge_data.get("cost"),
                meta_data=json.loads(
                    json.dumps(
                        {
                            "prediction": prediction_data,
                            "stress": {k: v for k, v in stress_data.items() if k != "stress_history"},
                        },
                        default=str,
                    )
                ),
            )
            db.add(prediction)
            await db.commit()
        except Exception as e:
            logger.error(f"Failed to persist dashboard prediction: {e}")
            try:
                await db.rollback()
            except Exception as e:
                logger.error(f"Failed to rollback: {e}")
                pass

        return {
            "symbol": symbol.upper(),
            "prediction": {
                "crash_probability": prediction_data.get("combined_probability", 0),
                "intensity": prediction_data.get("intensity", "unknown"),
                "confidence": max(
                    prediction_data.get("lppls", {}).get("confidence") or 0,
                    prediction_data.get("lstm", {}).get("confidence") or 0,
                ),
                "timestamp": prediction_data.get("timestamp"),
                "lppls": prediction_data.get("lppls", {}),
                "lstm": prediction_data.get("lstm", {}),
                "combined_score": prediction_data.get("combined_probability", 0),
            },
            "stress": stress_data,
            "hedge_recommendation": hedge_data,
            "history": history,
        }

    except Exception as e:
        logger.error(f"Crash dashboard failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Dashboard failed: {str(e)}")


@router.post("/alert/configure")
async def configure_crash_alerts(
    crash_threshold: float = Query(0.33, ge=0, le=1, description="Crash probability threshold for alerts"),
    stress_threshold: float = Query(0.7, ge=0, le=1, description="Stress index threshold for alerts"),
    email_enabled: bool = Query(True, description="Enable email alerts"),
    sms_enabled: bool = Query(False, description="Enable SMS alerts"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Configure crash alert preferences for the current user.
    """
    try:
        # Store preferences in user settings
        from ...models.user_settings import UserSettings

        result = await db.execute(select(UserSettings).where(UserSettings.user_id == current_user.id))
        settings = result.scalar_one_or_none()

        crash_alert_config = {
            "crash_threshold": crash_threshold,
            "stress_threshold": stress_threshold,
            "email_enabled": email_enabled,
            "sms_enabled": sms_enabled,
        }

        if settings:
            # Update existing settings
            existing_prefs = settings.alert_preferences or {}
            existing_prefs["crash_alerts"] = crash_alert_config
            settings.alert_preferences = existing_prefs
        else:
            # Create new settings
            new_settings = UserSettings(
                user_id=current_user.id,
                alert_preferences={"crash_alerts": crash_alert_config},
            )
            db.add(new_settings)

        await db.commit()

        return {
            "success": True,
            "message": "Crash alert preferences updated",
            "config": crash_alert_config,
        }

    except Exception as e:
        logger.error(f"Failed to configure crash alerts: {e}")
        try:
            await db.rollback()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Failed to configure alerts: {str(e)}")


@router.get("/accuracy/{symbol}")
async def get_historical_accuracy(
    symbol: str,
    stride: int = Query(20, ge=1, le=60, description="Days between evaluation points (default 20 = ~monthly)"),
    threshold: float = Query(0.33, ge=0.0, le=1.0, description="Crash signal threshold"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Run historical crash prediction backtest across 18+ years of data.

    Evaluates LPPLS + LSTM models against known crash events (GFC, COVID, etc.)
    and returns accuracy metrics, per-event breakdown, and time-series data for charting.

    First request computes the backtest (~5-10 min). Results are cached for 24h.
    """
    try:
        service = _get_accuracy_service()
        result = await service.run_historical_accuracy(
            symbol=symbol.upper(),
            user=current_user,
            db=db,
            stride_days=stride,
            threshold=threshold,
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Historical accuracy failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Accuracy backtest failed: {str(e)}")
