"""
Sector Scanner API routes — sector ranking, stock selection, strategy recommendations
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ...api.deps import get_current_active_user
from ...database import get_db
from ...models.user import User
from ...services.auth_service import AuthService
from ...services.sector_scanner_service import SectorScannerService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sector", tags=["Sector Scanner"])

_service: Optional[SectorScannerService] = None


def get_service() -> SectorScannerService:
    global _service
    if _service is None:
        _service = SectorScannerService()
    return _service


@router.get("/list")
async def list_sectors(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """List all sectors with metadata."""
    await AuthService.track_usage(db, current_user.id, "sector_list", {})

    try:
        service = get_service()
        return service.get_sectors()
    except Exception as e:
        logger.error(f"Error listing sectors: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scan")
async def scan_sectors(
    period: str = Query("6mo", description="Lookback period for ETF data"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Scan all sectors and rank by momentum and risk-adjusted returns.
    Uses sector ETF data (XLK, XLV, etc.) for analysis.
    """
    await AuthService.track_usage(db, current_user.id, "sector_scan", {"period": period})

    try:
        service = get_service()
        result = await service.scan_sectors(period=period)
        return result
    except Exception as e:
        logger.error(f"Error scanning sectors: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stocks/{sector}")
async def rank_sector_stocks(
    sector: str,
    top_n: int = Query(10, ge=1, le=50, description="Number of top stocks to return"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Rank stocks within a sector using multi-factor analysis.
    Uses momentum, volatility, quality, valuation, and growth factors.
    """
    await AuthService.track_usage(db, current_user.id, "sector_stocks", {"sector": sector, "top_n": top_n})

    try:
        service = get_service()
        ranked = await service.rank_stocks(sector=sector, top_n=top_n)
        return ranked
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error ranking stocks in {sector}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommend/{symbol}")
async def recommend_strategies(
    symbol: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Recommend trading strategies for a specific stock based on its current market regime.
    """
    await AuthService.track_usage(db, current_user.id, "sector_recommend", {"symbol": symbol})

    try:
        service = get_service()
        recommendations = await service.recommend_strategies(symbol=symbol.upper())
        return recommendations
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error recommending strategies for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
