"""
Market data routes
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.deps import get_current_active_user
from backend.app.database import get_db
from backend.app.models.user import User
from backend.app.services.market_service import get_market_service

router = APIRouter(prefix="/market", tags=["Market Data"])

# Get singleton market service instance
market_service = get_market_service()


@router.get("/quote/{symbol}")
async def get_quote(symbol: str, use_cache: bool = True, current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)):
    """
    Get real-time quote for a symbol

    Args:
        symbol: Stock symbol
        use_cache: Whether to use cached data (default: True)
        current_user: User
        db: AsyncSession
    """
    try:
        quote = await market_service.get_quote(db, symbol, use_cache=use_cache)
        return quote
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch quote: {str(e)} for user: {current_user}")


@router.post("/quotes")
async def get_quotes(
    symbols: List[str], use_cache: bool = True, current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)
):
    """
    Get quotes for multiple symbols

    Args:
        symbols: List of stock symbols
        use_cache: Whether to use cached data
        current_user: User
        db: AsyncSession
    """
    try:
        quotes = await market_service.get_quotes(db, symbols, use_cache=use_cache)
        return quotes
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch quotes: {str(e)} for user: {current_user}")


@router.get("/historical/{symbol}")
async def get_historical_data(
    symbol: str,
    period: str = Query("1mo", description="Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)"),
    interval: str = Query("1d", description="Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)"),
    start: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    use_cache: bool = False,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get historical OHLCV data
    """
    try:
        data = await market_service.get_historical_data(
            db, symbol=symbol, period=period, interval=interval, start=start, end=end, use_cache=use_cache
        )

        if "data" in data and isinstance(data["data"], list):
            formatted_data = []
            for item in data["data"]:
                if isinstance(item, dict):
                    date_val = item.get("Date") or item.get("date") or item.get("timestamp")

                    # Convert Timestamp to ISO string
                    if hasattr(date_val, "isoformat"):
                        date_str = date_val.isoformat()
                    elif hasattr(date_val, "strftime"):
                        date_str = date_val.strftime("%Y-%m-%d")
                    else:
                        date_str = str(date_val)

                    formatted_data.append(
                        {
                            "date": date_str,
                            "timestamp": date_str,
                            "open": float(item.get("Open", 0)),
                            "high": float(item.get("High", 0)),
                            "low": float(item.get("Low", 0)),
                            "close": float(item.get("Close", 0)),
                            "volume": int(item.get("Volume", 0)),
                        }
                    )

            return {"symbol": symbol, "period": period, "interval": interval, "data": formatted_data}

        if "dataframe" in data:
            del data["dataframe"]

        return data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch historical data: {str(e)}")


@router.get("/options/{symbol}")
async def get_option_chain(
    symbol: str,
    expiration: Optional[str] = Query(None, description="Expiration date (YYYY-MM-DD)"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get option chain data

    Args:
        symbol: Stock symbol
        expiration: Optional specific expiration date
        current_user: User
        db: AsyncSession
    """
    try:
        data = await market_service.get_option_chain(symbol, expiration=expiration)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch option chain: {str(e)} for user: {current_user}")


@router.get("/fundamentals/{symbol}")
async def get_fundamentals(symbol: str, current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)):
    """
    Get fundamental data for a symbol

    Args:
        symbol: Stock symbol
        current_user: User
        db: AsyncSession
    """
    try:
        data = await market_service.get_fundamentals(db, symbol)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch fundamentals: {str(e)} for user: {current_user}")


@router.get("/news/{symbol}")
async def get_news(
    symbol: str,
    limit: int = Query(10, ge=1, le=50, description="Maximum number of news items"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get news for a symbol

    Args:
        symbol: Stock symbol
        limit: Maximum number of news items (1-50)
        current_user: User
        db: AsyncSession
    """
    try:
        news = await market_service.get_news(symbol, limit=limit)
        return news
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch news: {str(e)} for user: {current_user}")


@router.get("/search")
async def search_symbols(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(10, ge=1, le=50),
    current_user: User = Depends(get_current_active_user),
):
    """
    Search for symbols

    Args:
        q: Search query
        limit: Maximum results
        current_user: User
    """
    try:
        results = await market_service.search_symbols(q, limit=limit)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)} for user: {current_user}")


@router.get("/validate/{symbol}")
async def validate_symbol(symbol: str, current_user: User = Depends(get_current_active_user)):
    """
    Validate if a symbol exists

    Args:
        symbol: Stock symbol to validate
        current_user: User
    """
    try:
        is_valid = await market_service.validate_symbol(symbol)
        return {"symbol": symbol, "valid": is_valid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)} for user: {current_user}")


@router.get("/status")
async def get_market_status(current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)):
    """
    Get market status (open/closed)
    """
    try:
        status = await market_service.get_market_status(db)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get market status: {str(e)} for user: {current_user}")


@router.get("/cache/stats")
async def get_cache_stats(current_user: User = Depends(get_current_active_user)):
    """
    Get cache statistics
    """
    try:
        stats = market_service.cache.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)} for user: {current_user}")


@router.post("/cache/clear")
async def clear_cache(
    data_type: Optional[str] = Query(None, description="Data type to clear (quote, historical, fundamentals, etc.)"),
    current_user: User = Depends(get_current_active_user),
):
    """
    Clear market data cache

    Args:
        data_type: Optional data type to clear. If not specified, clears all.
        current_user: User
    """
    try:
        market_service.cache.clear(data_type)
        return {"status": "success", "message": f"Cache cleared: {data_type or 'all'}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)} for user: {current_user}")


@router.post("/cache/cleanup")
async def cleanup_cache(current_user: User = Depends(get_current_active_user)):
    """
    Clean up expired cache entries
    """
    try:
        market_service.cache.cleanup_expired()
        return {"status": "success", "message": "Expired cache entries cleaned up"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cleanup cache: {str(e)} for user: {current_user}")
