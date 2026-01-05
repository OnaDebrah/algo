"""
Market data routes
"""

from typing import List

from fastapi import APIRouter, Depends

from backend.app.api.deps import get_current_active_user
from backend.app.models.user import User
from backend.app.schemas.market import HistoricalData, Quote, SymbolSearch
from backend.app.services.market_service import MarketService

router = APIRouter(prefix="/market", tags=["Market"])
market_service = MarketService()


@router.get("/quote/{symbol}", response_model=Quote)
async def get_quote(symbol: str, current_user: User = Depends(get_current_active_user)):
    """Get real-time quote for a symbol"""
    quote = await market_service.get_quote(symbol)
    return Quote(**quote)


@router.post("/quotes", response_model=List[Quote])
async def get_quotes(symbols: List[str], current_user: User = Depends(get_current_active_user)):
    """Get quotes for multiple symbols"""
    quotes = await market_service.get_quotes(symbols)
    return [Quote(**q) for q in quotes]


@router.get("/historical/{symbol}", response_model=HistoricalData)
async def get_historical_data(symbol: str, period: str = "1mo", interval: str = "1d", current_user: User = Depends(get_current_active_user)):
    """Get historical data for a symbol"""
    data = await market_service.get_historical_data(symbol, period, interval)
    return HistoricalData(**data)


@router.get("/search", response_model=List[SymbolSearch])
async def search_symbols(q: str, current_user: User = Depends(get_current_active_user)):
    """Search for symbols"""
    results = await market_service.search_symbols(q)
    return [SymbolSearch(**r) for r in results]
