"""
API routes for watchlists and stock screener
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ...api.deps import get_current_active_user
from ...database import get_db
from ...models.user import User
from ...schemas.watchlist import (
    ScreenerFilter,
    ScreenerResult,
    WatchlistCreate,
    WatchlistItemAdd,
    WatchlistItemOut,
    WatchlistOut,
)
from ...services.watchlist_service import WatchlistService

router = APIRouter(tags=["Watchlist"])


# ── Watchlist CRUD ─────────────────────────────────────────────────────────


@router.get("/watchlists", response_model=list[WatchlistOut])
async def get_watchlists(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Get all watchlists for the current user."""
    return await WatchlistService.get_watchlists(db, current_user.id)


@router.post("/watchlists", response_model=WatchlistOut, status_code=status.HTTP_201_CREATED)
async def create_watchlist(
    body: WatchlistCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Create a new watchlist."""
    return await WatchlistService.create_watchlist(db, current_user.id, body.name)


@router.delete("/watchlists/{watchlist_id}")
async def delete_watchlist(
    watchlist_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Delete a watchlist."""
    deleted = await WatchlistService.delete_watchlist(db, current_user.id, watchlist_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Watchlist not found")
    return {"message": "Watchlist deleted"}


# ── Watchlist Items ────────────────────────────────────────────────────────


@router.post("/watchlists/{watchlist_id}/symbols", response_model=WatchlistItemOut, status_code=status.HTTP_201_CREATED)
async def add_symbol(
    watchlist_id: int,
    body: WatchlistItemAdd,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Add a symbol to a watchlist."""
    try:
        return await WatchlistService.add_symbol(db, watchlist_id, current_user.id, body.symbol, body.notes)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.delete("/watchlists/{watchlist_id}/symbols/{symbol}")
async def remove_symbol(
    watchlist_id: int,
    symbol: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Remove a symbol from a watchlist."""
    try:
        removed = await WatchlistService.remove_symbol(db, watchlist_id, current_user.id, symbol)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    if not removed:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Symbol not found in watchlist")
    return {"message": f"Symbol {symbol.upper()} removed"}


# ── Live Quotes ────────────────────────────────────────────────────────────


@router.get("/watchlists/{watchlist_id}/quotes")
async def get_watchlist_quotes(
    watchlist_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Get live quotes for all symbols in a watchlist."""
    try:
        return await WatchlistService.get_watchlist_quotes(db, watchlist_id, current_user.id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


# ── Screener ───────────────────────────────────────────────────────────────


@router.post("/screener", response_model=list[ScreenerResult])
async def run_screener(
    body: ScreenerFilter,
    current_user: User = Depends(get_current_active_user),
):
    """Run a stock screener with filters."""
    return await WatchlistService.screen_symbols(body)
