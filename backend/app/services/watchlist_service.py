"""
Watchlist and screener service
"""

import asyncio
import logging
from typing import Optional

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..core.data.providers.providers import ProviderFactory
from ..models.watchlist import Watchlist, WatchlistItem
from ..schemas.watchlist import ScreenerFilter, ScreenerResult

logger = logging.getLogger(__name__)


class WatchlistService:
    """Service for managing watchlists and running screeners"""

    @staticmethod
    async def create_watchlist(db: AsyncSession, user_id: int, name: str) -> Watchlist:
        """Create a new watchlist for a user."""
        watchlist = Watchlist(user_id=user_id, name=name)
        db.add(watchlist)
        await db.commit()

        # Re-query with eager loading so `items` is available for serialization
        query = select(Watchlist).where(Watchlist.id == watchlist.id).options(selectinload(Watchlist.items))
        result = await db.execute(query)
        return result.scalar_one()

    @staticmethod
    async def get_watchlists(db: AsyncSession, user_id: int) -> list[Watchlist]:
        """Get all watchlists for a user with items eager-loaded."""
        query = select(Watchlist).where(Watchlist.user_id == user_id).options(selectinload(Watchlist.items)).order_by(Watchlist.created_at.desc())
        result = await db.execute(query)
        return list(result.scalars().all())

    @staticmethod
    async def delete_watchlist(db: AsyncSession, user_id: int, watchlist_id: int) -> bool:
        """Delete a watchlist if owned by the user. Returns True if deleted."""
        stmt = delete(Watchlist).where(
            Watchlist.id == watchlist_id,
            Watchlist.user_id == user_id,
        )
        result = await db.execute(stmt)
        await db.commit()
        return result.rowcount > 0

    @staticmethod
    async def add_symbol(
        db: AsyncSession,
        watchlist_id: int,
        user_id: int,
        symbol: str,
        notes: Optional[str] = None,
    ) -> WatchlistItem:
        """Add a symbol to a watchlist after verifying ownership."""
        # Verify ownership
        query = select(Watchlist).where(Watchlist.id == watchlist_id, Watchlist.user_id == user_id)
        result = await db.execute(query)
        watchlist = result.scalar_one_or_none()
        if not watchlist:
            raise ValueError("Watchlist not found or not owned by user")

        item = WatchlistItem(
            watchlist_id=watchlist_id,
            symbol=symbol.upper(),
            notes=notes,
        )
        db.add(item)
        await db.commit()
        await db.refresh(item)
        return item

    @staticmethod
    async def remove_symbol(db: AsyncSession, watchlist_id: int, user_id: int, symbol: str) -> bool:
        """Remove a symbol from a watchlist after verifying ownership. Returns True if removed."""
        # Verify ownership
        query = select(Watchlist).where(Watchlist.id == watchlist_id, Watchlist.user_id == user_id)
        result = await db.execute(query)
        watchlist = result.scalar_one_or_none()
        if not watchlist:
            raise ValueError("Watchlist not found or not owned by user")

        stmt = delete(WatchlistItem).where(
            WatchlistItem.watchlist_id == watchlist_id,
            WatchlistItem.symbol == symbol.upper(),
        )
        result = await db.execute(stmt)
        await db.commit()
        return result.rowcount > 0

    @staticmethod
    async def get_watchlist_quotes(db: AsyncSession, watchlist_id: int, user_id: int) -> list[dict]:
        """Get live quotes for all symbols in a watchlist."""
        # Verify ownership and get items
        query = select(Watchlist).where(Watchlist.id == watchlist_id, Watchlist.user_id == user_id).options(selectinload(Watchlist.items))
        result = await db.execute(query)
        watchlist = result.scalar_one_or_none()
        if not watchlist:
            raise ValueError("Watchlist not found or not owned by user")

        if not watchlist.items:
            return []

        provider = ProviderFactory()
        symbols = [item.symbol for item in watchlist.items]

        async def _fetch_quote(symbol: str) -> dict:
            try:
                raw = await provider.get_quote(symbol)

                # Normalize camelCase provider response to snake_case for frontend
                def _num(v: any, default=0) -> float:
                    """Safely convert to float, treating None/NaN as default."""
                    if v is None:
                        return default
                    try:
                        f = float(v)
                        return default if f != f else f  # NaN check
                    except (TypeError, ValueError):
                        return default

                return {
                    "symbol": raw.get("symbol", symbol),
                    "price": _num(raw.get("price")),
                    "change": _num(raw.get("change")),
                    "change_percent": _num(raw.get("changePercent")),
                    "volume": _num(raw.get("volume")),
                    "day_high": _num(raw.get("high", raw.get("dayHigh"))),
                    "day_low": _num(raw.get("low", raw.get("dayLow"))),
                    "name": raw.get("name", raw.get("shortName", "")),
                }
            except Exception as e:
                logger.warning(f"Failed to fetch quote for {symbol}: {e}")
                return {
                    "symbol": symbol,
                    "price": 0,
                    "change": 0,
                    "change_percent": 0,
                    "volume": 0,
                    "day_high": 0,
                    "day_low": 0,
                }

        quotes = await asyncio.gather(*[_fetch_quote(s) for s in symbols])
        return list(quotes)

    @staticmethod
    async def screen_symbols(filters: ScreenerFilter) -> list[ScreenerResult]:
        """Fetch quotes for symbols and apply screening filters."""
        provider = ProviderFactory()

        async def _fetch_quote(symbol: str) -> Optional[dict]:
            try:
                return await provider.get_quote(symbol)
            except Exception as e:
                logger.warning(f"Screener: failed to fetch {symbol}: {e}")
                return None

        raw_quotes = await asyncio.gather(*[_fetch_quote(s) for s in filters.symbols])

        results: list[ScreenerResult] = []
        for quote in raw_quotes:
            if quote is None or "error" in quote:
                continue

            price = quote.get("price", 0) or 0
            change = quote.get("change", 0) or 0
            change_pct = quote.get("changePercent", 0) or 0
            volume = quote.get("volume", 0) or 0

            # Apply filters
            if filters.min_price is not None and price < filters.min_price:
                continue
            if filters.max_price is not None and price > filters.max_price:
                continue
            if filters.min_change_pct is not None and change_pct < filters.min_change_pct:
                continue
            if filters.max_change_pct is not None and change_pct > filters.max_change_pct:
                continue
            if filters.min_volume is not None and volume < filters.min_volume:
                continue

            results.append(
                ScreenerResult(
                    symbol=quote.get("symbol", ""),
                    price=price,
                    change=change,
                    change_pct=change_pct,
                    volume=int(volume),
                    day_high=quote.get("high", 0) or 0,
                    day_low=quote.get("low", 0) or 0,
                    market_cap=quote.get("marketCap"),
                )
            )

        return results
