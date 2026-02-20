import logging
import time
from typing import Any, Dict, Optional

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.market_data import MarketDataCacheModel

logger = logging.getLogger(__name__)


class MarketDataCache:
    def __init__(self, ttl_seconds: int = 60):
        self.memory_cache: Dict[str, tuple[Any, float]] = {}
        self.memory_ttl = ttl_seconds

    async def get(self, db: AsyncSession, key: str) -> Optional[Any]:
        # 1. Hot Memory Tier
        if key in self.memory_cache:
            value, timestamp = self.memory_cache[key]
            if time.time() - timestamp < self.memory_ttl:
                return value
            del self.memory_cache[key]

        # 2. Cold Database Tier
        current_time = time.time()
        stmt = select(MarketDataCacheModel).where(MarketDataCacheModel.cache_key == key, MarketDataCacheModel.expires_at > current_time)
        result = await db.execute(stmt)
        cache_obj = result.scalar_one_or_none()

        if cache_obj:
            # Update stats asynchronously (Fire and forget if needed, or await)
            cache_obj.access_count += 1
            cache_obj.last_accessed = current_time

            # Re-fill memory cache
            self.memory_cache[key] = (cache_obj.data_json, current_time)
            return cache_obj.data_json

        return None

    async def set(self, db: AsyncSession, key: str, value: Any, data_type: str = "quote", ttl_override: int = None):
        current_time = time.time()

        # 1. Update Memory
        self.memory_cache[key] = (value, current_time)

        # 2. Update Database
        ttl_map = {"quote": 60, "historical": 3600, "fundamentals": 86400, "options": 300}
        ttl = ttl_override or ttl_map.get(data_type, self.memory_ttl)
        expires_at = current_time + ttl

        # UPSERT Logic (On Conflict Update)
        from sqlalchemy.dialects.postgresql import insert

        stmt = (
            insert(MarketDataCacheModel)
            .values(
                cache_key=key,
                data_type=data_type,
                data_json=value,  # SQLAlchemy handles dict to JSONB conversion
                created_at=current_time,
                expires_at=expires_at,
                last_accessed=current_time,
            )
            .on_conflict_do_update(
                index_elements=["cache_key"], set_={"data_json": value, "expires_at": expires_at, "last_accessed": current_time, "access_count": 0}
            )
        )
        await db.execute(stmt)

    async def clear(self, db: AsyncSession, data_type: Optional[str] = None):
        """
        Clear cached data from memory and database
        """
        # 1. Clear memory cache
        if data_type:
            keys_to_remove = [k for k in self.memory_cache.keys() if k.startswith(data_type)]
            for key in keys_to_remove:
                del self.memory_cache[key]
        else:
            self.memory_cache.clear()

        # 2. Clear database cache
        stmt = delete(MarketDataCacheModel)
        if data_type:
            stmt = stmt.where(MarketDataCacheModel.data_type == data_type)

        await db.execute(stmt)
        # Note: No need for conn.commit(), get_db handles it.
        logger.info(f"Database cache cleared: {data_type or 'all'}")

    async def cleanup_expired(self, db: AsyncSession):
        """Remove expired entries from database"""
        current_time = time.time()
        stmt = delete(MarketDataCacheModel).where(MarketDataCacheModel.expires_at < current_time)

        result = await db.execute(stmt)
        deleted_count = result.rowcount

        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} expired cache entries")
        return deleted_count

    async def get_stats(self, db: AsyncSession) -> Dict:
        """Get cache statistics using async aggregation"""
        current_time = time.time()

        # 1. Base memory stats
        stats = {"memory_entries": len(self.memory_cache), "database_enabled": True}

        # 2. Total Count and Accesses
        count_stmt = select(
            func.count(MarketDataCacheModel.cache_key).label("count"),
            func.coalesce(func.sum(MarketDataCacheModel.access_count), 0).label("total_accesses"),
        ).where(MarketDataCacheModel.expires_at > current_time)

        result = await db.execute(count_stmt)
        summary = result.first()

        stats["database_entries"] = summary.count if summary else 0
        stats["total_accesses"] = summary.total_accesses if summary else 0

        # 3. Breakdown by data type
        type_stmt = (
            select(
                MarketDataCacheModel.data_type,
                func.count(MarketDataCacheModel.cache_key).label("count"),
                func.avg(MarketDataCacheModel.access_count).label("avg_accesses"),
            )
            .where(MarketDataCacheModel.expires_at > current_time)
            .group_by(MarketDataCacheModel.data_type)
        )

        type_result = await db.execute(type_stmt)

        stats["by_type"] = {
            row.data_type: {"count": row.count, "avg_accesses": float(row.avg_accesses) if row.avg_accesses else 0} for row in type_result.all()
        }

        return stats
