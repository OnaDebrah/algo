# backend/app/core/data/macro/cache.py

import asyncio
import hashlib
import logging
import pickle
import zlib
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import and_, delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from .....models.market_data import MacroCacheEntry, RateLimitEntry

logger = logging.getLogger(__name__)


class MacroCache:
    """
    PostgreSQL-based caching system for macro data

    Features:
    - Persistent storage using your existing PostgreSQL
    - Async SQLAlchemy integration
    - Automatic cleanup of expired entries
    - Rate limit tracking
    - Compression for large datasets
    """

    def __init__(
        self,
        db: AsyncSession = None,
        default_ttl_hours: int = 24,
        compress_large: bool = True,
        compression_threshold: int = 1024 * 1024,  # 1MB
        enable_cleanup_scheduler: bool = True,
        cleanup_interval_hours: int = 6,
    ):
        self.db = db
        self.default_ttl = timedelta(hours=default_ttl_hours)
        self.compress_large = compress_large
        self.compression_threshold = compression_threshold

        # In-memory L1 cache (optional, for hot data)
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._memory_cache_size = 100
        self._memory_access_times: Dict[str, datetime] = {}

        # Statistics (use PostgreSQL for persistence)
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "bytes_saved": 0,
        }

        # Start cleanup scheduler if enabled
        if enable_cleanup_scheduler:
            self._start_cleanup_scheduler(cleanup_interval_hours)

        logger.info(f"MacroCache initialized with PostgreSQL (TTL={default_ttl_hours}h)")

    def _start_cleanup_scheduler(self, interval_hours: int):
        """Start background task to clean up expired entries"""

        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(interval_hours * 3600)
                    await self.cleanup_expired()
                except Exception as e:
                    logger.error(f"Cleanup scheduler error: {e}")

        asyncio.create_task(cleanup_loop())
        logger.info(f"Cleanup scheduler started (interval={interval_hours}h)")

    def _generate_cache_key(
        self,
        indicators: List[str],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        frequency: str,
        country: str = "USA",
        version: int = 1,
    ) -> str:
        """Generate unique cache key from parameters"""
        sorted_indicators = sorted(indicators) if indicators else []

        start_str = start_date.isoformat() if start_date else "none"
        end_str = end_date.isoformat() if end_date else "none"

        key_parts = ["macro", "_".join(sorted_indicators), start_str, end_str, frequency, country, f"v{version}"]

        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()[:64]

    async def get(
        self,
        indicators: List[str],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        frequency: str,
        country: str = "USA",
        max_age_hours: Optional[int] = None,
        use_memory_cache: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve cached macro data from PostgreSQL

        Args:
            indicators: List of indicator names
            start_date: Start date for data
            end_date: End date for data
            frequency: Data frequency
            country: Country code
            max_age_hours: Maximum age of data in hours
            use_memory_cache: Whether to use L1 memory cache

        Returns:
            DataFrame if found, None otherwise
        """
        cache_key = self._generate_cache_key(indicators, start_date, end_date, frequency, country)

        # Check memory cache first (L1)
        if use_memory_cache and cache_key in self._memory_cache:
            entry = self._memory_cache[cache_key]

            # Check expiration
            if not self._is_expired(entry, max_age_hours):
                self.stats["hits"] += 1
                self._update_access_time(cache_key)
                logger.debug(f"Memory cache hit: {cache_key}")
                return entry["data"]
            else:
                # Remove expired entry
                del self._memory_cache[cache_key]
                if cache_key in self._memory_access_times:
                    del self._memory_access_times[cache_key]

        # Check PostgreSQL (L2)
        async with self.db as session:
            try:
                # Build query
                query = select(MacroCacheEntry).where(MacroCacheEntry.cache_key == cache_key)

                # Apply max age filter if specified
                if max_age_hours:
                    min_created = datetime.utcnow() - timedelta(hours=max_age_hours)
                    query = query.where(MacroCacheEntry.created_at >= min_created)
                else:
                    # Apply TTL filter
                    query = query.where(and_(MacroCacheEntry.expires_at.is_(None), MacroCacheEntry.expires_at > datetime.utcnow()))

                result = await session.execute(query)
                entry = result.scalar_one_or_none()

                if entry:
                    # Deserialize data
                    df = self._deserialize_data(entry.data, entry.compression)

                    # Add to memory cache for next time
                    if use_memory_cache:
                        self._add_to_memory_cache(cache_key, df, entry)

                    self.stats["hits"] += 1
                    logger.debug(f"PostgreSQL cache hit: {cache_key} ({len(df)} rows)")
                    return df

            except Exception as e:
                logger.error(f"Error retrieving from cache: {e}")

        self.stats["misses"] += 1
        logger.debug(f"Cache miss: {cache_key}")
        return None

    async def set(
        self,
        indicators: List[str],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        frequency: str,
        data: pd.DataFrame,
        country: str = "USA",
        ttl_hours: Optional[int] = None,
        metadata: Optional[Dict] = None,
        use_memory_cache: bool = True,
    ):
        """
        Store macro data in PostgreSQL cache

        Args:
            indicators: List of indicator names
            start_date: Start date for data
            end_date: End date for data
            frequency: Data frequency
            data: DataFrame to cache
            country: Country code
            ttl_hours: Time-to-live in hours (None for default)
            metadata: Additional metadata to store
            use_memory_cache: Whether to also store in memory
        """
        if data.empty:
            logger.warning("Attempted to cache empty DataFrame")
            return

        cache_key = self._generate_cache_key(indicators, start_date, end_date, frequency, country)

        # Serialize data
        serialized, compression = self._serialize_data(data)

        # Calculate expiration
        ttl = timedelta(hours=ttl_hours or self.default_ttl.total_seconds() / 3600)
        expires_at = datetime.utcnow() + ttl

        # Prepare metadata
        meta = {
            "indicators": indicators,
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
            "frequency": frequency,
            "country": country,
            "rows": len(data),
            "columns": list(data.columns),
            "date_range": [
                data.index.min().isoformat() if isinstance(data.index, pd.DatetimeIndex) else None,
                data.index.max().isoformat() if isinstance(data.index, pd.DatetimeIndex) else None,
            ],
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": expires_at.isoformat(),
            **(metadata or {}),
        }

        # Store in PostgreSQL
        async with self.db as conn:
            try:
                # Use PostgreSQL upsert
                stmt = (
                    insert(MacroCacheEntry)
                    .values(
                        cache_key=cache_key,
                        data=serialized,
                        metadata=meta,
                        created_at=datetime.utcnow(),
                        expires_at=expires_at,
                        data_size=len(serialized),
                        compression=compression,
                        version=1,
                    )
                    .on_conflict_do_update(
                        index_elements=["cache_key"],
                        set_={
                            "data": serialized,
                            "metadata": meta,
                            "expires_at": expires_at,
                            "data_size": len(serialized),
                            "compression": compression,
                            "version": MacroCacheEntry.version + 1,
                        },
                    )
                )

                await conn.execute(stmt)
                await conn.commit()

                self.stats["sets"] += 1
                self.stats["bytes_saved"] += len(serialized)

                logger.debug(f"Cached data for {cache_key}: {len(data)} rows, expires {expires_at}")

            except Exception as e:
                logger.error(f"Error storing in cache: {e}")
                await conn.rollback()

        # Store in memory cache if enabled
        if use_memory_cache:
            self._add_to_memory_cache(cache_key, data, meta, expires_at)

    def _serialize_data(self, df: pd.DataFrame) -> Tuple[bytes, Optional[str]]:
        """Serialize DataFrame for storage"""
        # Use pickle for complex objects
        serialized = pickle.dumps(df)
        compression = None

        # Compress if needed
        if self.compress_large and len(serialized) > self.compression_threshold:
            serialized = zlib.compress(serialized)
            compression = "zlib"

        return serialized, compression

    def _deserialize_data(self, data: bytes, compression: Optional[str]) -> pd.DataFrame:
        """Deserialize DataFrame from storage"""
        try:
            if compression == "zlib":
                data = zlib.decompress(data)

            return pickle.loads(data)
        except Exception as e:
            logger.error(f"Error deserializing data: {e}")
            return pd.DataFrame()

    def _add_to_memory_cache(
        self,
        cache_key: str,
        df: pd.DataFrame,
        entry: Optional[MacroCacheEntry] = None,
        metadata: Optional[Dict] = None,
        expires_at: Optional[datetime] = None,
    ):
        """Add item to memory cache with LRU management"""

        # Check if we need to evict
        if len(self._memory_cache) >= self._memory_cache_size:
            self._evict_lru()

        # Store in memory
        self._memory_cache[cache_key] = {
            "data": df,
            "metadata": metadata or (entry.metadata if entry else {}),
            "expires_at": expires_at or (entry.expires_at if entry else None),
            "size_bytes": df.memory_usage(deep=True).sum(),
        }

        self._update_access_time(cache_key)

    def _update_access_time(self, cache_key: str):
        """Update last access time for LRU"""
        self._memory_access_times[cache_key] = datetime.utcnow()

    def _evict_lru(self):
        """Evict least recently used item from memory cache"""
        if not self._memory_access_times:
            return

        # Find least recently accessed
        lru_key = min(self._memory_access_times.items(), key=lambda x: x[1])[0]

        # Remove from both structures
        self._memory_cache.pop(lru_key, None)
        self._memory_access_times.pop(lru_key, None)

        logger.debug(f"Evicted LRU item: {lru_key}")

    def _is_expired(self, entry: Dict, max_age_hours: Optional[int] = None) -> bool:
        """Check if cache entry is expired"""
        expires_at = entry.get("expires_at")

        if expires_at and datetime.utcnow() > expires_at:
            return True

        if max_age_hours:
            metadata = entry.get("metadata", {})
            created_str = metadata.get("created_at")
            if created_str:
                created_at = datetime.fromisoformat(created_str)
                if datetime.utcnow() - created_at > timedelta(hours=max_age_hours):
                    return True

        return False

    # ==================== RATE LIMIT TRACKING ====================

    async def update_rate_limit(self, provider: str, remaining: int, reset_at: datetime):
        """Track API rate limits in PostgreSQL"""
        async with self.db as db:
            try:
                stmt = (
                    insert(RateLimitEntry)
                    .values(provider=provider, remaining=remaining, reset_at=reset_at, last_updated=datetime.utcnow())
                    .on_conflict_do_update(
                        index_elements=["provider"], set_={"remaining": remaining, "reset_at": reset_at, "last_updated": datetime.utcnow()}
                    )
                )

                await db.execute(stmt)
                await db.commit()

                logger.debug(f"Updated rate limit for {provider}: {remaining} remaining, resets at {reset_at}")

            except Exception as e:
                logger.error(f"Error updating rate limit: {e}")
                await db.rollback()

    async def check_rate_limit(self, provider: str) -> Tuple[bool, Optional[int]]:
        """
        Check if provider has rate limit capacity

        Returns:
            Tuple of (has_capacity, seconds_until_reset)
        """
        async with self.db as db:
            try:
                result = await db.execute(select(RateLimitEntry).where(RateLimitEntry.provider == provider))
                entry = result.scalar_one_or_none()

                if not entry:
                    return True, None

                if entry.remaining > 0:
                    return True, None

                # Check if reset time has passed
                if datetime.now(timezone.utc) > entry.reset_at:
                    return True, None

                # Rate limited, calculate seconds until reset
                seconds = (entry.reset_at - datetime.now(timezone.utc)).total_seconds()
                return False, int(max(1, seconds))

            except Exception as e:
                logger.error(f"Error checking rate limit: {e}")
                return True, None  # Assume we can proceed on error

    # ==================== CLEANUP & MAINTENANCE ====================

    async def cleanup_expired(self):
        """Remove expired entries from PostgreSQL"""
        async with self.db as db:
            try:
                # Delete expired entries
                result = await db.execute(delete(MacroCacheEntry).where(MacroCacheEntry.expires_at < datetime.utcnow()))
                await db.commit()

                deleted_count = result.rowcount

                # Also clean up old rate limit entries (keep only last 30 days)
                cutoff = datetime.utcnow() - timedelta(days=30)
                await db.execute(delete(RateLimitEntry).where(RateLimitEntry.last_updated < cutoff))
                await db.commit()

                logger.info(f"Cleaned up {deleted_count} expired cache entries")

            except Exception as e:
                logger.error(f"Error cleaning up cache: {e}")
                await db.rollback()

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        async with self.db as db:
            try:
                # Get total count and size
                count_result = await db.execute(select(func.count()).select_from(MacroCacheEntry))
                total_entries = count_result.scalar()

                size_result = await db.execute(select(func.sum(MacroCacheEntry.data_size)))
                total_size = size_result.scalar() or 0

                # Get oldest entry
                oldest_result = await db.execute(select(MacroCacheEntry.created_at).order_by(MacroCacheEntry.created_at).limit(1))
                oldest = oldest_result.scalar_one_or_none()

                # Get newest entry
                newest_result = await db.execute(select(MacroCacheEntry.created_at).order_by(MacroCacheEntry.created_at.desc()).limit(1))
                newest = newest_result.scalar_one_or_none()

                return {
                    "postgres": {
                        "total_entries": total_entries,
                        "total_size_mb": total_size / (1024 * 1024),
                        "oldest_entry": oldest.isoformat() if oldest else None,
                        "newest_entry": newest.isoformat() if newest else None,
                    },
                    "memory": {
                        "items": len(self._memory_cache),
                        "size_mb": sum(e.get("size_bytes", 0) for e in self._memory_cache.values()) / (1024 * 1024),
                        "hit_rate": self._calculate_hit_rate(),
                    },
                    "operations": {
                        "hits": self.stats["hits"],
                        "misses": self.stats["misses"],
                        "sets": self.stats["sets"],
                        "bytes_saved_mb": self.stats["bytes_saved"] / (1024 * 1024),
                    },
                }

            except Exception as e:
                logger.error(f"Error getting stats: {e}")
                return {}

    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.stats["hits"] + self.stats["misses"]
        if total == 0:
            return 0.0
        return (self.stats["hits"] / total) * 100

    async def invalidate(self, indicators: Optional[List[str]] = None, country: Optional[str] = None):
        """Invalidate cache entries matching criteria"""
        async with self.db as db:
            try:
                query = delete(MacroCacheEntry)

                if indicators or country:
                    # Need to filter by metadata
                    # This is a simplified version - in production you might want
                    # to add more indexes or use PostgreSQL JSON queries
                    entries = await db.execute(select(MacroCacheEntry))
                    to_delete = []

                    for entry in entries.scalars():
                        meta = entry.meta_data
                        if indicators and set(indicators) != set(meta.get("indicators", [])):
                            continue
                        if country and meta.get("country") != country:
                            continue
                        to_delete.append(entry.cache_key)

                    if to_delete:
                        await db.execute(delete(MacroCacheEntry).where(MacroCacheEntry.cache_key.in_(to_delete)))
                else:
                    # Clear all
                    await db.execute(query)

                await db.commit()

                # Clear memory cache too
                self._memory_cache.clear()
                self._memory_access_times.clear()

                logger.info("Cache invalidated")

            except Exception as e:
                logger.error(f"Error invalidating cache: {e}")
                await db.rollback()
