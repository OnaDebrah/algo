"""
Database configuration and session management
"""

import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool

from .config import settings

logger = logging.getLogger(__name__)

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DB_ECHO,  # Set to True for SQL query logging
    future=True,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_pre_ping=settings.DB_POOL_PRE_PING,  # Verify connections before using
    pool_recycle=settings.DB_POOL_RECYCLE,  # Recycle connections after N seconds
    # Use NullPool for serverless/lambda environments
    poolclass=NullPool if settings.ENVIRONMENT == "lambda" else None,
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Base class for all models
Base = declarative_base()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for getting async database sessions

    Usage:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(Item))
            return result.scalars().all()
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """
    Initialize database tables
    Call this on application startup
    """
    async with engine.begin() as conn:
        # Import all models here to ensure they're registered with Base

        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")


async def close_db() -> None:
    """
    Close database connections
    Call this on application shutdown
    """
    await engine.dispose()
    logger.info("Database connections closed")


async def check_db_connection() -> bool:
    """
    Check if database connection is working
    Returns True if connection is successful
    """
    try:
        async with AsyncSessionLocal() as session:
            await session.execute("SELECT 1")
            logger.info("Database connection successful")
            return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


async def reset_db() -> None:
    """
    Drop and recreate all tables
    ⚠️ WARNING: This will delete all data!
    Only use in development/testing
    """
    if settings.ENVIRONMENT == "production":
        raise RuntimeError("Cannot reset database in production!")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
        logger.warning("Database reset completed - all data deleted!")


class DatabaseSession:
    """
    Context manager for database sessions

    Usage:
        async with DatabaseSession() as db:
            result = await db.execute(select(User))
            users = result.scalars().all()
    """

    def __init__(self):
        self.session: AsyncSession | None = None

    async def __aenter__(self) -> AsyncSession:
        self.session = AsyncSessionLocal()
        return self.session

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            await self.session.rollback()
        else:
            await self.session.commit()
        await self.session.close()


async def get_or_create(db: AsyncSession, model, defaults=None, **kwargs):
    """
    Get an instance or create it if it doesn't exist

    Args:
        db: Database session
        model: SQLAlchemy model class
        defaults: Default values for creation
        **kwargs: Filter parameters

    Returns:
        Tuple of (instance, created)
    """
    from sqlalchemy import select

    result = await db.execute(select(model).filter_by(**kwargs))
    instance = result.scalar_one_or_none()

    if instance:
        return instance, False
    else:
        params = {**kwargs, **(defaults or {})}
        instance = model(**params)
        db.add(instance)
        await db.flush()
        return instance, True


async def bulk_create(db: AsyncSession, model, data_list: list[dict]) -> list:
    """
    Bulk create instances

    Args:
        db: Database session
        model: SQLAlchemy model class
        data_list: List of dictionaries with instance data

    Returns:
        List of created instances
    """
    instances = [model(**data) for data in data_list]
    db.add_all(instances)
    await db.flush()
    return instances


__all__ = [
    "engine",
    "AsyncSessionLocal",
    "Base",
    "get_db",
    "init_db",
    "close_db",
    "check_db_connection",
    "reset_db",
    "DatabaseSession",
    "get_or_create",
    "bulk_create",
]
