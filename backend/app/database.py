"""
Database setup with SQLAlchemy
"""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.declarative import declarative_base

from backend.app.config import settings

SQLALCHEMY_DATABASE_URL = f"sqlite+aiosqlite:///./{settings.DATABASE_PATH}"

engine = create_async_engine(SQLALCHEMY_DATABASE_URL, echo=False, future=True)

AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()


async def get_db():
    """Database session dependency"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def create_tables():
    """Create all tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
