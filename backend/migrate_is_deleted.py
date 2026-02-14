
import asyncio
import logging
from sqlalchemy import text
from backend.app.database import engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def migrate():
    """Add is_deleted column to live_strategies table"""
    from backend.app.config import settings
    from sqlalchemy.ext.asyncio import create_async_engine
    
    urls_to_try = [settings.DATABASE_URL]
    # Also try the constructed one from components if different
    constructed_url = f"postgresql+asyncpg://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    if constructed_url != settings.DATABASE_URL:
        urls_to_try.append(constructed_url)

    for url in urls_to_try:
        logger.info(f"Trying to migrate using URL: {url.replace(settings.POSTGRES_PASSWORD, '****') if settings.POSTGRES_PASSWORD else url}")
        try:
            temp_engine = create_async_engine(url)
            async with temp_engine.begin() as conn:
                # Check if column exists first
                result = await conn.execute(text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name='live_strategies' AND column_name='is_deleted';"
                ))
                if not result.fetchone():
                    logger.info("Adding column 'is_deleted' to 'live_strategies' table...")
                    await conn.execute(text(
                        "ALTER TABLE live_strategies ADD COLUMN is_deleted BOOLEAN DEFAULT FALSE NOT NULL;"
                    ))
                    logger.info("Column added successfully.")
                else:
                    logger.info("Column 'is_deleted' already exists.")
            await temp_engine.dispose()
            return # Success
        except Exception as e:
            logger.error(f"Migration attempt failed for {url.split('@')[-1]}: {e}")
    
    logger.error("All migration attempts failed.")

if __name__ == "__main__":
    asyncio.run(migrate())
