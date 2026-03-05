import asyncio
import logging
import os
import sys
sys.path.insert(0, os.getcwd())

from app.database import engine
from sqlalchemy import text

logger = logging.getLogger(__name__)

async def main():
    async with engine.begin() as conn:
        await conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
        logger.info("Dropped alembic_version table")

if __name__ == "__main__":
    asyncio.run(main())
