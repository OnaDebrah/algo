#!/usr/bin/env python3
"""
Encrypt existing plaintext strategy code in the custom_strategies table.

Usage:
    python -m scripts.encrypt_existing_strategies          # from backend/
    python backend/scripts/encrypt_existing_strategies.py   # from repo root

Requires ENCRYPTION_KEY and DATABASE_URL in the environment (or .env file).

Processes rows where is_encrypted=False in batches of 100, encrypting the
code column and setting is_encrypted=True.  Idempotent — safe to re-run.
"""

import asyncio
import logging
import os
import sys

# Ensure the backend package is importable when invoked directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 100


async def encrypt_strategies() -> None:
    from sqlalchemy import select, func as sa_func
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from app.config import settings
    from app.models.custom_strategy import CustomStrategy
    from app.services.encryption_service import get_encryption_service

    engine = create_async_engine(settings.DATABASE_URL, future=True)
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    svc = get_encryption_service()

    async with SessionLocal() as db:
        # Total count for progress reporting
        count_result = await db.execute(
            select(sa_func.count()).select_from(CustomStrategy).where(CustomStrategy.is_encrypted == False)  # noqa: E712
        )
        total = count_result.scalar() or 0
        logger.info(f"Found {total} unencrypted strategies to process")

        if total == 0:
            logger.info("Nothing to do — all strategies are already encrypted.")
            await engine.dispose()
            return

        processed = 0
        while True:
            result = await db.execute(
                select(CustomStrategy)
                .where(CustomStrategy.is_encrypted == False)  # noqa: E712
                .limit(BATCH_SIZE)
            )
            batch = result.scalars().all()

            if not batch:
                break

            for strategy in batch:
                # Skip if already encrypted (defensive check via prefix)
                if svc.is_encrypted(strategy.code):
                    strategy.is_encrypted = True
                    processed += 1
                    continue

                strategy.code = svc.encrypt_code(strategy.code)
                strategy.is_encrypted = True
                processed += 1

            await db.commit()
            logger.info(f"  Encrypted {processed}/{total} strategies ...")

        logger.info(f"Done. {processed} strategies encrypted successfully.")

    await engine.dispose()


def main() -> None:
    asyncio.run(encrypt_strategies())


if __name__ == "__main__":
    main()
