import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.config import settings
from app.database import get_db
from app.main import app

# Create test engine using the primary DATABASE_URL since we rely on transaction rollbacks
test_engine = create_async_engine(settings.DATABASE_URL)


@pytest_asyncio.fixture(scope="session")
async def setup_db():
    """Session-wide setup for the database."""
    # Assuming Alembic migrations have already been run against the database.
    yield
    await test_engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def db_session(setup_db):
    """
    Creates an async database session for a test.
    Crucially, it wraps the session in a transaction and a nested savepoint.
    When the test finishes, the transaction is rolled back so the database
    is never permanently altered by tests.
    """
    async with test_engine.connect() as conn:
        # Start a transaction block
        await conn.begin()
        
        # Start a nested transaction (savepoint)
        await conn.begin_nested()

        # Bind an AsyncSession to this specific connection
        session = AsyncSession(conn, expire_on_commit=False)

        try:
            yield session
        finally:
            await session.close()
            # Rollback the outer transaction, destroying the savepoint and any changes
            await conn.rollback()


@pytest_asyncio.fixture(scope="function")
async def client(db_session):
    """
    Provides an async test client.
    Overrides the FastAPI `get_db` dependency to inject our isolated `db_session`.
    """
    
    # We must use an async generator for get_db
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac

    # Clean up overrides after the test
    app.dependency_overrides.clear()
