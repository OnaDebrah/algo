"""Main FastAPI application"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from backend.app.api.routes import (
    advisor,
    alerts,
    analyst,
    analytics,
    auth,
    backtest,
    health,
    market,
    marketplace,
    mlstudio,
    optimise,
    options,
    portfolio,
    regime,
    root,
    settings as settings_router,
    social,
    strategy,
    websocket,
)
from backend.app.api.routes.live import live
from backend.app.config import settings
from backend.app.database import AsyncSessionLocal, init_db
from backend.app.init_data import init_default_data
from backend.app.services.execution_manager import get_execution_manager, start_execution_manager, stop_execution_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    try:
        await init_db()
        await init_default_data()

        await start_execution_manager(AsyncSessionLocal)

        manager = get_execution_manager(AsyncSessionLocal)

        logger.info(f"Execution Manager started with {manager.get_executor_count()} strategies")
        logger.info("Platform ready for trading!")

    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise
    yield

    logger.info("SHUTTING DOWN TRADING PLATFORM")
    try:
        await stop_execution_manager()
        logger.info("All strategies stopped cleanly")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Oraculum Backtesting Platform API",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

app.include_router(auth.router)
app.include_router(backtest.router)
app.include_router(portfolio.router)
app.include_router(market.router)
app.include_router(strategy.router)
app.include_router(analytics.router)
app.include_router(regime.router)
app.include_router(websocket.router)
app.include_router(analyst.router)
app.include_router(advisor.router)
app.include_router(alerts.router)
app.include_router(live.router)
app.include_router(marketplace.router)
app.include_router(mlstudio.router)
app.include_router(options.router)
app.include_router(optimise.router)
app.include_router(social.router)
app.include_router(settings_router.router)
app.include_router(health.router)
app.include_router(root.router)
