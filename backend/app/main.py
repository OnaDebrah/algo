"""Main FastAPI application"""

import logging
from contextlib import asynccontextmanager

import sentry_sdk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pythonjsonlogger.jsonlogger import JsonFormatter

from .api.middleware.rate_limit import RateLimitMiddleware
from .api.routes import (
    advisor,
    alerts,
    analyst,
    analytics,
    auth,
    backtest,
    crash_prediction,
    health,
    market,
    marketplace,
    mlstudio,
    optimise,
    options,
    portfolio,
    regime,
    root,
    sector,
    settings as settings_router,
    social,
    strategy,
    websocket,
)
from .api.routes.live import live
from .config import settings
from .database import AsyncSessionLocal, init_db
from .init_data import init_default_data
from .services.execution_manager import get_execution_manager, start_execution_manager, stop_execution_manager

if settings.SENTRY_DSN:
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        traces_sample_rate=1.0,
        profiles_sample_rate=1.0,
    )

logHandler = logging.StreamHandler()
formatter = JsonFormatter(fmt="%(asctime)s %(name)s %(levelname)s %(message)s", rename_fields={"levelname": "level", "asctime": "timestamp"})
logHandler.setFormatter(formatter)
logging.basicConfig(handlers=[logHandler], level=settings.LOG_LEVEL)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    try:
        if settings.ENVIRONMENT != "test":
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
        if settings.ENVIRONMENT != "test":
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

if settings.ENFORCE_HTTPS:
    app.add_middleware(HTTPSRedirectMiddleware)

# Middleware order: last added = outermost (processes first)
# Desired execution order: CORS → GZip → RateLimit → App
app.add_middleware(RateLimitMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Prometheus Metrics
Instrumentator().instrument(app).expose(app)

# CORS must be outermost to handle preflight OPTIONS before anything else
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
app.include_router(sector.router)
app.include_router(crash_prediction.router)
app.include_router(root.router)
