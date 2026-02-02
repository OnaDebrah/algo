"""Main FastAPI application"""

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
    live,
    market,
    marketplace,
    mlstudio,
    optimise,
    options,
    portfolio,
    regime,
    root,
    settings as settings_router,
    strategy,
    websocket,
)
from backend.app.config import settings
from backend.app.database import init_db
from backend.app.init_data import init_default_data


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    await init_db()
    await init_default_data()
    yield
    # Shutdown (cleanup if needed)


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
app.include_router(settings_router.router)
app.include_router(health.router)
app.include_router(root.router)
