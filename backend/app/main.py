"""Main FastAPI application"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from backend.app.config import settings
from backend.app.database import create_tables
from backend.app.init_data import init_default_data


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    await create_tables()
    await init_default_data()
    yield
    # Shutdown (cleanup if needed)


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Advanced Trading Platform API",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
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

# Import and include routers
from backend.app.api.routes import (
    auth, backtest, portfolio, market,
    strategy, analytics, regime, websocket
)

app.include_router(auth.router, prefix="/api")
app.include_router(backtest.router, prefix="/api")
app.include_router(portfolio.router, prefix="/api")
app.include_router(market.router, prefix="/api")
app.include_router(strategy.router, prefix="/api")
app.include_router(analytics.router, prefix="/api")
app.include_router(regime.router, prefix="/api")
app.include_router(websocket.router, prefix="/api")


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "version": settings.VERSION}


@app.get("/")
async def root():
    return {
        "message": "Trading Platform API",
        "version": settings.VERSION,
        "docs": "/api/docs"
    }