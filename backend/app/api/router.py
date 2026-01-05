"""
Main API router
"""

from fastapi import APIRouter

from .routes import advisor, auth, backtest, data, marketplace, portfolio, regime, strategies

api_router = APIRouter()

# Include all route modules
api_router.include_router(auth.router)
api_router.include_router(backtest.router)
api_router.include_router(portfolio.router)
api_router.include_router(strategies.router)
api_router.include_router(advisor.router)
api_router.include_router(marketplace.router)
api_router.include_router(regime.router)
api_router.include_router(data.router)
