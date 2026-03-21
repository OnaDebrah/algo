"""Watchlist and Screener schemas"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class WatchlistCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)


class WatchlistItemAdd(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)
    notes: Optional[str] = None


class WatchlistItemOut(BaseModel):
    id: int
    symbol: str
    notes: Optional[str] = None
    added_at: datetime

    class Config:
        from_attributes = True


class WatchlistOut(BaseModel):
    id: int
    name: str
    items: list[WatchlistItemOut] = []
    created_at: datetime

    class Config:
        from_attributes = True


class ScreenerFilter(BaseModel):
    symbols: list[str] = Field(
        default_factory=lambda: [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "TSLA",
            "META",
            "NVDA",
            "JPM",
            "V",
            "JNJ",
            "WMT",
            "PG",
            "UNH",
            "HD",
            "BAC",
            "XOM",
            "CVX",
            "PFE",
            "ABBV",
            "KO",
            "PEP",
            "MRK",
            "COST",
            "LLY",
            "TMO",
            "AVGO",
            "ORCL",
            "ACN",
            "CRM",
            "AMD",
        ]
    )
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_change_pct: Optional[float] = None
    max_change_pct: Optional[float] = None
    min_volume: Optional[int] = None


class ScreenerResult(BaseModel):
    symbol: str
    price: float
    change: float
    change_pct: float
    volume: int
    day_high: float
    day_low: float
    market_cap: Optional[float] = None
