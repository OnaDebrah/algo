"""
Market data schemas
"""

from typing import List, Optional

from pydantic import BaseModel


class Quote(BaseModel):
    symbol: str
    price: float
    change: float
    changePct: float
    volume: int
    marketCap: Optional[int] = None
    high: Optional[float] = None
    low: Optional[float] = None
    open: Optional[float] = None
    previousClose: Optional[float] = None
    timestamp: str


class HistoricalDataPoint(BaseModel):
    Date: str
    Open: float
    High: float
    Low: float
    Close: float
    Volume: int


class HistoricalData(BaseModel):
    symbol: str
    data: List[dict]


class SymbolSearch(BaseModel):
    symbol: str
    name: str
    type: str
    exchange: str
