"""Schemas for webhook-based trade triggers."""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class WebhookSignal(BaseModel):
    """Incoming webhook signal (compatible with TradingView alert format)."""

    # Required fields
    ticker: str = Field(..., description="Ticker symbol (e.g., AAPL)")
    action: Literal["buy", "sell", "close"] = Field(..., description="Trade action")

    # Optional fields
    strategy_id: Optional[int] = Field(None, description="Target live strategy ID. If omitted, routes to first matching running strategy.")
    quantity: Optional[float] = Field(None, ge=0, description="Number of shares. If omitted, uses strategy default sizing.")
    price: Optional[float] = Field(None, ge=0, description="Limit price. If omitted, uses market order.")
    comment: Optional[str] = Field(None, max_length=500, description="Signal comment / reason")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional signal data")

    # TradingView compatibility
    interval: Optional[str] = Field(None, description="Chart interval (e.g., '1h', '1D')")
    exchange: Optional[str] = Field(None, description="Exchange (e.g., 'NASDAQ')")


class WebhookExecutionResponse(BaseModel):
    """Response after processing a webhook signal."""

    trade_id: Optional[int] = None
    strategy_id: int
    strategy_name: str
    ticker: str
    action: str
    status: Literal["executed", "queued", "rejected"]
    message: str
    timestamp: str


class WebhookLogEntry(BaseModel):
    """Webhook execution log entry."""

    id: int
    user_id: int
    strategy_id: Optional[int]
    signal: Dict[str, Any]
    result: Dict[str, Any]
    source_ip: str
    api_key_prefix: str
    created_at: str
