"""
Webhook routes for external trade signal execution.

Allows external tools (TradingView, custom scripts, etc.) to trigger
trades on running live/paper strategies via API key authentication.

Usage:
    curl -X POST https://api.oraculum.io/webhooks/signal \
         -H "X-API-Key: orc_abc123..." \
         -H "Content-Type: application/json" \
         -d '{"ticker": "AAPL", "action": "buy", "quantity": 10}'
"""

import hashlib
import logging
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...database import get_db
from ...models.api_key import ApiKey
from ...models.live import DeploymentMode, LiveStrategy, LiveTrade, StrategyStatus
from ...models.user import User
from ...schemas.webhooks import WebhookExecutionResponse, WebhookSignal

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["Webhooks"])


async def _authenticate_api_key(
    db: AsyncSession,
    api_key: str,
    request: Request | None = None,
) -> tuple:
    """Validate API key and return (user, api_key_record).

    Raises HTTPException on invalid/expired/inactive keys or blocked IPs.
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header.",
        )

    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    result = await db.execute(
        select(ApiKey).where(ApiKey.key_hash == key_hash, ApiKey.is_active == True)  # noqa: E712
    )
    key_record = result.scalar_one_or_none()

    if not key_record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked API key.",
        )

    # Check expiry
    if key_record.expires_at and key_record.expires_at < datetime.now(timezone.utc):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key has expired.",
        )

    # Check IP whitelist
    if key_record.allowed_ips and request:
        from .api_keys import _check_ip_allowed

        client_ip = request.client.host if request.client else None
        if not client_ip or not _check_ip_allowed(client_ip, key_record.allowed_ips):
            logger.warning(
                f"API key IP whitelist violation: key_prefix={key_record.key_prefix} client_ip={client_ip} allowed_ips={key_record.allowed_ips}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"IP address '{client_ip}' is not in the allowed IP list for this API key.",
            )

    # Check permissions
    permissions = key_record.permissions or []
    if "trade" not in permissions and "write" not in permissions:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key lacks 'trade' permission. Regenerate with trade permission in Settings.",
        )

    # Update last used timestamp
    key_record.last_used_at = datetime.now(timezone.utc)
    await db.commit()

    # Fetch the user
    user_result = await db.execute(select(User).where(User.id == key_record.user_id))
    user = user_result.scalar_one_or_none()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive.",
        )

    return user, key_record


@router.post("/signal", response_model=WebhookExecutionResponse)
async def receive_signal(
    signal: WebhookSignal,
    request: Request,
    x_api_key: str = Header(None, alias="X-API-Key"),
    db: AsyncSession = Depends(get_db),
):
    """
    Receive an external trade signal and route it to a running strategy.

    **Authentication:** Requires an API key with 'trade' permission.
    Pass it via the `X-API-Key` header.

    **TradingView Integration:**
    Set your TradingView alert webhook URL to:
    `https://your-domain/webhooks/signal`

    Set the alert message to JSON:
    ```json
    {"ticker": "{{ticker}}", "action": "buy", "comment": "{{strategy.order.comment}}"}
    ```

    **Signal Routing:**
    - If `strategy_id` is provided, the signal is sent to that specific strategy.
    - If omitted, routes to the first running strategy that trades the given ticker.
    """
    user, key_record = await _authenticate_api_key(db, x_api_key, request)
    source_ip = request.client.host if request.client else "unknown"

    logger.info(f"Webhook signal received: {signal.action} {signal.ticker} from user={user.id} key_prefix={key_record.key_prefix} ip={source_ip}")

    # Find target strategy
    strategy_query = select(LiveStrategy).where(
        LiveStrategy.user_id == user.id,
        LiveStrategy.status == StrategyStatus.RUNNING,
        LiveStrategy.is_deleted == False,  # noqa: E712
    )

    if signal.strategy_id:
        strategy_query = strategy_query.where(LiveStrategy.id == signal.strategy_id)
    else:
        # Find any running strategy that trades this ticker
        strategy_query = strategy_query.where(LiveStrategy.symbols.any(signal.ticker.upper()))

    result = await db.execute(strategy_query.limit(1))
    strategy = result.scalar_one_or_none()

    if not strategy:
        detail = (
            f"No running strategy found for ticker '{signal.ticker}'"
            + (f" with id={signal.strategy_id}" if signal.strategy_id else "")
            + ". Deploy and start a strategy first."
        )
        logger.warning(f"Webhook rejected: {detail} (user={user.id})")
        return WebhookExecutionResponse(
            strategy_id=signal.strategy_id or 0,
            strategy_name="N/A",
            ticker=signal.ticker,
            action=signal.action,
            status="rejected",
            message=detail,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # Map action to trade side
    side_map = {"buy": "BUY", "sell": "SELL", "close": "SELL"}
    side = side_map.get(signal.action, "BUY")

    # Determine quantity
    quantity = signal.quantity or 0
    if quantity == 0:
        # Default: use 10% of strategy capital at current price
        # (Simple default — real systems would use strategy-specific sizing)
        quantity = 1  # Minimum 1 share as fallback

    # Create trade record
    trade = LiveTrade(
        strategy_id=strategy.id,
        symbol=signal.ticker.upper(),
        side=side,
        quantity=quantity,
        entry_price=signal.price or 0,  # 0 = market order
        status="OPEN",
        strategy_signal={
            "source": "webhook",
            "action": signal.action,
            "comment": signal.comment,
            "api_key_prefix": key_record.key_prefix,
            "source_ip": source_ip,
            "metadata": signal.metadata,
            "interval": signal.interval,
            "exchange": signal.exchange,
        },
    )
    db.add(trade)

    # Update strategy stats
    strategy.total_trades = (strategy.total_trades or 0) + 1
    strategy.last_trade_at = datetime.now(timezone.utc)

    await db.commit()
    await db.refresh(trade)

    logger.info(f"Webhook trade created: trade_id={trade.id} strategy={strategy.id} {side} {quantity} {signal.ticker.upper()}")

    return WebhookExecutionResponse(
        trade_id=trade.id,
        strategy_id=strategy.id,
        strategy_name=strategy.name,
        ticker=signal.ticker.upper(),
        action=signal.action,
        status="executed",
        message=f"Trade {side} {quantity} {signal.ticker.upper()} created on '{strategy.name}'",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/history")
async def get_webhook_history(
    request: Request,
    limit: int = 20,
    x_api_key: str = Header(None, alias="X-API-Key"),
    db: AsyncSession = Depends(get_db),
):
    """Get recent webhook-triggered trades for the authenticated user."""
    user, _ = await _authenticate_api_key(db, x_api_key, request)

    # Fetch trades triggered by webhooks (identified by strategy_signal.source = "webhook")
    result = await db.execute(
        select(LiveTrade)
        .join(LiveStrategy, LiveTrade.strategy_id == LiveStrategy.id)
        .where(LiveStrategy.user_id == user.id)
        .order_by(LiveTrade.id.desc())
        .limit(limit)
    )
    trades = result.scalars().all()

    # Filter to webhook-triggered trades
    webhook_trades = []
    for t in trades:
        sig = t.strategy_signal or {}
        if sig.get("source") == "webhook":
            webhook_trades.append(
                {
                    "trade_id": t.id,
                    "strategy_id": t.strategy_id,
                    "symbol": t.symbol,
                    "side": t.side,
                    "quantity": t.quantity,
                    "entry_price": t.entry_price,
                    "status": t.status,
                    "comment": sig.get("comment"),
                    "source_ip": sig.get("source_ip"),
                    "created_at": t.created_at.isoformat() if t.created_at else "",
                }
            )

    return {"trades": webhook_trades, "total": len(webhook_trades)}


@router.get("/test")
async def test_webhook_auth(
    request: Request,
    x_api_key: str = Header(None, alias="X-API-Key"),
    db: AsyncSession = Depends(get_db),
):
    """Test API key authentication without executing any trade. Useful for verifying setup."""
    user, key_record = await _authenticate_api_key(db, x_api_key, request)
    return {
        "status": "authenticated",
        "user_id": user.id,
        "username": user.username,
        "key_prefix": key_record.key_prefix,
        "permissions": key_record.permissions,
        "message": "API key is valid and has trade permissions. Ready to receive signals.",
    }
