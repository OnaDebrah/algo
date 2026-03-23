"""
Price alert service for creating, querying, and checking price alerts
"""

import logging
from collections import defaultdict
from datetime import datetime, timezone

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.notification import PriceAlert
from .notification_service import NotificationService

logger = logging.getLogger(__name__)


class PriceAlertService:
    """Service for managing price alerts"""

    @staticmethod
    async def create_alert(
        db: AsyncSession,
        user_id: int,
        symbol: str,
        condition: str,
        target_price: float,
    ) -> PriceAlert:
        """Create a new price alert."""
        alert = PriceAlert(
            user_id=user_id,
            symbol=symbol.upper(),
            condition=condition,
            target_price=target_price,
        )
        db.add(alert)
        await db.commit()
        await db.refresh(alert)
        return alert

    @staticmethod
    async def get_alerts(db: AsyncSession, user_id: int) -> list[PriceAlert]:
        """Get all price alerts for a user."""
        result = await db.execute(select(PriceAlert).where(PriceAlert.user_id == user_id).order_by(PriceAlert.created_at.desc()))
        return list(result.scalars().all())

    @staticmethod
    async def delete_alert(db: AsyncSession, user_id: int, alert_id: int):
        """Delete a price alert."""
        stmt = delete(PriceAlert).where(
            PriceAlert.id == alert_id,
            PriceAlert.user_id == user_id,
        )
        await db.execute(stmt)
        await db.commit()

    @staticmethod
    async def check_all_alerts(db: AsyncSession):
        """Check all active price alerts against current market prices.

        Groups alerts by symbol to minimize API calls, then triggers
        notifications for any alerts whose conditions are met.
        """
        try:
            # Fetch all active alerts
            result = await db.execute(
                select(PriceAlert).where(PriceAlert.is_active == True)  # noqa: E712
            )
            active_alerts = list(result.scalars().all())

            if not active_alerts:
                return

            # Group by symbol
            alerts_by_symbol: dict[str, list[PriceAlert]] = defaultdict(list)
            for alert in active_alerts:
                alerts_by_symbol[alert.symbol].append(alert)

            from ..core.data.providers.providers import ProviderFactory

            provider = ProviderFactory()

            for symbol, alerts in alerts_by_symbol.items():
                try:
                    quote = await provider.get_quote(symbol)
                    current_price = quote.get("price", 0)

                    if current_price <= 0:
                        continue

                    for alert in alerts:
                        triggered = False
                        if alert.condition == "above" and current_price >= alert.target_price:
                            triggered = True
                        elif alert.condition == "below" and current_price <= alert.target_price:
                            triggered = True

                        if triggered:
                            alert.is_active = False
                            alert.triggered_at = datetime.now(timezone.utc)

                            # Create notification
                            notif = await NotificationService.create_notification(
                                db=db,
                                user_id=alert.user_id,
                                type="price_alert",
                                title=f"Price Alert: {symbol}",
                                message=(
                                    f"{symbol} is now ${current_price:.2f}, which is {alert.condition} your target of ${alert.target_price:.2f}"
                                ),
                                data={
                                    "symbol": symbol,
                                    "current_price": current_price,
                                    "target_price": alert.target_price,
                                    "condition": alert.condition,
                                },
                            )
                            alert.notification_id = notif.id

                except Exception as e:
                    logger.warning(f"Error checking price for {symbol}: {e}")
                    continue

            await db.commit()

        except Exception as e:
            logger.error(f"Error in check_all_alerts: {e}")
