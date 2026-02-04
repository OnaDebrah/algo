"""
WebSocket Manager for Real-Time Strategy Updates
Broadcasts equity snapshots and trade updates to connected clients
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Set

from fastapi import WebSocket
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.live import LiveStrategy

logger = logging.getLogger(__name__)


class LiveStrategyWebSocketManager:
    """
    Manages WebSocket connections for live strategy monitoring

    Supports:
    - Multiple clients per strategy
    - Equity updates every 60 seconds
    - Trade execution notifications
    - Status change notifications
    """

    def __init__(self):
        # {strategy_id: Set[WebSocket]}
        self.active_connections: Dict[int, Set[WebSocket]] = {}
        self.connection_user_map: Dict[WebSocket, int] = {}  # Track user per connection

    async def connect(self, websocket: WebSocket, strategy_id: int, user_id: int):
        """
        Accept new WebSocket connection for a strategy
        """
        await websocket.accept()

        if strategy_id not in self.active_connections:
            self.active_connections[strategy_id] = set()

        self.active_connections[strategy_id].add(websocket)
        self.connection_user_map[websocket] = user_id

        logger.info(f"WebSocket connected: user={user_id}, strategy={strategy_id}")

        # Send initial connection confirmation
        await websocket.send_json(
            {
                "type": "connected",
                "strategy_id": strategy_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": "Connected to strategy updates",
            }
        )

    def disconnect(self, websocket: WebSocket, strategy_id: int):
        """
        Remove WebSocket connection
        """
        if strategy_id in self.active_connections:
            self.active_connections[strategy_id].discard(websocket)

            # Clean up empty strategy sets
            if not self.active_connections[strategy_id]:
                del self.active_connections[strategy_id]

        if websocket in self.connection_user_map:
            user_id = self.connection_user_map.pop(websocket)
            logger.info(f"WebSocket disconnected: user={user_id}, strategy={strategy_id}")

    async def broadcast_to_strategy(self, strategy_id: int, message: Dict[str, Any]):
        """
        Broadcast message to all clients watching a strategy
        """
        if strategy_id not in self.active_connections:
            return

        dead_connections = set()

        for connection in self.active_connections[strategy_id]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to WebSocket: {e}")
                dead_connections.add(connection)

        # Clean up dead connections
        for dead in dead_connections:
            self.disconnect(dead, strategy_id)

    async def broadcast_equity_update(
        self, strategy_id: int, equity: float, cash: float, positions_value: float, daily_pnl: float, total_pnl: float, drawdown_pct: float
    ):
        """
        Broadcast equity update to all connected clients
        """
        message = {
            "type": "equity_update",
            "strategy_id": strategy_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "equity": equity,
                "cash": cash,
                "positions_value": positions_value,
                "daily_pnl": daily_pnl,
                "total_pnl": total_pnl,
                "drawdown_pct": drawdown_pct,
            },
        }

        await self.broadcast_to_strategy(strategy_id, message)

    async def broadcast_trade_executed(self, strategy_id: int, trade_data: Dict[str, Any]):
        """
        Broadcast trade execution to all connected clients
        """
        message = {"type": "trade_executed", "strategy_id": strategy_id, "timestamp": datetime.now(timezone.utc).isoformat(), "data": trade_data}

        await self.broadcast_to_strategy(strategy_id, message)

    async def broadcast_status_change(self, strategy_id: int, old_status: str, new_status: str, reason: str = ""):
        """
        Broadcast status change to all connected clients
        """
        message = {
            "type": "status_change",
            "strategy_id": strategy_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {"old_status": old_status, "new_status": new_status, "reason": reason},
        }

        await self.broadcast_to_strategy(strategy_id, message)

    async def broadcast_error(self, strategy_id: int, error_message: str):
        """
        Broadcast error to all connected clients
        """
        message = {"type": "error", "strategy_id": strategy_id, "timestamp": datetime.now(timezone.utc).isoformat(), "data": {"error": error_message}}

        await self.broadcast_to_strategy(strategy_id, message)

    def get_connection_count(self, strategy_id: int) -> int:
        """
        Get number of active connections for a strategy
        """
        if strategy_id not in self.active_connections:
            return 0
        return len(self.active_connections[strategy_id])

    def get_total_connections(self) -> int:
        """
        Get total number of active connections across all strategies
        """
        return sum(len(connections) for connections in self.active_connections.values())


# Global WebSocket manager instance
ws_manager = LiveStrategyWebSocketManager()


async def equity_snapshot_background_task(db_session_maker):
    """
    Background task that runs every 60 seconds
    Calculates equity for all active strategies and broadcasts updates

    This should be started on app startup:

    @app.on_event("startup")
    async def startup():
        asyncio.create_task(equity_snapshot_background_task(SessionLocal))
    """
    logger.info("Starting equity snapshot background service")

    while True:
        try:
            # Create DB session
            db = db_session_maker()

            # Get all running strategies
            strategies = db.query(LiveStrategy).filter(LiveStrategy.status == "running").all()

            for strategy in strategies:
                try:
                    # Calculate current equity
                    # This would integrate with your broker API
                    equity_data = await calculate_live_equity(strategy, db)

                    if equity_data:
                        # Save snapshot to database
                        from backend.app.models.live import LiveEquitySnapshot

                        snapshot = LiveEquitySnapshot(
                            strategy_id=strategy.id,
                            timestamp=datetime.utcnow(),
                            equity=equity_data["equity"],
                            cash=equity_data["cash"],
                            positions_value=equity_data["positions_value"],
                            daily_pnl=equity_data["daily_pnl"],
                            total_pnl=equity_data["total_pnl"],
                            drawdown_pct=equity_data["drawdown_pct"],
                        )

                        db.add(snapshot)
                        db.commit()

                        # Update strategy
                        strategy.current_equity = equity_data["equity"]
                        strategy.daily_pnl = equity_data["daily_pnl"]
                        strategy.last_equity_update = datetime.utcnow()
                        db.commit()

                        # Broadcast to WebSocket clients
                        await ws_manager.broadcast_equity_update(
                            strategy_id=strategy.id,
                            equity=equity_data["equity"],
                            cash=equity_data["cash"],
                            positions_value=equity_data["positions_value"],
                            daily_pnl=equity_data["daily_pnl"],
                            total_pnl=equity_data["total_pnl"],
                            drawdown_pct=equity_data["drawdown_pct"],
                        )

                except Exception as e:
                    logger.error(f"Error processing strategy {strategy.id}: {e}")

            db.close()

        except Exception as e:
            logger.error(f"Error in equity snapshot service: {e}")

        # Wait 60 seconds before next update
        await asyncio.sleep(60)


async def calculate_live_equity(strategy: LiveStrategy, db: AsyncSession) -> Dict[str, float]:
    """
    Calculate current equity for a live strategy

    This would integrate with broker API to get:
    - Current cash balance
    - Open positions and their values
    - Calculate P&L

    For now, returns mock data
    """
    # TODO: Implement broker API integration

    # Mock implementation
    return {
        "equity": strategy.current_equity or strategy.initial_capital,
        "cash": strategy.initial_capital * 0.7,
        "positions_value": strategy.initial_capital * 0.3,
        "daily_pnl": 0.0,
        "total_pnl": (strategy.current_equity or strategy.initial_capital) - strategy.initial_capital,
        "drawdown_pct": 0.0,
    }
