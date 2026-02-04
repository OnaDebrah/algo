"""
WebSocket routes for real-time updates
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Set

import yfinance as yf
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, status as http_status
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models.live import LiveStrategy
from backend.app.utils.security import decode_access_token
from backend.app.websockets.manager import ws_manager

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/websocket", tags=["WebSocket"])


class ConnectionManager:
    """Manages WebSocket connections"""

    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.user_connections: Dict[int, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, user_id: int, channel: str = "general"):
        """Connect a new client"""
        await websocket.accept()

        # Add to channel
        if channel not in self.active_connections:
            self.active_connections[channel] = set()
        self.active_connections[channel].add(websocket)

        # Add to user connections
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(websocket)

    def disconnect(self, websocket: WebSocket, user_id: int, channel: str = "general"):
        """Disconnect a client"""
        if channel in self.active_connections:
            self.active_connections[channel].discard(websocket)

        if user_id in self.user_connections:
            self.user_connections[user_id].discard(websocket)

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to specific connection"""
        await websocket.send_json(message)

    async def broadcast_to_channel(self, message: dict, channel: str = "general"):
        """Broadcast message to all connections in a channel"""
        if channel in self.active_connections:
            disconnected = set()
            for connection in self.active_connections[channel]:
                try:
                    await connection.send_json(message)
                except Exception:
                    disconnected.add(connection)

            # Remove disconnected clients
            for connection in disconnected:
                self.active_connections[channel].discard(connection)

    async def send_to_user(self, message: dict, user_id: int):
        """Send message to all connections of a specific user"""
        if user_id in self.user_connections:
            disconnected = set()
            for connection in self.user_connections[user_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    disconnected.add(connection)

            # Remove disconnected clients
            for connection in disconnected:
                self.user_connections[user_id].discard(connection)


async def get_current_user_ws(token: str):
    """Get current user from WebSocket token"""
    try:
        payload = decode_access_token(token)
        user_id = payload.get("sub")
        if user_id is None:
            return None
        return int(user_id)
    except Exception as e:
        logger.error(f"User not found: {e}")
        return None


@router.websocket("/ws/market/{symbol}")
async def websocket_market_data(websocket: WebSocket, symbol: str, token: str = None):
    """WebSocket endpoint for real-time market data"""
    user_id = await get_current_user_ws(token) if token else 0

    await ws_manager.connect(websocket, user_id, f"market:{symbol}")

    try:
        # Send initial data
        ticker = yf.Ticker(symbol)
        info = ticker.info

        await ws_manager.send_personal_message(
            {
                "type": "initial_data",
                "symbol": symbol,
                "data": {
                    "price": info.get("currentPrice", 0),
                    "change": info.get("regularMarketChange", 0),
                    "changePct": info.get("regularMarketChangePercent", 0),
                    "volume": info.get("volume", 0),
                    "timestamp": datetime.now().isoformat(),
                },
            },
            websocket,
        )

        # Stream updates
        while True:
            # Fetch latest data every 5 seconds
            await asyncio.sleep(5)

            ticker = yf.Ticker(symbol)
            info = ticker.info

            message = {
                "type": "market_update",
                "symbol": symbol,
                "data": {
                    "price": info.get("currentPrice", 0),
                    "change": info.get("regularMarketChange", 0),
                    "changePct": info.get("regularMarketChangePercent", 0),
                    "volume": info.get("volume", 0),
                    "timestamp": datetime.now().isoformat(),
                },
            }

            await ws_manager.broadcast_to_channel(message, f"market:{symbol}")

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, user_id, f"market:{symbol}")


@router.websocket("/ws/portfolio/{portfolio_id}")
async def websocket_portfolio_updates(websocket: WebSocket, portfolio_id: int, token: str = None):
    """WebSocket endpoint for portfolio updates"""
    user_id = await get_current_user_ws(token) if token else 0

    await ws_manager.connect(websocket, user_id, f"portfolio:{portfolio_id}")

    try:
        while True:
            # Listen for messages from client
            data = await websocket.receive_text()
            message = json.loads(data)

            # Echo back or process
            await ws_manager.send_personal_message({"type": "portfolio_update", "portfolio_id": portfolio_id, "data": message}, websocket)

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, user_id, f"portfolio:{portfolio_id}")


@router.websocket("/ws/backtest/{run_id}")
async def websocket_backtest_progress(websocket: WebSocket, run_id: int, token: str = None):
    """WebSocket endpoint for backtest progress updates"""
    user_id = await get_current_user_ws(token) if token else 0

    await ws_manager.connect(websocket, user_id, f"backtest:{run_id}")

    try:
        while True:
            # In practice, this would be updated by the backtest service
            await asyncio.sleep(1)

            # Send progress update
            await ws_manager.send_personal_message(
                {
                    "type": "backtest_progress",
                    "run_id": run_id,
                    "progress": 50,  # This would be real progress
                    "status": "running",
                },
                websocket,
            )

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, user_id, f"backtest:{run_id}")


@router.websocket("/strategy/{strategy_id}")
async def strategy_websocket(websocket: WebSocket, strategy_id: int, db: Session = Depends(get_db)):
    """
    WebSocket endpoint for real-time strategy updates

    Connect via: ws://localhost:8000/ws/strategy/{strategy_id}

    Message types:
    - equity_update: Real-time equity snapshots
    - trade_executed: Trade execution notifications
    - status_change: Strategy status changes
    - error: Error notifications
    """
    # Note: WebSocket authentication is tricky with Depends
    # For production, implement token-based auth

    # Verify strategy exists (basic security)
    strategy = db.query(LiveStrategy).filter(LiveStrategy.id == strategy_id).first()
    if not strategy:
        await websocket.close(code=http_status.WS_1008_POLICY_VIOLATION)
        return

    # Connect WebSocket
    await ws_manager.connect(websocket, strategy_id, strategy.user_id)

    try:
        # Keep connection alive
        while True:
            # Wait for messages from client (ping/pong)
            try:
                data = await websocket.receive_text()

                # Handle ping
                if data == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": datetime.utcnow().isoformat()})

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break

    finally:
        ws_manager.disconnect(websocket, strategy_id)


# Helper function to send updates from services
async def notify_backtest_progress(run_id: int, user_id: int, progress: int, status: str):
    """Send backtest progress update to connected clients"""
    await ws_manager.send_to_user(
        {"type": "backtest_progress", "run_id": run_id, "progress": progress, "status": status, "timestamp": datetime.now().isoformat()}, user_id
    )


async def notify_portfolio_update(portfolio_id: int, user_id: int, update_data: dict):
    """Send portfolio update to connected clients"""
    await ws_manager.broadcast_to_channel(
        {"type": "portfolio_update", "portfolio_id": portfolio_id, "data": update_data, "timestamp": datetime.now().isoformat()},
        f"portfolio:{portfolio_id}",
    )
