"""
Updated Live Trading Routes
Properly integrates strategies with order execution
"""

from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.deps import get_current_user
from backend.app.database import get_db
from backend.app.models.backtest import BacktestRun
from backend.app.models.live import (
    DeploymentMode,
    LiveEquitySnapshot,
    LiveStrategy,
    LiveTrade,
    StrategyStatus,
)
from backend.app.schemas.live import (
    BrokerType,
    ConnectRequest,
    ControlRequest,
    EngineStatus,
    EquityPoint,
    ExecutionOrder,
    LiveStatus,
    TradeResponse,
)
from backend.app.schemas.strategy import DeployStrategyRequest, StrategyDetailsResponse, StrategyResponse

router = APIRouter(prefix="/live", tags=["Live Execution"])


# ============================================================================
# GLOBAL STATE MANAGEMENT (In production, use Redis or database)
# ============================================================================
class LiveTradingState:
    """Centralized state management for live trading"""

    def __init__(self):
        self.is_connected: bool = False
        self.engine_status: EngineStatus = EngineStatus.IDLE
        self.active_broker: BrokerType = BrokerType.PAPER
        self.connected_at: Optional[datetime] = None
        self.running_strategy_ids: List[int] = []

    def connect(self, broker: BrokerType):
        self.is_connected = True
        self.active_broker = broker
        self.connected_at = datetime.now(timezone.utc)

    def disconnect(self):
        self.is_connected = False
        self.engine_status = EngineStatus.IDLE
        self.running_strategy_ids = []

    def start_engine(self, strategy_ids: List[int] = None):
        if not self.is_connected:
            raise ValueError("Broker not connected")
        self.engine_status = EngineStatus.RUNNING
        if strategy_ids:
            self.running_strategy_ids = strategy_ids

    def stop_engine(self):
        self.engine_status = EngineStatus.IDLE
        self.running_strategy_ids = []


# Global state instance
trading_state = LiveTradingState()


# ============================================================================
# STATUS & CONNECTION ENDPOINTS
# ============================================================================


@router.get("/status", response_model=LiveStatus)
async def get_status():
    """Get current broker connection and engine status"""
    return {
        "is_connected": trading_state.is_connected,
        "engine_status": trading_state.engine_status,
        "active_broker": trading_state.active_broker,
    }


@router.post("/connect")
async def connect_broker(request: ConnectRequest):
    """Connect to trading broker"""
    # In production, validate credentials and establish actual connection
    trading_state.connect(request.broker)

    return {
        "status": "connected",
        "broker": request.broker,
        "connected_at": trading_state.connected_at.isoformat() if trading_state.connected_at else None,
    }


@router.post("/disconnect")
async def disconnect_broker(db: AsyncSession = Depends(get_db), current_user=Depends(get_current_user)):
    """Disconnect from broker and stop all running strategies"""

    # Stop all running strategies for this user
    if trading_state.running_strategy_ids:
        stmt = select(LiveStrategy).where(LiveStrategy.id.in_(trading_state.running_strategy_ids), LiveStrategy.user_id == current_user.id)
        result = await db.execute(stmt)
        strategies = result.scalars().all()

        for strategy in strategies:
            strategy.status = StrategyStatus.PAUSED

        await db.commit()

    trading_state.disconnect()

    return {"status": "disconnected"}


@router.post("/engine/start")
async def start_engine(strategy_ids: Optional[List[int]] = None, db: AsyncSession = Depends(get_db), current_user=Depends(get_current_user)):
    """
    Start execution engine

    If strategy_ids provided, only run those strategies.
    Otherwise, run all RUNNING strategies for the user.
    """
    if not trading_state.is_connected:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Broker not connected. Please connect to a broker first.")

    # Determine which strategies to run
    if strategy_ids:
        # Validate strategies belong to user
        stmt = select(LiveStrategy).where(LiveStrategy.id.in_(strategy_ids), LiveStrategy.user_id == current_user.id)
        result = await db.execute(stmt)
        strategies = result.scalars().all()

        if len(strategies) != len(strategy_ids):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="One or more strategies not found")

        # Update strategies to RUNNING status
        for strategy in strategies:
            if strategy.status != StrategyStatus.RUNNING:
                strategy.status = StrategyStatus.RUNNING
                strategy.started_at = datetime.now(timezone.utc)

        await db.commit()

        trading_state.start_engine(strategy_ids)
    else:
        # Get all RUNNING strategies for user
        stmt = select(LiveStrategy).where(LiveStrategy.user_id == current_user.id, LiveStrategy.status == StrategyStatus.RUNNING)
        result = await db.execute(stmt)
        strategies = result.scalars().all()

        strategy_ids = [s.id for s in strategies]
        trading_state.start_engine(strategy_ids)

    # In production, this would start the actual execution loop
    # For now, we just update the state

    return {"status": "started", "running_strategies": len(trading_state.running_strategy_ids), "strategy_ids": trading_state.running_strategy_ids}


@router.post("/engine/stop")
async def stop_engine(db: AsyncSession = Depends(get_db), current_user=Depends(get_current_user)):
    """Stop execution engine"""

    # Pause all running strategies
    if trading_state.running_strategy_ids:
        stmt = select(LiveStrategy).where(LiveStrategy.id.in_(trading_state.running_strategy_ids), LiveStrategy.user_id == current_user.id)
        result = await db.execute(stmt)
        strategies = result.scalars().all()

        for strategy in strategies:
            strategy.status = StrategyStatus.PAUSED

        await db.commit()

    trading_state.stop_engine()

    return {"status": "stopped"}


# ============================================================================
# ORDER MANAGEMENT
# ============================================================================


@router.get("/orders", response_model=List[ExecutionOrder])
async def get_orders(strategy_id: Optional[int] = None, db: AsyncSession = Depends(get_db), current_user=Depends(get_current_user)):
    """
    Get active orders

    If strategy_id provided, only return orders for that strategy.
    Otherwise, return all orders for running strategies.
    """

    # Determine which strategies to query
    if strategy_id:
        strategy_ids = [strategy_id]
    else:
        strategy_ids = trading_state.running_strategy_ids

    if not strategy_ids:
        # No running strategies, return empty list
        return []

    # Query LiveTrade table for open trades that represent active orders
    stmt = (
        select(LiveTrade)
        .where(
            LiveTrade.strategy_id.in_(strategy_ids),
            LiveTrade.status == "open",  # Only open trades
        )
        .order_by(LiveTrade.opened_at.desc())
    )

    result = await db.execute(stmt)
    trades = result.scalars().all()

    # Convert trades to ExecutionOrder format
    orders = []
    for trade in trades:
        orders.append(
            {
                "id": f"ORD-{trade.id}",
                "symbol": trade.symbol,
                "side": trade.side.value if hasattr(trade.side, "value") else trade.side,
                "qty": int(trade.quantity),
                "type": "MARKET",  # Default for now
                "status": "PENDING",  # Open trades are pending
                "price": float(trade.entry_price),
                "time": trade.opened_at.strftime("%H:%M:%S"),
                "strategy_id": trade.strategy_id,
            }
        )

    return orders


# ============================================================================
# STRATEGY DEPLOYMENT
# ============================================================================


@router.post("/strategy/deploy", status_code=status.HTTP_201_CREATED)
async def deploy_strategy(request: DeployStrategyRequest, db: AsyncSession = Depends(get_db), current_user=Depends(get_current_user)):
    """
    Deploy a strategy to live or paper trading

    This creates a new LiveStrategy record and starts monitoring
    """
    if request.deployment_mode not in ["paper", "live"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="deployment_mode must be 'paper' or 'live'")

    # Get backtest data if deploying from backtest
    backtest = None
    if request.source == "backtest" and request.backtest_id:
        stmt = select(BacktestRun).where(BacktestRun.id == request.backtest_id, BacktestRun.user_id == current_user.id)
        result = await db.execute(stmt)
        backtest = result.scalars().first()

        if not backtest:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Backtest {request.backtest_id} not found")

    now = datetime.now(timezone.utc)

    live_strategy = LiveStrategy(
        user_id=current_user.id,
        name=request.name,
        strategy_key=request.strategy_key,
        parameters=request.parameters,
        symbols=request.symbols,
        backtest_id=request.backtest_id,
        deployment_mode=DeploymentMode(request.deployment_mode),
        status=StrategyStatus.STOPPED,  # Start as STOPPED, user must start it
        initial_capital=request.initial_capital,
        current_equity=request.initial_capital,
        max_position_pct=request.max_position_pct,
        stop_loss_pct=request.stop_loss_pct,
        daily_loss_limit=request.daily_loss_limit,
        broker=request.broker or trading_state.active_broker.value,
        notes=request.notes,
        created_at=now,  # ✅ FIX: Explicitly set created_at
        deployed_at=now,
        # Copy backtest metrics if available
        backtest_return_pct=backtest.total_return_pct if backtest else None,
        backtest_sharpe=backtest.sharpe_ratio if backtest else None,
        backtest_max_drawdown=backtest.max_drawdown if backtest else None,
    )

    db.add(live_strategy)
    await db.flush()

    # Create initial equity snapshot
    initial_snapshot = LiveEquitySnapshot(
        strategy_id=live_strategy.id,
        timestamp=now,
        equity=request.initial_capital,
        cash=request.initial_capital,
        positions_value=0.0,
        daily_pnl=0.0,
        total_pnl=0.0,
        drawdown_pct=0.0,
    )

    db.add(initial_snapshot)
    await db.commit()
    await db.refresh(live_strategy)

    return {
        "strategy_id": live_strategy.id,
        "status": "deployed",
        "deployment_mode": request.deployment_mode,
        "deployed_at": now.isoformat(),
        "message": f"Strategy '{request.name}' deployed to {request.deployment_mode} trading. Start the strategy from the Live Execution page.",
    }


# ============================================================================
# STRATEGY MANAGEMENT
# ============================================================================


@router.get("/strategy", response_model=List[StrategyResponse])
async def list_strategies(
    status_: Optional[str] = None, mode: Optional[str] = None, db: AsyncSession = Depends(get_db), current_user=Depends(get_current_user)
):
    """List all strategies for the current user"""
    stmt = select(LiveStrategy).where(LiveStrategy.user_id == current_user.id)

    if status_:
        try:
            stmt = stmt.where(LiveStrategy.status == StrategyStatus(status_))
        except ValueError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid status: {status_}")

    if mode:
        try:
            stmt = stmt.where(LiveStrategy.deployment_mode == DeploymentMode(mode))
        except ValueError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid mode: {mode}")

    stmt = stmt.order_by(LiveStrategy.deployed_at.desc())

    result = await db.execute(stmt)
    strategies = result.scalars().all()

    return [
        StrategyResponse(
            id=s.id,
            name=s.name,
            strategy_key=s.strategy_key,
            symbols=s.symbols,
            status=s.status.value,
            deployment_mode=s.deployment_mode.value,
            current_equity=s.current_equity or s.initial_capital,
            total_return_pct=s.total_return_pct or 0.0,
            total_trades=s.total_trades or 0,
            deployed_at=s.deployed_at.isoformat() if s.deployed_at else None,
        )
        for s in strategies
    ]


@router.get("/strategy/{strategy_id}", response_model=StrategyDetailsResponse)
async def get_strategy_details(strategy_id: int, db: AsyncSession = Depends(get_db), current_user=Depends(get_current_user)):
    """Get complete details for a strategy including equity curve and trades"""
    stmt = select(LiveStrategy).where(LiveStrategy.id == strategy_id, LiveStrategy.user_id == current_user.id)
    result = await db.execute(stmt)
    strategy = result.scalars().first()

    if not strategy:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Strategy {strategy_id} not found")

    # Get equity curve
    stmt = select(LiveEquitySnapshot).where(LiveEquitySnapshot.strategy_id == strategy_id).order_by(LiveEquitySnapshot.timestamp.asc())

    result = await db.execute(stmt)
    snapshots = result.scalars().all()

    equity_curve = [EquityPoint(timestamp=s.timestamp.isoformat(), equity=s.equity, cash=s.cash, daily_pnl=s.daily_pnl) for s in snapshots]

    # Get trades
    stmt = select(LiveTrade).where(LiveTrade.strategy_id == strategy_id).order_by(LiveTrade.opened_at.desc())

    result = await db.execute(stmt)
    trades = result.scalars().all()

    trade_list = [
        TradeResponse(
            id=t.id,
            symbol=t.symbol,
            side=t.side.value if hasattr(t.side, "value") else t.side,
            quantity=t.quantity,
            entry_price=t.entry_price,
            exit_price=t.exit_price,
            status=t.status.value if hasattr(t.status, "value") else t.status,
            profit=t.profit,
            profit_pct=t.profit_pct,
            opened_at=t.opened_at.isoformat(),
            closed_at=t.closed_at.isoformat() if t.closed_at else None,
        )
        for t in trades
    ]

    # Backtest comparison
    comparison = None
    if strategy.backtest_return_pct:
        comparison = {
            "backtest_return_pct": strategy.backtest_return_pct,
            "live_return_pct": strategy.total_return_pct or 0.0,
            "backtest_sharpe": strategy.backtest_sharpe,
            "live_sharpe": strategy.sharpe_ratio,
            "backtest_max_drawdown": strategy.backtest_max_drawdown,
            "live_max_drawdown": strategy.max_drawdown,
        }

    return StrategyDetailsResponse(
        strategy=StrategyResponse(
            id=strategy.id,
            name=strategy.name,
            strategy_key=strategy.strategy_key,
            symbols=strategy.symbols,
            status=strategy.status.value,
            deployment_mode=strategy.deployment_mode.value,
            current_equity=strategy.current_equity or strategy.initial_capital,
            total_return_pct=strategy.total_return_pct or 0.0,
            total_trades=strategy.total_trades or 0,
            deployed_at=strategy.deployed_at.isoformat() if strategy.deployed_at else None,
        ),
        equity_curve=equity_curve,
        trades=trade_list,
        backtest_comparison=comparison,
    )


@router.post("/strategy/{strategy_id}/control")
async def control_strategy(strategy_id: int, request: ControlRequest, db: AsyncSession = Depends(get_db), current_user=Depends(get_current_user)):
    """
    Control strategy execution

    Actions: start, pause, stop, restart
    """
    stmt = select(LiveStrategy).where(LiveStrategy.id == strategy_id, LiveStrategy.user_id == current_user.id)

    result = await db.execute(stmt)
    strategy = result.scalars().first()

    if not strategy:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Strategy {strategy_id} not found")

    action = request.action.lower()
    now = datetime.now(timezone.utc)

    if action == "start":
        if strategy.status != StrategyStatus.RUNNING:
            strategy.status = StrategyStatus.RUNNING
            strategy.started_at = now

            # Add to running strategies if engine is running
            if trading_state.engine_status == EngineStatus.RUNNING:
                if strategy_id not in trading_state.running_strategy_ids:
                    trading_state.running_strategy_ids.append(strategy_id)

    elif action == "pause":
        if strategy.status == StrategyStatus.RUNNING:
            strategy.status = StrategyStatus.PAUSED

            # Remove from running strategies
            if strategy_id in trading_state.running_strategy_ids:
                trading_state.running_strategy_ids.remove(strategy_id)

    elif action == "stop":
        strategy.status = StrategyStatus.STOPPED
        strategy.stopped_at = now

        # Remove from running strategies
        if strategy_id in trading_state.running_strategy_ids:
            trading_state.running_strategy_ids.remove(strategy_id)

    elif action == "restart":
        strategy.status = StrategyStatus.RUNNING
        strategy.started_at = now

        # Add to running strategies if engine is running
        if trading_state.engine_status == EngineStatus.RUNNING:
            if strategy_id not in trading_state.running_strategy_ids:
                trading_state.running_strategy_ids.append(strategy_id)

    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid action: {action}. Must be start, pause, stop, or restart")

    await db.commit()

    return {
        "strategy_id": strategy_id,
        "status": strategy.status.value,
        "action": action,
        "message": f"Strategy {action}ed successfully",
        "timestamp": now.isoformat(),
    }


@router.delete("/strategy/{strategy_id}")
async def delete_strategy(strategy_id: int, db: AsyncSession = Depends(get_db), current_user=Depends(get_current_user)):
    """Delete a strategy (soft delete - sets status to stopped)"""
    stmt = select(LiveStrategy).where(LiveStrategy.id == strategy_id, LiveStrategy.user_id == current_user.id)

    result = await db.execute(stmt)
    strategy = result.scalars().first()

    if not strategy:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Strategy {strategy_id} not found")

    # Soft delete
    strategy.status = StrategyStatus.STOPPED
    strategy.stopped_at = datetime.now(timezone.utc)

    # Remove from running strategies
    if strategy_id in trading_state.running_strategy_ids:
        trading_state.running_strategy_ids.remove(strategy_id)

    await db.commit()

    return {"strategy_id": strategy_id, "message": "Strategy deleted successfully"}
