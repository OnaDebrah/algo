"""
Live Trading Routes with Broker Integration
Uses BrokerFactory to support multiple brokers (Paper, Alpaca, IB, etc.)
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.deps import get_current_user
from backend.app.api.routes.live.live_trading_state import LiveTradingState
from backend.app.database import AsyncSessionLocal, get_db
from backend.app.models.backtest import BacktestRun
from backend.app.models.live import (
    DeploymentMode,
    LiveEquitySnapshot,
    LiveStrategy,
    LiveStrategySnapshot,
    LiveTrade,
    StrategyStatus,
    TradeStatus,
)
from backend.app.models.user_settings import UserSettings as UserSettingsModel
from backend.app.schemas.live import (
    ConnectRequest,
    ControlRequest,
    EngineStatus,
    EquityPoint,
    ExecutionOrder,
    LiveStatus,
    TradeResponse,
)
from backend.app.schemas.strategy import (
    DeployStrategyRequest,
    StrategyDetailsResponse,
    StrategyResponse,
    UpdateStrategyRequest,
)
from backend.app.services.execution_manager import get_execution_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/live", tags=["Live Execution"])

trading_state = LiveTradingState()
execution_manager = get_execution_manager(AsyncSessionLocal)


async def get_user_broker_settings(db: AsyncSession, user_id: int) -> Optional[Dict[str, Any]]:
    """Get user's broker settings from database"""
    stmt = select(UserSettingsModel).where(UserSettingsModel.user_id == user_id)
    result = await db.execute(stmt)
    settings = result.scalars().first()

    if not settings:
        return None

    return {
        "broker_type": settings.default_broker or "paper",
        "api_key": settings.broker_api_key,
        "api_secret": settings.broker_api_secret,
        "base_url": settings.broker_base_url,
        "auto_connect": settings.auto_connect_broker or False,
        "data_source": settings.live_data_source or "alpaca",
        "initial_capital": settings.initial_capital or 100000.0,
    }


@router.get("/status", response_model=LiveStatus)
async def get_status(db: AsyncSession = Depends(get_db), current_user=Depends(get_current_user)):
    """Get current broker connection and engine status"""

    # Get user's data source preference
    user_broker = await get_user_broker_settings(db, current_user.id)
    data_source = user_broker["data_source"] if user_broker else "alpaca"

    # Check market status if connected
    market_open = False
    if trading_state.is_connected and trading_state.broker_client:
        try:
            market_open = await trading_state.broker_client.is_market_open()
        except Exception as e:
            logger.error(f"Error checking market status: {e}")

    return {
        "is_connected": trading_state.is_connected,
        "engine_status": trading_state.engine_status,
        "active_broker": trading_state.active_broker,
        "data_source": data_source,
        "connected_at": trading_state.connected_at.isoformat() if trading_state.connected_at else None,
        "market_open": market_open,
        "running_strategies": len(trading_state.running_strategy_ids),
    }


@router.post("/connect")
async def connect_broker(
    request: Optional[ConnectRequest] = None, use_settings: bool = True, db: AsyncSession = Depends(get_db), current_user=Depends(get_current_user)
):
    """
    Connect to trading broker using saved settings or manual override
    """

    broker_type = None
    credentials = {}

    if use_settings:
        # Get credentials from user settings
        user_broker = await get_user_broker_settings(db, current_user.id)

        if user_broker:
            broker_type = user_broker["broker_type"]

            # Build credentials dict
            if broker_type != "paper":
                credentials = {"api_key": user_broker["api_key"], "api_secret": user_broker["api_secret"], "base_url": user_broker["base_url"]}
            else:
                credentials = {"initial_capital": user_broker["initial_capital"]}

    # Override with request if provided
    if request:
        broker_type = request.broker or broker_type
        if request.api_key:
            credentials["api_key"] = request.api_key
        if request.api_secret:
            credentials["api_secret"] = request.api_secret

    if not broker_type:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No broker configured. Please configure broker in Settings.")

    # Connect using broker factory
    success = await trading_state.connect(broker_type, credentials)

    if not success:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to connect to {broker_type}")

    # Get account info
    account_info = {}
    if trading_state.broker_client:
        account_info = await trading_state.broker_client.get_account_info()

    return {
        "status": "connected",
        "broker": broker_type,
        "mode": "paper" if broker_type == "paper" else "live",
        "connected_at": trading_state.connected_at.isoformat(),
        "account": account_info,
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

    await trading_state.disconnect()

    return {"status": "disconnected"}


@router.post("/auto-connect")
async def auto_connect_broker(db: AsyncSession = Depends(get_db), current_user=Depends(get_current_user)):
    """Auto-connect using saved settings (called on app startup)"""

    user_broker = await get_user_broker_settings(db, current_user.id)

    if not user_broker:
        return {"status": "skipped", "message": "No broker settings configured"}

    if not user_broker["auto_connect"]:
        return {"status": "skipped", "message": "Auto-connect disabled in settings"}

    # Try to connect
    try:
        return await connect_broker(request=None, use_settings=True, db=db, current_user=current_user)
    except Exception as e:
        return {"status": "failed", "message": f"Auto-connect failed: {str(e)}"}


@router.post("/engine/start")
async def start_engine(strategy_ids: Optional[List[int]] = None, db: AsyncSession = Depends(get_db), current_user=Depends(get_current_user)):
    """Start execution engine with selected strategies"""

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

    # Start background execution for each strategy
    for sid in trading_state.running_strategy_ids:
        await execution_manager.deploy_strategy(sid)

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

    # Stop all running strategies in ExecutionManager
    for sid in list(trading_state.running_strategy_ids):
        await execution_manager.stop_strategy(sid)

    trading_state.stop_engine()

    return {"status": "stopped"}


# ============================================================================
# ORDER MANAGEMENT
# ============================================================================


@router.get("/orders", response_model=List[ExecutionOrder])
async def get_orders(strategy_id: Optional[int] = None, db: AsyncSession = Depends(get_db), current_user=Depends(get_current_user)):
    """Get active orders from broker"""

    if not trading_state.is_connected or not trading_state.broker_client:
        return []

    # Determine which strategies to query
    if strategy_id:
        strategy_ids = [strategy_id]
    else:
        strategy_ids = trading_state.running_strategy_ids

    if not strategy_ids:
        return []

    # Get open trades from database that represent pending orders
    stmt = select(LiveTrade).where(LiveTrade.strategy_id.in_(strategy_ids), LiveTrade.status == TradeStatus.OPEN).order_by(LiveTrade.opened_at.desc())

    result = await db.execute(stmt)
    trades = result.scalars().all()

    # Convert to ExecutionOrder format
    orders = []
    for trade in trades:
        orders.append(
            {
                "id": f"ORD-{trade.id}",
                "symbol": trade.symbol,
                "side": trade.side.value if hasattr(trade.side, "value") else trade.side,
                "qty": int(trade.quantity),
                "type": "MARKET",
                "status": "PENDING",
                "price": float(trade.entry_price),
                "time": trade.opened_at.strftime("%H:%M:%S"),
                "strategy_id": trade.strategy_id,
            }
        )

    return orders


@router.get("/account")
async def get_account():
    """Get account information from broker"""

    if not trading_state.is_connected or not trading_state.broker_client:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Not connected to broker")

    try:
        account_info = await trading_state.broker_client.get_account_info()
        return account_info
    except Exception as e:
        logger.error(f"Error getting account info: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get account info: {str(e)}")


@router.get("/positions")
async def get_positions():
    """Get current positions from broker"""

    if not trading_state.is_connected or not trading_state.broker_client:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Not connected to broker")

    try:
        positions = await trading_state.broker_client.get_positions()
        return {"positions": positions}
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get positions: {str(e)}")


@router.get("/market-hours")
async def get_market_hours():
    """Get market hours information"""

    if not trading_state.is_connected or not trading_state.broker_client:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Not connected to broker")

    try:
        # Check if broker has get_market_hours method (PaperClient does)
        if hasattr(trading_state.broker_client, "get_market_hours"):
            return await trading_state.broker_client.get_market_hours()
        else:
            # Fallback: just return is_open
            is_open = await trading_state.broker_client.is_market_open()
            return {"is_open": is_open, "message": "Detailed market hours not available for this broker"}
    except Exception as e:
        logger.error(f"Error getting market hours: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get market hours: {str(e)}")


# ============================================================================
# STRATEGY DEPLOYMENT
# ============================================================================


@router.post("/strategy/deploy", status_code=status.HTTP_201_CREATED)
async def deploy_strategy(request: DeployStrategyRequest, db: AsyncSession = Depends(get_db), current_user=Depends(get_current_user)):
    """Deploy a strategy to live or paper trading"""

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
        status=StrategyStatus.STOPPED,
        initial_capital=request.initial_capital,
        current_equity=request.initial_capital,
        max_position_pct=request.max_position_pct,
        stop_loss_pct=request.stop_loss_pct,
        daily_loss_limit=request.daily_loss_limit,
        broker=request.broker or trading_state.broker_type or "paper",
        notes=request.notes,
        created_at=now,
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
        "message": f"Strategy '{request.name}' deployed to {request.deployment_mode} trading",
    }


# ============================================================================
# STRATEGY MANAGEMENT (List, Get, Control, Update, Delete)
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
    """Get complete details for a strategy"""
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
    """Control strategy execution (start, pause, stop, restart)"""
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

    # Trigger ExecutionManager for the strategy
    if action in ["start", "restart"]:
        await execution_manager.deploy_strategy(strategy_id)
    elif action == "pause":
        await execution_manager.pause_strategy(strategy_id)
    elif action == "stop":
        await execution_manager.stop_strategy(strategy_id)

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
    """Delete a strategy (soft delete)"""
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


@router.patch("/strategy/{strategy_id}", response_model=StrategyResponse)
async def update_strategy(
    strategy_id: int, request: UpdateStrategyRequest, db: AsyncSession = Depends(get_db), current_user=Depends(get_current_user)
):
    """
    Update strategy parameters and create a versioned snapshot
    """
    stmt = select(LiveStrategy).where(LiveStrategy.id == strategy_id, LiveStrategy.user_id == current_user.id)
    result = await db.execute(stmt)
    strategy = result.scalars().first()

    if not strategy:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Strategy {strategy_id} not found")

    # Create snapshot of CURRENT state before updating
    snapshot = LiveStrategySnapshot(
        strategy_id=strategy.id,
        version=strategy.version,
        parameters=strategy.parameters,
        notes=f"Snapshot before update at {datetime.now(timezone.utc).isoformat()}",
    )
    db.add(snapshot)

    # Update strategy fields
    if request.name:
        strategy.name = request.name
    if request.parameters:
        strategy.parameters = request.parameters
    if request.symbols:
        strategy.symbols = request.symbols
    if request.notes:
        strategy.notes = request.notes

    # Increment version
    strategy.version += 1

    await db.commit()
    await db.refresh(strategy)

    # If running, we might need to notify ExecutionManager to reload parameters
    if strategy.status == StrategyStatus.RUNNING:
        await execution_manager.deploy_strategy(strategy_id)

    return StrategyResponse(
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
    )


@router.post("/strategy/{strategy_id}/rollback/{version_id}")
async def rollback_strategy(strategy_id: int, version_id: int, db: AsyncSession = Depends(get_db), current_user=Depends(get_current_user)):
    """
    Rollback strategy parameters to a previous version
    """
    # Verify strategy ownership
    stmt = select(LiveStrategy).where(LiveStrategy.id == strategy_id, LiveStrategy.user_id == current_user.id)
    result = await db.execute(stmt)
    strategy = result.scalars().first()

    if not strategy:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Strategy {strategy_id} not found")

    # Find the specific snapshot
    stmt = select(LiveStrategySnapshot).where(LiveStrategySnapshot.strategy_id == strategy_id, LiveStrategySnapshot.id == version_id)
    result = await db.execute(stmt)
    snapshot = result.scalars().first()

    if not snapshot:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Snapshot {version_id} not found for strategy {strategy_id}")

    # Create a snapshot of the current state BEFORE rolling back
    new_snapshot = LiveStrategySnapshot(
        strategy_id=strategy.id,
        version=strategy.version,
        parameters=strategy.parameters,
        notes=f"Snapshot before rollback to version {snapshot.version}",
    )
    db.add(new_snapshot)

    # Revert parameters
    strategy.parameters = snapshot.parameters
    strategy.version += 1

    await db.commit()

    # If running, reload
    if strategy.status == StrategyStatus.RUNNING:
        await execution_manager.deploy_strategy(strategy_id)

    return {"status": "success", "message": f"Rolled back to version {snapshot.version}", "current_version": strategy.version}


@router.get("/strategy/{strategy_id}/versions", response_model=List[Dict[str, Any]])
async def list_strategy_versions(strategy_id: int, db: AsyncSession = Depends(get_db), current_user=Depends(get_current_user)):
    """
    List all saved versions of a strategy
    """
    # Verify ownership
    stmt = select(LiveStrategy).where(LiveStrategy.id == strategy_id, LiveStrategy.user_id == current_user.id)
    result = await db.execute(stmt)
    strategy = result.scalars().first()

    if not strategy:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Strategy {strategy_id} not found")

    # Get all snapshots
    stmt = select(LiveStrategySnapshot).where(LiveStrategySnapshot.strategy_id == strategy_id).order_by(LiveStrategySnapshot.version.desc())
    result = await db.execute(stmt)
    snapshots = result.scalars().all()

    return [
        {
            "id": s.id,
            "version": s.version,
            "parameters": s.parameters,
            "created_at": s.created_at.isoformat(),
            "notes": s.notes,
        }
        for s in snapshots
    ]
