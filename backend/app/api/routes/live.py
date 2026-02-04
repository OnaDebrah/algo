from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.api.deps import get_current_user
from backend.app.database import get_db
from backend.app.models.backtest import BacktestRun
from backend.app.models.live import DeploymentMode, LiveEquitySnapshot, LiveStrategy, LiveTrade, StrategyStatus
from backend.app.schemas.live import (
    BrokerType,
    ConnectRequest,
    ControlRequest,
    EngineStatus,
    EquityPoint,
    ExecutionOrder,
    LiveStatus,
    OrderSide,
    OrderStatus,
    OrderType,
    TradeResponse,
)
from backend.app.schemas.strategy import DeployStrategyRequest, StrategyDetailsResponse, StrategyResponse

router = APIRouter(prefix="/live", tags=["Live Execution"])

# Mock state (in a real app, this would be in a service/database)
state = {
    "is_connected": False,
    "engine_status": EngineStatus.IDLE,
    "active_broker": BrokerType.PAPER,
    "orders": [
        {
            "id": "ORD-9921",
            "symbol": "AAPL",
            "side": OrderSide.BUY,
            "qty": 50,
            "type": OrderType.LIMIT,
            "status": OrderStatus.FILLED,
            "price": 182.45,
            "time": "14:02:11",
        },
        {
            "id": "ORD-9925",
            "symbol": "TSLA",
            "side": OrderSide.SELL,
            "qty": 10,
            "type": OrderType.MARKET,
            "status": OrderStatus.PENDING,
            "price": 234.10,
            "time": "14:05:45",
        },
    ],
}


@router.get("/status", response_model=LiveStatus)
async def get_status():
    return {"is_connected": state["is_connected"], "engine_status": state["engine_status"], "active_broker": state["active_broker"]}


@router.post("/connect")
async def connect_broker(request: ConnectRequest):
    # Simulate connection
    state["active_broker"] = request.broker
    state["is_connected"] = True
    return {"status": "connected", "broker": request.broker}


@router.post("/disconnect")
async def disconnect_broker():
    state["is_connected"] = False
    state["engine_status"] = EngineStatus.IDLE
    return {"status": "disconnected"}


@router.post("/engine/start")
async def start_engine():
    if not state["is_connected"]:
        raise HTTPException(status_code=400, detail="Broker not connected")
    state["engine_status"] = EngineStatus.RUNNING
    return {"status": "started"}


@router.post("/engine/stop")
async def stop_engine():
    state["engine_status"] = EngineStatus.IDLE
    return {"status": "stopped"}


@router.get("/orders", response_model=List[ExecutionOrder])
async def get_orders():
    return state["orders"]


@router.post("/deploy", status_code=status.HTTP_201_CREATED)
async def deploy_strategy(request: DeployStrategyRequest, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    """
    Deploy a strategy to live or paper trading

    This creates a new LiveStrategy record and starts monitoring
    """
    # Validate deployment mode
    if request.deployment_mode not in ["paper", "live"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="deployment_mode must be 'paper' or 'live'")

    # Get backtest data if deploying from backtest
    backtest = None
    if request.source == "backtest" and request.backtest_id:
        backtest = db.query(BacktestRun).filter(BacktestRun.id == request.backtest_id, BacktestRun.user_id == current_user.id).first()

        if not backtest:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Backtest {request.backtest_id} not found")

    # Create LiveStrategy
    now = datetime.utcnow()

    live_strategy = LiveStrategy(
        user_id=current_user.id,
        name=request.name,
        strategy_key=request.strategy_key,
        parameters=request.parameters,
        symbols=request.symbols,
        backtest_id=request.backtest_id,
        deployment_mode=DeploymentMode(request.deployment_mode),
        status=StrategyStatus.RUNNING,
        initial_capital=request.initial_capital,
        current_equity=request.initial_capital,
        max_position_pct=request.max_position_pct,
        stop_loss_pct=request.stop_loss_pct,
        daily_loss_limit=request.daily_loss_limit,
        broker=request.broker,
        notes=request.notes,
        deployed_at=now,
        started_at=now,
        # Copy backtest metrics
        backtest_return_pct=backtest.total_return_pct if backtest else None,
        backtest_sharpe=backtest.sharpe_ratio if backtest else None,
        backtest_max_drawdown=backtest.max_drawdown if backtest else None,
    )

    db.add(live_strategy)
    db.flush()

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
    db.commit()
    db.refresh(live_strategy)

    return {
        "strategy_id": live_strategy.id,
        "status": "deployed",
        "deployment_mode": request.deployment_mode,
        "started_at": now.isoformat(),
        "message": f"Strategy successfully deployed to {request.deployment_mode} trading",
    }


@router.get("/", response_model=List[StrategyResponse])
async def list_strategies(
    status: Optional[str] = None, mode: Optional[str] = None, db: Session = Depends(get_db), current_user=Depends(get_current_user)
):
    """
    List all strategies for the current user

    Query params:
    - status: Filter by status (running, paused, stopped)
    - mode: Filter by mode (paper, live)
    """
    query = db.query(LiveStrategy).filter(LiveStrategy.user_id == current_user.id)

    if status:
        try:
            query = query.filter(LiveStrategy.status == StrategyStatus(status))
        except ValueError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid status: {status}")

    if mode:
        try:
            query = query.filter(LiveStrategy.deployment_mode == DeploymentMode(mode))
        except ValueError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid mode: {mode}")

    strategies = query.order_by(LiveStrategy.deployed_at.desc()).all()

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


@router.get("/{strategy_id}", response_model=StrategyDetailsResponse)
async def get_strategy_details(strategy_id: int, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    """
    Get complete details for a strategy including equity curve and trades
    """
    strategy = db.query(LiveStrategy).filter(LiveStrategy.id == strategy_id, LiveStrategy.user_id == current_user.id).first()

    if not strategy:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Strategy {strategy_id} not found")

    # Get equity snapshots
    snapshots = db.query(LiveEquitySnapshot).filter(LiveEquitySnapshot.strategy_id == strategy_id).order_by(LiveEquitySnapshot.timestamp).all()

    equity_curve = [EquityPoint(timestamp=s.timestamp.isoformat(), equity=s.equity, cash=s.cash, daily_pnl=s.daily_pnl) for s in snapshots]

    # Get trades
    trades = db.query(LiveTrade).filter(LiveTrade.strategy_id == strategy_id).order_by(LiveTrade.opened_at.desc()).all()

    trade_list = [
        TradeResponse(
            id=t.id,
            symbol=t.symbol,
            side=t.side.value,
            quantity=t.quantity,
            entry_price=t.entry_price,
            exit_price=t.exit_price,
            status=t.status.value,
            profit=t.profit,
            profit_pct=t.profit_pct,
            opened_at=t.opened_at.isoformat(),
            closed_at=t.closed_at.isoformat() if t.closed_at else None,
        )
        for t in trades
    ]

    # Build backtest comparison
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


@router.post("/{strategy_id}/control")
async def control_strategy(strategy_id: int, request: ControlRequest, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    """
    Control strategy execution

    Actions: start, pause, stop, restart
    """
    strategy = db.query(LiveStrategy).filter(LiveStrategy.id == strategy_id, LiveStrategy.user_id == current_user.id).first()

    if not strategy:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Strategy {strategy_id} not found")

    action = request.action.lower()
    now = datetime.now(timezone.utc)

    if action == "start":
        if strategy.status != StrategyStatus.RUNNING:
            strategy.status = StrategyStatus.RUNNING
            strategy.started_at = now

    elif action == "pause":
        if strategy.status == StrategyStatus.RUNNING:
            strategy.status = StrategyStatus.PAUSED

    elif action == "stop":
        strategy.status = StrategyStatus.STOPPED
        strategy.stopped_at = now

    elif action == "restart":
        strategy.status = StrategyStatus.RUNNING
        strategy.started_at = now

    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid action: {action}. Must be start, pause, stop, or restart")

    db.commit()

    return {
        "strategy_id": strategy_id,
        "status": strategy.status.value,
        "action": action,
        "message": f"Strategy {action}ed successfully",
        "timestamp": now.isoformat(),
    }


@router.put("/{strategy_id}")
async def update_strategy(
    strategy_id: int,
    parameters: Optional[Dict[str, Any]] = None,
    max_position_pct: Optional[float] = None,
    stop_loss_pct: Optional[float] = None,
    daily_loss_limit: Optional[float] = None,
    notes: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Update strategy parameters

    Creates a new version of the strategy
    """
    strategy = db.query(LiveStrategy).filter(LiveStrategy.id == strategy_id, LiveStrategy.user_id == current_user.id).first()

    if not strategy:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Strategy {strategy_id} not found")

    # Update fields
    if parameters is not None:
        strategy.parameters = parameters
        strategy.version += 1

    if max_position_pct is not None:
        strategy.max_position_pct = max_position_pct

    if stop_loss_pct is not None:
        strategy.stop_loss_pct = stop_loss_pct

    if daily_loss_limit is not None:
        strategy.daily_loss_limit = daily_loss_limit

    if notes is not None:
        strategy.notes = notes

    db.commit()

    return {
        "strategy_id": strategy_id,
        "version": strategy.version,
        "updated_at": datetime.utcnow().isoformat(),
        "message": "Strategy updated successfully",
    }


@router.delete("/{strategy_id}")
async def delete_strategy(strategy_id: int, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    """
    Delete a strategy (soft delete - sets status to stopped)
    """
    strategy = db.query(LiveStrategy).filter(LiveStrategy.id == strategy_id, LiveStrategy.user_id == current_user.id).first()

    if not strategy:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Strategy {strategy_id} not found")

    # Soft delete
    strategy.status = StrategyStatus.STOPPED
    strategy.stopped_at = datetime.utcnow()

    db.commit()

    return {"strategy_id": strategy_id, "message": "Strategy deleted successfully"}
