"""
Enhanced Backtest service with database integration
"""

import asyncio
from datetime import datetime
from typing import List

from sqlalchemy.orm import Session

from backend.app.models.backtest import BacktestRun
from backend.app.schemas.backtest import (
    BacktestHistoryItem,
    BacktestRequest,
    BacktestResponse,
    BacktestResult,
    EquityCurvePoint,
    MultiAssetBacktestRequest,
    MultiAssetBacktestResponse,
    MultiAssetBacktestResult,
    SymbolStats,
    Trade,
)
from core.data_fetcher import fetch_stock_data
from core.database import DatabaseManager
from core.multi_asset_engine import MultiAssetEngine
from core.risk_manager import RiskManager
from core.trading_engine import TradingEngine
from strategies.strategy_catalog import get_catalog


class BacktestService:
    """Service for running backtests with database persistence"""

    def __init__(self, db: Session = None):
        self.catalog = get_catalog()
        self.db_manager = DatabaseManager()
        self.risk_manager = RiskManager()
        self.db = db

    def create_backtest_run(
        self, user_id: int, backtest_type: str, symbols: List[str], strategy_config: dict, period: str, interval: str, initial_capital: float
    ) -> BacktestRun:
        """Create a new backtest run in database"""
        backtest_run = BacktestRun(
            user_id=user_id,
            backtest_type=backtest_type,
            symbols=symbols,
            strategy_config=strategy_config,
            period=period,
            interval=interval,
            initial_capital=initial_capital,
            status="pending",
        )

        if self.db:
            self.db.add(backtest_run)
            self.db.commit()
            self.db.refresh(backtest_run)

        return backtest_run

    def update_backtest_status(self, backtest_id: int, status: str, results: dict = None, error_message: str = None):
        """Update backtest run status"""
        if not self.db:
            return

        backtest_run = self.db.query(BacktestRun).filter(BacktestRun.id == backtest_id).first()

        if backtest_run:
            backtest_run.status = status
            if results:
                backtest_run.total_return = results.get("total_return")
                backtest_run.total_return_pct = results.get("total_return_pct")
                backtest_run.sharpe_ratio = results.get("sharpe_ratio")
                backtest_run.max_drawdown = results.get("max_drawdown")
                backtest_run.win_rate = results.get("win_rate")
                backtest_run.total_trades = results.get("total_trades")
                backtest_run.final_equity = results.get("final_equity")

            if error_message:
                backtest_run.error_message = error_message

            if status in ["completed", "failed"]:
                backtest_run.completed_at = datetime.now()

            self.db.commit()

    async def run_single_backtest(self, request: BacktestRequest, user_id: int) -> BacktestResponse:
        """Run single asset backtest"""
        # Create database record
        backtest_run = None
        if self.db:
            backtest_run = self.create_backtest_run(
                user_id=user_id,
                backtest_type="single",
                symbols=[request.symbol],
                strategy_config={"strategy_key": request.strategy_key, "parameters": request.parameters},
                period=request.period,
                interval=request.interval,
                initial_capital=request.initial_capital,
            )

        try:
            # Update status to running
            if backtest_run:
                self.update_backtest_status(backtest_run.id, "running")

            # Fetch data
            data = await asyncio.to_thread(fetch_stock_data, request.symbol, request.period, request.interval)

            if data.empty:
                raise ValueError(f"No data available for {request.symbol}")

            # Create strategy
            strategy = self.catalog.create_strategy(request.strategy_key, **request.parameters)

            # Run backtest
            engine = TradingEngine(
                strategy=strategy,
                initial_capital=request.initial_capital,
                risk_manager=self.risk_manager,
                db_manager=self.db_manager,
                commission_rate=request.commission_rate,
                slippage_rate=request.slippage_rate,
            )

            await asyncio.to_thread(engine.run_backtest, request.symbol, data)

            # Calculate metrics
            from analytics.performance import calculate_performance_metrics

            metrics = calculate_performance_metrics(engine.trades, engine.equity_curve, request.initial_capital)

            # Update database with results
            if backtest_run:
                self.update_backtest_status(backtest_run.id, "completed", metrics)

            # Build response
            result = BacktestResult(**metrics)

            equity_curve = [
                EquityCurvePoint(timestamp=str(point["timestamp"]), equity=point["equity"], cash=point["cash"]) for point in engine.equity_curve
            ]

            trades = [
                Trade(
                    symbol=t["symbol"],
                    order_type=t["order_type"],
                    quantity=t["quantity"],
                    price=t["price"],
                    commission=t["commission"],
                    timestamp=t["timestamp"],
                    strategy=t["strategy"],
                    profit=t.get("profit"),
                    profit_pct=t.get("profit_pct"),
                )
                for t in engine.trades
            ]

            return BacktestResponse(result=result, equity_curve=equity_curve, trades=trades)

        except Exception as e:
            # Update database with error
            if backtest_run:
                self.update_backtest_status(backtest_run.id, "failed", error_message=str(e))
            raise ValueError(f"Backtest failed: {str(e)}")

    async def run_multi_asset_backtest(self, request: MultiAssetBacktestRequest, user_id: int) -> MultiAssetBacktestResponse:
        """Run multi-asset backtest"""
        # Create database record
        backtest_run = None
        if self.db:
            backtest_run = self.create_backtest_run(
                user_id=user_id,
                backtest_type="multi",
                symbols=request.symbols,
                strategy_config={
                    "strategy_configs": {
                        k: {"strategy_key": v.strategy_key, "parameters": v.parameters} for k, v in request.strategy_configs.items()
                    },
                    "allocation_method": request.allocation_method,
                    "custom_allocations": request.custom_allocations,
                },
                period=request.period,
                interval=request.interval,
                initial_capital=request.initial_capital,
            )

        try:
            # Update status
            if backtest_run:
                self.update_backtest_status(backtest_run.id, "running")

            # Create strategies for each symbol
            strategies = {}
            for symbol, config in request.strategy_configs.items():
                strategy = self.catalog.create_strategy(config.strategy_key, **config.parameters)
                strategies[symbol] = strategy

            # Create engine
            engine = MultiAssetEngine(
                strategies=strategies,
                initial_capital=request.initial_capital,
                risk_manager=self.risk_manager,
                db=self.db_manager,
                allocation_method=request.allocation_method,
                commission_rate=request.commission_rate,
                slippage_rate=request.slippage_rate,
            )

            # Override allocations if custom
            if request.custom_allocations:
                engine.allocations = request.custom_allocations

            # Run backtest
            await asyncio.to_thread(engine.run_backtest, request.symbols, request.period, request.interval)

            # Get results
            results = engine.get_results()

            # Update database
            if backtest_run:
                self.update_backtest_status(backtest_run.id, "completed", results)

            # Build response
            result = MultiAssetBacktestResult(
                total_return=results["total_return"],
                total_return_pct=results["total_return"],
                win_rate=results["win_rate"],
                sharpe_ratio=results["sharpe_ratio"],
                max_drawdown=results["max_drawdown"],
                total_trades=results["total_trades"],
                winning_trades=results.get("winning_trades", 0),
                losing_trades=results.get("losing_trades", 0),
                avg_profit=results["avg_profit"],
                avg_win=results.get("avg_win", 0),
                avg_loss=results.get("avg_loss", 0),
                profit_factor=results.get("profit_factor", 0),
                final_equity=results["final_equity"],
                initial_capital=request.initial_capital,
                symbol_stats={symbol: SymbolStats(**stats) for symbol, stats in results["symbol_stats"].items()},
                num_symbols=results["num_symbols"],
            )

            equity_curve = [
                EquityCurvePoint(timestamp=str(point["timestamp"]), equity=point["equity"], cash=point["cash"]) for point in engine.equity_curve
            ]

            trades = [
                Trade(
                    symbol=t["symbol"],
                    order_type=t["order_type"],
                    quantity=t["quantity"],
                    price=t["price"],
                    commission=t["commission"],
                    timestamp=t["timestamp"],
                    strategy=t["strategy"],
                    profit=t.get("profit"),
                    profit_pct=t.get("profit_pct"),
                )
                for t in engine.trades
            ]

            return MultiAssetBacktestResponse(result=result, equity_curve=equity_curve, trades=trades)

        except Exception as e:
            if backtest_run:
                self.update_backtest_status(backtest_run.id, "failed", error_message=str(e))
            raise ValueError(f"Multi-asset backtest failed: {str(e)}")

    def get_backtest_history(self, user_id: int, limit: int = 20, backtest_type: str = None) -> List[BacktestHistoryItem]:
        """Get user's backtest history"""
        if not self.db:
            return []

        query = self.db.query(BacktestRun).filter(BacktestRun.user_id == user_id)

        if backtest_type:
            query = query.filter(BacktestRun.backtest_type == backtest_type)

        runs = query.order_by(BacktestRun.created_at.desc()).limit(limit).all()

        return [
            BacktestHistoryItem(
                id=run.id,
                name=run.name,
                backtest_type=run.backtest_type,
                symbols=run.symbols,
                strategy_config=run.strategy_config,
                period=run.period,
                interval=run.interval,
                initial_capital=run.initial_capital,
                total_return_pct=run.total_return_pct,
                sharpe_ratio=run.sharpe_ratio,
                max_drawdown=run.max_drawdown,
                status=run.status,
                created_at=run.created_at.isoformat() if run.created_at else None,
                completed_at=run.completed_at.isoformat() if run.completed_at else None,
            )
            for run in runs
        ]
