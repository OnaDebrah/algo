"""
Enhanced Backtest service with database integration
"""

import asyncio
import logging
from datetime import datetime
from typing import List

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.analytics.performance import calculate_performance_metrics
from backend.app.core.benchmark_calculator import BenchmarkCalculator
from backend.app.core.data_fetcher import fetch_stock_data
from backend.app.core.database import DatabaseManager
from backend.app.core.multi_asset_engine import MultiAssetEngine
from backend.app.core.risk_manager import RiskManager
from backend.app.core.trading_engine import TradingEngine
from backend.app.models.backtest import BacktestRun
from backend.app.schemas.backtest import (
    BacktestRequest,
    BacktestResponse,
    BacktestResult,
    EquityCurvePoint,
    MultiAssetBacktestRequest,
    MultiAssetBacktestResponse,
    Trade,
)
from backend.app.strategies.strategy_catalog import get_catalog

logger = logging.getLogger(__name__)


class BacktestService:
    """Service for running backtests with database persistence"""

    def __init__(self, db: AsyncSession = None):
        """
        Initialize backtest service

        Args:
            db: AsyncSession for SQLAlchemy (BacktestRun model)
        """
        self.catalog = get_catalog()
        self.db_manager = DatabaseManager()  # For trades/performance (psycopg2)
        self.risk_manager = RiskManager()
        self.db = db  # For BacktestRun model (SQLAlchemy async)

    async def create_backtest_run(
        self, user_id: int, backtest_type: str, symbols: List[str], strategy_config: dict, period: str, interval: str, initial_capital: float
    ) -> BacktestRun:
        """Create a new backtest run in database"""
        if not self.db:
            raise ValueError("Database session not provided")

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

        self.db.add(backtest_run)
        await self.db.commit()
        await self.db.refresh(backtest_run)

        logger.info(f"Created backtest run ID: {backtest_run.id}")
        return backtest_run

    async def update_backtest_status(self, backtest_id: int, status: str, results: dict = None, error_message: str = None):
        """Update backtest run status"""
        if not self.db:
            logger.warning("No database session - skipping status update")
            return

        stmt = select(BacktestRun).where(BacktestRun.id == backtest_id)
        result = await self.db.execute(stmt)
        backtest_run = result.scalars().first()

        if not backtest_run:
            logger.error(f"Backtest run {backtest_id} not found")
            return

        backtest_run.status = status

        if results:
            backtest_run.total_return = results.get("total_return")
            backtest_run.total_return_pct = results.get("total_return_pct")
            backtest_run.sharpe_ratio = results.get("sharpe_ratio")
            backtest_run.max_drawdown = results.get("max_drawdown")
            backtest_run.win_rate = results.get("win_rate")
            backtest_run.total_trades = results.get("total_trades")
            backtest_run.final_equity = results.get("final_equity")
            backtest_run.equity_curve = results.get("equity_curve")
            backtest_run.trades_json = results.get("trades")

        if error_message:
            backtest_run.error_message = error_message

        if status in ["completed", "failed"]:
            backtest_run.completed_at = datetime.now()

        await self.db.commit()
        await self.db.refresh(backtest_run)

        logger.info(f"Updated backtest {backtest_id} - Status: {status}")
        if results:
            logger.info(f"  Equity curve points: {len(results.get('equity_curve', []))}")
            logger.info(f"  Trades: {len(results.get('trades', []))}")

    async def run_single_backtest(self, request: BacktestRequest, user_id: int) -> BacktestResponse:
        """Run single asset backtest"""
        backtest_run = None

        try:
            # Create database record
            if self.db:
                backtest_run = await self.create_backtest_run(
                    user_id=user_id,
                    backtest_type="single",
                    symbols=[request.symbol],
                    strategy_config={"strategy_key": request.strategy_key, "parameters": request.parameters},
                    period=request.period,
                    interval=request.interval,
                    initial_capital=request.initial_capital,
                )

                # Update status to running
                await self.update_backtest_status(backtest_run.id, "running")

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
            metrics: dict = calculate_performance_metrics(engine.trades, engine.equity_curve, request.initial_capital)

            # Create equity curve
            equity_curve: list[EquityCurvePoint] = [
                EquityCurvePoint(timestamp=str(point["timestamp"]), equity=point["equity"], cash=point["cash"]) for point in engine.equity_curve
            ]

            # Calculate drawdowns
            equity_series = pd.Series([point.equity for point in equity_curve])
            running_max = equity_series.expanding().max()
            drawdown = (equity_series - running_max) / running_max

            for i, point in enumerate(equity_curve):
                point.drawdown = drawdown.iloc[i]

            # Create result
            result = BacktestResult(**metrics)

            # Create trades
            trades = [
                Trade(
                    symbol=t["symbol"],
                    order_type=t["order_type"],
                    quantity=t["quantity"],
                    price=t["price"],
                    commission=t["commission"],
                    executed_at=t.get("executed_at") or t.get("timestamp"),
                    total_value=t.get("total_value") or (t["quantity"] * t["price"]),
                    side=t.get("side") or t.get("order_type"),
                    notes=t.get("notes"),
                    strategy=t["strategy"],
                    profit=t.get("profit"),
                    profit_pct=t.get("profit_pct"),
                )
                for t in engine.trades
            ]

            # Calculate benchmark
            benchmark_calc = BenchmarkCalculator(request.initial_capital)
            benchmark = benchmark_calc.calculate_spy_benchmark(
                period=request.period, interval=request.interval, commission_rate=request.commission_rate
            )

            if benchmark:
                benchmark["comparison"] = benchmark_calc.compare_to_benchmark(metrics, benchmark)

            # Update database with results
            if backtest_run:
                update_data = metrics.copy()
                update_data["equity_curve"] = [p.model_dump() for p in equity_curve]
                update_data["trades"] = [t.model_dump() for t in trades]

                # Note: Trades are already saved by TradingEngine via db_manager
                # We just update the BacktestRun record here
                await self.update_backtest_status(backtest_run.id, "completed", update_data)

            return BacktestResponse(result=result, equity_curve=equity_curve, trades=trades, price_data=engine.trades, benchmark=benchmark)

        except Exception as e:
            logger.error(f"Backtest failed: {e}", exc_info=True)
            if backtest_run:
                await self.update_backtest_status(backtest_run.id, "failed", error_message=str(e))
            raise ValueError(f"Backtest failed: {str(e)}")

    async def run_multi_asset_backtest(self, request: MultiAssetBacktestRequest, user_id: int) -> MultiAssetBacktestResponse:
        """Run multi-asset backtest"""
        backtest_run = None

        try:
            # Create database record
            if self.db:
                backtest_run = await self.create_backtest_run(
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

                await self.update_backtest_status(backtest_run.id, "running")

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

            results = engine.get_results()

            # Create equity curve
            equity_curve = [
                EquityCurvePoint(timestamp=str(point["timestamp"]), equity=point["equity"], cash=point["cash"]) for point in engine.equity_curve
            ]

            # Calculate drawdowns
            equity_series = pd.Series([point.equity for point in equity_curve])
            running_max = equity_series.expanding().max()
            drawdown = (equity_series - running_max) / running_max

            for i, point in enumerate(equity_curve):
                point.drawdown = drawdown.iloc[i]

            # Create trades
            trades = [
                Trade(
                    symbol=t["symbol"],
                    order_type=t["order_type"],
                    quantity=t["quantity"],
                    price=t["price"],
                    commission=t["commission"],
                    executed_at=t.get("executed_at") or t.get("timestamp"),
                    total_value=t.get("total_value") or (t["quantity"] * t["price"]),
                    side=t.get("side") or t.get("order_type"),
                    notes=t.get("notes"),
                    strategy=t["strategy"],
                    profit=t.get("profit"),
                    profit_pct=t.get("profit_pct"),
                )
                for t in engine.trades
            ]

            # Calculate benchmark
            benchmark_calc = BenchmarkCalculator(request.initial_capital)
            data_dict = {}

            for symbol in request.symbols:
                try:
                    data = await asyncio.to_thread(fetch_stock_data, symbol, request.period, request.interval)
                    if not data.empty:
                        data_dict[symbol] = data
                except Exception as e:
                    logger.warning(f"Failed to get benchmark data for {symbol}: {e}")

            benchmark = None
            if data_dict:
                benchmark = benchmark_calc.calculate_multi_benchmark(
                    symbols=list(data_dict.keys()),
                    data_dict=data_dict,
                    allocations=request.custom_allocations if request.allocation_method == "custom" else None,
                    commission_rate=request.commission_rate,
                )

                if benchmark:
                    benchmark["comparison"] = benchmark_calc.compare_to_benchmark(results.model_dump(), benchmark)

            # Update database
            if backtest_run:
                update_data = results.model_dump()
                update_data["equity_curve"] = [p.model_dump() for p in equity_curve]
                update_data["trades"] = [t.model_dump() for t in trades]

                await self.update_backtest_status(backtest_run.id, "completed", update_data)

            return MultiAssetBacktestResponse(result=results, equity_curve=equity_curve, trades=trades, price_data=engine.trades, benchmark=benchmark)

        except Exception as e:
            logger.error(f"Multi-asset backtest failed: {e}", exc_info=True)
            if backtest_run:
                await self.update_backtest_status(backtest_run.id, "failed", error_message=str(e))
            raise ValueError(f"Multi-asset backtest failed: {str(e)}")
