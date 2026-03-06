"""
Celery tasks for background backtest execution

These tasks run in a separate worker process to avoid blocking the API.
Each task creates its own DB engine + session to avoid event-loop conflicts
(asyncpg connections are tied to the loop that created them).
"""

import asyncio
import logging
from datetime import datetime, timezone

from celery.exceptions import SoftTimeLimitExceeded

from ..celery_app import celery_app

logger = logging.getLogger(__name__)


def _create_task_session():
    """Create a fresh async engine + session for this task invocation.

    We can't reuse the global AsyncSessionLocal because asyncio.run() creates
    a new event loop per call, and asyncpg connections are tied to their
    creating loop.
    """
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from ..config import settings

    task_engine = create_async_engine(
        settings.DATABASE_URL,
        future=True,
        pool_size=5,
        max_overflow=2,
    )
    TaskSession = async_sessionmaker(
        task_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )
    return task_engine, TaskSession


async def _run_single_backtest(backtest_id: int, request_data: dict, user_id: int):
    """Async implementation of single backtest task."""

    from ..schemas.backtest import BacktestRequest
    from ..services.backtest_service import BacktestService

    task_engine, TaskSession = _create_task_session()

    try:
        async with TaskSession() as db:
            try:
                # Run the backtest — service handles status updates + result storage
                request = BacktestRequest(**request_data)
                service = BacktestService(db)
                await service.run_single_backtest(request, user_id, backtest_run_id=backtest_id)

                logger.info(f"Backtest {backtest_id} completed successfully")
                return {"status": "completed", "backtest_id": backtest_id}

            except SoftTimeLimitExceeded:
                logger.warning(f"Backtest {backtest_id} timed out (soft limit)")
                error_msg = "Task timed out — backtest took too long"
                await _mark_failed(db, backtest_id, error_msg)
                return {"status": "failed", "backtest_id": backtest_id, "error": error_msg}

            except Exception as e:
                logger.error(f"Backtest {backtest_id} failed: {e}", exc_info=True)
                await _mark_failed(db, backtest_id, str(e)[:500])
                return {"status": "failed", "backtest_id": backtest_id, "error": str(e)}
    finally:
        await task_engine.dispose()


async def _mark_failed(db, backtest_id: int, error_message: str):
    """Mark a backtest run as failed in the database."""
    from sqlalchemy import select

    from ..models import BacktestRun

    try:
        result = await db.execute(select(BacktestRun).where(BacktestRun.id == backtest_id))
        run = result.scalars().first()
        if run and run.status != "failed":
            run.status = "failed"
            run.error_message = error_message
            run.completed_at = datetime.now(timezone.utc)
            await db.commit()
    except Exception as db_err:
        logger.error(f"Failed to update backtest status: {db_err}")


async def _run_multi_backtest(backtest_id: int, request_data: dict, user_id: int):
    """Async implementation of multi-asset backtest task."""

    from ..schemas.backtest import MultiAssetBacktestRequest
    from ..services.backtest_service import BacktestService

    task_engine, TaskSession = _create_task_session()

    try:
        async with TaskSession() as db:
            try:
                # Run the backtest — service handles status updates + result storage
                request = MultiAssetBacktestRequest(**request_data)
                service = BacktestService(db)
                await service.run_multi_asset_backtest(request, user_id, backtest_run_id=backtest_id)

                logger.info(f"Multi backtest {backtest_id} completed successfully")
                return {"status": "completed", "backtest_id": backtest_id}

            except SoftTimeLimitExceeded:
                logger.warning(f"Multi backtest {backtest_id} timed out (soft limit)")
                error_msg = "Task timed out — multi-asset backtest took too long"
                await _mark_failed(db, backtest_id, error_msg)
                return {"status": "failed", "backtest_id": backtest_id, "error": error_msg}

            except Exception as e:
                logger.error(f"Multi backtest {backtest_id} failed: {e}", exc_info=True)
                await _mark_failed(db, backtest_id, str(e)[:500])
                return {"status": "failed", "backtest_id": backtest_id, "error": str(e)}
    finally:
        await task_engine.dispose()


async def _run_wfa(backtest_id: int, request_data: dict, user_id: int):
    """Async implementation of Walk-Forward Analysis task."""
    from sqlalchemy import select

    from ..models import BacktestRun
    from ..schemas.backtest import WFARequest
    from ..services.walk_forward_service import WalkForwardService

    task_engine, TaskSession = _create_task_session()

    try:
        async with TaskSession() as db:
            try:
                # Update status to running
                result = await db.execute(select(BacktestRun).where(BacktestRun.id == backtest_id))
                run = result.scalars().first()
                if run:
                    run.status = "running"
                    await db.commit()

                # Run WFA
                request = WFARequest(**request_data)
                service = WalkForwardService(db, user_id)
                wfa_result = await service.run_wfa(request)

                # Update with results
                result = await db.execute(select(BacktestRun).where(BacktestRun.id == backtest_id))
                run = result.scalars().first()
                if run:
                    run.status = "completed"
                    run.completed_at = datetime.now(timezone.utc)

                    # Store WFA-specific results as JSON
                    if hasattr(wfa_result, "model_dump"):
                        wfa_data = wfa_result.model_dump()
                        run.trades_json = wfa_data

                        # Also store aggregated OOS metrics in the standard columns
                        # so the history list view shows meaningful summary data
                        agg = wfa_data.get("aggregated_oos_metrics") or {}
                        run.total_return_pct = agg.get("total_return_pct")
                        run.sharpe_ratio = agg.get("sharpe_ratio")
                        run.max_drawdown = agg.get("max_drawdown")
                        run.win_rate = agg.get("win_rate")
                        run.total_trades = agg.get("total_trades")
                        run.final_equity = agg.get("final_equity")

                    await db.commit()

                logger.info(f"WFA {backtest_id} completed successfully")
                return {"status": "completed", "backtest_id": backtest_id}

            except SoftTimeLimitExceeded:
                logger.warning(f"WFA {backtest_id} timed out (soft limit)")
                error_msg = "Task timed out — walk-forward analysis took too long"
                await _mark_failed(db, backtest_id, error_msg)
                return {"status": "failed", "backtest_id": backtest_id, "error": error_msg}

            except Exception as e:
                logger.error(f"WFA {backtest_id} failed: {e}", exc_info=True)
                await _mark_failed(db, backtest_id, str(e)[:500])
                return {"status": "failed", "backtest_id": backtest_id, "error": str(e)}
    finally:
        await task_engine.dispose()


async def _run_custom_backtest(backtest_id: int, request_data: dict, user_id: int):
    """Async implementation of custom strategy backtest task."""
    from sqlalchemy import select

    from ..analytics.performance import calculate_performance_metrics
    from ..core.benchmark_calculator import BenchmarkCalculator
    from ..core.custom_strategy_engine import CustomStrategyAdapter
    from ..core.data_fetcher import fetch_stock_data
    from ..core.risk_manager import RiskManager
    from ..core.trading_engine import TradingEngine
    from ..models import BacktestRun
    from ..schemas.custom_strategy import CustomBacktestRequest
    from ..services.backtest_service import _sanitize_value

    task_engine, TaskSession = _create_task_session()

    try:
        async with TaskSession() as db:
            try:
                request = CustomBacktestRequest(**request_data)

                # Mark as running
                result = await db.execute(select(BacktestRun).where(BacktestRun.id == backtest_id))
                run = result.scalars().first()
                if run:
                    run.status = "running"
                    await db.commit()

                # Create adapter from custom code
                adapter = CustomStrategyAdapter(
                    name=request.strategy_name or "Custom Strategy",
                    code=request.code,
                )

                # Fetch market data
                data = await fetch_stock_data(request.symbol, request.period, request.interval)
                if data is None or data.empty:
                    raise ValueError(f"No data available for {request.symbol}")

                # Run backtest using TradingEngine
                engine = TradingEngine(
                    strategy=adapter,
                    initial_capital=request.initial_capital,
                    risk_manager=RiskManager(),
                    commission_rate=request.commission_rate,
                    slippage_rate=request.slippage_rate,
                )
                await engine.run_backtest(request.symbol, data)

                # Calculate benchmark
                benchmark_calc = BenchmarkCalculator(request.initial_capital)
                benchmark = await benchmark_calc.calculate_spy_benchmark(
                    period=request.period,
                    interval=request.interval,
                    commission_rate=request.commission_rate,
                )
                benchmark_equity = benchmark.get("equity_curve") if benchmark else None

                # Calculate metrics
                metrics = calculate_performance_metrics(
                    engine.trades,
                    engine.equity_curve,
                    request.initial_capital,
                    benchmark_equity=benchmark_equity,
                )

                # Store results
                result = await db.execute(select(BacktestRun).where(BacktestRun.id == backtest_id))
                run = result.scalars().first()
                if run:
                    run.status = "completed"
                    run.total_return_pct = metrics.get("total_return_pct")
                    run.sharpe_ratio = metrics.get("sharpe_ratio")
                    run.max_drawdown = metrics.get("max_drawdown")
                    run.win_rate = metrics.get("win_rate")
                    run.total_trades = metrics.get("total_trades")
                    run.final_equity = metrics.get("final_equity")
                    run.equity_curve = _sanitize_value(engine.equity_curve)
                    run.trades_json = _sanitize_value(engine.trades)
                    run.extended_results = _sanitize_value(metrics)
                    run.completed_at = datetime.now(timezone.utc)
                    await db.commit()

                logger.info(f"Custom backtest {backtest_id} completed successfully")
                return {"status": "completed", "backtest_id": backtest_id}

            except SoftTimeLimitExceeded:
                logger.warning(f"Custom backtest {backtest_id} timed out (soft limit)")
                error_msg = "Task timed out — custom strategy backtest took too long"
                await _mark_failed(db, backtest_id, error_msg)
                return {"status": "failed", "backtest_id": backtest_id, "error": error_msg}

            except Exception as e:
                logger.error(f"Custom backtest {backtest_id} failed: {e}", exc_info=True)
                await _mark_failed(db, backtest_id, str(e)[:500])
                return {"status": "failed", "backtest_id": backtest_id, "error": str(e)}
    finally:
        await task_engine.dispose()


# ---------------------------------------------------------------------------
# Celery task wrappers (sync → async bridge using asyncio.run)
# ---------------------------------------------------------------------------


@celery_app.task(
    bind=True,
    name="app.tasks.backtest_tasks.run_single_backtest_task",
    time_limit=1800,  # 30 min hard kill
    soft_time_limit=1500,  # 25 min soft limit
)
def run_single_backtest_task(self, backtest_id: int, request_data: dict, user_id: int):
    """Celery task: run single-asset backtest in background."""
    logger.info(f"[Celery] Starting single backtest {backtest_id}")
    return asyncio.run(_run_single_backtest(backtest_id, request_data, user_id))


@celery_app.task(
    bind=True,
    name="app.tasks.backtest_tasks.run_multi_backtest_task",
    time_limit=3600,  # 60 min hard kill
    soft_time_limit=3300,  # 55 min soft limit
)
def run_multi_backtest_task(self, backtest_id: int, request_data: dict, user_id: int):
    """Celery task: run multi-asset backtest in background."""
    logger.info(f"[Celery] Starting multi backtest {backtest_id}")
    return asyncio.run(_run_multi_backtest(backtest_id, request_data, user_id))


@celery_app.task(
    bind=True,
    name="app.tasks.backtest_tasks.run_wfa_task",
    time_limit=7200,  # 2 hour hard kill (Optuna trials are slow)
    soft_time_limit=6900,  # 1h55m soft limit
)
def run_wfa_task(self, backtest_id: int, request_data: dict, user_id: int):
    """Celery task: run Walk-Forward Analysis in background."""
    logger.info(f"[Celery] Starting WFA {backtest_id}")
    return asyncio.run(_run_wfa(backtest_id, request_data, user_id))


@celery_app.task(
    bind=True,
    name="app.tasks.backtest_tasks.run_custom_backtest_task",
    time_limit=600,  # 10 min hard kill (custom code shouldn't run longer)
    soft_time_limit=540,  # 9 min soft limit
)
def run_custom_backtest_task(self, backtest_id: int, request_data: dict, user_id: int):
    """Celery task: run custom strategy backtest in background."""
    logger.info(f"[Celery] Starting custom backtest {backtest_id}")
    return asyncio.run(_run_custom_backtest(backtest_id, request_data, user_id))
