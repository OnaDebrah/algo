"""
Backtest service with pairs trading support
"""

import asyncio
import logging
import math
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.analytics.performance import calculate_performance_metrics
from backend.app.core.benchmark_calculator import BenchmarkCalculator
from backend.app.core.data_fetcher import fetch_stock_data
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
from backend.app.services.trading_service import TradingService
from backend.app.strategies.strategy_catalog import get_catalog

logger = logging.getLogger(__name__)


def _sanitize_value(value):
    """Recursively sanitize values to ensure JSON compliance (no inf/NaN)."""
    if isinstance(value, (float, np.floating)):
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return float(value)
    elif isinstance(value, (np.integer,)):
        return int(value)
    elif isinstance(value, dict):
        return {k: _sanitize_value(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    return value


# ML strategy keys that require training before backtesting.
# These strategies have is_trained flags and return zeros if not trained.
ML_STRATEGY_KEYS = {"ml_random_forest", "ml_gradient_boosting", "ml_svm", "ml_logistic", "ml_lstm"}

# ML strategies that self-train during signal generation (no explicit train() needed for backtest)
ML_SELF_TRAINING_KEYS = {"mc_ml_sentiment"}


class BacktestService:
    """Service for running backtests with database persistence"""

    def __init__(self, db: AsyncSession = None):
        """
        Initialize backtest service

        Args:
            db: AsyncSession for SQLAlchemy (BacktestRun model)
        """
        self.catalog = get_catalog()
        self.trading_service = TradingService()
        self.risk_manager = RiskManager()
        self.db = db

    async def create_backtest_run(
        self,
        user_id: int,
        backtest_type: str,
        symbols: List[str],
        strategy_config: dict,
        period: str,
        interval: str,
        initial_capital: float,
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
                await self.update_backtest_status(backtest_run.id, "running")

            # Validate strategy is compatible with single-asset mode
            strategy_info = self.catalog.get_info(request.strategy_key)
            if strategy_info and strategy_info.backtest_mode == "multi":
                raise ValueError(f"Strategy '{strategy_info.name}' requires multi-asset mode. " f"Please use the multi-asset backtest instead.")

            # Fetch data (ensure we have enough for warm-up)
            # We fetch 'max' or the requested period, then slice manually to ensure history exists
            full_data = await fetch_stock_data(request.symbol, "max" if request.start_date else request.period, request.interval)

            if full_data.empty:
                raise ValueError(f"No data available for {request.symbol}")

            # Slice data to target range + warm-up
            if request.start_date:
                try:
                    target_idx = full_data.index.get_indexer([pd.Timestamp(request.start_date)], method="bfill")[0]
                    warm_up_idx = max(0, target_idx - 250)
                    data = full_data.iloc[warm_up_idx:].loc[: request.end_date]
                except Exception as e:
                    logger.warning(f"Metadata slicing failed: {e}")
                    data = full_data
            else:
                data = full_data

            # Create strategy — for ML strategies, either load a deployed model or auto-train
            if request.strategy_key in ML_STRATEGY_KEYS:
                strategy = None
                if request.ml_model_id:
                    # Try to load a pre-trained deployed model from ML Studio
                    try:
                        from backend.app.api.routes.mlstudio import load_model

                        strategy = load_model(request.ml_model_id)
                        logger.info(f"Loaded deployed ML model '{request.ml_model_id}' for backtest")
                    except (ValueError, FileNotFoundError) as e:
                        logger.warning(f"Could not load ML model '{request.ml_model_id}': {e}. Falling back to auto-train.")

                if strategy is None:
                    # No deployed model or model not found — auto-train on the data
                    strategy = self.catalog.create_strategy(request.strategy_key, **request.parameters)
                    logger.info(f"Auto-training ML strategy '{request.strategy_key}' on backtest data")
                    try:
                        await asyncio.to_thread(strategy.train, data)
                        logger.info(f"ML strategy auto-trained: is_trained={getattr(strategy, 'is_trained', 'N/A')}")
                    except Exception as e:
                        raise ValueError(f"ML auto-training failed: {str(e)}")
            elif request.strategy_key in ML_SELF_TRAINING_KEYS:
                # Self-training ML strategies (e.g., MC ML Sentiment) — they train
                # during signal generation, so just create the strategy instance
                strategy = self.catalog.create_strategy(request.strategy_key, **request.parameters)
                logger.info(f"Created self-training ML strategy '{request.strategy_key}' — will train during signal generation")
            else:
                strategy = self.catalog.create_strategy(request.strategy_key, **request.parameters)

            # Run backtest
            engine = TradingEngine(
                strategy=strategy,
                initial_capital=request.initial_capital,
                risk_manager=self.risk_manager,
                trading_service=self.trading_service,
                commission_rate=request.commission_rate,
                slippage_rate=request.slippage_rate,
            )

            await asyncio.to_thread(
                engine.run_backtest, request.symbol, data, start_timestamp=pd.Timestamp(request.start_date) if request.start_date else None
            )

            # Calculate benchmark FIRST so we can pass its equity to factor metrics
            benchmark_calc = BenchmarkCalculator(request.initial_capital)
            benchmark = await benchmark_calc.calculate_spy_benchmark(
                period=request.period, interval=request.interval, commission_rate=request.commission_rate
            )

            # Extract benchmark equity curve for alpha/beta/R² calculation
            benchmark_equity = benchmark.get("equity_curve") if benchmark else None

            # Calculate metrics (with benchmark for factor attribution)
            metrics: dict = calculate_performance_metrics(
                engine.trades, engine.equity_curve, request.initial_capital, benchmark_equity=benchmark_equity
            )

            if benchmark:
                benchmark["comparison"] = benchmark_calc.compare_to_benchmark(metrics, benchmark)
                # Sanitize benchmark to prevent inf/NaN JSON serialization errors
                benchmark = _sanitize_value(benchmark)

            # Create equity curve
            equity_curve: list[EquityCurvePoint] = [
                EquityCurvePoint(timestamp=str(point["timestamp"]), equity=point["equity"], cash=point["cash"]) for point in engine.equity_curve
            ]

            # Calculate drawdowns
            equity_series = pd.Series([point.equity for point in equity_curve])
            running_max = equity_series.expanding().max()
            drawdown = (equity_series - running_max) / running_max
            # Replace NaN/inf values in drawdown (e.g. when running_max is 0)
            drawdown = drawdown.fillna(0.0).replace([float("inf"), float("-inf")], 0.0)

            for i, point in enumerate(equity_curve):
                point.drawdown = float(drawdown.iloc[i])

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

            # Build OHLC price data for frontend chart (candlestick + trade markers)
            ohlc_price_data = None
            try:
                ohlc_df = data[["Open", "High", "Low", "Close", "Volume"]].copy()
                ohlc_df.index = ohlc_df.index.astype(str)
                ohlc_price_data = [
                    {
                        "timestamp": ts,
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "volume": float(row["Volume"]),
                    }
                    for ts, row in ohlc_df.iterrows()
                ]
                ohlc_price_data = _sanitize_value(ohlc_price_data)
            except Exception as e:
                logger.warning(f"Failed to build OHLC price data: {e}")

            # Update database with results
            if backtest_run:
                update_data = metrics.copy()
                update_data["equity_curve"] = [p.model_dump() for p in equity_curve]
                update_data["trades"] = [t.model_dump() for t in trades]
                await self.update_backtest_status(backtest_run.id, "completed", update_data)

            return BacktestResponse(result=result, equity_curve=equity_curve, trades=trades, price_data=ohlc_price_data, benchmark=benchmark)

        except Exception as e:
            logger.error(f"Backtest failed: {e}", exc_info=True)
            if backtest_run:
                await self.update_backtest_status(backtest_run.id, "failed", error_message=str(e))
            raise ValueError(f"Backtest failed: {str(e)}")

    async def run_multi_asset_backtest(self, request: MultiAssetBacktestRequest, user_id: int) -> MultiAssetBacktestResponse:
        """Run multi-asset backtest (supports both independent and pairs trading)"""
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

            # Validate strategies are compatible with multi-asset mode
            for symbol, config in request.strategy_configs.items():
                strategy_info = self.catalog.get_info(config.strategy_key)
                if strategy_info and strategy_info.backtest_mode == "single":
                    raise ValueError(
                        f"Strategy '{strategy_info.name}' only supports single-asset mode. " f"It cannot be used in multi-asset backtests."
                    )

            # Detect if this is a pairs trading strategy
            is_pairs_strategy = self._is_pairs_strategy(request.strategy_configs, request.symbols)

            if is_pairs_strategy:
                # PAIRS TRADING MODE (e.g., Kalman Filter)
                logger.info("Running pairs trading backtest")
                engine = await self._create_pairs_engine(request)
            else:
                # INDEPENDENT STRATEGIES MODE
                logger.info("Running independent strategies backtest")
                engine = await self._create_independent_engine(request)

            # Run backtest (run_backtest is async — must be awaited directly, not via to_thread)
            await engine.run_backtest(request.symbols, request.period, request.interval)
            results = engine.get_results()
            # Create equity curve
            equity_curve = [
                EquityCurvePoint(timestamp=str(point["timestamp"]), equity=point["equity"], cash=point["cash"]) for point in engine.equity_curve
            ]

            # Calculate drawdowns
            equity_series = pd.Series([point.equity for point in equity_curve])
            running_max = equity_series.expanding().max()
            drawdown = (equity_series - running_max) / running_max
            # Replace NaN/inf values in drawdown (e.g. when running_max is 0)
            drawdown = drawdown.fillna(0.0).replace([float("inf"), float("-inf")], 0.0)

            for i, point in enumerate(equity_curve):
                point.drawdown = float(drawdown.iloc[i])

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
                    data = await fetch_stock_data(symbol, request.period, request.interval)
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
                    # Sanitize benchmark to prevent inf/NaN JSON serialization errors
                    benchmark = _sanitize_value(benchmark)

            # Build OHLC price data for frontend chart (use first symbol's data for multi-asset)
            ohlc_price_data = None
            try:
                if data_dict:
                    first_symbol = request.symbols[0]
                    if first_symbol in data_dict:
                        ohlc_df = data_dict[first_symbol][["Open", "High", "Low", "Close", "Volume"]].copy()
                        ohlc_df.index = ohlc_df.index.astype(str)
                        ohlc_price_data = [
                            {
                                "timestamp": ts,
                                "open": float(row["Open"]),
                                "high": float(row["High"]),
                                "low": float(row["Low"]),
                                "close": float(row["Close"]),
                                "volume": float(row["Volume"]),
                            }
                            for ts, row in ohlc_df.iterrows()
                        ]
                        ohlc_price_data = _sanitize_value(ohlc_price_data)
            except Exception as e:
                logger.warning(f"Failed to build OHLC price data for multi-asset: {e}")

            # Update database
            if backtest_run:
                update_data = results.model_dump()
                update_data["equity_curve"] = [p.model_dump() for p in equity_curve]
                update_data["trades"] = [t.model_dump() for t in trades]
                await self.update_backtest_status(backtest_run.id, "completed", update_data)

            return MultiAssetBacktestResponse(
                result=results, equity_curve=equity_curve, trades=trades, price_data=ohlc_price_data, benchmark=benchmark
            )

        except Exception as e:
            logger.error(f"Multi-asset backtest failed: {e}", exc_info=True)
            if backtest_run:
                await self.update_backtest_status(backtest_run.id, "failed", error_message=str(e))
            raise ValueError(f"Multi-asset backtest failed: {str(e)}")

    def _is_pairs_strategy(self, strategy_configs: dict, symbols: List[str] = None) -> bool:
        """
        Detect if this is a pairs trading strategy

        A pairs strategy typically:
        1. Has exactly 2 symbols (either in strategy_configs or provided symbols list)
        2. Uses the same strategy for both
        3. Strategy has pairs-specific parameters or key
        """
        # A pair must involve exactly 2 symbols
        num_symbols = len(symbols) if symbols else len(strategy_configs)
        if num_symbols != 2:
            logger.debug(f"Not a pairs strategy: expected 2 symbols, got {num_symbols}")
            return False

        # Get all strategy configs
        configs = list(strategy_configs.values())
        if not configs:
            return False

        # Check if all use the same strategy key (if multiple provided)
        strategy_keys = [s.strategy_key for s in configs]
        if len(set(strategy_keys)) != 1:
            logger.debug(f"Not a pairs strategy: different strategy keys {strategy_keys}")
            return False

        # Check for known pairs strategy keys or parameters
        strategy_key = strategy_keys[0]
        if strategy_key in ["kalman_filter", "pairs_trading"]:
            logger.info(f"Detected pairs strategy by key: {strategy_key}")
            return True

        # Fallback to parameter check
        params = configs[0].parameters
        if "asset_1" in params and "asset_2" in params:
            logger.info(f"Detected pairs strategy by parameters: {strategy_key}")
            return True

        return False

    async def _create_pairs_engine(self, request: MultiAssetBacktestRequest) -> MultiAssetEngine:
        """Create engine for pairs trading with correct asset mapping"""

        # Get the first strategy config
        first_config = list(request.strategy_configs.values())[0]
        params = first_config.parameters.copy()

        # Ensure asset_1 and asset_2 in params match the symbols in the request
        # if they were not explicitly provided or if we want to force alignment
        symbols = request.symbols
        if len(symbols) == 2:
            # Update params to match selected symbols if they are generic or missing
            # This handles cases where the user selects symbols in UI that differ from strategy defaults
            params["asset_1"] = symbols[0]
            params["asset_2"] = symbols[1]
            logger.info(f"Mapping pairs assets: {symbols[0]} and {symbols[1]}")

        # Create the pairs strategy with updated params
        pairs_strategy = self.catalog.create_strategy(first_config.strategy_key, **params)

        # Create engine in pairs mode
        engine = MultiAssetEngine(
            strategies=pairs_strategy,
            initial_capital=request.initial_capital,
            risk_manager=self.risk_manager,
            allocation_method=request.allocation_method,
            commission_rate=request.commission_rate,
            slippage_rate=request.slippage_rate,
            pairs_mode=True,
            pair_symbols=symbols,
        )

        if request.custom_allocations:
            engine.allocations = request.custom_allocations

        return engine

    async def _create_independent_engine(self, request: MultiAssetBacktestRequest) -> MultiAssetEngine:
        """Create engine for independent strategies (original behavior)"""

        # Create strategies for each symbol
        strategies = {}
        for symbol, config in request.strategy_configs.items():
            strategy = self.catalog.create_strategy(config.strategy_key, **config.parameters)
            strategies[symbol] = strategy

        # Create engine in independent mode
        engine = MultiAssetEngine(
            strategies=strategies,
            initial_capital=request.initial_capital,
            risk_manager=self.risk_manager,
            allocation_method=request.allocation_method,
            commission_rate=request.commission_rate,
            slippage_rate=request.slippage_rate,
            pairs_mode=False,  # Independent mode
        )

        # Override allocations if custom
        if request.custom_allocations:
            engine.allocations = request.custom_allocations

        return engine
