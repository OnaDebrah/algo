import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from ..analytics.performance import calculate_performance_metrics
from ..core.data_fetcher import fetch_stock_data
from ..optimise.walk_forward_optimiser import OptimizationConfig, OptimizationResult, PrunerType, SamplerType, WalkForwardOptimizer
from ..schemas.backtest import BacktestResult, EquityCurvePoint, WFAFoldResult, WFARequest, WFAResponse
from ..schemas.optimise import OptimizationRequest, ParamRange as OptParamRange
from ..services.backtest_service import BacktestService

logger = logging.getLogger(__name__)


class WalkForwardService:
    """
    Service for running Walk-Forward Analysis with modular optimization.
    Optimizes parameters on In-Sample windows and validates on Out-of-Sample windows.
    """

    def __init__(self, db: AsyncSession = None, user_id: int = None):
        self.db = db
        self.user_id = user_id
        self.backtest_service = BacktestService(db)
        self._optimizer_cache = {}  # Cache optimizers by config

    def _create_optimizer(self, request: WFARequest) -> WalkForwardOptimizer:
        """Create or retrieve cached optimizer for the request"""
        cache_key = f"{request.strategy_key}_{request.n_trials}"

        if cache_key not in self._optimizer_cache:
            config = OptimizationConfig(
                n_trials=request.n_trials,
                n_jobs=getattr(request, "n_jobs", 1),
                random_seed=request.random_seed if hasattr(request, "random_seed") else 42,
                sampler_type=SamplerType.TPE,
                pruner_type=PrunerType.MEDIAN,
                early_stopping_rounds=10,
                test_size=0.2,  # Use 20% of IS for validation
                save_study=request.save_optimization_studies if hasattr(request, "save_optimization_studies") else False,
                show_progress_bar=True,
                param_ranges=self._convert_param_ranges(request.param_ranges),
            )

            self._optimizer_cache[cache_key] = WalkForwardOptimizer(backtest_service=self.backtest_service, config=config, user_id=self.user_id)

        return self._optimizer_cache[cache_key]

    def _convert_param_ranges(self, param_ranges: Dict[str, OptParamRange]) -> Dict[str, Dict[str, Any]]:
        """Convert WFA param ranges to optimizer format"""
        converted = {}
        for name, pr in param_ranges.items():
            converted[name] = {
                "type": pr.type,
                "low": pr.min,
                "high": pr.max,
                "step": pr.step if pr.type == "int" else None,
                "log": getattr(pr, "log", False),
            }
        return converted

    def _generate_folds(self, data: pd.DataFrame, request: WFARequest) -> List[Dict[str, Any]]:
        """Generate IS/OOS date folds based on request parameters."""
        folds = []
        df_index = data.index

        # Validate window sizes
        if len(df_index) < request.is_window_days + request.oos_window_days:
            raise ValueError(
                f"Data period ({len(df_index)} days) insufficient for " f"IS ({request.is_window_days}) + OOS ({request.oos_window_days}) windows"
            )

        # Start from the beginning of data
        current_oos_start_idx = request.is_window_days
        fold_idx = 0
        max_folds = getattr(request, "max_folds", 100)  # Prevent infinite loops

        while current_oos_start_idx + request.oos_window_days <= len(df_index) and fold_idx < max_folds:
            # Calculate IS window
            if request.anchored:
                is_start_idx = 0
            else:
                is_start_idx = current_oos_start_idx - request.is_window_days

            is_end_idx = current_oos_start_idx

            # Calculate OOS window
            oos_start_idx = current_oos_start_idx
            oos_end_idx = current_oos_start_idx + request.oos_window_days

            # Ensure indices are valid
            if is_start_idx >= 0 and is_end_idx <= len(df_index) and oos_end_idx <= len(df_index):
                folds.append(
                    {
                        "fold_index": fold_idx,
                        "is_start": df_index[is_start_idx],
                        "is_end": df_index[is_end_idx - 1],
                        "oos_start": df_index[oos_start_idx],
                        "oos_end": df_index[oos_end_idx - 1],
                    }
                )

                # Walk forward
                current_oos_start_idx += request.step_days
                fold_idx += 1
            else:
                break

        return folds

    async def run_wfa(self, request: WFARequest) -> WFAResponse:
        """
        Run the complete Walk-Forward Analysis using modular optimization.

        Args:
            request: WFA request with strategy and parameters

        Returns:
            WFAResponse with fold results and aggregated metrics
        """
        logger.info(f"Starting WFA for {request.symbol} with strategy {request.strategy_key}")

        try:
            # 1. Fetch full data
            data = await fetch_stock_data(request.symbol, request.period, request.interval)

            if data.empty:
                raise ValueError(f"No data available for {request.symbol}")

            logger.info(f"Fetched {len(data)} rows of data for {request.symbol}")

            # 2. Generate folds
            folds_info = self._generate_folds(data, request)
            if not folds_info:
                raise ValueError("Data period insufficient for the requested IS/OOS window sizes")

            logger.info(f"Generated {len(folds_info)} folds for WFA")

            # 3. Initialize aggregators
            fold_results = []
            aggregated_oos_trades = []
            aggregated_oos_equity = []
            last_equity = request.initial_capital

            total_is_performance = 0
            total_oos_performance = 0
            successful_folds = 0

            # Create optimizer
            optimizer = self._create_optimizer(request)

            # 4. Iterate through folds
            for fold_idx, fold in enumerate(folds_info):
                logger.info(
                    f"Processing WFA Fold {fold['fold_index']}: "
                    f"IS {fold['is_start'].date()} to {fold['is_end'].date()} | "
                    f"OOS {fold['oos_start'].date()} to {fold['oos_end'].date()}"
                )

                # Check capital adequacy
                if last_equity < request.initial_capital * 0.1:  # 10% of initial
                    logger.warning(f"WFA: Account drawdown too severe (equity {last_equity:.2f}). " f"Stopping at fold {fold['fold_index']}")
                    break

                try:
                    # A. Run optimization on IS window
                    optimization_result = await self._optimize_fold(
                        optimizer=optimizer,
                        request=request,
                        start=fold["is_start"],
                        end=fold["is_end"],
                        capital=last_equity,
                        fold_index=fold["fold_index"],
                    )

                    if not optimization_result:
                        logger.warning(f"Fold {fold['fold_index']} optimization failed, skipping")
                        continue

                    best_params = optimization_result.best_params
                    is_metrics = optimization_result.best_metrics

                    # Log optimization insights
                    logger.info(f"Fold {fold['fold_index']} IS {request.metric}: " f"{optimization_result.best_value:.4f}")

                    if optimization_result.overfitting_ratio:
                        logger.info(f"Overfitting ratio: {optimization_result.overfitting_ratio:.2f}")

                    # B. Validate on OOS window
                    oos_metrics, oos_trades, oos_equity = await self._run_out_of_sample(
                        request=request, start=fold["oos_start"], end=fold["oos_end"], params=best_params, capital=last_equity
                    )

                    # C. Update state for next fold
                    if oos_equity:
                        last_equity = oos_equity[-1]["equity"]

                    # D. Track metrics for WFE calculation
                    is_ret = getattr(is_metrics, "total_return_pct", 0)
                    oos_ret = getattr(oos_metrics, "total_return_pct", 0)

                    total_is_performance += is_ret
                    total_oos_performance += oos_ret
                    successful_folds += 1

                    # E. Store fold results with optimization metadata
                    fold_results.append(
                        WFAFoldResult(
                            fold_index=fold["fold_index"],
                            is_start=str(fold["is_start"]),
                            is_end=str(fold["is_end"]),
                            oos_start=str(fold["oos_start"]),
                            oos_end=str(fold["oos_end"]),
                            best_params=best_params,
                            is_metrics=is_metrics,
                            oos_metrics=oos_metrics,
                            optimization_metadata={
                                "n_trials": len(optimization_result.study.trials),
                                "best_value": optimization_result.best_value,
                                "param_importances": optimization_result.param_importances,
                                "overfitting_ratio": optimization_result.overfitting_ratio,
                            },
                        )
                    )

                    # F. Aggregate OOS results
                    aggregated_oos_trades.extend(oos_trades)
                    aggregated_oos_equity.extend(oos_equity)

                except Exception as e:
                    logger.error(f"Fold {fold['fold_index']} failed: {e}", exc_info=True)
                    continue

            # 5. Calculate final aggregated metrics
            if not aggregated_oos_trades:
                logger.warning("No successful folds completed")
                return self._create_empty_response(request)

            final_oos_metrics_dict = calculate_performance_metrics(aggregated_oos_trades, aggregated_oos_equity, request.initial_capital)

            # 6. Calculate Walk-Forward Efficiency (WFE)
            wfe = self._calculate_wfe(total_is_performance, total_oos_performance, successful_folds)

            # 7. Calculate robustness metrics
            robustness_metrics = self._calculate_robustness_metrics(fold_results)

            logger.info(f"WFA completed successfully with {successful_folds}/{len(folds_info)} folds")
            logger.info(f"WFE: {wfe:.3f}, Robustness: {robustness_metrics['avg_wfe']:.3f}")

            # 8. Build and return response
            return WFAResponse(
                folds=fold_results,
                aggregated_oos_metrics=BacktestResult(**final_oos_metrics_dict),
                oos_equity_curve=[EquityCurvePoint(timestamp=str(p["timestamp"]), equity=p["equity"], cash=p["cash"]) for p in aggregated_oos_equity],
                wfe=wfe,
                robustness_metrics=robustness_metrics,
                strategy_key=request.strategy_key,
                symbol=request.symbol,
                successful_folds=successful_folds,
                total_folds=len(folds_info),
            )

        except Exception as e:
            logger.error(f"WFA failed: {e}", exc_info=True)
            raise

    async def _optimize_fold(
        self, optimizer: WalkForwardOptimizer, request: WFARequest, start: datetime, end: datetime, capital: float, fold_index: int
    ) -> Optional[OptimizationResult]:
        """Optimize parameters for a single fold"""
        try:
            # Create optimization request
            opt_request = self._create_optimization_request(request, capital)

            # Run optimization WITHOUT internal validation split.
            # WFA already has dedicated OOS windows for validation —
            # an additional 80/20 split inside 'IS' is redundant and
            # shrinks the effective trading window, causing slow
            # strategies (e.g. SMA Crossover) to produce zero trades.
            result = await optimizer.optimize_in_sample(
                request=opt_request, start=start, end=end, capital=capital, validation_split=False, fold_index=fold_index
            )

            return result

        except Exception as e:
            logger.error(f"Fold {fold_index} optimization failed: {e}")
            return None

    def _create_optimization_request(self, request: WFARequest, capital: float) -> OptimizationRequest:
        """Create an optimization request from WFA request"""

        return OptimizationRequest(
            symbol=request.symbol,
            strategy_key=request.strategy_key,
            interval=request.interval,
            metric=request.metric,
            initial_capital=capital,
            commission_rate=request.commission_rate,
            slippage_rate=request.slippage_rate,
            indicator_config={"returns": True, "volatility": True, "moving_averages": True, "rsi": True, "macd": True, "bollinger_bands": False},
        )

    async def _run_out_of_sample(
        self, request: WFARequest, start: datetime, end: datetime, params: Dict[str, Any], capital: float
    ) -> Tuple[BacktestResult, List, List]:
        """Run backtest on OOS segment with robust error handling."""
        try:
            # Fetch data (ensure we have enough for warm-up)
            data = await fetch_stock_data(request.symbol, "max", request.interval)

            # Find data segment with warm-up (e.g., 200 bars before start)
            try:
                # Find index of start date
                # Use asof search if exact date not found
                target_idx = data.index.get_indexer([pd.Timestamp(start)], method="bfill")[0]
                warm_up_idx = max(0, target_idx - 250)  # Approx 1 year

                # Slice from warm-up start to requested end
                oos_segment = data.iloc[warm_up_idx:].loc[:end]
            except Exception as e:
                logger.warning(f"Error slicing with warm-up: {e}. Using raw slice.")
                oos_segment = data.loc[:end]

            if oos_segment.empty:
                logger.warning(f"No OOS data available for range {start} to {end}")
                empty_result = BacktestResult(
                    total_return=0,
                    total_return_pct=0,
                    win_rate=0,
                    sharpe_ratio=0,
                    max_drawdown=0,
                    total_trades=0,
                    winning_trades=0,
                    losing_trades=0,
                    avg_profit=0,
                    avg_win=0,
                    avg_loss=0,
                    profit_factor=0,
                    final_equity=capital,
                    initial_capital=capital,
                )
                return empty_result, [], [{"timestamp": start, "equity": capital, "cash": capital}]

            return await self._run_test(request, oos_segment, params, capital, start_timestamp=start)

        except Exception as e:
            logger.error(f"OOS run failed: {e}")
            raise

    async def _run_test(
        self, request: WFARequest, data: pd.DataFrame, params: Dict[str, Any], capital: float, start_timestamp: Any = None
    ) -> Tuple[BacktestResult, List, List]:
        """Helper to run a single backtest on a data slice."""
        from ..core.risk_manager import RiskManager
        from ..core.trading_engine import TradingEngine
        from ..strategies.strategy_catalog import get_catalog

        # Ensure window parameters are integers
        sanitized_params = self._sanitize_params(params)

        logger.debug(f"Running test with params: {sanitized_params}")

        try:
            strategy = get_catalog().create_strategy(request.strategy_key, **sanitized_params)
            engine = TradingEngine(
                strategy=strategy,
                initial_capital=capital,
                risk_manager=RiskManager(),
                trading_service=self.backtest_service.trading_service,
                commission_rate=request.commission_rate,
                slippage_rate=request.slippage_rate,
                db=self.db,
            )

            await engine.run_backtest(request.symbol, data, start_timestamp=start_timestamp)

            if not engine.trades:
                logger.debug("No trades generated in this period")

            metrics_dict = calculate_performance_metrics(engine.trades, engine.equity_curve, capital)

            return BacktestResult(**metrics_dict), engine.trades, engine.equity_curve

        except Exception as e:
            logger.error(f"Error during WFA test run: {e}")
            raise

    def _sanitize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert parameters to appropriate types."""
        sanitized = {}
        for k, v in params.items():
            if "window" in k.lower() or "period" in k.lower() or "lookback" in k.lower():
                try:
                    sanitized[k] = int(float(v))  # Handle both int and float strings
                except (ValueError, TypeError):
                    sanitized[k] = v
            else:
                sanitized[k] = v
        return sanitized

    def _calculate_wfe(self, total_is: float, total_oos: float, num_folds: int) -> float:
        """Calculate Walk-Forward Efficiency"""
        if num_folds == 0 or total_is == 0:
            return 0.0

        avg_is = total_is / num_folds
        avg_oos = total_oos / num_folds

        # WFE = OOS performance / IS performance
        # Values > 0.5 indicate robustness
        return avg_oos / avg_is if avg_is != 0 else 0.0

    def _calculate_robustness_metrics(self, fold_results: List[WFAFoldResult]) -> Dict[str, float]:
        """Calculate robustness metrics across folds."""
        if not fold_results:
            return {}

        wfe_scores = []
        is_sharpes = []
        oos_sharpes = []

        for fold in fold_results:
            is_ret = getattr(fold.is_metrics, "total_return_pct", 0)
            oos_ret = getattr(fold.oos_metrics, "total_return_pct", 0)

            if is_ret != 0:
                wfe_scores.append(oos_ret / is_ret)

            is_sharpes.append(getattr(fold.is_metrics, "sharpe_ratio", 0))
            oos_sharpes.append(getattr(fold.oos_metrics, "sharpe_ratio", 0))

        return {
            "avg_wfe": np.mean(wfe_scores) if wfe_scores else 0,
            "std_wfe": np.std(wfe_scores) if wfe_scores else 0,
            "avg_is_sharpe": np.mean(is_sharpes),
            "avg_oos_sharpe": np.mean(oos_sharpes),
            "sharpe_decay": (np.mean(is_sharpes) - np.mean(oos_sharpes)) / np.mean(is_sharpes) if np.mean(is_sharpes) != 0 else 0,
            "positive_wfe_ratio": sum(1 for w in wfe_scores if w > 0.5) / len(wfe_scores) if wfe_scores else 0,
        }

    def _create_empty_response(self, request: WFARequest) -> WFAResponse:
        """Create an empty response when no folds succeed."""
        empty_metrics = BacktestResult(
            total_return=0,
            total_return_pct=0,
            win_rate=0,
            sharpe_ratio=0,
            max_drawdown=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            avg_profit=0,
            avg_win=0,
            avg_loss=0,
            profit_factor=0,
            final_equity=request.initial_capital,
            initial_capital=request.initial_capital,
        )

        return WFAResponse(
            folds=[],
            aggregated_oos_metrics=empty_metrics,
            oos_equity_curve=[EquityCurvePoint(timestamp=datetime.now().isoformat(), equity=request.initial_capital, cash=request.initial_capital)],
            wfe=0.0,
            robustness_metrics={},
            strategy_key=request.strategy_key,
            symbol=request.symbol,
            successful_folds=0,
            total_folds=0,
        )
