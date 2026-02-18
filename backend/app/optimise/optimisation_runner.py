"""
Bayesian Optimization Endpoint:
Connection management, error handling, and performance
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

import optuna
from fastapi import APIRouter
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from backend.app.config import settings
from backend.app.schemas.backtest import BacktestRequest as SingleBacktestRequest, MultiAssetBacktestRequest, StrategyConfig
from backend.app.schemas.optimise import BayesianOptimizationRequest
from backend.app.services.backtest_service import BacktestService

logger = logging.getLogger(__name__)
router = APIRouter()


class OptimizationRunner:
    """
    Encapsulates optimization logic with proper resource management
    """

    def __init__(self, request: BayesianOptimizationRequest, user_id: int, max_workers: int = 4):
        self.request = request
        self.user_id = user_id
        self.max_workers = max_workers

        # Create single engine for all trials
        self.engine = None
        self.executor = None

        # Track optimization state
        self.completed_trials = 0
        self.failed_trials = 0

    async def __aenter__(self):
        """Async context manager entry"""
        # Create engine once
        self.engine = create_async_engine(
            settings.DATABASE_URL,
            echo=False,
            poolclass=NullPool,  # No connection pooling for optimization
            pool_pre_ping=True,  # Verify connections before use
        )

        # Create thread pool executor
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        logger.info(f"Optimization initialized: {self.max_workers} workers")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup resources"""
        # Shutdown executor
        if self.executor:
            self.executor.shutdown(wait=True)

        # Dispose engine
        if self.engine:
            await self.engine.dispose()

        logger.info(f"Optimization cleanup: " f"{self.completed_trials} completed, " f"{self.failed_trials} failed")

    def create_objective(self):
        """Create objective function with proper error handling"""

        def objective(trial: optuna.Trial) -> float:
            """
            Objective function that runs in thread pool
            Returns metric value to maximize
            """
            try:
                # 1. Suggest parameters based on ranges
                params = self._suggest_parameters(trial)

                # 2. Run backtest in async context
                result_value = asyncio.run(self._run_trial_backtest(params))

                # 3. Track success
                self.completed_trials += 1

                # 4. Log progress
                if self.completed_trials % 5 == 0:
                    logger.info(f"Progress: {self.completed_trials}/{self.request.n_trials} trials, " f"best={trial.study.best_value:.4f}")

                return result_value

            except Exception as e:
                self.failed_trials += 1
                logger.error(f"Trial {trial.number} failed: {str(e)[:200]}", exc_info=True)
                # Return penalty value so optimization continues
                return float("-inf")

        return objective

    def _suggest_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest parameters based on defined ranges"""
        params = {}

        for name, param_range in self.request.param_ranges.items():
            try:
                if param_range.type == "int":
                    params[name] = trial.suggest_int(
                        name, int(param_range.min), int(param_range.max), step=int(param_range.step) if param_range.step else 1
                    )
                else:  # float
                    params[name] = trial.suggest_float(name, param_range.min, param_range.max, step=param_range.step)
            except Exception as e:
                logger.error(f"Error suggesting parameter {name}: {e}")
                raise

        return params

    async def _run_trial_backtest(self, params: Dict[str, Any]) -> float:
        """
        Run a single backtest trial
        Returns the metric value
        """
        try:
            # Create session from shared engine
            async with AsyncSession(self.engine) as session:
                service = BacktestService(session)

                # Single-asset backtest
                if len(self.request.tickers) == 1:
                    backtest_req = SingleBacktestRequest(
                        symbol=self.request.tickers[0],
                        strategy_key=self.request.strategy_key,
                        parameters=params,
                        period=self.request.period,
                        interval=self.request.interval,
                        initial_capital=self.request.initial_capital,
                    )

                    result = await service.run_single_backtest(backtest_req, self.user_id)

                    # Extract metric
                    metric_value = getattr(result.result, self.request.metric, 0)

                else:
                    # Multi-asset backtest
                    strategy_configs = {
                        ticker: StrategyConfig(strategy_key=self.request.strategy_key, parameters=params) for ticker in self.request.tickers
                    }

                    multi_req = MultiAssetBacktestRequest(
                        symbols=self.request.tickers,
                        strategy_configs=strategy_configs,
                        allocation_method="equal",
                        period=self.request.period,
                        interval=self.request.interval,
                        initial_capital=self.request.initial_capital,
                    )

                    result = await service.run_multi_asset_backtest(multi_req, self.user_id)

                    metric_value = getattr(result.result, self.request.metric, 0)

                # Handle None or invalid values
                if metric_value is None:
                    logger.warning(f"Metric {self.request.metric} returned None")
                    return float("-inf")

                return float(metric_value)

        except Exception as e:
            logger.error(f"Backtest execution failed: {e}")
            raise

    async def run_optimization(self) -> optuna.Study:
        """
        Run the optimization with proper configuration
        """
        # Create study with advanced settings
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=min(10, self.request.n_trials // 10),
                multivariate=True,
                seed=42,  # For reproducibility
            ),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )

        # Create objective function
        objective = self.create_objective()

        # Run optimization in thread pool
        logger.info(f"Starting optimization: " f"{self.request.n_trials} trials, " f"metric={self.request.metric}")

        try:
            # Run optimization (blocks until complete)
            await asyncio.to_thread(
                study.optimize,
                objective,
                n_trials=self.request.n_trials,
                n_jobs=1,  # Important: 1 job with our thread pool
                show_progress_bar=False,
                callbacks=[self._create_callback()],
            )

            logger.info(f"Optimization complete: " f"best_value={study.best_value:.4f}, " f"best_params={study.best_params}")

            return study

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise

    def _create_callback(self):
        """Create callback for logging progress"""

        def callback(study: optuna.Study, trial: optuna.Trial):
            if trial.value is not None and trial.value != float("-inf"):
                logger.info(f"Trial {trial.number}: " f"value={trial.value:.4f}, " f"params={trial.params}")

        return callback
