import asyncio
import logging
import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import HyperbandPruner, MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import CmaEsSampler, RandomSampler, TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances

from backend.app.schemas.backtest import BacktestResult
from backend.app.strategies.technical.pre_compute_indicators import PreComputeIndicators

logger = logging.getLogger(__name__)


class OptimizationMetric(Enum):
    """Supported optimization metrics"""

    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    TOTAL_RETURN = "total_return"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    MAX_DRAWDOWN = "max_drawdown"  # Note: this would be minimized
    VAR_95 = "var_95"  # Note: this would be minimized


class OptimizationDirection(Enum):
    """Optimization direction"""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class SamplerType(Enum):
    """Available Optuna samplers"""

    TPE = "tpe"
    RANDOM = "random"
    CMA_ES = "cma_es"
    GRID = "grid"


class PrunerType(Enum):
    """Available Optuna pruners"""

    MEDIAN = "median"
    HYPERBAND = "hyperband"
    SUCCESSIVE_HALVING = "successive_halving"
    NONE = "none"


@dataclass
class OptimizationConfig:
    """Configuration for optimization"""

    n_trials: int = 100
    n_jobs: int = 1
    timeout_seconds: Optional[int] = None
    random_seed: int = 42
    sampler_type: SamplerType = SamplerType.TPE
    pruner_type: PrunerType = PrunerType.MEDIAN
    early_stopping_rounds: Optional[int] = None
    cv_folds: int = 1  # For cross-validation within in-sample
    test_size: float = 0.2  # For validation split within in-sample
    save_study: bool = False
    study_directory: str = "optuna_studies"
    show_progress_bar: bool = True
    param_ranges: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def get_direction(self, metric: str) -> str:
        """Determine optimization direction based on metric"""
        if metric in [OptimizationMetric.MAX_DRAWDOWN.value, OptimizationMetric.VAR_95.value]:
            return OptimizationDirection.MINIMIZE.value
        return OptimizationDirection.MAXIMIZE.value


@dataclass
class OptimizationResult:
    """Container for optimization results"""

    best_params: Dict[str, Any]
    best_metrics: Any
    best_value: float
    study: optuna.Study
    optimization_history: pd.DataFrame
    param_importances: Optional[Dict[str, float]] = None
    validation_metrics: Optional[Dict[str, float]] = None
    overfitting_ratio: Optional[float] = None  # IS performance / OOS performance


class ParameterSampler:
    """Handles parameter sampling strategies"""

    @staticmethod
    def sample_from_trial(trial: optuna.Trial, param_config: Dict[str, Any]) -> Any:
        """Sample a parameter value based on its configuration"""
        param_type = param_config.get("type", "float")

        if param_type == "float":
            return trial.suggest_float(
                param_config.get("name", "param"),
                param_config["low"],
                param_config["high"],
                log=param_config.get("log", False),
                step=param_config.get("step", None),
            )
        elif param_type == "int":
            return trial.suggest_int(
                param_config.get("name", "param"),
                param_config["low"],
                param_config["high"],
                log=param_config.get("log", False),
                step=param_config.get("step", 1),
            )
        elif param_type == "categorical":
            return trial.suggest_categorical(param_config.get("name", "param"), param_config["choices"])
        elif param_type == "discrete_uniform":
            return trial.suggest_discrete_uniform(param_config.get("name", "param"), param_config["low"], param_config["high"], param_config["q"])
        else:
            raise ValueError(f"Unsupported parameter type: {param_type}")


class WalkForwardOptimizer:
    """
    Production-grade walk-forward optimization system
    """

    def __init__(self, backtest_service, config: Optional[OptimizationConfig] = None, user_id: str = None):
        """
        Initialize the optimizer

        Args:
            backtest_service: Service for running backtests
            config: Optimization configuration
        """
        self.user_id = user_id
        self.backtest_service = backtest_service
        self.config = config or OptimizationConfig()
        self.data_provider = PreComputeIndicators()
        self.parameter_sampler = ParameterSampler()
        self.logger = logging.getLogger(__name__)

    async def optimize_in_sample(
        self, request: Any, start: Union[str, datetime], end: Union[str, datetime], capital: float, validation_split: bool = True, fold_index: int = 0
    ) -> OptimizationResult:
        """
        Run optimization on in-sample data

        Args:
            request: Optimization request with strategy config
            start: Start date for in-sample period
            end: End date for in-sample period
            capital: Initial capital
            validation_split: Whether to hold out validation data

        Returns:
            OptimizationResult with best parameters and metadata
        """
        try:
            data = await self.data_provider.get_data(request.symbol, request.interval, start, end)

            data = self.data_provider.precalculate_indicators(data, getattr(request, "indicator_config", {}))

            if validation_split and self.config.cv_folds == 1:
                train_data, val_data = self._train_test_split(data)
            else:
                train_data = data
                val_data = None

            direction = self.config.get_direction(request.metric)

            # Create Optuna study — vary seed per fold so each fold
            # explores different regions of the parameter space.
            study = self._create_study(request.metric, direction, fold_index=fold_index)

            # The actual IS start date — trading should only begin here,
            # even though data extends earlier for indicator warm-up.
            is_start_str = str(start) if isinstance(start, datetime) else start

            async def objective(trial: optuna.Trial) -> float:
                return await self._objective_wrapper(trial, request, train_data, capital, val_data, is_start=is_start_str)

            study = await self._run_optimization(study, objective)

            # Get best parameters and re-run for full metrics
            start_str = str(start) if isinstance(start, datetime) else start
            end_str = str(end) if isinstance(end, datetime) else end

            best_metrics, _, _ = await self._run_with_params(request, data, study.best_params, capital, start_date=start_str, end_date=end_str)

            # Calculate validation metrics if validation data exists
            validation_metrics = None
            if val_data is not None:
                val_start = val_data.index.min()
                val_end = val_data.index.max()
                val_metrics, _, _ = await self._run_with_params(
                    request, val_data, study.best_params, capital, start_date=str(val_start), end_date=str(val_end)
                )
                validation_metrics = {"value": getattr(val_metrics, request.metric, 0), "full_metrics": val_metrics}

                # Calculate overfitting ratio
                train_value = study.best_value
                val_value = validation_metrics["value"]
                if val_value != 0:
                    overfitting_ratio = train_value / abs(val_value)
                else:
                    overfitting_ratio = float("inf")
            else:
                overfitting_ratio = None

            # Create optimization history dataframe
            history = self._create_history_dataframe(study)

            # Calculate parameter importances
            param_importances = self._calculate_param_importances(study)

            # Build result
            result = OptimizationResult(
                best_params=study.best_params,
                best_metrics=best_metrics,
                best_value=study.best_value,
                study=study,
                optimization_history=history,
                param_importances=param_importances,
                validation_metrics=validation_metrics,
                overfitting_ratio=overfitting_ratio,
            )

            # Save study if configured
            if self.config.save_study:
                self._save_study(study, request, start, end)

            self.logger.info(f"Optimization complete: best {request.metric} = {study.best_value:.4f}")
            return result

        except Exception as e:
            self.logger.error(f"Optimization failed: {e}", exc_info=True)
            raise

    async def _run_with_params(
        self,
        request: Any,
        data: pd.DataFrame,
        params: Dict[str, Any],
        capital: float,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Tuple:
        """Run backtest with given parameters using BacktestService"""
        from backend.app.optimise.request_adapter import RequestAdapter

        try:
            test_request = RequestAdapter.to_backtest_request(request, params, capital, start_date=start_date, end_date=end_date)

            response = await self.backtest_service.run_single_backtest(request=test_request, user_id=self.user_id)

            metrics = response.result
            trades = response.trades
            equity_curve = response.equity_curve

            equity_dicts = [
                {"timestamp": point.timestamp, "equity": point.equity, "cash": point.cash, "drawdown": getattr(point, "drawdown", 0)}
                for point in equity_curve
            ]

            return metrics, trades, equity_dicts

        except Exception as e:
            logger.error(f"Error in _run_with_params: {e}", exc_info=True)
            # Return empty results on failure
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
                final_equity=capital,
                initial_capital=capital,
            )
            return empty_metrics, [], [{"timestamp": datetime.now(), "equity": capital, "cash": capital}]

    def _create_study(self, metric: str, direction: str, fold_index: int = 0) -> optuna.Study:
        """Create Optuna study with configured sampler and pruner"""
        # Vary seed per fold so each fold explores independently
        fold_seed = self.config.random_seed + fold_index

        # Create sampler
        if self.config.sampler_type == SamplerType.TPE:
            sampler = TPESampler(
                seed=fold_seed,
                # Keep startup low so TPE kicks in quickly; at least 2 random
                # trials are needed before TPE can model, but we cap at 30%
                # of n_trials so the Bayesian phase gets the majority.
                n_startup_trials=max(2, min(5, self.config.n_trials // 3)),
            )
        elif self.config.sampler_type == SamplerType.RANDOM:
            sampler = RandomSampler(seed=fold_seed)
        elif self.config.sampler_type == SamplerType.CMA_ES:
            sampler = CmaEsSampler(seed=fold_seed)
        else:
            sampler = None  # Optuna default

        # Create pruner
        if self.config.pruner_type == PrunerType.MEDIAN:
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=5)
        elif self.config.pruner_type == PrunerType.HYPERBAND:
            pruner = HyperbandPruner(min_resource=1, max_resource=self.config.n_trials, reduction_factor=3)
        elif self.config.pruner_type == PrunerType.SUCCESSIVE_HALVING:
            pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0)
        else:
            pruner = None

        return optuna.create_study(
            direction=direction, sampler=sampler, pruner=pruner, study_name=f"optimize_{metric}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    async def _objective_wrapper(
        self,
        trial: optuna.Trial,
        request: Any,
        data: pd.DataFrame,
        capital: float,
        val_data: Optional[pd.DataFrame] = None,
        is_start: Optional[str] = None,
    ) -> float:
        """Wrapper for Optuna objective with error handling and pruning"""
        try:
            # Sample parameters
            params = {}
            for param_name, param_range in self.config.param_ranges.items():
                param_range["name"] = param_name
                params[param_name] = self.parameter_sampler.sample_from_trial(trial, param_range)

            # Run backtest — use actual IS start (not warm-up start) so that
            # only trades within the IS window count towards the metric.
            start_date = is_start if is_start else str(data.index.min())
            end_date = str(data.index.max())
            metrics, _, _ = await self._run_with_params(request, data, params, capital, start_date=start_date, end_date=end_date)

            # Extract metric value
            metric_value = getattr(metrics, request.metric, None)

            if metric_value is None:
                self.logger.warning(f"Metric {request.metric} not found")
                return float("-inf") if self.config.get_direction(request.metric) == "maximize" else float("inf")

            # Handle NaN/inf
            if pd.isna(metric_value) or np.isinf(metric_value):
                return float("-inf") if self.config.get_direction(request.metric) == "maximize" else float("inf")

            # Report intermediate values for pruning (if available)
            if hasattr(metrics, "intermediate_values"):
                for step, value in metrics.intermediate_values.items():
                    trial.report(value, step)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

            return float(metric_value)

        except optuna.TrialPruned:
            raise
        except Exception as e:
            self.logger.error(f"Trial failed: {e}")
            return float("-inf") if self.config.get_direction(request.metric) == "maximize" else float("inf")

    async def _run_optimization(self, study: optuna.Study, objective_func) -> optuna.Study:
        """Run optimization bridging sync Optuna with async objective.

        Optuna's study.optimize() is synchronous, but our objective function
        is async.  We run the synchronous optimize() in a worker thread and
        use asyncio.run_coroutine_threadsafe() to call the async objective
        back on the main event loop from that thread.
        """
        loop = asyncio.get_running_loop()

        def sync_objective(trial: optuna.Trial) -> float:
            """Bridge: called by Optuna in a worker thread, dispatches to async."""
            future = asyncio.run_coroutine_threadsafe(objective_func(trial), loop)
            return future.result()  # blocks the worker thread until coroutine completes

        def run_study():
            study.optimize(
                sync_objective,
                n_trials=self.config.n_trials,
                timeout=self.config.timeout_seconds,
                n_jobs=1,  # sequential within the thread
                show_progress_bar=self.config.show_progress_bar,
            )

        await loop.run_in_executor(None, run_study)

        return study

    def _train_test_split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and validation sets"""
        split_idx = int(len(data) * (1 - self.config.test_size))
        train_data = data.iloc[:split_idx].copy()
        val_data = data.iloc[split_idx:].copy()
        return train_data, val_data

    def _create_history_dataframe(self, study: optuna.Study) -> pd.DataFrame:
        """Create DataFrame with optimization history"""
        trials = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                trials.append(
                    {
                        "number": trial.number,
                        "value": trial.value,
                        "params": trial.params,
                        "datetime_start": trial.datetime_start,
                        "datetime_complete": trial.datetime_complete,
                        "duration": (trial.datetime_complete - trial.datetime_start).total_seconds() if trial.datetime_complete else None,
                    }
                )

        return pd.DataFrame(trials)

    def _calculate_param_importances(self, study: optuna.Study) -> Optional[Dict[str, float]]:
        """Calculate parameter importance using Optuna's built-in function"""
        try:
            from optuna.importance import get_param_importances

            importances = get_param_importances(study)
            return {k: float(v) for k, v in importances.items()}
        except Exception as e:
            self.logger.warning(f"Could not calculate param importances: {e}")
            return None

    def _save_study(self, study: optuna.Study, request: Any, start: Union[str, datetime], end: Union[str, datetime]):
        """Save Optuna study to disk"""
        os.makedirs(self.config.study_directory, exist_ok=True)

        filename = f"{request.symbol}_{start}_{end}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        filepath = os.path.join(self.config.study_directory, filename)

        with open(filepath, "wb") as f:
            pickle.dump(study, f)

        self.logger.info(f"Study saved to {filepath}")

    def get_visualizations(self, study: optuna.Study) -> Dict[str, Any]:
        """Generate optimization visualizations"""
        try:
            import plotly.graph_objects as go

            visualizations = {}

            # Optimization history
            fig_history = plot_optimization_history(study)
            visualizations["history"] = fig_history.to_json()

            # Parameter importances
            if self._calculate_param_importances(study):
                fig_importance = plot_param_importances(study)
                visualizations["importance"] = fig_importance.to_json()

            # Parallel coordinate plot
            df = self._create_history_dataframe(study)
            if not df.empty and len(df.columns) > 2:
                fig_parallel = go.Figure(
                    data=go.Parcoords(
                        line=dict(color=df["value"], colorscale="viridis", showscale=True),
                        dimensions=[
                            dict(label="Trial", values=df["number"]),
                            dict(label="Value", values=df["value"]),
                            *[dict(label=k, values=df["params"].apply(lambda x: x.get(k, 0))) for k in df["params"].iloc[0].keys()],
                        ],
                    )
                )
                visualizations["parallel"] = fig_parallel.to_json()

            return visualizations

        except Exception as e:
            self.logger.warning(f"Could not generate visualizations: {e}")
            return {}
