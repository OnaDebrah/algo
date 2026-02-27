"""
Scheduler for automated crash monitoring and hedge execution
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from alerts import AlertManager
from alpaca_trade_api.entity import Watchlist
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from models import User, UserSettings
from models.bubble_detection import BubbleDetection
from models.crash_prediction import CrashPrediction
from pygments.lexers import go
from schemas.alert import AlertCategory, AlertLevel
from sklearn.metrics import r2_score
from sqlalchemy import select
from tensorflow.python.keras.losses import mean_squared_error

from ...core.data.providers.providers import ProviderFactory
from ...database import AsyncSessionLocal
from ...services.brokers.broker_service import BrokerService
from ...services.execution.auto_hedge_executor import AutoHedgeExecutor
from ..analysis.dashboard import CrashPredictionDashboard
from ..portfolio_service import PortfolioService

logger = logging.getLogger(__name__)


class CrashMonitorScheduler:
    """
    Scheduler for automated crash monitoring and hedge execution
    """

    def __init__(self, alert_manager: AlertManager):
        self.scheduler = AsyncIOScheduler()
        self.provider_factory = ProviderFactory()
        self.broker = BrokerService()
        self.executors = {}
        self.dashboard = CrashPredictionDashboard()
        self.alert_manager = alert_manager

    async def start(self):
        """Start all scheduled tasks"""

        # Market data update - every hour during market hours
        self.scheduler.add_job(self.update_market_data, CronTrigger(day_of_week="mon-fri", hour="9-16", minute="0"), id="market_data_update")

        # ML model retraining - weekly on Sunday
        self.scheduler.add_job(self.retrain_models, CronTrigger(day_of_week="sun", hour="20", minute="0"), id="model_retraining")

        # Crash prediction - every 30 minutes during market hours
        self.scheduler.add_job(self.run_crash_predictions, CronTrigger(day_of_week="mon-fri", hour="9-16", minute="*/30"), id="crash_predictions")

        # Hedge execution - check every hour
        self.scheduler.add_job(self.check_hedge_executions, IntervalTrigger(hours=1), id="hedge_execution")

        # Dashboard update - every 5 minutes
        self.scheduler.add_job(self.update_dashboards, IntervalTrigger(minutes=5), id="dashboard_update")

        # Alert check - continuous
        self.scheduler.add_job(self.check_alerts, IntervalTrigger(minutes=1), id="alert_check")

        self.scheduler.start()
        logger.info("Crash monitor scheduler started")

    async def update_market_data(self):
        """Update market data for all tracked symbols"""
        logger.info("Updating market data")

        async with AsyncSessionLocal() as db:
            users = await self._get_active_users(db)

            for user in users:
                try:
                    await self._update_user_portfolio(user.id, db)

                    symbols = await self._get_user_symbols(user.id, db)
                    for symbol in symbols:
                        await self.provider_factory.get_quote(symbol, user, db)

                except Exception as e:
                    logger.error(f"Error updating data for user {user.id}: {e}")

    async def retrain_models(self):
        """Retrain ML models with latest data"""
        logger.info("=" * 50)
        logger.info("Starting ML model retraining")
        logger.info("=" * 50)

        retrain_results = {"timestamp": datetime.now().isoformat(), "models": {}, "success": False, "errors": []}

        try:
            logger.info("Step 1: Fetching latest market data for training...")
            training_data = await self._fetch_training_data()

            if training_data.empty:
                logger.error("No training data available")
                retrain_results["errors"].append("No training data available")
                return retrain_results

            logger.info(f"Fetched {len(training_data)} samples for training")

            logger.info("Step 2: Preparing features and targets...")
            X_train, X_test, y_train, y_test = await self._prepare_training_data(training_data)

            logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

            logger.info("Step 3: Retraining Enhanced Random Forest...")
            rf_result = await self._retrain_random_forest(X_train, X_test, y_train, y_test)
            retrain_results["models"]["random_forest"] = rf_result

            logger.info("Step 4: Retraining LSTM Stress Predictor...")
            lstm_result = await self._retrain_lstm(X_train, X_test, y_train, y_test)
            retrain_results["models"]["lstm"] = lstm_result

            logger.info("Step 5: Retraining Ensemble model...")
            ensemble_result = await self._retrain_ensemble(X_train, X_test, y_train, y_test)
            retrain_results["models"]["ensemble"] = ensemble_result

            logger.info("Step 6: Validating retrained models...")
            validation_results = await self._validate_models(retrain_results["models"], X_test, y_test)
            retrain_results["validation"] = validation_results

            logger.info("Step 7: Saving improved models...")
            save_results = await self._save_models_if_improved(retrain_results["models"], validation_results)
            retrain_results["saved"] = save_results

            logger.info("Step 8: Updating model registry...")
            await self._update_model_registry(retrain_results)

            logger.info("Step 9: Logging performance comparison...")
            self._log_performance_comparison(retrain_results)

            retrain_results["success"] = True

        except Exception as e:
            logger.error(f"Model retraining failed: {e}", exc_info=True)
            retrain_results["errors"].append(str(e))
            retrain_results["success"] = False

        await self._send_retraining_notification(retrain_results)

        logger.info("=" * 50)
        logger.info(f"Model retraining {'SUCCESSFUL' if retrain_results['success'] else 'FAILED'}")
        logger.info("=" * 50)

        return retrain_results

    async def _fetch_training_data(self) -> pd.DataFrame:
        """Fetch latest data for model retraining"""
        training_data = []

        async with AsyncSessionLocal() as db:
            users = await self._get_active_users(db)
            all_symbols = set()

            for user in users:
                symbols = await self._get_user_symbols(user.id, db)
                all_symbols.update(symbols)

        for symbol in list(all_symbols)[:50]:
            try:
                data = await self.provider_factory.fetch_data(symbol=symbol, period="5y", interval="1d")

                if not data.empty:
                    data = self._add_technical_indicators(data)
                    data["symbol"] = symbol
                    training_data.append(data)

            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                continue

        if not training_data:
            return pd.DataFrame()

        combined = pd.concat(training_data, axis=0)

        combined = combined.dropna()

        return combined

    async def _prepare_training_data(self, data: pd.DataFrame) -> Tuple:
        """Prepare features and targets for training"""

        # Define features (you may need to adjust based on your data)
        feature_columns = [
            "returns_1d",
            "returns_5d",
            "returns_21d",
            "volatility_21d",
            "volume_ratio",
            "rsi",
            "macd",
            "bb_position",
            "sma_50_ratio",
            "sma_200_ratio",
        ]

        # Define target (e.g., next day return)
        target_column = "forward_return_1d"

        # Ensure all required columns exist
        available_features = [col for col in feature_columns if col in data.columns]

        X = data[available_features].values
        y = data[target_column].values if target_column in data.columns else None

        if y is None:
            # Create target if not present (next day return)
            y = data["Close"].pct_change().shift(-1).dropna().values
            # Align X with y
            X = X[:-1]  # Remove last row (no target)

        # Split into train/test (time series split)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        return X_train, X_test, y_train, y_test

    async def _retrain_random_forest(self, X_train, X_test, y_train, y_test) -> Dict:
        """Retrain Random Forest model"""
        import os

        import joblib
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, r2_score

        result = {"success": False, "metrics": {}, "model_path": None}

        try:
            model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1)

            start_time = datetime.now()
            model.fit(X_train, y_train)
            training_time = (datetime.now() - start_time).total_seconds()

            # Evaluate
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            # Feature importance
            feature_importance = model.feature_importances_.tolist()

            result.update(
                {
                    "success": True,
                    "metrics": {
                        "mse": float(mse),
                        "rmse": float(rmse),
                        "r2": float(r2),
                        "training_time_seconds": training_time,
                        "n_features": X_train.shape[1],
                        "n_samples": len(X_train),
                    },
                    "feature_importance": feature_importance[:10],  # Top 10
                    "model_params": model.get_params(),
                }
            )

            # Save model temporarily
            model_path = f"models/random_forest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            os.makedirs("models", exist_ok=True)
            joblib.dump(model, model_path)
            result["model_path"] = model_path

            logger.info(f"Random Forest - RMSE: {rmse:.4f}, R2: {r2:.4f}")

        except Exception as e:
            logger.error(f"Random Forest retraining failed: {e}")
            result["error"] = str(e)

        return result

    async def _retrain_lstm(self, X_train, X_test, y_train, y_test) -> Dict:
        """Retrain LSTM model"""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        result = {"success": False, "metrics": {}, "model_path": None}

        try:
            # Reshape data for LSTM (samples, timesteps, features)
            # This assumes you want to use sequences
            seq_length = 20
            n_features = X_train.shape[1]

            # Create sequences
            X_train_seq = []
            y_train_seq = []

            for i in range(len(X_train) - seq_length):
                X_train_seq.append(X_train[i : i + seq_length])
                y_train_seq.append(y_train[i + seq_length])

            X_train_seq = np.array(X_train_seq)
            y_train_seq = np.array(y_train_seq)

            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train_seq)
            y_train_tensor = torch.FloatTensor(y_train_seq)

            # Create data loader
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

            # Define model
            class LSTMPredictor(nn.Module):
                def __init__(self, input_size, hidden_size=64, num_layers=2):
                    super().__init__()
                    self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2)
                    self.fc = nn.Linear(hidden_size, 1)

                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    last_output = lstm_out[:, -1, :]
                    output = self.fc(last_output)
                    return output.squeeze()

            model = LSTMPredictor(n_features)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Training loop
            start_time = datetime.now()
            n_epochs = 50
            train_losses = []

            for epoch in range(n_epochs):
                epoch_loss = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    output = model(batch_X)
                    loss = criterion(output, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                avg_loss = epoch_loss / len(train_loader)
                train_losses.append(avg_loss)

                if (epoch + 1) % 10 == 0:
                    logger.debug(f"LSTM Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")

            training_time = (datetime.now() - start_time).total_seconds()

            # Evaluate on test set
            model.eval()
            with torch.no_grad():
                # Prepare test sequences
                X_test_seq = []
                y_test_seq = []

                for i in range(len(X_test) - seq_length):
                    X_test_seq.append(X_test[i : i + seq_length])
                    y_test_seq.append(y_test[i + seq_length])

                if X_test_seq:
                    X_test_tensor = torch.FloatTensor(np.array(X_test_seq))
                    y_test_tensor = torch.FloatTensor(np.array(y_test_seq))

                    y_pred = model(X_test_tensor).numpy()
                    y_true = y_test_tensor.numpy()

                    mse = mean_squared_error(y_true, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_true, y_pred)
                else:
                    rmse = r2 = 0

            result.update(
                {
                    "success": True,
                    "metrics": {
                        "rmse": float(rmse),
                        "r2": float(r2),
                        "training_time_seconds": training_time,
                        "final_loss": train_losses[-1] if train_losses else 0,
                        "n_epochs": n_epochs,
                        "loss_history": train_losses[-10:],  # Last 10 losses
                    },
                    "model_params": {"hidden_size": 64, "num_layers": 2, "seq_length": seq_length},
                }
            )

            # Save model
            model_path = f"models/lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            os.makedirs("models", exist_ok=True)
            torch.save(
                {"model_state_dict": model.state_dict(), "model_params": {"input_size": n_features, "hidden_size": 64, "num_layers": 2}}, model_path
            )
            result["model_path"] = model_path

            logger.info(f"LSTM - RMSE: {rmse:.4f}, R2: {r2:.4f}")

        except Exception as e:
            logger.error(f"LSTM retraining failed: {e}")
            result["error"] = str(e)

        return result

    async def _retrain_ensemble(self, X_train, X_test, y_train, y_test) -> Dict:
        """Retrain ensemble model combining RF and LSTM"""
        result = {"success": False, "metrics": {}, "model_path": None}

        try:
            # This would combine predictions from both models
            # Simplified version - just returns RF result for now
            rf_result = await self._retrain_random_forest(X_train, X_test, y_train, y_test)

            if rf_result["success"]:
                result.update({"success": True, "metrics": rf_result["metrics"], "model_path": rf_result["model_path"]})

        except Exception as e:
            logger.error(f"Ensemble retraining failed: {e}")
            result["error"] = str(e)

        return result

    async def _validate_models(self, models: Dict, X_test, y_test) -> Dict:
        """Validate retrained models"""
        validation_results = {}

        for model_name, model_result in models.items():
            if not model_result.get("success"):
                validation_results[model_name] = {"valid": False, "reason": "Training failed"}
                continue

            # Check if metrics meet minimum requirements
            metrics = model_result.get("metrics", {})
            rmse = metrics.get("rmse", float("inf"))
            r2 = metrics.get("r2", -float("inf"))

            is_valid = (
                rmse < 0.1  # RMSE threshold
                and r2 > 0.3  # R2 threshold
            )

            validation_results[model_name] = {"valid": is_valid, "rmse": rmse, "r2": r2, "passed_rmse": rmse < 0.1, "passed_r2": r2 > 0.3}

            if not is_valid:
                logger.warning(f"Model {model_name} failed validation: RMSE={rmse:.4f}, R2={r2:.4f}")

        return validation_results

    async def _save_models_if_improved(self, models: Dict, validation_results: Dict) -> Dict:
        """Save models if they improved over current versions"""
        save_results = {}

        for model_name, model_result in models.items():
            if not model_result.get("success"):
                save_results[model_name] = {"saved": False, "reason": "Training failed"}
                continue

            validation = validation_results.get(model_name, {})
            if not validation.get("valid", False):
                save_results[model_name] = {"saved": False, "reason": "Validation failed"}
                continue

            # Compare with current model if exists
            current_metrics = await self._get_current_model_metrics(model_name)
            new_metrics = model_result.get("metrics", {})

            improved = True
            if current_metrics:
                # Check if new model is better
                improved = new_metrics.get("rmse", float("inf")) < current_metrics.get("rmse", float("inf"))

            if improved:
                # Move from temp path to final location
                import shutil

                final_path = f"models/current/{model_name}.pkl"
                os.makedirs("models/current", exist_ok=True)

                if os.path.exists(model_result["model_path"]):
                    shutil.copy2(model_result["model_path"], final_path)
                    save_results[model_name] = {
                        "saved": True,
                        "path": final_path,
                        "improved": improved,
                        "previous_rmse": current_metrics.get("rmse") if current_metrics else None,
                        "new_rmse": new_metrics.get("rmse"),
                    }
                    logger.info(f"Saved improved {model_name} model (RMSE: {new_metrics.get('rmse'):.4f})")
            else:
                save_results[model_name] = {
                    "saved": False,
                    "reason": "No improvement",
                    "current_rmse": current_metrics.get("rmse") if current_metrics else None,
                    "new_rmse": new_metrics.get("rmse"),
                }

        return save_results

    async def _get_current_model_metrics(self, model_name: str) -> Optional[Dict]:
        """Get metrics of current production model"""
        # This would load from a metrics database or file
        # Placeholder implementation
        return None

    async def _update_model_registry(self, retrain_results: Dict):
        """Update model registry with retraining results"""
        registry_entry = {"timestamp": retrain_results["timestamp"], "models": {}, "overall_success": retrain_results["success"]}

        for model_name, model_result in retrain_results.get("models", {}).items():
            if model_result.get("success"):
                registry_entry["models"][model_name] = {
                    "metrics": model_result.get("metrics", {}),
                    "path": model_result.get("model_path"),
                    "saved": retrain_results.get("saved", {}).get(model_name, {}).get("saved", False),
                }

        # Save to database or file
        import json

        registry_path = "models/registry.json"

        # Load existing registry
        registry = []
        if os.path.exists(registry_path):
            with open(registry_path, "r") as f:
                registry = json.load(f)

        # Add new entry
        registry.append(registry_entry)

        # Keep only last 10 entries
        if len(registry) > 10:
            registry = registry[-10:]

        # Save
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)

        logger.info(f"Model registry updated at {registry_path}")

    def _log_performance_comparison(self, retrain_results: Dict):
        """Log detailed performance comparison"""
        logger.info("-" * 50)
        logger.info("MODEL PERFORMANCE COMPARISON")
        logger.info("-" * 50)

        for model_name, model_result in retrain_results.get("models", {}).items():
            if model_result.get("success"):
                metrics = model_result.get("metrics", {})
                saved_info = retrain_results.get("saved", {}).get(model_name, {})

                logger.info(f"\n{model_name.upper()}:")
                logger.info(f"  RMSE: {metrics.get('rmse', 0):.4f}")
                logger.info(f"  R²: {metrics.get('r2', 0):.4f}")
                logger.info(f"  Training time: {metrics.get('training_time_seconds', 0):.1f}s")

                if saved_info.get("saved"):
                    logger.info(f"  ✓ Saved (improved from {saved_info.get('previous_rmse', 'N/A')} to {saved_info.get('new_rmse', 'N/A')})")
                else:
                    reason = saved_info.get("reason", "Not saved")
                    logger.info(f"  ✗ Not saved: {reason}")

        logger.info("-" * 50)

    async def _send_retraining_notification(self, results: Dict):
        """Send notification about retraining results"""
        if not self.alert_manager:
            return

        if results["success"]:
            # Success notification
            message = "✅ Model retraining completed successfully\n"
            for model_name, model_result in results.get("models", {}).items():
                if model_result.get("success"):
                    metrics = model_result.get("metrics", {})
                    message += f"\n{model_name}: RMSE={metrics.get('rmse', 0):.4f}, R²={metrics.get('r2', 0):.4f}"

            await self.alert_manager.send_info(message)

        else:
            # Failure notification
            message = "❌ Model retraining failed\n"
            message += f"Errors: {', '.join(results.get('errors', []))}"
            await self.alert_manager.send_error(message)

        async def run_crash_predictions(self):
            """Run crash predictions for all users"""
            logger.info("Running crash predictions")

            async with AsyncSessionLocal() as db:
                users = await self._get_active_users(db)

                for user in users:
                    try:
                        # Get user's executor
                        if user.id not in self.executors:
                            self.executors[user.id] = AutoHedgeExecutor(self.provider_factory, self.broker, db)

                        executor = self.executors[user.id]

                        await executor.monitor_and_execute(user.id)

                    except Exception as e:
                        logger.error(f"Error in predictions for user {user.id}: {e}")

    async def check_hedge_executions(self):
        """Check and execute hedges if needed"""
        logger.info("Checking hedge executions")

        async with AsyncSessionLocal() as db:
            users = await self._get_active_users(db)

            for user in users:
                if user.id in self.executors:
                    try:
                        await self.executors[user.id].monitor_and_execute(user.id)
                    except Exception as e:
                        logger.error(f"Hedge execution error for user {user.id}: {e}")

    async def update_dashboards(self):
        """Update all dashboards"""
        logger.info("Updating dashboards")

        async with AsyncSessionLocal() as db:
            users = await self._get_active_users(db)

            for user in users:
                try:
                    # Generate new dashboard
                    dashboard = await self._generate_user_dashboard(user.id, db)

                    # Save or send dashboard
                    await self._save_dashboard(user.id, dashboard)

                except Exception as e:
                    logger.error(f"Dashboard update error for user {user.id}: {e}")

    async def check_alerts(self):
        """Check and send alerts"""
        logger.info("Checking alerts")

        async with AsyncSessionLocal() as db:
            users = await self._get_active_users(db)

            for user in users:
                try:
                    crash_prob = await self._get_crash_probability(user.id, db)

                    if crash_prob > 0.7:
                        await self._send_alert(
                            user.id, f"⚠️ HIGH CRASH RISK: {crash_prob:.1%} probability", AlertLevel.CRITICAL, AlertCategory.CRASH_PREDICTION
                        )
                    elif crash_prob > 0.5:
                        await self._send_alert(
                            user.id, f"⚠️ Moderate crash risk: {crash_prob:.1%} probability", AlertLevel.WARNING, AlertCategory.CRASH_PREDICTION
                        )

                    # Check bubble detection
                    bubble = await self._check_bubble(user.id, db)
                    if bubble:
                        await self._send_alert(
                            user.id,
                            f" Bubble detected in {bubble['symbol']} with {bubble['confidence']:.1%} confidence",
                            AlertLevel.INFO,
                            AlertCategory.BUBBLE_DETECTION,
                        )

                except Exception as e:
                    logger.error(f"Alert check error for user {user.id}: {e}")

    async def _get_active_users(self, db) -> list:
        """Get users with active monitoring"""

        result = await db.execute(select(User).join(UserSettings).where(UserSettings.crash_monitoring, User.is_active))
        return result.scalars().all()

    async def _get_user_symbols(self, user_id: int, db) -> list:
        """Get symbols monitored by user"""

        result = await db.execute(select(Watchlist).where(Watchlist.user_id == user_id))
        watchlist = result.scalars().first()

        return watchlist.symbols if watchlist else []

    async def _get_crash_probability(self, user_id: int, db) -> float:
        """Get current crash probability for user"""

        result = await db.execute(
            select(CrashPrediction).where(CrashPrediction.user_id == user_id).order_by(CrashPrediction.timestamp.desc()).limit(1)
        )
        pred = result.scalars().first()

        return pred.probability if pred else 0.0

    async def _check_bubble(self, user_id: int, db) -> Optional[Dict]:
        """Check for bubble detection"""

        result = await db.execute(
            select(BubbleDetection).where(
                BubbleDetection.user_id == user_id, BubbleDetection.detected, BubbleDetection.timestamp > datetime.now() - timedelta(hours=24)
            )
        )
        bubble = result.scalars().first()

        if bubble:
            return {"symbol": bubble.symbol, "confidence": bubble.confidence, "crash_probability": bubble.crash_probability}

        return None

    async def _send_alert(self, user_id: int, message: str, level: AlertLevel, category: AlertCategory):
        """Send alert to user"""

        await self.alert_manager.send_alert(user_id=user_id, level=level, title="Crash Monitor Alert", message=message, category=category)

    async def _update_user_portfolio(self, user_id: int, db, portfolio_id: int = 1):
        """Update user portfolio data"""

        service = PortfolioService(db)
        await service.update_portfolio(portfolio_id, user_id)

    async def _generate_user_dashboard(self, user_id: int, db, portfolio_id: int = 1) -> go.Figure:
        """Generate dashboard for user"""
        service = PortfolioService(db)
        portfolio = await service.get_portfolio(user_id, portfolio_id)
        predictions = await self._get_user_predictions(user_id, db)

        dashboard = self.dashboard.create_dashboard(
            price_data=portfolio.price_history, predictions=predictions, stress_data=portfolio.stress_history, bubble_data=portfolio.bubble_data
        )

        return dashboard

    async def _save_dashboard(self, user_id: int, dashboard: go.Figure):
        """Save dashboard to file or database"""
        # Save as HTML
        dashboard.write_html(f"dashboards/user_{user_id}_dashboard.html")

    def stop(self):
        """Stop the scheduler"""
        self.scheduler.shutdown()
        logger.info("Crash monitor scheduler stopped")
