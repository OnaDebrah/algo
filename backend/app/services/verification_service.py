import logging
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.backtest import BacktestRun
from backend.app.models.live import LiveStrategy
from backend.app.core.marketplace import StrategyMarketplace

logger = logging.getLogger(__name__)

class VerificationService:
    """
    Service for verifying strategy performance by comparing 
    Live results vs. Backtest expectations.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.marketplace = StrategyMarketplace()

    async def verify_live_strategy(self, strategy_id: int) -> Dict:
        """
        Calculates drift and assigns a verification badge to a live strategy.
        """
        # 1. Fetch live strategy
        stmt = select(LiveStrategy).where(LiveStrategy.id == strategy_id)
        result = await self.db.execute(stmt)
        strategy = result.scalars().first()

        if not strategy:
            return {"status": "error", "message": "Strategy not found"}

        if not strategy.backtest_id:
            return {"status": "error", "message": "No associated backtest for this strategy"}

        # 2. Fetch associated backtest
        stmt = select(BacktestRun).where(BacktestRun.id == strategy.backtest_id)
        result = await self.db.execute(stmt)
        backtest = result.scalars().first()

        if not backtest:
            return {"status": "error", "message": "Backtest data not found"}

        # 3. Calculate Performance Drift
        # Metrics to compare: Sharpe Ratio, Max Drawdown, Annualized Return
        
        live_sharpe = strategy.sharpe_ratio or 0.0
        bt_sharpe = backtest.sharpe_ratio or 0.0
        
        live_drawdown = abs(strategy.max_drawdown or 0.0)
        bt_drawdown = abs(backtest.max_drawdown or 0.0)
        
        live_return = strategy.total_return_pct or 0.0
        bt_return = backtest.total_return_pct or 0.0

        # Drift Calculation (as a percentage of backtest)
        sharpe_drift = (live_sharpe - bt_sharpe) / (bt_sharpe if bt_sharpe != 0 else 1.0)
        drawdown_drift = (live_drawdown - bt_drawdown) / (bt_drawdown if bt_drawdown != 0 else 1.0)
        
        # 4. Assign Badge
        badge = "UNVERIFIED"
        score = 0.0
        
        # Simple scoring: 
        # - Low drawdown drift is good
        # - Higher or equal sharpe is good
        if strategy.total_trades and strategy.total_trades > 50:
            if abs(sharpe_drift) < 0.1 and drawdown_drift < 0.1:
                badge = "INSTITUTIONAL"  # Platinum: Matches backtest perfectly
                score = 0.95
            elif abs(sharpe_drift) < 0.25 and drawdown_drift < 0.25:
                badge = "VERIFIED"       # Gold: Consistent with backtest
                score = 0.80
            elif sharpe_drift < -0.5 or drawdown_drift > 0.5:
                badge = "DRIFTING"       # Warning: Dangerously different from backtest
                score = 0.30
            else:
                badge = "CONSISTENT"    # Silver: Reasonable performance
                score = 0.60
        else:
            badge = "PENDING"  # Not enough trades to verify
            score = 0.0

        verification_data = {
            "strategy_id": strategy_id,
            "badge": badge,
            "score": score,
            "sharpe_drift": sharpe_drift,
            "drawdown_drift": drawdown_drift,
            "live_trades": strategy.total_trades,
            "verified_at": datetime.now(timezone.utc).isoformat()
        }

        # 5. Update strategy if it's in marketplace
        # Note: We need a way to link LiveStrategy to MarketplaceStrategy
        # For now, we'll just return the info.
        
        return verification_data

    async def get_marketplace_badges(self) -> Dict[int, str]:
        """
        Global scan to return all verified strategy IDs and their badges
        """
        # In a real implementation, this would query a dedicated verification table
        # or join live metrics with marketplace listings.
        return {}
