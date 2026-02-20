import logging
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import delete, desc, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.marketplace import (
    MarketplaceStrategy,
    StrategyBacktest,
    StrategyDownload,
    StrategyFavorite,
    StrategyReview as StrategyReviewModel,
)

logger = logging.getLogger(__name__)


# ── Dataclasses (used by routes to build listings) ────────────────────


@dataclass
class StrategyReviewData:
    """User review of a strategy (data transfer object)"""

    id: Optional[int] = None
    strategy_id: int = 0
    user_id: int = 0
    username: str = ""
    rating: int = 5
    review_text: str = ""
    performance_achieved: Dict = None
    created_at: datetime = None

    def __post_init__(self):
        if self.performance_achieved is None:
            self.performance_achieved = {}
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class BacktestResults:
    """Complete backtest results for a strategy"""

    # Core metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    calmar_ratio: float = 0.0

    # Trading metrics
    num_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade_duration: float = 0.0

    # Risk metrics
    volatility: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0

    # Time series data (serialized as JSON/pickle)
    equity_curve: List[Dict] = None  # [{date, equity, drawdown}]
    trades: List[Dict] = None  # [{entry_date, exit_date, pnl, return, ...}]
    daily_returns: List[Dict] = None  # [{date, return}]

    # Backtest configuration
    start_date: datetime = None
    end_date: datetime = None
    initial_capital: float = 100000.0
    symbols: List[str] = None

    def __post_init__(self):
        if self.equity_curve is None:
            self.equity_curve = []
        if self.trades is None:
            self.trades = []
        if self.daily_returns is None:
            self.daily_returns = []
        if self.symbols is None:
            self.symbols = []

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "BacktestResults":
        """Create from dictionary"""
        return cls(**data)


@dataclass
class StrategyListing:
    """Strategy listing with backtest data"""

    id: Optional[int] = None
    name: str = ""
    description: str = ""
    creator_id: int = 0
    creator_name: str = ""
    strategy_type: str = ""
    category: str = ""
    complexity: str = ""

    # Strategy configuration
    parameters: Dict = None

    # Backtest results (complete)
    backtest_results: BacktestResults = None

    # Quick access metrics (denormalized for filtering)
    sharpe_ratio: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    num_trades: int = 0

    # Marketplace specific
    price: float = 0.0
    is_public: bool = True
    is_verified: bool = False
    verification_badge: Optional[str] = None
    version: str = "1.0.0"
    tags: List[str] = None

    # Social features
    downloads: int = 0
    rating: float = 0.0
    num_ratings: int = 0
    num_reviews: int = 0

    # User-provided content
    pros: List[str] = None
    cons: List[str] = None
    risk_level: str = "medium"
    recommended_capital: float = 10000.0

    # Timestamps
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.tags is None:
            self.tags = []
        if self.pros is None:
            self.pros = []
        if self.cons is None:
            self.cons = []
        if self.backtest_results is None:
            self.backtest_results = BacktestResults()
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


# ── MarketplaceService (fully async, SQLAlchemy) ──────────────────────


class MarketplaceService:
    @staticmethod
    async def publish_strategy_with_backtest(
        db: AsyncSession, listing: StrategyListing, equity_df: pd.DataFrame = None, trades_df: pd.DataFrame = None
    ) -> int:
        """Publish a strategy with complete backtest results."""

        br = listing.backtest_results

        new_strategy = MarketplaceStrategy(
            # Core fields from listing
            name=listing.name,
            description=listing.description,
            creator_id=listing.creator_id,
            creator_name=listing.creator_name,
            strategy_type=listing.strategy_type,
            category=listing.category,
            complexity=listing.complexity,
            parameters=listing.parameters,
            # Performance metrics from backtest_results
            sharpe_ratio=br.sharpe_ratio,
            total_return=br.total_return,
            max_drawdown=br.max_drawdown,
            win_rate=br.win_rate,
            num_trades=br.num_trades,
            # Marketplace fields from listing (NOT backtest_results)
            price=listing.price,
            is_public=listing.is_public,
            is_verified=listing.is_verified,
            verification_badge=listing.verification_badge,
            version=listing.version,
            tags=listing.tags,
            pros=listing.pros,
            cons=listing.cons,
            risk_level=listing.risk_level,
            recommended_capital=listing.recommended_capital,
        )
        db.add(new_strategy)
        await db.flush()  # Get strategy ID

        # 2. Store the Backtest Blobs
        backtest = StrategyBacktest(
            strategy_id=new_strategy.id,
            version=listing.version,
            backtest_data=br.to_dict(),
            equity_curve=pickle.dumps(equity_df) if equity_df is not None else None,
            trades_history=pickle.dumps(trades_df) if trades_df is not None else None,
            initial_capital=br.initial_capital,
        )
        db.add(backtest)
        return new_strategy.id

    @staticmethod
    async def get_strategy_backtest(strategy_id: int, db: AsyncSession) -> Optional[Dict]:
        """Get complete backtest results for a strategy."""
        stmt = select(StrategyBacktest).where(StrategyBacktest.strategy_id == strategy_id).order_by(StrategyBacktest.created_at.desc())
        result = await db.execute(stmt)
        row = result.scalars().first()

        if not row:
            return None

        return {
            "results": row.backtest_data,
            "equity_curve": pickle.loads(row.equity_curve) if row.equity_curve else None,
            "trades": pickle.loads(row.trades_history) if row.trades_history else None,
        }

    @staticmethod
    async def browse_strategies(
        db: AsyncSession,
        category: Optional[str] = None,
        complexity: Optional[str] = None,
        min_sharpe: Optional[float] = None,
        min_return: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        min_win_rate: Optional[float] = None,
        search_query: Optional[str] = None,
        sort_by: str = "sharpe_ratio",
        limit: int = 50,
        offset: int = 0,
    ) -> List[MarketplaceStrategy]:
        """Browse strategies with performance-based filters."""
        stmt = select(MarketplaceStrategy).where(MarketplaceStrategy.is_public == True)  # noqa: E712

        if category:
            stmt = stmt.where(MarketplaceStrategy.category == category)
        if complexity:
            stmt = stmt.where(MarketplaceStrategy.complexity == complexity)
        if min_sharpe is not None:
            stmt = stmt.where(MarketplaceStrategy.sharpe_ratio >= min_sharpe)
        if min_return is not None:
            stmt = stmt.where(MarketplaceStrategy.total_return >= min_return)
        if max_drawdown is not None:
            stmt = stmt.where(MarketplaceStrategy.max_drawdown >= max_drawdown)
        if min_win_rate is not None:
            stmt = stmt.where(MarketplaceStrategy.win_rate >= min_win_rate)
        if search_query:
            stmt = stmt.where(
                or_(
                    MarketplaceStrategy.name.ilike(f"%{search_query}%"),
                    MarketplaceStrategy.description.ilike(f"%{search_query}%"),
                )
            )

        sort_attr = getattr(MarketplaceStrategy, sort_by, MarketplaceStrategy.sharpe_ratio)
        stmt = stmt.order_by(desc(sort_attr)).limit(limit).offset(offset)

        result = await db.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def compare_strategies(db: AsyncSession, strategy_ids: List[int]) -> pd.DataFrame:
        """Compare multiple strategies side-by-side."""
        stmt = select(MarketplaceStrategy).where(MarketplaceStrategy.id.in_(strategy_ids))
        result = await db.execute(stmt)
        strategies = result.scalars().all()

        comparison_data = [
            {
                "Strategy": s.name,
                "Sharpe Ratio": s.sharpe_ratio,
                "Total Return": f"{s.total_return:.2f}%",
                "Max Drawdown": f"{s.max_drawdown:.2f}%",
                "Win Rate": f"{s.win_rate:.2f}%",
                "Num Trades": s.num_trades,
                "Rating": f"{s.rating:.1f}",
                "Downloads": s.downloads,
            }
            for s in strategies
        ]
        return pd.DataFrame(comparison_data)

    @staticmethod
    async def toggle_favorite(db: AsyncSession, strategy_id: int, user_id: int, favorite: bool = True) -> bool:
        """Add or remove a strategy from user's favorites."""
        try:
            if favorite:
                stmt = pg_insert(StrategyFavorite).values(strategy_id=strategy_id, user_id=user_id).on_conflict_do_nothing()
                await db.execute(stmt)
            else:
                stmt = delete(StrategyFavorite).where(
                    StrategyFavorite.strategy_id == strategy_id,
                    StrategyFavorite.user_id == user_id,
                )
                await db.execute(stmt)
            return True
        except Exception as e:
            logger.error(f"Favorite toggle error: {e}")
            return False

    @staticmethod
    async def record_download(db: AsyncSession, strategy_id: int, user_id: int) -> bool:
        """Record a download and increment counter."""
        try:
            db.add(StrategyDownload(strategy_id=strategy_id, user_id=user_id))
            await db.execute(
                update(MarketplaceStrategy).where(MarketplaceStrategy.id == strategy_id).values(downloads=MarketplaceStrategy.downloads + 1)
            )
            return True
        except Exception as e:
            logger.error(f"Download record error: {e}")
            return False

    @staticmethod
    async def add_review(
        db: AsyncSession,
        strategy_id: int,
        user_id: int,
        username: str,
        rating: int,
        review_text: str,
        performance: Dict = None,
    ) -> Optional[int]:
        """Upsert a review and update strategy aggregates."""
        # 1. Upsert review (uses ORM model, not dataclass)
        stmt = (
            pg_insert(StrategyReviewModel)
            .values(
                strategy_id=strategy_id,
                user_id=user_id,
                username=username,
                rating=rating,
                review_text=review_text,
                performance_achieved=performance,
            )
            .on_conflict_do_update(
                index_elements=["strategy_id", "user_id"],
                set_={
                    "rating": rating,
                    "review_text": review_text,
                    "performance_achieved": performance,
                    "created_at": func.current_timestamp(),
                },
            )
            .returning(StrategyReviewModel.id)
        )

        res = await db.execute(stmt)
        review_id = res.scalar()

        # 2. Recalculate aggregated rating
        stats_stmt = select(
            func.avg(StrategyReviewModel.rating),
            func.count(StrategyReviewModel.id),
        ).where(StrategyReviewModel.strategy_id == strategy_id)

        stats_res = await db.execute(stats_stmt)
        avg_rating, review_count = stats_res.first()

        await db.execute(
            update(MarketplaceStrategy)
            .where(MarketplaceStrategy.id == strategy_id)
            .values(
                rating=avg_rating or 0,
                num_ratings=review_count,
                num_reviews=review_count,
            )
        )
        return review_id

    @staticmethod
    async def get_reviews(db: AsyncSession, strategy_id: int, limit: int = 20) -> List[StrategyReviewModel]:
        """Get reviews for a strategy."""
        stmt = (
            select(StrategyReviewModel)
            .where(StrategyReviewModel.strategy_id == strategy_id)
            .order_by(desc(StrategyReviewModel.created_at))
            .limit(limit)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def is_favorite(db: AsyncSession, strategy_id: int, user_id: int) -> bool:
        """Check if a strategy is in user's favorites."""
        stmt = select(StrategyFavorite).where(
            StrategyFavorite.strategy_id == strategy_id,
            StrategyFavorite.user_id == user_id,
        )
        result = await db.execute(stmt)
        return result.scalars().first() is not None
