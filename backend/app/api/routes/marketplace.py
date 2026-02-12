import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.deps import get_current_active_user
from backend.app.core.marketplace import BacktestResults, StrategyListing as CoreListing, StrategyMarketplace
from backend.app.database import get_db
from backend.app.models import User
from backend.app.models.backtest import BacktestRun
from backend.app.schemas.marketplace import (
    BacktestResultsSchema,
    ReviewCreateRequest,
    StrategyListing as StrategyListingSchema,
    StrategyListingDetailed as StrategyListingDetailedSchema,
    StrategyPublishRequest,
    StrategyReviewSchema,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/marketplace", tags=["Marketplace"])

_marketplace: Optional[StrategyMarketplace] = None


def get_marketplace() -> StrategyMarketplace:
    """Get or initialize the marketplace instance (Lazy Initialization)"""
    global _marketplace
    if _marketplace is None:
        _marketplace = StrategyMarketplace()
    return _marketplace


def convert_core_to_schema(core_listing, marketplace: StrategyMarketplace = None, user_id: int = None) -> StrategyListingSchema:
    """Convert core StrategyListing to API schema"""
    is_fav = False
    if marketplace and user_id and core_listing.id:
        try:
            is_fav = marketplace.is_favorite(core_listing.id, user_id)
        except Exception:
            is_fav = False

    # Use user-provided pros/cons if available, otherwise auto-generate
    pros = core_listing.pros if core_listing.pros else _generate_pros(core_listing)
    cons = core_listing.cons if core_listing.cons else _generate_cons(core_listing)

    # Get backtest results for additional metrics
    br = core_listing.backtest_results

    return StrategyListingSchema(
        id=core_listing.id if core_listing.id else "unknown",
        name=core_listing.name,
        creator=core_listing.creator_name,
        description=core_listing.description,
        rating=core_listing.rating,
        reviews=core_listing.num_reviews,
        price=core_listing.price,
        category=core_listing.category,
        complexity=core_listing.complexity,
        time_horizon=getattr(core_listing, "time_horizon", "Medium-term"),
        total_return=core_listing.total_return,
        monthly_return=round(core_listing.total_return / 12, 2) if core_listing.total_return else 0,
        drawdown=abs(core_listing.max_drawdown),
        sharpe_ratio=core_listing.sharpe_ratio,
        win_rate=core_listing.win_rate,
        num_trades=core_listing.num_trades,
        avg_win=br.avg_win if br else 0.0,
        avg_loss=br.avg_loss if br else 0.0,
        profit_factor=br.profit_factor if br else 0.0,
        volatility=br.volatility if br else 0.0,
        sortino_ratio=br.sortino_ratio if br else 0.0,
        calmar_ratio=br.calmar_ratio if br else 0.0,
        var_95=br.var_95 if br else 0.0,
        initial_capital=br.initial_capital if br else 10000.0,
        total_downloads=core_listing.downloads,
        tags=core_listing.tags if core_listing.tags else [],
        best_for=core_listing.tags[:3] if core_listing.tags else [],
        pros=pros,
        cons=cons,
        is_favorite=is_fav,
        is_verified=core_listing.is_verified,
        verification_badge=core_listing.verification_badge,
        publish_date=core_listing.created_at.strftime("%Y-%m-%d") if core_listing.created_at else "",
    )


def _generate_pros(listing) -> List[str]:
    """Generate pros based on strategy characteristics"""
    pros = []
    if listing.sharpe_ratio > 1.5:
        pros.append("Excellent risk-adjusted returns")
    elif listing.sharpe_ratio > 1.0:
        pros.append("Good risk-adjusted returns")
    if listing.win_rate > 0.6:
        pros.append("High win rate")
    if abs(listing.max_drawdown) < 10:
        pros.append("Low drawdown")
    if listing.num_trades > 100:
        pros.append("Well-tested with many trades")
    if listing.is_verified:
        pros.append("Verified performance")
    if len(pros) < 2:
        pros.extend(["Systematic approach", "Clear entry/exit signals"])
    return pros[:4]


def _generate_cons(listing) -> List[str]:
    """Generate cons based on strategy characteristics"""
    cons = []
    if listing.sharpe_ratio < 1.0:
        cons.append("Lower risk-adjusted returns")
    if listing.win_rate < 0.5:
        cons.append("Win rate below 50%")
    if abs(listing.max_drawdown) > 15:
        cons.append("High drawdown potential")
    if listing.complexity == "Advanced":
        cons.append("Requires advanced knowledge")
    if listing.num_trades < 50:
        cons.append("Limited backtest history")
    if len(cons) < 2:
        cons.extend(["Requires monitoring", "Market-dependent performance"])
    return cons[:3]


@router.get("/", response_model=List[StrategyListingSchema])
async def get_strategies(
    category: Optional[str] = Query(None),
    complexity: Optional[str] = Query(None),
    min_sharpe: Optional[float] = Query(None),
    min_return: Optional[float] = Query(None),
    max_drawdown: Optional[float] = Query(None),
    search: Optional[str] = Query(None),
    sort_by: str = Query("sharpe_ratio"),
    limit: int = Query(50),
    current_user: User = Depends(get_current_active_user),
    marketplace: StrategyMarketplace = Depends(get_marketplace),
):
    """Browse marketplace strategies"""
    core_listings = marketplace.browse_strategies(
        category=None if category == "All" else category,
        complexity=None if complexity == "All" else complexity,
        min_sharpe=min_sharpe,
        min_return=min_return,
        max_drawdown=max_drawdown,
        search_query=search,
        sort_by=sort_by,
        limit=limit,
    )
    return [convert_core_to_schema(listing, marketplace, current_user.id) for listing in core_listings]


@router.get("/{strategy_id}", response_model=StrategyListingDetailedSchema)
async def get_strategy_details(
    strategy_id: int, current_user: User = Depends(get_current_active_user), marketplace: StrategyMarketplace = Depends(get_marketplace)
):
    """Get detailed information about a specific strategy including backtest results"""

    core_listings = marketplace.browse_strategies(limit=200)
    target_listing = next((listing for listing in core_listings if listing.id == strategy_id), None)

    if not target_listing:
        raise HTTPException(status_code=404, detail="Strategy not found")

    # Load full backtest data for the detail view
    backtest_data = marketplace.get_strategy_backtest(strategy_id)

    # Inject backtest results into listing so convert_core_to_schema has full metrics
    if backtest_data:
        target_listing.backtest_results = backtest_data["backtest_results"]

    detailed_schema = StrategyListingDetailedSchema(**convert_core_to_schema(target_listing, marketplace, current_user.id).dict())

    if backtest_data:
        res = backtest_data["backtest_results"]
        detailed_schema.backtest_results = BacktestResultsSchema(
            total_return=res.total_return,
            annualized_return=res.annualized_return,
            sharpe_ratio=res.sharpe_ratio,
            sortino_ratio=res.sortino_ratio,
            max_drawdown=res.max_drawdown,
            max_drawdown_duration=res.max_drawdown_duration,
            calmar_ratio=res.calmar_ratio,
            num_trades=res.num_trades,
            win_rate=res.win_rate,
            profit_factor=res.profit_factor,
            avg_win=res.avg_win,
            avg_loss=res.avg_loss,
            avg_trade_duration=res.avg_trade_duration,
            volatility=res.volatility,
            var_95=res.var_95,
            cvar_95=res.cvar_95,
            equity_curve=res.equity_curve,
            trades=res.trades,
            daily_returns=res.daily_returns,
            start_date=res.start_date.isoformat() if res.start_date else None,
            end_date=res.end_date.isoformat() if res.end_date else None,
            initial_capital=res.initial_capital,
            symbols=res.symbols,
        )

    # Populate reviews
    reviews = marketplace.get_reviews(strategy_id)
    detailed_schema.reviews_list = [
        StrategyReviewSchema(
            id=r.id,
            strategy_id=r.strategy_id,
            user_id=r.user_id,
            username=r.username,
            rating=r.rating,
            review_text=r.review_text,
            performance_achieved=r.performance_achieved,
            created_at=r.created_at.isoformat() if r.created_at else "",
        )
        for r in reviews
    ]

    return detailed_schema


@router.post("/{strategy_id}/favorite")
async def favorite_strategy(
    strategy_id: int, current_user: User = Depends(get_current_active_user), marketplace: StrategyMarketplace = Depends(get_marketplace)
):
    """Add a strategy to user's favorites"""
    success = marketplace.toggle_favorite(strategy_id, current_user.id, favorite=True)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to favorite strategy")
    return {"status": "success"}


@router.delete("/{strategy_id}/favorite")
async def unfavorite_strategy(
    strategy_id: int, current_user: User = Depends(get_current_active_user), marketplace: StrategyMarketplace = Depends(get_marketplace)
):
    """Remove a strategy from user's favorites"""
    success = marketplace.toggle_favorite(strategy_id, current_user.id, favorite=False)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to unfavorite strategy")
    return {"status": "success"}


@router.post("/{strategy_id}/download")
async def record_download(
    strategy_id: int, current_user: User = Depends(get_current_active_user), marketplace: StrategyMarketplace = Depends(get_marketplace)
):
    """Record a strategy download and increment download count"""
    success = marketplace.record_download(strategy_id, current_user.id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to record download")
    return {"status": "success"}


@router.post("/{strategy_id}/review", status_code=status.HTTP_201_CREATED)
async def create_review(
    strategy_id: int,
    request: ReviewCreateRequest,
    current_user: User = Depends(get_current_active_user),
    marketplace: StrategyMarketplace = Depends(get_marketplace),
):
    """Create or update a review for a strategy"""
    review_id = marketplace.add_review(
        strategy_id=strategy_id,
        user_id=current_user.id,
        username=current_user.username or "Unknown",
        rating=request.rating,
        review_text=request.review_text,
        performance_achieved=request.performance_achieved,
    )
    if not review_id:
        raise HTTPException(status_code=500, detail="Failed to create review")
    return {"id": review_id, "status": "created"}


@router.get("/{strategy_id}/reviews", response_model=List[StrategyReviewSchema])
async def get_reviews(
    strategy_id: int,
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
    marketplace: StrategyMarketplace = Depends(get_marketplace),
):
    """Get reviews for a strategy"""
    reviews = marketplace.get_reviews(strategy_id, limit)
    return [
        StrategyReviewSchema(
            id=r.id,
            strategy_id=r.strategy_id,
            user_id=r.user_id,
            username=r.username,
            rating=r.rating,
            review_text=r.review_text,
            performance_achieved=r.performance_achieved,
            created_at=r.created_at.isoformat() if r.created_at else "",
        )
        for r in reviews
    ]


@router.post("/publish", status_code=status.HTTP_201_CREATED)
async def publish_strategy(
    request: StrategyPublishRequest,
    current_user: User = Depends(get_current_active_user),
    marketplace: StrategyMarketplace = Depends(get_marketplace),
    db: AsyncSession = Depends(get_db),
):
    """Publish a new strategy to the marketplace"""
    try:
        backtest_results = BacktestResults()
        strategy_config = {}

        if request.backtest_id:
            # Fetch backtest from backtest_runs table
            result = await db.execute(
                select(BacktestRun).filter(
                    BacktestRun.id == request.backtest_id,
                    BacktestRun.user_id == current_user.id,
                )
            )
            backtest_run = result.scalar_one_or_none()

            if not backtest_run:
                raise HTTPException(status_code=404, detail="Backtest not found or does not belong to you")

            if backtest_run.status != "completed":
                raise HTTPException(status_code=400, detail="Backtest must be completed before publishing")

            strategy_config = backtest_run.strategy_config or {}

            # Map BacktestRun fields to BacktestResults
            backtest_results = BacktestResults(
                total_return=backtest_run.total_return_pct or 0.0,
                sharpe_ratio=backtest_run.sharpe_ratio or 0.0,
                max_drawdown=backtest_run.max_drawdown or 0.0,
                win_rate=backtest_run.win_rate or 0.0,
                num_trades=backtest_run.total_trades or 0,
                initial_capital=backtest_run.initial_capital or 100000.0,
                symbols=backtest_run.symbols or [],
                equity_curve=backtest_run.equity_curve or [],
                trades=backtest_run.trades_json or [],
            )

        # Determine strategy_type from request or backtest config
        strategy_type = request.strategy_key or strategy_config.get("strategy_key", "Custom")

        listing = CoreListing(
            id=None,
            name=request.name,
            description=request.description,
            creator_id=current_user.id,
            creator_name=current_user.username or "Unknown",
            strategy_type=strategy_type,
            category=request.category,
            complexity=request.complexity,
            parameters=strategy_config,
            backtest_results=backtest_results,
            price=request.price,
            is_public=request.is_public,
            tags=request.tags,
            pros=request.pros,
            cons=request.cons,
            risk_level=request.risk_level or "medium",
            recommended_capital=request.recommended_capital or 10000.0,
        )

        strategy_id = marketplace.publish_strategy_with_backtest(listing)

        return {"id": strategy_id, "status": "published"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to publish strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))
