import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...api.deps import get_current_active_user
from ...database import get_db
from ...models import User
from ...models.backtest import BacktestRun
from ...models.custom_strategy import CustomStrategy
from ...models.marketplace import StrategyPurchase
from ...schemas.marketplace import (
    BacktestResultsSchema,
    ReviewCreateRequest,
    StrategyListing as StrategyListingSchema,
    StrategyListingDetailed as StrategyListingDetailedSchema,
    StrategyPublishRequest,
    StrategyReviewSchema,
)
from ...services.marketplace_service import BacktestResults, MarketplaceService, StrategyListing
from ...services.social_service import ActivityService
from ...strategies.catelog.strategy_catalog import get_catalog

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/marketplace", tags=["Marketplace"])

_marketplace: Optional[MarketplaceService] = None


def get_marketplace() -> MarketplaceService:
    """Get or initialize the marketplace instance (Lazy Initialization)"""
    global _marketplace
    if _marketplace is None:
        _marketplace = MarketplaceService()
    return _marketplace


async def convert_core_to_schema(
    core_listing, db: AsyncSession, marketplace: MarketplaceService = None, user_id: int = None
) -> StrategyListingSchema:
    """Convert a MarketplaceStrategy ORM object to the API schema."""
    is_fav = False
    is_purchased = False
    if marketplace and user_id and core_listing.id:
        try:
            is_fav = await marketplace.is_favorite(db, core_listing.id, user_id)
        except Exception:
            is_fav = False
        # Check purchase status
        price = getattr(core_listing, "price", 0) or 0
        if price <= 0:
            is_purchased = True  # Free strategies are always "owned"
        else:
            try:
                purchase_result = await db.execute(
                    select(StrategyPurchase).where(
                        StrategyPurchase.user_id == user_id,
                        StrategyPurchase.strategy_id == core_listing.id,
                        StrategyPurchase.status == "completed",
                    )
                )
                is_purchased = purchase_result.scalar_one_or_none() is not None
            except Exception:
                is_purchased = False

    # Use user-provided pros/cons if available, otherwise auto-generate
    pros = core_listing.pros if core_listing.pros else _generate_pros(core_listing)
    cons = core_listing.cons if core_listing.cons else _generate_cons(core_listing)

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
        # These extended metrics default to 0 since MarketplaceStrategy ORM
        # doesn't store them — they come from the full backtest detail view
        avg_win=0.0,
        avg_loss=0.0,
        profit_factor=0.0,
        volatility=0.0,
        sortino_ratio=0.0,
        calmar_ratio=0.0,
        var_95=0.0,
        initial_capital=10000.0,
        total_downloads=core_listing.downloads,
        tags=core_listing.tags if core_listing.tags else [],
        best_for=core_listing.tags[:3] if core_listing.tags else [],
        pros=pros,
        cons=cons,
        is_favorite=is_fav,
        is_verified=core_listing.is_verified,
        is_proprietary=getattr(core_listing, "is_proprietary", False),
        is_purchased=is_purchased,
        status=getattr(core_listing, "status", "approved"),
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


@router.get("/leaderboard")
async def get_leaderboard(
    metric: str = Query("sharpe_ratio", enum=["sharpe_ratio", "total_return", "win_rate", "downloads", "rating"]),
    category: Optional[str] = Query(None),
    limit: int = Query(25, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
    marketplace: MarketplaceService = Depends(get_marketplace),
    db: AsyncSession = Depends(get_db),
):
    """Get strategy leaderboard ranked by a chosen metric."""
    sort_map = {
        "sharpe_ratio": "sharpe_ratio",
        "total_return": "total_return",
        "win_rate": "win_rate",
        "downloads": "downloads",
        "rating": "rating",
    }
    core_listings = await marketplace.browse_strategies(
        db,
        category=None if category == "All" else category,
        sort_by=sort_map.get(metric, "sharpe_ratio"),
        limit=limit,
    )
    results = []
    for rank, listing in enumerate(core_listings, 1):
        schema = await convert_core_to_schema(listing, db, marketplace, current_user.id)
        results.append(
            {
                "rank": rank,
                "strategy": schema.model_dump(),
            }
        )
    return results


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
    marketplace: MarketplaceService = Depends(get_marketplace),
    db: AsyncSession = Depends(get_db),
):
    """Browse marketplace strategies"""
    core_listings = await marketplace.browse_strategies(
        db,
        category=None if category == "All" else category,
        complexity=None if complexity == "All" else complexity,
        min_sharpe=min_sharpe,
        min_return=min_return,
        max_drawdown=max_drawdown,
        search_query=search,
        sort_by=sort_by,
        limit=limit,
    )
    return [await convert_core_to_schema(listing, db, marketplace, current_user.id) for listing in core_listings]


@router.get("/{strategy_id}", response_model=StrategyListingDetailedSchema)
async def get_strategy_details(
    strategy_id: int,
    current_user: User = Depends(get_current_active_user),
    marketplace: MarketplaceService = Depends(get_marketplace),
    db: AsyncSession = Depends(get_db),
):
    """Get detailed information about a specific strategy including backtest results"""

    core_listings = await marketplace.browse_strategies(db, limit=200)
    target_listing = next((listing for listing in core_listings if listing.id == strategy_id), None)

    if not target_listing:
        raise HTTPException(status_code=404, detail="Strategy not found")

    # Load full backtest data for the detail view
    backtest_data = await marketplace.get_strategy_backtest(strategy_id, db)

    # Build the base schema from the ORM listing
    base_schema = await convert_core_to_schema(target_listing, db, marketplace, current_user.id)
    detailed_schema = StrategyListingDetailedSchema(**base_schema.model_dump())

    # Populate full backtest results if available
    if backtest_data:
        # backtest_data["results"] is a dict (JSONB from DB), not a BacktestResults object
        res = backtest_data["results"]
        detailed_schema.backtest_results = BacktestResultsSchema(
            total_return=res.get("total_return", 0.0),
            annualized_return=res.get("annualized_return", 0.0),
            sharpe_ratio=res.get("sharpe_ratio", 0.0),
            sortino_ratio=res.get("sortino_ratio", 0.0),
            max_drawdown=res.get("max_drawdown", 0.0),
            max_drawdown_duration=res.get("max_drawdown_duration", 0),
            calmar_ratio=res.get("calmar_ratio", 0.0),
            num_trades=res.get("num_trades", 0),
            win_rate=res.get("win_rate", 0.0),
            profit_factor=res.get("profit_factor", 0.0),
            avg_win=res.get("avg_win", 0.0),
            avg_loss=res.get("avg_loss", 0.0),
            avg_trade_duration=res.get("avg_trade_duration", 0.0),
            volatility=res.get("volatility", 0.0),
            var_95=res.get("var_95", 0.0),
            cvar_95=res.get("cvar_95", 0.0),
            equity_curve=res.get("equity_curve", []),
            trades=res.get("trades", []),
            daily_returns=res.get("daily_returns", []),
            start_date=res.get("start_date"),
            end_date=res.get("end_date"),
            initial_capital=res.get("initial_capital", 100000.0),
            symbols=res.get("symbols", []),
        )

    # Populate reviews
    reviews = await marketplace.get_reviews(db, strategy_id)
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
    strategy_id: int,
    current_user: User = Depends(get_current_active_user),
    marketplace: MarketplaceService = Depends(get_marketplace),
    db: AsyncSession = Depends(get_db),
):
    """Add a strategy to user's favorites"""
    success = await marketplace.toggle_favorite(db, strategy_id, current_user.id, favorite=True)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to favorite strategy")
    return {"status": "success"}


@router.delete("/{strategy_id}/favorite")
async def unfavorite_strategy(
    strategy_id: int,
    current_user: User = Depends(get_current_active_user),
    marketplace: MarketplaceService = Depends(get_marketplace),
    db: AsyncSession = Depends(get_db),
):
    """Remove a strategy from user's favorites"""
    success = await marketplace.toggle_favorite(db, strategy_id, current_user.id, favorite=False)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to unfavorite strategy")
    return {"status": "success"}


@router.post("/{strategy_id}/download")
async def record_download(
    strategy_id: int,
    current_user: User = Depends(get_current_active_user),
    marketplace: MarketplaceService = Depends(get_marketplace),
    db: AsyncSession = Depends(get_db),
):
    """Record a strategy download and increment download count"""
    success = await marketplace.record_download(db, strategy_id, current_user.id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to record download")
    return {"status": "success"}


@router.post("/{strategy_id}/review", status_code=status.HTTP_201_CREATED)
async def create_review(
    strategy_id: int,
    request: ReviewCreateRequest,
    current_user: User = Depends(get_current_active_user),
    marketplace: MarketplaceService = Depends(get_marketplace),
    db: AsyncSession = Depends(get_db),
):
    """Create or update a review for a strategy"""
    review_id = await marketplace.add_review(
        db=db,
        strategy_id=strategy_id,
        user_id=current_user.id,
        username=current_user.username or "Unknown",
        rating=request.rating,
        review_text=request.review_text,
        performance=request.performance_achieved,
    )
    if not review_id:
        raise HTTPException(status_code=500, detail="Failed to create review")
    return {"id": review_id, "status": "created"}


@router.get("/{strategy_id}/reviews", response_model=List[StrategyReviewSchema])
async def get_reviews(
    strategy_id: int,
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
    marketplace: MarketplaceService = Depends(get_marketplace),
    db: AsyncSession = Depends(get_db),
):
    """Get reviews for a strategy"""
    reviews = await marketplace.get_reviews(db, strategy_id, limit)
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
    marketplace: MarketplaceService = Depends(get_marketplace),
    db: AsyncSession = Depends(get_db),
):
    """Publish a new strategy to the marketplace"""
    try:
        backtest_results = BacktestResults()
        strategy_config = {}

        # Determine if this is a proprietary (built-in) strategy
        catalog = get_catalog()
        strategy_key = request.strategy_key or ""
        is_proprietary = strategy_key in catalog.strategies

        # Access control: only superusers can publish proprietary strategies
        if is_proprietary and not current_user.is_superuser:
            raise HTTPException(
                status_code=403,
                detail="Only administrators can publish proprietary strategies to the marketplace",
            )

        # For custom strategies, verify ownership of the custom strategy
        if not is_proprietary and request.custom_strategy_id:
            result = await db.execute(
                select(CustomStrategy).filter(
                    CustomStrategy.id == request.custom_strategy_id,
                    CustomStrategy.user_id == current_user.id,
                )
            )
            if not result.scalar_one_or_none():
                raise HTTPException(status_code=404, detail="Custom strategy not found or does not belong to you")

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

            # Enforce minimum performance standards
            sharpe = backtest_run.sharpe_ratio or 0.0
            total_return = backtest_run.total_return_pct or 0.0
            if sharpe < 1.0:
                raise HTTPException(status_code=400, detail="Strategy must have Sharpe Ratio >= 1.0 to publish")
            if total_return < 10.0:
                raise HTTPException(status_code=400, detail="Strategy must have Total Return >= 10% to publish")

            # Map BacktestRun fields to BacktestResults (including extended metrics)
            ext = backtest_run.extended_results or {}
            backtest_results = BacktestResults(
                total_return=total_return,
                sharpe_ratio=sharpe,
                max_drawdown=backtest_run.max_drawdown or 0.0,
                win_rate=backtest_run.win_rate or 0.0,
                num_trades=backtest_run.total_trades or 0,
                initial_capital=backtest_run.initial_capital or 100000.0,
                symbols=backtest_run.symbols or [],
                equity_curve=backtest_run.equity_curve or [],
                trades=backtest_run.trades_json or [],
                # Extended metrics from the backtest's extended_results JSON
                avg_win=ext.get("avg_win", 0.0),
                avg_loss=ext.get("avg_loss", 0.0),
                profit_factor=ext.get("profit_factor", 0.0),
                volatility=ext.get("volatility", 0.0),
                sortino_ratio=ext.get("sortino_ratio", 0.0),
                calmar_ratio=ext.get("calmar_ratio", 0.0),
                var_95=ext.get("var_95", 0.0),
                cvar_95=ext.get("cvar_95", 0.0),
            )

        # Determine strategy_type from request or backtest config
        strategy_type = request.strategy_key or strategy_config.get("strategy_key", "Custom")

        # Superuser publishing proprietary → approved immediately
        # Regular user publishing custom → pending_review
        publish_status = "approved" if (current_user.is_superuser and is_proprietary) else "pending_review"

        listing = StrategyListing(
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
        # Attach access control fields to the listing
        listing.is_proprietary = is_proprietary
        listing.status = publish_status
        listing.custom_strategy_id = request.custom_strategy_id

        strategy_id = await marketplace.publish_strategy_with_backtest(db, listing)

        # Send email notification to admin for pending submissions
        if publish_status == "pending_review":
            try:
                from ...alerts.email_provider import EmailProvider
                from ...config import settings

                if settings.EMAIL_ENABLED and settings.ADMIN_EMAIL:
                    email_provider = EmailProvider(
                        smtp_host=settings.SMTP_SERVER,
                        smtp_port=settings.SMTP_PORT,
                        username=settings.SMTP_USERNAME,
                        password=settings.SMTP_PASSWORD,
                        from_email=settings.FROM_EMAIL,
                        from_name="Oraculum Platform",
                    )
                    await email_provider.send_email(
                        to_email=settings.ADMIN_EMAIL,
                        subject=f"New Strategy Submission: {request.name}",
                        body=(
                            f"A new strategy has been submitted for review.\n\n"
                            f"Name: {request.name}\n"
                            f"Creator: {current_user.username}\n"
                            f"Category: {request.category}\n"
                            f"Sharpe Ratio: {backtest_results.sharpe_ratio:.2f}\n"
                            f"Total Return: {backtest_results.total_return:.2f}%\n\n"
                            f"Please review in the Admin Dashboard."
                        ),
                    )
            except Exception as email_err:
                logger.warning(f"Failed to send submission notification email: {email_err}")

        response_status = "published" if publish_status == "approved" else "pending_review"
        message = (
            "Strategy published to marketplace!"
            if publish_status == "approved"
            else "Strategy submitted for review. You'll be notified when approved."
        )

        # Log activity for the social feed
        try:
            activity_service = ActivityService(db)
            activity_type = "STRATEGY_PUBLISHED" if publish_status == "approved" else "STRATEGY_SUBMITTED"
            activity_content = (
                f"published '{request.name}' to the marketplace" if publish_status == "approved" else f"submitted '{request.name}' for review"
            )
            await activity_service.log_activity(
                user_id=current_user.id,
                activity_type=activity_type,
                content=activity_content,
                metadata={
                    "strategy_id": strategy_id,
                    "strategy_name": request.name,
                    "category": request.category,
                    "sharpe_ratio": round(backtest_results.sharpe_ratio, 2),
                    "total_return": round(backtest_results.total_return, 2),
                },
            )
        except Exception as act_err:
            logger.warning(f"Failed to log publish activity: {act_err}")

        return {"id": strategy_id, "status": response_status, "message": message}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to publish strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))
