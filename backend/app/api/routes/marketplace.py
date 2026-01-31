from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.app.api.deps import get_current_active_user
from backend.app.config import settings
from backend.app.core.marketplace import BacktestResults, StrategyListing as CoreListing, StrategyMarketplace
from backend.app.models import User
from backend.app.schemas.marketplace import (
    BacktestResultsSchema,
    StrategyListing as StrategyListingSchema,
    StrategyListingDetailed as StrategyListingDetailedSchema,
    StrategyPublishRequest,
)

router = APIRouter(prefix="/marketplace", tags=["Marketplace"])

# Initialize marketplace (singleton pattern)
marketplace = StrategyMarketplace(db_path=settings.DATABASE_PATH)


def convert_core_to_schema(core_listing) -> StrategyListingSchema:
    """Convert core StrategyListing to API schema"""
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
        monthly_return=round(core_listing.total_return / 12, 2) if core_listing.total_return else 0,
        drawdown=abs(core_listing.max_drawdown),
        sharpe_ratio=core_listing.sharpe_ratio,
        total_downloads=core_listing.downloads,
        tags=core_listing.tags if core_listing.tags else [],
        best_for=core_listing.tags[:3] if core_listing.tags else [],
        pros=_generate_pros(core_listing),
        cons=_generate_cons(core_listing),
        is_favorite=False,  # This should be checked against user favorites
        is_verified=core_listing.is_verified,
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
    return [convert_core_to_schema(listing) for listing in core_listings]


@router.get("/{strategy_id}", response_model=StrategyListingDetailedSchema)
async def get_strategy_details(strategy_id: int, current_user: User = Depends(get_current_active_user)):
    """Get detailed information about a specific strategy including backtest results"""

    core_listings = marketplace.browse_strategies(limit=100)
    target_listing = next((listing for listing in core_listings if listing.id == strategy_id), None)

    if not target_listing:
        raise HTTPException(status_code=404, detail="Strategy not found")

    backtest_data = marketplace.get_strategy_backtest(strategy_id)

    detailed_schema = StrategyListingDetailedSchema(**convert_core_to_schema(target_listing).dict())

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

    return detailed_schema


@router.post("/{strategy_id}/favorite")
async def favorite_strategy(strategy_id: int, current_user: User = Depends(get_current_active_user)):
    """Add a strategy to user's favorites"""
    try:
        import sqlite3

        conn = sqlite3.connect(settings.DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO strategy_favorites (strategy_id, user_id) VALUES (?, ?)", (strategy_id, current_user.id))
        conn.commit()
        conn.close()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{strategy_id}/favorite")
async def unfavorite_strategy(strategy_id: int, current_user: User = Depends(get_current_active_user)):
    """Remove a strategy from user's favorites"""
    try:
        import sqlite3

        conn = sqlite3.connect(settings.DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM strategy_favorites WHERE strategy_id = ? AND user_id = ?", (strategy_id, current_user.id))
        conn.commit()
        conn.close()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{strategy_id}/download")
async def record_download(strategy_id: int, current_user: User = Depends(get_current_active_user)):
    """Record a strategy download and increment download count"""
    try:
        import sqlite3

        conn = sqlite3.connect(settings.DATABASE_PATH)
        cursor = conn.cursor()

        # 1. Record download
        cursor.execute("INSERT INTO strategy_downloads (strategy_id, user_id) VALUES (?, ?)", (strategy_id, current_user.id))

        # 2. Increment download count in main table
        cursor.execute("UPDATE marketplace_strategies SET downloads = downloads + 1 WHERE id = ?", (strategy_id,))

        conn.commit()
        conn.close()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/publish", status_code=status.HTTP_201_CREATED)
async def publish_strategy(request: StrategyPublishRequest, current_user: User = Depends(get_current_active_user)):
    """Publish a new strategy to the marketplace"""
    try:
        # 1. Fetch existing backtest results if backtest_id is provided
        # In a real app, we'd verify the backtest belongs to the user

        if request.backtest_id:
            # Mocking the retrieval of backtest data for now
            # In production, we'd fetch from the backtest_runs table
            pass

        # 2. Create core listing object
        listing = CoreListing(
            id=None,  # Will be set by DB
            name=request.name,
            description=request.description,
            creator_id=current_user.id,
            creator_name=current_user.full_name or "Unknown",
            strategy_type="Custom",
            category=request.category,
            complexity=request.complexity,
            parameters={},
            backtest_results=BacktestResults(),  # Should be populated from backtest_id
            price=request.price,
            is_public=request.is_public,
            tags=request.tags,
        )

        # 3. Publish to marketplace
        strategy_id = marketplace.publish_strategy_with_backtest(listing)

        return {"id": strategy_id, "status": "published"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
