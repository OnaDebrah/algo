"""
Strategy routes
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.deps import check_permission, get_current_active_user, get_db
from backend.app.core.permissions import Permission
from backend.app.models.user import User
from backend.app.schemas.strategy import StrategyInfo, StrategyParameter
from backend.app.services.auth_service import AuthService
from backend.app.strategies.strategy_catalog import get_catalog

router = APIRouter(prefix="/strategy", tags=["Strategy"])


@router.get("/list", response_model=List[StrategyInfo])
async def list_strategies(current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)):
    """List all available strategies"""
    # Track usage (informational)
    await AuthService.track_usage(db, current_user.id, "list_strategies")
    catalog = get_catalog()
    strategies = []

    for key, info in catalog.strategies.items():
        # Get parameters from catalog info (already defined in strategy_catalog.py)
        parameters = []
        for param_name, param_info in info.parameters.items():
            parameters.append(
                StrategyParameter(
                    name=param_name,
                    type=param_info.get("type", "number"),
                    default=param_info.get("default"),
                    min=param_info.get("min")
                    if param_info.get("min") is not None
                    else (param_info.get("range")[0] if param_info.get("range") else None),
                    max=param_info.get("max")
                    if param_info.get("max") is not None
                    else (param_info.get("range")[1] if param_info.get("range") else None),
                    description=param_info.get("description", ""),
                )
            )

        strategies.append(
            StrategyInfo(
                key=key,
                name=info.name,
                description=info.description,
                category=info.category.value if hasattr(info.category, "value") else str(info.category),
                complexity=info.complexity,
                time_horizon=info.time_horizon,
                best_for=info.best_for,
                parameters=parameters,
            )
        )

    return strategies


@router.get("/{strategy_key}", response_model=StrategyInfo)
async def get_strategy(
    strategy_key: str,
    current_user: User = Depends(check_permission(Permission.BASIC_BACKTEST)),  # Basic strategies require BASIC tier
    db: AsyncSession = Depends(get_db),
):
    """Get strategy details"""
    # Track usage
    await AuthService.track_usage(db, current_user.id, "get_strategy", {"strategy_key": strategy_key})
    catalog = get_catalog()

    if strategy_key not in catalog.strategies:
        raise HTTPException(status_code=404, detail="Strategy not found")

    info = catalog.strategies[strategy_key]

    # Get parameters from catalog info
    parameters = []
    for param_name, param_info in info.parameters.items():
        parameters.append(
            StrategyParameter(
                name=param_name,
                type=param_info.get("type", "number"),
                default=param_info.get("default"),
                min=param_info.get("min") if param_info.get("min") is not None else (param_info.get("range")[0] if param_info.get("range") else None),
                max=param_info.get("max") if param_info.get("max") is not None else (param_info.get("range")[1] if param_info.get("range") else None),
                description=param_info.get("description", ""),
            )
        )

    return StrategyInfo(
        key=strategy_key,
        name=info.name,
        description=info.description,
        category=info.category.value if hasattr(info.category, "value") else str(info.category),
        complexity=info.complexity,
        time_horizon=info.time_horizon,
        best_for=info.best_for,
        parameters=parameters,
    )
