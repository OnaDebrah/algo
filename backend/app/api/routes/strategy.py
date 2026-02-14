"""
Strategy routes
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.deps import check_permission, get_current_active_user, get_db
from backend.app.core.permissions import Permission
from backend.app.models.user import User
from backend.app.schemas.strategy import StrategyInfo, StrategyParameter
from backend.app.services.auth_service import AuthService
from backend.app.strategies.strategy_catalog import get_catalog

router = APIRouter(prefix="/strategy", tags=["Strategy"])


@router.get("/list", response_model=List[StrategyInfo])
async def list_strategies(
    mode: Optional[str] = Query(None, description="Filter by backtest mode: 'single' or 'multi'"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """List all available strategies, optionally filtered by backtest mode"""
    # Track usage (informational)
    await AuthService.track_usage(db, current_user.id, "list_strategies")
    catalog = get_catalog()
    strategies = []

    # Filter by mode if specified
    if mode and mode in ("single", "multi"):
        strategy_items = catalog.get_by_mode(mode).items()
    else:
        strategy_items = catalog.strategies.items()

    for key, info in strategy_items:
        # Get parameters from catalog info (already defined in strategy_catalog.py)
        parameters = []
        for param_name, param_info in info.parameters.items():
            param_range = param_info.get("range")
            param_type = param_info.get("type", "number")
            param_min = None
            param_max = None
            param_options = None

            # Determine parameter type and constraints from range
            if isinstance(param_range, (list, tuple)):
                # Check if it's a list of strings (select/enum type)
                if param_range and isinstance(param_range[0], str):
                    param_type = "select"
                    param_options = list(param_range)
                # Check if it's a list of booleans (boolean select)
                elif param_range and isinstance(param_range[0], bool):
                    param_type = "boolean"
                # Numeric range (tuple of two numbers)
                elif len(param_range) == 2 and all(isinstance(v, (int, float)) for v in param_range):
                    param_min = param_range[0]
                    param_max = param_range[1]
            elif param_range is None:
                # No range - check default type to infer param_type
                default_val = param_info.get("default")
                if isinstance(default_val, str):
                    param_type = "string"
                elif isinstance(default_val, bool):
                    param_type = "boolean"
                elif isinstance(default_val, list):
                    param_type = "string"  # Lists serialized as strings for UI

            # Allow explicit min/max overrides
            param_min = param_info.get("min") if param_info.get("min") is not None else param_min
            param_max = param_info.get("max") if param_info.get("max") is not None else param_max

            parameters.append(
                StrategyParameter(
                    name=param_name,
                    type=param_type,
                    default=param_info.get("default"),
                    min=param_min,
                    max=param_max,
                    description=param_info.get("description", ""),
                    options=param_options,
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
                backtest_mode=info.backtest_mode,
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
        param_range = param_info.get("range")
        param_type = param_info.get("type", "number")
        param_min = None
        param_max = None
        param_options = None

        if isinstance(param_range, (list, tuple)):
            if param_range and isinstance(param_range[0], str):
                param_type = "select"
                param_options = list(param_range)
            elif param_range and isinstance(param_range[0], bool):
                param_type = "boolean"
            elif len(param_range) == 2 and all(isinstance(v, (int, float)) for v in param_range):
                param_min = param_range[0]
                param_max = param_range[1]
        elif param_range is None:
            default_val = param_info.get("default")
            if isinstance(default_val, str):
                param_type = "string"
            elif isinstance(default_val, bool):
                param_type = "boolean"
            elif isinstance(default_val, list):
                param_type = "string"

        param_min = param_info.get("min") if param_info.get("min") is not None else param_min
        param_max = param_info.get("max") if param_info.get("max") is not None else param_max

        parameters.append(
            StrategyParameter(
                name=param_name,
                type=param_type,
                default=param_info.get("default"),
                min=param_min,
                max=param_max,
                description=param_info.get("description", ""),
                options=param_options,
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
        backtest_mode=info.backtest_mode,
    )
