"""
Strategy routes
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException

from backend.app.api.deps import get_current_active_user
from backend.app.models.user import User
from backend.app.schemas.strategy import StrategyInfo, StrategyParameter
from strategies.strategy_catalog import get_catalog

router = APIRouter(prefix="/strategy", tags=["Strategy"])


@router.get("/list", response_model=List[StrategyInfo])
async def list_strategies(current_user: User = Depends(get_current_active_user)):
    """List all available strategies"""
    catalog = get_catalog()
    strategies = []

    for key, info in catalog.strategies.items():
        strategy_class = info["class"]

        # Get parameters from strategy class
        parameters = []
        if hasattr(strategy_class, "get_parameters"):
            params = strategy_class.get_parameters()
            for param_name, param_info in params.items():
                parameters.append(
                    StrategyParameter(
                        name=param_name,
                        type=param_info.get("type", "number"),
                        default=param_info.get("default"),
                        min=param_info.get("min"),
                        max=param_info.get("max"),
                        description=param_info.get("description", ""),
                    )
                )

        strategies.append(
            StrategyInfo(
                key=key, name=info["name"], description=info.get("description", ""), category=info.get("category", "Technical"), parameters=parameters
            )
        )

    return strategies


@router.get("/{strategy_key}", response_model=StrategyInfo)
async def get_strategy(strategy_key: str, current_user: User = Depends(get_current_active_user)):
    """Get strategy details"""
    catalog = get_catalog()

    if strategy_key not in catalog.strategies:
        raise HTTPException(status_code=404, detail="Strategy not found")

    info = catalog.strategies[strategy_key]
    strategy_class = info["class"]

    # Get parameters
    parameters = []
    if hasattr(strategy_class, "get_parameters"):
        params = strategy_class.get_parameters()
        for param_name, param_info in params.items():
            parameters.append(
                StrategyParameter(
                    name=param_name,
                    type=param_info.get("type", "number"),
                    default=param_info.get("default"),
                    min=param_info.get("min"),
                    max=param_info.get("max"),
                    description=param_info.get("description", ""),
                )
            )

    return StrategyInfo(
        key=strategy_key,
        name=info["name"],
        description=info.get("description", ""),
        category=info.get("category", "Technical"),
        parameters=parameters,
    )
