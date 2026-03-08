"""
Strategy routes — catalog listing + custom strategy AI generation, validation, CRUD, and backtest
"""

import re
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from ...api.deps import check_permission, get_current_active_user, get_db
from ...config import settings
from ...core.custom_strategy_engine import SafeExecutionEnvironment, StrategyCodeGenerator
from ...core.permissions import Permission
from ...models.custom_strategy import CustomStrategy
from ...models.user import User
from ...schemas.custom_strategy import (
    CustomBacktestRequest,
    CustomStrategyCreate,
    CustomStrategyResponse,
    CustomStrategyUpdate,
    StrategyGenerateRequest,
    StrategyGenerateResponse,
    StrategyValidateRequest,
    StrategyValidateResponse,
)
from ...schemas.strategy import StrategyInfo, StrategyParameter
from ...services.auth_service import AuthService
from ...strategies.catelog.strategy_catalog import get_catalog
from ...utils.errors import safe_detail

router = APIRouter(prefix="/strategy", tags=["Strategy"])


# ── AI Code Generation ─────────────────────────────────────────


@router.post("/generate", response_model=StrategyGenerateResponse)
async def generate_strategy(
    request: StrategyGenerateRequest,
    current_user: User = Depends(check_permission(Permission.CUSTOM_STRATEGIES)),
    db: AsyncSession = Depends(get_db),
):
    """Generate strategy code from a natural language prompt using AI (Anthropic → DeepSeek → template)"""
    await AuthService.track_usage(db, current_user.id, "generate_strategy", {"prompt": request.prompt[:100]})

    generator = StrategyCodeGenerator(
        anthropic_api_key=settings.ANTHROPIC_API_KEY or None,
        deepseek_api_key=settings.DEEPSEEK_API_KEY or None,
    )

    try:
        code, explanation, example, provider = await generator.generate_strategy_code(request.prompt, request.style)
        return StrategyGenerateResponse(
            code=code,
            explanation=explanation,
            example_usage=example,
            provider=provider,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=safe_detail("Strategy generation failed", e),
        )


# ── Code Validation ────────────────────────────────────────────


@router.post("/validate", response_model=StrategyValidateResponse)
async def validate_strategy(
    request: StrategyValidateRequest,
    current_user: User = Depends(get_current_active_user),
):
    """Validate strategy code for syntax errors and forbidden operations"""
    env = SafeExecutionEnvironment()
    is_valid, error = env.validate_code(request.code)

    errors = [error] if not is_valid else []
    warnings: List[str] = []

    # Additional import checks
    if is_valid:
        imports = re.findall(r"import\s+(\w+)", request.code)
        allowed = {"pandas", "pd", "numpy", "np", "math", "datetime", "statistics"}
        for imp in imports:
            if imp not in allowed:
                warnings.append(f"Import '{imp}' may not be available in execution environment")

    return StrategyValidateResponse(is_valid=is_valid, errors=errors, warnings=warnings)


# ── Custom Strategy Backtest ───────────────────────────────────


@router.post("/backtest")
async def backtest_custom_strategy(
    request: CustomBacktestRequest,
    current_user: User = Depends(check_permission(Permission.CUSTOM_STRATEGIES)),
    db: AsyncSession = Depends(get_db),
):
    """Run a backtest using custom strategy code (dispatched to Celery worker)"""
    await AuthService.track_usage(db, current_user.id, "backtest_custom", {"symbol": request.symbol})

    # Validate code first
    env = SafeExecutionEnvironment()
    is_valid, error = env.validate_code(request.code)
    if not is_valid:
        raise HTTPException(status_code=400, detail=f"Invalid strategy code: {error}")

    # Create backtest run record
    from ...services.backtest_service import BacktestService

    service = BacktestService(db)
    backtest_run = await service.create_backtest_run(
        user_id=current_user.id,
        backtest_type="custom",
        symbols=[request.symbol],
        strategy_config={
            "custom_code": True,
            "strategy_name": request.strategy_name,
            "custom_strategy_id": request.custom_strategy_id,
        },
        period=request.period,
        interval=request.interval,
        initial_capital=request.initial_capital,
    )

    # Dispatch to Celery
    from ...tasks.backtest_tasks import run_custom_backtest_task

    task = run_custom_backtest_task.delay(backtest_run.id, request.model_dump(mode="json"), current_user.id)
    backtest_run.celery_task_id = task.id
    await db.commit()

    return {
        "backtest_id": backtest_run.id,
        "task_id": task.id,
        "status": "pending",
        "message": "Custom strategy backtest submitted",
    }


# ── Custom Strategy CRUD ───────────────────────────────────────


@router.post("/custom", response_model=CustomStrategyResponse, status_code=201)
async def create_custom_strategy(
    request: CustomStrategyCreate,
    current_user: User = Depends(check_permission(Permission.CUSTOM_STRATEGIES)),
    db: AsyncSession = Depends(get_db),
):
    """Save a custom strategy to the user's library"""
    # Auto-validate code
    env = SafeExecutionEnvironment()
    is_valid, _error = env.validate_code(request.code)

    strategy = CustomStrategy(
        user_id=current_user.id,
        name=request.name,
        description=request.description,
        code=request.code,
        strategy_type=request.strategy_type,
        parameters=request.parameters or {},
        is_validated=is_valid,
        ai_generated=request.ai_generated,
        ai_explanation=request.ai_explanation,
    )
    db.add(strategy)
    await db.commit()
    await db.refresh(strategy)

    return _strategy_to_response(strategy)


@router.get("/custom", response_model=List[CustomStrategyResponse])
async def list_custom_strategies(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """List all custom strategies for the current user"""
    result = await db.execute(select(CustomStrategy).filter(CustomStrategy.user_id == current_user.id).order_by(desc(CustomStrategy.updated_at)))
    strategies = result.scalars().all()
    return [_strategy_to_response(s) for s in strategies]


@router.get("/custom/{strategy_id}", response_model=CustomStrategyResponse)
async def get_custom_strategy(
    strategy_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a specific custom strategy"""
    strategy = await _get_owned_strategy(db, strategy_id, current_user.id)
    return _strategy_to_response(strategy)


@router.put("/custom/{strategy_id}", response_model=CustomStrategyResponse)
async def update_custom_strategy(
    strategy_id: int,
    request: CustomStrategyUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Update an existing custom strategy"""
    strategy = await _get_owned_strategy(db, strategy_id, current_user.id)

    if request.name is not None:
        strategy.name = request.name
    if request.description is not None:
        strategy.description = request.description
    if request.code is not None:
        strategy.code = request.code
        # Re-validate when code changes
        env = SafeExecutionEnvironment()
        is_valid, _error = env.validate_code(request.code)
        strategy.is_validated = is_valid
    if request.strategy_type is not None:
        strategy.strategy_type = request.strategy_type
    if request.parameters is not None:
        strategy.parameters = request.parameters

    await db.commit()
    await db.refresh(strategy)
    return _strategy_to_response(strategy)


@router.delete("/custom/{strategy_id}", status_code=204)
async def delete_custom_strategy(
    strategy_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a custom strategy"""
    strategy = await _get_owned_strategy(db, strategy_id, current_user.id)
    await db.delete(strategy)
    await db.commit()


# ── Helpers ────────────────────────────────────────────────────


async def _get_owned_strategy(db: AsyncSession, strategy_id: int, user_id: int) -> CustomStrategy:
    """Fetch a custom strategy ensuring the requesting user owns it."""
    result = await db.execute(
        select(CustomStrategy).filter(
            CustomStrategy.id == strategy_id,
            CustomStrategy.user_id == user_id,
        )
    )
    strategy = result.scalar_one_or_none()
    if not strategy:
        raise HTTPException(status_code=404, detail="Custom strategy not found")
    return strategy


def _strategy_to_response(s: CustomStrategy) -> CustomStrategyResponse:
    return CustomStrategyResponse(
        id=s.id,
        name=s.name,
        description=s.description,
        code=s.code,
        strategy_type=s.strategy_type,
        parameters=s.parameters,
        is_validated=s.is_validated,
        ai_generated=s.ai_generated,
        ai_explanation=s.ai_explanation,
        created_at=s.created_at.isoformat() if s.created_at else "",
        updated_at=s.updated_at.isoformat() if s.updated_at else "",
    )


# ── Catalog Routes (must come AFTER /custom, /generate, etc.) ──


@router.get("/list", response_model=List[StrategyInfo])
async def list_strategies(
    mode: Optional[str] = Query(None, description="Filter by backtest mode: 'single' or 'multi'"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """List all available strategies, optionally filtered by backtest mode"""
    await AuthService.track_usage(db, current_user.id, "list_strategies")
    catalog = get_catalog()
    strategies = []

    if mode and mode in ("single", "multi"):
        strategy_items = catalog.get_by_mode(mode).items()
    else:
        strategy_items = catalog.strategies.items()

    for key, info in strategy_items:
        parameters = _build_params(info)
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
    current_user: User = Depends(check_permission(Permission.BASIC_BACKTEST)),
    db: AsyncSession = Depends(get_db),
):
    """Get strategy details"""
    await AuthService.track_usage(db, current_user.id, "get_strategy", {"strategy_key": strategy_key})
    catalog = get_catalog()

    if strategy_key not in catalog.strategies:
        raise HTTPException(status_code=404, detail="Strategy not found")

    info = catalog.strategies[strategy_key]
    parameters = _build_params(info)

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


def _build_params(info) -> List[StrategyParameter]:
    """Build parameter list from catalog strategy info (DRY helper)."""
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
    return parameters
