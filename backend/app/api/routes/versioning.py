"""Strategy versioning and rollback routes."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from ...api.deps import get_current_active_user
from ...database import get_db
from ...models import User
from ...services.versioning_service import VersioningService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/strategies", tags=["Strategy Versioning"])


class CreateVersionRequest(BaseModel):
    strategy_type: str = "marketplace"
    parameters: dict
    performance: Optional[dict] = None
    description: Optional[str] = None


@router.get("/{strategy_id}/versions")
async def get_version_history(
    strategy_id: int,
    strategy_type: str = Query("marketplace"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Get version history for a strategy."""
    versions = await VersioningService.get_history(db, strategy_id, strategy_type)
    return [
        {
            "id": v.id,
            "version_number": v.version_number,
            "version_label": v.version_label,
            "parameters_snapshot": v.parameters_snapshot,
            "performance_snapshot": v.performance_snapshot,
            "change_description": v.change_description,
            "created_at": v.created_at.isoformat() if v.created_at else None,
        }
        for v in versions
    ]


@router.post("/{strategy_id}/versions")
async def create_version(
    strategy_id: int,
    request: CreateVersionRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Save current state as a new version."""
    version = await VersioningService.create_version(
        db, strategy_id,
        strategy_type=request.strategy_type,
        parameters=request.parameters,
        performance=request.performance,
        description=request.description,
        user_id=current_user.id,
    )
    return {
        "id": version.id,
        "version_label": version.version_label,
        "version_number": version.version_number,
        "message": f"Version {version.version_label} created",
    }


@router.get("/{strategy_id}/versions/{v1_id}/compare/{v2_id}")
async def compare_versions(
    strategy_id: int,
    v1_id: int,
    v2_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Compare two versions of a strategy."""
    result = await VersioningService.compare_versions(db, v1_id, v2_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.post("/{strategy_id}/versions/{version_id}/rollback")
async def rollback_version(
    strategy_id: int,
    version_id: int,
    strategy_type: str = Query("marketplace"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Rollback strategy parameters to a previous version."""
    result = await VersioningService.rollback(db, strategy_id, version_id, strategy_type)
    if not result:
        raise HTTPException(status_code=404, detail="Version not found")
    return result
