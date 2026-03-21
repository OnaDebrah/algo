"""Strategy versioning and rollback service."""

import logging

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.marketplace import MarketplaceStrategy
from ..models.strategy_version import StrategyVersion

logger = logging.getLogger(__name__)


class VersioningService:

    @staticmethod
    async def create_version(
        db: AsyncSession,
        strategy_id: int,
        strategy_type: str,
        parameters: dict,
        performance: dict | None = None,
        description: str | None = None,
        user_id: int | None = None,
    ) -> StrategyVersion:
        """Create a new version snapshot."""
        # Get next version number
        result = await db.execute(
            select(func.coalesce(func.max(StrategyVersion.version_number), 0))
            .where(StrategyVersion.strategy_id == strategy_id, StrategyVersion.strategy_type == strategy_type)
        )
        next_num = (result.scalar() or 0) + 1
        label = f"{next_num}.0.0"

        version = StrategyVersion(
            strategy_id=strategy_id,
            strategy_type=strategy_type,
            version_number=next_num,
            version_label=label,
            parameters_snapshot=parameters,
            performance_snapshot=performance,
            change_description=description,
            created_by=user_id,
        )
        db.add(version)
        await db.commit()
        await db.refresh(version)
        return version

    @staticmethod
    async def get_history(
        db: AsyncSession,
        strategy_id: int,
        strategy_type: str = "marketplace",
    ) -> list[StrategyVersion]:
        result = await db.execute(
            select(StrategyVersion)
            .where(StrategyVersion.strategy_id == strategy_id, StrategyVersion.strategy_type == strategy_type)
            .order_by(desc(StrategyVersion.version_number))
        )
        return list(result.scalars().all())

    @staticmethod
    async def compare_versions(
        db: AsyncSession,
        v1_id: int,
        v2_id: int,
    ) -> dict:
        result = await db.execute(
            select(StrategyVersion).where(StrategyVersion.id.in_([v1_id, v2_id]))
        )
        versions = {v.id: v for v in result.scalars().all()}
        if len(versions) != 2:
            return {"error": "One or both versions not found"}

        v1, v2 = versions[v1_id], versions[v2_id]
        params1 = v1.parameters_snapshot or {}
        params2 = v2.parameters_snapshot or {}

        # Compute diffs
        all_keys = set(params1.keys()) | set(params2.keys())
        param_diffs = {}
        for key in all_keys:
            val1, val2 = params1.get(key), params2.get(key)
            if val1 != val2:
                param_diffs[key] = {"v1": val1, "v2": val2}

        return {
            "v1": {"id": v1.id, "version": v1.version_label, "created_at": v1.created_at.isoformat()},
            "v2": {"id": v2.id, "version": v2.version_label, "created_at": v2.created_at.isoformat()},
            "parameter_diffs": param_diffs,
            "performance_v1": v1.performance_snapshot,
            "performance_v2": v2.performance_snapshot,
        }

    @staticmethod
    async def rollback(
        db: AsyncSession,
        strategy_id: int,
        version_id: int,
        strategy_type: str = "marketplace",
    ) -> dict | None:
        """Rollback strategy to a previous version's parameters."""
        result = await db.execute(
            select(StrategyVersion).where(
                StrategyVersion.id == version_id,
                StrategyVersion.strategy_id == strategy_id,
                StrategyVersion.strategy_type == strategy_type,
            )
        )
        version = result.scalar_one_or_none()
        if not version:
            return None

        if strategy_type == "marketplace":
            strat_result = await db.execute(
                select(MarketplaceStrategy).where(MarketplaceStrategy.id == strategy_id)
            )
            strategy = strat_result.scalar_one_or_none()
            if strategy:
                strategy.parameters = version.parameters_snapshot
                await db.commit()
                return {"restored_version": version.version_label, "parameters": version.parameters_snapshot}

        return None
