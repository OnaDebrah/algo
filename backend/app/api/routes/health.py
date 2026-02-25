from fastapi import APIRouter

from ...database import AsyncSessionLocal
from ...services.execution_manager import get_execution_manager

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check():
    manager = get_execution_manager(AsyncSessionLocal)

    return {"status": "ok", "active_strategies": manager.get_executor_count(), "executor_status": manager.get_executor_status()}
