from typing import List
from fastapi import APIRouter, HTTPException, Body
from backend.app.schemas.live import LiveStatus, ExecutionOrder, ConnectRequest, EngineStatus, BrokerType, OrderSide, OrderType, OrderStatus

router = APIRouter(prefix="/live", tags=["Live Execution"])

# Mock state (in a real app, this would be in a service/database)
state = {
    "is_connected": False,
    "engine_status": EngineStatus.IDLE,
    "active_broker": BrokerType.PAPER,
    "orders": [
         { "id": "ORD-9921", "symbol": "AAPL", "side": OrderSide.BUY, "qty": 50, "type": OrderType.LIMIT, "status": OrderStatus.FILLED, "price": 182.45, "time": "14:02:11" },
         { "id": "ORD-9925", "symbol": "TSLA", "side": OrderSide.SELL, "qty": 10, "type": OrderType.MARKET, "status": OrderStatus.PENDING, "price": 234.10, "time": "14:05:45" },
    ]
}

@router.get("/status", response_model=LiveStatus)
async def get_status():
    return {
        "is_connected": state["is_connected"],
        "engine_status": state["engine_status"],
        "active_broker": state["active_broker"]
    }

@router.post("/connect")
async def connect_broker(request: ConnectRequest):
    # Simulate connection
    state["active_broker"] = request.broker
    state["is_connected"] = True
    return {"status": "connected", "broker": request.broker}

@router.post("/disconnect")
async def disconnect_broker():
    state["is_connected"] = False
    state["engine_status"] = EngineStatus.IDLE
    return {"status": "disconnected"}

@router.post("/engine/start")
async def start_engine():
    if not state["is_connected"]:
        raise HTTPException(status_code=400, detail="Broker not connected")
    state["engine_status"] = EngineStatus.RUNNING
    return {"status": "started"}

@router.post("/engine/stop")
async def stop_engine():
    state["engine_status"] = EngineStatus.IDLE
    return {"status": "stopped"}

@router.get("/orders", response_model=List[ExecutionOrder])
async def get_orders():
    return state["orders"]
