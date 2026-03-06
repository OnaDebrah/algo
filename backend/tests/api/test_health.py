import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient):
    """Test the /health endpoint"""
    response = await client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "environment" in data
    assert data["environment"] == "test"


@pytest.mark.asyncio
async def test_metrics_endpoint(client: AsyncClient):
    """Test the Prometheus /metrics endpoint"""
    response = await client.get("/metrics")

    assert response.status_code == 200
    assert "python_info" in response.text
