
import asyncio
import json
import httpx
from datetime import datetime

BASE_URL = "http://localhost:8000"
# Replace with a valid token if testing against a real server
TOKEN = "YOUR_TOKEN_HERE" 

async def test_versioning():
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        # 1. Get current strategies
        headers = {"Authorization": f"Bearer {TOKEN}"} if TOKEN != "YOUR_TOKEN_HERE" else {}
        
        try:
            resp = await client.get("/live/strategy", headers=headers)
            strategies = resp.json()
            if not strategies:
                print("No live strategies found to test with.")
                return
            
            strategy = strategies[0]
            strategy_id = strategy["id"]
            print(f"Testing with Strategy ID: {strategy_id} ({strategy['name']})")
            
            # 2. Update strategy parameters (triggers snapshot)
            update_data = {
                "parameters": {"test_param": 100},
                "notes": "Testing versioning system"
            }
            print("Updating strategy parameters...")
            resp = await client.patch(f"/live/strategy/{strategy_id}", json=update_data, headers=headers)
            print(f"Update response: {resp.status_code}")
            
            # 3. Check versions
            print("Fetching version history...")
            resp = await client.get(f"/live/strategy/{strategy_id}/versions", headers=headers)
            versions = resp.json()
            print(f"Found {len(versions)} versions.")
            for v in versions:
                print(f"  Version {v['version']} (ID: {v['id']}): {v['created_at']}")
            
            if versions:
                # 4. Rollback to previous version
                latest_snapshot_id = versions[0]["id"]
                print(f"Rolling back to Snapshot ID {latest_snapshot_id}...")
                resp = await client.post(f"/live/strategy/{strategy_id}/rollback/{latest_snapshot_id}", headers=headers)
                print(f"Rollback response: {resp.json()}")
                
        except Exception as e:
            print(f"Test failed: {e}")

if __name__ == "__main__":
    # This script is for manual verification if the server is running.
    # We won't run it here as we don't have a reliable way to get a token and have a running server in this environment.
    print("Verification script created. Run this locally with a valid TOKEN to test the full flow.")
