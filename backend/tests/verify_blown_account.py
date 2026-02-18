import asyncio
import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from backend.app.schemas.backtest import ParamRange, WFARequest
from backend.app.services.walk_forward_service import WalkForwardService

# Mock heavy dependencies that might be missing
sys.modules["torch"] = MagicMock()
sys.modules["filterpy"] = MagicMock()
sys.modules["filterpy.common"] = MagicMock()
sys.modules["optuna"] = MagicMock()


async def verify_blown_account():
    print("\n[VERIFICATION] Testing WFA with potential blown account...")

    # Mock some data
    dates = pd.date_range(start="2020-01-01", periods=500, freq="D")
    data = pd.DataFrame(
        {
            "Open": np.random.uniform(100, 110, size=500),
            "High": np.random.uniform(110, 120, size=500),
            "Low": np.random.uniform(90, 100, size=500),
            "Close": np.linspace(100, 10, num=500),  # Steady decline to trigger loss
            "Volume": np.random.uniform(1000, 5000, size=500),
        },
        index=dates,
    )

    service = WalkForwardService()

    # Create a request that will likely blow the account
    request = WFARequest(
        symbol="DUMMY",
        strategy_key="sma_crossover",
        param_ranges={"short_window": ParamRange(min=5, max=10, type="int"), "long_window": ParamRange(min=20, max=50, type="int")},
        initial_capital=5000,  # Low capital to blow it easily
        period="max",
        interval="1d",
        is_window_days=100,
        oos_window_days=50,
        step_days=50,
        n_trials=2,
    )

    try:
        # Patch fetch_stock_data to return our dummy data
        import backend.app.services.walk_forward_service as wfs

        original_fetch = wfs.fetch_stock_data
        wfs.fetch_stock_data = lambda *args, **kwargs: data

        result = await service.run_wfa(request)

        print(f"SUCCESS: WFA completed with {len(result.folds)} folds.")
        print(f"Final WFE: {result.wfe:.4f}")

        wfs.fetch_stock_data = original_fetch
    except Exception as e:
        print(f"FAIL: WFA crashed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(verify_blown_account())
