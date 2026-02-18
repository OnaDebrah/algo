import asyncio
import os
import sys

import pandas as pd

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.app.schemas.backtest import ParamRange, WFARequest
from backend.app.services.walk_forward_service import WalkForwardService


async def verify_wfa():
    service = WalkForwardService(user_id=1)

    # Mock data for 300 points
    dates = pd.date_range(start="2023-01-01", periods=300, freq="D")
    data = pd.DataFrame(
        {"Open": [100.0] * 300, "High": [105.0] * 300, "Low": [95.0] * 300, "Close": [102.0] * 300, "Volume": [1000] * 300}, index=dates
    )

    request = WFARequest(
        symbol="MOCK",
        strategy_key="trend_following",
        param_ranges={"sma_period": ParamRange(min=10, max=50, type="int")},
        is_window_days=100,
        oos_window_days=30,
        step_days=30,
        anchored=False,
        period="1y",
    )

    print("Generating folds...")
    folds = service._generate_folds(data, request)
    print(f"Total Folds: {len(folds)}")

    for i, f in enumerate(folds):
        print(f"Fold {i}: IS {f['is_start'].date()} to {f['is_end'].date()} | OOS {f['oos_start'].date()} to {f['oos_end'].date()}")

    # Verify dates do not overlap incorrectly
    for i in range(len(folds) - 1):
        if folds[i + 1]["oos_start"] <= folds[i]["oos_end"]:
            print(f"Overlap detected between Fold {i} and {i+1} OOS!")
        else:
            print(f"Fold {i} -> {i+1} OOS sequence OK.")

    # Test a dummy optimization sampling
    best_params, metrics = await service._optimize_in_sample(request, folds[0]["is_start"], folds[0]["is_end"], 100000)
    print(f"Best Params from IS: {best_params}")
    print(f"IS Performance: {metrics.total_return_pct:.2f}%")


if __name__ == "__main__":
    asyncio.run(verify_wfa())
