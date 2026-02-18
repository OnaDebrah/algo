import asyncio
import pandas as pd
from datetime import datetime
import sys
from unittest.mock import MagicMock

# Mock out heavy dependencies that might be missing
sys.modules['torch'] = MagicMock()
sys.modules['filterpy'] = MagicMock()
sys.modules['filterpy.common'] = MagicMock()

# Add project root
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Mock BacktestService to avoid complex imports
from backend.app.schemas.backtest import WFARequest, ParamRange

async def verify_wfa_logic():
    # We can't easily import WalkForwardService if it does 'from backend.app.services.backtest_service import BacktestService'
    # and that file imports everything.
    # So we'll test the _generate_folds logic specifically by extracting it if needed, or by mocking.
    
    # Let's try to import it now that we've mocked torch
    try:
        from backend.app.services.walk_forward_service import WalkForwardService
    except ImportError as e:
        print(f"Still failing to import: {e}")
        return

    service = WalkForwardService(user_id=1)
    
    # Mock data for 300 points
    dates = pd.date_range(start="2023-01-01", periods=300, freq='D')
    data = pd.DataFrame({
        "Open": [100.0] * 300,
        "High": [105.0] * 300,
        "Low": [95.0] * 300,
        "Close": [102.0] * 300,
        "Volume": [1000] * 300
    }, index=dates)

    request = WFARequest(
        symbol="MOCK",
        strategy_key="trend_following",
        param_ranges={
            "sma_period": ParamRange(min=10, max=50, type="int")
        },
        is_window_days=100,
        oos_window_days=30,
        step_days=30,
        anchored=False,
        period="1y"
    )

    print("\n[VERIFICATION] Testing _generate_folds...")
    folds = service._generate_folds(data, request)
    print(f"Total Folds Generated: {len(folds)}")
    
    expected_folds = (len(data) - 100) // 30
    print(f"Expected approx {expected_folds} folds based on (300-100)/30")

    for i, f in enumerate(folds):
        print(f"Fold {i}: IS {f['is_start'].date()} -> {f['is_end'].date()} | OOS {f['oos_start'].date()} -> {f['oos_end'].date()}")

    # Check for OOS continuity
    continuity_error = False
    for i in range(len(folds) - 1):
        if folds[i+1]['oos_start'] <= folds[i]['oos_end']:
            print(f"FAIL: Fold {i} and {i+1} OOS overlap!")
            continuity_error = True
    
    if not continuity_error:
        print("SUCCESS: OOS periods are sequential and non-overlapping.")

    print("\n[VERIFICATION] Walk-Forward logic verification complete.")

if __name__ == "__main__":
    asyncio.run(verify_wfa_logic())
