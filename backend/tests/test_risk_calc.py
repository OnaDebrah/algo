import os
import sys

import numpy as np
import pandas as pd

# Add backend to path so imports work
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.app.analytics.performance import calculate_risk_metrics

try:
    # Build dummy equity curve mimicking multi-asset (daily changes)
    equity = []
    base_val = 100000
    dates = pd.date_range("2023-01-01", "2024-01-01", freq="B")

    # Simulate a steady +0.1% daily gain with some random noise
    np.random.seed(42)
    daily_returns = np.random.normal(0.0005, 0.01, len(dates))

    for i, d in enumerate(dates):
        if i > 0:
            base_val = base_val * (1 + daily_returns[i])
        equity.append({"timestamp": str(d), "equity": float(base_val)})

    trades = [{"profit": 100, "commission": 1}]

    res = calculate_risk_metrics(trades, equity)

    print("MOCK RISK METRICS:")
    for k, v in res.items():
        print(f"{k}: {v}")

except Exception:
    import traceback

    traceback.print_exc()
