import sys
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from backend.app.schemas.backtest import BacktestResult, ParamRange, WFARequest
from backend.app.services.walk_forward_service import WalkForwardService


# Robust mocks for torch to satisfy issubclass(cls, torch.Tensor)
class MockTensor:
    pass


mock_torch = MagicMock()
mock_torch.Tensor = MockTensor
sys.modules["torch"] = mock_torch

sys.modules["optuna"] = MagicMock()
sys.modules["filterpy"] = MagicMock()
sys.modules["filterpy.common"] = MagicMock()


class TestWFABlowout(unittest.IsolatedAsyncioTestCase):
    async def test_blown_account_stops_loop(self):
        # Mock dependencies
        with patch("backend.app.services.walk_forward_service.fetch_stock_data") as mock_fetch:
            # Create dummy data
            dates = pd.date_range(start="2020-01-01", periods=200, freq="D")
            mock_fetch.return_value = pd.DataFrame({"Close": [100] * 200}, index=dates)

            service = WalkForwardService()

            # Mock internal methods to simulate a blowout
            service._optimize_in_sample = MagicMock()
            service._optimize_in_sample.return_value = (
                {"p": 1},
                BacktestResult(
                    total_return=0,
                    total_return_pct=-10,
                    win_rate=0,
                    sharpe_ratio=0,
                    max_drawdown=0,
                    total_trades=0,
                    winning_trades=0,
                    losing_trades=0,
                    avg_profit=0,
                    avg_win=0,
                    avg_loss=0,
                    profit_factor=0,
                    final_equity=900,
                    initial_capital=1000,
                ),
            )

            service._run_out_of_sample = MagicMock()
            # First fold: equity goes to 500
            service._run_out_of_sample.side_effect = [
                (
                    BacktestResult(
                        total_return=0,
                        total_return_pct=-50,
                        win_rate=0,
                        sharpe_ratio=0,
                        max_drawdown=0,
                        total_trades=0,
                        winning_trades=0,
                        losing_trades=0,
                        avg_profit=0,
                        avg_win=0,
                        avg_loss=0,
                        profit_factor=0,
                        final_equity=500,
                        initial_capital=1000,
                    ),
                    [],
                    [{"timestamp": "2020-01-01", "equity": 500, "cash": 500}],
                ),
                # Second fold should not be called
                Exception("Should not be called"),
            ]

            request = WFARequest(
                symbol="TEST",
                strategy_key="test",
                param_ranges={"p": ParamRange(min=1, max=2)},
                initial_capital=1000,
                is_window_days=50,
                oos_window_days=50,
                step_days=50,
            )

            # This should complete without crashing and with only 1 fold
            result = await service.run_wfa(request)

            self.assertEqual(len(result.folds), 1)
            print(f"SUCCESS: WFA stopped early as expected. Folds: {len(result.folds)}")


if __name__ == "__main__":
    unittest.main()
