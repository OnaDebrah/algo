import asyncio

from backend.app.api.routes.optimise import bayesian_optimization
from backend.app.database import AsyncSessionLocal
from backend.app.models.user import User
from backend.app.schemas.optimise import BayesianOptimizationRequest, ParamRange


async def test_bayesian():
    # Mock user
    user = User(id=1, username="testuser", tier="PRO", is_active=True)

    # Request
    request = BayesianOptimizationRequest(
        ticker="AAPL",
        strategy_key="SMA_Crossover",
        param_ranges={"fast_window": ParamRange(min=5, max=20, type="int"), "slow_window": ParamRange(min=20, max=100, type="int")},
        n_trials=5,
        metric="sharpe_ratio",
    )

    async with AsyncSessionLocal() as db:
        try:
            print("Starting Bayesian Optimization test...")
            result = await bayesian_optimization(request, current_user=user, db=db)
            print("Optimization successful!")
            print(f"Best Params: {result.best_params}")
            print(f"Best Value: {result.best_value}")
            print(f"Total Trials: {len(result.trials)}")
        except Exception as e:
            print(f"Optimization failed: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_bayesian())
