import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.analytics.options_analytics import OptionsAnalytics
from backend.app.api.deps import get_current_active_user
from backend.app.core.options_engine import OptionsBacktestEngine, backtest_options_strategy
from backend.app.database import get_db
from backend.app.models import User
from backend.app.schemas.options import (
    ChainRequest, ChainResponse, BacktestRequest, BacktestResult,
    StrategyAnalysisRequest, StrategyAnalysisResponse,
    GreeksRequest, GreeksResponse,
    StrategyComparisonRequest, StrategyComparisonResponse,
    PayoffPoint,
    ProbabilityRequest, ProbabilityResponse,
    StrikeOptimizerRequest, StrikeOptimizerResponse, StrikeAnalysis,
    RiskMetricsRequest, RiskMetricsResponse,
    PortfolioStatsRequest, PortfolioStatsResponse,
    MonteCarloRequest, MonteCarloResponse
)
from backend.app.services.auth_service import AuthService
from backend.app.services.market_service import get_market_service
from backend.app.strategies.options_builder import OptionsStrategy
from backend.app.strategies.options_strategies import OptionsChain

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/options", tags=["Options"])

market_service = get_market_service()
# Initialize options engine (singleton pattern)
options_engine = OptionsBacktestEngine()


def fetch_real_option_chain(symbol: str) -> ChainResponse:
    """Fetch real option chain data from yfinance"""
    try:
        ticker = yf.Ticker(symbol)

        # Get current price
        hist = ticker.history(period="1d")
        if hist.empty:
            raise ValueError(f"No price data available for {symbol}")

        current_price = hist['Close'].iloc[-1]

        # Get expiration dates
        expirations = ticker.options
        if not expirations:
            raise ValueError(f"No options data available for {symbol}")

        # Get option chain for first few expirations
        expiration_dates = list(expirations[:5])  # First 5 expirations

        # Get calls and puts for the first expiration
        opt_chain = ticker.option_chain(expirations[0])

        # Convert calls to dict format
        calls = []
        for _, row in opt_chain.calls.iterrows():
            calls.append({
                "strike": float(row['strike']),
                "bid": float(row.get('bid', 0)),
                "ask": float(row.get('ask', 0)),
                "volume": int(row.get('volume', 0)) if pd.notna(row.get('volume')) else 0,
                "openInterest": int(row.get('openInterest', 0)) if pd.notna(row.get('openInterest')) else 0,
                "impliedVolatility": float(row.get('impliedVolatility', 0.3)),
                "delta": float(row.get('delta', 0.5)) if 'delta' in row else 0.5,
                "gamma": float(row.get('gamma', 0.05)) if 'gamma' in row else 0.05,
                "theta": float(row.get('theta', -0.1)) if 'theta' in row else -0.1,
                "vega": float(row.get('vega', 0.2)) if 'vega' in row else 0.2
            })

        # Convert puts to dict format
        puts = []
        for _, row in opt_chain.puts.iterrows():
            puts.append({
                "strike": float(row['strike']),
                "bid": float(row.get('bid', 0)),
                "ask": float(row.get('ask', 0)),
                "volume": int(row.get('volume', 0)) if pd.notna(row.get('volume')) else 0,
                "openInterest": int(row.get('openInterest', 0)) if pd.notna(row.get('openInterest')) else 0,
                "impliedVolatility": float(row.get('impliedVolatility', 0.3)),
                "delta": float(row.get('delta', -0.5)) if 'delta' in row else -0.5,
                "gamma": float(row.get('gamma', 0.05)) if 'gamma' in row else 0.05,
                "theta": float(row.get('theta', -0.1)) if 'theta' in row else -0.1,
                "vega": float(row.get('vega', 0.2)) if 'vega' in row else 0.2
            })

        return ChainResponse(
            symbol=symbol,
            underlying_price=float(current_price),
            expiration_dates=expiration_dates,
            calls=calls,
            puts=puts
        )

    except Exception as e:
        # Fallback to mock data if real data fetch fails
        print(f"Warning: Failed to fetch real option chain for {symbol}: {e}")
        return _generate_mock_chain(symbol, 150.0)


def _generate_mock_chain(symbol: str, price: float) -> ChainResponse:
    """Generate mock option chain data as fallback"""
    import random
    expiration_dates = [
        (datetime.now() + timedelta(days=d)).strftime("%Y-%m-%d")
        for d in [7, 14, 30, 60, 90]
    ]

    calls = []
    puts = []
    strikes = [price * (1 + i * 0.01) for i in range(-10, 11)]

    for k in strikes:
        calls.append({
            "strike": k,
            "bid": random.uniform(0.1, 10.0),
            "ask": random.uniform(0.2, 11.0),
            "volume": random.randint(100, 10000),
            "openInterest": random.randint(500, 50000),
            "impliedVolatility": 0.3 + random.uniform(-0.1, 0.1),
            "delta": 0.5 + (price - k) / price,
            "gamma": 0.05,
            "theta": -0.1,
            "vega": 0.2
        })
        puts.append({
            "strike": k,
            "bid": random.uniform(0.1, 10.0),
            "ask": random.uniform(0.2, 11.0),
            "volume": random.randint(100, 10000),
            "openInterest": random.randint(500, 50000),
            "impliedVolatility": 0.3 + random.uniform(-0.1, 0.1),
            "delta": -0.5 + (k - price) / price,
            "gamma": 0.05,
            "theta": -0.1,
            "vega": 0.2
        })

    return ChainResponse(
        symbol=symbol,
        underlying_price=price,
        expiration_dates=expiration_dates,
        calls=calls,
        puts=puts
    )


@router.post("/chain", response_model=ChainResponse)
async def get_option_chain(
        request: ChainRequest,
        current_user: User = Depends(get_current_active_user),
):
    try:
        chain = OptionsChain(request.symbol)

        # Get the raw data from yfinance
        data = chain.get_chain(expiration=request.expiration)

        # ✅ FIX: Properly structure the response
        import pandas as pd
        import numpy as np

        def clean_value(val):
            """Clean NaN and Inf values"""
            if pd.isna(val) or np.isinf(val):
                return None
            return float(val) if isinstance(val, (np.float64, np.float32)) else val

        # If no expiration provided, return available dates only
        if request.expiration is None:
            # Get all available expiration dates
            ticker = yf.Ticker(request.symbol)
            expiration_dates = ticker.options if hasattr(ticker, 'options') else []

            return {
                "symbol": request.symbol,
                "current_price": clean_value(ticker.info.get('currentPrice', 0)),
                "expiration_dates": list(expiration_dates),
                "calls": [],
                "puts": []
            }

        # If expiration is provided, return full chain data
        if isinstance(data, dict) and 'calls' in data and 'puts' in data:
            calls_df = data['calls']
            puts_df = data['puts']

            # Convert DataFrames to list of dicts
            calls_list = []
            for _, row in calls_df.iterrows():
                calls_list.append({
                    "strike": clean_value(row.get('strike', 0)),
                    "lastPrice": clean_value(row.get('lastPrice', 0)),
                    "bid": clean_value(row.get('bid', 0)),
                    "ask": clean_value(row.get('ask', 0)),
                    "volume": int(row.get('volume', 0)) if pd.notna(row.get('volume')) else 0,
                    "openInterest": int(row.get('openInterest', 0)) if pd.notna(row.get('openInterest')) else 0,
                    "impliedVolatility": clean_value(row.get('impliedVolatility', 0)),
                    "delta": clean_value(row.get('delta', 0)),  # May need calculation
                    "gamma": clean_value(row.get('gamma', 0)),
                    "theta": clean_value(row.get('theta', 0)),
                    "vega": clean_value(row.get('vega', 0)),
                    "inTheMoney": bool(row.get('inTheMoney', False))
                })

            puts_list = []
            for _, row in puts_df.iterrows():
                puts_list.append({
                    "strike": clean_value(row.get('strike', 0)),
                    "lastPrice": clean_value(row.get('lastPrice', 0)),
                    "bid": clean_value(row.get('bid', 0)),
                    "ask": clean_value(row.get('ask', 0)),
                    "volume": int(row.get('volume', 0)) if pd.notna(row.get('volume')) else 0,
                    "openInterest": int(row.get('openInterest', 0)) if pd.notna(row.get('openInterest')) else 0,
                    "impliedVolatility": clean_value(row.get('impliedVolatility', 0)),
                    "delta": clean_value(row.get('delta', 0)),
                    "gamma": clean_value(row.get('gamma', 0)),
                    "theta": clean_value(row.get('theta', 0)),
                    "vega": clean_value(row.get('vega', 0)),
                    "inTheMoney": bool(row.get('inTheMoney', False))
                })

            # Get current price and available dates
            ticker = yf.Ticker(request.symbol)
            current_price = clean_value(ticker.info.get('currentPrice', 0))
            expiration_dates = list(ticker.options) if hasattr(ticker, 'options') else [request.expiration]

            return {
                "symbol": request.symbol,
                "current_price": current_price,
                "expiration_dates": expiration_dates,
                "calls": calls_list,
                "puts": puts_list
            }

        # Fallback
        raise HTTPException(status_code=500, detail="Invalid data format returned from options chain")

    except Exception as e:
        import traceback
        print(f"❌ Option chain error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch option chain for {request.symbol}: {str(e)}"
        )
@router.post("/backtest", response_model=BacktestResult)
async def run_backtest(
    request: BacktestRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Run options strategy backtest using historical data
    """
    try:
        # Try to fetch real historical data
        try:
            ticker = yf.Ticker(request.symbol)
            data = ticker.history(start=request.start_date, end=request.end_date)

            if data.empty:
                raise ValueError("No historical data available")

            # Ensure we have OHLC columns
            if not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
                raise ValueError("Missing required OHLC columns")

        except Exception as e:
            print(f"⚠️ Failed to fetch real data for {request.symbol}: {e}")
            # Fallback to mock data
            dates = pd.date_range(start=request.start_date, end=request.end_date, freq='D')
            prices = 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.02)
            highs = prices * (1 + np.random.rand(len(dates)) * 0.01)
            lows = prices * (1 - np.random.rand(len(dates)) * 0.01)
            opens = prices * (1 + np.random.randn(len(dates)) * 0.005)

            data = pd.DataFrame({
                "Open": opens,
                "High": highs,
                "Low": lows,
                "Close": prices
            }, index=dates)

        logger.info(f"📊 Running backtest for {request.symbol} from {request.start_date} to {request.end_date}")
        logger.info(f"📊 Data shape: {data.shape}, Strategy: {request.strategy_type}")

        # ✅ FIX 1: Convert strategy type string to enum
        try:
            # Handle both uppercase and lowercase, with/without underscores
            strategy_name = request.strategy_type.upper().replace('-', '_').replace(' ', '_')
            strategy_enum = OptionsStrategy[strategy_name]
        except KeyError:
            # Try direct value match
            strategy_enum = OptionsStrategy(request.strategy_type)

        # ✅ FIX 2: Run backtest using the core engine
        results = backtest_options_strategy(
            symbol=request.symbol,
            data=data,
            strategy_type=strategy_enum,
            entry_rules=request.entry_rules or {},
            exit_rules=request.exit_rules or {},
            initial_capital=request.initial_capital,
            risk_free_rate=request.risk_free_rate
        )

        logger.info(f"✅ Backtest complete: {results['total_trades']} trades, {results['win_rate']:.1f}% win rate")

        # ✅ FIX 3: Get engine from results
        engine = results.get("engine")
        if not engine:
            raise ValueError("Backtest engine not found in results")

        # ✅ FIX 4: Safely serialize equity curve
        equity_curve_data = []
        for e in engine.equity_curve:
            date_val = e["date"]
            # Handle both Timestamp and datetime objects
            if hasattr(date_val, 'isoformat'):
                date_str = date_val.isoformat()
            else:
                date_str = str(date_val)

            equity_curve_data.append({
                "date": date_str,
                "equity": float(e["equity"])
            })

        # ✅ FIX 5: Safely serialize trades
        trades_data = []
        for t in engine.trades:
            date_val = t["date"]
            if hasattr(date_val, 'isoformat'):
                date_str = date_val.isoformat()
            else:
                date_str = str(date_val)

            trades_data.append({
                "date": date_str,
                "type": t["type"],
                "price": float(t["price"]),
                "pnl": float(t.get("pnl", 0)),
                "strategy": t["strategy"]
            })

        # Transform results for API response
        return BacktestResult(
            total_trades=int(results["total_trades"]),
            winning_trades=int(results["winning_trades"]),
            losing_trades=int(results["losing_trades"]),
            win_rate=float(results["win_rate"]),
            total_return=float(results["total_return"]),
            max_drawdown=float(results["max_drawdown"]),
            sharpe_ratio=float(results["sharpe_ratio"]),
            profit_factor=float(results["profit_factor"]),
            total_profit=float(results["total_profit"]),
            total_loss=float(results["total_loss"]),
            equity_curve=equity_curve_data,
            trades=trades_data
        )

    except Exception as e:
        import traceback
        logger.info(f"❌ Backtest error: {str(e)}")
        logger.info(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Backtest failed for {request.symbol}: {str(e)}"
        )

# ============================================================
# ENHANCED ENDPOINTS
# ============================================================

@router.post("/analyze", response_model=StrategyAnalysisResponse)
async def analyze_strategy(
        request: StrategyAnalysisRequest,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    Comprehensive strategy analysis including Greeks, payoff diagram, and probability of profit
    """
    await AuthService.track_usage(db, current_user.id, "analyze_strategy", {"symbol": request.symbol})
    try:
        from backend.app.strategies import OptionsStrategyBuilder, OptionType

        # Get current price
        ticker = yf.Ticker(request.symbol)
        hist = ticker.history(period="1d")
        if hist.empty:
            raise HTTPException(status_code=400, detail=f"No price data for {request.symbol}")

        current_price = float(hist['Close'].iloc[-1])

        # Build strategy
        builder = OptionsStrategyBuilder(request.symbol)
        builder.current_price = current_price

        # Add legs
        for leg in request.legs:
            option_type = OptionType.CALL if leg.option_type.upper() == 'CALL' else OptionType.PUT
            expiry = datetime.fromisoformat(leg.expiration)

            builder.add_leg(
                option_type=option_type,
                strike=leg.strike,
                expiry=expiry,
                quantity=leg.quantity,
                premium=leg.premium,
                volatility=request.volatility
            )

        # Calculate metrics
        greeks = builder.calculate_greeks(volatility=request.volatility)
        breakevens = builder.get_breakeven_points()
        max_profit, max_profit_cond = builder.get_max_profit()
        max_loss, max_loss_cond = builder.get_max_loss()
        prob_profit = builder.calculate_probability_of_profit(volatility=request.volatility)

        # Generate payoff diagram
        price_range = np.linspace(current_price * 0.7, current_price * 1.3, 100)
        payoffs = builder.calculate_payoff(price_range)
        payoff_diagram = [
            PayoffPoint(price=float(p), payoff=float(pf))
            for p, pf in zip(price_range, payoffs)
        ]

        return StrategyAnalysisResponse(
            symbol=request.symbol,
            current_price=current_price,
            initial_cost=builder.get_initial_cost(),
            greeks=GreeksResponse(**greeks),
            breakeven_points=breakevens,
            max_profit=max_profit,
            max_profit_condition=max_profit_cond,
            max_loss=max_loss,
            max_loss_condition=max_loss_cond,
            probability_of_profit=prob_profit,
            payoff_diagram=payoff_diagram
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Strategy analysis failed: {str(e)}"
        )


@router.post("/greeks", response_model=GreeksResponse)
async def calculate_greeks(
        request: GreeksRequest,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    Calculate real-time Greeks for an options position
    """
    await AuthService.track_usage(db, current_user.id, "calculate_greeks", {"symbol": request.symbol})
    try:
        from streamlit.strategies import OptionsStrategyBuilder, OptionType

        # Get current price
        ticker = yf.Ticker(request.symbol)
        hist = ticker.history(period="1d")
        if hist.empty:
            raise HTTPException(status_code=400, detail=f"No price data for {request.symbol}")

        current_price = float(hist['Close'].iloc[-1])

        # Build strategy
        builder = OptionsStrategyBuilder(request.symbol)
        builder.current_price = current_price

        # Add legs
        for leg in request.legs:
            option_type = OptionType.CALL if leg.option_type.upper() == 'CALL' else OptionType.PUT
            expiry = datetime.fromisoformat(leg.expiration)

            builder.add_leg(
                option_type=option_type,
                strike=leg.strike,
                expiry=expiry,
                quantity=leg.quantity,
                premium=leg.premium,
                volatility=request.volatility
            )

        # Calculate Greeks
        greeks = builder.calculate_greeks(volatility=request.volatility)

        return GreeksResponse(**greeks)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Greeks calculation failed: {str(e)}"
        )


@router.post("/compare", response_model=StrategyComparisonResponse)
async def compare_strategies(
        request: StrategyComparisonRequest,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    Compare multiple options strategies side-by-side
    """
    await AuthService.track_usage(db, current_user.id, "compare_strategies", {"symbol": request.symbol})
    try:
        from backend.app.strategies.options_builder import OptionsStrategyBuilder, OptionType, create_preset_strategy

        # Get current price
        ticker = yf.Ticker(request.symbol)
        hist = ticker.history(period="1d")
        if hist.empty:
            raise HTTPException(status_code=400, detail=f"No price data for {request.symbol}")

        current_price = float(hist['Close'].iloc[-1])

        comparisons = []

        for strategy_config in request.strategies:
            strategy_name = strategy_config.get("name", "Custom")
            strategy_type = strategy_config.get("type")
            legs = strategy_config.get("legs", [])

            # Build strategy
            if strategy_type and hasattr(OptionsStrategy, strategy_type.upper()):
                # Preset strategy
                from backend.app.strategies.options_strategies import OptionsStrategy as OptStrat
                strategy_enum = OptStrat[strategy_type.upper()]
                expiration = datetime.fromisoformat(strategy_config.get("expiration",
                                                                        (datetime.now() + timedelta(
                                                                            days=30)).isoformat()))

                builder = create_preset_strategy(
                    strategy_enum,
                    request.symbol,
                    current_price,
                    expiration,
                    **strategy_config.get("params", {})
                )
            else:
                # Custom strategy from legs
                builder = OptionsStrategyBuilder(request.symbol)
                builder.current_price = current_price

                for leg in legs:
                    option_type = OptionType.CALL if leg["option_type"].upper() == 'CALL' else OptionType.PUT
                    expiry = datetime.fromisoformat(leg["expiration"])

                    builder.add_leg(
                        option_type=option_type,
                        strike=leg["strike"],
                        expiry=expiry,
                        quantity=leg["quantity"],
                        premium=leg.get("premium")
                    )

            # Calculate metrics
            greeks = builder.calculate_greeks()
            breakevens = builder.get_breakeven_points()
            max_profit, max_profit_cond = builder.get_max_profit()
            max_loss, max_loss_cond = builder.get_max_loss()
            prob_profit = builder.calculate_probability_of_profit()

            comparisons.append({
                "name": strategy_name,
                "initial_cost": builder.get_initial_cost(),
                "greeks": greeks,
                "breakeven_points": breakevens,
                "max_profit": max_profit,
                "max_profit_condition": max_profit_cond,
                "max_loss": max_loss,
                "max_loss_condition": max_loss_cond,
                "probability_of_profit": prob_profit
            })

        return StrategyComparisonResponse(
            symbol=request.symbol,
            current_price=current_price,
            comparisons=comparisons
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Strategy comparison failed: {str(e)}"
        )


# ============================================================
# OPTIONS ANALYTICS ENDPOINTS
# ============================================================


@router.post("/analytics/probability", response_model=ProbabilityResponse)
async def calculate_probability(
        request: ProbabilityRequest,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    Calculate probability metrics for an option
    """
    await AuthService.track_usage(db, current_user.id, "calculate_probability", {"type": request.option_type})

    try:
        # Calculate probability ITM
        prob_itm = OptionsAnalytics.calculate_probability_itm(
            current_price=request.current_price,
            strike=request.strike,
            days_to_expiration=request.days_to_expiration,
            volatility=request.volatility,
            risk_free_rate=request.risk_free_rate,
            option_type=request.option_type
        )

        # Calculate probability of touching strike
        prob_touch = OptionsAnalytics.calculate_probability_touch(
            current_price=request.current_price,
            barrier=request.strike,
            days_to_expiration=request.days_to_expiration,
            volatility=request.volatility
        )

        # Calculate expected returns
        # Estimate premium (simplified)
        moneyness = request.strike / request.current_price
        time_value = request.volatility * np.sqrt(request.days_to_expiration / 365) * request.current_price
        intrinsic = max(request.current_price - request.strike, 0) if request.option_type == "call" else max(
            request.strike - request.current_price, 0)
        premium = intrinsic + time_value * (1 - abs(1 - moneyness))

        expected_return_long = OptionsAnalytics.calculate_expected_return(
            current_price=request.current_price,
            strike=request.strike,
            premium=premium,
            option_type=request.option_type,
            position="long",
            probability_itm=prob_itm
        )

        expected_return_short = OptionsAnalytics.calculate_expected_return(
            current_price=request.current_price,
            strike=request.strike,
            premium=premium,
            option_type=request.option_type,
            position="short",
            probability_itm=prob_itm
        )

        return ProbabilityResponse(
            probability_itm=prob_itm,
            probability_otm=1 - prob_itm,
            probability_touch=prob_touch,
            expected_return_long=expected_return_long,
            expected_return_short=expected_return_short
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Probability calculation failed: {str(e)}"
        )


@router.post("/analytics/optimize-strike", response_model=StrikeOptimizerResponse)
async def optimize_strike(
        request: StrikeOptimizerRequest,
        current_user: User = Depends(get_current_active_user),
        # db: AsyncSession = Depends(get_db())
):
    """
    Optimize strike selection for a strategy
    """
    # await AuthService.track_usage(db, current_user.id, "optimize_strike", {"type": request.option_type})

    try:
        # Calculate optimal strikes
        strike_analysis = OptionsAnalytics.calculate_optimal_strike_selection(
            current_price=request.current_price,
            volatility=request.volatility,
            days_to_expiration=request.days_to_expiration,
            strategy_type=request.strategy_type,
            num_strikes=request.num_strikes
        )

        # Convert to response format
        strikes = [
            StrikeAnalysis(**analysis)
            for analysis in strike_analysis
        ]

        return StrikeOptimizerResponse(
            symbol=request.symbol,
            strategy_type=request.strategy_type,
            current_price=request.current_price,
            strikes=strikes
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Strike optimization failed: {str(e)}"
        )


@router.post("/analytics/risk-metrics", response_model=RiskMetricsResponse)
async def calculate_risk_metrics(
        request: RiskMetricsRequest,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    Calculate portfolio risk metrics (VaR, CVaR, Kelly Criterion)
    """
    await AuthService.track_usage(db, current_user.id, "calculate_risk_metrics")
    try:
        returns_array = np.array(request.returns)

        # Calculate VaR
        var = OptionsAnalytics.calculate_var(
            portfolio_value=request.portfolio_value,
            returns=returns_array,
            confidence_level=request.confidence_level
        )

        # Calculate CVaR
        cvar = OptionsAnalytics.calculate_cvar(
            portfolio_value=request.portfolio_value,
            returns=returns_array,
            confidence_level=request.confidence_level
        )

        # Calculate Kelly Criterion if we have win/loss data
        kelly_fraction = None
        if len(returns_array) > 0:
            wins = returns_array[returns_array > 0]
            losses = returns_array[returns_array < 0]

            if len(wins) > 0 and len(losses) > 0:
                win_prob = len(wins) / len(returns_array)
                avg_win = wins.mean()
                avg_loss = losses.mean()

                kelly_fraction = OptionsAnalytics.calculate_kelly_criterion(
                    win_prob=win_prob,
                    avg_win=avg_win,
                    avg_loss=avg_loss
                )

        # Generate recommendation
        risk_pct = (cvar / request.portfolio_value) * 100 if request.portfolio_value > 0 else 0

        if risk_pct > 20:
            recommendation = "High risk detected - consider reducing position sizes significantly"
        elif risk_pct > 10:
            recommendation = "Moderate risk - consider reducing position sizes"
        elif risk_pct > 5:
            recommendation = "Normal risk levels - maintain current approach"
        else:
            recommendation = "Low risk - room for increased position sizes if desired"

        return RiskMetricsResponse(
            var_95=var,
            cvar_95=cvar,
            kelly_fraction=kelly_fraction,
            recommendation=recommendation
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Risk metrics calculation failed: {str(e)}"
        )


@router.post("/analytics/portfolio-stats", response_model=PortfolioStatsResponse)
async def calculate_portfolio_stats(
        request: PortfolioStatsRequest,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    Calculate comprehensive portfolio statistics
    """
    await AuthService.track_usage(db, current_user.id, "calculate_portfolio_stats")
    try:
        # Convert positions to dict format
        positions = [pos.dict() for pos in request.positions]

        # Calculate stats
        stats = OptionsAnalytics.calculate_options_portfolio_stats(positions)

        return PortfolioStatsResponse(**stats)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Portfolio stats calculation failed: {str(e)}"
        )


@router.post("/analytics/monte-carlo", response_model=MonteCarloResponse)
async def run_monte_carlo(
        request: MonteCarloRequest,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    Run Monte Carlo simulation for price distribution
    """
    await AuthService.track_usage(db, current_user.id, "run_monte_carlo")
    try:
        # Run simulation
        final_prices = OptionsAnalytics.monte_carlo_simulation(
            current_price=request.current_price,
            volatility=request.volatility,
            days=request.days,
            num_simulations=request.num_simulations,
            drift=request.drift
        )

        # Calculate statistics
        mean_price = float(np.mean(final_prices))
        median_price = float(np.median(final_prices))
        std_price = float(np.std(final_prices))
        percentile_5 = float(np.percentile(final_prices, 5))
        percentile_95 = float(np.percentile(final_prices, 95))
        prob_above_current = float(np.sum(final_prices > request.current_price) / len(final_prices))

        # Sample simulations for visualization (max 100)
        sample_size = min(100, len(final_prices))
        sampled_prices = np.random.choice(final_prices, size=sample_size, replace=False).tolist()

        return MonteCarloResponse(
            mean_final_price=mean_price,
            median_final_price=median_price,
            std_final_price=std_price,
            percentile_5=percentile_5,
            percentile_95=percentile_95,
            probability_above_current=prob_above_current,
            simulated_prices=sampled_prices
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Monte Carlo simulation failed: {str(e)}"
        )
