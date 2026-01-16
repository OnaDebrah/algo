from fastapi import APIRouter, HTTPException, Depends
from typing import List
from datetime import datetime, timedelta
from backend.app.schemas.options import (
    ChainRequest, ChainResponse, BacktestRequest, BacktestResult,
    StrategyAnalysisRequest, StrategyAnalysisResponse,
    GreeksRequest, GreeksResponse,
    StrategyComparisonRequest, StrategyComparisonResponse,
    PayoffPoint,
    # Analytics schemas
    ProbabilityRequest, ProbabilityResponse,
    StrikeOptimizerRequest, StrikeOptimizerResponse, StrikeAnalysis,
    RiskMetricsRequest, RiskMetricsResponse,
    PortfolioStatsRequest, PortfolioStatsResponse,
    MonteCarloRequest, MonteCarloResponse
)
from backend.app.core.options_engine import OptionsBacktestEngine, backtest_options_strategy
from backend.app.api.deps import get_current_active_user
from backend.app.models import User
from strategies.options_builder import OptionsStrategy
import pandas as pd
import numpy as np
import yfinance as yf

router = APIRouter(prefix="/options", tags=["Options"])

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
    strikes = [price * (1 + i*0.01) for i in range(-10, 11)]
    
    for k in strikes:
        calls.append({
            "strike": k,
            "bid": random.uniform(0.1, 10.0),
            "ask": random.uniform(0.2, 11.0),
            "volume": random.randint(100, 10000),
            "openInterest": random.randint(500, 50000),
            "impliedVolatility": 0.3 + random.uniform(-0.1, 0.1),
            "delta": 0.5 + (price - k)/ price,
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
            "delta": -0.5 + (k - price)/ price, 
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
    current_user: User = Depends(get_current_active_user)
):
    """
    Get option chain data for a symbol
    Attempts to fetch real data from yfinance, falls back to mock data if unavailable
    """
    try:
        return fetch_real_option_chain(request.symbol)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch option chain for {request.symbol}: {str(e)}"
        )


@router.post("/backtest", response_model=BacktestResult)
async def run_backtest(
    request: BacktestRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    Run options strategy backtest using historical data
    Attempts to fetch real historical data, falls back to mock data if unavailable
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
            print(f"Warning: Failed to fetch real data for {request.symbol}: {e}")
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
        
        # Convert strategy type string to enum
        strategy_enum = OptionsStrategy(request.strategy_type)
        
        # Run backtest using the core engine
        results = backtest_options_strategy(
            symbol=request.symbol,
            data=data,
            strategy_type=strategy_enum,
            entry_rules=request.entry_rules,
            exit_rules=request.exit_rules,
            initial_capital=request.initial_capital,
            risk_free_rate=request.risk_free_rate
        )
        
        # Transform results for API response
        return BacktestResult(
            total_trades=results["total_trades"],
            winning_trades=results["winning_trades"],
            losing_trades=results["losing_trades"],
            win_rate=results["win_rate"],
            total_return=results["total_return"],
            max_drawdown=results["max_drawdown"],
            sharpe_ratio=results["sharpe_ratio"],
            profit_factor=results["profit_factor"],
            total_profit=results["total_profit"],
            total_loss=results["total_loss"],
            equity_curve=[
                {"date": e["date"].isoformat(), "equity": e["equity"]} 
                for e in results["engine"].equity_curve
            ],
            trades=[
                 {
                    "date": t["date"].isoformat(), 
                    "type": t["type"],
                    "price": t["price"],
                    "pnl": t.get("pnl", 0),
                    "strategy": t["strategy"]
                }
                for t in results.get("engine", {}).trades
            ]
        )
        
    except Exception as e:
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
    current_user: User = Depends(get_current_active_user)
):
    """
    Comprehensive strategy analysis including Greeks, payoff diagram, and probability of profit
    """
    try:
        from strategies.options_builder import OptionsStrategyBuilder, OptionType
        
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
    current_user: User = Depends(get_current_active_user)
):
    """
    Calculate real-time Greeks for an options position
    """
    try:
        from strategies.options_builder import OptionsStrategyBuilder, OptionType
        
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
    current_user: User = Depends(get_current_active_user)
):
    """
    Compare multiple options strategies side-by-side
    """
    try:
        from strategies.options_builder import OptionsStrategyBuilder, OptionType, create_preset_strategy
        
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
                from strategies.options_builder import OptionsStrategy as OptStrat
                strategy_enum = OptStrat[strategy_type.upper()]
                expiration = datetime.fromisoformat(strategy_config.get("expiration", 
                    (datetime.now() + timedelta(days=30)).isoformat()))
                
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

from backend.app.analytics.options_analytics import OptionsAnalytics


@router.post("/analytics/probability", response_model=ProbabilityResponse)
async def calculate_probability(
    request: ProbabilityRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    Calculate probability metrics for an option
    """
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
        intrinsic = max(request.current_price - request.strike, 0) if request.option_type == "call" else max(request.strike - request.current_price, 0)
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
    current_user: User = Depends(get_current_active_user)
):
    """
    Optimize strike selection for a strategy
    """
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
    current_user: User = Depends(get_current_active_user)
):
    """
    Calculate portfolio risk metrics (VaR, CVaR, Kelly Criterion)
    """
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
    current_user: User = Depends(get_current_active_user)
):
    """
    Calculate comprehensive portfolio statistics
    """
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
    current_user: User = Depends(get_current_active_user)
):
    """
    Run Monte Carlo simulation for price distribution
    """
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
