"""
Performance analytics and metrics calculation
"""

import math
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats


def _sanitize_float(value, default=0.0, cap=999999.0):
    """
    Sanitize a numeric value to ensure JSON compliance.
    Replaces inf, -inf, NaN with default, and caps extreme values.
    """
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if math.isnan(value) or math.isinf(value):
            return default
        # Cap extreme values to prevent meaningless huge numbers
        if value > cap:
            return cap
        if value < -cap:
            return -cap
        return float(value)
    return value


def _sanitize_metrics(metrics: Dict) -> Dict:
    """
    Sanitize all numeric values in a metrics dictionary to ensure
    they are JSON-serializable (no inf, NaN, or extreme values).
    """
    sanitized = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            sanitized[key] = _sanitize_metrics(value)
        elif isinstance(value, list):
            sanitized[key] = value  # Don't modify lists (equity curves, etc.)
        else:
            sanitized[key] = _sanitize_float(value)
    return sanitized


def calculate_performance_metrics(trades: List[Dict], equity_curve: List[Dict], initial_capital: float, benchmark_equity: List[Dict] = None) -> Dict:
    """
    Calculate comprehensive performance metrics

    Args:
        trades: List of trade dictionaries
        equity_curve: List of equity snapshots
        initial_capital: Starting capital
        benchmark_equity: Benchmark equity

    Returns:
        Dictionary of performance metrics
    """
    if not trades or not equity_curve:
        return _get_empty_metrics()

    trades_df = pd.DataFrame(trades)
    completed_trades = trades_df[trades_df["profit"].notna()]

    if completed_trades.empty:
        return _get_empty_metrics(total_trades=len(trades))

    # Basic metrics
    final_equity = equity_curve[-1]["equity"]
    total_return = ((final_equity - initial_capital) / initial_capital) * 100

    # Win rate
    winning_trades = completed_trades[completed_trades["profit"] > 0]
    win_rate = (len(winning_trades) / len(completed_trades)) * 100

    # Profit metrics
    avg_profit = completed_trades["profit"].mean()
    total_profit = completed_trades["profit"].sum()
    avg_win = winning_trades["profit"].mean() if len(winning_trades) > 0 else 0

    losing_trades = completed_trades[completed_trades["profit"] < 0]
    avg_loss = losing_trades["profit"].mean() if len(losing_trades) > 0 else 0

    # Sharpe ratio - use equity curve daily returns (not per-trade profit_pct)
    # Per-trade profit_pct spans variable time periods, not daily,
    # so annualizing with sqrt(252) would be incorrect
    equity_values = pd.Series([e["equity"] for e in equity_curve])
    daily_returns = equity_values.pct_change().dropna()
    if len(daily_returns) > 1 and daily_returns.std() != 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0

    # Maximum drawdown
    max_dd = calculate_max_drawdown(equity_curve)

    # Profit factor
    total_wins = winning_trades["profit"].sum() if len(winning_trades) > 0 else 0
    total_losses = abs(losing_trades["profit"].sum()) if len(losing_trades) > 0 else 1
    profit_factor = total_wins / total_losses if total_losses != 0 else 0

    # Calculate advanced risk metrics
    risk_metrics = calculate_risk_metrics(trades, equity_curve)

    # Create equity series for factor calculations
    equity_df = pd.DataFrame(equity_curve)
    equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"])
    equity_df = equity_df.sort_values("timestamp")
    equity_df["returns"] = equity_df["equity"].pct_change().dropna()

    # Calculate monthly returns matrix
    monthly_matrix = calculate_monthly_returns_matrix(equity_df)

    # Beta, Alpha, R-Squared
    factors = {"alpha": 0.0, "beta": 0.0, "r_squared": 0.0}
    if benchmark_equity:
        factors = calculate_factor_metrics(equity_df, benchmark_equity)

    metrics = {
        "total_return": total_return,
        "total_return_pct": total_return,
        "win_rate": win_rate,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_dd,
        "total_trades": len(completed_trades),
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "avg_profit": avg_profit,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "total_profit": total_profit,
        "profit_factor": profit_factor,
        "final_equity": final_equity,
        "initial_capital": initial_capital,
        "monthly_returns_matrix": monthly_matrix,
        **factors,
    }

    metrics.update(risk_metrics)

    return _sanitize_metrics(metrics)


def calculate_risk_metrics(trades: List[Dict], equity_curve: List[Dict]) -> Dict:
    """
    Calculate risk metrics including VaR and CVaR using equity curve data.
    """

    df = pd.DataFrame(equity_curve)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    returns = df["equity"].pct_change().dropna()
    volatility = returns.std() * np.sqrt(252) if not returns.empty else 0

    # var_95 is the 5th percentile of returns
    var_95 = np.percentile(returns, 5) if not returns.empty else 0
    # cvar_95 is the average of returns that fall below the VaR threshold
    cvar_95 = returns[returns <= var_95].mean() if not returns.empty and len(returns[returns <= var_95]) > 0 else 0

    max_dd = calculate_max_drawdown(equity_curve)
    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)

    days = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).days
    years = max(days / 365.25, 0.1)  # Floor at 0.1 to prevent infinity
    total_return_pct = ((df["equity"].iloc[-1] - df["equity"].iloc[0]) / df["equity"].iloc[0]) * 100
    calmar = calculate_calmar_ratio(total_return_pct, max_dd, years)

    df_trades = pd.DataFrame(trades)

    closed_trades = df_trades[df_trades["profit"].notnull()]

    if not closed_trades.empty:
        gross_profits = closed_trades[closed_trades["profit"] > 0]["profit"].sum()
        gross_losses = abs(closed_trades[closed_trades["profit"] < 0]["profit"].sum())

        # Profit Factor: How many dollars earned for every dollar lost
        # When no losses, cap at 0 (no meaningful ratio) rather than returning raw gross_profits
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else 0.0

        # Win Rate
        win_rate = (len(closed_trades[closed_trades["profit"] > 0]) / len(closed_trades)) * 100

        # Expectancy: (Win% * Avg Win) - (Loss% * Avg Loss)
        winning = closed_trades[closed_trades["profit"] > 0]["profit"]
        losing = closed_trades[closed_trades["profit"] < 0]["profit"]
        avg_win = float(winning.mean()) if len(winning) > 0 else 0.0
        avg_loss = abs(float(losing.mean())) if len(losing) > 0 else 0.0
        expectancy = (win_rate / 100 * avg_win) - ((1 - win_rate / 100) * avg_loss)
    else:
        profit_factor, win_rate, expectancy = 0, 0, 0

    return _sanitize_metrics(
        {
            "volatility": float(volatility),
            "beta": 1.0,  # Placeholder for future benchmark comparison
            "var_95": float(var_95),
            "cvar_95": float(cvar_95),
            "max_drawdown": float(max_dd),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "calmar_ratio": float(calmar),
            "profit_factor": float(profit_factor),
            "win_rate": float(win_rate),
            "expectancy": float(expectancy),
            "total_trades": len(df_trades),
            "total_commission": float(df_trades["commission"].sum()) if "commission" in df_trades else 0,
        }
    )


def calculate_monthly_returns_matrix(equity_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Calculate a matrix of monthly percentage returns (Year -> Month -> Return%).
    """
    if equity_df.empty or "returns" not in equity_df.columns:
        return {}

    # Resample to monthly returns
    # We take the last equity of each month to calculate monthly pct change
    monthly_equity = equity_df.set_index("timestamp")["equity"].resample("M").last()
    monthly_returns = monthly_equity.pct_change().dropna()

    # Add the first month's return (from initial capital if possible, or just 0)
    # But pct_change() needs a previous value.
    # Let's use the first available price point as the base for the first month.

    matrix = {}
    for ts, ret in monthly_returns.items():
        year = str(ts.year)
        month = ts.strftime("%b")  # Jan, Feb, etc.
        if year not in matrix:
            matrix[year] = {}
        matrix[year][month] = float(ret * 100)

    return matrix


def calculate_factor_metrics(strategy_df: pd.DataFrame, benchmark_equity: List[Dict]) -> Dict:
    """
    Calculate Alpha, Beta, and R-Squared relative to a benchmark.
    """
    if not benchmark_equity or strategy_df.empty:
        return {"alpha": 0.0, "beta": 0.0, "r_squared": 0.0}

    bench_df = pd.DataFrame(benchmark_equity)
    bench_df["timestamp"] = pd.to_datetime(bench_df["timestamp"])
    bench_df = bench_df.sort_values("timestamp")
    bench_df["returns"] = bench_df["equity"].pct_change().dropna()

    # Align returns on timestamps
    combined = pd.merge(
        strategy_df[["timestamp", "returns"]], bench_df[["timestamp", "returns"]], on="timestamp", suffixes=("_strat", "_bench")
    ).dropna()

    if len(combined) < 5:
        return {"alpha": 0.0, "beta": 0.0, "r_squared": 0.0}

    # Linear Regression: R_strat = alpha + beta * R_bench
    slope, intercept, r_value, p_value, std_err = stats.linregress(combined["returns_bench"], combined["returns_strat"])

    # Annualize Alpha (daily returns -> annual)
    # Alpha is the intercept. Annual Alpha = (1 + intercept)^252 - 1
    # But usually, it's simplified as intercept * 252
    annual_alpha = intercept * 252

    return {
        "alpha": float(annual_alpha * 100),  # As percentage
        "beta": float(slope),
        "r_squared": float(r_value**2),
    }


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculate annualized Sharpe ratio.

    Args:
        returns: Series of daily returns
        risk_free_rate: Daily risk-free rate (usually 0.0)

    Returns:
        Annualized Sharpe ratio
    """
    excess_returns = returns - risk_free_rate

    std_dev = excess_returns.std()

    if std_dev == 0 or np.isnan(std_dev):
        return 0.0

    sharpe_ratio = (excess_returns.mean() / std_dev) * np.sqrt(252)

    return float(sharpe_ratio)


def calculate_max_drawdown(equity_curve: List[Dict]) -> float:
    """
    Calculate maximum drawdown percentage (vectorized)

    Args:
        equity_curve: List of equity snapshots

    Returns:
        Maximum drawdown as percentage
    """
    equity_values = np.array([e["equity"] for e in equity_curve])
    running_max = np.maximum.accumulate(equity_values)
    drawdowns = ((running_max - equity_values) / running_max) * 100
    return float(drawdowns.max()) if len(drawdowns) > 0 else 0


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sortino ratio (only penalizes downside volatility)

    Args:
        returns: Series of returns
        risk_free_rate: Risk-free rate

    Returns:
        Sortino ratio
    """
    excess_returns = returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return 0

    downside_std = np.std(downside_returns)

    if downside_std == 0:
        return 0

    result = np.mean(excess_returns) / downside_std * np.sqrt(252)
    return float(result) if np.isfinite(result) else 0.0


def calculate_calmar_ratio(total_return: float, max_drawdown: float, years: float = 1.0) -> float:
    """
    Calculate Calmar ratio (return / max drawdown)

    Args:
        total_return: Total return percentage
        max_drawdown: Maximum drawdown percentage
        years: Number of years

    Returns:
        Calmar ratio
    """
    if max_drawdown == 0:
        return 0

    annualized_return = total_return / years
    return annualized_return / max_drawdown


def _get_empty_metrics(total_trades: int = 0) -> Dict:
    """Return empty metrics dict"""
    return {
        "total_return": 0,
        "total_return_pct": 0,
        "win_rate": 0,
        "sharpe_ratio": 0,
        "max_drawdown": 0,
        "total_trades": total_trades,
        "winning_trades": 0,
        "losing_trades": 0,
        "avg_profit": 0,
        "avg_win": 0,
        "avg_loss": 0,
        "total_profit": 0,
        "profit_factor": 0,
        "final_equity": 0,
        "initial_capital": 0,
    }


def format_metrics_for_display(metrics: Dict) -> Dict:
    """
    Format metrics for display in UI

    Args:
        metrics: Raw metrics dictionary

    Returns:
        Formatted metrics dictionary
    """
    return {
        "Total Return": f"{metrics['total_return']:.2f}%",
        "Win Rate": f"{metrics['win_rate']:.1f}%",
        "Sharpe Ratio": f"{metrics['sharpe_ratio']:.2f}",
        "Max Drawdown": f"{metrics['max_drawdown']:.2f}%",
        "Total Trades": str(metrics["total_trades"]),
        "Winning Trades": str(metrics["winning_trades"]),
        "Average Profit": f"${metrics['avg_profit']:.2f}",
        "Profit Factor": f"{metrics['profit_factor']:.2f}",
        "Final Equity": f"${metrics['final_equity']:,.2f}",
    }
