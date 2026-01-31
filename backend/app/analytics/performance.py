"""
Performance analytics and metrics calculation
"""

from typing import Dict, List

import numpy as np
import pandas as pd


def calculate_performance_metrics(trades: List[Dict], equity_curve: List[Dict], initial_capital: float) -> Dict:
    """
    Calculate comprehensive performance metrics

    Args:
        trades: List of trade dictionaries
        equity_curve: List of equity snapshots
        initial_capital: Starting capital

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

    # Sharpe ratio
    returns = completed_trades["profit_pct"].values
    if len(returns) > 1 and np.std(returns) != 0:
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
    else:
        sharpe_ratio = 0

    # Maximum drawdown
    max_dd = calculate_max_drawdown(equity_curve)

    # Profit factor
    total_wins = winning_trades["profit"].sum() if len(winning_trades) > 0 else 0
    total_losses = abs(losing_trades["profit"].sum()) if len(losing_trades) > 0 else 1
    profit_factor = total_wins / total_losses if total_losses != 0 else 0

    return {
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
    }


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
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else gross_profits

        # Win Rate
        win_rate = (len(closed_trades[closed_trades["profit"] > 0]) / len(closed_trades)) * 100

        # Expectancy: (Win% * Avg Win) - (Loss% * Avg Loss)
        avg_win = closed_trades[closed_trades["profit"] > 0]["profit"].mean() or 0
        avg_loss = abs(closed_trades[closed_trades["profit"] < 0]["profit"].mean()) or 0
        expectancy = (win_rate / 100 * avg_win) - ((1 - win_rate / 100) * avg_loss)
    else:
        profit_factor, win_rate, expectancy = 0, 0, 0

    return {
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


def calculate_returns(trades: List[Dict]) -> Dict:
    """
    Calculates daily, monthly, and cumulative returns from trade data.
    """
    if not trades:
        return {"daily_returns": [], "monthly_returns": [], "cumulative_returns": []}

    df = pd.DataFrame(trades)
    df["executed_at"] = pd.to_datetime(df["executed_at"])
    df = df.sort_values("executed_at")

    # 2. Calculate Daily Returns
    daily_pnl = df.groupby(df["executed_at"].dt.date)["profit"].sum().reset_index()
    daily_pnl.columns = ["date", "pnl"]

    # Calculate daily percentage returns
    # Note: We use the cumulative total value to estimate the base for each day
    df["cumulative_profit"] = df["profit"].cumsum()
    # If initial capital isn't in this table, we derive it or assume a base
    # For a more accurate pct, we'd need the equity curve,
    # but here we'll provide the daily PnL series
    daily_returns = [{"date": str(row["date"]), "return": float(row["pnl"])} for _, row in daily_pnl.iterrows()]

    # 3. Calculate Cumulative Returns
    # Starting from 0, showing the growth of the account over time
    df["cum_return"] = df["profit"].cumsum()
    cumulative_returns = [{"date": row["executed_at"].isoformat(), "value": float(row["cum_return"])} for _, row in df.iterrows()]

    # 4. Calculate Monthly Returns (Heatmap Data)
    # Pivot the data into Year/Month format
    df["year"] = df["executed_at"].dt.year
    df["month"] = df["executed_at"].dt.month

    monthly_pnl = df.groupby(["year", "month"])["profit"].sum().reset_index()

    monthly_returns = [{"year": int(row["year"]), "month": int(row["month"]), "return": float(row["profit"])} for _, row in monthly_pnl.iterrows()]

    return {"daily_returns": daily_returns, "monthly_returns": monthly_returns, "cumulative_returns": cumulative_returns}


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
    Calculate maximum drawdown percentage

    Args:
        equity_curve: List of equity snapshots

    Returns:
        Maximum drawdown as percentage
    """
    equity_values = [e["equity"] for e in equity_curve]
    peak = equity_values[0]
    max_dd = 0

    for value in equity_values:
        if value > peak:
            peak = value
        dd = ((peak - value) / peak) * 100
        if dd > max_dd:
            max_dd = dd

    return max_dd


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

    return np.mean(excess_returns) / downside_std * np.sqrt(252)


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
