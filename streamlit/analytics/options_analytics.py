"""
Options Analytics Module
Advanced analytics for options trading strategies
"""

from typing import Dict, List

import numpy as np
import pandas as pd


class OptionsAnalytics:
    """Advanced analytics for options strategies"""

    @staticmethod
    def calculate_expected_return(
        current_price: float, strike: float, premium: float, option_type: str, position: str, probability_itm: float
    ) -> float:
        """
        Calculate expected return for an option position

        Args:
            current_price: Current underlying price
            strike: Option strike price
            premium: Option premium
            option_type: 'call' or 'put'
            position: 'long' or 'short'
            probability_itm: Probability of finishing in-the-money

        Returns:
            Expected return in dollars
        """
        if position == "long":
            # Long option
            if option_type == "call":
                expected_profit = probability_itm * max(current_price - strike, 0)
                return expected_profit - premium
            else:  # put
                expected_profit = probability_itm * max(strike - current_price, 0)
                return expected_profit - premium
        else:
            # Short option
            if option_type == "call":
                expected_loss = probability_itm * max(current_price - strike, 0)
                return premium - expected_loss
            else:  # put
                expected_loss = probability_itm * max(strike - current_price, 0)
                return premium - expected_loss

    @staticmethod
    def calculate_kelly_criterion(win_prob: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate optimal position size using Kelly Criterion

        Args:
            win_prob: Probability of winning trade
            avg_win: Average win amount
            avg_loss: Average loss amount

        Returns:
            Optimal fraction of capital to risk
        """
        if avg_loss == 0:
            return 0

        b = avg_win / abs(avg_loss)  # Win/loss ratio
        p = win_prob
        q = 1 - win_prob

        kelly = (b * p - q) / b

        # Apply Kelly half for safety
        return max(0, min(kelly / 2, 0.25))  # Cap at 25%

    @staticmethod
    def calculate_probability_itm(
        current_price: float, strike: float, days_to_expiration: int, volatility: float, risk_free_rate: float = 0.05, option_type: str = "call"
    ) -> float:
        """
        Calculate probability of option finishing in-the-money
        Uses log-normal distribution

        Args:
            current_price: Current stock price
            strike: Strike price
            days_to_expiration: Days until expiration
            volatility: Implied volatility (annualized)
            risk_free_rate: Risk-free rate
            option_type: 'call' or 'put'

        Returns:
            Probability (0 to 1)
        """
        from scipy.stats import norm

        T = days_to_expiration / 365.0

        if T <= 0:
            return 1.0 if ((option_type == "call" and current_price > strike) or (option_type == "put" and current_price < strike)) else 0.0

        # Calculate d2 from Black-Scholes
        d2 = (np.log(current_price / strike) + (risk_free_rate - 0.5 * volatility**2) * T) / (volatility * np.sqrt(T))

        if option_type == "call":
            return norm.cdf(d2)
        else:
            return norm.cdf(-d2)

    @staticmethod
    def calculate_probability_touch(current_price: float, barrier: float, days_to_expiration: int, volatility: float) -> float:
        """
        Calculate probability of price touching a barrier before expiration

        Args:
            current_price: Current price
            barrier: Price barrier
            days_to_expiration: Days until expiration
            volatility: Volatility

        Returns:
            Probability of touching barrier
        """
        T = days_to_expiration / 365.0

        if T <= 0:
            return 1.0 if current_price >= barrier else 0.0

        # Using barrier option formula
        h = np.log(barrier / current_price) / (volatility * np.sqrt(T))

        from scipy.stats import norm

        prob_touch = 2 * norm.cdf(-abs(h))

        return prob_touch

    @staticmethod
    def calculate_var(portfolio_value: float, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR)

        Args:
            portfolio_value: Current portfolio value
            returns: Historical returns array
            confidence_level: Confidence level (e.g., 0.95 for 95%)

        Returns:
            VaR in dollars
        """
        if len(returns) == 0:
            return 0.0

        var_percentile = np.percentile(returns, (1 - confidence_level) * 100)
        return abs(portfolio_value * var_percentile)

    @staticmethod
    def calculate_cvar(portfolio_value: float, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall

        Args:
            portfolio_value: Current portfolio value
            returns: Historical returns array
            confidence_level: Confidence level

        Returns:
            CVaR in dollars
        """
        if len(returns) == 0:
            return 0.0

        var_percentile = np.percentile(returns, (1 - confidence_level) * 100)
        cvar = np.mean(returns[returns <= var_percentile])

        return abs(portfolio_value * cvar)

    @staticmethod
    def calculate_options_portfolio_stats(closed_positions: List[Dict]) -> Dict:
        """
        Calculate comprehensive statistics for options portfolio

        Args:
            closed_positions: List of closed position dictionaries

        Returns:
            Dictionary of statistics
        """
        if not closed_positions:
            return {}

        df = pd.DataFrame(closed_positions)

        # Basic stats
        total_pnl = df["pnl"].sum()
        total_trades = len(df)
        winning_trades = len(df[df["pnl"] > 0])
        losing_trades = len(df[df["pnl"] <= 0])

        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # P&L stats
        avg_win = df[df["pnl"] > 0]["pnl"].mean() if winning_trades > 0 else 0
        avg_loss = df[df["pnl"] <= 0]["pnl"].mean() if losing_trades > 0 else 0

        largest_win = df["pnl"].max()
        largest_loss = df["pnl"].min()

        # Profit factor
        total_wins = df[df["pnl"] > 0]["pnl"].sum() if winning_trades > 0 else 0
        total_losses = abs(df[df["pnl"] <= 0]["pnl"].sum()) if losing_trades > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

        # Days held stats
        avg_days_held = df["days_held"].mean()
        max_days_held = df["days_held"].max()
        min_days_held = df["days_held"].min()

        # Return stats
        avg_return_pct = df["pnl_pct"].mean() * 100
        std_return_pct = df["pnl_pct"].std() * 100

        # Kelly criterion
        kelly_fraction = OptionsAnalytics.calculate_kelly_criterion(win_rate, avg_win, avg_loss)

        # Expectancy
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))

        return {
            "total_pnl": total_pnl,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate * 100,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "profit_factor": profit_factor,
            "avg_days_held": avg_days_held,
            "max_days_held": max_days_held,
            "min_days_held": min_days_held,
            "avg_return_pct": avg_return_pct,
            "std_return_pct": std_return_pct,
            "kelly_fraction": kelly_fraction,
            "expectancy": expectancy,
        }

    @staticmethod
    def calculate_rolling_metrics(equity_curve: List[Dict], window: int = 30) -> pd.DataFrame:
        """
        Calculate rolling performance metrics

        Args:
            equity_curve: List of equity curve data points
            window: Rolling window size

        Returns:
            DataFrame with rolling metrics
        """
        df = pd.DataFrame(equity_curve)

        if len(df) < window:
            return df

        # Calculate returns
        df["returns"] = df["equity"].pct_change()

        # Rolling metrics
        df["rolling_return"] = df["returns"].rolling(window).sum() * 100
        df["rolling_volatility"] = df["returns"].rolling(window).std() * np.sqrt(252) * 100
        df["rolling_sharpe"] = (df["rolling_return"] / df["rolling_volatility"]).fillna(0)

        # Rolling drawdown
        df["rolling_max"] = df["equity"].rolling(window, min_periods=1).max()
        df["rolling_dd"] = (df["equity"] - df["rolling_max"]) / df["rolling_max"] * 100

        return df

    @staticmethod
    def monte_carlo_simulation(current_price: float, volatility: float, days: int, num_simulations: int = 10000, drift: float = 0.0) -> np.ndarray:
        """
        Run Monte Carlo simulation for price paths

        Args:
            current_price: Starting price
            volatility: Annualized volatility
            days: Number of days to simulate
            num_simulations: Number of simulation paths
            drift: Expected daily drift

        Returns:
            Array of final prices (num_simulations,)
        """
        dt = 1 / 252  # Daily time step

        # Generate random price paths
        returns = np.random.normal(drift * dt, volatility * np.sqrt(dt), (num_simulations, days))

        # Calculate cumulative returns
        price_paths = current_price * np.exp(returns.cumsum(axis=1))

        # Return final prices
        return price_paths[:, -1]

    @staticmethod
    def calculate_optimal_strike_selection(
        current_price: float, volatility: float, days_to_expiration: int, strategy_type: str, num_strikes: int = 10
    ) -> List[Dict]:
        """
        Calculate optimal strikes for a strategy with risk/reward analysis

        Args:
            current_price: Current underlying price
            volatility: Implied volatility
            days_to_expiration: Days until expiration
            strategy_type: 'covered_call', 'cash_secured_put', etc.
            num_strikes: Number of strikes to analyze

        Returns:
            List of dictionaries with strike analysis
        """
        results = []

        # Generate strike range
        if strategy_type in ["covered_call", "call_spread"]:
            strikes = np.linspace(current_price * 1.01, current_price * 1.15, num_strikes)
        elif strategy_type in ["cash_secured_put", "put_spread"]:
            strikes = np.linspace(current_price * 0.85, current_price * 0.99, num_strikes)
        else:
            strikes = np.linspace(current_price * 0.90, current_price * 1.10, num_strikes)

        for strike in strikes:
            # Calculate probability ITM
            prob_itm = OptionsAnalytics.calculate_probability_itm(
                current_price, strike, days_to_expiration, volatility, option_type="call" if "call" in strategy_type else "put"
            )

            # Calculate expected premium (simplified)
            moneyness = strike / current_price
            time_value = volatility * np.sqrt(days_to_expiration / 365) * current_price

            intrinsic = max(current_price - strike, 0) if "call" in strategy_type else max(strike - current_price, 0)
            premium = intrinsic + time_value * (1 - abs(1 - moneyness))

            results.append(
                {
                    "strike": strike,
                    "moneyness": moneyness,
                    "premium_estimate": premium,
                    "prob_itm": prob_itm,
                    "prob_otm": 1 - prob_itm,
                    "expected_return": premium * (1 - prob_itm) if "short" in strategy_type else -premium + intrinsic,
                }
            )

        return results


def analyze_strategy_risk_reward(strategy_payoffs: np.ndarray, price_simulations: np.ndarray) -> Dict:
    """
    Analyze risk/reward profile of a strategy

    Args:
        strategy_payoffs: Array of payoffs at different prices
        price_simulations: Simulated final prices

    Returns:
        Dictionary with risk/reward metrics
    """
    # Calculate expected profit
    expected_profit = np.mean(strategy_payoffs)

    # Calculate probability of profit
    prob_profit = np.sum(strategy_payoffs > 0) / len(strategy_payoffs)

    # Calculate risk metrics
    var_95 = np.percentile(strategy_payoffs, 5)
    cvar_95 = np.mean(strategy_payoffs[strategy_payoffs <= var_95])

    # Max profit/loss
    max_profit = np.max(strategy_payoffs)
    max_loss = np.min(strategy_payoffs)

    # Return to risk ratio
    risk_reward_ratio = abs(expected_profit / cvar_95) if cvar_95 != 0 else 0

    return {
        "expected_profit": expected_profit,
        "prob_profit": prob_profit * 100,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "risk_reward_ratio": risk_reward_ratio,
        "profit_loss_ratio": abs(max_profit / max_loss) if max_loss != 0 else float("inf"),
    }
