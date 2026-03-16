from typing import Any, Dict


class PositionSizer:
    """
    Determines position sizes based on Monte Carlo simulation results
    """

    def __init__(self, risk_per_trade: float = 0.02):
        self.risk_per_trade = risk_per_trade

    def calculate_position_size(self, portfolio_value: float, simulation_stats: Dict[str, Any], current_price: float) -> float:
        """
        Calculate position size based on simulation results

        Uses Kelly Criterion adjusted for risk constraints
        """
        # Probability of profit from simulations
        win_prob = simulation_stats["prob_profit"]

        # Expected return
        expected_return = simulation_stats["expected_return"]

        # Risk (distance to lower bound)
        risk = abs((simulation_stats["lower_bound"] - current_price) / current_price)

        if risk == 0 or expected_return <= 0:
            return 0.0

        # Kelly Criterion: f = (p*b - q) / b
        # where p = win probability, q = loss probability, b = win/loss ratio
        win_loss_ratio = abs(expected_return / risk) if risk > 0 else 1
        kelly_fraction = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio

        # Apply safety factor (half Kelly)
        kelly_fraction = max(0, kelly_fraction * 0.5)

        # Cap at risk per trade limit
        position_fraction = min(kelly_fraction, int(self.risk_per_trade))

        # Calculate position size
        position_value = portfolio_value * position_fraction
        position_size = position_value / current_price

        return position_size
