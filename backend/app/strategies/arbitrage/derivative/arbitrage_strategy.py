from typing import Dict, List

from ... import BaseStrategy


class ArbitrageStrategy(BaseStrategy):
    """Base class for all arbitrage strategies"""

    def __init__(self, name: str, params: dict, universe: List[str]):
        self.universe = universe
        self.active_positions = {}
        self.position_history = []
        self.performance_metrics = {}
        super().__init__(name, params)

    def generate_signal(self, market_data: Dict) -> Dict[str, Dict]:
        """Generate trading signals - to be implemented by subclasses"""
        raise NotImplementedError

    def calculate_position_size(self, signal: Dict, capital: float) -> float:
        """Calculate position size based on risk parameters"""
        raise NotImplementedError

    def update_positions(self, market_data: Dict) -> Dict[str, Dict]:
        """Update open positions with new market data"""
        updates = {}
        for position_id, position in self.active_positions.items():
            if position["status"] == "open":
                position = self._update_single_position(position, market_data)
                updates[position_id] = position
        return updates

    def _update_single_position(self, position: Dict, market_data: Dict) -> Dict:
        """Update a single position - to be customized by subclasses"""
        return position
