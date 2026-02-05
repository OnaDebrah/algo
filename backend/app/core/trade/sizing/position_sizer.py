class PositionSizer:
    """
    Base class for position sizing algorithms
    """

    def calculate_size(self, equity: float, price: float, **kwargs) -> int:
        """
        Calculate position size

        Args:
            equity: Current account equity
            price: Current price of asset
            **kwargs: Additional parameters

        Returns:
            Number of shares to buy
        """
        raise NotImplementedError
