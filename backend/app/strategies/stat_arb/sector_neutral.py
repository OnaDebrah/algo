from typing import Dict

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from strategies.stat_arb.base_stat_arb import StatisticalArbitrageStrategy

# ============================================================================
# SPECIALIZED STATARB VARIATIONS
# ============================================================================


class SectorNeutralStrategy(StatisticalArbitrageStrategy):
    """
    Sector-Neutral Statistical Arbitrage
    Ensures baskets are sector-neutral
    """

    def __init__(self, sector_mapping: Dict[str, str], **kwargs):
        """
        Args:
            sector_mapping: Dictionary mapping assets to sectors
        """
        super().__init__(**kwargs)
        self.sector_mapping = sector_mapping

    def _construct_cointegration_basket(self, prices: pd.DataFrame, **kwargs):
        """Override to ensure sector neutrality"""
        # Group assets by sector
        sectors = {}
        for asset in self.universe:
            sector = self.sector_mapping.get(asset, "Unknown")
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(asset)

        # Construct basket with equal sector representation
        selected_assets = []

        for sector, sector_assets in sectors.items():
            if len(sector_assets) >= 2:
                # Select one asset from each sector
                selected = np.random.choice(sector_assets, size=1)[0]
                selected_assets.append(selected)

        if len(selected_assets) >= self.basket_size:
            selected_assets = selected_assets[: self.basket_size]

            # Get prices for selected assets
            basket_prices = prices[selected_assets].dropna()

            if len(basket_prices) < self.lookback_period // 2:
                return None

            # Perform cointegration test
            try:
                result = coint_johansen(basket_prices, det_order=0, k_ar_diff=1)
                if result.lr1[0] > result.cvt[0, 1]:
                    eigenvector = result.evec[:, 0]
                    eigenvector = eigenvector / eigenvector[0]
                    eigenvector = eigenvector - np.mean(eigenvector)
                    eigenvector = self._normalize_weights(eigenvector)

                    return selected_assets, eigenvector
            except Exception:
                pass

        return None
