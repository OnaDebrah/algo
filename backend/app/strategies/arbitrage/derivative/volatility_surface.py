from typing import Dict, cast

import numpy as np
import pandas as pd
from pandas.core.arraylike import OpsMixin


class VolatilitySurface:
    """Models and manages volatility surfaces"""

    def __init__(self, min_moneyness: float = 0.8, max_moneyness: float = 1.2, min_dte: int = 7, max_dte: int = 180):
        self.min_moneyness = min_moneyness
        self.max_moneyness = max_moneyness
        self.min_dte = min_dte
        self.max_dte = max_dte
        self.surfaces = {}

    def build_surface(self, asset: str, option_chain: pd.DataFrame, spot: float) -> Dict:
        """Build volatility surface for an asset"""

        # Filter options
        filtered = self._filter_options(option_chain, spot)

        if filtered.empty:
            return {}

        surface = {
            "asset": asset,
            "spot": spot,
            "timestamp": pd.Timestamp.now(),
            "data": filtered,
            "parameters": self._fit_svi_surface(filtered),
            "term_structure": self._extract_term_structure(filtered),
            "skew": self._calculate_skew(filtered),
            "smile": self._calculate_smile(filtered),
        }

        self.surfaces[asset] = surface
        return surface

    def _filter_options(self, option_chain: pd.DataFrame, spot: float) -> pd.DataFrame:
        """Filter options by moneyness and DTE"""
        df = option_chain.copy()

        df["moneyness"] = df["strike"] / spot

        if "dte" not in df.columns and "expiry" in df.columns:
            df["dte"] = (pd.to_datetime(df["expiry"]) - pd.Timestamp.now(tz="UTC")).dt.days

        return cast(
            pd.DataFrame,
            cast(
                OpsMixin,
                df[
                    (df["moneyness"] >= self.min_moneyness)
                    & (df["moneyness"] <= self.max_moneyness)
                    & (df["dte"] >= self.min_dte)
                    & (df["dte"] <= self.max_dte)
                ],
            ),
        )

    def _fit_svi_surface(self, option_data: pd.DataFrame) -> Dict:
        """Fit SVI parameters"""
        surfaces = {}

        for expiry, group in option_data.groupby("dte"):
            try:
                moneyness = group["moneyness"].values
                iv = group["iv"].values

                m_adj = moneyness - 1
                coeffs = np.polyfit(m_adj, iv, 2)

                surfaces[expiry] = {
                    "a": coeffs[2],  # ATM vol
                    "b": coeffs[1],  # Skew
                    "c": coeffs[0],  # Smile curvature
                }
            except Exception:
                continue

        return surfaces

    def _extract_term_structure(self, option_data: pd.DataFrame) -> pd.Series:
        """Extract ATM volatility term structure"""
        atm_options = option_data[np.abs(option_data["moneyness"] - 1) < 0.05]
        return atm_options.groupby("dte")["iv"].mean()

    def _calculate_skew(self, option_data: pd.DataFrame) -> Dict:
        """Calculate volatility skew"""
        skew = {}

        for expiry, group in option_data.groupby("dte"):
            otm_puts = group[group["inTheMoney"] < 0.95]["iv"].mean()
            otm_calls = group[group["inTheMoney"] > 1.05]["v"].mean()
            atm = group[np.abs(group["inTheMoney"] - 1) < 0.05]["iv"].mean()

            if pd.isna(atm).empty:
                skew[expiry] = {
                    "put_skew": (otm_puts - atm) if not pd.isna(otm_puts) else 0,
                    "call_skew": (otm_calls - atm) if not pd.isna(otm_calls) else 0,
                }

        return skew

    def _calculate_smile(self, option_data: pd.DataFrame) -> Dict:
        """Calculate smile curvature"""
        smile = {}

        for expiry, group in option_data.groupby("dte"):
            if len(group) > 3:
                moneyness = group["inTheMoney"].values
                iv = group["iv"].values
                coeffs = np.polyfit(moneyness, iv, 2)
                smile[expiry] = coeffs[0]
            else:
                smile[expiry] = 0

        return smile

    def get_volatility(self, asset: str, strike: float, expiry: str, spot: float) -> float:
        """Get interpolated volatility for specific strike/expiry"""
        if asset not in self.surfaces:
            return 0.3  # Default

        surface = self.surfaces[asset]
        moneyness = strike / spot
        dte = (pd.to_datetime(expiry) - pd.Timestamp.now()).days

        # Find closest expiry
        expiries = list(surface["parameters"].keys())
        if not expiries:
            return 0.3

        # Simple nearest neighbor interpolation
        expiry_dtes = [(e, abs((pd.to_datetime(e) - pd.Timestamp.now()).days - dte)) for e in expiries]
        closest_expiry = min(expiry_dtes, key=lambda x: x[1])[0]

        params = surface["parameters"][closest_expiry]

        # Quadratic smile
        m_adj = moneyness - 1
        iv = params["a"] + params["b"] * m_adj + params["c"] * m_adj**2

        return max(iv, 0.05)  # Floor at 5%
