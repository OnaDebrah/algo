from typing import Dict, List, Optional, cast

import numpy as np
import pandas as pd
from pandas.core.arraylike import OpsMixin

from ....config import DEFAULT_RISK_FREE_RATE
from ....core.options.pricers.engine import OptionsPricingEngine
from .greek_calculator import GreekCalculator
from .volatility_forecaster import VolatilityForecaster
from .volatility_surface import VolatilitySurface


class ArbitrageDetector:
    """Detects various types of options arbitrage"""

    def __init__(self, entry_threshold: float = 2.0, min_liquidity: float = 1.0):
        self.greek_calculator = GreekCalculator()
        self.entry_threshold = entry_threshold
        self.min_liquidity = min_liquidity

    def detect_all(
        self,
        asset: str,
        option_chain: pd.DataFrame,
        spot: float,
        vol_surface: VolatilitySurface,
        vol_forecaster: VolatilityForecaster,
        greek_calc: GreekCalculator,
    ) -> List[Dict]:
        """Detect all types of arbitrage"""

        opportunities = []

        liquid_options = self._filter_liquidity(option_chain)

        # Detect different arbitrage types
        opportunities.extend(self._volatility_arbitrage(asset, liquid_options, spot, vol_forecaster))
        opportunities.extend(self._put_call_parity(asset, liquid_options, spot))
        opportunities.extend(self._term_structure_arbitrage(asset, liquid_options, vol_surface))
        opportunities.extend(self._skew_arbitrage(asset, liquid_options, vol_surface))
        opportunities.extend(self._butterfly_arbitrage(asset, liquid_options, spot))
        opportunities.extend(self._box_spread_arbitrage(asset, liquid_options, spot))
        opportunities.extend(self._calendar_spread_arbitrage(asset, liquid_options, vol_surface, spot))

        return opportunities

    def _filter_liquidity(self, option_chain: pd.DataFrame) -> pd.DataFrame:
        """Filter options by liquidity"""
        if "openInterest" in option_chain.columns and "mid" in option_chain.columns:
            liquidity = option_chain["openInterest"] * option_chain["mid"] / 1e6
            return cast(pd.DataFrame, cast(OpsMixin, option_chain[liquidity >= self.min_liquidity]))
        return option_chain

    def _volatility_arbitrage(self, asset: str, options: pd.DataFrame, spot: float, vol_forecaster: VolatilityForecaster) -> List[Dict]:
        """Detect volatility arbitrage opportunities"""
        opportunities = []

        for _, option in options.iterrows():
            dte = option["dte"]
            forecast_vol = vol_forecaster.forecast(pd.Series(), dte)  # Need price series
            implied_vol = option["iv"]

            vol_spread = implied_vol - forecast_vol
            vol_zscore = vol_spread / (forecast_vol * 0.1)

            if abs(vol_zscore) > self.entry_threshold:
                opportunities.append(
                    {
                        "asset": asset,
                        "type": "volatility_arb",
                        "option_symbol": option["symbol"],
                        "option_type": option["option_type"],
                        "strike": option["strike"],
                        "expiry": option["dte"],
                        "dte": dte,
                        "implied_vol": implied_vol,
                        "forecast_vol": forecast_vol,
                        "vol_spread": vol_spread,
                        "vol_zscore": vol_zscore,
                        "direction": -1 if vol_zscore > 0 else 1,
                        "confidence": min(abs(vol_zscore) / self.entry_threshold, 3.0),
                    }
                )

        return opportunities

    def _put_call_parity(self, asset: str, options: pd.DataFrame, spot: float, r: float = 0.03) -> List[Dict]:
        """Detect put-call parity violations"""
        opportunities = []

        options["strike_expiry"] = options["strike"].astype(str) + "_" + options["dte"].astype(str)

        for key, group in options.groupby("strike_expiry"):
            calls = group[group["option_type"] == "call"]
            puts = group[group["option_type"] == "put"]

            if calls.empty or puts.empty:
                continue

            call = calls.iloc[0]
            put = puts.iloc[0]

            K = call["strike"]
            T = call["dte"] / 365
            discounted_K = K * np.exp(-r * T)

            actual_diff = call["mid"] - put["mid"]
            theoretical_diff = spot - discounted_K

            mispricing = actual_diff - theoretical_diff
            avg_price = (call["mid"] + put["mid"]) / 2

            if avg_price > 0:
                mispricing_pct = mispricing / avg_price

                if abs(mispricing_pct) > self.entry_threshold * 0.01:
                    opportunities.append(
                        {
                            "asset": asset,
                            "type": "put_call_parity",
                            "strike": K,
                            "expiry": call["expiry"],
                            "dte": call["dte"],
                            "call_symbol": call["symbol"],
                            "put_symbol": put["symbol"],
                            "call_price": call["mid"],
                            "put_price": put["mid"],
                            "actual_diff": actual_diff,
                            "theoretical_diff": theoretical_diff,
                            "mispricing": mispricing,
                            "mispricing_pct": mispricing_pct,
                            "direction": 1 if mispricing < 0 else -1,
                            "confidence": min(abs(mispricing_pct) / (self.entry_threshold * 0.01), 3.0),
                        }
                    )

        return opportunities

    def _term_structure_arbitrage(self, asset: str, options: pd.DataFrame, vol_surface: VolatilitySurface) -> List[Dict]:
        opportunities = []
        spot = options["spot"].iloc[0] if "spot" in options else 0  # Ensure you have spot

        # Group by moneyness to compare like-for-like strikes across time
        options["moneyness_bucket"] = pd.cut(
            options["strike"] / spot, bins=[0, 0.9, 0.95, 1.05, 1.1, np.inf], labels=["deep_otm_put", "otm_put", "atm", "otm_call", "deep_otm_call"]
        )

        for moneyness, group in options.groupby("moneyness_bucket"):
            if len(group) < 2:
                continue

            group = group.sort_values("dte")

            dtes = group["dte"].values
            vols = group["iv"].values
            strikes = group["strike"].values

            for i in range(len(dtes) - 1):
                T1, T2 = dtes[i] / 365, dtes[i + 1] / 365

                # USE VOL_SURFACE HERE: Get smoothed IVs instead of raw noisy ones
                iv1 = vol_surface.get_volatility(asset, strikes[i], dtes[i], spot)
                iv2 = vol_surface.get_volatility(asset, strikes[i + 1], dtes[i + 1], spot)

                var1, var2 = (iv1**2) * T1, (iv2**2) * T2

                # CALENDAR ARBITRAGE CHECK: Total variance must increase with time
                if var2 < var1:
                    # This is a "Pure" Arbitrage: Buy Far, Sell Near for guaranteed profit
                    forward_vol = 0
                    is_calendar_arb = True
                else:
                    forward_var = (var2 - var1) / (T2 - T1)
                    forward_vol = np.sqrt(forward_var)
                    is_calendar_arb = False

                vol_diff = forward_vol - iv2

                if is_calendar_arb or abs(vol_diff) > self.entry_threshold * 0.05:
                    opportunities.append(
                        {
                            "asset": asset,
                            "type": "term_structure_arb",
                            "moneyness": moneyness,
                            "near_expiry": group.iloc[i]["dte"],
                            "far_expiry": group.iloc[i + 1]["dte"],
                            "near_dte": dtes[i],
                            "far_dte": dtes[i + 1],
                            "near_vol": vols[i],
                            "far_vol": vols[i + 1],
                            "forward_vol": forward_vol,
                            "vol_diff": vol_diff,
                            "direction": 1 if vol_diff < 0 else -1,
                            "confidence": min(abs(vol_diff) / (self.entry_threshold * 0.05), 3.0),
                        }
                    )

        return opportunities

    def _skew_arbitrage(self, asset: str, options: pd.DataFrame, vol_surface: VolatilitySurface) -> List[Dict]:
        """Detect skew arbitrage opportunities"""
        opportunities = []

        skew_data = vol_surface._calculate_skew(options)

        for expiry, skew_values in skew_data.items():
            put_skew = skew_values["put_skew"]
            call_skew = skew_values["call_skew"]

            # Historical averages (would be dynamic in production)
            avg_put_skew = 0.02
            avg_call_skew = -0.01
            skew_std = 0.005

            put_skew_z = (put_skew - avg_put_skew) / skew_std
            call_skew_z = (call_skew - avg_call_skew) / skew_std

            expiry_options = options[options["expiry"] == expiry]

            if abs(put_skew_z) > self.entry_threshold:
                otm_puts = expiry_options[expiry_options["inTheMoney"] < 0.95]
                if not otm_puts.empty:
                    target_put = otm_puts.iloc[np.argmax(np.abs(otm_puts["iv"] - avg_put_skew))]
                    opportunities.append(self._create_skew_opportunity(asset, target_put, "put", put_skew_z))

            if abs(call_skew_z) > self.entry_threshold:
                otm_calls = expiry_options[expiry_options["inTheMoney"] > 1.05]
                if not otm_calls.empty:
                    target_call = otm_calls.iloc[np.argmax(np.abs(otm_calls["iv"] - avg_call_skew))]
                    opportunities.append(self._create_skew_opportunity(asset, target_call, "call", call_skew_z))

        return opportunities

    def _create_skew_opportunity(self, asset: str, option: pd.Series, option_type: str, zscore: float) -> Dict:
        """Create skew arbitrage opportunity dict"""
        return {
            "asset": asset,
            "type": "skew_arb",
            "subtype": f"{option_type}_skew",
            "expiry": option["expiry"],
            "option_symbol": option["symbol"],
            "strike": option["strike"],
            "current_iv": option[f"{option_type}_iv"],
            "zscore": zscore,
            "direction": -1 if zscore > 0 else 1,
            "confidence": min(abs(zscore) / self.entry_threshold, 3.0),
        }

    def _butterfly_arbitrage(self, asset: str, options: pd.DataFrame, spot: float) -> List[Dict]:
        """
        Detect butterfly arbitrage opportunities (convexity violations)

        A butterfly spread involves buying one option at low strike (K1),
        selling two options at middle strike (K2), and buying one option at high strike (K3).

        For no-arbitrage, the butterfly price must be non-negative:
        C(K1) - 2*C(K2) + C(K3) >= 0 (for calls)
        P(K1) - 2*P(K2) + P(K3) >= 0 (for puts)

        Also checks for convexity violations in the volatility smile.
        """
        opportunities = []

        # Group options by expiry
        for expiry, group in options.groupby("dte"):
            # Sort by strike
            group = group.sort_values("strike").reset_index(drop=True)

            if len(group) < 3:
                continue

            # Check call butterflies
            call_group = group[group["option_type"] == "call"].reset_index(drop=True)
            if len(call_group) >= 3:
                for i in range(len(call_group) - 2):
                    for j in range(i + 1, len(call_group) - 1):
                        for k in range(j + 1, len(call_group)):
                            K1 = call_group.iloc[i]["strike"]
                            K2 = call_group.iloc[j]["strike"]
                            K3 = call_group.iloc[k]["strike"]

                            # Ensure strikes are equally spaced for standard butterfly
                            if abs((K2 - K1) - (K3 - K2)) > 0.01 * K2:
                                continue

                            C1 = call_group.iloc[i]["mid"]
                            C2 = call_group.iloc[j]["mid"]
                            C3 = call_group.iloc[k]["mid"]

                            butterfly_price = C1 - 2 * C2 + C3

                            # Check for arbitrage (negative price or convexity violation)
                            if butterfly_price < -0.01:  # Small threshold for numerical errors
                                # Calculate theoretical minimum price (intrinsic value)
                                intrinsic = max(spot - K1, 0) - 2 * max(spot - K2, 0) + max(spot - K3, 0)

                                if butterfly_price < intrinsic - 0.01:
                                    opportunities.append(
                                        {
                                            "asset": asset,
                                            "type": "butterfly_arb",
                                            "subtype": "call",
                                            "expiry": expiry,
                                            "strikes": [K1, K2, K3],
                                            "option_symbols": [
                                                call_group.iloc[i]["symbol"],
                                                call_group.iloc[j]["symbol"],
                                                call_group.iloc[k]["symbol"],
                                            ],
                                            "prices": [C1, C2, C3],
                                            "butterfly_price": butterfly_price,
                                            "intrinsic_value": intrinsic,
                                            "mispricing": butterfly_price - intrinsic,
                                            "strategy": "buy_butterfly" if butterfly_price < 0 else "sell_butterfly",
                                            "direction": 1 if butterfly_price < 0 else -1,
                                            "max_profit": abs(butterfly_price),
                                            "max_loss": abs(butterfly_price) + (K2 - K1) if butterfly_price < 0 else (K2 - K1),
                                            "breakevens": [K1 + butterfly_price, K3 - butterfly_price],
                                            "confidence": min(abs(butterfly_price) / (0.01 * K2), 3.0),
                                        }
                                    )

                            # Check for volatility smile convexity violation
                            iv1 = call_group.iloc[i]["iv"]
                            iv2 = call_group.iloc[j]["iv"]
                            iv3 = call_group.iloc[k]["iv"]

                            # For a valid smile, IV should be convex (butterfly > 0)
                            # Check if IV butterfly is negative (arbitrage)
                            iv_convexity = iv1 - 2 * iv2 + iv3

                            if iv_convexity < -0.01:  # 1% convexity violation
                                opportunities.append(
                                    {
                                        "asset": asset,
                                        "type": "butterfly_arb",
                                        "subtype": "volatility_smile",
                                        "expiry": expiry,
                                        "strikes": [K1, K2, K3],
                                        "implied_vols": [iv1, iv2, iv3],
                                        "iv_convexity": iv_convexity,
                                        "strategy": "sell_volatility_butterfly",
                                        "direction": -1,
                                        "confidence": min(abs(iv_convexity) / 0.01, 3.0),
                                    }
                                )

            # Check put butterflies
            put_group = group[group["option_type"] == "put"].reset_index(drop=True)
            if len(put_group) >= 3:
                for i in range(len(put_group) - 2):
                    for j in range(i + 1, len(put_group) - 1):
                        for k in range(j + 1, len(put_group)):
                            K1 = put_group.iloc[i]["strike"]
                            K2 = put_group.iloc[j]["strike"]
                            K3 = put_group.iloc[k]["strike"]

                            if abs((K2 - K1) - (K3 - K2)) > 0.01 * K2:
                                continue

                            P1 = put_group.iloc[i]["mid"]
                            P2 = put_group.iloc[j]["mid"]
                            P3 = put_group.iloc[k]["mid"]

                            butterfly_price = P1 - 2 * P2 + P3

                            if butterfly_price < -0.01:
                                intrinsic = max(K1 - spot, 0) - 2 * max(K2 - spot, 0) + max(K3 - spot, 0)

                                if butterfly_price < intrinsic - 0.01:
                                    opportunities.append(
                                        {
                                            "asset": asset,
                                            "type": "butterfly_arb",
                                            "subtype": "put",
                                            "expiry": expiry,
                                            "strikes": [K1, K2, K3],
                                            "option_symbols": [put_group.iloc[i]["symbol"], put_group.iloc[j]["symbol"], put_group.iloc[k]["symbol"]],
                                            "prices": [P1, P2, P3],
                                            "butterfly_price": butterfly_price,
                                            "intrinsic_value": intrinsic,
                                            "mispricing": butterfly_price - intrinsic,
                                            "strategy": "buy_butterfly" if butterfly_price < 0 else "sell_butterfly",
                                            "direction": 1 if butterfly_price < 0 else -1,
                                            "confidence": min(abs(butterfly_price) / (0.01 * K2), 3.0),
                                        }
                                    )

        return opportunities

    def _box_spread_arbitrage(self, asset: str, options: pd.DataFrame, spot: float) -> List[Dict]:
        """
        Detect box spread arbitrage

        A box spread combines a bull call spread and a bear put spread with the same strikes.
        It should be priced at the present value of the difference in strikes.

        Box price = C(K1) - C(K2) + P(K2) - P(K1) = PV(K2 - K1)

        Any deviation from this theoretical price creates an arbitrage opportunity.
        """
        opportunities = []

        # Group options by expiry
        for expiry, group in options.groupby("dte"):
            # Get all strikes for this expiry
            strikes = sorted(group["strike"].unique())

            if len(strikes) < 2:
                continue

            # Check all strike pairs
            for i in range(len(strikes) - 1):
                for j in range(i + 1, len(strikes)):
                    K1 = strikes[i]
                    K2 = strikes[j]

                    # Get options at these strikes
                    opts_k1 = group[group["strike"] == K1]
                    opts_k2 = group[group["strike"] == K2]

                    # Need both calls and puts at each strike
                    call_k1 = opts_k1[opts_k1["option_type"] == "call"]
                    put_k1 = opts_k1[opts_k1["option_type"] == "put"]
                    call_k2 = opts_k2[opts_k2["option_type"] == "call"]
                    put_k2 = opts_k2[opts_k2["option_type"] == "put"]

                    if call_k1.empty or put_k1.empty or call_k2.empty or put_k2.empty:
                        continue

                    call_k1 = call_k1.iloc[0]
                    put_k1 = put_k1.iloc[0]
                    call_k2 = call_k2.iloc[0]
                    put_k2 = put_k2.iloc[0]

                    # Calculate box price
                    box_price = call_k1["mid"] - call_k2["mid"] + put_k2["mid"] - put_k1["mid"]

                    # Theoretical price = PV(K2 - K1)
                    T = call_k1["dte"] / 365
                    theoretical_price = (K2 - K1) * np.exp(-DEFAULT_RISK_FREE_RATE * T)

                    # Check for mispricing
                    mispricing = box_price - theoretical_price
                    mispricing_pct = mispricing / theoretical_price if theoretical_price > 0 else 0

                    if abs(mispricing_pct) > 0.01:  # 1% threshold
                        # Check liquidity
                        min_liquidity = (
                            min(
                                call_k1.get("openInterest", 0) * call_k1.get("mid", 0),
                                put_k1.get("openInterest", 0) * put_k1.get("mid", 0),
                                call_k2.get("openInterest", 0) * call_k2.get("mid", 0),
                                put_k2.get("openInterest", 0) * put_k2.get("mid", 0),
                            )
                            / 1e6
                        )

                        if min_liquidity >= self.min_liquidity:
                            opportunities.append(
                                {
                                    "asset": asset,
                                    "type": "box_spread_arb",
                                    "expiry": expiry,
                                    "strikes": [K1, K2],
                                    "option_symbols": {
                                        "call_k1": call_k1["symbol"],
                                        "put_k1": put_k1["symbol"],
                                        "call_k2": call_k2["symbol"],
                                        "put_k2": put_k2["symbol"],
                                    },
                                    "prices": {
                                        "call_k1": call_k1["mid"],
                                        "put_k1": put_k1["mid"],
                                        "call_k2": call_k2["mid"],
                                        "put_k2": put_k2["mid"],
                                    },
                                    "box_price": box_price,
                                    "theoretical_price": theoretical_price,
                                    "mispricing": mispricing,
                                    "mispricing_pct": mispricing_pct,
                                    "strategy": "buy_box" if mispricing < 0 else "sell_box",
                                    "direction": 1 if mispricing < 0 else -1,
                                    "max_profit": abs(mispricing),
                                    "time_value": T,
                                    "confidence": min(abs(mispricing_pct) / 0.01, 3.0),
                                }
                            )

        return opportunities

    def _calendar_spread_arbitrage(self, asset: str, options: pd.DataFrame, vol_surface: VolatilitySurface, spot: float) -> List[Dict]:
        """
        Detect calendar spread arbitrage

        A calendar spread involves buying a longer-term option and selling a shorter-term option
        at the same strike. Arbitrage opportunities arise when the term structure of volatility
        violates no-arbitrage conditions or when there are pricing inconsistencies.

        Key conditions checked:
        1. Forward volatility should be non-negative
        2. Calendar spread should not be cheaper than intrinsic value
        3. Term structure should be monotonically increasing in normal markets
        """
        opportunities = []

        # Get all expiries
        expiries = sorted(options["dte"].unique())

        if len(expiries) < 2:
            return opportunities

        # Get ATM strike for reference
        atm_strike = spot  # Using spot as reference for ATM

        # Check each strike across expiries
        for strike in options["strike"].unique():
            if abs(strike - atm_strike) > (0.2 * atm_strike):
                continue

            strike_options = options[options["strike"] == strike]

            if len(strike_options) < 2:
                continue

            calls = strike_options[strike_options["option_type"] == "call"].sort_values("dte")
            puts = strike_options[strike_options["option_type"] == "put"].sort_values("dte")

            # Check call calendar spreads
            if len(calls) >= 2:
                for i in range(len(calls) - 1):
                    for j in range(i + 1, len(calls)):
                        near_call = calls.iloc[i]
                        far_call = calls.iloc[j]

                        opportunity = self._analyze_calendar_spread(asset, near_call, far_call, "call", vol_surface, spot)
                        if opportunity:
                            opportunities.append(opportunity)

            # Check put calendar spreads
            if len(puts) >= 2:
                for i in range(len(puts) - 1):
                    for j in range(i + 1, len(puts)):
                        near_put = puts.iloc[i]
                        far_put = puts.iloc[j]

                        opportunity = self._analyze_calendar_spread(asset, near_put, far_put, "put", vol_surface, spot)
                        if opportunity:
                            opportunities.append(opportunity)

        self._analyze_diagonal_calendars(asset, options, vol_surface, spot, opportunities)

        return opportunities

    def _analyze_calendar_spread(
        self, asset: str, near_option: pd.Series, far_option: pd.Series, option_type: str, vol_surface: VolatilitySurface, spot: float
    ) -> Optional[Dict]:
        """Analyze a single calendar spread for arbitrage"""
        opportunities = []
        K = near_option["strike"]
        T1 = near_option["dte"] / 365
        T2 = far_option["dte"] / 365

        # Get implied vols
        iv1 = near_option[f"{option_type}_iv"]
        iv2 = far_option[f"{option_type}_iv"]

        spread_price = far_option["mid"] - near_option["mid"]

        if T2 > T1:
            var1 = iv1**2 * T1
            var2 = iv2**2 * T2
            forward_var = (var2 - var1) / (T2 - T1)

            if forward_var < -0.0001:  # Negative forward variance is arbitrage
                opportunities.append(
                    {
                        "asset": asset,
                        "type": "calendar_spread_arb",
                        "subtype": "negative_forward_vol",
                        "option_type": option_type,
                        "strike": K,
                        "near_expiry": near_option["dte"],
                        "far_expiry": far_option["dte"],
                        "near_iv": iv1,
                        "far_iv": iv2,
                        "forward_var": forward_var,
                        "strategy": "sell_calendar",
                        "direction": -1,
                        "confidence": min(abs(forward_var) / 0.0001, 3.0),
                    }
                )

        # Check 2: Calendar spread shouldn't be cheaper than intrinsic value
        if option_type == "call":
            intrinsic_near = max(spot - K, 0)
            intrinsic_far = max(spot - K, 0)  # Same for same strike
        else:
            intrinsic_near = max(K - spot, 0)
            intrinsic_far = max(K - spot, 0)

        min_spread = 0  # Calendar spread can't be negative for same strike

        if spread_price < min_spread - 0.01:
            opportunities.append(
                {
                    "asset": asset,
                    "type": "calendar_spread_arb",
                    "subtype": "negative_spread",
                    "option_type": option_type,
                    "strike": K,
                    "near_expiry": near_option["dte"],
                    "far_expiry": far_option["dte"],
                    "spread_price": spread_price,
                    "min_theoretical": min_spread,
                    "mispricing": spread_price - min_spread,
                    "strategy": "buy_calendar",
                    "direction": 1,
                    "confidence": min(abs(spread_price) / 0.01, 3.0),
                }
            )

        # Check 3: Term structure anomalies
        # In normal markets, longer-dated options should be more expensive (time value)
        # But this can reverse in special situations
        if option_type == "call":
            time_value_near = near_option["mid"] - intrinsic_near
            time_value_far = far_option["mid"] - intrinsic_far

            # If far option has less time value than near, might be arbitrage
            if time_value_far < time_value_near - 0.5 * time_value_near:
                opportunities.append(
                    {
                        "asset": asset,
                        "type": "calendar_spread_arb",
                        "subtype": "term_structure_reversal",
                        "option_type": option_type,
                        "strike": K,
                        "near_expiry": near_option["dte"],
                        "far_expiry": far_option["dte"],
                        "near_time_value": time_value_near,
                        "far_time_value": time_value_far,
                        "strategy": "sell_calendar",
                        "direction": -1,
                        "confidence": min((time_value_near - time_value_far) / time_value_near, 3.0),
                    }
                )

        return None

    def _analyze_diagonal_calendars(self, asset: str, options: pd.DataFrame, vol_surface: VolatilitySurface, spot: float, opportunities: List[Dict]):
        """
        Analyze diagonal calendar spreads (different strikes and expiries)

        A diagonal calendar spread involves buying a longer-term option at one strike
        and selling a shorter-term option at a different strike.
        """

        # Group by moneyness
        options["inTheMoney"] = options["strike"] / spot

        # Get expiries
        expiries = sorted(options["expiry"].unique())

        for i in range(len(expiries) - 1):
            for j in range(i + 1, len(expiries)):
                near_expiry = expiries[i]
                far_expiry = expiries[j]

                near_opts = options[options["dte"] == near_expiry]
                far_opts = options[options["dte"] == far_expiry]

                # Look for diagonal spreads where strikes are different
                for _, near_opt in near_opts.iterrows():
                    for _, far_opt in far_opts.iterrows():
                        if near_opt["option_type"] != far_opt["option_type"]:
                            continue

                        K_near = near_opt["strike"]
                        K_far = far_opt["strike"]

                        if abs(K_near - K_far) < 0.01 * spot:
                            continue  # Skip calendar spreads (already covered)

                        option_type = near_opt["option_type"]

                        # Calculate theoretical prices using the T values
                        T_near = near_opt["dte"] / 365
                        T_far = far_opt["dte"] / 365

                        # Get the "Fair" Volatility from your VolSurface
                        iv_near = vol_surface.get_volatility(asset, K_near, near_expiry, spot)
                        iv_far = vol_surface.get_volatility(asset, K_far, far_expiry, spot)

                        # Calculate what the options SHOULD cost based on your model
                        engine = OptionsPricingEngine()
                        theoretical_near = engine.price(spot, K_near, T_near, DEFAULT_RISK_FREE_RATE, iv_near, option_type)
                        theoretical_far = engine.price(spot, K_far, T_far, DEFAULT_RISK_FREE_RATE, iv_far, option_type)

                        actual_spread = far_opt["mid"] - near_opt["mid"]
                        theoretical_spread = theoretical_far - theoretical_near
                        mispricing = actual_spread - theoretical_spread

                        if abs(mispricing) > 0.05 * theoretical_spread:
                            opportunities.append(
                                {
                                    "asset": asset,
                                    "type": "diagonal_calendar_arb",
                                    "option_type": option_type,
                                    "near": {
                                        "expiry": near_expiry,
                                        "strike": K_near,
                                        "symbol": near_opt["symbol"],
                                        "price": near_opt["mid"],
                                        "iv": iv_near,
                                    },
                                    "far": {
                                        "expiry": far_expiry,
                                        "strike": K_far,
                                        "symbol": far_opt["symbol"],
                                        "price": far_opt["mid"],
                                        "iv": iv_far,
                                    },
                                    "actual_spread": actual_spread,
                                    "theoretical_spread": theoretical_spread,
                                    "mispricing": mispricing,
                                    "strategy": "buy_diagonal" if mispricing < 0 else "sell_diagonal",
                                    "direction": 1 if mispricing < 0 else -1,
                                    "confidence": min(abs(mispricing) / (0.05 * theoretical_spread), 3.0),
                                }
                            )
