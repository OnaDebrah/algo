# ============================================================================
# INTEGRATION WITH RISK MANAGER
# ============================================================================
from typing import Dict

import numpy as np
import pandas as pd

from backend.app.strategies.volatility.dynamic_scaling import DynamicVolatilityScalingStrategy
from backend.app.strategies.volatility.volatility_targeting import VolatilityTargetingStrategy


class VolatilityRiskManager:
    """
    Risk Manager with Volatility-Based Controls

    Integrates volatility strategies with overall risk management
    """

    def __init__(self, config: Dict):
        self.config = config

        # Volatility strategies
        self.vol_targeting = VolatilityTargetingStrategy(
            target_volatility=config.get("target_volatility", 0.15),
            vol_lookback=config.get("vol_lookback", 63),
        )

        self.dynamic_scaling = DynamicVolatilityScalingStrategy(
            target_volatility=config.get("target_volatility", 0.15),
            scaling_method=config.get("scaling_method", "multiplicative"),
        )

        # Risk limits
        self.position_limits = config.get("position_limits", {})
        self.sector_limits = config.get("sector_limits", {})
        self.var_limits = config.get("var_limits", {})

        # Monitoring
        self.risk_alerts = []
        self.breach_history = []

    def check_position_risk(self, asset: str, position: float, returns: pd.Series, metadata: Dict = None) -> Dict:
        """
        Check position risk with volatility-based controls
        """
        checks = {
            "approved": True,
            "scaled_position": position,
            "risk_metrics": {},
            "warnings": [],
            "adjustments": [],
        }

        # 1. Volatility scaling
        scaled_position = self.dynamic_scaling.scale_position(asset, position, returns)
        checks["scaled_position"] = scaled_position
        checks["risk_metrics"]["volatility_scale_factor"] = self.dynamic_scaling.scaling_factors.get(asset, 1.0)

        # 2. Position limits
        if asset in self.position_limits:
            limit = self.position_limits[asset]
            if abs(scaled_position) > limit["max"]:
                checks["approved"] = False
                checks["warnings"].append(f"Position limit exceeded: {abs(scaled_position):.2f} > {limit['max']:.2f}")
                checks["adjustments"].append(
                    {
                        "type": "position_cap",
                        "original": scaled_position,
                        "adjusted": np.sign(scaled_position) * limit["max"],
                    }
                )
                scaled_position = np.sign(scaled_position) * limit["max"]

        # 3. Portfolio volatility targeting
        if metadata and "portfolio_returns" in metadata:
            portfolio_signal = self.vol_targeting.generate_portfolio_signal(metadata["portfolio_returns"], current_exposure=abs(scaled_position))

            if portfolio_signal["adjustment"] != 0:
                checks["risk_metrics"]["portfolio_vol_adjustment"] = portfolio_signal["adjustment"]
                checks["risk_metrics"]["target_volatility"] = portfolio_signal["target_volatility"]
                checks["risk_metrics"]["current_portfolio_vol"] = portfolio_signal["current_volatility"]

                # Apply adjustment
                new_exposure = portfolio_signal["new_exposure"]
                if new_exposure != abs(scaled_position):
                    scaled_position = np.sign(scaled_position) * new_exposure
                    checks["adjustments"].append(
                        {
                            "type": "portfolio_vol_targeting",
                            "adjustment": portfolio_signal["adjustment"],
                            "new_exposure": new_exposure,
                        }
                    )

        # 4. VaR check (simplified)
        if asset in self.var_limits:
            var_limit = self.var_limits[asset]
            vol = self.dynamic_scaling.volatility_forecasts.get(asset, 0.15)
            estimated_var = abs(scaled_position) * vol * 2.33 / np.sqrt(252)  # 99% VaR

            if estimated_var > var_limit:
                checks["warnings"].append(f"VaR limit warning: ${estimated_var:.2f} > ${var_limit:.2f}")

        checks["scaled_position"] = scaled_position
        return checks

    def generate_risk_report(self, portfolio_positions: Dict) -> Dict:
        """
        Generate comprehensive risk report
        """
        report = {
            "timestamp": pd.Timestamp.now(),
            "portfolio_summary": {
                "total_positions": len(portfolio_positions),
                "gross_exposure": sum(abs(p) for p in portfolio_positions.values()),
                "net_exposure": sum(portfolio_positions.values()),
            },
            "volatility_metrics": self.dynamic_scaling.generate_risk_report(),
            "risk_limits": {
                "position_limits": self.position_limits,
                "sector_limits": self.sector_limits,
                "var_limits": self.var_limits,
            },
            "alerts": (self.risk_alerts[-10:] if len(self.risk_alerts) > 10 else self.risk_alerts),
            "breaches": (self.breach_history[-5:] if len(self.breach_history) > 5 else self.breach_history),
        }

        return report
