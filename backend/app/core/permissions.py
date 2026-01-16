from enum import Enum
from typing import Dict, List

class UserTier(str, Enum):
    """User subscription tiers"""
    FREE = "FREE"
    BASIC = "BASIC"
    PRO = "PRO"
    ENTERPRISE = "ENTERPRISE"

class Permission(str, Enum):
    """Feature permissions"""
    VIEW_DASHBOARD = "view_dashboard"
    BASIC_BACKTEST = "basic_backtest"
    MULTI_ASSET_BACKTEST = "multi_asset_backtest"
    ML_STRATEGIES = "ml_strategies"
    LIVE_TRADING = "live_trading"
    ADVANCED_ANALYTICS = "advanced_analytics"
    API_ACCESS = "api_access"
    UNLIMITED_BACKTESTS = "unlimited_backtests"
    PRIORITY_SUPPORT = "priority_support"
    CUSTOM_STRATEGIES = "custom_strategies"

# Tier permissions mapping
TIER_PERMISSIONS: Dict[UserTier, List[Permission]] = {
    UserTier.FREE: [
        Permission.VIEW_DASHBOARD,
        Permission.BASIC_BACKTEST,
    ],
    UserTier.BASIC: [
        Permission.VIEW_DASHBOARD,
        Permission.BASIC_BACKTEST,
        Permission.MULTI_ASSET_BACKTEST,
        Permission.ADVANCED_ANALYTICS,
    ],
    UserTier.PRO: [
        Permission.VIEW_DASHBOARD,
        Permission.BASIC_BACKTEST,
        Permission.MULTI_ASSET_BACKTEST,
        Permission.ML_STRATEGIES,
        Permission.ADVANCED_ANALYTICS,
        Permission.UNLIMITED_BACKTESTS,
        Permission.CUSTOM_STRATEGIES,
    ],
    UserTier.ENTERPRISE: [
        Permission.VIEW_DASHBOARD,
        Permission.BASIC_BACKTEST,
        Permission.MULTI_ASSET_BACKTEST,
        Permission.ML_STRATEGIES,
        Permission.LIVE_TRADING,
        Permission.ADVANCED_ANALYTICS,
        Permission.API_ACCESS,
        Permission.UNLIMITED_BACKTESTS,
        Permission.PRIORITY_SUPPORT,
        Permission.CUSTOM_STRATEGIES,
    ],
}
