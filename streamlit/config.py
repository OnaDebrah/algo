"""
Configuration settings for the trading platform
"""

import logging

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Database settings
DATABASE_PATH = "../trading_platform.db"

# Default trading parameters
DEFAULT_INITIAL_CAPITAL = 100000
DEFAULT_MAX_POSITION_SIZE = 0.1
DEFAULT_STOP_LOSS_PCT = 0.05
DEFAULT_MAX_DRAWDOWN = 0.15

# Strategy parameters
DEFAULT_SMA_SHORT = 20
DEFAULT_SMA_LONG = 50
DEFAULT_RSI_PERIOD = 14
DEFAULT_RSI_OVERSOLD = 30
DEFAULT_RSI_OVERBOUGHT = 70
DEFAULT_MACD_FAST = 12
DEFAULT_MACD_SLOW = 26
DEFAULT_MACD_SIGNAL = 9

# ML parameters
DEFAULT_ML_TEST_SIZE = 0.2
DEFAULT_ML_THRESHOLD = 0.02

# UI settings
PAGE_TITLE = "🏛️ ORACULUM"
PAGE_ICON = "📈"
LAYOUT = "wide"

# Data fetching
DEFAULT_PERIOD = "1mo"
DEFAULT_INTERVAL = "1d"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

# Options Configuration
OPTIONS_CONFIG = {
    "risk_free_rate": 0.05,
    "default_volatility": 0.30,
    "commission_per_contract": 0.65,
    "default_dte": 30,  # Days to expiration
    "max_positions": 10,
}

# Supported Asset Classes
SUPPORTED_ASSET_CLASSES = [
    "Stock",
    "Cryptocurrency",
    "Forex",
    "Commodity",
    "ETF",
    "Index",
    "Bond",
    "Futures",
]

ANTHROPIC_API_KEY = ""

# Database
AUTH_DB_PATH = "../auth.db"

# JWT Configuration
JWT_SECRET_KEY = "b4b9dec99638d32897d5b2705755cba147659e02675d4615011951ce24f6aff1"
JWT_EXPIRATION_DAYS = 7

# Security
PASSWORD_MIN_LENGTH = 8
MAX_LOGIN_ATTEMPTS = 5
SESSION_TIMEOUT_MINUTES = 30
