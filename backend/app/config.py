"""
Configuration settings for FastAPI backend
"""

import os
import secrets
from typing import List

from pydantic import model_validator
from pydantic_settings import BaseSettings

# Ephemeral dev-only secret — regenerated on each server restart.
# Production deployments MUST set JWT_SECRET_KEY and SECRET_KEY env vars.
_DEV_SECRET = secrets.token_hex(32)


class Settings(BaseSettings):
    # App
    APP_NAME: str = "ORACULUM API"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    # Security - JWT  (no hardcoded secret — uses ephemeral random in dev)
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", _DEV_SECRET)
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_MINUTES: int = 60 * 24 * 7  # 7 days

    # Security - General  (no hardcoded secret — uses ephemeral random in dev)
    SECRET_KEY: str = os.getenv("SECRET_KEY", _DEV_SECRET)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 10080  # 7 days

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
    ]

    # PostgreSQL (Primary - for production and development)
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "trading_user")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "password")
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "postgres")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5432")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "trading_platform")

    # Database Configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")

    @model_validator(mode="after")
    def assemble_db_url(self) -> "Settings":
        if not self.DATABASE_URL:
            self.DATABASE_URL = (
                f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
            )
        elif self.DATABASE_URL.startswith("postgresql://"):
            self.DATABASE_URL = self.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
        return self

    # SQLite paths (for legacy/testing)
    DATABASE_PATH: str = os.getenv("DATABASE_PATH", "trading_platform.db")
    AUTH_DB_PATH: str = os.getenv("AUTH_DB_PATH", "auth.db")

    # Database Connection Pool Settings
    DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "20"))
    DB_MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "10"))
    DB_POOL_RECYCLE: int = int(os.getenv("DB_POOL_RECYCLE", "3600"))  # 1 hour
    DB_POOL_PRE_PING: bool = os.getenv("DB_POOL_PRE_PING", "true").lower() == "true"
    DB_ECHO: bool = os.getenv("DB_ECHO", "false").lower() == "true"  # SQL logging

    # API Keys — set via .env file or environment variables, never hardcode
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL_HAIKU_3: str = os.getenv("ANTHROPIC_MODEL_HAIKU_3", "claude-3-haiku-20240307")
    ANTHROPIC_MODEL_SONNET_4: str = os.getenv("ANTHROPIC_MODEL_SONNET_4", "claude-sonnet-4-20250514")

    # DeepSeek (fallback for AI code generation)
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-coder")

    # Google Gemini (fallback for AI code generation)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    # OpenAI / ChatGPT (fallback for AI code generation)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Trading Defaults
    DEFAULT_INITIAL_CAPITAL: float = float(os.getenv("DEFAULT_INITIAL_CAPITAL", "100000"))
    DEFAULT_COMMISSION_RATE: float = float(os.getenv("DEFAULT_COMMISSION_RATE", "0.001"))
    DEFAULT_SLIPPAGE_RATE: float = float(os.getenv("DEFAULT_SLIPPAGE_RATE", "0.0005"))
    DEFAULT_RISK_FREE_RATE: float = float(os.getenv("DEFAULT_RISK_FREE_RATE", "0.05"))
    DEFAULT_IB_CLIENT_ID_MODULUS: int = int(os.getenv("DEFAULT_IB_CLIENT_ID_MODULUS", "32700"))
    DEFAULT_MAX_POSITION_SIZE: int = int(os.getenv("DEFAULT_MAX_POSITION_SIZE", "20"))
    DEFAULT_STOP_LOSS_PCT: float = float(os.getenv("DEFAULT_STOP_LOSS_PCT", "0.05"))
    DEFAULT_MAX_DRAWDOWN: float = float(os.getenv("DEFAULT_MAX_DRAWDOWN", "0.10"))

    # Strategy Specific Defaults
    DEFAULT_SMA_SHORT: int = 20
    DEFAULT_SMA_LONG: int = 50
    DEFAULT_RSI_PERIOD: int = 14
    DEFAULT_RSI_OVERBOUGHT: float = 70.0
    DEFAULT_RSI_OVERSOLD: float = 30.0
    DEFAULT_MACD_FAST: int = 12
    DEFAULT_MACD_SLOW: int = 26
    DEFAULT_MACD_SIGNAL: int = 9
    DEFAULT_ML_TEST_SIZE: float = 0.2
    DEFAULT_ML_THRESHOLD: float = 0.002
    ML_MODELS_DIR: str = os.getenv("ML_MODELS_DIR", "ml_models")

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))

    # Market Data
    DATA_PROVIDER: str = os.getenv("DATA_PROVIDER", "yahoo")  # 'yahoo', 'polygon', 'alpaca', or 'iex'
    YAHOO_SEARCH_URL: str = os.getenv("YAHOO_SEARCH_URL", "https://query2.finance.yahoo.com/v1/finance/search")
    POLYGON_API_KEY: str = os.getenv("POLYGON_API_KEY", "")
    IEX_API_KEY: str = os.getenv("IEX_API_KEY", "")

    ALPHA_VANTAGE_BASE_URL: str = os.getenv("ALPHA_VANTAGE_BASE_URL", "https://www.alphavantage.co/query")
    ALPHA_VANTAGE_API_KEY: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    FREE_TIER_CALLS_PER_MINUTE: int = int(os.getenv("FREE_TIER_CALLS_PER_MINUTE", "5"))
    PREMIUM_TIER_CALLS_PER_MINUTE: int = int(os.getenv("PREMIUM_TIER_CALLS_PER_MINUTE", "30"))
    CALL_DELAY: int = int(os.getenv("CALL_DELAY", "12"))  # seconds between calls for free tier (60/5 = 12)

    USER_AGENT: str = os.getenv(
        "USER_AGENT", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )

    # Macro Data
    # FRED API (free registration at https://fred.stlouisfed.org/docs/api/api_key.html)
    FRED_API_KEY: str = os.getenv("FRED_API_KEY", "")

    # BLS API (free registration at https://data.bls.gov/registrationEngine/)
    BLS_API_KEY: str = os.getenv("BLS_API_KEY", "")
    BLS_BASE_URL: str = os.getenv("BLS_BASE_URL", "https://api.bls.gov/publicAPI/v2/timeseries/data/")

    NEWSAPI_KEY: str = os.getenv("NEWSAPI_KEY", "")

    TWITTER_BEARER_TOKEN: str = os.getenv("TWITTER_BEARER_TOKEN", "")
    REDDIT_CLIENT_ID: str = os.getenv("REDDIT_CLIENT_ID", "")
    REDDIT_CLIENT_SECRET: str = os.getenv("REDDIT_CLIENT_SECRET", "")

    # Alerts (SMTP)
    SMTP_SERVER: str = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USERNAME: str = os.getenv("SMTP_USERNAME", "")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
    FROM_EMAIL: str = os.getenv("FROM_EMAIL", "alerts@oraculum.com")
    TO_EMAIL: str = os.getenv("TO_EMAIL", "")
    EMAIL_ENABLED: bool = os.getenv("EMAIL_ENABLED", "false").lower() == "true"

    # Alerts (Twilio)
    TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    TWILIO_FROM_NUMBER: str = os.getenv("TWILIO_FROM_NUMBER", "")
    TWILIO_TO_NUMBER: str = os.getenv("TWILIO_TO_NUMBER", "")
    SMS_ENABLED: bool = os.getenv("SMS_ENABLED", "false").lower() == "true"

    # Redis (for WebSocket/caching)
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
    REDIS_ENABLED: bool = os.getenv("REDIS_ENABLED", "false").lower() == "true"

    # Logging & Observability
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "app.log")

    SENTRY_DSN: str = os.getenv("SENTRY_DSN", "")
    ENFORCE_HTTPS: bool = os.getenv("ENFORCE_HTTPS", "false").lower() == "true"

    # Brokers
    ALPACA_PAPER_BASE_URL: str = os.getenv("ALPACA_PAPER_BASE_URL", "https://paper-api.alpaca.markets/v2")
    ALPACA_BASE_URL: str = os.getenv("ALPACA_BASE_URL", "https://api.alpaca.markets")
    ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
    ALPACA_SECRET: str = os.getenv("ALPACA_SECRET", "")

    IB_PAPER_PORT: int = int(os.getenv("IB_PAPER_PORT", "7497"))
    IB_LIVE_PORT: int = int(os.getenv("IB_LIVE_PORT", "7496"))
    IB_HOST: str = os.getenv("IB_HOST", "127.0.0.1")
    IB_CLIENT_ID: int = int(os.getenv("IB_CLIENT_ID", "1"))

    class Config:
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"


settings = Settings()

# Export settings as globals for compatibility with legacy imports
DEFAULT_INITIAL_CAPITAL = settings.DEFAULT_INITIAL_CAPITAL
DEFAULT_COMMISSION_RATE = settings.DEFAULT_COMMISSION_RATE
DEFAULT_SLIPPAGE_RATE = settings.DEFAULT_SLIPPAGE_RATE
DEFAULT_MAX_POSITION_SIZE = settings.DEFAULT_MAX_POSITION_SIZE
DEFAULT_STOP_LOSS_PCT = settings.DEFAULT_STOP_LOSS_PCT
DEFAULT_MAX_DRAWDOWN = settings.DEFAULT_MAX_DRAWDOWN

# Strategy Specific Defaults
DEFAULT_SMA_SHORT = settings.DEFAULT_SMA_SHORT
DEFAULT_SMA_LONG = settings.DEFAULT_SMA_LONG
DEFAULT_RSI_PERIOD = settings.DEFAULT_RSI_PERIOD
DEFAULT_RSI_OVERBOUGHT = settings.DEFAULT_RSI_OVERBOUGHT
DEFAULT_RSI_OVERSOLD = settings.DEFAULT_RSI_OVERSOLD
DEFAULT_MACD_FAST = settings.DEFAULT_MACD_FAST
DEFAULT_MACD_SLOW = settings.DEFAULT_MACD_SLOW
DEFAULT_MACD_SIGNAL = settings.DEFAULT_MACD_SIGNAL
DEFAULT_ML_TEST_SIZE = settings.DEFAULT_ML_TEST_SIZE
DEFAULT_ML_THRESHOLD = settings.DEFAULT_ML_THRESHOLD

ML_MODELS_DIR = settings.ML_MODELS_DIR

BLS_BASE_URL = settings.BLS_BASE_URL
ANTHROPIC_MODEL_HAIKU_3 = settings.ANTHROPIC_MODEL_HAIKU_3
ANTHROPIC_MODEL_SONNET_4 = settings.ANTHROPIC_MODEL_SONNET_4
GEMINI_MODEL = settings.GEMINI_MODEL
OPENAI_MODEL = settings.OPENAI_MODEL

NEWSAPI_KEY = settings.NEWSAPI_KEY
TWITTER_BEARER_TOKEN = settings.TWITTER_BEARER_TOKEN

REDDIT_CLIENT_ID = settings.REDDIT_CLIENT_ID
REDDIT_CLIENT_SECRET = settings.REDDIT_CLIENT_SECRET

ALPHA_VANTAGE_BASE_URL = settings.ALPHA_VANTAGE_BASE_URL
FREE_TIER_CALLS_PER_MINUTE = settings.FREE_TIER_CALLS_PER_MINUTE
PREMIUM_TIER_CALLS_PER_MINUTE = settings.PREMIUM_TIER_CALLS_PER_MINUTE
CALL_DELAY = settings.CALL_DELAY  # seconds between calls for free tier (60/5 = 12)

YAHOO_SEARCH_URL = settings.YAHOO_SEARCH_URL


def validate_settings():
    """Validate critical settings on startup"""
    import logging

    _logger = logging.getLogger("app.config")

    if settings.ENVIRONMENT not in ("development", "test"):
        # Production: require explicitly-set secrets (reject ephemeral dev defaults)
        if not settings.JWT_SECRET_KEY or settings.JWT_SECRET_KEY == _DEV_SECRET:
            raise ValueError(f"Environment '{settings.ENVIRONMENT}' must set JWT_SECRET_KEY env var")
        if not settings.SECRET_KEY or settings.SECRET_KEY == _DEV_SECRET:
            raise ValueError(f"Environment '{settings.ENVIRONMENT}' must set SECRET_KEY env var")
        if settings.POSTGRES_PASSWORD == "password" or not settings.POSTGRES_PASSWORD:
            raise ValueError(f"Environment '{settings.ENVIRONMENT}' must NOT use the default POSTGRES_PASSWORD")

    # Ensure database URL is set
    if not settings.DATABASE_URL:
        raise ValueError("DATABASE_URL must be configured")

    # ── Warn about missing optional API keys ──────────────────────────
    _optional_keys = {
        "ANTHROPIC_API_KEY": "AI Analyst / AI Advisor",
        "POLYGON_API_KEY": "Polygon data provider",
        "FRED_API_KEY": "FRED macro data",
        "BLS_API_KEY": "BLS economic data",
        "ALPACA_API_KEY": "Alpaca broker",
        "DEEPSEEK_API_KEY": "DeepSeek code generation (Strategy Builder fallback)",
        "GEMINI_API_KEY": "Gemini code generation (Strategy Builder fallback)",
        "OPENAI_API_KEY": "OpenAI/ChatGPT code generation (Strategy Builder fallback)",
    }
    for _key, _feature in _optional_keys.items():
        if not getattr(settings, _key, ""):
            _logger.warning(f"{_key} not set — {_feature} will be unavailable")

    # ── Validate data provider ↔ API key consistency ──────────────────
    if settings.DATA_PROVIDER == "polygon" and not settings.POLYGON_API_KEY:
        raise ValueError("DATA_PROVIDER is 'polygon' but POLYGON_API_KEY is not set")
    if settings.DATA_PROVIDER == "iex" and not settings.IEX_API_KEY:
        raise ValueError("DATA_PROVIDER is 'iex' but IEX_API_KEY is not set")
    if settings.DATA_PROVIDER == "alpaca" and not settings.ALPACA_API_KEY:
        raise ValueError("DATA_PROVIDER is 'alpaca' but ALPACA_API_KEY is not set")


validate_settings()
