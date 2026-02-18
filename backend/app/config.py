"""
Configuration settings for FastAPI backend
"""

import os
from typing import List

from pydantic import model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    APP_NAME: str = "Trading Platform API"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    # Security - JWT
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "b4b9dec99638d32897d5b2705755cba147659e02675d4615011951ce24f6aff1")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_MINUTES: int = 60 * 24 * 7  # 7 days

    # Security - General
    SECRET_KEY: str = os.getenv("SECRET_KEY", "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 10080  # 7 days

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
    ]

    # Database Configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")

    # PostgreSQL (Primary - for production and development)
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "trading_user")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "secure_password_here")
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5432")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "trading_platform")

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

    # API Keys
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Trading Defaults
    DEFAULT_INITIAL_CAPITAL: float = float(os.getenv("DEFAULT_INITIAL_CAPITAL", "100000"))
    DEFAULT_COMMISSION_RATE: float = float(os.getenv("DEFAULT_COMMISSION_RATE", "0.001"))
    DEFAULT_SLIPPAGE_RATE: float = float(os.getenv("DEFAULT_SLIPPAGE_RATE", "0.0005"))
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
    DEFAULT_ML_THRESHOLD: float = 0.5
    ML_MODELS_DIR: str = os.getenv("ML_MODELS_DIR", "ml_models")

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))

    # Market Data
    DATA_PROVIDER: str = os.getenv("DATA_PROVIDER", "yahoo")  # 'yahoo', 'polygon', 'alpaca', or 'iex'
    POLYGON_API_KEY: str = os.getenv("POLYGON_API_KEY", "")
    IEX_API_KEY: str = os.getenv("IEX_API_KEY", "")
    USER_AGENT: str = os.getenv(
        "USER_AGENT", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )

    # Alerts (SMTP)
    SMTP_SERVER: str = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USERNAME: str = os.getenv("SMTP_USERNAME", "")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
    FROM_EMAIL: str = os.getenv("FROM_EMAIL", "alerts@yourplatform.com")
    TO_EMAIL: str = os.getenv("TO_EMAIL", "")
    EMAIL_ENABLED: bool = os.getenv("EMAIL_ENABLED", "false").lower() == "true"

    # Alerts (Twilio)
    TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    TWILIO_FROM_NUMBER: str = os.getenv("TWILIO_FROM_NUMBER", "")
    TWILIO_TO_NUMBER: str = os.getenv("TWILIO_TO_NUMBER", "")
    SMS_ENABLED: bool = os.getenv("SMS_ENABLED", "false").lower() == "true"

    # Redis (for WebSocket/caching)
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    REDIS_ENABLED: bool = os.getenv("REDIS_ENABLED", "false").lower() == "true"

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "app.log")

    # Brokers
    ALPACA_PAPER_BASE_URL: str = os.getenv("ALPACA_PAPER_BASE_URL", "https://paper-api.alpaca.markets")
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


# Validation on import
def validate_settings():
    """Validate critical settings on startup"""
    if settings.ENVIRONMENT == "production":
        if settings.JWT_SECRET_KEY == "b4b9dec99638d32897d5b2705755cba147659e02675d4615011951ce24f6aff1":
            raise ValueError("Production environment must use custom JWT_SECRET_KEY")
        if settings.SECRET_KEY == "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7":
            raise ValueError("Production environment must use custom SECRET_KEY")

    # Ensure database URL is set
    if not settings.DATABASE_URL:
        raise ValueError("DATABASE_URL must be configured")


# Run validation
validate_settings()
