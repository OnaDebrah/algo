"""
Configuration settings for FastAPI backend
"""

import os
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    APP_NAME: str = "Trading Platform API"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    # Security
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "b4b9dec99638d32897d5b2705755cba147659e02675d4615011951ce24f6aff1")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_MINUTES: int = 60 * 24 * 7  # 7 days

    # Security
    SECRET_KEY: str = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 10080  # 7 days

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
    ]

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./trading_platform.db"

    # For async PostgreSQL in production:
    # DATABASE_URL: str = "postgresql+asyncpg://user:password@localhost/trading_platform"

    DATABASE_PATH: str = os.getenv("DATABASE_PATH", "trading_platform.db")
    AUTH_DB_PATH: str = os.getenv("AUTH_DB_PATH", "auth.db")

    # API Keys
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Trading Defaults
    DEFAULT_INITIAL_CAPITAL: float = 100000
    DEFAULT_COMMISSION_RATE: float = 0.001
    DEFAULT_SLIPPAGE_RATE: float = 0.0005

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60

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

    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()
