"""
Security utilities for authentication
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

# ── bcrypt 5.0 + passlib 1.7.4 compatibility shim ──────────────────
# passlib 1.7.4 (unmaintained) calls bcrypt.hashpw with >72-byte test
# strings during its internal wrap-bug detection.  bcrypt >=4.1 raises
# ValueError instead of silently truncating.  Monkey-patch hashpw
# *before* passlib loads the backend so the detection succeeds.
import bcrypt as _bcrypt
from fastapi import HTTPException, status
from jose import JWTError, jwt
from passlib.context import CryptContext

from ..config import settings

_orig_hashpw = _bcrypt.hashpw


def _patched_hashpw(password: bytes, salt: bytes) -> bytes:
    return _orig_hashpw(password[:72], salt)


_bcrypt.hashpw = _patched_hashpw


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash.
    Truncate to 72 bytes for bcrypt 5.0+ compatibility (passlib 1.7.4
    expects silent truncation but bcrypt 4.1+ raises ValueError instead).
    """
    return pwd_context.verify(plain_password[:72], hashed_password)


def get_password_hash(password: str) -> str:
    """Hash password (truncated to 72 bytes for bcrypt compat)."""
    return pwd_context.hash(password[:72])


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.JWT_EXPIRATION_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> dict:
    """Decode JWT token"""
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
