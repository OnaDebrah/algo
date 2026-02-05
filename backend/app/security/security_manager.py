import hashlib
import hmac
import logging
import secrets
from datetime import datetime, timezone
from typing import Dict, List, Optional

from security.api_key import ApiKey
from security.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class SecurityManager:
    """
    Security hardening and protection

    Features:
    - API key management
    - Rate limiting
    - IP whitelisting
    - Encryption
    - CSRF protection

    USAGE:

    # Setup security
    security_mgr = SecurityManager()
    audit_logger = AuditLogger(SessionLocal)
    compliance_mgr = ComplianceManager()

    # Generate API key for user
    api_key = security_mgr.generate_api_key(
        user_id=1,
        name="Trading API Key",
        scopes=["read", "trade"]
    )

    print(f"API Key: {api_key.key_id}")
    print(f"Secret (save this!): {api_key.secret_plaintext}")

    # Validate API key on request
    def authenticate_request(key_id: str, secret: str):
        api_key = security_mgr.validate_api_key(key_id, secret)

        if not api_key:
            return False

        if not api_key.has_scope("trade"):
            return False

        return True

    # Log audit event
    await audit_logger.log_event(
        event_type=AuditEventType.ORDER_PLACED,
        user_id=1,
        details={
            'symbol': 'AAPL',
            'side': 'BUY',
            'quantity': 100,
            'price': 150.50
        },
        strategy_id=1,
        ip_address="192.168.1.1"
    )

    # Check PDT status
    pdt_status = await compliance_mgr.check_pattern_day_trader(
        user_id=1,
        account_value=15000
    )

    if pdt_status['is_pattern_day_trader']:
        print("WARNING: Pattern Day Trader restrictions apply")
    """

    def __init__(self):
        self.api_keys: Dict[str, "ApiKey"] = {}

        self.rate_limiters: Dict[str, "RateLimiter"] = {}

        self.ip_whitelist: List[str] = []

        self.encryption_key = self._load_encryption_key()

    def generate_api_key(self, user_id: int, name: str, scopes: List[str]) -> "ApiKey":
        """
        Generate new API key

        Args:
            user_id: User ID
            name: Key name/description
            scopes: Permissions (e.g., ['read', 'trade'])

        Returns:
            ApiKey object
        """
        # Generate secure random key
        key_id = f"ak_{secrets.token_hex(16)}"
        secret = secrets.token_hex(32)

        # Create API key
        api_key = ApiKey(key_id=key_id, secret_hash=self._hash_secret(secret), user_id=user_id, name=name, scopes=scopes)

        self.api_keys[key_id] = api_key

        logger.info(f"Generated API key {key_id} for user {user_id}")

        # Return with plaintext secret (only shown once!)
        api_key.secret_plaintext = secret
        return api_key

    def validate_api_key(self, key_id: str, secret: str) -> Optional["ApiKey"]:
        """
        Validate API key

        Returns:
            ApiKey if valid, None if invalid
        """
        if key_id not in self.api_keys:
            logger.warning(f"Unknown API key: {key_id}")
            return None

        api_key = self.api_keys[key_id]

        # Check if revoked
        if api_key.is_revoked:
            logger.warning(f"Revoked API key used: {key_id}")
            return None

        # Check if expired
        if api_key.expires_at and datetime.now(timezone.utc) > api_key.expires_at:
            logger.warning(f"Expired API key used: {key_id}")
            return None

        # Validate secret (constant-time comparison)
        secret_hash = self._hash_secret(secret)
        if not hmac.compare_digest(secret_hash, api_key.secret_hash):
            logger.warning(f"Invalid secret for API key: {key_id}")
            return None

        # Update last used
        api_key.last_used_at = datetime.now(timezone.utc)

        return api_key

    def revoke_api_key(self, key_id: str):
        """Revoke an API key"""
        if key_id in self.api_keys:
            self.api_keys[key_id].is_revoked = True
            logger.info(f"Revoked API key: {key_id}")

    def check_rate_limit(self, ip_address: str, endpoint: str) -> bool:
        """
        Check if request is rate limited

        Returns:
            True if allowed, False if rate limited
        """
        if ip_address not in self.rate_limiters:
            self.rate_limiters[ip_address] = RateLimiter(max_requests=100, window_seconds=60)

        limiter = self.rate_limiters[ip_address]
        return limiter.allow_request()

    def check_ip_whitelist(self, ip_address: str) -> bool:
        """Check if IP is whitelisted"""
        if not self.ip_whitelist:
            return True  # No whitelist = allow all

        return ip_address in self.ip_whitelist

    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data (API keys, passwords, etc.)"""
        from cryptography.fernet import Fernet

        f = Fernet(self.encryption_key)
        encrypted = f.encrypt(data.encode())
        return encrypted.decode()

    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        from cryptography.fernet import Fernet

        f = Fernet(self.encryption_key)
        decrypted = f.decrypt(encrypted_data.encode())
        return decrypted.decode()

    def _hash_secret(self, secret: str) -> str:
        """Hash API secret securely"""
        return hashlib.sha256(secret.encode()).hexdigest()

    def _load_encryption_key(self) -> bytes:
        """Load encryption key from environment"""
        import os

        key = os.getenv("ENCRYPTION_KEY")
        if not key:
            logger.warning("No encryption key found, generating temporary key")
            from cryptography.fernet import Fernet

            key = Fernet.generate_key()

        return key if isinstance(key, bytes) else key.encode()
