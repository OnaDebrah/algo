from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class ApiKey:
    """API key model"""

    def __init__(self, key_id: str, secret_hash: str, user_id: int, name: str, scopes: List[str]):
        self.key_id = key_id
        self.secret_hash = secret_hash
        self.secret_plaintext: Optional[str] = None  # Only populated on creation
        self.user_id = user_id
        self.name = name
        self.scopes = scopes

        self.created_at = datetime.now(timezone.utc)
        self.last_used_at: Optional[datetime] = None
        self.expires_at: Optional[datetime] = None
        self.is_revoked = False

    def has_scope(self, required_scope: str) -> bool:
        """Check if key has required scope"""
        return required_scope in self.scopes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict (never include secret!)"""
        return {
            "key_id": self.key_id,
            "user_id": self.user_id,
            "name": self.name,
            "scopes": self.scopes,
            "created_at": self.created_at.isoformat(),
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_revoked": self.is_revoked,
        }
