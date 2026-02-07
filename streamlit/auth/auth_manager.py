"""
Authentication and authorization system
"""

import hashlib
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional

import jwt


class UserTier(Enum):
    """User subscription tiers"""

    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class Permission(Enum):
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
TIER_PERMISSIONS = {
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


class AuthManager:
    """Manage authentication and authorization"""

    def __init__(self, db_path: str = "auth.db", secret_key: str = None):
        self.db_path = db_path
        self.secret_key = secret_key or secrets.token_hex(32)
        self._init_database()

    def _init_database(self):
        """Initialize authentication database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                tier TEXT NOT NULL DEFAULT 'free',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                email_verified BOOLEAN DEFAULT 0
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS usage_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                action TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)

        conn.commit()
        conn.close()

    def _hash_password(self, password: str, salt: str = None) -> tuple:
        if salt is None:
            salt = secrets.token_hex(32)
        pwd_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
        return pwd_hash.hex(), salt

    def register_user(self, username: str, email: str, password: str, tier: UserTier = UserTier.FREE) -> Dict:
        if len(password) < 8:
            return {"success": False, "message": "Password must be at least 8 characters"}
        if "@" not in email:
            return {"success": False, "message": "Invalid email address"}

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
            if cursor.fetchone():
                return {"success": False, "message": "Username or email already exists"}

            pwd_hash, salt = self._hash_password(password)

            cursor.execute(
                """
                INSERT INTO users (username, email, password_hash, salt, tier)
                VALUES (?, ?, ?, ?, ?)
            """,
                (username, email, pwd_hash, salt, tier.value),
            )

            user_id = cursor.lastrowid
            conn.commit()
            conn.close()

            return {"success": True, "message": "User registered successfully", "user_id": user_id}
        except Exception as e:
            return {"success": False, "message": f"Registration failed: {str(e)}"}

    def login(self, username: str, password: str) -> Dict:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT id, username, email, password_hash, salt, tier, is_active
                FROM users WHERE username = ? OR email = ?
            """,
                (username, username),
            )

            user = cursor.fetchone()
            if not user:
                return {"success": False, "message": "Invalid credentials"}

            user_id, uname, email, pwd_hash, salt, tier, is_active = user

            if not is_active:
                return {"success": False, "message": "Account is inactive"}

            check_hash, _ = self._hash_password(password, salt)
            if check_hash != pwd_hash:
                return {"success": False, "message": "Invalid credentials"}

            cursor.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?", (user_id,))

            token = self._create_session(user_id, cursor)
            conn.commit()
            conn.close()

            return {
                "success": True,
                "message": "Page successful",
                "token": token,
                "user": {"id": user_id, "username": uname, "email": email, "tier": tier},
            }
        except Exception as e:
            return {"success": False, "message": f"Page failed: {str(e)}"}

    def _create_session(self, user_id: int, cursor) -> str:
        payload = {"user_id": user_id, "exp": datetime.now(timezone.utc) + timedelta(days=7), "iat": datetime.now(timezone.utc)}
        token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        expires_at = datetime.now(timezone.utc) + timedelta(days=7)
        cursor.execute("INSERT INTO sessions (user_id, token, expires_at) VALUES (?, ?, ?)", (user_id, token, expires_at))
        return token

    def verify_token(self, token: str) -> Optional[Dict]:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            user_id = payload["user_id"]

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT id, username, email, tier, is_active FROM users WHERE id = ?", (user_id,))
            user = cursor.fetchone()
            conn.close()

            if not user or not user[4]:
                return None

            return {"id": user[0], "username": user[1], "email": user[2], "tier": user[3]}
        except Exception:
            return None

    def logout(self, token: str) -> bool:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM sessions WHERE token = ?", (token,))
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False

    def has_permission(self, user_tier: str, permission: Permission) -> bool:
        try:
            tier = UserTier(user_tier)
            return permission in TIER_PERMISSIONS.get(tier, [])
        except ValueError:
            return False

    def get_user_permissions(self, user_tier: str) -> List[Permission]:
        try:
            tier = UserTier(user_tier)
            return TIER_PERMISSIONS.get(tier, [])
        except ValueError:
            return []

    def update_user_tier(self, user_id: int, new_tier: UserTier) -> bool:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("UPDATE users SET tier = ? WHERE id = ?", (new_tier.value, user_id))
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False

    def track_usage(self, user_id: int, action: str, metadata: str = None):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO usage_tracking (user_id, action, metadata) VALUES (?, ?, ?)", (user_id, action, metadata))
            conn.commit()
            conn.close()
        except Exception:
            pass
