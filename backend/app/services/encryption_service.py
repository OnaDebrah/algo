"""
Strategy code encryption service using Fernet (AES-128-CBC via cryptography library).

Encrypted values are prefixed with "ENC:" to distinguish from plaintext.
This allows backward-compatible reads of old unencrypted rows.
"""

import logging
import os

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)

# Prefix used to identify encrypted values in the database
_ENC_PREFIX = "ENC:"


class EncryptionService:
    """Thin wrapper around Fernet for encrypting/decrypting strategy code at rest."""

    _instance: "EncryptionService | None" = None

    def __init__(self, key: bytes | str | None = None):
        raw = key or os.getenv("ENCRYPTION_KEY")
        if not raw:
            logger.warning(
                "ENCRYPTION_KEY not set — generating an ephemeral key. "
                "Data encrypted with this key will be UNRECOVERABLE after restart. "
                "Set ENCRYPTION_KEY in your environment for production use."
            )
            raw = Fernet.generate_key()

        self._key: bytes = raw if isinstance(raw, bytes) else raw.encode()
        self._fernet = Fernet(self._key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encrypt_code(self, plaintext: str) -> str:
        """Encrypt strategy code and return an ``ENC:``-prefixed ciphertext string."""
        if not plaintext:
            return plaintext
        token = self._fernet.encrypt(plaintext.encode("utf-8"))
        return f"{_ENC_PREFIX}{token.decode('utf-8')}"

    def decrypt_code(self, ciphertext: str) -> str:
        """Decrypt strategy code.

        If *ciphertext* is not prefixed with ``ENC:`` it is assumed to be
        legacy plaintext and returned as-is (backward compatibility).
        """
        if not ciphertext:
            return ciphertext
        if not ciphertext.startswith(_ENC_PREFIX):
            # Legacy plaintext — return unchanged
            return ciphertext
        try:
            raw_token = ciphertext[len(_ENC_PREFIX) :]
            return self._fernet.decrypt(raw_token.encode("utf-8")).decode("utf-8")
        except InvalidToken:
            logger.error("Failed to decrypt strategy code — invalid token or wrong key")
            raise ValueError("Unable to decrypt strategy code. The encryption key may have changed.")

    def is_encrypted(self, value: str) -> bool:
        """Return *True* if *value* carries the ``ENC:`` prefix."""
        return bool(value) and value.startswith(_ENC_PREFIX)

    # ------------------------------------------------------------------
    # Singleton accessor
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "EncryptionService":
        """Return (or create) the module-level singleton."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


def get_encryption_service() -> EncryptionService:
    """Convenience function used as a FastAPI / general-purpose accessor."""
    return EncryptionService.get_instance()
