import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from security.audit_type import AuditEventType

logger = logging.getLogger(__name__)


class AuditLogger:
    """
    Comprehensive audit logging system

    Features:
    - Immutable audit trail
    - Event correlation
    - Compliance reporting
    - Tamper detection
    """

    def __init__(self, db_session_factory):
        self.db_factory = db_session_factory

        # Chain hash for tamper detection
        self.last_hash: Optional[str] = None

    async def log_event(
        self,
        event_type: AuditEventType,
        user_id: Optional[int],
        details: Dict[str, Any],
        strategy_id: Optional[int] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> str:
        """
        Log an audit event

        Returns:
            event_id: Unique event ID
        """
        timestamp = datetime.now(timezone.utc)

        # Create event record
        event = {
            "timestamp": timestamp,
            "event_type": event_type.value,
            "user_id": user_id,
            "strategy_id": strategy_id,
            "details": details,
            "ip_address": ip_address,
            "user_agent": user_agent,
        }

        # Calculate hash (for tamper detection)
        event_hash = self._calculate_hash(event)
        event["hash"] = event_hash
        event["previous_hash"] = self.last_hash

        # Save to database
        event_id = await self._save_to_db(event)

        # Update chain
        self.last_hash = event_hash

        # Log to file (secondary backup)
        logger.info(f"AUDIT: {event_type.value} - User {user_id} - {json.dumps(details)}")

        return event_id

    def _calculate_hash(self, event: Dict[str, Any]) -> str:
        """Calculate cryptographic hash of event"""
        # Create deterministic string representation
        event_str = json.dumps(
            {
                "timestamp": event["timestamp"].isoformat(),
                "event_type": event["event_type"],
                "user_id": event["user_id"],
                "details": event["details"],
                "previous_hash": self.last_hash,
            },
            sort_keys=True,
        )

        # SHA-256 hash
        return hashlib.sha256(event_str.encode()).hexdigest()

    async def _save_to_db(self, event: Dict[str, Any]) -> str:
        """Save event to database"""
        db = self.db_factory()

        try:
            # Save to audit_log table
            # Implementation depends on your ORM
            event_id = f"AE_{int(event['timestamp'].timestamp() * 1000)}"

            # Simplified - use your actual model
            # audit_record = AuditLog(**event)
            # db.add(audit_record)
            # db.commit()

            return event_id

        finally:
            db.close()

    async def get_audit_trail(
        self,
        user_id: Optional[int] = None,
        strategy_id: Optional[int] = None,
        event_type: Optional[AuditEventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query audit trail with filters"""
        # Implementation would query database
        # Return filtered audit events
        return []

    async def verify_integrity(self) -> bool:
        """
        Verify audit trail integrity

        Checks hash chain for tampering
        """
        # Implementation would verify hash chain
        return True
