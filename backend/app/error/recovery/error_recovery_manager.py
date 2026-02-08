"""
Error Recovery & Persistent State Management
Graceful error handling, state persistence, crash recovery
"""

import asyncio
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List

from error.recovery.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


class RecoveryAction(str, Enum):
    """Recovery actions"""

    RETRY = "retry"
    PAUSE = "pause"
    STOP = "stop"
    NOTIFY = "notify"
    RESTART = "restart"


class ErrorSeverity(str, Enum):
    """Error severity levels"""

    LOW = "low"  # Log and continue
    MEDIUM = "medium"  # Log and retry
    HIGH = "high"  # Pause strategy
    CRITICAL = "critical"  # Stop strategy


class ErrorRecoveryManager:
    """
    Manages error recovery for live trading

    Features:
    - Automatic retry with exponential backoff
    - Error classification
    - Circuit breaker pattern
    - Graceful degradation
    """

    def __init__(self):
        # Error history: {strategy_id: [errors]}
        self.error_history: Dict[int, List[Dict[str, Any]]] = {}

        # Circuit breaker state
        self.circuit_breakers: Dict[int, CircuitBreaker] = {}

        # Retry configuration
        self.max_retries = 3
        self.retry_delays = [1, 5, 15]  # seconds

    async def handle_error(self, strategy_id: int, error: Exception, context: Dict[str, Any]) -> RecoveryAction:
        """
        Handle an error and determine recovery action

        Args:
            strategy_id: Strategy that encountered error
            error: The exception
            context: Context information (operation, data, etc.)

        Returns:
            RecoveryAction to take
        """
        # Classify error
        severity = self._classify_error(error)

        # Log error
        error_record = {
            "timestamp": datetime.now(timezone.utc),
            "error": str(error),
            "type": type(error).__name__,
            "severity": severity.value,
            "context": context,
        }

        if strategy_id not in self.error_history:
            self.error_history[strategy_id] = []

        self.error_history[strategy_id].append(error_record)

        # Log
        logger.error(f"Strategy {strategy_id} error ({severity.value}): {error}", exc_info=True)

        # Determine action based on severity
        if severity == ErrorSeverity.CRITICAL:
            return RecoveryAction.STOP

        elif severity == ErrorSeverity.HIGH:
            # Check circuit breaker
            if self._should_trip_circuit(strategy_id):
                return RecoveryAction.STOP
            return RecoveryAction.PAUSE

        elif severity == ErrorSeverity.MEDIUM:
            # Retry with backoff
            if context.get("retry_count", 0) < self.max_retries:
                return RecoveryAction.RETRY
            else:
                return RecoveryAction.PAUSE

        else:  # LOW
            return RecoveryAction.NOTIFY

    def _classify_error(self, error: Exception) -> ErrorSeverity:
        """Classify error severity"""
        error_type = type(error).__name__
        error_msg = str(error).lower()

        # Critical errors - must stop
        if any(x in error_type for x in ["RuntimeError", "SystemError"]):
            return ErrorSeverity.CRITICAL

        if "database" in error_msg or "connection" in error_msg:
            return ErrorSeverity.CRITICAL

        # High severity - pause strategy
        if any(x in error_type for x in ["ValueError", "KeyError"]):
            return ErrorSeverity.HIGH

        if "broker" in error_msg or "order" in error_msg:
            return ErrorSeverity.HIGH

        # Medium severity - retry
        if any(x in error_type for x in ["TimeoutError", "ConnectionError"]):
            return ErrorSeverity.MEDIUM

        if "timeout" in error_msg or "network" in error_msg:
            return ErrorSeverity.MEDIUM

        # Low severity - log and continue
        return ErrorSeverity.LOW

    def _should_trip_circuit(self, strategy_id: int) -> bool:
        """Check if circuit breaker should trip"""
        if strategy_id not in self.circuit_breakers:
            self.circuit_breakers[strategy_id] = CircuitBreaker()

        breaker = self.circuit_breakers[strategy_id]
        return breaker.should_trip(self.error_history[strategy_id])

    async def retry_with_backoff(self, operation, *args, max_retries: int = 3, **kwargs):
        """
        Retry an operation with exponential backoff

        Usage:
            result = await error_mgr.retry_with_backoff(
                broker.place_order,
                symbol='AAPL',
                side='buy',
                quantity=10
            )
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                return await operation(*args, **kwargs)

            except Exception as e:
                last_error = e

                if attempt < max_retries - 1:
                    delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                    logger.warning(f"Operation failed, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Operation failed after {max_retries} attempts")

        raise last_error

    def get_error_summary(self, strategy_id: int) -> Dict[str, Any]:
        """Get error summary for a strategy"""
        if strategy_id not in self.error_history:
            return {"total_errors": 0}

        errors = self.error_history[strategy_id]

        return {
            "total_errors": len(errors),
            "recent_errors": errors[-10:],
            "error_types": self._count_error_types(errors),
            "last_error": errors[-1] if errors else None,
        }

    def _count_error_types(self, errors: List[Dict]) -> Dict[str, int]:
        """Count errors by type"""
        counts = {}
        for error in errors:
            error_type = error["type"]
            counts[error_type] = counts.get(error_type, 0) + 1
        return counts
