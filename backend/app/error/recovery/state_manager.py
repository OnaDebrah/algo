import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class StateManager:
    """
    Persistent state management for strategies

    Features:
    - State snapshots
    - Crash recovery
    - State versioning
    - Rollback capability
    """

    def __init__(self, state_dir: str = "./state"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # In-memory state cache
        self.state_cache: Dict[int, Dict[str, Any]] = {}

    async def save_state(self, strategy_id: int, state: Dict[str, Any]):
        """
        Save strategy state to disk

        State includes:
        - Current positions
        - Open orders
        - Equity history
        - Strategy parameters
        - Internal state variables
        """
        # Add metadata
        state_snapshot = {"strategy_id": strategy_id, "timestamp": datetime.utcnow().isoformat(), "version": 1, "state": state}

        # Save to disk (both JSON and pickle for redundancy)
        state_file = self.state_dir / f"strategy_{strategy_id}.json"
        pickle_file = self.state_dir / f"strategy_{strategy_id}.pkl"

        try:
            # JSON (human-readable)
            with open(state_file, "w") as f:
                json.dump(state_snapshot, f, indent=2, default=str)

            # Pickle (full Python objects)
            with open(pickle_file, "wb") as f:
                pickle.dump(state_snapshot, f)

            # Update cache
            self.state_cache[strategy_id] = state

            logger.debug(f"Saved state for strategy {strategy_id}")

        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    async def load_state(self, strategy_id: int) -> Optional[Dict[str, Any]]:
        """
        Load strategy state from disk

        Returns:
            State dict or None if not found
        """
        # Try pickle first (more reliable)
        pickle_file = self.state_dir / f"strategy_{strategy_id}.pkl"

        if pickle_file.exists():
            try:
                with open(pickle_file, "rb") as f:
                    snapshot = pickle.load(f)

                logger.info(f"Loaded state for strategy {strategy_id} from pickle")
                return snapshot["state"]

            except Exception as e:
                logger.warning(f"Failed to load pickle state: {e}")

        # Fallback to JSON
        state_file = self.state_dir / f"strategy_{strategy_id}.json"

        if state_file.exists():
            try:
                with open(state_file, "r") as f:
                    snapshot = json.load(f)

                logger.info(f"Loaded state for strategy {strategy_id} from JSON")
                return snapshot["state"]

            except Exception as e:
                logger.error(f"Failed to load JSON state: {e}")

        return None

    async def delete_state(self, strategy_id: int):
        """Delete strategy state"""
        state_file = self.state_dir / f"strategy_{strategy_id}.json"
        pickle_file = self.state_dir / f"strategy_{strategy_id}.pkl"

        if state_file.exists():
            state_file.unlink()

        if pickle_file.exists():
            pickle_file.unlink()

        if strategy_id in self.state_cache:
            del self.state_cache[strategy_id]

        logger.info(f"Deleted state for strategy {strategy_id}")

    async def create_checkpoint(self, strategy_id: int, label: str = "auto"):
        """
        Create a checkpoint of current state

        Useful before risky operations
        """
        if strategy_id not in self.state_cache:
            logger.warning(f"No state to checkpoint for strategy {strategy_id}")
            return

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = self.state_dir / f"strategy_{strategy_id}_checkpoint_{label}_{timestamp}.pkl"

        try:
            with open(checkpoint_file, "wb") as f:
                pickle.dump(self.state_cache[strategy_id], f)

            logger.info(f"Created checkpoint: {checkpoint_file.name}")

        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
