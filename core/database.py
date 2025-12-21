"""
Database management for trade persistence
"""

import logging
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional

from config import DATABASE_PATH

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manage SQLite database for trade persistence"""

    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Trades table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                order_type TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                timestamp TEXT NOT NULL,
                strategy TEXT NOT NULL,
                profit REAL,
                profit_pct REAL,
                portfolio_id INTEGER
            )
        """
        )

        # Positions table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                current_price REAL NOT NULL,
                unrealized_pnl REAL NOT NULL,
                entry_time TEXT NOT NULL,
                portfolio_id INTEGER
            )
        """
        )

        # Portfolio table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                initial_capital REAL NOT NULL,
                current_capital REAL NOT NULL,
                created_at TEXT NOT NULL
            )
        """
        )

        # Performance history
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS performance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id INTEGER,
                timestamp TEXT NOT NULL,
                equity REAL NOT NULL,
                cash REAL NOT NULL,
                total_return REAL
            )
        """
        )

        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")

    def save_trade(self, trade_data: Dict, portfolio_id: int = 1):
        """Save trade to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO trades (symbol, order_type, quantity, price, timestamp,
                              strategy, profit, profit_pct, portfolio_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                trade_data["symbol"],
                trade_data["order_type"],
                trade_data["quantity"],
                trade_data["price"],
                trade_data["timestamp"],
                trade_data["strategy"],
                trade_data.get("profit"),
                trade_data.get("profit_pct"),
                portfolio_id,
            ),
        )

        conn.commit()
        conn.close()

    def get_trades(self, portfolio_id: int = 1, limit: int = 100) -> List[Dict]:
        """Retrieve trades from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM trades WHERE portfolio_id = ?
            ORDER BY timestamp DESC LIMIT ?
        """,
            (portfolio_id, limit),
        )

        columns = [desc[0] for desc in cursor.description]
        trades = [dict(zip(columns, row)) for row in cursor.fetchall()]

        conn.close()
        return trades

    def save_performance(
        self, portfolio_id: int, equity: float, cash: float, total_return: float
    ):
        """Save performance snapshot"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO performance_history
            (portfolio_id, timestamp, equity, cash, total_return)
            VALUES (?, ?, ?, ?, ?)
        """,
            (portfolio_id, datetime.now().isoformat(), equity, cash, total_return),
        )

        conn.commit()
        conn.close()

    def create_portfolio(self, name: str, initial_capital: float) -> int:
        """Create new portfolio"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO portfolios
                (name, initial_capital, current_capital, created_at)
                VALUES (?, ?, ?, ?)
            """,
                (name, initial_capital, initial_capital, datetime.now().isoformat()),
            )

            portfolio_id = cursor.lastrowid
            conn.commit()
            logger.info(f"Created portfolio: {name} (ID: {portfolio_id})")
        except sqlite3.IntegrityError:
            # Portfolio already exists
            cursor.execute("SELECT id FROM portfolios WHERE name = ?", (name,))
            portfolio_id = cursor.fetchone()[0]
            logger.info(f"Portfolio {name} already exists (ID: {portfolio_id})")

        conn.close()
        return portfolio_id

    def get_portfolio(self, portfolio_id: int) -> Optional[Dict]:
        """Get portfolio by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM portfolios WHERE id = ?", (portfolio_id,))
        row = cursor.fetchone()

        if row:
            columns = [desc[0] for desc in cursor.description]
            portfolio = dict(zip(columns, row))
        else:
            portfolio = None

        conn.close()
        return portfolio
