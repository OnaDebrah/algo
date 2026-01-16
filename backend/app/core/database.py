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

        # Tables for options tracking
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS options_positions (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                strategy TEXT,
                entry_date TEXT,
                expiration TEXT,
                initial_cost REAL,
                status TEXT,
                pnl REAL
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS options_legs (
                id INTEGER PRIMARY KEY,
                position_id INTEGER,
                option_type TEXT,
                strike REAL,
                quantity INTEGER,
                premium REAL,
                FOREIGN KEY (position_id) REFERENCES options_positions(id)
            )
        """
        )

        # User settings table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER UNIQUE NOT NULL,
                data_source TEXT DEFAULT 'yahoo',
                slippage REAL DEFAULT 0.001,
                commission REAL DEFAULT 0.002,
                initial_capital REAL DEFAULT 100000.0,
                theme TEXT DEFAULT 'dark',
                notifications INTEGER DEFAULT 1,
                auto_refresh INTEGER DEFAULT 1,
                refresh_interval INTEGER DEFAULT 30,
                FOREIGN KEY (user_id) REFERENCES users(id)
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

    def save_performance(self, portfolio_id: int, equity: float, cash: float, total_return: float):
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

    def get_header_metrics(self, portfolio_id: int = 1) -> Dict:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # 1. Get Current Capital - Use a safer fetch
            cursor.execute("SELECT current_capital FROM portfolios WHERE id = ?", (portfolio_id,))
            row = cursor.fetchone()
            nav = row[0] if row else 0.0

            # 2. Get Total Exposure - Use COALESCE to handle NULL from SUM()
            cursor.execute(
                "SELECT COALESCE(SUM(entry_price * quantity), 0.0) FROM positions WHERE portfolio_id = ?",
                (portfolio_id,),
            )
            exposure = cursor.fetchone()[0]

            # 3. Get Unrealized PnL - Use COALESCE
            cursor.execute(
                "SELECT COALESCE(SUM(unrealized_pnl), 0.0) FROM positions WHERE portfolio_id = ?",
                (portfolio_id,),
            )
            unrealized = cursor.fetchone()[0]

            # 4. Get Previous NAV
            cursor.execute(
                """
                SELECT equity FROM performance_history
                WHERE portfolio_id = ? ORDER BY timestamp DESC LIMIT 1
            """,
                (portfolio_id,),
            )
            prev_row = cursor.fetchone()
            prev_nav = prev_row[0] if prev_row else nav

            return {
                "nav": nav,
                "prev_nav": prev_nav,
                "exposure": exposure,
                "unrealized_pnl": unrealized,
            }
        finally:
            conn.close()
