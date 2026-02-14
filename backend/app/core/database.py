"""
Database management for trade persistence (PostgreSQL version)
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

from backend.app.config import settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manage PostgreSQL database for trade persistence"""

    def __init__(self, db_url: str = None):
        """
        Initialize database manager with connection pooling

        Args:
            db_url: Database URL (defaults to settings.DATABASE_URL)
        """
        self.db_url = db_url or settings.DATABASE_URL

        # Convert asyncpg URL to psycopg2 format
        self.db_url = self.db_url.replace("postgresql+asyncpg://", "postgresql://")

        # Create connection pool for better performance
        try:
            self.connection_pool = pool.SimpleConnectionPool(minconn=1, maxconn=10, dsn=self.db_url)
            logger.info("Database connection pool created successfully")
        except Exception as e:
            logger.error(f"Failed to create connection pool: {e}")
            raise

        self.init_database()

    def get_connection(self):
        """Get a connection from the pool"""
        return self.connection_pool.getconn()

    def return_connection(self, conn):
        """Return a connection to the pool"""
        self.connection_pool.putconn(conn)

    def init_database(self):
        """Initialize database tables"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()

            # Trades table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    order_type VARCHAR(10) NOT NULL,
                    quantity NUMERIC NOT NULL,
                    price NUMERIC NOT NULL,
                    executed_at TIMESTAMP NOT NULL,
                    strategy VARCHAR(100) NOT NULL,
                    profit NUMERIC,
                    profit_pct NUMERIC,
                    commission NUMERIC,
                    slippage NUMERIC,
                    total_value NUMERIC NOT NULL,
                    side VARCHAR(10),
                    notes TEXT,
                    portfolio_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """
            )

            # Positions table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS positions (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    side VARCHAR(10) NOT NULL,
                    quantity NUMERIC NOT NULL,
                    entry_price NUMERIC NOT NULL,
                    current_price NUMERIC NOT NULL,
                    unrealized_pnl NUMERIC NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    portfolio_id INTEGER,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """
            )

            # Portfolio table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS portfolios (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    name VARCHAR(255) NOT NULL UNIQUE,
                    description TEXT,
                    initial_capital NUMERIC NOT NULL,
                    current_capital NUMERIC NOT NULL,
                    is_active BOOLEAN DEFAULT true,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """
            )

            # Performance history
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS performance_history (
                    id SERIAL PRIMARY KEY,
                    portfolio_id INTEGER REFERENCES portfolios(id),
                    timestamp TIMESTAMP NOT NULL,
                    equity NUMERIC NOT NULL,
                    cash NUMERIC NOT NULL,
                    total_return NUMERIC,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """
            )

            # Create index for performance queries
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_performance_portfolio_timestamp
                    ON performance_history(portfolio_id, timestamp DESC)
                """
            )

            # Tables for options tracking
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS options_positions (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20),
                    strategy VARCHAR(100),
                    entry_date TIMESTAMP,
                    expiration TIMESTAMP,
                    initial_cost NUMERIC,
                    status VARCHAR(20),
                    pnl NUMERIC,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS options_legs (
                    id SERIAL PRIMARY KEY,
                    position_id INTEGER REFERENCES options_positions(id),
                    option_type VARCHAR(10),
                    strike NUMERIC,
                    quantity INTEGER,
                    premium NUMERIC,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """
            )

            # User settings table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS user_settings (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER UNIQUE NOT NULL,

                    -- Backtest settings
                    data_source VARCHAR(50) DEFAULT 'yahoo',
                    slippage NUMERIC DEFAULT 0.001,
                    commission NUMERIC DEFAULT 0.002,
                    initial_capital NUMERIC DEFAULT 10000.0,

                    -- Live trading settings
                    live_data_source VARCHAR(50) DEFAULT 'alpaca',
                    default_broker VARCHAR(50) DEFAULT 'paper',
                    auto_connect_broker BOOLEAN DEFAULT false,

                    -- Broker credentials (consider encryption for production)
                    broker_api_key VARCHAR(255),
                    broker_api_secret VARCHAR(255),
                    broker_base_url VARCHAR(255),

                    broker_host VARCHAR(255),
                    broker_port INTEGER,
                    broker_client_id INTEGER,
                    user_ib_account_id VARCHAR(255),

                    -- General settings
                    theme VARCHAR(20) DEFAULT 'dark',
                    notifications BOOLEAN DEFAULT true,
                    auto_refresh BOOLEAN DEFAULT true,
                    refresh_interval INTEGER DEFAULT 30,

                    -- Timestamps
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    -- Foreign key constraint
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                    )
                """
            )

            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_portfolio ON trades(portfolio_id, executed_at DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_portfolio ON positions(portfolio_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_settings_user_id ON user_settings(user_id)")

            conn.commit()
            logger.info("Database tables initialized successfully")
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to initialize database: {e}")
            raise
        finally:
            cursor.close()
            self.return_connection(conn)

    def save_trade(self, trade_data: Dict, portfolio_id: int = 1):
        """Save trade to database"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO trades (
                    symbol, order_type, quantity, price, executed_at,
                    strategy, profit, profit_pct, commission, slippage,
                    total_value, side, notes, portfolio_id
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """,
                (
                    trade_data["symbol"],
                    trade_data["order_type"],
                    trade_data["quantity"],
                    trade_data["price"],
                    trade_data.get("executed_at") or trade_data.get("timestamp"),
                    trade_data["strategy"],
                    trade_data.get("profit"),
                    trade_data.get("profit_pct"),
                    trade_data.get("commission"),
                    trade_data.get("slippage"),
                    trade_data.get("total_value") or (trade_data["quantity"] * trade_data["price"]),
                    trade_data.get("side") or trade_data.get("order_type"),
                    trade_data.get("notes"),
                    trade_data.get("portfolio_id") or portfolio_id,
                ),
            )

            trade_id = cursor.fetchone()[0]
            conn.commit()
            logger.debug(f"Saved trade ID: {trade_id}")
            return trade_id
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to save trade: {e}")
            raise
        finally:
            cursor.close()
            self.return_connection(conn)

    def save_trades_bulk(self, trades: List[Dict], portfolio_id: int):
        """
        Save multiple trades in a single transaction for better performance
        """
        if not trades:
            return

        conn = self.get_connection()
        try:
            cursor = conn.cursor()

            # Prepare the list of tuples for executemany
            trade_tuples = [
                (
                    t.get("symbol"),
                    t.get("order_type"),
                    t.get("quantity"),
                    t.get("price"),
                    t.get("executed_at"),
                    t.get("strategy"),
                    t.get("profit"),
                    t.get("profit_pct"),
                    t.get("commission"),
                    t.get("slippage"),
                    t.get("total_value"),
                    t.get("side"),
                    t.get("notes"),
                    portfolio_id,
                )
                for t in trades
            ]

            cursor.executemany(
                """
                INSERT INTO trades (
                    symbol, order_type, quantity, price, executed_at,
                    strategy, profit, profit_pct, commission, slippage,
                    total_value, side, notes, portfolio_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                trade_tuples,
            )
            conn.commit()
            logger.info(f"Bulk saved {len(trades)} trades")
        except Exception as e:
            conn.rollback()
            logger.error(f"Error bulk saving trades: {e}")
            raise
        finally:
            cursor.close()
            self.return_connection(conn)

    def get_trades(self, portfolio_id: int = 1, limit: int = 100, start_date: datetime = None, end_date: datetime = None) -> List[Dict]:
        """Retrieve trades from database"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            query = "SELECT * FROM trades WHERE portfolio_id = %s"
            params = [portfolio_id]

            if start_date:
                query += " AND executed_at >= %s"
                params.append(start_date)
            if end_date:
                query += " AND executed_at <= %s"
                params.append(end_date)

            query += " ORDER BY executed_at DESC LIMIT %s"
            params.append(limit)

            cursor.execute(query, params)
            trades = cursor.fetchall()

            return [dict(row) for row in trades]
        finally:
            cursor.close()
            self.return_connection(conn)

    def get_equity_curve(self, portfolio_id: int, start_date: datetime = None, end_date: datetime = None) -> List[Dict]:
        """Retrieve equity curve from database"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            query = "SELECT * FROM performance_history WHERE portfolio_id = %s"
            params = [portfolio_id]

            if start_date:
                query += " AND timestamp >= %s"
                params.append(start_date)
            if end_date:
                query += " AND timestamp <= %s"
                params.append(end_date)

            query += " ORDER BY timestamp ASC"

            cursor.execute(query, params)
            curve = cursor.fetchall()

            return [dict(row) for row in curve]
        finally:
            cursor.close()
            self.return_connection(conn)

    def save_performance(self, portfolio_id: int, equity: float, cash: float, total_return: float):
        """Save performance snapshot"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO performance_history
                    (portfolio_id, timestamp, equity, cash, total_return)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (portfolio_id, datetime.now(), equity, cash, total_return),
            )

            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to save performance: {e}")
            raise
        finally:
            cursor.close()
            self.return_connection(conn)

    def save_performance_bulk(self, portfolio_id: int, performance_data: List[Dict]):
        """Save multiple performance snapshots in bulk"""
        if not performance_data:
            return

        conn = self.get_connection()
        try:
            cursor = conn.cursor()

            data_tuples = [
                (
                    portfolio_id,
                    p.get("timestamp"),
                    p.get("equity"),
                    p.get("cash"),
                    p.get("total_return"),
                )
                for p in performance_data
            ]

            cursor.executemany(
                """
                INSERT INTO performance_history
                    (portfolio_id, timestamp, equity, cash, total_return)
                VALUES (%s, %s, %s, %s, %s)
                """,
                data_tuples,
            )
            conn.commit()
            logger.info(f"Bulk saved {len(performance_data)} performance snapshots")
        except Exception as e:
            conn.rollback()
            logger.error(f"Error bulk saving performance: {e}")
            raise
        finally:
            cursor.close()
            self.return_connection(conn)

    def create_portfolio(self, name: str, initial_capital: float, user_id: int = 1) -> int:
        """Create new portfolio"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO portfolios
                    (user_id, name, initial_capital, current_capital, created_at)
                VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """,
                (user_id, name, initial_capital, initial_capital, datetime.now()),
            )

            portfolio_id = cursor.fetchone()[0]
            conn.commit()
            logger.info(f"Created portfolio: {name} (ID: {portfolio_id})")
            return portfolio_id
        except psycopg2.IntegrityError:
            # Portfolio already exists
            conn.rollback()
            cursor.execute("SELECT id FROM portfolios WHERE name = %s", (name,))
            portfolio_id = cursor.fetchone()[0]
            logger.info(f"Portfolio {name} already exists (ID: {portfolio_id})")
            return portfolio_id
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to create portfolio: {e}")
            raise
        finally:
            cursor.close()
            self.return_connection(conn)

    def get_portfolio(self, portfolio_id: int) -> Optional[Dict]:
        """Get portfolio by ID"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute("SELECT * FROM portfolios WHERE id = %s", (portfolio_id,))
            row = cursor.fetchone()

            return dict(row) if row else None
        finally:
            cursor.close()
            self.return_connection(conn)

    def update_portfolio_capital(self, portfolio_id: int, current_capital: float):
        """Update portfolio current capital"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE portfolios
                SET current_capital = %s, updated_at = %s
                WHERE id = %s
                """,
                (current_capital, datetime.now(), portfolio_id),
            )

            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to update portfolio capital: {e}")
            raise
        finally:
            cursor.close()
            self.return_connection(conn)

    def get_header_metrics(self, portfolio_id: int = 1) -> Dict:
        """Get header metrics for dashboard"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # 1. Get Current Capital
            cursor.execute("SELECT current_capital FROM portfolios WHERE id = %s", (portfolio_id,))
            row = cursor.fetchone()
            nav = row["current_capital"] if row else 0.0

            # 2. Get Total Exposure
            cursor.execute(
                "SELECT COALESCE(SUM(entry_price * quantity), 0.0) as exposure FROM positions WHERE portfolio_id = %s",
                (portfolio_id,),
            )
            exposure = cursor.fetchone()["exposure"]

            # 3. Get Unrealized PnL
            cursor.execute(
                "SELECT COALESCE(SUM(unrealized_pnl), 0.0) as unrealized FROM positions WHERE portfolio_id = %s",
                (portfolio_id,),
            )
            unrealized = cursor.fetchone()["unrealized"]

            # 4. Get Previous NAV
            cursor.execute(
                """
                SELECT equity FROM performance_history
                WHERE portfolio_id = %s
                ORDER BY timestamp DESC
                    LIMIT 1
                """,
                (portfolio_id,),
            )
            prev_row = cursor.fetchone()
            prev_nav = prev_row["equity"] if prev_row else nav

            return {
                "nav": float(nav),
                "prev_nav": float(prev_nav),
                "exposure": float(exposure),
                "unrealized_pnl": float(unrealized),
            }
        finally:
            cursor.close()
            self.return_connection(conn)

    def close(self):
        """Close all connections in the pool"""
        if hasattr(self, "connection_pool"):
            self.connection_pool.closeall()
            logger.info("Database connection pool closed")

    def __del__(self):
        """Cleanup on deletion"""
        self.close()
