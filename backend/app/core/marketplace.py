"""
Strategy Marketplace with Complete Backtest Storage
Stores not just strategy parameters, but complete backtest results,
equity curves, trade history, and performance analytics.
"""

import json
import pickle
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class StrategyReview:
    """User review of a strategy"""

    id: Optional[int] = None
    strategy_id: int = 0
    user_id: int = 0
    username: str = ""
    rating: int = 5
    review_text: str = ""
    performance_achieved: Dict = None
    created_at: datetime = None

    def __post_init__(self):
        if self.performance_achieved is None:
            self.performance_achieved = {}
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class BacktestResults:
    """Complete backtest results for a strategy"""

    # Core metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    calmar_ratio: float = 0.0

    # Trading metrics
    num_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade_duration: float = 0.0

    # Risk metrics
    volatility: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0

    # Time series data (serialized as JSON/pickle)
    equity_curve: List[Dict] = None  # [{date, equity, drawdown}]
    trades: List[Dict] = None  # [{entry_date, exit_date, pnl, return, ...}]
    daily_returns: List[Dict] = None  # [{date, return}]

    # Backtest configuration
    start_date: datetime = None
    end_date: datetime = None
    initial_capital: float = 100000.0
    symbols: List[str] = None

    def __post_init__(self):
        if self.equity_curve is None:
            self.equity_curve = []
        if self.trades is None:
            self.trades = []
        if self.daily_returns is None:
            self.daily_returns = []
        if self.symbols is None:
            self.symbols = []

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "BacktestResults":
        """Create from dictionary"""
        return cls(**data)


@dataclass
class StrategyListing:
    """Strategy listing with backtest data"""

    id: Optional[int] = None
    name: str = ""
    description: str = ""
    creator_id: int = 0
    creator_name: str = ""
    strategy_type: str = ""
    category: str = ""
    complexity: str = ""

    # Strategy configuration
    parameters: Dict = None

    # Backtest results (complete)
    backtest_results: BacktestResults = None

    # Quick access metrics (denormalized for filtering)
    sharpe_ratio: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    num_trades: int = 0

    # Marketplace specific
    price: float = 0.0
    is_public: bool = True
    is_verified: bool = False  # Admin verified performance
    version: str = "1.0.0"
    tags: List[str] = None

    # Social features
    downloads: int = 0
    rating: float = 0.0
    num_ratings: int = 0
    num_reviews: int = 0

    # Timestamps
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.tags is None:
            self.tags = []
        if self.backtest_results is None:
            self.backtest_results = BacktestResults()
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


class StrategyMarketplace:
    """
    Marketplace with complete backtest storage
    """

    def __init__(self, db_path: str = "trading_platform.db"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize enhanced database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Main strategy listings table (with quick access metrics)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS marketplace_strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                creator_id INTEGER NOT NULL,
                creator_name TEXT NOT NULL,
                strategy_type TEXT NOT NULL,
                category TEXT NOT NULL,
                complexity TEXT NOT NULL,

                -- Strategy configuration
                parameters TEXT NOT NULL,

                -- Quick access performance metrics (for filtering)
                sharpe_ratio REAL DEFAULT 0.0,
                total_return REAL DEFAULT 0.0,
                max_drawdown REAL DEFAULT 0.0,
                win_rate REAL DEFAULT 0.0,
                num_trades INTEGER DEFAULT 0,

                -- Marketplace
                price REAL DEFAULT 0.0,
                is_public BOOLEAN DEFAULT 1,
                is_verified BOOLEAN DEFAULT 0,
                version TEXT DEFAULT '1.0.0',
                tags TEXT,

                -- Social
                downloads INTEGER DEFAULT 0,
                rating REAL DEFAULT 0.0,
                num_ratings INTEGER DEFAULT 0,
                num_reviews INTEGER DEFAULT 0,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (creator_id) REFERENCES users(id)
            )
        """)

        # Backtest results table (stores complete backtest data)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_backtests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id INTEGER NOT NULL,
                version TEXT NOT NULL,

                -- Complete backtest results (JSON)
                backtest_data TEXT NOT NULL,  -- Serialized BacktestResults

                -- Time series data (can be large, stored separately)
                equity_curve BLOB,  -- Pickled pandas DataFrame
                trades_history BLOB,  -- Pickled pandas DataFrame
                daily_returns BLOB,  -- Pickled pandas DataFrame

                -- Backtest metadata
                start_date TEXT,
                end_date TEXT,
                initial_capital REAL,
                symbols TEXT,  -- JSON array

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (strategy_id) REFERENCES marketplace_strategies(id)
            )
        """)

        # Keep existing tables for reviews, downloads, favorites
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                username TEXT NOT NULL,
                rating INTEGER NOT NULL CHECK(rating >= 1 AND rating <= 5),
                review_text TEXT,
                performance_achieved TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (strategy_id) REFERENCES marketplace_strategies(id),
                FOREIGN KEY (user_id) REFERENCES users(id),
                UNIQUE(strategy_id, user_id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_downloads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                downloaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (strategy_id) REFERENCES marketplace_strategies(id),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_favorites (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (strategy_id) REFERENCES marketplace_strategies(id),
                FOREIGN KEY (user_id) REFERENCES users(id),
                UNIQUE(strategy_id, user_id)
            )
        """)

        conn.commit()
        conn.close()

    # ============================================================
    # PUBLISHING WITH BACKTEST RESULTS
    # ============================================================

    def publish_strategy_with_backtest(
        self, listing: StrategyListing, equity_curve_df: pd.DataFrame = None, trades_df: pd.DataFrame = None, daily_returns_df: pd.DataFrame = None
    ) -> int:
        """
        Publish strategy WITH complete backtest results

        Args:
            listing: StrategyListing with backtest_results populated
            equity_curve_df: DataFrame with columns [date, equity, drawdown]
            trades_df: DataFrame with trade history
            daily_returns_df: DataFrame with daily returns

        Returns:
            Strategy ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # 1. Insert strategy listing with quick-access metrics
            cursor.execute(
                """
                           INSERT INTO marketplace_strategies (name, description, creator_id, creator_name,
                                                                  strategy_type,
                                                                  category, complexity, parameters,
                                                                  sharpe_ratio, total_return, max_drawdown, win_rate,
                                                                  num_trades,
                                                                  price, is_public, version, tags)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                           """,
                (
                    listing.name,
                    listing.description,
                    listing.creator_id,
                    listing.creator_name,
                    listing.strategy_type,
                    listing.category,
                    listing.complexity,
                    json.dumps(listing.parameters),
                    listing.backtest_results.sharpe_ratio,
                    listing.backtest_results.total_return,
                    listing.backtest_results.max_drawdown,
                    listing.backtest_results.win_rate,
                    listing.backtest_results.num_trades,
                    listing.price,
                    listing.is_public,
                    listing.version,
                    json.dumps(listing.tags),
                ),
            )

            strategy_id = cursor.lastrowid

            # 2. Store complete backtest results
            backtest_data = json.dumps(listing.backtest_results.to_dict())

            # Serialize dataframes efficiently
            equity_blob = pickle.dumps(equity_curve_df) if equity_curve_df is not None else None
            trades_blob = pickle.dumps(trades_df) if trades_df is not None else None
            returns_blob = pickle.dumps(daily_returns_df) if daily_returns_df is not None else None

            cursor.execute(
                """
                           INSERT INTO strategy_backtests (strategy_id, version, backtest_data,
                                                           equity_curve, trades_history, daily_returns,
                                                           start_date, end_date, initial_capital, symbols)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                           """,
                (
                    strategy_id,
                    listing.version,
                    backtest_data,
                    equity_blob,
                    trades_blob,
                    returns_blob,
                    listing.backtest_results.start_date.isoformat() if listing.backtest_results.start_date else None,
                    listing.backtest_results.end_date.isoformat() if listing.backtest_results.end_date else None,
                    listing.backtest_results.initial_capital,
                    json.dumps(listing.backtest_results.symbols),
                ),
            )

            conn.commit()
            return strategy_id

        except Exception as e:
            conn.rollback()
            print(f"Failed to publish strategy: {e}")
            raise
        finally:
            conn.close()

    def get_strategy_backtest(self, strategy_id: int) -> Optional[Dict]:
        """
        Get complete backtest results for a strategy

        Returns:
            Dictionary with:
                - backtest_results: BacktestResults object
                - equity_curve: pandas DataFrame
                - trades: pandas DataFrame
                - daily_returns: pandas DataFrame
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
                       SELECT *
                       FROM strategy_backtests
                       WHERE strategy_id = ?
                       ORDER BY created_at DESC LIMIT 1
                       """,
            (strategy_id,),
        )

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        # Deserialize backtest data
        backtest_data = json.loads(row["backtest_data"])
        backtest_results = BacktestResults.from_dict(backtest_data)

        # Deserialize dataframes
        equity_curve = pickle.loads(row["equity_curve"]) if row["equity_curve"] else None
        trades = pickle.loads(row["trades_history"]) if row["trades_history"] else None
        daily_returns = pickle.loads(row["daily_returns"]) if row["daily_returns"] else None

        return {"backtest_results": backtest_results, "equity_curve": equity_curve, "trades": trades, "daily_returns": daily_returns}

    # ============================================================
    # ENHANCED BROWSING WITH PERFORMANCE FILTERS
    # ============================================================

    def browse_strategies(
        self,
        category: Optional[str] = None,
        complexity: Optional[str] = None,
        min_sharpe: Optional[float] = None,
        min_return: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        min_win_rate: Optional[float] = None,
        search_query: Optional[str] = None,
        sort_by: str = "sharpe_ratio",
        limit: int = 50,
        offset: int = 0,
    ) -> List[StrategyListing]:
        """
        Browse strategies with performance-based filters
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM marketplace_strategies WHERE is_public = 1"
        params = []

        if category:
            query += " AND category = ?"
            params.append(category)

        if complexity:
            query += " AND complexity = ?"
            params.append(complexity)

        if min_sharpe is not None:
            query += " AND sharpe_ratio >= ?"
            params.append(min_sharpe)

        if min_return is not None:
            query += " AND total_return >= ?"
            params.append(min_return)

        if max_drawdown is not None:
            query += " AND max_drawdown >= ?"  # Note: drawdown is negative
            params.append(max_drawdown)

        if min_win_rate is not None:
            query += " AND win_rate >= ?"
            params.append(min_win_rate)

        if search_query:
            query += " AND (name LIKE ? OR description LIKE ?)"
            params.extend([f"%{search_query}%", f"%{search_query}%"])

        # Sorting options
        sort_columns = {
            "sharpe_ratio": "sharpe_ratio DESC",
            "total_return": "total_return DESC",
            "rating": "rating DESC",
            "downloads": "downloads DESC",
            "created_at": "created_at DESC",
        }
        query += f" ORDER BY {sort_columns.get(sort_by, 'sharpe_ratio DESC')}"
        query += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        strategies = []
        for row in rows:
            # Get backtest results for this strategy
            backtest_data = self.get_strategy_backtest(row["id"])

            listing = StrategyListing(
                id=row["id"],
                name=row["name"],
                description=row["description"],
                creator_id=row["creator_id"],
                creator_name=row["creator_name"],
                strategy_type=row["strategy_type"],
                category=row["category"],
                complexity=row["complexity"],
                parameters=json.loads(row["parameters"]),
                backtest_results=backtest_data["backtest_results"] if backtest_data else BacktestResults(),
                sharpe_ratio=row["sharpe_ratio"],
                total_return=row["total_return"],
                max_drawdown=row["max_drawdown"],
                win_rate=row["win_rate"],
                num_trades=row["num_trades"],
                price=row["price"],
                is_public=bool(row["is_public"]),
                is_verified=bool(row["is_verified"]),
                version=row["version"],
                tags=json.loads(row["tags"]) if row["tags"] else [],
                downloads=row["downloads"],
                rating=row["rating"],
                num_ratings=row["num_ratings"],
                num_reviews=row["num_reviews"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )
            strategies.append(listing)

        return strategies

    # ============================================================
    # STRATEGY COMPARISON
    # ============================================================

    def compare_strategies(self, strategy_ids: List[int]) -> pd.DataFrame:
        """
        Compare multiple strategies side-by-side

        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []

        for strategy_id in strategy_ids:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                           SELECT *
                           FROM marketplace_strategies
                           WHERE id = ?
                           """,
                (strategy_id,),
            )

            row = cursor.fetchone()
            conn.close()

            if row:
                comparison_data.append(
                    {
                        "Strategy": row["name"],
                        "Sharpe Ratio": row["sharpe_ratio"],
                        "Total Return": f"{row['total_return']:.2f}%",
                        "Max Drawdown": f"{row['max_drawdown']:.2f}%",
                        "Win Rate": f"{row['win_rate']:.2f}%",
                        "Num Trades": row["num_trades"],
                        "Rating": f"{row['rating']:.1f}⭐",
                        "Downloads": row["downloads"],
                    }
                )

        return pd.DataFrame(comparison_data)

    # ============================================================
    # EXPORT BACKTEST DATA
    # ============================================================

    def export_backtest_data(self, strategy_id: int, format: str = "json") -> Optional[str]:
        """
        Export backtest data for external analysis

        Args:
            strategy_id: Strategy ID
            format: 'json', 'csv', or 'pickle'

        Returns:
            Serialized data or file path
        """
        backtest = self.get_strategy_backtest(strategy_id)

        if not backtest:
            return None

        if format == "json":
            return json.dumps(
                {
                    "backtest_results": backtest["backtest_results"].to_dict(),
                    "equity_curve": backtest["equity_curve"].to_dict("records") if backtest["equity_curve"] is not None else None,
                    "trades": backtest["trades"].to_dict("records") if backtest["trades"] is not None else None,
                    "daily_returns": backtest["daily_returns"].to_dict("records") if backtest["daily_returns"] is not None else None,
                }
            )

        elif format == "csv":
            # Would write to file or return CSV strings
            pass

        elif format == "pickle":
            return pickle.dumps(backtest)

        return None


# Helper function to convert backtest engine results to BacktestResults
def convert_engine_results_to_backtest(engine_results: Dict) -> BacktestResults:
    """
    Convert your trading engine backtest results to BacktestResults format

    Args:
        engine_results: Output from your TradingEngine.backtest()

    Returns:
        BacktestResults object
    """
    return BacktestResults(
        total_return=engine_results.get("total_return", 0),
        annualized_return=engine_results.get("annualized_return", 0),
        sharpe_ratio=engine_results.get("sharpe_ratio", 0),
        sortino_ratio=engine_results.get("sortino_ratio", 0),
        max_drawdown=engine_results.get("max_drawdown", 0),
        max_drawdown_duration=engine_results.get("max_drawdown_duration", 0),
        calmar_ratio=engine_results.get("calmar_ratio", 0),
        num_trades=engine_results.get("num_trades", 0),
        win_rate=engine_results.get("win_rate", 0),
        profit_factor=engine_results.get("profit_factor", 0),
        avg_win=engine_results.get("avg_win", 0),
        avg_loss=engine_results.get("avg_loss", 0),
        volatility=engine_results.get("volatility", 0),
        var_95=engine_results.get("var_95", 0),
        cvar_95=engine_results.get("cvar_95", 0),
        equity_curve=engine_results.get("equity_curve", []),
        trades=engine_results.get("trades", []),
        daily_returns=engine_results.get("daily_returns", []),
        start_date=engine_results.get("start_date"),
        end_date=engine_results.get("end_date"),
        initial_capital=engine_results.get("initial_capital", 100000),
        symbols=engine_results.get("symbols", []),
    )
