"""
Strategy Marketplace - Complete Implementation
Allows users to share, discover, rate, and clone trading strategies

Files structure:
- core/marketplace.py: Core marketplace logic
- pages/6_🏪_Marketplace.py: Streamlit UI
- Database schema additions
"""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class StrategyListing:
    """Strategy listing in marketplace"""

    id: Optional[int] = None
    name: str = ""
    description: str = ""
    creator_id: int = 0
    creator_name: str = ""
    strategy_type: str = ""  # From catalog
    category: str = ""
    complexity: str = ""
    parameters: Dict = None
    performance_metrics: Dict = None

    # Marketplace specific
    price: float = 0.0  # 0 = free
    is_public: bool = True
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
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


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


class StrategyMarketplace:
    """
    Strategy Marketplace Manager
    Handles strategy sharing, discovery, ratings, and cloning
    """

    def __init__(self, db_path: str = "trading_platform.db"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize marketplace database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Strategy listings table
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
               parameters TEXT NOT NULL,
               performance_metrics TEXT,
               price REAL DEFAULT 0.0,
               is_public BOOLEAN DEFAULT 1,
               version TEXT DEFAULT '1.0.0',
               tags TEXT,
               downloads INTEGER DEFAULT 0,
               rating REAL DEFAULT 0.0,
               num_ratings INTEGER DEFAULT 0,
               num_reviews INTEGER DEFAULT 0,
               created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
               updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
               FOREIGN KEY (creator_id) REFERENCES users(id)
               )
           """)

        # Strategy reviews table
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

        # User strategy downloads table
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

        # User favorites table
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
    # LISTING STRATEGIES
    # ============================================================

    def publish_strategy(self, listing: StrategyListing) -> int:
        """
        Publish a strategy to the marketplace

        Returns:
            Strategy ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
                       INSERT INTO marketplace_strategies (name, description, creator_id, creator_name, strategy_type,
                                                           category, complexity, parameters, performance_metrics,
                                                           price, is_public, version, tags)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                json.dumps(listing.performance_metrics),
                listing.price,
                listing.is_public,
                listing.version,
                json.dumps(listing.tags),
            ),
        )

        strategy_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return strategy_id

    def update_strategy(self, strategy_id: int, updates: Dict) -> bool:
        """Update a strategy listing"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Build update query dynamically
        update_fields = []
        values = []

        allowed_fields = ["name", "description", "parameters", "performance_metrics", "price", "is_public", "version", "tags"]

        for field, value in updates.items():
            if field in allowed_fields:
                update_fields.append(f"{field} = ?")
                if field in ["parameters", "performance_metrics", "tags"]:
                    values.append(json.dumps(value))
                else:
                    values.append(value)

        if not update_fields:
            return False

        update_fields.append("updated_at = CURRENT_TIMESTAMP")
        values.append(strategy_id)

        query = f"UPDATE marketplace_strategies SET {', '.join(update_fields)} WHERE id = ?"
        cursor.execute(query, values)

        success = cursor.rowcount > 0
        conn.commit()
        conn.close()

        return success

    def delete_strategy(self, strategy_id: int, user_id: int) -> bool:
        """Delete a strategy (only by creator)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM marketplace_strategies WHERE id = ? AND creator_id = ?", (strategy_id, user_id))

        success = cursor.rowcount > 0
        conn.commit()
        conn.close()

        return success

    # ============================================================
    # BROWSING AND DISCOVERY
    # ============================================================

    def browse_strategies(
        self,
        category: Optional[str] = None,
        complexity: Optional[str] = None,
        tags: Optional[List[str]] = None,
        search_query: Optional[str] = None,
        sort_by: str = "rating",
        limit: int = 50,
        offset: int = 0,
    ) -> List[StrategyListing]:
        """
        Browse marketplace strategies with filters

        Args:
            category: Filter by category
            complexity: Filter by complexity level
            tags: Filter by tags (AND logic)
            search_query: Search in name and description
            sort_by: rating, downloads, created_at, updated_at
            limit: Max results
            offset: Pagination offset
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

        if search_query:
            query += " AND (name LIKE ? OR description LIKE ?)"
            params.extend([f"%{search_query}%", f"%{search_query}%"])

        # Tag filtering (basic - checks if tag exists in JSON array)
        if tags:
            for tag in tags:
                query += " AND tags LIKE ?"
                params.append(f'%"{tag}"%')

        # Sorting
        sort_columns = {
            "rating": "rating DESC",
            "downloads": "downloads DESC",
            "created_at": "created_at DESC",
            "updated_at": "updated_at DESC",
            "name": "name ASC",
        }
        query += f" ORDER BY {sort_columns.get(sort_by, 'rating DESC')}"
        query += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, params)
        rows = cursor.fetchall()

        strategies = []
        for row in rows:
            listing = self._row_to_listing(row)
            strategies.append(listing)

        conn.close()
        return strategies

    def get_strategy(self, strategy_id: int) -> Optional[StrategyListing]:
        """Get a specific strategy by ID"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM marketplace_strategies WHERE id = ?", (strategy_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return self._row_to_listing(row)
        return None

    def get_user_strategies(self, user_id: int) -> List[StrategyListing]:
        """Get all strategies created by a user"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM marketplace_strategies WHERE creator_id = ? ORDER BY created_at DESC", (user_id,))
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_listing(row) for row in rows]

    def get_trending_strategies(self, days: int = 7, limit: int = 10) -> List[StrategyListing]:
        """Get trending strategies based on recent downloads and ratings"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Calculate trending score: recent downloads * rating
        cursor.execute(
            """
                       SELECT s.*,
                              COUNT(d.id)              as recent_downloads,
                              (COUNT(d.id) * s.rating) as trending_score
                       FROM marketplace_strategies s
                                LEFT JOIN strategy_downloads d
                                          ON s.id = d.strategy_id
                                              AND d.downloaded_at >= datetime('now', '-' || ? || ' days')
                       WHERE s.is_public = 1
                       GROUP BY s.id
                       ORDER BY trending_score DESC LIMIT ?
                       """,
            (days, limit),
        )

        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_listing(row) for row in rows]

    # ============================================================
    # DOWNLOADING AND CLONING
    # ============================================================

    def download_strategy(self, strategy_id: int, user_id: int) -> Optional[Dict]:
        """
        Download/clone a strategy

        Returns:
            Strategy configuration dict to be used with create_strategy()
        """
        strategy = self.get_strategy(strategy_id)
        if not strategy:
            return None

        # Record download
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
                       INSERT INTO strategy_downloads (strategy_id, user_id)
                       VALUES (?, ?)
                       """,
            (strategy_id, user_id),
        )

        # Increment download counter
        cursor.execute(
            """
                       UPDATE marketplace_strategies
                       SET downloads = downloads + 1
                       WHERE id = ?
                       """,
            (strategy_id,),
        )

        conn.commit()
        conn.close()

        # Return strategy configuration
        return {
            "strategy_type": strategy.strategy_type,
            "name": f"{strategy.name} (Clone)",
            "parameters": strategy.parameters,
            "description": strategy.description,
            "original_id": strategy_id,
            "original_creator": strategy.creator_name,
        }

    def get_user_downloads(self, user_id: int) -> List[StrategyListing]:
        """Get strategies downloaded by user"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
                       SELECT DISTINCT s.*
                       FROM marketplace_strategies s
                                JOIN strategy_downloads d ON s.id = d.strategy_id
                       WHERE d.user_id = ?
                       ORDER BY d.downloaded_at DESC
                       """,
            (user_id,),
        )

        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_listing(row) for row in rows]

    # ============================================================
    # RATINGS AND REVIEWS
    # ============================================================

    def add_review(self, review: StrategyReview) -> bool:
        """Add or update a strategy review"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Insert or replace review
        cursor.execute(
            """
            INSERT OR REPLACE INTO strategy_reviews (
                strategy_id, user_id, username, rating,
                review_text, performance_achieved
            ) VALUES (?, ?, ?, ?, ?, ?)
        """,
            (review.strategy_id, review.user_id, review.username, review.rating, review.review_text, json.dumps(review.performance_achieved)),
        )

        # Recalculate strategy rating
        cursor.execute(
            """
                       SELECT AVG(rating) as avg_rating, COUNT(*) as num_ratings
                       FROM strategy_reviews
                       WHERE strategy_id = ?
                       """,
            (review.strategy_id,),
        )

        result = cursor.fetchone()
        avg_rating, num_ratings = result

        cursor.execute(
            """
                       UPDATE marketplace_strategies
                       SET rating      = ?,
                           num_ratings = ?,
                           num_reviews = ?
                       WHERE id = ?
                       """,
            (avg_rating, num_ratings, num_ratings, review.strategy_id),
        )

        conn.commit()
        conn.close()

        return True

    def get_reviews(self, strategy_id: int, limit: int = 50) -> List[StrategyReview]:
        """Get reviews for a strategy"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
                       SELECT *
                       FROM strategy_reviews
                       WHERE strategy_id = ?
                       ORDER BY created_at DESC LIMIT ?
                       """,
            (strategy_id, limit),
        )

        rows = cursor.fetchall()
        conn.close()

        reviews = []
        for row in rows:
            review = StrategyReview(
                id=row["id"],
                strategy_id=row["strategy_id"],
                user_id=row["user_id"],
                username=row["username"],
                rating=row["rating"],
                review_text=row["review_text"],
                performance_achieved=json.loads(row["performance_achieved"]) if row["performance_achieved"] else {},
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            reviews.append(review)

        return reviews

    # ============================================================
    # FAVORITES
    # ============================================================

    def add_favorite(self, strategy_id: int, user_id: int) -> bool:
        """Add strategy to favorites"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                           INSERT INTO strategy_favorites (strategy_id, user_id)
                           VALUES (?, ?)
                           """,
                (strategy_id, user_id),
            )
            conn.commit()
            success = True
        except sqlite3.IntegrityError:
            success = False

        conn.close()
        return success

    def remove_favorite(self, strategy_id: int, user_id: int) -> bool:
        """Remove strategy from favorites"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
                       DELETE
                       FROM strategy_favorites
                       WHERE strategy_id = ?
                         AND user_id = ?
                       """,
            (strategy_id, user_id),
        )

        success = cursor.rowcount > 0
        conn.commit()
        conn.close()

        return success

    def get_favorites(self, user_id: int) -> List[StrategyListing]:
        """Get user's favorite strategies"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
                       SELECT s.*
                       FROM marketplace_strategies s
                                JOIN strategy_favorites f ON s.id = f.strategy_id
                       WHERE f.user_id = ?
                       ORDER BY f.created_at DESC
                       """,
            (user_id,),
        )

        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_listing(row) for row in rows]

    def is_favorited(self, strategy_id: int, user_id: int) -> bool:
        """Check if strategy is favorited by user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
                       SELECT COUNT(*)
                       FROM strategy_favorites
                       WHERE strategy_id = ?
                         AND user_id = ?
                       """,
            (strategy_id, user_id),
        )

        count = cursor.fetchone()[0]
        conn.close()

        return count > 0

    # ============================================================
    # STATISTICS AND ANALYTICS
    # ============================================================

    def get_marketplace_stats(self) -> Dict[str, Any]:
        """Get overall marketplace statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
                       SELECT COUNT(*)                   as total_strategies,
                              COUNT(DISTINCT creator_id) as total_creators,
                              SUM(downloads)             as total_downloads,
                              AVG(rating)                as avg_rating
                       FROM marketplace_strategies
                       WHERE is_public = 1
                       """)

        row = cursor.fetchone()

        stats = {"total_strategies": row[0], "total_creators": row[1], "total_downloads": row[2], "average_rating": round(row[3], 2) if row[3] else 0}

        # Category breakdown
        cursor.execute("""
                       SELECT category, COUNT(*) as count
                       FROM marketplace_strategies
                       WHERE is_public = 1
                       GROUP BY category
                       """)

        stats["by_category"] = {row[0]: row[1] for row in cursor.fetchall()}

        conn.close()
        return stats

    def get_creator_stats(self, user_id: int) -> Dict[str, Any]:
        """Get statistics for a creator"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
                       SELECT COUNT(*)         as total_published,
                              SUM(downloads)   as total_downloads,
                              AVG(rating)      as avg_rating,
                              SUM(num_ratings) as total_ratings
                       FROM marketplace_strategies
                       WHERE creator_id = ?
                       """,
            (user_id,),
        )

        row = cursor.fetchone()

        stats = {
            "strategies_published": row[0],
            "total_downloads": row[1] or 0,
            "average_rating": round(row[2], 2) if row[2] else 0,
            "total_ratings": row[3] or 0,
        }

        conn.close()
        return stats

    # ============================================================
    # HELPER METHODS
    # ============================================================

    def _row_to_listing(self, row: sqlite3.Row) -> StrategyListing:
        """Convert database row to StrategyListing"""
        return StrategyListing(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            creator_id=row["creator_id"],
            creator_name=row["creator_name"],
            strategy_type=row["strategy_type"],
            category=row["category"],
            complexity=row["complexity"],
            parameters=json.loads(row["parameters"]),
            performance_metrics=json.loads(row["performance_metrics"]) if row["performance_metrics"] else {},
            price=row["price"],
            is_public=bool(row["is_public"]),
            version=row["version"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            downloads=row["downloads"],
            rating=row["rating"],
            num_ratings=row["num_ratings"],
            num_reviews=row["num_reviews"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
