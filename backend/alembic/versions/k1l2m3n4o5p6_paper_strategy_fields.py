"""Add strategy fields to paper trading

Revision ID: k1l2m3n4o5p6
Revises: j0k1l2m3n4o5
Create Date: 2026-03-21
"""

from alembic import op
import sqlalchemy as sa

revision = "k1l2m3n4o5p6"
down_revision = "j0k1l2m3n4o5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Strategy fields on portfolio
    op.add_column("paper_portfolios", sa.Column("strategy_key", sa.String(100), nullable=True))
    op.add_column("paper_portfolios", sa.Column("strategy_params", sa.Text(), nullable=True))
    op.add_column("paper_portfolios", sa.Column("strategy_symbol", sa.String(20), nullable=True))
    op.add_column("paper_portfolios", sa.Column("trade_quantity", sa.Float(), nullable=True, server_default="100"))

    # Trade source field
    op.add_column("paper_trades", sa.Column("source", sa.String(20), nullable=False, server_default="manual"))


def downgrade() -> None:
    op.drop_column("paper_trades", "source")
    op.drop_column("paper_portfolios", "trade_quantity")
    op.drop_column("paper_portfolios", "strategy_symbol")
    op.drop_column("paper_portfolios", "strategy_params")
    op.drop_column("paper_portfolios", "strategy_key")
