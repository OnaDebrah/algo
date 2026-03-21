"""Add strategy cloning/forking fields

Revision ID: m3n4o5p6q7r8
Revises: l2m3n4o5p6q7
Create Date: 2026-03-21
"""

from alembic import op
import sqlalchemy as sa

revision = "m3n4o5p6q7r8"
down_revision = "l2m3n4o5p6q7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "marketplace_strategies",
        sa.Column("parent_strategy_id", sa.Integer(), sa.ForeignKey("marketplace_strategies.id"), nullable=True),
    )
    op.add_column(
        "marketplace_strategies",
        sa.Column("fork_count", sa.Integer(), server_default="0", nullable=True),
    )


def downgrade() -> None:
    op.drop_column("marketplace_strategies", "fork_count")
    op.drop_column("marketplace_strategies", "parent_strategy_id")
