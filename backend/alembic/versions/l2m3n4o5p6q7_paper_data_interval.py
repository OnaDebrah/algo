"""Add data_interval to paper portfolios

Revision ID: l2m3n4o5p6q7
Revises: k1l2m3n4o5p6
Create Date: 2026-03-21
"""

import sqlalchemy as sa

from alembic import op

revision = "l2m3n4o5p6q7"
down_revision = "k1l2m3n4o5p6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "paper_portfolios",
        sa.Column("data_interval", sa.String(10), nullable=True, server_default="1d"),
    )


def downgrade() -> None:
    op.drop_column("paper_portfolios", "data_interval")
