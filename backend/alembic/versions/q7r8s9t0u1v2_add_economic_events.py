"""Add economic events table

Revision ID: q7r8s9t0u1v2
Revises: p6q7r8s9t0u1
Create Date: 2026-03-21
"""

from alembic import op
import sqlalchemy as sa

revision = "q7r8s9t0u1v2"
down_revision = "p6q7r8s9t0u1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "economic_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("event_name", sa.String(200), nullable=False),
        sa.Column("country", sa.String(5), server_default="US"),
        sa.Column("event_date", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("impact", sa.String(10), nullable=False, index=True),
        sa.Column("previous_value", sa.String(50), nullable=True),
        sa.Column("forecast_value", sa.String(50), nullable=True),
        sa.Column("actual_value", sa.String(50), nullable=True),
        sa.Column("category", sa.String(50), nullable=True),
        sa.Column("source", sa.String(50), server_default="manual"),
        sa.Column("fetched_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("economic_events")
