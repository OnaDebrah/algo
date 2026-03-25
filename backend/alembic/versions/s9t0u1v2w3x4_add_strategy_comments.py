"""Add strategy_comments table for marketplace discussions

Revision ID: s9t0u1v2w3x4
Revises: r8s9t0u1v2w3
Create Date: 2026-03-24
"""

import sqlalchemy as sa

from alembic import op

revision = "s9t0u1v2w3x4"
down_revision = "r8s9t0u1v2w3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "strategy_comments",
        sa.Column("id", sa.Integer, primary_key=True, index=True),
        sa.Column("strategy_id", sa.Integer, sa.ForeignKey("marketplace_strategies.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("username", sa.String(100), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("parent_comment_id", sa.Integer, sa.ForeignKey("strategy_comments.id", ondelete="CASCADE"), nullable=True),
        sa.Column("is_edited", sa.Boolean, default=False),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("strategy_comments")
