"""add marketplace access control columns and strategy_purchases table

Revision ID: f6g7h8i9j0k1
Revises: e5f6g7h8i9j0
Create Date: 2026-03-16
"""

import sqlalchemy as sa

from alembic import op

# revision identifiers
revision = "f6g7h8i9j0k1"
down_revision = "e5f6g7h8i9j0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add access-control columns to marketplace_strategies
    op.add_column(
        "marketplace_strategies",
        sa.Column("is_proprietary", sa.Boolean(), server_default=sa.text("false"), nullable=False),
    )
    op.add_column(
        "marketplace_strategies",
        sa.Column("status", sa.String(), server_default=sa.text("'approved'"), nullable=False),
    )
    op.add_column(
        "marketplace_strategies",
        sa.Column("rejection_reason", sa.Text(), nullable=True),
    )
    op.add_column(
        "marketplace_strategies",
        sa.Column("custom_strategy_id", sa.Integer(), sa.ForeignKey("custom_strategies.id"), nullable=True),
    )
    op.create_index("ix_marketplace_strategies_status", "marketplace_strategies", ["status"])

    # Create strategy_purchases table
    op.create_table(
        "strategy_purchases",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("strategy_id", sa.Integer(), sa.ForeignKey("marketplace_strategies.id"), nullable=False),
        sa.Column("stripe_checkout_session_id", sa.String(), nullable=True),
        sa.Column("amount_paid", sa.Float(), nullable=False),
        sa.Column("currency", sa.String(), server_default=sa.text("'usd'")),
        sa.Column("status", sa.String(), server_default=sa.text("'completed'")),
        sa.Column("purchased_at", sa.DateTime(), server_default=sa.func.now()),
        sa.UniqueConstraint("user_id", "strategy_id", name="_user_strategy_purchase_uc"),
    )


def downgrade() -> None:
    op.drop_table("strategy_purchases")
    op.drop_index("ix_marketplace_strategies_status", table_name="marketplace_strategies")
    op.drop_column("marketplace_strategies", "custom_strategy_id")
    op.drop_column("marketplace_strategies", "rejection_reason")
    op.drop_column("marketplace_strategies", "status")
    op.drop_column("marketplace_strategies", "is_proprietary")
