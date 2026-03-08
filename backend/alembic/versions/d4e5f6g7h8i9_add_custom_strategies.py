"""add custom_strategies table

Revision ID: d4e5f6g7h8i9
Revises: c3d4e5f6g7h8
Create Date: 2026-03-05
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d4e5f6g7h8i9"
down_revision: Union[str, None] = "c3d4e5f6g7h8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "custom_strategies",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("code", sa.Text(), nullable=False),
        sa.Column("strategy_type", sa.String(50), nullable=False, server_default="custom"),
        sa.Column("parameters", postgresql.JSONB(), nullable=True),
        sa.Column("is_validated", sa.Boolean(), server_default=sa.text("false")),
        sa.Column("ai_generated", sa.Boolean(), server_default=sa.text("false")),
        sa.Column("ai_explanation", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_custom_strategies_id", "custom_strategies", ["id"])
    op.create_index("ix_custom_strategies_user_id", "custom_strategies", ["user_id"])
    op.create_index("ix_custom_strategies_created_at", "custom_strategies", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_custom_strategies_created_at", table_name="custom_strategies")
    op.drop_index("ix_custom_strategies_user_id", table_name="custom_strategies")
    op.drop_index("ix_custom_strategies_id", table_name="custom_strategies")
    op.drop_table("custom_strategies")
