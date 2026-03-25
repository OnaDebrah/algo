"""Add is_encrypted column to custom_strategies for code-at-rest encryption

Revision ID: u1v2w3x4y5z6
Revises: t0u1v2w3x4y5
Create Date: 2026-03-24
"""

import sqlalchemy as sa

from alembic import op

revision = "u1v2w3x4y5z6"
down_revision = "t0u1v2w3x4y5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "custom_strategies",
        sa.Column(
            "is_encrypted",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )


def downgrade() -> None:
    op.drop_column("custom_strategies", "is_encrypted")
