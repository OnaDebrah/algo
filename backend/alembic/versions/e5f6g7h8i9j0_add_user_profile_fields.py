"""add user profile fields (country, investor_type, risk_profile)

Revision ID: e5f6g7h8i9j0
Revises: d4e5f6g7h8i9
Create Date: 2026-03-06
"""

import sqlalchemy as sa

from alembic import op

# revision identifiers
revision = "e5f6g7h8i9j0"
down_revision = "d4e5f6g7h8i9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("country", sa.String(100), nullable=True))
    op.add_column("users", sa.Column("investor_type", sa.String(50), nullable=True))
    op.add_column("users", sa.Column("risk_profile", sa.String(50), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "risk_profile")
    op.drop_column("users", "investor_type")
    op.drop_column("users", "country")
