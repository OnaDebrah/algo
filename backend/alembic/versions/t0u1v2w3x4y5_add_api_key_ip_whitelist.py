"""Add allowed_ips column to api_keys table for IP whitelisting

Revision ID: t0u1v2w3x4y5
Revises: s9t0u1v2w3x4
Create Date: 2026-03-24
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision = "t0u1v2w3x4y5"
down_revision = "s9t0u1v2w3x4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("api_keys", sa.Column("allowed_ips", JSONB, nullable=True))


def downgrade() -> None:
    op.drop_column("api_keys", "allowed_ips")
