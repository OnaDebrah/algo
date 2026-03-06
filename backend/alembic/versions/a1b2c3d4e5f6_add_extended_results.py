"""add extended_results to backtest_runs

Revision ID: a1b2c3d4e5f6
Revises: 5ebb4254851c
Create Date: 2026-03-04 12:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "5ebb4254851c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("backtest_runs", sa.Column("extended_results", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("backtest_runs", "extended_results")
