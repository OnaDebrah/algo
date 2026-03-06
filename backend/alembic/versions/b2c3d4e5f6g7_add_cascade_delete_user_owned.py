"""add cascade delete to user-owned foreign keys

Revision ID: b2c3d4e5f6g7
Revises: a1b2c3d4e5f6
Create Date: 2026-03-05 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'b2c3d4e5f6g7'
down_revision: Union[str, None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # backtest_runs.user_id → CASCADE on delete
    op.drop_constraint('backtest_runs_user_id_fkey', 'backtest_runs', type_='foreignkey')
    op.create_foreign_key(
        'backtest_runs_user_id_fkey',
        'backtest_runs', 'users',
        ['user_id'], ['id'],
        ondelete='CASCADE',
    )

    # portfolios.user_id → CASCADE on delete
    op.drop_constraint('portfolios_user_id_fkey', 'portfolios', type_='foreignkey')
    op.create_foreign_key(
        'portfolios_user_id_fkey',
        'portfolios', 'users',
        ['user_id'], ['id'],
        ondelete='CASCADE',
    )


def downgrade() -> None:
    # Revert to FK without ondelete
    op.drop_constraint('portfolios_user_id_fkey', 'portfolios', type_='foreignkey')
    op.create_foreign_key(
        'portfolios_user_id_fkey',
        'portfolios', 'users',
        ['user_id'], ['id'],
    )

    op.drop_constraint('backtest_runs_user_id_fkey', 'backtest_runs', type_='foreignkey')
    op.create_foreign_key(
        'backtest_runs_user_id_fkey',
        'backtest_runs', 'users',
        ['user_id'], ['id'],
    )
