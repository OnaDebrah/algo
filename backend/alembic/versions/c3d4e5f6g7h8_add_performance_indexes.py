"""add performance indexes on FK, status, and created_at columns

Revision ID: c3d4e5f6g7h8
Revises: b2c3d4e5f6g7
Create Date: 2026-03-05 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'c3d4e5f6g7h8'
down_revision: Union[str, None] = 'b2c3d4e5f6g7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # backtest_runs: user_id, status, created_at
    op.create_index('ix_backtest_runs_user_id', 'backtest_runs', ['user_id'])
    op.create_index('ix_backtest_runs_status', 'backtest_runs', ['status'])
    op.create_index('ix_backtest_runs_created_at', 'backtest_runs', ['created_at'])

    # portfolios: user_id
    op.create_index('ix_portfolios_user_id', 'portfolios', ['user_id'])

    # positions: portfolio_id
    op.create_index('ix_positions_portfolio_id', 'positions', ['portfolio_id'])

    # trades: portfolio_id
    op.create_index('ix_trades_portfolio_id', 'trades', ['portfolio_id'])

    # live_strategies: user_id, status
    op.create_index('ix_live_strategies_user_id', 'live_strategies', ['user_id'])
    op.create_index('ix_live_strategies_status', 'live_strategies', ['status'])

    # live_trades: strategy_id
    op.create_index('ix_live_trades_strategy_id', 'live_trades', ['strategy_id'])

    # usage_tracking: user_id
    op.create_index('ix_usage_tracking_user_id', 'usage_tracking', ['user_id'])


def downgrade() -> None:
    op.drop_index('ix_usage_tracking_user_id', 'usage_tracking')
    op.drop_index('ix_live_trades_strategy_id', 'live_trades')
    op.drop_index('ix_live_strategies_status', 'live_strategies')
    op.drop_index('ix_live_strategies_user_id', 'live_strategies')
    op.drop_index('ix_trades_portfolio_id', 'trades')
    op.drop_index('ix_positions_portfolio_id', 'positions')
    op.drop_index('ix_portfolios_user_id', 'portfolios')
    op.drop_index('ix_backtest_runs_created_at', 'backtest_runs')
    op.drop_index('ix_backtest_runs_status', 'backtest_runs')
    op.drop_index('ix_backtest_runs_user_id', 'backtest_runs')
