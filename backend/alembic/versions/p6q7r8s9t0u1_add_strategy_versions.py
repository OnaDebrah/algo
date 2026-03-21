"""Add strategy versions and scheduled backtests tables

Revision ID: p6q7r8s9t0u1
Revises: o5p6q7r8s9t0
Create Date: 2026-03-21
"""

from alembic import op
import sqlalchemy as sa

revision = "p6q7r8s9t0u1"
down_revision = "o5p6q7r8s9t0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Strategy Versions
    op.create_table(
        "strategy_versions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("strategy_id", sa.Integer(), nullable=False, index=True),
        sa.Column("strategy_type", sa.String(20), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("version_label", sa.String(20), nullable=False),
        sa.Column("parameters_snapshot", sa.JSON(), nullable=False),
        sa.Column("performance_snapshot", sa.JSON(), nullable=True),
        sa.Column("change_description", sa.Text(), nullable=True),
        sa.Column("created_by", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # Scheduled Backtests
    op.create_table(
        "scheduled_backtests",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("strategy_key", sa.String(100), nullable=False),
        sa.Column("strategy_params", sa.JSON(), server_default="{}"),
        sa.Column("symbols", sa.JSON(), nullable=False),
        sa.Column("interval", sa.String(10), server_default="1d"),
        sa.Column("period", sa.String(10), server_default="1y"),
        sa.Column("initial_capital", sa.Float(), server_default="100000"),
        sa.Column("schedule_cron", sa.String(50), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default="true", index=True),
        sa.Column("last_run_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("next_run_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # Scheduled Backtest Runs
    op.create_table(
        "scheduled_backtest_runs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("scheduled_backtest_id", sa.Integer(), sa.ForeignKey("scheduled_backtests.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("status", sa.String(20), server_default="pending"),
        sa.Column("result_summary", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("scheduled_backtest_runs")
    op.drop_table("scheduled_backtests")
    op.drop_table("strategy_versions")
