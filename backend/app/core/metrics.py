"""Custom Prometheus metrics for Oraculum platform monitoring.

Exposes application-level metrics beyond the auto-instrumented HTTP metrics
from prometheus_fastapi_instrumentator.
"""

from prometheus_client import Counter, Gauge, Histogram, Info

# ── Platform Info ───────────────────────────────────────────────────────────

platform_info = Info("oraculum_platform", "Platform metadata")
platform_info.info({"version": "1.0.0", "name": "Oraculum"})

# ── Backtest Metrics ────────────────────────────────────────────────────────

backtest_runs_total = Counter(
    "oraculum_backtest_runs_total",
    "Total number of backtest runs",
    ["strategy_key", "status"],  # status: completed, failed, timeout
)

backtest_duration_seconds = Histogram(
    "oraculum_backtest_duration_seconds",
    "Backtest execution duration in seconds",
    ["strategy_key"],
    buckets=[1, 5, 10, 30, 60, 120, 300, 600],
)

backtest_queue_size = Gauge(
    "oraculum_backtest_queue_size",
    "Number of backtests currently queued/running",
)

# ── AI Analyst Metrics ──────────────────────────────────────────────────────

analyst_reports_total = Counter(
    "oraculum_analyst_reports_total",
    "Total AI analyst reports generated",
    ["ticker", "status"],
)

analyst_duration_seconds = Histogram(
    "oraculum_analyst_duration_seconds",
    "AI analyst report generation time",
    buckets=[5, 10, 30, 60, 120, 300],
)

# ── Data Provider Metrics ──────────────────────────────────────────────────

yfinance_requests_total = Counter(
    "oraculum_yfinance_requests_total",
    "Total yfinance API requests",
    ["endpoint", "status"],  # endpoint: quote, historical, options; status: success, error, timeout
)

yfinance_request_duration_seconds = Histogram(
    "oraculum_yfinance_request_duration_seconds",
    "yfinance request latency",
    ["endpoint"],
    buckets=[0.5, 1, 2, 5, 10, 30],
)

yfinance_data_quality_errors = Counter(
    "oraculum_yfinance_data_quality_errors_total",
    "NaN/None/Inf values received from yfinance",
    ["field"],
)

# ── Authentication Metrics ──────────────────────────────────────────────────

auth_attempts_total = Counter(
    "oraculum_auth_attempts_total",
    "Authentication attempts",
    ["method", "status"],  # method: login, signup, refresh; status: success, failed
)

active_users = Gauge(
    "oraculum_active_users",
    "Number of currently active users (with valid sessions)",
)

# ── Live Trading Metrics ────────────────────────────────────────────────────

live_strategies_active = Gauge(
    "oraculum_live_strategies_active",
    "Number of live/paper strategies currently running",
    ["mode"],  # mode: live, paper
)

trade_executions_total = Counter(
    "oraculum_trade_executions_total",
    "Total trade executions",
    ["mode", "status"],  # mode: live, paper; status: filled, rejected, error
)

# ── Marketplace Metrics ─────────────────────────────────────────────────────

marketplace_strategies_total = Gauge(
    "oraculum_marketplace_strategies_total",
    "Total strategies in the marketplace",
)

marketplace_downloads_total = Counter(
    "oraculum_marketplace_downloads_total",
    "Strategy downloads",
)

# ── Celery / Task Queue Metrics ─────────────────────────────────────────────

celery_tasks_total = Counter(
    "oraculum_celery_tasks_total",
    "Total Celery tasks dispatched",
    ["task_name", "status"],
)

celery_task_duration_seconds = Histogram(
    "oraculum_celery_task_duration_seconds",
    "Celery task execution duration",
    ["task_name"],
    buckets=[1, 5, 10, 30, 60, 120, 300, 600],
)

# ── Rate Limiting Metrics ──────────────────────────────────────────────────

rate_limit_hits_total = Counter(
    "oraculum_rate_limit_hits_total",
    "Number of requests that hit rate limits",
    ["path"],
)

# ── WebSocket Metrics ──────────────────────────────────────────────────────

websocket_connections_active = Gauge(
    "oraculum_websocket_connections_active",
    "Active WebSocket connections",
)

# ── Database Metrics ────────────────────────────────────────────────────────

db_query_duration_seconds = Histogram(
    "oraculum_db_query_duration_seconds",
    "Database query execution time",
    ["operation"],  # operation: select, insert, update, delete
    buckets=[0.01, 0.05, 0.1, 0.5, 1, 5],
)
