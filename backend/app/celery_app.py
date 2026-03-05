"""
Celery application configuration

Start the worker with:
    celery -A app.celery_app worker --loglevel=info

Start the beat scheduler with:
    celery -A app.celery_app beat --loglevel=info
"""

import os

from celery import Celery

from .config import settings

REDIS_URL = os.getenv("REDIS_URL", settings.REDIS_URL)

celery_app = Celery(
    "oraculum",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["app.tasks.backtest_tasks"],
)

celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",

    # Result expiry (24 hours)
    result_expires=86400,

    # Timezone
    timezone="UTC",
    enable_utc=True,

    # Task execution
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # Fair scheduling for long tasks

    # Retry policy
    task_default_retry_delay=60,
    task_max_retries=3,

    # All tasks use the default queue (single worker setup)
    task_default_queue="default",
)
