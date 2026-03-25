# Oraculum Trading Platform -- Self-Hosting Guide

Complete guide for deploying Oraculum on your own infrastructure using Docker Compose.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Quick Start](#2-quick-start)
3. [Environment Configuration](#3-environment-configuration)
4. [Architecture Overview](#4-architecture-overview)
5. [Production Hardening](#5-production-hardening)
6. [Scaling](#6-scaling)
7. [Monitoring](#7-monitoring)
8. [Updating](#8-updating)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Prerequisites

### System Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 4 GB | 8 GB+ |
| CPU | 2 cores | 4 cores+ |
| Disk | 20 GB | 50 GB+ (SSD) |
| Docker | 24.0+ | Latest stable |
| Docker Compose | v2.20+ | Latest stable |

### Required Ports

Ensure the following ports are available on your host:

| Port | Service | Purpose |
|------|---------|---------|
| 3000 | Next.js | Frontend UI |
| 8000 | FastAPI | Backend API |
| 5432 | PostgreSQL | Database |
| 6379 | Redis | Cache and message broker |
| 5555 | Flower | Celery task monitoring |
| 9090 | Prometheus | Metrics collection |
| 3001 | Grafana | Dashboards and alerting |

In production, you will typically only expose ports 80/443 through a reverse proxy and keep all other ports internal.

### Software

- **Docker Engine 24+**: [Install Docker](https://docs.docker.com/engine/install/)
- **Docker Compose v2**: Included with Docker Desktop; for Linux see [Install Compose Plugin](https://docs.docker.com/compose/install/linux/)
- **Git**: To clone the repository

Verify your installation:

```bash
docker --version        # Docker version 24.x or higher
docker compose version  # Docker Compose version v2.x
```

---

## 2. Quick Start

Get Oraculum running in five steps.

### Step 1: Clone the Repository

```bash
git clone https://github.com/OnaDebrah/algo.git
cd algo
```

### Step 2: Configure Environment Variables

```bash
cp backend/.env.example backend/.env
```

Open `backend/.env` and set **at minimum** the following values:

```bash
POSTGRES_PASSWORD=<strong-random-password>
SECRET_KEY=<generate-with-python-see-below>
JWT_SECRET_KEY=<generate-with-python-see-below>
```

Generate secure keys:

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

### Step 3: Build and Start All Services

```bash
docker compose up -d --build
```

This builds the backend and frontend images, pulls PostgreSQL/Redis/Prometheus/Grafana, runs database migrations automatically, and starts all services.

### Step 4: Verify Services Are Running

```bash
docker compose ps
```

All containers should show a `healthy` or `running` status. Then check:

```bash
# API health
curl http://localhost:8000/docs

# Frontend
curl -s -o /dev/null -w "%{http_code}" http://localhost:3000
```

### Step 5: Create an Admin Account

If `ADMIN_EMAIL`, `ADMIN_USERNAME`, and `ADMIN_PASSWORD` are set in your `.env` file, a superuser is created automatically on first startup. Otherwise, register through the frontend at `http://localhost:3000`.

---

## 3. Environment Configuration

All configuration is managed through `backend/.env`. A full reference is available in `backend/.env.example`.

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `POSTGRES_USER` | Database username | `trading_user` |
| `POSTGRES_PASSWORD` | Database password | *(strong random value)* |
| `POSTGRES_DB` | Database name | `trading_platform` |
| `POSTGRES_HOST` | Database host | `postgres` (Docker) |
| `POSTGRES_PORT` | Database port | `5432` |
| `SECRET_KEY` | Application secret for encryption | *(hex token, 32+ bytes)* |
| `JWT_SECRET_KEY` | JWT signing key | *(hex token, 32+ bytes)* |
| `REDIS_URL` | Redis connection string | `redis://redis:6379/0` |

### Market Data Providers

At least one data provider is needed for market data. Yahoo Finance (`yahoo`) works out of the box with no API key.

| Variable | Description |
|----------|-------------|
| `DATA_PROVIDER` | `yahoo`, `polygon`, `alpaca`, or `iex` |
| `ALPACA_API_KEY` | Alpaca API key (required for live trading) |
| `ALPACA_SECRET` | Alpaca API secret |
| `POLYGON_API_KEY` | Polygon.io API key |
| `IEX_API_KEY` | IEX Cloud API key |
| `ALPHA_VANTAGE_API_KEY` | Alpha Vantage API key |

### AI/LLM Integration

The AI analyst feature requires at least one LLM API key:

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Claude API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `GEMINI_API_KEY` | Google Gemini API key |
| `DEEPSEEK_API_KEY` | DeepSeek API key |

### Frontend

| Variable | Description | Default |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_URL` | Backend URL the frontend calls | `http://localhost:8000` |
| `NEXT_PUBLIC_WS_URL` | WebSocket URL | `ws://localhost:8000` |

### Monitoring

| Variable | Description | Default |
|----------|-------------|---------|
| `GRAFANA_ADMIN_USER` | Grafana admin username | `admin` |
| `GRAFANA_ADMIN_PASSWORD` | Grafana admin password | `oraculum` |
| `SENTRY_DSN` | Sentry error tracking DSN | *(empty, disabled)* |
| `LOG_LEVEL` | Logging verbosity | `INFO` |

### Notifications (Optional)

| Variable | Description |
|----------|-------------|
| `EMAIL_ENABLED` | Enable email alerts (`true`/`false`) |
| `SMTP_SERVER` | SMTP host |
| `SMTP_PORT` | SMTP port |
| `SMTP_USERNAME` | SMTP username |
| `SMTP_PASSWORD` | SMTP password |
| `SMS_ENABLED` | Enable SMS alerts via Twilio (`true`/`false`) |
| `TWILIO_ACCOUNT_SID` | Twilio account SID |
| `TWILIO_AUTH_TOKEN` | Twilio auth token |

### Payments (Optional)

| Variable | Description |
|----------|-------------|
| `STRIPE_SECRET_KEY` | Stripe secret key for marketplace |
| `STRIPE_PUBLISHABLE_KEY` | Stripe publishable key |
| `STRIPE_WEBHOOK_SECRET` | Stripe webhook signing secret |

---

## 4. Architecture Overview

```
                    Internet
                       |
               [Reverse Proxy]      (nginx/Caddy, optional)
                  /         \
            :3000            :8000
          +-------+        +--------+       +----------+
          |Next.js|------->|FastAPI  |------>|PostgreSQL|
          +-------+  API   +--------+       +----------+
                            |    |
                      +-----+    +-----+
                      |                |
                 +--------+      +---------+
                 | Redis  |      | Celery  |
                 +--------+      | Workers |
                      |          +---------+
                 +--------+          |
                 | Celery |     +---------+
                 | Beat   |     | Flower  |
                 +--------+     +---------+

          +------------+     +---------+
          | Prometheus |---->| Grafana |
          +------------+     +---------+
```

**Service roles:**

- **PostgreSQL 16**: Primary datastore for users, strategies, backtests, and marketplace data.
- **Redis 7**: Celery broker, caching, rate limiting, and WebSocket pub/sub.
- **FastAPI**: Async REST API and WebSocket server.
- **Celery Worker**: Executes backtests, price alerts, and other background jobs.
- **Celery Beat**: Schedules periodic tasks (market scans, alert checks).
- **Next.js**: Server-rendered React frontend.
- **Flower**: Web UI for monitoring Celery task queues.
- **Prometheus**: Scrapes `/metrics` from the API every 15 seconds.
- **Grafana**: Visualizes metrics with pre-configured dashboards.

---

## 5. Production Hardening

### 5.1 Reverse Proxy with SSL

Never expose the application services directly in production. Use a reverse proxy with TLS termination.

**Caddy** (automatic HTTPS with Let's Encrypt):

```Caddyfile
oraculum.example.com {
    handle /api/* {
        reverse_proxy fastapi:8000
    }
    handle /ws/* {
        reverse_proxy fastapi:8000
    }
    handle {
        reverse_proxy nextjs:3000
    }
}
```

**nginx** example:

```nginx
server {
    listen 443 ssl http2;
    server_name oraculum.example.com;

    ssl_certificate     /etc/letsencrypt/live/oraculum.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/oraculum.example.com/privkey.pem;

    location /api/ {
        proxy_pass http://fastapi:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /ws/ {
        proxy_pass http://fastapi:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    location / {
        proxy_pass http://nextjs:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Update your `.env` accordingly:

```bash
NEXT_PUBLIC_API_URL=https://oraculum.example.com
NEXT_PUBLIC_WS_URL=wss://oraculum.example.com
ENVIRONMENT=production
```

### 5.2 Secure Credentials

Before going to production:

1. **Change all default passwords** -- `POSTGRES_PASSWORD`, `GRAFANA_ADMIN_PASSWORD`, `ADMIN_PASSWORD`.
2. **Generate strong secrets** for `SECRET_KEY` and `JWT_SECRET_KEY` (minimum 32 bytes of entropy).
3. **Restrict port exposure** -- remove host port mappings from `docker-compose.yml` for all services except the reverse proxy. Internal services communicate over the Docker network.

```yaml
# Example: remove host port binding for postgres
postgres:
  image: postgres:16-alpine
  # ports:          # <-- remove or comment out
  #   - "5432:5432"
```

### 5.3 Environment Hardening

```bash
ENVIRONMENT=production
LOG_LEVEL=WARNING
```

### 5.4 Container Resource Limits

Add resource constraints to prevent any single service from consuming all host resources:

```yaml
services:
  fastapi:
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 2G
        reservations:
          cpus: "0.5"
          memory: 512M

  celery_worker:
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 2G

  postgres:
    deploy:
      resources:
        limits:
          cpus: "1.0"
          memory: 1G

  redis:
    deploy:
      resources:
        limits:
          cpus: "0.5"
          memory: 512M
```

### 5.5 PostgreSQL Backups

**Automated daily backups** using `pg_dump`:

```bash
#!/bin/bash
# backup-postgres.sh -- run via cron
BACKUP_DIR=/backups/postgres
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FILENAME="oraculum_${TIMESTAMP}.sql.gz"

docker compose exec -T postgres pg_dump \
  -U trading_user \
  -d trading_platform \
  --format=custom \
  | gzip > "${BACKUP_DIR}/${FILENAME}"

# Retain last 30 days
find "${BACKUP_DIR}" -name "oraculum_*.sql.gz" -mtime +30 -delete

echo "Backup completed: ${FILENAME}"
```

Add to crontab:

```bash
# Daily at 2 AM
0 2 * * * /opt/oraculum/backup-postgres.sh >> /var/log/oraculum-backup.log 2>&1
```

**Restore from backup:**

```bash
gunzip -c oraculum_20260324_020000.sql.gz | \
  docker compose exec -T postgres pg_restore \
    -U trading_user \
    -d trading_platform \
    --clean --if-exists
```

### 5.6 Log Aggregation

For centralized logging, configure Docker's logging driver:

```yaml
# In docker-compose.yml or docker-compose.override.yml
services:
  fastapi:
    logging:
      driver: "json-file"
      options:
        max-size: "50m"
        max-file: "5"
```

For production environments, consider forwarding logs to an external system (Loki, ELK, Datadog) by using the appropriate Docker logging driver or a sidecar container.

---

## 6. Scaling

### 6.1 Multiple Celery Workers

Scale the Celery worker horizontally to process more backtests concurrently:

```bash
docker compose up -d --scale celery_worker=4
```

Or define multiple workers with different queues in a `docker-compose.override.yml`:

```yaml
services:
  celery_worker_backtest:
    build: ./backend
    command: celery -A app.celery_app worker --loglevel=info -Q backtest -c 4
    env_file:
      - ./backend/.env
    environment:
      POSTGRES_HOST: postgres
    depends_on:
      redis:
        condition: service_healthy

  celery_worker_default:
    build: ./backend
    command: celery -A app.celery_app worker --loglevel=info -Q default -c 2
    env_file:
      - ./backend/.env
    environment:
      POSTGRES_HOST: postgres
    depends_on:
      redis:
        condition: service_healthy
```

### 6.2 PostgreSQL Read Replicas

For read-heavy workloads, add a streaming replica:

1. Configure the primary with `wal_level = replica` and `max_wal_senders = 3`.
2. Set up a standby with `primary_conninfo` pointing to the primary.
3. Route read queries (market data lookups, leaderboards) to the replica through application-level read/write splitting.

### 6.3 Redis Sentinel / Cluster

For Redis high availability:

- **Redis Sentinel**: Provides automatic failover with a master-replica setup. Update `REDIS_URL` to use the Sentinel connection string.
- **Redis Cluster**: For sharding across multiple nodes when cache size exceeds single-node memory.

### 6.4 Container Orchestration

For larger deployments, migrate from Docker Compose to **Kubernetes**:

- Convert services to Kubernetes Deployments and StatefulSets (use StatefulSets for PostgreSQL and Redis).
- Use Horizontal Pod Autoscaler (HPA) for FastAPI and Celery workers.
- Manage secrets with Kubernetes Secrets or an external vault (HashiCorp Vault, AWS Secrets Manager).
- Use persistent volume claims (PVCs) for database and Redis storage.

Tools like [Kompose](https://kompose.io/) can generate an initial set of Kubernetes manifests from the existing `docker-compose.yml`.

---

## 7. Monitoring

### 7.1 Prometheus

Prometheus scrapes the FastAPI `/metrics` endpoint every 15 seconds. It is pre-configured in `monitoring/prometheus.yml` with a 30-day retention period.

Access the Prometheus UI at `http://localhost:9090`.

Useful PromQL queries:

```promql
# Request rate (last 5 min)
sum(rate(http_requests_total[5m]))

# P95 latency
histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))

# Error rate percentage
sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m])) * 100

# Backtest queue size
oraculum_backtest_queue_size
```

### 7.2 Grafana

Grafana is available at `http://localhost:3001` with pre-provisioned dashboards and Prometheus as the default data source.

Default credentials:

- Username: `admin` (or `GRAFANA_ADMIN_USER`)
- Password: `oraculum` (or `GRAFANA_ADMIN_PASSWORD`)

Change these immediately in production via `GRAFANA_ADMIN_USER` and `GRAFANA_ADMIN_PASSWORD` in your `.env`.

### 7.3 Flower (Celery Monitoring)

Flower provides real-time monitoring of Celery workers and tasks at `http://localhost:5555`.

Use it to:

- View active, completed, and failed tasks.
- Monitor worker status and resource usage.
- Inspect task arguments and results.
- Identify slow or stuck tasks.

### 7.4 Pre-configured Alert Rules

The platform ships with alert rules in `monitoring/alert_rules.yml`:

| Alert | Condition | Severity |
|-------|-----------|----------|
| HighErrorRate | >5% of requests return 5xx for 5 min | Critical |
| HighLatency | P95 latency >10s for 5 min | Warning |
| APIDown | API unreachable for 1 min | Critical |
| BacktestQueueBacklog | >20 queued backtests for 10 min | Warning |
| BacktestFailureRate | >30% of backtests failing for 10 min | Warning |
| YFinanceHighErrorRate | >20% data provider errors for 10 min | Warning |
| HighAuthFailureRate | >10 failed auth/sec for 5 min | Warning |
| RateLimitSpike | >50 rate limit hits/sec for 5 min | Info |

To receive alert notifications, configure an Alertmanager instance or use Grafana's built-in alerting with notification channels (email, Slack, PagerDuty).

---

## 8. Updating

### 8.1 Standard Update Procedure

```bash
# 1. Pull latest code
git pull origin main

# 2. Rebuild images
docker compose build

# 3. Run database migrations
docker compose run --rm migrate

# 4. Restart services with zero downtime (rolling)
docker compose up -d

# 5. Verify
docker compose ps
curl http://localhost:8000/docs
```

### 8.2 Database Migrations

Migrations run automatically via the `migrate` service on `docker compose up`. To run them manually:

```bash
# Apply all pending migrations
docker compose run --rm migrate alembic upgrade head

# Check current migration version
docker compose run --rm migrate alembic current

# View migration history
docker compose run --rm migrate alembic history
```

### 8.3 Rollback

If an update causes issues:

```bash
# Roll back to a specific migration
docker compose run --rm migrate alembic downgrade -1

# Roll back to a specific git commit and rebuild
git checkout <commit-hash>
docker compose build
docker compose up -d
```

---

## 9. Troubleshooting

### Container fails to start

```bash
# Check logs for the failing service
docker compose logs <service-name>

# Example: check FastAPI logs
docker compose logs fastapi

# Follow logs in real time
docker compose logs -f fastapi
```

### Database connection refused

The `migrate` and `fastapi` services wait for PostgreSQL to be healthy before starting. If connections still fail:

```bash
# Verify PostgreSQL is running and healthy
docker compose ps postgres
docker compose exec postgres pg_isready -U trading_user

# Check if the database exists
docker compose exec postgres psql -U trading_user -l
```

Common causes:
- `POSTGRES_PASSWORD` is empty in `.env` -- set a password.
- `POSTGRES_HOST` is not set to `postgres` when running in Docker.

### Redis connection refused

```bash
docker compose ps redis
docker compose exec redis redis-cli ping
# Expected: PONG
```

### Migrations fail

```bash
# View detailed migration error
docker compose logs migrate

# Check current state
docker compose run --rm migrate alembic current

# If the database is in a bad state, stamp it and retry
docker compose run --rm migrate alembic stamp head
```

### Frontend cannot reach the API

- Verify `NEXT_PUBLIC_API_URL` matches the actual API address.
- If using a reverse proxy, confirm it forwards `/api/` requests to `fastapi:8000`.
- Check browser console for CORS errors -- the API must allow the frontend's origin.

### Celery tasks not executing

```bash
# Check worker status
docker compose logs celery_worker

# Verify Redis connectivity
docker compose exec celery_worker celery -A app.celery_app inspect ping

# Check active tasks
docker compose exec celery_worker celery -A app.celery_app inspect active
```

### High memory usage

- Add resource limits (see [Section 5.4](#54-container-resource-limits)).
- Reduce Celery worker concurrency: change the worker command to include `--concurrency=2`.
- Monitor with `docker stats`.

### Port conflicts

If a port is already in use on the host:

```bash
# Find what is using the port
lsof -i :8000

# Change the host port in docker-compose.override.yml
# Example: map FastAPI to host port 8080 instead
services:
  fastapi:
    ports:
      - "8080:8000"
```

### Reset everything (development only)

```bash
# Stop all containers, remove volumes, and rebuild
docker compose down -v
docker compose up -d --build
```

**Warning:** `docker compose down -v` permanently deletes all data in PostgreSQL, Redis, Prometheus, and Grafana.

---

## Additional Resources

- **API Documentation**: `http://localhost:8000/docs` (Swagger UI)
- **Flower Dashboard**: `http://localhost:5555`
- **Grafana Dashboards**: `http://localhost:3001`
- **Prometheus Targets**: `http://localhost:9090/targets`
