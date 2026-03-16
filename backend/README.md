"""
# Backtesting Platform Backend

FastAPI-based backend for the advanced trading platform.

## Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create .env file:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Run the application:
```bash
python run.py
```

Or using uvicorn directly:
```bash
uvicorn app.main:app --reload
```

## API Documentation

Once running, visit:
- Interactive docs: http://localhost:8000/api/docs
- ReDoc: http://localhost:8000/api/redoc

## Project Structure

```
backend/
├── app/
│   ├── api/
│   │   ├── deps.py           # Dependencies
│   │   └── routes/           # API routes
│   │       ├── auth.py
│   │       ├── backtest.py
│   │       ├── portfolio.py
│   │       ├── market.py
│   │       ├── strategy.py
│   │       ├── analytics.py
│   │       ├── regime.py
│   │       └── websocket.py
│   ├── core/
│   │   ├── config.py         # Configuration
│   │   └── security.py       # Authentication
│   ├── db/
│   │   ├── base.py           # Database setup
│   │   └── init_db.py        # Database initialization
│   ├── models/               # SQLAlchemy models
│   │   ├── user.py
│   │   ├── backtest.py
│   │   └── portfolio.py
│   ├── schemas/              # Pydantic schemas
│   │   ├── user.py
│   │   ├── backtest.py
│   │   ├── portfolio.py
│   │   ├── market.py
│   │   ├── strategy.py
│   │   └── regime.py
│   ├── services/             # Business logic
│   │   ├── backtest_service.py
│   │   └── market_service.py
│   └── main.py               # FastAPI app
├── core/                     # Your existing trading engine
├── strategies/               # Your existing strategies
├── analytics/                # Your existing analytics
├── .env.example
├── requirements.txt
├── run.py
└── README.md
```

## API Endpoints

### Authentication
- POST `/api/auth/register` - Register new user
- POST `/api/auth/login` - Page
- POST `/api/auth/refresh` - Refresh token
- GET `/api/auth/me` - Get current user

### Backtesting
- POST `/api/backtest/single` - Run single asset backtest
- POST `/api/backtest/multi` - Run multi-asset backtest
- POST `/api/backtest/options` - Run options backtest
- GET `/api/backtest/history` - Get backtest history

### Portfolio
- GET `/api/portfolio/` - List portfolios
- POST `/api/portfolio/` - Create portfolio
- GET `/api/portfolio/{id}` - Get portfolio
- GET `/api/portfolio/{id}/metrics` - Get portfolio metrics
- GET `/api/portfolio/{id}/positions` - Get positions
- GET `/api/portfolio/{id}/trades` - Get trades

### Market Data
- GET `/api/market/quote/{symbol}` - Get real-time quote
- POST `/api/market/quotes` - Get multiple quotes
- GET `/api/market/historical/{symbol}` - Get historical data
- GET `/api/market/search` - Search symbols

### Strategies
- GET `/api/strategy/list` - List all strategies
- GET `/api/strategy/{key}` - Get strategy details

### Analytics
- GET `/api/analytics/performance/{portfolio_id}` - Performance analytics
- GET `/api/analytics/returns/{portfolio_id}` - Returns analysis
- GET `/api/analytics/risk/{portfolio_id}` - Risk metrics
- GET `/api/analytics/drawdown/{portfolio_id}` - Drawdown analysis

### Market Regime
- GET `/api/regime/detect/{symbol}` - Detect current regime
- GET `/api/regime/history/{symbol}` - Get regime history
- POST `/api/regime/batch` - Batch regime detection

### WebSocket
- WS `/api/ws/market/{symbol}` - Real-time market data
- WS `/api/ws/portfolio/{portfolio_id}` - Portfolio updates
- WS `/api/ws/backtest/{run_id}` - Backtest progress

## Testing

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest
```

## Production Deployment

1. Set environment variables in production
2. Use PostgreSQL instead of SQLite
3. Set strong SECRET_KEY
4. Use proper CORS origins
5. Enable HTTPS
6. Use production ASGI server (gunicorn + uvicorn)

```bash
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```
