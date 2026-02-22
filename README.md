# Algorithmic Backtesting Platform

Algorithmic backtesting system with machine learning strategies, multi-symbol support, and comprehensive analytics capabilities.

## Features

- 📊 **Multiple Trading Strategies**
    - SMA Crossover
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - Custom ML Models (Random Forest, Gradient Boosting)

- 🔬 **Comprehensive Backtesting**
    - Historical data analysis
    - Performance metrics (Sharpe ratio, max drawdown, win rate)
    - Visual equity curves and price charts
    - Trade-by-trade analysis

- 🤖 **Machine Learning**
    - Train custom ML models on historical data
    - Feature engineering with technical indicators
    - Model performance validation

- 📈 **Analytics & Visualization**
    - Interactive charts with Plotly
    - Performance dashboards
    - Risk metrics
    - Trade history

- ⚠️ **Risk Management**
    - Position sizing
    - Stop loss management
    - Maximum drawdown limits

- 🔔 **Alert System**
    - Email notifications
    - SMS alerts via Twilio
    - Trade execution alerts

- 💾 **Data Persistence**
    - SQLite database
    - Trade history
    - Portfolio tracking
    - Performance snapshots

## Project Structure

```
trading_platform/
│
├── main.py                      # Streamlit entry point
├── config.py                    # Configuration settings
├── requirements.txt             # Dependencies
│
├── core/                        # Core components
│   ├── __init__.py
│   ├── database.py             # Database management
│   ├── trading_engine.py       # Trading engine
│   ├── risk_manager.py         # Risk management
│   └── data_fetcher.py         # Data fetching
│
├── strategies/                  # Trading strategies
│   ├── __init__.py
│   ├── base_strategy.py        # Base strategy class
│   ├── sma_crossover.py        # SMA strategy
│   ├── rsi_strategy.py         # RSI strategy
│   ├── macd_strategy.py        # MACD strategy
│   └── ml_strategy.py          # ML strategy
│
├── analytics/                   # Analytics & metrics
│   ├── __init__.py
│   └── performance.py          # Performance calculations
│
├── alerts/                      # Alert system
│   ├── __init__.py
│   └── alert_manager.py        # Alert manager
│
└── ui/                          # UI components
    ├── __init__.py
    ├── dashboard.py            # Portfolio dashboard
    ├── backtest.py             # Backtest interface
    ├── ml_builder.py           # ML model builder
    └── configuration.py        # Configuration panel
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/OnaDebrah/algo.git
cd trading_platform
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Run the application

```bash
streamlit run main.py
```

The application will open in your default web browser at `http://localhost:8501`

### Quick Start Guide

1. **Portfolio Dashboard**
    - View overall portfolio performance
    - See recent trades and metrics

2. **Backtest Tab**
    - Select a stock symbol (e.g., AAPL, MSFT, GOOGL)
    - Choose a trading strategy
    - Configure parameters
    - Run backtest and analyze results

3. **ML Strategy Builder**
    - Train custom ML models
    - Select training period
    - Evaluate model performance
    - Use trained models in backtests

4. **Configuration**
    - Set up email/SMS alerts
    - Configure risk management parameters
    - Export trade data
    - View system information

## Configuration

### Default Settings

Edit `config.py` to customize default settings:

- Initial capital
- Position sizing
- Stop loss percentages
- Strategy parameters
- Database path

### Email Alerts (Gmail)

1. Enable 2-Step Verification in your Google Account
2. Generate an App Password
3. Configure in the Configuration tab:
    - SMTP Server: `smtp.gmail.com`
    - SMTP Port: `587`
    - Use your App Password

### SMS Alerts (Twilio)

1. Create a Twilio account
2. Get your Account SID and Auth Token
3. Configure in the Configuration tab

## Development

### Project Architecture

- **Core Layer**: Database, trading engine, risk management
- **Strategy Layer**: Pluggable trading strategies
- **Analytics Layer**: Performance metrics and calculations
- **UI Layer**: Streamlit components
- **Alerts Layer**: Notification system

### Adding New Strategies

1. Create a new file in `strategies/`
2. Inherit from `BaseStrategy`
3. Implement `generate_signal()` method
4. Import and use in `ui/backtest.py`

Example:

```python
from streamlit.strategies import BaseStrategy


class MyStrategy(BaseStrategy):
    def __init__(self, param1, param2):
        params = {'param1': param1, 'param2': param2}
        super().__init__("My Strategy", params)

    def generate_signal(self, data):
        # Your logic here
        return 1  # Buy signal
```

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black .
flake8 .
```

## Troubleshooting

### Yahoo Finance Data Issues

If you encounter errors fetching data:

1. Check internet connection
2. Verify symbol is correct
3. Try a different period/interval
4. Update yfinance: `pip install --upgrade yfinance`

### IntelliJ/PyCharm Issues

The project includes user-agent headers to prevent blocking in different environments.

## Disclaimer

This software is for educational and research purposes only. It is not financial advice. Trading involves risk, and you should never trade with money you cannot afford to lose.

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For issues and questions:
- Open an issue on GitHub
- Check existing documentation
- Review the code comments

## Roadmap

- [ ] Live trading integration (Alpaca, Interactive Brokers)
- [ ] Additional technical indicators
- [ ] Portfolio optimization
- [ ] Multi-asset support
- [ ] Advanced order types
- [ ] Paper trading mode
- [ ] Web API for programmatic access
- [ ] Mobile app

## Acknowledgments

- Data provided by Yahoo Finance
- Built with Streamlit
- ML powered by scikit-learn
- Charts by Plotly
