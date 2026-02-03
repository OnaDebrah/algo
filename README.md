# Advanced Algorithmic Trading Platform

A production-ready algorithmic trading system with backtesting capabilities, machine learning strategies, multi-symbol support, and comprehensive analytics.

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

📊 Visualization & Analytics
1. Interactive Chart Features

Candlestick/OHLC charts instead of just line charts
Technical indicators overlay (show SMA lines, Bollinger Bands, RSI panel below)
Zoom and pan functionality on charts
Volume bars below price chart
Drawdown chart separate from equity curve
Rolling returns heatmap (monthly/yearly performance grid)

2. Advanced Performance Metrics

Sortino Ratio (downside deviation focused)
Calmar Ratio (return/max drawdown)
Information Ratio
Beta and Alpha (vs benchmark like SPY)
VAR (Value at Risk) and CVaR
Maximum consecutive wins/losses
Profit distribution histogram
Monthly/yearly returns table

3. Comparative Analysis

Compare multiple strategy results side-by-side
Strategy vs Buy-and-Hold benchmark overlay
Efficient frontier visualization (for multi-asset)
Correlation matrix heatmap (for multi-asset)
Rolling Sharpe/Sortino charts

🎯 Trading Intelligence
4. Trade Analysis

Trade duration histogram (how long positions are held)
Entry/exit efficiency analysis (did you buy near lows, sell near highs?)
Slippage and commission impact breakdown
MAE/MFE analysis (Maximum Adverse/Favorable Excursion)
Win/loss streaks visualization
Time-of-day/day-of-week performance patterns

5. Risk Management

Position sizing visualization (how much capital in each trade)
Exposure over time (how much of portfolio is invested)
Risk-adjusted returns by position
Leverage usage tracking
Stop-loss effectiveness analysis

🔄 Workflow & Productivity
6. Parameter Optimization

Grid search for optimal parameters (test ranges of parameters)
Walk-forward optimization (validate on out-of-sample data)
Monte Carlo simulation (test robustness with randomized data)
Genetic algorithm optimization
Parameter sensitivity heatmap (how sensitive is performance to each param)

7. Backtesting Enhancements

Out-of-sample testing (train/test split)
Rolling window backtest (test consistency across different periods)
Multiple timeframe analysis (run same strategy on 1h, 4h, 1d)
Transaction cost modeling (realistic slippage based on volume)
Market regime detection (bull/bear/sideways performance)

8. Save & Compare

Save backtest results with unique IDs
Portfolio of saved backtests to compare
Notes/tags on each backtest
Export results to CSV/JSON/PDF
Share backtest results via link

📱 Modern UX Features
9. Real-time Updates

Progress bar during backtest execution
Streaming results as backtest runs (show partial results)
Cancel running backtest button
Queue multiple backtests and run sequentially

10. Smart Defaults & Templates

Strategy presets (conservative/moderate/aggressive)
Pre-configured portfolios (tech stocks, crypto, dividend stocks)
Quick start templates ("Test my strategy on FAANG")
Copy strategy from successful backtests

11. Insights & Recommendations

AI-generated insights ("This strategy performs best in trending markets")
Risk warnings ("High drawdown detected")
Optimization suggestions ("Consider reducing position size")
Similar strategy recommendations

🎨 Specific UI/UX Improvements
12. For Single Asset:

Strategy builder wizard (step-by-step flow)
Price alerts on chart (show support/resistance levels)
News events overlay (show earnings dates, splits)
Comparison with other strategies on same asset

13. For Multi-Asset:

Portfolio rebalancing visualization (show when/how portfolio rebalances)
Individual asset contribution to total return
Correlation changes over time
Sector/category breakdown pie chart
Risk contribution by asset (which assets add most risk)
Pair trading opportunities identification
Portfolio optimization recommendations

14. Data & Execution

Custom date range picker with calendar
Intraday data support (minute-level)
Multiple data sources (choose between Yahoo, Alpha Vantage, etc.)
Paper trading mode (test strategies live without money)
Auto-trade integration (deploy successful strategies)

🚀 Advanced Features
15. Machine Learning Integration

ML-enhanced strategy suggestions
Anomaly detection in backtest results
Predictive analytics for strategy performance
Feature importance for multi-factor strategies

16. Social & Collaboration

Strategy marketplace (share/discover strategies)
Leaderboard of best performing strategies
Comments/discussion on strategies
Clone and modify others' strategies

17. Alerts & Monitoring

Email notifications when backtest completes
Performance degradation alerts
Webhook integrations for custom workflows

🎯 Quick Wins (Easiest to Implement)
If I had to prioritize for immediate impact:

Benchmark comparison (SPY buy-and-hold overlay) ⭐⭐⭐
Save/load backtest results ⭐⭐⭐
Export to CSV/PDF ⭐⭐⭐
Monthly returns table ⭐⭐⭐
Drawdown chart (separate from equity) ⭐⭐
Rolling Sharpe ratio chart ⭐⭐
Parameter grid search ⭐⭐
Trade duration analysis ⭐⭐
Progress indicator during backtest ⭐
Quick strategy templates ⭐
