"""
Research Lab routes — interactive Python code playground.

Provides sandboxed Python execution against platform data APIs,
code templates, and execution history.
"""

import asyncio
import io
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from typing import Any, Dict, List

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, HTTPException

from ...schemas.research import SuggestResponse, SuggestRequest, ExecuteResponse, ExecuteRequest, HistoryItem, CodeTemplate
from ...api.deps import enforce_endpoint_rate_limit, get_current_active_user
from ...config import DEFAULT_INITIAL_CAPITAL
from ...config import settings
from ...models.user import User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/research", tags=["Research"])

# ── Safe modules whitelist — user code can import these ───────────────────

_SAFE_MODULES = frozenset(
    {
        "math",
        "statistics",
        "datetime",
        "json",
        "collections",
        "itertools",
        "functools",
        "decimal",
        "fractions",
        "random",
        "string",
        "re",
        "dataclasses",
        "typing",
        "enum",
        "copy",
        "operator",
        "textwrap",
        "csv",
        "io",
        "base64",
        "hashlib",
        "uuid",
        # Data science
        "numpy",
        "np",
        "pandas",
        "pd",
        # Plotting
        "matplotlib",
        "matplotlib.pyplot",
        "matplotlib.patches",
        "matplotlib.dates",
        "matplotlib.ticker",
        "matplotlib.colors",
        "matplotlib.cm",
        "matplotlib.gridspec",
        "matplotlib.figure",
    }
)


def _safe_import(name, *args, **kwargs):
    """Custom __import__ that only allows whitelisted modules."""
    top_level = name.split(".")[0]
    if name in _SAFE_MODULES or top_level in _SAFE_MODULES:
        return __import__(name, *args, **kwargs)
    raise ImportError(
        f"Module '{name}' is not available in the sandbox. "
        "Available: numpy, pandas, matplotlib, math, statistics, datetime, "
        "json, collections, itertools, random, re"
    )


_BLOCKED_MODULES = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "signal",
        "socket",
        "http",
        "urllib",
        "requests",
        "ctypes",
        "importlib",
        "pathlib",
        "tempfile",
        "glob",
        "pickle",
        "shelve",
        "multiprocessing",
        "threading",
        "asyncio",
        "concurrent",
        "webbrowser",
        "code",
        "codeop",
        "compileall",
        "py_compile",
    }
)


# ── Sandbox execution (runs in a separate process) ────────────────────────


def _execute_in_sandbox(code: str) -> Dict[str, Any]:
    """Run user code in a restricted namespace. Called inside ProcessPoolExecutor."""
    import base64
    import math
    import statistics
    from datetime import datetime, timedelta, date

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Validate code doesn't try to import blocked modules
    for mod in _BLOCKED_MODULES:
        # Check for import statements targeting blocked modules
        if f"import {mod}" in code or f"from {mod}" in code:
            return {
                "output": "",
                "error": f"Import of '{mod}' is not allowed for security reasons.",
                "variables": {},
            }

    # Build safe builtins
    safe_builtins = {}
    import builtins as _builtins

    _ALLOWED_BUILTINS = {
        "abs",
        "all",
        "any",
        "bin",
        "bool",
        "bytes",
        "callable",
        "chr",
        "complex",
        "dict",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "format",
        "frozenset",
        "hash",
        "hex",
        "id",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "object",
        "oct",
        "ord",
        "pow",
        "print",
        "property",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
        "True",
        "False",
        "None",
        "Exception",
        "ValueError",
        "TypeError",
        "KeyError",
        "IndexError",
        "RuntimeError",
        "StopIteration",
        "ZeroDivisionError",
        "ArithmeticError",
        "AttributeError",
    }
    for name in _ALLOWED_BUILTINS:
        if hasattr(_builtins, name):
            safe_builtins[name] = getattr(_builtins, name)

    # Use the module-level whitelist-based __import__
    safe_builtins["__import__"] = _safe_import

    # Data helper functions — route through ProviderFactory
    # ProviderFactory is async, but sandbox runs in ProcessPoolExecutor (sync),
    # so we bridge with asyncio.run() in a fresh event loop per call.
    from ...core.data.providers.providers import ProviderFactory

    _provider = ProviderFactory()

    def fetch_data(symbol: str, period: str = "1y") -> "pd.DataFrame":
        """Fetch historical OHLCV data via ProviderFactory (auto-routes crypto to CoinGecko)."""
        import asyncio as _aio

        df = _aio.run(_provider.fetch_data(symbol, period, "1d"))
        if df.empty:
            raise ValueError(f"No data found for symbol '{symbol}'")
        return df

    def fetch_quote(symbol: str) -> dict:
        """Fetch current quote data via ProviderFactory."""
        import asyncio as _aio

        return _aio.run(_provider.get_quote(symbol))

    def fetch_crypto(symbol: str) -> dict:
        """Fetch crypto data via ProviderFactory. Use symbols like 'BTC-USD', 'ETH-USD', or just 'BTC'."""
        import asyncio as _aio

        if not symbol.upper().endswith(("-USD", "-USDT", "-BTC", "-ETH")):
            symbol = f"{symbol}-USD"
        return _aio.run(_provider.get_quote(symbol))

    # Import data-science libraries in process scope
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        return {
            "output": "",
            "error": "Required libraries (pandas, numpy) not available.",
            "variables": {},
            "plots": [],
        }

    def backtest_strategy(symbol: str, strategy_fn, period: str = "1y", initial_capital: float = DEFAULT_INITIAL_CAPITAL):
        """
        Backtest a simple strategy. strategy_fn receives a DataFrame and returns a Series of signals (1=buy, -1=sell, 0=hold).
        Returns a dict with performance metrics.
        """
        df = fetch_data(symbol, period=period)
        if df.empty:
            raise ValueError(f"No data for {symbol}")

        signals = strategy_fn(df)
        # Simple backtest engine
        position = 0
        cash = initial_capital
        shares = 0
        portfolio_values = []
        trades = []

        for i, (idx, row) in enumerate(df.iterrows()):
            signal = signals.iloc[i] if i < len(signals) else 0
            price = row["Close"]

            if signal == 1 and position == 0:  # Buy
                shares = int(cash / price)
                cash -= shares * price
                position = 1
                trades.append({"date": str(idx.date()), "action": "BUY", "price": round(price, 2), "shares": shares})
            elif signal == -1 and position == 1:  # Sell
                cash += shares * price
                trades.append({"date": str(idx.date()), "action": "SELL", "price": round(price, 2), "shares": shares})
                shares = 0
                position = 0

            portfolio_values.append(cash + shares * price)

        final_value = portfolio_values[-1] if portfolio_values else initial_capital
        total_return = (final_value - initial_capital) / initial_capital * 100

        # Buy and hold comparison
        buy_hold_return = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100

        # Max drawdown
        peak = pd.Series(portfolio_values).expanding().max()
        drawdown = (pd.Series(portfolio_values) - peak) / peak
        max_dd = drawdown.min() * 100

        return {
            "initial_capital": initial_capital,
            "final_value": round(final_value, 2),
            "total_return": round(total_return, 2),
            "buy_hold_return": round(buy_hold_return, 2),
            "max_drawdown": round(max_dd, 2),
            "num_trades": len(trades),
            "trades": trades[:20],  # limit to last 20 trades for display
            "portfolio_values": [round(v, 2) for v in portfolio_values[:: max(1, len(portfolio_values) // 100)]],
            # downsample to ~100 points
        }

    # Build execution namespace
    namespace: Dict[str, Any] = {
        "__builtins__": safe_builtins,
        "pd": pd,
        "np": np,
        "plt": plt,
        "datetime": datetime,
        "timedelta": timedelta,
        "date": date,
        "json": json,
        "math": math,
        "statistics": statistics,
        "fetch_data": fetch_data,
        "fetch_quote": fetch_quote,
        "fetch_crypto": fetch_crypto,
        "backtest_strategy": backtest_strategy,
    }

    # Capture stdout
    stdout_capture = io.StringIO()
    try:
        with redirect_stdout(stdout_capture):
            exec(compile(code, "<research>", "exec"), namespace)  # noqa: S102
    except Exception as exc:
        output = stdout_capture.getvalue()[:50_000]
        plt.close("all")
        return {
            "output": output,
            "error": f"{type(exc).__name__}: {exc}",
            "variables": {},
            "plots": [],
        }

    output = stdout_capture.getvalue()[:50_000]

    # Capture any open matplotlib figures
    plots: List[str] = []
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="#0f172a", edgecolor="none")
        buf.seek(0)
        plots.append(base64.b64encode(buf.read()).decode())
        plt.close(fig)

    # Collect user-defined variables (skip private/internal, callables, modules)
    user_vars: Dict[str, str] = {}
    _skip = {
        "__builtins__",
        "pd",
        "np",
        "plt",
        "datetime",
        "timedelta",
        "date",
        "json",
        "math",
        "statistics",
        "fetch_data",
        "fetch_quote",
        "fetch_crypto",
        "backtest_strategy",
    }
    for k, v in namespace.items():
        if k.startswith("_") or k in _skip:
            continue
        if callable(v) and not isinstance(v, (pd.DataFrame, pd.Series, np.ndarray)):
            continue
        try:
            r = repr(v)
            if len(r) > 2000:
                r = r[:2000] + "... (truncated)"
            user_vars[k] = r
        except Exception:
            user_vars[k] = "<unrepresentable>"

    return {
        "output": output,
        "error": None,
        "variables": user_vars,
        "plots": plots,
    }


# ── Process pool (module-level singleton) ─────────────────────────────────
_executor = ProcessPoolExecutor(max_workers=4)

# ── Templates ─────────────────────────────────────────────────────────────

TEMPLATES: List[Dict[str, str]] = [
    {
        "id": "fetch_stock",
        "name": "Fetch Stock Data",
        "description": "Pull historical OHLCV data for any stock symbol",
        "category": "Data",
        "code": (
            "# Fetch historical stock data\n"
            'symbol = "AAPL"\n'
            'df = fetch_data(symbol, period="6mo")\n'
            "\n"
            'print(f"=== {symbol} Historical Data ===")\n'
            'print(f"Period: {df.index[0].date()} to {df.index[-1].date()}")\n'
            'print(f"Data points: {len(df)}")\n'
            'print(f"\\nLatest 5 days:")\n'
            "print(df.tail())\n"
            'print(f"\\nSummary Statistics:")\n'
            'print(df[["Close", "Volume"]].describe())\n'
        ),
    },
    {
        "id": "technical_analysis",
        "name": "Technical Analysis",
        "description": "Compute SMA, RSI, and MACD indicators",
        "category": "Analysis",
        "code": (
            "# Technical Analysis — SMA, RSI, MACD\n"
            'symbol = "AAPL"\n'
            'df = fetch_data(symbol, period="1y")\n'
            "\n"
            "# Simple Moving Averages\n"
            'df["SMA_20"] = df["Close"].rolling(20).mean()\n'
            'df["SMA_50"] = df["Close"].rolling(50).mean()\n'
            "\n"
            "# RSI (14-period)\n"
            'delta = df["Close"].diff()\n'
            "gain = delta.where(delta > 0, 0).rolling(14).mean()\n"
            "loss = (-delta.where(delta < 0, 0)).rolling(14).mean()\n"
            "rs = gain / loss\n"
            'df["RSI"] = 100 - (100 / (1 + rs))\n'
            "\n"
            "# MACD\n"
            'ema12 = df["Close"].ewm(span=12).mean()\n'
            'ema26 = df["Close"].ewm(span=26).mean()\n'
            'df["MACD"] = ema12 - ema26\n'
            'df["Signal"] = df["MACD"].ewm(span=9).mean()\n'
            "\n"
            "latest = df.iloc[-1]\n"
            'print(f"=== {symbol} Technical Analysis ===")\n'
            "print(f\"Price:    ${latest['Close']:.2f}\")\n"
            "print(f\"SMA 20:   ${latest['SMA_20']:.2f}\")\n"
            "print(f\"SMA 50:   ${latest['SMA_50']:.2f}\")\n"
            "print(f\"RSI:      {latest['RSI']:.1f}\")\n"
            "print(f\"MACD:     {latest['MACD']:.4f}\")\n"
            "print(f\"Signal:   {latest['Signal']:.4f}\")\n"
            "\n"
            "# Signal interpretation\n"
            'if latest["RSI"] > 70:\n'
            '    print("\\n⚠ RSI indicates OVERBOUGHT")\n'
            'elif latest["RSI"] < 30:\n'
            '    print("\\n⚠ RSI indicates OVERSOLD")\n'
            "else:\n"
            '    print("\\n✓ RSI in neutral zone")\n'
        ),
    },
    {
        "id": "portfolio_analysis",
        "name": "Portfolio Analysis",
        "description": "Multi-stock correlation matrix and risk metrics",
        "category": "Analysis",
        "code": (
            "# Portfolio Correlation Analysis\n"
            'symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]\n'
            "\n"
            "returns = pd.DataFrame()\n"
            "for sym in symbols:\n"
            '    df = fetch_data(sym, period="1y")\n'
            '    returns[sym] = df["Close"].pct_change()\n'
            "\n"
            "returns = returns.dropna()\n"
            "\n"
            'print("=== Portfolio Correlation Matrix ===")\n'
            "corr = returns.corr()\n"
            "print(corr.round(3))\n"
            "\n"
            'print("\\n=== Annualized Metrics ===")\n'
            "ann_returns = returns.mean() * 252\n"
            "ann_vol = returns.std() * np.sqrt(252)\n"
            "sharpe = ann_returns / ann_vol\n"
            "\n"
            "metrics = pd.DataFrame({\n"
            '    "Ann Return": ann_returns.map(lambda x: f"{x:.2%}"),\n'
            '    "Ann Vol": ann_vol.map(lambda x: f"{x:.2%}"),\n'
            '    "Sharpe": sharpe.map(lambda x: f"{x:.2f}"),\n'
            "})\n"
            "print(metrics)\n"
        ),
    },
    {
        "id": "crypto_overview",
        "name": "Crypto Market Overview",
        "description": "Fetch and compare major cryptocurrency data",
        "category": "Crypto",
        "code": (
            "# Crypto Market Overview\n"
            'cryptos = ["BTC-USD", "ETH-USD", "SOL-USD"]\n'
            "\n"
            'print("=== Crypto Market Overview ===")\n'
            "for symbol in cryptos:\n"
            "    try:\n"
            "        q = fetch_crypto(symbol)\n"
            "        print(f\"\\n{q['name']} ({q['symbol']})\")\n"
            "        print(f\"  Price:      ${q['price']:,.2f}\")\n"
            "        print(f\"  Change:     {q['changePercent']:.2f}%\")\n"
            "        print(f\"  Volume:     {q['volume']:,.0f}\")\n"
            "        print(f\"  Market Cap: ${q['marketCap']:,.0f}\")\n"
            "    except Exception as e:\n"
            '        print(f"\\n{symbol}: Error — {e}")\n'
            "\n"
            "# Compare BTC vs ETH performance\n"
            'print("\\n=== 90-Day Performance Comparison ===")\n'
            'for symbol in ["BTC-USD", "ETH-USD"]:\n'
            '    df = fetch_data(symbol, period="3mo")\n'
            '    ret = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100\n'
            '    vol = df["Close"].pct_change().std() * np.sqrt(365) * 100\n'
            '    print(f"{symbol}: Return {ret:+.1f}%, Volatility {vol:.1f}%")\n'
        ),
    },
    {
        "id": "statistical_analysis",
        "name": "Statistical Analysis",
        "description": "Returns distribution, Sharpe ratio, and VaR calculation",
        "category": "Analysis",
        "code": (
            "# Statistical Analysis of Returns\n"
            'symbol = "SPY"\n'
            'df = fetch_data(symbol, period="2y")\n'
            'returns = df["Close"].pct_change().dropna()\n'
            "\n"
            'print(f"=== {symbol} Returns Distribution ===")\n'
            'print(f"Mean daily return:   {returns.mean():.6f}")\n'
            'print(f"Std deviation:       {returns.std():.6f}")\n'
            'print(f"Skewness:            {returns.skew():.4f}")\n'
            'print(f"Kurtosis:            {returns.kurtosis():.4f}")\n'
            "\n"
            "# Annualized metrics\n"
            "ann_return = returns.mean() * 252\n"
            "ann_vol = returns.std() * np.sqrt(252)\n"
            "risk_free = 0.05  # 5% risk-free rate\n"
            "sharpe = (ann_return - risk_free) / ann_vol\n"
            "\n"
            'print(f"\\n=== Annualized Metrics ===")\n'
            'print(f"Annual return:       {ann_return:.2%}")\n'
            'print(f"Annual volatility:   {ann_vol:.2%}")\n'
            'print(f"Sharpe ratio:        {sharpe:.3f}")\n'
            "\n"
            "# Value at Risk\n"
            "var_95 = np.percentile(returns, 5)\n"
            "var_99 = np.percentile(returns, 1)\n"
            "cvar_95 = returns[returns <= var_95].mean()\n"
            "\n"
            'print(f"\\n=== Risk Metrics ===")\n'
            'print(f"VaR (95%):           {var_95:.4%}")\n'
            'print(f"VaR (99%):           {var_99:.4%}")\n'
            'print(f"CVaR (95%):          {cvar_95:.4%}")\n'
            "\n"
            "# Max drawdown\n"
            "cumulative = (1 + returns).cumprod()\n"
            "peak = cumulative.expanding().max()\n"
            "drawdown = (cumulative - peak) / peak\n"
            "max_dd = drawdown.min()\n"
            'print(f"Max drawdown:        {max_dd:.2%}")\n'
        ),
    },
    {
        "id": "backtest_strategy",
        "name": "Backtest a Strategy",
        "description": "Test a simple SMA crossover strategy with performance metrics",
        "category": "Strategy",
        "code": (
            "# Backtest: SMA Crossover Strategy\n"
            "def sma_crossover(df):\n"
            '    """Buy when SMA20 crosses above SMA50, sell when it crosses below."""\n'
            '    sma20 = df["Close"].rolling(20).mean()\n'
            '    sma50 = df["Close"].rolling(50).mean()\n'
            "    signals = pd.Series(0, index=df.index)\n"
            "    signals[sma20 > sma50] = 1\n"
            "    signals[sma20 < sma50] = -1\n"
            "    # Only trigger on crossover (change in signal)\n"
            "    signals = signals.diff().fillna(0)\n"
            "    signals[signals > 0] = 1  # Buy signal\n"
            "    signals[signals < 0] = -1  # Sell signal\n"
            "    return signals\n"
            "\n"
            'result = backtest_strategy("AAPL", sma_crossover, period="2y")\n'
            "\n"
            'print(f"=== SMA Crossover Backtest ===")\n'
            "print(f\"Initial Capital: ${result['initial_capital']:,.2f}\")\n"
            "print(f\"Final Value:     ${result['final_value']:,.2f}\")\n"
            "print(f\"Total Return:    {result['total_return']:+.2f}%\")\n"
            "print(f\"Buy & Hold:      {result['buy_hold_return']:+.2f}%\")\n"
            "print(f\"Max Drawdown:    {result['max_drawdown']:.2f}%\")\n"
            "print(f\"Total Trades:    {result['num_trades']}\")\n"
            "\n"
            'print(f"\\n=== Trade Log ===")\n'
            'for t in result["trades"]:\n'
            "    print(f\"  {t['date']}  {t['action']}  ${t['price']:.2f}  x{t['shares']}\")\n"
        ),
    },
    {
        "id": "plot_chart",
        "name": "Plot Stock Chart",
        "description": "Create a price chart with moving averages using matplotlib",
        "category": "Visualization",
        "code": (
            "# Plot Stock Price with Moving Averages\n"
            "import matplotlib.pyplot as plt\n"
            "\n"
            'symbol = "AAPL"\n'
            'df = fetch_data(symbol, period="1y")\n'
            'df["SMA_20"] = df["Close"].rolling(20).mean()\n'
            'df["SMA_50"] = df["Close"].rolling(50).mean()\n'
            "\n"
            "fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1],\n"
            '                                gridspec_kw={"hspace": 0.3})\n'
            "\n"
            "# Price chart\n"
            'ax1.plot(df.index, df["Close"], color="#60a5fa", linewidth=1.5, label="Price")\n'
            'ax1.plot(df.index, df["SMA_20"], color="#f59e0b", linewidth=1, alpha=0.8, label="SMA 20")\n'
            'ax1.plot(df.index, df["SMA_50"], color="#ef4444", linewidth=1, alpha=0.8, label="SMA 50")\n'
            'ax1.fill_between(df.index, df["Close"], alpha=0.1, color="#60a5fa")\n'
            'ax1.set_title(f"{symbol} Price Chart", color="white", fontsize=14, fontweight="bold")\n'
            'ax1.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="white")\n'
            'ax1.set_facecolor("#1e293b")\n'
            'ax1.tick_params(colors="white")\n'
            'ax1.grid(True, alpha=0.2, color="#475569")\n'
            "\n"
            "# Volume chart\n"
            'colors = ["#22c55e" if c >= o else "#ef4444" for c, o in zip(df["Close"], df["Open"])]\n'
            'ax2.bar(df.index, df["Volume"], color=colors, alpha=0.7, width=1)\n'
            'ax2.set_title("Volume", color="white", fontsize=11)\n'
            'ax2.set_facecolor("#1e293b")\n'
            'ax2.tick_params(colors="white")\n'
            'ax2.grid(True, alpha=0.2, color="#475569")\n'
            "\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
            'print(f"Chart generated for {symbol}")\n'
        ),
    },
    {
        "id": "plot_distribution",
        "name": "Returns Distribution",
        "description": "Histogram of daily returns with normal distribution overlay",
        "category": "Visualization",
        "code": (
            "# Returns Distribution Analysis\n"
            "import matplotlib.pyplot as plt\n"
            "\n"
            'symbol = "SPY"\n'
            'df = fetch_data(symbol, period="2y")\n'
            'returns = df["Close"].pct_change().dropna() * 100\n'
            "\n"
            "fig, ax = plt.subplots(figsize=(10, 6))\n"
            'ax.hist(returns, bins=80, density=True, color="#8b5cf6", alpha=0.7, edgecolor="#1e293b")\n'
            "\n"
            "# Normal distribution overlay\n"
            "x = np.linspace(returns.min(), returns.max(), 200)\n"
            "mu, sigma = returns.mean(), returns.std()\n"
            "normal = (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sigma)**2)\n"
            'ax.plot(x, normal, color="#f59e0b", linewidth=2, label=f"Normal(\\u03bc={mu:.3f}, \\u03c3={sigma:.3f})")\n'
            "\n"
            "# VaR lines\n"
            "var95 = np.percentile(returns, 5)\n"
            'ax.axvline(var95, color="#ef4444", linestyle="--", linewidth=1.5, label=f"VaR 95%: {var95:.2f}%")\n'
            "\n"
            'ax.set_title(f"{symbol} Daily Returns Distribution", color="white", fontsize=14, fontweight="bold")\n'
            'ax.set_xlabel("Daily Return (%)", color="white")\n'
            'ax.set_ylabel("Density", color="white")\n'
            'ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="white")\n'
            'ax.set_facecolor("#1e293b")\n'
            'ax.tick_params(colors="white")\n'
            'ax.grid(True, alpha=0.2, color="#475569")\n'
            "\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
            "\n"
            'print(f"Skewness: {returns.skew():.4f}")\n'
            'print(f"Kurtosis: {returns.kurtosis():.4f}")\n'
        ),
    },
]


# ── Redis helper ──────────────────────────────────────────────────────────


async def _get_redis() -> aioredis.Redis:
    return aioredis.from_url(settings.REDIS_URL, decode_responses=True)


# ── Endpoints ─────────────────────────────────────────────────────────────


@router.post(
    "/execute",
    response_model=ExecuteResponse,
    dependencies=[Depends(enforce_endpoint_rate_limit("research_execute", max_requests=10, window_seconds=60))],
)
async def execute_code(
    body: ExecuteRequest,
    current_user: User = Depends(get_current_active_user),
) -> ExecuteResponse:
    """Execute Python code in a sandboxed environment."""
    code = body.code.strip()
    if not code:
        raise HTTPException(status_code=400, detail="Code cannot be empty")

    loop = asyncio.get_running_loop()
    start = time.perf_counter()

    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(_executor, _execute_in_sandbox, code),
            timeout=body.timeout,
        )
    except asyncio.TimeoutError:
        return ExecuteResponse(
            output="",
            error=f"Execution timed out after {body.timeout} seconds.",
            execution_time_ms=body.timeout * 1000,
        )
    except Exception as exc:
        logger.error(f"Research execution error for user {current_user.id}: {exc}", exc_info=True)
        return ExecuteResponse(
            output="",
            error=f"Execution error: {exc}",
            execution_time_ms=int((time.perf_counter() - start) * 1000),
        )

    elapsed_ms = int((time.perf_counter() - start) * 1000)

    response = ExecuteResponse(
        output=result.get("output", ""),
        error=result.get("error"),
        execution_time_ms=elapsed_ms,
        variables=result.get("variables", {}),
        plots=result.get("plots", []),
    )

    # Store in history (fire-and-forget)
    try:
        r = await _get_redis()
        history_entry = {
            "code": code[:10_000],  # limit stored code size
            "output": response.output[:10_000],
            "error": response.error,
            "execution_time_ms": elapsed_ms,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        key = f"research:history:{current_user.id}"
        await r.lpush(key, json.dumps(history_entry))
        await r.ltrim(key, 0, 19)  # keep last 20
        await r.expire(key, 86400)  # 24h TTL
        await r.aclose()
    except Exception as exc:
        logger.warning(f"Failed to store research history: {exc}")

    return response


# ── AI suggestion templates (rule-based, no external AI API needed) ──────

_SUGGESTION_PATTERNS = {
    "momentum": (
        "# Momentum Strategy\n"
        'symbol = "AAPL"\n'
        'df = fetch_data(symbol, period="1y")\n'
        "\n"
        "def momentum_strategy(df):\n"
        '    """Buy when RSI < 30 (oversold), sell when RSI > 70 (overbought)."""\n'
        '    delta = df["Close"].diff()\n'
        "    gain = delta.where(delta > 0, 0).rolling(14).mean()\n"
        "    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()\n"
        "    rs = gain / loss\n"
        "    rsi = 100 - (100 / (1 + rs))\n"
        "    signals = pd.Series(0, index=df.index)\n"
        "    signals[rsi < 30] = 1   # Buy when oversold\n"
        "    signals[rsi > 70] = -1  # Sell when overbought\n"
        "    signals = signals.diff().fillna(0)\n"
        "    signals[signals > 0] = 1\n"
        "    signals[signals < 0] = -1\n"
        "    return signals\n"
        "\n"
        'result = backtest_strategy(symbol, momentum_strategy, period="1y")\n'
        "print(f\"Return: {result['total_return']:+.2f}%  |  Buy&Hold: {result['buy_hold_return']:+.2f}%\")\n"
        "print(f\"Max DD: {result['max_drawdown']:.2f}%  |  Trades: {result['num_trades']}\")\n"
    ),
    "rsi": None,  # falls through to "momentum"
    "oversold": None,
    "overbought": None,
    "correlation": (
        "# Correlation Analysis\n"
        'symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]\n'
        "returns = pd.DataFrame()\n"
        "for sym in symbols:\n"
        '    df = fetch_data(sym, period="1y")\n'
        '    returns[sym] = df["Close"].pct_change()\n'
        "returns = returns.dropna()\n"
        "\n"
        'print("=== Correlation Matrix ===")\n'
        "print(returns.corr().round(3))\n"
        "\n"
        "# Plot heatmap\n"
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots(figsize=(8, 6))\n"
        'im = ax.imshow(returns.corr(), cmap="RdYlGn", vmin=-1, vmax=1)\n'
        "ax.set_xticks(range(len(symbols)))\n"
        "ax.set_yticks(range(len(symbols)))\n"
        'ax.set_xticklabels(symbols, color="white")\n'
        'ax.set_yticklabels(symbols, color="white")\n'
        "for i in range(len(symbols)):\n"
        "    for j in range(len(symbols)):\n"
        '        ax.text(j, i, f"{returns.corr().iloc[i,j]:.2f}", ha="center", va="center", fontsize=10)\n'
        'ax.set_title("Stock Correlation Heatmap", color="white", fontsize=14)\n'
        'ax.set_facecolor("#1e293b")\n'
        "plt.colorbar(im)\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
    ),
    "heatmap": None,
    "plot": (
        "# Plot Stock Data\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        'symbol = "AAPL"\n'
        'df = fetch_data(symbol, period="6mo")\n'
        "\n"
        "fig, ax = plt.subplots(figsize=(12, 6))\n"
        'ax.plot(df.index, df["Close"], color="#60a5fa", linewidth=1.5)\n'
        'ax.fill_between(df.index, df["Close"], alpha=0.1, color="#60a5fa")\n'
        'ax.set_title(f"{symbol} Price", color="white", fontsize=14, fontweight="bold")\n'
        'ax.set_facecolor("#1e293b")\n'
        'ax.tick_params(colors="white")\n'
        'ax.grid(True, alpha=0.2, color="#475569")\n'
        "plt.tight_layout()\n"
        "plt.show()\n"
    ),
    "chart": None,
    "candlestick": (
        "# Candlestick-style Chart\n"
        "import matplotlib.pyplot as plt\n"
        "from matplotlib.patches import Rectangle\n"
        "\n"
        'symbol = "AAPL"\n'
        'df = fetch_data(symbol, period="3mo").tail(60)\n'
        "\n"
        "fig, ax = plt.subplots(figsize=(14, 7))\n"
        "for i, (idx, row) in enumerate(df.iterrows()):\n"
        '    color = "#22c55e" if row["Close"] >= row["Open"] else "#ef4444"\n'
        "    # Wick\n"
        '    ax.plot([i, i], [row["Low"], row["High"]], color=color, linewidth=0.8)\n'
        "    # Body\n"
        '    body_bottom = min(row["Open"], row["Close"])\n'
        '    body_height = abs(row["Close"] - row["Open"])\n'
        "    rect = Rectangle((i-0.35, body_bottom), 0.7, body_height, facecolor=color, edgecolor=color)\n"
        "    ax.add_patch(rect)\n"
        "\n"
        'ax.set_title(f"{symbol} Candlestick Chart", color="white", fontsize=14, fontweight="bold")\n'
        'ax.set_facecolor("#1e293b")\n'
        'ax.tick_params(colors="white")\n'
        'ax.grid(True, alpha=0.2, color="#475569")\n'
        "plt.tight_layout()\n"
        "plt.show()\n"
    ),
    "portfolio": (
        "# Portfolio Optimization (Equal Weight vs Risk Parity)\n"
        'symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]\n'
        "returns = pd.DataFrame()\n"
        "for sym in symbols:\n"
        '    df = fetch_data(sym, period="1y")\n'
        '    returns[sym] = df["Close"].pct_change()\n'
        "returns = returns.dropna()\n"
        "\n"
        "# Equal weight\n"
        "eq_weights = np.array([1/len(symbols)] * len(symbols))\n"
        "eq_ret = (returns * eq_weights).sum(axis=1)\n"
        "\n"
        "# Inverse volatility (simple risk parity)\n"
        "vols = returns.std()\n"
        "inv_vol = 1 / vols\n"
        "rp_weights = inv_vol / inv_vol.sum()\n"
        "rp_ret = (returns * rp_weights.values).sum(axis=1)\n"
        "\n"
        'print("=== Portfolio Weights ===")\n'
        "for sym, ew, rw in zip(symbols, eq_weights, rp_weights):\n"
        '    print(f"  {sym}: Equal={ew:.1%}  RiskParity={rw:.1%}")\n'
        "\n"
        'print(f"\\n=== Annualized Return ===")\n'
        'print(f"  Equal Weight:  {eq_ret.mean()*252:.2%}")\n'
        'print(f"  Risk Parity:   {rp_ret.mean()*252:.2%}")\n'
        'print(f"\\n=== Annualized Volatility ===")\n'
        'print(f"  Equal Weight:  {eq_ret.std()*np.sqrt(252):.2%}")\n'
        'print(f"  Risk Parity:   {rp_ret.std()*np.sqrt(252):.2%}")\n'
    ),
    "optimal": None,
    "weight": None,
    "backtest": (
        "# Backtest: Mean Reversion Strategy\n"
        "def mean_reversion(df):\n"
        '    """Buy when price is >2 std below 20-day mean, sell when above."""\n'
        '    sma = df["Close"].rolling(20).mean()\n'
        '    std = df["Close"].rolling(20).std()\n'
        '    z_score = (df["Close"] - sma) / std\n'
        "    signals = pd.Series(0, index=df.index)\n"
        "    signals[z_score < -2] = 1   # Buy on dip\n"
        "    signals[z_score > 1] = -1   # Sell on recovery\n"
        "    signals = signals.diff().fillna(0)\n"
        "    signals[signals > 0] = 1\n"
        "    signals[signals < 0] = -1\n"
        "    return signals\n"
        "\n"
        'result = backtest_strategy("SPY", mean_reversion, period="2y")\n'
        "print(f\"Return: {result['total_return']:+.2f}%\")\n"
        "print(f\"Buy&Hold: {result['buy_hold_return']:+.2f}%\")\n"
        "print(f\"Max DD: {result['max_drawdown']:.2f}%\")\n"
    ),
    "strategy": None,
    "sma": None,
    "crossover": None,
    "compare": (
        "# Compare Stock Performance\n"
        'symbols = ["AAPL", "MSFT", "GOOGL"]\n'
        "import matplotlib.pyplot as plt\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(12, 6))\n"
        "for sym in symbols:\n"
        '    df = fetch_data(sym, period="1y")\n'
        '    normalized = df["Close"] / df["Close"].iloc[0] * 100\n'
        "    ax.plot(df.index, normalized, linewidth=1.5, label=sym)\n"
        "\n"
        'ax.set_title("Normalized Performance Comparison", color="white", fontsize=14, fontweight="bold")\n'
        'ax.set_ylabel("Normalized Price (base=100)", color="white")\n'
        'ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="white")\n'
        'ax.set_facecolor("#1e293b")\n'
        'ax.tick_params(colors="white")\n'
        'ax.grid(True, alpha=0.2, color="#475569")\n'
        "plt.tight_layout()\n"
        "plt.show()\n"
    ),
    "crypto": (
        "# Crypto Analysis\n"
        'cryptos = ["BTC", "ETH", "SOL"]\n'
        "\n"
        "for sym in cryptos:\n"
        "    q = fetch_crypto(sym)\n"
        "    print(f\"{q['name']}: ${q['price']:,.2f}  ({q['changePercent']:.2f}%)\")\n"
        "\n"
        "# Plot BTC vs ETH\n"
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots(figsize=(12, 6))\n"
        'for sym in ["BTC-USD", "ETH-USD"]:\n'
        '    df = fetch_data(sym, period="6mo")\n'
        '    norm = df["Close"] / df["Close"].iloc[0] * 100\n'
        "    ax.plot(df.index, norm, linewidth=1.5, label=sym)\n"
        'ax.set_title("BTC vs ETH (Normalized)", color="white", fontsize=14)\n'
        'ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="white")\n'
        'ax.set_facecolor("#1e293b")\n'
        'ax.tick_params(colors="white")\n'
        'ax.grid(True, alpha=0.2, color="#475569")\n'
        "plt.tight_layout()\n"
        "plt.show()\n"
    ),
}

# Resolve None aliases to actual patterns
for _k, _v in list(_SUGGESTION_PATTERNS.items()):
    if _v is None:
        # Find first non-None match by walking keys
        for _k2, _v2 in _SUGGESTION_PATTERNS.items():
            if _v2 is not None and _k in (_k2,):
                break
        # Fallback: find a related key
        _SUGGESTION_PATTERNS[_k] = _v

_DEFAULT_SUGGESTION = (
    "# Here's a starting point based on your request:\n"
    'symbol = "AAPL"\n'
    'df = fetch_data(symbol, period="6mo")\n'
    "\n"
    'print(f"=== {symbol} Data ===")\n'
    'print(f"Period: {df.index[0].date()} to {df.index[-1].date()}")\n'
    "print(f\"Current Price: ${df['Close'].iloc[-1]:.2f}\")\n"
    "print(df.tail())\n"
)


def _find_suggestion(prompt: str) -> str:
    """Match user prompt to a code suggestion template via keyword matching."""
    prompt_lower = prompt.lower()
    best_match = None
    best_score = 0

    for keyword, code in _SUGGESTION_PATTERNS.items():
        if code is None:
            continue
        if keyword in prompt_lower:
            # Longer keyword = more specific match = higher priority
            score = len(keyword)
            if score > best_score:
                best_score = score
                best_match = code

    return best_match or _DEFAULT_SUGGESTION


@router.post("/suggest", response_model=SuggestResponse)
async def suggest_code(
    body: SuggestRequest,
    current_user: User = Depends(get_current_active_user),
) -> SuggestResponse:
    """Suggest code based on a natural language description."""
    suggestion = _find_suggestion(body.prompt)
    return SuggestResponse(suggestion=suggestion)


@router.get("/templates", response_model=List[CodeTemplate])
async def get_templates(
    current_user: User = Depends(get_current_active_user),
) -> List[CodeTemplate]:
    """Return available code templates."""
    return [CodeTemplate(**t) for t in TEMPLATES]


@router.get("/history", response_model=List[HistoryItem])
async def get_history(
    current_user: User = Depends(get_current_active_user),
) -> List[HistoryItem]:
    """Return the user's last 20 code executions."""
    try:
        r = await _get_redis()
        key = f"research:history:{current_user.id}"
        raw_items = await r.lrange(key, 0, 19)
        await r.aclose()
        return [HistoryItem(**json.loads(item)) for item in raw_items]
    except Exception as exc:
        logger.warning(f"Failed to fetch research history: {exc}")
        return []
