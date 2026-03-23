"""
Paper trading service — manual + strategy-driven trading with virtual money.
"""

import json
import logging
from datetime import datetime, time
from typing import Optional

import numpy as np
import pytz
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..core.data_fetcher import fetch_quote, fetch_stock_data
from ..models.paper_trading import PaperEquitySnapshot, PaperPortfolio, PaperPosition, PaperTrade
from ..schemas.paper_trading import PaperPerformanceOut, StrategySignalResult
from ..strategies.base_strategy import normalize_signal
from ..strategies.catelog.strategy_catalog import get_catalog

logger = logging.getLogger(__name__)

SLIPPAGE_PCT = 0.0005  # 0.05%

# Mapping from data interval to the minimum lookback period for signal generation
INTERVAL_PERIOD_MAP = {
    "1m": "5d",
    "5m": "1mo",
    "15m": "1mo",
    "30m": "3mo",
    "1h": "3mo",
    "4h": "6mo",
    "1d": "1y",
    "1wk": "2y",
}

# US market hours (NYSE/NASDAQ)
_ET = pytz.timezone("US/Eastern")
_MARKET_OPEN = time(9, 30)
_MARKET_CLOSE = time(16, 0)


def is_market_open() -> bool:
    """Check if US stock market is currently open (Mon-Fri 9:30-16:00 ET)."""
    now_et = datetime.now(_ET)
    if now_et.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    return _MARKET_OPEN <= now_et.time() <= _MARKET_CLOSE


class PaperTradingService:
    """Service for managing paper trading portfolios."""

    # ── Portfolio CRUD ───────────────────────────────────────────────

    @staticmethod
    async def create_portfolio(
        db: AsyncSession,
        user_id: int,
        name: str,
        initial_cash: float,
        strategy_key: Optional[str] = None,
        strategy_params: Optional[dict] = None,
        strategy_symbol: Optional[str] = None,
        trade_quantity: float = 100,
        data_interval: str = "1d",
    ) -> PaperPortfolio:
        portfolio = PaperPortfolio(
            user_id=user_id,
            name=name,
            initial_cash=initial_cash,
            current_cash=initial_cash,
            strategy_key=strategy_key,
            strategy_params=json.dumps(strategy_params) if strategy_params else None,
            strategy_symbol=strategy_symbol.upper() if strategy_symbol else None,
            trade_quantity=trade_quantity,
            data_interval=data_interval or "1d",
        )
        db.add(portfolio)
        await db.commit()

        # Re-query with eager loading
        query = select(PaperPortfolio).where(PaperPortfolio.id == portfolio.id).options(selectinload(PaperPortfolio.positions))
        result = await db.execute(query)
        return result.scalar_one()

    @staticmethod
    async def get_portfolios(db: AsyncSession, user_id: int) -> list[PaperPortfolio]:
        query = (
            select(PaperPortfolio)
            .where(PaperPortfolio.user_id == user_id, PaperPortfolio.is_active == True)  # noqa: E712
            .options(selectinload(PaperPortfolio.positions))
            .order_by(PaperPortfolio.created_at.desc())
        )
        result = await db.execute(query)
        return list(result.scalars().all())

    @staticmethod
    async def get_portfolio(db: AsyncSession, portfolio_id: int, user_id: int) -> Optional[PaperPortfolio]:
        query = (
            select(PaperPortfolio)
            .where(PaperPortfolio.id == portfolio_id, PaperPortfolio.user_id == user_id)
            .options(selectinload(PaperPortfolio.positions))
        )
        result = await db.execute(query)
        return result.scalar_one_or_none()

    @staticmethod
    async def deactivate_portfolio(db: AsyncSession, portfolio_id: int, user_id: int) -> bool:
        query = select(PaperPortfolio).where(PaperPortfolio.id == portfolio_id, PaperPortfolio.user_id == user_id)
        result = await db.execute(query)
        portfolio = result.scalar_one_or_none()
        if not portfolio:
            return False
        portfolio.is_active = False
        await db.commit()
        return True

    # ── Trading ──────────────────────────────────────────────────────

    @staticmethod
    async def place_trade(
        db: AsyncSession,
        portfolio_id: int,
        user_id: int,
        symbol: str,
        side: str,
        quantity: float,
        source: str = "manual",
    ) -> PaperTrade:
        """Place a paper trade using real market quotes."""
        portfolio = await PaperTradingService.get_portfolio(db, portfolio_id, user_id)
        if not portfolio:
            raise ValueError("Portfolio not found")

        symbol = symbol.upper()

        try:
            quote = await fetch_quote(symbol)
            price = quote.get("price") or quote.get("latestPrice") or quote.get("close")
            if not price or price <= 0:
                raise ValueError(f"Invalid price for {symbol}")
        except Exception as e:
            raise ValueError(f"Cannot fetch quote for {symbol}: {e}")

        # Apply slippage
        slippage = price * SLIPPAGE_PCT
        if side == "buy":
            exec_price = price + slippage
        else:
            exec_price = price - slippage

        total_cost = exec_price * quantity
        realized_pnl = None

        if side == "buy":
            # Validate cash
            if total_cost > portfolio.current_cash:
                raise ValueError(f"Insufficient cash: need ${total_cost:.2f}, have ${portfolio.current_cash:.2f}")

            portfolio.current_cash -= total_cost

            # Update or create position
            existing = next((p for p in portfolio.positions if p.symbol == symbol), None)
            if existing:
                # Average down
                total_qty = existing.quantity + quantity
                existing.avg_entry_price = (existing.avg_entry_price * existing.quantity + exec_price * quantity) / total_qty
                existing.quantity = total_qty
                existing.current_price = price
            else:
                pos = PaperPosition(
                    portfolio_id=portfolio_id,
                    symbol=symbol,
                    quantity=quantity,
                    avg_entry_price=exec_price,
                    current_price=price,
                )
                db.add(pos)

        elif side == "sell":
            # Find position
            existing = next((p for p in portfolio.positions if p.symbol == symbol), None)
            if not existing or existing.quantity < quantity:
                avail = existing.quantity if existing else 0
                raise ValueError(f"Insufficient shares: need {quantity}, have {avail}")

            # Calculate realized P&L
            realized_pnl = (exec_price - existing.avg_entry_price) * quantity
            portfolio.current_cash += total_cost

            existing.quantity -= quantity
            existing.current_price = price

            # Remove position if fully closed
            if existing.quantity <= 0.0001:
                await db.delete(existing)

        # Record trade
        trade = PaperTrade(
            portfolio_id=portfolio_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=exec_price,
            slippage=slippage,
            total_cost=total_cost,
            realized_pnl=realized_pnl,
            source=source,
        )
        db.add(trade)
        await db.commit()
        await db.refresh(trade)

        # Snapshot equity after trade
        await PaperTradingService._snapshot_equity(db, portfolio)

        return trade

    # ── Queries ──────────────────────────────────────────────────────

    @staticmethod
    async def get_trades(db: AsyncSession, portfolio_id: int, user_id: int, limit: int = 100) -> list[PaperTrade]:
        # Verify ownership
        portfolio = await PaperTradingService.get_portfolio(db, portfolio_id, user_id)
        if not portfolio:
            raise ValueError("Portfolio not found")

        query = select(PaperTrade).where(PaperTrade.portfolio_id == portfolio_id).order_by(PaperTrade.executed_at.desc()).limit(limit)
        result = await db.execute(query)
        return list(result.scalars().all())

    @staticmethod
    async def get_equity_history(db: AsyncSession, portfolio_id: int, user_id: int) -> list[PaperEquitySnapshot]:
        portfolio = await PaperTradingService.get_portfolio(db, portfolio_id, user_id)
        if not portfolio:
            raise ValueError("Portfolio not found")

        query = select(PaperEquitySnapshot).where(PaperEquitySnapshot.portfolio_id == portfolio_id).order_by(PaperEquitySnapshot.timestamp)
        result = await db.execute(query)
        return list(result.scalars().all())

    @staticmethod
    async def get_performance(db: AsyncSession, portfolio_id: int, user_id: int) -> PaperPerformanceOut:
        portfolio = await PaperTradingService.get_portfolio(db, portfolio_id, user_id)
        if not portfolio:
            raise ValueError("Portfolio not found")

        # Get all trades
        query = select(PaperTrade).where(PaperTrade.portfolio_id == portfolio_id)
        result = await db.execute(query)
        trades = list(result.scalars().all())

        # Calculate current equity
        positions_value = sum((p.current_price or p.avg_entry_price) * p.quantity for p in portfolio.positions)
        equity = portfolio.current_cash + positions_value
        total_return = equity - portfolio.initial_cash
        total_return_pct = (total_return / portfolio.initial_cash) * 100 if portfolio.initial_cash > 0 else 0

        # Trade stats
        closed_trades = [t for t in trades if t.realized_pnl is not None]
        winning = [t for t in closed_trades if t.realized_pnl > 0]
        losing = [t for t in closed_trades if t.realized_pnl <= 0]
        win_rate = len(winning) / len(closed_trades) * 100 if closed_trades else 0
        avg_win = np.mean([t.realized_pnl for t in winning]) if winning else 0
        avg_loss = np.mean([t.realized_pnl for t in losing]) if losing else 0

        # Max drawdown from equity snapshots
        snapshots = await PaperTradingService.get_equity_history(db, portfolio_id, user_id)
        max_drawdown = 0.0
        if snapshots:
            equities = [s.equity for s in snapshots]
            peak = equities[0]
            for eq in equities:
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak * 100 if peak > 0 else 0
                if dd > max_drawdown:
                    max_drawdown = dd

        # Sharpe ratio (simplified: from equity snapshots)
        sharpe = None
        if len(snapshots) >= 3:
            equities = np.array([s.equity for s in snapshots])
            returns = np.diff(equities) / equities[:-1]
            if np.std(returns) > 0:
                sharpe = round(float(np.mean(returns) / np.std(returns) * np.sqrt(252)), 2)

        return PaperPerformanceOut(
            total_return=round(total_return, 2),
            total_return_pct=round(total_return_pct, 2),
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=round(win_rate, 1),
            avg_win=round(float(avg_win), 2),
            avg_loss=round(float(avg_loss), 2),
            max_drawdown=round(max_drawdown, 2),
            sharpe_ratio=sharpe,
        )

    # ── Enrich positions with live prices ────────────────────────────

    @staticmethod
    async def enrich_positions(portfolio: PaperPortfolio) -> tuple[list[dict], float]:
        """Fetch live prices for all positions, return enriched dicts and total positions value."""
        enriched = []
        positions_value = 0.0

        for pos in portfolio.positions:
            try:
                quote = await fetch_quote(pos.symbol)
                price = quote.get("price") or quote.get("latestPrice") or pos.avg_entry_price
            except Exception:
                price = pos.current_price or pos.avg_entry_price

            market_value = price * pos.quantity
            unrealized_pnl = (price - pos.avg_entry_price) * pos.quantity
            unrealized_pnl_pct = ((price - pos.avg_entry_price) / pos.avg_entry_price * 100) if pos.avg_entry_price > 0 else 0
            positions_value += market_value

            enriched.append(
                {
                    "id": pos.id,
                    "symbol": pos.symbol,
                    "quantity": pos.quantity,
                    "avg_entry_price": round(pos.avg_entry_price, 2),
                    "current_price": round(price, 2),
                    "unrealized_pnl": round(unrealized_pnl, 2),
                    "unrealized_pnl_pct": round(unrealized_pnl_pct, 2),
                    "market_value": round(market_value, 2),
                    "opened_at": pos.opened_at.isoformat() if pos.opened_at else None,
                }
            )

        return enriched, positions_value

    # ── Strategy Execution ────────────────────────────────────────────

    @staticmethod
    async def attach_strategy(
        db: AsyncSession,
        portfolio_id: int,
        user_id: int,
        strategy_key: str,
        strategy_symbol: str,
        strategy_params: Optional[dict] = None,
        trade_quantity: float = 100,
        data_interval: str = "1d",
    ) -> PaperPortfolio:
        """Attach a strategy to an existing paper portfolio."""
        portfolio = await PaperTradingService.get_portfolio(db, portfolio_id, user_id)
        if not portfolio:
            raise ValueError("Portfolio not found")

        # Validate strategy exists in catalog
        catalog = get_catalog()
        if strategy_key not in catalog.strategies:
            available = list(catalog.strategies.keys())
            raise ValueError(f"Unknown strategy '{strategy_key}'. Available: {available[:10]}...")

        portfolio.strategy_key = strategy_key
        portfolio.strategy_params = json.dumps(strategy_params) if strategy_params else None
        portfolio.strategy_symbol = strategy_symbol.upper()
        portfolio.trade_quantity = trade_quantity
        portfolio.data_interval = data_interval or "1d"
        await db.commit()

        # Re-query with eager load
        query = select(PaperPortfolio).where(PaperPortfolio.id == portfolio_id).options(selectinload(PaperPortfolio.positions))
        result = await db.execute(query)
        return result.scalar_one()

    @staticmethod
    async def detach_strategy(db: AsyncSession, portfolio_id: int, user_id: int) -> PaperPortfolio:
        """Remove strategy from a portfolio (keep it manual-only)."""
        portfolio = await PaperTradingService.get_portfolio(db, portfolio_id, user_id)
        if not portfolio:
            raise ValueError("Portfolio not found")

        portfolio.strategy_key = None
        portfolio.strategy_params = None
        portfolio.strategy_symbol = None
        await db.commit()

        query = select(PaperPortfolio).where(PaperPortfolio.id == portfolio_id).options(selectinload(PaperPortfolio.positions))
        result = await db.execute(query)
        return result.scalar_one()

    @staticmethod
    async def run_strategy_signal(
        db: AsyncSession,
        portfolio_id: int,
        user_id: int,
        auto_execute: bool = True,
    ) -> StrategySignalResult:
        """
        Run the attached strategy on current market data and optionally execute the trade.

        1. Fetches 3 months of daily data for the strategy symbol
        2. Instantiates the strategy with stored params
        3. Generates the latest signal (1=buy, -1=sell, 0=hold)
        4. If auto_execute and signal != 0, places the paper trade
        """
        portfolio = await PaperTradingService.get_portfolio(db, portfolio_id, user_id)
        if not portfolio:
            raise ValueError("Portfolio not found")
        if not portfolio.strategy_key or not portfolio.strategy_symbol:
            raise ValueError("No strategy attached to this portfolio")

        symbol = portfolio.strategy_symbol
        strategy_key = portfolio.strategy_key
        params = json.loads(portfolio.strategy_params) if portfolio.strategy_params else {}
        trade_qty = portfolio.trade_quantity or 100

        # 1. Fetch historical data for signal generation (use portfolio's configured interval)
        interval = portfolio.data_interval or "1d"
        period = INTERVAL_PERIOD_MAP.get(interval, "1y")
        try:
            data = await fetch_stock_data(symbol, period, interval)
            if data is None or data.empty:
                raise ValueError(f"No data returned for {symbol}")
        except Exception as e:
            raise ValueError(f"Failed to fetch data for {symbol}: {e}")

        # 2. Instantiate strategy
        catalog = get_catalog()
        try:
            strategy = catalog.create_strategy(strategy_key, **params)
        except Exception as e:
            raise ValueError(f"Failed to create strategy '{strategy_key}': {e}")

        # 3. Generate signal
        try:
            if hasattr(strategy, "generate_signals_vectorized"):
                signals = strategy.generate_signals_vectorized(data)
                raw_signal = int(signals.iloc[-1]) if len(signals) > 0 else 0
            else:
                raw_signal_out = strategy.generate_signal(data)
                normalized = normalize_signal(raw_signal_out)
                raw_signal = int(normalized["signal"])
        except Exception as e:
            logger.error(f"Strategy signal generation failed: {e}")
            raise ValueError(f"Strategy signal generation failed: {e}")

        # Get current price for the response
        try:
            quote = await fetch_quote(symbol)
            current_price = quote.get("price") or quote.get("latestPrice") or quote.get("close") or 0
        except Exception:
            close_col = "Close" if "Close" in data.columns else "close"
            current_price = float(data[close_col].iloc[-1])

        signal_label = {1: "BUY", -1: "SELL", 0: "HOLD"}.get(raw_signal, "HOLD")

        # Data freshness — last bar timestamp
        data_as_of = str(data.index[-1]) if len(data) > 0 else None

        result = StrategySignalResult(
            signal=raw_signal,
            signal_label=signal_label,
            strategy_key=strategy_key,
            symbol=symbol,
            current_price=round(current_price, 2),
            trade_executed=False,
            data_interval=interval,
            market_open=is_market_open(),
            data_as_of=data_as_of,
        )

        # 4. Auto-execute if signal is actionable
        if auto_execute and raw_signal != 0:
            side = "buy" if raw_signal == 1 else "sell"

            # For sell signals, check if we have a position; skip if not
            if side == "sell":
                existing = next((p for p in portfolio.positions if p.symbol == symbol), None)
                if not existing or existing.quantity <= 0:
                    result.trade_detail = f"SELL signal but no {symbol} position — skipped"
                    return result
                # Sell entire position
                trade_qty = existing.quantity

            try:
                trade = await PaperTradingService.place_trade(db, portfolio_id, user_id, symbol, side, trade_qty, source="strategy")
                result.trade_executed = True
                result.trade_detail = f"{side.upper()} {trade_qty} {symbol} @ ${trade.price:.2f}"
            except ValueError as e:
                result.trade_detail = f"Signal={signal_label} but execution failed: {e}"

        elif raw_signal == 0:
            result.trade_detail = "No action — strategy says HOLD"

        return result

    @staticmethod
    async def refresh_equity(db: AsyncSession, portfolio_id: int, user_id: int) -> PaperEquitySnapshot:
        """
        Take an equity snapshot with live prices — use for periodic P&L tracking
        even when no trades are placed (mark-to-market).
        """
        portfolio = await PaperTradingService.get_portfolio(db, portfolio_id, user_id)
        if not portfolio:
            raise ValueError("Portfolio not found")

        # Fetch live prices for all positions
        positions_value = 0.0
        for pos in portfolio.positions:
            try:
                quote = await fetch_quote(pos.symbol)
                price = quote.get("price") or quote.get("latestPrice") or pos.current_price or pos.avg_entry_price
                pos.current_price = price
            except Exception:
                price = pos.current_price or pos.avg_entry_price
            positions_value += price * pos.quantity

        equity = portfolio.current_cash + positions_value

        snapshot = PaperEquitySnapshot(
            portfolio_id=portfolio.id,
            equity=round(equity, 2),
            cash=round(portfolio.current_cash, 2),
            positions_value=round(positions_value, 2),
        )
        db.add(snapshot)
        await db.commit()
        await db.refresh(snapshot)
        return snapshot

    # ── Internal helpers ─────────────────────────────────────────────

    @staticmethod
    async def _snapshot_equity(db: AsyncSession, portfolio: PaperPortfolio):
        """Save an equity snapshot after a trade."""
        positions_value = sum((p.current_price or p.avg_entry_price) * p.quantity for p in portfolio.positions)
        equity = portfolio.current_cash + positions_value

        snapshot = PaperEquitySnapshot(
            portfolio_id=portfolio.id,
            equity=round(equity, 2),
            cash=round(portfolio.current_cash, 2),
            positions_value=round(positions_value, 2),
        )
        db.add(snapshot)
        await db.commit()
