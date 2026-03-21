"""Export routes for CSV and PDF report generation."""

import csv
import io
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...api.deps import get_current_active_user
from ...database import get_db
from ...models import User
from ...models.backtest import BacktestRun
from ...models.paper_trading import PaperPortfolio, PaperTrade

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/export", tags=["Export"])


def _csv_response(buf: io.StringIO, filename: str) -> StreamingResponse:
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Backtest CSV ─────────────────────────────────────────────────────────────
@router.get("/backtest/{run_id}/csv")
async def export_backtest_csv(
    run_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Export backtest results as CSV."""
    result = await db.execute(
        select(BacktestRun).where(BacktestRun.id == run_id, BacktestRun.user_id == current_user.id)
    )
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="Backtest not found")

    buf = io.StringIO()
    writer = csv.writer(buf)

    # Summary section
    writer.writerow(["Backtest Report"])
    writer.writerow(["Generated", datetime.utcnow().isoformat()])
    writer.writerow([])
    writer.writerow(["Metric", "Value"])
    writer.writerow(["Strategy", (run.strategy_config or {}).get("strategy", "N/A")])
    writer.writerow(["Symbols", ", ".join(run.symbols or [])])
    writer.writerow(["Period", run.period])
    writer.writerow(["Interval", run.interval])
    writer.writerow(["Initial Capital", f"${run.initial_capital:,.2f}"])
    writer.writerow(["Final Equity", f"${run.final_equity:,.2f}" if run.final_equity else "N/A"])
    writer.writerow(["Total Return", f"{run.total_return_pct:.2f}%" if run.total_return_pct else "N/A"])
    writer.writerow(["Sharpe Ratio", f"{run.sharpe_ratio:.2f}" if run.sharpe_ratio else "N/A"])
    writer.writerow(["Max Drawdown", f"{run.max_drawdown:.2f}%" if run.max_drawdown else "N/A"])
    writer.writerow(["Win Rate", f"{run.win_rate:.2f}%" if run.win_rate else "N/A"])
    writer.writerow(["Total Trades", run.total_trades or 0])

    # Extended metrics
    ext = run.extended_results or {}
    if ext:
        writer.writerow([])
        writer.writerow(["Extended Metrics"])
        for key in ["sortino_ratio", "calmar_ratio", "profit_factor", "avg_win", "avg_loss", "volatility", "var_95"]:
            if key in ext:
                writer.writerow([key.replace("_", " ").title(), f"{ext[key]:.4f}"])

    # Trades section
    trades = run.trades_json or []
    if trades:
        writer.writerow([])
        writer.writerow(["Trade History"])
        if trades:
            headers = list(trades[0].keys())
            writer.writerow(headers)
            for trade in trades:
                writer.writerow([trade.get(h, "") for h in headers])

    # Equity curve
    equity = run.equity_curve or []
    if equity:
        writer.writerow([])
        writer.writerow(["Equity Curve"])
        if isinstance(equity[0], dict):
            eq_headers = list(equity[0].keys())
            writer.writerow(eq_headers)
            for point in equity:
                writer.writerow([point.get(h, "") for h in eq_headers])
        else:
            writer.writerow(["Index", "Equity"])
            for i, val in enumerate(equity):
                writer.writerow([i, val])

    strategy_name = (run.strategy_config or {}).get("strategy", "backtest")
    filename = f"{strategy_name}_backtest_{run_id}.csv"
    return _csv_response(buf, filename)


# ── Backtest PDF ─────────────────────────────────────────────────────────────
@router.get("/backtest/{run_id}/pdf")
async def export_backtest_pdf(
    run_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Export backtest results as PDF report."""
    result = await db.execute(
        select(BacktestRun).where(BacktestRun.id == run_id, BacktestRun.user_id == current_user.id)
    )
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="Backtest not found")

    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError:
        raise HTTPException(status_code=500, detail="PDF generation not available — reportlab not installed")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=0.75 * inch, bottomMargin=0.5 * inch)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    title_style = ParagraphStyle("Title", parent=styles["Title"], fontSize=20, spaceAfter=20)
    elements.append(Paragraph("Oraculum Backtest Report", title_style))
    elements.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", styles["Normal"]))
    elements.append(Spacer(1, 20))

    # Strategy info
    strategy_name = (run.strategy_config or {}).get("strategy", "N/A")
    symbols = ", ".join(run.symbols or [])
    info_data = [
        ["Strategy", strategy_name],
        ["Symbols", symbols],
        ["Period / Interval", f"{run.period} / {run.interval}"],
        ["Initial Capital", f"${run.initial_capital:,.2f}"],
    ]
    info_table = Table(info_data, colWidths=[2 * inch, 4 * inch])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.Color(0.15, 0.15, 0.25)),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
        ("BACKGROUND", (1, 0), (1, -1), colors.Color(0.1, 0.1, 0.2)),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.Color(0.3, 0.3, 0.4)),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 20))

    # Performance metrics
    elements.append(Paragraph("Performance Summary", styles["Heading2"]))
    ext = run.extended_results or {}
    metrics_data = [
        ["Metric", "Value"],
        ["Total Return", f"{run.total_return_pct:.2f}%" if run.total_return_pct else "N/A"],
        ["Final Equity", f"${run.final_equity:,.2f}" if run.final_equity else "N/A"],
        ["Sharpe Ratio", f"{run.sharpe_ratio:.2f}" if run.sharpe_ratio else "N/A"],
        ["Max Drawdown", f"{run.max_drawdown:.2f}%" if run.max_drawdown else "N/A"],
        ["Win Rate", f"{run.win_rate:.2f}%" if run.win_rate else "N/A"],
        ["Total Trades", str(run.total_trades or 0)],
        ["Sortino Ratio", f"{ext.get('sortino_ratio', 0):.2f}"],
        ["Profit Factor", f"{ext.get('profit_factor', 0):.2f}"],
        ["Volatility", f"{ext.get('volatility', 0):.2f}%"],
    ]
    metrics_table = Table(metrics_data, colWidths=[3 * inch, 3 * inch])
    metrics_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.2, 0.15, 0.4)),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
        ("BACKGROUND", (0, 1), (-1, -1), colors.Color(0.1, 0.1, 0.2)),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.Color(0.3, 0.3, 0.4)),
    ]))
    elements.append(metrics_table)
    elements.append(Spacer(1, 20))

    # Trade summary (first 50 trades)
    trades = run.trades_json or []
    if trades:
        elements.append(Paragraph(f"Trade History ({len(trades)} trades)", styles["Heading2"]))
        trade_headers = ["#", "Type", "Price", "Date", "PnL"]
        trade_data = [trade_headers]
        for i, t in enumerate(trades[:50], 1):
            trade_data.append([
                str(i),
                str(t.get("type", t.get("side", ""))),
                f"${t.get('price', 0):.2f}" if t.get("price") else "",
                str(t.get("date", t.get("entry_date", "")))[:10],
                f"${t.get('pnl', t.get('profit', 0)):.2f}" if t.get("pnl", t.get("profit")) else "",
            ])
        trade_table = Table(trade_data, colWidths=[0.5 * inch, 1 * inch, 1.2 * inch, 1.5 * inch, 1.2 * inch])
        trade_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.2, 0.15, 0.4)),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
            ("BACKGROUND", (0, 1), (-1, -1), colors.Color(0.1, 0.1, 0.2)),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.Color(0.3, 0.3, 0.4)),
        ]))
        elements.append(trade_table)
        if len(trades) > 50:
            elements.append(Paragraph(f"... and {len(trades) - 50} more trades", styles["Normal"]))

    doc.build(elements)
    buf.seek(0)

    strategy_name = (run.strategy_config or {}).get("strategy", "backtest")
    filename = f"{strategy_name}_report_{run_id}.pdf"
    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Paper Trading CSV ────────────────────────────────────────────────────────
@router.get("/paper/{portfolio_id}/csv")
async def export_paper_trades_csv(
    portfolio_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Export paper trading history as CSV."""
    result = await db.execute(
        select(PaperPortfolio).where(
            PaperPortfolio.id == portfolio_id,
            PaperPortfolio.user_id == current_user.id,
        )
    )
    portfolio = result.scalar_one_or_none()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Paper portfolio not found")

    result = await db.execute(
        select(PaperTrade)
        .where(PaperTrade.portfolio_id == portfolio_id)
        .order_by(PaperTrade.executed_at.desc())
    )
    trades = result.scalars().all()

    buf = io.StringIO()
    writer = csv.writer(buf)

    writer.writerow(["Paper Trading Report"])
    writer.writerow(["Portfolio", portfolio.name])
    writer.writerow(["Cash Balance", f"${portfolio.cash_balance:,.2f}"])
    writer.writerow(["Generated", datetime.utcnow().isoformat()])
    writer.writerow([])
    writer.writerow(["Date", "Symbol", "Side", "Quantity", "Price", "Total", "Source"])

    for trade in trades:
        writer.writerow([
            trade.executed_at.isoformat() if trade.executed_at else "",
            trade.symbol,
            trade.side,
            trade.quantity,
            f"{trade.price:.2f}",
            f"{trade.quantity * trade.price:.2f}",
            getattr(trade, "source", "manual"),
        ])

    return _csv_response(buf, f"paper_trades_{portfolio.name}.csv")
