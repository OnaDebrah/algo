"""
Advanced Rolling News Ticker with Real-Time Updates
===================================================
Professional market data ribbon with auto-refresh, news feed,
and economic calendar integration.
"""

import logging
from datetime import datetime
from typing import Dict, List

import yfinance as yf

import streamlit as st
from streamlit.ui.components.theme import OracleTheme

logger = logging.getLogger(__name__)


@st.cache_data(ttl=120)  # Cache for 2 minutes
def fetch_market_data() -> List[Dict]:
    """
    Fetch real-time market data for major indices and assets.

    Returns:
        List of dictionaries with market data
    """
    symbols = {
        "^GSPC": {"name": "S&P 500", "icon": "📈"},
        "^DJI": {"name": "Dow Jones", "icon": "📊"},
        "^IXIC": {"name": "Nasdaq", "icon": "💻"},
        "^VIX": {"name": "VIX", "icon": "📉"},
        "BTC-USD": {"name": "Bitcoin", "icon": "₿"},
        "ETH-USD": {"name": "Ethereum", "icon": "⟠"},
        "GC=F": {"name": "Gold", "icon": "🥇"},
        "CL=F": {"name": "Crude Oil", "icon": "🛢️"},
        "^TNX": {"name": "10Y Treasury", "icon": "🏦"},
        "EURUSD=X": {"name": "EUR/USD", "icon": "💱"},
    }

    data = []
    for symbol, info in symbols.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")

            if len(hist) >= 2:
                current_price = hist["Close"].iloc[-1]
                prev_close = hist["Close"].iloc[-2]
                change_pct = ((current_price - prev_close) / prev_close) * 100

                data.append(
                    {
                        "symbol": symbol,
                        "name": info["name"],
                        "icon": info["icon"],
                        "price": current_price,
                        "change": change_pct,
                        "prev_close": prev_close,
                    }
                )
        except Exception as e:
            logger.warning(f"Failed to fetch {symbol}: {e}")
            continue

    return data


@st.cache_data(ttl=600)  # Cache for 10 minutes
def fetch_market_news() -> List[Dict]:
    """
    Fetch latest market news headlines.

    Returns:
        List of news items with headlines and timestamps
    """
    try:
        # Using yfinance to get news for major indices
        ticker = yf.Ticker("^GSPC")
        news = ticker.news

        news_items = []
        for item in news[:5]:  # Get top 5 news items
            news_items.append(
                {
                    "title": item.get("title", ""),
                    "publisher": item.get("publisher", ""),
                    "time": datetime.fromtimestamp(item.get("providerPublishTime", 0)),
                }
            )

        return news_items
    except Exception as e:
        logger.warning(f"Failed to fetch news: {e}")
        return []


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_economic_calendar() -> List[Dict]:
    """
    Fetch upcoming economic events (mock data - replace with real API).

    Returns:
        List of economic events
    """
    # This is mock data - integrate with real economic calendar API
    # Options: Trading Economics API, Investing.com API, Alpha Vantage

    events = [
        {"time": "10:00 AM EST", "event": "Fed Interest Rate Decision", "impact": "high", "icon": "🔴"},
        {"time": "2:30 PM EST", "event": "Unemployment Claims", "impact": "medium", "icon": "🟡"},
        {"time": "Tomorrow", "event": "CPI Data Release", "impact": "high", "icon": "🔴"},
    ]

    return events


# ============================================================================
# TICKER RENDERING FUNCTIONS
# ============================================================================


def render_scrolling_ticker(data: List[Dict]):
    """
    Render horizontal scrolling ticker (marquee style) aligned with Oraculum Theme.
    """

    C = OracleTheme.COLORS

    bg_card = C["bg_card"]
    border_color = C["border_subtle"]
    text_muted = C["text_muted"]

    ticker_items = []
    for item in data:
        color = "#00C853" if item["change"] >= 0 else "#FF5252"
        arrow = "▲" if item["change"] >= 0 else "▼"

        ticker_html = f"""
        <div class="ticker-item">
            <span class="ticker-icon">{item['icon']}</span>
            <span class="ticker-name">{item['name']}</span>
            <span class="ticker-price">${item['price']:,.2f}</span>
            <span class="ticker-change" style="color: {color};">
                {arrow} {abs(item['change']):.2f}%
            </span>
        </div>
        """
        ticker_items.append(ticker_html)

    ticker_html_all = "".join(ticker_items * 3)

    st.markdown(
        f"""
        <div class="ticker-wrapper">
            <div class="ticker-container">
                {ticker_html_all}
            </div>
        </div>

        <style>
        .ticker-wrapper {{
            /* Regulation: Matches page background or secondary background */
            background: {bg_card};
            border-top: 1px solid {border_color};
            border-bottom: 1px solid {border_color};

            overflow: hidden;
            padding: 8px 0;

            /* Regulation: Standardizes width to page margins */
            width: 100%;
            margin: 0 auto 2rem auto;
            border-radius: 12px;
            box-shadow: inset 0 0 20px rgba(0,0,0,0.2);
        }}

        .ticker-container {{
            display: flex;
            /* Regulation: Adjust speed based on item count */
            animation: scroll 40s linear infinite;
            width: fit-content;
        }}

        .ticker-item {{
            display: inline-flex;
            align-items: center;
            gap: 12px;
            padding: 6px 24px;
            border-right: 1px solid {border_color};
            white-space: nowrap;
        }}

        .ticker-name {{
            font-size: 0.7rem;
            color: {text_muted};
            text-transform: uppercase;
            font-weight: 700;
        }}

        @keyframes scroll {{
            0% {{ transform: translateX(0); }}
            100% {{ transform: translateX(-33.33%); }}
        }}

        .ticker-wrapper:hover .ticker-container {{
            animation-play-state: paused;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_grid_ticker(data: List[Dict]):
    """
    Render static grid layout ticker.

    Args:
        data: List of market data dictionaries
    """
    # Limit to top 8 items for clean grid
    display_data = data[:8]

    cols = st.columns(len(display_data))

    for i, item in enumerate(display_data):
        color = "#00C853" if item["change"] >= 0 else "#FF5252"
        arrow = "▲" if item["change"] >= 0 else "▼"

        with cols[i]:
            st.markdown(
                f"""
                <div class="grid-ticker-item">
                    <div class="grid-ticker-header">
                        <span class="grid-ticker-icon">{item['icon']}</span>
                        <span class="grid-ticker-name">{item['name']}</span>
                    </div>
                    <div class="grid-ticker-price">${item['price']:,.2f}</div>
                    <div class="grid-ticker-change" style="color: {color};">
                        {arrow} {abs(item['change']):.2f}%
                    </div>
                </div>

                <style>
                .grid-ticker-item {{
                    background: linear-gradient(135deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.01) 100%);
                    padding: 12px;
                    border-radius: 8px;
                    border-left: 3px solid {color};
                    text-align: center;
                    transition: all 0.3s ease;
                }}

                .grid-ticker-item:hover {{
                    background: rgba(255,255,255,0.08);
                    transform: translateY(-4px);
                    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
                }}

                .grid-ticker-header {{
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 6px;
                    margin-bottom: 8px;
                }}

                .grid-ticker-icon {{
                    font-size: 1.2rem;
                }}

                .grid-ticker-name {{
                    font-size: 0.7rem;
                    color: #9AA4B2;
                    text-transform: uppercase;
                    letter-spacing: 0.05em;
                    font-weight: 600;
                }}

                .grid-ticker-price {{
                    font-size: 1.1rem;
                    font-weight: 700;
                    color: #E6EAF2;
                    margin-bottom: 4px;
                }}

                .grid-ticker-change {{
                    font-size: 0.85rem;
                    font-weight: 600;
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )


def render_news_feed(news_items: List[Dict]):
    """
    Render scrolling news feed.

    Args:
        news_items: List of news dictionaries
    """
    if not news_items:
        return

    news_html = []
    for item in news_items:
        time_str = item["time"].strftime("%H:%M") if isinstance(item["time"], datetime) else "Now"
        news_html.append(f"""
            <div class="news-item">
                <span class="news-time">{time_str}</span>
                <span class="news-dot">•</span>
                <span class="news-title">{item['title']}</span>
                <span class="news-publisher">({item['publisher']})</span>
            </div>
        """)

    # Duplicate for seamless loop
    all_news = "".join(news_html * 3)

    st.markdown(
        f"""
        <div class="news-wrapper">
            <div class="news-label">📰 LATEST NEWS:</div>
            <div class="news-scroll">
                {all_news}
            </div>
        </div>

        <style>
        .news-wrapper {{
            background: linear-gradient(135deg, #1e2742 0%, #2a3250 100%);
            border-radius: 8px;
            padding: 10px 16px;
            margin: 1rem 0;
            display: flex;
            align-items: center;
            gap: 16px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }}

        .news-label {{
            color: #FFB300;
            font-weight: 700;
            font-size: 0.85rem;
            white-space: nowrap;
            letter-spacing: 0.05em;
        }}

        .news-scroll {{
            display: flex;
            animation: scrollNews 90s linear infinite;
            width: fit-content;
        }}

        .news-item {{
            display: inline-flex;
            align-items: center;
            gap: 10px;
            white-space: nowrap;
            margin-right: 40px;
        }}

        .news-time {{
            color: #9AA4B2;
            font-size: 0.8rem;
            font-weight: 600;
        }}

        .news-dot {{
            color: #4C78FF;
            font-weight: bold;
        }}

        .news-title {{
            color: #E6EAF2;
            font-size: 0.9rem;
        }}

        .news-publisher {{
            color: #6B7280;
            font-size: 0.75rem;
            font-style: italic;
        }}

        @keyframes scrollNews {{
            0% {{
                transform: translateX(0);
            }}
            100% {{
                transform: translateX(-33.33%);
            }}
        }}

        .news-wrapper:hover .news-scroll {{
            animation-play-state: paused;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_economic_calendar(events: List[Dict]):
    """
    Render upcoming economic events bar.

    Args:
        events: List of economic event dictionaries
    """
    if not events:
        return

    events_html = []
    for event in events:
        events_html.append(f"""
            <div class="econ-event">
                <span class="econ-icon">{event['icon']}</span>
                <div class="econ-details">
                    <div class="econ-time">{event['time']}</div>
                    <div class="econ-name">{event['event']}</div>
                </div>
            </div>
        """)

    st.markdown(
        f"""
        <div class="econ-wrapper">
            <div class="econ-label">📅 TODAY'S EVENTS:</div>
            <div class="econ-container">
                {"".join(events_html)}
            </div>
        </div>

        <style>
        .econ-wrapper {{
            background: linear-gradient(135deg, #2a1e3e 0%, #3a2d4e 100%);
            border-radius: 8px;
            padding: 12px 16px;
            margin: 1rem 0;
            display: flex;
            align-items: center;
            gap: 16px;
        }}

        .econ-label {{
            color: #FFB300;
            font-weight: 700;
            font-size: 0.85rem;
            white-space: nowrap;
            letter-spacing: 0.05em;
        }}

        .econ-container {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}

        .econ-event {{
            display: flex;
            align-items: center;
            gap: 10px;
            background: rgba(255, 255, 255, 0.05);
            padding: 8px 14px;
            border-radius: 6px;
            transition: all 0.3s ease;
        }}

        .econ-event:hover {{
            background: rgba(255, 255, 255, 0.1);
            transform: translateY(-2px);
        }}

        .econ-icon {{
            font-size: 1.2rem;
        }}

        .econ-details {{
            display: flex;
            flex-direction: column;
            gap: 2px;
        }}

        .econ-time {{
            color: #9AA4B2;
            font-size: 0.7rem;
            font-weight: 600;
        }}

        .econ-name {{
            color: #E6EAF2;
            font-size: 0.85rem;
            font-weight: 500;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# MAIN TICKER FUNCTION
# ============================================================================


def render_market_ticker(style: str = "scrolling", include_news: bool = False, include_calendar: bool = False):
    """
    Render complete market ticker with auto-refresh.

    Args:
        style: "scrolling" or "grid" layout
        include_news: Whether to show news feed
        include_calendar: Whether to show economic calendar
    """
    t_col1, t_col2 = st.columns([8, 1])

    with t_col2:
        inner_col1, inner_col2 = st.columns([1, 4])
        with inner_col2:
            auto_refresh = st.toggle("Auto", value=False, key="ticker_auto", help="Refresh every 2m")

    # Fetch data
    with st.spinner("Loading market data..."):
        market_data = fetch_market_data()
        news_items = fetch_market_news() if include_news else []
        events = fetch_economic_calendar() if include_calendar else []

    # Render based on style
    if style == "scrolling":
        render_scrolling_ticker(market_data)
    else:
        render_grid_ticker(market_data)

    # Render news feed
    if include_news and news_items:
        render_news_feed(news_items)

    # Render economic calendar
    if include_calendar and events:
        render_economic_calendar(events)

    # Auto-refresh logic
    if auto_refresh:
        import time

        time.sleep(120)  # Wait 2 minutes
        st.rerun()
