"""
Asset Class Selection UI Component
"""

import pandas as pd

import streamlit as st
from streamlit.core.asset_classes import AssetClass, get_asset_class_names, get_asset_manager


def render_asset_selector(key_prefix: str = "asset") -> tuple:
    """
    Render asset class selector with popular symbols

    Args:
        key_prefix: Prefix for widget keys

    Returns:
        Tuple of (symbols_list, asset_class)
    """
    asset_mgr = get_asset_manager()

    st.markdown("### 🎯 Select Assets")

    # Asset class selection
    col1, col2 = st.columns([1, 2])

    with col1:
        asset_class_name = st.selectbox(
            "Asset Class",
            get_asset_class_names(),
            help="Choose the type of asset to trade",
            key=f"{key_prefix}_class",
        )

        # Get asset class enum
        asset_class = next(ac for ac in AssetClass if ac.value == asset_class_name)

    with col2:
        # Show asset class info
        info = asset_mgr.get_asset_class_info(asset_class)

        if info:
            st.info(
                f"""
            **{info.get('description', '')}**
            - Risk: {info.get('risk_level', 'N/A')}
            - Liquidity: {info.get('liquidity', 'N/A')}
            - Hours: {info.get('trading_hours', 'N/A')}
            """
            )

    # Symbol input method
    input_method = st.radio(
        "Input Method",
        ["Popular Symbols", "Custom Symbol"],
        horizontal=True,
        key=f"{key_prefix}_method",
    )

    symbols = []

    if input_method == "Popular Symbols":
        # Show popular symbols for selected asset class
        popular = asset_mgr.get_popular_symbols(asset_class)

        st.markdown("**Select from popular symbols:**")

        # Create columns for better layout
        cols = st.columns(4)
        selected_symbols = []

        for i, symbol in enumerate(popular):
            with cols[i % 4]:
                if st.checkbox(symbol, key=f"{key_prefix}_pop_{symbol}"):
                    selected_symbols.append(symbol)

        if selected_symbols:
            symbols = selected_symbols
            st.success(f"Selected: {', '.join(symbols)}")
        else:
            st.warning("⚠️ Select at least one symbol")

    else:
        # Custom symbol input
        symbol_input = st.text_input(
            "Enter Symbol(s)",
            help="Enter one or more symbols (comma-separated)",
            placeholder=f"e.g., {', '.join(asset_mgr.get_popular_symbols(asset_class)[:3])}",
            key=f"{key_prefix}_custom",
        )

        if symbol_input:
            symbols = [s.strip().upper() for s in symbol_input.split(",") if s.strip()]

            # Validate symbols
            if symbols:
                st.markdown("**Validating symbols...**")

                valid_symbols = []
                for symbol in symbols:
                    is_valid, message = asset_mgr.validate_symbol(symbol)

                    if is_valid:
                        st.success(f"✅ {symbol}: {message}")
                        valid_symbols.append(symbol)
                    else:
                        st.error(f"❌ {symbol}: {message}")

                symbols = valid_symbols

    # Show selected assets info
    if symbols:
        with st.expander("📊 Selected Assets Details"):
            asset_data = []

            for symbol in symbols:
                asset_info = asset_mgr.get_asset_info(symbol)
                asset_data.append(
                    {
                        "Symbol": symbol,
                        "Name": asset_info.name,
                        "Type": asset_info.asset_class.value,
                        "Exchange": asset_info.exchange or "N/A",
                        "Currency": asset_info.currency,
                        "Trading Hours": asset_info.trading_hours,
                    }
                )

            df = pd.DataFrame(asset_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

    return symbols, asset_class


def render_asset_class_overview():
    """Render overview of all asset classes"""

    st.markdown("### 📚 Asset Class Overview")

    asset_mgr = get_asset_manager()

    # Create tabs for each asset class
    asset_classes = list(AssetClass)
    tabs = st.tabs([ac.value for ac in asset_classes])

    for i, asset_class in enumerate(asset_classes):
        with tabs[i]:
            info = asset_mgr.get_asset_class_info(asset_class)

            if info:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Description:**")
                    st.write(info.get("description", "N/A"))

                    st.markdown("**Risk Level:**")
                    st.write(info.get("risk_level", "N/A"))

                    st.markdown("**Liquidity:**")
                    st.write(info.get("liquidity", "N/A"))

                    st.markdown("**Trading Hours:**")
                    st.write(info.get("trading_hours", "N/A"))

                with col2:
                    st.markdown("**Leverage:**")
                    st.write(info.get("leverage", "N/A"))

                    st.markdown("**Typical Spread:**")
                    st.write(info.get("typical_spread", "N/A"))

                    st.markdown("**Best For:**")
                    st.write(info.get("best_for", "N/A"))

                    st.markdown("**Considerations:**")
                    st.write(info.get("considerations", "N/A"))

                # Show popular symbols
                st.markdown("**Popular Symbols:**")
                popular = asset_mgr.get_popular_symbols(asset_class)
                st.write(", ".join(popular[:12]))


def render_asset_comparison(symbols: list):
    """
    Render comparison of multiple assets

    Args:
        symbols: List of symbols to compare
    """
    if len(symbols) < 2:
        st.warning("Select at least 2 assets to compare")
        return

    st.markdown("### 📊 Asset Comparison")

    asset_mgr = get_asset_manager()

    comparison_data = []

    for symbol in symbols:
        asset_info = asset_mgr.get_asset_info(symbol)
        comparison_data.append(
            {
                "Symbol": symbol,
                "Name": asset_info.name[:30],  # Truncate long names
                "Type": asset_info.asset_class.value,
                "Currency": asset_info.currency,
                "Min Tick": asset_info.min_tick,
                "Lot Size": asset_info.lot_size,
                "Leverage": "✅" if asset_info.leverage_available else "❌",
            }
        )

    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Asset class distribution
    if len(symbols) > 1:
        st.markdown("**Asset Class Distribution:**")

        class_counts = df["Type"].value_counts()

        col1, col2 = st.columns([2, 1])

        with col1:
            st.bar_chart(class_counts)

        with col2:
            for asset_type, count in class_counts.items():
                pct = (count / len(symbols)) * 100
                st.metric(asset_type, f"{count} ({pct:.0f}%)")


def render_symbol_finder():
    """Render symbol finder/search tool"""

    st.markdown("### 🔍 Symbol Finder")

    asset_mgr = get_asset_manager()

    col1, col2 = st.columns(2)

    with col1:
        search_query = st.text_input("Search Symbol", placeholder="Enter symbol to validate", key="symbol_finder")

    with col2:
        _ = st.selectbox(
            "Filter by Asset Class",
            ["All"] + get_asset_class_names(),
            key="finder_class",
        )

    if search_query:
        symbol = search_query.upper()

        # Validate
        is_valid, message = asset_mgr.validate_symbol(symbol)

        if is_valid:
            st.success(f"✅ {symbol} is valid!")

            # Show details
            asset_info = asset_mgr.get_asset_info(symbol)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Name", asset_info.name[:20])
                st.metric("Type", asset_info.asset_class.value)

            with col2:
                st.metric("Exchange", asset_info.exchange or "N/A")
                st.metric("Currency", asset_info.currency)

            with col3:
                st.metric("Min Tick", asset_info.min_tick)
                st.metric("Leverage", "Yes" if asset_info.leverage_available else "No")

            # Trading hours
            st.info(f"**Trading Hours:** {asset_info.trading_hours}")

            # Show asset class info
            class_info = asset_mgr.get_asset_class_info(asset_info.asset_class)
            if class_info:
                with st.expander("📋 Asset Class Information"):
                    st.markdown(f"**Description:** {class_info.get('description')}")
                    st.markdown(f"**Risk Level:** {class_info.get('risk_level')}")
                    st.markdown(f"**Best For:** {class_info.get('best_for')}")
                    st.markdown(f"**Considerations:** {class_info.get('considerations')}")

        else:
            st.error(f"❌ {message}")

            # Suggest similar symbols if available
            detected_class = asset_mgr.detect_asset_class(symbol)
            popular = asset_mgr.get_popular_symbols(detected_class)

            if popular:
                st.info(f"💡 **Suggested {detected_class.value} symbols:** {', '.join(popular[:5])}")


def render_portfolio_builder(key_prefix: str = "portfolio") -> list:
    """
    Render portfolio builder for multi-asset selection

    Args:
        key_prefix: Prefix for widget keys

    Returns:
        List of selected symbols
    """
    st.markdown("### 🎯 Build Your Portfolio")

    asset_mgr = get_asset_manager()

    # Initialize session state
    if f"{key_prefix}_portfolio" not in st.session_state:
        st.session_state[f"{key_prefix}_portfolio"] = []

    # Portfolio templates
    col1, col2 = st.columns([2, 1])

    with col1:
        template = st.selectbox(
            "Quick Start Template",
            [
                "Empty Portfolio",
                "Conservative (70/30 Stocks/Bonds)",
                "Balanced (60/40)",
                "Aggressive (80/20)",
                "Tech Focus",
                "Crypto Portfolio",
                "Global Diversified",
            ],
            key=f"{key_prefix}_template",
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📥 Load Template", key=f"{key_prefix}_load"):
            portfolio = _get_portfolio_template(template)
            st.session_state[f"{key_prefix}_portfolio"] = portfolio
            st.success(f"✅ Loaded {template}")
            st.rerun()

    # Add asset section
    st.markdown("#### ➕ Add Assets")

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        add_class = st.selectbox("Asset Class", get_asset_class_names(), key=f"{key_prefix}_add_class")

        asset_class_enum = next(ac for ac in AssetClass if ac.value == add_class)

    with col2:
        popular = asset_mgr.get_popular_symbols(asset_class_enum)
        add_symbol = st.selectbox("Symbol", popular, key=f"{key_prefix}_add_symbol")

    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("➕ Add", key=f"{key_prefix}_add_btn"):
            if add_symbol not in st.session_state[f"{key_prefix}_portfolio"]:
                st.session_state[f"{key_prefix}_portfolio"].append(add_symbol)
                st.success(f"✅ Added {add_symbol}")
                st.rerun()
            else:
                st.warning(f"⚠️ {add_symbol} already in portfolio")

    # Display current portfolio
    portfolio = st.session_state[f"{key_prefix}_portfolio"]

    if portfolio:
        st.markdown(f"#### 📋 Current Portfolio ({len(portfolio)} assets)")

        # Create DataFrame
        portfolio_data = []
        for symbol in portfolio:
            asset_info = asset_mgr.get_asset_info(symbol)
            portfolio_data.append(
                {
                    "Symbol": symbol,
                    "Name": asset_info.name[:25],
                    "Type": asset_info.asset_class.value,
                    "Currency": asset_info.currency,
                }
            )

        df = pd.DataFrame(portfolio_data)

        # Display with remove buttons
        for idx, row in df.iterrows():
            col1, col2, col3, col4, col5 = st.columns([2, 3, 2, 2, 1])

            with col1:
                st.write(row["Symbol"])
            with col2:
                st.write(row["Name"])
            with col3:
                st.write(row["Type"])
            with col4:
                st.write(row["Currency"])
            with col5:
                if st.button("🗑️", key=f"{key_prefix}_remove_{idx}"):
                    st.session_state[f"{key_prefix}_portfolio"].remove(row["Symbol"])
                    st.rerun()

        # Portfolio statistics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Assets", len(portfolio))

        with col2:
            asset_types = [asset_mgr.detect_asset_class(s).value for s in portfolio]
            unique_types = len(set(asset_types))
            st.metric("Asset Classes", unique_types)

        with col3:
            if st.button("🗑️ Clear All", key=f"{key_prefix}_clear"):
                st.session_state[f"{key_prefix}_portfolio"] = []
                st.rerun()

        # Asset class breakdown
        with st.expander("📊 Portfolio Breakdown"):
            type_counts = pd.Series(asset_types).value_counts()

            for asset_type, count in type_counts.items():
                pct = (count / len(portfolio)) * 100
                st.markdown(f"**{asset_type}:** {count} ({pct:.1f}%)")

            st.bar_chart(type_counts)

    else:
        st.info("No assets in portfolio. Add assets above or load a template.")

    return portfolio


def _get_portfolio_template(template_name: str) -> list:
    """
    Get predefined portfolio template

    Args:
        template_name: Name of the template

    Returns:
        List of symbols
    """
    templates = {
        "Empty Portfolio": [],
        "Conservative (70/30 Stocks/Bonds)": [
            "SPY",
            "VTI",
            "VOO",  # Stocks
            "TLT",
            "AGG",
            "BND",  # Bonds
        ],
        "Balanced (60/40)": ["SPY", "QQQ", "IWM", "TLT", "IEF"],  # Stocks  # Bonds
        "Aggressive (80/20)": [
            "NVDA",
            "TSLA",
            "AAPL",
            "MSFT",  # Growth stocks
            "QQQ",
            "SPY",  # Tech-heavy ETFs
            "TLT",  # Some bonds
        ],
        "Tech Focus": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"],
        "Crypto Portfolio": ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD"],
        "Global Diversified": [
            "SPY",  # US Large Cap
            "IWM",  # US Small Cap
            "GLD",  # Gold
            "TLT",  # Bonds
            "EURUSD=X",  # Forex
            "BTC-USD",  # Crypto
        ],
    }

    return templates.get(template_name, [])


def render_asset_grid(symbols: list, key_prefix: str = "grid"):
    """
    Render asset grid with quick info

    Args:
        symbols: List of symbols to display
        key_prefix: Prefix for widget keys
    """
    if not symbols:
        return

    asset_mgr = get_asset_manager()

    # Display in grid format
    cols_per_row = 3

    for i in range(0, len(symbols), cols_per_row):
        cols = st.columns(cols_per_row)

        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(symbols):
                symbol = symbols[idx]
                asset_info = asset_mgr.get_asset_info(symbol)

                with col:
                    with st.container():
                        st.markdown(f"**{symbol}**")
                        st.caption(f"{asset_info.name[:20]}")
                        st.caption(f"Type: {asset_info.asset_class.value}")
                        st.caption(f"Currency: {asset_info.currency}")


def render_quick_add(key_prefix: str = "quick") -> str:
    """
    Render quick asset add widget

    Args:
        key_prefix: Prefix for widget keys

    Returns:
        Selected symbol or empty string
    """
    col1, col2 = st.columns([3, 1])

    with col1:
        symbol = st.text_input(
            "Quick Add Symbol",
            placeholder="Enter symbol (e.g., AAPL, BTC-USD)",
            key=f"{key_prefix}_quick_add",
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        validate = st.button("✓ Validate", key=f"{key_prefix}_validate")

    if symbol and validate:
        symbol = symbol.upper().strip()
        asset_mgr = get_asset_manager()

        is_valid, message = asset_mgr.validate_symbol(symbol)

        if is_valid:
            asset_info = asset_mgr.get_asset_info(symbol)
            st.success(f"✅ {symbol} - {asset_info.name} ({asset_info.asset_class.value})")
            return symbol
        else:
            st.error(f"❌ {message}")
            return ""

    return ""
