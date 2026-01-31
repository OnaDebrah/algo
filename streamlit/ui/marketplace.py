"""
Strategy Marketplace
"""

import pandas as pd

import streamlit as st
from streamlit.core.marketplace import StrategyListing, StrategyMarketplace, StrategyReview
from streamlit.strategies.strategy_catalog import get_catalog


def init_marketplace():
    """Initialize marketplace in session state"""
    if "marketplace" not in st.session_state:
        st.session_state.marketplace = StrategyMarketplace()
    if "current_user_id" not in st.session_state:
        st.session_state.current_user_id = st.session_state.get("user", {}).get("id", 1)
    if "current_username" not in st.session_state:
        st.session_state.current_username = st.session_state.get("user", {}).get("username", "demo_user")


def render_strategy_card(strategy: StrategyListing, show_actions: bool = True, key: str = ""):
    """Render a beautiful strategy card"""
    with st.container():
        st.markdown(
            """
            <style>
            .strategy-card {
                background: var(--bg-card);
                border-radius: 12px;
                padding: 1.5rem;
                margin-bottom: 1rem;
                border: 1px solid var(--border-subtle);
                transition: all 0.2s ease;
            }
            .strategy-card:hover {
                border-color: var(--primary-color);
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Create columns for layout
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            st.markdown(f"### {strategy.name}")
            st.markdown(f"*by {strategy.creator_name}*")
            st.caption(strategy.description[:150] + "..." if len(strategy.description) > 150 else strategy.description)

        with col2:
            # Rating display
            stars = "⭐" * int(strategy.rating)
            st.markdown(f"**{stars}**")
            st.caption(f"{strategy.rating:.1f} ({strategy.num_ratings} ratings)")

        with col3:
            st.metric("Downloads", strategy.downloads)

        # Tags and metadata
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f'<span style="background-color: #1f77b4; color: white; padding: 2px 8px; border-radius: 3px; font-size: 12px;">{strategy.category}</span>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<span style="background-color: #ff7f0e; color: white; padding: 2px 8px; border-radius: 3px; font-size: 12px;">{strategy.complexity}</span>',
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f'<span style="background-color: #2ca02c; color: white; padding: 2px 8px; border-radius: 3px; font-size: 12px;">{strategy.strategy_type}</span>',
                unsafe_allow_html=True,
            )
        with col4:
            if strategy.price == 0:
                st.markdown(
                    '<span style="background-color: #2ca02c; color: white; padding: 2px 8px; border-radius: 3px; font-size: 12px;">FREE</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<span style="background-color: #d62728; color: white; padding: 2px 8px; border-radius: 3px; font-size: 12px;">${strategy.price}</span>',
                    unsafe_allow_html=True,
                )

        # Tags
        if strategy.tags:
            st.write("🏷️ " + " • ".join([f"`{tag}`" for tag in strategy.tags[:5]]))

        # Performance metrics preview
        if strategy.performance_metrics:
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            metrics = strategy.performance_metrics

            if "sharpe_ratio" in metrics:
                metrics_col1.metric("Sharpe", f"{metrics['sharpe_ratio']:.2f}")
            if "total_return" in metrics:
                metrics_col2.metric("Return", f"{metrics['total_return']:.1f}%")
            if "max_drawdown" in metrics:
                metrics_col3.metric("Max DD", f"{metrics['max_drawdown']:.1f}%")
            if "win_rate" in metrics:
                metrics_col4.metric("Win Rate", f"{metrics['win_rate']:.1f}%")

        # Action buttons
        if show_actions:
            action_col1, action_col2, action_col3, action_col4 = st.columns(4)

            with action_col1:
                if st.button("📖 Details", key=f"details_{strategy.id}_{key}", use_container_width=True):
                    st.session_state.selected_strategy = strategy.id
                    st.rerun()

            with action_col2:
                if st.button("⬇️ Clone", key=f"clone_{strategy.id}_clone_{key}", use_container_width=True):
                    clone_strategy(strategy.id)

            with action_col3:
                is_fav = st.session_state.marketplace.is_favorited(strategy.id, st.session_state.current_user_id)
                icon = "❤️" if is_fav else "🤍"
                if st.button(f"{icon}", key=f"fav_{strategy.id}_{key}", use_container_width=True, help="Favorite"):
                    toggle_favorite(strategy.id)

            with action_col4:
                if st.button("⭐ Review", key=f"review_{strategy.id}_{key}", use_container_width=True):
                    st.session_state.review_strategy = strategy.id
                    st.rerun()

        st.divider()


def render_strategy_details(strategy_id: int):
    """Render detailed strategy view"""
    marketplace = st.session_state.marketplace
    strategy = marketplace.get_strategy(strategy_id)

    if not strategy:
        st.error("Strategy not found")
        return

    # Back button
    if st.button("← Back to Browse", type="secondary"):
        st.session_state.selected_strategy = None
        st.rerun()

    st.title(strategy.name)

    # Creator and metadata
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"**Created by:** {strategy.creator_name}")
        st.caption(f"Published: {strategy.created_at.strftime('%B %d, %Y')}")
        st.caption(f"Last updated: {strategy.updated_at.strftime('%B %d, %Y')}")
        st.caption(f"Version: {strategy.version}")

    with col2:
        stars = "⭐" * int(strategy.rating)
        st.markdown(f"### {stars}")
        st.markdown(f"**{strategy.rating:.1f}** / 5.0")
        st.caption(f"Based on {strategy.num_ratings} ratings")

    with col3:
        st.metric("Total Downloads", strategy.downloads)
        if strategy.price == 0:
            st.success("FREE")
        else:
            st.info(f"${strategy.price}")

    st.divider()

    # Description
    st.markdown("## 📝 Description")
    st.write(strategy.description)

    # Strategy details
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Strategy Information")
        st.write(f"**Type:** {strategy.strategy_type}")
        st.write(f"**Category:** {strategy.category}")
        st.write(f"**Complexity:** {strategy.complexity}")

        if strategy.tags:
            st.write("**Tags:**")
            st.write(" • ".join([f"`{tag}`" for tag in strategy.tags]))

    with col2:
        st.markdown("### ⚙️ Parameters")
        if strategy.parameters:
            params_df = pd.DataFrame([{"Parameter": k, "Value": v} for k, v in strategy.parameters.items()])
            st.dataframe(params_df, use_container_width=True)
        else:
            st.info("No custom parameters")

    # Performance metrics
    if strategy.performance_metrics:
        st.markdown("### 📈 Backtested Performance")

        metrics = strategy.performance_metrics
        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric("Total Return", f"{metrics.get('total_return', 0):.2f}%")
        col2.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
        col3.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2f}%")
        col4.metric("Win Rate", f"{metrics.get('win_rate', 0):.2f}%")
        col5.metric("# Trades", metrics.get("num_trades", 0))

    st.divider()

    # Action buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("⬇️ Clone This Strategy", type="primary", use_container_width=True):
            clone_strategy(strategy_id)

    with col2:
        is_fav = marketplace.is_favorited(strategy_id, st.session_state.current_user_id)
        fav_text = "❤️ Unfavorite" if is_fav else "🤍 Add to Favorites"
        if st.button(fav_text, use_container_width=True):
            toggle_favorite(strategy_id)

    with col3:
        if st.button("⭐ Write a Review", use_container_width=True):
            st.session_state.show_review_form = True

    # Reviews section
    st.divider()
    st.markdown("## 💬 Reviews")

    # Show review form if requested
    if st.session_state.get("show_review_form", False):
        render_review_form(strategy_id)

    # Display reviews
    reviews = marketplace.get_reviews(strategy_id)

    if reviews:
        for review in reviews:
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{review.username}**")
                    stars = "⭐" * review.rating
                    st.markdown(stars)
                with col2:
                    st.caption(review.created_at.strftime("%b %d, %Y"))

                st.write(review.review_text)

                if review.performance_achieved:
                    with st.expander("📊 User's Performance"):
                        perf_df = pd.DataFrame([{"Metric": k, "Value": v} for k, v in review.performance_achieved.items()])
                        st.dataframe(perf_df, use_container_width=True)

                st.divider()
    else:
        st.info("No reviews yet. Be the first to review this strategy!")


def render_review_form(strategy_id: int):
    """Render review submission form"""
    with st.form("review_form"):
        st.markdown("### ✍️ Write Your Review")

        rating = st.slider("Rating", 1, 5, 5)
        review_text = st.text_area("Your Review", height=150)

        st.markdown("**Your Performance (Optional)**")
        col1, col2, col3 = st.columns(3)
        with col1:
            total_return = st.number_input("Total Return (%)", value=0.0)
        with col2:
            sharpe = st.number_input("Sharpe Ratio", value=0.0)
        with col3:
            max_dd = st.number_input("Max Drawdown (%)", value=0.0)

        submitted = st.form_submit_button("Submit Review", type="primary")

        if submitted:
            performance = {}
            if total_return != 0 or sharpe != 0 or max_dd != 0:
                performance = {"total_return": total_return, "sharpe_ratio": sharpe, "max_drawdown": max_dd}

            review = StrategyReview(
                strategy_id=strategy_id,
                user_id=st.session_state.current_user_id,
                username=st.session_state.current_username,
                rating=rating,
                review_text=review_text,
                performance_achieved=performance,
            )

            if st.session_state.marketplace.add_review(review):
                st.success("Review submitted successfully!")
                st.session_state.show_review_form = False
                st.rerun()
            else:
                st.error("Failed to submit review")


def clone_strategy(strategy_id: int):
    """Clone a strategy to user's collection"""
    marketplace = st.session_state.marketplace
    config = marketplace.download_strategy(strategy_id, st.session_state.current_user_id)

    if config:
        st.success("✅ Strategy cloned successfully! Check your strategies page.")
        # Here you would integrate with your strategy management system
        # to actually create the strategy in the user's account
    else:
        st.error("Failed to clone strategy")


def toggle_favorite(strategy_id: int):
    """Toggle favorite status"""
    marketplace = st.session_state.marketplace

    if marketplace.is_favorited(strategy_id, st.session_state.current_user_id):
        marketplace.remove_favorite(strategy_id, st.session_state.current_user_id)
        st.success("Removed from favorites")
    else:
        marketplace.add_favorite(strategy_id, st.session_state.current_user_id)
        st.success("Added to favorites")

    st.rerun()


def render_publish_form():
    """Render form to publish a new strategy"""
    st.markdown("## 📤 Publish Your Strategy")

    catalog = get_catalog()

    with st.form("publish_strategy"):
        name = st.text_input("Strategy Name*", placeholder="My Awesome Momentum Strategy")
        description = st.text_area("Description*", height=150, placeholder="Describe what makes your strategy unique...")

        col1, col2 = st.columns(2)
        with col1:
            strategy_type = st.selectbox("Strategy Type*", options=list(catalog.strategies.keys()), format_func=lambda x: catalog.strategies[x].name)
        with col2:
            complexity = st.selectbox("Complexity*", ["Beginner", "Intermediate", "Advanced", "Institutional"])

        # Get strategy info from catalog
        strategy_info = catalog.strategies[strategy_type]

        st.markdown("### ⚙️ Parameters")
        parameters = {}
        for param_name, param_info in strategy_info.parameters.items():
            if param_info.get("range"):
                if isinstance(param_info["range"], tuple):
                    parameters[param_name] = st.slider(
                        param_name, min_value=param_info["range"][0], max_value=param_info["range"][1], value=param_info["default"]
                    )
                else:
                    parameters[param_name] = st.selectbox(
                        param_name, options=param_info["range"], index=param_info["range"].index(param_info["default"])
                    )
            else:
                parameters[param_name] = st.text_input(param_name, value=str(param_info["default"]))

        st.markdown("### 📊 Performance Metrics (Optional)")
        st.caption("Add backtested performance metrics to make your strategy more attractive")

        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        with perf_col1:
            total_return = st.number_input("Total Return (%)", value=0.0)
        with perf_col2:
            sharpe_ratio = st.number_input("Sharpe Ratio", value=0.0)
        with perf_col3:
            max_drawdown = st.number_input("Max Drawdown (%)", value=0.0)
        with perf_col4:
            win_rate = st.number_input("Win Rate (%)", value=0.0, max_value=100.0)

        st.markdown("### 🏷️ Tags")
        tags_input = st.text_input("Tags (comma-separated)", placeholder="momentum, trend-following, low-risk")

        col1, col2 = st.columns(2)
        with col1:
            price = st.number_input("Price (0 for free)", min_value=0.0, value=0.0)
        with col2:
            is_public = st.checkbox("Make Public", value=True)

        submitted = st.form_submit_button("🚀 Publish Strategy", type="primary")

        if submitted:
            if not name or not description:
                st.error("Please fill in all required fields (marked with *)")
                return

            # Parse tags
            tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()]

            # Build performance metrics
            performance_metrics = {}
            if total_return != 0 or sharpe_ratio != 0:
                performance_metrics = {"total_return": total_return, "sharpe_ratio": sharpe_ratio, "max_drawdown": max_drawdown, "win_rate": win_rate}

            # Create listing
            listing = StrategyListing(
                name=name,
                description=description,
                creator_id=st.session_state.current_user_id,
                creator_name=st.session_state.current_username,
                strategy_type=strategy_type,
                category=strategy_info.category.value,
                complexity=complexity,
                parameters=parameters,
                performance_metrics=performance_metrics,
                price=price,
                is_public=is_public,
                tags=tags,
            )

            # Publish to marketplace
            strategy_id = st.session_state.marketplace.publish_strategy(listing)

            if strategy_id:
                st.success(f"✅ Strategy published successfully! (ID: {strategy_id})")
                st.balloons()
            else:
                st.error("Failed to publish strategy")


def render_navigation_tabs():
    """Render main navigation tabs"""
    tabs = st.tabs(["🔍 Browse", "🔥 Trending", "📝 My Strategies", "❤️ Favorites", "⬇️ Downloads", "📤 Publish"])

    return tabs


def render_marketplace_header():
    """Render marketplace header with stats"""
    marketplace = st.session_state.marketplace
    stats = marketplace.get_marketplace_stats()

    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
            color: white;
        ">
            <div style="font-size: 2.5rem; font-weight: 800; margin-bottom: 0.5rem;">
                🏪 Strategy Marketplace
            </div>
            <div style="font-size: 1.125rem; opacity: 0.9;">
                Discover, share, and clone trading strategies from the community
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Stats row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 1rem; background: var(--bg-card); border-radius: 12px;">
                <div style="font-size: 2rem; font-weight: 700; color: var(--primary-color);">
                    {stats['total_strategies']}
                </div>
                <div style="font-size: 0.875rem; color: var(--text-muted);">
                    Total Strategies
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 1rem; background: var(--bg-card); border-radius: 12px;">
                <div style="font-size: 2rem; font-weight: 700; color: var(--primary-color);">
                    {stats['total_downloads']}
                </div>
                <div style="font-size: 0.875rem; color: var(--text-muted);">
                    Total Downloads
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 1rem; background: var(--bg-card); border-radius: 12px;">
                <div style="font-size: 2rem; font-weight: 700; color: var(--primary-color);">
                    {stats['average_rating']:.1f}⭐
                </div>
                <div style="font-size: 0.875rem; color: var(--text-muted);">
                    Average Rating
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 1rem; background: var(--bg-card); border-radius: 12px;">
                <div style="font-size: 2rem; font-weight: 700; color: var(--primary-color);">
                    {stats['total_creators']}
                </div>
                <div style="font-size: 0.875rem; color: var(--text-muted);">
                    Active Creators
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)


def marketplace():
    """Main marketplace page"""
    st.set_page_config(page_title="Strategy Marketplace", page_icon="🏪", layout="wide")

    init_marketplace()
    marketplace = st.session_state.marketplace

    # Check if viewing strategy details
    if st.session_state.get("selected_strategy"):
        render_strategy_details(st.session_state.selected_strategy)
        return

    # Render header with stats
    render_marketplace_header()

    # Navigation tabs
    tabs = render_navigation_tabs()

    # Browse Tab
    with tabs[0]:
        st.markdown("### 🔍 Browse Strategies")

        # Filters
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            search_query = st.text_input("🔍 Search", placeholder="Search strategies...", key="browse_search")

        with col2:
            categories = ["All"] + list(set([s.category for s in marketplace.browse_strategies()]))
            category = st.selectbox("Category", categories, key="browse_category")
            category = None if category == "All" else category

        with col3:
            complexities = ["All", "Beginner", "Intermediate", "Advanced", "Institutional"]
            complexity = st.selectbox("Complexity", complexities, key="browse_complexity")
            complexity = None if complexity == "All" else complexity

        with col4:
            sort_options = {"Rating": "rating", "Downloads": "downloads", "Newest": "created_at", "Recently Updated": "updated_at"}
            sort_by = st.selectbox("Sort by", list(sort_options.keys()), key="browse_sort")

        st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)

        # Browse strategies
        strategies = marketplace.browse_strategies(
            category=category, complexity=complexity, search_query=search_query or None, sort_by=sort_options[sort_by], limit=50
        )

        if strategies:
            st.caption(f"Showing {len(strategies)} strategies")
            for strategy in strategies:
                render_strategy_card(strategy, key="brows")
        else:
            st.info("No strategies found. Try adjusting your filters.")

    # Trending Tab
    with tabs[1]:
        st.markdown("### 🔥 Trending Strategies")
        st.caption("Most popular strategies in the last 7 days")

        trending = marketplace.get_trending_strategies(days=7, limit=20)

        if trending:
            for strategy in trending:
                render_strategy_card(strategy, key="trending")
        else:
            st.info("No trending strategies at the moment.")

    # My Strategies Tab
    with tabs[2]:
        st.markdown("### 📝 My Published Strategies")

        user_strategies = marketplace.get_user_strategies(st.session_state.current_user_id)

        if user_strategies:
            # Show creator stats
            creator_stats = marketplace.get_creator_stats(st.session_state.current_user_id)

            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            stat_col1.metric("Published", creator_stats["strategies_published"])
            stat_col2.metric("Total Downloads", creator_stats["total_downloads"])
            stat_col3.metric("Average Rating", f"{creator_stats['average_rating']:.1f}⭐")
            stat_col4.metric("Total Ratings", creator_stats["total_ratings"])

            st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)

            for strategy in user_strategies:
                render_strategy_card(strategy, key="published")
        else:
            st.info("You haven't published any strategies yet.")

    # Favorites Tab
    with tabs[3]:
        st.markdown("### ❤️ My Favorite Strategies")

        favorites = marketplace.get_favorites(st.session_state.current_user_id)

        if favorites:
            for strategy in favorites:
                render_strategy_card(strategy, key="favorite")
        else:
            st.info("You haven't favorited any strategies yet.")

    # Downloads Tab
    with tabs[4]:
        st.markdown("### ⬇️ My Downloaded Strategies")

        downloads = marketplace.get_user_downloads(st.session_state.current_user_id)

        if downloads:
            for strategy in downloads:
                render_strategy_card(strategy, key="downloaded")
        else:
            st.info("You haven't downloaded any strategies yet.")

    # Publish Tab
    with tabs[5]:
        render_publish_form()
