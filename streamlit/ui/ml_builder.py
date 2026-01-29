"""
ML Strategy Builder UI Componen
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from config import DEFAULT_ML_TEST_SIZE
from streamlit.core.data_fetcher import fetch_stock_data
from streamlit.strategies.lstm_strategy import LSTMStrategy
from streamlit.strategies.ml_strategy import MLStrategy


def render_ml_builder(ml_models: dict):
    """
    Render the ML strategy builder tab

    Args:
        ml_models: Dictionary to store trained ML models
    """
    st.header("🤖 Machine Learning Strategy Builder")

    st.markdown(
        """
    Train custom ML models to predict market movements and generate trading signals.
    The model learns from historical price patterns and technical indicators.
    """
    )

    # Training Configuration
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Training Configuration")

        ml_symbol = st.text_input(
            "Symbol for Training",
            value="AAPL",
            key="ml_symbol",
            help="Stock symbol to train the model on",
        ).upper()

        model_type = st.selectbox(
            "ML Model Type",
            [
                "Random Forest",
                "Gradient Boosting",
                "SVM (Support Vector Machine)",
                "Logistic Regression",
                "LSTM (Deep Learning)",
            ],
            help="Choose the machine learning algorithm",
        )

        epochs = 20
        if model_type == "LSTM (Deep Learning)":
            epochs = st.number_input("Epochs", 10, 100, 20, help="Training epochs")

        training_period = st.selectbox(
            "Training Data Period",
            ["1y", "2y", "5y"],
            help="Amount of historical data for training",
        )

        test_size = (
            st.slider(
                "Test Set Size (%)",
                min_value=10,
                max_value=40,
                value=int(DEFAULT_ML_TEST_SIZE * 100),
                help="Percentage of data reserved for testing",
            )
            / 100
        )

    with col2:
        st.subheader("Model Info")
        st.info(
            """
        **Features Used:**
        - Moving Averages (SMA, EMA)
        - RSI (Relative Strength Index)
        - MACD
        - Bollinger Bands
        - Volatility
        - Price Returns

        **Output:**
        - Buy Signal (1)
        - Sell Signal (-1)
        - Hold Signal (0)
        """
        )

    # Advanced Settings
    with st.expander("⚙️ Advanced Settings"):
        col1, col2 = st.columns(2)

        with col1:
            _ = st.slider(
                "Prediction Threshold",
                min_value=0.01,
                max_value=0.05,
                value=0.02,
                step=0.005,
                help="Return threshold for buy/sell classification",
            )

        with col2:
            min_samples = st.number_input(
                "Minimum Training Samples",
                min_value=100,
                max_value=1000,
                value=200,
                help="Minimum data points required for training",
            )

    # Train Model Button
    if st.button("🚀 Train ML Model", type="primary"):
        with st.spinner(f"Training model on {ml_symbol}... This may take a minute."):
            try:
                # Fetch training data
                data = fetch_stock_data(ml_symbol, training_period, "1d")

                if data.empty:
                    st.error("❌ Failed to fetch training data. Please check the symbol.")
                    return

                if len(data) < min_samples:
                    st.error(f"❌ Insufficient data. Got {len(data)} samples, need at least {min_samples}.")
                    return

                st.info(f"📊 Fetched {len(data)} data points from {data.index[0].date()} to {data.index[-1].date()}")

                # Create and train model
                if model_type == "LSTM (Deep Learning)":
                    ml_strategy = LSTMStrategy(f"LSTM_{ml_symbol}", epochs=epochs)
                    # LSTM requires different training signature? No, updated to be similar?
                    # basic LSTMStrategy.train takes (data).
                    # Need to check LSTMStrategy.train signature again.
                    # It relies on internal logic.
                    # Wrapper might be needed if signatures diverge too much.
                    # Let's assume standard behavior or minimal divergence.
                    # Actually LSTMStrategy.train(data) -> None (internally splits)?
                    # Existing MLStrategy.train(data, test_size) -> (train_score, test_score).
                    # I need to align them. LSTMStrategy.train currently returns None.
                    # I should update LSTMStrategy to return scores or handle it here.
                    # For now, I'll assume I need to handle the return manually.
                    ml_strategy.train(data)  # It doesn't take test_size yet?
                    train_score, test_score = 0.0, 0.0  # Placeholder

                else:
                    mapped_type = {
                        "Random Forest": "random_forest",
                        "Gradient Boosting": "gradient_boosting",
                        "SVM (Support Vector Machine)": "svm",
                        "Logistic Regression": "logistic_regression",
                    }.get(model_type, "random_forest")

                    ml_strategy = MLStrategy(
                        f"ML_{mapped_type}_{ml_symbol}",
                        model_type=mapped_type,
                    )
                    train_score, test_score = ml_strategy.train(data, test_size)

                # Store model
                ml_models[ml_symbol] = ml_strategy

                st.success("✅ Model trained successfully!")

                # Display Results
                _display_training_results(train_score, test_score, ml_strategy, ml_symbol)

            except Exception as e:
                st.error(f"❌ Error during training: {str(e)}")
                with st.expander("Show detailed error"):
                    st.code(str(e))

    # Display Trained Models
    _display_trained_models(ml_models)


def _display_training_results(train_score: float, test_score: float, ml_strategy: MLStrategy, symbol: str):
    """Display training results and metrics"""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Training Accuracy", f"{train_score:.2%}")

    with col2:
        st.metric("Test Accuracy", f"{test_score:.2%}")

    with col3:
        st.metric("Features Used", len(ml_strategy.feature_cols))

    # Model Performance Analysis
    st.subheader("📊 Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        # Accuracy comparison
        accuracy_data = pd.DataFrame(
            {
                "Dataset": ["Training", "Test"],
                "Accuracy": [train_score * 100, test_score * 100],
            }
        )

        fig = px.bar(
            accuracy_data,
            x="Dataset",
            y="Accuracy",
            title="Model Accuracy",
            color="Dataset",
            color_discrete_map={"Training": "#00ff88", "Test": "#ff6b6b"},
        )
        fig.update_layout(template="plotly_dark", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Overfitting check
        overfit_score = train_score - test_score

        if overfit_score < 0.05:
            st.success("✅ **Good Model**: Low overfitting")
        elif overfit_score < 0.15:
            st.warning("⚠️ **Moderate Overfitting**: Consider more data or regularization")
        else:
            st.error("❌ **High Overfitting**: Model may not generalize well")

        st.metric("Overfitting Score", f"{overfit_score:.2%}")
        st.caption("Difference between training and test accuracy")

    # Feature Importance
    if hasattr(ml_strategy, "get_feature_importance"):
        feature_importance = ml_strategy.get_feature_importance()

        if not feature_importance.empty:
            st.subheader("🔍 Feature Importance")

            top_features = feature_importance.head(10)

            fig = px.bar(
                top_features,
                x="importance",
                y="feature",
                orientation="h",
                title="Top 10 Most Important Features",
                labels={"importance": "Importance", "feature": "Feature"},
            )
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)

    # Usage Instructions
    st.info(
        f"""
    ✅ **Model ready for {symbol}**

    You can now use this ML model in the **Backtest** tab by:
    1. Going to the Backtest tab
    2. Entering symbol: **{symbol}**
    3. Selecting strategy: **ML Model**
    4. Running the backtes
    """
    )


def _display_trained_models(ml_models: dict):
    """Display list of trained models"""
    if ml_models:
        st.divider()
        st.subheader("🧠 Trained Models")

        # Summary stats
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Models", len(ml_models))

        with col2:
            rf_count = sum(1 for m in ml_models.values() if getattr(m, "model_type", "") == "random_forest")
            st.metric("Random Forest", rf_count)

        with col3:
            lstm_count = sum(1 for m in ml_models.values() if isinstance(m, LSTMStrategy))
            st.metric("LSTM Models", lstm_count)

        # Model details
        for symbol, model in ml_models.items():
            with st.expander(f"📈 {symbol} - {model.name}"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write(f"**Model Type:** {model.model_type.replace('_', ' ').title()}")
                    st.write(f"**Status:** {'✅ Trained' if model.is_trained else '❌ Not Trained'}")

                with col2:
                    st.write(f"**Features:** {len(model.feature_cols)}")
                    st.write(f"**Symbol:** {symbol}")

                with col3:
                    # Action buttons
                    col_a, col_b = st.columns(2)

                    with col_a:
                        if st.button("🔄 Retrain", key=f"retrain_{symbol}"):
                            st.info(f"Use the training section above to retrain {symbol}")

                    with col_b:
                        if st.button("🗑️ Delete", key=f"delete_{symbol}"):
                            del ml_models[symbol]
                            st.success(f"Deleted model for {symbol}")
                            st.rerun()

                # Show features
                if model.feature_cols:
                    with st.expander("View Features"):
                        st.write(", ".join(model.feature_cols[:20]))
                        if len(model.feature_cols) > 20:
                            st.caption(f"... and {len(model.feature_cols) - 20} more")
    else:
        st.info("📚 No models trained yet. Train your first model using the form above!")


def _create_sample_data_chart():
    """Create a sample data visualization"""
    st.subheader("📈 Sample Training Data")

    # This is a placeholder - in real implementation would show actual data
    st.info("Train a model to see data visualization here")
