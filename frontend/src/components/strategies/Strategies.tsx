import { Strategy } from "@/types/backtest";

export const strategies: Strategy[] = [
    // TECHNICAL INDICATORS - TREND FOLLOWING
    {
        id: 'sma_crossover',
        name: 'SMA Crossover',
        category: 'Trend Following',
        description: 'Trades based on moving average crossovers. Buy when short MA crosses above long MA, sell when it crosses below.',
        complexity: 'Beginner',
        time_horizon: 'Short to Medium-term',
        best_for: ["Trending markets", "Beginner traders", "Clear trends"],
        params: { short_window: 20, long_window: 50 }
    },
    {
        id: 'macd',
        name: 'MACD Strategy',
        category: 'Trend Following',
        description: 'Moving Average Convergence Divergence. Trades on crossovers of MACD line and signal line.',
        complexity: 'Intermediate',
        time_horizon: 'Medium-term',
        best_for: ["Trend identification", "Momentum confirmation", "Swing trading"],
        params: { fast: 12, slow: 26, signal: 9 }
    },
    {
        id: 'kama',
        name: 'KAMA Strategy',
        category: 'Trend Following',
        description: "Kaufman's Adaptive Moving Average. Adapts speed based on market efficiency - fast during trends, slow during consolidation.",
        complexity: 'Intermediate',
        time_horizon: 'Medium-term',
        best_for: ["Trending markets", "Swing trading", "Reduced whipsaws", "Adaptive trend following"],
        params: { period: 10, fast_ema: 2, slow_ema: 30, signal_threshold: 0.0 }
    },
    {
        id: 'multi_kama',
        name: 'MULTI KAMA Strategy',
        category: 'Trend Following',
        description: "Kaufman's Adaptive Moving Average. Adapts speed based on market efficiency across multiple timeframes.",
        complexity: 'Intermediate',
        time_horizon: 'Medium-term',
        best_for: ["Trending markets", "Swing trading", "Reduced whipsaws", "Adaptive trend following"],
        params: { short_period: 10, long_period: 15, fast_ema: 2, slow_ema: 30 }
    },
    {
        id: 'donchian',
        name: 'Donchian Channel Breakout',
        category: 'Trend Following',
        description: 'The classic Turtle Trader strategy. Goes long on N-day high breakouts, exits on M-day low breakouts.',
        complexity: 'Beginner',
        time_horizon: 'Medium to Long-term',
        best_for: ["Trending markets", "Futures trading", "Long-term trend following", "Systematic trading"],
        params: { entry_period: 20, exit_period: 10, use_both_sides: true }
    },
    {
        id: 'donchian_atr',
        name: 'Donchian ATR Strategy',
        category: 'Trend Following',
        description: 'Enhanced Donchian strategy with ATR-based risk management. Uses volatility for position sizing.',
        complexity: 'Intermediate',
        time_horizon: 'Medium to Long-term',
        best_for: ["Risk-managed trend following", "Volatile markets", "Professional trading"],
        params: { entry_period: 20, exit_period: 10, atr_period: 14, atr_multiplier: 2.0 }
    },
    {
        id: 'filtered_donchian',
        name: 'Filtered Donchian Strategy',
        category: 'Trend Following',
        description: 'Donchian breakouts with trend filter. Only takes trades in direction of longer-term trend.',
        complexity: 'Intermediate',
        time_horizon: 'Medium to Long-term',
        best_for: ["Reducing false breakouts", "Trend-aligned trading", "Lower drawdown tolerance"],
        params: { entry_period: 20, exit_period: 10, trend_period: 50 }
    },
    {
        id: 'parabolic_sar',
        name: 'Parabolic SAR',
        category: 'Trend Following',
        description: 'Stop And Reverse strategy. Follows price trends and reverses when price crosses the indicator.',
        complexity: 'Intermediate',
        time_horizon: 'Medium-term',
        best_for: ["Trending markets", "Trailing stops", "Crypto/Volatile assets"],
        params: { start: 0.02, increment: 0.02, maximum: 0.2 }
    },

    // MOMENTUM
    {
        id: 'rsi',
        name: 'RSI Strategy',
        category: 'Momentum',
        description: 'Uses Relative Strength Index to identify overbought and oversold conditions.',
        complexity: 'Beginner',
        time_horizon: 'Short-term',
        best_for: ["Range-bound markets", "Momentum trading", "Quick trades"],
        params: { period: 14, oversold: 30, overbought: 70 }
    },
    {
        id: 'ts_momentum',
        name: 'Time Series Momentum',
        category: 'Momentum',
        description: 'Exploits persistence in asset returns over time. Goes long when recent returns are positive.',
        complexity: 'Intermediate',
        time_horizon: 'Short to Medium-term',
        best_for: ["Trending assets", "Futures trading", "Systematic strategies"],
        params: { lookback: 12, holding_period: 1 }
    },
    {
        id: 'cs_momentum',
        name: 'Cross-Sectional Momentum',
        category: 'Momentum',
        description: 'Ranks assets by performance and goes long winners, short losers. Also known as relative strength.',
        complexity: 'Intermediate',
        time_horizon: 'Medium-term',
        best_for: ["Stock portfolios", "ETF rotation", "Long-short strategies"],
        params: { lookback: 6, top_pct: 0.2, bottom_pct: 0.2 }
    },

    // MEAN REVERSION
    {
        id: 'bb_mean_reversion',
        name: 'Bollinger Band Mean Reversion',
        category: 'Mean Reversion',
        description: 'Trades reversals from Bollinger Bands. Buy below lower band, sell above upper band.',
        complexity: 'Intermediate',
        time_horizon: 'Short-term',
        best_for: ["Range-bound markets", "Mean reversion", "Swing trading"],
        params: { period: 20, std_dev: 2.0 }
    },
    {
        id: 'pairs_trading',
        name: 'Pairs Trading',
        category: 'Pairs Trading',
        description: 'Identifies cointegrated asset pairs and trades their spread reversion to mean.',
        complexity: 'Advanced',
        time_horizon: 'Short to Medium-term',
        best_for: ["Market-neutral strategies", "Statistical arbitrage", "Hedge funds"],
        params: { lookback: 60, entry_threshold: 2.0, exit_threshold: 0.5 }
    },

    // VOLATILITY
    {
        id: 'volatility_breakout',
        name: 'Volatility Breakout',
        category: 'Volatility',
        description: 'Trades breakouts from volatility bands. Enters when price breaks out of Bollinger Bands.',
        complexity: 'Intermediate',
        time_horizon: 'Short-term',
        best_for: ["Volatile markets", "Breakout trading", "Crypto and commodities"],
        params: { period: 20, std_dev: 2.0 }
    },
    {
        id: 'volatility_targeting',
        name: 'Volatility Targeting',
        category: 'Volatility',
        description: 'Adjusts position size to maintain constant portfolio volatility. Scales down in high vol.',
        complexity: 'Advanced',
        time_horizon: 'All timeframes',
        best_for: ["Risk management", "Portfolio optimization", "Professional trading"],
        params: { target_vol: 0.15, lookback: 21 }
    },
    {
        id: 'dynamic_scaling',
        name: 'Dynamic Position Scaling',
        category: 'Volatility',
        description: 'Dynamically scales position sizes based on market conditions and volatility.',
        complexity: 'Advanced',
        time_horizon: 'All timeframes',
        best_for: ["Risk-adjusted returns", "Drawdown management", "Adaptive trading"],
        params: { base_size: 0.1, scaling_factor: 1.5 }
    },
    {
        id: 'variance_risk_premium',
        name: 'Variance Risk Premium',
        category: 'Volatility',
        description: 'Captures the premium between implied and realized volatility. Typically short volatility.',
        complexity: 'Advanced',
        time_horizon: 'Medium-term',
        best_for: ["Options trading", "Volatility arbitrage", "Institutional strategies"],
        params: { lookback: 30, threshold: 0.05 }
    },

    // STATISTICAL ARBITRAGE
    {
        id: 'sector_neutral',
        name: 'Sector Neutral Arbitrage',
        category: 'Statistical Arbitrage',
        description: 'Market-neutral strategy that is neutral within each sector. Exploits intra-sector relationships.',
        complexity: 'Advanced',
        time_horizon: 'Short to Medium-term',
        best_for: ["Hedge funds", "Market-neutral portfolios", "Statistical arbitrage"],
        params: { lookback: 60, rebalance_freq: 20 }
    },

    // MACHINE LEARNING
    {
        id: 'ml_random_forest',
        name: 'ML Random Forest',
        category: 'Machine Learning',
        description: 'Uses Random Forest classifier trained on technical indicators to predict market direction.',
        complexity: 'Advanced',
        time_horizon: 'Adaptable',
        best_for: ["Complex pattern recognition", "Multi-factor analysis"],
        params: { n_estimators: 100, max_depth: 10, test_size: 0.2 }
    },
    {
        id: 'ml_gradient_boosting',
        name: 'ML Gradient Boosting',
        category: 'Machine Learning',
        description: 'Uses Gradient Boosting classifier for sequential learning and improved predictions.',
        complexity: 'Advanced',
        time_horizon: 'Adaptable',
        best_for: ["Complex patterns", "Incremental learning"],
        params: { n_estimators: 100, learning_rate: 0.1, max_depth: 5 }
    },
    {
        id: 'ml_svm',
        name: 'ML SVM Classifier',
        category: 'Machine Learning',
        description: 'Uses Support Vector Machine (SVM) to classify market regimes and predict direction.',
        complexity: 'Advanced',
        time_horizon: 'Adaptable',
        best_for: ["Regime classification", "Non-linear boundaries"],
        params: { model_type: 'svm', test_size: 0.2 }
    },
    {
        id: 'ml_logistic',
        name: 'ML Logistic Regression',
        category: 'Machine Learning',
        description: 'Uses Logistic Regression for simple, interpretable market direction prediction.',
        complexity: 'Intermediate',
        time_horizon: 'Adaptable',
        best_for: ["Baseline models", "Interpretability"],
        params: { model_type: 'logistic_regression', test_size: 0.2 }
    },
    {
        id: 'ml_lstm',
        name: 'ML LSTM (Deep Learning)',
        category: 'Machine Learning',
        description: 'Uses Long Short-Term Memory (LSTM) neural network for time-series forecasting.',
        complexity: 'Expert',
        time_horizon: 'Short to Medium-term',
        best_for: ["Time-series forecasting", "Sequence patterns", "Complex temporal dependencies"],
        params: { lookback: 10, classes: 2, epochs: 20 }
    },

    // ADAPTIVE
    {
        id: 'adaptive_trend',
        name: 'Adaptive Trend Following',
        category: 'Adaptive Strategies',
        description: 'Dynamically adjusts trend-following parameters based on market conditions and volatility.',
        complexity: 'Advanced',
        time_horizon: 'Medium to Long-term',
        best_for: ["Changing market conditions", "Volatile markets", "Institutional trading"],
        params: { lookback_period: 50, volatility_threshold: 0.02 }
    },
    {
        id: 'kalman_filter',
        name: 'Kalman Filter Pairs Strategy',
        category: 'Adaptive Strategies',
        description: 'Statistical arbitrage using Kalman Filtering to dynamically estimate the hedge ratio.',
        complexity: 'Institutional',
        time_horizon: 'Intraday to Medium-term',
        best_for: ["Pairs Trading", "Statistical Arbitrage", "Mean Reversion"],
        params: { entry_z: 2.0, exit_z: 0.5, transitory_std: 0.01, observation_std: 0.1, decay_factor: 0.99, min_obs: 20 }
    },

    // OPTIONS
    {
        id: 'covered_call',
        name: 'Covered Call',
        category: 'Options Strategies',
        description: 'Hold stock and sell call options to generate income. Limited upside, downside protected by premium.',
        complexity: 'Intermediate',
        time_horizon: 'Short to Medium-term',
        best_for: ["Income generation", "Range-bound markets", "Conservative traders"],
        params: { strategy_type: 'covered_call', strike_pct: 0.05, dte: 30 }
    },
    {
        id: 'iron_condor',
        name: 'Iron Condor',
        category: 'Options Strategies',
        description: 'Market-neutral options strategy. Profits when underlying stays within a range.',
        complexity: 'Advanced',
        time_horizon: 'Short-term',
        best_for: ["Low volatility markets", "Income generation", "Range-bound stocks"],
        params: { strategy_type: 'iron_condor', wing_width: 0.05, dte: 30 }
    },
    {
        id: 'butterfly_spread',
        name: 'Butterfly Spread',
        category: 'Options Strategies',
        description: 'Limited risk strategy with concentrated profit zone. Profits when price stays near middle strike.',
        complexity: 'Advanced',
        time_horizon: 'Short-term',
        best_for: ["Neutral outlook", "Low volatility expected", "Precise targets"],
        params: { strategy_type: 'butterfly_spread', wing_width: 0.03, dte: 30 }
    },
    {
        id: 'straddle',
        name: 'Long Straddle',
        category: 'Options Strategies',
        description: 'Profits from large moves in either direction. Buy ATM call and put. Volatility play.',
        complexity: 'Intermediate',
        time_horizon: 'Short-term',
        best_for: ["Earnings events", "High expected volatility", "Direction unknown"],
        params: { strategy_type: 'straddle', dte: 30, iv_threshold: 0.30 }
    }
];