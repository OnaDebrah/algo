/* eslint-disable @typescript-eslint/no-explicit-any */

'use client'
import React, { useState } from 'react';
import {
    Activity,
    ArrowRight,
    BarChart3,
    BookOpen,
    BrainCircuit,
    ChevronDown,
    ChevronRight,
    Eye,
    Layers,
    LineChart,
    Play,
    Rocket,
    Search,
    Settings,
    Shield,
    Target,
    TrendingUp,
    Zap,
} from 'lucide-react';
import { useNavigationStore } from '@/store/useNavigationStore';

interface GuideSection {
    id: string;
    icon: any;
    title: string;
    description: string;
    steps: { title: string; detail: string }[];
    pageLink?: string;
    pageLinkLabel?: string;
}

const sections: GuideSection[] = [
    {
        id: 'quickstart',
        icon: Rocket,
        title: '1. Quick Start',
        description: 'Get up and running in 5 minutes.',
        steps: [
            { title: 'Create your account', detail: 'Sign up with email and password. All features are unlocked during the launch promo period.' },
            { title: 'Add stocks to your Watchlist', detail: 'Navigate to the Watchlist page and create a new list. Add ticker symbols (e.g., AAPL, TSLA, SPY) to track prices in real-time.' },
            { title: 'Run your first backtest', detail: 'Go to the Backtest Lab, pick a strategy (e.g., Moving Average Crossover), set a ticker and date range, then hit Run. Results appear in seconds.' },
            { title: 'Analyze with the AI Analyst', detail: 'Open the Deep Analyst, enter a ticker, and get a comprehensive report: technical signals, fundamentals, sentiment, and an AI-generated investment thesis.' },
        ],
        pageLink: 'watchlist',
        pageLinkLabel: 'Go to Watchlist',
    },
    {
        id: 'backtesting',
        icon: BarChart3,
        title: '2. Backtesting Strategies',
        description: 'Test trading strategies on historical data before risking real money.',
        steps: [
            { title: 'Choose a strategy', detail: 'Browse 100+ built-in strategies across categories: Trend Following, Mean Reversion, Momentum, Volatility, Options, ML-based, and more.' },
            { title: 'Configure parameters', detail: 'Each strategy has tunable parameters (e.g., lookback period, entry threshold). Adjust these or use the Bayesian Optimizer to find optimal values automatically.' },
            { title: 'Single vs Multi-Asset', detail: 'Single-asset mode tests one ticker. Multi-asset mode tests a portfolio with allocation weights across multiple symbols.' },
            { title: 'Read the results', detail: 'Key metrics: Sharpe Ratio (risk-adjusted return, >1.0 is good), Max Drawdown (worst peak-to-trough decline), Win Rate, and Total Return.' },
            { title: 'Optimize parameters', detail: 'Click the Optimizer button to run Bayesian optimization. It will try different parameter combinations and find the best performing set.' },
        ],
        pageLink: 'backtest',
        pageLinkLabel: 'Open Backtest Lab',
    },
    {
        id: 'analysis',
        icon: Search,
        title: '3. Market Analysis Tools',
        description: 'Deep research tools powered by AI and quantitative analysis.',
        steps: [
            { title: 'Deep Analyst', detail: 'AI-powered fundamental + technical + sentiment analysis with price targets, risk ratings, and an investment thesis. Works for any US-listed stock.' },
            { title: 'Options Desk', detail: 'Comprehensive options analytics: chain data, volatility surface, Greeks analysis, risk visualization, profit/loss forecasting, strategy builder, and ML pricing.' },
            { title: 'Sector Scanner', detail: 'Rank sectors by momentum, then drill into individual stocks within each sector. Get strategy recommendations tailored to each stock.' },
            { title: 'Market Regime Detector', detail: 'Uses Hidden Markov Models to classify the current market state (bull, bear, high-volatility, low-volatility) with transition probabilities.' },
            { title: 'Crash Prediction', detail: 'LSTM neural network that monitors market stress indicators and predicts crash probability. Includes hedge recommendations when risk is elevated.' },
        ],
        pageLink: 'analyst',
        pageLinkLabel: 'Open Deep Analyst',
    },
    {
        id: 'ml',
        icon: BrainCircuit,
        title: '4. ML Strategy Studio',
        description: 'Train machine learning models for trading signal prediction.',
        steps: [
            { title: 'Choose a model type', detail: 'Available models: Random Forest, Gradient Boosting, LSTM, Transformer, Reinforcement Learning (PPO/A2C). Each has different strengths.' },
            { title: 'Configure training', detail: 'Set the ticker, training period, test split, and hyperparameters. Enable feature engineering for auto-generated technical indicators.' },
            { title: 'Train and evaluate', detail: 'Training runs on the server. Results show accuracy, overfitting score, and feature importance charts. Models with >60% accuracy are worth deploying.' },
            { title: 'Deploy to backtests', detail: 'Trained models appear in the backtest strategy dropdown as ML Signal strategies. Test them with historical data before going live.' },
        ],
        pageLink: 'ml-studio',
        pageLinkLabel: 'Open ML Studio',
    },
    {
        id: 'live',
        icon: Play,
        title: '5. Paper & Live Trading',
        description: 'Deploy strategies to paper trading or real brokers.',
        steps: [
            { title: 'Paper Trading', detail: 'Start with the built-in paper trading simulator. It mimics real execution with simulated fills, slippage, and commissions. No broker needed.' },
            { title: 'Connect a broker', detail: 'Go to Settings > Broker Integration. Supported brokers: Alpaca (free paper + live), Interactive Brokers (professional). Enter API credentials.' },
            { title: 'Deploy a strategy', detail: 'After a successful backtest, click "Deploy to Live". Configure position sizing, stop losses, and risk limits. Start with paper mode first.' },
            { title: 'Monitor execution', detail: 'The Live Execution page shows real-time P&L, open positions, order history, and strategy performance. Set up alerts for important events.' },
        ],
        pageLink: 'paper-trading',
        pageLinkLabel: 'Open Paper Trading',
    },
    {
        id: 'marketplace',
        icon: Layers,
        title: '6. Strategy Marketplace',
        description: 'Browse, share, and discover community strategies.',
        steps: [
            { title: 'Browse strategies', detail: 'Filter by category, complexity, performance metrics, and price. Sort by Sharpe ratio, downloads, or rating.' },
            { title: 'Review performance', detail: 'Each strategy shows verified backtest results: equity curve, trade history, and risk metrics. Check the community reviews and discussion.' },
            { title: 'Fork a strategy', detail: 'Found a strategy you like? Fork it to create your own copy, then modify the parameters or logic to suit your style.' },
            { title: 'Publish your own', detail: 'Run a backtest with Sharpe >= 1.0 and Return >= 10%, then publish to the marketplace. Community strategies go through admin review.' },
        ],
        pageLink: 'marketplace',
        pageLinkLabel: 'Browse Marketplace',
    },
    {
        id: 'portfolio',
        icon: Target,
        title: '7. Portfolio Optimization',
        description: 'Construct optimal portfolios using modern portfolio theory.',
        steps: [
            { title: 'Add assets', detail: 'Enter ticker symbols for your portfolio. The optimizer fetches historical returns and computes the correlation matrix.' },
            { title: 'Choose optimization method', detail: 'Options: Maximum Sharpe (best risk-adjusted), Minimum Variance (lowest risk), Risk Parity (equal risk contribution), or custom target return.' },
            { title: 'Review allocation', detail: 'See the optimal weight for each asset, along with the efficient frontier chart showing the risk-return tradeoff.' },
            { title: 'Stress test', detail: 'Run stress tests against historical scenarios (2008 crash, COVID, dot-com) to see how your portfolio would have performed.' },
        ],
        pageLink: 'portfolio',
        pageLinkLabel: 'Open Portfolio Optimizer',
    },
    {
        id: 'glossary',
        icon: BookOpen,
        title: '8. Key Metrics Glossary',
        description: 'Understanding the numbers that matter.',
        steps: [
            { title: 'Sharpe Ratio', detail: 'Risk-adjusted return. Formula: (Return - Risk-Free Rate) / Volatility. Above 1.0 is good, above 2.0 is excellent. Below 0 means you lost money on a risk-adjusted basis.' },
            { title: 'Max Drawdown', detail: 'The largest peak-to-trough decline. A drawdown of -20% means at some point your portfolio fell 20% from its high. Lower is better.' },
            { title: 'Win Rate', detail: 'Percentage of trades that were profitable. A 60% win rate with good risk management is strong. Win rate alone doesn\'t determine profitability.' },
            { title: 'Sortino Ratio', detail: 'Like Sharpe but only penalizes downside volatility. Better for strategies with asymmetric returns. Above 1.5 is good.' },
            { title: 'Calmar Ratio', detail: 'Annualized return divided by max drawdown. Measures return per unit of drawdown risk. Above 1.0 is good.' },
            { title: 'Profit Factor', detail: 'Gross profit / gross loss. Above 1.5 is good. Below 1.0 means the strategy lost money overall.' },
            { title: 'VaR (Value at Risk)', detail: 'The maximum expected loss at a confidence level (typically 95%). A VaR of -2% means there\'s a 5% chance of losing more than 2% in a day.' },
        ],
    },
];

const GettingStarted = () => {
    const navigateTo = useNavigationStore(state => state.navigateTo);
    const [expandedSection, setExpandedSection] = useState<string>('quickstart');

    return (
        <div className="space-y-6 max-w-4xl mx-auto">
            {/* Header */}
            <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-emerald-500 to-teal-500 flex items-center justify-center shadow-lg shadow-emerald-500/20">
                    <BookOpen size={24} className="text-white" />
                </div>
                <div>
                    <h1 className="text-2xl font-bold text-slate-100">Getting Started</h1>
                    <p className="text-sm text-slate-500">Everything you need to start algorithmic trading with Oraculum</p>
                </div>
            </div>

            {/* Quick navigation */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {sections.slice(0, 8).map((s) => (
                    <button
                        key={s.id}
                        onClick={() => setExpandedSection(s.id)}
                        className={`p-3 rounded-xl border text-left transition-all ${
                            expandedSection === s.id
                                ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                                : 'bg-slate-800/30 border-slate-700/30 text-slate-400 hover:border-slate-600'
                        }`}
                    >
                        <s.icon size={18} className="mb-1" />
                        <p className="text-xs font-bold truncate">{s.title.replace(/^\d+\.\s*/, '')}</p>
                    </button>
                ))}
            </div>

            {/* Sections */}
            <div className="space-y-3">
                {sections.map((section) => {
                    const isOpen = expandedSection === section.id;
                    const Icon = section.icon;

                    return (
                        <div
                            key={section.id}
                            className={`border rounded-2xl overflow-hidden transition-all ${
                                isOpen
                                    ? 'bg-slate-800/40 border-slate-700/60'
                                    : 'bg-slate-800/20 border-slate-700/30 hover:border-slate-700/50'
                            }`}
                        >
                            <button
                                onClick={() => setExpandedSection(isOpen ? '' : section.id)}
                                className="w-full px-6 py-4 flex items-center justify-between"
                            >
                                <div className="flex items-center gap-3">
                                    <div className={`w-9 h-9 rounded-xl flex items-center justify-center ${
                                        isOpen ? 'bg-emerald-500/20 text-emerald-400' : 'bg-slate-700/40 text-slate-400'
                                    }`}>
                                        <Icon size={18} />
                                    </div>
                                    <div className="text-left">
                                        <h3 className="text-sm font-bold text-slate-200">{section.title}</h3>
                                        <p className="text-xs text-slate-500">{section.description}</p>
                                    </div>
                                </div>
                                {isOpen ? <ChevronDown size={18} className="text-slate-500" /> : <ChevronRight size={18} className="text-slate-500" />}
                            </button>

                            {isOpen && (
                                <div className="px-6 pb-5 space-y-3">
                                    {section.steps.map((step, idx) => (
                                        <div key={idx} className="flex gap-3">
                                            <div className="w-6 h-6 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center shrink-0 mt-0.5">
                                                <span className="text-xs font-bold">{idx + 1}</span>
                                            </div>
                                            <div>
                                                <p className="text-sm font-semibold text-slate-200">{step.title}</p>
                                                <p className="text-xs text-slate-400 leading-relaxed mt-0.5">{step.detail}</p>
                                            </div>
                                        </div>
                                    ))}
                                    {section.pageLink && (
                                        <button
                                            onClick={() => navigateTo(section.pageLink as any)}
                                            className="mt-2 flex items-center gap-2 px-4 py-2 bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 rounded-xl text-xs font-bold hover:bg-emerald-500/20 transition-all"
                                        >
                                            {section.pageLinkLabel}
                                            <ArrowRight size={14} />
                                        </button>
                                    )}
                                </div>
                            )}
                        </div>
                    );
                })}
            </div>

            {/* Tips */}
            <div className="bg-gradient-to-r from-violet-500/10 via-fuchsia-500/10 to-violet-500/10 border border-violet-500/30 rounded-2xl p-6">
                <div className="flex items-center gap-2 mb-3">
                    <Zap size={18} className="text-violet-400" />
                    <h3 className="text-sm font-bold text-violet-300">Pro Tips</h3>
                </div>
                <ul className="space-y-2 text-xs text-slate-400">
                    <li className="flex items-start gap-2">
                        <Shield size={12} className="text-violet-400 mt-0.5 shrink-0" />
                        <span>Always paper trade a strategy for at least 2 weeks before deploying with real money.</span>
                    </li>
                    <li className="flex items-start gap-2">
                        <TrendingUp size={12} className="text-violet-400 mt-0.5 shrink-0" />
                        <span>A Sharpe ratio above 1.0 with low max drawdown is more important than high total returns.</span>
                    </li>
                    <li className="flex items-start gap-2">
                        <Activity size={12} className="text-violet-400 mt-0.5 shrink-0" />
                        <span>Check the Market Regime Detector before deploying — trend-following strategies struggle in choppy markets.</span>
                    </li>
                    <li className="flex items-start gap-2">
                        <Settings size={12} className="text-violet-400 mt-0.5 shrink-0" />
                        <span>Configure risk limits in Settings before live trading: max position size, stop loss, and daily loss limit.</span>
                    </li>
                    <li className="flex items-start gap-2">
                        <LineChart size={12} className="text-violet-400 mt-0.5 shrink-0" />
                        <span>Use Walk-Forward Analysis to validate that your strategy isn't overfitted to historical data.</span>
                    </li>
                </ul>
            </div>
        </div>
    );
};

export default GettingStarted;
