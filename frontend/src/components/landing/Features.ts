import {BarChart3, Cpu, DollarSign, Sparkles, Target, Zap} from "lucide-react";

export const FEATURES = [
    {
        title: 'Backtesting Lab',
        desc: 'Single & multi-asset backtesting with walk-forward analysis, Monte Carlo simulation, and Bayesian optimization.',
        icon: BarChart3,
        gradient: 'from-blue-500 to-cyan-500',
    },
    {
        title: 'AI Strategy Builder',
        desc: 'Describe your strategy in plain English and let AI generate production-ready backtesting Python code.',
        icon: Sparkles,
        gradient: 'from-violet-500 to-purple-500',
    },
    {
        title: 'ML Strategy Studio',
        desc: 'Build and train ML models — Random Forest, Gradient Boosting, LSTM, and ensemble methods — with integrated feature engineering.',
        icon: Cpu,
        gradient: 'from-fuchsia-500 to-pink-500',
    },
    {
        title: 'Live Execution',
        desc: 'Deploy strategies to Alpaca and Interactive Brokers with real-time position tracking and automated risk controls.',
        icon: Zap,
        gradient: 'from-emerald-500 to-teal-500',
    },
    {
        title: 'Options Desk',
        desc: 'Options strategy builder with Greeks analytics, probability calculators, and volatility surface visualization.',
        icon: DollarSign,
        gradient: 'from-amber-500 to-orange-500',
    },
    {
        title: 'Portfolio Optimization',
        desc: 'Mean-variance, Black-Litterman, and risk parity optimization with efficient frontier visualization.',
        icon: Target,
        gradient: 'from-rose-500 to-red-500',
    },
];
