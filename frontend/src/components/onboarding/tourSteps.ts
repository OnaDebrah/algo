export interface TourStep {
    target: string;        // CSS selector or data-tour attribute
    title: string;
    content: string;
    placement: 'top' | 'bottom' | 'left' | 'right';
    page?: string;         // navigate to this page first
}

const tourSteps: TourStep[] = [
    {
        target: '[data-tour="sidebar"]',
        title: 'Navigation',
        content: 'Use the sidebar to navigate between sections — monitoring, analysis, research, and strategy building tools are all organized here.',
        placement: 'right',
    },
    {
        target: '[data-tour="activity-hub"]',
        title: 'Activity Hub',
        content: 'Your central dashboard showing backtest history, live execution status, and recent platform activity at a glance.',
        placement: 'bottom',
        page: 'dashboard',
    },
    {
        target: '[data-tour="notifications"]',
        title: 'Notifications',
        content: 'Real-time alerts for price triggers, strategy updates, and marketplace activity appear here. You can also set custom price alerts.',
        placement: 'bottom',
    },
    {
        target: '[data-tour="strategy-builder"]',
        title: 'Strategy Builder',
        content: 'Build, configure, and backtest trading strategies. Choose from 20+ built-in strategies or create your own with AI assistance.',
        placement: 'right',
        page: 'strategy-builder',
    },
    {
        target: '[data-tour="backtest-lab"]',
        title: 'Backtesting Lab',
        content: 'Run single-asset, multi-asset, and options backtests with detailed performance metrics, equity curves, and trade analysis.',
        placement: 'right',
        page: 'backtest',
    },
    {
        target: '[data-tour="marketplace"]',
        title: 'Strategy Marketplace',
        content: 'Browse, rate, and download community strategies. Publish your own winning strategies to share with other traders.',
        placement: 'right',
        page: 'marketplace',
    },
    {
        target: '[data-tour="watchlist"]',
        title: 'Watchlist & Screener',
        content: 'Create custom watchlists to track symbols with live quotes, and use the screener to discover new opportunities.',
        placement: 'right',
        page: 'watchlist',
    },
    {
        target: '[data-tour="options-desk"]',
        title: 'Options Desk',
        content: 'Analyze options chains, calculate Greeks, run Monte Carlo simulations, and backtest complex options strategies.',
        placement: 'right',
        page: 'options',
    },
    {
        target: '[data-tour="live-execution"]',
        title: 'Live Execution',
        content: 'Connect your broker and execute strategies in real-time with position tracking, order management, and risk controls.',
        placement: 'right',
        page: 'live',
    },
    {
        target: '[data-tour="leaderboard"]',
        title: 'Strategy Leaderboard',
        content: 'See which strategies top the rankings by Sharpe ratio, returns, downloads, and ratings. Find inspiration and benchmark your work.',
        placement: 'right',
        page: 'leaderboard',
    },
];

export default tourSteps;
