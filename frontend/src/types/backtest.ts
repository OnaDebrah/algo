
export interface BacktestConfig {
    symbols: string[];
    symbolInput: string;
    period: '1mo' | '1y' | '5y';
    interval: '1h' | '1d';
    strategyMode: 'same' | 'different';
    strategy: string;
    initialCapital: number;
    allocationMethod: 'equal' | 'custom' | 'risk-parity';
}

export interface MultiConfig {
    symbols: string[];
    symbolInput: string;
    period: string;
    interval: string;
    strategyMode: 'same' | 'different';
    strategy?: string;
    strategies: Record<string, string>;
    allocationMethod: string;
    allocations: Record<string, number>;
    initialCapital: number;
    maxPositionPct: number;
}

export interface Strategy {
    id: string;
    name: string;
    category: string;
    params: Record<string, string | number | boolean>;
    description: string;
    complexity: 'Beginner' | 'Intermediate' | 'Advanced' | 'Expert' | 'Institutional';
    time_horizon?: string;
    best_for?: string[];
    rating?: number;
    monthly_return?: number;
    drawdown?: number;
    sharpe_ratio?: number;
}

export interface Trade {
    id: number | string;
    date?: string;
    entry_time?: string;
    exit_time?: string;
    symbol: string;
    type?: string;
    side?: 'LONG' | 'SHORT';
    strategy?: string;
    quantity: number;
    price?: number;
    entry_price?: number;
    exit_price?: number;
    total?: number;
    pnl: number;
    status: 'open' | 'closed';
}

export interface BacktestResult {
    type: 'single' | 'multi';
    total_return: number;
    win_rate: number;
    sharpe_ratio: number;
    max_drawdown: number;
    total_trades: number;
    final_equity: number;
    equity_curve: Array<{
        timestamp: string;
        equity: number;
        num_positions?: number;
    }>;
    trades?: Trade[];
    avg_win?: number;
    avg_loss?: number;
    num_symbols?: number;
    avg_profit?: number;
    symbol_stats?: Record<string, {
        strategy: string;
        total_profit: number;
        num_trades: number;
        win_rate: number;
        avg_profit: number;
    }>;
}